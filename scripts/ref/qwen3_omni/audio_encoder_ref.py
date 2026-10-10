#!/usr/bin/env python3
"""M0 reference harness for Qwen3-Omni's AUDIO ENCODER (the port plan, M0 + M2 gate).

Instantiates ONLY `Qwen3OmniMoeAudioEncoder` from transformers 5.17 (`modeling_qwen3_omni_moe.py`,
"M:" below), loads the real `thinker.audio_tower.*` weights (fetch_audio_tower.py), runs it in
float32 on the CPU on log-mel features from the reference `WhisperFeatureExtractor`, and dumps the
features, the output and the per-layer hidden states. No full-model load: the encoder is 1.3 GB
of bf16 (2.6 GB as f32).

    VENV/bin/python audio_encoder_ref.py REF_DIR

REF_DIR must hold config.json + preprocessor_config.json (the model repo's, @26291f79),
audio_tower.safetensors, and speech_q.wav / speech_long.wav (16 kHz mono PCM16; made with macOS
`say`, sha256 recorded in the manifest). Output: REF_DIR/dump/<case>/... and
REF_DIR/dump/manifest.json.

The instrument checks (rule 8), each recorded in the manifest and asserted here:
- every forward runs TWICE and must be bit-identical;
- the token count equals the processor formula (13 per 100-frame chunk + the tail's convs,
  `M:152-159` / `P:108-116`) and the number of `<|audio_pad|>` rows the processor would emit;
- the mel of each generated M1 input equals the M1 fixture's stored frames bit for bit (the same
  numpy extractor path), so "the M1 fixtures' mel features" is literally what is encoded;
- the weights' state dict loads with strict=True (every parameter present, none left random).

Mel path: with torch installed the extractor would switch to its torch STFT path
(`feature_extraction_whisper.py:321-323`). M1 (and Arf) follow the numpy path, so this harness
forces it by patching `is_torch_available` in that module only. The encoder gate starts from
these features either way; the mel is the same array on both sides.

How the model calls the encoder (replicated, not imported): `get_audio_features` (M:1936-1943)
selects the valid frames of every clip with `feature_attention_mask`, concatenates them into one
`[128, sum(valid)]` and passes `feature_lens = mask.sum(-1)`. A batch is therefore NOT a batch of
independent encodes: every chunk of every clip is zero-padded to the longest chunk in the call
(`chunk_and_pad_features`, M:650-682) before the conv stem.
"""

import hashlib
import json
import math
import os
import sys
import time
import wave

import numpy as np
import torch
import transformers
from safetensors.torch import load_file
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as M
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig
from transformers.models.whisper import feature_extraction_whisper as W

REF = sys.argv[1]
OUT = os.path.join(REF, "dump")
M1_FIXTURES = os.path.join(os.path.dirname(__file__), "../../../crates/arf-core/testdata/qwen_audio")
PREFIX = "thinker.audio_tower."
# bf16 envelope (plan section 5) is run on these cases only: bf16 matmuls on this CPU are slow.
BF16_CASES = {"speech_q", "noise_3p7s"}

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)

# ---- inputs: the M1 generators (crates/arf-core/testdata/qwen_audio/gen_ref.py), verbatim ----
M64 = (1 << 64) - 1


def splitmix_i16(seed, n):
    s = seed & M64
    out = []
    for _ in range(n):
        s = (s + 0x9E3779B97F4A7C15) & M64
        z = s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        z = z ^ (z >> 31)
        v = z >> 48
        out.append(v - 65536 if v >= 32768 else v)
    return out


def tones_i16(n, sr, tones):
    out = []
    for i in range(n):
        v = 0.0
        for f, a in tones:
            v += a * math.sin(2.0 * math.pi * f * i / sr)
        out.append(max(-32768, min(32767, round(v * 32768.0))))
    return out


def fnv(i16s):
    h = 0xCBF29CE484222325
    for v in i16s:
        for byte in int(v).to_bytes(2, "little", signed=True):
            h = ((h ^ byte) * 0x100000001B3) & M64
    return f"{h:016x}"


def to_f32(i16s):
    return (np.asarray(i16s, dtype=np.int16).astype(np.float32) / np.float32(32768.0)).astype(np.float32)


def gen(meta):
    n = meta["n"]
    if meta["kind"] == "zeros":
        x = [0] * n
    elif meta["kind"] == "tones":
        x = tones_i16(n, meta["sr"], meta["tones"])
    elif meta["kind"] == "noise":
        x = [v >> meta.get("shift", 0) for v in splitmix_i16(meta["seed"], n)]
    elif meta["kind"] == "tones+noise":
        a = tones_i16(n, meta["sr"], meta["tones"])
        b = splitmix_i16(meta["seed"], n)
        x = [max(-32768, min(32767, p + (q >> meta["shift"]))) for p, q in zip(a, b)]
    else:
        raise ValueError(meta["kind"])
    assert fnv(x) == meta["fnv"], f"generator drifted from M1 for {meta}"
    return to_f32(x)


def read_wav(path):
    with wave.open(path, "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000), path
        raw = w.readframes(w.getnframes())
    i16 = np.frombuffer(raw, dtype="<i2")
    # soundfile / librosa int16 -> float32 is x / 32768 (M1 gen_ref.py docstring).
    return (i16.astype(np.float32) / np.float32(32768.0)).astype(np.float32)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for piece in iter(lambda: f.read(1 << 24), b""):
            h.update(piece)
    return h.hexdigest()


def md5(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


# ---- the reference feature extractor, numpy path (see docstring) ----
W.is_torch_available = lambda: False
fe = W.WhisperFeatureExtractor.from_pretrained(REF)
AUDIO_KWARGS = dict(sampling_rate=16000, padding=True, truncation=False, return_attention_mask=True)


def extract(clips):
    a = fe(clips, **AUDIO_KWARGS)
    b = fe(clips, **AUDIO_KWARGS)
    assert np.array_equal(a["input_features"], b["input_features"])
    return np.asarray(a["input_features"], dtype=np.float32), np.asarray(a["attention_mask"])


# ---- the encoder, alone ----
cfg_all = json.load(open(os.path.join(REF, "config.json")))
acfg = Qwen3OmniMoeAudioEncoderConfig(**cfg_all["thinker_config"]["audio_config"])
acfg._attn_implementation = "sdpa"  # the default a user gets; eager differs only in op order


def build(dtype):
    enc = M.Qwen3OmniMoeAudioEncoder(acfg)
    sd = load_file(os.path.join(REF, "audio_tower.safetensors"))
    sd = {k[len(PREFIX):]: v for k, v in sd.items()}
    missing, unexpected = enc.load_state_dict(sd, strict=True)
    assert not missing and not unexpected
    # The sinusoid table is a non-persistent buffer built in __init__ (M:94-112), in float32.
    return enc.to(dtype).eval()


def run(enc, feats, mask, dtype, taps=None):
    """M:1936-1943, then the tower. Returns the output as float32 numpy."""
    x = torch.from_numpy(feats).to(dtype)
    m = torch.from_numpy(mask).long()
    lens = m.sum(dim=1)
    x = x.permute(0, 2, 1)[m.bool()].permute(1, 0)
    hooks = []
    if taps is not None:
        hooks.append(enc.layers[0].register_forward_pre_hook(
            lambda mod, args: taps.append(args[0].detach().float().numpy().copy())))
        for layer in enc.layers:
            hooks.append(layer.register_forward_hook(
                lambda mod, args, out: taps.append(out[0].detach().float().numpy().copy())))
        hooks.append(enc.ln_post.register_forward_hook(
            lambda mod, args, out: taps.append(out.detach().float().numpy().copy())))
    try:
        with torch.no_grad():
            y = enc(x, feature_lens=lens).last_hidden_state
    finally:
        for h in hooks:
            h.remove()
    return y.float().numpy()


def token_formula(frames):
    chunk = 100
    leave = frames % chunk
    feat = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (frames // chunk) * 13


def windows(frames_list):
    """cu_seqlens as M:704-743 builds them, recomputed independently for the manifest."""
    chunks = []
    for f in frames_list:
        c = [100] * (f // 100) + ([f % 100] if f % 100 else [])
        chunks += c
    max_after = max(token_formula(c) for c in chunks)
    win = max_after * (800 // 100)
    out = []
    for f in frames_list:
        n = token_formula(f)
        out += [win] * (n // win) + ([n % win] if n % win else [])
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    m1 = json.load(open(os.path.join(M1_FIXTURES, "manifest.json")))
    m1_by_name = {c["name"]: c for c in m1["mel"]}
    cases = []
    for name in ["sine440_1s", "noise_3p7s", "silence_0p5s", "long_31s", "batch2", "short_170", "tones_24077"]:
        c = m1_by_name[name]
        cases.append((name, [gen(i) for i in c["inputs"]], {"m1": name, "inputs": c["inputs"]}))
    for wav in ["speech_q.wav", "speech_long.wav"]:
        p = os.path.join(REF, wav)
        cases.append((wav[:-4], [read_wav(p)], {"wav": wav, "sha256": sha256(p)}))

    t0 = time.time()
    enc32 = build(torch.float32)
    load_s = time.time() - t0
    enc16 = None
    man = {
        "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__,
        "threads": torch.get_num_threads(), "attn_implementation": acfg._attn_implementation,
        "weights_sha256": sha256(os.path.join(REF, "audio_tower.safetensors")),
        "audio_config": {k: getattr(acfg, k) for k in [
            "num_mel_bins", "encoder_layers", "encoder_attention_heads", "encoder_ffn_dim", "d_model",
            "activation_function", "max_source_positions", "n_window", "output_dim", "n_window_infer",
            "conv_chunksize", "downsample_hidden_size", "scale_embedding"]},
        "load_seconds_f32": round(load_s, 2),
        "cases": [],
    }
    for name, clips, meta in cases:
        feats, mask = extract(clips)
        valid = [int(v) for v in mask.sum(-1)]
        # M1 identity: the stored fixture frames must be these features, bit for bit.
        if "m1" in meta:
            fx = m1_by_name[meta["m1"]]
            assert fx["mask_sums"] == valid and fx["n_frames"] == feats.shape[2]
            for i, clip in enumerate(fx["clips"]):
                want = np.fromfile(os.path.join(M1_FIXTURES, clip["file"]), dtype="<f4").reshape(-1, 128)
                got = feats[i][:, clip["frames"]].T
                assert np.array_equal(got, want), f"{name}[{i}]: mel != M1 fixture"
        taps = []
        t1 = time.time()
        y = run(enc32, feats, mask, torch.float32, taps)
        dt = time.time() - t1
        y2 = run(enc32, feats, mask, torch.float32)
        assert np.array_equal(y, y2), f"{name}: two f32 runs differ"
        want_tokens = sum(token_formula(v) for v in valid)
        assert y.shape == (want_tokens, acfg.output_dim), (name, y.shape, want_tokens)
        assert len(taps) == 1 + acfg.encoder_layers + 1
        d = os.path.join(OUT, name)
        os.makedirs(d, exist_ok=True)
        feats.astype("<f4").tofile(os.path.join(d, "mel.f32"))
        y.astype("<f4").tofile(os.path.join(d, "out.f32"))
        np.save(os.path.join(d, "out.npy"), y)
        # hidden.f32: [34, N, 1280] = stem+pos (layer-0 input), after each of 32 layers, after ln_post.
        np.stack(taps).astype("<f4").tofile(os.path.join(d, "hidden.f32"))
        entry = {
            "name": name, **meta, "n_clips": len(clips), "samples": [len(c) for c in clips],
            "mel_shape": list(feats.shape), "valid_frames": valid,
            "tokens_per_clip": [token_formula(v) for v in valid], "tokens": int(y.shape[0]),
            "windows": windows(valid), "out_shape": list(y.shape), "out_md5": md5(y),
            "mel_md5": md5(feats), "hidden_taps": len(taps), "twice_identical": True,
            "cpu_seconds_f32": round(dt, 3),
            "out_rms": float(np.sqrt((y.astype(np.float64) ** 2).mean())),
        }
        if name in BF16_CASES:
            if enc16 is None:
                enc16 = build(torch.bfloat16)
            yb = run(enc16, feats, mask, torch.bfloat16)
            yb2 = run(enc16, feats, mask, torch.bfloat16)
            assert np.array_equal(yb, yb2)
            a, b = yb.astype(np.float64), y.astype(np.float64)
            cos = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
            entry["bf16_vs_f32"] = {"rel": float(np.linalg.norm(a - b) / np.linalg.norm(b)),
                                    "max_abs": float(np.abs(a - b).max()), "min_cos": float(cos.min())}
        man["cases"].append(entry)
        print(f"{name:14s} mel={feats.shape} valid={valid} tokens={y.shape[0]} windows={entry['windows']} "
              f"f32 {dt:.2f}s " + (f"bf16 rel={entry['bf16_vs_f32']['rel']:.3e}" if "bf16_vs_f32" in entry else ""))
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    print("ok", os.path.join(OUT, "manifest.json"))


if __name__ == "__main__":
    main()
