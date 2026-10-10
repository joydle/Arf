"""Reference fixtures for `arf_core::model::qwen_audio` (Qwen3-Omni audio front end, M1).

Run from this directory (numpy + transformers 5.17, NO torch; soxr only for the resample cases):

    python3 -m venv --system-site-packages /tmp/v && /tmp/v/bin/pip install soxr==1.1.0
    /tmp/v/bin/python gen_ref.py

What it runs, and why it is the reference:
- `WhisperFeatureExtractor.from_pretrained(".")` on `preprocessor_config.json`, byte-identical to
  Qwen/Qwen3-Omni-30B-A3B-Instruct @ 26291f793822fb6be9555850f06dfe95f2d7e695 (md5
  c9b23714cb566b484133ca01739c3907), called with the processor's audio kwargs
  (transformers 5.17 `processing_qwen3_omni_moe.py:98-104`: sampling_rate 16000, padding True,
  truncation False, return_attention_mask True). Without torch installed the extractor takes its
  numpy path (`feature_extraction_whisper.py:321-324` -> `:105-133`).
- the input side is qwen-omni-utils 0.0.9 `audio_process.py:84-91`: `librosa.load(sr=16000)`,
  which in librosa 1.0.0 is soundfile float32 -> `to_mono` (mean over channels) -> `resample`
  with `res_type="soxr_hq"` -> `fix_length(ceil(n * ratio))` (`librosa/core/audio.py:159-163,
  705-740, 1121-1173`). librosa itself is not installed here; the same three steps are done
  below with numpy + soxr directly (soundfile's int16 -> float is x / 32768).

Inputs are generated from integer-exact formulas that `qwen_audio.rs`'s tests reproduce (and
check against the `input_fnv` checksums below), so only the outputs are stored. Every run of the
extractor is done twice and must be bit-identical (the instrument check).
"""

import json
import math
import wave

import numpy as np
import transformers
from transformers import WhisperFeatureExtractor

M64 = (1 << 64) - 1


def splitmix_i16(seed, n):
    """splitmix64; each output's top 16 bits as a signed i16."""
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
    """sum of a*sin(2*pi*f*i/sr), times 32768, round-half-even, clamped to i16."""
    out = []
    for i in range(n):
        v = 0.0
        for f, a in tones:
            v += a * math.sin(2.0 * math.pi * f * i / sr)
        out.append(max(-32768, min(32767, round(v * 32768.0))))
    return out


def mix(a, b, shift):
    """clamp(a + (b >> shift)) -- arithmetic shift."""
    return [max(-32768, min(32767, x + (y >> shift))) for x, y in zip(a, b)]


def fnv(i16s):
    h = 0xCBF29CE484222325
    for v in i16s:
        for byte in int(v).to_bytes(2, "little", signed=True):
            h = ((h ^ byte) * 0x100000001B3) & M64
    return f"{h:016x}"


def to_f32(i16s):
    return (np.asarray(i16s, dtype=np.int16).astype(np.float32) / np.float32(32768.0)).astype(np.float32)


fe = WhisperFeatureExtractor.from_pretrained(".")
AUDIO_KWARGS = dict(sampling_rate=16000, padding=True, truncation=False, return_attention_mask=True)


def extract(clips):
    a = fe(clips, **AUDIO_KWARGS)
    b = fe(clips, **AUDIO_KWARGS)
    assert np.array_equal(a["input_features"], b["input_features"]), "reference not deterministic"
    feats = np.asarray(a["input_features"], dtype=np.float32)  # [B, 128, T]
    mask = np.asarray(a["attention_mask"])  # [B, T]
    return feats, mask


manifest = {
    "transformers": transformers.__version__,
    "numpy": np.__version__,
    "extractor": {k: getattr(fe, k) for k in ["feature_size", "sampling_rate", "hop_length", "n_fft", "chunk_length", "n_samples", "dither", "padding_value"]},
    "mel": [],
    "resample": [],
}


def pick_frames(t):
    """Frames stored verbatim (all of them if t <= 32): the first 3, ~24 evenly spaced, the last 3,
    and the frames around 3000 (the 30 s chunk_length) when the clip crosses it. Every other frame
    is covered only by the whole-array sum / sum of squares / min / max."""
    if t <= 32:
        return list(range(t))
    s = set(range(3)) | set(range(0, t, max(1, t // 24))) | set(range(t - 3, t))
    if t > 3003:
        s |= set(range(2997, 3004))
    return sorted(s)


def store_mel(name, clips, inputs_meta, wav=None):
    feats, mask = extract(clips)
    b, nm, t = feats.shape
    entry = {"name": name, "inputs": inputs_meta, "n_mels": nm, "n_frames": t,
             "mask_sums": [int(x) for x in mask.sum(-1)], "clips": []}
    if wav:
        entry["wav"] = wav
    for i in range(b):
        f = feats[i]
        frames = pick_frames(t)
        fname = f"{name}_{i}.f32"
        np.ascontiguousarray(f[:, frames].T).astype("<f4").tofile(fname)  # [k][128]
        f64 = f.astype(np.float64)
        entry["clips"].append({
            "file": fname, "frames": frames,
            "sum": float(f64.sum()), "sumsq": float((f64 * f64).sum()),
            "min": float(f.min()), "max": float(f.max()),
            "mask": [int(x) for x in mask[i]] if t <= 64 else None,
        })
    manifest["mel"].append(entry)
    print(f"mel {name}: feats {feats.shape} mask_sums {entry['mask_sums']}")


def meta(kind, n, **kw):
    return dict(kind=kind, n=n, **kw)


# (a) 440 Hz sine, 1 s
x = tones_i16(16000, 16000, [(440, 0.5)])
store_mel("sine440_1s", to_f32(x), [meta("tones", 16000, sr=16000, tones=[[440, 0.5]], fnv=fnv(x))])

# (b) white noise, 3.7 s, splitmix seed 1, full scale
x = splitmix_i16(1, 59200)
store_mel("noise_3p7s", to_f32(x), [meta("noise", 59200, seed=1, fnv=fnv(x))])

# (c) silence, 0.5 s
x = [0] * 8000
store_mel("silence_0p5s", to_f32(x), [meta("zeros", 8000, fnv=fnv(x))])

# (d) 31 s: longer than the extractor's 30 s chunk_length (truncation is False, so no cut)
t1 = tones_i16(496000, 16000, [(1000, 0.3), (3000, 0.1)])
nz = splitmix_i16(7, 496000)
x = mix(t1, nz, 3)
store_mel("long_31s", to_f32(x), [meta("tones+noise", 496000, sr=16000, tones=[[1000, 0.3], [3000, 0.1]], seed=7, shift=3, fnv=fnv(x))])

# (e) stereo 16 kHz WAV (written by the stdlib `wave` module), mixed down as librosa does
left = tones_i16(4000, 16000, [(440, 0.4)])
right = [v >> 1 for v in splitmix_i16(2, 4000)]
inter = np.empty(8000, dtype="<i2")
inter[0::2] = left
inter[1::2] = right
with wave.open("stereo_16k.wav", "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(16000)
    w.writeframes(inter.tobytes())
st = inter.reshape(-1, 2).T.astype(np.float32) / np.float32(32768.0)  # soundfile float32, (ch, n)
mono = np.mean(st, axis=0)  # librosa to_mono, norm=True
assert mono.dtype == np.float32
store_mel("stereo_16k", mono, [meta("wav", 4000, left_fnv=fnv(left), right_fnv=fnv(right))], wav="stereo_16k.wav")

# (f) a batch of two clips of different lengths, padded to the longest by the processor
xa = splitmix_i16(3, 16500)
xb = tones_i16(9770, 16000, [(440, 0.5), (2500, 0.2)])
store_mel("batch2", [to_f32(xa), to_f32(xb)],
          [meta("noise", 16500, seed=3, fnv=fnv(xa)), meta("tones", 9770, sr=16000, tones=[[440, 0.5], [2500, 0.2]], fnv=fnv(xb))])

# (g) 170 samples: shorter than the 200-sample reflect pad (numpy reflects repeatedly)
x = splitmix_i16(4, 170)
store_mel("short_170", to_f32(x), [meta("noise", 170, seed=4, fnv=fnv(x))])

# (h) a length that is not a multiple of the hop (1.5 s + 77 samples)
x = tones_i16(24077, 16000, [(300, 0.3), (1200, 0.2), (7000, 0.05)])
store_mel("tones_24077", to_f32(x), [meta("tones", 24077, sr=16000, tones=[[300, 0.3], [1200, 0.2], [7000, 0.05]], fnv=fnv(x))])

# ---- resample: the librosa.load path (to_mono, then soxr HQ, then fix_length) ----
try:
    import soxr

    manifest["soxr"] = soxr.__version__

    def store_rs(name, sr, chans, chan_meta):
        arr = np.stack([to_f32(c) for c in chans])  # (ch, n) float32
        mono = np.mean(arr, axis=0) if len(chans) > 1 else arr[0]
        y = soxr.resample(mono, sr, 16000, quality="soxr_hq")
        n_out = int(np.ceil(mono.shape[-1] * (16000.0 / sr)))
        y = np.asarray(y, dtype=np.float32)
        y = y[:n_out] if y.shape[0] >= n_out else np.pad(y, (0, n_out - y.shape[0]))
        fname = f"rs_{name}.f32"
        y.astype("<f4").tofile(fname)
        manifest["resample"].append({"name": name, "sr": sr, "channels": chan_meta, "n_in": len(chans[0]),
                                     "n_out": n_out, "file": fname})
        print(f"resample {name}: {sr} Hz x{len(chans)} {len(chans[0])} -> {n_out}")

    l = tones_i16(13230, 44100, [(300, 0.2), (1200, 0.15), (3400, 0.1), (7000, 0.1), (10000, 0.1)])
    r = [v >> 2 for v in splitmix_i16(5, 13230)]
    store_rs("44k1_stereo", 44100, [l, r], [
        meta("tones", 13230, sr=44100, tones=[[300, 0.2], [1200, 0.15], [3400, 0.1], [7000, 0.1], [10000, 0.1]], fnv=fnv(l)),
        meta("noise", 13230, seed=5, shift=2, fnv=fnv(r))])
    x = tones_i16(14417, 48000, [(200, 0.3), (5000, 0.2), (7700, 0.1), (12000, 0.1)])
    store_rs("48k_mono", 48000, [x], [meta("tones", 14417, sr=48000, tones=[[200, 0.3], [5000, 0.2], [7700, 0.1], [12000, 0.1]], fnv=fnv(x))])
    x = tones_i16(2403, 8000, [(500, 0.4), (3000, 0.3)])
    store_rs("8k_mono", 8000, [x], [meta("tones", 2403, sr=8000, tones=[[500, 0.4], [3000, 0.3]], fnv=fnv(x))])
    x = [v >> 1 for v in splitmix_i16(6, 6615)]
    store_rs("22k05_noise", 22050, [x], [meta("noise", 6615, seed=6, shift=1, fnv=fnv(x))])
except ImportError:
    print("soxr not installed: resample fixtures NOT regenerated")

with open("manifest.json", "w") as f:
    json.dump(manifest, f, indent=1)
print("wrote manifest.json")
