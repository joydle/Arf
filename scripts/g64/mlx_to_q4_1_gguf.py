#!/usr/bin/env python3
"""MLX affine 4-bit (group 64, bf16 scale + bias) projections -> a HYBRID GGUF (2026-09-25).

Takes our GGUF (unsloth Qwen3.8-27B Q4_K_M) as the template and replaces the target's big
projections with the MLX checkpoint's weights (mlx-community/Qwen3.8-27B-4bit — the weights another engine
1.0.2 ships), written as standard GGUF Q4_1. The conversion is EXACT: one MLX group of 64 with
scale s and bias b is two Q4_1 blocks of 32 with d = s, m = b. bf16 -> fp16 is exact except in
fp16's subnormal range: MEASURED over all 380,108,800 groups of the 27B, 522 scales and 2 biases
round, the largest absolute change 2.98e-8 (groups whose weights are ~1e-6). Allowed up to 1e-7. Every other tensor (norms, GDN conv/decay/dt, beta/alpha,
embeddings, lm_head, the MTP head blk.64.*) is copied verbatim from the template.

Why Q4_1: llama.cpp reads it (a same-engine quality check against the template), and Arf's loader
recombines each pair of Q4_1 blocks into one group-64 block at load (the fast format).

GDN head order: our value head h is the MLX/HF head 3*(h%16)+h//16 (found empirically for
another engine's repack 2026-09-22; another engine = MLX + joins). Applied to qkv's v rows, z rows and ssm_out
columns, in 128-wide head blocks. Attention and FFN projections map 1:1. Checked by correlation
against the template's dequantized weights before anything is written.

usage: mlx_to_q4_1_gguf.py <mlx snapshot dir> <template.gguf> <out.gguf>
       G64_LM_HEAD=1 also replaces output.weight (the lm_head) with MLX's group-64 one
"""
import json, os, sys
import numpy as np, gguf
from safetensors.numpy import load_file  # noqa: F401 (bf16 needs the torch-free path below)
from gguf import GGMLQuantizationType as Q

MLX, TEMPLATE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
H, INTER = 5120, 17408
VPERM = np.array([3 * (h % 16) + h // 16 for h in range(48)])

# ---------------- MLX safetensors reader (bf16-safe, memory-mapped) ----------------
_index = json.load(open(os.path.join(MLX, "model.safetensors.index.json")))["weight_map"]
_headers = {}
def _hdr(fn):
    if fn not in _headers:
        p = os.path.join(MLX, fn)
        with open(p, "rb") as f:
            n = int.from_bytes(f.read(8), "little"); h = json.loads(f.read(n))
        _headers[fn] = (p, 8 + n, h, np.memmap(p, dtype=np.uint8, mode="r"))
    return _headers[fn]
def mlx(name):
    fn = _index[name]; p, base, h, mm = _hdr(fn); e = h[name]
    a, b = e["data_offsets"]; raw = mm[base + a: base + b]
    dt = e["dtype"]; shape = e["shape"]
    if dt == "U32": return np.frombuffer(raw, dtype=np.uint32).reshape(shape)
    if dt == "BF16": return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
    if dt == "F32": return np.frombuffer(raw, dtype=np.float32).reshape(shape)
    raise ValueError(dt)
def mlx_q(prefix):
    """(q [N, K] uint8 0..15, scales [N, K/64] f32, biases [N, K/64] f32) — MLX packs 8 nibbles
    per uint32, element j at bits 4j..4j+3 (low first)."""
    w = mlx(prefix + ".weight"); s = mlx(prefix + ".scales"); b = mlx(prefix + ".biases")
    N, K8 = w.shape
    q = np.empty((N, K8 * 8), dtype=np.uint8)
    for j in range(8): q[:, j::8] = ((w >> (4 * j)) & 0xF).astype(np.uint8)
    return q, s, b

# ---------------- mapping: our GGUF projection -> (mlx prefix, row blocks, col blocks) ----------------
def rows_qkv(n):   # [q 2048 | k 2048] identity, v 48 heads x 128 permuted
    return np.concatenate([np.arange(4096), (4096 + (VPERM[:, None] * 128 + np.arange(128))).ravel()])
def rows_heads(n): return (VPERM[:, None] * 128 + np.arange(128)).ravel()
def plan_for(name):
    # the lm_head too (2026-09-25, G64_LM_HEAD=1): MLX quantizes it group-64 like the projections,
    # vocab rows in order. Without the flag the template's (Q6_K) lm_head is kept, as before.
    if name == "output.weight":
        return ("language_model.lm_head", None, None) if os.environ.get("G64_LM_HEAD") else None
    parts = name.split(".")
    if parts[0] != "blk" or int(parts[1]) >= 64: return None
    i = int(parts[1]); base = ".".join(parts[2:])
    L = f"language_model.model.layers.{i}"
    m = {
        "attn_q.weight": (f"{L}.self_attn.q_proj", None, None),
        "attn_k.weight": (f"{L}.self_attn.k_proj", None, None),
        "attn_v.weight": (f"{L}.self_attn.v_proj", None, None),
        "attn_output.weight": (f"{L}.self_attn.o_proj", None, None),
        "attn_qkv.weight": (f"{L}.linear_attn.in_proj_qkv", rows_qkv, None),
        "attn_gate.weight": (f"{L}.linear_attn.in_proj_z", rows_heads, None),
        "ssm_out.weight": (f"{L}.linear_attn.out_proj", None, rows_heads),
        "ffn_gate.weight": (f"{L}.mlp.gate_proj", None, None),
        "ffn_up.weight": (f"{L}.mlp.up_proj", None, None),
        "ffn_down.weight": (f"{L}.mlp.down_proj", None, None),
    }
    return m.get(base)

def mapped(name):
    pre, rows, cols = plan_for(name)
    q, s, b = mlx_q(pre)
    if rows is not None:
        r = rows(q.shape[0]); q, s, b = q[r], s[r], b[r]
    if cols is not None:
        c = cols(q.shape[1])                       # permutes 128-wide blocks = whole 64-groups
        g = c.reshape(-1, 64)[:, 0] // 64
        q, s, b = q[:, c], s[:, g], b[:, g]
    return q, s, b

def dequant_g64(q, s, b):
    N, K = q.shape
    return (q.reshape(N, K // 64, 64).astype(np.float32) * s[:, :, None] + b[:, :, None]).reshape(N, K)

def to_q4_1(q, s, b):
    """Two Q4_1 blocks per group of 64: {f16 d, f16 m, 16 bytes: x[j] | x[j+16] << 4}."""
    N, K = q.shape; nb = K // 32
    d = np.repeat(s, 2, axis=1).astype(np.float16); m = np.repeat(b, 2, axis=1).astype(np.float16)
    assert np.abs(d.astype(np.float32) - np.repeat(s, 2, axis=1)).max() <= 1e-7, "scale not ~exact in f16"
    assert np.abs(m.astype(np.float32) - np.repeat(b, 2, axis=1)).max() <= 1e-7, "bias not ~exact in f16"
    qb = q.reshape(N, nb, 32)
    qs = (qb[:, :, :16] | (qb[:, :, 16:] << 4)).astype(np.uint8)
    out = np.empty((N, nb, 20), dtype=np.uint8)
    out[:, :, 0:2] = d.view(np.uint8).reshape(N, nb, 2)
    out[:, :, 2:4] = m.view(np.uint8).reshape(N, nb, 2)
    out[:, :, 4:20] = qs
    return out.reshape(N, nb * 20)

rd = gguf.GGUFReader(TEMPLATE)
tensors = {t.name: t for t in rd.tensors}
# ---------------- self-check: mapping against the template ----------------
def corr(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))
for name in ["blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_out.weight", "blk.0.ffn_gate.weight",
             "blk.0.ffn_down.weight", "blk.3.attn_q.weight", "blk.3.attn_k.weight", "blk.3.attn_v.weight",
             "blk.3.attn_output.weight", "blk.62.ffn_up.weight", "blk.61.attn_qkv.weight"] + \
            (["output.weight"] if os.environ.get("G64_LM_HEAD") else []):
    t = tensors[name]
    ref = gguf.quants.dequantize(t.data, t.tensor_type).reshape(-1, t.shape[0]).astype(np.float32)
    q, s, b = mapped(name); val = dequant_g64(q, s, b)
    assert val.shape == ref.shape, (name, val.shape, ref.shape)
    c = corr(val, ref); print(f"check {name:<26} {val.shape} corr vs template {c:.4f}", flush=True)
    assert c > 0.98, f"{name}: mapping wrong (corr {c:.4f}) — aborting before writing"
    # the Q4_1 encoding round-trips exactly
    enc = to_q4_1(q, s, b)
    back = gguf.quants.dequantize(enc.reshape(-1), Q.Q4_1).reshape(val.shape)
    assert np.array_equal(back, val), f"{name}: Q4_1 encoding is not exact"
print("mapping + exact Q4_1 encoding verified", flush=True)

# ---------------- write ----------------
w = gguf.GGUFWriter(OUT, rd.fields["general.architecture"].contents())
for f in rd.fields.values():
    if f.name in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"): continue
    vt = f.types[0]; sub = f.types[1] if len(f.types) > 1 else None
    val = f.contents()
    if f.name == "general.name": val = str(val) + " + mlx-community 4-bit g64 projections (Q4_1, exact)" + (" + lm_head" if os.environ.get("G64_LM_HEAD") else "")
    w.add_key_value(f.name, val, vt, sub)
order = [t.name for t in rd.tensors]
for name in order:
    t = tensors[name]
    if plan_for(name) is None:
        w.add_tensor_info(name, list(t.data.shape), t.data.dtype, t.n_bytes, raw_dtype=t.tensor_type)
    else:
        N, K = int(t.shape[1]), int(t.shape[0])
        w.add_tensor_info(name, [N, K // 32 * 20], np.dtype(np.uint8), N * (K // 32) * 20, raw_dtype=Q.Q4_1)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for k, name in enumerate(order):
    t = tensors[name]
    if plan_for(name) is None:
        w.write_tensor_data(np.asarray(t.data))
    else:
        q, s, b = mapped(name); w.write_tensor_data(to_q4_1(q, s, b))
    if k % 50 == 0: print(f"[{k}/{len(order)}] {name}", flush=True)
w.close()
print("DONE", OUT, round(os.path.getsize(OUT) / 1e9, 2), "GB")
