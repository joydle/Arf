#!/usr/bin/env python3
"""OUR group-64 quantization (2026-09-25): an 8-bit GGUF -> group-64 affine 4-bit projections,
imatrix-weighted, written as exact Q4_1 pairs into a copy of our Q4_K_M GGUF.

Why: another engine's weights (mlx-community 4-bit, plain round-to-nearest from min/max) cost +2.1%
perplexity against our Q4_K_M (measured 2026-09-25), and our free-form draft
acceptance on them fell. The fast FORMAT is the win; the QUANTIZATION inside it is ours to choose.

Method — llama.cpp's Q4_1-with-imatrix recipe (`quantize_row_q4_1_impl` + `make_qkx3_quants`),
at group 64 instead of 32:
  importance  w_i = imatrix[col_i] * sqrt(sigma2_row + x_i^2),  sigma2_row = mean(x_row^2)
  candidates  iscale = (15 + rmin + rdelta*j) / (max - min), j = 0..nstep  (rmin -0.9, rdelta 0.05, 36)
              plus plain min/max; for each: q = clip(round(iscale*(x - min)), 0, 15), then the
              weighted least-squares (scale, bias) for those q; keep the lowest weighted error
  finally     scale and bias rounded to f16 (what the carrier stores), q re-rounded against them
Every step is elementwise over groups, run on the GPU through MLX.

Source: unsloth Qwen3.8-27B-Q8_0.gguf (same tensor names and layout as our Q4_K_M, so no head
permutations), imatrix: unsloth's `imatrix_unsloth.gguf` (the calibration behind our Q4_K_M).
Everything that is not one of the 400 trunk projections is copied from the TEMPLATE, so against the
MLX arm (scripts/g64/mlx_to_q4_1_gguf.py) the only variable is how the projections were quantized.

usage: quantize_g64.py <source Q8_0.gguf> <template.gguf> <imatrix.gguf> <out.gguf> [--check-only]
       G64_LM_HEAD=1 also quantizes output.weight (the lm_head) the same way
       G64_3BIT=ffn_gate,ffn_up[,ffn_down] quantizes those at 3 bits (codes 0..7, same carrier)
"""
import os, sys, time
import numpy as np, gguf
import mlx.core as mx
from gguf import GGMLQuantizationType as Q

SRC, TEMPLATE, IMAT, OUT = sys.argv[1:5]
CHECK_ONLY = "--check-only" in sys.argv
PROJ = {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_qkv.weight",
        "attn_gate.weight", "ssm_out.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}

LM_HEAD = bool(os.environ.get("G64_LM_HEAD"))  # also quantize output.weight (no imatrix entry:
                                               # llama.cpp's sqrt(sigma2 + x^2) weighting alone)
# G64_3BIT="ffn_gate,ffn_up[,ffn_down]" (2026-09-26): these projections at 3 BITS (codes 0..7), same
# fit with nmax 7. Written in the same Q4_1 carrier — a 3-bit group is exactly a 4-bit group whose codes
# stay below 8 — so the engine measures its quality unchanged and packs it to 3 bits at load.
BITS3 = {f"{t.strip()}.weight" for t in os.environ.get("G64_3BIT", "").split(",") if t.strip()}

def nmax_for(name):
    p = name.split(".")
    return 7 if len(p) > 2 and ".".join(p[2:]) in BITS3 else 15

def is_proj(name):
    if name == "output.weight":
        return LM_HEAD
    p = name.split(".")
    return p[0] == "blk" and int(p[1]) < 64 and ".".join(p[2:]) in PROJ

src = gguf.GGUFReader(SRC); S = {t.name: t for t in src.tensors}
rd = gguf.GGUFReader(TEMPLATE); T = {t.name: t for t in rd.tensors}
im = gguf.GGUFReader(IMAT); I = {t.name: np.asarray(t.data, dtype=np.float32) for t in im.tensors}

def dense(t):
    """[N, K] f32 from a GGUF tensor (ggml dims are [K, N])."""
    return gguf.quants.dequantize(t.data, t.tensor_type).reshape(-1, int(t.shape[0])).astype(np.float32)

def imatrix(name, K):
    s2, c = I.get(name + ".in_sum2"), I.get(name + ".counts")
    if s2 is None:
        return None
    assert s2.shape[0] == K, (name, s2.shape, K)
    return s2 / max(float(c[0]), 1.0)

def fit(x, qw, nmax=15):
    """x [G, 64] f32 (mx), qw [G, 64] importance -> (q uint8 [G, 64], d f16 [G], m f16 [G]); codes 0..nmax."""
    mn = mx.min(x, axis=1, keepdims=True); mxv = mx.max(x, axis=1, keepdims=True)
    rng = mx.maximum(mxv - mn, 1e-30)
    sw = mx.sum(qw, axis=1, keepdims=True); sx = mx.sum(qw * x, axis=1, keepdims=True)
    best_err = mx.full(mn.shape, float("inf")); best_d = rng / float(nmax); best_m = mn
    for j in [None] + list(range(37)):
        isc = float(nmax) / rng if j is None else (float(nmax) - 0.9 + 0.05 * j) / rng
        L = mx.clip(mx.round(isc * (x - mn)), 0, nmax)
        sl = mx.sum(qw * L, axis=1, keepdims=True); sl2 = mx.sum(qw * L * L, axis=1, keepdims=True)
        sxl = mx.sum(qw * x * L, axis=1, keepdims=True)
        D = sw * sl2 - sl * sl
        ok = D > 0
        Dsafe = mx.where(ok, D, 1.0)
        d = mx.where(ok, (sw * sxl - sx * sl) / Dsafe, rng / float(nmax))
        m = mx.where(ok, (sl2 * sx - sl * sxl) / Dsafe, mn)
        e = mx.sum(qw * mx.square(d * L + m - x), axis=1, keepdims=True)
        better = e < best_err
        best_err = mx.where(better, e, best_err); best_d = mx.where(better, d, best_d); best_m = mx.where(better, m, best_m)
        mx.eval(best_err, best_d, best_m)
    d16 = best_d.astype(mx.float16); m16 = best_m.astype(mx.float16)
    df = d16.astype(mx.float32); mf = m16.astype(mx.float32)
    q = mx.clip(mx.round((x - mf) / mx.where(df == 0, 1.0, df)), 0, nmax).astype(mx.uint8)
    q = mx.where(df == 0, mx.zeros_like(q), q)
    mx.eval(q, d16, m16)
    return np.array(q), np.array(d16).reshape(-1), np.array(m16).reshape(-1)

def rtn(x, nmax=15):
    """plain round-to-nearest from min/max (MLX's method), for the self-check only."""
    mn = mx.min(x, axis=1, keepdims=True); mxv = mx.max(x, axis=1, keepdims=True)
    d = (mxv - mn) / float(nmax); d = mx.where(d == 0, 1.0, d)
    q = mx.clip(mx.round((x - mn) / d), 0, nmax)
    return q * d + mn

CHUNK = 8192  # rows per pass: bounds memory (the lm_head is 248,320 x 5,120)

def quantize(name, xn):
    N, K = xn.shape
    imat = imatrix(name, K)
    q, d, m = [], [], []
    for r0 in range(0, N, CHUNK):
        xc = xn[r0:r0 + CHUNK]; n = xc.shape[0]
        x = mx.array(xc.reshape(-1, 64))
        sig2 = mx.array(np.repeat(np.mean(xc * xc, axis=1), K // 64)[:, None])
        base = mx.array(np.tile(imat.reshape(-1, 64), (n, 1))) if imat is not None else mx.ones(x.shape)
        qc, dc, mc = fit(x, base * mx.sqrt(sig2 + x * x), nmax_for(name))
        q.append(qc.reshape(n, K)); d.append(dc.reshape(n, K // 64)); m.append(mc.reshape(n, K // 64))
    return np.concatenate(q), np.concatenate(d), np.concatenate(m), imat

def quantize_tensor(name, t):
    """chunked from the GGUF rows too — never the whole tensor as f32"""
    K = int(t.shape[0]); raw = t.data
    q, d, m = [], [], []
    for r0 in range(0, raw.shape[0], CHUNK):
        xc = gguf.quants.dequantize(raw[r0:r0 + CHUNK], t.tensor_type).reshape(-1, K).astype(np.float32)
        qc, dc, mc, _ = quantize(name, xc)
        q.append(qc); d.append(dc); m.append(mc)
    return np.concatenate(q), np.concatenate(d), np.concatenate(m)

def to_q4_1(q, d, m):
    """two Q4_1 blocks per group of 64: {f16 d, f16 m, 16 bytes x[j] | x[j+16] << 4}."""
    N, K = q.shape; nb = K // 32
    dd = np.repeat(d, 2, axis=1); mm = np.repeat(m, 2, axis=1)
    qb = q.reshape(N, nb, 32)
    out = np.empty((N, nb, 20), dtype=np.uint8)
    out[:, :, 0:2] = dd.view(np.uint8).reshape(N, nb, 2)
    out[:, :, 2:4] = mm.view(np.uint8).reshape(N, nb, 2)
    out[:, :, 4:20] = (qb[:, :, :16] | (qb[:, :, 16:] << 4)).astype(np.uint8)
    return out.reshape(N, nb * 20)

def werr(a, ref, imat):
    """importance-weighted relative RMS error over columns (imatrix = E[x^2] per input column)."""
    w = imat[None, :] if imat is not None else 1.0
    return float(np.sqrt(np.sum(w * (a - ref) ** 2) / np.sum(w * ref ** 2)))

# ---------------- self-check before anything is written ----------------
for name in ["blk.0.attn_qkv.weight", "blk.0.ffn_down.weight", "blk.3.attn_q.weight", "blk.3.attn_v.weight",
             "blk.30.ffn_gate.weight", "blk.30.ffn_up.weight", "blk.62.ssm_out.weight"]:
    t0 = time.time()
    xn = dense(S[name]); ref_tpl = dense(T[name]); assert xn.shape == ref_tpl.shape, (name, xn.shape, ref_tpl.shape)
    q, d, m, imat = quantize(name, xn)
    ours = (q.reshape(q.shape[0], -1, 64).astype(np.float32) * d.astype(np.float32)[:, :, None]
            + m.astype(np.float32)[:, :, None]).reshape(xn.shape)
    plain = np.array(rtn(mx.array(xn.reshape(-1, 64)), nmax_for(name))).reshape(xn.shape)
    assert int(q.max()) <= nmax_for(name), (name, int(q.max()))
    enc = to_q4_1(q, d, m)
    back = gguf.quants.dequantize(enc.reshape(-1), Q.Q4_1).reshape(xn.shape)
    assert np.array_equal(back, ours), f"{name}: Q4_1 encoding not exact"
    print(f"check {name:<24} {nmax_for(name).bit_length()}-bit {str(xn.shape):<14} weighted rel err vs Q8_0: ours {werr(ours, xn, imat):.4f} | "
          f"plain RTN g64 {werr(plain, xn, imat):.4f} | template ({T[name].tensor_type.name}) {werr(ref_tpl, xn, imat):.4f}"
          f"  [{time.time() - t0:.1f} s]", flush=True)
if CHECK_ONLY:
    sys.exit(0)

# ---------------- write ----------------
w = gguf.GGUFWriter(OUT, rd.fields["general.architecture"].contents())
for f in rd.fields.values():
    if f.name in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"): continue
    vt = f.types[0]; sub = f.types[1] if len(f.types) > 1 else None
    val = f.contents()
    if f.name == "general.name": val = str(val) + " + Arf imatrix group-64 projections from Q8_0 (Q4_1 pairs)" + (" + lm_head" if LM_HEAD else "") + (f" + 3-bit {sorted(BITS3)}" if BITS3 else "")
    w.add_key_value(f.name, val, vt, sub)
order = [t.name for t in rd.tensors]
for name in order:
    t = T[name]
    if not is_proj(name):
        w.add_tensor_info(name, list(t.data.shape), t.data.dtype, t.n_bytes, raw_dtype=t.tensor_type)
    else:
        N, K = int(t.shape[1]), int(t.shape[0])
        w.add_tensor_info(name, [N, K // 32 * 20], np.dtype(np.uint8), N * (K // 32) * 20, raw_dtype=Q.Q4_1)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
t0 = time.time()
for k, name in enumerate(order):
    t = T[name]
    if not is_proj(name):
        w.write_tensor_data(np.asarray(t.data))
    else:
        q, d, m = quantize_tensor(name, S[name])
        assert q.shape == (int(t.shape[1]), int(t.shape[0])), (name, q.shape, t.shape)
        w.write_tensor_data(to_q4_1(q, d, m))
    if k % 50 == 0:
        print(f"[{k}/{len(order)}] {name}  {time.time() - t0:.0f} s", flush=True)
w.close()
print("DONE", OUT, round(os.path.getsize(OUT) / 1e9, 2), "GB")
