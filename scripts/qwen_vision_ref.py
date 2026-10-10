#!/usr/bin/env python3
"""Independent numpy reference for Arf's Qwen3.8 vision encoder (qwen3vl_merger mmproj).

Written from llama.cpp's graph, NOT from Arf's Rust:
  tools/mtmd/models/qwen3vl.cpp        (patch embed, spatial-merge permutes, blocks, merger)
  tools/mtmd/clip.cpp                  (resize_position_embeddings, the "positions" input)
  ggml/src/ggml-cpu/ops.cpp            (ggml_rope_multi VISION, ggml_interpolate align-corners,
                                        ggml_norm)
The layout work is done the ggml way -- the same reshape/permute chain qwen3vl.cpp builds,
translated to numpy axes (ggml ne0 is numpy's LAST axis) -- rather than with the explicit index
loops the Rust uses, so an index bug in one is not silently shared by the other.
Everything runs in float64.

Two knobs where references differ:
  --merger-gelu erf|tanh   erf = HF / another engine (Arf's choice), tanh = llama.cpp's ggml_gelu.

Usage:
  python3 scripts/qwen_vision_ref.py --tiny
      Rebuilds the synthetic tiny encoder of qwen_vision.rs's tests from the same xorshift32
      stream and prints (sum, sum|x|, first 4) for the golden constant TINY_GOLDEN.
  python3 scripts/qwen_vision_ref.py --mmproj models/qwen3.8-27b-gsqrco/mmproj-Qwen3.8-27B-BF16.gguf
      Runs the real encoder on the 64x64 test pattern (4x4 patches, no resize) -> REAL_GOLDEN.
Needs numpy (and the `gguf` package for --mmproj).
"""
import argparse
import math
import sys

import numpy as np


# ---------------------------------------------------------------- the synthetic tiny model
class Rng:
    """xorshift32, identical to qwen_vision.rs's test Rng."""

    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def next(self):
        x = self.s
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self.s = x
        return np.float32(np.float32(x >> 8) / np.float32(16777216.0) * np.float32(2.0) - np.float32(1.0))

    def vec(self, n, s):
        return np.array([np.float32(self.next() * np.float32(s)) for _ in range(n)], dtype=np.float32)

    def bf(self, n, s):
        return bf16_round(self.vec(n, s))


def bf16_round(a):
    """f32 -> bf16 (round to nearest even) -> f32, as arf_ml::dtype::f32_to_bf16."""
    b = a.astype(np.float32).view(np.uint32).astype(np.uint64)
    bias = ((b >> 16) & 1) + 0x7FFF
    top = ((b + bias) >> 16) & 0xFFFF
    return (top.astype(np.uint32) << 16).view(np.float32)


def tiny_model():
    d, f, pd, L = 16, 24, 48, 2
    r = Rng(0x12345678)
    m = {}
    # patch_w in Rust is the ALREADY-SUMMED [out, c*ky*kx]; model it as w0 = patch_w, w1 = 0.
    m["patch_w0"] = r.vec(d * pd, 0.2).reshape(d, 3, 4, 4)
    m["patch_w1"] = np.zeros_like(m["patch_w0"])
    m["patch_b"] = r.vec(d, 0.1)
    m["pos"] = r.vec(36 * d, 0.5).reshape(36, d)
    m["layers"] = []
    for _ in range(L):
        l = {}
        l["ln1_w"] = r.vec(d, 1.0)
        l["ln1_b"] = r.vec(d, 0.1)
        l["qkv_w"] = r.bf(3 * d * d, 0.4).reshape(3 * d, d)
        l["qkv_b"] = r.vec(3 * d, 0.1)
        l["o_w"] = r.bf(d * d, 0.3).reshape(d, d)
        l["o_b"] = r.vec(d, 0.1)
        l["ln2_w"] = r.vec(d, 1.0)
        l["ln2_b"] = r.vec(d, 0.1)
        l["up_w"] = r.bf(f * d, 0.3).reshape(f, d)
        l["up_b"] = r.vec(f, 0.1)
        l["down_w"] = r.bf(d * f, 0.3).reshape(d, f)
        l["down_b"] = r.vec(d, 0.1)
        m["layers"].append(l)
    m["post_w"] = r.vec(d, 1.0)
    m["post_b"] = r.vec(d, 0.1)
    m["mm0_w"] = r.bf(16 * d * d, 0.2).reshape(4 * d, 4 * d)
    m["mm0_b"] = r.vec(4 * d, 0.1)
    m["mm2_w"] = r.bf(12 * 4 * d, 0.2).reshape(12, 4 * d)
    m["mm2_b"] = r.vec(12, 0.1)
    cfg = dict(patch=4, hidden=d, heads=2, eps=1e-6, pos_grid=6, mean=[0.5] * 3, std=[0.5] * 3)
    return cfg, m


# ------------------------------------------------------------------------ the real mmproj
def load_mmproj(path):
    import gguf

    rd = gguf.GGUFReader(path)
    kv = {k: f.contents() for k, f in rd.fields.items() if not k.startswith("tokenizer")}
    assert kv["clip.projector_type"] == "qwen3vl_merger"
    T = {}
    for t in rd.tensors:
        a = np.asarray(t.data)
        if t.tensor_type.name == "BF16":
            a = (a.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        a = a.astype(np.float32)
        # numpy shape = reversed ggml dims
        T[t.name] = a.reshape([int(x) for x in reversed(t.shape)])
    d = int(kv["clip.vision.embedding_length"])
    m = {
        "patch_w0": T["v.patch_embd.weight"],  # (out, c, ky, kx)
        "patch_w1": T["v.patch_embd.weight.1"],
        "patch_b": T["v.patch_embd.bias"],
        "pos": T["v.position_embd.weight"],  # (2304, 1152)
        "post_w": T["v.post_ln.weight"],
        "post_b": T["v.post_ln.bias"],
        "mm0_w": T["mm.0.weight"],
        "mm0_b": T["mm.0.bias"],
        "mm2_w": T["mm.2.weight"],
        "mm2_b": T["mm.2.bias"],
        "layers": [],
    }
    for i in range(int(kv["clip.vision.block_count"])):
        p = f"v.blk.{i}."
        m["layers"].append(
            {
                "ln1_w": T[p + "ln1.weight"],
                "ln1_b": T[p + "ln1.bias"],
                "qkv_w": T[p + "attn_qkv.weight"],
                "qkv_b": T[p + "attn_qkv.bias"],
                "o_w": T[p + "attn_out.weight"],
                "o_b": T[p + "attn_out.bias"],
                "ln2_w": T[p + "ln2.weight"],
                "ln2_b": T[p + "ln2.bias"],
                "up_w": T[p + "ffn_up.weight"],
                "up_b": T[p + "ffn_up.bias"],
                "down_w": T[p + "ffn_down.weight"],
                "down_b": T[p + "ffn_down.bias"],
            }
        )
    cfg = dict(
        patch=int(kv["clip.vision.patch_size"]),
        hidden=d,
        heads=int(kv["clip.vision.attention.head_count"]),
        eps=float(kv["clip.vision.attention.layer_norm_epsilon"]),
        pos_grid=int(round(math.sqrt(m["pos"].shape[0]))),
        mean=list(kv["clip.vision.image_mean"]),
        std=list(kv["clip.vision.image_std"]),
    )
    return cfg, m


# --------------------------------------------------------------------------- the forward
def ggml_norm(x, w, b, eps):
    mean = x.mean(axis=-1, keepdims=True)
    v = x - mean
    var = (v * v).mean(axis=-1, keepdims=True)
    return v / np.sqrt(var + eps) * w + b


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def gelu_erf(x):
    erf = np.vectorize(math.erf)
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def spatial_merge(t, gw, gh, C):
    """qwen3vl.cpp's merge chain on a ggml tensor (C, gw, gh) held as numpy (gh, gw, C).
    cont_4d(2C, gw/2, gh) -> reshape_4d(2C, gw/2, 2, gh/2) -> permute(0,2,1,3) -> cont_3d(C, gw*gh)."""
    t = t.reshape(gh, gw // 2, 2 * C)  # ggml (2C, gw/2, gh)
    t = t.reshape(gh // 2, 2, gw // 2, 2 * C)  # ggml (2C, gw/2, 2, gh/2)
    t = t.transpose(0, 2, 1, 3)  # ggml permute(0,2,1,3): swap ggml dims 1,2
    return np.ascontiguousarray(t).reshape(gw * gh, C)


def interp_align_corners(pos, n, gw, gh):
    """ggml_interpolate BILINEAR|ALIGN_CORNERS of the (n x n) table to (gw x gh)."""
    C = pos.shape[1]
    src = pos.reshape(n, n, C)  # [y, x, C]
    if gw == n and gh == n:
        return src
    sf0 = (gw - 1) / (n - 1) if gw > 1 and n > 1 else gw / n
    sf1 = (gh - 1) / (n - 1) if gh > 1 and n > 1 else gh / n
    out = np.zeros((gh, gw, C))
    for i1 in range(gh):
        y = i1 / sf1
        y0 = min(max(int(math.floor(y)), 0), n - 1)
        y1 = min(max(int(math.floor(y)) + 1, 0), n - 1)
        dy = min(max(y - y0, 0.0), 1.0)
        for i0 in range(gw):
            x = i0 / sf0
            x0 = min(max(int(math.floor(x)), 0), n - 1)
            x1 = min(max(int(math.floor(x)) + 1, 0), n - 1)
            dx = min(max(x - x0, 0.0), 1.0)
            out[i1, i0] = (
                src[y0, x0] * (1 - dx) * (1 - dy)
                + src[y0, x1] * dx * (1 - dy)
                + src[y1, x0] * (1 - dx) * dy
                + src[y1, x1] * dx * dy
            )
    return out


def vision_positions(gw, gh):
    """clip.cpp's 'positions' input for QWEN3VL: 4 streams (y, x, y, x) in merge order."""
    pos = [[], [], [], []]
    for y in range(0, gh, 2):
        for x in range(0, gw, 2):
            for dy in range(2):
                for dx in range(2):
                    pos[0].append(y + dy)
                    pos[1].append(x + dx)
                    pos[2].append(y + dy)
                    pos[3].append(x + dx)
    return [np.array(p) for p in pos]


def rope_vision(x, pos, d_head):
    """ggml_rope_multi(n_dims=d_head/2, sections=[d/4]*4, GGML_ROPE_TYPE_VISION, base 1e4):
    x is (n_pos, n_head, d_head). ops.cpp: theta_scale = base^(-2/n_dims); for pair k
    (sector k, indep sections) theta = p_t*s^k for k < sec0, p_h*s^(k-sec0) for sec0 <= k < 2*sec0;
    rotate_pairs(ne0, n_offset=n_dims): (x[k], x[k+n_dims])."""
    n_dims = d_head // 2
    sec = d_head // 4
    s = 10000.0 ** (-2.0 / n_dims)
    k = np.arange(n_dims)
    p_t, p_h = pos[0][:, None], pos[1][:, None]
    theta = np.where(k[None, :] < sec, p_t * s ** k[None, :], p_h * s ** (k[None, :] - sec))
    c, sn = np.cos(theta)[:, None, :], np.sin(theta)[:, None, :]
    a, b = x[..., :n_dims], x[..., n_dims:]
    return np.concatenate([a * c - b * sn, a * sn + b * c], axis=-1)


def encode(cfg, m, pixels_chw, merger_gelu="erf"):
    P, C, H = cfg["patch"], cfg["hidden"], cfg["heads"]
    _, ih, iw = pixels_chw.shape
    gh, gw = ih // P, iw // P
    x = pixels_chw.astype(np.float64)
    # conv2d(k=P, s=P) with each temporal tap, summed (build_inp_with_temporal_merge, n_batch 1)
    patches = x.reshape(3, gh, P, gw, P).transpose(1, 3, 0, 2, 4).reshape(gh, gw, 3 * P * P)
    wsum = (m["patch_w0"].astype(np.float64) + m["patch_w1"].astype(np.float64)).reshape(C, -1)
    emb = patches @ wsum.T  # (gh, gw, C) == ggml (C, gw, gh) after the permute
    inp = spatial_merge(emb, gw, gh, C) + m["patch_b"]
    pe = interp_align_corners(m["pos"].astype(np.float64), cfg["pos_grid"], gw, gh)
    inp = inp + spatial_merge(pe, gw, gh, C)
    n = gw * gh
    pos = vision_positions(gw, gh)
    dh = C // H
    h = inp
    for l in m["layers"]:
        cur = ggml_norm(h, l["ln1_w"], l["ln1_b"], cfg["eps"])
        qkv = cur @ l["qkv_w"].T.astype(np.float64) + l["qkv_b"]
        q = qkv[:, :C].reshape(n, H, dh)
        k = qkv[:, C : 2 * C].reshape(n, H, dh)
        v = qkv[:, 2 * C :].reshape(n, H, dh)
        q, k = rope_vision(q, pos, dh), rope_vision(k, pos, dh)
        att = np.einsum("ihd,jhd->hij", q, k) / math.sqrt(dh)
        att = np.exp(att - att.max(axis=-1, keepdims=True))
        att /= att.sum(axis=-1, keepdims=True)
        o = np.einsum("hij,jhd->ihd", att, v).reshape(n, C)
        o = o @ l["o_w"].T.astype(np.float64) + l["o_b"]
        h = h + o
        cur = ggml_norm(h, l["ln2_w"], l["ln2_b"], cfg["eps"])
        up = gelu_tanh(cur @ l["up_w"].T.astype(np.float64) + l["up_b"])
        h = h + (up @ l["down_w"].T.astype(np.float64) + l["down_b"])
    h = ggml_norm(h, m["post_w"], m["post_b"], cfg["eps"])
    e = h.reshape(n // 4, 4 * C)
    e = e @ m["mm0_w"].T.astype(np.float64) + m["mm0_b"]
    e = gelu_erf(e) if merger_gelu == "erf" else gelu_tanh(e)
    return e @ m["mm2_w"].T.astype(np.float64) + m["mm2_b"]


def test_pattern(w, h):
    y, x = np.mgrid[0:h, 0:w]
    rgb = np.stack([(x * 7 + y * 3 + c * 50) % 256 for c in range(3)], axis=-1).astype(np.uint8)
    return rgb


def normalize(cfg, rgb):
    f = rgb.astype(np.float32) / np.float32(255.0)
    mean = np.array(cfg["mean"], dtype=np.float32)
    std = np.array(cfg["std"], dtype=np.float32)
    return ((f - mean) / std).transpose(2, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--mmproj")
    ap.add_argument("--merger-gelu", default="erf", choices=["erf", "tanh"])
    a = ap.parse_args()
    if a.tiny:
        cfg, m = tiny_model()
        rgb = test_pattern(24, 16)
    elif a.mmproj:
        cfg, m = load_mmproj(a.mmproj)
        rgb = test_pattern(64, 64)
    else:
        ap.print_help()
        sys.exit(2)
    out = encode(cfg, m, normalize(cfg, rgb), a.merger_gelu)
    print("shape", out.shape)
    print(f"sum={out.sum():.9f} abs={np.abs(out).sum():.9f}")
    print("first4", [f"{v:.9f}" for v in out.reshape(-1)[:4]])


if __name__ == "__main__":
    main()
