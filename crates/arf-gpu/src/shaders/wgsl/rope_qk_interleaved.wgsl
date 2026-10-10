// Fused RoPE over q and k, INTERLEAVED (ggml "NORM") pairing — the twin of rope_qk.wgsl, which
// is split-half ("NEOX").
//
// WHICH MODELS: llama.cpp returns LLAMA_ROPE_TYPE_NORM for muse-glimmer
// (src/llama-model.cpp:2624); Qwen3 / Llama / Gemma are NEOX. Muse Glimmer's converter states the
// contract (conversion/muse_glimmer.py:14-16) — "HF stores Q/K in rotate_half layout, llama.cpp
// consumes the interleaved (NORM) layout" — and permutes q/k at conversion, so its GGUF is
// ALREADY in interleaved order. Applying split-half rope to those weights scrambles every q and
// k in every layer, which is coherent-machinery-wrong-distribution, not a crash.
//
// THE ONLY DIFFERENCE from rope_qk.wgsl is which two elements form a rotation pair:
//     split-half (NEOX):   (x[i],  x[i + rot_half])   for i < rot_half
//     interleaved (NORM):  (x[2p], x[2p + 1])         for p < rot_half
// The ANGLE for frequency p is the same in both, so the cos/sin tables are reused unchanged
// (`cos[pos * rot_half + p]`). Only the memory stride between partners differs.
//
// WHY A SECOND SHADER instead of permuting weights at load (which is what llama's converter does
// on the way in): q/k are Q4_K_M here, so reordering rows crosses quantization super-blocks and
// would force a dequant+requant, destroying the lossless native-Q4_K path. This costs one
// pipeline and zero extra per-token work.
//
// Bindings and Dims are byte-identical to rope_qk.wgsl so the dispatch site only swaps the
// pipeline. Partial rotary is supported the same way: only the first `rotary_dim` head dims are
// rotated, as pairs (2p, 2p+1) within that leading sub-block.

struct Dims { tokens: u32, q_heads: u32, k_heads: u32, head_dim: u32, rotary_dim: u32, _p0: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k: array<f32>;
@group(0) @binding(2) var<storage, read>       cos: array<f32>;
@group(0) @binding(3) var<storage, read>       sin: array<f32>;
@group(0) @binding(4) var<storage, read>       positions: array<u32>;
@group(0) @binding(5) var<uniform>             d: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let hd = d.head_dim;
    let rot_half = d.rotary_dim / 2u;
    let q_total = d.tokens * d.q_heads * hd;
    let k_total = d.tokens * d.k_heads * hd;
    let idx = gid.x;
    if (idx >= q_total + k_total) {
        return;
    }

    let is_q = idx < q_total;
    var local = idx;
    var heads = d.q_heads;
    if (!is_q) {
        local = idx - q_total;
        heads = d.k_heads;
    }

    // local = (t * heads + h) * hd + i — identical decomposition to rope_qk.wgsl.
    let t = local / (heads * hd);
    let rem = local % (heads * hd);
    let h = rem / hd;
    let i = rem % hd;

    // Interleaved: the EVEN element of each pair drives the rotation and writes both halves.
    // Pair p = i/2 must be inside the rotary sub-block; dims >= rotary_dim pass through.
    if ((i & 1u) != 0u || i >= d.rotary_dim) {
        return;
    }
    let p = i / 2u;

    let pos = positions[t];
    let base = (t * heads + h) * hd;
    // Same table and stride as the split-half twin: frequency p, row stride rot_half.
    let c = cos[pos * rot_half + p];
    let s = sin[pos * rot_half + p];

    if (is_q) {
        let a = q[base + i];
        let b = q[base + i + 1u];
        q[base + i]      = a * c - b * s;
        q[base + i + 1u] = b * c + a * s;
    } else {
        let a = k[base + i];
        let b = k[base + i + 1u];
        k[base + i]      = a * c - b * s;
        k[base + i + 1u] = b * c + a * s;
    }
}
