// Per-head RMSNorm on V, in place — WEIGHTLESS (no learned gain). Gemma 4 normalizes
// the value projection per head exactly like Q/K, but with NO weight (llama.cpp:
// `Vcur_normed = RMS_NORM(Vcur)`, a bare RMS_NORM with no following MUL). This runs
// AFTER the v_proj matmul and BEFORE attention. Q/K get qk_norm.wgsl (with weight)
// + RoPE; V gets only this normalize (no weight, no RoPE).
//
//   y = x / sqrt(mean(x²) + eps)   — per (token, kv_head) row of head_dim.
//
// One workgroup per (token·kv_head) row; the row's sum-of-squares is reduced in
// shared memory, then every element is scaled. Mirrors qk_norm.wgsl's reduction.

struct Dims { rows: u32, head_dim: u32, eps: f32, _pad: u32 };

@group(0) @binding(0) var<storage, read_write> v: array<f32>;
@group(0) @binding(1) var<uniform>             d: Dims;

const WG: u32 = 64u;
var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    if (row >= d.rows) {
        return;
    }
    let lane = lid.x;
    let hd = d.head_dim;
    let base = row * hd;

    var acc = 0.0;
    var i = lane;
    while (i < hd) {
        let x = v[base + i];
        acc = acc + x * x;
        i = i + WG;
    }
    partial[lane] = acc;
    workgroupBarrier();

    var stride = WG / 2u;
    while (stride > 0u) {
        if (lane < stride) {
            partial[lane] = partial[lane] + partial[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let mean_sq = partial[0] / f32(hd);
    let inv_rms = 1.0 / sqrt(mean_sq + d.eps);

    i = lane;
    while (i < hd) {
        v[base + i] = v[base + i] * inv_rms;
        i = i + WG;
    }
}
