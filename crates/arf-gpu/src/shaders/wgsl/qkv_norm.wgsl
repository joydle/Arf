// Per-head RMSNorm on q, k, AND v in ONE dispatch (Gemma 4) — fuses qk_norm.wgsl
// (q/k, with weight) and v_norm.wgsl (v, weightless) so the decode loop issues one
// dispatch per layer instead of two. q/k/v are disjoint buffers; the row index
// selects which: [0, q_rows) → q (q_weight), [q_rows, q_rows+k_rows) → k (k_weight),
// [q_rows+k_rows, q_rows+k_rows+v_rows) → v (NO weight, llama.cpp Vcur_normed).
//
//   y = x / sqrt(mean(x²) + eps) [* weight for q/k]   — per (token, head) head_dim row.
//
// Bit-IDENTICAL to qk_norm followed by v_norm: same per-row sum-of-squares, same
// inv_rms = 1/sqrt(mean_sq+eps), same per-lane weight (q/k) / none (v). Gates on the
// CPU QkNorm::apply + the weightless V RMSNorm oracles unchanged.

struct Dims { q_rows: u32, k_rows: u32, v_rows: u32, head_dim: u32, eps: f32, _p0: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k: array<f32>;
@group(0) @binding(2) var<storage, read_write> v: array<f32>;
@group(0) @binding(3) var<storage, read>       q_weight: array<f32>;
@group(0) @binding(4) var<storage, read>       k_weight: array<f32>;
@group(0) @binding(5) var<uniform>             d: Dims;

const WG: u32 = 64u;
var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) ng: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    // 2D-safe row index (same as qk_norm.wgsl: row count can exceed the 65535/dim limit).
    let row = wid.y * ng.x + wid.x;
    let total = d.q_rows + d.k_rows + d.v_rows;
    if (row >= total) {
        return;
    }
    let lane = lid.x;
    let hd = d.head_dim;

    // Region select: 0 = q, 1 = k, 2 = v.
    let is_q = row < d.q_rows;
    let is_k = (row >= d.q_rows) && (row < d.q_rows + d.k_rows);
    var local_row = row;
    if (is_k) { local_row = row - d.q_rows; }
    else if (!is_q) { local_row = row - d.q_rows - d.k_rows; }
    let base = local_row * hd;

    // Sum of squares over this row (strided per lane).
    var acc = 0.0;
    var i = lane;
    while (i < hd) {
        var x = 0.0;
        if (is_q) { x = q[base + i]; }
        else if (is_k) { x = k[base + i]; }
        else { x = v[base + i]; }
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

    // Scale: q/k get the per-head weight (reused across heads); v is weightless.
    i = lane;
    while (i < hd) {
        if (is_q) {
            q[base + i] = q[base + i] * inv_rms * q_weight[i];
        } else if (is_k) {
            k[base + i] = k[base + i] * inv_rms * k_weight[i];
        } else {
            v[base + i] = v[base + i] * inv_rms;
        }
        i = i + WG;
    }
}
