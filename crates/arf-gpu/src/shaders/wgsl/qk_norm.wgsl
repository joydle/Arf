// Per-head RMSNorm on q and k, in place, before RoPE (Qwen3/Gemma).
//   y = x / sqrt(mean(x²) + eps) * weight   — per (token, head) row of head_dim.
//
// Fused over q and k in one dispatch, mirroring rope_qk.wgsl's index split:
// workgroup ids [0, q_rows) normalize q with q_norm weight; [q_rows, q_rows+k_rows)
// normalize k with k_norm weight. Each row is one head's head_dim slice at
// (t*heads + h)*head_dim. The norm weight is length head_dim, shared across all
// heads (so the per-head row reuses weight[i]). Bit-exact vs QkNorm::apply on CPU:
// same sum-of-squares, same inv_rms = 1/sqrt(mean_sq+eps), same per-lane weight.

struct Dims { q_rows: u32, k_rows: u32, head_dim: u32, eps: f32 };

@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k: array<f32>;
@group(0) @binding(2) var<storage, read>       q_weight: array<f32>;
@group(0) @binding(3) var<storage, read>       k_weight: array<f32>;
@group(0) @binding(4) var<uniform>             d: Dims;

const WG: u32 = 64u;
var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) ng: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    // 2D-safe row index: the (q_rows+k_rows) row count exceeds the 65535 workgroups-per-dim limit
    // at high resolution (e.g. 1024px → 196608 rows), so the dispatch grid is [x, y, 1] and the
    // row is reconstructed row-major. ng.x is the x-extent of the grid.
    let row = wid.y * ng.x + wid.x;
    if (row >= d.q_rows + d.k_rows) {
        return;
    }
    let lane = lid.x;
    let hd = d.head_dim;
    let is_q = row < d.q_rows;
    let local_row = select(row - d.q_rows, row, is_q);
    let base = local_row * hd;

    // Each lane sums the squares of its strided slice of the head_dim row.
    var acc = 0.0;
    var i = lane;
    while (i < hd) {
        let v = select(k[base + i], q[base + i], is_q);
        acc = acc + v * v;
        i = i + WG;
    }
    partial[lane] = acc;
    workgroupBarrier();

    // Tree reduction over the workgroup.
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

    // Scale every element by inv_rms * weight[i] (weight is head_dim, reused/head).
    i = lane;
    while (i < hd) {
        if (is_q) {
            q[base + i] = q[base + i] * inv_rms * q_weight[i];
        } else {
            k[base + i] = k[base + i] * inv_rms * k_weight[i];
        }
        i = i + WG;
    }
}
