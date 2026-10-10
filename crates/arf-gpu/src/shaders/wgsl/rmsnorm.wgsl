// RMSNorm (Llama-style: no mean subtraction, no bias).
//   y = x / sqrt(mean(x²) + eps) * weight   — per row.
//
// One workgroup per row; the row's sum-of-squares is reduced in shared memory,
// then every element is scaled. Matches RmsNorm::forward.

struct Dims { tokens: u32, hidden: u32, eps: f32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read>       weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;

const WG: u32 = 256u;
var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    let lane = lid.x;
    let base = row * d.hidden;

    // Each lane sums the squares of its strided slice of the row.
    var acc = 0.0;
    var i = lane;
    while (i < d.hidden) {
        let v = x[base + i];
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

    let mean_sq = partial[0] / f32(d.hidden);
    let inv_rms = 1.0 / sqrt(mean_sq + d.eps);

    // Scale every element by inv_rms * weight.
    i = lane;
    while (i < d.hidden) {
        y[base + i] = x[base + i] * inv_rms * weight[i];
        i = i + WG;
    }
}
