// Fused RMSNorm + residual-add, one workgroup per row (Gemma's post_attention /
// post_feedforward norm, which normalizes a sub-block output and adds it back to
// the residual stream):
//   normed[i] = src[i] / sqrt(mean(src²)+eps) * weight[i]
//   hidden[i] += normed[i]
// Bit-exact vs rmsnorm.wgsl (into a scratch) then add.wgsl: identical strided
// sum-of-squares with the same WG/stride tree reduction, same scale·weight, same
// accumulate order. The intermediate `normed` is only consumed by the add, so it
// is never materialized to global memory — one dispatch instead of two.

struct Dims { tokens: u32, hidden: u32, eps: f32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       src: array<f32>;
@group(0) @binding(1) var<storage, read>       weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> hidden: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;

const WG: u32 = 256u;
var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    let lane = lid.x;
    let base = row * d.hidden;

    // Each lane sums the squares of its strided slice of the row (same as rmsnorm).
    var acc = 0.0;
    var i = lane;
    while (i < d.hidden) {
        let v = src[base + i];
        acc = acc + v * v;
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

    let inv_rms = 1.0 / sqrt(partial[0] / f32(d.hidden) + d.eps);

    // Normalize and accumulate into the residual stream, in the same element order.
    i = lane;
    while (i < d.hidden) {
        hidden[base + i] = hidden[base + i] + src[base + i] * inv_rms * weight[i];
        i = i + WG;
    }
}
