// Fused residual-add + RMSNorm, one workgroup per row (replaces add then rmsnorm):
//   hidden[i] += residual[i]                       (store back to residual stream)
//   normed[i] = hidden[i] / sqrt(mean(hidden²)+eps) * weight[i]
// Bit-exact vs add.wgsl then rmsnorm.wgsl: same updated hidden, same strided
// sum-of-squares with the identical WG/stride tree reduction, same scale·weight.
// Each lane keeps its updated values in registers (`vals`) so pass 3 never re-reads
// global hidden — avoids relying on a barrier to publish storage writes across
// lanes, and saves a DRAM round-trip. `hidden` is read_write (add) and its updated
// value feeds the reduction — all on ONE buffer in ONE dispatch, which is allowed.

struct Dims { tokens: u32, hidden: u32, eps: f32, _pad: u32 };

@group(0) @binding(0) var<storage, read_write> hidden: array<f32>;
@group(0) @binding(1) var<storage, read>       residual: array<f32>;
@group(0) @binding(2) var<storage, read>       weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> normed: array<f32>;
@group(0) @binding(4) var<uniform>             d: Dims;

const WG: u32 = 256u;
var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    let lane = lid.x;
    let base = row * d.hidden;

    // Pass 1: add in place, store, accumulate sum-of-squares of the UPDATED value.
    // `vals` holds this lane's strided slice; sized for ceil(hidden/WG) (8 at h=2048).
    var vals: array<f32, 64>;
    var acc = 0.0;
    var i = lane;
    var s = 0u;
    while (i < d.hidden) {
        let v = hidden[base + i] + residual[base + i];
        hidden[base + i] = v;
        vals[s] = v;
        acc = acc + v * v;
        i = i + WG;
        s = s + 1u;
    }
    partial[lane] = acc;
    workgroupBarrier();

    // Same tree reduction as rmsnorm.wgsl (same WG, same stride schedule).
    var stride = WG / 2u;
    while (stride > 0u) {
        if (lane < stride) {
            partial[lane] = partial[lane] + partial[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let inv_rms = 1.0 / sqrt(partial[0] / f32(d.hidden) + d.eps);

    // Pass 3: scale the values this lane already holds, in the same order.
    i = lane;
    s = 0u;
    while (i < d.hidden) {
        normed[base + i] = vals[s] * inv_rms * weight[i];
        i = i + WG;
        s = s + 1u;
    }
}
