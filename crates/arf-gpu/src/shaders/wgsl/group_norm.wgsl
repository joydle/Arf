// GroupNorm over channel-major `[C, H, W]` activations, `groups` groups (VAE: 32, eps 1e-5).
// Each group covers `C/groups` consecutive channels × H*W spatial elements; normalize over that
// whole set, then apply the per-channel affine: y[c,p] = (x - mean)/sqrt(var+eps) * weight[c] + bias[c].
//
// One workgroup per group; lanes co-reduce the group's elements (mean, then variance), then
// write the normalized+affine result. Group element count = (C/groups)*H*W can be large
// (e.g. 512/32 * 128*128 = 262144), so reduce in a strided loop with a workgroup tree.

struct Dims { channels: u32, groups: u32, hw: u32, eps_bits: u32 };

@group(0) @binding(0) var<storage, read_write> x: array<f32>;       // [C, H*W]  (in place)
@group(0) @binding(1) var<storage, read>       weight: array<f32>;  // [C]
@group(0) @binding(2) var<storage, read>       bias: array<f32>;    // [C]
@group(0) @binding(3) var<uniform>             d: Dims;

const WG: u32 = 256u;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let g = wid.x;                       // group index
    let lane = lid.x;
    let cpg = d.channels / d.groups;     // channels per group
    let c0 = g * cpg;                    // first channel of this group
    let count = cpg * d.hw;              // elements in the group
    let base = c0 * d.hw;                // flat start of the group in x
    let eps = bitcast<f32>(d.eps_bits);

    // 1) mean
    var s = 0.0;
    var i = lane;
    while (i < count) { s = s + x[base + i]; i = i + WG; }
    red[lane] = s;
    workgroupBarrier();
    var st = WG / 2u;
    loop { if (st == 0u) { break; } if (lane < st) { red[lane] = red[lane] + red[lane + st]; } workgroupBarrier(); st = st / 2u; }
    let mean = red[0] / f32(count);
    workgroupBarrier();

    // 2) variance
    var vs = 0.0;
    i = lane;
    while (i < count) { let dd = x[base + i] - mean; vs = vs + dd * dd; i = i + WG; }
    red[lane] = vs;
    workgroupBarrier();
    st = WG / 2u;
    loop { if (st == 0u) { break; } if (lane < st) { red[lane] = red[lane] + red[lane + st]; } workgroupBarrier(); st = st / 2u; }
    let inv = 1.0 / sqrt(red[0] / f32(count) + eps);
    workgroupBarrier();

    // 3) normalize + per-channel affine
    i = lane;
    while (i < count) {
        let c = c0 + i / d.hw;           // channel of this element
        x[base + i] = (x[base + i] - mean) * inv * weight[c] + bias[c];
        i = i + WG;
    }
}
