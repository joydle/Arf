// Standard LayerNorm WITH learnable weight + bias (SigLIP / ViT), one workgroup per row:
//   mean = mean(x);  var = mean((x-mean)^2)
//   out[i] = (x[i] - mean) / sqrt(var + eps) * weight[i] + bias[i]
// Two tree reductions (sum for mean, sum-of-squares for var) over the row, mirroring
// rmsnorm's WG/stride pattern. `x` is read-only; `out` is a separate buffer.

struct Dims { tokens: u32, hidden: u32, eps: f32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read>       weight: array<f32>;
@group(0) @binding(2) var<storage, read>       bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform>             d: Dims;

const WG: u32 = 256u;
var<workgroup> partial: array<f32, 256>;

fn reduce_sum(lane: u32, v: f32) -> f32 {
    partial[lane] = v;
    workgroupBarrier();
    var stride = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (lane < stride) { partial[lane] = partial[lane] + partial[lane + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    return partial[0];
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    let lane = lid.x;
    let base = row * d.hidden;

    // Pass 1: sum for the mean.
    var s = 0.0;
    var i = lane;
    while (i < d.hidden) { s = s + x[base + i]; i = i + WG; }
    let total = reduce_sum(lane, s);
    let mean = total / f32(d.hidden);
    workgroupBarrier();

    // Pass 2: sum of squared deviations for the variance.
    var sq = 0.0;
    i = lane;
    while (i < d.hidden) { let dv = x[base + i] - mean; sq = sq + dv * dv; i = i + WG; }
    let varsum = reduce_sum(lane, sq);
    let inv = 1.0 / sqrt(varsum / f32(d.hidden) + d.eps);

    // Pass 3: normalize + affine.
    i = lane;
    while (i < d.hidden) {
        out[base + i] = (x[base + i] - mean) * inv * weight[i] + bias[i];
        i = i + WG;
    }
}
