// LayerNorm with NO affine (elementwise_affine=False) — FLUX's DiT norms (the affine comes from
// adaLN instead). out[i] = (x[i] - mean) / sqrt(var + eps). One workgroup per row, two tree
// reductions, mirroring layernorm_bias.wgsl minus the weight/bias.

struct Dims { tokens: u32, hidden: u32, eps: f32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             d: Dims;

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

    var s = 0.0;
    var i = lane;
    while (i < d.hidden) { s = s + x[base + i]; i = i + WG; }
    let mean = reduce_sum(lane, s) / f32(d.hidden);
    workgroupBarrier();

    var sq = 0.0;
    i = lane;
    while (i < d.hidden) { let dv = x[base + i] - mean; sq = sq + dv * dv; i = i + WG; }
    let inv = 1.0 / sqrt(reduce_sum(lane, sq) / f32(d.hidden) + d.eps);

    i = lane;
    while (i < d.hidden) { out[base + i] = (x[base + i] - mean) * inv; i = i + WG; }
}
