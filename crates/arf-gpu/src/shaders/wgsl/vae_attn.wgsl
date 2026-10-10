// Single-head full self-attention for the VAE mid-block, over the HW spatial sequence.
// q/k/v are `[seq, dim]` (seq = H*W, dim = channels, e.g. 512); scale = 1/sqrt(dim).
//   out[i] = softmax_j( q[i]·k[j] * scale ) · v[j]
// One workgroup per query row, lanes split the dim for the q·k dot and the V-weighted sum.
// Scores are STREAMED (recomputed in the V pass) rather than cached, so seq is unbounded — the
// VAE mid block runs at full input resolution (128² = 16384 tokens) where a score cache won't fit.
// Two streaming passes over j give the softmax max+sum, a third accumulates V (online-softmax
// would fuse these; kept explicit for parity clarity — the mid block is a small fraction of cost).

struct Dims { seq: u32, dim: u32, scale_bits: u32, _p: u32 };

@group(0) @binding(0) var<storage, read>       q: array<f32>;     // [seq, dim]
@group(0) @binding(1) var<storage, read>       k: array<f32>;
@group(0) @binding(2) var<storage, read>       v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;   // [seq, dim]
@group(0) @binding(4) var<uniform>             d: Dims;

const WG: u32 = 256u;
var<workgroup> red: array<f32, 256>;
var<workgroup> sh_score: f32;     // broadcast one j's score to all lanes

fn dot_qk(qrow: u32, j: u32, lane: u32) -> f32 {
    // lanes co-reduce q[qrow]·k[j] over dim
    var s = 0.0;
    var t = lane;
    while (t < d.dim) { s = s + q[qrow * d.dim + t] * k[j * d.dim + t]; t = t + WG; }
    red[lane] = s;
    workgroupBarrier();
    var st = WG / 2u;
    loop { if (st == 0u) { break; } if (lane < st) { red[lane] = red[lane] + red[lane + st]; } workgroupBarrier(); st = st / 2u; }
    return red[0];
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let qrow = wid.x;
    let lane = lid.x;
    if (qrow >= d.seq) { return; }
    let scale = bitcast<f32>(d.scale_bits);

    // pass 1: max score
    var m = -3.0e38;
    for (var j = 0u; j < d.seq; j = j + 1u) {
        let s = dot_qk(qrow, j, lane) * scale;
        m = max(m, s);
        workgroupBarrier();
    }
    // pass 2: sum exp
    var denom = 0.0;
    for (var j = 0u; j < d.seq; j = j + 1u) {
        let s = dot_qk(qrow, j, lane) * scale;
        denom = denom + exp(s - m);
        workgroupBarrier();
    }
    // pass 3: weighted V — lanes split dim, accumulate over j
    var t = lane;
    while (t < d.dim) {
        out[qrow * d.dim + t] = 0.0;
        t = t + WG;
    }
    workgroupBarrier();
    for (var j = 0u; j < d.seq; j = j + 1u) {
        let s = dot_qk(qrow, j, lane) * scale;   // red[0] holds the dot for all lanes after barrier
        if (lane == 0u) { sh_score = exp(s - m) / denom; }
        workgroupBarrier();
        let w = sh_score;
        var tt = lane;
        while (tt < d.dim) {
            out[qrow * d.dim + tt] = out[qrow * d.dim + tt] + w * v[j * d.dim + tt];
            tt = tt + WG;
        }
        workgroupBarrier();
    }
}
