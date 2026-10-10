// Multi-head self-attention for a TEXT encoder, shared by CLIP-L (causal) and T5-XXL (full +
// relative-position bias). Score-cached like vision_attn.wgsl (one q·k per key, then softmax
// + V-weighting read the cache) so it stays under Metal's per-submission watchdog.
//
// Generalizes vision_attn with two knobs in Dims:
//   causal   : 1 → key j masked when j > qrow (CLIP). 0 → full bidirectional (T5).
//   has_bias : 1 → add rel_bias[h*seq*seq + qrow*seq + j] to the score before softmax (T5's
//              relative-position bias). 0 → rel_bias unused (bound as a 1-elem dummy).
// scale_bits: f32 bitcast of the attention scale — CLIP uses 1/sqrt(head_dim); T5 uses 1.0
//             (T5 folds the scaling into its weights and applies NO 1/sqrt(d)).
//
// Q/K/V/out are [seq, dim] row-major, dim = heads*head_dim. seq ≤ 4096 (cache bound).

struct Dims { seq: u32, dim: u32, heads: u32, head_dim: u32, causal: u32, has_bias: u32, scale_bits: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       q: array<f32>;
@group(0) @binding(1) var<storage, read>       k: array<f32>;
@group(0) @binding(2) var<storage, read>       v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read>       rel_bias: array<f32>;
@group(0) @binding(5) var<uniform>             d: Dims;

const WG: u32 = 64u;
var<workgroup> red: array<f32, 64>;
var<workgroup> scores: array<f32, 4096>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let qrow = wid.x;
    let lane = lid.x;
    if (qrow >= d.seq) { return; }
    let hd = d.head_dim;
    let scale = bitcast<f32>(d.scale_bits);

    for (var h = 0u; h < d.heads; h = h + 1u) {
        let off = h * hd;
        let qbase = qrow * d.dim + off;
        let bias_base = (h * d.seq + qrow) * d.seq;
        // upper key bound: causal → keys 0..=qrow, else all keys.
        let kmax = select(d.seq, qrow + 1u, d.causal == 1u);

        // 0) score q·k once into the cache (lane-strided), + per-lane max for stability.
        var lmax = -3.0e38;
        var j = lane;
        while (j < kmax) {
            var s = 0.0;
            let kbase = j * d.dim + off;
            for (var t = 0u; t < hd; t = t + 1u) { s = s + q[qbase + t] * k[kbase + t]; }
            s = s * scale;
            if (d.has_bias == 1u) { s = s + rel_bias[bias_base + j]; }
            scores[j] = s;
            lmax = max(lmax, s);
            j = j + WG;
        }
        red[lane] = lmax;
        workgroupBarrier();
        var st = WG / 2u;
        loop { if (st == 0u) { break; } if (lane < st) { red[lane] = max(red[lane], red[lane + st]); } workgroupBarrier(); st = st / 2u; }
        let m = red[0];
        workgroupBarrier();

        // 1) sum of exp(score - max) over attended keys.
        var lsum = 0.0;
        j = lane;
        while (j < kmax) { lsum = lsum + exp(scores[j] - m); j = j + WG; }
        red[lane] = lsum;
        workgroupBarrier();
        st = WG / 2u;
        loop { if (st == 0u) { break; } if (lane < st) { red[lane] = red[lane] + red[lane + st]; } workgroupBarrier(); st = st / 2u; }
        let denom = red[0];
        workgroupBarrier();

        // 2) weighted sum of V over attended keys → out[qrow, off+t], lanes split head_dim.
        var t = lane;
        while (t < hd) {
            var acc = 0.0;
            for (var jj = 0u; jj < kmax; jj = jj + 1u) {
                let w = exp(scores[jj] - m) / denom;
                acc = acc + w * v[jj * d.dim + off + t];
            }
            out[qrow * d.dim + off + t] = acc;
            t = t + WG;
        }
        workgroupBarrier();
    }
}
