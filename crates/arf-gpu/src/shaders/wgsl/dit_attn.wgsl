// Full (non-causal, no-mask) multi-head self-attention for FLUX's joint DiT sequence
// (txt 256 + img 4096 = 4352 tokens). Score-cached like text_attn.wgsl but with a 4352-entry
// cache (17 KB workgroup mem, under Metal's 32 KB limit) since the joint seq exceeds 4096.
// q/k/v are PRE-RoPE'd by the caller; this kernel only dots + softmaxes + weights V.
//
// scale = 1/sqrt(head_dim) (head_dim 128). One workgroup per query row, loops all heads.

struct Dims { seq: u32, dim: u32, heads: u32, head_dim: u32, scale_bits: u32, _p0: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read>       q: array<f32>;
@group(0) @binding(1) var<storage, read>       k: array<f32>;
@group(0) @binding(2) var<storage, read>       v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform>             d: Dims;

const WG: u32 = 64u;
var<workgroup> red: array<f32, 64>;
var<workgroup> scores: array<f32, 4352>;

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

        // 0) score q·k once into the cache (lane-strided), + per-lane max.
        var lmax = -3.0e38;
        var j = lane;
        while (j < d.seq) {
            var s = 0.0;
            let kbase = j * d.dim + off;
            for (var t = 0u; t < hd; t = t + 1u) { s = s + q[qbase + t] * k[kbase + t]; }
            s = s * scale;
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

        // 1) sum of exp(score - max).
        var lsum = 0.0;
        j = lane;
        while (j < d.seq) { lsum = lsum + exp(scores[j] - m); j = j + WG; }
        red[lane] = lsum;
        workgroupBarrier();
        st = WG / 2u;
        loop { if (st == 0u) { break; } if (lane < st) { red[lane] = red[lane] + red[lane + st]; } workgroupBarrier(); st = st / 2u; }
        let denom = red[0];
        workgroupBarrier();

        // 2) weighted sum of V → out[qrow, off+t], lanes split head_dim.
        var t = lane;
        while (t < hd) {
            var acc = 0.0;
            for (var jj = 0u; jj < d.seq; jj = jj + 1u) {
                let w = exp(scores[jj] - m) / denom;
                acc = acc + w * v[jj * d.dim + off + t];
            }
            out[qrow * d.dim + off + t] = acc;
            t = t + WG;
        }
        workgroupBarrier();
    }
}
