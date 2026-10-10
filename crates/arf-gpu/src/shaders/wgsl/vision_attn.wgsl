// Bidirectional (full, unmasked) multi-head self-attention for a ViT encoder (SigLIP).
// Q/K/V are [seq, dim] row-major, dim = heads*head_dim. No causal mask, no RoPE, no
// KV-cache — every query attends to every key. One workgroup per (query row); it loops
// over all heads. scale = 1/sqrt(head_dim).
//
// PERF: the QK score q·k is computed ONCE per (query,key,head) into a workgroup-resident
// score cache, then the softmax max/sum and the V-weighting all read the cache. The naive
// "recompute the dot in every pass" version re-walked all seq keys × head_dim in pass 3
// for every output dim — ~head_dim× more FLOPs — which blew past Metal's per-submission
// GPU watchdog at seq=4096. Caching makes the whole layer a single fast submission.
//
// Layout note: head h occupies columns [h*hd, (h+1)*hd) of each row.
// Cache size: SigLIP seq = num_patches = 4096 (fixed for Gemma-3-4B). MAX_SEQ f32 = 16 KB
// of workgroup memory, well under the 32 KB Metal limit.

struct Dims { seq: u32, dim: u32, heads: u32, head_dim: u32 };

@group(0) @binding(0) var<storage, read>       q: array<f32>;
@group(0) @binding(1) var<storage, read>       k: array<f32>;
@group(0) @binding(2) var<storage, read>       v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform>             d: Dims;

const WG: u32 = 64u;
const MAX_SEQ: u32 = 4096u;
var<workgroup> red: array<f32, 64>;          // tree-reduction scratch
var<workgroup> scores: array<f32, 4096>;     // cached q·k per key (this head)

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let qrow = wid.x;            // query position
    let lane = lid.x;
    if (qrow >= d.seq) { return; }
    let hd = d.head_dim;
    let scale = 1.0 / sqrt(f32(hd));

    for (var h = 0u; h < d.heads; h = h + 1u) {
        let off = h * hd;        // column offset for this head
        let qbase = qrow * d.dim + off;

        // 0) compute every score q·k ONCE into the cache (lane-strided over keys), and
        //    track this lane's local max for stability.
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

        // 1) sum of exp(score - max) over the cached scores — lane-strided.
        var lsum = 0.0;
        j = lane;
        while (j < d.seq) { lsum = lsum + exp(scores[j] - m); j = j + WG; }
        red[lane] = lsum;
        workgroupBarrier();
        st = WG / 2u;
        loop { if (st == 0u) { break; } if (lane < st) { red[lane] = red[lane] + red[lane + st]; } workgroupBarrier(); st = st / 2u; }
        let denom = red[0];
        workgroupBarrier();

        // 2) weighted sum of V → out[qrow, off + t], lanes split the head_dim. Reads the
        //    cached softmax weight per key — NO score recompute.
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
        workgroupBarrier();   // protect `scores` before the next head overwrites it
    }
}
