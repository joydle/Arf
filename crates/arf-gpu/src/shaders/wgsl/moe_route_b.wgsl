// BATCHED MoE router top-k selection (B rows / sequences). One workgroup per row:
// dispatch B workgroups, wid.x = row r. Each workgroup runs row r's full softmax
// over `logits[r*num_experts .. r*num_experts + num_experts]`, picks the `top_k`
// highest by probability (ties to the lower index), optionally renormalizes the
// selected weights (Qwen3 norm_topk_prob), and writes ids[r*top_k .. ] +
// weights[r*top_k .. ].
//
// This is the per-row generalization of moe_route.wgsl: the per-row math (the
// shared-memory tree MAX, tree SUM of exp(), and k passes of tree ARGMAX with
// lowest-index-wins) is BYTE-IDENTICAL to the single-token kernel — only the
// per-row base offsets (lbase for logits, obase for ids/weights) are added. With
// B=1 (lbase=0, obase=0) it reduces exactly to moe_route.wgsl, so the B=1 route
// parity test stays green and the batched output for row r equals the single-token
// route of row r's logits. See moe_route.wgsl for the tie-break/monotonicity proof.
//
// d.norm_topk != 0 enables renormalization.

struct Dims { num_experts: u32, top_k: u32, norm_topk: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       logits: array<f32>;   // [B * num_experts]
@group(0) @binding(1) var<storage, read_write> ids: array<u32>;      // [B * top_k]
@group(0) @binding(2) var<storage, read_write> weights: array<f32>;  // [B * top_k]
@group(0) @binding(3) var<uniform>             d: Dims;

const WG: u32 = 128u;
const MAX_EXPERTS: u32 = 256u;
const NEG_INF: f32 = -3.4e38;

// Tree-reduction scratch. `red_*` hold the per-lane partial then the reduced result.
var<workgroup> red_val: array<f32, 128>;   // max / sum / argmax value
var<workgroup> red_idx: array<u32, 128>;   // argmax index carried alongside the value
var<workgroup> sh_max: f32;                 // broadcast softmax max
var<workgroup> sh_sum: f32;                 // broadcast softmax denominator
var<workgroup> taken: array<bool, MAX_EXPERTS>;  // experts already selected

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let n = d.num_experts;
    let k = min(d.top_k, n);
    // Per-row bases: this workgroup owns row `wid.x`.
    let row = wid.x;
    let lbase = row * n;          // row r's logit segment
    let obase = row * d.top_k;    // row r's ids/weights segment

    // --- Parallel MAX over logits[lbase .. lbase+n] (tree reduction). max is
    //     associative and order-independent, matching the CPU's f32::max fold. ---
    var m = NEG_INF;
    var i = lane;
    while (i < n) {
        m = max(m, logits[lbase + i]);
        i = i + WG;
    }
    red_val[lane] = m;
    workgroupBarrier();
    var stride = WG / 2u;
    while (stride > 0u) {
        if (lane < stride) {
            red_val[lane] = max(red_val[lane], red_val[lane + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lane == 0u) {
        sh_max = red_val[0];
    }
    workgroupBarrier();
    let maxv = sh_max;

    // --- Parallel SUM of exp(logit - max) (tree reduction). ---
    var s = 0.0;
    i = lane;
    while (i < n) {
        s = s + exp(logits[lbase + i] - maxv);
        i = i + WG;
    }
    red_val[lane] = s;
    workgroupBarrier();
    stride = WG / 2u;
    while (stride > 0u) {
        if (lane < stride) {
            red_val[lane] = red_val[lane] + red_val[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lane == 0u) {
        sh_sum = red_val[0];
    }
    workgroupBarrier();
    let sum = sh_sum;

    // --- Clear the taken mask in parallel. ---
    i = lane;
    while (i < n) {
        taken[i] = false;
        i = i + WG;
    }
    workgroupBarrier();

    // --- k iterations of a parallel ARGMAX over not-yet-taken experts. ---
    // Argmax on the raw logit (monotonic in prob); ties broken to the LOWEST index.
    for (var slot = 0u; slot < k; slot = slot + 1u) {
        var bv = NEG_INF;
        var bi = 0xFFFFFFFFu;
        i = lane;
        while (i < n) {
            if (!taken[i]) {
                let v = logits[lbase + i];
                if (v > bv) {
                    bv = v;
                    bi = i;
                }
            }
            i = i + WG;
        }
        red_val[lane] = bv;
        red_idx[lane] = bi;
        workgroupBarrier();

        stride = WG / 2u;
        while (stride > 0u) {
            if (lane < stride) {
                let ov = red_val[lane + stride];
                let oi = red_idx[lane + stride];
                let cv = red_val[lane];
                let ci = red_idx[lane];
                if (ov > cv || (ov == cv && oi < ci)) {
                    red_val[lane] = ov;
                    red_idx[lane] = oi;
                }
            }
            workgroupBarrier();
            stride = stride / 2u;
        }

        if (lane == 0u) {
            let best = red_idx[0];
            taken[best] = true;
            ids[obase + slot] = best;
            weights[obase + slot] = exp(logits[lbase + best] - maxv) / sum;
        }
        workgroupBarrier();
    }

    // --- Renormalize the k selected weights (lane 0; k is tiny). ---
    if (lane == 0u && d.norm_topk != 0u) {
        var wsum = 0.0;
        for (var slot = 0u; slot < k; slot = slot + 1u) {
            wsum = wsum + weights[obase + slot];
        }
        if (wsum > 0.0) {
            for (var slot = 0u; slot < k; slot = slot + 1u) {
                weights[obase + slot] = weights[obase + slot] / wsum;
            }
        }
    }
}
