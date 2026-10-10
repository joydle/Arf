// MoE router top-k selection (batch-1 decode). ONE workgroup of WG lanes:
// full softmax over `logits[num_experts]`, pick the `top_k` highest by probability
// (ties to the lower index, matching argsort), optionally renormalize the selected
// weights to sum to 1 (Qwen3 norm_topk_prob), and write ids[top_k] + weights[top_k].
//
// num_experts is small (n≤128, guarded n≤MAX_EXPERTS=256) and top_k tiny (k≤8). The
// single-thread version of this kernel was ~37% of decode time, so this is a full-
// workgroup rewrite: a shared-memory tree MAX, a shared-memory tree SUM of exp(),
// then k iterations of a shared-memory tree ARGMAX (over not-yet-taken experts).
//
// Bit-exact vs the CPU `top_k_softmax` oracle in model::moe (same full-softmax-then-
// select-then-renorm). The CPU writes weight[s] = probs[idx[s]] = exp(logit-max)/sum,
// so we write that exact expression for the winning expert.
//
// TIE-BREAK: the CPU sorts by (descending prob, then ascending index), i.e. on equal
// prob the LOWER index wins. The parallel argmax reduction below combines two
// (value,index) candidates with: `a` beats `b` iff a.val > b.val OR (a.val == b.val
// AND a.idx < b.idx) — exactly lowest-index-wins. We argmax on the raw LOGIT, not the
// prob: exp()/sum is strictly monotonic, so logit[i] > logit[j] ⟺ prob[i] > prob[j]
// and logit[i] == logit[j] ⟺ prob[i] == prob[j]; the selection (and its tie set) is
// therefore identical, while avoiding an exp() per element per pass plus any
// exp-rounding tie artifact. The WRITTEN weight is still the prob (exp(logit-max)/sum).
//
// d.norm_topk != 0 enables renormalization.

struct Dims { num_experts: u32, top_k: u32, norm_topk: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> weights: array<f32>;
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
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let n = d.num_experts;
    let k = min(d.top_k, n);

    // --- Parallel MAX over logits[0..n] (tree reduction). max is associative and
    //     order-independent, so this matches the CPU's f32::max fold exactly. ---
    var m = NEG_INF;
    var i = lane;
    while (i < n) {
        m = max(m, logits[i]);
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
        s = s + exp(logits[i] - maxv);
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
    // prob[i] = exp(logits[i] - maxv) / sum, computed on demand.

    // --- Clear the taken mask in parallel. ---
    i = lane;
    while (i < n) {
        taken[i] = false;
        i = i + WG;
    }
    workgroupBarrier();

    // --- k iterations of a parallel ARGMAX over not-yet-taken experts. ---
    // We argmax on the raw logit (monotonic in prob); the winner is the highest
    // logit, ties broken to the LOWEST index (matching the CPU argsort). k is tiny so
    // the outer loop over slots stays sequential.
    for (var slot = 0u; slot < k; slot = slot + 1u) {
        // Each lane scans its strided slice, keeping the best (logit, idx) it sees.
        // A taken expert is excluded by giving it value NEG_INF. Ties within a lane's
        // own slice resolve to the lower index because we only replace on STRICTLY
        // greater value (ascending scan, first max wins).
        var bv = NEG_INF;
        var bi = 0xFFFFFFFFu;
        i = lane;
        while (i < n) {
            if (!taken[i]) {
                let v = logits[i];
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

        // Tree reduction with the lowest-index-wins combine: keep `lane` unless the
        // partner strictly beats it (greater value, or equal value AND lower index).
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

        // Lane 0 commits the winner: mark taken, write id + prob weight.
        if (lane == 0u) {
            let best = red_idx[0];
            taken[best] = true;
            ids[slot] = best;
            weights[slot] = exp(logits[best] - maxv) / sum;
        }
        workgroupBarrier();
    }

    // --- Renormalize the k selected weights (lane 0; k is tiny). ---
    if (lane == 0u && d.norm_topk != 0u) {
        var wsum = 0.0;
        for (var slot = 0u; slot < k; slot = slot + 1u) {
            wsum = wsum + weights[slot];
        }
        if (wsum > 0.0) {
            for (var slot = 0u; slot < k; slot = slot + 1u) {
                weights[slot] = weights[slot] / wsum;
            }
        }
    }
}
