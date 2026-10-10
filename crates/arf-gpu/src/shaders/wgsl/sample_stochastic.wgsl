// Stochastic token sampling over logits[vocab] -> one token id in out[0].
//
// Faithful single-workgroup GPU port of arf_core::sampling::filtered_probs +
// the WeightedIndex draw, using the THRESHOLD method (no host sort):
//
//   1. temperature: scale logits by inv_temp (= 1/temperature).
//   2. top_k:  keep the k highest-logit tokens. We find the k-th largest
//      temperature-scaled logit `tau_k` by binary search on the logit axis
//      (count(scaled_logit >= tau) is monotonic decreasing in tau), then the
//      survivor set is {scaled_logit >= tau_k}. For DISTINCT logits this is
//      exactly the CPU sort-then-truncate set; at an exact tie on the boundary
//      both the CPU (sort_unstable) and this kernel realize an ambiguous spec
//      and pick a subset of the tied group.
//   3. softmax over the survivors with the standard max-subtract (max = the
//      single global max scaled logit; p_t = exp(scaled_logit_t - max), summed
//      over survivors).
//   4. top_p:  the CPU keeps the SMALLEST prefix of the prob-DESC-sorted
//      survivors whose cumulative prob >= top_p. Since prob is a strictly
//      increasing function of the logit, prob-DESC order == logit-DESC order, so
//      the kept set is the highest-prob survivors forming the minimal mass-
//      >=top_p prefix. We find the prob threshold `tau_p` = largest threshold
//      with mass(prob >= tau_p) >= top_p (binary search; mass is monotonic), and
//      the kept set is {survivor : prob >= tau_p}. For distinct probs this is
//      exactly the CPU prefix; the global-max token is always kept (it has the
//      largest prob, so it is in every {prob >= tau_p} set for tau_p in [0,1] —
//      this is the GPU analogue of the CPU's `keep.max(1)`).
//   5. weighted draw:  Skept = sum of kept (un-renormalized softmax) weights;
//      target = u * Skept where u in [0,1) comes from a deterministic counter-
//      based PCG seeded by (seed, position); walk the kept tokens in a fixed
//      canonical order accumulating weight and pick the first whose running
//      cumulative exceeds `target`. Inverse-CDF over the kept categorical — the
//      walk order does not change any token's selection probability, so this
//      matches the CPU's WeightedIndex distribution (the RNG STREAM differs from
//      the CPU's ChaCha, by design; only the distribution is asserted in tests).
//
// Single workgroup (WG=256), workgroup-memory tree reductions for max / count /
// mass / sum, mirroring sample.wgsl. Greedy (temperature==0) is handled by the
// separate sample.wgsl fast path; this kernel is only dispatched when temperature>0.

struct Dims {
    vocab: u32,        // number of logits
    inv_temp: f32,     // 1.0 / temperature (temperature>0 guaranteed by the host)
    top_k: u32,        // 0 = disabled, else keep the k highest-logit tokens
    top_p: f32,        // >=1 = disabled, else nucleus mass in (0,1]
    seed_lo: u32,      // RNG seed low 32 bits
    seed_hi: u32,      // RNG seed high 32 bits
    position: u32,     // decode position; feeds the RNG counter so each step differs
    _pad: u32,
};

@group(0) @binding(0) var<storage, read>       logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@group(0) @binding(2) var<uniform>             d: Dims;

const WG: u32 = 256u;
const NEG_INF: f32 = -3.4e38;

// Shared reduction scratch (reused across phases).
var<workgroup> red_f: array<f32, 256>;
var<workgroup> red_u: array<u32, 256>;
var<workgroup> red_i: array<u32, 256>; // argmax index companion to red_f
// Broadcast slots written by lane 0 and read by all lanes after a barrier.
var<workgroup> sh_maxlog: f32; // global max of inv_temp-scaled logits
var<workgroup> sh_minlog: f32; // global min of inv_temp-scaled logits
var<workgroup> sh_argmax: u32; // global argmax token id (lowest index on ties)
var<workgroup> sh_tau_k: f32;  // top-k logit threshold (scaled-logit axis)
var<workgroup> sh_softsum: f32; // sum of exp(scaled - max) over the top-k survivors
var<workgroup> sh_tau_p: f32;  // top-p prob threshold
var<workgroup> sh_skept: f32;  // sum of kept (un-renormalized softmax) weights
var<workgroup> sh_lane_excl: array<f32, 256>; // exclusive prefix of per-lane kept sums
var<workgroup> sh_winner: u32; // selected token id

// --- counter-based PCG RNG --------------------------------------------------
// pcg output permutation on a 32-bit LCG-advanced state. Deterministic and
// stateless: same inputs -> same output, so (seed, position) reproduces u.
fn pcg(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Hash (seed_lo, seed_hi, position) into a uniform f32 in [0,1). Fold each input
// through a pcg round so all three decorrelate the stream.
fn rng_uniform(seed_lo: u32, seed_hi: u32, pos: u32) -> f32 {
    var h = pcg(seed_lo);
    h = pcg(h ^ seed_hi);
    h = pcg(h ^ pos);
    h = pcg(h ^ 0x9e3779b9u); // one extra avalanche round
    // 24-bit mantissa -> [0,1); never reaches 1.0.
    return f32(h >> 8u) * (1.0 / 16777216.0);
}

// Tree-reduce red_f[] with max; result in red_f[0] (all lanes synced after).
fn reduce_max(lane: u32) {
    var stride = WG / 2u;
    while (stride > 0u) {
        if (lane < stride) {
            red_f[lane] = max(red_f[lane], red_f[lane + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
}

// Tree-reduce red_f[] with min; result in red_f[0].
fn reduce_min(lane: u32) {
    var stride = WG / 2u;
    while (stride > 0u) {
        if (lane < stride) {
            red_f[lane] = min(red_f[lane], red_f[lane + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
}

// Tree-reduce (red_f as value, red_i as index) keeping the max value with the
// lowest index on ties; result in slot 0. Matches sample.wgsl's argmax.
fn reduce_argmax(lane: u32) {
    var stride = WG / 2u;
    while (stride > 0u) {
        if (lane < stride) {
            let ov = red_f[lane + stride];
            let oi = red_i[lane + stride];
            if (ov > red_f[lane] || (ov == red_f[lane] && oi < red_i[lane])) {
                red_f[lane] = ov;
                red_i[lane] = oi;
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
}

// Tree-reduce red_f[] with sum; result in red_f[0].
fn reduce_sum(lane: u32) {
    var stride = WG / 2u;
    while (stride > 0u) {
        if (lane < stride) {
            red_f[lane] = red_f[lane] + red_f[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
}

// Tree-reduce red_u[] with sum; result in red_u[0].
fn reduce_sum_u(lane: u32) {
    var stride = WG / 2u;
    while (stride > 0u) {
        if (lane < stride) {
            red_u[lane] = red_u[lane] + red_u[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let inv_temp = d.inv_temp;

    // ---- Phase 1: global max, min, and argmax of the scaled logits ----------
    var lmax = NEG_INF;
    var lmin = -NEG_INF;
    var amax_v = NEG_INF;
    var amax_i = 0u;
    var i = lane;
    while (i < d.vocab) {
        let s = logits[i] * inv_temp;
        lmax = max(lmax, s);
        lmin = min(lmin, s);
        if (s > amax_v) { amax_v = s; amax_i = i; }
        i = i + WG;
    }
    // max
    red_f[lane] = lmax;
    workgroupBarrier();
    reduce_max(lane);
    if (lane == 0u) { sh_maxlog = red_f[0]; }
    workgroupBarrier();
    // min
    red_f[lane] = lmin;
    workgroupBarrier();
    reduce_min(lane);
    if (lane == 0u) { sh_minlog = red_f[0]; }
    workgroupBarrier();
    // argmax (lowest index on ties)
    red_f[lane] = amax_v;
    red_i[lane] = amax_i;
    workgroupBarrier();
    reduce_argmax(lane);
    if (lane == 0u) { sh_argmax = red_i[0]; }
    workgroupBarrier();

    let gmax = sh_maxlog;
    let gmin = sh_minlog;

    // ---- Phase 2: top-k logit threshold via binary search -------------------
    // Largest tau with count(scaled_logit >= tau) >= top_k. Survivors are
    // {scaled_logit >= tau_k}. top_k==0 (or >= vocab) disables: tau_k below gmin.
    if (d.top_k == 0u || d.top_k >= d.vocab) {
        if (lane == 0u) { sh_tau_k = gmin - 1.0; }
        workgroupBarrier();
    } else {
        var lo = gmin;
        var hi = gmax;
        // 50 bisections drive the interval far below f32 logit resolution.
        for (var it = 0u; it < 50u; it = it + 1u) {
            let mid = 0.5 * (lo + hi);
            var cnt = 0u;
            var j = lane;
            while (j < d.vocab) {
                if (logits[j] * inv_temp >= mid) { cnt = cnt + 1u; }
                j = j + WG;
            }
            red_u[lane] = cnt;
            workgroupBarrier();
            reduce_sum_u(lane);
            let total = red_u[0];
            workgroupBarrier();
            if (total >= d.top_k) {
                lo = mid; // threshold can rise (still keeps >= k)
            } else {
                hi = mid;
            }
            workgroupBarrier();
        }
        if (lane == 0u) { sh_tau_k = lo; }
        workgroupBarrier();
    }
    let tau_k = sh_tau_k;

    // ---- Phase 3: softmax sum over the top-k survivors ----------------------
    var ssum = 0.0;
    i = lane;
    while (i < d.vocab) {
        let s = logits[i] * inv_temp;
        if (s >= tau_k) {
            ssum = ssum + exp(s - gmax);
        }
        i = i + WG;
    }
    red_f[lane] = ssum;
    workgroupBarrier();
    reduce_sum(lane);
    if (lane == 0u) { sh_softsum = red_f[0]; }
    workgroupBarrier();
    let softsum = sh_softsum;

    // ---- Phase 4: top-p prob threshold via binary search --------------------
    // prob_t = exp(scaled_logit_t - gmax) / softsum (survivors only). Largest
    // tau_p with mass(prob >= tau_p) >= top_p; kept = {survivor : prob >= tau_p}.
    // top_p >= 1 disables: tau_p <= 0 keeps every survivor.
    if (d.top_p >= 1.0) {
        if (lane == 0u) { sh_tau_p = -1.0; }
        workgroupBarrier();
    } else {
        var lo = 0.0;
        var hi = 1.0; // a single prob can be at most 1.0
        for (var it = 0u; it < 50u; it = it + 1u) {
            let mid = 0.5 * (lo + hi);
            var mass = 0.0;
            var j = lane;
            while (j < d.vocab) {
                let s = logits[j] * inv_temp;
                if (s >= tau_k) {
                    let p = exp(s - gmax) / softsum;
                    if (p >= mid) { mass = mass + p; }
                }
                j = j + WG;
            }
            red_f[lane] = mass;
            workgroupBarrier();
            reduce_sum(lane);
            let total = red_f[0];
            workgroupBarrier();
            if (total >= d.top_p) {
                lo = mid; // threshold can rise (mass still covers top_p)
            } else {
                hi = mid;
            }
            workgroupBarrier();
        }
        if (lane == 0u) { sh_tau_p = lo; }
        workgroupBarrier();
    }
    let tau_p = sh_tau_p;

    // ---- Phase 5a: per-lane sum of kept (un-renormalized softmax) weights ----
    // A token is KEPT iff it survived top-k (scaled >= tau_k) AND passes top-p
    // (prob >= tau_p). We accumulate the UN-normalized softmax weight
    // exp(scaled-gmax); the draw target scales by this kept sum, exactly like
    // WeightedIndex normalizing by the kept total.
    var lane_sum = 0.0;
    i = lane;
    while (i < d.vocab) {
        let s = logits[i] * inv_temp;
        if (s >= tau_k) {
            let w = exp(s - gmax);
            if ((w / softsum) >= tau_p) {
                lane_sum = lane_sum + w;
            }
        }
        i = i + WG;
    }
    red_f[lane] = lane_sum;
    workgroupBarrier();
    reduce_sum(lane); // red_f[0] = Skept (all lanes see it after the barrier)
    if (lane == 0u) { sh_skept = red_f[0]; }
    workgroupBarrier();
    let skept = sh_skept;

    // Re-derive per-lane sums (red_f was overwritten by the reduction) and build
    // the exclusive prefix over lanes. Canonical walk order is lane-major
    // (lane 0's whole strided slice, then lane 1's, ...); each lane's cumulative
    // starts at the sum of all earlier lanes' kept weights.
    red_f[lane] = lane_sum;
    workgroupBarrier();
    if (lane == 0u) {
        var acc = 0.0;
        for (var l = 0u; l < WG; l = l + 1u) {
            sh_lane_excl[l] = acc;
            acc = acc + red_f[l];
        }
        // Default winner: the global argmax. Used when skept == 0 (degenerate)
        // or float rounding leaves no lane crossing the target.
        sh_winner = sh_argmax;
    }
    workgroupBarrier();

    // ---- Phase 5b: deterministic uniform + inverse-CDF crossover ------------
    let u = rng_uniform(d.seed_lo, d.seed_hi, d.position);
    let tgt = u * skept;

    // Only the lane whose [excl, excl+lane_sum) interval contains `tgt` can hold
    // the crossover; it serially walks its own strided slice (ascending index)
    // and records the first kept token whose running cumulative exceeds the
    // target. Exactly one lane qualifies, so no atomics are needed.
    let my_excl = sh_lane_excl[lane];
    if (skept > 0.0 && lane_sum > 0.0 && tgt >= my_excl && tgt < my_excl + lane_sum) {
        var cum = my_excl;
        var j = lane;
        loop {
            if (j >= d.vocab) { break; }
            let s = logits[j] * inv_temp;
            if (s >= tau_k) {
                let w = exp(s - gmax);
                if ((w / softsum) >= tau_p) {
                    cum = cum + w;
                    if (cum > tgt) {
                        sh_winner = j;
                        break;
                    }
                }
            }
            j = j + WG;
        }
    }
    workgroupBarrier();

    if (lane == 0u) {
        out[0] = sh_winner;
    }
}
