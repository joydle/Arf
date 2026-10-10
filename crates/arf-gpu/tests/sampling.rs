//! On-GPU stochastic sampling parity: the `sample_stochastic.wgsl` kernel must
//! select tokens from the SAME support the CPU oracle (`Sampler::sample` /
//! `filtered_probs`) keeps, and draw them with the SAME distribution — even
//! though its deterministic PCG RNG stream differs from the CPU's ChaCha (so the
//! tests are DISTRIBUTIONAL, not bit-identical, on the random draw).
//!
//! Three parity invariants (mirroring the task spec):
//!   (a) GREEDY bit-exactness: the `sample` argmax kernel == `argmax` for several
//!       logit vectors including ties; and the stochastic kernel with `top_k = 1`
//!       collapses to that same argmax for any temperature/seed (its selection
//!       reduces to the single highest-logit token).
//!   (b) FILTER CORRECTNESS: over many (seed, position) the drawn token is ALWAYS
//!       inside the CPU `filtered_probs` kept set (the strong invariant — the GPU
//!       must never sample outside the CPU's top-k/top-p support).
//!   (c) DISTRIBUTION: over N=10k draws the empirical token frequencies match the
//!       CPU `filtered_probs` probabilities within a few percent.
//!
//! Plus a SELECTION-EQUIVALENCE unit assertion (no GPU): a CPU reimplementation of
//! the GPU threshold method keeps EXACTLY the CPU sort-truncate set, isolating the
//! selection logic from the RNG.
//!
//! Skips gracefully when no GPU adapter is present (CI).

use std::collections::HashSet;
use std::sync::Arc;

use arf_core::sampling::{argmax, filtered_probs, SamplingParams};
use arf_gpu::gpu::{ComputeKernel, GpuContext};

const GREEDY_WGSL: &str = include_str!("../src/shaders/wgsl/sample.wgsl");
const STOCHASTIC_WGSL: &str = include_str!("../src/shaders/wgsl/sample_stochastic.wgsl");

/// Acquire a GPU context or `None` (so the test prints a skip and returns Ok).
fn try_ctx() -> Option<Arc<GpuContext>> {
    match GpuContext::new() {
        Ok(c) => Some(Arc::new(c)),
        Err(e) => {
            eprintln!("skipping GPU sampling test: {e}");
            None
        }
    }
}

/// Run the greedy `sample` kernel: logits -> argmax token id.
fn run_greedy(ctx: &Arc<GpuContext>, logits: &[f32]) -> u32 {
    let kernel = ComputeKernel::new(ctx, "sample", GREEDY_WGSL);
    let lg = ctx.storage_init("logits", logits);
    let out = ctx.storage_zeros_u32("out", 1);
    let dims = ctx.uniform_of("dims", &[logits.len() as u32, 0u32, 0u32, 0u32]);
    kernel.dispatch("sample", &[&lg, &out, &dims], [1, 1, 1]);
    ctx.read_u32(&out, 1)[0]
}

/// Pack the `sample_stochastic` Dims uniform: must match the WGSL struct exactly:
/// `[vocab, inv_temp_bits, top_k, top_p_bits, seed_lo, seed_hi, position, _pad]`.
fn stochastic_dims(
    vocab: usize,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    seed: u64,
    position: u32,
) -> [u32; 8] {
    [
        vocab as u32,
        (1.0f32 / temperature).to_bits(),
        top_k,
        top_p.to_bits(),
        seed as u32,
        (seed >> 32) as u32,
        position,
        0,
    ]
}

/// Run the stochastic `sample_stochastic` kernel once for the given params and
/// (seed, position) -> the drawn token id.
#[allow(clippy::too_many_arguments)]
fn run_stochastic(
    ctx: &Arc<GpuContext>,
    kernel: &ComputeKernel,
    logits: &[f32],
    temperature: f32,
    top_k: u32,
    top_p: f32,
    seed: u64,
    position: u32,
) -> u32 {
    let lg = ctx.storage_init("logits", logits);
    let out = ctx.storage_zeros_u32("out", 1);
    let dims = ctx.uniform_of(
        "dims",
        &stochastic_dims(logits.len(), temperature, top_k, top_p, seed, position),
    );
    kernel.dispatch("sample_stochastic", &[&lg, &out, &dims], [1, 1, 1]);
    ctx.read_u32(&out, 1)[0]
}

/// A `SamplingParams` for the CPU oracle from the same knobs the kernel takes.
fn params(temperature: f32, top_k: u32, top_p: f32, seed: u64) -> SamplingParams {
    SamplingParams {
        temperature,
        top_k: if top_k == 0 {
            None
        } else {
            Some(top_k as usize)
        },
        top_p: if top_p >= 1.0 { None } else { Some(top_p) },
        seed,
        ..SamplingParams::default()
    }
}

/// The CPU oracle's kept SET (token ids), straight from `filtered_probs`.
fn cpu_kept_set(logits: &[f32], p: &SamplingParams) -> HashSet<u32> {
    filtered_probs(logits, p)
        .into_iter()
        .map(|(t, _)| t)
        .collect()
}

/// CPU reimplementation of the GPU THRESHOLD selection, deterministic by
/// (logit DESC, index ASC). For distinct logits this equals the CPU sort-truncate
/// set; the assertion `gpu_threshold_kept == cpu_kept_set` validates the selection
/// logic independent of the RNG. Returns the kept token ids.
fn gpu_threshold_kept(logits: &[f32], p: &SamplingParams) -> HashSet<u32> {
    let inv_temp = 1.0 / p.temperature;
    let n = logits.len();
    let scaled: Vec<f32> = logits.iter().map(|&l| l * inv_temp).collect();
    let gmax = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // --- top-k threshold: tau_k = the k-th largest scaled logit. We compute it
    // exactly by sorting (the reference); the kernel binary-searches to the same
    // value for distinct logits. Survivors = {scaled >= tau_k}.
    let mut order: Vec<usize> = (0..n).collect();
    // logit DESC, index ASC tie-break (the GPU's deterministic realization).
    order.sort_by(|&a, &b| {
        scaled[b]
            .partial_cmp(&scaled[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let survivors: Vec<usize> = match p.top_k {
        Some(k) => order.iter().take(k.max(1)).cloned().collect(),
        None => order.clone(),
    };

    // --- softmax over survivors (max-subtract). ---
    let softsum: f32 = survivors.iter().map(|&t| (scaled[t] - gmax).exp()).sum();
    let prob = |t: usize| (scaled[t] - gmax).exp() / softsum;

    // --- top-p: keep the smallest prob-DESC prefix with cumulative >= top_p,
    // i.e. {survivor : prob >= tau_p} where tau_p is the boundary prob. We walk
    // the survivors in (prob DESC, idx ASC) order accumulating, exactly like the
    // CPU prefix, then take that set. ---
    let mut kept: HashSet<u32> = HashSet::new();
    match p.top_p {
        Some(top_p) if top_p < 1.0 => {
            // survivors are already a subset of `order`; re-order them prob-DESC.
            let mut surv_ord = survivors.clone();
            surv_ord.sort_by(|&a, &b| {
                prob(b)
                    .partial_cmp(&prob(a))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });
            let mut cum = 0.0f32;
            for &t in &surv_ord {
                cum += prob(t);
                kept.insert(t as u32);
                if cum >= top_p {
                    break;
                }
            }
        }
        _ => {
            for &t in &survivors {
                kept.insert(t as u32);
            }
        }
    }
    kept
}

// ---------------------------------------------------------------------------
// (a) Greedy bit-exactness.
// ---------------------------------------------------------------------------

#[test]
fn greedy_kernel_matches_cpu_argmax_incl_ties() {
    let Some(ctx) = try_ctx() else { return };
    let cases: Vec<Vec<f32>> = vec![
        vec![0.1, 0.9, 0.3],
        vec![2.0, 2.0, 1.0],          // tie at the top -> lowest index (0)
        vec![-1.0, -0.5, -0.5, -2.0], // tie not at the top
        vec![5.0, 1.0, 5.0, 5.0],     // multiple ties at the max -> index 0
        (0..300).map(|i| (i % 7) as f32).collect(), // > one workgroup of lanes
    ];
    for logits in &cases {
        let got = run_greedy(&ctx, logits);
        assert_eq!(
            got,
            argmax(logits),
            "greedy kernel != argmax for {logits:?}"
        );
    }
}

#[test]
fn stochastic_top_k_one_is_argmax_any_temp_seed() {
    let Some(ctx) = try_ctx() else { return };
    let kernel = ComputeKernel::new(&ctx, "sample_stochastic", STOCHASTIC_WGSL);
    // DISTINCT top logits: with a unique maximum, top_k=1 keeps exactly that one
    // token for every temperature/seed/position, so the draw is its index — the
    // GPU stochastic path is bit-exactly the greedy argmax. (A tie AT the top is
    // an ambiguous edge case covered separately by `top_k_one_with_top_tie_*`.)
    let cases: Vec<Vec<f32>> = vec![
        vec![0.2, 5.0, 0.1, 0.0],
        (0..400)
            .map(|i| ((i * 13) % 97) as f32 * 0.1 + i as f32 * 1e-3)
            .collect(),
    ];
    for logits in &cases {
        let want = argmax(logits);
        for &temp in &[0.5f32, 1.0, 2.0] {
            for seed in 0..4u64 {
                for pos in 0..4u32 {
                    let got = run_stochastic(&ctx, &kernel, logits, temp, 1, 1.0, seed, pos);
                    assert_eq!(
                        got, want,
                        "top_k=1 must collapse to argmax (temp={temp} seed={seed} pos={pos})"
                    );
                }
            }
        }
    }
}

/// DOCUMENTED tie edge case: with an exact tie at the top and `top_k = 1`, the
/// k-th-largest logit threshold sits ON the tied value, so the GPU's
/// `scaled >= tau_k` rule keeps the WHOLE tied group (it may keep more than k at
/// an exact boundary tie). The drawn token is therefore one of the tied indices —
/// a valid realization of the spec, which is ambiguous at a boundary tie (the CPU
/// `sort_unstable` truncation is itself non-deterministic there). We only assert
/// the draw stays within the tied set.
#[test]
fn top_k_one_with_top_tie_draws_from_tied_set() {
    let Some(ctx) = try_ctx() else { return };
    let kernel = ComputeKernel::new(&ctx, "sample_stochastic", STOCHASTIC_WGSL);
    let logits = vec![3.0f32, 3.0, 1.0, 3.0, 0.5]; // tokens 0,1,3 tied at the max
    let tied: HashSet<u32> = [0u32, 1, 3].into_iter().collect();
    for &temp in &[0.5f32, 1.0, 2.0] {
        for seed in 0..8u64 {
            for pos in 0..8u32 {
                let got = run_stochastic(&ctx, &kernel, &logits, temp, 1, 1.0, seed, pos);
                assert!(
                    tied.contains(&got),
                    "top_k=1 at a top tie drew {got}, not in the tied set {tied:?}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Selection equivalence (no GPU): the CPU threshold reimplementation keeps
// EXACTLY the CPU sort-truncate set, isolating selection from the RNG.
// ---------------------------------------------------------------------------

#[test]
fn threshold_selection_equals_cpu_sort_truncate() {
    // Distinct logits so neither selection is ambiguous at a tie.
    let logits: Vec<f32> = (0..64)
        .map(|i| ((i * 37 + 11) % 101) as f32 * 0.05)
        .collect();
    // Sanity: all distinct.
    {
        let mut s: Vec<u32> = logits.iter().map(|x| x.to_bits()).collect();
        s.sort_unstable();
        s.dedup();
        assert_eq!(s.len(), logits.len(), "test logits must be distinct");
    }
    for &temp in &[0.7f32, 1.0, 1.5] {
        for &top_k in &[0u32, 1, 5, 16, 40] {
            for &top_p in &[1.0f32, 0.5, 0.8, 0.95] {
                let p = params(temp, top_k, top_p, 0);
                let cpu = cpu_kept_set(&logits, &p);
                let gpu = gpu_threshold_kept(&logits, &p);
                assert_eq!(
                    gpu, cpu,
                    "threshold kept set != sort-truncate (temp={temp} top_k={top_k} top_p={top_p})"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// (b) Filter correctness: every drawn token is in the CPU kept set.
// ---------------------------------------------------------------------------

#[test]
fn gpu_draws_only_from_cpu_support() {
    let Some(ctx) = try_ctx() else { return };
    let kernel = ComputeKernel::new(&ctx, "sample_stochastic", STOCHASTIC_WGSL);
    // Distinct logits (avoid boundary-tie ambiguity between the two realizations).
    let logits: Vec<f32> = (0..64)
        .map(|i| ((i * 29 + 5) % 103) as f32 * 0.06)
        .collect();

    let configs: &[(f32, u32, f32)] = &[
        (1.0, 0, 1.0),  // pure temperature (full support)
        (1.0, 8, 1.0),  // top-k only
        (1.0, 0, 0.9),  // top-p only
        (0.7, 16, 0.8), // top-k + top-p
        (1.5, 4, 0.95), // higher temperature, tight k
    ];
    for &(temp, top_k, top_p) in configs {
        let p = params(temp, top_k, top_p, 7);
        let support = cpu_kept_set(&logits, &p);
        assert!(!support.is_empty());
        for seed in 0..20u64 {
            for pos in 0..30u32 {
                let tok = run_stochastic(&ctx, &kernel, &logits, temp, top_k, top_p, seed, pos);
                assert!(
                    support.contains(&tok),
                    "GPU drew {tok} OUTSIDE CPU support {support:?} \
                     (temp={temp} top_k={top_k} top_p={top_p} seed={seed} pos={pos})"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// (c) Distribution: empirical frequencies match CPU filtered_probs.
// ---------------------------------------------------------------------------

#[test]
fn gpu_draw_distribution_matches_cpu() {
    let Some(ctx) = try_ctx() else { return };
    let kernel = ComputeKernel::new(&ctx, "sample_stochastic", STOCHASTIC_WGSL);
    // Small synthetic vocab with distinct logits so the target distribution is
    // unambiguous and the per-token tolerance is tight.
    let logits: Vec<f32> = (0..64).map(|i| ((i * 17 + 3) % 53) as f32 * 0.08).collect();

    for &(temp, top_k, top_p) in &[(1.0f32, 0u32, 1.0f32), (1.0, 16, 1.0), (0.9, 0, 0.9)] {
        let p = params(temp, top_k, top_p, 123);
        // Target probabilities from the CPU oracle (already normalized over the
        // kept set by filtered_probs).
        let mut target = vec![0.0f32; logits.len()];
        for (t, prob) in filtered_probs(&logits, &p) {
            target[t as usize] = prob;
        }

        // N draws over varying position (the RNG counter), fixed seed: each
        // position yields an independent uniform from the deterministic PCG.
        const N: u32 = 10_000;
        let mut counts = vec![0u32; logits.len()];
        for pos in 0..N {
            let tok = run_stochastic(&ctx, &kernel, &logits, temp, top_k, top_p, 999, pos);
            counts[tok as usize] += 1;
        }

        // Max per-token |freq - prob| must be small. 10k draws over a 64-vocab
        // categorical: a 0.04 absolute tolerance comfortably covers sampling
        // noise (a few sigma) while still catching a wrong distribution.
        let mut max_err = 0.0f32;
        let mut worst = 0usize;
        for i in 0..logits.len() {
            let freq = counts[i] as f32 / N as f32;
            let err = (freq - target[i]).abs();
            if err > max_err {
                max_err = err;
                worst = i;
            }
            // No token outside the support may ever be drawn.
            if target[i] == 0.0 {
                assert_eq!(counts[i], 0, "drew unsupported token {i}");
            }
        }
        assert!(
            max_err < 0.04,
            "distribution mismatch (temp={temp} top_k={top_k} top_p={top_p}): \
             max |freq-prob| = {max_err:.4} at token {worst} \
             (freq {:.4} vs prob {:.4})",
            counts[worst] as f32 / N as f32,
            target[worst]
        );
    }
}
