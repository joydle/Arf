//! Token sampling: greedy, temperature, top-k, and top-p (nucleus).
//!
//! All strategies operate on a `[vocab]` slice of f32 logits and return a single
//! token id. A seeded RNG makes stochastic sampling reproducible.
//!
//! ## Grammar-constrained sampling
//!
//! The [`constraint`] sub-module provides the [`Constraint`] trait and
//! [`JsonConstraint`] for `response_format` JSON modes.  The constrained
//! entry-points are [`sample_at_constrained`] and [`sample_batch_constrained`]:
//! they apply the grammar mask before argmax/sampling, leaving the unconstrained
//! path ([`sample_at`], [`sample_batch`]) byte-identical to before.

pub mod constraint;

pub use constraint::{Constraint, JsonConstraint, TokenPieceMap};

use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::error::{ArfError, Result};

/// How to turn logits into the next token.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingParams {
    /// `0.0` means greedy (argmax). Otherwise logits are divided by this.
    pub temperature: f32,
    /// Keep only the `k` highest-probability tokens before sampling.
    pub top_k: Option<usize>,
    /// Keep the smallest set of tokens whose cumulative probability ≥ `top_p`.
    pub top_p: Option<f32>,
    /// Max tokens to generate for the request.
    pub max_tokens: usize,
    /// Generation stops when any of these tokens is produced.
    pub stop_tokens: Vec<u32>,
    /// RNG seed for reproducible stochastic sampling.
    pub seed: u64,
    /// Repetition penalty (CTRL/HF style): each already-generated token's logit is
    /// divided by this before sampling (positive logits) or multiplied (negative),
    /// discouraging loops. `1.0` = off (no penalty); typical anti-loop value ~1.1–1.3.
    /// Applies on BOTH greedy and stochastic paths (greedy loops are the common case).
    pub repetition_penalty: f32,
    /// Only the last `repeat_last_n` generated tokens are penalized (a sliding
    /// window). `0` = penalize the entire generated history. Bounds the cost and
    /// matches llama.cpp's `repeat_last_n`.
    pub repeat_last_n: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams {
            temperature: 0.0,
            top_k: None,
            top_p: None,
            max_tokens: 128,
            stop_tokens: Vec::new(),
            seed: 0,
            repetition_penalty: 1.0,
            repeat_last_n: 0,
        }
    }
}

impl SamplingParams {
    /// Greedy decoding with a token budget.
    pub fn greedy(max_tokens: usize) -> Self {
        SamplingParams {
            max_tokens,
            ..Default::default()
        }
    }

    /// Validate ranges. Cheap; call when a request is admitted.
    pub fn validate(&self) -> Result<()> {
        if self.temperature < 0.0 {
            return Err(ArfError::InvalidSampling(format!(
                "temperature must be >= 0, got {}",
                self.temperature
            )));
        }
        if let Some(p) = self.top_p {
            if !(0.0..=1.0).contains(&p) {
                return Err(ArfError::InvalidSampling(format!(
                    "top_p must be in [0,1], got {p}"
                )));
            }
        }
        if self.top_k == Some(0) {
            return Err(ArfError::InvalidSampling("top_k must be > 0".into()));
        }
        if self.repetition_penalty <= 0.0 {
            return Err(ArfError::InvalidSampling(format!(
                "repetition_penalty must be > 0 (1.0 = off), got {}",
                self.repetition_penalty
            )));
        }
        Ok(())
    }

    /// Whether a repetition penalty is active (not the 1.0 no-op).
    pub fn has_repetition_penalty(&self) -> bool {
        self.repetition_penalty != 1.0
    }

    /// Whether decoding is greedy (deterministic).
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }
}

/// A stateful sampler bound to one sequence's RNG stream.
#[derive(Debug)]
pub struct Sampler {
    rng: StdRng,
}

impl Sampler {
    pub fn new(seed: u64) -> Self {
        Sampler {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Sample the next token from a `[vocab]` logits slice.
    pub fn sample(&mut self, logits: &[f32], params: &SamplingParams) -> Result<u32> {
        if params.is_greedy() {
            return Ok(argmax(logits));
        }
        let probs = filtered_probs(logits, params);
        let dist = WeightedIndex::new(probs.iter().map(|(_, p)| *p))
            .map_err(|e| ArfError::InvalidSampling(format!("degenerate distribution: {e}")))?;
        Ok(probs[dist.sample(&mut self.rng)].0)
    }
}

/// Per-sequence sampling spec for one batched step.
///
/// `params` and `position` are used by both the unconstrained and constrained
/// paths.  `generated` is the slice of token ids emitted so far for this
/// sequence (NOT including the prompt); it is only accessed when a grammar
/// constraint is present (the unconstrained path ignores it).
#[derive(Debug, Clone, Copy)]
pub struct SeqSampling<'a> {
    pub params: &'a SamplingParams,
    /// Absolute position of the token being sampled (prompt_len + tokens_emitted).
    /// Used to key the per-position RNG for reproducible stochastic sampling.
    pub position: usize,
    /// Tokens emitted so far for this sequence (empty slice for unconstrained paths).
    pub generated: &'a [u32],
}

/// Stateless sampling: greedy argmax, or stochastic keyed by (seed, position).
/// Position-keyed RNG (vs the stateful [`Sampler`]) makes a recomputed
/// (preempted) sequence resample the SAME token at the same position, and
/// mirrors the GPU sampler's (seed, position)-keyed, stateless scheme (an
/// independent RNG — not a bit-identical stream) — the server path uses
/// this; the CPU engine keeps its per-sequence [`Sampler`].
pub fn sample_at(logits: &[f32], params: &SamplingParams, position: usize) -> Result<u32> {
    // No history available on this entry point → no repetition penalty.
    sample_at_with_history(logits, params, position, &[])
}

/// [`sample_at`] plus the repetition penalty over `generated` (the tokens emitted so
/// far for this sequence). The penalty is applied to a COPY of the logits BEFORE the
/// greedy/stochastic branch — greedy loops (the common case) need it too — so the
/// caller's logits are untouched and penalty-off (1.0) is byte-identical to before.
pub fn sample_at_with_history(
    logits: &[f32],
    params: &SamplingParams,
    position: usize,
    generated: &[u32],
) -> Result<u32> {
    // Fast path: no penalty → operate on the borrowed slice, zero allocation.
    if !params.has_repetition_penalty() || generated.is_empty() {
        if params.is_greedy() {
            return Ok(argmax(logits));
        }
        return sample_stochastic(logits, params, position);
    }
    let mut shaped = logits.to_vec();
    apply_repetition_penalty(&mut shaped, generated, params);
    if params.is_greedy() {
        return Ok(argmax(&shaped));
    }
    sample_stochastic(&shaped, params, position)
}

/// Position-keyed stochastic draw (top-k/top-p then weighted sample). Split out so
/// both the penalized and unpenalized paths share it.
fn sample_stochastic(logits: &[f32], params: &SamplingParams, position: usize) -> Result<u32> {
    // splitmix64-style mix so adjacent positions land in unrelated streams.
    let key = params
        .seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((position as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
    let mut rng = StdRng::seed_from_u64(key);
    let probs = filtered_probs(logits, params);
    let dist = WeightedIndex::new(probs.iter().map(|(_, p)| *p))
        .map_err(|e| ArfError::InvalidSampling(format!("degenerate distribution: {e}")))?;
    Ok(probs[dist.sample(&mut rng)].0)
}

/// CTRL/HF repetition penalty, in place: each token id in `generated` (last
/// `repeat_last_n`, or all when 0) has its logit divided by `repetition_penalty`
/// when positive, multiplied when negative — pushing already-seen tokens toward
/// less-likely regardless of sign. Out-of-range ids are skipped.
fn apply_repetition_penalty(logits: &mut [f32], generated: &[u32], params: &SamplingParams) {
    let p = params.repetition_penalty;
    let window = if params.repeat_last_n == 0 {
        generated
    } else {
        let start = generated.len().saturating_sub(params.repeat_last_n);
        &generated[start..]
    };
    for &tok in window {
        let i = tok as usize;
        if i >= logits.len() {
            continue;
        }
        let l = logits[i];
        logits[i] = if l > 0.0 { l / p } else { l * p };
    }
}

/// SPECULATIVE SAMPLING (2026-09-26): what a sampled request's verify rows are drawn with.
///
/// Every drafter here is DETERMINISTIC (the block draft's greedy selector chain, the MTP argmax
/// chain, the prompt-lookup majority vote), so the draft distribution q is a point mass on the
/// drafted token, and exact speculative sampling reduces to: draw each verify row's token with
/// the PLAIN path's own sampler — keyed by (seed, absolute position), with the history that row
/// would have — and keep the equality accept rule (`speculative::accepted_run`). Row r accepts
/// its draft x with probability P(y_r = x) = p_r(x); on a mismatch y_r ~ p_r given y_r != x,
/// which is exactly the residual norm(max(p_r - delta_x, 0)); after an all-accepted window the
/// last row's draw is the bonus. The emitted tokens are then the SAME tokens plain sampling
/// draws for the same seed (up to the verify record's logits differing from a one-row step's in
/// the last bits), not merely the same distribution.
#[derive(Debug, Clone)]
pub struct RowSampler {
    pub params: SamplingParams,
    /// The tokens this sequence has emitted so far, the pending one (the window's first row)
    /// included — the plain path's `generated`, which the repetition penalty reads.
    pub history: Vec<u32>,
    /// A SAMPLED draft's distributions (step 2): entry i is q over the candidates the draft drew
    /// window position i+1 from. Empty = the point-mass draft of step 1 (the equality rule).
    /// With it a row accepts its draft x with probability min(1, p(x) / q(x)) and a rejection
    /// draws from norm(max(p - q, 0)) — standard speculative sampling, exact for any q, and it
    /// accepts ~1 - TV(p, q) a position instead of p(x).
    pub draft_q: Vec<Vec<(u32, f32)>>,
}

/// What a keyed uniform is for — each purpose is its own stream, so the draft's draws, the
/// accept tests and the residual/bonus draws of one position never share a number.
pub mod purpose {
    pub const DRAFT: u64 = 1;
    pub const ACCEPT: u64 = 2;
    pub const RESIDUAL: u64 = 3;
}

/// A uniform in [0, 1) keyed by (seed, absolute position, purpose): splitmix64 over the mixed
/// key, top 24 bits. Stateless, so a recomputed or re-drafted position draws the same number.
pub fn keyed_uniform(seed: u64, position: usize, purpose: u64) -> f32 {
    let mut z = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((position as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add(purpose.wrapping_mul(0x94D0_49BB_1331_11EB));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 40) as f32 / (1u64 << 24) as f32
}

/// Inverse-CDF draw from an unnormalised distribution; `None` when it has no mass.
pub fn draw_from(dist: &[(u32, f32)], u: f32) -> Option<u32> {
    let total: f32 = dist.iter().map(|t| t.1).sum();
    // `is_nan` first: a NaN total is not "some mass"
    if total.is_nan() || total <= 0.0 {
        return None;
    }
    let target = u * total;
    let mut cum = 0.0f32;
    for &(t, w) in dist {
        cum += w;
        if cum > target {
            return Some(t);
        }
    }
    dist.iter().rev().find(|t| t.1 > 0.0).map(|t| t.0)
}

/// SAMPLED DRAFT, one position (speculative sampling step 2): the draft's q over its candidates
/// and the token drawn from it. `scored` = (score, token) per candidate, in any order; `best` = the
/// argmax (ties -> the lower id), which a degenerate softmax (NaN or zero mass) falls back to and
/// records as the point mass q = [(best, 1.0)] — the acceptor then uses the q the draft really
/// drew from (exact). `u` = the position's `keyed_uniform(seed, position, purpose::DRAFT)`.
/// Returns (the drawn token, q in TOKEN-ID order, so the draw is reproducible).
///
/// another engine's form (its `draft_select_dflash`): q = softmax(score / T) over the candidates. This is
/// the one arithmetic the CPU selector (`dflash_select`'s sampled branch) runs and the GPU chain
/// (`dflash_chain_sampled` in dflash2_msl.metal) mirrors, op for op — max-subtract, divide by
/// max(T, 1e-6), exp, a sequential sum, divide, then [`draw_from`].
pub fn sampled_draft_pick(
    scored: &[(f32, u32)],
    best: u32,
    temperature: f32,
    u: f32,
) -> (u32, Vec<(u32, f32)>) {
    let mut c = scored.to_vec();
    c.sort_by_key(|t| t.1);
    let m = c.iter().fold(f32::NEG_INFINITY, |a, t| a.max(t.0));
    let w: Vec<f32> = c
        .iter()
        .map(|t| ((t.0 - m) / temperature.max(1e-6)).exp())
        .collect();
    let z: f32 = w.iter().sum();
    let dist: Vec<(u32, f32)> = if z.is_finite() && z > 0.0 {
        c.iter().zip(&w).map(|(t, &x)| (t.1, x / z)).collect()
    } else {
        vec![(best, 1.0)]
    };
    let pick = draw_from(&dist, u).unwrap_or(best);
    (pick, dist)
}

impl RowSampler {
    /// Row `r`'s token: the one the plain path would draw after `window[..=r]`, i.e. at absolute
    /// position `prefix_len + r + 1`, with `window[1..=r]` appended to the history (those rows
    /// are the sequence's history exactly when the draw is used: every earlier draft accepted).
    pub fn sample_row(
        &self,
        logits: &[f32],
        prefix_len: usize,
        window: &[u32],
        r: usize,
    ) -> Result<u32> {
        let position = prefix_len + r + 1;
        if !self.params.has_repetition_penalty() || r == 0 {
            return sample_at_with_history(logits, &self.params, position, &self.history);
        }
        let mut hist = Vec::with_capacity(self.history.len() + r);
        hist.extend_from_slice(&self.history);
        hist.extend_from_slice(&window[1..=r.min(window.len() - 1)]);
        sample_at_with_history(logits, &self.params, position, &hist)
    }

    /// Row `r`'s TARGET distribution p (temperature, top-k, top-p and the repetition penalty
    /// with row r's history), normalised over its kept set.
    pub fn target_dist(&self, logits: &[f32], window: &[u32], r: usize) -> Vec<(u32, f32)> {
        debug_assert!(
            !self.params.is_greedy(),
            "target_dist is for a sampled request"
        );
        let mut hist;
        let logits: std::borrow::Cow<'_, [f32]> = if self.params.has_repetition_penalty() {
            hist = self.history.clone();
            if r > 0 {
                hist.extend_from_slice(&window[1..=r.min(window.len() - 1)]);
            }
            let mut shaped = logits.to_vec();
            apply_repetition_penalty(&mut shaped, &hist, &self.params);
            std::borrow::Cow::Owned(shaped)
        } else {
            std::borrow::Cow::Borrowed(logits)
        };
        let mut p = filtered_probs(&logits, &self.params);
        let sum: f32 = p.iter().map(|t| t.1).sum();
        for t in p.iter_mut() {
            t.1 /= sum;
        }
        p
    }

    /// Step 2: the rows of a window drafted from `draft_q`. Returns per-row predictions encoding
    /// the outcome for the equality rule downstream: an accepted row predicts its draft, the
    /// first rejected row predicts its residual draw (never the draft: a rejection means
    /// p(x) < q(x), so max(p - q, 0) is 0 at x), and an all-accepted window's last row its bonus.
    fn sample_rows_q(
        &self,
        logits: &[f32],
        vocab: usize,
        prefix_len: usize,
        window: &[u32],
    ) -> Result<Vec<u32>> {
        let rows = window.len();
        let seed = self.params.seed;
        let ps: Vec<Vec<(u32, f32)>> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..rows)
                .map(|r| {
                    let row = &logits[r * vocab..(r + 1) * vocab];
                    sc.spawn(move || self.target_dist(row, window, r))
                })
                .collect();
            hs.into_iter()
                .map(|h| h.join().expect("target_dist panicked"))
                .collect()
        });
        // INSTRUMENT (2026-09-26), ARF_ACCEPT_DUMP=<file>: one JSON line per sampled-draft window
        // with each tested row's target p, the draft's q and the drafted token — so q-shaping
        // (a sharper temperature, a mix with the argmax) can be sized OFFLINE against the exact
        // position-1 acceptance before anything is built (another engine accepts 3.67 vs our 3.45 tokens a
        // window on sampled requests, measured 2026-09-26). Off: nothing is read.
        if let Some(path) = std::env::var_os("ARF_ACCEPT_DUMP") {
            let fmt = |d: &[(u32, f32)]| {
                d.iter()
                    .map(|(t, w)| format!("[{t},{w}]"))
                    .collect::<Vec<_>>()
                    .join(",")
            };
            let rows_json: Vec<String> = (0..rows.saturating_sub(1).min(self.draft_q.len()))
                .map(|r| {
                    format!(
                        "{{\"x\":{},\"p\":[{}],\"q\":[{}]}}",
                        window[r + 1],
                        fmt(&ps[r]),
                        fmt(&self.draft_q[r])
                    )
                })
                .collect();
            let line = format!(
                "{{\"pos\":{prefix_len},\"rows\":[{}]}}\n",
                rows_json.join(",")
            );
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                use std::io::Write;
                let _ = f.write_all(line.as_bytes());
            }
        }
        let mut out = window[1..].to_vec();
        out.push(0);
        for r in 0..rows {
            let p = &ps[r];
            let pos = prefix_len + r + 1;
            if r + 1 < rows && r < self.draft_q.len() {
                let x = window[r + 1];
                let q = &self.draft_q[r];
                let q_of = |t: u32| q.iter().find(|c| c.0 == t).map_or(0.0, |c| c.1);
                let px = p.iter().find(|c| c.0 == x).map_or(0.0, |c| c.1);
                let qx = q_of(x);
                if qx > 0.0 && keyed_uniform(seed, pos, purpose::ACCEPT) * qx < px {
                    continue; // accepted: out[r] is already x
                }
                let resid: Vec<(u32, f32)> = p
                    .iter()
                    .map(|&(t, pt)| (t, (pt - q_of(t)).max(0.0)))
                    .filter(|&(t, w)| w > 0.0 && t != x)
                    .collect();
                let u = keyed_uniform(seed, pos, purpose::RESIDUAL);
                let without_x: Vec<(u32, f32)> = p.iter().copied().filter(|c| c.0 != x).collect();
                out[r] = draw_from(&resid, u)
                    .or_else(|| draw_from(&without_x, u))
                    .ok_or_else(|| ArfError::InvalidSampling("empty residual and target".into()))?;
                return Ok(out);
            }
            out[r] = draw_from(p, keyed_uniform(seed, pos, purpose::RESIDUAL))
                .ok_or_else(|| ArfError::InvalidSampling("empty target distribution".into()))?;
            return Ok(out);
        }
        Ok(out)
    }

    /// Every row of a verify window, rows side by side in `logits` (`[rows][vocab]`), each on
    /// its own thread (a row is one O(V) selection over the vocabulary, ~1 ms at 248K ids).
    pub fn sample_rows(
        &self,
        logits: &[f32],
        vocab: usize,
        prefix_len: usize,
        window: &[u32],
    ) -> Result<Vec<u32>> {
        let rows = window.len();
        // a greedy request (temperature 0, e.g. with a client-sent penalty) keeps the point-mass
        // rule even if a q was attached: target_dist divides by the temperature
        if !self.draft_q.is_empty() && !self.params.is_greedy() {
            assert!(
                logits.len() >= rows * vocab,
                "sample_rows: {} logits for {rows} x {vocab}",
                logits.len()
            );
            // every tested row needs its q; the record has run, so a mismatch cannot decline
            assert!(
                self.draft_q.len() + 1 >= rows,
                "sample_rows: {rows} rows but {} draft distributions",
                self.draft_q.len()
            );
            return self.sample_rows_q(logits, vocab, prefix_len, window);
        }
        assert!(
            logits.len() >= rows * vocab,
            "sample_rows: {} logits for {rows} x {vocab}",
            logits.len()
        );
        std::thread::scope(|sc| {
            let handles: Vec<_> = (0..rows)
                .map(|r| {
                    let row = &logits[r * vocab..(r + 1) * vocab];
                    sc.spawn(move || self.sample_row(row, prefix_len, window, r))
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("sample_rows: a row thread panicked"))
                .collect()
        })
    }
}

/// Sample one token per batched row. `rows[i]` is sequence i's `[vocab]`
/// last-token logits; `seqs[i]` its params + position. Used by the
/// `BatchedBackend::step` default impl (the serving loop's sampling stage).
pub fn sample_batch(rows: &[Vec<f32>], seqs: &[SeqSampling]) -> Result<Vec<u32>> {
    assert_eq!(rows.len(), seqs.len(), "one logits row per sequence");
    rows.iter()
        .zip(seqs)
        .map(|(row, s)| sample_at_with_history(row, s.params, s.position, s.generated))
        .collect()
}

/// Sample one token per batched row, with optional per-sequence grammar constraints.
///
/// When `constraints[i]` is `Some(c)`, the constraint's [`Constraint::apply_mask`]
/// is called on a **copy** of the logits row before sampling — so the unconstrained
/// rows (`None`) are absolutely unchanged.  The `piece_map` argument is required only
/// when at least one constraint is present (the constraint uses it to decode token ids
/// to string pieces for the grammar state machine).
///
/// This is the constrained variant of [`sample_batch`].  The actor calls this when any
/// sequence in the step has a `response_format` constraint; it falls back to
/// [`sample_batch`] when all constraints are `None`.
pub fn sample_batch_constrained(
    rows: &[Vec<f32>],
    seqs: &[SeqSampling],
    constraints: &[Option<&dyn Constraint>],
    piece_map: &TokenPieceMap,
) -> Result<Vec<u32>> {
    assert_eq!(rows.len(), seqs.len(), "one logits row per sequence");
    assert_eq!(
        rows.len(),
        constraints.len(),
        "one constraint slot per sequence"
    );
    rows.iter()
        .zip(seqs)
        .zip(constraints)
        .map(|((row, s), maybe_c)| match maybe_c {
            None => sample_at_with_history(row, s.params, s.position, s.generated),
            Some(c) => sample_at_constrained(row, s.params, s.position, s.generated, *c, piece_map),
        })
        .collect()
}

/// Constrained variant of [`sample_at`]: applies the grammar mask, then samples.
///
/// The logits are cloned internally so the caller's slice is unchanged.
/// `generated` must be the token ids emitted so far for this sequence (NOT
/// including the prompt).
///
/// The unconstrained path ([`sample_at`]) is entirely unaffected by this
/// function; it has no conditional logic for constraints.
pub fn sample_at_constrained(
    logits: &[f32],
    params: &SamplingParams,
    position: usize,
    generated: &[u32],
    constraint: &dyn Constraint,
    _piece_map: &TokenPieceMap,
) -> Result<u32> {
    // Clone, apply the repetition penalty (no-op at 1.0), then the grammar mask —
    // the mask is applied LAST so the grammar's hard constraints always win over the
    // softer penalty.
    let mut masked = logits.to_vec();
    if params.has_repetition_penalty() && !generated.is_empty() {
        apply_repetition_penalty(&mut masked, generated, params);
    }
    constraint.apply_mask(generated, &mut masked);

    // Belt-and-suspenders: if apply_mask somehow left every logit at -inf
    // (violating the never-all-masked invariant), fall back to the original
    // unmasked row rather than returning Err or silently picking token 0.
    // The constraint's own guard should prevent this; this is a second layer.
    if !masked.iter().any(|&v| v.is_finite()) {
        masked.copy_from_slice(logits);
    }

    // Now sample over the masked logits.
    if params.is_greedy() {
        return Ok(argmax(&masked));
    }
    let key = params
        .seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((position as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
    let mut rng = StdRng::seed_from_u64(key);
    let probs = filtered_probs(&masked, params);
    let dist = WeightedIndex::new(probs.iter().map(|(_, p)| *p)).map_err(|e| {
        ArfError::InvalidSampling(format!(
            "degenerate distribution after constraint mask: {e}"
        ))
    })?;
    Ok(probs[dist.sample(&mut rng)].0)
}

/// Largest candidate count `filtered_probs` selects by one streaming pass; above it, the
/// selection algorithm (an insert into the kept buffer is O(c)).
const STREAM_TOPK_MAX: usize = 1024;

/// Apply temperature, top-k and top-p, returning surviving `(token, prob)`.
///
/// This is the CPU sampling ORACLE the GPU `sample_stochastic` kernel replicates
/// (sort-then-truncate selection + max-subtract softmax); exposed so the GPU
/// parity tests can assert the GPU's threshold-selected kept set matches it.
///
/// O(V) SELECTION (2026-09-26). It used to sort all V indices on every sampled token — ~10 ms
/// at Qwen3.8's 248,320 ids, and speculative SAMPLING must sample every verify row (8 of them a
/// cycle). Now it selects only the candidates it can keep: the top-`k` with `top_k`, else a
/// best-first prefix grown until it holds `top_p` of the FULL-vocabulary mass (the full sort
/// remains only for `top_p` absent too). Same kept set and order as sorting everything —
/// under ONE total order, logit descending then id ascending: the old unstable sort left
/// equal logits in arbitrary order, so ties now resolve to the lower id, deterministically
/// (`filtered_probs_matches_full_sort` checks it against the full sort, ties included).
pub fn filtered_probs(logits: &[f32], params: &SamplingParams) -> Vec<(u32, f32)> {
    let inv_temp = 1.0 / params.temperature;
    let n = logits.len();
    let desc = |a: &u32, b: &u32| {
        logits[*b as usize]
            .partial_cmp(&logits[*a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(b))
    };
    // the `c` best token ids, best first
    let best = |c: usize| -> Vec<u32> {
        // STREAMING TOP-c (2026-09-26): for a small c, one pass keeping a sorted buffer of the
        // best c so far — one compare per id against the c-th, an insert only when it beats it.
        // `select_nth_unstable_by` over a 248,320-id index vector cost ~1.1-1.4 ms a row, and a
        // sampled verify draws 8 rows a cycle (measured draw 1.72 ms a window); this is ~0.07 ms a
        // row (unpublished probe topkbench, same kept set asserted). Same total order: ids arrive
        // ascending, so an equal later id never displaces an earlier one, and an insert goes after
        // its equals. NaN never enters (it compares false), where the old comparator placed it
        // arbitrarily.
        if c <= STREAM_TOPK_MAX && c < n {
            let mut v: Vec<(f32, u32)> = Vec::with_capacity(c + 1);
            let mut thr = f32::NEG_INFINITY;
            for (i, &x) in logits.iter().enumerate() {
                if x.is_nan() || (v.len() == c && x <= thr) {
                    continue;
                }
                let pos = v.partition_point(|&(y, _)| y >= x);
                v.insert(pos, (x, i as u32));
                if v.len() > c {
                    v.pop();
                }
                if v.len() == c {
                    thr = v[c - 1].0;
                }
            }
            return v.into_iter().map(|t| t.1).collect();
        }
        let mut idx: Vec<u32> = (0..n as u32).collect();
        if c < n {
            idx.select_nth_unstable_by(c - 1, desc);
            idx.truncate(c);
        }
        idx.sort_unstable_by(desc);
        idx
    };
    let probs_of = |idx: &[u32], max: f32, sum: f32| -> Vec<(u32, f32)> {
        idx.iter()
            .map(|&t| (t, (logits[t as usize] * inv_temp - max).exp() / sum))
            .collect()
    };
    let cut = |probs: &mut Vec<(u32, f32)>, top_p: f32| -> bool {
        let mut cum = 0.0f32;
        for (i, (_, p)) in probs.iter().enumerate() {
            cum += *p;
            if cum >= top_p {
                probs.truncate(i + 1);
                return true;
            }
        }
        false
    };
    if let Some(k) = params.top_k {
        // softmax over the top-k survivors, then the nucleus within them
        let idx = best(k.max(1).min(n));
        let max = logits[idx[0] as usize] * inv_temp;
        let sum: f32 = idx
            .iter()
            .map(|&t| (logits[t as usize] * inv_temp - max).exp())
            .sum();
        let mut probs = probs_of(&idx, max, sum);
        if let Some(top_p) = params.top_p {
            cut(&mut probs, top_p);
        }
        return probs;
    }
    // No top-k: the softmax spans the whole vocabulary, so normalise over all of it, then
    // take a best-first prefix large enough to reach top_p.
    let Some(top_p) = params.top_p else {
        let idx = best(n);
        let max = logits[idx[0] as usize] * inv_temp;
        let sum: f32 = idx
            .iter()
            .map(|&t| (logits[t as usize] * inv_temp - max).exp())
            .sum();
        return probs_of(&idx, max, sum);
    };
    let max = logits.iter().fold(f32::NEG_INFINITY, |m, &l| m.max(l)) * inv_temp;
    let sum: f32 = logits.iter().map(|&l| (l * inv_temp - max).exp()).sum();
    let mut c = 256usize.min(n);
    loop {
        let mut probs = probs_of(&best(c), max, sum);
        if cut(&mut probs, top_p) || c == n {
            probs.truncate(probs.len().max(1));
            return probs;
        }
        c = (c * 4).min(n);
    }
}

/// Index of the maximum logit (ties broken toward the lowest index).
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i as u32;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    /// The sampled draft's one-position arithmetic (shared by the CPU selector and mirrored by
    /// the GPU chain): q is softmax(score / T) in TOKEN-ID order whatever order the candidates
    /// arrive in, the draw follows q, and a degenerate softmax records the argmax point mass.
    #[test]
    fn sampled_draft_pick_orders_by_id_and_draws_from_q() {
        let scored = [(2.0f32, 900u32), (0.5, 12), (1.0, 407), (2.0, 55)];
        let (_, q) = sampled_draft_pick(&scored, 55, 0.7, 0.3);
        assert_eq!(
            q.iter().map(|t| t.0).collect::<Vec<_>>(),
            vec![12, 55, 407, 900]
        );
        let z: f32 = [0.5f32, 2.0, 1.0, 2.0]
            .iter()
            .map(|s| ((s - 2.0) / 0.7).exp())
            .sum();
        for (t, s) in [(12u32, 0.5f32), (55, 2.0), (407, 1.0), (900, 2.0)] {
            let want = ((s - 2.0) / 0.7).exp() / z;
            let got = q.iter().find(|c| c.0 == t).unwrap().1;
            assert!((got - want).abs() < 1e-6, "q({t}) = {got}, want {want}");
        }
        // the draw is draw_from over q with the given uniform: frequencies follow q
        let n = 40_000usize;
        let mut hits = std::collections::HashMap::<u32, usize>::new();
        for i in 0..n {
            let u = keyed_uniform(7, i, purpose::DRAFT);
            let (pick, q2) = sampled_draft_pick(&scored, 55, 0.7, u);
            assert_eq!(q2, q, "q does not depend on the uniform");
            assert_eq!(Some(pick), draw_from(&q, u));
            *hits.entry(pick).or_default() += 1;
        }
        for &(t, p) in &q {
            let f = hits.get(&t).copied().unwrap_or(0) as f32 / n as f32;
            assert!((f - p).abs() < 0.01, "token {t}: freq {f} vs q {p}");
        }
        // degenerate: a NaN score -> the point mass at `best`, and `best` is drawn
        let bad = [(f32::NAN, 3u32), (1.0, 9)];
        assert_eq!(sampled_draft_pick(&bad, 9, 0.7, 0.99), (9, vec![(9, 1.0)]));
        let none = [(f32::NEG_INFINITY, 3u32), (f32::NEG_INFINITY, 9)];
        assert_eq!(sampled_draft_pick(&none, 3, 0.7, 0.5), (3, vec![(3, 1.0)]));
    }

    /// EXACTNESS of step 2 (speculative sampling with a sampled draft): draft x ~ q, accept with
    /// min(1, p/q), else the residual — the emitted token must be distributed as p, whatever q
    /// is (here q over-weights some tokens, under-weights others, and puts mass OUTSIDE p's
    /// support). Chi-square over 200,000 independent seeds against p.
    #[test]
    fn sampled_draft_acceptance_emits_p() {
        let vocab = 64;
        let logits: Vec<f32> = (0..vocab).map(|i| ((i * 37 % 64) as f32) / 9.0).collect();
        let params = SamplingParams {
            temperature: 0.8,
            top_k: Some(12),
            top_p: Some(0.9),
            seed: 0,
            ..Default::default()
        };
        let probe = RowSampler {
            params: params.clone(),
            history: vec![],
            draft_q: vec![],
        };
        let p = probe.target_dist(&logits, &[0, 0], 0);
        // q: p's top candidates reweighted, plus two tokens outside p's kept set
        let outside: Vec<u32> = (0..vocab as u32)
            .filter(|t| !p.iter().any(|c| c.0 == *t))
            .take(2)
            .collect();
        let mut q: Vec<(u32, f32)> = p
            .iter()
            .enumerate()
            .map(|(i, &(t, w))| (t, if i % 2 == 0 { w * 1.8 } else { w * 0.3 }))
            .collect();
        q.extend(outside.iter().map(|&t| (t, 0.05)));
        let z: f32 = q.iter().map(|c| c.1).sum();
        for c in q.iter_mut() {
            c.1 /= z;
        }
        let n = 200_000u64;
        let mut hist = std::collections::HashMap::<u32, u64>::new();
        for s in 0..n {
            let x = draw_from(&q, keyed_uniform(s, 10_000 + s as usize, purpose::DRAFT)).unwrap();
            let rs = RowSampler {
                params: SamplingParams {
                    seed: s,
                    ..params.clone()
                },
                history: vec![],
                draft_q: vec![q.clone()],
            };
            // two rows: row 0 tests the draft x, row 1 would be the bonus
            let mut two = logits.clone();
            two.extend_from_slice(&logits);
            let out = rs.sample_rows(&two, vocab, 5, &[3, x]).unwrap();
            let emitted = if out[0] == x { x } else { out[0] };
            *hist.entry(emitted).or_default() += 1;
        }
        let mut chi2 = 0.0f64;
        for &(t, pt) in &p {
            let e = pt as f64 * n as f64;
            let o = *hist.get(&t).unwrap_or(&0) as f64;
            chi2 += (o - e).powi(2) / e;
        }
        let leaked: u64 = hist
            .iter()
            .filter(|(t, _)| !p.iter().any(|c| c.0 == **t))
            .map(|(_, c)| *c)
            .sum();
        assert_eq!(leaked, 0, "tokens outside p's support were emitted");
        // p has <= 12 kept tokens: chi-square with <= 11 dof; 99.9th percentile ~31.3
        assert!(
            chi2 < 31.3,
            "emitted distribution differs from p: chi2 {chi2:.1} over {} tokens",
            p.len()
        );
    }

    /// B1 (2026-09-26): temperature 0 with a penalty is not "all greedy", so it may reach the
    /// row sampler with a q attached — it must keep the point-mass (argmax) rule, not divide by 0.
    #[test]
    fn greedy_with_penalty_ignores_draft_q() {
        let vocab = 50;
        let logits: Vec<f32> = (0..2 * vocab)
            .map(|i| ((i * 13 % 50) as f32) / 7.0)
            .collect();
        let params = SamplingParams {
            temperature: 0.0,
            repetition_penalty: 1.05,
            seed: 3,
            ..Default::default()
        };
        let q = vec![vec![(1u32, 0.5f32), (2, 0.5)]];
        let with_q = RowSampler {
            params: params.clone(),
            history: vec![4, 5],
            draft_q: q,
        };
        let without = RowSampler {
            params,
            history: vec![4, 5],
            draft_q: vec![],
        };
        assert_eq!(
            with_q.sample_rows(&logits, vocab, 9, &[7, 1]).unwrap(),
            without.sample_rows(&logits, vocab, 9, &[7, 1]).unwrap()
        );
    }

    /// A verify row draws exactly the token the plain path draws at the same position with the
    /// same history — the property that makes speculative sampling exact (and seed-identical).
    #[test]
    fn row_sampler_matches_the_plain_path() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let vocab = 3000;
        let params = SamplingParams {
            temperature: 0.8,
            top_k: Some(20),
            top_p: Some(0.95),
            repetition_penalty: 1.1,
            seed: 42,
            ..Default::default()
        };
        let history: Vec<u32> = (0..40).map(|_| rng.random_range(0..vocab as u32)).collect();
        let window: Vec<u32> = (0..8).map(|_| rng.random_range(0..vocab as u32)).collect();
        let logits: Vec<f32> = (0..8 * vocab)
            .map(|_| rng.random_range(-6.0f32..6.0))
            .collect();
        let rs = RowSampler {
            params: params.clone(),
            history: history.clone(),
            draft_q: vec![],
        };
        let got = rs.sample_rows(&logits, vocab, 100, &window).unwrap();
        for (r, &g) in got.iter().enumerate() {
            let mut hist = history.clone();
            hist.extend_from_slice(&window[1..=r]);
            let want = sample_at_with_history(
                &logits[r * vocab..(r + 1) * vocab],
                &params,
                100 + r + 1,
                &hist,
            )
            .unwrap();
            assert_eq!(g, want, "row {r}");
        }
    }

    /// The O(V) selection against the full sort it replaced (with the canonical id tie-break),
    /// on random logits with deliberate ties, for every top_k / top_p combination.
    #[test]
    fn filtered_probs_never_keeps_nan() {
        // the streaming top-k skips NaN; the old comparator placed it arbitrarily. (Without
        // top_k the full-vocabulary normaliser still sums a NaN in — unchanged, not covered here.)
        let mut logits: Vec<f32> = (0..2000).map(|i| (i % 97) as f32 / 10.0).collect();
        logits[5] = f32::NAN;
        logits[1500] = f32::NAN;
        for top_k in [Some(20usize), Some(1)] {
            let params = SamplingParams {
                temperature: 0.7,
                top_k,
                top_p: Some(0.95),
                ..Default::default()
            };
            let kept = filtered_probs(&logits, &params);
            assert!(!kept.is_empty());
            assert!(
                kept.iter()
                    .all(|&(t, p)| t != 5 && t != 1500 && p.is_finite()),
                "{top_k:?}: {kept:?}"
            );
        }
    }

    #[test]
    fn filtered_probs_matches_full_sort() {
        use rand::{Rng, SeedableRng};
        fn reference(logits: &[f32], params: &SamplingParams) -> Vec<(u32, f32)> {
            let inv_temp = 1.0 / params.temperature;
            let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
            idx.sort_by(|&a, &b| {
                logits[b as usize]
                    .partial_cmp(&logits[a as usize])
                    .unwrap()
                    .then(a.cmp(&b))
            });
            if let Some(k) = params.top_k {
                idx.truncate(k.max(1));
            }
            let max = logits[idx[0] as usize] * inv_temp;
            let mut probs: Vec<(u32, f32)> = idx
                .iter()
                .map(|&t| (t, (logits[t as usize] * inv_temp - max).exp()))
                .collect();
            let sum: f32 = probs.iter().map(|(_, p)| p).sum();
            for (_, p) in probs.iter_mut() {
                *p /= sum;
            }
            if let Some(top_p) = params.top_p {
                let mut cum = 0.0f32;
                let mut keep = 0;
                for (_, p) in &probs {
                    cum += *p;
                    keep += 1;
                    if cum >= top_p {
                        break;
                    }
                }
                probs.truncate(keep.max(1));
            }
            probs
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        for trial in 0..60 {
            let n = [50usize, 1000, 5000][trial % 3];
            // coarse values make exact ties common
            let logits: Vec<f32> = (0..n)
                .map(|_| (rng.random_range(-800i32..800) as f32) / 100.0)
                .collect();
            for (top_k, top_p) in [
                (None, Some(0.95f32)),
                (Some(20usize), Some(0.95)),
                (Some(20), None),
                (None, None),
                (Some(1), Some(0.5)),
                (None, Some(0.3)),
            ] {
                let params = SamplingParams {
                    temperature: [0.7f32, 1.0, 1.3][trial % 3],
                    top_k,
                    top_p,
                    ..Default::default()
                };
                let a = filtered_probs(&logits, &params);
                let b = reference(&logits, &params);
                // Without top_k the full-vocabulary normaliser is summed in index order here and
                // in sorted order by the reference, so the top_p cut may move by ONE token when
                // the cumulative mass sits within float rounding of top_p. Nothing else may move.
                let common = a.len().min(b.len());
                assert!(
                    a.len().abs_diff(b.len()) <= 1,
                    "kept set sizes {} vs {} (n {n}, top_k {top_k:?}, top_p {top_p:?})",
                    a.len(),
                    b.len()
                );
                if a.len() != b.len() {
                    let cum: f32 = b.iter().take(common).map(|x| x.1).sum();
                    assert!(
                        top_k.is_none() && (cum - top_p.unwrap()).abs() < 1e-4,
                        "the cut moved away from the boundary (cum {cum}, top_p {top_p:?})"
                    );
                }
                assert_eq!(
                    a[..common].iter().map(|x| x.0).collect::<Vec<_>>(),
                    b[..common].iter().map(|x| x.0).collect::<Vec<_>>(),
                    "kept order differs (n {n}, top_k {top_k:?}, top_p {top_p:?})"
                );
                for (x, y) in a.iter().zip(&b) {
                    assert!(
                        (x.1 - y.1).abs() <= 1e-5 * y.1.max(1e-6),
                        "prob {} vs {}",
                        x.1,
                        y.1
                    );
                }
            }
        }
    }

    use super::*;

    #[test]
    fn argmax_picks_highest() {
        assert_eq!(argmax(&[0.1, 0.9, 0.3]), 1);
        assert_eq!(argmax(&[2.0, 2.0]), 0, "ties go to lowest index");
    }

    #[test]
    fn greedy_returns_argmax() {
        let mut s = Sampler::new(0);
        assert_eq!(
            s.sample(&[1.0, 3.0, 2.0, 0.5], &SamplingParams::greedy(8))
                .unwrap(),
            1
        );
    }

    #[test]
    fn top_k_one_is_deterministic_argmax() {
        let mut s = Sampler::new(42);
        let p = SamplingParams {
            temperature: 1.0,
            top_k: Some(1),
            ..Default::default()
        };
        for _ in 0..5 {
            assert_eq!(s.sample(&[0.2, 5.0, 0.1, 0.0], &p).unwrap(), 1);
        }
    }

    #[test]
    fn same_seed_is_reproducible() {
        let p = SamplingParams {
            temperature: 1.0,
            ..Default::default()
        };
        let logits = [0.5, 1.0, 0.8, 0.3, 0.9];
        let mut a = Sampler::new(7);
        let mut b = Sampler::new(7);
        for _ in 0..10 {
            assert_eq!(
                a.sample(&logits, &p).unwrap(),
                b.sample(&logits, &p).unwrap()
            );
        }
    }

    #[test]
    fn validate_rejects_bad_params() {
        assert!(SamplingParams {
            temperature: -1.0,
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(SamplingParams {
            top_p: Some(1.5),
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(SamplingParams {
            top_k: Some(0),
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(SamplingParams::greedy(4).validate().is_ok());
    }

    #[test]
    fn sample_at_greedy_is_argmax() {
        let p = SamplingParams::greedy(8);
        assert_eq!(sample_at(&[1.0, 3.0, 2.0], &p, 17).unwrap(), 1);
    }

    #[test]
    fn repetition_penalty_off_is_unchanged() {
        // penalty 1.0 (default) → identical to the no-history greedy argmax, even
        // when a "generated" history is present.
        let p = SamplingParams::greedy(8); // repetition_penalty defaults to 1.0
        assert!(!p.has_repetition_penalty());
        let logits = [1.0, 3.0, 2.0];
        assert_eq!(
            sample_at_with_history(&logits, &p, 0, &[1, 1, 1]).unwrap(),
            1,
            "1.0 penalty must not change the pick"
        );
    }

    #[test]
    fn repetition_penalty_breaks_a_greedy_loop() {
        // The demo failure mode: greedy keeps picking token 1 (highest logit) forever.
        // With a penalty and token 1 already in the history, its logit is divided down
        // so a DIFFERENT token wins — the loop breaks.
        let logits = [1.0, 3.0, 2.0]; // argmax = token 1
        let p = SamplingParams {
            repetition_penalty: 2.0, // 3.0 / 2.0 = 1.5 < 2.0 (token 2)
            ..SamplingParams::greedy(8)
        };
        assert!(p.has_repetition_penalty());
        // No history yet → still token 1.
        assert_eq!(sample_at_with_history(&logits, &p, 0, &[]).unwrap(), 1);
        // Token 1 emitted → penalized below token 2 → loop breaks to token 2.
        assert_eq!(sample_at_with_history(&logits, &p, 1, &[1]).unwrap(), 2);
    }

    #[test]
    fn repeat_last_n_windows_the_history() {
        // Only the last N tokens are penalized; an older repeat outside the window
        // is NOT penalized.
        let logits = [1.0, 3.0, 2.0];
        let p = SamplingParams {
            repetition_penalty: 2.0,
            repeat_last_n: 1, // only the most recent token
            ..SamplingParams::greedy(8)
        };
        // History = [1, 2]; window = [2] (only token 2 penalized). Token 1 (logit 3.0)
        // is OUTSIDE the window → still the argmax.
        assert_eq!(sample_at_with_history(&logits, &p, 2, &[1, 2]).unwrap(), 1);
    }

    #[test]
    fn repetition_penalty_validates() {
        let bad = SamplingParams {
            repetition_penalty: 0.0,
            ..Default::default()
        };
        assert!(bad.validate().is_err(), "penalty must be > 0");
    }

    #[test]
    fn sample_at_is_deterministic_per_seed_and_position() {
        let p = SamplingParams {
            temperature: 1.0,
            seed: 7,
            ..Default::default()
        };
        let logits = [0.5, 1.0, 0.8, 0.3, 0.9];
        assert_eq!(
            sample_at(&logits, &p, 3).unwrap(),
            sample_at(&logits, &p, 3).unwrap(),
            "same (seed, position) -> same token (preemption-recompute safe)"
        );
        // Different positions draw from different RNG streams; over many
        // positions at temperature 1.0 the draws cannot all collide.
        let varied: std::collections::HashSet<u32> = (0..64)
            .map(|pos| sample_at(&logits, &p, pos).unwrap())
            .collect();
        assert!(varied.len() > 1, "positions must decorrelate draws");
    }

    #[test]
    fn sample_batch_maps_rows_to_seqs() {
        let g = SamplingParams::greedy(8);
        let rows = vec![vec![0.0, 9.0], vec![9.0, 0.0]];
        let seqs = vec![
            SeqSampling {
                params: &g,
                position: 5,
                generated: &[],
            },
            SeqSampling {
                params: &g,
                position: 9,
                generated: &[],
            },
        ];
        assert_eq!(sample_batch(&rows, &seqs).unwrap(), vec![1, 0]);
    }
}
