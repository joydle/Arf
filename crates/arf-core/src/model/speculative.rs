//! Prompt-lookahead speculative decoding (CPU, the correctness oracle).
//!
//! Greedy speculative decoding is **exact**: it emits exactly the tokens plain
//! greedy decoding would, just faster when drafts hit. Each round:
//!   1. an n-gram drafter proposes up to `k` continuation tokens from history,
//!   2. ONE forward pass over `[committed, d1..dk]` yields the model's greedy
//!      prediction at every position (the weight stream — the cost — is paid
//!      once for up to k+1 tokens),
//!   3. accept the longest draft prefix the model agrees with, plus the model's
//!      own next token; reject the rest. The KV of rejected positions is simply
//!      overwritten next round (we advance the cache only by the accepted count).
//!
//! This module exists to MEASURE the acceptance rate before committing to a GPU
//! port: the batched/multi-position forward is compute-bound (scales ~linearly
//! with rows), so speculation only wins if `acceptance × k` beats that ~k× cost.
//! [`SpecStats`] reports the numbers that decide it.

use std::collections::HashMap;

use crate::cache::{slots_for, write_runs, PagedKvCache};
use crate::model::batch::{ForwardBatch, SeqAttn};
use crate::model::Llama;
use crate::sampling::argmax;

/// Outcome of a speculative run: the tokens (identical to plain greedy) plus the
/// stats that decide whether the GPU port is worth it.
#[derive(Debug, Clone)]
pub struct SpecResult {
    pub tokens: Vec<u32>,
    pub stats: SpecStats,
}

/// Acceptance accounting. `accepted_tokens / forward_passes` is the mean tokens
/// emitted per (expensive) forward — the speedup ceiling over plain greedy.
#[derive(Debug, Clone, Default)]
pub struct SpecStats {
    /// Forward passes run (each streams the full weight set once).
    pub forward_passes: usize,
    /// Total tokens emitted (== tokens.len()).
    pub accepted_tokens: usize,
    /// Draft tokens proposed across all rounds.
    pub drafted: usize,
    /// Draft tokens the model accepted (drafted that matched its own argmax).
    pub draft_hits: usize,
}

impl SpecStats {
    /// Mean tokens emitted per forward pass — the spec-dec speedup ceiling.
    pub fn tokens_per_forward(&self) -> f64 {
        if self.forward_passes == 0 {
            0.0
        } else {
            self.accepted_tokens as f64 / self.forward_passes as f64
        }
    }

    /// Fraction of drafted tokens the model accepted.
    pub fn draft_acceptance(&self) -> f64 {
        if self.drafted == 0 {
            0.0
        } else {
            self.draft_hits as f64 / self.drafted as f64
        }
    }
}

/// An n-gram drafter: remembers, for each (n-1)-token context seen, the token
/// that followed it most recently. `propose` walks that chain to draft tokens.
/// Order `n = 2` (bigram) is the cheapest useful lookahead; higher orders are
/// more precise but hit less often.
pub struct NgramDrafter {
    order: usize, // n: context is the last (order-1) tokens
    next: HashMap<Vec<u32>, u32>,
}

impl NgramDrafter {
    pub fn new(order: usize) -> Self {
        assert!(order >= 2, "n-gram order must be >= 2");
        NgramDrafter {
            order,
            next: HashMap::new(),
        }
    }

    /// Record that `tok` followed the (order-1) tokens ending at `tail` (the
    /// slice ending just before `tok`). Call for every committed token so the
    /// table tracks the realized sequence.
    pub fn observe(&mut self, history: &[u32], tok: u32) {
        let ctx = self.order - 1;
        if history.len() >= ctx {
            let key = history[history.len() - ctx..].to_vec();
            self.next.insert(key, tok);
        }
    }

    /// Draft up to `k` tokens by following the chain from `history`'s tail. Each
    /// drafted token extends the (virtual) context for the next lookup, so a
    /// run of tokens seen before is proposed in full.
    pub fn propose(&self, history: &[u32], k: usize) -> Vec<u32> {
        let ctx = self.order - 1;
        let mut draft = Vec::with_capacity(k);
        // Only the last (order-1) tokens ever key a lookup (drafts append to the tail),
        // so seed from the window, not the whole history — the full copy was an O(ctx)
        // alloc+memcpy per decode step (128KB per token at 32k context).
        let mut tail: Vec<u32> = history[history.len().saturating_sub(ctx)..].to_vec();
        for _ in 0..k {
            if tail.len() < ctx {
                break;
            }
            let key = &tail[tail.len() - ctx..];
            match self.next.get(key) {
                Some(&t) => {
                    draft.push(t);
                    tail.push(t);
                }
                None => break,
            }
        }
        draft
    }
}

/// Candidate suffix lengths tried longest-first when matching the tail of
/// `history` against prior occurrences. Kept small + fixed so `observe` stays
/// O(1) amortized (one hashmap insert per candidate per call) and `propose`
/// stays O(k) (a handful of lookups, not a scan of the whole history).
const SUFFIX_LENS: [usize; 3] = [8, 6, 4];

/// Shortest suffix match this drafter will ever draft from. Below this, a
/// "match" is too likely to be coincidental repetition (e.g. common short
/// token sequences) to be worth a verify round — see `MIN_OCCURRENCES_SHORT`.
const MIN_MATCH_LEN_DEFAULT: usize = SUFFIX_LENS[SUFFIX_LENS.len() - 1]; // 4

/// L238 — `ARF_SPEC_MIN_MATCH` overrides [`MIN_MATCH_LEN_DEFAULT`]. The default of 4 was
/// chosen for a cost model where a wasted verify is expensive. Ours is the opposite: a wrong
/// draft costs ONE verify window (~1.3 decode steps at k=2) while a MISSED draft costs a full
/// decode step, and L237 measured 141 of 159 proposals returning empty — an 89% miss rate — at
/// 72% acceptance on the ones that did fire. Lowering the bar trades a few wasted windows for
/// many more attempts, which is the right trade at those numbers. Measure, do not assume.
fn min_match_len() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("ARF_SPEC_MIN_MATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n: &usize| n >= 1)
            .unwrap_or(MIN_MATCH_LEN_DEFAULT)
    })
}

/// A match at the shortest tolerated length additionally needs this many
/// independent historical occurrences agreeing on the continuation (a simple
/// majority vote) before it's trusted — this is the confidence gate that
/// keeps the drafter from firing on weak/coincidental matches. Longer matches
/// (> MIN_MATCH_LEN) are trusted off a single occurrence: the suffix itself is
/// specific enough that a repeat is unlikely to be coincidence.
const MIN_OCCURRENCES_SHORT_DEFAULT: usize = 2;

/// L238 — `ARF_SPEC_MIN_OCC` overrides [`MIN_OCCURRENCES_SHORT_DEFAULT`]. Set 1 to trust a
/// shortest-length match off a single occurrence.
fn min_occurrences_short() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("ARF_SPEC_MIN_OCC")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n: &usize| n >= 1)
            .unwrap_or(MIN_OCCURRENCES_SHORT_DEFAULT)
    })
}

/// A SuffixDecoding-class drafter: rather than a fixed n-gram order, it
/// greedily matches the LONGEST suffix of the observed history against prior
/// occurrences of that same suffix anywhere in history (prompt + generated),
/// then extends forward from the best-agreeing occurrence(s) to build a draft.
///
/// This approximates a suffix automaton (which would give exact O(1)-extend /
/// O(k)-query behavior) with much less code: a hashmap from a *fixed-length*
/// tail suffix to the list of prior positions where that exact suffix ended,
/// tried across a small descending set of lengths (`SUFFIX_LENS`) so longer
/// (more specific, higher-confidence) matches are preferred over shorter ones.
///
/// CONFIDENCE GATE (the fix for the n-gram's measured 0.76x loss on prose):
/// `propose` returns an EMPTY draft — not a weak/low-quality one — whenever
/// the best match is shorter than `MIN_MATCH_LEN`, or sits at exactly
/// `MIN_MATCH_LEN` without at least `MIN_OCCURRENCES_SHORT` independent
/// occurrences agreeing on the next token. An empty draft is free: the m4
/// bridge (batch.rs) already treats "no draft" as "skip verify, do the plain
/// decode" — so this drafter can never make the net path slower than the
/// no-spec baseline, only ever tokens-for-free on strong repeats.
pub struct SuffixDrafter {
    /// For each candidate suffix length, a map from that suffix (as a key) to
    /// the list of `history` positions immediately AFTER a prior occurrence of
    /// it (i.e. `history[pos]` is the token that followed that occurrence).
    /// One map per length so a query can try longest-first without having to
    /// re-derive shorter suffixes from a single combined structure.
    by_len: Vec<HashMap<Vec<u32>, Vec<usize>>>,
    /// Cap on stored occurrences per suffix key (oldest evicted first) so a
    /// highly repetitive stream (e.g. a loop) can't grow a key's occurrence
    /// list unboundedly — keeps `propose`'s per-key work bounded.
    max_occurrences: usize,
}

impl Default for SuffixDrafter {
    fn default() -> Self {
        Self::new()
    }
}

impl SuffixDrafter {
    pub fn new() -> Self {
        SuffixDrafter {
            by_len: SUFFIX_LENS.iter().map(|_| HashMap::new()).collect(),
            max_occurrences: 32,
        }
    }

    /// Record that `tok` followed `history` (the history NOT yet including
    /// `tok`) — call for every committed token so the index tracks the
    /// realized sequence. O(1) amortized: one hashmap entry touched per
    /// candidate length in `SUFFIX_LENS`, never a scan of `history`.
    pub fn observe(&mut self, history: &[u32], tok: u32) {
        let _ = tok; // the position is what's stored; `tok` is read back via `history` later
        for (map, &len) in self.by_len.iter_mut().zip(SUFFIX_LENS.iter()) {
            if history.len() >= len {
                let key = history[history.len() - len..].to_vec();
                let occ = map.entry(key).or_default();
                occ.push(history.len()); // position of `tok` once pushed by the caller
                if occ.len() > self.max_occurrences {
                    occ.remove(0);
                }
            }
        }
    }

    /// Find the longest suffix of `history` (checked at `SUFFIX_LENS`,
    /// longest-first) with at least one PRIOR occurrence elsewhere in
    /// `history`. Returns the matched length and the occurrence positions
    /// (each position `p` means `history[p]` is a token that has followed
    /// this suffix before). Prior occurrences that happen to be the tail
    /// itself (nothing to extend from) are excluded. The occurrence list is
    /// capped (`max_occurrences`) at insert time, so this allocation is O(1)
    /// bounded, not O(history).
    fn best_match(&self, history: &[u32]) -> Option<(usize, Vec<usize>)> {
        for (map, &len) in self.by_len.iter().zip(SUFFIX_LENS.iter()) {
            if history.len() < len {
                continue;
            }
            let key = &history[history.len() - len..];
            if let Some(occ) = map.get(key) {
                // Exclude the current tail's own position (occ position ==
                // history.len() would mean "the token that follows THIS
                // exact tail", which doesn't exist yet — nothing to extend).
                let usable: Vec<usize> =
                    occ.iter().copied().filter(|&p| p < history.len()).collect();
                if !usable.is_empty() {
                    return Some((len, usable));
                }
            }
        }
        None
    }

    /// Draft up to `k` tokens by greedy longest-suffix match + majority vote
    /// across agreeing historical occurrences. Returns an EMPTY draft (never
    /// a low-confidence one) when the match is below the confidence gate
    /// (`MIN_MATCH_LEN` / `MIN_OCCURRENCES_SHORT`) — see the struct docs.
    /// O(k) plus a handful of hashmap lookups: no scan of `history`.
    pub fn propose(&self, history: &[u32], k: usize) -> Vec<u32> {
        if k == 0 || history.is_empty() {
            return Vec::new();
        }
        let (match_len, occurrences) = match self.best_match(history) {
            Some(m) => m,
            None => return Vec::new(),
        };
        // Confidence gate: below MIN_MATCH_LEN never fires; exactly the
        // shortest tolerated length needs corroborating occurrences.
        if match_len < min_match_len() {
            return Vec::new();
        }
        if match_len == min_match_len() && occurrences.len() < min_occurrences_short() {
            return Vec::new();
        }

        // Extend forward token-by-token: at each offset, take the majority
        // vote of what each occurrence's continuation says next; stop at the
        // first offset where no candidate has a strict majority, or once
        // every occurrence has run out of history to extend from.
        let mut draft = Vec::with_capacity(k);
        'outer: for i in 0..k {
            let mut votes: HashMap<u32, usize> = HashMap::new();
            for &pos in &occurrences {
                let idx = pos + i;
                if idx >= history.len() {
                    continue;
                }
                *votes.entry(history[idx]).or_default() += 1;
            }
            if votes.is_empty() {
                break 'outer;
            }
            let (&best_tok, &best_votes) = votes.iter().max_by_key(|&(_, &c)| c).unwrap();
            let total: usize = votes.values().sum();
            // Majority: strictly more than half the (still-live) occurrences
            // agree. With a single occurrence this is trivially satisfied
            // (already gated above by MIN_OCCURRENCES_SHORT when short).
            if best_votes * 2 <= total {
                break 'outer;
            }
            draft.push(best_tok);
        }
        draft
    }
}

/// Greedily generate up to `max_tokens` tokens from `prompt` using prompt-
/// lookahead speculative decoding. Output is identical to plain greedy decoding;
/// `stats` reports the acceptance rate. `k` is the max draft length per round;
/// `order` is the n-gram order (2 = bigram, cheapest; higher = more precise drafts
/// that hit less often).
///
/// `block_size`/`num_blocks` size the KV cache; the sequence uses a contiguous
/// block table, and the cache advances only by accepted tokens each round.
pub fn generate_speculative(
    model: &Llama,
    prompt: &[u32],
    max_tokens: usize,
    k: usize,
    order: usize,
    block_size: usize,
) -> SpecResult {
    assert!(!prompt.is_empty(), "empty prompt");
    let vocab = model.config().vocab_size;
    let total_cap = prompt.len() + max_tokens + k + 1;
    let num_blocks = total_cap.div_ceil(block_size);
    let mut cache = PagedKvCache::new(
        model.num_layers(),
        num_blocks,
        block_size,
        model.config().num_kv_heads,
        model.config().head_dim,
    );
    let block_table: Vec<u32> = (0..num_blocks as u32).collect();

    let mut drafter = NgramDrafter::new(order);
    let mut history: Vec<u32> = prompt.to_vec();
    let mut out: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut stats = SpecStats::default();

    // Run one forward over `tokens` at logical `past_len`; return per-position
    // greedy argmax (length tokens.len()). Advances the cache for all positions.
    let run = |cache: &mut PagedKvCache, tokens: &[u32], past_len: usize| -> Vec<u32> {
        let q_len = tokens.len();
        let ctx = past_len + q_len;
        let batch = ForwardBatch {
            positions: (past_len..ctx).map(|p| p as u32).collect(),
            seqs: vec![SeqAttn {
                // L162 — single-sequence draft/verify path: no distinct recurrent stream.
                stream_id: None,
                q_start: 0,
                q_len,
                past_len,
                slots: slots_for(&block_table, block_size, ctx),
                write_runs: write_runs(&block_table, block_size, past_len, q_len),
                image_spans: Vec::new(),
            }],
            image_embeds: None,
            mrope_positions: None,
        };
        let hidden = model.forward(tokens, &batch, cache);
        let logits = model.logits(&hidden); // [q_len, vocab], every position
        (0..q_len)
            .map(|p| argmax(&logits.row(p)[..vocab]))
            .collect()
    };

    // Prefill the prompt; its last-position argmax is the first generated token.
    let preds = run(&mut cache, prompt, 0);
    stats.forward_passes += 1;
    let mut committed = *preds.last().unwrap();
    let mut cache_len = prompt.len();

    while out.len() < max_tokens {
        out.push(committed);
        history.push(committed);
        drafter.observe(&history[..history.len() - 1], committed);
        if out.len() >= max_tokens {
            break;
        }

        // Draft from history, then verify [committed, d1..dj] in one forward.
        let draft = drafter.propose(&history, k.min(max_tokens - out.len()));
        stats.drafted += draft.len();
        let mut window = Vec::with_capacity(1 + draft.len());
        window.push(committed);
        window.extend_from_slice(&draft);

        let preds = run(&mut cache, &window, cache_len);
        stats.forward_passes += 1;

        // preds[i] is the model's greedy next token after window[i]. Accept
        // draft[i] iff it equals preds[i] (the model would have produced it);
        // stop at the first miss. preds[last_accepted] is the genuinely-new next
        // token (== plain greedy's next), so we always make >= 1 token of progress.
        let mut accepted = 0usize;
        while accepted < draft.len() && draft[accepted] == preds[accepted] {
            accepted += 1;
        }
        stats.draft_hits += accepted;

        // Commit the accepted draft tokens; the model's prediction at the last
        // accepted position becomes the next `committed`. KV for positions
        // [cache_len .. cache_len+1+accepted) is valid and kept; the rejected
        // draft tail's KV is abandoned (overwritten next round).
        for &d in &draft[..accepted] {
            out.push(d);
            history.push(d);
            drafter.observe(&history[..history.len() - 1], d);
            if out.len() >= max_tokens {
                break;
            }
        }
        committed = preds[accepted];
        cache_len += 1 + accepted;
    }

    out.truncate(max_tokens);
    stats.accepted_tokens = out.len();
    SpecResult { tokens: out, stats }
}

/// The accept rule, shared by the verify path and the actor so both name the same row.
///
/// `preds[i]` is greedy's token after `window[i]`, where `window[0]` is the step's own input
/// and `window[1..]` are the drafts. Drafts are accepted while `preds[i] == window[i + 1]`;
/// the first disagreement — or the end of the window — contributes `preds[i]` as the bonus,
/// greedy's genuine next token. The returned run `preds[..=a]` is therefore exactly what greedy
/// would have emitted, one token per verified position, and never empty for non-empty `preds`.
pub fn accepted_run(window: &[u32], preds: &[u32]) -> Vec<u32> {
    let mut run = Vec::with_capacity(preds.len());
    for (i, &p) in preds.iter().enumerate() {
        run.push(p);
        if window.get(i + 1) != Some(&p) {
            break;
        }
    }
    run
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feed every prefix of `history` into `observe` (as `generate_speculative`
    /// and the m4 bridge both do: one call per newly-committed token).
    fn observe_all(drafter: &mut SuffixDrafter, history: &[u32]) {
        for i in 0..history.len() {
            drafter.observe(&history[..i], history[i]);
        }
    }

    /// A long repeated pattern (e.g. a code loop) gets drafted with high
    /// confidence: once the pattern has repeated, the drafter should match
    /// the longest suffix and extend the SAME continuation forward.
    #[test]
    fn repeated_pattern_drafts_with_confidence() {
        let mut drafter = SuffixDrafter::new();
        // "for i in range(n): x[i] = i" repeated a few times, tokenized as ids.
        let pattern = [1u32, 2, 3, 4, 5, 6, 7, 8];
        let mut history = Vec::new();
        for _ in 0..4 {
            history.extend_from_slice(&pattern);
        }
        observe_all(&mut drafter, &history);

        // Tail is a full copy of `pattern` (8 tokens) — should match the
        // MIN(SUFFIX_LENS)=8 length and confidently draft the SAME pattern
        // continuing (since every prior occurrence is followed by `pattern` again).
        let draft = drafter.propose(&history, 4);
        assert!(
            !draft.is_empty(),
            "expected a non-empty confident draft on a repeated pattern"
        );
        assert_eq!(
            &draft[..],
            &pattern[..4],
            "draft should continue the repeated pattern"
        );
    }

    /// A novel tail with no prior occurrence anywhere in history returns an
    /// EMPTY draft (the confidence gate) — never a low-quality guess. This is
    /// the fix for the n-gram's measured loss: no match => zero verify cost.
    #[test]
    fn novel_tail_returns_empty_draft() {
        let mut drafter = SuffixDrafter::new();
        // Some history the drafter has seen...
        let history: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        observe_all(&mut drafter, &history);

        // ...but query with a tail that never occurred before (random-looking,
        // shares no suffix with anything indexed).
        let novel: Vec<u32> = vec![777, 888, 999, 111, 222, 333, 444, 555];
        let draft = drafter.propose(&novel, 4);
        assert!(
            draft.is_empty(),
            "novel/unseen tail must draft nothing, not a weak guess"
        );
    }

    /// When both a short and a long suffix match (with different, conflicting
    /// continuations recorded), the longer/more-specific match wins — it's
    /// tried first and, once it has ANY usable occurrence, short candidates
    /// are never consulted.
    #[test]
    fn longest_match_beats_shorter_fallback() {
        let mut drafter = SuffixDrafter::new();
        // Short suffix [1,2,3,4] is ambiguous: sometimes followed by 100,
        // sometimes by 200 (two different longer contexts).
        // Long suffix [9,9,9,9,1,2,3,4] is unambiguous: always followed by 100.
        let mut history = Vec::new();
        history.extend_from_slice(&[9, 9, 9, 9, 1, 2, 3, 4, 100, 101]); // long-context occurrence #1
        history.extend_from_slice(&[5, 5, 5, 5, 1, 2, 3, 4, 200, 201]); // short-context occurrence, different continuation
        history.extend_from_slice(&[9, 9, 9, 9, 1, 2, 3, 4]); // query tail: matches the LONG context exactly
        observe_all(&mut drafter, &history);

        let draft = drafter.propose(&history, 2);
        assert!(
            !draft.is_empty(),
            "expected a confident draft from the unambiguous long match"
        );
        assert_eq!(
            draft[0], 100,
            "the 8-token match (unambiguous) must win over the noisier 4-token match"
        );
    }

    /// A match at exactly MIN_MATCH_LEN with only ONE historical occurrence
    /// does not meet the confidence gate (needs MIN_OCCURRENCES_SHORT) and
    /// must draft nothing.
    #[test]
    fn short_match_single_occurrence_stays_gated_off() {
        let mut drafter = SuffixDrafter::new();
        // [1,2,3,4] (length == MIN_MATCH_LEN) occurs exactly once before the
        // query tail, followed by 42 — a single corroborating occurrence.
        let mut history = vec![1u32, 2, 3, 4, 42, 43];
        history.extend_from_slice(&[1, 2, 3, 4]); // query tail, same 4-gram
        observe_all(&mut drafter, &history);

        let draft = drafter.propose(&history, 2);
        assert!(
            draft.is_empty(),
            "a MIN_MATCH_LEN suffix with only one occurrence must stay below the confidence gate"
        );
    }

    #[test]
    fn accepted_run_stops_at_the_first_disagreement_and_keeps_the_bonus() {
        // window = [input, d1, d2]; greedy after input = d1 (accept), after d1 = 7 != d2 → bonus 7
        assert_eq!(accepted_run(&[1, 2, 3], &[2, 7, 9]), vec![2, 7]);
        // everything accepted: the last pred is the bonus
        assert_eq!(accepted_run(&[1, 2, 3], &[2, 3, 4]), vec![2, 3, 4]);
        // first draft wrong: only the bonus
        assert_eq!(accepted_run(&[1, 2], &[5, 6]), vec![5]);
    }
}
