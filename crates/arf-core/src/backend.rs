//! The backend seam : everything a batched inference backend must
//! provide for the serving loop. The scheduler, actor, admission, streaming,
//! and KV paging are written ONCE against this trait — wgpu/Metal implements
//! it today; another backend implements the same trait and the
//! entire server loop runs unchanged. No backend types may leak upward.

use crate::error::Result;
use crate::model::batch::ForwardBatch;
use crate::sampling::{sample_batch, SeqSampling};

/// A model backend capable of one batched scheduler step. `Send` (it is moved
/// onto the single owner thread) but deliberately NOT `Sync` — the actor is
/// the sole owner .
pub trait BatchedBackend: Send {
    /// Batched trunk pass over a ragged batch; returns each sequence's
    /// last-token logits `[vocab]`, in plan order. Writes this step's K/V
    /// into the backend's resident pool.
    fn forward_batch(&self, input_ids: &[u32], batch: &ForwardBatch) -> Vec<Vec<f32>>;

    /// `(num_blocks, block_size)` of the backend's KV pool. The actor asserts
    /// this matches the scheduler's `EngineConfig` at spawn — the two MUST
    /// agree or block ids would index past the pool (spec §3, memory safety).
    fn kv_geometry(&self) -> (usize, usize);

    /// A HINT for the next step (2026-09-26): `false` = its output token is ignored (a lone
    /// sequence's prefill chunk that does not finish the prompt — `SeqPlan::completes_prompt`), so
    /// the backend may skip computing it. Default: ignored.
    fn set_prefill_logits_needed(&self, _needed: bool) {}

    /// Wait until no GPU work this backend submitted is still running. The serving loop calls it
    /// once at shutdown, before the backend (and its GPU memory) is dropped. Default: nothing.
    fn quiesce(&self) {}

    /// `Some(why)` when this backend serves only plain greedy steps: no sampling, no repetition
    /// penalty, no logprobs, no grammar — anything that needs the logits back takes a path it
    /// cannot run. The serving loop then serves such a request greedily and says so, instead of
    /// reaching that path. Default: every step is served.
    fn greedy_only(&self) -> Option<&'static str> {
        None
    }

    /// One full step: forward + per-sequence sampling. The default reads
    /// logits back and samples on the CPU. A backend may
    /// override to sample on-device with NO readback — that single override
    /// is the entire on-GPU-sampling upgrade path.
    ///
    /// A sampling `Err` from a row means the logits were non-finite or
    /// degenerate after filtering — backend-level corruption, not a
    /// per-request input problem (params are validated at admission) — so
    /// callers treat a step error as FATAL rather than isolating the row.
    fn step(
        &self,
        input_ids: &[u32],
        batch: &ForwardBatch,
        sampling: &[SeqSampling],
    ) -> Result<Vec<u32>> {
        let logits = self.forward_batch(input_ids, batch);
        sample_batch(&logits, sampling)
    }

    /// SPECULATIVE VERIFY (G1 seam, 2026-08-08). Verify a `window` of draft tokens for ONE
    /// sequence sitting at logical `prefix_len` with paged `prefix_slots`, returning the greedy
    /// argmax at each window position. `window[i]`'s prediction is `out[i]`, so the caller accepts
    /// the longest run where `out[i] == window[i+1]` and takes `out[j]` as the bonus token.
    ///
    /// WHY THIS SEAM EXISTS: `generate_speculative` (wgpu/generate.rs, core/speculative.rs) owns a
    /// WHOLE generation loop — prompt in, full sequence out. The serve actor is step-driven over
    /// paged slots, so that API is shape-incompatible with the daemon and, measured 2026-08-08,
    /// `grep -rn "speculative" crates/arf-serve/` returned NOTHING: spec was engine-only and
    /// unreachable from serving. This method is the smallest seam that fixes that — it matches what
    /// the actor already holds per sequence (window, past_len, block-table slots).
    ///
    /// Default `None` = "this backend has no accelerated verify"; the caller then falls back to
    /// ordinary single-token decode, so adding this breaks no existing backend and changes no
    /// shipping behaviour until a caller opts in.
    /// L233 — `stream_id` is the SEQUENCE the window belongs to. It is not optional on a
    /// recurrent architecture: a hybrid model keeps per-sequence state in a bank keyed by this
    /// id, and passing `None` sends verify to the id-less SCRATCH bank, where it predicts from
    /// state belonging to no sequence (L232: preds were pure noise, and committing one poisoned
    /// the stream). Backends without recurrent state may ignore it.
    /// L255 — one draft token from a TRAINED multi-token-prediction head, if the model ships
    /// one. Unlike a suffix/n-gram drafter this proposes on EVERY step by construction — L238
    /// measured `SuffixDrafter` firing on 12% of steps, which diluted a real 1.16x per-window win
    /// to ~1.02x overall. `None` = no head, or the head declined; the caller then falls back to
    /// its own drafter or to ordinary decode. A draft is only ever a PROPOSAL — `verify_window`
    /// decides correctness — so a bad one costs a verify window and nothing else.
    fn mtp_draft(&self, _tok: u32, _past_len: usize, _slots: &[u32]) -> Option<u32> {
        None
    }

    /// L337 — CHAINED MTP: up to `k` draft tokens, each conditioning on the head's own output
    /// for the one before (the DeepSeek-style recurrence llama.cpp's draft-mtp implements).
    /// The default falls back to a single `mtp_draft`, so backends without a chain still
    /// speculate at k=1. `slots` must cover positions [0, past_len + k). A shorter-than-k
    /// return is a valid (shorter) window, not an error.
    fn mtp_draft_chain(&self, tok: u32, past_len: usize, slots: &[u32], _k: usize) -> Vec<u32> {
        self.mtp_draft(tok, past_len, slots).into_iter().collect()
    }

    /// HYBRID PREFIX CACHE: save sequence `stream`'s recurrent state under `key`, right after
    /// the step that left it at the block boundary `key` names. `(saved, key evicted to make
    /// room)`. Backends without recurrent layers, or without snapshots, never save.
    fn state_save(&self, _stream: u64, _key: u64) -> (bool, Option<u64>) {
        (false, None)
    }

    /// HYBRID PREFIX CACHE — an ANCHOR snapshot (`SeqPlan::snapshot_anchor`, 2026-09-26): the
    /// state at the end of a request's shared "system + tools" prefix, which NEW sessions of the
    /// same agent resume from. A backend with a separate anchor pool keeps it there, so the
    /// per-turn snapshots of a long session cannot evict it. Same contract as
    /// [`state_save`](Self::state_save), whose pool the default uses — a backend without an
    /// anchor pool behaves exactly as before anchors existed. The TOOLS anchor (the end of the
    /// tools block, `SeqPlan::snapshot_tools_anchor`, 2026-09-27) is saved here too.
    fn state_save_anchor(&self, stream: u64, key: u64) -> (bool, Option<u64>) {
        self.state_save(stream, key)
    }

    /// HYBRID PREFIX CACHE — a JUNCTION snapshot (`SeqPlan::snapshot_junction`, 2026-09-26): the
    /// state where a prompt LEFT cached history, which a later prompt sharing that longer prefix
    /// (a new agent session in the same project) resumes from. A backend with a junction pool
    /// keeps it there, apart from the anchors, so learned junctions cannot evict the anchors a
    /// template gave us. Same contract as [`state_save`](Self::state_save); the default files it
    /// as an anchor ([`state_save_anchor`](Self::state_save_anchor)), so a backend without a
    /// junction pool behaves as it did with anchors only.
    ///
    /// Since 2026-09-26 the serving loop also saves a conversation's FIRST end-of-prompt snapshot
    /// here (`SeqPlan::snapshot_session_start`), so the conversation's own later turn-end saves
    /// cannot evict it before a new session that opens the same way arrives.
    fn state_save_junction(&self, stream: u64, key: u64) -> (bool, Option<u64>) {
        self.state_save_anchor(stream, key)
    }

    /// HYBRID PREFIX CACHE: restore the state saved under `key` into `stream`, which has not
    /// run yet. `false` = not restored: the serving loop must not run the step as planned.
    /// ROLLING CHECKPOINT (`SeqPlan::snapshot_checkpoint`, 2026-10-07): a resume point inside a
    /// long prompt. A backend with a checkpoint pool keeps it there, so checkpoints — one every
    /// ~1,024 tokens of a long read — evict only each other. Same contract as
    /// [`state_save`](Self::state_save), whose pool the default uses.
    fn state_save_checkpoint(&self, stream: u64, key: u64) -> (bool, Option<u64>) {
        self.state_save(stream, key)
    }

    fn state_restore(&self, _stream: u64, _key: u64) -> bool {
        false
    }

    /// ON-DISK PREFIX CACHE: the state of a cached prefix as bytes — the snapshot
    /// under `key` and the KV of `slots`, the prefix's positions in order. `Err` = not supported
    /// or not available (no such snapshot, another KV format).
    fn prefix_export(&self, _key: u64, _slots: &[u32]) -> std::result::Result<Vec<u8>, String> {
        Err("this backend has no on-disk prefix cache".into())
    }

    /// The inverse of [`prefix_export`](Self::prefix_export): write the KV into `slots` and file
    /// the snapshot under `key`. Returns the snapshot key evicted to make room.
    fn prefix_import(
        &self,
        _key: u64,
        _slots: &[u32],
        _blob: &[u8],
    ) -> std::result::Result<Option<u64>, String> {
        Err("this backend has no on-disk prefix cache".into())
    }

    /// Sequence `stream` has FINISHED (stop, length or evicted by the scheduler): release any
    /// per-sequence state the backend keeps for it — on a hybrid model, its recurrent bank row.
    /// Called only on a real finish, never because a step did not schedule the sequence.
    fn release_stream(&self, _stream: u64) {}

    /// A BLOCK draft: several tokens after `tok` (the pending token of sequence `stream`, at
    /// position `past_len`) from one pass of an attached draft model. Empty = no draft this step
    /// (none attached, or it has no unbroken context for this sequence); the caller falls back.
    fn block_draft(&self, _tok: u32, _past_len: usize, _stream: u64) -> Vec<u32> {
        Vec::new()
    }

    /// Whether a block draft for `stream`'s pending token at `past_len` would run now: a draft
    /// is attached and the stream's context is an unbroken history up to `past_len`. A cheap
    /// question (no GPU work) — the actor's decode round-robin asks it before choosing.
    fn block_draft_ready(&self, _stream: u64, _past_len: usize) -> bool {
        false
    }

    /// The block draft with a SAMPLED selector (speculative sampling, step 2): the drafts after
    /// `tok` and, per drafted position, the distribution q each was drawn from (candidates and
    /// probabilities) at `temperature`, keyed by `seed`. `None` = no draft this step.
    fn block_draft_sampled(
        &self,
        _tok: u32,
        _past_len: usize,
        _stream: u64,
        _temperature: f32,
        _seed: u64,
    ) -> Option<(Vec<u32>, Vec<Vec<(u32, f32)>>)> {
        None
    }

    /// The block draft, GPU-SELECTED (2026-09-22): launch the draft AND its candidate selector
    /// on the GPU and leave the chosen window in a GPU token buffer instead of returning it — the
    /// CPU does not wait. `Some(n)` = launched, proposing `n` tokens; the caller verifies through
    /// [`verify_window_gpu_draft`](Self::verify_window_gpu_draft), which hands the real tokens
    /// back with the predictions. `None` = no draft this step; fall back exactly as for an empty
    /// [`block_draft`](Self::block_draft).
    fn block_draft_gpu(&self, _tok: u32, _past_len: usize, _stream: u64) -> Option<usize> {
        None
    }

    /// [`block_draft_gpu`](Self::block_draft_gpu) with the SAMPLED selector ON THE GPU
    /// (2026-09-26): the draft of [`block_draft_sampled`](Self::block_draft_sampled), drawn at
    /// `temperature` keyed by `seed`, without the CPU waiting for it. Its q stays on the GPU; the
    /// [`verify_window_gpu_draft_sampled`](Self::verify_window_gpu_draft_sampled) that follows reads
    /// it back and accepts with min(1, p/q). `None` = not launched — the caller may still take
    /// `block_draft_sampled`. Default: not launched.
    fn block_draft_gpu_sampled(
        &self,
        _tok: u32,
        _past_len: usize,
        _stream: u64,
        _temperature: f32,
        _seed: u64,
    ) -> Option<usize> {
        None
    }

    /// [`block_draft_gpu_sampled`](Self::block_draft_gpu_sampled), WAITED and READ BACK
    /// (2026-09-27): the same GPU-sampled draft, but the call returns once it has run, with the
    /// drafts and the q each was drawn from — [`block_draft_sampled`](Self::block_draft_sampled)'s
    /// result, from the GPU selector. For the MULTI-STREAM cycle: its record takes CPU token
    /// windows, and each sampled segment's q must reach that segment's
    /// [`RowSampler::draft_q`](crate::sampling::RowSampler::draft_q) before the record so the
    /// verify accepts with min(1, p/q). `None` = not run — the caller may still take
    /// `block_draft_sampled`. Default: not run.
    fn block_draft_gpu_sampled_waited(
        &self,
        _tok: u32,
        _past_len: usize,
        _stream: u64,
        _temperature: f32,
        _seed: u64,
    ) -> Option<(Vec<u32>, Vec<Vec<(u32, f32)>>)> {
        None
    }

    /// Verify a `k`-row window whose tokens are the GPU-resident draft left by
    /// [`block_draft_gpu`](Self::block_draft_gpu). Returns `(window, preds)`: the window's real
    /// tokens, read back only after the verify completed, and greedy's prediction after each row.
    fn verify_window_gpu_draft(
        &self,
        _k: usize,
        _prefix_len: usize,
        _prefix_slots: &[u32],
        _stream_id: u64,
    ) -> Option<(Vec<u32>, Vec<u32>)> {
        None
    }

    fn verify_window(
        &self,
        _window: &[u32],
        _prefix_len: usize,
        _prefix_slots: &[u32],
        _stream_id: u64,
    ) -> Option<Vec<u32>> {
        None
    }

    /// [`verify_window`](Self::verify_window) for a SAMPLED request (speculative sampling,
    /// 2026-09-26): each row's prediction is the token `sampler` draws from that row's logits
    /// (see [`crate::sampling::RowSampler`]) instead of the argmax, and the backend's accept
    /// decision, draft-context commit and state rollback follow those draws. `None` = declined
    /// before anything ran. Default: declined.
    fn verify_window_sampled(
        &self,
        _window: &[u32],
        _prefix_len: usize,
        _prefix_slots: &[u32],
        _stream_id: u64,
        _sampler: &crate::sampling::RowSampler,
    ) -> Option<Vec<u32>> {
        None
    }

    /// [`verify_window_gpu_draft`](Self::verify_window_gpu_draft) for a SAMPLED request — see
    /// [`verify_window_sampled`](Self::verify_window_sampled). Default: declined.
    fn verify_window_gpu_draft_sampled(
        &self,
        _k: usize,
        _prefix_len: usize,
        _prefix_slots: &[u32],
        _stream_id: u64,
        _sampler: &crate::sampling::RowSampler,
    ) -> Option<(Vec<u32>, Vec<u32>)> {
        None
    }

    /// MULTI-STREAM verify (C2, 2026-09-26): several sequences' windows in ONE pass, one
    /// contiguous segment each. Returns each segment's predictions (greedy's token after each of
    /// its rows), in order; the backend commits every segment's accepted rows and rolls back the
    /// rest exactly as [`verify_window`](Self::verify_window) does for one. `None` = declined
    /// before anything ran — the caller takes the ordinary step. Default: declined.
    fn verify_segments(&self, _segs: &[SegReq<'_>]) -> Option<Vec<Vec<u32>>> {
        None
    }

    /// [`verify_segments`](Self::verify_segments) with SAMPLED segments (2026-09-27): `samplers[i]`
    /// is segment i's [`RowSampler`](crate::sampling::RowSampler), or `None` for a plain greedy
    /// segment. A sampled segment's predictions are the tokens its sampler DRAWS from its own rows
    /// of the record's logits ([`segment_predictions`] names the rows), and the backend's accept
    /// count, draft-ring commit and recurrent restore for that segment follow those draws —
    /// exactly what [`verify_window_sampled`](Self::verify_window_sampled) does for one stream. A
    /// greedy segment keeps the argmax. `None` = declined before anything ran; once the record
    /// has run, a failure to draw is a panic, never the argmax (that would serve greedy text to a
    /// sampled request).
    ///
    /// WHY A SEPARATE METHOD and not a sampler field on [`SegReq`]: a backend that implements only
    /// `verify_segments` would ignore a field it does not know and return the ARGMAX for a sampled
    /// segment — greedy text for a sampled request, silently, the bug fixed on 2026-09-26. Here the
    /// default DECLINES whenever any segment is sampled (the caller takes the plain step), and with
    /// no sampler at all it is `verify_segments`, unchanged.
    fn verify_segments_sampled(
        &self,
        segs: &[SegReq<'_>],
        samplers: &[Option<&crate::sampling::RowSampler>],
    ) -> Option<Vec<Vec<u32>>> {
        if samplers.iter().any(Option::is_some) {
            return None;
        }
        self.verify_segments(segs)
    }
}

/// One sequence's part of a [`BatchedBackend::verify_segments`] pass.
#[derive(Debug, Clone, Copy)]
pub struct SegReq<'a> {
    pub stream: u64,
    /// The pending token, then the drafts.
    pub window: &'a [u32],
    /// Tokens the sequence has computed (the window's first position).
    pub prefix_len: usize,
    /// Its slot table covering at least `prefix_len + window.len()` positions.
    pub slots: &'a [u32],
}

/// Each segment's first row in a multi-segment record: the windows laid end to end in `segs`
/// order (the layout `verify_segments` records them in).
pub fn segment_starts(segs: &[SegReq<'_>]) -> Vec<usize> {
    segs.iter()
        .scan(0usize, |at, g| {
            let s = *at;
            *at += g.window.len();
            Some(s)
        })
        .collect()
}

/// C2 SAMPLED (2026-09-27): every segment's predictions from ONE multi-segment record — the step
/// between the record and everything that uses its predictions (accept count, ring commit,
/// recurrent restore), shared by the Metal backend and the CPU tests so the row map is tested
/// where it is used.
///
/// THE ROW MAP: segment g owns the record's rows `[start_g, start_g + w_g)` ([`segment_starts`]).
/// `argmax` is the record's prediction per row and `logits` its `[rows][vocab]` lm_head output,
/// both in that row order. A segment with a sampler gets ITS rows drawn by it
/// ([`RowSampler::sample_rows`](crate::sampling::RowSampler::sample_rows) at its own `prefix_len`,
/// over its own window: the plain path's keyed positions and history, as a lone stream's verify
/// draws them); a segment without one keeps its argmax slice untouched. `logits` is read only for a
/// sampled segment, so a record with none may pass `&[]`. `samplers` is empty (all greedy) or one
/// per segment. Sampled segments are drawn side by side (each row on its own thread, as for one
/// stream).
pub fn segment_predictions(
    segs: &[SegReq<'_>],
    samplers: &[Option<&crate::sampling::RowSampler>],
    argmax: &[u32],
    logits: &[f32],
    vocab: usize,
) -> Result<Vec<Vec<u32>>> {
    use crate::error::ArfError;
    let total: usize = segs.iter().map(|g| g.window.len()).sum();
    if argmax.len() != total {
        return Err(ArfError::Other(format!(
            "segment_predictions: {} predictions for a {total}-row record",
            argmax.len()
        )));
    }
    if !samplers.is_empty() && samplers.len() != segs.len() {
        return Err(ArfError::Other(format!(
            "segment_predictions: {} samplers for {} segments",
            samplers.len(),
            segs.len()
        )));
    }
    let sampler = |i: usize| samplers.get(i).copied().flatten();
    if (0..segs.len()).any(|i| sampler(i).is_some()) && logits.len() < total * vocab {
        return Err(ArfError::Other(format!(
            "segment_predictions: {} logits for {total} rows x {vocab}",
            logits.len()
        )));
    }
    let starts = segment_starts(segs);
    let draw = |i: usize, s: &crate::sampling::RowSampler| {
        let (rs, g) = (starts[i], &segs[i]);
        let w = g.window.len();
        s.sample_rows(
            &logits[rs * vocab..(rs + w) * vocab],
            vocab,
            g.prefix_len,
            g.window,
        )
    };
    let sampled: Vec<usize> = (0..segs.len()).filter(|&i| sampler(i).is_some()).collect();
    let mut drawn: Vec<Option<Result<Vec<u32>>>> = (0..segs.len()).map(|_| None).collect();
    if let [i] = sampled[..] {
        drawn[i] = sampler(i).map(|s| draw(i, s));
    } else if !sampled.is_empty() {
        std::thread::scope(|sc| {
            let hs: Vec<_> = sampled
                .iter()
                .map(|&i| {
                    let s = sampler(i).expect("sampled segment");
                    (i, sc.spawn(move || draw(i, s)))
                })
                .collect();
            for (i, h) in hs {
                drawn[i] = Some(
                    h.join()
                        .expect("segment_predictions: a segment thread panicked"),
                );
            }
        });
    }
    segs.iter()
        .zip(&starts)
        .zip(drawn)
        .map(|((g, &rs), d)| match d {
            Some(r) => r,
            None => Ok(argmax[rs..rs + g.window.len()].to_vec()),
        })
        .collect()
}

/// Each segment's LAST row of a multi-segment record's `[rows][vocab]` logits (2026-09-27): the
/// row whose logits give the token after the segment's window — for a one-row decode segment,
/// its only row. Segment g's last row is `start_g + w_g - 1` ([`segment_starts`]). An error, not
/// a guess, when the logits do not cover the record.
pub fn segment_last_rows(
    segs: &[SegReq<'_>],
    logits: &[f32],
    vocab: usize,
) -> Result<Vec<Vec<f32>>> {
    let total: usize = segs.iter().map(|g| g.window.len()).sum();
    if segs.iter().any(|g| g.window.is_empty()) || logits.len() < total * vocab {
        return Err(crate::error::ArfError::Other(format!(
            "segment_last_rows: {} logits for {total} rows x {vocab} (or an empty segment)",
            logits.len()
        )));
    }
    Ok(segment_starts(segs)
        .iter()
        .zip(segs)
        .map(|(&s, g)| {
            let r = s + g.window.len() - 1;
            logits[r * vocab..(r + 1) * vocab].to_vec()
        })
        .collect())
}

/// BATCHED DECODE LOGITS (2026-09-27): split `n` sequences into records of at most `cap`, as
/// evenly as they go (8 at a cap of 7 -> 4 + 4, not 7 + 1: a one-segment record is not a
/// multi-segment record, and even records cost the same passes). Empty for `n == 0`.
pub fn decode_row_chunks(n: usize, cap: usize) -> Vec<std::ops::Range<usize>> {
    let cap = cap.max(1);
    let k = n.div_ceil(cap);
    let mut out = Vec::with_capacity(k);
    let mut at = 0;
    for c in 0..k {
        let len = n / k + usize::from(c < n % k);
        out.push(at..at + len);
        at += len;
    }
    out
}

/// BATCHED DECODE LOGITS (2026-09-27) — the driver, shared by the Metal backend and the CPU
/// tests. Every one of `n` sequences is a single decode row, and each needs its last-row logits.
/// Before this, a hybrid model ran ONE full record per sequence for them (`step`'s sampled
/// fallback: B=2 took 166 ms a step against 75 for a greedy pair, measured 2026-09-27);
/// here each chunk of [`decode_row_chunks`]`(n, cap)` is ONE multi-segment record through
/// `record`, which returns one logits row per sequence of the chunk, in order — or `None` when
/// it DECLINED BEFORE ANYTHING RAN. A declined chunk falls back to `single(i)` per sequence (the
/// one-record-per-sequence path), whose `None` likewise means declined before running.
///
/// Returns `None` only when nothing has run yet — the caller may still take another path, as
/// the per-sequence loop's first window always could. A decline after any record has run is a
/// panic: those sequences' state has advanced and nothing can hand them to another path. A
/// record that ran but returned the wrong number of rows is a panic too.
pub fn batched_decode_logits<R, S>(
    n: usize,
    cap: usize,
    mut record: R,
    mut single: S,
) -> Option<Vec<Vec<f32>>>
where
    R: FnMut(std::ops::Range<usize>) -> Option<Vec<Vec<f32>>>,
    S: FnMut(usize) -> Option<Vec<f32>>,
{
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(n);
    for chunk in decode_row_chunks(n, cap) {
        if chunk.len() >= 2 {
            if let Some(rows) = record(chunk.clone()) {
                assert_eq!(
                    rows.len(),
                    chunk.len(),
                    "batched decode logits: a record for sequences {chunk:?} returned {} rows",
                    rows.len()
                );
                out.extend(rows);
                continue;
            }
        }
        for i in chunk {
            match single(i) {
                Some(row) => out.push(row),
                None if out.is_empty() => return None,
                None => panic!(
                    "batched decode logits: sequence {i} declined after {} sequences ran",
                    out.len()
                ),
            }
        }
    }
    Some(out)
}

/// A vision encoder that turns a decoded RGB image into LM-hidden-space soft-tokens — the
/// backend-agnostic seam for multimodal input. The actor holds an optional
/// `Box<dyn ImageEncoder>` and runs it during admission; the wgpu `VisionPipeline` implements
/// it. No wgpu types leak into the actor. `Send` (moved onto the owner thread), not `Sync`.
pub trait ImageEncoder: Send {
    /// Encode `rgb` (`[h*w*3]` u8, row-major) of size `w×h` into `[num_tokens(), hidden()]`
    /// soft-tokens, row-major — preprocessing (resize/normalize) included. Ready to splice as
    /// `ImagePrompt.embeds`.
    fn encode(&self, rgb: &[u8], w: usize, h: usize) -> Vec<f32>;
    /// Soft-tokens produced per image (256 for Gemma-3).
    fn num_tokens(&self) -> usize;
    /// The LM hidden dim the soft-tokens occupy (2560 for Gemma-3-4B).
    fn hidden(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::batch::SeqAttn;
    use crate::sampling::SamplingParams;

    /// Logits row = one-hot at (past_len + q_len) % vocab.
    struct Fake {
        vocab: usize,
    }
    impl BatchedBackend for Fake {
        fn forward_batch(&self, _ids: &[u32], batch: &ForwardBatch) -> Vec<Vec<f32>> {
            batch
                .seqs
                .iter()
                .map(|s| {
                    let mut row = vec![0.0; self.vocab];
                    row[(s.past_len + s.q_len) % self.vocab] = 1.0;
                    row
                })
                .collect()
        }
        fn kv_geometry(&self) -> (usize, usize) {
            (64, 4)
        }
    }

    #[test]
    fn default_step_samples_each_row() {
        let b = Fake { vocab: 8 };
        let batch = ForwardBatch {
            positions: vec![0, 1, 2, 0],
            seqs: vec![
                SeqAttn {
                    stream_id: None,
                    q_start: 0,
                    q_len: 3,
                    past_len: 0,
                    slots: vec![],
                    write_runs: Vec::new(),
                    image_spans: Vec::new(),
                },
                SeqAttn {
                    stream_id: None,
                    q_start: 3,
                    q_len: 1,
                    past_len: 5,
                    slots: vec![],
                    write_runs: Vec::new(),
                    image_spans: Vec::new(),
                },
            ],
            image_embeds: None,
            mrope_positions: None,
        };
        let g = SamplingParams::greedy(8);
        let samp = vec![
            SeqSampling {
                params: &g,
                position: 3,
                generated: &[],
            },
            SeqSampling {
                params: &g,
                position: 6,
                generated: &[],
            },
        ];
        let toks = b.step(&[1, 2, 3, 4], &batch, &samp).unwrap();
        assert_eq!(toks, vec![3, 6]); // (0+3)%8, (5+1)%8
    }

    /// Compile-time: the trait is object-safe (the actor holds Box<dyn ...>).
    #[allow(dead_code)]
    fn assert_object_safe(_: &dyn BatchedBackend) {}

    /// A backend with ONE snapshot pool (it implements only `state_save`) still saves anchors:
    /// the default `state_save_anchor` is that pool, so adding the method changed no backend.
    #[test]
    fn anchor_save_defaults_to_the_ordinary_pool() {
        struct OnePool(std::cell::RefCell<Vec<u64>>);
        impl BatchedBackend for OnePool {
            fn forward_batch(&self, _ids: &[u32], _b: &ForwardBatch) -> Vec<Vec<f32>> {
                Vec::new()
            }
            fn kv_geometry(&self) -> (usize, usize) {
                (64, 4)
            }
            fn state_save(&self, _stream: u64, key: u64) -> (bool, Option<u64>) {
                self.0.borrow_mut().push(key);
                (true, Some(7))
            }
        }
        let b = OnePool(Default::default());
        assert_eq!(b.state_save_anchor(1, 42), (true, Some(7)));
        assert_eq!(*b.0.borrow(), vec![42]);
        // A junction (2026-09-26) defaults to the anchor save, i.e. here the same one pool.
        assert_eq!(b.state_save_junction(1, 43), (true, Some(7)));
        assert_eq!(*b.0.borrow(), vec![42, 43]);
        // And a backend with no snapshots at all saves none of them.
        assert_eq!(Fake { vocab: 8 }.state_save_anchor(1, 42), (false, None));
        assert_eq!(Fake { vocab: 8 }.state_save_junction(1, 42), (false, None));
    }

    fn sampled(seed: u64) -> crate::sampling::RowSampler {
        crate::sampling::RowSampler {
            params: SamplingParams {
                temperature: 0.7,
                top_k: Some(20),
                top_p: Some(0.95),
                seed,
                ..Default::default()
            },
            history: Vec::new(),
            draft_q: Vec::new(),
        }
    }

    /// A backend that knows only the greedy multi-segment verify (2026-09-27): the default
    /// `verify_segments_sampled` must DECLINE a record with any sampled segment — never hand back
    /// its argmax for a sampled request — and with no sampler it is `verify_segments` itself.
    #[test]
    fn a_greedy_only_backend_declines_sampled_segments() {
        struct GreedyOnly(std::cell::Cell<usize>);
        impl BatchedBackend for GreedyOnly {
            fn forward_batch(&self, _ids: &[u32], _b: &ForwardBatch) -> Vec<Vec<f32>> {
                Vec::new()
            }
            fn kv_geometry(&self) -> (usize, usize) {
                (64, 4)
            }
            fn verify_segments(&self, segs: &[SegReq<'_>]) -> Option<Vec<Vec<u32>>> {
                self.0.set(self.0.get() + 1);
                Some(segs.iter().map(|g| vec![7; g.window.len()]).collect())
            }
        }
        let b = GreedyOnly(std::cell::Cell::new(0));
        let slots: Vec<u32> = (0..16).collect();
        let segs = [
            SegReq {
                stream: 1,
                window: &[3, 4],
                prefix_len: 5,
                slots: &slots,
            },
            SegReq {
                stream: 2,
                window: &[9],
                prefix_len: 8,
                slots: &slots,
            },
        ];
        let s = sampled(1);
        assert_eq!(b.verify_segments_sampled(&segs, &[None, Some(&s)]), None);
        assert_eq!(b.0.get(), 0, "a declined record must not have run");
        assert_eq!(
            b.verify_segments_sampled(&segs, &[None, None]),
            Some(vec![vec![7, 7], vec![7]])
        );
        assert_eq!(
            b.verify_segments_sampled(&segs, &[]),
            Some(vec![vec![7, 7], vec![7]])
        );
        assert_eq!(b.0.get(), 2);
    }

    /// THE ROW MAP of a multi-segment record: segment g's predictions come from rows
    /// `[start_g, start_g + w_g)` — drawn by its sampler from THOSE rows' logits at its own
    /// positions, or its argmax slice untouched when it has none. Every row's logits here point at
    /// a different token, so a draw from any other row (an off-by-one start, the wrong segment's
    /// rows, row 0 for everything) names the wrong token.
    #[test]
    fn segment_predictions_draw_each_segment_from_its_own_rows() {
        let vocab = 32;
        // windows 3 + 1 + 4 = the 8-row record; row r's logits: one token far ahead of the rest
        let hot = |r: usize| ((r * 5 + 1) % vocab) as u32;
        let mut logits = vec![0.0f32; 8 * vocab];
        for r in 0..8 {
            for t in 0..vocab {
                logits[r * vocab + t] = ((r * 7 + t * 3) % 11) as f32 / 10.0;
            }
            logits[r * vocab + hot(r) as usize] = 40.0;
        }
        let argmax: Vec<u32> = (100..108).collect(); // the record's own predictions, recognisable
        let slots: Vec<u32> = (0..64).collect();
        let w0 = [1u32, hot(0), 2];
        let w1 = [4u32];
        let w2 = [6u32, hot(4), hot(5), 9];
        let segs = [
            SegReq {
                stream: 1,
                window: &w0,
                prefix_len: 10,
                slots: &slots,
            },
            SegReq {
                stream: 2,
                window: &w1,
                prefix_len: 20,
                slots: &slots,
            },
            SegReq {
                stream: 3,
                window: &w2,
                prefix_len: 30,
                slots: &slots,
            },
        ];
        assert_eq!(segment_starts(&segs), vec![0, 3, 4]);
        let (s1, s2) = (sampled(11), sampled(12));
        let got = segment_predictions(
            &segs,
            &[None, Some(&s1), Some(&s2)],
            &argmax,
            &logits,
            vocab,
        )
        .unwrap();
        assert_eq!(
            got[0],
            vec![100, 101, 102],
            "a greedy segment keeps its argmax slice"
        );
        assert_eq!(got[1], vec![hot(3)], "segment 1 is row 3");
        assert_eq!(
            got[2],
            vec![hot(4), hot(5), hot(6), hot(7)],
            "segment 2 is rows 4..8"
        );
        // and each draw is the lone stream's: the sampler over that segment's rows alone
        assert_eq!(
            got[2],
            s2.sample_rows(&logits[4 * vocab..], vocab, 30, &w2)
                .unwrap()
        );
        // no sampler = the argmax, split by segment, with no logits needed
        assert_eq!(
            segment_predictions(&segs, &[], &argmax, &[], vocab).unwrap(),
            vec![vec![100, 101, 102], vec![103], vec![104, 105, 106, 107]]
        );
        // a record whose shape does not match is an error, not a guess
        assert!(segment_predictions(&segs, &[None, Some(&s1), None], &argmax, &[], vocab).is_err());
        assert!(segment_predictions(&segs, &[None], &argmax, &logits, vocab).is_err());
        assert!(segment_predictions(&segs, &[], &argmax[..7], &[], vocab).is_err());
    }

    /// Each segment's LAST row: row r of the record carries r in every column, so any other row
    /// (the segment's first, an off-by-one, the previous segment's) reads a different value.
    #[test]
    fn segment_last_rows_take_each_segments_last_row() {
        let vocab = 5;
        let slots: Vec<u32> = (0..64).collect();
        let (w0, w1, w2) = ([1u32, 2, 3], [4u32], [5u32, 6, 7, 8]);
        let segs = [
            SegReq {
                stream: 1,
                window: &w0,
                prefix_len: 10,
                slots: &slots,
            },
            SegReq {
                stream: 2,
                window: &w1,
                prefix_len: 20,
                slots: &slots,
            },
            SegReq {
                stream: 3,
                window: &w2,
                prefix_len: 30,
                slots: &slots,
            },
        ];
        let logits: Vec<f32> = (0..8).flat_map(|r| vec![r as f32; vocab]).collect();
        let rows = segment_last_rows(&segs, &logits, vocab).unwrap();
        assert_eq!(
            rows,
            vec![vec![2.0; vocab], vec![3.0; vocab], vec![7.0; vocab]]
        );
        assert!(segment_last_rows(&segs, &logits[..7 * vocab], vocab).is_err());
    }

    /// Chunks never exceed the cap, cover 0..n in order, and are as even as they go.
    #[test]
    fn decode_row_chunks_are_even_and_capped() {
        let lens = |n| {
            decode_row_chunks(n, 7)
                .iter()
                .map(|r| r.len())
                .collect::<Vec<_>>()
        };
        assert!(decode_row_chunks(0, 7).is_empty());
        assert_eq!(lens(1), vec![1]);
        assert_eq!(lens(2), vec![2]);
        assert_eq!(lens(7), vec![7]);
        assert_eq!(lens(8), vec![4, 4]);
        assert_eq!(lens(9), vec![5, 4]);
        assert_eq!(lens(15), vec![5, 5, 5]);
        for n in 1..64 {
            let c = decode_row_chunks(n, 7);
            assert_eq!(c.first().unwrap().start, 0);
            assert_eq!(c.last().unwrap().end, n);
            assert!(c.windows(2).all(|w| w[0].end == w[1].start));
            assert!(c.iter().all(|r| r.len() <= 7 && (n < 2 || r.len() >= 2)));
        }
    }

    /// A mock of the two paths `batched_decode_logits` drives: a sequence is (stream, position)
    /// and its logits a vector naming both — so the batched rows can be compared with the
    /// per-sequence rows directly. `record` builds the chunk's one-row segments and maps rows
    /// with `segment_last_rows`, as the Metal backend does after its record.
    struct DecodeMock {
        seqs: Vec<(u64, usize)>,
        /// chunks whose record declines (before running)
        decline_record: Vec<usize>,
        /// sequences whose single window declines (before running)
        decline_single: Vec<usize>,
        records: Vec<std::ops::Range<usize>>,
        singles: Vec<usize>,
    }

    const MV: usize = 6;

    fn row_of(stream: u64, pos: usize) -> Vec<f32> {
        (0..MV)
            .map(|j| (stream * 1000 + pos as u64 * 10 + j as u64) as f32)
            .collect()
    }

    impl DecodeMock {
        fn new(seqs: Vec<(u64, usize)>) -> Self {
            DecodeMock {
                seqs,
                decline_record: Vec::new(),
                decline_single: Vec::new(),
                records: Vec::new(),
                singles: Vec::new(),
            }
        }
        fn run(&mut self, cap: usize) -> Option<Vec<Vec<f32>>> {
            let seqs = self.seqs.clone();
            let (dr, ds) = (self.decline_record.clone(), self.decline_single.clone());
            let records = std::cell::RefCell::new(Vec::new());
            let singles = std::cell::RefCell::new(Vec::new());
            let out = batched_decode_logits(
                seqs.len(),
                cap,
                |r: std::ops::Range<usize>| {
                    if dr.contains(&r.start) {
                        return None;
                    }
                    records.borrow_mut().push(r.clone());
                    let toks: Vec<[u32; 1]> = r.clone().map(|i| [i as u32]).collect();
                    let slots: Vec<u32> = (0..4096).collect();
                    let segs: Vec<SegReq<'_>> = r
                        .clone()
                        .zip(&toks)
                        .map(|(i, t)| SegReq {
                            stream: seqs[i].0,
                            window: t,
                            prefix_len: seqs[i].1,
                            slots: &slots,
                        })
                        .collect();
                    // the record's logits: row k is the k-th segment's (stream, position)
                    let logits: Vec<f32> = segs
                        .iter()
                        .flat_map(|g| row_of(g.stream, g.prefix_len))
                        .collect();
                    Some(segment_last_rows(&segs, &logits, MV).unwrap())
                },
                |i| {
                    if ds.contains(&i) {
                        return None;
                    }
                    singles.borrow_mut().push(i);
                    Some(row_of(seqs[i].0, seqs[i].1))
                },
            );
            self.records = records.into_inner();
            self.singles = singles.into_inner();
            out
        }
    }

    fn seqs(n: usize) -> Vec<(u64, usize)> {
        (0..n).map(|i| (100 + i as u64 * 7, 50 + i * 13)).collect()
    }

    fn per_sequence(s: &[(u64, usize)]) -> Vec<Vec<f32>> {
        s.iter().map(|&(st, p)| row_of(st, p)).collect()
    }

    /// Every sequence gets its OWN row — the row the per-sequence path gives it — through one
    /// record per chunk, and the per-sequence path never runs.
    #[test]
    fn batched_decode_logits_map_each_sequence_to_its_own_row() {
        for n in [2, 5, 7, 8, 9, 15] {
            let mut m = DecodeMock::new(seqs(n));
            let out = m.run(7).expect("ran");
            assert_eq!(out, per_sequence(&m.seqs), "n={n}");
            assert_eq!(m.records, decode_row_chunks(n, 7), "n={n}");
            assert!(m.singles.is_empty(), "n={n}: {:?}", m.singles);
        }
        // one sequence is not a multi-segment record: the per-sequence path
        let mut m = DecodeMock::new(seqs(1));
        assert_eq!(m.run(7).unwrap(), per_sequence(&m.seqs));
        assert!(m.records.is_empty());
        assert_eq!(m.singles, vec![0]);
    }

    /// A record that declines (before running) sends ITS chunk's sequences to the per-sequence
    /// path, in order; the other chunks still take their records.
    #[test]
    fn a_declined_record_falls_back_to_one_window_per_sequence() {
        let mut m = DecodeMock::new(seqs(9)); // chunks 0..5, 5..9
        m.decline_record = vec![0];
        assert_eq!(m.run(7).unwrap(), per_sequence(&m.seqs));
        assert_eq!(m.records, vec![5..9]);
        assert_eq!(m.singles, vec![0, 1, 2, 3, 4]);
        let mut m = DecodeMock::new(seqs(9));
        m.decline_record = vec![5];
        assert_eq!(m.run(7).unwrap(), per_sequence(&m.seqs));
        assert_eq!(m.records, vec![0..5]);
        assert_eq!(m.singles, vec![5, 6, 7, 8]);
    }

    /// Nothing has run: the caller may still take another path (`None`), as before.
    #[test]
    fn a_decline_before_anything_ran_returns_none() {
        let mut m = DecodeMock::new(seqs(3));
        m.decline_record = vec![0];
        m.decline_single = vec![0];
        assert!(m.run(7).is_none());
        assert!(m.records.is_empty() && m.singles.is_empty());
    }

    /// A decline after a record has run cannot be handed to another path: panic.
    #[test]
    #[should_panic(expected = "declined after")]
    fn a_decline_after_a_record_ran_panics() {
        let mut m = DecodeMock::new(seqs(9));
        m.decline_record = vec![5];
        m.decline_single = vec![6];
        let _ = m.run(7);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// L358 — THE BACKEND SEAM, in the crate both backends can actually see.
// ═══════════════════════════════════════════════════════════════════════════════════════════

/// What a decode layer structurally IS. The nine architectures this engine runs differ in
/// exactly two structural ways, and every backend needs the same three-way answer.
///
/// This is the ONE piece of the seam that is genuinely backend-independent: it names a property
/// of the *model*, not of any GPU API, so it belongs here rather than beside Metal handles.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LayerKind {
    /// Full attention + dense gate/up/down FFN. Llama, Gemma, muse-glimmer.
    DenseAttn,
    /// Full attention + routed mixture-of-experts. Qwen3-Coder-30B.
    MoeAttn,
    /// Gated-delta-net recurrence in place of attention. 48 of Qwen3.8-27B's 64 layers.
    /// The recurrent state is destructive and per-row, which is why a verify window must chain
    /// rows serially — the property that caps speculative decoding on this model.
    Recurrent,
}

/// **THE BACKEND SEAM.** One method that matters.
///
/// L358 — this trait was first written in `arf-gpu` (L354), which was wrong and is worth
/// recording: any other backend depends on `arf-core` only, and should — a backend must
/// not pull in the Metal one to reach a trait. The compile probe that "proved" the trait
/// implementable lived in `arf-gpu`, where it trivially resolved, so it could not catch
/// this. A seam is only a seam if both sides can see it.
///
/// **Why one method and not eighty-one.** llama.cpp puts a seam between every *operation* — 81
/// ops dispatched per graph node — which buys 151 architectures and costs a dispatch per op. We
/// measured that cost on our own code: folding ~100 per-row dispatches into one kernel took a
/// verify from 119 ms to 78 ms (L336), a 34% win from nothing but not dispatching. So the seam
/// goes one level up: a backend receives a DESCRIBED LAYER and emits its own fused dispatches.
/// Crossed 64 times per token against ~800 dispatches, the indirection is unmeasurable — while
/// buying what vLLM (281 models over 17 shared layer modules) and SGLang (222 over 34) get.
///
/// **Generic over the backend's own types**, because the layer description borrows raw handles
/// that cannot live in this crate: `Q4ksMtl` and `MTLBuffer` on Metal, its own handles elsewhere.
/// The trait fixes the SHAPE of the contract — a layer, model constants, per-step inputs, an
/// encoder — and lets each backend name its own handles.
pub trait LayerEncoder {
    /// This backend's layer description (Metal: `MegaLayer`).
    type Layer<'a>;
    /// Values bound once at model load: final norm, lm_head, embed table, geometry.
    type ModelConstants<'a>;
    /// Values that change per token: input tokens, batch rows, slot tables, past lengths.
    type StepInputs<'a>;
    /// What this backend records into — on Metal, a command encoder.
    type Encoder<'a>;

    /// The layer's structural kind, from the backend's own description of it.
    fn layer_kind(layer: &Self::Layer<'_>) -> LayerKind;

    /// Encode ONE layer. The layer knows what it is; the backend knows how to dispatch it.
    fn encode_layer(
        &self,
        layer: &Self::Layer<'_>,
        mc: &Self::ModelConstants<'_>,
        step: &Self::StepInputs<'_>,
        enc: &mut Self::Encoder<'_>,
    ) -> std::result::Result<(), String>;

    /// Encode the trunk epilogue: final norm, lm_head, sampling. Separate because it runs once
    /// per step rather than once per layer, and a backend may fuse it differently — ours takes
    /// a GEMV below b=8 and a tiled GEMM above (L188).
    fn encode_epilogue(
        &self,
        mc: &Self::ModelConstants<'_>,
        step: &Self::StepInputs<'_>,
        enc: &mut Self::Encoder<'_>,
    ) -> std::result::Result<(), String>;
}
