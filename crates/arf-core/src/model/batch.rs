//! Description of a forward batch.
//!
//! The engine flattens every scheduled sequence's tokens for this step into one
//! run. [`ForwardBatch`] carries per-token positions (for RoPE) and, per
//! sequence, where its query tokens sit and how to read/write its KV cache.

use crate::cache::WriteRun;

/// Attention bookkeeping for one sequence within a batch.
#[derive(Debug, Clone, Default)]
pub struct SeqAttn {
    /// L162 — STABLE SEQUENCE IDENTITY, carried from `SeqPlan::id`.
    ///
    /// Every other field here is POSITIONAL: `q_start` and the row's index in `batch.seqs`
    /// describe where this sequence sits in *this step's* batch, and that shifts as sequences are
    /// admitted and evicted. Attention does not care — the slot table carries per-sequence
    /// identity, so a row can move freely between steps.
    ///
    /// A RECURRENT layer does care. Gated-delta-net keeps a conv ring and a delta-net matrix per
    /// stream, advanced in place (token t+1 depends on t), so the stream must follow the SEQUENCE,
    /// not the row slot. Keying the banked state by row index (L161) meant row 0 could be
    /// sequence A one step and sequence B the next — each inheriting the other's memory. Measured:
    /// 4 concurrent sequences produced garbage while the same 4 run serially were perfect.
    ///
    /// `None` = no identity supplied (single-stream callers, tests, the m=1 path). Those run one
    /// stream and are unaffected.
    pub stream_id: Option<u64>,
    /// Offset of this sequence's query tokens within the flattened batch.
    pub q_start: usize,
    /// Query tokens this step (prefill: prompt len; decode: 1).
    pub q_len: usize,
    /// Cached tokens that already existed before this step.
    pub past_len: usize,
    /// Physical slot ids for the full context (`past_len + q_len` entries).
    pub slots: Vec<u32>,
    /// Where this step's new K/V rows are written in the pool.
    pub write_runs: Vec<WriteRun>,
    /// Vision (Gemma-3): contiguous spans of image soft-tokens within this sequence's query
    /// range, as `(local_start, len)` where `local_start` is the offset within `q_len` (so
    /// the absolute context key index is `past_len + local_start`). Image tokens attend
    /// BIDIRECTIONALLY within their own span (causal elsewhere), so a query inside a span
    /// extends its causal frontier to the span's end. Empty for text sequences; one entry per
    /// image present in this prefill chunk.
    pub image_spans: Vec<(usize, usize)>,
}

impl SeqAttn {
    /// Total context length attended over (`past_len + q_len`).
    pub fn context_len(&self) -> usize {
        self.past_len + self.q_len
    }
}

/// Precomputed image soft-tokens to splice into the embedding stream (Gemma-3 vision).
/// `embeds` is row-major `[count, hidden]` (the projector output, already in the LM hidden
/// space — no √hidden scale). `rows[i]` is the FLATTENED batch row that image token `i`
/// occupies (i.e. the `<image>` placeholder position). The backend's embed step copies
/// `embeds[i]` into `hidden[rows[i]]` instead of looking up the token id.
#[derive(Debug, Clone)]
pub struct ImageEmbeds {
    pub embeds: Vec<f32>,
    pub hidden: usize,
    pub rows: Vec<usize>,
}

/// A full forward batch.
#[derive(Debug, Clone, Default)]
pub struct ForwardBatch {
    /// Absolute positions for RoPE, one per flattened token.
    pub positions: Vec<u32>,
    /// One entry per scheduled sequence, in batch order.
    pub seqs: Vec<SeqAttn>,
    /// Image soft-tokens to splice at `<image>` placeholder rows (vision). `None` = text.
    pub image_embeds: Option<ImageEmbeds>,
    /// Qwen3.8 vision: per flattened token, the (t, h, w) interleaved M-RoPE positions
    /// (`model::mrope`), parallel to `positions`. `Some` iff some sequence in the batch carries
    /// an image layout; rows of other sequences then hold `[p, p, p]` with `p` their plain
    /// position, which rotates identically. A backend that sees `Some` MUST rotate attention
    /// q/k with these (pair `i` reads axis `imrope_axis(i)`) or refuse the batch; `positions`
    /// stays the KV/causal index. `None` for every text batch — the path is then untouched.
    pub mrope_positions: Option<Vec<[u32; 3]>>,
}
