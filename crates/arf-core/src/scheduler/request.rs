//! Requests and their in-flight state.

use crate::sampling::{Sampler, SamplingParams};

/// A generation request submitted to the engine.
#[derive(Debug, Clone)]
pub struct Request {
    pub id: u64,
    /// Prompt token ids (already tokenized).
    pub prompt: Vec<u32>,
    pub params: SamplingParams,
    /// Vision (Gemma-3): precomputed image soft-tokens `[count, hidden]` row-major, and the
    /// LOCAL prompt positions (indices into `prompt`) they occupy (the `<image>` placeholder
    /// tokens). Empty for text requests. The engine splices `embeds[i]` into the hidden
    /// stream at `image_positions[i]` during prefill instead of embedding that token id.
    pub image: Option<ImagePrompt>,
    /// PREFIX ANCHOR (2026-09-26): how many leading prompt tokens are the request's shared
    /// "system + tools" prefix — the point where two sessions of the same agent, same system
    /// prompt and same tools, but different first user messages, stop agreeing. The HTTP layer
    /// computes it from the chat template (`arf-serve`'s `prefix_anchor`); `None` = no anchor
    /// (a raw completion, a short system prompt, or a path that could not compute one).
    /// With state snapshots on, the scheduler snapshots the recurrent state there too
    /// (`SeqPlan::snapshot_anchor`), so a NEW session can resume at it: measured 2026-09-26,
    /// Claude Code's first turn was 88-90 s on every new session because the only snapshot a
    /// prompt took was one window before its own end, inside the first user message.
    pub prefix_anchor: Option<usize>,
    /// TOOLS ANCHOR (2026-09-27): how many leading prompt tokens are the request's TOOLS block
    /// alone, when its template renders the tools BEFORE the system text (Qwen3.8's does) — the
    /// point where two sessions of the same agent with the same tools but a DIFFERENT system
    /// text stop agreeing. Claude Code's system text carries the working directory, so a new
    /// session in another directory shares only the ~12K-token tool schemas with the last one
    /// and never matched its `prefix_anchor` snapshot (measured 2026-09-27: `anchor snapshot at
    /// 13056` for session 1, no resume for sessions 2 and 3 in other directories). The scheduler
    /// snapshots here too, into the anchor pool (`SeqPlan::snapshot_tools_anchor`). `None` = no
    /// tools, a template that renders the system text first, or `ARF_NO_TOOLS_ANCHOR=1`.
    pub tools_anchor: Option<usize>,
    /// HEADER TAIL (2026-10-05): how many of the prompt's LAST tokens the next turn of the same
    /// conversation renders differently — the part of the assistant header the template appends
    /// that the next turn does not share — measured from the template by the server (`arf-serve`'s `prefix_anchor::header_tail`). The end-of-prompt snapshot backs off
    /// this far instead of the fixed [`crate::scheduler::SNAPSHOT_TAIL_TOKENS`]. Measured
    /// 2026-10-05: a repeated 34,972-token prompt resumed at 34,816 and re-prefilled 156 tokens
    /// (1.56 s to the first token) where the last window boundary before the header is 34,944.
    /// `None` = unknown (a raw completion, a path that could not measure it): the fixed tail.
    pub header_tail: Option<usize>,
    /// STOP STRINGS: the request's text stop condition (OpenAI `stop`, Anthropic
    /// `stop_sequences`), checked after every generated token. `None` = none (the default).
    pub stop_check: Option<StopCheck>,
}

/// A text stop condition (OpenAI `stop`, Anthropic `stop_sequences`). The scheduler holds only
/// token ids; the server, which owns the tokenizer, supplies the check. It is called with every
/// generated token so far, the newest last, after each token is appended, and returns true when
/// that newest token's text completed a stop string. The sequence then retires exactly where a
/// stop token retires it (`advance_and_retire`: a speculative run is truncated there, the KV
/// freed, held snapshots published) with [`FinishReason::StopString`] — and unlike a stop token,
/// the token IS output: its text up to the stop string is the end of the reply.
///
/// Stateless (it sees the whole output each call), so a `Request` stays `Clone`.
#[derive(Clone)]
pub struct StopCheck(std::sync::Arc<dyn Fn(&[u32]) -> bool + Send + Sync>);

impl StopCheck {
    pub fn new(check: impl Fn(&[u32]) -> bool + Send + Sync + 'static) -> Self {
        StopCheck(std::sync::Arc::new(check))
    }

    /// Did the newest of `output` (every generated token so far) complete a stop string?
    pub fn hit(&self, output: &[u32]) -> bool {
        (self.0)(output)
    }
}

impl std::fmt::Debug for StopCheck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("StopCheck")
    }
}

/// Precomputed image embeddings + their prompt positions (see `Request::image`).
#[derive(Debug, Clone, Default)]
pub struct ImagePrompt {
    pub embeds: Vec<f32>,
    pub hidden: usize,
    pub positions: Vec<usize>,
    /// Qwen3.8 vision: the prompt's interleaved M-RoPE layout (see `model::mrope`). `Some`
    /// means (a) every step of this sequence carries per-token (t, h, w) rope positions —
    /// including its decode steps, which rotate at `kv_index - delta` — and (b) image tokens
    /// attend CAUSALLY like text (no bidirectional span: llama.cpp's
    /// `mtmd_decode_use_non_causal` is false for Qwen-VL). `None` = Gemma-3's behaviour,
    /// unchanged: plain positions and bidirectional image spans.
    pub mrope: Option<crate::model::mrope::MropeLayout>,
    /// The embedded rows attend CAUSALLY, like text, with plain positions (2026-09-27, Qwen3-Omni
    /// AUDIO: the Thinker's mask is causal over every token, and an audio span's rope positions
    /// are `(p, p, p)`, so `mrope` stays `None`). `false` keeps Gemma-3's bidirectional image
    /// span. A layout (`mrope: Some`) is causal whatever this says.
    pub causal: bool,
}

impl Request {
    pub fn new(id: u64, prompt: Vec<u32>, params: SamplingParams) -> Self {
        Request {
            id,
            prompt,
            params,
            image: None,
            prefix_anchor: None,
            tools_anchor: None,
            header_tail: None,
            stop_check: None,
        }
    }

    /// The same request with its text stop condition set (see [`StopCheck`]).
    pub fn with_stop_check(mut self, check: Option<StopCheck>) -> Self {
        self.stop_check = check;
        self
    }

    /// The same request with its shared-prefix anchor set (see [`Request::prefix_anchor`]).
    pub fn with_prefix_anchor(mut self, anchor: Option<usize>) -> Self {
        self.prefix_anchor = anchor;
        self
    }

    /// The same request with its tools-block anchor set (see [`Request::tools_anchor`]).
    pub fn with_tools_anchor(mut self, anchor: Option<usize>) -> Self {
        self.tools_anchor = anchor;
        self
    }

    /// The same request with its header tail set (see [`Request::header_tail`]).
    pub fn with_header_tail(mut self, tail: Option<usize>) -> Self {
        self.header_tail = tail;
        self
    }

    /// A vision request: prompt + precomputed image soft-tokens at the given positions.
    pub fn with_image(
        id: u64,
        prompt: Vec<u32>,
        params: SamplingParams,
        image: ImagePrompt,
    ) -> Self {
        Request {
            id,
            prompt,
            params,
            image: Some(image),
            prefix_anchor: None,
            tools_anchor: None,
            header_tail: None,
            stop_check: None,
        }
    }
}

/// Why a sequence stopped generating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// Hit a stop token.
    Stop,
    /// Reached the `max_tokens` budget.
    Length,
    /// Evicted by the server (client disconnected or stalled past its backlog).
    Evicted,
    /// The newest token completed a stop string ([`StopCheck`]). That token is part of the
    /// output — the server cuts the text at the stop string — where a stop token is not.
    StopString,
}

/// Lifecycle state of a sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStatus {
    Waiting,
    Running,
    Finished(FinishReason),
}

/// A sequence: a request plus its evolving token list and cache bookkeeping.
#[derive(Debug)]
pub struct Sequence {
    pub id: u64,
    pub params: SamplingParams,
    /// Prompt followed by all generated tokens.
    pub tokens: Vec<u32>,
    pub prompt_len: usize,
    /// Number of leading tokens whose KV is already in the cache.
    pub num_computed: usize,
    /// Physical block ids backing this sequence's KV.
    pub block_table: Vec<u32>,
    pub status: SequenceStatus,
    /// Prefix-cache bookkeeping : leading FULL blocks whose KV is in
    /// the shared cache (reused on admission or registered as this sequence
    /// filled them). Always `0` when prefix caching is disabled.
    pub num_cached_blocks: usize,
    /// Chained content hash of the first `num_cached_blocks` blocks — the
    /// resume point for incremental block-hash registration. Seeded when reset.
    pub prefix_hash: u64,
    /// Vision: image soft-tokens + their prompt positions, spliced during prefill. `None`
    /// for text. (Image requests disable prefix caching for the sequence — the cached
    /// blocks would hash token ids, not the spliced image embeddings.)
    pub image: Option<ImagePrompt>,
    /// STATE SNAPSHOTS (hybrid recurrent models): the snapshot key the prefix match landed on.
    /// The backend must restore that recurrent state into this sequence BEFORE its first step
    /// runs; the scheduler hands it over once, in that step's `SeqPlan::restore_key`.
    pub restore_key: Option<u64>,
    /// STATE SNAPSHOTS: a recurrent-state snapshot the BACKEND has taken for this sequence at a
    /// block boundary, held back until the sequence FINISHES. It must not be advertised earlier:
    /// the KV blocks covering the same tokens are published only on completion (L125 — publishing
    /// mid-prefill hands out live write targets), so a turn attaching to the state sooner would
    /// read KV from a still-prefilling sequence. Measured doing exactly that, 2026-09-21.
    pub pending_snapshot: Option<u64>,
    /// PREFIX ANCHOR: the request's shared-prefix length in tokens ([`Request::prefix_anchor`]).
    pub prefix_anchor: Option<usize>,
    /// PREFIX ANCHOR: the key of the anchor snapshot the scheduler planned for this sequence, so
    /// `Scheduler::hold_snapshot` can tell the anchor from the end-of-prompt snapshot.
    pub anchor_key: Option<u64>,
    /// PREFIX ANCHOR: the anchor snapshot the backend took, HELD until the sequence finishes for
    /// exactly the reason `pending_snapshot` is — both are published together.
    pub pending_anchor: Option<u64>,
    /// TOOLS ANCHOR (2026-09-27): the request's tools-block length in tokens
    /// ([`Request::tools_anchor`]).
    pub tools_anchor: Option<usize>,
    /// TOOLS ANCHOR: the key of the tools-anchor snapshot planned for this sequence, so
    /// `Scheduler::hold_snapshot` can tell it from the (system) anchor and the others.
    pub tools_anchor_key: Option<u64>,
    /// TOOLS ANCHOR: the tools-anchor snapshot the backend took, HELD until the sequence
    /// finishes and published with the others, for the reason `pending_snapshot` is.
    pub pending_tools_anchor: Option<u64>,
    /// HEADER TAIL (2026-10-05): [`Request::header_tail`].
    pub header_tail: Option<usize>,
    /// JUNCTION SNAPSHOTS (2026-09-26): how many leading prompt tokens the prefix cache held KV
    /// for when this sequence was admitted — the verified chain hit, BEFORE the hybrid match
    /// backed it off to the deepest recurrent-state snapshot (`BlockManager::match_prefix`).
    /// Block-aligned. 0 when prefix caching or state snapshots are off, or nothing matched.
    pub kv_match_len: usize,
    /// JUNCTION SNAPSHOTS: the window-aligned boundary where this prompt leaves cached history
    /// (from `kv_match_len`), when it lies far enough past the resume point to be worth a
    /// snapshot. Set at admission by the scheduler; `None` = no junction for this sequence.
    pub junction_at: Option<usize>,
    /// JUNCTION SNAPSHOTS: the key of the junction snapshot planned for this sequence, so
    /// `Scheduler::hold_snapshot` can tell it from the anchor and the end-of-prompt snapshot.
    pub junction_key: Option<u64>,
    /// JUNCTION SNAPSHOTS: the junction snapshot the backend took, HELD until the sequence
    /// finishes and published with the others, for the reason `pending_snapshot` is.
    pub pending_junction: Option<u64>,
    /// ROLLING CHECKPOINT (2026-10-07): the key and position of the checkpoint planned for this
    /// sequence's current chunk (`SeqPlan::snapshot_checkpoint`), so `hold_snapshot` can tell it apart.
    pub checkpoint_key: Option<(u64, usize)>,
    /// The latest checkpoint the backend took, and where: HELD and published with the others when
    /// the sequence finishes or is evicted. A newer one replaces it (only the latest is kept).
    pub pending_checkpoint: Option<(u64, usize)>,
    /// SESSION-START SNAPSHOTS (2026-09-26): this sequence is the FIRST turn of a conversation —
    /// it did not resume from the turn-end snapshot of an earlier turn of its own conversation
    /// (it resumed at 0, at or before its anchor, or at a published anchor / junction) — and its
    /// end-of-prompt boundary is worth a long-lived slot. Its end-of-prompt snapshot is then
    /// filed in the JUNCTION pool (`SeqPlan::snapshot_session_start`), so the turns that follow
    /// cannot evict it and a later session that opens the same way resumes there. Set at
    /// admission by the scheduler (`Scheduler::is_session_start`).
    pub session_start: bool,
    /// STOP STRINGS: the request's text stop condition ([`Request::stop_check`]).
    pub stop_check: Option<StopCheck>,
    sampler: Sampler,
}

/// Seed of the per-block chained content hash (FNV-1a basis), so block 0's hash
/// is `H(PREFIX_HASH_SEED, tokens[0..block_size])`.
pub const PREFIX_HASH_SEED: u64 = 0xcbf2_9ce4_8422_2325;

impl Sequence {
    pub fn from_request(req: Request) -> Self {
        let sampler = Sampler::new(req.params.seed);
        let prompt_len = req.prompt.len();
        Sequence {
            id: req.id,
            params: req.params,
            tokens: req.prompt,
            prompt_len,
            num_computed: 0,
            block_table: Vec::new(),
            status: SequenceStatus::Waiting,
            num_cached_blocks: 0,
            prefix_hash: PREFIX_HASH_SEED,
            image: req.image,
            restore_key: None,
            pending_snapshot: None,
            prefix_anchor: req.prefix_anchor,
            anchor_key: None,
            pending_anchor: None,
            tools_anchor: req.tools_anchor,
            tools_anchor_key: None,
            pending_tools_anchor: None,
            header_tail: req.header_tail,
            kv_match_len: 0,
            junction_at: None,
            junction_key: None,
            pending_junction: None,
            checkpoint_key: None,
            pending_checkpoint: None,
            session_start: false,
            stop_check: req.stop_check,
            sampler,
        }
    }

    /// Total tokens (prompt + generated).
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Tokens awaiting a forward pass this step (`len - num_computed`).
    pub fn num_pending(&self) -> usize {
        self.tokens.len() - self.num_computed
    }

    /// Generated tokens only (excludes the prompt).
    pub fn output_tokens(&self) -> &[u32] {
        &self.tokens[self.prompt_len..]
    }

    /// Sample the next token from this sequence's logits (`[vocab]`) WITHOUT
    /// appending it — the scheduler appends via `advance_and_retire`, which
    /// also handles the chunked-prefill case where a sampled token is discarded.
    pub fn sample_next(&mut self, logits: &[f32]) -> crate::error::Result<u32> {
        self.sampler.sample(logits, &self.params)
    }

    /// Reset KV progress so the sequence is recomputed from scratch (preemption).
    /// The block table is freed by the caller (`BlockManager::free`) first; here
    /// we drop the prefix-cache bookkeeping so re-admission re-matches cleanly.
    pub fn reset_for_recompute(&mut self) {
        self.num_computed = 0;
        self.num_cached_blocks = 0;
        self.prefix_hash = PREFIX_HASH_SEED;
        self.restore_key = None;
        self.pending_snapshot = None;
        self.anchor_key = None;
        self.pending_anchor = None;
        self.tools_anchor_key = None;
        self.pending_tools_anchor = None;
        self.kv_match_len = 0;
        self.junction_at = None;
        self.junction_key = None;
        self.pending_junction = None;
        self.checkpoint_key = None;
        self.pending_checkpoint = None;
        self.session_start = false;
        self.status = SequenceStatus::Waiting;
    }
}
