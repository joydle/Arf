//! DFlash 2 draft, stage 3 — the CONTEXT RING.
//!
//! The draft attends over the target's history, not its own: every committed position's five
//! tapped hiddens go through `fc` and `hidden_norm` once, and then EACH draft layer projects that
//! one context vector with its own `k_proj`/`v_proj` (k-norm + RoPE on K) into that layer's ring
//! of the last `sliding_window` positions. The context never runs through the draft's layers.
//!
//! Everything here is encoded into ONE command buffer on the island's queue, committed without
//! waiting: the queue orders it after the target record that wrote the taps and before the next
//! one that overwrites them. The matmuls are the target's own MPP Q4 path.
//!
//! CONCURRENT ENCODE (2026-09-29): the commit was a `Serial` encoder, and every layer's k/v
//! projection wrote the SAME `kraw`/`vraw` pair, so its 10 projections and 5 ring writes ran one
//! after another — 0.95 ms of GPU a speculative window on the critical path (labelled Metal
//! trace, 2026-09-26), ~3x its byte floor. Now each layer has its own K/V scratch and each
//! projection its own split-K `parts`, and the encoder is `Concurrent` with a buffer barrier only
//! at a real dependency: fc -> hidden_norm -> 10 matmuls -> 10 split-K sums -> ring writes. Same
//! kernels, same inputs, same reductions: the ring must be BYTE-identical to the serial encode
//! (gate: `dflash2_context ring-ab`). UNMEASURED on the GPU as of this commit.
//! `ARF_DFLASH_COMMIT_SERIAL=1` restores the serial encode exactly (one scratch pair, `st.parts`).
//! Group-64 draft weights (`ARF_DFLASH_G64`) always take the serial encode: `g64_encode` shares
//! ONE split-K `partials`/arrival-`counters` set across every matmul.
//!
//! A commit that does not continue where its ring stopped marks the ring invalid (a draft over a
//! context with a hole is exactly as fast as a correct one).
//!
//! ONE RING PER STREAM (C1, 2026-09-26) — a pool of `ARF_DFLASH_RINGS` (default 4) ring sets,
//! 84 MB each in f32. It was ONE ring set: a second request's prefill adopted it (`start == 0`)
//! and the first stream's draft was dead for the rest of its answer — and plain B>=2 steps never
//! committed at all, so even a stream that kept the ring fell behind `past_len` while it shared
//! the batch. The concurrent harness is total tokens over the LONGEST answer's time, and that
//! answer's solo tail ran plain at ~21 tok/s. Now: a new history takes its own ring or a free one
//! (never a live stream's), `release_stream` frees it, and every batched record's taps are
//! committed to their streams' rings (`dflash_commit_runs`) — one `fc`/norm/k-v pass over the
//! record's rows, a ring write per stream. `ARF_NO_STREAM_RINGS=1` restores the single stolen
//! ring and no plain-step commits (A/B).

use super::dflash2::Dflash2Draft;
use super::island::{
    MetalIsland, MtlBuf, Q4TileScratch, Q4ksMtl, Q4rmPhase, Q4RM_PARTS, Q4TILE_MAX_ROWS,
    Q4TILE_ROWS, Q4TILE_WIDE_MAX_ROWS, Q4TILE_WIDE_UNIT,
};
use crate::gpu::GpuContext;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBarrierScope, MTLBuffer, MTLCommandBuffer as _, MTLCommandEncoder as _,
    MTLCommandQueue as _, MTLComputeCommandEncoder, MTLDevice as _, MTLDispatchType,
    MTLResourceUsage, MTLSize,
};
use std::cell::{Cell, RefCell};
use std::sync::Arc;
use wgpu::hal::api::Metal;

/// The widest commit: one prefill window.
pub const DFLASH_CTX_ROWS: usize = Q4TILE_WIDE_MAX_ROWS;

/// The loaded draft plus everything the context commit needs, owned by the island.
pub struct Dflash2State {
    pub draft: Dflash2Draft,
    /// `[DFLASH_CTX_ROWS, hidden]` — `fc`'s output, then `hidden_norm`'s.
    pub(super) feat: MtlBuf,
    pub(super) ctxn: MtlBuf,
    /// `[DFLASH_CTX_ROWS, kv_dim]` — raw K and V, one pair PER DRAFT LAYER so the concurrent
    /// commit's projections are independent. The serial encode (`ARF_DFLASH_COMMIT_SERIAL`)
    /// reuses pair 0 layer after layer, as before.
    pub(super) kraw: Vec<MtlBuf>,
    pub(super) vraw: Vec<MtlBuf>,
    /// Split-K partials of the concurrent commit's projections, one per (layer, k|v) —
    /// `[2 * layers]`, k at `2l`, v at `2l + 1`. Empty when the split matmul is not ready.
    pub(super) kv_parts: Vec<MtlBuf>,
    /// The ring commit's encode: `true` = the serial encode (`ARF_DFLASH_COMMIT_SERIAL=1`).
    /// A `Cell` so a gate can run both arms in one process (`dflash_set_commit_serial`).
    pub(super) commit_serial: Cell<bool>,
    /// Commits encoded (serial, concurrent) — rule 7: which encode really ran.
    pub(super) commit_counts: Cell<(u64, u64)>,
    /// `[head_dim / 2]`, theta^(-2i/head_dim) computed in f64.
    pub(super) inv_freq: MtlBuf,
    /// The ring pool — one per stream with a history (see the module note).
    pub(super) rings: Vec<DflashRing>,
    pub(super) scratch: Arc<Q4TileScratch>,
    pub(super) parts: Option<MtlBuf>,
    pub(super) dims_fc: MtlBuf,
    pub(super) dims_kv: MtlBuf,
    /// The candidate range the last block used: its logits' row width.
    pub(super) block_vocab: Cell<usize>,
    /// Speculative sampling, step 2: when set, the CPU selector DRAWS each position from
    /// softmax(score / temperature) over its candidates instead of taking the best, keyed by
    /// (seed, absolute position), and records each position's distribution in `last_q`.
    /// `(temperature, seed, past_len)`; set and cleared around one draft.
    pub(super) draft_sampling: Cell<Option<(f32, u64, usize)>>,
    pub(super) last_q: RefCell<Vec<Vec<(u32, f32)>>>,
    /// The LAST GPU-select draft's sampling key: `Some((temperature, seed, past_len))` when it
    /// ran `dflash_chain_sampled` (its q is in the block's `q_id`/`q_p`), `None` when it ran the
    /// greedy `dflash_chain`. Written at every GPU-select encode, so a point-mass draft can never
    /// hand a stale q to the acceptor.
    pub(super) gpu_sampled: Cell<Option<(f32, u64, usize)>>,
    /// Tokens the DRAFT must never propose, sorted. MEASURED 2026-09-22: on plain prose the
    /// drafter emits `</think>` / `<|im_end|>` mid-generation on 4 of 27 zero-accept windows
    /// (14.8%) and on 0 of 69 windows that accepted anything — three of the four identical
    /// (a newline anchor makes it predict the thinking block ends). The target wanted a special
    /// token in NONE of those cases, so the proposal is simply wrong, and each one throws away a
    /// whole ~120 ms cycle. Worth ~1-2% end to end — small, but exact and free: a suppressed
    /// draft token can only lower acceptance, never change the text, because every draft is
    /// verified. `ARF_NO_DRAFT_SPECIAL_GUARD=1` opts out.
    pub(super) no_propose: Vec<u32>,
    /// GATE ONLY: a host copy of every tap row committed, so a reference can be computed from
    /// exactly what the GPU consumed. `None` in serving.
    pub(super) tap_log: Option<RefCell<Vec<f32>>>,
    /// Stage 4 — the block forward's buffers (`dflash2_block.rs`).
    pub(super) block: super::dflash2_block::Dflash2BlockBufs,
}

/// One stream's context ring.
pub struct DflashRing {
    /// Per draft layer, `[kv_heads, window, head_dim]` f32.
    pub(super) ring_k: Vec<MtlBuf>,
    pub(super) ring_v: Vec<MtlBuf>,
    /// Positions in the ring == the position the next commit must start at.
    pub(super) committed: Cell<usize>,
    /// The largest token id this sequence has committed (prompt and output) — the draft's
    /// candidate range covers it (`dflash_vocab_eff`).
    pub(super) max_tok: Cell<u32>,
    /// `None` = free; `Some(stream)` = holds `stream`'s history (`Some(None)`: a caller that
    /// passes no stream id — the examples and gates, which run one stream).
    pub(super) owner: Cell<Option<Option<u64>>>,
    pub(super) valid: Cell<bool>,
}

/// One run of a record's rows for one stream: `rows` tap rows from `row0`, positions from
/// `start` (`start == 0` begins a new history).
#[derive(Clone, Copy, Debug)]
pub struct CommitRun {
    pub stream: Option<u64>,
    pub row0: usize,
    pub rows: usize,
    pub start: usize,
}

/// `ARF_NO_STREAM_RINGS=1`: the single-ring behaviour before C1 (A/B only).
pub(crate) fn stream_rings_off() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var_os("ARF_NO_STREAM_RINGS").is_some())
}

impl Dflash2State {
    /// The ring holding `stream`'s history.
    pub(super) fn ring_of(&self, stream: Option<u64>) -> Option<&DflashRing> {
        self.rings.iter().find(|r| r.owner.get() == Some(stream))
    }

    /// A ring for a history of `stream` that begins now: its own, else a free one — never
    /// another live stream's (a newcomer that took it would silence that stream's draft for the
    /// rest of its answer). `None` = the pool is full: this stream simply does not draft. With a
    /// single ring (`ARF_NO_STREAM_RINGS`) the newcomer takes it, as before C1.
    pub(super) fn ring_claim(&self, stream: Option<u64>) -> Option<&DflashRing> {
        let r = self
            .ring_of(stream)
            .or_else(|| self.rings.iter().find(|r| r.owner.get().is_none()))
            .or_else(|| (self.rings.len() == 1).then(|| &self.rings[0]))?;
        r.owner.set(Some(stream));
        r.committed.set(0);
        r.max_tok.set(0);
        r.valid.set(true);
        Some(r)
    }

    /// The ring a gate reads: the first one holding a history.
    fn ring_for_gate(&self) -> Option<&DflashRing> {
        self.rings.iter().find(|r| r.owner.get().is_some())
    }
}

fn pad_rows(rows: usize) -> usize {
    if rows <= Q4TILE_ROWS {
        Q4TILE_ROWS
    } else if rows <= Q4TILE_MAX_ROWS {
        Q4TILE_MAX_ROWS
    } else {
        rows.next_multiple_of(Q4TILE_WIDE_UNIT)
    }
}

impl MetalIsland {
    /// Can the draft run on this OS? Its matmuls are the MPP Q4 units, which need the tensor ops of
    /// Metal 4 (macOS 26); on an older OS they do not compile and the model serves without it.
    pub fn dflash_supported(&self) -> bool {
        self.q4tile_unit_compiled(Q4TILE_ROWS)
            && self.q4tile_unit_compiled(Q4TILE_MAX_ROWS)
            && self.q4tile_wide_compiled()
    }

    /// Take ownership of a loaded draft: enable the taps, compile the draft's kernels, allocate
    /// the rings and every scratch the context commit uses — all HERE, while nothing is in
    /// flight, so a commit never allocates (the shared MPP scratch is regrown to `fc`'s 25600-wide
    /// input now, not under a running record).
    pub fn dflash_attach(
        &mut self,
        ctx: &GpuContext,
        draft: Dflash2Draft,
        log_taps: bool,
    ) -> Result<(), String> {
        let c = draft.cfg.clone();
        let (h, kv_dim) = (c.hidden, c.kv_heads * c.head_dim);
        let fc_in = c.target_layer_ids.len() * h;
        if !self.dflash_supported() {
            return Err(
                "dflash: the MPP Q4 matmul units are not compiled (needs macOS 26 tensor ops)"
                    .into(),
            );
        }
        self.dflash_enable_taps(ctx, &c.target_layer_ids, h, DFLASH_CTX_ROWS)?;
        let src = include_str!("../../shaders/metal/dflash2_msl.metal");
        for k in [
            "dflash_rmsnorm",
            "dflash_rmsnorm_tg",
            "dflash_ctx_commit",
            "dflash_conv",
            "dflash_head_norm_rope",
            "dflash_attn_fused",
            "dflash_attn_scores",
            "dflash_attn_softmax",
            "dflash_attn_mix",
            "dflash_swiglu",
            // the GPU candidate selector (ARF_DFLASH_GPU_SELECT): top-16, edge table, chain
            "dflash_topk",
            "dflash_edges",
            "dflash_chain",
            // the special-token tail's top-1 merge (2026-09-28, `DflashTail`)
            "dflash_tail_argmax",
        ] {
            self.compile_msl(ctx, k, src, k)?;
        }
        // optional: a compile failure leaves the per-(row, head) attention kernel in use
        for k in ["dflash_attn_gqa", "dflash_attn_gqa_combine"] {
            if let Err(e) = self.compile_msl(ctx, k, src, k) {
                eprintln!("[dflash] {k} not compiled ({e}) — the per-(row, head) attention serves");
            }
        }
        // optional (2026-09-26): the SAMPLED chain. Without it a sampled request's sampled draft
        // takes the CPU selector (`dflash_select`), exactly as before the kernel existed.
        if let Err(e) = self.compile_msl(ctx, "dflash_chain_sampled", src, "dflash_chain_sampled") {
            eprintln!("[dflash] dflash_chain_sampled not compiled ({e}) — sampled drafts use the CPU selector");
        }
        let scratch = self
            .q4tile_scratch_for(ctx, fc_in)
            .ok_or("dflash: mpp scratch")?;
        let parts = if self.q4rm_split_ready(Q4TILE_ROWS) && self.q4rm_split_ready(Q4TILE_MAX_ROWS)
        {
            Some(
                self.q4rm_parts_for(ctx, h.max(c.intermediate))
                    .ok_or("dflash: parts")?,
            )
        } else {
            None
        };
        let dims_fc = self.q4rm_dims_for(ctx, h, fc_in).ok_or("dflash: fc dims")?;
        let dims_kv = self
            .q4rm_dims_for(ctx, kv_dim, h)
            .ok_or("dflash: kv dims")?;
        let block_dims = super::dflash2_block::Dflash2BlockBufs::dims(self, ctx, &c)?;
        let guard = unsafe { ctx.device.as_hal::<Metal>() }.ok_or("not metal")?;
        let dev = guard.raw_device();
        let mk = |n: usize| {
            dev.newBufferWithLength_options(
                n.max(16),
                objc2_metal::MTLResourceOptions::StorageModeShared,
            )
            .map(MtlBuf)
            .ok_or_else(|| "dflash: alloc".to_string())
        };
        let inv_freq = mk(c.head_dim / 2 * 4)?;
        unsafe {
            let p = inv_freq.0.contents().as_ptr() as *mut f32;
            for i in 0..c.head_dim / 2 {
                *p.add(i) = (c.rope_theta as f64).powf(-2.0 * i as f64 / c.head_dim as f64) as f32;
            }
        }
        let ring = c.kv_heads * c.sliding_window * c.head_dim * 4;
        let n_rings = if stream_rings_off() {
            1
        } else {
            std::env::var("ARF_DFLASH_RINGS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(4)
                .max(1)
        };
        let mut rings = Vec::with_capacity(n_rings);
        for _ in 0..n_rings {
            let (mut ring_k, mut ring_v) = (vec![], vec![]);
            for _ in 0..c.layers {
                ring_k.push(mk(ring)?);
                ring_v.push(mk(ring)?);
            }
            rings.push(DflashRing {
                ring_k,
                ring_v,
                committed: Cell::new(0),
                max_tok: Cell::new(0),
                owner: Cell::new(None),
                valid: Cell::new(false),
            });
        }
        eprintln!(
            "[dflash] {n_rings} context ring(s), {:.0} MB each",
            (2 * c.layers * ring) as f64 / 1e6
        );
        let block = super::dflash2_block::Dflash2BlockBufs::alloc(&c, block_dims, &mk)?;
        let st = Dflash2State {
            block,
            feat: mk(DFLASH_CTX_ROWS * h * 4)?,
            ctxn: mk(DFLASH_CTX_ROWS * h * 4)?,
            kraw: (0..c.layers)
                .map(|_| mk(DFLASH_CTX_ROWS * kv_dim * 4))
                .collect::<Result<_, _>>()?,
            vraw: (0..c.layers)
                .map(|_| mk(DFLASH_CTX_ROWS * kv_dim * 4))
                .collect::<Result<_, _>>()?,
            // q4rm_parts_for's size at n = kv_dim
            kv_parts: if parts.is_some() {
                (0..2 * c.layers)
                    .map(|_| mk(Q4RM_PARTS * Q4TILE_MAX_ROWS * kv_dim * 4))
                    .collect::<Result<_, _>>()?
            } else {
                vec![]
            },
            commit_serial: Cell::new(std::env::var_os("ARF_DFLASH_COMMIT_SERIAL").is_some()),
            commit_counts: Cell::new((0, 0)),
            inv_freq,
            rings,
            scratch,
            parts,
            dims_fc,
            dims_kv,
            block_vocab: Cell::new(0),
            draft_sampling: Cell::new(None),
            last_q: RefCell::new(Vec::new()),
            gpu_sampled: Cell::new(None),
            tap_log: log_taps.then(|| RefCell::new(vec![])),
            no_propose: Self::draft_suppressed_tokens(),
            draft,
        };
        drop(guard);
        *self.dflash_state.borrow_mut() = Some(st);
        Ok(())
    }

    /// Tokens the draft must never propose (see `Dflash2State::no_propose`). Defaults to the
    /// Qwen3.8 control tokens observed causing the miss — `</think>` **248069** and `<|im_end|>`
    /// **248046** — and is overridable with `ARF_DRAFT_SUPPRESS=id,id,...` so another
    /// checkpoint's ids can be set without a rebuild. `ARF_NO_DRAFT_SPECIAL_GUARD=1` disables it.
    ///
    /// ⚠️ The ids are NOT guessable and were wrong on the first attempt: 248067 is
    /// `</tool_response>` and 248061 is `<|fim_middle|>` in this vocab. They were read out of the
    /// GGUF with `cargo run -p arf-core --example dump_vocab` and matched BY NAME. Any change
    /// here must be re-checked the same way, never by adjacency to another id.
    ///
    /// These are TARGET-vocab ids: the drafter has no lm_head of its own and scores the target's
    /// logits, so the ids are the target's. They are verified at attach against the log, not
    /// assumed — `dflash2_load` prints the decoded form of each.
    fn draft_suppressed_tokens() -> Vec<u32> {
        // ⛔ DEFAULT OFF — MEASURED 2026-09-22 and it buys NOTHING. The guard does exactly what
        // it says (special tokens at position 0: 4 -> 0, text identical 4/4) and the acceptance
        // did not move: mean accepted 2.00 -> 2.01, zero-accept 28.1% -> 36.0% on 86/96 windows.
        // Suppressing the wrong token does not make the draft right — it proposes the NEXT wrong
        // token and the window still dies at position 0. Opt IN with ARF_DRAFT_SPECIAL_GUARD=1;
        // kept because it is exact and is the instrument that priced the tic at zero.
        if std::env::var_os("ARF_DRAFT_SPECIAL_GUARD").is_none() {
            return vec![];
        }
        let mut v: Vec<u32> = match std::env::var("ARF_DRAFT_SUPPRESS") {
            Ok(s) => s.split(',').filter_map(|t| t.trim().parse().ok()).collect(),
            Err(_) => vec![248046, 248069],
        };
        v.sort_unstable();
        v.dedup();
        v
    }

    pub fn dflash_attached(&self) -> bool {
        self.dflash_state.borrow().is_some()
    }

    /// Positions in the ring, and whether it is an unbroken history of one stream. GATES: the
    /// first ring holding a history (`(0, false)` when none does).
    pub fn dflash_context_len(&self) -> Option<(usize, bool)> {
        let st = self.dflash_state.borrow();
        st.as_ref().map(|s| {
            s.ring_for_gate()
                .map_or((0, false), |r| (r.committed.get(), r.valid.get()))
        })
    }

    /// A draft for `stream`'s pending token at `past_len` would run: its ring is valid and holds
    /// exactly positions `..past_len` (the check `dflash_draft_block_cut` makes).
    pub fn dflash_ready(&self, stream: Option<u64>, past_len: usize) -> bool {
        self.dflash_state.borrow().as_ref().is_some_and(|st| {
            st.ring_of(stream)
                .is_some_and(|r| r.valid.get() && r.committed.get() == past_len)
        })
    }

    /// `stream` has finished: its ring is free for the next history.
    pub fn dflash_release_stream(&self, stream: u64) {
        if let Some(st) = self.dflash_state.borrow().as_ref() {
            if let Some(r) = st.ring_of(Some(stream)) {
                r.owner.set(None);
                r.valid.set(false);
                r.committed.set(0);
            }
        }
    }

    /// Tokens this stream has committed — widens the draft's candidate range to cover them.
    pub fn dflash_note_tokens(&self, stream: Option<u64>, tokens: &[u32]) {
        if let Some(st) = self.dflash_state.borrow().as_ref() {
            if let Some(ring) = st.ring_of(stream) {
                // control tokens (the chat template's, one block at the top of the vocabulary) do
                // not count: the template alone would pin the range at its maximum (measured: the
                // first English prompt set it to 248068). The draft then cannot propose them — at
                // most one position a response (`</think>`), which the target still emits.
                //
                // NOR does the word right after one (2026-09-26): the template's role header
                // `<|im_start|>assistant` commits 'assistant' = 74,455 — ordinary vocabulary — and
                // pinned EVERY chat request at 98,304 rows (125 of 125 logged range jumps naming
                // 74455; ~0.26-0.30 ms of draft lm_head a cycle). A role word the answer really
                // uses is committed again later, where no control token precedes it.
                // ARF_DFLASH_NOTE_ROLE=1 restores the old accounting (A/B).
                //
                // 2026-09-28 — "at most one position a response" above was not free: the
                // teacher-forced runs (measured 2026-09-28) show windows dying at exactly
                // those positions where another engine's full-vocabulary draft proposes `</think>` /
                // `<|im_end|>`; the full-vocabulary arm recovered 3 of 120 windows on p2-p5. The
                // exclusion STAYS (it still keeps the range small); the control block is now
                // scored separately every block — the special-token tail (`DflashTail`,
                // dflash2_block.rs), a few hundred lm_head rows. ARF_NO_DFLASH_SPECIAL_TAIL=1 =
                // the range alone, the behaviour this comment first described.
                let floor = st.draft.cfg.special_floor;
                let keep_role = std::env::var_os("ARF_DFLASH_NOTE_ROLE").is_some();
                let m = tokens
                    .iter()
                    .enumerate()
                    .filter(|&(i, &t)| t < floor && (keep_role || i == 0 || tokens[i - 1] < floor))
                    .map(|(_, &t)| t)
                    .max()
                    .unwrap_or(0);
                if m > ring.max_tok.get() {
                    ring.max_tok.set(m);
                }
            }
        }
    }

    /// Commit rows `0..rows` of the most recent record's taps as positions `start..start+rows` of
    /// `stream`'s context. Call after the record that wrote them and before the next record:
    /// a prefill window commits all its rows, a verify window its ACCEPTED prefix.
    /// `start == 0` begins a new history (in the stream's own ring or a free one). No-op without
    /// a draft.
    pub fn dflash_commit_context(
        &self,
        stream: Option<u64>,
        rows: usize,
        start: usize,
    ) -> Result<(), String> {
        self.dflash_commit_runs(&[CommitRun {
            stream,
            row0: 0,
            rows,
            start,
        }])
    }

    /// Commit the most recent record's tap rows to their streams' rings, one [`CommitRun`] per
    /// stream: ONE `fc` / `hidden_norm` / k-v projection pass over the record's rows, then a ring
    /// write per run. A run whose stream holds no ring (the pool is full, or its history began
    /// elsewhere) is skipped; one that does not continue its ring invalidates that ring and is
    /// reported, and the other runs still commit.
    pub fn dflash_commit_runs(&self, runs: &[CommitRun]) -> Result<(), String> {
        let state = self.dflash_state.borrow();
        let Some(st) = state.as_ref() else {
            return Ok(());
        };
        let (tap_rows, taps) = self
            .dflash_taps_buf()
            .ok_or("dflash: taps are not enabled")?;
        let c = &st.draft.cfg;
        let fc_in = c.target_layer_ids.len() * c.hidden;
        let mut live: Vec<(&DflashRing, CommitRun)> = Vec::with_capacity(runs.len());
        let mut err = None;
        for &run in runs {
            let CommitRun {
                stream,
                row0,
                rows,
                start,
            } = run;
            let ring = if start == 0 {
                let r = st.ring_claim(stream);
                if r.is_some() {
                    if let Some(log) = &st.tap_log {
                        log.borrow_mut().clear();
                    }
                }
                r
            } else {
                st.ring_of(stream)
            };
            let Some(ring) = ring else {
                continue; // no ring for this stream — it does not draft
            };
            if !ring.valid.get() {
                continue;
            }
            if start != ring.committed.get()
                || rows == 0
                || row0 + rows > tap_rows
                || rows > DFLASH_CTX_ROWS
            {
                ring.valid.set(false);
                err = Some(format!(
                    "dflash context: commit of {rows} rows (record rows {row0}..) at {start} does \
                     not continue a ring of {} (record had {tap_rows} rows) — ring invalidated",
                    ring.committed.get()
                ));
                continue;
            }
            // advanced now, so a later run of the same stream in this call must continue it
            ring.committed.set(start + rows);
            live.push((ring, run));
        }
        if live.is_empty() {
            return err.map_or(Ok(()), Err);
        }
        let rows = live.iter().map(|(_, r)| r.row0 + r.rows).max().unwrap_or(0);
        if let Some(log) = &st.tap_log {
            // Gate only. A non-final prefill window skips the lm_head and may not have been waited
            // on, so fence the queue before reading what the record wrote.
            let fence = self
                .queue
                .commandBuffer()
                .ok_or("dflash: no command buffer")?;
            super::island::label_cb(&fence, "arf-ring-fence");
            fence.commit();
            fence.waitUntilCompleted();
            let v = unsafe {
                std::slice::from_raw_parts(taps.0.contents().as_ptr() as *const f32, rows * fc_in)
            };
            for (_, r) in &live {
                log.borrow_mut()
                    .extend_from_slice(&v[r.row0 * fc_in..(r.row0 + r.rows) * fc_in]);
            }
        }
        // The rings were advanced above; an encoding failure leaves them unwritten, so it
        // invalidates every one of them rather than leave a hole the draft cannot see.
        if let Err(e) = self.dflash_commit_encode(st, &taps, rows, &live) {
            for (ring, _) in &live {
                ring.valid.set(false);
            }
            return Err(e);
        }
        err.map_or(Ok(()), Err)
    }

    fn dflash_commit_encode(
        &self,
        st: &Dflash2State,
        taps: &MtlBuf,
        rows: usize,
        live: &[(&DflashRing, CommitRun)],
    ) -> Result<(), String> {
        let c = &st.draft.cfg;
        let (h, kv_dim) = (c.hidden, c.kv_heads * c.head_dim);
        let fc_in = c.target_layer_ids.len() * h;
        let padded = pad_rows(rows);
        // `g64_encode` writes ONE shared split-K `partials` + arrival `counters` set for every
        // matmul, so group-64 projections must never overlap: they take the serial encode.
        let g64 = st.draft.fc.g64.is_some()
            || st
                .draft
                .layers
                .iter()
                .any(|ly| ly.k_proj.g64.is_some() || ly.v_proj.g64.is_some());
        let serial = st.commit_serial.get() || g64;
        {
            // rule 7: say which encode ran, the first time each one does
            let (ns, nc) = st.commit_counts.get();
            let (ns, nc) = if serial { (ns + 1, nc) } else { (ns, nc + 1) };
            st.commit_counts.set((ns, nc));
            if (serial && ns == 1) || (!serial && nc == 1) {
                eprintln!(
                    "[dflash] ring commit: {} encode{}",
                    if serial { "SERIAL" } else { "CONCURRENT" },
                    if serial && g64 {
                        " (group-64 draft weights share one split-K scratch)"
                    } else if serial {
                        " (ARF_DFLASH_COMMIT_SERIAL)"
                    } else {
                        " (per-layer K/V scratch, barriers at the real dependencies)"
                    }
                );
            }
        }
        let cmd = self
            .queue
            .commandBuffer()
            .ok_or("dflash: no command buffer")?;
        super::island::label_cb(&cmd, "arf-ring-commit");
        let enc = cmd
            .computeCommandEncoderWithDispatchType(if serial {
                MTLDispatchType::Serial
            } else {
                MTLDispatchType::Concurrent
            })
            .ok_or("dflash: no encoder")?;
        // A buffer barrier on the concurrent encoder; nothing on the serial one, which already
        // orders every dispatch after the one before it.
        let fence = || {
            if !serial {
                enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
            }
        };
        type Hook<'a> =
            dyn Fn(&[&ProtocolObject<dyn MTLBuffer>], &[&ProtocolObject<dyn MTLBuffer>]) + 'a;
        // The matmul helpers' dependency hooks: `full` honours every stage boundary a helper
        // declares; `none` is for a matmul whose operand is already prepared and whose output
        // nothing reads until an explicit `fence`.
        let full: &Hook = &|_, _| fence();
        let none: &Hook = &|_, _| {};
        let matmul = |w: &Q4ksMtl,
                      dims: &MtlBuf,
                      n: usize,
                      k: usize,
                      a: &MtlBuf,
                      out: &MtlBuf,
                      reuse: bool,
                      hook: &Hook,
                      parts: Option<&ProtocolObject<dyn MTLBuffer>>,
                      phase: Q4rmPhase|
         -> Result<(), String> {
            unsafe {
                // group-64 draft weights (ARF_DFLASH_G64): up to 32 lanes of 8 rows in one
                // dispatch. Serial encode only (above) — the whole matmul in one phase.
                if let Some(g) = w.g64.as_ref() {
                    if phase == Q4rmPhase::Sum {
                        return Ok(());
                    }
                    return self.g64_encode(&enc, padded, g, n, &a.0, &out.0, hook, false, None);
                }
                if padded > Q4TILE_MAX_ROWS {
                    // no split-K on the wide path: the matmul writes `out` itself
                    if phase == Q4rmPhase::Sum {
                        return Ok(());
                    }
                    self.q4rm_encode_wide(
                        &enc,
                        padded,
                        w,
                        &dims.0,
                        n,
                        k,
                        &st.scratch,
                        &a.0,
                        &out.0,
                        hook,
                        reuse,
                    )
                } else {
                    self.q4rm_encode_phased(
                        &enc,
                        padded,
                        w,
                        &dims.0,
                        n,
                        k,
                        &st.scratch,
                        &a.0,
                        &out.0,
                        hook,
                        reuse,
                        parts,
                        phase,
                    )
                }
            }
        };
        let pso = |k: &str| {
            self.pipelines
                .get(k)
                .map(|p| p.pso.clone())
                .ok_or_else(|| format!("pso {k}"))
        };
        let bind = |bufs: &[&ProtocolObject<dyn MTLBuffer>]| {
            for (i, b) in bufs.iter().enumerate() {
                enc.useResource_usage(
                    ProtocolObject::from_ref(*b),
                    MTLResourceUsage::Read | MTLResourceUsage::Write,
                );
                unsafe { enc.setBuffer_offset_atIndex(Some(*b), 0, i) };
            }
        };
        let bytes = |v: &[u32], idx: usize| unsafe {
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(v.as_ptr() as *mut std::ffi::c_void).unwrap(),
                std::mem::size_of_val(v),
                idx,
            );
        };
        let st_parts = st.parts.as_ref().map(|b| &*b.0);

        matmul(
            &st.draft.fc,
            &st.dims_fc,
            h,
            fc_in,
            taps,
            &st.feat,
            false,
            full,
            st_parts,
            Q4rmPhase::Both,
        )?;
        fence(); // hidden_norm reads `feat`
        let p_norm = pso("dflash_rmsnorm_tg")?;
        enc.setComputePipelineState(&p_norm);
        bind(&[&st.feat.0, &st.draft.hidden_norm.0, &st.ctxn.0]);
        bytes(&[padded as u32, h as u32, c.rms_eps.to_bits(), 0], 3);
        // One 256-thread threadgroup per row (`dflash_rmsnorm_tg`, 2026-09-23). It was one
        // threadgroup holding one THREAD per row, each walking the whole hidden width serially.
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: padded,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        let p_commit = pso("dflash_ctx_commit")?;
        // one ring write per run: its rows of layer `l`'s raw K/V (a buffer offset — the kernel
        // indexes from row 0) into its own ring at its own positions
        let commit =
            |l: usize, kraw: &MtlBuf, vraw: &MtlBuf, ring: &DflashRing, run: &CommitRun| {
                let ly = &st.draft.layers[l];
                enc.setComputePipelineState(&p_commit);
                bind(&[
                    &kraw.0,
                    &vraw.0,
                    &ly.k_norm.0,
                    &st.inv_freq.0,
                    &ring.ring_k[l].0,
                    &ring.ring_v[l].0,
                ]);
                let off = run.row0 * kv_dim * 4;
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&kraw.0), off, 0);
                    enc.setBuffer_offset_atIndex(Some(&vraw.0), off, 1);
                }
                bytes(
                    &[
                        run.rows as u32,
                        run.start as u32,
                        c.sliding_window as u32,
                        c.kv_heads as u32,
                        c.head_dim as u32,
                        c.rms_eps.to_bits(),
                        0,
                        0,
                    ],
                    6,
                );
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: 1,
                        height: run.rows,
                        depth: 1,
                    },
                    MTLSize {
                        width: c.kv_heads,
                        height: 1,
                        depth: 1,
                    },
                );
            };
        if serial {
            // THE ENCODE BEFORE 2026-09-29, exactly: one K/V scratch pair reused layer after
            // layer, every projection's split-K through `st.parts`.
            let (kraw, vraw) = (&st.kraw[0], &st.vraw[0]);
            for (l, ly) in st.draft.layers.iter().enumerate() {
                // All ten projections read the SAME `ctxn` at the same width: narrowed once.
                matmul(
                    &ly.k_proj,
                    &st.dims_kv,
                    kv_dim,
                    h,
                    &st.ctxn,
                    kraw,
                    l > 0,
                    none,
                    st_parts,
                    Q4rmPhase::Both,
                )?;
                matmul(
                    &ly.v_proj,
                    &st.dims_kv,
                    kv_dim,
                    h,
                    &st.ctxn,
                    vraw,
                    true,
                    none,
                    st_parts,
                    Q4rmPhase::Both,
                )?;
                for (ring, run) in live {
                    commit(l, kraw, vraw, ring, run);
                }
            }
        } else {
            // All ten projections read the SAME `ctxn`: the FIRST narrows/prepares it (`full`
            // hook: its stages are ordered, and the barrier before its matmul also orders every
            // later matmul after the prepared operand); the other nine only read that scratch.
            // Each writes its own K/V scratch and its own split-K parts — no shared written
            // buffer — so all ten matmuls share one barrier interval, then all ten sums.
            for phase in [Q4rmPhase::Mm, Q4rmPhase::Sum] {
                if phase == Q4rmPhase::Sum {
                    fence(); // every sum reads its own matmul's parts
                }
                for (l, ly) in st.draft.layers.iter().enumerate() {
                    for (j, (w, out)) in [(&ly.k_proj, &st.kraw[l]), (&ly.v_proj, &st.vraw[l])]
                        .into_iter()
                        .enumerate()
                    {
                        let first = l == 0 && j == 0;
                        matmul(
                            w,
                            &st.dims_kv,
                            kv_dim,
                            h,
                            &st.ctxn,
                            out,
                            !first,
                            if first && phase == Q4rmPhase::Mm {
                                full
                            } else {
                                none
                            },
                            st.kv_parts.get(2 * l + j).map(|b| &*b.0),
                            phase,
                        )?;
                    }
                }
            }
            fence(); // the ring writes read every layer's K/V
            for l in 0..st.draft.layers.len() {
                // Two runs of ONE stream in a call write the same ring (later positions); the
                // serial encode ordered them, so a barrier keeps that order here (a wrapped ring
                // could map both to one slot).
                let mut seen: Vec<*const DflashRing> = Vec::with_capacity(live.len());
                for (ring, run) in live {
                    let id = *ring as *const DflashRing;
                    if seen.contains(&id) {
                        fence();
                        seen.clear();
                    }
                    seen.push(id);
                    commit(l, &st.kraw[l], &st.vraw[l], ring, run);
                }
            }
        }
        enc.endEncoding();
        super::island::seam_tag(&cmd, "ring-commit");
        cmd.commit();
        Ok(())
    }

    /// GATES: set the ring commit's encode (`true` = serial, as `ARF_DFLASH_COMMIT_SERIAL=1`),
    /// returning the previous setting; and how many commits each encode has run (serial,
    /// concurrent) — the proof an arm ran the encode it names.
    pub fn dflash_set_commit_serial(&self, serial: bool) -> Option<bool> {
        let state = self.dflash_state.borrow();
        let st = state.as_ref()?;
        Some(st.commit_serial.replace(serial))
    }

    pub fn dflash_commit_counts(&self) -> Option<(u64, u64)> {
        self.dflash_state
            .borrow()
            .as_ref()
            .map(|s| s.commit_counts.get())
    }

    /// GATES: wait for the queue, then fill every ring of every stream with a NaN pattern, so a
    /// commit that wrote nothing cannot pass a comparison against an earlier one.
    pub fn dflash_poison_rings(&self) {
        self.dflash_fence();
        if let Some(st) = self.dflash_state.borrow().as_ref() {
            for r in &st.rings {
                for b in r.ring_k.iter().chain(&r.ring_v) {
                    let n = b.0.length() / 4;
                    let p = b.0.contents().as_ptr() as *mut u32;
                    for i in 0..n {
                        unsafe { *p.add(i) = 0x7fc0_dead };
                    }
                }
            }
        }
    }

    /// GATES: wait for the queue, then the raw bits of `stream`'s ring, every layer, K then V.
    pub fn dflash_ring_bits(&self, stream: Option<u64>) -> Option<Vec<u32>> {
        self.dflash_fence();
        let state = self.dflash_state.borrow();
        let ring = state.as_ref()?.ring_of(stream)?;
        let mut v = vec![];
        for b in ring.ring_k.iter().chain(&ring.ring_v) {
            v.extend_from_slice(unsafe {
                std::slice::from_raw_parts(b.0.contents().as_ptr() as *const u32, b.0.length() / 4)
            });
        }
        Some(v)
    }

    /// HOST READ, gates only: wait for the queue, then layer `l`'s rings of the first ring holding
    /// a history
    /// (`[kv_heads, window, head_dim]` each) and the tap log (`[committed, taps*hidden]`).
    pub fn dflash_read_ring(&self, l: usize) -> Option<(Vec<f32>, Vec<f32>)> {
        let state = self.dflash_state.borrow();
        let st = state.as_ref()?;
        let cmd = self.queue.commandBuffer()?;
        cmd.commit();
        cmd.waitUntilCompleted();
        let c = &st.draft.cfg;
        let n = c.kv_heads * c.sliding_window * c.head_dim;
        let rd = |b: &MtlBuf| unsafe {
            std::slice::from_raw_parts(b.0.contents().as_ptr() as *const f32, n).to_vec()
        };
        let ring = st.ring_for_gate()?;
        Some((rd(ring.ring_k.get(l)?), rd(ring.ring_v.get(l)?)))
    }

    /// Wait until everything committed to the island's queue so far has run.
    pub fn dflash_fence(&self) {
        if let Some(cmd) = self.queue.commandBuffer() {
            cmd.commit();
            cmd.waitUntilCompleted();
        }
    }

    pub fn dflash_tap_log(&self) -> Option<Vec<f32>> {
        let state = self.dflash_state.borrow();
        state.as_ref()?.tap_log.as_ref().map(|l| l.borrow().clone())
    }
}
