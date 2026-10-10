//! Continuous-batching scheduler.
//!
//! Each [`Scheduler::schedule`] call returns a [`BatchPlan`]: the sequences to
//! run this step, with their query offsets and block tables. The caller turns
//! that into tensors and runs the model, then commits the result to append
//! sampled tokens and retire finished sequences. Two commit paths share one
//! `advance_and_retire` helper: the CPU engine calls [`Scheduler::commit`] with
//! logits (it samples here), while the GPU server samples on its backend and
//! calls [`Scheduler::commit_tokens`] with the already-sampled ids.
//!
//! Scheduling is greedy: continue running sequences first (preempting the
//! newest when the KV pool is exhausted), then admit waiting requests while the
//! batch-size and prefill-token budgets allow.

use std::collections::{HashSet, VecDeque};

use crate::config::EngineConfig;
use crate::error::Result;
use crate::scheduler::block_manager::BlockManager;
use crate::scheduler::request::{FinishReason, Request, Sequence, SequenceStatus};
use crate::tensor::Tensor;

/// Per-sequence entry in a scheduled batch.
#[derive(Debug, Clone)]
pub struct SeqPlan {
    pub id: u64,
    /// Offset of this sequence's query tokens in the flattened batch.
    pub q_start: usize,
    /// Query tokens this step (prefill: pending prompt; decode: 1).
    pub q_len: usize,
    /// Tokens already cached (the RoPE/causal offset).
    pub past_len: usize,
    /// Physical blocks backing this sequence (snapshot).
    pub block_table: Vec<u32>,
    /// Vision: image soft-tokens to splice in THIS chunk, as `(flat_row, embedding[hidden])`.
    /// `flat_row` is the position in the flattened batch; the embedding overrides the token
    /// embed at that row. Empty for text seqs / chunks with no image positions.
    pub image_rows: Vec<(usize, Vec<f32>)>,
    /// Qwen3.8 vision: this step's `q_len` rows' (t, h, w) rope positions, from the
    /// sequence's `ImagePrompt::mrope` layout. `None` for every sequence without one (all text,
    /// and Gemma-3 images), whose rows rotate at their plain `positions` exactly as before.
    pub mrope: Option<Vec<[u32; 3]>>,
    /// `image_rows` attend causally (`ImagePrompt::causal`, Qwen3-Omni audio): no bidirectional
    /// span even without an M-RoPE layout. `false` for text and for Gemma-3 images.
    pub image_causal: bool,
    /// STATE SNAPSHOTS: restore the recurrent state stored under this key into the sequence
    /// BEFORE running this step (its prefix was a cache hit down to that boundary).
    pub restore_key: Option<u64>,
    /// STATE SNAPSHOTS: this chunk ends exactly on the block boundary the key names — save the
    /// sequence's recurrent state under it AFTER running the step, then tell the scheduler
    /// (`Scheduler::note_snapshot`).
    pub snapshot_key: Option<u64>,
    /// STATE SNAPSHOTS: `snapshot_key` is an ANCHOR — it sits at the end of the request's shared
    /// "system + tools" prefix (`Request::prefix_anchor`), where a NEW session of the same agent
    /// will match, rather than one window before this prompt's end, where the next turn of THIS
    /// conversation will. The backend keeps anchors in their own pool
    /// (`BatchedBackend::state_save_anchor`) so a long session's per-turn snapshots cannot
    /// evict them. Always false when `snapshot_key` is `None`.
    pub snapshot_anchor: bool,
    /// STATE SNAPSHOTS: this ANCHOR (`snapshot_anchor` is true too — same pool) is the TOOLS
    /// ANCHOR (2026-09-27): the end of the request's tools block (`Request::tools_anchor`), where
    /// a new session of the same agent whose SYSTEM text differs (Claude Code in another working
    /// directory) still matches. A prompt carrying both anchors takes two anchor snapshots in one
    /// prefill: the tools anchor first, then the system anchor. Always false when
    /// `snapshot_anchor` is.
    pub snapshot_tools_anchor: bool,
    /// STATE SNAPSHOTS: `snapshot_key` is a JUNCTION (2026-09-26) — the window-aligned point
    /// where this prompt LEAVES cached history (`Sequence::junction_at`): an earlier prompt's KV
    /// runs on past the snapshot this one resumed from, so a later prompt sharing that longer
    /// prefix (a new agent session with the same project context) can resume here instead of at
    /// the anchor. The backend keeps junctions in a pool of their own
    /// (`BatchedBackend::state_save_junction`). Never true together with `snapshot_anchor`;
    /// always false when `snapshot_key` is `None`.
    pub snapshot_junction: bool,
    /// STATE SNAPSHOTS: `snapshot_key` is the END-OF-PROMPT snapshot of a conversation's FIRST
    /// turn (`Sequence::session_start`, 2026-09-26). It is filed in the backend's JUNCTION pool
    /// (`BatchedBackend::state_save_junction`) instead of the turn-end pool: an agent session
    /// runs 6-7 turns, each saving a turn-end snapshot, so in the 4-slot turn-end pool the
    /// first turn's snapshot is gone by the time the next session starts, and a new session
    /// whose first request is the same (or the same up to its last window) resumes only at the
    /// anchor (`Scheduler::is_session_start` has the measurement and its caveat). Never true together with `snapshot_anchor` or `snapshot_junction`; always false
    /// when `snapshot_key` is `None`. The scheduler's bookkeeping is the turn-end snapshot's
    /// (held and published as `pending_snapshot`); only the pool differs.
    pub snapshot_session_start: bool,
    /// ROLLING CHECKPOINT (2026-10-07): `snapshot_key` is a resume point inside a long prompt
    /// (`checkpoint_tokens`), filed in the backend's checkpoint pool
    /// (`BatchedBackend::state_save_checkpoint`). Never true together with another snapshot flag.
    pub snapshot_checkpoint: bool,
    /// This step's `q_len` finishes the sequence's pending tokens (a decode step, or the chunk that
    /// ends its prompt). A prefill chunk that does NOT is never sampled (`commit`), so a backend
    /// may skip the prediction for it (2026-09-26: the 1-row lm_head pass every 512-token chunk
    /// paid, ~0.95 s of TTFT at 7,433 tokens).
    pub completes_prompt: bool,
}

/// A scheduled step's plan: flattened tokens, positions, and per-sequence info.
#[derive(Debug, Clone)]
pub struct BatchPlan {
    pub input_ids: Vec<u32>,
    pub positions: Vec<u32>,
    pub seqs: Vec<SeqPlan>,
}

/// One token of output for a request.
#[derive(Debug, Clone)]
pub struct RequestOutput {
    pub id: u64,
    pub token: u32,
    pub finished: bool,
    pub finish_reason: Option<FinishReason>,
}

/// The scheduler: waiting queue, running set, and the block manager.
#[derive(Debug)]
pub struct Scheduler {
    waiting: VecDeque<Sequence>,
    running: Vec<Sequence>,
    blocks: BlockManager,
    cfg: EngineConfig,
    /// Per-seq (running index, q_len) scheduled by the LAST `build_plan`, in
    /// plan order. Set by `schedule()`, consumed+cleared by commit. This is how
    /// commit advances `num_computed` by what was actually processed.
    scheduled: Vec<(usize, usize)>,
    /// One-shot (the next `build_plan` takes it): of the sequences in decode, plan ONLY this one
    /// (`set_decode_only`). The others keep their pending token and run a later step.
    decode_only: Option<u64>,
    /// One-shot (the next `build_plan` takes it): the most prompt tokens the next plan reads
    /// while a sequence decodes beside them (`set_prefill_cap`). `Some(0)` plans decode only.
    prefill_cap: Option<usize>,
    /// ABANDONED READS (`set_background`): running sequences nobody is waiting for. They are
    /// planned only in a step no other sequence has work in.
    background: Vec<u64>,
    /// Every running sequence the last `set_background` named, background or promoted.
    abandoned: Vec<u64>,
    /// JUNCTION SNAPSHOTS are planned: state snapshots on and `ARF_NO_JUNCTION_SNAPSHOT` unset.
    junctions: bool,
    /// SESSION-START SNAPSHOTS are planned: state snapshots on and
    /// `ARF_NO_SESSION_START_SNAPSHOT` unset (2026-09-26).
    session_starts: bool,
    /// TOOLS-ANCHOR SNAPSHOTS are planned: `ARF_NO_TOOLS_ANCHOR` unset (2026-09-27).
    tools_anchors: bool,
    /// EARLY ANCHORS: anchors are published when prefill passes them, and a request waits for an
    /// anchor in flight (`early_anchors_enabled`, 2026-10-05).
    early_anchors: bool,
    /// SESSION-START SNAPSHOTS: the published snapshots that are ANCHORS or JUNCTIONS — the
    /// shared-prefix class, which a new session resumes from. A sequence resuming at one of these
    /// is the first turn of its conversation; one resuming at any other snapshot resumed from an
    /// earlier turn's end. Added when a sequence's anchor / junction is published, dropped when
    /// the backend evicts it (`forget_snapshot`).
    shared_prefix_keys: HashSet<u64>,
}

/// Prompts shorter than this are not worth a ~150 MB snapshot: they prefill in under a second.
const SNAPSHOT_MIN_TOKENS: usize = 256;

/// PREFIX ANCHOR (2026-09-26): an anchor boundary below this is not planned — a shared prefix
/// that short prefills cheaply, and the anchor pool is small (`ARF_ANCHOR_SNAPSHOT_SLOTS`,
/// default 2, each slot ~235 MB of host RAM; 4 since the tools anchor, 2026-09-27). The HTTP
/// layer applies the same floor before it sends an anchor at all. The tools anchor
/// (`Request::tools_anchor`) has the same floor.
pub const ANCHOR_MIN_TOKENS: usize = 1024;

/// JUNCTION SNAPSHOTS (2026-09-26): no junction below this boundary — the same floor as an
/// anchor, for the same reasons (a short shared prefix prefills cheaply; each slot is ~235 MB).
pub const JUNCTION_MIN_TOKENS: usize = ANCHOR_MIN_TOKENS;

/// JUNCTION SNAPSHOTS: a junction is planned only when the prompt's cached-KV match runs at
/// least this far past the snapshot it resumes from. Below it the re-prefill it would save is a
/// few windows, not worth a slot another prefix could use. It also keeps the NEXT TURN of one
/// conversation from taking junctions: there the match runs past the previous turn's
/// end-of-prompt snapshot by that snapshot's one-window back-off plus the assistant header,
/// 128-256 tokens.
pub const JUNCTION_MIN_EXTRA_TOKENS: usize = 512;

/// `ARF_NO_JUNCTION_SNAPSHOT=1` turns junction snapshots off (the A/B control arm). Read once.
pub fn junction_snapshots_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ARF_NO_JUNCTION_SNAPSHOT").is_none())
}

/// `ARF_NO_TOOLS_ANCHOR=1` plans no TOOLS-ANCHOR snapshot (`SeqPlan::snapshot_tools_anchor`,
/// 2026-09-27): a request's `tools_anchor` is ignored, so every plan is the one taken before it
/// existed (the A/B control arm). The HTTP layer reads the same switch and computes none. Read
/// once.
pub fn tools_anchor_snapshots_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ARF_NO_TOOLS_ANCHOR").is_none())
}

/// `ARF_EARLY_ANCHOR=1` — EARLY ANCHORS (2026-10-05, opt-in until measured on the GPU). Two changes
/// that only make sense together:
///
/// 1. PUBLISH AN ANCHOR WHEN ITS PREFILL PASSES IT, not when its request finishes. Measured the same
///    day: four coding agents started together on the 27B prefilled their shared ~15K-token system +
///    tools prefix FOUR times (8 of 13 requests `cached 0`, first responses 83-276 s, the Claude Code
///    runs timed out), because an anchor taken by a request that was still running was invisible to
///    the others until that request finished its whole answer. What is published early is exactly
///    the KV blocks wholly before the anchor's window-aligned boundary plus the snapshot taken
///    there: the actor saves the snapshot (which waits for the step's GPU work) before this commit,
///    and the registrar's prefill continues into NEW blocks past the boundary, so nothing published
///    is written again. (L125's rule — never publish a still-prefilling sequence's blocks — is about
///    blocks the sequence will keep writing; these it will not. NOT verified on the GPU yet: the
///    prefix-cache gate and a 4-agent run decide whether this becomes the default.)
/// 2. WAIT FOR AN ANCHOR IN FLIGHT instead of computing it again: a queued request whose tools or
///    system anchor is being prefilled right now by a running request stays queued — without
///    holding up unrelated requests behind it — until that anchor is published, then resumes from
///    it. If the running request stops before reaching it, the wait ends with it.
pub fn early_anchors_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ARF_EARLY_ANCHOR").is_some())
}

/// `ARF_NO_SESSION_START_SNAPSHOT=1` files a conversation's first end-of-prompt snapshot in the
/// turn-end pool again, like every other turn's (the A/B control arm, 2026-09-26). Read once.
pub fn session_start_snapshots_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ARF_NO_SESSION_START_SNAPSHOT").is_none())
}

/// The step the ANCHOR and JUNCTION boundaries are a multiple of: `max(block_size,
/// ARF_SNAPSHOT_ALIGN)`, the alignment defaulting to the backend's 128-token prefill window. Since
/// 2026-10-05 that is a cost rule, not a correctness one (the comment where `build_plan` uses
/// it): a boundary inside a window splits the cold prompt into smaller windows, and those cost
/// more a row. One function so the anchor and junction boundaries cannot drift apart.
fn snapshot_step(block_size: usize) -> usize {
    block_size.max(snapshot_align().unwrap_or(128))
}

/// The step the END-OF-PROMPT boundary is a multiple of (2026-10-05): the KV block by default —
/// every resume re-prefills the tail after it, ~15-17 ms a row on the 27B, so it pays to sit as
/// close to the prompt's end as a block allows; the one window it splits is at the very end.
/// `ARF_SNAPSHOT_ALIGN` overrides it too (`=128` is the whole old rule, the A/B control).
fn end_step(block_size: usize) -> usize {
    block_size.max(snapshot_align().unwrap_or(1))
}

/// `ARF_SNAPSHOT_ALIGN`, when set.
fn snapshot_align() -> Option<usize> {
    std::env::var("ARF_SNAPSHOT_ALIGN")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|a| a.max(1))
}

/// A prompt's end-of-prompt snapshot boundary: the last `step` boundary at least
/// [`SNAPSHOT_TAIL_TOKENS`] before the prompt's end (why: the comments where `build_plan` uses
/// it). One function so admission (`Scheduler::is_session_start`) and `build_plan` agree on it.
///
/// 2026-09-27: was "the last boundary leaving >= 1 token, backed off one more `step`" — a
/// 129-255-token back-off to step over the few tokens the next turn renders differently (the
/// assistant header and `<think>`). Measured on agent sessions (`[done]` lines): a turn adds
/// ~150-500 tokens, so the boundary often landed where the previous turn's already was and did
/// not advance (an OpenCode turn 3 resumed at 5,248 of 5,703 tokens, the same point as turn 2),
/// and every turn re-prefilled 200-700 tokens where another engine re-prefills ~150-250. A tail of 32
/// tokens clears the header with room to spare; a tail too short would only make the snapshot
/// unmatchable (its key is the content hash), never wrong. `ARF_SNAPSHOT_TAIL=N` overrides it;
/// `ARF_SNAPSHOT_TAIL=0` restores the old one-step back-off (the A/B control).
///
/// 2026-10-05: `header` — the request's [`Request::header_tail`], the tokens the next turn
/// renders differently, measured from the template — replaces the fixed 32 when known, plus one
/// token of margin (a reply's first characters can merge with the last shared token). Measured on
/// SPEED-Bench's repeated 34,972-token prompt: the boundary was 34,816 (156 tokens re-prefilled,
/// 1.56 s to the first token; the other engines 240-357 ms); with Qwen3.8's header it is 34,944.
/// `ARF_SNAPSHOT_TAIL` still overrides both.
fn end_of_prompt_boundary(prompt_len: usize, step: usize, header: Option<usize>) -> usize {
    // `ARF_SNAPSHOT_TAIL`: unset = `None`; `0` = `Some(None)`, the one-step back-off; `N` = `Some(Some(N))`.
    static ENV: std::sync::OnceLock<Option<Option<usize>>> = std::sync::OnceLock::new();
    let env = *ENV.get_or_init(|| {
        std::env::var("ARF_SNAPSHOT_TAIL")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|t| (t > 0).then_some(t))
    });
    let tail = env.unwrap_or(Some(header.map_or(SNAPSHOT_TAIL_TOKENS, |h| h + 1)));
    match tail {
        None => ((prompt_len.saturating_sub(1) / step) * step).saturating_sub(step),
        Some(t) => (prompt_len.saturating_sub(t.max(1)) / step) * step,
    }
}

/// ROLLING CHECKPOINTS: a long prompt saves a resume point every this many tokens
/// (`SeqPlan::snapshot_checkpoint`). A multiple of the 128-token prefill window. `0` = off;
/// `ARF_CHECKPOINT_TOKENS=N` sets it (rounded down to the snapshot step).
pub const CHECKPOINT_TOKENS: usize = 1024;

/// The shortest prompt that takes rolling checkpoints: below it a re-read costs seconds.
pub const CHECKPOINT_MIN_PROMPT: usize = 8192;

/// The most prompt tokens a step of background reads alone carries (`build_plan`): one prefill
/// window, well under a second of GPU on the 27B.
pub const BACKGROUND_STEP_TOKENS: usize = 128;

/// SHORT READS FIRST: a prompt with at most this many tokens left to read is planned ahead of
/// longer ones (`build_plan`). ~11 s of prefill on the 27B on an M4 Max.
pub const SHORT_READ_TOKENS: usize = 2048;

fn checkpoint_tokens() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("ARF_CHECKPOINT_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|n| n / 128 * 128)
            .unwrap_or(CHECKPOINT_TOKENS)
    })
}

/// Default cached-KV soft cap in tokens (`BlockManager::set_cache_soft_cap`): eight of the Metal
/// pool's 16,384-slot mapping chunks, ~4.4 GB of Qwen3.8's 8-bit KV when full. Live sequences may
/// still grow past it; only cached prefixes are kept inside it. `ARF_KV_CACHE_CAP_TOKENS=N`
/// overrides it, `0` = no cap (the control arm: cached blocks reclaimed only when the free list is
/// empty).
///
/// WAS 49,152 (three chunks, ~1.6 GB) UNTIL 2026-10-07. An agent in auto mode keeps TWO prompts
/// cached: its session and its safety classifier's constant ~30,400-token system prompt. Measured
/// through Claude Code 2.1.292 on a real task: while the session was under ~18,700 tokens its
/// turns resumed from the cache (16,832 cached, 10-20 s); the turn that took it past that — 19,579
/// + 30,400 > 49,152 — and every one after came back `cached 0` (110-128 s each), the session and
/// the classifier evicting each other. A session with a 47K-token project context is past the old
/// cap before its first tool call. The cost is memory, when a server really caches this much:
/// ~33 MB per 1,000 tokens. A budget that follows what the machine has free is the better answer
/// and is not built yet.
pub const KV_CACHE_CAP_TOKENS: usize = 131_072;

fn kv_cache_cap_tokens() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("ARF_KV_CACHE_CAP_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(KV_CACHE_CAP_TOKENS)
    })
}

/// Tokens left after a prompt's end-of-prompt snapshot at least (the next turn re-renders the
/// assistant header); `ARF_SNAPSHOT_TAIL` overrides it, `0` = the old one-window back-off.
pub const SNAPSHOT_TAIL_TOKENS: usize = 32;

/// A request's anchor boundary: its `prefix_anchor` floored to `step`, when that is at least
/// [`ANCHOR_MIN_TOKENS`] and leaves >= 1 prompt token after it. Shared by admission and
/// `build_plan` for the reason `end_of_prompt_boundary` is.
fn anchor_boundary(prefix_anchor: Option<usize>, prompt_len: usize, step: usize) -> Option<usize> {
    prefix_anchor
        .map(|n| (n / step) * step)
        .filter(|&a| a >= ANCHOR_MIN_TOKENS && a < prompt_len)
}

impl Scheduler {
    /// A sequence's `(tools anchor, system anchor)` boundaries, exactly as `build_plan` plans them:
    /// each floored to the snapshot step, the tools anchor only when it lies strictly before the
    /// system anchor (at the same boundary it would be the same snapshot).
    fn anchor_points(&self, seq: &Sequence) -> (Option<usize>, Option<usize>) {
        let step = snapshot_step(self.blocks.block_size());
        let anchor_at = anchor_boundary(seq.prefix_anchor, seq.prompt_len, step);
        let tools_at = anchor_boundary(
            seq.tools_anchor.filter(|_| self.tools_anchors),
            seq.prompt_len,
            step,
        )
        .filter(|&t| anchor_at.is_none_or(|a| t < a));
        (tools_at, anchor_at)
    }

    /// EARLY ANCHORS, part 1: publish `ri`'s held tools / system anchor once its committed prefill
    /// has passed the boundary — the KV blocks wholly before it and the snapshot together, the same
    /// pair a finished sequence publishes (`early_anchors_enabled` has the reasoning).
    fn publish_early_anchors(&mut self, ri: usize) {
        let (tools_at, anchor_at) = self.anchor_points(&self.running[ri]);
        let bs = self.blocks.block_size();
        for (at, tools) in [(tools_at, true), (anchor_at, false)] {
            let Some(at) = at else { continue };
            let seq = &self.running[ri];
            let held = if tools {
                seq.pending_tools_anchor
            } else {
                seq.pending_anchor
            };
            let Some(key) = held else { continue };
            if seq.num_computed < at || self.blocks.chain_hash(&seq.tokens, at / bs) != key {
                continue;
            }
            self.blocks
                .register_blocks_upto(&mut self.running[ri], at / bs);
            self.blocks.note_snapshot(key);
            self.shared_prefix_keys.insert(key);
            let seq = &mut self.running[ri];
            if tools {
                seq.pending_tools_anchor = None;
            } else {
                seq.pending_anchor = None;
            }
        }
    }

    /// Where `seq`'s end-of-prompt snapshot is taken, as `build_plan` plans it; `None` when it
    /// takes none (no snapshots, an image prompt).
    fn end_point(&self, seq: &Sequence) -> Option<usize> {
        if !self.cfg.state_snapshots || seq.image.is_some() {
            return None;
        }
        let at = end_of_prompt_boundary(
            seq.prompt_len,
            end_step(self.blocks.block_size()),
            seq.header_tail,
        );
        (at > 0).then_some(at)
    }

    /// Publish what `ri`'s committed prefill has passed and holds: its anchors
    /// (`publish_early_anchors`), its junction and its end-of-prompt snapshot — each with the KV
    /// blocks wholly before it. For a sequence that will write no more (an evicted one): every
    /// block before such a boundary is final, and so is the state saved there.
    fn publish_passed(&mut self, ri: usize) {
        self.publish_early_anchors(ri);
        if let Some((key, at)) = self.running[ri].pending_checkpoint {
            let bs = self.blocks.block_size();
            if self.running[ri].num_computed >= at
                && self.blocks.chain_hash(&self.running[ri].tokens, at / bs) == key
            {
                self.blocks
                    .register_blocks_upto(&mut self.running[ri], at / bs);
                self.blocks.note_snapshot(key);
                self.running[ri].pending_checkpoint = None;
            }
        }
        let bs = self.blocks.block_size();
        let end = self.end_point(&self.running[ri]);
        let junction = self.running[ri].junction_at;
        for (at, junction_slot) in [(junction, true), (end, false)] {
            let Some(at) = at else { continue };
            let seq = &self.running[ri];
            let held = if junction_slot {
                seq.pending_junction
            } else {
                seq.pending_snapshot
            };
            let Some(key) = held else { continue };
            if seq.num_computed < at || self.blocks.chain_hash(&seq.tokens, at / bs) != key {
                continue;
            }
            self.blocks
                .register_blocks_upto(&mut self.running[ri], at / bs);
            self.blocks.note_snapshot(key);
            let seq = &mut self.running[ri];
            if junction_slot {
                seq.pending_junction = None;
                self.shared_prefix_keys.insert(key);
            } else {
                seq.pending_snapshot = None;
            }
        }
    }

    /// `seq` (queued) extends the prompt an ABANDONED read (`set_background`) is reading and has
    /// not yet passed the end of: it should wait for that read's end-of-prompt snapshot instead
    /// of reading the same tokens beside it. Claude Code's classifier in auto mode sends the same
    /// prompt plus a few tokens before every tool call and gives up after 60 s.
    fn waits_for_background(&self, seq: &Sequence) -> bool {
        let bs = self.blocks.block_size();
        self.running
            .iter()
            .filter(|r| self.abandoned.contains(&r.id))
            .any(|r| self.extends_unread(seq, r, bs))
    }

    /// `seq` shares `r`'s tokens through a boundary `r` saves on its way to its end of prompt
    /// (`saved_on_the_way`), and nothing is published there yet: `r` has not reached it, or has
    /// and publishes it when it is evicted, right after (an abandoned read is evicted once it
    /// has passed what it was worth finishing for, `read_worth_finishing`).
    ///
    /// Not when `seq` already resumes at or past that boundary from the cache. Claude Code's check
    /// in a project carries the project's context (~45,400 tokens) and resumes ~45,000 tokens in
    /// from its own saved state, but shares ~30,000 with a check read without one: in the
    /// 2026-10-08 release test it waited for that read and gave up, four times in a row.
    fn extends_unread(&self, seq: &Sequence, r: &Sequence, bs: usize) -> bool {
        self.saved_on_the_way(seq, r).is_some_and(|at| {
            !self
                .blocks
                .has_snapshot(self.blocks.chain_hash(&r.tokens, at / bs))
                && self.blocks.resume_point(&seq.tokens) < at
        })
    }

    /// The deepest boundary `r` publishes when it is evicted at its end of prompt that `seq`
    /// shares and reads past: the end-of-prompt boundary itself, an anchor, or `r`'s last
    /// rolling checkpoint (the one it holds then). Claude Code 2.1.294's safety check has two
    /// stages whose ~30,500-token prompts part ~300 tokens before their end; when the first is
    /// abandoned part-way (60 s), the next one shares ~30,200 tokens with it but not its end of
    /// prompt, and was read from the start beside it (measured 2026-10-08). It waits for the
    /// checkpoint at 29,696 instead.
    fn saved_on_the_way(&self, seq: &Sequence, r: &Sequence) -> Option<usize> {
        let end = self.end_point(r)?;
        let shared = seq
            .tokens
            .iter()
            .zip(&r.tokens)
            .take_while(|(a, b)| a == b)
            .count()
            .min(seq.tokens.len() - 1);
        let cp = checkpoint_tokens();
        let checkpoint = (cp > 0 && r.prompt_len >= CHECKPOINT_MIN_PROMPT && end >= cp + cp / 2)
            .then(|| (end - cp / 2) / cp * cp);
        let (tools_at, anchor_at) = self.anchor_points(r);
        [Some(end), checkpoint, anchor_at, tools_at]
            .into_iter()
            .flatten()
            .filter(|&at| at <= shared)
            .max()
    }

    /// EARLY ANCHORS, part 2: `seq` (queued) shares a tools or system anchor that a running
    /// sequence is prefilling right now and has not published yet — so it should wait for it
    /// rather than compute the same prefix again.
    fn waits_for_anchor(&self, seq: &Sequence) -> bool {
        if !self.early_anchors {
            return false;
        }
        let bs = self.blocks.block_size();
        let (tools_at, anchor_at) = self.anchor_points(seq);
        for at in [tools_at, anchor_at].into_iter().flatten() {
            let key = self.blocks.chain_hash(&seq.tokens, at / bs);
            if self.blocks.has_snapshot(key) {
                continue; // published: admission matches to it
            }
            let in_flight = self.running.iter().any(|r| {
                let (rt, ra) = self.anchor_points(r);
                [rt, ra].contains(&Some(at))
                    && r.num_computed < at
                    && self.blocks.chain_hash(&r.tokens, at / bs) == key
            });
            if in_flight {
                return true;
            }
        }
        false
    }

    pub fn new(cfg: EngineConfig) -> Self {
        assert!(
            cfg.max_prefill_tokens >= 1,
            "max_prefill_tokens must be >= 1"
        );
        let mut blocks = BlockManager::with_prefix_cache(
            cfg.num_blocks,
            cfg.block_size,
            cfg.enable_prefix_cache,
        );
        // Reserve one block per sequence that may run: the lookahead never takes a block another
        // running sequence could need for its next real token.
        blocks.set_decode_lookahead(cfg.decode_lookahead_tokens, cfg.max_batch_size);
        if cfg.enable_prefix_cache {
            blocks.set_cache_soft_cap(kv_cache_cap_tokens() / cfg.block_size.max(1));
        }
        if cfg.state_snapshots {
            blocks.require_state_snapshots();
        }
        let junctions = cfg.state_snapshots && junction_snapshots_enabled();
        let session_starts = cfg.state_snapshots && session_start_snapshots_enabled();
        let tools_anchors = cfg.state_snapshots && tools_anchor_snapshots_enabled();
        let early_anchors = cfg.state_snapshots && early_anchors_enabled();
        Scheduler {
            waiting: VecDeque::new(),
            running: Vec::new(),
            blocks,
            cfg,
            scheduled: Vec::new(),
            decode_only: None,
            prefill_cap: None,
            background: Vec::new(),
            abandoned: Vec::new(),
            junctions,
            session_starts,
            tools_anchors,
            early_anchors,
            shared_prefix_keys: HashSet::new(),
        }
    }

    /// Running sequences in decode (one pending token): `(id, num_computed)`.
    pub fn decode_seqs(&self) -> Vec<(u64, usize)> {
        self.running
            .iter()
            .filter(|s| s.num_pending() == 1)
            .map(|s| (s.id, s.num_computed))
            .collect()
    }

    /// A RUNNING sequence's tokens (prompt, then whatever it has generated), or `None` when `id`
    /// is not running. Read-only; the serving loop uses it to say whether a sequence that
    /// resumed from an anchor snapshot is a NEW session or the same conversation (2026-09-26).
    pub fn running_tokens(&self, id: u64) -> Option<&[u32]> {
        self.running
            .iter()
            .find(|s| s.id == id)
            .map(|s| s.tokens.as_slice())
    }

    /// Running sequences still reading their prompt: `(id, prompt tokens computed, prompt length)`
    /// — the serving loop's progress line and `/v1/arf/status` (2026-10-06).
    pub fn prefill_progress(&self) -> Vec<(u64, usize, usize)> {
        self.running
            .iter()
            .filter(|s| s.num_computed < s.prompt_len)
            .map(|s| (s.id, s.num_computed, s.prompt_len))
            .collect()
    }

    /// A prompt is still being prefilled, or a request is waiting to be admitted.
    pub fn has_prefill_work(&self) -> bool {
        !self.waiting.is_empty() || self.running.iter().any(|s| s.num_pending() > 1)
    }

    /// DECODE ROUND-ROBIN (2026-09-26): the next plan carries only decode sequence `id` (prefill
    /// chunks still fill the budget). The actor uses it to give two speculating streams one
    /// speculative window each in turn instead of one plain row each per step. Ignored when `id`
    /// is not a running decode sequence by then (a preemption), so a plan is never emptied by it.
    pub fn set_decode_only(&mut self, id: Option<u64>) {
        self.decode_only = id;
    }

    /// ABANDONED READS (2026-10-07): `ids` are sequences whose client has gone but whose read is
    /// worth finishing up to an anchor (`anchor_ahead`). They keep their place and their state,
    /// and are planned only in a step that would otherwise be empty, so a request someone is
    /// waiting for never shares the GPU with one nobody is. Measured through Claude Code 2.1.292
    /// in auto mode: its safety classifier gives up on a 30,500-token prompt after 60 s, and the
    /// session's next turn then waited 450 s behind two reads of it that nobody was waiting for.
    /// Replaces the previous set; an id that is not running is ignored.
    pub fn set_background(&mut self, ids: &[u64]) {
        // A read a queued request waits for (`waits_for_background`) is no longer one nobody
        // waits for: it is planned as any other, or the waiter would wait behind every live step.
        let bs = self.blocks.block_size();
        self.background = ids
            .iter()
            .copied()
            .filter(|id| {
                self.running
                    .iter()
                    .find(|r| r.id == *id)
                    .is_none_or(|r| !self.waiting.iter().any(|w| self.extends_unread(w, r, bs)))
            })
            .collect();
        // Kept for `waits_for_background`: every abandoned read, background or promoted.
        self.abandoned = ids.to_vec();
    }

    /// Is `id` queued and not yet started?
    pub fn is_waiting(&self, id: u64) -> bool {
        self.waiting.iter().any(|s| s.id == id)
    }

    /// Does running sequence `id` still have a tools or system anchor ahead of its committed
    /// prefill that nobody has published yet? That is what an abandoned read is worth finishing
    /// for: once passed (or when another sequence published it first), evicting the sequence
    /// leaves the anchor (`evict_seqs`) and nothing more is gained by reading on.
    pub fn anchor_ahead(&self, id: u64) -> bool {
        self.anchor_ahead_of(id)
    }

    /// Is an abandoned read worth finishing? While an anchor nobody has published is ahead of it
    /// ([`anchor_ahead`](Self::anchor_ahead)), or its own end-of-prompt boundary is and nothing is
    /// saved there yet: the same prompt plus a few tokens is what a client that gave up sends next.
    pub fn read_worth_finishing(&self, id: u64) -> bool {
        if self.anchor_ahead_of(id) {
            return true;
        }
        let Some(seq) = self.running.iter().find(|s| s.id == id) else {
            return false;
        };
        let bs = self.blocks.block_size();
        self.end_point(seq).is_some_and(|at| {
            seq.num_computed < at
                && !self
                    .blocks
                    .has_snapshot(self.blocks.chain_hash(&seq.tokens, at / bs))
        })
    }

    fn anchor_ahead_of(&self, id: u64) -> bool {
        let Some(seq) = self.running.iter().find(|s| s.id == id) else {
            return false;
        };
        let bs = self.blocks.block_size();
        let (tools_at, anchor_at) = self.anchor_points(seq);
        [tools_at, anchor_at].into_iter().flatten().any(|at| {
            seq.num_computed < at
                && !self
                    .shared_prefix_keys
                    .contains(&self.blocks.chain_hash(&seq.tokens, at / bs))
        })
    }

    /// DECODE SHARE (2026-10-07): the next plan reads at most `cap` prompt tokens, ending the
    /// chunk on a multiple of `cap`; `Some(0)` plans decode only. One step could carry
    /// `max_prefill_tokens` (4,096) prompt tokens — ~24 s of GPU on an M4 Max with the 27B — so
    /// a sequence decoding beside a long prompt got one token per step. The actor sets a cap of
    /// one prefill window while anything decodes, and 0 while it owes the decoders time. Ignored
    /// when no sequence is in decode by then, so a plan is never emptied by it.
    pub fn set_prefill_cap(&mut self, cap: Option<usize>) {
        self.prefill_cap = cap;
    }

    /// Enqueue a new request.
    pub fn add(&mut self, req: Request) {
        self.waiting.push_back(Sequence::from_request(req));
    }

    /// Whether any work remains.
    pub fn has_unfinished(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty()
    }

    pub fn num_waiting(&self) -> usize {
        self.waiting.len()
    }

    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    pub fn num_free_blocks(&self) -> usize {
        self.blocks.num_free_blocks()
    }

    /// Total number of physical KV blocks in the pool (free + used).
    pub fn num_total_blocks(&self) -> usize {
        self.blocks.total_blocks()
    }

    /// Total prompt-prefix tokens served from the prefix cache , i.e.
    /// tokens whose KV was reused instead of recomputed. Always 0 when prefix
    /// caching is disabled. A measurement hook for the prefill-saved metric.
    pub fn prefix_cache_reused_tokens(&self) -> u64 {
        self.blocks.reused_tokens()
    }

    /// Plan the next step. Returns `None` when nothing can run.
    pub fn schedule(&mut self) -> Result<Option<BatchPlan>> {
        debug_assert!(
            self.scheduled.is_empty(),
            "schedule() called without committing the previous plan"
        );
        self.reschedule_running()?;
        self.admit_waiting()?;
        if self.running.is_empty() {
            return Ok(None);
        }
        Ok(Some(self.build_plan()))
    }

    /// Keep running sequences, preempting the newest when blocks run short.
    fn reschedule_running(&mut self) -> Result<()> {
        let mut candidates: VecDeque<Sequence> = std::mem::take(&mut self.running).into();
        let mut survivors: Vec<Sequence> = Vec::with_capacity(candidates.len());

        'next: while let Some(mut seq) = candidates.pop_front() {
            while !self.blocks.can_allocate(&seq) {
                match candidates.pop_back() {
                    // Preempt the newest not-yet-scheduled sequence and recompute it later.
                    Some(mut victim) => {
                        self.blocks.free(&mut victim);
                        victim.reset_for_recompute();
                        self.waiting.push_front(victim);
                    }
                    // Nothing left to preempt: this sequence yields its turn.
                    None => {
                        self.blocks.free(&mut seq);
                        seq.reset_for_recompute();
                        self.waiting.push_front(seq);
                        continue 'next;
                    }
                }
            }
            self.blocks.allocate(&mut seq)?;
            seq.status = SequenceStatus::Running;
            survivors.push(seq);
        }
        self.running = survivors;
        Ok(())
    }

    /// Admit waiting requests while batch-size, block, and step-budget room
    /// remain. A prompt larger than the remaining `max_prefill_tokens` is
    /// still admitted — `build_plan` chunks its compute across steps (its
    /// blocks are allocated up front; chunking splits compute, not memory).
    fn admit_waiting(&mut self) -> Result<()> {
        // A background read (`set_background`) steps aside for whatever is admitted here, so it
        // does not use up the step budget. Until 2026-10-08 it did: a 30,593-token background
        // read kept the user's 15,621-token first message queued until it ended (first token
        // 221 s, ~130 s of it waiting), and an abandoned check did the same to the agent's turns.
        let mut batch_tokens: usize = self
            .running
            .iter()
            .filter(|s| !self.background.contains(&s.id))
            .map(|s| s.num_pending())
            .sum();
        // EARLY ANCHORS: requests waiting for an anchor in flight, set aside for this pass and put
        // back at the front, in order, afterwards — they keep their place in the queue.
        let mut deferred: VecDeque<Sequence> = VecDeque::new();
        while self.running.len() < self.cfg.max_batch_size {
            if self.waiting.front().is_none() {
                break;
            }
            if self
                .waiting
                .front()
                .is_some_and(|s| self.waits_for_anchor(s) || self.waits_for_background(s))
            {
                deferred.push_back(self.waiting.pop_front().expect("front exists"));
                continue;
            }
            // Tentatively admit: claim any shared prefix FIRST (which shrinks the
            // blocks this sequence still needs), then check it fits. On a miss,
            // roll the match back and leave the sequence at the front.
            let mut seq = self.waiting.pop_front().expect("front exists");
            self.blocks.match_prefix(&mut seq);
            // No step budget left: nothing admitted now can compute this step — except a SHORT
            // read, which `build_plan` puts ahead of the long ones that used the budget up.
            if batch_tokens >= self.cfg.max_prefill_tokens && seq.num_pending() > SHORT_READ_TOKENS
            {
                self.blocks.unmatch_prefix(&mut seq);
                self.waiting.push_front(seq);
                break;
            }
            if !self.blocks.can_allocate(&seq) {
                self.blocks.unmatch_prefix(&mut seq);
                self.waiting.push_front(seq);
                break;
            }
            self.blocks.allocate(&mut seq)?;
            seq.junction_at = self.junction_boundary(&seq);
            seq.session_start = self.is_session_start(&seq);
            seq.status = SequenceStatus::Running;
            batch_tokens += seq.num_pending();
            self.running.push(seq);
        }
        while let Some(seq) = deferred.pop_back() {
            self.waiting.push_front(seq);
        }
        Ok(())
    }

    /// JUNCTION SNAPSHOTS (2026-09-26): where a just-admitted sequence leaves cached history, if
    /// a snapshot there is worth taking. `kv_match_len` is what the block manager verified the
    /// cache holds for this prompt; `num_computed` is where it resumes (the deepest snapshot on
    /// that chain, or 0). Floored to the same `snapshot_step` as every other snapshot boundary —
    /// the window alignment is a correctness requirement, not a tuning knob. The checks that
    /// depend on the plan (the anchor, the end-of-prompt boundary, keys already taken) are made
    /// in `build_plan`.
    fn junction_boundary(&self, seq: &Sequence) -> Option<usize> {
        if !self.junctions || seq.image.is_some() {
            return None;
        }
        let resume = seq.num_computed;
        if seq.kv_match_len < resume + JUNCTION_MIN_EXTRA_TOKENS {
            return None;
        }
        let step = snapshot_step(self.blocks.block_size());
        Some((seq.kv_match_len / step) * step).filter(|&j| j > resume && j >= JUNCTION_MIN_TOKENS)
    }

    /// SESSION-START SNAPSHOTS (2026-09-26): is a just-admitted sequence the FIRST turn of a
    /// conversation, whose end-of-prompt snapshot belongs in the long-lived junction pool?
    ///
    /// Measured the same day (measured 2026-09-26): Claude Code
    /// sessions ran 6-7 turns, and a second session with the IDENTICAL task resumed only at the
    /// anchor ("resumes from the anchor snapshot at 13184 tokens"): first turn 14.4 s, another engine
    /// 2.0 s. Each turn saves a turn-end snapshot into the 4-slot turn-end pool, so the first
    /// turn's is evicted by the fifth — the first suspect for that gap, not a confirmed
    /// cause. The effect of this change was NOT measured when written; measured since
    /// (measured 2026-09-26): the identical task
    /// repeated, first turn 14.4 -> 3.9 s, whole task 35.6 -> 20.5 s.
    ///
    /// The first turn is told apart by WHERE IT RESUMED (`num_computed` / `restore_key`, as the
    /// prefix match left them): at 0, at or before its own anchor boundary, or at a published
    /// anchor or junction (`shared_prefix_keys`) — the snapshots a NEW session resumes from. Any
    /// other resume point is an earlier turn's end-of-prompt snapshot: a later turn of the same
    /// conversation, which is unchanged (turn-end pool). Session-start snapshots are themselves
    /// NOT in `shared_prefix_keys`, so turn 2, which resumes at turn 1's, is a later turn.
    ///
    /// Floors, the junction's, for the same reason (a slot in the long-lived pool must be worth
    /// it): the boundary is >= [`JUNCTION_MIN_TOKENS`] and >= [`JUNCTION_MIN_EXTRA_TOKENS`] past
    /// the resume point. A short first request (an agent's side query, say) re-prefills cheaply
    /// and would otherwise evict a session start or a junction worth keeping; below the floors
    /// its snapshot is an ordinary turn-end one, as before.
    ///
    /// KNOWN MISCLASSIFICATIONS (how often is not measured): a later turn that resumes at 0 or
    /// at the anchor — every deeper snapshot of its conversation evicted, or a first message so
    /// short that the anchor was turn 1's only snapshot (`build_plan`, "Corrected 2026-09-26
    /// (review)") — counts as a first turn and takes a junction-pool slot.
    fn is_session_start(&self, seq: &Sequence) -> bool {
        if !self.session_starts || seq.image.is_some() {
            return false;
        }
        let step = snapshot_step(self.blocks.block_size());
        let at = end_of_prompt_boundary(
            seq.prompt_len,
            end_step(self.blocks.block_size()),
            seq.header_tail,
        );
        let resume = seq.num_computed;
        if at < JUNCTION_MIN_TOKENS || at < resume + JUNCTION_MIN_EXTRA_TOKENS {
            return false;
        }
        resume == 0
            || anchor_boundary(seq.prefix_anchor, seq.prompt_len, step).is_some_and(|a| resume <= a)
            || seq
                .restore_key
                .is_some_and(|k| self.shared_prefix_keys.contains(&k))
    }

    /// Flatten the running set into a [`BatchPlan`] under the per-step token
    /// budget. Decode sequences (1 pending token) are scheduled FIRST so a
    /// long prompt's prefill chunks can never stall token streaming; prefill chunks fill the remaining
    /// budget. A sequence that gets no budget this step simply isn't in the
    /// plan and resumes next step.
    ///
    /// Ordering is stable across steps (decode-first, then FCFS by running
    /// order among prefills), so when the budget is smaller than the number of
    /// decode sequences, the same tail is omitted every step — configs must
    /// keep `max_prefill_tokens >= expected concurrent decode sequences` (the
    /// serve CLI warns).
    fn build_plan(&mut self) -> BatchPlan {
        let mut budget = self.cfg.max_prefill_tokens;
        let only = self.decode_only.take().filter(|id| {
            self.running
                .iter()
                .any(|s| s.id == *id && s.num_pending() == 1)
        });
        let decode = self
            .running
            .iter()
            .enumerate()
            .filter(|(_, s)| s.num_pending() == 1)
            .filter(move |(_, s)| only.is_none_or(|id| s.id == id))
            .map(|(i, _)| i);
        // SHORT READS FIRST (2026-10-08): among prefills, one with at most `SHORT_READ_TOKENS`
        // left goes ahead of longer ones; FCFS within each group. Measured through Claude Code
        // 2.1.294 in auto mode: its safety check (~30,500 tokens) was being read when the
        // agent's next turn came in, ~230 tokens past a saved state, and the turn waited for
        // the whole read on the 512-token step (first token 64 s and 125 s). The long read
        // gives up a step or two (~3 s each) for it.
        let mut prefill: Vec<usize> = self
            .running
            .iter()
            .enumerate()
            .filter(|(_, s)| s.num_pending() > 1)
            .map(|(i, _)| i)
            .collect();
        prefill.sort_by_key(|&i| self.running[i].num_pending() > SHORT_READ_TOKENS);
        let mut order: Vec<usize> = decode.chain(prefill).collect();
        // ABANDONED READS: step aside whenever anything else has work (`set_background`).
        if order
            .iter()
            .any(|&i| !self.background.contains(&self.running[i].id))
        {
            order.retain(|&i| !self.background.contains(&self.running[i].id));
        }
        let cap = self
            .prefill_cap
            .take()
            .filter(|_| self.running.iter().any(|s| s.num_pending() == 1));
        let mut prefill_left = cap.unwrap_or(usize::MAX);
        // BACKGROUND STEPS ARE SHORT (2026-10-08): a step that carries only background reads is
        // capped at `BACKGROUND_STEP_TOKENS`. A step cannot be interrupted once on the GPU, so a
        // request arriving during a full 512-token background step waited ~3 s for it — on every
        // turn of an agent session while Claude Code's safety check was read in the background.
        if !order.is_empty()
            && order
                .iter()
                .all(|&i| self.background.contains(&self.running[i].id))
        {
            prefill_left = prefill_left.min(BACKGROUND_STEP_TOKENS);
        }

        let mut input_ids = Vec::new();
        let mut positions = Vec::new();
        let mut seqs = Vec::new();
        let mut q_start = 0;
        // Anchor keys planned earlier in THIS plan: two new sessions admitted together must not
        // both save the same ~235 MB anchor (neither is held yet — holding follows the step).
        // Junction keys too (2026-09-26), for the same reason: two new sessions of one project
        // admitted together leave the same cached history at the same point.
        let mut planned_anchors: Vec<u64> = Vec::new();
        self.scheduled.clear();
        for ri in order {
            if budget == 0 {
                break;
            }
            let past = self.running[ri].num_computed;
            let mut q_len = self.running[ri].num_pending().min(budget);
            let is_prefill = self.running[ri].num_pending() > 1;
            if is_prefill {
                if prefill_left == 0 {
                    continue;
                }
                q_len = q_len.min(prefill_left);
                if let Some(c) = cap {
                    q_len = q_len.min(c - past % c);
                }
            }
            // STATE SNAPSHOTS: a prompt's last block boundary (leaving >= 1 token after it) is
            // where the NEXT turn of the same conversation will match to — a thinking model's
            // template drops the earlier <think> text, so the shared prefix ends with the
            // prompt, not with the answer. End one chunk exactly there and have it snapshotted.
            let mut snapshot_key = None;
            let mut snapshot_anchor = false;
            let mut snapshot_tools_anchor = false;
            let mut snapshot_junction = false;
            let mut snapshot_session_start = false;
            let mut snapshot_checkpoint = false;
            if self.cfg.state_snapshots && self.running[ri].image.is_none() {
                let bs = self.blocks.block_size();
                // ... backed off ONE more block: a chat prompt ends with the assistant header (and
                // a thinking model's `<think>`), which the next turn renders differently, so the
                // last few tokens of this prompt are not a prefix of the next one.
                // WAS (until 2026-10-05) ALSO A MULTIPLE OF THE BACKEND'S PREFILL WINDOW (128),
                // held as a correctness requirement: a sequence resuming from a snapshot at 672
                // splits its remaining prompt into windows starting at 672, while an uncached one
                // splits at 640/768 — different chunkings of the same recurrence. MEASURED
                // 2026-09-21: turn 2 of a conversation differed from a `--no-prefix-cache`
                // server at align=1 and was character-identical at align=128.
                // 2026-10-05: no longer a correctness requirement. Prompt windows now run through
                // the verify record, which steps the recurrence one ROW at a time, so a different
                // chunking moves the numbers only as a different batch shape does — never to
                // another sequence's text. The hybrid prefix-cache gate (identical, or parting
                // only at a near-tie of < 0.1 nats on both servers) PASSED three times with every
                // boundary at the KV block (16): turn 2 identical; partings at near-ties of 0.009
                // and 0.004 nats, the kind the speculative verify's batch shape already makes.
                // So the END-OF-PROMPT boundary is block-aligned (`end_step`): the re-prefilled
                // tail costs ~15-17 ms a ROW on the 27B (the record steps the recurrent layers
                // per row), and a repeated 34,972-token prompt took 510 ms to its first token at
                // 128 (28 rows) and 198 ms at 16 (9 rows), against 240-357 ms in other engines.
                // The ANCHOR and JUNCTION boundaries stay on the 128-token window
                // (`snapshot_step`): with them at 16 too, cold requests with the cache on got
                // SLOWER than cache off (the gate's two concurrent 1,010-token requests: 15.1 s
                // vs 11.5 s; at 128, 12.2 s vs 14.1 s) — a boundary inside the prompt splits it
                // into smaller windows, which cost more a row. `ARF_SNAPSHOT_ALIGN=N` sets both.
                // (2026-09-26: computed by `snapshot_step`, which the junction boundary shares.)
                let step = snapshot_step(bs);
                // (2026-09-26: computed by `end_of_prompt_boundary`, shared with admission.)
                let at = end_of_prompt_boundary(
                    self.running[ri].prompt_len,
                    end_step(bs),
                    self.running[ri].header_tail,
                );
                // PREFIX ANCHOR (2026-09-26). The end-of-prompt snapshot above serves the next
                // turn of THIS conversation; nothing served a NEW session of the same agent,
                // which shares the system prompt and tools and diverges at its first user
                // message. Measured: Claude Code's first turn re-prefilled its ~13.4K-token
                // system + tools prefix on every new session (88-90 s; an identical request
                // hit in 3 s). So a request that knows where its shared prefix ends
                // (`prefix_anchor`, from the HTTP layer) ALSO ends a chunk there — floored to
                // the SAME `step`, because the window alignment above is a correctness
                // requirement for this boundary exactly as for that one — and snapshots it
                // into the backend's anchor pool. `match_prefix` needs no change: it already
                // resumes at the deepest snapshotted boundary along the prompt's chain, and a
                // new session's chain runs through the anchor and stops a few blocks later.
                // Skipped when the key is already published or held (it will be matched, or will
                // be once its holder finishes), when this sequence resumed at or past it, and
                // when it is not before the end-of-prompt boundary (that snapshot covers it —
                // and is itself filed as an anchor below when it lies inside the shared prefix).
                //
                // **Corrected 2026-09-26 (review):** "filed as an anchor below" was wrong for a
                // SHORT first message. Its end-of-prompt boundary lands BEFORE the anchor
                // (1,320 tokens, anchor 1,300: `at` = 1,152 < `a` = 1,280), so turn 1 saved
                // 1,152 into the anchor pool, and turn 2 of the same conversation — resuming
                // there, with `a` still ahead — saved a SECOND anchor at 1,280. With the default
                // two anchor slots one agent then held both and the next save evicted another
                // agent's anchor (its next new session re-prefilled its whole prefix — the miss
                // the pool exists to prevent), while 1,152 was dead weight (~235 MB; every later
                // session resumes at the deeper 1,280). Now: when `at <= a` the ONE snapshot is
                // taken at `a` — it serves both the next turn (every token before `a` precedes
                // the user's text, so that turn shares them) and new sessions — and no
                // end-of-prompt snapshot is planned (it would be shallower than `a`). `a` must
                // leave >= 1 prompt token after it, like `at`.
                // (2026-09-26: computed by `anchor_boundary`, shared with admission.)
                let anchor_at = anchor_boundary(
                    self.running[ri].prefix_anchor,
                    self.running[ri].prompt_len,
                    step,
                );
                // TOOLS ANCHOR (2026-09-27). The anchor above is where the system text ENDS, so it
                // serves a new session only when that session's system text is the same. Claude
                // Code's is not across working directories (it carries the cwd and environment),
                // and Qwen3.8's template renders the tools block BEFORE the system text: a
                // session in another directory shares the ~12K-token tool schemas and nothing
                // after them, and resumed from nothing (measured 2026-09-27: `anchor snapshot at
                // 13056` for session 1; sessions 2 and 3, other directories, no resume, ~75 s
                // each). So a request that knows where its tools block ends
                // (`tools_anchor`, from the HTTP layer) ALSO ends a chunk there — floored to the
                // same `step`, for the same correctness reason — and snapshots it into the
                // anchor pool. Planned BEFORE the system anchor (it lies before it), and only
                // when its floored boundary is strictly before the system anchor's (at the same
                // boundary it would be the same snapshot: nothing gained). Skipped, like the
                // anchor, when published, held, planned in this step or resumed past.
                // `match_prefix` needs no change: a session whose prompt agrees only up to the
                // tools block resumes at the deepest snapshot on its chain, this one.
                let tools_at = anchor_boundary(
                    self.running[ri].tools_anchor.filter(|_| self.tools_anchors),
                    self.running[ri].prompt_len,
                    step,
                )
                .filter(|&t| anchor_at.is_none_or(|a| t < a));
                if let Some(t) = tools_at.filter(|&t| past < t) {
                    let key = self.blocks.chain_hash(&self.running[ri].tokens, t / bs);
                    if !self.snapshot_known(key) && !planned_anchors.contains(&key) {
                        q_len = q_len.min(t - past);
                        if past + q_len == t {
                            snapshot_key = Some(key);
                            snapshot_anchor = true;
                            snapshot_tools_anchor = true;
                        }
                    }
                }
                if let Some(a) = anchor_at.filter(|&a| past < a && snapshot_key.is_none()) {
                    let key = self.blocks.chain_hash(&self.running[ri].tokens, a / bs);
                    if !self.snapshot_known(key) && !planned_anchors.contains(&key) {
                        q_len = q_len.min(a - past);
                        if past + q_len == a {
                            snapshot_key = Some(key);
                            snapshot_anchor = true;
                        }
                    }
                }
                // JUNCTION (2026-09-26). The anchor serves a new session only down to the end of
                // the system + tools prefix. With Claude Code 2.1.283 the same day the anchor sat
                // at 13,184 tokens (measured 2026-09-26), and what follows it in a first
                // request (a <system-reminder> user block, then the environment block) is the
                // SAME across new sessions in one project, ahead of the task text — re-prefilled
                // every time, because the only snapshots were the anchor and each prompt's own
                // end (the size of that stretch is an unrecorded trace; the
                // effect of this change is NOT measured). another engine learns
                // "junctions" instead: a snapshot where a new prompt diverges from cached
                // history (read in its source: `CacheLookup::junctionBoundary()` in
                // runtime/engine/Cache.hpp is `kvBoundary > resumeBoundary ? kvBoundary : 0`,
                // planned as a state boundary by `Engine::configureDraftStatePlan`, with NO
                // minimum distance — ours has the two floors below, and the window alignment).
                // Same here: when this prompt's cached-KV match
                // (`Sequence::kv_match_len`, taken at admission BEFORE the backoff to a
                // snapshot) ran >= JUNCTION_MIN_EXTRA_TOKENS past where it resumed, end a chunk
                // at that divergence point, floored to the same `step` (window alignment is a
                // correctness requirement here exactly as above), and snapshot it. The next
                // session that shares the longer prefix resumes there. Order within a prompt:
                // anchor < junction < end-of-prompt — a chunk carries one key, and the `min`s
                // cut each chunk at the first boundary still ahead. Skipped when the key is
                // published, held or planned in this step, when it is not past the anchor (the
                // anchor covers it), and when it is not before the end-of-prompt boundary (that
                // snapshot covers it). A cut here can leave a one-token chunk `[j-1, j)`; it is
                // not a decode row (`completes_prompt` is false), so the serving loop does not
                // speculate on it (the 2026-09-26 anchor review fix).
                if snapshot_key.is_none() {
                    let junction_at = self.running[ri]
                        .junction_at
                        .filter(|&j| past < j && j < at && anchor_at.is_none_or(|a| j > a));
                    if let Some(j) = junction_at {
                        let key = self.blocks.chain_hash(&self.running[ri].tokens, j / bs);
                        if !self.snapshot_known(key) && !planned_anchors.contains(&key) {
                            q_len = q_len.min(j - past);
                            if past + q_len == j {
                                snapshot_key = Some(key);
                                snapshot_junction = true;
                            }
                        }
                    }
                }
                // ROLLING CHECKPOINT (2026-10-07): a long prompt also ends a chunk every
                // `checkpoint_tokens` and saves a resume point there. Measured through Claude Code
                // 2.1.292 in auto mode with a 46 KB AGENTS.md: its safety check is ~45,400 tokens,
                // and the next one differs ~70 tokens before the end (the new action goes in before
                // a fixed closing instruction), so neither the end-of-prompt snapshot nor the
                // anchor 15,000 tokens back served it: every check read ~15,000 tokens, longer
                // than its 60 s, and was abandoned. Only the latest checkpoint of a sequence is
                // kept; not within half a step of the end-of-prompt boundary, which is saved anyway;
                // and only in a prompt of `CHECKPOINT_MIN_PROMPT` tokens or more: a shorter one is
                // read again in seconds, and its chunk plan stays as it was.
                let cp = checkpoint_tokens();
                if snapshot_key.is_none()
                    && cp > 0
                    && past < at
                    && self.running[ri].prompt_len >= CHECKPOINT_MIN_PROMPT
                {
                    let c = (past / cp + 1) * cp;
                    // The hash only once `c` is known to lie inside this chunk: past the prompt's
                    // end it would read beyond its tokens (a 13,780-token prompt panicked here on
                    // the GPU, 2026-10-07, before this order). A chunk that ENDS on `c` takes it
                    // too (`<=`). Until 2026-10-08 it was `<`, and on the server's 512-token step
                    // every multiple of 1,024 is a chunk end: a 30,483-token prompt read from the
                    // start saved no checkpoint at all, and only a prompt whose chunks an anchor
                    // had shifted off that grid saved any (measured through Claude Code 2.1.294:
                    // its safety check's two stages share ~30,200 tokens and part before the
                    // anchor at 30,336, so the second stage read all ~30,700 again, 170 s).
                    if c <= past + q_len && c + cp / 2 <= at {
                        let key = self.blocks.chain_hash(&self.running[ri].tokens, c / bs);
                        if !self.snapshot_known(key) && !planned_anchors.contains(&key) {
                            q_len = c - past;
                            snapshot_key = Some(key);
                            snapshot_checkpoint = true;
                        }
                    }
                }
                // The anchor at or beyond the end-of-prompt boundary covers it (above).
                let end_covered = anchor_at.is_some_and(|a| at <= a);
                if snapshot_key.is_none() && !end_covered && at >= SNAPSHOT_MIN_TOKENS && past < at
                {
                    q_len = q_len.min(at - past);
                    if past + q_len == at {
                        snapshot_key =
                            Some(self.blocks.chain_hash(&self.running[ri].tokens, at / bs));
                        // SESSION START (2026-09-26): a conversation's first end-of-prompt
                        // snapshot goes to the junction pool, where its own later turns cannot
                        // evict it (`is_session_start`). Same key, chunk, hold and publish as
                        // before; only the pool differs.
                        snapshot_session_start = self.running[ri].session_start;
                    }
                }
                if snapshot_tools_anchor {
                    planned_anchors.extend(snapshot_key);
                    self.running[ri].tools_anchor_key = snapshot_key;
                } else if snapshot_anchor {
                    planned_anchors.extend(snapshot_key);
                    self.running[ri].anchor_key = snapshot_key;
                }
                if snapshot_junction {
                    planned_anchors.extend(snapshot_key);
                    self.running[ri].junction_key = snapshot_key;
                }
                if snapshot_checkpoint {
                    self.running[ri].checkpoint_key = snapshot_key.map(|k| (k, past + q_len));
                }
            }
            let restore_key = self.running[ri].restore_key.take();
            let seq = &self.running[ri];
            budget -= q_len;
            if is_prefill {
                prefill_left = prefill_left.saturating_sub(q_len);
            }
            input_ids.extend_from_slice(&seq.tokens[past..past + q_len]);
            positions.extend((past..past + q_len).map(|p| p as u32));
            // Vision: collect the image soft-tokens whose prompt position falls in THIS
            // prefill chunk [past, past+q_len). Local position p → flat row q_start+(p-past);
            // the embed index is p's ordinal among the sequence's image positions.
            let image_rows = seq
                .image
                .as_ref()
                .map(|im| {
                    im.positions
                        .iter()
                        .enumerate()
                        .filter(|(_, &p)| p >= past && p < past + q_len)
                        .map(|(ei, &p)| {
                            let row = q_start + (p - past);
                            let emb = im.embeds[ei * im.hidden..(ei + 1) * im.hidden].to_vec();
                            (row, emb)
                        })
                        .collect()
                })
                .unwrap_or_default();
            let mrope = seq
                .image
                .as_ref()
                .and_then(|im| im.mrope.as_ref())
                .map(|l| (past..past + q_len).map(|p| l.at(p)).collect());
            let completes_prompt = q_len >= seq.num_pending();
            seqs.push(SeqPlan {
                id: seq.id,
                q_start,
                q_len,
                past_len: past,
                block_table: seq.block_table.clone(),
                image_rows,
                mrope,
                image_causal: seq.image.as_ref().is_some_and(|im| im.causal),
                restore_key,
                snapshot_key,
                snapshot_anchor,
                snapshot_tools_anchor,
                snapshot_junction,
                snapshot_session_start,
                snapshot_checkpoint,
                completes_prompt,
            });
            self.scheduled.push((ri, q_len));
            q_start += q_len;
        }
        BatchPlan {
            input_ids,
            positions,
            seqs,
        }
    }

    /// The backend saved a recurrent-state snapshot for sequence `id` under `key`. It is HELD,
    /// not published: `match_prefix` may not reach it until the sequence finishes and its KV
    /// blocks are registered, so that both halves of a hit become valid at the same moment.
    pub fn hold_snapshot(&mut self, id: u64, key: u64) {
        if let Some(seq) = self.running.iter_mut().find(|s| s.id == id) {
            // A sequence may hold FOUR: its tools anchor (end of the tools block; 2026-09-27),
            // its anchor (end of the shared system prefix), its junction (where it left cached
            // history; 2026-09-26) and its end-of-prompt snapshot. All are published when it
            // finishes.
            if let Some((k, at)) = seq.checkpoint_key.filter(|(k, _)| *k == key) {
                // The latest only: an older one stays in the backend's checkpoint pool until that
                // pool's LRU drops it, unpublished.
                seq.pending_checkpoint = Some((k, at));
            } else if seq.tools_anchor_key == Some(key) {
                seq.pending_tools_anchor = Some(key);
            } else if seq.anchor_key == Some(key) {
                seq.pending_anchor = Some(key);
            } else if seq.junction_key == Some(key) {
                seq.pending_junction = Some(key);
            } else {
                seq.pending_snapshot = Some(key);
            }
        }
    }

    /// The backend saved a recurrent-state snapshot under `key` (a `SeqPlan::snapshot_key`).
    pub fn note_snapshot(&mut self, key: u64) {
        self.blocks.note_snapshot(key);
    }

    /// ON-DISK PREFIX CACHE (issue #17): the KV slots of `tokens`' first `at` positions when they
    /// are all published cached blocks (`BlockManager::cached_prefix_slots`) — what a save reads.
    /// `at` is a snapshot boundary, a whole number of blocks.
    pub fn prefix_cached_slots(&self, tokens: &[u32], at: usize) -> Option<Vec<u32>> {
        let bs = self.blocks.block_size();
        if at == 0 || !at.is_multiple_of(bs) {
            return None;
        }
        self.blocks.cached_prefix_slots(tokens, at / bs)
    }

    /// The snapshot key of `tokens`' first `at` positions (`BlockManager::chain_hash`).
    pub fn prefix_key(&self, tokens: &[u32], at: usize) -> u64 {
        self.blocks
            .chain_hash(tokens, at / self.blocks.block_size())
    }

    /// Fresh blocks for a prefix read back from disk — `tokens` must be a whole number of
    /// blocks. Returns a holder and the slots to write the KV into; then
    /// [`prefix_import_commit`](Self::prefix_import_commit), or
    /// [`prefix_import_abort`](Self::prefix_import_abort) if the write failed.
    pub fn prefix_import_begin(&mut self, tokens: Vec<u32>) -> Option<(Sequence, Vec<u32>)> {
        let bs = self.blocks.block_size();
        if tokens.is_empty() || !tokens.len().is_multiple_of(bs) {
            return None;
        }
        let mut seq = Sequence::from_request(Request {
            id: u64::MAX - 64,
            prompt: tokens,
            params: Default::default(),
            image: None,
            prefix_anchor: None,
            tools_anchor: None,
            header_tail: None,
            stop_check: None,
        });
        let slots = self.blocks.import_begin(&mut seq).ok()?;
        Some((seq, slots))
    }

    /// The KV and the snapshot under `key` are written: publish the blocks and `key` as a
    /// shared-prefix snapshot, exactly as an anchor is published when its sequence finishes.
    pub fn prefix_import_commit(&mut self, mut seq: Sequence, key: u64) {
        self.blocks.import_commit(&mut seq, key);
        if self.blocks.has_snapshot(key) {
            self.shared_prefix_keys.insert(key);
        }
    }

    /// The write failed: return the blocks unpublished.
    pub fn prefix_import_abort(&mut self, mut seq: Sequence) {
        self.blocks.free(&mut seq);
    }

    /// The backend dropped the snapshot stored under `key`.
    ///
    /// 2026-09-26: ALSO un-holds it. A snapshot evicted while the sequence that took it is still
    /// running used to be published anyway when that sequence finished (`forget` ran first and
    /// found nothing to remove, then the finish `note`d the key), so a later prompt matched to
    /// a snapshot the backend no longer had — and the serving loop treats that restore miss as
    /// fatal. Reachable once more sequences take snapshots concurrently than the pool holds;
    /// the anchor pool (2 slots by default) makes it easy to reach.
    ///
    /// Returns the running sequence that still HELD the key, if any (issue #12): that
    /// conversation's next turn will not find its snapshot and re-prefills. The serving loop
    /// logs it, so whether this happens under real agent load is counted before anything is
    /// built to prevent it.
    pub fn forget_snapshot(&mut self, key: u64) -> Option<u64> {
        self.blocks.forget_snapshot(key);
        self.shared_prefix_keys.remove(&key);
        let mut holder = None;
        for seq in &mut self.running {
            let held = [
                seq.pending_snapshot,
                seq.pending_anchor,
                seq.pending_tools_anchor,
                seq.pending_junction,
            ]
            .contains(&Some(key));
            if held && holder.is_none() {
                holder = Some(seq.id);
            }
            if seq.pending_snapshot == Some(key) {
                seq.pending_snapshot = None;
            }
            if seq.pending_anchor == Some(key) {
                seq.pending_anchor = None;
            }
            if seq.pending_tools_anchor == Some(key) {
                seq.pending_tools_anchor = None;
            }
            if seq.pending_junction == Some(key) {
                seq.pending_junction = None;
            }
            // A checkpoint dropped from its pool is not a held turn snapshot (issue #12): the
            // sequence just has no resume point to leave until it saves the next one.
            if seq.pending_checkpoint.is_some_and(|(k, _)| k == key) {
                seq.pending_checkpoint = None;
            }
        }
        holder
    }

    /// A snapshot under `key` is published, or held by a running sequence (published when it
    /// finishes). Either way a new one there would duplicate it.
    fn snapshot_known(&self, key: u64) -> bool {
        self.blocks.has_snapshot(key)
            || self.running.iter().any(|s| {
                s.pending_anchor == Some(key)
                    || s.pending_tools_anchor == Some(key)
                    || s.pending_junction == Some(key)
                    || s.pending_snapshot == Some(key)
                    || s.pending_checkpoint.is_some_and(|(k, _)| k == key)
            })
    }

    /// Tokens per KV block — the geometry `build_forward` needs. Exposing it
    /// here keeps the serving loop off backend-specific types .
    pub fn block_size(&self) -> usize {
        self.blocks.block_size()
    }

    /// Commit externally-sampled tokens (one per scheduled sequence, in the
    /// last plan's seq order). The backend-agnostic GPU path: the backend
    /// samples, the scheduler only does bookkeeping .
    pub fn commit_tokens(&mut self, tokens: &[u32]) -> Result<Vec<RequestOutput>> {
        let runs: Vec<Vec<u32>> = tokens.iter().map(|&t| vec![t]).collect();
        self.advance_and_retire(&runs)
    }

    /// Commit a RUN of tokens per scheduled sequence — the speculative case. All but the last
    /// token of a run were verified against the model with their K/V and recurrent state left
    /// in place, so they count as computed; the last is the next input, exactly as a
    /// single-token commit's is. A finish inside the run truncates it there.
    pub fn commit_runs(&mut self, runs: &[Vec<u32>]) -> Result<Vec<RequestOutput>> {
        self.advance_and_retire(runs)
    }

    /// Retire the given sequences NOW (free their KV, drop them) without a
    /// token. For server-side eviction of disconnected/stalled clients
    /// . Must NOT be called between `schedule()` and commit — the
    /// pending plan's row alignment would break. An id that matches nothing
    /// (e.g. a sequence that already finished in this step's commit) is a
    /// silent no-op.
    pub fn evict_seqs(&mut self, ids: &[u64]) {
        assert!(
            self.scheduled.is_empty(),
            "evict_seqs called between schedule() and commit — this would \
             misalign the pending plan's rows"
        );
        for ri in (0..self.running.len()).rev() {
            if ids.contains(&self.running[ri].id) {
                // AN EVICTED SEQUENCE LEAVES ITS ANCHORS (2026-10-07). An anchor was published
                // only when its request FINISHED, so a request whose client gave up took its
                // shared prefix with it — and a client that gives up on a long prompt asks again.
                // Measured through Claude Code 2.1.292 in auto mode: its safety classifier sends a
                // constant ~30,500-token system prompt before each tool call, gives up before a
                // cold read of it ends (276 s on a loaded M4 Max), and every retry read it from
                // the start again (`cached 96`, every time): a session that never ended. What is
                // published is what an early anchor publishes — the KV blocks wholly before the
                // boundary and the snapshot taken there, both final — and this sequence writes
                // nothing more at all.
                // AND ITS END-OF-PROMPT SNAPSHOT AND JUNCTION, when its prefill passed them (later
                // the same day): the classifier's next request is the same prompt plus a few
                // tokens, so the anchor alone left ~15,000 tokens of project context to read again
                // on every tool call (`publish_passed`).
                self.publish_passed(ri);
                let mut seq = self.running.remove(ri);
                seq.status = SequenceStatus::Finished(FinishReason::Evicted);
                self.blocks.free(&mut seq);
            }
        }
        let kept: VecDeque<Sequence> = std::mem::take(&mut self.waiting)
            .into_iter()
            .filter_map(|mut seq| {
                if ids.contains(&seq.id) {
                    self.blocks.free(&mut seq); // no-op for never-allocated seqs
                    None
                } else {
                    Some(seq)
                }
            })
            .collect();
        self.waiting = kept;
    }

    /// Sample one token per scheduled sequence from `logits` (rows align with
    /// the last plan's seq order) and commit them. Sequences that did NOT
    /// complete their pending tokens this step (mid-chunked-prefill) are not
    /// sampled — their row is ignored. On a sampling error the step's plan is
    /// dropped (no sequence advances) and the error is returned — a long-lived
    /// scheduler stays usable.
    pub fn commit(&mut self, logits: &Tensor) -> Result<Vec<RequestOutput>> {
        let mut tokens = Vec::with_capacity(self.scheduled.len());
        let mut failed = None;
        for (si, &(ri, q)) in self.scheduled.iter().enumerate() {
            let seq = &mut self.running[ri];
            if seq.num_computed + q == seq.tokens.len() {
                match seq.sample_next(logits.row(si)) {
                    Ok(token) => tokens.push(token),
                    Err(e) => {
                        failed = Some(e);
                        break;
                    }
                }
            } else {
                tokens.push(0); // mid-prefill: discarded by advance_and_retire
            }
        }
        if let Some(e) = failed {
            // Failure-atomic: drop the plan so a long-lived scheduler is not
            // poisoned (the next schedule() re-plans the same pending tokens).
            self.scheduled.clear();
            return Err(e);
        }
        let runs: Vec<Vec<u32>> = tokens.iter().map(|&t| vec![t]).collect();
        self.advance_and_retire(&runs)
    }

    /// Advance every scheduled sequence by its processed q_len, append the
    /// given token for sequences that completed their pending tokens, retire
    /// finished sequences, and free their blocks. `tokens` aligns with the last
    /// plan's seq order. Tokens for mid-prefill sequences are DISCARDED — a
    /// mid-prefill last-row logit predicts a token already in the prompt.
    fn advance_and_retire(&mut self, runs: &[Vec<u32>]) -> Result<Vec<RequestOutput>> {
        assert_eq!(
            runs.len(),
            self.scheduled.len(),
            "one run per scheduled sequence"
        );
        let mut outputs = Vec::with_capacity(self.scheduled.len());
        let mut finished_idx = Vec::new();

        let scheduled = self.scheduled.clone();
        for (si, &(ri, q)) in scheduled.iter().enumerate() {
            self.running[ri].num_computed += q;
            // Register any blocks this step just completed so later requests
            // with the same prefix reuse their KV (no-op if caching is off).
            // Before the push: a block is full when its tokens are COMPUTED.
            self.blocks.register_full_blocks(&mut self.running[ri]);
            if self.early_anchors {
                self.publish_early_anchors(ri);
            }
            let seq = &mut self.running[ri];
            if seq.num_computed < seq.tokens.len() {
                continue; // still prefilling (or recomputing): no output
            }
            let mut pushed = 0usize;
            for &token in &runs[si] {
                seq.tokens.push(token);
                pushed += 1;
                let reason = finish_reason(seq, token);
                if let Some(reason) = reason {
                    seq.status = SequenceStatus::Finished(reason);
                    finished_idx.push(ri);
                }
                outputs.push(RequestOutput {
                    id: seq.id,
                    token,
                    finished: reason.is_some(),
                    finish_reason: reason,
                });
                if reason.is_some() {
                    break;
                }
            }
            // Every token of the run but the last was verified in place; the last is the next
            // input (or the terminal token). A single-token run leaves this at zero — and
            // registers nothing new, exactly as before runs existed.
            if pushed > 1 {
                seq.num_computed += pushed - 1;
                self.blocks.register_full_blocks(&mut self.running[ri]);
            }
            // STATE SNAPSHOTS — PUBLISH BOTH HALVES TOGETHER. The backend took this snapshot at a
            // block boundary during prefill, but its KV blocks only became shareable in the
            // `register_full_blocks` above (which no-ops until the sequence is complete). Only now
            // are the recurrent state and the KV for those same tokens both final, so only now may
            // a later turn attach to them. Announcing it at capture time is the 2026-09-21 bug:
            // turn 2 diverged because it read KV that was still being written.
            let done = matches!(self.running[ri].status, SequenceStatus::Finished(_));
            if done {
                if let Some(key) = self.running[ri].pending_snapshot.take() {
                    self.blocks.note_snapshot(key);
                }
                // The anchor lies inside the prompt too, so its KV blocks were registered above
                // with the rest: both halves are final now, exactly as for the turn-end one.
                // Anchors and junctions are the shared-prefix class a NEW session resumes from
                // (`shared_prefix_keys`, 2026-09-26: `is_session_start` reads it).
                if let Some(key) = self.running[ri].pending_anchor.take() {
                    self.blocks.note_snapshot(key);
                    self.shared_prefix_keys.insert(key);
                }
                // The tools anchor (2026-09-27) lies before the anchor: the same.
                if let Some(key) = self.running[ri].pending_tools_anchor.take() {
                    self.blocks.note_snapshot(key);
                    self.shared_prefix_keys.insert(key);
                }
                // So does the junction (2026-09-26): it lies before the end-of-prompt boundary.
                if let Some(key) = self.running[ri].pending_junction.take() {
                    self.blocks.note_snapshot(key);
                    self.shared_prefix_keys.insert(key);
                }
                // And the latest rolling checkpoint (2026-10-07).
                if let Some((key, _)) = self.running[ri].pending_checkpoint.take() {
                    self.blocks.note_snapshot(key);
                }
            }
        }
        self.scheduled.clear();

        // Free finished sequences' blocks and drop them (descending index order
        // keeps earlier indices valid).
        finished_idx.sort_unstable_by(|a, b| b.cmp(a));
        for i in finished_idx {
            let mut seq = self.running.remove(i);
            self.blocks.free(&mut seq);
        }
        Ok(outputs)
    }
}

/// Determine whether the just-appended `token` ends the sequence. A stop string is checked
/// before the budget: a token that both completes one and spends the last of `max_tokens` is a
/// stop (as in vLLM, whose stop-string check overrides a length finish).
fn finish_reason(seq: &Sequence, token: u32) -> Option<FinishReason> {
    if seq.params.stop_tokens.contains(&token) {
        Some(FinishReason::Stop)
    } else if seq
        .stop_check
        .as_ref()
        .is_some_and(|c| c.hit(seq.output_tokens()))
    {
        Some(FinishReason::StopString)
    } else if seq.output_tokens().len() >= seq.params.max_tokens {
        Some(FinishReason::Length)
    } else {
        None
    }
}
