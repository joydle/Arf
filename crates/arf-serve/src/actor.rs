//! The model actor: single owner of the resident model, now a CONTINUOUS
//! BATCHING step-loop .
//!
//! ## Shape
//!
//! axum handlers (tokio reactor) build [`Job`]s and send them over a channel.
//! ONE dedicated OS thread owns the model — `GpuModel` is `Send` but `!Sync`,
//! and the GPU is one resource: throughput comes from folding N requests into
//! ONE batched forward pass per step (the weight stream is read once for the
//! whole batch), not from concurrent submitters.
//!
//! Each loop iteration: drain new jobs (non-blocking `try_recv` while busy;
//! blocking `recv` only when fully idle — the the design invariant), run one
//! scheduler step (`schedule -> build_forward -> backend.step ->
//! commit_tokens`), then demux each sequence's token to its request's channel
//! with NON-BLOCKING `try_send` : a full or closed channel evicts
//! that sequence (KV freed by the scheduler) and never stalls the batch.
//!
//! The actor owns a `Box<dyn BatchedBackend>` — no wgpu types appear in this
//! file, so the same loop serves any other backend .

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;

use arf_core::backend::{BatchedBackend, ImageEncoder};
use arf_core::config::EngineConfig;
use arf_core::engine::{build_forward_cached, SlotsCache};
use arf_core::sampling::{
    sample_batch, sample_batch_constrained, Constraint, JsonConstraint, SamplingParams,
    SeqSampling, TokenPieceMap,
};
use arf_core::scheduler::ImagePrompt;
use arf_core::scheduler::{FinishReason as CoreFinish, Request, RequestOutput, Scheduler};

use crate::metrics::Metrics;

mod forced;

/// One unit of work: decode `prompt_ids` under `params`, streaming tokens to
/// `token_tx`. `params` carries max_tokens/stop_tokens/temperature/seed — the
/// backend-agnostic form (no GPU sampling types here).
/// L229 — draft window size. 0 (default) = speculation OFF, so shipping behaviour is
/// unchanged until it is armed.
///
/// L324 — CLAMPED TO `SPEC_K_MAX`. A hybrid verify window is `k + 1` rows and needs one
/// recurrent checkpoint slot per non-final row, so `gdn_prepare_verify` refuses `k > 3`
/// (SPEC_CKPT_ROWS = 3, concurrent_metal.rs). Unclamped, `ARF_SPEC_K=5` parsed fine and
/// then made EVERY verify window bail — speculation silently degraded to plain decode with
/// no message unless ARF_PREFILL_FAST_DEBUG was set. Safe, but silently useless is its own
/// bug. Clamping makes the ceiling visible at the knob instead of deep in the record.
///
/// On the sweet spot: the "k=2 nets 1.64x" claim that used to sit here came from L220 and was
/// SUPERSEDED on 2026-08-31 (measured, L323) — it assumed `1 + k*p`
/// expected tokens (drafts are chained, so it is `1 + p + p^2`) and charged nothing for the
/// verify's measured 19 ms of non-traffic overhead. Corrected projection at p=0.80: k=1 is
/// parity, k=2 is ~1.11x, and k saturates by 3. Do not quote a k here as "measured best"
/// until the owed quiet-box interleaved A/B exists.
pub(crate) const SPEC_K_MAX: usize = 3;

/// PROMPT-LOOKUP SPECULATION (2026-09-20). Single stream is pinned at the bandwidth roofline; the
/// only way past it is more than one token a pass, and the draft that needs no training is a
/// COPY: where the last few tokens occurred earlier in prompt+answer, propose what followed.
/// An 8-row verify window costs ~124 ms against ~48 ms for a plain step, so it pays from 3
/// accepted drafts up and LOSES below that — replayed on real answers (measured,
/// a standalone probe): ungated 0.88x on prose; with this controller
/// 1.95x on a code edit and never under 1.0x. The drafter's own gate (>= 4-token match) decides
/// WHETHER there is a copy; this decides whether copying has been PAYING.
const LOOKUP_K: usize = 7; // drafts per window: 1 + 7 = the 8-row MPP unit
const LOOKUP_MIN_DRAFT: usize = 4; // a shorter copy is not worth a 2.6-step pass
const LOOKUP_PAYS_FROM: usize = 3; // accepted drafts at which a window broke even
const LOOKUP_BACKOFF: usize = 4; // plain steps to sit out after a miss, doubling per miss in a row

// THE BLOCK DRAFT'S PAY-OR-BACK-OFF (DFlash 2). A block cycle is a ~124 ms verify + a ~22 ms
// draft + ~5 ms of commit/restore ~= 150 ms; the fallback it displaces — the MTP head's 2-row
// window — lands ~1.76 tokens in ~82 ms. So a block window pays only from ~2.2 accepted drafts.
// On code, arithmetic and plain prose it lands 3-7; on a model THINKING ALOUD it averaged 2.1 and
// the first head-to-head read 18.6-19.2 tok/s against 20.9 without the draft (2026-09-21). The
// running mean decides; while it is low the head drafts instead — and its windows still feed the
// draft's context ring, so the draft can be probed again at any time.
// 2.2 -> 1.0 ON 2026-09-22: the split-K sg matmul made the 8-row verify pass 27% cheaper, and the
// sweep (`ARF_BLOCK_PAYS_FROM`, 3 interleaved rounds on a quiet box, every arm 598 tokens) says
// free-form 22.8 (2.2) / 24.8 (1.6) / 27.6 (1.0) / 27.3 (no back-off) tok/s with code edit flat
// at ~79 — a block window now pays from ONE accepted draft. The controller stays (it still rests
// the draft on text where even one proposal fails); only its bar moved.
// 1.0 -> 0.0 ON 2026-09-23 (the draft never rests): with q4sg2 the 8-row verify is 85 ms and the
// draft 12 ms (its RMSnorm had been one thread per row), while the 2-row MTP window the backoff
// falls back to still costs ~78 ms for ~1.5 tokens — a block window now pays at ANY acceptance
// the controller could see. Profiled free-form, same prompt, one server at a time: 41.0 tok/s
// never resting (306 windows, acceptance 3.10) vs 37.5 at 1.0 (246 block + 101 MTP windows).
// `ARF_BLOCK_PAYS_FROM=1.0` is the control arm.
const BLOCK_PAYS_FROM: f32 = 0.0; // mean accepted drafts at which a block window breaks even
/// `ARF_BLOCK_PAYS_FROM` overrides the break-even for a sweep (2026-09-22: the 8-row verify pass
/// got 27% cheaper with the split-K sg matmul, so the constant tuned for a 137 ms pass is stale;
/// on prose the controller was sending 171 of 216 windows to the 2-row MTP head). Read once.
fn block_pays_from() -> f32 {
    static V: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("ARF_BLOCK_PAYS_FROM")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(BLOCK_PAYS_FROM)
    })
}
const BLOCK_EMA: f32 = 0.3; // weight of the newest window in the running mean
const BLOCK_EMA_START: f32 = 3.5; // optimistic: a new sequence gets a fair run of windows first
const BLOCK_BACKOFF: usize = 8; // head-drafted steps to sit out, doubling per failed probe

fn lookup_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ARF_NO_PROMPT_LOOKUP").is_none())
}

fn spec_k() -> usize {
    static K: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *K.get_or_init(|| {
        let want = std::env::var("ARF_SPEC_K")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        if want > SPEC_K_MAX {
            eprintln!(
                "[spec] ARF_SPEC_K={want} exceeds the {SPEC_K_MAX} recurrent checkpoint \
                 slots; clamping to {SPEC_K_MAX} (a larger window makes every hybrid verify \
                 bail, which is speculation OFF, not faster)"
            );
            return SPEC_K_MAX;
        }
        want
    })
}

fn spec_dbg() -> bool {
    static D: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *D.get_or_init(|| std::env::var_os("ARF_SPEC_DEBUG").is_some())
}

/// `ARF_MTP_PROBE=1` — run the MTP draft every step, throw it away, and score it against
/// the token the ordinary step then emits. That is the head's top-1 hit rate with NO
/// speculative state interaction at all: the number that says whether the head is right,
/// separately from whether verify/commit is.
fn mtp_probe() -> bool {
    static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *P.get_or_init(|| std::env::var_os("ARF_MTP_PROBE").is_some())
}

fn mtp_probe_observe(draft: u32, actual: Option<u32>) {
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
    static HITS: AtomicUsize = AtomicUsize::new(0);
    static SEEN: AtomicUsize = AtomicUsize::new(0);
    let hit = actual == Some(draft);
    let n = SEEN.fetch_add(1, Relaxed) + 1;
    let h = HITS.fetch_add(hit as usize, Relaxed) + hit as usize;
    if spec_dbg() || n.is_multiple_of(32) {
        eprintln!(
            "[mtp-probe] {} draft={draft} actual={actual:?}  hits {h}/{n} = {:.1}%",
            if hit { "HIT " } else { "miss" },
            100.0 * h as f64 / n as f64
        );
    }
}

pub struct Job {
    /// Unique request id, assigned by the HTTP layer.
    pub id: u64,
    /// Fully-prepared prompt token ids (BOS/template applied by the caller).
    pub prompt_ids: Vec<u32>,
    /// Sampling + stopping spec for this request.
    pub params: SamplingParams,
    /// Per-request token sink. The actor try_sends each token : the
    /// channel's capacity IS the slow-client backlog; overflow or closure
    /// evicts the sequence without stalling the batch.
    pub token_tx: tokio::sync::mpsc::Sender<StreamItem>,
    /// If true, the actor computes per-token logprobs from the forward logits
    /// and attaches them to each `StreamItem::Token`. The non-logprob path is
    /// unchanged (no overhead when this is false).
    pub want_logprobs: bool,
    /// Number of top-alternative logprobs to return (0 = chosen token only).
    pub top_logprobs: usize,
    /// Optional `response_format.type` from the OpenAI request.
    /// `"json_object"` or `"json_schema"` → attach a `JsonConstraint` in the actor.
    /// `None` or `"text"` → unconstrained (default, zero overhead).
    pub response_format: Option<String>,
    /// Vision input (Gemma-3): decoded RGB images `(rgb[h*w*3], w, h)` plus the LOCAL prompt
    /// positions of the `<image_soft_token>` placeholders (256 per image, in image order).
    /// The actor runs the vision pipeline on these (GPU-side) and builds `Request::with_image`.
    /// Empty for text requests.
    pub images: Vec<(Vec<u8>, usize, usize)>,
    /// Flat list of soft-token positions across all images, in image order (256 × n_images).
    pub image_positions: Vec<usize>,
    /// Qwen3.8 vision: the image embeddings ALREADY ENCODED by the HTTP layer (the CPU encoder
    /// runs there, off the actor thread) with their prompt positions and M-RoPE layout. Takes
    /// precedence over `images`; `None` for text requests and for the Gemma-3 path.
    pub image_prompt: Option<ImagePrompt>,
    /// PREFIX ANCHOR (2026-09-26): how many leading `prompt_ids` are the request's shared
    /// "system + tools" prefix (`crate::prefix_anchor`), so the scheduler can snapshot the
    /// recurrent state there for the NEXT session of the same agent. `None` = no anchor (no
    /// system prompt or tools, a prefix under 1,024 tokens, a raw completion, a warm-up job).
    pub prefix_anchor: Option<usize>,
    /// TOOLS ANCHOR (2026-09-27): how many leading `prompt_ids` are the request's tools block
    /// alone, when the template renders it before the system text (`crate::prefix_anchor::
    /// tools_anchor`) — so a new session whose system text differs (Claude Code in another
    /// working directory) still resumes past the tool schemas. `None` = no tools, a template
    /// that renders the system text first, `ARF_NO_TOOLS_ANCHOR=1`, or no `prefix_anchor`.
    pub tools_anchor: Option<usize>,
    /// HEADER TAIL (2026-10-05): how many of the prompt's last tokens the next turn renders
    /// differently (`crate::prefix_anchor::header_tail`), so the end-of-prompt snapshot backs off
    /// only that far. `None` = unknown: the scheduler's fixed tail.
    pub header_tail: Option<usize>,
    /// STOP STRINGS: the request's text stop condition (`crate::stop`), handed to the
    /// scheduler, which retires the sequence at the token that completes a stop string. `None` =
    /// no stop strings.
    pub stop: Option<arf_core::scheduler::StopCheck>,
    /// BACKGROUND (2026-10-08): read this prompt only when nothing else has work, as an
    /// abandoned read is (`Scheduler::set_background`); a request that shares its prefix waits
    /// for it and moves it up. The `x-arf-background: 1` header sets it (`arf launch` reading an
    /// agent's fixed prompts ahead of the first session).
    pub background: bool,
}

/// `Job::top_logprobs` value that asks for EVERY token's logprob, in vocabulary order
/// (`top_logprobs[i] = (i, logprob_i)`, no top-N selection and no sort). The System One
/// score-only job (`http::systemone`) reads its label tokens out of this row. OpenAI requests are
/// clamped to 20 alternatives in the HTTP layer and never ask for it.
pub const ALL_LOGPROBS: usize = usize::MAX;

/// Per-token logprob data attached to `StreamItem::Token` when the request set
/// `logprobs: true`. Computed from the SAME logits row the backend returns.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenLogprob {
    /// The sampled token id.
    pub token_id: u32,
    /// Natural-log probability of the chosen token: logit`chosen` - logsumexp(row).
    pub logprob: f32,
    /// Top-N alternatives sorted by logprob descending (includes the chosen token
    /// when it is in the top-N; matches OpenAI's `top_logprobs` semantics).
    pub top_logprobs: Vec<(u32, f32)>,
}

/// An item the actor streams back for a job.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamItem {
    /// A generated token id — plus optional logprob data when the job requested it.
    Token(u32, Option<Box<TokenLogprob>>),
    /// Generation finished, and why.
    Done(FinishReason),
}

/// Why a generation stopped — maps to the OpenAI `finish_reason` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// Hit the `max_tokens` budget.
    Length,
    /// Sampled a stop / EOS token.
    Stop,
    /// Evicted: the client disconnected or stalled past its backlog.
    Evicted,
}

impl FinishReason {
    /// The OpenAI-compatible `finish_reason` string.
    pub fn as_str(self) -> &'static str {
        match self {
            FinishReason::Length => "length",
            FinishReason::Stop => "stop",
            FinishReason::Evicted => "evicted",
        }
    }
}

/// Handle for submitting jobs; clonable + `Send + Sync` (just a sender), so a
/// router can hold a set of these.
#[derive(Clone)]
pub struct ModelHandle {
    tx: Sender<Job>,
}

impl ModelHandle {
    /// Queue a job. Errors only if the actor thread has died.
    pub fn submit(&self, job: Job) -> Result<(), String> {
        self.tx
            .send(job)
            .map_err(|_| "model actor thread is gone".to_string())
    }

    /// A handle whose jobs land on `tx` instead of an actor: a handler test takes the job and
    /// plays the actor's part (feeds `token_tx`), so the handler runs end to end without a model.
    #[cfg(test)]
    pub(crate) fn from_sender(tx: Sender<Job>) -> Self {
        ModelHandle { tx }
    }
}

/// Spawn the actor on its dedicated OS thread, transferring ownership of the
/// backend. Asserts the backend's KV pool matches the scheduler config — the
/// memory-safety gate from the spec (block ids must never exceed the pool).
///
/// Returns `(ModelHandle, Arc<Metrics>, JoinHandle)`.
/// - `ModelHandle` — the job submission channel (cloneable, `Send + Sync`).
/// - `Arc<Metrics>` — shared with the actor; the async HTTP handler reads it.
///   The actor is the SOLE WRITER (relaxed atomics, no lock); the handler is a
///   pure reader. This preserves the single-owner discipline: no tokio code
///   ever touches the scheduler.
/// - `JoinHandle` — for controlled shutdown.
/// CLEAN SHUTDOWN (2026-10-08): set by the server when it stops; the serving loop leaves after the
/// step in flight, waits for the GPU (`BatchedBackend::quiesce`) and drops the backend on its own
/// thread, which releases the model's resident GPU memory (`ResidencySet::drop`). Before, the
/// process exited around a model thread that could be mid-step, and the kernel tore down ~20 GB
/// of resident memory and sparse mappings with work in flight: a kernel panic in Apple's GPU
/// driver (IOGPUFamily, "Kernel data abort") named a 285 MB `arf-serve` as the panicked task on a
/// 36 GB M4 Max — a process that small is starting or exiting.
pub static SHUTDOWN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// `rx.recv()` for an idle serving loop, waking every 200 ms to see [`SHUTDOWN`]: other threads
/// hold senders, so a plain `recv` would never return at shutdown. `Err` when stopping or when
/// every sender is gone; the loop then returns, as it did on a disconnect.
fn recv_unless_stopping(rx: &Receiver<Job>) -> Result<Job, ()> {
    loop {
        if SHUTDOWN.load(Ordering::Relaxed) {
            return Err(());
        }
        match rx.recv_timeout(std::time::Duration::from_millis(200)) {
            Ok(job) => return Ok(job),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return Err(()),
        }
    }
}

pub fn spawn(
    backend: Box<dyn BatchedBackend>,
    cfg: EngineConfig,
    piece_map: Arc<TokenPieceMap>,
    vision: Option<Box<dyn ImageEncoder>>,
) -> (ModelHandle, Arc<Metrics>, JoinHandle<()>) {
    let (nb, bs) = backend.kv_geometry();
    assert_eq!(
        nb, cfg.num_blocks,
        "backend KV pool ({nb} blocks) != scheduler num_blocks ({})",
        cfg.num_blocks
    );
    assert_eq!(
        bs, cfg.block_size,
        "backend block_size ({bs}) != scheduler block_size ({})",
        cfg.block_size
    );
    let metrics = Arc::new(Metrics::default());
    let metrics_actor = Arc::clone(&metrics);
    let (tx, rx): (Sender<Job>, Receiver<Job>) = std::sync::mpsc::channel();
    let handle = std::thread::Builder::new()
        .name("arf-model-actor".to_string())
        .spawn(move || {
            raise_actor_qos();
            // A PANIC IN THE MODEL THREAD ENDS THE SERVER (2026-10-07). It used to leave a process
            // that still listened and answered `/v1/models` while every request failed with "model
            // actor thread is gone": nothing a client, `arf run` or launchd could tell from a
            // healthy server. The panic message is printed by the default hook before this runs.
            let run = std::panic::AssertUnwindSafe(|| {
                actor_loop(backend, cfg, rx, metrics_actor, piece_map, vision)
            });
            if std::panic::catch_unwind(run).is_err() {
                eprintln!("[serve] the model thread panicked (above); the server exits");
                std::process::exit(70);
            }
        })
        .expect("spawn model actor thread");
    (ModelHandle { tx }, metrics, handle)
}

/// The actor thread encodes every GPU window (~1,100 dispatches of a speculative cycle) and
/// spin-waits on its completion: it IS the GPU's feed. At the default QoS macOS may place it on an
/// efficiency core and preempt it for background work, and the GPU starves — measured 2026-09-26:
/// under load (another process's browser job, load 5-29) Arf fell ~12% below its clean decode while
/// another engine, whose completion work runs at `QOS_CLASS_USER_INITIATED`, held its own. USER_INTERACTIVE
/// keeps the feeder on a performance core. `ARF_NO_ACTOR_QOS=1` leaves the default (the A/B control).
fn raise_actor_qos() {
    #[cfg(target_os = "macos")]
    {
        if std::env::var_os("ARF_NO_ACTOR_QOS").is_some() {
            return;
        }
        extern "C" {
            // <pthread/qos.h>; QOS_CLASS_USER_INTERACTIVE = 0x21
            fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
        }
        // SAFETY: sets the calling thread's own QoS; no pointers, no shared state.
        let rc = unsafe { pthread_set_qos_class_self_np(0x21, 0) };
        eprintln!(
            "[actor] model actor thread QoS: USER_INTERACTIVE{} (ARF_NO_ACTOR_QOS=1 to leave the default)",
            if rc == 0 { "" } else { " — refused, left at the default" }
        );
        if std::env::var_os("ARF_ACTOR_RT").is_some() {
            set_actor_time_constraint();
        }
    }
}

/// ARF_ACTOR_RT=1 (2026-09-29, A/B candidate): put the actor thread in Mach's TIME-CONSTRAINT
/// scheduling class (Core Audio's real-time threads use it). Why: after the decode waits stopped
/// spinning (measured 2026-09-29) the thread is ~8% busy, but under an oversubscribed CPU its
/// WAKE-UPS — twice a speculative window, to read the verify and launch the next draft — queue behind
/// busy work while the GPU idles (Arf lost ~12% under 12 busy loops; another engine ~1%). A time-constraint
/// thread is scheduled ahead of ordinary threads; the kernel demotes it if it overruns its
/// computation budget, so a runaway cannot starve the machine.
#[cfg(target_os = "macos")]
fn set_actor_time_constraint() {
    #[repr(C)]
    struct TimebaseInfo {
        numer: u32,
        denom: u32,
    }
    #[repr(C)]
    struct TimeConstraintPolicy {
        period: u32,
        computation: u32,
        constraint: u32,
        preemptible: i32,
    }
    extern "C" {
        fn mach_thread_self() -> u32;
        fn mach_timebase_info(info: *mut TimebaseInfo) -> i32;
        fn thread_policy_set(
            thread: u32,
            flavor: i32,
            info: *const TimeConstraintPolicy,
            count: u32,
        ) -> i32;
    }
    const THREAD_TIME_CONSTRAINT_POLICY: i32 = 2;
    let mut tb = TimebaseInfo { numer: 0, denom: 0 };
    // SAFETY: fills a plain struct.
    unsafe { mach_timebase_info(&mut tb) };
    let ticks = |ns: u64| -> u32 {
        if tb.numer == 0 {
            return ns as u32;
        }
        (ns * tb.denom as u64 / tb.numer as u64) as u32
    };
    let policy = TimeConstraintPolicy {
        period: 0,                     // aperiodic: woken by GPU completions, not a clock
        computation: ticks(5_000_000), // ~5 ms of CPU a burst (a window's encode + readback)
        constraint: ticks(10_000_000), // ... to be done within 10 ms of becoming runnable
        preemptible: 1,
    };
    // SAFETY: sets the calling thread's own scheduling policy; the struct matches
    // thread_time_constraint_policy_data_t (4 x 32-bit), count 4.
    let rc = unsafe {
        thread_policy_set(
            mach_thread_self(),
            THREAD_TIME_CONSTRAINT_POLICY,
            &policy,
            4,
        )
    };
    eprintln!(
        "[actor] model actor thread: TIME-CONSTRAINT scheduling {} (ARF_ACTOR_RT)",
        if rc == 0 { "ON" } else { "REFUSED" }
    );
}

/// Step 2 of speculative sampling: the block draft is SAMPLED from its own distribution for a
/// sampled request (accept with min(1, p/q)). OPT-IN (`ARF_SAMPLED_DRAFT=1`): MEASURED
/// 2026-09-26 it did not pay — 1.74 accepted drafts a window against the point-mass draft's
/// 1.70 at penalty 1.0 (1.28 vs 1.19 at 1.1), 33.7 / 31.6 vs 34.7 / 33.9 tok/s. The draft's
/// softmax over its candidates is flatter than the target's distribution, and a flat q can
/// accept LESS than a point mass (p = (0.9, 0.1), q = (0.5, 0.5): 0.6 vs 0.9). Shaping q is the
/// open lever. `ARF_NO_SAMPLED_DRAFT` is kept as a no-op.
///
/// CORRECTION 2026-09-26: the "did not pay" verdict above was sample size. On 60 sampled requests
/// (3 prompts x 20 seeds) the sampled draft accepts 3.770 tokens a window vs the point mass's
/// 3.453 (+9.2%; another engine) — but decodes 1-3% SLOWER, because it left the GPU-select path for
/// a waited draft + CPU selector (measured 2026-09-26, "The sampled draft, re-measured
/// properly"). So the sampled draft now runs ON THE GPU SELECTOR (`dflash_chain_sampled`) unless
/// `ARF_SAMPLED_DRAFT_CPU=1` (see `sampled_draft_cpu`). STILL OPT-IN: the GPU chain has not been
/// run live yet, and a q that does not match the draws would bias sampled text silently. When
/// the live gates pass (the `ARF_DFLASH_GPU_SELECT_CHECK=1` parity against the CPU sampled
/// selector, then an interleaved wall-clock A/B vs the point mass with a ledger row), the default
/// flips here — and `ARF_NO_SAMPLED_DRAFT` becomes the opt-out again.
///
/// DEFAULT ON (2026-09-27): the live gates passed — the GPU chain's parity probe (2,000 blocks, 0
/// draft mismatches vs the CPU sampled selector, max |dq| 1.2e-7), the dispatch line, the same
/// text as the CPU selector, and acceptance on 90 fresh sampled requests: 3.834 tokens a window vs
/// the point mass's 3.519 (+9.0%, every prompt 3-9 SE) and another engine's 3.691 on the same
/// requests (measured 2026-09-27). `ARF_NO_SAMPLED_DRAFT=1` = the point mass;
/// `ARF_SAMPLED_DRAFT` (the old opt-in) is kept and changes nothing.
///
/// 2026-09-27 (later): the SAME switch governs the multi-stream cycle's sampled segments
/// (`multi_sampled_draft`), so one variable puts every sampled stream, lone or not, on the point
/// mass.
fn sampled_draft_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let _ = std::env::var_os("ARF_SAMPLED_DRAFT");
        std::env::var_os("ARF_NO_SAMPLED_DRAFT").is_none()
    })
}

/// `ARF_SAMPLED_DRAFT_CPU=1`: the sampled draft takes the CPU selector (`dflash_select`'s sampled
/// branch, the waited draft) even on the GPU-select path — the reference the GPU chain
/// (`dflash_chain_sampled`) is gated against, and the A/B arm for it.
fn sampled_draft_cpu() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var_os("ARF_SAMPLED_DRAFT_CPU").is_some())
}

/// Speculative sampling for a lone sampled stream (see the gate in `actor_loop`). DEFAULT ON;
/// `ARF_NO_SPEC_SAMPLING=1` opts out (the plain sampled path, A/B) — for every sampled stream,
/// lone or among several (see `multi_spec_sampling_on`).
fn spec_sampling_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var_os("ARF_NO_SPEC_SAMPLING").is_none())
}

/// Multi-stream speculation serves SAMPLED streams too (2026-09-27): each sampled segment's
/// verify rows are drawn by its own stream's `RowSampler` (see `multi_spec_step`). Why, as
/// measured on 2026-09-27 (load ~22; measured, the entry for this
/// change): 2-4 concurrent sampled requests ran at 9-18 tok/s AGGREGATE against another engine's
/// 39-47 — below ONE sampled stream (~48) — because the multi-stream gate was greedy-only and a
/// sampled pair took plain one-token steps together. This change was NOT MEASURED when written;
/// MEASURED since (measured 2026-09-27):
/// concurrent sampled requests 13-19 -> 51-56 tok/s aggregate, level with another engine.
/// DEFAULT ON under `spec_sampling_on`; `ARF_NO_MULTI_SPEC_SAMPLING=1` opts out (sampled streams
/// among several then take the plain step together, as before — the A/B control).
fn multi_spec_sampling_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        spec_sampling_on() && std::env::var_os("ARF_NO_MULTI_SPEC_SAMPLING").is_none()
    })
}

/// Whether a verify row's raw ARGMAX is this request's next token: greedy with no repetition
/// penalty (a penalty changes the argmax even at temperature 0). Every other request speculates
/// only through a `RowSampler` (`row_sampler_for`).
fn plain_greedy(params: &SamplingParams) -> bool {
    params.is_greedy() && !params.has_repetition_penalty()
}

/// The sampler a speculative verify draws a (not plain-greedy) stream's rows with: the plain
/// path's own — its params and seed (draws keyed by (seed, absolute position)), and its emitted
/// history, the pending token included, which the repetition penalty reads. ONE builder for the
/// lone stream and every multi-stream segment, so both draw exactly what the plain step would.
/// The draft's q starts empty (the point-mass rule); a sampled draft fills it later.
fn row_sampler_for(j: &JobState) -> arf_core::sampling::RowSampler {
    arf_core::sampling::RowSampler {
        params: j.params.clone(),
        history: j.generated.clone(),
        draft_q: Vec::new(),
    }
}

/// A SAMPLED stream's block draft in a multi-stream cycle (2026-09-27), with the q it was drawn
/// from, for its segment's `RowSampler::draft_q` (see `multi_spec_step`). In order:
///   1. the GPU selector, WAITED (`block_draft_gpu_sampled_waited`: `dflash_chain_sampled`, the
///      lone stream's default draw, read back to the host) — unless `ARF_SAMPLED_DRAFT_CPU=1` or
///      `ARF_NO_DFLASH_GPU_SELECT=1`, as on the lone path;
///   2. the CPU sampled selector (`block_draft_sampled`: the same draws — the GPU chain is gated
///      against it, 2,000 blocks, 0 mismatches) when the GPU chain did not run;
///   3. the point mass (`block_draft`, empty q — the equality rule, exact) when neither ran: a
///      backend without sampled drafts behaves as it did before this existed.
///
/// Whichever runs, the q returned is the one the returned draft was drawn from — the one thing
/// min(1, p/q) needs to be exact.
fn multi_sampled_draft(
    backend: &dyn BatchedBackend,
    tok: u32,
    past_len: usize,
    stream: u64,
    rs: &arf_core::sampling::RowSampler,
) -> (Vec<u32>, Vec<Vec<(u32, f32)>>) {
    let (t, seed) = (rs.params.temperature, rs.params.seed);
    let gpu = std::env::var_os("ARF_NO_DFLASH_GPU_SELECT").is_none() && !sampled_draft_cpu();
    if gpu {
        if let Some(dq) = backend.block_draft_gpu_sampled_waited(tok, past_len, stream, t, seed) {
            static SHOWN: std::sync::Once = std::sync::Once::new();
            SHOWN.call_once(|| {
                eprintln!(
                    "[spec] multi-stream sampled segments draft on the GPU selector (q per segment)"
                )
            });
            return dq;
        }
    }
    if let Some(dq) = backend.block_draft_sampled(tok, past_len, stream, t, seed) {
        static SHOWN: std::sync::Once = std::sync::Once::new();
        SHOWN.call_once(|| {
            eprintln!(
                "[spec] multi-stream sampled segments draft on the CPU sampled selector (q per \
                 segment)"
            )
        });
        return dq;
    }
    (backend.block_draft(tok, past_len, stream), Vec::new())
}

/// Multi-stream speculation (C2). DEFAULT ON since 2026-09-26: it won every round at 2, 3, 4 and
/// 8 streams against the decode round-robin (62.8 / 65.9 / 70.0 / 76.6 vs 44.7 / 46.8 / 53.3 / 68.4
/// tok/s, measured); a lone stream never reaches it. `ARF_NO_MULTI_SPEC=1` opts out (the
/// round-robin then serves two streams); `ARF_MULTI_SPEC` is kept as a no-op.
fn multi_spec_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let _ = std::env::var_os("ARF_MULTI_SPEC");
        std::env::var_os("ARF_NO_MULTI_SPEC").is_none()
    })
}

fn multi_prof() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var_os("ARF_MULTI_PROF").is_some())
}

/// Streams one multi-stream verify serves: the draft ring pool's default.
const MULTI_MAX_STREAMS: usize = 4;

/// Rows of one multi-stream verify record (2026-10-05): 16 — four streams as 7 + 7 + 1 + 1, two as
/// 8 + 8, three as 6 + 5 + 5 (each window still <= 8, the block draft's 7 tokens plus the pending
/// one). It was 8: measured that day, 89% of the 3-row windows of a 3 + 3 + 1 + 1 record were
/// accepted whole — the windows capped the streams, not the acceptance. `ARF_MULTI_ROWS=8` is
/// the control (the backend takes 16 only on its fused / replay path and caps at 8 otherwise).
fn multi_rows() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("ARF_MULTI_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16usize)
            .clamp(4, 16)
    })
}

/// C2 (2026-09-26) — MULTI-STREAM SPECULATION: up to four greedy decode streams verify their
/// windows in ONE 8-row record instead of taking a plain step together. The drafts are the
/// CPU-selected block draft, run one after the other (the GPU-selected one shares a single token
/// window). The shape follows the measured per-position survival on the harness prompts
/// (P(a>=j) = 0.82, 0.61, 0.46 ...; T2 1.82, T3 2.43, T4 2.89 ) with a draft's
/// ~7 ms priced in:
///   2 streams: 4 + 4                3 streams: 3 + 3 + 2 (the 2-row stream rotates)
///   4 streams: 3 + 3 + 1 + 1 — two draft, two verify only their pending token (a plain row);
///              the drafted pair rotates every cycle so no answer falls behind (the harness is
///              paced by its slowest answer)
/// A stream whose draft comes back empty gets one row and its rows go to the others. Each window
/// is clamped to the slots its block table backs. Returns each sequence's accepted run (drafts
/// accepted + greedy's bonus), aligned to `plan.seqs`; `None` = no draft at all, or the backend
/// declined before running anything — the caller takes the plain step.
///
/// SAMPLED STREAMS (2026-09-27): `samplers` (aligned to `plan.seqs`) holds a `RowSampler` for each
/// stream that is not plain greedy. Its segment's rows are then DRAWN by that sampler from its own
/// rows of the record (`BatchedBackend::verify_segments_sampled`), so its run is the accepted
/// drafts plus the draw after them — exactly the lone sampled stream's rule. Its draft was, at
/// first, the same point-mass block draft as a greedy stream's, for which the equality rule is
/// exact and the draws are the plain sampler's own (seed-identical to plain sampling).
///
/// SAMPLED DRAFTS PER SEGMENT (2026-09-27): a sampled stream (temperature > 0) now drafts from the
/// draft's own distribution, as the lone stream has by default since 2026-09-27 (+9% acceptance:
/// 3.834 vs 3.519 tokens a window, measured 2026-09-27) — see `multi_sampled_draft`. The
/// q each draft was drawn from goes into ITS segment's `RowSampler::draft_q` before the record,
/// so that segment's rows accept with min(1, p/q) (`sample_rows_q`) — distribution-exact, not
/// seed-identical to plain sampling (the draft spends its own uniforms). The GPU chain is WAITED
/// per stream: this record takes CPU token windows, and the streams draft one after another into
/// the ONE set of draft block buffers — the point-mass draft waited here too (and ran its CPU
/// selector on top). Greedy streams, and greedy-with-a-penalty ones, keep the point mass
/// (`block_draft`), byte for byte. `ARF_NO_SAMPLED_DRAFT=1` = the point mass for every stream,
/// lone or multi (one switch for both paths). NOT MEASURED live yet.
fn multi_spec_step(
    plan: &arf_core::scheduler::BatchPlan,
    input_ids: &[u32],
    backend: &dyn BatchedBackend,
    tick: usize,
    samplers: &mut [Option<arf_core::sampling::RowSampler>],
) -> Option<Vec<Vec<u32>>> {
    let rows_cap = multi_rows();
    let n = plan.seqs.len();
    let bs = backend.kv_geometry().1;
    // which streams draft this cycle (all of them below 4 streams; a rotating pair at 4)
    // (2026-10-05: all four drafting, 4 + 4 + 4 + 4 rows, measured no better than 7 + 7 + 1 + 1 —
    // 89.9 / 96.8 vs 94.2 / 95.4 tok/s — for twice the draft time a cycle: 30 vs 16 ms)
    let drafted = |i: usize| n < 4 || (i + tick).is_multiple_of(2);
    // ARF_MULTI_PROF=1: wall time of each draft and of the verify record (no queue fence, unlike
    // ARF_SPEC_PROF — the shares of a multi-stream cycle, rule 10)
    let prof = multi_prof();
    let mut draft_ms: Vec<f64> = Vec::new();
    let t_cycle = std::time::Instant::now();
    let drafts: Vec<Vec<u32>> = plan
        .seqs
        .iter()
        .zip(samplers.iter_mut())
        .enumerate()
        .map(|(i, (sp, rs))| {
            if drafted(i) {
                let t = std::time::Instant::now();
                let tok = input_ids[sp.q_start];
                // a SAMPLED stream's draft comes with its q, which its segment's sampler must
                // carry into the record (min(1, p/q)); every other stream drafts the point mass
                let d = match rs
                    .as_mut()
                    .filter(|r| sampled_draft_on() && !r.params.is_greedy())
                {
                    Some(r) => {
                        let (d, q) = multi_sampled_draft(backend, tok, sp.past_len, sp.id, r);
                        r.draft_q = q;
                        d
                    }
                    None => backend.block_draft(tok, sp.past_len, sp.id),
                };
                if prof {
                    draft_ms.push(t.elapsed().as_secs_f64() * 1e3);
                }
                d
            } else {
                Vec::new()
            }
        })
        .collect();
    let with: Vec<usize> = (0..n).filter(|&i| !drafts[i].is_empty()).collect();
    if with.is_empty() {
        return None;
    }
    // rows: 1 for every stream without a draft, the rest split over the drafted ones, the
    // remainder going to a rotating subset (3 streams: 3 + 3 + 2 with the short one rotating)
    let spare = rows_cap - (n - with.len());
    let (base, extra) = (spare / with.len(), spare % with.len());
    let mut want = vec![1usize; n];
    for (k, &i) in with.iter().enumerate() {
        let rot = (k + with.len() - tick % with.len()) % with.len();
        want[i] = base + usize::from(rot < extra);
    }
    let mut windows: Vec<Vec<u32>> = Vec::with_capacity(n);
    let mut slots: Vec<Vec<u32>> = Vec::with_capacity(n);
    for ((sp, d), want) in plan.seqs.iter().zip(&drafts).zip(want) {
        let backed = (sp.block_table.len() * bs).saturating_sub(sp.past_len);
        let w = want.min(1 + d.len()).min(backed).max(1);
        let mut win = Vec::with_capacity(w);
        win.push(input_ids[sp.q_start]);
        win.extend_from_slice(&d[..w - 1]);
        slots.push(arf_core::cache::slots_for(
            &sp.block_table,
            bs,
            sp.past_len + w,
        ));
        windows.push(win);
    }
    let segs: Vec<arf_core::backend::SegReq<'_>> = plan
        .seqs
        .iter()
        .zip(windows.iter().zip(&slots))
        .map(|(sp, (w, sl))| arf_core::backend::SegReq {
            stream: sp.id,
            window: w,
            prefix_len: sp.past_len,
            slots: sl,
        })
        .collect();
    let t_verify = std::time::Instant::now();
    // No sampled stream: the greedy record, the same call as before sampled streams could come.
    let preds = if samplers.iter().all(Option::is_none) {
        backend.verify_segments(&segs)?
    } else {
        let refs: Vec<Option<&arf_core::sampling::RowSampler>> =
            samplers.iter().map(Option::as_ref).collect();
        let p = backend.verify_segments_sampled(&segs, &refs)?;
        // once per segment count (rule 7: shown to have run, at each width)
        static SHOWN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let bit = 1u32 << n.min(31);
        if SHOWN.fetch_or(bit, Ordering::Relaxed) & bit == 0 {
            eprintln!(
                "[spec] multi-stream speculation serves sampled streams ({n} segments, rows drawn \
                 by each stream's sampler) — {} of them sampled",
                refs.iter().filter(|s| s.is_some()).count()
            );
        }
        p
    };
    if preds.len() != n || preds.iter().zip(&windows).any(|(p, w)| p.len() != w.len()) {
        // the record ran and advanced every stream: a malformed answer cannot be declined
        panic!(
            "multi-stream verify: predictions {:?} for windows {:?}",
            preds.iter().map(Vec::len).collect::<Vec<_>>(),
            windows.iter().map(Vec::len).collect::<Vec<_>>()
        );
    }
    if prof {
        eprintln!(
            "[multi-prof] B={n} rows={:?} drafts={:?} ms verify={:.2} ms cycle={:.2} ms",
            windows.iter().map(|w| w.len()).collect::<Vec<_>>(),
            draft_ms
                .iter()
                .map(|x| (x * 100.0).round() / 100.0)
                .collect::<Vec<_>>(),
            t_verify.elapsed().as_secs_f64() * 1e3,
            t_cycle.elapsed().as_secs_f64() * 1e3
        );
    }
    let runs: Vec<Vec<u32>> = windows
        .iter()
        .zip(&preds)
        .map(|(w, p)| arf_core::model::speculative::accepted_run(w, p))
        .collect();
    if spec_dbg() {
        eprintln!(
            "[spec-multi] windows {:?} accepted {:?}",
            windows.iter().map(|w| w.len()).collect::<Vec<_>>(),
            runs.iter()
                .map(|r| r.len().saturating_sub(1))
                .collect::<Vec<_>>()
        );
        for ((sp, w), p) in plan.seqs.iter().zip(&windows).zip(&preds) {
            eprintln!(
                "[spec-multi-seg] id={} pos={} window={w:?} preds={p:?}",
                sp.id, sp.past_len
            );
        }
    }
    Some(runs)
}

/// DECODE ROUND-ROBIN (2026-09-26). Speculation serves one sequence a step, so two greedy streams
/// decoding together used to take PLAIN steps: one token each per ~56 ms record, 35.6 tok/s — the
/// same-session head-to-head had another engine at 61.3 there. With a context ring per stream (C1) a
/// stream can wait a step without losing its draft, so give the two streams one SPECULATIVE window
/// each, in turn: a ~65 ms cycle yields 1 + the accepted drafts, the solo rate (~48.6 on the
/// harness prompts). Only at exactly two decode streams, nothing prefilling or waiting, both
/// greedy and unconstrained, both drafts ready, and while the mean accepted drafts per window
/// beat what two plain rows give: (ema + 1) / 65 ms > 2 / 56.2 ms, i.e. ema > ~1.3
/// (`ARF_DECODE_RR_FROM`). Below it the pair decodes plain, and after `RR_REPROBE` plain steps
/// the means are lifted just above the bar to re-probe (they only learn in speculative windows).
/// At three streams one plain record (3 tokens / ~58 ms) already beats a solo window. Multi-
/// stream verify (C2) is what beats both. `ARF_NO_DECODE_RR=1` opts out (A/B).
#[derive(Default)]
struct DecodeRr {
    last: Option<u64>,
    plain_steps: usize,
}

const RR_REPROBE: usize = 32;

fn decode_rr_from() -> f32 {
    static V: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("ARF_DECODE_RR_FROM")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.3)
    })
}

/// DECODE SHARE (2026-10-07): what a prompt being read owes the sequences decoding beside it.
/// One step could carry 4,096 prompt tokens (~24 s on an M4 Max with the 27B) and the decoders
/// got one token each per step. While anything decodes, a step now reads at most
/// `ARF_DECODE_QUANTUM` (128) prompt tokens, and then the decoders get decode-only steps for
/// `ARF_DECODE_SHARE` (0.5) of that step's wall time. `ARF_DECODE_SHARE=0` restores the old plan.
///
/// THE QUANTUM IS 128, NOT THE 256-ROW PREFILL WINDOW (2026-10-07, the 27B, one M4 Max under
/// other load, a stream beside a 7,628-token prompt, 5 interleaved rounds): in 4 of them the
/// stream's longest gap was 1.22-1.58 s at 128 against 2.20-5.34 s at 256 and its rate during the
/// read higher (10.2-13.3 against 4.9-10.3 chunks/s), the prompt ~9% slower (74 against 81-82
/// tok/s in the two steady rounds). One round went the other way (an 8.83 s gap at 128, with the
/// stream's solo rate far off the others' — the machine, by that sign, but not re-measured quiet).
/// The gaps at 128 are more even: p95 ~1.0 s against 0.1-1.4 s.
/// ⛔ 64 MEASURED-OUT (2 rounds): the longest gap stays at 1.15-1.41 s and the prompt reads 54-62
/// tok/s against 84-87 — a read step has a floor near a second that a smaller chunk does not
/// remove. Getting under a second needs the step itself shorter, not fewer rows in it.
#[derive(Default)]
struct DecodeShare {
    /// Decode-only time still owed.
    debt: std::time::Duration,
    /// The cap handed to the scheduler for the step in flight.
    cap: Option<usize>,
}

/// The most a debt accumulates to: decoders that cannot use their share (a stream that ends)
/// must not hold the prompt back afterwards.
const DECODE_DEBT_MAX: std::time::Duration = std::time::Duration::from_secs(2);

fn decode_share() -> f64 {
    static V: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("ARF_DECODE_SHARE")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|v: &f64| v.is_finite() && *v >= 0.0)
            .unwrap_or(0.5)
    })
}

fn decode_quantum() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("ARF_DECODE_QUANTUM")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|v| *v >= 1)
            .unwrap_or(128)
    })
}

impl DecodeShare {
    /// The prefill cap of the next plan: `None` when nothing decodes beside a prompt.
    fn next_cap(&mut self, sched: &Scheduler) -> Option<usize> {
        self.cap = None;
        if decode_share() == 0.0 || !sched.has_prefill_work() || sched.decode_seqs().is_empty() {
            self.debt = std::time::Duration::ZERO;
            return None;
        }
        self.cap = Some(if self.debt.is_zero() {
            decode_quantum()
        } else {
            0
        });
        self.cap
    }

    /// Account one finished step: a step that read prompt tokens adds to the debt, a decode-only
    /// step pays it.
    fn observe(&mut self, prompt_tokens: usize, took: std::time::Duration) {
        match self.cap {
            Some(0) => self.debt = self.debt.saturating_sub(took),
            Some(_) if prompt_tokens > 0 => {
                self.debt = (self.debt + took.mul_f64(decode_share())).min(DECODE_DEBT_MAX);
            }
            _ => {}
        }
    }
}

impl DecodeRr {
    /// The one decode sequence to plan this step, or `None` for the ordinary plan.
    fn pick(
        &mut self,
        sched: &Scheduler,
        jobs: &HashMap<u64, JobState>,
        backend: &dyn BatchedBackend,
    ) -> Option<u64> {
        static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *OFF.get_or_init(|| std::env::var_os("ARF_NO_DECODE_RR").is_some())
            || multi_spec_on()
            || spec_k() == 0
            || sched.has_prefill_work()
        {
            return None;
        }
        let d = sched.decode_seqs();
        if d.len() != 2 {
            return None;
        }
        let mut ema = 0.0f32;
        for &(id, past) in &d {
            let j = jobs.get(&id)?;
            if !j.params.is_greedy()
                || j.params.has_repetition_penalty()
                || j.want_logprobs
                || j.constraint.is_some()
                || !backend.block_draft_ready(id, past)
            {
                return None;
            }
            ema += j.block_ema.get() / 2.0;
        }
        if ema < decode_rr_from() {
            self.plain_steps += 1;
            if self.plain_steps >= RR_REPROBE {
                self.plain_steps = 0;
                for &(id, _) in &d {
                    if let Some(j) = jobs.get(&id) {
                        j.block_ema.set(decode_rr_from() + 0.4);
                    }
                }
            }
            return None;
        }
        self.plain_steps = 0;
        let pick = if self.last == Some(d[0].0) {
            d[1].0
        } else {
            d[0].0
        };
        self.last = Some(pick);
        static SHOWN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !SHOWN.swap(true, std::sync::atomic::Ordering::Relaxed) || spec_dbg() {
            eprintln!(
                "[spec] decode round-robin: 2 streams {:?}, mean accepted {ema:.2} -> seq {pick}",
                d.iter().map(|x| x.0).collect::<Vec<_>>()
            );
        }
        Some(pick)
    }
}

/// Per-request state the actor tracks between steps.
struct JobState {
    params: SamplingParams,
    token_tx: tokio::sync::mpsc::Sender<StreamItem>,
    want_logprobs: bool,
    top_logprobs: usize,
    /// Grammar constraint for `response_format` JSON modes.
    /// `None` = unconstrained (default, no overhead on the sampling path).
    constraint: Option<Box<dyn Constraint>>,
    /// Tokens emitted so far for this sequence (post-prompt). Fed to the
    /// constraint's state machine at each decode step so it can track JSON structure.
    generated: Vec<u32>,
    /// `Job::background`.
    background: bool,
    /// L229 — speculative drafting state: the realized token history (prompt + emitted)
    /// this sequence has produced, and the suffix index built over it. Only populated when
    /// speculation is armed for the job, so a non-speculative request pays nothing.
    spec_history: Vec<u32>,
    spec_drafter: Option<arf_core::model::speculative::SuffixDrafter>,
    /// Prompt-lookup controller: plain steps left before a lookup window may fire again, and how
    /// many lookup windows in a row failed to pay. See `LOOKUP_*`.
    /// `Cell`s: the step holds shared borrows of every job (the sampling params) while it
    /// speculates, and this is bookkeeping, not state anyone else reads.
    lookup_cooldown: std::cell::Cell<usize>,
    lookup_misses: std::cell::Cell<u32>,
    /// Running mean of drafts accepted per BLOCK window, steps left to sit out, and failed
    /// probes in a row. See `BLOCK_*`.
    block_ema: std::cell::Cell<f32>,
    block_cooldown: std::cell::Cell<usize>,
    block_misses: std::cell::Cell<u32>,
    /// EXPERIMENT (`ARF_BLOCK_AFTER_FULL`): drafted tokens accepted by the previous block window.
    block_last_accepted: std::cell::Cell<usize>,
    /// The per-request `[done]` line (`ReqStats`).
    stats: ReqStats,
}

/// One line a finished request (2026-09-27): what a user of an agent feels, per turn — prompt
/// tokens, how many of them came from the prefix cache, output tokens, time to first token and
/// the decode rate after it. another engine prints the same per request; without it an agent session's
/// slow turn cannot be told from a long answer. `ARF_NO_REQUEST_LOG=1` = silent.
struct ReqStats {
    t0: std::time::Instant,
    prompt: usize,
    /// Tokens of the prompt the first step started past (a prefix-cache / snapshot resume).
    cached: Option<usize>,
    first: Option<std::time::Instant>,
    out: usize,
    /// Read as background work (`Job::background`): the line says so, so a warm-up's read can be
    /// told from a user's.
    background: bool,
    /// Steps that gave this request tokens, and the last of them (`STEP`): output / passes is
    /// the tokens a verify pass yields — 1.00 without speculation, the draft's acceptance with it.
    passes: usize,
    last_step: u64,
}

/// The step the serving loop is committing (one per `commit_runs`), for [`ReqStats::passes`].
static STEP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl ReqStats {
    fn new(prompt: usize, background: bool) -> Self {
        ReqStats {
            t0: std::time::Instant::now(),
            prompt,
            cached: None,
            first: None,
            out: 0,
            background,
            passes: 0,
            last_step: u64::MAX,
        }
    }
    fn token(&mut self) {
        self.out += 1;
        self.first.get_or_insert_with(std::time::Instant::now);
        let step = STEP.load(Ordering::Relaxed);
        if step != self.last_step {
            self.last_step = step;
            self.passes += 1;
        }
    }
    fn report(&self, why: &str) {
        static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *OFF.get_or_init(|| std::env::var_os("ARF_NO_REQUEST_LOG").is_some()) {
            return;
        }
        let now = std::time::Instant::now();
        let ttft = self
            .first
            .map(|f| (f - self.t0).as_secs_f64())
            .unwrap_or(0.0);
        let dec = match self.first {
            Some(f) if self.out > 1 => (self.out - 1) as f64 / (now - f).as_secs_f64().max(1e-9),
            _ => 0.0,
        };
        // Tokens a pass: the first pass is the prompt's (one token), so it is left out.
        let per_pass = match self.passes {
            p if p > 1 => (self.out - 1) as f64 / (p - 1) as f64,
            _ => 1.0,
        };
        eprintln!(
            "[done] input {} · cached {} · output {} · TTFT {ttft:.2} s · decode {dec:.1} tok/s · {per_pass:.2} tokens a pass · wall {:.2} s · {why}{}",
            self.prompt,
            self.cached.unwrap_or(0),
            self.out,
            (now - self.t0).as_secs_f64(),
            if self.background { " · background" } else { "" }
        );
    }
}

/// The step loop. Exits when the job channel closes (all handles dropped) and
/// no work remains, or on a backend/scheduler error (fatal: clients see their
/// streams close).
/// Where an anchor snapshot came from: the sequence that took it and the prompt tokens that
/// followed the anchor in it (up to [`ANCHOR_TAIL_CAP`]) — what tells a later resume from it
/// apart as a NEW session or the same conversation ([`anchor_resume_is_new_session`]).
struct AnchorOrigin {
    seq: u64,
    tail: Vec<u32>,
    /// The TOOLS anchor (end of the tools block, `SeqPlan::snapshot_tools_anchor`, 2026-09-27)
    /// rather than the end of the system text — only the log lines differ.
    tools: bool,
}

/// Prompts with at least this many tokens left to read get progress lines and a status entry.
const PROGRESS_MIN: usize = 4096;
/// How often a long read logs its progress.
const PROGRESS_EVERY: std::time::Duration = std::time::Duration::from_secs(10);

/// `12345` -> `12,345`.
fn thousands(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// After a committed step: refresh `/v1/arf/status`'s list of long prompts being read, and log a
/// progress line for each one every [`PROGRESS_EVERY`] (`reading` keeps when each was first seen).
fn report_prefill_progress(
    sched: &Scheduler,
    reading: &mut HashMap<u64, (std::time::Instant, usize, std::time::Instant)>,
    metrics: &Metrics,
) {
    let progress = sched.prefill_progress();
    if progress.is_empty() && reading.is_empty() {
        return;
    }
    let now = std::time::Instant::now();
    reading.retain(|id, _| progress.iter().any(|p| p.0 == *id));
    let mut statuses = Vec::new();
    for &(id, done, total) in &progress {
        let (t0, first, last) = reading.entry(id).or_insert((now, done, now));
        if total.saturating_sub(*first) < PROGRESS_MIN {
            continue;
        }
        let secs = now.duration_since(*t0).as_secs_f64();
        let rate = if secs >= 1.0 {
            done.saturating_sub(*first) as f64 / secs
        } else {
            0.0
        };
        let eta = if rate > 0.0 {
            (total - done) as f64 / rate
        } else {
            0.0
        };
        if now.duration_since(*last) >= PROGRESS_EVERY {
            *last = now;
            eprintln!(
                "[prefill] {}: {} / {} prompt tokens · {rate:.0} tok/s · ~{} left",
                if id >= crate::metrics::INTERNAL_IDS {
                    "re-reading the last agent prompt (background)".to_string()
                } else {
                    format!("request {id}")
                },
                thousands(done),
                thousands(total),
                if eta >= 90.0 {
                    format!("{:.1} min", eta / 60.0)
                } else {
                    format!("{eta:.0} s")
                }
            );
        }
        statuses.push(crate::metrics::PrefillStatus {
            id,
            done,
            total,
            cached: *first,
            tok_per_s: rate,
            eta_s: eta,
            replay: id >= crate::metrics::INTERNAL_IDS,
        });
    }
    *metrics.prefills.lock().unwrap() = statuses;
}

/// Prompt tokens kept per anchor for that comparison: a first user message plus the assistant
/// header, generously. Bounded by the anchor pool's size (four by default since 2026-09-27).
const ANCHOR_TAIL_CAP: usize = 4096;

/// How many trailing tokens of the anchor-taker's prompt a LATER TURN of the same conversation
/// may render differently: the assistant header and a thinking model's `<think>` cue, which the
/// next turn replaces with the answer (the same reason the end-of-prompt snapshot backs off one
/// window). Generous on purpose: a new session whose first message differs only in its last few
/// tokens is then called the same conversation — the log under-claims, never over-claims.
const ANCHOR_SAME_CONVERSATION_SLACK: usize = 16;

fn common_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// A sequence resuming from an anchor is a NEW session when its tokens after the anchor
/// (`after`) leave the anchor-taker's (`tail`) before the part of `tail` a next turn could
/// re-render: a new session diverges inside its first user message; the same conversation
/// follows the whole earlier prompt except its assistant header.
fn anchor_resume_is_new_session(tail: &[u32], after: &[u32]) -> bool {
    common_len(tail, after) + ANCHOR_SAME_CONVERSATION_SLACK < tail.len()
}

fn actor_loop(
    backend: Box<dyn BatchedBackend>,
    cfg: EngineConfig,
    rx: Receiver<Job>,
    metrics: Arc<Metrics>,
    piece_map: Arc<TokenPieceMap>,
    vision: Option<Box<dyn ImageEncoder>>,
) {
    // L13 PROOF INSTRUMENTATION (ARF_MBS_PROBE=1) — the "--max-batch-size moves B=1" dig.
    // A measured 33.9 (mbs=8) vs 41.8 (mbs=64) tok/s on IDENTICAL single-stream deep work
    // implied something on the B=1 path was keyed off max_batch_size. This one-shot line dumps
    // the ENTIRE EngineConfig the actor was built with, so the two arms can be diffed field by
    // field instead of assumed equal. (Same discipline as L12's `[m1-attn] LIVE BRANCH`: L9 lost
    // a whole strike optimizing a path that was never live — prove the divergence, then edit.)
    // STATIC RESULT this probe corroborates: `max_batch_size` is consumed at EXACTLY ONE site in
    // the engine — `Scheduler::admit_waiting` (scheduler/scheduler.rs:176, `while
    // self.running.len() < self.cfg.max_batch_size`). It is NEVER handed to the GPU backend:
    // arf-serve/src/main.rs:149 passes only `engine_cfg.num_blocks` into `load_resident`, and
    // every GPU scratch/arena is sized from `total` (the ACTUAL batch rows this step), not from
    // any configured maximum. At B=1 the admission predicate is `1 < 8` vs `1 < 64` — both true,
    // same plan, same kernels. So mbs CANNOT alter the B=1 code path by construction.
    if std::env::var_os("ARF_MBS_PROBE").is_some() {
        eprintln!(
            "[l13-mbs] ENGINE CONFIG: max_batch_size={} num_blocks={} block_size={} \
             max_prefill_tokens={} kv_quant={:?} prefix_cache={}",
            cfg.max_batch_size,
            cfg.num_blocks,
            cfg.block_size,
            cfg.max_prefill_tokens,
            cfg.kv_quant,
            cfg.enable_prefix_cache,
        );
    }
    let mbs_probe = std::env::var_os("ARF_MBS_PROBE").is_some();
    let cfg_mbs = cfg.max_batch_size;
    let mut mbs_probe_steps = 0usize;
    let mut sched = Scheduler::new(cfg);
    let mut jobs: HashMap<u64, JobState> = HashMap::new();
    let greedy_only = backend.greedy_only();
    let mut rr = DecodeRr::default();
    let mut share = DecodeShare::default();
    let mut multi_tick = 0usize;
    // Finishing seqs whose Done couldn't be delivered yet (client momentarily
    // full). Flushed opportunistically each iteration — guarantees every stream
    // gets its terminator WITHOUT the loop ever blocking on a slow client.
    let mut pending_close: Vec<(tokio::sync::mpsc::Sender<StreamItem>, FinishReason)> = Vec::new();
    // ARF_ACTOR_PROF=1 — hoisted out of the step loop (L3: it was a per-token getenv syscall
    // on the hot path). Process-constant, like every other ARF_* flag.
    let actor_prof = std::env::var_os("ARF_ACTOR_PROF").is_some();
    // L3 — per-seq slots cache: build_forward_cached extends each running seq's slots vec by
    // q_len per step (O(1) amortized) instead of rebuilding the O(ctx) vec every decode token;
    // reclaim() below hands the vecs back after the forward. See SlotsCache docs for the
    // byte-identity guard. Predicted (2026-08-04): +0.3-1.5ms/tok, growing
    // with depth — NOT yet measured; perf measurement gated on the post-reboot clean race
    // (ARF_ACTOR_PROF=1's build= line measures it for free).
    let mut slots_cache = SlotsCache::default();
    // PREFIX ANCHORS (2026-09-26): the anchor snapshots the backend holds (added on save, dropped
    // when evicted), so a restore can say it resumed from one — the log line that proves a new
    // session was served from the shared system prefix, not merely that an anchor was planned.
    //
    // **Corrected 2026-09-26 (review):** a restore of an anchor key is NOT by itself a new
    // session — turn 2 of the conversation that took the anchor restores it too (when the anchor
    // was that turn's only snapshot, or its turn-end snapshot was evicted) — and the line was
    // once per key, so a genuine new-session resume after such a restore was never logged. So
    // each anchor now remembers the prompt tokens that FOLLOWED it in the sequence that took it
    // (`AnchorOrigin`), and a resume is classified against them (`anchor_resume_is_new_session`):
    // a NEW session is logged every time (one line per new agent session — that line is the
    // evidence the 88-90 s fix ran), the same conversation once per key.
    let mut anchor_keys: HashMap<u64, AnchorOrigin> = HashMap::new();
    // Set once a step has committed: the recurrent banks a loaded prefix state needs exist then.
    let mut ran_a_step = false;
    // PROMPT READING PROGRESS (2026-10-06): an agent's first message can be 40-50K tokens —
    // minutes of prefill that look like a hang from the client (a teammate's M5 took ~5 min). Each
    // prompt with PROGRESS_MIN or more tokens to read gets a log line every PROGRESS_EVERY and its
    // numbers in `/v1/arf/status`. Per sequence: (first seen, tokens computed then, last line).
    let mut reading: HashMap<u64, (std::time::Instant, usize, std::time::Instant)> = HashMap::new();
    let mut anchor_resumes_said: std::collections::HashSet<u64> = std::collections::HashSet::new();
    // JUNCTION SNAPSHOTS (2026-09-26): the junction snapshots the backend holds (added on save,
    // dropped when evicted), so a restore can say it resumed from one — the line that proves a
    // session skipped the shared prefix PAST the anchor. Once per key (every time under
    // ARF_SPEC_DEBUG).
    let mut junction_keys: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut junction_resumes_said: std::collections::HashSet<u64> =
        std::collections::HashSet::new();
    // SESSION-START SNAPSHOTS (2026-09-26): the first-turn end-of-prompt snapshots the backend
    // holds in its junction pool (added on save, dropped when evicted), so a restore can say it
    // resumed from one — the line that proves a session that opened the same way as an earlier
    // one was not re-prefilled down to the anchor. Kept apart from `junction_keys`: turn 2 of the
    // conversation restores its own session-start snapshot too, which is not "an earlier prompt
    // left cached history". The anchor log's once-per-key slip (its "Corrected 2026-09-26
    // (review)" above) would recur here — turn 2 restores the key first and would spend the
    // line — so each key remembers `(seq, prompt length)` of the prompt that took it: a resuming
    // prompt NO LONGER than that one ends where the first prompt ended (a new session that
    // opened the same way) and is logged every time; a longer one (the next turn, or a new
    // session that goes on past it) once per key (every time under ARF_SPEC_DEBUG).
    let mut session_start_keys: HashMap<u64, (u64, usize)> = HashMap::new();
    let mut session_start_resumes_said: std::collections::HashSet<u64> =
        std::collections::HashSet::new();
    // L57: FULL-CYCLE timer. `[actor-prof] step=` measures only the GPU forward and
    // stops BEFORE the emit path (detokenize / SSE serialize / channel send / commit_tokens /
    // metrics). At conc1 the GPU step is 11.3-11.5ms while the client sees 12.2ms/token — so
    // ~0.9ms/token lives in the part the old probe could not see, and the whole gap to llama
    // (~1.1ms/token) is that size. This measures the gap directly instead of guessing.
    let mut _cycle_prev: Option<std::time::Instant> = None;
    loop {
        if SHUTDOWN.load(Ordering::Relaxed) {
            backend.quiesce();
            eprintln!("[serve] model thread: GPU idle, releasing its memory");
            return; // drops the backend here, on this thread
        }
        // Opportunistically flush undelivered terminators (non-blocking). Retain
        // those still full; drop those delivered OR whose channel closed (client
        // gone — nothing to deliver to).
        pending_close.retain(|(tx, reason)| {
            match tx.try_send(StreamItem::Done(*reason)) {
                Ok(()) => false,                                                 // delivered → drop
                Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => true,    // retry later
                Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => false, // client gone → drop
            }
        });

        // ON-DISK PREFIX CACHE (`prefix_disk`): idle is when saved prefix states load (once the
        // warm-up's first step has built the recurrent banks) and published anchors are saved.
        // Idle includes "background reads only" (`Job::background`, or a client that left): in
        // the 2026-10-08 release test the background read of Claude Code's check kept the server
        // from ever being idle, so nothing was saved, and after a restart the 47,551-token first
        // message was read from 0 (281 s against 57 s). The export and the import each wait for
        // the step in flight (`dflash_fence`) and touch only published slots.
        if !sched.has_unfinished()
            || jobs
                .values()
                .all(|j| j.background || j.token_tx.is_closed())
        {
            for l in crate::prefix_disk::on_idle(&mut sched, backend.as_ref(), ran_a_step) {
                anchor_keys.insert(
                    l.key,
                    AnchorOrigin {
                        seq: u64::MAX,
                        tail: Vec::new(),
                        tools: l.tools,
                    },
                );
            }
        }

        // the design recv discipline: block only when fully idle.
        // When idle but pending_close is non-empty, use recv_timeout(1ms) so
        // the loop keeps spinning to retry the flush without busy-spinning. Likewise while a
        // prefix state waits to be saved (`prefix_disk::exports_ready`): one is written a pass.
        if !sched.has_unfinished() {
            if pending_close.is_empty() && !crate::prefix_disk::exports_ready(&sched) {
                match recv_unless_stopping(&rx) {
                    Ok(job) => admit(
                        &mut sched,
                        &mut jobs,
                        &metrics,
                        job,
                        &piece_map,
                        vision.as_deref(),
                        greedy_only,
                    ),
                    Err(_) => {
                        backend.quiesce(); // stopping, or all senders gone
                        return;
                    }
                }
            } else {
                // Idle but terminators outstanding — short timed wait so we
                // retry the flush on the next iteration without blocking forever.
                match rx.recv_timeout(std::time::Duration::from_millis(1)) {
                    Ok(job) => admit(
                        &mut sched,
                        &mut jobs,
                        &metrics,
                        job,
                        &piece_map,
                        vision.as_deref(),
                        greedy_only,
                    ),
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue, // loop → retry flush
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return,
                }
            }
        }
        while let Ok(job) = rx.try_recv() {
            admit(
                &mut sched,
                &mut jobs,
                &metrics,
                job,
                &piece_map,
                vision.as_deref(),
                greedy_only,
            );
        }

        // ABANDONED READS (2026-10-07). A request whose client has gone (its receiver is dropped)
        // was read to the end like any other, ahead of requests someone was waiting for, and its
        // state was then thrown away. Now: one with a shared prefix or its own end of prompt still
        // ahead of it is kept as BACKGROUND — planned only when nothing else has work, unless a
        // queued request waits for it — until it has passed them; any other is retired at once. Either way `evict_seqs` leaves the anchors it passed.
        // (`Scheduler::set_background` has the measurement.)
        let (mut background, gone): (Vec<u64>, Vec<u64>) = jobs
            .iter()
            .filter(|(_, j)| j.token_tx.is_closed())
            .map(|(id, _)| *id)
            .partition(|id| sched.read_worth_finishing(*id));
        // And a request that asked to be background work (`Job::background`) while it reads.
        background.extend(jobs.iter().filter(|(_, j)| j.background).map(|(id, _)| *id));
        sched.set_background(&background);
        if !gone.is_empty() {
            for id in &gone {
                backend.release_stream(*id);
            }
            sched.evict_seqs(&gone);
            for id in &gone {
                jobs.remove(id);
            }
            metrics
                .evictions
                .fetch_add(gone.len() as u64, Ordering::Relaxed);
            eprintln!(
                "[gone] {} request(s) whose client left were retired; what they had read of a \
                 shared prefix is kept",
                gone.len()
            );
            metrics
                .running
                .store(sched.num_running(), Ordering::Relaxed);
            metrics
                .waiting
                .store(sched.num_waiting(), Ordering::Relaxed);
            if !sched.has_unfinished() {
                continue;
            }
        }
        sched.set_decode_only(rr.pick(&sched, &jobs, &*backend));
        sched.set_prefill_cap(share.next_cap(&sched));
        let t_share = std::time::Instant::now();
        let plan = match sched.schedule() {
            Ok(Some(p)) => p,
            // Nothing runnable this step (e.g. the waiting head can't be
            // admitted yet). There is no model work until a new job arrives, so
            // wait for one instead of spinning. (Normally unreachable in the
            // server config — main.rs sizes the pool so any single request fits
            // — but the loop must not busy-spin if it ever happens.) If a
            // terminator is still pending, use the 1ms timed wait so its flush
            // is retried rather than stalled behind the blocking recv.
            Ok(None) => {
                if pending_close.is_empty() {
                    match recv_unless_stopping(&rx) {
                        Ok(job) => admit(
                            &mut sched,
                            &mut jobs,
                            &metrics,
                            job,
                            &piece_map,
                            vision.as_deref(),
                            greedy_only,
                        ),
                        Err(_) => {
                            backend.quiesce();
                            return;
                        }
                    }
                } else {
                    match rx.recv_timeout(std::time::Duration::from_millis(1)) {
                        Ok(job) => admit(
                            &mut sched,
                            &mut jobs,
                            &metrics,
                            job,
                            &piece_map,
                            vision.as_deref(),
                            greedy_only,
                        ),
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return,
                    }
                }
                continue;
            }
            Err(e) => {
                eprintln!("actor: scheduler error (fatal): {e}");
                return;
            }
        };
        // `[done]` line: the first step a sequence appears in starts at its cached prefix.
        for sp in &plan.seqs {
            if let Some(j) = jobs.get_mut(&sp.id) {
                j.stats.cached.get_or_insert(sp.past_len);
            }
        }
        // HYBRID PREFIX CACHE: a sequence whose prefix matched down to a snapshot boundary gets
        // that recurrent state BEFORE its first step. The scheduler only matches to keys it was
        // told exist and is told the moment one is evicted, so a miss here is a logic error —
        // and running the step anyway would serve fluent text from another sequence's state.
        for sp in &plan.seqs {
            if let Some(key) = sp.restore_key {
                // a state saved to disk that a request resumes from is kept (`prefix_disk::touch`)
                crate::prefix_disk::touch(key);
                if spec_dbg() {
                    eprintln!(
                        "[state-snapshot] restore seq {} key {key:#x}, resuming at {} tokens",
                        sp.id, sp.past_len
                    );
                }
                // A NEW session: every time. The conversation that took the anchor: once per
                // anchor (every time under ARF_SPEC_DEBUG). See `anchor_keys`.
                if let Some(origin) = anchor_keys.get(&key) {
                    crate::anchor_replay::touch(key);
                    let after = sched
                        .running_tokens(sp.id)
                        .and_then(|t| t.get(sp.past_len..))
                        .unwrap_or(&[]);
                    // TOOLS ANCHOR (2026-09-27): the same classification; the line names it, so a
                    // session in another working directory is seen resuming past the tools.
                    let (which, prefix) = if origin.tools {
                        ("tools anchor", "tools block")
                    } else {
                        ("anchor", "system prefix")
                    };
                    if anchor_resume_is_new_session(&origin.tail, after) {
                        eprintln!(
                            "[state-snapshot] seq {} resumes from the {which} snapshot at {} \
                             tokens — a NEW session (it leaves seq {}'s prompt within {} tokens \
                             of the {which}): the shared {prefix} was not prefilled again",
                            sp.id,
                            sp.past_len,
                            origin.seq,
                            common_len(&origin.tail, after)
                        );
                    } else if anchor_resumes_said.insert(key) || spec_dbg() {
                        eprintln!(
                            "[state-snapshot] seq {} resumes from the {which} snapshot at {} \
                             tokens — the SAME conversation as seq {}, which took it (no deeper \
                             snapshot of its own), NOT a new session",
                            sp.id, sp.past_len, origin.seq
                        );
                    }
                }
                if junction_keys.contains(&key) && (junction_resumes_said.insert(key) || spec_dbg())
                {
                    eprintln!(
                        "[state-snapshot] seq {} resumes from the junction snapshot at {} tokens \
                         (where an earlier prompt left cached history): the shared prefix past \
                         the anchor was not prefilled again",
                        sp.id, sp.past_len
                    );
                }
                if let Some(&(origin, origin_len)) = session_start_keys.get(&key) {
                    let len = sched.running_tokens(sp.id).map_or(0, |t| t.len());
                    if len <= origin_len {
                        eprintln!(
                            "[state-snapshot] seq {} resumes from the session-start snapshot at \
                             {} tokens — a NEW session that opens the same way as seq {} (its \
                             {len}-token prompt ends within that one's {origin_len}): the first \
                             prompt was not prefilled again",
                            sp.id, sp.past_len, origin
                        );
                    } else if session_start_resumes_said.insert(key) || spec_dbg() {
                        eprintln!(
                            "[state-snapshot] seq {} resumes from the session-start snapshot at \
                             {} tokens — a prompt that goes on past seq {}'s first prompt (its \
                             next turn, or a new session that opens the same way and continues)",
                            sp.id, sp.past_len, origin
                        );
                    }
                }
                if !backend.state_restore(sp.id, key) {
                    eprintln!(
                        "actor: recurrent-state snapshot {key:#x} for sequence {} is gone (fatal): \
                         its prefix was skipped on the strength of it",
                        sp.id
                    );
                    return;
                }
            }
        }
        // ARF_ACTOR_PROF=1 — per-step wall breakdown for the daemon path.
        // The in-process benches never exercise this loop, which is exactly why a ~100x
        // serving-layer gap (48s for 16 tokens) went unseen while the engine measured 89-441 tok/s.
        let _t_step = std::time::Instant::now();
        let (input_ids, mut batch) =
            build_forward_cached(&plan, sched.block_size(), &mut slots_cache);
        let _t_built = std::time::Instant::now();
        let sampling: Vec<SeqSampling> = plan
            .seqs
            .iter()
            .map(|sp| SeqSampling {
                params: &jobs[&sp.id].params,
                position: sp.past_len + sp.q_len,
                // Feed the emitted-token history: the repetition penalty reads it (and
                // it's empty — zero cost — for plain requests, since `generated` is only
                // populated when a penalty or constraint is active).
                generated: &jobs[&sp.id].generated,
            })
            .collect();

        // Determine if ANY sequence in this batch wants logprobs or has a grammar
        // constraint. Either case requires calling `forward_batch` to get the raw
        // logits before sampling — the unconstrained, no-logprobs path uses
        // `backend.step()` (a single call with zero overhead).
        let any_logprobs = plan
            .seqs
            .iter()
            .any(|sp| jobs.get(&sp.id).map(|j| j.want_logprobs).unwrap_or(false));
        let any_constrained = plan.seqs.iter().any(|sp| {
            jobs.get(&sp.id)
                .map(|j| j.constraint.is_some())
                .unwrap_or(false)
        });

        // ── L229 SPECULATIVE STEP ──────────────────────────────────────────────────────
        // Applies to exactly one shape: a SINGLE decode sequence, greedy, unconstrained, no
        // logprobs, with a drafter that has a proposal. Everything else falls through to the
        // ordinary path below, unchanged.
        //
        // The accept rule is the seam's contract (core/backend.rs): verify returns the greedy
        // argmax at each window position, so `out[i]` is what greedy would emit after
        // `window[i]`. Accept the longest run where out[i] == window[i+1]; out[j] at the first
        // disagreement is the BONUS token — it is greedy's own next token, so it is always
        // correct to take. Output is therefore byte-identical to greedy by construction.
        // CORRECTED 2026-10-05: identical up to NEAR-TIES. "What greedy would emit" is computed
        // here from verify-shaped logits, which differ in the last bits from a one-row step, so
        // a near-tie can pick the other token: a 0.067-nat tie flipped at the 4th token of a
        // reasoning trace (measured 2026-10-05). The accept rule is still sound.
        // A lone prefill chunk that does not finish its prompt: its token is never sampled, so
        // the backend may skip the prediction (2026-09-26, ~0.95 s TTFT at 7,433 tokens).
        backend.set_prefill_logits_needed(
            !(plan.seqs.len() == 1 && plan.seqs[0].q_len > 1 && !plan.seqs[0].completes_prompt),
        );
        let mut spec_tokens: Option<Vec<u32>> = None;
        let mut probe_draft: Option<u32> = None;
        if spec_dbg() && spec_k() > 0 {
            eprintln!("[spec] gate: seqs={} q_len={:?} logprobs={any_logprobs} constrained={any_constrained}",
                plan.seqs.len(), plan.seqs.first().map(|s| s.q_len));
        }
        // GREEDY ONLY, and it is checked HERE (2026-09-26). Verify returns the RAW argmax, so a
        // sampled request (temperature > 0) or a repetition penalty (which changes the argmax even
        // at temperature 0) must not speculate. Until today only the lookup drafter's seeding
        // checked the temperature (`spec_on` below); the block draft and the MTP chain came through
        // this gate unchecked, and every sampled request with a draft attached was served GREEDY
        // text — temperature 1.0 with seeds 1 and 2 returned the greedy answer byte for byte.
        let all_greedy = plan
            .seqs
            .iter()
            .all(|sp| jobs.get(&sp.id).is_some_and(|j| plain_greedy(&j.params)));
        // DECODE ROWS ONLY (2026-09-26, review of the anchor snapshot). Both speculation gates
        // below tested `q_len == 1` alone, which a MID-PROMPT one-token prefill chunk also has:
        // a snapshot cut at `a` (the prefix anchor, or the end-of-prompt boundary) leaves the
        // chunk `[a-1, a)` whenever an earlier chunk ended one short — one step beside a
        // decode row is enough. The verify then ran `[tok_{a-1}, drafts..]` inside the prompt;
        // accepted drafts stay in the recurrent state (the verify contract), the run was thrown
        // away because the sequence was still prefilling, and the snapshot saved right after
        // held `tokens[..a]` PLUS the drafts — restored by every later new session of that
        // agent. `completes_prompt` is true for a decode row and for a prompt's LAST token (whose
        // verify starts the answer, as before); false exactly for a chunk inside the prompt.
        // Tested: tests/actor_spec_gate.rs (both gates; both failed without this).
        //
        // Qwen3.8 IMAGE sequences never speculate (2026-09-27): every verify window and MTP draft
        // builds its own rows at plain positions (`verify_window` sets `positions` = KV index),
        // but an image sequence's decode rows rotate at `kv_index - rope_delta`
        // (`batch.mrope_positions`). A verify at the wrong angle would ACCEPT the wrong tokens,
        // not just fewer. Plain steps carry the M-RoPE table; speculation for images is future work.
        let decode_rows_only = batch.mrope_positions.is_none()
            && plan.seqs.iter().all(|s| s.q_len == 1 && s.completes_prompt);
        // C2 — MULTI-STREAM speculation (see `multi_spec_step`): two greedy decode streams verify
        // their windows in ONE record instead of taking a plain step together.
        // 2026-09-27: SAMPLED streams too (`multi_spec_sampling_on`), mixed freely with greedy
        // ones — each non-plain-greedy stream's rows are drawn by its own `RowSampler`, built as
        // the lone stream's is (`row_sampler_for`: params, seed, the emitted history a repetition
        // penalty reads — so a greedy request WITH a penalty takes the penalised argmax, as alone).
        // Logprobs and grammar constraints still take the plain step: neither is served by a draw.
        let mut multi_runs: Option<Vec<Vec<u32>>> = None;
        if multi_spec_on()
            && spec_k() > 0
            && (all_greedy || multi_spec_sampling_on())
            && !any_logprobs
            && !any_constrained
            && (2..=MULTI_MAX_STREAMS).contains(&plan.seqs.len())
            && decode_rows_only
        {
            let mut samplers: Vec<Option<arf_core::sampling::RowSampler>> = plan
                .seqs
                .iter()
                .map(|sp| {
                    let j = &jobs[&sp.id];
                    (!plain_greedy(&j.params)).then(|| row_sampler_for(j))
                })
                .collect();
            multi_tick = multi_tick.wrapping_add(1);
            multi_runs = multi_spec_step(&plan, &input_ids, &*backend, multi_tick, &mut samplers);
        }
        // SPECULATIVE SAMPLING (2026-09-26): a lone SAMPLED stream speculates too. Every verify
        // row is drawn with the plain path's own sampler (`RowSampler`), so the accept rule and
        // everything after it follow the draws — exact, and seed-identical to plain sampling.
        // (That identity holds for point-mass drafts — the block draft's default selector, MTP,
        // lookup. The opt-in SAMPLED draft, `ARF_SAMPLED_DRAFT=1`, is distribution-exact but not
        // seed-identical: it spends extra uniforms. And identity is up to the verify record's
        // logits differing from a one-row step's in the last bits.)
        // Measured before it: sampled 16.1 tok/s vs another engine 62-64. `ARF_NO_SPEC_SAMPLING=1` = off.
        let mut row_sampler: Option<arf_core::sampling::RowSampler> =
            (!all_greedy && plan.seqs.len() == 1 && spec_sampling_on())
                .then(|| row_sampler_for(&jobs[&plan.seqs[0].id]));
        // TEACHER-FORCED ACCEPTANCE (`ARF_FORCED_TOKENS`, see `forced.rs`): a lone greedy request
        // emits a fixed continuation, and each cycle logs the draft's proposals against it. When
        // the sequence is forced, no other speculative path runs this step. Off = never entered.
        let mut forced_seq = false;
        if forced::forced_on() {
            for sp in &plan.seqs {
                forced::observe(sp, sched.running_tokens(sp.id));
            }
            forced_seq = plan.seqs.len() == 1 && forced::is_forced(&plan.seqs[0]);
            if forced_seq && all_greedy && !any_logprobs && !any_constrained && decode_rows_only {
                let sp = &plan.seqs[0];
                spec_tokens = forced::step(&*backend, sp, input_ids[sp.q_start]);
            }
        }
        if !forced_seq
            && spec_k() > 0
            && (all_greedy || row_sampler.is_some())
            && !any_logprobs
            && !any_constrained
            && plan.seqs.len() == 1
            && decode_rows_only
        {
            let sp = &plan.seqs[0];
            let k = spec_k();
            // L255 — MTP HEAD FIRST, suffix drafter as fallback.
            //
            // The trained head (blk.64) proposes on EVERY step; `SuffixDrafter` needs a >=4-token
            // suffix seen before and fired on 12% of steps (L238), which is why speculation has
            // measured as parity so far. Both feed the SAME verify path, so correctness is
            // unchanged either way — this only changes how often a window is attempted.
            let last_tok = input_ids[sp.q_start];
            let bs_k = backend.kv_geometry().1;
            // L337 — the chain drafts up to k tokens at positions [past_len, past_len+k), each
            // needing a slot [0, past_len+k). CLAMP to what the CURRENT block table backs: the
            // scheduler allocated for this step's single token, so past_len+k may not be backed
            // yet near a block boundary. `slots_for` indexes block_table[p/bs] and panics past
            // it (paged.rs:310) — the exact crash the first chain build hit. `k_draft` is what
            // fits; k=1 gives the old single-draft sizing byte-for-byte.
            let cap_k = bs_k * sp.block_table.len();
            let k_draft = k.min(cap_k.saturating_sub(sp.past_len));
            let mtp_slots =
                arf_core::cache::slots_for(&sp.block_table, bs_k, sp.past_len + k_draft);
            // A long COPY first: it is free to propose (a hash lookup) and, when it exists and has
            // been paying, worth far more than the head's single token.
            let lookup: Vec<u32> = jobs
                .get(&sp.id)
                .filter(|j| lookup_on() && !mtp_probe() && j.lookup_cooldown.get() == 0)
                .and_then(|j| {
                    j.spec_drafter
                        .as_ref()
                        .map(|d| d.propose(&j.spec_history, LOOKUP_K))
                })
                // ROWS THE BLOCK TABLE BACKS RIGHT NOW. The scheduler allocates for this step's one
                // token, so a window is cut at the end of the current 16-slot block. The first
                // build judged a window cut to 2-4 rows as a MISS (it can only accept 1-3) and
                // backed off through exactly the code block that was paying: 40 of 51 proposals
                // suppressed. So: trim the draft to what is backed, and do not fire at all unless
                // that still leaves a window worth its ~124 ms.
                .map(|mut d| {
                    d.truncate(cap_k.saturating_sub(sp.past_len).saturating_sub(1));
                    d
                })
                .filter(|d| d.len() >= LOOKUP_MIN_DRAFT)
                .unwrap_or_default();
            if spec_dbg() {
                if let Some(j) = jobs.get(&sp.id) {
                    let raw = j
                        .spec_drafter
                        .as_ref()
                        .map(|d| d.propose(&j.spec_history, LOOKUP_K).len())
                        .unwrap_or(0);
                    eprintln!(
                        "[lookup] pos={} raw_draft={raw} cooldown={} misses={} used={}",
                        sp.past_len,
                        j.lookup_cooldown.get(),
                        j.lookup_misses.get(),
                        lookup.len()
                    );
                }
            }
            let used_lookup = !lookup.is_empty();
            if let Some(j) = jobs.get(&sp.id) {
                j.lookup_cooldown
                    .set(j.lookup_cooldown.get().saturating_sub(1));
            }
            // THE BLOCK DRAFT (DFlash 2), behind a paying copy and ahead of the MTP head: one pass
            // of an attached draft model proposes the next 7 tokens. Empty when no draft is
            // attached or its context is not an unbroken history of this sequence — then the
            // head drafts as before. Trimmed below to the rows the block table backs, like
            // every other window.
            // `ARF_NO_BLOCK_BACKOFF=1`: never rest the draft — for measuring its acceptance, which
            // the controller otherwise samples unevenly (it probes most where the draft is worst).
            let block_resting = std::env::var_os("ARF_NO_BLOCK_BACKOFF").is_none()
                && jobs.get(&sp.id).is_some_and(|j| {
                    let c = j.block_cooldown.get();
                    j.block_cooldown.set(c.saturating_sub(1));
                    c > 0
                });
            // EXPERIMENT `ARF_BLOCK_AFTER_FULL=1` — MANIPULATE, don't observe. The block draft
            // runs ONLY when the previous block window accepted all 7. If the 33.3%-vs-5.3%
            // zero-accept asymmetry is damage carried by a partial accept, suppressing the
            // partial-accept cycles should leave the survivors at ~5.3%; if it is the draft
            // being in a different regime, the survivors will drift back toward the pooled rate.
            // Observational analysis cannot separate those (permutation p=0.0001 says the effect
            // is real, but not what causes it).
            let after_full_only = std::env::var_os("ARF_BLOCK_AFTER_FULL").is_some()
                && jobs
                    .get(&sp.id)
                    .is_some_and(|j| !matches!(j.block_last_accepted.get(), 7 | usize::MAX));
            // GPU-SELECTED draft (ARF_DFLASH_GPU_SELECT=1, 2026-09-22): the block draft is
            // launched with its selector ON THE GPU and the chosen tokens stay in a GPU buffer the
            // verify record reads directly — the CPU never waits for the draft. MEASURED before
            // this: ~42 ms of idle GPU per cycle between draft and verify (measured).
            // `block` then holds a PLACEHOLDER of the right length: only its LENGTH is used before
            // the verify (window shape, block-table clamp), and the real tokens replace `window`
            // the moment the verify returns, before anything reads their content.
            // DEFAULT ON since 2026-09-25 (ARF_NO_DFLASH_GPU_SELECT=1 opts out): on group-64 weights
            // 67.1 -> 66.2 ms a cycle over 6 prompts x 2 rounds, identical text (it was a tie at the
            // ~80 ms cycles it was built under). The old opt-in ARF_DFLASH_GPU_SELECT is retired.
            let block_gpu_mode = std::env::var_os("ARF_NO_DFLASH_GPU_SELECT").is_none();
            let mut block_gpu = false;
            let block: Vec<u32> = if used_lookup || mtp_probe() || block_resting || after_full_only
            {
                Vec::new()
            } else if let Some(rs) = row_sampler
                .as_mut()
                .filter(|r| sampled_draft_on() && !r.params.is_greedy())
            {
                // SAMPLED draft (speculative sampling, step 2): drawn from the draft's own
                // distribution at the request's temperature; the verify accepts with min(1, p/q).
                // 2026-09-26: ON THE GPU SELECTOR when the GPU-select path is on — the chain draws
                // (`dflash_chain_sampled`) and the verify reads q back, so the CPU never waits for
                // the draft (the CPU selector's wait cost 1-3% wall-clock against the point mass
                // despite +9.2% acceptance, measured 2026-09-26). `rs.draft_q` stays empty
                // here: the backend fills it after the record. `ARF_SAMPLED_DRAFT_CPU=1` (or
                // `ARF_NO_DFLASH_GPU_SELECT=1`) keeps the CPU selector, which is also the fallback
                // whenever the GPU chain is not launched. NOT MEASURED on the GPU when written;
                // acceptance measured since (measured 2026-09-27, "The GPU-sampled draft
                // is the DEFAULT": 3.834 vs the point mass's 3.519 tokens a window, 90 requests).
                // A clean-box wall-clock A/B is still owed.
                let gpu = (block_gpu_mode && !sampled_draft_cpu())
                    .then(|| {
                        backend.block_draft_gpu_sampled(
                            last_tok,
                            sp.past_len,
                            sp.id,
                            rs.params.temperature,
                            rs.params.seed,
                        )
                    })
                    .flatten();
                match gpu {
                    Some(n) => {
                        block_gpu = true;
                        vec![u32::MAX; n]
                    }
                    None => match backend.block_draft_sampled(
                        last_tok,
                        sp.past_len,
                        sp.id,
                        rs.params.temperature,
                        rs.params.seed,
                    ) {
                        Some((d, q)) => {
                            rs.draft_q = q;
                            d
                        }
                        None => Vec::new(),
                    },
                }
            } else if block_gpu_mode {
                match backend.block_draft_gpu(last_tok, sp.past_len, sp.id) {
                    Some(n) => {
                        block_gpu = true;
                        vec![u32::MAX; n]
                    }
                    None => Vec::new(), // no ring for this stream: fall back as an empty draft does
                }
            } else {
                backend.block_draft(last_tok, sp.past_len, sp.id)
            };
            // The draft's context is fed by EVERY verify window and by nothing else, so while it
            // is live even a window trimmed to the pending token alone must go through verify:
            // an ordinary step would leave a hole and silence the draft for the rest of the
            // sequence.
            let block_live = !block.is_empty();
            if spec_dbg() && block_live {
                eprintln!("[dflash] pos={} draft={block:?}", sp.past_len);
            }
            let mtp: Vec<u32> = if used_lookup {
                lookup
            } else if block_live {
                block
            } else if std::env::var_os("ARF_NO_MTP").is_some() || k_draft == 0 {
                Vec::new()
            } else {
                backend.mtp_draft_chain(last_tok, sp.past_len, &mtp_slots, k_draft)
            };
            // Probe mode keeps the draft for scoring and speculates on nothing, so the
            // ordinary step below emits the greedy token the draft is scored against.
            if mtp_probe() {
                probe_draft = mtp.first().copied();
            }
            let draft: Vec<u32> = match Some(mtp).filter(|d| !d.is_empty() && !mtp_probe()) {
                Some(d) => {
                    if spec_dbg() {
                        eprintln!("[spec] MTP draft ({}): {d:?}", d.len());
                    }
                    d
                }
                None if mtp_probe() => Vec::new(),
                None => jobs
                    .get(&sp.id)
                    .and_then(|j| j.spec_drafter.as_ref().map(|d| (d, &j.spec_history)))
                    .map(|(d, h)| {
                        let p = d.propose(h, k);
                        if spec_dbg() {
                            eprintln!("[spec] propose: id={} hist={} -> {:?}", sp.id, h.len(), p);
                        }
                        p
                    })
                    .unwrap_or_default(),
            };
            if spec_dbg() && draft.is_empty() {
                let hl = jobs.get(&sp.id).map(|j| j.spec_history.len()).unwrap_or(0);
                let has_d = jobs
                    .get(&sp.id)
                    .map(|j| j.spec_drafter.is_some())
                    .unwrap_or(false);
                let tail: Vec<u32> = jobs
                    .get(&sp.id)
                    .map(|j| j.spec_history.iter().rev().take(8).rev().copied().collect())
                    .unwrap_or_default();
                eprintln!("[spec] no draft: history={hl} drafter={has_d} k={k} tail={tail:?}");
            }
            if !draft.is_empty() {
                // The window is [last_committed_token, draft...]: verify predicts the token
                // AFTER each window position, so position 0 must be the token the sequence is
                // currently sitting on.
                // The window's position 0 must be the token the sequence is SITTING ON, i.e.
                // the last committed token — which is exactly this decode step's input. Using
                // the drafter's own history tail instead would desync if the two ever diverge.
                let last = input_ids[sp.q_start];
                if spec_dbg() {
                    let ht: Vec<u32> = jobs
                        .get(&sp.id)
                        .map(|j| j.spec_history.iter().rev().take(3).rev().copied().collect())
                        .unwrap_or_default();
                    eprintln!("[spec] window0={last} hist_tail={ht:?} draft={draft:?}");
                }
                let mut window = Vec::with_capacity(1 + draft.len());
                window.push(last);
                window.extend_from_slice(&draft);
                let bs = backend.kv_geometry().1;
                // CLAMP the window to the slots the CURRENT block table actually covers. The
                // scheduler allocates for THIS step's single token, so near a block boundary
                // past_len+1+k is not yet backed — skipping entirely there loses ~2 of every 16
                // steps for no reason. Trimming the draft keeps speculation alive on those steps.
                let cap = sp.block_table.len() * bs;
                if sp.past_len + window.len() > cap {
                    window.truncate(cap.saturating_sub(sp.past_len));
                }
                let need = sp.past_len + window.len();
                if window.len() >= 2 || (block_live && !window.is_empty()) {
                    let prefix_slots = arf_core::cache::slots_for(&sp.block_table, bs, need);
                    let vres = if block_gpu {
                        // The window's tokens are on the GPU; the verify hands the real ones back.
                        let r = match row_sampler.as_ref() {
                            Some(rs) => backend.verify_window_gpu_draft_sampled(
                                window.len(),
                                sp.past_len,
                                &prefix_slots,
                                sp.id,
                                rs,
                            ),
                            None => backend.verify_window_gpu_draft(
                                window.len(),
                                sp.past_len,
                                &prefix_slots,
                                sp.id,
                            ),
                        };
                        match r {
                            Some((real, preds)) => {
                                window = real;
                                Some(preds)
                            }
                            None => None,
                        }
                    } else {
                        match row_sampler.as_ref() {
                            Some(rs) => backend.verify_window_sampled(
                                &window,
                                sp.past_len,
                                &prefix_slots,
                                sp.id,
                                rs,
                            ),
                            None => {
                                backend.verify_window(&window, sp.past_len, &prefix_slots, sp.id)
                            }
                        }
                    };
                    if spec_dbg() && vres.is_none() {
                        eprintln!(
                            "[spec] verify DECLINED (window {} past_len {})",
                            window.len(),
                            sp.past_len
                        );
                    }
                    if let Some(preds) = vres {
                        if spec_dbg() {
                            eprintln!("[spec] verify ok: window={window:?} preds={preds:?}");
                        }
                        // preds[i] = greedy's token after window[i]. The rule lives in
                        // arf-core so the backend's state restore names the same row.
                        let accepted = arf_core::model::speculative::accepted_run(&window, &preds);
                        if spec_dbg() {
                            eprintln!(
                                "[spec] drafted {} accepted {} (window {})",
                                draft.len(),
                                accepted.len().saturating_sub(1),
                                window.len()
                            );
                        }
                        // `ARF_SPEC_MISS=1` — READ THE MISSES, do not just count them.
                        // Everything measured on the 26% zero-accept rate so far has been a
                        // COUNT (rate, mean, permutation test), and every mechanism proposed
                        // from counts has been eliminated (state damage: killed by the
                        // manipulation experiment; text autocorrelation: killed by the MTP
                        // control; committed rows: another engine does the same). What has never been
                        // looked at is WHICH TOKEN the draft proposes when it misses at
                        // position 0 versus what the target wanted there.
                        if std::env::var_os("ARF_SPEC_MISS").is_some() && window.len() >= 6 {
                            let got = accepted.len().saturating_sub(1);
                            // draft[i] is the proposal verified at window position i; preds[0]
                            // is what greedy actually wanted after the anchor.
                            let proposed = window.get(1).copied();
                            let wanted = preds.first().copied();
                            if let (Some(pr), Some(wa)) = (proposed, wanted) {
                                // IDs only: the actor has no tokenizer. Decoded afterwards by
                                // scripts/spec_miss_report.py, which is also what turns these
                                // into the distribution that matters.
                                eprintln!(
                                    "[miss] accepted={got} anchor={:?} pos0_proposed={pr} pos0_wanted={wa}",
                                    window.first().copied().unwrap_or(0),
                                );
                            }
                        }
                        if block_live {
                            if let Some(j) = jobs.get(&sp.id) {
                                j.block_last_accepted.set(accepted.len().saturating_sub(1));
                            }
                            // Only a window the block table did not cut says anything about the
                            // draft; a short one can only accept a few.
                            if let Some(j) = jobs.get(&sp.id).filter(|_| window.len() >= 6) {
                                let got = accepted.len().saturating_sub(1) as f32;
                                let ema = j.block_ema.get() * (1.0 - BLOCK_EMA) + got * BLOCK_EMA;
                                j.block_ema.set(ema);
                                if ema < block_pays_from() {
                                    // Sit out, then probe with ONE window's worth of credit: the
                                    // mean restarts just above the bar, so a probe that lands
                                    // keeps the draft and one that does not rests it for longer.
                                    j.block_cooldown
                                        .set(BLOCK_BACKOFF << j.block_misses.get().min(3));
                                    j.block_misses.set(j.block_misses.get() + 1);
                                    j.block_ema.set(block_pays_from() + 0.4);
                                    if spec_dbg() {
                                        eprintln!(
                                            "[dflash] not paying (mean {ema:.2}) — resting {} steps",
                                            j.block_cooldown.get()
                                        );
                                    }
                                } else if got >= block_pays_from() {
                                    j.block_misses.set(0);
                                }
                            }
                        }
                        if used_lookup {
                            let got = accepted.len().saturating_sub(1);
                            if let Some(j) = jobs.get(&sp.id) {
                                if got < LOOKUP_PAYS_FROM {
                                    // cap at 4 << 2 = 16 steps: a thinking model misses a lot while it
                                    // reasons, and a 128-step penalty then slept through the code block
                                    // it was about to copy (measured: 10 windows in 320 tokens).
                                    j.lookup_misses.set((j.lookup_misses.get() + 1).min(2));
                                    j.lookup_cooldown
                                        .set(LOOKUP_BACKOFF << j.lookup_misses.get());
                                } else {
                                    j.lookup_misses.set(0);
                                }
                            }
                        }
                        spec_tokens = Some(accepted);
                    }
                }
            }
        }
        // The accepted run is committed by the SHIPPED path below, one token at a time —
        // `commit_tokens` takes exactly one per scheduled sequence. Reusing it means the
        // speculative path inherits stop/length handling, metrics and streaming unchanged.
        // L229 — when the speculative step already produced the accepted run, SKIP the
        // ordinary forward entirely. Running it would do a full weight-stream pass whose
        // result is discarded, which turns the speculation win into a loss.
        let (tokens, maybe_logits): (Vec<u32>, Option<Vec<Vec<f32>>>) =
            if spec_tokens.is_some() || multi_runs.is_some() {
                (Vec::new(), None)
            } else if any_logprobs || any_constrained {
                let logits = backend.forward_batch(&input_ids, &batch);
                let toks = if any_constrained {
                    // Build per-seq constraint refs and generated-token slices.
                    let constraints: Vec<Option<&dyn Constraint>> = plan
                        .seqs
                        .iter()
                        .map(|sp| jobs.get(&sp.id).and_then(|j| j.constraint.as_deref()))
                        .collect();
                    // Build SeqSampling with generated tokens for constrained seqs.
                    let sampling_c: Vec<SeqSampling> = plan
                        .seqs
                        .iter()
                        .map(|sp| {
                            let j = &jobs[&sp.id];
                            SeqSampling {
                                params: &j.params,
                                position: sp.past_len + sp.q_len,
                                generated: &j.generated,
                            }
                        })
                        .collect();
                    match sample_batch_constrained(&logits, &sampling_c, &constraints, &piece_map) {
                        Ok(t) => t,
                        Err(e) => {
                            eprintln!("actor: constrained sampling error (fatal): {e}");
                            return;
                        }
                    }
                } else {
                    match sample_batch(&logits, &sampling) {
                        Ok(t) => t,
                        Err(e) => {
                            eprintln!("actor: backend step error (fatal): {e}");
                            return;
                        }
                    }
                };
                (toks, Some(logits))
            } else {
                let _t_step = std::time::Instant::now();
                let _r = backend.step(&input_ids, &batch, &sampling);
                if std::env::var("ARF_STEP_TIMING").is_ok() {
                    let us = _t_step.elapsed().as_secs_f64() * 1e6;
                    let rows = batch.seqs.len();
                    let qtot: usize = batch.seqs.iter().map(|s| s.q_len).sum();
                    eprintln!("[step] {us:8.1} us · {rows} seq · {qtot} q-tok");
                }
                match _r {
                    Ok(t) => (t, None),
                    Err(e) => {
                        eprintln!("actor: backend step error (fatal): {e}");
                        return;
                    }
                }
            };
        drop(sampling);
        if let Some(d) = probe_draft {
            mtp_probe_observe(d, tokens.first().copied());
        }
        // L3 — hand this step's slots vecs back to the cache (O(1) moves; `batch` is not read
        // past this point) so the next step extends instead of rebuilding.
        share.observe(
            plan.seqs
                .iter()
                .filter(|sp| sp.q_len > 1)
                .map(|sp| sp.q_len)
                .sum(),
            t_share.elapsed(),
        );
        slots_cache.reclaim(&plan, &mut batch);
        // L309 — a verified run is committed WHOLE. Its accepted drafts' K/V and recurrent
        // state are already in place (verify keeps them and restores only past the last
        // accepted row), so the scheduler advances by the run and the next input is its last
        // token, the bonus. (L231 queued the tail in a `spec_pending` nothing ever read: every
        // cycle committed one token, and a verify window bought exactly nothing.)
        let multi = multi_runs.is_some();
        let forced_plain = forced::forced_on() && spec_tokens.is_none() && multi_runs.is_none();
        let mut runs: Vec<Vec<u32>> = match (spec_tokens, multi_runs) {
            (Some(run), _) => vec![run],
            (None, Some(runs)) => runs,
            (None, None) => tokens.iter().map(|&t| vec![t]).collect(),
        };
        // `ARF_FORCED_TOKENS`: a plain step of a forced sequence emits the continuation's token
        // (the prompt's first token, or a cycle whose forced verify declined).
        if forced_plain {
            forced::override_plain(&plan.seqs, &mut runs);
        }
        // every sequence's history, when the step was a multi-stream verify; seqs[0] otherwise
        let fed = if multi { plan.seqs.len() } else { 1 };
        for (sp, run) in plan.seqs.iter().zip(&runs).take(fed) {
            if let Some(j) = jobs.get_mut(&sp.id) {
                for &t in run {
                    if let Some(d) = j.spec_drafter.as_mut() {
                        d.observe(&j.spec_history, t);
                    }
                    j.spec_history.push(t);
                }
            }
        }
        // HYBRID PREFIX CACHE: this step left a sequence exactly on the boundary the scheduler
        // picked — save its recurrent state there, and keep the scheduler's view of which
        // snapshots exist in step with the backend's.
        for sp in &plan.seqs {
            if let Some(key) = sp.snapshot_key {
                // An ANCHOR (end of the shared system + tools prefix) goes to the backend's
                // anchor pool, where per-turn snapshots cannot evict it.
                // The "anchor snapshot at N" line is printed only once the save SUCCEEDED
                // (Corrected 2026-09-26, review: it printed before the call and spent its
                // once-per-process flag, so a failed first save claimed a snapshot and hid the
                // first real one).
                // A JUNCTION (where this prompt left cached history, 2026-09-26) goes to the
                // backend's junction pool, where it cannot evict an anchor.
                // So does a SESSION START (a conversation's first end-of-prompt snapshot,
                // 2026-09-26), where the conversation's own later turn-end saves cannot evict it.
                let (saved, evicted) = if sp.snapshot_anchor {
                    backend.state_save_anchor(sp.id, key)
                } else if sp.snapshot_junction || sp.snapshot_session_start {
                    backend.state_save_junction(sp.id, key)
                } else if sp.snapshot_checkpoint {
                    backend.state_save_checkpoint(sp.id, key)
                } else {
                    backend.state_save(sp.id, key)
                };
                if saved && sp.snapshot_session_start {
                    static SAID: std::sync::atomic::AtomicBool =
                        std::sync::atomic::AtomicBool::new(false);
                    if !SAID.swap(true, Ordering::Relaxed) {
                        eprintln!(
                            "[state-snapshot] session-start snapshot at {} tokens (kept for the \
                             next session that opens the same way)",
                            sp.past_len + sp.q_len
                        );
                    }
                }
                if saved && sp.snapshot_junction {
                    static SAID: std::sync::atomic::AtomicBool =
                        std::sync::atomic::AtomicBool::new(false);
                    if !SAID.swap(true, Ordering::Relaxed) {
                        eprintln!(
                            "[state-snapshot] junction snapshot at {} tokens (where this prompt \
                             leaves cached history)",
                            sp.past_len + sp.q_len
                        );
                    }
                }
                if saved && sp.snapshot_tools_anchor {
                    static SAID: std::sync::atomic::AtomicBool =
                        std::sync::atomic::AtomicBool::new(false);
                    if !SAID.swap(true, Ordering::Relaxed) {
                        eprintln!(
                            "[state-snapshot] tools anchor snapshot at {} tokens (end of the \
                             tools block)",
                            sp.past_len + sp.q_len
                        );
                    }
                } else if saved && sp.snapshot_anchor {
                    static SAID: std::sync::atomic::AtomicBool =
                        std::sync::atomic::AtomicBool::new(false);
                    if !SAID.swap(true, Ordering::Relaxed) {
                        eprintln!(
                            "[state-snapshot] anchor snapshot at {} tokens (end of the shared \
                             system prefix)",
                            sp.past_len + sp.q_len
                        );
                    }
                }
                if spec_dbg() {
                    eprintln!(
                        "[state-snapshot] save seq {} key {key:#x} at {} tokens{}: {saved}",
                        sp.id,
                        sp.past_len + sp.q_len,
                        if sp.snapshot_tools_anchor {
                            " (tools anchor)"
                        } else if sp.snapshot_anchor {
                            " (anchor)"
                        } else if sp.snapshot_junction {
                            " (junction)"
                        } else if sp.snapshot_session_start {
                            " (session-start)"
                        } else {
                            ""
                        }
                    );
                }
                if let Some(old) = evicted {
                    // Issue #12's measurement: a snapshot evicted while the request that took it
                    // is still running. Always printed (it should be rare); count these lines
                    // during concurrent agent runs before building eviction protection.
                    if let Some(holder) = sched.forget_snapshot(old) {
                        static HELD_EVICTIONS: std::sync::atomic::AtomicU64 =
                            std::sync::atomic::AtomicU64::new(0);
                        let n = HELD_EVICTIONS.fetch_add(1, Ordering::Relaxed) + 1;
                        eprintln!(
                            "[state-snapshot] evicted key {old:#x} while seq {holder} still holds \
                             it (its next turn re-prefills; {n} so far)"
                        );
                    }
                    anchor_keys.remove(&old);
                    junction_keys.remove(&old);
                    session_start_keys.remove(&old);
                }
                if saved && sp.snapshot_junction {
                    junction_keys.insert(key);
                    // A junction is where a later session left the cached history — for an agent,
                    // the end of the system prompt + tools + the project's AGENTS.md (which Claude
                    // Code puts in the first user message). Saved to disk like an anchor, so a
                    // restarted server resumes there and not at the anchor (`prefix_disk`).
                    if let Some(prompt) = sched.running_tokens(sp.id) {
                        crate::prefix_disk::note_anchor(key, sp.past_len + sp.q_len, false, prompt);
                    }
                }
                if saved && sp.snapshot_session_start {
                    // Still prefilling (the boundary leaves >= 1 prompt token), so its tokens are
                    // exactly its prompt.
                    let len = sched.running_tokens(sp.id).map_or(0, |t| t.len());
                    session_start_keys.insert(key, (sp.id, len));
                }
                if saved && sp.snapshot_anchor {
                    let at = sp.past_len + sp.q_len;
                    // The anchor's token prefix outlives a restart (`anchor_replay`).
                    if let Some(prompt) = sched.running_tokens(sp.id) {
                        crate::anchor_replay::record(key, at, sp.snapshot_tools_anchor, prompt);
                        crate::prefix_disk::note_anchor(key, at, sp.snapshot_tools_anchor, prompt);
                    }
                    let tail = sched
                        .running_tokens(sp.id)
                        .and_then(|t| t.get(at..))
                        .map(|t| t[..t.len().min(ANCHOR_TAIL_CAP)].to_vec())
                        .unwrap_or_default();
                    anchor_keys.insert(
                        key,
                        AnchorOrigin {
                            seq: sp.id,
                            tail,
                            tools: sp.snapshot_tools_anchor,
                        },
                    );
                }
                if saved {
                    // HELD until this sequence finishes — see Scheduler::hold_snapshot. The KV
                    // blocks for the same tokens are not shareable until then.
                    sched.hold_snapshot(sp.id, key);
                }
            }
        }
        STEP.fetch_add(1, Ordering::Relaxed);
        let outs = match sched.commit_runs(&runs) {
            Ok(o) => {
                ran_a_step = true;
                report_prefill_progress(&sched, &mut reading, &metrics);
                o
            }
            Err(e) => {
                eprintln!("actor: commit error (fatal): {e}");
                return;
            }
        };

        // Publish a metrics snapshot: gauges (current state) + step counter.
        // All stores are Relaxed — the /metrics handler needs only eventual
        // consistency, and we never need ordering relative to other atomics.
        metrics.steps.fetch_add(1, Ordering::Relaxed);
        metrics
            .running
            .store(sched.num_running(), Ordering::Relaxed);
        metrics
            .waiting
            .store(sched.num_waiting(), Ordering::Relaxed);
        let total = sched.num_total_blocks();
        let free = sched.num_free_blocks();
        metrics
            .kv_blocks_used
            .store(total.saturating_sub(free), Ordering::Relaxed);
        metrics.kv_blocks_total.store(total, Ordering::Relaxed);
        metrics
            .prefix_cache_tokens_reused
            .store(sched.prefix_cache_reused_tokens(), Ordering::Relaxed);

        // L13: per-step geometry for the first few decode steps. Everything the GPU sees is
        // derived from THESE numbers (rows / q_len / past_len), never from cfg_mbs — printing
        // them side by side is what turns "mbs shouldn't matter" into "mbs demonstrably does
        // not reach here". Capped so it can be left on for a whole deep run without flooding.
        if mbs_probe && mbs_probe_steps < 12 {
            mbs_probe_steps += 1;
            let rows: usize = plan.seqs.iter().map(|s| s.q_len).sum();
            eprintln!(
                "[l13-mbs] step#{mbs_probe_steps} cfg_mbs={cfg_mbs} B={} rows={rows} \
                 q_lens={:?} past_lens={:?}",
                plan.seqs.len(),
                plan.seqs.iter().map(|s| s.q_len).collect::<Vec<_>>(),
                plan.seqs.iter().map(|s| s.past_len).collect::<Vec<_>>(),
            );
        }
        if actor_prof {
            let now = std::time::Instant::now();
            // cycle = start-of-this-step MINUS start-of-previous-step: the true per-iteration
            // wall cost, including everything after `step=` (emit, detok, SSE, commit, metrics).
            let cycle_ms = _cycle_prev
                .map(|p| (_t_step - p).as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            let step_ms = (now - _t_built).as_secs_f64() * 1000.0;
            eprintln!("[actor-prof] B={} build={:.1}ms step={:.1}ms total={:.1}ms cycle={:.1}ms non_gpu={:.1}ms",
                plan.seqs.len(),
                (_t_built - _t_step).as_secs_f64() * 1000.0,
                step_ms,
                (now - _t_step).as_secs_f64() * 1000.0,
                cycle_ms,
                if cycle_ms > 0.0 { cycle_ms - step_ms } else { 0.0 });
        }
        _cycle_prev = Some(_t_step);
        // Count generated tokens + finished requests from this step's outputs.
        let mut step_tokens: u64 = 0;
        let mut step_finished: u64 = 0;
        for out in &outs {
            if out.finish_reason.is_none() {
                // Normal mid-generation token (no finish — count it).
                step_tokens += 1;
            } else {
                // Finished (Stop, StopString or Length). The final token (if any) is
                // delivered by demux; count it only if it was a Length or StopString finish
                // (demux emits the last token for those but not for Stop).
                if matches!(
                    out.finish_reason,
                    Some(CoreFinish::Length | CoreFinish::StopString)
                ) {
                    step_tokens += 1;
                }
                step_finished += 1;
            }
        }
        metrics
            .tokens_generated
            .fetch_add(step_tokens, Ordering::Relaxed);
        metrics
            .requests_finished
            .fetch_add(step_finished, Ordering::Relaxed);

        // Demux to per-request channels; collect eviction candidates. Eviction
        // happens AFTER commit (never between schedule and commit — row
        // alignment, see Scheduler::evict_seqs).
        let mut evict: Vec<u64> = Vec::new();
        // Map from sequence id → index in plan.seqs (to look up the logits row).
        let seq_to_plan_idx: HashMap<u64, usize> = plan
            .seqs
            .iter()
            .enumerate()
            .map(|(i, sp)| (sp.id, i))
            .collect();
        for out in outs {
            // Track generated tokens when the NEXT step needs the history: a grammar
            // constraint's state machine, OR an active repetition penalty. Zero
            // overhead for plain greedy/stochastic requests (neither set).
            if let Some(j) = jobs.get_mut(&out.id) {
                if j.constraint.is_some() || j.params.has_repetition_penalty() {
                    j.generated.push(out.token);
                }
            }

            let logprob_data = if let Some(logits_vec) = &maybe_logits {
                if let Some(&plan_idx) = seq_to_plan_idx.get(&out.id) {
                    if let Some(j) = jobs.get(&out.id) {
                        if j.want_logprobs {
                            Some(compute_token_logprob(
                                &logits_vec[plan_idx],
                                out.token,
                                j.top_logprobs,
                            ))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };
            let finished = out.finish_reason.is_some().then_some(out.id);
            demux(&mut jobs, out, logprob_data, &mut evict, &mut pending_close);
            if let Some(id) = finished {
                backend.release_stream(id);
            }
        }
        for id in &evict {
            backend.release_stream(*id);
        }
        if !evict.is_empty() {
            let evict_count = evict.len() as u64;
            sched.evict_seqs(&evict);
            for id in &evict {
                if let Some(j) = jobs.remove(id) {
                    let _ = j.token_tx.try_send(StreamItem::Done(FinishReason::Evicted));
                }
            }
            metrics.evictions.fetch_add(evict_count, Ordering::Relaxed);
            // The gauges above were published before this eviction; an idle actor publishes no
            // further step, so a cancelled lone stream left `arf_running_sequences 1` on an idle
            // server until the next request (measured live, 2026-09-26). Republish them.
            metrics
                .running
                .store(sched.num_running(), Ordering::Relaxed);
            metrics
                .waiting
                .store(sched.num_waiting(), Ordering::Relaxed);
            metrics.kv_blocks_used.store(
                sched
                    .num_total_blocks()
                    .saturating_sub(sched.num_free_blocks()),
                Ordering::Relaxed,
            );
        }
    }
}

/// Validate + enqueue one job. Degenerate jobs finish immediately (preserves
/// the old single-stream actor's behavior for empty prompt / zero budget).
fn admit(
    sched: &mut Scheduler,
    jobs: &mut HashMap<u64, JobState>,
    metrics: &Metrics,
    mut job: Job,
    piece_map: &Arc<TokenPieceMap>,
    vision: Option<&dyn ImageEncoder>,
    greedy_only: Option<&str>,
) {
    if job.prompt_ids.is_empty() || job.params.max_tokens == 0 {
        let _ = job
            .token_tx
            .try_send(StreamItem::Done(FinishReason::Length));
        return;
    }
    // A request's arrival, with what it waits behind — a long wait is then told from a hang
    // (asked for 2026-10-06 by a teammate whose agent showed nothing for 9 minutes). Not for the
    // server's own warm-up and replay requests.
    if job.id < crate::metrics::INTERNAL_IDS {
        let ahead = sched.prefill_progress();
        let reading: usize = ahead.iter().map(|&(_, done, total)| total - done).sum();
        eprintln!(
            "[start] request {}: {} prompt tokens · {} running, {} waiting{}",
            job.id,
            thousands(job.prompt_ids.len()),
            sched.num_running(),
            sched.num_waiting(),
            if reading > 0 {
                format!(
                    " · {} prompt tokens still to read ahead of it",
                    thousands(reading)
                )
            } else {
                String::new()
            }
        );
    }
    // A BACKEND THAT SERVES GREEDY STEPS ONLY (2026-10-07): a sampled request reached a path the
    // backend cannot run and the model thread panicked — every later request then answered "model
    // actor thread is gone" from a server that still looked healthy. Reported on Qwen3-Coder-30B:
    // any OpenAI client's default temperature took the server down on its first message. Such a
    // request is now served greedily, and the log says so once per kind. Sampling there needs the
    // island to sample from its own logits; that is the real fix and is not built yet.
    if let Some(why) = greedy_only {
        let mut dropped = Vec::new();
        if !job.params.is_greedy() {
            dropped.push("sampling (temperature > 0)");
            job.params.temperature = 0.0;
        }
        if (job.params.repetition_penalty - 1.0).abs() >= f32::EPSILON {
            dropped.push("a repetition penalty");
            job.params.repetition_penalty = 1.0;
        }
        if job.want_logprobs {
            dropped.push("logprobs");
            job.want_logprobs = false;
        }
        if job.response_format.is_some() {
            dropped.push("a structured-output grammar");
            job.response_format = None;
        }
        if !dropped.is_empty() {
            static SAID: std::sync::OnceLock<()> = std::sync::OnceLock::new();
            let first = SAID.set(()).is_ok();
            eprintln!(
                "[greedy-only] request {}: served greedily, without {}{}",
                job.id,
                dropped.join(", "),
                if first {
                    format!(" — {why}")
                } else {
                    String::new()
                }
            );
        }
    }
    // Build a grammar constraint if the job specifies a structured output format.
    // "json_object" and "json_schema" both use JsonConstraint v1 (structural JSON).
    // v2 will add schema-level enforcement on top.
    //
    // Stop tokens are plumbed into the constraint so it can:
    //   (a) leave them unmasked when the JSON value is complete (Done state),
    //       enabling the model to terminate via the normal stop-token path
    //       instead of padding to max_tokens; and
    //   (b) use them as the never-all-masked fallback (defense in depth).
    let constraint: Option<Box<dyn Constraint>> = match job.response_format.as_deref() {
        Some("json_object") | Some("json_schema") => Some(Box::new(
            JsonConstraint::new_with_stops(Arc::clone(piece_map), job.params.stop_tokens.clone()),
        )),
        _ => None,
    };
    // Vision: encode each image (GPU-side) and build the positional embed-override. The
    // HTTP layer reserved 256 <image_soft_token> slots per image and recorded their LOCAL
    // positions; here we fill them with the projector's soft-tokens. A vision job with no
    // configured encoder, or a token-count mismatch, falls back to a text-only request
    // (the placeholder tokens then embed normally — degraded but not a crash).
    let image_prompt = match job.image_prompt.take() {
        Some(pre) => Some(pre),
        None => build_image_prompt(&job, vision),
    };

    metrics.requests_admitted.fetch_add(1, Ordering::Relaxed);
    // L229 — speculation is GREEDY-ONLY. The accept rule (out[i] == window[i+1]) is an argmax
    // equality, so it reproduces greedy exactly and nothing else. A sampled request must never
    // take this path.
    // 2026-09-26: still true for THIS drafter — the prompt-lookup seeding. Sampled requests now
    // speculate through the block draft and MTP instead, with each verify row DRAWN by the plain
    // sampler (`RowSampler`), which makes the same equality rule exact for them.
    let spec_on = spec_k() > 0 && job.params.is_greedy() && !job.params.has_repetition_penalty();
    jobs.insert(
        job.id,
        JobState {
            params: job.params.clone(),
            token_tx: job.token_tx,
            want_logprobs: job.want_logprobs,
            top_logprobs: job.top_logprobs,
            constraint,
            generated: Vec::new(),
            background: job.background,
            // L229 — seed the drafter with the PROMPT so a request that quotes its own
            // context (RAG, "rewrite this", a coding agent applying an edit) can draft from
            // it on the very first generated token. That is lookup drafting: no draft model,
            // no extra weights, proposals come from text already in the window.
            spec_history: if spec_on {
                job.prompt_ids.clone()
            } else {
                Vec::new()
            },
            spec_drafter: spec_on.then(|| {
                // INDEX the prompt, do not merely store it. `observe(history_before, tok)`
                // records a position per suffix key, so a drafter handed a populated
                // `spec_history` it never observed has ZERO keys and `propose` returns empty
                // forever — which is exactly what the first wiring did (39 gate passes, 0
                // drafts, history=67). Feeding every prefix is what the reference loop does
                // (speculative.rs tests::observe_all) and it is what makes lookup drafting
                // work on the FIRST generated token for a prompt that quotes itself.
                let mut d = arf_core::model::speculative::SuffixDrafter::new();
                for i in 0..job.prompt_ids.len() {
                    d.observe(&job.prompt_ids[..i], job.prompt_ids[i]);
                }
                d
            }),
            lookup_cooldown: std::cell::Cell::new(0),
            lookup_misses: std::cell::Cell::new(0),
            block_ema: std::cell::Cell::new(BLOCK_EMA_START),
            block_cooldown: std::cell::Cell::new(0),
            block_misses: std::cell::Cell::new(0),
            block_last_accepted: std::cell::Cell::new(usize::MAX),
            stats: ReqStats::new(job.prompt_ids.len(), job.background),
        },
    );
    let request = match image_prompt {
        Some(img) => Request::with_image(job.id, job.prompt_ids, job.params, img),
        None => Request::new(job.id, job.prompt_ids, job.params),
    };
    sched.add(
        request
            .with_prefix_anchor(job.prefix_anchor)
            .with_tools_anchor(job.tools_anchor)
            .with_header_tail(job.header_tail)
            .with_stop_check(job.stop),
    );
}

/// Encode the job's images into an `ImagePrompt`, or `None` for a text-only job (no images,
/// no encoder, or a slot/token-count mismatch). Concatenates all images' soft-tokens in
/// order; `job.image_positions` are the matching LOCAL prompt slots.
fn build_image_prompt(job: &Job, vision: Option<&dyn ImageEncoder>) -> Option<ImagePrompt> {
    if job.images.is_empty() {
        return None;
    }
    let enc = match vision {
        Some(e) => e,
        None => {
            eprintln!("actor: image request but no vision encoder configured — ignoring image");
            return None;
        }
    };
    let hidden = enc.hidden();
    let per_image = enc.num_tokens();
    let mut embeds = Vec::with_capacity(job.images.len() * per_image * hidden);
    for (rgb, w, h) in &job.images {
        let soft = enc.encode(rgb, *w, *h); // [per_image, hidden]
        if soft.len() != per_image * hidden {
            eprintln!(
                "actor: vision encode produced {} floats, expected {}",
                soft.len(),
                per_image * hidden
            );
            return None;
        }
        embeds.extend_from_slice(&soft);
    }
    let expected = job.images.len() * per_image;
    if job.image_positions.len() != expected {
        eprintln!(
            "actor: {} image positions but {} soft-tokens — dropping image",
            job.image_positions.len(),
            expected
        );
        return None;
    }
    Some(ImagePrompt {
        embeds,
        hidden,
        positions: job.image_positions.clone(),
        mrope: None,
        // Gemma-3: each image attends bidirectionally within its span.
        causal: false,
    })
}

/// Compute the logprob of `chosen_token` in `logits_row`, plus the top-N
/// alternatives. Uses logsumexp for numerical stability (natural log).
fn compute_token_logprob(logits: &[f32], chosen_token: u32, top_n: usize) -> Box<TokenLogprob> {
    // logsumexp for numerical stability.
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f64 = logits.iter().map(|&l| ((l - max_l) as f64).exp()).sum();
    let logsumexp = max_l as f64 + sum_exp.ln();

    let chosen_logprob = if (chosen_token as usize) < logits.len() {
        logits[chosen_token as usize] as f64 - logsumexp
    } else {
        f64::NEG_INFINITY
    };

    // The whole row, in vocabulary order (the System One score-only job).
    if top_n == ALL_LOGPROBS {
        return Box::new(TokenLogprob {
            token_id: chosen_token,
            logprob: chosen_logprob as f32,
            top_logprobs: logits
                .iter()
                .enumerate()
                .map(|(id, &l)| (id as u32, (l as f64 - logsumexp) as f32))
                .collect(),
        });
    }

    // Build top-N list sorted descending by logprob (= logit - logsumexp).
    let mut top_logprobs: Vec<(u32, f32)> = if top_n > 0 {
        let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
        // Partial select: bring top_n largest to the front.
        let keep = top_n.min(logits.len());
        indexed.select_nth_unstable_by(keep - 1, |(_, a), (_, b)| {
            b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut top = indexed[..keep].to_vec();
        top.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        top.into_iter()
            .map(|(id, logit)| (id as u32, (logit as f64 - logsumexp) as f32))
            .collect()
    } else {
        Vec::new()
    };

    // Ensure the chosen token is always present in the list (OpenAI convention).
    if top_n > 0 && !top_logprobs.iter().any(|(id, _)| *id == chosen_token) {
        top_logprobs.push((chosen_token, chosen_logprob as f32));
    }

    Box::new(TokenLogprob {
        token_id: chosen_token,
        logprob: chosen_logprob as f32,
        top_logprobs,
    })
}

/// Route one step's output for one sequence to its channel — NON-blocking.
///
/// A failed send on a **live** sequence marks it for eviction (slow or gone
/// client); finished sequences are guaranteed their `Done` terminator:
/// if the channel is momentarily full, the terminator is stashed in
/// `pending_close` and flushed opportunistically on subsequent loop iterations
/// (never by blocking). A closed channel (client gone) is silently dropped —
/// there is nobody to deliver to.
///
/// Guarantee: every normally-finished stream (Stop/Length) eventually receives
/// `StreamItem::Done` — it is NEVER silently lost under backpressure.
fn demux(
    jobs: &mut HashMap<u64, JobState>,
    out: RequestOutput,
    logprob_data: Option<Box<TokenLogprob>>,
    evict: &mut Vec<u64>,
    pending_close: &mut Vec<(tokio::sync::mpsc::Sender<StreamItem>, FinishReason)>,
) {
    // Clone the (cheap, Arc-backed) sender so the `jobs` borrow ends before the
    // removes below — `get` + `remove` in one scope trips the borrow checker.
    let Some(tx) = jobs.get(&out.id).map(|j| j.token_tx.clone()) else {
        return;
    };
    match out.finish_reason {
        // Stop token: not emitted (matches generate_streaming semantics).
        Some(CoreFinish::Stop) => {
            if let Some(j) = jobs.get(&out.id) {
                j.stats.report("stop");
            }
            match tx.try_send(StreamItem::Done(FinishReason::Stop)) {
                Ok(()) => {}
                Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
                    pending_close.push((tx, FinishReason::Stop));
                }
                Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {} // client gone
            }
            jobs.remove(&out.id);
        }
        // Length: the final token IS emitted (best-effort), then Done (guaranteed).
        // A stop string the same way, as `Stop`: its last token carries the text
        // before the stop string, which the HTTP layer cuts there.
        Some(core @ (CoreFinish::Length | CoreFinish::StopString)) => {
            let (reason, label) = if core == CoreFinish::Length {
                (FinishReason::Length, "length")
            } else {
                (FinishReason::Stop, "stop string")
            };
            if let Some(j) = jobs.get_mut(&out.id) {
                j.stats.token();
                j.stats.report(label);
            }
            let _ = tx.try_send(StreamItem::Token(out.token, logprob_data)); // best-effort
            match tx.try_send(StreamItem::Done(reason)) {
                Ok(()) => {}
                Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
                    pending_close.push((tx, reason));
                }
                Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {} // client gone
            }
            jobs.remove(&out.id);
        }
        // commit never produces Evicted (only evict_seqs sets it); drop state.
        Some(CoreFinish::Evicted) => {
            jobs.remove(&out.id);
        }
        None => {
            if let Some(j) = jobs.get_mut(&out.id) {
                j.stats.token();
            }
            if tx
                .try_send(StreamItem::Token(out.token, logprob_data))
                .is_err()
            {
                // full (slow) or closed (disconnected). A speculative step hands one sequence
                // several outputs, and each failed send landed here: one cancelled stream counted
                // 2 evictions in `arf_evictions_total` (measured live, 2026-09-26). Once per id.
                if !evict.contains(&out.id) {
                    evict.push(out.id);
                }
            }
        }
    }
}

#[cfg(test)]
mod anchor_resume_tests {
    use super::*;

    #[test]
    fn a_new_session_diverges_inside_the_first_message_and_a_next_turn_does_not() {
        // The anchor-taker's prompt after the anchor: 40 tokens of first message, then an
        // 8-token assistant header.
        let tail: Vec<u32> = (100..140).chain(900..908).collect();
        // Next turn of the same conversation: the first message, then (a different rendering
        // of) the assistant turn, then more.
        let next_turn: Vec<u32> = (100..140).chain(950..1000).collect();
        assert!(!anchor_resume_is_new_session(&tail, &next_turn));
        // A new session: another first message.
        let new_session: Vec<u32> = (200..260).collect();
        assert!(anchor_resume_is_new_session(&tail, &new_session));
        // Sharing the first 10 tokens of the message ("Fix the ...") is still a new session.
        let similar: Vec<u32> = (100..110).chain(300..350).collect();
        assert!(anchor_resume_is_new_session(&tail, &similar));
        // A tail shorter than the slack cannot be told apart: never claimed as a new session.
        assert!(!anchor_resume_is_new_session(&[1, 2, 3], &[9, 9, 9]));
    }
}

#[cfg(test)]
mod progress_tests {
    use super::thousands;

    #[test]
    fn thousands_groups_digits() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1000), "1,000");
        assert_eq!(thousands(45120), "45,120");
        assert_eq!(thousands(1234567), "1,234,567");
    }
}
