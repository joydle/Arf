//! Lock-free serving metrics: the actor thread publishes a snapshot each step
//! into these atomics; the async /metrics handler reads them. No lock, no
//! scheduler access from tokio (the scheduler is single-owner on the actor).
//!
//! The Prometheus text exposition format (v0.0.4) is hand-rolled here — no
//! prometheus crate dependency. Each metric entry is three lines:
//!
//! ```text
//! # HELP arf_<name> <description>
//! # TYPE arf_<name> <gauge|counter>
//! arf_<name> <value>
//! ```
//!
//! Gauges are overwritten each step (current state); counters are monotonically
//! incremented (summed over process life) and carry the `_total` suffix per
//! Prometheus naming convention.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

#[derive(Default)]
pub struct Metrics {
    // ---- Gauges (current state, overwritten each step via store) --------------
    /// Number of sequences currently in the running batch.
    pub running: AtomicUsize,
    /// Number of sequences waiting to be admitted to the batch.
    pub waiting: AtomicUsize,
    /// KV cache blocks currently in use (total – free).
    pub kv_blocks_used: AtomicUsize,
    /// Total KV cache blocks in the pool (free + used, constant at runtime).
    pub kv_blocks_total: AtomicUsize,

    // ---- Counters (monotonic, incremented via fetch_add, never reset) ---------
    /// Requests admitted to the scheduler since process start.
    pub requests_admitted: AtomicU64,
    /// Requests that reached a terminal state (Stop/Length) since process start.
    pub requests_finished: AtomicU64,
    /// Output tokens generated (sampled) since process start.
    pub tokens_generated: AtomicU64,
    /// Scheduler steps executed since process start.
    pub steps: AtomicU64,
    /// Sequences evicted (client disconnect / backpressure) since process start.
    pub evictions: AtomicU64,
    /// Prompt tokens served from the prefix cache (KV reused, not recomputed).
    /// Zero when prefix caching is disabled (--no-prefix-cache).
    pub prefix_cache_tokens_reused: AtomicU64,

    // ---- Prompt reading (2026-10-06) -------------------------------------------
    /// Long prompts being read right now, for `/v1/arf/status` and `arf status`: rewritten by the
    /// actor after each step while a long prompt is being read (one short lock, never per token).
    pub prefills: std::sync::Mutex<Vec<PrefillStatus>>,
    /// Low Power Mode is on (the GPU runs at lower clocks): `[power]`, `/v1/arf/status`.
    pub low_power: std::sync::atomic::AtomicBool,
    /// The thermal state: 0 nominal, 1 fair, 2 serious, 3 critical.
    pub thermal: std::sync::atomic::AtomicU8,
}

/// The name of a [`Metrics::thermal`] value.
pub fn thermal_name(t: u8) -> &'static str {
    ["nominal", "fair", "serious", "critical"]
        .get(t as usize)
        .copied()
        .unwrap_or("unknown")
}

/// One prompt being read: how far, how fast, how long to go.
#[derive(Clone, Debug, serde::Serialize)]
pub struct PrefillStatus {
    pub id: u64,
    /// Prompt tokens computed so far (including any taken from the cache).
    pub done: usize,
    /// The prompt's length in tokens.
    pub total: usize,
    /// Tokens the cache supplied when the request started.
    pub cached: usize,
    /// Tokens read per second since the request started.
    pub tok_per_s: f64,
    /// Seconds left at that rate.
    pub eta_s: f64,
    /// A background re-read of an agent prompt recorded before a restart (`anchor_replay`), not
    /// a client's request.
    pub replay: bool,
}

/// Request ids at or above this are the server's own (warm-up and anchor replay), never a client's.
pub const INTERNAL_IDS: u64 = u64::MAX - 64;

impl Metrics {
    /// Render the Prometheus text exposition format (v0.0.4).
    ///
    /// Called from the async `/metrics` handler on the tokio thread pool; reads
    /// only the shared atomics — never touches the scheduler or the actor.
    pub fn render(&self) -> String {
        let g = |a: &AtomicUsize| a.load(Ordering::Relaxed);
        let c = |a: &AtomicU64| a.load(Ordering::Relaxed);

        // Each helper emits: # HELP, # TYPE, value line.
        macro_rules! gauge {
            ($name:expr, $help:expr, $val:expr) => {
                concat_to_string(&[
                    "# HELP arf_",
                    $name,
                    " ",
                    $help,
                    "\n",
                    "# TYPE arf_",
                    $name,
                    " gauge\n",
                    &format!("arf_{} {}\n\n", $name, $val),
                ])
            };
        }
        macro_rules! counter {
            ($name:expr, $help:expr, $val:expr) => {
                concat_to_string(&[
                    "# HELP arf_",
                    $name,
                    "_total ",
                    $help,
                    "\n",
                    "# TYPE arf_",
                    $name,
                    "_total counter\n",
                    &format!("arf_{}_total {}\n\n", $name, $val),
                ])
            };
        }

        let mut s = String::with_capacity(1024);

        s.push_str(&gauge!(
            "running_sequences",
            "Number of sequences currently in the running batch.",
            g(&self.running)
        ));
        s.push_str(&gauge!(
            "waiting_sequences",
            "Number of sequences waiting to be admitted to the batch.",
            g(&self.waiting)
        ));
        s.push_str(&gauge!(
            "kv_blocks_used",
            "KV cache blocks currently allocated (total minus free).",
            g(&self.kv_blocks_used)
        ));
        s.push_str(&gauge!(
            "kv_blocks_total",
            "Total KV cache blocks in the pool (free + used). Constant at runtime.",
            g(&self.kv_blocks_total)
        ));
        s.push_str(&counter!(
            "requests_admitted",
            "Requests admitted to the scheduler since process start.",
            c(&self.requests_admitted)
        ));
        s.push_str(&counter!(
            "requests_finished",
            "Requests that reached a terminal state (Stop or Length) since process start.",
            c(&self.requests_finished)
        ));
        s.push_str(&counter!(
            "tokens_generated",
            "Output tokens generated (sampled) since process start.",
            c(&self.tokens_generated)
        ));
        s.push_str(&counter!(
            "steps",
            "Scheduler steps executed since process start.",
            c(&self.steps)
        ));
        s.push_str(&counter!(
            "evictions",
            "Sequences evicted due to client disconnect or backpressure since process start.",
            c(&self.evictions)
        ));
        s.push_str(&counter!(
            "prefix_cache_tokens_reused",
            "Prompt tokens served from the prefix cache (KV reused, not recomputed). \
             Zero when prefix caching is disabled.",
            c(&self.prefix_cache_tokens_reused)
        ));

        s
    }
}

/// Concatenate a slice of string pieces into a single `String`.
fn concat_to_string(pieces: &[&str]) -> String {
    let cap: usize = pieces.iter().map(|s| s.len()).sum();
    let mut s = String::with_capacity(cap);
    for p in pieces {
        s.push_str(p);
    }
    s
}
