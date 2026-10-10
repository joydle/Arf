//! Structured GPU profiler sink. The per-kernel timestamp aggregates that
//! `CommandPass::submit_profiled` computes were previously emitted ONLY as ephemeral
//! `tracing` events (gone into the log). This collects them into a process-global,
//! queryable form so they can be (a) dumped as a shareable JSON artifact at exit and
//! (b) snapshotted live by a HUD each frame.
//!
//! Active only under the `profiling` feature (the hot decode path pays nothing
//! otherwise). The collector is a global `Mutex` — profiling is a diagnostic mode, not
//! the throughput path, so the lock cost is irrelevant.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::OnceLock;

/// One kernel's accumulated cost across the run.
#[derive(Clone, Debug, Default)]
pub struct KernelStat {
    /// Number of dispatches recorded for this kernel name.
    pub calls: u64,
    /// Total GPU time across those dispatches (microseconds).
    pub total_us: f64,
    /// Sum of effective bandwidth samples (GB/s) and their count, for a mean. Only
    /// matmul-class kernels carry a byte count, so this is `None`-equivalent (count 0)
    /// for the rest.
    pub gbps_sum: f64,
    pub gbps_n: u64,
}

impl KernelStat {
    /// Mean effective bandwidth (GB/s) over the sampled dispatches, or 0 if none.
    pub fn mean_gbps(&self) -> f64 {
        if self.gbps_n == 0 {
            0.0
        } else {
            self.gbps_sum / self.gbps_n as f64
        }
    }
}

/// A point-in-time view of the profiler: per-kernel totals + frame/step counters.
#[derive(Clone, Debug, Default)]
pub struct ProfileSnapshot {
    pub kernels: BTreeMap<String, KernelStat>,
    /// Number of profiled steps (decode tokens / forward passes) seen.
    pub steps: u64,
    /// GPU time of the most recent step (microseconds) — the "current frame" cost.
    pub last_step_us: f64,
    /// Sum of all profiled steps' GPU time (microseconds).
    pub total_us: f64,
}

impl ProfileSnapshot {
    /// Kernels sorted by descending total GPU time — the "where the time goes" view.
    pub fn by_time(&self) -> Vec<(String, KernelStat)> {
        let mut v: Vec<_> = self
            .kernels
            .iter()
            .map(|(k, s)| (k.clone(), s.clone()))
            .collect();
        v.sort_by(|a, b| {
            b.1.total_us
                .partial_cmp(&a.1.total_us)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        v
    }
}

static COLLECTOR: OnceLock<Mutex<ProfileSnapshot>> = OnceLock::new();

fn collector() -> &'static Mutex<ProfileSnapshot> {
    COLLECTOR.get_or_init(|| Mutex::new(ProfileSnapshot::default()))
}

/// Record one profiled step: `samples` is `(kernel_name, gpu_us, opt_gbps)` per
/// dispatch. Accumulates into the global collector and bumps the step counters.
pub fn record_step(samples: &[(String, f64, Option<f64>)]) {
    let mut c = collector().lock().unwrap_or_else(|p| p.into_inner());
    let mut step_us = 0.0;
    for (name, us, gbps) in samples {
        step_us += *us;
        let e = c.kernels.entry(name.clone()).or_default();
        e.calls += 1;
        e.total_us += *us;
        if let Some(g) = gbps {
            e.gbps_sum += *g;
            e.gbps_n += 1;
        }
    }
    c.steps += 1;
    c.last_step_us = step_us;
    c.total_us += step_us;
}

/// A clone of the current profiler state — the HUD reads this each frame.
pub fn snapshot() -> ProfileSnapshot {
    collector()
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clone()
}

/// Reset the collector (e.g. between benchmark phases).
pub fn reset() {
    *collector().lock().unwrap_or_else(|p| p.into_inner()) = ProfileSnapshot::default();
}

/// Serialize the snapshot to a JSON string (hand-rolled — no serde dep needed for this
/// diagnostic artifact). Shareable: feeds a flamegraph / web timeline / the HUD export.
/// `roofline_gbps` is the device's peak memory bandwidth, used to print roofline %.
pub fn to_json(snap: &ProfileSnapshot, roofline_gbps: f64) -> String {
    let mut s = String::from("{\n");
    s.push_str(&format!("  \"steps\": {},\n", snap.steps));
    s.push_str(&format!("  \"total_us\": {:.3},\n", snap.total_us));
    s.push_str(&format!("  \"last_step_us\": {:.3},\n", snap.last_step_us));
    s.push_str(&format!("  \"roofline_gbps\": {roofline_gbps:.1},\n"));
    s.push_str("  \"kernels\": [\n");
    let by_time = snap.by_time();
    let total = snap.total_us.max(1e-9);
    for (i, (name, st)) in by_time.iter().enumerate() {
        let comma = if i + 1 < by_time.len() { "," } else { "" };
        let gbps = st.mean_gbps();
        let roof = if roofline_gbps > 0.0 {
            100.0 * gbps / roofline_gbps
        } else {
            0.0
        };
        s.push_str(&format!(
            "    {{ \"kernel\": \"{}\", \"calls\": {}, \"total_us\": {:.3}, \"pct_frame\": {:.2}, \"mean_gbps\": {:.1}, \"roofline_pct\": {:.1} }}{}\n",
            name, st.calls, st.total_us, 100.0 * st.total_us / total, gbps, roof, comma
        ));
    }
    s.push_str("  ]\n}\n");
    s
}
