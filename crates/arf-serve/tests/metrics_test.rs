//! Tests for the Prometheus metrics: render format correctness and actor
//! publishing (proves the actor actually writes to the shared Arc<Metrics>).

use std::sync::atomic::Ordering;
use std::sync::Arc;

use arf_core::backend::BatchedBackend;
use arf_core::config::EngineConfig;
use arf_core::model::batch::ForwardBatch;
use arf_core::sampling::{SamplingParams, TokenPieceMap};
use arf_serve::actor::{spawn, Job, StreamItem};
use arf_serve::metrics::Metrics;

// ---------------------------------------------------------------------------
// Unit tests: Metrics::render() format validation
// ---------------------------------------------------------------------------

/// Verify that render() produces valid Prometheus text for every metric:
/// correct # HELP / # TYPE lines, proper _total suffix on counters, and that
/// the numeric values appear in the output.
#[test]
fn render_contains_all_metric_lines() {
    let m = Metrics::default();

    // Set representative values.
    m.running.store(3, Ordering::Relaxed);
    m.waiting.store(7, Ordering::Relaxed);
    m.kv_blocks_used.store(42, Ordering::Relaxed);
    m.kv_blocks_total.store(1024, Ordering::Relaxed);
    m.requests_admitted.store(100, Ordering::Relaxed);
    m.requests_finished.store(90, Ordering::Relaxed);
    m.tokens_generated.store(5000, Ordering::Relaxed);
    m.steps.store(200, Ordering::Relaxed);
    m.evictions.store(2, Ordering::Relaxed);
    m.prefix_cache_tokens_reused.store(1234, Ordering::Relaxed);

    let out = m.render();

    // --- Gauges: no _total suffix ---
    assert!(
        out.contains("# TYPE arf_running_sequences gauge"),
        "missing gauge TYPE line for running_sequences"
    );
    assert!(
        out.contains("arf_running_sequences 3"),
        "missing running_sequences value"
    );

    assert!(
        out.contains("# TYPE arf_waiting_sequences gauge"),
        "missing gauge TYPE line for waiting_sequences"
    );
    assert!(
        out.contains("arf_waiting_sequences 7"),
        "missing waiting_sequences value"
    );

    assert!(
        out.contains("# TYPE arf_kv_blocks_used gauge"),
        "missing gauge TYPE for kv_blocks_used"
    );
    assert!(
        out.contains("arf_kv_blocks_used 42"),
        "missing kv_blocks_used value"
    );

    assert!(
        out.contains("# TYPE arf_kv_blocks_total gauge"),
        "missing gauge TYPE for kv_blocks_total"
    );
    assert!(
        out.contains("arf_kv_blocks_total 1024"),
        "missing kv_blocks_total value"
    );

    // --- Counters: _total suffix ---
    assert!(
        out.contains("# TYPE arf_requests_admitted_total counter"),
        "missing counter TYPE for requests_admitted_total"
    );
    assert!(
        out.contains("arf_requests_admitted_total 100"),
        "missing requests_admitted_total value"
    );

    assert!(
        out.contains("# TYPE arf_requests_finished_total counter"),
        "missing counter TYPE for requests_finished_total"
    );
    assert!(
        out.contains("arf_requests_finished_total 90"),
        "missing requests_finished_total value"
    );

    assert!(
        out.contains("# TYPE arf_tokens_generated_total counter"),
        "missing counter TYPE for tokens_generated_total"
    );
    assert!(
        out.contains("arf_tokens_generated_total 5000"),
        "missing tokens_generated_total value"
    );

    assert!(
        out.contains("# TYPE arf_steps_total counter"),
        "missing counter TYPE for steps_total"
    );
    assert!(
        out.contains("arf_steps_total 200"),
        "missing steps_total value"
    );

    assert!(
        out.contains("# TYPE arf_evictions_total counter"),
        "missing counter TYPE for evictions_total"
    );
    assert!(
        out.contains("arf_evictions_total 2"),
        "missing evictions_total value"
    );

    assert!(
        out.contains("# TYPE arf_prefix_cache_tokens_reused_total counter"),
        "missing counter TYPE for prefix_cache_tokens_reused_total"
    );
    assert!(
        out.contains("arf_prefix_cache_tokens_reused_total 1234"),
        "missing prefix_cache_tokens_reused_total value"
    );

    // Every metric must have a # HELP line (spot check a few).
    assert!(
        out.contains("# HELP arf_running_sequences"),
        "missing HELP for running_sequences"
    );
    assert!(
        out.contains("# HELP arf_tokens_generated_total"),
        "missing HELP for tokens_generated_total"
    );
    assert!(
        out.contains("# HELP arf_steps_total"),
        "missing HELP for steps_total"
    );
}

/// Render with all-zero defaults still produces valid (zero-valued) output.
#[test]
fn render_all_zeros_is_valid() {
    let m = Metrics::default();
    let out = m.render();
    // Gauge lines must be present even at zero.
    assert!(out.contains("arf_running_sequences 0"));
    assert!(out.contains("arf_kv_blocks_total 0"));
    assert!(out.contains("arf_steps_total 0"));
}

// ---------------------------------------------------------------------------
// Integration test: actor actually publishes metrics
// ---------------------------------------------------------------------------

/// A deterministic mock backend mirroring actor_loop.rs.
struct MockMetrics {
    vocab: usize,
}
impl BatchedBackend for MockMetrics {
    fn forward_batch(&self, _ids: &[u32], batch: &ForwardBatch) -> Vec<Vec<f32>> {
        batch
            .seqs
            .iter()
            .map(|s| {
                let mut row = vec![0.0f32; self.vocab];
                row[(s.past_len + s.q_len) % self.vocab] = 1.0;
                row
            })
            .collect()
    }
    fn kv_geometry(&self) -> (usize, usize) {
        (64, 4)
    }
}

fn test_cfg() -> EngineConfig {
    EngineConfig {
        block_size: 4,
        num_blocks: 64,
        max_batch_size: 8,
        max_prefill_tokens: 16,
        ..Default::default()
    }
}

fn empty_piece_map() -> Arc<TokenPieceMap> {
    Arc::new(TokenPieceMap::new(vec![]))
}

/// Drive the mock-backend actor through two jobs and assert that the shared
/// Arc<Metrics> reflects actual work: tokens_generated > 0, steps > 0,
/// requests_admitted == 2, requests_finished == 2.
#[tokio::test]
async fn actor_publishes_metrics_after_jobs() {
    let (handle, metrics, _join) = spawn(
        Box::new(MockMetrics { vocab: 16 }),
        test_cfg(),
        empty_piece_map(),
        None,
    );

    // Submit two jobs with known output counts.
    let (tx1, mut rx1) = tokio::sync::mpsc::channel::<StreamItem>(256);
    handle
        .submit(Job {
            id: 0,
            prompt_ids: vec![1u32; 3], // 3-token prompt → generates 2 tokens
            params: SamplingParams::greedy(2),
            token_tx: tx1,
            want_logprobs: false,
            top_logprobs: 0,
            response_format: None,
            images: Vec::new(),
            image_positions: Vec::new(),
            image_prompt: None,
            prefix_anchor: None,
            tools_anchor: None,
            header_tail: None,
            background: false,
            stop: None,
        })
        .unwrap();

    let (tx2, mut rx2) = tokio::sync::mpsc::channel::<StreamItem>(256);
    handle
        .submit(Job {
            id: 1,
            prompt_ids: vec![1u32; 5], // 5-token prompt → generates 3 tokens
            params: SamplingParams::greedy(3),
            token_tx: tx2,
            want_logprobs: false,
            top_logprobs: 0,
            response_format: None,
            images: Vec::new(),
            image_positions: Vec::new(),
            image_prompt: None,
            prefix_anchor: None,
            tools_anchor: None,
            header_tail: None,
            background: false,
            stop: None,
        })
        .unwrap();

    // Drain both streams to completion (waits for the actor to finish).
    let mut tok_count = 0usize;
    while let Some(item) = rx1.recv().await {
        match item {
            StreamItem::Token(_, _) => tok_count += 1,
            StreamItem::Done(_) => break,
        }
    }
    while let Some(item) = rx2.recv().await {
        match item {
            StreamItem::Token(_, _) => tok_count += 1,
            StreamItem::Done(_) => break,
        }
    }

    // By the time both streams are drained the actor has committed all steps
    // and published the final snapshot.  Assert the published counters match.
    assert_eq!(tok_count, 5, "expected 2 + 3 = 5 tokens across both jobs");

    assert!(
        metrics.steps.load(Ordering::Relaxed) > 0,
        "steps counter must be > 0 after work"
    );
    assert_eq!(
        metrics.requests_admitted.load(Ordering::Relaxed),
        2,
        "both requests must have been admitted"
    );
    assert_eq!(
        metrics.requests_finished.load(Ordering::Relaxed),
        2,
        "both requests must have finished"
    );
    assert!(
        metrics.tokens_generated.load(Ordering::Relaxed) >= 5,
        "tokens_generated must account for at least the 5 output tokens"
    );
    assert_eq!(
        metrics.kv_blocks_total.load(Ordering::Relaxed),
        64,
        "kv_blocks_total must match the pool size from the config"
    );
}
