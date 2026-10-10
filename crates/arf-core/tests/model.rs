//! Forward-pass correctness: shapes, determinism, and the KV-cache invariant.

mod common;

use arf_core::cache::{slots_for, write_runs, PagedKvCache};
use arf_core::config::KvQuant;
use arf_core::model::{ForwardBatch, Llama, SeqAttn};

const BLOCK_SIZE: usize = 4;

/// Run one chunk of `tokens` (starting at logical `past_len`) for a single
/// sequence and return the last token's logits (length `vocab`).
fn process(
    model: &Llama,
    cache: &mut PagedKvCache,
    block_table: &[u32],
    tokens: &[u32],
    past_len: usize,
) -> Vec<f32> {
    let q_len = tokens.len();
    let ctx = past_len + q_len;
    let batch = ForwardBatch {
        positions: (past_len..ctx).map(|p| p as u32).collect(),
        seqs: vec![SeqAttn {
            stream_id: None,
            q_start: 0,
            q_len,
            past_len,
            slots: slots_for(block_table, BLOCK_SIZE, ctx),
            write_runs: write_runs(block_table, BLOCK_SIZE, past_len, q_len),
            image_spans: Vec::new(),
        }],
        image_embeds: None,
        mrope_positions: None,
    };
    let hidden = model.forward(tokens, &batch, cache);
    model.logits_last(&hidden, &batch).into_vec()
}

fn fresh_cache(model: &Llama) -> PagedKvCache {
    PagedKvCache::new(
        model.num_layers(),
        4,
        BLOCK_SIZE,
        model.config().num_kv_heads,
        model.config().head_dim,
    )
}

fn fresh_cache_quant(model: &Llama, kv_quant: KvQuant) -> PagedKvCache {
    PagedKvCache::with_quant(
        model.num_layers(),
        4,
        BLOCK_SIZE,
        model.config().num_kv_heads,
        model.config().head_dim,
        kv_quant,
    )
}

/// Relative L2 error of `got` against `want`.
fn rel_l2(got: &[f32], want: &[f32]) -> f32 {
    let err: f32 = got.iter().zip(want).map(|(a, b)| (a - b).powi(2)).sum();
    let sig: f32 = want.iter().map(|x| x * x).sum();
    (err / sig).sqrt()
}

#[test]
fn forward_produces_vocab_logits() {
    let model = common::tiny_model();
    let mut cache = fresh_cache(&model);
    let logits = process(&model, &mut cache, &[0, 1], &[3, 7], 0);
    assert_eq!(logits.len(), model.config().vocab_size);
    assert!(
        logits.iter().all(|x| x.is_finite()),
        "logits must be finite"
    );
}

#[test]
fn forward_is_deterministic() {
    let model = common::tiny_model();
    let a = process(&model, &mut fresh_cache(&model), &[0, 1], &[5, 9, 2], 0);
    let b = process(&model, &mut fresh_cache(&model), &[0, 1], &[5, 9, 2], 0);
    assert_eq!(a, b);
}

/// The central correctness property: prefilling a whole prompt at once must
/// yield the same final-token logits as feeding the tokens one at a time
/// through the KV cache.
#[test]
fn chunked_prefill_matches_token_by_token() {
    let model = common::tiny_model();
    let prompt = [3u32, 7, 1, 4, 2]; // 5 tokens -> 2 blocks at block_size 4
    let block_table = [0u32, 1];

    let mut cache_a = fresh_cache(&model);
    let a = process(&model, &mut cache_a, &block_table, &prompt, 0);

    let mut cache_b = fresh_cache(&model);
    let mut last = Vec::new();
    for (i, &tok) in prompt.iter().enumerate() {
        last = process(&model, &mut cache_b, &block_table, &[tok], i);
    }

    let max_diff = a
        .iter()
        .zip(&last)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-4,
        "prefill vs incremental decode diverged by {max_diff}"
    );
}

/// TurboQuant KV must track the exact `f32` KV oracle within a bounded drift —
/// the same contract Q4 weights have against bf16 (close, not bit-exact). More
/// bits ⇒ closer. This is the end-to-end correctness gate for the Tq path.
#[test]
fn tq_prefill_matches_f32_oracle_within_drift() {
    let model = common::tiny_model();
    let prompt = [3u32, 7, 1, 4, 2];
    let block_table = [0u32, 1];

    let oracle = process(&model, &mut fresh_cache(&model), &block_table, &prompt, 0);

    let mut prev = f32::INFINITY;
    for bits in [2u8, 3, 4] {
        let q = KvQuant::Tq { bits, qjl: false };
        let got = process(
            &model,
            &mut fresh_cache_quant(&model, q),
            &block_table,
            &prompt,
            0,
        );
        assert!(got.iter().all(|x| x.is_finite()));
        let drift = rel_l2(&got, &oracle);
        assert!(
            drift < 0.25,
            "{bits}-bit Tq logits drifted {drift} from the f32 oracle"
        );
        assert!(
            drift <= prev + 1e-3,
            "{bits}-bit drift {drift} should not exceed lower-bit drift {prev}"
        );
        prev = drift;
    }
}

/// The KV-cache invariant must hold under Tq too: one-shot prefill equals
/// token-by-token decode (both quantize the same vectors the same way).
#[test]
fn tq_chunked_prefill_matches_token_by_token() {
    let model = common::tiny_model();
    let prompt = [3u32, 7, 1, 4, 2];
    let block_table = [0u32, 1];
    let q = KvQuant::Tq {
        bits: 4,
        qjl: false,
    };

    let a = process(
        &model,
        &mut fresh_cache_quant(&model, q),
        &block_table,
        &prompt,
        0,
    );

    let mut cache_b = fresh_cache_quant(&model, q);
    let mut last = Vec::new();
    for (i, &tok) in prompt.iter().enumerate() {
        last = process(&model, &mut cache_b, &block_table, &[tok], i);
    }

    let max_diff = a
        .iter()
        .zip(&last)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-4,
        "Tq prefill vs incremental decode diverged by {max_diff}"
    );
}

#[test]
fn gemma_forward_produces_finite_logits() {
    let model = common::tiny_gemma_model();
    let mut cache = fresh_cache(&model);
    let logits = process(&model, &mut cache, &[0, 1], &[3, 7], 0);
    assert_eq!(logits.len(), model.config().vocab_size);
    assert!(
        logits.iter().all(|x| x.is_finite()),
        "Gemma logits must be finite"
    );
}

/// The KV-cache invariant for the Gemma path: the prompt (5 tokens) is longer
/// than the sliding window (3), so this also exercises that the windowed mask is
/// consistent between a single prefill and token-by-token decode.
#[test]
fn gemma_chunked_prefill_matches_token_by_token() {
    let model = common::tiny_gemma_model();
    let prompt = [3u32, 7, 1, 4, 2];
    let block_table = [0u32, 1];

    let mut cache_a = fresh_cache(&model);
    let a = process(&model, &mut cache_a, &block_table, &prompt, 0);

    let mut cache_b = fresh_cache(&model);
    let mut last = Vec::new();
    for (i, &tok) in prompt.iter().enumerate() {
        last = process(&model, &mut cache_b, &block_table, &[tok], i);
    }

    let max_diff = a
        .iter()
        .zip(&last)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-4,
        "Gemma prefill vs incremental decode diverged by {max_diff}"
    );
}

/// DOES tq4's ATTENTION-LEVEL DRIFT SURVIVE TO THE LOGITS, AND DOES IT GROW
/// WITH CONTEXT DEPTH?
///
/// This is the question that decides whether long contexts on a memory-bound GPU
/// can ship at tq4. A GPU kernel can be verified exact against the codec, so the
/// port is not in question. What IS in question is the codec's own cost on
/// ATTENTION OUTPUT, which can exceed the 0.25 bar
/// `tq_prefill_matches_f32_oracle_within_drift` holds logits to.
///
/// Attention output is not logits. Between them sit o_proj, a residual add, a
/// second norm and the FFN — and the residual in particular is a large signal
/// the attention delta is added TO, so a big relative error on the attention
/// branch can arrive at the logits much smaller. That is the effect this
/// measures rather than assumes.
///
/// DEPTH is the second half of the question, and the half that matters for 32K:
/// the existing tq gate runs a 5-token prompt. If drift compounds with context
/// length, a bound measured at 5 tokens says nothing about 32,768. The tiny
/// model's max_position_embeddings is 256, so this sweeps what it can and
/// reports the TREND; the absolute numbers belong to this model, the shape of
/// the curve is the transferable part.
#[test]
fn tq4_logit_drift_versus_context_depth() {
    let model = common::tiny_model();
    // The shared `fresh_cache` helpers allocate 4 blocks (16 slots at
    // BLOCK_SIZE 4) — enough for the 5-token prompts the other tq tests use and
    // NOT for this sweep, which is the point of the sweep. Size both the cache
    // and the block table to the deepest point instead.
    const DEPTHS: [usize; 6] = [5, 16, 32, 64, 128, 224];
    let max_depth = DEPTHS[DEPTHS.len() - 1];
    let n_blocks = max_depth.div_ceil(BLOCK_SIZE) + 1;
    let block_table: Vec<u32> = (0..n_blocks as u32).collect();
    let mk_cache = |q: Option<KvQuant>| match q {
        None => PagedKvCache::new(
            model.num_layers(),
            n_blocks,
            BLOCK_SIZE,
            model.config().num_kv_heads,
            model.config().head_dim,
        ),
        Some(q) => PagedKvCache::with_quant(
            model.num_layers(),
            n_blocks,
            BLOCK_SIZE,
            model.config().num_kv_heads,
            model.config().head_dim,
            q,
        ),
    };
    let q = KvQuant::Tq {
        bits: 4,
        qjl: false,
    };

    println!("  tq4 logit drift vs depth (tiny model, head_dim 4):");
    let mut drifts = Vec::new();
    for &n in &DEPTHS {
        // A deterministic, non-repeating prompt: a constant token would let the
        // cache be trivially compressible and flatter the quantizer.
        let prompt: Vec<u32> = (0..n).map(|i| ((i * 13 + 7) % 48) as u32).collect();
        let oracle = process(&model, &mut mk_cache(None), &block_table, &prompt, 0);
        let got = process(&model, &mut mk_cache(Some(q)), &block_table, &prompt, 0);
        assert!(
            got.iter().all(|x| x.is_finite()),
            "tq4 produced non-finite logits at depth {n}"
        );
        let d = rel_l2(&got, &oracle);
        println!("    depth {n:>4}: logit rel-L2 {d:.4}");
        drifts.push((n, d));
    }

    // THE BAR. Same 0.25 the existing tq gate uses, applied at every depth
    // rather than only at 5 tokens. If this fails at depth, tq4 is not a safe
    // default for long context and the 32K plan needs more bits or a smaller
    // base — which is exactly the decision this test exists to inform.
    for &(n, d) in &drifts {
        assert!(
            d < 0.25,
            "tq4 logit drift {d} at depth {n} exceeds the 0.25 bar the tq gate holds"
        );
    }

    // AND IT MUST NOT COMPOUND. A drift that grows steadily with depth would
    // extrapolate badly to 32K even if every point here passes. Compare the
    // deepest against the shallowest rather than asserting monotonicity, which
    // a softmax makes noisy (see tq_kv_distribution.rs).
    let (shallow, deep) = (drifts[0].1, drifts.last().expect("swept").1);
    assert!(
        deep < shallow.max(0.05) * 3.0,
        "tq4 drift grew from {shallow} at depth {} to {deep} at depth {} — that \
         compounds, and extrapolating it to 32K is not safe",
        drifts[0].0,
        drifts.last().expect("swept").0
    );
}
