//! End-to-end engine tests on the synthetic model.

mod common;

use std::collections::HashMap;

use arf_core::config::EngineConfig;
use arf_core::engine::LlmEngine;
use arf_core::sampling::SamplingParams;

fn engine() -> LlmEngine {
    let cfg = EngineConfig {
        block_size: 4,
        num_blocks: 64,
        max_batch_size: 8,
        max_prefill_tokens: 4096,
        ..Default::default()
    };
    LlmEngine::new(common::tiny_model(), cfg).unwrap()
}

#[test]
fn greedy_generation_is_bounded_and_deterministic() {
    let mut e1 = engine();
    let mut e2 = engine();
    let prompt = vec![3u32, 7, 1];
    let a = e1
        .generate(prompt.clone(), SamplingParams::greedy(5))
        .unwrap();
    let b = e2.generate(prompt, SamplingParams::greedy(5)).unwrap();
    assert!(!a.is_empty() && a.len() <= 5);
    assert_eq!(a, b, "greedy decoding must be deterministic");
}

#[test]
fn stop_token_halts_generation() {
    let mut e = engine();
    // Discover the first greedily-produced token, then make it a stop token.
    let first = e
        .generate(vec![2u32, 5], SamplingParams::greedy(8))
        .unwrap()[0];

    let mut e2 = engine();
    let params = SamplingParams {
        max_tokens: 8,
        stop_tokens: vec![first],
        ..Default::default()
    };
    let out = e2.generate(vec![2u32, 5], params).unwrap();
    assert!(out.is_empty(), "stop token on first step yields no output");
}

#[test]
fn concurrent_requests_each_complete() {
    let mut e = engine();
    let id_a = e
        .submit(vec![3u32, 1, 4], SamplingParams::greedy(4))
        .unwrap();
    let id_b = e.submit(vec![2u32, 7], SamplingParams::greedy(6)).unwrap();

    let mut outputs: HashMap<u64, Vec<u32>> = HashMap::new();
    while e.has_unfinished() {
        for out in e.step().unwrap() {
            if out.finish_reason != Some(arf_core::scheduler::FinishReason::Stop) {
                outputs.entry(out.id).or_default().push(out.token);
            }
        }
    }
    assert_eq!(outputs[&id_a].len(), 4);
    assert_eq!(outputs[&id_b].len(), 6);
}

#[test]
fn rejects_invalid_requests() {
    use arf_core::error::ArfError;
    let mut e = engine();
    let vocab = e.model().config().vocab_size;

    // Empty prompt.
    assert!(matches!(
        e.submit(vec![], SamplingParams::greedy(4)),
        Err(ArfError::EmptyPrompt)
    ));
    // Token id past the vocabulary must not reach (and panic) the embedding.
    assert!(matches!(
        e.submit(vec![vocab as u32], SamplingParams::greedy(4)),
        Err(ArfError::TokenOutOfRange { .. })
    ));
    // Bad sampling params.
    let bad = SamplingParams {
        temperature: -1.0,
        ..Default::default()
    };
    assert!(e.submit(vec![1], bad).is_err());
}

#[test]
fn streaming_yields_same_tokens_as_generate() {
    use arf_core::engine::StepInfo;

    // Two identical engines from the same fixed-seed synthetic model.
    let mut e1 = engine();
    let mut e2 = engine();
    let prompt = vec![3u32, 7, 1];
    let params = SamplingParams::greedy(8);

    let collected = e1.generate(prompt.clone(), params.clone()).unwrap();

    let mut streamed: Vec<u32> = Vec::new();
    e2.generate_streaming(prompt, params, &mut |info: StepInfo| {
        streamed.push(info.token);
    })
    .unwrap();

    assert_eq!(
        collected, streamed,
        "generate_streaming must yield the same token sequence as generate"
    );
}

#[test]
fn streaming_reports_plausible_kv_occupancy() {
    use arf_core::engine::StepInfo;

    let mut e = engine();
    let mut infos: Vec<StepInfo> = Vec::new();
    e.generate_streaming(vec![3, 7, 1], SamplingParams::greedy(6), &mut |info| {
        infos.push(info);
    })
    .unwrap();

    assert!(!infos.is_empty());
    for info in &infos {
        assert!(info.kv_blocks_total > 0);
        assert!(info.kv_blocks_used <= info.kv_blocks_total);
        assert!(
            info.context_len >= 3,
            "context grows beyond the 3-token prompt"
        );
    }
    // Context length is non-decreasing across decode steps.
    for w in infos.windows(2) {
        assert!(w[1].context_len >= w[0].context_len);
    }
}

#[test]
fn chunked_prefill_is_bit_exact_vs_whole_prompt_prefill() {
    // THE invariant of the design (spec §5 #3): chunking a prompt's prefill
    // across steps must produce the identical token stream as one-shot prefill.
    let mk = |budget: usize| {
        LlmEngine::new(
            common::tiny_model(),
            EngineConfig {
                block_size: 4,
                num_blocks: 64,
                max_batch_size: 8,
                max_prefill_tokens: budget,
                ..Default::default()
            },
        )
        .unwrap()
    };
    let prompt = vec![3u32, 7, 1, 4, 2, 9, 6, 5]; // 8 tokens
    let whole = mk(4096)
        .generate(prompt.clone(), SamplingParams::greedy(6))
        .unwrap();
    let chunked = mk(3) // chunks of 3/3/2
        .generate(prompt.clone(), SamplingParams::greedy(6))
        .unwrap();
    assert_eq!(whole, chunked, "chunked prefill must be bit-exact");

    // Boundary: prompt length an exact multiple of the budget (chunks 4/4 —
    // the final chunk exactly fills the budget).
    let prompt = vec![3u32, 7, 1, 4, 2, 9, 6, 5];
    let whole = mk(4096)
        .generate(prompt.clone(), SamplingParams::greedy(6))
        .unwrap();
    let exact = mk(4).generate(prompt, SamplingParams::greedy(6)).unwrap();
    assert_eq!(whole, exact, "exact-multiple chunking must be bit-exact");
}
