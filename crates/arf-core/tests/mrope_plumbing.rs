//! Qwen3.8 image prompts through scheduler -> build_forward: the (t, h, w) rope positions reach
//! `ForwardBatch::mrope_positions` for the image sequence (prefill AND decode), image tokens get
//! no bidirectional span, and a text batch is byte-for-byte what it was (`mrope_positions: None`,
//! plain `positions`). Model-free.

use arf_core::config::EngineConfig;
use arf_core::engine::build_forward;
use arf_core::model::mrope::{mrope_layout, ImageGrid};
use arf_core::sampling::SamplingParams;
use arf_core::scheduler::{ImagePrompt, Request, Scheduler};

fn cfg() -> EngineConfig {
    EngineConfig {
        block_size: 4,
        num_blocks: 32,
        max_batch_size: 8,
        max_prefill_tokens: 4096,
        ..Default::default()
    }
}

/// The hand-worked prompt of `mrope::tests`: `A B <vs> [2x3 image] <ve> C` (11 tokens).
fn image_request(id: u64) -> Request {
    let grid = ImageGrid {
        start: 3,
        grid_h: 2,
        grid_w: 3,
    };
    let prompt: Vec<u32> = vec![10, 11, 12, 99, 99, 99, 99, 99, 99, 13, 14];
    let hidden = 2;
    let img = ImagePrompt {
        embeds: (0..6 * hidden).map(|i| i as f32).collect(),
        hidden,
        positions: (3..9).collect(),
        mrope: Some(mrope_layout(prompt.len(), &[grid])),
        causal: true,
    };
    Request::with_image(id, prompt, SamplingParams::greedy(4), img)
}

#[test]
fn text_only_batches_are_unchanged() {
    let mut s = Scheduler::new(cfg());
    s.add(Request::new(0, vec![1; 7], SamplingParams::greedy(4)));
    s.add(Request::new(1, vec![2; 3], SamplingParams::greedy(4)));
    let plan = s.schedule().unwrap().expect("a batch");
    let (_ids, fb) = build_forward(&plan, 4);
    assert!(
        fb.mrope_positions.is_none(),
        "a text batch carries no M-RoPE table"
    );
    assert_eq!(fb.positions, vec![0, 1, 2, 3, 4, 5, 6, 0, 1, 2]);
    assert!(fb.seqs.iter().all(|s| s.image_spans.is_empty()));
    assert!(plan.seqs.iter().all(|sp| sp.mrope.is_none()));
    // and decode
    s.commit_tokens(&[5, 5]).unwrap();
    let plan = s.schedule().unwrap().expect("decode");
    let (_ids, fb) = build_forward(&plan, 4);
    assert!(fb.mrope_positions.is_none());
    assert_eq!(fb.positions, vec![7, 3]);
}

#[test]
fn image_sequence_carries_mrope_rows_and_no_bidirectional_span() {
    let mut s = Scheduler::new(cfg());
    s.add(Request::new(0, vec![1; 2], SamplingParams::greedy(4))); // a text neighbour
    s.add(image_request(1));
    let plan = s.schedule().unwrap().expect("a batch");
    let (_ids, fb) = build_forward(&plan, 4);
    let m = fb
        .mrope_positions
        .as_ref()
        .expect("image batch has a table");
    assert_eq!(m.len(), fb.positions.len());
    // text neighbour: [p,p,p]
    assert_eq!(&m[..2], &[[0, 0, 0], [1, 1, 1]]);
    // the image sequence, straight from the hand-worked layout
    assert_eq!(
        &m[2..],
        &[
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [3, 3, 4],
            [3, 3, 5],
            [3, 4, 3],
            [3, 4, 4],
            [3, 4, 5],
            [6, 6, 6],
            [7, 7, 7]
        ]
    );
    // positions (the KV / causal index) are untouched
    assert_eq!(&fb.positions[2..], &(0..11).collect::<Vec<u32>>()[..]);
    // embeds still override the pad rows; spans are Gemma-only
    assert_eq!(
        fb.image_embeds.as_ref().unwrap().rows,
        (5..11).collect::<Vec<_>>()
    );
    assert!(fb.seqs.iter().all(|s| s.image_spans.is_empty()));

    // DECODE: the image sequence's first generated token (KV index 11) rotates at 11 - 3 = 8.
    s.commit_tokens(&[5, 5]).unwrap();
    let plan = s.schedule().unwrap().expect("decode");
    let (_ids, fb) = build_forward(&plan, 4);
    assert_eq!(fb.positions, vec![2, 11]);
    assert_eq!(fb.mrope_positions, Some(vec![[2, 2, 2], [8, 8, 8]]));
}

/// Two image requests with the same token ids (same grid, different pictures) must not share
/// KV: the second one prefills from zero. A text request with the same ids would reuse.
#[test]
fn an_image_sequence_never_claims_cached_blocks() {
    let cfg = EngineConfig {
        enable_prefix_cache: true,
        ..cfg()
    };
    let mut s = Scheduler::new(cfg);
    s.add(image_request(1));
    let plan = s.schedule().unwrap().expect("prefill");
    assert_eq!(plan.seqs[0].past_len, 0);
    // finish it (max_tokens 4)
    for _ in 0..4 {
        s.commit_tokens(&[5]).unwrap();
        if s.schedule().unwrap().is_none() {
            break;
        }
    }
    s.add(image_request(2));
    let plan = s.schedule().unwrap().expect("second prefill");
    let sp = plan
        .seqs
        .iter()
        .find(|sp| sp.id == 2)
        .expect("seq 2 scheduled");
    assert_eq!(
        sp.past_len, 0,
        "an image prompt must not reuse another image's KV"
    );
}
