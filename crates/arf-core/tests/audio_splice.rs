//! Qwen3-Omni AUDIO prompts through scheduler -> build_forward (M3, 2026-09-27): the encoder's
//! rows land on exactly the `<|audio_pad|>` rows, in order, across chunked prefill; the rows attend
//! causally (no bidirectional span); positions stay plain and no M-RoPE table appears, so the
//! batch reaches the backend as a text batch with an embedding override. Model-free.

use arf_core::config::EngineConfig;
use arf_core::engine::build_forward;
use arf_core::sampling::SamplingParams;
use arf_core::scheduler::{ImagePrompt, Request, Scheduler};

const AUDIO_START: u32 = 151_669;
const AUDIO_PAD: u32 = 151_675;
const AUDIO_END: u32 = 151_670;
const HIDDEN: usize = 3;

fn cfg(max_prefill_tokens: usize) -> EngineConfig {
    EngineConfig {
        block_size: 4,
        num_blocks: 32,
        max_batch_size: 8,
        max_prefill_tokens,
        ..Default::default()
    }
}

/// `A B <as> [5 pads] <ae> C D`: pads at 3..8. Row `i` of the embedding is `[i, i, i] + 0.5`.
fn audio_request(id: u64, causal: bool) -> (Request, Vec<u32>) {
    let prompt: Vec<u32> = vec![
        10,
        11,
        AUDIO_START,
        AUDIO_PAD,
        AUDIO_PAD,
        AUDIO_PAD,
        AUDIO_PAD,
        AUDIO_PAD,
        AUDIO_END,
        12,
        13,
    ];
    let img = ImagePrompt {
        embeds: (0..5)
            .flat_map(|i| [i as f32 + 0.5; HIDDEN])
            .collect::<Vec<f32>>(),
        hidden: HIDDEN,
        positions: (3..8).collect(),
        mrope: None,
        causal,
    };
    (
        Request::with_image(id, prompt.clone(), SamplingParams::greedy(4), img),
        prompt,
    )
}

fn rows_of(fb: &arf_core::model::batch::ForwardBatch) -> Vec<(usize, f32)> {
    match &fb.image_embeds {
        None => Vec::new(),
        Some(ie) => ie
            .rows
            .iter()
            .enumerate()
            .map(|(i, &r)| (r, ie.embeds[i * ie.hidden]))
            .collect(),
    }
}

#[test]
fn audio_rows_land_on_the_pads_causally_with_plain_positions() {
    let mut s = Scheduler::new(cfg(4096));
    s.add(Request::new(0, vec![1; 2], SamplingParams::greedy(4))); // a text neighbour
    let (req, prompt) = audio_request(1, true);
    s.add(req);
    let plan = s.schedule().unwrap().expect("a batch");
    let (ids, fb) = build_forward(&plan, 4);
    // the neighbour occupies flat rows 0..2, the audio prompt 2..13
    assert_eq!(&ids[2..], &prompt[..]);
    let pads: Vec<usize> = ids
        .iter()
        .enumerate()
        .filter(|(_, &t)| t == AUDIO_PAD)
        .map(|(i, _)| i)
        .collect();
    let got = rows_of(&fb);
    assert_eq!(
        got.iter().map(|g| g.0).collect::<Vec<_>>(),
        pads,
        "every pad row, and only pad rows, is overridden"
    );
    assert_eq!(
        got.iter().map(|g| g.1).collect::<Vec<_>>(),
        vec![0.5, 1.5, 2.5, 3.5, 4.5],
        "encoder rows in order"
    );
    assert!(
        fb.seqs.iter().all(|s| s.image_spans.is_empty()),
        "audio attends causally: no bidirectional span"
    );
    assert!(fb.mrope_positions.is_none(), "no M-RoPE table for audio");
    assert_eq!(&fb.positions[2..], &(0..11).collect::<Vec<u32>>()[..]);

    // decode: an ordinary text step
    s.commit_tokens(&[5, 5]).unwrap();
    let plan = s.schedule().unwrap().expect("decode");
    let (_ids, fb) = build_forward(&plan, 4);
    assert!(fb.image_embeds.is_none());
    assert_eq!(fb.positions, vec![2, 11]);
}

/// A prompt chunked mid-audio: each chunk carries exactly its own pads' rows, with the right
/// embedding for each (ordinal among the pads, not the chunk).
#[test]
fn chunked_prefill_splits_the_audio_rows() {
    let mut s = Scheduler::new(cfg(5));
    let (req, _prompt) = audio_request(1, true);
    s.add(req);
    let plan = s.schedule().unwrap().expect("chunk 1");
    let (_ids, fb) = build_forward(&plan, 4);
    assert_eq!(plan.seqs[0].q_len, 5);
    assert_eq!(rows_of(&fb), vec![(3, 0.5), (4, 1.5)]);
    assert!(fb.seqs[0].image_spans.is_empty());
    s.commit_tokens(&[0]).unwrap(); // mid-prefill: the token is discarded
    let plan = s.schedule().unwrap().expect("chunk 2");
    let (_ids, fb) = build_forward(&plan, 4);
    assert_eq!(plan.seqs[0].past_len, 5);
    // local rows 0..3 of this chunk are prompt positions 5..8
    assert_eq!(rows_of(&fb), vec![(0, 2.5), (1, 3.5), (2, 4.5)]);
    assert!(fb.seqs[0].image_spans.is_empty());
}

/// The control: the SAME prompt with `causal: false` (Gemma-3's rule) gets its span, so the flag,
/// not an accident of the fixture, is what keeps audio causal.
#[test]
fn the_causal_flag_is_what_removes_the_span() {
    let mut s = Scheduler::new(cfg(4096));
    let (req, _) = audio_request(1, false);
    s.add(req);
    let plan = s.schedule().unwrap().expect("a batch");
    let (_ids, fb) = build_forward(&plan, 4);
    assert_eq!(fb.seqs[0].image_spans, vec![(3, 5)]);
}
