//! The actor's SPECULATION GATES against a mid-prompt one-token prefill chunk (2026-09-26,
//! review of the anchor-snapshot commit). Its own test binary because the gates read
//! `ARF_SPEC_K` once per process (a `OnceLock`), and this file turns speculation ON.
//!
//! The failure it pins: a snapshot cut (the prefix anchor at `a`, or the end-of-prompt boundary)
//! can leave a prefill chunk of ONE token, `[a-1, a)` — whenever an earlier chunk ended at
//! `a-1`, e.g. because a decode row took one token of the budget. That row has `q_len == 1`, and
//! both gates tested only `q_len == 1`: the lone gate (`seqs.len() == 1`) and the multi-stream
//! gate (`all(q_len == 1)`). So a verify window `[tok_{a-1}, drafts..]` ran MID-PROMPT; accepted
//! drafts stay in the recurrent state (the verify contract), the run is thrown away because the
//! sequence is still prefilling — and the snapshot taken right after (`state_save_anchor`) held
//! the state of `tokens[..a]` PLUS the drafts. Every later new session restored that.

use std::sync::{Arc, Mutex};

use arf_core::backend::{BatchedBackend, SegReq};
use arf_core::config::EngineConfig;
use arf_core::model::batch::ForwardBatch;
use arf_core::sampling::{SamplingParams, TokenPieceMap};
use arf_serve::actor::{spawn, FinishReason, Job, StreamItem};

/// A backend that always has a one-token draft (MTP chain, block draft) and records every verify
/// it is asked for as `(stream, prefix_len)`. Its verifies decline (`None`), so the actor falls
/// back to the plain step it would have taken — the recording is the whole point.
struct DraftingMock {
    vocab: usize,
    verifies: Arc<Mutex<Vec<(u64, usize)>>>,
    /// Keys saved so far — a restore succeeds only for one of them, as the Metal store answers
    /// (2026-09-26: the junction test below is the first here to restore anything).
    saved: Mutex<Vec<u64>>,
}

impl BatchedBackend for DraftingMock {
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
        (512, 16)
    }
    fn mtp_draft_chain(&self, _tok: u32, _past_len: usize, _slots: &[u32], _k: usize) -> Vec<u32> {
        vec![5]
    }
    fn block_draft(&self, _tok: u32, _past_len: usize, _stream: u64) -> Vec<u32> {
        vec![5]
    }
    fn verify_window(
        &self,
        _window: &[u32],
        prefix_len: usize,
        _prefix_slots: &[u32],
        stream_id: u64,
    ) -> Option<Vec<u32>> {
        self.verifies.lock().unwrap().push((stream_id, prefix_len));
        None
    }
    fn verify_segments(&self, segs: &[SegReq<'_>]) -> Option<Vec<Vec<u32>>> {
        let mut v = self.verifies.lock().unwrap();
        v.extend(segs.iter().map(|s| (s.stream, s.prefix_len)));
        None
    }
    fn state_save(&self, _stream: u64, key: u64) -> (bool, Option<u64>) {
        self.saved.lock().unwrap().push(key);
        (true, None)
    }
    fn state_restore(&self, _stream: u64, key: u64) -> bool {
        self.saved.lock().unwrap().contains(&key)
    }
}

fn job(
    id: u64,
    prompt: Vec<u32>,
    max_tokens: usize,
    anchor: Option<usize>,
) -> (Job, tokio::sync::mpsc::Receiver<StreamItem>) {
    let (tx, rx) = tokio::sync::mpsc::channel(4096);
    let job = Job {
        id,
        prompt_ids: prompt,
        params: SamplingParams::greedy(max_tokens),
        token_tx: tx,
        want_logprobs: false,
        top_logprobs: 0,
        response_format: None,
        images: Vec::new(),
        image_positions: Vec::new(),
        image_prompt: None,
        prefix_anchor: anchor,
        tools_anchor: None,
        header_tail: None,
        background: false,
        stop: None,
    };
    (job, rx)
}

async fn finish(mut rx: tokio::sync::mpsc::Receiver<StreamItem>) -> Option<FinishReason> {
    while let Some(item) = rx.recv().await {
        if let StreamItem::Done(r) = item {
            return Some(r);
        }
    }
    None
}

fn cfg(max_prefill_tokens: usize) -> EngineConfig {
    EngineConfig {
        block_size: 16,
        num_blocks: 512,
        max_batch_size: 8,
        max_prefill_tokens,
        enable_prefix_cache: true,
        state_snapshots: true,
        ..Default::default()
    }
}

/// Speculation ON for this process, before any actor reads the gate.
fn spec_on() {
    std::env::set_var("ARF_SPEC_K", "1");
}

/// No verify may run at a position before the prompt's last token: `prefix_len + 1 < len`
/// is a verify inside the prompt.
fn assert_no_mid_prompt_verify(verifies: &[(u64, usize)], id: u64, prompt_len: usize) {
    let bad: Vec<_> = verifies
        .iter()
        .filter(|&&(s, p)| s == id && p + 1 < prompt_len)
        .collect();
    assert!(
        bad.is_empty(),
        "a verify ran inside stream {id}'s prompt ({prompt_len} tokens): {bad:?}"
    );
}

/// ALONE: a 1,023-token budget makes the first chunk `[0, 1023)`, and the prefix anchor at
/// 1,100 -> 1,024 cuts the next one to `[1023, 1024)` — one token, mid-prompt, lone. The
/// lone gate must not verify there; it still speculates once the sequence decodes.
#[tokio::test]
async fn a_lone_one_token_anchor_chunk_does_not_speculate() {
    spec_on();
    let verifies = Arc::new(Mutex::new(Vec::new()));
    let backend = DraftingMock {
        vocab: 64,
        verifies: Arc::clone(&verifies),
        saved: Mutex::new(Vec::new()),
    };
    let (handle, _metrics, _join) = spawn(
        Box::new(backend),
        cfg(1023),
        Arc::new(TokenPieceMap::new(vec![])),
        None,
    );
    let mut prompt: Vec<u32> = (0..1100).map(|t| t % 60).collect();
    prompt.extend((0..600).map(|t| (t * 7) % 60));
    let len = prompt.len();
    let (j, rx) = job(1, prompt, 4, Some(1100));
    handle.submit(j).unwrap();
    assert_eq!(finish(rx).await, Some(FinishReason::Length));
    let v = verifies.lock().unwrap().clone();
    assert_no_mid_prompt_verify(&v, 1, len);
    assert!(
        v.iter().any(|&(s, p)| s == 1 && p + 1 >= len),
        "the gate still speculates on decode rows (the mock always drafts): {v:?}"
    );
}

/// WITH A DECODE ROW: stream 1 is decoding when stream 2 (the new session) arrives, so stream
/// 2's first chunk is one token short of the budget and a later one is `[a-1, a)` beside
/// stream 1's decode row: `[D:1, A:1]`. The multi-stream gate must not verify stream 2 there.
#[tokio::test]
async fn a_one_token_anchor_chunk_beside_a_decode_row_does_not_speculate() {
    spec_on();
    let verifies = Arc::new(Mutex::new(Vec::new()));
    let backend = DraftingMock {
        vocab: 64,
        verifies: Arc::clone(&verifies),
        saved: Mutex::new(Vec::new()),
    };
    let (handle, _metrics, _join) = spawn(
        Box::new(backend),
        cfg(1024),
        Arc::new(TokenPieceMap::new(vec![])),
        None,
    );
    // Stream 1: a short prompt that decodes for a long time.
    let (d, mut drx) = job(1, (0..20).collect(), 400, None);
    handle.submit(d).unwrap();
    // Wait until it is decoding (its first token is out).
    match drx.recv().await {
        Some(StreamItem::Token(..)) => {}
        Some(StreamItem::Done(r)) => panic!("stream 1 finished early: {r:?}"),
        None => panic!("stream 1 closed"),
    }
    let mut prompt: Vec<u32> = (0..1100).map(|t| t % 60).collect();
    prompt.extend((0..600).map(|t| (t * 7) % 60));
    let len = prompt.len();
    let (a, arx) = job(2, prompt, 4, Some(1100));
    handle.submit(a).unwrap();
    assert_eq!(finish(arx).await, Some(FinishReason::Length));
    let _ = finish(drx).await;
    let v = verifies.lock().unwrap().clone();
    assert_no_mid_prompt_verify(&v, 2, len);
}

/// A JUNCTION CUT (2026-09-26) leaves the same one-token chunk. Session 1 = S (1,100, anchored)
/// + R (2,000 shared) + its task; session 2 shares S + R, resumes at the anchor (1,024) and plans
/// a junction at 3,072 (its KV match, 3,088, floored to the window). A 2,047-token budget makes
/// its first chunk `[1024, 3071)` and the next `[3071, 3072)` — one token, mid-prompt, lone,
/// carrying the junction key. The lone gate must not verify there: the junction saved right
/// after would hold the drafts too, and every later session of the project would restore them.
#[tokio::test]
async fn a_lone_one_token_junction_chunk_does_not_speculate() {
    spec_on();
    let verifies = Arc::new(Mutex::new(Vec::new()));
    let backend = DraftingMock {
        vocab: 64,
        verifies: Arc::clone(&verifies),
        saved: Mutex::new(Vec::new()),
    };
    let (handle, _metrics, _join) = spawn(
        Box::new(backend),
        cfg(2047),
        Arc::new(TokenPieceMap::new(vec![])),
        None,
    );
    let session = |task: u32| {
        let mut p: Vec<u32> = (0..1100).map(|t| t % 60).collect();
        p.extend((0..2000).map(|t| (t * 7) % 60));
        p.extend((0..600).map(|t| (t * 11 + task) % 60));
        p
    };
    let (j1, rx1) = job(1, session(1), 4, Some(1100));
    handle.submit(j1).unwrap();
    assert_eq!(finish(rx1).await, Some(FinishReason::Length));
    let p2 = session(5);
    let len = p2.len();
    let (j2, rx2) = job(2, p2, 4, Some(1100));
    handle.submit(j2).unwrap();
    assert_eq!(finish(rx2).await, Some(FinishReason::Length));
    let v = verifies.lock().unwrap().clone();
    assert_no_mid_prompt_verify(&v, 2, len);
    assert!(
        v.iter().any(|&(s, p)| s == 2 && p + 1 >= len),
        "the gate still speculates on session 2's decode rows (the mock always drafts): {v:?}"
    );
}
