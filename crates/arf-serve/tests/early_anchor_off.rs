//! EARLY ANCHORS, the control arm (`ARF_EARLY_ANCHOR` unset): two sessions sharing a system prefix,
//! started together, each prefill it — the second cannot resume from an anchor that is published
//! only when the first finishes. See `tests/early_anchor.rs`.

use std::sync::{Arc, Mutex};

use arf_core::backend::BatchedBackend;
use arf_core::config::EngineConfig;
use arf_core::model::batch::ForwardBatch;
use arf_core::sampling::{SamplingParams, TokenPieceMap};
use arf_serve::actor::{spawn, FinishReason, Job, ModelHandle, StreamItem};

#[derive(Debug, Clone, PartialEq)]
enum SnapCall {
    Save(u64, u64),
    SaveAnchor(u64, u64),
    SaveJunction(u64, u64),
    Restore(u64, u64),
}

/// Deterministic fake model with a snapshot store that restores only keys it saved, as the Metal
/// store answers (the `SnapMock` of `tests/actor_loop.rs`).
struct SnapMock {
    calls: Arc<Mutex<Vec<SnapCall>>>,
}
impl BatchedBackend for SnapMock {
    fn forward_batch(&self, _ids: &[u32], batch: &ForwardBatch) -> Vec<Vec<f32>> {
        batch
            .seqs
            .iter()
            .map(|s| {
                let mut row = vec![0.0f32; 16];
                row[(s.past_len + s.q_len) % 16] = 1.0;
                row
            })
            .collect()
    }
    fn kv_geometry(&self) -> (usize, usize) {
        (1024, 16)
    }
    fn state_save(&self, stream: u64, key: u64) -> (bool, Option<u64>) {
        self.calls.lock().unwrap().push(SnapCall::Save(stream, key));
        (true, None)
    }
    fn state_save_anchor(&self, stream: u64, key: u64) -> (bool, Option<u64>) {
        let mut c = self.calls.lock().unwrap();
        c.push(SnapCall::SaveAnchor(stream, key));
        (true, None)
    }
    fn state_save_junction(&self, stream: u64, key: u64) -> (bool, Option<u64>) {
        let mut c = self.calls.lock().unwrap();
        c.push(SnapCall::SaveJunction(stream, key));
        (true, None)
    }
    fn state_restore(&self, stream: u64, key: u64) -> bool {
        let mut calls = self.calls.lock().unwrap();
        let saved = calls.iter().any(|c| {
            matches!(c, SnapCall::Save(_, k) | SnapCall::SaveAnchor(_, k)
                | SnapCall::SaveJunction(_, k) if *k == key)
        });
        calls.push(SnapCall::Restore(stream, key));
        saved
    }
}

fn server() -> (ModelHandle, Arc<Mutex<Vec<SnapCall>>>) {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let cfg = EngineConfig {
        block_size: 16,
        num_blocks: 1024,
        max_batch_size: 8,
        max_prefill_tokens: 4096,
        enable_prefix_cache: true,
        state_snapshots: true,
        ..Default::default()
    };
    let backend = SnapMock {
        calls: Arc::clone(&calls),
    };
    let (handle, _metrics, _join) = spawn(
        Box::new(backend),
        cfg,
        Arc::new(TokenPieceMap::new(vec![])),
        None,
    );
    (handle, calls)
}

async fn run(
    h: &ModelHandle,
    id: u64,
    prompt_ids: Vec<u32>,
    max_tokens: usize,
    prefix_anchor: Option<usize>,
    tools_anchor: Option<usize>,
) {
    let (tx, mut rx) = tokio::sync::mpsc::channel(256);
    h.submit(Job {
        id,
        prompt_ids,
        params: SamplingParams::greedy(max_tokens),
        token_tx: tx,
        want_logprobs: false,
        top_logprobs: 0,
        response_format: None,
        images: Vec::new(),
        image_positions: Vec::new(),
        image_prompt: None,
        prefix_anchor,
        tools_anchor,
        header_tail: None,
        background: false,
        stop: None,
    })
    .unwrap();
    let mut fin = None;
    while let Some(item) = rx.recv().await {
        if let StreamItem::Done(r) = item {
            fin = Some(r);
        }
    }
    assert_eq!(fin, Some(FinishReason::Length), "job {id}");
}

/// A session: a 1,100-token shared system prefix (anchor 1,100 -> 1,024) and its own message.
fn session(user: u32) -> Vec<u32> {
    let mut p: Vec<u32> = (0..1100).collect();
    p.extend(user..user + 600);
    p
}

#[tokio::test]
async fn without_early_anchors_sessions_started_together_share_nothing() {
    std::env::remove_var("ARF_EARLY_ANCHOR");
    std::env::set_var("ARF_SNAPSHOT_TAIL", "0");
    std::env::set_var("ARF_SNAPSHOT_ALIGN", "128");
    let (h, calls) = server();
    let (a, b) = tokio::join!(
        run(&h, 1, session(10_000), 4, Some(1100), None),
        run(&h, 2, session(20_000), 4, Some(1100), None)
    );
    let _ = (a, b);
    let c = calls.lock().unwrap().clone();
    assert!(
        !c.iter().any(|x| matches!(x, SnapCall::Restore(..))),
        "nothing to resume from while both are running: {c:?}"
    );
}
