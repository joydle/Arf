//! ANCHOR REPLAY through the actor, no GPU: anchors a server took are recorded to
//! disk, and after a "restart" (a new actor over a backend with an empty snapshot store) replaying
//! the recorded prefixes re-takes the SAME keys, so a new session resumes from them as it would
//! have before the restart. Its own test binary: the recording store is process-global.

use std::sync::{Arc, Mutex};

use arf_core::backend::BatchedBackend;
use arf_core::config::EngineConfig;
use arf_core::model::batch::ForwardBatch;
use arf_core::sampling::{SamplingParams, TokenPieceMap};
use arf_serve::actor::{spawn, FinishReason, Job, ModelHandle, StreamItem};
use arf_serve::anchor_replay;

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

/// A Claude Code-shaped session: a 1,100-token tools block (tools anchor 1,100 -> 1,024), a
/// 1,300-token system text (system anchor 2,400 -> 2,304) and the session's own message.
fn session(user: u32) -> Vec<u32> {
    let mut p: Vec<u32> = (0..2400).collect();
    p.extend(user..user + 600);
    p
}

fn anchors(calls: &[SnapCall]) -> Vec<(u64, u64)> {
    calls
        .iter()
        .filter_map(|c| match c {
            SnapCall::SaveAnchor(s, k) => Some((*s, *k)),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn replayed_anchors_are_the_same_keys_and_a_new_session_resumes_from_them() {
    std::env::set_var("ARF_SNAPSHOT_TAIL", "0");
    std::env::set_var("ARF_SNAPSHOT_ALIGN", "128");
    let dir = std::env::temp_dir().join(format!("arf-anchor-replay-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let path = dir.join("anchors.json");
    assert!(anchor_replay::init(path.clone(), "test-model".into()).is_empty());

    // Before the restart: one session takes the tools anchor and the system anchor.
    let (a, a_calls) = server();
    run(&a, 1, session(10_000), 2, Some(2400), Some(1100)).await;
    let taken = anchors(&a_calls.lock().unwrap());
    assert_eq!(taken.len(), 2, "tools + system anchor: {taken:?}");
    let (tools_key, system_key) = (taken[0].1, taken[1].1);

    let recorded = anchor_replay::load(&path, "test-model");
    let mut keys: Vec<u64> = recorded.iter().map(|e| e.key).collect();
    keys.sort();
    let mut want = vec![tools_key, system_key];
    want.sort();
    assert_eq!(keys, want, "both anchors recorded: {recorded:?}");
    for e in &recorded {
        assert_eq!(e.tokens, session(10_000)[..e.at + 1].to_vec());
    }

    // The restart: a new actor whose backend has no snapshots. Replay as the server does.
    let (b, b_calls) = server();
    for (i, e) in anchor_replay::replay_order(recorded)
        .into_iter()
        .enumerate()
    {
        let (prompt, prefix_anchor, tools_anchor) = e.replay();
        run(&b, 100 + i as u64, prompt, 1, prefix_anchor, tools_anchor).await;
    }
    let c = b_calls.lock().unwrap().clone();
    assert_eq!(
        anchors(&c),
        vec![(100, tools_key), (101, system_key)],
        "the replay re-takes the same keys, tools first: {c:?}"
    );
    assert!(
        c.contains(&SnapCall::Restore(101, tools_key)),
        "the system anchor's replay resumes from the replayed tools anchor: {c:?}"
    );

    // A new session after the restart resumes past the whole shared prefix.
    run(&b, 2, session(20_000), 2, Some(2400), Some(1100)).await;
    let c = b_calls.lock().unwrap().clone();
    assert!(
        c.contains(&SnapCall::Restore(2, system_key)),
        "a new session after the restart resumes from the system anchor: {c:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
