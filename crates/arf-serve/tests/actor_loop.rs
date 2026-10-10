//! The actor step-loop, tested through a mock backend (no GPU): admission of
//! concurrent jobs, per-request streams, finish reasons, eviction isolation.

use std::sync::Arc;

use arf_core::backend::BatchedBackend;
use arf_core::config::EngineConfig;
use arf_core::model::batch::ForwardBatch;
use arf_core::sampling::{SamplingParams, TokenPieceMap};
use arf_serve::actor::{spawn, FinishReason, Job, StreamItem};

/// Deterministic fake model: next token = (past_len + q_len) % vocab. A prompt
/// of length L therefore generates L%V, (L+1)%V, (L+2)%V, ...
struct Mock {
    vocab: usize,
}
impl BatchedBackend for Mock {
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

fn cfg() -> EngineConfig {
    EngineConfig {
        block_size: 4,
        num_blocks: 64,
        max_batch_size: 8,
        max_prefill_tokens: 16,
        ..Default::default()
    }
}

fn greedy(max_tokens: usize) -> SamplingParams {
    SamplingParams::greedy(max_tokens)
}

/// An empty piece map — sufficient for tests that don't use grammar constraints.
fn empty_piece_map() -> Arc<TokenPieceMap> {
    Arc::new(TokenPieceMap::new(vec![]))
}

async fn collect(
    mut rx: tokio::sync::mpsc::Receiver<StreamItem>,
) -> (Vec<u32>, Option<FinishReason>) {
    let mut toks = Vec::new();
    let mut fin = None;
    while let Some(item) = rx.recv().await {
        match item {
            StreamItem::Token(t, _lp) => toks.push(t),
            StreamItem::Done(r) => {
                fin = Some(r);
                break;
            }
        }
    }
    (toks, fin)
}

#[tokio::test]
async fn concurrent_jobs_each_get_their_own_correct_stream() {
    let (handle, _metrics, _join) =
        spawn(Box::new(Mock { vocab: 16 }), cfg(), empty_piece_map(), None);
    let mut rxs = Vec::new();
    for (id, plen) in [(0u64, 3usize), (1, 5), (2, 2)] {
        let (tx, rx) = tokio::sync::mpsc::channel(256);
        handle
            .submit(Job {
                id,
                prompt_ids: vec![1; plen],
                params: greedy(4),
                token_tx: tx,
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
        rxs.push((plen, rx));
    }
    for (plen, rx) in rxs {
        let (toks, fin) = collect(rx).await;
        let want: Vec<u32> = (0..4).map(|i| ((plen + i) % 16) as u32).collect();
        assert_eq!(toks, want, "prompt len {plen}");
        assert_eq!(fin, Some(FinishReason::Length));
    }
}

#[tokio::test]
async fn stop_token_finishes_without_emitting_it() {
    let (handle, _metrics, _join) =
        spawn(Box::new(Mock { vocab: 16 }), cfg(), empty_piece_map(), None);
    let (tx, rx) = tokio::sync::mpsc::channel(256);
    // First generated token for a 3-prompt is 3 -> make it a stop token.
    let params = SamplingParams {
        max_tokens: 8,
        stop_tokens: vec![3],
        ..SamplingParams::default()
    };
    handle
        .submit(Job {
            id: 0,
            prompt_ids: vec![1; 3],
            params,
            token_tx: tx,
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
    let (toks, fin) = collect(rx).await;
    assert!(toks.is_empty(), "stop token is not emitted");
    assert_eq!(fin, Some(FinishReason::Stop));
}

/// STOP STRINGS through the real actor and scheduler: the job's stop check fires on
/// the token that completes the stop string; that token IS emitted (its text before the stop
/// string is the end of the reply — the HTTP layer cuts it), the stream ends `Stop`, and the
/// sequence is retired there: the check is never asked about a later token, so nothing more was
/// generated for it.
#[tokio::test]
async fn stop_check_ends_the_job_on_the_completing_token_and_emits_it() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let (handle, metrics, _join) =
        spawn(Box::new(Mock { vocab: 16 }), cfg(), empty_piece_map(), None);
    let (tx, rx) = tokio::sync::mpsc::channel(256);
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&calls);
    // A 3-token prompt generates 3, 4, 5, 6, ...; the "stop string" is completed by 5 after 4.
    let check = arf_core::scheduler::StopCheck::new(move |out: &[u32]| {
        seen.fetch_add(1, Ordering::Relaxed);
        out.ends_with(&[4, 5])
    });
    handle
        .submit(Job {
            id: 0,
            prompt_ids: vec![1; 3],
            params: greedy(8),
            token_tx: tx,
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
            stop: Some(check),
        })
        .unwrap();
    let (toks, fin) = collect(rx).await;
    assert_eq!(
        toks,
        vec![3, 4, 5],
        "the completing token is emitted, nothing after it"
    );
    assert_eq!(fin, Some(FinishReason::Stop));
    assert_eq!(
        calls.load(Ordering::Relaxed),
        3,
        "asked once per generated token, then retired"
    );
    assert_eq!(
        metrics
            .requests_finished
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );
}

#[tokio::test]
async fn dropped_client_is_evicted_and_others_complete() {
    let (handle, _metrics, _join) =
        spawn(Box::new(Mock { vocab: 16 }), cfg(), empty_piece_map(), None);
    // Victim: receiver dropped immediately (client disconnect).
    let (tx_dead, rx_dead) = tokio::sync::mpsc::channel(256);
    drop(rx_dead);
    handle
        .submit(Job {
            id: 0,
            prompt_ids: vec![1; 4],
            params: greedy(64),
            token_tx: tx_dead,
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
    // Survivor submitted after: must complete normally despite the dead peer.
    let (tx, rx) = tokio::sync::mpsc::channel(256);
    handle
        .submit(Job {
            id: 1,
            prompt_ids: vec![1; 2],
            params: greedy(3),
            token_tx: tx,
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
    let (toks, fin) = collect(rx).await;
    assert_eq!(toks, vec![2, 3, 4]);
    assert_eq!(fin, Some(FinishReason::Length));
}

/// 2026-09-26: a lone cancelled request left `arf_running_sequences 1` on an idle server — the
/// gauges were published before the eviction and an idle actor publishes no further step.
#[tokio::test]
async fn lone_eviction_republishes_the_gauges() {
    use std::sync::atomic::Ordering;
    let (handle, metrics, _join) =
        spawn(Box::new(Mock { vocab: 16 }), cfg(), empty_piece_map(), None);
    let (tx_dead, rx_dead) = tokio::sync::mpsc::channel(256);
    drop(rx_dead);
    handle
        .submit(Job {
            id: 0,
            prompt_ids: vec![1; 4],
            params: greedy(64),
            token_tx: tx_dead,
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
    // Poll for the settled state (the stores are Relaxed and land in some order); 5 s bound.
    for _ in 0..500 {
        if metrics.evictions.load(Ordering::Relaxed) > 0
            && metrics.running.load(Ordering::Relaxed) == 0
        {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(metrics.evictions.load(Ordering::Relaxed), 1);
    assert_eq!(metrics.running.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.kv_blocks_used.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn empty_prompt_and_zero_budget_finish_immediately() {
    let (handle, _metrics, _join) =
        spawn(Box::new(Mock { vocab: 16 }), cfg(), empty_piece_map(), None);
    let (tx, rx) = tokio::sync::mpsc::channel(8);
    handle
        .submit(Job {
            id: 0,
            prompt_ids: vec![],
            params: greedy(4),
            token_tx: tx,
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
    let (toks, fin) = collect(rx).await;
    assert!(toks.is_empty());
    assert_eq!(fin, Some(FinishReason::Length));
}

// ---------------------------------------------------------------------------
// Regression tests for the Ok(None) busy-spin guard
// ---------------------------------------------------------------------------

/// A mock backend with a tiny KV pool: 2 blocks × 4 tokens = 8 token slots.
/// Used to force a job whose prompt needs more blocks than the pool holds.
struct SmallMock {
    vocab: usize,
}
impl BatchedBackend for SmallMock {
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
        (2, 4) // 2 blocks × 4 tokens = 8 slots total
    }
}

fn small_cfg() -> EngineConfig {
    EngineConfig {
        block_size: 4,
        num_blocks: 2,
        max_batch_size: 8,
        max_prefill_tokens: 64,
        ..Default::default()
    }
}

/// Regression: when `schedule()` returns `Ok(None)` (a waiting job that can't
/// be admitted — its prompt needs more blocks than the pool holds), the actor
/// must block on `recv`, NOT spin at 100% CPU.
///
/// Without the fix (`Ok(None) => continue`), the actor loops forever and the
/// join never returns. With the fix, dropping the handle closes the channel;
/// the actor wakes from `recv` with `Err(_)` and exits → join completes.
///
/// We run the join in a helper thread with a 5-second timeout so that if the
/// spin is ever re-introduced the test fails immediately with a diagnostic
/// message instead of blocking the test suite indefinitely.
#[test]
fn actor_blocks_not_spins_when_nothing_is_schedulable() {
    // 9-token prompt needs ceil(9/4) = 3 blocks; pool only has 2 → can never
    // be admitted. schedule() will see: running=[], waiting=[oversized] →
    // admit_waiting pushes it back → running still [] → Ok(None).
    let (handle, _metrics, join) = spawn(
        Box::new(SmallMock { vocab: 16 }),
        small_cfg(),
        empty_piece_map(),
        None,
    );

    let (tx, _rx) = tokio::sync::mpsc::channel(8);
    handle
        .submit(Job {
            id: 0,
            prompt_ids: vec![1u32; 9], // needs 3 blocks, pool has 2
            params: greedy(1),
            token_tx: tx,
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

    // Drop the handle to close the job channel. With the fix the actor is
    // blocking on rx.recv() and will return immediately on Err(_). With the
    // spin it never reaches recv and loops forever.
    drop(handle);

    // Join with a timeout via a wrapper thread + channel.
    let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
    std::thread::spawn(move || {
        let _ = join.join();
        let _ = done_tx.send(());
    });
    done_rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("actor did not exit within 5 s — likely busy-spinning on Ok(None)");
}

/// Complementary: after the actor serves one job (goes idle, blocking on recv),
/// a second job submitted later must complete correctly. This proves the loop
/// correctly transitions through the idle-block state and resumes work.
#[tokio::test]
async fn actor_resumes_after_going_idle_between_sequential_jobs() {
    let (handle, _metrics, _join) =
        spawn(Box::new(Mock { vocab: 16 }), cfg(), empty_piece_map(), None);

    // First job: 3-token prompt, generate 2 tokens.
    let (tx1, rx1) = tokio::sync::mpsc::channel(256);
    handle
        .submit(Job {
            id: 0,
            prompt_ids: vec![1u32; 3],
            params: greedy(2),
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
    let (toks1, fin1) = collect(rx1).await;
    // Deterministic: tokens are (3%16), (4%16) = 3, 4
    assert_eq!(toks1, vec![3, 4]);
    assert_eq!(fin1, Some(FinishReason::Length));

    // First job is done. Actor is now idle (has_unfinished() == false) and is
    // blocking on rx.recv(). Submit a second job — it must wake up and complete.
    let (tx2, rx2) = tokio::sync::mpsc::channel(256);
    handle
        .submit(Job {
            id: 1,
            prompt_ids: vec![1u32; 5],
            params: greedy(2),
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
    let (toks2, fin2) = collect(rx2).await;
    // Deterministic: tokens are (5%16), (6%16) = 5, 6
    assert_eq!(toks2, vec![5, 6]);
    assert_eq!(fin2, Some(FinishReason::Length));
}

// ---------------------------------------------------------------------------
// pending_close outbox: Done terminator is never lost under backpressure
// ---------------------------------------------------------------------------

/// A finishing seq whose channel is FULL at the moment it finishes must still
/// eventually receive its `Done` terminator (stashed in the pending-close
/// outbox, flushed once the client drains).
///
/// Setup:
///   - prompt len 3, max_tokens 2 → generates tokens 3, 4 (deterministic Mock)
///     then finishes with FinishReason::Length.
///   - channel capacity 2: exactly fits the two payload tokens, leaving NO
///     room for Done when the seq finishes.
///   - we do NOT drain the receiver until generation is complete, so at finish
///     time the channel is full and Done goes to pending_close.
///   - then we drain fully: the actor flushes Done on its next iteration once
///     room is available. `recv().await` naturally waits for that flush.
///
/// The test is deterministic in the success path (no sleeps). A bounded
/// `recv().await` on the async side waits for the actor's next iteration
/// (which runs in ≤1ms) to deliver the stashed terminator.
#[tokio::test]
async fn done_terminator_is_flushed_after_backpressure() {
    let (handle, _metrics, _join) =
        spawn(Box::new(Mock { vocab: 16 }), cfg(), empty_piece_map(), None);

    // Capacity 2: exactly fits the two generated tokens, no room for Done.
    let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamItem>(2);
    handle
        .submit(Job {
            id: 0,
            prompt_ids: vec![1u32; 3], // prompt len 3
            params: greedy(2),         // max_tokens 2 → tokens [3, 4], then Length
            token_tx: tx,
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

    // Wait until both token slots are filled. We poll with a short bounded
    // wait to avoid a sleep in the fast path; the actor delivers tokens within
    // one or two step iterations.
    //
    // First recv: waits for token 3 (prefill result, 3%16).
    let item0 = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
        .await
        .expect("timed out waiting for token 0")
        .expect("channel closed unexpectedly");
    assert!(
        matches!(item0, StreamItem::Token(3, _)),
        "first token should be 3, got {item0:?}"
    );

    // Second recv: waits for token 4 (decode step, 4%16). Channel now empty
    // again after consuming, so Done can be delivered immediately if the actor
    // races past the final step — but typically the channel briefly hit 2/2
    // before we drained, so Done ended up in pending_close. Either way we must
    // receive Done as the last item.
    let item1 = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
        .await
        .expect("timed out waiting for token 1")
        .expect("channel closed unexpectedly");

    // The channel was full (2/2) at finish. The next item is either the last
    // payload token or already the Done if the actor raced past us. Handle
    // both orderings robustly.
    let final_item = match item1 {
        StreamItem::Token(t, _lp) => {
            // Got token 4 — Done must come next (from pending_close flush).
            assert_eq!(t, 4, "second token should be 4");
            tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
                .await
                .expect("timed out waiting for Done after draining tokens")
                .expect("channel closed without Done")
        }
        done @ StreamItem::Done(_) => done, // actor delivered Done directly
    };

    assert_eq!(
        final_item,
        StreamItem::Done(FinishReason::Length),
        "Done(Length) terminator must not be lost under backpressure"
    );
}

// ---- PREFIX ANCHORS through the actor (2026-09-26) ----

/// What the snapshotting mock was asked to do, in order.
#[derive(Debug, Clone, PartialEq)]
enum SnapCall {
    Save(u64, u64),
    SaveAnchor(u64, u64),
    SaveJunction(u64, u64),
    Restore(u64, u64),
}

/// `Mock` plus recurrent-state snapshots: it records every save / anchor save / junction save /
/// restore and restores only keys it saved, exactly as the Metal store answers.
struct SnapMock {
    calls: Arc<std::sync::Mutex<Vec<SnapCall>>>,
}
impl BatchedBackend for SnapMock {
    fn forward_batch(&self, ids: &[u32], batch: &ForwardBatch) -> Vec<Vec<f32>> {
        Mock { vocab: 16 }.forward_batch(ids, batch)
    }
    fn kv_geometry(&self) -> (usize, usize) {
        (512, 16)
    }
    fn state_save(&self, stream: u64, key: u64) -> (bool, Option<u64>) {
        self.calls.lock().unwrap().push(SnapCall::Save(stream, key));
        (true, None)
    }
    fn state_save_anchor(&self, stream: u64, key: u64) -> (bool, Option<u64>) {
        self.calls
            .lock()
            .unwrap()
            .push(SnapCall::SaveAnchor(stream, key));
        (true, None)
    }
    fn state_save_junction(&self, stream: u64, key: u64) -> (bool, Option<u64>) {
        self.calls
            .lock()
            .unwrap()
            .push(SnapCall::SaveJunction(stream, key));
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

/// THE WHOLE CHAIN, no GPU: a job's `prefix_anchor` reaches the scheduler, the anchor chunk's
/// snapshot goes to the backend's ANCHOR pool (`state_save_anchor`), and a second job — same
/// 1,100-token system prefix, different first message — is restored from that anchor. The same
/// two jobs WITHOUT an anchor: no anchor save, and the second job restores nothing.
///
/// **Updated 2026-09-26 (junction snapshots):** the no-anchor arm asserted that EVERY snapshot
/// call was an ordinary save. Job 2 now also takes a JUNCTION there: its cached-KV match runs
/// through job 1's 1,100 shared tokens (1,088, a whole block short of the divergence) while it
/// resumes at 0, so it snapshots 1,024 for the next session — the anchor-less, learned version
/// of the same shared prefix. The assertion is now what the arm was for: no anchor save, and
/// nothing for job 2 to restore.
#[tokio::test]
async fn a_jobs_prefix_anchor_is_saved_to_the_anchor_pool_and_resumed_by_a_new_session() {
    async fn run(anchor: Option<usize>) -> Vec<SnapCall> {
        let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
        // the boundary positions below are the pre-2026-09-27 one-window back-off (the control arm);
        // the 32-token tail default is tested in arf-core `tests/scheduler_snapshot_tail.rs`
        std::env::set_var("ARF_SNAPSHOT_TAIL", "0");
        std::env::set_var("ARF_SNAPSHOT_ALIGN", "128");
        let cfg = EngineConfig {
            block_size: 16,
            num_blocks: 512,
            max_batch_size: 8,
            max_prefill_tokens: 4096,
            enable_prefix_cache: true,
            state_snapshots: true,
            ..Default::default()
        };
        let backend = SnapMock {
            calls: Arc::clone(&calls),
        };
        let (handle, _metrics, _join) = spawn(Box::new(backend), cfg, empty_piece_map(), None);
        for (id, user) in [(1u64, 10_000u32), (2, 20_000)] {
            let mut prompt: Vec<u32> = (0..1100).collect();
            prompt.extend(user..user + 600);
            let (tx, rx) = tokio::sync::mpsc::channel(256);
            handle
                .submit(Job {
                    id,
                    prompt_ids: prompt,
                    params: greedy(2),
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
                })
                .unwrap();
            let (_, fin) = collect(rx).await;
            assert_eq!(fin, Some(FinishReason::Length));
        }
        let c = calls.lock().unwrap().clone();
        c
    }

    let with = run(Some(1100)).await;
    let anchor_key = match with.first() {
        Some(SnapCall::SaveAnchor(1, k)) => *k,
        other => panic!("job 1's first snapshot must be the anchor, got {other:?} in {with:?}"),
    };
    assert!(
        with.contains(&SnapCall::Restore(2, anchor_key)),
        "job 2 (a new session) resumes from the anchor: {with:?}"
    );
    assert_eq!(
        with.iter()
            .filter(|c| matches!(c, SnapCall::SaveAnchor(..)))
            .count(),
        1,
        "the anchor is saved once: {with:?}"
    );

    let without = run(None).await;
    assert!(
        without
            .iter()
            .all(|c| !matches!(c, SnapCall::SaveAnchor(..) | SnapCall::Restore(..))),
        "no anchor, no anchor save, and nothing for job 2 to restore: {without:?}"
    );
    assert!(
        without
            .iter()
            .any(|c| matches!(c, SnapCall::SaveJunction(2, _))),
        "job 2 leaves job 1's cached history and takes a junction there: {without:?}"
    );
}

/// TOOLS ANCHORS through the actor (2026-09-27), no GPU: the whole chain for a Claude Code
/// session in a second working directory. Prompts are a 1,100-token tools block (tools anchor
/// 1,100 -> 1,024), a 1,300-token system text that differs per directory (system anchor 2,400
/// -> 2,304) and a 600-token first message. Job 1 saves BOTH anchors to the backend's anchor
/// pool, the tools anchor first. Job 2 (another directory) resumes from the TOOLS anchor and
/// saves its own system anchor. Without the tools anchor job 2 restores nothing — the 2026-09-27
/// measurement (`anchor snapshot at 13056`, no resume for the next two directories).
#[tokio::test]
async fn a_jobs_tools_anchor_is_saved_and_resumed_by_a_session_with_another_system_text() {
    async fn run(tools_anchor: Option<usize>) -> Vec<SnapCall> {
        let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
        // the boundary positions below are the pre-2026-09-27 one-window back-off (the control arm);
        // the 32-token tail default is tested in arf-core `tests/scheduler_snapshot_tail.rs`
        std::env::set_var("ARF_SNAPSHOT_TAIL", "0");
        std::env::set_var("ARF_SNAPSHOT_ALIGN", "128");
        let cfg = EngineConfig {
            block_size: 16,
            num_blocks: 512,
            max_batch_size: 8,
            max_prefill_tokens: 4096,
            enable_prefix_cache: true,
            state_snapshots: true,
            ..Default::default()
        };
        let backend = SnapMock {
            calls: Arc::clone(&calls),
        };
        let (handle, _metrics, _join) = spawn(Box::new(backend), cfg, empty_piece_map(), None);
        for (id, system, user) in [(1u64, 100_000u32, 10_000u32), (2, 200_000, 20_000)] {
            let mut prompt: Vec<u32> = (0..1100).collect();
            prompt.extend(system..system + 1300);
            prompt.extend(user..user + 600);
            let (tx, rx) = tokio::sync::mpsc::channel(256);
            handle
                .submit(Job {
                    id,
                    prompt_ids: prompt,
                    params: greedy(2),
                    token_tx: tx,
                    want_logprobs: false,
                    top_logprobs: 0,
                    response_format: None,
                    images: Vec::new(),
                    image_positions: Vec::new(),
                    image_prompt: None,
                    prefix_anchor: Some(2400),
                    tools_anchor,
                    header_tail: None,
                    background: false,
                    stop: None,
                })
                .unwrap();
            let (_, fin) = collect(rx).await;
            assert_eq!(fin, Some(FinishReason::Length));
        }
        let c = calls.lock().unwrap().clone();
        c
    }

    let with = run(Some(1100)).await;
    let (tools_key, system_key) = match with.as_slice() {
        [SnapCall::SaveAnchor(1, t), SnapCall::SaveAnchor(1, s), ..] => (*t, *s),
        other => panic!("job 1 must save the tools anchor, then the system anchor: {other:?}"),
    };
    assert!(
        with.contains(&SnapCall::Restore(2, tools_key)),
        "job 2 (another system text) resumes from the tools anchor: {with:?}"
    );
    assert!(!with.contains(&SnapCall::Restore(2, system_key)));
    assert_eq!(
        with.iter()
            .filter(|c| matches!(c, SnapCall::SaveAnchor(2, _)))
            .count(),
        1,
        "job 2 saves its own system anchor only: {with:?}"
    );

    let without = run(None).await;
    assert!(
        without
            .iter()
            .all(|c| !matches!(c, SnapCall::Restore(2, _))),
        "without the tools anchor job 2 has nothing to resume from: {without:?}"
    );
}

// ---- JUNCTION SNAPSHOTS through the actor (2026-09-26) ----

/// THE WHOLE CHAIN for junctions, no GPU: three sessions of one project, S (1,100-token system
/// prefix, anchored) + R (2,000 tokens every session shares after it) + their own 600-token task.
/// Job 1 saves the anchor. Job 2 resumes from the anchor and saves a JUNCTION where it leaves
/// job 1's cached history (to the backend's junction pool, `state_save_junction`). Job 3
/// resumes from that junction — not from the anchor.
#[tokio::test]
async fn a_junction_is_saved_to_the_junction_pool_and_resumed_by_the_next_session() {
    let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
    // the boundary positions below are the pre-2026-09-27 one-window back-off (the control arm);
    // the 32-token tail default is tested in arf-core `tests/scheduler_snapshot_tail.rs`
    std::env::set_var("ARF_SNAPSHOT_TAIL", "0");
    std::env::set_var("ARF_SNAPSHOT_ALIGN", "128");
    let cfg = EngineConfig {
        block_size: 16,
        num_blocks: 512, // SnapMock's KV pool
        max_batch_size: 8,
        max_prefill_tokens: 4096,
        enable_prefix_cache: true,
        state_snapshots: true,
        ..Default::default()
    };
    let backend = SnapMock {
        calls: Arc::clone(&calls),
    };
    let (handle, _metrics, _join) = spawn(Box::new(backend), cfg, empty_piece_map(), None);
    for (id, task) in [(1u64, 10_000u32), (2, 20_000), (3, 30_000)] {
        let mut prompt: Vec<u32> = (0..1100).collect();
        prompt.extend(100_000..102_000);
        prompt.extend(task..task + 600);
        let (tx, rx) = tokio::sync::mpsc::channel(256);
        handle
            .submit(Job {
                id,
                prompt_ids: prompt,
                params: greedy(2),
                token_tx: tx,
                want_logprobs: false,
                top_logprobs: 0,
                response_format: None,
                images: Vec::new(),
                image_positions: Vec::new(),
                image_prompt: None,
                prefix_anchor: Some(1100),
                tools_anchor: None,
                header_tail: None,
                background: false,
                stop: None,
            })
            .unwrap();
        let (_, fin) = collect(rx).await;
        assert_eq!(fin, Some(FinishReason::Length));
    }
    let c = calls.lock().unwrap().clone();
    let anchor = c
        .iter()
        .find_map(|x| match x {
            SnapCall::SaveAnchor(1, k) => Some(*k),
            _ => None,
        })
        .expect("job 1 saves the anchor");
    let junction = c
        .iter()
        .find_map(|x| match x {
            SnapCall::SaveJunction(2, k) => Some(*k),
            _ => None,
        })
        .unwrap_or_else(|| panic!("job 2 saves a junction: {c:?}"));
    assert!(c.contains(&SnapCall::Restore(2, anchor)), "{c:?}");
    assert!(
        c.contains(&SnapCall::Restore(3, junction)),
        "job 3 resumes from the junction: {c:?}"
    );
    assert!(!c.contains(&SnapCall::Restore(3, anchor)), "{c:?}");
    // **Updated 2026-09-26 (session-start snapshots):** this counted every junction-pool save
    // as "one junction for the three sessions". Each session here is the FIRST turn of its
    // conversation, so its end-of-prompt snapshot can go to the junction pool as well
    // (`SeqPlan::snapshot_session_start`): job 1 saves one (its session start, from 0), job 2
    // two (the junction, then its session start, from the anchor). Job 3 none: its end (3,456)
    // is only 384 tokens past the junction it resumed from, under the 512-token floor, so it is
    // an ordinary turn-end save. Still exactly one JUNCTION.
    let per_job = |id: u64| {
        c.iter()
            .filter(|x| matches!(x, SnapCall::SaveJunction(j, _) if *j == id))
            .count()
    };
    assert_eq!(
        (per_job(1), per_job(2), per_job(3)),
        (1, 2, 0),
        "one junction (job 2) plus the session starts above the floors: {c:?}"
    );
}

// ---- SESSION-START SNAPSHOTS through the actor (2026-09-26) ----

/// `SnapMock` with the Metal store's EVICTION (`state_snapshots.rs`, `file_snapshot`): a 4-slot
/// turn-end LRU, a 2-slot junction LRU and a 2-slot anchor LRU, a restore bumps recency, and a
/// save returns the key it evicted — which the serving loop hands to `forget_snapshot`. A
/// restore of an evicted key answers `false`, which the loop treats as fatal.
struct PoolMock {
    calls: Arc<std::sync::Mutex<Vec<SnapCall>>>,
    /// (key, pool: 0 turn / 1 junction / 2 anchor, tick)
    store: std::sync::Mutex<(Vec<(u64, u8, u64)>, u64)>,
}
impl PoolMock {
    fn file(&self, key: u64, pool: u8) -> Option<u64> {
        let mut st = self.store.lock().unwrap();
        let (entries, tick) = &mut *st;
        let mut pool = pool;
        if let Some(i) = entries.iter().position(|e| e.0 == key) {
            pool = pool.max(entries[i].1);
            entries.swap_remove(i);
        }
        *tick += 1;
        let cap = [4, 2, 2][pool as usize];
        let mut evicted = None;
        if entries.iter().filter(|e| e.1 == pool).count() >= cap {
            let (i, _) = entries
                .iter()
                .enumerate()
                .filter(|(_, e)| e.1 == pool)
                .min_by_key(|(_, e)| e.2)
                .unwrap();
            evicted = Some(entries.swap_remove(i).0);
        }
        entries.push((key, pool, *tick));
        evicted
    }
}
impl BatchedBackend for PoolMock {
    fn forward_batch(&self, ids: &[u32], batch: &ForwardBatch) -> Vec<Vec<f32>> {
        Mock { vocab: 16 }.forward_batch(ids, batch)
    }
    fn kv_geometry(&self) -> (usize, usize) {
        (1024, 16)
    }
    fn state_save(&self, stream: u64, key: u64) -> (bool, Option<u64>) {
        self.calls.lock().unwrap().push(SnapCall::Save(stream, key));
        (true, self.file(key, 0))
    }
    fn state_save_junction(&self, stream: u64, key: u64) -> (bool, Option<u64>) {
        self.calls
            .lock()
            .unwrap()
            .push(SnapCall::SaveJunction(stream, key));
        (true, self.file(key, 1))
    }
    fn state_save_anchor(&self, stream: u64, key: u64) -> (bool, Option<u64>) {
        self.calls
            .lock()
            .unwrap()
            .push(SnapCall::SaveAnchor(stream, key));
        (true, self.file(key, 2))
    }
    fn state_restore(&self, stream: u64, key: u64) -> bool {
        self.calls
            .lock()
            .unwrap()
            .push(SnapCall::Restore(stream, key));
        let mut st = self.store.lock().unwrap();
        let (entries, tick) = &mut *st;
        *tick += 1;
        match entries.iter_mut().find(|e| e.0 == key) {
            Some(e) => {
                e.2 = *tick;
                true
            }
            None => false,
        }
    }
}

/// THE WHOLE CHAIN for session starts, no GPU: a 7-turn agent conversation (1,100-token anchored
/// system prefix, a 2,000-token first message, 600 new tokens a turn), then a NEW session whose
/// first request is identical to turn 1's. Turn 1's end-of-prompt snapshot goes to the junction
/// pool (`state_save_junction`); turns 2-7 save theirs with `state_save`, and the 4-slot turn-end
/// LRU evicts two of them; the new session is restored from turn 1's snapshot — not from the
/// anchor, which is where it resumed before this change (measured 2026-09-26: the
/// repeated Claude Code task, "resumes from the anchor snapshot at 13184 tokens").
///
/// Mutation-checked: routing a session start through `state_save` (the pre-change call) fails
/// this test at "turn 1's end goes to the junction pool". What job 8 then resumes at (the
/// anchor, once turn 5's save has evicted turn 1's end) is asserted at the scheduler level by
/// `tests/scheduler_no_session_start.rs` (arf-core).
#[tokio::test]
async fn a_first_turns_end_snapshot_survives_the_session_and_restores_an_identical_new_one() {
    let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
    // the boundary positions below are the pre-2026-09-27 one-window back-off (the control arm);
    // the 32-token tail default is tested in arf-core `tests/scheduler_snapshot_tail.rs`
    std::env::set_var("ARF_SNAPSHOT_TAIL", "0");
    std::env::set_var("ARF_SNAPSHOT_ALIGN", "128");
    let cfg = EngineConfig {
        block_size: 16,
        num_blocks: 1024, // PoolMock's KV pool
        max_batch_size: 8,
        max_prefill_tokens: 4096,
        enable_prefix_cache: true,
        state_snapshots: true,
        ..Default::default()
    };
    let backend = PoolMock {
        calls: Arc::clone(&calls),
        store: Default::default(),
    };
    let (handle, _metrics, _join) = spawn(Box::new(backend), cfg, empty_piece_map(), None);
    let mut turn1: Vec<u32> = (0..1100).collect();
    turn1.extend(10_000..12_000);
    let mut prompts = vec![turn1.clone()];
    for k in 2..=7u32 {
        let mut p = prompts.last().unwrap().clone();
        p.extend(20_000 + 1000 * k..20_000 + 1000 * k + 600);
        prompts.push(p);
    }
    prompts.push(turn1); // job 8: a new session, identical first request
    for (i, prompt) in prompts.into_iter().enumerate() {
        let (tx, rx) = tokio::sync::mpsc::channel(256);
        handle
            .submit(Job {
                id: i as u64 + 1,
                prompt_ids: prompt,
                params: greedy(2),
                token_tx: tx,
                want_logprobs: false,
                top_logprobs: 0,
                response_format: None,
                images: Vec::new(),
                image_positions: Vec::new(),
                image_prompt: None,
                prefix_anchor: Some(1100),
                tools_anchor: None,
                header_tail: None,
                background: false,
                stop: None,
            })
            .unwrap();
        let (_, fin) = collect(rx).await;
        assert_eq!(fin, Some(FinishReason::Length), "job {}", i + 1);
    }
    let c = calls.lock().unwrap().clone();
    let anchor = c
        .iter()
        .find_map(|x| match x {
            SnapCall::SaveAnchor(1, k) => Some(*k),
            _ => None,
        })
        .expect("turn 1 saves the anchor");
    let start = c
        .iter()
        .find_map(|x| match x {
            SnapCall::SaveJunction(1, k) => Some(*k),
            _ => None,
        })
        .unwrap_or_else(|| panic!("turn 1's end goes to the junction pool: {c:?}"));
    for turn in 2..=7u64 {
        let own: Vec<_> = c
            .iter()
            .filter(|x| {
                matches!(x, SnapCall::Save(j, _) | SnapCall::SaveJunction(j, _)
                    | SnapCall::SaveAnchor(j, _) if *j == turn)
            })
            .collect();
        assert!(
            matches!(own.as_slice(), [SnapCall::Save(..)]),
            "turn {turn} saves its end to the turn-end pool, once: {own:?}"
        );
    }
    assert!(
        c.contains(&SnapCall::Restore(8, start)),
        "the new identical session resumes at turn 1's end: {c:?}"
    );
    assert!(!c.contains(&SnapCall::Restore(8, anchor)), "{c:?}");
}

/// The System One score-only job (`http::systemone`): greedy, one token, `ALL_LOGPROBS`. It must
/// come back as exactly one token carrying the prefill's WHOLE logprob row in vocabulary order,
/// then `Done(Length)` — no decode step, the stream released like any finished one.
#[tokio::test]
async fn score_only_job_returns_the_whole_logprob_row_once() {
    let (handle, _metrics, _join) =
        spawn(Box::new(Mock { vocab: 16 }), cfg(), empty_piece_map(), None);
    let (tx, mut rx) = tokio::sync::mpsc::channel(256);
    handle
        .submit(Job {
            id: 7,
            prompt_ids: vec![1; 5],
            params: greedy(1),
            token_tx: tx,
            want_logprobs: true,
            top_logprobs: arf_serve::actor::ALL_LOGPROBS,
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
    let Some(StreamItem::Token(tok, Some(lp))) = rx.recv().await else {
        panic!("expected one token with logprobs");
    };
    assert_eq!(tok, 5); // the mock's argmax after a 5-token prompt
    assert_eq!(lp.top_logprobs.len(), 16);
    for (i, &(id, l)) in lp.top_logprobs.iter().enumerate() {
        assert_eq!(id as usize, i);
        assert!(l <= 0.0);
    }
    let total: f64 = lp.top_logprobs.iter().map(|&(_, l)| (l as f64).exp()).sum();
    assert!((total - 1.0).abs() < 1e-5, "{total}");
    assert_eq!(lp.top_logprobs[5].1, lp.logprob);
    assert_eq!(
        rx.recv().await,
        Some(StreamItem::Done(FinishReason::Length))
    );
}
