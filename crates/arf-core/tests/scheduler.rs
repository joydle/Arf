//! Scheduler behavior: admission under capacity, block accounting, finishing,
//! and preemption with recompute. These tests are model-free.

use arf_core::config::EngineConfig;
use arf_core::sampling::SamplingParams;
use arf_core::scheduler::{FinishReason, Request, Scheduler};
use arf_core::Tensor;

fn cfg(block_size: usize, num_blocks: usize) -> EngineConfig {
    EngineConfig {
        block_size,
        num_blocks,
        max_batch_size: 8,
        max_prefill_tokens: 4096,
        ..Default::default()
    }
}

fn req(id: u64, prompt_len: usize, max_tokens: usize) -> Request {
    Request::new(id, vec![1; prompt_len], SamplingParams::greedy(max_tokens))
}

fn cfg_prefix(block_size: usize, num_blocks: usize) -> EngineConfig {
    EngineConfig {
        block_size,
        num_blocks,
        max_batch_size: 8,
        max_prefill_tokens: 4096,
        enable_prefix_cache: true,
        ..Default::default()
    }
}

fn req_tokens(id: u64, tokens: Vec<u32>, max_tokens: usize) -> Request {
    Request::new(id, tokens, SamplingParams::greedy(max_tokens))
}

/// Constant logits over `vocab` tokens (argmax is token 0).
fn logits(rows: usize, vocab: usize) -> Tensor {
    let mut data = vec![0f32; rows * vocab];
    for r in 0..rows {
        data[r * vocab] = 1.0; // token 0 wins
    }
    Tensor::from_vec(data, vec![rows, vocab])
}

#[test]
fn admits_until_blocks_exhausted() {
    let mut s = Scheduler::new(cfg(4, 4)); // 4 blocks
    s.add(req(0, 5, 8)); // needs 2 blocks
    s.add(req(1, 3, 8)); // needs 1 block
    s.add(req(2, 9, 8)); // needs 3 blocks -> won't fit (only 1 left)

    let plan = s.schedule().unwrap().expect("a batch");
    assert_eq!(s.num_running(), 2);
    assert_eq!(s.num_waiting(), 1);
    assert_eq!(s.num_free_blocks(), 1); // 4 - (2 + 1)
    assert_eq!(plan.seqs.len(), 2);
    assert_eq!(plan.input_ids.len(), 5 + 3);
    // Positions for each prefilled sequence run 0..len.
    assert_eq!(&plan.positions[..5], &[0, 1, 2, 3, 4]);
    assert_eq!(&plan.positions[5..], &[0, 1, 2]);
}

/// STOP STRINGS: a `StopCheck` that fires inside a speculative run retires the
/// sequence at THAT token — the run is truncated there, as at a stop token, and the KV is freed —
/// with `StopString`, and the token that completed the stop string is the last output.
#[test]
fn stop_check_retires_mid_run_like_a_stop_token() {
    use arf_core::scheduler::StopCheck;
    let mut s = Scheduler::new(cfg(4, 4));
    // "Text" = the token ids; the stop "string" is completed by a 7 that follows a 6.
    let check = StopCheck::new(|out: &[u32]| out.ends_with(&[6, 7]));
    s.add(req(0, 2, 32).with_stop_check(Some(check)));
    let _ = s.schedule().unwrap().expect("batch");
    let out = s.commit_runs(&[vec![5, 6, 7, 8, 9]]).unwrap();
    let toks: Vec<u32> = out.iter().map(|o| o.token).collect();
    assert_eq!(
        toks,
        vec![5, 6, 7],
        "the run is cut at the completing token"
    );
    assert_eq!(out[2].finish_reason, Some(FinishReason::StopString));
    assert!(out[..2].iter().all(|o| o.finish_reason.is_none()));
    assert_eq!(s.num_running(), 0);
    assert_eq!(
        s.num_free_blocks(),
        4,
        "retired and freed, not left running"
    );
}

/// A stop token is still a stop token with a stop check set, and a stop string beats the budget
/// when one token does both.
#[test]
fn stop_check_order_against_stop_tokens_and_the_budget() {
    use arf_core::scheduler::StopCheck;
    let fire = || StopCheck::new(|out: &[u32]| out.last() == Some(&3));
    let mut params = SamplingParams::greedy(32);
    params.stop_tokens = vec![3];
    let mut s = Scheduler::new(cfg(4, 4));
    s.add(Request::new(0, vec![1; 2], params).with_stop_check(Some(fire())));
    let _ = s.schedule().unwrap().expect("batch");
    let out = s.commit_runs(&[vec![3]]).unwrap();
    assert_eq!(out[0].finish_reason, Some(FinishReason::Stop));

    let mut s = Scheduler::new(cfg(4, 4));
    s.add(req(0, 2, 2).with_stop_check(Some(fire())));
    let _ = s.schedule().unwrap().expect("batch");
    let out = s.commit_runs(&[vec![2, 3]]).unwrap();
    assert_eq!(out[1].finish_reason, Some(FinishReason::StopString));
}

#[test]
fn finishing_frees_blocks() {
    let mut s = Scheduler::new(cfg(4, 4));
    s.add(req(0, 2, 1)); // generate exactly one token then stop on length
    let _ = s.schedule().unwrap().expect("batch");
    assert_eq!(s.num_free_blocks(), 3); // one block held during the step

    let out = s.commit(&logits(1, 10)).unwrap();
    assert_eq!(out.len(), 1);
    assert!(out[0].finished);
    assert_eq!(out[0].finish_reason, Some(FinishReason::Length));
    assert_eq!(s.num_running(), 0);
    assert_eq!(s.num_free_blocks(), 4); // fully reclaimed
}

#[test]
fn preempts_newest_when_pool_is_full() {
    let mut s = Scheduler::new(cfg(4, 2)); // only 2 blocks total
    s.add(req(0, 4, 64)); // 1 block
    s.add(req(1, 4, 64)); // 1 block

    // Step 1: both prefill, pool now full.
    let _ = s.schedule().unwrap().expect("batch");
    assert_eq!(s.num_running(), 2);
    assert_eq!(s.num_free_blocks(), 0);
    let _ = s.commit(&logits(2, 10)).unwrap(); // each grows to 5 tokens

    // Step 2: each now needs a 2nd block but none are free; the newest is
    // preempted so the oldest can continue.
    let plan = s.schedule().unwrap().expect("batch");
    assert_eq!(s.num_running(), 1, "one sequence preempted");
    assert_eq!(s.num_waiting(), 1);
    assert_eq!(plan.seqs[0].id, 0, "oldest survives");
    assert_eq!(s.num_free_blocks(), 0);
}

#[test]
fn empty_scheduler_plans_nothing() {
    let mut s = Scheduler::new(cfg(4, 4));
    assert!(s.schedule().unwrap().is_none());
    assert!(!s.has_unfinished());
}

#[test]
fn scheduler_reports_total_blocks() {
    let sched = Scheduler::new(cfg(4, 7));
    assert_eq!(sched.num_total_blocks(), 7);
    // Nothing allocated yet: free == total.
    assert_eq!(sched.num_free_blocks(), 7);
}

#[test]
fn commit_advances_num_computed_by_scheduled_qlen_only() {
    // 2-token prompt, 2 steps of decode. After each commit the pending count
    // must be exactly 1 (the newly sampled token), proving num_computed tracks
    // the scheduled q_len rather than jumping to tokens.len().
    let mut s = Scheduler::new(cfg(4, 8));
    s.add(req(0, 2, 4));
    let p1 = s.schedule().unwrap().expect("prefill step");
    assert_eq!(p1.seqs[0].q_len, 2);
    let out1 = s.commit(&logits(1, 10)).unwrap();
    assert_eq!(out1.len(), 1, "prefill completes -> first token emitted");

    let p2 = s.schedule().unwrap().expect("decode step");
    assert_eq!(p2.seqs[0].q_len, 1, "exactly the one pending token");
    assert_eq!(p2.seqs[0].past_len, 2);
    let out2 = s.commit(&logits(1, 10)).unwrap();
    assert_eq!(out2.len(), 1);
}

#[test]
fn commit_tokens_matches_commit_logits_for_greedy() {
    // Same requests through both commit paths produce identical outputs/state.
    let mut a = Scheduler::new(cfg(4, 8));
    let mut b = Scheduler::new(cfg(4, 8));
    a.add(req(0, 3, 2));
    b.add(req(0, 3, 2));

    for _ in 0..2 {
        let pa = a.schedule().unwrap().expect("plan");
        let pb = b.schedule().unwrap().expect("plan");
        assert_eq!(pa.input_ids, pb.input_ids);
        let oa = a.commit(&logits(pa.seqs.len(), 10)).unwrap();
        // Greedy over `logits()` always argmaxes to token 0.
        let ob = b.commit_tokens(&vec![0u32; pb.seqs.len()]).unwrap();
        assert_eq!(oa.len(), ob.len());
        for (x, y) in oa.iter().zip(&ob) {
            assert_eq!(
                (x.id, x.token, x.finished, x.finish_reason),
                (y.id, y.token, y.finished, y.finish_reason)
            );
        }
    }
    assert_eq!(a.num_running(), b.num_running());
    assert_eq!(a.num_free_blocks(), b.num_free_blocks());
}

#[test]
fn evict_seqs_frees_blocks_and_drops_running_and_waiting() {
    let mut s = Scheduler::new(cfg(4, 8));
    s.add(req(0, 3, 8));
    s.add(req(1, 3, 8));
    let p = s.schedule().unwrap().expect("plan");
    let _ = s.commit_tokens(&vec![0u32; p.seqs.len()]).unwrap();
    assert_eq!(s.num_running(), 2);
    s.add(req(2, 3, 8)); // still waiting

    s.evict_seqs(&[0, 2]);
    assert_eq!(s.num_running(), 1);
    assert_eq!(s.num_waiting(), 0);
    // Only seq 1's single block remains held.
    assert_eq!(s.num_free_blocks(), s.num_total_blocks() - 1);
}

#[test]
fn two_seqs_finishing_in_one_commit_both_retire_and_free_blocks() {
    let mut s = Scheduler::new(cfg(4, 8));
    s.add(req(0, 3, 1)); // length-limited after one token
    s.add(req(1, 5, 1));
    let plan = s.schedule().unwrap().expect("both prefill");
    assert_eq!(plan.seqs.len(), 2);

    let out = s.commit(&logits(2, 10)).unwrap();
    assert_eq!(out.len(), 2);
    assert!(out.iter().all(|o| o.finished));
    assert_eq!(s.num_running(), 0, "both retired in the same commit");
    assert_eq!(
        s.num_free_blocks(),
        s.num_total_blocks(),
        "all blocks reclaimed"
    );
}

#[test]
fn evicting_a_preempted_seq_is_clean() {
    // Force a preemption (2 blocks, 2 seqs that outgrow them), then evict the
    // preempted seq (sitting in waiting with an EMPTY block table after
    // reset_for_recompute) — must not double-free or change the free count.
    let mut s = Scheduler::new(cfg(4, 2));
    s.add(req(0, 4, 64)); // 1 block
    s.add(req(1, 4, 64)); // 1 block
    let p = s.schedule().unwrap().expect("both prefill");
    let _ = s.commit(&logits(p.seqs.len(), 10)).unwrap(); // each grows to 5 tokens

    // Next step: each needs a 2nd block, none free -> newest (id 1) preempted.
    let p2 = s.schedule().unwrap().expect("survivor runs");
    assert_eq!(s.num_running(), 1);
    assert_eq!(s.num_waiting(), 1);
    let _ = s.commit(&logits(p2.seqs.len(), 10)).unwrap();

    let free_before = s.num_free_blocks();
    s.evict_seqs(&[1]); // the preempted seq: empty block_table, in waiting
    assert_eq!(s.num_waiting(), 0, "preempted seq evicted from waiting");
    assert_eq!(s.num_running(), 1, "survivor untouched");
    assert_eq!(
        s.num_free_blocks(),
        free_before,
        "evicting a block-less seq must not change the free count"
    );
}

// ── Chunked-prefill tests ──────────────────────────────────────────

fn cfg_budget(block_size: usize, num_blocks: usize, budget: usize) -> EngineConfig {
    EngineConfig {
        block_size,
        num_blocks,
        max_batch_size: 8,
        max_prefill_tokens: budget,
        ..Default::default()
    }
}

// ── Prefix caching tests  ─────────────────────────────────────────

#[test]
fn prefix_cache_reuses_shared_prefix_after_first_request() {
    let mut s = Scheduler::new(cfg_prefix(4, 16));
    let prompt = vec![1u32, 2, 3, 4, 5, 6, 7, 8, 9];

    // First request: nothing to reuse, prefill the whole prompt.
    s.add(req_tokens(0, prompt.clone(), 1));
    let p = s.schedule().unwrap().expect("prefill");
    assert_eq!((p.seqs[0].past_len, p.seqs[0].q_len), (0, 9));
    let out = s.commit(&logits(p.seqs.len(), 10)).unwrap();
    assert!(out[0].finished, "max_tokens=1 -> finishes");
    assert_eq!(s.num_running(), 0);
    assert_eq!(
        s.prefix_cache_reused_tokens(),
        0,
        "no reuse on a cold cache"
    );

    // Identical prompt: the two full 4-token blocks (positions 0..8) are reused;
    // only the uncached tail (position 8) is recomputed.
    s.add(req_tokens(1, prompt.clone(), 4));
    let p2 = s.schedule().unwrap().expect("prefill with reuse");
    assert_eq!(p2.seqs[0].id, 1);
    assert_eq!(p2.seqs[0].past_len, 8, "two cached blocks reused");
    assert_eq!(p2.seqs[0].q_len, 1, "only the uncached tail is prefilled");
    assert_eq!(
        p2.input_ids,
        vec![9u32],
        "recompute only the last prompt token"
    );
    assert_eq!(p2.positions, vec![8u32]);
    assert_eq!(s.prefix_cache_reused_tokens(), 8);
}

#[test]
fn prefix_cache_disabled_recomputes_everything() {
    // Same workload without the flag: no reuse, full prefill both times.
    let mut s = Scheduler::new(cfg(4, 16)); // prefix caching off (default)
    let prompt = vec![1u32, 2, 3, 4, 5, 6, 7, 8, 9];
    s.add(req_tokens(0, prompt.clone(), 1));
    let p = s.schedule().unwrap().unwrap();
    let _ = s.commit(&logits(p.seqs.len(), 10)).unwrap();

    s.add(req_tokens(1, prompt.clone(), 4));
    let p2 = s.schedule().unwrap().unwrap();
    assert_eq!(
        (p2.seqs[0].past_len, p2.seqs[0].q_len),
        (0, 9),
        "no reuse when off"
    );
    assert_eq!(s.prefix_cache_reused_tokens(), 0);
}

#[test]
fn prefix_cache_keeps_at_least_one_pending_token() {
    // A prompt that is an exact multiple of block_size must not be fully cached
    // away — the step needs a query token. Prime with an identical prompt first.
    let mut s = Scheduler::new(cfg_prefix(4, 16));
    let prompt = vec![1u32, 2, 3, 4, 5, 6, 7, 8]; // exactly 2 blocks
    s.add(req_tokens(0, prompt.clone(), 1));
    let p = s.schedule().unwrap().unwrap();
    let _ = s.commit(&logits(p.seqs.len(), 10)).unwrap();

    s.add(req_tokens(1, prompt.clone(), 4));
    let p2 = s.schedule().unwrap().unwrap();
    assert!(p2.seqs[0].q_len >= 1, "must leave a token to forward");
    assert_eq!(
        p2.seqs[0].past_len, 4,
        "first block reused, last block recomputed"
    );
    assert_eq!((p2.seqs[0].q_len, p2.seqs[0].past_len), (4, 4));
}

#[test]
fn prefix_cache_conserves_blocks_across_churn() {
    // Repeatedly run the same prompt to completion. Every block must be
    // reclaimable (free or evictable-cached) when the scheduler goes idle —
    // proving the refcount/eviction path never leaks — and reuse must kick in.
    let mut s = Scheduler::new(cfg_prefix(4, 16));
    let total = s.num_total_blocks();
    let prompt = vec![1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    for round in 0..5u64 {
        s.add(req_tokens(round, prompt.clone(), 2));
        while s.has_unfinished() {
            let p = s.schedule().unwrap().expect("plan");
            let _ = s.commit(&logits(p.seqs.len(), 64)).unwrap();
        }
        assert_eq!(
            s.num_free_blocks(),
            total,
            "all blocks reclaimable when idle after round {round}"
        );
    }
    assert!(
        s.prefix_cache_reused_tokens() > 0,
        "the repeated prefix should have been reused"
    );
}

#[test]
fn prefix_cache_evicts_cached_blocks_under_pressure() {
    // Tiny pool. After one request finishes, its full blocks sit cached
    // (evictable). A new, unrelated request must still be admittable by
    // evicting them — blocks stay conserved and the cache makes room.
    let mut s = Scheduler::new(cfg_prefix(4, 3)); // 3 blocks
    s.add(req_tokens(0, vec![1, 2, 3, 4, 5, 6, 7, 8, 9], 1)); // 3 blocks, 2 cacheable
    let p = s.schedule().unwrap().unwrap();
    let _ = s.commit(&logits(p.seqs.len(), 10)).unwrap();
    assert_eq!(s.num_running(), 0);
    assert_eq!(s.num_free_blocks(), 3, "cached blocks count as reclaimable");

    // Unrelated 9-token prompt also needs 3 blocks -> must evict the 2 cached
    // ones plus take the free one.
    s.add(req_tokens(1, vec![20, 21, 22, 23, 24, 25, 26, 27, 28], 1));
    let p2 = s
        .schedule()
        .unwrap()
        .expect("admitted by evicting cached blocks");
    assert_eq!(p2.seqs[0].id, 1);
    assert_eq!(
        (p2.seqs[0].past_len, p2.seqs[0].q_len),
        (0, 9),
        "no false reuse"
    );
    assert_eq!(s.num_free_blocks(), 0, "all 3 blocks now in use");
}

#[test]
fn chunked_prefill_splits_prompt_and_discards_midchunk_tokens() {
    let mut s = Scheduler::new(cfg_budget(4, 16, 4));
    s.add(req(0, 10, 4)); // 10-token prompt, budget 4 -> chunks 4/4/2

    let p1 = s.schedule().unwrap().expect("chunk 1");
    assert_eq!((p1.seqs[0].q_len, p1.seqs[0].past_len), (4, 0));
    assert_eq!(p1.input_ids.len(), 4);
    assert!(
        s.commit(&logits(1, 10)).unwrap().is_empty(),
        "mid-prefill emits nothing"
    );

    let p2 = s.schedule().unwrap().expect("chunk 2");
    assert_eq!((p2.seqs[0].q_len, p2.seqs[0].past_len), (4, 4));
    assert!(s.commit(&logits(1, 10)).unwrap().is_empty());

    let p3 = s.schedule().unwrap().expect("final chunk");
    assert_eq!((p3.seqs[0].q_len, p3.seqs[0].past_len), (2, 8));
    let out = s.commit(&logits(1, 10)).unwrap();
    assert_eq!(out.len(), 1, "prefill complete -> first real token");
    assert_eq!(out[0].token, 0);
}

#[test]
fn decode_seqs_schedule_before_prefill_chunks() {
    let mut s = Scheduler::new(cfg_budget(4, 32, 4));
    s.add(req(0, 2, 8));
    let p = s.schedule().unwrap().expect("prefill seq 0");
    let _ = s.commit_tokens(&vec![0u32; p.seqs.len()]).unwrap(); // seq 0 now decoding

    s.add(req(1, 20, 8)); // long prompt arrives mid-flight
    let plan = s.schedule().unwrap().expect("mixed step");
    // Decode seq first (1 token), then the prefill chunk fills the rest (3).
    assert_eq!((plan.seqs[0].id, plan.seqs[0].q_len), (0, 1));
    assert_eq!((plan.seqs[1].id, plan.seqs[1].q_len), (1, 3));
    assert_eq!(plan.input_ids.len(), 4, "budget respected across the batch");
}

/// DECODE SHARE: beside a decoding sequence a capped plan reads at most the cap, ending on a
/// multiple of it; a cap of 0 plans decode only; with nothing in decode the cap is ignored. The
/// cap is one-shot.
#[test]
fn prefill_cap_limits_the_prompt_read_beside_a_decoder() {
    let mut s = Scheduler::new(cfg_budget(4, 64, 64));
    s.add(req(0, 2, 32));
    let p = s.schedule().unwrap().expect("prefill seq 0");
    let _ = s.commit_tokens(&vec![0u32; p.seqs.len()]).unwrap(); // seq 0 now decoding
    s.add(req(1, 40, 8));

    s.set_prefill_cap(Some(8));
    let plan = s.schedule().unwrap().expect("capped step");
    assert_eq!((plan.seqs[0].id, plan.seqs[0].q_len), (0, 1));
    assert_eq!((plan.seqs[1].id, plan.seqs[1].q_len), (1, 8));
    let _ = s.commit_tokens(&vec![0u32; plan.seqs.len()]).unwrap();

    s.set_prefill_cap(Some(0));
    let plan = s.schedule().unwrap().expect("decode-only step");
    assert_eq!(plan.seqs.len(), 1);
    assert_eq!((plan.seqs[0].id, plan.seqs[0].q_len), (0, 1));
    let _ = s.commit_tokens(&vec![0u32; plan.seqs.len()]).unwrap();

    // One-shot: the next plan is uncapped and reads the rest of the prompt (40 - 8).
    let plan = s.schedule().unwrap().expect("uncapped step");
    assert_eq!(
        (plan.seqs[1].id, plan.seqs[1].q_len, plan.seqs[1].past_len),
        (1, 32, 8)
    );
}

#[test]
fn prefill_cap_is_ignored_when_nothing_decodes() {
    let mut s = Scheduler::new(cfg_budget(4, 64, 64));
    s.add(req(0, 40, 8));
    s.set_prefill_cap(Some(0));
    let plan = s
        .schedule()
        .unwrap()
        .expect("a plan is never emptied by the cap");
    assert_eq!((plan.seqs[0].id, plan.seqs[0].q_len), (0, 40));
}

#[test]
fn oversized_prompt_is_admitted_and_chunked_not_stuck() {
    // A prompt bigger than the whole budget must still be admitted (and
    // chunked), not wait forever.
    let mut s = Scheduler::new(cfg_budget(4, 32, 4));
    s.add(req(0, 9, 2));
    let p1 = s.schedule().unwrap().expect("admitted");
    assert_eq!(s.num_waiting(), 0);
    assert_eq!(p1.seqs[0].q_len, 4);
}

#[test]
fn two_prefills_share_one_steps_budget_fcfs() {
    // Budget 4, two 3-token prompts: step 1 gives seq 0 its full 3 and spills
    // the remaining 1 into seq 1; step 2 finishes seq 1's remaining 2. This
    // pins the FCFS budget-spillover behavior (decode-first only reorders
    // decode ahead of prefill; among prefills it is FCFS by running order).
    let mut s = Scheduler::new(cfg_budget(4, 16, 4));
    s.add(req(0, 3, 8));
    s.add(req(1, 3, 8));

    let p1 = s.schedule().unwrap().expect("step 1");
    assert_eq!(p1.seqs.len(), 2);
    assert_eq!(
        (p1.seqs[0].id, p1.seqs[0].q_len, p1.seqs[0].past_len),
        (0, 3, 0)
    );
    assert_eq!(
        (p1.seqs[1].id, p1.seqs[1].q_len, p1.seqs[1].past_len),
        (1, 1, 0)
    );
    let out1 = s.commit(&logits(2, 10)).unwrap();
    assert_eq!(out1.len(), 1, "only seq 0 finished its prefill");
    assert_eq!(out1[0].id, 0);

    let p2 = s.schedule().unwrap().expect("step 2");
    // Seq 0 is now decode (1 pending) -> scheduled first; seq 1 finishes prefill.
    assert_eq!((p2.seqs[0].id, p2.seqs[0].q_len), (0, 1));
    assert_eq!(
        (p2.seqs[1].id, p2.seqs[1].q_len, p2.seqs[1].past_len),
        (1, 2, 1)
    );
    let out2 = s.commit(&logits(2, 10)).unwrap();
    assert_eq!(
        out2.len(),
        2,
        "both emit: seq 0 decodes, seq 1 finishes prefill"
    );
}

// ---- STATE SNAPSHOTS: the prefix cache on a model with recurrent layers ----

fn cfg_snapshots(num_blocks: usize) -> EngineConfig {
    // Every snapshot test here was worked out on the pre-2026-09-27 end-of-prompt boundary (one
    // window backed off); they keep testing that logic on the control arm. The 32-token tail
    // default has its own process: `tests/scheduler_snapshot_tail.rs`. And the 128-token window
    // alignment (the default until 2026-10-05; block-aligned since, the same file).
    std::env::set_var("ARF_SNAPSHOT_TAIL", "0");
    std::env::set_var("ARF_SNAPSHOT_ALIGN", "128");
    EngineConfig {
        block_size: 16,
        num_blocks,
        max_batch_size: 8,
        max_prefill_tokens: 4096,
        enable_prefix_cache: true,
        state_snapshots: true,
        ..Default::default()
    }
}

/// Drive one sequence to completion, returning every (past_len, q_len, snapshot_key,
/// restore_key) the scheduler planned for it. A snapshot key is reported back as saved.
fn run_to_end(s: &mut Scheduler, id: u64) -> Vec<(usize, usize, Option<u64>, Option<u64>)> {
    let mut seen = vec![];
    while s.has_unfinished() {
        let plan = s.schedule().unwrap().expect("a plan");
        for sp in plan.seqs.iter().filter(|sp| sp.id == id) {
            seen.push((sp.past_len, sp.q_len, sp.snapshot_key, sp.restore_key));
            if let Some(k) = sp.snapshot_key {
                // The real serving loop HOLDS the key until the sequence finishes (both halves of
                // a hit must become valid together); using note_snapshot here would test a path
                // the daemon does not take.
                s.hold_snapshot(sp.id, k);
            }
        }
        s.commit_tokens(&vec![7; plan.seqs.len()]).unwrap();
    }
    seen
}

/// A long prompt is prefilled in TWO chunks: one ending exactly on the last block boundary that
/// still leaves a token after it (where the snapshot is taken), then the rest. A short prompt
/// is left alone.
#[test]
fn snapshot_chunk_ends_on_the_prompts_last_block_boundary() {
    let mut s = Scheduler::new(cfg_snapshots(64));
    let prompt: Vec<u32> = (0..700).collect(); // align 128: last boundary 640, backed off to 512
    s.add(req_tokens(1, prompt, 2));
    let seen = run_to_end(&mut s, 1);
    assert_eq!((seen[0].0, seen[0].1), (0, 512));
    assert!(
        seen[0].2.is_some(),
        "the boundary chunk carries a snapshot key"
    );
    assert_eq!((seen[1].0, seen[1].1), (512, 188));
    assert!(seen[1..].iter().all(|c| c.2.is_none()));

    let mut short = Scheduler::new(cfg_snapshots(64));
    short.add(req_tokens(2, (0..100).collect(), 2));
    let seen = run_to_end(&mut short, 2);
    assert_eq!((seen[0].0, seen[0].1, seen[0].2), (0, 100, None));
}

/// A snapshot is NOT visible while the sequence that took it is still running: its KV blocks are
/// not published until it finishes, and attaching to the state sooner reads KV that is still being
/// written (measured 2026-09-21 — turn 2 of a conversation diverged).
#[test]
fn a_snapshot_is_invisible_until_its_sequence_finishes() {
    let mut s = Scheduler::new(cfg_snapshots(128));
    let turn1: Vec<u32> = (0..700).collect();
    s.add(req_tokens(1, turn1.clone(), 64)); // 64 output tokens: still running for many steps
                                             // Step until the snapshot chunk has been planned (and held), but the sequence is NOT done.
    let mut key = None;
    for _ in 0..4 {
        let plan = s.schedule().unwrap().expect("a plan");
        for sp in &plan.seqs {
            if let Some(k) = sp.snapshot_key {
                s.hold_snapshot(sp.id, k);
                key = Some(k);
            }
        }
        s.commit_tokens(&vec![7; plan.seqs.len()]).unwrap();
    }
    assert!(key.is_some(), "the snapshot chunk was planned");
    // A second turn arriving NOW must not attach: the first sequence is still generating.
    let mut turn2 = turn1.clone();
    turn2.extend(1000..1100);
    s.add(req_tokens(2, turn2, 2));
    let plan = s.schedule().unwrap().expect("a plan");
    let sp = plan
        .seqs
        .iter()
        .find(|sp| sp.id == 2)
        .expect("turn 2 scheduled");
    assert_eq!(
        (sp.past_len, sp.restore_key),
        (0, None),
        "turn 2 must prefill from zero while turn 1 is still running"
    );
}

/// The next turn of the same conversation matches down to the snapshot boundary — and ONLY
/// there: cached KV blocks without a recurrent snapshot are not a usable prefix on this model.
#[test]
fn prefix_hit_is_taken_only_down_to_a_snapshot() {
    let mut s = Scheduler::new(cfg_snapshots(128));
    let turn1: Vec<u32> = (0..700).collect();
    s.add(req_tokens(1, turn1.clone(), 2));
    let key = run_to_end(&mut s, 1)[0].2.expect("snapshot key");

    // Turn 2 = turn 1's prompt + more. Its first chunk starts AT the boundary, carrying the key.
    let mut turn2 = turn1.clone();
    turn2.extend(1000..1200);
    s.add(req_tokens(2, turn2.clone(), 2));
    let seen = run_to_end(&mut s, 2);
    assert_eq!(seen[0].0, 512, "prefill resumes at the snapshot boundary");
    assert_eq!(seen[0].3, Some(key), "and asks for that snapshot first");
    assert!(
        seen[1..].iter().all(|c| c.3.is_none()),
        "the restore is asked for once"
    );

    // The backend evicts that snapshot: the same prompt now prefills from zero, even though
    // every one of its KV blocks is still cached. Turn 1 finished, so nothing held it.
    assert_eq!(s.forget_snapshot(key), None);
    // (turn 2 saved its own snapshot further on; forget that too so nothing matches)
    for k in seen.iter().filter_map(|c| c.2) {
        s.forget_snapshot(k);
    }
    s.add(req_tokens(3, turn2, 2));
    let seen = run_to_end(&mut s, 3);
    assert_eq!((seen[0].0, seen[0].3), (0, None));
}

// ---- PREFIX ANCHORS: a snapshot at the end of the shared system + tools prefix ----

/// Drive `id` to completion like `run_to_end`, keeping each whole `SeqPlan` (the anchor flag
/// included). Every snapshot key is HELD, as the serving loop holds it.
fn run_plans(s: &mut Scheduler, id: u64) -> Vec<arf_core::scheduler::SeqPlan> {
    let mut seen = vec![];
    while s.has_unfinished() {
        let plan = s.schedule().unwrap().expect("a plan");
        for sp in &plan.seqs {
            if let Some(k) = sp.snapshot_key {
                s.hold_snapshot(sp.id, k);
            }
        }
        seen.extend(plan.seqs.iter().filter(|sp| sp.id == id).cloned());
        s.commit_tokens(&vec![7; plan.seqs.len()]).unwrap();
    }
    seen
}

/// A 1,100-token "system + tools" prefix followed by one session's first user message.
fn session(user: std::ops::Range<u32>) -> Vec<u32> {
    let mut p: Vec<u32> = (0..1100).collect();
    p.extend(user);
    p
}

fn anchored(id: u64, tokens: Vec<u32>, anchor: Option<usize>) -> Request {
    req_tokens(id, tokens, 2).with_prefix_anchor(anchor)
}

/// THE CASE THIS EXISTS FOR (Claude Code, 2026-09-26: 88-90 s on every new session). P1 = S + U1
/// takes an ANCHOR snapshot at |S| floored to the 128-token window (1,100 -> 1,024), in its own
/// chunk, then its end-of-prompt snapshot as before. P2 = S + U2 — a NEW session, different first
/// message — then resumes at the anchor, not at 0. The control (no anchor on P1) prefills P2 from
/// zero, which is what the first live run measured.
#[test]
fn a_new_session_resumes_at_the_anchor_of_the_shared_prefix() {
    let mut s = Scheduler::new(cfg_snapshots(512));
    s.add(anchored(1, session(10_000..10_600), Some(1100)));
    let p1 = run_plans(&mut s, 1);
    // 1,700 tokens: anchor chunk [0, 1024), then [1024, 1536) ending on the end-of-prompt
    // boundary (1,699 -> 1,664, backed off one window), then the rest.
    assert_eq!((p1[0].past_len, p1[0].q_len), (0, 1024));
    assert!(p1[0].snapshot_anchor, "the first chunk ends on the anchor");
    let anchor = p1[0].snapshot_key.expect("an anchor key");
    assert_eq!((p1[1].past_len, p1[1].q_len), (1024, 512));
    assert!(p1[1].snapshot_key.is_some() && !p1[1].snapshot_anchor);
    assert_ne!(p1[1].snapshot_key, Some(anchor));
    assert_eq!(p1[2].past_len, 1536);

    s.add(anchored(2, session(20_000..20_600), Some(1100)));
    let p2 = run_plans(&mut s, 2);
    assert_eq!(
        (p2[0].past_len, p2[0].restore_key),
        (1024, Some(anchor)),
        "a new session resumes at the anchor"
    );
    assert!(
        p2.iter().all(|sp| !sp.snapshot_anchor),
        "and does not take the anchor again"
    );
    assert!(
        p2.iter().any(|sp| sp.snapshot_key.is_some()),
        "its own end-of-prompt snapshot is still taken, for its next turn"
    );

    // Control: the same two prompts WITHOUT an anchor — P2 finds nothing to resume from.
    let mut c = Scheduler::new(cfg_snapshots(512));
    c.add(anchored(1, session(10_000..10_600), None));
    let c1 = run_plans(&mut c, 1);
    assert!(c1.iter().all(|sp| !sp.snapshot_anchor));
    c.add(anchored(2, session(20_000..20_600), None));
    let c2 = run_plans(&mut c, 2);
    assert_eq!((c2[0].past_len, c2[0].restore_key), (0, None));
}

/// A request evicted after its prefill passed the anchor (its client gave up) leaves the anchor:
/// the retry, and every later request with the same prefix, resumes there. Evicted before the
/// anchor, it leaves nothing.
#[test]
fn an_evicted_request_leaves_the_anchor_it_had_passed() {
    let mut s = Scheduler::new(cfg_snapshots(512));
    s.add(anchored(1, session(10_000..10_600), Some(1100)));
    // One step: the anchor chunk [0, 1024), its snapshot held as the actor does.
    let plan = s.schedule().unwrap().expect("the anchor chunk");
    assert!(plan.seqs[0].snapshot_anchor);
    let anchor = plan.seqs[0].snapshot_key.expect("an anchor key");
    s.hold_snapshot(1, anchor);
    s.commit_tokens(&vec![7; plan.seqs.len()]).unwrap();
    s.evict_seqs(&[1]);

    s.add(anchored(2, session(20_000..20_600), Some(1100)));
    let p2 = run_plans(&mut s, 2);
    assert_eq!(
        (p2[0].past_len, p2[0].restore_key),
        (1024, Some(anchor)),
        "the retry resumes at the anchor the evicted request took"
    );

    // Evicted BEFORE the anchor chunk ran: nothing to leave.
    let mut c = Scheduler::new(cfg_snapshots(512));
    c.add(anchored(1, session(10_000..10_600), Some(1100)));
    c.evict_seqs(&[1]);
    c.add(anchored(2, session(20_000..20_600), Some(1100)));
    let c2 = run_plans(&mut c, 2);
    assert_eq!((c2[0].past_len, c2[0].restore_key), (0, None));
}

/// An abandoned read steps aside for any sequence with work, keeps its place and its state, and
/// runs when the step would otherwise be empty. It is worth finishing while an anchor nobody has
/// published lies ahead of it, and not after.
#[test]
fn a_background_read_runs_only_when_nothing_else_has_work() {
    let mut s = Scheduler::new(cfg_snapshots(512));
    s.add(anchored(1, session(10_000..10_600), Some(1100)));
    assert!(!s.anchor_ahead(1), "not running yet");
    assert!(s.is_waiting(1));
    // Admit both; the plain request has a different prompt and no anchor.
    s.add(req_tokens(2, (50_000..50_300).collect(), 2));
    s.set_background(&[1]);
    let plan = s.schedule().unwrap().expect("a plan");
    assert!(
        plan.seqs.iter().all(|sp| sp.id == 2),
        "the abandoned read is not planned beside a request with work"
    );
    assert!(s.anchor_ahead(1), "its anchor is still ahead");
    for sp in &plan.seqs {
        if let Some(k) = sp.snapshot_key {
            s.hold_snapshot(sp.id, k);
        }
    }
    s.commit_tokens(&vec![7; plan.seqs.len()]).unwrap();
    // Drive request 2 to its end with 1 still in the background.
    while s.num_running() > 1 {
        s.set_background(&[1]);
        let plan = s.schedule().unwrap().expect("a plan");
        assert!(plan.seqs.iter().all(|sp| sp.id == 2));
        for sp in &plan.seqs {
            if let Some(k) = sp.snapshot_key {
                s.hold_snapshot(sp.id, k);
            }
        }
        s.commit_tokens(&vec![7; plan.seqs.len()]).unwrap();
    }
    // Alone, it runs, in short background steps (`BACKGROUND_STEP_TOKENS`), to its anchor at 1,024.
    let anchor = loop {
        s.set_background(&[1]);
        let plan = s.schedule().unwrap().expect("the background read, alone");
        assert_eq!((plan.seqs[0].id, plan.seqs[0].q_len), (1, 128));
        let sp = &plan.seqs[0];
        let at = sp.past_len + sp.q_len;
        let key = sp.snapshot_key;
        if let Some(k) = key {
            s.hold_snapshot(1, k);
        }
        s.commit_tokens(&vec![7; plan.seqs.len()]).unwrap();
        if at == 1024 {
            break key.expect("an anchor key at 1,024");
        }
    };
    assert!(!s.anchor_ahead(1), "passed: nothing more to gain");
    s.evict_seqs(&[1]);
    // And what it read serves the next request with that prefix.
    s.add(anchored(3, session(20_000..20_600), Some(1100)));
    let p3 = run_plans(&mut s, 3);
    assert_eq!((p3[0].past_len, p3[0].restore_key), (1024, Some(anchor)));
}

/// An abandoned read is worth finishing to its own end of prompt, and evicting it there leaves
/// that snapshot: the same prompt plus a few tokens — what a client that gave up sends next —
/// resumes from it.
#[test]
fn an_abandoned_read_leaves_its_end_of_prompt_snapshot() {
    let mut s = Scheduler::new(cfg_snapshots(512));
    let prompt: Vec<u32> = (0..1700).collect();
    s.add(req_tokens(1, prompt.clone(), 2));
    // The end-of-prompt boundary on this config: 1,699 floored to 128 and backed off a window,
    // reached in short background steps (`BACKGROUND_STEP_TOKENS`).
    let key = loop {
        s.set_background(&[1]);
        let plan = s.schedule().unwrap().expect("its read, alone");
        let sp = &plan.seqs[0];
        assert_eq!(sp.q_len, 128);
        let (at, key) = (sp.past_len + sp.q_len, sp.snapshot_key);
        assert!(
            s.read_worth_finishing(1),
            "the boundary is ahead until the step commits"
        );
        if let Some(k) = key {
            s.hold_snapshot(1, k);
        }
        s.commit_tokens(&vec![7; plan.seqs.len()]).unwrap();
        if at == 1536 {
            break key.expect("the end-of-prompt snapshot");
        }
    };
    assert!(!s.read_worth_finishing(1), "passed: nothing more to gain");
    s.evict_seqs(&[1]);

    let mut next = prompt;
    next.extend(5000..5020);
    s.add(req_tokens(2, next, 2));
    let plan = s.schedule().unwrap().expect("the next request");
    assert_eq!(
        (plan.seqs[0].past_len, plan.seqs[0].restore_key),
        (1536, Some(key)),
        "resumes where the abandoned read stopped"
    );
}

/// ROLLING CHECKPOINTS: a long prompt ends a chunk every 1,024 tokens and saves a resume point
/// there; abandoned part-way, it leaves its latest one. The next request shares the prompt only up
/// to a point past it (Claude Code's next safety check differs ~70 tokens before the end) and
/// resumes there, not from the start.
#[test]
fn an_abandoned_long_read_leaves_its_latest_checkpoint() {
    let mut s = Scheduler::new(cfg_snapshots(2048));
    let prompt: Vec<u32> = (0..10_000).collect();
    s.add(req_tokens(1, prompt.clone(), 2));
    let mut cps = vec![];
    for _ in 0..6 {
        let plan = s.schedule().unwrap().expect("a chunk");
        let sp = &plan.seqs[0];
        assert!(sp.snapshot_checkpoint, "every chunk ends on a checkpoint");
        assert_eq!(sp.past_len + sp.q_len, (cps.len() + 1) * 1024);
        let key = sp.snapshot_key.unwrap();
        cps.push(key);
        s.hold_snapshot(1, key);
        s.commit_tokens(&vec![7; plan.seqs.len()]).unwrap();
    }
    // The client gave up at 6,144 tokens read.
    s.evict_seqs(&[1]);

    let mut next: Vec<u32> = prompt[..6_500].to_vec();
    next.extend(50_000..52_000);
    s.add(req_tokens(2, next, 2));
    let plan = s.schedule().unwrap().expect("the next request");
    assert_eq!(
        (plan.seqs[0].past_len, plan.seqs[0].restore_key),
        (6144, Some(cps[5])),
        "resumes from the latest checkpoint before it differs"
    );

    // A long prompt read to its end: the last checkpoint step lies past the prompt, and is not
    // looked at (it panicked on the GPU, a 13,780-token prompt, before the fix).
    let mut u = Scheduler::new(cfg_snapshots(2048));
    u.add(req_tokens(4, (0..9_100).collect(), 2));
    let seen = run_plans(&mut u, 4);
    assert_eq!(seen.iter().filter(|sp| sp.snapshot_checkpoint).count(), 8);

    // On the server's 512-token step every multiple of 1,024 is a chunk END, and those chunks
    // take the checkpoint too (until 2026-10-08 only a chunk the step lay strictly inside did:
    // a prompt read from the start took none).
    let mut c = cfg_snapshots(2048);
    c.max_prefill_tokens = 512;
    let mut v = Scheduler::new(c);
    v.add(req_tokens(5, (0..10_000).collect(), 2));
    let ends: Vec<usize> = run_plans(&mut v, 5)
        .iter()
        .filter(|sp| sp.snapshot_checkpoint)
        .map(|sp| sp.past_len + sp.q_len)
        .collect();
    assert_eq!(ends, (1..=9).map(|i| i * 1024).collect::<Vec<_>>());

    // A prompt under 8,192 tokens takes none.
    let mut t = Scheduler::new(cfg_snapshots(2048));
    t.add(req_tokens(3, (0..8_000).collect(), 2));
    let plan = t.schedule().unwrap().expect("a plan");
    assert!(!plan.seqs[0].snapshot_checkpoint);
}

/// SHORT READS FIRST: a short prompt admitted behind a long read is planned ahead of it, so an
/// agent's next turn does not wait for a whole safety check to be read; two long reads stay
/// first come, first served.
#[test]
fn a_short_read_goes_ahead_of_a_long_one() {
    let mut s = Scheduler::new(cfg_budget(16, 4096, 512));
    s.add(req_tokens(1, (0..20_000).collect(), 2));
    let plan = s.schedule().unwrap().expect("the long read");
    assert_eq!((plan.seqs[0].id, plan.seqs[0].q_len), (1, 512));
    s.commit_tokens(&[7]).unwrap();

    s.add(req_tokens(2, (50_000..50_300).collect(), 2));
    let plan = s.schedule().unwrap().expect("both");
    assert_eq!(
        plan.seqs
            .iter()
            .map(|sp| (sp.id, sp.q_len))
            .collect::<Vec<_>>(),
        vec![(2, 300), (1, 212)],
        "the short read first, the long one takes the rest of the step"
    );
    s.commit_tokens(&[7, 7]).unwrap();

    s.add(req_tokens(3, (60_000..70_000).collect(), 2));
    let plan = s.schedule().unwrap().expect("decode and the long reads");
    let prefills: Vec<u64> = plan
        .seqs
        .iter()
        .filter(|sp| sp.q_len > 1)
        .map(|sp| sp.id)
        .collect();
    assert_eq!(prefills, vec![1], "the older long read keeps its place");
}

/// A request that extends an abandoned read still short of its end of prompt waits for it, and
/// the read it waits for is no longer kept in the background; then it resumes from that read.
#[test]
fn a_request_extending_an_abandoned_read_waits_for_it() {
    let mut s = Scheduler::new(EngineConfig {
        max_prefill_tokens: 512,
        ..cfg_snapshots(512)
    });
    let prompt: Vec<u32> = (0..1700).collect();
    s.add(req_tokens(1, prompt.clone(), 2));
    let step = |s: &mut Scheduler| {
        let plan = s.schedule().unwrap().expect("a plan");
        for sp in &plan.seqs {
            if let Some(k) = sp.snapshot_key {
                s.hold_snapshot(sp.id, k);
            }
        }
        s.commit_tokens(&vec![7; plan.seqs.len()]).unwrap();
        plan
    };
    step(&mut s); // [0, 512)
    let mut next = prompt;
    next.extend(5000..5020);
    s.add(req_tokens(2, next, 2));
    // The client of 1 has gone; 2 extends what it is reading.
    while s.read_worth_finishing(1) {
        s.set_background(&[1]);
        let plan = step(&mut s);
        assert!(
            plan.seqs.iter().all(|sp| sp.id == 1),
            "2 waits; 1 runs although abandoned, because 2 waits for it"
        );
    }
    s.evict_seqs(&[1]);
    s.set_background(&[]);
    let plan = s.schedule().unwrap().expect("2 now");
    assert_eq!(
        (plan.seqs[0].id, plan.seqs[0].past_len),
        (2, 1536),
        "and resumes from its end of prompt"
    );
    assert!(plan.seqs[0].restore_key.is_some());
}

/// A request that shares a long abandoned read's prompt up to a point before its end — Claude
/// Code's second safety-check stage beside an abandoned first — waits for it too, and resumes from
/// the read's last checkpoint rather than reading the same tokens beside it.
#[test]
fn a_request_sharing_an_abandoned_read_waits_for_its_last_checkpoint() {
    let mut s = Scheduler::new(EngineConfig {
        max_prefill_tokens: 512,
        ..cfg_snapshots(4096)
    });
    let prompt: Vec<u32> = (0..10_000).collect();
    s.add(req_tokens(1, prompt.clone(), 2));
    let mut checkpoint = None;
    let mut step = |s: &mut Scheduler| {
        let plan = s.schedule().unwrap().expect("a plan");
        for sp in &plan.seqs {
            if let Some(k) = sp.snapshot_key {
                s.hold_snapshot(sp.id, k);
                if sp.snapshot_checkpoint {
                    checkpoint = Some((sp.past_len + sp.q_len, k));
                }
            }
        }
        s.commit_tokens(&vec![7; plan.seqs.len()]).unwrap();
        plan
    };
    for _ in 0..4 {
        step(&mut s); // [0, 2048)
    }
    let mut next: Vec<u32> = prompt[..9_800].to_vec();
    next.extend(50_000..50_300);
    s.add(req_tokens(2, next, 2));
    while s.read_worth_finishing(1) {
        s.set_background(&[1]);
        let plan = step(&mut s);
        assert!(
            plan.seqs.iter().all(|sp| sp.id == 1),
            "2 waits for the read it shares 9,800 tokens with"
        );
    }
    let (at, key) = checkpoint.expect("the read took checkpoints");
    assert_eq!(
        at, 9216,
        "its last: 9,216 + 512 <= its end of prompt at 9,856"
    );
    s.evict_seqs(&[1]);
    s.set_background(&[]);
    let plan = s.schedule().unwrap().expect("2 now");
    assert_eq!(
        (
            plan.seqs[0].id,
            plan.seqs[0].past_len,
            plan.seqs[0].restore_key
        ),
        (2, 9216, Some(key)),
        "and resumes from that checkpoint"
    );
}

/// A background read does not use up the step budget: a long request that arrives while one runs
/// is admitted at once and planned ahead of it.
#[test]
fn a_background_read_does_not_hold_up_admission() {
    let mut s = Scheduler::new(cfg_budget(16, 4096, 512));
    s.add(req_tokens(1, (0..20_000).collect(), 2));
    s.set_background(&[1]);
    let plan = s.schedule().unwrap().expect("the background read, alone");
    assert_eq!(plan.seqs[0].id, 1);
    s.commit_tokens(&[7]).unwrap();

    s.add(req_tokens(2, (50_000..60_000).collect(), 2));
    s.set_background(&[1]);
    let plan = s.schedule().unwrap().expect("the new request");
    assert_eq!(
        plan.seqs.iter().map(|sp| sp.id).collect::<Vec<_>>(),
        vec![2],
        "admitted although the background read has 19,488 tokens left, and planned alone"
    );
}

/// A request that resumes from its own saved state past the boundary a background read would
/// leave does not wait for that read (the 2026-10-08 release test: a project's safety check, ~45,000
/// tokens in from its own state, waited for a check read without the project and gave up).
#[test]
fn a_request_with_a_deeper_saved_state_does_not_wait_for_a_background_read() {
    let mut s = Scheduler::new(EngineConfig {
        max_prefill_tokens: 512,
        ..cfg_snapshots(4096)
    });
    // The check with the project's context, read to its end: 9,900 shared tokens, 5,000 of
    // context, a 100-token tail. It leaves its end-of-prompt state, at 14,848.
    let shared: Vec<u32> = (0..9_900).collect();
    let mut project = shared.clone();
    project.extend(20_000..25_000);
    let mut first = project.clone();
    first.extend(30_000..30_100);
    s.add(req_tokens(1, first, 2));
    run_plans(&mut s, 1);
    // The check without the project (the warm-up's), read in the background from the start: it
    // shares 9,900 tokens with the project's and would leave its last checkpoint at 9,216.
    let mut bare = shared.clone();
    bare.extend(40_000..40_100);
    s.add(req_tokens(2, bare, 2));
    s.set_background(&[2]);
    s.schedule().unwrap().expect("the background read starts");
    s.commit_tokens(&[7]).unwrap();
    // The project's next check resumes at 14,848 by itself: it does not wait for 9,216.
    let mut next = project;
    next.extend(50_000..50_100);
    s.add(req_tokens(3, next, 2));
    s.set_background(&[2]);
    let plan = s.schedule().unwrap().expect("the project's check");
    let sp = plan.seqs.iter().find(|sp| sp.id == 3);
    assert_eq!(
        sp.map(|sp| sp.past_len),
        Some(14_848),
        "admitted, not waiting"
    );
}

/// A step of background reads alone is short, so a request that arrives meanwhile waits for one
/// window, not a whole step budget.
#[test]
fn a_background_step_is_short() {
    let mut s = Scheduler::new(cfg_budget(16, 4096, 512));
    s.add(req_tokens(1, (0..10_000).collect(), 2));
    s.set_background(&[1]);
    let plan = s.schedule().unwrap().expect("the background read");
    assert_eq!(plan.seqs[0].q_len, 128);
    s.commit_tokens(&[7]).unwrap();
    s.set_background(&[]);
    let plan = s.schedule().unwrap().expect("the same read, foreground");
    assert_eq!(plan.seqs[0].q_len, 512);
}

/// No anchor below 1,024 tokens (a short system prompt prefills cheaply), none for a sequence that
/// resumed at or past it (the next turn of an anchored conversation), and none twice: a new
/// session admitted in the same step as another, or while the other still HOLDS the anchor, takes
/// no second copy of the same state.
#[test]
fn an_anchor_is_skipped_when_short_resumed_past_or_already_taken() {
    // Short: 1,000 floors to 896 < 1,024.
    let mut s = Scheduler::new(cfg_snapshots(512));
    s.add(anchored(1, session(10_000..10_600), Some(1000)));
    assert!(run_plans(&mut s, 1).iter().all(|sp| !sp.snapshot_anchor));

    // Resumed past it: turn 2 of an anchored conversation resumes at turn 1's end-of-prompt
    // snapshot (1,536), beyond the anchor (1,024).
    let mut s = Scheduler::new(cfg_snapshots(512));
    let turn1 = session(10_000..10_600);
    s.add(anchored(1, turn1.clone(), Some(1100)));
    run_plans(&mut s, 1);
    let mut turn2 = turn1;
    turn2.extend(30_000..30_300);
    s.add(anchored(2, turn2, Some(1100)));
    let t2 = run_plans(&mut s, 2);
    assert_eq!(
        t2[0].past_len, 1536,
        "turn 2 resumes at turn 1's end snapshot"
    );
    assert!(t2.iter().all(|sp| !sp.snapshot_anchor));

    // Taken in the same step: two new sessions admitted together plan ONE anchor between them.
    let mut s = Scheduler::new(cfg_snapshots(512));
    s.add(anchored(1, session(10_000..10_600), Some(1100)));
    s.add(anchored(2, session(20_000..20_600), Some(1100)));
    let plan = s.schedule().unwrap().expect("a plan");
    let anchors: Vec<u64> = plan
        .seqs
        .iter()
        .filter(|sp| sp.snapshot_anchor)
        .map(|sp| sp.id)
        .collect();
    assert_eq!(anchors, vec![1], "only the first session takes the anchor");
    for sp in &plan.seqs {
        if let Some(k) = sp.snapshot_key {
            s.hold_snapshot(sp.id, k);
        }
    }
    s.commit_tokens(&vec![7; plan.seqs.len()]).unwrap();

    // Held: a third session arriving while session 1 still holds the anchor takes none either.
    s.add(anchored(3, session(40_000..40_600), Some(1100)));
    let p3 = run_plans(&mut s, 3);
    assert!(p3.iter().all(|sp| !sp.snapshot_anchor));
}

/// A short first message puts the end-of-prompt boundary INSIDE the shared prefix (1,320 tokens:
/// 1,319 -> 1,280, backed off to 1,152 <= the 1,280 anchor). That snapshot covers shared-prefix
/// tokens only, so it is filed as the anchor rather than as a turn-end snapshot.
///
/// **Corrected 2026-09-26 (review):** filing 1,152 as the anchor cost a SECOND anchor on turn 2
/// (it resumed at 1,152 with the 1,280 anchor still ahead), so one agent held both default
/// anchor slots and evicted another agent's. Turn 1 now takes ONE snapshot, at the anchor
/// (1,280), and no end-of-prompt one; turn 2 resumes there and takes only its turn-end snapshot;
/// a new session resumes there too. One anchor-pool entry for the conversation, not two.
#[test]
fn an_end_of_prompt_snapshot_inside_the_shared_prefix_is_the_anchor() {
    let mut s = Scheduler::new(cfg_snapshots(512));
    let mut p: Vec<u32> = (0..1300).collect();
    p.extend(50_000..50_020);
    s.add(anchored(1, p.clone(), Some(1300)));
    let plans = run_plans(&mut s, 1);
    let snaps: Vec<(usize, bool)> = plans
        .iter()
        .filter(|sp| sp.snapshot_key.is_some())
        .map(|sp| (sp.past_len + sp.q_len, sp.snapshot_anchor))
        .collect();
    assert_eq!(snaps, vec![(1280, true)], "one snapshot, at the anchor");
    let anchor = plans
        .iter()
        .find_map(|sp| sp.snapshot_key)
        .expect("the anchor key");

    // Turn 2 of the SAME conversation: resumes at the anchor, and no second anchor.
    let mut turn2 = p;
    turn2.extend(60_000..60_400);
    s.add(anchored(2, turn2, Some(1300)));
    let t2 = run_plans(&mut s, 2);
    assert_eq!((t2[0].past_len, t2[0].restore_key), (1280, Some(anchor)));
    let t2_snaps: Vec<(usize, bool)> = t2
        .iter()
        .filter(|sp| sp.snapshot_key.is_some())
        .map(|sp| (sp.past_len + sp.q_len, sp.snapshot_anchor))
        .collect();
    assert_eq!(
        t2_snaps,
        vec![(1536, false)],
        "turn 2 takes its turn-end snapshot only (1,719 -> 1,664, backed off to 1,536)"
    );

    // A NEW session (same 1,300-token prefix, another short first message) resumes there too.
    let mut other: Vec<u32> = (0..1300).collect();
    other.extend(70_000..70_020);
    s.add(anchored(3, other, Some(1300)));
    let p3 = run_plans(&mut s, 3);
    assert_eq!((p3[0].past_len, p3[0].restore_key), (1280, Some(anchor)));
    assert!(
        p3.iter().all(|sp| sp.snapshot_key.is_none()),
        "nothing new to save: {p3:?}"
    );
}

/// A snapshot the backend EVICTED while the sequence that took it was still running must not be
/// published when that sequence finishes: a later prompt would match to a key the backend no
/// longer has, and the serving loop treats that restore miss as fatal. (Before 2026-09-26 the
/// finish published it anyway — `forget` had run first and found nothing to remove.)
#[test]
fn a_snapshot_evicted_while_held_is_never_published() {
    let mut s = Scheduler::new(cfg_snapshots(128));
    let turn1: Vec<u32> = (0..700).collect();
    s.add(req_tokens(1, turn1.clone(), 16));
    let mut key = None;
    while key.is_none() {
        let plan = s.schedule().unwrap().expect("a plan");
        for sp in &plan.seqs {
            if let Some(k) = sp.snapshot_key {
                s.hold_snapshot(sp.id, k);
                key = Some(k);
            }
        }
        s.commit_tokens(&vec![7; plan.seqs.len()]).unwrap();
    }
    // The backend's LRU drops it while sequence 1 is still prefilling / decoding — and the
    // scheduler names sequence 1 as the holder (the agent bench's evicted-while-held count counts these).
    assert_eq!(s.forget_snapshot(key.unwrap()), Some(1));
    run_to_end(&mut s, 1);
    let mut turn2 = turn1;
    turn2.extend(1000..1100);
    s.add(req_tokens(2, turn2, 2));
    let seen = run_to_end(&mut s, 2);
    assert_eq!(
        (seen[0].0, seen[0].3),
        (0, None),
        "turn 2 must not be sent to a snapshot the backend evicted"
    );
}

// ---- JUNCTION SNAPSHOTS: where a prompt leaves cached history (2026-09-26) ----

/// S = a 1,100-token "system + tools" prefix (anchor 1,100 -> 1,024), R = 2,000 tokens every
/// session of one project shares after it (Claude Code's <system-reminder> + environment
/// blocks), T = the session's own task text (600 tokens from `task`).
fn project_session(task: u32) -> Vec<u32> {
    let mut p: Vec<u32> = (0..1100).collect();
    p.extend(100_000..102_000);
    p.extend(task..task + 600);
    p
}

/// The snapshots a sequence's plans took: `(boundary, anchor, junction)`.
fn snaps_of(plans: &[arf_core::scheduler::SeqPlan]) -> Vec<(usize, bool, bool)> {
    plans
        .iter()
        .filter(|sp| sp.snapshot_key.is_some())
        .map(|sp| {
            (
                sp.past_len + sp.q_len,
                sp.snapshot_anchor,
                sp.snapshot_junction,
            )
        })
        .collect()
}

fn junction_key(plans: &[arf_core::scheduler::SeqPlan]) -> Option<u64> {
    plans
        .iter()
        .find(|sp| sp.snapshot_junction)
        .and_then(|sp| sp.snapshot_key)
}

/// THE CASE THIS EXISTS FOR. P1 = S + R + T1 takes the anchor (1,024) and its end-of-prompt
/// snapshot (3,699 -> 3,584, backed off to 3,456 — inside T1, so no later session reaches it).
/// P2 = S + R + T2, a new session: the cache holds KV for S + R down to 3,088 (the last full
/// block before T2 diverges), but the only snapshot on that chain is the anchor, so it resumes
/// at 1,024 — 2,064 tokens short of the match — and plans a JUNCTION at 3,088 floored to the
/// 128 window: 3,072, in a chunk of its own, before its end-of-prompt snapshot. P3 = S + R + T3
/// then resumes at that junction, not at the anchor, and takes no junction of its own (its
/// match runs only 16 tokens past it).
#[test]
fn a_third_session_resumes_at_the_junction_the_second_one_took() {
    let mut s = Scheduler::new(cfg_snapshots(1024));
    s.add(anchored(1, project_session(10_000), Some(1100)));
    let p1 = run_plans(&mut s, 1);
    assert_eq!(
        snaps_of(&p1),
        vec![(1024, true, false), (3456, false, false)]
    );
    let anchor = p1[0].snapshot_key.expect("the anchor key");

    s.add(anchored(2, project_session(20_000), Some(1100)));
    let p2 = run_plans(&mut s, 2);
    assert_eq!(
        (p2[0].past_len, p2[0].restore_key),
        (1024, Some(anchor)),
        "P2 resumes at the anchor"
    );
    assert_eq!(
        (p2[0].q_len, p2[0].snapshot_junction),
        (2048, true),
        "and its first chunk ends exactly on the junction, 3,072"
    );
    assert_eq!(
        snaps_of(&p2),
        vec![(3072, false, true), (3456, false, false)],
        "anchor < junction < end-of-prompt, one key per chunk"
    );
    let junction = junction_key(&p2).expect("a junction key");

    s.add(anchored(3, project_session(30_000), Some(1100)));
    let p3 = run_plans(&mut s, 3);
    assert_eq!(
        (p3[0].past_len, p3[0].restore_key),
        (3072, Some(junction)),
        "P3 resumes at the junction, 2,048 tokens past the anchor"
    );
    assert_eq!(
        snaps_of(&p3),
        vec![(3456, false, false)],
        "P3 takes no junction (its match runs 16 tokens past it) — only its end-of-prompt one"
    );
}

/// No junction when the cached match runs < 512 tokens past the resume point (a 300-token R:
/// 1,392 - 1,024 = 368), when the boundary falls below 1,024 (no anchor, a 900-token shared
/// prefix: 896), or when a snapshot for that key already exists — held by a session still
/// running, or planned in the same step by a session admitted together.
#[test]
fn no_junction_when_the_extra_match_is_short_or_the_key_is_taken() {
    // Short extra match.
    let mut s = Scheduler::new(cfg_snapshots(1024));
    let short = |task: u32| {
        let mut p: Vec<u32> = (0..1100).collect();
        p.extend(100_000..100_300);
        p.extend(task..task + 600);
        p
    };
    s.add(anchored(1, short(10_000), Some(1100)));
    run_plans(&mut s, 1);
    s.add(anchored(2, short(20_000), Some(1100)));
    let p2 = run_plans(&mut s, 2);
    assert_eq!(p2[0].past_len, 1024, "resumes at the anchor");
    assert!(p2.iter().all(|sp| !sp.snapshot_junction), "{p2:?}");

    // Below the 1,024 floor: no anchor, 900 shared tokens — the match (896) is >= 512 past the
    // resume point (0) but the boundary is not worth a slot.
    let mut s = Scheduler::new(cfg_snapshots(1024));
    let small = |task: u32| {
        let mut p: Vec<u32> = (0..900).collect();
        p.extend(task..task + 600);
        p
    };
    s.add(anchored(1, small(10_000), None));
    run_plans(&mut s, 1);
    s.add(anchored(2, small(20_000), None));
    assert!(run_plans(&mut s, 2).iter().all(|sp| !sp.snapshot_junction));

    // Held: P2 took the junction and is still decoding when P3 arrives. P3's KV match reaches
    // the same boundary (P1's published blocks), but the key is held — no second copy.
    let mut s = Scheduler::new(cfg_snapshots(1024));
    s.add(anchored(1, project_session(10_000), Some(1100)));
    run_plans(&mut s, 1);
    s.add(req_tokens(2, project_session(20_000), 64).with_prefix_anchor(Some(1100)));
    let mut held = None;
    while held.is_none() {
        let plan = s.schedule().unwrap().expect("a plan");
        for sp in &plan.seqs {
            if let Some(k) = sp.snapshot_key {
                s.hold_snapshot(sp.id, k);
                if sp.snapshot_junction {
                    held = Some(k);
                }
            }
        }
        s.commit_tokens(&vec![7; plan.seqs.len()]).unwrap();
    }
    s.add(anchored(3, project_session(30_000), Some(1100)));
    let p3 = run_plans(&mut s, 3);
    assert_eq!(p3[0].past_len, 1024, "the junction is not published yet");
    assert!(
        p3.iter().all(|sp| !sp.snapshot_junction),
        "and P3 does not take it again while P2 holds it: {p3:?}"
    );

    // Same step: two new sessions admitted together plan ONE junction between them.
    let mut s = Scheduler::new(cfg_snapshots(1024));
    s.add(anchored(1, project_session(10_000), Some(1100)));
    run_plans(&mut s, 1);
    s.add(anchored(2, project_session(20_000), Some(1100)));
    s.add(anchored(3, project_session(30_000), Some(1100)));
    let plan = s.schedule().unwrap().expect("a plan");
    let junctions: Vec<u64> = plan
        .seqs
        .iter()
        .filter(|sp| sp.snapshot_junction)
        .map(|sp| sp.id)
        .collect();
    assert_eq!(
        junctions,
        vec![2],
        "only the first session takes the junction"
    );
}

/// The junction boundary is a multiple of the 128-token window (a CORRECTNESS requirement — a
/// resume anywhere else returns different text), the chunk ends exactly there even when the
/// step budget ends one token short of it, and the one-token chunk that leaves, `[3071, 3072)`,
/// is NOT a decode row (`completes_prompt` false) — the serving loop speculates only on decode
/// rows, so no draft verify can run inside the prompt and leak into the saved state (the
/// 2026-09-26 anchor review fix; tests/actor_spec_gate.rs covers the actor side).
#[test]
fn the_junction_is_window_aligned_and_its_one_token_chunk_is_not_a_decode_row() {
    let mut s = Scheduler::new(EngineConfig {
        max_prefill_tokens: 2047,
        ..cfg_snapshots(1024)
    });
    s.add(anchored(1, project_session(10_000), Some(1100)));
    run_plans(&mut s, 1);
    let p2_prompt = project_session(20_000);
    let len = p2_prompt.len();
    s.add(anchored(2, p2_prompt, Some(1100)));
    let p2 = run_plans(&mut s, 2);
    let j = p2
        .iter()
        .find(|sp| sp.snapshot_junction)
        .expect("a junction");
    assert_eq!(
        (j.past_len, j.q_len, j.past_len + j.q_len),
        (3071, 1, 3072),
        "a 2,047-token first chunk from 1,024 leaves ONE token before the junction"
    );
    assert_eq!((j.past_len + j.q_len) % 128, 0, "window-aligned");
    assert!(!j.completes_prompt, "a mid-prompt chunk, not a decode row");
    for sp in &p2 {
        if sp.q_len == 1 && sp.past_len + 1 < len {
            assert!(
                !sp.completes_prompt,
                "a one-token chunk inside the prompt must never look like a decode row: {sp:?}"
            );
        }
    }
}

/// The eviction-while-held fix holds for junctions: a junction the backend evicted while the
/// session that took it was still running is NOT published when that session finishes, so the
/// next session resumes at the anchor (not at a key the backend no longer has, which the serving
/// loop treats as fatal) — and takes the junction again.
#[test]
fn a_junction_evicted_while_held_is_never_published() {
    let mut s = Scheduler::new(cfg_snapshots(1024));
    s.add(anchored(1, project_session(10_000), Some(1100)));
    let anchor = run_plans(&mut s, 1)[0].snapshot_key.expect("the anchor");
    s.add(req_tokens(2, project_session(20_000), 16).with_prefix_anchor(Some(1100)));
    let mut junction = None;
    while junction.is_none() {
        let plan = s.schedule().unwrap().expect("a plan");
        for sp in &plan.seqs {
            if let Some(k) = sp.snapshot_key {
                s.hold_snapshot(sp.id, k);
                if sp.snapshot_junction {
                    junction = Some(k);
                }
            }
        }
        s.commit_tokens(&vec![7; plan.seqs.len()]).unwrap();
    }
    // The backend's LRU drops it while session 2 is still prefilling.
    s.forget_snapshot(junction.unwrap());
    run_plans(&mut s, 2);

    s.add(anchored(3, project_session(30_000), Some(1100)));
    let p3 = run_plans(&mut s, 3);
    assert_eq!(
        (p3[0].past_len, p3[0].restore_key),
        (1024, Some(anchor)),
        "P3 must not be sent to the evicted junction"
    );
    assert_eq!(
        junction_key(&p3),
        junction,
        "and it takes the junction again, at the same boundary"
    );
}

// ---- SESSION-START SNAPSHOTS: a conversation's first prompt end, kept (2026-09-26) ----

mod snapshot_pools;
use snapshot_pools::{first_turn, next_turn, run_pooled, saves, Pool, Pools};

/// THE CASE THIS EXISTS FOR (measured 2026-09-26: a repeated Claude Code task's first
/// turn 14.4 s, resumed only at the anchor; another engine s). A 7-turn conversation through the
/// modelled pools (4 turn-end, 2 junction, 4 anchor slots — 2 before 2026-09-27). Turn 1 takes the anchor (1,024) and
/// its end-of-prompt snapshot (2,944), which — the first turn — goes to the JUNCTION pool. Turns
/// 2-7 each resume at the previous turn's end and file their own in the turn-end pool (six saves
/// into four slots: turns 2 and 3 are evicted). Turn 1's survives them all, and a NEW session
/// whose first request is identical — or the same up to its last window — resumes at it, not at
/// the anchor. A new session with a different first message resumes at the anchor and files its
/// own end in the junction pool.
///
/// Mutation-checked: with the session-start flag never set, turn 1's snapshot is evicted by
/// turn 5's save and the identical session resumes at the anchor (1,024), which is what
/// `tests/scheduler_no_session_start.rs` asserts for the opt-out; with it always set, turns 2-7
/// file in the junction pool and the per-turn assertion fails.
#[test]
fn a_first_turns_end_snapshot_survives_seven_turns_and_serves_the_next_identical_session() {
    assert!(arf_core::scheduler::session_start_snapshots_enabled());
    let mut s = Scheduler::new(cfg_snapshots(1024));
    let mut pools = Pools::default();

    let mut prompt = first_turn();
    s.add(anchored(1, prompt.clone(), Some(1100)));
    let t1 = run_pooled(&mut s, &mut pools, 1);
    assert_eq!(
        saves(&t1),
        vec![(1024, Pool::Anchor), (2944, Pool::Junction)],
        "turn 1: the anchor, then its end-of-prompt snapshot as a SESSION START"
    );
    assert!(t1[1].snapshot_session_start && !t1[1].snapshot_junction);
    let anchor = t1[0].snapshot_key.expect("the anchor");
    let start = t1[1].snapshot_key.expect("the session-start snapshot");

    // Turns 2..=7: each resumes at the previous turn's end, and files its own end-of-prompt
    // snapshot in the TURN-END pool — later turns are unchanged.
    let mut prev_end = (2944, start);
    let mut turn_keys = vec![];
    for k in 2..=7u32 {
        prompt = next_turn(&prompt, k);
        let id = k as u64;
        s.add(anchored(id, prompt.clone(), Some(1100)));
        let t = run_pooled(&mut s, &mut pools, id);
        assert_eq!(
            (t[0].past_len, t[0].restore_key),
            (prev_end.0, Some(prev_end.1)),
            "turn {k} resumes at turn {}'s end",
            k - 1
        );
        let sv = saves(&t);
        assert_eq!(
            sv.iter().map(|x| x.1).collect::<Vec<_>>(),
            vec![Pool::Turn],
            "turn {k}: one snapshot, its end, in the turn-end pool: {t:?}"
        );
        assert!(t.iter().all(|sp| !sp.snapshot_session_start));
        let key = t.iter().find_map(|sp| sp.snapshot_key).unwrap();
        prev_end = (sv[0].0, key);
        turn_keys.push(key);
    }
    // Six turn-end saves into four slots: turns 2 and 3 are gone, turn 1's session start is not.
    assert_eq!(pools.pool(turn_keys[0]), None, "turn 2's end was evicted");
    assert_eq!(pools.pool(turn_keys[1]), None, "turn 3's end was evicted");
    assert_eq!(pools.pool(start), Some(Pool::Junction), "turn 1's survived");
    assert_eq!(pools.pool(anchor), Some(Pool::Anchor));

    // A NEW session, identical first request: resumes at turn 1's end, not at the anchor.
    s.add(anchored(8, first_turn(), Some(1100)));
    let n = run_pooled(&mut s, &mut pools, 8);
    assert_eq!(
        (n[0].past_len, n[0].restore_key),
        (2944, Some(start)),
        "the identical first request resumes at the session-start snapshot"
    );
    assert!(saves(&n).is_empty(), "nothing new to save: {n:?}");

    // The same up to its last window (its last 100 tokens differ): the same resume point.
    let mut near = first_turn();
    near.truncate(3000);
    near.extend(90_000..90_100);
    s.add(anchored(9, near, Some(1100)));
    let n = run_pooled(&mut s, &mut pools, 9);
    assert_eq!((n[0].past_len, n[0].restore_key), (2944, Some(start)));

    // A new session with a DIFFERENT first message resumes at the anchor, and is itself a
    // session start.
    let mut other: Vec<u32> = (0..1100).collect();
    other.extend(50_000..52_000);
    s.add(anchored(10, other, Some(1100)));
    let n = run_pooled(&mut s, &mut pools, 10);
    assert_eq!((n[0].past_len, n[0].restore_key), (1024, Some(anchor)));
    assert_eq!(saves(&n), vec![(2944, Pool::Junction)]);
    assert!(n.iter().any(|sp| sp.snapshot_session_start));
}

/// Which resume points make a first turn: 0 (P1), the anchor (P2) and a JUNCTION another session
/// took (P3, as in `a_third_session_resumes_at_the_junction_the_second_one_took`) — each files its
/// end in the junction pool as a session start; the junction itself stays a junction, never both.
/// P3's next turn resumes at P3's session start and is a later turn (turn-end pool).
/// Mutation-checked: without the published-junction check (`shared_prefix_keys`) P3's end is a
/// turn-end snapshot and this fails.
#[test]
fn a_session_resuming_at_zero_the_anchor_or_a_junction_is_a_session_start() {
    let kinds = |p: &[arf_core::scheduler::SeqPlan]| -> Vec<(usize, bool, bool, bool)> {
        p.iter()
            .filter(|sp| sp.snapshot_key.is_some())
            .map(|sp| {
                (
                    sp.past_len + sp.q_len,
                    sp.snapshot_anchor,
                    sp.snapshot_junction,
                    sp.snapshot_session_start,
                )
            })
            .collect()
    };
    let mut s = Scheduler::new(cfg_snapshots(1024));
    s.add(anchored(1, project_session(10_000), Some(1100)));
    let p1 = run_plans(&mut s, 1);
    assert_eq!(p1[0].past_len, 0);
    assert_eq!(
        kinds(&p1),
        vec![(1024, true, false, false), (3456, false, false, true)]
    );
    s.add(anchored(2, project_session(20_000), Some(1100)));
    let p2 = run_plans(&mut s, 2);
    assert_eq!(p2[0].past_len, 1024, "P2 resumes at the anchor");
    assert_eq!(
        kinds(&p2),
        vec![(3072, false, true, false), (3456, false, false, true)]
    );
    // P3's task is 1,000 tokens, not 600: from the junction (3,072) a 600-token task ends at
    // 3,456, under the 512-token floor (`a_short_first_turn_is_not_a_session_start`).
    let mut p3_prompt = project_session(30_000);
    p3_prompt.truncate(3100);
    p3_prompt.extend(30_000..31_000);
    s.add(anchored(3, p3_prompt.clone(), Some(1100)));
    let p3 = run_plans(&mut s, 3);
    assert_eq!(p3[0].past_len, 3072, "P3 resumes at P2's junction");
    assert_eq!(kinds(&p3), vec![(3968, false, false, true)]);

    let mut t2 = p3_prompt;
    t2.extend(60_000..60_600);
    s.add(anchored(4, t2, Some(1100)));
    let p4 = run_plans(&mut s, 4);
    assert_eq!(
        p4[0].past_len, 3968,
        "P3's turn 2 resumes at its session start"
    );
    assert_eq!(kinds(&p4), vec![(4480, false, false, false)]);
}

/// The junction floors apply (a slot in the long-lived pool must be worth it): no session start
/// below 1,024 tokens, and none less than 512 tokens past the resume point — those end-of-prompt
/// snapshots stay ordinary turn-end ones, as before.
#[test]
fn a_short_first_turn_is_not_a_session_start() {
    // 700 tokens from 0: the end boundary is 512 < 1,024.
    let mut s = Scheduler::new(cfg_snapshots(1024));
    s.add(req_tokens(1, (0..700).collect(), 2));
    let p = run_plans(&mut s, 1);
    assert_eq!(p[0].past_len + p[0].q_len, 512);
    assert!(p[0].snapshot_key.is_some() && !p[0].snapshot_session_start);

    // Resuming at the anchor (1,024) with a 400-token first message: the end boundary (1,499 ->
    // 1,408 -> 1,280) is only 256 past the resume point.
    let mut s = Scheduler::new(cfg_snapshots(1024));
    s.add(anchored(1, session(10_000..10_600), Some(1100)));
    run_plans(&mut s, 1);
    s.add(anchored(2, session(20_000..20_400), Some(1100)));
    let p2 = run_plans(&mut s, 2);
    assert_eq!(p2[0].past_len, 1024);
    let ends: Vec<_> = p2.iter().filter(|sp| sp.snapshot_key.is_some()).collect();
    assert_eq!(ends.len(), 1);
    assert_eq!(ends[0].past_len + ends[0].q_len, 1280);
    assert!(!ends[0].snapshot_session_start, "{p2:?}");
}

// ---- TOOLS ANCHORS: a second anchor at the end of the tools block (2026-09-27) ----

/// An agent prompt as Qwen3.8 renders it with tools: the TOOLS block first (1,100 tokens, the
/// same for every session of the agent: tools anchor 1,100 -> 1,024), then the SYSTEM text
/// (1,300 tokens from `system` — Claude Code's carries the working directory, so it differs
/// between directories: system anchor 2,400 -> 2,304), then the first user message (600 tokens
/// from `user`). 3,000 tokens; the end-of-prompt boundary is 2,999 -> 2,944, backed off to 2,816.
fn tools_session(id: u64, system: u32, user: u32) -> Request {
    let mut p: Vec<u32> = (0..1100).collect();
    p.extend(system..system + 1300);
    p.extend(user..user + 600);
    req_tokens(id, p, 2)
        .with_prefix_anchor(Some(2400))
        .with_tools_anchor(Some(1100))
}

/// The snapshots a sequence's plans took: `(boundary, anchor, tools anchor)`.
fn anchor_snaps(plans: &[arf_core::scheduler::SeqPlan]) -> Vec<(usize, bool, bool)> {
    plans
        .iter()
        .filter(|sp| sp.snapshot_key.is_some())
        .map(|sp| {
            (
                sp.past_len + sp.q_len,
                sp.snapshot_anchor,
                sp.snapshot_tools_anchor,
            )
        })
        .collect()
}

/// THE CASE THIS EXISTS FOR (2026-09-27: `anchor snapshot at 13056` for a Claude Code session,
/// and no resume for the next two, in other working directories — ~75 s of re-prefill each).
/// P1 carries both anchors: it takes TWO anchor snapshots in one prefill, each ending a chunk —
/// the tools anchor (1,024) first, then the system anchor (2,304) — then its end-of-prompt one.
/// P2 = same tools, ANOTHER system text: its prompt agrees with P1's only up to the tools block,
/// so the system anchor is not on its chain, and it resumes at the TOOLS anchor — where before
/// it prefilled from zero (the control, last). It takes its own system anchor (another key),
/// not the tools anchor again. P3 = P1's system text, another first message: it resumes at
/// P1's SYSTEM anchor, the deeper of the two, exactly as without tools anchors.
#[test]
fn a_session_in_another_directory_resumes_at_the_tools_anchor() {
    assert!(arf_core::scheduler::tools_anchor_snapshots_enabled());
    let mut s = Scheduler::new(cfg_snapshots(1024));
    s.add(tools_session(1, 100_000, 10_000));
    let p1 = run_plans(&mut s, 1);
    let chunks: Vec<(usize, usize)> = p1
        .iter()
        .take(4)
        .map(|sp| (sp.past_len, sp.q_len))
        .collect();
    assert_eq!(
        chunks,
        vec![(0, 1024), (1024, 1280), (2304, 512), (2816, 184)],
        "a chunk ends at each anchor, then at the end-of-prompt boundary"
    );
    assert_eq!(
        anchor_snaps(&p1),
        vec![
            (1024, true, true),
            (2304, true, false),
            (2816, false, false)
        ]
    );
    let tools_key = p1[0].snapshot_key.unwrap();
    let system_key = p1[1].snapshot_key.unwrap();

    s.add(tools_session(2, 200_000, 20_000));
    let p2 = run_plans(&mut s, 2);
    assert_eq!(
        (p2[0].past_len, p2[0].restore_key),
        (1024, Some(tools_key)),
        "another system text resumes at the tools anchor"
    );
    assert_eq!(
        anchor_snaps(&p2),
        vec![(2304, true, false), (2816, false, false)],
        "its own system anchor; the tools anchor is not taken twice"
    );
    assert_ne!(p2[0].snapshot_key, Some(system_key));

    s.add(tools_session(3, 100_000, 30_000));
    let p3 = run_plans(&mut s, 3);
    assert_eq!(
        (p3[0].past_len, p3[0].restore_key),
        (2304, Some(system_key)),
        "the same system text resumes at the deeper, system anchor"
    );
    assert!(p3.iter().all(|sp| !sp.snapshot_anchor), "{p3:?}");

    // Control: P1 and P2 without the tools anchor — P2 finds nothing to resume from.
    let mut c = Scheduler::new(cfg_snapshots(1024));
    c.add(tools_session(1, 100_000, 10_000).with_tools_anchor(None));
    let c1 = run_plans(&mut c, 1);
    assert_eq!(
        anchor_snaps(&c1),
        vec![(2304, true, false), (2816, false, false)]
    );
    c.add(tools_session(2, 200_000, 20_000).with_tools_anchor(None));
    let c2 = run_plans(&mut c, 2);
    assert_eq!((c2[0].past_len, c2[0].restore_key), (0, None));
}

/// No tools anchor when it would add nothing: at the SAME window as the system anchor (a short
/// system text: 1,100 and 1,150 both floor to 1,024 — one snapshot, the system anchor), past
/// it (never from the HTTP layer; the scheduler does not trust that), or under 1,024 tokens.
/// And none twice: two sessions admitted in one step share one tools anchor.
#[test]
fn a_tools_anchor_is_skipped_when_it_adds_nothing_or_is_already_taken() {
    let prompt: Vec<u32> = (0..3000).collect();
    for (tools, system, want) in [
        (1100, 1150, vec![(1024, true, false)]),
        (2400, 1100, vec![(1024, true, false)]),
        (1000, 2400, vec![(2304, true, false)]),
    ] {
        let mut s = Scheduler::new(cfg_snapshots(1024));
        s.add(
            req_tokens(1, prompt.clone(), 2)
                .with_prefix_anchor(Some(system))
                .with_tools_anchor(Some(tools)),
        );
        let p = run_plans(&mut s, 1);
        let anchors: Vec<_> = anchor_snaps(&p).into_iter().filter(|a| a.1).collect();
        assert_eq!(anchors, want, "tools {tools}, system {system}");
    }

    let mut s = Scheduler::new(cfg_snapshots(1024));
    s.add(tools_session(1, 100_000, 10_000));
    s.add(tools_session(2, 200_000, 20_000));
    let plan = s.schedule().unwrap().expect("a plan");
    let tools: Vec<u64> = plan
        .seqs
        .iter()
        .filter(|sp| sp.snapshot_tools_anchor)
        .map(|sp| sp.id)
        .collect();
    assert_eq!(
        tools,
        vec![1],
        "only the first session takes the tools anchor"
    );
}

/// Through the modelled backend pools (4 anchor slots since 2026-09-27; was 2). Sessions in
/// three directories: each after the first resumes at the tools anchor (a restore miss panics
/// in `run_pooled`) and files its own system anchor — four anchors, all kept, so a new session
/// back in directory 1 resumes at directory 1's SYSTEM anchor. (Mutation-checked: with the
/// model at 2 anchor slots that session resumes at the tools anchor — directory 1's system
/// anchor was evicted by directory 2's.) Then directories 4 and 5: every anchor save now evicts
/// the least recently used anchor, and the tools anchor — restored by every new directory — is
/// never it.
#[test]
fn the_tools_anchor_survives_new_directories_in_the_anchor_pool() {
    let mut s = Scheduler::new(cfg_snapshots(4096));
    let mut pools = Pools::default();
    s.add(tools_session(1, 100_000, 10_000));
    let p1 = run_pooled(&mut s, &mut pools, 1);
    assert_eq!(
        saves(&p1),
        vec![
            (1024, Pool::Anchor),
            (2304, Pool::Anchor),
            (2816, Pool::Junction)
        ]
    );
    let tools_key = p1[0].snapshot_key.unwrap();
    let system_key = p1[1].snapshot_key.unwrap();
    let new_directory = |s: &mut Scheduler, pools: &mut Pools, id: u64, dir: u32| {
        s.add(tools_session(id, 100_000 * dir, 10_000 * id as u32));
        let p = run_pooled(s, pools, id);
        assert_eq!(
            (p[0].past_len, p[0].restore_key),
            (1024, Some(tools_key)),
            "directory {dir}"
        );
        assert_eq!(pools.pool(tools_key), Some(Pool::Anchor));
    };
    new_directory(&mut s, &mut pools, 2, 2);
    new_directory(&mut s, &mut pools, 3, 3);
    // Directory 1 again, another task: its system anchor is still held.
    s.add(tools_session(4, 100_000, 40_000));
    let p4 = run_pooled(&mut s, &mut pools, 4);
    assert_eq!(
        (p4[0].past_len, p4[0].restore_key),
        (2304, Some(system_key))
    );
    for (id, dir) in [(5, 4), (6, 5)] {
        new_directory(&mut s, &mut pools, id, dir);
    }
}
