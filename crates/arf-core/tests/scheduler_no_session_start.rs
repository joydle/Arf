//! `ARF_NO_SESSION_START_SNAPSHOT=1` — the session-start-snapshot control arm (2026-09-26). Its own
//! test binary because the switch is read once per process (a `OnceLock`) and this file turns it
//! OFF.

use arf_core::config::EngineConfig;
use arf_core::sampling::SamplingParams;
use arf_core::scheduler::{Request, Scheduler};

mod snapshot_pools;
use snapshot_pools::{first_turn, next_turn, run_pooled, saves, Pool, Pools};

fn cfg() -> EngineConfig {
    EngineConfig {
        block_size: 16,
        num_blocks: 1024,
        max_batch_size: 8,
        max_prefill_tokens: 4096,
        enable_prefix_cache: true,
        state_snapshots: true,
        ..Default::default()
    }
}

fn anchored(id: u64, tokens: Vec<u32>) -> Request {
    Request::new(id, tokens, SamplingParams::greedy(2)).with_prefix_anchor(Some(1100))
}

/// With the switch set, the 7-turn conversation of
/// `a_first_turns_end_snapshot_survives_seven_turns_and_serves_the_next_identical_session`
/// (tests/scheduler.rs) files turn 1's end-of-prompt snapshot in the TURN-END pool like every
/// other turn's; turn 5's save evicts it, and a new session with the identical first request
/// resumes only at the anchor (1,024) — the behaviour measured before this change (a repeated
/// Claude Code task's first turn resumed "from the anchor snapshot at 13184 tokens",
/// measured 2026-09-26), i.e. the A/B control.
#[test]
fn the_opt_out_files_the_first_turn_in_the_turn_pool_and_a_repeat_resumes_at_the_anchor() {
    std::env::set_var("ARF_NO_SESSION_START_SNAPSHOT", "1");
    std::env::set_var("ARF_SNAPSHOT_ALIGN", "128"); // worked out on the 128-token rule
    assert!(!arf_core::scheduler::session_start_snapshots_enabled());
    let mut s = Scheduler::new(cfg());
    let mut pools = Pools::default();

    let mut prompt = first_turn();
    s.add(anchored(1, prompt.clone()));
    let t1 = run_pooled(&mut s, &mut pools, 1);
    assert_eq!(
        saves(&t1),
        vec![(1024, Pool::Anchor), (2944, Pool::Turn)],
        "turn 1's end goes to the turn-end pool"
    );
    assert!(t1.iter().all(|sp| !sp.snapshot_session_start));
    let anchor = t1[0].snapshot_key.expect("the anchor");
    let end1 = t1[1].snapshot_key.expect("turn 1's end");
    for k in 2..=7u32 {
        prompt = next_turn(&prompt, k);
        s.add(anchored(k as u64, prompt.clone()));
        let t = run_pooled(&mut s, &mut pools, k as u64);
        assert!(t.iter().all(|sp| !sp.snapshot_session_start));
    }
    assert_eq!(pools.pool(end1), None, "turn 1's end was evicted");

    s.add(anchored(8, first_turn()));
    let n = run_pooled(&mut s, &mut pools, 8);
    assert_eq!(
        (n[0].past_len, n[0].restore_key),
        (1024, Some(anchor)),
        "the identical first request resumes only at the anchor"
    );
}
