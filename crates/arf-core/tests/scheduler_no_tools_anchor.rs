//! `ARF_NO_TOOLS_ANCHOR=1` — the tools-anchor control arm (2026-09-27). Its own test binary
//! because the switch is read once per process (a `OnceLock`) and this file turns it OFF.

use arf_core::config::EngineConfig;
use arf_core::sampling::SamplingParams;
use arf_core::scheduler::{Request, Scheduler, SeqPlan};

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

/// Tools block (1,100) + system text (1,300 from `system`) + first message (600 from `user`),
/// as `tools_session` in tests/scheduler.rs.
fn session(id: u64, system: u32, user: u32, tools_anchor: Option<usize>) -> Request {
    let mut p: Vec<u32> = (0..1100).collect();
    p.extend(system..system + 1300);
    p.extend(user..user + 600);
    Request::new(id, p, SamplingParams::greedy(2))
        .with_prefix_anchor(Some(2400))
        .with_tools_anchor(tools_anchor)
}

fn run_plans(s: &mut Scheduler, id: u64) -> Vec<SeqPlan> {
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

/// Every field of a plan the scheduler decides, but the block table and image rows.
fn shape(p: &[SeqPlan]) -> Vec<String> {
    p.iter()
        .map(|sp| {
            format!(
                "{} {} {:?} {:?} anchor={} tools={} junction={} start={} completes={}",
                sp.past_len,
                sp.q_len,
                sp.restore_key,
                sp.snapshot_key,
                sp.snapshot_anchor,
                sp.snapshot_tools_anchor,
                sp.snapshot_junction,
                sp.snapshot_session_start,
                sp.completes_prompt
            )
        })
        .collect()
}

/// With the switch set, the two sessions of `a_session_in_another_directory_resumes_at_the_
/// tools_anchor` (tests/scheduler.rs) plan EXACTLY what they plan with no tools anchor sent at
/// all — the behaviour before 2026-09-27: no tools snapshot, and the second session (another
/// system text) prefills from zero.
#[test]
fn the_opt_out_plans_exactly_as_without_a_tools_anchor() {
    std::env::set_var("ARF_NO_TOOLS_ANCHOR", "1");
    assert!(!arf_core::scheduler::tools_anchor_snapshots_enabled());
    let run = |tools: Option<usize>| {
        let mut s = Scheduler::new(cfg());
        s.add(session(1, 100_000, 10_000, tools));
        let p1 = run_plans(&mut s, 1);
        s.add(session(2, 200_000, 20_000, tools));
        let p2 = run_plans(&mut s, 2);
        (shape(&p1), shape(&p2), p2[0].past_len)
    };
    let off = run(Some(1100));
    assert_eq!(
        off,
        run(None),
        "the opt-out plans as if no tools anchor was sent"
    );
    assert_eq!(off.2, 0, "so the second directory prefills from zero");
    assert!(
        off.0.iter().all(|l| l.contains("tools=false")),
        "{:?}",
        off.0
    );
}
