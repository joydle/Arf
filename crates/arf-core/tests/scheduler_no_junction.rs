//! `ARF_NO_JUNCTION_SNAPSHOT=1` — the junction-snapshot control arm (2026-09-26). Its own test
//! binary because the switch is read once per process (a `OnceLock`) and this file turns it OFF.

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

/// S (1,100, anchor -> 1,024) + R (2,000 shared) + the session's own 600 tokens.
fn project_session(id: u64, task: u32) -> Request {
    let mut p: Vec<u32> = (0..1100).collect();
    p.extend(100_000..102_000);
    p.extend(task..task + 600);
    Request::new(id, p, SamplingParams::greedy(2)).with_prefix_anchor(Some(1100))
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

/// With the switch set, the three sessions of
/// `a_third_session_resumes_at_the_junction_the_second_one_took` (tests/scheduler.rs) take NO
/// junction: P2 and P3 both resume at the anchor — the behaviour before junctions, the A/B
/// control.
#[test]
fn the_opt_out_plans_no_junction_and_every_new_session_resumes_at_the_anchor() {
    std::env::set_var("ARF_NO_JUNCTION_SNAPSHOT", "1");
    std::env::set_var("ARF_SNAPSHOT_ALIGN", "128"); // worked out on the 128-token rule
    assert!(!arf_core::scheduler::junction_snapshots_enabled());
    let mut s = Scheduler::new(cfg());
    s.add(project_session(1, 10_000));
    let anchor = run_plans(&mut s, 1)[0].snapshot_key.expect("the anchor");
    for (id, task) in [(2u64, 20_000u32), (3, 30_000)] {
        s.add(project_session(id, task));
        let p = run_plans(&mut s, id);
        assert_eq!((p[0].past_len, p[0].restore_key), (1024, Some(anchor)));
        assert!(p.iter().all(|sp| !sp.snapshot_junction), "{p:?}");
    }
}
