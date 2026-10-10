//! The end-of-prompt snapshot boundary at its DEFAULTS: the last KV-block boundary (16 tokens here;
//! the 128-token window until 2026-10-05) at least `SNAPSHOT_TAIL_TOKENS` (32) before the prompt's
//! end (2026-09-27), or the request's measured header tail plus one when it has one (2026-10-05) —
//! not the old one-window back-off on the 128-token rule, which the other scheduler tests pin
//! (`ARF_SNAPSHOT_TAIL=0`, `ARF_SNAPSHOT_ALIGN=128`, the control arms). Its own test binary: the
//! tail is read once per process.
use arf_core::config::EngineConfig;
use arf_core::sampling::SamplingParams;
use arf_core::scheduler::{Request, Scheduler, SNAPSHOT_TAIL_TOKENS};

fn cfg() -> EngineConfig {
    EngineConfig {
        block_size: 16,
        num_blocks: 256,
        max_batch_size: 8,
        max_prefill_tokens: 4096,
        enable_prefix_cache: true,
        state_snapshots: true,
        ..Default::default()
    }
}

/// (past_len, q_len, snapshot_key, restore_key) of every chunk `id` ran, holding keys as the
/// serving loop does.
fn run(s: &mut Scheduler, id: u64) -> Vec<(usize, usize, Option<u64>, Option<u64>)> {
    let mut seen = vec![];
    while s.has_unfinished() {
        let plan = s.schedule().unwrap().expect("a plan");
        for sp in plan.seqs.iter().filter(|sp| sp.id == id) {
            seen.push((sp.past_len, sp.q_len, sp.snapshot_key, sp.restore_key));
            if let Some(k) = sp.snapshot_key {
                s.hold_snapshot(sp.id, k);
            }
        }
        s.commit_tokens(&vec![7; plan.seqs.len()]).unwrap();
    }
    seen
}

#[test]
fn the_end_snapshot_sits_one_tail_before_the_end_and_each_turn_resumes_further() {
    assert_eq!(SNAPSHOT_TAIL_TOKENS, 32);
    let mut s = Scheduler::new(cfg());
    // 700 tokens: (700 - 32) floored to 16 = 656 (128-aligned it was 640; the one-window rule, 512).
    let turn1: Vec<u32> = (0..700).collect();
    s.add(Request::new(1, turn1.clone(), SamplingParams::greedy(2)));
    let t1 = run(&mut s, 1);
    assert_eq!((t1[0].0, t1[0].1), (0, 656), "turn 1 snapshots at 656");
    let k1 = t1[0].2.expect("a key");

    // Turn 2 adds 200 tokens (an agent turn): resumes at 656, snapshots at (900 - 32) -> 864.
    let mut turn2 = turn1.clone();
    turn2.extend(1000..1200);
    s.add(Request::new(2, turn2.clone(), SamplingParams::greedy(2)));
    let t2 = run(&mut s, 2);
    assert_eq!(
        (t2[0].0, t2[0].3),
        (656, Some(k1)),
        "turn 2 resumes at turn 1's 656"
    );
    assert_eq!(t2[0].0 + t2[0].1, 864, "and snapshots at 864");

    // Turn 3 adds 150: resumes at 864 — the boundary ADVANCED (the one-window rule left turns 2
    // and 3 resuming at the same point when a turn added less than two windows).
    let mut turn3 = turn2;
    turn3.extend(2000..2150);
    s.add(Request::new(3, turn3, SamplingParams::greedy(2)));
    let t3 = run(&mut s, 3);
    assert_eq!(t3[0].0, 864, "turn 3 resumes at turn 2's 864");
}

#[test]
fn a_prompt_ending_inside_the_tail_backs_off_to_the_previous_window() {
    // 650 tokens: 650 - 32 = 618 -> 608 (624 or 640 would leave fewer than 32 tokens: the
    // fixed tail's margin for an assistant header the next turn renders differently).
    let mut s = Scheduler::new(cfg());
    s.add(Request::new(
        1,
        (0..650).collect(),
        SamplingParams::greedy(2),
    ));
    let t = run(&mut s, 1);
    assert_eq!((t[0].0, t[0].1), (0, 608));
}

#[test]
fn a_measured_header_moves_the_end_snapshot_to_the_last_window_before_it() {
    // 1,050 tokens whose template header is 5 tokens: (1050 - 5 - 1) floored to 16 = 1,040, so a
    // repeat re-prefills 10 tokens. The fixed tail gives (1050 - 32) -> 1,008: 42 tokens.
    // Measured 2026-10-05 on a repeated 34,972-token prompt (128-aligned then): 1.56 s to the
    // first token with the fixed tail, 510 ms with the header; block-aligned, 198 ms.
    let mut s = Scheduler::new(cfg());
    s.add(
        Request::new(1, (0..1050).collect(), SamplingParams::greedy(2)).with_header_tail(Some(5)),
    );
    let t = run(&mut s, 1);
    assert_eq!(t[0].0 + t[0].1, 1040, "with the header: snapshot at 1,040");

    let mut s = Scheduler::new(cfg());
    s.add(Request::new(
        1,
        (0..1050).collect(),
        SamplingParams::greedy(2),
    ));
    let t = run(&mut s, 1);
    assert_eq!(t[0].0 + t[0].1, 1008, "without it: the fixed tail, 1,008");
}
