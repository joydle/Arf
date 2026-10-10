//! What does speculation actually BUY on the CPU path, in the one currency that is
//! machine-independent: tokens emitted per forward pass?
//!
//! Wall-clock on the CPU says nothing about the GPU (different bottleneck entirely — the CPU
//! path is compute-bound, decode on Metal is latency-bound on outstanding loads). But
//! `tokens_per_forward` is the MECHANISM's own ratio: how many tokens each weight-pass yields.
//! On a bandwidth-bound engine that ratio IS the speedup ceiling, before overheads.
mod common;

use arf_core::model::speculative::generate_speculative;

#[test]
fn report_tokens_per_forward_across_k() {
    let model = common::tiny_model();
    println!("\n  k | tokens | forwards | tokens/forward | drafted | hits | accept%");
    println!("  --+--------+----------+----------------+---------+------+--------");
    for k in [1usize, 2, 3, 4, 8] {
        let r = generate_speculative(&model, &[1, 2, 3], 64, k, 2, 16);
        let s = &r.stats;
        let acc = if s.drafted > 0 {
            100.0 * s.draft_hits as f64 / s.drafted as f64
        } else {
            0.0
        };
        println!(
            "  {k} | {:6} | {:8} | {:14.3} | {:7} | {:4} | {:6.1}",
            r.tokens.len(),
            s.forward_passes,
            s.tokens_per_forward(),
            s.drafted,
            s.draft_hits,
            acc
        );
        // The invariant that makes the number meaningful.
        assert!(s.tokens_per_forward() >= 1.0);
    }
    println!();
}

/// Does speculation actually go FASTER on the CPU path, in wall clock?
///
/// This is the question `tokens_per_forward` cannot answer: a forward over k+1 rows costs more
/// than a forward over 1 row, so fewer-but-fatter passes only win if the extra rows are cheap
/// relative to the weight read. On the CPU that ratio is DIFFERENT from the GPU's — the CPU path
/// is compute-bound (extra rows cost real arithmetic), while Metal decode is latency-bound on
/// outstanding loads (extra rows ride along nearly free, measured marginal 0.279). So a CPU
/// result does NOT predict the GPU. It does prove the mechanism converts tokens/forward into
/// time, and it is the one speed check that a loaded box cannot corrupt into a false negative.
///
/// ARMS ARE INTERLEAVED (rule 3): greedy, spec, greedy, spec, ... and the MEDIAN is reported,
/// so a scheduler hiccup cannot land entirely on one arm.
#[test]
fn does_speculation_actually_save_wall_clock_on_cpu() {
    use std::time::Instant;
    let model = common::tiny_model();
    let (prompt, n) = (&[1u32, 2, 3][..], 96);

    let mut g_ms: Vec<f64> = Vec::new();
    let mut s_ms: Vec<f64> = Vec::new();
    const REPS: usize = 5;
    for _ in 0..REPS {
        let t = Instant::now();
        let g = generate_speculative(&model, prompt, n, 0, 2, 16); // k=0 = no drafting
        g_ms.push(t.elapsed().as_secs_f64() * 1e3);
        let t = Instant::now();
        let sp = generate_speculative(&model, prompt, n, 4, 2, 16);
        s_ms.push(t.elapsed().as_secs_f64() * 1e3);
        assert_eq!(
            g.tokens, sp.tokens,
            "spec must stay byte-identical to greedy"
        );
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let (gm, sm) = (med(&mut g_ms), med(&mut s_ms));
    println!("\n  CPU wall clock, {REPS} interleaved pairs, {n} tokens, median:");
    println!("    greedy (k=0) : {gm:8.2} ms");
    println!("    spec   (k=4) : {sm:8.2} ms");
    println!("    speedup      : {:8.3}x\n", gm / sm);
}
