//! GATE for `dflash_chain_sampled` (2026-09-26) — the SAMPLED GPU chain, NO MODEL: synthetic
//! selector tables, milliseconds. Runs the kernel on the GPU and compares, block by block, with
//!   * `dflash_chain_sampled_ref` — the kernel's statement-for-statement CPU reference, and
//!   * `dflash_chain_sampled_cpu` — the CPU SAMPLED SELECTOR's logic (`dflash_select`'s sampled
//!     branch, via `arf_core::sampling::sampled_draft_pick`), keyed exactly as it keys draws.
//!
//! The CPU-side agreement of the two references is a `cargo test` (dflash2_block tests). This is
//! the GPU half: it proves the kernel RAN (sentinel-filled outputs, anchor echoed — rule 8) and
//! that its drafts and q match. Expected: q candidate ids identical everywhere; |dq| at the float
//! noise of `precise::exp` vs Rust's `exp` (~1e-7); drafts identical except where a uniform
//! lands within that noise of a CDF boundary (rare — each is printed).
//!
//! Run: cargo run --release -p arf-gpu --example dflash_chain_sampled_probe [-- <cases, default 2000>]

#[cfg(target_os = "macos")]
fn main() {
    use arf_gpu::gpu::concurrent_metal::MetalIsland;
    use arf_gpu::gpu::metal::dflash2_block::{
        dflash_chain_greedy_ref, dflash_chain_sampled_cpu, dflash_chain_sampled_ref,
        dflash_chain_synthetic_tables, dflash_chain_uniforms, dflash_q_rows, DFLASH_BLOCK_ROWS,
        DFLASH_SEL_TOPK,
    };
    use arf_gpu::gpu::GpuContext;

    const K: usize = DFLASH_SEL_TOPK;
    const R: usize = DFLASH_BLOCK_ROWS;
    let cases: u64 = std::env::args()
        .nth(1)
        .map(|s| s.parse().expect("CASES"))
        .unwrap_or(2000);
    let ctx = GpuContext::new().expect("gpu context");
    let mut island = MetalIsland::new(&ctx).expect("metal island (need a Metal backend)");

    let (mut blocks, mut positions) = (0usize, 0usize);
    let (mut win_bad_ref, mut win_bad_cpu, mut qid_bad, mut left_greedy) = (0usize, 0, 0, 0);
    let mut dq_max = 0.0f32;
    for case in 0..cases {
        let (mut cands, mut unary, edges) = dflash_chain_synthetic_tables(case, case % 3 == 0);
        if case % 5 == 0 {
            cands[3 * K + 5] = u32::MAX; // holes: a position with 14 legal candidates
            cands[3 * K + 11] = u32::MAX;
        }
        if case % 97 == 0 {
            unary[2 * K + 4] = f32::NAN; // the degenerate-softmax fallback
            for j in 0..K {
                unary[5 * K + j] = f32::NEG_INFINITY;
            }
        }
        let temp = [0.7f32, 1.0, 0.3, 1.5, 1e-9, 0.6][case as usize % 6];
        let (seed, past) = (case.wrapping_mul(7919) + 3, 100 + case as usize * 13);
        let anchor = 42_000 + case as u32;
        let unif = dflash_chain_uniforms(seed, past);
        let (gw, gqi, gqp) = island
            .dflash_chain_sampled_selftest(&ctx, &cands, &unary, &edges, anchor, &unif, temp)
            .expect("dflash_chain_sampled dispatch");
        // rule 8: a kernel that never ran leaves the sentinels
        assert_eq!(
            gw[0], anchor,
            "case {case}: the anchor was not echoed — did the kernel run?"
        );
        assert!(
            !gw.contains(&0xDEAD_BEEF) && !gqi.contains(&0xDEAD_BEEF),
            "case {case}: sentinel left in the output — the kernel did not write it"
        );
        let gq = dflash_q_rows(&gqi, &gqp, R);
        let (rw, rq) = dflash_chain_sampled_ref(&cands, &unary, &edges, anchor, &unif, temp);
        let (cw, _) = dflash_chain_sampled_cpu(&cands, &unary, &edges, anchor, R, temp, seed, past);
        blocks += 1;
        positions += R - 1;
        if gw != rw {
            win_bad_ref += 1;
            eprintln!("case {case} (T {temp}): gpu {gw:?}\n              ref {rw:?}");
        }
        if gw != cw {
            win_bad_cpu += 1;
        }
        // q compared up to and including the first diverging position (its q is still built
        // from the same prev); after it the two sides score different edge rows, legitimately
        let agree = (1..R).take_while(|&p| gw[p] == rw[p]).count();
        for p in 1..(agree + 2).min(R) {
            let (a, b) = (&gq[p - 1], &rq[p - 1]);
            if a.len() != b.len() || a.iter().zip(b).any(|(x, y)| x.0 != y.0) {
                qid_bad += 1;
                eprintln!("case {case} pos {p}: q ids gpu {a:?} ref {b:?}");
                continue;
            }
            for (x, y) in a.iter().zip(b) {
                dq_max = dq_max.max((x.1 - y.1).abs());
            }
        }
        if temp > 0.1 {
            let g = dflash_chain_greedy_ref(&cands, &unary, &edges, anchor);
            left_greedy += gw.iter().zip(&g).filter(|(a, b)| a != b).count();
        }
    }
    println!("=== dflash_chain_sampled GPU PARITY (synthetic, no model) ===");
    println!("  {blocks} blocks, {positions} drafted positions");
    println!("  drafts != kernel reference     : {win_bad_ref} blocks");
    println!("  drafts != CPU sampled selector : {win_bad_cpu} blocks");
    println!("  q candidate ids differ         : {qid_bad} positions");
    println!("  max |q_gpu - q_ref|            : {dq_max:.3e}");
    println!("  positions off the greedy chain : {left_greedy} (T > 0.1 — the draw is drawing)");
    assert_eq!(qid_bad, 0, "q ids / order differ from the reference");
    assert!(
        dq_max < 1e-5,
        "q values differ beyond float noise: {dq_max:.3e}"
    );
    assert!(
        left_greedy > positions / 10,
        "the GPU chain barely leaves the greedy chain — is it sampling?"
    );
    assert!(
        win_bad_ref * 100 <= blocks,
        "drafts differ from the reference in {win_bad_ref} of {blocks} blocks (> 1%)"
    );
    println!("\nGATE PASS: dflash_chain_sampled == its CPU reference (drafts and q).");
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("dflash_chain_sampled_probe: the MSL kernel is macOS-only (Metal island). Skipped.");
}
