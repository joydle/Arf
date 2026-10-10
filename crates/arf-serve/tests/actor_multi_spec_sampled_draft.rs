//! SAMPLED DRAFTS PER SEGMENT (2026-09-27). Multi-stream speculation served sampled streams with
//! the POINT-MASS draft; the lone stream has drafted from the draft's own distribution by default
//! since 2026-09-27 (+9% acceptance). Now each sampled stream in a multi-stream cycle drafts SAMPLED
//! (`multi_sampled_draft`: the GPU selector, waited) and its segment's `RowSampler` carries the q
//! that draft was drawn from, so its rows accept with min(1, p/q).
//!
//! What these pin, through the actor with a mock backend (no GPU):
//! - a sampled segment's `draft_q` is EXACTLY the q its own draft was drawn from (its stream, its
//!   position — not another segment's, not a stale one), and its window carries that draft;
//! - its predictions follow min(1, p/q) with the residual norm(max(p - q, 0)) on a rejection —
//!   recomputed here independently of `sample_rows_q`, and both branches exercised;
//! - greedy streams keep the point-mass draft and the argmax, byte for byte;
//! - EXACTNESS in distribution: over thousands of tokens emitted through q-bearing records, the
//!   emitted tokens are distributed as the target p (a randomized probability-integral transform,
//!   chi-square on 10 bins). With a sampled draft the text is NOT seed-identical to plain sampling
//!   (the draft spends its own uniforms), so identity against `reference` would be the wrong test.
//!
//! Its own test binary: the actor reads `ARF_SPEC_K` and `ARF_NO_SAMPLED_DRAFT` once per process.
//! The opt-out is `actor_multi_spec_sampled_draft_off.rs`.

mod common;

use std::sync::Arc;

use arf_core::model::speculative::accepted_run;
use arf_core::sampling::{
    draw_from, keyed_uniform, purpose, RowSampler, SamplingParams, TokenPieceMap,
};
use arf_serve::actor::{spawn, FinishReason};
use common::*;

fn spec_on() {
    std::env::set_var("ARF_SPEC_K", "1");
}

/// The accepted run a q-bearing sampled window must produce, from the RULE, written out here
/// rather than taken from `sample_rows_q`: row r tests the draft x = window[r+1] and accepts it
/// when u_accept * q(x) < p(x) (probability min(1, p(x) / q(x))); the first rejection emits a draw
/// from norm(max(p - q, 0)) (x excluded); an all-accepted window emits the last row's draw from p.
fn expected_run(
    s: &RowSampler,
    prefix_len: usize,
    window: &[u32],
    logits_rows: &[Vec<f32>],
) -> Vec<u32> {
    let seed = s.params.seed;
    let mut run = Vec::new();
    for r in 0..window.len() {
        let p = s.target_dist(&logits_rows[r], window, r);
        let pos = prefix_len + r + 1;
        let of = |d: &[(u32, f32)], t: u32| d.iter().find(|c| c.0 == t).map_or(0.0, |c| c.1);
        if r + 1 < window.len() {
            let q = &s.draft_q[r];
            let x = window[r + 1];
            if of(q, x) > 0.0 && keyed_uniform(seed, pos, purpose::ACCEPT) * of(q, x) < of(&p, x) {
                run.push(x);
                continue;
            }
            let resid: Vec<(u32, f32)> = p
                .iter()
                .map(|&(t, pt)| (t, (pt - of(q, t)).max(0.0)))
                .filter(|&(t, w)| w > 0.0 && t != x)
                .collect();
            let u = keyed_uniform(seed, pos, purpose::RESIDUAL);
            let without_x: Vec<(u32, f32)> = p.iter().copied().filter(|c| c.0 != x).collect();
            run.push(
                draw_from(&resid, u)
                    .or_else(|| draw_from(&without_x, u))
                    .expect("residual"),
            );
            return run;
        }
        run.push(draw_from(&p, keyed_uniform(seed, pos, purpose::RESIDUAL)).expect("bonus"));
    }
    run
}

/// Two sampled streams and a greedy one share records (3 + 3 + 2). Every sampled segment carries
/// its own draft's q and accepts by min(1, p/q); the greedy stream keeps the point mass and emits
/// exactly the greedy text.
#[tokio::test]
async fn sampled_segments_carry_their_own_drafts_q_and_accept_with_min_p_over_q() {
    spec_on();
    let (mock, rec, go) = QMock::new();
    let (handle, _m, _join) = spawn(
        Box::new(mock),
        cfg(),
        Arc::new(TokenPieceMap::new(vec![])),
        None,
    );
    let n = 96;
    let params = [
        sampled(n, 31, 1.0),
        SamplingParams::greedy(n),
        sampled(n, 32, 1.0),
    ];
    let mut rxs = Vec::new();
    for (i, p) in params.iter().enumerate() {
        let id = 40 + i as u64;
        let (j, rx) = job(id, prompt(id), p.clone());
        handle.submit(j).unwrap();
        rxs.push(rx);
    }
    go.store(true, std::sync::atomic::Ordering::Release); // every job is queued
    for (i, rx) in rxs.into_iter().enumerate() {
        let (toks, fin) = collect(rx).await;
        assert_eq!(fin, Some(FinishReason::Length));
        assert_eq!(toks.len(), n);
        if i == 1 {
            assert_eq!(
                toks,
                reference(&prompt(41), &params[1], n),
                "the greedy stream's text must not change"
            );
        }
    }
    let rec = rec.lock().unwrap();
    assert!(
        !rec.draft_calls.contains(&41),
        "the greedy stream must keep the point-mass draft: {:?}",
        rec.draft_calls
    );
    assert!(
        rec.draft_calls.contains(&40) && rec.draft_calls.contains(&42),
        "both sampled streams must draft sampled: {:?}",
        rec.draft_calls
    );
    let (mut tested, mut accepted, mut rejected) = (0usize, 0usize, 0usize);
    for g in &rec.segs {
        let Some(s) = g.sampler.as_ref() else {
            assert_eq!(g.stream, 41, "only the greedy stream has no sampler");
            continue;
        };
        if g.window.len() < 2 {
            continue; // a one-row segment tests no draft
        }
        // THE q IS THIS SEGMENT'S OWN: its stream's draft at its position
        let (seed, d, q) = rec
            .drafts
            .get(&(g.stream, g.prefix_len))
            .unwrap_or_else(|| panic!("segment {g:?} has no sampled draft at its position"));
        assert_eq!(
            *seed, s.params.seed,
            "stream {}: the draft's seed",
            g.stream
        );
        assert_eq!(
            &g.window[1..],
            &d[..g.window.len() - 1],
            "stream {} at {}: the window must carry its draft",
            g.stream,
            g.prefix_len
        );
        assert_eq!(
            &s.draft_q, q,
            "stream {} at {}: the segment's sampler must carry the q ITS draft was drawn from",
            g.stream, g.prefix_len
        );
        // THE RULE: min(1, p/q), residual norm(max(p - q, 0))
        let rows: Vec<Vec<f32>> = g
            .window
            .iter()
            .enumerate()
            .map(|(r, &t)| model_logits(g.prefix_len + r, t))
            .collect();
        let want = expected_run(s, g.prefix_len, &g.window, &rows);
        let got = accepted_run(&g.window, &g.preds);
        assert_eq!(
            got, want,
            "stream {} at {}: window {:?}",
            g.stream, g.prefix_len, g.window
        );
        tested += 1;
        accepted += got.len() - 1;
        if got.len() < g.window.len() {
            rejected += 1;
        }
    }
    eprintln!(
        "q-bearing segments: {tested}, drafts accepted {accepted}, rejections {rejected}, \
         sampled draft calls {}",
        rec.draft_calls.len()
    );
    assert!(tested >= 20, "only {tested} q-bearing segments were served");
    assert!(
        accepted > 0 && rejected > 0,
        "both branches of min(1, p/q) must run: {accepted} accepted, {rejected} rejections"
    );
}

/// EXACTNESS IN DISTRIBUTION. Four sampled streams (3 + 3 + 1 + 1, the drafted pair rotating),
/// 3,000 tokens each: every emitted token's randomized PIT under the target p (the plain sampler's
/// distribution at its position) must be uniform. A segment accepting with a q that is not the one
/// its draft was drawn from — another segment's, a shifted one — skews the output away from p.
/// (A MISSING q does not: the equality rule is exact for any draft — `actor_multi_spec_sampling.rs`
/// — so the first test, not this one, is what catches a dropped q.)
#[tokio::test]
async fn sampled_drafts_in_multi_stream_records_emit_p() {
    spec_on();
    let (mock, rec, go) = QMock::with_blocks(1024);
    let (handle, _m, _join) = spawn(
        Box::new(mock),
        cfg_blocks(1024),
        Arc::new(TokenPieceMap::new(vec![])),
        None,
    );
    let n = 3000;
    let params: Vec<SamplingParams> = (0..4).map(|i| sampled(n, 500 + i, 1.0)).collect();
    let mut rxs = Vec::new();
    for (i, p) in params.iter().enumerate() {
        let id = 60 + i as u64;
        let (j, rx) = job(id, prompt(id), p.clone());
        handle.submit(j).unwrap();
        rxs.push(rx);
    }
    go.store(true, std::sync::atomic::Ordering::Release); // every job is queued
    const BINS: usize = 10;
    let mut hist = [0u64; BINS];
    let mut total = 0u64;
    for (i, rx) in rxs.into_iter().enumerate() {
        let id = 60 + i as u64;
        let (out, fin) = collect(rx).await;
        assert_eq!(fin, Some(FinishReason::Length));
        let pr = prompt(id);
        let mut toks = pr.clone();
        let probe = RowSampler {
            params: params[i].clone(),
            history: Vec::new(),
            draft_q: Vec::new(),
        };
        for &t in &out {
            let pos = toks.len() - 1;
            let mut p = probe.target_dist(&model_logits(pos, toks[pos]), &[toks[pos]], 0);
            p.sort_by_key(|c| c.0);
            let lo: f32 = p.iter().filter(|c| c.0 < t).map(|c| c.1).sum();
            let pt = p
                .iter()
                .find(|c| c.0 == t)
                .map(|c| c.1)
                .unwrap_or_else(|| panic!("stream {id} at {pos}: {t} is outside p's support"));
            let v = keyed_uniform(0xD1CE_0000 + id, pos, 77);
            let u = ((lo + v * pt) as f64).clamp(0.0, 1.0 - 1e-9);
            hist[(u * BINS as f64) as usize] += 1;
            total += 1;
            toks.push(t);
        }
    }
    let e = total as f64 / BINS as f64;
    let chi2: f64 = hist.iter().map(|&o| (o as f64 - e).powi(2) / e).sum();
    eprintln!("PIT over {total} tokens: chi2 {chi2:.2}, bins {hist:?}");
    // 9 degrees of freedom: the 99.9th percentile is 27.88
    assert!(
        chi2 < 27.88,
        "emitted tokens are not distributed as p: chi2 {chi2:.1}, bins {hist:?}"
    );
    // not vacuous: most tokens came through q-bearing records
    let rec = rec.lock().unwrap();
    let with_q: usize = rec
        .segs
        .iter()
        .filter(|g| g.sampler.as_ref().is_some_and(|s| !s.draft_q.is_empty()))
        .map(|g| accepted_run(&g.window, &g.preds).len())
        .sum();
    eprintln!("{with_q} of {total} tokens came through q-bearing segments");
    assert!(
        with_q as u64 * 2 > total,
        "only {with_q} of {total} tokens came through q-bearing segments"
    );
}
