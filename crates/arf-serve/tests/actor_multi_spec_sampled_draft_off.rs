//! The OPT-OUT of sampled drafts, `ARF_NO_SAMPLED_DRAFT=1` (2026-09-27): the one switch that puts
//! the lone sampled stream on the point-mass draft does the same for every sampled segment of a
//! multi-stream record — the sampled draft is never asked for, no segment carries a q, and each
//! stream emits exactly the plain sampler's tokens (point-mass drafts are seed-identical). Its own
//! test binary: the actor reads the variable once per process.

mod common;

use std::sync::Arc;

use arf_core::sampling::TokenPieceMap;
use arf_serve::actor::{spawn, FinishReason};
use common::*;

#[tokio::test]
async fn opted_out_sampled_segments_draft_the_point_mass() {
    std::env::set_var("ARF_SPEC_K", "1");
    std::env::set_var("ARF_NO_SAMPLED_DRAFT", "1");
    let (mock, rec, go) = QMock::new();
    let (handle, _m, _join) = spawn(
        Box::new(mock),
        cfg(),
        Arc::new(TokenPieceMap::new(vec![])),
        None,
    );
    let n = 48;
    let params = [sampled(n, 7, 1.0), sampled(n, 8, 1.0)];
    let mut rxs = Vec::new();
    for (i, p) in params.iter().enumerate() {
        let (j, rx) = job(i as u64, prompt(i as u64), p.clone());
        handle.submit(j).unwrap();
        rxs.push(rx);
    }
    go.store(true, std::sync::atomic::Ordering::Release); // every job is queued
    for (i, rx) in rxs.into_iter().enumerate() {
        let (toks, fin) = collect(rx).await;
        assert_eq!(fin, Some(FinishReason::Length));
        assert_eq!(
            toks,
            reference(&prompt(i as u64), &params[i], n),
            "stream {i}"
        );
    }
    let rec = rec.lock().unwrap();
    assert!(
        rec.draft_calls.is_empty(),
        "the opt-out must never ask for a sampled draft: {:?}",
        rec.draft_calls
    );
    assert!(
        rec.segs.iter().any(|g| g.window.len() > 1),
        "the point-mass drafts must still reach the record"
    );
    assert!(
        rec.segs
            .iter()
            .all(|g| g.sampler.as_ref().is_some_and(|s| s.draft_q.is_empty())),
        "no segment may carry a q under the opt-out"
    );
}
