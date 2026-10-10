//! The OPT-OUT of multi-stream speculation for sampled streams, `ARF_NO_MULTI_SPEC_SAMPLING=1`
//! (2026-09-27): concurrent sampled streams take the plain step together, as before — no
//! multi-stream record at all — and still emit the plain sampler's tokens; greedy streams keep
//! their multi-stream record. Its own test binary: the actor reads both variables once per process.

mod common;

use std::sync::Arc;

use arf_core::sampling::{SamplingParams, TokenPieceMap};
use arf_serve::actor::{spawn, FinishReason};
use common::*;

fn env() {
    std::env::set_var("ARF_SPEC_K", "1");
    std::env::set_var("ARF_NO_MULTI_SPEC_SAMPLING", "1");
}

#[tokio::test]
async fn opted_out_sampled_streams_take_the_plain_step() {
    env();
    let (mock, rec, go) = MultiMock::new();
    let (handle, _m, _join) = spawn(
        Box::new(mock),
        cfg(),
        Arc::new(TokenPieceMap::new(vec![])),
        None,
    );
    let n = 32;
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
        rec.sampled_records.is_empty() && rec.greedy_records == 0,
        "the opt-out must keep sampled streams off the multi-stream record: {rec:?}"
    );
}

#[tokio::test]
async fn opting_out_leaves_greedy_streams_on_the_multi_stream_record() {
    env();
    let (mock, rec, go) = MultiMock::new();
    let (handle, _m, _join) = spawn(
        Box::new(mock),
        cfg(),
        Arc::new(TokenPieceMap::new(vec![])),
        None,
    );
    let n = 32;
    let mut rxs = Vec::new();
    for id in [30u64, 31] {
        let (j, rx) = job(id, prompt(id), SamplingParams::greedy(n));
        handle.submit(j).unwrap();
        rxs.push((id, rx));
    }
    go.store(true, std::sync::atomic::Ordering::Release); // every job is queued
    for (id, rx) in rxs {
        let (toks, fin) = collect(rx).await;
        assert_eq!(fin, Some(FinishReason::Length));
        assert_eq!(
            toks,
            reference(&prompt(id), &SamplingParams::greedy(n), n),
            "stream {id}"
        );
    }
    let rec = rec.lock().unwrap();
    assert!(
        rec.greedy_records > 0,
        "greedy streams lost their record: {rec:?}"
    );
    assert!(rec.sampled_records.is_empty(), "{rec:?}");
}
