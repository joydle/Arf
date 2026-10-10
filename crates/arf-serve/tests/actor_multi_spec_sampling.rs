//! MULTI-STREAM SPECULATION SERVES SAMPLED STREAMS (2026-09-27). Before it, the multi-stream gate
//! was greedy-only: 2-4 concurrent sampled requests took plain one-token steps together and ran at
//! 9-18 tok/s aggregate (another engine: 39-47) — below one sampled stream alone. Now each sampled
//! segment of the multi-stream record is drawn by its own stream's `RowSampler`.
//!
//! What these pin, through the actor with a mock backend (no GPU):
//! - concurrent sampled streams take the multi-stream record (`verify_segments_sampled`), each
//!   segment flagged sampled, each stream with its own seed;
//! - greedy, sampled and greedy-with-a-penalty streams share one record, each segment treated as
//!   its own request says;
//! - EXACTNESS: every stream emits exactly what the plain sampler draws for its seed at each
//!   position (the lone-stream `RowSampler`'s own draws), with drafts both accepted and rejected.
//!
//! Its own test binary: the actor reads `ARF_SPEC_K` once per process. The opt-out
//! (`ARF_NO_MULTI_SPEC_SAMPLING=1`) is `actor_multi_spec_sampling_off.rs`.

mod common;

use std::sync::Arc;

use arf_core::sampling::{SamplingParams, TokenPieceMap};
use arf_serve::actor::{spawn, FinishReason};
use common::*;

fn spec_on() {
    std::env::set_var("ARF_SPEC_K", "1");
}

/// Two sampled requests decoding together verify in ONE multi-stream record, both segments
/// sampled — and each emits exactly the plain sampler's tokens for its seed.
#[tokio::test]
async fn two_concurrent_sampled_streams_share_one_record_and_draw_what_plain_sampling_draws() {
    spec_on();
    let (mock, rec, go) = MultiMock::new();
    let (handle, _m, _join) = spawn(
        Box::new(mock),
        cfg(),
        Arc::new(TokenPieceMap::new(vec![])),
        None,
    );
    let n = 64;
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
            "stream {i}: the multi-stream record must emit the plain sampler's tokens"
        );
    }
    let rec = rec.lock().unwrap();
    assert!(
        !rec.sampled_records.is_empty(),
        "two sampled streams never took the multi-stream record: {rec:?}"
    );
    assert_eq!(rec.greedy_records, 0, "no record here is all greedy");
    for r in &rec.sampled_records {
        assert_eq!(r.len(), 2, "{r:?}");
        assert!(r.iter().all(|s| s.1), "every segment is sampled: {r:?}");
    }
    assert!(
        rec.sampled_records
            .iter()
            .any(|r| r.iter().all(|s| s.2 > 1)),
        "the drafts must reach the record (multi-row segments)"
    );
    assert!(
        rec.sampled_accepted > 0 && rec.sampled_rejections > 0,
        "both branches of the accept rule must run: {} accepted, {} rejections",
        rec.sampled_accepted,
        rec.sampled_rejections
    );
    for &(stream, seed) in &rec.seeds {
        assert_eq!(
            seed, params[stream as usize].seed,
            "stream {stream} drew with another seed"
        );
    }
}

/// A MIX in one record: a plain greedy stream (argmax), a sampled stream with a repetition penalty
/// (its history reaches the draws), and a greedy stream WITH a penalty (the penalised argmax — a
/// `RowSampler` at temperature 0, as it gets alone). Three streams: the 3 + 3 + 2 shape.
#[tokio::test]
async fn greedy_and_sampled_streams_share_one_record() {
    spec_on();
    let (mock, rec, go) = MultiMock::new();
    let (handle, _m, _join) = spawn(
        Box::new(mock),
        cfg(),
        Arc::new(TokenPieceMap::new(vec![])),
        None,
    );
    let n = 48;
    let params = [
        SamplingParams::greedy(n),
        sampled(n, 21, 1.3),
        SamplingParams {
            repetition_penalty: 1.3,
            repeat_last_n: 64,
            ..SamplingParams::greedy(n)
        },
    ];
    let mut rxs = Vec::new();
    for (i, p) in params.iter().enumerate() {
        let id = 10 + i as u64;
        let (j, rx) = job(id, prompt(id), p.clone());
        handle.submit(j).unwrap();
        rxs.push(rx);
    }
    go.store(true, std::sync::atomic::Ordering::Release); // every job is queued
    for (i, rx) in rxs.into_iter().enumerate() {
        let id = 10 + i as u64;
        let (toks, fin) = collect(rx).await;
        assert_eq!(fin, Some(FinishReason::Length));
        assert_eq!(toks, reference(&prompt(id), &params[i], n), "stream {id}");
    }
    let rec = rec.lock().unwrap();
    assert!(
        rec.sampled_records.iter().any(|r| r.len() == 3),
        "the three streams never shared a record: {rec:?}"
    );
    for r in &rec.sampled_records {
        for &(stream, is_sampled, _) in r {
            assert_eq!(
                is_sampled,
                stream != 10,
                "stream {stream}: only the plain greedy stream keeps the argmax ({r:?})"
            );
        }
    }
}

/// Four sampled streams: the 3 + 3 + 1 + 1 shape, where two streams verify only their pending
/// token (a one-row segment: its draw is the plain step's) and the drafted pair rotates.
#[tokio::test]
async fn four_sampled_streams_including_one_row_segments_are_exact() {
    spec_on();
    let (mock, rec, go) = MultiMock::new();
    let (handle, _m, _join) = spawn(
        Box::new(mock),
        cfg(),
        Arc::new(TokenPieceMap::new(vec![])),
        None,
    );
    let n = 40;
    let params: Vec<SamplingParams> = (0..4).map(|i| sampled(n, 100 + i, 1.0)).collect();
    let mut rxs = Vec::new();
    for (i, p) in params.iter().enumerate() {
        let id = 20 + i as u64;
        let (j, rx) = job(id, prompt(id), p.clone());
        handle.submit(j).unwrap();
        rxs.push(rx);
    }
    go.store(true, std::sync::atomic::Ordering::Release); // every job is queued
    for (i, rx) in rxs.into_iter().enumerate() {
        let id = 20 + i as u64;
        let (toks, fin) = collect(rx).await;
        assert_eq!(fin, Some(FinishReason::Length));
        assert_eq!(toks, reference(&prompt(id), &params[i], n), "stream {id}");
    }
    let rec = rec.lock().unwrap();
    assert!(
        rec.sampled_records
            .iter()
            .any(|r| r.len() == 4 && r.iter().filter(|s| s.2 == 1).count() == 2),
        "four streams never shared a record with two one-row segments: {rec:?}"
    );
}
