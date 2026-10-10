//! A mock backend for MULTI-STREAM SPECULATION with sampled streams (2026-09-27), shared by the
//! test binaries that need it (each is its own process, because the actor reads its gates once
//! per process — `ARF_SPEC_K`, `ARF_NO_MULTI_SPEC_SAMPLING`).
//!
//! The mock is a tiny deterministic "model": the logits for the token after `tok` at position
//! `pos` are a fixed function of `(pos, tok)`: a spread of 3.0 with one favoured token lifted by
//! 2.0, so a draft of the argmax chain is accepted often but not always at the tests' temperature
//! (the two-stream test: 42 drafts accepted, 75 segments that rejected one). Every
//! path computes those logits the same way — the plain step's `forward_batch`, the block draft's
//! argmax chain, and the verify record's rows — so the reference text is a plain sampler run over
//! the same function ([`reference`]), and any drift in positions, seeds, history or the row map
//! shows up as a different token.
#![allow(dead_code)] // each test binary uses a different subset

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use arf_core::backend::{segment_predictions, BatchedBackend, SegReq};
use arf_core::config::EngineConfig;
use arf_core::model::batch::ForwardBatch;
use arf_core::sampling::{argmax, sample_at_with_history, RowSampler, SamplingParams};
use arf_serve::actor::{FinishReason, Job, StreamItem};

pub const V: usize = 32;
pub const DRAFT: usize = 7;

/// The logits for the token after `tok`, which sits at position `pos`.
pub fn model_logits(pos: usize, tok: u32) -> Vec<f32> {
    let favoured = (pos * 7 + tok as usize * 3) % V;
    (0..V)
        .map(|t| {
            let base = ((pos * 31 + tok as usize * 17 + t * 13) % 23) as f32 / 23.0 * 3.0;
            if t == favoured {
                base + 2.0
            } else {
                base
            }
        })
        .collect()
}

/// What the PLAIN path emits for this prompt: each token drawn by the plain sampler at its
/// absolute position, with the emitted history the repetition penalty reads.
pub fn reference(prompt: &[u32], params: &SamplingParams, n: usize) -> Vec<u32> {
    let mut toks = prompt.to_vec();
    let mut out: Vec<u32> = Vec::new();
    for _ in 0..n {
        let pos = toks.len() - 1;
        let t = sample_at_with_history(&model_logits(pos, toks[pos]), params, pos + 1, &out)
            .expect("reference draw");
        out.push(t);
        toks.push(t);
    }
    out
}

/// What the mock saw.
#[derive(Default, Debug)]
pub struct Rec {
    /// Multi-segment records taken through `verify_segments` (every segment greedy).
    pub greedy_records: usize,
    /// Multi-segment records taken through `verify_segments_sampled`: per record, each segment's
    /// (stream, sampled?, rows).
    pub sampled_records: Vec<Vec<(u64, bool, usize)>>,
    /// Over the sampled records' SAMPLED segments: drafts accepted, and segments that rejected a
    /// draft (both branches of the accept rule must be exercised for exactness to mean anything).
    pub sampled_accepted: usize,
    pub sampled_rejections: usize,
    /// Sampler seeds seen per stream (the actor must hand each stream its own).
    pub seeds: Vec<(u64, u64)>,
}

pub struct MultiMock {
    pub rec: Arc<Mutex<Rec>>,
    /// Held false until the test has submitted every job: the first step waits on it, so all the
    /// jobs are queued before any stream can run ahead (a mock step takes microseconds, and a
    /// stream that finished before the last job arrived would never share a record with it).
    go: Arc<AtomicBool>,
}

impl MultiMock {
    /// The mock, what it saw, and the switch that releases its first step.
    pub fn new() -> (Self, Arc<Mutex<Rec>>, Arc<AtomicBool>) {
        let rec = Arc::new(Mutex::new(Rec::default()));
        let go = Arc::new(AtomicBool::new(false));
        (
            MultiMock {
                rec: Arc::clone(&rec),
                go: Arc::clone(&go),
            },
            rec,
            go,
        )
    }

    /// The record: every segment's rows, laid end to end — (argmax per row, `[rows][V]` logits).
    fn record(segs: &[SegReq<'_>]) -> (Vec<u32>, Vec<f32>) {
        let mut am = Vec::new();
        let mut logits = Vec::new();
        for g in segs {
            for (r, &tok) in g.window.iter().enumerate() {
                let l = model_logits(g.prefix_len + r, tok);
                am.push(argmax(&l));
                logits.extend_from_slice(&l);
            }
        }
        (am, logits)
    }
}

impl BatchedBackend for MultiMock {
    fn forward_batch(&self, ids: &[u32], batch: &ForwardBatch) -> Vec<Vec<f32>> {
        while !self.go.load(Ordering::Acquire) {
            std::thread::yield_now();
        }
        batch
            .seqs
            .iter()
            .map(|s| model_logits(s.past_len + s.q_len - 1, ids[s.q_start + s.q_len - 1]))
            .collect()
    }
    fn kv_geometry(&self) -> (usize, usize) {
        (512, 16)
    }
    /// The block draft: the favoured chain after `tok` (a point mass, like the real selector).
    fn block_draft(&self, tok: u32, past_len: usize, _stream: u64) -> Vec<u32> {
        let (mut pos, mut t) = (past_len, tok);
        (0..DRAFT)
            .map(|_| {
                t = argmax(&model_logits(pos, t));
                pos += 1;
                t
            })
            .collect()
    }
    fn verify_segments(&self, segs: &[SegReq<'_>]) -> Option<Vec<Vec<u32>>> {
        self.rec.lock().unwrap().greedy_records += 1;
        let (am, _) = Self::record(segs);
        Some(segment_predictions(segs, &[], &am, &[], V).expect("greedy segments"))
    }
    fn verify_segments_sampled(
        &self,
        segs: &[SegReq<'_>],
        samplers: &[Option<&RowSampler>],
    ) -> Option<Vec<Vec<u32>>> {
        let (am, logits) = Self::record(segs);
        // the SAME row map and draws the Metal backend runs after its record
        let preds = segment_predictions(segs, samplers, &am, &logits, V).expect("sampled segments");
        let mut rec = self.rec.lock().unwrap();
        rec.sampled_records.push(
            segs.iter()
                .zip(samplers)
                .map(|(g, s)| (g.stream, s.is_some(), g.window.len()))
                .collect(),
        );
        for ((g, s), p) in segs.iter().zip(samplers).zip(&preds) {
            if let Some(s) = s {
                rec.seeds.push((g.stream, s.params.seed));
                let a = arf_core::model::speculative::accepted_run(g.window, p).len() - 1;
                rec.sampled_accepted += a;
                if a + 1 < g.window.len() {
                    rec.sampled_rejections += 1;
                }
            }
        }
        Some(preds)
    }
}

pub fn cfg() -> EngineConfig {
    cfg_blocks(512)
}

/// [`cfg`] with `num_blocks` KV blocks of 16.
pub fn cfg_blocks(num_blocks: usize) -> EngineConfig {
    EngineConfig {
        block_size: 16,
        num_blocks,
        max_batch_size: 8,
        max_prefill_tokens: 512,
        decode_lookahead_tokens: 8,
        ..Default::default()
    }
}

pub fn sampled(max_tokens: usize, seed: u64, penalty: f32) -> SamplingParams {
    SamplingParams {
        max_tokens,
        temperature: 0.7,
        top_p: Some(0.95),
        top_k: Some(20),
        seed,
        repetition_penalty: penalty,
        repeat_last_n: 64,
        ..Default::default()
    }
}

/// A distinct prompt per stream, all the same length (so the streams prefill together and then
/// decode side by side).
pub fn prompt(i: u64) -> Vec<u32> {
    (0..12)
        .map(|t| ((t * 5 + i as usize * 11) % V) as u32)
        .collect()
}

pub fn job(
    id: u64,
    prompt: Vec<u32>,
    params: SamplingParams,
) -> (Job, tokio::sync::mpsc::Receiver<StreamItem>) {
    let (tx, rx) = tokio::sync::mpsc::channel(4096);
    (
        Job {
            id,
            prompt_ids: prompt,
            params,
            token_tx: tx,
            want_logprobs: false,
            top_logprobs: 0,
            response_format: None,
            images: Vec::new(),
            image_positions: Vec::new(),
            image_prompt: None,
            prefix_anchor: None,
            tools_anchor: None,
            header_tail: None,
            background: false,
            stop: None,
        },
        rx,
    )
}

pub async fn collect(
    mut rx: tokio::sync::mpsc::Receiver<StreamItem>,
) -> (Vec<u32>, Option<FinishReason>) {
    let mut toks = Vec::new();
    while let Some(item) = rx.recv().await {
        match item {
            StreamItem::Token(t, _) => toks.push(t),
            StreamItem::Done(r) => return (toks, Some(r)),
        }
    }
    (toks, None)
}

// ------------------------------------------------------------------------------------------------
// SAMPLED DRAFTS PER SEGMENT (2026-09-27): a mock whose block draft can also be SAMPLED, with the
// q it was drawn from — the multi-stream cycle's `block_draft_gpu_sampled_waited`.
// ------------------------------------------------------------------------------------------------

/// The mock draft's temperature: SHARPER than the tests' target (0.7), so q over-weights the
/// favourite and min(1, p/q) rejects it often enough for both branches to run.
pub const DRAFT_T: f32 = 0.5;
/// The mock draft's candidates per position: its q has a smaller support than p (top-k 20).
pub const DRAFT_K: usize = 8;

/// The mock draft's q for the token after `tok` at position `pos`: softmax(logits / DRAFT_T) over
/// the top DRAFT_K ids, in id order (the GPU chain's order).
pub fn draft_q_at(pos: usize, tok: u32) -> Vec<(u32, f32)> {
    let l = model_logits(pos, tok);
    let mut ids: Vec<u32> = (0..V as u32).collect();
    ids.sort_by(|a, b| l[*b as usize].total_cmp(&l[*a as usize]).then(a.cmp(b)));
    ids.truncate(DRAFT_K);
    ids.sort_unstable();
    let m = ids
        .iter()
        .map(|&t| l[t as usize])
        .fold(f32::NEG_INFINITY, f32::max);
    let w: Vec<f32> = ids
        .iter()
        .map(|&t| ((l[t as usize] - m) / DRAFT_T).exp())
        .collect();
    let z: f32 = w.iter().sum();
    ids.iter().zip(&w).map(|(&t, &x)| (t, x / z)).collect()
}

/// A SAMPLED block draft after `tok` (sitting at `past_len`): each position drawn from
/// [`draft_q_at`] with the draft's keyed uniform `keyed_uniform(seed, past_len + i, DRAFT)` — the
/// keying the real selectors use. Returns the drafts and the q each was drawn from.
pub fn sampled_block_draft(
    tok: u32,
    past_len: usize,
    seed: u64,
) -> (Vec<u32>, Vec<Vec<(u32, f32)>>) {
    use arf_core::sampling::{draw_from, keyed_uniform, purpose};
    let (mut pos, mut t) = (past_len, tok);
    let (mut d, mut qs) = (Vec::with_capacity(DRAFT), Vec::with_capacity(DRAFT));
    for i in 1..=DRAFT {
        let q = draft_q_at(pos, t);
        let x = draw_from(&q, keyed_uniform(seed, past_len + i, purpose::DRAFT)).expect("draft q");
        d.push(x);
        qs.push(q);
        t = x;
        pos += 1;
    }
    (d, qs)
}

/// One segment a record served, as the backend saw it.
#[derive(Debug, Clone)]
pub struct SegSeen {
    pub stream: u64,
    pub prefix_len: usize,
    pub window: Vec<u32>,
    /// `None` = a greedy segment (no sampler).
    pub sampler: Option<RowSampler>,
    pub preds: Vec<u32>,
}

/// What the q-bearing mock saw.
#[derive(Default, Debug)]
pub struct QRec {
    /// Every sampled draft handed out: (stream, past_len) -> (seed, drafts, q).
    pub drafts: std::collections::HashMap<(u64, usize), (u64, Vec<u32>, Vec<Vec<(u32, f32)>>)>,
    /// Streams that asked for a sampled draft, in call order.
    pub draft_calls: Vec<u64>,
    /// Point-mass drafts handed out, per stream.
    pub point_mass_calls: Vec<u64>,
    /// Every segment of every multi-segment record, in order.
    pub segs: Vec<SegSeen>,
}

/// [`MultiMock`]'s model with a SAMPLED draft (`block_draft_gpu_sampled_waited`).
pub struct QMock {
    pub rec: Arc<Mutex<QRec>>,
    go: Arc<AtomicBool>,
    /// KV blocks of 16 (the actor checks it against its `EngineConfig`, see [`cfg_blocks`]).
    blocks: usize,
}

impl QMock {
    pub fn new() -> (Self, Arc<Mutex<QRec>>, Arc<AtomicBool>) {
        Self::with_blocks(512)
    }

    /// A mock with `blocks` KV blocks of 16 — for long runs; pair it with [`cfg_blocks`].
    pub fn with_blocks(blocks: usize) -> (Self, Arc<Mutex<QRec>>, Arc<AtomicBool>) {
        let rec = Arc::new(Mutex::new(QRec::default()));
        let go = Arc::new(AtomicBool::new(false));
        (
            QMock {
                rec: Arc::clone(&rec),
                go: Arc::clone(&go),
                blocks,
            },
            rec,
            go,
        )
    }

    fn serve(
        &self,
        segs: &[SegReq<'_>],
        samplers: &[Option<&RowSampler>],
    ) -> Option<Vec<Vec<u32>>> {
        let (am, logits) = MultiMock::record(segs);
        let preds = segment_predictions(segs, samplers, &am, &logits, V).expect("segments");
        let mut rec = self.rec.lock().unwrap();
        for (i, (g, p)) in segs.iter().zip(&preds).enumerate() {
            rec.segs.push(SegSeen {
                stream: g.stream,
                prefix_len: g.prefix_len,
                window: g.window.to_vec(),
                sampler: samplers.get(i).copied().flatten().cloned(),
                preds: p.clone(),
            });
        }
        Some(preds)
    }
}

impl BatchedBackend for QMock {
    fn forward_batch(&self, ids: &[u32], batch: &ForwardBatch) -> Vec<Vec<f32>> {
        while !self.go.load(Ordering::Acquire) {
            std::thread::yield_now();
        }
        batch
            .seqs
            .iter()
            .map(|s| model_logits(s.past_len + s.q_len - 1, ids[s.q_start + s.q_len - 1]))
            .collect()
    }
    fn kv_geometry(&self) -> (usize, usize) {
        (self.blocks, 16)
    }
    fn block_draft(&self, tok: u32, past_len: usize, stream: u64) -> Vec<u32> {
        self.rec.lock().unwrap().point_mass_calls.push(stream);
        let (mut pos, mut t) = (past_len, tok);
        (0..DRAFT)
            .map(|_| {
                t = argmax(&model_logits(pos, t));
                pos += 1;
                t
            })
            .collect()
    }
    fn block_draft_gpu_sampled_waited(
        &self,
        tok: u32,
        past_len: usize,
        stream: u64,
        _temperature: f32,
        seed: u64,
    ) -> Option<(Vec<u32>, Vec<Vec<(u32, f32)>>)> {
        let (d, q) = sampled_block_draft(tok, past_len, seed);
        let mut rec = self.rec.lock().unwrap();
        rec.draft_calls.push(stream);
        rec.drafts
            .insert((stream, past_len), (seed, d.clone(), q.clone()));
        Some((d, q))
    }
    fn verify_segments(&self, segs: &[SegReq<'_>]) -> Option<Vec<Vec<u32>>> {
        self.serve(segs, &[])
    }
    fn verify_segments_sampled(
        &self,
        segs: &[SegReq<'_>],
        samplers: &[Option<&RowSampler>],
    ) -> Option<Vec<Vec<u32>>> {
        self.serve(segs, samplers)
    }
}
