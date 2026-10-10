//! Public single-stream generation API for [`GpuModel`]: `generate*`,
//! `prefill`, `forward_logits`, streaming, and speculative decode.
//! Split out of the original monolithic gpu.rs; behavior unchanged.

use super::*;

impl GpuModel {
    /// Logits `[vocab]` for the last token of `tokens`, prefilling the KV pool
    /// from scratch. Each token is processed once (O(n)). The CPU `Llama` is the
    /// correctness oracle.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(level = "info", name = "gpu_forward_logits", skip_all, fields(tokens = tokens.len()))
    )]
    pub fn forward_logits(&self, tokens: &[u32]) -> Vec<f32> {
        assert!(
            tokens.len() <= self.max_ctx,
            "context {} exceeds the KV pool capacity {} (load with a larger max_positions)",
            tokens.len(),
            self.max_ctx
        );
        let last = tokens.len() - 1;
        for (pos, &token) in tokens.iter().enumerate() {
            // Logits-only path: never samples (the caller reads logits back), so
            // greedy (None) regardless of any request temperature.
            self.decode_token(Some(token), pos, pos == last, None, None);
        }
        self.ctx.read_f32(&self.scratch.logits, self.cfg.vocab_size)
    }

    /// Greedily generate `max_tokens` tokens from `prompt` on the GPU.
    ///
    /// Prefills the prompt's KV once, then decodes incrementally: each new token
    /// computes only its own K/V and attends over the resident context — O(n)
    /// total, not O(n²). Returns the generated tokens.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(
            level = "info",
            name = "gpu_generate",
            skip_all,
            fields(prompt = prompt.len(), max_tokens = max_tokens)
        )
    )]
    pub fn generate(&self, prompt: &[u32], max_tokens: usize) -> Vec<u32> {
        self.generate_stop(prompt, max_tokens, &[])
    }

    /// Like [`Self::generate`] but halts as soon as a `stop` token (e.g. Gemma's
    /// `<end_of_turn>`, or `<eos>`) is sampled — the stop token is NOT included in
    /// the output, matching the engine/scheduler's stop semantics and what a real
    /// serving loop does.
    ///
    /// With `stop` empty this is the fully-fused fast path: every decode step is
    /// queued on the GPU and the tokens are drained in ONE readback at the end (the
    /// on-GPU feedback loop, no per-token CPU↔GPU stall). With stop tokens, each
    /// emitted token is read back before the next step so generation can end early
    /// — the honest cost of EOS detection (one 4-byte readback per token).
    pub fn generate_stop(&self, prompt: &[u32], max_tokens: usize, stop: &[u32]) -> Vec<u32> {
        self.generate_stop_sampled(prompt, max_tokens, stop, None)
    }

    /// Like [`Self::generate_stop`] but with an explicit on-GPU sampling spec.
    /// `sampling == None` is greedy (the fully-fused fast path); `Some(spec)`
    /// runs the on-GPU stochastic kernel (temperature + top-k + top-p + a
    /// deterministic (seed, position) RNG) — still writing `next_token` on-GPU,
    /// so the fused feedback loop and single-4-byte readback design are intact.
    /// L321 — REFUSE A HYBRID RATHER THAN EMIT GARBAGE. This path's decode
    /// (`decode_token`) has no gated-delta-net branch: it runs attention over the 48
    /// recurrent layers, which produces fluent nonsense instead of an error. The server
    /// path is unaffected — `forward_batch_impl` routes hybrids through
    /// `try_gdn_serial_prefill` + the batched megakernel (batch.rs:3109), which is why
    /// every measurement and the parity gates were correct while this was broken.
    ///
    /// Wiring the hybrid decode into this loop is real work (the recurrent bank, its
    /// per-sequence row, and the reset at a sequence boundary all have to come with it).
    /// Until then a clear refusal that names the working command beats output a user
    /// would have to be an expert to recognise as wrong.
    pub fn generate_stop_sampled(
        &self,
        prompt: &[u32],
        max_tokens: usize,
        stop: &[u32],
        sampling: Option<GpuSampling>,
    ) -> Vec<u32> {
        assert!(!prompt.is_empty(), "empty prompt");
        // L322 — HYBRIDS TAKE THEIR OWN LOOP. `decode_token` has no gated-delta-net
        // branch: it would run attention over the 48 recurrent layers and emit fluent
        // nonsense. `generate_hybrid` below drives the same one-token-at-a-time
        // megakernel step the server uses, which is the only shape a recurrent state
        // admits — the state IS the history, so every token must advance it in order.
        if self.layers.iter().any(|l| l.gdn.is_some()) {
            // L364 — this fn returns tokens, not a Result, so off macOS it panics with a clear
            // message rather than returning an empty Vec that reads as "the model said nothing".
            #[cfg(not(target_os = "macos"))]
            panic!(
                "hybrid (GDN) generation needs the native-Metal island, which is macOS-only \
                 (L364). Use the server path, or a non-hybrid model, on this platform."
            );
            #[cfg(target_os = "macos")]
            return self.generate_hybrid(prompt, max_tokens, stop, sampling);
        }
        assert!(
            prompt.len() + max_tokens <= self.max_ctx,
            "context {} exceeds the KV pool capacity {} (load with a larger max_positions)",
            prompt.len() + max_tokens,
            self.max_ctx
        );

        self.prefill(prompt, sampling);

        // Decode: each step embeds the id in next_token (on-GPU feedback,
        // token == None), stashes it into out_tokens[i], runs the layers, and
        // samples the next id back into next_token. The token never leaves the
        // GPU and nothing is read back mid-loop — the N submits queue on the GPU
        // while the CPU keeps recording, and we drain exactly once at the end.
        // This is the win: the old per-token read_u32 forced a full pipeline
        // flush + CPU↔GPU stall every step.
        if stop.is_empty() {
            // MULTI-TOKEN-PER-CMDBUF (ARF_MEGA_KTOK=K): on the greedy megakernel path, batch K
            // consecutive decode tokens into ONE island command buffer (1 commit / K tokens),
            // killing the per-buffer ~20ms inter-buffer GPU idle gap. The K sub-tokens chain on-
            // GPU through next_token (each embed reads the prior argmax's output, ordered by the
            // island's barriers) → byte-identical greedy output. K=1 (unset) = the original path.
            #[cfg(target_os = "macos")]
            let ktok: usize = if sampling.is_none()
                && crate::weights::megakernel_on()
                && std::env::var_os("ARF_MEGA_SINGLEQ").is_some()
            {
                std::env::var("ARF_MEGA_KTOK")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(1usize)
                    .max(1)
            } else {
                1
            };
            #[cfg(not(target_os = "macos"))]
            let ktok: usize = 1;
            if ktok > 1 {
                #[cfg(target_os = "macos")]
                {
                    let mut i = 0usize;
                    while i < max_tokens {
                        let k = ktok.min(max_tokens - i);
                        // decode_tokens_k records k sub-tokens at positions prompt.len()+i .. +k-1
                        // into one cmd buffer, storing out_tokens[(i+1)..(i+k+1)] (the +1 off-by-
                        // one: the island argmax produces the NEXT token). collect_slot = i (the
                        // base is collect_slot+1 = i+1 inside the megakernel).
                        let did = self.decode_tokens_k(prompt.len() + i, i as u64, k);
                        i += did;
                    }
                }
            } else {
                for i in 0..max_tokens {
                    self.decode_token(None, prompt.len() + i, true, Some(i as u64), sampling);
                }
            }
            // NOWAIT pipeline: the last MEGA_DEPTH tokens may still be in flight on the island
            // queue — drain it before reading out_tokens (a wgpu poll won't cover the island).
            #[cfg(target_os = "macos")]
            if let Some(isl) = self.island.as_ref() {
                isl.lock().unwrap().drain_pipeline();
            }
            let toks = self.ctx.read_u32(&self.scratch.out_tokens, max_tokens);
            if std::env::var_os("ARF_TOKDUMP").is_some() {
                eprintln!("[tokdump ktok={ktok}] {toks:?}");
            }
            #[cfg(target_os = "macos")]
            if std::env::var_os("ARF_KVPOOL_DUMP").is_some() {
                // Per-position L0 K checksum (fully drained) — diff vs K=1 to localize whether
                // kt>=1 corrupts KV (early-position drift = bug) vs pure post-flip cascade.
                let kvd = self.layers[0].kv_dim();
                for p in 0..(prompt.len() + max_tokens) {
                    let kr = self
                        .ctx
                        .read_f32_at(&self.kv.keys[0], p as u64 * kvd as u64 * 4, 4);
                    eprintln!(
                        "[kvpool ktok={ktok} L0 p={p}] K4={:?}",
                        &kr[..4.min(kr.len())]
                    );
                }
            }
            return toks;
        }

        // EOS-aware path: next_token currently holds the token to emit this step.
        // Read it back, stop if it's a stop token (not emitted), else record it and
        // advance one step (embed it as feedback + sample the successor). Same token
        // stream as the fused path, just ended early at a stop token — at the honest
        // cost of one 4-byte readback per token (no fused on-GPU drain).
        let mut out = Vec::with_capacity(max_tokens);
        for i in 0..max_tokens {
            let tok = self.ctx.read_u32(&self.scratch.next_token, 1)[0];
            if stop.contains(&tok) {
                break;
            }
            out.push(tok);
            if i + 1 < max_tokens {
                self.decode_token(None, prompt.len() + i, true, None, sampling);
            }
        }
        out
    }

    /// Prefill the prompt's K/V into the resident pool and seed `next_token` with
    /// the first generated id. Shared by [`Self::generate_stop`] and
    /// [`Self::generate_stream`] so both enter their decode loop in the same state.
    ///
    /// Batches all but the last token through ONE `forward_batch` (weights streamed
    /// once for the whole prefix — measured 1.36× Q4 / 2.10× int8 on TTFT, P20),
    /// then runs the LAST prompt token through the single-stream decode step so its
    /// sample seeds `next_token` on-GPU — the decode loop's starting feedback.
    /// `forward_batch` and `decode_token` write the SAME resident KV slots, so the
    /// last step and the decode loop read a consistent pool. A 1-token prompt skips
    /// the batch (nothing to prefix). MoE still has no batched-prefill kernel (the
    /// fused experts have no m>1 path), so it prefills token-by-token. GEMMA NOW USES
    /// the batched prefill: `forward_batch` was made gemma-correct (the GeGLU-vs-SiLU
    /// fix + per-layer global/sliding geometry), validated token-identical to the
    /// single-stream oracle (gemma4_batched_parity + 57/57 real-weight trajectories).
    /// This is the long-prompt TTFT fix: a 4096-token prompt was 4096 sequential
    /// ~47ms `decode_token` forwards (~95s); batched prefill streams the weights once
    /// over the whole prefix instead. All paths write the same KV slots.
    ///
    /// `sampling` is applied to the LAST prompt token only (the one whose sample
    /// becomes the first generated id); the prefix tokens never sample
    /// (`want_logits == false`), so it is irrelevant there. `None` = greedy.
    /// L322 — generation for a HYBRID (gated delta net + attention) model.
    ///
    /// The fused fast path this file is built around cannot serve one: it queues N decode
    /// steps on the GPU and drains once, and `decode_token` has no recurrent branch. A
    /// recurrent layer's state is the history — every token must advance it, in order —
    /// so the only correct shape is one megakernel step per token, prompt included.
    ///
    /// That is exactly what the server does (`try_gdn_serial_prefill` + the batched
    /// megakernel, batch.rs), and this reuses the same primitive rather than growing a
    /// second implementation of it. Slower than the fused path by design; correct, which
    /// the fused path was not.
    /// L364 — macOS-only. Every token here goes through the batched megakernel (a recurrent
    /// layer's state must advance one step per token, in order), and that record is the island.
    /// Off macOS the caller below returns a clear error instead of silently running a path that
    /// would advance the recurrent state wrongly.
    #[cfg(target_os = "macos")]
    fn generate_hybrid(
        &self,
        prompt: &[u32],
        max_tokens: usize,
        stop: &[u32],
        sampling: Option<GpuSampling>,
    ) -> Vec<u32> {
        use arf_core::cache::paged::WriteRun;
        use arf_core::model::batch::{ForwardBatch, SeqAttn};

        let _ = sampling; // greedy only on this path; the sampled variant is server-side.
        let bs = self.kv.block_size;
        let total = prompt.len() + max_tokens;
        // One contiguous slot table for the whole generation: block b holds positions
        // [b*bs, (b+1)*bs). The pool is ours alone here (single in-process sequence).
        let slots: Vec<u32> = (0..total.div_ceil(bs) * bs).map(|i| i as u32).collect();

        // `stream_id` pins the recurrent bank row for this sequence's lifetime. A fixed id
        // is right: there is exactly one sequence in a `arf generate` run.
        let step_for = |pos: usize, tok: u32| -> (Vec<u32>, ForwardBatch) {
            let mut s = slots.clone();
            s.truncate(pos + 1);
            (
                vec![tok],
                ForwardBatch {
                    positions: vec![pos as u32],
                    seqs: vec![SeqAttn {
                        stream_id: Some(0),
                        q_start: 0,
                        q_len: 1,
                        past_len: pos,
                        slots: s,
                        write_runs: vec![WriteRun {
                            src_off: 0,
                            phys_start: slots[pos] as usize,
                            count: 1,
                        }],
                        image_spans: Vec::new(),
                    }],
                    image_embeds: None,
                    mrope_positions: None,
                },
            )
        };

        let mut out: Vec<u32> = Vec::with_capacity(max_tokens);
        let mut next: Option<u32> = None;

        // Prompt, then generation — the same step either way, which is the point.
        for (i, &tok) in prompt.iter().enumerate() {
            let (ids, batch) = step_for(i, tok);
            match self.try_batched_megakernel(&ids, &batch) {
                Some(v) if !v.is_empty() => next = Some(v[0]),
                // A refusal mid-prompt leaves the recurrent state half-advanced and there is
                // no recovery: fail loudly rather than return text computed from a dirty state.
                _ => panic!(
                    "hybrid prefill step {i} refused — recurrent state is now half-advanced; \
                     this is a bug, not a fallback condition"
                ),
            }
        }

        for i in 0..max_tokens {
            let Some(tok) = next else { break };
            if stop.contains(&tok) {
                break;
            }
            out.push(tok);
            let pos = prompt.len() + i;
            if pos + 1 >= total {
                break;
            }
            let (ids, batch) = step_for(pos, tok);
            match self.try_batched_megakernel(&ids, &batch) {
                Some(v) if !v.is_empty() => next = Some(v[0]),
                _ => break,
            }
        }
        out
    }

    pub(super) fn prefill(&self, prompt: &[u32], sampling: Option<GpuSampling>) {
        // ARF_PREFILL_PROF=1 — where does a chat's FIRST TURN actually spend its time?
        // (turn 1 = 12-14 tok/s, later turns = 50-51.)
        let _t_pf = std::time::Instant::now();
        let pf_prof = std::env::var_os("ARF_PREFILL_PROF").is_some();
        let _guard = scopeguard_prefill(pf_prof, _t_pf, prompt.len());
        let is_moe = matches!(self.layers.first().map(|l| &l.mlp), Some(GpuMlp::Moe(_)));
        // MoE PREFILL — the first-turn tax. This used to be unconditional
        // token-by-token ("Only MoE still needs token-by-token prefill (no batched expert
        // kernel)"), i.e. ONE FULL FORWARD PASS PER PROMPT TOKEN: a 20-token prompt = 20 passes
        // through 48 layers. MEASURED 2026-08-03 on real chat: turn 1 = 12-14 tok/s while every
        // later turn = 50-51, and a one-shot CLI run reads 1 tok/s because that prefill dominates
        // the whole wall.
        //
        // That comment is STALE: the batched MoE expert kernels it says do not exist have been
        // shipping for weeks — they are what produce 441 tok/s at conc64 (sort-by-expert mm_id +
        // batched lm_head). So MoE can take the SAME chunked `forward_batch` path gemma takes.
        // ⚠️ MEASURED 2026-08-03 — DO NOT "MODERNISE" THIS AGAIN WITHOUT AN A/B.
        // I switched MoE to the chunked forward_batch path on the theory that the comment above
        // was stale (the batched expert kernels DO exist and drive 441 tok/s at conc64). The A/B
        // at a 50-token prompt says the opposite, decisively:
        //     token-by-token :    275.7 ms      <- 85x FASTER
        //     batched        : 23,549.9 ms
        // The batched MoE kernels are tuned for MANY SEQUENCES x 1 token (expert weight reuse
        // across the batch). Prefill is ONE sequence x many tokens: each row routes to its own
        // top-k experts, so the batched path re-streams expert weights per row and wins nothing,
        // while the m=1 decode path amortises them. Hence token-by-token stays the default for
        // MoE. ARF_MOE_BATCHED_PREFILL=1 opts INTO the (slower) batched path for A/B work.
        // L321 — A HYBRID MUST PREFILL SERIALLY, ALWAYS. The 48 gated-delta-net layers carry
        // recurrent state that advances one token at a time: there is no slot table to fix up
        // afterwards, the state IS the history. The batched branch below runs ATTENTION over
        // those layers instead, which produces fluent garbage rather than an error — exactly
        // what `try_gdn_serial_prefill` exists to prevent on the server path (batch.rs:3109,
        // L143). This path never consulted it, so `arf generate` on Qwen3.8-27B emitted
        // gibberish that got worse with prompt length while `arf serve` was correct.
        let is_hybrid = self.layers.iter().any(|l| l.gdn.is_some());
        let token_by_token =
            is_hybrid || (is_moe && std::env::var_os("ARF_MOE_BATCHED_PREFILL").is_none());
        let last = prompt.len() - 1;
        if last > 0 && token_by_token {
            for (p, &tok) in prompt[..last].iter().enumerate() {
                self.decode_token(Some(tok), p, false, None, None);
            }
        } else if last > 0 {
            use arf_core::cache::{slots_for, write_runs};
            use arf_core::model::batch::{ForwardBatch, SeqAttn};
            let bs = self.kv.block_size;
            let tbl: Vec<u32> = (0..self.kv.num_blocks as u32).collect();
            // Chunk the prefix: one giant forward_batch over thousands of query rows
            // overflows a per-row dispatch grid (wgpu caps each grid dimension at
            // 65535) AND balloons the single command buffer. Process the prefix in
            // ≤PREFILL_CHUNK-token segments, each attending all prior chunks via
            // `past_len` — the KV the earlier chunks wrote. Bounds grids + cmd buffers
            // and keeps the weight-amortized batched prefill (vs token-by-token).
            // Chunk size capped so the largest per-row dispatch grid (≈ q_len × q_dim
            // / 256) stays under wgpu's 65535 per-dimension limit — q_len 2048 already
            // overflows (gemma global q_dim). 1024 is safe and measured equal to 512
            // (prefill is not weight-amortization-bound across the chunk, so a bigger
            // chunk doesn't speed it up — deeper prefill-throughput work is separate).
            // ARF_PREFILL_CHUNK overrides for tuning.
            let prefill_chunk: usize = std::env::var("ARF_PREFILL_CHUNK")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1024);
            let mut past = 0usize;
            while past < last {
                let q_len = (last - past).min(prefill_chunk);
                let ctx = past + q_len;
                let batch = ForwardBatch {
                    positions: (past as u32..ctx as u32).collect(),
                    seqs: vec![SeqAttn {
                        stream_id: None,
                        q_start: 0,
                        q_len,
                        past_len: past,
                        // `slots`/`row_meta` index by `past_len + r`, so cover the full
                        // prefix so far; `write_runs` writes only this chunk's q_len rows.
                        slots: slots_for(&tbl, bs, ctx),
                        write_runs: write_runs(&tbl, bs, past, q_len),
                        image_spans: Vec::new(),
                    }],
                    image_embeds: None,
                    mrope_positions: None,
                };
                // Discard the logits — we only need the KV it wrote into the pool.
                let _ = self.forward_batch(&prompt[past..ctx], &batch);
                past = ctx;
            }
        }
        // Last prompt token: single-stream, want_logits → samples the first
        // generated id into next_token (the decode loop's starting feedback).
        self.decode_token(Some(prompt[last]), last, true, None, sampling);
    }

    /// Streaming decode: like [`Self::generate_stop`], but invokes `on_token`
    /// with each generated id AS IT IS PRODUCED (before the next decode step) rather
    /// than returning the whole vector at the end — the hook a serving loop needs to
    /// push tokens to a client incrementally. Halts at a `stop` token (not emitted)
    /// or after `max_tokens`. If `on_token` returns `false` the loop stops early
    /// (e.g. the client disconnected); the model stays valid for the next request.
    ///
    /// Uses the same per-token readback structure as `generate_stop`'s EOS-aware
    /// branch: `next_token` holds the id to emit this step, so we read it back
    /// (one 4-byte readback/token — the honest cost of incremental emission), hand
    /// it to the callback, then advance one decode step.
    ///
    /// `sampling == None` is greedy (on-GPU argmax into `next_token`); `Some(spec)`
    /// runs the on-GPU stochastic kernel (temperature + top-k + top-p + a
    /// deterministic (seed, position) RNG), still writing `next_token` on-GPU — so
    /// streaming temperature requests no longer fall back to greedy.
    pub fn generate_stream<F: FnMut(u32) -> bool>(
        &self,
        prompt: &[u32],
        max_tokens: usize,
        stop: &[u32],
        sampling: Option<GpuSampling>,
        mut on_token: F,
    ) {
        assert!(!prompt.is_empty(), "empty prompt");
        assert!(
            prompt.len() + max_tokens <= self.max_ctx,
            "context {} exceeds the KV pool capacity {} (load with a larger max_positions)",
            prompt.len() + max_tokens,
            self.max_ctx
        );

        self.prefill(prompt, sampling);

        for i in 0..max_tokens {
            // next_token holds the id to emit this step (prefill/the previous step
            // sampled it there on-GPU). Read it back, stop if it's a stop token
            // (not emitted), else emit it via the callback and advance one step.
            let tok = self.ctx.read_u32(&self.scratch.next_token, 1)[0];
            if stop.contains(&tok) {
                break;
            }
            if !on_token(tok) {
                break;
            }
            if i + 1 < max_tokens {
                self.decode_token(None, prompt.len() + i, true, None, sampling);
            }
        }
    }

    /// Speculative decode on the GPU (P19): an n-gram drafter proposes `k` tokens
    /// from the running history; the resident model VERIFIES the `k`-token window
    /// via the SINGLE-STREAM decode path (one cheap m==1 forward per window token,
    /// which Q4 supports — unlike the heavy batched forward), accepts the longest
    /// correct prefix, and advances. The output is **bit-identical to greedy**
    /// [`Self::generate`] — every accepted draft token equals what greedy would
    /// have produced — a pure latency win, gated by acceptance making the window's
    /// `1+k` cheap forwards emit more than `1+k` tokens of progress on average.
    ///
    /// Stacks on the weight quantization (Q4/int8): the verify reuses the resident
    /// decode kernels + P14/P15 (merged pass, cached binds), so the per-token
    /// bandwidth win compounds. Returns `(tokens, forward_passes, drafted,
    /// accepted)` so the caller can report the speedup and accept rate.
    pub fn generate_speculative(
        &self,
        prompt: &[u32],
        max_tokens: usize,
        k: usize,
        order: usize,
    ) -> (Vec<u32>, usize, usize, usize) {
        use arf_core::model::speculative::NgramDrafter;

        assert!(!prompt.is_empty(), "empty prompt");
        assert!(k >= 1, "k must be >= 1");
        let cap = prompt.len() + max_tokens + k + 1;
        assert!(
            cap <= self.max_ctx,
            "spec-decode context {cap} exceeds the KV pool capacity {} (load with a larger max_positions)",
            self.max_ctx
        );

        // Verify `window` at logical `past_len`, returning per-position greedy argmax.
        // BATCHED (ARF_SPEC_BATCHED, default ON): ONE forward_batch_all over the whole
        // window — weights streamed ONCE (the coop batched path = our advantage; coop_batch_bench
        // N=4 sustains 2.3× agg throughput, so a k=4 verify costs ~1.75 single-decodes, not 4).
        // This is what flips spec-dec from the old 0.55× (sequential m==1 = k full forwards/
        // round, no amortization) to a win. Sequential path kept for A/B (ARF_SPEC_SEQ).
        use arf_core::cache::{slots_for, write_runs};
        use arf_core::model::batch::{ForwardBatch, SeqAttn};
        let bs = self.kv.block_size;
        let tbl: Vec<u32> = (0..self.kv.num_blocks as u32).collect();
        let seq_verify = std::env::var_os("ARF_SPEC_SEQ").is_some();
        let run = |window: &[u32], past_len: usize| -> Vec<u32> {
            if seq_verify {
                for (i, &tok) in window.iter().enumerate() {
                    self.decode_token_impl(super::decode::DecodeStep {
                        token: Some(tok),
                        pos: past_len + i,
                        want_logits: true,
                        collect_after: Some(i as u64),
                        k: 1,
                        ..Default::default()
                    });
                }
                return self.ctx.read_u32(&self.scratch.out_tokens, window.len());
            }
            let ctxn = past_len + window.len();
            // SHARED-PREFIX VERIFY MEGAKERNEL (MTP / spec-decode lever, DEFAULT-ON): stage the shared
            // prefix [0, past_len) ONCE and let all k=window rows consume it (k×→1× prefix reuse) —
            // the ~3.3×→~2× verify-cost cut that flips MTP from a loss to a win at low conc. Returns
            // the k greedy argmax tokens directly. Falls back to the WGSL forward_batch_all oracle
            // (below) when the island can't run it or ARF_NO_VERIFY_MEGA is set — the parity oracle.
            #[cfg(target_os = "macos")]
            {
                let prefix_slots = slots_for(&tbl, bs, ctxn); // [0, past_len+k) progressive table
                if let Some(toks) =
                    self.try_batched_megakernel_verify(window, past_len, &prefix_slots, None)
                {
                    return toks;
                }
            }
            let batch = ForwardBatch {
                positions: (past_len as u32..ctxn as u32).collect(),
                seqs: vec![SeqAttn {
                    stream_id: None,
                    q_start: 0,
                    q_len: window.len(),
                    past_len,
                    slots: slots_for(&tbl, bs, ctxn),
                    write_runs: write_runs(&tbl, bs, past_len, window.len()),
                    image_spans: Vec::new(),
                }],
                image_embeds: None,
                mrope_positions: None,
            };
            let all = self.forward_batch_all(window, &batch);
            all.iter()
                .map(|logits| {
                    logits
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .map(|(i, _)| i as u32)
                        .unwrap_or(0)
                })
                .collect()
        };

        let mut drafter = NgramDrafter::new(order);
        let mut history: Vec<u32> = prompt.to_vec();
        let mut out: Vec<u32> = Vec::with_capacity(max_tokens);
        let (mut forward_passes, mut drafted, mut accepted_total) = (0usize, 0usize, 0usize);

        // Prefill: the prompt's last-position argmax is the first generated token.
        let preds = run(prompt, 0);
        forward_passes += 1;
        let mut committed = *preds.last().unwrap();
        let mut cache_len = prompt.len();

        while out.len() < max_tokens {
            out.push(committed);
            history.push(committed);
            drafter.observe(&history[..history.len() - 1], committed);
            if out.len() >= max_tokens {
                break;
            }

            // Draft, then verify [committed, d1..dj] in one batched forward.
            let draft = drafter.propose(&history, k.min(max_tokens - out.len()));
            drafted += draft.len();
            let mut window = Vec::with_capacity(1 + draft.len());
            window.push(committed);
            window.extend_from_slice(&draft);

            let preds = run(&window, cache_len);
            forward_passes += 1;

            // Accept draft[i] iff it equals the model's greedy preds[i]; stop at
            // the first miss. preds[accepted] is the genuinely-new next token.
            let mut accepted = 0usize;
            while accepted < draft.len() && draft[accepted] == preds[accepted] {
                accepted += 1;
            }
            accepted_total += accepted;

            for &d in &draft[..accepted] {
                out.push(d);
                history.push(d);
                drafter.observe(&history[..history.len() - 1], d);
                if out.len() >= max_tokens {
                    break;
                }
            }
            committed = preds[accepted];
            cache_len += 1 + accepted;
        }

        out.truncate(max_tokens);
        (out, forward_passes, drafted, accepted_total)
    }

    /// Greedily decode `n` sequences concurrently through the batched forward —
    /// the continuous-batching path. Each sequence gets a disjoint block range in
    /// the resident KV pool; every step builds one [`arf_core::model::ForwardBatch`] of all live
    /// sequences (one token each) and runs them in a single batched trunk pass,
    /// so the weight stream is read ONCE for the whole batch instead of once per
    /// sequence. Returns each sequence's generated tokens.
    ///
    /// This is where a from-scratch engine can exceed a tuned single-stream one:
    /// aggregate tok/s scales with the batch because decode is bandwidth-bound on
    /// the shared weights. Greedy via CPU argmax on the per-seq logits (simple and
    /// correct for the throughput measurement; on-GPU batched argmax is future work).
    pub fn generate_batch(&self, prompts: &[Vec<u32>], max_tokens: usize) -> Vec<Vec<u32>> {
        use arf_core::cache::{slots_for, write_runs};
        use arf_core::model::batch::{ForwardBatch, SeqAttn};

        let bs = self.kv.block_size;
        let n = prompts.len();
        assert!(n > 0, "empty batch");

        // Disjoint block range per sequence: sequence s owns a contiguous run of
        // `blocks_per_seq` blocks, sized for prompt + all generated tokens.
        let max_ctx = prompts.iter().map(|p| p.len()).max().unwrap() + max_tokens;
        let blocks_per_seq = max_ctx.div_ceil(bs);
        assert!(
            n * blocks_per_seq <= self.kv.num_blocks,
            "batch needs {} KV blocks but the pool has {}",
            n * blocks_per_seq,
            self.kv.num_blocks
        );
        let block_tables: Vec<Vec<u32>> = (0..n)
            .map(|s| {
                let base = (s * blocks_per_seq) as u32;
                (0..blocks_per_seq as u32).map(|b| base + b).collect()
            })
            .collect();

        // Prefill each sequence's prompt KV via the fast one-shot batched `forward_batch`.
        // (Gemma's batched path is now correct: forward_batch picks the gemma GeGLU
        // activation by `gate_act` — a prior hardcoded SiLU was THE batched-gemma bug.
        // MoE still has no correct batched-prefill kernel, so it stays token-by-token.)
        let is_moe = matches!(self.layers.first().map(|l| &l.mlp), Some(GpuMlp::Moe(_)));
        let prefill_token_by_token = is_moe;
        let mut next: Vec<u32> = prompts
            .iter()
            .enumerate()
            .map(|(s, prompt)| {
                let len = prompt.len();
                if prefill_token_by_token {
                    // Single-stream prefill into seq s's block range. decode_token reads
                    // `self.decode_block_table` for its slot mapping; set it for this seq.
                    *self.decode_block_table.borrow_mut() = Some(block_tables[s].clone());
                    let last = len - 1;
                    for (p, &tok) in prompt[..last].iter().enumerate() {
                        self.decode_token(Some(tok), p, false, None, None);
                    }
                    // Last token: want_logits → its argmax is seq s's first new token.
                    self.decode_token(Some(prompt[last]), last, true, None, None);
                    let logits = self.ctx.read_f32(&self.scratch.logits, self.cfg.vocab_size);
                    *self.decode_block_table.borrow_mut() = None;
                    arf_core::sampling::argmax(&logits)
                } else {
                    let batch = ForwardBatch {
                        positions: (0..len as u32).collect(),
                        seqs: vec![SeqAttn {
                            stream_id: None,
                            q_start: 0,
                            q_len: len,
                            past_len: 0,
                            slots: slots_for(&block_tables[s], bs, len),
                            write_runs: write_runs(&block_tables[s], bs, 0, len),
                            image_spans: Vec::new(),
                        }],
                        image_embeds: None,
                        mrope_positions: None,
                    };
                    let logits = self.forward_batch(prompt, &batch);
                    arf_core::sampling::argmax(&logits[0])
                }
            })
            .collect();

        let mut out: Vec<Vec<u32>> = vec![Vec::with_capacity(max_tokens); n];
        let mut past: Vec<usize> = prompts.iter().map(|p| p.len()).collect();

        for _ in 0..max_tokens {
            for s in 0..n {
                out[s].push(next[s]);
            }
            if prefill_token_by_token {
                // Gemma/MoE: decode each sequence through the single-stream
                // `decode_token` (the correct four-norm path), one token each into its
                // own block range. forward_batch's m>1 batched decode diverges for
                // Gemma the same way its prefill does (reassociation flips greedy
                // argmax under softcap). Serial-per-seq within a step — correctness
                // over the batched-decode throughput, which only Llama/Qwen get.
                for s in 0..n {
                    *self.decode_block_table.borrow_mut() = Some(block_tables[s].clone());
                    self.decode_token(Some(next[s]), past[s], true, None, None);
                    let logits = self.ctx.read_f32(&self.scratch.logits, self.cfg.vocab_size);
                    next[s] = arf_core::sampling::argmax(&logits);
                    past[s] += 1;
                }
                *self.decode_block_table.borrow_mut() = None;
            } else {
                // One batched step: all n sequences, one token each (Llama/Qwen).
                let mut input_ids = Vec::with_capacity(n);
                let mut positions = Vec::with_capacity(n);
                let mut seqs = Vec::with_capacity(n);
                for s in 0..n {
                    input_ids.push(next[s]);
                    positions.push(past[s] as u32);
                    let ctx = past[s] + 1;
                    seqs.push(SeqAttn {
                        // L162 — generate_batch runs n INDEPENDENT sequences, one row each; `s`
                        // is that sequence's index and is stable for the whole call, so it is a
                        // valid stream identity here.
                        stream_id: Some(s as u64),
                        q_start: s,
                        q_len: 1,
                        past_len: past[s],
                        slots: slots_for(&block_tables[s], bs, ctx),
                        write_runs: write_runs(&block_tables[s], bs, past[s], 1),
                        image_spans: Vec::new(),
                    });
                }
                let batch = ForwardBatch {
                    positions,
                    seqs,
                    image_embeds: None,
                    mrope_positions: None,
                };
                let logits = self.forward_batch(&input_ids, &batch);
                for s in 0..n {
                    next[s] = arf_core::sampling::argmax(&logits[s]);
                    past[s] += 1;
                }
            }
        }
        out
    }
}

/// Prints prefill wall time on drop when ARF_PREFILL_PROF is set (diagnostics).
fn scopeguard_prefill(on: bool, t0: std::time::Instant, n: usize) -> impl Drop {
    struct G(bool, std::time::Instant, usize);
    impl Drop for G {
        fn drop(&mut self) {
            if self.0 {
                eprintln!(
                    "[prefill] {} prompt tokens took {:.1}ms",
                    self.2,
                    self.1.elapsed().as_secs_f64() * 1000.0
                );
            }
        }
    }
    G(on, t0, n)
}

/// Whole-`forward_batch_impl` timer for the daemon path (diagnostics).
pub(crate) fn scopeguard_fb(on: bool, t0: std::time::Instant, b: usize, q_len: usize) -> impl Drop {
    struct G(bool, std::time::Instant, usize, usize);
    impl Drop for G {
        fn drop(&mut self) {
            if self.0 {
                eprintln!(
                    "[forward_batch] B={} q_len={} TOTAL {:.1}ms",
                    self.2,
                    self.3,
                    self.1.elapsed().as_secs_f64() * 1000.0
                );
            }
        }
    }
    G(on, t0, b, q_len)
}
