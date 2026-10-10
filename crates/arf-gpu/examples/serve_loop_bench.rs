//! Standalone serving-loop harness: runs the EXACT host hot-path the actor runs
//! (`schedule -> build_forward -> backend.step -> commit_tokens`) over N requests
//! to completion, then exits cleanly. Three uses:
//!   1. A lasting batched-serving throughput harness (no HTTP, no client).
//!   2. The PGO training workload (a clean exit flushes `.profraw`).
//!   3. A CPU-vs-GPU split profiler: it times the host scheduler/encode work
//!      (`schedule + build_forward + commit`) separately from `backend.step`
//!      (the GPU forward + sample), so we can see PGO's ceiling — PGO only
//!      optimizes host code, so if `step` dominates wall time, PGO can't move
//!      tok/s much.
//!
//! Run:
//!   ARF_MODEL_PATH=models/llama-3.2-1b QUANT=q4k CONC=32 REQS=64 GEN=64 \
//!   PREFIX=256 PREFIX_CACHE=1 \
//!     cargo run --release -p arf-gpu --example serve_loop_bench

// L364 — this whole harness drives the macOS-only island API, so off macOS every import here
// serves a `main` that is not compiled. One file-level allow rather than ~10 per-import cfgs
// (the cascade documented in docs/PLATFORMS.md).
#![cfg_attr(not(target_os = "macos"), allow(unused_imports, dead_code))]

use std::path::{Path, PathBuf};
use std::time::Instant;

use arf_core::backend::BatchedBackend;
use arf_core::config::{EngineConfig, KvQuant, ModelConfig, Quant};
use arf_core::engine::build_forward;
use arf_core::model::batch::ForwardBatch;
use arf_core::sampling::{SamplingParams, SeqSampling};
use arf_core::scheduler::{BatchPlan, Request, Scheduler};
use arf_gpu::gpu::GpuContext;
use arf_gpu::WgpuBatched;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}

// L364 — macOS-only harness. It drives the island's depth-2 / k-step submission API directly
// (`depth2_eligible`, `submit_depth2`, `read_depth2`, `submit_kstep`), which is Metal-gated, so
// off macOS this compiles to a `main` that says why instead of failing the Linux build.
#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!(
        "serve_loop_bench drives the native-Metal island's depth-2 submission API, which is \
         macOS-only (L364). See docs/PLATFORMS.md."
    );
}

#[cfg(target_os = "macos")]
fn main() {
    // ── DEFAULT THE FAST-PATH LEVERS (the footgun that burned entire sessions) ────────────────
    // The serve DAEMON (`make dev`) defaults the megakernel + island on. This BENCH did NOT — it
    // relied on the caller remembering `ARF_MSL_GEMV=1 ARF_MEGAKERNEL=1` on the command line.
    // Forget them → the bench silently measures the SLOW portable path → "we lost our speed" panic,
    // over and over, because the number people measure with (this harness) didn't match the product
    // (the daemon). Fixed for good: the bench now defaults the SAME levers the daemon does, so
    // it is IMPOSSIBLE to measure the wrong path by omission. Opt out explicitly for A/B:
    //   ARF_NO_MSL_GEMV=1 / ARF_NO_MEGAKERNEL=1 / ARF_NO_BATCH_MEGA=1.
    // (ARF_NO_BATCH_MEGA is already the daemon's opt-out; the island/megakernel mirror it here.)
    if std::env::var_os("ARF_NO_MSL_GEMV").is_none() {
        std::env::set_var("ARF_MSL_GEMV", "1");
    }
    if std::env::var_os("ARF_NO_MEGAKERNEL").is_none() {
        std::env::set_var("ARF_MEGAKERNEL", "1");
    }
    // L89: ARF_KV_F16 — the daemon has defaulted this ON since L5b (main.rs:306) but this bench
    // never set it, so EVERY conc-N number measured here was taken on the f32 KV pool the product
    // does NOT ship. Measured cost of the omission at conc1 (ABAB, GEN=32):
    //     without: 85, 84   with: 88, 88   -> +4.1%
    // Exactly the failure mode the block above exists to prevent ("IMPOSSIBLE to measure the wrong
    // path by omission") — the list was just incomplete. Opt out with ARF_NO_KV_F16=1, which is
    // also the daemon's own opt-out, so both stay in step.
    if std::env::var_os("ARF_NO_KV_F16").is_none() {
        std::env::set_var("ARF_KV_F16", "1");
    }
    // L107: prefill pack width — daemon defaults 256 (WIN_CONFIG + batch.rs unwrap_or). Pin here
    // so serve_loop_bench measures the same packed-prefill path the product ships. Override with
    // ARF_PREFILL_CHUNK_ROWS=N for A/B (e.g. =128 for the old default).
    if std::env::var_os("ARF_PREFILL_CHUNK_ROWS").is_none() {
        std::env::set_var("ARF_PREFILL_CHUNK_ROWS", "256");
    }
    // The single-chat BURST path (SINGLEQ island-embed + BLOCKING island fence) is what makes
    // single-stream ~89 tok/s instead of ~11 — and the DAEMON defaults BOTH on (arf-serve
    // main.rs:260-268). The bench did NOT, so measuring conc1 here without remembering these two
    // flags gave 11 tok/s and a false "we lost single-stream" panic. `batch.rs:459` needs BOTH
    // present together to arm the fast B=1 path. Default them here so the bench measures the SAME
    // path the product ships. Opt out with ARF_NO_MEGA_SINGLEQ=1.
    // (KTOK defaults to 1 in batch.rs:741 via unwrap_or(1) — no set_var needed. This comment used
    //  to say "8"; the burst lever measured -26.9% and the default was flipped to 1 for +37% conc1.)
    // CONC==1 (single chat) → arm SINGLEQ + BLOCKING (the B=1 burst fence — REQUIRED for the
    // 4-byte-read burst that gives ~89 tok/s instead of ~11). CONC>1 (batched serving) → do NOT
    // arm BLOCKING: it forces waitUntilCompleted() every step on the batched path, which DRAINS the
    // GPU between steps → 79% inter-step starvation → the DVFS governor pins ~250MHz (measured
    // 2026-07-24, ARF_BMEGA_GPUIDLE). The async completion-handler ring (BLOCKING off) is the
    // batched fast path (mega_pipelined=true). Was: blanket-armed BLOCKING for ALL conc → throttled
    // conc64 to ~340 vs the 439 async peak. Opt out entirely with ARF_NO_MEGA_SINGLEQ=1.
    let _conc_for_singleq = env_usize("CONC", 32);
    if std::env::var_os("ARF_NO_MEGA_SINGLEQ").is_none() {
        if std::env::var_os("ARF_MEGA_SINGLEQ").is_none() {
            std::env::set_var("ARF_MEGA_SINGLEQ", "1");
        }
        // BLOCKING only for single-stream. Batched (conc>1) stays on the async ring.
        if _conc_for_singleq == 1 && std::env::var_os("ARF_MEGA_BLOCKING").is_none() {
            std::env::set_var("ARF_MEGA_BLOCKING", "1");
        }
    }

    // ARCH=qwen3-coder + GGUF=<path> → load a Qwen3-Coder-30B MoE GGUF directly (the concurrency
    // win measurement). Else the legacy llama-1b safetensors path.
    let arch = std::env::var("ARCH").unwrap_or_default();
    let gguf_path = std::env::var("GGUF").ok();
    let (cfg, is_gguf) = match arch.as_str() {
        "qwen3-coder" | "qwen3-coder-30b" | "qwen3moe" => (ModelConfig::qwen3_coder_30b(), true),
        _ => (ModelConfig::llama_3_2_1b(), false),
    };
    let dir = PathBuf::from(std::env::var("ARF_MODEL_PATH").unwrap_or_else(|_| ".".into()));
    let mf = dir.join("model.safetensors");
    let paths: [&Path; 1] = [mf.as_path()];

    let conc = env_usize("CONC", 32);
    let reqs = env_usize("REQS", 64);
    let gen = env_usize("GEN", 64);
    let prefix = env_usize("PREFIX", 0);
    let prefix_cache = env_usize("PREFIX_CACHE", if prefix > 0 { 1 } else { 0 }) != 0;
    let quant = match std::env::var("QUANT").as_deref() {
        Ok("int8") => Quant::Int8,
        Ok("q4") => Quant::Q4,
        Ok("q4k") => Quant::Q4K,
        Ok("q4ks") => Quant::Q4KS, // the island MoE indirect-GEMV path (megakernel + batched mega)
        _ => Quant::None,
    };

    // KV pool must hold `conc` sequences each up to (prefix + tail + gen) tokens.
    let per_seq = (prefix + 8 + gen + 16).next_multiple_of(16);
    let max_ctx = env_usize("MAX_CTX", conc * per_seq);
    let block_size = 16usize;
    let num_blocks = max_ctx.div_ceil(block_size);

    // ── ARF_KV_F16 SELF-CHECK (real-prompt prefill→decode argmax parity) ────────────
    // The kernel-level gate (batched_mega_parity: prefill_mirror_f16_selftest) proves the CONVERT
    // kernel mirrors the f32 pool → f16 correctly. This mode proves the END-TO-END wiring: run the
    // SAME prompts (a real prefill>1 then several decode steps) through the SAME serving loop TWICE
    // — once with ARF_KV_F16 OFF (the f32 oracle) and once with it ON (the f16 pool + the new
    // prefill-mirror) — and assert every sequence's generated token stream is IDENTICAL. The f16 KV
    // cache is lossy but greedy-argmax-robust, so the tokens MUST match; a mismatch means the
    // prefill slots were not mirrored (the exact gap the prefill mirror closes) or the convert is wrong.
    //
    // The flag is read at MODEL LOAD (weights.rs gates the f16 pool alloc on it), so each run
    // loads its own model with the flag set accordingly. HEAVY: this loads the real model twice →
    // DEFER on the shared machine. Run when the machine is free:
    //   ARF_KV_F16_SELFCHECK=1 ARF_MSL_GEMV=1 ARF_MEGAKERNEL=1 \
    //   ARF_ATTN_BATCHED=1 ARF_ATTN_COALESCED=1 \
    //   ARCH=qwen3-coder GGUF=<path/to/qwen3-coder-30b.gguf> QUANT=q4ks \
    //   CONC=4 REQS=4 GEN=16 PREFIX=32 \
    //     cargo run --release -p arf-gpu --example serve_loop_bench
    if std::env::var_os("ARF_KV_F16_SELFCHECK").is_some() {
        // f16 needs the coalesced batched island attention; make sure those levers are on so the
        // f16 pool is actually the read path (else the check would pass trivially on the f32 path).
        for k in ["ARF_ATTN_BATCHED", "ARF_ATTN_COALESCED"] {
            if std::env::var_os(k).is_none() {
                std::env::set_var(k, "1");
            }
        }
        // Build the prompt set ONCE (shared across both runs) so the token streams are comparable.
        let shared_p: Vec<u32> = (0..prefix.max(8)).map(|i| (i % 97 + 3) as u32).collect();
        let mut prompts: Vec<(u64, Vec<u32>)> = Vec::with_capacity(reqs.max(1));
        for i in 0..reqs.max(1) {
            let mut p = shared_p.clone();
            p.extend_from_slice(&[(i % 50 + 200) as u32, (i / 50 + 250) as u32, 300]);
            prompts.push((i as u64, p));
        }
        // One full serving run at the current flag state; returns each seq's token stream.
        let run = |flag_on: bool| -> std::collections::BTreeMap<u64, Vec<u32>> {
            if flag_on {
                std::env::set_var("ARF_KV_F16", "1");
            } else {
                std::env::remove_var("ARF_KV_F16");
            }
            let gctx = std::sync::Arc::new(GpuContext::new().expect("gpu"));
            let model = if is_gguf {
                let gp = PathBuf::from(
                    gguf_path
                        .clone()
                        .expect("set GGUF=<path> for ARCH=qwen3-coder"),
                );
                arf_gpu::weights::load_gguf_gpu_kv_quant_pooled(
                    &cfg,
                    &gp,
                    max_ctx,
                    &gctx,
                    quant,
                    KvQuant::None,
                    Some(num_blocks),
                )
                .expect("load gguf")
            } else {
                arf_gpu::weights::load_safetensors_gpu_kv_quant_pooled(
                    &cfg,
                    &paths,
                    max_ctx,
                    &gctx,
                    quant,
                    KvQuant::None,
                    Some(num_blocks),
                )
                .expect("load")
            };
            let backend = WgpuBatched(model);
            let mut sched = Scheduler::new(EngineConfig {
                block_size,
                num_blocks,
                max_batch_size: conc,
                max_prefill_tokens: 512,
                enable_prefix_cache: false,
                ..Default::default()
            });
            for (id, p) in &prompts {
                sched.add(Request::new(*id, p.clone(), SamplingParams::greedy(gen)));
            }
            let g = SamplingParams::greedy(gen);
            let mut streams: std::collections::BTreeMap<u64, Vec<u32>> =
                std::collections::BTreeMap::new();
            while sched.has_unfinished() {
                let Some(plan) = sched.schedule().unwrap() else {
                    break;
                };
                let (ids, batch) = build_forward(&plan, sched.block_size());
                let samp: Vec<SeqSampling> = plan
                    .seqs
                    .iter()
                    .map(|sp| SeqSampling {
                        params: &g,
                        position: sp.past_len + sp.q_len,
                        generated: &[],
                    })
                    .collect();
                let toks = backend.step(&ids, &batch, &samp).unwrap();
                for out in sched.commit_tokens(&toks).unwrap() {
                    streams.entry(out.id).or_default().push(out.token);
                }
            }
            streams
        };
        eprintln!("[kv-f16-selfcheck] run 1/2: ARF_KV_F16 OFF (f32 oracle)...");
        let off = run(false);
        eprintln!("[kv-f16-selfcheck] run 2/2: ARF_KV_F16 ON (f16 pool + prefill-mirror)...");
        let on = run(true);
        let mut mism = 0usize;
        for (id, off_stream) in &off {
            let on_stream = on.get(id);
            if on_stream != Some(off_stream) {
                mism += 1;
                eprintln!(
                    "  seq {id}: DIVERGED\n    off={:?}\n    on ={:?}",
                    &off_stream[..off_stream.len().min(24)],
                    on_stream.map(|v| &v[..v.len().min(24)]).unwrap_or(&[])
                );
            }
        }
        if mism == 0 {
            println!("KV_F16 SELFCHECK: PASS — f16-on token streams IDENTICAL to f16-off (f32 oracle) for all {} seqs (real prefill→decode).", off.len());
        } else {
            println!("KV_F16 SELFCHECK: FAIL — {mism}/{} seqs diverged (prefill NOT mirrored, or convert bug).", off.len());
            std::process::exit(1);
        }
        return;
    }

    let gctx = std::sync::Arc::new(GpuContext::new().expect("gpu"));
    if std::env::var_os("PROFILE").is_some() {
        gctx.set_profiling(true);
        eprintln!("[profile] per-kernel GPU timing ON");
    }
    eprintln!(
        "device {} | weights {quant:?} | conc {conc} reqs {reqs} gen {gen} | prefix {prefix} \
         (cache {}) | pool {num_blocks}x{block_size}\n",
        gctx.info(),
        if prefix_cache { "ON" } else { "OFF" }
    );

    let model = if is_gguf {
        let gp = PathBuf::from(gguf_path.expect("set GGUF=<path> for ARCH=qwen3-coder"));
        eprintln!("loading qwen3-coder-30b MoE GGUF ({quant:?})...");
        arf_gpu::weights::load_gguf_gpu_kv_quant_pooled(
            &cfg,
            &gp,
            max_ctx,
            &gctx,
            quant,
            KvQuant::None,
            Some(num_blocks),
        )
        .expect("load gguf")
    } else {
        arf_gpu::weights::load_safetensors_gpu_kv_quant_pooled(
            &cfg,
            &paths,
            max_ctx,
            &gctx,
            quant,
            KvQuant::None,
            Some(num_blocks),
        )
        .expect("load")
    };
    let backend = WgpuBatched(model);

    let mut sched = Scheduler::new(EngineConfig {
        block_size,
        num_blocks,
        max_batch_size: conc,
        max_prefill_tokens: 512,
        enable_prefix_cache: prefix_cache,
        ..Default::default()
    });

    // Each request = [shared prefix][unique tail]. The prefix is identical across
    // requests (so prefix caching can reuse it); the tail makes each distinct.
    let shared: Vec<u32> = (0..prefix).map(|i| (i % 97 + 3) as u32).collect();
    for i in 0..reqs {
        let mut prompt = shared.clone();
        prompt.extend_from_slice(&[(i % 50 + 200) as u32, (i / 50 + 250) as u32, 300]);
        sched.add(Request::new(i as u64, prompt, SamplingParams::greedy(gen)));
    }

    let g = SamplingParams::greedy(gen);
    let mut host_ns = 0u128;
    let mut gpu_ns = 0u128;
    let mut generated = 0usize;
    // DUMP=1 → collect each sequence's generated token stream into a fingerprint so the
    // ARF_MSL_MOE island run can be diffed token-for-token against the wgpu baseline (the
    // kernels are parity-1e-3 → greedy argmax should be near-identical). Off by default (no cost).
    let dump = std::env::var_os("DUMP").is_some();
    let mut streams: std::collections::BTreeMap<u64, Vec<u32>> = std::collections::BTreeMap::new();
    // DECODE-ONLY window: to compare against llama's S_TG (decode-only tok/s) we must exclude the
    // prefill steps. A step is pure decode when every seq contributes exactly one token
    // (ids.len() == nseq) — the SAME eligibility the batched megakernel checks (batch.rs). We
    // accumulate wall-time + tokens ONLY over those steps, so `decode` below is the steady-state
    // number. The old `agg` (whole-run, prefill included) is kept for continuity but is NOT the
    // conc-tier comparison figure.
    let mut decode_ns = 0u128;
    let mut decode_toks = 0usize;
    // Per-decode-step latencies (ms), for the p50/p90/p95 tail report — the numbers that
    // actually matter for serving (a peak/best-token is a vanity figure).
    let mut decode_step_ms: Vec<f64> = Vec::new();
    // ── DEPTH-2 GATE — default OFF ────────────────────────────────────────────────────
    // ARF_BATCH_MEGA_DEPTH2 set AND ARF_NO_BATCH_MEGA_DEPTH2 unset arms the scheduler-side
    // depth-2 pipeline: submit step N+1 (reading step N's out_tokens bank GPU-side via
    // TokenSrc::GpuBank) BEFORE fencing/reading step N, so the GPU never idles across the
    // read→commit→schedule→build CPU gap that starves the conc64 clock. Two out_tokens banks
    // ping-pong. When the flag is OFF this variable is false and the loop runs the byte-identical
    // depth-1 path below (backend.step every step) — no behavior change.
    // ── K-STEP GATE — default OFF ─────────────────────────────────────────────────────
    // ARF_BATCH_MEGA_KSTEP=K chains up to K decode steps into ONE island command buffer (no CPU
    // seam between them) so the GPU runs continuously → the AGX DVFS governor ramps the pinned
    // 338MHz clock (the PROVEN conc64 root cause). At a KV block boundary or a roster change the
    // burst shrinks to k=1 to resync (the caller caps k to the block boundary + requires a stable
    // roster, exactly like the m=1 'ktok burst). Mutually exclusive with depth-2 (both feed the
    // GPU token-relay — kstep is the stronger, no-seam version). OFF ⇒ kstep_max==1 ⇒ no change.
    let kstep_max = std::env::var("ARF_BATCH_MEGA_KSTEP")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1);
    let kstep_on = kstep_max > 1;
    let depth2 = std::env::var_os("ARF_BATCH_MEGA_DEPTH2").is_some()
        && std::env::var_os("ARF_NO_BATCH_MEGA_DEPTH2").is_none()
        && !kstep_on;

    // DEPTH-2 IN-FLIGHT STEP: (handle, out-bank index it WROTE, roster it was submitted for).
    // `Some` == a GPU step for the CURRENT scheduler position was submitted (record+async-commit,
    // reading the PREVIOUS step's out_tokens bank GPU-side) but its tokens are NOT yet read. Every
    // iteration `schedule()`s exactly this step, submits the FOLLOWING step ahead (self-derived
    // geometry, async → GPU starts it), THEN reads+commits THIS step — so the following step's GPU
    // work overlaps this step's fence, yet the scheduler's schedule→commit pairing stays strictly
    // serial: ONE committed token per input step, in order. The self-derived geometry is used ONLY
    // to enqueue the GPU work early; the COMMIT always uses the scheduler's real `scheduled` plan,
    // and a roster match (below) proves the self-derived geometry equalled it. On any mismatch (a
    // retire/admit changed the roster) OR a KV block-boundary (a slot the scheduler must allocate)
    // the queued-ahead step is DROPPED unread and the step re-runs depth-1 — the design's Phase-1
    // drain.
    let mut pending: Option<(arf_gpu::gpu::BatchPending, usize, Vec<u32>)> = None;
    let block_size_toks = sched.block_size();

    // roster = the per-seq stable key (slots[0]). A roster that matches across the self-derived step
    // and the scheduler's fresh plan proves no retire/admit happened, so the self-derived geometry
    // (which assumed a stable roster) equalled the real one and the queued-ahead GPU step is valid.
    fn roster_of(batch: &ForwardBatch) -> Vec<u32> {
        batch
            .seqs
            .iter()
            .map(|s| s.slots.first().copied().unwrap_or(0))
            .collect()
    }
    // Same roster key, derived DIRECTLY from a scheduler `BatchPlan` WITHOUT build_forward.
    // roster_of reads `slots.first()`; build_forward fills `slots = slots_for(block_table, bs, ctx)`,
    // and slots_for maps logical position 0 → block_table[0]*bs + 0 (p=0 ⇒ p/bs=0, p%bs=0). So the
    // per-seq roster key is exactly block_table[0]*bs — no slot table, no write_runs needed. Mirrors
    // the `.unwrap_or(0)` fallback for an (impossible here) empty block_table so the two agree byte-
    // for-byte. This lets the post-burst commit loop detect a roster change without the ~full-forward
    // build_forward that was the per-step CPU wall.
    fn roster_of_plan(plan: &BatchPlan, block_size: usize) -> Vec<u32> {
        plan.seqs
            .iter()
            .map(|sp| {
                sp.block_table
                    .first()
                    .copied()
                    .map(|b| b * block_size as u32)
                    .unwrap_or(0)
            })
            .collect()
    }
    // Self-derive step N+1's batch from step N's real batch (past_len+1, one more contiguous slot).
    // Returns None when a seq would cross a KV block boundary (the next slot lives in a block only
    // `schedule()` can allocate) — the caller then does NOT submit-ahead (it reads N and re-primes),
    // exactly as the design's drain specifies. Only valid for pure q_len==1 decode. `positions`/
    // `write_runs` are left empty: the batched megakernel derives positions from past_len and scatters
    // via slots[past_len], so it reads neither (verified: submit_batched_step uses past_len + slots).
    fn derive_next_batch(prev: &ForwardBatch, block_size: usize) -> Option<ForwardBatch> {
        let mut seqs = Vec::with_capacity(prev.seqs.len());
        for s in &prev.seqs {
            if s.q_len != 1 {
                return None;
            }
            let new_pos = s.past_len + 1; // logical position of the token N+1 will WRITE
            if new_pos % block_size == 0 {
                return None; // crosses into a new block — cannot self-derive; do not submit-ahead.
            }
            let last_slot = *s.slots.last()?; // slot of position past_len (the token this step wrote)
            let mut slots = s.slots.clone();
            slots.push(last_slot + 1); // within-block: the next slot is contiguous
            seqs.push(arf_core::model::batch::SeqAttn {
                q_start: 0,
                q_len: 1,
                past_len: new_pos,
                slots,
                write_runs: Vec::new(),
                image_spans: Vec::new(),
                stream_id: None,
            });
        }
        Some(ForwardBatch {
            positions: Vec::new(),
            seqs,
            image_embeds: None,
            mrope_positions: None,
        })
    }

    // K-STEP burst geometry: given the CURRENT step's batch, build ONE batch whose per-seq
    // `slots` cover the burst's k new contiguous positions past_len+1..past_len+k (all within the
    // seq's current KV block). Returns the built batch + the achieved k (== requested k_req, capped
    // to the block boundary so no step writes a slot the scheduler hasn't reserved). Returns None if
    // the roster isn't pure q_len==1 decode. The megakernel derives each step ks's write slot from
    // slots[past_len+ks] and its pos from past_len+ks, so filling the contiguous slots here is the
    // whole geometry — the scheduler re-derives IDENTICAL slots when we commit the k tokens in order
    // (roster + block-boundary stable ⇒ the self-derived slots equal the scheduler's, same invariant
    // the depth-2 derive_next_batch relies on).
    fn derive_kstep_batch(
        cur: &ForwardBatch,
        block_size: usize,
        k_req: usize,
    ) -> Option<(ForwardBatch, usize)> {
        if cur.seqs.is_empty() || !cur.seqs.iter().all(|s| s.q_len == 1) {
            return None;
        }
        // Cap k to the block boundary, min over seqs. The burst's step ks (ks in 0..k) writes logical
        // position p+ks. Position p (ks=0) is THIS step's already-allocated slot. Positions p+1..p+k-1
        // are the k-1 self-derived slots; each is invalid iff it's a block multiple (== 0 mod block —
        // a slot only schedule() allocs), the SAME test derive_next_batch uses. So the largest safe k
        // is the smallest k where p+k would be a block multiple: k_max = block_size - (p % block_size).
        // (Positions p+1..p+k_max-1 are all in-block; p+k_max is the boundary, excluded.)
        let mut k = k_req;
        for s in &cur.seqs {
            let k_max_seq = block_size - (s.past_len % block_size); // p+k_max_seq is the next boundary
            k = k.min(k_max_seq);
        }
        if k == 0 {
            return None;
        } // at a boundary → caller falls back to k=1 (a single step).
        let mut seqs = Vec::with_capacity(cur.seqs.len());
        for s in &cur.seqs {
            // s.slots has length past_len+1; slots[past_len] is step 0's write slot. Step ks writes
            // slots[past_len+ks], so push k-1 MORE contiguous slots → indices past_len..past_len+k-1.
            let last_slot = *s.slots.last()?; // slot of position past_len (step 0's write slot)
            let mut slots = s.slots.clone();
            for i in 1..k {
                slots.push(last_slot + i as u32);
            } // k-1 contiguous in-block slots
            seqs.push(arf_core::model::batch::SeqAttn {
                q_start: 0,
                q_len: 1,
                past_len: s.past_len,
                slots,
                write_runs: Vec::new(),
                image_spans: Vec::new(),
                stream_id: None,
            });
        }
        Some((
            ForwardBatch {
                positions: Vec::new(),
                seqs,
                image_embeds: None,
                mrope_positions: None,
            },
            k,
        ))
    }

    // Shared per-decode-step accounting (both depth-1 and depth-2 land here).
    let record = |toks: &[u32],
                  step_ns: u128,
                  is_decode: bool,
                  b: usize,
                  gpu_ns: &mut u128,
                  decode_ns: &mut u128,
                  decode_toks: &mut usize,
                  decode_step_ms: &mut Vec<f64>,
                  generated: &mut usize,
                  streams: &mut std::collections::BTreeMap<u64, Vec<u32>>,
                  sched: &mut Scheduler|
     -> u128 {
        *gpu_ns += step_ns;
        if is_decode {
            *decode_ns += step_ns;
            *decode_toks += toks.len();
            decode_step_ms.push(step_ns as f64 / 1e6);
            if std::env::var_os("STEP_WALL").is_some() {
                eprintln!(
                    "[step-wall] decode B={} wall {:.1}ms",
                    b,
                    step_ns as f64 / 1e6
                );
            }
        }
        let tc = Instant::now();
        for out in sched.commit_tokens(toks).unwrap() {
            if dump {
                streams.entry(out.id).or_default().push(out.token);
            }
            if !out.finished || out.finish_reason == Some(arf_core::scheduler::FinishReason::Length)
            {
                *generated += 1;
            }
        }
        tc.elapsed().as_nanos()
    };

    let wall = Instant::now();
    while sched.has_unfinished() {
        // ── (1) SCHEDULE the current step S (the scheduler is at S; `pending`, if any, is S's GPU
        //        work already submitted last iteration). ──────────────────────────────────────────
        let t = Instant::now();
        let Some(plan) = sched.schedule().unwrap() else {
            break;
        };
        let (ids, batch) = build_forward(&plan, sched.block_size());
        let is_pure_decode = ids.len() == plan.seqs.len();
        let samp: Vec<SeqSampling> = plan
            .seqs
            .iter()
            .map(|sp| SeqSampling {
                params: &g,
                position: sp.past_len + sp.q_len,
                generated: &[],
            })
            .collect();
        let all_greedy = !samp.is_empty()
            && samp.iter().all(|s| {
                s.params.is_greedy() && (s.params.repetition_penalty - 1.0).abs() < f32::EPSILON
            });
        let roster = roster_of(&batch);
        let d2_ok = depth2 && all_greedy && backend.depth2_eligible(&batch);
        host_ns += t.elapsed().as_nanos();

        if std::env::var_os("ARF_TRACE_STEP").is_some() {
            eprintln!(
                "[step] {} ids, {} seqs, pure_decode={} d2_ok={} pending={} → START",
                ids.len(),
                plan.seqs.len(),
                is_pure_decode,
                d2_ok,
                pending.is_some()
            );
        }

        // ── (K) K-STEP BURST — chain up to kstep_max decode steps into ONE island command
        //        buffer (no CPU seam) so the GPU runs continuously → the DVFS clock ramps. Engaged
        //        only for a pure greedy decode step that's batched-mega eligible; the burst is capped
        //        to the KV block boundary + a stable roster (derive_kstep_batch). k==1 (boundary) →
        //        fall through to the single-step paths below (a k=1 resync). The megakernel samples
        //        all k steps' tokens in ONE record writing out_bank[ks*B+r]; we then advance the
        //        scheduler ONE step at a time (schedule()+commit_tokens per step, as the scheduler
        //        requires tokens aligned to each step's plan), verifying the roster still matches the
        //        submitted burst. On the first mismatch (a seq finished mid-burst → retire) we stop
        //        and DROP the remaining speculative tokens (their KV writes are re-planned + reused
        //        by the depth-1 re-run — the same Phase-1 drain as depth-2). ────────────────────────
        let kstep_ok = kstep_on && is_pure_decode && all_greedy && backend.depth2_eligible(&batch);
        if kstep_ok {
            if let Some((kbatch, k)) = derive_kstep_batch(&batch, sched.block_size(), kstep_max) {
                if k > 1 {
                    let tg = Instant::now();
                    // Submit the k-step chain (CPU ids into bank 0; steps 1..k read GPU-side). Read
                    // k*B tokens step-major (toks[ks*B..ks*B+B] = step ks's B sampled ids).
                    if let Some(ph) = backend.submit_kstep(0, &ids, &kbatch, k) {
                        let all_toks = backend.read_depth2(&ph, 0);
                        let burst_ns = tg.elapsed().as_nanos();
                        let bsz = plan.seqs.len();
                        // Per-step wall = burst_ns / k (the honest per-step latency for the p50/tps).
                        let per_step_ns = burst_ns / (k as u128);
                        // Commit each step in order. Step 0 uses the ALREADY-scheduled `plan`; steps
                        // 1..k re-schedule (advancing the scheduler). We stop the burst the moment a
                        // seq FINISHES (a retire → the next schedule's roster changes; committing more
                        // speculative tokens would misalign advance_and_retire). Committing inline (not
                        // via `record`) lets us inspect the finished flag. The scheduler always has a
                        // fresh plan committed here (schedule()→commit_tokens per step), so `scheduled`
                        // is never left dangling. Correctness: every committed token had a real plan.
                        let mut committed_any = false;
                        let cpu_loop_t = Instant::now();
                        for ks in 0..k {
                            // Ensure a plan exists + matches the burst roster for this step.
                            if ks > 0 {
                                // Advance the scheduler to step ks. This sets `scheduled` — which we
                                // MUST then consume (commit) or the next top-of-loop schedule() asserts.
                                // derive the roster to compare DIRECTLY from the plan (a cheap
                                // block_table[0]*bs per seq) instead of running the full build_forward
                                // (slots_for/write_runs for all B seqs). build_forward is ONLY needed to
                                // actually RUN the megakernel step on a mismatch, so it is built lazily in
                                // the mismatch branch. NB (measured, BURST_BREAKDOWN=1): at B=64/K=8 the
                                // per-burst CPU (this whole loop) is <2ms and the k× build_forward it once
                                // held was NOT the wall — the ~1.8s/burst is the GPU submit+read itself
                                // (~225ms/step). This is a correctness-preserving CPU cleanup, not the win.
                                let ta = Instant::now();
                                match sched.schedule().unwrap() {
                                    None => {
                                        host_ns += ta.elapsed().as_nanos();
                                        break;
                                    } // drained mid-burst → nothing dangling.
                                    Some(p2) => {
                                        if roster_of_plan(&p2, sched.block_size()) != roster {
                                            // Roster changed WITHOUT a mid-burst retire (an admit brought
                                            // a new seq in). The burst's k*B tokens don't cover this new
                                            // roster, so we can't reuse them. Consume the already-set plan
                                            // via the depth-1 path (a real megakernel step for p2) — NOW we
                                            // build_forward — commit it, then stop the burst. Keeps
                                            // `scheduled` consumed (correct). Behavior UNCHANGED from before
                                            // the fix: same plan p2, same build_forward, same step, same
                                            // record — only the cheap roster check moved ahead of the build.
                                            let (ids2, b2) = build_forward(&p2, sched.block_size());
                                            host_ns += ta.elapsed().as_nanos();
                                            let samp2: Vec<SeqSampling> = p2
                                                .seqs
                                                .iter()
                                                .map(|sp| SeqSampling {
                                                    params: &g,
                                                    position: sp.past_len + sp.q_len,
                                                    generated: &[],
                                                })
                                                .collect();
                                            let tg2 = Instant::now();
                                            let toks2 = backend.step(&ids2, &b2, &samp2).unwrap();
                                            let step_ns2 = tg2.elapsed().as_nanos();
                                            host_ns += record(
                                                &toks2,
                                                step_ns2,
                                                ids2.len() == p2.seqs.len(),
                                                b2.seqs.len(),
                                                &mut gpu_ns,
                                                &mut decode_ns,
                                                &mut decode_toks,
                                                &mut decode_step_ms,
                                                &mut generated,
                                                &mut streams,
                                                &mut sched,
                                            );
                                            committed_any = true;
                                            break;
                                        }
                                        host_ns += ta.elapsed().as_nanos();
                                    }
                                }
                            }
                            let toks_ks = &all_toks[ks * bsz..ks * bsz + bsz];
                            // Accounting (mirror `record`, but keep the outputs to check finished).
                            gpu_ns += per_step_ns;
                            decode_ns += per_step_ns;
                            decode_toks += toks_ks.len();
                            decode_step_ms.push(per_step_ns as f64 / 1e6);
                            if std::env::var_os("STEP_WALL").is_some() {
                                eprintln!(
                                    "[step-wall] kstep B={} step {}/{} wall {:.1}ms",
                                    bsz,
                                    ks + 1,
                                    k,
                                    per_step_ns as f64 / 1e6
                                );
                            }
                            let tc = Instant::now();
                            let mut any_finished = false;
                            for out in sched.commit_tokens(toks_ks).unwrap() {
                                if dump {
                                    streams.entry(out.id).or_default().push(out.token);
                                }
                                if !out.finished
                                    || out.finish_reason
                                        == Some(arf_core::scheduler::FinishReason::Length)
                                {
                                    generated += 1;
                                }
                                if out.finished {
                                    any_finished = true;
                                }
                            }
                            host_ns += tc.elapsed().as_nanos();
                            committed_any = true;
                            if any_finished {
                                break;
                            } // a seq retired → stop before scheduling ks+1.
                        }
                        if std::env::var_os("BURST_BREAKDOWN").is_some() {
                            eprintln!("[burst] k={} GPU(submit+read)={:.1}ms CPU(sched+commit loop)={:.1}ms",
                                k, burst_ns as f64 / 1e6, cpu_loop_t.elapsed().as_nanos() as f64 / 1e6);
                        }
                        if committed_any {
                            continue;
                        }
                    }
                }
            }
        }

        // ── (2) If a step S is in flight, does its submitted geometry still match the scheduler's
        //        real plan? (roster stable ⇒ the self-derived slots equalled the real ones). ───────
        let stale = pending
            .as_ref()
            .map(|(_, _, proster)| *proster != roster || !d2_ok)
            .unwrap_or(false);
        if stale {
            // A retire/admit changed the roster (or S is no longer d2-eligible). The queued-ahead
            // GPU step S ran with the OLD geometry — DROP it unread. Its KV writes went to slots the
            // scheduler will re-plan; this depth-1 re-run of S overwrites them before any read. No
            // token is committed for the dropped work (correctness: the depth-1 run below produces
            // S's one token). This is the Phase-1 one-step drain on roster change.
            pending = None;
        }

        // ── (3) DEPTH-2: submit the FOLLOWING step ahead (overlap), then read+commit THIS step. ──
        if let Some((ph_s, bank_s, _)) = pending.take() {
            // S is already in flight (submitted last iteration into bank_s). Self-derive S+1 and
            // submit it NOW reading bank_s GPU-side → its command buffer queues behind S, so the GPU
            // runs S then S+1 back-to-back while the CPU fences S below (the overlap).
            let tg = Instant::now();
            let mut next_pending = None;
            if let Some(next_batch) = derive_next_batch(&batch, block_size_toks) {
                let other = 1 - bank_s;
                if let Some(p_next) = backend.submit_depth2(Some(bank_s), other, &[], &next_batch) {
                    next_pending = Some((p_next, other, roster_of(&next_batch)));
                }
            }
            // Fence + read S's REAL tokens (S+1, if queued, runs meanwhile). Commit against S's plan.
            let toks_s = backend.read_depth2(&ph_s, bank_s);
            let step_ns = tg.elapsed().as_nanos();
            host_ns += record(
                &toks_s,
                step_ns,
                true,
                toks_s.len(),
                &mut gpu_ns,
                &mut decode_ns,
                &mut decode_toks,
                &mut decode_step_ms,
                &mut generated,
                &mut streams,
                &mut sched,
            );
            pending = next_pending; // S+1 is the new in-flight step (or None at a boundary → re-prime)
            continue;
        }

        // ── (4) PRIME (no in-flight step, but S is depth-2 eligible): submit S (Cpu ids into bank
        //        0), read+commit it, then submit the self-derived S+1 ahead so the pipeline is armed
        //        for the next iteration. S's token is byte-identical to depth-1 (TokenSrc::Cpu). ──
        if d2_ok {
            let tg = Instant::now();
            let p_s = backend
                .submit_depth2(None, 0, &ids, &batch)
                .expect("depth2 prime submit must record");
            // Arm S+1 ahead (reads S's bank 0 GPU-side) before fencing S — same overlap as (3).
            let mut next_pending = None;
            if let Some(next_batch) = derive_next_batch(&batch, block_size_toks) {
                if let Some(p_next) = backend.submit_depth2(Some(0), 1, &[], &next_batch) {
                    next_pending = Some((p_next, 1, roster_of(&next_batch)));
                }
            }
            let toks_s = backend.read_depth2(&p_s, 0);
            let step_ns = tg.elapsed().as_nanos();
            host_ns += record(
                &toks_s,
                step_ns,
                is_pure_decode,
                toks_s.len(),
                &mut gpu_ns,
                &mut decode_ns,
                &mut decode_toks,
                &mut decode_step_ms,
                &mut generated,
                &mut streams,
                &mut sched,
            );
            pending = next_pending;
            continue;
        }

        // ── (5) PLAIN DEPTH-1 (default path; flag off, or prefill/ineligible step). ─────────────
        if std::env::var_os("ARF_TRACE_STEP").is_some() {
            eprintln!("[step] depth-1 backend.step START");
        }
        let tg = Instant::now();
        let tokens = backend.step(&ids, &batch, &samp).unwrap();
        if std::env::var_os("ARF_TRACE_STEP").is_some() {
            eprintln!("[step] backend.step DONE ({} tokens)", tokens.len());
        }
        let step_ns = tg.elapsed().as_nanos();
        host_ns += record(
            &tokens,
            step_ns,
            is_pure_decode,
            batch.seqs.len(),
            &mut gpu_ns,
            &mut decode_ns,
            &mut decode_toks,
            &mut decode_step_ms,
            &mut generated,
            &mut streams,
            &mut sched,
        );
    }

    // ── POST-LOOP DRAIN ───────────────────────────────────────────────────────────────
    // The loop always reads+commits step S in the SAME iteration it submits S+1 ahead, so a still-
    // in-flight `pending` here means the LAST armed S+1 was never scheduled (has_unfinished() went
    // false — every seq finished at S). That S+1 is speculative work for positions the finished seqs
    // will never reach: DROP it unread (no scheduler plan exists to commit it against, and its tokens
    // are not part of any request's output). Correctness: every INPUT step got exactly one committed
    // token inside the loop; the dropped speculative S+1 has no input step. This preserves the
    // one-token-per-input invariant (a spurious commit here would ADD a phantom token).
    let _ = pending.take();
    if dump {
        // Global checksum + seq-0 head: a stable fingerprint to diff baseline vs island.
        let mut sum: u64 = 0;
        let mut count = 0usize;
        for toks in streams.values() {
            for &t in toks {
                sum = sum.wrapping_mul(1000003).wrapping_add(t as u64);
                count += 1;
            }
        }
        let head: Vec<u32> = streams
            .values()
            .next()
            .map(|v| v.iter().take(16).copied().collect())
            .unwrap_or_default();
        println!("DUMP: tokens={count} checksum={sum:#018x} seq0_head={head:?}");
        // DUMP_ALL=1 → print EVERY sequence's full token stream (to diff baseline vs island
        // per-sequence and find exactly which seq/token diverges).
        if std::env::var_os("DUMP_ALL").is_some() {
            for (id, toks) in &streams {
                println!("SEQ {id}: {toks:?}");
            }
        }
    }
    let secs = wall.elapsed().as_secs_f64();
    let host_ms = host_ns as f64 / 1e6;
    let gpu_ms = gpu_ns as f64 / 1e6;
    let wall_ms = secs * 1e3;
    let decode_tps = if decode_ns > 0 {
        decode_toks as f64 / (decode_ns as f64 / 1e9)
    } else {
        0.0
    };
    // ── THE HEADLINE: canonical steady-state aggregate throughput ────────────────────────────
    // The p50 of the per-decode-step wall, converted to aggregate tok/s (conc * 1000 / p50_ms),
    // with the COLD first token (Metal-pipeline JIT + first-cmdbuf stall) dropped. THIS is the
    // number that was measured and compares apples-to-apples with llama's
    // aggregate S t/s. It is printed FIRST and labeled so nobody ever again quotes `agg` (which
    // is prefill+cold-stall polluted) or best-token (a vanity peak) by accident. Fixed for
    // good: the honest number is now the DEFAULT headline, not an env-gated afterthought.
    if decode_step_ms.len() > 2 {
        let mut lat: Vec<f64> = decode_step_ms[1..].to_vec(); // drop cold token 0
        lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = lat.len();
        let pct = |p: f64| lat[((n as f64 * p / 100.0) as usize).min(n - 1)];
        let tps = |ms: f64| conc as f64 * 1000.0 / ms;
        let (p50, p90, p95, p99) = (pct(50.0), pct(90.0), pct(95.0), pct(99.0));
        println!(
            "▶ CANONICAL conc{conc} = {:.0} tok/s aggregate (steady p50; compares to llama agg S t/s). \
             cold-token dropped, {n} warm steps.",
            tps(p50)
        );
        println!(
            "  distribution: p50 {p50:.1}ms->{:.0} | p90 {p90:.1}ms->{:.0} | p95 {p95:.1}ms->{:.0} | \
             p99 {p99:.1}ms->{:.0} tok/s  (per-user latency = the p-values; aggregate = conc/step)",
            tps(p50), tps(p90), tps(p95), tps(p99),
        );
        // L118 — HONEST GPU-BUSY (ARF_BMEGA_GPUBUSY=1): summed over EVERY command buffer, not
        // read off the final chunk like ARF_BMEGA_GPUIDLE (whose mis-read produced the
        // retracted L114). busy% below ~90% means the GPU really is under-fed.
        #[cfg(target_os = "macos")]
        if std::env::var_os("ARF_BMEGA_GPUBUSY").is_some() {
            if let Some(isl) = backend.island_for_probe() {
                let (busy_ns, bufs, span_ns) = isl.lock().unwrap().take_gpubusy();
                if span_ns > 0 && bufs > 0 {
                    println!(
                        "  [gpu-busy] {:.1} ms busy over {:.1} ms span = {:.1}% BUSY | {} buffers, \
                         {:.2} ms/buffer  (summed over ALL buffers)",
                        busy_ns as f64 / 1e6, span_ns as f64 / 1e6,
                        100.0 * busy_ns as f64 / span_ns as f64,
                        bufs, busy_ns as f64 / 1e6 / bufs as f64,
                    );
                }
            }
        }
    } else {
        println!("▶ CANONICAL: run too short for a steady p50 (need >2 decode steps) — increase REQS/GEN.");
    }

    // Secondary/diagnostic. `agg` is whole-run (prefill + cold-stall included) → NOT the comparison
    // metric; kept only for continuity. DECODE = decode-only mean (drifts below p50 under jitter).
    println!(
        "  [diag] agg {:.1} (whole-run, NOT the comparison metric) | DECODE-mean {decode_tps:.1} | \
         wall {wall_ms:.0}ms | gpu {:.1}% | host {:.1}% | reused {} tok",
        generated as f64 / secs,
        100.0 * gpu_ms / wall_ms,
        100.0 * host_ms / wall_ms,
        sched.prefix_cache_reused_tokens(),
    );
}
