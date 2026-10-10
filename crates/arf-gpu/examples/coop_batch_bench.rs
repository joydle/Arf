//! Batched-decode aggregate throughput, the vLLM-axis metric: N concurrent
//! sequences decoded in one continuous batch. Compares the cooperative-matrix GEMM
//! path (default) against the scalar path (ARF_NO_COOP=1) — the test of
//! whether the matrix units lift P8's compute-bound batched ceiling (~106 bf16 /
//! ~122 int8 tok/s at N=16, *below* single-stream).
//!
//! bf16 weights (coop is wired for the bf16 batched path; Q4/int8-into-coop is C4).
//! Run both:
//!   ARF_MODEL_PATH=models/llama-3.2-1b GEN=32 \
//!     cargo run --release -p arf-gpu --example coop_batch_bench
//!   ARF_NO_COOP=1 ARF_MODEL_PATH=models/llama-3.2-1b GEN=32 \
//!     cargo run --release -p arf-gpu --example coop_batch_bench

use std::path::{Path, PathBuf};
use std::time::Instant;

use arf_core::config::{KvQuant, ModelConfig, Quant};
use arf_gpu::gpu::{GpuContext, GpuModel};

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}

fn main() {
    let dir = PathBuf::from(std::env::var("ARF_MODEL_PATH").expect("set ARF_MODEL_PATH"));
    let cfg = ModelConfig::llama_3_2_1b();
    let mf = dir.join("model.safetensors");
    let paths: [&Path; 1] = [mf.as_path()];
    let gen = env_usize("GEN", 32);
    let ns: Vec<usize> = std::env::var("N")
        .unwrap_or_else(|_| "1,8,16,32".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let max_n = ns.iter().copied().max().unwrap_or(32);
    // The KV pool must hold ALL N sequences concurrently: N × per-seq slots.
    let per_seq = (8 + gen + 16).next_multiple_of(16);
    let max_ctx = max_n * per_seq;

    let quant = match std::env::var("QUANT").as_deref() {
        Ok("int8") => Quant::Int8,
        Ok("q4") => Quant::Q4,
        Ok("q4k") => Quant::Q4K,
        _ => Quant::None,
    };
    let gctx = std::sync::Arc::new(GpuContext::new().expect("gpu"));
    let coop = gctx.has_coop() && std::env::var("ARF_NO_COOP").is_err();
    eprintln!(
        "device {} | weights {quant:?} | coop-matrix path: {} | gen {gen}/seq\n",
        gctx.info(),
        if coop { "ON" } else { "OFF" }
    );

    let model: GpuModel = arf_gpu::weights::load_safetensors_gpu_kv_quant(
        &cfg,
        &paths,
        max_ctx,
        &gctx,
        quant,
        KvQuant::None,
    )
    .expect("load");

    let short = vec![1u32, 7, 3, 9, 4, 2, 8, 5];
    let _ = model.generate_batch(std::slice::from_ref(&short), 4); // warm

    println!("{:>4} {:>14} {:>14}", "N", "agg tok/s", "per-seq tok/s");
    for &n in &ns {
        let prompts = vec![short.clone(); n];
        let t = Instant::now();
        let _ = model.generate_batch(&prompts, gen);
        let secs = t.elapsed().as_secs_f64();
        // Decode work = N sequences × gen tokens each (continuous batch lockstep).
        let agg = (n * gen) as f64 / secs;
        println!("{n:>4} {agg:>14.1} {:>14.1}", agg / n as f64);
    }
}
