//! Long-context KV-cache bench: decode throughput and KV footprint for f32 vs
//! TurboQuant-packed KV (tq4/tq3/tq2). Everything except the KV path is identical
//! (same weights, same prompt), so the tok/s delta isolates the attention-read
//! bandwidth / cache-footprint win — the metric that bounds long-context decode
//! (ollama-style single stream) and large-batch serving (vLLM-style).
//!
//! Run (real weights):
//!   ARF_MODEL_PATH=models/llama-3.2-1b \
//!   CTX="1024,4096,8192" GEN=64 MAX_CTX=9000 \
//!   cargo run --release -p arf-gpu --example gpu_kv_quant_bench
//!
//! Reports, per (kv-mode, ctx): KV-pool bytes, decode tok/s, and the speedup vs
//! the f32 baseline. Token ids are synthetic (throughput, not text quality —
//! quality drift is covered by the parity tests).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use arf_core::config::{KvQuant, ModelConfig, Quant};
use arf_gpu::gpu::{GpuContext, GpuModel};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Resident KV-pool bytes for a given mode at `max_ctx` tokens (one sequence):
/// f32 stores 2·layers·slots·kv_dim f32; Tq stores packed codes (bits/coord) plus
/// one f32 norm per head-vector, ×2 for keys+values.
fn kv_pool_bytes(cfg: &ModelConfig, max_ctx: usize, mode: KvQuant) -> usize {
    let block = 16;
    let slots = max_ctx.div_ceil(block) * block;
    let kv_dim = cfg.num_kv_heads * cfg.head_dim;
    match mode {
        KvQuant::None => 2 * cfg.num_layers * slots * kv_dim * 4,
        KvQuant::Tq { bits, .. } => {
            let n_vectors = slots * cfg.num_kv_heads;
            let code_bytes = (n_vectors * cfg.head_dim * bits as usize).div_ceil(8);
            let norm_bytes = n_vectors * 4;
            2 * cfg.num_layers * (code_bytes + norm_bytes)
        }
    }
}

fn load(
    cfg: &ModelConfig,
    paths: &[&Path],
    max_ctx: usize,
    ctx: &std::sync::Arc<GpuContext>,
    quant: Quant,
    kv: KvQuant,
) -> GpuModel {
    arf_gpu::weights::load_safetensors_gpu_kv_quant(cfg, paths, max_ctx, ctx, quant, kv)
        .expect("load")
}

fn main() {
    let dir = PathBuf::from(std::env::var("ARF_MODEL_PATH").expect("set ARF_MODEL_PATH"));
    let cfg = ModelConfig::llama_3_2_1b();
    let mf = dir.join("model.safetensors");
    let paths: [&Path; 1] = [mf.as_path()];

    let ctxs: Vec<usize> = std::env::var("CTX")
        .unwrap_or_else(|_| "1024,4096,8192".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let gen = env_usize("GEN", 64);
    let max_ctx = env_usize(
        "MAX_CTX",
        ctxs.iter().copied().max().unwrap_or(8192) + gen + 64,
    );
    let quant = match std::env::var("QUANT").as_deref() {
        Ok("q4") => Quant::Q4,
        Ok("int8") => Quant::Int8,
        _ => Quant::None,
    };

    let gctx = std::sync::Arc::new(GpuContext::new().expect("gpu"));
    eprintln!(
        "device {} | weights {quant:?} | max_ctx {max_ctx} | gen {gen}/run\n",
        gctx.info()
    );

    let modes = [
        ("f32", KvQuant::None),
        (
            "tq4",
            KvQuant::Tq {
                bits: 4,
                qjl: false,
            },
        ),
        (
            "tq3",
            KvQuant::Tq {
                bits: 3,
                qjl: false,
            },
        ),
        (
            "tq2",
            KvQuant::Tq {
                bits: 2,
                qjl: false,
            },
        ),
    ];

    // Context grows by INCREMENTAL DECODE (q_len=1 per step) from a short prompt —
    // never a giant single-shot prefill (that blows past the 65535-workgroup
    // dispatch cap and isn't the decode metric anyway). For each target `c` we time
    // ONE from-scratch decode of `c` tokens and report the average decode tok/s
    // over context 0→c. One run per (mode, c) → stable (no run-to-run subtraction);
    // the same-`c` comparison across modes is apples-to-apples (only the KV path
    // differs). `gen` is unused here (kept for the env contract).
    let _ = gen;
    let short = vec![1u32, 7, 3, 9, 4, 2, 8, 5];

    println!(
        "{:>5} {:>6} {:>12} {:>10} {:>10} {:>8}",
        "mode", "ctx", "kv-bytes", "kv-vs-f32", "dec-tok/s", "speedup"
    );
    let f32_bytes = kv_pool_bytes(&cfg, max_ctx, KvQuant::None) as f64;
    let mut f32_tps: std::collections::HashMap<usize, f64> = Default::default();
    for (label, mode) in modes {
        let model = load(&cfg, &paths, max_ctx, &gctx, quant, mode);
        let _ = model.generate(&short, 8); // warm shaders/clocks
        for &c in &ctxs {
            if c + short.len() >= max_ctx {
                continue;
            }
            let t = Instant::now();
            let _ = model.generate(&short, c);
            let secs = t.elapsed().as_secs_f64();
            let tps = c as f64 / secs.max(1e-6);

            let bytes = kv_pool_bytes(&cfg, max_ctx, mode);
            if matches!(mode, KvQuant::None) {
                f32_tps.insert(c, tps);
            }
            let speedup = f32_tps.get(&c).map(|b| tps / b).unwrap_or(1.0);
            println!(
                "{label:>5} {c:>6} {:>10.1}MB {:>8.2}x {tps:>10.1} {speedup:>7.2}x",
                bytes as f64 / 1.0e6,
                bytes as f64 / f32_bytes,
            );
            std::io::stdout().flush().ok();
        }
    }
}
