//! GATE for the MoE megakernel wiring: load the qwen3-coder-30B MoE GGUF, run SINGLE-STREAM
//! greedy `generate` (the path that reaches `run_megakernel` via decode_token — NOT the batched
//! `forward_batch` serve_loop_bench uses), and report (a) whether the megakernel fired, (b) the
//! generated token stream (coherence: identical greedy ids with the megakernel ON vs OFF), and
//! (c) decode tok/s (the speed gate: must beat the ~13 tok/s serial wgpu MoE path).
//!
//! Run (megakernel ON):
//!   ARF_MSL_GEMV=1 ARF_MEGAKERNEL=1 ARF_MEGA_SINGLEQ=1 \
//!   GGUF=<blob> GEN=24 cargo run --release -p arf-gpu --example mega_moe_gate
//! Run (oracle, wgpu MoE path — drop the megakernel envs):
//!   ARF_MSL_GEMV=1 GGUF=<blob> GEN=24 cargo run --release -p arf-gpu --example mega_moe_gate

#[cfg(target_os = "macos")]
fn main() {
    use arf_core::config::{KvQuant, ModelConfig, Quant};
    use arf_gpu::gpu::GpuContext;
    use std::path::PathBuf;
    use std::time::Instant;

    let gguf = std::env::var("GGUF").expect("set GGUF=<qwen3-coder-30B blob>");
    let gen = std::env::var("GEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(24usize);
    let quant = match std::env::var("QUANT").as_deref() {
        Ok("q4k") => Quant::Q4K,
        _ => Quant::Q4KS, // the megakernel indirect GEMV needs Q4KS .mtl
    };
    let cfg = ModelConfig::qwen3_coder_30b();
    let max_ctx = (16 + gen + 16).next_multiple_of(16);
    let block_size = 16usize;
    let num_blocks = max_ctx.div_ceil(block_size);

    let ctx = std::sync::Arc::new(GpuContext::new().expect("gpu"));
    eprintln!("device {} | weights {quant:?} | gen {gen}", ctx.info());
    let mega_on = std::env::var_os("ARF_MEGAKERNEL").is_some();
    eprintln!(
        "megakernel envs: ARF_MEGAKERNEL={} ARF_MEGA_SINGLEQ={} ARF_MSL_GEMV={}",
        mega_on,
        std::env::var_os("ARF_MEGA_SINGLEQ").is_some(),
        std::env::var_os("ARF_MSL_GEMV").is_some()
    );

    eprintln!("loading qwen3-coder-30B MoE GGUF ({quant:?})...");
    let t_load = Instant::now();
    let model = arf_gpu::weights::load_gguf_gpu_kv_quant_pooled(
        &cfg,
        &PathBuf::from(&gguf),
        max_ctx,
        &ctx,
        quant,
        KvQuant::None,
        Some(num_blocks),
    )
    .expect("load gguf");
    eprintln!("LOAD TIME: {:.2}s", t_load.elapsed().as_secs_f64());

    // A short deterministic prompt (raw token ids — no tokenizer dependency; coherence is the
    // GREEDY ID MATCH between megakernel ON and OFF, both reading the same weights).
    let prompt: Vec<u32> = vec![
        151644, 872, 198, 9707, 11, 1879, 0, 151645, 198, 151644, 77091, 198,
    ];

    // Warm (one short generate) so the island compiles + buffers alloc outside the timed run.
    let _ = model.generate(&prompt, 2);

    let t = Instant::now();
    let out = model.generate(&prompt, gen);
    let dt = t.elapsed().as_secs_f64();
    let toks = out.len();
    let tps = toks as f64 / dt;

    println!(
        "\n=== MoE megakernel gate (megakernel {}) ===",
        if mega_on { "ON" } else { "OFF (wgpu oracle)" }
    );
    println!("generated {toks} tokens in {dt:.3}s = {tps:.2} tok/s");
    println!("ids: {:?}", out);
    // Coherence sniff: a degenerate stream (all-same id, or a 1-2 token loop) = the MoE path is
    // broken even if it ran. A varied stream that MATCHES the wgpu oracle's ids = correct.
    let distinct: std::collections::BTreeSet<u32> = out.iter().copied().collect();
    println!("distinct ids: {} / {}", distinct.len(), toks);
    if distinct.len() <= 2 && toks > 4 {
        println!(
            "WARNING: near-degenerate stream — likely INCOHERENT (compare to the OFF run's ids)."
        );
    } else {
        println!("stream varied (>2 distinct) — diff these ids against the megakernel-OFF run for exact coherence.");
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("mega_moe_gate is macOS-only.");
}
