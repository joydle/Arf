//! PROOF: the modern agentic gemma-4-12B (yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2
//! -3.5x-tau2 — the tau2-bench agentic fine-tune, ships its MTP draft) runs FAST on Arf's
//! native-Metal megakernel engine. Loads the real model, proves coherent generation, and reports
//! single-stream tok/s on the megakernel path + the speculative (n-gram batched-coop verify) path.
//!
//! Run (files already on disk from the HF repo's gemma4-v2-Q4_K_M.gguf + MTP-Q8_0.gguf):
//!   TARGET=$TMPDIR/g4-agentic/gemma4-v2-Q4_K_M.gguf \
//!   ARF_MSL_GEMV=1 ARF_MEGAKERNEL=1 ARF_MEGA_SINGLEQ=1 ARF_MEGA_NOWAIT=1 \
//!     cargo run --release -p arf-gpu --example agentic_proof

use std::path::PathBuf;
use std::time::Instant;

use arf_core::config::{ModelConfig, Quant};
use arf_core::Tokenizer;
use arf_gpu::gpu::{GpuContext, GpuModel};

fn env(k: &str) -> Option<String> {
    std::env::var(k).ok()
}
fn env_usize(k: &str, d: usize) -> usize {
    env(k).and_then(|s| s.parse().ok()).unwrap_or(d)
}

fn main() {
    let target = env("TARGET").map(PathBuf::from).unwrap_or_else(|| {
        std::env::temp_dir()
            .join("g4-agentic")
            .join("gemma4-v2-Q4_K_M.gguf")
    });
    assert!(target.exists(), "model not found: {target:?}\n(pull yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF → gemma4-v2-Q4_K_M.gguf)");

    let n = env_usize("N", 64);
    let max_ctx = env_usize("MAX_CTX", 2048);
    let prompt_text = env("PROMPT").unwrap_or_else(|| {
        "You are a terminal debugging agent. List the first three steps to diagnose a service \
         that returns HTTP 500 on every request."
            .into()
    });

    let ctx = std::sync::Arc::new(GpuContext::new().expect("gpu"));
    let cfg: ModelConfig = ModelConfig::gemma4_12b();
    println!("=== Arf · modern agentic gemma-4-12B proof ===");
    println!("device: {}", ctx.info());
    println!(
        "model:  {} (agentic-tau2 fine-tune, Q4_K_M)",
        target.display()
    );

    eprint!(
        "loading {:.1}GB... ",
        std::fs::metadata(&target)
            .map(|m| m.len() as f64 / 1e9)
            .unwrap_or(0.0)
    );
    let t = Instant::now();
    let model: GpuModel =
        arf_gpu::weights::load_gguf_gpu_quant(&cfg, &target, max_ctx, &ctx, Quant::Q4K)
            .expect("load gguf");
    println!("loaded in {:.1}s", t.elapsed().as_secs_f64());

    // Tokenizer: sidecar if present, else GGUF-embedded, else seed ids (still proves throughput).
    let tok: Option<Tokenizer> = env("TOKENIZER")
        .map(PathBuf::from)
        .or_else(|| {
            let s = PathBuf::from("/tmp/g4-12b-st/tokenizer.json");
            s.exists().then_some(s)
        })
        .and_then(|p| Tokenizer::from_file(&p).ok())
        .or_else(|| Tokenizer::from_gguf_path(&target).ok());

    // gemma-4 is INSTRUCT: wrap in the turn template (marker SPECIAL-token ids 105/106).
    let prompt: Vec<u32> = match &tok {
        Some(t) => {
            let id = |s: &str| t.token_to_id(s);
            let nl: Vec<u32> = id("\n").into_iter().collect();
            let mut ids: Vec<u32> = id("<bos>").into_iter().collect();
            if let Some(turn) = id("<|turn>") {
                ids.push(turn);
                ids.extend(t.encode("user", false).unwrap_or_default());
                ids.extend(&nl);
            }
            ids.extend(t.encode(prompt_text.as_str(), false).expect("encode"));
            if let Some(turne) = id("<turn|>") {
                ids.push(turne);
            }
            ids.extend(&nl);
            if let Some(turn) = id("<|turn>") {
                ids.push(turn);
                ids.extend(t.encode("model", false).unwrap_or_default());
                ids.extend(&nl);
            }
            ids
        }
        None => {
            println!("(no tokenizer — seeding ids; throughput still real, text not decodable)");
            vec![2u32, 818, 1956, 6914, 7233, 708]
        }
    };
    println!("prompt: {} tokens\n", prompt.len());

    // warm the clocks (load lazies, JIT the island pipelines).
    let _ = model.generate(&prompt, 8);

    // === PROOF 1: single-stream megakernel generate — coherence + tok/s ===
    let t = Instant::now();
    let out = model.generate(&prompt, n);
    let dt = t.elapsed().as_secs_f64();
    let toks = out.len();
    println!("── single-stream (megakernel: Concurrent+DEPBAR+fused-attn) ──");
    println!(
        "  {toks} tokens in {dt:.2}s = {:.1} tok/s",
        toks as f64 / dt
    );
    if let Some(t) = &tok {
        let text = t
            .decode(&out, true)
            .unwrap_or_else(|_| "<decode failed>".into());
        println!(
            "  output: \"{}\"",
            text.chars().take(280).collect::<String>()
        );
    }

    // === PROOF 2: speculative (n-gram + batched-coop verify) — the agentic-workload lever ===
    let t = Instant::now();
    let (sp, fwd, drafted, accepted) = model.generate_speculative(&prompt, n, 4, 3);
    let sdt = t.elapsed().as_secs_f64();
    println!("\n── speculative (n-gram k=4, batched-coop verify) ──");
    println!(
        "  {} tokens in {sdt:.2}s = {:.1} tok/s | {fwd} forwards | accept {:.0}% | {:.2} tok/fwd",
        sp.len(),
        sp.len() as f64 / sdt,
        if drafted > 0 {
            100.0 * accepted as f64 / drafted as f64
        } else {
            0.0
        },
        sp.len() as f64 / fwd.max(1) as f64,
    );
    let spec_speedup = (sp.len() as f64 / sdt) / (toks as f64 / dt);
    println!("  speculative speedup: {spec_speedup:.2}× vs single-stream greedy");

    println!("\n✓ PROOF: modern agentic gemma-4-12B runs on Arf's megakernel engine.");
}
