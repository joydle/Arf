//! A/B tok/s bench for the Gemma speed levers — loads a gemma GGUF ONCE, warms up, and times
//! `n` single-stream decode tokens (median of `REPS` passes) on the native-Metal megakernel path.
//! Prints ONE canonical parseable line so a runner can diff lever flags across processes.
//!
//! The lever flags are read at LOAD/COMPILE time (ARF_ATTN_VEC4 specializes attention_decode;
//! ARF_GEMMA_FUSE_ADDNORM2 / ARF_Q4_DOTY / ARF_SSM_CONV_BATCHED gate at record/compile),
//! so each flag configuration must be a FRESH PROCESS. Drive with an A/B script, which sets
//! one config per invocation under a hard timeout. This bench does NOT set the flags itself.
//!
//! Run (single config):
//!   TARGET=models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf N=64 REPS=3 \
//!   ARF_MSL_GEMV=1 ARF_MEGAKERNEL=1 \
//!     cargo run --release -p arf-gpu --example gemma_ab
//!
//! Output line (stdout, exactly one):
//!   GEMMA_AB tokps=<median> quant=<q> n=<n> reps=<r> flags=[<active ARF_* levers>]

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
    // ARCH selects the config (default gemma-4-12b). gemma3-4b supported for a cheaper A/B.
    let arch = env("ARCH").unwrap_or_else(|| "gemma-4-12b".into());
    let (cfg, default_target): (ModelConfig, &str) = match arch.as_str() {
        "gemma-4-12b" | "gemma4-12b" => (
            ModelConfig::gemma4_12b(),
            "models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf",
        ),
        "gemma3-4b" | "gemma3" => (
            ModelConfig::gemma3_4b(),
            "models/gemma-3-4b-vision/gemma-3-4b-it-Q4_0.gguf",
        ),
        other => panic!("unknown ARCH {other:?} (gemma-4-12b | gemma3-4b)"),
    };
    let target = PathBuf::from(env("TARGET").unwrap_or_else(|| default_target.into()));
    assert!(target.exists(), "model not found: {target:?}");

    // Q4_0 QAT GGUFs load via the Q4K native path (dense loader self-quants Q4_0→Q4_K lossless-ish).
    let quant = match env("QUANT").as_deref() {
        Some("q4") => Quant::Q4,
        Some("q4k") | None => Quant::Q4K,
        Some("q4ks") => Quant::Q4KS,
        Some("bf16") | Some("none") => Quant::None,
        Some(o) => panic!("unknown QUANT {o:?}"),
    };
    let n = env_usize("N", 64);
    let reps = env_usize("REPS", 3);
    let max_ctx = env_usize("MAX_CTX", 2048);

    // Report which lever flags are active this process (for the output line + provenance).
    let levers: Vec<&str> = [
        ("ARF_ATTN_VEC4", "vec4"),
        ("ARF_GEMMA_FUSE_ADDNORM2", "normpair"),
        ("ARF_Q4_DOTY", "doty"),
        ("ARF_GEMV_NSG2", "nsg2"),
        ("ARF_SSM_CONV_BATCHED", "convf4"),
    ]
    .iter()
    .filter(|(k, _)| std::env::var_os(k).is_some())
    .map(|(_, tag)| *tag)
    .collect();
    let flags = if levers.is_empty() {
        "baseline".to_string()
    } else {
        levers.join("+")
    };

    let ctx = std::sync::Arc::new(GpuContext::new().expect("gpu"));
    eprintln!(
        "[gemma_ab] {arch} {quant:?} flags=[{flags}] device={}",
        ctx.info()
    );
    eprint!(
        "[gemma_ab] loading {:.1}GB... ",
        std::fs::metadata(&target)
            .map(|m| m.len() as f64 / 1e9)
            .unwrap_or(0.0)
    );
    let t = Instant::now();
    // The RAM guard in the loader refuses an oversized wired footprint (safety on a 36GB Mac).
    let model: GpuModel =
        arf_gpu::weights::load_gguf_gpu_quant(&cfg, &target, max_ctx, &ctx, quant)
            .expect("load gguf");
    eprintln!("loaded in {:.1}s", t.elapsed().as_secs_f64());

    let tok: Option<Tokenizer> = env("TOKENIZER")
        .map(PathBuf::from)
        .and_then(|p| Tokenizer::from_file(&p).ok())
        .or_else(|| Tokenizer::from_gguf_path(&target).ok());

    // gemma-4 instruct turn template (same as agentic_proof); falls back to seed ids.
    let prompt_text = env("PROMPT").unwrap_or_else(|| {
        "List three steps to diagnose a service returning HTTP 500 on every request.".into()
    });
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
        None => vec![2u32, 818, 1956, 6914, 7233, 708],
    };

    // Warm: load lazies + JIT the island pipelines (the levers' first-use compile happens here,
    // OUTSIDE the timed window — the warm-up lesson from the verify-kernel measurement).
    let _ = model.generate(&prompt, 8);

    // Time REPS passes, take the MEDIAN (robust to a thermal blip vs a mean).
    let mut tokps: Vec<f64> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        let out = model.generate(&prompt, n);
        let dt = t.elapsed().as_secs_f64();
        tokps.push(out.len() as f64 / dt);
    }
    tokps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = tokps[tokps.len() / 2];

    // ONE canonical parseable line (stdout).
    println!(
        "GEMMA_AB tokps={median:.2} quant={quant:?} n={n} reps={reps} flags=[{flags}] all={:?}",
        tokps.iter().map(|x| format!("{x:.1}")).collect::<Vec<_>>()
    );
}
