//! Wide in-process correctness sweep for the Gemma coop-GEMM unlock. Loads the
//! real GGUF ONCE, then for a broad prompt suite generates greedily under the
//! bit-exact GEMV (ARF_COOP_MAX_K=256) and the cooperative-matrix GEMM
//! (ARF_COOP_MAX_K=4096) and reports token-for-token match. This estimates the
//! true reassociation flip-rate far faster than the shell harness (which reloaded
//! the 12 GB model every invocation).
//!
//! The model uses the batched server path (`generate_batch`) at batch=BATCH so the
//! coop GEMM (m>=8) is actually exercised — single-stream m=1 never hits coop.
//!
//! Usage: cargo run --release -p arf-gpu --example coop_gemma_match -- <gguf>
//! Env: ARF_CGM_BATCH (default 8), ARF_CGM_NTOK (default 24).

use arf_core::config::{KvQuant, ModelConfig, Quant};
use std::io::Write;
use std::sync::Arc;

fn flush() {
    let _ = std::io::stdout().flush();
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: coop_gemma_match <gguf>");
    let batch: usize = std::env::var("ARF_CGM_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let ntok: usize = std::env::var("ARF_CGM_NTOK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let nproto: usize = std::env::var("ARF_CGM_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);
    // ARF_CGM_PLEN: pad each prompt body to ~this many tokens (long-context test:
    // coop's reassociation accumulates over more tokens, and the per-step drift
    // compounds over more decode steps — long prompt + long ntok is the hard case).
    let plen: usize = std::env::var("ARF_CGM_PLEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let ctx = Arc::new(arf_gpu::gpu::GpuContext::new().expect("gpu"));
    let cfg = ModelConfig::gemma4_12b();
    // KV pool must hold batch × ceil((prompt+ntok)/block) blocks. Size for the long
    // case: batch × (prompt_len + ntok + slack), block_size=16.
    let max_ctx = (batch * (plen + ntok + 64)).max(512);
    let m = arf_gpu::weights::load_gguf_gpu_kv_quant(
        &cfg,
        std::path::Path::new(&path),
        max_ctx,
        &ctx,
        Quant::Q4K,
        KvQuant::None,
    )
    .expect("load");

    // A wide suite of short factual/closed prompts (greedy → most likely to hit
    // near-tie argmax decisions, where the reassociation delta could flip a token).
    // Pre-tokenized chat-templated bodies would need the tokenizer; instead reuse
    // the raw-token chat scaffold and vary the body token. We approximate breadth by
    // sweeping the FIRST body token across a spread of vocab ids — each yields a
    // different greedy trajectory, exercising many distinct argmax decisions.
    let scaffold_pre: Vec<u32> = vec![2, 105, 2364, 107]; // <bos><|turn>user\n
    let scaffold_post: Vec<u32> = vec![106, 107, 105, 4368, 107]; // <turn|>\n<|turn>model\n
                                                                  // N distinct body seeds → N distinct prompts/trajectories (ARF_CGM_N).
    let body_seeds: Vec<u32> = (0..nproto as u32).map(|i| 1000 + i * 211).collect();

    let gen_at = |coop_max_k: &str, prompt: &[u32]| -> Vec<u32> {
        std::env::set_var("ARF_COOP_MAX_K", coop_max_k);
        // generate_batch takes a slice of distinct prompts; pass `batch` copies of
        // the same prompt so the coop GEMM (m = batch >= 8) is exercised, take seq 0.
        let prompts: Vec<Vec<u32>> = (0..batch).map(|_| prompt.to_vec()).collect();
        let outs = m.generate_batch(&prompts, ntok);
        outs.into_iter().next().unwrap_or_default()
    };

    let mut matches = 0usize;
    let mut diffs = 0usize;
    let total = body_seeds.len();
    println!("### coop_gemma_match — batch={batch} plen={plen} ntok={ntok} prompts={total}");
    flush();
    for (i, &seed) in body_seeds.iter().enumerate() {
        let mut prompt = scaffold_pre.clone();
        prompt.push(seed);
        // Long-context filler: deterministic pseudo-text body of `plen` tokens, varied
        // by seed so each trajectory differs. Kept in a safe vocab range (avoid
        // specials). 0 = short prompt (original behavior).
        for j in 0..plen {
            prompt.push(500 + ((seed.wrapping_mul(31).wrapping_add(j as u32 * 17)) % 20000));
        }
        prompt.extend_from_slice(&scaffold_post);
        let gemv = gen_at("256", &prompt);
        let coop = gen_at("4096", &prompt);
        if gemv == coop {
            matches += 1;
            println!("[MATCH {:>2}] seed={seed}", i + 1);
        } else {
            diffs += 1;
            // first divergent position
            let fd = gemv.iter().zip(&coop).position(|(a, b)| a != b);
            println!("[DIFF  {:>2}] seed={seed} first_diff_tok={fd:?}", i + 1);
            println!("   GEMV: {gemv:?}");
            println!("   COOP: {coop:?}");
        }
        flush();
    }
    println!(
        "### RESULT: {matches}/{total} match, {diffs}/{total} diff (flip-rate {diffs}/{total})"
    );
    flush();
}
