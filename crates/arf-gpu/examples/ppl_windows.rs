//! TEACHER-FORCED PERPLEXITY through the path that serves a hybrid model: the windowed prefill's
//! batched record, 16 rows at a time, scoring each true next token from the record's own logits.
//!
//! This is the quality instrument for any change to the NUMBERS in the weights (a quantiser, a
//! regrouping, a re-fit). `arf perplexity` cannot be it for Qwen3.8-27B: its all-positions path
//! reads 2.7e6 on plain English. And the block draft's acceptance cannot be it either — on
//! 2026-09-21 it "found" a fidelity bug on one prompt that sixteen prompts did not confirm.
//! Compare two builds on the SAME text and window; against `llama-perplexity` only loosely (it
//! chunks and scores differently).
//!
//! Run: cargo run --release -p arf-gpu --example ppl_windows -- <text file> [gguf] [window=16]

#[cfg(target_os = "macos")]
fn main() {
    use arf_core::config::{KvQuant, ModelConfig, Quant};
    use arf_gpu::gpu::GpuContext;
    use std::sync::Arc;

    let arg = |i: usize, d: &str| std::env::args().nth(i).unwrap_or_else(|| d.into());
    let text = std::fs::read_to_string(arg(1, "docs/fixtures/ppl_sample.txt")).expect("text file");
    let gguf = arg(2, "models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf");
    let win: usize = arg(3, "16").parse().expect("window");
    arf_gpu::apply_fast_path_defaults("ppl_windows");
    std::env::remove_var("ARF_KV_F16");
    std::env::set_var("ARF_NO_KV_F16", "1");

    let tok = arf_core::tokenizer::Tokenizer::from_gguf_path(&gguf).expect("gguf tokenizer");
    let ids = tok.encode(&text, false).expect("encode");
    let n = ids.len();
    let cfg = ModelConfig::qwen35_27b();
    let blocks = n.div_ceil(16) + 2;
    let ctx = Arc::new(GpuContext::new().expect("gpu context"));
    eprintln!("{n} tokens; loading {gguf} ...");
    let model = arf_gpu::weights::load_gguf_gpu_kv_quant_pooled(
        &cfg,
        std::path::Path::new(&gguf),
        blocks * 16,
        &ctx,
        Quant::Q4KS,
        KvQuant::None,
        Some(blocks),
    )
    .expect("load gguf");
    // One sequence owning the pool from slot 0: position p lives in slot p.
    let slots: Vec<u32> = (0..(blocks * 16) as u32).collect();
    let (mut nll, mut hits, mut scored) = (0.0f64, 0usize, 0usize);
    // Mean NLL by the row's position INSIDE its window (2026-09-23): a fault that corrupts only
    // late rows of wide windows (the >= 120-row prefill bug) shows up as one bucket blowing up.
    let buckets: [(usize, usize); 4] = [(0, 64), (64, 96), (96, 120), (120, 128)];
    let mut bnll = [0.0f64; 4];
    let mut bn = [0usize; 4];
    let t = std::time::Instant::now();
    let mut at = 0usize;
    while at + 1 < n {
        let k = win.min(n - 1 - at);
        let lp = model
            .window_logprobs(
                &ids[at..at + k],
                at,
                &slots[..at + k],
                1,
                &ids[at + 1..at + 1 + k],
            )
            .expect("the windowed record declined — see ARF_PREFILL_FAST_DEBUG");
        for (r, (l, hit)) in lp.into_iter().enumerate() {
            if let Some(bi) = buckets.iter().position(|&(a, b)| r >= a && r < b) {
                bnll[bi] -= l as f64;
                bn[bi] += 1;
            }
            nll -= l as f64;
            hits += hit as usize;
            scored += 1;
        }
        at += k;
    }
    println!(
        "{scored} tokens scored in {:.1} s, window {win}: perplexity {:.4} (mean NLL {:.5} nats), top-1 agreement with the text {:.2}%",
        t.elapsed().as_secs_f64(),
        (nll / scored as f64).exp(),
        nll / scored as f64,
        100.0 * hits as f64 / scored as f64
    );
    for (i, (a, b)) in buckets.iter().enumerate() {
        if bn[i] > 0 {
            println!(
                "  rows {a:3}..{b:3} of the window: {:5} tokens, mean NLL {:.4}",
                bn[i],
                bnll[i] / bn[i] as f64
            );
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("ppl_windows drives the native-Metal island; macOS only.");
}
