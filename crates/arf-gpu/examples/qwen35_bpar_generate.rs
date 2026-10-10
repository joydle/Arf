//! L141 — drive `generate_qwen35_batched`, the B-PARALLEL gated-delta-net engine.
//!
//! WHY THIS EXISTS. `ssm_qwen35.rs` ships TWO GPU generation paths and only one had a driver:
//!   `generate_qwen35`          — m=1, per-op host round-trip. The restored `qwen35_generate`
//!                                example drives this. It is CORRECT but slow by construction
//!                                (see the M6 header at ssm_qwen35.rs:94).
//!   `generate_qwen35_batched`  — the B-parallel path: `m7_linear_layer_b` / `m7_gqa_layer_b`,
//!                                ONE CommandPass per layer, ONE submit, no poll, with the
//!                                recurrent conv/ssm state resident and sized xB. NO CALLER.
//!
//! This driver closes that gap so the fast path can be measured and text-gated. Same tokenizer +
//! BOS handling as `qwen35_generate`, so the two are directly comparable on one prompt.
//!
//! Run: cargo run --release -p arf-gpu --example qwen35_bpar_generate -- [gguf] [prompt] [n_new]

#[cfg(target_os = "macos")]
fn main() {
    use arf_core::config::ModelConfig;
    use arf_core::model::gguf::LazyGguf;
    use arf_core::tokenizer::Tokenizer;
    use arf_gpu::gpu::ssm_qwen35::{generate_qwen35_batched, read_hparams};
    use arf_gpu::gpu::GpuContext;

    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf".to_string());
    let prompt = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "What is 2+2?".to_string());
    let n_new: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);

    println!("opening GGUF: {path}");
    let g = LazyGguf::open_raw(std::path::Path::new(&path)).expect("open_raw");
    let cfg = ModelConfig::qwen35_27b();
    let hp = read_hparams(&g, &cfg).expect("read_hparams");
    println!(
        "hparams: n_embd={} n_head={} n_head_kv={} head_dim={} layers={} (nextn {}) vocab={}",
        hp.n_embd,
        hp.n_head,
        hp.n_head_kv,
        hp.n_embd_head_k,
        hp.n_layer,
        hp.n_layer_nextn,
        hp.n_vocab
    );

    let tok = Tokenizer::from_gguf_path(&path).expect("tokenizer");
    let mut ids = tok.encode(&prompt, false).expect("encode");
    let bos = g
        .get_metadata_u32("tokenizer.ggml.bos_token_id")
        .unwrap_or(248044);
    ids.insert(0, bos);
    println!("prompt: {prompt:?}\ntokens ({}): {:?}", ids.len(), ids);

    let ctx = std::sync::Arc::new(GpuContext::new().expect("gpu context"));

    // B=1: the same single stream the m=1 driver runs, but through the B-parallel kernels, so any
    // text difference between the two drivers isolates to the _b path itself.
    let t0 = std::time::Instant::now();
    let out = generate_qwen35_batched(&g, &hp, std::slice::from_ref(&ids), n_new, &ctx, None)
        .expect("generate_qwen35_batched");
    let secs = t0.elapsed().as_secs_f64();

    let gen = &out[0];
    let text = tok.decode(gen, false).unwrap_or_default();
    let tps = gen.len() as f64 / secs;
    println!(
        "\n=== B-PARALLEL DONE ({} tokens in {secs:.1}s, {tps:.2} tok/s) ===",
        gen.len()
    );
    println!("gen ids: {gen:?}");
    println!("\n----- FULL TEXT -----\n{prompt}{text}\n---------------------");
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("qwen35_bpar_generate is macOS-only (Metal island). No-op on this platform.");
}
