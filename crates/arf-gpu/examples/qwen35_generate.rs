//! M5 GATE: REAL TEXT GENERATION on the Qwen3.6-27B (`qwen35`) hybrid gated-delta-net beast.
//!
//! Loads the model ONCE, tokenizes a prompt (GGUF-embedded tokenizer + BOS), prefills it,
//! then generates `n_new` tokens AUTOREGRESSIVELY (greedy argmax) — carrying the recurrent
//! state INCREMENTALLY (option B): the 48 LINEAR layers advance O(1) per token (conv_state +
//! ssm_state in place, no growing KV); the 16 FULL-GQA layers keep the standard growing KV
//! cache. The three new gated-delta-net ops dispatch on the Metal island; GEMVs / norms /
//! attention / FFN run on the proven host code (M4b reference speed — SLOW but correct).
//!
//! THE M5 GATE: coherent generated text on the real model. For "The capital of France is"
//! the continuation should be sensible (" Paris. …"). The first generated token must match
//! the M4a/M4b known "Paris" — if it then degrades, the bug is in step-to-step state.
//!
//! Run (macOS / Metal):
//!   cargo run --release -p arf-gpu --example qwen35_generate -- [gguf] [prompt] [n_new]

#[cfg(target_os = "macos")]
fn main() {
    use arf_core::config::ModelConfig;
    use arf_core::model::gguf::LazyGguf;
    use arf_core::tokenizer::Tokenizer;
    use arf_gpu::gpu::concurrent_metal::MetalIsland;
    use arf_gpu::gpu::ssm_qwen35::{generate_qwen35, read_hparams};
    use arf_gpu::gpu::GpuContext;

    let path = std::env::args().nth(1).unwrap_or_else(|| {
        format!(
            "{}/models/qwen3.6-27b-mtp/Qwen3.6-27B-Q4_K_S.gguf",
            std::env::var("HOME").unwrap()
        )
    });
    let prompt = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "The capital of France is".to_string());
    let n_new: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);

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

    // Tokenize (GGUF-embedded tokenizer). Add BOS so the prompt matches a normal prefill.
    let tok = Tokenizer::from_gguf_path(&path).expect("tokenizer");
    let mut ids = tok.encode(&prompt, false).expect("encode");
    let bos = g
        .get_metadata_u32("tokenizer.ggml.bos_token_id")
        .unwrap_or(248044);
    ids.insert(0, bos);
    println!("prompt: {prompt:?}\ntokens ({}): {:?}", ids.len(), ids);
    println!("generating {n_new} tokens (greedy, incremental O(1) linear state)...\n");

    let ctx = std::sync::Arc::new(GpuContext::new().expect("gpu context"));
    let mut island = MetalIsland::new(&ctx).expect("metal island (need a Metal backend)");

    // Stream each token's decode as it is produced (so progress is visible on a slow forward).
    let t0 = std::time::Instant::now();
    let mut step = 0usize;
    let gen_ids = {
        let tok_ref = &tok;
        let mut on_token = move |id: u32| {
            step += 1;
            let piece = tok_ref.decode(&[id], false).unwrap_or_default();
            let secs = t0.elapsed().as_secs_f64();
            println!("  [{step:2}] id={id:7}  {piece:?}   (t={secs:.1}s)");
        };
        generate_qwen35(&g, &hp, &ids, n_new, &mut island, &ctx, Some(&mut on_token))
            .expect("generate")
    };
    let secs = t0.elapsed().as_secs_f64();

    // Detokenize the full continuation and print prompt + generated text.
    let cont = tok.decode(&gen_ids, false).unwrap_or_default();
    let tps = gen_ids.len() as f64 / secs.max(1e-9);
    println!(
        "\n=== GENERATION DONE ({} tokens in {secs:.1}s, {tps:.2} tok/s) ===",
        gen_ids.len()
    );
    println!("gen ids: {gen_ids:?}");
    println!("\n----- FULL TEXT -----\n{prompt}{cont}\n---------------------");

    // M5 GATE check: the FIRST generated token should be the known "Paris" continuation for
    // the default prompt (M4a/M4b proved the single forward). Soft-assert (print, don't panic
    // on a non-default prompt) but hard-check for the canonical France prompt.
    if prompt == "The capital of France is" {
        let first = tok
            .decode(&gen_ids[..1.min(gen_ids.len())], false)
            .unwrap_or_default();
        let ok = first.trim() == "Paris";
        println!(
            "\nM5 GATE: first token {first:?} {} (expected \"Paris\")",
            if ok { "✓" } else { "✗ DIVERGED" }
        );
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("qwen35_generate is macOS-only (Metal island). No-op on this platform.");
}
