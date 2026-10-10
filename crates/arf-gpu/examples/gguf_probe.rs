//! Validate the GGUF loader against a real file: parse it, then report tensor
//! count + sanity stats (finite, magnitude) on a few mapped tensors. Catches
//! offset/layout/dequant bugs before a full model load.
//!
//! Usage: cargo run --example gguf_probe -- <path-to.gguf>

use arf_core::config::ModelConfig;
use arf_core::model::gguf::load_gguf;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: gguf_probe <file.gguf>");
    let cfg = ModelConfig::qwen3_coder_30b();
    let started = std::time::Instant::now();
    let w = load_gguf(std::path::Path::new(&path), &cfg).expect("load_gguf");
    eprintln!("loaded GGUF in {:?}", started.elapsed());

    // A few representative tensors across the pipeline.
    let probes = [
        (
            "model.embed_tokens.weight",
            cfg.vocab_size * cfg.hidden_size,
        ),
        (
            "model.layers.0.self_attn.q_proj.weight",
            cfg.num_attention_heads * cfg.head_dim * cfg.hidden_size,
        ),
        ("model.layers.0.self_attn.q_norm.weight", cfg.head_dim),
        // Q5_K tensors (the new dequant path): attn_v + a ffn_down expert in early layers.
        (
            "model.layers.0.self_attn.v_proj.weight",
            cfg.num_kv_heads * cfg.head_dim * cfg.hidden_size,
        ),
        (
            "model.layers.0.self_attn.k_proj.weight",
            cfg.num_kv_heads * cfg.head_dim * cfg.hidden_size,
        ),
        (
            "model.layers.0.mlp.experts.0.down_proj.weight",
            cfg.hidden_size * 768,
        ),
        ("model.layers.0.mlp.gate.weight", 128 * cfg.hidden_size),
        (
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            768 * cfg.hidden_size,
        ),
        (
            "model.layers.47.mlp.experts.127.down_proj.weight",
            cfg.hidden_size * 768,
        ),
        ("model.norm.weight", cfg.hidden_size),
    ];
    for (name, expect_len) in probes {
        match w.get_f32(name) {
            Some(v) => {
                let n = v.len();
                let finite = v.iter().all(|x| x.is_finite());
                let max = v.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
                let mean = v.iter().sum::<f32>() / n.max(1) as f32;
                let ok = if n == expect_len {
                    "ok"
                } else {
                    "LEN MISMATCH"
                };
                eprintln!(
                    "{name:50} n={n} (expect {expect_len}, {ok}) finite={finite} max|.|={max:.4} mean={mean:.5}"
                );
            }
            None => eprintln!("{name:50} MISSING"),
        }
    }
}
