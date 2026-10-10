//! Measure speculative-decoding acceptance on real weights (P11a decision gate).
//! Run: ARF_MODEL_PATH=... cargo run --release -p arf-gpu --example spec_accept
use std::io::Write;
use std::path::{Path, PathBuf};

use arf_core::config::ModelConfig;
use arf_core::model::speculative::generate_speculative;
use arf_core::model::weights;
use arf_core::Tokenizer;

fn main() {
    let dir = PathBuf::from(std::env::var("ARF_MODEL_PATH").expect("ARF_MODEL_PATH"));
    let cfg = ModelConfig::llama_3_2_1b();
    let mf = dir.join("model.safetensors");
    let paths: [&Path; 1] = [mf.as_path()];
    eprint!("loading real weights... ");
    std::io::stderr().flush().ok();
    let model = weights::load_safetensors(&cfg, &paths, 8192).expect("load");
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tok");
    eprintln!("done");

    let prompts = [
        ("natural", "The capital of France is"),
        ("code", "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n\ndef factorial(n):\n    if n < 2:\n        return"),
        ("repetitive", "the quick brown fox jumps. the quick brown fox jumps. the quick brown fox jumps."),
    ];
    let n = 48;
    let k = 4;
    for (label, p) in prompts {
        let ids = tok.encode(p, true).expect("encode");
        for order in [2usize, 3] {
            let r = generate_speculative(&model, &ids, n, k, order, 16);
            println!(
                "[{label:>10}] k={k} order={order}: {:.2} tok/forward  ({} fwd / {} tok)  draft-accept {:.0}%",
                r.stats.tokens_per_forward(),
                r.stats.forward_passes,
                r.stats.accepted_tokens,
                r.stats.draft_acceptance() * 100.0,
            );
            std::io::stdout().flush().ok();
        }
    }
}
