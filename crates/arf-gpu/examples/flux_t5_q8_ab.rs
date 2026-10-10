//! F1 Step-5 A/B parity: T5-XXL encode via the NEW Q8_0 quantized-resident path vs the legacy
//! bf16 path (ARF_FLUX_FORCE_BF16), on identical tokens. High cosine ⇒ the Q8_0 dequant-in-
//! kernel GEMMs reproduce the bf16 result (residual = Q8_0-vs-bf16 weight precision, expected).
//!
//! Run: cargo run --release -p arf-gpu --example flux_t5_q8_ab

use arf_gpu::gpu::flux::{GpuT5, T5Config};
use arf_gpu::gpu::GpuContext;
use std::sync::Arc;

const T5: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/flux-schnell/t5xxl-Q8_0.gguf"
);

fn cos_rl2(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut dot, mut na, mut nb, mut diff) = (0.0f64, 0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += (x as f64).powi(2);
        nb += (y as f64).powi(2);
        diff += ((x - y) as f64).powi(2);
    }
    (dot / (na.sqrt() * nb.sqrt()), diff.sqrt() / nb.sqrt())
}

fn run(bf16: bool, tokens: &[u32]) -> Vec<f32> {
    if bf16 {
        std::env::set_var("ARF_FLUX_FORCE_BF16", "1");
    } else {
        std::env::remove_var("ARF_FLUX_FORCE_BF16");
    }
    let ctx = Arc::new(GpuContext::new().expect("gpu"));
    let t5 = GpuT5::load(&ctx, std::path::Path::new(T5), T5Config::default()).expect("load t5");
    t5.encode(tokens)
}

fn main() {
    // A short prompt padded to 256 (T5 seq), EOS=1 then zeros.
    let mut tokens = vec![3i32 as u32, 9, 1991, 5, 27, 1]; // arbitrary ids + </s>
    tokens.resize(256, 0);

    println!("T5 Q8_0 (new) path…");
    let q = run(false, &tokens);
    println!("T5 bf16 (legacy) path…");
    let b = run(true, &tokens);

    let (cos, rl2) = cos_rl2(&q, &b);
    println!(
        "\nT5 encode  Q8_0 vs bf16   cos={cos:.6}  rel_l2={rl2:.4e}  (n={})",
        q.len()
    );
    // Q8_0 and bf16 are TWO DIFFERENT lossy quantizations of the same f32 checkpoint, and the gap
    // compounds across T5's 24 layers — so a cosine ~0.998 between them is EXPECTED, not a bug.
    // (The loader split is proven bit-exact vs ggml dequant by examples/t5_q8_split_check, and the
    // kernel is parity-exact vs matmul_nt_q8_0 by examples/coop_q8_0_probe. Q8_0 is in fact closer
    // to the true f32 weights than bf16's 7-bit mantissa.) Threshold guards against a real wiring
    // break (which would collapse the cosine well below this), not the quant delta.
    if cos >= 0.995 {
        println!("PASS — Q8_0 T5 wiring correct (matches bf16 to cross-quant precision).");
        std::process::exit(0);
    } else {
        println!("FAIL — wiring bug (cos too low for mere quant precision).");
        std::process::exit(1);
    }
}
