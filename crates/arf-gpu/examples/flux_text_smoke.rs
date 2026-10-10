//! Smoke test for the FLUX P0 text encoders: load CLIP-L + T5-XXL on the real weights, run a
//! fixed token sequence through each on the GPU, and assert the outputs are finite + non-
//! degenerate with the right shapes. This is the first gate (catches orientation/NaN/missing-
//! tensor bugs) BEFORE the numerical parity gate vs a diffusers/llama.cpp oracle.
//!
//! Usage: cargo run --release --example flux_text_smoke

use std::sync::Arc;

use arf_gpu::gpu::flux::{ClipConfig, GpuClipText, GpuT5, T5Config};
use arf_gpu::gpu::GpuContext;

fn stats(name: &str, v: &[f32], expect_len: usize) {
    assert_eq!(
        v.len(),
        expect_len,
        "{name}: len {} != {expect_len}",
        v.len()
    );
    let finite = v.iter().all(|x| x.is_finite());
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32;
    let (lo, hi) = v
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &x| {
            (a.min(x), b.max(x))
        });
    eprintln!(
        "[{name}] len={} finite={finite} mean={mean:.4} var={var:.4} range=[{lo:.3},{hi:.3}]",
        v.len()
    );
    assert!(finite, "{name}: non-finite output");
    assert!(var > 1e-6, "{name}: degenerate (var≈0)");
}

fn main() {
    let base = std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../models/flux-schnell"
    ));
    let ctx = Arc::new(GpuContext::new().expect("gpu"));

    // CLIP-L: a short token sequence (valid ids < vocab) ending in an EOS-marker position.
    let clip_path = base.join("clip_l.safetensors");
    if clip_path.exists() {
        let clip = GpuClipText::load(&ctx, &clip_path, ClipConfig::default()).expect("load clip");
        let toks: Vec<u32> = vec![49406, 320, 1125, 539, 320, 2368, 49407]; // <bos> a photo of a cat <eos>
        let eos = toks.len() - 1;
        let pooled = clip.encode(&toks, eos);
        stats("CLIP pooled", &pooled, 768);
    } else {
        eprintln!("SKIP CLIP: {clip_path:?} not found");
    }

    // T5-XXL: a short token sequence → [seq, 4096] sequence embedding.
    let t5_path = base.join("t5xxl-Q8_0.gguf");
    if t5_path.exists() {
        let t5 = GpuT5::load(&ctx, &t5_path, T5Config::default()).expect("load t5");
        let toks: Vec<u32> = vec![71, 1712, 13, 3, 9, 1712, 1]; // arbitrary valid ids + </s>=1
        let seq = t5.encode(&toks);
        stats("T5 seq", &seq, toks.len() * 4096);
    } else {
        eprintln!("SKIP T5: {t5_path:?} not found");
    }

    eprintln!("FLUX P0 text-encoder smoke test PASSED");
}
