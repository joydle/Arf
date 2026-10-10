//! Discrimination probe: does the vision pipeline produce DISTINCT soft-tokens for distinct
//! images? Encodes several synthetic images (solid colors + shapes) through the real
//! VisionPipeline and reports pairwise cosine similarity of their [256,2560] soft-tokens.
//! If very different images score ~1.0, the encoder/preprocess collapses (bug). If they
//! score clearly < 1, the encoder discriminates and any weak grounding is downstream.
//!
//! Usage: cargo run --release --example vision_discriminate

use std::sync::Arc;

use arf_core::ImageEncoder;
use arf_gpu::gpu::vision::VisionPipeline;
use arf_gpu::gpu::GpuContext;

fn solid(w: usize, h: usize, rgb: [u8; 3]) -> Vec<u8> {
    let mut v = vec![0u8; w * h * 3];
    for i in 0..w * h {
        v[i * 3] = rgb[0];
        v[i * 3 + 1] = rgb[1];
        v[i * 3 + 2] = rgb[2];
    }
    v
}

fn circle(w: usize, h: usize, fg: [u8; 3], bg: [u8; 3]) -> Vec<u8> {
    let mut v = vec![0u8; w * h * 3];
    let (cx, cy, r2) = (w as f32 / 2.0, h as f32 / 2.0, (w as f32 / 3.0).powi(2));
    for y in 0..h {
        for x in 0..w {
            let d = (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2);
            let c = if d < r2 { fg } else { bg };
            let i = (y * w + x) * 3;
            v[i] = c[0];
            v[i + 1] = c[1];
            v[i + 2] = c[2];
        }
    }
    v
}

fn cos(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        d += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    d / (na.sqrt() * nb.sqrt())
}

fn main() {
    let path = std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../models/gemma-3-4b-vision/mmproj-F16.gguf"
    ));
    assert!(path.exists(), "mmproj not found");
    let ctx = Arc::new(GpuContext::new().expect("gpu"));
    let vp = VisionPipeline::load(&ctx, path, 2560).expect("load");

    let imgs: Vec<(&str, Vec<u8>)> = vec![
        ("red", solid(128, 128, [220, 30, 30])),
        ("blue", solid(128, 128, [30, 30, 220])),
        ("green", solid(128, 128, [30, 200, 30])),
        (
            "red_circle",
            circle(128, 128, [220, 30, 30], [255, 255, 255]),
        ),
        (
            "blue_circle",
            circle(128, 128, [30, 30, 220], [255, 255, 255]),
        ),
    ];
    let embeds: Vec<(&str, Vec<f32>)> = imgs
        .iter()
        .map(|(name, rgb)| (*name, vp.encode(rgb, 128, 128)))
        .collect();

    // per-image stats
    for (name, e) in &embeds {
        let mean = e.iter().sum::<f32>() / e.len() as f32;
        let var = e.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / e.len() as f32;
        eprintln!("[{name:>11}] len={} mean={mean:.4} var={var:.4}", e.len());
    }
    eprintln!("\npairwise cosine (1.0 = identical → BAD if images differ):");
    for i in 0..embeds.len() {
        for j in (i + 1)..embeds.len() {
            let c = cos(&embeds[i].1, &embeds[j].1);
            eprintln!("  {:>11} vs {:<11} cos={c:.4}", embeds[i].0, embeds[j].0);
        }
    }
}
