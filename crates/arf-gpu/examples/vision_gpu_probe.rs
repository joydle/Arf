//! Fast GPU SigLIP localization probe. Loads the real mmproj GGUF, builds the
//! `GpuVisionEncoder`, runs the cheap CPU `patch_embed`, then runs `encode_layers` with
//! ARF_VISION_PROBE=1 to print per-stage finiteness/range on layer 0 — WITHOUT the slow
//! (~195s) CPU reference encode. Used to localize the NaN in the GPU forward.
//!
//! Usage: ARF_VISION_PROBE=1 cargo run --release --example vision_gpu_probe

use std::sync::Arc;

use arf_core::config::ModelConfig;
use arf_core::model::gguf::LazyGguf;
use arf_core::model::vision::{VisionConfig, VisionEncoder};
use arf_gpu::gpu::vision::GpuVisionEncoder;
use arf_gpu::gpu::GpuContext;

fn main() {
    let path = std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../models/gemma-3-4b-vision/mmproj-F16.gguf"
    ));
    assert!(path.exists(), "mmproj GGUF not found at {path:?}");
    let ctx = Arc::new(GpuContext::new().expect("gpu"));
    let cfg = VisionConfig::default();
    let g = LazyGguf::open(path, &ModelConfig::gemma3_4b()).expect("open mmproj");
    let cpu = VisionEncoder::load(&g, cfg).expect("load CPU vision encoder");
    let gpu = GpuVisionEncoder::new(&ctx, cfg, cpu.weights());

    // synthetic gradient image (matches the parity test input)
    let n = cfg.image_size;
    let mut px = vec![0.0f32; 3 * n * n];
    for ch in 0..3 {
        for y in 0..n {
            for x in 0..n {
                px[ch * n * n + y * n + x] = ((x + y + ch * 50) as f32 / (2 * n) as f32) - 0.5;
            }
        }
    }
    // Localize the gelu input magnitude: run JUST the up-proj on the GPU and inspect the
    // value at the index the probe flagged as NaN-after-gelu (3032218).
    let hidden0 = cpu.patch_embed(&px);
    let h0_bad = hidden0.iter().position(|v| !v.is_finite());
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in &hidden0 {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    eprintln!("[probe hidden0] range=[{lo:.4},{hi:.4}] first_nonfinite={h0_bad:?}");

    let out = gpu.encode_layers(&hidden0);
    let bad = out.iter().position(|v| !v.is_finite());
    eprintln!("[probe OUT] len={} first_nonfinite={:?}", out.len(), bad);
}
