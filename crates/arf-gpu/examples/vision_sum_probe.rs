//! Numerical localization probe: decode an image exactly as serve does, run the CPU SigLIP
//! pipeline with ARF_VISION_SUM=1, and print per-stage element sums to compare against
//! llama.cpp's MTMD_DEBUG_GRAPH `sum=` values — pinpointing the first diverging op.
//!
//! Usage: ARF_VISION_SUM=1 cargo run --release --example vision_sum_probe -- /tmp/sun.png

use std::sync::Arc;

use arf_core::model::gguf::LazyGguf;
use arf_core::model::vision::{VisionConfig, VisionEncoder, VisionPreprocess, VisionProjector};
use arf_gpu::gpu::GpuContext;

fn main() {
    let img_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/sun.png".into());
    let mmproj = std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../models/gemma-3-4b-vision/mmproj-F16.gguf"
    ));
    let _ctx = Arc::new(GpuContext::new().expect("gpu")); // not used (CPU path) but keeps parity
    let cfg = VisionConfig::default();
    let g = LazyGguf::open(mmproj, &arf_core::config::ModelConfig::gemma3_4b()).expect("open");
    let enc = VisionEncoder::load(&g, cfg).expect("load encoder");
    let proj = VisionProjector::load(&g, cfg.hidden, 2560).expect("load projector");
    let pp = VisionPreprocess::from_gguf(&g);

    // decode image exactly as serve does (image crate → RGB8)
    let img = image::open(&img_path).expect("open image").to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let rgb = img.into_raw();
    eprintln!("image {img_path}: {w}x{h}, preprocess size={}", pp.size);

    let px = pp.preprocess(&rgb, w, h);
    // a few input-pixel sums for the resize/normalize sanity (compare to llama.cpp inp_raw if needed)
    let pxsum: f64 = px.iter().map(|&v| v as f64).sum();
    eprintln!(
        "[vsum] {:30} sum={pxsum:.4} len={}",
        "preprocessed pixels",
        px.len()
    );

    let patches = enc.encode(&px); // prints patch_matmul / patch_bias / pos_embed + (post-norm)
    let post_sum: f64 = patches.iter().map(|&v| v as f64).sum();
    eprintln!(
        "[vsum] {:30} sum={post_sum:.4} (post-27-layers+post_ln)",
        "encode out"
    );
    let _soft = proj.project(&patches, cfg.grid()); // prints pooled / rms·soft_emb / projection
}
