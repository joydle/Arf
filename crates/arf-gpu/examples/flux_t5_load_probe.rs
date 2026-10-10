//! Measure the memory footprint of loading ONLY the T5-XXL encoder (the current bf16 path),
//! to size the F1 Step-5 win before building it. Run under `/usr/bin/time -l`.
//!
//! Run: /usr/bin/time -l cargo run --release -p arf-gpu --example flux_t5_load_probe

use arf_gpu::gpu::flux::{GpuT5, T5Config};
use arf_gpu::gpu::GpuContext;
use std::sync::Arc;

const T5: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/flux-schnell/t5xxl-Q8_0.gguf"
);

fn main() {
    let ctx = Arc::new(GpuContext::new().expect("gpu"));
    let t = std::time::Instant::now();
    let t5 = GpuT5::load(&ctx, std::path::Path::new(T5), T5Config::default()).expect("load t5");
    println!("T5 loaded in {:.1}s", t.elapsed().as_secs_f32());
    std::hint::black_box(&t5);
}
