//! Measure the memory of loading ONLY the DiT (no T5/CLIP/VAE), to isolate whether the
//! Q4_K_S path actually shrinks resident + transient memory vs the bf16 path. Run under
//! `/usr/bin/time -l` and compare ARF_FLUX_FORCE_BF16=1 vs unset.
//!
//! Run: /usr/bin/time -l cargo run --release -p arf-gpu --example flux_dit_load_probe [grid]

use arf_gpu::gpu::flux::{DitConfig, GpuDiT};
use arf_gpu::gpu::GpuContext;
use std::sync::Arc;

const MODELS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/flux-schnell/flux1-schnell-Q4_K_S.gguf"
);

fn rss_mb() -> u64 {
    // current RSS via mach task_info would need a crate; use ps on self.
    let pid = std::process::id();
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output();
    out.ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|kb| kb / 1024)
        .unwrap_or(0)
}

fn main() {
    let grid: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let bf16 = std::env::var("ARF_FLUX_FORCE_BF16").is_ok();
    println!(
        "path: {}",
        if bf16 {
            "bf16 (legacy)"
        } else {
            "Q4_K_S (new)"
        }
    );
    println!("RSS before load: {} MB", rss_mb());
    let ctx = Arc::new(GpuContext::new().expect("gpu"));
    let cfg = DitConfig::schnell(grid, 256);
    let t = std::time::Instant::now();
    let dit = GpuDiT::load(&ctx, std::path::Path::new(MODELS), cfg).expect("load dit");
    println!(
        "RSS after DiT load: {} MB  (loaded in {:.1}s)",
        rss_mb(),
        t.elapsed().as_secs_f32()
    );
    std::hint::black_box(&dit);
}
