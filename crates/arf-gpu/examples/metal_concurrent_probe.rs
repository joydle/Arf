//! GO/NO-GO: does Metal's MTLDispatchTypeConcurrent actually overlap independent
//! bandwidth-bound kernels on this Mac? This decides whether the as_hal concurrent-encoder
//! island can close the llama.cpp decode gap, OR whether the gap is raw GEMV roofline
//! (in which case we redirect to hand-MSL kernels, reusing the same island plumbing).
//!
//! Run: cargo run --release -p arf-gpu --example metal_concurrent_probe

#[cfg(target_os = "macos")]
fn main() {
    use arf_gpu::gpu::concurrent_metal::MetalIsland;
    use arf_gpu::gpu::GpuContext;
    use std::sync::Arc;

    let ctx = Arc::new(GpuContext::new().expect("gpu"));
    eprintln!("device: {}", ctx.info());
    let mut island = MetalIsland::new(&ctx).expect("metal island (needs Metal backend)");
    eprintln!("Metal island up.\n");

    // Sweep buffer size (per-kernel DRAM traffic) and N (independent kernels). Concurrent
    // should help most when N kernels each UNDER-utilize the GPU (small buffers, few cores
    // busy) — and help LEAST when each already saturates DRAM (big buffers). The decode
    // GEMVs are big weight reads (saturating), so we expect the big-buffer rows to show ~no
    // concurrent win — that's the bandwidth-bound prediction.
    println!(
        "{:>10} {:>4} {:>12} {:>12} {:>8}",
        "bytes/kern", "N", "serial_ms", "concur_ms", "speedup"
    );
    for &mb in &[1usize, 4, 16, 64] {
        for &n in &[2usize, 3, 7] {
            let bytes = mb * 1024 * 1024;
            let (s, c) = island.microbench(&ctx, n, bytes, 30).expect("microbench");
            println!(
                "{:>10} {:>4} {:>12.4} {:>12.4} {:>7.2}×",
                format!("{mb}MB"),
                n,
                s,
                c,
                s / c
            );
        }
    }
    println!(
        "\nspeedup > ~1.15× on the SMALL-buffer rows = concurrency helps when cores are idle."
    );
    println!(
        "speedup ~1.0× on the BIG-buffer (16/64MB) rows = bandwidth-bound (decode GEMV case)."
    );
    println!("=> if even small buffers show ~1.0×, MTLDispatchTypeConcurrent is a no-op here → redirect to hand-MSL GEMV roofline.");
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("metal_concurrent_probe is macOS-only.");
}
