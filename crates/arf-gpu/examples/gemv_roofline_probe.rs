//! Profile our WGSL Q4_K decode GEMV's achieved DRAM bandwidth vs the M4 Max roofline,
//! at the real gemma-4-12B matmul shapes. This is the ground truth for the 1.93x llama.cpp
//! decode gap: if we're at ~30% of roofline and llama.cpp is at ~70%, the gap IS the GEMV
//! and a hand-MSL kernel closes it. The kernel is compiled from the EXACT WGSL the wgpu
//! decode path uses (matmul_vec_q4k_sg.wgsl), so the GB/s is what the real engine achieves.
//!
//! Run: cargo run --release -p arf-gpu --example gemv_roofline_probe

#[cfg(target_os = "macos")]
fn main() {
    use arf_gpu::gpu::concurrent_metal::MetalIsland;
    use arf_gpu::gpu::GpuContext;
    use std::sync::Arc;

    // M4 Max: 410 GB/s unified memory bandwidth (advertised peak).
    const ROOFLINE: f64 = 410.0;
    let wgsl = include_str!("../src/shaders/wgsl/matmul_vec_q4k_sg.wgsl");

    let ctx = Arc::new(GpuContext::new().expect("gpu"));
    eprintln!("device: {} | roofline {ROOFLINE} GB/s\n", ctx.info());
    let mut island = MetalIsland::new(&ctx).expect("metal island");

    // gemma-4-12B Q4_K GEMV shapes (k=in, n=out). The decode frame is dominated by these.
    // (matmul_vec_q4k_sg: WG=64, one workgroup per output column → cols_per_wg=1.)
    let shapes: &[(&str, usize, usize)] = &[
        ("q_proj   ", 3840, 4096), // q: 16 heads × 256
        ("kv_proj  ", 3840, 2048), // k/v: 8 heads × 256
        ("o_proj   ", 4096, 3840),
        ("gate/up  ", 3840, 15360),
        ("down_proj", 15360, 3840),
        ("lm_head  ", 3840, 262144),
    ];

    println!(
        "{:>10} {:>8} {:>8} {:>10} {:>10} {:>8}",
        "shape", "k", "n", "gpu_us", "GB/s", "%roof"
    );
    let mut tot_us = 0.0;
    for (name, k, n) in shapes {
        let (ms, gb_s, pct) = island
            .gemv_bench(&ctx, "q4k_sg", wgsl, *k, *n, 64, 1, ROOFLINE, 50)
            .expect("bench");
        tot_us += ms * 1000.0;
        println!(
            "{name:>10} {k:>8} {n:>8} {:>10.2} {gb_s:>10.1} {pct:>7.1}%",
            ms * 1000.0
        );
    }
    println!("\nsum of one decode layer's q+kv+o+gate/up+down ≈ the per-layer GEMV cost.");
    println!("total of all listed GEMVs: {tot_us:.1} us");
    println!("\n=> if %roof is ~20-40%, the GEMV is the gap. llama.cpp's hand-MSL hits ~60-70%.");
    println!("   The hand-MSL Q4_K GEMV (2 rows/simdgroup, float4/dot, simd_sum) is the lever.");
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("gemv_roofline_probe is macOS-only.");
}
