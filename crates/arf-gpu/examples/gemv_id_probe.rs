//! PROVE the indirect (MoE) GEMV — the kernel_mul_mv_id port. Parity vs CPU oracle (<1e-3) AND
//! speed (%roofline) at real qwen3-coder-30B MoE dims. NO model load — synthetic experts built
//! internally, so it's fast + machine-safe. If parity passes AND it hits high %roofline, the port
//! is CORRECT (== fast, by definition) and ready to wire into the concurrent megakernel.
//!
//! Run: cargo run --release -p arf-gpu --example gemv_id_probe

#[cfg(target_os = "macos")]
fn main() {
    use arf_gpu::gpu::concurrent_metal::MetalIsland;
    use arf_gpu::gpu::GpuContext;
    use std::sync::Arc;

    const ROOFLINE: f64 = 410.0;
    let ctx = Arc::new(GpuContext::new().expect("gpu"));
    eprintln!("device: {} | roofline {ROOFLINE} GB/s\n", ctx.info());
    let mut island = MetalIsland::new(&ctx).expect("metal island");

    let msl = include_str!("../src/shaders/metal/gemv_q4ks_id_msl.metal");
    island
        .compile_msl(&ctx, "gemv_id", msl, "gemv_q4ks_id")
        .expect("gemv_q4ks_id compile");
    let msl_dn = include_str!("../src/shaders/metal/gemv_q4ks_id_down_msl.metal");
    island
        .compile_msl(&ctx, "gemv_id_down", msl_dn, "gemv_q4ks_id_down")
        .expect("down compile");
    // NSG=2 variants (fc3=2, 64 threads/tg): 2 simdgroups on the same rows — the llama
    // kernel_mul_mv_id N_SG=2 occupancy lever (deleted L29: flat in situ). A/B'd below vs NSG=1.
    island
        .compile_msl_uint_fc(&ctx, "gemv_id_nsg2", msl, "gemv_q4ks_id", &[(3usize, 2u32)])
        .expect("gemv_q4ks_id NSG=2 compile");
    island
        .compile_msl_uint_fc(
            &ctx,
            "gemv_id_down_nsg2",
            msl_dn,
            "gemv_q4ks_id_down",
            &[(3usize, 2u32)],
        )
        .expect("gemv_q4ks_id_down NSG=2 compile");
    // Compile-check the route + swiglu MoE kernels (full parity for these comes with the wiring).
    match island.compile_msl(
        &ctx,
        "moe_route",
        include_str!("../src/shaders/metal/moe_route_msl.metal"),
        "moe_route",
    ) {
        Ok(()) => println!("  moe_route MSL compiles ✓"),
        Err(e) => {
            println!("  moe_route FAILED ✗: {e}");
            std::process::exit(1);
        }
    }
    match island.compile_msl(
        &ctx,
        "moe_swiglu",
        include_str!("../src/shaders/metal/moe_swiglu_msl.metal"),
        "moe_swiglu",
    ) {
        Ok(()) => println!("  moe_swiglu MSL compiles ✓"),
        Err(e) => {
            println!("  moe_swiglu FAILED ✗: {e}");
            std::process::exit(1);
        }
    }

    // Real qwen3-coder-30B MoE dims: hidden=2048, inter=768, top_k=8, 128 experts.
    // gate/up: k=hidden=2048, n=inter=768. down: k=inter=768, n=hidden=2048.
    let shapes: &[(&str, usize, usize)] = &[("gate/up  ", 2048, 768), ("down     ", 768, 2048)];
    let (top_k, experts) = (8usize, 128usize);

    println!(
        "=== indirect MoE GEMV (kernel_mul_mv_id port) — top_k={top_k}, experts={experts} ==="
    );
    println!(
        "{:>16} {:>6} {:>6}   {:>9} {:>8} {:>6}",
        "shape", "k", "n", "max_rel", "GB/s", "%roof"
    );
    let mut all_ok = true;
    for (name, k, n) in shapes {
        for (arm, key, tgw) in [("nsg1", "gemv_id", 32usize), ("NSG2", "gemv_id_nsg2", 64)] {
            let (rel, gb, pct) = island
                .gemv_q4ks_id_check(&ctx, key, *k, *n, top_k, experts, ROOFLINE, 80, tgw)
                .expect("id check");
            let ok = rel < 1e-2;
            all_ok &= ok;
            println!(
                "{name:>10} {arm} {k:>6} {n:>6}   {:>9.6}{} {gb:>8.1} {pct:>5.1}%",
                rel,
                if ok { " ✓" } else { " ✗DIVERGE" }
            );
        }
    }
    // DOWN-indirect-reduce (silu_all + ids + wts → mlp_down): k=inter=768, n=hidden=2048.
    println!("\n=== indirect MoE DOWN GEMV + router-reduce (down half of the port) ===");
    for (arm, key, tgw) in [
        ("nsg1", "gemv_id_down", 32usize),
        ("NSG2", "gemv_id_down_nsg2", 64),
    ] {
        let (drel, dgb, dpct) = island
            .gemv_q4ks_id_down_check(&ctx, key, 768, 2048, top_k, experts, ROOFLINE, 80, tgw)
            .expect("down check");
        let dok = drel < 1e-2;
        all_ok &= dok;
        println!(
            "{:>10} {arm} {:>6} {:>6}   {:>9.6}{} {dgb:>8.1} {dpct:>5.1}%",
            "down",
            768,
            2048,
            drel,
            if dok { " ✓" } else { " ✗DIVERGE" }
        );
    }

    println!();
    if all_ok {
        println!("PARITY ✓ — BOTH indirect GEMVs (gate/up + down-reduce) reuse the dense kernel correctly.");
        println!("If %roofline is high, the port is CORRECT == FAST. Wire it into the concurrent megakernel.");
    } else {
        println!("✗ DIVERGED — the expert-offset indexing is wrong; fix before wiring.");
        std::process::exit(1);
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("gemv_id_probe is macOS-only.");
}
