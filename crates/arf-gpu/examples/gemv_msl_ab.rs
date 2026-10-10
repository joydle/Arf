//! A/B: our WGSL Q4_K decode GEMV vs a hand-tuned MSL kernel (2 cols/simdgroup, simd_sum,
//! activation reuse — llama.cpp's techniques) at real gemma-4-12B shapes. The decode frame
//! is 83% matmul at only 40% roofline; if the hand-MSL GEMV lifts that toward llama.cpp's
//! ~70%, it's the lever for the 1.93x gap. Both compiled + GPU-timed via the Metal island.
//!
//! Run: cargo run --release -p arf-gpu --example gemv_msl_ab

#[cfg(target_os = "macos")]
fn main() {
    use arf_gpu::gpu::concurrent_metal::MetalIsland;
    use arf_gpu::gpu::GpuContext;
    use std::sync::Arc;

    const ROOFLINE: f64 = 410.0;
    // Use the NON-subgroup WGSL (tree reduction) as the oracle: the subgroup variant's
    // `subgroupAdd` needs the SUBGROUP pipeline feature, which our raw naga→MSL→PSO path
    // doesn't enable (it returns 0). The tree-reduction kernel is the portable, correct ref.
    let wgsl = include_str!("../src/shaders/wgsl/matmul_vec_q4k.wgsl");
    let msl = include_str!("../src/shaders/metal/matmul_vec_q4k_msl.metal");

    let ctx = Arc::new(GpuContext::new().expect("gpu"));
    eprintln!("device: {} | roofline {ROOFLINE} GB/s\n", ctx.info());
    let mut island = MetalIsland::new(&ctx).expect("metal island");

    // Pre-compile both under their keys (gemv_bench's compile() no-ops if key exists).
    island.compile(&ctx, "wgsl", wgsl).expect("wgsl compile");
    island
        .compile_msl(&ctx, "msl", msl, "gemv_q4k")
        .expect("msl compile");

    let shapes: &[(&str, usize, usize)] = &[
        ("q_proj   ", 3840, 4096),
        ("kv_proj  ", 3840, 2048),
        ("o_proj   ", 4096, 3840),
        ("gate/up  ", 3840, 15360),
        ("down_proj", 15360, 3840),
        ("lm_head  ", 3840, 262144),
    ];

    // BRIDGE TEST: a SharedBuffer round-trips — bytes written via the wgpu view are visible
    // through the raw MTLBuffer view (same memory, one allocation, no copy).
    {
        let got = MetalIsland::bridge_selftest(&ctx).expect("bridge");
        let want: [f32; 7] = [1.0, -2.5, 3.25, 0.0, 42.0, -7.0, 99.5];
        let ok = got
            .iter()
            .zip(want.iter())
            .all(|(a, b)| (a - b).abs() < 1e-6);
        println!(
            "=== SharedBuffer bridge: wgpu-write → MTL-read = {got:?} → {}\n",
            if ok {
                "ROUND-TRIP OK ✓ (one allocation, two views)"
            } else {
                "MISMATCH ✗"
            }
        );
    }

    // PARITY GATE: same inputs → WGSL and MSL outputs must match (only simd_sum reassoc).
    println!("=== parity (WGSL vs MSL, same inputs) ===");
    let mut parity_ok = true;
    for (name, k, n) in &[("q_proj", 3840usize, 4096usize), ("gate/up", 3840, 15360)] {
        let (abs, rel, norm) = island
            .gemv_parity(&ctx, "wgsl", 64, 1, "msl", 32, 2, *k, *n)
            .expect("parity");
        let ok = rel < 1e-3 || abs < 1e-2;
        parity_ok &= ok;
        println!(
            "  {name:>8}: max_abs={abs:.5} max_rel={rel:.5} ‖ref‖={norm:.2} → {}",
            if ok { "MATCH ✓" } else { "DIVERGE ✗" }
        );
    }
    if !parity_ok {
        println!("\n⚠️ PARITY FAILED — MSL kernel has a real bug (indexing/dequant), NOT just reassoc. Fix before wiring.\n");
    } else {
        println!("  parity OK — MSL math == WGSL.\n");
    }

    println!(
        "{:>10} {:>8} {:>8}   {:>9} {:>6}   {:>9} {:>6}   {:>7}",
        "shape", "k", "n", "wgsl_us", "%roof", "msl_us", "%roof", "speedup"
    );
    for (name, k, n) in shapes {
        // WGSL: WG=64, 1 col/workgroup.
        let (wms, _wg, wpct) = island
            .gemv_bench(&ctx, "wgsl", wgsl, *k, *n, 64, 1, ROOFLINE, 50)
            .expect("wgsl bench");
        // MSL: 32-lane simdgroup, 2 cols/threadgroup.
        let (mms, _mg, mpct) = island
            .gemv_bench(&ctx, "msl", msl, *k, *n, 32, 2, ROOFLINE, 50)
            .expect("msl bench");
        println!(
            "{name:>10} {k:>8} {n:>8}   {:>9.2} {wpct:>5.1}%   {:>9.2} {mpct:>5.1}%   {:>6.2}×",
            wms * 1000.0,
            mms * 1000.0,
            wms / mms
        );
    }
    // === Q4_K_S — the REAL gemma-4-12B decode format (codes + u8/i8 scales + dd) ===
    println!("\n=== Q4_K_S hand-MSL GEMV (the REAL decode format) ===");
    let q4ks_msl = include_str!("../src/shaders/metal/matmul_vec_q4ks_msl.metal");
    island
        .compile_msl(&ctx, "q4ks", q4ks_msl, "gemv_q4ks")
        .expect("q4ks compile");
    // Fused attention+scatter compile-check.
    let attn_fused = include_str!("../src/shaders/metal/attention_fused_msl.metal");
    match island.compile_msl(&ctx, "attn_fused", attn_fused, "attention_decode_fused") {
        Ok(()) => println!("  attention_decode_fused MSL compiles ✓"),
        Err(e) => println!("  attention_decode_fused FAILED ✗: {e}"),
    }
    // Q3K (3-bit) GEMV compile-check — the bandwidth-floor breaker for the FFN.
    let q3k_msl = include_str!("../src/shaders/metal/matmul_vec_q3k_msl.metal");
    match island.compile_msl(&ctx, "q3k", q3k_msl, "gemv_q3k") {
        Ok(()) => {
            println!("  gemv_q3k MSL compiles ✓");
            for &(name, k, n) in &[("gate/up", 3840usize, 15360usize), ("down", 15360, 3840)] {
                let rel = island.gemv_q3k_check(&ctx, "q3k", k, n).expect("q3k check");
                println!(
                    "  gemv_q3k {name} (k={k},n={n}): max_rel={rel:.5} → {}",
                    if rel < 1e-2 {
                        "MATCH ✓"
                    } else {
                        "DIVERGE ✗ (kernel bug)"
                    }
                );
            }
        }
        Err(e) => println!("  gemv_q3k MSL FAILED ✗: {e}"),
    }
    println!(
        "{:>10} {:>8} {:>8}   {:>10} {:>9} {:>6}",
        "shape", "k", "n", "max_rel", "msl_GB/s", "%roof"
    );
    for (name, k, n) in shapes {
        let (rel, _norm, gb_s, pct) = island
            .gemv_q4ks_check(&ctx, "q4ks", *k, *n, ROOFLINE, 50)
            .expect("q4ks check");
        let ok = rel < 1e-2;
        println!(
            "{name:>10} {k:>8} {n:>8}   {:>9.5}{} {gb_s:>9.1} {pct:>5.1}%",
            rel,
            if ok { " ✓" } else { " ✗" }
        );
    }
    // === MSL OP PORT: RMSNorm (megakernel building block) — parity + timing ===
    println!("\n=== MSL RMSNorm port (vs CPU oracle) ===");
    let rms_msl = include_str!("../src/shaders/metal/rmsnorm_msl.metal");
    island
        .compile_msl(&ctx, "rmsnorm", rms_msl, "rmsnorm")
        .expect("rmsnorm compile");
    for &(name, h) in &[("hidden", 3840usize), ("q_dim", 4096)] {
        let (rel, norm, us) = island
            .rmsnorm_check(&ctx, "rmsnorm", h, 1e-6, 256, 100)
            .expect("rmsnorm check");
        println!(
            "  {name:>8} (h={h}): max_rel={rel:.6} ‖ref‖={norm:.2} {us:.2}us → {}",
            if rel < 1e-3 {
                "MATCH ✓"
            } else {
                "DIVERGE ✗"
            }
        );
    }

    // === MSL element-wise op ports (megakernel building blocks) ===
    println!("\n=== MSL decode op ports (vs CPU oracle) ===");
    let ops_msl = include_str!("../src/shaders/metal/decode_ops_msl.metal");
    island
        .compile_msl(&ctx, "geglu", ops_msl, "geglu")
        .expect("geglu compile");
    let grel = island.geglu_check(&ctx, "geglu", 15360).expect("geglu");
    println!(
        "  geglu (n=15360):     max_rel={grel:.6} → {}",
        if grel < 1e-3 {
            "MATCH ✓"
        } else {
            "DIVERGE ✗"
        }
    );
    island
        .compile_msl(&ctx, "rmsnorm_add", ops_msl, "rmsnorm_add")
        .expect("rmsnorm_add compile");
    let rarel = island
        .rmsnorm_add_check(&ctx, "rmsnorm_add", 3840, 1e-6)
        .expect("rmsnorm_add");
    println!(
        "  rmsnorm_add (h=3840):max_rel={rarel:.6} → {}",
        if rarel < 1e-3 {
            "MATCH ✓"
        } else {
            "DIVERGE ✗"
        }
    );
    // verify the NEW megakernel kernels compile (qkv_norm, scale_inplace, kv_scatter).
    for entry in &["qkv_norm", "scale_inplace", "kv_scatter"] {
        match island.compile_msl(&ctx, entry, ops_msl, entry) {
            Ok(()) => println!("  kernel '{entry}' compiles ✓"),
            Err(e) => println!("  kernel '{entry}' FAILED ✗: {e}"),
        }
    }
    // attention (the big port) — decode q_len=1, vs CPU softmax-attention oracle.
    let attn_msl = include_str!("../src/shaders/metal/attention_msl.metal");
    island
        .compile_msl(&ctx, "attn", attn_msl, "attention_decode")
        .expect("attn compile");
    for &(name, hd, nk) in &[
        ("hd256 sliding", 256usize, 64usize),
        ("hd512 global", 512, 200),
    ] {
        let arel = island.attention_check(&ctx, "attn", hd, nk).expect("attn");
        println!(
            "  attention {name} (nkeys={nk}): max_rel={arel:.6} → {}",
            if arel < 1e-3 {
                "MATCH ✓"
            } else {
                "DIVERGE ✗"
            }
        );
    }
    // === S0 DE-RISK: full-GQA attention at the REAL gemma-4-12B per-layer geometries ===
    println!(
        "\n=== S0: attention at REAL 12B geometries (full GQA + DPL) — megakernel gate (<1e-4) ==="
    );
    // (name, nh, nkv, head_dim, nkeys, window)
    let geoms = &[
        (
            "sliding hd256 nkv8 grp2 win1024",
            16usize,
            8usize,
            256usize,
            300usize,
            1024usize,
        ),
        ("sliding short-ctx (no window cut)", 16, 8, 256, 64, 1024),
        ("global  hd512 nkv1 grp16 DPL2", 16, 1, 512, 300, 0),
    ];
    let mut s0_ok = true;
    for &(name, nh, nkv, hd, nk, win) in geoms {
        let arel = island
            .attention_geom_check(&ctx, "attn", nh, nkv, hd, nk, win)
            .expect("attn geom");
        let ok = arel < 1e-4;
        s0_ok &= ok;
        println!(
            "  {name}: max_rel={arel:.7} → {}",
            if ok {
                "MATCH ✓"
            } else {
                "DIVERGE ✗ (BLOCKS megakernel)"
            }
        );
    }
    println!(
        "  S0 {}",
        if s0_ok {
            "PASS ✓ — attention bit-exact at both 12B profiles, megakernel unblocked"
        } else {
            "FAIL ✗ — fix attention before the megakernel"
        }
    );

    // === PER-TOKEN ENCODER PoC: the whole FFN sub-block in ONE island command buffer ===
    println!("\n=== FFN block in ONE island encoder (per-token-megakernel proof) ===");
    // ffn_block_selftest expects the GEMV under key "gemv_q4ks" (+ geglu, rmsnorm_add already up).
    island
        .compile_msl(&ctx, "gemv_q4ks", q4ks_msl, "gemv_q4ks")
        .expect("gemv_q4ks compile");
    let (frel, fus) = island
        .ffn_block_selftest(&ctx, 3840, 15360, 1e-6, 50)
        .expect("ffn block");
    println!(
        "  FFN (gate+up+geglu+down+rmsnorm_add, 5 ops, 1 cmd buffer): max_rel={frel:.6} {fus:.1}us → {}",
        if frel < 1e-2 { "MATCH ✓" } else { "DIVERGE ✗" }
    );

    // === SYNC-COST PROBE: is per-GEMV island↔wgpu interleave viable, or megakernel-only? ===
    println!(
        "\n=== sync cost: island GEMV interleaved with wgpu flush (the in-frame question) ==="
    );
    let (inter, isl, ratio) = island
        .sync_cost_probe(&ctx, "q4ks", 3840, 15360, 100)
        .expect("sync probe");
    println!(
        "  island-only {isl:.4} ms/GEMV | interleaved (per-GEMV wgpu flush) {inter:.4} ms/GEMV | overhead {ratio:.2}×"
    );
    println!(
        "  → ratio ~1.0 = per-GEMV routing FREE (wire it directly). >>1 = cross-queue sync dominates → batch per-LAYER or full megakernel."
    );

    println!("\nspeedup > 1.0× on the matmul-dominant shapes → hand-MSL GEMV is the lever.");
    println!("(parity: the MSL math == WGSL; verify token-identical decode before wiring in.)");
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("gemv_msl_ab is macOS-only.");
}
