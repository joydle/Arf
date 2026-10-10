//! PARITY + SPEED probe for the tiled dense Q4_K_S GEMM (matmul_mm_q4ks) vs the per-row GEMV
//! (gemv_q4ks_batch) — the dense q/k/v/o projection accelerator for the batched megakernel.
//! Runs BOTH kernels on identical synthetic Q4_K_S weights + B-row activations and reports
//! max_abs / max_rel + each kernel's GPU ms. The two compute the SAME math (Σ_k a·dequant(W));
//! only the f32 accumulation order differs, so parity must hold to reduction-reorder noise.
//!
//! Run: cargo run --release -p arf-gpu --example mm_q4ks_probe
//!
//! HARD GATE: max_rel < 1e-3 at B=16 AND B=32, for n=4096 (q_proj) and n=2048 (o_proj).

#[cfg(target_os = "macos")]
fn main() {
    use arf_gpu::gpu::concurrent_metal::MetalIsland;
    use arf_gpu::gpu::GpuContext;
    use std::sync::Arc;

    let ctx = Arc::new(GpuContext::new().expect("gpu"));
    eprintln!("device: {}\n", ctx.info());
    let mut island = MetalIsland::new(&ctx).expect("metal island (needs Metal backend)");

    // qwen3-coder-30B dense-projection shapes:
    //   q_proj: k=h=2048 → n=q_dim=4096   (nh=32, hd=128)
    //   k/v_proj: k=2048 → n=kv_dim=512   (nkv=4, hd=128)
    //   o_proj: k=q_dim=4096 → n=h=2048
    //   lm_head: k=2048 → n=vocab (Q8 usually, skip here)
    let shapes: &[(&str, usize, usize)] = &[
        ("q_proj  k=2048 n=4096", 2048, 4096),
        ("o_proj  k=4096 n=2048", 4096, 2048),
        ("kv_proj k=2048 n=512 ", 2048, 512),
    ];
    let mut all_ok = true;
    for &b in &[8usize, 16, 32, 64] {
        eprintln!("== B={b} ==");
        for &(name, k, n) in shapes {
            let (abs, rel, gemv_ms, mm_ms) = island
                .mm_q4ks_parity(&ctx, b, k, n, 40)
                .expect("mm_q4ks_parity");
            // gate only B>=16 (the prompt's B16/B32 bar); B=8 informational.
            let gated = b >= 16;
            let ok = rel < 1e-3;
            if gated {
                all_ok &= ok;
            }
            let speedup = gemv_ms / mm_ms;
            eprintln!(
                "  {name}: max_abs={abs:.3e} max_rel={rel:.3e}  gemv={gemv_ms:.4}ms mm={mm_ms:.4}ms  speedup={speedup:.2}x  {}{}",
                if ok { "PASS" } else { "FAIL" },
                if gated { "" } else { " (informational)" },
            );
        }
    }
    eprintln!(
        "\n{}",
        if all_ok {
            "MM_Q4KS PARITY: PASS (rel<1e-3 @B16,B32)"
        } else {
            "MM_Q4KS PARITY: FAIL"
        }
    );
    assert!(all_ok, "matmul_mm_q4ks parity FAILED");
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("mm_q4ks_probe is macOS/Metal-only");
}
