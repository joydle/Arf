//! L347 — THE b-ROW MICROBENCHMARK. Times ONE isolated layer at b=1 vs b=2 to find the
//! 368 us/layer the second row costs (L346), after every cheaper explanation was killed:
//! not the fixed cost or lm_head (L344), not the record build or checkpoint restore (L331),
//! not a missing parallel path and not CPU/scheduling (L345, 97% GPU-busy), not a depth cliff
//! and not the activation-column term (L346, COLS=8 LOSES at b=2).
//!
//! WHY A SEPARATE HARNESS. AGX refuses per-dispatch GPU timestamps (`sampleCountersInBuffer`
//! asserts — L16/L80, and ARF_BMEGA_KERNPROF says so itself), so inside the 384-dispatch
//! megakernel record a single kernel cannot be isolated. Out here one dispatch group is the
//! whole command buffer, so a CPU-side wall timer around commit+wait IS the kernel's time.
//!
//! Run: cargo run --release -p arf-gpu --example brow_microbench
#[cfg(target_os = "macos")]
fn main() {
    use arf_gpu::gpu::concurrent_metal::MetalIsland;
    use arf_gpu::gpu::GpuContext;
    use std::time::Instant;

    // Qwen3.8-27B geometry (config.rs qwen35_27b): the real shapes, so the numbers transfer.
    let h = 5120usize;
    let nh = 40usize;
    let nkv = 8usize;
    let hd = 128usize;
    let _inter = 17408usize;
    let ctx_len = 256usize;
    let eps = 1e-6f32;

    let ctx = GpuContext::new().expect("gpu context");
    let mut island = MetalIsland::new(&ctx).expect("metal island (need a Metal backend)");

    // MetalIsland::new does NOT compile the B-row set — weights.rs does that during model load.
    // A standalone example must compile exactly what the selftest dispatches, or every call
    // fails with the pipeline's own name (which is how the first run of this harness failed).
    let batch_ops = include_str!("../src/shaders/metal/decode_ops_batch_msl.metal");
    let ops = include_str!("../src/shaders/metal/decode_ops_msl.metal");
    let attn = include_str!("../src/shaders/metal/attention_msl.metal");
    let gemv_b = include_str!("../src/shaders/metal/matmul_vec_q4ks_batch_msl.metal");
    for (key, src) in [
        ("rmsnorm_b", batch_ops),
        ("qk_norm_b", batch_ops),
        ("rope_qk_b", batch_ops),
        ("add_residual_b", batch_ops),
        ("kv_scatter", ops),
        ("attention_decode", attn),
        ("gemv_q4ks_batch", gemv_b),
    ] {
        island
            .compile_msl(&ctx, key, src, key)
            .unwrap_or_else(|e| panic!("compile {key}: {e}"));
    }

    // Warm: the first call pays pipeline JIT and allocation.
    let _ = island.batched_attn_block_selftest(&ctx, 1, h, nh, nkv, hd, ctx_len, eps);

    let time_at = |isl: &mut MetalIsland, b: usize, reps: usize| -> f64 {
        // Median of `reps` — a single sample on a shared GPU is noise.
        let mut ms: Vec<f64> = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t = Instant::now();
            let r = isl.batched_attn_block_selftest(&ctx, b, h, nh, nkv, hd, ctx_len, eps);
            let el = t.elapsed().as_secs_f64() * 1e3;
            match r {
                Ok(_) => ms.push(el),
                Err(e) => {
                    eprintln!("  selftest b={b} FAILED: {e}");
                    return f64::NAN;
                }
            }
        }
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if ms.is_empty() {
            f64::NAN
        } else {
            ms[ms.len() / 2]
        }
    };

    println!("L347 b-row microbenchmark — ONE isolated layer, Qwen3.8-27B geometry");
    println!("  h={h} nh={nh} nkv={nkv} hd={hd} inter=n/a ctx={ctx_len}\n");
    println!("   b |  median ms | vs b=1 | marginal per extra row");
    println!("  ---+------------+--------+-----------------------");
    let base = time_at(&mut island, 1, 9);
    println!("   1 | {base:10.3} |      - | -");
    for b in [2usize, 3, 4] {
        let t = time_at(&mut island, b, 9);
        let marg = (t - base) / (b - 1) as f64;
        println!(
            "   {b} | {t:10.3} | {:5.2}x | {marg:.3} ms  ({:.0} us)",
            t / base,
            marg * 1000.0
        );
    }
    println!("\n  A second row streams NO extra weights and needs <1 ms of arithmetic across");
    println!("  all 64 layers (L342). If the marginal here is ~0, the b-penalty is NOT in the");
    println!("  layer kernels and the megakernel record is where to look next. If it is large,");
    println!("  this harness can bisect it kernel by kernel.");
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("macOS/Metal only");
}
