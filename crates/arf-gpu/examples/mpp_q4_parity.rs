//! The SHIPPED MPP Q4 matmul (row-major strided view, no copy of the weights) inside the engine,
//! against an f64 reference.
//!
//! Two questions, neither of which the Swift probes could answer:
//!  1. does the kernel produce the right numbers through OUR compile path and OUR layout?
//!  2. what does narrowing f32 activations to per-row-scaled half cost, on activations shaped
//!     like a real residual stream — including the outlier channels that would overflow a plain
//!     `half` cast?
//!
//! Run: cargo run --release -p arf-gpu --example mpp_q4_parity
#[cfg(target_os = "macos")]
fn main() {
    use arf_core::tensor::Q4KSMatrix;
    use arf_gpu::gpu::concurrent_metal::MetalIsland;
    use arf_gpu::gpu::GpuContext;

    let ctx = GpuContext::new().expect("gpu context");
    let mut island = MetalIsland::new(&ctx).expect("metal island");

    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut rnd = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    };

    // (label, out, in) — the 27B's real projection shapes.
    let shapes = [
        ("ffn gate/up 5120->17408", 17408usize, 5120usize),
        ("ffn down   17408->5120", 5120, 17408),
        ("attn q     5120->12288", 12288, 5120),
        ("attn k/v   5120->1024", 1024, 5120),
        // One tile of output: what is left is almost purely the activation narrowing + sums.
        ("narrow-only 5120->256", 256, 5120),
        ("narrow-only 17408->256", 256, 17408),
        ("gdn qkv    5120->10240", 10240, 5120),
        ("gdn z/out  5120<->6144", 6144, 5120),
    ];
    // Activation profiles. "outliers" plants a few channels thousands of times larger than the
    // rest — the shape a plain half cast cannot hold (the largest exceeds 65,504).
    let profiles: [(&str, fn(usize, f32) -> f32); 3] = [
        ("unit-scale", |_, r| r),
        ("post-norm, wide spread", |i, r| {
            r * (1.0 + (i % 97) as f32 * 0.35)
        }),
        ("outlier channels (max ~2e5)", |i, r| {
            if i % 1701 == 3 {
                r * 2.0e5
            } else {
                r * 4.0
            }
        }),
    ];

    let mut worst = 0.0f64;
    for (label, n, k) in shapes {
        let wf: Vec<f32> = (0..n * k).map(|_| rnd() * 0.05).collect();
        let q = Q4KSMatrix::from_f32(&wf, n, k);
        for (pname, f) in profiles {
            for rows in [8usize, 16, 128] {
                let x: Vec<f32> = (0..rows * k).map(|i| f(i % k, rnd())).collect();
                let (y, gpu_s, wd) = match island.mpp_q4_selftest(&ctx, &q, &x, rows) {
                    Ok(v) => v,
                    Err(e) => {
                        println!("MPP tensor ops unavailable on this OS/GPU — feature off: {e}");
                        return;
                    }
                };
                // f64 reference on a sample of outputs (every 61st), all 8 rows.
                let (mut max_rel, mut max_abs, mut sum_sq_err, mut sum_sq_ref) =
                    (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                for r in 0..rows {
                    for o in (0..n).step_by(61) {
                        let mut acc = 0.0f64;
                        for i in 0..k {
                            acc += x[r * k + i] as f64 * wd[o * k + i] as f64;
                        }
                        let got = y[r * n + o] as f64;
                        let err = (got - acc).abs();
                        max_abs = max_abs.max(err);
                        sum_sq_err += err * err;
                        sum_sq_ref += acc * acc;
                        // relative to the row's typical output magnitude, not to a near-zero element
                        max_rel = max_rel.max(err);
                    }
                }
                let rms_ref = (sum_sq_ref / (rows as f64 * n.div_ceil(61) as f64)).sqrt();
                let rel_rms = (sum_sq_err / sum_sq_ref.max(1e-300)).sqrt();
                let rel_max = max_rel / rms_ref.max(1e-300);
                worst = worst.max(rel_max);
                // 2026-09-23: `f64::max` DROPS NaN — a kernel writing NaN everywhere printed
                // "worst 2.18e-3" and exited 0 through this gate. Non-finite output is a failure.
                if !y.iter().all(|v| v.is_finite()) || !rel_rms.is_finite() {
                    worst = f64::INFINITY;
                }
                println!(
                "{label:26} | {pname:30} | {rows:2} rows | err/rms(ref): rms {rel_rms:.2e}  max {rel_max:.2e} | finite {} | gpu {:.0} us",
                y.iter().all(|v| v.is_finite()),
                gpu_s * 1e6
            );
                let _ = max_abs;
            }
        }
    }
    println!("worst max-error / rms(reference) over all arms: {worst:.2e}");
    std::process::exit(if worst < 5e-3 { 0 } else { 1 });
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("mpp_q4tile_parity: macOS only (Metal 4 MPP tensor ops)");
}
