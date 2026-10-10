//! GATE + TIMING for `ARF_G64_PFS` (2026-09-27) — another engine's sg4 prefill matmul
//! (`g64pfs_mm`, shaders/metal/matmul_g64_pfs_msl.metal) against the default prefill matmul
//! (`g64pf2_mm`), NO MODEL. NOT RUN when written:
//! nothing here has produced a number yet.
//!
//! For each row count (default 128, 256, 512) and the verify shapes — 5120->17408 (gate/up),
//! 17408->5120 (down), 5120->10240 (GDN qkv), 5120->6144, 6144->5120 — random group-64 weights in
//! the StorageN=256 tiling (f16 scale / bias, as `G64Mtl` holds them) and random f32 activations
//! with hot channels go through BOTH shipped encode routines on the SAME inputs
//! (`MetalIsland::g64_pfs_selftest`; `g64pf2_encode` in the record's windows of <= 256 rows, the
//! port in one call):
//!   * every output is sentinel-filled (NaN) first — a kernel that never ran cannot pass (rule 8),
//!     and the run must report the expected dispatch counts;
//!   * both vs an f64 CPU reference (the exact product of the f32 activations), all rows, 256
//!     sampled columns (tile edges included): rms error / rms(ref) <= 2^-8 (one bf16 ulp), worst
//!     element <= 2^-5 x rms(ref);
//!   * the port vs the f64 product of ITS OWN operand (bf16-rounded x): rms <= 1e-4 — it computes
//!     exactly what it claims, the bf16 input aside;
//!   * the two kernels against each other on EVERY output, the same bounds;
//!   * the fused forms (ported, NOT wired into the record): residual == plain + r (f32);
//!     up-silu vs bf16(silu(gate) x ref) on the sampled columns; its sums vs its own bf16 output.
//!
//! Then the median GPU time per call (command-buffer GPUStartTime/GPUEndTime; the four arms — pf2
//! and the port, each with its operand pass and with the operand reused — interleaved per command
//! buffer; 64 samples of 4 calls on rotated weight copies, >= 512 MiB of them so the weight is not
//! cache-resident) and the effective TFLOPS. Standalone numbers only: the claim, if any, is the
//! in-context TTFT A/B (`ARF_G64_PFS=1` vs unset, interleaved).
//!
//! Run on a quiet GPU (nothing else resident), from the repo root:
//!   cargo run --release -p arf-gpu --example g64_pfs_probe              # rows 128 256 512
//!   cargo run --release -p arf-gpu --example g64_pfs_probe -- 256 2048   # other row counts

#[cfg(target_os = "macos")]
fn main() {
    use arf_gpu::gpu::concurrent_metal::{f16_bits_to_f32, f32_to_f16_bits, MetalIsland};
    use arf_gpu::gpu::metal::{G64PFS_SENTINEL_BF16, G64PFS_SENTINEL_F32};
    use arf_gpu::gpu::GpuContext;

    const SAMPLES: usize = 64;
    const SAMPLED_COLS: usize = 256;
    const BF16_ULP: f64 = 1.0 / 256.0; // 2^-8
    const WORST: f64 = 1.0 / 32.0; // 2^-5 of the reference's rms
    let rows_list: Vec<usize> = {
        let v: Vec<usize> = std::env::args()
            .skip(1)
            .map(|a| a.parse().expect("row counts (multiples of 32)"))
            .collect();
        if v.is_empty() {
            vec![128, 256, 512]
        } else {
            v
        }
    };
    let shapes: [(usize, usize, &str); 5] = [
        (5120, 17408, "gate/up 5120->17408"),
        (17408, 5120, "down 17408->5120"),
        (5120, 10240, "GDN qkv 5120->10240"),
        (5120, 6144, "5120->6144"),
        (6144, 5120, "6144->5120"),
    ];
    let ctx = GpuContext::new().expect("gpu context");
    let mut island = MetalIsland::new(&ctx).expect("metal island (need a Metal backend)");

    let mut seed = 0x243F_6A88_85A3_08D3u64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    // round to nearest even, as Metal's bfloat(float)
    let bf16 = |x: f32| -> f32 {
        let b = x.to_bits();
        f32::from_bits((b.wrapping_add(((b >> 16) & 1) + 0x7FFF)) & 0xFFFF_0000)
    };
    let silu = |g: f64| g / (1.0 + (-g).exp());
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let mut fails: Vec<String> = Vec::new();

    for &(k, n, label) in &shapes {
        let groups = k / 64;
        // weights, generated directly in the tiled layout: codes 0..15, scale in [0.002, 0.012),
        // bias in [-0.03, 0.02) (the ranges a standalone probe used)
        let w: Vec<u8> = (0..n * k / 2).map(|_| rnd() as u8).collect();
        let s: Vec<u16> = (0..n * groups)
            .map(|_| f32_to_f16_bits(0.002 + (rnd() % 1000) as f32 / 1e5))
            .collect();
        let b: Vec<u16> = (0..n * groups)
            .map(|_| f32_to_f16_bits(-0.03 + (rnd() % 1000) as f32 / 2e4))
            .collect();
        // sampled columns: the edges of the 128-column tiles and 256-column storage tiles, then random
        let mut cols: Vec<usize> = [0, 1, 63, 64, 127, 128, 129, 255, 256, 383, 384]
            .into_iter()
            .chain([n - 129, n - 128, n - 65, n - 64, n - 1])
            .collect();
        while cols.len() < SAMPLED_COLS {
            cols.push(rnd() as usize % n);
        }
        cols.sort_unstable();
        cols.dedup();
        // the dequantized weight of each sampled column, f64, from the TILED bytes (what the GPU reads)
        let wd: Vec<Vec<f64>> = cols
            .iter()
            .map(|&col| {
                (0..k)
                    .map(|kk| {
                        let p = ((col / 256) * groups + kk / 64) * 256 + col % 256;
                        let byte = w[p * 32 + (kk % 64) / 2];
                        let q = if kk % 2 == 0 { byte & 15 } else { byte >> 4 };
                        f64::from(f16_bits_to_f32(s[p])) * f64::from(q)
                            + f64::from(f16_bits_to_f32(b[p]))
                    })
                    .collect()
            })
            .collect();

        for &rows in &rows_list {
            let x: Vec<f32> = (0..rows * k)
                .map(|i| {
                    let v = ((rnd() % 2001) as f32 - 1000.0) / 250.0;
                    if i % 97 == 5 {
                        v * 30.0
                    } else {
                        v
                    }
                })
                .collect();
            let residual: Vec<f32> = (0..rows * n)
                .map(|_| ((rnd() % 2001) as f32 - 1000.0) / 500.0)
                .collect();
            let gate: Vec<f32> = (0..rows * n)
                .map(|_| ((rnd() % 2001) as f32 - 1000.0) / 250.0)
                .collect();
            let r = match island
                .g64_pfs_selftest(&ctx, n, k, &w, &s, &b, &x, rows, &residual, &gate, SAMPLES)
            {
                Ok(r) => r,
                Err(e) => {
                    let m = format!("{label} rows {rows}: selftest error: {e}");
                    println!("  {m}");
                    fails.push(m);
                    continue;
                }
            };
            let tag = format!("{label} rows {rows}");
            let mut check = |ok: bool, what: String| {
                if !ok {
                    println!("    FAIL {what}");
                    fails.push(format!("{tag}: {what}"));
                }
            };

            // rule 8: the kernels ran, and wrote every output
            let windows = rows.div_ceil(256) as u64;
            check(
                r.dispatched == (windows, 3),
                format!(
                    "dispatches (pf2, pfs) = {:?}, expected ({windows}, 3)",
                    r.dispatched
                ),
            );
            let bad_f32 = |v: &[f32]| {
                v.iter()
                    .filter(|x| x.to_bits() == G64PFS_SENTINEL_F32 || !x.is_finite())
                    .count()
            };
            let (b1, b2, b3) = (bad_f32(&r.pf2), bad_f32(&r.pfs), bad_f32(&r.pfs_residual));
            let b4 = r
                .pfs_up_silu
                .iter()
                .filter(|&&h| h == G64PFS_SENTINEL_BF16 || !bf16_bits(h).is_finite())
                .count();
            let b5 = bad_f32(&r.pfs_up_silu_sums);
            check(
                b1 + b2 + b3 + b4 + b5 == 0,
                format!("sentinel / non-finite outputs: pf2 {b1} pfs {b2} residual {b3} up-silu {b4} sums {b5}"),
            );

            // f64 references on the sampled columns, all rows: exact x, and the port's own bf16 x
            let xb: Vec<f32> = x.iter().map(|&v| bf16(v)).collect();
            let nc = cols.len();
            let mut ref_exact = vec![0f64; rows * nc];
            let mut ref_bf = vec![0f64; rows * nc];
            std::thread::scope(|sc| {
                let per = rows.div_ceil(threads);
                for ((re, rb), r0) in ref_exact
                    .chunks_mut(per * nc)
                    .zip(ref_bf.chunks_mut(per * nc))
                    .zip((0..rows).step_by(per))
                {
                    let (x, xb, wd) = (&x, &xb, &wd);
                    sc.spawn(move || {
                        for (i, (oe, ob)) in re.chunks_mut(nc).zip(rb.chunks_mut(nc)).enumerate() {
                            let row = r0 + i;
                            let (xr, xbr) = (&x[row * k..][..k], &xb[row * k..][..k]);
                            for (j, wc) in wd.iter().enumerate() {
                                let (mut e, mut f) = (0f64, 0f64);
                                for kk in 0..k {
                                    e += f64::from(xr[kk]) * wc[kk];
                                    f += f64::from(xbr[kk]) * wc[kk];
                                }
                                oe[j] = e;
                                ob[j] = f;
                            }
                        }
                    });
                }
            });
            let rms = |v: &[f64]| (v.iter().map(|a| a * a).sum::<f64>() / v.len() as f64).sqrt();
            // (rms error / rms(ref), worst |error| / rms(ref)) of a kernel output on the samples
            let err = |out: &dyn Fn(usize, usize) -> f64, re: &[f64]| {
                let (mut e2, mut worst) = (0f64, 0f64);
                for row in 0..rows {
                    for (j, &col) in cols.iter().enumerate() {
                        let d = out(row, col) - re[row * nc + j];
                        e2 += d * d;
                        worst = worst.max(d.abs());
                    }
                }
                let s = rms(re);
                ((e2 / re.len() as f64).sqrt() / s, worst / s)
            };
            let (pf2_rms, pf2_worst) = err(&|row, col| f64::from(r.pf2[row * n + col]), &ref_exact);
            let (pfs_rms, pfs_worst) = err(&|row, col| f64::from(r.pfs[row * n + col]), &ref_exact);
            let (own_rms, _) = err(&|row, col| f64::from(r.pfs[row * n + col]), &ref_bf);
            check(
                pf2_rms <= BF16_ULP && pf2_worst <= WORST,
                format!("g64pf2_mm vs f64: rms {pf2_rms:.2e} worst {pf2_worst:.2e}"),
            );
            check(
                pfs_rms <= BF16_ULP && pfs_worst <= WORST,
                format!("g64pfs_mm vs f64: rms {pfs_rms:.2e} worst {pfs_worst:.2e}"),
            );
            check(
                own_rms <= 1e-4,
                format!("g64pfs_mm vs f64 of its own bf16 operand: rms {own_rms:.2e} (> 1e-4)"),
            );
            // the two kernels, every output
            let (mut d2, mut p2, mut dworst) = (0f64, 0f64, 0f64);
            for (a, c) in r.pfs.iter().zip(&r.pf2) {
                let d = f64::from(*a) - f64::from(*c);
                d2 += d * d;
                p2 += f64::from(*c) * f64::from(*c);
                dworst = dworst.max(d.abs());
            }
            let prms = (p2 / r.pf2.len() as f64).sqrt();
            let (x_rms, x_worst) = ((d2 / p2).sqrt(), dworst / prms);
            check(
                x_rms <= BF16_ULP && x_worst <= WORST,
                format!(
                    "g64pfs_mm vs g64pf2_mm (all outputs): rms {x_rms:.2e} worst {x_worst:.2e}"
                ),
            );
            // residual form == plain + residual, every output
            let (mut res_worst, mut res_exact) = (0f64, 0usize);
            for ((o, p), q) in r.pfs_residual.iter().zip(&r.pfs).zip(&residual) {
                let want = p + q;
                res_exact += usize::from(o.to_bits() == want.to_bits());
                res_worst = res_worst.max((f64::from(*o) - f64::from(want)).abs());
            }
            let res_scale = prms + rms(&residual.iter().map(|&v| f64::from(v)).collect::<Vec<_>>());
            check(
                res_worst <= 1e-4 * res_scale,
                format!(
                    "residual form vs plain + r: worst {res_worst:.2e} (scale {res_scale:.2e})"
                ),
            );
            // up-silu-sums: the bf16 out vs bf16(silu(gate) * ref) on the samples ...
            let want_up: Vec<f64> = (0..rows * nc)
                .map(|i| {
                    let (row, j) = (i / nc, i % nc);
                    silu(f64::from(gate[row * n + cols[j]])) * ref_bf[i]
                })
                .collect();
            let (up_rms, _) = err(
                &|row, col| f64::from(bf16_bits(r.pfs_up_silu[row * n + col])),
                &want_up,
            );
            // per element: within one bf16 ulp of the exact value (its own rounding is half of one)
            let up_floor = 1e-5 * rms(&want_up);
            let up_bad = (0..rows * nc)
                .filter(|&i| {
                    let got = f64::from(bf16_bits(r.pfs_up_silu[(i / nc) * n + cols[i % nc]]));
                    (got - want_up[i]).abs() > BF16_ULP * want_up[i].abs() + up_floor
                })
                .count();
            check(
                up_rms <= BF16_ULP && up_bad == 0,
                format!(
                    "up-silu form vs silu(gate) x ref: rms {up_rms:.2e}, {up_bad} of {} beyond one bf16 ulp",
                    rows * nc
                ),
            );
            // ... and its sums vs its OWN bf16 output, every (row, 64-group)
            let mut sums_bad = 0usize;
            for row in 0..rows {
                for g in 0..n / 64 {
                    let vals = &r.pfs_up_silu[row * n + g * 64..][..64];
                    let (sum, mag) = vals.iter().fold((0f64, 0f64), |(a, m), &h| {
                        let v = f64::from(bf16_bits(h));
                        (a + v, m + v.abs())
                    });
                    let got =
                        f64::from(r.pfs_up_silu_sums[((row / 32) * (n / 64) + g) * 32 + row % 32]);
                    if (got - sum).abs() > 1e-5 * mag + 1e-30 {
                        sums_bad += 1;
                    }
                }
            }
            check(
                sums_bad == 0,
                format!(
                    "up-silu output sums: {sums_bad} of {} differ from the bf16 output's own sums",
                    rows * n / 64
                ),
            );

            let flops = 2.0 * (rows * n * k) as f64;
            let tf = |t: f64| flops / t / 1e12;
            println!(
                "{label:<22} rows {rows:>4} | vs f64 rms: pf2 {pf2_rms:.1e} pfs {pfs_rms:.1e} (own operand {own_rms:.1e}) | pfs-pf2 {x_rms:.1e} | residual exact {res_exact}/{} | up-silu {up_rms:.1e}",
                rows * n
            );
            println!(
                "{:<22}           | GPU us/call, operand pass incl.: pf2 {:8.1} ({:5.2} TFLOPS)  pfs {:8.1} ({:5.2})  pf2/pfs {:.3} | operand reused: pf2 {:8.1}  pfs {:8.1}  pf2/pfs {:.3}   [{} x {} calls, {} weight copies]",
                "",
                r.t_pf2_full * 1e6,
                tf(r.t_pf2_full),
                r.t_pfs_full * 1e6,
                tf(r.t_pfs_full),
                r.t_pf2_full / r.t_pfs_full,
                r.t_pf2_mm * 1e6,
                r.t_pfs_mm * 1e6,
                r.t_pf2_mm / r.t_pfs_mm,
                r.samples,
                r.calls_per_sample,
                r.copies,
            );
        }
    }
    println!();
    if fails.is_empty() {
        println!("GATE PASS: g64pfs_mm (+ residual, up-silu-sums) correct on every shape and row count; both kernels dispatched, no sentinel left.");
    } else {
        println!("GATE FAIL: {} check(s) failed:", fails.len());
        for f in &fails {
            println!("  {f}");
        }
        std::process::exit(1);
    }
}

/// bf16 bits -> f32.
#[cfg(target_os = "macos")]
fn bf16_bits(h: u16) -> f32 {
    f32::from_bits(u32::from(h) << 16)
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("g64_pfs_probe: the MSL kernels are macOS-only (Metal island). Skipped.");
}
