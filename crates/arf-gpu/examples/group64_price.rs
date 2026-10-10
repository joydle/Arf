//! WHAT DOES A 64-WIDE QUANT GROUP ACTUALLY COST? — from bf16, the way a quantiser really works.
//!
//! another engine ships group-64 (`q4_mpp_tiles.h:5`, `quant_groups = input_size / 64`) with a **bfloat**
//! scale and bias per group (:96-98). Arf ships Q4_K_S: group-32, but its scale is TWO-LEVEL (a
//! u8 sub-scale against a per-256 f32), which is what makes a 64-wide MPP sub-product impossible
//! without re-grouping. `ARF_TMP_PAIR=64` priced a re-fit of already-4-bit weights at +0.71%
//! perplexity — an UPPER BOUND, because it inherited Q4_K's coarse scale grid AND a second
//! rounding. This measures the real thing: one pass from bf16, three formats, the same weights.
//!
//! The metric is the error of the MATMUL OUTPUT, not of the weights: that is what reaches the
//! model. Inputs carry a few hot channels, as real activations do (the 27B's residual stream has
//! channels ~90x its RMS — measured 2026-09-21).
//!
//! Run: cargo run --release -p arf-gpu --example group64_price -- [safetensors] [tensors...]

use arf_core::tensor::Q4KSMatrix;

/// bf16 scale+bias per `group` inputs, quantised once from f32 — another engine's format.
fn quant_bf16_scales(w: &[f32], rows: usize, cols: usize, group: usize) -> Vec<f32> {
    let bf = |x: f32| f32::from_bits(((x.to_bits() + 0x8000) >> 16) << 16); // round-to-nearest bf16
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for g in (0..cols).step_by(group) {
            let blk = &w[r * cols + g..][..group];
            let (lo, hi) = blk
                .iter()
                .fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
            let scale = bf(if hi > lo { (hi - lo) / 15.0 } else { 1.0 });
            let bias = bf(lo);
            let s = if scale > 0.0 { scale } else { 1.0 };
            for (j, &x) in blk.iter().enumerate() {
                let code = ((x - bias) / s).round().clamp(0.0, 15.0);
                out[r * cols + g + j] = code * scale + bias;
            }
        }
    }
    out
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .unwrap_or_else(|| "models/qwen3.8-27b-dflash2/model.safetensors".into());
    let names: Vec<String> = args.collect();
    let names = if names.is_empty() {
        vec![
            "layers.0.mlp.down_proj.weight".to_string(),
            "layers.2.self_attn.q_proj.weight".to_string(),
            "layers.4.mlp.gate_proj.weight".to_string(),
            "fc.weight".to_string(),
        ]
    } else {
        names
    };
    let w = arf_core::model::weights::read_weights_lazy(&[std::path::PathBuf::from(&path)])
        .expect("safetensors");

    let mut seed = 0x243F_6A88_85A3_08D3u64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f32 / (1u64 << 53) as f32 - 0.5
    };
    println!(
        "{:<44} {:>12} {:>9} {:>9} {:>9}",
        "tensor", "shape", "Q4KS-32", "bf16-64", "bf16-32"
    );
    let (mut t32, mut b64, mut b32, mut n) = (0.0f64, 0.0f64, 0.0f64, 0usize);
    for name in &names {
        let Some(d) = w.shape(name) else {
            eprintln!("{name}: not found — skipped");
            continue;
        };
        let (rows, cols) = (d[0], d[1]);
        let m = w.bf16_matrix(name, rows, cols).expect("bf16");
        let f: Vec<f32> = m
            .bits()
            .iter()
            .map(|b| f32::from_bits((*b as u32) << 16))
            .collect();
        // The three formats, all from the SAME bf16 source.
        let q4ks = Q4KSMatrix::from_f32(&f, rows, cols).to_f32();
        let g64 = quant_bf16_scales(&f, rows, cols, 64);
        let g32 = quant_bf16_scales(&f, rows, cols, 32);
        // Activations: bell-shaped, every 41st channel 30x (the residual stream's hot channels).
        let xs: Vec<Vec<f32>> = (0..12)
            .map(|_| {
                (0..cols)
                    .map(|c| {
                        let g: f32 = (0..6).map(|_| rnd()).sum::<f32>() * 0.05;
                        if c % 41 == 0 {
                            g * 30.0
                        } else {
                            g
                        }
                    })
                    .collect()
            })
            .collect();
        let err = |q: &[f32]| -> f64 {
            let (mut e, mut d) = (0.0f64, 0.0f64);
            for x in &xs {
                for r in (0..rows).step_by((rows / 64).max(1)) {
                    let (mut a, mut b) = (0.0f64, 0.0f64);
                    for c in 0..cols {
                        a += (x[c] * f[r * cols + c]) as f64;
                        b += (x[c] * q[r * cols + c]) as f64;
                    }
                    e += (a - b) * (a - b);
                    d += a * a;
                }
            }
            (e / d).sqrt() * 100.0
        };
        let (a, b, c) = (err(&q4ks), err(&g64), err(&g32));
        println!(
            "{name:<44} {:>12} {a:>8.3}% {b:>8.3}% {c:>8.3}%",
            format!("{rows}x{cols}")
        );
        t32 += a;
        b64 += b;
        b32 += c;
        n += 1;
    }
    if n > 0 {
        let (a, b, c) = (t32 / n as f64, b64 / n as f64, b32 / n as f64);
        println!(
            "\nmean matmul-output error vs the bf16 original:\n  Q4_K_S, group 32, two-level u8 scale (ships today): {a:.3}%\n  bf16 scale+bias, group 64 (group-64 affine format):        {b:.3}%\n  bf16 scale+bias, group 32:                          {c:.3}%"
        );
        println!(
            "\ngroup-64 with bf16 scales vs what ships today: {:+.1}% relative error",
            (b / a - 1.0) * 100.0
        );
    }
}
