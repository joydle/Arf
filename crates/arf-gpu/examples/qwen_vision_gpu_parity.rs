//! GATE for the Qwen3.8 GPU vision encoder: run the GPU encoder and the CPU reference on the SAME
//! preprocessed image, report their difference over the whole `[tokens, proj_dim]` output, and
//! time both.
//!
//!   cargo run --release -p arf-gpu --example qwen_vision_gpu_parity -- \
//!       models/qwen3.8-27b-gsqrco/mmproj-Qwen3.8-27B-BF16.gguf [WxH ...] [--reps N]
//!       [--image some.png] [--no-cpu]
//!
//!   cargo run --release -p arf-gpu --example qwen_vision_gpu_parity -- --synthetic
//!       (no mmproj: the synthetic tiny + medium encoders over sizes that hit every tile tail)
//!
//! Sizes default to `512x384 1024x768`. Without `--image` the input is the test pattern
//! `(x*7 + y*3 + c*50) % 256`; with it, that file is resized to each size first (the encoder's
//! own preprocess then applies its pixel budget, as the server does). Per size it prints:
//!   - the GPU wall per encode (host input prep + GPU + readback — what the server pays), the
//!     first (cold: scratch allocation) and min/median of `--reps` warm ones;
//!   - the CPU wall (one encode) unless `--no-cpu`;
//!   - max |gpu - cpu|, rms(gpu - cpu), rms(cpu), rel = rms(diff)/rms(cpu), cosine, the worst
//!     token, and whether two GPU runs are bit-identical;
//!   - GATE PASS/FAIL: rel <= 1e-3 and cosine >= 0.99999 and finite (see `REL_BOUND`); two GPU
//!     runs that are not bit-identical also fail.
//!
//! Exit code 1 if any size fails the gate or the GPU encoder does not come up.

#[cfg(target_os = "macos")]
fn main() {
    use arf_core::model::qwen_vision::{preprocess, synthetic, QwenImage, QwenVisionEncoder};
    use arf_gpu::gpu::metal::qwen_vision::QwenVisionGpu;
    use std::time::Instant;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut sizes: Vec<(usize, usize)> = Vec::new();
    let mut reps = 5usize;
    let mut image: Option<String> = None;
    let mut no_cpu = false;
    let mut synthetic_mode = false;
    let mut mmproj: Option<String> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--reps" => reps = it.next().and_then(|v| v.parse().ok()).expect("--reps N"),
            "--image" => image = Some(it.next().expect("--image PATH").clone()),
            "--no-cpu" => no_cpu = true,
            "--synthetic" => synthetic_mode = true,
            s if s.contains('x') && s.split('x').all(|p| p.parse::<usize>().is_ok()) => {
                let mut p = s.split('x').map(|v| v.parse::<usize>().unwrap());
                sizes.push((p.next().unwrap(), p.next().unwrap()));
            }
            s => mmproj = Some(s.to_string()),
        }
    }

    let pattern = |w: usize, h: usize| -> Vec<u8> {
        let mut rgb = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    rgb[(y * w + x) * 3 + c] = ((x * 7 + y * 3 + c * 50) % 256) as u8;
                }
            }
        }
        rgb
    };

    // (label, encoder, preprocessed images)
    let mut cases: Vec<(String, QwenVisionEncoder, Vec<QwenImage>)> = Vec::new();
    if synthetic_mode {
        let tiny = synthetic::tiny();
        let imgs = [(24, 16), (40, 32), (8, 8)]
            .iter()
            .map(|&(w, h)| synthetic::pattern_image(&tiny, w, h))
            .collect();
        cases.push(("synthetic tiny (head_dim 8)".into(), tiny, imgs));
        let med = synthetic::medium();
        let imgs = [(96, 64), (160, 128), (64, 64), (352, 320)]
            .iter()
            .map(|&(w, h)| synthetic::pattern_image(&med, w, h))
            .collect();
        cases.push(("synthetic medium (head_dim 72, ffn 200)".into(), med, imgs));
    } else {
        let path =
            mmproj.expect("usage: qwen_vision_gpu_parity <mmproj.gguf> [WxH ...] | --synthetic");
        if sizes.is_empty() {
            sizes = vec![(512, 384), (1024, 768)];
        }
        let t0 = Instant::now();
        let enc = QwenVisionEncoder::load(std::path::Path::new(&path)).expect("load mmproj");
        eprintln!("CPU encoder load: {:.2}s", t0.elapsed().as_secs_f64());
        let src = image.as_ref().map(|p| {
            image::open(p)
                .unwrap_or_else(|e| panic!("{p}: {e}"))
                .to_rgb8()
        });
        let imgs = sizes
            .iter()
            .map(|&(w, h)| {
                let rgb = match &src {
                    Some(s) => image::imageops::resize(
                        s,
                        w as u32,
                        h as u32,
                        image::imageops::FilterType::CatmullRom,
                    )
                    .into_raw(),
                    None => pattern(w, h),
                };
                let tp = Instant::now();
                let img = preprocess(&enc.cfg, &rgb, w, h);
                eprintln!(
                    "preprocess {w}x{h} -> {}x{} ({} patches, {} tokens): {:.3}s (CPU, shared by both paths)",
                    img.width,
                    img.height,
                    img.grid_w * img.grid_h,
                    img.n_tokens(),
                    tp.elapsed().as_secs_f64()
                );
                img
            })
            .collect();
        cases.push((path.clone(), enc, imgs));
    }

    let mut failed = false;
    for (label, enc, imgs) in &cases {
        let t0 = Instant::now();
        let gpu = match QwenVisionGpu::new(enc) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("FAIL: GPU encoder did not come up for {label}: {e}");
                std::process::exit(1);
            }
        };
        println!(
            "== {label}: GPU encoder up in {:.2}s on {} ({:.3} GB weights)",
            t0.elapsed().as_secs_f64(),
            gpu.device_name(),
            gpu.weight_bytes() as f64 / 1e9
        );
        for img in imgs {
            let tag = format!("{}x{} ({} tokens)", img.width, img.height, img.n_tokens());
            let tc = Instant::now();
            let first = match gpu.encode(img) {
                Ok(v) => v,
                Err(e) => {
                    println!("{tag}: GATE FAIL — GPU encode error: {e}");
                    failed = true;
                    continue;
                }
            };
            let cold = tc.elapsed().as_secs_f64();
            let mut times = Vec::new();
            let mut identical = true;
            for _ in 0..reps {
                let t = Instant::now();
                let v = gpu.encode(img).expect("GPU encode (warm)");
                times.push(t.elapsed().as_secs_f64());
                identical &= v
                    .iter()
                    .zip(&first)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let (tmin, tmed) = if times.is_empty() {
                (cold, cold)
            } else {
                (times[0], times[times.len() / 2])
            };
            println!(
                "{tag}: GPU cold {:.1} ms | warm min {:.1} ms, median {:.1} ms over {} | repeat bit-identical: {}",
                cold * 1e3,
                tmin * 1e3,
                tmed * 1e3,
                times.len(),
                if identical { "yes" } else { "NO" }
            );
            // No atomics and fixed reduction orders: two runs that differ are a race, a FAIL.
            failed |= !identical;
            if no_cpu {
                continue;
            }
            let tcpu = Instant::now();
            let cpu = enc.encode(img);
            let cpu_s = tcpu.elapsed().as_secs_f64();
            let st = compare(&first, &cpu, enc.cfg.proj_dim);
            let pass = st.finite && st.rel <= REL_BOUND && st.cos >= COS_BOUND;
            failed |= !pass;
            println!(
                "{tag}: CPU {:.2} s (GPU/CPU speedup {:.1}x) | max|d| {:.3e} rms(d) {:.3e} rms(cpu) {:.3e} rel {:.3e} cos {:.8} worst token {} (max|d| {:.3e}) | GATE {}",
                cpu_s,
                cpu_s / tmed,
                st.max_abs,
                st.rms_d,
                st.rms_ref,
                st.rel,
                st.cos,
                st.worst_tok,
                st.worst_tok_abs,
                if pass { "PASS" } else { "FAIL" }
            );
        }
    }
    if failed {
        std::process::exit(1);
    }
}

/// rms(gpu - cpu) / rms(cpu) bound. Both paths are f32 end to end with bit-identical inputs and
/// weights; they differ only in summation order, f32 vs f64 LayerNorm sums, the softmax's
/// normalisation point and the transcendental implementations. An indexing or tiling bug moves
/// this to O(1); f32 reordering over 27 residual blocks should leave it near 1e-5..1e-4
/// (NOT MEASURED when written; the first run measured rel 1.95e-5 at 512x384 and 2.57e-5 at
/// 1024x768 — measured 2026-09-27).
#[cfg(target_os = "macos")]
const REL_BOUND: f64 = 1e-3;
#[cfg(target_os = "macos")]
const COS_BOUND: f64 = 0.99999;

#[cfg(target_os = "macos")]
struct Stats {
    max_abs: f64,
    rms_d: f64,
    rms_ref: f64,
    rel: f64,
    cos: f64,
    worst_tok: usize,
    worst_tok_abs: f64,
    finite: bool,
}

#[cfg(target_os = "macos")]
fn compare(gpu: &[f32], cpu: &[f32], width: usize) -> Stats {
    assert_eq!(gpu.len(), cpu.len(), "GPU and CPU output lengths differ");
    let (mut d2, mut r2, mut g2, mut dot, mut max_abs) = (0f64, 0f64, 0f64, 0f64, 0f64);
    let (mut worst_tok, mut worst_tok_abs) = (0usize, 0f64);
    let mut finite = true;
    for (i, (&g, &c)) in gpu.iter().zip(cpu).enumerate() {
        finite &= g.is_finite();
        let (g, c) = (g as f64, c as f64);
        let d = (g - c).abs();
        d2 += d * d;
        r2 += c * c;
        g2 += g * g;
        dot += g * c;
        if d > max_abs {
            max_abs = d;
        }
        if d > worst_tok_abs {
            worst_tok_abs = d;
            worst_tok = i / width;
        }
    }
    let n = gpu.len().max(1) as f64;
    Stats {
        max_abs,
        rms_d: (d2 / n).sqrt(),
        rms_ref: (r2 / n).sqrt(),
        rel: (d2 / r2.max(f64::MIN_POSITIVE)).sqrt(),
        cos: dot / (g2.sqrt() * r2.sqrt()).max(f64::MIN_POSITIVE),
        worst_tok,
        worst_tok_abs,
        finite,
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("qwen_vision_gpu_parity: the GPU vision encoder is Metal-only");
}
