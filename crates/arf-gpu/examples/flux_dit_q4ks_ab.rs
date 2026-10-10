//! F1 A/B parity: GpuDiT.forward via the NEW Q4_K_S quantized-resident path vs the legacy
//! bf16 path (ARF_FLUX_FORCE_BF16), on identical fixed inputs. Confirms the quantized
//! wiring is correct — a high cosine means the dequant-in-kernel GEMMs reproduce the bf16
//! result (any small residual is just Q4_K_S-vs-bf16 weight precision, which is EXPECTED and
//! correct since Q4_K_S is the true checkpoint precision bf16 was approximating).
//!
//! Run: cargo run --release -p arf-gpu --example flux_dit_q4ks_ab [grid] [txt]

use arf_gpu::gpu::flux::{DitConfig, GpuDiT};
use arf_gpu::gpu::GpuContext;
use std::sync::Arc;

const MODELS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/flux-schnell/flux1-schnell-Q4_K_S.gguf"
);

fn det_noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    let mut next = || {
        s = s.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        (((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
    };
    (0..n).map(|_| next()).collect()
}

fn cos_rl2(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut dot, mut na, mut nb, mut diff) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += (x as f64).powi(2);
        nb += (y as f64).powi(2);
        diff += ((x - y) as f64).powi(2);
    }
    (dot / (na.sqrt() * nb.sqrt()), (diff.sqrt()) / nb.sqrt())
}

fn run(
    force_bf16: bool,
    grid: usize,
    txt: usize,
    img: &[f32],
    t5: &[f32],
    pooled: &[f32],
    ts: f32,
) -> Vec<f32> {
    if force_bf16 {
        std::env::set_var("ARF_FLUX_FORCE_BF16", "1");
    } else {
        std::env::remove_var("ARF_FLUX_FORCE_BF16");
    }
    let ctx = Arc::new(GpuContext::new().expect("gpu"));
    let cfg = DitConfig::schnell(grid, txt);
    let dit = GpuDiT::load(&ctx, std::path::Path::new(MODELS), cfg).expect("load dit");
    dit.forward(img, t5, pooled, ts)
}

fn main() {
    let grid: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(16); // 256px
    let txt: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let it = grid * grid;
    let img = det_noise(it * 64, 1);
    let t5 = det_noise(txt * 4096, 2);
    let pooled = det_noise(768, 3);
    let ts = 1.0f32;

    println!("loading + running Q4_K_S (new) path…");
    let q = run(false, grid, txt, &img, &t5, &pooled, ts);
    println!("loading + running bf16 (legacy) path…");
    let b = run(true, grid, txt, &img, &t5, &pooled, ts);

    let (cos, rl2) = cos_rl2(&q, &b);
    println!(
        "\nDiT velocity  Q4_K_S vs bf16   cos={cos:.6}  rel_l2={rl2:.4e}  (n={})",
        q.len()
    );
    // bf16 and Q4_K_S are two quantizations of the SAME checkpoint; expect cos ≥ ~0.995.
    if cos >= 0.99 {
        println!("PASS — quantized DiT wiring is correct (matches bf16 to weight-precision).");
        std::process::exit(0);
    } else {
        println!("FAIL — wiring bug (cos too low to be mere quant precision).");
        std::process::exit(1);
    }
}
