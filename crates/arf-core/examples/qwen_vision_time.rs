//! Time the Qwen3.8 CPU vision encoder on a synthetic W x H image (no GPU involved).
//!
//!   cargo run --release -p arf-core --example qwen_vision_time -- <mmproj.gguf> [W H] [repeats]
//!
//! Prints the resized size, merged-token count, load time and per-encode wall time, plus the
//! output checksum (so two runs can be compared for identity). `ARF_QWEN_VISION_MAX_TOKENS`
//! changes the pixel budget exactly as it does in the server.
use arf_core::model::qwen_vision::{preprocess, QwenVisionEncoder};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let path = a
        .get(1)
        .expect("usage: qwen_vision_time <mmproj.gguf> [W H] [repeats]");
    let w: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(512);
    let h: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(512);
    let reps: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(1);
    let t0 = std::time::Instant::now();
    let enc = QwenVisionEncoder::load(std::path::Path::new(path)).expect("load mmproj");
    eprintln!("load: {:.2}s", t0.elapsed().as_secs_f64());
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                rgb[(y * w + x) * 3 + c] = ((x * 7 + y * 3 + c * 50) % 256) as u8;
            }
        }
    }
    let tp = std::time::Instant::now();
    let img = preprocess(&enc.cfg, &rgb, w, h);
    eprintln!(
        "preprocess {w}x{h} -> {}x{} ({} patches, {} tokens): {:.3}s",
        img.width,
        img.height,
        img.grid_w * img.grid_h,
        img.n_tokens(),
        tp.elapsed().as_secs_f64()
    );
    for r in 0..reps {
        let te = std::time::Instant::now();
        let out = enc.encode(&img);
        let s: f64 = out.iter().map(|&v| v as f64).sum();
        println!(
            "encode #{r}: {:.3}s  tokens={}  sum={s:.6}",
            te.elapsed().as_secs_f64(),
            out.len() / enc.cfg.proj_dim
        );
    }
}
