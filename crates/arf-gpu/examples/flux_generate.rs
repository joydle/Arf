//! FLUX end-to-end: prompt → image, all on GPU. Tokenizes with the real CLIP-L (BPE) + T5
//! (Unigram) tokenizers, runs the FluxPipeline (CLIP+T5 → DiT Euler loop → VAE), and writes a PNG.
//! This is the "first real image" artifact + the reference the serve /v1/images/generations
//! endpoint reuses.
//!
//! Usage: cargo run --release -p arf-gpu --example flux_generate -- "a cat" out.png [steps] [size]

use std::sync::Arc;

use arf_gpu::gpu::flux::{FluxPipeline, GenConfig};
use arf_gpu::gpu::GpuContext;
use tokenizers::Tokenizer;

const MODELS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/flux-schnell");

/// CLIP-L tokens: BOS(49406) + text + EOS(49407), padded to 77 with EOS. Returns (tokens, eos_pos).
fn clip_tokens(tok: &Tokenizer, prompt: &str) -> (Vec<u32>, usize) {
    let enc = tok.encode(prompt, false).expect("clip encode");
    let mut ids = vec![49406u32];
    ids.extend(enc.get_ids().iter().take(75));
    ids.push(49407);
    let eos_pos = ids.len() - 1;
    ids.resize(77, 49407);
    (ids, eos_pos)
}

/// T5 tokens: text + EOS(1), padded to `len` with 0. (T5 has no BOS.)
fn t5_tokens(tok: &Tokenizer, prompt: &str, len: usize) -> Vec<u32> {
    let enc = tok.encode(prompt, false).expect("t5 encode");
    let mut ids: Vec<u32> = enc.get_ids().iter().take(len - 1).copied().collect();
    ids.push(1); // </s>
    ids.resize(len, 0);
    ids
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let prompt = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "a cinematic photo of a red fox in snow".into());
    let out = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "/tmp/flux_out.png".into());
    let steps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);
    let size: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(512);

    let clip_tok =
        Tokenizer::from_file(format!("{MODELS}/tokenizers/clip_tokenizer.json")).expect("clip tok");
    let t5_tok =
        Tokenizer::from_file(format!("{MODELS}/tokenizers/t5_tokenizer.json")).expect("t5 tok");

    let gen = GenConfig {
        height: size,
        width: size,
        steps,
        seed: 0,
        t5_len: 256,
    };
    let (ct, eos) = clip_tokens(&clip_tok, &prompt);
    let tt = t5_tokens(&t5_tok, &prompt, gen.t5_len);

    println!("loading FLUX pipeline ({size}×{size}, {steps} steps)…");
    let ctx = Arc::new(GpuContext::new().expect("gpu"));
    let t_load = std::time::Instant::now();
    let pipe = FluxPipeline::load(&ctx, std::path::Path::new(MODELS), &gen).expect("load pipeline");
    println!(
        "loaded in {:.1}s. prompt = {prompt:?}",
        t_load.elapsed().as_secs_f32()
    );

    let t_gen = std::time::Instant::now();
    let res = pipe.generate(&ct, eos, &tt, &gen);
    let gen_s = t_gen.elapsed().as_secs_f32();
    for s in &res.trace {
        println!(
            "  step {} sigma={:.3} ‖latent‖={:.1} dit={:.0}ms",
            s.step, s.sigma, s.latent_l2, s.dit_ms
        );
    }
    println!("generated in {gen_s:.2}s");

    // RGB [3,H,W] in ~[-1,1] (channel-major) → u8 RGB [H,W,3]
    let (h, w) = (res.height, res.width);
    let mut img = image::RgbImage::new(w as u32, h as u32);
    let plane = h * w;
    for y in 0..h {
        for x in 0..w {
            let px = |c: usize| {
                let v = res.rgb[c * plane + y * w + x];
                (((v * 0.5 + 0.5).clamp(0.0, 1.0)) * 255.0).round() as u8
            };
            img.put_pixel(x as u32, y as u32, image::Rgb([px(0), px(1), px(2)]));
        }
    }
    img.save(&out).expect("save png");
    println!("wrote {out}");
}
