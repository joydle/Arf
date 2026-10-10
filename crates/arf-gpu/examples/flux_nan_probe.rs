//! Which FLUX stage first produces NaN? Runs the two text encoders on a fixed prompt and
//! reports finite-ness of each conditioning tensor before the DiT ever sees them.
use arf_gpu::gpu::flux::{ClipConfig, GpuClipText, GpuT5, T5Config};
use arf_gpu::gpu::GpuContext;
use std::sync::Arc;
use tokenizers::Tokenizer;

const MODELS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/flux-schnell");

fn report(name: &str, v: &[f32]) {
    let nan = v.iter().filter(|x| x.is_nan()).count();
    let inf = v.iter().filter(|x| x.is_infinite()).count();
    let fin: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
    let l2 = fin.iter().map(|x| x * x).sum::<f32>().sqrt();
    let amax = fin.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    println!(
        "  {name:<12} len={:<7} nan={nan:<6} inf={inf:<4} l2={l2:<11.3} absmax={amax:.4}",
        v.len()
    );
}

fn main() {
    let clip_tok =
        Tokenizer::from_file(format!("{MODELS}/tokenizers/clip_tokenizer.json")).expect("clip tok");
    let t5_tok =
        Tokenizer::from_file(format!("{MODELS}/tokenizers/t5_tokenizer.json")).expect("t5 tok");
    let prompt = "a red fox in snow";
    let t5_len = 256usize;

    let enc = clip_tok.encode(prompt, false).expect("clip encode");
    let mut ct = vec![49406u32];
    ct.extend(enc.get_ids().iter().take(75));
    ct.push(49407);
    let eos = ct.len() - 1;
    ct.resize(77, 49407);

    let enc5 = t5_tok.encode(prompt, false).expect("t5 encode");
    let mut tt: Vec<u32> = enc5.get_ids().iter().take(t5_len - 1).copied().collect();
    tt.push(1);
    tt.resize(t5_len, 0);

    let ctx = Arc::new(GpuContext::new().expect("gpu"));
    println!("stage finite-ness (prompt = {prompt:?}):");

    let clip = GpuClipText::load(
        &ctx,
        std::path::Path::new(&format!("{MODELS}/clip_l.safetensors")),
        ClipConfig::default(),
    )
    .expect("clip load");
    report("clip_pooled", &clip.encode(&ct, eos));

    let t5 = GpuT5::load(
        &ctx,
        std::path::Path::new(&format!("{MODELS}/t5xxl-Q8_0.gguf")),
        T5Config::default(),
    )
    .expect("t5 load");
    report("t5_seq", &t5.encode(&tt));
}
