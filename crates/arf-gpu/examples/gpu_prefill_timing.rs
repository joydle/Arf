//! P20: measure Q4 prefill speedup — batched prefill (one ragged forward_batch,
//! weights streamed ONCE for the whole prompt via the batched-Q4 GEMV) vs the
//! sequential per-token prefill (`forward_logits`, weights streamed once PER token).
//! Time-to-first-token is dominated by prefill, so this is the user-visible win.
//!
//! Run: ARF_MODEL_PATH=models/llama-3.2-1b QUANT=q4 \
//!   cargo run --release -p arf-gpu --features gpu --example gpu_prefill_timing

fn main() {
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use arf_core::cache::{slots_for, write_runs};
    use arf_core::config::{ModelConfig, Quant};
    use arf_core::model::batch::{ForwardBatch, SeqAttn};
    use arf_gpu::gpu::GpuContext;

    let dir = PathBuf::from(std::env::var("ARF_MODEL_PATH").expect("ARF_MODEL_PATH"));
    let cfg = ModelConfig::llama_3_2_1b();
    let mf = dir.join("model.safetensors");
    let paths: [&Path; 1] = [mf.as_path()];
    let quant = match std::env::var("QUANT").as_deref() {
        Ok("int8") => Quant::Int8,
        Ok("bf16") | Ok("none") => Quant::None,
        _ => Quant::Q4,
    };

    let ctx = std::sync::Arc::new(GpuContext::new().expect("gpu"));
    eprint!("loading {quant:?} weights onto {}... ", ctx.info());
    std::io::stderr().flush().ok();
    let model = arf_gpu::weights::load_safetensors_gpu_quant(&cfg, &paths, 2048, &ctx, quant)
        .expect("load");
    eprintln!("done");

    let bs = 16usize; // KV block size used by build_gpu_quant
    let tbl: Vec<u32> = (0..(2048 / bs as u32) + 1).collect();
    // Deterministic prompt ids (content doesn't matter for prefill cost).
    let mk = |len: usize| -> Vec<u32> { (0..len as u32).map(|i| (i * 131 % 30000) + 1).collect() };

    println!("{quant:?} prefill: sequential (per-token) vs batched (one forward), median of 5\n");
    println!(
        "{:>8} {:>12} {:>12} {:>8} {:>8}",
        "prompt", "seq_ms", "batch_ms", "speedup", "match"
    );
    for &len in &[16usize, 64, 128, 256, 512, 1024] {
        let ids = mk(len);
        let batch = ForwardBatch {
            positions: (0..len as u32).collect(),
            seqs: vec![SeqAttn {
                q_start: 0,
                q_len: len,
                past_len: 0,
                slots: slots_for(&tbl, bs, len),
                write_runs: write_runs(&tbl, bs, 0, len),
                image_spans: Vec::new(),
                stream_id: None,
            }],
            image_embeds: None,
            mrope_positions: None,
        };

        // Warm up both paths.
        let _ = model.forward_logits(&ids);
        let _ = model.forward_batch(&ids, &batch);

        let mut seq_ms = Vec::new();
        let mut bat_ms = Vec::new();
        let (mut last_seq, mut last_bat) = (Vec::new(), Vec::new());
        for _ in 0..5 {
            let t = Instant::now();
            last_seq = model.forward_logits(&ids);
            seq_ms.push(t.elapsed().as_secs_f64() * 1e3);

            let t = Instant::now();
            let lg = model.forward_batch(&ids, &batch);
            last_bat = lg[0].clone();
            bat_ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let med = |mut v: Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let s = med(seq_ms);
        let b = med(bat_ms);
        // Sanity: both prefill the same prompt -> same last-token argmax.
        let am = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        };
        let ok = am(&last_seq) == am(&last_bat);
        println!(
            "{len:>8} {s:>12.2} {b:>12.2} {:>7.2}x {:>8}",
            s / b,
            if ok { "ok" } else { "DIFF" }
        );
        std::io::stdout().flush().ok();
    }
}
