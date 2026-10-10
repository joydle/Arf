//! DFlash 2 — WHAT DOES 4-BIT COST THE DRAFT'S ACCEPTANCE? An experiment, not a gate.
//!
//! For several real-text contexts: the target (27B, Q4_K) prefills the context and greedily
//! continues 8 tokens — the truth a draft is scored against. Then the SAME block is proposed
//! twice: by the shipped GPU draft (Q4_K_S weights), and by the CPU reference running the
//! PUBLISHED bf16 weights, whose final hidden is handed to the same lm_head + selector on the
//! GPU. Both attend over the same context ring (built by the shipped 4-bit k/v projections), so
//! this isolates the BLOCK's weights; it is a lower bound on what full precision would recover.
//! Score = how many leading proposals the target would accept.
//!
//! Run: cargo run --release -p arf-gpu --example dflash2_precision -- [draft dir] [gguf]

#[cfg(target_os = "macos")]
const TEXTS: [&str; 3] = [
    "A bicycle stays upright mostly because of how it is steered, not because of any single \
     stabilising force. When the bicycle begins to lean to one side, the front wheel turns \
     slightly into the lean, and the contact patches of the tyres move back underneath the \
     centre of mass. Riders do this without thinking about it, and a moving bicycle with nobody \
     on it will do a version of it by itself, because the geometry of the front fork makes the \
     wheel steer towards the side the frame is falling to. Gyroscopic effects from the spinning \
     wheels contribute a little, but experiments with counter-rotating wheels have shown that a \
     bicycle can still balance when those effects are cancelled out. Speed matters because the \
     faster the bicycle moves, the smaller the steering correction that is needed to bring the \
     wheels back under the rider, which is why balancing at walking pace is so much harder than \
     balancing at a normal riding speed. In short, a bicycle is kept upright by constant small \
     steering corrections, made partly by the rider and partly by the machine itself.",
    "Here is a function that parses a configuration file. It reads the file line by line, skips \
     blank lines and comments, splits each remaining line at the first equals sign, and stores \
     the key and value in a map.\n\n```rust\nuse std::collections::HashMap;\n\npub fn \
     parse_config(text: &str) -> Result<HashMap<String, String>, String> {\n    let mut map = \
     HashMap::new();\n    for (line_no, line) in text.lines().enumerate() {\n        let line = \
     line.trim();\n        if line.is_empty() || line.starts_with('#') {\n            \
     continue;\n        }\n        let (key, value) = line\n            .split_once('=')\n      \
           .ok_or_else(|| format!(\"line {}: expected key=value\", line_no + 1))?;\n        \
     map.insert(key.trim().to_string(), value.trim().to_string());\n    }\n    Ok(map)\n}\n```\n\n\
     The function returns an error that names the line number when a line has no equals sign, \
     and otherwise returns the map of every key to its value. A caller can then look up the \
     port with map.get(\"port\") and parse it into a number, handling the case where the key is \
     missing or the value is not a valid integer.",
    "To find how long the two trains take to meet, first work out how quickly the gap between \
     them closes. The first train travels at 60 kilometres per hour and the second at 90 \
     kilometres per hour, and they are moving towards each other, so the distance between them \
     shrinks at 60 plus 90, which is 150 kilometres per hour. They start 450 kilometres apart. \
     Dividing the distance by the closing speed gives 450 divided by 150, which is 3 hours. In \
     those 3 hours the first train covers 3 times 60, which is 180 kilometres, and the second \
     covers 3 times 90, which is 270 kilometres. As a check, 180 plus 270 is 450 kilometres, \
     which matches the starting distance, so the trains meet after 3 hours, 180 kilometres from \
     where the first train started.",
];

#[cfg(target_os = "macos")]
fn main() {
    use arf_core::backend::BatchedBackend;
    use arf_core::config::{EngineConfig, KvQuant, ModelConfig, Quant};
    use arf_core::engine::build_forward;
    use arf_core::sampling::{SamplingParams, SeqSampling};
    use arf_core::scheduler::{Request, Scheduler};
    use arf_gpu::gpu::metal::dflash2::Dflash2Config;
    use arf_gpu::gpu::metal::dflash2_ref::Dflash2BlockReference;
    use arf_gpu::gpu::GpuContext;
    use arf_gpu::WgpuBatched;
    use std::sync::Arc;

    let arg = |i: usize, d: &str| std::env::args().nth(i).unwrap_or_else(|| d.into());
    let dir = std::path::PathBuf::from(arg(1, "models/qwen3.8-27b-dflash2"));
    let gguf = arg(2, "models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf");
    arf_gpu::apply_fast_path_defaults("dflash2_precision");
    std::env::remove_var("ARF_KV_F16");
    std::env::set_var("ARF_NO_KV_F16", "1");

    let tok = arf_core::tokenizer::Tokenizer::from_gguf_path(&gguf).expect("gguf tokenizer");
    let cfg = ModelConfig::qwen35_27b();
    let (block_size, num_blocks) = (16usize, 64usize);
    let ctx = Arc::new(GpuContext::new().expect("gpu context"));
    eprintln!("loading {gguf} ...");
    let model = arf_gpu::weights::load_gguf_gpu_kv_quant_pooled(
        &cfg,
        std::path::Path::new(&gguf),
        block_size * num_blocks,
        &ctx,
        Quant::Q4KS,
        KvQuant::None,
        Some(num_blocks),
    )
    .expect("load gguf");
    let backend = WgpuBatched(model);
    backend.0.dflash_attach(&dir).expect("attach draft");
    let dc = Dflash2Config::from_json(&std::fs::read_to_string(dir.join("config.json")).unwrap())
        .unwrap();
    eprintln!("loading the PUBLISHED bf16 block weights (~6 GB of f32) ...");
    let published = Dflash2BlockReference::load(&dir, &dc, true).expect("published");

    let next_id = std::cell::Cell::new(1u64);
    let (mut sum_q4, mut sum_pub, mut n) = (0usize, 0usize, 0usize);
    let (mut first_q4, mut first_pub) = (0usize, 0usize);
    // DFLASH_PRECISION_TEXT=<file>: score ONE text from a file instead of the three built-ins, at
    // 14 positions — for the model's OWN thinking-mode output (2026-09-22), where serving
    // acceptance is 2.09 against 3.2 on the plain built-ins; the 15-block verdict never saw it.
    let file_text = std::env::var("DFLASH_PRECISION_TEXT")
        .ok()
        .map(|f| std::fs::read_to_string(f).expect("read DFLASH_PRECISION_TEXT"));
    let texts: Vec<&str> = match file_text.as_deref() {
        Some(t) => vec![t],
        None => TEXTS.to_vec(),
    };
    let fracs: Vec<f32> = if file_text.is_some() {
        (0..14).map(|i| 0.12 + 0.06 * i as f32).collect()
    } else {
        vec![0.35, 0.5, 0.65, 0.8, 0.92]
    };
    for (ti, text) in texts.iter().enumerate() {
        let ids = tok.encode(text, false).expect("encode");
        for &frac in &fracs {
            let p = ((ids.len() as f32 * frac) as usize).max(24);
            let id = next_id.get();
            next_id.set(id + 1);
            let mut sched = Scheduler::new(EngineConfig {
                block_size,
                num_blocks,
                max_batch_size: 1,
                max_prefill_tokens: 512,
                enable_prefix_cache: false,
                ..Default::default()
            });
            sched.add(Request::new(
                id,
                ids[..p].to_vec(),
                SamplingParams::greedy(8),
            ));
            let g = SamplingParams::greedy(8);
            let mut truth: Vec<u32> = vec![];
            while sched.has_unfinished() {
                let Some(plan) = sched.schedule().unwrap() else {
                    break;
                };
                let (inp, batch) = build_forward(&plan, sched.block_size());
                let samp: Vec<SeqSampling> = plan
                    .seqs
                    .iter()
                    .map(|sp| SeqSampling {
                        params: &g,
                        position: sp.past_len + sp.q_len,
                        generated: &[],
                    })
                    .collect();
                let toks = backend.step(&inp, &batch, &samp).unwrap();
                truth.extend(
                    sched
                        .commit_tokens(&toks)
                        .unwrap()
                        .into_iter()
                        .map(|o| o.token),
                );
            }
            // truth[0] is the anchor (position p); truth[1..8] is what the draft must predict.
            // The b=1 decode steps run the m=1 record, which has no taps: the ring is still
            // exactly the p prompt positions.
            let q4 = backend.0.dflash_draft(truth[0], p, id);
            if q4.is_empty() {
                eprintln!("text {ti} p={p}: the draft declined (ring not at p) — skipped");
                continue;
            }
            let isl = backend.0.island.as_ref().unwrap().lock().unwrap();
            let (mut rk, mut rv) = (vec![], vec![]);
            for l in 0..dc.layers {
                let (k, v) = isl.dflash_read_ring(l).expect("ring");
                rk.push(k);
                rv.push(v);
            }
            drop(isl);
            let mut blk = vec![dc.mask_token_id; dc.block_size];
            blk[0] = truth[0];
            let embeds = backend.0.dflash_embed_rows(&blk).expect("embed rows");
            let hidden = published.forward(&embeds, &rk, &rv, p);
            let pb = backend
                .0
                .dflash_propose_from_hidden(&hidden, truth[0])
                .expect("propose");
            let run = |d: &[u32]| {
                d.iter()
                    .zip(&truth[1..])
                    .take_while(|(a, b)| a == b)
                    .count()
            };
            let (a, b) = (run(&q4), run(&pb));
            println!(
                "text {ti} ctx {p:3}: accepted of 7 — Q4_K_S draft {a}, published-bf16 draft {b}   | truth {:?} | q4 {:?} | bf16 {:?}",
                &truth[1..],
                q4,
                pb
            );
            sum_q4 += a;
            sum_pub += b;
            first_q4 += (a > 0) as usize;
            first_pub += (b > 0) as usize;
            n += 1;
        }
    }
    println!(
        "\n{n} blocks: mean accepted Q4_K_S {:.2} vs published bf16 {:.2}; first token right {}/{n} vs {}/{n}",
        sum_q4 as f64 / n as f64,
        sum_pub as f64 / n as f64,
        first_q4,
        first_pub
    );
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("dflash2_precision drives the native-Metal island; macOS only.");
}
