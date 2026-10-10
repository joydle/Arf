//! Real-model batched Gemma-4 collapse probe. Loads the real GGUF and compares
//! single-stream `generate_stop` vs batched `forward_batch` (public APIs) on the
//! SAME tokens — to localize the `of of of` collapse on the real weights (the
//! learned sink channel that random unit-test weights can't reproduce).
//!
//! Usage: cargo run --release --example batched_gemma4_diff -- <gguf-path>
//! Env: ARF_N=<seqs>, ARF_PLEN=<prompt-len>, ARF_RAW=1 — see the prompt
//! builder. (The per-op `ARF_FB_*` dump probes were removed post-fix; this
//! example reproduces the single-stream-vs-batched comparison that found the bug.)

use arf_core::cache::{slots_for, write_runs};
use arf_core::config::{KvQuant, ModelConfig, Quant};
use arf_core::model::{ForwardBatch, SeqAttn};
use arf_core::sampling::argmax;
use std::sync::Arc;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: batched_gemma4_diff <gguf>");
    let ctx = Arc::new(arf_gpu::gpu::GpuContext::new().expect("gpu"));
    let cfg = ModelConfig::gemma4_12b();
    let max_ctx = 256;
    // Chat-templated structure (gemma-4): <bos> <|turn>(105) user \n(107) {body}
    // <turn|>(106) \n(107) <|turn>(105) model \n(107) — mirrors the CLI --chat path
    // that COLLAPSES at batch=4. ARF_RAW=1 uses the short raw prompt instead.
    let full: Vec<u32> = vec![
        2, 105, 2364, 107, 3689, 563, 506, 5279, 529, 7001, 236881, 18925, 528, 1217, 3309, 236761,
        106, 107, 105, 4368, 107,
    ];
    // ARF_PLEN truncates the chat prompt to N tokens — bisect length vs content.
    let prompt: Vec<u32> = match std::env::var("ARF_PLEN")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        Some(n) => full[..n.min(full.len())].to_vec(),
        None => full,
    };

    let m = arf_gpu::weights::load_gguf_gpu_kv_quant(
        &cfg,
        std::path::Path::new(&path),
        max_ctx,
        &ctx,
        Quant::Q4K,
        KvQuant::None,
    )
    .expect("load");

    // Single-stream: the first generated token (oracle).
    let ss_tok = *m.generate_stop(&prompt, 1, &[]).first().unwrap();

    // Batched: forward_batch over the whole prompt as ONE sequence; argmax of the
    // last-token logits = the first generated token (same as generate_batch prefill).
    let bs = 16usize; // gemma-4 KV block size
    let tbl: Vec<u32> = (0..(max_ctx / bs) as u32).collect();
    let batch = ForwardBatch {
        positions: (0..prompt.len() as u32).collect(),
        seqs: vec![SeqAttn {
            q_start: 0,
            q_len: prompt.len(),
            past_len: 0,
            slots: slots_for(&tbl, bs, prompt.len()),
            write_runs: write_runs(&tbl, bs, 0, prompt.len()),
            image_spans: Vec::new(),
            stream_id: None,
        }],
        image_embeds: None,
        mrope_positions: None,
    };
    let fb_logits = m.forward_batch(&prompt, &batch);
    let fb_tok = argmax(&fb_logits[0]);

    // forward_batch prefilled TOKEN-BY-TOKEN (m=1 each) — if THIS matches SS but the
    // m=n version doesn't, the bug is the m>1 prefill accumulation (fix = per-token).
    let tbl2: Vec<u32> = ((max_ctx / bs) as u32..(2 * max_ctx / bs) as u32).collect();
    let mut t1_logits = Vec::new();
    for (p, &tok) in prompt.iter().enumerate() {
        let b = ForwardBatch {
            positions: vec![p as u32],
            seqs: vec![SeqAttn {
                q_start: 0,
                q_len: 1,
                past_len: p,
                slots: slots_for(&tbl2, bs, p + 1),
                write_runs: write_runs(&tbl2, bs, p, 1),
                stream_id: None,
                image_spans: Vec::new(),
            }],
            image_embeds: None,
            mrope_positions: None,
        };
        t1_logits = m.forward_batch(&[tok], &b).pop().unwrap();
    }
    let t1_tok = argmax(&t1_logits);

    eprintln!("=== real gemma-4-12B, prompt len {} ===", prompt.len());
    eprintln!("[1-seq] SS={ss_tok}  FB(m=n)={fb_tok}  FB(token-by-token m=1)={t1_tok}");
    eprintln!(
        "  m=n MATCH={}   token-by-token MATCH={}",
        ss_tok == fb_tok,
        ss_tok == t1_tok
    );

    // ---- N-SEQ multi-step decode, mirroring generate_batch's block_tables EXACTLY
    // (blocks_per_seq = max_ctx.div_ceil(bs), seq s owns [s*bps .. (s+1)*bps)). ----
    let past = prompt.len();
    let nseq = std::env::var("ARF_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4usize);
    let bps = max_ctx.div_ceil(bs); // EXACTLY generate_batch's blocks_per_seq
    let tbls: Vec<Vec<u32>> = (0..nseq)
        .map(|s| ((s * bps) as u32..((s + 1) * bps) as u32).collect())
        .collect();
    // Prefill each seq (single-seq forward_batch).
    for tbl in &tbls {
        let b = ForwardBatch {
            positions: (0..prompt.len() as u32).collect(),
            seqs: vec![SeqAttn {
                q_start: 0,
                q_len: prompt.len(),
                past_len: 0,
                slots: slots_for(tbl, bs, prompt.len()),
                write_runs: write_runs(tbl, bs, 0, prompt.len()),
                stream_id: None,
                image_spans: Vec::new(),
            }],
            image_embeds: None,
            mrope_positions: None,
        };
        m.forward_batch(&prompt, &b);
    }
    let mut next = vec![ss_tok; nseq];
    let mut p = vec![past; nseq];
    let mut toks0 = Vec::new();
    for _ in 0..10 {
        let seqs: Vec<SeqAttn> = (0..nseq)
            .map(|s| SeqAttn {
                q_start: s,
                q_len: 1,
                past_len: p[s],
                slots: slots_for(&tbls[s], bs, p[s] + 1),
                write_runs: write_runs(&tbls[s], bs, p[s], 1),
                image_spans: Vec::new(),
                stream_id: None,
            })
            .collect();
        let dec = ForwardBatch {
            positions: p.iter().map(|&x| x as u32).collect(),
            seqs,
            image_embeds: None,
            mrope_positions: None,
        };
        let lg = m.forward_batch(&next, &dec);
        for s in 0..nseq {
            next[s] = argmax(&lg[s]);
            p[s] += 1;
        }
        toks0.push(next[0]);
    }
    eprintln!("[{nseq}-seq manual] seq0 tokens = {toks0:?} (collapse = same tok repeated)");

    // ---- THE ACTUAL generate_batch (the CLI path). If THIS collapses but the manual
    // replication above doesn't, the bug is inside generate_batch's orchestration. ----
    let prompts: Vec<Vec<u32>> = vec![prompt.clone(); nseq];
    let out = m.generate_batch(&prompts, 10);
    eprintln!("[generate_batch] seq0 tokens = {:?}", out[0]);
}
