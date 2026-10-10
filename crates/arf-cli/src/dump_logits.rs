//! `dump-logits` subcommand: teacher-forced per-position logit dump.
//!
//! Feeds a fixed token sequence through the model (teacher forcing) and writes
//! one JSON object per position to a JSONL file.  The output is the schema
//! consumed by `scripts/compare_logits.py` — any engine (HF, llama.cpp, …)
//! that emits the same schema can be compared against this dump.
//!
//! **Output schema (JSONL)**
//!
//! Line 0 — meta header:
//! ```json
//! {"meta": {"model": "…", "quant": "…", "device": "…",
//!           "vocab": N, "positions": N, "engine": "arf"}}
//! ```
//!
//! Lines 1…N — one per position:
//! ```json
//! {"pos": 0, "token_id": <input tok>, "argmax": <id>, "top": [[id, logit], …],
//!  "logsumexp": <f32>}
//! ```
//! `logsumexp` = log(Σ exp(logit_i)) over the FULL vocab row.  It lets the
//! compare script reconstruct exact probabilities for any dumped (id, logit)
//! pair without needing the full row: p_i = exp(logit_i − logsumexp).
//!
//! With `--full`, an additional `"logits": [… vocab floats …]` key is appended
//! to each position line, enabling exact KL over all vocab positions.
//!
//! **NOTE (v1):** CPU all-position logits require running a separate forward
//! pass per position (the CPU `Llama` exposes `logits_last` / `forward`, not a
//! batched all-positions call).  For v1 the dump is **GPU-only**.  Passing
//! `--device cpu` returns an explanatory error.  A future CPU path is trivial
//! to add by looping `engine.forward_single` and collecting each row; it is not
//! wired here to keep the scope minimal.

use std::error::Error;
use std::io::Write as IoWrite;
use std::path::PathBuf;

use clap::Parser;

use crate::{
    collect_safetensors, is_gguf, resolve_config, resolve_device, resolve_tokenizer, DeviceChoice,
};

// Args

#[derive(Parser)]
pub struct DumpLogitsArgs {
    /// Path to a safetensors file/dir, or a single-file GGUF.
    #[arg(long)]
    pub model: PathBuf,

    /// Model architecture (required for GGUF without a sibling config.json).
    #[arg(long)]
    pub arch: Option<String>,

    /// Path to tokenizer.json. Required unless --tokens is used.
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,

    /// Prompt text (requires --tokenizer).
    #[arg(long)]
    pub prompt: Option<String>,

    /// Raw prompt token ids, comma-separated (bypasses the tokenizer).
    #[arg(long, value_delimiter = ',')]
    pub tokens: Option<Vec<u32>>,

    /// Output file path (default: logits.jsonl).
    #[arg(long, default_value = "logits.jsonl")]
    pub out: PathBuf,

    /// Number of top-(id,logit) pairs to dump per position (also see --full).
    /// Internally clamped to max(top, 64) to preserve enough mass for KL.
    #[arg(long, default_value_t = 5)]
    pub top: usize,

    /// Dump the full vocab logit row per position (large files; enables exact KL).
    #[arg(long)]
    pub full: bool,

    /// Backend: `cpu` (not supported in v1) or `gpu`.
    #[arg(long, default_value = "gpu")]
    pub device: String,

    /// Size of the RoPE table / max context.
    #[arg(long, default_value_t = 8192)]
    pub max_context: usize,

    /// Override the model's `vocab_size`.
    #[arg(long)]
    pub vocab: Option<usize>,

    /// Weight quantization: `none`, `int8`, `q4`, `q4k`, `q4ks`, `q3k`.
    #[arg(long, default_value = "none")]
    pub quant: String,

    /// KV-cache quantization (GPU only): `none`, `tq2`, `tq3`, `tq4`, or the
    /// qjl variants `tq2q`/`tq3q`/`tq4q`.
    #[arg(long, default_value = "none")]
    pub kv_quant: String,
}

// Entry point

pub fn dump_logits(args: DumpLogitsArgs) -> Result<(), Box<dyn Error>> {
    use arf_core::config::{KvQuant, Quant};

    // config
    let mut cfg = resolve_config(&args.model, args.arch.as_deref())?;
    if let Some(v) = args.vocab {
        cfg.vocab_size = v;
    }

    let quant: Quant = args.quant.to_ascii_lowercase().parse()?;
    let kv_quant = match args.kv_quant.to_ascii_lowercase().as_str() {
        "none" => KvQuant::None,
        "tq2" => KvQuant::Tq {
            bits: 2,
            qjl: false,
        },
        "tq3" => KvQuant::Tq {
            bits: 3,
            qjl: false,
        },
        "tq4" => KvQuant::Tq {
            bits: 4,
            qjl: false,
        },
        "tq2q" => KvQuant::Tq { bits: 2, qjl: true },
        "tq3q" => KvQuant::Tq { bits: 3, qjl: true },
        "tq4q" => KvQuant::Tq { bits: 4, qjl: true },
        other => {
            return Err(format!(
                "unknown --kv-quant {other:?} (expected none, tq2, tq3, tq4, or qjl variants)"
            )
            .into())
        }
    };

    // device
    let device = resolve_device(&args.device)?;
    if matches!(device, DeviceChoice::Cpu) {
        return Err(
            "dump-logits v1 is GPU-only (the CPU path lacks a batched all-positions forward; \
             pass --device gpu). A per-position loop on the CPU Llama can be added in v2."
                .into(),
        );
    }
    if kv_quant.is_quantized() && matches!(device, DeviceChoice::Cpu) {
        return Err("--kv-quant is GPU-only".into());
    }

    let DeviceChoice::Gpu(ctx) = &device else {
        unreachable!("already checked above")
    };

    // tokenizer + prompt
    let tokenizer = resolve_tokenizer(args.tokenizer.as_ref(), &args.model)?;
    let prompt_ids = resolve_prompt_raw(
        args.tokens.as_ref(),
        args.prompt.as_deref(),
        tokenizer.as_ref(),
    )?;

    if prompt_ids.is_empty() {
        return Err("prompt is empty — provide at least one token".into());
    }

    // load model
    let model_is_gguf = is_gguf(&args.model);
    let paths = if model_is_gguf {
        vec![args.model.clone()]
    } else {
        collect_safetensors(&args.model)?
    };

    eprintln!("loading model for dump-logits ({})...", device.describe());
    let model = if model_is_gguf {
        arf_gpu::weights::load_gguf_gpu_kv_quant(
            &cfg,
            &args.model,
            args.max_context,
            ctx,
            quant,
            kv_quant,
        )?
    } else {
        arf_gpu::weights::load_safetensors_gpu_kv_quant(
            &cfg,
            &paths,
            args.max_context,
            ctx,
            quant,
            kv_quant,
        )?
    };

    // build teacher-forced ForwardBatch
    //
    // We want logits at EVERY position, so we issue a single prefill over the
    // full prompt using forward_batch_all.  The ForwardBatch mirrors what
    // generate_stop_inner does for the prefill window:
    //   - q_start 0, q_len = N, past_len 0
    //   - slots: physical slot ids for positions 0..N
    //   - write_runs: write all N K/V rows starting at position 0
    //   - positions: [0, 1, …, N-1]
    let n = prompt_ids.len();
    let batch = {
        use arf_core::cache::{slots_for, write_runs};
        use arf_core::model::batch::{ForwardBatch, SeqAttn};
        let bs = model.kv.block_size;
        // A fresh block table: we own all blocks from 0..num_blocks.
        // slots_for and write_runs only need the table to span positions 0..n,
        // so we only need ceil(n / block_size) blocks.
        let num_blocks_needed = n.div_ceil(bs);
        let tbl: Vec<u32> = (0..num_blocks_needed as u32).collect();
        ForwardBatch {
            positions: (0..n as u32).collect(),
            seqs: vec![SeqAttn {
                stream_id: None,
                q_start: 0,
                q_len: n,
                past_len: 0,
                slots: slots_for(&tbl, bs, n),
                write_runs: write_runs(&tbl, bs, 0, n),
                image_spans: Vec::new(),
            }],
            image_embeds: None,
            mrope_positions: None,
        }
    };

    // forward pass
    eprintln!(
        "running teacher-forced forward over {} tokens...",
        prompt_ids.len()
    );
    let all_logits: Vec<Vec<f32>> = model.forward_batch_all(&prompt_ids, &batch);
    assert_eq!(
        all_logits.len(),
        n,
        "forward_batch_all returned {} rows for {} tokens",
        all_logits.len(),
        n
    );

    // write JSONL
    let top_k = args.top.max(64); // at least 64 for KL approximation quality
    let vocab_size = cfg.vocab_size;
    let device_desc = device.describe();
    let quant_str = args.quant.to_ascii_lowercase();
    let model_str = args.model.display().to_string();
    let out_path = &args.out;

    eprintln!("writing JSONL to {}...", out_path.display());
    let file = std::fs::File::create(out_path)?;
    let mut writer = std::io::BufWriter::new(file);

    // Header line
    writeln!(
        writer,
        "{{\"meta\":{{\"model\":{},\"quant\":{},\"device\":{},\"vocab\":{},\"positions\":{},\"engine\":\"arf\"}}}}",
        json_str(&model_str),
        json_str(&quant_str),
        json_str(&device_desc),
        vocab_size,
        n,
    )?;

    for (pos, logits) in all_logits.iter().enumerate() {
        let token_id = prompt_ids[pos];

        // logsumexp over the full row (needed for probability reconstruction).
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f64 = logits.iter().map(|&l| ((l - max_l) as f64).exp()).sum();
        let logsumexp = max_l as f64 + sum_exp.ln();

        // argmax
        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);

        // top-k (id, logit) pairs sorted descending by logit
        let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
        // Partial sort: partition so the top_k highest-logit pairs land in [..top_k]
        // (unordered among themselves — we sort just that prefix below).
        indexed.select_nth_unstable_by(top_k.min(logits.len()) - 1, |(_, a), (_, b)| {
            b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut top: Vec<(usize, f32)> = indexed[..top_k.min(logits.len())].to_vec();
        top.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Build the JSON line manually (stdlib-only, no serde dependency).
        let mut top_json = String::from("[");
        for (i, (id, logit)) in top.iter().enumerate() {
            if i > 0 {
                top_json.push(',');
            }
            top_json.push_str(&format!("[{},{}]", id, logit_f32_to_str(*logit)));
        }
        top_json.push(']');

        if args.full {
            // Full logit array (large — one float per vocab token).
            let logits_json: String = {
                let mut s = String::from("[");
                for (i, &l) in logits.iter().enumerate() {
                    if i > 0 {
                        s.push(',');
                    }
                    s.push_str(&logit_f32_to_str(l));
                }
                s.push(']');
                s
            };
            writeln!(
                writer,
                "{{\"pos\":{},\"token_id\":{},\"argmax\":{},\"top\":{},\"logsumexp\":{},\"logits\":{}}}",
                pos, token_id, argmax, top_json,
                logit_f32_to_str(logsumexp as f32),
                logits_json,
            )?;
        } else {
            writeln!(
                writer,
                "{{\"pos\":{},\"token_id\":{},\"argmax\":{},\"top\":{},\"logsumexp\":{}}}",
                pos,
                token_id,
                argmax,
                top_json,
                logit_f32_to_str(logsumexp as f32),
            )?;
        }
    }

    writer.flush()?;
    eprintln!("done: {} positions written to {}", n, out_path.display());
    Ok(())
}

// Helpers

/// Resolve prompt tokens from raw fields (usable by both GenerateArgs and
/// DumpLogitsArgs without coupling to the struct type).
pub(crate) fn resolve_prompt_raw(
    tokens: Option<&Vec<u32>>,
    prompt: Option<&str>,
    tokenizer: Option<&arf_core::Tokenizer>,
) -> Result<Vec<u32>, Box<dyn Error>> {
    if let Some(t) = tokens {
        return Ok(t.clone());
    }
    let p = prompt.ok_or("provide --prompt (with --tokenizer) or --tokens")?;
    let tok = tokenizer.ok_or("--prompt requires --tokenizer")?;
    let mut ids = tok.encode(p, true)?;
    // Prepend BOS if the tokenizer's post-processor didn't insert one already.
    if let Some(bos) = crate::bos_token(tok) {
        if ids.first() != Some(&bos) {
            ids.insert(0, bos);
        }
    }
    Ok(ids)
}

// BOS token id from the tokenizer (by name, no per-model hardcoding).

/// Format a float for JSON output — finite as a decimal, NaN/Inf as null.
fn logit_f32_to_str(v: f32) -> String {
    if v.is_finite() {
        format!("{:.6}", v)
    } else {
        "null".to_string()
    }
}

/// Minimal JSON string escaping (path strings; no control chars expected).
fn json_str(s: &str) -> String {
    let escaped = s.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{}\"", escaped)
}
