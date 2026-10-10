//! `perplexity` subcommand: teacher-forced perplexity over a fixed text corpus.
//!
//! Computes PPL = exp(mean NLL per token) by teacher-forcing the corpus through
//! the GPU model's `forward_batch_all` — the same path used by `dump-logits`.
//! This is **Correctness Ladder Layer 5**: if PPL is within ~1–2% of a reference
//! implementation (HF transformers f32, llama.cpp) the engine is numerically
//! healthy end-to-end, not just on individual token outputs.
//!
//! # Algorithm
//!
//! 1. Tokenize the input text (with BOS prepended if absent).
//! 2. Run `forward_batch_all` to get per-position logits (same call as dump-logits).
//! 3. For each position i (1..N), compute
//!    `log p(token[i] | context) = logit[i-1][token[i]] - logsumexp(logit[i-1])`
//!    (the "next-token" NLL at each position — position 0 is the BOS context, not
//!    scored itself; we score positions 1..N against their preceding context).
//! 4. PPL = exp(−mean(log_p)).
//!
//! # Reference
//!
//! `scripts/hf_perplexity.py` computes the same quantity via HF transformers f32
//! in the existing HF env at `/tmp/hf_l4_env`. Run it on the same fixture file
//! to obtain a reference ratio.

use std::error::Error;
use std::path::PathBuf;

use clap::Parser;

use crate::{
    collect_safetensors, is_gguf, resolve_config, resolve_device, resolve_tokenizer, DeviceChoice,
};

// Args

#[derive(Parser)]
pub struct PerplexityArgs {
    /// Path to a safetensors file/dir, or a single-file GGUF.
    #[arg(long)]
    pub model: PathBuf,

    /// Model architecture (required for GGUF without a sibling config.json).
    #[arg(long)]
    pub arch: Option<String>,

    /// Path to tokenizer.json. Required unless --tokens is used.
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,

    /// Text file to compute perplexity over (UTF-8).
    /// Mutually exclusive with --text.
    #[arg(long)]
    pub text_file: Option<PathBuf>,

    /// Inline text to compute perplexity over.
    /// Mutually exclusive with --text-file.
    #[arg(long)]
    pub text: Option<String>,

    /// Override the model's `vocab_size`.
    #[arg(long)]
    pub vocab: Option<usize>,

    /// Weight quantization: `none`, `int8`, `q4`, `q4k`, `q4ks`, `q3k`.
    #[arg(long, default_value = "none")]
    pub quant: String,

    /// KV-cache quantization (GPU only): `none`, `tq2`, `tq3`, `tq4`.
    #[arg(long, default_value = "none")]
    pub kv_quant: String,

    /// Size of the RoPE table / max context.
    #[arg(long, default_value_t = 8192)]
    pub max_context: usize,
}

// Entry point

pub fn perplexity(args: PerplexityArgs) -> Result<(), Box<dyn Error>> {
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
                "unknown --kv-quant {other:?} (expected none, tq2, tq3, tq4, or the qjl variants)"
            )
            .into())
        }
    };

    // device
    let device = resolve_device("gpu")?;
    if kv_quant.is_quantized() && matches!(device, DeviceChoice::Cpu) {
        return Err("--kv-quant is GPU-only".into());
    }
    let DeviceChoice::Gpu(ctx) = &device else {
        return Err("perplexity requires GPU (the CPU path lacks forward_batch_all)".into());
    };

    // tokenizer + text
    let tokenizer = resolve_tokenizer(args.tokenizer.as_ref(), &args.model)?;
    let tok = tokenizer
        .as_ref()
        .ok_or("--tokenizer is required for perplexity (need to tokenize the input text)")?;

    let raw_text = if let Some(path) = &args.text_file {
        std::fs::read_to_string(path).map_err(|e| format!("reading {}: {e}", path.display()))?
    } else if let Some(text) = &args.text {
        text.clone()
    } else {
        return Err("provide --text or --text-file".into());
    };

    // Tokenize: encode with BOS prepended if absent.
    let mut token_ids: Vec<u32> = tok.encode(&raw_text, true)?;
    if let Some(bos) = crate::bos_token(tok) {
        if token_ids.first() != Some(&bos) {
            token_ids.insert(0, bos);
        }
    }
    if token_ids.len() < 2 {
        return Err("corpus too short — need at least 2 tokens to compute PPL".into());
    }

    let n = token_ids.len();
    eprintln!("corpus: {} tokens", n);

    // load model
    let model_is_gguf = is_gguf(&args.model);
    let paths = if model_is_gguf {
        vec![args.model.clone()]
    } else {
        collect_safetensors(&args.model)?
    };

    eprintln!("loading model for perplexity ({})...", device.describe());
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

    // teacher-forced forward (same pattern as dump-logits)
    let batch = {
        use arf_core::cache::{slots_for, write_runs};
        use arf_core::model::batch::{ForwardBatch, SeqAttn};
        let bs = model.kv.block_size;
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

    eprintln!("running teacher-forced forward over {} tokens...", n);
    let all_logits: Vec<Vec<f32>> = model.forward_batch_all(&token_ids, &batch);
    assert_eq!(
        all_logits.len(),
        n,
        "forward_batch_all returned {} rows for {} tokens",
        all_logits.len(),
        n
    );

    // compute PPL
    //
    // Positions 0..N-1: all_logits[i] is the next-token distribution AFTER
    // consuming token i. We score the actual next token: token_ids[i+1].
    // So we accumulate NLL over positions 1..N (scoring tokens 1..N-1 is the
    // standard: skip BOS as a "target" but use it as context).
    let mut total_nll: f64 = 0.0;
    let scored = n - 1; // number of positions scored (tokens 1..N-1)

    for i in 0..scored {
        let logits = &all_logits[i];
        let next_token = token_ids[i + 1] as usize;

        // logsumexp for numerical stability
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f64 = logits.iter().map(|&l| ((l - max_l) as f64).exp()).sum();
        let logsumexp = max_l as f64 + sum_exp.ln();

        let log_p = if next_token < logits.len() {
            logits[next_token] as f64 - logsumexp
        } else {
            f64::NEG_INFINITY
        };
        total_nll -= log_p; // NLL = -log_p
    }

    let mean_nll = total_nll / scored as f64;
    let ppl = mean_nll.exp();

    let quant_label = args.quant.to_ascii_lowercase();
    let quant_label = if quant_label == "none" {
        "bf16"
    } else {
        &quant_label
    };

    println!(
        "perplexity [{quant_label}] over {scored} tokens: {ppl:.4} (mean NLL: {mean_nll:.6} nats)"
    );

    Ok(())
}

// Helpers
