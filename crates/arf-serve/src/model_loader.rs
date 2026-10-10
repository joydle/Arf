//! Model + config resolution at startup. These helpers mirror the patterns in
//! `arf-cli`'s `main.rs` (`config_for_arch`, `is_gguf`, `collect_safetensors`,
//! quant parsing) so the server loads the exact same way the CLI does — the only
//! difference is the model is loaded ONCE here and kept resident, instead of per
//! invocation.

use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arf_core::config::{KvQuant, ModelConfig, Quant};
use arf_gpu::{weights, GpuContext, GpuModel};

/// Parse a `--quant` string into a [`Quant`] (same set the CLI accepts).
pub fn parse_quant(s: &str) -> Result<Quant, Box<dyn Error>> {
    match s.to_ascii_lowercase().as_str() {
        "none" => Ok(Quant::None),
        "int8" => Ok(Quant::Int8),
        "q4" => Ok(Quant::Q4),
        "q4k" | "q4_k" => Ok(Quant::Q4K),
        "q4ks" | "q4_k_s" => Ok(Quant::Q4KS),
        "q3k" | "q3_k" => Ok(Quant::Q3K),
        other => Err(format!(
            "unknown --quant {other:?} (expected none, int8, q4, q4k, q4ks, or q3k)"
        )
        .into()),
    }
}

/// Parse a `--kv-quant` string into a [`KvQuant`] (same set the CLI accepts):
/// `none`, TurboQuant `tq2`/`tq3`/`tq4`, and the QJL-residual variants
/// `tq2q`/`tq3q`/`tq4q`.
pub fn parse_kv_quant(s: &str) -> Result<KvQuant, Box<dyn Error>> {
    match s.to_ascii_lowercase().as_str() {
        "none" => Ok(KvQuant::None),
        "tq2" => Ok(KvQuant::Tq {
            bits: 2,
            qjl: false,
        }),
        "tq3" => Ok(KvQuant::Tq {
            bits: 3,
            qjl: false,
        }),
        "tq4" => Ok(KvQuant::Tq {
            bits: 4,
            qjl: false,
        }),
        "tq2q" => Ok(KvQuant::Tq { bits: 2, qjl: true }),
        "tq3q" => Ok(KvQuant::Tq { bits: 3, qjl: true }),
        "tq4q" => Ok(KvQuant::Tq { bits: 4, qjl: true }),
        other => Err(format!(
            "unknown --kv-quant {other:?} (expected none, tq2, tq3, tq4, or tq2q/tq3q/tq4q)"
        )
        .into()),
    }
}

/// Map an `--arch` name to a built-in [`ModelConfig`].
///
/// Delegates to `arf_core::config::config_for_arch`. This function's doc comment used to say
/// "Mirrors the CLI's `config_for_arch`" — and it was the THIRD copy of that table. All three had
/// drifted from each other, so which architectures were supported depended on which binary and
/// which code path you happened to hit.
pub fn config_for_arch(arch: &str) -> Result<ModelConfig, Box<dyn Error>> {
    arf_core::config::config_for_arch(arch).map_err(|e| e.into())
}

/// Resolve the model config: an explicit `--arch` wins (required for a GGUF blob,
/// which has no `config.json`); otherwise read a sibling `config.json`; otherwise
/// fall back to the Llama-3.2-1B config. Mirrors the CLI's `resolve_config`.
pub fn resolve_config(model: &Path, arch: Option<&str>) -> Result<ModelConfig, Box<dyn Error>> {
    // Qwen3-Omni's Thinker (GGUF arch `qwen3vlmoe`, M3 2026-09-27): its header carries every
    // hyperparameter, so it is the config — with no `--arch`, or with `--arch qwen3-omni`. An
    // explicit other `--arch` still wins, exactly as before.
    let omni_or_none =
        arch.is_none_or(|a| arf_core::config::canonical_arch(a) == Some("qwen3-omni-30b"));
    if omni_or_none && is_gguf(model) {
        let g = arf_core::model::gguf::LazyGguf::open_raw(model)?;
        if let Some(h) = arf_core::config::qwen3moe_family_from_gguf(&g)? {
            eprintln!(
                "architecture from the GGUF header: {} ({} layers, hidden {}, vocab {}, rope base \
                 {}, {} experts top-{}; M-RoPE sections {:?} unused for text/audio; deepstack \
                 layers {})",
                h.arch,
                h.cfg.num_layers,
                h.cfg.hidden_size,
                h.cfg.vocab_size,
                h.cfg.rope_theta,
                match h.cfg.mlp {
                    arf_core::config::MlpKind::Moe { num_experts, .. } => num_experts,
                    _ => 0,
                },
                match h.cfg.mlp {
                    arf_core::config::MlpKind::Moe { top_k, .. } => top_k,
                    _ => 0,
                },
                h.mrope_sections,
                h.deepstack_layers
            );
            return Ok(h.cfg);
        }
    }
    if let Some(arch) = arch {
        let cfg = config_for_arch(arch)?;
        cfg.validate()?;
        eprintln!("using architecture --arch {arch}");
        return Ok(cfg);
    }
    let dir = if model.is_dir() {
        Some(model.to_path_buf())
    } else {
        model.parent().map(|p| p.to_path_buf())
    };
    if let Some(dir) = dir {
        let cfg_path = dir.join("config.json");
        if cfg_path.is_file() {
            let cfg = arf_core::model::hf_config::load_config_json(&cfg_path)?;
            cfg.validate()?;
            eprintln!("loaded architecture from {}", cfg_path.display());
            return Ok(cfg);
        }
    }
    if is_gguf(model) {
        // The file's own header (2026-10-07): a GGUF pulled from any repository names its
        // architecture and geometry, and when both are a built-in config's, that is the config.
        let g = arf_core::model::gguf::LazyGguf::open_raw(model)?;
        if let Some(arch) = arf_core::config::arch_from_gguf(&g) {
            let cfg = config_for_arch(arch)?;
            cfg.validate()?;
            eprintln!("architecture from the GGUF header: {arch}");
            return Ok(cfg);
        }
        return Err(arf_core::config::gguf_header_mismatch(&g).into());
    }
    // A DIRECTORY WITH NOTHING TO LOAD IS AN ERROR, NOT A LLAMA (2026-10-07): a folder holding a
    // GGUF under another name fell through to the Llama-3.2-1B config below and failed later on
    // its tokenizer, a message about a file that was never the problem.
    if model.is_dir()
        && !std::fs::read_dir(model).is_ok_and(|d| {
            d.flatten()
                .any(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
        })
    {
        return Err(format!(
            "{} holds no model: looked for model.gguf, a single *.gguf, or config.json with \
             *.safetensors",
            model.display()
        )
        .into());
    }
    Ok(ModelConfig::llama_3_2_1b())
}

/// True if `path` is a single file whose first four bytes are the `GGUF` magic.
/// Mirrors the CLI's `is_gguf`.
pub fn is_gguf(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    let mut buf = [0u8; 4];
    std::fs::File::open(path)
        .and_then(|mut f| std::io::Read::read_exact(&mut f, &mut buf))
        .is_ok()
        && &buf == b"GGUF"
}

/// Collect safetensors shards from a file or directory path. Mirrors the CLI's
/// `collect_safetensors`.
pub fn collect_safetensors(path: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    if path.is_dir() {
        let mut shards: Vec<PathBuf> = std::fs::read_dir(path)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        shards.sort();
        if shards.is_empty() {
            return Err(format!("no .safetensors files in {}", path.display()).into());
        }
        return Ok(shards);
    }
    Err(format!("model path not found: {}", path.display()).into())
}

/// Load the resident [`GpuModel`] ONCE. Picks the GGUF or safetensors loader by
/// magic detection, exactly like the CLI's GPU path. This is the one heavy step
/// at startup; after it returns the weights live on the GPU for the process life.
///
/// `kv_blocks` sizes the GPU KV pool: `Some(n)` for the serving path (MUST equal
/// the scheduler's `EngineConfig::num_blocks` — the actor asserts it at spawn),
/// `None` for the single-sequence sizing.
pub fn load_resident(
    cfg: &ModelConfig,
    model_path: &Path,
    quant: Quant,
    kv_quant: KvQuant,
    max_context: usize,
    ctx: &Arc<GpuContext>,
    kv_blocks: Option<usize>,
) -> Result<GpuModel, Box<dyn Error>> {
    if is_gguf(model_path) {
        Ok(weights::load_gguf_gpu_kv_quant_pooled(
            cfg,
            model_path,
            max_context,
            ctx,
            quant,
            kv_quant,
            kv_blocks,
        )?)
    } else {
        let paths = collect_safetensors(model_path)?;
        Ok(weights::load_safetensors_gpu_kv_quant_pooled(
            cfg,
            &paths,
            max_context,
            ctx,
            quant,
            kv_quant,
            kv_blocks,
        )?)
    }
}
