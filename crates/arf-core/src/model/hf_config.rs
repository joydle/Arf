//! Load a [`ModelConfig`] from a HuggingFace `config.json`, detecting the
//! architecture (Llama / Qwen3-MoE / Gemma 3) and mapping its fields onto our
//! multi-architecture config. Keeps the hand-built `ModelConfig::llama_3_2_1b()`
//! etc. as fallbacks; this is the path for loading arbitrary checkpoints by dir.
//!
//! We parse with `serde_json::Value` (no derive structs) — we only pull a handful
//! of fields and the schemas differ per architecture, so untyped access is simpler
//! and robust to extra keys.

use std::path::Path;

use serde_json::Value;

use crate::config::{AttnKind, MlpKind, ModelConfig, NormStyle, RopeScaling};
use crate::error::{ArfError, Result};

/// Parse `config.json` at `path` into a [`ModelConfig`].
pub fn load_config_json(path: &Path) -> Result<ModelConfig> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| ArfError::model_load(format!("read {path:?}: {e}")))?;
    let v: Value = serde_json::from_str(&text)
        .map_err(|e| ArfError::model_load(format!("parse {path:?}: {e}")))?;
    from_value(&v)
}

/// Build a [`ModelConfig`] from a parsed `config.json` value.
pub fn from_value(v: &Value) -> Result<ModelConfig> {
    // Multimodal wrappers (Gemma 4 `Gemma4ForConditionalGeneration`) nest the text
    // backbone fields under `text_config`, with the multimodal `model_type` at top
    // level (e.g. "gemma4") and the text `model_type` inside (e.g. "gemma4_text").
    // Descend into `text_config` when present so the rest of this fn reads the text
    // tower. Flat text-only configs (Gemma 3, our tests) are used as-is.
    if let Some(tc) = v.get("text_config").filter(|x| x.is_object()) {
        let mut cfg = from_value(tc)?;
        // The wrapper may state the tie itself, outside `text_config`.
        if let Some(t) = v.get("tie_word_embeddings").and_then(|x| x.as_bool()) {
            cfg.tie_word_embeddings = t;
        }
        return Ok(cfg);
    }
    // Helpers: read a field, with a clear error if missing/wrong type.
    let u = |k: &str| -> Result<usize> {
        v.get(k)
            .and_then(|x| x.as_u64())
            .map(|n| n as usize)
            .ok_or_else(|| ArfError::model_load(format!("config.json missing usize {k:?}")))
    };
    let uf = |k: &str, default: usize| {
        v.get(k)
            .and_then(|x| x.as_u64())
            .map_or(default, |n| n as usize)
    };
    let f = |k: &str, default: f64| v.get(k).and_then(|x| x.as_f64()).unwrap_or(default);
    let b = |k: &str, default: bool| v.get(k).and_then(|x| x.as_bool()).unwrap_or(default);

    let hidden_size = u("hidden_size")?;
    let num_attention_heads = u("num_attention_heads")?;
    // head_dim is explicit in Gemma/Qwen3; default to hidden/heads (Llama).
    let head_dim = uf("head_dim", hidden_size / num_attention_heads.max(1));
    let num_kv_heads = uf("num_key_value_heads", num_attention_heads);

    // Architecture detection from `model_type` (lowercased) or `architectures[0]`.
    let model_type = v
        .get("model_type")
        .and_then(|x| x.as_str())
        .map(str::to_ascii_lowercase)
        .or_else(|| {
            v.get("architectures")
                .and_then(|a| a.get(0))
                .and_then(|x| x.as_str())
                .map(str::to_ascii_lowercase)
        })
        .unwrap_or_default();

    // MoE (Qwen3-MoE): presence of `num_experts` / `num_experts_per_tok`.
    let mlp = if let Some(num_experts) = v.get("num_experts").and_then(|x| x.as_u64()) {
        MlpKind::Moe {
            num_experts: num_experts as usize,
            top_k: uf("num_experts_per_tok", 8),
            shared_experts: uf("num_shared_experts", 0)
                .max(uf("shared_expert_intermediate_size", 0).min(1)),
            moe_intermediate: uf("moe_intermediate_size", uf("intermediate_size", 0)),
            norm_topk: b("norm_topk_prob", false),
        }
    } else {
        MlpKind::Dense
    };

    // QK-norm: Qwen3 + Gemma (3 and 4). Detect by model_type (q/k_norm tensors
    // confirm at load). Gemma 4 shares Gemma 3's transformer architecture (the new
    // work in Gemma 4 is its encoder-free multimodal *inputs*; the text backbone is
    // the same family), so it maps onto the same Gemma config path here.
    let is_qwen3 = model_type.contains("qwen3");
    let is_gemma3 = model_type.contains("gemma3") || model_type.contains("gemma_3");
    let is_gemma4 = model_type.contains("gemma4") || model_type.contains("gemma_4");
    let is_gemma = is_gemma3 || is_gemma4;
    let qk_norm = is_qwen3 || is_gemma;

    let norm_style = if is_gemma {
        NormStyle::Gemma
    } else {
        NormStyle::Llama
    };

    // Attention: Gemma interleaves sliding-window local layers with periodic global
    // ones — but only when the checkpoint declares a `sliding_window`. Without it
    // (or for any non-Gemma model) attention is plain causal.
    let has_sliding_window = v.get("sliding_window").and_then(|x| x.as_u64()).is_some();
    let attn = if is_gemma && has_sliding_window {
        // RoPE thetas: Gemma 4 nests them in `rope_parameters.{sliding,full}_attention`;
        // Gemma 3 uses the flat `rope_local_base_freq` / `rope_theta`. Prefer the
        // nested form when present, else fall back to the flat keys.
        let rope_params = v.get("rope_parameters");
        let nested_f = |group: &str, key: &str| -> Option<f64> {
            rope_params
                .and_then(|rp| rp.get(group))
                .and_then(|g| g.get(key))
                .and_then(|x| x.as_f64())
        };
        let local_theta = nested_f("sliding_attention", "rope_theta")
            .unwrap_or_else(|| f("rope_local_base_freq", 10_000.0));
        let global_theta = nested_f("full_attention", "rope_theta")
            .unwrap_or_else(|| f("rope_theta", 1_000_000.0));
        // Per-layer-type geometry. The base head_dim/num_kv_heads describe the
        // SLIDING (local) layers; Gemma 4 gives GLOBAL layers a distinct head_dim
        // (`global_head_dim`), KV-head count (`num_global_key_value_heads`), and
        // RoPE rotary fraction (`rope_parameters.full_attention.partial_rotary_factor`).
        // Default to the sliding geometry (== Gemma 3) when those keys are absent.
        let global_head_dim = uf("global_head_dim", head_dim);
        let global_kv_heads = uf("num_global_key_value_heads", num_kv_heads);
        let partial_rotary_factor = nested_f("full_attention", "partial_rotary_factor")
            .map(|x| x as f32)
            .unwrap_or(1.0);
        AttnKind::HybridLocalGlobal {
            window: uf("sliding_window", 1024),
            global_every: uf("sliding_window_pattern", 6),
            local_theta,
            global_theta,
            global_head_dim,
            global_kv_heads,
            partial_rotary_factor,
        }
    } else {
        AttnKind::Causal
    };

    let embedding_scale = if is_gemma {
        Some((hidden_size as f32).sqrt())
    } else {
        None
    };
    // Pre-softmax query scale. Gemma 4 uses attention scale = 1.0: its per-head
    // qk_norm (RMS_NORM + weight on Q/K) already normalizes the queries/keys, so the
    // usual 1/√d pre-attention scaling is dropped (llama.cpp gemma4.cpp sets
    // `f_attention_scale = 1.0f`). The `query_pre_attn_scalar` field is still present
    // in Gemma 4 config.json but is NOT applied as the attention scale — applying
    // 1/√256 flattens the softmax and breaks the BOS attention sink. Gemma 3 (and any
    // other arch carrying the field) keeps the 1/√query_pre_attn_scalar behavior.
    let query_pre_attn_scalar = if is_gemma4 {
        Some(1.0)
    } else {
        v.get("query_pre_attn_scalar")
            .and_then(|x| x.as_f64())
            .map(|q| 1.0 / (q as f32).sqrt())
    };

    // Final-logit soft-cap (Gemma 4 = 30.0; absent on Gemma 3 / Llama / Qwen → None).
    let final_logit_softcap = v
        .get("final_logit_softcapping")
        .and_then(|x| x.as_f64())
        .map(|c| c as f32);
    // Gemma 4 weightless per-head V-norm (`value_norm`). HF marks it via arch, not a
    // bool, so default off; the `--arch` constructors carry the known truth.
    let value_norm = b("value_norm", false);

    // RoPE scaling (Llama 3 piecewise) — only if rope_scaling.rope_type == "llama3".
    let rope_scaling = parse_rope_scaling(v);

    // The global RoPE base drives `ModelConfig.rope_theta`. Gemma 4 nests it under
    // `rope_parameters.full_attention.rope_theta`; otherwise read the flat key.
    let rope_theta = v
        .get("rope_parameters")
        .and_then(|rp| rp.get("full_attention"))
        .and_then(|g| g.get("rope_theta"))
        .and_then(|x| x.as_f64())
        .unwrap_or_else(|| f("rope_theta", 10_000.0));

    Ok(ModelConfig {
        // HF configs do not describe an MTP head; GGUF sets this from nextn_predict_layers.
        nextn_layers: 0,
        vocab_size: u("vocab_size")?,
        hidden_size,
        intermediate_size: uf("intermediate_size", 0),
        num_layers: u("num_hidden_layers")?,
        num_attention_heads,
        num_kv_heads,
        head_dim,
        rms_norm_eps: f("rms_norm_eps", 1e-6),
        rope_theta,
        rope_scaling,
        max_position_embeddings: uf("max_position_embeddings", 32_768),
        // Absent, Gemma ties the output head to the embeddings (HF's Gemma configs default it on,
        // and unsloth/gemma-3-4b-it states it nowhere: without this its load failed on a missing
        // `lm_head.weight`, 2026-10-06).
        tie_word_embeddings: b("tie_word_embeddings", is_gemma),
        mlp,
        qk_norm,
        norm_style,
        attn,
        embedding_scale,
        query_pre_attn_scalar,
        final_logit_softcap,
        value_norm,
        // Gemma's FFN is GELU-tanh-gated (gelu_pytorch_tanh); Llama/Qwen are SiLU.
        gate_act: if is_gemma {
            crate::config::GateAct::GeluTanh
        } else {
            crate::config::GateAct::Silu
        },
    })
}

fn parse_rope_scaling(v: &Value) -> RopeScaling {
    let Some(rs) = v.get("rope_scaling").filter(|x| !x.is_null()) else {
        return RopeScaling::None;
    };
    let kind = rs
        .get("rope_type")
        .or_else(|| rs.get("type"))
        .and_then(|x| x.as_str())
        .unwrap_or("");
    if kind == "llama3" {
        RopeScaling::Llama3 {
            factor: rs.get("factor").and_then(|x| x.as_f64()).unwrap_or(8.0),
            low_freq_factor: rs
                .get("low_freq_factor")
                .and_then(|x| x.as_f64())
                .unwrap_or(1.0),
            high_freq_factor: rs
                .get("high_freq_factor")
                .and_then(|x| x.as_f64())
                .unwrap_or(4.0),
            original_max_position: rs
                .get("original_max_position_embeddings")
                .and_then(|x| x.as_u64())
                .map(|n| n as usize)
                .unwrap_or(8192),
        }
    } else {
        RopeScaling::None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_llama_3_2_1b_shape() {
        let j = serde_json::json!({
            "model_type": "llama",
            "vocab_size": 128256, "hidden_size": 2048, "intermediate_size": 8192,
            "num_hidden_layers": 16, "num_attention_heads": 32, "num_key_value_heads": 8,
            "rms_norm_eps": 1e-5, "rope_theta": 500000.0, "tie_word_embeddings": true,
            "max_position_embeddings": 131072,
            "rope_scaling": {"rope_type": "llama3", "factor": 32.0,
                "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192}
        });
        let c = from_value(&j).unwrap();
        assert_eq!(c.hidden_size, 2048);
        assert_eq!(c.head_dim, 64); // 2048/32 default
        assert_eq!(c.mlp, MlpKind::Dense);
        assert!(!c.qk_norm);
        assert_eq!(c.norm_style, NormStyle::Llama);
        assert!(matches!(c.rope_scaling, RopeScaling::Llama3 { .. }));
        c.validate().unwrap();
    }

    #[test]
    fn parses_qwen3_moe_shape() {
        let j = serde_json::json!({
            "model_type": "qwen3_moe",
            "vocab_size": 151936, "hidden_size": 2048, "intermediate_size": 768,
            "num_hidden_layers": 48, "num_attention_heads": 32, "num_key_value_heads": 4,
            "head_dim": 64, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
            "tie_word_embeddings": true, "max_position_embeddings": 32768,
            "num_experts": 128, "num_experts_per_tok": 8, "moe_intermediate_size": 768,
            "norm_topk_prob": true
        });
        let c = from_value(&j).unwrap();
        assert!(c.qk_norm, "Qwen3 has QK-norm");
        match c.mlp {
            MlpKind::Moe {
                num_experts,
                top_k,
                moe_intermediate,
                norm_topk,
                ..
            } => {
                assert_eq!((num_experts, top_k, moe_intermediate), (128, 8, 768));
                assert!(norm_topk);
            }
            _ => panic!("expected MoE"),
        }
        c.validate().unwrap();
    }

    #[test]
    fn a_gemma3_multimodal_wrapper_without_the_key_ties_the_head() {
        let v: Value = serde_json::from_str(
            r#"{"architectures": ["Gemma3ForConditionalGeneration"], "model_type": "gemma3",
                "text_config": {"model_type": "gemma3_text", "hidden_size": 2560,
                "intermediate_size": 10240, "num_hidden_layers": 34, "num_attention_heads": 8,
                "num_key_value_heads": 4, "head_dim": 256, "vocab_size": 262208,
                "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "sliding_window": 1024,
                "max_position_embeddings": 131072}}"#,
        )
        .unwrap();
        assert!(from_value(&v).unwrap().tie_word_embeddings);
        let mut untied = v.clone();
        untied["tie_word_embeddings"] = serde_json::json!(false);
        assert!(!from_value(&untied).unwrap().tie_word_embeddings);
    }

    #[test]
    fn parses_gemma3_shape() {
        let j = serde_json::json!({
            "model_type": "gemma3_text",
            "vocab_size": 262208, "hidden_size": 2304, "intermediate_size": 9216,
            "num_hidden_layers": 26, "num_attention_heads": 8, "num_key_value_heads": 4,
            "head_dim": 256, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
            "tie_word_embeddings": true, "max_position_embeddings": 131072,
            "sliding_window": 1024, "rope_local_base_freq": 10000.0
        });
        let c = from_value(&j).unwrap();
        assert_eq!(c.head_dim, 256); // explicit, != hidden/heads
        assert_eq!(c.norm_style, NormStyle::Gemma);
        assert!(c.qk_norm);
        assert!(c.embedding_scale.is_some());
        assert!(matches!(c.attn, AttnKind::HybridLocalGlobal { .. }));
        c.validate().unwrap(); // Gemma head_dim check is relaxed
    }

    #[test]
    fn parses_gemma4_shape() {
        // Gemma 4 12B reuses the Gemma transformer family, so it maps onto the
        // same Gemma config path (4 norms, QK-norm, embedding scale, hybrid
        // local/global attention with two RoPE bases).
        let j = serde_json::json!({
            "model_type": "gemma4_text",
            "vocab_size": 262_208, "hidden_size": 3840, "intermediate_size": 15_360,
            "num_hidden_layers": 48, "num_attention_heads": 16, "num_key_value_heads": 8,
            "head_dim": 256, "rms_norm_eps": 1e-6, "rope_theta": 1_000_000.0,
            "tie_word_embeddings": true, "max_position_embeddings": 131_072,
            "sliding_window": 1024, "sliding_window_pattern": 6,
            "rope_local_base_freq": 10_000.0
        });
        let c = from_value(&j).unwrap();
        assert_eq!(c.norm_style, NormStyle::Gemma);
        assert!(c.qk_norm, "Gemma 4 has QK-norm");
        assert!(c.embedding_scale.is_some());
        assert_eq!(c.mlp, MlpKind::Dense, "Gemma 4 12B is dense, not MoE");
        match c.attn {
            AttnKind::HybridLocalGlobal {
                window,
                global_every,
                local_theta,
                global_theta,
                global_head_dim,
                global_kv_heads,
                partial_rotary_factor,
            } => {
                assert_eq!((window, global_every), (1024, 6));
                assert_eq!(local_theta, 10_000.0);
                assert_eq!(global_theta, 1_000_000.0);
                // No global-geometry keys in this flat config → defaults to sliding.
                assert_eq!(global_head_dim, 256);
                assert_eq!(global_kv_heads, 8);
                assert_eq!(partial_rotary_factor, 1.0);
            }
            _ => panic!("expected hybrid local/global attention"),
        }
        // No Gemma-4 deltas declared here → gemma3-equivalent defaults.
        assert_eq!(c.final_logit_softcap, None);
        assert!(!c.value_norm);
        c.validate().unwrap();
    }

    #[test]
    fn parses_gemma4_31b_text_config_deltas() {
        // The real `Gemma4ForConditionalGeneration` config nests the text backbone
        // under `text_config` and carries the three Gemma-4 deltas: final-logit
        // softcap, K==V shared projection, and per-layer-type geometry (global
        // layers differ from sliding ones), with RoPE thetas nested in
        // `rope_parameters`.
        let j = serde_json::json!({
            "model_type": "gemma4",
            "architectures": ["Gemma4ForConditionalGeneration"],
            "tie_word_embeddings": true,
            "text_config": {
                "model_type": "gemma4_text",
                "vocab_size": 262_144, "hidden_size": 5376, "intermediate_size": 21_504,
                "num_hidden_layers": 60, "num_attention_heads": 32,
                "num_key_value_heads": 16, "head_dim": 256,
                "global_head_dim": 512, "num_global_key_value_heads": 4,
                "rms_norm_eps": 1e-6, "tie_word_embeddings": true,
                "max_position_embeddings": 262_144,
                "sliding_window": 1024,
                "final_logit_softcapping": 30.0,
                "value_norm": true,
                "query_pre_attn_scalar": 256,
                "rope_parameters": {
                    "full_attention": {
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 1_000_000.0,
                        "rope_type": "proportional"
                    },
                    "sliding_attention": {
                        "rope_theta": 10_000.0,
                        "rope_type": "default"
                    }
                }
            }
        });
        let c = from_value(&j).unwrap();
        assert_eq!(c.norm_style, NormStyle::Gemma);
        assert!(c.qk_norm);
        assert_eq!(c.hidden_size, 5376);
        assert_eq!(c.num_layers, 60);
        assert_eq!(c.num_attention_heads, 32);
        assert_eq!(c.num_kv_heads, 16); // sliding geometry on the base fields
        assert_eq!(c.head_dim, 256);
        assert_eq!(c.rope_theta, 1_000_000.0); // global base from rope_parameters
        assert_eq!(c.final_logit_softcap, Some(30.0));
        assert!(c.value_norm);
        match c.attn {
            AttnKind::HybridLocalGlobal {
                window,
                global_every,
                local_theta,
                global_theta,
                global_head_dim,
                global_kv_heads,
                partial_rotary_factor,
            } => {
                assert_eq!((window, global_every), (1024, 6));
                assert_eq!(local_theta, 10_000.0);
                assert_eq!(global_theta, 1_000_000.0);
                assert_eq!(global_head_dim, 512);
                assert_eq!(global_kv_heads, 4);
                assert_eq!(partial_rotary_factor, 0.25);
            }
            _ => panic!("expected hybrid local/global attention"),
        }
        c.validate().unwrap();

        // The hand-built constructor must agree with the parsed config.
        assert_eq!(c, ModelConfig::gemma4_31b());
    }

    #[test]
    fn gemma_without_sliding_window_is_causal() {
        // A Gemma-family text config that omits `sliding_window` falls back to
        // plain causal attention rather than windowing every layer.
        let j = serde_json::json!({
            "model_type": "gemma4_text",
            "vocab_size": 1000, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 16, "rms_norm_eps": 1e-6, "rope_theta": 1_000_000.0,
            "tie_word_embeddings": true, "max_position_embeddings": 8192
        });
        let c = from_value(&j).unwrap();
        assert_eq!(c.norm_style, NormStyle::Gemma);
        assert!(matches!(c.attn, AttnKind::Causal));
        c.validate().unwrap();
    }
}
