//! The tensor-name vocabulary every loader shares.
//!
//! Both loaders built HuggingFace tensor names inline, as `&format!("{p}.self_attn.q_proj.weight")`
//! repeated at each use — the same twenty-five strings spelled out two to four times each, with
//! nothing checking that two spellings of "the q projection" agreed.
//!
//! This is the equivalent of llama.cpp's `LLM_TENSOR_*` table : a name is written once,
//! and a loader composes it. The point is not brevity. It is that adding an architecture should
//! mean *using* this vocabulary, never inventing a twenty-sixth spelling of an existing tensor.
//!
//! # Layout
//!
//! HuggingFace checkpoints name per-layer tensors `model.layers.{i}.<suffix>`. [`layer`] builds the
//! prefix; the `&'static str` constants are the suffixes. A GGUF uses different names entirely and
//! is mapped by the GGUF reader, so this module describes the *safetensors* convention only.

/// Per-layer tensor prefix: `model.layers.{i}`.
///
/// Callers that need several tensors from one layer should build this once and reuse it, which is
/// also how the existing loaders are written.
#[must_use]
pub fn layer(i: usize) -> String {
    format!("model.layers.{i}")
}

/// Full name of a per-layer tensor: `layer(i) + "." + suffix`.
///
/// ```
/// # use arf_core::model::tensor_names as tn;
/// assert_eq!(tn::layer_tensor(3, tn::ATTN_Q), "model.layers.3.self_attn.q_proj.weight");
/// ```
#[must_use]
pub fn layer_tensor(i: usize, suffix: &str) -> String {
    format!("model.layers.{i}.{suffix}")
}

// ---- model-level -----------------------------------------------------------------------------

/// The embedding table, `[vocab, hidden]`. Doubles as the lm_head when weights are tied.
pub const TOKEN_EMBD: &str = "model.embed_tokens.weight";
/// Final norm before the lm_head.
pub const OUTPUT_NORM: &str = "model.norm.weight";
/// The lm_head, when it is NOT tied to the embedding table.
pub const LM_HEAD: &str = "lm_head.weight";

// ---- attention -------------------------------------------------------------------------------

/// Query projection.
pub const ATTN_Q: &str = "self_attn.q_proj.weight";
/// Key projection.
pub const ATTN_K: &str = "self_attn.k_proj.weight";
/// Value projection.
pub const ATTN_V: &str = "self_attn.v_proj.weight";
/// Output projection.
pub const ATTN_O: &str = "self_attn.o_proj.weight";
/// Per-head RMSNorm on Q, applied before RoPE (Qwen3, Gemma 3/4).
pub const ATTN_Q_NORM: &str = "self_attn.q_norm.weight";
/// Per-head RMSNorm on K, applied before RoPE.
pub const ATTN_K_NORM: &str = "self_attn.k_norm.weight";
/// Attention output gate (Muse Glimmer's full-attention layers).
pub const ATTN_GATE: &str = "self_attn.gate_proj.weight";

// ---- norms -----------------------------------------------------------------------------------

/// Norm before attention. Present on every architecture.
pub const INPUT_NORM: &str = "input_layernorm.weight";
/// Norm after attention.
pub const POST_ATTN_NORM: &str = "post_attention_layernorm.weight";
/// Norm before the FFN — Gemma's four-norm block only.
pub const PRE_FFN_NORM: &str = "pre_feedforward_layernorm.weight";
/// Norm after the FFN — Gemma's four-norm block only.
pub const POST_FFN_NORM: &str = "post_feedforward_layernorm.weight";

// ---- feed-forward ----------------------------------------------------------------------------

/// SwiGLU gate projection.
pub const FFN_GATE: &str = "mlp.gate_proj.weight";
/// SwiGLU up projection.
pub const FFN_UP: &str = "mlp.up_proj.weight";
/// Down projection.
pub const FFN_DOWN: &str = "mlp.down_proj.weight";
/// MoE router: `[num_experts, hidden]`.
pub const FFN_ROUTER: &str = "mlp.gate.weight";

// ---- gated delta net (the hybrid's recurrent layers) ------------------------------------------

/// Fused QKV projection for a recurrent layer.
pub const GDN_QKV: &str = "linear_attn.qkv_proj.weight";
/// Causal conv1d over the recurrent ring.
pub const GDN_CONV1D: &str = "linear_attn.conv1d.weight";
/// Per-head decay gate.
pub const GDN_ALPHA: &str = "linear_attn.alpha_proj.weight";
/// Per-head write strength (the delta rule's β).
pub const GDN_BETA: &str = "linear_attn.beta_proj.weight";
/// Norm on the recurrent readout.
pub const GDN_NORM: &str = "linear_attn.norm.weight";
/// Recurrent output projection.
pub const GDN_OUT: &str = "linear_attn.out_proj.weight";

// ---- multi-token prediction head (`blk.64`) ---------------------------------------------------

/// Projects `[normed embedding ‖ normed hidden]` down to hidden width.
pub const MTP_EH_PROJ: &str = "nextn.eh_proj.weight";
/// Norm applied to the token embedding before concatenation.
pub const MTP_ENORM: &str = "nextn.enorm.weight";
/// Norm applied to the trunk hidden before concatenation.
pub const MTP_HNORM: &str = "nextn.hnorm.weight";
/// Norm before the shared lm_head.
pub const MTP_SHARED_HEAD_NORM: &str = "nextn.shared_head_norm.weight";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_prefix_and_full_name_agree() {
        assert_eq!(layer(7), "model.layers.7");
        assert_eq!(layer_tensor(7, ATTN_Q), format!("{}.{}", layer(7), ATTN_Q));
    }

    /// Every constant must be a distinct string. Two names for one tensor is the failure this
    /// module exists to make impossible.
    #[test]
    fn no_two_constants_share_a_name() {
        let all = [
            TOKEN_EMBD,
            OUTPUT_NORM,
            LM_HEAD,
            ATTN_Q,
            ATTN_K,
            ATTN_V,
            ATTN_O,
            ATTN_Q_NORM,
            ATTN_K_NORM,
            ATTN_GATE,
            INPUT_NORM,
            POST_ATTN_NORM,
            PRE_FFN_NORM,
            POST_FFN_NORM,
            FFN_GATE,
            FFN_UP,
            FFN_DOWN,
            FFN_ROUTER,
            GDN_QKV,
            GDN_CONV1D,
            GDN_ALPHA,
            GDN_BETA,
            GDN_NORM,
            GDN_OUT,
            MTP_EH_PROJ,
            MTP_ENORM,
            MTP_HNORM,
            MTP_SHARED_HEAD_NORM,
        ];
        let unique: std::collections::HashSet<_> = all.iter().collect();
        assert_eq!(unique.len(), all.len(), "a tensor name is duplicated");
    }

    /// Per-layer suffixes must NOT carry the `model.layers.{i}` prefix — that is `layer_tensor`'s
    /// job, and baking it in would produce `model.layers.3.model.layers.3.…`.
    #[test]
    fn suffixes_are_relative() {
        for s in [ATTN_Q, INPUT_NORM, FFN_UP, GDN_QKV, MTP_ENORM] {
            assert!(!s.starts_with("model."), "{s} should be a bare suffix");
        }
        for s in [TOKEN_EMBD, OUTPUT_NORM] {
            assert!(
                s.starts_with("model."),
                "{s} is model-level and should be absolute"
            );
        }
    }
}
