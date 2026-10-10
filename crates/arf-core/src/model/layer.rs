//! A single decoder block.
//!
//! Llama/Qwen3: pre-norm attention and pre-norm MLP, each with a residual
//! connection — two norms (`input` before attention, `pre_ffn` before the MLP).
//!
//! Gemma adds a *post*-norm on each sub-block's output, applied before the
//! residual add — four norms total:
//! `x + post_attn(attn(input(x)))`, then `h + post_ffn(mlp(pre_ffn(h)))`.

use crate::cache::PagedKvCache;
use crate::model::attention::Attention;
use crate::model::batch::ForwardBatch;
use crate::model::mlp::Mlp;
use crate::model::moe::MoeMlp;
use crate::model::rms_norm::RmsNorm;
use crate::model::rope::Rope;
use crate::tensor::Tensor;

/// The feed-forward block: dense SwiGLU (Llama/Gemma) or mixture-of-experts
/// (Qwen3). One `forward` signature so [`DecoderLayer`] doesn't branch per arch.
#[derive(Debug, Clone)]
pub enum MlpBlock {
    Dense(Mlp),
    Moe(MoeMlp),
}

impl MlpBlock {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        match self {
            MlpBlock::Dense(m) => m.forward(x),
            MlpBlock::Moe(m) => m.forward(x),
        }
    }
}

/// `x + post_attn?(attn(input(x)))`, then `h + post_ffn?(mlp(pre_ffn(h)))`. The
/// `post_attn`/`post_ffn` norms are `Some` only for Gemma; Llama/Qwen leave them
/// `None` and the block reduces to the classic two-norm pre-norm transformer.
#[derive(Debug, Clone)]
pub struct DecoderLayer {
    input_norm: RmsNorm,
    attn: Attention,
    /// Gemma only: norm on the attention output, before the residual add.
    post_attn_norm: Option<RmsNorm>,
    /// Norm before the MLP (Llama `post_attention_layernorm` / Gemma
    /// `pre_feedforward_layernorm`).
    pre_ffn_norm: RmsNorm,
    /// Gemma only: norm on the MLP output, before the residual add.
    post_ffn_norm: Option<RmsNorm>,
    mlp: MlpBlock,
}

impl DecoderLayer {
    pub fn new(
        input_norm: RmsNorm,
        attn: Attention,
        post_attn_norm: Option<RmsNorm>,
        pre_ffn_norm: RmsNorm,
        post_ffn_norm: Option<RmsNorm>,
        mlp: MlpBlock,
    ) -> Self {
        DecoderLayer {
            input_norm,
            attn,
            post_attn_norm,
            pre_ffn_norm,
            post_ffn_norm,
            mlp,
        }
    }

    pub fn forward(
        &self,
        x: &Tensor,
        layer: usize,
        rope_global: &Rope,
        rope_local: &Rope,
        batch: &ForwardBatch,
        cache: &mut PagedKvCache,
    ) -> Tensor {
        let mut attn = self.attn.forward(
            &self.input_norm.forward(x),
            layer,
            rope_global,
            rope_local,
            batch,
            cache,
        );
        if let Some(n) = &self.post_attn_norm {
            attn = n.forward(&attn);
        }
        let h = x.add(&attn);
        let mut mlp = self.mlp.forward(&self.pre_ffn_norm.forward(&h));
        if let Some(n) = &self.post_ffn_norm {
            mlp = n.forward(&mlp);
        }
        h.add(&mlp)
    }
}
