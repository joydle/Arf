//! The Llama model: token embedding → N decoder layers → final norm → LM head.

use crate::cache::PagedKvCache;
use crate::config::ModelConfig;
use crate::model::batch::ForwardBatch;
use crate::model::layer::DecoderLayer;
use crate::model::nn::{Embedding, Linear};
use crate::model::rms_norm::RmsNorm;
use crate::model::rope::Rope;
use crate::tensor::Tensor;

/// A loaded Llama model, ready for inference.
#[derive(Debug, Clone)]
pub struct Llama {
    embed: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    rope: Rope,
    /// Gemma's local (sliding-window) RoPE base; `None` for single-base models
    /// (Llama/Qwen), where the global `rope` is used everywhere.
    rope_local: Option<Rope>,
    cfg: ModelConfig,
}

impl Llama {
    /// Assemble from parts. Prefer [`crate::model::weights`] to build these.
    pub fn new(
        embed: Embedding,
        layers: Vec<DecoderLayer>,
        norm: RmsNorm,
        lm_head: Linear,
        rope: Rope,
        rope_local: Option<Rope>,
        cfg: ModelConfig,
    ) -> Self {
        Llama {
            embed,
            layers,
            norm,
            lm_head,
            rope,
            rope_local,
            cfg,
        }
    }

    pub fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Run the transformer trunk. Returns post-final-norm hidden states
    /// `[total_tokens, hidden]`.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(
            level = "debug",
            name = "cpu_forward",
            skip_all,
            fields(tokens = input_ids.len())
        )
    )]
    pub fn forward(
        &self,
        input_ids: &[u32],
        batch: &ForwardBatch,
        cache: &mut PagedKvCache,
    ) -> Tensor {
        let mut h = self.embed.forward(input_ids);
        // Gemma scales the embeddings by √hidden after lookup.
        if let Some(scale) = self.cfg.embedding_scale {
            for v in h.as_mut_slice() {
                *v *= scale;
            }
        }
        let rope_local = self.rope_local.as_ref().unwrap_or(&self.rope);
        for (layer, block) in self.layers.iter().enumerate() {
            h = block.forward(&h, layer, &self.rope, rope_local, batch, cache);
        }
        self.norm.forward(&h)
    }

    /// Project hidden states to vocabulary logits: `[n, hidden]` → `[n, vocab]`.
    pub fn logits(&self, hidden: &Tensor) -> Tensor {
        self.lm_head.forward(hidden)
    }

    /// Logits for only the last token of each sequence in `batch`
    /// (`[num_seqs, vocab]`) — avoids projecting the whole prefill.
    pub fn logits_last(&self, hidden: &Tensor, batch: &ForwardBatch) -> Tensor {
        let (_, hidden_size) = hidden.dims2();
        let mut rows = Vec::with_capacity(batch.seqs.len() * hidden_size);
        for s in &batch.seqs {
            rows.extend_from_slice(hidden.row(s.q_start + s.q_len - 1));
        }
        let last = Tensor::from_vec(rows, vec![batch.seqs.len(), hidden_size]);
        self.logits(&last)
    }
}
