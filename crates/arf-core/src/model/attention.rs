//! Grouped-query attention with RoPE and a paged KV cache.
//!
//! Per layer and batch:
//! 1. project Q/K/V for every token,
//! 2. apply RoPE to Q and K,
//! 3. write the new K/V into the paged cache,
//! 4. for each sequence and head, gather the context and run causal scaled
//!    dot-product attention for its query tokens.

use crate::cache::PagedKvCache;
use crate::config::ModelConfig;
use crate::model::batch::{ForwardBatch, SeqAttn};
use crate::model::nn::Linear;
use crate::model::qk_norm::QkNorm;
use crate::model::rope::Rope;
use crate::tensor::{matmul, matmul_nt, Tensor};

/// One attention layer's projections and shape metadata.
#[derive(Debug, Clone)]
pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    /// Per-head RMSNorm on q/k before RoPE (Qwen3/Gemma). `None` for Llama.
    q_norm: Option<QkNorm>,
    k_norm: Option<QkNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    group: usize,
    scale: f32,
    /// The config's `query_pre_attn_scalar` (`None` = use `1/√head_dim`). Kept so a
    /// per-layer geometry change can recompute the default scale for its head_dim
    /// while a pinned scalar stays fixed.
    query_pre_attn_scalar: Option<f32>,
    /// Number of leading head dims RoPE rotates (partial rotary). Equals `head_dim`
    /// for full rotation (Llama/Qwen/Gemma-3 and Gemma-4 sliding layers); smaller for
    /// Gemma-4 global layers (e.g. 128 of 512). The non-rotated tail passes through.
    rotary_dim: usize,
    /// Sliding-window size for a Gemma *local* layer (each query attends only to
    /// the last `window` keys); `None` is a global/causal layer (Llama, Qwen3, and
    /// Gemma's periodic global layers). Local layers also use the local RoPE base.
    window: Option<usize>,
}

impl Attention {
    pub fn new(
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        cfg: &ModelConfig,
    ) -> Self {
        Attention {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: None,
            k_norm: None,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            group: cfg.gqa_group_size(),
            scale: cfg
                .query_pre_attn_scalar
                .unwrap_or(1.0 / (cfg.head_dim as f32).sqrt()),
            query_pre_attn_scalar: cfg.query_pre_attn_scalar,
            window: None,
            rotary_dim: cfg.head_dim,
        }
    }

    /// Override the per-layer attention geometry (Gemma 4: global layers differ from
    /// the base/sliding geometry). Recomputes `group` and, when no explicit
    /// `query_pre_attn_scalar` is set, the `1/√head_dim` scale for THIS layer's
    /// head_dim. Builder-style; non-Gemma-4 layers pass the base values (a no-op vs
    /// `new`). `scale` is left as-is only when the config pinned `query_pre_attn_scalar`.
    pub fn with_geometry(
        mut self,
        head_dim: usize,
        num_kv_heads: usize,
        rotary_dim: usize,
    ) -> Self {
        self.head_dim = head_dim;
        self.num_kv_heads = num_kv_heads;
        self.rotary_dim = rotary_dim;
        self.group = self.num_heads / num_kv_heads.max(1);
        // Only recompute the default 1/√head_dim scale; a config-pinned
        // query_pre_attn_scalar (set in `new`) stays fixed across layers.
        if self.query_pre_attn_scalar.is_none() {
            self.scale = 1.0 / (head_dim as f32).sqrt();
        }
        self
    }

    /// Attach per-head q/k RMSNorm (Qwen3/Gemma). Builder-style so the Llama path
    /// (which never calls this) is unchanged.
    pub fn with_qk_norm(mut self, q_norm: QkNorm, k_norm: QkNorm) -> Self {
        self.q_norm = Some(q_norm);
        self.k_norm = Some(k_norm);
        self
    }

    /// Mark this as a Gemma sliding-window *local* layer (`Some(window)`) or leave
    /// it global/causal (`None`). Builder-style; the Llama/Qwen path never calls it.
    pub fn with_window(mut self, window: Option<usize>) -> Self {
        self.window = window;
        self
    }

    /// `x`: `[total_tokens, hidden]` → `[total_tokens, hidden]`.
    ///
    /// `rope_global`/`rope_local` are the two RoPE bases: a local (sliding-window)
    /// layer rotates with `rope_local`, a global/causal layer with `rope_global`.
    /// For Llama/Qwen both arguments are the same table.
    pub fn forward(
        &self,
        x: &Tensor,
        layer: usize,
        rope_global: &Rope,
        rope_local: &Rope,
        batch: &ForwardBatch,
        cache: &mut PagedKvCache,
    ) -> Tensor {
        let total = x.dims2().0;
        let (h, kv, hd) = (self.num_heads, self.num_kv_heads, self.head_dim);
        // Local (sliding-window) layers use the local RoPE base; everyone else global.
        let rope = if self.window.is_some() {
            rope_local
        } else {
            rope_global
        };

        // Project, split into heads, per-head-RMSNorm (Qwen3/Gemma), then rotate Q/K.
        let mut q = self.q_proj.forward(x).reshape(vec![total, h, hd]);
        let mut k = self.k_proj.forward(x).reshape(vec![total, kv, hd]);
        let v = self.v_proj.forward(x).reshape(vec![total, kv, hd]);
        if let Some(qn) = &self.q_norm {
            qn.apply(q.as_mut_slice());
        }
        if let Some(kn) = &self.k_norm {
            kn.apply(k.as_mut_slice());
        }
        let q = rope.apply(&q, &batch.positions);
        let k = rope.apply(&k, &batch.positions);

        // Persist this step's K/V per sequence.
        let row = kv * hd; // floats per cached token
        for seq in &batch.seqs {
            let lo = seq.q_start * row;
            let hi = (seq.q_start + seq.q_len) * row;
            cache.write(
                layer,
                &k.as_slice()[lo..hi],
                &v.as_slice()[lo..hi],
                &seq.write_runs,
            );
        }

        // Per-sequence causal attention; assemble [total, h*hd].
        let mut out = vec![0.0f32; total * h * hd];
        for seq in &batch.seqs {
            self.attend_sequence(seq, &q, layer, cache, &mut out);
        }
        let attn = Tensor::from_vec(out, vec![total, h * hd]);
        self.o_proj.forward(&attn)
    }

    /// Attention for a single sequence, writing into `out` (`[total, h*hd]`).
    fn attend_sequence(
        &self,
        seq: &SeqAttn,
        q: &Tensor,
        layer: usize,
        cache: &PagedKvCache,
        out: &mut [f32],
    ) {
        let (h, kv, hd) = (self.num_heads, self.num_kv_heads, self.head_dim);
        let q_len = seq.q_len;
        let ctx = seq.context_len();
        let (k_all, v_all) = cache.gather(layer, &seq.slots); // [ctx, kv, hd] each

        for head in 0..h {
            let kv_head = head / self.group;
            let q_h = gather_head(q.as_slice(), h, hd, head, seq.q_start, q_len);
            let k_h = gather_head(&k_all, kv, hd, kv_head, 0, ctx);
            let v_h = gather_head(&v_all, kv, hd, kv_head, 0, ctx);

            // scores = scale * Q Kᵀ  -> [q_len, ctx]
            let mut scores = matmul_nt(&q_h, q_len, hd, &k_h, ctx);
            for s in scores.iter_mut() {
                *s *= self.scale;
            }
            causal_softmax(&mut scores, q_len, ctx, seq.past_len, self.window);

            // out_h = softmax(scores) V  -> [q_len, hd]
            let out_h = matmul(&scores, q_len, ctx, &v_h, hd);

            // Scatter into the per-token, per-head output layout.
            for r in 0..q_len {
                let dst = ((seq.q_start + r) * h + head) * hd;
                out[dst..dst + hd].copy_from_slice(&out_h[r * hd..(r + 1) * hd]);
            }
        }
    }
}

/// Gather `row_count` rows of one head into a contiguous `[row_count, hd]`
/// buffer, from a `[rows, n_heads, hd]` source starting at `row_start`.
fn gather_head(
    src: &[f32],
    n_heads: usize,
    hd: usize,
    head: usize,
    row_start: usize,
    row_count: usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(row_count * hd);
    for r in 0..row_count {
        let base = ((row_start + r) * n_heads + head) * hd;
        out.extend_from_slice(&src[base..base + hd]);
    }
    out
}

/// In-place causal mask + row softmax on `scores [q_len, ctx]`. Query `i` (at
/// absolute position `past_len + i`) attends to keys `lo..=past_len+i`, where
/// `lo` is `0` for a global/causal layer or `last + 1 - window` for a Gemma
/// sliding-window layer (each query sees only the most recent `window` keys).
fn causal_softmax(
    scores: &mut [f32],
    q_len: usize,
    ctx: usize,
    past_len: usize,
    window: Option<usize>,
) {
    for i in 0..q_len {
        let row = &mut scores[i * ctx..(i + 1) * ctx];
        let last = past_len + i; // highest attendable key index
        let lo = match window {
            Some(w) => (last + 1).saturating_sub(w),
            None => 0,
        };
        let mut max = f32::NEG_INFINITY;
        for &v in row.iter().take(last + 1).skip(lo) {
            max = max.max(v);
        }
        let mut sum = 0.0f32;
        for (j, v) in row.iter_mut().enumerate() {
            if j >= lo && j <= last {
                *v = (*v - max).exp();
                sum += *v;
            } else {
                *v = 0.0;
            }
        }
        let inv = 1.0 / sum;
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}
