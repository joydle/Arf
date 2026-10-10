//! GPU CLIP-L text encoder (FLUX's pooled-conditioning encoder). Runs the 12-layer CLIP text
//! transformer entirely on the GPU — token+position embed, then per layer: LN1 → causal MHA →
//! residual, LN2 → fc1 → quick_gelu → fc2 → residual, then final LayerNorm. The pooled output
//! is the hidden state at the EOS token (CLIP's `[768]` pooled embedding that feeds FLUX's
//! adaLN modulation).
//!
//! Self-contained in the `GpuVisionEncoder` mould: owns its `ComputeKernel`s, uploads resident
//! bf16 weights (matmuls) + f32 norms/biases, reuses the shared kernel library
//! (`matmul`, `layernorm_bias`, `add_bias`, `text_attn`, `add`) + the CLIP-only `quick_gelu`.
//! All compute is on Metal; the CPU only gathers the token/position embeddings into the initial
//! hidden buffer (input prep) and reads back the final pooled vector.

use super::wg;
use std::sync::Arc;

use arf_core::model::weights::read_weights;
use arf_core::{ArfError, Result};

use crate::gpu::{ComputeKernel, GpuContext};

/// CLIP-L config (fixed for the FLUX text encoder).
#[derive(Debug, Clone, Copy)]
pub struct ClipConfig {
    pub layers: usize,  // 12
    pub hidden: usize,  // 768
    pub ffn: usize,     // 3072
    pub heads: usize,   // 12  (head_dim 64)
    pub max_pos: usize, // 77
    pub vocab: usize,   // 49408
    pub ln_eps: f32,    // 1e-5
}

impl Default for ClipConfig {
    fn default() -> Self {
        ClipConfig {
            layers: 12,
            hidden: 768,
            ffn: 3072,
            heads: 12,
            max_pos: 77,
            vocab: 49408,
            ln_eps: 1e-5,
        }
    }
}

/// One CLIP layer's resident GPU weights (bf16 matmuls + f32 norms/biases).
struct ClipLayer {
    ln1_w: wgpu::Buffer,
    ln1_b: wgpu::Buffer,
    q_w: wgpu::Buffer,
    q_b: wgpu::Buffer,
    k_w: wgpu::Buffer,
    k_b: wgpu::Buffer,
    v_w: wgpu::Buffer,
    v_b: wgpu::Buffer,
    o_w: wgpu::Buffer,
    o_b: wgpu::Buffer,
    ln2_w: wgpu::Buffer,
    ln2_b: wgpu::Buffer,
    fc1_w: wgpu::Buffer,
    fc1_b: wgpu::Buffer,
    fc2_w: wgpu::Buffer,
    fc2_b: wgpu::Buffer,
}

pub struct GpuClipText {
    ctx: Arc<GpuContext>,
    cfg: ClipConfig,
    /// Token embedding table [vocab, hidden] kept on the CPU (tiny per-prompt gather is input
    /// prep, not a forward op) + position embedding [max_pos, hidden].
    token_embd: Vec<f32>,
    pos_embd: Vec<f32>,
    layers: Vec<ClipLayer>,
    final_ln_w: wgpu::Buffer,
    final_ln_b: wgpu::Buffer,
    // kernels
    matmul: ComputeKernel,
    layernorm: ComputeKernel,
    add_bias: ComputeKernel,
    attn: ComputeKernel,
    gelu: ComputeKernel,
    add: ComputeKernel,
}

impl GpuClipText {
    /// Load CLIP-L from its safetensors file and build the GPU kernels.
    pub fn load(ctx: &Arc<GpuContext>, path: &std::path::Path, cfg: ClipConfig) -> Result<Self> {
        let w = read_weights(&[path])?;
        let get = |n: &str| -> Result<Vec<f32>> {
            w.get_f32(n)
                .ok_or_else(|| ArfError::model_load(format!("clip: missing {n}")))
        };
        let bf16 = |name: &str, d: &[f32]| ctx.storage_init_bf16(name, d);
        let f32b = |name: &str, d: &[f32]| ctx.storage_init(name, d);

        let layers = (0..cfg.layers)
            .map(|i| {
                let p = format!("text_model.encoder.layers.{i}");
                Ok(ClipLayer {
                    ln1_w: f32b("c.ln1w", &get(&format!("{p}.layer_norm1.weight"))?),
                    ln1_b: f32b("c.ln1b", &get(&format!("{p}.layer_norm1.bias"))?),
                    q_w: bf16("c.qw", &get(&format!("{p}.self_attn.q_proj.weight"))?),
                    q_b: f32b("c.qb", &get(&format!("{p}.self_attn.q_proj.bias"))?),
                    k_w: bf16("c.kw", &get(&format!("{p}.self_attn.k_proj.weight"))?),
                    k_b: f32b("c.kb", &get(&format!("{p}.self_attn.k_proj.bias"))?),
                    v_w: bf16("c.vw", &get(&format!("{p}.self_attn.v_proj.weight"))?),
                    v_b: f32b("c.vb", &get(&format!("{p}.self_attn.v_proj.bias"))?),
                    o_w: bf16("c.ow", &get(&format!("{p}.self_attn.out_proj.weight"))?),
                    o_b: f32b("c.ob", &get(&format!("{p}.self_attn.out_proj.bias"))?),
                    ln2_w: f32b("c.ln2w", &get(&format!("{p}.layer_norm2.weight"))?),
                    ln2_b: f32b("c.ln2b", &get(&format!("{p}.layer_norm2.bias"))?),
                    fc1_w: bf16("c.fc1w", &get(&format!("{p}.mlp.fc1.weight"))?),
                    fc1_b: f32b("c.fc1b", &get(&format!("{p}.mlp.fc1.bias"))?),
                    fc2_w: bf16("c.fc2w", &get(&format!("{p}.mlp.fc2.weight"))?),
                    fc2_b: f32b("c.fc2b", &get(&format!("{p}.mlp.fc2.bias"))?),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(GpuClipText {
            ctx: ctx.clone(),
            cfg,
            token_embd: get("text_model.embeddings.token_embedding.weight")?,
            pos_embd: get("text_model.embeddings.position_embedding.weight")?,
            layers,
            final_ln_w: f32b("c.flnw", &get("text_model.final_layer_norm.weight")?),
            final_ln_b: f32b("c.flnb", &get("text_model.final_layer_norm.bias")?),
            matmul: ComputeKernel::new(
                ctx,
                "cmatmul",
                include_str!("../../shaders/wgsl/matmul.wgsl"),
            ),
            layernorm: ComputeKernel::new(
                ctx,
                "cln",
                include_str!("../../shaders/wgsl/layernorm_bias.wgsl"),
            ),
            add_bias: ComputeKernel::new(
                ctx,
                "cab",
                include_str!("../../shaders/wgsl/add_bias.wgsl"),
            ),
            attn: ComputeKernel::new(
                ctx,
                "cattn",
                include_str!("../../shaders/wgsl/text_attn.wgsl"),
            ),
            gelu: ComputeKernel::new(
                ctx,
                "cqgelu",
                include_str!("../../shaders/wgsl/quick_gelu.wgsl"),
            ),
            add: ComputeKernel::new(ctx, "cadd", include_str!("../../shaders/wgsl/add.wgsl")),
        })
    }

    /// Encode token ids → the pooled `[hidden]` CLIP embedding (hidden state at `eos_pos`,
    /// which for CLIP is the highest token id = the EOS/end-of-text marker, i.e. the last
    /// non-pad position). `tokens` length = seq (≤ max_pos).
    pub fn encode(&self, tokens: &[u32], eos_pos: usize) -> Vec<f32> {
        let c = &self.cfg;
        let n = tokens.len();
        let h = c.hidden;
        let ctx = &self.ctx;

        // Initial hidden = token_embd[id] + pos_embd[pos]  (gather is input prep, not compute).
        let mut hidden0 = vec![0.0f32; n * h];
        for (pos, &id) in tokens.iter().enumerate() {
            let te = &self.token_embd[id as usize * h..(id as usize + 1) * h];
            let pe = &self.pos_embd[pos * h..(pos + 1) * h];
            for j in 0..h {
                hidden0[pos * h + j] = te[j] + pe[j];
            }
        }

        let hidden = ctx.storage_init("c.hidden", &hidden0);
        let norm = ctx.storage_init("c.norm", &vec![0.0f32; n * h]);
        let q = ctx.storage_init("c.q", &vec![0.0f32; n * h]);
        let k = ctx.storage_init("c.k", &vec![0.0f32; n * h]);
        let v = ctx.storage_init("c.v", &vec![0.0f32; n * h]);
        let attn = ctx.storage_init("c.attn", &vec![0.0f32; n * h]);
        let proj = ctx.storage_init("c.proj", &vec![0.0f32; n * h]);
        let up = ctx.storage_init("c.up", &vec![0.0f32; n * c.ffn]);
        let dummy_bias = ctx.storage_init("c.nobias", &[0.0f32]);

        let ln_dims = ctx.uniform_of("cln", &[n as u32, h as u32, c.ln_eps.to_bits(), 0u32]);
        let mm =
            |kk: usize, nn: usize| ctx.uniform_of("cmm", &[n as u32, kk as u32, nn as u32, 0u32]);
        let bias_dims = |cols: usize| ctx.uniform_of("cba", &[n as u32, cols as u32, 0u32, 0u32]);
        let act_dims = |len: usize| ctx.uniform_of("cact", &[len as u32, 0u32, 0u32, 0u32]);
        let add_dims = ctx.uniform_of("cadd", &[(n * h) as u32, 0u32, 0u32, 0u32]);
        // causal attention, no bias, scale = 1/sqrt(head_dim).
        let head_dim = h / c.heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let attn_dims = ctx.uniform_of(
            "cattn",
            &[
                n as u32,
                h as u32,
                c.heads as u32,
                head_dim as u32,
                1u32,
                0u32,
                scale.to_bits(),
                0u32,
            ],
        );
        let mm_wg = |nn: usize| [(n as u32).div_ceil(16), (nn as u32).div_ceil(16), 1u32];

        for l in &self.layers {
            // attn block: norm = LN1(hidden); q/k/v = norm·W+b; attn = causal MHA; proj = attn·Wo+bo; hidden += proj
            self.layernorm.dispatch(
                "ln1",
                &[&hidden, &l.ln1_w, &l.ln1_b, &norm, &ln_dims],
                [n as u32, 1, 1],
            );
            for (w, b, dst) in [
                (&l.q_w, &l.q_b, &q),
                (&l.k_w, &l.k_b, &k),
                (&l.v_w, &l.v_b, &v),
            ] {
                let md = mm(h, h);
                self.matmul.dispatch("qkv", &[&norm, w, dst, &md], mm_wg(h));
                let bd = bias_dims(h);
                self.add_bias.dispatch("qkvb", &[dst, b, &bd], wg(n * h));
            }
            self.attn.dispatch(
                "attn",
                &[&q, &k, &v, &attn, &dummy_bias, &attn_dims],
                [n as u32, 1, 1],
            );
            let md = mm(h, h);
            self.matmul
                .dispatch("o", &[&attn, &l.o_w, &proj, &md], mm_wg(h));
            let bd = bias_dims(h);
            self.add_bias
                .dispatch("ob", &[&proj, &l.o_b, &bd], wg(n * h));
            self.add
                .dispatch("res1", &[&hidden, &proj, &add_dims], wg(n * h));

            // mlp block: norm = LN2(hidden); up = norm·fc1+b; quick_gelu(up); proj = up·fc2+b; hidden += proj
            self.layernorm.dispatch(
                "ln2",
                &[&hidden, &l.ln2_w, &l.ln2_b, &norm, &ln_dims],
                [n as u32, 1, 1],
            );
            let mdu = mm(h, c.ffn);
            self.matmul
                .dispatch("fc1", &[&norm, &l.fc1_w, &up, &mdu], mm_wg(c.ffn));
            let bdu = bias_dims(c.ffn);
            self.add_bias
                .dispatch("fc1b", &[&up, &l.fc1_b, &bdu], wg(n * c.ffn));
            let gd = act_dims(n * c.ffn);
            self.gelu.dispatch("qgelu", &[&up, &gd], wg(n * c.ffn));
            let mdd = mm(c.ffn, h);
            self.matmul
                .dispatch("fc2", &[&up, &l.fc2_w, &proj, &mdd], mm_wg(h));
            let bd2 = bias_dims(h);
            self.add_bias
                .dispatch("fc2b", &[&proj, &l.fc2_b, &bd2], wg(n * h));
            self.add
                .dispatch("res2", &[&hidden, &proj, &add_dims], wg(n * h));
        }
        // final LayerNorm, then read back only the eos_pos row as the pooled embedding.
        self.layernorm.dispatch(
            "fln",
            &[&hidden, &self.final_ln_w, &self.final_ln_b, &norm, &ln_dims],
            [n as u32, 1, 1],
        );
        let all = ctx.read_f32(&norm, n * h);
        all[eos_pos * h..(eos_pos + 1) * h].to_vec()
    }
}
