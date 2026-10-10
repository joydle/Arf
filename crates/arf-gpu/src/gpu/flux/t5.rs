//! GPU T5-XXL encoder (FLUX's sequence-conditioning text encoder). Runs the 24-layer T5
//! encoder entirely on the GPU and returns the `[seq, 4096]` sequence embedding that becomes
//! FLUX's text stream. T5 specifics vs a vanilla encoder: RMSNorm (no bias, plain weight),
//! NO attention 1/√d scaling (folded into the weights → scale 1.0), gated-GELU FFN
//! (`down(gelu(gate(x)) * up(x))`), and a learned relative-position bias added to the
//! attention logits (computed from the layer-0 `attn_rel_b` table, shared across all layers).
//!
//! Self-contained in the `GpuVisionEncoder` mould: owns its `ComputeKernel`s, uploads resident
//! bf16 matmul weights (read by raw ggml name via `get_f32_raw`) + f32 norms, reuses the shared
//! kernels (`matmul`, `rmsnorm`, `text_attn`, `gelu_tanh`, `add`) + an elementwise `mul`. All
//! compute on Metal; the CPU only gathers token embeddings + builds the position-only rel-bias.

use super::wg;
use std::sync::Arc;

use arf_core::model::gguf::LazyGguf;
use arf_core::{ArfError, Result};

use crate::gpu::{ComputeKernel, GpuContext};

#[derive(Debug, Clone, Copy)]
pub struct T5Config {
    pub layers: usize,       // 24
    pub hidden: usize,       // 4096
    pub ffn: usize,          // 10240
    pub heads: usize,        // 64 (head_dim 64)
    pub rel_buckets: usize,  // 32
    pub rel_max_dist: usize, // 128
    pub rms_eps: f32,        // 1e-6
}

impl Default for T5Config {
    fn default() -> Self {
        T5Config {
            layers: 24,
            hidden: 4096,
            ffn: 10240,
            heads: 64,
            rel_buckets: 32,
            rel_max_dist: 128,
            rms_eps: 1e-6,
        }
    }
}

/// A T5 GEMM weight resident on the GPU: native **Q8_0** (i8 codes + per-32-block f32 scale,
/// dequant in-kernel) for the Q8_0 GGUF tensors — ~5 GB vs ~9.5 GB bf16 AND faster (4× fewer
/// weight bytes read) — or bf16 for any non-Q8_0 tensor.
enum T5W {
    Bf16 {
        w: wgpu::Buffer,
    },
    Q8_0 {
        q: wgpu::Buffer,
        dscale: wgpu::Buffer,
    },
}

struct T5Layer {
    attn_norm: wgpu::Buffer,
    q: T5W,
    k: T5W,
    v: T5W,
    o: T5W,
    ffn_norm: wgpu::Buffer,
    gate: T5W,
    up: T5W,
    down: T5W,
}

pub struct GpuT5 {
    ctx: Arc<GpuContext>,
    cfg: T5Config,
    token_embd: Vec<f32>,  // [vocab, hidden] — per-prompt gather is input prep
    rel_b_table: Vec<f32>, // [heads, rel_buckets] from layer-0 attn_rel_b (ggml [buckets, heads])
    layers: Vec<T5Layer>,
    output_norm: wgpu::Buffer,
    // kernels
    matmul: ComputeKernel,
    /// Cooperative-matrix Q8_0 GEMM (dequant-in-kernel) for the Q8_0-resident weights. T5 always
    /// runs at m = seq = 256 (≥8), so this covers every T5 GEMM; parity-gated to matmul_nt_q8_0
    /// (examples/coop_q8_0_probe). `None` without coop-matrix → those devices keep weights bf16.
    matmul_coop_q8_0: Option<ComputeKernel>,
    rmsnorm: ComputeKernel,
    attn: ComputeKernel,
    gelu: ComputeKernel,
    mul: ComputeKernel,
    add: ComputeKernel,
}

impl GpuT5 {
    /// Load T5-XXL from its GGUF and build the GPU kernels.
    pub fn load(ctx: &Arc<GpuContext>, path: &std::path::Path, cfg: T5Config) -> Result<Self> {
        // Raw-name open: T5's `t5encoder` arch is foreign to the text indexer, so we skip it
        // and read every tensor by its ggml name (get_f32_raw). Composable — no T5 knowledge
        // leaks into the llama/gemma loader.
        let g = LazyGguf::open_raw(path)?;
        let raw = |n: &str| -> Result<Vec<f32>> {
            g.get_f32_raw(n)
                .ok_or_else(|| ArfError::model_load(format!("t5: missing {n}")))
        };
        let f32b = |name: &str, d: &[f32]| ctx.storage_init(name, d);
        // Q8_0 dequant-in-kernel needs the coop-matrix GEMM (T5 always runs m=256≥8); without it,
        // keep weights bf16. ARF_FLUX_FORCE_BF16 forces the legacy path (A/B parity).
        let use_q8 = ctx.has_coop() && std::env::var("ARF_FLUX_FORCE_BF16").is_err();
        // Load a T5 GEMM weight: native Q8_0 transcode (no f32 materialization) when the GGUF
        // stores it as Q8_0 and coop is available, splitting the raw 34-byte block into aligned
        // GPU buffers — q (i8 codes, 4/u32) + dscale (one f32 per 32-block). Else bf16.
        let weight = |gname: &str| -> Result<T5W> {
            if use_q8 {
                if let Some((blocks, rows, cols)) = g.get_q8_0_blocks_raw(gname) {
                    let bpr = cols / 32; // blocks per row
                    let mut codes_u32 = Vec::with_capacity(rows * cols / 4);
                    let mut dscale = Vec::with_capacity(rows * bpr);
                    // pack i8 codes 4/u32; collect one f32 scale per block.
                    let mut pending: [u8; 4] = [0; 4];
                    let mut pc = 0usize;
                    for r in 0..rows {
                        for blk in 0..bpr {
                            let p = (r * bpr + blk) * 34;
                            let dh = u16::from_le_bytes([blocks[p], blocks[p + 1]]);
                            dscale.push(arf_core::model::gguf::f16_to_f32(dh));
                            for t in 0..32 {
                                pending[pc] = blocks[p + 2 + t];
                                pc += 1;
                                if pc == 4 {
                                    codes_u32.push(
                                        (pending[0] as u32)
                                            | ((pending[1] as u32) << 8)
                                            | ((pending[2] as u32) << 16)
                                            | ((pending[3] as u32) << 24),
                                    );
                                    pc = 0;
                                }
                            }
                        }
                    }
                    return Ok(T5W::Q8_0 {
                        q: ctx.storage_init_u32_tracked("t5.q.codes", &codes_u32),
                        dscale: ctx.storage_init("t5.q.dscale", &dscale),
                    });
                }
            }
            Ok(T5W::Bf16 {
                w: ctx.storage_init_bf16("t5.w", &raw(gname)?),
            })
        };

        let layers = (0..cfg.layers)
            .map(|i| {
                let p = format!("enc.blk.{i}");
                Ok(T5Layer {
                    attn_norm: f32b("t.an", &raw(&format!("{p}.attn_norm.weight"))?),
                    q: weight(&format!("{p}.attn_q.weight"))?,
                    k: weight(&format!("{p}.attn_k.weight"))?,
                    v: weight(&format!("{p}.attn_v.weight"))?,
                    o: weight(&format!("{p}.attn_o.weight"))?,
                    ffn_norm: f32b("t.fn", &raw(&format!("{p}.ffn_norm.weight"))?),
                    gate: weight(&format!("{p}.ffn_gate.weight"))?,
                    up: weight(&format!("{p}.ffn_up.weight"))?,
                    down: weight(&format!("{p}.ffn_down.weight"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(GpuT5 {
            ctx: ctx.clone(),
            cfg,
            token_embd: raw("token_embd.weight")?,
            // ggml stores attn_rel_b as [ne0=buckets=32, ne1=heads=64] → raw flat is
            // buckets-contiguous = [heads][buckets] row-major, which is what we index below.
            rel_b_table: raw("enc.blk.0.attn_rel_b.weight")?,
            layers,
            output_norm: f32b("t.on", &raw("enc.output_norm.weight")?),
            matmul: ComputeKernel::new(
                ctx,
                "tmatmul",
                include_str!("../../shaders/wgsl/matmul.wgsl"),
            ),
            matmul_coop_q8_0: ctx.has_coop().then(|| {
                ComputeKernel::new(
                    ctx,
                    "tmmcq8",
                    include_str!("../../shaders/wgsl/matmul_coop_q8_0.wgsl"),
                )
            }),
            rmsnorm: ComputeKernel::new(
                ctx,
                "trms",
                include_str!("../../shaders/wgsl/rmsnorm.wgsl"),
            ),
            attn: ComputeKernel::new(
                ctx,
                "tattn",
                include_str!("../../shaders/wgsl/text_attn.wgsl"),
            ),
            gelu: ComputeKernel::new(
                ctx,
                "tgelu",
                include_str!("../../shaders/wgsl/gelu_tanh.wgsl"),
            ),
            mul: ComputeKernel::new(ctx, "tmul", include_str!("../../shaders/wgsl/mul.wgsl")),
            add: ComputeKernel::new(ctx, "tadd", include_str!("../../shaders/wgsl/add.wgsl")),
        })
    }

    /// T5's bidirectional relative-position bucket (num_buckets=32, max_distance=128). Maps a
    /// signed relative position (key - query) to a bucket index. Half the buckets are for
    /// "to the left/right" sign, then exact for small distances + log-spaced for large.
    fn rel_bucket(&self, rel: i32) -> usize {
        let nb = self.cfg.rel_buckets as i32;
        let max_d = self.cfg.rel_max_dist as i32;
        let mut ret = 0i32;
        let num_buckets = nb / 2; // bidirectional: split sign into the top bit
        if rel > 0 {
            ret += num_buckets;
        }
        let n = rel.abs();
        let max_exact = num_buckets / 2;
        if n < max_exact {
            ret += n;
        } else {
            // log-spaced bucket for larger distances, clamped to the last bucket.
            let v = (max_exact as f32)
                + ((n as f32 / max_exact as f32).ln() / (max_d as f32 / max_exact as f32).ln()
                    * (num_buckets - max_exact) as f32);
            ret += (v as i32).min(num_buckets - 1);
        }
        ret as usize
    }

    /// Build the `[heads, seq, seq]` additive attention bias from the rel-bucket table.
    /// (A fixed function of positions — input prep, not model compute.)
    fn build_rel_bias(&self, seq: usize) -> Vec<f32> {
        let h = self.cfg.heads;
        let mut bias = vec![0.0f32; h * seq * seq];
        for q in 0..seq {
            for k in 0..seq {
                let bucket = self.rel_bucket(k as i32 - q as i32);
                for head in 0..h {
                    // rel_b_table is HF's relative_attention_bias embedding [buckets, heads]
                    // (ggml stores attn_rel_b buckets-major) → index [bucket * heads + head].
                    bias[(head * seq + q) * seq + k] = self.rel_b_table[bucket * h + head];
                }
            }
        }
        bias
    }

    /// Encode token ids → `[seq, hidden]` sequence embedding (FLUX's text stream).
    pub fn encode(&self, tokens: &[u32]) -> Vec<f32> {
        let c = &self.cfg;
        let n = tokens.len();
        let h = c.hidden;
        let f = c.ffn;
        let ctx = &self.ctx;

        // Initial hidden = token_embd[id]  (T5 has NO additive position embedding — position
        // info enters only via the relative bias). Gather = input prep.
        let mut hidden0 = vec![0.0f32; n * h];
        for (pos, &id) in tokens.iter().enumerate() {
            hidden0[pos * h..(pos + 1) * h]
                .copy_from_slice(&self.token_embd[id as usize * h..(id as usize + 1) * h]);
        }

        let probe = std::env::var("ARF_T5_SUM").is_ok();
        let psum = |tag: &str, b: &wgpu::Buffer, len: usize| {
            if probe {
                let v = ctx.read_f32(b, len);
                let s: f64 = v.iter().map(|&x| x as f64).sum();
                eprintln!("[t5sum] {tag:24} sum={s:.4} len={len}");
            }
        };
        if probe {
            let s: f64 = hidden0.iter().map(|&x| x as f64).sum();
            eprintln!(
                "[t5sum] {:24} sum={s:.4} len={}",
                "embed (hidden0)",
                hidden0.len()
            );
        }

        let hidden = ctx.storage_init("t.hidden", &hidden0);
        let norm = ctx.storage_init("t.norm", &vec![0.0f32; n * h]);
        let q = ctx.storage_init("t.q", &vec![0.0f32; n * h]);
        let k = ctx.storage_init("t.k", &vec![0.0f32; n * h]);
        let v = ctx.storage_init("t.v", &vec![0.0f32; n * h]);
        let attn = ctx.storage_init("t.attn", &vec![0.0f32; n * h]);
        let proj = ctx.storage_init("t.proj", &vec![0.0f32; n * h]);
        let gate = ctx.storage_init("t.gate", &vec![0.0f32; n * f]);
        let up = ctx.storage_init("t.up", &vec![0.0f32; n * f]);
        let rel_bias = ctx.storage_init("t.relb", &self.build_rel_bias(n));

        let rms_dims = ctx.uniform_of("trms", &[n as u32, h as u32, c.rms_eps.to_bits(), 0u32]);
        let mm =
            |kk: usize, nn: usize| ctx.uniform_of("tmm", &[n as u32, kk as u32, nn as u32, 0u32]);
        let act_dims = |len: usize| ctx.uniform_of("tact", &[len as u32, 0u32, 0u32, 0u32]);
        let add_h = ctx.uniform_of("tadd", &[(n * h) as u32, 0u32, 0u32, 0u32]);
        let mul_f = ctx.uniform_of("tmul", &[(n * f) as u32, 0u32, 0u32, 0u32]);
        // full attention, WITH rel-bias, scale = 1.0 (T5 folds 1/√d into the weights).
        let head_dim = h / c.heads;
        let attn_dims = ctx.uniform_of(
            "tattn",
            &[
                n as u32,
                h as u32,
                c.heads as u32,
                head_dim as u32,
                0u32,
                1u32,
                1.0f32.to_bits(),
                0u32,
            ],
        );
        let mm_wg = |nn: usize| [(n as u32).div_ceil(16), (nn as u32).div_ceil(16), 1u32];
        let coop_wg = |nn: usize| [(n as u32).div_ceil(8), (nn as u32).div_ceil(8), 1u32];
        // out[n,nn] = in[n,kk]·Wᵀ. Q8_0 weights route through the coop dequant-in-kernel GEMM
        // (T5 m=n=256≥8, coop present); bf16 weights keep the scalar matmul. Both = C=A·Bᵀ,
        // parity-proven (matmul_nt_q8_0 / bf16 oracle).
        let lin = |inp: &wgpu::Buffer, w: &T5W, out: &wgpu::Buffer, kk: usize, nn: usize| match (
            w,
            &self.matmul_coop_q8_0,
        ) {
            (T5W::Q8_0 { q, dscale }, Some(coop)) => {
                coop.dispatch("t.q8", &[inp, q, out, &mm(kk, nn), dscale], coop_wg(nn))
            }
            (T5W::Q8_0 { .. }, None) => unreachable!("Q8_0 weights only built when coop present"),
            (T5W::Bf16 { w }, _) => {
                self.matmul
                    .dispatch("t.bf", &[inp, w, out, &mm(kk, nn)], mm_wg(nn))
            }
        };

        for (li, l) in self.layers.iter().enumerate() {
            let p0 = probe && li == 0;
            // attn: norm=RMS(hidden); q/k/v=norm·W (no bias); attn=full MHA+relbias; proj=attn·Wo; hidden+=proj
            self.rmsnorm.dispatch(
                "an",
                &[&hidden, &l.attn_norm, &norm, &rms_dims],
                [n as u32, 1, 1],
            );
            if p0 {
                psum("L0 attn_norm", &norm, n * h);
            }
            for (w, dst) in [(&l.q, &q), (&l.k, &k), (&l.v, &v)] {
                lin(&norm, w, dst, h, h);
            }
            if p0 {
                psum("L0 q", &q, n * h);
                psum("L0 k", &k, n * h);
                psum("L0 v", &v, n * h);
            }
            self.attn.dispatch(
                "attn",
                &[&q, &k, &v, &attn, &rel_bias, &attn_dims],
                [n as u32, 1, 1],
            );
            if p0 {
                psum("L0 attn", &attn, n * h);
            }
            lin(&attn, &l.o, &proj, h, h);
            self.add
                .dispatch("res1", &[&hidden, &proj, &add_h], wg(n * h));
            if p0 {
                psum("L0 after_attn_res", &hidden, n * h);
            }

            // ffn (gated-GELU): norm=RMS(hidden); gate=norm·Wg; up=norm·Wu; gate=gelu(gate)*up; proj=gate·Wd; hidden+=proj
            self.rmsnorm.dispatch(
                "fn",
                &[&hidden, &l.ffn_norm, &norm, &rms_dims],
                [n as u32, 1, 1],
            );
            lin(&norm, &l.gate, &gate, h, f);
            lin(&norm, &l.up, &up, h, f);
            let gd = act_dims(n * f);
            self.gelu.dispatch("gelu", &[&gate, &gd], wg(n * f));
            self.mul
                .dispatch("gate_up", &[&gate, &up, &mul_f], wg(n * f)); // gate = gelu(gate)*up
            lin(&gate, &l.down, &proj, f, h);
            self.add
                .dispatch("res2", &[&hidden, &proj, &add_h], wg(n * h));
        }
        // final RMSNorm → the [seq, hidden] sequence embedding.
        self.rmsnorm.dispatch(
            "on",
            &[&hidden, &self.output_norm, &norm, &rms_dims],
            [n as u32, 1, 1],
        );
        ctx.read_f32(&norm, n * h)
    }
}
