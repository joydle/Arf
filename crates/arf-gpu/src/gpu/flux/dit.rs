//! GPU FLUX MMDiT denoiser (the rectified-flow transformer). Runs one denoising step entirely
//! on Metal: img latent + T5 sequence + CLIP-pooled + timestep → velocity. 19 double-stream
//! blocks (separate img/txt weights, joint attention) + 38 single-stream blocks (merged, fused
//! qkv+mlp) + final adaLN layer. Self-contained in the `GpuVisionEncoder` mould: owns its
//! `ComputeKernel`s, uploads resident bf16 matmul weights + f32 norms, reuses the shared kernel
//! library (matmul, qk_norm, gelu_tanh, add, mul) + the FLUX kernels (dit_attn, flux_rope_3axis,
//! adaln_modulate, layernorm_plain, silu).
//!
//! All compute on Metal; the CPU only patchifies the input latent, builds the 3-axis RoPE table
//! + sinusoidal timestep embedding (fixed functions of position/time = input prep), and reads
//! back the final velocity.

use super::wg;
use std::sync::Arc;

use arf_core::model::gguf::LazyGguf;
use arf_core::tensor::Q4KSMatrix;
use arf_core::{ArfError, Result};

use crate::gpu::{ComputeKernel, GpuContext};

#[derive(Debug, Clone, Copy)]
pub struct DitConfig {
    pub hidden: usize,        // 3072
    pub heads: usize,         // 24  (head_dim 128)
    pub double_blocks: usize, // 19
    pub single_blocks: usize, // 38
    pub mlp: usize,           // 12288
    pub txt_len: usize,       // 256 (T5 sequence the pipeline pads/truncates to)
    pub patch_grid: usize,    // 64  (1024/16: latent 128/8=16 → /2 patch → 64×64 = 4096 tokens)
    pub theta: f32,           // 10000
}

impl DitConfig {
    /// FLUX.1-schnell defaults for a 1024×1024 image (4096 image tokens).
    pub fn schnell_1024() -> Self {
        Self::schnell(64, 256)
    }
    /// FLUX.1-schnell with a given patch grid (image = grid² tokens) + T5 sequence length —
    /// lets the parity test run a small problem (grid 8, txt 16) on the same weights.
    pub fn schnell(patch_grid: usize, txt_len: usize) -> Self {
        DitConfig {
            hidden: 3072,
            heads: 24,
            double_blocks: 19,
            single_blocks: 38,
            mlp: 12288,
            txt_len,
            patch_grid,
            theta: 10000.0,
        }
    }
    pub fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }
    pub fn img_tokens(&self) -> usize {
        self.patch_grid * self.patch_grid
    }
    pub fn seq(&self) -> usize {
        self.txt_len + self.img_tokens()
    }
}

/// The resident weight of a DiT linear: either bf16-packed (the lossless fallback for the
/// handful of non-Q4_K tensors) or native **Q4_K_S** (codes + u8/i8 sub-scales + per-super
/// `dd` = `[d,dmin]`), dequantized inside the GEMM kernel. Keeping the 304 big GEMM weights
/// Q4_K_S-resident is ~6.7 GB vs ~24 GB bf16 — the F1 memory win (and FASTER: bandwidth-bound
/// GEMMs read 4× fewer weight bytes).
enum LinW {
    Bf16 {
        w: wgpu::Buffer,
    },
    Q4ks {
        codes: wgpu::Buffer,
        scales: wgpu::Buffer,
        mins: wgpu::Buffer,
        dd: wgpu::Buffer,
    },
}

/// A linear weight (bf16 or Q4_K_S) + optional bias (f32) resident on the GPU.
struct Lin {
    w: LinW,
    b: Option<wgpu::Buffer>,
}

struct DoubleBlock {
    img_mod: Lin,
    txt_mod: Lin, // 3072 -> 18432 (+bias)
    img_qkv: Lin,
    txt_qkv: Lin, // 3072 -> 9216
    img_qn: wgpu::Buffer,
    img_kn: wgpu::Buffer, // QK-norm .scale [head_dim]
    txt_qn: wgpu::Buffer,
    txt_kn: wgpu::Buffer,
    img_proj: Lin,
    txt_proj: Lin, // 3072 -> 3072
    img_mlp0: Lin,
    img_mlp2: Lin, // 3072->12288, 12288->3072
    txt_mlp0: Lin,
    txt_mlp2: Lin,
}

struct SingleBlock {
    mod_lin: Lin, // 3072 -> 9216 (3 chunks)
    linear1: Lin, // 3072 -> 21504 (fused qkv + mlp-up)
    linear2: Lin, // 15360 -> 3072 (fused out)
    qn: wgpu::Buffer,
    kn: wgpu::Buffer,
}

pub struct GpuDiT {
    ctx: Arc<GpuContext>,
    cfg: DitConfig,
    // embedders
    img_in: Lin,
    txt_in: Lin,
    time_in0: Lin,
    time_in1: Lin, // MLPEmbedder in/out
    vec_in0: Lin,
    vec_in1: Lin,
    doubles: Vec<DoubleBlock>,
    singles: Vec<SingleBlock>,
    final_adaln: Lin, // 3072 -> 6144
    final_lin: Lin,   // 3072 -> 64
    // kernels
    matmul: ComputeKernel,
    /// Cooperative-matrix GEMM (Apple simdgroup_matrix) — a drop-in for `matmul` (same Dims +
    /// bindings + C=A·Bᵀ bf16 contract) used for the large-m DiT GEMMs (joint seq ≥ 8 rows), where
    /// it runs the inner product on the matrix units (~3.5× the scalar tiled GEMM). `None` when the
    /// device lacks the feature → falls back to `matmul`. The dominant FLUX denoise cost.
    matmul_coop: Option<ComputeKernel>,
    /// Cooperative-matrix Q4_K_S GEMM — the quantized-resident analog of `matmul_coop`, with the
    /// two-level u8/i8→f32 dequant folded into the staged weight element. Used for the Q4_K_S DiT
    /// GEMMs at m≥8 (the dominant denoise cost). Parity-gated bit-tight to `matmul_nt_q4ks`
    /// (examples/coop_q4ks_probe). `None` when the device lacks coop-matrix.
    matmul_coop_q4ks: Option<ComputeKernel>,
    /// Batched Q4_K_S GEMV — the m<8 (or no-coop) fallback for Q4_K_S weights, same dequant
    /// contract as the coop kernel and the GEMV oracle.
    matmul_vec_q4ks: ComputeKernel,
    add_bias: ComputeKernel,
    ln: ComputeKernel, // layernorm_plain (affine-free)
    qknorm: ComputeKernel,
    // FLUX 3-axis RoPE. NOT shaders/wgsl/rope_interleaved.wgsl — that name was taken over by the
    // LLM decode path (single tensor, 4 bindings); this one is fused q/k with 5. See the shader.
    rope: ComputeKernel, // flux_rope_3axis
    attn: ComputeKernel, // dit_attn
    gelu: ComputeKernel,
    silu: ComputeKernel,
    add: ComputeKernel,
    adaln: ComputeKernel, // adaln_modulate (modulate + gate)
    slice: ComputeKernel, // slice_cols (deinterleave fused qkv/mlp column-blocks)
}

/// 2D-safe grid for ONE workgroup PER `rows` (no /256), splitting across x/y past the
/// 65535-per-dim limit (1024px qk-norm needs 196608 rows). The shader reconstructs the row as
/// `wid.y * num_workgroups.x + wid.x`, so `num_workgroups.x` MUST equal the x-extent here.
fn wg_rows(rows: usize) -> [u32; 3] {
    let r = rows as u32;
    const MAX: u32 = 65535;
    if r <= MAX {
        [r, 1, 1]
    } else {
        [MAX, r.div_ceil(MAX), 1]
    }
}

impl GpuDiT {
    pub fn load(ctx: &Arc<GpuContext>, path: &std::path::Path, cfg: DitConfig) -> Result<Self> {
        let g = LazyGguf::open_raw(path)?;
        let raw = |n: &str| -> Result<Vec<f32>> {
            g.get_f32_raw(n)
                .ok_or_else(|| ArfError::model_load(format!("dit: missing {n}")))
        };
        let f32b = |d: &[f32]| ctx.storage_init("dit.b", d);
        // Load a weight tensor: native Q4_K_S transcode (no f32 materialization → the memory win)
        // when the GGUF stores it as Q4_K, else dequant→bf16 for the handful of non-Q4K weights.
        // The Q4_K_S resident form is (codes u32, scales u8×4/word, mins i8×4/word, dd=[d,dmin]/super)
        // — the exact buffers matmul_coop_q4ks / matmul_vec_q4ks_batch read.
        // Escape hatch for A/B parity: ARF_FLUX_FORCE_BF16=1 forces the legacy bf16 path on
        // every weight (the pre-F1 behavior) so the new Q4_K_S path can be compared against it.
        let force_bf16 = std::env::var("ARF_FLUX_FORCE_BF16").is_ok();
        let weight = |name: &str| -> Result<LinW> {
            let wname = format!("{name}.weight");
            if let Some((blocks, rows, cols)) = g.get_q4k_blocks_raw(&wname).filter(|_| !force_bf16)
            {
                let q = Q4KSMatrix::from_ggml_q4k(blocks, rows, cols);
                // scales/mins are read as array<u32> (4/word); upload raw bytes reinterpreted.
                let pack_u8 = |b: &[u8]| -> Vec<u32> {
                    b.chunks(4)
                        .map(|c| {
                            let mut w = 0u32;
                            for (i, &x) in c.iter().enumerate() {
                                w |= (x as u32) << (8 * i);
                            }
                            w
                        })
                        .collect()
                };
                let mut dd = Vec::with_capacity(q.d().len() * 2);
                for (dv, dmv) in q.d().iter().zip(q.dmin()) {
                    dd.push(*dv);
                    dd.push(*dmv);
                }
                Ok(LinW::Q4ks {
                    // codes are the big buffer (6.66 GB across the DiT) → tracked upload so the
                    // backlog flushes periodically (avoids the silent-drop bug); a single
                    // end-of-load flush is too late at this scale.
                    codes: ctx.storage_init_u32_tracked("dit.q.codes", q.codes()),
                    scales: ctx.storage_init("dit.q.scales", &pack_u8(q.scales())),
                    mins: ctx.storage_init(
                        "dit.q.mins",
                        &pack_u8(&q.mins().iter().map(|&x| x as u8).collect::<Vec<u8>>()),
                    ),
                    dd: ctx.storage_init("dit.q.dd", &dd),
                })
            } else {
                Ok(LinW::Bf16 {
                    w: ctx.storage_init_bf16("dit.w", &raw(&wname)?),
                })
            }
        };
        // a Linear (weight + optional bias), reading ggml `{name}.weight` / `{name}.bias`.
        let lin = |name: &str, bias: bool| -> Result<Lin> {
            Ok(Lin {
                w: weight(name)?,
                b: if bias {
                    Some(f32b(&raw(&format!("{name}.bias"))?))
                } else {
                    None
                },
            })
        };

        let doubles = (0..cfg.double_blocks)
            .map(|i| {
                let p = format!("double_blocks.{i}");
                Ok(DoubleBlock {
                    img_mod: lin(&format!("{p}.img_mod.lin"), true)?,
                    txt_mod: lin(&format!("{p}.txt_mod.lin"), true)?,
                    img_qkv: lin(&format!("{p}.img_attn.qkv"), true)?,
                    txt_qkv: lin(&format!("{p}.txt_attn.qkv"), true)?,
                    img_qn: f32b(&raw(&format!("{p}.img_attn.norm.query_norm.scale"))?),
                    img_kn: f32b(&raw(&format!("{p}.img_attn.norm.key_norm.scale"))?),
                    txt_qn: f32b(&raw(&format!("{p}.txt_attn.norm.query_norm.scale"))?),
                    txt_kn: f32b(&raw(&format!("{p}.txt_attn.norm.key_norm.scale"))?),
                    img_proj: lin(&format!("{p}.img_attn.proj"), true)?,
                    txt_proj: lin(&format!("{p}.txt_attn.proj"), true)?,
                    img_mlp0: lin(&format!("{p}.img_mlp.0"), true)?,
                    img_mlp2: lin(&format!("{p}.img_mlp.2"), true)?,
                    txt_mlp0: lin(&format!("{p}.txt_mlp.0"), true)?,
                    txt_mlp2: lin(&format!("{p}.txt_mlp.2"), true)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let singles = (0..cfg.single_blocks)
            .map(|i| {
                let p = format!("single_blocks.{i}");
                Ok(SingleBlock {
                    mod_lin: lin(&format!("{p}.modulation.lin"), true)?,
                    linear1: lin(&format!("{p}.linear1"), true)?,
                    linear2: lin(&format!("{p}.linear2"), true)?,
                    qn: f32b(&raw(&format!("{p}.norm.query_norm.scale"))?),
                    kn: f32b(&raw(&format!("{p}.norm.key_norm.scale"))?),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // wgpu defers `create_buffer_init` upload copies and drops the backlog under the
        // multi-GB allocation churn of a 12B resident model (the silent-zero-weight bug).
        // `storage_init_*` auto-flushes past a byte threshold; flush once more here so the
        // tail of the uploads is committed before the first forward uses them.
        ctx.flush_uploads();

        Ok(GpuDiT {
            ctx: ctx.clone(),
            cfg,
            img_in: lin("img_in", true)?,
            txt_in: lin("txt_in", true)?,
            time_in0: lin("time_in.in_layer", true)?,
            time_in1: lin("time_in.out_layer", true)?,
            vec_in0: lin("vector_in.in_layer", true)?,
            vec_in1: lin("vector_in.out_layer", true)?,
            doubles,
            singles,
            final_adaln: lin("final_layer.adaLN_modulation.1", true)?,
            final_lin: lin("final_layer.linear", true)?,
            matmul: ComputeKernel::new(ctx, "dmm", include_str!("../../shaders/wgsl/matmul.wgsl")),
            matmul_coop: ctx.has_coop().then(|| {
                ComputeKernel::new(
                    ctx,
                    "dmmc",
                    include_str!("../../shaders/wgsl/matmul_coop.wgsl"),
                )
            }),
            matmul_coop_q4ks: ctx.has_coop().then(|| {
                ComputeKernel::new(
                    ctx,
                    "dmmcq",
                    include_str!("../../shaders/wgsl/matmul_coop_q4ks.wgsl"),
                )
            }),
            matmul_vec_q4ks: ComputeKernel::new(
                ctx,
                "dmvq",
                include_str!("../../shaders/wgsl/matmul_vec_q4ks_batch.wgsl"),
            ),
            add_bias: ComputeKernel::new(
                ctx,
                "dab",
                include_str!("../../shaders/wgsl/add_bias.wgsl"),
            ),
            ln: ComputeKernel::new(
                ctx,
                "dln",
                include_str!("../../shaders/wgsl/layernorm_plain.wgsl"),
            ),
            qknorm: ComputeKernel::new(
                ctx,
                "dqkn",
                include_str!("../../shaders/wgsl/qk_norm.wgsl"),
            ),
            rope: ComputeKernel::new(
                ctx,
                "drope",
                include_str!("../../shaders/wgsl/flux_rope_3axis.wgsl"),
            ),
            attn: ComputeKernel::new(
                ctx,
                "dattn",
                include_str!("../../shaders/wgsl/dit_attn.wgsl"),
            ),
            gelu: ComputeKernel::new(
                ctx,
                "dgelu",
                include_str!("../../shaders/wgsl/gelu_tanh.wgsl"),
            ),
            silu: ComputeKernel::new(ctx, "dsilu", include_str!("../../shaders/wgsl/silu.wgsl")),
            add: ComputeKernel::new(ctx, "dadd", include_str!("../../shaders/wgsl/add.wgsl")),
            adaln: ComputeKernel::new(
                ctx,
                "dadaln",
                include_str!("../../shaders/wgsl/adaln_modulate.wgsl"),
            ),
            slice: ComputeKernel::new(
                ctx,
                "dslice",
                include_str!("../../shaders/wgsl/slice_cols.wgsl"),
            ),
        })
    }

    /// Sinusoidal timestep embedding (BFL: t scaled ×1000, half=128, cat([cos, sin]),
    /// max_period 10000) → `[256]`. Host-built (a fixed function of t = input prep).
    fn timestep_embed(t: f32) -> Vec<f32> {
        let half = 128usize;
        let mut e = vec![0.0f32; 2 * half];
        let arg_scale = t * 1000.0;
        for i in 0..half {
            let freq = (-(10000f32.ln()) * i as f32 / half as f32).exp();
            let a = arg_scale * freq;
            e[i] = a.cos();
            e[half + i] = a.sin();
        }
        e
    }

    /// Build the `[seq, head_dim]` interleaved cos/sin RoPE tables. axes_dim [16,56,56]:
    /// head dims [0:16] = axis-0 (always pos 0 → identity), [16:72] = row, [72:128] = col.
    /// Each axis angle repeat-interleaved across its consecutive pair. txt tokens (first
    /// `txt_len`) have all-zero positions → identity everywhere.
    fn rope_tables(&self) -> (Vec<f32>, Vec<f32>) {
        let c = &self.cfg;
        let hd = c.head_dim();
        let seq = c.seq();
        let grid = c.patch_grid;
        let axes = [16usize, 56, 56];
        let mut cos = vec![1.0f32; seq * hd];
        let mut sin = vec![0.0f32; seq * hd];
        for tok in 0..seq {
            // position per axis: txt → (0,0,0); img token j → (0, row, col).
            let (p0, p1, p2) = if tok < c.txt_len {
                (0i32, 0i32, 0i32)
            } else {
                let j = tok - c.txt_len;
                (0, (j / grid) as i32, (j % grid) as i32)
            };
            let pos = [p0, p1, p2];
            let mut dim_off = 0usize;
            for (ax, &adim) in axes.iter().enumerate() {
                let pairs = adim / 2;
                for pair in 0..pairs {
                    // omega = 1/theta^(2*pair/adim); angle = pos * omega.
                    let omega = (c.theta).powf(-(2.0 * pair as f32) / adim as f32);
                    let ang = pos[ax] as f32 * omega;
                    let (cv, sv) = (ang.cos(), ang.sin());
                    let i0 = dim_off + 2 * pair;
                    cos[tok * hd + i0] = cv;
                    cos[tok * hd + i0 + 1] = cv;
                    sin[tok * hd + i0] = sv;
                    sin[tok * hd + i0 + 1] = sv;
                }
                dim_off += adim;
            }
        }
        (cos, sin)
    }

    /// One denoising forward: `img_latent` patchified `[img_tokens, 64]`, `t5_seq` raw T5
    /// `[txt_len, 4096]`, `clip_pooled` `[768]`, `timestep` scalar in `0,1`. Returns velocity
    /// `[img_tokens, 64]` (caller unpatchifies).
    pub fn forward(
        &self,
        img_patches: &[f32],
        t5_seq: &[f32],
        clip_pooled: &[f32],
        timestep: f32,
    ) -> Vec<f32> {
        let c = &self.cfg;
        let ctx = &self.ctx;
        let h = c.hidden;
        let nh = c.heads;
        let hd = c.head_dim();
        let it = c.img_tokens(); // 4096
        let tl = c.txt_len; // 256
        let seq = c.seq(); // 4352
        let mlp = c.mlp;
        let scale = 1.0f32 / (hd as f32).sqrt();

        // scratch (resident across the block loop). io = also a copy DST (qkv split / concat).
        let zeros = |n: usize| ctx.storage_init("dit.z", &vec![0.0f32; n]);
        let io = |n: usize| ctx.storage_io_f32("dit.io", n);
        let dummy = ctx.storage_init("dit.dummy", &[0.0f32]);

        // uniforms helpers
        let mm = |m: usize, k: usize, n: usize| {
            ctx.uniform_of("dmm", &[m as u32, k as u32, n as u32, 0u32])
        };
        let mm_wg = |m: usize, n: usize| [(m as u32).div_ceil(16), (n as u32).div_ceil(16), 1u32];
        let bias_d = |rows: usize, cols: usize| {
            ctx.uniform_of("dba", &[rows as u32, cols as u32, 0u32, 0u32])
        };
        let ln_d =
            |rows: usize| ctx.uniform_of("dln", &[rows as u32, h as u32, 1e-6f32.to_bits(), 0u32]);
        let act_d = |n: usize| ctx.uniform_of("dact", &[n as u32, 0u32, 0u32, 0u32]);

        // coop GEMM uses 8×8 output tiles (workgroup_size 8,8); scalar matmul uses 16×16.
        let coop_wg = |m: usize, n: usize| [(m as u32).div_ceil(8), (n as u32).div_ceil(8), 1u32];
        // Q4_K_S batched-GEMV fallback: one workgroup per output column (n groups), 64 lanes.
        let q4ks_wg = |n: usize| [n as u32, 1u32, 1u32];
        // Q4_K_S Dims carry row_off (0 — no chunking; the fallback only runs at m<8 ≤ MAXM).
        let mmq = |m: usize, k: usize, n: usize| {
            ctx.uniform_of("dmmq", &[m as u32, k as u32, n as u32, 0u32])
        };
        // linear: out[m,n] = in[m,k]·Wᵀ (+bias). W is bf16 OR Q4_K_S [n,k]. Q4_K_S routes through the
        // cooperative-matrix dequant-in-kernel GEMM at m≥8 (the denoise GEMMs, m = joint seq ≫ 8),
        // else the batched-GEMV fallback; bf16 weights keep the original coop/scalar path. All four
        // share the C=A·Bᵀ contract and are parity-proven to matmul_nt_q4ks / the bf16 oracle.
        let linear =
            |inp: &wgpu::Buffer, l: &Lin, out: &wgpu::Buffer, m: usize, k: usize, n: usize| {
                match &l.w {
                    LinW::Bf16 { w } => match (m >= 8, &self.matmul_coop) {
                        (true, Some(coop)) => {
                            coop.dispatch("linc", &[inp, w, out, &mm(m, k, n)], coop_wg(m, n))
                        }
                        _ => self
                            .matmul
                            .dispatch("lin", &[inp, w, out, &mm(m, k, n)], mm_wg(m, n)),
                    },
                    LinW::Q4ks {
                        codes,
                        scales,
                        mins,
                        dd,
                    } => match (m >= 8, &self.matmul_coop_q4ks) {
                        (true, Some(coop)) => coop.dispatch(
                            "lincq",
                            &[inp, codes, out, &mmq(m, k, n), scales, mins, dd],
                            coop_wg(m, n),
                        ),
                        _ => self.matmul_vec_q4ks.dispatch(
                            "linvq",
                            &[inp, codes, out, &mmq(m, k, n), scales, mins, dd],
                            q4ks_wg(n),
                        ),
                    },
                }
                if let Some(b) = &l.b {
                    self.add_bias
                        .dispatch("linb", &[out, b, &bias_d(m, n)], wg(m * n));
                }
            };

        let probe = std::env::var("ARF_DIT_SUM").is_ok();
        let psum = |tag: &str, b: &wgpu::Buffer, len: usize| {
            if probe {
                let v = ctx.read_f32(b, len);
                let s: f64 = v.iter().map(|&x| x as f64).sum();
                eprintln!("[ditsum] {tag:22} sum={s:.4} len={len}");
            }
        };

        // ---- input prep ----
        // img_in: [it,64]→[it,3072]; txt_in: [tl,4096]→[tl,3072]
        let img_raw = ctx.storage_init("dit.imgraw", img_patches);
        let img = zeros(it * h);
        linear(&img_raw, &self.img_in, &img, it, 64, h);
        let txt_raw = ctx.storage_init("dit.txtraw", t5_seq);
        let txt = zeros(tl * h);
        linear(&txt_raw, &self.txt_in, &txt, tl, 4096, h);

        // vec[1,3072] = time_in(timestep_emb) + vector_in(clip_pooled). (schnell: no guidance.)
        let temb = ctx.storage_init("dit.temb", &Self::timestep_embed(timestep)); // [256]
        let vt0 = zeros(h);
        linear(&temb, &self.time_in0, &vt0, 1, 256, h);
        self.silu.dispatch("tsilu", &[&vt0, &act_d(h)], wg(h));
        let vec = zeros(h);
        linear(&vt0, &self.time_in1, &vec, 1, h, h);
        let cp = ctx.storage_init("dit.cp", clip_pooled); // [768]
        let vv0 = zeros(h);
        linear(&cp, &self.vec_in0, &vv0, 1, 768, h);
        self.silu.dispatch("vsilu", &[&vv0, &act_d(h)], wg(h));
        let vvec = zeros(h);
        linear(&vv0, &self.vec_in1, &vvec, 1, h, h);
        self.add.dispatch("vadd", &[&vec, &vvec, &act_d(h)], wg(h)); // vec += vector_in(clip)
        psum("vec", &vec, h);
        psum("img_in", &img, it * h);
        psum("txt_in", &txt, tl * h);

        // rope tables [seq, hd]
        let (cos, sin) = self.rope_tables();
        let rcos = ctx.storage_init("dit.cos", &cos);
        let rsin = ctx.storage_init("dit.sin", &sin);

        // ---- scratch (resident across the block loop). io() = copy DST (qkv split / concat). ----
        let qj = io(seq * h); // joint q  [txt;img]
        let kj = io(seq * h); // joint k
        let vj = io(seq * h); // joint v
        let aout = zeros(seq * h); // attention out
        let mod_p = zeros(6 * h); // modulation params (≤6 chunks of [h])
        let norm = zeros(seq * h); // LN output
        let qkv = zeros(it.max(tl) * 3 * h); // per-stream qkv (rows ≤ max(it,tl))
        let qtmp = io(it.max(tl) * h); // per-stream q slice (for qk-norm before scatter)
        let ktmp = io(it.max(tl) * h);
        let proj = io(seq * h); // attn-proj / mlp-down scratch (copy dst)
        let up = io(seq * mlp); // mlp-up scratch (also a copy dst in single blocks)
        let lin1 = io(seq * (3 * h + mlp)); // single fused qkv+mlp (split via copy)

        // modulation: SiLU(vec)·Wmod → [1, nchunk*h] in mod_p.
        let modparams = |l: &Lin| {
            let sv = zeros(h);
            self.add.dispatch("mc", &[&sv, &vec, &act_d(h)], wg(h)); // sv = vec
            self.silu.dispatch("ms", &[&sv, &act_d(h)], wg(h));
            linear(&sv, l, &mod_p, 1, h, 6 * h);
        };
        let modulate = |buf: &wgpu::Buffer, rows: usize, sc: usize, sh: usize| {
            let u = ctx.uniform_of(
                "dm0",
                &[
                    rows as u32,
                    h as u32,
                    0u32,
                    (sc * h) as u32,
                    (sh * h) as u32,
                    0u32,
                    0u32,
                    0u32,
                ],
            );
            self.adaln
                .dispatch("mod", &[buf, &dummy, &mod_p, &u], wg(rows * h));
        };
        let gate_add = |acc: &wgpu::Buffer, y: &wgpu::Buffer, rows: usize, gc: usize| {
            let u = ctx.uniform_of(
                "dm1",
                &[
                    rows as u32,
                    h as u32,
                    1u32,
                    (gc * h) as u32,
                    0u32,
                    0u32,
                    0u32,
                    0u32,
                ],
            );
            self.adaln
                .dispatch("gate", &[acc, y, &mod_p, &u], wg(rows * h));
        };
        // Gather a column-block (`col_off..col_off+width`) of every row of a row-major
        // `[rows, stride]` matrix into `dst` as dense `[rows, width]`, written at row `dst_tok`.
        // The fused qkv/mlp matmul output is interleaved per token, so q/k/v are COLUMN slices,
        // not contiguous row-blocks — this deinterleaves them on the GPU.
        let slice_cols = |mat: &wgpu::Buffer,
                          dst: &wgpu::Buffer,
                          rows: usize,
                          stride: usize,
                          col_off: usize,
                          width: usize,
                          dst_tok: usize| {
            let u = ctx.uniform_of(
                "dsl",
                &[
                    rows as u32,
                    stride as u32,
                    col_off as u32,
                    width as u32,
                    dst_tok as u32,
                    0u32,
                    0u32,
                    0u32,
                ],
            );
            self.slice
                .dispatch("slice", &[mat, dst, &u], wg(rows * width));
        };
        // Inverse of `slice_cols`: scatter a dense `[rows, width]` into the column-block
        // `col_off..col_off+width` of every row of a `[rows, stride]` matrix (build a per-token
        // concatenation, e.g. linear2's [attn | gelu(mlp)] input).
        let place_cols = |mat: &wgpu::Buffer,
                          src: &wgpu::Buffer,
                          rows: usize,
                          stride: usize,
                          col_off: usize,
                          width: usize| {
            let u = ctx.uniform_of(
                "dsl",
                &[
                    rows as u32,
                    stride as u32,
                    col_off as u32,
                    width as u32,
                    0u32,
                    1u32,
                    0u32,
                    0u32,
                ],
            );
            self.slice
                .dispatch("place", &[mat, src, &u], wg(rows * width));
        };
        // per-stream QK-norm: deinterleave the [rows,3h] qkv into q/k (own weights) + v,
        // norm q & k, then scatter q,k,v into the joint buffers at token offset `dsttok`.
        let qknorm_scatter = |rows: usize, qn: &wgpu::Buffer, kn: &wgpu::Buffer, dsttok: usize| {
            // q = cols[0..h], k = cols[h..2h], v = cols[2h..3h] of each row of qkv[rows,3h].
            slice_cols(&qkv, &qtmp, rows, 3 * h, 0, h, 0);
            slice_cols(&qkv, &ktmp, rows, 3 * h, h, h, 0);
            let rws = (rows * nh) as u32;
            let qd = ctx.uniform_of("qkn", &[rws, rws, hd as u32, 1e-6f32.to_bits()]);
            self.qknorm.dispatch(
                "qkn",
                &[&qtmp, &ktmp, qn, kn, &qd],
                wg_rows(2 * rws as usize),
            );
            ctx.copy_range(
                &qtmp,
                0,
                &qj,
                (dsttok * h * 4) as u64,
                (rows * h * 4) as u64,
            );
            ctx.copy_range(
                &ktmp,
                0,
                &kj,
                (dsttok * h * 4) as u64,
                (rows * h * 4) as u64,
            );
            slice_cols(&qkv, &vj, rows, 3 * h, 2 * h, h, dsttok);
        };
        // RoPE the WHOLE joint q,k (shared tables) after both streams scattered.
        let rope_joint = || {
            let rws = (seq * nh) as u32;
            let rd = ctx.uniform_of(
                "dr",
                &[rws, rws, nh as u32, hd as u32, seq as u32, 0u32, 0u32, 0u32],
            );
            self.rope.dispatch(
                "rope",
                &[&qj, &kj, &rcos, &rsin, &rd],
                wg_rows(2 * rws as usize),
            );
        };
        let joint_attn = || {
            let ad = ctx.uniform_of(
                "da",
                &[
                    seq as u32,
                    h as u32,
                    nh as u32,
                    hd as u32,
                    scale.to_bits(),
                    0u32,
                    0u32,
                    0u32,
                ],
            );
            self.attn
                .dispatch("attn", &[&qj, &kj, &vj, &aout, &ad], [seq as u32, 1, 1]);
        };

        // coarse per-section CPU timing (env ARF_DIT_TIME): drains the queue at each marker so
        // wall-clock attributes to the right section. Off by default (one read_f32 sync/marker).
        let timing = std::env::var("ARF_DIT_TIME").is_ok();
        let mark = |tag: &str, since: &std::time::Instant| {
            if timing {
                let _ = ctx.read_f32(&img, 1); // forces submit+poll → all prior work done
                eprintln!(
                    "[dittime] {tag:20} {:.0}ms",
                    since.elapsed().as_secs_f32() * 1000.0
                );
            }
        };
        let t_dbl = std::time::Instant::now();

        // ================= DOUBLE BLOCKS ×19 (txt FIRST in joint seq) =================
        for (di, b) in self.doubles.iter().enumerate() {
            let dbg0 = probe && di == 0;
            // img attention input → scatter img qkv into joint [tl..seq)
            modparams(&b.img_mod);
            self.ln
                .dispatch("iln1", &[&img, &norm, &ln_d(it)], [it as u32, 1, 1]);
            modulate(&norm, it, 1, 0); // (1+scale_msa)*norm + shift_msa
            linear(&norm, &b.img_qkv, &qkv, it, h, 3 * h);
            qknorm_scatter(it, &b.img_qn, &b.img_kn, tl); // img rows at offset tl
                                                          // txt attention input → scatter txt qkv into joint [0..tl)
            modparams(&b.txt_mod);
            self.ln
                .dispatch("tln1", &[&txt, &norm, &ln_d(tl)], [tl as u32, 1, 1]);
            modulate(&norm, tl, 1, 0);
            linear(&norm, &b.txt_qkv, &qkv, tl, h, 3 * h);
            qknorm_scatter(tl, &b.txt_qn, &b.txt_kn, 0);
            // joint rope + attention
            rope_joint();
            joint_attn();

            // img: proj(attn[tl..]) gated; then mlp gated.
            ctx.copy_range(&aout, (tl * h * 4) as u64, &proj, 0, (it * h * 4) as u64);
            linear(&proj, &b.img_proj, &qkv, it, h, h);
            modparams(&b.img_mod);
            gate_add(&img, &qkv, it, 2); // img += gate_msa*proj
            self.ln
                .dispatch("iln2", &[&img, &norm, &ln_d(it)], [it as u32, 1, 1]);
            modulate(&norm, it, 4, 3); // (1+scale_mlp)*norm + shift_mlp
            linear(&norm, &b.img_mlp0, &up, it, h, mlp);
            self.gelu
                .dispatch("ig", &[&up, &act_d(it * mlp)], wg(it * mlp));
            linear(&up, &b.img_mlp2, &qkv, it, mlp, h);
            gate_add(&img, &qkv, it, 5); // img += gate_mlp*mlp
                                         // txt: same with attn[0..tl]
            ctx.copy_range(&aout, 0, &proj, 0, (tl * h * 4) as u64);
            linear(&proj, &b.txt_proj, &qkv, tl, h, h);
            modparams(&b.txt_mod);
            gate_add(&txt, &qkv, tl, 2);
            self.ln
                .dispatch("tln2", &[&txt, &norm, &ln_d(tl)], [tl as u32, 1, 1]);
            modulate(&norm, tl, 4, 3);
            linear(&norm, &b.txt_mlp0, &up, tl, h, mlp);
            self.gelu
                .dispatch("tg", &[&up, &act_d(tl * mlp)], wg(tl * mlp));
            linear(&up, &b.txt_mlp2, &qkv, tl, mlp, h);
            gate_add(&txt, &qkv, tl, 5);
            if dbg0 {
                psum("after double0 img", &img, it * h);
                psum("after double0 txt", &txt, tl * h);
            }
        }
        psum("after doubles img", &img, it * h);
        mark("19 double blocks", &t_dbl);
        let t_sgl = std::time::Instant::now();

        // ================= SINGLE BLOCKS ×38 =================
        // x = cat([txt, img]) into a joint residual stream.
        let x = io(seq * h);
        ctx.copy_range(&txt, 0, &x, 0, (tl * h * 4) as u64);
        ctx.copy_range(&img, 0, &x, (tl * h * 4) as u64, (it * h * 4) as u64);
        for b in &self.singles {
            modparams(&b.mod_lin); // 3 chunks: shift(0),scale(1),gate(2)
            self.ln
                .dispatch("sln", &[&x, &norm, &ln_d(seq)], [seq as u32, 1, 1]);
            modulate(&norm, seq, 1, 0); // (1+scale)*norm + shift
            linear(&norm, &b.linear1, &lin1, seq, h, 3 * h + mlp);
            // split fused [seq, 3h+mlp] row-major: q/k/v/mlp are COLUMN blocks of each row.
            let stride1 = 3 * h + mlp;
            slice_cols(&lin1, &qj, seq, stride1, 0, h, 0);
            slice_cols(&lin1, &kj, seq, stride1, h, h, 0);
            slice_cols(&lin1, &vj, seq, stride1, 2 * h, h, 0);
            // QK-norm (shared weights, whole joint) + rope
            {
                let rws = (seq * nh) as u32;
                let qd = ctx.uniform_of("sqkn", &[rws, rws, hd as u32, 1e-6f32.to_bits()]);
                self.qknorm.dispatch(
                    "sqkn",
                    &[&qj, &kj, &b.qn, &b.kn, &qd],
                    wg_rows(2 * rws as usize),
                );
            }
            rope_joint();
            joint_attn(); // aout = attn
                          // mlp branch: gelu(mlp cols [3h..3h+mlp] of lin1) into dense `up` [seq,mlp]
            slice_cols(&lin1, &up, seq, stride1, 3 * h, mlp, 0);
            self.gelu
                .dispatch("sg", &[&up, &act_d(seq * mlp)], wg(seq * mlp));
            // out = linear2( cat([aout, gelu(mlp)] ) ) per token, width h+mlp=15360. Build the
            // row-major [seq, h+mlp] cat in lin1: aout → cols[0..h], gelu(mlp) → cols[h..h+mlp].
            place_cols(&lin1, &aout, seq, h + mlp, 0, h);
            place_cols(&lin1, &up, seq, h + mlp, h, mlp);
            linear(&lin1, &b.linear2, &proj, seq, h + mlp, h);
            gate_add(&x, &proj, seq, 2); // x += gate*out
        }

        mark("38 single blocks", &t_sgl);
        // img tokens = x[tl.., :]
        psum("after singles x", &x, seq * h);
        let img_final = io(it * h);
        ctx.copy_range(&x, (tl * h * 4) as u64, &img_final, 0, (it * h * 4) as u64);

        // ================= FINAL LAYER =================
        {
            let sv = zeros(h);
            self.add.dispatch("fmc", &[&sv, &vec, &act_d(h)], wg(h));
            self.silu.dispatch("fms", &[&sv, &act_d(h)], wg(h));
            linear(&sv, &self.final_adaln, &mod_p, 1, h, 2 * h); // chunk0=shift, chunk1=scale
        }
        self.ln
            .dispatch("fln", &[&img_final, &norm, &ln_d(it)], [it as u32, 1, 1]);
        modulate(&norm, it, 1, 0); // (1+scale)*norm + shift
        let vel = zeros(it * 64);
        linear(&norm, &self.final_lin, &vel, it, h, 64);
        psum("velocity", &vel, it * 64);
        ctx.read_f32(&vel, it * 64)
    }
}
