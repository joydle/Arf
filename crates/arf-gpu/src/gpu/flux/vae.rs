//! GPU FLUX VAE decoder (latent → RGB image). Runs the BFL/ldm autoencoder decoder entirely on
//! Metal: a 16-channel latent `[16, H, W]` → RGB `[3, 8H, 8W]`. Architecture (BFL tensor names):
//!   conv_in (16→512, 3×3)
//!   → mid: block_1 (resnet) → attn_1 (spatial self-attn) → block_2 (resnet)
//!   → up.3, up.2, up.1, up.0  (each: 3 resnets, then a 2× upsample+conv except up.0)
//!   → norm_out (GroupNorm 32) → SiLU → conv_out (128→3, 3×3)
//! Channels-per-stage: 512,512,512,256→128. Spatial: ×2 at up.3/up.2/up.1 (8× total).
//!
//! Self-contained in the `GpuDiT`/`GpuVisionEncoder` mould: owns its `ComputeKernel`s, uploads
//! resident bf16 conv/linear weights + f32 norms/biases (flush_uploads-safe), runs a fixed-shape
//! multi-dispatch forward, reads back only the final RGB. The conv kernel family (im2col→matmul→
//! epilogue, group_norm, upsample_nearest, vae_attn) is NEW; SiLU + add are reused.
//!
//! Conv is lowered to a matmul via im2col: patches `[out_pixels, C_in·KH·KW]` · `Wᵀ` (W bf16
//! `[C_out, C_in·KH·KW]`, flattened c_in,ky,kx with kx fastest to match im2col column order),
//! then a transpose+bias epilogue back to channel-major `[C_out, H, W]`.

use super::wg;
use std::sync::Arc;

use arf_core::model::weights::read_weights;
use arf_core::{ArfError, Result};

use crate::gpu::{ComputeKernel, GpuContext};

/// FLUX VAE config (the 16-channel autoencoder shared by schnell/dev).
#[derive(Debug, Clone, Copy)]
pub struct VaeConfig {
    pub latent_channels: usize, // 16
    pub groups: usize,          // 32 (GroupNorm)
    pub eps: f32,               // 1e-5
    pub scaling_factor: f32,    // 0.3611
    pub shift_factor: f32,      // 0.1159
}

impl Default for VaeConfig {
    fn default() -> Self {
        VaeConfig {
            latent_channels: 16,
            groups: 32,
            eps: 1e-5,
            scaling_factor: 0.3611,
            shift_factor: 0.1159,
        }
    }
}

/// A conv weight (bf16 `[c_out, c_in·kh·kw]`) + bias (f32 `[c_out]`) + its shape.
struct Conv {
    w: wgpu::Buffer,
    b: wgpu::Buffer,
    c_out: usize,
    c_in: usize,
    k: usize, // kernel size (1 or 3); stride 1, pad = k/2
}

/// A GroupNorm's affine params (`weight`/`bias`, length = channels).
struct Norm {
    w: wgpu::Buffer,
    b: wgpu::Buffer,
}

/// A resnet block: norm1 → SiLU → conv1 → norm2 → SiLU → conv2 → (+ shortcut). `shortcut` is the
/// 1×1 `nin_shortcut` conv when in≠out channels, else None (identity residual).
struct Resnet {
    norm1: Norm,
    conv1: Conv,
    norm2: Norm,
    conv2: Conv,
    shortcut: Option<Conv>,
}

/// The mid-block spatial self-attention (`attn_1`): GroupNorm → q,k,v (1×1 conv) → attn → proj_out.
struct MidAttn {
    norm: Norm,
    q: Conv,
    k: Conv,
    v: Conv,
    proj_out: Conv,
}

/// One decoder up-stage: `resnets` (the first may change channels via a shortcut) + optional
/// 2× nearest upsample followed by a 3×3 conv.
struct UpBlock {
    resnets: Vec<Resnet>,
    upsample: Option<Conv>, // the post-upsample 3×3 conv (None for up.0)
}

pub struct GpuVae {
    ctx: Arc<GpuContext>,
    cfg: VaeConfig,
    conv_in: Conv,
    mid_block1: Resnet,
    mid_attn: MidAttn,
    mid_block2: Resnet,
    ups: Vec<UpBlock>, // ordered as the forward runs them: up.3, up.2, up.1, up.0
    norm_out: Norm,
    conv_out: Conv,
    // kernels
    matmul: ComputeKernel,
    im2col: ComputeKernel,
    conv_epi: ComputeKernel, // transpose + bias → channel-major
    gnorm: ComputeKernel,
    silu: ComputeKernel,
    upsample: ComputeKernel,
    attn: ComputeKernel,
    add: ComputeKernel,
    row_bias: ComputeKernel,
}

impl GpuVae {
    /// Load the BFL VAE decoder from `ae.safetensors` and build the GPU kernels.
    pub fn load(ctx: &Arc<GpuContext>, path: &std::path::Path, cfg: VaeConfig) -> Result<Self> {
        let w = read_weights(&[path])?;
        let get = |n: &str| -> Result<Vec<f32>> {
            w.get_f32(n)
                .ok_or_else(|| ArfError::model_load(format!("vae: missing {n}")))
        };
        let shape = |n: &str| -> Result<Vec<usize>> {
            w.shape(n)
                .ok_or_else(|| ArfError::model_load(format!("vae: missing shape {n}")))
        };
        // a conv: weight `[c_out, c_in, kh, kw]` → flatten to `[c_out, c_in*kh*kw]` (row-major,
        // already c_in,ky,kx with kx fastest) as bf16; bias `[c_out]` as f32.
        let conv = |name: &str| -> Result<Conv> {
            let s = shape(&format!("{name}.weight"))?; // [c_out, c_in, kh, kw]
            let (c_out, c_in, k) = (s[0], s[1], s[2]);
            Ok(Conv {
                w: ctx.storage_init_bf16("vae.cw", &get(&format!("{name}.weight"))?),
                b: ctx.storage_init("vae.cb", &get(&format!("{name}.bias"))?),
                c_out,
                c_in,
                k,
            })
        };
        let norm = |name: &str| -> Result<Norm> {
            let wd = get(&format!("{name}.weight"))?;
            Ok(Norm {
                w: ctx.storage_init("vae.nw", &wd),
                b: ctx.storage_init("vae.nb", &get(&format!("{name}.bias"))?),
            })
        };
        // a resnet at `p`: norm1/conv1/norm2/conv2 (+ nin_shortcut if present).
        let resnet = |p: &str| -> Result<Resnet> {
            let shortcut = if w.shape(&format!("{p}.nin_shortcut.weight")).is_some() {
                Some(conv(&format!("{p}.nin_shortcut"))?)
            } else {
                None
            };
            Ok(Resnet {
                norm1: norm(&format!("{p}.norm1"))?,
                conv1: conv(&format!("{p}.conv1"))?,
                norm2: norm(&format!("{p}.norm2"))?,
                conv2: conv(&format!("{p}.conv2"))?,
                shortcut,
            })
        };

        let mid_block1 = resnet("decoder.mid.block_1")?;
        let mid_attn = MidAttn {
            norm: norm("decoder.mid.attn_1.norm")?,
            q: conv("decoder.mid.attn_1.q")?,
            k: conv("decoder.mid.attn_1.k")?,
            v: conv("decoder.mid.attn_1.v")?,
            proj_out: conv("decoder.mid.attn_1.proj_out")?,
        };
        let mid_block2 = resnet("decoder.mid.block_2")?;

        // BFL decoder iterates up blocks in REVERSE (up.3 first … up.0 last). Each up.N has
        // `block.0..2` resnets; up.1/2/3 have an `upsample.conv`, up.0 does not.
        let mut ups = Vec::new();
        for n in (0..4).rev() {
            let mut resnets = Vec::new();
            for b in 0..3 {
                resnets.push(resnet(&format!("decoder.up.{n}.block.{b}"))?);
            }
            let upsample = if w
                .shape(&format!("decoder.up.{n}.upsample.conv.weight"))
                .is_some()
            {
                Some(conv(&format!("decoder.up.{n}.upsample.conv"))?)
            } else {
                None
            };
            ups.push(UpBlock { resnets, upsample });
        }

        let vae = GpuVae {
            ctx: ctx.clone(),
            cfg,
            conv_in: conv("decoder.conv_in")?,
            mid_block1,
            mid_attn,
            mid_block2,
            ups,
            norm_out: norm("decoder.norm_out")?,
            conv_out: conv("decoder.conv_out")?,
            matmul: ComputeKernel::new(ctx, "vmm", include_str!("../../shaders/wgsl/matmul.wgsl")),
            im2col: ComputeKernel::new(ctx, "vi2c", include_str!("../../shaders/wgsl/im2col.wgsl")),
            conv_epi: ComputeKernel::new(
                ctx,
                "vepi",
                include_str!("../../shaders/wgsl/conv_epilogue.wgsl"),
            ),
            gnorm: ComputeKernel::new(
                ctx,
                "vgn",
                include_str!("../../shaders/wgsl/group_norm.wgsl"),
            ),
            silu: ComputeKernel::new(ctx, "vsilu", include_str!("../../shaders/wgsl/silu.wgsl")),
            upsample: ComputeKernel::new(
                ctx,
                "vup",
                include_str!("../../shaders/wgsl/upsample_nearest.wgsl"),
            ),
            attn: ComputeKernel::new(
                ctx,
                "vattn",
                include_str!("../../shaders/wgsl/vae_attn.wgsl"),
            ),
            add: ComputeKernel::new(ctx, "vadd", include_str!("../../shaders/wgsl/add.wgsl")),
            row_bias: ComputeKernel::new(
                ctx,
                "vrb",
                include_str!("../../shaders/wgsl/row_bias.wgsl"),
            ),
        };
        ctx.flush_uploads(); // commit the deferred weight uploads before the first decode
        Ok(vae)
    }

    /// Decode a latent `[latent_channels, h, w]` (raw, pre-scaling) to RGB `[3, 8h, 8w]` in
    /// `[c, y, x]` channel-major, range roughly [-1, 1]. The caller un-normalizes to `0,255`.
    pub fn decode(&self, latent: &[f32], h: usize, w: usize) -> Vec<f32> {
        let ctx = &self.ctx;
        let probe = std::env::var("ARF_VAE_SUM").is_ok();
        let psum = |tag: &str, b: &wgpu::Buffer, n: usize| {
            if probe {
                let s: f64 = ctx.read_f32(b, n).iter().map(|&x| x as f64).sum();
                eprintln!("[vaesum] {tag:24} sum={s:.4} n={n}");
            }
        };

        // z = z / scaling_factor + shift_factor (the pipeline's pre-decode transform).
        let cin = self.cfg.latent_channels;
        let z: Vec<f32> = latent
            .iter()
            .map(|&v| v / self.cfg.scaling_factor + self.cfg.shift_factor)
            .collect();
        let z_buf = ctx.storage_init("vae.x", &z);

        // conv_in: 16 → 512, 3×3. Activations stay channel-major [c, h, w] through the conv chain.
        let mut cur = self.conv2d(&z_buf, cin, h, w, &self.conv_in);
        let mut c = self.conv_in.c_out;
        let (mut hh, mut ww) = (h, w);
        psum("conv_in", &cur, c * hh * ww);

        // mid: block1 → attn → block2
        cur = self.resnet(&cur, c, hh, ww, &self.mid_block1);
        psum("mid_block1", &cur, c * hh * ww);
        cur = self.mid_attention(&cur, c, hh, ww);
        psum("mid_attn", &cur, c * hh * ww);
        cur = self.resnet(&cur, c, hh, ww, &self.mid_block2);
        psum("mid_block2", &cur, c * hh * ww);

        // up blocks (already in forward order up.3..up.0)
        for (ui, up) in self.ups.iter().enumerate() {
            for rs in &up.resnets {
                cur = self.resnet(&cur, c, hh, ww, rs);
                c = rs.conv2.c_out;
            }
            if let Some(upconv) = &up.upsample {
                // nearest 2× then 3×3 conv
                let big = ctx.storage_io_f32("vae.up", c * (2 * hh) * (2 * ww));
                let u = ctx.uniform_of("vup", &[c as u32, hh as u32, ww as u32, 0u32]);
                self.upsample
                    .dispatch("up", &[&cur, &big, &u], wg(c * 2 * hh * 2 * ww));
                hh *= 2;
                ww *= 2;
                cur = self.conv2d(&big, c, hh, ww, upconv);
                c = upconv.c_out;
            }
            psum(&format!("after up[{ui}]"), &cur, c * hh * ww);
        }

        // norm_out → SiLU → conv_out (128 → 3)
        self.group_norm(&cur, c, hh, ww, &self.norm_out);
        self.silu
            .dispatch("vsilu", &[&cur, &act_d(ctx, c * hh * ww)], wg(c * hh * ww));
        let rgb = self.conv2d(&cur, c, hh, ww, &self.conv_out);
        psum("conv_out(rgb)", &rgb, 3 * hh * ww);
        ctx.read_f32(&rgb, 3 * hh * ww)
    }

    /// conv2d (stride 1, pad k/2) via im2col → matmul → transpose+bias epilogue. Input + output
    /// are channel-major `[c, h, w]`; output channels = `cv.c_out` at the SAME spatial size.
    fn conv2d(
        &self,
        inp: &wgpu::Buffer,
        c_in: usize,
        h: usize,
        w: usize,
        cv: &Conv,
    ) -> wgpu::Buffer {
        let ctx = &self.ctx;
        debug_assert_eq!(c_in, cv.c_in);
        let pad = cv.k / 2;
        let out_pixels = h * w; // stride-1 'same'
        let patch = cv.c_in * cv.k * cv.k;

        // The full output [c_out, out_pixels] is the only buffer sized to the whole image; the
        // transient im2col `cols` is TILED by output-row band so its peak stays bounded regardless
        // of resolution (at 1024px the un-tiled cols is ~4.8 GB — the dominant VAE spike). Pick a
        // row band so cols ≈ band_rows·w·patch·4B ≤ COLS_CAP; the input stays fully resident so a
        // band's patches read across its boundary freely.
        let out = ctx.storage_io_f32("vae.conv", cv.c_out * out_pixels);

        const COLS_CAP: usize = 256 << 20; // 256 MiB cap on the transient cols buffer
        let bytes_per_row = w * patch * 4;
        let band_rows = (COLS_CAP / bytes_per_row.max(1)).clamp(1, h);
        // Reuse one cols/mm pair sized for a full band across all tiles (last tile ≤ band_rows).
        let band_pixels = band_rows * w;
        let cols = ctx.storage_io_f32("vae.cols", band_pixels * patch);
        let mm = ctx.storage_io_f32("vae.mm", band_pixels * cv.c_out);

        let mut row0 = 0usize;
        while row0 < h {
            let rows = (h - row0).min(band_rows);
            let tile_pixels = rows * w;
            // im2col → cols[tile_pixels, patch] for output rows [row0, row0+rows).
            let id = ctx.uniform_of(
                "vi2c",
                &[
                    cv.c_in as u32,
                    h as u32,
                    w as u32,
                    cv.k as u32,
                    cv.k as u32,
                    pad as u32,
                    rows as u32,
                    w as u32,
                    row0 as u32,
                    0,
                    0,
                    0,
                ],
            );
            self.im2col
                .dispatch("i2c", &[inp, &cols, &id], wg(tile_pixels * patch));
            // matmul: mm[tile_pixels, c_out] = cols · Wᵀ
            let md = ctx.uniform_of(
                "vmm",
                &[tile_pixels as u32, patch as u32, cv.c_out as u32, 0u32],
            );
            let mwg = [
                (tile_pixels as u32).div_ceil(16),
                (cv.c_out as u32).div_ceil(16),
                1u32,
            ];
            self.matmul
                .dispatch("conv_mm", &[&cols, &cv.w, &mm, &md], mwg);
            // epilogue: scatter tile into out[c_out, out_pixels] at column offset row0*w.
            let ed = ctx.uniform_of(
                "vepi",
                &[
                    tile_pixels as u32,
                    cv.c_out as u32,
                    (row0 * w) as u32,
                    out_pixels as u32,
                ],
            );
            self.conv_epi
                .dispatch("epi", &[&mm, &cv.b, &out, &ed], wg(tile_pixels * cv.c_out));
            row0 += rows;
        }
        out
    }

    /// GroupNorm (in place) over channel-major `[c, h, w]`.
    fn group_norm(&self, x: &wgpu::Buffer, c: usize, h: usize, w: usize, n: &Norm) {
        let ctx = &self.ctx;
        let d = ctx.uniform_of(
            "vgn",
            &[
                c as u32,
                self.cfg.groups as u32,
                (h * w) as u32,
                self.cfg.eps.to_bits(),
            ],
        );
        self.gnorm
            .dispatch("gn", &[x, &n.w, &n.b, &d], [self.cfg.groups as u32, 1, 1]);
    }

    /// Resnet block: norm1 → SiLU → conv1 → norm2 → SiLU → conv2 → (+shortcut). Input channel-major.
    fn resnet(
        &self,
        inp: &wgpu::Buffer,
        c_in: usize,
        h: usize,
        w: usize,
        rs: &Resnet,
    ) -> wgpu::Buffer {
        let ctx = &self.ctx;
        // h = norm1(x); h = silu(h); h = conv1(h)  — copy x first so the residual stays intact.
        let hbuf = ctx.storage_io_f32("vae.rs", c_in * h * w);
        ctx.copy_range(inp, 0, &hbuf, 0, (c_in * h * w * 4) as u64);
        self.group_norm(&hbuf, c_in, h, w, &rs.norm1);
        self.silu.dispatch(
            "rsilu",
            &[&hbuf, &act_d(ctx, c_in * h * w)],
            wg(c_in * h * w),
        );
        let mut cur = self.conv2d(&hbuf, c_in, h, w, &rs.conv1);
        let c_mid = rs.conv1.c_out;
        // h = norm2(h); h = silu(h); h = conv2(h)
        self.group_norm(&cur, c_mid, h, w, &rs.norm2);
        self.silu.dispatch(
            "rsilu2",
            &[&cur, &act_d(ctx, c_mid * h * w)],
            wg(c_mid * h * w),
        );
        cur = self.conv2d(&cur, c_mid, h, w, &rs.conv2);
        let c_out = rs.conv2.c_out;
        // residual: + (shortcut(x) or x)
        let res = match &rs.shortcut {
            Some(sc) => self.conv2d(inp, c_in, h, w, sc),
            None => {
                let r = ctx.storage_io_f32("vae.res", c_out * h * w);
                ctx.copy_range(inp, 0, &r, 0, (c_out * h * w * 4) as u64);
                r
            }
        };
        self.add.dispatch(
            "radd",
            &[&cur, &res, &act_d(ctx, c_out * h * w)],
            wg(c_out * h * w),
        );
        cur
    }

    /// Mid-block spatial self-attention: GroupNorm → q,k,v (1×1 conv) → softmax(qkᵀ/√c)·v →
    /// proj_out (1×1 conv) → + residual. Works in pixel-major `[seq=h*w, dim=c]` for the attention.
    fn mid_attention(&self, inp: &wgpu::Buffer, c: usize, h: usize, w: usize) -> wgpu::Buffer {
        let ctx = &self.ctx;
        let seq = h * w;
        // normed = group_norm(x) (on a copy)
        let normed = ctx.storage_io_f32("vae.an", c * h * w);
        ctx.copy_range(inp, 0, &normed, 0, (c * h * w * 4) as u64);
        self.group_norm(&normed, c, h, w, &self.mid_attn.norm);
        // q,k,v: 1×1 conv → pixel-major [seq, c] via im2col(1×1)+matmul WITHOUT the channel-major
        // epilogue (we want [seq, dim] for attention). 1×1 im2col cols == the input transposed to
        // [seq, c]; matmul gives [seq, c_out] directly.
        let q = self.qkv_proj(&normed, c, seq, &self.mid_attn.q);
        let k = self.qkv_proj(&normed, c, seq, &self.mid_attn.k);
        let v = self.qkv_proj(&normed, c, seq, &self.mid_attn.v);
        // attention out [seq, c]
        let ao = ctx.storage_io_f32("vae.ao", seq * c);
        let scale = 1.0f32 / (c as f32).sqrt();
        let ad = ctx.uniform_of("vattn", &[seq as u32, c as u32, scale.to_bits(), 0u32]);
        self.attn
            .dispatch("attn", &[&q, &k, &v, &ao, &ad], [seq as u32, 1, 1]);
        // proj_out: 1×1 conv on pixel-major [seq,c] (a plain linear, no transpose) → [seq,c],
        // then conv_epilogue transposes [seq,c]→channel-major [c,seq] (zero bias).
        let projected = self.linear_seq(&ao, c, seq, &self.mid_attn.proj_out);
        let chw = ctx.storage_io_f32("vae.aproj", c * h * w);
        // untiled transpose over all seq pixels: pixel_off=0, full_pixels=seq.
        let td = ctx.uniform_of("vepi", &[seq as u32, c as u32, 0u32, seq as u32]);
        let zb = ctx.storage_init("vae.zb", &vec![0.0f32; c]);
        self.conv_epi
            .dispatch("atr", &[&projected, &zb, &chw, &td], wg(c * seq));
        // residual add onto the ORIGINAL input
        let out = ctx.storage_io_f32("vae.aout", c * h * w);
        ctx.copy_range(inp, 0, &out, 0, (c * h * w * 4) as u64);
        self.add
            .dispatch("aadd", &[&out, &chw, &act_d(ctx, c * h * w)], wg(c * h * w));
        out
    }

    /// 1×1 conv from CHANNEL-major `[c_in, seq]` → PIXEL-major `[seq, c_out]` (q/k/v projection).
    /// A 1×1 conv is `out[p,co] = sum_ci in[ci,p]·W[co,ci] + b[co]`. A 1×1 im2col transposes
    /// `[c_in,seq]→[seq,c_in]`; matmul by Wᵀ (bf16 `[c_out,c_in]`); add per-row bias.
    fn qkv_proj(&self, inp: &wgpu::Buffer, c_in: usize, seq: usize, cv: &Conv) -> wgpu::Buffer {
        let ctx = &self.ctx;
        let a = ctx.storage_io_f32("vae.a1", seq * c_in);
        let id = ctx.uniform_of(
            "vi2c",
            &[
                c_in as u32,
                seq as u32,
                1u32,
                1u32,
                1u32,
                0u32,
                seq as u32,
                1u32,
                0u32,
                0,
                0,
                0,
            ],
        );
        self.im2col
            .dispatch("i2c1", &[inp, &a, &id], wg(seq * c_in));
        self.linear_into(&a, c_in, seq, cv)
    }

    /// 1×1 conv on a PIXEL-major `[seq, c_in]` activation → `[seq, c_out]` (proj_out: input is
    /// already pixel-major, so no transpose — just matmul by Wᵀ + bias).
    fn linear_seq(&self, a: &wgpu::Buffer, c_in: usize, seq: usize, cv: &Conv) -> wgpu::Buffer {
        self.linear_into(a, c_in, seq, cv)
    }

    /// matmul `[seq,c_in]·Wᵀ` (W bf16 `[c_out,c_in]`) + per-row bias → `[seq,c_out]`.
    fn linear_into(&self, a: &wgpu::Buffer, c_in: usize, seq: usize, cv: &Conv) -> wgpu::Buffer {
        let ctx = &self.ctx;
        let mm = ctx.storage_io_f32("vae.m1", seq * cv.c_out);
        let md = ctx.uniform_of("vmm", &[seq as u32, c_in as u32, cv.c_out as u32, 0u32]);
        let mwg = [
            (seq as u32).div_ceil(16),
            (cv.c_out as u32).div_ceil(16),
            1u32,
        ];
        self.matmul.dispatch("mm1", &[a, &cv.w, &mm, &md], mwg);
        let bd = ctx.uniform_of("vrb", &[seq as u32, cv.c_out as u32, 0u32, 0u32]);
        self.row_bias
            .dispatch("rb", &[&mm, &cv.b, &bd], wg(seq * cv.c_out));
        mm
    }
}

fn act_d(ctx: &GpuContext, n: usize) -> wgpu::Buffer {
    ctx.uniform_of("vact", &[n as u32, 0u32, 0u32, 0u32])
}
