//! Gemma-3 SigLIP vision encoder (CPU, f32). Loads the `vision.*` tensors a multimodal
//! GGUF (mmproj) exposes and runs the image → patch embeddings → 27 SigLIP layers →
//! post-norm forward. The projector (pool + 1152→2560) is in `vision_projector.rs`-style
//! follow-up; this module produces the per-patch hidden states `[num_patches, 1152]`.
//!
//! SigLIP differs from Arf's text path: standard LayerNorm WITH bias (not RMSNorm),
//! Q/K/V/O biases, gelu (tanh) MLP, a Conv2d patch embedding, and learned position
//! embeddings. All compute is plain f32 here (the mmproj weights dequantize to f32) —
//! correctness first; a GPU port can follow once the end-to-end path is proven.

use crate::model::gguf::LazyGguf;
use crate::tensor::matmul_nt;
use crate::{ArfError, Result};

/// SigLIP config (read from the mmproj `clip.vision.*` metadata; hard-defaulted to the
/// Gemma-3-4B values, which are fixed for this model).
#[derive(Debug, Clone, Copy)]
pub struct VisionConfig {
    pub image_size: usize, // 896
    pub patch_size: usize, // 14
    pub hidden: usize,     // 1152
    pub ffn: usize,        // 4304
    pub layers: usize,     // 27
    pub heads: usize,      // 16
    pub ln_eps: f32,       // ~1e-6
}

impl Default for VisionConfig {
    fn default() -> Self {
        VisionConfig {
            image_size: 896,
            patch_size: 14,
            hidden: 1152,
            ffn: 4304,
            layers: 27,
            heads: 16,
            ln_eps: 1e-6,
        }
    }
}

impl VisionConfig {
    /// Patches per side / total (image_size / patch_size).
    pub fn grid(&self) -> usize {
        self.image_size / self.patch_size
    }
    pub fn num_patches(&self) -> usize {
        self.grid() * self.grid()
    }

    /// Read `image_size` / `patch_size` from the mmproj `clip.vision.*` metadata, falling
    /// back to the hard-coded Gemma-3-4B defaults when a key is absent. The transformer
    /// dims (hidden/ffn/layers/heads) stay defaulted — they're fixed for this tower and
    /// the loaded tensor shapes already encode them.
    pub fn from_gguf(g: &LazyGguf) -> Self {
        let mut c = VisionConfig::default();
        if let Some(s) = g.get_metadata_u32("clip.vision.image_size") {
            c.image_size = s as usize;
        }
        if let Some(p) = g.get_metadata_u32("clip.vision.patch_size") {
            c.patch_size = p as usize;
        }
        c
    }
}

/// Image preprocessing parameters: the per-channel mean/std the SigLIP tower was trained
/// with (read from the mmproj `clip.vision.image_mean` / `image_std`) and the target square
/// size. Gemma-3 normalizes as `(px/255 - mean) / std` per RGB channel.
#[derive(Debug, Clone)]
pub struct VisionPreprocess {
    pub size: usize,    // 896
    pub mean: [f32; 3], // per-channel, ~[0.5, 0.5, 0.5]
    pub std: [f32; 3],  // per-channel, ~[0.5, 0.5, 0.5]
}

impl VisionPreprocess {
    /// Load mean/std/size from the mmproj metadata, defaulting to Gemma-3's 0.5/0.5 and
    /// 896 when absent. The mmproj DOES carry the real values (verified), so this normally
    /// reads them exactly rather than guessing.
    pub fn from_gguf(g: &LazyGguf) -> Self {
        let size = g
            .get_metadata_u32("clip.vision.image_size")
            .map(|s| s as usize)
            .unwrap_or(896);
        let to3 = |v: Option<Vec<f32>>, d: f32| -> [f32; 3] {
            match v {
                Some(a) if a.len() >= 3 => [a[0], a[1], a[2]],
                Some(a) if a.len() == 1 => [a[0], a[0], a[0]],
                _ => [d, d, d],
            }
        };
        VisionPreprocess {
            size,
            mean: to3(g.get_metadata_f32_array("clip.vision.image_mean"), 0.5),
            std: to3(g.get_metadata_f32_array("clip.vision.image_std"), 0.5),
        }
    }

    /// Preprocess a decoded RGB image into the encoder's input layout: bilinearly resize
    /// `[src_h, src_w, 3]` (row-major, u8 0..255) to `size×size`, normalize each channel
    /// `(px/255 - mean)/std`, and emit channel-major `[3, size, size]` f32 (what
    /// `VisionEncoder::patch_embed` / `encode` expect). No aspect-ratio preservation —
    /// Gemma-3's processor resizes to a fixed square.
    pub fn preprocess(&self, rgb: &[u8], src_w: usize, src_h: usize) -> Vec<f32> {
        let n = self.size;
        assert_eq!(rgb.len(), src_w * src_h * 3, "rgb must be [h,w,3] u8");
        let mut out = vec![0.0f32; 3 * n * n];
        // Bilinear sample with the half-pixel-center convention (align_corners=false),
        // matching torchvision/PIL bilinear that the HF Gemma-3 processor uses.
        let sx = src_w as f32 / n as f32;
        let sy = src_h as f32 / n as f32;
        for dy in 0..n {
            let fy = ((dy as f32 + 0.5) * sy - 0.5).max(0.0);
            let y0 = fy.floor() as usize;
            let y1 = (y0 + 1).min(src_h - 1);
            let wy = fy - y0 as f32;
            for dx in 0..n {
                let fx = ((dx as f32 + 0.5) * sx - 0.5).max(0.0);
                let x0 = fx.floor() as usize;
                let x1 = (x0 + 1).min(src_w - 1);
                let wx = fx - x0 as f32;
                for ch in 0..3 {
                    let p = |y: usize, x: usize| rgb[(y * src_w + x) * 3 + ch] as f32;
                    let top = p(y0, x0) * (1.0 - wx) + p(y0, x1) * wx;
                    let bot = p(y1, x0) * (1.0 - wx) + p(y1, x1) * wx;
                    let v = top * (1.0 - wy) + bot * wy;
                    let norm = (v / 255.0 - self.mean[ch]) / self.std[ch];
                    out[ch * n * n + dy * n + dx] = norm;
                }
            }
        }
        out
    }
}

/// One SigLIP transformer layer's weights (all f32). Public so the GPU encoder
/// (arf-gpu) can upload them; the field layout is the lossless mmproj data.
pub struct VisionLayer {
    pub ln1_w: Vec<f32>,
    pub ln1_b: Vec<f32>,
    pub q_w: Vec<f32>, // [hidden, hidden] row-major [out, in]
    pub q_b: Vec<f32>,
    pub k_w: Vec<f32>,
    pub k_b: Vec<f32>,
    pub v_w: Vec<f32>,
    pub v_b: Vec<f32>,
    pub o_w: Vec<f32>,
    pub o_b: Vec<f32>,
    pub ln2_w: Vec<f32>,
    pub ln2_b: Vec<f32>,
    pub up_w: Vec<f32>, // [ffn, hidden]
    pub up_b: Vec<f32>,
    pub down_w: Vec<f32>, // [hidden, ffn]
    pub down_b: Vec<f32>,
}

/// The vision-tower weights the GPU encoder uploads (layers + post-norm).
pub struct VisionWeights {
    pub layers: Vec<VisionLayer>,
    pub post_ln_w: Vec<f32>,
    pub post_ln_b: Vec<f32>,
}

/// The full vision tower (patch embed + position embed + layers + post-norm).
pub struct VisionEncoder {
    pub cfg: VisionConfig,
    patch_w: Vec<f32>, // flattened [hidden, 3*patch*patch] after reshape (see note)
    patch_b: Vec<f32>, // [hidden]
    pos_embd: Vec<f32>, // [num_pos, hidden]
    /// Per-layer + post-norm weights. Owned here; `weights()` lends them to the GPU encoder.
    w: VisionWeights,
}

impl VisionEncoder {
    /// Load the encoder from a mmproj GGUF that was opened with the vision tensors indexed
    /// under `vision.*` (Phase 1). Returns `None`-equivalent error if a tensor is missing.
    pub fn load(g: &LazyGguf, cfg: VisionConfig) -> Result<Self> {
        let get = |name: &str| -> Result<Vec<f32>> {
            g.get_f32(name)
                .ok_or_else(|| ArfError::model_load(format!("vision: missing {name}")))
        };
        let layers = (0..cfg.layers)
            .map(|i| {
                let p = format!("vision.v.blk.{i}");
                Ok(VisionLayer {
                    ln1_w: get(&format!("{p}.ln1.weight"))?,
                    ln1_b: get(&format!("{p}.ln1.bias"))?,
                    q_w: get(&format!("{p}.attn_q.weight"))?,
                    q_b: get(&format!("{p}.attn_q.bias"))?,
                    k_w: get(&format!("{p}.attn_k.weight"))?,
                    k_b: get(&format!("{p}.attn_k.bias"))?,
                    v_w: get(&format!("{p}.attn_v.weight"))?,
                    v_b: get(&format!("{p}.attn_v.bias"))?,
                    o_w: get(&format!("{p}.attn_out.weight"))?,
                    o_b: get(&format!("{p}.attn_out.bias"))?,
                    ln2_w: get(&format!("{p}.ln2.weight"))?,
                    ln2_b: get(&format!("{p}.ln2.bias"))?,
                    up_w: get(&format!("{p}.ffn_up.weight"))?,
                    up_b: get(&format!("{p}.ffn_up.bias"))?,
                    down_w: get(&format!("{p}.ffn_down.weight"))?,
                    down_b: get(&format!("{p}.ffn_down.bias"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(VisionEncoder {
            cfg,
            patch_w: get("vision.v.patch_embd.weight")?,
            patch_b: get("vision.v.patch_embd.bias")?,
            pos_embd: get("vision.v.position_embd.weight")?,
            w: VisionWeights {
                layers,
                post_ln_w: get("vision.v.post_ln.weight")?,
                post_ln_b: get("vision.v.post_ln.bias")?,
            },
        })
    }

    /// Borrow the layer + post-norm weights so the GPU encoder can upload them once.
    pub fn weights(&self) -> &VisionWeights {
        &self.w
    }

    /// Patch-embed + position-embed only: pixels `[3,H,W]` → hidden states `[np, hidden]`,
    /// the input to the SigLIP transformer stack. Split out so the GPU encoder can reuse
    /// the (cheap, CPU-bound) conv+pos part and run only the 27 layers on the GPU.
    pub fn patch_embed(&self, pixels: &[f32]) -> Vec<f32> {
        let c = &self.cfg;
        let grid = c.grid();
        let np = c.num_patches();
        // 1) Patch embedding: a Conv2d(3→hidden, k=patch, stride=patch) is exactly a linear
        //    over each non-overlapping patch's flattened pixels. patch_w is [hidden,
        //    3*patch*patch] (ggml stores the 4D conv [patch,patch,3,hidden] as that matrix).
        let pdim = 3 * c.patch_size * c.patch_size;
        let mut patches = vec![0.0f32; np * pdim];
        for py in 0..grid {
            for px in 0..grid {
                let patch_idx = py * grid + px;
                let dst = &mut patches[patch_idx * pdim..(patch_idx + 1) * pdim];
                let mut o = 0;
                for ch in 0..3 {
                    for ky in 0..c.patch_size {
                        for kx in 0..c.patch_size {
                            let y = py * c.patch_size + ky;
                            let x = px * c.patch_size + kx;
                            dst[o] =
                                pixels[ch * c.image_size * c.image_size + y * c.image_size + x];
                            o += 1;
                        }
                    }
                }
            }
        }
        // hidden = patches[np,pdim] · patch_w[hidden,pdim]ᵀ + patch_b
        let mut h = matmul_nt(&patches, np, pdim, &self.patch_w, c.hidden);
        vsum("patch_matmul (node_3)", &h);
        add_bias(&mut h, &self.patch_b, np, c.hidden);
        vsum("patch_bias", &h);
        // 2) + learned position embeddings (first np rows of pos_embd).
        for i in 0..np {
            for j in 0..c.hidden {
                h[i * c.hidden + j] += self.pos_embd[i * c.hidden + j];
            }
        }
        vsum("pos_embed (hidden0)", &h);
        h
    }

    /// Encode a preprocessed image into per-patch hidden states `[num_patches, hidden]`.
    /// `pixels` is the normalized RGB image as `[3, H, W]` (channel-major, H=W=image_size,
    /// values already (px/255 - mean)/std). Returns the post-norm hidden states.
    pub fn encode(&self, pixels: &[f32]) -> Vec<f32> {
        let c = &self.cfg;
        let np = c.num_patches();
        let mut h = self.patch_embed(pixels);
        // 3) 27 SigLIP transformer layers.
        for layer in &self.w.layers {
            self.block(&mut h, layer, np);
        }
        // 4) final LayerNorm.
        layer_norm(
            &mut h,
            &self.w.post_ln_w,
            &self.w.post_ln_b,
            np,
            c.hidden,
            c.ln_eps,
        );
        h
    }

    /// One pre-norm SigLIP block: x += attn(ln1(x)); x += mlp(ln2(x)).
    fn block(&self, h: &mut [f32], l: &VisionLayer, n: usize) {
        let d = self.cfg.hidden;
        // --- attention ---
        let mut x = h.to_vec();
        layer_norm(&mut x, &l.ln1_w, &l.ln1_b, n, d, self.cfg.ln_eps);
        let mut q = matmul_nt(&x, n, d, &l.q_w, d);
        add_bias(&mut q, &l.q_b, n, d);
        let mut k = matmul_nt(&x, n, d, &l.k_w, d);
        add_bias(&mut k, &l.k_b, n, d);
        let mut v = matmul_nt(&x, n, d, &l.v_w, d);
        add_bias(&mut v, &l.v_b, n, d);
        let attn = self.mha(&q, &k, &v, n);
        let mut o = matmul_nt(&attn, n, d, &l.o_w, d);
        add_bias(&mut o, &l.o_b, n, d);
        for i in 0..n * d {
            h[i] += o[i];
        }
        // --- MLP ---
        let mut y = h.to_vec();
        layer_norm(&mut y, &l.ln2_w, &l.ln2_b, n, d, self.cfg.ln_eps);
        let f = self.cfg.ffn;
        let mut up = matmul_nt(&y, n, d, &l.up_w, f);
        add_bias(&mut up, &l.up_b, n, f);
        for v in up.iter_mut() {
            *v = gelu_tanh(*v);
        }
        let mut down = matmul_nt(&up, n, f, &l.down_w, d);
        add_bias(&mut down, &l.down_b, n, d);
        for i in 0..n * d {
            h[i] += down[i];
        }
    }

    /// Bidirectional multi-head self-attention (no mask, no RoPE — SigLIP is a plain
    /// ViT encoder). q/k/v are `[n, d]`; returns `[n, d]`.
    fn mha(&self, q: &[f32], k: &[f32], v: &[f32], n: usize) -> Vec<f32> {
        let d = self.cfg.hidden;
        let h = self.cfg.heads;
        let hd = d / h;
        let scale = 1.0 / (hd as f32).sqrt();
        let mut out = vec![0.0f32; n * d];
        for head in 0..h {
            let off = head * hd;
            for i in 0..n {
                // scores over all j, softmax, weighted sum of v.
                let mut scores = vec![0.0f32; n];
                let mut max = f32::NEG_INFINITY;
                for j in 0..n {
                    let mut s = 0.0;
                    for t in 0..hd {
                        s += q[i * d + off + t] * k[j * d + off + t];
                    }
                    s *= scale;
                    scores[j] = s;
                    if s > max {
                        max = s;
                    }
                }
                let mut sum = 0.0;
                for s in scores.iter_mut() {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                let inv = 1.0 / sum;
                for j in 0..n {
                    let w = scores[j] * inv;
                    for t in 0..hd {
                        out[i * d + off + t] += w * v[j * d + off + t];
                    }
                }
            }
        }
        out
    }
}

/// The Gemma-3 multimodal projector: pools the SigLIP per-patch output down to a fixed
/// token count, RMS-normalizes, and projects 1152 → 2560 (the LM hidden dim), yielding the
/// image *soft tokens* spliced into the language model's input sequence.
///
/// Gemma-3 reduces the 64×64 patch grid to a 16×16 grid (256 tokens) by average-pooling
/// over 4×4 patch windows (`mm_tokens_per_image = 256`). `mm.soft_emb_norm` is a Gemma RMS
/// norm (scale only, no bias, `(1+w)`-free here — ggml stores the raw scale) applied before
/// `mm.input_projection`.
pub struct VisionProjector {
    soft_emb_norm: Vec<f32>,    // [hidden] RMS scale
    input_proj: Vec<f32>,       // [out=2560, in=hidden] row-major [out,in]
    pub hidden: usize,          // 1152
    pub out_dim: usize,         // 2560
    pub tokens_per_side: usize, // 16 → 256 tokens
}

impl VisionProjector {
    pub fn load(g: &LazyGguf, hidden: usize, out_dim: usize) -> Result<Self> {
        let get = |n: &str| -> Result<Vec<f32>> {
            g.get_f32(n)
                .ok_or_else(|| ArfError::model_load(format!("vision: missing {n}")))
        };
        Ok(VisionProjector {
            soft_emb_norm: get("vision.mm.soft_emb_norm.weight")?,
            // `mm.input_projection.weight` is the one vision matmul where ggml's ne0 is the
            // OUT dim, not the in dim: it's stored {ne0=out=2560, ne1=in=1152}, whereas the
            // SigLIP layer weights are {ne0=in, ne1=out}. The generic 2D vision indexer assumes
            // ne0=in, so it hands us a [1152, 2560] = [in, out] matrix; `matmul_nt` wants
            // [out, in]. Transpose once here. (llama.cpp's graph TRANSPOSEs this exact tensor
            // and nothing else — the bug that made image soft-tokens ~2× off and ungrounded.)
            input_proj: transpose(&get("vision.mm.input_projection.weight")?, hidden, out_dim),
            hidden,
            out_dim,
            tokens_per_side: 16,
        })
    }

    /// Project the encoder output `patches[grid*grid, hidden]` into image soft tokens
    /// `[tokens_per_side², out_dim]`. `grid` = encoder.cfg.grid() (64 for Gemma-3-4B).
    pub fn project(&self, patches: &[f32], grid: usize) -> Vec<f32> {
        let d = self.hidden;
        let ts = self.tokens_per_side; // 16
        let win = grid / ts; // 4 (64/16)
        let ntok = ts * ts; // 256
                            // 1) average-pool the grid×grid patches into ts×ts windows.
        let mut pooled = vec![0.0f32; ntok * d];
        let area = (win * win) as f32;
        for ty in 0..ts {
            for tx in 0..ts {
                let dst = &mut pooled[(ty * ts + tx) * d..(ty * ts + tx + 1) * d];
                for wy in 0..win {
                    for wx in 0..win {
                        let py = ty * win + wy;
                        let px = tx * win + wx;
                        let src = &patches[(py * grid + px) * d..(py * grid + px + 1) * d];
                        for j in 0..d {
                            dst[j] += src[j];
                        }
                    }
                }
                for j in 0..d {
                    dst[j] /= area;
                }
            }
        }
        vsum("pooled (node_858)", &pooled);
        // 2) RMSNorm (scale only) per token.
        rms_norm(&mut pooled, &self.soft_emb_norm, ntok, d, 1e-6);
        vsum("rms·soft_emb (node_863)", &pooled);
        // 3) project hidden → out_dim. input_proj is [out_dim, hidden] → matmul_nt.
        let out = matmul_nt(&pooled, ntok, d, &self.input_proj, self.out_dim);
        vsum("projection (node_864)", &out);
        out
    }
}

// ---- small f32 helpers ----------------------------------------------------

/// Transpose a row-major `[rows, cols]` matrix into `[cols, rows]`.
fn transpose(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = m[r * cols + c];
        }
    }
    out
}

/// Debug: print a tensor's element sum when ARF_VISION_SUM is set, to compare against
/// llama.cpp's `MTMD_DEBUG_GRAPH` per-node `sum=` values and localize numerical divergence.
fn vsum(tag: &str, v: &[f32]) {
    if std::env::var("ARF_VISION_SUM").is_ok() {
        let s: f64 = v.iter().map(|&x| x as f64).sum();
        eprintln!("[vsum] {tag:30} sum={s:.4} len={}", v.len());
    }
}

/// Gemma-style RMSNorm (scale only, no bias, no mean-subtraction), in place, row-wise.
fn rms_norm(x: &mut [f32], w: &[f32], rows: usize, cols: usize, eps: f32) {
    for r in 0..rows {
        let row = &mut x[r * cols..(r + 1) * cols];
        let ms = row.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for c in 0..cols {
            row[c] = row[c] * inv * w[c];
        }
    }
}

fn add_bias(x: &mut [f32], b: &[f32], rows: usize, cols: usize) {
    for r in 0..rows {
        for c in 0..cols {
            x[r * cols + c] += b[c];
        }
    }
}

/// Standard LayerNorm WITH learnable weight + bias, in place, row-wise.
fn layer_norm(x: &mut [f32], w: &[f32], b: &[f32], rows: usize, cols: usize, eps: f32) {
    for r in 0..rows {
        let row = &mut x[r * cols..(r + 1) * cols];
        let mean = row.iter().sum::<f32>() / cols as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for c in 0..cols {
            row[c] = (row[c] - mean) * inv * w[c] + b[c];
        }
    }
}

/// GELU (tanh approximation — matches ggml's `use_gelu`).
fn gelu_tanh(x: f32) -> f32 {
    const K: f32 = 0.797_884_6; // sqrt(2/pi)
    0.5 * x * (1.0 + (K * (x + 0.044715 * x * x * x)).tanh())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_layernorm_sanity() {
        assert!((gelu_tanh(0.0)).abs() < 1e-6);
        assert!(gelu_tanh(10.0) > 9.9 && gelu_tanh(-10.0).abs() < 1e-3);
        // layer_norm of a row → zero mean, unit-ish var, then affine identity (w=1,b=0).
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0; 4];
        let b = vec![0.0; 4];
        layer_norm(&mut x, &w, &b, 1, 4, 1e-6);
        let mean: f32 = x.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-4, "normalized mean ~0");
    }

    /// The mmproj carries real preprocessing metadata — read it + sanity-check + exercise
    /// the bilinear resize/normalize. Skips if the mmproj GGUF isn't present.
    #[test]
    fn vision_preprocess_from_real_mmproj() {
        let path = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../models/gemma-3-4b-vision/mmproj-F16.gguf"
        ));
        if !path.exists() {
            eprintln!("SKIP: mmproj GGUF not found");
            return;
        }
        let g = LazyGguf::open(path, &crate::config::ModelConfig::gemma3_4b()).expect("open");
        let pp = VisionPreprocess::from_gguf(&g);
        eprintln!(
            "mmproj preprocess: size={} mean={:?} std={:?}",
            pp.size, pp.mean, pp.std
        );
        assert_eq!(pp.size, 896, "Gemma-3 image size");
        // Gemma-3 SigLIP normalization is 0.5/0.5 per channel (the values in the mmproj).
        for c in 0..3 {
            assert!(pp.mean[c] > 0.0 && pp.mean[c] < 1.0, "mean[{c}] in (0,1)");
            assert!(pp.std[c] > 0.0 && pp.std[c] <= 1.0, "std[{c}] in (0,1]");
        }
        let cfg = VisionConfig::from_gguf(&g);
        assert_eq!(cfg.image_size, 896);
        assert_eq!(cfg.patch_size, 14);

        // Preprocess a tiny synthetic 4×4 RGB image → [3,896,896], all finite, resize works.
        let (sw, sh) = (4usize, 4usize);
        let mut rgb = vec![0u8; sw * sh * 3];
        for i in 0..sw * sh {
            rgb[i * 3] = (i * 16) as u8;
            rgb[i * 3 + 1] = 128;
            rgb[i * 3 + 2] = 255 - (i * 16) as u8;
        }
        let px = pp.preprocess(&rgb, sw, sh);
        assert_eq!(px.len(), 3 * 896 * 896);
        assert!(px.iter().all(|v| v.is_finite()));
        // green channel is constant 128 → normalized value is uniform across the plane.
        let g_plane = &px[896 * 896..2 * 896 * 896];
        let g0 = g_plane[0];
        assert!(
            g_plane.iter().all(|&v| (v - g0).abs() < 1e-4),
            "constant channel stays constant"
        );
        assert!(
            (g0 - (128.0 / 255.0 - pp.mean[1]) / pp.std[1]).abs() < 1e-4,
            "normalization formula"
        );
    }

    /// Load the REAL Gemma-3 vision tower from the mmproj GGUF and run a forward pass on a
    /// dummy normalized image — output must be [num_patches, hidden] = [4096, 1152], all
    /// finite + non-degenerate. Skips if the GGUF isn't present (~851MB, not committed).
    ///
    /// Ignored by default: a full SigLIP forward on the CPU takes minutes in a debug build
    /// (it ran 18 minutes at 12 cores inside `cargo test --workspace` on 2026-08-30). Run it
    /// deliberately: `cargo test --release -p arf-core vision_encoder_forward_real_gguf -- --ignored`.
    #[test]
    #[ignore = "minutes on the CPU in debug; run with --release -- --ignored"]
    fn vision_encoder_forward_real_gguf() {
        let path = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../models/gemma-3-4b-vision/mmproj-F16.gguf"
        ));
        if !path.exists() {
            eprintln!("SKIP: mmproj GGUF not found");
            return;
        }
        let cfg = VisionConfig::default();
        let g = LazyGguf::open(path, &crate::config::ModelConfig::gemma3_4b()).expect("open");
        let enc = VisionEncoder::load(&g, cfg).expect("load vision encoder");

        // a synthetic normalized image [3, 896, 896] — a smooth gradient so attention has
        // structure (not a constant, which would collapse).
        let n = cfg.image_size;
        let mut px = vec![0.0f32; 3 * n * n];
        for ch in 0..3 {
            for y in 0..n {
                for x in 0..n {
                    px[ch * n * n + y * n + x] = ((x + y + ch * 50) as f32 / (2 * n) as f32) - 0.5;
                }
            }
        }
        let out = enc.encode(&px);
        assert_eq!(out.len(), cfg.num_patches() * cfg.hidden, "[4096,1152]");
        assert!(out.iter().all(|v| v.is_finite()), "all finite");
        // non-degenerate: a real spread of values (a broken forward tends to collapse).
        let mean = out.iter().sum::<f32>() / out.len() as f32;
        let var = out.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / out.len() as f32;
        assert!(var > 1e-4, "output has variance (not collapsed): var={var}");

        // Phase 3: project → image soft tokens [256, 2560].
        let proj = VisionProjector::load(&g, cfg.hidden, 2560).expect("load projector");
        let soft = proj.project(&out, cfg.grid());
        assert_eq!(soft.len(), 256 * 2560, "[256,2560] soft tokens");
        assert!(soft.iter().all(|v| v.is_finite()));
        let pm = soft.iter().sum::<f32>() / soft.len() as f32;
        let pv = soft.iter().map(|v| (v - pm).powi(2)).sum::<f32>() / soft.len() as f32;
        assert!(pv > 1e-4, "soft tokens have variance: var={pv}");
    }
}
