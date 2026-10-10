//! Qwen3.8 vision encoder (mmproj `clip.projector_type = qwen3vl_merger`), CPU f32.
//!
//! Correctness first: this is a plain transcription of llama.cpp's `clip_graph_qwen3vl`
//! (`tools/mtmd/models/qwen3vl.cpp`) plus its preprocessing (`mtmd-image.cpp`
//! `mtmd_image_preprocessor_dyn_size`), checked against an independent numpy reference
//! (`scripts/qwen_vision_ref.py`). The GPU port (`arf_gpu::gpu::metal::qwen_vision`, 2026-09-27)
//! builds its inputs with this file's `patches_in_merge_order` / `pos_table_resized` /
//! `rope_tables` and is gated against [`QwenVisionEncoder::encode`]; this encoder stays the
//! reference and the server's fallback (`ARF_VISION_CPU=1` selects it).
//!
//! The mmproj on disk (ISTA-DASLab `mmproj-Qwen3.8-27B-BF16.gguf`, 334 tensors, inventoried
//! 2026-09-27):
//!
//! | tensor | ggml dims | type |
//! |---|---|---|
//! | `v.patch_embd.weight`, `v.patch_embd.weight.1` | `[16,16,3,1152]` | F32 (two temporal taps) |
//! | `v.patch_embd.bias` | `[1152]` | F32 |
//! | `v.position_embd.weight` | `[1152, 2304]` (a 48x48 grid) | F32 |
//! | `v.blk.{0..26}.attn_qkv.weight` / `.bias` | `[1152, 3456]` / `[3456]` | BF16 / F32 |
//! | `v.blk.N.attn_out.weight` / `.bias` | `[1152, 1152]` / `[1152]` | BF16 / F32 |
//! | `v.blk.N.ffn_up.weight` / `.bias` | `[1152, 4304]` / `[4304]` | BF16 / F32 |
//! | `v.blk.N.ffn_down.weight` / `.bias` | `[4304, 1152]` / `[1152]` | BF16 / F32 |
//! | `v.blk.N.ln1` / `ln2` `.weight` `.bias` | `[1152]` | F32 |
//! | `v.post_ln.weight` / `.bias` | `[1152]` | F32 |
//! | `mm.0.weight` / `.bias` | `[4608, 4608]` / `[4608]` | BF16 / F32 |
//! | `mm.2.weight` / `.bias` | `[4608, 5120]` / `[5120]` | BF16 / F32 |
//!
//! Metadata: `image_size 768`, `patch_size 16`, `embedding_length 1152`,
//! `feed_forward_length 4304`, `block_count 27`, `head_count 16`, `layer_norm_epsilon 1e-6`,
//! `image_mean = image_std = [0.5; 3]`, `spatial_merge_size 2`, `projection_dim 5120`,
//! `use_gelu true`, `is_deepstack_layers` all false (no deepstack tensors — refused if present).
//!
//! Forward, per llama.cpp:
//! 1. patch embed = conv(w0) + conv(w1) over the SAME frame (a still image fills both temporal
//!    taps), + bias, emitted in 2x2-MERGE order (each 4 consecutive patches are one merged
//!    token: (0,0),(0,1),(1,0),(1,1); merged cells row-major). VIDEO (2026-09-27,
//!    [`QwenVisionEncoder::encode_frame_pair`]): conv(w0) over frame 2k + conv(w1) over frame
//!    2k+1, everything after it unchanged — see `qwen_video` for why a pair is encoded alone;
//! 2. + the learned 48x48 position table, bilinearly resized (align-corners) to the patch
//!    grid, in the same merge order;
//! 3. 27 pre-LN blocks: LN1 -> fused qkv (+bias) -> 2-D vision rope -> full (non-causal)
//!    attention, scale 1/sqrt(72) -> out proj (+bias) -> residual; LN2 -> up (+bias) ->
//!    GELU(tanh) -> down (+bias) -> residual;
//! 4. post-LN, then 4 consecutive rows -> one 4608 row -> mm.0 (+bias) -> GELU -> mm.2 (+bias):
//!    `[n_patches/4, 5120]`, already in the LM's embedding space.
//!
//! Vision rope: head_dim 72, rotary over all 72 as NeoX pairs `(d, d + 36)`; pairs 0..17 rotate
//! by the patch ROW, 18..35 by the patch COLUMN, each at `10000^(-2k/36)`, k restarting at 0 for
//! the column half (ggml `GGML_ROPE_TYPE_VISION` with sections `[18,18,18,18]`, indep sections).
//!
//! Two deliberate choices where references differ (both documented, neither measurable in
//! text output as far as is known — NOT MEASURED):
//! - merger GELU: HF `Qwen3VLVisionPatchMerger` uses `nn.GELU()` (erf) and so does another engine
//!   (`Vision.mm`); llama.cpp uses `ggml_gelu` (tanh). Arf follows the checkpoint's own erf.
//!   The ViT MLP is tanh everywhere (`gelu_pytorch_tanh`).
//! - resize: bicubic, Pillow-exact (a = -0.5, llama.cpp's port), STRETCHED to the aligned size
//!   as HF and another engine do. llama.cpp pads (`PAD_CEIL`); for an input whose sides are already
//!   multiples of 32 and inside the pixel budget, all three agree bit-for-bit on the pixels.
//!   Pixel budget: min 64 merged tokens (HF / another engine `shortest_edge 65536`), max 1024 by
//!   default on the CPU (`ARF_QWEN_VISION_MAX_TOKENS` raises it; llama.cpp and another engine use 4096).

use std::path::Path;

use crate::model::gguf::LazyGguf;
use crate::tensor::matmul::matmul_nt_bf16;
use crate::{ArfError, Result};

/// Config read from the mmproj metadata.
#[derive(Debug, Clone)]
pub struct QwenVisionConfig {
    pub patch: usize,    // 16
    pub merge: usize,    // 2
    pub hidden: usize,   // 1152
    pub ffn: usize,      // 4304
    pub layers: usize,   // 27
    pub heads: usize,    // 16
    pub eps: f32,        // 1e-6
    pub proj_dim: usize, // 5120 (the LM hidden size)
    pub pos_grid: usize, // 48 (sqrt of the position table's rows)
    pub mean: [f32; 3],
    pub std: [f32; 3],
    /// Pixel budget of the resized image (llama.cpp `image_min/max_pixels`).
    pub min_pixels: usize,
    pub max_pixels: usize,
}

impl QwenVisionConfig {
    /// Side alignment of the resized image: one merged token = `patch * merge` pixels (32).
    pub fn align(&self) -> usize {
        self.patch * self.merge
    }
    pub fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }

    /// Default pixel budget for `tokens_min..=tokens_max` merged tokens.
    fn budget(&mut self, tokens_min: usize, tokens_max: usize) {
        let area = self.align() * self.align();
        self.min_pixels = tokens_min * area;
        self.max_pixels = tokens_max * area;
    }

    /// Read from an mmproj opened with [`LazyGguf::open_raw`]. Errors unless the projector is
    /// `qwen3vl_merger` (the only one this module implements).
    pub fn from_gguf(g: &LazyGguf) -> Result<Self> {
        let proj = g
            .get_metadata_string("clip.projector_type")
            .unwrap_or_default();
        if proj != "qwen3vl_merger" {
            return Err(ArfError::model_load(format!(
                "qwen vision: projector_type '{proj}' is not qwen3vl_merger"
            )));
        }
        let u = |k: &str, d: usize| g.get_metadata_u32(k).map(|v| v as usize).unwrap_or(d);
        let to3 = |v: Option<Vec<f32>>| -> [f32; 3] {
            match v {
                Some(a) if a.len() >= 3 => [a[0], a[1], a[2]],
                _ => [0.5; 3],
            }
        };
        let pos_rows = g
            .shape_raw("v.position_embd.weight")
            .and_then(|s| s.get(1).copied())
            .ok_or_else(|| ArfError::model_load("qwen vision: no v.position_embd.weight"))?;
        let pos_grid = (pos_rows as f64).sqrt().round() as usize;
        if pos_grid * pos_grid != pos_rows {
            return Err(ArfError::model_load(format!(
                "qwen vision: position table has {pos_rows} rows, not a square grid"
            )));
        }
        let mut c = QwenVisionConfig {
            patch: u("clip.vision.patch_size", 16),
            merge: u("clip.vision.spatial_merge_size", 2),
            hidden: u("clip.vision.embedding_length", 1152),
            ffn: u("clip.vision.feed_forward_length", 4304),
            layers: u("clip.vision.block_count", 27),
            heads: u("clip.vision.attention.head_count", 16),
            eps: g
                .get_metadata_f32("clip.vision.attention.layer_norm_epsilon")
                .unwrap_or(1e-6),
            proj_dim: u("clip.vision.projection_dim", 5120),
            pos_grid,
            mean: to3(g.get_metadata_f32_array("clip.vision.image_mean")),
            std: to3(g.get_metadata_f32_array("clip.vision.image_std")),
            min_pixels: 0,
            max_pixels: 0,
        };
        if c.merge != 2 {
            return Err(ArfError::model_load(format!(
                "qwen vision: spatial_merge_size {} (only 2 is implemented)",
                c.merge
            )));
        }
        let max_tokens = std::env::var("ARF_QWEN_VISION_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1024usize)
            .max(64);
        c.budget(64, max_tokens);
        Ok(c)
    }
}

/// A preprocessed image: normalized pixels `[3, height, width]` and its patch grid.
#[derive(Debug, Clone)]
pub struct QwenImage {
    pub pixels: Vec<f32>,
    pub width: usize,
    pub height: usize,
    /// Patches per side (width / 16, height / 16) — both even.
    pub grid_w: usize,
    pub grid_h: usize,
}

impl QwenImage {
    /// Merged-token grid `(rows, cols)` — the `<|image_pad|>` layout the LM sees.
    pub fn token_grid(&self) -> (usize, usize) {
        (self.grid_h / 2, self.grid_w / 2)
    }
    pub fn n_tokens(&self) -> usize {
        (self.grid_h / 2) * (self.grid_w / 2)
    }
}

/// Target size `(w, h)` for an input `w x h`: sides rounded to multiples of `align`, then
/// scaled into `[min_pixels, max_pixels]` keeping the aspect ratio — llama.cpp's
/// `img_tool::calc_size_preserved_ratio` (the transformers `smart_resize`), in its f32 math.
pub fn smart_resize(
    width: usize,
    height: usize,
    align: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> (usize, usize) {
    if width == 0 || height == 0 {
        return (0, 0);
    }
    let f = align as f32;
    let round_by = |x: f32| ((x / f).round() as i64 * align as i64) as usize;
    let ceil_by = |x: f32| ((x / f).ceil() as i64 * align as i64) as usize;
    let floor_by = |x: f32| ((x / f).floor() as i64 * align as i64) as usize;
    let (wf, hf) = (width as f32, height as f32);
    let mut w_bar = align.max(round_by(wf));
    let mut h_bar = align.max(round_by(hf));
    if max_pixels > 0 && h_bar * w_bar > max_pixels {
        let beta = (hf * width as f32 / max_pixels as f32).sqrt();
        h_bar = align.max(floor_by(hf / beta));
        w_bar = align.max(floor_by(wf / beta));
    } else if min_pixels > 0 && h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f32 / (hf * width as f32)).sqrt();
        h_bar = ceil_by(hf * beta);
        w_bar = ceil_by(wf * beta);
    }
    (w_bar, h_bar)
}

/// Pillow-compatible separable BICUBIC resize (a = -0.5, 22-bit fixed point, horizontal pass
/// then vertical, u8 intermediate) of `rgb` (`[h, w, 3]`). A port of llama.cpp's
/// `img_tool::resize_pillow`, itself adapted from Pillow's `Resample.c`.
pub fn resize_bicubic_pillow(rgb: &[u8], sw: usize, sh: usize, tw: usize, th: usize) -> Vec<u8> {
    assert_eq!(rgb.len(), sw * sh * 3, "rgb must be [h, w, 3]");
    const PRECISION_BITS: u32 = 32 - 8 - 2;
    const SUPPORT: f64 = 2.0;
    let filter = |x: f64| -> f64 {
        let x = x.abs();
        const A: f64 = -0.5;
        if x < 1.0 {
            ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
        } else if x < 2.0 {
            (((x - 5.0) * x + 8.0) * x - 4.0) * A
        } else {
            0.0
        }
    };
    // (bounds [xmin, xcnt] per output, fixed-point weights [out * ksize])
    let precompute = |in_size: usize, out_size: usize| -> (Vec<(usize, usize)>, Vec<i64>, usize) {
        let scale = in_size as f64 / out_size as f64;
        let filterscale = scale.max(1.0);
        let support = SUPPORT * filterscale;
        let ksize = support.ceil() as usize * 2 + 1;
        let mut bounds = Vec::with_capacity(out_size);
        let mut weights = vec![0i64; out_size * ksize];
        let fxp = (1u64 << PRECISION_BITS) as f64;
        for xx in 0..out_size {
            let center = (xx as f64 + 0.5) * scale;
            let ss = 1.0 / filterscale;
            let xmin = ((center - support + 0.5) as i64).max(0) as usize;
            let xmax = ((center + support + 0.5) as i64).min(in_size as i64) as usize - xmin;
            let mut pre = vec![0.0f64; ksize];
            let mut ww = 0.0;
            for (x, p) in pre.iter_mut().enumerate().take(xmax) {
                let w = filter((x as f64 + xmin as f64 - center + 0.5) * ss);
                *p = w;
                ww += w;
            }
            if ww != 0.0 {
                for p in pre.iter_mut().take(xmax) {
                    *p /= ww;
                }
            }
            for (k, p) in pre.iter().enumerate() {
                let r = p * fxp + if *p < 0.0 { -0.5 } else { 0.5 };
                weights[xx * ksize + k] = r as i32 as i64;
            }
            bounds.push((xmin, xmax));
        }
        (bounds, weights, ksize)
    };
    let clip8 = |v: i64| -> u8 { (v >> PRECISION_BITS).clamp(0, 255) as u8 };
    let round_bias: i64 = 1 << (PRECISION_BITS - 1);

    let mut cur = rgb.to_vec();
    let mut cw = sw;
    if tw != sw {
        let (b, w, k) = precompute(sw, tw);
        let mut out = vec![0u8; tw * sh * 3];
        for y in 0..sh {
            let src = &cur[y * sw * 3..(y + 1) * sw * 3];
            for (xx, &(xmin, xcnt)) in b.iter().enumerate() {
                let kk = &w[xx * k..xx * k + k];
                let mut acc = [round_bias; 3];
                for x in 0..xcnt {
                    let p = &src[(xmin + x) * 3..(xmin + x) * 3 + 3];
                    for c in 0..3 {
                        acc[c] += p[c] as i64 * kk[x];
                    }
                }
                for c in 0..3 {
                    out[(y * tw + xx) * 3 + c] = clip8(acc[c]);
                }
            }
        }
        cur = out;
        cw = tw;
    }
    if th != sh {
        let (b, w, k) = precompute(sh, th);
        let row = cw * 3;
        let mut out = vec![0u8; row * th];
        for (yy, &(ymin, ycnt)) in b.iter().enumerate() {
            let kk = &w[yy * k..yy * k + k];
            let mut acc = vec![round_bias; row];
            for y in 0..ycnt {
                let src = &cur[(ymin + y) * row..(ymin + y + 1) * row];
                let wy = kk[y];
                for (a, &s) in acc.iter_mut().zip(src) {
                    *a += s as i64 * wy;
                }
            }
            for (o, a) in out[yy * row..(yy + 1) * row].iter_mut().zip(&acc) {
                *o = clip8(*a);
            }
        }
        cur = out;
    }
    cur
}

/// Resize to the aligned budgeted size and normalize: `(px/255 - mean) / std`, channel-major.
pub fn preprocess(cfg: &QwenVisionConfig, rgb: &[u8], w: usize, h: usize) -> QwenImage {
    let (tw, th) = smart_resize(w, h, cfg.align(), cfg.min_pixels, cfg.max_pixels);
    preprocess_to(cfg, rgb, w, h, tw, th)
}

/// [`preprocess`] at an explicit target size `tw x th` (multiples of `2 * patch`) — the video path,
/// whose size comes from the whole video's budget (`qwen_video::video_frame_size`), not the image
/// budget.
pub fn preprocess_to(
    cfg: &QwenVisionConfig,
    rgb: &[u8],
    w: usize,
    h: usize,
    tw: usize,
    th: usize,
) -> QwenImage {
    let resized = if (tw, th) == (w, h) {
        rgb.to_vec()
    } else {
        resize_bicubic_pillow(rgb, w, h, tw, th)
    };
    let mut pixels = vec![0.0f32; 3 * tw * th];
    for i in 0..tw * th {
        for c in 0..3 {
            let v = resized[i * 3 + c] as f32 / 255.0;
            pixels[c * tw * th + i] = (v - cfg.mean[c]) / cfg.std[c];
        }
    }
    QwenImage {
        pixels,
        width: tw,
        height: th,
        grid_w: tw / cfg.patch,
        grid_h: th / cfg.patch,
    }
}

/// One ViT block. Matrices are bf16 bit patterns, row-major `[out, in]` (the mmproj stores them
/// bf16; widening in the dot is exact, so this is the f32 forward at half the memory).
pub struct QwenVisionLayer {
    pub ln1_w: Vec<f32>,
    pub ln1_b: Vec<f32>,
    pub qkv_w: Vec<u16>, // [3*hidden, hidden]: rows 0..h = Q, h..2h = K, 2h..3h = V
    pub qkv_b: Vec<f32>,
    pub o_w: Vec<u16>, // [hidden, hidden]
    pub o_b: Vec<f32>,
    pub ln2_w: Vec<f32>,
    pub ln2_b: Vec<f32>,
    pub up_w: Vec<u16>, // [ffn, hidden]
    pub up_b: Vec<f32>,
    pub down_w: Vec<u16>, // [hidden, ffn]
    pub down_b: Vec<f32>,
}

/// All encoder weights. Public so a test (or a GPU port) can build one without a file.
pub struct QwenVisionWeights {
    /// `w0 + w1` (the two temporal taps summed), `[hidden, 3*patch*patch]`, inner order
    /// `[channel][ky][kx]` (ggml's `[kx, ky, c, out]` conv layout read row-major).
    pub patch_w: Vec<f32>,
    /// The two taps side by side for VIDEO, `[hidden, 2*3*patch*patch]`: row `o` is `[w0 row o |
    /// w1 row o]`, so a frame pair's patch `[frame 2k | frame 2k+1]` dotted with it is
    /// `w0.x0 + w1.x1` — the Conv3d over two different frames. `w0` is `v.patch_embd.weight`
    /// (HF temporal index 0), `w1` is `v.patch_embd.weight.1` (llama.cpp `conversion/qwen3vl.py`
    /// slices `[:, :, 0]` and `[:, :, 1]`). Images keep using the pre-summed `patch_w`, so the
    /// image path is bit-for-bit what it was.
    pub patch_w_pair: Vec<f32>,
    pub patch_b: Vec<f32>,
    /// `[pos_grid*pos_grid, hidden]`, row-major over the grid (row = y*grid + x).
    pub pos_embd: Vec<f32>,
    pub layers: Vec<QwenVisionLayer>,
    pub post_ln_w: Vec<f32>,
    pub post_ln_b: Vec<f32>,
    pub mm0_w: Vec<u16>, // [4*hidden, 4*hidden]
    pub mm0_b: Vec<f32>,
    pub mm2_w: Vec<u16>, // [proj_dim, 4*hidden]
    pub mm2_b: Vec<f32>,
}

pub struct QwenVisionEncoder {
    pub cfg: QwenVisionConfig,
    pub w: QwenVisionWeights,
}

/// True when `path` is an mmproj whose projector this module implements.
pub fn is_qwen3vl_mmproj(path: &Path) -> bool {
    LazyGguf::open_raw(path)
        .ok()
        .and_then(|g| g.get_metadata_string("clip.projector_type"))
        .is_some_and(|p| p == "qwen3vl_merger")
}

impl QwenVisionEncoder {
    /// Load every tensor from an mmproj GGUF. ~0.93 GB resident (bf16 matrices).
    pub fn load(path: &Path) -> Result<Self> {
        let g = LazyGguf::open_raw(path)?;
        let cfg = QwenVisionConfig::from_gguf(&g)?;
        if g.tensor_names().any(|n| n.starts_with("v.deepstack")) {
            return Err(ArfError::model_load(
                "qwen vision: deepstack layers are not implemented",
            ));
        }
        let f = |n: &str| -> Result<Vec<f32>> {
            g.get_f32_raw(n)
                .ok_or_else(|| ArfError::model_load(format!("qwen vision: missing {n}")))
        };
        // The file's matrices are BF16, so truncating the dequantized f32 back to its top 16
        // bits is lossless. (An F32/F16 mmproj would round here — acceptable, and loud below.)
        let m = |n: &str| -> Result<Vec<u16>> {
            if g.raw_ggml_type_id(n) != Some(30) {
                eprintln!("[qwen-vision] {n} is not BF16 in the mmproj; rounding to bf16");
            }
            Ok(f(n)?
                .iter()
                .map(|&x| crate::tensor::dtype::f32_to_bf16(x))
                .collect())
        };
        let w0 = f("v.patch_embd.weight")?;
        let w1 = f("v.patch_embd.weight.1")?;
        let patch_w: Vec<f32> = w0.iter().zip(&w1).map(|(a, b)| a + b).collect();
        let patch_w_pair = interleave_taps(&w0, &w1, 3 * cfg.patch * cfg.patch);
        let layers = (0..cfg.layers)
            .map(|i| {
                let p = format!("v.blk.{i}");
                Ok(QwenVisionLayer {
                    ln1_w: f(&format!("{p}.ln1.weight"))?,
                    ln1_b: f(&format!("{p}.ln1.bias"))?,
                    qkv_w: m(&format!("{p}.attn_qkv.weight"))?,
                    qkv_b: f(&format!("{p}.attn_qkv.bias"))?,
                    o_w: m(&format!("{p}.attn_out.weight"))?,
                    o_b: f(&format!("{p}.attn_out.bias"))?,
                    ln2_w: f(&format!("{p}.ln2.weight"))?,
                    ln2_b: f(&format!("{p}.ln2.bias"))?,
                    up_w: m(&format!("{p}.ffn_up.weight"))?,
                    up_b: f(&format!("{p}.ffn_up.bias"))?,
                    down_w: m(&format!("{p}.ffn_down.weight"))?,
                    down_b: f(&format!("{p}.ffn_down.bias"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let w = QwenVisionWeights {
            patch_w,
            patch_w_pair,
            patch_b: f("v.patch_embd.bias")?,
            pos_embd: f("v.position_embd.weight")?,
            layers,
            post_ln_w: f("v.post_ln.weight")?,
            post_ln_b: f("v.post_ln.bias")?,
            mm0_w: m("mm.0.weight")?,
            mm0_b: f("mm.0.bias")?,
            mm2_w: m("mm.2.weight")?,
            mm2_b: f("mm.2.bias")?,
        };
        Self::from_weights(cfg, w)
    }

    /// Build from in-memory weights, checking every shape against `cfg`.
    pub fn from_weights(cfg: QwenVisionConfig, w: QwenVisionWeights) -> Result<Self> {
        let (d, f, pd) = (cfg.hidden, cfg.ffn, 3 * cfg.patch * cfg.patch);
        let chk = |what: &str, got: usize, want: usize| -> Result<()> {
            if got == want {
                Ok(())
            } else {
                Err(ArfError::model_load(format!(
                    "qwen vision: {what} has {got} elements, expected {want}"
                )))
            }
        };
        if !cfg.hidden.is_multiple_of(cfg.heads) || !(cfg.hidden / cfg.heads).is_multiple_of(4) {
            return Err(ArfError::model_load(
                "qwen vision: head_dim must be a multiple of 4",
            ));
        }
        chk("patch_w", w.patch_w.len(), d * pd)?;
        chk("patch_w_pair", w.patch_w_pair.len(), d * 2 * pd)?;
        chk("patch_b", w.patch_b.len(), d)?;
        chk(
            "pos_embd",
            w.pos_embd.len(),
            cfg.pos_grid * cfg.pos_grid * d,
        )?;
        chk("layers", w.layers.len(), cfg.layers)?;
        for l in &w.layers {
            chk("qkv_w", l.qkv_w.len(), 3 * d * d)?;
            chk("qkv_b", l.qkv_b.len(), 3 * d)?;
            chk("o_w", l.o_w.len(), d * d)?;
            chk("up_w", l.up_w.len(), f * d)?;
            chk("up_b", l.up_b.len(), f)?;
            chk("down_w", l.down_w.len(), d * f)?;
            for v in [&l.ln1_w, &l.ln1_b, &l.ln2_w, &l.ln2_b, &l.o_b, &l.down_b] {
                chk("layer vector", v.len(), d)?;
            }
        }
        chk("mm0_w", w.mm0_w.len(), 16 * d * d)?;
        chk("mm0_b", w.mm0_b.len(), 4 * d)?;
        chk("mm2_w", w.mm2_w.len(), cfg.proj_dim * 4 * d)?;
        chk("mm2_b", w.mm2_b.len(), cfg.proj_dim)?;
        Ok(QwenVisionEncoder { cfg, w })
    }

    /// Decode-side entry: raw RGB (`[h, w, 3]` u8) -> `[n_tokens, proj_dim]` embeddings and
    /// the merged token grid `(rows, cols)`.
    pub fn encode_rgb(&self, rgb: &[u8], w: usize, h: usize) -> (Vec<f32>, (usize, usize)) {
        let img = preprocess(&self.cfg, rgb, w, h);
        let out = self.encode(&img);
        (out, img.token_grid())
    }

    /// Patches in merge order (see [`patches_in_merge_order`]).
    fn patches(&self, img: &QwenImage) -> (Vec<f32>, Vec<(usize, usize)>) {
        patches_in_merge_order(&self.cfg, img)
    }

    /// The position table resized to the patch grid (see [`pos_table_resized`]).
    fn pos_table(&self, gw: usize, gh: usize) -> Vec<f32> {
        pos_table_resized(&self.cfg, &self.w.pos_embd, gw, gh)
    }

    /// Full forward: `[n_patches/4, proj_dim]`.
    pub fn encode(&self, img: &QwenImage) -> Vec<f32> {
        let pd = 3 * self.cfg.patch * self.cfg.patch;
        check_grid(img);
        let (patches, rc) = self.patches(img);
        // patch embed with both temporal taps pre-summed: the still image fills both
        self.forward(&patches, pd, &self.w.patch_w, &rc, img.grid_w)
    }

    /// One VIDEO frame pair `(a, b)` = frames `(2k, 2k+1)`: the image forward with the Conv3d's
    /// taps applied to two frames (`a` through `w0`, `b` through `w1`). A pair is encoded on its
    /// own — the reference's ViT attention is per frame pair (`qwen_video` module docs) — and
    /// `encode_frame_pair(x, x)` equals `encode(x)` up to f32 summation order
    /// (`video_pair_of_one_image_equals_the_image`).
    pub fn encode_frame_pair(&self, a: &QwenImage, b: &QwenImage) -> Vec<f32> {
        let pd = 3 * self.cfg.patch * self.cfg.patch;
        check_grid(a);
        assert_eq!(
            (a.width, a.height),
            (b.width, b.height),
            "qwen vision: a frame pair must share one size"
        );
        let (patches, rc) = patches_pair_in_merge_order(&self.cfg, a, b);
        self.forward(&patches, 2 * pd, &self.w.patch_w_pair, &rc, a.grid_w)
    }

    /// The forward from unfolded patches (`[n, k]` in merge order, `rc` their grid cells) through
    /// the merger, `patch_w` being `[hidden, k]`.
    fn forward(
        &self,
        patches: &[f32],
        k: usize,
        patch_w: &[f32],
        rc: &[(usize, usize)],
        grid_w: usize,
    ) -> Vec<f32> {
        let c = &self.cfg;
        let d = c.hidden;
        let n = rc.len();
        let grid_h = n / grid_w;
        // 1) patch embed + bias.
        let mut h = crate::tensor::matmul_nt(patches, n, k, patch_w, d);
        add_bias(&mut h, &self.w.patch_b);
        // 2) + position table, gathered into merge order.
        let pos = self.pos_table(grid_w, grid_h);
        for (i, &(py, px)) in rc.iter().enumerate() {
            let src = &pos[(py * grid_w + px) * d..(py * grid_w + px + 1) * d];
            for (a, b) in h[i * d..(i + 1) * d].iter_mut().zip(src) {
                *a += b;
            }
        }
        vsum("inp_pos_emb", &h);
        // Vision rope tables, per patch: [n, head_dim/2] cos and sin.
        let (cos, sin) = self.rope_tables(rc);
        for (li, l) in self.w.layers.iter().enumerate() {
            self.block(&mut h, l, n, &cos, &sin);
            if li == 0 {
                vsum("layer_out 0", &h);
            }
        }
        vsum("layer_out last", &h);
        layer_norm(&mut h, &self.w.post_ln_w, &self.w.post_ln_b, d, c.eps);
        // 4) merger: rows are already in merge order, so [n, d] IS [n/4, 4d].
        let (m, d4) = (n / 4, 4 * d);
        let mut x = matmul_nt_bf16(&h, m, d4, &self.w.mm0_w, d4);
        add_bias(&mut x, &self.w.mm0_b);
        for v in x.iter_mut() {
            *v = gelu_erf(*v);
        }
        let mut out = matmul_nt_bf16(&x, m, d4, &self.w.mm2_w, c.proj_dim);
        add_bias(&mut out, &self.w.mm2_b);
        vsum("embeddings", &out);
        out
    }

    fn rope_tables(&self, rc: &[(usize, usize)]) -> (Vec<f32>, Vec<f32>) {
        rope_tables(&self.cfg, rc)
    }

    fn block(&self, h: &mut [f32], l: &QwenVisionLayer, n: usize, cos: &[f32], sin: &[f32]) {
        let c = &self.cfg;
        let d = c.hidden;
        let mut x = h.to_vec();
        layer_norm(&mut x, &l.ln1_w, &l.ln1_b, d, c.eps);
        let mut qkv = matmul_nt_bf16(&x, n, d, &l.qkv_w, 3 * d);
        add_bias(&mut qkv, &l.qkv_b);
        let hd = c.head_dim();
        let half = hd / 2;
        // rope q and k in place (NeoX pairs (j, j+half) within each head).
        for i in 0..n {
            let (cs, sn) = (
                &cos[i * half..(i + 1) * half],
                &sin[i * half..(i + 1) * half],
            );
            for part in 0..2 {
                for head in 0..c.heads {
                    let base = i * 3 * d + part * d + head * hd;
                    let v = &mut qkv[base..base + hd];
                    for j in 0..half {
                        let (a, b) = (v[j], v[j + half]);
                        v[j] = a * cs[j] - b * sn[j];
                        v[j + half] = a * sn[j] + b * cs[j];
                    }
                }
            }
        }
        let attn = attention(&qkv, n, d, c.heads);
        let mut o = matmul_nt_bf16(&attn, n, d, &l.o_w, d);
        add_bias(&mut o, &l.o_b);
        for (a, b) in h.iter_mut().zip(&o) {
            *a += b;
        }
        let mut y = h.to_vec();
        layer_norm(&mut y, &l.ln2_w, &l.ln2_b, d, c.eps);
        let mut up = matmul_nt_bf16(&y, n, d, &l.up_w, c.ffn);
        add_bias(&mut up, &l.up_b);
        for v in up.iter_mut() {
            *v = gelu_tanh(*v);
        }
        let mut down = matmul_nt_bf16(&up, n, c.ffn, &l.down_w, d);
        add_bias(&mut down, &l.down_b);
        for (a, b) in h.iter_mut().zip(&down) {
            *a += b;
        }
    }
}

/// Patches in merge order, `[n_patches, 3*patch*patch]`, and each patch's (row, col). Each 4
/// consecutive patches are one merged token: (0,0),(0,1),(1,0),(1,1) of a 2x2 cell, cells
/// row-major. Public so the GPU encoder builds its input from the same host code.
pub fn patches_in_merge_order(
    cfg: &QwenVisionConfig,
    img: &QwenImage,
) -> (Vec<f32>, Vec<(usize, usize)>) {
    let (ps, gw, gh) = (cfg.patch, img.grid_w, img.grid_h);
    let (iw, ih) = (img.width, img.height);
    let pd = 3 * ps * ps;
    let mut out = Vec::with_capacity(gw * gh * pd);
    let mut rc = Vec::with_capacity(gw * gh);
    for by in (0..gh).step_by(2) {
        for bx in (0..gw).step_by(2) {
            for dy in 0..2 {
                for dx in 0..2 {
                    let (py, px) = (by + dy, bx + dx);
                    rc.push((py, px));
                    for c in 0..3 {
                        for ky in 0..ps {
                            let row = c * iw * ih + (py * ps + ky) * iw + px * ps;
                            out.extend_from_slice(&img.pixels[row..row + ps]);
                        }
                    }
                }
            }
        }
    }
    (out, rc)
}

/// A video frame pair's patches in merge order, `[n_patches, 2*3*patch*patch]`: each row is frame
/// `a`'s patch followed by frame `b`'s at the same cell (the layout
/// [`QwenVisionWeights::patch_w_pair`] dots against). The cells and their order are
/// [`patches_in_merge_order`]'s. Public so the GPU encoder builds its input from the same code.
pub fn patches_pair_in_merge_order(
    cfg: &QwenVisionConfig,
    a: &QwenImage,
    b: &QwenImage,
) -> (Vec<f32>, Vec<(usize, usize)>) {
    let pd = 3 * cfg.patch * cfg.patch;
    let (pa, rc) = patches_in_merge_order(cfg, a);
    let (pb, _) = patches_in_merge_order(cfg, b);
    let mut out = Vec::with_capacity(pa.len() * 2);
    for (x, y) in pa.chunks(pd).zip(pb.chunks(pd)) {
        out.extend_from_slice(x);
        out.extend_from_slice(y);
    }
    (out, rc)
}

/// `[hidden, pd]` taps `w0`, `w1` -> `[hidden, 2*pd]` rows `[w0 row | w1 row]`.
pub fn interleave_taps(w0: &[f32], w1: &[f32], pd: usize) -> Vec<f32> {
    assert_eq!(w0.len(), w1.len(), "the two temporal taps differ in size");
    let mut out = Vec::with_capacity(w0.len() * 2);
    for (x, y) in w0.chunks(pd).zip(w1.chunks(pd)) {
        out.extend_from_slice(x);
        out.extend_from_slice(y);
    }
    out
}

fn check_grid(img: &QwenImage) {
    assert!(
        img.grid_w.is_multiple_of(2)
            && img.grid_h.is_multiple_of(2)
            && img.grid_w > 0
            && img.grid_h > 0,
        "qwen vision: patch grid {}x{} must be even",
        img.grid_w,
        img.grid_h
    );
}

/// The position table `pos_embd` (`[pos_grid*pos_grid, hidden]`) resized to `gw x gh` (bilinear,
/// align-corners — ggml `ggml_interpolate(.., GGML_SCALE_MODE_BILINEAR |
/// GGML_SCALE_FLAG_ALIGN_CORNERS)`), raster order `[gh*gw, hidden]`.
pub fn pos_table_resized(
    cfg: &QwenVisionConfig,
    pos_embd: &[f32],
    gw: usize,
    gh: usize,
) -> Vec<f32> {
    let (n, d) = (cfg.pos_grid, cfg.hidden);
    let src = pos_embd;
    if gw == n && gh == n {
        return src.to_vec();
    }
    // ggml: sf = (ne-1)/(ne_src-1) when both > 1 (else the plain ratio), pixel_offset 0.
    let sf = |out: usize| -> f32 {
        if out > 1 && n > 1 {
            (out - 1) as f32 / (n - 1) as f32
        } else {
            out as f32 / n as f32
        }
    };
    let (sf0, sf1) = (sf(gw), sf(gh));
    let mut out = vec![0.0f32; gw * gh * d];
    for i1 in 0..gh {
        let y = i1 as f32 / sf1;
        let y0 = (y.floor() as i64).clamp(0, n as i64 - 1) as usize;
        let y1 = (y.floor() as i64 + 1).clamp(0, n as i64 - 1) as usize;
        let dy = (y - y0 as f32).clamp(0.0, 1.0);
        for i0 in 0..gw {
            let x = i0 as f32 / sf0;
            let x0 = (x.floor() as i64).clamp(0, n as i64 - 1) as usize;
            let x1 = (x.floor() as i64 + 1).clamp(0, n as i64 - 1) as usize;
            let dx = (x - x0 as f32).clamp(0.0, 1.0);
            let (a, b) = (&src[(y0 * n + x0) * d..], &src[(y0 * n + x1) * d..]);
            let (c, e) = (&src[(y1 * n + x0) * d..], &src[(y1 * n + x1) * d..]);
            let o = &mut out[(i1 * gw + i0) * d..(i1 * gw + i0 + 1) * d];
            for k in 0..d {
                o[k] = a[k] * (1.0 - dx) * (1.0 - dy)
                    + b[k] * dx * (1.0 - dy)
                    + c[k] * (1.0 - dx) * dy
                    + e[k] * dx * dy;
            }
        }
    }
    out
}

/// Per-patch cos/sin for the `head_dim/2` rotary pairs (`[n, head_dim/2]` each): pairs 0..17 at
/// the patch row, 18..35 at the column, frequency `theta_scale^k` with k restarting at the column
/// half. The angle is built by repeated f32 multiplication exactly as ggml's
/// `ggml_mrope_cache_init` does. The GPU encoder uploads these same tables, so its angles are
/// bit-identical to the CPU's.
pub fn rope_tables(cfg: &QwenVisionConfig, rc: &[(usize, usize)]) -> (Vec<f32>, Vec<f32>) {
    let half = cfg.head_dim() / 2; // 36 pairs
    let quarter = half / 2; // 18 per axis
    let theta_scale = 10000f32.powf(-2.0 / half as f32);
    let mut cos = vec![0.0f32; rc.len() * half];
    let mut sin = vec![0.0f32; rc.len() * half];
    for (i, &(py, px)) in rc.iter().enumerate() {
        let mut t_row = py as f32;
        let mut t_col = px as f32;
        for k in 0..half {
            if k == quarter {
                t_col = px as f32; // indep sections: reset at the section boundary
            }
            let theta = if k < quarter { t_row } else { t_col };
            cos[i * half + k] = theta.cos();
            sin[i * half + k] = theta.sin();
            t_row *= theta_scale;
            t_col *= theta_scale;
        }
    }
    (cos, sin)
}

/// Full bidirectional multi-head attention over `qkv` rows `[q | k | v]` (`[n, 3d]`), scale
/// `1/sqrt(head_dim)`. Rows are split across threads; each writes its own output rows.
fn attention(qkv: &[f32], n: usize, d: usize, heads: usize) -> Vec<f32> {
    let hd = d / heads;
    let scale = 1.0 / (hd as f32).sqrt();
    // K per head, contiguous: kh[head][j][hd].
    let mut kh = vec![0.0f32; heads * n * hd];
    for j in 0..n {
        for head in 0..heads {
            let src = &qkv[j * 3 * d + d + head * hd..j * 3 * d + d + (head + 1) * hd];
            kh[(head * n + j) * hd..(head * n + j + 1) * hd].copy_from_slice(src);
        }
    }
    let mut out = vec![0.0f32; n * d];
    let threads = std::thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(1)
        .min(n.max(1));
    let rows_per = n.div_ceil(threads);
    std::thread::scope(|s| {
        for (t, chunk) in out.chunks_mut(rows_per * d).enumerate() {
            let kh = &kh;
            s.spawn(move || {
                let mut scores = vec![0.0f32; n];
                for (r, orow) in chunk.chunks_mut(d).enumerate() {
                    let i = t * rows_per + r;
                    for head in 0..heads {
                        let q = &qkv[i * 3 * d + head * hd..i * 3 * d + (head + 1) * hd];
                        let mut mx = f32::NEG_INFINITY;
                        for (j, sc) in scores.iter_mut().enumerate() {
                            let k = &kh[(head * n + j) * hd..(head * n + j + 1) * hd];
                            let mut acc = 0.0f32;
                            for e in 0..hd {
                                acc += q[e] * k[e];
                            }
                            *sc = acc * scale;
                            mx = mx.max(*sc);
                        }
                        let mut sum = 0.0f32;
                        for sc in scores.iter_mut() {
                            *sc = (*sc - mx).exp();
                            sum += *sc;
                        }
                        let inv = 1.0 / sum;
                        let o = &mut orow[head * hd..(head + 1) * hd];
                        for (j, &w) in scores.iter().enumerate() {
                            let v = &qkv[j * 3 * d + 2 * d + head * hd..];
                            let w = w * inv;
                            for e in 0..hd {
                                o[e] += w * v[e];
                            }
                        }
                    }
                }
            });
        }
    });
    out
}

/// Debug: element sum of a tensor when `ARF_VISION_SUM` is set (same instrument as the Gemma
/// tower), to compare against llama.cpp's `MTMD_DEBUG_GRAPH` node sums.
fn vsum(tag: &str, v: &[f32]) {
    if std::env::var_os("ARF_VISION_SUM").is_some() {
        let s: f64 = v.iter().map(|&x| x as f64).sum();
        eprintln!("[qwen-vsum] {tag:24} sum={s:.6} len={}", v.len());
    }
}

fn add_bias(x: &mut [f32], b: &[f32]) {
    for row in x.chunks_mut(b.len()) {
        for (a, bb) in row.iter_mut().zip(b) {
            *a += bb;
        }
    }
}

/// ggml `ggml_norm` + affine: mean and variance accumulated in f64, `1/sqrt(var + eps)` in f32.
pub(crate) fn layer_norm(x: &mut [f32], w: &[f32], b: &[f32], cols: usize, eps: f32) {
    for row in x.chunks_mut(cols) {
        let mean = (row.iter().map(|&v| v as f64).sum::<f64>() / cols as f64) as f32;
        let mut s2 = 0.0f64;
        for v in row.iter_mut() {
            *v -= mean;
            s2 += (*v * *v) as f64;
        }
        let var = (s2 / cols as f64) as f32;
        let scale = 1.0 / (var + eps).sqrt();
        for ((v, ww), bb) in row.iter_mut().zip(w).zip(b) {
            *v = *v * scale * ww + bb;
        }
    }
}

/// GELU, tanh approximation (`gelu_pytorch_tanh`; ggml `ggml_gelu`).
fn gelu_tanh(x: f32) -> f32 {
    const K: f32 = 0.797_884_6; // sqrt(2/pi)
    0.5 * x * (1.0 + (K * (x + 0.044715 * x * x * x)).tanh())
}

/// GELU, exact: `0.5 x (1 + erf(x / sqrt 2))` (torch `nn.GELU()`), erf in f64.
pub(crate) fn gelu_erf(x: f32) -> f32 {
    let xd = x as f64;
    (0.5 * xd * (1.0 + erf(xd / std::f64::consts::SQRT_2))) as f32
}

/// erf(x) to ~1e-15 relative: a Taylor series for |x| < 2.5, the continued fraction for erfc
/// beyond (both converge fast there). std has no stable `erf`.
fn erf(x: f64) -> f64 {
    let ax = x.abs();
    let r = if ax < 2.5 {
        // erf(x) = 2/sqrt(pi) * sum_n (-1)^n x^(2n+1) / (n! (2n+1))
        let x2 = ax * ax;
        let mut term = ax;
        let mut sum = ax;
        let mut n = 0.0f64;
        loop {
            n += 1.0;
            term *= -x2 / n;
            let add = term / (2.0 * n + 1.0);
            sum += add;
            if add.abs() <= 1e-17 * sum.abs() {
                break;
            }
        }
        sum * 2.0 / std::f64::consts::PI.sqrt()
    } else {
        // erfc(x) = exp(-x^2)/sqrt(pi) * 1/(x + 1/2/(x + 1/(x + 3/2/(x + ...)))) (Lentz)
        let mut f = 0.0f64;
        for k in (1..60).rev() {
            f = (k as f64 / 2.0) / (ax + f);
        }
        1.0 - (-ax * ax).exp() / std::f64::consts::PI.sqrt() / (ax + f)
    };
    if x < 0.0 {
        -r
    } else {
        r
    }
}

/// Synthetic encoders for tests and GPU parity gates: weights drawn from a xorshift32 stream,
/// matrices rounded to bf16. [`synthetic::tiny`] is the encoder behind the numpy-checked golden
/// (`tiny_encoder_matches_numpy_reference`); the SAME generator is in
/// `scripts/qwen_vision_ref.py`, so the draw order below must not change.
pub mod synthetic {
    use super::*;

    /// xorshift32 -> uniform in [-scale, scale], rounded to bf16 where a matrix is bf16.
    struct Rng(u32);
    impl Rng {
        fn next(&mut self) -> f32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.0 = x;
            (x >> 8) as f32 / 16_777_216.0 * 2.0 - 1.0
        }
        fn vec(&mut self, n: usize, s: f32) -> Vec<f32> {
            (0..n).map(|_| self.next() * s).collect()
        }
        fn bf(&mut self, n: usize, s: f32) -> Vec<u16> {
            (0..n)
                .map(|_| crate::tensor::dtype::f32_to_bf16(self.next() * s))
                .collect()
        }
    }

    /// An encoder of shape `cfg` with weights drawn from `Rng(seed)` in a fixed order (the order
    /// `tiny` has always used). `cfg.min_pixels`/`max_pixels` are left as given.
    pub fn encoder(cfg: QwenVisionConfig, seed: u32) -> QwenVisionEncoder {
        let (d, f, pd) = (cfg.hidden, cfg.ffn, 3 * cfg.patch * cfg.patch);
        let mut r = Rng(seed);
        let patch_w = r.vec(d * pd, 0.2);
        let patch_b = r.vec(d, 0.1);
        let pos_embd = r.vec(cfg.pos_grid * cfg.pos_grid * d, 0.5);
        let layers = (0..cfg.layers)
            .map(|_| QwenVisionLayer {
                ln1_w: r.vec(d, 1.0),
                ln1_b: r.vec(d, 0.1),
                qkv_w: r.bf(3 * d * d, 0.4),
                qkv_b: r.vec(3 * d, 0.1),
                o_w: r.bf(d * d, 0.3),
                o_b: r.vec(d, 0.1),
                ln2_w: r.vec(d, 1.0),
                ln2_b: r.vec(d, 0.1),
                up_w: r.bf(f * d, 0.3),
                up_b: r.vec(f, 0.1),
                down_w: r.bf(d * f, 0.3),
                down_b: r.vec(d, 0.1),
            })
            .collect();
        // The video taps are DERIVED, not drawn (the draw order is the golden's): w0 = 0.75 p,
        // w1 = p - w0. Both are exact (Sterbenz), so w0 + w1 == p bit for bit — the invariant
        // the real mmproj has by construction — and w0 != w1, so a swapped frame pair shows.
        let w0: Vec<f32> = patch_w.iter().map(|&p| p * 0.75).collect();
        let w1: Vec<f32> = patch_w.iter().zip(&w0).map(|(&p, &a)| p - a).collect();
        let patch_w_pair = interleave_taps(&w0, &w1, pd);
        let w = QwenVisionWeights {
            patch_w,
            patch_w_pair,
            patch_b,
            pos_embd,
            layers,
            post_ln_w: r.vec(d, 1.0),
            post_ln_b: r.vec(d, 0.1),
            mm0_w: r.bf(16 * d * d, 0.2),
            mm0_b: r.vec(4 * d, 0.1),
            mm2_w: r.bf(cfg.proj_dim * 4 * d, 0.2),
            mm2_b: r.vec(cfg.proj_dim, 0.1),
        };
        QwenVisionEncoder::from_weights(cfg, w).expect("synthetic weights")
    }

    fn cfg(
        patch: usize,
        hidden: usize,
        ffn: usize,
        layers: usize,
        heads: usize,
        proj_dim: usize,
        pos_grid: usize,
    ) -> QwenVisionConfig {
        QwenVisionConfig {
            patch,
            merge: 2,
            hidden,
            ffn,
            layers,
            heads,
            eps: 1e-6,
            proj_dim,
            pos_grid,
            mean: [0.5; 3],
            std: [0.5; 3],
            min_pixels: 0,
            max_pixels: 0,
        }
    }

    /// The tiny encoder of the numpy golden: patch 4, hidden 16, ffn 24, 2 layers, 2 heads
    /// (head_dim 8), projection 12, a 6x6 position table.
    pub fn tiny() -> QwenVisionEncoder {
        encoder(cfg(4, 16, 24, 2, 2, 12, 6), 0x1234_5678)
    }

    /// The real model's head geometry at a small width: patch 16, hidden 144 = 2 heads x 72,
    /// ffn 200 (not a multiple of 32 — every tiled kernel's K tail runs), 2 layers, projection
    /// 40, an 8x8 position table. For GPU parity gates (no golden of its own: the CPU is the
    /// reference).
    pub fn medium() -> QwenVisionEncoder {
        encoder(cfg(16, 144, 200, 2, 2, 40, 8), 0x0bad_5eed)
    }

    /// `w x h` RGB test pattern `(x*7 + y*3 + c*50) % 256`, preprocessed with NO pixel budget
    /// (the size is used as is, so its sides must be multiples of `2 * patch`).
    pub fn pattern_image(enc: &QwenVisionEncoder, w: usize, h: usize) -> QwenImage {
        let mut rgb = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    rgb[(y * w + x) * 3 + c] = ((x * 7 + y * 3 + c * 50) % 256) as u8;
                }
            }
        }
        let mut c = enc.cfg.clone();
        c.min_pixels = 0;
        c.max_pixels = 0;
        preprocess(&c, &rgb, w, h)
    }

    /// The tiny image: 24 wide x 16 tall (a 6x4 patch grid, 3x2 = 6 merged tokens — a grid
    /// that is NOT the 6x6 position table, so the bilinear resize is exercised).
    pub fn tiny_image(enc: &QwenVisionEncoder) -> QwenImage {
        pattern_image(enc, 24, 16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_real_budget() -> QwenVisionConfig {
        let mut c = QwenVisionConfig {
            patch: 16,
            merge: 2,
            hidden: 1152,
            ffn: 4304,
            layers: 27,
            heads: 16,
            eps: 1e-6,
            proj_dim: 5120,
            pos_grid: 48,
            mean: [0.5; 3],
            std: [0.5; 3],
            min_pixels: 0,
            max_pixels: 0,
        };
        c.budget(64, 1024);
        c
    }

    /// Shape math, worked by hand from llama.cpp's `calc_size_preserved_ratio`:
    /// - 512x384: already multiples of 32, 16x12 tokens = 192 in [64, 1024] -> unchanged.
    /// - 500x375: round(15.6)=16, round(11.7)=12 -> 512x384.
    /// - 64x64: 2x2 = 4 tokens < 64 -> beta = sqrt(65536/4096) = 4 -> 256x256 (8x8 = 64).
    /// - 4000x3000: 125x94 -> way over 1024 tokens; beta = sqrt(12e6/1048576) = 3.3829...,
    ///   floor(3000/3.3829/32)=27 -> 864, floor(4000/3.3829/32)=36 -> 1152: 36x27 = 972 tokens.
    /// - 1x1000 (a thin line): w rounds to max(32, 0) = 32; h 992 -> 1x31 = 31 < 64 tokens ->
    ///   beta = sqrt(65536/1000) = 8.095, ceil(8.095/32)*32 = 32, ceil(8095.4/32)*32 = 8096.
    #[test]
    fn smart_resize_shapes() {
        let c = cfg_real_budget();
        let r = |w, h| smart_resize(w, h, c.align(), c.min_pixels, c.max_pixels);
        assert_eq!(r(512, 384), (512, 384));
        assert_eq!(r(500, 375), (512, 384));
        assert_eq!(r(64, 64), (256, 256));
        assert_eq!(r(4000, 3000), (1152, 864));
        assert_eq!(r(1, 1000), (32, 8096));
        for (w, h) in [
            (512, 384),
            (500, 375),
            (64, 64),
            (4000, 3000),
            (777, 333),
            (33, 2000),
        ] {
            let (tw, th) = r(w, h);
            assert_eq!(tw % 32, 0);
            assert_eq!(th % 32, 0);
            let tok = (tw / 32) * (th / 32);
            assert!(
                (64..=1024).contains(&tok),
                "{w}x{h} -> {tw}x{th} = {tok} tokens"
            );
        }
    }

    /// Resizing to the same size is the identity; a constant image stays constant under
    /// bicubic (the weights are normalized); and a 2x downscale of a checkerboard averages.
    #[test]
    fn bicubic_resize_basics() {
        let (w, h) = (40usize, 30usize);
        let flat = vec![77u8; w * h * 3];
        let out = resize_bicubic_pillow(&flat, w, h, 64, 32);
        assert_eq!(out.len(), 64 * 32 * 3);
        assert!(out.iter().all(|&v| v == 77), "constant stays constant");
        let mut grad = vec![0u8; w * h * 3];
        for (i, p) in grad.chunks_mut(3).enumerate() {
            p.copy_from_slice(&[(i % w * 6) as u8, (i / w * 8) as u8, 9]);
        }
        let out = resize_bicubic_pillow(&grad, w, h, w * 2, h * 2);
        // monotone along x in row 0 (a ramp stays a ramp, allowing bicubic overshoot at ends)
        let r0: Vec<u8> = (4..w * 2 - 4).map(|x| out[x * 3]).collect();
        assert!(r0.windows(2).all(|p| p[1] >= p[0]), "{r0:?}");
    }

    /// Preprocess output dims + normalization of a known pixel.
    #[test]
    fn preprocess_dims_and_norm() {
        let c = cfg_real_budget();
        let (w, h) = (512usize, 384usize);
        let mut rgb = vec![0u8; w * h * 3];
        rgb[0] = 255; // top-left red = 255 -> (1 - .5)/.5 = 1
        let img = preprocess(&c, &rgb, w, h);
        assert_eq!(
            (img.width, img.height, img.grid_w, img.grid_h),
            (512, 384, 32, 24)
        );
        assert_eq!(img.token_grid(), (12, 16));
        assert_eq!(img.n_tokens(), 192);
        assert_eq!(img.pixels.len(), 3 * 512 * 384);
        assert_eq!(img.pixels[0], 1.0);
        assert_eq!(img.pixels[1], -1.0);
    }

    #[test]
    fn erf_and_gelu() {
        // reference values (Abramowitz & Stegun table / mpmath)
        for (x, e) in [
            (0.0, 0.0),
            (0.5, 0.520_499_877_813_046_5),
            (1.0, 0.842_700_792_949_714_9),
            (2.0, 0.995_322_265_018_952_7),
            (3.0, 0.999_977_909_503_001_4),
            (-1.5, -0.966_105_146_475_310_7),
        ] {
            assert!((erf(x) - e).abs() < 1e-13, "erf({x}) = {} vs {e}", erf(x));
        }
        assert!((gelu_erf(1.0) - 0.841_344_7).abs() < 1e-6);
        assert!((gelu_tanh(1.0) - 0.841_192).abs() < 1e-5);
    }

    // ---- synthetic-weights golden (independent numpy reference) --------------------------

    fn tiny() -> QwenVisionEncoder {
        synthetic::tiny()
    }

    fn tiny_image(enc: &QwenVisionEncoder) -> QwenImage {
        synthetic::tiny_image(enc)
    }

    /// GOLDEN: the tiny encoder's output must match `scripts/qwen_vision_ref.py --tiny`
    /// (numpy, float64, written from llama.cpp's `qwen3vl.cpp` graph, `ggml_rope_multi`
    /// VISION semantics and `ggml_interpolate` align-corners — not from this file). Values
    /// printed by that script on 2026-09-27; tolerance covers f32-vs-f64 accumulation.
    #[test]
    fn tiny_encoder_matches_numpy_reference() {
        let enc = tiny();
        let img = tiny_image(&enc);
        assert_eq!((img.grid_w, img.grid_h), (6, 4));
        let out = enc.encode(&img);
        assert_eq!(out.len(), 6 * 12);
        let sum: f64 = out.iter().map(|&v| v as f64).sum();
        let abs: f64 = out.iter().map(|&v| (v as f64).abs()).sum();
        eprintln!("tiny: sum={sum:.6} abs={abs:.6} first={:?}", &out[..4]);
        // Filled in from the numpy reference (see the script's --tiny output).
        let (want_sum, want_abs, want_first) = TINY_GOLDEN;
        assert!((sum - want_sum).abs() < 1e-3, "sum {sum} vs {want_sum}");
        assert!((abs - want_abs).abs() < 1e-3, "abs {abs} vs {want_abs}");
        for (a, b) in out.iter().zip(want_first.iter()) {
            assert!(
                (*a as f64 - b).abs() < 1e-4,
                "first {:?} vs {want_first:?}",
                &out[..4]
            );
        }
    }

    /// (sum, sum|x|, first four values) of the tiny forward, from the numpy reference.
    const TINY_GOLDEN: (f64, f64, [f64; 4]) = (
        1.790_162_395,
        22.488_194_138,
        [
            0.502_010_689,
            -0.229_807_648,
            -0.013_536_033,
            -0.774_382_088,
        ],
    );

    /// Real mmproj, tiny image: patch embedding through the merger on a 64x64 pixel input
    /// (4x4 patches -> 4 tokens, bypassing the min-pixel upscale), checked against the numpy
    /// reference run on the same GGUF. Ignored by default (reads 0.93 GB); run with
    /// `cargo test --release -p arf-core qwen_vision_real -- --ignored --nocapture`.
    #[test]
    #[ignore = "reads the 0.93 GB mmproj; run with --release -- --ignored"]
    fn qwen_vision_real_mmproj_matches_numpy_reference() {
        // ARF_QWEN_MMPROJ overrides the in-repo path (a fresh checkout has no models/ dir).
        let path = std::env::var("ARF_QWEN_MMPROJ")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                std::path::PathBuf::from(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../models/qwen3.8-27b-gsqrco/mmproj-Qwen3.8-27B-BF16.gguf"
                ))
            });
        let path = path.as_path();
        if !path.exists() {
            eprintln!(
                "SKIP: mmproj not found at {} -- this run verified NOTHING",
                path.display()
            );
            return;
        }
        let t0 = std::time::Instant::now();
        let enc = QwenVisionEncoder::load(path).expect("load");
        eprintln!("load: {:.2}s", t0.elapsed().as_secs_f64());
        assert_eq!(enc.cfg.hidden, 1152);
        assert_eq!(enc.cfg.pos_grid, 48);
        assert_eq!(enc.cfg.proj_dim, 5120);
        let (w, h) = (64usize, 64usize);
        let mut rgb = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    rgb[(y * w + x) * 3 + c] = ((x * 7 + y * 3 + c * 50) % 256) as u8;
                }
            }
        }
        // No resize: feed the 64x64 straight in (a 4x4 patch grid), like the reference.
        let mut c = enc.cfg.clone();
        c.min_pixels = 0;
        let img = preprocess(&c, &rgb, w, h);
        assert_eq!((img.grid_w, img.grid_h), (4, 4));
        let t1 = std::time::Instant::now();
        let out = enc.encode(&img);
        eprintln!("encode 16 patches: {:.2}s", t1.elapsed().as_secs_f64());
        assert_eq!(out.len(), 4 * 5120);
        let sum: f64 = out.iter().map(|&v| v as f64).sum();
        let abs: f64 = out.iter().map(|&v| (v as f64).abs()).sum();
        eprintln!("real: sum={sum:.6} abs={abs:.6} first={:?}", &out[..4]);
        let (want_sum, want_abs, want_first) = REAL_GOLDEN;
        let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1.0);
        assert!(rel(abs, want_abs) < 1e-4, "abs {abs} vs {want_abs}");
        assert!(
            (sum - want_sum).abs() < 1e-4 * want_abs,
            "sum {sum} vs {want_sum}"
        );
        for (a, b) in out.iter().zip(want_first.iter()) {
            assert!(
                (*a as f64 - b).abs() < 1e-3,
                "first {:?} vs {want_first:?}",
                &out[..4]
            );
        }
    }

    const REAL_GOLDEN: (f64, f64, [f64; 4]) = (
        -138.545_677_567,
        8_418.956_519_681,
        [
            0.220_128_320,
            -0.388_358_038,
            -0.464_051_424,
            -0.202_071_108,
        ],
    );

    // ---- video: the Conv3d's two temporal taps over a frame pair ---------------------------

    /// `(rel rms, max |d|)` of `a` against `b`.
    fn diff(a: &[f32], b: &[f32]) -> (f64, f64) {
        assert_eq!(a.len(), b.len());
        let (mut d2, mut r2, mut mx) = (0f64, 0f64, 0f64);
        for (&x, &y) in a.iter().zip(b) {
            let d = (x as f64 - y as f64).abs();
            d2 += d * d;
            r2 += (y as f64) * (y as f64);
            mx = mx.max(d);
        }
        ((d2 / r2).sqrt(), mx)
    }

    /// The taps the synthetic encoders derive sum to the image weight BIT FOR BIT — the property
    /// the real loader has by construction (`patch_w = w0 + w1`) — and differ from each other.
    #[test]
    fn synthetic_taps_sum_to_the_image_weight() {
        for enc in [synthetic::tiny(), synthetic::medium()] {
            let pd = 3 * enc.cfg.patch * enc.cfg.patch;
            let w = &enc.w;
            let mut differ = false;
            for (o, row) in w.patch_w_pair.chunks(2 * pd).enumerate() {
                let (w0, w1) = row.split_at(pd);
                for k in 0..pd {
                    assert_eq!(
                        (w0[k] + w1[k]).to_bits(),
                        w.patch_w[o * pd + k].to_bits(),
                        "row {o} col {k}"
                    );
                    differ |= w0[k] != w1[k];
                }
            }
            assert!(differ, "w0 == w1 would hide a swapped frame pair");
        }
    }

    /// The pair layout IS HF's Conv3d. `Qwen3VLVisionPatchEmbed` views each patch as
    /// `[channel, temporal, ky, kx]` (`patchify` permutes to `(.., channel, temporal, ky, kx)`) and
    /// convolves it with a `[out, channel, temporal, ky, kx]` kernel whose temporal slices 0 and 1
    /// llama.cpp stores as `v.patch_embd.weight` / `.weight.1`. Built here from those definitions,
    /// independently of `patches_pair_in_merge_order` / `interleave_taps`, on two DIFFERENT frames.
    #[test]
    fn frame_pair_embedding_is_the_hf_conv3d() {
        let enc = synthetic::medium();
        let c = &enc.cfg;
        let (ps, d, pd) = (c.patch, c.hidden, 3 * c.patch * c.patch);
        let a = synthetic::pattern_image(&enc, 96, 64);
        let mut b = a.clone();
        for (i, v) in b.pixels.iter_mut().enumerate() {
            *v = ((i * 37 % 101) as f32 - 50.0) / 60.0; // unrelated second frame
        }
        // Ours.
        let (pairs, rc) = patches_pair_in_merge_order(c, &a, &b);
        let ours = crate::tensor::matmul_nt(&pairs, rc.len(), 2 * pd, &enc.w.patch_w_pair, d);
        // The taps as the synthetic encoder derives them, [out][c][ky][kx] each.
        let w0: Vec<f32> = enc.w.patch_w.iter().map(|&p| p * 0.75).collect();
        let w1: Vec<f32> = enc
            .w
            .patch_w
            .iter()
            .zip(&w0)
            .map(|(&p, &q)| p - q)
            .collect();
        let (iw, ih) = (a.width, a.height);
        let mut worst = 0f64;
        for (i, &(py, px)) in rc.iter().enumerate() {
            for o in 0..d {
                let mut acc = 0f64;
                for ch in 0..3 {
                    for t in 0..2 {
                        let (frame, tap) = if t == 0 { (&a, &w0) } else { (&b, &w1) };
                        for ky in 0..ps {
                            for kx in 0..ps {
                                let x =
                                    frame.pixels[ch * iw * ih + (py * ps + ky) * iw + px * ps + kx];
                                let w = tap[o * pd + (ch * ps + ky) * ps + kx];
                                acc += x as f64 * w as f64;
                            }
                        }
                    }
                }
                worst = worst.max((ours[i * d + o] as f64 - acc).abs() / acc.abs().max(1.0));
            }
        }
        assert!(worst < 1e-5, "pair embedding vs HF Conv3d: rel {worst:.3e}");
        // And the order matters: (a, b) is not (b, a).
        let (swapped, _) = patches_pair_in_merge_order(c, &b, &a);
        let other = crate::tensor::matmul_nt(&swapped, rc.len(), 2 * pd, &enc.w.patch_w_pair, d);
        assert!(
            diff(&other, &ours).0 > 1e-2,
            "swapping the frames changed nothing"
        );
    }

    /// PROVEN, not assumed: a 2-frame video of one image encodes to the image's own embedding.
    /// `w0.x + w1.x = (w0 + w1).x` is exact in real arithmetic; in f32 the two sides sum in a
    /// different order, so this bounds the difference (measured on this data: see the eprintln)
    /// rather than claiming bit equality. Everything after the patch embedding is shared code.
    #[test]
    fn video_pair_of_one_image_equals_the_image() {
        for (enc, sizes) in [
            (synthetic::tiny(), vec![(24, 16), (40, 32)]),
            (synthetic::medium(), vec![(96, 64), (160, 128)]),
        ] {
            for (w, h) in sizes {
                let img = synthetic::pattern_image(&enc, w, h);
                let one = enc.encode(&img);
                let pair = enc.encode_frame_pair(&img, &img);
                let (rel, mx) = diff(&pair, &one);
                eprintln!(
                    "hidden {} {w}x{h}: pair(x, x) vs image(x) rel {rel:.3e} max|d| {mx:.3e}",
                    enc.cfg.hidden
                );
                assert!(rel < 1e-5, "{w}x{h}: rel {rel:.3e}");
            }
        }
    }

    /// The same identity on the REAL mmproj (the 64x64 input of the golden above): a frame pair of
    /// one image against the image encoding, and that against the numpy golden. Ignored by
    /// default (reads 0.93 GB); `cargo test --release -p arf-core qwen_vision_real -- --ignored`.
    #[test]
    #[ignore = "reads the 0.93 GB mmproj; run with --release -- --ignored"]
    fn qwen_vision_real_video_pair_equals_the_image() {
        let path = std::env::var("ARF_QWEN_MMPROJ")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                std::path::PathBuf::from(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../models/qwen3.8-27b-gsqrco/mmproj-Qwen3.8-27B-BF16.gguf"
                ))
            });
        if !path.exists() {
            eprintln!(
                "SKIP: mmproj not found at {} -- this run verified NOTHING",
                path.display()
            );
            return;
        }
        let enc = QwenVisionEncoder::load(&path).expect("load");
        let (w, h) = (64usize, 64usize);
        let mut rgb = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    rgb[(y * w + x) * 3 + c] = ((x * 7 + y * 3 + c * 50) % 256) as u8;
                }
            }
        }
        let mut c = enc.cfg.clone();
        c.min_pixels = 0;
        let img = preprocess(&c, &rgb, w, h);
        let one = enc.encode(&img);
        let pair = enc.encode_frame_pair(&img, &img);
        let (rel, mx) = diff(&pair, &one);
        let abs: f64 = pair.iter().map(|&v| (v as f64).abs()).sum();
        eprintln!("real: pair(x, x) vs image(x) rel {rel:.3e} max|d| {mx:.3e}; pair abs {abs:.6}");
        // Measured 2026-09-27: rel 5.9e-6, max|d| 3.8e-4 (27 layers amplify the patch-embed
        // rounding more than the 2-layer synthetic encoders' ~1e-7). A wrong or missing tap is
        // O(1e-1); 1e-4 is a bug detector with margin, not a precision claim.
        assert!(rel < 1e-4, "rel {rel:.3e}");
        let (_, want_abs, _) = REAL_GOLDEN;
        assert!(
            (abs - want_abs).abs() / want_abs < 1e-4,
            "abs {abs} vs {want_abs}"
        );
        // A pair of two different frames is NOT the image (the taps are really applied apart).
        let mut rgb2 = rgb.clone();
        rgb2.reverse();
        let img2 = preprocess(&c, &rgb2, w, h);
        let mixed = enc.encode_frame_pair(&img, &img2);
        assert!(diff(&mixed, &one).0 > 1e-2);
    }
}
