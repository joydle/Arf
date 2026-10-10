//! Qwen3-Omni AUDIO ENCODER ("AuT", M2 of the port, 2026-09-27): the `[128, T]`
//! log-mel from [`super::qwen_audio`] -> `[N, 2048]` rows in the Thinker's embedding space, 13 per
//! second of audio. CPU, f32 activations over bf16-resident weights (a bf16 weight widens to f32
//! exactly, so this is the reference's f32 computation, not an approximation of it).
//!
//! The reference is transformers 5.17.0 `models/qwen3_omni_moe/modeling_qwen3_omni_moe.py` ("M:")
//! with the real `thinker.audio_tower.*` weights of Qwen/Qwen3-Omni-30B-A3B-Instruct @ 26291f79
//! (shard 1 of 15). Every step below was read there, not assumed:
//!
//! | step | what | source |
//! |---|---|---|
//! | call | the model concatenates every clip's VALID frames into one `[128, sum]` and passes `feature_lens` | `get_audio_features`, M:1936-1943 |
//! | chunk | each clip is cut into 100-frame chunks (`n_window` 50 x 2), the tail keeps `len % 100`; **every chunk in the call is zero-padded to the longest chunk in the call** | `chunk_and_pad_features`, M:650-682 |
//! | stem | 3 x Conv2d(3x3, stride 2, pad 1), 1->480->480->480, each followed by erf GELU; `[128, 100]` -> `[480, 16, 13]` | M:776-778, 824-828 |
//! | flatten | `permute(0,3,1,2)` so feature index = `c * 16 + f`, then `conv_out` 7680->1280, no bias | M:779-783, 830-832 |
//! | position | sinusoid (`sin` half then `cos` half, timescale 1e4, float32) added per chunk, **restarting at 0 in every chunk** | M:94-112, 834-839 |
//! | select | keep the first `out_len(chunk_len)` positions of each chunk | `get_valid_indices` M:685-701, M:840 |
//! | windows | non-causal attention, block-diagonal over windows of `max_len_after_cnn * (800 / 100)` tokens (104 = 8 s whenever any chunk is full), per clip, the last window of a clip taking the remainder | `get_audio_cu_seqlens` M:704-743; eager/sdpa run each window separately, M:574-593 |
//! | layers | 32 x pre-LN (eps 1e-5): LN -> q,k,v (biased) -> 20 heads x 64, scale 1/8 -> out_proj -> residual; LN -> fc1 5120 -> erf GELU -> fc2 -> residual | M:506-647 |
//! | head | `ln_post` -> `proj1` 1280x1280 -> erf GELU -> `proj2` 1280->2048 | M:850-853 |
//!
//! Traps, each checked against the reference dumps (`scripts/ref/qwen3_omni/audio_encoder_ref.py`):
//! - **Padding changes the tail.** A zero-padded frame is zero only at the stem's INPUT. After
//!   conv1 + GELU a padded column is `gelu(bias)`, not zero, and conv2's last valid output reads
//!   it. So a tail chunk encoded alone (padded to its own length) and the same chunk inside a clip
//!   with full chunks (padded to 100) give different tail tokens. [`QwenAudioEncoder::encode_batch`]
//!   pads exactly as the reference does, which also makes a batch differ from separate encodes
//!   (the M1 "batches change features" trap, one stage later).
//! - **The window is set by the LONGEST chunk in the call**, not by 8 s of the clip: a call whose
//!   chunks are all shorter than 100 frames gets a window of `8 * out_len(longest)`. A clip under
//!   one chunk is always a single window, so this only matters for batches of short clips.
//! - The positional table has 1500 rows (`max_source_positions`) but only rows 0..13 are ever
//!   read, because positions restart per chunk.

use std::path::Path;

use crate::model::qwen_audio::LogMel;
use crate::model::qwen_vision::{gelu_erf, layer_norm};
use crate::model::weights::{read_weights_lazy, Weights};
use crate::tensor::dtype::f32_to_bf16;
use crate::tensor::matmul::matmul_nt_bf16;
use crate::{ArfError, Result};

/// The tensor-name prefix inside the Qwen3-Omni checkpoint.
pub const PREFIX: &str = "thinker.audio_tower.";

/// `thinker_config.audio_config` of Qwen3-Omni-30B-A3B-Instruct (`config.json` @ 26291f79).
#[derive(Debug, Clone, PartialEq)]
pub struct AudioEncoderConfig {
    pub n_mels: usize,
    pub d_model: usize,
    pub layers: usize,
    pub heads: usize,
    pub ffn: usize,
    /// Conv stem channels (`downsample_hidden_size`).
    pub conv_ch: usize,
    pub output_dim: usize,
    /// Chunks are `2 * n_window` mel frames.
    pub n_window: usize,
    /// Attention windows span `n_window_infer` mel frames (800 = 8 s).
    pub n_window_infer: usize,
    pub max_source_positions: usize,
    /// `nn.LayerNorm` default.
    pub eps: f32,
}

impl AudioEncoderConfig {
    pub fn qwen3_omni() -> AudioEncoderConfig {
        AudioEncoderConfig {
            n_mels: 128,
            d_model: 1280,
            layers: 32,
            heads: 20,
            ffn: 5120,
            conv_ch: 480,
            output_dim: 2048,
            n_window: 50,
            n_window_infer: 800,
            max_source_positions: 1500,
            eps: 1e-5,
        }
    }

    /// Mel frames per chunk (100).
    pub fn chunk(&self) -> usize {
        2 * self.n_window
    }

    /// Frequency rows after the three stride-2 convs (128 -> 64 -> 32 -> 16).
    pub fn freq_out(&self) -> usize {
        conv_out_len(conv_out_len(conv_out_len(self.n_mels)))
    }
}

/// Output length of one 3-tap, stride-2, pad-1 conv.
pub fn conv_out_len(n: usize) -> usize {
    (n - 1) / 2 + 1
}

/// Tokens the encoder emits for `frames` valid mel frames (M:152-159). The same as
/// [`super::qwen_audio::audio_token_count`], restated here for chunk lengths.
pub fn tokens_for_frames(frames: usize) -> usize {
    super::qwen_audio::audio_token_count(frames)
}

/// One encoder layer's weights. Public (read-only by convention) so the Metal port
/// (`arf_gpu::gpu::metal::qwen_audio`) uploads the SAME bits this encoder computes with.
pub struct Layer {
    pub ln1_w: Vec<f32>,
    pub ln1_b: Vec<f32>,
    /// q, k, v stacked: `[3 * d, d]` bf16, bias `[3 * d]`.
    pub qkv_w: Vec<u16>,
    pub qkv_b: Vec<f32>,
    pub o_w: Vec<u16>,
    pub o_b: Vec<f32>,
    pub ln2_w: Vec<f32>,
    pub ln2_b: Vec<f32>,
    pub fc1_w: Vec<u16>,
    pub fc1_b: Vec<f32>,
    pub fc2_w: Vec<u16>,
    pub fc2_b: Vec<f32>,
}

/// The loaded encoder. The weight fields are public for the Metal port, which uploads them as
/// they are; nothing outside this file mutates them.
pub struct QwenAudioEncoder {
    pub cfg: AudioEncoderConfig,
    /// conv1 `[480, 9]` (in-channels 1), f32.
    pub conv1_w: Vec<f32>,
    pub conv1_b: Vec<f32>,
    /// conv2/conv3 `[480, 480 * 9]` bf16 (im2col order `c_in, kh, kw`, PyTorch's).
    pub conv2_w: Vec<u16>,
    pub conv2_b: Vec<f32>,
    pub conv3_w: Vec<u16>,
    pub conv3_b: Vec<f32>,
    /// `[1280, 7680]`.
    pub conv_out_w: Vec<u16>,
    pub layers: Vec<Layer>,
    pub ln_post_w: Vec<f32>,
    pub ln_post_b: Vec<f32>,
    pub proj1_w: Vec<u16>,
    pub proj1_b: Vec<f32>,
    pub proj2_w: Vec<u16>,
    pub proj2_b: Vec<f32>,
    /// Sinusoid rows `[max_source_positions, d]`, float32 as the reference builds them.
    pub pos: Vec<f32>,
}

/// Round an f32 vector to bf16 bits. Lossless for a bf16 checkpoint (the only kind shipped);
/// an f32 checkpoint would be rounded, which the parity gate would then have to absorb.
fn to_bf16(v: Vec<f32>) -> Vec<u16> {
    v.into_iter().map(f32_to_bf16).collect()
}

/// The sinusoid table exactly as M:104-108 builds it in float32: `inc = ln(1e4) / (d/2 - 1)`
/// (a Python float, used as a float32 scalar), `inv[i] = exp(-inc * i)`, `t * inv[i]`, then
/// `[sin | cos]`.
fn sinusoids(length: usize, channels: usize) -> Vec<f32> {
    let half = channels / 2;
    let inc = ((10000.0f64).ln() / (half - 1) as f64) as f32;
    let inv: Vec<f32> = (0..half).map(|i| (-inc * i as f32).exp()).collect();
    let mut out = vec![0.0f32; length * channels];
    for t in 0..length {
        let row = &mut out[t * channels..(t + 1) * channels];
        for i in 0..half {
            let a = t as f32 * inv[i];
            row[i] = a.sin();
            row[half + i] = a.cos();
        }
    }
    out
}

/// How one call of the encoder is cut up: the chunks, the stem's padded width, the attention
/// windows. Computed here ONCE and used by both the CPU encoder and the Metal port
/// (`arf_gpu::gpu::metal::qwen_audio`), so the two cannot disagree on the batch-padding and
/// windowing semantics the module doc lists as traps.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioPlan {
    /// Every clip's chunks, in order: (clip, first frame, valid length).
    pub chunks: Vec<(usize, usize, usize)>,
    /// Frames every chunk is zero-padded to before the stem: the longest chunk IN THE CALL.
    pub width: usize,
    /// Attention windows over the concatenated token rows: (first row, rows).
    pub windows: Vec<(usize, usize)>,
    /// Token rows of the whole call (`sum tokens_for_frames(valid)`).
    pub n_tokens: usize,
}

impl AudioPlan {
    /// The plan for `clips` (`(mel, valid frames)`), with the attention window forced to
    /// `window` tokens when given (a test control; `None` is the reference's rule).
    pub fn new(
        c: &AudioEncoderConfig,
        clips: &[(&LogMel, usize)],
        window: Option<usize>,
    ) -> Result<AudioPlan> {
        if clips.is_empty() {
            return Err(ArfError::other("audio encoder: no clips"));
        }
        // Chunks of every clip, in order: (clip, first frame, length).
        let mut chunks = Vec::new();
        for (k, &(mel, valid)) in clips.iter().enumerate() {
            if mel.n_mels != c.n_mels || valid == 0 || valid > mel.n_frames {
                return Err(ArfError::other(format!(
                    "audio encoder: clip {k} is [{}, {}] with {valid} valid frames",
                    mel.n_mels, mel.n_frames
                )));
            }
            let mut s = 0;
            while s < valid {
                let len = (valid - s).min(c.chunk());
                chunks.push((k, s, len));
                s += len;
            }
        }
        let width = chunks.iter().map(|x| x.2).max().unwrap();
        // Attention windows (M:704-743), in token rows.
        let max_after = tokens_for_frames(width);
        let win = window.unwrap_or(max_after * (c.n_window_infer / c.chunk()));
        let mut windows = Vec::new();
        let mut start = 0;
        for &(_, valid) in clips {
            let n = tokens_for_frames(valid);
            let mut off = 0;
            while off < n {
                let len = (n - off).min(win);
                windows.push((start + off, len));
                off += len;
            }
            start += n;
        }
        Ok(AudioPlan {
            chunks,
            width,
            windows,
            n_tokens: start,
        })
    }

    /// Chunk `i` as the stem's input image `[n_mels, width]`: its frames, zero past its length
    /// (M:650-682).
    pub fn padded_chunk(
        &self,
        c: &AudioEncoderConfig,
        clips: &[(&LogMel, usize)],
        i: usize,
    ) -> Vec<f32> {
        let (k, s, len) = self.chunks[i];
        let mel = clips[k].0;
        let width = self.width;
        let mut img = vec![0.0f32; c.n_mels * width];
        for m in 0..c.n_mels {
            img[m * width..m * width + len]
                .copy_from_slice(&mel.data[m * mel.n_frames + s..m * mel.n_frames + s + len]);
        }
        img
    }
}

impl QwenAudioEncoder {
    /// Load from a safetensors file holding `thinker.audio_tower.*`: the checkpoint's shard 1, or
    /// the audio-only extract `scripts/ref/qwen3_omni/fetch_audio_tower.py` writes.
    pub fn load(path: &Path) -> Result<QwenAudioEncoder> {
        let w = read_weights_lazy(&[path])?;
        QwenAudioEncoder::from_weights(&w, PREFIX, AudioEncoderConfig::qwen3_omni())
    }

    pub fn from_weights(w: &Weights, prefix: &str, cfg: AudioEncoderConfig) -> Result<Self> {
        let d = cfg.d_model;
        let ch = cfg.conv_ch;
        let t = |name: &str, shape: &[usize]| w.get_f32_shaped(&format!("{prefix}{name}"), shape);
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = format!("layers.{i}.");
            let lt = |name: &str, shape: &[usize]| t(&format!("{p}{name}"), shape);
            let mut qkv_w = lt("self_attn.q_proj.weight", &[d, d])?;
            qkv_w.extend(lt("self_attn.k_proj.weight", &[d, d])?);
            qkv_w.extend(lt("self_attn.v_proj.weight", &[d, d])?);
            let mut qkv_b = lt("self_attn.q_proj.bias", &[d])?;
            qkv_b.extend(lt("self_attn.k_proj.bias", &[d])?);
            qkv_b.extend(lt("self_attn.v_proj.bias", &[d])?);
            layers.push(Layer {
                ln1_w: lt("self_attn_layer_norm.weight", &[d])?,
                ln1_b: lt("self_attn_layer_norm.bias", &[d])?,
                qkv_w: to_bf16(qkv_w),
                qkv_b,
                o_w: to_bf16(lt("self_attn.out_proj.weight", &[d, d])?),
                o_b: lt("self_attn.out_proj.bias", &[d])?,
                ln2_w: lt("final_layer_norm.weight", &[d])?,
                ln2_b: lt("final_layer_norm.bias", &[d])?,
                fc1_w: to_bf16(lt("fc1.weight", &[cfg.ffn, d])?),
                fc1_b: lt("fc1.bias", &[cfg.ffn])?,
                fc2_w: to_bf16(lt("fc2.weight", &[d, cfg.ffn])?),
                fc2_b: lt("fc2.bias", &[d])?,
            });
        }
        let flat = ch * cfg.freq_out();
        Ok(QwenAudioEncoder {
            conv1_w: t("conv2d1.weight", &[ch, 1, 3, 3])?,
            conv1_b: t("conv2d1.bias", &[ch])?,
            conv2_w: to_bf16(t("conv2d2.weight", &[ch, ch, 3, 3])?),
            conv2_b: t("conv2d2.bias", &[ch])?,
            conv3_w: to_bf16(t("conv2d3.weight", &[ch, ch, 3, 3])?),
            conv3_b: t("conv2d3.bias", &[ch])?,
            conv_out_w: to_bf16(t("conv_out.weight", &[d, flat])?),
            layers,
            ln_post_w: t("ln_post.weight", &[d])?,
            ln_post_b: t("ln_post.bias", &[d])?,
            proj1_w: to_bf16(t("proj1.weight", &[d, d])?),
            proj1_b: t("proj1.bias", &[d])?,
            proj2_w: to_bf16(t("proj2.weight", &[cfg.output_dim, d])?),
            proj2_b: t("proj2.bias", &[cfg.output_dim])?,
            pos: sinusoids(cfg.max_source_positions, d),
            cfg,
        })
    }

    /// One clip: `mel` is `[128, T]`, of which the first `valid_frames` are real (the processor's
    /// `feature_attention_mask.sum()`). Returns `[tokens_for_frames(valid_frames), 2048]`.
    pub fn encode(&self, mel: &LogMel, valid_frames: usize) -> Result<Vec<f32>> {
        self.encode_batch(&[(mel, valid_frames)], None)
    }

    /// A processor batch, as ONE call of the reference encoder: the rows of every clip, in order,
    /// concatenated (`[sum tokens, 2048]`). This is what the model scatters over the
    /// `<|audio_pad|>` rows. See the module doc for why a batch is not a list of separate encodes.
    ///
    /// `taps`, when given, receives `2 + layers` tensors of `[N, 1280]`: the layer-0 input (stem +
    /// position), each layer's output, and `ln_post`'s output (the reference dump's `hidden.f32`).
    pub fn encode_batch(
        &self,
        clips: &[(&LogMel, usize)],
        taps: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<Vec<f32>> {
        self.encode_batch_windowed(clips, taps, None)
    }

    /// [`Self::encode_batch`] with the attention window forced to `window` tokens (the parity
    /// test's negative control: one window over a > 8 s clip must FAIL the gate).
    fn encode_batch_windowed(
        &self,
        clips: &[(&LogMel, usize)],
        mut taps: Option<&mut Vec<Vec<f32>>>,
        window: Option<usize>,
    ) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let d = c.d_model;
        let plan = AudioPlan::new(c, clips, window)?;
        let width = plan.width;
        // Stem over every chunk, padded to `width` frames (M:650-682), then the token rows.
        let mut x = Vec::with_capacity(plan.n_tokens * d);
        for (i, &(_, _, len)) in plan.chunks.iter().enumerate() {
            let img = plan.padded_chunk(c, clips, i);
            let rows = self.stem(&img, width);
            let keep = tokens_for_frames(len);
            for (t, row) in rows.chunks(d).take(keep).enumerate() {
                x.extend(
                    row.iter()
                        .zip(&self.pos[t * d..(t + 1) * d])
                        .map(|(a, p)| a + p),
                );
            }
        }
        let windows = &plan.windows;
        let n = x.len() / d;
        debug_assert_eq!(plan.n_tokens, n);
        if let Some(t) = taps.as_deref_mut() {
            t.push(x.clone());
        }
        for l in &self.layers {
            self.layer(l, &mut x, n, windows);
            if let Some(t) = taps.as_deref_mut() {
                t.push(x.clone());
            }
        }
        layer_norm(&mut x, &self.ln_post_w, &self.ln_post_b, d, c.eps);
        if let Some(t) = taps {
            t.push(x.clone());
        }
        let mut h = matmul_nt_bf16(&x, n, d, &self.proj1_w, d);
        add_bias(&mut h, &self.proj1_b);
        h.iter_mut().for_each(|v| *v = gelu_erf(*v));
        let mut out = matmul_nt_bf16(&h, n, d, &self.proj2_w, c.output_dim);
        add_bias(&mut out, &self.proj2_b);
        Ok(out)
    }

    /// The conv stem on one padded chunk `[128, w]` -> token rows `[w3, d]` (all `w3` positions,
    /// valid or not; the caller keeps the valid prefix).
    fn stem(&self, img: &[f32], w: usize) -> Vec<f32> {
        let c = &self.cfg;
        let ch = c.conv_ch;
        let (h0, w0) = (c.n_mels, w);
        // conv1: 1 -> 480, direct (9 taps).
        let (h1, w1) = (conv_out_len(h0), conv_out_len(w0));
        let mut a1 = vec![0.0f32; ch * h1 * w1];
        for o in 0..ch {
            let k = &self.conv1_w[o * 9..o * 9 + 9];
            for y in 0..h1 {
                for xo in 0..w1 {
                    let mut acc = 0.0f32;
                    for ky in 0..3 {
                        let iy = (2 * y + ky) as isize - 1;
                        if iy < 0 || iy >= h0 as isize {
                            continue;
                        }
                        for kx in 0..3 {
                            let ix = (2 * xo + kx) as isize - 1;
                            if ix < 0 || ix >= w0 as isize {
                                continue;
                            }
                            acc += k[ky * 3 + kx] * img[iy as usize * w0 + ix as usize];
                        }
                    }
                    a1[(o * h1 + y) * w1 + xo] = gelu_erf(acc + self.conv1_b[o]);
                }
            }
        }
        let (a2, h2, w2) = conv3x3_s2(&a1, ch, h1, w1, &self.conv2_w, &self.conv2_b, ch);
        let (a3, h3, w3) = conv3x3_s2(&a2, ch, h2, w2, &self.conv3_w, &self.conv3_b, ch);
        debug_assert_eq!(h3, c.freq_out());
        // [c, f, t] -> rows t of (c * 16 + f), then conv_out.
        let flat = ch * h3;
        let mut rows = vec![0.0f32; w3 * flat];
        for cc in 0..ch {
            for f in 0..h3 {
                for t in 0..w3 {
                    rows[t * flat + cc * h3 + f] = a3[(cc * h3 + f) * w3 + t];
                }
            }
        }
        matmul_nt_bf16(&rows, w3, flat, &self.conv_out_w, c.d_model)
    }

    fn layer(&self, l: &Layer, x: &mut [f32], n: usize, windows: &[(usize, usize)]) {
        let c = &self.cfg;
        let d = c.d_model;
        let mut h = x.to_vec();
        layer_norm(&mut h, &l.ln1_w, &l.ln1_b, d, c.eps);
        let mut qkv = matmul_nt_bf16(&h, n, d, &l.qkv_w, 3 * d);
        add_bias(&mut qkv, &l.qkv_b);
        let attn = windowed_attention(&qkv, n, d, c.heads, windows);
        let mut o = matmul_nt_bf16(&attn, n, d, &l.o_w, d);
        add_bias(&mut o, &l.o_b);
        for (a, b) in x.iter_mut().zip(&o) {
            *a += b;
        }
        let mut h = x.to_vec();
        layer_norm(&mut h, &l.ln2_w, &l.ln2_b, d, c.eps);
        let mut up = matmul_nt_bf16(&h, n, d, &l.fc1_w, c.ffn);
        add_bias(&mut up, &l.fc1_b);
        up.iter_mut().for_each(|v| *v = gelu_erf(*v));
        let mut down = matmul_nt_bf16(&up, n, c.ffn, &l.fc2_w, d);
        add_bias(&mut down, &l.fc2_b);
        for (a, b) in x.iter_mut().zip(&down) {
            *a += b;
        }
    }
}

fn add_bias(x: &mut [f32], b: &[f32]) {
    for row in x.chunks_mut(b.len()) {
        for (a, bb) in row.iter_mut().zip(b) {
            *a += bb;
        }
    }
}

/// Conv2d 3x3, stride 2, padding 1, `cin -> cout`, + bias, + erf GELU, by im2col and a bf16
/// matmul. `x` is `[cin, h, w]`; returns (`[cout, h', w']`, h', w').
fn conv3x3_s2(
    x: &[f32],
    cin: usize,
    h: usize,
    w: usize,
    wt: &[u16],
    b: &[f32],
    cout: usize,
) -> (Vec<f32>, usize, usize) {
    let (ho, wo) = (conv_out_len(h), conv_out_len(w));
    let k = cin * 9;
    let p = ho * wo;
    // col[pos][ci * 9 + ky * 3 + kx], PyTorch's weight order.
    let mut col = vec![0.0f32; p * k];
    for y in 0..ho {
        for xo in 0..wo {
            let row = &mut col[(y * wo + xo) * k..(y * wo + xo + 1) * k];
            for ky in 0..3 {
                let iy = (2 * y + ky) as isize - 1;
                if iy < 0 || iy >= h as isize {
                    continue;
                }
                for kx in 0..3 {
                    let ix = (2 * xo + kx) as isize - 1;
                    if ix < 0 || ix >= w as isize {
                        continue;
                    }
                    let src = iy as usize * w + ix as usize;
                    for ci in 0..cin {
                        row[ci * 9 + ky * 3 + kx] = x[ci * h * w + src];
                    }
                }
            }
        }
    }
    let y = matmul_nt_bf16(&col, p, k, wt, cout); // [p, cout]
    let mut out = vec![0.0f32; cout * p];
    for (pos, r) in y.chunks(cout).enumerate() {
        for (o, &v) in r.iter().enumerate() {
            out[o * p + pos] = gelu_erf(v + b[o]);
        }
    }
    (out, ho, wo)
}

/// Non-causal multi-head attention within each window of `qkv` rows (`[n, 3d]`, `[q | k | v]`),
/// scale `1/sqrt(hd)`. Every (window, head) pair is one task, spread over threads; each task
/// returns its `[len, hd]` block and the blocks are scattered into `[n, d]`.
fn windowed_attention(
    qkv: &[f32],
    n: usize,
    d: usize,
    heads: usize,
    windows: &[(usize, usize)],
) -> Vec<f32> {
    let hd = d / heads;
    let scale = 1.0 / (hd as f32).sqrt();
    let tasks: Vec<(usize, usize, usize)> = windows
        .iter()
        .flat_map(|&(s, len)| (0..heads).map(move |hh| (s, len, hh)))
        .collect();
    let threads = std::thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(1)
        .min(tasks.len());
    let per = tasks.len().div_ceil(threads);
    let results: Vec<Vec<f32>> = std::thread::scope(|sc| {
        let handles: Vec<_> = tasks
            .chunks(per)
            .map(|group| {
                sc.spawn(move || {
                    group
                        .iter()
                        .map(|&(s, len, hh)| {
                            let q = |i: usize| &qkv[i * 3 * d + hh * hd..i * 3 * d + (hh + 1) * hd];
                            let k = |j: usize| {
                                &qkv[j * 3 * d + d + hh * hd..j * 3 * d + d + (hh + 1) * hd]
                            };
                            let v = |j: usize| {
                                &qkv[j * 3 * d + 2 * d + hh * hd..j * 3 * d + 2 * d + (hh + 1) * hd]
                            };
                            let mut o = vec![0.0f32; len * hd];
                            let mut sc = vec![0.0f32; len];
                            for i in 0..len {
                                let qi = q(s + i);
                                let mut mx = f32::NEG_INFINITY;
                                for (j, v) in sc.iter_mut().enumerate() {
                                    let kj = k(s + j);
                                    let mut acc = 0.0f32;
                                    for e in 0..hd {
                                        acc += qi[e] * kj[e];
                                    }
                                    *v = acc * scale;
                                    mx = mx.max(*v);
                                }
                                let mut sum = 0.0f32;
                                for v in sc.iter_mut() {
                                    *v = (*v - mx).exp();
                                    sum += *v;
                                }
                                let inv = 1.0 / sum;
                                let oi = &mut o[i * hd..(i + 1) * hd];
                                for (j, &p) in sc.iter().enumerate() {
                                    let vj = v(s + j);
                                    let p = p * inv;
                                    for e in 0..hd {
                                        oi[e] += p * vj[e];
                                    }
                                }
                            }
                            o
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("attention thread"))
            .collect()
    });
    let mut out = vec![0.0f32; n * d];
    for (&(s, len, hh), o) in tasks.iter().zip(&results) {
        for i in 0..len {
            out[(s + i) * d + hh * hd..(s + i) * d + (hh + 1) * hd]
                .copy_from_slice(&o[i * hd..(i + 1) * hd]);
        }
    }
    out
}

/// Synthetic encoders for GPU parity gates (`arf-gpu/tests/qwen_audio_gpu.rs`): weights drawn
/// from a xorshift32 stream, matrices rounded to bf16, the real chunk/window geometry (100-frame
/// chunks, 13 tokens each, 104-token windows) at a small width. No reference implementation
/// checks these; they exist so the Metal port can be compared with THIS encoder, which the
/// real-weights gate above checks against transformers.
pub mod synthetic {
    use super::*;

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
        fn around(&mut self, n: usize, c: f32, s: f32) -> Vec<f32> {
            (0..n).map(|_| c + self.next() * s).collect()
        }
        fn bf(&mut self, n: usize, s: f32) -> Vec<u16> {
            (0..n).map(|_| f32_to_bf16(self.next() * s)).collect()
        }
    }

    /// An encoder of shape `cfg`, weights from `Rng(seed)`. Scales keep every activation O(1):
    /// matrices ~ U(-1,1) / sqrt(fan_in).
    pub fn encoder(cfg: AudioEncoderConfig, seed: u32) -> QwenAudioEncoder {
        let (d, f, ch) = (cfg.d_model, cfg.ffn, cfg.conv_ch);
        let flat = ch * cfg.freq_out();
        let inv = |k: usize| 1.5 / (k as f32).sqrt();
        let mut r = Rng(seed);
        let layers = (0..cfg.layers)
            .map(|_| Layer {
                ln1_w: r.around(d, 1.0, 0.2),
                ln1_b: r.vec(d, 0.1),
                qkv_w: r.bf(3 * d * d, inv(d)),
                qkv_b: r.vec(3 * d, 0.1),
                o_w: r.bf(d * d, inv(d)),
                o_b: r.vec(d, 0.1),
                ln2_w: r.around(d, 1.0, 0.2),
                ln2_b: r.vec(d, 0.1),
                fc1_w: r.bf(f * d, inv(d)),
                fc1_b: r.vec(f, 0.1),
                fc2_w: r.bf(d * f, inv(f)),
                fc2_b: r.vec(d, 0.1),
            })
            .collect();
        QwenAudioEncoder {
            conv1_w: r.vec(ch * 9, inv(9)),
            conv1_b: r.vec(ch, 0.1),
            conv2_w: r.bf(ch * ch * 9, inv(ch * 9)),
            conv2_b: r.vec(ch, 0.1),
            conv3_w: r.bf(ch * ch * 9, inv(ch * 9)),
            conv3_b: r.vec(ch, 0.1),
            conv_out_w: r.bf(d * flat, inv(flat)),
            layers,
            ln_post_w: r.around(d, 1.0, 0.2),
            ln_post_b: r.vec(d, 0.1),
            proj1_w: r.bf(d * d, inv(d)),
            proj1_b: r.vec(d, 0.1),
            proj2_w: r.bf(cfg.output_dim * d, inv(d)),
            proj2_b: r.vec(cfg.output_dim, 0.1),
            pos: sinusoids(cfg.max_source_positions, d),
            cfg,
        }
    }

    /// 32 mel bins (stem 32 -> 16 -> 8 -> 4), 16 conv channels, d 128 = 2 heads x 64 (the real
    /// head_dim), ffn 200 (not a multiple of any tile), 2 layers, output 96. Chunking and
    /// windowing are the real ones.
    pub fn small() -> QwenAudioEncoder {
        encoder(
            AudioEncoderConfig {
                n_mels: 32,
                d_model: 128,
                layers: 2,
                heads: 2,
                ffn: 200,
                conv_ch: 16,
                output_dim: 96,
                n_window: 50,
                n_window_infer: 800,
                max_source_positions: 1500,
                eps: 1e-5,
            },
            0x0a0d_10e5,
        )
    }

    /// A log-mel-like input `[n_mels, frames]` in about [-1, 1] (smooth in time and frequency,
    /// plus noise), different for every `seed`.
    pub fn mel(n_mels: usize, frames: usize, seed: u32) -> LogMel {
        let mut r = Rng(seed | 1);
        let ph = r.next() * 3.0;
        let mut data = Vec::with_capacity(n_mels * frames);
        for m in 0..n_mels {
            for t in 0..frames {
                let a = (t as f32 * 0.07 + m as f32 * 0.3 + ph).sin() * 0.6;
                data.push(a + r.next() * 0.3);
            }
        }
        LogMel {
            n_mels,
            n_frames: frames,
            data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_and_window_arithmetic_matches_the_reference_formulas() {
        let c = AudioEncoderConfig::qwen3_omni();
        assert_eq!(c.freq_out(), 16);
        assert_eq!(c.conv_ch * c.freq_out(), 7680);
        // 100 frames -> 50 -> 25 -> 13 positions: 13 tokens per full second.
        assert_eq!(conv_out_len(conv_out_len(conv_out_len(100))), 13);
        assert_eq!(tokens_for_frames(100), 13);
        assert_eq!(tokens_for_frames(370), 3 * 13 + 9);
        assert_eq!(tokens_for_frames(3100), 403);
        // 8 s windows: 13 * 800 / 100.
        assert_eq!(tokens_for_frames(100) * (c.n_window_infer / c.chunk()), 104);
    }

    #[test]
    fn sinusoid_table_is_sin_then_cos_and_starts_at_zero() {
        let p = sinusoids(13, 1280);
        assert!(p[..640].iter().all(|&v| v == 0.0));
        assert!(p[640..1280].iter().all(|&v| v == 1.0));
        // Channel 0 has timescale 1: sin(t).
        assert!((p[5 * 1280] - 5.0f32.sin()).abs() < 1e-7);
        // Last channel has timescale 1e4: sin(t / 1e4).
        assert!((p[5 * 1280 + 639] - (5.0f32 / 1e4).sin()).abs() < 1e-9);
    }

    /// The synthetic encoder the GPU gate compares against: finite, O(1) outputs, and its plan
    /// has the real geometry (a > 8 s clip gets 104-token windows; a batch pads to the longest
    /// chunk). CPU only; this does not test the GPU.
    #[test]
    fn synthetic_encoder_is_finite_and_uses_the_real_plan() {
        let enc = synthetic::small();
        let c = &enc.cfg;
        let long = synthetic::mel(c.n_mels, 1234, 3);
        let short = synthetic::mel(c.n_mels, 57, 5);
        let plan = AudioPlan::new(c, &[(&long, 1234), (&short, 57)], None).unwrap();
        assert_eq!(plan.width, 100);
        assert_eq!(plan.n_tokens, 161 + tokens_for_frames(57));
        assert_eq!(plan.windows, vec![(0, 104), (104, 57), (161, 8)]);
        let out = enc
            .encode_batch(&[(&long, 1234), (&short, 57)], None)
            .unwrap();
        assert_eq!(out.len(), plan.n_tokens * c.output_dim);
        assert!(out.iter().all(|v| v.is_finite()));
        let rms = (out.iter().map(|v| (v * v) as f64).sum::<f64>() / out.len() as f64).sqrt();
        assert!((0.05..20.0).contains(&rms), "synthetic output rms {rms}");
        // Two short clips: width 60, window 8 x tokens(60) = 64.
        let a = synthetic::mel(c.n_mels, 60, 7);
        let b = synthetic::mel(c.n_mels, 45, 9);
        let p2 = AudioPlan::new(c, &[(&a, 60), (&b, 45)], None).unwrap();
        assert_eq!(p2.width, 60);
        assert_eq!(p2.windows, vec![(0, 8), (8, tokens_for_frames(45))]);
    }

    #[test]
    fn windowed_attention_is_block_diagonal() {
        // Two windows; perturbing a row in window 1 must not move any output of window 0.
        let (n, d, heads) = (7, 8, 2);
        let mut s = 1u32;
        let mut qkv: Vec<f32> = (0..n * 3 * d)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s as f32 / u32::MAX as f32) - 0.5
            })
            .collect();
        let w = [(0, 4), (4, 3)];
        let a = windowed_attention(&qkv, n, d, heads, &w);
        for v in &mut qkv[5 * 3 * d..6 * 3 * d] {
            *v += 1.0;
        }
        let b = windowed_attention(&qkv, n, d, heads, &w);
        assert_eq!(a[..4 * d], b[..4 * d]);
        assert_ne!(a[4 * d..], b[4 * d..]);
        // One window over everything differs from two.
        let full = windowed_attention(&qkv, n, d, heads, &[(0, 7)]);
        assert_ne!(full[..4 * d], b[..4 * d]);
    }

    /// The M2 gate against the reference dumps (`scripts/ref/qwen3_omni/audio_encoder_ref.py`).
    ///
    /// Needs the audio tower's weights, so it is ignored by default. `ARF_QWEN_OMNI_AUDIO_TOWER`
    /// names the safetensors file (shard 1 of the checkpoint, or the `fetch_audio_tower.py`
    /// extract); the reference dumps are read from `dump/` next to it. Gate (plan section 6, M2):
    /// rel = ||a-b|| / ||b|| <= 1e-4 and min per-token cosine >= 0.9999 on `[N, 2048]`, token
    /// count exact, on every case in the dump manifest.
    #[test]
    #[ignore = "needs the 1.3 GB audio tower; ARF_QWEN_OMNI_AUDIO_TOWER=... cargo test --release -p arf-core qwen_audio_encoder -- --ignored --nocapture"]
    fn audio_encoder_matches_the_transformers_reference() {
        let path = std::env::var("ARF_QWEN_OMNI_AUDIO_TOWER")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                std::path::PathBuf::from(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../models/qwen3-omni-ref/audio_tower.safetensors"
                ))
            });
        if !path.exists() {
            eprintln!(
                "SKIP: audio tower not found at {} -- this run verified NOTHING",
                path.display()
            );
            return;
        }
        let dump = path.parent().unwrap().join("dump");
        let man: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dump.join("manifest.json")).unwrap())
                .unwrap();
        let t0 = std::time::Instant::now();
        let enc = QwenAudioEncoder::load(&path).unwrap();
        eprintln!(
            "[qwen-audio-enc] loaded {} in {:.2}s (CPU)",
            path.display(),
            t0.elapsed().as_secs_f64()
        );
        let read = |p: std::path::PathBuf| -> Vec<f32> {
            std::fs::read(&p)
                .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let d = enc.cfg.d_model;
        let od = enc.cfg.output_dim;
        let mut cases = 0;
        let mut controls = 0;
        for case in man["cases"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let shape: Vec<usize> = case["mel_shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let valid: Vec<usize> = case["valid_frames"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let (b, nm, t) = (shape[0], shape[1], shape[2]);
            let mel = read(dump.join(name).join("mel.f32"));
            assert_eq!(mel.len(), b * nm * t);
            let mels: Vec<LogMel> = (0..b)
                .map(|i| LogMel {
                    n_mels: nm,
                    n_frames: t,
                    data: mel[i * nm * t..(i + 1) * nm * t].to_vec(),
                })
                .collect();
            let clips: Vec<(&LogMel, usize)> = mels.iter().zip(valid.iter().copied()).collect();
            let mut taps = Vec::new();
            let t1 = std::time::Instant::now();
            let got = enc.encode_batch(&clips, Some(&mut taps)).unwrap();
            let secs = t1.elapsed().as_secs_f64();
            let want = read(dump.join(name).join("out.f32"));
            let n_want = case["tokens"].as_u64().unwrap() as usize;
            let n_formula: usize = valid.iter().map(|&v| tokens_for_frames(v)).sum();
            assert_eq!(n_formula, n_want, "{name}: token formula");
            assert_eq!(got.len(), n_want * od, "{name}: token count");
            assert_eq!(want.len(), got.len());
            let (rel, max_abs, min_cos, row_rel) = compare(&got, &want, od);
            // Per-layer rel, for locating a divergence (hidden.f32 = [34, N, 1280]).
            let hid = read(dump.join(name).join("hidden.f32"));
            assert_eq!(hid.len(), taps.len() * n_want * d);
            let layer_rel: Vec<f64> = taps
                .iter()
                .enumerate()
                .map(|(i, tp)| compare(tp, &hid[i * n_want * d..(i + 1) * n_want * d], d).0)
                .collect();
            let worst_layer = layer_rel.iter().cloned().fold(0.0f64, f64::max);
            eprintln!(
                "[qwen-audio-enc] {name:14} clips={b} frames={valid:?} tokens={n_want} \
                 rel={rel:.3e} max_token_rel={row_rel:.3e} max_abs={max_abs:.3e} 1-min_cos={:.1e} \
                 stem_rel={:.2e} worst_hidden_rel={worst_layer:.2e} cpu={secs:.2}s \
                 (reference torch cpu {}s)",
                1.0 - min_cos,
                layer_rel[0],
                case["cpu_seconds_f32"]
            );
            assert!(rel <= 1e-4, "{name}: rel {rel:.3e} > 1e-4");
            assert!(row_rel <= 1e-3, "{name}: worst token rel {row_rel:.3e}");
            assert!(min_cos >= 0.9999, "{name}: min cosine {min_cos}");
            cases += 1;
            // Negative controls (rule 8: the gate must be able to fail on this data).
            // (a) A clip longer than one window, attended as ONE window, must fail.
            if n_want > 104 && b == 1 {
                let one = enc
                    .encode_batch_windowed(&clips, None, Some(usize::MAX))
                    .unwrap();
                let (bad, ..) = compare(&one, &want, od);
                eprintln!("[qwen-audio-enc] {name:14} CONTROL one window over {n_want} tokens: rel={bad:.3e} (must fail)");
                assert!(bad > 1e-3, "{name}: windowing made no difference");
                controls += 1;
            }
            // (b) A batch's short clip encoded ALONE (padded to its own length, not the batch's
            // 100 frames) must differ from the reference's batched rows: the padding trap.
            if b > 1 {
                let last = clips.len() - 1;
                let alone = enc.encode(clips[last].0, clips[last].1).unwrap();
                let tail = &want[want.len() - alone.len()..];
                let (bad, ..) = compare(&alone, tail, od);
                eprintln!("[qwen-audio-enc] {name:14} CONTROL clip {last} alone: rel={bad:.3e} vs its batched rows (must differ)");
                assert!(bad > 1e-4, "{name}: batch padding made no difference");
                controls += 1;
            }
        }
        assert!(cases >= 3, "only {cases} reference cases in the dump");
        assert!(controls >= 2, "the negative controls did not run");
    }

    /// (rel = ||a-b|| / ||b||, max-abs, min per-row cosine, max per-row rel), in f64.
    fn compare(a: &[f32], b: &[f32], cols: usize) -> (f64, f64, f64, f64) {
        let (mut num, mut den, mut mx) = (0.0f64, 0.0f64, 0.0f64);
        let mut min_cos = 1.0f64;
        let mut max_row_rel = 0.0f64;
        for (ra, rb) in a.chunks(cols).zip(b.chunks(cols)) {
            let (mut ab, mut aa, mut bb, mut dd) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for (&x, &y) in ra.iter().zip(rb) {
                let (x, y) = (x as f64, y as f64);
                num += (x - y) * (x - y);
                den += y * y;
                mx = mx.max((x - y).abs());
                ab += x * y;
                aa += x * x;
                bb += y * y;
                dd += (x - y) * (x - y);
            }
            min_cos = min_cos.min(ab / (aa.sqrt() * bb.sqrt()).max(1e-300));
            max_row_rel = max_row_rel.max((dd / bb.max(1e-300)).sqrt());
        }
        ((num / den.max(1e-300)).sqrt(), mx, min_cos, max_row_rel)
    }
}
