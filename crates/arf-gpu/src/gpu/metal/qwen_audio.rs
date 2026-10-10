//! Qwen3-Omni audio encoder ("AuT") on the GPU (plain Metal, 2026-09-27) — the port of the CPU
//! encoder in `arf_core::model::qwen_audio_encoder`, which stays the reference and the fallback
//! (`ARF_AUDIO_CPU=1` selects it in `QwenAudio`).
//!
//! WHY: ~21 GFLOP per second of audio (16.4 in the 32 layers, 4.5 in the stem; the port plan
//! section 4, ESTIMATED). The CPU encoder takes 11.1 s for the 31 s reference clip
//! (`audio_encoder_matches_the_transformers_reference`, 2026-09-27 run, a CPU wall time and not a
//! ledger row).
//!
//! WHAT RUNS WHERE: the host cuts the call exactly as the CPU encoder does, through the SAME
//! public code (`AudioPlan::new`, `AudioPlan::padded_chunk`): chunks, the padded width (the
//! longest chunk IN THE CALL — the batch-padding trap), the attention windows. It also gathers the
//! per-token sinusoid rows (restarting at 0 in every chunk) from the encoder's own table. The GPU
//! runs everything with a FLOP in it, in ONE command buffer:
//!
//! | step | kernel |
//! |---|---|
//! | 3 x Conv2d 3x3 s2 + bias + erf GELU | `aud_conv_mm_{f32,bf16}` (implicit im2col), in groups of `STEM_GROUP` chunks |
//! | flatten `c*16+f` of the valid positions | `aud_flatten` |
//! | `conv_out` (no bias) + sinusoid | `vit_mm_bf16` with a zero bias and the position rows as its residual |
//! | 32 x (LN, qkv, windowed attention, out + residual, LN, fc1 + erf GELU, fc2 + residual) | `vit_layernorm` (eps 1e-5), `vit_mm_bf16`, `aud_attn_win_hd64` |
//! | `ln_post`, `proj1` + erf GELU, `proj2` | `vit_layernorm`, `vit_mm_bf16` |
//!
//! PRECISION: f32 activations and accumulation over the checkpoint's bf16 matrices widened exactly,
//! as the CPU encoder. NOT bf16 activations: their envelope on speech had a worst-token cosine of
//! 0.875 (the port plan, M2 progress note). GPU vs CPU differs by summation order, f32 LayerNorm
//! sums (the CPU uses f64) and the kernels' A&S erf (1.5e-7 absolute; the CPU's is f64).
//!
//! NOT RUN at the time of writing: the port was not run on a GPU when written;
//! the kernels were compiled offline with `xcrun metal` (the audio file appended to the
//! vision file, as the host does), which checks the MSL but not its results. The gates are
//! `tests/qwen_audio_gpu.rs`: synthetic weights vs the CPU encoder, and (ignored, needs the tower)
//! real weights vs the CPU encoder AND the transformers dumps. Its speed is NOT MEASURED.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use arf_core::model::qwen_audio::LogMel;
use arf_core::model::qwen_audio_encoder::{
    conv_out_len, tokens_for_frames, AudioEncoderConfig, AudioPlan, QwenAudioEncoder,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLLibrary,
};

use super::qwen_vision::{
    bind, bind_at, dispatch, new_buffer, new_buffer_with, read_f32, set_bytes, slice_bytes,
    write_f32, Buf, Enc, Pso, Rec, MSL as VISION_MSL,
};

const AUDIO_MSL: &str = include_str!("../../shaders/metal/qwen_audio_msl.metal");

/// Query rows per attention threadgroup (4 simdgroups x 8 rows), as in the vision encoder.
const ATTN_ROWS: usize = 32;

/// Chunks (1 s each) whose stem runs per group. The stem's activations are per group, not per
/// call: conv1's output alone is 6.1 MB per chunk at the real width (64 x 50 x 480 f32), so a
/// 5-minute clip in one group would need 1.8 GB. 8 chunks bound the stem scratch to ~65 MB.
pub const STEM_GROUP: usize = 8;

struct Pipes {
    mm_bf16: Pso,
    conv_bf16: Pso,
    conv_f32: Pso,
    flatten: Pso,
    ln: Pso,
    attn: Pso,
}

struct GpuLayer {
    ln1_w: Buf,
    ln1_b: Buf,
    qkv_w: Buf,
    qkv_b: Buf,
    o_w: Buf,
    o_b: Buf,
    ln2_w: Buf,
    ln2_b: Buf,
    fc1_w: Buf,
    fc1_b: Buf,
    fc2_w: Buf,
    fc2_b: Buf,
}

/// The stem's per-group activations, sized for [`STEM_GROUP`] chunks at the full chunk width.
struct StemScratch {
    a1: Buf,   // [G, h1, w1, ch]
    a2: Buf,   // [G, h2, w2, ch]
    a3: Buf,   // [G, h3, w3, ch]
    flat: Buf, // [G * w3, ch * h3]
}

/// Token activations for up to `rows` tokens; regrown on demand.
struct Scratch {
    rows: usize,
    h: Buf,    // [rows, d] residual stream
    x: Buf,    // [rows, d] LayerNorm output
    qkv: Buf,  // [rows + ATTN_ROWS rounded, 3d]; the rows past n are zeroed per encode
    attn: Buf, // [rows, d]
    up: Buf,   // [rows, ffn]
    p1: Buf,   // [rows, d] proj1 output
    out: Buf,  // [rows, output_dim]
}

/// The resident GPU audio encoder: weights uploaded once (~1.3 GB for the real tower), pipelines
/// compiled once, activations reused across calls.
pub struct QwenAudioGpu {
    cfg: AudioEncoderConfig,
    /// Host copy of the sinusoid rows a chunk can reach (`tokens_for_frames(chunk)` rows).
    pos: Vec<f32>,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipes: Pipes,
    conv1_w: Buf, // f32 [ch, 9]
    conv1_b: Buf,
    conv2_w: Buf, // bf16 [ch, ch*9]
    conv2_b: Buf,
    conv3_w: Buf,
    conv3_b: Buf,
    conv_out_w: Buf, // bf16 [d, ch*F]
    /// f32 zeros `[d]`: `conv_out` has no bias, `vit_mm` always adds one.
    zero_bias: Buf,
    layers: Vec<GpuLayer>,
    ln_post_w: Buf,
    ln_post_b: Buf,
    proj1_w: Buf,
    proj1_b: Buf,
    proj2_w: Buf,
    proj2_b: Buf,
    weight_bytes: usize,
    stem: Mutex<StemScratch>,
    scratch: Mutex<Option<Scratch>>,
}

// SAFETY: as `QwenVisionGpu`: device, queue, pipelines and buffers are thread-safe Metal objects;
// command buffers and encoders never leave `encode_batch_windowed`; the activations are behind the
// two mutexes, which it holds from the first write to the last read (stem first, then scratch —
// the only order they are ever taken in).
unsafe impl Send for QwenAudioGpu {}
unsafe impl Sync for QwenAudioGpu {}

impl QwenAudioGpu {
    /// Upload `enc`'s weights and compile the kernels. Errors (never panics) when there is no
    /// Metal device, a kernel does not compile, or the geometry is one the kernels do not cover
    /// (head_dim other than 64) — the caller keeps the CPU encoder then.
    pub fn new(enc: &QwenAudioEncoder) -> Result<Self, String> {
        let cfg = enc.cfg.clone();
        let (d, ch) = (cfg.d_model, cfg.conv_ch);
        if cfg.heads == 0 || d % cfg.heads != 0 || d / cfg.heads != 64 {
            return Err(format!(
                "d_model {d} / {} heads: only head_dim 64 has a GPU attention kernel",
                cfg.heads
            ));
        }
        super::super::relax_agx_watchdog();
        let device = objc2_metal::MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
        let queue = device.newCommandQueue().ok_or("no Metal command queue")?;
        // One library: the audio kernels use the vision file's tile constants and GELU, and the
        // host reuses its `vit_mm_bf16` and `vit_layernorm`.
        let src = format!("{VISION_MSL}\n{AUDIO_MSL}");
        let lib = device
            .newLibraryWithSource_options_error(
                &NSString::from_str(&src),
                Some(&super::island::msl_compile_options()),
            )
            .map_err(|e| format!("qwen_vision_msl.metal + qwen_audio_msl.metal: {e:?}"))?;
        let pso = |name: &str, threads: usize| -> Result<Pso, String> {
            let f = lib
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| format!("no kernel {name}"))?;
            let p = device
                .newComputePipelineStateWithFunction_error(&f)
                .map_err(|e| format!("pipeline {name}: {e:?}"))?;
            if p.maxTotalThreadsPerThreadgroup() < threads {
                return Err(format!(
                    "{name}: at most {} threads per threadgroup, needs {threads}",
                    p.maxTotalThreadsPerThreadgroup()
                ));
            }
            Ok(p)
        };
        let pipes = Pipes {
            mm_bf16: pso("vit_mm_bf16", 128)?,
            conv_bf16: pso("aud_conv_mm_bf16", 128)?,
            conv_f32: pso("aud_conv_mm_f32", 128)?,
            flatten: pso("aud_flatten", 256)?,
            ln: pso("vit_layernorm", 256)?,
            attn: pso("aud_attn_win_hd64", 128)?,
        };
        let mut bytes = 0usize;
        let mut up = |v: &[u8]| -> Result<Buf, String> {
            bytes += v.len();
            new_buffer_with(&device, v)
        };
        let mut layers = Vec::with_capacity(enc.layers.len());
        for l in &enc.layers {
            layers.push(GpuLayer {
                ln1_w: up(slice_bytes(&l.ln1_w))?,
                ln1_b: up(slice_bytes(&l.ln1_b))?,
                qkv_w: up(slice_bytes(&l.qkv_w))?,
                qkv_b: up(slice_bytes(&l.qkv_b))?,
                o_w: up(slice_bytes(&l.o_w))?,
                o_b: up(slice_bytes(&l.o_b))?,
                ln2_w: up(slice_bytes(&l.ln2_w))?,
                ln2_b: up(slice_bytes(&l.ln2_b))?,
                fc1_w: up(slice_bytes(&l.fc1_w))?,
                fc1_b: up(slice_bytes(&l.fc1_b))?,
                fc2_w: up(slice_bytes(&l.fc2_w))?,
                fc2_b: up(slice_bytes(&l.fc2_b))?,
            });
        }
        let conv1_w = up(slice_bytes(&enc.conv1_w))?;
        let conv1_b = up(slice_bytes(&enc.conv1_b))?;
        let conv2_w = up(slice_bytes(&enc.conv2_w))?;
        let conv2_b = up(slice_bytes(&enc.conv2_b))?;
        let conv3_w = up(slice_bytes(&enc.conv3_w))?;
        let conv3_b = up(slice_bytes(&enc.conv3_b))?;
        let conv_out_w = up(slice_bytes(&enc.conv_out_w))?;
        let ln_post_w = up(slice_bytes(&enc.ln_post_w))?;
        let ln_post_b = up(slice_bytes(&enc.ln_post_b))?;
        let proj1_w = up(slice_bytes(&enc.proj1_w))?;
        let proj1_b = up(slice_bytes(&enc.proj1_b))?;
        let proj2_w = up(slice_bytes(&enc.proj2_w))?;
        let proj2_b = up(slice_bytes(&enc.proj2_b))?;
        let zero_bias = new_buffer_with(&device, slice_bytes(&vec![0.0f32; d]))?;

        // The stem scratch, at the widest chunk a call can hold.
        let (h1, w1) = (conv_out_len(cfg.n_mels), conv_out_len(cfg.chunk()));
        let (h2, w2) = (conv_out_len(h1), conv_out_len(w1));
        let (h3, w3) = (conv_out_len(h2), conv_out_len(w2));
        let g = STEM_GROUP;
        let f = |len: usize| new_buffer(&device, len * 4);
        let stem = StemScratch {
            a1: f(g * h1 * w1 * ch)?,
            a2: f(g * h2 * w2 * ch)?,
            a3: f(g * h3 * w3 * ch)?,
            flat: f(g * w3 * ch * h3)?,
        };
        let keep = tokens_for_frames(cfg.chunk());
        if enc.pos.len() < keep * d {
            return Err(format!(
                "sinusoid table of {} rows, a chunk needs {keep}",
                enc.pos.len() / d
            ));
        }
        Ok(QwenAudioGpu {
            pos: enc.pos[..keep * d].to_vec(),
            cfg,
            device,
            queue,
            pipes,
            conv1_w,
            conv1_b,
            conv2_w,
            conv2_b,
            conv3_w,
            conv3_b,
            conv_out_w,
            zero_bias,
            layers,
            ln_post_w,
            ln_post_b,
            proj1_w,
            proj1_b,
            proj2_w,
            proj2_b,
            weight_bytes: bytes,
            stem: Mutex::new(stem),
            scratch: Mutex::new(None),
        })
    }

    pub fn cfg(&self) -> &AudioEncoderConfig {
        &self.cfg
    }

    /// Bytes of weights resident on the device.
    pub fn weight_bytes(&self) -> usize {
        self.weight_bytes
    }

    /// The device's name (`Apple M4 Max`), for the load line.
    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    /// One clip — the GPU twin of [`QwenAudioEncoder::encode`]: `[tokens_for_frames(valid), out]`.
    pub fn encode(&self, mel: &LogMel, valid_frames: usize) -> Result<Vec<f32>, String> {
        self.encode_batch(&[(mel, valid_frames)])
    }

    /// A processor batch as ONE call — the GPU twin of [`QwenAudioEncoder::encode_batch`], with
    /// the same padding and windowing (both come from [`AudioPlan`]). Errors on any GPU failure
    /// (a command-buffer error, a non-finite output) so the caller can fall back to the CPU.
    pub fn encode_batch(&self, clips: &[(&LogMel, usize)]) -> Result<Vec<f32>, String> {
        self.encode_batch_windowed(clips, None)
    }

    /// [`Self::encode_batch`] with the attention window forced to `window` tokens: a TEST
    /// CONTROL (one window over a > 8 s clip must fail the reference gate), not a feature.
    pub fn encode_batch_windowed(
        &self,
        clips: &[(&LogMel, usize)],
        window: Option<usize>,
    ) -> Result<Vec<f32>, String> {
        let c = &self.cfg;
        let d = c.d_model;
        let plan = AudioPlan::new(c, clips, window).map_err(|e| e.to_string())?;
        let n = plan.n_tokens;
        let width = plan.width;
        let nch = plan.chunks.len();
        let (h1, w1) = (conv_out_len(c.n_mels), conv_out_len(width));
        let (h2, w2) = (conv_out_len(h1), conv_out_len(w1));
        let (h3, w3) = (conv_out_len(h2), conv_out_len(w2));

        // Host inputs. Mel chunks [chunk, n_mels, width] (NHWC with cin = 1), the token table
        // (token -> chunk * w3 + position) and the per-token sinusoid rows.
        let mut mel = Vec::with_capacity(nch * c.n_mels * width);
        let mut src: Vec<u32> = Vec::with_capacity(n);
        let mut pos = Vec::with_capacity(n * d);
        let mut chunk_tok0 = Vec::with_capacity(nch + 1);
        for (ci, &(_, _, len)) in plan.chunks.iter().enumerate() {
            mel.extend(plan.padded_chunk(c, clips, ci));
            chunk_tok0.push(src.len());
            let keep = tokens_for_frames(len);
            if keep > w3 || keep * d > self.pos.len() {
                return Err(format!(
                    "chunk {ci}: {keep} tokens from {len} frames, stem width {w3}"
                ));
            }
            for t in 0..keep {
                src.push((ci * w3 + t) as u32);
                pos.extend_from_slice(&self.pos[t * d..(t + 1) * d]);
            }
        }
        chunk_tok0.push(src.len());
        if src.len() != n {
            return Err(format!("{} token rows, the plan says {n}", src.len()));
        }
        // Attention blocks: 32 query rows of one window each, (k_begin, k_end, q_start, 0).
        let mut blocks: Vec<u32> = Vec::new();
        for &(s, len) in &plan.windows {
            for b in (0..len).step_by(ATTN_ROWS) {
                blocks.extend_from_slice(&[s as u32, (s + len) as u32, (s + b) as u32, 0]);
            }
        }
        let n_blocks = blocks.len() / 4;
        let mel_buf = new_buffer_with(&self.device, slice_bytes(&mel))?;
        let src_buf = new_buffer_with(&self.device, slice_bytes(&src))?;
        let pos_buf = new_buffer_with(&self.device, slice_bytes(&pos))?;
        let blk_buf = new_buffer_with(&self.device, slice_bytes(&blocks))?;

        let stem = self
            .stem
            .lock()
            .map_err(|_| "audio stem scratch poisoned")?;
        let mut guard = self.scratch.lock().map_err(|_| "audio scratch poisoned")?;
        if guard.as_ref().is_none_or(|s| s.rows < n) {
            *guard = None; // free the old set before allocating the bigger one
            *guard = Some(self.scratch_for(n)?);
        }
        let s = guard.as_ref().expect("allocated above");
        // The attention kernel reads Q in whole 32-row blocks from each window's start: rows up
        // to n + 31 are read and must be finite.
        write_f32(&s.qkv, n * 3 * d, &vec![0.0f32; ATTN_ROWS * 3 * d]);

        let rec = Rec::labeled(&self.queue, "arf-qwen-audio")?;
        let enc = &rec.enc;
        let ch = c.conv_ch;
        for g0 in (0..nch).step_by(STEM_GROUP) {
            let g1 = (g0 + STEM_GROUP).min(nch);
            let gn = g1 - g0;
            let geo = |h, w, cin, ho, wo| ConvGeo {
                imgs: gn,
                h,
                w,
                cin,
                ho,
                wo,
                cout: ch,
            };
            self.conv(
                enc,
                (&mel_buf, g0 * c.n_mels * width),
                &self.conv1_w,
                false,
                &self.conv1_b,
                &stem.a1,
                geo(c.n_mels, width, 1, h1, w1),
            );
            self.conv(
                enc,
                (&stem.a1, 0),
                &self.conv2_w,
                true,
                &self.conv2_b,
                &stem.a2,
                geo(h1, w1, ch, h2, w2),
            );
            self.conv(
                enc,
                (&stem.a2, 0),
                &self.conv3_w,
                true,
                &self.conv3_b,
                &stem.a3,
                geo(h2, w2, ch, h3, w3),
            );
            let (t0, t1) = (chunk_tok0[g0], chunk_tok0[g1]);
            let nt = t1 - t0;
            let flat = ch * h3;
            enc.setComputePipelineState(&self.pipes.flatten);
            bind_at(enc, &stem.a3, 0);
            bind_off(enc, &src_buf, t0, 1);
            bind_at(enc, &stem.flat, 2);
            set_bytes(
                enc,
                &[
                    nt as u32,
                    ch as u32,
                    h3 as u32,
                    w3 as u32,
                    (g0 * w3) as u32,
                    0,
                    0,
                    0,
                ],
                3,
            );
            dispatch(enc, [(nt * flat).div_ceil(256), 1, 1], 256);
            // conv_out, + the sinusoid rows (fused as the residual: h = (flat . W^T + 0) + pos,
            // the CPU's `a + p` with the same operands).
            self.mm(
                enc,
                (&stem.flat, 0),
                &self.conv_out_w,
                (&s.h, t0 * d),
                nt,
                flat,
                d,
                &self.zero_bias,
                0,
                Some((&pos_buf, t0 * d)),
            );
        }
        for l in &self.layers {
            self.layer(enc, s, l, n, &blk_buf, n_blocks);
        }
        self.ln(enc, &s.h, &self.ln_post_w, &self.ln_post_b, &s.x, n, d);
        self.mm(
            enc,
            (&s.x, 0),
            &self.proj1_w,
            (&s.p1, 0),
            n,
            d,
            d,
            &self.proj1_b,
            2,
            None,
        );
        self.mm(
            enc,
            (&s.p1, 0),
            &self.proj2_w,
            (&s.out, 0),
            n,
            d,
            c.output_dim,
            &self.proj2_b,
            0,
            None,
        );
        rec.finish()?;
        let out = read_f32(&s.out, n * c.output_dim);
        if let Some(i) = out.iter().position(|v| !v.is_finite()) {
            return Err(format!("non-finite output at element {i} of {}", out.len()));
        }
        Ok(out)
    }

    fn scratch_for(&self, n: usize) -> Result<Scratch, String> {
        let c = &self.cfg;
        let d = c.d_model;
        let rows = n.div_ceil(ATTN_ROWS) * ATTN_ROWS;
        let f = |len: usize| new_buffer(&self.device, len * 4);
        Ok(Scratch {
            rows,
            h: f(rows * d)?,
            x: f(rows * d)?,
            qkv: f((rows + ATTN_ROWS) * 3 * d)?,
            attn: f(rows * d)?,
            up: f(rows * c.ffn)?,
            p1: f(rows * d)?,
            out: f(rows * c.output_dim)?,
        })
    }

    /// One pre-LN encoder layer on `s.h` (n rows).
    fn layer(&self, enc: &Enc, s: &Scratch, l: &GpuLayer, n: usize, blk: &Buf, n_blocks: usize) {
        let c = &self.cfg;
        let d = c.d_model;
        self.ln(enc, &s.h, &l.ln1_w, &l.ln1_b, &s.x, n, d);
        self.mm(
            enc,
            (&s.x, 0),
            &l.qkv_w,
            (&s.qkv, 0),
            n,
            d,
            3 * d,
            &l.qkv_b,
            0,
            None,
        );
        let scale = 1.0f32 / ((d / c.heads) as f32).sqrt();
        enc.setComputePipelineState(&self.pipes.attn);
        bind(enc, &[&s.qkv, &s.attn]);
        set_bytes(
            enc,
            &[n as u32, d as u32, c.heads as u32, scale.to_bits()],
            2,
        );
        bind_at(enc, blk, 3);
        dispatch(enc, [n_blocks, c.heads, 1], 128);
        self.mm(
            enc,
            (&s.attn, 0),
            &l.o_w,
            (&s.h, 0),
            n,
            d,
            d,
            &l.o_b,
            0,
            Some((&s.h, 0)),
        );
        self.ln(enc, &s.h, &l.ln2_w, &l.ln2_b, &s.x, n, d);
        self.mm(
            enc,
            (&s.x, 0),
            &l.fc1_w,
            (&s.up, 0),
            n,
            d,
            c.ffn,
            &l.fc1_b,
            2,
            None,
        );
        self.mm(
            enc,
            (&s.up, 0),
            &l.fc2_w,
            (&s.h, 0),
            n,
            c.ffn,
            d,
            &l.fc2_b,
            0,
            Some((&s.h, 0)),
        );
    }

    /// `c[m, n] = act(a[m, k] . w[n, k]^T + bias) (+ res)` with `vit_mm_bf16`; every buffer
    /// operand is `(buffer, element offset)`. `res` may be `c` itself (same offset).
    #[allow(clippy::too_many_arguments)]
    fn mm(
        &self,
        enc: &Enc,
        a: (&Buf, usize),
        w: &Buf,
        c: (&Buf, usize),
        m: usize,
        k: usize,
        n: usize,
        bias: &Buf,
        act: u32,
        res: Option<(&Buf, usize)>,
    ) {
        let dims = [
            m as u32,
            k as u32,
            n as u32,
            act,
            res.is_some() as u32,
            0,
            0,
            0,
        ];
        enc.setComputePipelineState(&self.pipes.mm_bf16);
        bind_off(enc, a.0, a.1, 0);
        bind_at(enc, w, 1);
        bind_off(enc, c.0, c.1, 2);
        set_bytes(enc, &dims, 3);
        bind_at(enc, bias, 4);
        let (r, ro) = res.unwrap_or(c);
        bind_off(enc, r, ro, 5);
        dispatch(enc, [m.div_ceil(32), n.div_ceil(64), 1], 128);
    }

    /// Conv2d 3x3 stride 2 pad 1 + bias + erf GELU over `g.imgs` NHWC images at `x` (element
    /// offset) into `out` (`[imgs*ho*wo, cout]`).
    #[allow(clippy::too_many_arguments)]
    fn conv(
        &self,
        enc: &Enc,
        x: (&Buf, usize),
        w: &Buf,
        bf16: bool,
        bias: &Buf,
        out: &Buf,
        g: ConvGeo,
    ) {
        let m = g.imgs * g.ho * g.wo;
        let k = g.cin * 9;
        let dims = [
            m as u32,
            k as u32,
            g.cout as u32,
            2, // erf GELU
            g.h as u32,
            g.w as u32,
            g.cin as u32,
            g.ho as u32,
            g.wo as u32,
            0,
            0,
            0,
        ];
        enc.setComputePipelineState(if bf16 {
            &self.pipes.conv_bf16
        } else {
            &self.pipes.conv_f32
        });
        bind_off(enc, x.0, x.1, 0);
        bind_at(enc, w, 1);
        bind_at(enc, out, 2);
        set_bytes(enc, &dims, 3);
        bind_at(enc, bias, 4);
        dispatch(enc, [m.div_ceil(32), g.cout.div_ceil(64), 1], 128);
    }

    fn ln(&self, enc: &Enc, x: &Buf, w: &Buf, b: &Buf, y: &Buf, rows: usize, cols: usize) {
        enc.setComputePipelineState(&self.pipes.ln);
        bind(enc, &[x, w, b, y]);
        set_bytes(
            enc,
            &[rows as u32, cols as u32, self.cfg.eps.to_bits(), 0],
            4,
        );
        dispatch(enc, [rows, 1, 1], 256);
    }
}

/// Geometry of one stem conv over a group of chunks.
#[derive(Clone, Copy)]
struct ConvGeo {
    imgs: usize,
    h: usize,
    w: usize,
    cin: usize,
    ho: usize,
    wo: usize,
    cout: usize,
}

/// Bind `b` at element offset `off` (f32 or u32 elements: 4 bytes each).
fn bind_off(enc: &Enc, b: &Buf, off: usize, i: usize) {
    // SAFETY: a live buffer; every caller's offset is inside it (the extents are the plan's).
    unsafe { enc.setBuffer_offset_atIndex(Some(b), off * 4, i) };
}

/// Whether this machine has a Metal device at all (tests skip without one).
pub fn device_available() -> bool {
    super::qwen_vision::device_available()
}

/// The audio encoder a caller holds: the GPU port when it comes up, the CPU encoder otherwise —
/// the switch the vision encoder has in `arf-serve` (`QwenVision`), here so a future caller
/// (M3's `input_audio`) switches with one line.
///
/// - `ARF_AUDIO_CPU=1` (read at load) selects the CPU encoder and never builds the GPU one.
/// - With the GPU up, the CPU encoder's weights are DROPPED after the upload (the GPU holds its
///   own copy); a GPU failure on a call logs one line and falls back to the CPU encoder, which is
///   reloaded from the tower file the first time that happens and kept after.
pub struct QwenAudio {
    pub cfg: AudioEncoderConfig,
    path: Option<PathBuf>,
    gpu: Option<QwenAudioGpu>,
    cpu: Mutex<Option<Arc<QwenAudioEncoder>>>,
}

impl QwenAudio {
    /// Load the tower (`thinker.audio_tower.*` safetensors) and, unless `ARF_AUDIO_CPU` is set,
    /// bring up the GPU encoder. A GPU that does not come up is one log line and the CPU
    /// encoder, not an error.
    pub fn load(path: &Path) -> Result<Self, String> {
        let enc = QwenAudioEncoder::load(path).map_err(|e| e.to_string())?;
        let mut me = QwenAudio::from_encoder(enc);
        me.path = Some(path.to_path_buf());
        if me.gpu.is_some() {
            *me.cpu.lock().map_err(|_| "audio cpu poisoned")? = None;
        }
        Ok(me)
    }

    /// Wrap an already-loaded encoder (kept as the fallback: there is no file to reload from).
    pub fn from_encoder(enc: QwenAudioEncoder) -> Self {
        let cpu_only = std::env::var_os("ARF_AUDIO_CPU").is_some();
        let gpu = if cpu_only {
            None
        } else {
            match QwenAudioGpu::new(&enc) {
                Ok(g) => {
                    eprintln!(
                        "[qwen-audio] GPU encoder on {} ({:.2} GB of weights)",
                        g.device_name(),
                        g.weight_bytes() as f64 / 1e9
                    );
                    Some(g)
                }
                Err(e) => {
                    eprintln!("[qwen-audio] GPU encoder unavailable ({e}); using the CPU encoder");
                    None
                }
            }
        };
        QwenAudio {
            cfg: enc.cfg.clone(),
            path: None,
            gpu,
            cpu: Mutex::new(Some(Arc::new(enc))),
        }
    }

    /// "GPU" or "CPU": which encoder serves calls (before any fallback).
    pub fn backend(&self) -> &'static str {
        if self.gpu.is_some() {
            "GPU"
        } else {
            "CPU"
        }
    }

    /// One clip; see [`QwenAudioEncoder::encode`].
    pub fn encode(&self, mel: &LogMel, valid_frames: usize) -> Result<Vec<f32>, String> {
        self.encode_batch(&[(mel, valid_frames)])
    }

    /// A processor batch as one call; see [`QwenAudioEncoder::encode_batch`].
    pub fn encode_batch(&self, clips: &[(&LogMel, usize)]) -> Result<Vec<f32>, String> {
        if let Some(g) = &self.gpu {
            match g.encode_batch(clips) {
                Ok(v) => return Ok(v),
                Err(e) => eprintln!("[qwen-audio] GPU encode failed ({e}); using the CPU encoder"),
            }
        }
        self.cpu_encoder()?
            .encode_batch(clips, None)
            .map_err(|e| e.to_string())
    }

    fn cpu_encoder(&self) -> Result<Arc<QwenAudioEncoder>, String> {
        let mut g = self.cpu.lock().map_err(|_| "audio cpu poisoned")?;
        if let Some(e) = g.as_ref() {
            return Ok(e.clone());
        }
        let path = self
            .path
            .as_ref()
            .ok_or("no CPU audio encoder to fall back to")?;
        let e = Arc::new(QwenAudioEncoder::load(path).map_err(|e| e.to_string())?);
        *g = Some(e.clone());
        Ok(e)
    }
}
