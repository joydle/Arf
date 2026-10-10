//! Qwen3.8 vision encoder on the GPU (plain Metal, 2026-09-27) — the port of the CPU encoder in
//! `arf_core::model::qwen_vision`, which stays the reference and the fallback.
//!
//! WHY: the CPU encoder takes 5.16 s for a 512x384 image (192 merged tokens,
//! measured 2026-09-27), and an agent sends a screenshot on every image turn.
//! The forward is ~0.72 TFLOP at that size (27 blocks of qkv/o/up/down over 768 patches, ~26
//! GFLOP each, plus the merger), which is a GPU-sized job.
//!
//! WHAT RUNS WHERE: the host does exactly what the CPU encoder's host code does — resize and
//! normalise (`preprocess`, unchanged), cut the patches in merge order, resize the position table,
//! build the rope tables — through the SAME public functions (`patches_in_merge_order`,
//! `pos_table_resized`, `rope_tables`), so every input the GPU sees is bit-identical to the CPU's.
//! The GPU runs everything with a FLOP in it (`qwen_vision_msl.metal`): the patch embedding as a
//! GEMM over unfolded patches (f32 weights, bias and position fused into the store), 27 x
//! (LayerNorm, qkv GEMM, rope, attention, out GEMM + residual, LayerNorm, up GEMM + GELU, down GEMM
//! + residual), the post-LN and the two merger GEMMs — ~220 dispatches in ONE command buffer.
//!
//! PRECISION: f32 activations, f32 accumulation, the mmproj's bf16 matrices widened exactly (the
//! device buffers hold the same bf16 bits the CPU encoder holds). Nothing is rounded to f16 or
//! bf16, so GPU vs CPU differs only by summation order and a few transcendental implementations —
//! see the kernel file's header.
//!
//! DEVICE: `MTLCreateSystemDefaultDevice` (the one GPU of an Apple-silicon Mac — the device the
//! text model's wgpu context also opens) with a queue of its own. It does not share the island's
//! queue: an image encode is a one-shot job between requests, and the island's queue carries the
//! decode's pipelined records.
//!
//! NOT MEASURED at the time of writing (the port was not run on a GPU when written): its speed, and
//! its agreement with the CPU encoder. `examples/qwen_vision_gpu_parity.rs` measures both.
//! MEASURED since (measured 2026-09-27): 512x384 in 285 ms
//! warm vs the CPU's 5.22 s, rel 1.95e-5; 1024x768 in 1.23 s vs 41.0 s, rel 2.57e-5.

use std::sync::Mutex;

use arf_core::model::qwen_vision::{
    patches_in_merge_order, patches_pair_in_merge_order, pos_table_resized, rope_tables, QwenImage,
    QwenVisionConfig, QwenVisionEncoder,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLDispatchType, MTLLibrary,
    MTLResourceOptions, MTLSize,
};

// `pub(super)` from here down: the Qwen3-Omni audio encoder (`qwen_audio.rs`) is built on these
// same helpers and on this file's kernels.
pub(super) type Buf = Retained<ProtocolObject<dyn MTLBuffer>>;
pub(super) type Pso = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub(super) type Enc = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;
type Cmd = Retained<ProtocolObject<dyn MTLCommandBuffer>>;

pub(super) const MSL: &str = include_str!("../../shaders/metal/qwen_vision_msl.metal");

/// Query rows per attention threadgroup (the kernel's 4 simdgroups x 8 rows). The qkv scratch
/// holds a whole number of these blocks, and the rows past `n` are zeroed before every encode.
const ATTN_ROWS: usize = 32;

struct Pipes {
    mm_bf16: Pso,
    mm_f32: Pso,
    ln: Pso,
    rope: Pso,
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
    up_w: Buf,
    up_b: Buf,
    down_w: Buf,
    down_b: Buf,
}

/// Activations, sized for `rows` patches (a multiple of [`ATTN_ROWS`]); regrown on demand.
struct Scratch {
    rows: usize,
    /// Patch width the `patches` buffer holds: `3*patch*patch` for images, twice that once a video
    /// frame pair has been encoded.
    patch_k: usize,
    patches: Buf, // [rows, patch_k]
    pos: Buf,     // [rows, hidden], position rows gathered into merge order
    cos: Buf,     // [rows, head_dim/2]
    sin: Buf,
    h: Buf,    // [rows, hidden] residual stream
    x: Buf,    // [rows, hidden] LayerNorm output
    qkv: Buf,  // [rows, 3*hidden]
    attn: Buf, // [rows, hidden]
    up: Buf,   // [rows, ffn]
    m1: Buf,   // [rows/4, 4*hidden] merger hidden
    out: Buf,  // [rows/4, proj_dim]
}

/// The resident GPU encoder: weights uploaded once (~0.93 GB for the 27B's mmproj), pipelines
/// compiled once, activations reused across images.
pub struct QwenVisionGpu {
    cfg: QwenVisionConfig,
    /// Host copy of the learned position table (`[pos_grid^2, hidden]`, 10.6 MB): it is resized
    /// on the host by the CPU encoder's own function.
    pos_embd: Vec<f32>,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipes: Pipes,
    patch_w: Buf, // f32 [hidden, 3*patch*patch] (both temporal taps, pre-summed)
    /// f32 [hidden, 2*3*patch*patch]: the taps side by side, for video frame pairs
    /// (`QwenVisionWeights::patch_w_pair`, 7 MB for the 27B's mmproj).
    patch_w_pair: Buf,
    patch_b: Buf,
    layers: Vec<GpuLayer>,
    post_ln_w: Buf,
    post_ln_b: Buf,
    mm0_w: Buf,
    mm0_b: Buf,
    mm2_w: Buf,
    mm2_b: Buf,
    weight_bytes: usize,
    scratch: Mutex<Option<Scratch>>,
}

// SAFETY: MTLDevice, MTLCommandQueue, MTLComputePipelineState and MTLBuffer are documented
// thread-safe (the island wraps its buffers the same way, `MtlBuf`). Command buffers and
// encoders are created per `encode` and never leave it, and the shared activations are behind
// the `scratch` mutex, which `encode` holds from the first write to the last read.
unsafe impl Send for QwenVisionGpu {}
unsafe impl Sync for QwenVisionGpu {}

pub(super) fn slice_bytes<T: bytemuck::Pod>(v: &[T]) -> &[u8] {
    bytemuck::cast_slice(v)
}

/// Whether this machine has a Metal device at all (tests skip without one; any OTHER failure of
/// [`QwenVisionGpu::new`] is a real failure).
pub fn device_available() -> bool {
    objc2_metal::MTLCreateSystemDefaultDevice().is_some()
}

impl QwenVisionGpu {
    /// Upload `enc`'s weights and compile the kernels. Errors (never panics) when there is no
    /// Metal device, a kernel does not compile, or the geometry is one the kernels do not cover
    /// (head_dim not in 8/16/32/64/72/80) — the caller keeps the CPU encoder then.
    pub fn new(enc: &QwenVisionEncoder) -> Result<Self, String> {
        let cfg = enc.cfg.clone();
        let hd = cfg.head_dim();
        if !matches!(hd, 8 | 16 | 32 | 64 | 72 | 80) {
            return Err(format!("head_dim {hd} has no GPU attention kernel"));
        }
        if cfg.merge != 2 {
            return Err(format!("spatial_merge_size {} (only 2)", cfg.merge));
        }
        super::super::relax_agx_watchdog();
        let device = objc2_metal::MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
        let queue = device.newCommandQueue().ok_or("no Metal command queue")?;
        let lib = device
            .newLibraryWithSource_options_error(
                &NSString::from_str(MSL),
                Some(&super::island::msl_compile_options()),
            )
            .map_err(|e| format!("qwen_vision_msl.metal: {e:?}"))?;
        let pso = |name: &str, threads: usize| -> Result<Pso, String> {
            let f = lib
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| format!("no kernel {name}"))?;
            let p = device
                .newComputePipelineStateWithFunction_error(&f)
                .map_err(|e| format!("pipeline {name}: {e:?}"))?;
            // A kernel whose register use caps the threadgroup below what it is dispatched with
            // would run a partial threadgroup silently; refuse it instead.
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
            mm_f32: pso("vit_mm_f32", 128)?,
            ln: pso("vit_layernorm", 256)?,
            rope: pso("vit_rope", 256)?,
            attn: pso(&format!("vit_attn_hd{hd}"), 128)?,
        };
        let mut bytes = 0usize;
        let mut up = |v: &[u8]| -> Result<Buf, String> {
            bytes += v.len();
            new_buffer_with(&device, v)
        };
        let w = &enc.w;
        let mut layers = Vec::with_capacity(w.layers.len());
        for l in &w.layers {
            layers.push(GpuLayer {
                ln1_w: up(slice_bytes(&l.ln1_w))?,
                ln1_b: up(slice_bytes(&l.ln1_b))?,
                qkv_w: up(slice_bytes(&l.qkv_w))?,
                qkv_b: up(slice_bytes(&l.qkv_b))?,
                o_w: up(slice_bytes(&l.o_w))?,
                o_b: up(slice_bytes(&l.o_b))?,
                ln2_w: up(slice_bytes(&l.ln2_w))?,
                ln2_b: up(slice_bytes(&l.ln2_b))?,
                up_w: up(slice_bytes(&l.up_w))?,
                up_b: up(slice_bytes(&l.up_b))?,
                down_w: up(slice_bytes(&l.down_w))?,
                down_b: up(slice_bytes(&l.down_b))?,
            });
        }
        let patch_w = up(slice_bytes(&w.patch_w))?;
        let patch_w_pair = up(slice_bytes(&w.patch_w_pair))?;
        let patch_b = up(slice_bytes(&w.patch_b))?;
        let post_ln_w = up(slice_bytes(&w.post_ln_w))?;
        let post_ln_b = up(slice_bytes(&w.post_ln_b))?;
        let mm0_w = up(slice_bytes(&w.mm0_w))?;
        let mm0_b = up(slice_bytes(&w.mm0_b))?;
        let mm2_w = up(slice_bytes(&w.mm2_w))?;
        let mm2_b = up(slice_bytes(&w.mm2_b))?;
        Ok(QwenVisionGpu {
            cfg,
            pos_embd: w.pos_embd.clone(),
            device,
            queue,
            pipes,
            patch_w,
            patch_w_pair,
            patch_b,
            layers,
            post_ln_w,
            post_ln_b,
            mm0_w,
            mm0_b,
            mm2_w,
            mm2_b,
            weight_bytes: bytes,
            scratch: Mutex::new(None),
        })
    }

    pub fn cfg(&self) -> &QwenVisionConfig {
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

    fn scratch_for(&self, n: usize, patch_k: usize) -> Result<Scratch, String> {
        let c = &self.cfg;
        let rows = n.div_ceil(ATTN_ROWS) * ATTN_ROWS;
        let (d, half) = (c.hidden, c.head_dim() / 2);
        let f = |len: usize| new_buffer(&self.device, len * 4);
        Ok(Scratch {
            rows,
            patch_k,
            patches: f(rows * patch_k)?,
            pos: f(rows * d)?,
            cos: f(rows * half)?,
            sin: f(rows * half)?,
            h: f(rows * d)?,
            x: f(rows * d)?,
            qkv: f(rows * 3 * d)?,
            attn: f(rows * d)?,
            up: f(rows * c.ffn)?,
            m1: f(rows / 4 * 4 * d)?,
            out: f(rows / 4 * c.proj_dim)?,
        })
    }

    /// Full forward on a preprocessed image: `[n_patches/4, proj_dim]`, the same layout as
    /// [`QwenVisionEncoder::encode`]. Errors on any GPU failure (command-buffer error, a
    /// non-finite output) so the caller can fall back to the CPU.
    pub fn encode(&self, img: &QwenImage) -> Result<Vec<f32>, String> {
        check_grid(img)?;
        // Host inputs, from the CPU encoder's own code.
        let (patches, rc) = patches_in_merge_order(&self.cfg, img);
        let pd = 3 * self.cfg.patch * self.cfg.patch;
        self.forward(&patches, pd, &self.patch_w, &rc, img.grid_w, img.grid_h)
    }

    /// One VIDEO frame pair (frames 2k, 2k+1) — the GPU twin of
    /// [`QwenVisionEncoder::encode_frame_pair`]: the image forward with `[a | b]` patches against
    /// the side-by-side taps. NOT RUN on a GPU at the time of writing (
    /// `tests/qwen_vision_gpu.rs` `qwen_vision_gpu_frame_pair_matches_cpu` is its gate). Run
    /// since: the gate passes, rel 2.1e-6 vs the CPU (measured 2026-09-27, "Qwen3.8
    /// watches video").
    pub fn encode_frame_pair(&self, a: &QwenImage, b: &QwenImage) -> Result<Vec<f32>, String> {
        check_grid(a)?;
        if (a.width, a.height) != (b.width, b.height) {
            return Err(format!(
                "frame pair of {}x{} and {}x{}",
                a.width, a.height, b.width, b.height
            ));
        }
        let (patches, rc) = patches_pair_in_merge_order(&self.cfg, a, b);
        let pd = 3 * self.cfg.patch * self.cfg.patch;
        self.forward(
            &patches,
            2 * pd,
            &self.patch_w_pair,
            &rc,
            a.grid_w,
            a.grid_h,
        )
    }

    /// The forward from host-cut patches (`[n, pk]`, merge order, cells `rc`) against the f32
    /// patch weight `patch_w` (`[hidden, pk]`).
    fn forward(
        &self,
        patches: &[f32],
        pk: usize,
        patch_w: &Buf,
        rc: &[(usize, usize)],
        grid_w: usize,
        grid_h: usize,
    ) -> Result<Vec<f32>, String> {
        let c = &self.cfg;
        let d = c.hidden;
        let n = rc.len();
        let table = pos_table_resized(c, &self.pos_embd, grid_w, grid_h);
        let mut pos = Vec::with_capacity(n * d);
        for &(py, px) in rc {
            let r = py * grid_w + px;
            pos.extend_from_slice(&table[r * d..(r + 1) * d]);
        }
        let (cos, sin) = rope_tables(c, rc);

        let mut guard = self.scratch.lock().map_err(|_| "vision scratch poisoned")?;
        if guard.as_ref().is_none_or(|s| s.rows < n || s.patch_k < pk) {
            // Keep the wider patch buffer once a video has needed it.
            let pk = pk.max(guard.as_ref().map_or(0, |s| s.patch_k));
            *guard = None; // free the old set before allocating the bigger one
            *guard = Some(self.scratch_for(n, pk)?);
        }
        let s = guard.as_ref().expect("allocated above");
        write_f32(&s.patches, 0, patches);
        write_f32(&s.pos, 0, &pos);
        write_f32(&s.cos, 0, &cos);
        write_f32(&s.sin, 0, &sin);
        // The attention kernel reads Q from whole 32-row blocks: the rows past n must be finite.
        let pad = n.div_ceil(ATTN_ROWS) * ATTN_ROWS - n;
        write_f32(&s.qkv, n * 3 * d, &vec![0.0f32; pad * 3 * d]);

        let sums = std::env::var_os("ARF_VISION_SUM").is_some();
        let mut rec = Rec::new(&self.queue)?;
        // 1) patch embed + bias, + the position rows (fused: h = (P.W^T + b) + pos)
        self.mm(
            &rec.enc,
            &s.patches,
            patch_w,
            false,
            &s.h,
            n,
            pk,
            d,
            &self.patch_b,
            0,
            Some(&s.pos),
        );
        if sums {
            rec = rec.flush(&self.queue)?;
            vsum_gpu("inp_pos_emb", &s.h, n * d);
        }
        for (li, l) in self.layers.iter().enumerate() {
            self.block(&rec.enc, s, l, n);
            if sums && (li == 0 || li + 1 == self.layers.len()) {
                rec = rec.flush(&self.queue)?;
                vsum_gpu(
                    if li == 0 {
                        "layer_out 0"
                    } else {
                        "layer_out last"
                    },
                    &s.h,
                    n * d,
                );
            }
        }
        // post-LN, then the merger over [n/4, 4d] (rows are already in merge order)
        self.ln(&rec.enc, &s.h, &self.post_ln_w, &self.post_ln_b, &s.x, n, d);
        let (m, d4) = (n / 4, 4 * d);
        self.mm(
            &rec.enc,
            &s.x,
            &self.mm0_w,
            true,
            &s.m1,
            m,
            d4,
            d4,
            &self.mm0_b,
            2,
            None,
        );
        self.mm(
            &rec.enc,
            &s.m1,
            &self.mm2_w,
            true,
            &s.out,
            m,
            d4,
            c.proj_dim,
            &self.mm2_b,
            0,
            None,
        );
        rec.finish()?;
        let out = read_f32(&s.out, m * c.proj_dim);
        if sums {
            vsum_gpu("embeddings", &s.out, m * c.proj_dim);
        }
        if let Some(i) = out.iter().position(|v| !v.is_finite()) {
            return Err(format!("non-finite output at element {i} of {}", out.len()));
        }
        Ok(out)
    }

    /// One pre-LN ViT block on `s.h` (n rows), encoded into `enc`.
    fn block(&self, enc: &Enc, s: &Scratch, l: &GpuLayer, n: usize) {
        let c = &self.cfg;
        let d = c.hidden;
        self.ln(enc, &s.h, &l.ln1_w, &l.ln1_b, &s.x, n, d);
        self.mm(
            enc,
            &s.x,
            &l.qkv_w,
            true,
            &s.qkv,
            n,
            d,
            3 * d,
            &l.qkv_b,
            0,
            None,
        );
        self.rope(enc, s, n);
        self.attn(enc, &s.qkv, &s.attn, n);
        self.mm(
            enc,
            &s.attn,
            &l.o_w,
            true,
            &s.h,
            n,
            d,
            d,
            &l.o_b,
            0,
            Some(&s.h),
        );
        self.ln(enc, &s.h, &l.ln2_w, &l.ln2_b, &s.x, n, d);
        self.mm(
            enc, &s.x, &l.up_w, true, &s.up, n, d, c.ffn, &l.up_b, 1, None,
        );
        self.mm(
            enc,
            &s.up,
            &l.down_w,
            true,
            &s.h,
            n,
            c.ffn,
            d,
            &l.down_b,
            0,
            Some(&s.h),
        );
    }

    /// `c[m, n] = act(a[m, k] . w[n, k]^T + bias) (+ res)`; `res` may be `c` itself.
    fn mm(
        &self,
        enc: &Enc,
        a: &Buf,
        w: &Buf,
        bf16: bool,
        c: &Buf,
        m: usize,
        k: usize,
        n: usize,
        bias: &Buf,
        act: u32,
        res: Option<&Buf>,
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
        enc.setComputePipelineState(if bf16 {
            &self.pipes.mm_bf16
        } else {
            &self.pipes.mm_f32
        });
        bind(enc, &[a, w, c]);
        set_bytes(enc, &dims, 3);
        bind_at(enc, bias, 4);
        bind_at(enc, res.unwrap_or(c), 5);
        dispatch(enc, [m.div_ceil(32), n.div_ceil(64), 1], 128);
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

    fn rope(&self, enc: &Enc, s: &Scratch, n: usize) {
        let c = &self.cfg;
        let half = c.head_dim() / 2;
        enc.setComputePipelineState(&self.pipes.rope);
        bind(enc, &[&s.qkv, &s.cos, &s.sin]);
        set_bytes(
            enc,
            &[n as u32, c.hidden as u32, c.heads as u32, half as u32],
            3,
        );
        dispatch(enc, [(n * 2 * c.heads * half).div_ceil(256), 1, 1], 256);
    }

    fn attn(&self, enc: &Enc, qkv: &Buf, out: &Buf, n: usize) {
        let c = &self.cfg;
        let scale = 1.0f32 / (c.head_dim() as f32).sqrt();
        enc.setComputePipelineState(&self.pipes.attn);
        bind(enc, &[qkv, out]);
        set_bytes(
            enc,
            &[n as u32, c.hidden as u32, c.heads as u32, scale.to_bits()],
            2,
        );
        dispatch(enc, [n.div_ceil(ATTN_ROWS), c.heads, 1], 128);
    }
}

fn check_grid(img: &QwenImage) -> Result<(), String> {
    if img.grid_w == 0
        || img.grid_h == 0
        || !img.grid_w.is_multiple_of(2)
        || !img.grid_h.is_multiple_of(2)
    {
        return Err(format!(
            "patch grid {}x{} must be even and non-empty",
            img.grid_w, img.grid_h
        ));
    }
    Ok(())
}

/// One command buffer with one serial compute encoder: every dispatch sees the previous one's
/// writes (the dflash context commit encodes the same way).
pub(super) struct Rec {
    cmd: Cmd,
    pub(super) enc: Enc,
}

impl Rec {
    fn new(queue: &ProtocolObject<dyn MTLCommandQueue>) -> Result<Rec, String> {
        Rec::labeled(queue, "arf-qwen-vision")
    }

    /// A recorder whose command buffer carries `label` (what a GPU capture shows).
    pub(super) fn labeled(
        queue: &ProtocolObject<dyn MTLCommandQueue>,
        label: &str,
    ) -> Result<Rec, String> {
        let cmd = queue.commandBuffer().ok_or("no command buffer")?;
        super::island::label_cb(&cmd, label);
        let enc = cmd
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Serial)
            .ok_or("no compute encoder")?;
        Ok(Rec { cmd, enc })
    }

    /// End, commit, wait; an error status (a GPU fault, a watchdog kill) is returned, not lost.
    pub(super) fn finish(self) -> Result<(), String> {
        self.enc.endEncoding();
        self.cmd.commit();
        self.cmd.waitUntilCompleted();
        let st = self.cmd.status();
        if st != MTLCommandBufferStatus::Completed {
            return Err(format!(
                "command buffer status {st:?}: {:?}",
                self.cmd
                    .error()
                    .map(|e| e.localizedDescription().to_string())
            ));
        }
        Ok(())
    }

    /// `finish` and start a new one (only for the ARF_VISION_SUM readbacks).
    fn flush(self, queue: &ProtocolObject<dyn MTLCommandQueue>) -> Result<Rec, String> {
        self.finish()?;
        Rec::new(queue)
    }
}

pub(super) fn new_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: usize,
) -> Result<Buf, String> {
    device
        .newBufferWithLength_options(bytes.max(16), MTLResourceOptions::empty())
        .ok_or_else(|| format!("Metal buffer of {bytes} B"))
}

pub(super) fn new_buffer_with(
    device: &ProtocolObject<dyn MTLDevice>,
    data: &[u8],
) -> Result<Buf, String> {
    if data.is_empty() {
        return new_buffer(device, 16);
    }
    // SAFETY: `data` is valid for `data.len()` bytes; Metal copies it before returning.
    unsafe {
        device.newBufferWithBytes_length_options(
            std::ptr::NonNull::new(data.as_ptr() as *mut std::ffi::c_void).expect("non-null"),
            data.len(),
            MTLResourceOptions::empty(),
        )
    }
    .ok_or_else(|| format!("Metal buffer of {} B", data.len()))
}

/// Host write into a shared buffer at element `off` (the GPU is idle: `encode` writes before it
/// commits and holds the scratch lock).
pub(super) fn write_f32(b: &Buf, off: usize, v: &[f32]) {
    assert!(
        (off + v.len()) * 4 <= b.length(),
        "vision scratch write out of bounds"
    );
    // SAFETY: bounds checked above; shared storage is CPU-addressable; nothing on the GPU is
    // using this buffer (see above).
    unsafe {
        let dst = (b.contents().as_ptr() as *mut f32).add(off);
        std::ptr::copy_nonoverlapping(v.as_ptr(), dst, v.len());
    }
}

pub(super) fn read_f32(b: &Buf, len: usize) -> Vec<f32> {
    assert!(len * 4 <= b.length(), "vision scratch read out of bounds");
    // SAFETY: bounds checked; the command buffer that wrote it has completed.
    unsafe { std::slice::from_raw_parts(b.contents().as_ptr() as *const f32, len).to_vec() }
}

pub(super) fn bind(enc: &Enc, bufs: &[&Buf]) {
    for (i, b) in bufs.iter().enumerate() {
        bind_at(enc, b, i);
    }
}

pub(super) fn bind_at(enc: &Enc, b: &Buf, i: usize) {
    // SAFETY: a live buffer at offset 0.
    unsafe { enc.setBuffer_offset_atIndex(Some(b), 0, i) };
}

pub(super) fn set_bytes(enc: &Enc, v: &[u32], i: usize) {
    // SAFETY: Metal copies the bytes during the call.
    unsafe {
        enc.setBytes_length_atIndex(
            std::ptr::NonNull::new(v.as_ptr() as *mut std::ffi::c_void).expect("non-null"),
            std::mem::size_of_val(v),
            i,
        )
    };
}

pub(super) fn dispatch(enc: &Enc, groups: [usize; 3], threads: usize) {
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: groups[0],
            height: groups[1],
            depth: groups[2],
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
}

/// `ARF_VISION_SUM` on the GPU path: the same tags and format as the CPU encoder's `vsum`, so
/// the two logs diff line by line.
fn vsum_gpu(tag: &str, b: &Buf, len: usize) {
    let v = read_f32(b, len);
    let s: f64 = v.iter().map(|&x| x as f64).sum();
    eprintln!("[qwen-vsum] {tag:24} sum={s:.6} len={len} (GPU)");
}
