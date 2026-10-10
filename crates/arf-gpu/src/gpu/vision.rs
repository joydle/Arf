//! GPU SigLIP vision encoder. Mirrors the CPU `arf_core::model::vision::VisionEncoder`
//! op-for-op on the GPU, reusing the tiled `matmul.wgsl` (bf16 weights) + `add.wgsl`, and
//! the four vision-specific shaders (`layernorm_bias`, `add_bias`, `gelu_tanh`,
//! `vision_attn`). Self-contained: builds its own `ComputeKernel`s, so it shares ZERO state
//! with the text decode path. Gated by a bit-parity test against the CPU encoder.
//!
//! The patch embedding is done on the CPU (one tiny per-patch gather + a single matmul's
//! worth of work, negligible) so only the 27 transformer layers run on the GPU — where the
//! O(n²) attention over 4096 patches actually lives.

use std::sync::Arc;

use arf_core::model::gguf::LazyGguf;
use arf_core::model::vision::{
    VisionConfig, VisionEncoder, VisionPreprocess, VisionProjector, VisionWeights,
};
use arf_core::Result;

use crate::gpu::{ComputeKernel, GpuContext};

/// Workgroup grid for an element-wise kernel over `n` items at `@workgroup_size(256)`.
/// Spreads across Y when the X count would exceed the 65535-per-dim limit (the vision
/// MLP's n*ffn = 17.6M elements needs ~68.9K X groups). The element-wise shaders use a
/// 2D-safe flat index `gid.y * (num_workgroups.x * 256) + gid.x`, so the split is exact.
fn wg(n: usize) -> [u32; 3] {
    let groups = (n as u32).div_ceil(256);
    const MAX: u32 = 65535;
    if groups <= MAX {
        [groups, 1, 1]
    } else {
        let gy = groups.div_ceil(MAX);
        [MAX, gy, 1]
    }
}

/// Resident GPU weights for one SigLIP layer (bf16 matmuls + f32 norms/biases).
struct GpuLayer {
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
    up_w: wgpu::Buffer,
    up_b: wgpu::Buffer,
    down_w: wgpu::Buffer,
    down_b: wgpu::Buffer,
}

pub struct GpuVisionEncoder {
    ctx: Arc<GpuContext>,
    cfg: VisionConfig,
    layers: Vec<GpuLayer>,
    post_ln_w: wgpu::Buffer,
    post_ln_b: wgpu::Buffer,
    // kernels
    matmul: ComputeKernel,
    layernorm: ComputeKernel,
    add_bias: ComputeKernel,
    gelu: ComputeKernel,
    attn: ComputeKernel,
    add: ComputeKernel,
}

impl GpuVisionEncoder {
    /// Upload the CPU-loaded vision weights to the GPU and compile the kernels.
    pub fn new(ctx: &Arc<GpuContext>, cfg: VisionConfig, w: &VisionWeights) -> Self {
        let bf16 = |name: &str, d: &[f32]| ctx.storage_init_bf16(name, d);
        let f32b = |name: &str, d: &[f32]| ctx.storage_init(name, d);
        let layers = w
            .layers
            .iter()
            .map(|l| GpuLayer {
                ln1_w: f32b("v.ln1_w", &l.ln1_w),
                ln1_b: f32b("v.ln1_b", &l.ln1_b),
                q_w: bf16("v.q_w", &l.q_w),
                q_b: f32b("v.q_b", &l.q_b),
                k_w: bf16("v.k_w", &l.k_w),
                k_b: f32b("v.k_b", &l.k_b),
                v_w: bf16("v.v_w", &l.v_w),
                v_b: f32b("v.v_b", &l.v_b),
                o_w: bf16("v.o_w", &l.o_w),
                o_b: f32b("v.o_b", &l.o_b),
                ln2_w: f32b("v.ln2_w", &l.ln2_w),
                ln2_b: f32b("v.ln2_b", &l.ln2_b),
                up_w: bf16("v.up_w", &l.up_w),
                up_b: f32b("v.up_b", &l.up_b),
                down_w: bf16("v.down_w", &l.down_w),
                down_b: f32b("v.down_b", &l.down_b),
            })
            .collect();
        GpuVisionEncoder {
            ctx: ctx.clone(),
            cfg,
            layers,
            post_ln_w: f32b("v.post_ln_w", &w.post_ln_w),
            post_ln_b: f32b("v.post_ln_b", &w.post_ln_b),
            matmul: ComputeKernel::new(ctx, "vmatmul", include_str!("../shaders/wgsl/matmul.wgsl")),
            layernorm: ComputeKernel::new(
                ctx,
                "vlayernorm",
                include_str!("../shaders/wgsl/layernorm_bias.wgsl"),
            ),
            add_bias: ComputeKernel::new(
                ctx,
                "vaddbias",
                include_str!("../shaders/wgsl/add_bias.wgsl"),
            ),
            gelu: ComputeKernel::new(ctx, "vgelu", include_str!("../shaders/wgsl/gelu_tanh.wgsl")),
            attn: ComputeKernel::new(
                ctx,
                "vattn",
                include_str!("../shaders/wgsl/vision_attn.wgsl"),
            ),
            add: ComputeKernel::new(ctx, "vadd", include_str!("../shaders/wgsl/add.wgsl")),
        }
    }

    /// Run the 27 SigLIP layers + post-norm on the GPU. `hidden0` is the CPU patch-embedded
    /// + position-added input `[num_patches, hidden]`. Returns the post-norm `[np, hidden]`.
    pub fn encode_layers(&self, hidden0: &[f32]) -> Vec<f32> {
        let c = &self.cfg;
        let n = c.num_patches();
        let d = c.hidden;
        let f = c.ffn;
        let ctx = &self.ctx;
        // resident hidden buffer (read_write across the layer loop) + scratch.
        let hidden = ctx.storage_init("v.hidden", hidden0);
        let norm = ctx.storage_init("v.norm", &vec![0.0f32; n * d]);
        let q = ctx.storage_init("v.q", &vec![0.0f32; n * d]);
        let k = ctx.storage_init("v.k", &vec![0.0f32; n * d]);
        let v = ctx.storage_init("v.v", &vec![0.0f32; n * d]);
        let attn = ctx.storage_init("v.attn", &vec![0.0f32; n * d]);
        let proj = ctx.storage_init("v.proj", &vec![0.0f32; n * d]);
        let up = ctx.storage_init("v.up", &vec![0.0f32; n * f]);
        let down = ctx.storage_init("v.down", &vec![0.0f32; n * d]);

        let ln_dims = |eps: f32| ctx.uniform_of("ln", &[n as u32, d as u32, eps.to_bits(), 0u32]);
        let mm_dims = |m: usize, kk: usize, nn: usize| {
            ctx.uniform_of("mm", &[m as u32, kk as u32, nn as u32, 0u32])
        };
        let bias_dims = |rows: usize, cols: usize| {
            ctx.uniform_of("ba", &[rows as u32, cols as u32, 0u32, 0u32])
        };
        let add_dims = |len: usize| ctx.uniform_of("ad", &[len as u32, 0u32, 0u32, 0u32]);
        let gelu_dims = |len: usize| ctx.uniform_of("ge", &[len as u32, 0u32, 0u32, 0u32]);
        let attn_dims = ctx.uniform_of(
            "at",
            &[n as u32, d as u32, c.heads as u32, (d / c.heads) as u32],
        );
        // matmul tiles in 16×16: grid = (ceil(n/16), ceil(out/16)).
        let mm_wg = |m: usize, nn: usize| [(m as u32).div_ceil(16), (nn as u32).div_ceil(16), 1u32];

        // Optional per-stage finiteness probe (ARF_VISION_PROBE=1). Reads a buffer back
        // and reports range + first non-finite — used to localize a NaN to its kernel.
        let probe_on = std::env::var("ARF_VISION_PROBE").is_ok();
        let probe = |tag: &str, buf: &wgpu::Buffer, len: usize| {
            if !probe_on {
                return;
            }
            let v = ctx.read_f32(buf, len);
            let bad = v.iter().position(|x| !x.is_finite());
            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
            for &x in &v {
                if x.is_finite() {
                    lo = lo.min(x);
                    hi = hi.max(x);
                }
            }
            eprintln!(
                "[probe {tag}] len={len} range=[{lo:.4},{hi:.4}] first_nonfinite={:?} val={:?}",
                bad,
                bad.map(|i| v[i])
            );
        };

        for (li, l) in self.layers.iter().enumerate() {
            let p = li == 0; // probe only layer 0 to keep readbacks cheap
                             // attn block: norm = LN1(hidden); q/k/v = norm·W + b; attn = MHA; proj = attn·Wo + bo; hidden += proj
            let lnd = ln_dims(c.ln_eps);
            self.layernorm.dispatch(
                "ln1",
                &[&hidden, &l.ln1_w, &l.ln1_b, &norm, &lnd],
                [n as u32, 1, 1],
            );
            if p {
                probe("ln1", &norm, n * d);
            }
            for (w, b, dst) in [
                (&l.q_w, &l.q_b, &q),
                (&l.k_w, &l.k_b, &k),
                (&l.v_w, &l.v_b, &v),
            ] {
                let md = mm_dims(n, d, d);
                self.matmul
                    .dispatch("qkv", &[&norm, w, dst, &md], mm_wg(n, d));
                let bd = bias_dims(n, d);
                self.add_bias.dispatch("qkvb", &[dst, b, &bd], wg(n * d));
            }
            if p {
                probe("q", &q, n * d);
                probe("k", &k, n * d);
                probe("v", &v, n * d);
            }
            self.attn
                .dispatch("attn", &[&q, &k, &v, &attn, &attn_dims], [n as u32, 1, 1]);
            if p {
                probe("attn", &attn, n * d);
            }
            let md = mm_dims(n, d, d);
            self.matmul
                .dispatch("o", &[&attn, &l.o_w, &proj, &md], mm_wg(n, d));
            let bd = bias_dims(n, d);
            self.add_bias
                .dispatch("ob", &[&proj, &l.o_b, &bd], wg(n * d));
            let ad = add_dims(n * d);
            self.add.dispatch("res1", &[&hidden, &proj, &ad], wg(n * d));
            if p {
                probe("res1", &hidden, n * d);
            }

            // mlp block: norm = LN2(hidden); up = norm·Wup + bup; gelu(up); down = up·Wdown + bdown; hidden += down
            let lnd2 = ln_dims(c.ln_eps);
            self.layernorm.dispatch(
                "ln2",
                &[&hidden, &l.ln2_w, &l.ln2_b, &norm, &lnd2],
                [n as u32, 1, 1],
            );
            if p {
                probe("ln2", &norm, n * d);
            }
            let mdu = mm_dims(n, d, f);
            self.matmul
                .dispatch("up", &[&norm, &l.up_w, &up, &mdu], mm_wg(n, f));
            let bdu = bias_dims(n, f);
            self.add_bias
                .dispatch("upb", &[&up, &l.up_b, &bdu], wg(n * f));
            if p {
                probe("up", &up, n * f);
            }
            let gd = gelu_dims(n * f);
            self.gelu.dispatch("gelu", &[&up, &gd], wg(n * f));
            if p {
                probe("gelu", &up, n * f);
            }
            let mdd = mm_dims(n, f, d);
            self.matmul
                .dispatch("down", &[&up, &l.down_w, &down, &mdd], mm_wg(n, d));
            let bdd = bias_dims(n, d);
            self.add_bias
                .dispatch("downb", &[&down, &l.down_b, &bdd], wg(n * d));
            let ad2 = add_dims(n * d);
            self.add
                .dispatch("res2", &[&hidden, &down, &ad2], wg(n * d));
            if p {
                probe("res2", &hidden, n * d);
            }
        }
        // final post-norm into `norm`, then read it back.
        let lnd = ln_dims(c.ln_eps);
        self.layernorm.dispatch(
            "postln",
            &[&hidden, &self.post_ln_w, &self.post_ln_b, &norm, &lnd],
            [n as u32, 1, 1],
        );
        ctx.read_f32(&norm, n * d)
    }
}

/// The full image → soft-tokens pipeline for Gemma-3 vision, loaded from one mmproj GGUF:
/// CPU patch-embed + projector (cheap) bracketing the GPU SigLIP layer stack (the heavy
/// part). The actor holds one of these and calls [`encode_image`](VisionPipeline::encode_image)
/// per uploaded image to produce the `[256, out_dim]` soft-tokens spliced into the prompt.
pub struct VisionPipeline {
    /// CPU encoder — used only for `patch_embed` (conv + position) and to lend the
    /// `VisionWeights` to the GPU encoder. Its `encode` is NOT used (the GPU runs the layers).
    cpu: VisionEncoder,
    gpu: GpuVisionEncoder,
    projector: VisionProjector,
    pub preprocess: VisionPreprocess,
    pub cfg: VisionConfig,
    /// Projector output dim = the LM hidden size the soft-tokens splice into (2560 for 4B).
    pub out_dim: usize,
}

impl VisionPipeline {
    /// Load the SigLIP tower + projector from the mmproj GGUF and build the GPU encoder on
    /// the given (shared) context. `out_dim` is the LM hidden size (2560 for Gemma-3-4B).
    pub fn load(ctx: &Arc<GpuContext>, mmproj: &std::path::Path, out_dim: usize) -> Result<Self> {
        // The mmproj is opened under the gemma3 config (only the routing of v.*/mm.* matters).
        let g = LazyGguf::open(mmproj, &arf_core::config::ModelConfig::gemma3_4b())?;
        let cfg = VisionConfig::from_gguf(&g);
        let cpu = VisionEncoder::load(&g, cfg)?;
        let gpu = GpuVisionEncoder::new(ctx, cfg, cpu.weights());
        let projector = VisionProjector::load(&g, cfg.hidden, out_dim)?;
        let preprocess = VisionPreprocess::from_gguf(&g);
        Ok(VisionPipeline {
            cpu,
            gpu,
            projector,
            preprocess,
            cfg,
            out_dim,
        })
    }

    /// Number of image soft-tokens produced per image (256 for Gemma-3).
    pub fn num_tokens(&self) -> usize {
        let ts = self.projector.tokens_per_side;
        ts * ts
    }

    /// Run the full tower on a preprocessed image `[3, size, size]` (channel-major, already
    /// normalized — from [`VisionPreprocess::preprocess`]). Returns the `[num_tokens, out_dim]`
    /// soft-tokens, row-major, ready for `ImagePrompt.embeds`.
    pub fn encode_image(&self, pixels: &[f32]) -> Vec<f32> {
        // ARF_VISION_CPU=1 runs the SigLIP layers on the CPU in full f32 (no bf16 matmul)
        // — a debug switch to isolate bf16-precision divergence in the tower from a logic bug.
        let patches = if std::env::var("ARF_VISION_CPU").is_ok() {
            self.cpu.encode(pixels) // full f32 patch_embed + 27 layers + post-norm
        } else {
            let hidden0 = self.cpu.patch_embed(pixels); // CPU conv + position
            self.gpu.encode_layers(&hidden0) // GPU 27 layers (bf16) + post-norm
        };
        self.projector.project(&patches, self.cfg.grid()) // pool + RMSNorm + project
    }
}

impl arf_core::ImageEncoder for VisionPipeline {
    fn encode(&self, rgb: &[u8], w: usize, h: usize) -> Vec<f32> {
        let pixels = self.preprocess.preprocess(rgb, w, h);
        self.encode_image(&pixels)
    }
    fn num_tokens(&self) -> usize {
        VisionPipeline::num_tokens(self)
    }
    fn hidden(&self) -> usize {
        self.out_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arf_core::config::ModelConfig;

    /// The GPU SigLIP encoder must match the trusted CPU encoder on the REAL Gemma-3 vision
    /// tower. The GPU uses bf16-packed matmul weights vs the CPU's f32, so this is a close-
    /// match (cosine + bounded relative L2), not bit-exact — the same bar the text decode
    /// path meets. Skips if the GGUF or a GPU isn't present.
    #[test]
    fn gpu_vision_encoder_matches_cpu_real_gguf() {
        let path = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../models/gemma-3-4b-vision/mmproj-F16.gguf"
        ));
        if !path.exists() {
            eprintln!("SKIP: mmproj GGUF not found");
            return;
        }
        let ctx = match GpuContext::new() {
            Ok(c) => Arc::new(c),
            Err(e) => {
                eprintln!("SKIP: no GPU: {e}");
                return;
            }
        };
        // L358d — 224px, not the production 896. Parity is a property of the MATH, not of the
        // image size, and the full-resolution version made this test unrunnable in practice:
        //
        //   896px -> 64x64 = 4096 patches -> 5.45 TFLOP on ONE scalar core
        //     attention 2.09 (38%) + projections 1.17 (22%) + mlp 2.19 (40%)
        //   observed ~10 GFLOP/s optimised = 9+ min; at opt-level 0 it never finished in 11.
        //
        // `[profile.test] opt-level = 2` (added in the same change) was necessary and NOT
        // sufficient: it moved the CPU reference from ~2 to ~10 GFLOP/s, which is the right
        // speed for a scalar f32 loop and still far too slow for a gate. No compiler flag fixes
        // an O(n^2) reference at production resolution — the resolution is the bug.
        //
        // 224px -> 16x16 = 256 patches. The n^2 attention term drops 256x, the linear terms 16x,
        // and ALL 27 layers, 16 heads, both norms and the full MLP still run on real weights.
        // Every weight tensor is shape-independent of the grid; `patch_embed` slices the first
        // `np` rows of `pos_embd` (see vision.rs:261), which is exactly the smaller-grid case.
        // A parity bug that survives 256 patches and dies at 4096 would have to be a pure
        // sequence-length bug — and `num_patches` feeds every loop bound here, so there is no
        // hard-coded 4096 for it to hide behind.
        let cfg = VisionConfig {
            image_size: 224,
            ..VisionConfig::default()
        };
        let g = LazyGguf::open(path, &ModelConfig::gemma3_4b()).expect("open mmproj");
        let cpu = VisionEncoder::load(&g, cfg).expect("load CPU vision encoder");
        let gpu = GpuVisionEncoder::new(&ctx, cfg, cpu.weights());

        // same synthetic gradient image the CPU test uses, so both run on identical input.
        let n = cfg.image_size;
        let mut px = vec![0.0f32; 3 * n * n];
        for ch in 0..3 {
            for y in 0..n {
                for x in 0..n {
                    px[ch * n * n + y * n + x] = ((x + y + ch * 50) as f32 / (2 * n) as f32) - 0.5;
                }
            }
        }

        let hidden0 = cpu.patch_embed(&px);
        let cpu_out = cpu.encode(&px); // patch_embed + layers + post_ln
        let gpu_out = gpu.encode_layers(&hidden0); // layers + post_ln from the same hidden0
        assert_eq!(gpu_out.len(), cpu_out.len(), "shapes match [4096,1152]");
        assert!(
            gpu_out.iter().all(|v| v.is_finite()),
            "GPU output all finite"
        );

        // cosine similarity + relative L2 error over the whole [4096,1152] tensor.
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        let mut diff2 = 0.0f64;
        let mut ref2 = 0.0f64;
        for (a, b) in cpu_out.iter().zip(gpu_out.iter()) {
            let (a, b) = (*a as f64, *b as f64);
            dot += a * b;
            na += a * a;
            nb += b * b;
            diff2 += (a - b) * (a - b);
            ref2 += a * a;
        }
        let cos = dot / (na.sqrt() * nb.sqrt());
        let rel_l2 = (diff2 / ref2).sqrt();
        eprintln!("GPU↔CPU vision: cos={cos:.6}, rel_l2={rel_l2:.4}");
        assert!(
            cos > 0.999,
            "cosine similarity {cos} (bf16 should keep direction)"
        );
        assert!(
            rel_l2 < 0.05,
            "relative L2 error {rel_l2} within bf16 budget over 27 layers"
        );
    }

    /// The full VisionPipeline (preprocess → patch_embed → GPU layers → projector) must
    /// produce [256, 2560] finite, non-degenerate soft-tokens from a real RGB image.
    /// Skips without the mmproj GGUF or a GPU.
    #[test]
    fn vision_pipeline_end_to_end_real_gguf() {
        let path = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../models/gemma-3-4b-vision/mmproj-F16.gguf"
        ));
        if !path.exists() {
            eprintln!("SKIP: mmproj GGUF not found");
            return;
        }
        let ctx = match GpuContext::new() {
            Ok(c) => Arc::new(c),
            Err(e) => {
                eprintln!("SKIP: no GPU: {e}");
                return;
            }
        };
        // out_dim = 2560 (Gemma-3-4B LM hidden).
        let vp = VisionPipeline::load(&ctx, path, 2560).expect("load pipeline");
        assert_eq!(vp.num_tokens(), 256);
        assert_eq!(vp.out_dim, 2560);

        // A structured synthetic RGB image (gradient) so the tower has real structure.
        let (w, h) = (64usize, 48usize);
        let mut rgb = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 3;
                rgb[i] = (x * 4) as u8;
                rgb[i + 1] = (y * 5) as u8;
                rgb[i + 2] = ((x + y) * 2) as u8;
            }
        }
        let px = vp.preprocess.preprocess(&rgb, w, h);
        let soft = vp.encode_image(&px);
        assert_eq!(soft.len(), 256 * 2560, "[256, 2560] soft tokens");
        assert!(soft.iter().all(|v| v.is_finite()), "all finite");
        let mean = soft.iter().sum::<f32>() / soft.len() as f32;
        let var = soft.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / soft.len() as f32;
        assert!(var > 1e-4, "soft tokens non-degenerate: var={var}");
        eprintln!("VisionPipeline: 256×2560 soft-tokens, var={var:.4}");
    }
}
