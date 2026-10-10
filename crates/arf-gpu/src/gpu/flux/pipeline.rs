//! `FluxPipeline` — the P3 façade that composes the four parity-proven FLUX modules
//! (CLIP-L, T5-XXL, MMDiT, VAE) with the flow-match Euler scheduler into prompt → image. Thin by
//! design: it owns the four modules + the host schedule and orchestrates the denoise loop; ALL
//! compute stays inside the modules' GPU kernels (the CPU only packs/unpacks latents, seeds the
//! initial noise, and runs the Euler arithmetic).
//!
//! Tokenization lives in the CALLER (the serve layer has arf-core's tokenizers): the pipeline
//! takes CLIP + T5 token ids, so arf-gpu stays free of tokenizer deps and mirrors the LLM
//! path (serve tokenizes, engine takes tokens).

use std::sync::Arc;

use arf_core::Result;

use crate::gpu::flux::{
    pack, ClipConfig, DitConfig, FlowMatchSchedule, GpuClipText, GpuDiT, GpuT5, GpuVae, T5Config,
    VaeConfig,
};
use crate::gpu::GpuContext;

/// A captured denoise trajectory step (for the HUD timeline): the decoded-preview-free latent
/// summary + the wall-clock of the DiT forward. The full latent is optionally retained so the UI
/// can VAE-decode intermediate previews.
#[derive(Debug, Clone)]
pub struct StepTrace {
    pub step: usize,
    pub sigma: f32,
    pub latent_l2: f32, // ‖latent‖₂ after this step (cheap evolution signal for the scrubber)
    pub dit_ms: f32,    // DiT forward wall-clock for this step
}

/// The full generation result: the final RGB `[3, H, W]` (channel-major, ~[-1,1]) + the per-step
/// trace the UI renders as the denoise timeline.
pub struct GenResult {
    pub rgb: Vec<f32>,
    pub height: usize,
    pub width: usize,
    pub trace: Vec<StepTrace>,
}

/// Resolution + step config for one generation.
#[derive(Debug, Clone, Copy)]
pub struct GenConfig {
    pub height: usize, // image px (multiple of 16); latent = h/8, packed grid = h/16
    pub width: usize,
    pub steps: usize, // schnell: 4
    pub seed: u64,
    pub t5_len: usize, // T5 sequence length the DiT is built for (256)
}

impl Default for GenConfig {
    fn default() -> Self {
        GenConfig {
            height: 1024,
            width: 1024,
            steps: 4,
            seed: 0,
            t5_len: 256,
        }
    }
}

pub struct FluxPipeline {
    ctx: Arc<GpuContext>,
    clip: GpuClipText,
    t5: GpuT5,
    dit: GpuDiT,
    vae: GpuVae,
}

impl FluxPipeline {
    /// Load all four modules from `models/flux-schnell/`. Paths are the standard filenames the
    /// README documents (clip_l.safetensors, t5xxl-Q8_0.gguf, flux1-schnell-Q4_K_S.gguf,
    /// ae.safetensors). The DiT is built per-call for the target resolution, so `load` takes only
    /// the encoders + VAE; the DiT path is held and (re)built in `generate`.
    pub fn load(ctx: &Arc<GpuContext>, dir: &std::path::Path, gen: &GenConfig) -> Result<Self> {
        let clip = GpuClipText::load(ctx, &dir.join("clip_l.safetensors"), ClipConfig::default())?;
        let t5 = GpuT5::load(ctx, &dir.join("t5xxl-Q8_0.gguf"), T5Config::default())?;
        let grid = gen.height / 16; // 2× patch on the /8 latent
        let dit = GpuDiT::load(
            ctx,
            &dir.join("flux1-schnell-Q4_K_S.gguf"),
            DitConfig::schnell(grid, gen.t5_len),
        )?;
        let vae = GpuVae::load(ctx, &dir.join("ae.safetensors"), VaeConfig::default())?;
        Ok(FluxPipeline {
            ctx: ctx.clone(),
            clip,
            t5,
            dit,
            vae,
        })
    }

    /// prompt-tokens → image. `clip_tokens` (≤77, the caller pads/truncates + sets `clip_eos`) and
    /// `t5_tokens` (padded to `gen.t5_len`). Returns the final RGB + denoise trace.
    pub fn generate(
        &self,
        clip_tokens: &[u32],
        clip_eos: usize,
        t5_tokens: &[u32],
        gen: &GenConfig,
    ) -> GenResult {
        let lc = gen.height / 8; // latent spatial (VAE 8× compression)
        let lw = gen.width / 8;
        let grid_h = lc / 2; // packed token grid
        let grid_w = lw / 2;
        let seq = grid_h * grid_w; // img tokens

        // text conditioning (parity-proven encoders)
        let clip_pooled = self.clip.encode(clip_tokens, clip_eos); // [768]
        let t5_seq = self.t5.encode(t5_tokens); // [t5_len, 4096]

        // initial latent noise [16, lc, lw] (deterministic from seed), then pack → [seq, 64]
        let latent0 = seeded_noise(16 * lc * lw, gen.seed);
        let mut packed = pack::pack_latents(&latent0, 16, lc, lw); // [seq, 64]

        let sched = FlowMatchSchedule::schnell(gen.steps, seq);
        let mut trace = Vec::with_capacity(gen.steps);
        for i in 0..gen.steps {
            let sigma = sched.sigmas[i];
            // DiT velocity at this step (timestep = sigma; the DiT scales ×1000 internally)
            let t0 = std::time::Instant::now();
            let vel = self.dit.forward(&packed, &t5_seq, &clip_pooled, sigma);
            let dit_ms = t0.elapsed().as_secs_f32() * 1000.0;
            sched.step(i, &mut packed, &vel);
            let l2 = (packed
                .iter()
                .map(|&x| (x as f64) * (x as f64))
                .sum::<f64>()
                .sqrt()) as f32;
            trace.push(StepTrace {
                step: i,
                sigma,
                latent_l2: l2,
                dit_ms,
            });
        }

        // unpack → channel-major latent, VAE decode → RGB
        let latent = pack::unpack_latents(&packed, 16, lc, lw);
        let rgb = self.vae.decode(&latent, lc, lw);
        let _ = &self.ctx; // ctx retained for future intermediate-preview decodes
        GenResult {
            rgb,
            height: gen.height,
            width: gen.width,
            trace,
        }
    }
}

/// Deterministic standard-normal noise of length `n` from `seed` — a splitmix64 → Box–Muller so
/// the initial latent is reproducible without pulling in an RNG crate (and seed-stable across
/// runs for the parity gate). Not cryptographic; just a fixed, portable noise source.
fn seeded_noise(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9E3779B97F4A7C15);
    let mut next = || {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z = z ^ (z >> 31);
        (z >> 11) as f64 / (1u64 << 53) as f64 // uniform [0,1)
    };
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = next().max(1e-12);
        let u2 = next();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        out.push((r * theta.cos()) as f32);
        if out.len() < n {
            out.push((r * theta.sin()) as f32);
        }
    }
    out
}
