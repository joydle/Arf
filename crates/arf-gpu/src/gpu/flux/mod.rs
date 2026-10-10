//! FLUX.1-schnell text→image, built the `vision.rs` way: small self-contained GPU modules
//! that own their `ComputeKernel`s and reuse Arf's existing kernel library. All compute is
//! WGSL on Metal; the CPU only orchestrates the diffusion loop, uploads GGUF→GPU, and
//! PNG-encodes the final readback. See `README.md` for the phase plan + module map.

pub mod clip; // CLIP-L pooled text encoder (P0)
pub mod dit; // MMDiT denoiser (P1)
pub mod pack; // latent pack/unpack for the DiT (P3, pure host)
pub mod pipeline;
pub mod scheduler; // flow-match Euler schedule (P3, pure host)
pub mod t5; // T5-XXL sequence text encoder (P0)
pub mod vae; // VAE conv decoder: latent → RGB (P2) // FluxPipeline façade: prompt → image (P3)

pub use clip::{ClipConfig, GpuClipText};
pub use dit::{DitConfig, GpuDiT};
pub use pipeline::{FluxPipeline, GenConfig, GenResult, StepTrace};
pub use scheduler::FlowMatchSchedule;
pub use t5::{GpuT5, T5Config};
pub use vae::{GpuVae, VaeConfig};

/// Dispatch grid for a flat, 256-wide kernel over `n` elements.
///
/// The `x` extent of a dispatch is capped at 65,535 workgroups. A 1024-px FLUX latent needs more
/// than that, so past the cap the grid spills into `y` and the kernel reconstructs its index as
/// `y * num_workgroups.x + x`. Every flat FLUX kernel is written to that convention.
///
/// This lived as four byte-identical copies in clip/dit/t5/vae, none of which explained the 65,535.
pub(super) fn wg(n: usize) -> [u32; 3] {
    let g = (n as u32).div_ceil(256);
    const MAX: u32 = 65535;
    if g <= MAX {
        [g, 1, 1]
    } else {
        [MAX, g.div_ceil(MAX), 1]
    }
}
