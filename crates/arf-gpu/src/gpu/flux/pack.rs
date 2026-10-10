//! FLUX latent packing (host-side, pure — no GPU). The DiT operates on a sequence of patch
//! tokens, not the raw conv latent: a `[C=16, H, W]` latent is folded into `[(H/2)·(W/2), C·4]`
//! by grouping each 2×2 spatial block's channels. The VAE, by contrast, wants the channel-major
//! `[C, H, W]` latent — so the pipeline packs before the DiT and unpacks after.
//!
//! Mirrors diffusers `FluxPipeline._pack_latents` / `_unpack_latents` exactly:
//!   pack:   view(C, H/2, 2, W/2, 2) → permute(H/2, W/2, C, 2, 2) → reshape((H/2)·(W/2), C·4)
//!   unpack: the inverse.
//! Token order is row-major over (H/2, W/2); within a token the 64 values are (c, dy, dx) with
//! dx fastest — matching the DiT's `img_in` expectation and the RoPE img position (row, col).

/// Pack a channel-major latent `[c, h, w]` (row-major per channel) into DiT patch tokens
/// `[(h/2)·(w/2), c·4]`. `h` and `w` must be even.
pub fn pack_latents(latent: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let (hh, ww) = (h / 2, w / 2);
    let tok_dim = c * 4;
    let mut out = vec![0.0f32; hh * ww * tok_dim];
    for by in 0..hh {
        for bx in 0..ww {
            let tok = by * ww + bx; // row-major token index
            for ci in 0..c {
                for dy in 0..2 {
                    for dx in 0..2 {
                        let y = by * 2 + dy;
                        let x = bx * 2 + dx;
                        let src = ci * h * w + y * w + x;
                        // within-token layout: (c, dy, dx), dx fastest
                        let dst = tok * tok_dim + ci * 4 + dy * 2 + dx;
                        out[dst] = latent[src];
                    }
                }
            }
        }
    }
    out
}

/// Inverse of [`pack_latents`]: patch tokens `[(h/2)·(w/2), c·4]` → channel-major `[c, h, w]`.
pub fn unpack_latents(packed: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let (hh, ww) = (h / 2, w / 2);
    let tok_dim = c * 4;
    let mut out = vec![0.0f32; c * h * w];
    for by in 0..hh {
        for bx in 0..ww {
            let tok = by * ww + bx;
            for ci in 0..c {
                for dy in 0..2 {
                    for dx in 0..2 {
                        let y = by * 2 + dy;
                        let x = bx * 2 + dx;
                        let dst = ci * h * w + y * w + x;
                        let src = tok * tok_dim + ci * 4 + dy * 2 + dx;
                        out[dst] = packed[src];
                    }
                }
            }
        }
    }
    out
}
