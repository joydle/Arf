//! The native-Metal island and its platform stand-ins.
//!
//! L364b — grouped out of a flat `gpu/` that held 16 files, three of which (21.4k lines) are
//! touched by NO portable code. This is the contiguous move CLAUDE.md endorses: the files moved
//! whole, nothing was split, and every existing `concurrent_metal::` path still resolves via the
//! re-export in `gpu/mod.rs`. `island.rs` was `concurrent_metal.rs`, `selftest.rs` was
//! `concurrent_metal_selftest.rs`, `shim.rs` was `metal_shim.rs`.
//!
//! What did NOT move, and why: `types.rs`, `batch.rs`, `decode.rs` and `ssm_qwen35.rs` interleave
//! portable code with macOS branches (27–37 `cfg` gates each). Sorting them under `metal/` would
//! misdescribe them; separating them is the 33k-line split measured and rejected in
//! docs/PLATFORMS.md.

#[cfg(target_os = "macos")]
pub mod island;

/// The DFlash 2 block-diffusion draft model. Stage 1: the loader.
#[cfg(target_os = "macos")]
pub mod dflash2;
// These four use the island, objc2 and `wgpu::hal::api::Metal`. Ungated, they broke the Linux CI
// job (E0432) from the day they landed; every caller is already macOS-only.
#[cfg(target_os = "macos")]
pub mod dflash2_block;
#[cfg(target_os = "macos")]
pub mod dflash2_ctx;
#[cfg(target_os = "macos")]
pub mod dflash2_ref;
#[cfg(target_os = "macos")]
pub mod state_snapshots;

/// The Qwen3.8 vision encoder (mmproj `qwen3vl_merger`) on the GPU: plain Metal on the system
/// device, its own queue, its own kernels (`qwen_vision_msl.metal`).
#[cfg(target_os = "macos")]
pub mod qwen_vision;

/// The Qwen3-Omni audio encoder on the GPU: the vision encoder's kernels and helpers plus
/// `qwen_audio_msl.metal` (conv stem, flatten, windowed attention).
#[cfg(target_os = "macos")]
pub mod qwen_audio;

/// Grow-on-use KV memory: the 8-bit pool as placement-sparse buffers.
#[cfg(target_os = "macos")]
pub mod sparse_kv;

#[cfg(all(target_os = "macos", feature = "selftest"))]
mod selftest; // MetalIsland self-test/parity/probe methods (example-driven)
/// What `MetalIsland::g64_pfs_selftest` returns, and its output sentinels (examples/g64_pfs_probe.rs).
#[cfg(all(target_os = "macos", feature = "selftest"))]
pub use selftest::{G64PfsSelftest, G64PFS_SENTINEL_BF16, G64PFS_SENTINEL_F32};

#[cfg(not(target_os = "macos"))]
pub mod shim;
