//! # arf-gpu
//!
//! The **wgpu/Metal GPU backend** for the [`arf-core`](arf_core) engine,
//! built on from-scratch WGSL kernels (no ML framework, no vendor BLAS). It
//! provides two execution paths:
//!
//! - **Resident decode** ([`gpu::GpuModel`]): every weight, the RoPE tables, and
//!   the reused scratch live in resident GPU buffers; a whole token is recorded
//!   into one command buffer and submitted once. This is the fast path (e.g.
//!   Qwen3-Coder-30B Q4_K). Built via [`weights::load_gguf_gpu_quant`] and friends.
//! - **Eager `Linear`** ([`GpuContext`] as an [`EagerBackend`]):
//!   each projection's weight is uploaded once and run on demand, activations
//!   round-tripping the host. The simple per-`Linear` seam, selected by building a
//!   core model on [`Device::eager`](arf_core::device::Device::eager).
//!
//! The backend-agnostic transformer logic, GGUF/safetensors parsing, paged KV
//! cache, scheduler, and sampling live in `arf-core`; this crate depends on
//! it and never reimplements them.

// L364 — DEAD CODE OFF macOS. The native-Metal island is this crate's shipping fast path, and a
// large amount of code exists only to serve it: the speculation drafter, the depth-2 bank ring,
// the GDN recurrence helpers. Off macOS every one is genuinely unreachable, so a Linux build
// reports ~20 dead-code warnings for code that is correct and load-bearing where it runs.
//
// Per-item `#[cfg(target_os = "macos")]` was TRIED (2026-09-08) and abandoned: gating a type
// dead-ends its impl, then its struct field, then that field's initializer, then the helpers
// that touched it — eight rounds, still not closed, and every future island helper would join
// the queue. Churn that grows without bound to silence a lint telling the truth about the wrong
// platform. (A `[target.'cfg(..)'.lints]` table is not allowed in a virtual manifest, so this
// lives here, in the one crate that actually has an island.)
//
// On macOS these stay DENIED by CI — the island IS compiled there, so genuinely unused code is
// still caught on the platform where the check means something. See docs/PLATFORMS.md.
#![cfg_attr(
    not(target_os = "macos"),
    allow(dead_code, unused_imports, unused_variables)
)]

pub mod arena;
pub mod backend;
pub mod fast_path;
pub mod gpu;
/// Transcode cache + mmap zero-copy weight loading (macOS island path).
#[cfg(target_os = "macos")]
pub mod weight_cache;
pub mod weights;

use std::sync::Arc;

use arf_core::device::{Device, EagerBackend, EagerWeight};
use arf_core::error::Result;

pub use backend::WgpuBatched;
pub use fast_path::{apply_fast_path_defaults, decline_fast_path_for_model};

/// How many sequences a HYBRID (recurrent) model can decode concurrently: the recurrent bank's
/// row count minus row 0, which is reserved for id-less rows (prefill packing, MTP verify — L206).
/// A scheduler that admits more gets a refusal from the bank and a silent fall-back to a path with
/// different recurrent state, i.e. wrong text (measured 2026-09-19: 27B, 7 streams sane, 8
/// garbage). Callers clamp `--max-batch-size` to this for hybrid models.
pub fn recurrent_stream_capacity() -> usize {
    #[cfg(target_os = "macos")]
    {
        gpu::recurrent_bank_rows().saturating_sub(1).max(1)
    }
    #[cfg(not(target_os = "macos"))]
    {
        usize::MAX
    }
}
pub use gpu::{GpuContext, GpuModel, GpuWeight};

/// An eager GPU backend for core's `Linear` path: uploads a single weight and
/// hands back a resident [`GpuWeight`] that runs `x → x·Wᵀ`. A newtype over
/// `Arc<GpuContext>` because the orphan rule forbids `impl EagerBackend for
/// Arc<GpuContext>` (both foreign), and [`GpuWeight`] needs the `Arc` to dispatch.
pub struct WgpuEager(pub Arc<GpuContext>);

impl EagerBackend for WgpuEager {
    fn upload_weight(&self, data: &[f32], rows: usize, cols: usize) -> Arc<dyn EagerWeight> {
        Arc::new(GpuContext::upload_weight(&self.0, data, rows, cols))
    }

    fn info(&self) -> String {
        self.0.info().to_string()
    }
}

impl EagerWeight for GpuWeight {
    fn linear(&self, x: &[f32], m: usize) -> Vec<f32> {
        GpuWeight::linear(self, x, m)
    }

    fn out_dim(&self) -> usize {
        GpuWeight::out_dim(self)
    }
}

/// Acquire the best available GPU adapter as an eager [`Device`] for core's
/// `Linear` path, or error if none is present. The resident-decode path uses
/// [`GpuContext::new`] + the [`weights`] loaders directly instead.
pub fn eager_device() -> Result<Device> {
    let ctx = Arc::new(GpuContext::new()?);
    Ok(Device::eager(Arc::new(WgpuEager(ctx))))
}

/// 8-bit KV cache REQUESTED (2026-09-23): int8 K/V with one f32 scale per (token, kv head),
/// 4x the context of the f32 pool in the same memory. On by default; `ARF_NO_KV_Q8=1` opts out.
/// Whether a given model GETS it is [`kv_q8_for`] — ask that, not this. Read once.
fn kv_q8_requested() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var_os("ARF_NO_KV_Q8").is_none())
}

/// Does THIS model's KV pool go 8-bit? The single predicate the memory guard, `--max-context
/// auto` and the pool allocation all share — they must agree, because the guard and the context
/// rule price the pool before it exists. Pricing 8-bit and then allocating f32 is a 4x
/// under-charge on the largest allocation there is (17 GB at 131K on Qwen3.8).
///
/// Scoped to what was VERIFIED: the Metal island (the only place the int8 kernels exist) on a
/// hybrid recurrent/attention model (Qwen3.8), whose every attention dispatch goes through the
/// batched record that binds the 8-bit pool. Other archs still have paths that read the f32
/// pool directly (`run_megakernel`, the wgpu layer loop) — with the f32 pool reduced to a
/// placeholder those would attend garbage, so they keep f32 until each path is ported and checked.
pub fn kv_q8_for(cfg: &arf_core::config::ModelConfig) -> bool {
    cfg!(target_os = "macos")
        && kv_q8_requested()
        && std::env::var_os("ARF_MSL_GEMV").is_some()
        && weights::model_is_hybrid(cfg)
}

/// Is the 8-bit pool ALSO sparse — memory mapped as slots are written (`gpu/metal/sparse_kv.rs`)
/// instead of committed at load? `ARF_NO_SPARSE_KV=1` commits it all, as before 2026-09-23.
pub fn kv_sparse_for(cfg: &arf_core::config::ModelConfig) -> bool {
    kv_q8_for(cfg) && std::env::var_os("ARF_NO_SPARSE_KV").is_none() && sparse_kv_supported()
}

/// Grow-on-use KV (`gpu/metal/sparse_kv.rs`) is possible on this machine: Metal 4 placement-sparse
/// buffers. Asked before the model (and its device) exist, to size `--max-context auto`.
fn sparse_kv_supported() -> bool {
    #[cfg(target_os = "macos")]
    {
        static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *V.get_or_init(|| {
            objc2_metal::MTLCreateSystemDefaultDevice()
                .is_some_and(|d| gpu::metal::sparse_kv::supported(&d))
        })
    }
    #[cfg(not(target_os = "macos"))]
    false
}

/// What the Mac's power state does to the GPU (2026-10-06): `(Low Power Mode on, thermal state)`,
/// the thermal state as 0 nominal, 1 fair, 2 serious, 3 critical. Low Power Mode lowers the GPU's
/// clocks — a teammate's M5 Max, in Low Power Mode on battery, read a prompt with the GPU at
/// 35-41 % and decoded at 3-8 tok/s — so the server says so (`[power]`).
#[cfg(target_os = "macos")]
pub fn power_state() -> (bool, u8) {
    use objc2_foundation::{NSProcessInfo, NSProcessInfoThermalState};
    let p = NSProcessInfo::processInfo();
    let t = p.thermalState();
    let thermal = if t == NSProcessInfoThermalState::Critical {
        3
    } else if t == NSProcessInfoThermalState::Serious {
        2
    } else if t == NSProcessInfoThermalState::Fair {
        1
    } else {
        0
    };
    (p.isLowPowerModeEnabled(), thermal)
}

/// This process's physical footprint in bytes — what Activity Monitor calls its memory: the
/// anonymous and GPU memory it owns. The weights wrapped from the cache file are NOT in it (they
/// are file-backed), so the memory the model holds is this plus the weights. MEASURED 2026-10-07
/// on the 27B with its draft and vision tower: 6.6 GB after a first request, beside 14.6 GB of
/// weights; the device's `currentAllocatedSize` read 24.0-26.5 GB for the same state, moving with
/// the sparse KV reservation, so it is not the figure to budget on.
#[cfg(target_os = "macos")]
pub fn phys_footprint_bytes() -> Option<u64> {
    // `rusage_info_v0` (libproc): a 16-byte uuid, then u64 fields; `ri_phys_footprint` is the 8th.
    #[repr(C)]
    #[derive(Default)]
    struct RusageInfoV0 {
        uuid: [u8; 16],
        fields: [u64; 10],
    }
    extern "C" {
        fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut RusageInfoV0) -> i32;
    }
    let mut info = RusageInfoV0::default();
    // SAFETY: flavor 0 writes exactly `rusage_info_v0` (96 bytes), the layout above.
    let rc = unsafe { proc_pid_rusage(std::process::id() as i32, 0, &mut info) };
    (rc == 0).then_some(info.fields[7])
}

/// Off macOS: not read.
#[cfg(not(target_os = "macos"))]
pub fn phys_footprint_bytes() -> Option<u64> {
    None
}

/// Off macOS: no power state to report.
#[cfg(not(target_os = "macos"))]
pub fn power_state() -> (bool, u8) {
    (false, 0)
}
