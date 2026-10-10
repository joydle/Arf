//! Compute backend selection for the **eager** (CPU-mixed) execution path.
//!
//! `Cpu` runs the SIMD + threaded kernels. `Eager(backend)` keeps `Linear`
//! weights resident on a GPU backend and runs the projections there, one
//! projection at a time (activations round-trip the host). This is the simple
//! per-`Linear` seam — distinct from the fully-resident decode path, which
//! lives in `arf-gpu` behind its own `GpuBackend` trait.
//!
//! The dependency points *into* core: a GPU backend crate (e.g. `arf-gpu`)
//! implements [`EagerBackend`]/[`EagerWeight`]; core never names a concrete GPU
//! type. That inversion is what lets `arf-core` build with no GPU deps.

use std::sync::Arc;

/// An eager GPU backend: uploads a single `[out, in]` bf16 weight and hands back
/// a resident handle that can run `x → x·Wᵀ` on its own. Implemented by a GPU
/// backend crate; core only sees the trait objects.
pub trait EagerBackend: Send + Sync {
    /// Upload a row-major `[rows, cols]` (= `[out, in]`) f32 weight; return a
    /// resident handle. The host copy may be dropped by the caller afterwards.
    fn upload_weight(&self, data: &[f32], rows: usize, cols: usize) -> Arc<dyn EagerWeight>;
    /// Short human-readable description of the backend.
    fn info(&self) -> String;
}

/// A resident eager weight: runs `x [m, in] → [m, out]` on the backend.
pub trait EagerWeight: Send + Sync + std::fmt::Debug {
    /// `x` is `m` rows of length `in`; returns `m * out` flattened row-major.
    fn linear(&self, x: &[f32], m: usize) -> Vec<f32>;
    /// The output dimension (`out`).
    fn out_dim(&self) -> usize;
}

/// Where a model's heavy matmuls run, for the eager `Linear` path.
#[derive(Clone, Default)]
pub enum Device {
    /// CPU with SIMD/threaded matmul (the default).
    #[default]
    Cpu,
    /// An eager GPU backend (e.g. wgpu), holding resident `Linear` weights.
    Eager(Arc<dyn EagerBackend>),
}

impl Device {
    pub fn cpu() -> Self {
        Device::Cpu
    }

    /// Wrap an eager GPU backend.
    pub fn eager(backend: Arc<dyn EagerBackend>) -> Self {
        Device::Eager(backend)
    }

    /// Short human-readable description of the backend.
    pub fn describe(&self) -> String {
        match self {
            Device::Cpu => "cpu (simd + threads)".to_string(),
            Device::Eager(b) => format!("gpu: {}", b.info()),
        }
    }
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Device({})", self.describe())
    }
}
