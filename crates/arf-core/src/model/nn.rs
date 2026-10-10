//! The two parameterized building blocks: a bias-free `Linear` and a token
//! `Embedding`. Both hold a bf16 weight matrix (half the memory of f32),
//! shareable via `Arc` so tied embedding/LM-head weights are stored once.

use std::sync::Arc;

use crate::device::{Device, EagerWeight};
use crate::tensor::{Bf16Matrix, Q3KMatrix, Q4KMatrix, Q4KSMatrix, Q4Matrix, Q8Matrix, Tensor};

/// A CPU-resident weight, in one of the supported formats.
#[derive(Debug, Clone)]
enum CpuWeight {
    Bf16(Arc<Bf16Matrix>),
    Q8(Arc<Q8Matrix>),
    Q4(Arc<Q4Matrix>),
    Q4K(Arc<Q4KMatrix>),
    Q4KS(Arc<Q4KSMatrix>),
    Q3K(Arc<Q3KMatrix>),
}

impl CpuWeight {
    fn forward(&self, x: &Tensor) -> Tensor {
        match self {
            CpuWeight::Bf16(w) => w.linear(x),
            CpuWeight::Q8(w) => w.linear(x),
            CpuWeight::Q4(w) => w.linear(x),
            CpuWeight::Q4K(w) => w.linear(x),
            CpuWeight::Q4KS(w) => w.linear(x),
            CpuWeight::Q3K(w) => w.linear(x),
        }
    }
}

/// A bias-free linear layer; the weight is `[out, in]` (HuggingFace layout).
///
/// On a GPU device the weight is uploaded and the host copy dropped; on CPU it
/// stays resident as bf16 (widened to f32 in the matmul) or per-row int8.
#[derive(Debug, Clone)]
pub struct Linear {
    cpu: Option<CpuWeight>,
    /// An eager GPU-resident weight (e.g. wgpu), set when built on a GPU device.
    /// `None` for the CPU path. The fully-resident decode path does not use this.
    gpu: Option<Arc<dyn EagerWeight>>,
}

impl Linear {
    /// A CPU linear layer over a (possibly shared) bf16 weight.
    pub fn from_matrix(weight: Arc<Bf16Matrix>) -> Self {
        Linear {
            cpu: Some(CpuWeight::Bf16(weight)),
            gpu: None,
        }
    }

    /// A CPU linear layer over a per-row int8 weight.
    pub fn from_q8(weight: Arc<Q8Matrix>) -> Self {
        Linear {
            cpu: Some(CpuWeight::Q8(weight)),
            gpu: None,
        }
    }

    /// A CPU linear layer over a Q4_0 block-32 weight (the GPU-Q4 parity oracle).
    pub fn from_q4(weight: Arc<Q4Matrix>) -> Self {
        Linear {
            cpu: Some(CpuWeight::Q4(weight)),
            gpu: None,
        }
    }

    /// A CPU linear layer over a Q4_K-lite weight (the GPU-Q4_K parity oracle).
    pub fn from_q4k(weight: Arc<Q4KMatrix>) -> Self {
        Linear {
            cpu: Some(CpuWeight::Q4K(weight)),
            gpu: None,
        }
    }

    /// A CPU linear layer over a Q4_K_S weight (the GPU-Q4_K_S parity oracle).
    pub fn from_q4ks(weight: Arc<Q4KSMatrix>) -> Self {
        Linear {
            cpu: Some(CpuWeight::Q4KS(weight)),
            gpu: None,
        }
    }

    /// A CPU linear layer over a Q3_K weight (the GPU-Q3_K parity oracle).
    pub fn from_q3k(weight: Arc<Q3KMatrix>) -> Self {
        Linear {
            cpu: Some(CpuWeight::Q3K(weight)),
            gpu: None,
        }
    }

    /// Build for `device`. On GPU the weight is uploaded and the host copy freed.
    pub fn on_device(weight: Arc<Bf16Matrix>, device: &Device) -> Self {
        if let Device::Eager(backend) = device {
            let f32 = weight.to_f32();
            let gpu = backend.upload_weight(&f32, weight.rows(), weight.cols());
            // `weight` (and the temporary f32) are dropped here, freeing host RAM.
            return Linear {
                cpu: None,
                gpu: Some(gpu),
            };
        }
        Self::from_matrix(weight)
    }

    /// `x [tokens, in]` → `[tokens, out]`.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        if let Some(g) = &self.gpu {
            let (m, _) = x.dims2();
            return Tensor::from_vec(g.linear(x.as_slice(), m), vec![m, g.out_dim()]);
        }
        self.cpu
            .as_ref()
            .expect("cpu weight present when gpu is absent")
            .forward(x)
    }
}

/// A token embedding table; the matrix is `[vocab, hidden]`, shared by `Arc` so
/// a tied LM head can reuse it without a second copy.
#[derive(Debug, Clone)]
pub struct Embedding {
    matrix: Arc<Bf16Matrix>,
}

impl Embedding {
    pub fn new(matrix: Arc<Bf16Matrix>) -> Self {
        Embedding { matrix }
    }

    /// Gather rows for `ids`, widened to f32: `[ids.len(), hidden]`.
    pub fn forward(&self, ids: &[u32]) -> Tensor {
        self.matrix.gather(ids)
    }
}
