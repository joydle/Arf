//! A minimal, row-major, `f32` tensor — just enough linear algebra to run a
//! transformer on the CPU, in plain safe Rust.
//!
//! Design choices that keep it small and honest:
//! - **`f32` only.** Inference here is CPU-side; weights are up-converted to f32
//!   at load time.
//! - **Always contiguous, row-major.** No lazy strides or views; every op
//!   produces a fresh owned tensor. This trades some copies for code you can read.
//! - **Shape mismatches panic.** They are bugs in the model wiring, not runtime
//!   conditions to recover from (the same stance as `ndarray`).
//!
//! The matmul kernels are cache-aware and row-parallel via [`std::thread::scope`]
//! — no external dependencies. They are not a tuned BLAS; they are clear.
//!
//! # What lives here, and what does not
//!
//! This crate is the numeric floor: an `f32` tensor, the block-quantized weight formats the
//! loaders produce (Q4_0, Q4_K, Q4_K_S, Q3_K, Q8, bf16), TurboQuant KV compression, and CPU
//! matmul. It knows nothing about transformers, GGUF files, schedulers or GPUs.
//!
//! It is split out of `arf-core` because it had no dependency on the rest of that crate — the
//! separation already existed in fact, and this makes it checkable by the compiler. `arf-core`
//! holds the model logic above it; `arf-gpu` holds the GPU backend.
//!
//! The CPU kernels here are also the **correctness oracle**: every GPU kernel is verified against
//! them. That is why they favour code you can read over code that is fast.

pub mod dtype;
pub mod matmul;
pub mod qk_norm;
pub mod rms_norm;
pub mod turboquant;
pub mod weight;

pub use matmul::{matmul, matmul_nt, matmul_nt_q4ks, matmul_nt_q8_0};
pub use qk_norm::QkNorm;
pub use rms_norm::RmsNorm;
pub use turboquant::{Codebook, QjlProjection, Rotation, RotationKind, TqKvBlock};
pub use weight::{Bf16Matrix, Q3KMatrix, Q4KMatrix, Q4KSMatrix, Q4Matrix, Q8Matrix};

/// A dense, row-major `f32` tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Tensor {
    /// Wrap `data` with `shape`. Panics if they disagree.
    pub fn from_vec(data: Vec<f32>, shape: impl Into<Vec<usize>>) -> Self {
        let shape = shape.into();
        let n: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            n,
            "data length {} does not match shape {:?}",
            data.len(),
            shape
        );
        Tensor { data, shape }
    }

    /// A zero-filled tensor of the given shape.
    pub fn zeros(shape: impl Into<Vec<usize>>) -> Self {
        let shape = shape.into();
        let n = shape.iter().product();
        Tensor {
            data: vec![0.0; n],
            shape,
        }
    }

    /// A one-filled tensor of the given shape.
    pub fn ones(shape: impl Into<Vec<usize>>) -> Self {
        let shape = shape.into();
        let n = shape.iter().product();
        Tensor {
            data: vec![1.0; n],
            shape,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Mutable view of the backing data. Shape is unchanged; for in-place ops.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn into_vec(self) -> Vec<f32> {
        self.data
    }

    /// Interpret as a 2-D `(rows, cols)` shape. Panics if rank != 2.
    pub fn dims2(&self) -> (usize, usize) {
        assert_eq!(
            self.rank(),
            2,
            "expected rank-2 tensor, got {:?}",
            self.shape
        );
        (self.shape[0], self.shape[1])
    }

    /// Interpret as a 3-D shape. Panics if rank != 3.
    pub fn dims3(&self) -> (usize, usize, usize) {
        assert_eq!(
            self.rank(),
            3,
            "expected rank-3 tensor, got {:?}",
            self.shape
        );
        (self.shape[0], self.shape[1], self.shape[2])
    }

    /// Reinterpret with a new shape (same element count). Cheap: no copy.
    pub fn reshape(mut self, shape: impl Into<Vec<usize>>) -> Self {
        let shape = shape.into();
        let n: usize = shape.iter().product();
        assert_eq!(self.data.len(), n, "reshape changes element count");
        self.shape = shape;
        self
    }

    /// Row `i` of a 2-D tensor.
    pub fn row(&self, i: usize) -> &[f32] {
        let (_, cols) = self.dims2();
        &self.data[i * cols..(i + 1) * cols]
    }

    /// Elementwise add, same shape.
    pub fn add(&self, other: &Tensor) -> Tensor {
        self.zip_map(other, |a, b| a + b)
    }

    /// Elementwise multiply, same shape.
    pub fn mul(&self, other: &Tensor) -> Tensor {
        self.zip_map(other, |a, b| a * b)
    }

    /// Apply SiLU (`x * sigmoid(x)`) elementwise.
    pub fn silu(&self) -> Tensor {
        self.map(|x| x / (1.0 + (-x).exp()))
    }

    /// Map a function over every element.
    pub fn map(&self, f: impl Fn(f32) -> f32) -> Tensor {
        Tensor {
            data: self.data.iter().copied().map(f).collect(),
            shape: self.shape.clone(),
        }
    }

    fn zip_map(&self, other: &Tensor, f: impl Fn(f32, f32) -> f32) -> Tensor {
        assert_eq!(self.shape, other.shape, "elementwise op shape mismatch");
        Tensor {
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(&a, &b)| f(a, b))
                .collect(),
            shape: self.shape.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn elementwise_ops() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        assert_eq!(a.add(&b).as_slice(), &[4.0, 6.0]);
        assert_eq!(a.mul(&b).as_slice(), &[3.0, 8.0]);
    }
}
