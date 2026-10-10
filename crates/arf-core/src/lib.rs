//! # arf-core
//!
//! The backend-agnostic core of the **arf** LLM inference engine, in pure,
//! safe Rust with no GPU dependency. It implements the transformer architectures
//! (RMSNorm, RoPE, grouped-query attention, SwiGLU / MoE MLP), GGUF + safetensors
//! loading, a paged KV cache, a continuous-batching scheduler, and sampling — all
//! on top of a tiny [`tensor`] core (`f32`, SIMD + threaded matmul on CPU).
//!
//! GPU backends live in separate crates (`arf-gpu` here; another backend
//! the same way) and depend on this crate. The seam is [`device::EagerBackend`] for the
//! eager per-`Linear` path; the fully-resident decode path is defined by each
//! backend crate over its own command-recorder trait.
//!
//! ## Layout
//! - [`tensor`] — the `f32` tensor core and SIMD matmul kernels.
//! - [`device`] — [`device::Device`] backend selection + the eager-GPU seam.
//! - [`config`] — model and engine configuration.
//! - [`bundle`] — a model directory with its draft and projector, resolved into launch flags.
//! - [`model`] — the transformer, its ops, and weight loading.
//! - [`cache`] — the paged KV cache and its block allocator.
//! - [`sampling`] — token sampling strategies.
//! - [`scheduler`] — request lifecycle and continuous batching.
//! - [`engine`] — the [`engine::LlmEngine`] that ties it together.
//! - [`tokenizer`] — a thin `tokenizers` wrapper.

pub mod backend;
pub mod bundle;
pub mod cache;
pub mod chat_template;
pub mod config;
pub mod device;
pub mod engine;
pub mod env_registry;
pub mod error;
pub mod model;
pub mod sampling;
pub mod scheduler;
/// The numeric floor — re-exported from the `arf-ml` crate.
///
/// This was a module here until it was split out; nothing about the API changed, so
/// `arf_core::tensor::…` still resolves and every existing caller keeps working.
pub use arf_ml as tensor;
pub mod tokenizer;

pub use backend::{BatchedBackend, ImageEncoder};
pub use config::{EngineConfig, ModelConfig};
pub use device::Device;
pub use engine::LlmEngine;
pub use error::{ArfError, Result};
pub use sampling::SamplingParams;
pub use tensor::Tensor;
pub use tokenizer::Tokenizer;
