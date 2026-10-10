//! Error type for the engine.
//!
//! Tensor algebra treats shape mismatches as programming bugs and panics with a
//! clear message (like `ndarray`); recoverable failures — IO, weight loading,
//! tokenization, capacity — surface as [`ArfError`].

use thiserror::Error;

/// Errors produced anywhere in arf.
#[derive(Debug, Error)]
pub enum ArfError {
    /// Filesystem / IO failure (loading weights, tokenizer, configs).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// The safetensors file could not be parsed.
    #[error("safetensors error: {0}")]
    SafeTensors(String),

    /// The tokenizer could not be loaded or applied.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    /// A model config or weight tensor was missing or had the wrong shape/dtype.
    #[error("model load error: {0}")]
    ModelLoad(String),

    /// The KV cache / block pool ran out of capacity.
    #[error("out of kv-cache blocks: requested {requested}, free {free}")]
    OutOfBlocks { requested: usize, free: usize },

    /// Invalid sampling parameters (e.g. temperature < 0, top_p out of range).
    #[error("invalid sampling params: {0}")]
    InvalidSampling(String),

    /// A request's prompt was empty.
    #[error("empty prompt")]
    EmptyPrompt,

    /// A prompt token id is outside the model vocabulary.
    #[error("token id {token} out of range for vocab size {vocab}")]
    TokenOutOfRange { token: u32, vocab: usize },

    /// Any other invariant violation, with a human-readable explanation.
    #[error("{0}")]
    Other(String),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, ArfError>;

impl ArfError {
    /// Build an [`ArfError::Other`] from anything stringy.
    pub fn other(msg: impl Into<String>) -> Self {
        ArfError::Other(msg.into())
    }

    /// Build a [`ArfError::ModelLoad`].
    pub fn model_load(msg: impl Into<String>) -> Self {
        ArfError::ModelLoad(msg.into())
    }
}

impl From<safetensors::SafeTensorError> for ArfError {
    fn from(e: safetensors::SafeTensorError) -> Self {
        ArfError::SafeTensors(e.to_string())
    }
}
