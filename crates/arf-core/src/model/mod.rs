//! The Llama transformer: ops, layers, the assembled model, and weight loading.

pub mod attention;
pub mod batch;
pub mod gguf;
pub mod gguf_tokenizer;
pub mod hf_config;
pub mod layer;
pub mod llama;
pub mod mlp;
pub mod moe;
pub mod mrope;
pub mod nn;
pub mod pcm;
/// Per-head QK-norm — re-exported from `arf-ml`.
pub use arf_ml::qk_norm;
/// RMSNorm — re-exported from `arf-ml`, where the pure numerics live.
pub use arf_ml::rms_norm;
pub mod qwen_audio;
pub mod qwen_audio_encoder;
pub mod qwen_video;
pub mod qwen_vision;
pub mod rope;
pub mod speculative;
pub mod tensor_names;
pub mod torch_ckpt;
pub mod vision;
pub mod weights;

pub use batch::{ForwardBatch, SeqAttn};
pub use llama::Llama;
pub use nn::{Embedding, Linear};
pub use rope::Rope;
pub use speculative::{generate_speculative, SpecResult, SpecStats};
pub use weights::Weights;
