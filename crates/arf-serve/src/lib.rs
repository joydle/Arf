//! Library surface of arf-serve. The whole server is here ([`server::run`]); `main.rs` only
//! passes it the Metal backend. Exposed so integration tests can drive the actor loop with a
//! mock backend, and so an embedding binary can inject its own backend: build this crate with `default-features = false` and call [`server::run`] with a
//! [`server::BackendFactory`] from its own binary.
pub mod actor;
pub mod anchor_replay; // prefix anchors recorded on disk and replayed after a restart
pub mod chat;
pub mod dashboard;
#[cfg(feature = "wgpu")]
pub mod flux; // FLUX text→image actor + /v1/images/generations handler
pub mod gpu_panic; // after a GPU driver panic naming this server: GPU memory committed at load
pub mod http;
pub mod jinja_chat; // the model's own chat template, for tool requests
pub mod metrics;
#[cfg(feature = "wgpu")]
pub mod model_loader;
#[cfg(feature = "postgres")]
pub mod persist;
pub mod prefix_anchor;
pub mod prefix_disk; // a cached prefix's state written to disk and loaded at start
pub mod qwen_audio; // Qwen3-Omni audio input: decode (WAV natively, ffmpeg for the rest), prompt layout, encode (only with --audio-tower)
pub mod qwen_image; // Qwen3.8 image input: prompt layout + CPU encode (only with a qwen mmproj) // where a chat request's shared system + tools prefix ends
pub mod qwen_video; // Qwen3.8 video input: ffmpeg frame decode (only with a qwen mmproj)
pub mod remote_media; // http(s) image/audio/video references, only with ARF_MEDIA_URLS=1
pub mod server; // the server entry point, with the backend injected
pub mod stop; // stop strings: OpenAI `stop`, Anthropic `stop_sequences`
pub mod tool_parse; // native tool-call parsers + the streaming reply splitter
