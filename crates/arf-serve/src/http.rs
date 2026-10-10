//! The OpenAI-compatible HTTP layer: axum router, request/response types, and
//! the four handlers (plus the Anthropic-format `/v1/messages` pair in `http/anthropic.rs`).
//! Handlers are async tokio tasks; they NEVER touch the
//! `GpuModel`. They build a [`Job`], hand it to the model actor over a channel,
//! and stream the resulting tokens back (decoding ids → text with the resident
//! tokenizer, which is `Send + Sync` and lives in the shared state).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::http::header;

use arf_core::sampling::SamplingParams;
use arf_core::Tokenizer;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use crate::actor::{FinishReason, Job, ModelHandle, StreamItem, TokenLogprob};
use crate::chat::encode_prompt;
use crate::jinja_chat::ToolFormat;
use crate::metrics::Metrics;
use crate::stop::{text_of, StopFilter, StopStrings};
use crate::tool_parse::{settle_native, EndNote, Piece, ReplySplitter};

/// The Anthropic Messages wire format (`/v1/messages`) — a translation onto the same prompt
/// path, sampler and actor as the OpenAI handlers below.
mod anthropic;

/// The OpenAI Responses API (`/v1/responses`) — an adapter over `/v1/chat/completions`.
mod responses;

/// System One (`POST /v1/systemone`): typed decisions about a state from a decision model's
/// label-token logits — scored through the same actor, one score-only job per question.
pub mod systemone;

/// Shared, `Send + Sync` server state handed to every handler. The `GpuModel` is
/// NOT here — it is owned exclusively by the actor thread, reached only through
/// `model` (a channel sender). The `Tokenizer` is `Send + Sync` so it can be
/// shared directly for encode/decode.
#[derive(Clone)]
pub struct AppState {
    /// `Some(why)` when the model serves plain greedy steps only (`BatchedBackend::greedy_only`):
    /// a sampled request is served greedily, and every generation response says so in the
    /// `x-arf-warning` header, so a client learns its settings were not applied.
    pub greedy_only: Option<&'static str>,
    pub model: ModelHandle,
    pub tokenizer: Arc<Tokenizer>,
    pub model_id: String,
    /// Stop token ids resolved once from the tokenizer at startup.
    pub stop_tokens: Arc<Vec<u32>>,
    /// The `--max-context` budget: prompt + generation must fit in it. A prompt that fills it is
    /// refused (`context_length_exceeded`), and `max_tokens` is clamped to what is left.
    pub max_tokens_cap: usize,
    /// The model's own chat template, compiled at startup — used for requests with tools or tool
    /// history when it teaches a tool format Arf parses (`jinja_chat`). `None`: no template,
    /// it did not compile, or `ARF_NO_NATIVE_TOOLS=1`.
    pub chat_template: Option<Arc<crate::jinja_chat::ChatTemplate>>,
    /// Monotonic request-id source (Job.id / scheduler sequence id).
    pub next_id: Arc<AtomicU64>,
    /// Live serving counters published by the actor each step; read-only here.
    /// The actor is the sole writer (relaxed atomics); the /metrics handler is a
    /// pure reader. No lock, no scheduler access from tokio threads.
    pub metrics: Arc<Metrics>,
    /// True when a vision encoder is loaded (the actor can splice image soft-tokens).
    /// When false, image content parts are ignored (text-only).
    pub vision_enabled: bool,
    /// Image soft-tokens per image (256 for Gemma-3) — the placeholder run length.
    pub vision_tokens: usize,
    /// Qwen3.8 vision (`--mmproj` with a `qwen3vl_merger` projector): the GPU encoder, or the
    /// CPU one (`qwen_image::QwenVision`). Images are laid out and encoded in the HTTP layer
    /// (`qwen_image`), the actor only splices them. `None` = no Qwen vision; image parts then
    /// follow the Gemma-3 rules above.
    pub qwen_vision: Option<Arc<crate::qwen_image::QwenVision>>,
    /// Qwen3-Omni audio (`--audio-tower`): the audio encoder, GPU or CPU
    /// (`qwen_audio::AudioTower`). Audio parts are decoded, laid out and encoded in the HTTP
    /// layer (`qwen_audio`); the actor only splices the rows. `None` = a request carrying audio
    /// is refused with 400, never silently answered without it.
    pub qwen_audio: Option<Arc<crate::qwen_audio::AudioTower>>,
    /// Where each distinct system + tools prefix ends in tokens (`prefix_anchor`), remembered so
    /// an agent's 13K-token prefix is rendered and tokenized for it once, not every turn.
    pub prefix_anchors: Arc<crate::prefix_anchor::AnchorCache>,
    /// What `/v1/systemone` can serve with this model (`systemone::DecisionSupport`): decided at
    /// startup from the GGUF's `{arch}.decision.*` metadata. `None` (the default) = not a
    /// decision model, and the endpoint answers 501.
    pub systemone: Arc<systemone::DecisionSupport>,
    /// Optional Postgres pool for chat persistence (feature = "postgres"). `None`
    /// when `DATABASE_URL` is unset or unreachable — the `/conversations` routes
    /// then return `503`. The field only exists when the feature is enabled, so
    /// the zero-DB default build carries no sqlx state at all.
    #[cfg(feature = "postgres")]
    pub db: Option<sqlx::postgres::PgPool>,
}

/// Build the axum router with all routes wired to the shared state.
pub fn router(state: AppState) -> Router {
    // Permissive CORS so a browser-based client on another origin can poll /hud and stream
    // /v1/chat/completions. This is a LOCAL DEVELOPMENT DAEMON: it binds loopback by default and
    // has no authentication, so `Any` origin costs nothing it does not already give away. If you
    // ever expose this server beyond localhost, this layer and the missing auth are the first two
    // things to fix — do not treat the current defaults as production-safe.
    let cors = tower_http::cors::CorsLayer::new()
        .allow_origin(tower_http::cors::Any)
        .allow_methods(tower_http::cors::Any)
        .allow_headers(tower_http::cors::Any)
        // A browser page reads a response header only when it is exposed (`greedy_only_warning`).
        .expose_headers([axum::http::HeaderName::from_static("x-arf-warning")]);
    Router::new()
        .route("/health", get(health))
        .route("/healthz", get(healthz))
        .route("/metrics", get(metrics_handler))
        .route("/v1/arf/status", get(arf_status))
        .route("/hud", get(hud_handler))
        .route("/profile", get(profile_handler))
        .route("/v1/models", get(list_models))
        .route("/hf/search", get(hf_search))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        // The OpenAI Responses API, translated onto the chat path (see `http/responses.rs`).
        .route("/v1/responses", post(responses::responses))
        // The other dialect coding agents speak. Same engine path; see `http/anthropic.rs`.
        .route("/v1/messages", post(anthropic::messages))
        .route("/v1/messages/count_tokens", post(anthropic::count_tokens))
        .route("/v1/systemone", post(systemone::systemone))
        // FLUX text→image (lazy-loaded actor; see flux.rs). Always present so a client
        // can offer the image panel; the heavy pipeline loads on the first request.
        .route("/v1/images/generations", post(images_generations_handler))
        .route("/flux/trace", get(flux_trace_handler))
        // Opt-in chat persistence. These routes always exist (so a client can
        // probe them and fall back to localStorage); without the `postgres` feature
        // or a live pool they return 503 and the client uses its local store.
        .route(
            "/conversations",
            get(list_conversations_handler).post(create_conversation_handler),
        )
        .route(
            "/conversations/{id}",
            get(get_conversation_handler).delete(delete_conversation_handler),
        )
        .route("/conversations/{id}/messages", post(add_message_handler))
        .layer(cors)
        .layer(axum::middleware::from_fn_with_state(
            state.greedy_only,
            greedy_only_warning,
        ))
        .layer(axum::middleware::from_fn(background_request))
        .with_state(state)
}

// ----------------------------------------------------------------------------
// Chat-persistence handlers (feature = "postgres"). Without a live pool every
// route returns 503 so the browser client transparently falls back to
// localStorage. Request bodies are tiny JSON; responses mirror the persist types.
// ----------------------------------------------------------------------------

/// `503 Service Unavailable` with an OpenAI-shaped error body — the standard
/// "persistence disabled" reply when there is no Postgres pool.
fn persistence_unavailable() -> Response {
    error_response(
        StatusCode::SERVICE_UNAVAILABLE,
        "chat persistence disabled (no DATABASE_URL / postgres feature)",
    )
}

// Fields are only read in the `postgres` cfg arm; the request bodies are still
// deserialized in the default build (the routes exist so the client can probe them).
#[derive(Deserialize)]
#[cfg_attr(not(feature = "postgres"), allow(dead_code))]
struct CreateConversationBody {
    title: Option<String>,
    model: Option<String>,
}

#[derive(Deserialize)]
#[cfg_attr(not(feature = "postgres"), allow(dead_code))]
struct AddMessageBody {
    role: String,
    content: String,
}

/// `GET /conversations` → list conversations (most-recent first).
async fn list_conversations_handler(State(state): State<AppState>) -> Response {
    #[cfg(feature = "postgres")]
    {
        let Some(pool) = state.db.as_ref() else {
            return persistence_unavailable();
        };
        return match crate::persist::list_conversations(pool).await {
            Ok(rows) => Json(rows).into_response(),
            Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &format!("db: {e}")),
        };
    }
    #[cfg(not(feature = "postgres"))]
    {
        let _ = state;
        persistence_unavailable()
    }
}

/// `POST /conversations` → create a conversation.
async fn create_conversation_handler(
    State(state): State<AppState>,
    Json(body): Json<CreateConversationBody>,
) -> Response {
    #[cfg(feature = "postgres")]
    {
        let Some(pool) = state.db.as_ref() else {
            return persistence_unavailable();
        };
        return match crate::persist::create_conversation(
            pool,
            body.title.as_deref(),
            body.model.as_deref(),
        )
        .await
        {
            Ok(conv) => (StatusCode::CREATED, Json(conv)).into_response(),
            Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &format!("db: {e}")),
        };
    }
    #[cfg(not(feature = "postgres"))]
    {
        let _ = (state, body);
        persistence_unavailable()
    }
}

/// `GET /conversations/:id` → one conversation with its ordered messages.
async fn get_conversation_handler(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    #[cfg(feature = "postgres")]
    {
        let Some(pool) = state.db.as_ref() else {
            return persistence_unavailable();
        };
        return match crate::persist::get_conversation_with_messages(pool, &id).await {
            Ok(Some(detail)) => Json(detail).into_response(),
            Ok(None) => error_response(StatusCode::NOT_FOUND, "conversation not found"),
            Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &format!("db: {e}")),
        };
    }
    #[cfg(not(feature = "postgres"))]
    {
        let _ = (state, id);
        persistence_unavailable()
    }
}

/// `POST /conversations/:id/messages` → append a message to a conversation.
async fn add_message_handler(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
    Json(body): Json<AddMessageBody>,
) -> Response {
    #[cfg(feature = "postgres")]
    {
        let Some(pool) = state.db.as_ref() else {
            return persistence_unavailable();
        };
        return match crate::persist::add_message(pool, &id, &body.role, &body.content).await {
            Ok(Some(msg)) => (StatusCode::CREATED, Json(msg)).into_response(),
            Ok(None) => error_response(StatusCode::NOT_FOUND, "conversation not found"),
            Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &format!("db: {e}")),
        };
    }
    #[cfg(not(feature = "postgres"))]
    {
        let _ = (state, id, body);
        persistence_unavailable()
    }
}

/// `DELETE /conversations/:id` → delete a conversation (messages cascade).
async fn delete_conversation_handler(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    #[cfg(feature = "postgres")]
    {
        let Some(pool) = state.db.as_ref() else {
            return persistence_unavailable();
        };
        return match crate::persist::delete_conversation(pool, &id).await {
            Ok(true) => StatusCode::NO_CONTENT.into_response(),
            Ok(false) => error_response(StatusCode::NOT_FOUND, "conversation not found"),
            Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &format!("db: {e}")),
        };
    }
    #[cfg(not(feature = "postgres"))]
    {
        let _ = (state, id);
        persistence_unavailable()
    }
}

// ----------------------------------------------------------------------------
// Request / response types (OpenAI-compatible subset).
// ----------------------------------------------------------------------------

/// One chat message. `content` is the text (concatenated text parts); `images` holds any
/// `image_url` data-URLs from an OpenAI content-parts array. Both the plain-string form
/// (`"content": "hi"`) and the multimodal form (`"content": [{type,...}]`) deserialize here.
///
/// The tool fields were added 2026-09-26. Before that only `role` and `content` were read, so a
/// coding agent's history reached the model with every tool call ERASED (an assistant turn that
/// only called a tool rendered empty), and the redundant edit OpenCode made.
#[derive(Debug, Clone, Default)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    /// `image_url.url` values (typically `data:image/...;base64,...`) for vision input.
    /// Empty for text-only messages.
    pub images: Vec<String>,
    /// Byte offset in `content` where each of `images` appeared among the content parts, so the
    /// Qwen3.8 image path can put the image where the client did (`qwen_image::stage_messages`).
    /// Parallel to `images`; empty when a message is built without offsets (images go first).
    pub image_at: Vec<usize>,
    /// Qwen3.8 VIDEO parts (`video_url`, or `video` with a URL or a list of frames), 2026-09-27:
    /// `(i, video)` says `images[i]` is a video (that slot holds only a label, `VideoInput::label`),
    /// so a video keeps its place among the images and its `image_at` offset. Empty for every
    /// request without one.
    pub videos: Vec<(usize, crate::qwen_video::VideoInput)>,
    /// Qwen3-Omni AUDIO parts (2026-09-27, M3): OpenAI's `input_audio` (`{data: <base64>,
    /// format}`, kept as a `data:audio/<format>;base64,..` URL) or vLLM's `audio_url` (`{url}`).
    /// Decoded by `qwen_audio::load_audio`. Empty for every request without one.
    pub audios: Vec<String>,
    /// Byte offset in `content` where each of `audios` appeared among the content parts
    /// (parallel to `audios`), so the audio block lands where the client put it.
    pub audio_at: Vec<usize>,
    /// An assistant turn's `tool_calls`, `arguments` kept as the JSON string the client sent.
    pub tool_calls: Vec<ToolCall>,
    /// A `tool` message: the id of the call it answers.
    pub tool_call_id: Option<String>,
    /// A `tool` message's function name, when the client sends it.
    pub name: Option<String>,
    /// An assistant turn's reasoning, when the client replays it (`reasoning_content`).
    pub reasoning_content: Option<String>,
}

impl ChatMessage {
    /// Convenience for internal construction (system/tool prompts) — text only.
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        ChatMessage {
            role: role.into(),
            content: content.into(),
            ..Default::default()
        }
    }

    /// The video in slot `i` of `images`, if that slot is one.
    pub fn video_at(&self, i: usize) -> Option<&crate::qwen_video::VideoInput> {
        self.videos.iter().find(|(k, _)| *k == i).map(|(_, v)| v)
    }

    /// True for the turns that make a conversation a tool conversation: a `tool` result, or an
    /// assistant turn that called tools.
    pub fn is_tool_turn(&self) -> bool {
        self.role == "tool" || !self.tool_calls.is_empty()
    }
}

impl<'de> Deserialize<'de> for ChatMessage {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        // content: either a bare string, or an array of
        // {type: "text" | "image_url" | "video_url" | "video", ...}.
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Content {
            Text(String),
            Parts(Vec<ContentPart>),
        }
        #[derive(Deserialize)]
        #[serde(tag = "type", rename_all = "snake_case")]
        enum ContentPart {
            Text {
                text: String,
            },
            ImageUrl {
                image_url: ImageUrl,
            },
            /// vLLM's shape: `{"type": "video_url", "video_url": {"url", "fps"?, "max_frames"?}}`.
            VideoUrl {
                video_url: VideoUrl,
            },
            /// qwen-vl-utils / DashScope's shape: `{"type": "video", "video": "<url>" | [frames]}`.
            Video {
                video: VideoField,
                #[serde(default)]
                fps: Option<f64>,
                #[serde(default)]
                max_frames: Option<usize>,
            },
            /// OpenAI's shape: `{"type": "input_audio", "input_audio": {"data": <base64>,
            /// "format": "wav" | "mp3" | ..}}`. The bytes are sniffed; `format` is a hint.
            InputAudio {
                input_audio: InputAudio,
            },
            /// vLLM's shape: `{"type": "audio_url", "audio_url": {"url": <data: URL or path>}}`.
            AudioUrl {
                audio_url: ImageUrl,
            },
        }
        #[derive(Deserialize)]
        struct InputAudio {
            data: String,
            #[serde(default)]
            format: Option<String>,
        }
        #[derive(Deserialize)]
        struct ImageUrl {
            url: String,
        }
        #[derive(Deserialize)]
        struct VideoUrl {
            url: String,
            #[serde(default)]
            fps: Option<f64>,
            #[serde(default)]
            max_frames: Option<usize>,
        }
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum VideoField {
            Url(String),
            Frames(Vec<String>),
        }
        #[derive(Deserialize)]
        struct Raw {
            role: String,
            #[serde(default)]
            content: Option<Content>,
            // `null` is what clients send on a turn without calls; both mean "none".
            #[serde(default)]
            tool_calls: Option<Vec<ToolCall>>,
            #[serde(default)]
            tool_call_id: Option<String>,
            #[serde(default)]
            name: Option<String>,
            #[serde(default)]
            reasoning_content: Option<String>,
        }
        let raw = Raw::deserialize(d)?;
        let mut msg = ChatMessage {
            role: raw.role,
            tool_calls: raw.tool_calls.unwrap_or_default(),
            tool_call_id: raw.tool_call_id,
            name: raw.name,
            reasoning_content: raw.reasoning_content.filter(|r| !r.is_empty()),
            ..Default::default()
        };
        match raw.content {
            Some(Content::Text(s)) => msg.content = s,
            Some(Content::Parts(parts)) => {
                let mut text = String::new();
                for p in parts {
                    match p {
                        ContentPart::Text { text: t } => {
                            if !text.is_empty() {
                                text.push('\n');
                            }
                            text.push_str(&t);
                        }
                        ContentPart::ImageUrl { image_url } => {
                            msg.image_at.push(text.len());
                            msg.images.push(image_url.url);
                        }
                        ContentPart::VideoUrl { video_url } => {
                            let v = crate::qwen_video::VideoInput {
                                source: crate::qwen_video::VideoSource::Url(video_url.url),
                                fps: video_url.fps,
                                max_frames: video_url.max_frames,
                            };
                            msg.image_at.push(text.len());
                            msg.videos.push((msg.images.len(), v.clone()));
                            msg.images.push(v.label());
                        }
                        ContentPart::Video {
                            video,
                            fps,
                            max_frames,
                        } => {
                            let source = match video {
                                VideoField::Url(u) => crate::qwen_video::VideoSource::Url(u),
                                VideoField::Frames(f) => crate::qwen_video::VideoSource::Frames(f),
                            };
                            let v = crate::qwen_video::VideoInput {
                                source,
                                fps,
                                max_frames,
                            };
                            msg.image_at.push(text.len());
                            msg.videos.push((msg.images.len(), v.clone()));
                            msg.images.push(v.label());
                        }
                        ContentPart::InputAudio { input_audio } => {
                            let data = input_audio.data.trim();
                            let url = if data.starts_with("data:") {
                                data.to_string()
                            } else {
                                let fmt = input_audio.format.as_deref().unwrap_or("wav");
                                format!("data:audio/{fmt};base64,{data}")
                            };
                            msg.audio_at.push(text.len());
                            msg.audios.push(url);
                        }
                        ContentPart::AudioUrl { audio_url } => {
                            msg.audio_at.push(text.len());
                            msg.audios.push(audio_url.url);
                        }
                    }
                }
                msg.content = text;
            }
            None => {}
        }
        Ok(msg)
    }
}

/// OpenAI `response_format` object.
///
/// `type` is `"text"` (default), `"json_object"`, or `"json_schema"`.
/// For `"json_schema"`, `json_schema` carries the schema value (stored for v2;
/// v1 enforces structural JSON validity only — identical to `"json_object"`).
#[derive(Debug, Deserialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
    /// The schema value for `json_schema` mode (v2: key/type enforcement).
    pub json_schema: Option<serde_json::Value>,
}

/// OpenAI `stream_options`. Only `include_usage` exists.
#[derive(Debug, Default, Deserialize)]
pub struct StreamOptions {
    /// Send a final `{"choices": [], "usage": {…}}` chunk before `[DONE]`. Both OpenCode and Pi
    /// ask for it; without it their token meters read 0 and auto-compaction never fires.
    #[serde(default)]
    pub include_usage: bool,
}

/// `POST /v1/chat/completions` request body.
#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,
    /// OpenAI's newer name for `max_tokens`. Two fields rather than a serde alias: a client that
    /// sends BOTH would otherwise get a duplicate-field 400. When both are set, this one wins.
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    /// OpenAI `stream_options` (`include_usage`).
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// OpenAI `stop`: a string or a list of up to four. Enforced (`crate::stop`):
    /// the reply ends at the first one, which is not part of it, with `finish_reason` `stop`.
    /// Until then it was accepted and ignored.
    #[serde(default)]
    pub stop: Option<serde_json::Value>,
    /// `0` (or absent) = greedy; `> 0` = on-GPU stochastic sampling.
    pub temperature: Option<f32>,
    /// OpenAI nucleus sampling; `None`/`1.0` disables.
    pub top_p: Option<f32>,
    /// Non-standard extension; keep only the `k` highest-prob tokens.
    pub top_k: Option<usize>,
    /// Reproducible RNG seed (OpenAI `seed`). Unset = a fresh seed per request (OpenAI's
    /// behaviour); set it to get the same sample back.
    pub seed: Option<u64>,
    /// Repetition penalty (`1.0` = off; ~1.1–1.3 suppresses loops). Penalizes
    /// already-emitted tokens, on both greedy and stochastic decoding.
    /// `repeat_penalty` alias: the web UI / llama.cpp use that name — without the alias
    /// serde silently drops it, leaving the penalty at 1.0 (OFF) → runaway repetition loops.
    #[serde(alias = "repeat_penalty")]
    pub repetition_penalty: Option<f32>,
    /// Qwen3.8 reasoning effort: `xhigh` (default), `medium`, `low` — the model's own template.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub stream: bool,
    /// Return per-token log probabilities in the response (OpenAI `logprobs`).
    #[serde(default)]
    pub logprobs: bool,
    /// Number of top-alternative logprobs to return per token (0–20).
    /// Only honoured when `logprobs: true`.
    #[serde(default)]
    pub top_logprobs: Option<usize>,
    /// OpenAI structured output format.
    /// `"json_object"` / `"json_schema"` → grammar-constrained sampling (v1:
    /// structural JSON; v2 will add schema key/type enforcement).
    pub response_format: Option<ResponseFormat>,
    /// OpenAI function/tool calling: the functions the model may call. When present, the prompt
    /// is rendered by the model's OWN chat template when it teaches a tool format Arf parses
    /// (`jinja_chat`), else their schemas are injected with the fenced-JSON protocol
    /// (`chat::tool_system_prompt`, written for gemma-4). Either way the model's calls come back
    /// as `tool_calls` on the response (finish_reason `tool_calls`).
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    /// OpenAI `tool_choice`. `"none"` is honoured: the tools are neither shown to the model nor
    /// parsed out of its reply, so a call-shaped reply comes back as text (until 2026-09-26 the
    /// field was not declared, serde dropped it, and a `"none"` request could still come back
    /// with `tool_calls`). `"auto"` is the default. `"required"` and a forced function
    /// (`{"type": "function", "function": {"name": …}}`) were ACCEPTED AND NOT ENFORCED until
    /// 2026-10-04 — the model still chose. They are now enforced by PREFILL
    /// (`enforce_tool_choice`): the reply is begun with the opening of a call in the prompt's
    /// format, and with the function's name when one is forced. (A grammar over the template's
    /// call syntax would be the stronger guarantee; the sampler's JSON grammar does not cover it,
    /// and a constrained job leaves the fast decode step for its whole reply.)
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    /// vLLM / llama.cpp `chat_template_kwargs`: variables for the model's chat template —
    /// `enable_thinking`, `reasoning_effort`, `preserve_thinking` are read (others ignored).
    /// `enable_thinking: false` and `reasoning_effort` apply to plain chat too (the hand-written
    /// Qwen3.8 template reads the effort, `"none"` = non-thinking); `preserve_thinking` reaches
    /// only the model's own template (tool requests).
    #[serde(default)]
    pub chat_template_kwargs: Option<ChatTemplateKwargs>,
}

/// The `chat_template_kwargs` Arf reads (see [`ChatRequest::chat_template_kwargs`]).
#[derive(Debug, Default, Deserialize)]
pub struct ChatTemplateKwargs {
    #[serde(default)]
    pub enable_thinking: Option<bool>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub preserve_thinking: Option<bool>,
}

/// One OpenAI tool (only `type: "function"` is supported).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Tool {
    #[serde(default = "default_tool_type")]
    pub r#type: String,
    pub function: ToolFunction,
}
fn default_tool_type() -> String {
    "function".into()
}

/// An OpenAI function definition (JSON-schema parameters).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolFunction {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    /// JSON-schema for the arguments object (passed through verbatim).
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
}

/// A tool call the model decided to make (OpenAI shape) — and, deserialized, one an assistant
/// turn in the history made.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    #[serde(default)]
    pub id: String,
    #[serde(default = "default_tool_type")]
    pub r#type: String, // "function"
    pub function: ToolCallFunction,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    /// Arguments as a JSON **string** (OpenAI encodes the args object as a string). A client
    /// that sends an object instead is tolerated: it is kept as that object's JSON text.
    #[serde(default, deserialize_with = "arguments_as_string")]
    pub arguments: String,
}

/// `function.arguments` as the JSON string OpenAI specifies, accepting an object (or any other
/// JSON value) from a lenient client, and `null` as "no arguments".
fn arguments_as_string<'de, D: serde::Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    Ok(match serde_json::Value::deserialize(d)? {
        serde_json::Value::String(s) => s,
        serde_json::Value::Null => String::new(),
        other => other.to_string(),
    })
}

/// `POST /v1/completions` request body.
#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub prompt: String,
    pub max_tokens: Option<usize>,
    /// `0` (or absent) = greedy; `> 0` = on-GPU stochastic sampling.
    pub temperature: Option<f32>,
    /// OpenAI nucleus sampling; `None`/`1.0` disables.
    pub top_p: Option<f32>,
    /// Non-standard extension; keep only the `k` highest-prob tokens.
    pub top_k: Option<usize>,
    /// Reproducible RNG seed (OpenAI `seed`). Unset = a fresh seed per request (OpenAI's
    /// behaviour); set it to get the same sample back.
    pub seed: Option<u64>,
    /// Repetition penalty (`1.0` = off; ~1.1–1.3 suppresses loops). Penalizes
    /// already-emitted tokens, on both greedy and stochastic decoding.
    /// `repeat_penalty` alias: the web UI / llama.cpp use that name — without the alias
    /// serde silently drops it, leaving the penalty at 1.0 (OFF) → runaway repetition loops.
    #[serde(alias = "repeat_penalty")]
    pub repetition_penalty: Option<f32>,
    /// OpenAI `stop`, as on `ChatRequest`.
    #[serde(default)]
    pub stop: Option<serde_json::Value>,
    #[serde(default)]
    pub stream: bool,
    /// Return per-token log probabilities in the response (OpenAI `logprobs`).
    #[serde(default)]
    pub logprobs: bool,
    /// Number of top-alternative logprobs to return per token (0–20).
    /// Only honoured when `logprobs: true`.
    #[serde(default)]
    pub top_logprobs: Option<usize>,
    /// OpenAI structured output format (same semantics as `ChatRequest`).
    pub response_format: Option<ResponseFormat>,
}

// ----------------------------------------------------------------------------
// Logprobs response types (OpenAI-compatible).
// ----------------------------------------------------------------------------

/// One token's logprob entry in the response (OpenAI `logprobs.content[i]`).
#[derive(Serialize)]
struct TokenLogprobEntry {
    token: String,
    logprob: f32,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    top_logprobs: Vec<TopLogprobEntry>,
}

/// One alternative token in the `top_logprobs` list.
#[derive(Serialize)]
struct TopLogprobEntry {
    token: String,
    logprob: f32,
}

/// The `logprobs` object that appears in a non-streaming choice.
#[derive(Serialize)]
struct ChoiceLogprobs {
    content: Vec<TokenLogprobEntry>,
}

/// Build a `ChoiceLogprobs` from the accumulated per-token logprob data,
/// decoding each token id to its text representation using the tokenizer.
fn build_choice_logprobs(
    tok: &arf_core::Tokenizer,
    lp_data: &[Box<TokenLogprob>],
) -> ChoiceLogprobs {
    let content = lp_data
        .iter()
        .map(|lp| {
            let token_text = decode(tok, &[lp.token_id]);
            let top_logprobs = lp
                .top_logprobs
                .iter()
                .map(|(id, logprob)| TopLogprobEntry {
                    token: decode(tok, &[*id]),
                    logprob: *logprob,
                })
                .collect();
            TokenLogprobEntry {
                token: token_text,
                logprob: lp.logprob,
                top_logprobs,
            }
        })
        .collect();
    ChoiceLogprobs { content }
}

/// Build a single `TokenLogprobEntry` for a streaming chunk.
fn build_chunk_logprob_entry(tok: &arf_core::Tokenizer, lp: &TokenLogprob) -> TokenLogprobEntry {
    let token_text = decode(tok, &[lp.token_id]);
    let top_logprobs = lp
        .top_logprobs
        .iter()
        .map(|(id, logprob)| TopLogprobEntry {
            token: decode(tok, &[*id]),
            logprob: *logprob,
        })
        .collect();
    TokenLogprobEntry {
        token: token_text,
        logprob: lp.logprob,
        top_logprobs,
    }
}

#[derive(Serialize)]
struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelCard>,
}

#[derive(Serialize)]
struct ModelCard {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
    /// The context budget (`--max-context`): prompt + generation must fit in it. Agents read it to
    /// decide when to compact. Under two names: `max_model_len` is vLLM's, `context_length` the
    /// common one; both are the same number.
    max_model_len: usize,
    context_length: usize,
    /// What this server can take besides text, as loaded — not what the model family could do.
    capabilities: ModelCapabilities,
}

#[derive(Serialize)]
struct ModelCapabilities {
    /// Image (and video) input: a vision encoder is loaded.
    vision: bool,
    /// Audio input: an audio tower is loaded.
    audio: bool,
    /// Tool calls rendered through the model's own chat template.
    tools: bool,
}

#[derive(Serialize)]
struct ChatCompletion {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

/// OpenAI's token accounting. Required on every non-streaming completion — clients read it to
/// show context consumption and to bill, and its ABSENCE is not a neutral omission: an SDK that
/// expects the field reports zero usage rather than "unknown", which reads as an engine that
/// generated nothing.
///
/// On a streaming response OpenAI omits usage unless the caller asks via
/// `stream_options.include_usage`, so leaving it out of the chunk path is the spec behaviour,
/// not the same gap.
#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
struct ChatChoice {
    index: u32,
    message: ChatCompletionMessage,
    finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<ChoiceLogprobs>,
}

#[derive(Serialize)]
struct ChatCompletionMessage {
    role: &'static str,
    /// `null` when the message is a tool call (OpenAI returns content: null then).
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    /// Present only when the model decided to call tool(s).
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
    /// The model's thinking, when the prompt pre-opened `<think>` (Qwen3.8's own template).
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
}

#[derive(Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatChunkChoice>,
}

/// The final chunk of a stream that asked for `stream_options.include_usage`: OpenAI's shape is
/// an EMPTY `choices` array plus `usage`, sent after the finish chunk and before `[DONE]`.
/// `completion_tokens` counts the tokens the model generated (the raw stream, before any reply
/// cleanup), as the non-streamed `usage` does.
fn usage_chunk_json(
    id: &str,
    created: u64,
    model: &str,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> String {
    serde_json::json!({
        "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    })
    .to_string()
}

#[derive(Serialize)]
struct ChatChunkChoice {
    index: u32,
    delta: ChatDelta,
    finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<ChunkLogprobs>,
}

/// Logprobs object in a streaming chunk (wraps a single token entry).
#[derive(Serialize)]
struct ChunkLogprobs {
    content: Vec<TokenLogprobEntry>,
}

#[derive(Serialize, Default)]
struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    /// The model's thinking, streamed apart from the answer (Qwen3.8's own template pre-opens
    /// `<think>`, so every token up to `</think>` is reasoning).
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
}

#[derive(Serialize)]
struct TextCompletion {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<TextChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct TextChoice {
    index: u32,
    text: String,
    finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<ChoiceLogprobs>,
}

#[derive(Serialize)]
struct TextCompletionChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<TextChunkChoice>,
}

#[derive(Serialize)]
struct TextChunkChoice {
    index: u32,
    text: String,
    finish_reason: Option<String>,
}

// ----------------------------------------------------------------------------
// Handlers.
// ----------------------------------------------------------------------------

/// `GET /health` → `200 OK` (legacy, kept for compatibility).
async fn health() -> StatusCode {
    StatusCode::OK
}

/// `GET /healthz` → router readiness probe.
///
/// Returns `200 OK` with a tiny JSON body. The model is resident by the time
/// the server binds the port (load happens before `axum::serve`), so a
/// successful response means the replica is ready to accept inference traffic.
async fn healthz(State(state): State<AppState>) -> impl IntoResponse {
    let body = serde_json::json!({
        "status": "ok",
        "model": state.model_id,
    });
    (StatusCode::OK, Json(body))
}

/// `GET /metrics` → Prometheus text exposition format (v0.0.4).
///
/// Reads the shared `Arc<Metrics>` atomics published by the actor thread each
/// step. NEVER touches the scheduler or the actor directly — the single-owner
/// discipline is preserved: this handler is a pure atomic reader on the tokio
/// thread pool.
/// `GET /v1/arf/status` — what the server is doing right now, for `arf status`: the model, the
/// requests running and waiting, and each long prompt being read (how far, how fast, how long to
/// go). An agent's first message can take minutes to read; this is where that shows.
async fn arf_status(State(state): State<AppState>) -> impl IntoResponse {
    use std::sync::atomic::Ordering::Relaxed;
    let prefills = state.metrics.prefills.lock().unwrap().clone();
    Json(serde_json::json!({
        "model": state.model_id,
        "running": state.metrics.running.load(Relaxed),
        "waiting": state.metrics.waiting.load(Relaxed),
        "prefill": prefills,
        "low_power": state.metrics.low_power.load(Relaxed),
        "thermal": crate::metrics::thermal_name(state.metrics.thermal.load(Relaxed)),
        "footprint_gb": footprint_gb(),
    }))
}

/// The process's physical footprint in GB (its own memory; the file-backed weights are beside
/// it), or `null` where it is not read.
fn footprint_gb() -> Option<f64> {
    #[cfg(feature = "wgpu")]
    {
        arf_gpu::phys_footprint_bytes().map(|b| (b as f64 / 1e7).round() / 100.0)
    }
    #[cfg(not(feature = "wgpu"))]
    {
        None
    }
}

async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    let body = state.metrics.render();
    (
        StatusCode::OK,
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
}

/// Live snapshot for the Arf HUD (the web app polls this ~4 Hz). A flat JSON
/// view of the `Metrics` atomics + the model id — everything the HUD renders, in one
/// cheap lock-free read.
#[derive(Serialize)]
pub struct HudSnapshot {
    pub model_id: String,
    pub running: usize,
    pub waiting: usize,
    pub kv_used: usize,
    pub kv_total: usize,
    pub max_batch: usize,
    pub tokens_generated: u64,
    pub requests_admitted: u64,
    pub requests_finished: u64,
    pub steps: u64,
    pub prefix_reused: u64,
}

/// `GET /hud` → the HUD JSON snapshot.
async fn hud_handler(State(state): State<AppState>) -> Json<HudSnapshot> {
    use std::sync::atomic::Ordering::Relaxed;
    let m = &state.metrics;
    Json(HudSnapshot {
        model_id: state.model_id.clone(),
        running: m.running.load(Relaxed),
        waiting: m.waiting.load(Relaxed),
        kv_used: m.kv_blocks_used.load(Relaxed),
        kv_total: m.kv_blocks_total.load(Relaxed),
        max_batch: state
            .max_tokens_cap
            .min(64)
            .max(state.metrics.running.load(Relaxed)), // best-effort cap
        tokens_generated: m.tokens_generated.load(Relaxed),
        requests_admitted: m.requests_admitted.load(Relaxed),
        requests_finished: m.requests_finished.load(Relaxed),
        steps: m.steps.load(Relaxed),
        prefix_reused: m.prefix_cache_tokens_reused.load(Relaxed),
    })
}

/// `GET /profile` → the live GPU profiler snapshot as JSON (real per-kernel time /
/// bandwidth / roofline). Feeds any monitoring client with REAL per-kernel data.
/// Empty (`steps: 0`) until decode has run with profiling enabled.
async fn profile_handler() -> axum::response::Response {
    use axum::response::IntoResponse;
    let body = profile_json();
    (
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        body,
    )
        .into_response()
}

/// Render the profiler snapshot JSON. wgpu backend = real data; otherwise an empty doc.
#[cfg(feature = "wgpu")]
fn profile_json() -> String {
    use arf_gpu::gpu::profile;
    // M4 Max unified-memory bandwidth roofline (GB/s) — used for the roofline_pct column.
    const ROOFLINE_GBPS: f64 = 546.0;
    profile::to_json(&profile::snapshot(), ROOFLINE_GBPS)
}

#[cfg(not(feature = "wgpu"))]
fn profile_json() -> String {
    "{ \"steps\": 0, \"total_us\": 0, \"last_step_us\": 0, \"roofline_gbps\": 0, \"kernels\": [] }"
        .to_string()
}

/// Where the FLUX weights live (clip_l.safetensors, t5xxl-Q8_0.gguf, flux1-schnell-Q4_K_S.gguf,
/// ae.safetensors, tokenizers/): `ARF_FLUX_DIR` when set, else `flux-schnell` in the models
/// folder every other model is found in (`arf_cli`'s `default_models_dir`: `ARF_MODELS_DIR`,
/// else `./models` when it exists, else `~/.arf/models`).
///
/// It was `ARF_FLUX_DIR` or `./models/flux-schnell` relative to wherever the server was started
/// (until 2026-10-07): an installed `arf` reached image generation only with the variable set.
#[cfg(feature = "wgpu")] // its only callers are the FLUX handlers, which need wgpu
fn flux_models_dir() -> std::path::PathBuf {
    let set = |v: Option<std::ffi::OsString>| v.filter(|v| !v.is_empty());
    flux_dir_from(
        set(std::env::var_os("ARF_FLUX_DIR")),
        set(std::env::var_os("ARF_MODELS_DIR")),
        std::path::Path::new("./models").is_dir(),
        set(std::env::var_os("HOME")),
    )
}

/// [`flux_models_dir`] on its inputs: the two variables, whether `./models` exists, and `HOME`.
#[cfg_attr(not(feature = "wgpu"), allow(dead_code))]
fn flux_dir_from(
    flux_dir: Option<std::ffi::OsString>,
    models_dir: Option<std::ffi::OsString>,
    local_models: bool,
    home: Option<std::ffi::OsString>,
) -> std::path::PathBuf {
    use std::path::PathBuf;
    if let Some(d) = flux_dir {
        return PathBuf::from(d);
    }
    let models = match (models_dir, local_models, home) {
        (Some(d), _, _) => PathBuf::from(d),
        (None, true, _) | (None, false, None) => PathBuf::from("./models"),
        (None, false, Some(h)) => PathBuf::from(h).join(".arf").join("models"),
    };
    models.join("flux-schnell")
}

/// The `x-arf-warning` header on a generation response from a greedy-only model: sampling,
/// logprobs, a grammar and a repetition penalty are not applied there (`BatchedBackend::
/// greedy_only`). The server log says it per request; this is how a client can tell.
async fn greedy_only_warning(
    State(why): State<Option<&'static str>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let generation = matches!(
        req.uri().path(),
        "/v1/chat/completions" | "/v1/completions" | "/v1/messages" | "/v1/responses"
    );
    let mut resp = next.run(req).await;
    if let (true, Some(why)) = (generation, why) {
        let text = format!(
            "this model is served greedily: temperature, top_p, top_k, logprobs, response_format \
             and repetition_penalty are not applied ({why})"
        );
        if let Ok(v) = axum::http::HeaderValue::from_str(&text) {
            resp.headers_mut().insert("x-arf-warning", v);
        }
    }
    resp
}

tokio::task_local! {
    /// Set while a request with `x-arf-background: 1` is handled ([`background_request`]).
    static BACKGROUND: bool;
}

/// `x-arf-background: 1` asks that the request's prompt be read only when nothing else has work
/// (`actor::Job::background`): `arf launch` sends an agent's fixed prompts this way ahead of the
/// first session, so a user's own request never waits behind them.
async fn background_request(req: axum::extract::Request, next: axum::middleware::Next) -> Response {
    let on = req
        .headers()
        .get("x-arf-background")
        .is_some_and(|v| v.as_bytes() == b"1");
    BACKGROUND.scope(on, next.run(req)).await
}

/// The refusal of an image sent to a model serving text only.
const IMAGE_NEEDS_ENCODER: &str = "image: this model is serving text only; image input needs its \
vision projector (start the server with --mmproj <mmproj.gguf>, or serve a bundle that has one)";

/// `POST /v1/images/generations` — prompt → PNG (base64), OpenAI-images-shaped + a denoise trace.
#[cfg(feature = "wgpu")]
async fn images_generations_handler(
    Json(req): Json<crate::flux::ImageRequest>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    match crate::flux::images_generations(
        req.prompt,
        req.steps,
        req.size,
        req.seed,
        flux_models_dir(),
    )
    .await
    {
        Ok(resp) => Json(resp).into_response(),
        Err(e) => (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
    }
}
#[cfg(not(feature = "wgpu"))]
async fn images_generations_handler() -> axum::response::Response {
    use axum::response::IntoResponse;
    (
        axum::http::StatusCode::NOT_IMPLEMENTED,
        "image generation requires the wgpu backend",
    )
        .into_response()
}

/// `GET /flux/trace` — the last generation's per-step denoise trace (for the HUD timeline).
#[cfg(feature = "wgpu")]
async fn flux_trace_handler() -> axum::response::Response {
    use axum::response::IntoResponse;
    Json(crate::flux::last_trace(flux_models_dir())).into_response()
}
#[cfg(not(feature = "wgpu"))]
async fn flux_trace_handler() -> axum::response::Response {
    use axum::response::IntoResponse;
    Json::<Vec<()>>(vec![]).into_response()
}

/// `GET /v1/models` → the single loaded model: its id, its context budget, and what it accepts.
async fn list_models(State(state): State<AppState>) -> Json<ModelsResponse> {
    Json(models_response(&state))
}

fn models_response(state: &AppState) -> ModelsResponse {
    ModelsResponse {
        object: "list",
        data: vec![ModelCard {
            id: state.model_id.clone(),
            object: "model",
            created: now(),
            owned_by: "arf",
            max_model_len: state.max_tokens_cap,
            context_length: state.max_tokens_cap,
            capabilities: ModelCapabilities {
                vision: state.vision_enabled || state.qwen_vision.is_some(),
                audio: state.qwen_audio.is_some(),
                tools: state.chat_template.is_some(),
            },
        }],
    }
}

// ----------------------------------------------------------------------------
// HuggingFace model search (GET /hf/search).
//
// A thin SERVER-SIDE proxy to the HuggingFace models API, filtered to GGUF repos.
// The optional HF_TOKEN (read from the process env, populated by `load_dotenv`)
// is attached as a Bearer header HERE — server-side only. The token is NEVER
// included in the response and NEVER reaches the browser; the client just sees
// the public model fields. Best-effort: a too-short query, a network failure, or
// any decode error yields an empty list rather than an error status.
// ----------------------------------------------------------------------------

/// Query string for `GET /hf/search`: `q` is the search term, `limit` caps results.
#[derive(Deserialize)]
struct HfSearchQuery {
    #[serde(default)]
    q: String,
    #[serde(default = "default_hf_limit")]
    limit: usize,
}

fn default_hf_limit() -> usize {
    20
}

/// One model card returned to the client. Only PUBLIC fields — no token, ever.
#[derive(Serialize, Deserialize)]
struct HfModel {
    id: String,
    #[serde(default)]
    downloads: u64,
    #[serde(default)]
    likes: u64,
    /// HuggingFace reports `gated` as either a bool OR a string (e.g. `"manual"`,
    /// `"auto"`, `false`). A custom deserializer coerces both to a bool so a quirky
    /// value can't fail the whole response (which would silently empty the list).
    #[serde(default, deserialize_with = "deserialize_gated")]
    gated: bool,
    #[serde(default)]
    tags: Vec<String>,
}

/// Coerce HF's bool-or-string `gated` field into a bool. `false`/missing → false;
/// `true` or any non-empty string other than `"false"` → true.
fn deserialize_gated<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v = serde_json::Value::deserialize(deserializer)?;
    Ok(match v {
        serde_json::Value::Bool(b) => b,
        serde_json::Value::String(s) => {
            let s = s.trim();
            !s.is_empty() && !s.eq_ignore_ascii_case("false")
        }
        serde_json::Value::Null => false,
        // Any other JSON type (number/array/object) — treat presence as gated.
        _ => true,
    })
}

/// `GET /hf/search?q=<term>&limit=<n>` → a list of GGUF models from HuggingFace.
///
/// Best-effort: returns an empty vec on a query under 2 characters or on ANY error
/// (network, non-2xx, decode). The HF_TOKEN, when set in the env, is attached as a
/// `Authorization: Bearer` header on the upstream request ONLY — it is never echoed
/// back to the caller.
async fn hf_search(Query(params): Query<HfSearchQuery>) -> Json<Vec<HfModel>> {
    let q = params.q.trim();
    if q.len() < 2 {
        return Json(Vec::new());
    }
    let limit = params.limit.clamp(1, 100);
    Json(hf_search_inner(q, limit).await.unwrap_or_default())
}

/// The fallible body of `hf_search`; any `Err` becomes an empty list at the caller.
async fn hf_search_inner(q: &str, limit: usize) -> Result<Vec<HfModel>, reqwest::Error> {
    let url = format!(
        "https://huggingface.co/api/models?search={q}&filter=gguf&sort=downloads&direction=-1&limit={limit}&full=true",
        q = urlencode(q),
        limit = limit,
    );
    let mut req = reqwest::Client::new().get(&url);
    // Attach the token server-side ONLY when present. Never logged, never returned.
    if let Ok(token) = std::env::var("HF_TOKEN") {
        if !token.trim().is_empty() {
            req = req.bearer_auth(token);
        }
    }
    let resp = req.send().await?;
    if !resp.status().is_success() {
        return Ok(Vec::new());
    }
    let models: Vec<HfModel> = resp.json().await?;
    Ok(models)
}

/// Minimal percent-encoding for the `search` query param (alnum, `-_.~` pass
/// through; everything else is `%XX`). Avoids pulling in a urlencoding crate.
fn urlencode(s: &str) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            // fmt::Write into a String is infallible.
            _ => write!(out, "%{b:02X}").unwrap(),
        }
    }
    out
}

/// How a chat request's prompt was built, which decides how its reply is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PromptPath {
    /// No tools and no tool history: the hand-written templates, byte-for-byte as before.
    Plain,
    /// The model's own chat template (`jinja_chat`), replies parsed in its tool format.
    Native(ToolFormat),
    /// Tools or tool history without a usable template: the fenced-JSON protocol.
    Fenced,
}

/// Log, once per process, which path tool requests take — the proof, in the server log, that the
/// native template is really DISPATCHED (Note: a flag that changes nothing looks exactly
/// like a flag that works). A render failure is logged once with its message.
///
/// Bits: 0 / 1 which of the two paths `/v1/messages` took (`anthropic::plan_messages` — its own
/// bits, so the proof for that route is not hidden behind this one's); 2 a cut call dropped,
/// 3 unparsed text after a marker, 4 a bare marker dropped ([`log_end_note`]); 5 the fenced
/// protocol; 6 a render failure; 7 a forcing `tool_choice` that could not be enforced; 8 one
/// enforced by prefill ([`enforce_tool_choice`]). The native path logs once per distinct thinking
/// mode instead ([`log_native_mode_once`]).
fn log_tool_path_once(bit: u8, msg: impl FnOnce() -> String) {
    use std::sync::atomic::AtomicU16;
    static SEEN: AtomicU16 = AtomicU16::new(0);
    if SEEN.fetch_or(1 << bit, Ordering::Relaxed) & (1 << bit) == 0 {
        eprintln!("[serve] {}", msg());
    }
}

/// Log, once per distinct mode, that a tool request was rendered by the model's own template and
/// with which thinking settings — so an agent run's log says which reasoning mode it ran under
/// (the 22 s / 62 s baselines are only comparable at the same one).
fn log_native_mode_once(format: ToolFormat, vars: &crate::jinja_chat::TemplateVars) {
    static SEEN: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    let show = |v: Option<String>| v.unwrap_or_else(|| "template default".into());
    let line = format!(
        "tool request rendered by the model's own chat template ({} calls; enable_thinking={}, \
         reasoning_effort={}, preserve_thinking={})",
        format.as_str(),
        show(vars.enable_thinking.map(|b| b.to_string())),
        show(vars.reasoning_effort.clone()),
        show(vars.preserve_thinking.map(|b| b.to_string())),
    );
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&line) {
        eprintln!("[serve] {line}");
        seen.push(line);
    }
}

/// What the end of a native reply is worth logging ([`EndNote`]), once each per process.
fn log_end_note(note: Option<EndNote>) {
    match note {
        Some(EndNote::CutCall) => log_tool_path_once(2, || {
            "a tool call cut off by max_tokens was NOT sent as a call (its text went back as \
             content, finish_reason length; logged once)"
                .into()
        }),
        Some(EndNote::Unparsed) => log_tool_path_once(3, || {
            "text after a tool-call marker did not parse as a call — sent as content (logged once)"
                .into()
        }),
        Some(EndNote::BareMarker) => log_tool_path_once(4, || {
            "a tool-call marker with no call after it was dropped from the reply (logged once)"
                .into()
        }),
        None => {}
    }
}

/// The request's reasoning effort as the prompt paths read it — ONE resolution for plain and
/// tool requests alike. `"none"` = non-thinking. A grammar-constrained reply (JSON mode) is always
/// `"none"`; then `chat_template_kwargs.enable_thinking: false`; then
/// `chat_template_kwargs.reasoning_effort`; then the request's `reasoning_effort`. (`None`: the
/// server default, `ARF_REASONING_EFFORT`, else the template's.)
fn request_effort(req: &ChatRequest, constrained: bool) -> Option<&str> {
    if constrained {
        return Some("none");
    }
    let kw = req.chat_template_kwargs.as_ref();
    if kw.and_then(|k| k.enable_thinking) == Some(false) {
        return Some("none");
    }
    kw.and_then(|k| k.reasoning_effort.as_deref())
        .or(req.reasoning_effort.as_deref())
}

/// The model's-own-template variables for a request.
///
/// `enable_thinking`: false when the effort is `"none"`; else the request's
/// `chat_template_kwargs.enable_thinking`; else false when the operator turned the thinking
/// template off (`ARF_NO_QWEN_THINK_TEMPLATE`, `no_think_env`) — plain chat then runs ChatML
/// without `<think>`, and tool requests must not think where plain ones do not; else UNDEFINED,
/// the template's own default. Until 2026-09-26 (review) it was `Some(effort != "none")`: every
/// tool request thought at the template default even on a server whose plain chat did not.
/// `env_effort` is `ARF_REASONING_EFFORT`, the same server default the hand-written template
/// reads.
fn template_vars(
    effort: Option<&str>,
    kwargs: Option<&ChatTemplateKwargs>,
    no_think_env: bool,
    env_effort: Option<String>,
) -> crate::jinja_chat::TemplateVars {
    let effort = effort.map(str::to_string).or(env_effort);
    let enable_thinking = if effort.as_deref() == Some("none") {
        Some(false)
    } else if let Some(b) = kwargs.and_then(|k| k.enable_thinking) {
        Some(b)
    } else if no_think_env {
        Some(false)
    } else {
        None
    };
    crate::jinja_chat::TemplateVars {
        add_generation_prompt: true,
        enable_thinking,
        reasoning_effort: template_effort(effort.as_deref()),
        preserve_thinking: kwargs.and_then(|k| k.preserve_thinking),
    }
}

/// `reasoning_effort` as Qwen3.8's template accepts it. Its own mapping sends `high` to `xhigh`
/// and RAISES on anything else, which would drop the request to the fallback protocol — so
/// `minimal` (OpenAI's lowest) maps to `low` and an unknown value to the template's default.
fn template_effort(effort: Option<&str>) -> Option<String> {
    match effort? {
        "minimal" | "low" => Some("low".into()),
        "medium" => Some("medium".into()),
        "high" | "xhigh" => Some("xhigh".into()),
        _ => None,
    }
}

/// Render a tool request through the model's own template. `tools` comes from the raw body so
/// its key order is the client's (HF passes the request's dicts through in order); with
/// `offer_tools` false (`tool_choice: "none"`) it is left undefined.
fn render_native(
    tpl: &crate::jinja_chat::ChatTemplate,
    body: &[u8],
    req: &ChatRequest,
    vars: &crate::jinja_chat::TemplateVars,
    offer_tools: bool,
) -> Result<String, String> {
    tpl.render_chat(&req.messages, native_tools_of(body, offer_tools)?, vars)
}

/// The request's `tools` as the native template takes them — read from the raw body, in the
/// client's key order; `None` when none are offered. (Lifted out of `render_native` unchanged,
/// so the prefix anchor's sentinel renders get exactly the same tools value.)
fn native_tools_of(body: &[u8], offer_tools: bool) -> Result<Option<minijinja::Value>, String> {
    #[derive(Deserialize)]
    struct RawTools {
        #[serde(default)]
        tools: Option<minijinja::Value>,
    }
    Ok(serde_json::from_slice::<RawTools>(body)
        .map_err(|e| format!("tools: {e}"))?
        .tools
        .filter(|t| offer_tools && t.len().is_some_and(|n| n > 0)))
}

/// A `/v1/chat/completions` request's PREFIX ANCHOR (`crate::prefix_anchor`): where its shared
/// system + tools prefix ends in `prompt_ids`, computed on the path its prompt ACTUALLY took —
/// the sentinel renders go through the same template, tools and flags as the real one:
/// - native: the model's own template with the body's tools and the request's `vars`;
/// - fenced: `legacy_tool_messages` (tool prompt + leading system text merged) and the
///   hand-written template at the request's effort;
/// - plain: the hand-written template at the request's effort.
///
/// `None` for a request with images (it takes no snapshots) and wherever `anchor` finds none.
///
/// And its TOOLS ANCHOR (2026-09-27, `crate::prefix_anchor::tools_anchor`) on the same path,
/// when it offers tools and has a system text: the native and fenced renders both take the
/// system text as the leading system message(s) (`system_lead`), the fenced one merging its
/// tool prompt in ahead of it. Plain offers no tools, so it has none.
#[allow(clippy::too_many_arguments)]
fn chat_prefix_anchor(
    cache: &crate::prefix_anchor::AnchorCache,
    tok: &Tokenizer,
    tpl: Option<&crate::jinja_chat::ChatTemplate>,
    req: &ChatRequest,
    body: &[u8],
    path: PromptPath,
    vars: &crate::jinja_chat::TemplateVars,
    effort: Option<&str>,
    tools: &[Tool],
    prompt_ids: &[u32],
) -> crate::prefix_anchor::Anchors {
    use crate::prefix_anchor::{
        anchor, anchors, header_tail, leading_system, system_lead, Anchors,
    };
    if req
        .messages
        .iter()
        .any(|m| !m.images.is_empty() || !m.audios.is_empty())
    {
        return Anchors::default();
    }
    let lead = leading_system(&req.messages);
    let has_tools = !tools.is_empty();
    // The tools anchor needs tools AND a system text to vary (module docs).
    let has_system = lead.iter().any(|m| !m.content.trim().is_empty());
    let tools_lead: Option<&dyn Fn(&str) -> Vec<ChatMessage>> =
        (has_tools && has_system).then_some(&system_lead);
    match path {
        PromptPath::Native(_) => {
            let Some(tpl) = tpl else {
                return Anchors::default();
            };
            let Ok(raw) = native_tools_of(body, has_tools) else {
                return Anchors::default();
            };
            let shape = ("chat/native", format!("{vars:?}"), raw.clone());
            let render = |m: &[ChatMessage]| tpl.render_chat(m, raw.clone(), vars).ok();
            anchors(
                cache, tok, lead, has_tools, shape, render, prompt_ids, tools_lead,
            )
        }
        PromptPath::Fenced => {
            let shape = ("chat/fenced", effort, serde_json::to_string(tools).ok());
            let offered = has_tools.then_some(tools);
            let render = |m: &[ChatMessage]| {
                let msgs = crate::chat::legacy_tool_messages(m, offered);
                Some(crate::chat::apply_chat_template_with(tok, &msgs, effort))
            };
            anchors(
                cache, tok, lead, has_tools, shape, render, prompt_ids, tools_lead,
            )
        }
        PromptPath::Plain => {
            let render =
                |m: &[ChatMessage]| Some(crate::chat::apply_chat_template_with(tok, m, effort));
            Anchors {
                system: anchor(
                    cache,
                    tok,
                    lead,
                    false,
                    ("chat/plain", effort),
                    render,
                    prompt_ids,
                ),
                tools: None,
                header: header_tail(cache, tok, ("chat/plain", effort), render, prompt_ids),
            }
        }
    }
}

/// A `/v1/chat/completions` body as the handler reads it: parsed, then Claude Code's per-request
/// billing header taken out of every system / developer message ([`strip_billing_headers`]).
fn chat_request(body: &[u8]) -> Result<ChatRequest, serde_json::Error> {
    let mut req: ChatRequest = serde_json::from_slice(body)?;
    strip_billing_headers(&mut req.messages);
    Ok(req)
}

/// Claude Code's billing header (`x-anthropic-billing-header: cc_version=…; …`), removed from
/// system and developer message text — `/v1/messages` has dropped it since 2026-09-26
/// (`anthropic::system_message`, which says why: its value changes with every request, and at the
/// head of the prompt it made every prefix cache, anchor and snapshot miss). Claude Code behind
/// an OpenAI-translating gateway reaches THIS route with the header still in the system text,
/// and until 2026-10-04 it went to the model here. A message that held nothing else is dropped,
/// as `/v1/messages` drops an empty system prompt. Messages carrying images or audio are left
/// alone: their byte offsets index the text (`ChatMessage::image_at`).
fn strip_billing_headers(messages: &mut Vec<ChatMessage>) {
    messages.retain_mut(|m| {
        if !matches!(m.role.as_str(), "system" | "developer")
            || !m.images.is_empty()
            || !m.audios.is_empty()
        {
            return true;
        }
        match anthropic::strip_billing_header(&m.content) {
            Some(rest) => {
                m.content = rest;
                !m.content.is_empty()
            }
            None => true,
        }
    });
}

/// A `tool_choice` that asks for more than `auto`: a call to ANY offered tool, or to the NAMED
/// one. Both dialects' shapes are read here — OpenAI's `"required"` and
/// `{"type": "function", "function": {"name"}}`, Anthropic's `{"type": "any"}` and
/// `{"type": "tool", "name"}` — so the two endpoints enforce one rule.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ForcedCall {
    Any,
    Named(String),
}

/// The [`ForcedCall`] a `tool_choice` value asks for; `None` for `auto`, `none`, an absent field
/// or a shape this file does not know.
fn forced_call(choice: Option<&serde_json::Value>) -> Option<ForcedCall> {
    use serde_json::Value;
    let (kind, name) = match choice? {
        Value::String(s) => (s.as_str(), None),
        Value::Object(o) => (
            o.get("type")?.as_str()?,
            o.get("name")
                .or_else(|| o.get("function").and_then(|f| f.get("name")))
                .and_then(Value::as_str),
        ),
        _ => return None,
    };
    match kind {
        "required" | "any" => Some(ForcedCall::Any),
        "tool" | "function" => name.map(|n| ForcedCall::Named(n.to_string())),
        _ => None,
    }
}

/// The opening of a call in the format the prompt taught — what the model itself writes first
/// when it calls a tool (`jinja_chat::ToolFormat`'s shapes; `chat::tool_system_prompt`'s fenced
/// JSON) — up to the function's name when one is forced, so the model writes only the rest.
fn call_prefill(forced: &ForcedCall, format: ToolFormat) -> String {
    let name = match forced {
        ForcedCall::Any => None,
        ForcedCall::Named(n) => Some(n.as_str()),
    };
    let quoted = |n: &str| serde_json::Value::from(n).to_string();
    match (format, name) {
        (ToolFormat::QwenXml | ToolFormat::HermesJson, None) => "<tool_call>\n".into(),
        (ToolFormat::QwenXml, Some(n)) => format!("<tool_call>\n<function={n}>\n"),
        (ToolFormat::HermesJson, Some(n)) => {
            format!("<tool_call>\n{{\"name\": {}, \"arguments\": ", quoted(n))
        }
        (ToolFormat::None, None) => "```json\n".into(),
        (ToolFormat::None, Some(n)) => {
            format!("```json\n{{\"function\": {}, \"arguments\": ", quoted(n))
        }
    }
}

/// The PREFILL that enforces a forcing `tool_choice` ([`forced_call`]) on a prompt that teaches
/// `format` with `tools` on offer, or `None` when there is nothing to force — or it cannot be
/// enforced, which is said once in the log: no tools on offer, a JSON-constrained reply
/// (`constrained`: the grammar would refuse the call's opening), or a name that is not on offer.
///
/// The caller appends it to the prompt with [`prefill_call`] and reads the reply as
/// `prefill + reply`. This is the cheapest enforcement Arf's machinery has: the sampler's only
/// grammar is JSON, and a grammar-constrained job leaves the fast decode step (it samples on the
/// CPU from full logits) for its whole reply. What prefill does NOT guarantee: that the model
/// completes a call that parses. If it does not, the reply comes back as text, as an unforced one
/// would. Until 2026-10-04 `required` / `any` / a named tool were accepted and not enforced.
fn enforce_tool_choice(
    choice: Option<&serde_json::Value>,
    tools: &[Tool],
    format: ToolFormat,
    constrained: bool,
) -> Option<String> {
    let forced = forced_call(choice)?;
    let why_not = if tools.is_empty() {
        Some("no tools are on offer")
    } else if constrained {
        Some("response_format constrains the reply to JSON")
    } else {
        match &forced {
            ForcedCall::Named(n) if !tools.iter().any(|t| t.function.name == *n) => {
                Some("it names a tool that is not on offer")
            }
            _ => None,
        }
    };
    let c = choice.map(|c| c.to_string()).unwrap_or_default();
    if let Some(why) = why_not {
        log_tool_path_once(7, || {
            format!("tool_choice {c} is NOT enforced: {why} — the model chooses (logged once)")
        });
        return None;
    }
    let prefill = call_prefill(&forced, format);
    log_tool_path_once(8, || {
        format!(
            "tool_choice {c} enforced by prefill: the reply is begun with {prefill:?} ({} \
             calls), after an open <think> is closed (logged once)",
            match format {
                ToolFormat::None => "fenced-JSON",
                f => f.as_str(),
            }
        )
    });
    Some(prefill)
}

/// Append a forced call's opening to a rendered prompt. A prompt whose cue pre-opened `<think>`
/// has it closed first, empty, as Qwen3's own non-thinking cue renders it (`<think>\n\n</think>\n\n`):
/// the call is the answer, never the reasoning. Anthropic refuses extended thinking with a forced
/// `tool_choice` for the same reason. Closing it HERE, after the conversation, rather than
/// rendering the prompt non-thinking leaves the system and tools prefix byte-identical to the
/// same agent's unforced requests, so their prefix cache and anchor are shared.
fn prefill_call(prompt: &mut String, prefill: &str) {
    if crate::chat::prompt_opens_think(prompt) {
        prompt.truncate(prompt.trim_end().len());
        prompt.push_str("\n\n</think>\n\n");
    }
    prompt.push_str(prefill);
}

/// The tools a request offers the model: none when `tool_choice` is `"none"`.
fn offered_tools(req: &ChatRequest) -> &[Tool] {
    if req.tool_choice.as_ref().and_then(|c| c.as_str()) == Some("none") {
        return &[];
    }
    req.tools.as_deref().unwrap_or(&[])
}

/// How a chat request's prompt is built.
#[derive(Debug)]
enum Plan {
    /// The model's own template rendered `prompt`; `thinking` is read off it (its cue pre-opened
    /// `<think>`).
    Native {
        prompt: String,
        format: ToolFormat,
        thinking: bool,
    },
    /// The hand-written templates: [`PromptPath::Plain`] or [`PromptPath::Fenced`].
    HandWritten(PromptPath),
}

/// Choose a chat request's prompt path. Pure apart from the once-per-process log lines, so every
/// branch is unit-tested (`plan_chat_*`):
/// - no tools offered and no tool history → Plain (the hand-written templates, byte-for-byte);
/// - otherwise the model's own template when it teaches a format Arf parses and renders —
///   except with vision on and images present (the Gemma-3 image splice is hand-written), and
///   the same for audio with an audio tower loaded (the native render reads only `content`, so
///   the audio sentinels must go through the hand-written templates);
/// - otherwise, or when the render fails, the fenced-JSON protocol.
///
/// `offer_tools`: the request has tools and `tool_choice` is not `"none"`. `vision_enabled`: a
/// vision OR audio encoder is loaded.
fn plan_chat(
    req: &ChatRequest,
    body: &[u8],
    tpl: Option<&crate::jinja_chat::ChatTemplate>,
    vision_enabled: bool,
    vars: &crate::jinja_chat::TemplateVars,
    offer_tools: bool,
) -> Plan {
    if !offer_tools && !req.messages.iter().any(ChatMessage::is_tool_turn) {
        return Plan::HandWritten(PromptPath::Plain);
    }
    let has_images = req
        .messages
        .iter()
        .any(|m| !m.images.is_empty() || !m.audios.is_empty());
    if let Some(tpl) = tpl
        .filter(|t| t.tool_format() != ToolFormat::None)
        .filter(|_| !(vision_enabled && has_images))
    {
        match render_native(tpl, body, req, vars, offer_tools) {
            Ok(prompt) => {
                let format = tpl.tool_format();
                log_native_mode_once(format, vars);
                return Plan::Native {
                    thinking: crate::chat::prompt_opens_think(&prompt),
                    prompt,
                    format,
                };
            }
            Err(e) => log_tool_path_once(6, || {
                format!(
                    "chat template failed to render a tool request ({e}) — falling back to the \
                     fenced-JSON protocol (logged once)"
                )
            }),
        }
    }
    log_tool_path_once(5, || {
        "tool request rendered with the fenced-JSON protocol (no usable chat template)".into()
    });
    Plan::HandWritten(PromptPath::Fenced)
}

/// `Ok(max_tokens)` for a prompt of `prompt_tokens` under a context of `cap`: the request's own
/// limit (or everything left) clamped to what is left. `Err` when the prompt alone fills the
/// context — until 2026-09-26 such a prompt was queued anyway, and one the KV pool could not hold
/// blocked the head of the queue.
///
/// A request that omits the limit still gets EVERYTHING left, as under the old
/// `clamp_max_tokens(requested, cap)` this replaces — so a client that doesn't set `max_tokens`
/// gets a long generation instead of a 1024-token cut; `--max-context` stays the single knob that
/// bounds output length. What changed is that the prompt now counts against it.
fn context_budget(
    prompt_tokens: usize,
    requested: Option<usize>,
    cap: usize,
) -> Result<usize, String> {
    if prompt_tokens >= cap {
        return Err(format!(
            "This model's maximum context length is {cap} tokens. However, your messages resulted \
             in {prompt_tokens} tokens. Please reduce the length of the messages."
        ));
    }
    let left = cap - prompt_tokens;
    Ok(requested.unwrap_or(left).clamp(1, left))
}

/// HTTP 400 in OpenAI's context-overflow shape: `code: context_length_exceeded` and OpenAI's own
/// message wording, the two things a client can match to tell "compact and retry" from any
/// other failure. (Which of them OpenCode and Pi actually match is not verified here — that is
/// part of the live run.)
fn context_length_error(message: &str, param: &str) -> Response {
    let body = serde_json::json!({ "error": {
        "message": message,
        "type": "invalid_request_error",
        "param": param,
        "code": "context_length_exceeded",
    }});
    (StatusCode::BAD_REQUEST, Json(body)).into_response()
}

/// A finished native-format reply, read the same way whether it was streamed or not: run the
/// whole text through the splitter the stream uses, then parse the buffered call text.
struct NativeReply {
    reasoning: Option<String>,
    content: String,
    calls: Option<Vec<ToolCall>>,
}

/// `finish`: why the job stopped — a call the token budget cut off is not a call
/// ([`settle_native`]).
fn read_native_reply(
    text: &str,
    format: ToolFormat,
    thinking: bool,
    tools: &[Tool],
    finish: FinishReason,
) -> NativeReply {
    let mut s = ReplySplitter::new(format, thinking, true);
    let mut pieces = s.push(text);
    let end = s.finish();
    pieces.extend(end.pieces);
    let mut reasoning = String::new();
    let mut content = String::new();
    for p in pieces {
        match p {
            Piece::Reasoning(t) => reasoning.push_str(&t),
            Piece::Content(t) => content.push_str(&t),
        }
    }
    let settled = settle_native(
        format,
        &end.gap,
        &end.buffered,
        tools,
        finish != FinishReason::Stop,
    );
    log_end_note(settled.note);
    content.push_str(&settled.content);
    NativeReply {
        reasoning: (!reasoning.is_empty()).then_some(reasoning),
        content,
        calls: (!settled.calls.is_empty()).then_some(settled.calls),
    }
}

/// `POST /v1/chat/completions`. Applies the chat template, tokenizes, and either streams SSE
/// chunks (`stream: true`) or returns one completion.
///
/// The body is parsed here rather than by `Json<ChatRequest>`: the native-template path reads
/// `tools` a second time with its key order intact, and a malformed body now gets an
/// OpenAI-shaped 400 rather than axum's plain-text 422.
async fn chat_completions(State(state): State<AppState>, body: axum::body::Bytes) -> Response {
    let req = match chat_request(&body) {
        Ok(r) => r,
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!("invalid request body: {e}"),
            )
        }
    };
    if req.messages.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "messages must not be empty");
    }
    let stops = match StopStrings::from_openai(req.stop.as_ref()) {
        Ok(s) => s,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, &e),
    };
    // `tool_choice: "none"`: the tools are neither rendered nor parsed. `required` and a named
    // function are enforced by prefill once the prompt is rendered (`enforce_tool_choice`).
    let tools = offered_tools(&req);
    let has_tools = !tools.is_empty();
    let tool_names: Vec<&str> = tools.iter().map(|t| t.function.name.as_str()).collect();

    // A grammar-constrained reply (JSON mode) is rendered NON-THINKING: the constraint applies
    // from the first generated token, so it must not be spent inside a pre-opened <think>
    // (2026-09-23: JSON mode returned an EMPTY `content` — the JSON had gone into the reasoning).
    let constrained = req
        .response_format
        .as_ref()
        .is_some_and(|rf| rf.format_type != "text");
    let effort = request_effort(&req, constrained);

    // Tools or tool history: the model's OWN template when it teaches a format Arf parses.
    let vars = template_vars(
        effort,
        req.chat_template_kwargs.as_ref(),
        std::env::var_os("ARF_NO_QWEN_THINK_TEMPLATE").is_some(),
        std::env::var("ARF_REASONING_EFFORT").ok(),
    );
    // Qwen3-Omni audio (M3, 2026-09-27): refused outright without an audio tower — answering a
    // spoken question as if the audio were not there would be worse than an error.
    let has_audio = req.messages.iter().any(|m| !m.audios.is_empty());
    if has_audio {
        if state.qwen_audio.is_none() {
            return error_response(
                StatusCode::BAD_REQUEST,
                "audio: this server has no audio encoder (start arf-serve with --audio-tower, \
                 for a Qwen3-Omni Thinker)",
            );
        }
        if req.messages.iter().any(|m| !m.images.is_empty()) {
            return error_response(
                StatusCode::BAD_REQUEST,
                "audio: audio and image/video parts in one request are not supported yet",
            );
        }
    }
    let plan = plan_chat(
        &req,
        &body,
        state.chat_template.as_deref(),
        state.vision_enabled || state.qwen_vision.is_some() || state.qwen_audio.is_some(),
        &vars,
        has_tools,
    );
    // Qwen3.8 images and videos: staged here, laid out after tokenizing, encoded before submit.
    let mut qwen_images: Vec<crate::qwen_image::QwenMedia> = Vec::new();
    // Qwen3-Omni audio: decoded and featurised with the messages, encoded after tokenizing.
    let mut staged_audio: Option<crate::qwen_audio::StagedAudio> = None;

    let mut images: Vec<(Vec<u8>, usize, usize)> = Vec::new();
    let (mut prompt, path, mut thinking) = match plan {
        Plan::Native {
            prompt,
            format,
            thinking,
        } => (prompt, PromptPath::Native(format), thinking),
        Plan::HandWritten(path) => {
            // Tool calling without a usable template: the fenced-JSON protocol, with the tool
            // history rendered readably (`legacy_tool_messages`). Plain chat: the messages as
            // they came, through the hand-written templates exactly as before.
            let base: Vec<ChatMessage> = if path == PromptPath::Fenced {
                crate::chat::legacy_tool_messages(&req.messages, has_tools.then_some(tools))
            } else {
                req.messages.clone()
            };
            // Qwen3-Omni audio: a sentinel where each clip sat, and every clip decoded (ffmpeg for
            // compressed formats) and featurised as ONE processor call, off the async runtime.
            let base = if has_audio {
                let (msgs, refs) = crate::qwen_audio::stage_messages(&base);
                let cap = state.max_tokens_cap;
                let staged = tokio::task::spawn_blocking(move || {
                    let pcm = crate::qwen_audio::decode_all(&refs)?;
                    crate::qwen_audio::StagedAudio::from_pcm(&pcm, cap)
                })
                .await
                .map_err(|e| format!("audio decode task: {e}"))
                .and_then(|r| r);
                match staged {
                    Ok(s) => staged_audio = Some(s),
                    Err(e) => {
                        return error_response(StatusCode::BAD_REQUEST, &format!("audio: {e}"))
                    }
                }
                msgs
            } else {
                base
            };
            // Vision (Gemma-3): decode any image_url parts and splice a `<start_of_image>` +
            // 256×`<image_soft_token>` + `<end_of_image>` block into each message that carried
            // them (ahead of that turn's text). The actor fills those positions with the
            // projector's soft-tokens. Images are decoded here (CPU) in message order; the GPU
            // encode is the actor's. When vision isn't configured, an image request is refused.
            // (2026-10-07: no longer ignored. A model started without a vision encoder answered
            // an image request from the text alone — `gemma3:4b` from its safetensors called a red
            // square "Blue", 20 prompt tokens — which reads as a model that sees badly. Refused.)
            if state.qwen_vision.is_none()
                && !state.vision_enabled
                && base.iter().any(|m| !m.images.is_empty())
            {
                return error_response(StatusCode::BAD_REQUEST, IMAGE_NEEDS_ENCODER);
            }
            let templated_messages: Vec<ChatMessage> = if let Some(enc) = state
                .qwen_vision
                .as_ref()
                .filter(|_| base.iter().any(|m| !m.images.is_empty()))
            {
                match crate::qwen_image::stage_messages(enc, &base) {
                    Ok((msgs, imgs)) => {
                        qwen_images = imgs;
                        msgs
                    }
                    Err(e) => {
                        return error_response(StatusCode::BAD_REQUEST, &format!("image: {e}"))
                    }
                }
            } else if state.vision_enabled {
                let mut out = Vec::with_capacity(base.len());
                for m in &base {
                    if m.images.is_empty() {
                        out.push(m.clone());
                        continue;
                    }
                    if !m.videos.is_empty() {
                        return error_response(
                            StatusCode::BAD_REQUEST,
                            "video: video input needs a Qwen3.8 mmproj (qwen3vl_merger); this \
                             server's vision tower is Gemma-3's",
                        );
                    }
                    let mut block = String::new();
                    for url in &m.images {
                        match crate::chat::decode_image_data_url(url) {
                            // gemma3_image_block already includes the \n\n wrapping the HF
                            // processor injects on both sides of the image block.
                            Ok((rgb, w, h)) => {
                                block.push_str(&crate::chat::gemma3_image_block(
                                    state.vision_tokens,
                                ));
                                images.push((rgb, w, h));
                            }
                            Err(e) => {
                                return error_response(
                                    StatusCode::BAD_REQUEST,
                                    &format!("image: {e}"),
                                )
                            }
                        }
                    }
                    // Canonical Gemma-3 single-image layout is text-FIRST, then the image block
                    // (the documented order). `| trim` semantics: the text is used as-is.
                    out.push(ChatMessage {
                        role: m.role.clone(),
                        content: format!("{}{block}", m.content),
                        ..Default::default()
                    });
                }
                out
            } else {
                base
            };
            // `thinking` must follow the effort ACTUALLY rendered: `reasoning_effort: "none"`
            // from the client renders the non-thinking template too, and keying this on
            // `constrained` alone filed the whole reply under reasoning — empty `content` for
            // every "none" request (2026-09-23).
            let thinking = crate::chat::qwen_think_template_active() && effort != Some("none");
            let prompt = crate::chat::apply_chat_template_with(
                &state.tokenizer,
                &templated_messages,
                effort,
            );
            (prompt, path, thinking)
        }
    };
    // A forcing `tool_choice`: the reply is begun with the call's opening, after the reasoning
    // (if the prompt opened it) is closed, and read back as `prefill + reply` on every path below.
    let prefill = enforce_tool_choice(
        req.tool_choice.as_ref(),
        tools,
        match path {
            PromptPath::Native(f) => f,
            _ => ToolFormat::None,
        },
        constrained,
    )
    .unwrap_or_default();
    if !prefill.is_empty() {
        prefill_call(&mut prompt, &prefill);
        thinking = false;
    }
    let prompt = if qwen_images.is_empty() {
        prompt
    } else {
        match crate::qwen_image::expand_sentinels(&prompt, &qwen_images) {
            Ok(p) => p,
            Err(e) => return error_response(StatusCode::BAD_REQUEST, &format!("image: {e}")),
        }
    };
    let prompt = match &staged_audio {
        None => prompt,
        Some(s) => match crate::qwen_audio::expand_sentinels(&prompt, &s.tokens) {
            Ok(p) => p,
            Err(e) => return error_response(StatusCode::BAD_REQUEST, &format!("audio: {e}")),
        },
    };
    let prompt_ids = match encode_prompt(&state.tokenizer, &prompt) {
        Ok(ids) => ids,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, &format!("tokenize: {e}")),
    };
    let image_prompt = match staged_audio {
        Some(staged) => match audio_encode(&state, staged, &prompt_ids).await {
            Ok(p) => Some(p),
            Err(e) => return error_response(StatusCode::BAD_REQUEST, &format!("audio: {e}")),
        },
        None => match qwen_encode(&state, &qwen_images, &prompt_ids).await {
            Ok(p) => p,
            Err(e) => return error_response(StatusCode::BAD_REQUEST, &format!("image: {e}")),
        },
    };
    // Qwen3.8 images (an M-RoPE layout) are served GREEDY, without logprobs or a grammar: their
    // splice lives on the greedy island record only. Audio rows carry no layout and take the
    // generic path, which serves every sampling mode.
    let island_greedy_only = image_prompt.as_ref().is_some_and(|p| p.mrope.is_some());
    // Positions of the image soft-token slots in the tokenized prompt (256 per image).
    let image_positions = if images.is_empty() {
        Vec::new()
    } else {
        crate::chat::image_soft_token_positions(&state.tokenizer, &prompt_ids)
    };
    // Where the shared system + tools prefix ends — the scheduler snapshots there for the next
    // NEW session of the same agent (`prefix_anchor`, 2026-09-26).
    let anchors = chat_prefix_anchor(
        &state.prefix_anchors,
        &state.tokenizer,
        state.chat_template.as_deref(),
        &req,
        &body,
        path,
        &vars,
        effort,
        tools,
        &prompt_ids,
    );
    // Captured before `prompt_ids` is moved into `submit_job` below — this is the count the
    // model actually attended to (post-template, post-BOS), not a re-encode of the raw text.
    let prompt_tokens = prompt_ids.len();
    let max_tokens = match context_budget(
        prompt_tokens,
        req.max_completion_tokens.or(req.max_tokens),
        state.max_tokens_cap,
    ) {
        Ok(n) => n,
        Err(msg) => return context_length_error(&msg, "messages"),
    };
    let model_id = req.model.clone().unwrap_or_else(|| state.model_id.clone());

    let mut sampling = match build_sampling(
        req.temperature,
        req.top_p,
        req.top_k,
        req.seed,
        req.repetition_penalty,
    ) {
        Ok(s) => s,
        Err((status, msg)) => return error_response(status, &msg),
    };
    if island_greedy_only {
        crate::qwen_image::force_greedy(&mut sampling);
    }
    let want_logprobs = req.logprobs && !island_greedy_only;
    let top_logprobs_n = req.top_logprobs.unwrap_or(0).min(20);
    let include_usage = req.stream_options.as_ref().is_some_and(|o| o.include_usage);
    let response_format_type = req
        .response_format
        .as_ref()
        .map(|rf| rf.format_type.clone())
        .filter(|_| !island_greedy_only);
    let mut rx = match submit_job(
        &state,
        prompt_ids,
        max_tokens,
        sampling,
        want_logprobs,
        top_logprobs_n,
        response_format_type,
        images,
        image_positions,
        image_prompt,
        anchors,
        &stops,
    ) {
        Ok(rx) => rx,
        Err((status, msg)) => return error_response(status, &msg),
    };

    if req.stream && has_tools {
        // Reasoning and prose stream live; only the call itself is buffered (it is only
        // recognisable once complete). Until 2026-09-26 the WHOLE reply was buffered, so an
        // agent turn was silent until it finished.
        let format = match path {
            PromptPath::Native(f) => f,
            _ => ToolFormat::None,
        };
        chat_stream_tools_response(
            state.tokenizer.clone(),
            model_id,
            rx,
            ToolStream {
                format,
                thinking,
                tools: tools.to_vec(),
                include_usage,
                prompt_tokens,
                stops,
                prefill,
            },
        )
    } else if req.stream {
        chat_stream_response(
            state.tokenizer.clone(),
            model_id,
            rx,
            want_logprobs,
            thinking,
            include_usage,
            prompt_tokens,
            stops,
        )
    } else {
        // Accumulate every token, decode once, return a single completion.
        let mut ids = Vec::new();
        let mut lp_data: Vec<Box<crate::actor::TokenLogprob>> = Vec::new();
        let mut finish = FinishReason::Length;
        while let Some(item) = rx.recv().await {
            match item {
                StreamItem::Token(id, lp) => {
                    ids.push(id);
                    if let Some(lp) = lp {
                        lp_data.push(lp);
                    }
                }
                StreamItem::Done(reason) => {
                    finish = reason;
                    break;
                }
            }
        }
        let logprobs_out = if want_logprobs {
            Some(build_choice_logprobs(&state.tokenizer, &lp_data))
        } else {
            None
        };
        // The reply up to its first stop string, which ends it as `stop`.
        let decoded = decode(&state.tokenizer, &ids);
        let (decoded, matched) = stops.cut(&decoded);
        if matched.is_some() {
            finish = FinishReason::Stop;
        }
        let (message, finish_reason) = match path {
            // The model's own format: the same splitter + parser the stream uses, so a streamed
            // and a non-streamed reply agree on what is reasoning, prose and call. (No gemma
            // thought-channel or Muse channel cleanup here: those templates teach no tool format
            // Arf parses, so their requests never take this path.)
            PromptPath::Native(format) if has_tools => {
                // The stop strings cut the GENERATED text; a forced call's opening (never
                // generated, never matched against) is put back in front for the parse.
                let text = format!("{prefill}{decoded}");
                let r = read_native_reply(&text, format, thinking, tools, finish);
                let finish_reason = if r.calls.is_some() {
                    "tool_calls"
                } else {
                    finish.as_str()
                };
                (
                    ChatCompletionMessage {
                        role: "assistant",
                        content: (!r.content.is_empty() || r.calls.is_none()).then_some(r.content),
                        tool_calls: r.calls,
                        reasoning_content: r.reasoning,
                    },
                    finish_reason.to_string(),
                )
            }
            _ => {
                // Strip a leaked gemma "thought" channel marker so it never reaches the user.
                let text = crate::chat::strip_thought(&format!("{prefill}{decoded}"));
                // Muse Glimmer replies on channels (`to=self` thinking, then `to=user` answer).
                // Keep only the answer — the same thing llama surfaces. A reply with no channel
                // header is returned unchanged, so this is a no-op for every other model.
                let (_reasoning, text) = crate::chat::split_muse_channels(&text);
                // Qwen3.8's own template pre-opens `<think>`: split the reply into reasoning +
                // answer.
                let (qwen_reasoning, text) = if thinking {
                    crate::chat::split_pretag_think(&text)
                } else {
                    (None, text)
                };
                // Tool calling: if the request offered tools and the reply contains a JSON tool
                // call, surface it as `tool_calls` with finish_reason `tool_calls` (OpenAI
                // shape). The reasoning is kept either way (it was dropped on a call until
                // 2026-09-26 while the streamed path sent it).
                match has_tools
                    .then(|| crate::chat::parse_tool_calls(&text, &tool_names))
                    .flatten()
                {
                    Some((tool_calls, lead)) => (
                        ChatCompletionMessage {
                            reasoning_content: qwen_reasoning,
                            role: "assistant",
                            content: if lead.is_empty() { None } else { Some(lead) },
                            tool_calls: Some(tool_calls),
                        },
                        "tool_calls".to_string(),
                    ),
                    None => (
                        ChatCompletionMessage {
                            reasoning_content: qwen_reasoning,
                            role: "assistant",
                            content: Some(text),
                            tool_calls: None,
                        },
                        finish.as_str().to_string(),
                    ),
                }
            }
        };
        Json(ChatCompletion {
            id: gen_id("chatcmpl"),
            object: "chat.completion",
            created: now(),
            model: model_id,
            choices: vec![ChatChoice {
                index: 0,
                message,
                finish_reason,
                logprobs: logprobs_out,
            }],
            // `ids` is the raw sampled stream, counted before the thought-channel strip and
            // tool-call parse rewrite the text: usage reports what the model GENERATED, which
            // is the quantity that consumed context and time.
            usage: Usage {
                prompt_tokens,
                completion_tokens: ids.len(),
                total_tokens: prompt_tokens + ids.len(),
            },
        })
        .into_response()
    }
}

/// `POST /v1/completions`. Raw prompt in, text out; same streaming/non-streaming
/// split as chat.
async fn completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    let prompt_ids = match encode_prompt(&state.tokenizer, &req.prompt) {
        Ok(ids) => ids,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, &format!("tokenize: {e}")),
    };
    // Captured before the move into `submit_job`, as in the chat handler.
    let prompt_tokens = prompt_ids.len();
    let max_tokens = match context_budget(prompt_tokens, req.max_tokens, state.max_tokens_cap) {
        Ok(n) => n,
        Err(msg) => return context_length_error(&msg, "prompt"),
    };
    let model_id = req.model.clone().unwrap_or_else(|| state.model_id.clone());
    let stops = match StopStrings::from_openai(req.stop.as_ref()) {
        Ok(s) => s,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, &e),
    };

    let sampling = match build_sampling(
        req.temperature,
        req.top_p,
        req.top_k,
        req.seed,
        req.repetition_penalty,
    ) {
        Ok(s) => s,
        Err((status, msg)) => return error_response(status, &msg),
    };
    let want_logprobs = req.logprobs;
    let top_logprobs_n = req.top_logprobs.unwrap_or(0).min(20);
    let response_format_type = req
        .response_format
        .as_ref()
        .map(|rf| rf.format_type.clone());
    let mut rx = match submit_job(
        &state,
        prompt_ids,
        max_tokens,
        sampling,
        want_logprobs,
        top_logprobs_n,
        response_format_type,
        Vec::new(),
        Vec::new(),
        None,
        // A raw completion has no chat structure: no system prefix to anchor.
        Default::default(),
        &stops,
    ) {
        Ok(rx) => rx,
        Err((status, msg)) => return error_response(status, &msg),
    };

    if req.stream {
        text_stream_response(state, model_id, rx, want_logprobs, stops)
    } else {
        let mut ids = Vec::new();
        let mut lp_data: Vec<Box<crate::actor::TokenLogprob>> = Vec::new();
        let mut finish = FinishReason::Length;
        while let Some(item) = rx.recv().await {
            match item {
                StreamItem::Token(id, lp) => {
                    ids.push(id);
                    if let Some(lp) = lp {
                        lp_data.push(lp);
                    }
                }
                StreamItem::Done(reason) => {
                    finish = reason;
                    break;
                }
            }
        }
        let text = decode(&state.tokenizer, &ids);
        let (text, matched) = stops.cut(&text);
        if matched.is_some() {
            finish = FinishReason::Stop;
        }
        let text = text.to_string();
        let logprobs_out = if want_logprobs {
            Some(build_choice_logprobs(&state.tokenizer, &lp_data))
        } else {
            None
        };
        Json(TextCompletion {
            id: gen_id("cmpl"),
            object: "text_completion",
            created: now(),
            model: model_id,
            choices: vec![TextChoice {
                index: 0,
                text,
                finish_reason: finish.as_str().to_string(),
                logprobs: logprobs_out,
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens: ids.len(),
                total_tokens: prompt_tokens + ids.len(),
            },
        })
        .into_response()
    }
}

// ----------------------------------------------------------------------------
// Shared handler helpers.
// ----------------------------------------------------------------------------

/// Build backend-agnostic sampling params from the OpenAI request fields.
/// `max_tokens`/`stop_tokens` are filled in by `submit_job`. Invalid params
/// (negative temperature, `top_p` out of `[0,1]`, `top_k == 0`) surface as a
/// `400` via `SamplingParams::validate`.
/// A per-request seed for clients that did not send one: wall-clock nanoseconds mixed with a
/// process-wide counter (two requests in the same nanosecond still differ), through splitmix64.
fn fresh_seed() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    let mut z = t ^ N
        .fetch_add(1, Ordering::Relaxed)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The model's own sampling defaults (GGUF `general.sampling.top_k` / `top_p`), used when a SAMPLED
/// request omits them. Set once at startup (`main.rs`).
static MODEL_SAMPLING: std::sync::OnceLock<(Option<usize>, Option<f32>)> =
    std::sync::OnceLock::new();

pub fn set_model_sampling_defaults(top_k: Option<usize>, top_p: Option<f32>) {
    let _ = MODEL_SAMPLING.set((top_k, top_p));
}

fn build_sampling(
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    seed: Option<u64>,
    repetition_penalty: Option<f32>,
) -> Result<SamplingParams, (StatusCode, String)> {
    let temp = temperature.unwrap_or(0.0);
    let stochastic = temp > 0.0;
    // THE MODEL'S OWN DEFAULTS (2026-09-26) for what a sampled request leaves out: top_k and
    // top_p from the GGUF's `general.sampling.*` (Qwen3.8: 20 / 0.95), and NO repetition
    // penalty (the model's config names none; another engine and HF apply none). The temperature default
    // stays 0 (greedy). `ARF_LEGACY_SAMPLING_DEFAULTS=1` restores the previous defaults (A/B).
    let legacy = std::env::var_os("ARF_LEGACY_SAMPLING_DEFAULTS").is_some();
    let (model_k, model_p) = if legacy {
        (None, None)
    } else {
        MODEL_SAMPLING.get().copied().unwrap_or((None, None))
    };
    let params = SamplingParams {
        temperature: temp,
        top_k: top_k.or(if stochastic { model_k } else { None }),
        // When sampling stochastically, default to nucleus top_p=0.95 if the client sent
        // none — full-vocab sampling at temp≈0.7 lets low-probability/off-distribution
        // tokens through (the source of the `_`-mangled tokens). Greedy (temp=0) untouched.
        top_p: top_p.or(if stochastic {
            Some(model_p.unwrap_or(0.95))
        } else {
            None
        }),
        // Unseeded = a FRESH seed per request. Defaulting to 0 made every unseeded sampled
        // request identical — temperature 1.0 returned the same text three times running
        // (2026-09-24). An explicit `seed` still reproduces exactly.
        seed: seed.unwrap_or_else(fresh_seed),
        // 1.0 = off. Clients send ~1.1; when stochastic and none given, default to 1.1 so a
        // misconfigured client can't fall into a runaway repetition loop to the token cap.
        // 2026-09-26: the default is now the model's (none). MEASURED: the implicit 1.1 cost
        // speculative sampling ~0.5 accepted drafts a window (1.19 vs 1.70 of 7 — it lowers the
        // very tokens the draft proposes, and the drafter never sees it), and it compounds per
        // occurrence (unlike HF / llama.cpp). A client that sends a penalty still gets it; the
        // runaway-loop guard is top_k/top_p plus the token cap. Legacy: 1.1.
        repetition_penalty: repetition_penalty.unwrap_or(if stochastic && legacy {
            1.1
        } else {
            1.0
        }),
        // Sliding window (llama.cpp's repeat_last_n=64): penalize only recent tokens, so a
        // long generation doesn't crush legitimately-common tokens by penalizing all history.
        repeat_last_n: 64,
        ..SamplingParams::default()
    };
    params
        .validate()
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("sampling: {e}")))?;
    Ok(params)
}

/// Qwen3.8 images (and video frame pairs) of a tokenized prompt -> the request's `ImagePrompt`
/// (`None` when there are none). Locates each image's pad run, then runs the encoder on a blocking thread — seconds per
/// image on the CPU, a GPU wait otherwise, neither of which may stall the async runtime. Also
/// used by `/v1/messages`.
pub(crate) async fn qwen_encode(
    state: &AppState,
    images: &[crate::qwen_image::QwenMedia],
    prompt_ids: &[u32],
) -> Result<Option<arf_core::scheduler::ImagePrompt>, String> {
    if images.is_empty() {
        return Ok(None);
    }
    let Some(enc) = state.qwen_vision.clone() else {
        return Err("no Qwen vision encoder is loaded".into());
    };
    let pad = crate::qwen_image::pad_id(&state.tokenizer)?;
    let video_pad = crate::qwen_image::video_pad_id(&state.tokenizer);
    let grids = crate::qwen_image::locate(prompt_ids, pad, video_pad, images)?;
    let images = images.to_vec();
    let n = prompt_ids.len();
    tokio::task::spawn_blocking(move || {
        crate::qwen_image::build_image_prompt(&enc, &images, &grids, n)
    })
    .await
    .map_err(|e| format!("vision encode task: {e}"))?
    .map(Some)
}

/// Qwen3-Omni audio of a tokenized prompt -> the request's `ImagePrompt` (M3): find each clip's
/// `<|audio_pad|>` run, then run the encoder on a blocking thread (a GPU wait, or seconds of CPU).
pub(crate) async fn audio_encode(
    state: &AppState,
    staged: crate::qwen_audio::StagedAudio,
    prompt_ids: &[u32],
) -> Result<arf_core::scheduler::ImagePrompt, String> {
    let Some(tower) = state.qwen_audio.clone() else {
        return Err("no audio encoder is loaded".into());
    };
    let pad = crate::qwen_audio::pad_id(&state.tokenizer)?;
    let starts = crate::qwen_audio::locate(prompt_ids, pad, &staged.tokens)?;
    tokio::task::spawn_blocking(move || {
        crate::qwen_audio::build_audio_prompt(&tower, &staged, &starts)
    })
    .await
    .map_err(|e| format!("audio encode task: {e}"))?
}

/// Fold the per-request limits into the params, assign the request id, and
/// queue the job. The channel's capacity (256) is the per-request backlog a
/// slow client gets before eviction .
#[allow(clippy::too_many_arguments)]
fn submit_job(
    state: &AppState,
    prompt_ids: Vec<u32>,
    max_tokens: usize,
    mut params: SamplingParams,
    want_logprobs: bool,
    top_logprobs: usize,
    response_format: Option<String>,
    images: Vec<(Vec<u8>, usize, usize)>,
    image_positions: Vec<usize>,
    image_prompt: Option<arf_core::scheduler::ImagePrompt>,
    anchors: crate::prefix_anchor::Anchors,
    stops: &StopStrings,
) -> Result<tokio::sync::mpsc::Receiver<StreamItem>, (StatusCode, String)> {
    params.max_tokens = max_tokens;
    params.stop_tokens = (*state.stop_tokens).clone();
    let (token_tx, token_rx) = tokio::sync::mpsc::channel(256);
    let job = Job {
        id: state.next_id.fetch_add(1, Ordering::Relaxed),
        prompt_ids,
        params,
        token_tx,
        want_logprobs,
        top_logprobs,
        response_format,
        images,
        image_positions,
        image_prompt,
        prefix_anchor: anchors.system,
        tools_anchor: anchors.tools,
        header_tail: anchors.header,
        background: BACKGROUND.try_with(|b| *b).unwrap_or(false),
        stop: stops.check(state.tokenizer.clone()),
    };
    state
        .model
        .submit(job)
        .map_err(|e| (StatusCode::SERVICE_UNAVAILABLE, format!("model: {e}")))?;
    Ok(token_rx)
}

/// SSE response for chat: a leading `delta.role` chunk, one `delta.content` chunk
/// per token, a final chunk carrying `finish_reason`, the usage chunk when the request asked
/// for it (`stream_options.include_usage`), then `data: [DONE]`.
///
/// With stop strings a token's chunk carries what the [`StopFilter`] released — the
/// text a stop string could still begin with is held back, so a chunk may be empty — and the
/// final chunk carries whatever was held when the reply ended without one.
#[allow(clippy::too_many_arguments)]
fn chat_stream_response(
    tok: Arc<Tokenizer>,
    model_id: String,
    rx: tokio::sync::mpsc::Receiver<StreamItem>,
    want_logprobs: bool,
    thinking: bool,
    include_usage: bool,
    prompt_tokens: usize,
    stops: StopStrings,
) -> Response {
    let id = gen_id("chatcmpl");
    let created = now();
    // Tokens seen by the chunk mapper below, read by the usage chunk chained after it (the chain
    // only polls that once this stream has ended, so the count is final).
    let generated = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let generated_in_map = Arc::clone(&generated);
    let (usage_id, usage_model) = (id.clone(), model_id.clone());

    // First chunk announces the assistant role (OpenAI convention).
    let role_chunk = ChatCompletionChunk {
        id: id.clone(),
        object: "chat.completion.chunk",
        created,
        model: model_id.clone(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta: ChatDelta {
                role: Some("assistant"),
                content: None,
                reasoning_content: None,
            },
            finish_reason: None,
            logprobs: None,
        }],
    };
    let first = Event::default().data(serde_json::to_string(&role_chunk).unwrap_or_default());

    // Qwen3.8's own template pre-opened `<think>`: tokens are reasoning until `</think>`, then
    // the answer (leading whitespace after the close tag dropped, as the non-streamed path trims).
    let think_end = tok.token_to_id("</think>");
    let mut in_think = thinking && think_end.is_some();
    let mut answer_started = false;
    let mut stop = StopFilter::new(stops);
    let stream = ReceiverStream::new(rx).map(move |item| {
        let ev = match item {
            StreamItem::Token(token_id, lp) => {
                generated_in_map.fetch_add(1, Ordering::Relaxed);
                let raw = decode(&tok, &[token_id]);
                let part = ThinkPart::of(token_id, think_end, &mut in_think);
                let (reasoning, text) = think_delta(stop.push(&raw, part), &mut answer_started);
                let logprobs_chunk = if want_logprobs {
                    lp.as_deref().map(|lp_data| ChunkLogprobs {
                        content: vec![build_chunk_logprob_entry(&tok, lp_data)],
                    })
                } else {
                    None
                };
                let chunk = ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: model_id.clone(),
                    choices: vec![ChatChunkChoice {
                        index: 0,
                        delta: ChatDelta {
                            role: None,
                            content: (!text.is_empty()).then_some(text),
                            reasoning_content: reasoning,
                        },
                        finish_reason: None,
                        logprobs: logprobs_chunk,
                    }],
                };
                Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
            }
            StreamItem::Done(reason) => {
                // What the stop strings held back, when none matched (nothing otherwise).
                let (reasoning, text) = think_delta(stop.finish(), &mut answer_started);
                let reason = if stop.matched().is_some() {
                    FinishReason::Stop
                } else {
                    reason
                };
                let chunk = ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: model_id.clone(),
                    choices: vec![ChatChunkChoice {
                        index: 0,
                        delta: ChatDelta {
                            role: None,
                            content: (!text.is_empty()).then_some(text),
                            reasoning_content: reasoning.filter(|r| !r.is_empty()),
                        },
                        finish_reason: Some(reason.as_str().to_string()),
                        logprobs: None,
                    }],
                };
                Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
            }
        };
        Ok::<Event, std::convert::Infallible>(ev)
    });

    // Emit the role chunk first, then the token/finish chunks, the usage chunk (if asked),
    // then `[DONE]`.
    let usage = tokio_stream::once(()).filter_map(move |()| {
        include_usage.then(|| {
            let n = generated.load(Ordering::Relaxed);
            Ok(Event::default().data(usage_chunk_json(
                &usage_id,
                created,
                &usage_model,
                prompt_tokens,
                n,
            )))
        })
    });
    let done = tokio_stream::once(Ok(Event::default().data("[DONE]")));
    let full = tokio_stream::once(Ok(first))
        .chain(stream)
        .chain(usage)
        .chain(done);
    Sse::new(full)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// How a streamed tool request's reply is read.
struct ToolStream {
    /// The template's format (`None` = the fenced-JSON fallback, whose answer stays buffered).
    format: ToolFormat,
    /// The prompt pre-opened `<think>`.
    thinking: bool,
    /// The offered tools (the XML parser types parameters by their schemas).
    tools: Vec<Tool>,
    include_usage: bool,
    prompt_tokens: usize,
    /// The request's stop strings.
    stops: StopStrings,
    /// A forced call's opening, already in the prompt ([`enforce_tool_choice`]): fed to the
    /// splitter ahead of the reply, so the call reads whole. Empty for every other request.
    prefill: String,
}

/// Which part of a reply that opens inside `<think>` a token's text belongs to — the tag its
/// text carries through the [`StopFilter`], so held-back reasoning is still reasoning when a
/// later token releases it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum ThinkPart {
    Reasoning,
    /// The `</think>` token itself: matched against, never sent.
    ThinkEnd,
    Answer,
}

impl ThinkPart {
    /// The part `token_id` belongs to; `in_think` flips at `</think>`.
    pub(crate) fn of(token_id: u32, think_end: Option<u32>, in_think: &mut bool) -> Self {
        if !*in_think {
            ThinkPart::Answer
        } else if Some(token_id) == think_end {
            *in_think = false;
            ThinkPart::ThinkEnd
        } else {
            ThinkPart::Reasoning
        }
    }
}

/// Released pieces as one chat delta: the reasoning (`Some` when any reasoning piece came, even
/// empty — a reasoning token's chunk always carried `reasoning_content`) and the answer text,
/// its leading whitespace dropped until the answer has begun.
fn think_delta(
    pieces: Vec<(String, ThinkPart)>,
    answer_started: &mut bool,
) -> (Option<String>, String) {
    let mut reasoning: Option<String> = None;
    let mut answer = String::new();
    for (text, part) in pieces {
        match part {
            ThinkPart::Reasoning => reasoning.get_or_insert_with(String::new).push_str(&text),
            ThinkPart::ThinkEnd => {}
            ThinkPart::Answer => {
                let text = if *answer_started {
                    &text
                } else {
                    text.trim_start()
                };
                *answer_started |= !text.is_empty();
                answer.push_str(text);
            }
        }
    }
    (reasoning, answer)
}

/// Token ids → text, one token at a time, without the two failures of decoding each id alone:
/// a multi-byte character split across tokens arriving as U+FFFD, and a
/// tokenizer that drops a word's leading space when it is first in the sequence. It decodes a
/// window that starts one emitted chunk back and sends only what that window adds — the
/// prefix/read-offset scheme of HF text-generation-inference's `decode_token`. Held while the
/// window ends in U+FFFD (an incomplete character); `finish` flushes whatever is left.
#[derive(Default)]
struct IncrementalDecoder {
    ids: Vec<u32>,
    prefix: usize,
    read: usize,
}

impl IncrementalDecoder {
    fn push(&mut self, tok: &Tokenizer, id: u32) -> String {
        self.ids.push(id);
        let before = decode(tok, &self.ids[self.prefix..self.read]);
        let now = decode(tok, &self.ids[self.prefix..]);
        if now.len() > before.len() && !now.ends_with('\u{FFFD}') {
            if let Some(delta) = now
                .strip_prefix(before.as_str())
                .or_else(|| now.get(before.len()..))
            {
                let delta = delta.to_string();
                self.prefix = self.read;
                self.read = self.ids.len();
                return delta;
            }
        }
        String::new()
    }

    fn finish(&mut self, tok: &Tokenizer) -> String {
        if self.read >= self.ids.len() {
            return String::new();
        }
        let before = decode(tok, &self.ids[self.prefix..self.read]);
        let now = decode(tok, &self.ids[self.prefix..]);
        self.read = self.ids.len();
        now.strip_prefix(before.as_str())
            .or_else(|| now.get(before.len()..))
            .unwrap_or("")
            .to_string()
    }
}

/// SSE response for a chat request that offered `tools`.
///
/// Reasoning streams live as `reasoning_content` up to `</think>`, and the answer's prose streams
/// live as `content` up to the start of a tool call in the template's format (`<tool_call>`);
/// from there the text is buffered, parsed at the end, and sent as `tool_calls` deltas — one per
/// call: `index`, `id`, `type`, `function.name`, `function.arguments` — then `finish_reason`
/// `tool_calls`. On the fenced-JSON fallback the answer is buffered whole (a fenced call is only
/// recognisable once its JSON closes) while reasoning still streams. Then the usage chunk when
/// asked for, and `[DONE]`.
///
/// History: until 2026-09-21 this streamed raw tokens (the JSON call arrived as prose, never a
/// `tool_calls` delta); until 2026-09-26 it buffered the WHOLE reply — reasoning included — so an
/// agent turn was silent for its full length, and it drained the job to the end even after the
/// client had gone (the GPU kept decoding an aborted turn to `max_tokens`).
fn chat_stream_tools_response(
    tok: Arc<Tokenizer>,
    model_id: String,
    mut rx: tokio::sync::mpsc::Receiver<StreamItem>,
    cfg: ToolStream,
) -> Response {
    let (tx, out) = tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(64);
    tokio::spawn(async move {
        let id = gen_id("chatcmpl");
        let created = now();
        let chunk = |delta: serde_json::Value, finish: Option<&str>| {
            let body = serde_json::json!({
                "id": id, "object": "chat.completion.chunk", "created": created, "model": model_id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            });
            Ok(Event::default().data(body.to_string()))
        };
        let piece = |p: Piece| match p {
            Piece::Reasoning(t) => chunk(serde_json::json!({"reasoning_content": t}), None),
            Piece::Content(t) => chunk(serde_json::json!({"content": t}), None),
        };
        if tx
            .send(chunk(serde_json::json!({"role": "assistant"}), None))
            .await
            .is_err()
        {
            return;
        }
        let native = cfg.format != ToolFormat::None;
        let mut splitter = ReplySplitter::new(cfg.format, cfg.thinking, native);
        // A forced call's opening goes to the splitter only, never through the stop filter: it
        // was never generated (a stop string cannot match inside it — the job's own check reads
        // generated tokens only) and the splitter buffers it as the call it begins.
        for p in splitter.push(&cfg.prefill) {
            if tx.send(piece(p)).await.is_err() {
                return;
            }
        }
        let mut dec = IncrementalDecoder::default();
        let mut stop = StopFilter::new(cfg.stops.clone());
        let mut generated = 0usize;
        let mut finish = FinishReason::Length;
        loop {
            let item = tokio::select! {
                // The client went away. Returning drops `rx`, so the actor's next `try_send`
                // fails and it evicts the sequence  — the same path a dropped plain
                // stream takes when axum drops its body.
                _ = tx.closed() => return,
                item = rx.recv() => item,
            };
            match item {
                Some(StreamItem::Token(token_id, _)) => {
                    generated += 1;
                    let text = text_of(stop.push(&dec.push(&tok, token_id), ()));
                    for p in splitter.push(&text) {
                        if tx.send(piece(p)).await.is_err() {
                            return;
                        }
                    }
                }
                Some(StreamItem::Done(reason)) => {
                    finish = reason;
                    break;
                }
                None => break,
            }
        }
        let mut events = Vec::with_capacity(8);
        let mut rest = text_of(stop.push(&dec.finish(&tok), ()));
        rest += &text_of(stop.finish());
        if stop.matched().is_some() {
            finish = FinishReason::Stop;
        }
        let tail = splitter.push(&rest);
        let end = splitter.finish();
        for p in tail.into_iter().chain(end.pieces) {
            events.push(piece(p));
        }
        let (calls, extra) =
            settle_tool_stream(cfg.format, &end.gap, &end.buffered, &cfg.tools, finish);
        if !extra.is_empty() {
            events.push(chunk(serde_json::json!({"content": extra}), None));
        }
        let finish_reason = if calls.is_empty() {
            finish.as_str()
        } else {
            for (index, c) in calls.iter().enumerate() {
                events.push(chunk(
                    serde_json::json!({"tool_calls": [{
                        "index": index, "id": c.id, "type": c.r#type,
                        "function": {"name": c.function.name, "arguments": c.function.arguments},
                    }]}),
                    None,
                ));
            }
            "tool_calls"
        };
        events.push(chunk(serde_json::json!({}), Some(finish_reason)));
        if cfg.include_usage {
            events.push(Ok(Event::default().data(usage_chunk_json(
                &id,
                created,
                &model_id,
                cfg.prompt_tokens,
                generated,
            ))));
        }
        events.push(Ok(Event::default().data("[DONE]")));
        for ev in events {
            if tx.send(ev).await.is_err() {
                return;
            }
        }
    });
    Sse::new(ReceiverStream::new(out))
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// The end of a streamed tool reply, from what [`ReplySplitter::finish`] held back (`gap`) and
/// buffered: the complete calls, and the text still owed after what was streamed. One decision
/// for both tool streams — this one and `/v1/messages`' (`anthropic::stream_tool_blocks`) — so
/// the two dialects never disagree on what is a call.
///
/// What the buffered text is: calls (and the prose before them), or — when nothing
/// parses, or the budget cut the call off — text that must still reach the client. The
/// native `content` carries its own leading whitespace (the gap the splitter held back);
/// the fenced answer was never streamed, so nothing precedes it.
fn settle_tool_stream(
    format: ToolFormat,
    gap: &str,
    buffered: &str,
    tools: &[Tool],
    finish: FinishReason,
) -> (Vec<ToolCall>, String) {
    if format != ToolFormat::None {
        let settled = settle_native(format, gap, buffered, tools, finish != FinishReason::Stop);
        log_end_note(settled.note);
        (settled.calls, settled.content)
    } else {
        let names: Vec<&str> = tools.iter().map(|t| t.function.name.as_str()).collect();
        let answer = crate::chat::strip_thought(buffered);
        let (_reasoning, answer) = crate::chat::split_muse_channels(&answer);
        match crate::chat::parse_tool_calls(&answer, &names) {
            Some((calls, lead)) => (calls, lead),
            None => (Vec::new(), answer),
        }
    }
}

/// SSE response for raw completions: one `text` chunk per token, a final chunk
/// with `finish_reason`, then `data: [DONE]`.
fn text_stream_response(
    state: AppState,
    model_id: String,
    rx: tokio::sync::mpsc::Receiver<StreamItem>,
    _want_logprobs: bool,
    stops: StopStrings,
) -> Response {
    let id = gen_id("cmpl");
    let created = now();
    let tok = state.tokenizer.clone();
    // Stop strings: a chunk carries what the filter released; the final chunk what
    // it still held when the reply ended without one.
    let mut stop = StopFilter::new(stops);

    let stream = ReceiverStream::new(rx).map(move |item| {
        let ev = match item {
            StreamItem::Token(token_id, _lp) => {
                let text = text_of(stop.push(&decode(&tok, &[token_id]), ()));
                let chunk = TextCompletionChunk {
                    id: id.clone(),
                    object: "text_completion",
                    created,
                    model: model_id.clone(),
                    choices: vec![TextChunkChoice {
                        index: 0,
                        text,
                        finish_reason: None,
                    }],
                };
                Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
            }
            StreamItem::Done(reason) => {
                let text = text_of(stop.finish());
                let reason = if stop.matched().is_some() {
                    FinishReason::Stop
                } else {
                    reason
                };
                let chunk = TextCompletionChunk {
                    id: id.clone(),
                    object: "text_completion",
                    created,
                    model: model_id.clone(),
                    choices: vec![TextChunkChoice {
                        index: 0,
                        text,
                        finish_reason: Some(reason.as_str().to_string()),
                    }],
                };
                Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
            }
        };
        Ok::<Event, std::convert::Infallible>(ev)
    });

    let done = tokio_stream::once(Ok(Event::default().data("[DONE]")));
    let full = stream.chain(done);
    Sse::new(full)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Decode token ids to text, skipping special tokens (best-effort: a decode error
/// yields an empty string rather than failing the whole stream).
fn decode(tok: &Tokenizer, ids: &[u32]) -> String {
    tok.decode(ids, true).unwrap_or_default()
}

/// Seconds since the Unix epoch (the OpenAI `created` field).
fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// A unique response id (`<prefix>-<nanos>-<seq>`). The nanos give rough
/// ordering; a process-lifetime atomic counter guarantees uniqueness even for
/// requests that enter in the same nanosecond.
fn gen_id(prefix: &str) -> String {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}-{nanos}-{seq}")
}

/// Build a JSON error response in the OpenAI `{ "error": { "message": .., "type": .. } }`
/// shape with the given status. `type` follows OpenAI's split (a 4xx is the request's fault,
/// anything else the server's); it was missing until 2026-09-26.
fn error_response(status: StatusCode, message: &str) -> Response {
    let kind = if status.is_client_error() {
        "invalid_request_error"
    } else {
        "server_error"
    };
    let body = serde_json::json!({ "error": { "message": message, "type": kind } });
    (status, Json(body)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(body: serde_json::Value) -> ChatRequest {
        serde_json::from_value(body).expect("a valid chat request")
    }

    /// `video_url` (vLLM) and `video` (qwen-vl-utils: a URL or a frame list) parts keep their
    /// place among the images and in the text; an image-only message has no `videos` at all.
    #[test]
    fn video_parts_parse_into_their_image_slots() {
        use crate::qwen_video::{VideoInput, VideoSource};
        let r = req(
            serde_json::json!({"messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                {"type": "text", "text": "Compare"},
                {"type": "video_url", "video_url": {"url": "file:///tmp/a.mp4", "fps": 1.0}},
                {"type": "video", "video": ["data:image/png;base64,AA==", "data:image/png;base64,BB=="],
                 "fps": 4.0},
                {"type": "video", "video": "data:video/mp4;base64,AAAA", "max_frames": 16},
                {"type": "text", "text": "please."}
            ]}]}),
        );
        let m = &r.messages[0];
        assert_eq!(m.content, "Compare\nplease.");
        assert_eq!(m.images.len(), 4);
        assert_eq!(m.image_at, vec![0, 7, 7, 7]);
        assert!(m.video_at(0).is_none());
        assert_eq!(
            m.video_at(1),
            Some(&VideoInput {
                source: VideoSource::Url("file:///tmp/a.mp4".into()),
                fps: Some(1.0),
                max_frames: None
            })
        );
        assert_eq!(
            m.video_at(2).map(|v| &v.source),
            Some(&VideoSource::Frames(vec![
                "data:image/png;base64,AA==".into(),
                "data:image/png;base64,BB==".into()
            ]))
        );
        assert_eq!(m.video_at(2).and_then(|v| v.fps), Some(4.0));
        assert_eq!(m.video_at(3).and_then(|v| v.max_frames), Some(16));
        assert_eq!(m.images[1], "video:file:///tmp/a.mp4");
        assert_eq!(m.images[3], "video:data-url");
        let plain = req(
            serde_json::json!({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            {"type": "text", "text": "hi"}]}]}),
        );
        assert!(plain.messages[0].videos.is_empty());
    }

    /// `input_audio` (OpenAI: bare base64 + a format hint, kept as a data URL) and `audio_url`
    /// (vLLM) parts keep their offsets in the text; a request with neither has no audio at all,
    /// and with an audio encoder loaded an audio + tools request stays off the native template
    /// (whose render reads only `content`).
    #[test]
    fn audio_parts_parse_with_their_offsets() {
        let r = req(
            serde_json::json!({"messages": [{"role": "user", "content": [
                {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}},
                {"type": "text", "text": "What was asked?"},
                {"type": "audio_url", "audio_url": {"url": "data:audio/mp3;base64,SUQz"}},
                {"type": "input_audio", "input_audio": {"data": "data:audio/flac;base64,ZkxhQw=="}}
            ]}]}),
        );
        let m = &r.messages[0];
        assert_eq!(m.content, "What was asked?");
        assert_eq!(
            m.audios,
            vec![
                "data:audio/wav;base64,UklGRg==",
                "data:audio/mp3;base64,SUQz",
                "data:audio/flac;base64,ZkxhQw=="
            ]
        );
        assert_eq!(m.audio_at, vec![0, 15, 15]);
        assert!(m.images.is_empty());
        let plain = req(serde_json::json!({"messages": [{"role": "user", "content": "hi"}]}));
        assert!(plain.messages[0].audios.is_empty());

        let with_tools = serde_json::json!({"messages": [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}}]}],
            "tools": [{"type": "function", "function": {"name": "read",
                "parameters": {"type": "object"}}}]});
        let r = req(with_tools.clone());
        let bytes = serde_json::to_vec(&with_tools).unwrap();
        let tpl = crate::jinja_chat::ChatTemplate::new(
            "{{ messages[0].content }}<function=x><parameter=y>".into(),
            None,
        )
        .unwrap();
        assert!(matches!(
            plan_chat(
                &r,
                &bytes,
                Some(&tpl),
                true,
                &template_vars(None, None, false, None),
                true
            ),
            Plan::HandWritten(PromptPath::Fenced)
        ));
        assert!(matches!(
            plan_chat(
                &r,
                &bytes,
                Some(&tpl),
                false,
                &template_vars(None, None, false, None),
                true
            ),
            Plan::Native { .. }
        ));
    }

    /// A byte-level tokenizer with a fixed word-level vocabulary — enough to drive the streaming
    /// path without a model: `Ġ` is a space, `Ċ` a newline, and `Ã` + `©` are the two bytes of
    /// `é` (each alone decodes to U+FFFD, as a real BPE split does).
    pub(super) fn test_tokenizer() -> Tokenizer {
        let vocab: Vec<&str> = vec![
            "We",
            "Ġneed",
            "Ġto",
            "Ġread.",
            "Ċ",
            "</think>",
            "ĊĊ",
            "Hi",
            "<tool_call>",
            "<function=read>",
            "<parameter=path>",
            "calc.py",
            "</parameter>",
            "</function>",
            "</tool_call>",
            "Ã",
            "©",
            "```json",
            "{\"function\":Ġ\"read\",Ġ\"arguments\":Ġ{\"path\":Ġ\"a.py\"}}",
            "```",
            "[UNK]",
            // 21..=23 (2026-09-26): an `edit(path, oldText, newText)` call in qwen-xml, whose
            // argument order is NOT sorted order — the key-order tests. Appended, so no id above
            // moves.
            "<function=edit>",
            "<parameter=oldText>",
            "<parameter=newText>",
        ];
        let map: serde_json::Map<String, serde_json::Value> = vocab
            .iter()
            .enumerate()
            .map(|(i, t)| (t.to_string(), serde_json::json!(i)))
            .collect();
        let json = serde_json::json!({
            "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
            "normalizer": null, "pre_tokenizer": null, "post_processor": null,
            "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
            "model": {"type": "WordLevel", "vocab": map, "unk_token": "[UNK]"},
        });
        let path = std::env::temp_dir().join(format!(
            "arf_serve_test_tokenizer_{}_{}.json",
            std::process::id(),
            new_call_id_suffix()
        ));
        std::fs::write(&path, json.to_string()).unwrap();
        let tok = Tokenizer::from_file(&path).expect("test tokenizer.json parses");
        let _ = std::fs::remove_file(&path);
        tok
    }

    /// A unique suffix so concurrently running tests never share a temp file.
    fn new_call_id_suffix() -> String {
        crate::tool_parse::new_call_id()
    }

    /// The SSE body as the `data:` payloads, in order.
    pub(super) async fn sse_data(resp: Response) -> Vec<String> {
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        String::from_utf8(body.to_vec())
            .unwrap()
            .lines()
            .filter_map(|l| l.strip_prefix("data: ").map(str::to_string))
            .collect()
    }

    /// Plain chat — no tools, no tool history — must reach the hand-written templates exactly as
    /// before this change: the new message fields stay empty and the rendered bytes are the ones
    /// those templates always produced (the goldens are their documented shapes).
    #[test]
    fn plain_chat_keeps_the_old_path_byte_for_byte() {
        let r = req(serde_json::json!({"messages": [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
            {"role": "assistant", "content": "Hello."},
            {"role": "user", "content": "Why?"},
        ]}));
        assert!(r.tools.is_none());
        assert!(!r.messages.iter().any(ChatMessage::is_tool_turn));
        assert!(r.messages.iter().all(|m| m.tool_calls.is_empty()
            && m.tool_call_id.is_none()
            && m.name.is_none()
            && m.reasoning_content.is_none()));
        assert_eq!(
            crate::chat::qwen3_think_template(&r.messages, Some("medium")),
            "<|im_start|>system\nBe terse.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n\
<|im_start|>assistant\nHello.<|im_end|>\n<|im_start|>user\nWhy?<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
        assert_eq!(
            crate::chat::chatml_template(&r.messages),
            "<|im_start|>system\nBe terse.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n\
<|im_start|>assistant\nHello.<|im_end|>\n<|im_start|>user\nWhy?<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(
            crate::chat::gemma4_template(&r.messages),
            "<|turn>system\nBe terse.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\nHello.<turn|>\n\
<|turn>user\nWhy?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
        );
    }

    /// The OpenAI tool history now survives deserialization: `tool_calls` (arguments as the
    /// spec's JSON string, or an object from a lenient client, or null), `content: null`,
    /// `tool_call_id`, `name`, `reasoning_content`.
    #[test]
    fn tool_history_deserializes() {
        let r = req(serde_json::json!({"messages": [
            {"role": "user", "content": "fix it"},
            {"role": "assistant", "content": null, "reasoning_content": "look first",
             "tool_calls": [
                {"id": "call_a", "type": "function", "function": {"name": "read", "arguments": "{\"path\":\"calc.py\"}"}},
                {"id": "call_b", "type": "function", "function": {"name": "ls", "arguments": {"dir": "."}}},
                {"id": "call_c", "function": {"name": "status", "arguments": null}}]},
            {"role": "tool", "tool_call_id": "call_a", "name": "read", "content": "def add(a, b): ..."},
            {"role": "assistant", "content": "ok", "tool_calls": null},
        ], "max_completion_tokens": 64, "stream_options": {"include_usage": true}}));
        let a = &r.messages[1];
        assert!(a.is_tool_turn());
        assert_eq!(a.content, "");
        assert_eq!(a.reasoning_content.as_deref(), Some("look first"));
        assert_eq!(a.tool_calls.len(), 3);
        assert_eq!(a.tool_calls[0].function.arguments, "{\"path\":\"calc.py\"}");
        assert_eq!(a.tool_calls[1].function.arguments, "{\"dir\":\".\"}");
        assert_eq!(a.tool_calls[2].function.arguments, "");
        assert_eq!(a.tool_calls[2].r#type, "function");
        let t = &r.messages[2];
        assert!(t.is_tool_turn());
        assert_eq!(t.tool_call_id.as_deref(), Some("call_a"));
        assert_eq!(t.name.as_deref(), Some("read"));
        assert!(!r.messages[3].is_tool_turn());
        assert_eq!(r.max_completion_tokens, Some(64));
        assert!(r.stream_options.unwrap().include_usage);
    }

    /// Claude Code's billing header reaches `/v1/chat/completions` inside a system message when a
    /// gateway translated its Anthropic request: as its own content part, or joined into one
    /// string. Either way the handler's request no longer carries it — two requests whose headers
    /// differ read the same — a header-only system message is gone, and a user message is never
    /// touched.
    #[test]
    fn chat_request_drops_claude_codes_billing_header() {
        let header = |h: &str| {
            format!("x-anthropic-billing-header: cc_version=2.1.283.{h}; cc_entrypoint=cli;")
        };
        let system = "You are Claude Code.\nBe brief.";
        let read = |v: serde_json::Value| chat_request(&serde_json::to_vec(&v).unwrap()).unwrap();
        for h in ["e08", "4aa"] {
            for sys in [
                serde_json::json!([{"type": "text", "text": header(h)}, {"type": "text", "text": system}]),
                serde_json::json!(format!("{}\n\n{system}", header(h))),
            ] {
                for role in ["system", "developer"] {
                    let r = read(serde_json::json!({"messages": [
                        {"role": role, "content": sys},
                        {"role": "user", "content": "hi"}]}));
                    assert_eq!(r.messages.len(), 2);
                    assert_eq!(r.messages[0].role, role);
                    assert_eq!(r.messages[0].content, system, "{sys}");
                }
            }
        }
        let only = read(serde_json::json!({"messages": [
            {"role": "system", "content": header("e08")},
            {"role": "user", "content": header("e08")}]}));
        assert_eq!(only.messages.len(), 1);
        assert_eq!(only.messages[0].role, "user");
        assert_eq!(only.messages[0].content, header("e08"));
        // No header: the text is exactly as sent, whitespace included.
        let plain = read(serde_json::json!({"messages": [
            {"role": "system", "content": " keep me \n"}, {"role": "user", "content": "hi"}]}));
        assert_eq!(plain.messages[0].content, " keep me \n");
    }

    /// The fallback path's history: one merged system turn, each call replayed as the fenced JSON
    /// the model writes (parseable by the same parser), results as named user turns, parallel
    /// results in one turn.
    #[test]
    fn legacy_path_renders_tool_history_readably() {
        let r = req(serde_json::json!({
        "tools": [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}],
        "messages": [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "developer", "content": "Be careful."},
            {"role": "user", "content": "fix it"},
            {"role": "assistant", "content": "Reading both.", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{\"path\":\"a.py\"}"}},
                {"id": "c2", "type": "function", "function": {"name": "read", "arguments": "{\"path\":\"b.py\"}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "A"},
            {"role": "tool", "tool_call_id": "c2", "content": "B"},
        ]}));
        let m = crate::chat::legacy_tool_messages(&r.messages, r.tools.as_deref());
        let roles: Vec<&str> = m.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, vec!["system", "user", "assistant", "user"]);
        assert!(m[0].content.starts_with("You can call functions"));
        assert!(m[0]
            .content
            .ends_with("\n\nYou are a coding agent.\n\nBe careful."));
        assert!(m[2].content.starts_with("Reading both.\n```json\n"));
        let (calls, lead) = crate::chat::parse_tool_calls(&m[2].content, &["read"]).unwrap();
        assert_eq!(lead, "Reading both.");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1].function.arguments, "{\"path\":\"b.py\"}");
        assert_eq!(
            m[3].content,
            "Function `read` returned:\nA\n\nFunction `read` returned:\nB"
        );
    }

    #[test]
    fn usage_chunk_has_empty_choices_and_the_counts() {
        let v: serde_json::Value =
            serde_json::from_str(&usage_chunk_json("chatcmpl-1", 7, "m", 100, 23)).unwrap();
        assert_eq!(v["object"], "chat.completion.chunk");
        assert_eq!(v["choices"], serde_json::json!([]));
        assert_eq!(
            v["usage"],
            serde_json::json!({"prompt_tokens": 100, "completion_tokens": 23, "total_tokens": 123})
        );
    }

    #[test]
    fn context_budget_refuses_a_full_prompt_and_clamps_the_rest() {
        assert!(context_budget(100, None, 100).is_err());
        assert!(context_budget(101, Some(1), 100).is_err());
        assert_eq!(context_budget(90, None, 100), Ok(10)); // absent: everything left
        assert_eq!(context_budget(90, Some(4096), 100), Ok(10)); // clamped to what is left
        assert_eq!(context_budget(10, Some(5), 100), Ok(5));
        assert_eq!(context_budget(10, Some(0), 100), Ok(1));
    }

    #[tokio::test]
    async fn context_length_error_is_openai_shaped() {
        let msg = context_budget(70_000, None, 65_536).unwrap_err();
        let resp = context_length_error(&msg, "messages");
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["code"], "context_length_exceeded");
        let m = v["error"]["message"].as_str().unwrap();
        assert!(m.contains("65536") && m.contains("70000"), "{m}");
        assert_eq!(v["error"]["param"], "messages");
    }

    #[test]
    fn effort_maps_onto_what_the_template_accepts() {
        assert_eq!(template_effort(None), None);
        assert_eq!(template_effort(Some("minimal")).as_deref(), Some("low"));
        assert_eq!(template_effort(Some("high")).as_deref(), Some("xhigh"));
        assert_eq!(template_effort(Some("medium")).as_deref(), Some("medium"));
        assert_eq!(template_effort(Some("turbo")), None); // the template would raise
        assert_eq!(template_effort(Some("none")), None); // rendered as enable_thinking=false
    }

    /// Non-streamed native reply: reasoning, prose, and every call — the same split the stream
    /// makes. A marker with text after it that is not a call: the wrapper line is dropped and the
    /// text kept, after the whitespace it followed; a bare marker is dropped whole. A call the
    /// budget cut off (`FinishReason::Length`) is not a call: its text is the content.
    #[test]
    fn native_reply_non_streamed() {
        let text = "Need to read.\n</think>\n\nReading it.\n\n<tool_call>\n<function=read>\n\
<parameter=path>\ncalc.py\n</parameter>\n</function>\n</tool_call>";
        let r = read_native_reply(text, ToolFormat::QwenXml, true, &[], FinishReason::Stop);
        assert_eq!(r.reasoning.as_deref(), Some("Need to read."));
        assert_eq!(r.content, "Reading it.");
        let calls = r.calls.expect("a call");
        assert_eq!(calls[0].function.arguments, "{\"path\":\"calc.py\"}");
        let broken = read_native_reply(
            "Done.\n<tool_call>\noops",
            ToolFormat::QwenXml,
            false,
            &[],
            FinishReason::Stop,
        );
        assert!(broken.calls.is_none());
        assert_eq!(broken.content, "Done.\noops");
        let bare = read_native_reply(
            "All fixed.\n\n<tool_call>\n",
            ToolFormat::QwenXml,
            false,
            &[],
            FinishReason::Stop,
        );
        assert!(bare.calls.is_none());
        assert_eq!(bare.content, "All fixed.");
        let cut = "Writing it.\n<tool_call>\n<function=write>\n<parameter=content>\ndef add(a, b):";
        let r = read_native_reply(cut, ToolFormat::QwenXml, false, &[], FinishReason::Length);
        assert!(r.calls.is_none(), "a cut call must never be sent as a call");
        assert_eq!(r.content, cut);
        // The same text on EOS: a sloppy model, parsed leniently.
        let r = read_native_reply(cut, ToolFormat::QwenXml, false, &[], FinishReason::Stop);
        assert_eq!(r.calls.map(|c| c.len()), Some(1));
    }

    /// A character split across two tokens is held, not sent as U+FFFD, and the chunks
    /// concatenate to the one-shot decode.
    #[test]
    fn incremental_decoder_holds_split_characters() {
        let tok = test_tokenizer();
        let mut d = IncrementalDecoder::default();
        let ids = [7u32, 15, 16, 1, 2];
        let chunks: Vec<String> = ids.iter().map(|&i| d.push(&tok, i)).collect();
        assert_eq!(chunks, vec!["Hi", "", "é", " need", " to"]);
        assert_eq!(chunks.concat(), decode(&tok, &ids));
        assert_eq!(d.finish(&tok), "");
        // Cut off mid-character: `finish` flushes what is left.
        let mut d = IncrementalDecoder::default();
        assert_eq!(d.push(&tok, 15), "");
        assert_eq!(d.finish(&tok), "\u{FFFD}");
    }

    /// The streamed tool request end to end, fed token ids the way the actor sends them: role
    /// chunk, reasoning deltas, prose deltas (the split `é` intact, no partial `<tool_call>`),
    /// one `tool_calls` delta, finish `tool_calls`, the usage chunk, `[DONE]`.
    #[tokio::test]
    async fn tools_stream_end_to_end() {
        let tok = Arc::new(test_tokenizer());
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let ids = [
            0u32, 1, 2, 3, 4, 5, 6, 7, 15, 16, 4, 8, 4, 9, 4, 10, 4, 11, 4, 12, 4, 13, 4, 14,
        ];
        for id in ids {
            tx.send(StreamItem::Token(id, None)).await.unwrap();
        }
        tx.send(StreamItem::Done(FinishReason::Stop)).await.unwrap();
        drop(tx);
        let cfg = ToolStream {
            format: ToolFormat::QwenXml,
            thinking: true,
            tools: Vec::new(),
            include_usage: true,
            prompt_tokens: 7,
            stops: StopStrings::default(),
            prefill: String::new(),
        };
        let data = sse_data(chat_stream_tools_response(tok, "m".into(), rx, cfg)).await;
        assert_eq!(data.last().map(String::as_str), Some("[DONE]"));
        let chunks: Vec<serde_json::Value> = data[..data.len() - 1]
            .iter()
            .map(|d| serde_json::from_str(d).unwrap())
            .collect();
        assert_eq!(
            chunks[0]["choices"][0]["delta"],
            serde_json::json!({"role": "assistant"})
        );
        let (mut reasoning, mut content) = (String::new(), String::new());
        let mut calls = Vec::new();
        let mut finish = None;
        for c in &chunks[1..chunks.len() - 1] {
            let d = &c["choices"][0]["delta"];
            if let Some(t) = d["reasoning_content"].as_str() {
                reasoning.push_str(t);
            }
            if let Some(t) = d["content"].as_str() {
                assert!(!t.contains('<') && !t.contains('\u{FFFD}'), "leaked: {t:?}");
                content.push_str(t);
            }
            if let Some(tc) = d["tool_calls"].as_array() {
                calls.extend(tc.iter().cloned());
            }
            if let Some(f) = c["choices"][0]["finish_reason"].as_str() {
                finish = Some(f.to_string());
            }
        }
        assert_eq!(reasoning, "We need to read.");
        assert_eq!(content, "Hié");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["index"], 0);
        assert_eq!(calls[0]["type"], "function");
        assert!(calls[0]["id"].as_str().unwrap().starts_with("call_"));
        assert_eq!(calls[0]["function"]["name"], "read");
        assert_eq!(calls[0]["function"]["arguments"], "{\"path\":\"calc.py\"}");
        assert_eq!(finish.as_deref(), Some("tool_calls"));
        let usage = chunks.last().unwrap();
        assert_eq!(usage["choices"], serde_json::json!([]));
        assert_eq!(
            usage["usage"],
            serde_json::json!({"prompt_tokens": 7, "completion_tokens": ids.len(), "total_tokens": 7 + ids.len()})
        );
    }

    /// The fenced-JSON fallback on the stream: the answer is buffered and parsed at the end.
    #[tokio::test]
    async fn tools_stream_fenced_fallback() {
        let tok = Arc::new(test_tokenizer());
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        for id in [17u32, 4, 18, 4, 19] {
            tx.send(StreamItem::Token(id, None)).await.unwrap();
        }
        tx.send(StreamItem::Done(FinishReason::Stop)).await.unwrap();
        drop(tx);
        let tools = vec![Tool {
            r#type: "function".into(),
            function: ToolFunction {
                name: "read".into(),
                description: None,
                parameters: None,
            },
        }];
        let cfg = ToolStream {
            format: ToolFormat::None,
            thinking: false,
            tools,
            include_usage: false,
            prompt_tokens: 3,
            stops: StopStrings::default(),
            prefill: String::new(),
        };
        let data = sse_data(chat_stream_tools_response(tok, "m".into(), rx, cfg)).await;
        let joined = data.join("\n");
        assert!(joined.contains("\"tool_calls\""), "{joined}");
        assert!(joined.contains("{\\\"path\\\":\\\"a.py\\\"}"), "{joined}");
        assert!(
            joined.contains("\"finish_reason\":\"tool_calls\""),
            "{joined}"
        );
        assert!(
            !joined.contains("\"usage\""),
            "usage was not asked for: {joined}"
        );
    }

    /// Cancellation: when the client goes away MID-TURN (after the role chunk, while the task
    /// waits for the next token), the task stops reading and drops the job's receiver — which is
    /// what makes the actor evict the sequence
    /// (`tests/actor_loop.rs::dropped_client_is_evicted_and_others_complete`). The version
    /// before 2026-09-26 fails this: it sat in `rx.recv()` until the job ended on its own, so an
    /// aborted agent turn kept the GPU decoding to `max_tokens`.
    #[tokio::test]
    async fn tools_stream_drops_the_job_when_the_client_leaves() {
        let tok = Arc::new(test_tokenizer());
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        tx.send(StreamItem::Token(0, None)).await.unwrap();
        let cfg = ToolStream {
            format: ToolFormat::QwenXml,
            thinking: true,
            tools: Vec::new(),
            include_usage: true,
            prompt_tokens: 1,
            stops: StopStrings::default(),
            prefill: String::new(),
        };
        let resp = chat_stream_tools_response(tok, "m".into(), rx, cfg);
        let mut body = resp.into_body().into_data_stream();
        let first = body.next().await.expect("the role chunk").unwrap();
        assert!(String::from_utf8_lossy(&first).contains("\"role\":\"assistant\""));
        let second = body
            .next()
            .await
            .expect("the first reasoning delta")
            .unwrap();
        assert!(String::from_utf8_lossy(&second).contains("\"reasoning_content\":\"We\""));
        // The task now waits on the job (which never finishes on its own). The client leaves.
        drop(body);
        tokio::time::timeout(std::time::Duration::from_secs(5), tx.closed())
            .await
            .expect("the stream task must drop the job's receiver once the client is gone");
    }

    // ------------------------------------------------------------------------
    // Dispatch: which prompt path a request takes (`plan_chat`), and its thinking settings.
    // ------------------------------------------------------------------------

    const QWEN38: &str = include_str!("../testdata/qwen38_chat_template.jinja");

    pub(super) fn qwen38() -> crate::jinja_chat::ChatTemplate {
        crate::jinja_chat::ChatTemplate::new(QWEN38.into(), Some("<|im_end|>".into())).unwrap()
    }

    /// `<tool_call>\n<function=edit>\n<parameter=path>\ncalc.py\n</parameter>\n<parameter=oldText>
    /// \nWe\n</parameter>\n<parameter=newText>\nHi\n</parameter>\n</function>\n</tool_call>` — the
    /// model writes `path, oldText, newText`; sorted, that is `newText, oldText, path`.
    pub(super) const XML_EDIT: [u32; 25] = [
        8, 4, 21, 4, 10, 4, 11, 4, 12, 4, 22, 4, 0, 4, 12, 4, 23, 4, 7, 4, 12, 4, 13, 4, 14,
    ];

    /// [`XML_EDIT`]'s arguments, byte for byte, in the model's order.
    pub(super) const EDIT_ARGS: &str = r#"{"path":"calc.py","oldText":"We","newText":"Hi"}"#;

    /// `/v1/chat/completions`, non-streamed: `tool_calls[].function.arguments` is a JSON STRING,
    /// so the parser's text — in the model's key order — goes out verbatim, with no object in
    /// between to re-sort it. Confirmed end to end on 2026-09-26, when `/v1/messages` was found
    /// to sort the same arguments (`anthropic::ReplyBlock`).
    #[tokio::test]
    async fn chat_completions_non_streamed_arguments_keep_the_models_key_order() {
        let (state, jobs) = app_state(Some(qwen38()));
        // Non-thinking, so the reply is answer from its first token whatever the shell's defaults.
        let body = r#"{"max_tokens": 256, "chat_template_kwargs": {"enable_thinking": false},
            "tools": [{"type": "function", "function": {"name": "edit", "parameters":
                {"type": "object", "properties": {"path": {"type": "string"},
                 "oldText": {"type": "string"}, "newText": {"type": "string"}}}}}],
            "messages": [{"role": "user", "content": "fix calc.py"}]}"#;
        let actor = std::thread::spawn(move || {
            let job = jobs.recv().unwrap();
            for id in [7u32, 4].into_iter().chain(XML_EDIT) {
                job.token_tx
                    .blocking_send(StreamItem::Token(id, None))
                    .unwrap();
            }
            job.token_tx
                .blocking_send(StreamItem::Done(FinishReason::Stop))
                .unwrap();
        });
        let resp = chat_completions(State(state), axum::body::Bytes::from(body)).await;
        actor.join().unwrap();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["choices"][0]["finish_reason"], "tool_calls", "{v}");
        let call = &v["choices"][0]["message"]["tool_calls"][0]["function"];
        assert_eq!(call["name"], "edit");
        assert_eq!(call["arguments"], EDIT_ARGS, "{v}");
    }

    /// `x-arf-background: 1` makes the queued job background work; without it, it is not.
    #[tokio::test]
    async fn the_background_header_reaches_the_job() {
        use axum::body::Body;
        use axum::http::Request;
        for (header, want) in [(Some("1"), true), (Some("0"), false), (None, false)] {
            let (state, jobs) = app_state(None);
            let actor = std::thread::spawn(move || {
                let job = jobs.recv().unwrap();
                job.token_tx
                    .blocking_send(StreamItem::Done(FinishReason::Length))
                    .unwrap();
                job.background
            });
            let mut req = Request::post("/v1/messages").header("content-type", "application/json");
            if let Some(v) = header {
                req = req.header("x-arf-background", v);
            }
            let body = r#"{"model": "test", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}"#;
            let resp = tower_service_call(router(state), req.body(Body::from(body)).unwrap()).await;
            assert_eq!(resp.status(), StatusCode::OK);
            assert_eq!(actor.join().unwrap(), want, "header {header:?}");
        }
    }

    /// Play the actor for one queued job the way the scheduler and `demux` treat a stop string
    ///: each token is appended and the job's own stop check asked; the token that
    /// completes a stop string is sent and ends the job as `Stop`. Otherwise every id is sent and
    /// the job ends on its budget. Returns how many tokens were sent.
    pub(super) fn play_with_stops(
        jobs: std::sync::mpsc::Receiver<Job>,
        ids: Vec<u32>,
    ) -> std::thread::JoinHandle<usize> {
        std::thread::spawn(move || {
            let job = jobs.recv().unwrap();
            let mut out = Vec::new();
            for id in ids {
                out.push(id);
                job.token_tx
                    .blocking_send(StreamItem::Token(id, None))
                    .unwrap();
                if job.stop.as_ref().is_some_and(|c| c.hit(&out)) {
                    job.token_tx
                        .blocking_send(StreamItem::Done(FinishReason::Stop))
                        .unwrap();
                    return out.len();
                }
            }
            job.token_tx
                .blocking_send(StreamItem::Done(FinishReason::Length))
                .unwrap();
            out.len()
        })
    }

    /// The scheduler-side check fires on the token that completes a stop string and on no other:
    /// not on half of a split character (U+FFFD), and not again for a match already passed.
    #[test]
    fn stop_check_fires_on_the_completing_token_only() {
        let tok = Arc::new(test_tokenizer());
        assert!(StopStrings::default().check(tok.clone()).is_none());
        // "é" is the two tokens `Ã` `©`.
        let c = StopStrings::new(["\u{e9}".to_string()])
            .check(tok.clone())
            .unwrap();
        assert!(!c.hit(&[7, 15]));
        assert!(c.hit(&[7, 15, 16]));
        // "d to" split across " need" and " to".
        let c = StopStrings::new(["d to".to_string()]).check(tok).unwrap();
        assert!(!c.hit(&[0]));
        assert!(!c.hit(&[0, 1]));
        assert!(c.hit(&[0, 1, 2]));
        assert!(!c.hit(&[0, 1, 2, 3]));
    }

    async fn body_json(resp: Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    /// Stop strings (a), non-streamed chat: "." sits inside the token " read."; the job ends on that
    /// token, the reply is cut before the stop string, and `finish_reason` is `stop`. The tokens
    /// the model would have generated next are never asked for.
    #[tokio::test]
    async fn chat_stop_string_inside_one_token_non_streamed() {
        let (state, jobs) = app_state(None);
        let body = r#"{"max_tokens": 64, "stop": ".",
            "messages": [{"role": "user", "content": "go"}]}"#;
        let actor = play_with_stops(jobs, vec![0, 1, 2, 3, 4, 7]);
        let resp = chat_completions(State(state), axum::body::Bytes::from(body)).await;
        assert_eq!(
            actor.join().unwrap(),
            4,
            "the job ends at the completing token"
        );
        let v = body_json(resp).await;
        assert_eq!(
            v["choices"][0]["message"]["content"], "We need to read",
            "{v}"
        );
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert_eq!(v["usage"]["completion_tokens"], 4);
    }

    /// Stop strings (b) and (d), streamed chat: "d to" is split across " need" and " to". No byte of
    /// it is ever sent, and the finish chunk says `stop`. Without a match, what was held back is
    /// sent with the finish chunk and the reply is whole.
    #[tokio::test]
    async fn chat_stop_string_split_across_tokens_streamed() {
        async fn stream(stop: &str, ids: Vec<u32>) -> (String, String) {
            let (state, jobs) = app_state(None);
            let body = format!(
                r#"{{"max_tokens": 64, "stream": true, "stop": {stop},
                "messages": [{{"role": "user", "content": "go"}}]}}"#
            );
            let actor = play_with_stops(jobs, ids);
            let resp = chat_completions(State(state), axum::body::Bytes::from(body)).await;
            let data = sse_data(resp).await;
            actor.join().unwrap();
            assert_eq!(data.last().map(String::as_str), Some("[DONE]"));
            let chunks: Vec<serde_json::Value> = data[..data.len() - 1]
                .iter()
                .map(|d| serde_json::from_str(d).unwrap())
                .collect();
            let text: String = chunks
                .iter()
                .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
                .collect();
            let finish = chunks.last().unwrap()["choices"][0]["finish_reason"].clone();
            (text, finish.as_str().unwrap_or_default().to_string())
        }
        let (text, finish) = stream(r#"["d to"]"#, vec![0, 1, 2, 3]).await;
        assert_eq!((text.as_str(), finish.as_str()), ("We nee", "stop"));
        let (text, finish) = stream(r#"["XYZ", "zz"]"#, vec![0, 1, 2, 3, 4, 7]).await;
        assert_eq!(
            (text.as_str(), finish.as_str()),
            ("We need to read.\nHi", "length")
        );
    }

    /// Stop strings on `/v1/completions`: the cut and `finish_reason` non-streamed and streamed, a
    /// reply that never matches flushed whole, and more than four stop strings refused.
    #[tokio::test]
    async fn completions_enforce_stop() {
        async fn run(body: serde_json::Value, ids: Vec<u32>) -> Response {
            let (state, jobs) = app_state(None);
            let req: CompletionRequest = serde_json::from_value(body).unwrap();
            let actor = play_with_stops(jobs, ids);
            let resp = completions(State(state), Json(req)).await;
            let resp = if resp.status() == StatusCode::OK {
                resp
            } else {
                return resp;
            };
            // Read the body before joining: a stream is fed while it is read.
            let (parts, body) = resp.into_parts();
            let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
            actor.join().unwrap();
            Response::from_parts(parts, axum::body::Body::from(bytes))
        }
        let ids = vec![0, 1, 2, 3, 4, 7];
        let v = body_json(
            run(
                serde_json::json!({"prompt": "go", "max_tokens": 64, "stop": ["\n"]}),
                ids.clone(),
            )
            .await,
        )
        .await;
        assert_eq!(v["choices"][0]["text"], "We need to read.", "{v}");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        for (stop, want, finish) in [
            ("\n", "We need to read.", "stop"),
            ("nope", "We need to read.\nHi", "length"),
        ] {
            let resp = run(
                serde_json::json!({"prompt": "go", "max_tokens": 64, "stream": true, "stop": stop}),
                ids.clone(),
            )
            .await;
            let data = sse_data(resp).await;
            let chunks: Vec<serde_json::Value> = data[..data.len() - 1]
                .iter()
                .map(|d| serde_json::from_str(d).unwrap())
                .collect();
            let text: String = chunks
                .iter()
                .filter_map(|c| c["choices"][0]["text"].as_str())
                .collect();
            assert_eq!(text, want);
            assert_eq!(
                chunks.last().unwrap()["choices"][0]["finish_reason"],
                finish
            );
        }
        let (state, _jobs) = app_state(None);
        let req: CompletionRequest = serde_json::from_value(
            serde_json::json!({"prompt": "go", "stop": ["a", "b", "c", "d", "e"]}),
        )
        .unwrap();
        let resp = completions(State(state), Json(req)).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// A greedy-only model's generation responses carry `x-arf-warning`; other routes and other
    /// models do not.
    #[tokio::test]
    async fn a_greedy_only_model_says_so_in_a_header() {
        use axum::body::Body;
        use axum::http::Request;
        let ask = |why: Option<&'static str>, path: &'static str| async move {
            let (mut state, _jobs) = app_state(None);
            state.greedy_only = why;
            let req = Request::get(path).body(Body::empty()).unwrap();
            let next = axum::Router::new()
                .route(path, axum::routing::get(|| async { "ok" }))
                .layer(axum::middleware::from_fn_with_state(
                    state.greedy_only,
                    greedy_only_warning,
                ));
            let resp = tower_service_call(next, req).await;
            resp.headers()
                .get("x-arf-warning")
                .map(|v| v.to_str().unwrap().to_string())
        };
        let w = ask(Some("island MoE"), "/v1/chat/completions")
            .await
            .unwrap();
        assert!(w.contains("temperature") && w.contains("island MoE"), "{w}");
        assert!(ask(Some("island MoE"), "/v1/models").await.is_none());
        assert!(ask(None, "/v1/chat/completions").await.is_none());
    }

    /// One request through a router, without a listener.
    async fn tower_service_call(
        router: axum::Router,
        req: axum::http::Request<axum::body::Body>,
    ) -> Response {
        use tower_service::Service;
        let mut svc = router.into_service::<axum::body::Body>();
        std::future::poll_fn(|cx| Service::poll_ready(&mut svc, cx))
            .await
            .unwrap();
        svc.call(req).await.unwrap()
    }

    /// The route table builds. The HTTP library rejects a pattern it does not accept by PANICKING
    /// when the router is built, which is at server start: with its 0.8, `/conversations/:id`
    /// (the 0.7 capture syntax) stopped every start, and nothing in the test suite built the
    /// router, so the dependency update was green everywhere (2026-10-07).
    #[test]
    fn the_route_table_builds() {
        let (state, _jobs) = app_state(None);
        let _ = router(state);
    }

    /// FLUX is looked for where every other model is: `ARF_FLUX_DIR` wins, then the models
    /// folder (`ARF_MODELS_DIR`, `./models` when it exists, `~/.arf/models`).
    #[test]
    fn flux_is_found_in_the_models_folder() {
        use std::path::PathBuf;
        let os = |s: &str| Some(std::ffi::OsString::from(s));
        assert_eq!(
            flux_dir_from(os("/x/flux"), os("/m"), true, os("/h")),
            PathBuf::from("/x/flux")
        );
        assert_eq!(
            flux_dir_from(None, os("/m"), true, os("/h")),
            PathBuf::from("/m/flux-schnell")
        );
        assert_eq!(
            flux_dir_from(None, None, true, os("/h")),
            PathBuf::from("./models/flux-schnell")
        );
        // An installed `arf`, started anywhere: the folder `arf pull` uses.
        assert_eq!(
            flux_dir_from(None, None, false, os("/h")),
            PathBuf::from("/h/.arf/models/flux-schnell")
        );
        assert_eq!(
            flux_dir_from(None, None, false, None),
            PathBuf::from("./models/flux-schnell")
        );
    }

    /// An image sent to a server with no vision encoder is refused, not answered from the text.
    #[tokio::test]
    async fn an_image_without_a_vision_encoder_is_refused() {
        let (state, _jobs) = app_state(None);
        let body = serde_json::to_vec(
            &serde_json::json!({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            {"type": "text", "text": "What colour is this?"}]}]}),
        )
        .unwrap();
        let resp = chat_completions(State(state), axum::body::Bytes::from(body)).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(resp.into_body(), 1 << 16)
            .await
            .unwrap();
        assert!(String::from_utf8_lossy(&bytes).contains("serving text only"));
    }

    /// An `AppState` that drives the handler without a model: the test takes each queued job off
    /// the returned receiver and plays the actor's part (feeds the job's `token_tx`). The test
    /// tokenizer encodes any prompt; its hand-written template (the generic fallback) opens no
    /// `<think>`. (In `anthropic::tests` until 2026-09-26; here so `/v1/chat/completions` is
    /// driven the same way.)
    pub(super) fn app_state(
        tpl: Option<crate::jinja_chat::ChatTemplate>,
    ) -> (AppState, std::sync::mpsc::Receiver<Job>) {
        let (jobs_tx, jobs) = std::sync::mpsc::channel();
        let state = AppState {
            greedy_only: None,
            model: ModelHandle::from_sender(jobs_tx),
            tokenizer: Arc::new(test_tokenizer()),
            model_id: "test".into(),
            stop_tokens: Arc::new(Vec::new()),
            max_tokens_cap: 4096,
            chat_template: tpl.map(Arc::new),
            next_id: Arc::new(AtomicU64::new(0)),
            metrics: Arc::new(Metrics::default()),
            vision_enabled: false,
            vision_tokens: 0,
            qwen_vision: None,
            qwen_audio: None,
            prefix_anchors: Arc::new(Default::default()),
            systemone: Default::default(),
            #[cfg(feature = "postgres")]
            db: None,
        };
        (state, jobs)
    }

    /// `/v1/models` carries the context budget under both names, and capabilities as loaded.
    #[test]
    fn models_list_carries_the_context_budget() {
        let (state, _jobs) = app_state(None);
        let v = serde_json::to_value(models_response(&state)).unwrap();
        let m = &v["data"][0];
        assert_eq!(m["id"], "test");
        assert_eq!(m["max_model_len"], 4096);
        assert_eq!(m["context_length"], 4096);
        assert_eq!(m["capabilities"]["vision"], false);
        assert_eq!(m["capabilities"]["audio"], false);
        assert_eq!(m["capabilities"]["tools"], false);
    }

    fn default_vars() -> crate::jinja_chat::TemplateVars {
        template_vars(None, None, false, None)
    }

    /// `plan_chat` over the body a client would send (the native path reads `tools` from it).
    fn plan(
        body: serde_json::Value,
        tpl: Option<&crate::jinja_chat::ChatTemplate>,
        vision: bool,
    ) -> Plan {
        let r = req(body.clone());
        let bytes = serde_json::to_vec(&body).unwrap();
        let offer = !offered_tools(&r).is_empty();
        plan_chat(&r, &bytes, tpl, vision, &default_vars(), offer)
    }

    fn read_tool() -> serde_json::Value {
        serde_json::json!([{"type": "function", "function": {"name": "read",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}])
    }

    fn tool_history() -> serde_json::Value {
        serde_json::json!([
            {"role": "user", "content": "fix it"},
            {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function",
                "function": {"name": "read", "arguments": "{\"path\":\"a.py\"}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "A"},
        ])
    }

    #[test]
    fn plan_chat_plain_request_takes_the_hand_written_path() {
        let body = serde_json::json!({"messages": [{"role": "user", "content": "hi"}]});
        assert!(matches!(
            plan(body, Some(&qwen38()), false),
            Plan::HandWritten(PromptPath::Plain)
        ));
    }

    /// Tools → the model's own template, the tools in the CLIENT's key order (read from the raw
    /// body; `serde_json::Value` would sort them); `thinking` is read off the rendered prompt
    /// (its cue pre-opens `<think>` at the template's default).
    #[test]
    fn plan_chat_tools_render_natively_and_thinking_comes_from_the_prompt() {
        let body = r#"{"messages": [{"role": "user", "content": "read a.py"}],
            "tools": [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]}"#;
        let r: ChatRequest = serde_json::from_str(body).unwrap();
        match plan_chat(
            &r,
            body.as_bytes(),
            Some(&qwen38()),
            false,
            &default_vars(),
            true,
        ) {
            Plan::Native {
                prompt,
                format,
                thinking,
            } => {
                assert_eq!(format, ToolFormat::QwenXml);
                assert!(
                    prompt.contains("<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"read\", \"parameters\": {\"type\": \"object\"}}}\n</tools>"),
                    "{prompt}"
                );
                assert!(
                    prompt.ends_with("<|im_start|>assistant\n<think>\n"),
                    "{prompt}"
                );
                assert!(thinking);
            }
            other => panic!("expected the native path, got {other:?}"),
        }
        // enable_thinking=false pre-closes the block: not thinking, read off the same prompt.
        let r = req(
            serde_json::json!({"messages": [{"role": "user", "content": "x"}],
            "tools": read_tool()}),
        );
        let bytes = serde_json::to_vec(&serde_json::json!({"tools": read_tool()})).unwrap();
        let off = template_vars(Some("none"), None, false, None);
        match plan_chat(&r, &bytes, Some(&qwen38()), false, &off, true) {
            Plan::Native {
                thinking, prompt, ..
            } => {
                assert!(!thinking, "{prompt}");
                assert!(prompt.ends_with("<think>\n\n</think>\n\n"), "{prompt}");
            }
            other => panic!("expected the native path, got {other:?}"),
        }
    }

    /// Tool history without tools still renders natively (so the model sees its own calls), with
    /// no `<tools>` block.
    #[test]
    fn plan_chat_history_only_renders_natively_without_tools() {
        let body = serde_json::json!({"messages": tool_history()});
        match plan(body, Some(&qwen38()), false) {
            Plan::Native { prompt, .. } => {
                assert!(!prompt.contains("<tools>"), "{prompt}");
                assert!(prompt.contains("<tool_call>\n<function=read>"), "{prompt}");
                assert!(
                    prompt.contains("<tool_response>\nA\n</tool_response>"),
                    "{prompt}"
                );
            }
            other => panic!("expected the native path, got {other:?}"),
        }
    }

    /// `tool_choice: "none"`: tools neither rendered nor parsed. With no history that is plain
    /// chat; with history the native render carries no `<tools>` block. Other choices keep the
    /// tools on offer (`required` and a named function were not enforced until 2026-10-04; they
    /// are now prefilled after the render — `chat_completions_required_tool_choice_is_prefilled`).
    #[test]
    fn plan_chat_tool_choice_none_withdraws_the_tools() {
        let none = serde_json::json!({"messages": [{"role": "user", "content": "hi"}],
            "tools": read_tool(), "tool_choice": "none"});
        assert!(offered_tools(&req(none.clone())).is_empty());
        assert!(matches!(
            plan(none, Some(&qwen38()), false),
            Plan::HandWritten(PromptPath::Plain)
        ));
        let with_history = serde_json::json!({"messages": tool_history(),
            "tools": read_tool(), "tool_choice": "none"});
        match plan(with_history, Some(&qwen38()), false) {
            Plan::Native { prompt, .. } => assert!(!prompt.contains("<tools>"), "{prompt}"),
            other => panic!("expected the native path, got {other:?}"),
        }
        for choice in [
            serde_json::json!("auto"),
            serde_json::json!("required"),
            serde_json::json!({"type": "function", "function": {"name": "read"}}),
        ] {
            let r = req(
                serde_json::json!({"messages": [{"role": "user", "content": "x"}],
                "tools": read_tool(), "tool_choice": choice}),
            );
            assert_eq!(offered_tools(&r).len(), 1);
        }
    }

    /// No template, a template that teaches no format, a template that RAISES, and images with
    /// vision on: the fenced-JSON protocol. Images with vision off do not block the native path.
    #[test]
    fn plan_chat_falls_back_to_the_fenced_protocol() {
        let tools = serde_json::json!({"messages": [{"role": "user", "content": "x"}],
            "tools": read_tool()});
        assert!(matches!(
            plan(tools.clone(), None, false),
            Plan::HandWritten(PromptPath::Fenced)
        ));
        let plain_tpl =
            crate::jinja_chat::ChatTemplate::new("{{ messages[0].content }}".into(), None).unwrap();
        assert!(matches!(
            plan(tools.clone(), Some(&plain_tpl), false),
            Plan::HandWritten(PromptPath::Fenced)
        ));
        let raises = crate::jinja_chat::ChatTemplate::new(
            "{{ raise_exception('no') }}<function=x><parameter=y>".into(),
            None,
        )
        .unwrap();
        assert_eq!(raises.tool_format(), ToolFormat::QwenXml);
        assert!(matches!(
            plan(tools, Some(&raises), false),
            Plan::HandWritten(PromptPath::Fenced)
        ));
        let image = serde_json::json!({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}],
            "tools": read_tool()});
        assert!(matches!(
            plan(image.clone(), Some(&qwen38()), true),
            Plan::HandWritten(PromptPath::Fenced)
        ));
        assert!(matches!(
            plan(image, Some(&qwen38()), false),
            Plan::Native { .. }
        ));
    }

    /// One effort for both prompt paths; the template's `enable_thinking` follows the operator's
    /// switch unless the request sets it; `chat_template_kwargs` reach the template.
    #[test]
    fn thinking_settings_resolve_once_for_both_paths() {
        let r = |body: serde_json::Value| {
            let mut b = body;
            b["messages"] = serde_json::json!([{"role": "user", "content": "x"}]);
            req(b)
        };
        assert_eq!(request_effort(&r(serde_json::json!({})), false), None);
        assert_eq!(
            request_effort(&r(serde_json::json!({"reasoning_effort": "low"})), true),
            Some("none")
        );
        let kw = r(serde_json::json!({"reasoning_effort": "low",
            "chat_template_kwargs": {"reasoning_effort": "medium", "preserve_thinking": true, "other": 1}}));
        assert_eq!(request_effort(&kw, false), Some("medium"));
        let off = r(serde_json::json!({"reasoning_effort": "low",
            "chat_template_kwargs": {"enable_thinking": false}}));
        assert_eq!(request_effort(&off, false), Some("none"));

        // Nothing set: the template's own defaults.
        let v = template_vars(None, None, false, None);
        assert_eq!(
            (v.enable_thinking, v.reasoning_effort, v.preserve_thinking),
            (None, None, None)
        );
        // The operator turned the thinking template off: tool requests do not think either.
        assert_eq!(
            template_vars(None, None, true, None).enable_thinking,
            Some(false)
        );
        // "none" (JSON mode, or the client) always wins.
        let on = ChatTemplateKwargs {
            enable_thinking: Some(true),
            ..Default::default()
        };
        assert_eq!(
            template_vars(Some("none"), Some(&on), false, None).enable_thinking,
            Some(false)
        );
        // An explicit request beats the operator default.
        assert_eq!(
            template_vars(None, Some(&on), true, None).enable_thinking,
            Some(true)
        );
        // The server default effort, and preserve_thinking passed through.
        let v = template_vars(
            None,
            kw.chat_template_kwargs.as_ref(),
            false,
            Some("high".into()),
        );
        assert_eq!(v.reasoning_effort.as_deref(), Some("xhigh"));
        assert_eq!(v.preserve_thinking, Some(true));
        assert_eq!(
            template_vars(None, None, false, Some("none".into())).enable_thinking,
            Some(false)
        );
    }

    /// The plain (no tools) stream, channel-driven: role, reasoning up to `</think>`, content,
    /// then the finish chunk, THEN the usage chunk with the generated count, THEN `[DONE]`.
    #[tokio::test]
    async fn plain_stream_sends_finish_then_usage_then_done() {
        let tok = Arc::new(test_tokenizer());
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let ids = [0u32, 1, 5, 6, 7];
        for id in ids {
            tx.send(StreamItem::Token(id, None)).await.unwrap();
        }
        tx.send(StreamItem::Done(FinishReason::Stop)).await.unwrap();
        drop(tx);
        let data = sse_data(chat_stream_response(
            tok,
            "m".into(),
            rx,
            false,
            true,
            true,
            3,
            StopStrings::default(),
        ))
        .await;
        assert_eq!(data.last().map(String::as_str), Some("[DONE]"));
        let chunks: Vec<serde_json::Value> = data[..data.len() - 1]
            .iter()
            .map(|d| serde_json::from_str(d).unwrap())
            .collect();
        let n = chunks.len();
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        let (mut reasoning, mut content) = (String::new(), String::new());
        for c in &chunks[1..n - 2] {
            let d = &c["choices"][0]["delta"];
            reasoning.push_str(d["reasoning_content"].as_str().unwrap_or(""));
            content.push_str(d["content"].as_str().unwrap_or(""));
            assert!(c["choices"][0]["finish_reason"].is_null());
        }
        assert_eq!((reasoning.as_str(), content.as_str()), ("We need", "Hi"));
        assert_eq!(chunks[n - 2]["choices"][0]["finish_reason"], "stop");
        assert_eq!(chunks[n - 1]["choices"], serde_json::json!([]));
        assert_eq!(
            chunks[n - 1]["usage"],
            serde_json::json!({"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8})
        );
        // Not asked for: no usage chunk at all.
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        tx.send(StreamItem::Token(7, None)).await.unwrap();
        tx.send(StreamItem::Done(FinishReason::Stop)).await.unwrap();
        drop(tx);
        let tok = Arc::new(test_tokenizer());
        let data = sse_data(chat_stream_response(
            tok,
            "m".into(),
            rx,
            false,
            false,
            false,
            3,
            StopStrings::default(),
        ))
        .await;
        assert!(!data.iter().any(|d| d.contains("\"usage\"")), "{data:?}");
    }

    /// A streamed call cut off by `max_tokens`: no `tool_calls` delta, its text arrives as
    /// content right after the prose it followed, and finish_reason is `length`.
    #[tokio::test]
    async fn tools_stream_cut_call_finishes_length_without_a_call() {
        let tok = Arc::new(test_tokenizer());
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        // "Hi" "\n" "<tool_call>" "\n" "<function=read>" "\n" "<parameter=path>" "\n" "calc.py"
        for id in [7u32, 4, 8, 4, 9, 4, 10, 4, 11] {
            tx.send(StreamItem::Token(id, None)).await.unwrap();
        }
        tx.send(StreamItem::Done(FinishReason::Length))
            .await
            .unwrap();
        drop(tx);
        let cfg = ToolStream {
            format: ToolFormat::QwenXml,
            thinking: false,
            tools: Vec::new(),
            include_usage: false,
            prompt_tokens: 1,
            stops: StopStrings::default(),
            prefill: String::new(),
        };
        let data = sse_data(chat_stream_tools_response(tok, "m".into(), rx, cfg)).await;
        let joined = data.join("\n");
        assert!(!joined.contains("\"tool_calls\""), "{joined}");
        let mut content = String::new();
        let mut finish = None;
        for d in &data[..data.len() - 1] {
            let c: serde_json::Value = serde_json::from_str(d).unwrap();
            content.push_str(c["choices"][0]["delta"]["content"].as_str().unwrap_or(""));
            if let Some(f) = c["choices"][0]["finish_reason"].as_str() {
                finish = Some(f.to_string());
            }
        }
        assert_eq!(
            content,
            "Hi\n<tool_call>\n<function=read>\n<parameter=path>\ncalc.py"
        );
        assert_eq!(finish.as_deref(), Some("length"));
    }

    /// The forcing `tool_choice` shapes of both dialects, and the call opening each format
    /// prefills: up to the name when one is forced, the format's call marker otherwise.
    #[test]
    fn forced_tool_choice_reads_both_dialects_and_prefills_the_format() {
        use serde_json::json;
        let named = Some(ForcedCall::Named("read".into()));
        for (choice, want) in [
            (json!("required"), Some(ForcedCall::Any)),
            (json!({"type": "any"}), Some(ForcedCall::Any)),
            (json!({"type": "tool", "name": "read"}), named.clone()),
            (
                json!({"type": "function", "function": {"name": "read"}}),
                named.clone(),
            ),
            (json!("auto"), None),
            (json!("none"), None),
            (json!({"type": "auto"}), None),
            (json!({"type": "tool"}), None),
        ] {
            assert_eq!(forced_call(Some(&choice)), want, "{choice}");
        }
        assert_eq!(forced_call(None), None);
        let any = ForcedCall::Any;
        let read = ForcedCall::Named("read".into());
        assert_eq!(call_prefill(&any, ToolFormat::QwenXml), "<tool_call>\n");
        assert_eq!(
            call_prefill(&read, ToolFormat::QwenXml),
            "<tool_call>\n<function=read>\n"
        );
        assert_eq!(call_prefill(&any, ToolFormat::HermesJson), "<tool_call>\n");
        assert_eq!(
            call_prefill(&read, ToolFormat::HermesJson),
            "<tool_call>\n{\"name\": \"read\", \"arguments\": "
        );
        assert_eq!(call_prefill(&any, ToolFormat::None), "```json\n");
        assert_eq!(
            call_prefill(&read, ToolFormat::None),
            "```json\n{\"function\": \"read\", \"arguments\": "
        );
        // Not enforceable: nothing to force, nothing on offer, JSON mode, a name not on offer.
        let tools: Vec<Tool> = serde_json::from_value(read_tool()).unwrap();
        let req = json!("required");
        let f = ToolFormat::QwenXml;
        assert_eq!(enforce_tool_choice(None, &tools, f, false), None);
        assert_eq!(enforce_tool_choice(Some(&req), &[], f, false), None);
        assert_eq!(enforce_tool_choice(Some(&req), &tools, f, true), None);
        let other = json!({"type": "tool", "name": "write"});
        assert_eq!(enforce_tool_choice(Some(&other), &tools, f, false), None);
        assert_eq!(
            enforce_tool_choice(Some(&req), &tools, f, false).as_deref(),
            Some("<tool_call>\n")
        );
    }

    /// The prefill goes after an EMPTY, CLOSED `<think>` when the cue opened one — the call is the
    /// answer — and straight after the cue otherwise.
    #[test]
    fn prefill_closes_an_open_think_first() {
        let mut thinking = "<|im_start|>assistant\n<think>\n".to_string();
        prefill_call(&mut thinking, "<tool_call>\n");
        assert_eq!(
            thinking,
            "<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n"
        );
        assert!(!crate::chat::prompt_opens_think(&thinking));
        let mut plain = "<|im_start|>assistant\n".to_string();
        prefill_call(&mut plain, "```json\n");
        assert_eq!(plain, "<|im_start|>assistant\n```json\n");
    }

    /// `tool_choice: "required"` END TO END on the model's own template: the prompt was begun with
    /// `<tool_call>\n`, so a reply that continues it (`<function=read>…`) reads as the call — at
    /// the template's default (thinking) settings, where the reply would otherwise be reasoning.
    /// Until 2026-10-04 `required` was not enforced and this reply was not a call.
    #[tokio::test]
    async fn chat_completions_required_tool_choice_is_prefilled() {
        let (state, jobs) = app_state(Some(qwen38()));
        let body = serde_json::json!({"max_tokens": 256, "tools": read_tool(),
            "tool_choice": "required",
            "messages": [{"role": "user", "content": "read calc.py"}]});
        let actor = std::thread::spawn(move || {
            let job = jobs.recv().unwrap();
            // `<function=read>\n<parameter=path>\ncalc.py\n</parameter>\n</function>\n</tool_call>`
            for id in [9u32, 4, 10, 4, 11, 4, 12, 4, 13, 4, 14] {
                job.token_tx
                    .blocking_send(StreamItem::Token(id, None))
                    .unwrap();
            }
            job.token_tx
                .blocking_send(StreamItem::Done(FinishReason::Stop))
                .unwrap();
        });
        let resp = chat_completions(
            State(state),
            axum::body::Bytes::from(serde_json::to_vec(&body).unwrap()),
        )
        .await;
        actor.join().unwrap();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["choices"][0]["finish_reason"], "tool_calls", "{v}");
        let m = &v["choices"][0]["message"];
        assert!(m["reasoning_content"].is_null(), "{v}");
        assert_eq!(m["tool_calls"][0]["function"]["name"], "read");
        assert_eq!(
            m["tool_calls"][0]["function"]["arguments"], "{\"path\":\"calc.py\"}",
            "{v}"
        );
    }

    /// A forced `tool_choice` AND a stop string, streamed and not.
    /// The prompt is begun with `<tool_call>\n`; the stop string is `<tool_call>` itself. The model
    /// completes the forced call, then opens a SECOND one: the job ends on that token and the
    /// reply is cut there, so exactly one call comes back. The stop string never matches inside
    /// the prefill — the stop filter and the job's check see generated text only — or there
    /// would be no call at all. Without the prefill this reply is reasoning with no call; without
    /// the stop string it is two calls and 23 tokens.
    #[tokio::test]
    async fn chat_completions_forced_call_with_a_stop_string() {
        // `<function=read>…</tool_call>` `\n`, then a second `<tool_call>` and its call.
        let call = [9u32, 4, 10, 4, 11, 4, 12, 4, 13, 4, 14];
        let mut ids: Vec<u32> = call.to_vec();
        ids.extend([4, 8]);
        ids.extend(call);
        for stream in [false, true] {
            let (state, jobs) = app_state(Some(qwen38()));
            let body = serde_json::json!({"max_tokens": 256, "stream": stream,
                "tools": read_tool(), "tool_choice": "required", "stop": ["<tool_call>"],
                "messages": [{"role": "user", "content": "read calc.py"}]});
            let actor = play_with_stops(jobs, ids.clone());
            let resp = chat_completions(
                State(state),
                axum::body::Bytes::from(serde_json::to_vec(&body).unwrap()),
            )
            .await;
            let (calls, finish) = if stream {
                let data = sse_data(resp).await;
                assert_eq!(data.last().map(String::as_str), Some("[DONE]"));
                let chunks: Vec<serde_json::Value> = data[..data.len() - 1]
                    .iter()
                    .map(|d| serde_json::from_str(d).unwrap())
                    .collect();
                let calls: Vec<serde_json::Value> = chunks
                    .iter()
                    .filter_map(|c| c["choices"][0]["delta"]["tool_calls"].as_array())
                    .flatten()
                    .map(|c| c["function"].clone())
                    .collect();
                let finish = chunks.iter().find_map(|c| {
                    c["choices"][0]["finish_reason"]
                        .as_str()
                        .map(str::to_string)
                });
                (calls, finish.unwrap_or_default())
            } else {
                let v = body_json(resp).await;
                let calls = v["choices"][0]["message"]["tool_calls"]
                    .as_array()
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .map(|c| c["function"].clone())
                    .collect();
                (
                    calls,
                    v["choices"][0]["finish_reason"]
                        .as_str()
                        .unwrap()
                        .to_string(),
                )
            };
            assert_eq!(actor.join().unwrap(), call.len() + 2, "stream {stream}");
            assert_eq!(finish, "tool_calls", "stream {stream}");
            assert_eq!(calls.len(), 1, "stream {stream}: {calls:?}");
            assert_eq!(calls[0]["name"], "read");
            assert_eq!(calls[0]["arguments"], "{\"path\":\"calc.py\"}");
        }
    }

    #[test]
    fn tool_choice_and_template_kwargs_deserialize() {
        let r = req(
            serde_json::json!({"messages": [{"role": "user", "content": "x"}],
            "tool_choice": {"type": "function", "function": {"name": "read"}},
            "chat_template_kwargs": {"enable_thinking": false, "unknown": [1]}}),
        );
        assert_eq!(r.tool_choice.unwrap()["function"]["name"], "read");
        let kw = r.chat_template_kwargs.unwrap();
        assert_eq!(kw.enable_thinking, Some(false));
        assert_eq!(kw.reasoning_effort, None);
    }
}
