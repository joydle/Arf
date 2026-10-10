//! The Anthropic Messages wire format: `POST /v1/messages` and `POST /v1/messages/count_tokens`.
//!
//! Coding agents speak one of two dialects. `/v1/chat/completions` covers the OpenAI one; this
//! covers the other, so a client that only knows the Anthropic shape can point at a local `arf`.
//! It is a TRANSLATION LAYER and nothing else: a request is flattened into the same
//! [`ChatMessage`] list, templated, tokenized and queued by the very helpers the OpenAI handler
//! uses (`super::`), so both endpoints share one prompt path, one sampler and one actor.
//!
//! What is translated:
//! - `system` (string or text blocks) → a leading system message.
//! - `content` as a string or as blocks: `text`; `tool_use` (assistant history) rendered back
//!   into the fenced-JSON call the model itself emits (`chat::tool_system_prompt`); `tool_result`
//!   rendered into the user turn it arrives in. One Anthropic message stays ONE chat message.
//!   That is the fenced path. On the native path (since 2026-09-26, review — [`plan_messages`])
//!   the history is translated to the OpenAI shape the model's own template renders: a
//!   `tool_use` becomes the assistant turn's `tool_calls` entry (same id, `input` in the
//!   client's key order), a `tool_result` a `tool` message, a `thinking` block the turn's
//!   `reasoning_content` ([`native_messages`]).
//! - `tools` → the existing tool prompt; a parsed call comes back as a `tool_use` block with
//!   `stop_reason: "tool_use"`. `tool_choice: {"type": "none"}` withdraws them — neither shown to
//!   the model nor parsed — as `"none"` does on `/v1/chat/completions`. On the native path the
//!   tools reach the template as OpenAI-shaped functions ([`native_tools`]), and the reply is
//!   read in the template's format by the same splitter and parser as that endpoint's.
//!   A `tool_use`'s `input` is the model's own arguments text, its key order kept, streamed or
//!   not ([`call_input`]) — the history a client replays then renders the bytes the model
//!   generated. Until 2026-09-26 the non-streamed reply sorted the keys ([`ReplyBlock`]).
//!   (Corrected the same day, review: one parser path still sorted them, streamed or not — the
//!   fenced fallback's bare-key `call:NAME{key:"v"}` shorthand, `chat::coerce_loose_args`, which
//!   also garbled non-ASCII text. Both fixed there. Arguments that are a JSON string must pass a
//!   strict parse to be sent verbatim — [`call_input`].)
//! - `stream: true` → the Anthropic SSE event sequence (`message_start`, `content_block_*`,
//!   `message_delta`, `message_stop`), each with its `event:` name.
//! - A prompt whose template pre-opens `<think>` (Qwen3.8's): the reasoning comes back as a
//!   `thinking` block (streamed live without tools — and, since 2026-09-26, with them) and tool
//!   calls are parsed from the answer only.
//! - A prompt that fills the context is refused with Anthropic's own `invalid_request_error`
//!   wording, `prompt is too long: N tokens > M maximum`; `max_tokens` is clamped to what is left.
//!   A reply that runs into that clamp — the client asked for more than the context had left, or
//!   set no `max_tokens` — stops with `model_context_window_exceeded`, Anthropic's value for a
//!   response that filled the context window; one that runs into the client's own `max_tokens`
//!   stops with `max_tokens` ([`window_stop`]). Until 2026-10-04 both said `max_tokens`, and
//!   Claude Code then told its user to raise `CLAUDE_CODE_MAX_OUTPUT_TOKENS`, which cannot help.
//!
//! What is NOT, stated so nobody assumes it:
//! - `stop_sequences` are enforced (`crate::stop`): the reply ends at the first
//!   one, which is not part of it, with `stop_reason: "stop_sequence"` and `stop_sequence` set to
//!   the string matched. Until then they were accepted and ignored.
//! - `image` blocks are refused with a 400 rather than silently dropped — unless the server runs
//!   Qwen3.8 with its mmproj (`--mmproj`, a `qwen3vl_merger` projector), which takes
//!   `{"type": "image", "source": {"type": "base64", "media_type", "data"}}` (and `"url"` sources:
//!   `data:` URLs, or local paths with `ARF_IMAGE_FILES=1`) on the hand-written template path —
//!   an image request never takes the native tool template (as on `/v1/chat/completions`).
//!   (Added 2026-09-27.)
//! - `thinking`, `metadata` and unknown block types are ignored. So was `tool_choice` until
//!   2026-09-26; now `none` is honoured (above) and `any` / `tool` are accepted and NOT enforced —
//!   the model chooses (logged once), as on `/v1/chat/completions`.
//!   **Corrected 2026-10-04:** `any` and `tool` are enforced by prefill, as on
//!   `/v1/chat/completions` (`http::enforce_tool_choice`): the reply is begun with the opening of
//!   a call — up to the tool's name for `tool` — after an empty, closed `<think>` (Anthropic
//!   refuses extended thinking with a forced `tool_choice`, so the call is the answer). A model
//!   that then fails to complete a call that parses still gets its reply back as text.
//!   **Corrected 2026-09-26:** `thinking` is no longer ignored ([`thinking_vars`]). It maps onto
//!   the variables `/v1/chat/completions` takes from `chat_template_kwargs.enable_thinking`:
//!   `{"type": "disabled"}` renders the prompt non-thinking on both paths (effort `"none"`,
//!   `enable_thinking=false`), so the reply carries no `thinking` block; `{"type": "enabled"}`
//!   sets `enable_thinking=true`. Its `budget_tokens` is accepted and NOT enforced — there is no
//!   thinking budget to map it to (logged once). Absent: the server's defaults, as before.
//!   `metadata` is still ignored.
//!   **Corrected again 2026-09-26 (review):** that left the real client out. Claude Code sends
//!   `thinking: {"type": "adaptive"}` and `output_config: {"effort": …}`; both were dropped
//!   without a log line. `output_config.effort` is now the request's reasoning effort (`low`,
//!   `medium`, `high`; `max` → `xhigh`), and `adaptive` is read as the server's default
//!   thinking mode on purpose, logged once ([`thinking_vars`] says why).
//! - Tools use the fenced-JSON protocol here even when the model's own chat template teaches a
//!   native format — that path (`jinja_chat`) is wired into `/v1/chat/completions` only.
//!   **Corrected 2026-09-26 (review):** no longer. [`plan_messages`] makes `http::plan_chat`'s
//!   choice: tools or tool history → the model's own template when it teaches a format Arf
//!   parses (qwen-xml, hermes-json); otherwise — no template, one that teaches no format or fails
//!   to render, `ARF_NO_NATIVE_TOOLS` — the fenced-JSON protocol as before. A request with
//!   neither tools nor tool history renders exactly as it did. Which path a request took is
//!   logged once per process (`log_tool_path_once` bits 0 and 1).
//! - With `tools` present a streamed reply is BUFFERED and its blocks are emitted at the end:
//!   a tool call is only recognisable once the JSON object has closed. Text-only streams are
//!   token by token.
//!   **Corrected 2026-09-26:** no longer so. A tool stream now runs through the OpenAI tool
//!   stream's splitter and end-of-reply decision ([`stream_tool_blocks`]): the reasoning streams
//!   live as a `thinking` block, and each complete call becomes a `tool_use` block. What stays
//!   buffered is the fenced protocol's ANSWER — the prose and the call are settled when the
//!   reply ends — exactly as `/v1/chat/completions` buffers the answer on its fenced fallback.
//!   Answer prose streams live up to the call only with a native format (qwen-xml, hermes),
//!   which this route does not render yet (the point above). The buffered version measured
//!   the first delta at 9.58 s with tools vs 1.04 s without, 2 deltas vs 298 (measured,
//!   2026-09-26, "Coding agents after the tool-protocol merge"); the live one is NOT measured.
//!   **Corrected again 2026-09-26 (review):** that correction left the production case where it
//!   was. The handler always passed `ToolFormat::None` (the prompt was always fenced), so on
//!   Qwen3.8 — whose template teaches qwen-xml — the answer's prose arrived as ONE `text_delta`
//!   when the reply ended, and a reply that does not open in `<think>` sent nothing after
//!   `message_start` until then: the 9.58 s symptom, unchanged. The route now renders natively
//!   (the point above) and passes the template's format, so the prose streams live up to the
//!   call as on `/v1/chat/completions`. Only the fenced fallback still settles the answer when
//!   the reply ends — as `/v1/chat/completions` does on that fallback. Still NOT measured live.
//! - `x-api-key` / `anthropic-version` headers are not read. The server has no auth (SECURITY.md).
//! - Mid-conversation tool changes (2026-10-04): a tool declared with `defer_loading: true` is
//!   withheld, and `tool_addition` / `tool_removal` blocks (in a `role: "system"` message) add
//!   and withdraw tools, applied in message order ([`resolved_tools`]). Until then the blocks
//!   fell into `Block::Other` and every declared tool was offered from the first turn: a tool
//!   added mid-session was offered from the start anyway, and a removed one was never removed.
//!   What is NOT done: the model's template renders ONE tool list, at the head of the prompt, so
//!   the RESOLVED list is offered for the whole conversation rather than "from that point
//!   onward" — an added tool is appended (the tools before it render as they did, so the cached
//!   prefix holds up to it), a removed one disappears from the head and the prefix cache misses
//!   from there. MCP references (`mcp_tool_reference`, `mcp_toolset_reference`, an
//!   `mcp_toolset` definition) are ignored and logged once: Arf has no MCP client. A system
//!   message that carried only tool changes is not rendered as a turn.

use super::*;
use serde_json::{json, Value};

/// `POST /v1/messages` request body. Unknown fields are ignored (serde's default).
#[derive(Debug, Deserialize)]
pub(super) struct MessagesRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<AnthMessage>,
    #[serde(default)]
    system: Option<SystemPrompt>,
    /// Required by Anthropic; tolerated when absent (the server's cap applies).
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    stream: bool,
    /// Enforced — see the module docs.
    #[serde(default)]
    stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    tools: Option<Vec<AnthTool>>,
    /// Anthropic `tool_choice`: `{"type": "auto" | "any" | "tool" | "none"}`. `none` is honoured
    /// as `/v1/chat/completions` honours `"none"` ([`tools_on_offer`]); `any` and `tool` were
    /// ACCEPTED AND NOT ENFORCED until 2026-10-04, and are now enforced by prefill
    /// ([`prompt_of`], `http::enforce_tool_choice`). Until 2026-09-26 the field was not
    /// declared, serde dropped it, and a `none` request could still come back with a `tool_use`.
    #[serde(default)]
    tool_choice: Option<Value>,
    /// Anthropic extended thinking: `{"type": "enabled", "budget_tokens": N}` or
    /// `{"type": "disabled"}` — read as `/v1/chat/completions` reads
    /// `chat_template_kwargs.enable_thinking` ([`MessagesRequest::enable_thinking`],
    /// [`thinking_vars`]). Absent, or a shape this file does not know: the server's defaults, as
    /// before. A `Value`, leniently read like `tool_choice`: until 2026-09-26 the field was not
    /// declared and ANY value was accepted (and ignored), so a shape it cannot read must not
    /// start failing the request.
    ///
    /// `{"type": "adaptive"}` (what Claude Code sends, with `output_config.effort`) is read as
    /// the server's default thinking mode, on purpose — see [`thinking_vars`] (2026-09-26).
    #[serde(default)]
    thinking: Option<Value>,
    /// Anthropic `output_config`: its `effort` (`low` / `medium` / `high` / `max`) is Arf's
    /// reasoning effort ([`MessagesRequest::output_effort`]); other fields are ignored. Until
    /// 2026-09-26 (review) it was not declared, serde dropped it, and a client that asked for
    /// `low` got the `xhigh` sentence without a word in the log. A `Value`, leniently read, for
    /// the same reason as `thinking`.
    #[serde(default)]
    output_config: Option<Value>,
}

impl MessagesRequest {
    /// `tool_choice`'s `type` (a bare string is read too, leniently).
    fn tool_choice_type(&self) -> Option<&str> {
        let c = self.tool_choice.as_ref()?;
        c.get("type").and_then(Value::as_str).or_else(|| c.as_str())
    }

    /// `thinking`'s `type`, as sent.
    fn thinking_type(&self) -> Option<&str> {
        self.thinking.as_ref()?.get("type").and_then(Value::as_str)
    }

    /// `thinking` as the `enable_thinking` it asks for: `Some(true)` for `enabled`, `Some(false)`
    /// for `disabled`, `None` for another type (the server's defaults apply) — `adaptive`
    /// included, deliberately ([`thinking_vars`]).
    ///
    /// NO `thinking` FIELD AND NO EFFORT IS `disabled` (2026-10-07), as in the Messages API, where extended
    /// thinking is off unless a request asks for it. It was `None`, the template's default — on,
    /// for Qwen3.8 — and that template opens the system block with its reasoning instructions,
    /// so the same system prompt sent with `thinking: disabled` and with no `thinking` field were
    /// two prompts that shared nothing. Claude Code's auto-mode classifier does exactly that with
    /// a constant ~30,500-token system prompt (stage 1 `disabled`, stage 2 no field): the second
    /// stage read all of it again and never answered inside its 60 s.
    fn enable_thinking(&self) -> Option<bool> {
        match self.thinking_type() {
            Some("enabled") => Some(true),
            Some("disabled") => Some(false),
            Some(_) => None,
            // A request that names an effort wants reasoning at that effort, with or without the
            // field (`output_config.effort`, below): the server's default applies, as before.
            None if self.thinking.is_none() && self.effort_field().is_none() => Some(false),
            None => None,
        }
    }

    /// `output_config.effort`, as sent, when it is a string.
    fn effort_field(&self) -> Option<&str> {
        self.output_config.as_ref()?.get("effort")?.as_str()
    }

    /// `output_config.effort` as Arf's reasoning effort ([`anthropic_effort`]); `None` when absent
    /// or a level Arf does not know (the server's default then applies, and it is logged once).
    fn output_effort(&self) -> Option<&'static str> {
        anthropic_effort(self.effort_field()?)
    }

    /// `thinking.budget_tokens`, when `enabled` carries one — accepted and NOT enforced
    /// ([`thinking_vars`]).
    fn thinking_budget(&self) -> Option<u64> {
        let t = self.thinking.as_ref()?;
        (self.enable_thinking() == Some(true))
            .then(|| t.get("budget_tokens").and_then(Value::as_u64))
            .flatten()
    }
}

/// A request's reasoning settings — the effort the hand-written templates take and the variables
/// the model's own template takes — resolved as `/v1/chat/completions` resolves a request whose
/// `chat_template_kwargs.enable_thinking` is what this one's `thinking` asks for
/// ([`MessagesRequest::enable_thinking`]): `http::request_effort` (`false` IS the effort
/// `"none"`), then `http::template_vars` with the server's defaults (`no_think_env` =
/// `ARF_NO_QWEN_THINK_TEMPLATE`, `env_effort` = `ARF_REASONING_EFFORT`), passed in so a test names
/// them rather than inheriting the shell's.
///
/// - `disabled` → effort `"none"`: the hand-written Qwen3.8 template renders its non-thinking
///   form (a pre-CLOSED `<think>` block, `chat::qwen3_think_template`), and the model's own
///   template gets `enable_thinking=false` (Qwen3.8's pre-closes it too, and drops its effort
///   sentence from the system turn, as the hand-written form does). The prompt no longer
///   opens `<think>`, so the reply is read as answer from its first token and carries no
///   `thinking` block, streamed or not.
/// - `enabled` → `enable_thinking=true` for the model's own template (it overrides the operator's
///   `ARF_NO_QWEN_THINK_TEMPLATE` there, exactly as the OpenAI kwarg does); the hand-written
///   templates have no such variable and render at the server's effort — as for an OpenAI
///   request that sets the kwarg.
/// - absent (or another type) → `effort` `None` and the variables of a request that sets
///   nothing: byte-for-byte the prompt of before 2026-09-26.
///
/// `budget_tokens` is NOT mapped. Arf has a reasoning-EFFORT knob (a sentence in the system
/// prompt: `xhigh` / `medium` / `low`), not a thinking-token budget, and turning a token count
/// into one of those three would be thresholds nobody measured — silently changing the prompt a
/// client's runs are compared at (`http::log_native_mode_once`: agent baselines are only
/// comparable at the same reasoning mode). It is accepted, logged once, and the reasoning ends
/// on `</think>` or `max_tokens` as before.
///
/// **Corrected 2026-09-26 (review):** that reasoning missed that the protocol HAS a direct
/// effort field, and the real client uses it. Claude Code sends neither `enabled` nor
/// `disabled` but `thinking: {"type": "adaptive"}` with `output_config: {"effort": …}`; both
/// were read as "nothing set", with no log line. Now:
/// - `output_config.effort` is the request's effort, as `reasoning_effort` is on
///   `/v1/chat/completions` ([`anthropic_effort`]: `low`, `medium`, `high` 1:1, `max` → `xhigh`,
///   Arf's highest). It wins over `ARF_REASONING_EFFORT`, as the OpenAI field does, and loses to
///   `disabled` (`"none"`), as `reasoning_effort` loses to `enable_thinking: false`. An unknown
///   level is the server's default, logged once. `high` renders the same bytes as no effort
///   under the default settings (both templates send `high` to `xhigh`) — Claude Code's own
///   `high` changes nothing; a client's `low` now gets the `low` sentence.
/// - `adaptive` is the server's default thinking mode, ON PURPOSE, and logged once: it asks the
///   model to decide, which here is the operator's setting — the template's own (thinking), or
///   off under `ARF_NO_QWEN_THINK_TEMPLATE`. So `enabled` and `adaptive` differ under that
///   switch, and that is the intent: `enabled` is an explicit request and overrides it, as the
///   OpenAI `enable_thinking: true` kwarg does; if `adaptive` did too, an operator could not turn
///   thinking off for the one client that always sends it.
///
/// `budget_tokens` stays unmapped: a token count still has no measured effort to map to.
fn thinking_vars(
    req: &MessagesRequest,
    no_think_env: bool,
    env_effort: Option<String>,
) -> (Option<&'static str>, crate::jinja_chat::TemplateVars) {
    let kwargs = ChatTemplateKwargs {
        enable_thinking: req.enable_thinking(),
        ..Default::default()
    };
    let effort = if kwargs.enable_thinking == Some(false) {
        Some("none")
    } else {
        req.output_effort()
    };
    let vars = template_vars(effort, Some(&kwargs), no_think_env, env_effort);
    (effort, vars)
}

/// An Anthropic effort level as Arf's reasoning effort: the value `/v1/chat/completions` takes as
/// `reasoning_effort`, which both prompt paths read — the hand-written Qwen3.8 template
/// (`chat::qwen3_think_template`) and the model's own (`http::template_effort`). `low`,
/// `medium` and `high` are the same words there; `max` (and `xhigh`) is Arf's highest, `xhigh`.
/// Anything else is `None` — the server's default — rather than passed through: a word the
/// hand-written template does not know renders `xhigh` while `template_effort` may map it
/// elsewhere (`minimal`), and `"none"` would turn thinking off, which is `disabled`'s job.
fn anthropic_effort(effort: &str) -> Option<&'static str> {
    match effort {
        "low" => Some("low"),
        "medium" => Some("medium"),
        "high" => Some("high"),
        "max" | "xhigh" => Some("xhigh"),
        _ => None,
    }
}

/// Log a `thinking` note once per process (bit 0: `budget_tokens` not enforced; bit 1: a
/// `disabled` request whose template opened `<think>` anyway; bit 2: `adaptive` read as the
/// server's default; bit 3: an `output_config.effort` honoured; bit 4: one Arf does not know).
/// Its own bits, apart from `log_tool_path_once`'s, whose eight are all taken. Bit 5 is not about
/// thinking: an MCP tool change ignored ([`resolved_tools`]) — the one other once-only note here.
fn log_thinking_once(bit: u8, msg: impl FnOnce() -> String) {
    use std::sync::atomic::AtomicU8;
    static SEEN: AtomicU8 = AtomicU8::new(0);
    if SEEN.fetch_or(1 << bit, Ordering::Relaxed) & (1 << bit) == 0 {
        eprintln!("[serve] {}", msg());
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum SystemPrompt {
    Text(String),
    Blocks(Vec<Block>),
}

#[derive(Debug, Deserialize)]
struct AnthMessage {
    role: String,
    content: Content,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Content {
    Text(String),
    Blocks(Vec<Block>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Block {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        /// Order-preserving (`minijinja`'s map keeps insertion order here; `serde_json::Value`
        /// would sort it): the native path replays the call with the model's own argument
        /// order, so the history matches what it generated.
        #[serde(default)]
        input: minijinja::Value,
    },
    ToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: Option<Content>,
        #[serde(default)]
        is_error: bool,
    },
    Image {
        #[serde(default)]
        source: Option<ImageSource>,
    },
    /// Assistant history: the reasoning of that turn. The native path replays it as the turn's
    /// `reasoning_content`; the fenced path drops it, as it dropped it when this was `Other`.
    Thinking {
        #[serde(default)]
        thinking: String,
    },
    /// Offers a tool from here on ([`resolved_tools`]): `{"type": "tool_addition", "tool":
    /// {"type": "tool_reference", "name"}}`, or with `{"type": "tool_definition", "definition":
    /// {name, description, input_schema}}` a tool defined in the message itself.
    ToolAddition {
        #[serde(default)]
        tool: ToolChangeRef,
    },
    /// Withdraws a tool from here on: `{"type": "tool_removal", "tool": {"type":
    /// "tool_reference", "name"}}`.
    ToolRemoval {
        #[serde(default)]
        tool: ToolChangeRef,
    },
    /// `redacted_thinking`, and anything newer than this file.
    #[serde(other)]
    Other,
}

/// The `tool` of a `tool_addition` / `tool_removal` block. Read leniently — every field optional —
/// because the content enums are untagged: a shape this file does not know must not turn the
/// whole request into a 400. The MCP references are not read (module docs).
#[derive(Debug, Default, Deserialize)]
struct ToolChangeRef {
    /// `tool_reference` or `tool_definition`.
    #[serde(rename = "type", default)]
    kind: String,
    /// A `tool_reference`'s tool name.
    #[serde(default)]
    name: Option<String>,
    /// A `tool_definition`'s tool, order-preserving like [`AnthTool::input_schema`].
    #[serde(default)]
    definition: Option<minijinja::Value>,
}

impl ToolChangeRef {
    /// The name of the tool this refers to, by reference or by definition; `None` for an MCP
    /// reference or toolset.
    fn name(&self) -> Option<String> {
        match self.kind.as_str() {
            "tool_reference" => self.name.clone(),
            "tool_definition" => self
                .definition
                .as_ref()?
                .get_attr("name")
                .ok()?
                .as_str()
                .map(str::to_string),
            _ => None,
        }
    }

    /// The tool an ADDITION offers: the declared tool it names, or the one it defines. `Ok(None)`
    /// for what Arf cannot offer (an MCP reference or toolset, an unknown kind). `Err` when a
    /// reference names a tool `tools` does not declare — Anthropic's `tool_reference_unresolved`
    /// 400.
    fn resolve(&self, declared: &[AnthTool]) -> Result<Option<AnthTool>, String> {
        let Some(name) = self.name() else {
            return Ok(None);
        };
        if self.kind == "tool_definition" {
            let def = self.definition.as_ref().expect("name() read it");
            let field = |k: &str| def.get_attr(k).ok().filter(|v| !v.is_undefined());
            return Ok(Some(AnthTool {
                name,
                description: field("description").and_then(|v| v.as_str().map(str::to_string)),
                input_schema: field("input_schema"),
                defer_loading: false,
            }));
        }
        match declared.iter().find(|t| t.name == name) {
            Some(t) => Ok(Some(t.clone())),
            None => Err(format!(
                "tool_reference_unresolved: tool_addition references tool `{name}`, which is not \
                 declared in `tools`"
            )),
        }
    }
}

/// The request's tool changes in message order: `(true, tool)` for an addition, `(false, tool)`
/// for a removal.
fn tool_changes(req: &MessagesRequest) -> impl Iterator<Item = (bool, &ToolChangeRef)> {
    req.messages
        .iter()
        .filter_map(|m| match &m.content {
            Content::Blocks(blocks) => Some(blocks),
            Content::Text(_) => None,
        })
        .flatten()
        .filter_map(|b| match b {
            Block::ToolAddition { tool } => Some((true, tool)),
            Block::ToolRemoval { tool } => Some((false, tool)),
            _ => None,
        })
}

/// True for a message whose blocks are all tool changes: it says nothing to the model itself, so
/// neither path renders it as a turn (an empty system turn mid-conversation would make Qwen3.8's
/// template raise and send the request to the fenced fallback).
fn only_tool_changes(content: &Content) -> bool {
    match content {
        Content::Blocks(blocks) => {
            !blocks.is_empty()
                && blocks
                    .iter()
                    .all(|b| matches!(b, Block::ToolAddition { .. } | Block::ToolRemoval { .. }))
        }
        Content::Text(_) => false,
    }
}

/// The tools the conversation ends up offering: the declared `tools` without the
/// `defer_loading` ones, then each `tool_addition` / `tool_removal` applied in message order. An
/// added tool is appended (one already offered stays where it is); removing a tool that is not
/// offered changes nothing (Anthropic's rule). An addition Arf cannot resolve is skipped here —
/// [`tool_change_error`] has refused the request already when it names an undeclared tool.
fn resolved_tools(req: &MessagesRequest) -> Vec<AnthTool> {
    let declared = req.tools.as_deref().unwrap_or(&[]);
    let mut out: Vec<AnthTool> = declared
        .iter()
        .filter(|t| !t.defer_loading)
        .cloned()
        .collect();
    for (add, change) in tool_changes(req) {
        let Some(name) = change.name() else {
            log_thinking_once(5, || {
                "a tool_addition / tool_removal names an MCP tool or toolset — ignored, Arf has \
                 no MCP client (logged once)"
                    .into()
            });
            continue;
        };
        if !add {
            out.retain(|t| t.name != name);
        } else if !out.iter().any(|t| t.name == name) {
            if let Ok(Some(t)) = change.resolve(declared) {
                out.push(t);
            }
        }
    }
    out
}

/// The 400 for a `tool_addition` that references a tool `tools` does not declare, as Anthropic
/// refuses it (`tool_reference_unresolved`).
fn tool_change_error(req: &MessagesRequest) -> Option<String> {
    let declared = req.tools.as_deref().unwrap_or(&[]);
    tool_changes(req)
        .filter(|(add, _)| *add)
        .find_map(|(_, change)| change.resolve(declared).err())
}

/// An `image` block's `source`: `{"type": "base64", "media_type", "data"}` or
/// `{"type": "url", "url"}`.
#[derive(Debug, Deserialize)]
struct ImageSource {
    #[serde(rename = "type", default)]
    kind: String,
    #[serde(default)]
    media_type: Option<String>,
    #[serde(default)]
    data: Option<String>,
    #[serde(default)]
    url: Option<String>,
}

impl ImageSource {
    /// The source as the image reference `qwen_image::load_image` reads (a data URL, or the URL).
    fn reference(&self) -> Result<String, String> {
        match (self.kind.as_str(), &self.data, &self.url) {
            ("base64", Some(d), _) => Ok(format!(
                "data:{};base64,{d}",
                self.media_type.as_deref().unwrap_or("image/png")
            )),
            ("url", _, Some(u)) => Ok(u.clone()),
            _ => Err(
                "image source must be {type: base64, media_type, data} or {type: url, url}".into(),
            ),
        }
    }
}

/// True when any message carries an `image` block.
fn has_images(req: &MessagesRequest) -> bool {
    req.messages.iter().any(|m| match &m.content {
        Content::Blocks(b) => b.iter().any(|b| matches!(b, Block::Image { .. })),
        Content::Text(_) => false,
    })
}

/// An Anthropic tool: `{name, description, input_schema}`, and `defer_loading`.
#[derive(Debug, Clone, Deserialize)]
struct AnthTool {
    name: String,
    #[serde(default)]
    description: Option<String>,
    /// Order-preserving, like `Block::ToolUse::input`: the native path hands the schema to the
    /// template in the CLIENT's key order, as `/v1/chat/completions` does (`http::render_native`).
    #[serde(default)]
    input_schema: Option<minijinja::Value>,
    /// Withheld until a `tool_addition` offers it ([`resolved_tools`]).
    #[serde(default)]
    defer_loading: bool,
}

/// Anthropic's error envelope, with the matching HTTP status.
fn anth_error(status: StatusCode, kind: &str, message: &str) -> Response {
    (
        status,
        Json(json!({"type": "error", "error": {"type": kind, "message": message}})),
    )
        .into_response()
}

fn bad_request(message: &str) -> Response {
    anth_error(StatusCode::BAD_REQUEST, "invalid_request_error", message)
}

/// The refusal of a prompt that fills the context, in Anthropic's wording. Anthropic-dialect
/// clients match `prompt is too long` to compact and retry; the OpenAI wording
/// (`context_budget`'s, used until 2026-09-26 review) read to them as a generic 400. Claude Code
/// is believed to key on that text — not verified against its source here.
fn prompt_too_long(input_tokens: usize, cap: usize) -> Response {
    bad_request(&format!(
        "prompt is too long: {input_tokens} tokens > {cap} maximum"
    ))
}

/// The text blocks of a list, joined by newlines; everything else skipped.
fn blocks_text(blocks: &[Block]) -> String {
    blocks
        .iter()
        .filter_map(|b| match b {
            Block::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Plain text of a content value.
fn text_of(content: &Content) -> String {
    match content {
        Content::Text(s) => s.clone(),
        Content::Blocks(blocks) => blocks_text(blocks),
    }
}

/// The tools a request offers the model, in the OpenAI shape the tool prompt and the parsers
/// take: none when `tool_choice` is `none` (as `http::offered_tools` for `/v1/chat/completions`).
/// The prompt, the non-streamed parse and the stream all read this one list — the conversation's
/// [`resolved_tools`], so a deferred or removed tool is neither shown nor parsed.
fn tools_on_offer(req: &MessagesRequest) -> Vec<Tool> {
    if req.tool_choice_type() == Some("none") {
        return Vec::new();
    }
    resolved_tools(req)
        .iter()
        .map(|t| Tool {
            r#type: "function".into(),
            function: ToolFunction {
                name: t.name.clone(),
                description: t.description.clone(),
                // Sorted here, as it was when the field was a `serde_json::Value`: the fenced
                // prompt's bytes do not change.
                parameters: t.input_schema.as_ref().map(json_value),
            },
        })
        .collect()
}

/// An order-preserving request value as a `serde_json::Value` (keys sorted — what the fenced
/// path and the parsers always read).
fn json_value(v: &minijinja::Value) -> Value {
    serde_json::to_value(v).unwrap_or(Value::Null)
}

/// True when the conversation carries tool turns — a `tool_use` or a `tool_result` block —
/// which render natively even with no tools on offer (`http::plan_chat`'s rule), so the model
/// sees its own calls.
fn has_tool_history(req: &MessagesRequest) -> bool {
    req.messages.iter().any(|m| match &m.content {
        Content::Blocks(blocks) => blocks
            .iter()
            .any(|b| matches!(b, Block::ToolUse { .. } | Block::ToolResult { .. })),
        Content::Text(_) => false,
    })
}

/// The refusal of an `image` block when no Qwen3.8 vision encoder is loaded, on either path.
const IMAGE_REFUSED: &str = "image blocks are not supported on /v1/messages by this server; \
they need Qwen3.8 served with its mmproj (--mmproj)";

/// The offered tools as the model's own template takes them — the OpenAI shape
/// (`{"type": "function", "function": {"name", "description", "parameters"}}`), which is what
/// HF templates are written against — with each schema in the client's key order. `None` when
/// nothing is on offer (`tool_choice: none` included): the template's `tools` stays undefined.
fn native_tools(req: &MessagesRequest) -> Option<minijinja::Value> {
    use minijinja::Value as J;
    if tools_on_offer(req).is_empty() {
        return None;
    }
    let tools = resolved_tools(req);
    let tools = tools.iter().map(|t| {
        let mut function: Vec<(&str, J)> = vec![("name", J::from(t.name.as_str()))];
        if let Some(d) = &t.description {
            function.push(("description", J::from(d.as_str())));
        }
        if let Some(s) = &t.input_schema {
            function.push(("parameters", s.clone()));
        }
        J::from_iter([
            ("type", J::from("function")),
            ("function", J::from_iter(function)),
        ])
    });
    Some(J::from_iter(tools))
}

/// The conversation in the OpenAI shape the model's own template renders — the native path's
/// counterpart of [`flatten`]. `system` leads; an assistant turn keeps its text as `content`,
/// its `tool_use` blocks as `tool_calls` (the Anthropic id, `input` as compact JSON in the
/// client's key order) and its `thinking` as `reasoning_content`; a `tool_result` becomes a
/// `tool` message in place (Anthropic puts results FIRST in a user turn, any text after them —
/// that text becomes the user message that follows the results). A failed result keeps the
/// failure as an `Error: ` prefix: the OpenAI shape has no `is_error`, and the fenced path says
/// "failed with". `Err` carries a client-facing message (an image).
fn native_messages(req: &MessagesRequest) -> Result<Vec<ChatMessage>, String> {
    let mut out = Vec::with_capacity(req.messages.len() + 1);
    out.extend(system_message(req));
    let mut call_names: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
    for m in req
        .messages
        .iter()
        .filter(|m| !only_tool_changes(&m.content))
    {
        let blocks = match &m.content {
            Content::Text(s) => {
                out.push(ChatMessage::text(m.role.clone(), s.clone()));
                continue;
            }
            Content::Blocks(blocks) => blocks,
        };
        let mut turn = Turn::default();
        let mut results = false;
        for b in blocks {
            match b {
                Block::Text { text } => turn.texts.push(text),
                Block::Thinking { thinking } => turn.reasoning.push(thinking),
                Block::ToolUse { id, name, input } => {
                    call_names.insert(id.as_str(), name.as_str());
                    let arguments = if input.is_undefined() || input.is_none() {
                        "{}".to_string()
                    } else {
                        crate::jinja_chat::to_compact_json(input)
                    };
                    turn.calls.push(ToolCall {
                        id: id.clone(),
                        r#type: "function".into(),
                        function: ToolCallFunction {
                            name: name.clone(),
                            arguments,
                        },
                    });
                }
                Block::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => {
                    // What came before the result stays before it.
                    if !turn.is_empty() {
                        out.push(turn.message(&m.role));
                    }
                    let body = content.as_ref().map(text_of).unwrap_or_default();
                    out.push(ChatMessage {
                        role: "tool".into(),
                        content: if *is_error {
                            format!("Error: {body}")
                        } else {
                            body
                        },
                        tool_call_id: Some(tool_use_id.clone()),
                        name: call_names.get(tool_use_id.as_str()).map(|n| n.to_string()),
                        ..Default::default()
                    });
                    results = true;
                }
                // Images never take the native path (`plan_messages`); reaching here means the
                // request is being planned without a vision encoder.
                Block::Image { .. } => return Err(IMAGE_REFUSED.into()),
                // Read by `resolved_tools`, not rendered.
                Block::ToolAddition { .. } | Block::ToolRemoval { .. } | Block::Other => {}
            }
        }
        // A message of results alone ends with them; any other is one turn, even an empty one
        // (as `flatten` keeps it).
        if !turn.is_empty() || !results {
            out.push(turn.message(&m.role));
        }
    }
    Ok(out)
}

/// The project context at the head of a request's FIRST message, when it is a user message of two
/// or more text blocks (and nothing else): every text block but the last, joined as
/// [`native_messages`] joins them. Claude Code's first message is its `<system-reminder>` blocks
/// (AGENTS.md / CLAUDE.md, the date, the environment) and then the user's own words.
///
/// The context stops before the first block that carries the session's state
/// ([`is_session_state`]): the git status. MEASURED 2026-10-09 (that day's last release test,
/// Claude Code 2.1.295 with a developer's own settings): the first message was the AGENTS.md
/// block (47,678 characters), then the git status, then the attribution, then the user's words,
/// and the context anchor ended after the git status (71,657 tokens of 76,452). A new session
/// after any file changed differed at 71,469, before the anchor's snapshot at 71,552, so it
/// resumed from the tools anchor at 56,704 and read ~15K identical tokens again (272 s to the
/// first token on an M4 Max, against ~20 s from the anchor).
fn first_user_context(req: &MessagesRequest) -> Option<String> {
    let first = req.messages.first().filter(|m| m.role == "user")?;
    let Content::Blocks(blocks) = &first.content else {
        return None;
    };
    let texts: Vec<&str> = blocks
        .iter()
        .map(|b| match b {
            Block::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Option<_>>()?;
    if texts.len() < 2 {
        return None;
    }
    let head = &texts[..texts.len() - 1];
    let stable = head
        .iter()
        .position(|t| is_session_state(t))
        .unwrap_or(head.len());
    (stable > 0).then(|| head[..stable].join("\n"))
}

/// A first-message block that changes between two sessions of one project: Claude Code's git
/// status ("a snapshot in time" of the working tree, the branch and the recent commits).
fn is_session_state(text: &str) -> bool {
    text.contains("# gitStatus") || text.contains("This is the git status at the start")
}

/// One turn of [`native_messages`] as its blocks arrive.
#[derive(Default)]
struct Turn<'a> {
    texts: Vec<&'a str>,
    reasoning: Vec<&'a str>,
    calls: Vec<ToolCall>,
}

impl Turn<'_> {
    fn is_empty(&self) -> bool {
        self.texts.is_empty() && self.reasoning.is_empty() && self.calls.is_empty()
    }

    /// The turn as a chat message (texts joined by newlines, as `flatten` joins them); leaves
    /// it empty for what follows.
    fn message(&mut self, role: &str) -> ChatMessage {
        let reasoning = self.reasoning.join("\n");
        let m = ChatMessage {
            role: role.to_string(),
            content: self.texts.join("\n"),
            tool_calls: std::mem::take(&mut self.calls),
            reasoning_content: (!reasoning.is_empty()).then_some(reasoning),
            ..Default::default()
        };
        self.texts.clear();
        self.reasoning.clear();
        m
    }
}

/// Claude Code's per-request billing/attribution header block (see [`system_message`]).
fn is_billing_header(text: &str) -> bool {
    text.trim_start().starts_with("x-anthropic-billing-header")
}

/// `text` without its billing-header line(s) ([`is_billing_header`]), trimmed; `None` when it
/// has none. The `/v1/chat/completions` counterpart of [`system_message`]'s block filter
/// (`http::strip_billing_headers`): a gateway that translates Claude Code's Anthropic request
/// into the OpenAI shape hands the header over INSIDE a system message's text — as its own
/// content part, which `ChatMessage` joins with `\n`, or already joined into one string — so it
/// is found by line here rather than by block. The header is one line.
pub(super) fn strip_billing_header(text: &str) -> Option<String> {
    if !text.lines().any(is_billing_header) {
        return None;
    }
    let kept: String = text
        .split_inclusive('\n')
        .filter(|line| !is_billing_header(line))
        .collect();
    Some(kept.trim().to_string())
}

/// The request's `system` (a string or text blocks) as the leading system message, or `None`
/// when it is absent or empty. Shared by both paths' message lists — and by the prefix anchor,
/// whose sentinel renders must lead with exactly these bytes.
fn system_message(req: &MessagesRequest) -> Option<ChatMessage> {
    // Claude Code prepends `x-anthropic-billing-header: cc_version=2.1.283.<hash>; ...` as the FIRST
    // system block, and the hash changes with the task (measured 2026-09-26: `.e08` vs `.4aa` for
    // two first messages). It is routing/billing metadata for Anthropic's API, not model input —
    // and at the very start of the prompt it made every new session diverge at token ~10, so no
    // prefix, anchor or junction snapshot was ever shared between two Claude Code sessions with
    // different tasks (three sessions: first turn 96-104 s each). another engine drops it the same way
    // (server/api_shapes.py: "Claude Code prepends a billing-header text block; it is not user text").
    let text = match req.system.as_ref()? {
        SystemPrompt::Text(s) if is_billing_header(s) => String::new(),
        SystemPrompt::Text(s) => s.clone(),
        // `blocks_text` without the header block(s)
        SystemPrompt::Blocks(b) => b
            .iter()
            .filter_map(|blk| match blk {
                Block::Text { text } if !is_billing_header(text) => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
    };
    (!text.is_empty()).then(|| ChatMessage::text("system", text))
}

/// The head of [`flatten`]'s list: the fenced tool prompt when tools are on offer, then the
/// request's system message.
fn flatten_head(req: &MessagesRequest, tools: &[Tool]) -> Vec<ChatMessage> {
    let mut out = Vec::with_capacity(2);
    if !tools.is_empty() {
        out.push(ChatMessage::text(
            "system",
            crate::chat::tool_system_prompt(tools),
        ));
    }
    out.extend(system_message(req));
    out
}

/// Flatten an Anthropic conversation into the chat messages the template understands.
/// `Err` carries a client-facing message (an unsupported block).
fn flatten(req: &MessagesRequest) -> Result<Vec<ChatMessage>, String> {
    let mut out = Vec::with_capacity(req.messages.len() + 2);
    out.extend(flatten_head(req, &tools_on_offer(req)));
    // tool_use id -> function name, so a later tool_result can say whose result it is.
    let mut call_names: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
    for m in req
        .messages
        .iter()
        .filter(|m| !only_tool_changes(&m.content))
    {
        // Image blocks are kept with their offset in the joined text (`ChatMessage::image_at`);
        // `prompt_of` refuses them unless a Qwen3.8 vision encoder is loaded.
        let mut images: Vec<String> = Vec::new();
        let mut image_at: Vec<usize> = Vec::new();
        let text = match &m.content {
            Content::Text(s) => s.clone(),
            Content::Blocks(blocks) => {
                let mut parts: Vec<String> = Vec::with_capacity(blocks.len());
                for b in blocks {
                    match b {
                        Block::Image { source } => {
                            let r = source
                                .as_ref()
                                .ok_or_else(|| "image block without a source".to_string())?
                                .reference()?;
                            let joined: usize = parts.iter().map(|p| p.len()).sum::<usize>()
                                + parts.len().saturating_sub(1);
                            image_at.push(joined);
                            images.push(r);
                        }
                        Block::Text { text } => parts.push(text.clone()),
                        Block::ToolUse { id, name, input } => {
                            call_names.insert(id.as_str(), name.as_str());
                            parts.push(crate::chat::render_fenced_call(name, &json_value(input)));
                        }
                        Block::ToolResult {
                            tool_use_id,
                            content,
                            is_error,
                        } => {
                            let name = call_names
                                .get(tool_use_id.as_str())
                                .copied()
                                .unwrap_or("function");
                            let body = content.as_ref().map(text_of).unwrap_or_default();
                            let label = if *is_error { "failed with" } else { "returned" };
                            parts.push(format!("Function `{name}` {label}:\n{body}"));
                        }
                        Block::Thinking { .. }
                        | Block::ToolAddition { .. }
                        | Block::ToolRemoval { .. }
                        | Block::Other => {}
                    }
                }
                parts.join("\n")
            }
        };
        out.push(ChatMessage {
            images,
            image_at,
            ..ChatMessage::text(m.role.clone(), text)
        });
    }
    Ok(out)
}

/// Choose this route's prompt path — `http::plan_chat`'s rule, in this dialect, so one model
/// gets one protocol whichever endpoint an agent speaks:
/// - no tools on offer and no tool history → the hand-written templates ([`flatten`]), exactly
///   as before;
/// - otherwise the model's own template when it teaches a format Arf parses: [`native_messages`]
///   with [`native_tools`], `thinking` read off the rendered prompt;
/// - otherwise, or when that render fails, the fenced-JSON protocol ([`flatten`]).
///
/// Until 2026-09-26 (review) this route had only the last two: a Qwen3.8 agent on
/// `/v1/messages` was taught gemma-4's fenced protocol while `/v1/chat/completions` taught it
/// its own, and the stream could not send prose live (module docs). `Err` is a client-facing
/// message (an image block).
fn plan_messages(
    req: &MessagesRequest,
    tpl: Option<&crate::jinja_chat::ChatTemplate>,
    vars: &crate::jinja_chat::TemplateVars,
) -> Result<Plan, String> {
    if tools_on_offer(req).is_empty() && !has_tool_history(req) {
        return Ok(Plan::HandWritten(PromptPath::Plain));
    }
    // Images: the hand-written templates only (the image block is spliced there), as
    // `http::plan_chat` does with vision on.
    if let Some(tpl) = tpl
        .filter(|t| t.tool_format() != ToolFormat::None)
        .filter(|_| !has_images(req))
    {
        let messages = native_messages(req)?;
        match tpl.render_chat(&messages, native_tools(req), vars) {
            Ok(prompt) => {
                let format = tpl.tool_format();
                log_native_mode_once(format, vars);
                log_tool_path_once(0, || {
                    format!(
                        "/v1/messages: tool request rendered by the model's own chat template \
                         ({} calls) — answer prose streams live (logged once)",
                        format.as_str()
                    )
                });
                return Ok(Plan::Native {
                    thinking: crate::chat::prompt_opens_think(&prompt),
                    prompt,
                    format,
                });
            }
            Err(e) => log_tool_path_once(6, || {
                format!(
                    "chat template failed to render a tool request ({e}) — falling back to the \
                     fenced-JSON protocol (logged once)"
                )
            }),
        }
    }
    log_tool_path_once(1, || {
        "/v1/messages: tool request rendered with the fenced-JSON protocol (no usable chat \
         template) — a streamed answer is settled when the reply ends (logged once)"
            .into()
    });
    Ok(Plan::HandWritten(PromptPath::Fenced))
}

/// A request's prompt, tokenized, and how to read the reply.
struct Prompt {
    ids: Vec<u32>,
    /// The rendered prompt pre-opened `<think>` (Qwen3.8's cue): the reply begins as reasoning.
    thinking: bool,
    /// The tool format the prompt taught — `ToolFormat::None` for the hand-written templates
    /// (plain, or the fenced-JSON protocol).
    format: ToolFormat,
    /// Where the shared system + tools prefix ends in `ids` (`crate::prefix_anchor`), and where
    /// the tools block alone ends (its tools anchor, 2026-09-27); only computed for a request
    /// that will run ([`prompt_of`]), never for a count.
    anchor: crate::prefix_anchor::Anchors,
    /// Qwen3.8 images, preprocessed, in prompt order — encoded by the handler before submit
    /// (`http::qwen_encode`). Empty for every text request.
    images: Vec<crate::qwen_image::QwenMedia>,
    /// A forced call's opening, appended to the prompt (`http::enforce_tool_choice`): the reply
    /// is read as `prefill + reply`. Empty unless `tool_choice` is `any` or `tool`.
    prefill: String,
}

/// Plan + template + tokenize a request. Shared by the two handlers so a count is the count.
/// `Err` is a client-facing message; every failure here is the request's fault (a 400).
fn prompt_ids_of(state: &AppState, req: &MessagesRequest) -> Result<Prompt, String> {
    prompt_of(state, req, false)
}

/// [`prompt_ids_of`], and with `with_anchor` the prompt's prefix anchor too.
///
/// The template variables are those of a `/v1/chat/completions` request that sets no effort:
/// this dialect's `thinking` field is not read (module docs), so the server's defaults apply —
/// `ARF_REASONING_EFFORT`, and `ARF_NO_QWEN_THINK_TEMPLATE` turning thinking off.
/// **Corrected 2026-09-26:** it is read now ([`thinking_vars`]): `disabled` renders both prompt
/// paths non-thinking, `enabled` turns the model's own template's `enable_thinking` on. A
/// request without it gets exactly the settings above. `thinking` below is still read off the
/// rendered prompt, so it follows what was ACTUALLY rendered, whichever path and template.
/// (And, since the review the same day, `output_config.effort` is the request's effort.)
fn prompt_of(state: &AppState, req: &MessagesRequest, with_anchor: bool) -> Result<Prompt, String> {
    if req.messages.is_empty() {
        return Err("messages must not be empty".into());
    }
    if let Some(e) = tool_change_error(req) {
        return Err(e);
    }
    let (effort, vars) = thinking_vars(
        req,
        std::env::var_os("ARF_NO_QWEN_THINK_TEMPLATE").is_some(),
        std::env::var("ARF_REASONING_EFFORT").ok(),
    );
    let with_images = has_images(req);
    if with_images && state.qwen_vision.is_none() {
        return Err(IMAGE_REFUSED.into());
    }
    let mut images = Vec::new();
    let (mut prompt, format) = match plan_messages(req, state.chat_template.as_deref(), &vars)? {
        Plan::Native { prompt, format, .. } => (prompt, format),
        Plan::HandWritten(_) if with_images => {
            let enc = state.qwen_vision.as_ref().expect("checked above");
            let (msgs, imgs) = crate::qwen_image::stage_messages(enc, &flatten(req)?)?;
            let prompt = crate::chat::apply_chat_template_with(&state.tokenizer, &msgs, effort);
            let prompt = crate::qwen_image::expand_sentinels(&prompt, &imgs)?;
            images = imgs;
            (prompt, ToolFormat::None)
        }
        // `effort` is `None` unless `thinking` is `disabled`, and `apply_chat_template` is
        // exactly `apply_chat_template_with(.., None)`: a request without the field renders the
        // bytes it always did. (Corrected 2026-09-26, review: or unless `output_config.effort`
        // names a level — `thinking_vars`. A request with neither field still renders them.)
        Plan::HandWritten(_) => (
            crate::chat::apply_chat_template_with(&state.tokenizer, &flatten(req)?, effort),
            ToolFormat::None,
        ),
    };
    // `any` / `tool`: the reply is begun with the call (`http::enforce_tool_choice`).
    let prefill = enforce_tool_choice(
        req.tool_choice.as_ref(),
        &tools_on_offer(req),
        format,
        false,
    )
    .unwrap_or_default();
    if !prefill.is_empty() {
        prefill_call(&mut prompt, &prefill);
    }
    let thinking = crate::chat::prompt_opens_think(&prompt);
    let ids = encode_prompt(&state.tokenizer, &prompt).map_err(|e| format!("tokenize: {e}"))?;
    // An image request takes no snapshots (the cached blocks would hash pad ids, not pixels).
    let anchor = if with_anchor && !with_images {
        messages_prefix_anchor(state, req, format, &vars, effort, &ids)
    } else {
        Default::default()
    };
    Ok(Prompt {
        ids,
        thinking,
        format,
        anchor,
        images,
        prefill,
    })
}

/// A `/v1/messages` request's PREFIX ANCHOR (`crate::prefix_anchor`) on the path its prompt
/// took — this is the route Claude Code speaks, and the 88-90 s first turn was measured on it:
/// - native (`format` is the template's): the model's own template, the request's system
///   message and [`native_tools`], at `vars`;
/// - hand-written (plain or fenced): [`flatten`]'s head — the fenced tool prompt, then the
///   system message — through the hand-written template AT THE REQUEST'S `effort`.
///
/// **Corrected 2026-09-26 (review):** the hand-written branch rendered the sentinels with
/// `apply_chat_template` (effort `None`, i.e. `ARF_REASONING_EFFORT` or `xhigh`) under the fixed
/// cache key `"messages/hand"`, while the real prompt renders at [`thinking_vars`]' effort.
/// Qwen3.8's hand-written template writes the effort sentence FIRST in the system turn, so a
/// request at `low`/`medium` or with thinking `disabled` shared only `<|im_start|>system\n`
/// with the sentinel render: under 1,024 tokens, no anchor, and nothing logged. And whichever
/// effort rendered first fixed the cached prefix for every other effort. The effort is now
/// both the render's and part of the key (as `/v1/chat/completions`' plain path always had it).
///
/// And the TOOLS ANCHOR (2026-09-27, `crate::prefix_anchor::tools_anchor`) on the same path,
/// when the request offers tools and has a system text — the case it was built for: Claude
/// Code's system text carries the working directory, so its sessions in two directories share
/// only the tools block Qwen3.8 renders ahead of it. Native: the system text is the one system
/// message (`system_lead`); hand-written: [`flatten_head`]'s shape, the fenced tool prompt then
/// the system text.
fn messages_prefix_anchor(
    state: &AppState,
    req: &MessagesRequest,
    format: ToolFormat,
    vars: &crate::jinja_chat::TemplateVars,
    effort: Option<&str>,
    ids: &[u32],
) -> crate::prefix_anchor::Anchors {
    use crate::prefix_anchor::{anchors, system_lead};
    let (cache, tok) = (&state.prefix_anchors, &state.tokenizer);
    if format != ToolFormat::None {
        let Some(tpl) = state.chat_template.as_deref() else {
            return Default::default();
        };
        let tools = native_tools(req);
        let lead: Vec<ChatMessage> = system_message(req).into_iter().collect();
        let shape = ("messages/native", format!("{vars:?}"), tools.clone());
        let render = |m: &[ChatMessage]| tpl.render_chat(m, tools.clone(), vars).ok();
        let tools_lead: Option<&dyn Fn(&str) -> Vec<ChatMessage>> =
            (tools.is_some() && !lead.is_empty()).then_some(&system_lead);
        let mut a = anchors(
            cache,
            tok,
            &lead,
            tools.is_some(),
            &shape,
            render,
            ids,
            tools_lead,
        );
        // CONTEXT ANCHOR: the project context at the head of the first user message (Claude
        // Code's `<system-reminder>` blocks with AGENTS.md / CLAUDE.md) — the deeper anchor
        // takes the system anchor's place (`prefix_anchor::context_anchor`).
        if let Some(context) = first_user_context(req) {
            let with_user_text = |text: &str| {
                let mut m = lead.clone();
                m.push(ChatMessage::text("user", format!("{context}\n{text}")));
                m
            };
            if let Some(n) = crate::prefix_anchor::context_anchor(
                cache,
                tok,
                with_user_text,
                &shape,
                render,
                ids,
                a.system,
            ) {
                a.system = Some(n);
            }
        }
        a
    } else {
        hand_written_anchor(state, req, effort, ids, |m, e| {
            crate::chat::apply_chat_template_with(tok, m, e)
        })
    }
}

/// [`messages_prefix_anchor`]'s hand-written branch, with the template as a parameter so a test
/// can hand it Qwen3.8's thinking template without the process-wide `ARF_QWEN_THINK_TEMPLATE`.
/// `render` is called with the request's `effort` — the sentinels must render as the real prompt
/// did — and the effort is part of the cache key.
fn hand_written_anchor(
    state: &AppState,
    req: &MessagesRequest,
    effort: Option<&str>,
    ids: &[u32],
    render: impl Fn(&[ChatMessage], Option<&str>) -> String,
) -> crate::prefix_anchor::Anchors {
    let tools = tools_on_offer(req);
    let lead = flatten_head(req, &tools);
    // `flatten_head` with the system text replaced: the fenced tool prompt, then `text`.
    let tool_prompt = (!tools.is_empty()).then(|| crate::chat::tool_system_prompt(&tools));
    let lead_with = |text: &str| {
        let mut out: Vec<ChatMessage> = tool_prompt
            .iter()
            .map(|p| ChatMessage::text("system", p.clone()))
            .collect();
        out.push(ChatMessage::text("system", text));
        out
    };
    let tools_lead: Option<&dyn Fn(&str) -> Vec<ChatMessage>> =
        (tool_prompt.is_some() && system_message(req).is_some()).then_some(&lead_with);
    crate::prefix_anchor::anchors(
        &state.prefix_anchors,
        &state.tokenizer,
        &lead,
        !tools.is_empty(),
        ("messages/hand", effort),
        |m: &[ChatMessage]| Some(render(m, effort)),
        ids,
        tools_lead,
    )
}

/// `POST /v1/messages/count_tokens` — the prompt length the model would actually attend to.
pub(super) async fn count_tokens(
    State(state): State<AppState>,
    Json(req): Json<MessagesRequest>,
) -> Response {
    match prompt_ids_of(&state, &req) {
        Ok(p) => Json(json!({"input_tokens": p.ids.len()})).into_response(),
        Err(msg) => bad_request(&msg),
    }
}

/// A finished reply as Anthropic content blocks: a `thinking` block first when the prompt
/// pre-opened `<think>`, then the answer's blocks (`blocks_of`). Tool calls are parsed from the
/// ANSWER only — until 2026-09-26 this route never split `<think>`, so the reasoning and a
/// literal `</think>` reached the visible text and the tool parser scanned the reasoning.
/// `tool_names`: the offered tools (empty = none offered).
fn reply_blocks(
    text: &str,
    thinking: bool,
    tool_names: &[&str],
    finish: FinishReason,
) -> (Vec<ReplyBlock>, &'static str) {
    let (reasoning, answer) = if thinking {
        crate::chat::split_pretag_think(text)
    } else {
        (None, text.to_string())
    };
    let (mut blocks, stop) = blocks_of(&answer, tool_names, finish);
    if let Some(r) = reasoning {
        blocks.insert(0, ReplyBlock::thinking(r));
    }
    (blocks, stop)
}

fn stop_reason(finish: FinishReason) -> &'static str {
    match finish {
        FinishReason::Length => "max_tokens",
        FinishReason::Stop | FinishReason::Evicted => "end_turn",
    }
}

/// The stop reason as sent: `max_tokens` becomes `model_context_window_exceeded` when the limit
/// the reply ran into was the context window's, not the client's (`context_bound`: the request's
/// `max_tokens` was absent or above what the context had left, so [`messages`] clamped it).
/// Anthropic's documented value for "the response filled the model's context window"; a client
/// reading `max_tokens` asks its user to raise the output limit, which would change nothing.
fn window_stop(stop: &'static str, context_bound: bool) -> &'static str {
    if context_bound && stop == "max_tokens" {
        "model_context_window_exceeded"
    } else {
        stop
    }
}

/// Turn a finished reply into Anthropic content blocks and the stop reason. With tools on
/// offer (`tool_names` non-empty), a parsed call becomes a `tool_use` block (preceded by any
/// prose the model wrote). The names are checked as on the OpenAI fenced path
/// (`chat::json_call_from_object`): this route called the unchecked `parse_tool_call` until
/// 2026-09-26 (review), and once that parser scanned EVERY object instead of the first, any
/// JSON with a `name` key anywhere in an answer became a `tool_use`.
fn blocks_of(
    text: &str,
    tool_names: &[&str],
    finish: FinishReason,
) -> (Vec<ReplyBlock>, &'static str) {
    if !tool_names.is_empty() {
        if let Some((calls, lead)) = crate::chat::parse_tool_calls(text, tool_names) {
            let mut blocks = Vec::with_capacity(calls.len() + 1);
            if !lead.is_empty() {
                blocks.push(ReplyBlock::Text { text: lead });
            }
            blocks.extend(calls.iter().map(ReplyBlock::tool_use));
            return (blocks, "tool_use");
        }
    }
    (
        vec![ReplyBlock::Text {
            text: text.to_string(),
        }],
        stop_reason(finish),
    )
}

/// One content block of a finished (non-streamed) reply, serialized in the order its fields are
/// declared (`type` first).
///
/// TYPED, and a `tool_use`'s `input` is the call's own JSON text, inserted verbatim
/// ([`call_input`]). Until 2026-09-26 the blocks were `serde_json::Value`s, and a `Value` object
/// is a SORTED map: a call the model wrote as `{"path", "oldText", "newText"}` went out as
/// `{"newText", "oldText", "path"}`, while the stream sent the model's order in `partial_json`.
/// The client replays `input` in the next request's history and the native path renders it in
/// the order it arrives ([`native_messages`]) — so after a non-streamed turn the replayed call
/// was bytes the model never generated, and the prefix cache missed from that call on.
/// (`serde_json`'s `preserve_order` would have fixed it workspace-wide, and changed every other
/// `Value`'s order with it; `RawValue` changes this one field. `/v1/chat/completions` never had
/// the problem: its `arguments` is a JSON string, sent as the parser wrote it —
/// `http::tests::chat_completions_non_streamed_arguments_keep_the_models_key_order`.)
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ReplyBlock {
    Thinking {
        thinking: String,
        /// Required by the shape; there is no signing here, so it is empty.
        signature: &'static str,
    },
    Text {
        text: String,
    },
    ToolUse {
        /// Unique per call: clients pair a tool_result to its call by this id.
        id: String,
        name: String,
        input: Box<serde_json::value::RawValue>,
    },
}

impl ReplyBlock {
    fn thinking(thinking: String) -> Self {
        ReplyBlock::Thinking {
            thinking,
            signature: "",
        }
    }

    /// A parsed call as a finished `tool_use` block.
    fn tool_use(c: &ToolCall) -> Self {
        ReplyBlock::ToolUse {
            id: gen_id("toolu"),
            name: c.function.name.clone(),
            input: call_input(&c.function.arguments),
        }
    }
}

/// A call's `arguments` (a JSON string on the OpenAI side) as the `input` object Anthropic wants:
/// the parser's text VERBATIM — compact, in the model's key order — when it is a JSON object, and
/// `{}` otherwise. ONE rule for both replies: the stream sends this same text as `partial_json`
/// ([`BlockWriter::tool_use`]). (Before 2026-09-26 the non-streamed reply kept a valid non-object
/// value — `input: [1]` — where the stream sent `{}`; `input` must be an object.)
///
/// **Corrected 2026-09-26 (review):** the first version of this checked the text with
/// `from_str::<Box<RawValue>>` alone, and that is a SCAN, not a parse: serde_json's
/// `ignore_value` has no recursion limit, does not range-check a number and does not check a
/// `\u` escape's code point. So `{"n": 1e400}`, `{"path": "\ud800"}` and 200-deep nesting —
/// reachable when a Hermes or fenced call writes its `arguments` as a string, which
/// `chat::json_call_from_object` passes through unparsed — went out verbatim, and a strict
/// client parser (serde_json, the Rust SDKs' default) rejected the WHOLE response, where the
/// `Value` parse this replaced had sent `{}` for that one input. The gate is that strict parse
/// again; the text that passes it is still sent as written, so the key order is kept.
fn call_input(arguments: &str) -> Box<serde_json::value::RawValue> {
    let text = arguments.trim();
    let raw = serde_json::from_str::<Value>(text)
        .is_ok_and(|v| v.is_object())
        .then(|| serde_json::value::RawValue::from_string(text.to_string()).ok())
        .flatten();
    raw.unwrap_or_else(|| {
        serde_json::value::RawValue::from_string("{}".into()).expect("`{}` is JSON")
    })
}

/// A finished (non-streamed) reply, typed: a `json!` body would turn each `tool_use`'s `input`
/// back into a sorted `Value` ([`ReplyBlock`]). The fields are Anthropic's, in its documented
/// order.
#[derive(Serialize)]
struct MessageReply {
    id: String,
    r#type: &'static str,
    role: &'static str,
    model: String,
    content: Vec<ReplyBlock>,
    stop_reason: &'static str,
    /// The stop sequence the reply ended at (`stop_reason: "stop_sequence"`), else `null`.
    stop_sequence: Option<String>,
    usage: MessageUsage,
}

#[derive(Serialize)]
struct MessageUsage {
    input_tokens: usize,
    output_tokens: usize,
}

/// A finished reply to a prompt in the model's own tool format, as content blocks: read by the
/// splitter and `settle_native` exactly as `/v1/chat/completions` reads its non-streamed native
/// reply (`http::read_native_reply`), so the two dialects — and a streamed and a non-streamed
/// reply — agree on what is reasoning, prose and call, and a call the budget cut off is not a
/// `tool_use`. Same shape as [`reply_blocks`]: a `thinking` block when there was reasoning, the
/// prose as a `text` block (always one when there is no call), then the `tool_use` blocks.
fn native_blocks(
    text: &str,
    format: ToolFormat,
    thinking: bool,
    tools: &[Tool],
    finish: FinishReason,
) -> (Vec<ReplyBlock>, &'static str) {
    let r = read_native_reply(text, format, thinking, tools, finish);
    let mut blocks = Vec::new();
    if let Some(t) = r.reasoning {
        blocks.push(ReplyBlock::thinking(t));
    }
    if !r.content.is_empty() || r.calls.is_none() {
        blocks.push(ReplyBlock::Text { text: r.content });
    }
    match r.calls {
        Some(calls) => {
            blocks.extend(calls.iter().map(ReplyBlock::tool_use));
            (blocks, "tool_use")
        }
        None => (blocks, stop_reason(finish)),
    }
}

/// Send one named SSE event; false when the client has gone.
async fn send_event(
    tx: &tokio::sync::mpsc::Sender<Result<Event, std::convert::Infallible>>,
    name: &'static str,
    data: Value,
) -> bool {
    tx.send(Ok(Event::default().event(name).data(data.to_string())))
        .await
        .is_ok()
}

/// Stream a reply without tools token by token as content blocks: with the thinking template
/// the reply opens inside `<think>`, so it streams as a `thinking` block (index 0) until the
/// `</think>` token, then the answer as a `text` block (index 1), its leading whitespace dropped
/// as the non-streamed split trims. Without it, one `text` block. Every opened block is closed.
/// Returns the stop reason and the token count, or `None` when the client went away.
///
/// The text goes through `stop`: held back while it could begin a stop string, cut at
/// the first one — the reason is then `stop_sequence`, and `stop.matched()` the string. Each
/// piece keeps the part it was generated in, so held-back reasoning stays in the `thinking` block.
async fn stream_text_blocks(
    tok: &Tokenizer,
    rx: &mut tokio::sync::mpsc::Receiver<StreamItem>,
    tx: &tokio::sync::mpsc::Sender<Result<Event, std::convert::Infallible>>,
    thinking: bool,
    stop: &mut StopFilter<ThinkPart>,
) -> Option<(&'static str, usize)> {
    let think_end = tok.token_to_id("</think>");
    let mut in_think = thinking && think_end.is_some();
    // The block open now is `thinking` (it closes at the released `</think>`).
    let mut in_think_block = in_think;
    let mut answer_started = !in_think;
    let mut index = 0usize;
    let open = |index: usize, block: Value| json!({"type": "content_block_start", "index": index, "content_block": block});
    let first = if in_think {
        open(0, json!({"type": "thinking", "thinking": ""}))
    } else {
        open(0, json!({"type": "text", "text": ""}))
    };
    if !send_event(tx, "content_block_start", first).await {
        return None;
    }
    let mut n = 0usize;
    let mut finish = FinishReason::Length;
    loop {
        let (pieces, done) = match rx.recv().await {
            Some(StreamItem::Token(token_id, _)) => {
                n += 1;
                let part = ThinkPart::of(token_id, think_end, &mut in_think);
                (stop.push(&decode(tok, &[token_id]), part), false)
            }
            Some(StreamItem::Done(reason)) => {
                finish = reason;
                (stop.finish(), true)
            }
            None => (stop.finish(), true),
        };
        for (mut text, part) in pieces {
            let delta = match part {
                ThinkPart::ThinkEnd => {
                    if in_think_block {
                        in_think_block = false;
                        let stop = json!({"type": "content_block_stop", "index": index});
                        if !(send_event(tx, "content_block_stop", stop).await
                            && send_event(
                                tx,
                                "content_block_start",
                                open(index + 1, json!({"type": "text", "text": ""})),
                            )
                            .await)
                        {
                            return None;
                        }
                        index += 1;
                    }
                    continue;
                }
                ThinkPart::Reasoning => json!({"type": "thinking_delta", "thinking": text}),
                ThinkPart::Answer => {
                    if !answer_started {
                        text = text.trim_start().to_string();
                        answer_started = !text.is_empty();
                        if !answer_started {
                            continue;
                        }
                    }
                    json!({"type": "text_delta", "text": text})
                }
            };
            let delta = json!({"type": "content_block_delta", "index": index, "delta": delta});
            if !send_event(tx, "content_block_delta", delta).await {
                return None;
            }
        }
        if done {
            break;
        }
    }
    let reason = if stop.matched().is_some() {
        "stop_sequence"
    } else {
        stop_reason(finish)
    };
    let stop = json!({"type": "content_block_stop", "index": index});
    send_event(tx, "content_block_stop", stop)
        .await
        .then_some((reason, n))
}

/// The content blocks of a streamed reply as they open and close: at most one open at a time,
/// indexed 0, 1, 2… in the order they open, each closed exactly once. Every method returns false
/// (or `None`) when the client has gone.
struct BlockWriter<'a> {
    tx: &'a tokio::sync::mpsc::Sender<Result<Event, std::convert::Infallible>>,
    /// The open `thinking` / `text` block: its type and index.
    current: Option<(&'static str, usize)>,
    /// The index the next block gets — also how many were opened.
    next: usize,
    /// A `text` block was opened.
    text: bool,
}

impl BlockWriter<'_> {
    async fn close(&mut self) -> bool {
        match self.current.take() {
            Some((_, index)) => {
                let stop = json!({"type": "content_block_stop", "index": index});
                send_event(self.tx, "content_block_stop", stop).await
            }
            None => true,
        }
    }

    /// Close the open block and start `block`; its index.
    async fn start(&mut self, block: Value) -> Option<usize> {
        if !self.close().await {
            return None;
        }
        let index = self.next;
        self.next += 1;
        let start = json!({"type": "content_block_start", "index": index, "content_block": block});
        send_event(self.tx, "content_block_start", start)
            .await
            .then_some(index)
    }

    /// Reasoning into a `thinking` block, answer text into a `text` block — the open one when it
    /// is of that type, else a new one.
    async fn piece(&mut self, p: Piece) -> bool {
        let (kind, block, delta) = match p {
            Piece::Reasoning(t) => (
                "thinking",
                json!({"type": "thinking", "thinking": ""}),
                json!({"type": "thinking_delta", "thinking": t}),
            ),
            Piece::Content(t) => (
                "text",
                json!({"type": "text", "text": ""}),
                json!({"type": "text_delta", "text": t}),
            ),
        };
        let index = match self.current {
            Some((k, index)) if k == kind => index,
            _ => {
                let Some(index) = self.start(block).await else {
                    return false;
                };
                self.current = Some((kind, index));
                self.text |= kind == "text";
                index
            }
        };
        let delta = json!({"type": "content_block_delta", "index": index, "delta": delta});
        send_event(self.tx, "content_block_delta", delta).await
    }

    /// A complete call as one `tool_use` block: opened with an empty `input`, its arguments in
    /// one `input_json_delta`, closed. The arguments go out as the parser wrote them (compact,
    /// the model's key order); ones that are not a JSON object become `{}`, as the non-streamed
    /// path turns them into `input: {}`.
    /// **Corrected 2026-09-26:** the non-streamed path turned only INVALID JSON into `{}` — a
    /// valid non-object stayed as it was — and sorted an object's keys. Both replies now take
    /// [`call_input`], the same text.
    async fn tool_use(&mut self, call: &ToolCall) -> bool {
        let input = call_input(&call.function.arguments);
        let partial = input.get();
        let open = json!({
            "type": "tool_use",
            // Unique per call: clients pair a tool_result to its call by this id.
            "id": gen_id("toolu"),
            "name": call.function.name,
            "input": {},
        });
        let Some(index) = self.start(open).await else {
            return false;
        };
        let delta = json!({"type": "content_block_delta", "index": index,
            "delta": {"type": "input_json_delta", "partial_json": partial}});
        let stop = json!({"type": "content_block_stop", "index": index});
        send_event(self.tx, "content_block_delta", delta).await
            && send_event(self.tx, "content_block_stop", stop).await
    }
}

/// Stream a reply to a request that offered tools, LIVE, the way the OpenAI tool stream does
/// (`http::chat_stream_tools_response`: the same [`ReplySplitter`], the same end-of-reply
/// decision `http::settle_tool_stream`) — reasoning as a `thinking` block while it arrives, the
/// answer's prose as a `text` block up to the start of a call in `format`, then each complete
/// call as a `tool_use` block ([`BlockWriter::tool_use`]). A block opens when its first text
/// arrives. A call the token budget cut off is never a `tool_use`: its text follows the prose in
/// the `text` block and the stop reason stays `max_tokens` (`tool_parse::settle_native`). A reply
/// with no call streams the same events as [`stream_text_blocks`].
/// **Corrected 2026-09-26 (review):** not the same EVENTS — the same blocks, carrying the same
/// text up to the whitespace around each part. The splitter trims it (the reasoning's before
/// `</think>` included) as the non-streamed reply does, where the no-tools stream sends every
/// token; it joins what it held back into one delta; and on the fenced protocol the whole
/// answer is ONE delta when the reply ends. The claim was tested on qwen-xml alone, a format
/// the handler never passed then, and it missed a reply that thinks and then answers
/// nothing: that one sent no `text` block at all. Now a reply that makes no call always
/// ends with a `text` block (empty when the answer was), as the non-streamed reply does and the
/// no-tools stream sends at `</think>`; only a reply cut off inside its reasoning is the
/// `thinking` block alone, as the no-tools stream sends it.
///
/// `format` is the prompt's tool format; `ToolFormat::None` (the fenced-JSON protocol, which is
/// what this route's prompt uses) buffers the answer whole — a fenced call is only recognisable
/// once its JSON closes — and settles it when the reply ends, while the reasoning still streams.
/// (Since 2026-09-26, review, the fenced protocol is only this route's FALLBACK: a template that
/// teaches a format is rendered and its format passed here — [`plan_messages`].)
///
/// Returns the stop reason and the token count, or `None` when the client went away. That is
/// noticed while waiting for the next token too (a buffered call sends nothing for a long time):
/// the caller then returns and drops `rx`, which evicts the job.
///
/// History: until 2026-09-26 a streamed reply with tools was drained to the end and its blocks
/// were sent at once — the first delta at 9.58 s vs 1.04 s without tools (measured).
async fn stream_tool_blocks(
    tok: &Tokenizer,
    rx: &mut tokio::sync::mpsc::Receiver<StreamItem>,
    tx: &tokio::sync::mpsc::Sender<Result<Event, std::convert::Infallible>>,
    format: ToolFormat,
    thinking: bool,
    tools: &[Tool],
    stop: &mut StopFilter,
    prefill: &str,
) -> Option<(&'static str, usize)> {
    let mut splitter = ReplySplitter::new(format, thinking, format != ToolFormat::None);
    let mut dec = IncrementalDecoder::default();
    let mut blocks = BlockWriter {
        tx,
        current: None,
        next: 0,
        text: false,
    };
    // A forced call's opening (`prefill`), already in the prompt: read ahead of the reply, by the
    // splitter only — `stop` sees generated text alone, so no stop sequence matches inside it.
    for p in splitter.push(prefill) {
        if !blocks.piece(p).await {
            return None;
        }
    }
    let mut n = 0usize;
    let mut finish = FinishReason::Length;
    loop {
        let item = tokio::select! {
            _ = tx.closed() => return None,
            item = rx.recv() => item,
        };
        match item {
            Some(StreamItem::Token(token_id, _)) => {
                n += 1;
                for p in splitter.push(&crate::stop::text_of(
                    stop.push(&dec.push(tok, token_id), ()),
                )) {
                    if !blocks.piece(p).await {
                        return None;
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
    let mut rest = crate::stop::text_of(stop.push(&dec.finish(tok), ()));
    rest += &crate::stop::text_of(stop.finish());
    if stop.matched().is_some() {
        finish = FinishReason::Stop;
    }
    let tail = splitter.push(&rest);
    // Read before `finish` consumes the splitter: was the reply cut off before `</think>`?
    let cut_in_reasoning = splitter.in_reasoning();
    let end = splitter.finish();
    for p in tail.into_iter().chain(end.pieces) {
        if !blocks.piece(p).await {
            return None;
        }
    }
    let (calls, extra) = settle_tool_stream(format, &end.gap, &end.buffered, tools, finish);
    if !extra.is_empty() && !blocks.piece(Piece::Content(extra)).await {
        return None;
    }
    for call in &calls {
        if !blocks.tool_use(call).await {
            return None;
        }
    }
    // No call and no `text` block — the answer was empty, or nothing was generated at all: one
    // empty `text` block, as the non-streamed path returns. Cut off inside the reasoning: the
    // `thinking` block alone, as `stream_text_blocks` sends it. (Until 2026-09-26, review, only
    // "nothing sent at all" got the empty block, so thinking + an empty answer had none.)
    if calls.is_empty() && !blocks.text && (!cut_in_reasoning || blocks.next == 0) {
        let index = blocks.start(json!({"type": "text", "text": ""})).await?;
        blocks.current = Some(("text", index));
    }
    let reason = if !calls.is_empty() {
        "tool_use"
    } else if stop.matched().is_some() {
        "stop_sequence"
    } else {
        stop_reason(finish)
    };
    blocks.close().await.then_some((reason, n))
}

/// The `stop_sequence` of a streamed reply: the string matched, when that is why it stopped.
fn sequence_of<K: Copy>(stop: &str, filter: &StopFilter<K>) -> Option<String> {
    (stop == "stop_sequence")
        .then(|| filter.matched().map(str::to_string))
        .flatten()
}

/// Drain a job to completion: the token ids and why it stopped.
async fn drain(rx: &mut tokio::sync::mpsc::Receiver<StreamItem>) -> (Vec<u32>, FinishReason) {
    let mut ids = Vec::new();
    let mut finish = FinishReason::Length;
    while let Some(item) = rx.recv().await {
        match item {
            StreamItem::Token(id, _) => ids.push(id),
            StreamItem::Done(reason) => {
                finish = reason;
                break;
            }
        }
    }
    (ids, finish)
}

/// The visible reply: thought-channel markers and Muse's `to=self` channel removed, exactly as
/// the OpenAI handler does for a non-streamed reply.
fn visible_text(text: &str) -> String {
    let text = crate::chat::strip_thought(text);
    crate::chat::split_muse_channels(&text).1
}

/// `POST /v1/messages`.
pub(super) async fn messages(
    State(state): State<AppState>,
    Json(req): Json<MessagesRequest>,
) -> Response {
    let Prompt {
        ids: prompt_ids,
        thinking,
        format,
        anchor: prefix_anchor,
        images,
        prefill,
    } = match prompt_of(&state, &req, true) {
        Ok(p) => p,
        Err(msg) => return bad_request(&msg),
    };
    let image_prompt = match crate::http::qwen_encode(&state, &images, &prompt_ids).await {
        Ok(p) => p,
        Err(msg) => return bad_request(&format!("image: {msg}")),
    };
    // `none` withdraws the tools (`tools_on_offer`); `any` and `tool` were enforced by prefill
    // in `prompt_of`, as on `/v1/chat/completions`.
    // `thinking`: what is not honoured is said once, as for `tool_choice` ([`thinking_vars`]).
    if let Some(n) = req.thinking_budget() {
        log_thinking_once(0, || {
            format!(
                "thinking.budget_tokens {n} is accepted but NOT enforced — Arf has no thinking \
                 budget; the reasoning runs at the server's effort and ends on </think> or \
                 max_tokens (logged once)"
            )
        });
    }
    if req.enable_thinking() == Some(false) && thinking {
        log_thinking_once(1, || {
            "thinking {type: disabled} was requested, but the chat template pre-opened <think> \
             anyway (it has no enable_thinking switch) — the reasoning comes back as a thinking \
             block rather than as answer text (logged once)"
                .into()
        });
    }
    if req.thinking_type() == Some("adaptive") {
        log_thinking_once(2, || {
            format!(
                "thinking {{type: adaptive}} is read as the server's default thinking mode \
                 (this request's prompt {} <think>; ARF_NO_QWEN_THINK_TEMPLATE turns it off, \
                 {{type: enabled}} forces it on) (logged once)",
                if thinking { "opens" } else { "does not open" }
            )
        });
    }
    // `disabled` wins over any effort (`thinking_vars`); otherwise say what the effort became.
    if let Some(e) = req
        .effort_field()
        .filter(|_| req.enable_thinking() != Some(false))
    {
        match anthropic_effort(e) {
            Some(mapped) => log_thinking_once(3, || {
                format!(
                    "output_config.effort {e:?} honoured as reasoning effort {mapped:?} — \
                     read by Qwen3.8's templates; others have no effort (first request that \
                     set one; logged once)"
                )
            }),
            None => log_thinking_once(4, || {
                format!(
                    "output_config.effort {e:?} is not a level Arf knows (low, medium, high, \
                     max) — the server's default effort applies (logged once)"
                )
            }),
        }
    }
    let tools = tools_on_offer(&req);
    let has_tools = !tools.is_empty();
    let stops = StopStrings::new(req.stop_sequences.clone().unwrap_or_default());
    let input_tokens = prompt_ids.len();
    let max_tokens = match context_budget(input_tokens, req.max_tokens, state.max_tokens_cap) {
        Ok(n) => n,
        Err(_) => return prompt_too_long(input_tokens, state.max_tokens_cap),
    };
    // The clamp, not the client, set the limit: a reply cut there filled the context window.
    let context_bound = req.max_tokens.is_none_or(|n| n > max_tokens);
    let model_id = req.model.clone().unwrap_or_else(|| state.model_id.clone());
    let mut sampling = match build_sampling(req.temperature, req.top_p, req.top_k, None, None) {
        Ok(s) => s,
        Err((status, msg)) => return anth_error(status, "invalid_request_error", &msg),
    };
    if image_prompt.is_some() {
        crate::qwen_image::force_greedy(&mut sampling);
    }
    let mut rx = match submit_job(
        &state,
        prompt_ids,
        max_tokens,
        sampling,
        false,
        0,
        None,
        Vec::new(),
        Vec::new(),
        image_prompt,
        prefix_anchor,
        &stops,
    ) {
        Ok(rx) => rx,
        Err((status, msg)) => return anth_error(status, "overloaded_error", &msg),
    };
    let id = gen_id("msg");

    if !req.stream {
        let (ids, finish) = drain(&mut rx).await;
        // The reply up to its first stop sequence.
        let decoded = decode(&state.tokenizer, &ids);
        let (text, matched) = stops.cut(&decoded);
        let finish = if matched.is_some() {
            FinishReason::Stop
        } else {
            finish
        };
        let (content, stop) = if has_tools && format != ToolFormat::None {
            // The model's own format: read as the stream reads it (`native_blocks`). (No gemma
            // thought-channel or Muse cleanup, as on `/v1/chat/completions`: those templates
            // teach no format Arf parses, so their requests never take this path.)
            // The prefill was never generated: the stop sequences cut the generated text
            // only, and the forced call is read whole as `prefill + reply`.
            let text = format!("{prefill}{text}");
            native_blocks(&text, format, thinking, &tools, finish)
        } else {
            let names: Vec<&str> = tools.iter().map(|t| t.function.name.as_str()).collect();
            let text = format!("{prefill}{}", visible_text(text));
            reply_blocks(&text, thinking, &names, finish)
        };
        // A stop sequence is the reason unless the reply still made a call.
        let (stop, stop_sequence) = match matched {
            Some(m) if stop != "tool_use" => ("stop_sequence", Some(m.to_string())),
            _ => (stop, None),
        };
        // Typed, not `json!`: a `Value` body would sort each `tool_use` input's keys again.
        return Json(MessageReply {
            id,
            r#type: "message",
            role: "assistant",
            model: model_id,
            content,
            stop_reason: window_stop(stop, context_bound),
            stop_sequence,
            usage: MessageUsage {
                input_tokens,
                output_tokens: ids.len(),
            },
        })
        .into_response();
    }

    // Streaming. One input item can become several events (the last token closes a block, a
    // message and the stream), so a task writes events into a channel the SSE body reads.
    let (tx, out) = tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(64);
    tokio::spawn(async move {
        let start = json!({"type": "message_start", "message": {
            "id": id, "type": "message", "role": "assistant", "model": model_id,
            "content": [], "stop_reason": null, "stop_sequence": null,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        }});
        if !send_event(&tx, "message_start", start).await {
            return;
        }
        let (stop, output_tokens, stop_sequence) = if has_tools {
            // Live, as `/v1/chat/completions` streams a tool reply. This route's tool prompt is
            // the fenced-JSON protocol (module docs), so the reply is read as that endpoint reads
            // its fenced fallback: the reasoning streams, the answer is settled at the end. (Until
            // 2026-09-26 the whole reply was drained first — `stream_tool_blocks`' history.)
            // Corrected 2026-09-26 (review): the prompt is the model's own template when it
            // teaches a format (`plan_messages`), and `format` is that format — prose streams
            // live up to the call. `ToolFormat::None` only on the fenced fallback, where the
            // answer is still settled at the end, as on that endpoint. The literal `None` that
            // stood here made the prose wait for the end of the turn on every model.
            let mut filter = StopFilter::new(stops);
            let r = stream_tool_blocks(
                &state.tokenizer,
                &mut rx,
                &tx,
                format,
                thinking,
                &tools,
                &mut filter,
                &prefill,
            )
            .await;
            match r {
                Some((stop, n)) => (stop, n, sequence_of(stop, &filter)),
                None => return, // client went away; dropping `rx` evicts the job
            }
        } else {
            let mut filter = StopFilter::new(stops);
            match stream_text_blocks(&state.tokenizer, &mut rx, &tx, thinking, &mut filter).await {
                Some((stop, n)) => (stop, n, sequence_of(stop, &filter)),
                None => return, // client went away; dropping `rx` evicts the job
            }
        };
        let done = json!({"type": "message_delta",
            "delta": {"stop_reason": window_stop(stop, context_bound), "stop_sequence": stop_sequence},
            "usage": {"output_tokens": output_tokens}});
        if send_event(&tx, "message_delta", done).await {
            let _ = send_event(&tx, "message_stop", json!({"type": "message_stop"})).await;
        }
    });
    Sse::new(ReceiverStream::new(out))
        .keep_alive(KeepAlive::default())
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::super::tests::app_state;
    use super::*;

    /// The project context ends before Claude Code's git status (`first_user_context`): a new
    /// session after a file changed shares the AGENTS.md block, not the status after it.
    #[test]
    fn the_project_context_stops_before_the_git_status() {
        let req = |status: &str| {
            parse(
                json!({"max_tokens": 8, "messages": [{"role": "user", "content": [
                {"type": "text", "text": "<system-reminder>\nContents of AGENTS.md\n</system-reminder>"},
                {"type": "text", "text": format!("<system-reminder>\n# gitStatus\nThis is the git status at the start of the conversation.\nStatus:\n{status}\n</system-reminder>")},
                {"type": "text", "text": "<system-reminder>\nAttribution\n</system-reminder>"},
                {"type": "text", "text": "hey"}]}]}),
            )
        };
        let (a, b) = (req("(clean)"), req("?? calc.py"));
        assert_eq!(
            first_user_context(&a).as_deref(),
            Some("<system-reminder>\nContents of AGENTS.md\n</system-reminder>")
        );
        assert_eq!(first_user_context(&a), first_user_context(&b));
        // Without a git status every block but the user's words is context, as before.
        let plain = parse(
            json!({"max_tokens": 8, "messages": [{"role": "user", "content": [
            {"type": "text", "text": "ctx one"}, {"type": "text", "text": "ctx two"},
            {"type": "text", "text": "hey"}]}]}),
        );
        assert_eq!(
            first_user_context(&plain).as_deref(),
            Some("ctx one\nctx two")
        );
        // A git status first leaves no context to anchor.
        let first = parse(
            json!({"max_tokens": 8, "messages": [{"role": "user", "content": [
            {"type": "text", "text": "# gitStatus\n..."}, {"type": "text", "text": "hey"}]}]}),
        );
        assert_eq!(first_user_context(&first), None);
    }

    fn parse(body: Value) -> MessagesRequest {
        serde_json::from_value(body).expect("a valid Messages request")
    }

    /// Stop strings (e), `/v1/messages`: a reply cut at a stop sequence says so —
    /// `stop_reason: "stop_sequence"` and `stop_sequence` the string matched — non-streamed and in
    /// the stream's `message_delta`; the text stops before it. Without a match `stop_sequence`
    /// stays `null`.
    #[tokio::test]
    async fn messages_report_the_stop_sequence() {
        let ids = vec![0u32, 1, 2, 3, 4, 7];
        let request = |stream: bool, stops: Value| {
            parse(
                json!({"max_tokens": 64, "stream": stream, "stop_sequences": stops,
                "messages": [{"role": "user", "content": "go"}]}),
            )
        };
        // Non-streamed.
        let (state, jobs) = app_state(None);
        let actor = super::super::tests::play_with_stops(jobs, ids.clone());
        let resp = messages(State(state), Json(request(false, json!(["\n", "zz"])))).await;
        assert_eq!(actor.join().unwrap(), 5);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["stop_reason"], "stop_sequence", "{v}");
        assert_eq!(v["stop_sequence"], "\n");
        assert_eq!(
            v["content"],
            json!([{"type": "text", "text": "We need to read."}])
        );
        // Streamed, matched and not.
        for (stops, text, reason, seq) in [
            (
                json!(["\n"]),
                "We need to read.",
                "stop_sequence",
                json!("\n"),
            ),
            (
                json!(["zz"]),
                "We need to read.\nHi",
                "max_tokens",
                Value::Null,
            ),
        ] {
            let (state, jobs) = app_state(None);
            let actor = super::super::tests::play_with_stops(jobs, ids.clone());
            let resp = messages(State(state), Json(request(true, stops))).await;
            let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap();
            actor.join().unwrap();
            let ev = events_of(std::str::from_utf8(&body).unwrap());
            let streamed: String = ev
                .iter()
                .filter_map(|e| e["delta"]["text"].as_str())
                .collect();
            assert_eq!(streamed, text);
            let delta = ev.iter().find(|e| e["type"] == "message_delta").unwrap();
            assert_eq!(delta["delta"]["stop_reason"], reason, "{delta}");
            assert_eq!(delta["delta"]["stop_sequence"], seq);
        }
    }

    /// Finished-reply blocks as the JSON a client parses — through the real serializer, so a
    /// `tool_use` input is read back from the bytes that go out.
    fn values(blocks: &[ReplyBlock]) -> Vec<Value> {
        blocks
            .iter()
            .map(|b| serde_json::from_str(&serde_json::to_string(b).unwrap()).unwrap())
            .collect()
    }

    #[test]
    fn string_and_block_content_both_flatten_to_one_message_each() {
        let req = parse(json!({
            "model": "x", "max_tokens": 16, "system": "be brief",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
                {"role": "user", "content": [
                    {"type": "text", "text": "one"}, {"type": "text", "text": "two"}]},
            ],
        }));
        let m = flatten(&req).unwrap();
        let got: Vec<(&str, &str)> = m
            .iter()
            .map(|m| (m.role.as_str(), m.content.as_str()))
            .collect();
        assert_eq!(
            got,
            vec![
                ("system", "be brief"),
                ("user", "hello"),
                ("assistant", "hi"),
                ("user", "one\ntwo")
            ]
        );
    }

    #[test]
    fn claude_codes_billing_header_block_is_not_model_input() {
        // the header's hash changes with the task; left in, no two sessions shared a prefix
        let mk = |h: &str| {
            parse(json!({
                "system": [{"type": "text", "text": format!("x-anthropic-billing-header: cc_version=2.1.283.{h}; cc_entrypoint=sdk-cli;")},
                           {"type": "text", "text": "You are Claude Code."}],
                "messages": [{"role": "user", "content": "hi"}],
            }))
        };
        let (a, b) = (flatten(&mk("e08")).unwrap(), flatten(&mk("4aa")).unwrap());
        assert_eq!(a[0].content, "You are Claude Code.");
        assert_eq!(a[0].content, b[0].content);
        let only = parse(
            json!({"system": "x-anthropic-billing-header: x;", "messages": [{"role": "user", "content": "hi"}]}),
        );
        assert!(flatten(&only).unwrap().iter().all(|m| m.role != "system"));
    }

    #[test]
    fn system_as_blocks_and_unknown_blocks_are_tolerated() {
        let req = parse(json!({
            "system": [{"type": "text", "text": "a"}, {"type": "text", "text": "b", "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "assistant", "content": [
                {"type": "thinking", "thinking": "hmm", "signature": "s"},
                {"type": "text", "text": "answer"}]}],
        }));
        let m = flatten(&req).unwrap();
        assert_eq!(m[0].content, "a\nb");
        assert_eq!(m[1].content, "answer"); // the thinking block is dropped, not an error
    }

    #[test]
    fn tool_history_replays_as_the_models_own_call_and_a_named_result() {
        let req = parse(json!({
            "tools": [{"name": "read_file", "description": "read", "input_schema": {"type": "object"}}],
            "messages": [
                {"role": "user", "content": "open it"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a.rs"}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "fn main() {}"}]}]},
            ],
        }));
        let m = flatten(&req).unwrap();
        assert_eq!(m[0].role, "system"); // the tool prompt leads
        assert!(m[0].content.contains("read_file"));
        // The replayed call parses with the same parser that reads the model's live output.
        let (calls, _) = crate::chat::parse_tool_call(&m[2].content).expect("a parseable call");
        assert_eq!(calls[0].function.name, "read_file");
        assert_eq!(m[3].role, "user");
        assert_eq!(m[3].content, "Function `read_file` returned:\nfn main() {}");
    }

    /// `input_schema` and a `tool_use`'s `input` became order-preserving values (2026-09-26,
    /// review, for the native path). The fenced path must render the SAME bytes it did when
    /// they were `serde_json::Value`s: here the old values are built the old way — parsed
    /// straight into `serde_json::Value` — and the prompts compared, over unsorted keys, ints,
    /// floats, a negative, a null and nesting.
    #[test]
    fn fenced_prompt_bytes_are_unchanged_by_the_order_preserving_fields() {
        let schema = r#"{"type": "object", "properties": {"path": {"type": "string"},
            "line": {"type": "integer", "minimum": -1, "default": 1.5}, "all": {"type": "null"}},
            "required": ["path"]}"#;
        let input =
            r#"{"path": "a.rs", "line": 3, "ratio": 0.25, "opts": {"z": null, "a": [1, 2.0]}}"#;
        let body = format!(
            r#"{{"tools": [{{"name": "read", "description": "Read", "input_schema": {schema}}}],
            "messages": [{{"role": "assistant", "content": [
                {{"type": "tool_use", "id": "t1", "name": "read", "input": {input}}}]}}]}}"#
        );
        let req: MessagesRequest = serde_json::from_str(&body).unwrap();
        let m = flatten(&req).unwrap();
        let old_tools = [Tool {
            r#type: "function".into(),
            function: ToolFunction {
                name: "read".into(),
                description: Some("Read".into()),
                parameters: Some(serde_json::from_str(schema).unwrap()),
            },
        }];
        assert_eq!(m[0].content, crate::chat::tool_system_prompt(&old_tools));
        let old_input: Value = serde_json::from_str(input).unwrap();
        assert_eq!(
            m[1].content,
            crate::chat::render_fenced_call("read", &old_input)
        );
    }

    #[test]
    fn a_failed_tool_result_says_so_and_an_image_is_kept_for_the_vision_path() {
        let failed = parse(json!({"messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "nope", "content": "boom", "is_error": true}]}]}));
        assert_eq!(
            flatten(&failed).unwrap()[0].content,
            "Function `function` failed with:\nboom"
        );
        // 2026-09-27: an image is no longer refused by `flatten` — it is carried, with its offset,
        // to `prompt_of`, which refuses it unless a Qwen3.8 vision encoder is loaded.
        let image = parse(json!({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "What is"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}},
            {"type": "text", "text": "this?"}]}]}));
        let m = flatten(&image).unwrap();
        assert_eq!(m[0].content, "What is\nthis?");
        assert_eq!(m[0].images, vec!["data:image/png;base64,AA==".to_string()]);
        assert_eq!(m[0].image_at, vec![7]);
    }

    #[test]
    fn a_tool_call_becomes_a_tool_use_block_with_an_object_input() {
        let reply = "Let me look.\n```json\n{\"function\": \"read_file\", \"arguments\": {\"path\": \"a.rs\"}}\n```";
        let (blocks, stop) = blocks_of(reply, &["read_file"], FinishReason::Stop);
        let blocks = values(&blocks);
        assert_eq!(stop, "tool_use");
        assert_eq!(blocks[0], json!({"type": "text", "text": "Let me look."}));
        assert_eq!(blocks[1]["type"], "tool_use");
        assert_eq!(blocks[1]["name"], "read_file");
        assert_eq!(blocks[1]["input"], json!({"path": "a.rs"})); // an object, not a JSON string
        assert!(blocks[1]["id"].as_str().unwrap().starts_with("toolu"));
    }

    /// With the thinking template the reasoning becomes a `thinking` block, and a call-shaped
    /// object INSIDE the reasoning is not a call — only the answer is parsed.
    #[test]
    fn thinking_is_split_off_and_tools_come_from_the_answer_only() {
        let reply =
            "Maybe {\"function\": \"rm_rf\", \"arguments\": {}}? No — read first.\n</think>\n\n\
```json\n{\"function\": \"read_file\", \"arguments\": {\"path\": \"a.rs\"}}\n```";
        let (blocks, stop) = reply_blocks(reply, true, &["read_file"], FinishReason::Stop);
        let blocks = values(&blocks);
        assert_eq!(stop, "tool_use");
        assert_eq!(blocks[0]["type"], "thinking");
        assert!(blocks[0]["thinking"]
            .as_str()
            .unwrap()
            .ends_with("read first."));
        assert_eq!(blocks[1]["type"], "tool_use");
        assert_eq!(blocks[1]["name"], "read_file");
        assert_eq!(blocks.len(), 2);
        // No `</think>` in any visible block.
        assert!(!blocks.iter().any(|b| b.to_string().contains("</think>")));
        // Without the thinking template the text passes through as before.
        let (plain, stop) = reply_blocks("hello", false, &[], FinishReason::Stop);
        assert_eq!(stop, "end_turn");
        assert_eq!(
            values(&plain),
            vec![json!({"type": "text", "text": "hello"})]
        );
    }

    #[test]
    fn without_tools_the_same_text_is_just_text_and_stop_reasons_map() {
        let reply = "```json\n{\"function\": \"x\", \"arguments\": {}}\n```";
        let (blocks, stop) = blocks_of(reply, &[], FinishReason::Stop);
        assert_eq!(stop, "end_turn");
        assert_eq!(
            values(&blocks),
            vec![json!({"type": "text", "text": reply})]
        );
        assert_eq!(stop_reason(FinishReason::Length), "max_tokens");
    }

    /// The offered names are checked: JSON data with a `name` key after another object is an
    /// answer, not a `tool_use` (the unchecked parser made it a call to a tool named "web").
    #[test]
    fn json_data_in_an_answer_is_not_a_tool_use() {
        let reply =
            "Example {\"id\": 1}, then the service: {\"name\": \"web\", \"image\": \"nginx\"}";
        let (blocks, stop) = blocks_of(reply, &["read_file"], FinishReason::Stop);
        assert_eq!(stop, "end_turn");
        assert_eq!(
            values(&blocks),
            vec![json!({"type": "text", "text": reply})]
        );
    }

    /// A prompt that fills the context: Anthropic's envelope AND Anthropic's wording, which is
    /// what its clients match to compact and retry.
    #[tokio::test]
    async fn prompt_too_long_uses_anthropics_wording() {
        let resp = prompt_too_long(70_000, 65_536);
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            v,
            json!({"type": "error", "error": {"type": "invalid_request_error",
                "message": "prompt is too long: 70000 tokens > 65536 maximum"}})
        );
    }

    /// Run `stream_text_blocks` over `ids` then `Done(finish)`; the events' JSON, in order, and
    /// what it returned.
    /// A stream's stop-string filter with no stop strings: every piece passes straight through.
    fn no_stops<K: Copy>() -> StopFilter<K> {
        StopFilter::new(StopStrings::default())
    }

    async fn text_blocks(
        ids: &[u32],
        finish: FinishReason,
        thinking: bool,
    ) -> (Vec<Value>, Option<(&'static str, usize)>) {
        let tok = super::super::tests::test_tokenizer();
        let (itx, mut irx) = tokio::sync::mpsc::channel(64);
        for &id in ids {
            itx.send(StreamItem::Token(id, None)).await.unwrap();
        }
        itx.send(StreamItem::Done(finish)).await.unwrap();
        let (tx, out) = tokio::sync::mpsc::channel(64);
        let r = stream_text_blocks(&tok, &mut irx, &tx, thinking, &mut no_stops()).await;
        drop(tx);
        let resp = Sse::new(ReceiverStream::new(out)).into_response();
        let data = super::super::tests::sse_data(resp).await;
        let events = data
            .iter()
            .map(|d| serde_json::from_str(d).unwrap())
            .collect();
        (events, r)
    }

    /// The streamed thinking split: `thinking` block 0 up to the `</think>` token, then `text`
    /// block 1 (its leading newlines dropped), each opened and closed once, in order.
    #[tokio::test]
    async fn streamed_thinking_then_text_blocks_are_indexed_in_order() {
        // "We" " need" "</think>" "\n\n" "Hi"
        let (ev, r) = text_blocks(&[0, 1, 5, 6, 7], FinishReason::Stop, true).await;
        assert_eq!(r, Some(("end_turn", 5)));
        let shape: Vec<(String, u64)> = ev
            .iter()
            .map(|e| {
                (
                    e["type"].as_str().unwrap().to_string(),
                    e["index"].as_u64().unwrap(),
                )
            })
            .collect();
        let expect = [
            ("content_block_start", 0),
            ("content_block_delta", 0),
            ("content_block_delta", 0),
            ("content_block_stop", 0),
            ("content_block_start", 1),
            ("content_block_delta", 1),
            ("content_block_stop", 1),
        ];
        assert_eq!(
            shape,
            expect.map(|(t, i)| (t.to_string(), i)).to_vec(),
            "{ev:?}"
        );
        assert_eq!(ev[0]["content_block"]["type"], "thinking");
        assert_eq!(
            ev[1]["delta"],
            json!({"type": "thinking_delta", "thinking": "We"})
        );
        assert_eq!(ev[4]["content_block"]["type"], "text");
        assert_eq!(ev[5]["delta"], json!({"type": "text_delta", "text": "Hi"}));
    }

    /// Cut off by `max_tokens` inside `<think>`: the one thinking block is still closed, and the
    /// stop reason is `max_tokens`. Without the thinking template it is one text block.
    #[tokio::test]
    async fn streamed_reply_cut_inside_thinking_closes_its_block() {
        let (ev, r) = text_blocks(&[0, 1], FinishReason::Length, true).await;
        assert_eq!(r, Some(("max_tokens", 2)));
        assert_eq!(ev.len(), 4);
        assert_eq!(ev[0]["content_block"]["type"], "thinking");
        assert_eq!(ev[3], json!({"type": "content_block_stop", "index": 0}));
        let (ev, _) = text_blocks(&[7], FinishReason::Stop, false).await;
        assert_eq!(ev[0]["content_block"]["type"], "text");
        assert_eq!(ev[1]["delta"], json!({"type": "text_delta", "text": "Hi"}));
        assert_eq!(ev.len(), 3);
    }

    // ------------------------------------------------------------------------
    // The live tool stream (`stream_tool_blocks`). Token ids are `http::tests::test_tokenizer`'s:
    // 0 "We", 1 " need", 2 " to", 3 " read.", 4 "\n", 5 "</think>", 6 "\n\n", 7 "Hi",
    // 8..=14 the pieces of a qwen-xml `read(path=calc.py)` call, 17..=19 a fenced-JSON call.
    // ------------------------------------------------------------------------

    /// `<tool_call>\n<function=read>\n<parameter=path>\ncalc.py\n</parameter>\n</function>\n</tool_call>`
    const XML_CALL: [u32; 13] = [8, 4, 9, 4, 10, 4, 11, 4, 12, 4, 13, 4, 14];

    fn read_tool() -> Vec<Tool> {
        vec![Tool {
            r#type: "function".into(),
            function: ToolFunction {
                name: "read".into(),
                description: None,
                parameters: Some(
                    json!({"type": "object", "properties": {"path": {"type": "string"}}}),
                ),
            },
        }]
    }

    /// The events in an SSE body (one frame or the whole of it), each checked to carry its own
    /// type as its `event:` name — the Anthropic convention clients dispatch on.
    fn events_of(sse: &str) -> Vec<Value> {
        let mut out = Vec::new();
        let mut name = None;
        for line in sse.lines() {
            if let Some(n) = line.strip_prefix("event: ") {
                name = Some(n.to_string());
            } else if let Some(d) = line.strip_prefix("data: ") {
                let v: Value = serde_json::from_str(d).unwrap();
                assert_eq!(name.take().as_deref(), v["type"].as_str(), "{v}");
                out.push(v);
            }
        }
        out
    }

    /// `(type, index)` of each event — the stream's shape.
    fn shape(ev: &[Value]) -> Vec<(String, u64)> {
        ev.iter()
            .map(|e| {
                (
                    e["type"].as_str().unwrap().to_string(),
                    e["index"].as_u64().unwrap(),
                )
            })
            .collect()
    }

    fn expect(s: &[(&str, u64)]) -> Vec<(String, u64)> {
        s.iter().map(|&(t, i)| (t.to_string(), i)).collect()
    }

    /// Every block opened once, closed once, after it opened, before the next one opened.
    fn assert_blocks_balanced(ev: &[Value]) {
        let mut open: Option<u64> = None;
        let mut next = 0;
        for e in ev {
            let i = e["index"].as_u64().unwrap();
            match e["type"].as_str().unwrap() {
                "content_block_start" => {
                    assert_eq!((open, i), (None, next), "{ev:?}");
                    open = Some(i);
                    next += 1;
                }
                "content_block_delta" => assert_eq!(open, Some(i), "{ev:?}"),
                "content_block_stop" => {
                    assert_eq!(open, Some(i), "{ev:?}");
                    open = None;
                }
                other => panic!("unexpected event {other}"),
            }
        }
        assert_eq!(open, None, "a block was left open: {ev:?}");
    }

    /// Run `stream_tool_blocks` over `ids` then `Done(finish)`; the events, in order, and what it
    /// returned.
    async fn tool_blocks(
        ids: &[u32],
        finish: FinishReason,
        format: ToolFormat,
        thinking: bool,
        tools: &[Tool],
    ) -> (Vec<Value>, Option<(&'static str, usize)>) {
        let tok = super::super::tests::test_tokenizer();
        let (itx, mut irx) = tokio::sync::mpsc::channel(64);
        for &id in ids {
            itx.send(StreamItem::Token(id, None)).await.unwrap();
        }
        itx.send(StreamItem::Done(finish)).await.unwrap();
        let (tx, out) = tokio::sync::mpsc::channel(256);
        let r = stream_tool_blocks(
            &tok,
            &mut irx,
            &tx,
            format,
            thinking,
            tools,
            &mut no_stops(),
            "",
        )
        .await;
        drop(tx);
        let resp = Sse::new(ReceiverStream::new(out)).into_response();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        (events_of(std::str::from_utf8(&body).unwrap()), r)
    }

    /// (1) Thinking, prose, one qwen-xml call: `thinking` block 0, `text` block 1, `tool_use`
    /// block 2, stop reason `tool_use`. LIVE: the task is fed the reasoning and the prose only,
    /// and the prose's `text_delta` must arrive while the call has not been generated yet; only
    /// then are the call's tokens sent.
    #[tokio::test]
    async fn streamed_tool_call_after_live_thinking_and_prose() {
        let (itx, mut irx) = tokio::sync::mpsc::channel(64);
        let (tx, out) = tokio::sync::mpsc::channel(64);
        let task = tokio::spawn(async move {
            let tok = super::super::tests::test_tokenizer();
            let tools = read_tool();
            stream_tool_blocks(
                &tok,
                &mut irx,
                &tx,
                ToolFormat::QwenXml,
                true,
                &tools,
                &mut no_stops(),
                "",
            )
            .await
        });
        let mut body = Sse::new(ReceiverStream::new(out))
            .into_response()
            .into_body()
            .into_data_stream();
        // "We" " need" " to" " read." "</think>" "\n\n" "Hi" "\n" — no call token yet.
        for id in [0u32, 1, 2, 3, 5, 6, 7, 4] {
            itx.send(StreamItem::Token(id, None)).await.unwrap();
        }
        let mut ev: Vec<Value> = Vec::new();
        while !ev.iter().any(|e| e["delta"]["type"] == "text_delta") {
            let frame = tokio::time::timeout(std::time::Duration::from_secs(5), body.next())
                .await
                .expect("the prose must stream before the call is generated")
                .expect("the stream is open")
                .unwrap();
            ev.extend(events_of(&String::from_utf8_lossy(&frame)));
        }
        // Everything so far went out live; the text block is still open (the "\n" is held back:
        // it may precede a call marker).
        assert_eq!(
            shape(&ev),
            expect(&[
                ("content_block_start", 0),
                ("content_block_delta", 0),
                ("content_block_delta", 0),
                ("content_block_delta", 0),
                ("content_block_delta", 0),
                ("content_block_stop", 0),
                ("content_block_start", 1),
                ("content_block_delta", 1),
            ]),
            "{ev:?}"
        );
        // Now the call, and the end of the reply.
        for id in XML_CALL {
            itx.send(StreamItem::Token(id, None)).await.unwrap();
        }
        itx.send(StreamItem::Done(FinishReason::Stop))
            .await
            .unwrap();
        while let Some(frame) = body.next().await {
            ev.extend(events_of(&String::from_utf8_lossy(&frame.unwrap())));
        }
        assert_eq!(task.await.unwrap(), Some(("tool_use", 8 + XML_CALL.len())));
        assert_eq!(
            shape(&ev[8..]),
            expect(&[
                ("content_block_stop", 1),
                ("content_block_start", 2),
                ("content_block_delta", 2),
                ("content_block_stop", 2),
            ]),
            "{ev:?}"
        );
        assert_blocks_balanced(&ev);
        assert_eq!(
            ev[0]["content_block"],
            json!({"type": "thinking", "thinking": ""})
        );
        let thought: String = ev[1..5]
            .iter()
            .map(|e| e["delta"]["thinking"].as_str().unwrap())
            .collect();
        assert_eq!(thought, "We need to read.");
        assert_eq!(ev[6]["content_block"], json!({"type": "text", "text": ""}));
        assert_eq!(ev[7]["delta"], json!({"type": "text_delta", "text": "Hi"}));
        let open = &ev[9]["content_block"];
        assert_eq!(open["type"], "tool_use");
        assert_eq!(open["name"], "read");
        assert_eq!(open["input"], json!({}));
        assert!(open["id"].as_str().unwrap().starts_with("toolu"), "{open}");
        assert_eq!(
            ev[10]["delta"],
            json!({"type": "input_json_delta", "partial_json": "{\"path\":\"calc.py\"}"})
        );
        // No markup reached a visible block.
        assert!(!ev.iter().any(|e| e.to_string().contains("<tool_call>")));
    }

    /// (2) Two calls in one reply: two `tool_use` blocks, distinct ids, after the prose.
    #[tokio::test]
    async fn streamed_two_calls_are_two_tool_use_blocks() {
        let mut ids = vec![7u32, 4];
        ids.extend(XML_CALL);
        ids.push(4);
        ids.extend(XML_CALL);
        let (ev, r) = tool_blocks(
            &ids,
            FinishReason::Stop,
            ToolFormat::QwenXml,
            false,
            &read_tool(),
        )
        .await;
        assert_eq!(r, Some(("tool_use", ids.len())));
        assert_eq!(
            shape(&ev),
            expect(&[
                ("content_block_start", 0),
                ("content_block_delta", 0),
                ("content_block_stop", 0),
                ("content_block_start", 1),
                ("content_block_delta", 1),
                ("content_block_stop", 1),
                ("content_block_start", 2),
                ("content_block_delta", 2),
                ("content_block_stop", 2),
            ]),
            "{ev:?}"
        );
        assert_blocks_balanced(&ev);
        assert_eq!(ev[1]["delta"], json!({"type": "text_delta", "text": "Hi"}));
        let (a, b) = (&ev[3]["content_block"], &ev[6]["content_block"]);
        assert_eq!(a["type"], "tool_use");
        assert_eq!(b["type"], "tool_use");
        assert_ne!(a["id"], b["id"], "clients pair results to calls by id");
        for d in [&ev[4], &ev[7]] {
            assert_eq!(d["delta"]["partial_json"], "{\"path\":\"calc.py\"}");
        }
    }

    /// (3) A call cut off by `max_tokens` is never a `tool_use`: its text follows the prose in
    /// the text block (as `/v1/chat/completions` sends it as content), the stop reason is
    /// `max_tokens`, and every opened block is closed.
    #[tokio::test]
    async fn streamed_cut_call_is_text_and_stops_on_max_tokens() {
        // "We" " need" "</think>" "\n\n" "Hi" "\n" then a call cut inside its value.
        let ids = [0u32, 1, 5, 6, 7, 4, 8, 4, 9, 4, 10, 4, 11];
        let (ev, r) = tool_blocks(
            &ids,
            FinishReason::Length,
            ToolFormat::QwenXml,
            true,
            &read_tool(),
        )
        .await;
        assert_eq!(r, Some(("max_tokens", ids.len())));
        assert!(
            !ev.iter().any(|e| e.to_string().contains("tool_use")),
            "{ev:?}"
        );
        assert_blocks_balanced(&ev);
        assert_eq!(
            shape(&ev),
            expect(&[
                ("content_block_start", 0),
                ("content_block_delta", 0),
                ("content_block_delta", 0),
                ("content_block_stop", 0),
                ("content_block_start", 1),
                ("content_block_delta", 1),
                ("content_block_delta", 1),
                ("content_block_stop", 1),
            ]),
            "{ev:?}"
        );
        let text: String = ev[5..7]
            .iter()
            .map(|e| e["delta"]["text"].as_str().unwrap())
            .collect();
        assert_eq!(
            text,
            "Hi\n<tool_call>\n<function=read>\n<parameter=path>\ncalc.py"
        );
        // Cut before any answer at all: the thinking block alone, closed.
        let (ev, r) = tool_blocks(
            &[0, 1],
            FinishReason::Length,
            ToolFormat::QwenXml,
            true,
            &read_tool(),
        )
        .await;
        assert_eq!(r, Some(("max_tokens", 2)));
        assert_blocks_balanced(&ev);
        assert_eq!(ev.len(), 4);
    }

    /// The blocks of a stream as `(type, text)`: each block's deltas joined.
    fn folded(ev: &[Value]) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = Vec::new();
        for e in ev {
            match e["type"].as_str().unwrap() {
                "content_block_start" => out.push((
                    e["content_block"]["type"].as_str().unwrap().to_string(),
                    String::new(),
                )),
                "content_block_delta" => {
                    let d = &e["delta"];
                    let t = d["text"]
                        .as_str()
                        .or(d["thinking"].as_str())
                        .or(d["partial_json"].as_str())
                        .unwrap();
                    out.last_mut().unwrap().1.push_str(t);
                }
                _ => {}
            }
        }
        out
    }

    /// (4) Tools offered, no call made. On qwen-xml, with no whitespace to trim: the SAME events
    /// as the no-tools stream. In general, on BOTH formats — qwen-xml and the fenced protocol,
    /// the one the handler takes on fallback — the same blocks carrying the same text up to the
    /// whitespace around each part (the splitter trims it as the non-streamed reply does; the
    /// no-tools stream sends every token), and the same stop reason and count. The first
    /// version of this test ran qwen-xml alone, a format the handler never passed then.
    #[tokio::test]
    async fn streamed_reply_without_a_call_matches_the_no_tools_stream() {
        for (ids, thinking) in [(&[0u32, 1, 5, 6, 7][..], true), (&[7u32][..], false)] {
            let (with_tools, r) = tool_blocks(
                ids,
                FinishReason::Stop,
                ToolFormat::QwenXml,
                thinking,
                &read_tool(),
            )
            .await;
            let (without, r0) = text_blocks(ids, FinishReason::Stop, thinking).await;
            assert_eq!(with_tools, without);
            assert_eq!(r, r0);
            assert_eq!(r, Some(("end_turn", ids.len())));
        }
        let cases: [(&[u32], bool); 7] = [
            (&[0, 1, 5, 6, 7], true), // "We need" | "Hi"
            (&[7, 0], false),         // "HiWe", two tokens
            (&[7, 4, 7], false),      // "Hi\nHi"
            (&[0, 4, 5, 6, 7], true), // "We\n" | "\n\nHi": whitespace around </think>
            (&[4, 0, 5, 7, 4], true), // "\nWe" | "Hi\n"
            (&[0, 1, 5], true),       // thinking, then an empty answer
            (&[0, 5, 6], true),       // thinking, then only whitespace
        ];
        for format in [ToolFormat::QwenXml, ToolFormat::None] {
            for (ids, thinking) in cases {
                let (with_tools, r) =
                    tool_blocks(ids, FinishReason::Stop, format, thinking, &read_tool()).await;
                let (without, r0) = text_blocks(ids, FinishReason::Stop, thinking).await;
                assert_blocks_balanced(&with_tools);
                // Trimmed on both sides: the fenced settle keeps an answer's trailing
                // whitespace (as `/v1/chat/completions` sends it on that fallback).
                let trimmed = |ev: &[Value]| -> Vec<(String, String)> {
                    folded(ev)
                        .into_iter()
                        .map(|(t, s)| (t, s.trim().to_string()))
                        .collect()
                };
                assert_eq!(
                    trimmed(&with_tools),
                    trimmed(&without),
                    "{format:?} {ids:?}"
                );
                assert_eq!(r, r0, "{format:?} {ids:?}");
            }
        }
        // Where they differ on purpose: qwen-xml streams the answer token by token; the fenced
        // protocol sends it as ONE delta when the reply ends (a fenced call is only
        // recognisable once its JSON closes).
        let deltas = |ev: &[Value]| {
            ev.iter()
                .filter(|e| e["delta"]["type"] == "text_delta")
                .count()
        };
        let (native, _) = tool_blocks(
            &[7, 0],
            FinishReason::Stop,
            ToolFormat::QwenXml,
            false,
            &read_tool(),
        )
        .await;
        assert_eq!(deltas(&native), 2, "{native:?}");
        let (fenced, _) = tool_blocks(
            &[7, 0],
            FinishReason::Stop,
            ToolFormat::None,
            false,
            &read_tool(),
        )
        .await;
        assert_eq!(deltas(&fenced), 1, "{fenced:?}");
    }

    /// Thinking, then an empty answer (or only whitespace), no call: the `thinking` block, then
    /// ONE empty `text` block — on both formats, as the no-tools stream sends it and the
    /// non-streamed reply returns it. The first live version ended on the thinking block alone.
    /// Cut off inside the reasoning is still the thinking block alone (test 3).
    #[tokio::test]
    async fn streamed_thinking_then_an_empty_answer_ends_with_an_empty_text_block() {
        for format in [ToolFormat::QwenXml, ToolFormat::None] {
            for ids in [&[0u32, 1, 5][..], &[0, 1, 5, 6][..]] {
                let (ev, r) =
                    tool_blocks(ids, FinishReason::Stop, format, true, &read_tool()).await;
                assert_eq!(r, Some(("end_turn", ids.len())));
                assert_blocks_balanced(&ev);
                assert_eq!(
                    folded(&ev),
                    vec![
                        ("thinking".to_string(), "We need".to_string()),
                        ("text".to_string(), String::new()),
                    ],
                    "{format:?} {ids:?}: {ev:?}"
                );
            }
        }
        // The non-streamed reply, both paths: the same two blocks.
        let (fenced, stop) = reply_blocks("We need</think>", true, &["read"], FinishReason::Stop);
        assert_eq!(stop, "end_turn");
        let (native, stop2) = native_blocks(
            "We need</think>",
            ToolFormat::QwenXml,
            true,
            &read_tool(),
            FinishReason::Stop,
        );
        assert_eq!(stop2, "end_turn");
        for blocks in [values(&fenced), values(&native)] {
            assert_eq!(
                blocks,
                vec![
                    json!({"type": "thinking", "thinking": "We need", "signature": ""}),
                    json!({"type": "text", "text": ""}),
                ]
            );
        }
    }

    /// The fenced-JSON protocol — the one this route's prompt uses: the reasoning streams, the
    /// answer is settled at the end — a call becomes a `tool_use` block, anything else a `text`
    /// block — exactly as `/v1/chat/completions` reads its fenced fallback. An empty reply is one
    /// empty text block, as the non-streamed path returns.
    #[tokio::test]
    async fn streamed_fenced_protocol_streams_thinking_and_settles_the_answer() {
        // "We" " need" "</think>" "\n\n" "```json" "\n" {…read a.py…} "\n" "```"
        let ids = [0u32, 1, 5, 6, 17, 4, 18, 4, 19];
        let (ev, r) = tool_blocks(
            &ids,
            FinishReason::Stop,
            ToolFormat::None,
            true,
            &read_tool(),
        )
        .await;
        assert_eq!(r, Some(("tool_use", ids.len())));
        assert_blocks_balanced(&ev);
        assert_eq!(
            shape(&ev),
            expect(&[
                ("content_block_start", 0),
                ("content_block_delta", 0),
                ("content_block_delta", 0),
                ("content_block_stop", 0),
                ("content_block_start", 1),
                ("content_block_delta", 1),
                ("content_block_stop", 1),
            ]),
            "{ev:?}"
        );
        assert_eq!(ev[4]["content_block"]["name"], "read");
        assert_eq!(ev[5]["delta"]["partial_json"], "{\"path\":\"a.py\"}");
        // A plain answer on the fenced protocol: one text block, sent when the reply ends.
        let (ev, r) = tool_blocks(
            &[7],
            FinishReason::Stop,
            ToolFormat::None,
            false,
            &read_tool(),
        )
        .await;
        assert_eq!(r, Some(("end_turn", 1)));
        assert_eq!(ev[1]["delta"], json!({"type": "text_delta", "text": "Hi"}));
        assert_eq!(ev.len(), 3);
        // Nothing generated: one empty text block, opened and closed.
        let (ev, r) = tool_blocks(
            &[],
            FinishReason::Stop,
            ToolFormat::None,
            false,
            &read_tool(),
        )
        .await;
        assert_eq!(r, Some(("end_turn", 0)));
        assert_eq!(
            ev,
            vec![
                json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
                json!({"type": "content_block_stop", "index": 0}),
            ]
        );
    }

    /// Cancellation: the client goes away mid-turn, while the task waits for the next token. The
    /// task must stop and drop the job's receiver — what makes the actor evict the sequence — as
    /// the OpenAI tool stream does (`http::tests::tools_stream_drops_the_job_when_the_client_leaves`).
    #[tokio::test]
    async fn streamed_tools_drop_the_job_when_the_client_leaves() {
        let (itx, mut irx) = tokio::sync::mpsc::channel(64);
        let (tx, out) = tokio::sync::mpsc::channel(64);
        let task = tokio::spawn(async move {
            let tok = super::super::tests::test_tokenizer();
            stream_tool_blocks(
                &tok,
                &mut irx,
                &tx,
                ToolFormat::QwenXml,
                true,
                &[],
                &mut no_stops(),
                "",
            )
            .await
        });
        itx.send(StreamItem::Token(0, None)).await.unwrap();
        let mut body = Sse::new(ReceiverStream::new(out))
            .into_response()
            .into_body()
            .into_data_stream();
        let mut ev = Vec::new();
        while ev.len() < 2 {
            let frame = tokio::time::timeout(std::time::Duration::from_secs(5), body.next())
                .await
                .expect("the first reasoning delta must stream")
                .expect("the stream is open")
                .unwrap();
            ev.extend(events_of(&String::from_utf8_lossy(&frame)));
        }
        assert_eq!(
            ev[1]["delta"],
            json!({"type": "thinking_delta", "thinking": "We"})
        );
        // The job never finishes on its own. The client leaves.
        drop(body);
        tokio::time::timeout(std::time::Duration::from_secs(5), itx.closed())
            .await
            .expect("the stream task must drop the job's receiver once the client is gone");
        assert_eq!(task.await.unwrap(), None);
    }

    /// `tool_choice: none` withdraws the tools: no tool prompt, nothing parsed. `auto`, `any` and
    /// `tool` keep them on offer (the last two were not enforced until 2026-10-04; they are now
    /// enforced by prefill — `messages_forced_tool_choice_is_prefilled`).
    #[test]
    fn tool_choice_none_withdraws_the_tools() {
        let with = |choice: Value| {
            parse(json!({
                "tools": [{"name": "read_file", "input_schema": {"type": "object"}}],
                "tool_choice": choice,
                "messages": [{"role": "user", "content": "hi"}],
            }))
        };
        let none = with(json!({"type": "none"}));
        assert!(tools_on_offer(&none).is_empty());
        let m = flatten(&none).unwrap();
        assert_eq!(m.len(), 1, "no tool prompt: {m:?}");
        assert_eq!(m[0].role, "user");
        for choice in [
            json!({"type": "auto"}),
            json!({"type": "any"}),
            json!({"type": "tool", "name": "read_file"}),
        ] {
            let req = with(choice);
            assert_eq!(tools_on_offer(&req)[0].function.name, "read_file");
            assert!(flatten(&req).unwrap()[0].content.contains("read_file"));
        }
        // Absent: the tools are on offer, as before.
        let req = parse(json!({"tools": [{"name": "read_file"}],
            "messages": [{"role": "user", "content": "hi"}]}));
        assert_eq!(tools_on_offer(&req).len(), 1);
    }

    // ------------------------------------------------------------------------
    // The prompt path (`plan_messages`), the native reply, and the handler end to end.
    // Bodies whose key order matters are parsed from TEXT: `json!` builds a
    // `serde_json::Value`, which has already sorted them.
    // ------------------------------------------------------------------------

    fn qwen38() -> crate::jinja_chat::ChatTemplate {
        super::super::tests::qwen38()
    }

    /// The template variables of a request that sets nothing, env-independent.
    fn vars() -> crate::jinja_chat::TemplateVars {
        template_vars(None, None, false, None)
    }

    /// Tools, a system prompt and tool history, rendered by Qwen3.8's own template: the tools as
    /// OpenAI-shaped functions in the client's key order (`path` before `line`), the call
    /// replayed in the model's format with ITS argument order, the turn's reasoning replayed,
    /// the result as a `<tool_response>`, the text after it as the next user turn — and none of
    /// the fenced protocol.
    #[test]
    fn plan_messages_renders_tools_and_history_with_the_models_own_template() {
        let body = r#"{"system": "Be brief.",
            "tools": [{"name": "read", "description": "Read a file", "input_schema":
                {"type": "object", "properties": {"path": {"type": "string"}, "line": {"type": "integer"}}}}],
            "messages": [
                {"role": "user", "content": "open a.rs"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "Read it first.", "signature": "s"},
                    {"type": "text", "text": "Reading."},
                    {"type": "tool_use", "id": "toolu_1", "name": "read", "input": {"path": "a.rs", "line": 3}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "fn main() {}"}]},
                    {"type": "text", "text": "now fix it"}]}]}"#;
        let req: MessagesRequest = serde_json::from_str(body).unwrap();
        let (prompt, thinking) = match plan_messages(&req, Some(&qwen38()), &vars()) {
            Ok(Plan::Native {
                prompt,
                format,
                thinking,
            }) => {
                assert_eq!(format, ToolFormat::QwenXml);
                (prompt, thinking)
            }
            other => panic!("expected the native path, got {other:?}"),
        };
        assert!(
            prompt.contains(
                "<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"read\", \
\"description\": \"Read a file\", \"parameters\": {\"type\": \"object\", \"properties\": \
{\"path\": {\"type\": \"string\"}, \"line\": {\"type\": \"integer\"}}}}}\n</tools>"
            ),
            "{prompt}"
        );
        assert!(
            prompt.contains("</IMPORTANT>\n\nBe brief.<|im_end|>"),
            "{prompt}"
        );
        assert!(
            prompt.contains(
                "<|im_start|>assistant\n<think>\nRead it first.\n</think>\n\nReading.\n\n\
<tool_call>\n<function=read>\n<parameter=path>\na.rs\n</parameter>\n<parameter=line>\n3\n\
</parameter>\n</function>\n</tool_call><|im_end|>"
            ),
            "{prompt}"
        );
        assert!(
            prompt.contains(
                "<|im_start|>user\n<tool_response>\nfn main() {}\n</tool_response><|im_end|>\n\
<|im_start|>user\nnow fix it<|im_end|>"
            ),
            "{prompt}"
        );
        assert!(!prompt.contains("```json"), "the fenced protocol: {prompt}");
        // `thinking` is read off the prompt: the template's default cue pre-opens `<think>`.
        assert!(
            prompt.ends_with("<|im_start|>assistant\n<think>\n"),
            "{prompt}"
        );
        assert!(thinking);
    }

    /// The offered tool names (`tools_on_offer`) of a request.
    fn offered_names(req: &MessagesRequest) -> Vec<String> {
        tools_on_offer(req)
            .into_iter()
            .map(|t| t.function.name)
            .collect()
    }

    /// Mid-conversation tool changes: a `defer_loading` tool is withheld until a `tool_addition`
    /// offers it; `tool_addition` / `tool_removal` blocks in `role: "system"` messages apply in
    /// order; an added tool is appended; a tool can be defined in the message itself; removing a
    /// tool that is not offered changes nothing. A `tool_reference` to an undeclared tool is
    /// Anthropic's `tool_reference_unresolved` 400. Until 2026-10-04 every declared tool was
    /// offered from the first turn and the change blocks were ignored.
    #[test]
    fn tool_additions_and_removals_resolve_the_offered_tools_in_order() {
        let reference = |kind: &str, name: &str| json!({"type": kind, "tool": {"type": "tool_reference", "name": name}});
        let request = |changes: Vec<Value>| {
            let mut messages = vec![json!({"role": "user", "content": "go"})];
            for c in changes {
                messages.push(json!({"role": "system", "content": [c]}));
                messages.push(json!({"role": "assistant", "content": "ok"}));
                messages.push(json!({"role": "user", "content": "next"}));
            }
            parse(json!({"tools": [
                {"name": "read", "input_schema": {"type": "object"}},
                {"name": "grep", "defer_loading": true, "input_schema": {"type": "object"}},
                {"name": "edit", "input_schema": {"type": "object"}}],
                "messages": messages}))
        };
        assert_eq!(offered_names(&request(vec![])), ["read", "edit"]);
        let added = request(vec![reference("tool_addition", "grep")]);
        assert_eq!(offered_names(&added), ["read", "edit", "grep"]);
        assert_eq!(
            offered_names(&request(vec![
                reference("tool_addition", "grep"),
                reference("tool_removal", "read"),
            ])),
            ["edit", "grep"]
        );
        // Order matters: removed, then offered again — appended.
        assert_eq!(
            offered_names(&request(vec![
                reference("tool_removal", "read"),
                reference("tool_addition", "read"),
            ])),
            ["edit", "read"]
        );
        // Added twice, or removed when not offered: no change.
        assert_eq!(
            offered_names(&request(vec![
                reference("tool_addition", "read"),
                reference("tool_removal", "grep"),
                reference("tool_removal", "nope"),
            ])),
            ["read", "edit"]
        );
        // Defined in the message itself (`tool_definition`), schema and all.
        let defined = request(vec![json!({"type": "tool_addition", "tool": {
            "type": "tool_definition", "definition": {"name": "db_query",
                "description": "Run SQL", "input_schema": {"type": "object",
                    "properties": {"sql": {"type": "string"}}}}}})]);
        let tools = tools_on_offer(&defined);
        assert_eq!(tools.len(), 3);
        assert_eq!(tools[2].function.name, "db_query");
        assert_eq!(tools[2].function.description.as_deref(), Some("Run SQL"));
        assert_eq!(
            tools[2].function.parameters,
            Some(json!({"type": "object", "properties": {"sql": {"type": "string"}}}))
        );
        // An MCP reference is ignored, not a 400.
        let mcp = request(vec![json!({"type": "tool_addition", "tool": {
            "type": "tool_reference", "mcp_tool_reference": {"server_name": "c", "name": "e"}}})]);
        assert_eq!(offered_names(&mcp), ["read", "edit"]);
        assert_eq!(tool_change_error(&mcp), None);
        // An undeclared reference: refused, on the prompt path and the count alike.
        let (state, _jobs) = app_state(None);
        let unresolved = request(vec![reference("tool_addition", "write")]);
        let err = prompt_ids_of(&state, &unresolved).err().expect("refused");
        assert!(err.contains("tool_reference_unresolved"), "{err}");
        assert!(err.contains("`write`"), "{err}");
    }

    /// With the model's own template the RESOLVED list is what renders: the added tool is in the
    /// `<tools>` block, the deferred and removed ones are not, and a system message that carried
    /// only tool changes is no turn at all — the request stays on the native path (an empty
    /// system turn mid-conversation makes Qwen3.8's template raise).
    #[test]
    fn tool_changes_render_the_resolved_tools_natively() {
        let body = json!({"tools": [
            {"name": "read", "description": "Read a file", "input_schema": {"type": "object"}},
            {"name": "grep", "description": "Search", "defer_loading": true,
                "input_schema": {"type": "object"}},
            {"name": "edit", "description": "Edit a file", "input_schema": {"type": "object"}}],
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "system", "content": [
                    {"type": "tool_addition", "tool": {"type": "tool_reference", "name": "grep"}},
                    {"type": "tool_removal", "tool": {"type": "tool_reference", "name": "edit"}}]},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "search it"}]});
        let prompt = match plan_messages(&parse(body), Some(&qwen38()), &vars()) {
            Ok(Plan::Native { prompt, .. }) => prompt,
            other => panic!("expected the native path, got {other:?}"),
        };
        let tools = &prompt[prompt.find("<tools>").unwrap()..prompt.find("</tools>").unwrap()];
        assert!(tools.contains("\"name\": \"read\""), "{tools}");
        assert!(tools.contains("\"name\": \"grep\""), "{tools}");
        assert!(!tools.contains("\"name\": \"edit\""), "{tools}");
        assert!(tools.find("read").unwrap() < tools.find("grep").unwrap());
        assert_eq!(prompt.matches("<|im_start|>system").count(), 1, "{prompt}");
        assert!(
            prompt.contains("<|im_start|>user\ngo<|im_end|>\n<|im_start|>assistant\n"),
            "{prompt}"
        );
    }

    /// Where `http::plan_chat` keeps the hand-written templates, so does this route: no tools and
    /// no history (even with a template); no template (`ARF_NO_NATIVE_TOOLS` leaves none), one
    /// that teaches no format, or one that raises → the fenced protocol. `tool_choice: none`
    /// with history still renders natively, without `<tools>`. An image is refused either way.
    #[test]
    fn plan_messages_takes_the_hand_written_paths_where_plan_chat_does() {
        let v = vars();
        let tools = parse(json!({"tools": [{"name": "read"}],
            "messages": [{"role": "user", "content": "x"}]}));
        let plain = parse(json!({"messages": [{"role": "user", "content": "x"}]}));
        assert!(matches!(
            plan_messages(&plain, Some(&qwen38()), &v),
            Ok(Plan::HandWritten(PromptPath::Plain))
        ));
        assert!(matches!(
            plan_messages(&tools, None, &v),
            Ok(Plan::HandWritten(PromptPath::Fenced))
        ));
        let no_format =
            crate::jinja_chat::ChatTemplate::new("{{ messages[0].content }}".into(), None).unwrap();
        assert!(matches!(
            plan_messages(&tools, Some(&no_format), &v),
            Ok(Plan::HandWritten(PromptPath::Fenced))
        ));
        let raises = crate::jinja_chat::ChatTemplate::new(
            "{{ raise_exception('no') }}<function=x><parameter=y>".into(),
            None,
        )
        .unwrap();
        assert!(matches!(
            plan_messages(&tools, Some(&raises), &v),
            Ok(Plan::HandWritten(PromptPath::Fenced))
        ));
        let none = parse(
            json!({"tools": [{"name": "read"}], "tool_choice": {"type": "none"},
            "messages": [{"role": "user", "content": "x"}]}),
        );
        assert!(matches!(
            plan_messages(&none, Some(&qwen38()), &v),
            Ok(Plan::HandWritten(PromptPath::Plain))
        ));
        let none_with_history = parse(json!({"tools": [{"name": "read"}],
            "tool_choice": {"type": "none"}, "messages": [
                {"role": "user", "content": "x"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t1", "name": "read", "input": {}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "A"}]}]}));
        match plan_messages(&none_with_history, Some(&qwen38()), &v) {
            Ok(Plan::Native { prompt, .. }) => {
                assert!(!prompt.contains("<tools>"), "{prompt}");
                assert!(
                    prompt.contains("<tool_response>\nA\n</tool_response>"),
                    "{prompt}"
                );
            }
            other => panic!("expected the native path, got {other:?}"),
        }
        // An image request never takes the native template: the fenced path splices it.
        let image = parse(json!({"tools": [{"name": "read"}], "messages": [
            {"role": "user", "content": [{"type": "image", "source": {}}]}]}));
        assert!(matches!(
            plan_messages(&image, Some(&qwen38()), &v),
            Ok(Plan::HandWritten(PromptPath::Fenced))
        ));
    }

    /// The translation itself: calls keep their ids and argument order (`null` input = no
    /// arguments), results become `tool` messages named after their call, a failed one says so,
    /// and a message of results alone adds no empty turn — while an empty message stays one.
    #[test]
    fn native_messages_translate_blocks_to_the_openai_shape() {
        let body = r#"{"system": [{"type": "text", "text": "S"}], "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "edit", "input": {"path": "a", "old": "x", "new": "y"}},
                {"type": "tool_use", "id": "t2", "name": "bash", "input": null}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
                {"type": "tool_result", "tool_use_id": "t2", "is_error": true,
                    "content": [{"type": "text", "text": "boom"}]}]},
            {"role": "assistant", "content": []}]}"#;
        let req: MessagesRequest = serde_json::from_str(body).unwrap();
        let m = native_messages(&req).unwrap();
        let roles: Vec<&str> = m.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(
            roles,
            ["system", "user", "assistant", "tool", "tool", "assistant"]
        );
        assert_eq!(m[0].content, "S");
        assert_eq!(m[2].content, "");
        let calls: Vec<(&str, &str, &str)> = m[2]
            .tool_calls
            .iter()
            .map(|c| {
                (
                    c.id.as_str(),
                    c.function.name.as_str(),
                    c.function.arguments.as_str(),
                )
            })
            .collect();
        assert_eq!(
            calls,
            [
                ("t1", "edit", r#"{"path":"a","old":"x","new":"y"}"#),
                ("t2", "bash", "{}"),
            ]
        );
        assert_eq!(
            (m[3].tool_call_id.as_deref(), m[3].name.as_deref()),
            (Some("t1"), Some("edit"))
        );
        assert_eq!(m[3].content, "ok");
        assert_eq!(m[4].content, "Error: boom");
        assert_eq!(m[4].name.as_deref(), Some("bash"));
        assert!(m[5].tool_calls.is_empty() && m[5].content.is_empty());
    }

    /// The non-streamed reply in the model's own format, read as the stream reads it: reasoning,
    /// prose and the call as blocks; a call cut off by `max_tokens` is text, not a `tool_use`.
    #[test]
    fn native_reply_becomes_blocks_as_the_stream_reads_it() {
        let call = "<tool_call>\n<function=read>\n<parameter=path>\ncalc.py\n</parameter>\n\
</function>\n</tool_call>";
        let text = format!("We need to read.\n</think>\n\nHi\n{call}");
        let (blocks, stop) = native_blocks(
            &text,
            ToolFormat::QwenXml,
            true,
            &read_tool(),
            FinishReason::Stop,
        );
        let blocks = values(&blocks);
        assert_eq!(stop, "tool_use");
        assert_eq!(blocks.len(), 3, "{blocks:?}");
        assert_eq!(
            blocks[0],
            json!({"type": "thinking", "thinking": "We need to read.", "signature": ""})
        );
        assert_eq!(blocks[1], json!({"type": "text", "text": "Hi"}));
        assert_eq!(blocks[2]["type"], "tool_use");
        assert_eq!(blocks[2]["name"], "read");
        assert_eq!(blocks[2]["input"], json!({"path": "calc.py"}));
        let cut = "Hi\n<tool_call>\n<function=read>\n<parameter=path>\ncalc";
        let (blocks, stop) = native_blocks(
            cut,
            ToolFormat::QwenXml,
            false,
            &read_tool(),
            FinishReason::Length,
        );
        assert_eq!(stop, "max_tokens");
        assert_eq!(values(&blocks), vec![json!({"type": "text", "text": cut})]);
    }

    /// What Claude Code sends: tools, and a user turn.
    fn agent_request(stream: bool) -> MessagesRequest {
        parse(json!({"max_tokens": 256, "stream": stream,
            "tools": [{"name": "read", "description": "Read a file",
                "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}],
            "messages": [{"role": "user", "content": "read calc.py"}]}))
    }

    /// Without a Qwen3.8 vision encoder an image block is still a 400, on both endpoints of this
    /// route (the prompt and the count), and names what would enable it.
    #[test]
    fn an_image_without_a_vision_encoder_is_refused() {
        let (state, _jobs) = app_state(Some(qwen38()));
        assert!(state.qwen_vision.is_none());
        let req = parse(json!({"messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}},
            {"type": "text", "text": "what is this?"}]}]}));
        let err = prompt_ids_of(&state, &req).err().expect("refused");
        assert!(err.contains("--mmproj"), "{err}");
    }

    /// The handler END TO END with the model's own template (Qwen3.8's, qwen-xml): the request
    /// takes the native path, and the answer's prose goes out as a `text_delta` BEFORE the call
    /// has been generated — the job is fed the prose only, and it must reach the wire first. The
    /// first live version never ran this configuration: its handler hard-coded the fenced
    /// format, the prose waited for the end of the turn, and its tests passed.
    #[tokio::test]
    async fn messages_with_the_models_template_streams_prose_before_the_call() {
        let (state, jobs) = app_state(Some(qwen38()));
        let req = agent_request(true);
        let p = prompt_ids_of(&state, &req).unwrap();
        assert_eq!(p.format, ToolFormat::QwenXml, "the native path must run");
        let resp = messages(State(state), Json(req)).await;
        let job = jobs.try_recv().expect("the handler queued a job");
        let mut body = resp.into_body().into_data_stream();
        // The reasoning (when the prompt pre-opened `<think>`, as at the template's default),
        // then "Hi" "\n" — no call token yet.
        let mut fed: Vec<u32> = if p.thinking { vec![0, 1, 5, 6] } else { vec![] };
        fed.extend([7, 4]);
        for &id in &fed {
            job.token_tx
                .send(StreamItem::Token(id, None))
                .await
                .unwrap();
        }
        let mut ev: Vec<Value> = Vec::new();
        while !ev.iter().any(|e| e["delta"]["type"] == "text_delta") {
            let frame = tokio::time::timeout(std::time::Duration::from_secs(5), body.next())
                .await
                .expect("the prose must stream before the call is generated")
                .expect("the stream is open")
                .unwrap();
            ev.extend(events_of(&String::from_utf8_lossy(&frame)));
        }
        for id in XML_CALL {
            job.token_tx
                .send(StreamItem::Token(id, None))
                .await
                .unwrap();
        }
        job.token_tx
            .send(StreamItem::Done(FinishReason::Stop))
            .await
            .unwrap();
        while let Some(frame) = body.next().await {
            ev.extend(events_of(&String::from_utf8_lossy(&frame.unwrap())));
        }
        let n = ev.len();
        assert_eq!(ev[0]["type"], "message_start");
        assert_eq!(
            ev[n - 2],
            json!({"type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": null},
                "usage": {"output_tokens": fed.len() + XML_CALL.len()}})
        );
        assert_eq!(ev[n - 1], json!({"type": "message_stop"}));
        let blocks = &ev[1..n - 2];
        assert_blocks_balanced(blocks);
        let mut expect = Vec::new();
        if p.thinking {
            expect.push(("thinking".to_string(), "We need".to_string()));
        }
        expect.push(("text".to_string(), "Hi".to_string()));
        expect.push(("tool_use".to_string(), "{\"path\":\"calc.py\"}".to_string()));
        assert_eq!(folded(blocks), expect, "{ev:?}");
    }

    /// The same request non-streamed, same template: the reply is read in the model's format
    /// (the fenced reader would have returned the XML as text).
    #[tokio::test]
    async fn messages_non_streamed_reads_the_models_own_format() {
        let (state, jobs) = app_state(Some(qwen38()));
        let req = agent_request(false);
        let thinking = prompt_ids_of(&state, &req).unwrap().thinking;
        let mut ids: Vec<u32> = if thinking { vec![0, 1, 5, 6] } else { vec![] };
        ids.extend([7, 4]);
        ids.extend(XML_CALL);
        let n = ids.len();
        let actor = std::thread::spawn(move || {
            let job = jobs.recv().unwrap();
            for id in ids {
                job.token_tx
                    .blocking_send(StreamItem::Token(id, None))
                    .unwrap();
            }
            job.token_tx
                .blocking_send(StreamItem::Done(FinishReason::Stop))
                .unwrap();
        });
        let resp = messages(State(state), Json(req)).await;
        actor.join().unwrap();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["stop_reason"], "tool_use");
        assert_eq!(v["usage"]["output_tokens"], n);
        let content = v["content"].as_array().unwrap();
        assert_eq!(content.len(), if thinking { 3 } else { 2 }, "{v}");
        let tail = &content[content.len() - 2..];
        assert_eq!(tail[0], json!({"type": "text", "text": "Hi"}));
        assert_eq!(tail[1]["type"], "tool_use");
        assert_eq!(tail[1]["name"], "read");
        assert_eq!(tail[1]["input"], json!({"path": "calc.py"}));
    }

    /// The handler's stop reason for a plain reply of "Hi" that ends with `finish`, streamed or
    /// not, under a request with `max_tokens` (absent when `None`), and the budget it queued.
    async fn plain_stop_reason(
        max_tokens: Option<usize>,
        finish: FinishReason,
        stream: bool,
    ) -> (String, usize) {
        let (state, jobs) = app_state(None);
        let mut body = json!({"stream": stream, "messages": [{"role": "user", "content": "hi"}]});
        if let Some(n) = max_tokens {
            body["max_tokens"] = json!(n);
        }
        let actor = std::thread::spawn(move || {
            let job = jobs.recv().unwrap();
            let budget = job.params.max_tokens;
            job.token_tx
                .blocking_send(StreamItem::Token(7, None))
                .unwrap();
            job.token_tx
                .blocking_send(StreamItem::Done(finish))
                .unwrap();
            budget
        });
        let resp = messages(State(state), Json(parse(body))).await;
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let budget = actor.join().unwrap();
        let stop = if stream {
            let ev = events_of(std::str::from_utf8(&bytes).unwrap());
            let n = ev.len();
            assert_eq!(ev[n - 2]["type"], "message_delta", "{ev:?}");
            ev[n - 2]["delta"]["stop_reason"].clone()
        } else {
            serde_json::from_slice::<Value>(&bytes).unwrap()["stop_reason"].clone()
        };
        (stop.as_str().unwrap().to_string(), budget)
    }

    /// A reply cut by the CONTEXT WINDOW (the client asked for more than was left, or set no
    /// limit, so the budget was clamped) stops with `model_context_window_exceeded`, streamed or
    /// not; one cut by the client's own `max_tokens` stops with `max_tokens`; a reply that ended
    /// on its own is `end_turn` either way. Until 2026-10-04 the first said `max_tokens`, and
    /// Claude Code then asked its user to raise an output limit that was not the one binding.
    #[tokio::test]
    async fn a_reply_cut_by_the_context_window_says_so() {
        let left = 4096 - 1; // the test tokenizer reads the whole prompt as one token
        for stream in [false, true] {
            for (max_tokens, finish, expect, budget) in [
                (
                    Some(1 << 20),
                    FinishReason::Length,
                    "model_context_window_exceeded",
                    left,
                ),
                (
                    None,
                    FinishReason::Length,
                    "model_context_window_exceeded",
                    left,
                ),
                (Some(left), FinishReason::Length, "max_tokens", left),
                (Some(16), FinishReason::Length, "max_tokens", 16),
                (Some(1 << 20), FinishReason::Stop, "end_turn", left),
            ] {
                let got = plain_stop_reason(max_tokens, finish, stream).await;
                assert_eq!(
                    got,
                    (expect.to_string(), budget),
                    "max_tokens {max_tokens:?}, {finish:?}, stream {stream}"
                );
            }
        }
    }

    /// `tool_choice: {"type": "tool", "name": "read"}` END TO END, non-streamed and streamed, on
    /// the model's own template: the prompt was begun with `<tool_call>\n<function=read>\n`
    /// after a closed `<think>`, so a reply that only continues it (`<parameter=path>…`) is a
    /// `read` `tool_use` with no thinking block. Until 2026-10-04 `tool` was not enforced, and at
    /// the template's default this reply would have been read as reasoning.
    #[tokio::test]
    async fn messages_forced_tool_choice_is_prefilled() {
        // `<parameter=path>\ncalc.py\n</parameter>\n</function>\n</tool_call>`
        let rest = [10u32, 4, 11, 4, 12, 4, 13, 4, 14];
        for stream in [false, true] {
            let (state, jobs) = app_state(Some(qwen38()));
            let body = json!({"max_tokens": 256, "stream": stream,
                "tool_choice": {"type": "tool", "name": "read"},
                "tools": [{"name": "read", "description": "Read a file",
                    "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}},
                    {"name": "list", "input_schema": {"type": "object"}}],
                "messages": [{"role": "user", "content": "read calc.py"}]});
            let req = parse(body.clone());
            let p = prompt_ids_of(&state, &req).unwrap();
            assert_eq!(p.prefill, "<tool_call>\n<function=read>\n");
            assert!(!p.thinking);
            let actor = std::thread::spawn(move || {
                let job = jobs.recv().unwrap();
                for id in rest {
                    job.token_tx
                        .blocking_send(StreamItem::Token(id, None))
                        .unwrap();
                }
                job.token_tx
                    .blocking_send(StreamItem::Done(FinishReason::Stop))
                    .unwrap();
            });
            let resp = messages(State(state), Json(parse(body))).await;
            let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap();
            actor.join().unwrap();
            if stream {
                let ev = events_of(std::str::from_utf8(&bytes).unwrap());
                let n = ev.len();
                assert_eq!(ev[n - 2]["delta"]["stop_reason"], "tool_use", "{ev:?}");
                let blocks = &ev[1..n - 2];
                assert_blocks_balanced(blocks);
                assert_eq!(
                    folded(blocks),
                    [("tool_use".to_string(), "{\"path\":\"calc.py\"}".to_string())],
                    "{ev:?}"
                );
            } else {
                let v: Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(v["stop_reason"], "tool_use", "{v}");
                let content = v["content"].as_array().unwrap();
                assert_eq!(content.len(), 1, "{v}");
                assert_eq!(content[0]["type"], "tool_use");
                assert_eq!(content[0]["name"], "read");
                assert_eq!(content[0]["input"], json!({"path": "calc.py"}));
            }
        }
    }

    /// A forced `tool_choice` AND a stop sequence on `/v1/messages`, streamed and not.
    /// The prompt is begun with `<tool_call>\n<function=read>\n`; the stop
    /// sequence is `<tool_call>`, which the prefill itself contains. The model completes the
    /// forced call and opens a second one: the job ends on that token, the reply is cut there,
    /// and exactly one `tool_use` comes back (`stop_reason: tool_use` — a call was made). Were the
    /// prefill matched against, there would be no call; without the prefill the reply would be
    /// reasoning; without the stop sequence, two calls.
    #[tokio::test]
    async fn messages_forced_call_with_a_stop_sequence() {
        let rest = [10u32, 4, 11, 4, 12, 4, 13, 4, 14];
        let mut ids: Vec<u32> = rest.to_vec();
        ids.extend([4, 8, 9, 4]);
        ids.extend(rest);
        for stream in [false, true] {
            let (state, jobs) = app_state(Some(qwen38()));
            let body = json!({"max_tokens": 256, "stream": stream,
                "tool_choice": {"type": "tool", "name": "read"},
                "stop_sequences": ["<tool_call>"],
                "tools": [{"name": "read", "description": "Read a file",
                    "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}],
                "messages": [{"role": "user", "content": "read calc.py"}]});
            let actor = super::super::tests::play_with_stops(jobs, ids.clone());
            let resp = messages(State(state), Json(parse(body))).await;
            let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap();
            assert_eq!(actor.join().unwrap(), rest.len() + 2, "stream {stream}");
            let (stop, uses) = if stream {
                let ev = events_of(std::str::from_utf8(&bytes).unwrap());
                let n = ev.len();
                let blocks = &ev[1..n - 2];
                assert_blocks_balanced(blocks);
                (
                    ev[n - 2]["delta"]["stop_reason"].clone(),
                    folded(blocks)
                        .into_iter()
                        .map(|(kind, body)| json!([kind, body]))
                        .collect::<Vec<_>>(),
                )
            } else {
                let v: Value = serde_json::from_slice(&bytes).unwrap();
                let uses = v["content"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|b| match b["type"].as_str() {
                        Some("tool_use") => json!(["tool_use", b["input"].to_string()]),
                        _ => json!([b["type"], b["text"]]),
                    })
                    .collect();
                (v["stop_reason"].clone(), uses)
            };
            assert_eq!(stop, "tool_use", "stream {stream}");
            assert_eq!(
                uses,
                vec![json!(["tool_use", "{\"path\":\"calc.py\"}"])],
                "stream {stream}"
            );
        }
    }

    /// No usable template (none loaded, or `ARF_NO_NATIVE_TOOLS`): the fenced-JSON protocol, as
    /// `/v1/chat/completions` falls back — the answer settled when the reply ends, the call still
    /// a `tool_use` block.
    #[tokio::test]
    async fn messages_without_a_template_falls_back_to_the_fenced_protocol() {
        let (state, jobs) = app_state(None);
        let req = agent_request(true);
        let p = prompt_ids_of(&state, &req).unwrap();
        assert_eq!(p.format, ToolFormat::None);
        assert!(!p.thinking);
        let resp = messages(State(state), Json(req)).await;
        let job = jobs.try_recv().expect("the handler queued a job");
        // "```json" "\n" {…read a.py…} "\n" "```"
        for id in [17u32, 4, 18, 4, 19] {
            job.token_tx
                .send(StreamItem::Token(id, None))
                .await
                .unwrap();
        }
        job.token_tx
            .send(StreamItem::Done(FinishReason::Stop))
            .await
            .unwrap();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let ev = events_of(std::str::from_utf8(&body).unwrap());
        let n = ev.len();
        assert_eq!(ev[n - 2]["delta"]["stop_reason"], "tool_use");
        let blocks = &ev[1..n - 2];
        assert_blocks_balanced(blocks);
        assert_eq!(
            folded(blocks),
            [("tool_use".to_string(), "{\"path\":\"a.py\"}".to_string())]
        );
    }

    // ------------------------------------------------------------------------
    // The request's `thinking` (`thinking_vars`), since 2026-09-26: the variables it maps to, the
    // prompt it renders, and the replies read off that prompt.
    // ------------------------------------------------------------------------

    /// [`agent_request`] with a `thinking` field.
    fn agent_request_thinking(stream: bool, thinking: Value) -> MessagesRequest {
        parse(
            json!({"max_tokens": 256, "stream": stream, "thinking": thinking,
            "tools": [{"name": "read", "description": "Read a file",
                "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}],
            "messages": [{"role": "user", "content": "read calc.py"}]}),
        )
    }

    /// `(enable_thinking, reasoning_effort, preserve_thinking)` — `TemplateVars` has no `PartialEq`.
    fn fields(v: &crate::jinja_chat::TemplateVars) -> (Option<bool>, Option<String>, Option<bool>) {
        (
            v.enable_thinking,
            v.reasoning_effort.clone(),
            v.preserve_thinking,
        )
    }

    /// `thinking` resolves as `chat_template_kwargs.enable_thinking` does on
    /// `/v1/chat/completions`. Absent — or a shape this route does not know, which must still
    /// parse (the field was accepted and ignored before) — is EXACTLY the settings of before,
    /// under every server default.
    #[test]
    fn thinking_maps_onto_the_chat_completions_variables() {
        let plain = |thinking: Option<Value>| {
            let mut body = json!({"messages": [{"role": "user", "content": "x"}]});
            if let Some(t) = thinking {
                body["thinking"] = t;
            }
            parse(body)
        };
        let absent = plain(None);
        for unknown in [json!({"type": "adaptive"}), json!(true), json!("enabled")] {
            let req = plain(Some(unknown.clone()));
            assert_eq!(req.enable_thinking(), None, "{unknown}");
            assert_eq!(req.thinking_budget(), None, "{unknown}");
        }
        // No field (2026-10-07) — `null` is no field — is `disabled`, as in the Messages API.
        assert_eq!(absent.enable_thinking(), Some(false));
        assert_eq!(plain(Some(json!(null))).enable_thinking(), Some(false));
        for (no_think_env, env_effort) in [
            (false, None),
            (true, None),
            (false, Some("low".to_string())),
        ] {
            let adaptive = plain(Some(json!({"type": "adaptive"})));
            let (effort, v) = thinking_vars(&adaptive, no_think_env, env_effort.clone());
            assert_eq!(effort, None);
            assert_eq!(
                fields(&v),
                fields(&template_vars(None, None, no_think_env, env_effort.clone())),
                "{no_think_env} {env_effort:?}"
            );
            // No field: not thinking, whatever the server's defaults (as `disabled` below).
            let (effort, v) = thinking_vars(&absent, no_think_env, env_effort.clone());
            assert_eq!(effort, Some("none"));
            assert_eq!(fields(&v), (Some(false), None, None));
        }

        // Disabled: the effort "none" (the hand-written templates' non-thinking form) and
        // enable_thinking=false — whatever the server's defaults.
        let off = plain(Some(json!({"type": "disabled"})));
        for no_think_env in [false, true] {
            let (effort, v) = thinking_vars(&off, no_think_env, Some("xhigh".into()));
            assert_eq!(effort, Some("none"));
            assert_eq!(fields(&v), (Some(false), None, None));
        }
        assert_eq!(off.thinking_budget(), None);

        // Enabled: enable_thinking=true, and it wins over the operator's switch on the model's own
        // template, as the OpenAI kwarg does. The budget is read (for the log) and not mapped.
        let on = plain(Some(json!({"type": "enabled", "budget_tokens": 10000})));
        assert_eq!(on.thinking_budget(), Some(10000));
        for no_think_env in [false, true] {
            let (effort, v) = thinking_vars(&on, no_think_env, None);
            assert_eq!(effort, None);
            assert_eq!(fields(&v), (Some(true), None, None));
        }
        let (_, v) = thinking_vars(&on, false, Some("low".into()));
        assert_eq!(fields(&v), (Some(true), Some("low".into()), None));
        let on_no_budget = plain(Some(json!({"type": "enabled"})));
        assert_eq!(on_no_budget.enable_thinking(), Some(true));
        assert_eq!(on_no_budget.thinking_budget(), None);
    }

    /// The prompt with and without thinking, on both paths. The model's own template (a tool
    /// request): `disabled` pre-CLOSES the block and the reply is not read as reasoning;
    /// `enabled` and absent pre-open it, and absent renders the very bytes of before. The
    /// hand-written path (a plain request): `disabled` hands it the effort `"none"`, which the
    /// Qwen3.8 template renders pre-closed, and absent hands it nothing — `apply_chat_template`.
    #[test]
    fn thinking_renders_the_prompt_with_and_without_thinking() {
        let tpl = qwen38();
        let render = |req: &MessagesRequest| {
            let (_, v) = thinking_vars(req, false, None);
            match plan_messages(req, Some(&tpl), &v) {
                Ok(Plan::Native {
                    prompt, thinking, ..
                }) => (prompt, thinking),
                other => panic!("expected the native path, got {other:?}"),
            }
        };
        // (`absent` is the server's default, which `adaptive` asks for; a request with no field
        // at all renders as `disabled` — asserted with `off` below.)
        let (absent, t_absent) =
            render(&agent_request_thinking(false, json!({"type": "adaptive"})));
        let (before, _) = match plan_messages(&agent_request(false), Some(&tpl), &vars()) {
            Ok(Plan::Native {
                prompt, thinking, ..
            }) => (prompt, thinking),
            other => panic!("expected the native path, got {other:?}"),
        };
        assert_eq!(
            absent, before,
            "an `adaptive` request renders at the server's defaults, as before"
        );
        assert!(
            absent.ends_with("<|im_start|>assistant\n<think>\n"),
            "{absent}"
        );
        assert!(t_absent);

        let (on, t_on) = render(&agent_request_thinking(
            false,
            json!({"type": "enabled", "budget_tokens": 4000}),
        ));
        assert!(on.ends_with("<|im_start|>assistant\n<think>\n"), "{on}");
        assert!(t_on);
        // Qwen3.8's template reads `undefined or true` as one branch: under the server's default
        // settings, `enabled` is the SAME prompt as no field — nothing for the prefix cache to miss.
        assert_eq!(on, absent);

        let (off, t_off) = render(&agent_request_thinking(false, json!({"type": "disabled"})));
        assert!(
            off.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "{off}"
        );
        assert!(!t_off);
        // No `thinking` field is the same prompt (2026-10-07): one cached prefix for both.
        let (none, t_none) = render(&agent_request(false));
        assert_eq!(none, off);
        assert!(!t_none);
        // Besides the cue, the template drops its effort sentence when not thinking (as the
        // hand-written "none" form does); the tools and the conversation render the same.
        let head = |p: &str| p[..p.rfind("<|im_start|>assistant").unwrap()].to_string();
        let xhigh = format!("{}\n\n", crate::chat::QWEN3_EFFORT_XHIGH);
        assert!(absent.contains(&xhigh), "{absent}");
        assert!(!off.contains("Reasoning effort"), "{off}");
        assert_eq!(head(&off), head(&absent).replacen(&xhigh, "", 1));

        // The hand-written path: a request with neither tools nor tool history.
        let plain = |thinking: Value| {
            parse(json!({"thinking": thinking, "messages": [{"role": "user", "content": "hi"}]}))
        };
        let off = plain(json!({"type": "disabled"}));
        let (effort, v) = thinking_vars(&off, false, None);
        assert!(matches!(
            plan_messages(&off, Some(&tpl), &v),
            Ok(Plan::HandWritten(PromptPath::Plain))
        ));
        assert_eq!(effort, Some("none"));
        // What `apply_chat_template_with` renders for Qwen3.8 with that effort.
        let p = crate::chat::qwen3_think_template(&flatten(&off).unwrap(), effort);
        assert_eq!(
            p,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        assert!(!crate::chat::prompt_opens_think(&p));
        let on = plain(json!({"type": "enabled", "budget_tokens": 2048}));
        assert_eq!(thinking_vars(&on, false, None).0, None);
        // `null` is no field: not thinking, like `disabled` above.
        assert_eq!(
            thinking_vars(&plain(json!(null)), false, None).0,
            Some("none")
        );
    }

    /// Feed a job the prose and one qwen-xml call (after the reasoning, when the prompt opened
    /// `<think>`), then `Done(Stop)`. The ids fed.
    fn reply_ids(thinking: bool) -> Vec<u32> {
        let mut ids: Vec<u32> = if thinking { vec![0, 1, 5, 6] } else { vec![] };
        ids.extend([7, 4]);
        ids.extend(XML_CALL);
        ids
    }

    /// The handler END TO END, streamed, with `thinking: disabled` on the model's own template:
    /// the prompt does not open `<think>`, the reply streams as `text` then `tool_use`, and no
    /// `thinking` block is ever opened. With `enabled`, the reasoning is a `thinking` block again.
    #[tokio::test]
    async fn messages_streamed_honours_thinking_disabled_and_enabled() {
        // `ARF_REASONING_EFFORT=none` in the shell would turn `enabled` off too (the operator's
        // default wins over the request, as on `/v1/chat/completions`): read, never written.
        let env_none = std::env::var("ARF_REASONING_EFFORT").as_deref() == Ok("none");
        for (thinking, want) in [
            (json!({"type": "disabled"}), false),
            (json!({"type": "enabled", "budget_tokens": 1024}), !env_none),
        ] {
            let (state, jobs) = app_state(Some(qwen38()));
            let req = agent_request_thinking(true, thinking.clone());
            let p = prompt_ids_of(&state, &req).unwrap();
            assert_eq!(p.format, ToolFormat::QwenXml, "the native path must run");
            assert_eq!(p.thinking, want, "{thinking}");
            let resp = messages(State(state), Json(req)).await;
            let job = jobs.try_recv().expect("the handler queued a job");
            let ids = reply_ids(want);
            for &id in &ids {
                job.token_tx
                    .send(StreamItem::Token(id, None))
                    .await
                    .unwrap();
            }
            job.token_tx
                .send(StreamItem::Done(FinishReason::Stop))
                .await
                .unwrap();
            let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap();
            let ev = events_of(std::str::from_utf8(&body).unwrap());
            let n = ev.len();
            assert_eq!(ev[n - 2]["delta"]["stop_reason"], "tool_use", "{ev:?}");
            let blocks = &ev[1..n - 2];
            assert_blocks_balanced(blocks);
            let mut expect = Vec::new();
            if want {
                expect.push(("thinking".to_string(), "We need".to_string()));
            }
            expect.push(("text".to_string(), "Hi".to_string()));
            expect.push(("tool_use".to_string(), "{\"path\":\"calc.py\"}".to_string()));
            assert_eq!(folded(blocks), expect, "{thinking}: {ev:?}");
            if !want {
                assert!(
                    !ev.iter().any(|e| e.to_string().contains("thinking")),
                    "{thinking}: {ev:?}"
                );
            }
        }
    }

    /// The same, non-streamed: `disabled` → no `thinking` block in `content`; `enabled` → one.
    #[tokio::test]
    async fn messages_non_streamed_honours_thinking_disabled_and_enabled() {
        let env_none = std::env::var("ARF_REASONING_EFFORT").as_deref() == Ok("none");
        for (thinking, want) in [
            (json!({"type": "disabled"}), false),
            (json!({"type": "enabled", "budget_tokens": 1024}), !env_none),
        ] {
            let (state, jobs) = app_state(Some(qwen38()));
            let req = agent_request_thinking(false, thinking.clone());
            assert_eq!(prompt_ids_of(&state, &req).unwrap().thinking, want);
            let ids = reply_ids(want);
            let n = ids.len();
            let actor = std::thread::spawn(move || {
                let job = jobs.recv().unwrap();
                for id in ids {
                    job.token_tx
                        .blocking_send(StreamItem::Token(id, None))
                        .unwrap();
                }
                job.token_tx
                    .blocking_send(StreamItem::Done(FinishReason::Stop))
                    .unwrap();
            });
            let resp = messages(State(state), Json(req)).await;
            actor.join().unwrap();
            let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap();
            let v: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(v["stop_reason"], "tool_use", "{v}");
            assert_eq!(v["usage"]["output_tokens"], n);
            let types: Vec<&str> = v["content"]
                .as_array()
                .unwrap()
                .iter()
                .map(|b| b["type"].as_str().unwrap())
                .collect();
            let expect: &[&str] = if want {
                &["thinking", "text", "tool_use"]
            } else {
                &["text", "tool_use"]
            };
            assert_eq!(types, expect, "{thinking}: {v}");
            let text = &v["content"][if want { 1 } else { 0 }];
            assert_eq!(text, &json!({"type": "text", "text": "Hi"}));
        }
    }

    // ------------------------------------------------------------------------
    // `output_config.effort` and `thinking: adaptive` — the shape Claude Code actually sends
    // (review, 2026-09-26; the captured request carries `adaptive` + `{"effort": "high"}`).
    // ------------------------------------------------------------------------

    /// [`agent_request`] as Claude Code shapes it: `thinking: adaptive` and an effort.
    fn claude_code_request(stream: bool, effort: &str) -> MessagesRequest {
        let mut req = agent_request_thinking(stream, json!({"type": "adaptive"}));
        req.output_config = Some(json!({"effort": effort}));
        req
    }

    /// The effort maps onto the knob `/v1/chat/completions` takes as `reasoning_effort`: the
    /// variables are exactly an OpenAI request's with that effort, under every server default.
    /// `disabled` still wins; an unknown or malformed effort is the server's default, exactly as
    /// if absent; and `adaptive` defers to the operator's switch where `enabled` overrides it.
    #[test]
    fn output_config_effort_is_the_requests_reasoning_effort() {
        for (level, want) in [
            ("low", Some("low")),
            ("medium", Some("medium")),
            ("high", Some("high")),
            ("max", Some("xhigh")),
            ("xhigh", Some("xhigh")),
            ("none", None),
            ("minimal", None),
            ("turbo", None),
            ("LOW", None),
            ("", None),
        ] {
            assert_eq!(anthropic_effort(level), want, "{level:?}");
        }
        let with = |thinking: Value, output_config: Value| {
            parse(json!({"thinking": thinking, "output_config": output_config,
                "messages": [{"role": "user", "content": "x"}]}))
        };
        let adaptive = || json!({"type": "adaptive"});
        let defaults = [
            (false, None),
            (true, None),
            (false, Some("medium".to_string())),
            (true, Some("low".to_string())),
        ];
        for (no_think_env, env_effort) in defaults.clone() {
            let openai = |effort: Option<&str>| {
                fields(&template_vars(
                    effort,
                    None,
                    no_think_env,
                    env_effort.clone(),
                ))
            };
            for (level, mapped) in [
                ("low", "low"),
                ("medium", "medium"),
                ("high", "high"),
                ("max", "xhigh"),
            ] {
                let req = with(adaptive(), json!({"effort": level}));
                assert_eq!(req.effort_field(), Some(level));
                let (effort, v) = thinking_vars(&req, no_think_env, env_effort.clone());
                assert_eq!(effort, Some(mapped), "{level}");
                assert_eq!(
                    fields(&v),
                    openai(Some(mapped)),
                    "{level} {no_think_env} {env_effort:?}"
                );
            }
            // Unknown or malformed: the settings of a request without the field.
            for oc in [
                json!({"effort": "turbo"}),
                json!({"effort": 3}),
                json!({"effort": null}),
                json!({}),
                json!({"format": {"type": "json_schema"}}),
                json!("high"),
                json!(null),
            ] {
                let (effort, v) = thinking_vars(
                    &with(adaptive(), oc.clone()),
                    no_think_env,
                    env_effort.clone(),
                );
                assert_eq!(effort, None, "{oc}");
                assert_eq!(fields(&v), openai(None), "{oc}");
            }
            // `disabled` wins over the effort, as `enable_thinking: false` wins over
            // `reasoning_effort` on `/v1/chat/completions`.
            let off = with(json!({"type": "disabled"}), json!({"effort": "low"}));
            let (effort, v) = thinking_vars(&off, no_think_env, env_effort.clone());
            assert_eq!(effort, Some("none"));
            assert_eq!(fields(&v), (Some(false), None, None));
            // `enabled` with an effort: thinking forced on, at that effort.
            let on = with(json!({"type": "enabled"}), json!({"effort": "medium"}));
            let (effort, v) = thinking_vars(&on, no_think_env, env_effort.clone());
            assert_eq!(effort, Some("medium"));
            assert_eq!(fields(&v), (Some(true), Some("medium".into()), None));
        }
        // Concretely: Claude Code's `low` reaches the template, and over the operator's
        // `ARF_REASONING_EFFORT` (the request wins, as on `/v1/chat/completions`).
        let low = with(adaptive(), json!({"effort": "low"}));
        assert_eq!(
            fields(&thinking_vars(&low, false, Some("xhigh".into())).1),
            (None, Some("low".into()), None)
        );
        // The deliberate difference: under ARF_NO_QWEN_THINK_TEMPLATE `adaptive` does not think
        // (the operator's setting) and `enabled` does (an explicit request, as the OpenAI kwarg).
        let (_, a) = thinking_vars(&low, true, None);
        let (_, e) = thinking_vars(
            &with(json!({"type": "enabled"}), json!({"effort": "low"})),
            true,
            None,
        );
        assert_eq!(
            (a.enable_thinking, e.enable_thinking),
            (Some(false), Some(true))
        );
        // Neither field: not thinking (2026-10-07, as in the Messages API), under every default.
        let absent = parse(json!({"messages": [{"role": "user", "content": "x"}]}));
        assert_eq!(absent.effort_field(), None);
        for (no_think_env, env_effort) in defaults {
            let (effort, v) = thinking_vars(&absent, no_think_env, env_effort.clone());
            assert_eq!(effort, Some("none"));
            assert_eq!(fields(&v), (Some(false), None, None));
        }
    }

    /// The prompt it renders, on both paths. Claude Code's own `high` (and `max`) is the SAME
    /// bytes as no effort under the default settings — nothing moves for the captured client's
    /// runs; `low` puts the `low` sentence where the `xhigh` one was; `medium` has none; every one
    /// still opens `<think>` (`adaptive` thinks as the template does).
    #[test]
    fn output_config_effort_renders_the_effort_sentence() {
        let tpl = qwen38();
        let render = |req: &MessagesRequest| {
            let (_, v) = thinking_vars(req, false, None);
            match plan_messages(req, Some(&tpl), &v) {
                Ok(Plan::Native { prompt, .. }) => prompt,
                other => panic!("expected the native path, got {other:?}"),
            }
        };
        let absent = render(&agent_request_thinking(false, json!({"type": "adaptive"})));
        for level in ["high", "max", "xhigh"] {
            assert_eq!(
                render(&claude_code_request(false, level)),
                absent,
                "{level}"
            );
        }
        let head = |p: &str| p[..p.rfind("<|im_start|>assistant").unwrap()].to_string();
        let xhigh = crate::chat::QWEN3_EFFORT_XHIGH;
        let low_s = crate::chat::QWEN3_EFFORT_LOW;
        let low = render(&claude_code_request(false, "low"));
        assert!(low.ends_with("<|im_start|>assistant\n<think>\n"), "{low}");
        assert!(low.contains(low_s) && !low.contains(xhigh), "{low}");
        assert_eq!(head(&low), head(&absent).replacen(xhigh, low_s, 1));
        let medium = render(&claude_code_request(false, "medium"));
        assert!(
            medium.ends_with("<|im_start|>assistant\n<think>\n"),
            "{medium}"
        );
        assert!(!medium.contains("Reasoning effort"), "{medium}");
        // An effort Arf does not know renders as no effort — never a template exception.
        assert_eq!(render(&claude_code_request(false, "turbo")), absent);

        // The hand-written path (neither tools nor tool history): the effort reaches
        // `apply_chat_template_with`, i.e. Qwen3.8's hand-written template.
        let plain = |effort: &str| {
            parse(
                json!({"thinking": {"type": "adaptive"}, "output_config": {"effort": effort},
                "messages": [{"role": "user", "content": "hi"}]}),
            )
        };
        let msgs = flatten(&plain("low")).unwrap();
        for (level, want) in [
            ("low", Some(low_s)),
            ("high", Some(xhigh)),
            ("medium", None),
        ] {
            let req = plain(level);
            let (effort, v) = thinking_vars(&req, false, None);
            assert!(matches!(
                plan_messages(&req, Some(&tpl), &v),
                Ok(Plan::HandWritten(PromptPath::Plain))
            ));
            let p = crate::chat::qwen3_think_template(&msgs, effort);
            assert!(crate::chat::prompt_opens_think(&p), "{level}: {p}");
            match want {
                Some(s) => assert!(p.starts_with(&format!("<|im_start|>system\n{s}")), "{p}"),
                None => assert!(!p.contains("Reasoning effort"), "{p}"),
            }
        }
        // `high` is the hand-written template's default sentence too — the same bytes as xhigh.
        let (high, _) = thinking_vars(&plain("high"), false, None);
        assert_eq!(
            crate::chat::qwen3_think_template(&msgs, high),
            crate::chat::qwen3_think_template(&msgs, Some("xhigh"))
        );
    }

    /// The handler END TO END with Claude Code's shape, streamed and not: the request is
    /// accepted with `output_config`, and the reply is read off the prompt `adaptive` rendered —
    /// a `thinking` block when the server's default thinks, none when the operator turned it off
    /// (`ARF_NO_QWEN_THINK_TEMPLATE`, read from the shell, never written). It does NOT prove the
    /// effort reached the prompt — the test tokenizer cannot tell one prompt's ids from another's
    /// (checked: it passes with the mapping stubbed out); the two tests above do (both fail so).
    #[tokio::test]
    async fn messages_accepts_claude_codes_adaptive_thinking_and_effort() {
        // The request's effort wins over `ARF_REASONING_EFFORT` (even `none`), so only the
        // operator's thinking switch decides.
        let want = std::env::var_os("ARF_NO_QWEN_THINK_TEMPLATE").is_none();
        for stream in [false, true] {
            let (state, jobs) = app_state(Some(qwen38()));
            let req = claude_code_request(stream, "low");
            let p = prompt_ids_of(&state, &req).unwrap();
            assert_eq!(p.format, ToolFormat::QwenXml, "the native path must run");
            assert_eq!(p.thinking, want);
            let ids = reply_ids(want);
            let actor = std::thread::spawn(move || {
                let job = jobs.recv().unwrap();
                for id in ids {
                    job.token_tx
                        .blocking_send(StreamItem::Token(id, None))
                        .unwrap();
                }
                job.token_tx
                    .blocking_send(StreamItem::Done(FinishReason::Stop))
                    .unwrap();
            });
            let resp = messages(State(state), Json(req)).await;
            let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap();
            actor.join().unwrap();
            let mut expect = Vec::new();
            if want {
                expect.push(("thinking".to_string(), "We need".to_string()));
            }
            expect.push(("text".to_string(), "Hi".to_string()));
            expect.push(("tool_use".to_string(), "{\"path\":\"calc.py\"}".to_string()));
            let got = if stream {
                let ev = events_of(std::str::from_utf8(&body).unwrap());
                let n = ev.len();
                assert_eq!(ev[n - 2]["delta"]["stop_reason"], "tool_use", "{ev:?}");
                assert_blocks_balanced(&ev[1..n - 2]);
                folded(&ev[1..n - 2])
            } else {
                let v: Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(v["stop_reason"], "tool_use", "{v}");
                v["content"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|b| {
                        let t = b["type"].as_str().unwrap().to_string();
                        let s = match t.as_str() {
                            "thinking" => b["thinking"].as_str().unwrap().to_string(),
                            "text" => b["text"].as_str().unwrap().to_string(),
                            _ => b["input"].to_string(),
                        };
                        (t, s)
                    })
                    .collect()
            };
            assert_eq!(got, expect, "stream={stream}");
        }
    }

    // ------------------------------------------------------------------------
    // A `tool_use`'s `input` in the model's key order (`ReplyBlock`, 2026-09-26). The client
    // replays it in the next request's history, and the native path renders it in the order it
    // arrives: sorted keys there are bytes the model never generated, a prefix-cache miss.
    // ------------------------------------------------------------------------

    /// An `edit(path, oldText, newText)` tool, as Claude Code's is shaped (not sorted order).
    fn edit_request(stream: bool) -> MessagesRequest {
        let body = format!(
            r#"{{"max_tokens": 256, "stream": {stream}, "thinking": {{"type": "disabled"}},
            "tools": [{{"name": "edit", "description": "Edit a file", "input_schema":
                {{"type": "object", "properties": {{"path": {{"type": "string"}},
                 "oldText": {{"type": "string"}}, "newText": {{"type": "string"}}}}}}}}],
            "messages": [{{"role": "user", "content": "fix calc.py"}}]}}"#
        );
        serde_json::from_str(&body).unwrap()
    }

    /// The non-streamed reply END TO END: the response BYTES carry `input` exactly as the model
    /// wrote the arguments — `{"path":…,"oldText":…,"newText":…}` — and exactly as the stream
    /// sends them in `partial_json`. Until 2026-09-26 the non-streamed body read
    /// `{"newText":"Hi","oldText":"We","path":"calc.py"}` (a `serde_json::Value` sorts).
    #[tokio::test]
    async fn tool_use_input_keeps_the_models_key_order_streamed_and_not() {
        use super::super::tests::{EDIT_ARGS, XML_EDIT};
        let ids: Vec<u32> = [7u32, 4].into_iter().chain(XML_EDIT).collect();

        let (state, jobs) = app_state(Some(qwen38()));
        let req = edit_request(false);
        let p = prompt_ids_of(&state, &req).unwrap();
        assert_eq!((p.format, p.thinking), (ToolFormat::QwenXml, false));
        let fed = ids.clone();
        let actor = std::thread::spawn(move || {
            let job = jobs.recv().unwrap();
            for id in fed {
                job.token_tx
                    .blocking_send(StreamItem::Token(id, None))
                    .unwrap();
            }
            job.token_tx
                .blocking_send(StreamItem::Done(FinishReason::Stop))
                .unwrap();
        });
        let resp = messages(State(state), Json(req)).await;
        actor.join().unwrap();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let raw = std::str::from_utf8(&body).unwrap();
        assert!(
            raw.contains(&format!(r#""name":"edit","input":{EDIT_ARGS}}}"#)),
            "{raw}"
        );
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["stop_reason"], "tool_use");
        assert_eq!(v["content"][0], json!({"type": "text", "text": "Hi"}));
        // The typed envelope carries the fields the `json!` one did, and no others.
        let mut keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "content",
                "id",
                "model",
                "role",
                "stop_reason",
                "stop_sequence",
                "type",
                "usage"
            ]
        );
        assert_eq!(
            (&v["type"], &v["role"], &v["model"], &v["stop_sequence"]),
            (
                &json!("message"),
                &json!("assistant"),
                &json!("test"),
                &Value::Null
            )
        );
        assert!(v["id"].as_str().unwrap().starts_with("msg"), "{v}");
        assert!(v["content"][1]["id"].as_str().unwrap().starts_with("toolu"));
        assert_eq!(v["usage"]["output_tokens"], ids.len());

        // Streamed: the same bytes as `partial_json`.
        let (state, jobs) = app_state(Some(qwen38()));
        let resp = messages(State(state), Json(edit_request(true))).await;
        let job = jobs.try_recv().expect("the handler queued a job");
        for &id in &ids {
            job.token_tx
                .send(StreamItem::Token(id, None))
                .await
                .unwrap();
        }
        job.token_tx
            .send(StreamItem::Done(FinishReason::Stop))
            .await
            .unwrap();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let ev = events_of(std::str::from_utf8(&body).unwrap());
        let partial: Vec<&str> = ev
            .iter()
            .filter_map(|e| e["delta"]["partial_json"].as_str())
            .collect();
        assert_eq!(partial, [EDIT_ARGS]);
    }

    /// The fenced-JSON protocol's call too: its arguments are parsed order-preserving, and the
    /// serialized block keeps that order.
    #[test]
    fn fenced_tool_use_input_keeps_the_models_key_order() {
        let reply = "```json\n{\"function\": \"edit\", \"arguments\": \
            {\"path\": \"a.rs\", \"oldText\": \"x\", \"newText\": \"y\"}}\n```";
        let (blocks, stop) = blocks_of(reply, &["edit"], FinishReason::Stop);
        assert_eq!(stop, "tool_use");
        let s = serde_json::to_string(&blocks).unwrap();
        assert!(
            s.contains(r#""input":{"path":"a.rs","oldText":"x","newText":"y"}"#),
            "{s}"
        );
    }

    /// `call_input`, the one rule both replies take: an object's text verbatim (inner spacing
    /// and order kept; only the whitespace around it goes), anything else `{}`.
    #[test]
    fn call_input_is_the_object_verbatim_or_empty() {
        for (args, want) in [
            (r#"{"b":2,"a":1}"#, r#"{"b":2,"a":1}"#),
            (
                " {\"b\": 2, \"a\": [1, {\"z\": 0, \"y\": null}]}\n",
                r#"{"b": 2, "a": [1, {"z": 0, "y": null}]}"#,
            ),
            ("{}", "{}"),
            ("[1]", "{}"),
            (r#""x""#, "{}"),
            ("null", "{}"),
            ("", "{}"),
            ("{", "{}"),
            (r#"{"a":1} trailing"#, "{}"),
        ] {
            assert_eq!(call_input(args).get(), want, "{args:?}");
        }
    }

    /// Arguments a `RawValue` scan lets through and a strict parser rejects — an out-of-range
    /// number, a lone surrogate, nesting past serde_json's 128 — are `{}`, as they were before
    /// the order-keeping change (review, 2026-09-26). Each reaches `call_input` the way it can in
    /// production: a fenced / Hermes call whose `arguments` is a JSON STRING, passed through
    /// unparsed by `json_call_from_object`. The finished body must parse with serde_json, the
    /// strict parser a Rust client uses.
    #[test]
    fn call_input_refuses_what_a_strict_parser_refuses() {
        let deep = format!("{{\"a\":{}1{}}}", "[".repeat(200), "]".repeat(200));
        for args in [r#"{"n": 1e400}"#, r#"{"path": "\ud800"}"#, deep.as_str()] {
            // The RawValue scan alone accepted each — the hole this closes.
            assert!(
                serde_json::from_str::<Box<serde_json::value::RawValue>>(args).is_ok(),
                "{args}"
            );
            assert!(serde_json::from_str::<Value>(args).is_err(), "{args}");
            assert_eq!(call_input(args).get(), "{}", "{args}");

            // End to end through the fenced parser and the finished reply's serializer.
            let call = json!({"name": "calc", "arguments": args}).to_string();
            let (blocks, stop) = blocks_of(&call, &["calc"], FinishReason::Stop);
            assert_eq!(stop, "tool_use");
            let body = serde_json::to_string(&blocks).unwrap();
            let v: Value = serde_json::from_str(&body).expect("the body parses strictly");
            assert_eq!(v[0]["input"], json!({}), "{body}");
        }
        // A string-shaped arguments object that DOES parse is still sent as written.
        let call = json!({"name": "calc", "arguments": r#"{"z": 1, "a": 2}"#}).to_string();
        let (blocks, _) = blocks_of(&call, &["calc"], FinishReason::Stop);
        let body = serde_json::to_string(&blocks).unwrap();
        assert!(body.contains(r#""input":{"z": 1, "a": 2}"#), "{body}");
    }

    // --------------------------------------------------------------------------------------
    // PREFIX ANCHORS through the real handlers: the job each one queues carries the anchor, on
    // every prompt path, and it ends before the user's text in the prompt that job will run.
    // --------------------------------------------------------------------------------------

    /// `app_state` with a tokenizer that encodes any text losslessly (the word-level test vocab
    /// maps a whole prompt to one unknown token, which has no prefixes to anchor).
    fn anchor_state(
        tpl: Option<crate::jinja_chat::ChatTemplate>,
    ) -> (AppState, std::sync::mpsc::Receiver<Job>) {
        let (mut state, jobs) = app_state(tpl);
        state.tokenizer = Arc::new(crate::prefix_anchor::testing::qwen_like_tokenizer());
        (state, jobs)
    }

    /// Play the actor for ONE job: take it, finish it at once, hand back what it carried.
    fn finish_one(jobs: std::sync::mpsc::Receiver<Job>) -> std::thread::JoinHandle<Job> {
        std::thread::spawn(move || {
            let job = jobs.recv().unwrap();
            job.token_tx
                .blocking_send(StreamItem::Done(FinishReason::Stop))
                .unwrap();
            job
        })
    }

    /// Assert the job's anchor sits before `user` in the prompt it will run and keeps `keep`.
    fn assert_job_anchor(state: &AppState, job: &Job, user: &str, keep: &[&str]) -> usize {
        let n = job.prefix_anchor.expect("the job carries a prefix anchor");
        let prompt = state.tokenizer.decode(&job.prompt_ids, false).unwrap();
        crate::prefix_anchor::testing::assert_anchor_before_user(
            &state.tokenizer,
            &prompt,
            &job.prompt_ids,
            n,
            user,
            keep,
        );
        n
    }

    /// Claude Code's shape on `/v1/messages`: a long `system` (as text blocks), tools, a first
    /// user message.
    fn claude_code_like(user: &str, tools: bool) -> MessagesRequest {
        let system = crate::prefix_anchor::testing::long_system(3000);
        let mut body = json!({"max_tokens": 16,
            "system": [{"type": "text", "text": system}],
            "messages": [{"role": "user", "content": user}]});
        if tools {
            body["tools"] = json!([{"name": "read_file", "description": "Read a file",
                "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}},
                {"name": "run_shell", "description": "Run a command",
                "input_schema": {"type": "object", "properties": {"cmd": {"type": "string"}}}}]);
        }
        parse(body)
    }

    #[tokio::test]
    async fn messages_jobs_carry_the_anchor_on_the_native_path() {
        let (state, jobs) = anchor_state(Some(qwen38()));
        let actor = finish_one(jobs);
        let req = claude_code_like("Fix the failing test", true);
        assert_ne!(
            prompt_ids_of(&state, &req).unwrap().format,
            ToolFormat::None
        );
        assert_eq!(state.prefix_anchors.len(), 0, "a count computes no anchor");
        let _ = messages(State(state.clone()), Json(req)).await;
        let job = actor.join().unwrap();
        let sys = crate::prefix_anchor::testing::long_system(3000);
        assert_job_anchor(
            &state,
            &job,
            "Fix the failing test",
            &[sys.trim(), "read_file", "run_shell"],
        );
        // Two entries since 2026-09-27: the system anchor's and the tools anchor's. (Two small
        // schemas plus Qwen3.8's tool-calling instructions clear the floor on this byte-level
        // vocabulary: a tools anchor, before the system anchor.)
        assert_eq!(state.prefix_anchors.len(), 2);
        let t = job.tools_anchor.expect("a tools anchor");
        assert!(t < job.prefix_anchor.unwrap());
    }

    /// Claude Code's first message with project context: `<system-reminder>` blocks (AGENTS.md)
    /// then the user's words. The job's anchor is the CONTEXT anchor — past the project context,
    /// before the words — and two sessions with the same context and another question share it.
    #[tokio::test]
    async fn messages_jobs_anchor_past_the_project_context() {
        let agents = format!(
            "<system-reminder>\nAGENTS.md for this project. {}\n</system-reminder>",
            "Keep handlers small; validate input; test every change. ".repeat(10)
        );
        let with_context = |user: &str| {
            let mut req = claude_code_like("x", true);
            req.messages[0].content = Content::Blocks(vec![
                Block::Text {
                    text: agents.clone(),
                },
                Block::Text {
                    text: user.to_string(),
                },
            ]);
            req
        };
        let mut seen = Vec::new();
        for user in ["Fix the failing test", "What does this repository do"] {
            let (mut state, jobs) = anchor_state(Some(qwen38()));
            // The test state's context is 4,096 tokens and this tokenizer is byte-level: the
            // system prompt alone is ~3,900. A refused request queues no job, and the join below
            // would wait forever.
            state.max_tokens_cap = 8192;
            let actor = finish_one(jobs);
            let resp = messages(State(state.clone()), Json(with_context(user))).await;
            assert_eq!(resp.status(), StatusCode::OK);
            let job = actor.join().unwrap();
            let n = assert_job_anchor(
                &state,
                &job,
                user,
                &[
                    "read_file",
                    "AGENTS.md for this project",
                    "test every change",
                ],
            );
            assert!(job.tools_anchor.is_some_and(|t| t < n));
            seen.push((n, job.prompt_ids[..n].to_vec()));
        }
        assert_eq!(
            seen[0], seen[1],
            "the same project context, the same anchor"
        );
        // Without context blocks the anchor stays the system anchor (before the user's words).
        let (state, jobs) = anchor_state(Some(qwen38()));
        let actor = finish_one(jobs);
        let _ = messages(
            State(state.clone()),
            Json(claude_code_like("Fix the failing test", true)),
        )
        .await;
        let plain = assert_job_anchor(
            &state,
            &actor.join().unwrap(),
            "Fix the failing test",
            &["read_file"],
        );
        assert!(
            plain < seen[0].0,
            "the context anchor is deeper than the system anchor"
        );
    }

    /// No template: the fenced protocol (tool prompt + system text, hand-written template); and
    /// no tools: the plain hand-written path. Both anchor. A short system prompt and no tools
    /// does not (under 1,024 tokens).
    #[tokio::test]
    async fn messages_jobs_carry_the_anchor_on_the_hand_written_paths() {
        let sys = crate::prefix_anchor::testing::long_system(3000);
        for (tools, keep) in [
            (true, vec!["read_file", sys.trim()]),
            (false, vec![sys.trim()]),
        ] {
            let (state, jobs) = anchor_state(None);
            let actor = finish_one(jobs);
            let req = claude_code_like("List the files", tools);
            let _ = messages(State(state.clone()), Json(req)).await;
            let job = actor.join().unwrap();
            assert_job_anchor(&state, &job, "List the files", &keep);
        }
        let (state, jobs) = anchor_state(Some(qwen38()));
        let actor = finish_one(jobs);
        let short = parse(json!({"max_tokens": 16, "system": "Be terse.",
            "messages": [{"role": "user", "content": "hi"}]}));
        let _ = messages(State(state), Json(short)).await;
        assert_eq!(actor.join().unwrap().prefix_anchor, None);
    }

    /// `/v1/chat/completions` on each of its three paths — native (Qwen3.8 template with tools),
    /// fenced (tools without a template) and plain (no tools) — queues the anchor too.
    #[tokio::test]
    async fn chat_completions_jobs_carry_the_anchor_on_every_path() {
        let sys = crate::prefix_anchor::testing::long_system(3000);
        let tools = json!([{"type": "function", "function": {"name": "read_file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]);
        let body = |with_tools: bool| {
            let mut b = json!({"max_tokens": 16, "messages": [
                {"role": "system", "content": sys},
                {"role": "user", "content": "Why does the build fail?"}]});
            if with_tools {
                b["tools"] = tools.clone();
            }
            axum::body::Bytes::from(serde_json::to_vec(&b).unwrap())
        };
        for (tpl, with_tools, keep) in [
            (
                Some(qwen38()),
                true,
                vec![sys.trim(), "read_file", "</tools>"],
            ),
            (None, true, vec![sys.trim(), "read_file"]),
            (Some(qwen38()), false, vec![sys.trim()]),
        ] {
            let (state, jobs) = anchor_state(tpl);
            let actor = finish_one(jobs);
            let _ = super::super::chat_completions(State(state.clone()), body(with_tools)).await;
            let job = actor.join().unwrap();
            assert_job_anchor(&state, &job, "Why does the build fail?", &keep);
        }
    }

    /// The hand-written path at the request's EFFORT (review, 2026-09-26). Qwen3.8's thinking
    /// template writes the effort sentence first in the system turn, so sentinels rendered at the
    /// server default (`xhigh`) shared only `<|im_start|>system\n` with a `low`, `medium` or
    /// `disabled` prompt: no anchor, silently. And the cache key had no effort in it, so the first
    /// effort seen fixed the prefix for every other. Each effort now anchors, in any order.
    #[test]
    fn a_hand_written_messages_anchor_follows_the_requests_effort() {
        let (state, _jobs) = anchor_state(None);
        let sys = crate::prefix_anchor::testing::long_system(3000);
        for (i, (thinking, effort_field)) in [
            (Some(json!({"type": "adaptive"})), None),
            (None, Some("low")),
            (None, Some("medium")),
            (Some(json!({"type": "disabled"})), None),
            (None, Some("max")),
        ]
        .into_iter()
        .enumerate()
        {
            let mut body = json!({"max_tokens": 16,
                "system": [{"type": "text", "text": sys}],
                "messages": [{"role": "user", "content": "List the files"}]});
            if let Some(t) = thinking {
                body["thinking"] = t;
            }
            if let Some(e) = effort_field {
                body["output_config"] = json!({"effort": e});
            }
            let req = parse(body);
            let (effort, _) = thinking_vars(&req, false, None);
            let prompt = crate::chat::qwen3_think_template(&flatten(&req).unwrap(), effort);
            let ids = encode_prompt(&state.tokenizer, &prompt).unwrap();
            let n = hand_written_anchor(&state, &req, effort, &ids, |m, e| {
                crate::chat::qwen3_think_template(m, e)
            })
            .system
            .unwrap_or_else(|| panic!("case {i} ({effort:?}): no anchor on {prompt:.200}"));
            crate::prefix_anchor::testing::assert_anchor_before_user(
                &state.tokenizer,
                &prompt,
                &ids,
                n,
                "List the files",
                &[sys.trim()],
            );
        }
        // One cached prefix per distinct effort (none/low/medium/xhigh; `None` and `max` both
        // render xhigh but are distinct keys).
        assert_eq!(state.prefix_anchors.len(), 5);
    }

    // --------------------------------------------------------------------------------------
    // TOOLS ANCHORS (2026-09-27) through the real handlers: Claude Code sessions in two working
    // directories share the tools block and not the system text; each job carries a tools
    // anchor at the end of the tools block, the same for both, before the system text.
    // --------------------------------------------------------------------------------------

    /// Play the actor for `n` jobs in turn: take each, finish it at once, hand back all of them.
    fn finish_n(
        jobs: std::sync::mpsc::Receiver<Job>,
        n: usize,
    ) -> std::thread::JoinHandle<Vec<Job>> {
        std::thread::spawn(move || {
            (0..n)
                .map(|_| {
                    // A handler that refused its request queues nothing: fail, do not hang.
                    let job = jobs
                        .recv_timeout(std::time::Duration::from_secs(30))
                        .expect("the handler queued a job");
                    job.token_tx
                        .blocking_send(StreamItem::Done(FinishReason::Stop))
                        .unwrap();
                    job
                })
                .collect()
        })
    }

    /// Claude Code in `cwd`: its instructions, then the environment block naming the directory.
    fn system_in(cwd: &str) -> String {
        format!(
            "{}\nWorking directory: {cwd}\nPlatform: darwin",
            crate::prefix_anchor::testing::long_system(3000)
        )
    }

    /// Both sessions' jobs carry a tools anchor that ends before their system text and includes
    /// `keep` — the same anchor, the same tokens before it — and is before the system anchor.
    fn assert_tools_anchors(state: &AppState, jobs: &[Job], systems: &[String], keep: &[&str]) {
        let mut anchors = Vec::new();
        for (job, sys) in jobs.iter().zip(systems) {
            let t = job.tools_anchor.expect("the job carries a tools anchor");
            let s = job.prefix_anchor.expect("and a system anchor");
            assert!(t < s, "tools {t}, system {s}");
            let prompt = state.tokenizer.decode(&job.prompt_ids, false).unwrap();
            let head = state.tokenizer.decode(&job.prompt_ids[..t], false).unwrap();
            assert!(prompt.starts_with(&head));
            let sys_at = prompt
                .find(sys.trim())
                .expect("the system text is in the prompt");
            assert!(
                head.len() <= sys_at,
                "the tools anchor ends before the system text"
            );
            for k in keep {
                assert!(head.contains(k), "the tools anchor must include {k:?}");
            }
            anchors.push(t);
        }
        assert_eq!(
            anchors[0], anchors[1],
            "the same tools anchor in both directories"
        );
        let t = anchors[0];
        assert_eq!(jobs[0].prompt_ids[..t], jobs[1].prompt_ids[..t]);
    }

    /// `/v1/messages` on the native path (Qwen3.8: tools rendered first) and on the hand-written
    /// fenced path (no template: the tool prompt, then the system text).
    #[tokio::test]
    async fn messages_jobs_carry_a_tools_anchor_across_working_directories() {
        let tools: Vec<Value> = (0..30)
            .map(|i| {
                json!({"name": format!("tool_{i}"),
                    "description": format!("Tool number {i} does one careful thing"),
                    "input_schema": {"type": "object", "properties": {"arg": {
                        "type": "string", "description": "the input"}}}})
            })
            .collect();
        for (tpl, keep) in [
            (Some(qwen38()), vec!["tool_0", "tool_29", "</tools>"]),
            (None, vec!["tool_0", "tool_29"]),
        ] {
            let native = tpl.is_some();
            let (mut state, rx) = anchor_state(tpl);
            state.max_tokens_cap = 1 << 20; // 30 tool schemas on a byte-level vocabulary
            let actor = finish_n(rx, 2);
            let mut systems = Vec::new();
            for (cwd, user) in [("/work/a/one", "Fix the test"), ("/work/a/two", "List it")] {
                let sys = system_in(cwd);
                let req = parse(json!({"max_tokens": 16,
                    "system": [{"type": "text", "text": sys}],
                    "tools": tools,
                    "messages": [{"role": "user", "content": user}]}));
                assert_eq!(
                    prompt_ids_of(&state, &req).unwrap().format != ToolFormat::None,
                    native
                );
                let _ = messages(State(state.clone()), Json(req)).await;
                systems.push(sys);
            }
            assert_tools_anchors(&state, &actor.join().unwrap(), &systems, &keep);
        }
    }

    /// `/v1/chat/completions` on the native path with tools, and the plain path without: the
    /// tools anchor is there with tools and absent without.
    #[tokio::test]
    async fn chat_completions_jobs_carry_a_tools_anchor_only_with_tools() {
        let tools: Vec<Value> = (0..30)
            .map(|i| {
                json!({"type": "function", "function": {"name": format!("tool_{i}"),
                    "description": format!("Tool number {i} does one careful thing"),
                    "parameters": {"type": "object", "properties": {"arg": {"type": "string"}}}}})
            })
            .collect();
        let body = |sys: &str, with_tools: bool| {
            let mut b = json!({"max_tokens": 16, "messages": [
                {"role": "system", "content": sys},
                {"role": "user", "content": "Why does the build fail?"}]});
            if with_tools {
                b["tools"] = Value::Array(tools.clone());
            }
            axum::body::Bytes::from(serde_json::to_vec(&b).unwrap())
        };
        let (mut state, rx) = anchor_state(Some(qwen38()));
        state.max_tokens_cap = 1 << 20; // 30 tool schemas on a byte-level vocabulary
        let actor = finish_n(rx, 3);
        let systems = vec![system_in("/work/a/one"), system_in("/work/a/two")];
        for (sys, with_tools) in [
            (&systems[0], true),
            (&systems[1], true),
            (&systems[0], false),
        ] {
            let _ =
                super::super::chat_completions(State(state.clone()), body(sys, with_tools)).await;
        }
        let mut jobs = actor.join().unwrap();
        let plain = jobs.pop().unwrap();
        assert_tools_anchors(&state, &jobs, &systems, &["tool_0", "tool_29", "</tools>"]);
        assert!(plain.prefix_anchor.is_some());
        assert_eq!(plain.tools_anchor, None, "no tools, no tools anchor");
    }
}
