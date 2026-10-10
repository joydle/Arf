//! The model's OWN chat template — the GGUF's `tokenizer.chat_template` Jinja — rendered with
//! minijinja, for requests that carry tools or tool history.
//!
//! WHY (2026-09-26, the first coding-agent sessions): the hand-written templates in `chat.rs`
//! know nothing about tools, so a tool request was served gemma-4's fenced-JSON protocol
//! (`chat::tool_system_prompt`) and the model's own history was flattened to role + content.
//! Qwen3.8 was trained on a different protocol — tools as a `<tools>` JSON list in the system
//! turn, calls as `<tool_call><function=NAME><parameter=K>…` XML, results as `<tool_response>`
//! inside user turns — and the live runs showed exactly that seam: OpenCode repeated an edit it
//! could no longer see (the history carried no calls), and two bare `<tool_call>` lines leaked
//! into its final text, which is the model starting its NATIVE format under a foreign prompt.
//!
//! So a tool request is rendered by the template the model shipped with, and the reply is parsed
//! in the format that template taught ([`ToolFormat`], detected from the template's source).
//! That is model-agnostic by construction: whatever family the GGUF is, its own template decides.
//! A template with no tool format Arf can parse (Gemma 3/4, Llama 3/4, Muse Glimmer) keeps the
//! fenced-JSON protocol, and so does any render failure — the caller falls back, never errors.
//!
//! Plain chat does NOT come through here: its hand-written templates are byte-for-byte what they
//! were (the Qwen3.8 thinking template's acceptance numbers were measured on those bytes).
//!
//! HF-compat details, each one a way a template silently renders differently than in
//! `transformers` (which is what the model's authors ran):
//! - `trim_blocks` + `lstrip_blocks` on, as in HF's `ImmutableSandboxedEnvironment`;
//! - `tojson` is Python's `json.dumps` (`", "` / `": "` separators, `ensure_ascii=False`, key
//!   order kept, NO HTML escaping) — minijinja's built-in escapes `<`/`>`/`&`/`'` and is compact;
//! - Python string/dict methods (`.startswith`, `.items()`, `.keys()`…) via `minijinja-contrib`;
//! - `raise_exception(msg)` fails the render (→ fallback) instead of being an unknown function;
//! - `bos_token` renders EMPTY: `encode_prompt` adds BOS itself, and a template that also wrote
//!   it would double it;
//! - `x | string` and `{{ x }}` are Python's `str()` — `['string', 'null']`, `True`, `None`,
//!   `1e-05` — not minijinja's JSON-ish Display (`["string", "null"]`, `true`, `none`,
//!   `0.00001`). Qwen3-Coder's template prints every non-string schema field and every
//!   non-string replayed argument this way. NOT covered: `~` concatenation of a list, map, float
//!   or bool still uses minijinja's Display (it cannot be overridden); no template on this disk
//!   does that — Qwen3-Coder applies `| string` before `~` in every such place.

use std::path::Path;

use minijinja::value::{Kwargs, Value, ValueKind};
use minijinja::{Environment, Error, ErrorKind, UndefinedBehavior};

use crate::http::ChatMessage;

/// How a template teaches the model to write a tool call, decided from the template's source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolFormat {
    /// `<tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n</function>\n</tool_call>`
    /// — Qwen3-Coder and Qwen3.8.
    QwenXml,
    /// `<tool_call>\n{"name": NAME, "arguments": {…}}\n</tool_call>` — Hermes; Qwen2.5 / Qwen3.
    HermesJson,
    /// Nothing Arf can parse (Gemma 3/4, Llama 3/4, Muse Glimmer, or no tool section at all).
    None,
}

impl ToolFormat {
    /// The short name used in logs.
    pub fn as_str(self) -> &'static str {
        match self {
            ToolFormat::QwenXml => "qwen-xml",
            ToolFormat::HermesJson => "hermes-json",
            ToolFormat::None => "none",
        }
    }
}

/// Detect the tool-call format a template teaches. XML is checked first: Qwen3.8's template also
/// contains `<tool_call>`, and would otherwise be taken for Hermes. Hermes needs a literal JSON
/// `"name"` key (bare, or escaped inside a double-quoted Jinja string, as Qwen3 writes it) — a
/// template that merely wraps some other syntax in `<tool_call>` (GLM's `<arg_key>` XML) is not
/// JSON and must not be parsed as it.
pub fn detect_tool_format(src: &str) -> ToolFormat {
    if src.contains("<function=") && src.contains("<parameter=") {
        ToolFormat::QwenXml
    } else if src.contains("<tool_call>")
        && (src.contains("\"name\"") || src.contains("\\\"name\\\""))
        && src.contains("arguments")
    {
        ToolFormat::HermesJson
    } else {
        ToolFormat::None
    }
}

/// The template variables a request sets beyond `messages` and `tools`.
#[derive(Debug, Clone, Default)]
pub struct TemplateVars {
    /// Append the assistant cue (always true for a completion request).
    pub add_generation_prompt: bool,
    /// Qwen's `enable_thinking`. `None` leaves it undefined (the template's own default).
    pub enable_thinking: Option<bool>,
    /// Qwen3.8's `reasoning_effort` (`xhigh` / `medium` / `low`). `None` = template default.
    pub reasoning_effort: Option<String>,
    /// Qwen3.8's `preserve_thinking` (replay earlier turns' reasoning). `None` = template default.
    pub preserve_thinking: Option<bool>,
}

/// A compiled chat template plus the tool format it teaches.
pub struct ChatTemplate {
    env: Environment<'static>,
    tool_format: ToolFormat,
    eos_token: String,
}

const TEMPLATE_NAME: &str = "chat_template";

impl ChatTemplate {
    /// Compile `source`. `Err` carries minijinja's message (a syntax error in the template).
    pub fn new(source: String, eos_token: Option<String>) -> Result<Self, String> {
        let tool_format = detect_tool_format(&source);
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_undefined_behavior(UndefinedBehavior::Lenient);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        minijinja_contrib::add_to_environment(&mut env);
        env.add_filter("tojson", tojson_filter);
        env.add_filter("string", py_string_filter);
        env.set_formatter(py_formatter);
        env.add_function("raise_exception", raise_exception);
        env.add_function("strftime_now", strftime_now);
        env.add_template_owned(TEMPLATE_NAME, source)
            .map_err(|e| format!("chat template does not compile: {e}"))?;
        Ok(ChatTemplate {
            env,
            tool_format,
            eos_token: eos_token.unwrap_or_default(),
        })
    }

    pub fn tool_format(&self) -> ToolFormat {
        self.tool_format
    }

    /// Render the conversation. `messages` is a sequence of message maps (see
    /// [`message_values`]); `tools` is the request's `tools` array exactly as sent (key order
    /// kept), or `None` to leave it undefined. `Err` is the render error — a `raise_exception`
    /// in the template, or an operation it could not do — and the caller falls back on it.
    pub fn render(
        &self,
        messages: Value,
        tools: Option<Value>,
        vars: &TemplateVars,
    ) -> Result<String, String> {
        let mut ctx: Vec<(&str, Value)> = vec![
            ("messages", messages),
            (
                "add_generation_prompt",
                Value::from(vars.add_generation_prompt),
            ),
            ("bos_token", Value::from("")),
            ("eos_token", Value::from(self.eos_token.as_str())),
        ];
        if let Some(t) = tools {
            ctx.push(("tools", t));
        }
        if let Some(b) = vars.enable_thinking {
            ctx.push(("enable_thinking", Value::from(b)));
        }
        if let Some(e) = &vars.reasoning_effort {
            ctx.push(("reasoning_effort", Value::from(e.as_str())));
        }
        if let Some(b) = vars.preserve_thinking {
            ctx.push(("preserve_thinking", Value::from(b)));
        }
        let tmpl = self
            .env
            .get_template(TEMPLATE_NAME)
            .map_err(|e| e.to_string())?;
        tmpl.render(Value::from_iter(ctx)).map_err(|e| {
            let mut msg = e.to_string();
            if let Some(d) = e.detail() {
                if !msg.contains(d) {
                    msg.push_str(": ");
                    msg.push_str(d);
                }
            }
            msg
        })
    }

    /// Render a request's conversation ([`message_values`]), retrying once with any LATE
    /// system/developer message demoted to a user turn when the first render fails.
    ///
    /// Why: Qwen3.8's template raises `System message must be at the beginning.` for a system
    /// message after the first turn (HF and vLLM would refuse the request outright). Here a
    /// failed render drops the request to the fenced-JSON protocol, and doing that in the middle
    /// of an agent session switches the protocol under the model's feet — the history it sees
    /// suddenly stops matching what it wrote. A late instruction read as a user turn keeps the
    /// session on its own format. Templates that accept late system turns (Qwen3's) render on
    /// the first try and are untouched.
    pub fn render_chat(
        &self,
        messages: &[ChatMessage],
        tools: Option<Value>,
        vars: &TemplateVars,
    ) -> Result<String, String> {
        let first = self.render(message_values(messages), tools.clone(), vars);
        let Err(e) = first else { return first };
        let is_sys = |m: &ChatMessage| m.role == "system" || m.role == "developer";
        let lead = messages.iter().take_while(|m| is_sys(m)).count();
        if !messages[lead..].iter().any(is_sys) {
            return Err(e);
        }
        let mut demoted = messages.to_vec();
        for m in demoted[lead..].iter_mut().filter(|m| is_sys(m)) {
            m.role = "user".into();
        }
        self.render(message_values(&demoted), tools, vars)
            .map_err(|_| e)
    }
}

/// The request's messages as template values: `role`, `content` (always a string — templates
/// concatenate it), and when present `reasoning_content`, `tool_calls` (arguments parsed into a
/// MAPPING, see [`arguments_value`]), `tool_call_id` and `name`.
///
/// `developer` is renamed `system`: Qwen3.8's template treats them alike, and Qwen3's and
/// Qwen3-Coder's have no `developer` branch at all — the message would render as nothing.
pub fn message_values(messages: &[ChatMessage]) -> Value {
    Value::from_iter(messages.iter().map(|m| {
        let role = if m.role == "developer" {
            "system"
        } else {
            m.role.as_str()
        };
        let mut fields: Vec<(&str, Value)> = vec![
            ("role", Value::from(role)),
            ("content", Value::from(m.content.as_str())),
        ];
        if let Some(r) = &m.reasoning_content {
            fields.push(("reasoning_content", Value::from(r.as_str())));
        }
        if !m.tool_calls.is_empty() {
            let calls = m.tool_calls.iter().map(|c| {
                let function: Vec<(&str, Value)> = vec![
                    ("name", Value::from(c.function.name.as_str())),
                    ("arguments", arguments_value(&c.function.arguments)),
                ];
                let call: Vec<(&str, Value)> = vec![
                    ("id", Value::from(c.id.as_str())),
                    ("type", Value::from("function")),
                    ("function", Value::from_iter(function)),
                ];
                Value::from_iter(call)
            });
            fields.push(("tool_calls", Value::from_iter(calls)));
        }
        if let Some(id) = &m.tool_call_id {
            fields.push(("tool_call_id", Value::from(id.as_str())));
        }
        if let Some(n) = &m.name {
            fields.push(("name", Value::from(n.as_str())));
        }
        Value::from_iter(fields)
    }))
}

/// An OpenAI `function.arguments` STRING as the mapping templates need.
///
/// Qwen3.8's template REQUIRES a mapping (a non-empty string raises), and Qwen3-Coder's iterates
/// `arguments|items`; vLLM likewise `json.loads` the string before rendering. Decisions:
/// - empty / whitespace → `{}` (a no-argument call);
/// - a JSON object → that object, key order kept;
/// - a JSON string that itself holds an object (double-encoded) → the inner object;
/// - anything else (unparseable, or an array / number) → `{"_raw": <the string>}`. Never an error:
///   a raise here would drop the whole conversation to the fallback protocol mid-session. The
///   model then sees what the client says it called, and the tool result that follows tells it
///   how that went. Arf's own parsers always emit an object, so this only fires for a history a
///   client edited or produced elsewhere.
pub fn arguments_value(raw: &str) -> Value {
    let t = raw.trim();
    if t.is_empty() {
        return Value::from_iter(Vec::<(&str, Value)>::new());
    }
    if let Ok(v) = serde_json::from_str::<Value>(t) {
        match v.kind() {
            ValueKind::Map => return v,
            ValueKind::String => {
                if let Some(inner) = v
                    .as_str()
                    .and_then(|s| serde_json::from_str::<Value>(s).ok())
                {
                    if inner.kind() == ValueKind::Map {
                        return inner;
                    }
                }
            }
            _ => {}
        }
    }
    Value::from_iter([("_raw", Value::from(raw))])
}

/// Parse JSON into an order-preserving value (minijinja's map is an `IndexMap` here). Used for
/// the request's `tools` and by the tool-call parsers, so key order survives a round trip —
/// `serde_json::Value` would sort it (the workspace does not enable `preserve_order`, and
/// turning it on would change JSON output order in every crate).
pub fn parse_ordered(json: &str) -> Option<Value> {
    serde_json::from_str::<Value>(json).ok()
}

/// Compact JSON of a value, key order kept (`{"a":1,"b":"x"}`) — the OpenAI `arguments` form.
pub fn to_compact_json(v: &Value) -> String {
    serde_json::to_string(v).unwrap_or_else(|_| "{}".into())
}

fn raise_exception(msg: String) -> Result<Value, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, msg))
}

// ----------------------------------------------------------------------------
// `tojson`, as HF transformers defines it: Python's json.dumps.
// ----------------------------------------------------------------------------

/// `x | tojson(indent=None, ensure_ascii=False, sort_keys=False, separators=None)`.
///
/// transformers overrides Jinja's `tojson` with `json.dumps(x, ensure_ascii=False, indent=…,
/// separators=…, sort_keys=…)` precisely because the Jinja/minijinja one HTML-escapes. The
/// difference is visible to the model: Qwen3.8 is shown every tool schema through this filter,
/// so `{"type": "function", …}` (with spaces) is what it was trained on, and a `<` in a
/// description must stay `<`, not `\u003c`.
fn tojson_filter(value: &Value, kwargs: Kwargs) -> Result<Value, Error> {
    let indent: Option<usize> = kwargs.get("indent")?;
    let ensure_ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    let sort_keys: Option<bool> = kwargs.get("sort_keys")?;
    let separators: Option<Vec<String>> = kwargs.get("separators")?;
    kwargs.assert_all_used()?;
    let (item_sep, key_sep) = match separators {
        Some(s) if s.len() == 2 => (s[0].clone(), s[1].clone()),
        Some(_) => {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "tojson: separators must be a pair",
            ))
        }
        // Python's defaults: ", " without indent, "," with it (the newline does the spacing).
        None if indent.is_some() => (",".into(), ": ".into()),
        None => (", ".into(), ": ".into()),
    };
    let opts = PyJson {
        indent,
        item_sep,
        key_sep,
        ensure_ascii: ensure_ascii.unwrap_or(false),
        sort_keys: sort_keys.unwrap_or(false),
    };
    let mut out = String::new();
    opts.write(value, 0, &mut out);
    Ok(Value::from_safe_string(out))
}

/// Python `json.dumps` with the options above. Public for the unit tests and for callers that
/// need the exact bytes a template would print.
pub fn py_json_dumps(value: &Value) -> String {
    let opts = PyJson {
        indent: None,
        item_sep: ", ".into(),
        key_sep: ": ".into(),
        ensure_ascii: false,
        sort_keys: false,
    };
    let mut out = String::new();
    opts.write(value, 0, &mut out);
    out
}

struct PyJson {
    indent: Option<usize>,
    item_sep: String,
    key_sep: String,
    ensure_ascii: bool,
    sort_keys: bool,
}

impl PyJson {
    fn newline(&self, depth: usize, out: &mut String) {
        if let Some(n) = self.indent {
            out.push('\n');
            out.push_str(&" ".repeat(n * depth));
        }
    }

    fn write(&self, v: &Value, depth: usize, out: &mut String) {
        match v.kind() {
            ValueKind::Undefined | ValueKind::None => out.push_str("null"),
            ValueKind::Bool => out.push_str(if v.is_true() { "true" } else { "false" }),
            ValueKind::Number => {
                if v.is_integer() {
                    out.push_str(&v.to_string());
                } else {
                    out.push_str(&py_float_repr(f64::try_from(v.clone()).unwrap_or(f64::NAN)));
                }
            }
            ValueKind::String => py_json_string(v.as_str().unwrap_or(""), self.ensure_ascii, out),
            ValueKind::Seq | ValueKind::Iterable => {
                let items: Vec<Value> = v.try_iter().map(|i| i.collect()).unwrap_or_default();
                if items.is_empty() {
                    out.push_str("[]");
                    return;
                }
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(&self.item_sep);
                    }
                    self.newline(depth + 1, out);
                    self.write(item, depth + 1, out);
                }
                self.newline(depth, out);
                out.push(']');
            }
            ValueKind::Map => {
                let mut keys: Vec<Value> = v.try_iter().map(|i| i.collect()).unwrap_or_default();
                if keys.is_empty() {
                    out.push_str("{}");
                    return;
                }
                if self.sort_keys {
                    keys.sort_by_key(|k| k.to_string());
                }
                out.push('{');
                for (i, k) in keys.iter().enumerate() {
                    if i > 0 {
                        out.push_str(&self.item_sep);
                    }
                    self.newline(depth + 1, out);
                    // Python turns non-string keys into strings; so does this.
                    let key = match k.as_str() {
                        Some(s) => s.to_string(),
                        None => k.to_string(),
                    };
                    py_json_string(&key, self.ensure_ascii, out);
                    out.push_str(&self.key_sep);
                    let item = v.get_item(k).unwrap_or(Value::UNDEFINED);
                    self.write(&item, depth + 1, out);
                }
                self.newline(depth, out);
                out.push('}');
            }
            // Bytes, plain objects, invalid values: Python would raise; print the string form
            // rather than fail a render over a value no real template passes to tojson.
            _ => py_json_string(&v.to_string(), self.ensure_ascii, out),
        }
    }
}

/// Python's JSON string encoder: `\"`, `\\`, `\b \f \n \r \t`, other controls as lowercase
/// `\u00XX`; with `ensure_ascii` every non-ASCII char as `\uXXXX` (surrogate pairs above U+FFFF).
pub(crate) fn py_json_string(s: &str, ensure_ascii: bool, out: &mut String) {
    use std::fmt::Write;
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c if ensure_ascii && !c.is_ascii() => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    let _ = write!(out, "\\u{:04x}", unit);
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python's `repr(float)`: shortest round-trip digits, `.0` on integral values, exponent form
/// below 1e-4 and from 1e16 with a sign and at least two exponent digits (`1e-05`, `1e+16`).
/// Rust's `{:?}` uses the same shortest digits and the same two thresholds; only the exponent
/// spelling differs.
fn py_float_repr(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 {
            "Infinity".into()
        } else {
            "-Infinity".into()
        };
    }
    let s = format!("{f:?}");
    match s.split_once('e') {
        Some((mant, exp)) => {
            let e: i32 = exp.parse().unwrap_or(0);
            format!("{mant}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
        }
        None => s,
    }
}

// ----------------------------------------------------------------------------
// `| string` and `{{ x }}` as Python's str(), which is what Jinja2 does.
// ----------------------------------------------------------------------------

/// `x | string`: Python `str(x)`. A string passes through; undefined is empty (as in Jinja2).
fn py_string_filter(value: &Value) -> Value {
    match value.kind() {
        ValueKind::String => value.clone(),
        ValueKind::Undefined => Value::from(""),
        _ => Value::from(py_str(value)),
    }
}

/// `{{ x }}` prints Python's `str(x)` for the values whose Python spelling differs from
/// minijinja's (none, bools, floats, lists, maps); everything else (strings, undefined, macro
/// output) through minijinja's own formatter.
fn py_formatter(
    out: &mut minijinja::Output,
    state: &minijinja::State,
    value: &Value,
) -> Result<(), Error> {
    match value.kind() {
        ValueKind::None | ValueKind::Bool | ValueKind::Seq | ValueKind::Map => {
            out.write_str(&py_str(value)).map_err(Error::from)
        }
        ValueKind::Number if !value.is_integer() => {
            out.write_str(&py_str(value)).map_err(Error::from)
        }
        _ => minijinja::escape_formatter(out, state, value),
    }
}

/// Python `str()` of a JSON-shaped value: strings as themselves, containers as their `repr`.
pub fn py_str(v: &Value) -> String {
    match v.kind() {
        ValueKind::String => v.as_str().unwrap_or("").to_string(),
        _ => {
            let mut out = String::new();
            py_repr(v, &mut out);
            out
        }
    }
}

/// Python `repr()`: `'single'` quoted strings, `True`/`False`/`None`, `[a, b]`, `{k: v}` with
/// `", "` and `": "`, floats as `repr(float)`.
fn py_repr(v: &Value, out: &mut String) {
    match v.kind() {
        ValueKind::Undefined | ValueKind::None => out.push_str("None"),
        ValueKind::Bool => out.push_str(if v.is_true() { "True" } else { "False" }),
        ValueKind::Number => {
            if v.is_integer() {
                out.push_str(&v.to_string());
            } else {
                let f = f64::try_from(v.clone()).unwrap_or(f64::NAN);
                out.push_str(&match py_float_repr(f).as_str() {
                    "NaN" => "nan".to_string(),
                    "Infinity" => "inf".to_string(),
                    "-Infinity" => "-inf".to_string(),
                    s => s.to_string(),
                });
            }
        }
        ValueKind::String => py_repr_str(v.as_str().unwrap_or(""), out),
        ValueKind::Seq | ValueKind::Iterable => {
            out.push('[');
            if let Ok(items) = v.try_iter() {
                for (i, item) in items.enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    py_repr(&item, out);
                }
            }
            out.push(']');
        }
        ValueKind::Map => {
            out.push('{');
            if let Ok(keys) = v.try_iter() {
                for (i, k) in keys.enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    py_repr(&k, out);
                    out.push_str(": ");
                    py_repr(&v.get_item(&k).unwrap_or(Value::UNDEFINED), out);
                }
            }
            out.push('}');
        }
        _ => out.push_str(&v.to_string()),
    }
}

/// Python's `repr(str)`: single quotes unless the text has a `'` and no `"`; `\\`, the quote,
/// `\n \r \t` escaped; other non-printables as `\xNN` / `\uNNNN`. Printable non-ASCII stays
/// as is. "Non-printable" here is the controls plus the separators and format
/// characters JSON text realistically carries (NBSP, U+2028/9, zero-width and BOM) — Python's
/// full rule is the Unicode category table, which this crate does not carry.
fn py_repr_str(s: &str, out: &mut String) {
    use std::fmt::Write;
    let q = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    out.push(q);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == q => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || (0x7f..=0xa0).contains(&(c as u32)) || c == '\u{ad}' => {
                let _ = write!(out, "\\x{:02x}", c as u32);
            }
            '\u{1680}'
            | '\u{2000}'..='\u{200f}'
            | '\u{2028}'..='\u{202f}'
            | '\u{205f}'..='\u{206f}'
            | '\u{180e}'
            | '\u{3000}'
            | '\u{feff}'
            | '\u{e000}'..='\u{f8ff}' => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push(q);
}

// ----------------------------------------------------------------------------
// `strftime_now(fmt)` — HF exposes it to templates (Llama 3.x prints today's date with it).
// ----------------------------------------------------------------------------

/// The current date/time formatted with the common `strftime` codes. UTC (HF uses local time;
/// this crate carries no timezone database, and a date line is all templates ask it for).
fn strftime_now(fmt: String) -> Result<Value, Error> {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64);
    Ok(Value::from(strftime_utc(&fmt, secs)))
}

/// `strftime` over a Unix timestamp (UTC): `%Y %y %m %d %e %j %H %I %M %S %p %b %B %a %A %%`.
/// An unknown code is printed as written, as glibc does.
fn strftime_utc(fmt: &str, secs: i64) -> String {
    const MONTHS: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    const DAYS: [&str; 7] = [
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
    ];
    let days = secs.div_euclid(86_400);
    let tod = secs.rem_euclid(86_400);
    let (h, mi, s) = (tod / 3600, (tod % 3600) / 60, tod % 60);
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    const CUM: [i64; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let yday = CUM[(m - 1) as usize] + d + i64::from(leap && m > 2);
    let weekday = DAYS[days.rem_euclid(7) as usize];
    let month = MONTHS[(m - 1) as usize];
    let mut out = String::new();
    let mut chars = fmt.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('Y') => out.push_str(&y.to_string()),
            Some('y') => out.push_str(&format!("{:02}", y.rem_euclid(100))),
            Some('m') => out.push_str(&format!("{m:02}")),
            Some('d') => out.push_str(&format!("{d:02}")),
            Some('e') => out.push_str(&format!("{d:>2}")),
            Some('j') => out.push_str(&format!("{yday:03}")),
            Some('H') => out.push_str(&format!("{h:02}")),
            Some('I') => out.push_str(&format!("{:02}", if h % 12 == 0 { 12 } else { h % 12 })),
            Some('M') => out.push_str(&format!("{mi:02}")),
            Some('S') => out.push_str(&format!("{s:02}")),
            Some('p') => out.push_str(if h < 12 { "AM" } else { "PM" }),
            Some('b') => out.push_str(&month[..3]),
            Some('B') => out.push_str(month),
            Some('a') => out.push_str(&weekday[..3]),
            Some('A') => out.push_str(weekday),
            Some('%') => out.push('%'),
            Some(other) => {
                out.push('%');
                out.push(other);
            }
            None => out.push('%'),
        }
    }
    out
}

// ----------------------------------------------------------------------------
// Loading the template at startup.
// ----------------------------------------------------------------------------

/// Find the model's chat template, in order:
/// 1. `ARF_CHAT_TEMPLATE=<file>` — an explicit override (a broken or missing embedded template);
/// 2. a GGUF's `tokenizer.chat_template`;
/// 3. beside a safetensors model: `chat_template.jinja`, else `tokenizer_config.json`'s
///    `chat_template` (a string, or HF's list of named templates — `tool_use` preferred).
///
/// Returns `(source, where it came from, eos token text)`.
pub fn find_template_source(
    model: &Path,
    tok: &arf_core::Tokenizer,
) -> Option<(String, String, Option<String>)> {
    if let Ok(p) = std::env::var("ARF_CHAT_TEMPLATE") {
        return match std::fs::read_to_string(&p) {
            Ok(s) => Some((s, format!("ARF_CHAT_TEMPLATE={p}"), None)),
            Err(e) => {
                eprintln!("[serve] ARF_CHAT_TEMPLATE={p}: cannot read ({e}) — ignored");
                None
            }
        };
    }
    if arf_core::model::gguf::is_gguf_file(model) {
        let g = arf_core::model::gguf::LazyGguf::open_raw(model).ok()?;
        let src = g.get_metadata_string("tokenizer.chat_template")?;
        let eos = g
            .get_metadata_u32("tokenizer.ggml.eos_token_id")
            .and_then(|id| tok.decode(&[id], false).ok())
            .filter(|s| !s.is_empty());
        return Some((src, "GGUF tokenizer.chat_template".into(), eos));
    }
    let dir = if model.is_dir() {
        model
    } else {
        model.parent()?
    };
    if let Ok(s) = std::fs::read_to_string(dir.join("chat_template.jinja")) {
        return Some((s, "chat_template.jinja".into(), None));
    }
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("tokenizer_config.json")).ok()?)
            .ok()?;
    let eos = match &cfg["eos_token"] {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(o) => o.get("content").and_then(|c| c.as_str()).map(String::from),
        _ => None,
    };
    let src = match &cfg["chat_template"] {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(list) => {
            let named = |n: &str| {
                list.iter()
                    .find(|t| t["name"] == n)
                    .and_then(|t| t["template"].as_str())
                    .map(String::from)
            };
            named("tool_use").or_else(|| named("default"))?
        }
        _ => return None,
    };
    Some((src, "tokenizer_config.json chat_template".into(), eos))
}

/// Load and compile the model's chat template for tool requests, logging what was found. `None`
/// (tool requests then use the fenced-JSON protocol) when there is no template, it does not
/// compile, or `ARF_NO_NATIVE_TOOLS=1` — the A/B opt-out.
pub fn load(model: &Path, tok: &arf_core::Tokenizer) -> Option<ChatTemplate> {
    if std::env::var_os("ARF_NO_NATIVE_TOOLS").is_some() {
        eprintln!(
            "[serve] ARF_NO_NATIVE_TOOLS set: the model's chat template is NOT used; tool \
             requests use the fenced-JSON protocol"
        );
        return None;
    }
    let Some((src, origin, eos)) = find_template_source(model, tok) else {
        eprintln!(
            "[serve] no chat template found with the model: tool requests use the fenced-JSON \
             protocol"
        );
        return None;
    };
    let bytes = src.len();
    match ChatTemplate::new(src, eos) {
        Ok(t) => {
            let fmt = t.tool_format();
            if fmt == ToolFormat::None {
                eprintln!(
                    "[serve] chat template: {origin} ({bytes} bytes) teaches no tool-call format \
                     Arf parses — tool requests use the fenced-JSON protocol"
                );
            } else {
                eprintln!(
                    "[serve] chat template: {origin} ({bytes} bytes), tool format {} — requests \
                     with tools or tool history render through it (ARF_NO_NATIVE_TOOLS=1 opts out)",
                    fmt.as_str()
                );
            }
            Some(t)
        }
        Err(e) => {
            eprintln!(
                "[serve] chat template: {origin} ({bytes} bytes) — {e}; tool requests use the \
                 fenced-JSON protocol"
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::{ToolCall, ToolCallFunction};

    /// Qwen3.8's template, verbatim from the GGUF (`tokenizer.chat_template` of
    /// Qwen3.8-27B-arf-g64-imat-Q4_1.gguf, identical to the Q4_K_M release's).
    const QWEN38: &str = include_str!("../testdata/qwen38_chat_template.jinja");
    /// Qwen3-Coder-30B-A3B-Instruct's template, verbatim from its GGUF.
    const QWEN3_CODER: &str = include_str!("../testdata/qwen3_coder_chat_template.jinja");
    /// THE REFERENCE, not written by hand: `agent_history()` + `tools_value()` rendered by Python
    /// Jinja2 3.1.6 set up exactly as HF transformers' `_compile_jinja_template` sets it up
    /// (`ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)`, loopcontrols,
    /// `tojson` = `json.dumps(ensure_ascii=False)`, `raise_exception`), arguments passed as dicts
    /// as vLLM does, `bos_token=""`, `add_generation_prompt=True`. For Qwen3.8 also
    /// `enable_thinking=True, reasoning_effort="medium"`. The minijinja render must match it byte
    /// for byte — the instrument proven against an independent implementation, not against itself.
    const QWEN38_HF: &str = include_str!("../testdata/qwen38_three_turn.hf.txt");
    const QWEN3_CODER_HF: &str = include_str!("../testdata/qwen3_coder_three_turn.hf.txt");

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage::text(role, content)
    }

    fn call(id: &str, name: &str, args: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            r#type: "function".into(),
            function: ToolCallFunction {
                name: name.into(),
                arguments: args.into(),
            },
        }
    }

    fn tools_value() -> Value {
        parse_ordered(
            r#"[{"type":"function","function":{"name":"read","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
                {"type":"function","function":{"name":"edit","description":"Replace oldText with newText in a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"oldText":{"type":"string"},"newText":{"type":"string"}},"required":["path","oldText","newText"]}}}]"#,
        )
        .unwrap()
    }

    /// The 3-turn agent conversation from the live-test plan: a request, the model's `read`
    /// call, its result, the model's `edit` call (a multi-line value), its result.
    fn agent_history() -> Vec<ChatMessage> {
        let mut read = msg("assistant", "");
        read.tool_calls = vec![call("call_1", "read", r#"{"path":"calc.py"}"#)];
        let mut read_result = msg("tool", "def add(a, b):\n    return a - b\n");
        read_result.tool_call_id = Some("call_1".into());
        let mut edit = msg("assistant", "The bug is the minus sign.");
        edit.reasoning_content = Some("add() subtracts.".into());
        edit.tool_calls = vec![call(
            "call_2",
            "edit",
            r#"{"path":"calc.py","oldText":"return a - b","newText":"return a + b"}"#,
        )];
        let mut edit_result = msg("tool", "Edited calc.py");
        edit_result.tool_call_id = Some("call_2".into());
        vec![
            msg("system", "You are a coding agent."),
            msg("user", "calc.py has a bug in add(). Fix it."),
            read,
            read_result,
            edit,
            edit_result,
        ]
    }

    #[test]
    fn detects_the_tool_format_from_the_source() {
        assert_eq!(detect_tool_format(QWEN38), ToolFormat::QwenXml);
        assert_eq!(detect_tool_format(QWEN3_CODER), ToolFormat::QwenXml);
        // Qwen2.5 / Qwen3 (Hermes) shape: JSON inside <tool_call>.
        let hermes = r#"{{- '<tool_call>\n{"name": "' }}{{- tool_call.name }}{{- '", "arguments": ' }}{{- tool_call.arguments | tojson }}{{- '}\n</tool_call>' }}"#;
        assert_eq!(detect_tool_format(hermes), ToolFormat::HermesJson);
        // The instruction half of Qwen3's template, escaped inside a double-quoted string.
        let escaped = r#"{{- "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>" }}"#;
        assert_eq!(detect_tool_format(escaped), ToolFormat::HermesJson);
        // `<tool_call>` around a non-JSON syntax (GLM-4.5's `<arg_key>`) is not Hermes.
        let glm = "{{ '<tool_call>' + tc.name }}{% for k, v in tc.arguments.items() %}<arg_key>{{ k }}</arg_key>{% endfor %}";
        assert_eq!(detect_tool_format(glm), ToolFormat::None);
        // Gemma 4's own markers are `<|tool_call>` — not a format Arf parses.
        assert_eq!(
            detect_tool_format("{{ '<|tool_call>call:' + name }}{{ arguments }}"),
            ToolFormat::None
        );
        assert_eq!(
            detect_tool_format("{% for m in messages %}{{ m.content }}{% endfor %}"),
            ToolFormat::None
        );
    }

    /// The exact structure Qwen3.8's own template produces for a 3-turn tool conversation:
    /// tools as a `<tools>` JSON list (Python json.dumps spacing), each call replayed as
    /// `<tool_call><function=…><parameter=…>` XML in the model's argument order, each result
    /// as `<tool_response>` inside a user turn, and the pre-opened `<think>` cue.
    #[test]
    fn qwen38_renders_a_three_turn_tool_conversation() {
        let t = ChatTemplate::new(QWEN38.into(), Some("<|im_end|>".into())).unwrap();
        assert_eq!(t.tool_format(), ToolFormat::QwenXml);
        let vars = TemplateVars {
            add_generation_prompt: true,
            enable_thinking: Some(true),
            reasoning_effort: Some("medium".into()),
            preserve_thinking: None,
        };
        let p = t
            .render(message_values(&agent_history()), Some(tools_value()), &vars)
            .unwrap();
        let expected_system = concat!(
            "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n",
            r#"{"type": "function", "function": {"name": "read", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}"#,
            "\n",
            r#"{"type": "function", "function": {"name": "edit", "description": "Replace oldText with newText in a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "oldText": {"type": "string"}, "newText": {"type": "string"}}, "required": ["path", "oldText", "newText"]}}}"#,
            "\n</tools>"
        );
        assert!(p.starts_with(expected_system), "{p}");
        // The client's system text is merged after the tool block, in the SAME system turn.
        assert!(
            p.contains("</IMPORTANT>\n\nYou are a coding agent.<|im_end|>\n"),
            "{p}"
        );
        assert_eq!(p.matches("<|im_start|>system").count(), 1, "{p}");
        let expected_turns = concat!(
            "<|im_start|>user\ncalc.py has a bug in add(). Fix it.<|im_end|>\n",
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            "<tool_call>\n<function=read>\n<parameter=path>\ncalc.py\n</parameter>\n</function>\n</tool_call><|im_end|>\n",
            "<|im_start|>user\n<tool_response>\ndef add(a, b):\n    return a - b\n</tool_response><|im_end|>\n",
            "<|im_start|>assistant\n<think>\nadd() subtracts.\n</think>\n\nThe bug is the minus sign.\n\n",
            "<tool_call>\n<function=edit>\n<parameter=path>\ncalc.py\n</parameter>\n",
            "<parameter=oldText>\nreturn a - b\n</parameter>\n<parameter=newText>\nreturn a + b\n</parameter>\n",
            "</function>\n</tool_call><|im_end|>\n",
            "<|im_start|>user\n<tool_response>\nEdited calc.py\n</tool_response><|im_end|>\n",
            "<|im_start|>assistant\n<think>\n",
        );
        assert!(p.ends_with(expected_turns), "{p}");
        // And the whole prompt, byte for byte, is what HF transformers renders.
        assert_eq!(p, QWEN38_HF);
    }

    /// `reasoning_effort` and `enable_thinking` reach the template: low adds its instruction;
    /// thinking off pre-closes the think block; an unknown effort raises inside the template.
    #[test]
    fn qwen38_effort_and_thinking_switches() {
        let t = ChatTemplate::new(QWEN38.into(), None).unwrap();
        let msgs = message_values(&[msg("user", "hi")]);
        let low = TemplateVars {
            add_generation_prompt: true,
            reasoning_effort: Some("low".into()),
            ..Default::default()
        };
        let p = t.render(msgs.clone(), Some(tools_value()), &low).unwrap();
        assert!(
            p.starts_with("<|im_start|>system\nReasoning effort is set to low."),
            "{p}"
        );
        let off = TemplateVars {
            add_generation_prompt: true,
            enable_thinking: Some(false),
            ..Default::default()
        };
        let p = t.render(msgs.clone(), None, &off).unwrap();
        assert!(
            p.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "{p}"
        );
        let bad = TemplateVars {
            add_generation_prompt: true,
            reasoning_effort: Some("turbo".into()),
            ..Default::default()
        };
        let err = t.render(msgs, None, &bad).unwrap_err();
        assert!(err.contains("Unexpected reasoning effort"), "{err}");
    }

    /// A system message after the first turn: Qwen3.8's template raises, so `render_chat`
    /// retries with it demoted to a user turn instead of failing the request over to the other
    /// protocol. A render error with no late system message is returned as is.
    #[test]
    fn late_system_message_is_demoted_not_fatal() {
        let t = ChatTemplate::new(QWEN38.into(), None).unwrap();
        let vars = TemplateVars {
            add_generation_prompt: true,
            ..Default::default()
        };
        let msgs = vec![
            msg("system", "You are a coding agent."),
            msg("user", "fix it"),
            msg("assistant", "Done."),
            msg("system", "Plan mode is now OFF."),
            msg("user", "thanks"),
        ];
        let raw = t.render(message_values(&msgs), Some(tools_value()), &vars);
        assert!(raw
            .unwrap_err()
            .contains("System message must be at the beginning"));
        let p = t.render_chat(&msgs, Some(tools_value()), &vars).unwrap();
        assert!(
            p.contains("<|im_start|>user\nPlan mode is now OFF.<|im_end|>\n"),
            "{p}"
        );
        assert_eq!(p.matches("<|im_start|>system").count(), 1, "{p}");
        let bad = TemplateVars {
            add_generation_prompt: true,
            reasoning_effort: Some("turbo".into()),
            ..Default::default()
        };
        let err = t.render_chat(&msgs[..2], None, &bad).unwrap_err();
        assert!(err.contains("Unexpected reasoning effort"), "{err}");
    }

    /// Arguments that are not an object must not raise (the template demands a mapping).
    #[test]
    fn unparseable_arguments_render_as_raw_not_an_error() {
        let t = ChatTemplate::new(QWEN38.into(), None).unwrap();
        let mut a = msg("assistant", "");
        a.tool_calls = vec![call("c", "bash", "ls -la {oops")];
        let mut empty = msg("assistant", "");
        empty.tool_calls = vec![call("d", "status", "")];
        let vars = TemplateVars {
            add_generation_prompt: true,
            ..Default::default()
        };
        let p = t
            .render(message_values(&[msg("user", "go"), a, empty]), None, &vars)
            .unwrap();
        assert!(
            p.contains(
                "<function=bash>\n<parameter=_raw>\nls -la {oops\n</parameter>\n</function>"
            ),
            "{p}"
        );
        assert!(p.contains("<function=status>\n</function>"), "{p}");
    }

    /// Qwen3-Coder's template exercises `.keys()`, `reject("in", …)` and `|items` — the Python
    /// methods come from minijinja-contrib's pycompat.
    #[test]
    fn qwen3_coder_template_renders_tools_and_history() {
        let t = ChatTemplate::new(QWEN3_CODER.into(), None).unwrap();
        assert_eq!(t.tool_format(), ToolFormat::QwenXml);
        let vars = TemplateVars {
            add_generation_prompt: true,
            ..Default::default()
        };
        let p = t
            .render(message_values(&agent_history()), Some(tools_value()), &vars)
            .unwrap();
        assert!(
            p.contains("<function>\n<name>read</name>\n<description>Read a file</description>"),
            "{p}"
        );
        assert!(
            p.contains("<parameter>\n<name>path</name>\n<type>string</type>\n</parameter>"),
            "{p}"
        );
        assert!(p.contains("<tool_call>\n<function=read>\n<parameter=path>\ncalc.py\n</parameter>\n</function>\n</tool_call>"), "{p}");
        assert!(
            p.contains(
                "<|im_start|>user\n<tool_response>\nEdited calc.py\n</tool_response>\n<|im_end|>\n"
            ),
            "{p}"
        );
        assert!(p.ends_with("<|im_start|>assistant\n"), "{p}");
        assert_eq!(p, QWEN3_CODER_HF);
    }

    /// `| string` and `{{ x }}` are Python's `str()`. Qwen3-Coder's template prints every
    /// non-string schema field (`type` lists, `anyOf`, float/bool/None defaults) and every
    /// non-string replayed argument (OpenCode's `todowrite.todos`) with `| string`. The reference
    /// is HF-configured Jinja2 3.1.6 again (`testdata/hf_render_pyrepr.py` over the tools and
    /// history files below, arguments passed as dicts as vLLM does). Before the `string` filter
    /// and formatter overrides minijinja printed `["string", "null"]`, `0.00001`, `true`, `none`
    /// — shapes the model never saw in training; `qwen3_coder_template_renders_tools_and_history`
    /// passed only because its conversation has no list, map, float or bool anywhere.
    #[test]
    fn qwen3_coder_python_str_matches_hf() {
        const TOOLS: &str = include_str!("../testdata/pyrepr_tools.json");
        const HISTORY: &str = include_str!("../testdata/pyrepr_history.json");
        const HF: &str = include_str!("../testdata/qwen3_coder_python_str.hf.txt");
        // The reference itself carries the Python forms (the instrument is not blind to them).
        for form in [
            "<type>['string', 'null']</type>",
            "<anyOf>[{'type': 'string'}, {'type': 'null'}]</anyOf>",
            "<default>1e-05</default>",
            "<maximum>1e+16</maximum>",
            "<default>True</default>",
            "<default>None</default>",
            "[{'content': 'fix add()', 'status': 'pending', 'n': 2.5}]",
        ] {
            assert!(HF.contains(form), "{form}");
        }
        let t = ChatTemplate::new(QWEN3_CODER.into(), None).unwrap();
        let history: Vec<ChatMessage> = serde_json::from_str(HISTORY).unwrap();
        let vars = TemplateVars {
            add_generation_prompt: true,
            ..Default::default()
        };
        let p = t
            .render(message_values(&history), parse_ordered(TOOLS), &vars)
            .unwrap();
        assert_eq!(p, HF);
    }

    #[test]
    fn py_str_is_python_str() {
        let v = parse_ordered(
            r#"{"s": "it's", "d": "say \"hi\"", "both": "'\"", "ctl": "a\u0001\u00a0é\u2028", "l": [1, 2.5, true, null, "x"], "e": {}, "f": 1e16}"#,
        )
        .unwrap();
        assert_eq!(
            py_str(&v),
            r#"{'s': "it's", 'd': 'say "hi"', 'both': '\'"', 'ctl': 'a\x01\xa0é\u2028', 'l': [1, 2.5, True, None, 'x'], 'e': {}, 'f': 1e+16}"#
        );
        assert_eq!(py_str(&Value::from("plain")), "plain");
        // Through a template: the filter and `{{ }}` agree with Jinja2.
        let t = ChatTemplate::new(
            "{{ x | string }}|{{ x }}|{{ b }}|{{ n }}|{{ f }}|{{ u | string }}|{{ s }}".into(),
            None,
        )
        .unwrap();
        let ctx = Value::from_iter([
            ("x", parse_ordered(r#"["a", 1]"#).unwrap()),
            ("b", Value::from(false)),
            ("n", Value::from(())),
            ("f", Value::from(0.1)),
            ("s", Value::from("<s>")),
        ]);
        let out = t
            .env
            .get_template(TEMPLATE_NAME)
            .unwrap()
            .render(ctx)
            .unwrap();
        assert_eq!(out, "['a', 1]|['a', 1]|False|None|0.1||<s>");
    }

    /// The Hermes shape (Qwen2.5/Qwen3): the template's tool sections as the Qwen2.5-Instruct
    /// release ships them, REPRODUCED for this test (no Hermes-format GGUF is on this disk; the
    /// reference render below is still HF-configured Jinja2's, not hand-written). The model's
    /// own calls replay as `{"name": …, "arguments": {…}}` with Python spacing, argument order kept.
    #[test]
    fn hermes_template_renders_tools_and_history() {
        const HERMES: &str = include_str!("../testdata/hermes_qwen25_style_chat_template.jinja");
        const HERMES_HF: &str = include_str!("../testdata/hermes_three_turn.hf.txt");
        let t = ChatTemplate::new(HERMES.into(), None).unwrap();
        assert_eq!(t.tool_format(), ToolFormat::HermesJson);
        let vars = TemplateVars {
            add_generation_prompt: true,
            ..Default::default()
        };
        let p = t
            .render(message_values(&agent_history()), Some(tools_value()), &vars)
            .unwrap();
        assert!(p.contains("<tool_call>\n{\"name\": \"edit\", \"arguments\": {\"path\": \"calc.py\", \"oldText\": \"return a - b\", \"newText\": \"return a + b\"}}\n</tool_call>"), "{p}");
        assert_eq!(p, HERMES_HF);
    }

    #[test]
    fn tojson_is_python_json_dumps() {
        let v = parse_ordered(
            r#"{"b": 1, "a": [true, null, 2.5, 1e-5, 1e16, 3.0], "s": "<tag> & 'q' \"d\" \\ \n\t\u0001 é 😀", "e": {}, "l": []}"#,
        )
        .unwrap();
        assert_eq!(
            py_json_dumps(&v),
            r#"{"b": 1, "a": [true, null, 2.5, 1e-05, 1e+16, 3.0], "s": "<tag> & 'q' \"d\" \\ \n\t\u0001 é 😀", "e": {}, "l": []}"#
        );
        // Through the filter, with HF's keyword options.
        let t = ChatTemplate::new(
            "{{ x | tojson }}|{{ x | tojson(indent=2) }}|{{ x | tojson(ensure_ascii=true) }}|{{ x | tojson(sort_keys=true) }}".into(),
            None,
        )
        .unwrap();
        let x = parse_ordered(r#"{"z": "é", "a": [1, 2]}"#).unwrap();
        let out = t
            .env
            .get_template(TEMPLATE_NAME)
            .unwrap()
            .render(Value::from_iter([("x", x)]))
            .unwrap();
        assert_eq!(
            out,
            "{\"z\": \"é\", \"a\": [1, 2]}|{\n  \"z\": \"é\",\n  \"a\": [\n    1,\n    2\n  ]\n}|{\"z\": \"\\u00e9\", \"a\": [1, 2]}|{\"a\": [1, 2], \"z\": \"é\"}"
        );
    }

    #[test]
    fn raise_exception_fails_the_render() {
        let t = ChatTemplate::new("{{ raise_exception('nope: ' ~ 1) }}".into(), None).unwrap();
        let err = t.render(
            Value::from_iter(Vec::<Value>::new()),
            None,
            &TemplateVars::default(),
        );
        assert!(err.unwrap_err().contains("nope: 1"));
    }

    #[test]
    fn arguments_value_decisions() {
        assert_eq!(py_json_dumps(&arguments_value("")), "{}");
        assert_eq!(
            py_json_dumps(&arguments_value(r#"{"b":1,"a":2}"#)),
            r#"{"b": 1, "a": 2}"#
        );
        assert_eq!(
            py_json_dumps(&arguments_value(r#""{\"a\":1}""#)),
            r#"{"a": 1}"#
        );
        assert_eq!(py_json_dumps(&arguments_value("[1]")), r#"{"_raw": "[1]"}"#);
        assert_eq!(
            py_json_dumps(&arguments_value("{oops")),
            r#"{"_raw": "{oops"}"#
        );
        assert_eq!(
            to_compact_json(&arguments_value(r#"{ "b" : 1, "a" : 2 }"#)),
            r#"{"b":1,"a":2}"#
        );
    }

    #[test]
    fn strftime_formats_a_known_instant() {
        // 2024-07-26 13:05:09 UTC was a Friday, day 208 of a leap year.
        let t = 1_722_000_000 - 1_722_000_000 % 86_400 + 13 * 3600 + 5 * 60 + 9;
        assert_eq!(
            strftime_utc("%d %b %Y|%A %B %e|%H:%M:%S %I%p|%j|%y%m|%%|%Q", t),
            "26 Jul 2024|Friday July 26|13:05:09 01PM|208|2407|%|%Q"
        );
    }

    #[test]
    fn developer_becomes_system_for_templates() {
        let v = message_values(&[msg("developer", "rules")]);
        let first = v.get_item(&Value::from(0)).unwrap();
        assert_eq!(first.get_attr("role").unwrap().as_str(), Some("system"));
    }
}
