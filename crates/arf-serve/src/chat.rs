//! Prompt construction: BOS / stop-token resolution (copied from the CLI's
//! name-based approach) and a per-model chat template.

use arf_core::Tokenizer;

use crate::http::{ChatMessage, Tool, ToolCall};

/// Build the tool-use system instruction injected ahead of the conversation when a
/// request carries `tools`. gemma-4 (this QAT GGUF) doesn't reliably use the
/// `<|tool_call>` special tokens, but it RELIABLY emits a clean JSON object when asked —
/// so we steer it to a fenced ```json tool-call block we can parse robustly.
pub fn tool_system_prompt(tools: &[Tool]) -> String {
    let mut s = String::from(
        "You can call functions to get information or take actions. When a function is \
needed, respond with ONLY a JSON object in a fenced code block. The JSON MUST include both \
the function name and its arguments, in EXACTLY this shape (do not omit \"function\"):\n\
```json\n{\"function\": \"the_function_name\", \"arguments\": {\"arg\": \"value\"}}\n```\n\
For example, to compute a number you would output:\n\
```json\n{\"function\": \"run_code\", \"arguments\": {\"code\": \"2450 * 0.17\"}}\n```\n\
Call one function at a time. NEVER describe that you are about to call a function or say \
\"I will check\" — instead, immediately emit the JSON call. After you receive a result, if \
more information is still needed, emit the next JSON call; otherwise give the final answer. \
If no function is needed at all, just answer normally.\n\nAvailable functions:\n",
    );
    for t in tools {
        let params = t
            .function
            .parameters
            .as_ref()
            .map(|p| p.to_string())
            .unwrap_or_else(|| "{}".into());
        let desc = t.function.description.clone().unwrap_or_default();
        s.push_str(&format!(
            "- {}: {}\n  parameters: {}\n",
            t.function.name, desc, params
        ));
    }
    s
}

/// A call rendered exactly as `tool_system_prompt` asks the model to write it, so a replayed
/// assistant turn reads like something the model said. Shared by `/v1/messages` (`tool_use`
/// history) and the OpenAI fallback path (`tool_calls` history).
pub fn render_fenced_call(name: &str, input: &serde_json::Value) -> String {
    let args = if input.is_null() {
        serde_json::json!({})
    } else {
        input.clone()
    };
    format!(
        "```json\n{}\n```",
        serde_json::json!({"function": name, "arguments": args})
    )
}

/// The conversation for the fenced-JSON fallback path, with its tool history made readable.
///
/// Until 2026-09-26 the OpenAI path deserialized only `role` and `content`, so an assistant turn
/// that called a tool arrived EMPTY and a `tool` result arrived as an off-template `tool` turn:
/// the model never saw its own earlier calls, and OpenCode was observed repeating an edit it had
/// already made. Here:
/// - the tool prompt (when tools are offered) and every LEADING system/developer message become
///   ONE system message — the tool prompt used to take index 0 and push the client's system
///   prompt into a second system turn, which most hand-written templates render literally;
/// - an assistant's `tool_calls` are replayed after its prose as the fenced JSON the model writes;
/// - `tool` results become a user turn, `Function `name` returned:` (the name from the message,
///   else from the call its `tool_call_id` answers), consecutive results merged into one turn —
///   the same wording `/v1/messages` uses for `tool_result`.
pub fn legacy_tool_messages(messages: &[ChatMessage], tools: Option<&[Tool]>) -> Vec<ChatMessage> {
    let mut out: Vec<ChatMessage> = Vec::with_capacity(messages.len() + 1);
    let lead = messages
        .iter()
        .take_while(|m| m.role == "system" || m.role == "developer")
        .count();
    let mut system: Vec<String> = Vec::new();
    if let Some(t) = tools.filter(|t| !t.is_empty()) {
        system.push(tool_system_prompt(t));
    }
    system.extend(
        messages[..lead]
            .iter()
            .map(|m| m.content.trim().to_string())
            .filter(|s| !s.is_empty()),
    );
    if !system.is_empty() {
        out.push(ChatMessage::text("system", system.join("\n\n")));
    }
    let mut names: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
    let mut prev_was_tool = false;
    for m in &messages[lead..] {
        let is_tool = m.role == "tool";
        match m.role.as_str() {
            "assistant" if !m.tool_calls.is_empty() => {
                let mut parts: Vec<String> = Vec::with_capacity(m.tool_calls.len() + 1);
                if !m.content.trim().is_empty() {
                    parts.push(m.content.trim().to_string());
                }
                for c in &m.tool_calls {
                    names.insert(c.id.as_str(), c.function.name.as_str());
                    // Replayed as written: an object when the string parses, `{}` when there are
                    // no arguments, else the client's string verbatim.
                    let raw = c.function.arguments.trim();
                    let args = if raw.is_empty() {
                        serde_json::json!({})
                    } else {
                        serde_json::from_str::<serde_json::Value>(raw)
                            .unwrap_or_else(|_| serde_json::Value::String(raw.to_string()))
                    };
                    parts.push(render_fenced_call(&c.function.name, &args));
                }
                out.push(ChatMessage {
                    role: "assistant".into(),
                    content: parts.join("\n"),
                    images: m.images.clone(),
                    videos: m.videos.clone(),
                    ..Default::default()
                });
            }
            "tool" => {
                let name = m
                    .name
                    .as_deref()
                    .or_else(|| {
                        m.tool_call_id
                            .as_deref()
                            .and_then(|id| names.get(id).copied())
                    })
                    .unwrap_or("function");
                let text = format!("Function `{name}` returned:\n{}", m.content);
                // Consecutive results (parallel calls) share one user turn.
                match out.last_mut() {
                    Some(prev) if prev_was_tool => {
                        prev.content.push_str("\n\n");
                        prev.content.push_str(&text);
                    }
                    _ => out.push(ChatMessage::text("user", text)),
                }
            }
            "developer" => {
                let mut s = m.clone();
                s.role = "system".into();
                out.push(s);
            }
            _ => out.push(m.clone()),
        }
        prev_was_tool = is_tool;
    }
    out
}

/// Strip gemma-4 thought-channel content from a reply.
///
/// Transcribed from the `strip_thinking` macro in the GGUF's own `tokenizer.chat_template`:
/// split on `<channel|>`, and in any part that contains `<|channel>` keep only what precedes
/// it. That removes a whole `<|channel>thought\n…<channel|>` block wherever it appears, which
/// the previous version could not do — it only stripped a bare leading `thought`, so a real
/// block arrived intact and the brackets reached the user.
///
/// The leading-`thought` strip is kept for the QAT file's habit of emitting that token bare,
/// but is now guarded: it only fires when `thought` is a whole word, so ordinary prose like
/// "thoughtful design" survives. The unguarded `strip_prefix` silently ate four characters
/// off any reply that happened to start that way.
pub fn strip_thought(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for part in text.split("<channel|>") {
        match part.find("<|channel>") {
            Some(i) => out.push_str(&part[..i]),
            None => out.push_str(part),
        }
    }
    let t = out.trim();
    let t = match t.strip_prefix("thought") {
        Some(r) if r.is_empty() || r.starts_with(char::is_whitespace) => r,
        _ => t,
    };
    t.trim().to_string()
}

/// Split a Muse Glimmer reply into `(reasoning, answer)`.
///
/// This model answers on CHANNELS, chosen by the recipient it names in its own header. The
/// raw reply for a plain question looks like
///
/// ```text
///  to=self<|message|> …chain of thought… <|start|>assistant to=user<|message|> …answer…
/// ```
///
/// `to=self` is the model thinking out loud; `to=user` is the reply. Both are real, correct
/// output — llama shows only the second, which is why its answers look clean and a raw dump
/// looks like the model is rambling past the question.
///
/// Returns the LAST `to=user` segment as the answer (a long chain of thought can re-enter
/// `self` several times) and everything before it as reasoning. When no channel header is
/// present the text is already a plain answer, so it passes through with no reasoning.
pub fn split_muse_channels(reply: &str) -> (Option<String>, String) {
    const USER_CH: &str = "to=user<|message|>";
    match reply.rfind(USER_CH) {
        Some(i) => {
            let answer = reply[i + USER_CH.len()..].trim().to_string();
            // Trim the reasoning's own `to=self<|message|>` header and the assistant
            // re-header that introduces the final channel.
            let head = reply[..i]
                .trim()
                .trim_end_matches("<|start|>assistant")
                .trim();
            let reasoning = head
                .strip_prefix("to=self<|message|>")
                .unwrap_or(head)
                .trim()
                .to_string();
            ((!reasoning.is_empty()).then_some(reasoning), answer)
        }
        // No user channel: either a plain reply, or the model was cut off mid-thought
        // (n_predict limit). Strip a leading `to=self` header so the text is readable
        // either way, and report it as the answer — dropping it would show nothing.
        None => {
            let t = reply.trim();
            let t = t.strip_prefix("to=self<|message|>").unwrap_or(t);
            (None, t.trim().to_string())
        }
    }
}

/// Parse tool calls out of the model's reply with no tool-name check — see [`parse_tool_calls`].
pub fn parse_tool_call(reply: &str) -> Option<(Vec<ToolCall>, String)> {
    parse_tool_calls(reply, &[])
}

/// Parse EVERY tool call out of a fenced-JSON reply. Returns `Some((tool_calls, leading_text))`
/// when at least one `{"function"/"name": ..., "arguments"/"parameters": ...}` object is found
/// (in a ```json fence or bare), else `None`. The leading text (prose before the first call) is
/// returned for display. Each call gets a fresh, unique id.
///
/// 2026-09-26 hardening. The first version took the FIRST `{` in the reply and
/// stopped there, so:
///   * prose with a brace before the call (`f"Hello, {name}!"` in an explanation) hid the call —
///     the reply came back as plain text and the agent ended its run;
///   * a second call in the same reply was silently dropped;
///   * every call to `read` had the id `call_read`, so two results could not be told apart;
///   * a string with a RAW newline (the model writing code into an `edit` argument) is invalid
///     JSON, and the whole call fell back to text.
///
/// Now every balanced object is scanned in order: a call is taken; any other VALID JSON object is
/// skipped whole (its braces are data); an invalid one advances one character (prose). Raw
/// control characters inside strings are escaped and the parse retried. With `tool_names`
/// non-empty, the name is checked (see `json_call_from_object`, crate-private: no intra-doc link
/// from this public fn, which `cargo doc -D warnings` refuses) — an answer that is itself
/// JSON data (`{"name": "Alice", "age": 3}`) is not a call to a function named "Alice".
pub fn parse_tool_calls(reply: &str, tool_names: &[&str]) -> Option<(Vec<ToolCall>, String)> {
    // 1) JSON tool calls: {"function"/"name": ..., "arguments": ...} objects (fenced or bare).
    let mut calls = Vec::new();
    let mut first = None;
    let mut i = 0;
    while let Some(off) = reply[i..].find('{') {
        let start = i + off;
        let Some(end) = balanced_object_end(reply, start) else {
            i = start + 1;
            continue;
        };
        let obj = &reply[start..=end];
        if let Some(c) = json_call_from_object(obj, tool_names) {
            first.get_or_insert(start);
            calls.push(c);
            i = end + 1;
        } else if serde_json::from_str::<serde_json::Value>(obj).is_ok() {
            i = end + 1;
        } else {
            i = start + 1;
        }
    }
    if let Some(start) = first {
        let lead = reply[..start]
            .trim()
            .trim_end_matches("```json")
            .trim_end_matches("```")
            .trim()
            .trim_start_matches("thought")
            .trim()
            .to_string();
        return Some((calls, lead));
    }
    // 2) gemma shorthand: `call:NAME{...}` (it sometimes ignores the JSON instruction).
    //    e.g. `call:get_gpu_profile{}.`  or  `call:web_search{"query":"M4 Max"}`.
    let mut lead_end = None;
    let mut from = 0;
    while let Some(off) = reply[from..].find("call:") {
        let idx = from + off;
        from = idx + 5;
        let after = &reply[idx + 5..];
        let Some(brace) = after.find('{') else { break };
        let name = after[..brace].trim().trim_end_matches(['(', ' ']);
        if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            continue;
        }
        // The braces are its arguments, so an unknown name is still a call (see
        // `json_call_from_object`); a known one in the wrong case gets the declared spelling.
        let name = declared_name(name, tool_names).unwrap_or(name).to_string();
        // take the balanced brace span; coerce loose `{k:"v"}` to JSON if needed.
        let Some(end) = balanced_object_end(after, brace) else {
            break;
        };
        let braces = &after[brace..=end];
        let arguments = crate::jinja_chat::parse_ordered(braces)
            .map(|v| crate::jinja_chat::to_compact_json(&v))
            .unwrap_or_else(|| coerce_loose_args(braces));
        lead_end.get_or_insert(idx);
        calls.push(crate::tool_parse::make_call(&name, arguments));
        from = idx + 5 + end + 1;
    }
    let idx = lead_end?;
    let lead = reply[..idx]
        .trim()
        .trim_start_matches("thought")
        .trim()
        .to_string();
    Some((calls, lead))
}

/// The offered tool `name` matches, compared ignoring ASCII case, in its DECLARED spelling.
fn declared_name<'a>(name: &str, tool_names: &[&'a str]) -> Option<&'a str> {
    tool_names
        .iter()
        .find(|t| t.eq_ignore_ascii_case(name))
        .copied()
}

/// One JSON object as a tool call, or `None` if it is not one. Accepts `function` or `name` for
/// the function, `arguments` or `parameters` for its arguments (an object, or a string passed
/// through as the model wrote it), and the OpenAI echo `{"type":"function","function":{"name":…,
/// "arguments":…}}`. Arguments keep the model's key order.
///
/// With `tool_names` (the offered tools) non-empty, the name is checked ignoring case and
/// returned in the declared spelling (`Read` → `read`). An UNKNOWN name is still a call when the
/// object carries `arguments` / `parameters`: that is a misnamed call, and the client can repair
/// it — OpenCode's `experimental_repairToolCall` lowercases it or routes it to its `invalid` tool
/// so the model retries. Only an unknown name WITHOUT arguments is data, not a call. (The first
/// name check, 2026-09-26, was exact and case-sensitive: `{"function": "Read", …}` from a Gemma
/// or Llama model came back as plain text with finish_reason `stop`, ending the agent's turn.)
pub(crate) fn json_call_from_object(obj: &str, tool_names: &[&str]) -> Option<ToolCall> {
    use crate::jinja_chat::{parse_ordered, to_compact_json};
    use minijinja::value::{Value, ValueKind};
    let v = parse_ordered(obj).or_else(|| parse_ordered(&escape_raw_controls(obj)))?;
    if v.kind() != ValueKind::Map {
        return None;
    }
    let get = |v: &Value, k: &str| v.get_attr(k).unwrap_or(Value::UNDEFINED);
    let first_defined = |a: Value, b: Value| if a.is_undefined() { b } else { a };
    let f = get(&v, "function");
    let (name, args) = if f.kind() == ValueKind::Map {
        (
            get(&f, "name"),
            first_defined(get(&f, "arguments"), get(&f, "parameters")),
        )
    } else {
        (
            if f.is_undefined() { get(&v, "name") } else { f },
            first_defined(get(&v, "arguments"), get(&v, "parameters")),
        )
    };
    let name = name.as_str()?.trim();
    if name.is_empty() {
        return None;
    }
    let name = match declared_name(name, tool_names) {
        Some(declared) => declared,
        None if tool_names.is_empty() || !args.is_undefined() => name,
        None => return None,
    };
    let arguments = match args.kind() {
        ValueKind::Undefined | ValueKind::None => "{}".to_string(),
        ValueKind::String => args.as_str().unwrap_or("{}").to_string(),
        _ => to_compact_json(&args),
    };
    Some(crate::tool_parse::make_call(name, arguments))
}

/// Escape raw control characters that sit INSIDE JSON strings (`\n` → `\\n`, …). A model writing
/// code into an argument often emits a real newline there, which strict JSON rejects.
fn escape_raw_controls(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 16);
    let mut in_str = false;
    let mut esc = false;
    for c in s.chars() {
        if in_str {
            if esc {
                esc = false;
            } else if c == '\\' {
                esc = true;
            } else if c == '"' {
                in_str = false;
            } else if (c as u32) < 0x20 {
                match c {
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    c => out.push_str(&format!("\\u{:04x}", c as u32)),
                }
                continue;
            }
        } else if c == '"' {
            in_str = true;
        }
        out.push(c);
    }
    out
}

/// Coerce gemma's loose `{key:"val", n:5}` (unquoted keys) into valid JSON. Best-effort:
/// quote bare identifier keys; leave already-quoted strings + numbers alone. Falls back to
/// `{}` if it can't produce parseable JSON.
///
/// **Corrected 2026-09-26 (review), twice over.** (1) It walked `s.as_bytes()` and pushed each
/// byte `as char` — a Latin-1 decode — so `{content:"héllo ✓"}` came out as
/// `{"content":"hÃ©llo â\u{9c}\u{93}"}`: valid JSON, an object, and mojibake in the file an
/// agent then wrote. It now copies whole characters (the decisions are still made on ASCII
/// bytes, which never occur inside a multi-byte UTF-8 sequence). (2) It re-serialized through
/// `serde_json::Value`, a SORTED map (`preserve_order` is off), so the model's
/// `{path, oldText, newText}` went out as `{newText, oldText, path}` on this path alone. It now
/// parses order-preserving, as the quoted form and `tool_parse`'s Python-literal fallback do.
fn coerce_loose_args(s: &str) -> String {
    // quote bare keys: `{city:` / `, n:` → `{"city":` / `,"n":`
    let mut out = String::with_capacity(s.len() + 8);
    let bytes = s.as_bytes();
    // `i` stays on a char boundary: it moves by one ASCII byte or by one whole char.
    let whole_char = |i: usize| s[i..].chars().next().expect("i < len, on a char boundary");
    // ASCII whitespace only: a byte `as char` above 0x7F is half a character, not a space.
    let space = |b: u8| b.is_ascii() && (b as char).is_whitespace();
    let mut i = 0;
    let mut in_str = false;
    while i < bytes.len() {
        let c = whole_char(i);
        if in_str {
            out.push(c);
            if c == '"' && (i == 0 || bytes[i - 1] != b'\\') {
                in_str = false;
            }
            i += c.len_utf8();
            continue;
        }
        if c == '"' {
            in_str = true;
            out.push(c);
            i += 1;
        } else if (c == '{' || c == ',') && {
            // peek: next non-space is a bare identifier followed (eventually) by ':'
            let mut j = i + 1;
            while j < bytes.len() && space(bytes[j]) {
                j += 1;
            }
            j < bytes.len() && (bytes[j].is_ascii_alphabetic() || bytes[j] == b'_')
        } {
            out.push(c);
            i += 1;
            while i < bytes.len() && space(bytes[i]) {
                out.push(bytes[i] as char);
                i += 1;
            }
            let key_start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            out.push('"');
            out.push_str(&s[key_start..i]);
            out.push('"');
        } else {
            out.push(c);
            i += c.len_utf8();
        }
    }
    crate::jinja_chat::parse_ordered(&out)
        .map(|v| crate::jinja_chat::to_compact_json(&v))
        .unwrap_or_else(|| "{}".into())
}

/// The byte index of the `}` that closes the object opening at `text[start]` (a `{`), honouring
/// JSON strings and escapes, or `None` if it never closes. (Was `extract_json_object`, which
/// always started at the FIRST `{` of the text — see `parse_tool_calls` for why that failed.)
pub(crate) fn balanced_object_end(text: &str, start: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    if bytes.get(start) != Some(&b'{') {
        return None;
    }
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        let c = b as char;
        if in_str {
            if esc {
                esc = false;
            } else if c == '\\' {
                esc = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// The model's begin-of-sequence id, resolved by NAME (no per-model hardcoding):
/// SentencePiece `<bos>` (Gemma/Llama) or GPT-style `<|begin_of_text|>`. `None`
/// if the tokenizer has no BOS. Mirrors the CLI's `bos_token`.
pub fn bos_token(tok: &Tokenizer) -> Option<u32> {
    ["<bos>", "<|begin_of_text|>"]
        .iter()
        .find_map(|t| tok.token_to_id(t))
}

/// End-of-turn / EOS ids to stop decode on, resolved by NAME across model families:
/// Gemma `<end_of_turn>`, SentencePiece `<eos>`, GPT-style `<|endoftext|>`,
/// Qwen/ChatML `<|im_end|>`, and Llama-3 `<|eot_id|>` / `<|end_of_text|>`.
/// Only those present in the vocab are kept. Mirrors the CLI's `chat_stop_tokens`.
pub fn chat_stop_tokens(tok: &Tokenizer) -> Vec<u32> {
    [
        "<end_of_turn>",
        "<turn|>", // gemma-4 channel format: the turn terminator
        "<eos>",
        "<|endoftext|>",
        "<|im_end|>",
        "<|eot_id|>",
        "<|end_of_text|>",
        // Muse Glimmer's TURN terminator. `<|eom|>` is deliberately absent: it ends a MESSAGE
        // inside a turn, and this model emits it to close its `to=self` chain of thought before
        // re-heading into `to=user` with the answer. Treating it as a stop truncates the reply
        // to the reasoning alone. Non-Muse vocabs lack this name and drop out below.
        "<|eot|>",
    ]
    .iter()
    .filter_map(|t| tok.token_to_id(t))
    .collect()
}

/// Gemma chat template (no tokenizer required — pure string logic).
///
/// Each `user` or `system` turn becomes `<start_of_turn>user\n{content}<end_of_turn>\n`.
/// Gemma folds `system` into the user turn (matches ollama's authoritative template).
/// Each `assistant` turn becomes `<start_of_turn>model\n{content}<end_of_turn>\n`.
/// After all messages appends `<start_of_turn>model\n` to cue the reply.
pub fn gemma_template(messages: &[ChatMessage]) -> String {
    let mut p = String::new();
    for m in messages {
        if m.role == "assistant" {
            p.push_str("<start_of_turn>model\n");
            p.push_str(&m.content);
            p.push_str("<end_of_turn>\n");
        } else {
            // "user" and "system" both map to the user turn.
            p.push_str("<start_of_turn>user\n");
            p.push_str(&m.content);
            p.push_str("<end_of_turn>\n");
        }
    }
    p.push_str("<start_of_turn>model\n");
    p
}

/// Gemma-4 *channel* chat template (no tokenizer required — pure string logic).
///
/// gemma-4 dropped `<start_of_turn>`/`<end_of_turn>` for a turn/channel format. Transcribed
/// from the GGUF's OWN `tokenizer.chat_template` Jinja rather than inferred from the vocab —
/// the three places this used to differ from it are all visible in the model's output:
///
///   * A turn header is `<|turn>{role}\n`, NOT `<|turn>{role}<channel|>`. `<channel|>` is the
///     THOUGHT-channel terminator, not a role separator, so using it here handed the model a
///     half-open channel where a turn header belonged.
///   * A turn ends `<turn|>\n` — the trailing newline is part of the format.
///   * The generation cue is `<|turn>model\n` plus, when thinking is off (the template's own
///     default), an immediately closed thought channel `<|channel>thought\n<channel|>`.
///
/// That last one is the load-bearing fix. The cue's empty thought block is what tells the
/// model its thinking turn is already spent; without it the model OPENS ONE ITSELF, which is
/// why replies arrived as `…Paris.\n<|channel>thought\n<channel|>…` with the answer echoed
/// on both sides of a marker the user was never meant to see.
///
/// `assistant` maps to `model`; every other role passes through (gemma-4 has a real `system`
/// role, unlike gemma-2/3 which folded it into the user turn). A leading system/developer
/// message becomes the dedicated opening system turn and is dropped from the main loop.
/// `<turn|>` is the stop token. BOS is added by `encode_prompt`, matching the template's own
/// `{{- bos_token -}}`.
pub fn gemma4_template(messages: &[ChatMessage]) -> String {
    let mut p = String::new();
    let mut rest = messages;
    if let Some(first) = messages.first() {
        if first.role == "system" || first.role == "developer" {
            p.push_str("<|turn>system\n");
            p.push_str(first.content.trim());
            p.push_str("<turn|>\n");
            rest = &messages[1..];
        }
    }
    for m in rest {
        let role = match m.role.as_str() {
            "assistant" => "model",
            // A `tool` result is fed back inside a user turn: the template's own
            // `<|tool_response>` block is only reachable when `tools` were declared to it,
            // which this plain-chat path does not do.
            "tool" => "user",
            other => other,
        };
        p.push_str("<|turn>");
        p.push_str(role);
        p.push('\n');
        if m.role == "tool" {
            p.push_str("Function result:\n");
        }
        p.push_str(&m.content);
        p.push_str("<turn|>\n");
    }
    p.push_str("<|turn>model\n<|channel>thought\n<channel|>");
    p
}

/// Qwen3.8's OWN chat template with thinking on (2026-09-23), transcribed from the model's
/// `chat_template.jinja`. Two things the plain ChatML rendering below does not do:
///   1. a SYSTEM message carrying the reasoning-effort instruction (`xhigh` by default; `medium`
///      adds nothing; `low` asks for brevity), merged ahead of any user-provided system text;
///   2. the generation cue ends with a PRE-OPENED `<think>\n`.
/// WHY IT MATTERS (measured 2026-09-23): served off-template the model thinks in a
/// different register ("The user wants... Let me think...") from its on-template one ("We need
/// to answer user's request... Need produce final..."), and the DFlash drafter was trained on
/// the latter — acceptance 2.09 -> 3.09 drafts a window on the same prompt, above another engine's
/// 2.70, with no engine change. It is also simply the model's intended behaviour.
pub const QWEN3_EFFORT_XHIGH: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
pub const QWEN3_EFFORT_LOW: &str = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";

/// On when the served architecture is Qwen3.8 (`arf-serve` sets `ARF_QWEN_THINK_TEMPLATE`
/// from `--arch`); `ARF_NO_QWEN_THINK_TEMPLATE=1` restores the plain ChatML rendering (A/B).
pub fn qwen_think_template_active() -> bool {
    std::env::var_os("ARF_QWEN_THINK_TEMPLATE").is_some()
        && std::env::var_os("ARF_NO_QWEN_THINK_TEMPLATE").is_none()
}

pub fn qwen3_think_template(messages: &[ChatMessage], effort: Option<&str>) -> String {
    let effort = effort
        .map(str::to_string)
        .or_else(|| std::env::var("ARF_REASONING_EFFORT").ok())
        .unwrap_or_else(|| "xhigh".into());
    // "none" = Qwen's NON-THINKING rendering (its template's enable_thinking=false): no effort
    // instruction and a PRE-CLOSED think block. Used for grammar-constrained output (JSON mode),
    // whose constraint starts at the first generated token and must land in the answer.
    let thinking = effort != "none";
    let instr = match effort.as_str() {
        "low" => Some(QWEN3_EFFORT_LOW),
        "medium" | "none" => None,
        _ => Some(QWEN3_EFFORT_XHIGH),
    };
    let mut p = String::new();
    let first_sys = messages.first().filter(|m| m.role == "system");
    match (instr, first_sys) {
        (Some(i), Some(m)) if !m.content.trim().is_empty() => {
            p.push_str("<|im_start|>system\n");
            p.push_str(i);
            p.push_str("\n\n");
            p.push_str(m.content.trim());
            p.push_str("<|im_end|>\n");
        }
        (Some(i), _) => {
            p.push_str("<|im_start|>system\n");
            p.push_str(i);
            p.push_str("<|im_end|>\n");
        }
        (None, Some(m)) if !m.content.trim().is_empty() => {
            p.push_str("<|im_start|>system\n");
            p.push_str(m.content.trim());
            p.push_str("<|im_end|>\n");
        }
        (None, _) => {}
    }
    for (idx, m) in messages.iter().enumerate() {
        if idx == 0 && m.role == "system" {
            continue;
        }
        p.push_str("<|im_start|>");
        p.push_str(&m.role);
        p.push('\n');
        p.push_str(m.content.trim());
        p.push_str("<|im_end|>\n");
    }
    p.push_str(if thinking {
        "<|im_start|>assistant\n<think>\n"
    } else {
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    });
    p
}

/// True when a rendered prompt ENDS inside an open `<think>` — its cue pre-opened the reasoning
/// block, so the reply starts as reasoning. Read off the prompt itself rather than off the model
/// family: any template that does this (Qwen3.8's thinking cue does; with `enable_thinking`
/// false it pre-CLOSES the block, which this correctly reads as not thinking).
pub fn prompt_opens_think(prompt: &str) -> bool {
    prompt.trim_end().ends_with("<think>")
}

/// The reply to a prompt whose cue pre-opened `<think>\n`: everything before `</think>` is the
/// model's reasoning, the rest is the answer. A reply cut off before `</think>` is all reasoning.
pub fn split_pretag_think(reply: &str) -> (Option<String>, String) {
    let t = reply.strip_prefix("<think>").unwrap_or(reply);
    match t.find("</think>") {
        Some(i) => {
            let reasoning = t[..i].trim().to_string();
            let answer = t[i + "</think>".len()..].trim().to_string();
            ((!reasoning.is_empty()).then_some(reasoning), answer)
        }
        None => {
            let r = t.trim().to_string();
            ((!r.is_empty()).then_some(r), String::new())
        }
    }
}

/// ChatML / Qwen chat template (no tokenizer required — pure string logic).
///
/// Each turn becomes `<|im_start|>{role}\n{content}<|im_end|>\n`.
/// After all messages appends `<|im_start|>assistant\n` to cue the reply.
pub fn chatml_template(messages: &[ChatMessage]) -> String {
    let mut p = String::new();
    for m in messages {
        p.push_str("<|im_start|>");
        p.push_str(&m.role);
        p.push('\n');
        p.push_str(&m.content);
        p.push_str("<|im_end|>\n");
    }
    p.push_str("<|im_start|>assistant\n");
    p
}

/// Muse Glimmer chat template (no tokenizer required — pure string logic).
///
/// Each turn is `<|start|>{role}<|message|>{content}<|eot|>`, and the generation cue is
/// `<|start|>assistant` with **no** trailing `<|message|>` — the model emits that itself,
/// because the assistant header is where it selects a recipient (`assistant to=…`). Adding
/// `<|message|>` here would pre-commit that choice and suppress its tool-call channel.
///
/// The GGUF's own Jinja is far larger than this, but the extra machinery is all optional
/// (tool definitions, a reasoning-strength directive, `<|eom|>` for consecutive same-role
/// turns). None of it applies to plain chat, so this renders the same bytes llama does for
/// the same messages, without pulling in a Jinja engine.
pub fn muse_glimmer_template(messages: &[ChatMessage]) -> String {
    let mut p = String::new();
    if !messages.iter().any(|m| m.role == "system") {
        p.push_str(muse_glimmer_system().as_str());
    }
    for m in messages {
        p.push_str("<|start|>");
        p.push_str(&m.role);
        p.push_str("<|message|>");
        p.push_str(&m.content);
        p.push_str("<|eot|>");
    }
    p.push_str("<|start|>assistant");
    p
}

/// The system turn Muse Glimmer's template INJECTS when the caller supplies none. It is not
/// decoration: `# Valid recipients` is what tells the model which channels exist, and without
/// it the first sampled token after `<|start|>assistant` is a bare ` to`, which then repeats
/// forever — the model is trying to address a recipient it was never told about.
///
/// `Reasoning strength: high` is the template's own default (`render_reasoning()`), and the
/// knowledge-cutoff line is the template's literal default too. The current-date line is
/// omitted deliberately: the template only emits it when the caller passes `current_date`,
/// and baking a build-time date into a prompt would make output depend on when the binary was
/// compiled.
pub use arf_core::chat_template::muse_glimmer_system;

/// Render chat messages into a prompt string, dispatching to the correct
/// per-model chat template based on the tokenizer's special tokens:
///
/// Detection order (first match wins):
/// 1. **Llama-3** (`<|start_header_id|>` + `<|eot_id|>` present): header format.
/// 2. **Gemma** (`<start_of_turn>` present): turn-based format with `model` role.
/// 3. **ChatML/Qwen** (`<|im_start|>` + `<|im_end|>` present): `im_start`/`im_end` format.
/// 4. **Generic fallback** (unknown): `role\ncontent\n…assistant\n`.
pub fn apply_chat_template(tok: &Tokenizer, messages: &[ChatMessage]) -> String {
    apply_chat_template_with(tok, messages, None)
}

/// `apply_chat_template` with the request's `reasoning_effort` (Qwen3.8 only; ignored elsewhere).
pub fn apply_chat_template_with(
    tok: &Tokenizer,
    messages: &[ChatMessage],
    effort: Option<&str>,
) -> String {
    // 1. Llama-3 instruct header format when the required special tokens are present.
    if tok.token_to_id("<|start_header_id|>").is_some() && tok.token_to_id("<|eot_id|>").is_some() {
        let mut p = String::new();
        for m in messages {
            p.push_str("<|start_header_id|>");
            p.push_str(&m.role);
            p.push_str("<|end_header_id|>\n\n");
            p.push_str(&m.content);
            p.push_str("<|eot_id|>");
        }
        p.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        return p;
    }

    // 2a. Gemma-4 channel format (newer: <|turn>…<channel|>…<turn|>). Detected by the
    // turn/channel token pairs that REPLACED <start_of_turn> in gemma-4's vocab.
    if tok.token_to_id("<|turn>").is_some() && tok.token_to_id("<channel|>").is_some() {
        return gemma4_template(messages);
    }

    // 2b. Gemma-2/3 turn format (system folded into user; assistant role mapped to "model").
    if tok.token_to_id("<start_of_turn>").is_some() {
        return gemma_template(messages);
    }

    // 3. ChatML / Qwen im_start format — Qwen3.8's own thinking template when it is the model.
    if tok.token_to_id("<|im_start|>").is_some() && tok.token_to_id("<|im_end|>").is_some() {
        if qwen_think_template_active() && tok.token_to_id("<think>").is_some() {
            return qwen3_think_template(messages, effort);
        }
        return chatml_template(messages);
    }

    // 3b. Muse Glimmer start/message format. Keyed on the PAIR (both tokens) so a vocab that
    // merely happens to contain one of them cannot claim this arm.
    if tok.token_to_id("<|start|>").is_some() && tok.token_to_id("<|message|>").is_some() {
        return muse_glimmer_template(messages);
    }

    // 4. Generic fallback (unknown model family) — unchanged behavior.
    let mut prompt = String::new();
    for m in messages {
        prompt.push_str(&m.role);
        prompt.push('\n');
        prompt.push_str(&m.content);
        prompt.push('\n');
    }
    prompt.push_str("assistant\n");
    prompt
}

/// Tokenize a prompt the same way the CLI does: `encode` with `add_special_tokens`
/// so the tokenizer's post-processor can insert template tokens, then prepend the
/// model's BOS if the encoding does not already start with it (some `tokenizer.json`s
/// lack a BOS post-processor, and models like Gemma loop without a leading BOS).
pub fn encode_prompt(tok: &Tokenizer, prompt: &str) -> Result<Vec<u32>, String> {
    let mut ids = tok.encode(prompt, true).map_err(|e| e.to_string())?;
    if let Some(bos) = bos_token(tok) {
        if ids.first() != Some(&bos) {
            ids.insert(0, bos);
        }
    }
    Ok(ids)
}

// ----------------------------------------------------------------------------
// Vision (Gemma-3): image decode + the image-token block + position lookup.
// ----------------------------------------------------------------------------

/// Gemma-3 image marker tokens (present in the GGUF tokenizer as CONTROL tokens). The
/// `<image_soft_token>` placeholder positions are where the projector's soft-tokens splice
/// in; the embedding is overridden by position, so the id is not load-bearing, but the
/// real `<start_of_image>`/`<end_of_image>` brackets ARE tokens the LM was trained with.
pub const SOFT_IMAGE_TOKEN: &str = "<image_soft_token>";
pub const START_OF_IMAGE: &str = "<start_of_image>";
pub const END_OF_IMAGE: &str = "<end_of_image>";

/// The text block for one image, matching the HF Gemma-3 processor's `full_image_sequence`:
/// `\n\n<start_of_image>` + N×`<image_soft_token>` + `<end_of_image>\n\n`. The `\n\n` on both
/// sides is what the processor injects (not the chat template); omitting it measurably hurts
/// grounding. N = projector tokens (256).
pub fn gemma3_image_block(n_tokens: usize) -> String {
    let mut s = String::with_capacity(
        4 + START_OF_IMAGE.len() + n_tokens * SOFT_IMAGE_TOKEN.len() + END_OF_IMAGE.len(),
    );
    s.push_str("\n\n");
    s.push_str(START_OF_IMAGE);
    for _ in 0..n_tokens {
        s.push_str(SOFT_IMAGE_TOKEN);
    }
    s.push_str(END_OF_IMAGE);
    s.push_str("\n\n");
    s
}

/// Decode a `data:image/...;base64,...` URL (or a bare base64 string) into RGB8 bytes
/// `[h, w, 3]` row-major plus `(width, height)`. Errors on a malformed URL or undecodable
/// image. Only the formats compiled into the `image` crate (png/jpeg/webp) are accepted.
pub fn decode_image_data_url(url: &str) -> Result<(Vec<u8>, usize, usize), String> {
    use base64::Engine;
    // Strip an optional `data:<mime>;base64,` prefix; accept a bare base64 body too.
    let b64 = match url.find("base64,") {
        Some(i) => &url[i + "base64,".len()..],
        None if url.starts_with("data:") => {
            return Err("image data URL must be base64-encoded".into());
        }
        None => url, // assume a bare base64 string
    };
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .map_err(|e| format!("image base64 decode: {e}"))?;
    let img = image::load_from_memory(&bytes).map_err(|e| format!("image decode: {e}"))?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    Ok((rgb.into_raw(), w, h))
}

/// The token ids (in `ids`) that equal `<image_soft_token>`, in order — the LOCAL prompt
/// positions where image soft-tokens splice in. Empty if the tokenizer lacks the token or
/// the prompt has no image block.
pub fn image_soft_token_positions(tok: &Tokenizer, ids: &[u32]) -> Vec<usize> {
    let Some(soft_id) = tok.token_to_id(SOFT_IMAGE_TOKEN) else {
        return Vec::new();
    };
    ids.iter()
        .enumerate()
        .filter(|(_, &id)| id == soft_id)
        .map(|(i, _)| i)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::ChatMessage;

    #[test]
    fn parse_tool_call_fenced_json() {
        let reply =
            "```json\n{\"function\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n```";
        let (calls, lead) = parse_tool_call(reply).expect("should parse");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, "{\"city\":\"Paris\"}");
        assert_eq!(lead, "");
    }

    #[test]
    fn parse_tool_call_name_alias_and_lead_text() {
        // accepts "name" as well as "function", and keeps any prose before the call.
        let reply = "Let me check.\n{\"name\": \"calc\", \"arguments\": {\"x\": 2}}";
        let (calls, lead) = parse_tool_call(reply).expect("should parse");
        assert_eq!(calls[0].function.name, "calc");
        assert_eq!(lead, "Let me check.");
    }

    #[test]
    fn parse_tool_call_none_when_plain_text() {
        assert!(parse_tool_call("The capital of France is Paris.").is_none());
    }

    #[test]
    fn parse_tool_call_gemma_shorthand_empty_args() {
        // gemma sometimes emits `call:NAME{}.` instead of JSON (A3/A4 in the demo).
        let (calls, _) = parse_tool_call("call:get_gpu_profile{}.").expect("shorthand");
        assert_eq!(calls[0].function.name, "get_gpu_profile");
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn parse_tool_call_gemma_shorthand_loose_args() {
        // `call:web_search{"query":"M4 Max"}` and the unquoted-key variant both parse.
        let (a, _) = parse_tool_call("call:web_search{\"query\":\"M4 Max\"}").expect("q");
        assert_eq!(a[0].function.name, "web_search");
        assert_eq!(a[0].function.arguments, "{\"query\":\"M4 Max\"}");
        let (b, _) = parse_tool_call("call:get_weather{city:\"Tokyo\"}").expect("loose");
        assert_eq!(b[0].function.name, "get_weather");
        assert_eq!(b[0].function.arguments, "{\"city\":\"Tokyo\"}");
    }

    /// The bare-key shorthand keeps the model's key order (it went through a sorted
    /// `serde_json::Value` until 2026-09-26) — the same bytes the quoted form gives.
    #[test]
    fn gemma_shorthand_loose_args_keep_the_models_key_order() {
        let (calls, _) = parse_tool_calls(
            "call:edit{path:\"calc.py\", oldText:\"We\", newText:\"Hi\"}",
            &["edit"],
        )
        .expect("loose");
        assert_eq!(
            calls[0].function.arguments,
            r#"{"path":"calc.py","oldText":"We","newText":"Hi"}"#
        );
        let (quoted, _) = parse_tool_calls(
            "call:edit{\"path\":\"calc.py\", \"oldText\":\"We\", \"newText\":\"Hi\"}",
            &["edit"],
        )
        .expect("quoted");
        assert_eq!(calls[0].function.arguments, quoted[0].function.arguments);
    }

    /// Non-ASCII text survives the bare-key shorthand, inside strings and out: it was pushed
    /// byte by byte `as char` (a Latin-1 decode) until 2026-09-26, so `héllo ✓` became
    /// `hÃ©llo â\u{9c}\u{93}` — valid JSON, and mojibake in the file the agent then wrote.
    #[test]
    fn gemma_shorthand_loose_args_keep_non_ascii_text() {
        let (calls, _) = parse_tool_calls(
            "call:write{content:\"héllo ✓ — 日本\", path:\"ü.txt\"}",
            &["write"],
        )
        .expect("loose");
        assert_eq!(
            calls[0].function.arguments,
            r#"{"content":"héllo ✓ — 日本","path":"ü.txt"}"#
        );
        // Unquoted non-ASCII after a key (not valid JSON either way) still falls back to `{}`
        // rather than panicking on a char boundary.
        let (calls, _) = parse_tool_calls("call:write{n: é}", &["write"]).expect("loose");
        assert_eq!(calls[0].function.arguments, "{}");
        assert_eq!(coerce_loose_args("{a:\"x\", \u{a0}b:1}"), "{}");
    }

    #[test]
    fn balanced_object_end_handles_nested_braces_and_strings() {
        let s = "junk {\"a\": {\"b\": 1}, \"s\": \"}{\"} tail";
        let start = s.find('{').unwrap();
        let end = balanced_object_end(s, start).unwrap();
        assert_eq!(&s[start..=end], "{\"a\": {\"b\": 1}, \"s\": \"}{\"}");
        assert_eq!(balanced_object_end("{\"open\": 1", 0), None);
        assert_eq!(balanced_object_end("x{}", 0), None); // not at a brace
    }

    /// Prose with braces BEFORE the call used to hide it: the parser took the first `{`, found
    /// `{name}`, failed, and returned the whole reply as text.
    #[test]
    fn parse_tool_calls_skips_prose_braces_before_the_call() {
        let reply = "The greeting should be f\"Hello, {name}!\" — I'll fix it.\n```json\n\
{\"function\": \"edit\", \"arguments\": {\"path\": \"hello.py\", \"newText\": \"return f\\\"Hello, {name}!\\\"\"}}\n```";
        let (calls, lead) = parse_tool_calls(reply, &["edit"]).expect("the call");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "edit");
        assert_eq!(
            calls[0].function.arguments,
            "{\"path\":\"hello.py\",\"newText\":\"return f\\\"Hello, {name}!\\\"\"}"
        );
        assert_eq!(
            lead,
            "The greeting should be f\"Hello, {name}!\" — I'll fix it."
        );
    }

    #[test]
    fn parse_tool_calls_returns_every_call_with_unique_ids() {
        let reply = "```json\n{\"function\": \"read\", \"arguments\": {\"path\": \"a.py\"}}\n```\n\
```json\n{\"function\": \"read\", \"arguments\": {\"path\": \"b.py\"}}\n```";
        let (calls, lead) = parse_tool_calls(reply, &[]).expect("two calls");
        assert_eq!(lead, "");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1].function.arguments, "{\"path\":\"b.py\"}");
        assert_ne!(calls[0].id, calls[1].id);
        assert!(calls[0].id.starts_with("call_"));
    }

    /// A raw newline inside an argument string (the model writing code) is repaired, and a
    /// multi-line value with quotes survives intact.
    #[test]
    fn parse_tool_calls_repairs_raw_newlines_in_strings() {
        let reply = "{\"function\": \"edit\", \"arguments\": {\"newText\": \"def add(a, b):\n    return a + b  # \\\"fixed\\\"\n\"}}";
        let (calls, _) = parse_tool_calls(reply, &[]).expect("repaired");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(
            args["newText"],
            "def add(a, b):\n    return a + b  # \"fixed\"\n"
        );
    }

    /// With the offered tool names known, JSON data that merely has a `name` key is an answer,
    /// not a call; the OpenAI-shaped echo `{"type":"function","function":{…}}` is a call.
    #[test]
    fn parse_tool_calls_checks_names_and_accepts_the_openai_echo() {
        let data = "Here you go: {\"name\": \"Alice\", \"age\": 3}";
        assert!(parse_tool_calls(data, &["read"]).is_none());
        assert!(parse_tool_calls(data, &[]).is_some()); // unchecked: today's lenient behaviour
        let echo = "{\"type\": \"function\", \"function\": {\"name\": \"read\", \"arguments\": \"{\\\"path\\\": \\\"x\\\"}\"}}";
        let (calls, _) = parse_tool_calls(echo, &["read"]).expect("echo");
        assert_eq!(calls[0].function.arguments, "{\"path\": \"x\"}");
        assert!(parse_tool_calls("No call here: just {braces} and text.", &["read"]).is_none());
    }

    /// A misnamed call reaches the client (which can repair it) instead of ending the turn as
    /// text: the wrong case gets the declared spelling, an unknown name with arguments is kept,
    /// and an unknown name without arguments is still data.
    #[test]
    fn parse_tool_calls_repairable_names() {
        let wrong_case = "{\"function\": \"Read\", \"arguments\": {\"path\": \"a.py\"}}";
        let (calls, _) = parse_tool_calls(wrong_case, &["read"]).expect("a call");
        assert_eq!(calls[0].function.name, "read");
        let unknown =
            "```json\n{\"function\": \"read_file\", \"arguments\": {\"path\": \"a.py\"}}\n```";
        let (calls, _) =
            parse_tool_calls(unknown, &["read"]).expect("kept for the client to repair");
        assert_eq!(calls[0].function.name, "read_file");
        let (calls, _) = parse_tool_calls("call:READ{}", &["read"]).expect("shorthand");
        assert_eq!(calls[0].function.name, "read");
        // Data after another object: no arguments key, unknown name — not a call.
        let data = "Example {\"id\": 1} and the service: {\"name\": \"web\", \"image\": \"nginx\"}";
        assert!(parse_tool_calls(data, &["read"]).is_none());
    }

    const LLAMA3_TOKENIZER: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../models/llama-3.2-1b/tokenizer.json"
    );

    // Llama-3.2 special token ids (verified against tokenizer.json).
    const BOS: u32 = 128000; // <|begin_of_text|>
    const EOT: u32 = 128009; // <|eot_id|>
    const EOT2: u32 = 128001; // <|end_of_text|>
    const START_HEADER: u32 = 128006; // <|start_header_id|>
    const END_HEADER: u32 = 128007; // <|end_header_id|>

    fn load_llama3_tokenizer() -> Option<Tokenizer> {
        let path = std::path::Path::new(LLAMA3_TOKENIZER);
        if !path.exists() {
            return None;
        }
        Some(Tokenizer::from_file(path).expect("tokenizer.json should parse"))
    }

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage::text(role, content)
    }

    /// chat_stop_tokens must include both llama-3 stop ids when the tokenizer is present.
    #[test]
    fn stop_tokens_include_llama3_eot() {
        let Some(tok) = load_llama3_tokenizer() else {
            eprintln!("SKIP: llama-3.2 tokenizer not found at {LLAMA3_TOKENIZER}");
            return;
        };
        let stops = chat_stop_tokens(&tok);
        assert!(
            stops.contains(&EOT),
            "chat_stop_tokens missing <|eot_id|> (128009); got: {stops:?}"
        );
        assert!(
            stops.contains(&EOT2),
            "chat_stop_tokens missing <|end_of_text|> (128001); got: {stops:?}"
        );
    }

    /// apply_chat_template must produce Llama-3 header format for the llama-3.2 tokenizer.
    #[test]
    fn template_llama3_format() {
        let Some(tok) = load_llama3_tokenizer() else {
            eprintln!("SKIP: llama-3.2 tokenizer not found at {LLAMA3_TOKENIZER}");
            return;
        };
        let messages = vec![msg("user", "Hello")];
        let prompt = apply_chat_template(&tok, &messages);
        assert!(
            prompt.contains("<|start_header_id|>user<|end_header_id|>"),
            "Llama-3 template missing user header; got: {prompt:?}"
        );
        assert!(
            prompt.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"),
            "Llama-3 template must end with assistant header cue; got: {prompt:?}"
        );
        // Must NOT contain generic "assistant\n" fallback at the end.
        assert!(
            !prompt.ends_with("assistant\n"),
            "Llama-3 template should not use generic fallback tail; got: {prompt:?}"
        );
    }

    // ── Gemma template (tokenizer-free string-shape tests) ──────────────────

    /// gemma_template: user message emits <start_of_turn>user marker.
    #[test]
    fn gemma_template_user_turn() {
        let messages = vec![msg("user", "Hello world")];
        let prompt = gemma_template(&messages);
        assert!(
            prompt.contains("<start_of_turn>user\nHello world<end_of_turn>\n"),
            "gemma user turn shape wrong; got: {prompt:?}"
        );
        assert!(
            prompt.ends_with("<start_of_turn>model\n"),
            "gemma must end with model cue; got: {prompt:?}"
        );
    }

    /// gemma_template: assistant message maps to <start_of_turn>model (NOT assistant).
    #[test]
    fn gemma_template_assistant_maps_to_model() {
        let messages = vec![msg("user", "Hi"), msg("assistant", "Hello!")];
        let prompt = gemma_template(&messages);
        assert!(
            prompt.contains("<start_of_turn>model\nHello!<end_of_turn>\n"),
            "gemma assistant must emit model role; got: {prompt:?}"
        );
        // Must not emit the literal string "assistant" as a role.
        assert!(
            !prompt.contains("<start_of_turn>assistant"),
            "gemma must never emit <start_of_turn>assistant; got: {prompt:?}"
        );
    }

    /// gemma_template: system message is folded into the user turn.
    #[test]
    fn gemma_template_system_folded_into_user() {
        let messages = vec![msg("system", "Be helpful.")];
        let prompt = gemma_template(&messages);
        assert!(
            prompt.contains("<start_of_turn>user\nBe helpful.<end_of_turn>\n"),
            "gemma system must be folded into user turn; got: {prompt:?}"
        );
        assert!(
            !prompt.contains("<start_of_turn>system"),
            "gemma must not emit <start_of_turn>system; got: {prompt:?}"
        );
    }

    /// gemma_template: end-of-turn marker is present.
    #[test]
    fn gemma_template_contains_end_of_turn() {
        let messages = vec![msg("user", "Test")];
        let prompt = gemma_template(&messages);
        assert!(
            prompt.contains("<end_of_turn>"),
            "gemma prompt must contain <end_of_turn>; got: {prompt:?}"
        );
    }

    // ── Gemma-4 channel template (tokenizer-free string-shape tests) ─────────

    /// gemma4_template: a turn header is `<|turn>{role}\n`, closed by `<turn|>\n`.
    ///
    /// The shapes here are transcribed from the GGUF's own `tokenizer.chat_template`. The
    /// previous revision of this test asserted `<|turn>user<channel|>…`, which is where the
    /// leaked `<|channel>thought` markers in real replies came from — the test agreed with the
    /// code and both disagreed with the model.
    #[test]
    fn gemma4_template_user_turn() {
        let messages = vec![msg("user", "Hello world")];
        let prompt = gemma4_template(&messages);
        assert!(
            prompt.contains("<|turn>user\nHello world<turn|>\n"),
            "gemma-4 user turn shape wrong; got: {prompt:?}"
        );
        // must NOT use the old gemma-2/3 delimiters
        assert!(
            !prompt.contains("<start_of_turn>") && !prompt.contains("<end_of_turn>"),
            "gemma-4 must not emit old start/end_of_turn; got: {prompt:?}"
        );
    }

    /// gemma4_template: the generation cue opens AND closes an empty thought channel.
    ///
    /// This is the whole fix for the leaked markers: the cue spends the model's thinking turn
    /// for it (the template's `add_generation_prompt` branch with `enable_thinking` false).
    /// Without the trailing `<|channel>thought\n<channel|>` the model emits that header itself,
    /// mid-reply.
    #[test]
    fn gemma4_template_generation_cue_closes_thought_channel() {
        let prompt = gemma4_template(&[msg("user", "Hi")]);
        assert!(
            prompt.ends_with("<|turn>model\n<|channel>thought\n<channel|>"),
            "gemma-4 cue must pre-close the thought channel; got: {prompt:?}"
        );
    }

    /// gemma4_template: assistant maps to "model" role.
    #[test]
    fn qwen3_think_template_renders_the_official_form() {
        let messages = vec![msg(
            "user",
            "Explain how a bicycle stays upright, in about 200 words.",
        )];
        let prompt = qwen3_think_template(&messages, None);
        assert_eq!(
            prompt,
            format!(
                "<|im_start|>system\n{QWEN3_EFFORT_XHIGH}<|im_end|>\n<|im_start|>user\nExplain how a bicycle stays upright, in about 200 words.<|im_end|>\n<|im_start|>assistant\n<think>\n"
            )
        );
        assert!(qwen3_think_template(&messages, Some("medium")).starts_with("<|im_start|>user\n"));
        assert!(qwen3_think_template(&messages, Some("low")).contains(QWEN3_EFFORT_LOW));
        let off = qwen3_think_template(&messages, Some("none"));
        assert!(off.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        assert!(!off.contains("Reasoning effort"));
    }

    #[test]
    fn split_pretag_think_splits_at_the_close_tag() {
        let (r, a) =
            split_pretag_think("We need answer.\n</think>\n\nA bicycle stays upright because...");
        assert_eq!(r.as_deref(), Some("We need answer."));
        assert_eq!(a, "A bicycle stays upright because...");
        let (r, a) = split_pretag_think("still thinking");
        assert_eq!(r.as_deref(), Some("still thinking"));
        assert_eq!(a, "");
    }

    #[test]
    fn gemma4_template_assistant_maps_to_model() {
        let messages = vec![msg("user", "Hi"), msg("assistant", "Hey!")];
        let prompt = gemma4_template(&messages);
        assert!(
            prompt.contains("<|turn>model\nHey!<turn|>\n"),
            "gemma-4 assistant must map to model role; got: {prompt:?}"
        );
        assert!(
            !prompt.contains("<|turn>assistant"),
            "gemma-4 must never emit assistant role; got: {prompt:?}"
        );
    }

    /// strip_thought removes a whole `<|channel>thought…<channel|>` block, mirroring the
    /// template's own `strip_thinking` macro — this is the exact shape that reached users
    /// before the cue was fixed.
    #[test]
    fn strip_thought_removes_channel_block() {
        let raw = "The capital of France is Paris.\n<|channel>thought\n<channel|>";
        assert_eq!(strip_thought(raw), "The capital of France is Paris.");
    }

    /// strip_thought must not eat ordinary prose that merely begins with "thought".
    #[test]
    fn strip_thought_keeps_thoughtful_prose() {
        assert_eq!(
            strip_thought("thoughtful design matters"),
            "thoughtful design matters"
        );
    }

    /// gemma4_template: a LEADING system message becomes the dedicated opening system turn
    /// and is dropped from the message loop. gemma-4 has a real `system` role — unlike
    /// gemma-2/3, where this folded into the user turn.
    #[test]
    fn gemma4_template_leading_system_becomes_system_turn() {
        let messages = vec![msg("system", "Be terse."), msg("user", "Hi")];
        let prompt = gemma4_template(&messages);
        assert!(
            prompt.starts_with("<|turn>system\nBe terse.<turn|>\n"),
            "gemma-4 leading system turn wrong; got: {prompt:?}"
        );
        assert!(
            prompt.contains("<|turn>user\nHi<turn|>\n"),
            "user turn must still follow the system turn; got: {prompt:?}"
        );
        assert_eq!(
            prompt.matches("Be terse.").count(),
            1,
            "system content must not be duplicated into the loop; got: {prompt:?}"
        );
    }

    // ── ChatML / Qwen template (tokenizer-free string-shape tests) ───────────

    /// chatml_template: user message wrapped in im_start/im_end.
    #[test]
    fn chatml_template_user_turn() {
        let messages = vec![msg("user", "Hello")];
        let prompt = chatml_template(&messages);
        assert!(
            prompt.contains("<|im_start|>user\nHello<|im_end|>\n"),
            "chatml user turn shape wrong; got: {prompt:?}"
        );
        assert!(
            prompt.ends_with("<|im_start|>assistant\n"),
            "chatml must end with assistant cue; got: {prompt:?}"
        );
    }

    /// chatml_template: system message uses system role (not folded).
    #[test]
    fn chatml_template_system_role_preserved() {
        let messages = vec![msg("system", "Be helpful.")];
        let prompt = chatml_template(&messages);
        assert!(
            prompt.contains("<|im_start|>system\nBe helpful.<|im_end|>\n"),
            "chatml system turn shape wrong; got: {prompt:?}"
        );
    }

    /// chatml_template: assistant message uses assistant role as-is.
    #[test]
    fn chatml_template_assistant_turn() {
        let messages = vec![msg("user", "Hi"), msg("assistant", "Hello!")];
        let prompt = chatml_template(&messages);
        assert!(
            prompt.contains("<|im_start|>assistant\nHello!<|im_end|>\n"),
            "chatml assistant turn shape wrong; got: {prompt:?}"
        );
    }

    /// chatml_template: im_end token is present.
    #[test]
    fn chatml_template_contains_im_end() {
        let messages = vec![msg("user", "Test")];
        let prompt = chatml_template(&messages);
        assert!(
            prompt.contains("<|im_end|>"),
            "chatml prompt must contain <|im_end|>; got: {prompt:?}"
        );
    }

    // ── Llama-3 tests (unchanged) ─────────────────────────────────────────────

    /// encode_prompt on the Llama-3 template must start with exactly one BOS
    /// and encode the header special tokens as single atomic ids (not split).
    #[test]
    fn encode_prompt_llama3_bos_and_atomic_headers() {
        let Some(tok) = load_llama3_tokenizer() else {
            eprintln!("SKIP: llama-3.2 tokenizer not found at {LLAMA3_TOKENIZER}");
            return;
        };
        let messages = vec![msg("user", "Hi")];
        let prompt = apply_chat_template(&tok, &messages);
        let ids = encode_prompt(&tok, &prompt).expect("encode_prompt failed");

        // Exactly one BOS at the start.
        assert_eq!(
            ids.first().copied(),
            Some(BOS),
            "encoded ids must start with BOS (128000); got: {ids:?}"
        );
        assert!(
            ids.get(1).copied() != Some(BOS),
            "BOS must appear exactly once; got duplicate at index 1: {ids:?}"
        );

        // Header tokens must appear as atomic single ids (not split into BPE pieces).
        assert!(
            ids.contains(&START_HEADER),
            "ids must contain <|start_header_id|> (128006) as atomic id; got: {ids:?}"
        );
        assert!(
            ids.contains(&END_HEADER),
            "ids must contain <|end_header_id|> (128007) as atomic id; got: {ids:?}"
        );
        assert!(
            ids.contains(&EOT),
            "ids must contain <|eot_id|> (128009) as atomic id; got: {ids:?}"
        );
    }
}

#[cfg(test)]
mod muse_glimmer_tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: content.into(),
            ..Default::default()
        }
    }

    /// The template must match llama's `/apply-template` output byte for byte. This is the
    /// exact string llama-server produced for this message (with the date line rendered from
    /// the clock, so only the fixed parts are asserted here).
    #[test]
    fn template_matches_llama_wire_format() {
        let p =
            muse_glimmer_template(&[msg("user", "Write a Rust function that reverses a string.")]);
        assert!(p.starts_with("<|start|>system<|message|>You are a helpful AI assistant.\nKnowledge cutoff: 2026-01-04.\nCurrent date: "), "{p}");
        assert!(
            p.contains(
                "\n\nReasoning strength: high.\n\n# Valid recipients: \"self\", \"user\".<|eot|>"
            ),
            "{p}"
        );
        assert!(
            p.contains(
                "<|start|>user<|message|>Write a Rust function that reverses a string.<|eot|>"
            ),
            "{p}"
        );
        // The generation cue is a BARE assistant header — no trailing <|message|>.
        assert!(p.ends_with("<|start|>assistant"), "{p}");
        assert!(!p.ends_with("<|message|>"), "{p}");
    }

    /// A caller-supplied system message must SUPPRESS the injected default (llama's
    /// `ns.has_system` guard), not stack on top of it.
    #[test]
    fn caller_system_suppresses_the_default() {
        let p = muse_glimmer_template(&[msg("system", "Be terse."), msg("user", "hi")]);
        assert!(
            p.starts_with("<|start|>system<|message|>Be terse.<|eot|>"),
            "{p}"
        );
        assert!(!p.contains("You are a helpful AI assistant."), "{p}");
    }

    /// The real shape, taken from an actual reference generation: think on `to=self`, then
    /// re-header into `to=user` for the answer. Only the answer is user-facing.
    #[test]
    fn channel_split_keeps_only_the_user_answer() {
        let raw = " to=self<|message|>We need to write a function. Probably simple.\
                   <|start|>assistant to=user<|message|>```rust\nfn f() {}\n```";
        let (reasoning, answer) = split_muse_channels(raw);
        assert_eq!(answer, "```rust\nfn f() {}\n```");
        assert_eq!(
            reasoning.as_deref(),
            Some("We need to write a function. Probably simple.")
        );
    }

    /// A chain of thought can re-enter `self`; the LAST user channel is the answer.
    #[test]
    fn channel_split_takes_the_last_user_channel() {
        let raw = " to=self<|message|>a<|start|>assistant to=user<|message|>first\
                   <|start|>assistant to=self<|message|>b<|start|>assistant to=user<|message|>final";
        let (_r, answer) = split_muse_channels(raw);
        assert_eq!(answer, "final");
    }

    /// Cut off mid-thought (hit the token limit before any `to=user`): show the thinking
    /// rather than an empty reply, with its header stripped.
    #[test]
    fn channel_split_truncated_thought_still_returns_text() {
        let (reasoning, answer) = split_muse_channels(" to=self<|message|>still thinking");
        assert_eq!(answer, "still thinking");
        assert!(reasoning.is_none());
    }

    /// Every other model: no channel header, so the reply passes through untouched.
    #[test]
    fn channel_split_is_a_noop_without_headers() {
        let (reasoning, answer) = split_muse_channels("Just a normal reply.");
        assert_eq!(answer, "Just a normal reply.");
        assert!(reasoning.is_none());
    }

    /// The dispatcher must not claim a vocab that lacks the pair, and must not shadow ChatML.
    #[test]
    fn stop_tokens_names_include_the_muse_terminators() {
        // Name-level check (no tokenizer needed): both must be in the candidate list.
        let src = include_str!("chat.rs");
        assert!(
            src.contains("\"<|eot|>\","),
            "chat_stop_tokens missing <|eot|>"
        );
        // <|eom|> must NOT be a stop: it closes the to=self thought, and the answer follows.
        assert!(
            !src.contains("\n        \"<|eom|>\","),
            "<|eom|> must not be a stop token"
        );
    }
}
