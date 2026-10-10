//! `POST /v1/responses` — the OpenAI Responses API, as an adapter over `/v1/chat/completions`.
//!
//! The request is translated to a Chat Completions body, run through the chat handler unchanged
//! (so tools, stop strings, the prefix cache and speculation all apply), and its answer is mapped
//! back: a `response` object with `reasoning`, `message` and `function_call` output items, or —
//! streamed — the Responses event sequence (`response.created`, `response.output_item.added`,
//! `response.output_text.delta`, `response.function_call_arguments.delta`, the `.done` events,
//! `response.completed`).
//!
//! PORTED SOURCE: the request translation follows llama.cpp's
//! `server_chat_convert_responses_to_chatcmpl` (tools/server/server-chat.cpp) and the output shapes
//! its `to_json_oaicompat_resp*` (tools/server/server-task.cpp), MIT — see NOTICE.
//!
//! What is NOT supported, stated so nobody assumes it: `previous_response_id` and stored responses
//! (refused with a 400), `input_file` parts, and non-function tools (skipped).

use super::*;
use serde_json::{json, Value};

/// A short random-enough suffix for item ids (`resp_…`, `msg_…`, `fc_…`).
fn suffix() -> String {
    let n = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{n:x}")
}

fn bad_request(msg: impl Into<String>) -> Response {
    (
        axum::http::StatusCode::BAD_REQUEST,
        Json(json!({"error": {"message": msg.into(), "type": "invalid_request_error"}})),
    )
        .into_response()
}

/// Responses request → Chat Completions request.
pub(super) fn to_chat(body: &Value) -> Result<Value, String> {
    let obj = body
        .as_object()
        .ok_or("the request body must be a JSON object")?;
    let input = obj.get("input").ok_or("'input' is required")?;
    if obj
        .get("previous_response_id")
        .and_then(Value::as_str)
        .is_some_and(|s| !s.is_empty())
    {
        return Err(
            "'previous_response_id' is not supported: send the conversation in 'input'".into(),
        );
    }
    let mut chat = obj.clone();
    for k in [
        "input",
        "instructions",
        "tools",
        "max_output_tokens",
        "reasoning",
        "store",
        "previous_response_id",
        "text",
        "truncation",
        "include",
    ] {
        chat.remove(k);
    }
    let mut messages: Vec<Value> = Vec::new();
    if let Some(instr) = obj.get("instructions").and_then(Value::as_str) {
        messages.push(json!({"role": "system", "content": instr}));
    }
    match input {
        Value::String(s) => messages.push(json!({"role": "user", "content": s})),
        Value::Array(items) => {
            for item in items {
                push_item(&mut messages, item)?;
            }
        }
        _ => return Err("'input' must be a string or an array of items".into()),
    }
    chat.insert("messages".into(), Value::Array(messages));

    if let Some(tools) = obj.get("tools") {
        let tools = tools.as_array().ok_or("'tools' must be an array")?;
        let mut out = Vec::new();
        for t in tools {
            if t.get("type").and_then(Value::as_str) != Some("function") {
                continue; // no Chat Completions equivalent (web search, file search, ...)
            }
            let mut f = t.as_object().cloned().unwrap_or_default();
            f.remove("type");
            out.push(json!({"type": "function", "function": f}));
        }
        if !out.is_empty() {
            chat.insert("tools".into(), Value::Array(out));
        }
    }
    // tool_choice: Responses' {"type": "function", "name": X} is Chat's
    // {"type": "function", "function": {"name": X}}; the string forms carry over.
    if let Some(tc) = obj.get("tool_choice") {
        if let Some(name) = tc.get("name").and_then(Value::as_str) {
            chat.insert(
                "tool_choice".into(),
                json!({"type": "function", "function": {"name": name}}),
            );
        }
    }
    if let Some(m) = obj.get("max_output_tokens") {
        chat.insert("max_tokens".into(), m.clone());
    }
    if let Some(effort) = obj.get("reasoning").and_then(|r| r.get("effort")) {
        chat.insert("reasoning_effort".into(), effort.clone());
    }
    if chat.get("stream").and_then(Value::as_bool) == Some(true) {
        chat.insert("stream_options".into(), json!({"include_usage": true}));
    }
    Ok(Value::Object(chat))
}

/// Append one Responses input item to the chat messages. An assistant `message`, `function_call`
/// or `reasoning` item directly after an assistant message merges into it (one assistant turn).
fn push_item(messages: &mut Vec<Value>, item: &Value) -> Result<(), String> {
    let ty = item
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("message");
    let role = item.get("role").and_then(Value::as_str);
    let prev_assistant = messages
        .last()
        .is_some_and(|m| m.get("role").and_then(Value::as_str) == Some("assistant"));
    match (ty, role) {
        ("message", Some(r @ ("user" | "system" | "developer"))) => {
            let content = match item.get("content") {
                Some(Value::String(s)) => Value::String(s.clone()),
                Some(Value::Array(parts)) => {
                    let mut out = Vec::new();
                    for p in parts {
                        match p.get("type").and_then(Value::as_str) {
                            Some("input_text") => {
                                let t = p
                                    .get("text")
                                    .and_then(Value::as_str)
                                    .ok_or("'input_text' needs 'text'")?;
                                out.push(json!({"type": "text", "text": t}));
                            }
                            Some("input_image") => {
                                let url = p
                                    .get("image_url")
                                    .and_then(Value::as_str)
                                    .ok_or("'input_image' needs 'image_url'")?;
                                out.push(json!({"type": "image_url", "image_url": {"url": url}}));
                            }
                            Some("input_file") => {
                                return Err("'input_file' parts are not supported".into())
                            }
                            _ => {
                                return Err(
                                    "a content part must be 'input_text' or 'input_image'".into()
                                )
                            }
                        }
                    }
                    Value::Array(out)
                }
                _ => return Err("a message needs 'content'".into()),
            };
            messages.push(json!({"role": r, "content": content}));
        }
        ("message", Some("assistant")) => {
            let text = match item.get("content") {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Array(parts)) => parts
                    .iter()
                    .filter_map(|p| p.get("text").and_then(Value::as_str))
                    .collect::<Vec<_>>()
                    .join(""),
                _ => String::new(),
            };
            if prev_assistant {
                let m = messages.last_mut().unwrap();
                let old = m
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                m["content"] = Value::String(old + &text);
            } else {
                messages.push(json!({"role": "assistant", "content": text}));
            }
        }
        ("function_call", _) => {
            let call = json!({
                "id": item.get("call_id").and_then(Value::as_str).ok_or("'function_call' needs 'call_id'")?,
                "type": "function",
                "function": {
                    "name": item.get("name").and_then(Value::as_str).ok_or("'function_call' needs 'name'")?,
                    "arguments": item.get("arguments").and_then(Value::as_str).unwrap_or("{}"),
                },
            });
            if prev_assistant {
                let m = messages.last_mut().unwrap();
                match m.get_mut("tool_calls").and_then(Value::as_array_mut) {
                    Some(calls) => calls.push(call),
                    None => m["tool_calls"] = json!([call]),
                }
            } else {
                messages.push(json!({"role": "assistant", "content": null, "tool_calls": [call]}));
            }
        }
        ("function_call_output", _) => {
            let id = item
                .get("call_id")
                .and_then(Value::as_str)
                .ok_or("'function_call_output' needs 'call_id'")?;
            let output = match item.get("output") {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Array(parts)) => parts
                    .iter()
                    .filter_map(|p| p.get("text").and_then(Value::as_str))
                    .collect::<Vec<_>>()
                    .join(""),
                _ => return Err("'function_call_output' needs 'output'".into()),
            };
            messages.push(json!({"role": "tool", "tool_call_id": id, "content": output}));
        }
        ("reasoning", _) => {
            // Earlier reasoning is not replayed into the prompt (the chat templates drop it too).
        }
        _ => return Err(format!("unsupported input item type '{ty}'")),
    }
    Ok(())
}

/// A finished Chat Completion (JSON) → a Responses `response` object.
pub(super) fn from_chat(chat: &Value, resp_id: &str) -> Value {
    let choice = &chat["choices"][0];
    let msg = &choice["message"];
    let mut output = Vec::new();
    if let Some(r) = msg["reasoning_content"].as_str().filter(|r| !r.is_empty()) {
        output.push(reasoning_item(&format!("rs_{}", suffix()), r, "completed"));
    }
    if let Some(t) = msg["content"].as_str().filter(|t| !t.is_empty()) {
        output.push(message_item(&format!("msg_{}", suffix()), t, "completed"));
    }
    if let Some(calls) = msg["tool_calls"].as_array() {
        for c in calls {
            let id = c["id"].as_str().unwrap_or("");
            output.push(call_item(
                id,
                c["function"]["name"].as_str().unwrap_or(""),
                c["function"]["arguments"].as_str().unwrap_or(""),
                "completed",
            ));
        }
    }
    let (inp, out) = (
        chat["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
        chat["usage"]["completion_tokens"].as_u64().unwrap_or(0),
    );
    response_obj(
        resp_id,
        chat["model"].as_str().unwrap_or(""),
        output,
        choice["finish_reason"].as_str(),
        inp,
        out,
    )
}

fn reasoning_item(id: &str, text: &str, status: &str) -> Value {
    json!({"id": id, "type": "reasoning", "summary": [], "status": status,
           "content": [{"type": "reasoning_text", "text": text}]})
}

fn message_item(id: &str, text: &str, status: &str) -> Value {
    json!({"id": id, "type": "message", "role": "assistant", "status": status,
           "content": [{"type": "output_text", "text": text, "annotations": [], "logprobs": []}]})
}

fn call_item(call_id: &str, name: &str, arguments: &str, status: &str) -> Value {
    json!({"id": format!("fc_{call_id}"), "type": "function_call", "status": status,
           "call_id": call_id, "name": name, "arguments": arguments})
}

fn response_obj(
    id: &str,
    model: &str,
    output: Vec<Value>,
    finish: Option<&str>,
    inp: u64,
    out: u64,
) -> Value {
    let now = now();
    let mut r = json!({
        "id": id, "object": "response", "created_at": now, "model": model,
        "status": "completed", "output": output,
        "usage": {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out},
    });
    if finish == Some("length") {
        r["status"] = json!("incomplete");
        r["incomplete_details"] = json!({"reason": "max_output_tokens"});
    } else {
        r["completed_at"] = json!(now);
    }
    r
}

/// The handler.
pub(super) async fn responses(State(state): State<AppState>, body: axum::body::Bytes) -> Response {
    let req: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return bad_request(format!("invalid JSON: {e}")),
    };
    let chat_body = match to_chat(&req) {
        Ok(c) => c,
        Err(e) => return bad_request(e),
    };
    let stream = chat_body.get("stream").and_then(Value::as_bool) == Some(true);
    let resp_id = format!("resp_{}", suffix());
    let resp =
        super::chat_completions(State(state), axum::body::Bytes::from(chat_body.to_string())).await;
    if !resp.status().is_success() {
        return resp; // the chat handler's own error, already in OpenAI's shape
    }
    if !stream {
        let bytes = match axum::body::to_bytes(resp.into_body(), usize::MAX).await {
            Ok(b) => b,
            Err(e) => return bad_request(format!("reading the reply: {e}")),
        };
        let chat: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        return Json(from_chat(&chat, &resp_id)).into_response();
    }
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(64);
    let model = req["model"].as_str().unwrap_or("").to_string();
    tokio::spawn(async move {
        let mut conv = StreamConverter::new(resp_id, model);
        let mut body = resp.into_body().into_data_stream();
        let mut buf = String::new();
        use tokio_stream::StreamExt;
        while let Some(chunk) = body.next().await {
            let Ok(chunk) = chunk else { break };
            buf.push_str(&String::from_utf8_lossy(&chunk));
            while let Some(end) = buf.find("\n\n") {
                let frame: String = buf.drain(..end + 2).collect();
                for line in frame.lines() {
                    let Some(data) = line.strip_prefix("data:") else {
                        continue;
                    };
                    let data = data.trim();
                    if data == "[DONE]" {
                        continue;
                    }
                    if let Ok(v) = serde_json::from_str::<Value>(data) {
                        for (name, ev) in conv.chunk(&v) {
                            if tx
                                .send(Ok(Event::default().event(name).data(ev.to_string())))
                                .await
                                .is_err()
                            {
                                return;
                            }
                        }
                    }
                }
            }
        }
        for (name, ev) in conv.finish() {
            if tx
                .send(Ok(Event::default().event(name).data(ev.to_string())))
                .await
                .is_err()
            {
                return;
            }
        }
    });
    Sse::new(ReceiverStream::new(rx))
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Chat stream chunks → Responses stream events.
pub(super) struct StreamConverter {
    id: String,
    model: String,
    started: bool,
    reasoning: Option<(String, String)>,  // (item id, text)
    message: Option<(String, String)>,    // (item id, text)
    calls: Vec<(String, String, String)>, // (call id, name, arguments), by stream index
    finish: Option<String>,
    usage: (u64, u64),
}

impl StreamConverter {
    pub(super) fn new(id: String, model: String) -> Self {
        StreamConverter {
            id,
            model,
            started: false,
            reasoning: None,
            message: None,
            calls: Vec::new(),
            finish: None,
            usage: (0, 0),
        }
    }

    fn shell(&self, status: &str) -> Value {
        json!({"id": self.id, "object": "response", "status": status, "model": self.model})
    }

    pub(super) fn chunk(&mut self, v: &Value) -> Vec<(&'static str, Value)> {
        let mut ev = Vec::new();
        if !self.started {
            self.started = true;
            ev.push((
                "response.created",
                json!({"type": "response.created", "response": self.shell("in_progress")}),
            ));
            ev.push((
                "response.in_progress",
                json!({"type": "response.in_progress", "response": self.shell("in_progress")}),
            ));
        }
        if let Some(u) = v.get("usage").filter(|u| u.is_object()) {
            self.usage = (
                u["prompt_tokens"].as_u64().unwrap_or(0),
                u["completion_tokens"].as_u64().unwrap_or(0),
            );
        }
        let Some(choice) = v["choices"].get(0) else {
            return ev;
        };
        let d = &choice["delta"];
        if let Some(r) = d["reasoning_content"].as_str().filter(|s| !s.is_empty()) {
            if self.reasoning.is_none() {
                let id = format!("rs_{}", suffix());
                ev.push(("response.output_item.added", json!({"type": "response.output_item.added",
                    "item": {"id": id, "type": "reasoning", "summary": [], "content": [], "status": "in_progress"}})));
                self.reasoning = Some((id, String::new()));
            }
            let (id, text) = self.reasoning.as_mut().unwrap();
            text.push_str(r);
            ev.push((
                "response.reasoning_text.delta",
                json!({"type": "response.reasoning_text.delta", "item_id": id, "delta": r}),
            ));
        }
        if let Some(t) = d["content"].as_str().filter(|s| !s.is_empty()) {
            if self.message.is_none() {
                let id = format!("msg_{}", suffix());
                ev.push(("response.output_item.added", json!({"type": "response.output_item.added",
                    "item": {"id": id, "type": "message", "role": "assistant", "content": [], "status": "in_progress"}})));
                ev.push((
                    "response.content_part.added",
                    json!({"type": "response.content_part.added",
                    "item_id": id, "part": {"type": "output_text", "text": ""}}),
                ));
                self.message = Some((id, String::new()));
            }
            let (id, text) = self.message.as_mut().unwrap();
            text.push_str(t);
            ev.push((
                "response.output_text.delta",
                json!({"type": "response.output_text.delta", "item_id": id, "delta": t}),
            ));
        }
        if let Some(calls) = d["tool_calls"].as_array() {
            for c in calls {
                let idx = c["index"].as_u64().unwrap_or(self.calls.len() as u64) as usize;
                if idx >= self.calls.len() {
                    let call_id = c["id"]
                        .as_str()
                        .map(str::to_string)
                        .unwrap_or_else(|| format!("call_{}", suffix()));
                    let name = c["function"]["name"].as_str().unwrap_or("").to_string();
                    ev.push((
                        "response.output_item.added",
                        json!({"type": "response.output_item.added",
                        "item": call_item(&call_id, &name, "", "in_progress")}),
                    ));
                    self.calls.push((call_id, name, String::new()));
                }
                if let Some(a) = c["function"]["arguments"]
                    .as_str()
                    .filter(|s| !s.is_empty())
                {
                    let at = idx.min(self.calls.len() - 1);
                    let (call_id, _, args) = &mut self.calls[at];
                    args.push_str(a);
                    ev.push((
                        "response.function_call_arguments.delta",
                        json!({"type": "response.function_call_arguments.delta",
                        "item_id": format!("fc_{call_id}"), "delta": a}),
                    ));
                }
            }
        }
        if let Some(f) = choice["finish_reason"].as_str() {
            self.finish = Some(f.to_string());
        }
        ev
    }

    pub(super) fn finish(mut self) -> Vec<(&'static str, Value)> {
        let mut ev = Vec::new();
        if !self.started {
            ev.extend(self.chunk(&json!({})));
        }
        let mut output = Vec::new();
        if let Some((id, text)) = &self.reasoning {
            let item = reasoning_item(id, text, "completed");
            ev.push((
                "response.output_item.done",
                json!({"type": "response.output_item.done", "item": item}),
            ));
            output.push(item);
        }
        if let Some((id, text)) = &self.message {
            ev.push((
                "response.output_text.done",
                json!({"type": "response.output_text.done", "item_id": id, "text": text}),
            ));
            ev.push((
                "response.content_part.done",
                json!({"type": "response.content_part.done", "item_id": id,
                "part": {"type": "output_text", "text": text, "annotations": []}}),
            ));
            let item = message_item(id, text, "completed");
            ev.push((
                "response.output_item.done",
                json!({"type": "response.output_item.done", "item": item}),
            ));
            output.push(item);
        }
        for (call_id, name, args) in &self.calls {
            ev.push((
                "response.function_call_arguments.done",
                json!({"type": "response.function_call_arguments.done",
                "item_id": format!("fc_{call_id}"), "arguments": args}),
            ));
            let item = call_item(call_id, name, args, "completed");
            ev.push((
                "response.output_item.done",
                json!({"type": "response.output_item.done", "item": item}),
            ));
            output.push(item);
        }
        let r = response_obj(
            &self.id,
            &self.model,
            output,
            self.finish.as_deref(),
            self.usage.0,
            self.usage.1,
        );
        let name = if r["status"] == "incomplete" {
            "response.incomplete"
        } else {
            "response.completed"
        };
        ev.push((name, json!({"type": name, "response": r})));
        ev
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::app_state;
    use super::*;

    #[test]
    fn input_items_translate_to_chat_messages() {
        let req = json!({
        "model": "m", "instructions": "Be brief.", "max_output_tokens": 64,
        "reasoning": {"effort": "low"},
        "tools": [{"type": "function", "name": "read", "parameters": {"type": "object"}},
                  {"type": "web_search"}],
        "tool_choice": {"type": "function", "name": "read"},
        "input": [
            {"role": "user", "content": "fix calc.py"},
            {"type": "function_call", "call_id": "c1", "name": "read", "arguments": "{\"path\":\"calc.py\"}"},
            {"type": "function_call_output", "call_id": "c1", "output": "x = 1"},
            {"role": "user", "content": [{"type": "input_text", "text": "now?"}]},
        ]});
        let c = to_chat(&req).unwrap();
        let m = c["messages"].as_array().unwrap();
        assert_eq!(m[0], json!({"role": "system", "content": "Be brief."}));
        assert_eq!(m[1], json!({"role": "user", "content": "fix calc.py"}));
        assert_eq!(m[2]["role"], "assistant");
        assert_eq!(m[2]["tool_calls"][0]["function"]["name"], "read");
        assert_eq!(
            m[3],
            json!({"role": "tool", "tool_call_id": "c1", "content": "x = 1"})
        );
        assert_eq!(m[4]["content"][0], json!({"type": "text", "text": "now?"}));
        assert_eq!(c["max_tokens"], 64);
        assert_eq!(c["reasoning_effort"], "low");
        assert_eq!(
            c["tools"].as_array().unwrap().len(),
            1,
            "non-function tools are skipped"
        );
        assert_eq!(c["tools"][0]["function"]["name"], "read");
        assert_eq!(
            c["tool_choice"],
            json!({"type": "function", "function": {"name": "read"}})
        );
        assert!(c.get("input").is_none() && c.get("instructions").is_none());
    }

    #[test]
    fn unsupported_requests_are_refused() {
        assert!(to_chat(&json!({"model": "m"})).is_err(), "no input");
        assert!(to_chat(&json!({"input": "x", "previous_response_id": "resp_1"})).is_err());
        assert!(to_chat(
            &json!({"input": [{"role": "user", "content": [{"type": "input_file"}]}]})
        )
        .is_err());
        assert!(to_chat(&json!({"input": 3})).is_err());
    }

    #[test]
    fn a_chat_completion_maps_to_response_items() {
        let chat = json!({"model": "m", "choices": [{"finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": "On it.", "reasoning_content": "think",
            "tool_calls": [{"id": "c9", "type": "function", "function": {"name": "read", "arguments": "{\"path\":\"a.py\"}"}}]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}});
        let r = from_chat(&chat, "resp_x");
        assert_eq!(r["object"], "response");
        assert_eq!(r["status"], "completed");
        let out = r["output"].as_array().unwrap();
        assert_eq!(out[0]["type"], "reasoning");
        assert_eq!(out[1]["content"][0]["text"], "On it.");
        assert_eq!(out[2]["type"], "function_call");
        assert_eq!(out[2]["call_id"], "c9");
        assert_eq!(out[2]["arguments"], "{\"path\":\"a.py\"}");
        assert_eq!(r["usage"]["total_tokens"], 15);
        let cut = json!({"model": "m", "choices": [{"finish_reason": "length", "message": {"content": "a"}}]});
        assert_eq!(from_chat(&cut, "r")["status"], "incomplete");
    }

    #[test]
    fn stream_chunks_become_response_events() {
        let mut c = StreamConverter::new("resp_1".into(), "m".into());
        let mut names: Vec<&str> = Vec::new();
        for ch in [
            json!({"choices": [{"delta": {"role": "assistant"}}]}),
            json!({"choices": [{"delta": {"content": "Hi"}}]}),
            json!({"choices": [{"delta": {"content": " there"}}]}),
            json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "read", "arguments": "{\"p\""}}]}}]}),
            json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ":1}"}}]}}]}),
            json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
            json!({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 4}}),
        ] {
            names.extend(c.chunk(&ch).into_iter().map(|(n, _)| n));
        }
        let fin = c.finish();
        let last = fin.last().unwrap().1.clone();
        names.extend(fin.into_iter().map(|(n, _)| n));
        assert_eq!(&names[..2], ["response.created", "response.in_progress"]);
        assert!(names.contains(&"response.output_text.delta"));
        assert!(names.contains(&"response.function_call_arguments.delta"));
        assert_eq!(*names.last().unwrap(), "response.completed");
        let out = last["response"]["output"].as_array().unwrap();
        assert_eq!(out[0]["content"][0]["text"], "Hi there");
        assert_eq!(out[1]["arguments"], "{\"p\":1}");
        assert_eq!(last["response"]["usage"]["total_tokens"], 7);
    }

    /// Streamed, end to end: the chat handler's real SSE frames are parsed and re-emitted as
    /// Responses events, ending in `response.completed` with the assembled text.
    #[tokio::test]
    async fn responses_stream_through_the_chat_path() {
        let (state, jobs) = app_state(None);
        let actor = std::thread::spawn(move || {
            let job = jobs.recv().unwrap();
            for id in [0u32, 1, 2] {
                job.token_tx
                    .blocking_send(StreamItem::Token(id, None))
                    .unwrap();
            }
            job.token_tx
                .blocking_send(StreamItem::Done(FinishReason::Stop))
                .unwrap();
        });
        let body = r#"{"model": "test", "input": "go", "stream": true, "max_output_tokens": 8}"#;
        let resp = responses(State(state), axum::body::Bytes::from(body)).await;
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        actor.join().unwrap();
        let text = String::from_utf8_lossy(&bytes);
        let events: Vec<&str> = text
            .lines()
            .filter_map(|l| l.strip_prefix("event: "))
            .collect();
        assert_eq!(events.first(), Some(&"response.created"), "{text}");
        assert!(events.contains(&"response.output_text.delta"), "{text}");
        assert_eq!(events.last(), Some(&"response.completed"), "{text}");
        let last = text
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .next_back()
            .unwrap();
        let v: Value = serde_json::from_str(last).unwrap();
        assert_eq!(
            v["response"]["output"][0]["content"][0]["text"], "We need to",
            "{v}"
        );
    }

    /// End to end through the real chat handler, with the actor played by the test.
    #[tokio::test]
    async fn responses_endpoint_answers_through_the_chat_path() {
        let (state, jobs) = app_state(None);
        let actor = std::thread::spawn(move || {
            let job = jobs.recv().unwrap();
            job.token_tx
                .blocking_send(StreamItem::Token(7, None))
                .unwrap();
            job.token_tx
                .blocking_send(StreamItem::Done(FinishReason::Stop))
                .unwrap();
        });
        let body = r#"{"model": "test", "input": "say hi", "max_output_tokens": 8}"#;
        let resp = responses(State(state), axum::body::Bytes::from(body)).await;
        actor.join().unwrap();
        assert!(resp.status().is_success());
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"], "response", "{v}");
        assert_eq!(v["output"][0]["type"], "message", "{v}");
        assert_eq!(v["output"][0]["content"][0]["text"], "Hi", "{v}");
    }
}
