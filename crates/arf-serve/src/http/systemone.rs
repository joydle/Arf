//! System One: `POST /v1/systemone`, the TypeSafe-compatible decision API.
//!
//! A decision model answers typed questions about a `state` without generating anything: each
//! question is rendered with the GGUF's own `tokenizer.chat_template.systemone` template,
//! prefilled once, and one raw score per option is read at the LAST prompt token. The scores
//! become probabilities by a softmax at a temperature stored in the GGUF; the answer is read off
//! those probabilities.
//!
//! **Request:** `state` (string, object or array — anything that is not a string is shown to the
//! model as JSON text), optional `images` (data URLs), and `questions`, an ordered object of id →
//! `{type, instructions, criteria}`:
//! - `choice`: `criteria` is a non-empty object, option key → description (may be `null`);
//! - `score`: `criteria` is an array of 2 to 10 ordered levels, lowest first;
//! - `noul`: a yes/no question; `criteria` is optional, `{"true": …, "false": …}`.
//!
//! **Response:** `answers` in question order — `choice` + `probabilities` + `confidence`;
//! `score` (expected level) + `legend` + `probabilities` + `confidence`; `noul` (probability of
//! true) — and `usage` (`input_tokens` = every question's prompt, `output_tokens` = 0).
//!
//! **Supported:** the openjev family (label-token logits; openjev and its tiny test model are
//! CC-BY-NC-4.0, non-commercial). NOT YET: image input (refused with 501), and the other decision
//! families — lev, nimble, kev, laya / Julia-1, clef — which answer 501 naming their type.
//!
//! PORTED SOURCE: the request parsing, the prompt inputs, the temperature lookup and the answer
//! math are a port of llama.cpp's `tools/server/server-decision.cpp` (MIT; System One landed in
//! a4cb4c61f, ported as of 050439614), the endpoint of `post_systemone` in
//! `tools/server/server-context.cpp`. Parity is gated against
//! llama.cpp's own server on the same GGUF and requests: `scripts/probes/systemone_parity.py`
//! records and diffs; `tests/systemone_parity.rs` runs the recorded set through this module and
//! Arf's CPU reference forward.
//!
//! **The forward pass** is an ordinary actor [`Job`](crate::actor::Job): greedy, `max_tokens`
//! 1, `want_logprobs` with `top_logprobs` = [`ALL_LOGPROBS`](crate::actor::ALL_LOGPROBS) — the whole logprob row of the
//! prefill's last token, in vocabulary order. A label's logprob differs from its logit by the
//! row's logsumexp, a constant the softmax cancels, so the probabilities are the logits'. The job
//! finishes on its one token and its stream is released like any other; nothing in the backend
//! or the decode path knows it is a decision.

use std::collections::HashMap;
use std::path::Path;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use minijinja::value::{Value, ValueKind};
use minijinja::Environment;

use super::{context_budget, context_length_error, submit_job, AppState};
use crate::actor::{StreamItem, ALL_LOGPROBS};
use arf_core::sampling::SamplingParams;
use arf_core::Tokenizer;

/// openjev's option labels, in order: one letter per option, each a single token.
const OPENJEV_LETTERS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
/// llama.cpp `DECISION_MAX_IMAGES`.
const MAX_IMAGES: usize = 8;
const TEMPLATE: &str = "systemone";

/// What the loaded model offers this endpoint. Decided once at startup.
#[derive(Default)]
pub enum DecisionSupport {
    /// No `{arch}.decision.type` in the GGUF: not a decision model (501).
    #[default]
    None,
    /// A decision model this server cannot serve yet, or one whose GGUF is unusable (501, with
    /// the reason).
    Unsupported(String),
    Ready(Box<DecisionModel>),
}

impl DecisionSupport {
    /// Read the GGUF at `path` (and resolve the labels with the server's tokenizer).
    pub fn load(path: &Path, tok: &Tokenizer) -> Self {
        match arf_core::model::gguf::LazyGguf::open_raw(path) {
            Ok(g) => match DecisionModel::from_gguf(&g, tok) {
                Ok(Some(m)) => DecisionSupport::Ready(Box::new(m)),
                Ok(None) => DecisionSupport::None,
                Err(e) => DecisionSupport::Unsupported(e),
            },
            Err(_) => DecisionSupport::None,
        }
    }
}

/// A decision model's template, temperatures and label tokens (llama.cpp
/// `server_decision_context::init`, openjev branch).
pub struct DecisionModel {
    env: Environment<'static>,
    /// `{arch}.decision.temperature.<name>` → value, keyed by `<name>` (`choice`, `score.3_5`…).
    temperatures: HashMap<String, f32>,
    /// One token id per option position.
    labels: Vec<u32>,
}

/// A question type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuestionType {
    Choice,
    Score,
    Noul,
}

impl QuestionType {
    fn name(self) -> &'static str {
        match self {
            QuestionType::Choice => "choice",
            QuestionType::Score => "score",
            QuestionType::Noul => "noul",
        }
    }
}

/// One parsed question. `options` are (key, description) in the order the model is shown them.
#[derive(Clone, Debug)]
pub struct Question {
    pub id: String,
    pub kind: QuestionType,
    pub instructions: Value,
    pub options: Vec<(String, Value)>,
}

/// A parsed request.
#[derive(Debug)]
pub struct DecisionRequest {
    pub state: Value,
    pub questions: Vec<Question>,
    /// Images in `images` and in chat-message parts of `state`. Not supported yet: a request with
    /// any is refused with 501 after it has validated.
    pub n_images: usize,
}

/// An error with its HTTP status, in llama.cpp's shape (`{"error": {code, message, type}}`).
#[derive(Debug, PartialEq, Eq)]
pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
}

impl ApiError {
    fn bad(message: impl Into<String>) -> Self {
        ApiError {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }
    fn not_supported(message: impl Into<String>) -> Self {
        ApiError {
            status: StatusCode::NOT_IMPLEMENTED,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let kind = match self.status {
            StatusCode::BAD_REQUEST => "invalid_request_error",
            StatusCode::NOT_IMPLEMENTED => "not_supported_error",
            _ => "server_error",
        };
        let body = serde_json::json!({ "error": {
            "code": self.status.as_u16(),
            "message": self.message,
            "type": kind,
        }});
        (self.status, axum::Json(body)).into_response()
    }
}

impl DecisionModel {
    /// `Ok(None)`: not a decision model. `Err`: a decision model this server cannot serve.
    pub fn from_gguf(
        g: &arf_core::model::gguf::LazyGguf,
        tok: &Tokenizer,
    ) -> Result<Option<Self>, String> {
        let Some(arch) = g.get_metadata_string("general.architecture") else {
            return Ok(None);
        };
        let prefix = format!("{arch}.decision.");
        let Some(kind) = g.get_metadata_string(&format!("{prefix}type")) else {
            return Ok(None);
        };
        if kind != "openjev" {
            return Err(format!(
                "decision model type \"{kind}\" is not supported yet (supported: openjev)"
            ));
        }
        let source = g
            .get_metadata_string("tokenizer.chat_template.systemone")
            .ok_or("decision model has no \"systemone\" template")?;

        let prefix_temp = format!("{prefix}temperature.");
        let mut temperatures = HashMap::new();
        for key in g.metadata_keys() {
            let Some(name) = key.strip_prefix(&prefix_temp) else {
                continue;
            };
            let t = g
                .get_metadata_f32(&key)
                .ok_or_else(|| format!("invalid decision temperature: {key} is not a number"))?;
            if t.is_nan() || t <= 0.0 {
                return Err(format!("invalid decision temperature: {key} = {t}"));
            }
            temperatures.insert(name.to_string(), t);
        }

        let mut labels = Vec::new();
        for c in OPENJEV_LETTERS.chars() {
            let ids = tok
                .encode(&c.to_string(), false)
                .map_err(|e| format!("tokenize label: {e}"))?;
            if ids.len() != 1 {
                return Err(format!("decision label '{c}' is not a single token"));
            }
            labels.push(ids[0]);
        }
        Self::new(source, temperatures, labels).map(Some)
    }

    /// Build from parts (the template source, the temperatures by name, the label tokens).
    pub fn new(
        source: String,
        temperatures: HashMap<String, f32>,
        labels: Vec<u32>,
    ) -> Result<Self, String> {
        let mut env = Environment::new();
        env.add_filter("tojson", llama_tojson);
        env.add_template_owned(TEMPLATE, source)
            .map_err(|e| format!("systemone template does not compile: {e}"))?;
        Ok(DecisionModel {
            env,
            temperatures,
            labels,
        })
    }

    /// The most options a question can have (one label each).
    pub fn max_options(&self) -> usize {
        self.labels.len()
    }

    /// The label tokens read for `q`, one per option.
    pub fn labels_for(&self, q: &Question) -> &[u32] {
        &self.labels[..q.options.len()]
    }

    /// The prompt of one question (llama.cpp `server_decision_context::render`). The template is
    /// given raw JSON values; it serializes the ones that are not strings.
    pub fn render(&self, state: &Value, q: &Question) -> Result<String, String> {
        let options: Vec<Value> = q
            .options
            .iter()
            .map(|(key, desc)| {
                Value::from_iter([
                    ("key", Value::from(key.as_str())),
                    ("description", desc.clone()),
                ])
            })
            .collect();
        let ctx = Value::from_iter([
            ("id", Value::from(q.id.as_str())),
            ("type", Value::from(q.kind.name())),
            ("instructions", q.instructions.clone()),
            ("state", state.clone()),
            ("options", Value::from(options)),
            ("images", Value::from(Vec::<Value>::new())),
        ]);
        self.env
            .get_template(TEMPLATE)
            .and_then(|t| t.render(ctx))
            .map_err(|e| format!("systemone template: {e}"))
    }

    /// The softmax temperature of `q` (llama.cpp `get_temperature`, non-lev buckets):
    /// `<type>.<bucket>`, then `<type>`, then 1.0.
    pub fn temperature(&self, q: &Question) -> f32 {
        let n = q.options.len();
        let bucket = match n {
            0..=2 => "2",
            3..=5 => "3_5",
            6..=10 => "6_10",
            _ => "11",
        };
        let ty = q.kind.name();
        self.temperatures
            .get(&format!("{ty}.{bucket}"))
            .or_else(|| self.temperatures.get(ty))
            .copied()
            .unwrap_or(1.0)
    }

    /// Parse and validate a request body (llama.cpp `parse_questions` then `parse_state`, in
    /// that order, so the same request fails with the same message).
    pub fn parse(&self, body: &[u8]) -> Result<DecisionRequest, ApiError> {
        parse_request(body, self.max_options())
    }
}

// ----------------------------------------------------------------------------
// Request parsing.
// ----------------------------------------------------------------------------

/// `v[key]` when `v` is an object holding a non-null value there. llama.cpp treats a missing key
/// and an explicit `null` alike everywhere this is used.
fn field(v: &Value, key: &str) -> Option<Value> {
    if v.kind() != ValueKind::Map {
        return None;
    }
    v.get_item(&Value::from(key))
        .ok()
        .filter(|x| !x.is_undefined() && !x.is_none())
}

/// The entries of an object value, in order.
fn entries(v: &Value) -> Vec<(String, Value)> {
    let keys: Vec<Value> = v.try_iter().map(|i| i.collect()).unwrap_or_default();
    keys.into_iter()
        .map(|k| {
            let item = v.get_item(&k).unwrap_or(Value::UNDEFINED);
            (
                k.as_str().map(str::to_string).unwrap_or(k.to_string()),
                item,
            )
        })
        .collect()
}

fn items(v: &Value) -> Vec<Value> {
    v.try_iter().map(|i| i.collect()).unwrap_or_default()
}

/// Parse a request body against a model with `max_options` labels.
pub fn parse_request(body: &[u8], max_options: usize) -> Result<DecisionRequest, ApiError> {
    // Order-preserving: the questions, the criteria and an object state keep the client's order.
    let body: Value = serde_json::from_slice(body).map_err(|e| ApiError::bad(e.to_string()))?;
    let questions = parse_questions(&body, max_options)?;
    let n_images = count_images(&body)?;
    let state = field(&body, "state").expect("checked by parse_questions");
    Ok(DecisionRequest {
        state,
        questions,
        n_images,
    })
}

fn parse_questions(body: &Value, max_options: usize) -> Result<Vec<Question>, ApiError> {
    if field(body, "state").is_none() {
        return Err(ApiError::bad("\"state\" must be provided"));
    }
    let qs = field(body, "questions")
        .filter(|q| q.kind() == ValueKind::Map && q.len().unwrap_or(0) > 0)
        .ok_or_else(|| ApiError::bad("\"questions\" must be a non-empty object"))?;

    let mut out = Vec::new();
    for (id, q) in entries(&qs) {
        let err = |msg: &str| ApiError::bad(format!("questions.{id}: {msg}"));
        if q.kind() != ValueKind::Map {
            return Err(err("must be an object"));
        }
        let instructions =
            field(&q, "instructions").ok_or_else(|| err("\"instructions\" must be provided"))?;
        let type_name = field(&q, "type")
            .and_then(|t| t.as_str().map(str::to_string))
            .unwrap_or_default();
        let criteria = field(&q, "criteria");

        let (kind, options) = match type_name.as_str() {
            "choice" => {
                let c = criteria
                    .filter(|c| c.kind() == ValueKind::Map && c.len().unwrap_or(0) > 0)
                    .ok_or_else(|| err("\"criteria\" must be a non-empty object"))?;
                (QuestionType::Choice, entries(&c))
            }
            "score" => {
                let levels = criteria
                    .filter(|c| c.kind() == ValueKind::Seq)
                    .map(|c| items(&c))
                    .filter(|l| (2..=10).contains(&l.len()))
                    .ok_or_else(|| err("\"criteria\" must be an array of 2 to 10 levels"))?;
                let options = levels
                    .into_iter()
                    .enumerate()
                    .map(|(i, d)| (i.to_string(), d))
                    .collect();
                (QuestionType::Score, options)
            }
            "noul" => {
                if criteria
                    .as_ref()
                    .is_some_and(|c| c.kind() != ValueKind::Map)
                {
                    return Err(err("\"criteria\" must be an object"));
                }
                // openjev shows "true" first (llama.cpp `noul_true_first`).
                let desc = |k: &str| {
                    criteria
                        .as_ref()
                        .and_then(|c| c.get_item(&Value::from(k)).ok())
                        .filter(|d| !d.is_undefined())
                        .unwrap_or(Value::from(()))
                };
                let options = vec![
                    ("true".into(), desc("true")),
                    ("false".into(), desc("false")),
                ];
                (QuestionType::Noul, options)
            }
            _ => return Err(err("\"type\" must be one of: choice, score, noul")),
        };
        if options.len() > max_options {
            return Err(err(&format!(
                "too many options ({}), this model supports at most {max_options}",
                options.len()
            )));
        }
        out.push(Question {
            id,
            kind,
            instructions,
            options,
        });
    }
    Ok(out)
}

/// Validate and count the images: the `images` array, then the `image_url` parts of chat
/// messages in `state` (a bare array of messages or `{"messages": [...]}`), as llama.cpp's
/// `parse_state` loads them.
fn count_images(body: &Value) -> Result<usize, ApiError> {
    let mut n = 0usize;
    let mut load = |url: &Value| -> Result<(), ApiError> {
        let ok = url.as_str().filter(|s| s.starts_with("data:image/"));
        let Some(url) = ok else {
            return Err(ApiError::bad(
                "images must be data URLs (data:image/...;base64,...)",
            ));
        };
        if n >= MAX_IMAGES {
            return Err(ApiError::bad(format!(
                "too many images, the maximum is {MAX_IMAGES}"
            )));
        }
        use base64::Engine as _;
        let payload = url.split_once(";base64,").map(|(_, b)| b);
        if payload
            .and_then(|b| base64::engine::general_purpose::STANDARD.decode(b).ok())
            .is_none()
        {
            return Err(ApiError::bad("invalid image data URL"));
        }
        n += 1;
        Ok(())
    };
    if let Some(images) = field(body, "images") {
        if images.kind() != ValueKind::Seq {
            return Err(ApiError::bad("\"images\" must be an array"));
        }
        for url in items(&images) {
            load(&url)?;
        }
    }
    let state = field(body, "state").unwrap_or_default();
    let messages = field(&state, "messages").unwrap_or(state);
    if messages.kind() == ValueKind::Seq {
        for msg in items(&messages) {
            let Some(content) = field(&msg, "content").filter(|c| c.kind() == ValueKind::Seq)
            else {
                continue;
            };
            for part in items(&content) {
                let is_image = field(&part, "type").and_then(|t| t.as_str().map(str::to_string))
                    == Some("image_url".into());
                if let (true, Some(iu)) = (is_image, field(&part, "image_url")) {
                    load(&field(&iu, "url").unwrap_or(iu))?;
                }
            }
        }
    }
    Ok(n)
}

// ----------------------------------------------------------------------------
// `tojson` as llama.cpp's Jinja engine prints it (common/jinja/value.cpp `value_to_json`).
// ----------------------------------------------------------------------------

/// `x | tojson`, byte for byte what llama.cpp's own Jinja engine prints, since llama.cpp is
/// what the parity gate compares against: `", "` / `": "` separators, keys in the client's
/// order, UTF-8 kept as is (no `\uXXXX` beyond control characters) — the same as Python's
/// `json.dumps(ensure_ascii=False)` (`jinja_chat::py_json_dumps`) EXCEPT for floats, which it
/// streams with C++'s default `ostream` format (`%g`, 6 significant digits): `2.0` prints `2`,
/// `1.23456789` prints `1.23457`, `1e-7` prints `1e-07`. Recorded by the parity probe's
/// `floats` case.
fn llama_tojson(value: &Value) -> Value {
    let mut out = String::new();
    write_json(value, &mut out);
    Value::from_safe_string(out)
}

fn write_json(v: &Value, out: &mut String) {
    match v.kind() {
        ValueKind::Undefined | ValueKind::None => out.push_str("null"),
        ValueKind::Bool => out.push_str(if v.is_true() { "true" } else { "false" }),
        ValueKind::Number if v.is_integer() => out.push_str(&v.to_string()),
        ValueKind::Number => out.push_str(&cpp_float(f64::try_from(v.clone()).unwrap_or(0.0))),
        ValueKind::String => {
            crate::jinja_chat::py_json_string(v.as_str().unwrap_or(""), false, out)
        }
        ValueKind::Seq | ValueKind::Iterable => {
            out.push('[');
            for (i, item) in items(v).iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_json(item, out);
            }
            out.push(']');
        }
        ValueKind::Map => {
            out.push('{');
            for (i, (k, item)) in entries(v).iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                crate::jinja_chat::py_json_string(k, false, out);
                out.push_str(": ");
                write_json(item, out);
            }
            out.push('}');
        }
        _ => out.push_str("null"),
    }
}

/// C++ `std::ostream << double` with default flags: `printf("%g")` at precision 6.
pub fn cpp_float(f: f64) -> String {
    if f.is_nan() {
        return if f.is_sign_negative() { "-nan" } else { "nan" }.into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf" } else { "-inf" }.into();
    }
    const P: i32 = 6;
    // The exponent after rounding to P significant digits decides the style.
    let sci = format!("{:.*e}", (P - 1) as usize, f);
    let (mant, exp) = sci.split_once('e').expect("exponent form");
    let x: i32 = exp.parse().expect("exponent");
    let trim = |s: &str| -> String {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s.to_string()
        }
    };
    if (-4..P).contains(&x) {
        trim(&format!("{:.*}", (P - 1 - x) as usize, f))
    } else {
        format!(
            "{}e{}{:02}",
            trim(mant),
            if x < 0 { '-' } else { '+' },
            x.abs()
        )
    }
}

// ----------------------------------------------------------------------------
// The answer.
// ----------------------------------------------------------------------------

/// TypeSafe's confidence of a choice: how far the top probability is above uniform.
fn confidence_choice(p: &[f64]) -> f64 {
    if p.len() < 2 {
        return 1.0;
    }
    let uniform = 1.0 / p.len() as f64;
    let p_max = p.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    ((p_max - uniform) / (1.0 - uniform)).max(0.0)
}

/// TypeSafe's confidence of a score: the mean distance to the mode, relative to that of a
/// uniform distribution around the centre.
fn confidence_score(p: &[f64]) -> f64 {
    let n = p.len();
    if n < 2 {
        return 1.0;
    }
    let mode = argmax(p);
    let mut dist = 0.0;
    let mut dist_uniform = 0.0;
    for (i, &pi) in p.iter().enumerate() {
        dist += pi * (i as f64 - mode as f64).abs();
        dist_uniform += (i as f64 - (n - 1) as f64 / 2.0).abs() / n as f64;
    }
    (1.0 - dist / dist_uniform).max(0.0)
}

/// The first index of the maximum (C++ `std::max_element`).
fn argmax(p: &[f64]) -> usize {
    let mut best = 0;
    for (i, &v) in p.iter().enumerate() {
        if v > p[best] {
            best = i;
        }
    }
    best
}

/// Softmax of `scores` at `temperature`, in f64 after the f32 max subtraction, as llama.cpp
/// `format_answer` computes it.
pub fn probabilities(scores: &[f32], temperature: f32) -> Vec<f64> {
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f64> = scores
        .iter()
        .map(|&s| ((s - max) as f64 / temperature as f64).exp())
        .collect();
    let sum: f64 = e.iter().sum();
    e.into_iter().map(|x| x / sum).collect()
}

/// One question's answer object (llama.cpp `format_answer`, one variant), keys in llama.cpp's
/// order.
pub fn answer(q: &Question, probs: &[f64]) -> Value {
    let ty = ("type", Value::from(q.kind.name()));
    if q.kind == QuestionType::Noul {
        let p_true = q
            .options
            .iter()
            .position(|(k, _)| k == "true")
            .map(|i| probs[i])
            .unwrap_or(0.0);
        return Value::from_iter([ty, ("noul", Value::from(p_true))]);
    }
    let probabilities = Value::from_iter(
        q.options
            .iter()
            .zip(probs)
            .map(|((k, _), &p)| (k.clone(), Value::from(p))),
    );
    if q.kind == QuestionType::Choice {
        return Value::from_iter([
            ty,
            ("choice", Value::from(q.options[argmax(probs)].0.as_str())),
            ("probabilities", probabilities),
            ("confidence", Value::from(confidence_choice(probs))),
        ]);
    }
    let expected: f64 = probs.iter().enumerate().map(|(i, p)| i as f64 * p).sum();
    let legend = Value::from_iter(q.options.iter().map(|(k, d)| (k.clone(), d.clone())));
    Value::from_iter([
        ty,
        ("score", Value::from(expected)),
        ("legend", legend),
        ("probabilities", probabilities),
        ("confidence", Value::from(confidence_score(probs))),
    ])
}

// ----------------------------------------------------------------------------
// The handler.
// ----------------------------------------------------------------------------

/// `POST /v1/systemone`.
pub(super) async fn systemone(State(state): State<AppState>, body: Bytes) -> Response {
    match run(&state, &body).await {
        Ok(v) => (
            [(header::CONTENT_TYPE, "application/json")],
            serde_json::to_string(&v).unwrap_or_else(|_| "{}".into()),
        )
            .into_response(),
        Err(r) => r,
    }
}

async fn run(state: &AppState, body: &[u8]) -> Result<Value, Response> {
    let model = match &*state.systemone {
        DecisionSupport::Ready(m) => m,
        DecisionSupport::None => {
            return Err(
                ApiError::not_supported("This model is not a decision model").into_response(),
            )
        }
        DecisionSupport::Unsupported(why) => {
            return Err(ApiError::not_supported(why.clone()).into_response())
        }
    };
    let req = model.parse(body).map_err(IntoResponse::into_response)?;
    if req.n_images > 0 {
        return Err(ApiError::not_supported(
            "This server does not support image input for decisions yet",
        )
        .into_response());
    }

    // Render and tokenize every question first: a bad prompt fails the request before any work
    // is queued. Tokenized as llama.cpp does: no BOS, special tokens parsed.
    let mut prompts = Vec::with_capacity(req.questions.len());
    for q in &req.questions {
        let text = model.render(&req.state, q).map_err(server_error)?;
        let ids = state
            .tokenizer
            .encode(&text, false)
            .map_err(|e| server_error(format!("tokenize: {e}")))?;
        if ids.is_empty() {
            return Err(server_error("empty decision prompt".into()));
        }
        context_budget(ids.len(), Some(1), state.max_tokens_cap)
            .map_err(|m| context_length_error(&m, "state"))?;
        prompts.push(ids);
    }
    let input_tokens: usize = prompts.iter().map(Vec::len).sum();

    // One score-only job per question, all queued before any is awaited so the actor can batch
    // their prefills.
    let mut rxs = Vec::with_capacity(prompts.len());
    for ids in prompts {
        let rx = submit_job(
            state,
            ids,
            1,
            SamplingParams::default(),
            true,
            ALL_LOGPROBS,
            None,
            Vec::new(),
            Vec::new(),
            None,
            Default::default(),
            &Default::default(),
        )
        .map_err(|(status, msg)| super::error_response(status, &msg))?;
        rxs.push(rx);
    }

    let mut answers = Vec::with_capacity(req.questions.len());
    for (q, mut rx) in req.questions.iter().zip(rxs) {
        let mut row = None;
        while let Some(item) = rx.recv().await {
            match item {
                StreamItem::Token(_, lp) => row = lp,
                StreamItem::Done(_) => break,
            }
        }
        let row = row.ok_or_else(|| server_error("the model returned no logits".into()))?;
        let scores = model
            .labels_for(q)
            .iter()
            .map(|&t| row.top_logprobs.get(t as usize).map(|&(_, lp)| lp))
            .collect::<Option<Vec<f32>>>()
            .ok_or_else(|| server_error("a label token is outside the logits row".into()))?;
        let probs = probabilities(&scores, model.temperature(q));
        answers.push((q.id.clone(), answer(q, &probs)));
    }

    Ok(Value::from_iter([
        ("model", Value::from(state.model_id.as_str())),
        ("answers", Value::from_iter(answers)),
        (
            "usage",
            Value::from_iter([
                ("input_tokens", Value::from(input_tokens)),
                ("output_tokens", Value::from(0)),
            ]),
        ),
    ]))
}

fn server_error(message: String) -> Response {
    ApiError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message,
    }
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(kind: QuestionType, n: usize) -> Question {
        Question {
            id: "q".into(),
            kind,
            instructions: Value::from("x"),
            options: (0..n).map(|i| (i.to_string(), Value::from(()))).collect(),
        }
    }

    fn model(temps: &[(&str, f32)]) -> DecisionModel {
        let temperatures = temps.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        DecisionModel::new("{{ state }}".into(), temperatures, (0..52).collect()).unwrap()
    }

    fn parse(v: serde_json::Value) -> Result<DecisionRequest, ApiError> {
        parse_request(v.to_string().as_bytes(), 52)
    }

    fn err(v: serde_json::Value) -> String {
        let e = parse(v).unwrap_err();
        assert_eq!(e.status, StatusCode::BAD_REQUEST);
        e.message
    }

    #[test]
    fn parses_in_the_clients_order() {
        let body = r#"{"state": {"z": 1, "a": 2}, "questions": {
            "route": {"type": "choice", "instructions": "Which?", "criteria": {"zeta": "z", "alpha": null}},
            "urgency": {"type": "score", "instructions": "How?", "criteria": ["low", "high"]},
            "angry": {"type": "noul", "instructions": "Angry?"},
            "paid": {"type": "noul", "instructions": "Paid?", "criteria": {"false": "no", "true": "yes"}}}}"#;
        let r = parse_request(body.as_bytes(), 52).unwrap();
        let ids: Vec<&str> = r.questions.iter().map(|q| q.id.as_str()).collect();
        assert_eq!(ids, ["route", "urgency", "angry", "paid"]);
        let keys = |q: &Question| q.options.iter().map(|o| o.0.clone()).collect::<Vec<_>>();
        assert_eq!(keys(&r.questions[0]), ["zeta", "alpha"]);
        assert_eq!(keys(&r.questions[1]), ["0", "1"]);
        // openjev: true first, whatever order the criteria came in
        assert_eq!(keys(&r.questions[2]), ["true", "false"]);
        assert_eq!(keys(&r.questions[3]), ["true", "false"]);
        assert!(r.questions[2].options[0].1.is_none());
        assert_eq!(r.questions[3].options[0].1.as_str(), Some("yes"));
        assert_eq!(llama_tojson(&r.state).as_str(), Some(r#"{"z": 1, "a": 2}"#));
        assert_eq!(r.n_images, 0);
    }

    /// Every 400 of llama.cpp's `test_systemone_invalid_request`, plus the ones the parity
    /// probe records, with llama.cpp's messages.
    #[test]
    fn invalid_requests_are_400_with_llama_cpp_messages() {
        use serde_json::json;
        let s = "a state";
        assert_eq!(err(json!({"questions": {}})), "\"state\" must be provided");
        assert_eq!(
            err(json!({"state": null, "questions": {}})),
            "\"state\" must be provided"
        );
        assert_eq!(err(json!([1, 2])), "\"state\" must be provided");
        let nq = "\"questions\" must be a non-empty object";
        assert_eq!(err(json!({"state": s})), nq);
        assert_eq!(err(json!({"state": s, "questions": {}})), nq);
        assert_eq!(err(json!({"state": s, "questions": [1]})), nq);
        let one = |q: serde_json::Value| err(json!({"state": s, "questions": {"q": q}}));
        assert_eq!(one(json!("x")), "questions.q: must be an object");
        assert_eq!(
            one(json!({"type": "noul"})),
            "questions.q: \"instructions\" must be provided"
        );
        assert_eq!(
            one(json!({"type": "unknown", "instructions": "x"})),
            "questions.q: \"type\" must be one of: choice, score, noul"
        );
        assert_eq!(
            one(json!({"type": 3, "instructions": "x"})),
            "questions.q: \"type\" must be one of: choice, score, noul"
        );
        let ne = "questions.q: \"criteria\" must be a non-empty object";
        assert_eq!(one(json!({"type": "choice", "instructions": "x"})), ne);
        assert_eq!(
            one(json!({"type": "choice", "instructions": "x", "criteria": {}})),
            ne
        );
        assert_eq!(
            one(json!({"type": "choice", "instructions": "x", "criteria": ["a"]})),
            ne
        );
        let lv = "questions.q: \"criteria\" must be an array of 2 to 10 levels";
        assert_eq!(
            one(json!({"type": "score", "instructions": "x", "criteria": ["a"]})),
            lv
        );
        let eleven: Vec<String> = (0..11).map(|i| i.to_string()).collect();
        assert_eq!(
            one(json!({"type": "score", "instructions": "x", "criteria": eleven})),
            lv
        );
        assert_eq!(
            one(json!({"type": "noul", "instructions": "x", "criteria": ["no", "yes"]})),
            "questions.q: \"criteria\" must be an object"
        );
        let many: serde_json::Map<String, serde_json::Value> =
            (0..53).map(|i| (format!("o{i}"), json!(null))).collect();
        assert_eq!(
            one(json!({"type": "choice", "instructions": "x", "criteria": many})),
            "questions.q: too many options (53), this model supports at most 52"
        );
        assert!(parse_request(b"{\"state\": ", 52).is_err());
    }

    #[test]
    fn images_are_validated_then_counted() {
        use serde_json::json;
        let qs = json!({"q": {"type": "noul", "instructions": "x"}});
        let png = "data:image/png;base64,iVBORw0KGgo=";
        assert_eq!(
            err(json!({"state": "s", "questions": qs, "images": png})),
            "\"images\" must be an array"
        );
        assert_eq!(
            err(json!({"state": "s", "questions": qs, "images": ["https://x/y.png"]})),
            "images must be data URLs (data:image/...;base64,...)"
        );
        assert_eq!(
            err(json!({"state": "s", "questions": qs, "images": vec![png; 9]})),
            "too many images, the maximum is 8"
        );
        assert_eq!(
            parse(json!({"state": "s", "questions": qs, "images": [png, png]}))
                .unwrap()
                .n_images,
            2
        );
        // an image part of a chat message counts too, wrapped in {"messages"} or not
        let msgs = json!([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": png}}, {"type": "text", "text": "hi"}]}]);
        assert_eq!(
            parse(json!({"state": msgs, "questions": qs}))
                .unwrap()
                .n_images,
            1
        );
        let wrapped = json!({"messages": msgs});
        assert_eq!(
            parse(json!({"state": wrapped, "questions": qs}))
                .unwrap()
                .n_images,
            1
        );
        assert_eq!(
            parse(json!({"state": "s", "questions": qs, "images": null}))
                .unwrap()
                .n_images,
            0
        );
    }

    #[test]
    fn temperature_buckets_fall_back_to_the_type_then_one() {
        let m = model(&[
            ("choice", 0.85),
            ("choice.2", 0.5),
            ("choice.6_10", 0.7),
            ("score.11", 2.0),
            ("noul", 1.5),
        ]);
        assert_eq!(m.temperature(&q(QuestionType::Choice, 2)), 0.5);
        assert_eq!(m.temperature(&q(QuestionType::Choice, 1)), 0.5);
        assert_eq!(m.temperature(&q(QuestionType::Choice, 3)), 0.85); // no 3_5 → type
        assert_eq!(m.temperature(&q(QuestionType::Choice, 5)), 0.85);
        assert_eq!(m.temperature(&q(QuestionType::Choice, 6)), 0.7);
        assert_eq!(m.temperature(&q(QuestionType::Choice, 10)), 0.7);
        assert_eq!(m.temperature(&q(QuestionType::Choice, 11)), 0.85);
        assert_eq!(m.temperature(&q(QuestionType::Score, 4)), 1.0); // nothing → 1.0
        assert_eq!(m.temperature(&q(QuestionType::Noul, 2)), 1.5);
    }

    #[test]
    fn answer_math_matches_the_formulas() {
        // softmax at a temperature
        let p = probabilities(&[1.0, 2.0, 3.0], 0.5);
        let e: Vec<f64> = [-4.0f64, -2.0, 0.0].iter().map(|x| x.exp()).collect();
        let s: f64 = e.iter().sum();
        for (a, b) in p.iter().zip(e.iter().map(|x| x / s)) {
            assert!((a - b).abs() < 1e-12);
        }
        // choice: argmax (first on ties), confidence = (p_max - 1/n) / (1 - 1/n)
        let c = answer(&q(QuestionType::Choice, 4), &[0.1, 0.4, 0.4, 0.1]);
        assert_eq!(c.get_attr("choice").unwrap().as_str(), Some("1"));
        let conf = f64::try_from(c.get_attr("confidence").unwrap()).unwrap();
        assert!((conf - (0.4 - 0.25) / 0.75).abs() < 1e-12);
        assert_eq!(confidence_choice(&[0.25; 4]), 0.0);
        assert_eq!(confidence_choice(&[1.0]), 1.0);
        // score: expected level; confidence = 1 - E|i - mode| / mean |i - (n-1)/2|
        let probs = [0.1, 0.2, 0.6, 0.1];
        let s = answer(&q(QuestionType::Score, 4), &probs);
        let score = f64::try_from(s.get_attr("score").unwrap()).unwrap();
        assert!((score - (0.2 + 1.2 + 0.3)).abs() < 1e-12);
        let dist = 0.1 * 2.0 + 0.2 * 1.0 + 0.0 + 0.1 * 1.0;
        let uni = (1.5 + 0.5 + 0.5 + 1.5) / 4.0;
        let conf = f64::try_from(s.get_attr("confidence").unwrap()).unwrap();
        assert!((conf - (1.0 - dist / uni)).abs() < 1e-12);
        assert_eq!(confidence_score(&[0.0, 0.0, 1.0]), 1.0);
        assert_eq!(confidence_score(&[0.5, 0.0, 0.5]), 0.0); // clamped at 0
                                                             // keys in llama.cpp's order
        let keys: Vec<String> = s.try_iter().unwrap().map(|k| k.to_string()).collect();
        assert_eq!(
            keys,
            ["type", "score", "legend", "probabilities", "confidence"]
        );
        // noul: p(true), wherever true sits
        let mut nq = q(QuestionType::Noul, 0);
        nq.options = vec![
            ("true".into(), Value::from(())),
            ("false".into(), Value::from(())),
        ];
        let n = answer(&nq, &[0.3, 0.7]);
        assert_eq!(f64::try_from(n.get_attr("noul").unwrap()).unwrap(), 0.3);
    }

    #[test]
    fn floats_print_like_cpp_ostream() {
        for (f, s) in [
            (0.1, "0.1"),
            (2.0, "2"),
            (1.23456789, "1.23457"),
            (1e-7, "1e-07"),
            (12345678.9, "1.23457e+07"),
            (-0.5, "-0.5"),
            (100000.0, "100000"),
            (1000000.0, "1e+06"),
            (0.0001, "0.0001"),
            (0.00001234, "1.234e-05"),
            (999999.5, "1e+06"),
            (0.0, "0"),
            (1.5, "1.5"),
        ] {
            assert_eq!(cpp_float(f), s, "{f}");
        }
    }

    /// The openjev template as the converter writes it: labels by letter, `tojson` for anything
    /// that is not a string, and the falsy-description cases (`null`, `0`, `""`) print nothing.
    #[test]
    fn renders_the_openjev_template_shape() {
        let src = "{% set letters = 'ABC' %}S={{ state if state is string else state | tojson }}|\
                   {% for o in options %}[{{ letters[loop.index0] }}] {{ o.key }}:\
                   {% if o.description %}{{ o.description if o.description is string else o.description | tojson }}{% endif %};\
                   {% endfor %}";
        let m = DecisionModel::new(src.into(), HashMap::new(), vec![1, 2, 3]).unwrap();
        let body = r#"{"state": {"b": [1, 2.0, "é\n"], "a": null},
            "questions": {"q": {"type": "choice", "instructions": "i",
            "criteria": {"x": {"w": 1.5}, "y": 0, "z": ""}}}}"#;
        let r = parse_request(body.as_bytes(), 3).unwrap();
        assert_eq!(
            m.render(&r.state, &r.questions[0]).unwrap(),
            r#"S={"b": [1, 2, "é\n"], "a": null}|[A] x:{"w": 1.5};[B] y:;[C] z:;"#
        );
    }

    /// The handler end to end with the test harness playing the actor: one score-only job per
    /// question (greedy, one token, the whole logprob row), answers and usage assembled in order.
    #[tokio::test]
    async fn handler_queues_score_only_jobs_and_reads_the_label_rows() {
        let (mut st, jobs) = super::super::tests::app_state(None);
        let src = "{{ instructions }}";
        let m = DecisionModel::new(src.into(), HashMap::new(), vec![0, 1, 2]).unwrap();
        st.systemone = std::sync::Arc::new(DecisionSupport::Ready(Box::new(m)));
        let actor = std::thread::spawn(move || {
            let mut seen = Vec::new();
            for _ in 0..2 {
                let job = jobs.recv().unwrap();
                assert_eq!(job.params.max_tokens, 1);
                assert!(job.params.is_greedy());
                assert!(job.want_logprobs);
                assert_eq!(job.top_logprobs, ALL_LOGPROBS);
                seen.push(job.prompt_ids.len());
                // label 1 is the most likely
                let row: Vec<(u32, f32)> = (0..5u32)
                    .map(|i| (i, if i == 1 { -0.1 } else { -3.0 }))
                    .collect();
                let lp = crate::actor::TokenLogprob {
                    token_id: 1,
                    logprob: -0.1,
                    top_logprobs: row,
                };
                job.token_tx
                    .try_send(StreamItem::Token(1, Some(Box::new(lp))))
                    .unwrap();
                job.token_tx
                    .try_send(StreamItem::Done(crate::actor::FinishReason::Length))
                    .unwrap();
            }
            seen
        });
        let body = r#"{"state": "s", "questions": {
            "b": {"type": "choice", "instructions": "pick", "criteria": {"x": null, "y": null, "z": null}},
            "a": {"type": "noul", "instructions": "yes or no"}}}"#;
        let resp = systemone(State(st), Bytes::from(body)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        let seen = actor.join().unwrap();
        // answers in question order, not sorted
        assert!(
            text.find("\"b\"").unwrap() < text.find("\"a\"").unwrap(),
            "{text}"
        );
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["answers"]["b"]["choice"], "y");
        assert_eq!(v["usage"]["input_tokens"], seen.iter().sum::<usize>());
        assert_eq!(v["usage"]["output_tokens"], 0);
        // noul with no temperature key: p(true) = label A (score -3.0) vs B (-0.1) at T = 1
        let p_true = 1.0 / (1.0 + (2.9f64).exp());
        assert!((v["answers"]["a"]["noul"].as_f64().unwrap() - p_true).abs() < 1e-6);
    }

    #[tokio::test]
    async fn a_model_without_decision_metadata_is_501() {
        let (st, _jobs) = super::super::tests::app_state(None);
        let resp = systemone(State(st), Bytes::from_static(b"{}")).await;
        assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
    }

    #[tokio::test]
    async fn images_are_501_after_they_validate() {
        let (mut st, _jobs) = super::super::tests::app_state(None);
        let m = DecisionModel::new("x".into(), HashMap::new(), vec![0, 1]).unwrap();
        st.systemone = std::sync::Arc::new(DecisionSupport::Ready(Box::new(m)));
        let body = r#"{"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}},
            "images": ["data:image/png;base64,iVBORw0KGgo="]}"#;
        let resp = systemone(State(st.clone()), Bytes::from(body)).await;
        assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
        let bad = r#"{"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}},
            "images": ["https://example.com/a.png"]}"#;
        let resp = systemone(State(st), Bytes::from(bad)).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
}
