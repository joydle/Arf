//! Tool calls out of a model's reply, in the format its own chat template taught
//! ([`ToolFormat`]), plus the splitter the streaming path uses to send reasoning and prose live
//! while the call itself is buffered.
//!
//! The fenced-JSON fallback (gemma-4's protocol, `chat::parse_tool_calls`) lives in `chat.rs`
//! beside the prompt that asks for it.

use crate::http::{Tool, ToolCall, ToolCallFunction};
use crate::jinja_chat::{parse_ordered, to_compact_json, ToolFormat};

/// A fresh, process-unique tool-call id (`call_` + 24 hex digits). Clients pair each result to
/// its call by this id, so two calls in one reply — or the same function called on two turns —
/// must never share one. The previous `call_{name}` did exactly that.
pub fn new_call_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    // splitmix64 of the clock, then the counter in the low digits: unique within the process
    // (the counter) and unlikely to repeat across restarts (the clock).
    let mut z = nanos.wrapping_add(seq.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    format!("call_{z:016x}{:08x}", seq as u32)
}

/// Build one OpenAI-shaped call with a fresh id.
pub fn make_call(name: &str, arguments: String) -> ToolCall {
    ToolCall {
        id: new_call_id(),
        r#type: "function".into(),
        function: ToolCallFunction {
            name: name.to_string(),
            arguments,
        },
    }
}

/// The text that opens a native tool call. `<function=` is included for the XML format: the
/// model sometimes drops the `<tool_call>` wrapper, and the parser accepts it bare.
pub fn call_markers(format: ToolFormat) -> &'static [&'static str] {
    match format {
        ToolFormat::QwenXml => &["<tool_call>", "<function="],
        ToolFormat::HermesJson => &["<tool_call>"],
        ToolFormat::None => &[],
    }
}

/// The byte offset of the earliest of `pats` in `s`.
fn earliest(s: &str, pats: &[&str]) -> Option<usize> {
    pats.iter().filter_map(|p| s.find(p)).min()
}

const TC_OPEN: &str = "<tool_call>";
const TC_CLOSE: &str = "</tool_call>";
const FN_OPEN: &str = "<function=";
const FN_CLOSE: &str = "</function>";
const P_OPEN: &str = "<parameter=";
const P_CLOSE: &str = "</parameter>";

/// True when `s`, after leading whitespace, starts with one of `tags`.
fn next_is(s: &str, tags: &[&str]) -> bool {
    let t = s.trim_start();
    tags.iter().any(|m| t.starts_with(m))
}

/// Qwen3-Coder / Qwen3.8 XML:
///
/// ```text
/// <tool_call>
/// <function=edit>
/// <parameter=path>
/// calc.py
/// </parameter>
/// <parameter=newText>
/// def add(a, b):
///     return a + b
/// </parameter>
/// </function>
/// </tool_call>
/// ```
///
/// Every `<function=…>` block is a call — lenient: a call cut off mid-value is still returned.
/// [`settle_native`] is what the handlers use; it drops a call the token budget cut off. See
/// `qwen_xml_calls` (private — not linked from this public fn) for the rules.
pub fn parse_qwen_xml(reply: &str, tools: &[Tool]) -> Option<(Vec<ToolCall>, String)> {
    qwen_xml_calls(reply, tools)
        .map(|(calls, lead)| (calls.into_iter().map(|(c, _)| c).collect(), lead))
}

/// The XML calls in `reply`, each with whether its block was CLOSED, and the prose before the
/// first one (trimmed).
///
/// A parameter value is the raw text between its tags with exactly one leading and one trailing
/// newline removed (the template writes `>\n` + value + `\n</`); code with its own blank lines,
/// quotes and braces survives untouched. When the tool's JSON schema says the parameter is a
/// string, that text IS the value. Otherwise it is JSON-parsed (`5`, `true`, `["a"]`, `{…}`), then
/// read as a Python literal (Qwen3-Coder's own template replays lists and dicts in Python repr,
/// so the model writes them back that way), falling back to the string. `arguments` keeps the
/// model's parameter order.
///
/// Where a value ENDS (2026-09-26 review): at the first `</parameter>` in STRUCTURAL position —
/// followed, after whitespace, by `<parameter=`, `</function>`, `</tool_call>` or the end of the
/// text — else at the first `</parameter>` at all; never past the start of the NEXT call. The
/// first version took the first `</parameter>` outright, so an `edit` whose `newText` contained
/// that literal text (an agent editing this very parser) was cut there and the truncated edit
/// was still returned as a complete call; and a value missing its `</parameter>` ran on into the
/// next call's. A missing `</parameter>` ends the value at the next `<parameter=` or
/// `</function>` (a sloppy model), or at the end of the text.
///
/// A header counts only when what follows it is a parameter, a close, or the end of the text:
/// prose that mentions "`<function=NAME>` blocks" is not a call to `NAME`.
///
/// CLOSED means the block ended on `</function>` or `</tool_call>` — or on the next call, the
/// model having moved on. A reply cut off by `max_tokens` inside a value is NOT closed: its last
/// value is a prefix of what the model meant, and a `write` or `edit` executed with it truncates
/// a file in the user's repository. The first version returned such a call as finished, with
/// finish_reason `tool_calls`, and both agents run a call the moment its arguments parse.
pub(crate) fn qwen_xml_calls(
    reply: &str,
    tools: &[Tool],
) -> Option<(Vec<(ToolCall, bool)>, String)> {
    let mut calls = Vec::new();
    let mut lead_end = None;
    let mut pos = 0;
    while let Some(i) = reply[pos..].find(FN_OPEN) {
        let hdr = pos + i;
        let name_start = hdr + FN_OPEN.len();
        let Some(gt) = reply[name_start..].find('>') else {
            break;
        };
        let name = reply[name_start..name_start + gt].trim();
        let mut body = &reply[name_start + gt + 1..];
        // A function name never spans lines; a `>` further away means this was not a header.
        if name.is_empty()
            || name.contains('\n')
            || !(body.trim().is_empty() || next_is(body, &[P_OPEN, FN_CLOSE, TC_CLOSE]))
        {
            pos = name_start;
            continue;
        }
        let mut args: Vec<(String, String)> = Vec::new();
        loop {
            let b = body.trim_start();
            let Some(p) = b.strip_prefix(P_OPEN) else {
                body = b;
                break;
            };
            let Some(gt) = p.find('>') else {
                body = b;
                break;
            };
            let pname = p[..gt].trim().to_string();
            let vstart = &p[gt + 1..];
            let (e, resume) = param_value_end(vstart);
            let (raw, next) = (&vstart[..e], &vstart[resume..]);
            let v = raw.strip_prefix('\n').unwrap_or(raw);
            let v = v.strip_suffix('\n').unwrap_or(v);
            let json = if param_is_string(tools, name, &pname) {
                serde_json::to_string(v).unwrap_or_else(|_| "\"\"".into())
            } else {
                non_string_value(v)
            };
            args.push((pname, json));
            body = next;
        }
        let closed = next_is(body, &[FN_CLOSE, TC_CLOSE, TC_OPEN, FN_OPEN]);
        let mut arguments = String::from("{");
        for (i, (k, v)) in args.iter().enumerate() {
            if i > 0 {
                arguments.push(',');
            }
            arguments.push_str(&serde_json::to_string(k).unwrap_or_else(|_| "\"\"".into()));
            arguments.push(':');
            arguments.push_str(v);
        }
        arguments.push('}');
        // The prose ends where this call's `<tool_call>` wrapper (if any) begins.
        lead_end.get_or_insert_with(|| {
            let before = reply[..hdr].trim_end();
            before.strip_suffix(TC_OPEN).map_or(hdr, str::len)
        });
        calls.push((make_call(name, arguments), closed));
        pos = reply.len() - body.len();
    }
    let lead_end = lead_end?;
    Some((calls, reply[..lead_end].trim().to_string()))
}

/// Where the parameter value `v` ends, and where the text after it resumes. See
/// [`qwen_xml_calls`]: the first `</parameter>` in structural position, else the first at all —
/// searched only up to the next call (`</function>` then, past an optional `</tool_call>`, a
/// `<tool_call>` or `<function=`); with none, the next `<parameter=` or `</function>`.
fn param_value_end(v: &str) -> (usize, usize) {
    let window = next_call_boundary(v).unwrap_or(v.len());
    let w = &v[..window];
    let mut first = None;
    let mut from = 0;
    while let Some(off) = w[from..].find(P_CLOSE) {
        let e = from + off;
        first.get_or_insert(e);
        let after = &v[e + P_CLOSE.len()..];
        if after.trim().is_empty() || next_is(after, &[P_OPEN, FN_CLOSE, TC_CLOSE]) {
            return (e, e + P_CLOSE.len());
        }
        from = e + P_CLOSE.len();
    }
    match first {
        Some(e) => (e, e + P_CLOSE.len()),
        None => {
            let e = earliest(w, &[P_OPEN, FN_CLOSE]).unwrap_or(window);
            (e, e)
        }
    }
}

/// The offset of a `</function>` that is followed by another call — the end of this call.
fn next_call_boundary(v: &str) -> Option<usize> {
    let mut from = 0;
    while let Some(off) = v[from..].find(FN_CLOSE) {
        let at = from + off;
        let after = v[at + FN_CLOSE.len()..].trim_start();
        let after = after.strip_prefix(TC_CLOSE).unwrap_or(after);
        if next_is(after, &[TC_OPEN, FN_OPEN]) {
            return Some(at);
        }
        from = at + FN_CLOSE.len();
    }
    None
}

/// A parameter the schema does not declare a string: JSON if it parses (compact, key order kept);
/// a list or dict in Python literal form as that JSON; a Python-style `True`/`False`/`None`;
/// else the text as a string.
///
/// The Python form is not hypothetical: Qwen3-Coder's template replays a non-string argument
/// with `| string`, i.e. Python `str()` — `[{'content': 'fix', 'status': 'pending'}]` — so that
/// is the shape the model has seen and may write back. vLLM's qwen3_coder parser falls back to
/// `ast.literal_eval` for the same reason. Without this, OpenCode's `todowrite` would receive
/// `todos` as a STRING, which its schema rejects (expected array) — reasoned from the review,
/// not yet seen in a live session.
fn non_string_value(v: &str) -> String {
    let t = v.trim();
    if let Some(parsed) = parse_ordered(t) {
        return to_compact_json(&parsed);
    }
    if t.starts_with('[') || t.starts_with('{') {
        if let Some(json) = python_literal_to_json(t) {
            return json;
        }
    }
    match t {
        "True" => "true".into(),
        "False" => "false".into(),
        "None" => "null".into(),
        _ => serde_json::to_string(v).unwrap_or_else(|_| "\"\"".into()),
    }
}

/// A Python literal of lists, dicts, strings, numbers, `True`/`False`/`None` as compact JSON, or
/// `None` when it is not one. Single- or double-quoted strings with Python's escapes, and a
/// trailing comma before `]` / `}`. Anything else — a name, a call, `inf` — is not a literal.
fn python_literal_to_json(s: &str) -> Option<String> {
    let cs: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len() + 8);
    let mut i = 0;
    while i < cs.len() {
        let c = cs[i];
        match c {
            '\'' | '"' => {
                let (text, next) = python_string(&cs, i)?;
                out.push_str(&serde_json::to_string(&text).ok()?);
                i = next;
                continue;
            }
            ',' => {
                let mut j = i + 1;
                while j < cs.len() && cs[j].is_whitespace() {
                    j += 1;
                }
                if !matches!(cs.get(j), Some(']' | '}')) {
                    out.push(',');
                }
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                // An exponent inside a number (`1e-05`) is not a name.
                if out.ends_with(|p: char| p.is_ascii_digit() || p == '.') {
                    out.push(c);
                } else {
                    let mut j = i;
                    while j < cs.len() && (cs[j].is_ascii_alphanumeric() || cs[j] == '_') {
                        j += 1;
                    }
                    let word: String = cs[i..j].iter().collect();
                    out.push_str(match word.as_str() {
                        "True" => "true",
                        "False" => "false",
                        "None" => "null",
                        _ => return None,
                    });
                    i = j;
                    continue;
                }
            }
            c => out.push(c),
        }
        i += 1;
    }
    parse_ordered(&out).map(|v| to_compact_json(&v))
}

/// The Python string literal opening at `cs[start]` (a quote): its value and the index after the
/// closing quote. `None` when it never closes.
fn python_string(cs: &[char], start: usize) -> Option<(String, usize)> {
    let q = cs[start];
    let mut out = String::new();
    let mut i = start + 1;
    let hex = |from: usize, n: usize| -> Option<char> {
        let digits: String = cs.get(from..from + n)?.iter().collect();
        char::from_u32(u32::from_str_radix(&digits, 16).ok()?)
    };
    while i < cs.len() {
        let c = cs[i];
        if c == q {
            return Some((out, i + 1));
        }
        if c != '\\' {
            out.push(c);
            i += 1;
            continue;
        }
        let e = *cs.get(i + 1)?;
        i += 2;
        match e {
            'n' => out.push('\n'),
            't' => out.push('\t'),
            'r' => out.push('\r'),
            'b' => out.push('\u{08}'),
            'f' => out.push('\u{0c}'),
            'v' => out.push('\u{0b}'),
            'a' => out.push('\u{07}'),
            '\\' | '\'' | '"' => out.push(e),
            '\n' => {} // a line continuation
            'x' => {
                out.push(hex(i, 2)?);
                i += 2;
            }
            'u' => {
                out.push(hex(i, 4)?);
                i += 4;
            }
            'U' => {
                out.push(hex(i, 8)?);
                i += 8;
            }
            '0'..='7' => {
                let mut n = e.to_digit(8)?;
                let mut k = 0;
                while k < 2 && cs.get(i).is_some_and(|d| d.is_digit(8)) {
                    n = n * 8 + cs[i].to_digit(8)?;
                    i += 1;
                    k += 1;
                }
                out.push(char::from_u32(n)?);
            }
            // Python keeps an unknown escape as written, backslash included.
            other => {
                out.push('\\');
                out.push(other);
            }
        }
    }
    None
}

/// True when `tools` declares `function.parameter` as a string: `"type": "string"`, a type list
/// containing it (`["string", "null"]`), or an `anyOf` / `oneOf` member that is one (Pydantic's
/// `Optional[str]`). Unknown tool or parameter → false (the value is JSON-parsed, then kept as
/// text if it is not JSON).
fn param_is_string(tools: &[Tool], function: &str, param: &str) -> bool {
    let Some(schema) = tools
        .iter()
        .find(|t| t.function.name == function)
        .and_then(|t| t.function.parameters.as_ref())
        .and_then(|p| p.get("properties"))
        .and_then(|p| p.get(param))
    else {
        return false;
    };
    fn is_string(s: &serde_json::Value) -> bool {
        match s.get("type") {
            Some(serde_json::Value::String(t)) => t == "string",
            Some(serde_json::Value::Array(ts)) => ts.iter().any(|t| t == "string"),
            _ => ["anyOf", "oneOf"].iter().any(|k| {
                s.get(k)
                    .and_then(|v| v.as_array())
                    .is_some_and(|alts| alts.iter().any(is_string))
            }),
        }
    }
    is_string(schema)
}

/// Hermes / Qwen2.5 / Qwen3 JSON: every `<tool_call>{"name": …, "arguments": {…}}</tool_call>`.
/// A block whose JSON does not parse (even after `chat::json_call_from_object`'s raw-newline
/// repair) is skipped; `None` when no block parsed. `lead` is the prose before the first call.
///
/// The object is found by BALANCING it from the `{` right after the tag — which skips braces and
/// tags inside JSON strings — and only then is the `</tool_call>` after it skipped. The first
/// version cut the block at the first `</tool_call>`, so a call whose argument contained that
/// text (an agent writing a template or this parser) became unbalanced JSON, was dropped, and its
/// raw text went out as content. A `<tool_call>` not followed by `{` is prose mentioning the tag.
pub fn parse_hermes(reply: &str) -> Option<(Vec<ToolCall>, String)> {
    let mut calls = Vec::new();
    let mut lead_end = None;
    let mut pos = 0;
    while let Some(i) = reply[pos..].find(TC_OPEN) {
        let open = pos + i;
        let after = open + TC_OPEN.len();
        let s = reply.len() - reply[after..].trim_start().len();
        let Some(end) = reply[s..]
            .starts_with('{')
            .then(|| crate::chat::balanced_object_end(reply, s))
            .flatten()
        else {
            pos = after;
            continue;
        };
        pos = end + 1;
        if let Some(c) = crate::chat::json_call_from_object(&reply[s..=end], &[]) {
            lead_end.get_or_insert(open);
            calls.push(c);
            let rest = reply[pos..].trim_start();
            if rest.starts_with(TC_CLOSE) {
                pos = reply.len() - rest.len() + TC_CLOSE.len();
            }
        }
    }
    let lead_end = lead_end?;
    Some((calls, reply[..lead_end].trim().to_string()))
}

// ----------------------------------------------------------------------------
// The end of a native reply: calls, or text — the same decision streamed or not.
// ----------------------------------------------------------------------------

/// What the text buffered from the first call marker on (`SplitEnd::buffered`) turned out to be.
#[derive(Debug, Default)]
pub struct NativeEnd {
    /// Text to send as `content` AFTER the prose already sent, the whitespace the splitter held
    /// back before the marker included. Empty: nothing more.
    pub content: String,
    /// The complete calls, in order. Empty: the reply finishes with the job's own reason.
    pub calls: Vec<ToolCall>,
    /// Something the server log should show once (see [`EndNote`]).
    pub note: Option<EndNote>,
}

/// The buffered-text outcomes worth a once-per-process log line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndNote {
    /// The token budget cut a call off before it closed; it was not sent as a call.
    CutCall,
    /// A call marker with no call after it (`<tool_call>` then EOS); the marker was dropped.
    BareMarker,
    /// A call marker followed by text that is not a call; the text was sent as content.
    Unparsed,
}

/// Lines that are nothing but a call wrapper: dropped from text that did not parse as a call.
fn is_marker_line(line: &str) -> bool {
    matches!(
        line.trim(),
        "<tool_call>" | "</tool_call>" | "<function=" | "<function=>" | "</function>"
    )
}

/// Decide what `buffered` is. `gap` is the whitespace the splitter held back between the prose
/// and the marker ([`SplitEnd::gap`]). `cut`: the job did NOT end on a stop token (it hit
/// `max_tokens`, or was evicted) — so the last call may be a prefix of what the model meant.
///
/// - Calls parsed: send them, and any prose between the buffer start and the first REAL call as
///   content (a prose mention of `<tool_call>` before the call opened the buffer early; that
///   text used to be dropped). When `cut` and the last call never closed, it is DROPPED; if no
///   call is left, the buffered text goes back verbatim as content and the finish reason stays
///   the job's (`length`), so the agent sees a truncated turn instead of executing a partial
///   `write`. A reply that ended on EOS keeps the lenient parse (a sloppy model, not a cut one).
/// - Nothing parsed: lines that are only a wrapper (`<tool_call>`, `</tool_call>`, a nameless
///   `<function=`) are dropped — the two stray `<tool_call>` lines of the first OpenCode session
///   were this — and what is left is sent verbatim, after the SAME whitespace it followed.
///   (It used to be `"\n\n" + trimmed`: a prose mention of the tag read "the `\n\n<tool_call>`".)
pub fn settle_native(
    format: ToolFormat,
    gap: &str,
    buffered: &str,
    tools: &[Tool],
    cut: bool,
) -> NativeEnd {
    if buffered.trim().is_empty() {
        return NativeEnd::default();
    }
    let parsed = match format {
        ToolFormat::QwenXml => qwen_xml_calls(buffered, tools),
        ToolFormat::HermesJson => parse_hermes(buffered)
            .map(|(calls, lead)| (calls.into_iter().map(|c| (c, true)).collect(), lead)),
        ToolFormat::None => None,
    };
    let verbatim = |text: &str| format!("{gap}{}", text.trim_end());
    match parsed {
        Some((mut calls, lead)) => {
            let mut note = None;
            if cut && calls.last().is_some_and(|(_, closed)| !closed) {
                calls.pop();
                note = Some(EndNote::CutCall);
                if calls.is_empty() {
                    return NativeEnd {
                        content: verbatim(buffered),
                        calls: Vec::new(),
                        note,
                    };
                }
            }
            NativeEnd {
                content: if lead.is_empty() {
                    String::new()
                } else {
                    verbatim(&lead)
                },
                calls: calls.into_iter().map(|(c, _)| c).collect(),
                note,
            }
        }
        // Cut off before anything parsed: the text as it came, nothing dropped.
        None if cut => NativeEnd {
            content: verbatim(buffered),
            calls: Vec::new(),
            note: None,
        },
        None => {
            let kept: Vec<&str> = buffered.lines().filter(|l| !is_marker_line(l)).collect();
            let kept = kept.join("\n");
            if kept.trim().is_empty() {
                NativeEnd {
                    note: Some(EndNote::BareMarker),
                    ..Default::default()
                }
            } else {
                NativeEnd {
                    content: verbatim(&kept),
                    calls: Vec::new(),
                    note: Some(EndNote::Unparsed),
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// The streaming splitter: reasoning live, prose live, the call buffered.
// ----------------------------------------------------------------------------

/// One piece of a streamed reply, ready to become a `delta`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Piece {
    Reasoning(String),
    Content(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Think,
    Answer,
    Calls,
}

/// Splits a reply, as its text arrives, into reasoning (up to `</think>`), answer prose (up to a
/// tool-call marker) and the call text (the rest, buffered for the parser).
///
/// Never emits a partial marker: a suffix that could still become `</think>` or a call marker is
/// held back until the next text decides it, and so is trailing whitespace (the non-streamed
/// path trims both ends, so a streamed reply must not carry a newline the other would drop).
/// With no markers (`ToolFormat::None`, the fenced fallback) the answer is buffered whole — a
/// fenced call is only recognisable once its JSON closes — while reasoning still streams.
///
/// `detect_think_open`: a reply that itself begins with `<think>` (Qwen3's hybrid template, whose
/// cue does not pre-open it) enters the reasoning phase.
#[derive(Debug)]
pub struct ReplySplitter {
    phase: Phase,
    markers: &'static [&'static str],
    detect_think_open: bool,
    pending: String,
    reasoning_started: bool,
    answer_started: bool,
    /// Fenced mode: the whole answer. Native mode: the call text from the first marker on.
    buffered: String,
    /// Native mode: the whitespace between the prose and the first marker, held back and not
    /// sent — returned by `finish` so text sent after the prose follows it exactly as written.
    gap: String,
}

/// A finished [`ReplySplitter`]: the last pieces to send, and the text it buffered.
#[derive(Debug)]
pub struct SplitEnd {
    pub pieces: Vec<Piece>,
    /// The whitespace between the sent prose and `buffered` (native mode; else empty).
    pub gap: String,
    /// The call text from the first marker on (native), or the whole answer (fenced). Empty
    /// when there is none.
    pub buffered: String,
}

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

/// The length of the tail of `s` to hold back: the longest suffix that is a proper prefix of
/// one of `markers`, extended over the whitespace before it.
fn holdback(s: &str, markers: &[&str]) -> usize {
    let mut k = 0;
    for m in markers {
        for n in (1..m.len()).rev() {
            if n <= s.len() && s.ends_with(&m[..n]) {
                k = k.max(n);
                break;
            }
        }
    }
    // `m` is ASCII, so `s.len() - k` is a char boundary whenever the suffix matched.
    let cut = s.len() - k;
    s.len() - s[..cut].trim_end().len()
}

impl ReplySplitter {
    /// `in_think`: the prompt pre-opened `<think>` (the reply starts inside reasoning).
    pub fn new(format: ToolFormat, in_think: bool, detect_think_open: bool) -> Self {
        ReplySplitter {
            phase: if in_think {
                Phase::Think
            } else {
                Phase::Answer
            },
            markers: call_markers(format),
            detect_think_open,
            pending: String::new(),
            reasoning_started: false,
            answer_started: false,
            buffered: String::new(),
            gap: String::new(),
        }
    }

    fn reasoning(&mut self, s: &str, out: &mut Vec<Piece>) {
        let s = if self.reasoning_started {
            s
        } else {
            s.trim_start()
        };
        if !s.is_empty() {
            self.reasoning_started = true;
            out.push(Piece::Reasoning(s.to_string()));
        }
    }

    fn content(&mut self, s: &str, out: &mut Vec<Piece>) {
        if !s.is_empty() {
            out.push(Piece::Content(s.to_string()));
        }
    }

    /// Feed the next decoded text; returns what can be sent now.
    pub fn push(&mut self, text: &str) -> Vec<Piece> {
        self.pending.push_str(text);
        let mut out = Vec::new();
        loop {
            match self.phase {
                Phase::Think => {
                    if let Some(i) = self.pending.find(THINK_CLOSE) {
                        let r = self.pending[..i].trim_end().to_string();
                        self.reasoning(&r, &mut out);
                        self.pending = self.pending[i + THINK_CLOSE.len()..].to_string();
                        self.phase = Phase::Answer;
                        self.detect_think_open = false;
                        continue;
                    }
                    let keep = holdback(&self.pending, &[THINK_CLOSE]);
                    let emit = self.pending[..self.pending.len() - keep].to_string();
                    self.pending.drain(..emit.len());
                    self.reasoning(&emit, &mut out);
                    break;
                }
                Phase::Answer => {
                    if !self.answer_started {
                        let t = self.pending.trim_start();
                        if t.is_empty() {
                            self.pending.clear();
                            break;
                        }
                        if self.detect_think_open {
                            if let Some(rest) = t.strip_prefix(THINK_OPEN) {
                                self.pending = rest.to_string();
                                self.phase = Phase::Think;
                                self.detect_think_open = false;
                                continue;
                            }
                            if THINK_OPEN.starts_with(t) {
                                self.pending = t.to_string();
                                break; // could still become `<think>`
                            }
                        }
                        self.pending = t.to_string();
                        self.answer_started = true;
                    }
                    if self.markers.is_empty() {
                        self.buffered.push_str(&self.pending);
                        self.pending.clear();
                        break;
                    }
                    if let Some(i) = earliest(&self.pending, self.markers) {
                        let head = &self.pending[..i];
                        let prose = head.trim_end().to_string();
                        self.gap = head[prose.len()..].to_string();
                        self.content(&prose, &mut out);
                        self.buffered = self.pending[i..].to_string();
                        self.pending.clear();
                        self.phase = Phase::Calls;
                        break;
                    }
                    let keep = holdback(&self.pending, self.markers);
                    let emit = self.pending[..self.pending.len() - keep].to_string();
                    self.pending.drain(..emit.len());
                    self.content(&emit, &mut out);
                    break;
                }
                Phase::Calls => {
                    self.buffered.push_str(&self.pending);
                    self.pending.clear();
                    break;
                }
            }
        }
        out
    }

    /// True while the reply is still inside its reasoning (no `</think>` yet). Read before
    /// [`finish`](Self::finish), it says the reply was cut off mid-reasoning — which `/v1/messages`
    /// sends as the thinking block alone, where a reply that reached its answer always ends
    /// with a text or tool_use block.
    pub fn in_reasoning(&self) -> bool {
        self.phase == Phase::Think
    }

    /// The reply is complete: the last pieces, the gap and the buffered text ([`SplitEnd`]).
    pub fn finish(mut self) -> SplitEnd {
        let mut out = Vec::new();
        let rest = std::mem::take(&mut self.pending);
        match self.phase {
            // Cut off before `</think>`: all of it was reasoning (as `split_pretag_think`).
            Phase::Think => self.reasoning(rest.trim_end(), &mut out),
            Phase::Answer if self.markers.is_empty() => self.buffered.push_str(&rest),
            Phase::Answer => {
                let t = if self.answer_started {
                    rest.trim_end()
                } else {
                    rest.trim()
                };
                self.content(t, &mut out);
            }
            Phase::Calls => self.buffered.push_str(&rest),
        }
        SplitEnd {
            pieces: out,
            gap: self.gap,
            buffered: self.buffered,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::ToolFunction;

    fn tools() -> Vec<Tool> {
        let t = |name: &str, params: serde_json::Value| Tool {
            r#type: "function".into(),
            function: ToolFunction {
                name: name.into(),
                description: None,
                parameters: Some(params),
            },
        };
        vec![
            t(
                "edit",
                serde_json::json!({"type": "object", "properties": {
                    "path": {"type": "string"}, "oldText": {"type": "string"},
                    "newText": {"type": "string"}, "replaceAll": {"type": "boolean"},
                    "count": {"type": ["integer", "null"]}}}),
            ),
            t(
                "bash",
                serde_json::json!({"type": "object", "properties": {
                    "command": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "timeout": {"type": "number"}}}),
            ),
        ]
    }

    fn args(c: &ToolCall) -> serde_json::Value {
        serde_json::from_str(&c.function.arguments).expect("arguments are JSON")
    }

    /// A real Qwen3.8-shaped reply: prose (with braces) before the call, then an edit whose
    /// values are multi-line code with quotes, braces and a blank line.
    #[test]
    fn qwen_xml_multiline_code_values_and_lead_prose() {
        let reply = "The f-string needs {name} braces; fixing it.\n\n<tool_call>\n<function=edit>\n\
<parameter=path>\nhello.py\n</parameter>\n<parameter=oldText>\ndef greet(name):\n    return \"Hello, \" + name\n</parameter>\n\
<parameter=newText>\ndef greet(name):\n\n    return f\"Hello, {name}!\"  # \\n stays\n</parameter>\n\
<parameter=replaceAll>\nfalse\n</parameter>\n</function>\n</tool_call>";
        let (calls, lead) = parse_qwen_xml(reply, &tools()).expect("a call");
        assert_eq!(lead, "The f-string needs {name} braces; fixing it.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "edit");
        let a = args(&calls[0]);
        assert_eq!(a["path"], "hello.py");
        assert_eq!(
            a["oldText"],
            "def greet(name):\n    return \"Hello, \" + name"
        );
        assert_eq!(
            a["newText"],
            "def greet(name):\n\n    return f\"Hello, {name}!\"  # \\n stays"
        );
        assert_eq!(a["replaceAll"], false); // boolean per the schema, not "false"
                                            // The model's parameter order survives into the arguments string.
        assert!(calls[0]
            .function
            .arguments
            .starts_with("{\"path\":\"hello.py\",\"oldText\":"));
    }

    #[test]
    fn qwen_xml_two_calls_get_unique_ids() {
        let reply = "<tool_call>\n<function=bash>\n<parameter=command>\npython3 test_calc.py\n</parameter>\n\
<parameter=timeout>\n30\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=bash>\n\
<parameter=command>\n42\n</parameter>\n</function>\n</tool_call>";
        let (calls, lead) = parse_qwen_xml(reply, &tools()).expect("calls");
        assert_eq!(lead, "");
        assert_eq!(calls.len(), 2);
        assert_ne!(calls[0].id, calls[1].id);
        assert!(calls[0].id.starts_with("call_"));
        assert_eq!(
            args(&calls[0]),
            serde_json::json!({"command": "python3 test_calc.py", "timeout": 30})
        );
        // `command` is a string (anyOf) — "42" stays text, not a number.
        assert_eq!(args(&calls[1]), serde_json::json!({"command": "42"}));
    }

    /// Unknown parameters are JSON-parsed then kept as text; missing `</parameter>`s inside a
    /// CLOSED function are tolerated; a bare `<function=` without the wrapper is accepted. A reply
    /// cut off mid-value still parses here (the lenient parser) but is marked not closed — what
    /// `settle_native` drops when the token budget cut it (`cut_call_is_never_sent_as_a_call`).
    #[test]
    fn qwen_xml_lenient_shapes() {
        let reply = "<function=read>\n<parameter=path>\na.rs\n<parameter=limit>\n20\n<parameter=opts>\n{\"b\": 1, \"a\": [1]}\n</function>";
        let (calls, _) = parse_qwen_xml(reply, &[]).expect("a call");
        assert_eq!(calls[0].function.name, "read");
        assert_eq!(
            calls[0].function.arguments,
            r#"{"path":"a.rs","limit":20,"opts":{"b":1,"a":[1]}}"#
        );
        let (calls, _) = qwen_xml_calls(reply, &[]).unwrap();
        assert!(calls[0].1, "closed by </function>");
        let cut = "Reading.\n<tool_call>\n<function=read>\n<parameter=path>\nsrc/main";
        let (calls, lead) = parse_qwen_xml(cut, &[]).expect("a call");
        assert_eq!(lead, "Reading.");
        assert_eq!(args(&calls[0]), serde_json::json!({"path": "src/main"}));
        assert!(
            !qwen_xml_calls(cut, &[]).unwrap().0[0].1,
            "cut mid-value: not closed"
        );
        let no_args = "<tool_call>\n<function=status>\n</function>\n</tool_call>";
        assert_eq!(
            parse_qwen_xml(no_args, &[]).unwrap().0[0]
                .function
                .arguments,
            "{}"
        );
    }

    #[test]
    fn qwen_xml_plain_reply_is_none() {
        assert!(parse_qwen_xml("The fix is done: add() now returns a + b.", &tools()).is_none());
        // Prose that MENTIONS a header is not a call to it.
        assert!(parse_qwen_xml("Qwen writes `<function=NAME>` blocks.", &[]).is_none());
        assert!(settle_native(ToolFormat::None, "", "text", &[], false)
            .calls
            .is_empty());
    }

    /// A value that CONTAINS `</parameter>` (an agent editing a parser or a template) ends at the
    /// `</parameter>` in structural position, not the first one. The first version cut the value
    /// at the literal and still returned the call as complete.
    #[test]
    fn qwen_xml_close_tag_inside_a_value_is_data() {
        let new_text = "const P_CLOSE: &str = \"</parameter>\";\nlet end = v.find(P_CLOSE); // </parameter> twice";
        let reply = format!(
            "<tool_call>\n<function=edit>\n<parameter=path>\nsrc/tool_parse.rs\n</parameter>\n\
<parameter=newText>\n{new_text}\n</parameter>\n<parameter=oldText>\nx\n</parameter>\n</function>\n</tool_call>"
        );
        let (calls, _) = qwen_xml_calls(&reply, &tools()).expect("a call");
        assert_eq!(calls.len(), 1);
        assert!(calls[0].1);
        let a = args(&calls[0].0);
        assert_eq!(a["newText"], new_text);
        assert_eq!(a["oldText"], "x");
        // The last value is followed by `</function>`: that `</parameter>` is structural too.
        let last =
            "<function=edit>\n<parameter=newText>\na</parameter>b\n</parameter>\n</function>";
        assert_eq!(
            args(&parse_qwen_xml(last, &tools()).unwrap().0[0])["newText"],
            "a</parameter>b"
        );
    }

    /// Qwen3-Coder's template replays list/dict arguments in Python repr, and the model writes
    /// them back that way: read as JSON, not kept as a string the client's schema rejects.
    #[test]
    fn qwen_xml_python_literal_values_become_json() {
        let todo = |params: serde_json::Value| Tool {
            r#type: "function".into(),
            function: ToolFunction {
                name: "todowrite".into(),
                description: None,
                parameters: Some(params),
            },
        };
        let t = vec![todo(serde_json::json!({"type": "object", "properties": {
            "todos": {"type": "array"}, "note": {"type": "string"}}}))];
        let reply = "<tool_call>\n<function=todowrite>\n<parameter=todos>\n\
[{'content': 'fix', 'status': 'pending'}, {'content': \"it's \\\"done\\\"\\n\", 'ok': True, 'n': None, 'x': 1e-05,},]\n\
</parameter>\n<parameter=note>\n['stays', 'text']\n</parameter>\n</function>\n</tool_call>";
        let (calls, _) = parse_qwen_xml(reply, &t).expect("a call");
        assert_eq!(
            args(&calls[0]),
            serde_json::json!({
                "todos": [{"content": "fix", "status": "pending"},
                          {"content": "it's \"done\"\n", "ok": true, "n": null, "x": 1e-05}],
                "note": "['stays', 'text']",
            })
        );
        // Not literals: a call, an unclosed string, a tuple.
        assert_eq!(python_literal_to_json("[print('x')]"), None);
        assert_eq!(python_literal_to_json("['open"), None);
        assert_eq!(python_literal_to_json("{'b': (1, 2)}"), None);
        assert_eq!(
            python_literal_to_json(r"{'a': '\x41\u00e9\d'}").as_deref(),
            Some(r#"{"a":"Aé\\d"}"#)
        );
    }

    #[test]
    fn hermes_every_block_and_lead_ignores_tags_inside_strings() {
        // `</tool_call>` inside an argument string is data: the object is balanced first.
        let reply = "The `<tool_call>` tag wraps a call.\n<tool_call>\n{\"name\": \"write\", \"arguments\": \
{\"content\": \"end with </tool_call> and {\"}}\n</tool_call>";
        let (calls, lead) = parse_hermes(reply).expect("a call");
        assert_eq!(lead, "The `<tool_call>` tag wraps a call.");
        assert_eq!(calls.len(), 1);
        assert_eq!(
            args(&calls[0]),
            serde_json::json!({"content": "end with </tool_call> and {"})
        );
        assert!(parse_hermes("Use the <tool_call> tag.").is_none());
    }

    #[test]
    fn hermes_every_block_and_lead() {
        let reply = "I'll check both files.\n<tool_call>\n{\"name\": \"read\", \"arguments\": {\"path\": \"a.py\", \"limit\": 5}}\n</tool_call>\n\
<tool_call>\n{\"name\": \"read\", \"arguments\": \"{\\\"path\\\": \\\"b.py\\\"}\"}\n</tool_call>";
        let (calls, lead) = parse_hermes(reply).expect("calls");
        assert_eq!(lead, "I'll check both files.");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.arguments, r#"{"path":"a.py","limit":5}"#);
        assert_eq!(args(&calls[1]), serde_json::json!({"path": "b.py"}));
        assert_ne!(calls[0].id, calls[1].id);
        assert!(parse_hermes("no calls here {\"name\": \"x\"}").is_none());
        // Raw newlines inside a string (an edit with code) are repaired, not dropped.
        let raw = "<tool_call>\n{\"name\": \"write\", \"arguments\": {\"content\": \"a\nb\"}}\n</tool_call>";
        assert_eq!(
            args(&parse_hermes(raw).unwrap().0[0]),
            serde_json::json!({"content": "a\nb"})
        );
    }

    fn run(s: &mut ReplySplitter, chunks: &[&str]) -> Vec<Piece> {
        chunks.iter().flat_map(|c| s.push(c)).collect()
    }

    fn joined(pieces: &[Piece]) -> (String, String) {
        let mut r = String::new();
        let mut c = String::new();
        for p in pieces {
            match p {
                Piece::Reasoning(t) => r.push_str(t),
                Piece::Content(t) => c.push_str(t),
            }
        }
        (r, c)
    }

    /// Token-sized chunks, markers split across chunk boundaries: reasoning streams up to
    /// `</think>`, prose streams up to `<tool_call>`, no partial marker is ever emitted, and the
    /// call text is handed back whole.
    #[test]
    fn splitter_streams_reasoning_and_prose_and_buffers_the_call() {
        let mut s = ReplySplitter::new(ToolFormat::QwenXml, true, false);
        let chunks = [
            "We need",
            " to read",
            " the file.\n",
            "</th",
            "ink>\n\n",
            "Let me",
            " look",
            ".\n\n<tool",
            "_call>\n<function=read>",
            "\n<parameter=path>\ncalc.py\n</parameter>\n</function>\n</tool_call>",
        ];
        let mut pieces = Vec::new();
        for c in chunks {
            let got = s.push(c);
            for p in &got {
                let t = match p {
                    Piece::Reasoning(t) | Piece::Content(t) => t,
                };
                assert!(!t.contains('<'), "a partial marker leaked: {t:?}");
            }
            pieces.extend(got);
        }
        let end = s.finish();
        assert!(end.pieces.is_empty());
        let (r, c) = joined(&pieces);
        assert_eq!(r, "We need to read the file.");
        assert_eq!(c, "Let me look.");
        assert_eq!(end.gap, "\n\n");
        assert!(end.buffered.starts_with("<tool_call>\n<function=read>"));
        assert!(parse_qwen_xml(&end.buffered, &[]).is_some());
    }

    #[test]
    fn splitter_without_a_call_flushes_the_answer_trimmed() {
        let mut s = ReplySplitter::new(ToolFormat::HermesJson, false, true);
        let pieces = run(
            &mut s,
            &[
                "\n", "<th", "ink>", "hmm", "</think>", "\n\nAll", " done. <", "3 ", "\n",
            ],
        );
        let end = s.finish();
        let (r, c) = joined(&pieces.into_iter().chain(end.pieces).collect::<Vec<_>>());
        assert_eq!(r, "hmm"); // a reply that opened <think> itself
        assert_eq!(c, "All done. <3"); // `<` held back, then released; trailing newline dropped
        assert!(end.buffered.is_empty());
    }

    #[test]
    fn splitter_cut_off_in_reasoning_is_all_reasoning() {
        let mut s = ReplySplitter::new(ToolFormat::QwenXml, true, false);
        let pieces = run(&mut s, &["still ", "thinking </thi"]);
        let end = s.finish();
        let (r, c) = joined(&pieces.into_iter().chain(end.pieces).collect::<Vec<_>>());
        assert_eq!(r, "still thinking </thi");
        assert_eq!(c, "");
        assert!(end.buffered.is_empty());
    }

    /// The fenced fallback buffers the whole answer (its call is only recognisable once the JSON
    /// closes) but still streams reasoning.
    #[test]
    fn splitter_fenced_mode_buffers_the_answer() {
        let mut s = ReplySplitter::new(ToolFormat::None, true, false);
        let pieces = run(
            &mut s,
            &[
                "think",
                "</think>\n",
                "```json\n{\"function\": \"x\"}",
                "\n```",
            ],
        );
        let end = s.finish();
        assert_eq!(joined(&pieces).0, "think");
        assert!(end.pieces.is_empty());
        assert_eq!(end.buffered, "```json\n{\"function\": \"x\"}\n```");
    }

    /// A reply the token budget cut off inside a call's value (a `write` mid-file): with `cut`
    /// the call is NOT sent — the text goes back verbatim and the caller keeps finish_reason
    /// `length` — while a reply that ended on EOS keeps the lenient parse. An earlier complete
    /// call survives the cut of a later one.
    #[test]
    fn cut_call_is_never_sent_as_a_call() {
        let cut = "<tool_call>\n<function=write>\n<parameter=path>\na.py\n</parameter>\n\
<parameter=content>\ndef add(a, b):\n    return";
        let end = settle_native(ToolFormat::QwenXml, "\n\n", cut, &[], true);
        assert!(end.calls.is_empty());
        assert_eq!(end.content, format!("\n\n{cut}"));
        assert_eq!(end.note, Some(EndNote::CutCall));
        // The same text ending on EOS: a sloppy model, parsed leniently.
        let eos = settle_native(ToolFormat::QwenXml, "", cut, &[], false);
        assert_eq!(eos.calls.len(), 1);
        assert_eq!(eos.note, None);
        // A missing `</parameter>` inside a CLOSED function is still a complete call when cut.
        let closed =
            "<tool_call>\n<function=read>\n<parameter=path>\na.py\n</function>\n</tool_call>";
        assert_eq!(
            settle_native(ToolFormat::QwenXml, "", closed, &[], true)
                .calls
                .len(),
            1
        );
        // Two calls, the second cut: only the first is sent.
        let two = format!("{closed}\n{cut}");
        let end = settle_native(ToolFormat::QwenXml, "", &two, &[], true);
        assert_eq!(end.calls.len(), 1);
        assert_eq!(end.calls[0].function.name, "read");
        assert_eq!(end.content, "");
        assert_eq!(end.note, Some(EndNote::CutCall));
        // Cut before any call parsed (mid-header): verbatim, nothing dropped.
        let head = settle_native(
            ToolFormat::QwenXml,
            "",
            "<tool_call>\n<function=wri",
            &[],
            true,
        );
        assert_eq!(head.content, "<tool_call>\n<function=wri");
        assert!(head.calls.is_empty());
    }

    /// A marker with nothing after it is dropped, not sent as text (the stray `<tool_call>` lines
    /// of the first OpenCode session); a marker in PROSE comes back verbatim after the same
    /// whitespace it followed; prose between a mention and the real call is kept.
    #[test]
    fn marker_residue_and_prose_mentions() {
        for bare in [
            "<tool_call>",
            "<tool_call>\n</tool_call>",
            "<tool_call>\n<function=\n",
        ] {
            let end = settle_native(ToolFormat::QwenXml, "\n", bare, &[], false);
            assert_eq!(end.content, "", "{bare:?}");
            assert!(end.calls.is_empty());
            assert_eq!(end.note, Some(EndNote::BareMarker));
        }
        let end = settle_native(ToolFormat::QwenXml, "\n", "<tool_call>\noops\n", &[], false);
        assert_eq!(end.content, "\noops");
        assert_eq!(end.note, Some(EndNote::Unparsed));

        // "Qwen wraps calls in the <tool_call> tag." streamed through the splitter.
        let mut s = ReplySplitter::new(ToolFormat::QwenXml, false, false);
        let pieces = run(
            &mut s,
            &["Qwen wraps calls in the", " <tool_call> tag", "."],
        );
        let end = s.finish();
        assert_eq!(joined(&pieces).1, "Qwen wraps calls in the");
        assert_eq!(end.gap, " ");
        let settled = settle_native(ToolFormat::QwenXml, &end.gap, &end.buffered, &[], false);
        assert_eq!(settled.content, " <tool_call> tag.");
        assert!(settled.calls.is_empty());

        // A mention, then a real call: the text between them is sent before the call.
        let buffered = "<tool_call> tag. Reading now.\n<tool_call>\n<function=read>\n\
<parameter=path>\na.py\n</parameter>\n</function>\n</tool_call>";
        let end = settle_native(ToolFormat::QwenXml, " ", buffered, &[], false);
        assert_eq!(end.content, " <tool_call> tag. Reading now.");
        assert_eq!(end.calls.len(), 1);
        assert_eq!(end.note, None);
    }

    #[test]
    fn holdback_keeps_marker_prefixes_and_trailing_space() {
        assert_eq!(holdback("abc <tool_", &["<tool_call>"]), 7);
        assert_eq!(holdback("abc", &["<tool_call>"]), 0);
        assert_eq!(holdback("abc \n", &["<tool_call>"]), 2);
        assert_eq!(holdback("é<", &["<tool_call>", "<function="]), 1);
    }
}
