//! PREFIX ANCHOR — where a chat request's shared "system + tools" prefix ends, in tokens.
//!
//! WHY (2026-09-26, measured): on a
//! hybrid model a prefix-cache hit can only resume at a recurrent-state SNAPSHOT, and the
//! scheduler took one per prompt, one window before the prompt's end — the next turn of the same
//! conversation resumes there. A NEW session of the same agent shares the system prompt and the
//! tool schemas (Claude Code: ~13.4K tokens; OpenCode ~5-6K) and diverges at its first user
//! message, which no snapshot sat before, so it re-prefilled everything: Claude Code's first turn
//! took 88-90 s on every new session, while an identical request hit in 3 s. This module tells
//! the scheduler where the shared part ends (`Job::prefix_anchor`); the scheduler snapshots there
//! too (`SeqPlan::snapshot_anchor`), into a pool of its own.
//!
//! HOW — template-agnostic, because every template renders the system turn differently (Qwen3.8
//! merges an effort instruction into it and lists the tools there; Qwen3-Coder invents a default
//! system line when tools come without one; gemma folds system into a user turn). Render the
//! request's own leading system messages and tools, THE SAME WAY ITS PATH RENDERS THE REAL
//! PROMPT, followed by one user message twice — once with [`SENTINEL_A`], once with
//! [`SENTINEL_B`] — tokenize both, and take the longest common TOKEN prefix of the two renders
//! and the real prompt. The sentinels differ in their first character, so the two renders part
//! at the first token that holds any user content: the anchor is before the user's text by
//! construction, whatever the template and however BPE merges across the seam. The real prompt
//! bounds it too, so it is always a token prefix of what the model actually runs.
//!
//! COST: two renders and two tokenizations of the prefix — for a 13.4K-token agent prompt, the
//! same order of work as the request's own render. Cached per (render inputs) in a small bounded
//! map ([`AnchorCache`]): an agent session sends the same system prompt and tools every turn, so
//! only its first request pays. What is cached is the sentinel renders' shared prefix; the real
//! prompt's bound is taken per request (a slice comparison).
//!
//! NOT MEASURED when written: the end-to-end effect on a new session's first turn. Everything
//! here is CPU and unit-tested; the win needs a live Claude Code / OpenCode session with the log
//! lines (`[state-snapshot] anchor snapshot at N tokens`, `... resumes from the anchor snapshot`)
//! as proof it ran. Measured since: measured 2026-09-26 (a new
//! Claude Code session's first turn 88 -> 14.4 s, text identical to a no-cache server).
//!
//! TOOLS ANCHOR (2026-09-27, [`tools_anchor`]) — a SECOND anchor, at the end of the tools block.
//! Measured that day: three Claude Code sessions in three working directories on one server —
//! `anchor snapshot at 13056` for the first, and NO `resumes from the anchor snapshot` for the
//! other two, each re-prefilling ~13K tokens (~75 s). Claude Code's system text carries the
//! working directory and environment, and Qwen3.8's template renders the tools block BEFORE the
//! system text, so two sessions in different directories share the ~12K-token tool schemas and
//! nothing after them: the anchor above (end of the system text) is never on the second one's
//! chain. Found the same way, varying the SYSTEM text instead of the user's: render
//! `[system(A), user(A)]` and `[system(B), user(A)]` with the same tools, take the common token
//! prefix, bound it by the real prompt. It is the tools block exactly when the template puts the
//! tools first; a template that writes the system text first (Qwen3-Coder) parts the two renders
//! right after `<|im_start|>system`, under [`MIN_TOKENS`] — no tools anchor, as before. Carried as
//! `Job::tools_anchor`; the scheduler snapshots there too (`SeqPlan::snapshot_tools_anchor`).
//! `ARF_NO_TOOLS_ANCHOR=1` is its control arm. NOT MEASURED end to end when written: the proof
//! it ran is `[state-snapshot] tools anchor snapshot at N tokens` on the first session and
//! `... resumes from the tools anchor snapshot at N tokens — a NEW session` on the next one in
//! another directory. Measured since (measured 2026-09-27, "Agent-session caching,
//! live"): both lines logged; a new session's first turn 98.4 -> 27.5 / 27.1 s.

use std::collections::VecDeque;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, Mutex};

use arf_core::Tokenizer;

use crate::chat::encode_prompt;
use crate::http::ChatMessage;

/// The two stand-in user messages. Any two strings work if they differ in their FIRST character
/// (so the renders part where the content starts) and carry nothing a template reacts to
/// (no surrounding whitespace — templates `trim`; no `<tool_response>` — Qwen's reads it).
pub const SENTINEL_A: &str = "Qz7 arf prefix anchor sentinel";
pub const SENTINEL_B: &str = "Jx4 arf prefix anchor sentinel";

/// The smallest anchor worth a snapshot — the scheduler's own floor (below it, prefill is cheap).
pub const MIN_TOKENS: usize = arf_core::scheduler::ANCHOR_MIN_TOKENS;

/// Distinct system prefixes remembered. A handful of agents at once is the realistic case; each
/// entry is one prefix's tokens (~54 KB for Claude Code's).
const CACHE_SLOTS: usize = 16;

/// A request's anchors, in tokens of its prompt: the end of the shared system + tools prefix
/// ([`anchor`]) and, when the template renders the tools before the system text, the end of
/// the tools block ([`tools_anchor`]). `tools < system` whenever both are set.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Anchors {
    pub system: Option<usize>,
    pub tools: Option<usize>,
    /// The prompt's HEADER TAIL ([`header_tail`]): not an anchor, but measured the same way.
    pub header: Option<usize>,
}

/// A tools anchor this close to the system anchor is dropped: the two would be one prefill window
/// apart at most — nothing gained for a ~235 MB anchor-pool slot. The backend's 128-token prefill
/// window (the scheduler's snapshot step until 2026-10-05); the scheduler checks the floored
/// boundaries itself too.
pub const TOOLS_ANCHOR_GAP: usize = 128;

/// `ARF_NO_TOOLS_ANCHOR=1` — no tools anchor is computed (and the scheduler plans none): the
/// A/B control arm for [`tools_anchor`]. The same switch the scheduler reads. Read once.
pub fn tools_enabled() -> bool {
    arf_core::scheduler::tools_anchor_snapshots_enabled()
}

/// `ARF_NO_ANCHOR_SNAPSHOT=1` — no anchor is computed, so none is planned or saved (the A/B
/// control arm: the scheduler then snapshots one window before each prompt's end only, as before
/// 2026-09-26). Read once.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ARF_NO_ANCHOR_SNAPSHOT").is_none())
}

/// The sentinel renders' shared token prefix, per hash of what shapes them (see [`anchor`]).
/// Most recently used first; `None` entries remember a render that could not be made, so a
/// template that cannot render the sentinel variant is not retried on every request.
pub struct AnchorCache {
    entries: Mutex<VecDeque<(u64, Option<Arc<[u32]>>)>>,
    /// [`header_tail`]'s probes, apart from the anchors so they never evict one.
    headers: Mutex<VecDeque<(u64, Option<Arc<[u32]>>)>>,
    /// Off: [`anchor`] computes nothing. A server whose scheduler takes no recurrent-state
    /// snapshots (a model without recurrent layers matches cached KV blocks directly, and a
    /// server with the prefix cache off matches nothing) has no use for an anchor.
    on: bool,
}

impl Default for AnchorCache {
    fn default() -> Self {
        Self::new(true)
    }
}

impl AnchorCache {
    /// `on`: the scheduler this server feeds takes state snapshots (`EngineConfig::state_snapshots`).
    pub fn new(on: bool) -> Self {
        AnchorCache {
            entries: Mutex::new(VecDeque::new()),
            headers: Mutex::new(VecDeque::new()),
            on,
        }
    }

    fn get_or_compute(
        &self,
        key: u64,
        compute: impl FnOnce() -> Option<Vec<u32>>,
    ) -> Option<Arc<[u32]>> {
        Self::slots_get_or_compute(&self.entries, key, compute)
    }

    fn slots_get_or_compute(
        slots: &Mutex<VecDeque<(u64, Option<Arc<[u32]>>)>>,
        key: u64,
        compute: impl FnOnce() -> Option<Vec<u32>>,
    ) -> Option<Arc<[u32]>> {
        {
            let mut e = slots.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(i) = e.iter().position(|(k, _)| *k == key) {
                let hit = e.remove(i).expect("position is in range");
                let v = hit.1.clone();
                e.push_front(hit);
                return v;
            }
        }
        // Computed OUTSIDE the lock: a second request for a new prefix computes it again rather
        // than waiting behind a 13K-token tokenization. Same inputs, same answer.
        let v: Option<Arc<[u32]>> = compute().map(Into::into);
        let mut e = slots.lock().unwrap_or_else(|p| p.into_inner());
        if !e.iter().any(|(k, _)| *k == key) {
            e.push_front((key, v.clone()));
            e.truncate(CACHE_SLOTS);
        }
        v
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.lock().unwrap().len()
    }
}

/// The leading system / developer messages — the "system" half of the shared prefix.
pub fn leading_system(messages: &[ChatMessage]) -> &[ChatMessage] {
    let n = messages
        .iter()
        .take_while(|m| m.role == "system" || m.role == "developer")
        .count();
    &messages[..n]
}

/// Length of the longest common prefix of two token sequences.
pub fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// A chat request's prefix anchor: how many leading tokens of `prompt_ids` are its shared
/// "system + tools" prefix, or `None` (anchors off; no system message and no tools; under
/// [`MIN_TOKENS`]; or `render` could not render the sentinel variant — no anchor rather than a
/// guess).
///
/// - `lead`: the request's leading system messages, in the form the path's own message
///   transform takes them (what `render` is then given, followed by one sentinel user turn);
/// - `has_tools`: tools are rendered into the prompt (a prefix without any system message);
/// - `shape`: everything besides `lead` that changes what `render` renders — the path, the
///   template variables or effort, the tools. Hashed with `lead` into the cache key. (A 64-bit
///   collision would hand back another prefix's tokens; the real prompt's bound still keeps the
///   anchor a prefix of this prompt, so the worst case is a snapshot at a less useful boundary.)
/// - `render`: renders a message list EXACTLY as this request's path renders its real one;
/// - `prompt_ids`: the real prompt, tokenized.
pub fn anchor(
    cache: &AnchorCache,
    tok: &Tokenizer,
    lead: &[ChatMessage],
    has_tools: bool,
    shape: impl Hash,
    render: impl Fn(&[ChatMessage]) -> Option<String>,
    prompt_ids: &[u32],
) -> Option<usize> {
    if !cache.on || !enabled() || (lead.is_empty() && !has_tools) || prompt_ids.len() < MIN_TOKENS {
        return None;
    }
    let t0 = std::time::Instant::now();
    let mut h = DefaultHasher::new();
    shape.hash(&mut h);
    for m in lead {
        m.role.hash(&mut h);
        m.content.hash(&mut h);
    }
    let shared = cache.get_or_compute(h.finish(), || sentinel_prefix(tok, lead, &render))?;
    let n = common_prefix(&shared, prompt_ids);
    if n < MIN_TOKENS {
        return None;
    }
    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
        eprintln!(
            "[serve] prefix anchor: {n} tokens (the shared system + tools prefix), found in \
             {:.1} ms; requests with the same prefix reuse it (logged once)",
            t0.elapsed().as_secs_f64() * 1e3
        );
    }
    Some(n)
}

/// A chat request's TOOLS ANCHOR (module docs, 2026-09-27): how many leading tokens of
/// `prompt_ids` are its tools block alone — shared by sessions of one agent whose SYSTEM text
/// differs — or `None`: anchors off, `ARF_NO_TOOLS_ANCHOR`, no `system` anchor, under
/// [`MIN_TOKENS`] (a template that renders the system text before the tools lands here), not at
/// least [`TOOLS_ANCHOR_GAP`] before `system`, or `render` could not render the sentinels.
///
/// - `lead_with(text)`: the request's leading messages, in the form `render` takes them (as
///   [`anchor`]'s `lead`), with the request's system text replaced by `text` — tools and
///   everything else as the real request has them. Called with two sentinel texts.
/// - `shape`, `render`, `prompt_ids`: as for [`anchor`]; `shape` is hashed with a tag and
///   `lead_with(SENTINEL_A)`, so a cache entry never collides with an [`anchor`] entry.
/// - `system`: this request's [`anchor`] — the tools anchor must come before it.
///
/// The caller computes it only when the request has tools AND a system text: without a system
/// text, [`anchor`] already ends where the tools do.
pub fn tools_anchor(
    cache: &AnchorCache,
    tok: &Tokenizer,
    lead_with: impl Fn(&str) -> Vec<ChatMessage>,
    shape: impl Hash,
    render: impl Fn(&[ChatMessage]) -> Option<String>,
    prompt_ids: &[u32],
    system: Option<usize>,
) -> Option<usize> {
    let system = system?;
    if !cache.on || !enabled() || !tools_enabled() || system < MIN_TOKENS + TOOLS_ANCHOR_GAP {
        return None;
    }
    let t0 = std::time::Instant::now();
    let mut h = DefaultHasher::new();
    "tools-anchor".hash(&mut h);
    shape.hash(&mut h);
    for m in lead_with(SENTINEL_A) {
        m.role.hash(&mut h);
        m.content.hash(&mut h);
    }
    let shared = cache.get_or_compute(h.finish(), || {
        let with = |system_text: &str| {
            let mut msgs = lead_with(system_text);
            msgs.push(ChatMessage::text("user", SENTINEL_A));
            encode_prompt(tok, &render(&msgs)?).ok()
        };
        let mut a = with(SENTINEL_A)?;
        let b = with(SENTINEL_B)?;
        a.truncate(common_prefix(&a, &b));
        Some(a)
    })?;
    let n = common_prefix(&shared, prompt_ids);
    if n < MIN_TOKENS || n + TOOLS_ANCHOR_GAP > system {
        return None;
    }
    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
        eprintln!(
            "[serve] tools anchor: {n} tokens (the tools block, rendered before the system \
             text; the system anchor is at {system}), found in {:.1} ms; sessions with the same \
             tools and another system text resume there (logged once)",
            t0.elapsed().as_secs_f64() * 1e3
        );
    }
    Some(n)
}

/// `ARF_NO_CONTEXT_ANCHOR=1` — no context anchor ([`context_anchor`]): the A/B control arm. Read once.
pub fn context_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ARF_NO_CONTEXT_ANCHOR").is_none())
}

/// A chat request's CONTEXT ANCHOR (2026-10-06): where the project context an agent puts at the
/// head of its FIRST user message ends — Claude Code sends that message as text blocks, its
/// `<system-reminder>`s with the project's `AGENTS.md` / `CLAUDE.md` first and the user's own
/// words last. Measured that day in a Batuhan-like setup (a 46 KB `AGENTS.md`, 3 MCP servers): a
/// 44,723-token first message whose anchor sat at 26,496 — the end of the system prompt and the
/// tools — so every new session in the same project, and every restart, read the ~18K tokens of
/// project context again (136-172 s). Found as [`anchor`] is, with the first user message's leading
/// blocks kept and only its last text replaced by the two sentinels: `with_user_text(text)` is the
/// request's leading messages plus that first user message so rebuilt. `None` when off, under
/// [`MIN_TOKENS`], not at least [`TOOLS_ANCHOR_GAP`] past `system` (no context worth a snapshot),
/// or `render` cannot render it. The caller uses it IN PLACE of the system anchor: the scheduler
/// takes two anchors a request, and the tools anchor still serves another project's sessions.
pub fn context_anchor(
    cache: &AnchorCache,
    tok: &Tokenizer,
    with_user_text: impl Fn(&str) -> Vec<ChatMessage>,
    shape: impl Hash,
    render: impl Fn(&[ChatMessage]) -> Option<String>,
    prompt_ids: &[u32],
    system: Option<usize>,
) -> Option<usize> {
    if !cache.on || !enabled() || !context_enabled() || prompt_ids.len() < MIN_TOKENS {
        return None;
    }
    let t0 = std::time::Instant::now();
    let mut h = DefaultHasher::new();
    "context-anchor".hash(&mut h);
    shape.hash(&mut h);
    for m in with_user_text(SENTINEL_A) {
        m.role.hash(&mut h);
        m.content.hash(&mut h);
    }
    let shared = cache.get_or_compute(h.finish(), || {
        let with = |text: &str| encode_prompt(tok, &render(&with_user_text(text))?).ok();
        let mut a = with(SENTINEL_A)?;
        let b = with(SENTINEL_B)?;
        a.truncate(common_prefix(&a, &b));
        Some(a)
    })?;
    let n = common_prefix(&shared, prompt_ids);
    if n < MIN_TOKENS || system.is_some_and(|s| n < s + TOOLS_ANCHOR_GAP) {
        return None;
    }
    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
        eprintln!(
            "[serve] context anchor: {n} tokens (the system prompt, the tools and the project \
             context at the head of the first user message; the system anchor was at {}), found \
             in {:.1} ms; new sessions in the same project resume there (logged once)",
            system.map_or("none".into(), |s| s.to_string()),
            t0.elapsed().as_secs_f64() * 1e3
        );
    }
    Some(n)
}

/// Both anchors of one request: [`anchor`], then — when `tools_lead` is given (the request has
/// tools and a system text) — [`tools_anchor`] with it as `lead_with`. The same `shape` and
/// `render` serve both.
#[allow(clippy::too_many_arguments)]
pub fn anchors(
    cache: &AnchorCache,
    tok: &Tokenizer,
    lead: &[ChatMessage],
    has_tools: bool,
    shape: impl Hash,
    render: impl Fn(&[ChatMessage]) -> Option<String>,
    prompt_ids: &[u32],
    tools_lead: Option<&dyn Fn(&str) -> Vec<ChatMessage>>,
) -> Anchors {
    let system = anchor(cache, tok, lead, has_tools, &shape, &render, prompt_ids);
    let tools = tools_lead.and_then(|lead_with| {
        tools_anchor(cache, tok, lead_with, &shape, &render, prompt_ids, system)
    });
    let header = header_tail(cache, tok, &shape, &render, prompt_ids);
    Anchors {
        system,
        tools,
        header,
    }
}

/// A chat request's HEADER TAIL (2026-10-05): how many of `prompt_ids`'s last tokens the NEXT
/// turn of its conversation renders differently — the part of the assistant header the template
/// appends (a thinking template's `<think>`, say) that the next turn's render of the same reply
/// does not repeat. The scheduler's end-of-prompt snapshot backs off exactly this far (plus one)
/// instead of a fixed 32 tokens rounded down to a 128-token window, which re-prefilled up to 159
/// tokens on every resume: measured 2026-10-05, a repeated 34,972-token prompt resumed at 34,816
/// and took 1.56 s to its first token, against 240-357 ms elsewhere.
///
/// Measured by rendering, through the request's own `render`, one sentinel user turn and that
/// turn followed by a sentinel reply and a sentinel user turn, and counting what the first render
/// has past the two's common TOKEN prefix. It can be 0: Qwen3.8's template renders a past reply
/// with the same `<think>\n\n</think>\n\n` its header opens, so with thinking off the next turn
/// repeats the whole header. Kept only when `prompt_ids` really ends the way every prompt of this
/// shape does — the common token SUFFIX of two probes with different user text (the user turn's
/// end and the header) — so a prompt that does not (a forced tool call appended after the
/// header) gets `None`, and with it the fixed tail. `None` too when snapshots are off or `render`
/// fails. (The first cut checked the probe's last 8 tokens; Qwen3.8's ending is 7, so the 8th was
/// the probe's own text, no real prompt matched, and a repeated 32K prompt still resumed at 34,816
/// — measured 2026-10-05.)
pub fn header_tail(
    cache: &AnchorCache,
    tok: &Tokenizer,
    shape: impl Hash,
    render: impl Fn(&[ChatMessage]) -> Option<String>,
    prompt_ids: &[u32],
) -> Option<usize> {
    if !cache.on {
        return None;
    }
    let mut h = DefaultHasher::new();
    "header-tail".hash(&mut h);
    shape.hash(&mut h);
    // Cached as `[tail, the probe's last tokens...]`.
    let probe = AnchorCache::slots_get_or_compute(&cache.headers, h.finish(), || {
        let one = vec![ChatMessage::text("user", SENTINEL_A)];
        let mut two = one.clone();
        two.push(ChatMessage::text("assistant", SENTINEL_B));
        two.push(ChatMessage::text("user", SENTINEL_A));
        let a = encode_prompt(tok, &render(&one)?).ok()?;
        let b = encode_prompt(tok, &render(&two)?).ok()?;
        // A user text that ends differently from SENTINEL_A (the two sentinels share their last
        // words, so their renders' common suffix would run into the user's text).
        let other = encode_prompt(tok, &render(&[ChatMessage::text("user", "0")])?).ok()?;
        let tail = a.len() - common_prefix(&a, &b);
        let ending = a
            .iter()
            .rev()
            .zip(other.iter().rev())
            .take_while(|(x, y)| x == y)
            .count();
        if ending < tail {
            return None; // the tail reaches into the user's text: no header to measure
        }
        Some([&[u32::try_from(tail).ok()?][..], &a[a.len() - ending..]].concat())
    })?;
    let (tail, check) = probe.split_first()?;
    let hit = prompt_ids.ends_with(check);
    // once per outcome (rule 7: shown to have run, and shown when it did not apply)
    static SAID: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
    let bit = if hit { 1 } else { 2 };
    if SAID.fetch_or(bit, std::sync::atomic::Ordering::Relaxed) & bit == 0 {
        if hit {
            eprintln!(
                "[serve] header tail: {tail} token(s) the next turn renders differently; the \
                 end-of-prompt snapshot backs off that far, not {} (logged once)",
                arf_core::scheduler::SNAPSHOT_TAIL_TOKENS
            );
        } else {
            eprintln!(
                "[serve] header tail: a prompt does not end with its template's {}-token \
                 ending; it keeps the fixed tail (logged once)",
                check.len()
            );
        }
    }
    hit.then_some(*tail as usize)
}

/// [`tools_anchor`]'s `lead_with` for a path whose `render` takes the request's system text as
/// ONE leading system message (the native templates, and the fenced path, whose `render` merges
/// the tool prompt in itself).
pub fn system_lead(text: &str) -> Vec<ChatMessage> {
    vec![ChatMessage::text("system", text)]
}

/// The token prefix two renders share when all that differs between them is the user's text.
fn sentinel_prefix(
    tok: &Tokenizer,
    lead: &[ChatMessage],
    render: &impl Fn(&[ChatMessage]) -> Option<String>,
) -> Option<Vec<u32>> {
    let with = |text: &str| {
        let mut msgs = lead.to_vec();
        msgs.push(ChatMessage::text("user", text));
        encode_prompt(tok, &render(&msgs)?).ok()
    };
    let mut a = with(SENTINEL_A)?;
    let b = with(SENTINEL_B)?;
    a.truncate(common_prefix(&a, &b));
    Some(a)
}

/// Test fixtures shared with the handler tests (`http`, `http::anthropic`).
#[cfg(test)]
pub(crate) mod testing {
    use super::*;

    /// A byte-level BPE tokenizer shaped like Qwen's: the same pre-tokenizer (Qwen2's split regex,
    /// then byte-level), the chat control strings as atomic added tokens, and a few merges so
    /// tokens are not all single bytes. No model needed, and — unlike the handler tests'
    /// word-level vocabulary — it encodes ANY text losslessly, so token prefixes mean something.
    pub(crate) fn qwen_like_tokenizer() -> Tokenizer {
        // GPT-2's bytes_to_unicode: printable bytes map to themselves, the rest to 256 + n.
        let mut bs: Vec<u32> = (u32::from(b'!')..=u32::from(b'~'))
            .chain(0xA1..=0xAC)
            .chain(0xAE..=0xFF)
            .collect();
        let mut cs = bs.clone();
        let mut extra = 0;
        for b in 0..256u32 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(256 + extra);
                extra += 1;
            }
        }
        let mut vocab = serde_json::Map::new();
        for c in &cs {
            let id = vocab.len();
            vocab.insert(char::from_u32(*c).unwrap().to_string(), id.into());
        }
        let pairs = [
            ("Ġ", "t"),
            ("h", "e"),
            ("Ġt", "he"),
            ("i", "n"),
            ("Ġ", "a"),
            ("e", "r"),
            ("o", "n"),
            ("u", "s"),
            ("us", "er"),
            ("s", "y"),
            ("sy", "s"),
            ("sys", "t"),
            ("Ġ", "s"),
            ("o", "u"),
            ("Ċ", "Ċ"),
            ("Ġ", "Ġ"),
            ("ĠĠ", "ĠĠ"),
        ];
        let mut merges = Vec::new();
        for (a, b) in pairs {
            let id = vocab.len();
            vocab.insert(format!("{a}{b}"), id.into());
            merges.push(serde_json::json!([a, b]));
        }
        let specials = [
            "<|im_start|>",
            "<|im_end|>",
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<tool_response>",
            "</tool_response>",
        ];
        let added: Vec<serde_json::Value> = specials
            .iter()
            .enumerate()
            .map(|(i, s)| {
                serde_json::json!({"id": vocab.len() + i, "content": s, "single_word": false,
                    "lstrip": false, "rstrip": false, "normalized": false, "special": true})
            })
            .collect();
        let json = serde_json::json!({
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": added,
            "normalizer": null,
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},
                 "behavior": "Isolated", "invert": false},
                {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}]},
            "post_processor": null,
            "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false},
            "model": {"type": "BPE", "dropout": null, "unk_token": null,
                "continuing_subword_prefix": null, "end_of_word_suffix": null, "fuse_unk": false,
                "byte_fallback": false, "ignore_merges": false, "vocab": vocab, "merges": merges},
        });
        let path = std::env::temp_dir().join(format!(
            "arf_serve_anchor_tokenizer_{}_{}.json",
            std::process::id(),
            crate::tool_parse::new_call_id()
        ));
        std::fs::write(&path, json.to_string()).unwrap();
        let tok = Tokenizer::from_file(&path).expect("the test tokenizer.json parses");
        let _ = std::fs::remove_file(&path);
        tok
    }

    /// A system prompt of about `chars` characters — an agent's instructions, long enough to
    /// clear [`MIN_TOKENS`] on a byte-level vocabulary.
    pub(crate) fn long_system(chars: usize) -> String {
        let line = "You are a careful coding agent. Read before you edit, run the tests, and say \
                    what you verified.\n";
        line.repeat(chars / line.len() + 1)
    }

    /// Assert `n` is a good anchor for `prompt`, whose first user message is `user`: the tokens
    /// before it decode to a prefix of the prompt that ends before the user's text, and that
    /// still holds every one of `must_include` (the system text, the tools).
    pub(crate) fn assert_anchor_before_user(
        tok: &Tokenizer,
        prompt: &str,
        ids: &[u32],
        n: usize,
        user: &str,
        must_include: &[&str],
    ) {
        assert!(n <= ids.len());
        let head = tok.decode(&ids[..n], false).unwrap();
        assert!(
            prompt.starts_with(&head),
            "the anchor's tokens decode to a prefix of the prompt"
        );
        let user_at = prompt.find(user).expect("the prompt carries the user text");
        assert!(
            head.len() <= user_at,
            "the anchor ({} bytes) must end before the user text (at byte {user_at})",
            head.len()
        );
        for s in must_include {
            assert!(head.contains(s), "the shared prefix must include {s:?}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;
    use crate::jinja_chat::{ChatTemplate, TemplateVars};

    const QWEN38: &str = include_str!("../testdata/qwen38_chat_template.jinja");
    const QWEN3_CODER: &str = include_str!("../testdata/qwen3_coder_chat_template.jinja");

    fn tools() -> minijinja::Value {
        crate::jinja_chat::parse_ordered(
            r#"[{"type": "function", "function": {"name": "read_file", "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string",
                "description": "The file to read"}}, "required": ["path"]}}},
               {"type": "function", "function": {"name": "run_shell", "description": "Run a command",
                "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}}}]"#,
        )
        .unwrap()
    }

    fn vars() -> TemplateVars {
        TemplateVars {
            add_generation_prompt: true,
            ..Default::default()
        }
    }

    /// Render `messages` with `tpl`, tokenize, and compute the anchor through a fresh cache.
    fn anchor_of(
        tpl: &ChatTemplate,
        tok: &Tokenizer,
        messages: &[ChatMessage],
        tools: Option<minijinja::Value>,
    ) -> (String, Vec<u32>, Option<usize>) {
        let prompt = tpl.render_chat(messages, tools.clone(), &vars()).unwrap();
        let ids = encode_prompt(tok, &prompt).unwrap();
        let cache = AnchorCache::default();
        let n = anchor(
            &cache,
            tok,
            leading_system(messages),
            tools.is_some(),
            ("test", tools.clone().map(|t| t.to_string())),
            |m| tpl.render_chat(m, tools.clone(), &vars()).ok(),
            &ids,
        );
        (prompt, ids, n)
    }

    /// HEADER TAIL (2026-10-05): through Qwen3.8's own template, thinking on and off, the next
    /// turn of a conversation shares EVERY token of this turn's prompt before the measured tail —
    /// the property that keeps the end-of-prompt snapshot matchable — and the tail is a few
    /// tokens, not the fixed 32. A prompt that does not end with the header (a forced tool call
    /// appended after it) gets `None`.
    #[test]
    fn the_header_tail_is_what_the_next_turn_does_not_share() {
        let tpl = ChatTemplate::new(QWEN38.into(), Some("<|im_end|>".into())).unwrap();
        let tok = qwen_like_tokenizer();
        for think in [None, Some(false)] {
            let vars = TemplateVars {
                add_generation_prompt: true,
                enable_thinking: think,
                ..Default::default()
            };
            let render = |m: &[ChatMessage]| tpl.render_chat(m, None, &vars).ok();
            let turn1 = vec![
                ChatMessage::text("system", long_system(400)),
                ChatMessage::text("user", "What does this repository do?"),
            ];
            let mut turn2 = turn1.clone();
            turn2.push(ChatMessage::text("assistant", "It serves language models."));
            turn2.push(ChatMessage::text("user", "And how fast?"));
            let ids1 = encode_prompt(&tok, &render(&turn1).unwrap()).unwrap();
            let ids2 = encode_prompt(&tok, &render(&turn2).unwrap()).unwrap();
            let cache = AnchorCache::default();
            let h = header_tail(&cache, &tok, ("test", think), render, &ids1)
                .unwrap_or_else(|| panic!("a header tail (thinking {think:?})"));
            assert!(h < 8, "a few tokens at most, not the fixed 32: {h}");
            assert!(
                common_prefix(&ids1, &ids2) >= ids1.len() - h,
                "the next turn shares every token before the tail (thinking {think:?})"
            );
            let mut forced = ids1.clone();
            forced.extend(encode_prompt(&tok, "<tool_call>").unwrap());
            assert_eq!(
                header_tail(&cache, &tok, ("test", think), render, &forced),
                None
            );
        }
    }

    /// Qwen3.8's own template, the shape of an agent request: a long system prompt, tools, a
    /// first user message. The anchor ends before the user's text, covers the system prompt and
    /// every tool schema, and is a token prefix of the real prompt. A second session (another
    /// first message) gets the SAME anchor and shares every token before it — which is what makes
    /// the snapshot there matchable.
    #[test]
    fn qwen38_with_tools_anchors_before_the_first_user_message() {
        let tpl = ChatTemplate::new(QWEN38.into(), Some("<|im_end|>".into())).unwrap();
        let tok = qwen_like_tokenizer();
        let sys = long_system(3000);
        let s1 = vec![
            ChatMessage::text("system", sys.clone()),
            ChatMessage::text("user", "Fix the failing test in calc.py"),
        ];
        let (prompt, ids, n) = anchor_of(&tpl, &tok, &s1, Some(tools()));
        let n = n.expect("an anchor: system + tools is well over 1,024 tokens");
        assert_anchor_before_user(
            &tok,
            &prompt,
            &ids,
            n,
            "Fix the failing test",
            &[sys.trim(), "read_file", "run_shell", "</tools>"],
        );

        let s2 = vec![
            ChatMessage::text("system", sys.clone()),
            ChatMessage::text("user", "Add a --verbose flag to the CLI"),
        ];
        let (_, ids2, n2) = anchor_of(&tpl, &tok, &s2, Some(tools()));
        assert_eq!(n2, Some(n), "same system prefix, same anchor");
        assert_eq!(ids2[..n], ids[..n], "and the same tokens before it");

        // WHY TWO SENTINELS: a first message that happens to begin with sentinel A's own text.
        // One sentinel would agree with it for 30 characters and put the anchor INSIDE the
        // user's text; the second sentinel differs in its first character, so it cannot.
        let echo = format!("{SENTINEL_A} and then the actual question");
        let s3 = vec![
            ChatMessage::text("system", sys.clone()),
            ChatMessage::text("user", echo.clone()),
        ];
        let (prompt3, ids3, n3) = anchor_of(&tpl, &tok, &s3, Some(tools()));
        assert_eq!(n3, Some(n));
        assert_anchor_before_user(&tok, &prompt3, &ids3, n, &echo, &["</tools>"]);
    }

    /// Without tools, Qwen3.8 still leads with a system turn (effort instruction + the system
    /// prompt): the anchor is its end. Without a system prompt AND without tools there is no
    /// shared prefix to anchor.
    #[test]
    fn qwen38_without_tools_anchors_on_the_system_turn_and_nothing_without_one() {
        let tpl = ChatTemplate::new(QWEN38.into(), Some("<|im_end|>".into())).unwrap();
        let tok = qwen_like_tokenizer();
        let sys = long_system(2000);
        let msgs = vec![
            ChatMessage::text("system", sys.clone()),
            ChatMessage::text("user", "What does this repository do?"),
        ];
        let (prompt, ids, n) = anchor_of(&tpl, &tok, &msgs, None);
        let n = n.expect("an anchor");
        assert_anchor_before_user(
            &tok,
            &prompt,
            &ids,
            n,
            "What does this repository do?",
            &[sys.trim()],
        );

        let bare = vec![ChatMessage::text("user", long_system(2000))];
        assert_eq!(anchor_of(&tpl, &tok, &bare, None).2, None);
    }

    /// A MULTI-TURN conversation (turn 3 of a session) anchors at the same point as its first
    /// turn: the anchor is the system prefix, not the conversation so far.
    #[test]
    fn a_later_turn_anchors_at_the_same_system_prefix() {
        let tpl = ChatTemplate::new(QWEN38.into(), Some("<|im_end|>".into())).unwrap();
        let tok = qwen_like_tokenizer();
        let sys = long_system(3000);
        let first = vec![
            ChatMessage::text("system", sys.clone()),
            ChatMessage::text("user", "Fix calc.py"),
        ];
        let mut later = first.clone();
        later.push(ChatMessage::text(
            "assistant",
            "Done — the sign was flipped.",
        ));
        later.push(ChatMessage::text("user", "Now add a test for it"));
        let (_, ids1, n1) = anchor_of(&tpl, &tok, &first, Some(tools()));
        let (_, ids3, n3) = anchor_of(&tpl, &tok, &later, Some(tools()));
        assert_eq!(n1, n3);
        let n = n1.unwrap();
        assert_eq!(ids1[..n], ids3[..n]);
    }

    /// Qwen3-Coder's template (the other native qwen-xml template on this disk): with a system
    /// prompt, and with tools but NO system prompt, where the template writes its own default
    /// system line — shared by every session, so it belongs in the anchor.
    #[test]
    fn qwen3_coder_anchors_with_and_without_a_system_prompt() {
        let tpl = ChatTemplate::new(QWEN3_CODER.into(), Some("<|im_end|>".into())).unwrap();
        let tok = qwen_like_tokenizer();
        let sys = long_system(3000);
        let msgs = vec![
            ChatMessage::text("system", sys.clone()),
            ChatMessage::text("user", "List the files"),
        ];
        let (prompt, ids, n) = anchor_of(&tpl, &tok, &msgs, Some(tools()));
        assert_anchor_before_user(
            &tok,
            &prompt,
            &ids,
            n.expect("an anchor"),
            "List the files",
            &[sys.trim(), "read_file", "run_shell"],
        );

        // Tools only: enough schema text to clear the floor.
        let many: Vec<String> = (0..40)
            .map(|i| {
                format!(
                    r#"{{"type": "function", "function": {{"name": "tool_{i}", "description":
                    "Tool number {i} does one careful thing", "parameters": {{"type": "object",
                    "properties": {{"arg": {{"type": "string", "description": "the input"}}}}}}}}}}"#
                )
            })
            .collect();
        let many = crate::jinja_chat::parse_ordered(&format!("[{}]", many.join(","))).unwrap();
        let msgs = vec![ChatMessage::text("user", "List the files")];
        let (prompt, ids, n) = anchor_of(&tpl, &tok, &msgs, Some(many));
        assert_anchor_before_user(
            &tok,
            &prompt,
            &ids,
            n.expect("an anchor from the tools alone"),
            "List the files",
            &["You are Qwen", "tool_0", "tool_39"],
        );
    }

    /// The cache: the second request with the same prefix does not render again (a render that
    /// panics proves it), a different prefix is a different entry, and the map stays bounded.
    #[test]
    fn the_cache_renders_each_prefix_once_and_stays_bounded() {
        let tpl = ChatTemplate::new(QWEN38.into(), Some("<|im_end|>".into())).unwrap();
        let tok = qwen_like_tokenizer();
        let cache = AnchorCache::default();
        let sys = long_system(2000);
        let msgs = vec![
            ChatMessage::text("system", sys.clone()),
            ChatMessage::text("user", "hello"),
        ];
        let prompt = tpl.render_chat(&msgs, None, &vars()).unwrap();
        let ids = encode_prompt(&tok, &prompt).unwrap();
        let render = |m: &[ChatMessage]| tpl.render_chat(m, None, &vars()).ok();
        let lead = leading_system(&msgs);
        let n = anchor(&cache, &tok, lead, false, "k", render, &ids);
        assert!(n.is_some());
        let no_render = |_: &[ChatMessage]| -> Option<String> { panic!("rendered again") };
        assert_eq!(anchor(&cache, &tok, lead, false, "k", no_render, &ids), n);
        assert_eq!(cache.len(), 1);

        // A path that cannot render the sentinel variant gives no anchor — and is remembered.
        let fail = |_: &[ChatMessage]| -> Option<String> { None };
        assert_eq!(anchor(&cache, &tok, lead, false, "other", fail, &ids), None);
        assert_eq!(cache.len(), 2);
        assert_eq!(
            anchor(&cache, &tok, lead, false, "other", no_render, &ids),
            None
        );

        for i in 0..40 {
            anchor(&cache, &tok, lead, false, i, fail, &ids);
        }
        assert_eq!(cache.len(), CACHE_SLOTS);

        // A server whose scheduler takes no state snapshots computes nothing at all.
        let off = AnchorCache::new(false);
        assert_eq!(anchor(&off, &tok, lead, false, "k", no_render, &ids), None);
        assert_eq!(off.len(), 0);
    }

    // ---- TOOLS ANCHOR (2026-09-27) ----

    /// `n` tool schemas — enough text for the tools block alone to clear [`MIN_TOKENS`] (Claude
    /// Code's is ~12K tokens).
    fn many_tools(n: usize) -> minijinja::Value {
        let many: Vec<String> = (0..n)
            .map(|i| {
                format!(
                    r#"{{"type": "function", "function": {{"name": "tool_{i}", "description":
                    "Tool number {i} does one careful thing", "parameters": {{"type": "object",
                    "properties": {{"arg": {{"type": "string", "description": "the input"}}}}}}}}}}"#
                )
            })
            .collect();
        crate::jinja_chat::parse_ordered(&format!("[{}]", many.join(","))).unwrap()
    }

    /// Claude Code's system text: the same instructions, then the session's environment — the
    /// working directory differs between sessions started in different directories.
    fn system_in(cwd: &str) -> String {
        format!(
            "{}\nWorking directory: {cwd}\nIs directory a git repo: yes",
            long_system(3000)
        )
    }

    /// Both anchors of a rendered request, the way the handlers compute them (`anchors` with
    /// `system_lead` when the request has tools and a system text).
    fn anchors_of(
        tpl: &ChatTemplate,
        tok: &Tokenizer,
        cache: &AnchorCache,
        messages: &[ChatMessage],
        tools: Option<minijinja::Value>,
    ) -> (String, Vec<u32>, Anchors) {
        let prompt = tpl.render_chat(messages, tools.clone(), &vars()).unwrap();
        let ids = encode_prompt(tok, &prompt).unwrap();
        let lead = leading_system(messages);
        let tools_lead: Option<&dyn Fn(&str) -> Vec<ChatMessage>> =
            (tools.is_some() && !lead.is_empty()).then_some(&system_lead);
        let a = anchors(
            cache,
            tok,
            lead,
            tools.is_some(),
            ("test", tools.clone().map(|t| t.to_string())),
            |m| tpl.render_chat(m, tools.clone(), &vars()).ok(),
            &ids,
            tools_lead,
        );
        (prompt, ids, a)
    }

    /// THE CASE THIS EXISTS FOR, on Qwen3.8's own template: two sessions with the same tools and
    /// system texts that differ (another working directory). The tools anchor ends EXACTLY at
    /// the end of the tools block — only the whitespace the template puts between the block and
    /// the system text is left before the system text starts — it is the same for both
    /// sessions, their tokens agree up to it, and it is well before either system anchor. The
    /// system anchors differ in tokens (that is the problem): the second session is on the first
    /// one's chain only up to the tools anchor.
    #[test]
    fn qwen38_tools_anchor_ends_the_tools_block_before_any_system_text() {
        let tpl = ChatTemplate::new(QWEN38.into(), Some("<|im_end|>".into())).unwrap();
        let tok = qwen_like_tokenizer();
        let cache = AnchorCache::default();
        let mut seen = Vec::new();
        for (cwd, user) in [
            ("/work/a/project-one", "Fix the failing test"),
            ("/work/a/other-project", "Add a --verbose flag"),
        ] {
            let sys = system_in(cwd);
            let msgs = vec![
                ChatMessage::text("system", sys.clone()),
                ChatMessage::text("user", user),
            ];
            let (prompt, ids, a) = anchors_of(&tpl, &tok, &cache, &msgs, Some(many_tools(40)));
            let t = a
                .tools
                .expect("a tools anchor: the tools render before the system text");
            let s = a.system.expect("a system anchor");
            assert!(t + TOOLS_ANCHOR_GAP <= s, "tools {t}, system {s}");
            let head = tok.decode(&ids[..t], false).unwrap();
            assert!(prompt.starts_with(&head));
            let sys_at = prompt
                .find(sys.trim())
                .expect("the prompt carries the system text");
            assert!(
                head.len() <= sys_at,
                "the tools anchor must end before the system text"
            );
            assert!(
                prompt[head.len()..sys_at].trim().is_empty(),
                "and exactly at the end of the tools block: {:?} is left before it",
                &prompt[head.len()..sys_at]
            );
            for s in ["tool_0", "tool_39", "</tools>", "</IMPORTANT>"] {
                assert!(head.contains(s), "the tools anchor must include {s:?}");
            }
            assert_anchor_before_user(&tok, &prompt, &ids, s, user, &[sys.trim(), "</tools>"]);
            seen.push((ids, t, s));
        }
        let (ids1, t1, s1) = &seen[0];
        let (ids2, t2, _) = &seen[1];
        assert_eq!(t1, t2, "same tools, same tools anchor");
        assert_eq!(ids1[..*t1], ids2[..*t1], "and the same tokens before it");
        assert!(
            common_prefix(ids1, ids2) < *s1,
            "the second session is NOT on the first one's chain as far as its system anchor"
        );
        // One cache entry per anchor kind; the tools entry is shared by both sessions.
        assert_eq!(cache.len(), 3);
    }

    /// No tools: no tools anchor (nothing is computed — `anchors` is given no `tools_lead`),
    /// and the system anchor is today's. And a render without tools, if one were handed to
    /// `tools_anchor` anyway, parts at the system text right after the effort sentence: under
    /// the floor, `None`.
    #[test]
    fn no_tools_no_tools_anchor() {
        let tpl = ChatTemplate::new(QWEN38.into(), Some("<|im_end|>".into())).unwrap();
        let tok = qwen_like_tokenizer();
        let cache = AnchorCache::default();
        let msgs = vec![
            ChatMessage::text("system", system_in("/tmp/x")),
            ChatMessage::text("user", "What does this repository do?"),
        ];
        let (_, ids, a) = anchors_of(&tpl, &tok, &cache, &msgs, None);
        assert!(a.system.is_some());
        assert_eq!(a.tools, None);
        assert_eq!(cache.len(), 1, "no tools-anchor render at all");
        let t = tools_anchor(
            &cache,
            &tok,
            system_lead,
            "no tools",
            |m| tpl.render_chat(m, None, &vars()).ok(),
            &ids,
            a.system,
        );
        assert_eq!(t, None);
    }

    /// A template that renders the system text BEFORE the tools (Qwen3-Coder): the two sentinel
    /// renders part right after `<|im_start|>system\n` — no tools anchor, and the system anchor
    /// is exactly what it was without this feature.
    #[test]
    fn a_template_with_the_system_text_first_has_no_tools_anchor() {
        let tpl = ChatTemplate::new(QWEN3_CODER.into(), Some("<|im_end|>".into())).unwrap();
        let tok = qwen_like_tokenizer();
        let msgs = vec![
            ChatMessage::text("system", system_in("/work/a/project-one")),
            ChatMessage::text("user", "List the files"),
        ];
        let cache = AnchorCache::default();
        let (_, _, a) = anchors_of(&tpl, &tok, &cache, &msgs, Some(many_tools(40)));
        assert_eq!(a.tools, None);
        let (_, _, alone) = anchor_of(&tpl, &tok, &msgs, Some(many_tools(40)));
        assert_eq!(a.system, alone);
        assert!(a.system.is_some());
    }

    /// A short system text after a long tools block: the tools anchor would land within one
    /// window of the system anchor — the same snapshot, give or take a window — so it is
    /// dropped. And the tools-anchor cache entry renders once: a second request with the same
    /// tools does not render the sentinels again.
    #[test]
    fn a_tools_anchor_next_to_the_system_anchor_is_dropped_and_each_is_cached_once() {
        let tpl = ChatTemplate::new(QWEN38.into(), Some("<|im_end|>".into())).unwrap();
        let tok = qwen_like_tokenizer();
        let cache = AnchorCache::default();
        let short = vec![
            ChatMessage::text("system", "Be terse."),
            ChatMessage::text("user", "hi"),
        ];
        let (_, _, a) = anchors_of(&tpl, &tok, &cache, &short, Some(many_tools(40)));
        assert!(a.system.is_some());
        assert_eq!(a.tools, None, "{a:?}");

        let msgs = vec![
            ChatMessage::text("system", system_in("/work/a/project-one")),
            ChatMessage::text("user", "hi"),
        ];
        let tools = many_tools(40);
        let (_, ids, a) = anchors_of(&tpl, &tok, &cache, &msgs, Some(tools.clone()));
        let n = cache.len();
        let no_render = |_: &[ChatMessage]| -> Option<String> { panic!("rendered again") };
        let shape = ("test", Some(tools.to_string()));
        let again = anchors(
            &cache,
            &tok,
            leading_system(&msgs),
            true,
            shape,
            no_render,
            &ids,
            Some(&system_lead),
        );
        assert_eq!(again, a);
        assert!(a.tools.is_some());
        assert_eq!(cache.len(), n);
    }

    /// Opt-in check against the REAL tokenizer (`ARF_TOKENIZER=<Qwen3.8 tokenizer.json>`): the
    /// synthetic vocabulary above shares Qwen's pre-tokenizer but not its 151K merges. Without the
    /// variable this says so and checks nothing — it is not a pass.
    #[test]
    fn real_qwen_tokenizer_when_available() {
        let Some(path) = std::env::var_os("ARF_TOKENIZER") else {
            eprintln!("real_qwen_tokenizer_when_available: ARF_TOKENIZER unset — NOT RUN");
            return;
        };
        let tok = Tokenizer::from_file(&path).expect("ARF_TOKENIZER is a tokenizer.json");
        let tpl = ChatTemplate::new(QWEN38.into(), Some("<|im_end|>".into())).unwrap();
        let sys = long_system(6000);
        for user in [
            "Fix the failing test",
            "  leading spaces",
            "!bang first",
            "12345 digits",
        ] {
            let msgs = vec![
                ChatMessage::text("system", sys.clone()),
                ChatMessage::text("user", user),
            ];
            let (prompt, ids, n) = anchor_of(&tpl, &tok, &msgs, Some(tools()));
            let n = n.expect("an anchor");
            assert_anchor_before_user(
                &tok,
                &prompt,
                &ids,
                n,
                user.trim(),
                &[sys.trim(), "read_file", "run_shell"],
            );
            eprintln!(
                "real tokenizer, user {user:?}: prompt {} tokens, anchor {n}, tail after it {:?}",
                ids.len(),
                tok.decode(&ids[n..], false).unwrap()
            );
        }
        // The tools anchor (2026-09-27), with system texts that start with a letter, a digit, a
        // space and punctuation: it ends where the system text starts, whatever BPE does at
        // the seam.
        let cache = AnchorCache::default();
        for start in ["You are", "12 rules", " leading space", "# Heading"] {
            let sys = format!("{start}\n{}", system_in("/work/a/project-one"));
            let msgs = vec![
                ChatMessage::text("system", sys.clone()),
                ChatMessage::text("user", "Fix the failing test"),
            ];
            let (prompt, ids, a) = anchors_of(&tpl, &tok, &cache, &msgs, Some(many_tools(40)));
            let t = a.tools.expect("a tools anchor");
            let head = tok.decode(&ids[..t], false).unwrap();
            let sys_at = prompt.find(sys.trim()).unwrap();
            assert!(prompt.starts_with(&head) && head.len() <= sys_at);
            assert!(head.contains("</IMPORTANT>"));
            eprintln!(
                "real tokenizer, system {start:?}: tools anchor {t} of {} tokens, system anchor \
                 {:?}, between it and the system text {:?}",
                ids.len(),
                a.system,
                &prompt[head.len()..sys_at]
            );
        }
    }
}
