//! TEACHER-FORCED ACCEPTANCE (2026-09-28) — `ARF_FORCED_TOKENS=<path>`.
//!
//! Why: on the README greedy prompts another engine accepts ~7% more drafts a window than we do
//! (4.378 vs 4.079, measured 2026-09-27), and every comparison so far compared DIFFERENT
//! texts — the two engines' greedy outputs diverge within a few hundred tokens, and the audit that
//! retracted "the gap is in our draft pipeline" found the counts could not localise anything. This
//! feeds BOTH engines the SAME token sequence (prompt + a fixed continuation, ideally another engine's own
//! greedy output) and records, at every window start, what the draft proposed. Identical text makes
//! the proposals comparable position by position; the first positions where they differ are where
//! a pipeline difference lives (taps, context K/V, block inputs, selector, vocabulary, commit).
//! another engine's side is `dev/benchmarks/forced_accept.mm` in its tree; the comparer is
//! a standalone probe.
//!
//! What it does, for a LONE GREEDY request while the variable is set (nothing else changes):
//!   * the first time a sequence finishes its prompt, the prompt's token ids go to
//!     `<path>.prompt` (the another engine harness reads them, so both engines see the same ids) and the
//!     continuation is read from `<path>` (ids separated by spaces, commas or newlines; a JSON array
//!     works). No file = nothing is forced (the run records the prompt and generates as usual).
//!   * every emitted token is the continuation's, never the target's own: the prompt's first token
//!     (`override_plain`) and each speculative cycle's (`step`). The target still runs; its argmax
//!     is only logged (`target=`), so a near-tie where it disagrees with the continuation is visible.
//!   * each cycle drafts with the CPU-selected block draft (`block_draft`, waited — the proposals
//!     are needed on the host) and verifies a window made of CONTINUATION tokens, so the recurrent
//!     state, the KV and the draft's context ring hold exactly the forced text.
//!
//! `ARF_FORCED_WINDOW` picks which window is verified — i.e. where the next window starts:
//!   * `one` (default): `[anchor]` — one token a cycle, so EVERY position gets a draft window and
//!     another engine's windows (wherever its acceptance puts them) always have an Arf counterpart.
//!   * `draft`: `[anchor, the continuation tokens the draft matched]` — the natural stride
//!     (accepted + 1), as another engine commits under forcing; window starts agree with another engine only
//!     until the first position where the two engines' proposals differ.
//!   * `full`: `[anchor, 7 continuation tokens]` — the target's argmax on every row (`target=`),
//!     stride = how far the TARGET agrees with the continuation.
//!
//! The draft's context ring is fed only by the rows a verify keeps; a row's taps do not depend on
//! how the text was chunked, so the three modes should draft the same proposals at a shared
//! position up to the verify record's row-count numerics — `one` vs `draft` is that check.
//!
//! One `[fa]` line per cycle on stderr (a standalone probe reads them):
//! `[fa] engine=arf j=<anchor's index in the continuation> pos=<its position> anchor=<id>
//!  draft=<ids> forced=<next 7 continuation ids> match=<proposals matching them, in prefix order>
//!  target=<verify argmax per window row> commit=<tokens committed> window=<rows verified>`.
//! NOT MEASURED when written (built compile-only; to be run on a GPU).

use std::collections::HashMap;
use std::sync::Mutex;

use arf_core::backend::BatchedBackend;
use arf_core::scheduler::SeqPlan;

/// The forced-continuation mode is on (`ARF_FORCED_TOKENS` set). Read once.
pub(super) fn forced_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var_os("ARF_FORCED_TOKENS").is_some())
}

fn forced_path() -> Option<std::path::PathBuf> {
    std::env::var_os("ARF_FORCED_TOKENS").map(std::path::PathBuf::from)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum WindowMode {
    One,
    Draft,
    Full,
}

fn window_mode() -> WindowMode {
    static V: std::sync::OnceLock<WindowMode> = std::sync::OnceLock::new();
    *V.get_or_init(|| match std::env::var("ARF_FORCED_WINDOW").as_deref() {
        Ok("draft") => WindowMode::Draft,
        Ok("full") => WindowMode::Full,
        Ok("one") | Err(_) => WindowMode::One,
        Ok(other) => {
            eprintln!("[fa] ARF_FORCED_WINDOW={other:?} is not one|draft|full; using one");
            WindowMode::One
        }
    })
}

/// A forced sequence: where its prompt ends and the continuation it must emit.
struct ForcedSeq {
    prompt_len: usize,
    cont: Vec<u32>,
    /// Set once the sequence left the forced text (an anchor that is not the continuation's):
    /// nothing is forced or logged for it after that.
    broken: bool,
}

static SEQS: Mutex<Option<HashMap<u64, ForcedSeq>>> = Mutex::new(None);

/// Parse token ids: separated by whitespace, commas, or wrapped in a JSON array.
pub(super) fn parse_ids(text: &str) -> Result<Vec<u32>, String> {
    text.split(|c: char| c.is_whitespace() || c == ',' || c == '[' || c == ']')
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<u32>().map_err(|e| format!("token id {s:?}: {e}")))
        .collect()
}

fn join(ids: &[u32]) -> String {
    ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
}

/// Call for every scheduled sequence before the step runs. The first time a sequence reaches the
/// end of its prompt, record the prompt (`<path>.prompt`, and a `[fa-prompt]` line) and load the
/// continuation. `tokens` = the sequence's tokens so far (the scheduler's), `prompt_len` = where
/// this step leaves it (`past_len + q_len` of the chunk that completes the prompt).
pub(super) fn observe(sp: &SeqPlan, tokens: Option<&[u32]>) {
    if !sp.completes_prompt {
        return;
    }
    let mut g = SEQS.lock().unwrap();
    let map = g.get_or_insert_with(HashMap::new);
    if map.contains_key(&sp.id) {
        return;
    }
    let prompt_len = sp.past_len + sp.q_len;
    let prompt: Vec<u32> = tokens
        .and_then(|t| t.get(..prompt_len))
        .map(<[u32]>::to_vec)
        .unwrap_or_default();
    let Some(path) = forced_path() else {
        return;
    };
    let mut ppath = path.clone().into_os_string();
    ppath.push(".prompt");
    match std::fs::write(&ppath, join(&prompt).replace(',', " ") + "\n") {
        Ok(()) => {}
        Err(e) => eprintln!("[fa] cannot write the prompt ids to {ppath:?}: {e}"),
    }
    eprintln!(
        "[fa-prompt] engine=arf id={} n={} ids={}",
        sp.id,
        prompt.len(),
        join(&prompt)
    );
    let cont = match std::fs::read_to_string(&path) {
        Ok(text) => match parse_ids(&text) {
            Ok(ids) => ids,
            Err(e) => {
                eprintln!("[fa] {path:?}: {e} — nothing forced");
                Vec::new()
            }
        },
        Err(_) => {
            eprintln!(
                "[fa] no continuation at {path:?} — recording the prompt only ({ppath:?}); \
                 this request generates as usual"
            );
            Vec::new()
        }
    };
    if !cont.is_empty() {
        eprintln!(
            "[fa-cont] engine=arf id={} n={} window={:?} ids={}",
            sp.id,
            cont.len(),
            window_mode(),
            join(&cont)
        );
    }
    let broken = cont.is_empty() || prompt.len() != prompt_len;
    map.insert(
        sp.id,
        ForcedSeq {
            prompt_len,
            cont,
            broken,
        },
    );
}

/// The continuation token a plain step at `sp` must emit, if `sp` is forced: index
/// `past_len + q_len - prompt_len` (0 for the step that completes the prompt).
fn forced_next(sp: &SeqPlan) -> Option<u32> {
    let g = SEQS.lock().unwrap();
    let f = g.as_ref()?.get(&sp.id).filter(|f| !f.broken)?;
    let idx = (sp.past_len + sp.q_len).checked_sub(f.prompt_len)?;
    f.cont.get(idx).copied()
}

/// Whether this step's sequence is forced (lone, a decode row or the prompt's last chunk, a
/// continuation loaded and not left): the caller then skips every other speculative path.
pub(super) fn is_forced(sp: &SeqPlan) -> bool {
    sp.completes_prompt && forced_next(sp).is_some()
}

/// A PLAIN step's committed tokens, forced: each forced sequence's one token becomes the
/// continuation's (the target's own pick is logged when it differs, `[fa-plain]`).
pub(super) fn override_plain(seqs: &[SeqPlan], runs: &mut [Vec<u32>]) {
    for (sp, run) in seqs.iter().zip(runs.iter_mut()) {
        if !sp.completes_prompt || run.len() != 1 {
            continue;
        }
        if let Some(t) = forced_next(sp) {
            eprintln!(
                "[fa-plain] engine=arf id={} pos={} target={} forced={t}",
                sp.id,
                sp.past_len + sp.q_len,
                run[0]
            );
            run[0] = t;
        }
    }
}

/// One teacher-forced speculative cycle for the lone sequence `sp` (a decode row: its pending
/// token `anchor` sits at `past_len`). Drafts, verifies a window of CONTINUATION tokens and
/// returns the run to commit — the continuation's next tokens, never the target's. `None` =
/// nothing ran (the caller takes the plain step, which `override_plain` forces).
pub(super) fn step(backend: &dyn BatchedBackend, sp: &SeqPlan, anchor: u32) -> Option<Vec<u32>> {
    let (j, cont) = {
        let mut g = SEQS.lock().unwrap();
        let f = g.as_mut()?.get_mut(&sp.id).filter(|f| !f.broken)?;
        let j = sp.past_len.checked_sub(f.prompt_len)?;
        if f.cont.get(j) != Some(&anchor) {
            eprintln!(
                "[fa] engine=arf id={} LEFT the forced text at j={j}: anchor {anchor}, continuation \
                 {:?} — no longer forced",
                sp.id,
                f.cont.get(j)
            );
            f.broken = true;
            return None;
        }
        // the continuation from the anchor on: cont[0] = the anchor
        (j, f.cont[j..].to_vec())
    };
    if cont.len() < 2 {
        return None; // nothing left to force after the anchor
    }
    let draft = backend.block_draft(anchor, sp.past_len, sp.id);
    let forced: Vec<u32> = cont[1..].iter().take(7).copied().collect();
    let matched = draft
        .iter()
        .zip(&forced)
        .take_while(|(d, f)| d == f)
        .count();
    let want = match window_mode() {
        WindowMode::One => 1,
        WindowMode::Draft => 1 + matched,
        WindowMode::Full => 1 + forced.len(),
    };
    // the window must stay inside the continuation and the slots the block table backs now
    let bs = backend.kv_geometry().1;
    let cap = (sp.block_table.len() * bs).saturating_sub(sp.past_len);
    let rows = want.min(cont.len()).min(cap).max(1);
    let window: Vec<u32> = cont[..rows].to_vec();
    let prefix_slots = arf_core::cache::slots_for(&sp.block_table, bs, sp.past_len + rows);
    let Some(preds) = backend.verify_window(&window, sp.past_len, &prefix_slots, sp.id) else {
        eprintln!(
            "[fa] engine=arf id={} j={j} verify DECLINED (window {rows}) — plain step",
            sp.id
        );
        return None;
    };
    // The backend kept rows 0..=a (its accept rule on these tokens against its own argmax);
    // commit exactly those rows' tokens after the anchor, then the continuation's next token
    // in place of the target's bonus. If the continuation ends there, the target's bonus stands.
    let run_target = arf_core::model::speculative::accepted_run(&window, &preds);
    let a = run_target.len().saturating_sub(1);
    let mut run: Vec<u32> = window[1..=a].to_vec();
    run.push(cont.get(a + 1).copied().unwrap_or(run_target[a]));
    eprintln!(
        "[fa] engine=arf j={j} pos={} anchor={anchor} draft={} forced={} match={matched} \
         target={} commit={} window={rows}",
        sp.past_len,
        join(&draft),
        join(&forced),
        join(&preds),
        run.len()
    );
    Some(run)
}

#[cfg(test)]
mod tests {
    use super::parse_ids;

    #[test]
    fn parse_ids_takes_spaces_commas_newlines_and_json() {
        assert_eq!(parse_ids("1 2\n3,4").unwrap(), vec![1, 2, 3, 4]);
        assert_eq!(parse_ids("[5, 6, 7]\n").unwrap(), vec![5, 6, 7]);
        assert_eq!(parse_ids("").unwrap(), Vec::<u32>::new());
        assert!(parse_ids("1 x 2").is_err());
    }
}
