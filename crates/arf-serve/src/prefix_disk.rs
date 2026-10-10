//! ON-DISK PREFIX CACHE (issue #17, 2026-10-06): a cached agent prompt survives a restart as STATE,
//! not as tokens to prefill again.
//!
//! An agent's first message carries its system prompt and tool definitions — Claude Code's is
//! ~23K tokens, minutes of prefill on a Mac. The server keeps the state at the end of that prefix
//! (an anchor snapshot + the KV blocks before it) for later sessions, but only in memory. The
//! token-prefix replay (`anchor_replay`) re-derives it after a restart by prefilling it again:
//! 155 s in the background on an M4 Max (measured 2026-10-06). This module writes the state
//! itself — the snapshot and the KV of the prefix's positions (`BatchedBackend::prefix_export`,
//! ~1 GB for 20K tokens) — and reads it back at start in about a second, the way llama.cpp's slot
//! save/restore and another engine's SSD cache do. Measured 2026-10-06 (M4 Max, load ~100): an
//! 8,960-token system prefix saved in 2.1 s (584 MB) and loaded in 2.2 s after a restart; the
//! next request resumed from it (first token 3.3 s against ~75 s cold) with greedy text and 64
//! logprobs IDENTICAL to a cold run.
//!
//! `~/.arf/prefix/<fnv of identity>/<key>.arfpx`: magic, a JSON header (format version, model
//! identity, engine version, key, length, tools anchor or not), the token ids, the backend's blob.
//! A file is used only when every check holds: same identity, the key recomputed from its tokens
//! (`Scheduler::prefix_key`: tokenizer and block size), and the backend's own (KV bytes a slot,
//! recurrent layers and row lengths). A file that fails one is deleted.
//!
//! SAVE: an anchor is queued when it is taken (`note_anchor`) and written at the first idle moment
//! after its sequence has published the blocks (`Scheduler::prefix_cached_slots` finds them all):
//! both halves are final then. The blit runs on the serving thread (well under a second); the
//! file is written on its own thread, to a temporary name and renamed. LOAD: at the first idle
//! moment after the warm-up has run (the recurrent banks exist only after a first step), the
//! most recent files, each through `Scheduler::prefix_import_begin` -> `prefix_import` ->
//! `prefix_import_commit`: the scheduler can match the key only once its KV is written.
//!
//! On whenever anchor recording is (`anchor_replay::enabled`, which `arf run` / `arf launch` turn
//! on); `ARF_PREFIX_DISK=0` keeps only the token replay.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use arf_core::backend::BatchedBackend;
use arf_core::scheduler::Scheduler;

const MAGIC: &[u8; 8] = b"ARFPXF01";
const VERSION: u32 = 1;
/// Files kept, and loaded at start: for two agents (Claude Code and OpenCode side by side), each
/// one's tools anchor, its system or context anchor (the end of the project's AGENTS.md at the
/// head of the first message), and a junction. ~0.4-2.5 GB each.
const KEEP_FILES: usize = 6;

#[derive(serde::Serialize, serde::Deserialize)]
struct Header {
    version: u32,
    identity: String,
    key: u64,
    at: usize,
    tools: bool,
}

struct Pending {
    key: u64,
    at: usize,
    tools: bool,
    tokens: Vec<u32>,
}

struct State {
    dir: PathBuf,
    identity: String,
    exports: Vec<Pending>,
    imports: Vec<PathBuf>,
}

static STATE: OnceLock<Mutex<State>> = OnceLock::new();

/// `ARF_PREFIX_DISK=0` turns the state files off; otherwise they follow anchor recording.
pub fn enabled() -> bool {
    crate::anchor_replay::enabled() && std::env::var_os("ARF_PREFIX_DISK").is_none_or(|v| v != "0")
}

fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// The files of this model's directory, most recently written first.
fn files(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "arfpx"))
        .filter_map(|p| Some((std::fs::metadata(&p).ok()?.modified().ok()?, p)))
        .collect();
    v.sort_by(|a, b| b.0.cmp(&a.0));
    v.into_iter().map(|(_, p)| p).collect()
}

/// Turn the state files on for this process: `model` names the weights. Returns how many files
/// will be loaded at the first idle moment.
pub fn init(model: &Path) -> usize {
    let identity = format!(
        "{}|arf {}",
        crate::anchor_replay::model_identity(model),
        env!("CARGO_PKG_VERSION")
    );
    let Some(home) = std::env::var_os("HOME") else {
        return 0;
    };
    let dir = PathBuf::from(home)
        .join(".arf")
        .join("prefix")
        .join(format!("{:016x}", fnv(&identity)));
    let imports: Vec<PathBuf> = files(&dir).into_iter().take(KEEP_FILES).collect();
    let n = imports.len();
    let _ = STATE.set(Mutex::new(State {
        dir,
        identity,
        exports: Vec::new(),
        imports,
    }));
    n
}

/// Whether a state file for `key` is in place (the token replay then skips it).
pub fn has_file(key: u64) -> bool {
    STATE.get().is_some_and(|s| {
        s.lock()
            .unwrap()
            .dir
            .join(format!("{key:016x}.arfpx"))
            .is_file()
    })
}

/// The serving loop took an anchor (or junction) snapshot under `key` at `at`; `prompt` is the
/// sequence's prompt. Queued; written once its blocks are published ([`on_idle`]).
pub fn note_anchor(key: u64, at: usize, tools: bool, prompt: &[u32]) {
    let Some(st) = STATE.get() else { return };
    let mut st = st.lock().unwrap();
    if at > prompt.len() || st.exports.iter().any(|p| p.key == key) {
        return;
    }
    let path = st.dir.join(format!("{key:016x}.arfpx"));
    if path.is_file() {
        mark_used(&path);
        return;
    }
    st.exports.push(Pending {
        key,
        at,
        tools,
        tokens: prompt[..at].to_vec(),
    });
    // a queue that never drains (an anchor evicted before it was published) stays small
    if st.exports.len() > 2 * KEEP_FILES {
        st.exports.remove(0);
    }
}

/// A request resumed from the state under `key`: its file, if any, counts as used now.
///
/// Files are kept and loaded most recent first (`files`, `KEEP_FILES`), and that order was
/// the WRITE time: a state is written once and never again (`note_anchor` skips a key on disk),
/// so the tools anchor every session resumes from was the oldest file within days, and the first
/// pruned when a new project, a second agent or the safety check wrote theirs. Its use now moves
/// it up, so the files kept are the ones in use.
pub fn touch(key: u64) {
    let Some(st) = STATE.get() else { return };
    let path = st.lock().unwrap().dir.join(format!("{key:016x}.arfpx"));
    mark_used(&path);
}

fn mark_used(path: &Path) {
    if let Ok(f) = std::fs::File::options().write(true).open(path) {
        let _ = f.set_modified(std::time::SystemTime::now());
    }
}

/// An anchor loaded from disk: its key and whether it is the tools anchor.
pub struct Loaded {
    pub key: u64,
    pub tools: bool,
}

/// The serving loop is idle: load the state files (once, after the first step has run: `ready`)
/// and write any anchor whose blocks are now published. Returns the anchors loaded.
pub fn on_idle(sched: &mut Scheduler, backend: &dyn BatchedBackend, ready: bool) -> Vec<Loaded> {
    let Some(st) = STATE.get() else {
        return Vec::new();
    };
    let mut loaded = Vec::new();
    if ready {
        let imports = std::mem::take(&mut st.lock().unwrap().imports);
        let identity = st.lock().unwrap().identity.clone();
        for path in imports {
            let t = std::time::Instant::now();
            match load_one(&path, &identity, sched, backend) {
                Ok((key, at, tools, mb)) => {
                    eprintln!(
                        "[prefix-disk] loaded {} at {at} tokens ({mb} MB) from {} in {:.2} s",
                        if tools {
                            "tools anchor"
                        } else {
                            "prefix state"
                        },
                        path.display(),
                        t.elapsed().as_secs_f64()
                    );
                    loaded.push(Loaded { key, tools });
                }
                Err(e) => {
                    eprintln!("[prefix-disk] {}: {e}; removed", path.display());
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
    }
    // one export a call: an idle moment never holds a request back by more than one blit
    let next = {
        let mut st = st.lock().unwrap();
        let i = st
            .exports
            .iter()
            .position(|p| sched.prefix_cached_slots(&p.tokens, p.at).is_some());
        i.map(|i| (st.exports.remove(i), st.dir.clone(), st.identity.clone()))
    };
    if let Some((p, dir, identity)) = next {
        let slots = sched
            .prefix_cached_slots(&p.tokens, p.at)
            .expect("checked above");
        match backend.prefix_export(p.key, &slots) {
            Ok(blob) => {
                std::thread::spawn(move || {
                    if let Err(e) = write_one(&dir, &identity, &p, &blob) {
                        eprintln!("[prefix-disk] write failed: {e}");
                    }
                });
            }
            Err(e) => eprintln!("[prefix-disk] anchor at {} tokens not saved: {e}", p.at),
        }
    }
    loaded
}

/// Is a state ready to be saved ([`on_idle`] writes one a call)? The serving loop keeps calling
/// while this holds instead of sleeping until the next request.
///
/// MEASURED 2026-10-09 (the 27B, one 35,902-token Claude Code first message): the loop slept as
/// soon as it was idle, so of the three states that request left (the tools anchor, the context
/// anchor, the session start) only the first was written, and 200 s later the server was stopped
/// with the other two unsaved. The next start resumed from the tools anchor and read the project
/// context again — 15-19K tokens, minutes on a Mac, on every session after a restart.
pub fn exports_ready(sched: &Scheduler) -> bool {
    STATE.get().is_some_and(|st| {
        st.lock()
            .unwrap()
            .exports
            .iter()
            .any(|p| sched.prefix_cached_slots(&p.tokens, p.at).is_some())
    })
}

fn write_one(dir: &Path, identity: &str, p: &Pending, blob: &[u8]) -> std::io::Result<()> {
    let t = std::time::Instant::now();
    std::fs::create_dir_all(dir)?;
    let header = serde_json::to_vec(&Header {
        version: VERSION,
        identity: identity.to_string(),
        key: p.key,
        at: p.at,
        tools: p.tools,
    })?;
    let path = dir.join(format!("{:016x}.arfpx", p.key));
    let tmp = path.with_extension("tmp");
    {
        use std::io::Write;
        let mut f = std::io::BufWriter::with_capacity(8 << 20, std::fs::File::create(&tmp)?);
        f.write_all(MAGIC)?;
        f.write_all(&(header.len() as u64).to_le_bytes())?;
        f.write_all(&header)?;
        for t in &p.tokens {
            f.write_all(&t.to_le_bytes())?;
        }
        f.write_all(blob)?;
        f.flush()?;
    }
    std::fs::rename(&tmp, &path)?;
    eprintln!(
        "[prefix-disk] saved {} at {} tokens ({} MB) to {} in {:.2} s",
        if p.tools {
            "tools anchor"
        } else {
            "prefix state"
        },
        p.at,
        blob.len() >> 20,
        path.display(),
        t.elapsed().as_secs_f64()
    );
    for old in files(dir).into_iter().skip(KEEP_FILES) {
        let _ = std::fs::remove_file(old);
    }
    Ok(())
}

/// One file into the server: `(key, at, tools, megabytes)`.
fn load_one(
    path: &Path,
    identity: &str,
    sched: &mut Scheduler,
    backend: &dyn BatchedBackend,
) -> Result<(u64, usize, bool, usize), String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    if bytes.get(..8) != Some(MAGIC.as_slice()) {
        return Err("not a prefix state file".into());
    }
    let hlen =
        u64::from_le_bytes(bytes.get(8..16).ok_or("truncated")?.try_into().unwrap()) as usize;
    let h: Header = serde_json::from_slice(bytes.get(16..16 + hlen).ok_or("truncated header")?)
        .map_err(|e| e.to_string())?;
    if h.version != VERSION || h.identity != identity {
        return Err("written for another model or engine version".into());
    }
    let tok_end = 16 + hlen + 4 * h.at;
    let tokens: Vec<u32> = bytes
        .get(16 + hlen..tok_end)
        .ok_or("truncated tokens")?
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    if sched.prefix_key(&tokens, h.at) != h.key {
        return Err("its key does not match its tokens (tokenizer or block size changed)".into());
    }
    let blob = &bytes[tok_end..];
    let (seq, slots) = sched
        .prefix_import_begin(tokens)
        .ok_or("no free KV blocks for it")?;
    match backend.prefix_import(h.key, &slots, blob) {
        Ok(evicted) => {
            if let Some(old) = evicted {
                sched.forget_snapshot(old);
            }
            sched.prefix_import_commit(seq, h.key);
            Ok((h.key, h.at, h.tools, bytes.len() >> 20))
        }
        Err(e) => {
            sched.prefix_import_abort(seq);
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_in_use_is_kept_ahead_of_newer_ones() {
        let dir = std::env::temp_dir().join(format!("arf-prefix-disk-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hour_ago = std::time::SystemTime::now() - std::time::Duration::from_secs(3600);
        let (tools, newer) = (dir.join("a.arfpx"), dir.join("b.arfpx"));
        for p in [&tools, &newer] {
            std::fs::write(p, b"x").unwrap();
        }
        let old = std::fs::File::options().write(true).open(&tools).unwrap();
        old.set_modified(hour_ago).unwrap();
        assert_eq!(files(&dir)[0], newer, "written later, ranked first");
        mark_used(&tools);
        assert_eq!(files(&dir)[0], tools, "used now, ranked first");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
