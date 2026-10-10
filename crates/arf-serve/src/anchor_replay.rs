//! ANCHOR REPLAY: the prefix anchors a server learned survive a restart.
//!
//! An anchor snapshot (`crate::prefix_anchor`, `SeqPlan::snapshot_anchor`) is what lets a NEW
//! session of an agent skip the shared system + tools prefix: 88-90 s for Claude Code's first
//! turn without one, 14.4 s with one (measured 2026-09-26). It lives in host RAM and the
//! KV it pairs with in GPU memory, so a restart loses both and the next session pays the 88 s.
//!
//! What is persisted is NOT the state. It is the anchor's TOKEN PREFIX (a few hundred KB per
//! anchor): `~/.arf/anchors/<model>.json`, written when the server takes a new anchor. At
//! startup, after the warm-up requests, each recorded prefix is sent through the actor as an
//! ordinary one-token request carrying the same anchor hint, so the scheduler takes the same
//! snapshot under the same key (a key is a hash of token ids, `BlockManager::chain_hash`). The
//! replay runs on its own thread after the listener is up, shortest prefix first, so the tools
//! anchor is published before the system anchors that restore from it.
//!
//! Why replay rather than serialise the snapshot and KV: the KV lives in private sparse GPU
//! buffers and the block manager's cache would have to be rebuilt to match, all keyed to a model,
//! a KV format, a draft and a build. A replay goes through the exact path a cold session takes,
//! so it is correct by construction and has no format to version beyond the token ids. What it
//! costs: the prefill time of each anchor (~88 s for the 13K-token Claude Code prefix on the 27B)
//! spent on the GPU after a restart; a session that starts before the replay finishes gets no
//! hit from it and shares the GPU with it. NOT MEASURED on a real restart yet.
//!
//! DEFERRED, NOT DROPPED — full serialisation (still to do): writing the anchor
//! snapshot itself (GDN rows + draft ring, ~235 MB) and its KV blocks (~33 KB a token in q8 across
//! the 16 attention layers, blitted out of the private sparse buffers through a shared staging
//! buffer) to disk, and on start re-registering those blocks in the block manager (`cached`,
//! `hash_of`, `block_tokens`, `evictable` at refcount 0) and `note_snapshot`ing the keys. That
//! makes the first session after a restart warm at once instead of after the replay's prefill. It
//! needs a versioned file keyed to model, KV format, block size, snapshot step and draft, and a
//! GPU to verify text identical to a cold start. Build it if a measured restart shows sessions
//! arriving before the replay finishes; this replay is the fallback that serialisation would keep
//! for a stale or missing file.
//!
//! OPT-IN since 2026-10-05: `ARF_ANCHOR_REPLAY=1` turns on both the recording and the replay. It was
//! on by default for a day and MEASURED HARMFUL that way: the replay runs as ordinary requests, so the first
//! request after a start queued behind it — 109 s against 5.7 s in the prefix-cache gate, and a 1,021-token
//! SPEED-Bench answer at 10.3 tok/s for 108 s (the other seven ran 67-83 tok/s), which pulled that run's
//! single-stream figure from 74.0 to 40.3 tok/s. It becomes the default again only when the replay is
//! background work that yields to every waiting request.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

/// Anchors kept: `ANCHOR_SNAPSHOT_SLOTS`, the anchor pool's default size. Replaying more than the
/// pool holds would only evict the earlier replays.
pub const MAX_ENTRIES: usize = 4;

/// The on-disk format. Bump on any change; a file with another version is ignored.
const VERSION: u32 = 1;

/// One recorded anchor: the prompt tokens up to and including the first token past the snapshot
/// boundary `at` (a snapshot needs at least one prompt token after it), and which anchor it was.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Entry {
    pub key: u64,
    pub at: usize,
    pub tools: bool,
    pub tokens: Vec<u32>,
    /// Recency: the store's clock when the anchor was last taken or resumed from.
    pub used: u64,
}

impl Entry {
    /// The replay request's `(prompt, prefix_anchor, tools_anchor)`. A tools anchor is replayed
    /// with no system anchor, so the scheduler plans exactly it.
    pub fn replay(&self) -> (Vec<u32>, Option<usize>, Option<usize>) {
        if self.tools {
            (self.tokens.clone(), None, Some(self.at))
        } else {
            (self.tokens.clone(), Some(self.at), None)
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct File {
    version: u32,
    model: String,
    entries: Vec<Entry>,
}

struct Store {
    path: PathBuf,
    model: String,
    entries: Vec<Entry>,
    clock: u64,
}

impl Store {
    /// Add or refresh `key`; returns true when the set of anchors changed (worth a write).
    fn record(&mut self, key: u64, at: usize, tools: bool, tokens: &[u32]) -> bool {
        self.clock += 1;
        if let Some(e) = self.entries.iter_mut().find(|e| e.key == key) {
            e.used = self.clock;
            return false;
        }
        if self.entries.len() >= MAX_ENTRIES {
            let oldest = (0..self.entries.len())
                .min_by_key(|&i| self.entries[i].used)
                .expect("full");
            self.entries.swap_remove(oldest);
        }
        self.entries.push(Entry {
            key,
            at,
            tools,
            tokens: tokens.to_vec(),
            used: self.clock,
        });
        true
    }

    fn touch(&mut self, key: u64) {
        self.clock += 1;
        if let Some(e) = self.entries.iter_mut().find(|e| e.key == key) {
            e.used = self.clock;
        }
    }

    fn save(&self) -> Result<(), String> {
        let file = File {
            version: VERSION,
            model: self.model.clone(),
            entries: self.entries.clone(),
        };
        let body = serde_json::to_vec(&file).map_err(|e| e.to_string())?;
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        let tmp = self.path.with_extension("json.tmp");
        std::fs::write(&tmp, body).map_err(|e| format!("{}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, &self.path).map_err(|e| format!("{}: {e}", self.path.display()))
    }
}

static STORE: OnceLock<Mutex<Store>> = OnceLock::new();

/// `ARF_ANCHOR_REPLAY` is set (opt-in; module docs say why).
pub fn enabled() -> bool {
    std::env::var_os("ARF_ANCHOR_REPLAY").is_some_and(|v| !v.is_empty() && v != "0")
}

/// What identifies the model the token ids belong to: the weights' canonical path, size and
/// modification time (the weight cache's `gguf_identity` convention). A changed file makes the
/// recorded anchors stale; they are then ignored, not replayed.
pub fn model_identity(model: &Path) -> String {
    let canon = model.canonicalize().unwrap_or_else(|_| model.to_path_buf());
    let md = std::fs::metadata(&canon).ok();
    let len = md.as_ref().map_or(0, |m| m.len());
    let mtime = md
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos());
    format!("{}|{len}|{mtime}", canon.display())
}

/// `~/.arf/anchors/<fnv of identity>.json` (`ARF_ANCHOR_DIR` overrides the directory).
pub fn default_path(identity: &str) -> Option<PathBuf> {
    let dir = match std::env::var_os("ARF_ANCHOR_DIR") {
        Some(d) => PathBuf::from(d),
        None => PathBuf::from(std::env::var_os("HOME")?).join(".arf/anchors"),
    };
    let h = identity.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ b as u64).wrapping_mul(0x0100_0000_01b3)
    });
    Some(dir.join(format!("{h:016x}.json")))
}

/// Read the anchors recorded for `model` at `path` (empty when absent, unreadable, another
/// version or another model), in no particular order: see [`replay_order`].
pub fn load(path: &Path, model: &str) -> Vec<Entry> {
    let Ok(bytes) = std::fs::read(path) else {
        return Vec::new();
    };
    match serde_json::from_slice::<File>(&bytes) {
        Ok(f) if f.version == VERSION && f.model == model => f.entries,
        Ok(_) => {
            eprintln!(
                "[anchor-replay] {} is for another model or format; ignored",
                path.display()
            );
            Vec::new()
        }
        Err(e) => {
            eprintln!("[anchor-replay] {}: {e}; ignored", path.display());
            Vec::new()
        }
    }
}

/// Anchors replayed after a start: the most recently used ones only (one agent's tools anchor and
/// system anchor), so a start re-reads the prompt the next session most likely sends and no other.
pub const REPLAY_LATEST: usize = 2;

/// The [`REPLAY_LATEST`] most recently used anchors, shortest prefix first: a tools anchor before
/// the system anchors that extend it, so each later replay resumes from the earlier one instead of
/// prefilling the shared part again.
pub fn replay_order(mut entries: Vec<Entry>) -> Vec<Entry> {
    entries.sort_by_key(|e| std::cmp::Reverse(e.used));
    entries.truncate(REPLAY_LATEST);
    entries.sort_by_key(|e| (e.at, !e.tools));
    entries
}

/// Turn recording on for this process (the server calls it once, with snapshots and anchors
/// on). Returns the anchors already recorded, to replay.
pub fn init(path: PathBuf, model: String) -> Vec<Entry> {
    let entries = load(&path, &model);
    let clock = entries.iter().map(|e| e.used).max().unwrap_or(0);
    let _ = STORE.set(Mutex::new(Store {
        path,
        model,
        entries: entries.clone(),
        clock,
    }));
    entries
}

/// The serving loop took an anchor snapshot under `key` at `at`; `prompt` is that sequence's
/// prompt. A no-op unless [`init`] ran.
pub fn record(key: u64, at: usize, tools: bool, prompt: &[u32]) {
    let Some(store) = STORE.get() else { return };
    let Some(tokens) = prompt.get(..at + 1) else {
        return;
    };
    let mut s = store.lock().unwrap_or_else(|p| p.into_inner());
    if s.record(key, at, tools, tokens) {
        if let Err(e) = s.save() {
            eprintln!("[anchor-replay] could not record the anchor: {e}");
        }
    }
}

/// A sequence resumed from the anchor under `key`: keep it ahead of colder ones. Persisted with
/// the next write.
pub fn touch(key: u64) {
    if let Some(store) = STORE.get() {
        store.lock().unwrap_or_else(|p| p.into_inner()).touch(key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(dir: &Path) -> Store {
        Store {
            path: dir.join("a.json"),
            model: "m|1|2".into(),
            entries: Vec::new(),
            clock: 0,
        }
    }

    fn tmpdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("arf-anchor-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn round_trips_and_ignores_another_model() {
        let d = tmpdir("rt");
        let mut s = store(&d);
        let prompt: Vec<u32> = (0..300).collect();
        assert!(s.record(7, 256, false, &prompt[..257]));
        assert!(
            !s.record(7, 256, false, &prompt[..257]),
            "same key: no write"
        );
        s.save().unwrap();
        let back = load(&s.path, "m|1|2");
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].tokens, prompt[..257].to_vec());
        assert_eq!(back[0].replay(), (prompt[..257].to_vec(), Some(256), None));
        assert!(load(&s.path, "other|1|2").is_empty());
        assert!(load(&d.join("missing.json"), "m|1|2").is_empty());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn keeps_the_most_recently_used_anchors() {
        let d = tmpdir("lru");
        let mut s = store(&d);
        let p: Vec<u32> = (0..10).collect();
        for k in 0..MAX_ENTRIES as u64 {
            s.record(k, 4, false, &p[..5]);
        }
        s.touch(0); // resumed from: now the newest
        s.record(99, 4, false, &p[..5]);
        let keys: Vec<u64> = s.entries.iter().map(|e| e.key).collect();
        assert!(keys.contains(&0) && keys.contains(&99), "{keys:?}");
        assert!(!keys.contains(&1), "the least recently used goes: {keys:?}");
        assert_eq!(keys.len(), MAX_ENTRIES);
    }

    #[test]
    fn tools_anchors_replay_first_and_alone() {
        let e = |key, at, tools, used| Entry {
            key,
            at,
            tools,
            tokens: vec![1; at + 1],
            used,
        };
        let keys = |v: Vec<Entry>| replay_order(v).iter().map(|e| e.key).collect::<Vec<u64>>();
        // the two most recently used, shortest first: the tools anchor, then the system anchor
        assert_eq!(
            keys(vec![
                e(1, 4096, false, 3),
                e(2, 1024, true, 2),
                e(3, 2048, false, 1)
            ]),
            vec![2, 1]
        );
        assert_eq!(
            keys(vec![
                e(1, 4096, false, 1),
                e(2, 1024, true, 3),
                e(3, 2048, false, 2)
            ]),
            vec![2, 3]
        );
        assert_eq!(e(2, 1024, true, 0).replay().1, None);
        assert_eq!(e(2, 1024, true, 0).replay().2, Some(1024));
    }
}
