//! RECURRENT-STATE SNAPSHOTS — what makes a prefix-cache hit valid on a hybrid model.
//!
//! The prefix cache shares KV blocks. On Qwen3.8-27B 48 of the 64 layers have no KV: their
//! state is a recurrence over every token so far, so a hit that skips the cached tokens leaves
//! those layers having never seen them (measured 2026-09-19: request 0 correct, requests 1-4
//! return unrelated text — prefix caching has been REFUSED on hybrid models since). A snapshot is
//! one sequence's recurrent bank row across every GDN layer (~151 MB on the 27B) plus, when a
//! block draft is attached, its context ring (84 MB), taken at a block boundary the scheduler
//! chose (`SeqPlan::snapshot_key`) and restored into a NEW sequence whose prompt matches down to
//! that boundary (`SeqPlan::restore_key`). Same design as vLLM's `mamba_cache_mode = "align"`.
//!
//! HOST copies of shared-storage buffers: a save or a restore is a ~150-240 MB memcpy, once per
//! request, against the seconds-to-minutes of prefill it replaces. The snapshots live in plain
//! RAM, not in the GPU's working set. A small LRU; the evicted key is returned so the scheduler
//! stops matching to it.
//!
//! TWO POOLS since 2026-09-26. The turn-end snapshots (one window before each prompt's end —
//! the next turn of the same conversation resumes there) and the ANCHOR snapshots (the end of
//! a request's shared "system + tools" prefix — a new session of the same agent resumes there,
//! `SeqPlan::snapshot_anchor`). One shared 4-slot LRU would lose the anchor within a session:
//! every turn saves a turn-end snapshot, so the fifth turn evicts it, and the next new session
//! re-prefills ~13.4K tokens (Claude Code's system prompt + 20 tool schemas, measured 88-90 s
//! for that first turn on 2026-09-26). Each pool is its own LRU; a key lives in one of them.
//!
//! THREE POOLS since later on 2026-09-26: JUNCTION snapshots (`SeqPlan::snapshot_junction`) —
//! the point where a prompt left cached history, learned from traffic the way another engine's
//! lazy junctions are (its `CacheLookup::junctionBoundary()`, runtime/engine/Cache.hpp, keeps
//! them in one state cache with a purpose priority rather than a pool of their own), so a new
//! session sharing a longer prefix than the anchor (Claude Code's
//! per-project `<system-reminder>` and environment blocks after the system + tools prefix) resumes
//! there. Why a pool of their own rather than the anchor pool: an anchor comes from the chat
//! template and serves every new session of an agent; a junction is a guess from history, and
//! some are dead weight (the next turn of a conversation whose own turn-end snapshot was evicted
//! plans one near the previous prompt's end, which no new session will reach). Sharing the
//! anchor pool's two slots would let those evict anchors — the 88-90 s miss the anchor pool
//! exists to prevent — and one agent in one project would fill both slots by itself (its anchor
//! plus its junction), so a second project's junction would evict the first's. A separate pool
//! costs host RAM (up to `JUNCTION_SNAPSHOT_SLOTS` x ~235 MB, NOT MEASURED on a box in swap);
//! `ARF_JUNCTION_SNAPSHOT_SLOTS=0` files junctions in the anchor pool instead.
//!
//! SESSION STARTS share the junction pool (2026-09-26, `SeqPlan::snapshot_session_start`): the
//! end-of-prompt snapshot of a conversation's FIRST turn is saved with `state_save_junction`, not
//! into the turn-end pool — a 6-7 turn agent session saves a turn-end snapshot every turn, so in
//! 4 slots the first turn's was gone before the next session started, and a new session with the
//! identical first request resumed only at the anchor (measured 2026-09-26: first turn
//! 14.4 s vs another engine's 2.0 s; the eviction is the first suspect, not a confirmed cause).
//! No change here: the pool, its LRU and `JUNCTION_SNAPSHOT_SLOTS` are as above. What it costs:
//! a session that takes a junction AND a session start fills both default slots by itself, so
//! the next such session evicts the previous one's session start — NOT measured whether 2 slots
//! are enough for a real agent workload (`ARF_JUNCTION_SNAPSHOT_SLOTS`, ~235 MB each).

use super::island::MetalIsland;
use objc2_metal::MTLBuffer as _;

/// Snapshots kept. Each is ~235 MB with a draft attached.
pub const STATE_SNAPSHOT_SLOTS: usize = 4;

/// ANCHOR snapshots kept, in their own pool (2026-09-26). Two: one agent's system prefix plus a
/// second (another agent, or the same agent with its tool set changed) — ~470 MB of host RAM
/// when both are full. `ARF_ANCHOR_SNAPSHOT_SLOTS` overrides; `0` files anchors in the turn-end
/// pool, i.e. the single shared LRU this store had before anchors (an A/B control arm). NOT
/// MEASURED: what the extra RAM costs a box already in swap.
///
/// FOUR since 2026-09-27: the TOOLS ANCHOR (`SeqPlan::snapshot_tools_anchor`, the end of the
/// tools block, which Qwen3.8 renders before the system text) is filed here too, so one agent
/// now takes two anchors — its tools anchor and, per working directory, a system anchor. With
/// two slots a Claude Code session in a second directory would evict the first directory's
/// system anchor with its own, and a third would evict the tools anchor they share. Four keeps
/// the tools anchor plus three directories' system anchors (the tools anchor is restored by
/// every new session, so the LRU keeps it). Up to ~940 MB of host RAM when all four are full,
/// nothing until an anchor is taken — NOT MEASURED on a box in swap.
pub const ANCHOR_SNAPSHOT_SLOTS: usize = 4;

/// `ARF_ANCHOR_SNAPSHOT_SLOTS`, read once.
/// Slots of the checkpoint pool (rolling checkpoints inside long prompts, 2026-10-07): 2 — the
/// latest of the read in progress and of the one before it. ~150 MB each on the 27B.
/// `ARF_CHECKPOINT_SNAPSHOT_SLOTS=0` files checkpoints in the turn-end pool.
pub fn checkpoint_snapshot_slots() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("ARF_CHECKPOINT_SNAPSHOT_SLOTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2)
    })
}

pub fn anchor_snapshot_slots() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("ARF_ANCHOR_SNAPSHOT_SLOTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(ANCHOR_SNAPSHOT_SLOTS)
    })
}

/// JUNCTION snapshots kept, in their own pool (2026-09-26; module docs). Two: the junctions of
/// two projects (or two agents) at once — ~470 MB of host RAM when both are full, and nothing
/// until a junction is actually taken. `ARF_JUNCTION_SNAPSHOT_SLOTS` overrides; `0` files
/// junctions in the anchor pool. NOT MEASURED: what the extra RAM costs a box already in swap.
pub const JUNCTION_SNAPSHOT_SLOTS: usize = 2;

/// `ARF_JUNCTION_SNAPSHOT_SLOTS`, read once.
pub fn junction_snapshot_slots() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("ARF_JUNCTION_SNAPSHOT_SLOTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(JUNCTION_SNAPSHOT_SLOTS)
    })
}

/// The LRU a snapshot lives in (module docs). Ordered by how long a key should survive: a key
/// re-saved under a lower pool KEEPS the higher one it was filed in (`file_snapshot`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SnapshotPool {
    /// A resume point inside a long prompt (2026-10-07): the shortest-lived.
    Checkpoint,
    /// One window before a prompt's end: the next turn of the same conversation.
    Turn,
    /// Where a prompt left cached history: a later session sharing that longer prefix.
    Junction,
    /// The end of the shared system + tools prefix: every new session of the agent.
    Anchor,
}

/// Slots per pool. A pool with 0 slots files its snapshots in the next pool down: junctions in
/// the anchor pool, anchors in the turn-end pool (with both 0: one shared LRU, as before
/// anchors).
#[derive(Clone, Copy, Debug)]
pub struct PoolSlots {
    pub checkpoint: usize,
    pub turn: usize,
    pub anchor: usize,
    pub junction: usize,
}

impl PoolSlots {
    /// The configured slots: [`STATE_SNAPSHOT_SLOTS`], `ARF_ANCHOR_SNAPSHOT_SLOTS`,
    /// `ARF_JUNCTION_SNAPSHOT_SLOTS`.
    pub fn configured() -> Self {
        PoolSlots {
            checkpoint: checkpoint_snapshot_slots(),
            turn: STATE_SNAPSHOT_SLOTS,
            anchor: anchor_snapshot_slots(),
            junction: junction_snapshot_slots(),
        }
    }

    /// The pool a snapshot asked for `pool` is actually filed in.
    fn effective(&self, pool: SnapshotPool) -> SnapshotPool {
        match pool {
            SnapshotPool::Junction if self.junction == 0 => self.effective(SnapshotPool::Anchor),
            SnapshotPool::Anchor if self.anchor == 0 => SnapshotPool::Turn,
            SnapshotPool::Checkpoint if self.checkpoint == 0 => SnapshotPool::Turn,
            p => p,
        }
    }

    fn of(&self, pool: SnapshotPool) -> usize {
        match pool {
            SnapshotPool::Checkpoint => self.checkpoint,
            SnapshotPool::Turn => self.turn,
            SnapshotPool::Anchor => self.anchor,
            SnapshotPool::Junction => self.junction,
        }
    }
}

pub struct StateSnapshot {
    key: u64,
    tick: u64,
    /// The pool it is filed in (see the module docs) — already the EFFECTIVE pool.
    pool: SnapshotPool,
    gdn: Vec<(usize, Vec<f32>, Vec<f32>)>,
    /// The draft's context ring at the same position: (positions, largest committed token,
    /// per-layer K, per-layer V).
    ring: Option<(usize, u32, Vec<Vec<f32>>, Vec<Vec<f32>>)>,
}

/// File `snap` into `store`, in the pool its `pool` names (made effective by `slots`), evicting
/// that pool's least recently used entry when it is full. Returns the key evicted (at most one —
/// only the pool `snap` joins can overflow). Pure bookkeeping, so it is tested without a Metal
/// device.
///
/// One key, one entry: a re-save replaces the old copy (same tokens, so the same state), and a
/// key that was ever an anchor STAYS one — the end-of-prompt snapshot of a prompt whose first
/// message is short can land on the very boundary an anchor was saved at, and filing that
/// re-save in the turn-end pool would put the anchor back within reach of per-turn evictions.
/// The same holds for a junction (2026-09-26): a key keeps the highest pool it was filed in.
/// `anchor_slots == 0` files anchors in the turn-end pool (one shared LRU, as before anchors).
fn file_snapshot(
    store: &mut Vec<StateSnapshot>,
    mut snap: StateSnapshot,
    slots: PoolSlots,
) -> Option<u64> {
    if let Some(i) = store.iter().position(|s| s.key == snap.key) {
        snap.pool = snap.pool.max(store[i].pool);
        store.swap_remove(i);
    }
    snap.pool = slots.effective(snap.pool);
    let cap = slots.of(snap.pool);
    snap.tick = store.iter().map(|s| s.tick).max().unwrap_or(0) + 1;
    let mut evicted = None;
    if store.iter().filter(|s| s.pool == snap.pool).count() >= cap.max(1) {
        let (i, _) = store
            .iter()
            .enumerate()
            .filter(|(_, s)| s.pool == snap.pool)
            .min_by_key(|(_, s)| s.tick)
            .unwrap();
        evicted = Some(store.swap_remove(i).key);
    }
    store.push(snap);
    evicted
}

impl MetalIsland {
    /// Save `stream`'s recurrent state under `key`. The caller has just run the step that left
    /// the sequence exactly at the boundary `key` names. Returns the key evicted to make room.
    pub fn state_snapshot_save_checkpoint(
        &self,
        stream: u64,
        key: u64,
    ) -> Result<Option<u64>, String> {
        self.state_snapshot_save_in(stream, key, SnapshotPool::Checkpoint)
    }

    pub fn state_snapshot_save(&self, stream: u64, key: u64) -> Result<Option<u64>, String> {
        self.state_snapshot_save_in(stream, key, SnapshotPool::Turn)
    }

    /// [`state_snapshot_save`](Self::state_snapshot_save) into the ANCHOR pool (module docs).
    pub fn state_snapshot_save_anchor(&self, stream: u64, key: u64) -> Result<Option<u64>, String> {
        static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
            match anchor_snapshot_slots() {
                0 => eprintln!(
                    "[state-snapshot] anchor snapshots share the {STATE_SNAPSHOT_SLOTS} turn-end \
                     slots (ARF_ANCHOR_SNAPSHOT_SLOTS=0)"
                ),
                n => eprintln!(
                    "[state-snapshot] anchor snapshots keep their own pool of {n} (apart from \
                     the {STATE_SNAPSHOT_SLOTS} turn-end slots; ARF_ANCHOR_SNAPSHOT_SLOTS)"
                ),
            }
        }
        self.state_snapshot_save_in(stream, key, SnapshotPool::Anchor)
    }

    /// [`state_snapshot_save`](Self::state_snapshot_save) into the JUNCTION pool (module docs,
    /// 2026-09-26).
    pub fn state_snapshot_save_junction(
        &self,
        stream: u64,
        key: u64,
    ) -> Result<Option<u64>, String> {
        static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
            match junction_snapshot_slots() {
                0 => eprintln!(
                    "[state-snapshot] junction snapshots share the anchor pool \
                     (ARF_JUNCTION_SNAPSHOT_SLOTS=0)"
                ),
                n => eprintln!(
                    "[state-snapshot] junction snapshots keep their own pool of {n} (apart from \
                     the anchors and the {STATE_SNAPSHOT_SLOTS} turn-end slots; \
                     ARF_JUNCTION_SNAPSHOT_SLOTS)"
                ),
            }
        }
        self.state_snapshot_save_in(stream, key, SnapshotPool::Junction)
    }

    fn state_snapshot_save_in(
        &self,
        stream: u64,
        key: u64,
        pool: SnapshotPool,
    ) -> Result<Option<u64>, String> {
        // A non-final prefill chunk skips the lm_head and may not have been waited on.
        self.dflash_fence();
        let row = self
            .gdn_row_of(stream)
            .ok_or("state snapshot: the sequence has no recurrent bank row")?;
        let gdn = self.gdn_snapshot_row(row);
        if gdn.is_empty() {
            return Err("state snapshot: no recurrent layers".into());
        }
        let ring = self.dflash_state.borrow().as_ref().and_then(|st| {
            let r = st.ring_of(Some(stream)).filter(|r| r.valid.get())?;
            let rd = |b: &super::island::MtlBuf| unsafe {
                std::slice::from_raw_parts(b.0.contents().as_ptr() as *const f32, b.0.length() / 4)
                    .to_vec()
            };
            Some((
                r.committed.get(),
                r.max_tok.get(),
                r.ring_k.iter().map(rd).collect(),
                r.ring_v.iter().map(rd).collect(),
            ))
        });
        // The single-pool LRU that stood here (retain the key out, evict the oldest of all at
        // STATE_SNAPSHOT_SLOTS, push) is `file_snapshot` with the pool split added.
        let mut store = self.state_snapshots.borrow_mut();
        let snap = StateSnapshot {
            key,
            tick: 0,
            pool,
            gdn,
            ring,
        };
        Ok(file_snapshot(&mut store, snap, PoolSlots::configured()))
    }

    /// Restore the snapshot stored under `key` into `stream`, a sequence that has not run yet:
    /// it is given a bank row now, so the first record finds it mapped and does not zero it.
    /// `Ok(false)` = no such snapshot (the caller must then NOT skip the prefix).
    pub fn state_snapshot_restore(
        &self,
        stream: u64,
        key: u64,
        banked: usize,
    ) -> Result<bool, String> {
        let mut store = self.state_snapshots.borrow_mut();
        let tick = store.iter().map(|s| s.tick).max().unwrap_or(0) + 1;
        let Some(snap) = store.iter_mut().find(|s| s.key == key) else {
            return Ok(false);
        };
        snap.tick = tick;
        self.dflash_fence();
        let row = self.gdn_map_streams(&[Some(stream)], banked)?[0];
        self.gdn_restore_row(row, &snap.gdn);
        if std::env::var_os("ARF_SPEC_DEBUG").is_some() {
            let after = self.gdn_snapshot_row(row);
            let same = after.len() == snap.gdn.len()
                && after
                    .iter()
                    .zip(&snap.gdn)
                    .all(|(a, b)| a.1 == b.1 && a.2 == b.2);
            eprintln!(
                "[state-snapshot] restored seq {stream} into bank row {row}; readback matches: {same}"
            );
        }
        // The restored sequence gets its own ring (a free one; `None` when the pool is full, and
        // then it does not draft). No ring saved: the draft stays silent for this sequence — it
        // claims no ring, so its commits (which do not start at 0) find none. Before C1 that case
        // invalidated THE ring, i.e. whichever live stream held it.
        if let (Some(st), Some((committed, max_tok, ks, vs))) =
            (self.dflash_state.borrow().as_ref(), &snap.ring)
        {
            let Some(r) = st.ring_claim(Some(stream)) else {
                return Ok(true);
            };
            for (dst, src) in r.ring_k.iter().zip(ks).chain(r.ring_v.iter().zip(vs)) {
                let n = src.len().min(dst.0.length() / 4);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src.as_ptr(),
                        dst.0.contents().as_ptr() as *mut f32,
                        n,
                    );
                }
            }
            r.committed.set(*committed);
            r.max_tok.set(*max_tok);
            r.valid.set(true);
        }
        Ok(true)
    }

    pub fn state_snapshot_count(&self) -> usize {
        self.state_snapshots.borrow().len()
    }

    /// The snapshot under `key` as bytes, for the on-disk prefix cache ([`snapshot_to_bytes`]).
    /// `None` = no such snapshot.
    pub fn state_snapshot_export(&self, key: u64) -> Option<Vec<u8>> {
        let store = self.state_snapshots.borrow();
        store
            .iter()
            .find(|s| s.key == key)
            .map(|s| snapshot_to_bytes(&s.gdn, &s.ring))
    }

    /// File a snapshot read back from disk under `key` in the ANCHOR pool. Its recurrent rows must
    /// match this model's layers and row lengths — a mismatch is an error, never a partial restore.
    /// Returns the key evicted to make room.
    pub fn state_snapshot_import(&self, key: u64, bytes: &[u8]) -> Result<Option<u64>, String> {
        let (gdn, ring) = snapshot_from_bytes(bytes)?;
        // sorted: the banks are a HashMap, so another process lists the layers in another order
        // (`gdn_restore_row` matches them by layer)
        let mut live = self.gdn_snapshot_shape();
        live.sort_unstable();
        let mut saved: Vec<(usize, usize, usize)> =
            gdn.iter().map(|(l, c, s)| (*l, c.len(), s.len())).collect();
        saved.sort_unstable();
        if live.is_empty() || live != saved {
            return Err(format!(
                "state snapshot import: {} recurrent layers saved, the model has {} \
                 (or their row lengths differ)",
                saved.len(),
                live.len()
            ));
        }
        let mut store = self.state_snapshots.borrow_mut();
        let snap = StateSnapshot {
            key,
            tick: 0,
            pool: SnapshotPool::Anchor,
            gdn,
            ring,
        };
        Ok(file_snapshot(&mut store, snap, PoolSlots::configured()))
    }
}

type Ring = Option<(usize, u32, Vec<Vec<f32>>, Vec<Vec<f32>>)>;

/// A snapshot's contents as little-endian bytes: the recurrent rows `(layer, conv, ssm)`, then the
/// draft ring when there is one. No header: the on-disk prefix cache's file carries the identity.
pub fn snapshot_to_bytes(gdn: &[(usize, Vec<f32>, Vec<f32>)], ring: &Ring) -> Vec<u8> {
    let mut b = Vec::new();
    let u64s = |b: &mut Vec<u8>, v: u64| b.extend_from_slice(&v.to_le_bytes());
    let f32s = |b: &mut Vec<u8>, v: &[f32]| {
        b.extend_from_slice(&(v.len() as u64).to_le_bytes());
        for x in v {
            b.extend_from_slice(&x.to_le_bytes());
        }
    };
    u64s(&mut b, gdn.len() as u64);
    for (l, c, s) in gdn {
        u64s(&mut b, *l as u64);
        f32s(&mut b, c);
        f32s(&mut b, s);
    }
    match ring {
        None => b.push(0),
        Some((committed, max_tok, ks, vs)) => {
            b.push(1);
            u64s(&mut b, *committed as u64);
            u64s(&mut b, *max_tok as u64);
            for side in [ks, vs] {
                u64s(&mut b, side.len() as u64);
                for v in side {
                    f32s(&mut b, v);
                }
            }
        }
    }
    b
}

/// A little-endian reader over a byte slice: every read is checked against the end.
struct Cursor<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self
            .at
            .checked_add(n)
            .ok_or("state snapshot import: length overflow")?;
        let s = self
            .b
            .get(self.at..end)
            .ok_or("state snapshot import: truncated")?;
        self.at = end;
        Ok(s)
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32s(&mut self) -> Result<Vec<f32>, String> {
        let n = self.u64()? as usize;
        let raw = self.take(
            n.checked_mul(4)
                .ok_or("state snapshot import: length overflow")?,
        )?;
        Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }
}

/// The inverse of [`snapshot_to_bytes`]; an error on any truncation or trailing bytes.
#[allow(clippy::type_complexity)]
pub fn snapshot_from_bytes(
    bytes: &[u8],
) -> Result<(Vec<(usize, Vec<f32>, Vec<f32>)>, Ring), String> {
    let mut c = Cursor { b: bytes, at: 0 };
    let n = c.u64()? as usize;
    let mut gdn = Vec::with_capacity(n.min(1024));
    for _ in 0..n {
        let l = c.u64()? as usize;
        let conv = c.f32s()?;
        let ssm = c.f32s()?;
        gdn.push((l, conv, ssm));
    }
    let ring = match c.take(1)?[0] {
        0 => None,
        _ => {
            let committed = c.u64()? as usize;
            let max_tok = c.u64()? as u32;
            let mut side = || -> Result<Vec<Vec<f32>>, String> {
                let m = c.u64()? as usize;
                (0..m).map(|_| c.f32s()).collect()
            };
            let ks = side()?;
            let vs = side()?;
            Some((committed, max_tok, ks, vs))
        }
    };
    if c.at != bytes.len() {
        return Err("state snapshot import: trailing bytes".into());
    }
    Ok((gdn, ring))
}

#[cfg(test)]
mod tests {
    use super::*;

    use SnapshotPool::{Anchor, Junction, Turn};

    fn snap_in(key: u64, pool: SnapshotPool) -> StateSnapshot {
        StateSnapshot {
            key,
            tick: 0,
            pool,
            gdn: Vec::new(),
            ring: None,
        }
    }

    /// An anchor (`true`) or a turn-end snapshot (`false`).
    fn snap(key: u64, anchor: bool) -> StateSnapshot {
        snap_in(key, if anchor { Anchor } else { Turn })
    }

    fn slots(turn: usize, anchor: usize) -> PoolSlots {
        PoolSlots {
            checkpoint: 2,
            turn,
            anchor,
            junction: JUNCTION_SNAPSHOT_SLOTS,
        }
    }

    fn keys(store: &[StateSnapshot], anchor: bool) -> Vec<u64> {
        pool_keys(store, if anchor { Anchor } else { Turn })
    }

    fn pool_keys(store: &[StateSnapshot], pool: SnapshotPool) -> Vec<u64> {
        let mut k: Vec<u64> = store
            .iter()
            .filter(|s| s.pool == pool)
            .map(|s| s.key)
            .collect();
        k.sort_unstable();
        k
    }

    /// THE REASON FOR THE SECOND POOL: an anchor survives a long session's turn-end saves. Ten
    /// turns evict turn-end snapshots only (each eviction reported for the scheduler to forget),
    /// and the anchor is still there to be matched by the next new session.
    #[test]
    fn anchors_survive_ten_turn_end_saves() {
        let mut store = Vec::new();
        assert_eq!(
            file_snapshot(&mut store, snap(1000, true), slots(4, 2)),
            None
        );
        let mut evicted = Vec::new();
        for turn in 0..10u64 {
            evicted.extend(file_snapshot(&mut store, snap(turn, false), slots(4, 2)));
        }
        assert_eq!(keys(&store, true), vec![1000], "the anchor survived");
        assert_eq!(
            keys(&store, false),
            vec![6, 7, 8, 9],
            "the four newest turns"
        );
        assert_eq!(
            evicted,
            vec![0, 1, 2, 3, 4, 5],
            "oldest first, anchor never"
        );

        // Control: ONE shared pool (anchor slots 0, the store before 2026-09-26) — the fifth
        // turn-end save evicts the anchor, which is the new-session miss this pool prevents.
        let mut shared = Vec::new();
        file_snapshot(&mut shared, snap(1000, true), slots(4, 0));
        let mut gone = Vec::new();
        for turn in 0..10u64 {
            gone.extend(file_snapshot(&mut shared, snap(turn, false), slots(4, 0)));
        }
        assert_eq!(gone[0], 1000, "in one pool the anchor goes first");
        assert_eq!(shared.len(), 4);
    }

    /// The anchor pool is its own LRU: a third anchor evicts the least recently USED one (a
    /// restore bumps the tick, as `state_snapshot_restore` does), and never a turn-end entry.
    #[test]
    fn the_anchor_pool_is_its_own_lru() {
        let mut store = Vec::new();
        file_snapshot(&mut store, snap(10, false), slots(4, 2));
        file_snapshot(&mut store, snap(1, true), slots(4, 2));
        file_snapshot(&mut store, snap(2, true), slots(4, 2));
        // Anchor 1 is used again (a new session resumed from it).
        let tick = store.iter().map(|s| s.tick).max().unwrap() + 1;
        store.iter_mut().find(|s| s.key == 1).unwrap().tick = tick;
        assert_eq!(
            file_snapshot(&mut store, snap(3, true), slots(4, 2)),
            Some(2)
        );
        assert_eq!(keys(&store, true), vec![1, 3]);
        assert_eq!(keys(&store, false), vec![10]);
    }

    /// One key, one entry: a re-save replaces in place and evicts nothing; a key that was ever an
    /// anchor stays in the anchor pool even when re-saved as a turn-end snapshot.
    #[test]
    fn a_resave_replaces_and_an_anchor_stays_an_anchor() {
        let mut store = Vec::new();
        for k in 0..4 {
            file_snapshot(&mut store, snap(k, false), slots(4, 2));
        }
        assert_eq!(file_snapshot(&mut store, snap(2, false), slots(4, 2)), None);
        assert_eq!(store.len(), 4);

        file_snapshot(&mut store, snap(50, true), slots(4, 2));
        assert_eq!(
            file_snapshot(&mut store, snap(50, false), slots(4, 2)),
            None
        );
        assert_eq!(keys(&store, true), vec![50]);
        assert_eq!(keys(&store, false), vec![0, 1, 2, 3]);

        // A turn-end key re-saved as an anchor moves pools: out of the turn-end pool (no
        // eviction — the key still exists) and into the anchor pool.
        assert_eq!(file_snapshot(&mut store, snap(3, true), slots(4, 2)), None);
        assert_eq!(keys(&store, true), vec![3, 50]);
        assert_eq!(keys(&store, false), vec![0, 1, 2]);
    }

    /// JUNCTIONS (2026-09-26) keep a pool of their own: learned junctions churn among themselves
    /// and never evict an anchor or a turn-end snapshot, and a key filed as a junction stays one
    /// when re-saved as a turn-end snapshot (an end-of-prompt boundary can land on it).
    #[test]
    fn junctions_have_their_own_lru_and_never_evict_an_anchor() {
        let mut store = Vec::new();
        let s = slots(4, 2);
        file_snapshot(&mut store, snap(1000, true), s);
        file_snapshot(&mut store, snap(1, false), s);
        let mut evicted = Vec::new();
        for j in 100..105u64 {
            evicted.extend(file_snapshot(&mut store, snap_in(j, Junction), s));
        }
        assert_eq!(
            evicted,
            vec![100, 101, 102],
            "junctions evict junctions, oldest first"
        );
        assert_eq!(pool_keys(&store, Junction), vec![103, 104]);
        assert_eq!(pool_keys(&store, Anchor), vec![1000]);
        assert_eq!(pool_keys(&store, Turn), vec![1]);

        // A turn-end re-save of a junction key keeps it a junction, evicting nothing.
        assert_eq!(file_snapshot(&mut store, snap(104, false), s), None);
        assert_eq!(pool_keys(&store, Junction), vec![103, 104]);
        assert_eq!(pool_keys(&store, Turn), vec![1]);
    }

    /// `ARF_JUNCTION_SNAPSHOT_SLOTS=0`: junctions are filed in the ANCHOR pool (the other design,
    /// kept as the control arm) — where they do compete with anchors. With anchors at 0 too,
    /// everything shares the turn-end LRU, as before anchors.
    #[test]
    fn zero_junction_slots_file_junctions_as_anchors() {
        let mut store = Vec::new();
        let s = PoolSlots {
            checkpoint: 2,
            turn: 4,
            anchor: 2,
            junction: 0,
        };
        file_snapshot(&mut store, snap(1000, true), s);
        file_snapshot(&mut store, snap_in(7, Junction), s);
        assert_eq!(pool_keys(&store, Anchor), vec![7, 1000]);
        assert_eq!(
            file_snapshot(&mut store, snap_in(8, Junction), s),
            Some(1000),
            "in the anchor pool a junction can evict the anchor"
        );

        let mut one = Vec::new();
        let s = PoolSlots {
            checkpoint: 2,
            turn: 4,
            anchor: 0,
            junction: 0,
        };
        file_snapshot(&mut one, snap_in(9, Junction), s);
        assert_eq!(pool_keys(&one, Turn), vec![9]);
    }

    #[test]
    fn snapshot_bytes_round_trip() {
        let gdn = vec![
            (0usize, vec![1.0f32, -2.5], vec![3.25f32; 5]),
            (4, vec![], vec![0.5]),
        ];
        let ring = Some((7usize, 99u32, vec![vec![1.0f32, 2.0]], vec![vec![-1.0f32]]));
        let b = snapshot_to_bytes(&gdn, &ring);
        let (g2, r2) = snapshot_from_bytes(&b).unwrap();
        assert_eq!(g2, gdn);
        assert_eq!(r2, ring);
        let none = snapshot_to_bytes(&gdn, &None);
        assert_eq!(snapshot_from_bytes(&none).unwrap().1, None);
        assert!(snapshot_from_bytes(&b[..b.len() - 1]).is_err(), "truncated");
        let mut extra = b.clone();
        extra.push(0);
        assert!(snapshot_from_bytes(&extra).is_err(), "trailing bytes");
    }
}
