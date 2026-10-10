//! A model of the backend's recurrent-state snapshot pools (2026-09-26), for scheduler tests that
//! need EVICTION: `crates/arf-gpu/src/gpu/metal/state_snapshots.rs` keeps a 4-slot turn-end LRU,
//! a 2-slot junction LRU and a 4-slot anchor LRU (2 before the tools anchor, 2026-09-27); a restore bumps an entry's recency; a key keeps
//! the highest pool it was filed in (`file_snapshot`). The scheduler learns of an eviction only
//! through `forget_snapshot`, exactly as the serving loop tells it (`actor.rs`).
//!
//! Shared by `tests/scheduler.rs` and `tests/scheduler_no_session_start.rs` (the opt-out needs its
//! own binary); each uses a different subset, hence the allow.
#![allow(dead_code)]

use arf_core::scheduler::{Scheduler, SeqPlan};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Pool {
    Turn,
    Junction,
    Anchor,
}

/// Default slots: `STATE_SNAPSHOT_SLOTS`, `JUNCTION_SNAPSHOT_SLOTS`, `ANCHOR_SNAPSHOT_SLOTS`.
fn slots(pool: Pool) -> usize {
    match pool {
        Pool::Turn => 4,
        Pool::Junction => 2,
        Pool::Anchor => 4,
    }
}

#[derive(Default)]
pub struct Pools {
    /// (key, pool, tick)
    entries: Vec<(u64, Pool, u64)>,
    tick: u64,
}

impl Pools {
    /// The pool the serving loop files `sp`'s snapshot in (`actor.rs`'s save routing).
    pub fn pool_of(sp: &SeqPlan) -> Pool {
        if sp.snapshot_anchor {
            Pool::Anchor
        } else if sp.snapshot_junction || sp.snapshot_session_start {
            Pool::Junction
        } else {
            Pool::Turn
        }
    }

    /// File `key` in `pool`; returns the key evicted to make room.
    pub fn save(&mut self, key: u64, mut pool: Pool) -> Option<u64> {
        if let Some(i) = self.entries.iter().position(|e| e.0 == key) {
            pool = pool.max(self.entries[i].1);
            self.entries.swap_remove(i);
        }
        self.tick += 1;
        let mut evicted = None;
        if self.entries.iter().filter(|e| e.1 == pool).count() >= slots(pool) {
            let (i, _) = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, e)| e.1 == pool)
                .min_by_key(|(_, e)| e.2)
                .unwrap();
            evicted = Some(self.entries.swap_remove(i).0);
        }
        self.entries.push((key, pool, self.tick));
        evicted
    }

    /// Restore `key` (bumps its recency). `false` = the backend no longer has it.
    pub fn restore(&mut self, key: u64) -> bool {
        self.tick += 1;
        match self.entries.iter_mut().find(|e| e.0 == key) {
            Some(e) => {
                e.2 = self.tick;
                true
            }
            None => false,
        }
    }

    pub fn pool(&self, key: u64) -> Option<Pool> {
        self.entries.iter().find(|e| e.0 == key).map(|e| e.1)
    }
}

/// Drive `id` to completion like the serving loop does: restore before the step (a miss is the
/// loop's fatal error, so it panics here), save after it into the pool the plan names, forget
/// what that evicted, then HOLD the key. Returns every plan of `id`.
pub fn run_pooled(s: &mut Scheduler, pools: &mut Pools, id: u64) -> Vec<SeqPlan> {
    let mut seen = vec![];
    while s.has_unfinished() {
        let plan = s.schedule().unwrap().expect("a plan");
        for sp in &plan.seqs {
            if let Some(k) = sp.restore_key {
                assert!(pools.restore(k), "seq {} restores {k:#x}, evicted", sp.id);
            }
        }
        for sp in &plan.seqs {
            if let Some(k) = sp.snapshot_key {
                if let Some(old) = pools.save(k, Pools::pool_of(sp)) {
                    s.forget_snapshot(old);
                }
                s.hold_snapshot(sp.id, k);
            }
        }
        seen.extend(plan.seqs.iter().filter(|sp| sp.id == id).cloned());
        s.commit_tokens(&vec![7; plan.seqs.len()]).unwrap();
    }
    seen
}

/// The snapshots `plans` took: `(boundary, pool)`.
pub fn saves(plans: &[SeqPlan]) -> Vec<(usize, Pool)> {
    plans
        .iter()
        .filter(|sp| sp.snapshot_key.is_some())
        .map(|sp| (sp.past_len + sp.q_len, Pools::pool_of(sp)))
        .collect()
}

/// Turn 1 of an agent conversation: a 1,100-token "system + tools" prefix (anchor 1,100 ->
/// 1,024) and a 2,000-token first user message (Claude Code's task + environment blocks are
/// ~2K tokens after its anchor, measured 2026-09-26). 3,100 tokens; the end-of-prompt
/// boundary is 3,099 -> 3,072, backed off one window to 2,944.
pub fn first_turn() -> Vec<u32> {
    let mut p: Vec<u32> = (0..1100).collect();
    p.extend(10_000..12_000);
    p
}

/// Turn `k` (2..=7) of the same conversation: turn `k-1`'s prompt plus 600 new tokens (a
/// thinking model's template drops the earlier answer, so the next prompt extends the last one).
pub fn next_turn(prev: &[u32], k: u32) -> Vec<u32> {
    let mut p = prev.to_vec();
    p.extend(20_000 + 1000 * k..20_000 + 1000 * k + 600);
    p
}
