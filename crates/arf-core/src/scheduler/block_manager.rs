//! Maps sequences onto physical KV blocks via a [`BlockAllocator`], with
//! optional **automatic prefix caching** .
//!
//! This is intentionally tensor-free so the scheduler can be tested without a
//! model or a GPU. Prefix caching is a pure block-bookkeeping feature: the KV
//! for a block-aligned shared prompt prefix depends only on its tokens (and
//! their absolute positions, which a leading prefix fixes), so two requests
//! whose prompts share leading blocks can share the **physical** blocks holding
//! that KV. The GPU pool is keyed by physical block id and shared blocks are
//! FULL and immutable, so nothing downstream (attention, KV scatter, sampling)
//! changes — the win is purely "prefill the shared prefix once".
//!
//! Mechanics:
//! - **Refcounts** per physical block. A block is reclaimable only at refcount 0.
//! - **Content hash → block** map of registered FULL blocks. A block's hash
//!   chains the previous block's hash, so the key encodes the whole prefix; the
//!   block's tokens are stored and verified on a hit (no silent hash-collision
//!   reuse).
//! - **Eviction**: a cached block that drops to refcount 0 is retained (its KV
//!   is still valid) on an approximate-LRU `evictable` queue and only physically
//!   reclaimed when the free list is empty and a new block is needed.

use std::collections::{HashMap, VecDeque};

use crate::cache::{blocks_needed, BlockAllocator};
use crate::error::Result;
use crate::scheduler::request::{Sequence, PREFIX_HASH_SEED};

/// Owns the block free-list and grows/frees sequences' block tables, plus the
/// optional prefix cache.
#[derive(Debug, Clone)]
pub struct BlockManager {
    allocator: BlockAllocator,
    block_size: usize,
    /// Live references per physical block; 0 = not held by any running sequence.
    refcount: Vec<u32>,
    /// Whether automatic prefix caching is active.
    prefix_enabled: bool,
    /// Registered FULL blocks: chained content hash → physical block id.
    cached: HashMap<u64, u32>,
    /// Reverse map: physical block → its registered hash (for cleanup on evict).
    hash_of: Vec<Option<u64>>,
    /// The tokens of each registered block, verified on a cache hit so a hash
    /// collision can never reuse the wrong KV. Empty for non-cached blocks.
    block_tokens: Vec<Vec<u32>>,
    /// Cached blocks at refcount 0, oldest at the front (approximate LRU).
    evictable: VecDeque<u32>,
    /// Stat: total prefix tokens served from cache (blocks reused × block_size).
    reused_tokens: u64,
    /// STATE SNAPSHOTS — `Some` on a model with recurrent layers. A prefix hit there is only
    /// usable where the BACKEND holds a snapshot of the recurrent state (a cached KV block says
    /// nothing about the 48 layers that have no KV), so `match_prefix` stops at the deepest
    /// block boundary whose chained hash is in this set. Keys are added by the serving loop
    /// after the backend saved one, removed when it evicted one.
    snapshots: Option<std::collections::HashSet<u64>>,
    /// See `set_decode_lookahead`. 0 = off.
    lookahead: usize,
    lookahead_reserve: usize,
    /// See `set_cache_soft_cap`. `usize::MAX` = off.
    soft_cap_blocks: usize,
}

impl BlockManager {
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        Self::with_prefix_cache(num_blocks, block_size, false)
    }

    /// Build a manager, optionally enabling automatic prefix caching.
    pub fn with_prefix_cache(num_blocks: usize, block_size: usize, prefix_enabled: bool) -> Self {
        BlockManager {
            allocator: BlockAllocator::new(num_blocks),
            block_size,
            refcount: vec![0; num_blocks],
            prefix_enabled,
            cached: HashMap::new(),
            hash_of: vec![None; num_blocks],
            block_tokens: vec![Vec::new(); num_blocks],
            evictable: VecDeque::new(),
            reused_tokens: 0,
            snapshots: None,
            lookahead: 0,
            lookahead_reserve: 0,
            soft_cap_blocks: usize::MAX,
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Reclaimable blocks: the free list PLUS cached-but-unreferenced blocks
    /// (those can be evicted on demand). This is what admission/growth budget
    /// against, and what `StepInfo`'s "free" count reports.
    pub fn num_free_blocks(&self) -> usize {
        self.allocator.num_free() + self.evictable.len()
    }

    /// Blocks pinned by live sequences (cannot be reclaimed without preemption).
    pub fn num_used_blocks(&self) -> usize {
        self.total_blocks() - self.num_free_blocks()
    }

    pub fn total_blocks(&self) -> usize {
        self.allocator.total()
    }

    /// Total prefix tokens served from the cache so far (a measurement hook).
    pub fn reused_tokens(&self) -> u64 {
        self.reused_tokens
    }

    /// Gate prefix hits on recurrent-state snapshots (see the `snapshots` field).
    pub fn require_state_snapshots(&mut self) {
        self.snapshots.get_or_insert_with(Default::default);
    }

    pub fn note_snapshot(&mut self, key: u64) {
        if let Some(s) = self.snapshots.as_mut() {
            s.insert(key);
        }
    }

    pub fn forget_snapshot(&mut self, key: u64) {
        if let Some(s) = self.snapshots.as_mut() {
            s.remove(&key);
        }
    }

    /// Whether a recurrent-state snapshot under `key` is PUBLISHED (matchable). A snapshot the
    /// backend holds for a still-running sequence is not in this set yet — see
    /// `Sequence::pending_snapshot`.
    pub fn has_snapshot(&self, key: u64) -> bool {
        self.snapshots.as_ref().is_some_and(|s| s.contains(&key))
    }

    /// ON-DISK PREFIX CACHE (2026-10-06): the slots of `tokens`' first `blocks` full
    /// blocks, in position order, when every one of them is a registered cached block holding
    /// exactly those tokens — the KV a save reads. `None` when any block is gone or not published.
    pub fn cached_prefix_slots(&self, tokens: &[u32], blocks: usize) -> Option<Vec<u32>> {
        let bs = self.block_size;
        if tokens.len() < blocks * bs {
            return None;
        }
        let mut h = PREFIX_HASH_SEED;
        let mut slots = Vec::with_capacity(blocks * bs);
        for b in 0..blocks {
            let toks = &tokens[b * bs..(b + 1) * bs];
            h = block_hash(h, toks);
            let phys = *self.cached.get(&h)?;
            if self.block_tokens[phys as usize] != toks {
                return None;
            }
            slots.extend((0..bs as u32).map(|i| phys * bs as u32 + i));
        }
        Some(slots)
    }

    /// Blocks for a prefix read back from disk: `seq` (a sequence of exactly `blocks` full
    /// blocks of tokens) is given fresh blocks, unregistered, and their slots are returned. The
    /// caller writes the KV into those slots and then calls [`import_commit`](Self::import_commit)
    /// — or [`free`](Self::free) when the write failed, which returns the blocks unpublished.
    pub fn import_begin(&mut self, seq: &mut Sequence) -> Result<Vec<u32>> {
        let bs = self.block_size;
        let blocks = seq.len() / bs;
        let saved = self.lookahead;
        self.lookahead = 0; // exactly the prefix's blocks, no decode lookahead
        let r = self.allocate(seq);
        self.lookahead = saved;
        r?;
        Ok(seq.block_table[..blocks]
            .iter()
            .flat_map(|&p| (0..bs as u32).map(move |i| p * bs as u32 + i))
            .collect())
    }

    /// The KV of `seq`'s blocks is written: publish them (cached, at refcount 0 on the evictable
    /// queue, exactly as a finished sequence leaves its prompt) and the snapshot `key` at their
    /// end. A block whose hash another sequence already published stays private and the chain
    /// stops there, so a later match never reaches `key` — no sharing, never a wrong one.
    pub fn import_commit(&mut self, seq: &mut Sequence, key: u64) {
        seq.num_computed = seq.len();
        let n = seq.block_table.len();
        self.register_blocks_upto(seq, n);
        let complete = seq.num_cached_blocks * self.block_size == seq.len()
            && self
                .cached_prefix_slots(&seq.tokens, seq.num_cached_blocks)
                .is_some();
        self.free(seq);
        if complete {
            self.note_snapshot(key);
        }
    }

    /// The chained content hash of `tokens`' first `blocks` full blocks — the key a snapshot
    /// taken at position `blocks * block_size` is stored under, and the value `match_prefix`
    /// reaches at that boundary.
    pub fn chain_hash(&self, tokens: &[u32], blocks: usize) -> u64 {
        let bs = self.block_size;
        (0..blocks).fold(PREFIX_HASH_SEED, |h, b| {
            block_hash(h, &tokens[b * bs..(b + 1) * bs])
        })
    }

    /// DECODE LOOKAHEAD — back `tokens` positions PAST the sequence's end, best effort.
    ///
    /// A speculative verify window writes K/V for up to 8 positions at once, but the table is
    /// sized for the ONE token a step commits, so a window was cut at the end of the current
    /// 16-slot block: measured 2026-09-20 on the block draft, 69 of 168 windows cut, 2.6 tokens a
    /// cycle against 3.4 on the uncut ones. With a lookahead the next block is taken early.
    /// BEST EFFORT: never part of `can_allocate` (admission and preemption see only what a
    /// sequence truly needs), and only taken while more than `reserve` blocks are free, so under
    /// pressure speculation gets shorter windows, exactly as before, and nothing else changes.
    pub fn set_decode_lookahead(&mut self, tokens: usize, reserve: usize) {
        self.lookahead = tokens;
        self.lookahead_reserve = reserve;
    }

    /// CACHED-KV SOFT CAP (2026-09-27) — keep the prefix cache inside the first `blocks` block ids.
    ///
    /// The Metal KV pool maps memory up to the highest slot ever written and never unmaps it
    /// (`sparse_kv.rs`). Cached blocks at refcount 0 were reclaimed only when the free list ran
    /// dry, so every new session took fresh, ever-higher ids while the old sessions' KV sat in the
    /// cache: measured 2026-09-27, 24 distinct agent-style sessions (~3.5K-token prompts) grew the
    /// server's footprint 6.50 -> 9.54 GB in +545 MB steps, still climbing, the device at 30.1 of
    /// its 30.2 GB working set — and it would have kept going to the whole 8.1 GB KV budget.
    /// With a cap, a block whose next free id is at or past it is taken from the oldest cached
    /// block instead, while any is reclaimable; with none, the pool grows as before (live
    /// sequences are never refused by this). Admission, preemption and `num_free_blocks` are
    /// unchanged — cached blocks already counted as free. What it costs: an old cached prefix is
    /// evicted sooner (LRU, oldest first). On a hybrid model that matters little — a hit needs a
    /// recurrent-state snapshot, and only ~10 are kept.
    pub fn set_cache_soft_cap(&mut self, blocks: usize) {
        self.soft_cap_blocks = if blocks == 0 { usize::MAX } else { blocks };
    }

    /// Blocks the sequence still needs so its table covers all its tokens.
    pub fn additional_blocks_needed(&self, seq: &Sequence) -> usize {
        let want = blocks_needed(seq.len(), self.block_size);
        want.saturating_sub(seq.block_table.len())
    }

    /// Whether `seq` can be grown to cover all its tokens right now.
    pub fn can_allocate(&self, seq: &Sequence) -> bool {
        self.additional_blocks_needed(seq) <= self.num_free_blocks()
    }

    /// Grow `seq`'s block table to cover all its tokens. No-op if already
    /// covered. New blocks come from the free list first, then by evicting the
    /// oldest reclaimable cached block.
    pub fn allocate(&mut self, seq: &mut Sequence) -> Result<()> {
        let need = self.additional_blocks_needed(seq);
        for _ in 0..need {
            let phys = self.acquire_block()?;
            seq.block_table.push(phys);
        }
        if self.lookahead > 0 {
            let want = blocks_needed(seq.len() + self.lookahead, self.block_size);
            while seq.block_table.len() < want && self.num_free_blocks() > self.lookahead_reserve {
                let Ok(phys) = self.acquire_block() else {
                    break;
                };
                seq.block_table.push(phys);
            }
        }
        Ok(())
    }

    /// Free all blocks held by `seq` and clear its block table. A block whose
    /// refcount hits 0 is returned to the free list, unless it is a registered
    /// cached block — that is retained (KV still valid) on the evictable queue.
    pub fn free(&mut self, seq: &mut Sequence) {
        for &b in &seq.block_table {
            let i = b as usize;
            debug_assert!(self.refcount[i] > 0, "freeing a block with refcount 0");
            self.refcount[i] -= 1;
            if self.refcount[i] == 0 {
                if self.hash_of[i].is_some() {
                    self.evictable.push_back(b);
                } else {
                    self.allocator.free([b]);
                }
            }
        }
        seq.block_table.clear();
        seq.num_cached_blocks = 0;
        seq.prefix_hash = PREFIX_HASH_SEED;
    }

    /// Acquire one fresh physical block (refcount set to 1), preferring the free
    /// list and falling back to evicting the oldest reclaimable cached block.
    fn acquire_block(&mut self) -> Result<u32> {
        // Past the soft cap, reuse the oldest cached block rather than map a new id (see
        // `set_cache_soft_cap`).
        let past_cap = self
            .allocator
            .lowest_free()
            .is_some_and(|b| b as usize >= self.soft_cap_blocks)
            && !self.evictable.is_empty();
        let phys = if self.allocator.num_free() > 0 && !past_cap {
            self.allocator.allocate(1)?[0]
        } else {
            // Evict the oldest cached block and repurpose it.
            let b = self
                .evictable
                .pop_front()
                .ok_or(crate::error::ArfError::OutOfBlocks {
                    requested: 1,
                    free: 0,
                })?;
            self.unregister(b);
            b
        };
        self.refcount[phys as usize] = 1;
        Ok(phys)
    }

    /// Drop a physical block from the prefix cache (its KV is being repurposed).
    fn unregister(&mut self, phys: u32) {
        let i = phys as usize;
        if let Some(h) = self.hash_of[i].take() {
            self.cached.remove(&h);
        }
        self.block_tokens[i].clear();
    }

    fn drop_from_evictable(&mut self, phys: u32) {
        if let Some(pos) = self.evictable.iter().position(|&b| b == phys) {
            self.evictable.remove(pos);
        }
    }

    /// Match `seq`'s prompt against the prefix cache, reusing as many leading
    /// FULL blocks as hit (and verify). Sets `block_table`, `num_computed`,
    /// `num_cached_blocks`, and `prefix_hash` to the matched prefix. At least one
    /// token is always left uncached so the step has a query to forward. No-op
    /// when prefix caching is disabled or the sequence has already started.
    /// With state snapshots required, also sets `kv_match_len` — the KV hit before
    /// the backoff to a snapshot boundary.
    /// Where `tokens` would resume from the cache now, without claiming anything: the deepest
    /// block boundary on its chain that is cached and, with state snapshots, also snapshotted —
    /// the point [`match_prefix`](Self::match_prefix) would leave `num_computed` at.
    pub fn resume_point(&self, tokens: &[u32]) -> usize {
        if !self.prefix_enabled {
            return 0;
        }
        let bs = self.block_size;
        let (mut h, mut deepest) = (PREFIX_HASH_SEED, 0usize);
        for b in 0..tokens.len().saturating_sub(1) / bs {
            let toks = &tokens[b * bs..(b + 1) * bs];
            h = block_hash(h, toks);
            match self.cached.get(&h) {
                Some(&phys) if self.block_tokens[phys as usize].as_slice() == toks => {
                    if self.snapshots.as_ref().is_none_or(|s| s.contains(&h)) {
                        deepest = b + 1;
                    }
                }
                _ => break,
            }
        }
        deepest * bs
    }

    pub fn match_prefix(&mut self, seq: &mut Sequence) {
        if !self.prefix_enabled || seq.num_computed != 0 || !seq.block_table.is_empty() {
            return;
        }
        // An IMAGE sequence never claims cached blocks (2026-09-27). The block hash is over token
        // ids, and every image of the same grid is the same run of `<|image_pad|>` ids — a second
        // request with a DIFFERENT picture would match the first one's KV and see its pixels.
        // (`Request::image`'s doc already promised this; nothing enforced it.)
        if seq.image.is_some() {
            return;
        }
        let bs = self.block_size;
        // Keep ≥1 pending token: never cache the block holding the last token.
        let mut max_blocks = seq.len().saturating_sub(1) / bs;
        if let Some(snaps) = self.snapshots.as_ref() {
            // Read-only pass: how deep does the chain hit, and what is the deepest boundary on
            // it that ALSO has a recurrent-state snapshot? Claim blocks only up to there.
            let (mut h, mut deepest, mut hit) = (PREFIX_HASH_SEED, 0usize, 0usize);
            for b in 0..max_blocks {
                let toks = &seq.tokens[b * bs..(b + 1) * bs];
                h = block_hash(h, toks);
                match self.cached.get(&h) {
                    Some(&phys) if self.block_tokens[phys as usize].as_slice() == toks => {
                        hit = b + 1;
                        if snaps.contains(&h) {
                            deepest = b + 1;
                        }
                    }
                    _ => break,
                }
            }
            // JUNCTION SNAPSHOTS (2026-09-26): the chain hit BEFORE the backoff — verified
            // blocks, token for token, that the cache holds right now. The scheduler plans a
            // snapshot where it ends (`Sequence::junction_at`). Blocks past `deepest` are NOT
            // claimed and may be evicted later; nothing reads them — the junction's state and
            // KV are this sequence's own prefill.
            seq.kv_match_len = hit * bs;
            max_blocks = deepest;
        }
        let mut h = PREFIX_HASH_SEED;
        for b in 0..max_blocks {
            let toks = &seq.tokens[b * bs..(b + 1) * bs];
            h = block_hash(h, toks);
            match self.cached.get(&h) {
                Some(&phys) if self.block_tokens[phys as usize].as_slice() == toks => {
                    if self.refcount[phys as usize] == 0 {
                        self.drop_from_evictable(phys);
                    }
                    self.refcount[phys as usize] += 1;
                    seq.block_table.push(phys);
                    seq.num_cached_blocks += 1;
                    seq.num_computed += bs;
                    seq.prefix_hash = h;
                    self.reused_tokens += bs as u64;
                }
                _ => break,
            }
        }
        if self.snapshots.is_some() && seq.num_cached_blocks > 0 {
            seq.restore_key = Some(seq.prefix_hash);
        }
    }

    /// Undo a [`match_prefix`](Self::match_prefix) when the sequence ultimately
    /// can't be admitted this step — release the shared refs and reset state so
    /// the sequence returns to the waiting queue exactly as it arrived.
    pub fn unmatch_prefix(&mut self, seq: &mut Sequence) {
        // free() already decrements refcounts and (since these are cached blocks)
        // pushes them back to evictable, then resets the bookkeeping fields.
        self.reused_tokens -= (seq.num_cached_blocks * self.block_size) as u64;
        self.free(seq);
        seq.num_computed = 0;
        seq.restore_key = None;
        seq.kv_match_len = 0;
    }

    /// Register any leading FULL blocks of `seq` that just became complete (by
    /// committed `num_computed`) into the prefix cache, so later requests with
    /// the same prefix can reuse them. Incremental: resumes from the sequence's
    /// `num_cached_blocks`/`prefix_hash`. No-op when prefix caching is disabled.
    pub fn register_full_blocks(&mut self, seq: &mut Sequence) {
        if !self.prefix_enabled {
            return;
        }
        if seq.num_computed < seq.tokens.len() {
            return;
        }
        let full_blocks = seq.num_computed / self.block_size;
        self.register_blocks_upto(seq, full_blocks);
    }

    /// Register `seq`'s leading blocks up to `n_blocks` (and no further than its committed
    /// `num_computed`) into the prefix cache, resuming from `num_cached_blocks`. WITHOUT the
    /// completed-prompt guard of [`register_full_blocks`](Self::register_full_blocks): only for
    /// blocks the sequence will never write again — an anchor's blocks, wholly before its
    /// window-aligned boundary (`Scheduler::publish_early_anchors`, 2026-10-05).
    pub fn register_blocks_upto(&mut self, seq: &mut Sequence, n_blocks: usize) {
        if !self.prefix_enabled {
            return;
        }
        // ⚠️ L125 — PUBLISH-AFTER-PREFILL. Do NOT expose a sequence's blocks while it is still
        // prefilling. Prefill is CHUNKED, so a 940-token prompt
        // reaches this function after chunk 1 with `num_computed` covering blocks the sequence
        // will keep writing during chunks 2..n. Publishing them lets a later request
        // `match_prefix` onto memory that is still a live write target: the two sequences then
        // share physical blocks whose KV is not final, and the REGISTRAR reads back clobbered KV
        // and emits fluent garbage.
        //
        // That is exactly the L123 signature: corruption only COLD (the registrar must still be
        // prefilling when another request attaches — warm runs are clean because the prefix was
        // finalized by an earlier, completed request), and the victim is always the registrar
        // (agent 1), never the attacher.
        //
        // Registration is not lost, only deferred: the sequence calls this again on the step that
        // completes its prompt, and every full block is published then — the loop below resumes
        // from `num_cached_blocks`, so a deferred publish is exactly as complete as an eager one.
        // Cost: a cold burst of identical prompts shares nothing on its FIRST wave (each request
        // prefills its own copy, as it must — the KV genuinely does not exist yet) and shares
        // everything from the second wave on, which is where the advantage actually pays.
        let bs = self.block_size;
        let full_blocks = n_blocks.min(seq.num_computed / bs);
        let mut h = seq.prefix_hash;
        for b in seq.num_cached_blocks..full_blocks {
            let toks = &seq.tokens[b * bs..(b + 1) * bs];
            h = block_hash(h, toks);
            let phys = seq.block_table[b];
            let i = phys as usize;
            // Register only if this block isn't already cached and the hash is
            // free. A lost race (another seq registered the same prefix first)
            // leaves this block private — correct, just no sharing gained.
            if self.hash_of[i].is_none() && !self.cached.contains_key(&h) {
                self.hash_of[i] = Some(h);
                self.block_tokens[i] = toks.to_vec();
                self.cached.insert(h, phys);
            }
            seq.num_cached_blocks = b + 1;
            seq.prefix_hash = h;
        }
    }
}

/// Chained FNV-1a-style content hash: block `b`'s hash mixes the previous
/// block's hash with this block's tokens, so the value encodes the whole prefix.
fn block_hash(prev: u64, tokens: &[u32]) -> u64 {
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = prev ^ 0x9e37_79b9_7f4a_7c15; // mix prev so chaining is non-trivial
    h = h.wrapping_mul(FNV_PRIME);
    for &t in tokens {
        h ^= t as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

#[cfg(test)]
mod l124_aliasing_tests {
    use super::*;
    use crate::scheduler::request::{Request, Sequence};

    fn seq_with(tokens: Vec<u32>) -> Sequence {
        Sequence::from_request(Request {
            id: 0,
            prompt: tokens,
            params: Default::default(),
            image: None,
            prefix_anchor: None,
            tools_anchor: None,
            header_tail: None,
            stop_check: None,
        })
    }

    /// A prefix read back from disk is published like a finished prompt: its blocks match, its
    /// slots read back the same, and its snapshot key is matched at its end.
    #[test]
    fn an_imported_prefix_matches_like_a_prefilled_one() {
        let mut bm = BlockManager::with_prefix_cache(64, 16, true);
        bm.require_state_snapshots();
        let toks: Vec<u32> = (0..64).collect();
        let key = bm.chain_hash(&toks, 4);
        assert_eq!(bm.cached_prefix_slots(&toks, 4), None);
        let mut s = seq_with(toks.clone());
        let slots = bm.import_begin(&mut s).unwrap();
        assert_eq!(slots.len(), 64);
        bm.import_commit(&mut s, key);
        assert!(bm.has_snapshot(key));
        assert_eq!(bm.cached_prefix_slots(&toks, 4), Some(slots));
        let mut later: Vec<u32> = toks.clone();
        later.extend(100..120);
        let mut s2 = seq_with(later);
        bm.match_prefix(&mut s2);
        assert_eq!(s2.restore_key, Some(key));
    }

    /// The lookahead backs positions past the sequence's end, is never counted by
    /// `can_allocate`, and steps aside when blocks are scarce.
    #[test]
    fn decode_lookahead_is_best_effort() {
        let mut bm = BlockManager::new(8, 16);
        bm.set_decode_lookahead(8, 2);
        // 10 tokens need 1 block; 10 + 8 = 18 positions need 2.
        let mut a = seq_with(vec![7; 10]);
        assert_eq!(bm.additional_blocks_needed(&a), 1);
        bm.allocate(&mut a).unwrap();
        assert_eq!(a.block_table.len(), 2);
        // 5 tokens + 8 fits in one block: no extra.
        let mut b = seq_with(vec![7; 5]);
        bm.allocate(&mut b).unwrap();
        assert_eq!(b.block_table.len(), 1);
        // Scarce: 5 free, this needs 4 -> 1 left, under the reserve of 2, so no lookahead block.
        let mut c = seq_with(vec![7; 16 * 3 + 10]);
        bm.allocate(&mut c).unwrap();
        assert_eq!(c.block_table.len(), 4);
        assert_eq!(bm.num_free_blocks(), 1);
        for s in [&mut a, &mut b, &mut c] {
            bm.free(s);
        }
        assert_eq!(bm.num_free_blocks(), 8);
    }

    /// L124 — THE PREFIX-CACHE ALIASING BUG (found via garbage output under concurrency, L123).
    ///
    /// A cached block sits on `evictable` only while `refcount == 0`. `free()` pushes it there
    /// UNCONDITIONALLY on the 0-transition with no membership check, so a block that cycles
    /// free -> match -> free can be pushed TWICE. `acquire_block()` then pops the two copies on
    /// two different allocations and hands the SAME physical block to two live sequences, setting
    /// `refcount = 1` each time (an assignment, not an increment).
    ///
    /// Result: two sequences write KV into one block. No error is possible — the id is valid and
    /// in bounds — so the victim just reads someone else's KV and emits fluent garbage.
    /// The cached-KV soft cap (`set_cache_soft_cap`): a new session reuses an old session's
    /// cached blocks instead of taking fresh ids past the cap — the pool's mapped memory stops
    /// climbing — while a live sequence that needs more than the cap still gets it.
    #[test]
    fn soft_cap_reuses_cached_blocks_before_growing_past_it() {
        let bs = 4;
        let run = |cap: usize| {
            let mut bm = BlockManager::with_prefix_cache(16, bs, true);
            bm.set_cache_soft_cap(cap);
            // Session A: 4 full blocks, registered, then finished (cached at refcount 0).
            let mut a = seq_with((0..17).collect());
            bm.match_prefix(&mut a);
            bm.allocate(&mut a).unwrap();
            a.num_computed = a.tokens.len();
            bm.register_full_blocks(&mut a);
            let a_ids = a.block_table.clone();
            bm.free(&mut a);
            // Session B: different tokens, same size.
            let mut b = seq_with((500..517).collect());
            bm.match_prefix(&mut b);
            bm.allocate(&mut b).unwrap();
            (bm, a_ids, b)
        };
        // No cap (0 = off): B takes fresh ids past A's (A's partial last block was not cached,
        // so B gets that one back first), and A's four full blocks stay cached.
        let (_, a_ids, b) = run(0);
        let a_max = *a_ids.iter().max().unwrap();
        assert!(
            b.block_table.iter().any(|&x| x > a_max),
            "no cap: {:?}",
            b.block_table
        );
        // Cap of 5 blocks: A took ids 0..=4, so B's first block would be id 5 (past the cap) —
        // it reuses A's cached blocks instead and never goes past A's high-water mark.
        let (mut bm, a_ids, mut b) = run(5);
        let a_max = *a_ids.iter().max().unwrap();
        assert!(
            b.block_table.iter().all(|&x| x <= a_max),
            "cap: {:?}",
            b.block_table
        );
        // A live sequence that needs more than the cap still gets it once nothing is cached.
        bm.free(&mut b);
        let mut c = seq_with((900..957).collect()); // 57 tokens -> 15 blocks
        bm.match_prefix(&mut c);
        bm.allocate(&mut c).unwrap();
        assert_eq!(c.block_table.len(), 15);
        assert!(c.block_table.iter().any(|&x| x >= 5));
    }

    #[test]
    fn a_block_is_never_queued_for_eviction_twice() {
        let bs = 4;
        let mut bm = BlockManager::with_prefix_cache(8, bs, true);

        // One shared 2-block prefix, distinct tails — the agent-swarm shape.
        let prefix: Vec<u32> = (100..108).collect();
        let mk = |tail: u32| {
            let mut t = prefix.clone();
            t.extend_from_slice(&[tail, tail + 1, tail + 2, tail + 3, tail + 4]);
            seq_with(t)
        };

        // Seq A: prefill the prefix and register it.
        let mut a = mk(1);
        bm.match_prefix(&mut a);
        bm.allocate(&mut a).unwrap();
        a.num_computed = a.tokens.len(); // prefill COMPLETE (L125 publishes only then)
        bm.register_full_blocks(&mut a);
        bm.free(&mut a);

        // Cycle the shared blocks through match -> free several times. Each free() that lands on
        // refcount 0 pushes again; nothing dedups.
        for tail in 2..6 {
            let mut s = mk(tail);
            bm.match_prefix(&mut s);
            bm.allocate(&mut s).unwrap();
            bm.free(&mut s);
        }

        // INVARIANT: `evictable` is a set, not a bag. A duplicate means the same physical block
        // can be handed out twice while both holders believe they own it exclusively.
        let mut seen = std::collections::HashSet::new();
        let dupes: Vec<u32> = bm
            .evictable
            .iter()
            .copied()
            .filter(|b| !seen.insert(*b))
            .collect();
        assert!(
            dupes.is_empty(),
            "block(s) {dupes:?} queued for eviction more than once — acquire_block() will hand \
             the same physical block to two live sequences (evictable = {:?})",
            bm.evictable
        );
    }

    /// L124 — THE INVARIANT THAT ACTUALLY MATTERS: no physical block may appear in two LIVE
    /// sequences' block tables unless it is a genuinely SHARED cached prefix block (refcount
    /// tracks the sharers). A block handed out by `acquire_block()` is PRIVATE and must appear
    /// exactly once.
    ///
    /// `acquire_block()` pops from `evictable` and then does `refcount[phys] = 1` — an
    /// ASSIGNMENT. If the popped block still has live readers (refcount > 0), that assignment
    /// silently erases them and the block is now written by a new sequence while others read it.
    #[test]
    fn acquire_never_steals_a_block_that_still_has_readers() {
        let bs = 4;
        let n = 6; // deliberately tight: force eviction
        let mut bm = BlockManager::with_prefix_cache(n, bs, true);
        let prefix: Vec<u32> = (300..308).collect();
        let mk = |tail: u32| {
            let mut t = prefix.clone();
            t.extend_from_slice(&[tail, tail + 1, tail + 2, tail + 3, tail + 4]);
            seq_with(t)
        };

        // Register the shared prefix, then release it so it sits on `evictable` with rc 0.
        let mut a = mk(1);
        bm.match_prefix(&mut a);
        bm.allocate(&mut a).unwrap();
        a.num_computed = a.tokens.len(); // prefill COMPLETE — required to publish (L125)
        bm.register_full_blocks(&mut a);
        bm.free(&mut a);

        // Two LIVE sequences now share that prefix (refcount > 0 on those blocks).
        let mut b = mk(20);
        bm.match_prefix(&mut b);
        bm.allocate(&mut b).unwrap();
        let mut c = mk(40);
        bm.match_prefix(&mut c);
        bm.allocate(&mut c).unwrap();

        let shared: Vec<u32> = b.block_table[..b.num_cached_blocks].to_vec();
        assert!(
            !shared.is_empty(),
            "test setup: expected a shared prefix hit"
        );

        // Drain the pool so the next allocation MUST evict.
        let mut hogs = Vec::new();
        while bm.num_free_blocks() > 0 {
            let mut h = seq_with(vec![7000 + hogs.len() as u32; bs + 1]);
            // Under L125's publish-after-prefill rule fewer blocks are recyclable, so running dry
            // here is the CORRECT outcome — the pool refuses to over-commit rather than stealing.
            if bm.allocate(&mut h).is_err() {
                break;
            }
            hogs.push(h);
        }

        // Any block still referenced by the live sequences b/c must NOT have been handed to a hog.
        for h in &hogs {
            for blk in &h.block_table {
                assert!(
                    !shared.contains(blk),
                    "block {blk} is held by LIVE sequences b/c (refcount>0) yet was re-issued to a \
                     new sequence — acquire_block() stole a block that still has readers"
                );
            }
        }
    }

    /// `num_free_blocks()` is what `can_allocate()` gates admission on. It counts
    /// `evictable.len()`, so a duplicated entry inflates the count and lets the scheduler admit a
    /// sequence there is no real capacity for — over-admission on top of the aliasing.
    #[test]
    fn free_block_count_never_exceeds_the_pool() {
        let bs = 4;
        let n = 8;
        let mut bm = BlockManager::with_prefix_cache(n, bs, true);
        let prefix: Vec<u32> = (200..208).collect();
        for tail in 0..6u32 {
            let mut t = prefix.clone();
            t.extend_from_slice(&[900 + tail, 901 + tail, 902 + tail, 903 + tail, 904 + tail]);
            let mut s = seq_with(t);
            bm.match_prefix(&mut s);
            bm.allocate(&mut s).unwrap();
            s.num_computed = s.tokens.len(); // prefill COMPLETE (L125)
            bm.register_full_blocks(&mut s);
            bm.free(&mut s);
            assert!(
                bm.num_free_blocks() <= n,
                "num_free_blocks() = {} exceeds the {n}-block pool — evictable holds duplicates \
                 ({:?}), so admission control will over-commit",
                bm.num_free_blocks(),
                bm.evictable
            );
        }
    }
}

#[cfg(test)]
mod l125_registration_race_tests {
    use super::*;
    use crate::scheduler::request::{Request, Sequence};

    fn seq_with(tokens: Vec<u32>) -> Sequence {
        Sequence::from_request(Request {
            id: 0,
            prompt: tokens,
            params: Default::default(),
            image: None,
            prefix_anchor: None,
            tools_anchor: None,
            header_tail: None,
            stop_check: None,
        })
    }

    /// L125 — THE REGISTRATION RACE (L123's corruption; COLD-only, victim = the registrar).
    ///
    /// `register_full_blocks` publishes a sequence's completed blocks the moment `num_computed`
    /// covers them — INCLUDING while that sequence is still mid-prefill (prefill is chunked, so a
    /// 940-token prompt registers blocks after chunk 1 and keeps writing during chunks 2..n).
    ///
    /// A later request may then `match_prefix` onto those blocks and become a co-owner of memory
    /// the registrar is STILL WRITING. Sharing FINAL KV is safe; sharing KV that is still being
    /// produced is not.
    ///
    /// This asserts the invariant that makes it safe: **a block may only be published once the
    /// sequence that owns it has finished prefilling it** — i.e. registration must never expose a
    /// block belonging to a sequence whose prefill is incomplete.
    #[test]
    fn a_mid_prefill_sequence_never_publishes_its_blocks() {
        let bs = 4;
        let mut bm = BlockManager::with_prefix_cache(32, bs, true);

        // A 20-token prompt = 5 blocks; prefill it in CHUNKS like the real scheduler does.
        let prompt: Vec<u32> = (500..520).collect();
        let mut a = seq_with(prompt.clone());
        bm.match_prefix(&mut a);
        bm.allocate(&mut a).unwrap();

        // Chunk 1: only the first 8 tokens (2 blocks) are actually computed on the GPU.
        a.num_computed = 8;
        bm.register_full_blocks(&mut a);

        // A is STILL PREFILLING (num_computed 8 < prompt 20). Nothing it owns may be visible to
        // another sequence yet — those blocks are live write targets.
        let mut b = seq_with(prompt.clone());
        bm.match_prefix(&mut b);

        assert_eq!(
            b.num_cached_blocks,
            0,
            "sequence B attached to {} block(s) of a sequence that is STILL PREFILLING \
             (num_computed {} < len {}). B now co-owns memory A is still writing — A's KV gets \
             clobbered and A (the registrar) emits garbage. This is the L123 corruption.",
            b.num_cached_blocks,
            a.num_computed,
            a.tokens.len()
        );
    }
}

#[cfg(test)]
mod junction_match_tests {
    use super::*;
    use crate::scheduler::request::{Request, Sequence};

    fn seq_with(tokens: Vec<u32>) -> Sequence {
        Sequence::from_request(Request::new(0, tokens, Default::default()))
    }

    /// JUNCTION SNAPSHOTS (2026-09-26): on a hybrid model the match backs off to the deepest
    /// snapshot, and `kv_match_len` still reports how far the cached KV itself matched — the
    /// input the scheduler places a junction from. Only VERIFIED blocks count: the hit stops at
    /// the first block that differs.
    #[test]
    fn kv_match_len_is_the_hit_before_the_snapshot_backoff() {
        let bs = 4;
        let mut bm = BlockManager::with_prefix_cache(64, bs, true);
        bm.require_state_snapshots();
        // A: 40 tokens, fully prefilled and registered (10 blocks); a snapshot at block 2.
        let a_toks: Vec<u32> = (0..40).collect();
        let mut a = seq_with(a_toks.clone());
        bm.match_prefix(&mut a);
        bm.allocate(&mut a).unwrap();
        a.num_computed = a.tokens.len();
        bm.register_full_blocks(&mut a);
        bm.note_snapshot(bm.chain_hash(&a_toks, 2));
        bm.free(&mut a);

        // B shares A's first 26 tokens: blocks 0..6 hit (24 tokens), block 6 differs at 26.
        let mut b_toks: Vec<u32> = (0..26).collect();
        b_toks.extend(900..920);
        let mut b = seq_with(b_toks);
        bm.match_prefix(&mut b);
        assert_eq!(b.num_computed, 8, "resumes at the snapshot (block 2)");
        assert_eq!(b.kv_match_len, 24, "the cached KV matched six blocks");
        bm.unmatch_prefix(&mut b);
        assert_eq!(b.kv_match_len, 0, "a rolled-back admission forgets it");

        // No snapshots required (a plain transformer): the match is not backed off and the
        // field is left at 0 — there is no recurrent state to snapshot.
        let mut plain = BlockManager::with_prefix_cache(64, bs, true);
        let mut c = seq_with(a_toks.clone());
        plain.match_prefix(&mut c);
        assert_eq!(c.kv_match_len, 0);
    }
}
