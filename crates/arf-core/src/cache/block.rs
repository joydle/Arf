//! Physical block allocator: a free list over `[0, num_blocks)`.
//!
//! The allocator knows nothing about tensors — it just hands out and reclaims
//! integer block ids. The paged KV cache maps those ids onto storage.

use crate::error::{ArfError, Result};

/// A LIFO free-list allocator of physical KV block ids.
#[derive(Debug, Clone)]
pub struct BlockAllocator {
    free: Vec<u32>,
    total: usize,
}

impl BlockAllocator {
    /// Create an allocator owning `total` blocks, all initially free.
    pub fn new(total: usize) -> Self {
        // Hand out low ids first; reverse so `pop` yields 0,1,2,...
        let free = (0..total as u32).rev().collect();
        BlockAllocator { free, total }
    }

    /// Number of blocks currently available.
    pub fn num_free(&self) -> usize {
        self.free.len()
    }

    /// Number of blocks currently handed out.
    pub fn num_used(&self) -> usize {
        self.total - self.free.len()
    }

    /// Total capacity.
    pub fn total(&self) -> usize {
        self.total
    }

    /// The id the next `allocate(1)` would hand out (the lowest free one), if any.
    pub fn lowest_free(&self) -> Option<u32> {
        self.free.last().copied()
    }

    /// Allocate `n` blocks, or fail without partially allocating.
    pub fn allocate(&mut self, n: usize) -> Result<Vec<u32>> {
        if n > self.free.len() {
            return Err(ArfError::OutOfBlocks {
                requested: n,
                free: self.free.len(),
            });
        }
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.free.pop().expect("checked capacity above"));
        }
        Ok(out)
    }

    /// Return blocks to the pool. Order is irrelevant for correctness — but not for MEMORY: the
    /// list is kept sorted (highest first) so `pop` always yields the LOWEST free id. The Metal KV
    /// pool maps memory up to the highest slot ever written (grow-on-use, `sparse_kv.rs`); with a
    /// plain LIFO a short request after a long one popped the long one's highest block and pinned
    /// its whole range. Sorting costs O(n log n) over at most a few thousand ids per free.
    pub fn free(&mut self, blocks: impl IntoIterator<Item = u32>) {
        for b in blocks {
            debug_assert!((b as usize) < self.total, "freeing out-of-range block");
            self.free.push(b);
        }
        self.free.sort_unstable_by(|a, b| b.cmp(a));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ArfError;

    #[test]
    fn alloc_and_free_conserves_blocks() {
        let mut a = BlockAllocator::new(8);
        assert_eq!(a.num_free(), 8);
        let b = a.allocate(3).unwrap();
        assert_eq!(b.len(), 3);
        assert_eq!(a.num_free(), 5);
        assert_eq!(a.num_used(), 3);
        a.free(b);
        assert_eq!(a.num_free(), 8);
        assert_eq!(a.num_used(), 0);
    }

    #[test]
    fn allocation_is_disjoint() {
        let mut a = BlockAllocator::new(4);
        let x = a.allocate(2).unwrap();
        let y = a.allocate(2).unwrap();
        for b in &x {
            assert!(!y.contains(b), "blocks must not be handed out twice");
        }
    }

    #[test]
    fn over_allocation_fails_without_consuming() {
        let mut a = BlockAllocator::new(2);
        let err = a.allocate(3).unwrap_err();
        assert!(matches!(
            err,
            ArfError::OutOfBlocks {
                requested: 3,
                free: 2
            }
        ));
        // Nothing was consumed by the failed request.
        assert_eq!(a.num_free(), 2);
    }
}
