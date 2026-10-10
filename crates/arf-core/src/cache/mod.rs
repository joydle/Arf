//! KV cache: a physical block allocator and a paged cache built on it.

pub mod block;
pub mod paged;

pub use block::BlockAllocator;
pub use paged::{blocks_needed, slots_for, write_runs, LayerKvGeom, PagedKvCache, WriteRun};
