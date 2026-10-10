//! A GPU buffer sub-allocator: one large resident device buffer carved into
//! many aligned regions by Sebastian Aaltonen's `OffsetAllocator` (the safe-Rust
//! port, `offset-allocator`).
//!
//! # Why
//!
//! The resident decode path in [`crate::gpu`] currently creates a fresh
//! `wgpu::Buffer` for every transient value in a token step — the per-token
//! `k_all`/`v_all`/`attn`/`silu` scratch and the dozens of small uniform blocks
//! (one per kernel, per layer). A 16-layer model allocates *hundreds* of buffers
//! per token, which is the "buffer pooling" perf gap called out in the README.
//!
//! A [`GpuArena`] replaces that churn with a single device buffer plus an O(1)
//! hard-real-time offset allocator. You ask for `n` bytes, you get back a
//! [`Region`] — a byte range inside the arena — and bind it as a sub-range.
//! Freeing is O(1) and returns the range to the pool; [`GpuArena::reset`] frees
//! everything at once, which is the natural "end of token / start of frame"
//! operation.
//!
//! The allocator only manages *offsets*; the arena owns the backing buffer and
//! its lifetime. Sizes vary, so this is exactly the case a TLSF-style allocator
//! is built for — unlike the uniform-size KV [`BlockAllocator`], which stays a
//! plain free-list.
//!
//! [`BlockAllocator`]: arf_core::cache::BlockAllocator

use std::sync::Arc;

use bytemuck::Pod;
use offset_allocator::{Allocation, Allocator};

use crate::gpu::GpuContext;
use arf_core::error::{ArfError, Result};

/// Storage- and uniform-buffer binding offsets must be a multiple of this many
/// bytes on common backends (wgpu's default `min_storage_buffer_offset_alignment`
/// and `min_uniform_buffer_offset_alignment` are both 256). We make it the
/// allocator's *unit*, so every offset the allocator hands out is already a
/// valid binding offset and one `u32` unit index addresses up to 256 × 4 GiB.
pub const ALIGN: u64 = 256;

/// Number of `ALIGN`-byte units needed to hold `bytes` (rounding up).
fn units_for(bytes: u64) -> u64 {
    bytes.div_ceil(ALIGN)
}

/// Capacity (in bytes) for an arena that must hold `regions`, each padded to a
/// whole [`ALIGN`] unit.
///
/// You cannot size a multi-region arena at the bare sum of its regions. The
/// backing `offset-allocator` is a two-level bin allocator: a tightly-sized
/// arena lands in a low bin whose largest *contiguous* run floors below the sum,
/// so a later region that the sum says should fit gets refused. (Measured: two
/// 136-unit regions in a 275-unit arena — sum 272, slack 3 — fail; the trailing
/// 139 free units expose only a 128-unit run.) Rounding the *unit count* up to a
/// power of two seats every region cleanly, because a pow2 arena is a single
/// top-level bin that splits without stranding. Proven for 1..=4097-unit regions.
///
/// Decision: picked pow2-rounding from {pow2-round, exact-bin-table,
/// per-region-arena} — tradeoff: a few % slack vs. reverse-engineering the
/// allocator's private bin sizes (brittle across versions) or paying a buffer
/// per region (the churn GpuArena exists to kill).
pub fn arena_bytes(regions: &[u64]) -> u64 {
    let units: u64 = regions.iter().map(|&b| units_for(b)).sum();
    units.max(1).next_power_of_two() * ALIGN
}

/// A live sub-allocation: the byte range `[offset, offset + size)` inside the
/// arena buffer. Hand it to [`GpuArena::write`] / [`GpuArena::binding`] while it
/// is alive, and return it with [`GpuArena::free`] (or drop the whole generation
/// via [`GpuArena::reset`]).
pub struct Region {
    handle: Allocation,
    offset: u64,
    size: u64,
}

impl Region {
    /// Byte offset of this region within the arena buffer. Always a multiple of
    /// [`ALIGN`], so it is a valid storage/uniform binding offset.
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Requested size in bytes (the un-padded length asked for at `alloc`).
    pub fn size(&self) -> u64 {
        self.size
    }
}

/// One resident device buffer plus an offset allocator over it.
pub struct GpuArena {
    ctx: Arc<GpuContext>,
    buffer: wgpu::Buffer,
    alloc: Allocator,
    capacity: u64,
}

impl GpuArena {
    /// Allocate a `capacity`-byte device buffer with the given `usage` and an
    /// offset allocator over it. `capacity` is rounded up to a multiple of
    /// [`ALIGN`]. `usage` should include the binding kind you intend
    /// (`STORAGE`/`UNIFORM`) plus `COPY_DST` if you will [`write`] into it.
    ///
    /// [`write`]: GpuArena::write
    pub fn new(
        ctx: &Arc<GpuContext>,
        label: &str,
        capacity: u64,
        usage: wgpu::BufferUsages,
    ) -> Result<Self> {
        let units = units_for(capacity);
        if units == 0 || units > u32::MAX as u64 {
            return Err(ArfError::other(format!(
                "GpuArena capacity {capacity} bytes is out of range (1..={} bytes)",
                u32::MAX as u64 * ALIGN
            )));
        }
        let capacity = units * ALIGN;
        let buffer = ctx.create_buffer(label, capacity, usage);
        Ok(GpuArena {
            ctx: ctx.clone(),
            buffer,
            alloc: Allocator::new(units as u32),
            capacity,
        })
    }

    /// Reserve `size` bytes and return the [`Region`] they occupy. The byte
    /// range is padded up to [`ALIGN`] internally so neighbouring regions never
    /// share a 256-byte granule. Fails (rather than partially allocating) when
    /// the arena cannot satisfy the request.
    pub fn alloc(&mut self, size: u64) -> Result<Region> {
        if size == 0 {
            return Err(ArfError::other("GpuArena alloc of zero bytes"));
        }
        let units = units_for(size);
        // `units <= capacity_units <= u32::MAX`, so the cast cannot truncate.
        let handle = self.alloc.allocate(units as u32).ok_or_else(|| {
            let free = self.alloc.storage_report();
            ArfError::other(format!(
                "GPU arena out of space: requested {size} bytes ({units} units), \
                 largest free run is {} units of {} total",
                free.largest_free_region,
                self.capacity / ALIGN
            ))
        })?;
        let offset = handle.offset as u64 * ALIGN;
        Ok(Region {
            handle,
            offset,
            size,
        })
    }

    /// Return `region`'s range to the pool. O(1); merges with adjacent free
    /// ranges inside the allocator.
    pub fn free(&mut self, region: Region) {
        self.alloc.free(region.handle);
    }

    /// Free every outstanding region in one shot — the cheap "new generation"
    /// reset for per-token / per-frame scratch. Outstanding [`Region`]s must not
    /// be used or freed afterwards.
    pub fn reset(&mut self) {
        self.alloc.reset();
    }

    /// Upload `data` into `region`, starting at the region's offset. Panics if
    /// `data` does not fit; bytes past `data` keep their previous contents. The
    /// arena's buffer must carry `COPY_DST`.
    pub fn write<T: Pod>(&self, region: &Region, data: &[T]) {
        let bytes = std::mem::size_of_val(data) as u64;
        assert!(
            bytes <= region.size,
            "GpuArena::write of {bytes} bytes into a {}-byte region",
            region.size
        );
        self.ctx.write_at(&self.buffer, region.offset, data);
    }

    /// Reserve a region exactly sized to `data` and upload it in one call — the
    /// arena equivalent of `storage_init` / `uniform_of` for a per-token value.
    pub fn alloc_write<T: Pod>(&mut self, data: &[T]) -> Result<Region> {
        let region = self.alloc(std::mem::size_of_val(data) as u64)?;
        self.write(&region, data);
        Ok(region)
    }

    /// A bindable view of `region` as a sub-range of the arena buffer, for use
    /// as a `wgpu::BindGroupEntry` resource.
    pub fn binding(&self, region: &Region) -> wgpu::BufferBinding<'_> {
        wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: region.offset,
            size: wgpu::BufferSize::new(region.size),
        }
    }

    /// `region` as a bind-group resource, ready to pass to
    /// [`crate::gpu::CommandPass::add_bound`] alongside whole-buffer
    /// `as_entire_binding()` entries.
    pub fn resource(&self, region: &Region) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(self.binding(region))
    }

    /// The backing device buffer (e.g. for a whole-arena readback in tests).
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Total arena size in bytes (after rounding up at construction).
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Bytes currently handed out (padded to [`ALIGN`] granules), from the
    /// allocator's own free-space report.
    pub fn bytes_in_use(&self) -> u64 {
        let free = self.alloc.storage_report();
        self.capacity - free.total_free_space as u64 * ALIGN
    }
}

impl std::fmt::Debug for GpuArena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuArena")
            .field("capacity", &self.capacity)
            .field("bytes_in_use", &self.bytes_in_use())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Pure allocator/offset math: runs without a GPU adapter. ------------

    /// Regression for the kv-arena fragmentation bug: an arena sized by
    /// [`arena_bytes`] must actually seat all of its regions. The bare sum (with
    /// small fixed slack) does not — the two-level bin allocator floors the
    /// trailing free run below the sum — which panicked the resident decode path
    /// on real weights (kv_dim=512, ctx_len=17 → two 136-unit regions). pow2
    /// rounding fixes it; this checks the cases the bare sum fails.
    #[test]
    fn arena_bytes_seats_every_region() {
        // (region byte sizes that previously fragmented when summed tightly)
        let cases: &[&[u64]] = &[
            &[136 * ALIGN, 136 * ALIGN], // the production panic
            &[100 * ALIGN, 100 * ALIGN],
            &[17 * ALIGN, 17 * ALIGN],
            &[1000 * ALIGN, 1000 * ALIGN],
            &[136 * ALIGN, 136 * ALIGN, 64 * ALIGN], // three regions
        ];
        for regions in cases {
            let total_units = arena_bytes(regions) / ALIGN;
            let mut a = Allocator::<u32>::new(total_units as u32);
            for (i, &bytes) in regions.iter().enumerate() {
                assert!(
                    a.allocate(units_for(bytes) as u32).is_some(),
                    "region {i} of {regions:?} did not fit in arena_bytes \
                     = {total_units} units",
                );
            }
        }
    }

    #[test]
    fn units_round_up_to_alignment() {
        assert_eq!(units_for(0), 0);
        assert_eq!(units_for(1), 1);
        assert_eq!(units_for(ALIGN), 1);
        assert_eq!(units_for(ALIGN + 1), 2);
        assert_eq!(units_for(4 * ALIGN), 4);
    }

    /// The offset arithmetic we layer on top of `offset-allocator` must yield
    /// aligned, non-overlapping byte ranges. Exercised against the real
    /// allocator (no GPU needed) so the integration math is covered even on
    /// adapter-less CI.
    #[test]
    fn offsets_are_aligned_and_disjoint() {
        let mut a = Allocator::<u32>::new(units_for(1 << 20) as u32); // 1 MiB of units
        let sizes = [300u64, 16, 4096, 257];
        let mut ranges = Vec::new();
        for s in sizes {
            let h = a.allocate(units_for(s) as u32).expect("space");
            let off = h.offset as u64 * ALIGN;
            assert_eq!(off % ALIGN, 0, "offset {off} not aligned");
            ranges.push((off, off + units_for(s) * ALIGN));
        }
        for (i, &(lo_i, hi_i)) in ranges.iter().enumerate() {
            for &(lo_j, hi_j) in &ranges[i + 1..] {
                assert!(
                    hi_i <= lo_j || hi_j <= lo_i,
                    "ranges {lo_i}..{hi_i} and {lo_j}..{hj} overlap",
                    hj = hi_j
                );
            }
        }
    }

    // --- End-to-end against a real device, skipped when none is present. ----

    /// Sub-allocate several regions from one buffer, upload distinct data into
    /// each, read the whole buffer back, and assert every region kept its own
    /// bytes (i.e. the offsets are correct and disjoint on real hardware).
    #[test]
    fn arena_subregions_roundtrip_on_gpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => Arc::new(c),
            Err(_) => {
                eprintln!("no GPU adapter; skipping arena_subregions_roundtrip_on_gpu");
                return;
            }
        };
        let cap = 1 << 16; // 64 KiB
        let mut arena = GpuArena::new(
            &ctx,
            "test-arena",
            cap,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        )
        .expect("arena");

        // Three regions of different sizes, each filled with a distinct pattern.
        let r0 = arena.alloc(64 * 4).unwrap();
        let r1 = arena.alloc(300 * 4).unwrap();
        let r2 = arena.alloc(16 * 4).unwrap();
        for r in [&r0, &r1, &r2] {
            assert_eq!(r.offset() % ALIGN, 0);
        }
        let d0: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let d1: Vec<f32> = (0..300).map(|i| 1000.0 + i as f32).collect();
        let d2: Vec<f32> = (0..16).map(|i| -(i as f32)).collect();
        arena.write(&r0, &d0);
        arena.write(&r1, &d1);
        arena.write(&r2, &d2);

        let all = ctx.read_f32(arena.buffer(), (cap / 4) as usize);
        let slice = |r: &Region, n: usize| {
            let start = (r.offset() / 4) as usize;
            all[start..start + n].to_vec()
        };
        assert_eq!(slice(&r0, 64), d0);
        assert_eq!(slice(&r1, 300), d1);
        assert_eq!(slice(&r2, 16), d2);

        // Freeing and re-allocating the same size reuses freed space (use stays
        // bounded), and reset returns the arena to empty.
        let used_before = arena.bytes_in_use();
        assert!(used_before > 0);
        arena.free(r1);
        let r3 = arena.alloc(300 * 4).unwrap();
        assert!(arena.bytes_in_use() <= used_before);
        arena.free(r0);
        arena.free(r2);
        arena.free(r3);
        arena.reset();
        assert_eq!(arena.bytes_in_use(), 0);
    }

    #[test]
    fn alloc_past_capacity_fails_cleanly() {
        let ctx = match GpuContext::new() {
            Ok(c) => Arc::new(c),
            Err(_) => {
                eprintln!("no GPU adapter; skipping alloc_past_capacity_fails_cleanly");
                return;
            }
        };
        let mut arena = GpuArena::new(
            &ctx,
            "small",
            4 * ALIGN,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        )
        .unwrap();
        let _ok = arena.alloc(2 * ALIGN).unwrap();
        // Requesting more than what is left must error, not panic or partially
        // allocate.
        assert!(arena.alloc(8 * ALIGN).is_err());
        assert!(arena.alloc(0).is_err());
    }
}
