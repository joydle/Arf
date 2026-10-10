//! GROW-ON-USE KV MEMORY (2026-09-23): the 8-bit KV pool as Metal 4 placement-sparse buffers.
//!
//! Every KV buffer is created at its FULL virtual size (the whole `--max-context`) and costs no
//! memory until pages are mapped into it from a placement heap. `SparseKv::ensure` maps 16K-slot
//! chunks as the highest slot a step WRITES climbs, so a server started at 262K context holds only
//! what its sequences have actually used. The kernels are unchanged: a slot's address in the
//! buffer is the same whether its page is mapped at load or an hour later.
//!
//! This is what the MEASURED-OUT note in `weights.rs` (the residency-set block) asked for — "one
//! big buffer per layer can never be lazy; adaptive context needs the pool GROWN" — without the
//! realloc + copy and without interior mutability on the consumers.
//!
//! PROVEN FIRST in a standalone probe on this box (M4 Max, macOS 26.5): an 8 GB and a 16 GB
//! sparse buffer cost +0.00 GB until mapped; a kernel on an ordinary MTLCommandQueue wrote and read
//! back 536,870,912 of 536,870,912 floats through a 2 GB mapped window; reads of UNMAPPED pages
//! return 0 and the command buffer still completes (a flash tile running past the last mapped slot
//! cannot fault). The same probe WITHOUT the event wait below read back 86,756 of 67,108,864: the
//! mapping update runs asynchronously on the MTL4 queue, and a compute pass that is not fenced
//! behind it races it. Never drop the wait.
//!
//! Memory is only ever ADDED (high-water mark): the allocator hands out the lowest free block ids
//! (`BlockAllocator::free` keeps the list sorted), so the high-water mark tracks the peak context
//! actually served.
//!
//! THE GPU IS IDLE WHEN A MAPPING CHANGES (2026-10-09). Two kernel panics on one M4 Max (macOS 26.5,
//! 2026-10-08 01:30 and 2026-10-09 01:12) were the same "Kernel data abort" at the same place in
//! Apple's GPU driver (`IOGPUFamily`), `arf-serve` the panicked task both times. The second came
//! 8 s after `[kv] sparse KV grew to 49152 slots | device 29.3 of 30.2 GB working set`, with two
//! prompts being read — the next growth step was due. A growth step rewrote a buffer's page table
//! on the MTL4 queue and committed the island's residency set while the island queue could still be
//! running the previous step against the same buffers (steps are pipelined). Now every growth that
//! maps memory is preceded by a wait for the island queue to drain (`idle_for_mapping`); at most a
//! few dozen growths a server's life. That the race is the panic's cause is NOT proven — the panic
//! has not been reproduced on demand — but it is the one GPU operation of ours the driver crash
//! lines up with.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::Message;
use objc2_foundation::NSObjectProtocol;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTL4CommandQueue, MTL4UpdateSparseBufferMappingOperation, MTLBlitCommandEncoder, MTLBuffer,
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLDevice, MTLHeap, MTLHeapDescriptor,
    MTLHeapType, MTLResidencySet, MTLResourceOptions, MTLSharedEvent, MTLSparsePageSize,
    MTLSparseTextureMappingMode, MTLStorageMode,
};

use super::island::MtlBuf;

/// Sparse page (tile) size: 256 KiB, the largest Metal offers — fewest mapping ops.
const PAGE: usize = 256 << 10;
/// Slots mapped per growth step. 16,384 slots of Qwen3.8's 8-bit pool = ~545 MB across its 16
/// attention layers; a 262K context is at most 16 steps (16 heaps).
pub const CHUNK_SLOTS: usize = 16_384;

/// KV slots mapped at the first step in safe memory (`premap`): six growth steps, ~3.3 GB on the
/// 27B — an agent's ~47K-token first message beside its ~31K-token safety check, with room.
pub const PREMAP_KV_SLOTS: usize = 6 * CHUNK_SLOTS;

struct SparseBuf {
    buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    bytes_per_slot: usize,
    pages_total: usize,
    pages_mapped: usize,
}

pub struct SparseKv {
    dev: Retained<ProtocolObject<dyn MTLDevice>>,
    q4: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    event_value: u64,
    bufs: Vec<SparseBuf>,
    heaps: Vec<Retained<ProtocolObject<dyn MTLHeap>>>,
    max_slots: usize,
    mapped_slots: usize,
    mapped_bytes: usize,
    /// Slots mapped per growth step (`CHUNK_SLOTS` for the KV pool, 1 row for the GDN bank).
    chunk: usize,
    /// Log label ("KV" / "GDN").
    label: &'static str,
    /// Zero newly mapped pages (placement-heap memory is not guaranteed zero; a buffer that
    /// replaced a zero-initialized one must stay zero until written). Needs `fill_q`.
    fill_q: Option<Retained<ProtocolObject<dyn MTLCommandQueue>>>,
}

// SAFETY: the Retained Metal objects are thread-safe handles; every mutation goes through the
// `Mutex<SparseKv>` that owns this value.
unsafe impl Send for SparseKv {}
unsafe impl Sync for SparseKv {}

/// Can this device make placement-sparse buffers and an MTL4 queue (Metal 4, macOS 26+)? objc2
/// panics on a missing selector, so ask first.
pub fn supported(dev: &ProtocolObject<dyn MTLDevice>) -> bool {
    dev.respondsToSelector(objc2::sel!(newBufferWithLength:options:placementSparsePageSize:))
        && dev.respondsToSelector(objc2::sel!(newMTL4CommandQueue))
}

impl SparseKv {
    /// `None` when this device/OS cannot make placement-sparse buffers (pre-Metal-4): the caller
    /// then allocates ordinary buffers, as before.
    pub fn new(dev: &ProtocolObject<dyn MTLDevice>, max_slots: usize) -> Option<Self> {
        if std::env::var_os("ARF_NO_SPARSE_KV").is_some() {
            return None;
        }
        Self::new_with(dev, max_slots, CHUNK_SLOTS, "KV", false)
    }

    /// The general form: `chunk` slots per growth step, `zero_new` fills newly mapped pages with
    /// zeros (synchronously, before `ensure` returns). The caller owns its own opt-out switch.
    pub fn new_with(
        dev: &ProtocolObject<dyn MTLDevice>,
        max_slots: usize,
        chunk: usize,
        label: &'static str,
        zero_new: bool,
    ) -> Option<Self> {
        if !supported(dev) {
            return None;
        }
        let q4 = dev.newMTL4CommandQueue()?;
        let event = dev.newSharedEvent()?;
        let fill_q = if zero_new {
            Some(dev.newCommandQueue()?)
        } else {
            None
        };
        Some(Self {
            dev: dev.retain(),
            q4,
            event,
            event_value: 0,
            bufs: Vec::new(),
            heaps: Vec::new(),
            max_slots,
            mapped_slots: 0,
            mapped_bytes: 0,
            chunk: chunk.max(1),
            label,
            fill_q,
        })
    }

    /// Zero slot `slot` of every buffer on the GPU, synchronously (the buffers are private, so the
    /// CPU cannot). For a recycled GDN bank row.
    pub fn zero_slot(&self, slot: usize) -> Result<(), String> {
        let q = self.fill_q.as_ref().ok_or("sparse: no fill queue")?;
        let cb = q.commandBuffer().ok_or("sparse: no command buffer")?;
        let enc = cb.blitCommandEncoder().ok_or("sparse: no blit encoder")?;
        for b in &self.bufs {
            let (lo, len) = (slot * b.bytes_per_slot, b.bytes_per_slot);
            if (lo + len).div_ceil(PAGE) <= b.pages_mapped {
                enc.fillBuffer_range_value(&b.buf, NSRange::new(lo, len), 0);
            }
        }
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        Ok(())
    }

    /// A sparse buffer of `max_slots * bytes_per_slot` bytes (rounded up to a page), unmapped.
    pub fn alloc(&mut self, bytes_per_slot: usize) -> Option<MtlBuf> {
        let pages_total = (self.max_slots * bytes_per_slot).div_ceil(PAGE);
        let buf = unsafe {
            self.dev
                .newBufferWithLength_options_placementSparsePageSize(
                    pages_total * PAGE,
                    MTLResourceOptions::StorageModePrivate,
                    MTLSparsePageSize::Size256,
                )
        }?;
        self.bufs.push(SparseBuf {
            buf: buf.clone(),
            bytes_per_slot,
            pages_total,
            pages_mapped: 0,
        });
        Some(MtlBuf(buf))
    }

    pub fn mapped_slots(&self) -> usize {
        self.mapped_slots
    }

    pub fn mapped_bytes(&self) -> usize {
        self.mapped_bytes
    }

    /// The most slots this pool can map (its virtual size).
    pub fn max_slots(&self) -> usize {
        self.max_slots
    }

    /// `ARF_SPARSE_PREMAP=1` (set by the server after a GPU driver panic, `gpu_panic`): the first
    /// growth maps a fixed working set (`PREMAP_KV_SLOTS`, the bank's first rows) at load with the
    /// GPU idle; past it, growth is behind `idle_for_mapping` as ever.
    pub fn premap() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("ARF_SPARSE_PREMAP").is_some_and(|v| v != "0"))
    }

    /// Would [`ensure`](Self::ensure)`(slots)` map new memory? The caller then lets the GPU go
    /// idle first (`MetalIsland::idle_for_mapping`): see the module doc's last section.
    pub fn would_grow(&self, slots: usize) -> bool {
        slots > self.mapped_slots
            || self.bufs.iter().any(|b| {
                b.pages_mapped
                    < (self.mapped_slots * b.bytes_per_slot)
                        .div_ceil(PAGE)
                        .min(b.pages_total)
            })
    }

    /// Make slots `[0, slots)` of every buffer backed by memory. Returns `Ok(true)` when it grew.
    /// New heaps join `rset` (the island's residency set, bound to its queue) before any command
    /// buffer can reference the new pages, and the CPU waits for the mapping to land.
    pub fn ensure(
        &mut self,
        slots: usize,
        rset: Option<&ProtocolObject<dyn MTLResidencySet>>,
    ) -> Result<bool, String> {
        // A buffer allocated AFTER an earlier mapping lags it (the GDN bank allocates layer by
        // layer during the first step): map it up to the current high-water too.
        let lagging = self.bufs.iter().any(|b| {
            b.pages_mapped
                < (self.mapped_slots * b.bytes_per_slot)
                    .div_ceil(PAGE)
                    .min(b.pages_total)
        });
        if slots <= self.mapped_slots && !lagging {
            return Ok(false);
        }
        let target = slots
            .div_ceil(self.chunk)
            .saturating_mul(self.chunk)
            .min(self.max_slots);
        let target = target.max(slots.min(self.max_slots)).max(self.mapped_slots);
        let new_pages: Vec<usize> = self
            .bufs
            .iter()
            .map(|b| {
                (target * b.bytes_per_slot)
                    .div_ceil(PAGE)
                    .min(b.pages_total)
                    .saturating_sub(b.pages_mapped)
            })
            .collect();
        let total: usize = new_pages.iter().sum();
        if total == 0 {
            self.mapped_slots = target;
            return Ok(false);
        }
        let desc = MTLHeapDescriptor::new();
        desc.setType(MTLHeapType::Placement);
        desc.setStorageMode(MTLStorageMode::Private);
        desc.setSize(total * PAGE);
        desc.setMaxCompatiblePlacementSparsePageSize(MTLSparsePageSize::Size256);
        let heap = self.dev.newHeapWithDescriptor(&desc).ok_or_else(|| {
            format!(
                "sparse {}: no {} MB placement heap",
                self.label,
                (total * PAGE) >> 20
            )
        })?;
        let mut heap_off = 0usize;
        let fresh: Vec<(usize, usize)> = self
            .bufs
            .iter()
            .zip(&new_pages)
            .map(|(b, &n)| (b.pages_mapped, n))
            .collect();
        for (b, &n) in self.bufs.iter_mut().zip(&new_pages) {
            if n == 0 {
                continue;
            }
            let op = MTL4UpdateSparseBufferMappingOperation {
                mode: MTLSparseTextureMappingMode::Map,
                bufferRange: NSRange::new(b.pages_mapped, n),
                heapOffset: heap_off,
            };
            unsafe {
                self.q4.updateBufferMappings_heap_operations_count(
                    &b.buf,
                    Some(&heap),
                    std::ptr::NonNull::from(&op),
                    1,
                )
            };
            b.pages_mapped += n;
            heap_off += n;
        }
        self.event_value += 1;
        self.q4
            .signalEvent_value(ProtocolObject::from_ref(&*self.event), self.event_value);
        if let Some(rs) = rset {
            rs.addAllocation(ProtocolObject::from_ref(&*heap));
            rs.commit();
        }
        // See the module doc: an unfenced compute pass read back 0.13% of a freshly mapped window.
        if !self
            .event
            .waitUntilSignaledValue_timeoutMS(self.event_value, 10_000)
        {
            return Err(format!(
                "sparse {}: mapping update did not complete in 10 s",
                self.label
            ));
        }
        // Newly mapped placement-heap pages hold whatever the heap held: zero them where the
        // buffer's contract is "zero until written" (the GDN bank — a fresh row is an empty state).
        if let Some(q) = self.fill_q.as_ref() {
            let cb = q.commandBuffer().ok_or("sparse: no command buffer")?;
            let enc = cb.blitCommandEncoder().ok_or("sparse: no blit encoder")?;
            for (b, &(from, n)) in self.bufs.iter().zip(&fresh) {
                if n > 0 {
                    enc.fillBuffer_range_value(&b.buf, NSRange::new(from * PAGE, n * PAGE), 0);
                }
            }
            enc.endEncoding();
            cb.commit();
            cb.waitUntilCompleted();
        }
        self.heaps.push(heap);
        self.mapped_bytes += total * PAGE;
        self.mapped_slots = target;
        let ws = self.dev.recommendedMaxWorkingSetSize() as f64;
        let alloc = self.dev.currentAllocatedSize() as f64;
        eprintln!(
            "[kv] sparse {} grew to {target} slots: {:.2} GB mapped (+{:.0} MB) | device {:.1} of \
             {:.1} GB working set{}",
            self.label,
            self.mapped_bytes as f64 / 1e9,
            (total * PAGE) as f64 / 1e6,
            alloc / 1e9,
            ws / 1e9,
            if alloc > ws {
                " — ⚠ device allocations are past the recommended working set. They count reserved but \
                 untouched buffers too (the MTP head's pool: 0 resident while the block draft \
                 serves), so this alone does not mean paging; if decode slows, it does"
            } else {
                ""
            }
        );
        Ok(true)
    }
}
