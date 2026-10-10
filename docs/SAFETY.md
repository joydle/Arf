# Unsafe code: what it is and the invariants it relies on

Arf contains roughly 700 `unsafe` blocks (`grep -rn 'unsafe {' crates --include='*.rs' | wc -l`;
on 2026-10-03: 693 in `arf-gpu`, 3 in `arf-core`, 3 in `arf-serve`; 620 on 2026-09-20) — and
about half of `arf-gpu`'s (326) are in `gpu/metal/selftest.rs`, the self-test that exercises the
same paths. That number
looks alarming out of context, so this document states what they are and which rules keep them
sound. None of them is unsafe *logic* — there is no pointer arithmetic over Rust data structures,
no `transmute` of user types, no hand-rolled synchronisation primitive.

They fall into two dominant categories:

| what | share | why it is unsafe |
|---|---|---|
| `MTLBuffer::contents()` and the reads/writes through it | most | Returns a raw pointer into GPU-visible memory. Rust cannot know its length, lifetime, or whether the GPU is concurrently writing it. |
| `wgpu::Device::as_hal` / `raw_device` | many | Reaches through wgpu to the underlying `MTLDevice`. Unsafe because wgpu cannot guarantee the backend is Metal or that the handle outlives the guard. |
| Objective-C message sends (`setBuffer`, `dispatchThreadgroups`, `useResource`, `memoryBarrier`, `newBuffer*`) | a few dozen | FFI into Metal. Unsafe by declaration in `objc2`; each is a normal Metal API call. |

The engine is fast **because** it talks to `MTLBuffer` directly rather than round-tripping through
wgpu's safe abstractions. That is the trade, and it is deliberate.

---

## The four invariants

Every `unsafe` block in this codebase relies on at least one of these. When adding one, say which.

### 1. A mapped pointer is only valid while its buffer is alive

`contents()` returns a pointer owned by the `MTLBuffer`. The buffer must outlive every read and
write through that pointer.

In practice this means the `Retained<ProtocolObject<dyn MTLBuffer>>` is held in a long-lived
struct — `MegakernelBufs`, `BatchedScratch`, `GdnScratch`, the KV pools — and the pointer is
derived fresh at each use rather than cached. **Never store a raw pointer across a call boundary.**

The one exception is documented at its site: weights wrapped with
`newBufferWithBytesNoCopy` alias an `Arc<Mmap>` that the caller stores on `GpuModel`, so the mmap
outlives the buffers by construction. A `None` deallocator is passed precisely because Metal must
*not* free memory the `Arc` owns.

### 2. Host reads must be fenced against the GPU

The GPU and CPU share these buffers. Reading a buffer the GPU is still writing yields torn or
stale data, silently — no crash, no error, just wrong numbers.

Every host read of a result buffer must happen after the producing command buffer completes. The
paths that do this correctly: `read_pending` fences on `slot_done` before touching `out_tokens`;
`wait_batched_step(pos)` blocks on the step at that position; the gpustats and parity paths force
`waitUntilCompleted` per chunk *because* they read timings.

> This invariant has been violated in practice and the failure is instructive: an MTP draft read
> its output token before the record had run, and got a stale `0` every time. It looked exactly
> like a model predicting token 0.

### 3. No-copy wrapping requires page alignment

`newBufferWithBytesNoCopy` requires the host pointer to be page-aligned and the length to be a
page multiple — 16 KB on Apple Silicon. The transcode cache pads every region to a page boundary
specifically so the wrap is legal (`weight_cache.rs`, `PAGE`).

If you wrap a region that is not aligned, Metal fails the allocation and the loader falls back to
a copy — a slow path, not a crash. Do not "fix" that by relaxing the alignment.

### 4. Row and slot indices must be bounds-checked before pointer arithmetic

Most `contents()` uses index a `[rows, width]` slab: `ptr.add(row * stride + col)`. Rust cannot
check that, so the code must, and the helpers do — `gdn_snapshot_row`, `kv_snapshot_slots`,
`bmega_hidden_row` all compare `start + len` against `buffer.length() / 4` and return `None`
rather than read out of range.

> Also instructive: a kernel indexing `acc[cc * ACC_ROWS + r]` without clamping `r` by
> `ACC_ROWS` would write into the *next column's* accumulators — silent cross-column corruption,
> not an out-of-bounds fault, because the array is large enough to absorb it. Bounds are about
> correctness here, not just memory safety.

---

## Reviewing a change that adds `unsafe`

1. Which of the four invariants does it rely on? Write it as a `// SAFETY:` comment.
2. If it reads a buffer the GPU writes, where is the fence?
3. If it does pointer arithmetic, where is the bounds check?
4. Can it be done with a safe wrapper that already exists? `dispatch_serial` and
   `dispatch_concurrent` cover most kernel launches without new `unsafe` at the call site.

## What is deliberately *not* here

`arf-core` has **three** `unsafe` sites, all on read-only memory maps, each carrying a
`// SAFETY:` comment (two until the PyTorch checkpoint reader landed; counted 2026-10-03):

- `mmap_file` (`model/weights.rs`) — the read-only `Mmap::map`. The standard mmap contract: the
  file must not be modified while mapped.
- `unchecked_advise_range(DontNeed)` (`model/weights.rs`) — drops clean page-cache pages after a
  tensor is copied out. Sound because the mapping is `Mmap`, not `MmapMut`, so there are no
  un-synced writes to lose; a page touched again is transparently re-read from the file.
- `TorchCheckpoint::open` (`model/torch_ckpt.rs`, behind `arf inspect`) — the same read-only
  `Mmap::map` contract as `mmap_file`.

`arf-serve` has **three**, all macOS FFI in `actor.rs` that sets the model actor thread's own
scheduling: `pthread_set_qos_class_self_np` (no pointers), and, only under the opt-in
`ARF_ACTOR_RT`, `mach_timebase_info` (fills a plain `#[repr(C)]` struct) and `thread_policy_set`
(the struct matches `thread_time_constraint_policy_data_t`, count 4). None touches shared state.

Everything else in the core crate — the tokenizer, the GGUF reader, the scheduler, the paged KV
bookkeeping, the CPU tensor kernels — is safe Rust with no FFI. The CPU matmul parallelises over
disjoint `chunks_mut` rather than raw pointers, deliberately. If you are auditing this project and
want to start where the logic lives rather than where the pointers are, start there.
