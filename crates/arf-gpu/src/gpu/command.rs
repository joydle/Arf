//! Compute-dispatch machinery: a compiled `ComputeKernel`, bind-group
//! cursor, and the `CommandPass` recorder (optional GPU timestamp `Timing`).
//! Split out of gpu/mod.rs; behavior unchanged.

use super::*;

/// A compiled compute shader (one `.wgsl` entry point) plus its bind-group
/// layout, ready to dispatch against resident buffers.
///
/// This is the reusable primitive every Phase 2 kernel is built on: construct
/// once from WGSL source, then `dispatch` it with the buffers it binds and the
/// number of workgroups. The layout is inferred from the pipeline so each kernel
/// declares its own bindings in WGSL without hand-writing a layout in Rust.
pub struct ComputeKernel {
    ctx: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

impl ComputeKernel {
    /// Compile `wgsl` (entry point `main`) into a dispatchable kernel.
    pub fn new(ctx: &Arc<GpuContext>, label: &str, wgsl: &str) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(wgsl)),
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    zero_initialize_workgroup_memory: false,
                },
                cache: None,
            });
        let layout = pipeline.get_bind_group_layout(0);
        ComputeKernel {
            ctx: ctx.clone(),
            pipeline,
            layout,
        }
    }

    /// Encode and submit one dispatch. `buffers` are bound at successive
    /// `@binding(0..)` of `@group(0)`; `workgroups` is the `(x, y, z)` grid.
    pub fn dispatch(&self, label: &str, buffers: &[&wgpu::Buffer], workgroups: [u32; 3]) {
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let bind = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.layout,
                entries: &entries,
            });
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
        }
        self.ctx.queue.submit(std::iter::once(enc.finish()));
    }

    /// Build the bind group for this kernel from `buffers` (bound at successive
    /// `@binding(0..)`). Used by [`CommandPass`] to record without submitting.
    fn bind(&self, label: &str, buffers: &[&wgpu::Buffer]) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        self.ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.layout,
                entries: &entries,
            })
    }

    /// Build a bind group from arbitrary binding resources — the cacheable
    /// counterpart of [`bind`](ComputeKernel::bind) (which takes whole buffers).
    /// The caller (the P15 decode cache) builds one ONCE and reuses it across
    /// tokens; see `add_prebuilt`.
    pub(crate) fn bind_resources(
        &self,
        label: &str,
        resources: &[wgpu::BindingResource<'_>],
    ) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry> = resources
            .iter()
            .enumerate()
            .map(|(i, r)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: r.clone(),
            })
            .collect();
        self.ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.layout,
                entries: &entries,
            })
    }

    /// The compute pipeline handle, for recording a prebuilt bind group.
    pub(crate) fn pipeline(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }
}

/// A running cursor over the P15 decode bind-group cache: `binds` is the model's
/// persistent `decode_binds` vec, `slot` is the next site index (advances per
/// `add_bound_cached`, reset to 0 at the top of each decode_token). Built lazily —
/// the first decode step fills `binds`; later steps hit.
///
/// Decision: keyed implicitly by *dispatch order within one fixed call
/// shape* (the steady decode step: `token.is_none() && want_logits`, m=1), NOT
/// by a (pipeline, resources) hash — tradeoff: a positional cursor is zero-cost
/// to look up (just `binds[slot]`) and needs no per-dispatch keying, but it is
/// only valid when the dispatch sequence is byte-for-byte identical every call.
/// That holds for steady decode (same graph, hoisted arenas → stable offsets)
/// and nowhere else, so prefill/forward_batch pass `None` and rebuild. A hashed
/// cache would cover those too but pays a key+lookup on the hot path we just
/// spent P14 shrinking — the wrong trade for the one shape that runs N times.
pub struct BindCursor<'a> {
    binds: &'a mut Vec<wgpu::BindGroup>,
    slot: usize,
}

impl<'a> BindCursor<'a> {
    pub fn new(binds: &'a mut Vec<wgpu::BindGroup>) -> Self {
        BindCursor { binds, slot: 0 }
    }
}

/// Records many compute dispatches into ONE command buffer, submitted once.
///
/// This is the structural heart of resident decode: a decode token chains ~115
/// passes (embed → 16 layers → norm → lm_head → sample) into a single
/// `wgpu::CommandBuffer` so the GPU runs the whole step without returning to the
/// CPU between passes. Bind groups are held until [`CommandPass::submit`] so they
/// outlive the recording.
pub struct CommandPass {
    ctx: Arc<GpuContext>,
    encoder: wgpu::CommandEncoder,
    binds: Vec<wgpu::BindGroup>,
    /// Deferred op list (P14). `add`/`add_bound`/`copy_word` only *record* into
    /// this Vec; the actual compute passes are encoded in [`submit`], where
    /// consecutive `Dispatch`es are coalesced into ONE `begin_compute_pass`
    /// (breaking only at a `Copy`, which cannot live inside a pass). This cuts
    /// ~258 per-token pass-encodes down to ~1, recovering part of the ~3900us
    /// per-token dispatch overhead. Native wgpu-core auto-inserts the per-
    /// resource hazard barriers between dispatches *within* a pass on Metal, so
    /// the merge is bit-identical: same pipelines, bind groups, grids, order —
    /// only fewer pass boundaries.
    ///
    /// [`submit`]: CommandPass::submit
    ops: Vec<Op>,
    /// Per-pass GPU timestamp queries, when the `profiling` feature is on and
    /// the device supports `TIMESTAMP_QUERY`. `None` otherwise (and always when
    /// profiling is off, where this field does not exist).
    #[cfg(feature = "profiling")]
    timing: Option<Timing>,
}

/// A recorded-but-not-yet-encoded operation. Built by `add`/`add_bound`/
/// `copy_word`, replayed in `submit`. A run of consecutive `Dispatch`es merges
/// into one compute pass; a `Copy` forces the pass to close (a
/// `copy_buffer_to_buffer` cannot occur inside an open compute pass).
enum Op {
    Dispatch {
        /// Cloned pipeline handle. `wgpu::ComputePipeline` is `Arc`-backed in
        /// wgpu 29, so this clone is a refcount bump (not a recompile); it lets
        /// the `Op` own the pipeline past the `&ComputeKernel` borrow that
        /// `add`/`add_bound` take.
        pipeline: wgpu::ComputePipeline,
        /// Index into [`CommandPass::binds`]; that Vec keeps the bind group (and
        /// its bound resources) alive for the whole recording, so the `Op` never
        /// borrows them.
        bind: usize,
        workgroups: [u32; 3],
        /// Pass label, kept only for the profiling path's per-pass
        /// `timestamp_writes` + label attribution.
        #[cfg(feature = "profiling")]
        label: String,
    },
    /// `copy_buffer_to_buffer(src, 0, dst, word_idx * 4, 4)` — closes any open
    /// pass before it runs. `wgpu::Buffer` is `Arc`-backed, so the clones are
    /// refcount bumps.
    Copy {
        src: wgpu::Buffer,
        dst: wgpu::Buffer,
        word_idx: u64,
    },
}

/// Maximum number of timestamp slots per command buffer (2 per pass). 4096 is the
/// HARD per-`QuerySet` limit on wgpu/Metal (`create_query_set` validates it), so
/// this cannot be raised. A Llama-1B decode is ~220 passes; a 48-layer MoE
/// (Qwen3-30B) is ~430 passes after the MoE fusion — both fit. Past the cap,
/// overflow dispatches simply get no timestamp (the `ts_begin` guard in
/// `submit_profiled`) so the run still completes; it just under-reports the tail.
#[cfg(feature = "profiling")]
const TS_CAPACITY: u32 = 4096;

/// A `QuerySet` plus the running cursor and the per-pass labels, so resolved
/// timestamps can be attributed back to kernels.
#[cfg(feature = "profiling")]
struct Timing {
    query_set: wgpu::QuerySet,
    cap: u32,
    next: u32,
    labels: Vec<String>,
}

#[cfg(feature = "profiling")]
impl Timing {
    fn new(ctx: &GpuContext) -> Option<Self> {
        // Runtime opt-in: no timestamps / readback unless `--profile` turned it on,
        // even in a profiling-feature build. This is the per-RUN overhead gate.
        if !ctx.ts_supported || !ctx.profiling_on() {
            return None;
        }
        let query_set = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("decode-timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: TS_CAPACITY,
        });
        Some(Timing {
            query_set,
            cap: TS_CAPACITY,
            next: 0,
            labels: Vec::new(),
        })
    }
}

impl CommandPass {
    /// Open a new command buffer to record dispatches into.
    pub fn new(ctx: &Arc<GpuContext>) -> Self {
        let encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("decode-step"),
            });
        CommandPass {
            ctx: ctx.clone(),
            encoder,
            binds: Vec::new(),
            ops: Vec::new(),
            #[cfg(feature = "profiling")]
            timing: Timing::new(ctx),
        }
    }

    /// The GPU context this pass records against (for building its uniforms).
    pub fn ctx(&self) -> &Arc<GpuContext> {
        &self.ctx
    }

    /// Record one dispatch of `kernel` over `buffers` with the given workgroup
    /// grid. Does not submit and does not encode a pass — appends a deferred
    /// `Op::Dispatch`; chain as many as needed, then [`submit`] (which coalesces
    /// consecutive dispatches into one compute pass).
    ///
    /// [`submit`]: CommandPass::submit
    pub fn add(
        &mut self,
        kernel: &ComputeKernel,
        label: &str,
        buffers: &[&wgpu::Buffer],
        workgroups: [u32; 3],
    ) {
        let bind = kernel.bind(label, buffers);
        // Hold the bind group until submit; the index keeps it alive past this
        // `&ComputeKernel` borrow and across the replay pass in `submit`.
        let idx = self.binds.len();
        self.binds.push(bind);
        self.ops.push(Op::Dispatch {
            pipeline: kernel.pipeline.clone(), // Arc refcount bump (wgpu 29)
            bind: idx,
            workgroups,
            #[cfg(feature = "profiling")]
            label: label.to_string(),
        });
    }

    /// Like [`add`](CommandPass::add), but binds arbitrary
    /// [`wgpu::BindingResource`]s rather than whole buffers — so an entry can be
    /// a *sub-range* of a buffer (e.g. a [`crate::arena::GpuArena`]
    /// region) bound at its own offset. Mix whole buffers via
    /// `buf.as_entire_binding()` and arena regions via `arena.resource(&region)`.
    pub fn add_bound(
        &mut self,
        kernel: &ComputeKernel,
        label: &str,
        resources: &[wgpu::BindingResource<'_>],
        workgroups: [u32; 3],
    ) {
        // Build the bind group now (identically to before — arena sub-ranges
        // included) so the deferred op never borrows the resources.
        let entries: Vec<wgpu::BindGroupEntry> = resources
            .iter()
            .enumerate()
            .map(|(i, r)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: r.clone(),
            })
            .collect();
        let bind = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &kernel.layout,
                entries: &entries,
            });
        // Hold the bind group until submit (it keeps its bound resources alive).
        let idx = self.binds.len();
        self.binds.push(bind);
        self.ops.push(Op::Dispatch {
            pipeline: kernel.pipeline.clone(), // Arc refcount bump (wgpu 29)
            bind: idx,
            workgroups,
            #[cfg(feature = "profiling")]
            label: label.to_string(),
        });
    }

    /// Record a dispatch with an ALREADY-BUILT bind group (P15 cache hit) —
    /// skips `create_bind_group`. The caller guarantees the bind group's captured
    /// (buffer, offset, size) entries are byte-identical to what `add_bound` would
    /// build this token (the decode graph is ctx_len-invariant for these sites;
    /// see the cache in GpuModel). The recorded `Op::Dispatch` is indistinguishable
    /// from a freshly-built one, so submit()'s pass-coalescing + hazard barriers
    /// are byte-for-byte unchanged.
    #[cfg_attr(not(feature = "profiling"), allow(unused_variables))]
    pub fn add_prebuilt(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        bind: &wgpu::BindGroup,
        label: &str,
        workgroups: [u32; 3],
    ) {
        let idx = self.binds.len();
        self.binds.push(bind.clone()); // Arc refcount bump; kept alive for submit
        self.ops.push(Op::Dispatch {
            pipeline: pipeline.clone(),
            bind: idx,
            workgroups,
            #[cfg(feature = "profiling")]
            label: label.to_string(),
        });
    }

    /// Like [`add_bound`](CommandPass::add_bound), but caches the bind group across
    /// calls via `cursor` (P15). On a cache hit (the slot is already filled) it
    /// reuses the stored bind group via `add_prebuilt` — skipping
    /// `create_bind_group`. On a miss it builds the bind group, stores it at the
    /// cursor's slot, and records it. The caller MUST present the SAME dispatch
    /// sequence every call so slot N is always the same site (decode's steady
    /// shape guarantees this); `cursor.slot` advances by one each call.
    ///
    /// CORRECTNESS: a cached bind group captures (buffer, offset, size) per entry;
    /// reusing it is valid only because every `resources` entry is byte-identical
    /// across calls (resident buffers + deterministic arena offsets). The uniform
    /// CONTENTS are refreshed by the caller's `alloc_write` before this call; the
    /// bind group references the buffer, so it sees the fresh bytes.
    /// [`add_bound_cached`](CommandPass::add_bound_cached) when `cursor` is `Some`
    /// (the steady decode shape, where the dispatch sequence repeats every token),
    /// else plain [`add_bound`](CommandPass::add_bound). Lets a decode-path call site
    /// participate in the P15 bind-group cache without the caller branching — pass
    /// `cursor.as_deref_mut()`. CRITICAL: every site that takes the cursor advances
    /// `cursor.slot`, so the set of cursor-using sites must be identical and in the
    /// same order every token (true for steady decode: fixed shapes, `token=None`).
    pub fn add_maybe_cached(
        &mut self,
        cursor: Option<&mut BindCursor<'_>>,
        kernel: &ComputeKernel,
        label: &str,
        resources: &[wgpu::BindingResource<'_>],
        workgroups: [u32; 3],
    ) {
        match cursor {
            Some(c) => self.add_bound_cached(c, kernel, label, resources, workgroups),
            None => self.add_bound(kernel, label, resources, workgroups),
        }
    }

    pub fn add_bound_cached(
        &mut self,
        cursor: &mut BindCursor<'_>,
        kernel: &ComputeKernel,
        label: &str,
        resources: &[wgpu::BindingResource<'_>],
        workgroups: [u32; 3],
    ) {
        let slot = cursor.slot;
        cursor.slot += 1;
        if slot < cursor.binds.len() {
            // Hit: reuse the cached bind group (skip create_bind_group).
            let bind = cursor.binds[slot].clone();
            self.add_prebuilt(kernel.pipeline(), &bind, label, workgroups);
        } else {
            // Miss (first decode step): build, store at this slot, record.
            debug_assert_eq!(slot, cursor.binds.len(), "cache slots must fill in order");
            let bind = kernel.bind_resources(label, resources);
            cursor.binds.push(bind.clone());
            self.add_prebuilt(kernel.pipeline(), &bind, label, workgroups);
        }
    }

    /// Record a one-`u32` copy `src[0] -> dst[word_idx]`. Appends an `Op::Copy`,
    /// which forces `submit` to close any open compute pass before encoding it
    /// (a `copy_buffer_to_buffer` cannot occur inside a pass) and to open a fresh
    /// pass for any following dispatches. Used by the decode loop to stash each
    /// step's emitted token into a resident output buffer so the whole run drains
    /// once instead of once per token.
    pub fn copy_word(&mut self, src: &wgpu::Buffer, dst: &wgpu::Buffer, word_idx: u64) {
        // `wgpu::Buffer` is Arc-backed; these clones are refcount bumps.
        self.ops.push(Op::Copy {
            src: src.clone(),
            dst: dst.clone(),
            word_idx,
        });
    }

    /// Finish recording and submit the whole command buffer in one go.
    ///
    /// Replays the deferred `ops`, coalescing each maximal run of consecutive
    /// `Dispatch`es into ONE compute pass and breaking the pass at every `Copy`.
    /// This is where the P14 win lands: ~258 pass-encodes per token collapse to
    /// ~1. Native wgpu-core inserts the per-resource hazard barriers between the
    /// dispatches within a pass on Metal, so ordering/visibility — and thus the
    /// math — are unchanged from the old one-pass-per-dispatch encoding.
    /// True when no dispatches were recorded — submitting it would create + commit an empty
    /// wgpu command buffer, which periodically triggers staging-belt/queue maintenance (a
    /// per-token spike on the full-single-queue megakernel path, where the island owns the
    /// whole token and nothing is added to the wgpu pass).
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    pub fn submit(self) {
        #[cfg(feature = "profiling")]
        if self.timing.is_some() {
            self.submit_profiled();
            return;
        }
        // Destructure so the replay owns `encoder` (a `ComputePass` mutably
        // borrows it) while reading `binds`/`ops` as disjoint shared borrows —
        // no `&mut self` method is called while a pass is open, which is exactly
        // what the in-struct-open-pass design could not express.
        let CommandPass {
            ctx,
            encoder,
            binds,
            ops,
            ..
        } = self;
        Self::encode_and_submit(&ctx, encoder, &binds, &ops);
    }

    /// Submit the dispatches accumulated SO FAR and reopen a fresh encoder, leaving
    /// the pass reusable for more `add_*` calls. This is the submit-SPLIT that keeps a
    /// long forward (e.g. a 48-layer batched prefill = ~720 dispatches) from packing
    /// every dispatch into ONE Metal command buffer — a single monolithic
    /// `queue.submit` blocks indefinitely in IOGPU `submitCommandBuffers` (the
    /// machine-burning long-context / large-model hang). Flushing every few layers
    /// matches how llama.cpp pipelines work, so each submit is a digestible size.
    ///
    /// Safe across the boundary because every buffer the dispatches read/write is a
    /// RESIDENT arena/weight owned by `GpuModel` (not transient to the pass): the
    /// submit just commits the queued GPU work; the next band's dispatches see the
    /// GPU-updated values. Only the per-pass `binds`/`ops` reset.
    ///
    /// PROFILING: the conc serve path (`forward_batch_impl`) flushes ~6×/token (every
    /// `flush_layers_hint()` layers), so those bands hold ALL the attention + MoE +
    /// dense matmuls — 99% of the real work. A plain submit here left them INVISIBLE
    /// to the profiler (only the final `submit_profiled` tail band — final_norm,
    /// gather_last, lm_head, sample — got timed, producing the degenerate "matmul
    /// 99% x1" dump). So when profiling is on, this routes the band through the SAME
    /// per-dispatch-timestamp encode as `submit_profiled` and feeds the global
    /// `profile::record_step` collector, which accumulates ALL bands → the whole
    /// token's honest per-kernel breakdown. The `Timing` query_set persists across
    /// flushes (it's a `CommandPass` field); each band resolves its own range then
    /// resets the cursor so the next band reuses the slots (one band ≈ 8 layers ≈
    /// ~130 dispatches, far under the 4096-slot cap).
    pub fn flush(&mut self) {
        if self.ops.is_empty() {
            return;
        }
        // Take the current encoder + recorded ops/binds, submit them, and reopen.
        let encoder = std::mem::replace(
            &mut self.encoder,
            self.ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("decode-step"),
                }),
        );
        let binds = std::mem::take(&mut self.binds);
        let ops = std::mem::take(&mut self.ops);
        #[cfg(feature = "profiling")]
        if let Some(t) = self.timing.as_mut() {
            // Profile this band: per-dispatch timestamps, resolve, feed record_step.
            // Reset the cursor afterwards so the next band reuses the query slots.
            Self::encode_profiled_and_record(&self.ctx, encoder, &binds, &ops, t);
            t.next = 0;
            t.labels.clear();
            return;
        }
        Self::encode_and_submit(&self.ctx, encoder, &binds, &ops);
    }

    /// Shared encode+submit: replay `ops` (coalescing consecutive dispatches into one
    /// compute pass, breaking on `Copy`) into `encoder`, then submit it. `binds` is
    /// kept alive by the caller until after submit.
    fn encode_and_submit(
        ctx: &Arc<GpuContext>,
        mut encoder: wgpu::CommandEncoder,
        binds: &[wgpu::BindGroup],
        ops: &[Op],
    ) {
        let mut i = 0;
        while i < ops.len() {
            match &ops[i] {
                Op::Dispatch { .. } => {
                    // Open ONE pass and drain the whole run of consecutive
                    // dispatches into it.
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("decode-merged"),
                        timestamp_writes: None,
                    });
                    while i < ops.len() {
                        let Op::Dispatch {
                            pipeline,
                            bind,
                            workgroups,
                            ..
                        } = &ops[i]
                        else {
                            break; // hit a Copy: close the pass (drop below)
                        };
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, &binds[*bind], &[]);
                        pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
                        i += 1;
                    }
                    // `pass` drops here -> compute pass closes, freeing `encoder`.
                }
                Op::Copy { src, dst, word_idx } => {
                    encoder.copy_buffer_to_buffer(src, 0, dst, word_idx * 4, 4);
                    i += 1;
                }
            }
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Submit, then resolve the timestamp queries and emit per-kernel GPU times
    /// as `tracing` events (per-pass at TRACE, a per-kernel and per-step summary
    /// at DEBUG, both on target `arf::gpu`). Adds a readback + device wait,
    /// so it is only on the profiling path.
    #[cfg(feature = "profiling")]
    fn submit_profiled(self) {
        let CommandPass {
            ctx,
            encoder,
            // Keep the bind groups alive until after submit + wait below.
            binds,
            ops,
            timing,
        } = self;
        let mut t = timing.expect("submit_profiled requires timing");
        // This is the FINAL band (whatever was recorded since the last flush — on the
        // conc serve path that's the tail: gather_last/final_norm/lm_head/sample). The
        // earlier bands were already profiled + recorded by `flush`. Same per-dispatch
        // timestamp encode + readback + record_step accumulation.
        Self::encode_profiled_and_record(&ctx, encoder, &binds, &ops, &mut t);
    }

    /// Encode `ops` with a per-dispatch GPU timestamp pair, submit, read the
    /// timestamps back, and feed the per-kernel aggregate to `profile::record_step`
    /// (the global collector → JSON artifact + `ARF_PROFILE_DUMP` table). Shared by
    /// `flush` (each mid-token band) and `submit_profiled` (the final band) so the
    /// WHOLE token is captured, not just the tail. `t.next`/`t.labels` accumulate the
    /// labels for THIS band; the caller resets them after if the query_set is reused.
    #[cfg(feature = "profiling")]
    fn encode_profiled_and_record(
        ctx: &Arc<GpuContext>,
        mut encoder: wgpu::CommandEncoder,
        binds: &[wgpu::BindGroup],
        ops: &[Op],
        t: &mut Timing,
    ) {
        // Replay the deferred ops, but give EACH dispatch its own pass with a
        // reserved timestamp pair — `timestamp_writes` is a pass-descriptor
        // field, so per-dispatch GPU timings require per-dispatch passes. Slot
        // order == op order == the production-build dispatch order, so the
        // aggregation below attributes ticks correctly. Copies are encoded
        // between passes (unchanged).
        for op in ops {
            match op {
                Op::Dispatch {
                    pipeline,
                    bind,
                    workgroups,
                    label,
                } => {
                    // Reserve a slot pair inline against the owned `timing`
                    // (the old `ts_reserve`/`ts_writes` `&mut self` helpers no
                    // longer fit — `self` is destructured here).
                    let ts_begin = if t.next + 1 < t.cap {
                        let b = t.next;
                        t.next += 2;
                        t.labels.push(label.clone());
                        Some(b)
                    } else {
                        None
                    };
                    let timestamp_writes = ts_begin.map(|b| wgpu::ComputePassTimestampWrites {
                        query_set: &t.query_set,
                        beginning_of_pass_write_index: Some(b),
                        end_of_pass_write_index: Some(b + 1),
                    });
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(label),
                        timestamp_writes,
                    });
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &binds[*bind], &[]);
                    pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
                }
                Op::Copy { src, dst, word_idx } => {
                    encoder.copy_buffer_to_buffer(src, 0, dst, word_idx * 4, 4);
                }
            }
        }

        // Resolve ONLY the slots actually written this band = 2 per labeled dispatch.
        // `t.next` can sit at `t.cap` after the overflow guard stops reserving (a band
        // wider than cap/2 dispatches — the conc16 prefill), but the trailing slots are
        // then UNWRITTEN; resolving an unwritten query is a Metal validation error that
        // destroys the readback buffer (the conc16 panic). `labels.len()` is the exact
        // count of reserved pairs, so this never over-resolves.
        let count = t.labels.len() as u32 * 2;
        if count == 0 {
            ctx.queue.submit(std::iter::once(encoder.finish()));
            return;
        }
        let bytes = count as u64 * std::mem::size_of::<u64>() as u64;
        let resolve = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ts-resolve"),
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ts-readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.resolve_query_set(&t.query_set, 0..count, &resolve, 0);
        encoder.copy_buffer_to_buffer(&resolve, 0, &staging, 0, bytes);
        ctx.queue.submit(std::iter::once(encoder.finish()));

        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        let view = staging.slice(..).get_mapped_range();
        let ticks: &[u64] = bytemuck::cast_slice(&view);
        let period = ctx.ts_period as f64;

        let mut agg: std::collections::BTreeMap<&str, (u32, f64)> =
            std::collections::BTreeMap::new();
        let mut total_ns = 0.0;
        // Structured per-dispatch samples for the profile collector (the HUD/JSON sink).
        let mut samples: Vec<(String, f64, Option<f64>)> = Vec::with_capacity(t.labels.len());
        for (i, label) in t.labels.iter().enumerate() {
            let begin = ticks[2 * i];
            let end = ticks[2 * i + 1];
            let ns = end.saturating_sub(begin) as f64 * period;
            total_ns += ns;
            // A matmul label is "matmul/<weight-bytes>"; split off the bytes so the
            // aggregate keys on the plain name and we can report effective GB/s
            // (bytes / gpu_us) — directly comparable to the bandwidth probe.
            // ARF_PROFILE_SHAPES=1: KEEP the full "matmul/<bytes>" label as the
            // aggregate key so the profile breaks the GEMV out PER SHAPE — reveals the
            // bimodal roofline distribution (which matmul sizes are slow) for the
            // frame-debt attack, instead of one blended "matmul" line.
            let keep_shape = std::env::var("ARF_PROFILE_SHAPES").is_ok();
            let (name, bytes) = match label.split_once('/') {
                Some((n, b)) => (
                    if keep_shape { label.as_str() } else { n },
                    b.parse::<f64>().ok(),
                ),
                None => (label.as_str(), None),
            };
            // GB/s = bytes/ns (1 byte/ns == 1 GB/s); only meaningful for matmul labels.
            let gbps = bytes.filter(|_| ns > 0.0).map(|b| b / ns);
            samples.push((name.to_string(), ns / 1000.0, gbps));
            let e = agg.entry(name).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += ns;
            match bytes {
                Some(b) if ns > 0.0 => tracing::trace!(
                    target: "arf::gpu",
                    kernel = name,
                    gpu_us = ns / 1000.0,
                    gbps = b / ns, // bytes/ns == GB/s
                ),
                _ => {
                    tracing::trace!(target: "arf::gpu", kernel = name, gpu_us = ns / 1000.0)
                }
            }
        }
        for (kernel, (calls, ns)) in &agg {
            tracing::debug!(target: "arf::gpu", kernel = *kernel, calls = *calls, total_us = ns / 1000.0);
        }
        tracing::debug!(
            target: "arf::gpu",
            passes = t.labels.len(),
            gpu_total_us = total_ns / 1000.0,
            "decode step gpu time"
        );
        // Feed the structured collector (JSON artifact + live HUD snapshot) FIRST so the
        // dump below reads the GLOBAL accumulated state (every band of every token), not
        // just this one band. On the conc serve path one token is ~6 flush bands; dumping
        // each band's local `agg` would show a degenerate slice (and 6×/token of noise).
        super::profile::record_step(&samples);
        // ARF_PROFILE_DUMP: direct stderr per-kernel table (no tracing subscriber needed) —
        // sorted by total GPU time so the CRITICAL PATH kernel jumps out. Reads the ACCUMULATED
        // snapshot so it shows the WHOLE frame (all bands: attention + MoE gate/up + down-reduce
        // + router + dense q/k/v/o + lm_head), the honest breakdown to attack (Amdahl). It
        // reprints each band, so the FINAL print of the run is the complete picture.
        if std::env::var_os("ARF_PROFILE_DUMP").is_some() {
            let snap = super::profile::snapshot();
            let rows = snap.by_time();
            let grand_us: f64 = snap.total_us.max(1e-9);
            eprintln!(
                "[prof] === per-kernel GPU time (cumulative {:.1} us over {} step-bands) ===",
                snap.total_us, snap.steps
            );
            for (kernel, st) in rows.iter().take(20) {
                eprintln!(
                    "[prof]  {:>9.1} us {:>5.1}%  x{:<6} {}",
                    st.total_us,
                    100.0 * st.total_us / grand_us,
                    st.calls,
                    kernel
                );
            }
        }
        drop(view);
        staging.unmap();
    }
}
