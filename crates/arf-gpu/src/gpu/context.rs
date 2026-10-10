//! GPU device/adapter context: feature detection (timestamps, subgroups,
//! cooperative matrix), buffer creation, and resident-weight upload.
//! Split out of gpu/mod.rs; behavior unchanged.

use super::*;

/// Pack an even-length `f32` weight slice into bf16-as-`u32` words: two weights
/// per word, even index in the low 16 bits. bf16 = the high 16 bits of the f32
/// (truncating round-toward-zero), so this is exactly the CPU `Bf16Matrix`
/// representation and the matmul kernels unpack it bit-exactly via `<< 16`.
fn pack_bf16(weight: &[f32]) -> Vec<u32> {
    assert!(
        weight.len().is_multiple_of(2),
        "bf16-packed weights need even length; got {}",
        weight.len()
    );
    weight
        .chunks_exact(2)
        .map(|pair| {
            let lo = pair[0].to_bits() >> 16;
            let hi = pair[1].to_bits() >> 16;
            lo | (hi << 16)
        })
        .collect()
}

/// A live GPU device, queue, and the matmul pipeline.
pub struct GpuContext {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    pub(crate) info: String,
    /// Nanoseconds per timestamp-query tick (`queue.get_timestamp_period()`),
    /// or `1.0` when timestamps are unavailable. Read only on the profiling path.
    #[cfg_attr(not(feature = "profiling"), allow(dead_code))]
    pub(crate) ts_period: f32,
    /// Whether `TIMESTAMP_QUERY` was requested and granted (only ever true with
    /// the `profiling` feature and an adapter that supports it).
    #[cfg_attr(not(feature = "profiling"), allow(dead_code))]
    pub(crate) ts_supported: bool,
    /// RUNTIME profiling switch. A `profiling`-feature build still pays NOTHING per
    /// step unless this is set (no timestamp queries, no per-step readback, no
    /// collector) — overhead is opt-in per RUN (`--profile`), not just per build.
    #[cfg_attr(not(feature = "profiling"), allow(dead_code))]
    pub(crate) profiling_enabled: std::sync::atomic::AtomicBool,
    /// Whether subgroup ops (`subgroupAdd`) were granted — GEMV kernels use the
    /// subgroup reduction when true, and the shared-memory tree otherwise.
    subgroups: bool,
    /// Whether cooperative-matrix ops (Apple `simdgroup_matrix` / tensor cores)
    /// were granted. The batched/prefill GEMM path uses the coop-matrix kernel
    /// when true (m≥8), the scalar tiled path otherwise — so a device without the
    /// feature is unchanged.
    coop_matrix: bool,
    /// Bytes of `create_buffer_init` weight uploads queued but not yet submitted.
    /// wgpu defers these init copies; under enough allocation churn (a multi-GB
    /// resident model like the 12B FLUX DiT) the un-submitted backlog gets dropped
    /// and the buffer silently reads zero. We track the backlog and force a queue
    /// flush once it crosses [`UPLOAD_FLUSH_BYTES`], bounding pending uploads so
    /// none are ever lost. See `flush_uploads`.
    pending_upload_bytes: std::sync::atomic::AtomicU64,
}

/// Flush queued weight uploads once this many bytes are pending (≈1 GiB). Empirically
/// the backlog survives well past this on M4 Max (3.6 GB+ of 113 MB buffers recovered
/// with a flush every 32), so 1 GiB is a conservative margin; the cost is ~1 empty
/// submit per gigabyte uploaded — negligible against multi-GB model loads.
const UPLOAD_FLUSH_BYTES: u64 = 1 << 30;

/// Relax the macOS AGX interactivity watchdog (kIOGPUCommandBufferCallbackError-
/// ImpactingInteractivity) that preempts long-running command buffers — worst on
/// our conc32/64 megakernel frames. Port of llama.cpp 7fc1c4ef7 (ggml-metal.cpp:926,
/// llama.cpp issue #20141). Must be in the env BEFORE the Metal device is created inside
/// request_adapter; llama sets it unconditionally, we keep an explicit user
/// setting (e.g. =0 to A/B the watchdog) intact.
///
/// A function (2026-09-27) because the Qwen vision encoder (`gpu/metal/qwen_vision.rs`) opens
/// the Metal device on its own, and the server loads it BEFORE the text model's `GpuContext`:
/// whichever creates the device first must have set this.
pub(crate) fn relax_agx_watchdog() {
    #[cfg(target_os = "macos")]
    if std::env::var_os("AGX_RELAX_CDM_CTXSTORE_TIMEOUT").is_none() {
        std::env::set_var("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1");
    }
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GpuContext({})", self.info)
    }
}

impl GpuContext {
    /// Acquire a GPU adapter and build the matmul pipeline.
    pub fn new() -> Result<Self> {
        relax_agx_watchdog();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .map_err(|e| ArfError::other(format!("no GPU adapter: {e}")))?;

        let info = {
            let a = adapter.get_info();
            format!("{} ({:?})", a.name, a.backend)
        };

        // The default limits cap a single buffer at 256 MB, but Llama's embedding
        // table is ~1 GB. Request the adapter's real maxima (M4 Max allows far
        // more) so the whole model fits in resident buffers.
        let adapter_limits = adapter.limits();
        let limits = wgpu::Limits {
            max_buffer_size: adapter_limits.max_buffer_size,
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
            // The sort-by-expert grouped GU kernel binds 10 storage buffers in one
            // compute stage (codes/scales/mins/dd ×2 for gate+up, plus CSR/slots), but
            // wgpu::Limits::default() caps a stage at 8 — pipeline creation would fail
            // at model build with "Too many bindings of type StorageBuffers". Request
            // the adapter's real per-stage maximum (Metal/M-series allow far more) so
            // every kernel's implicit layout fits. PORTABILITY: on an adapter whose
            // reported max is exactly 8 (the WebGPU/DX12 baseline minimum), this still
            // yields 8 and the grouped path fails at pipeline creation — only the
            // batched (non-grouped) MoE path is portable to those devices.
            max_storage_buffers_per_shader_stage: adapter_limits
                .max_storage_buffers_per_shader_stage,
            ..wgpu::Limits::default()
        };

        // Only ask for timestamp queries when profiling is compiled in and the
        // adapter actually supports them; otherwise device creation is unchanged.
        let want_ts = cfg!(feature = "profiling");
        let ts_supported = want_ts && adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let mut required_features = wgpu::Features::default();
        if ts_supported {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        // Subgroup ops (subgroupAdd / metal::simd_sum) are DISABLED BY DEFAULT.
        // Measured on Qwen3-30B Q4_K (M4 Max, same thermal window, back-to-back):
        //   feature OFF + tree kernels      14.5 tok/s   (fastest)
        //   feature ON  + tree kernels      10.9 tok/s   (1.33× slower)
        //   feature ON  + simd_sum kernels   2.8 tok/s   (5.2× slower)
        // Two independent penalties: (a) the simd_sum reduction kernels are ~3.9×
        // slower than the shared-mem tree on this GEMV shape (the op appears to
        // force full-simdgroup convergence and inhibit Metal's latency hiding), and
        // (b) merely *requesting* the SUBGROUP device feature pessimizes ALL
        // pipelines (even ones that never call a subgroup op) by ~1.33×. So we
        // neither request the feature nor select any `_sg` kernel by default; every
        // GEMV dispatch falls back to its tree variant via `.unwrap_or(&tree)`.
        // Set ARF_SUBGROUP=1 to re-enable (e.g. to re-measure the small int8
        // model where P10 (2026-06-02) once claimed a win — unverified on 30B).
        let subgroups = adapter.features().contains(wgpu::Features::SUBGROUP)
            && std::env::var("ARF_SUBGROUP").is_ok();
        if subgroups {
            required_features |= wgpu::Features::SUBGROUP;
        }

        // Cooperative-matrix (Apple `simdgroup_matrix` / NVIDIA tensor cores / AMD
        // WMMA), reachable from WGSL since wgpu 29.0.1. Measured 3.15–3.55× over the
        // tuned scalar GEMM on this GPU with NO penalty to other pipelines (unlike
        // SUBGROUP) — so we opt in whenever the adapter advertises it. The batched/
        // prefill GEMM kernel uses it (m≥8); m=1 decode is a GEMV and is unchanged.
        // Opt out with ARF_NO_COOP=1 (e.g. to A/B the path). Adapters without
        // the feature take the scalar path — portability preserved.
        let coop_matrix = adapter
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
            && std::env::var("ARF_NO_COOP").is_err();
        if coop_matrix {
            required_features |= wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX;
        }
        // PREFILL DEPENDS ON THIS. `matmul_coop_q4ks` — the tiled GEMM that makes prefill
        // compute-efficient — is only COMPILED when coop_matrix is true (weights.rs:207). If it
        // is false there is no tiled path in prefill at all and every projection falls to the
        // per-row GEMV, which is exactly the 1.5%-of-peak we measured (L160). Never let this be
        // a silent false: it is the difference between ~4 and ~100 tok/s prefill.
        // 2026-09-27: the FALSE case always prints; the true case only under ARF_VERBOSE — it
        // leaked into user commands (`arf doctor` opened with it).
        if !coop_matrix || std::env::var_os("ARF_VERBOSE").is_some() {
            eprintln!(
                "[coop] EXPERIMENTAL_COOPERATIVE_MATRIX = {coop_matrix} \
(tiled prefill GEMM {})",
                if coop_matrix {
                    "AVAILABLE"
                } else {
                    "MISSING -> prefill runs per-row GEMV"
                }
            );
        }

        // SAFETY: `ExperimentalFeatures::enabled()` is `unsafe` only because it opts
        // into not-yet-stable wgpu features (it performs no memory-unsafe operation).
        // We pair it with `EXPERIMENTAL_COOPERATIVE_MATRIX` above, which is the sole
        // experimental feature we use, and only when the adapter advertises it. This
        // is the project's only `unsafe`; everything else stays safe Rust.
        let experimental_features = if coop_matrix {
            unsafe { wgpu::ExperimentalFeatures::enabled() }
        } else {
            wgpu::ExperimentalFeatures::default()
        };

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("arf"),
            required_features,
            required_limits: limits,
            experimental_features,
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::default(),
        }))
        .map_err(|e| ArfError::other(format!("request_device failed: {e}")))?;

        let ts_period = if ts_supported {
            queue.get_timestamp_period()
        } else {
            1.0
        };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matmul_nt"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL)),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matmul_nt"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[],
                zero_initialize_workgroup_memory: false,
            },
            cache: None,
        });
        let layout = pipeline.get_bind_group_layout(0);

        Ok(GpuContext {
            device,
            queue,
            pipeline,
            layout,
            info,
            ts_period,
            ts_supported,
            profiling_enabled: std::sync::atomic::AtomicBool::new(false),
            subgroups,
            coop_matrix,
            pending_upload_bytes: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Record a deferred `create_buffer_init` upload of `bytes`. When the un-submitted
    /// backlog crosses [`UPLOAD_FLUSH_BYTES`], flush the queue so wgpu commits the
    /// uploads before they can be dropped (the silent-zero-weight bug). Called by the
    /// large-buffer init paths (`storage_init_bf16`, `storage_init_q8`, …).
    fn note_upload(&self, bytes: u64) {
        use std::sync::atomic::Ordering;
        let prev = self
            .pending_upload_bytes
            .fetch_add(bytes, Ordering::Relaxed);
        if prev + bytes >= UPLOAD_FLUSH_BYTES {
            self.pending_upload_bytes.store(0, Ordering::Relaxed);
            self.flush_uploads();
        }
    }

    /// Enable/disable per-step GPU profiling at runtime. Off by default; `--profile`
    /// flips it on. No effect without the `profiling` feature (the timing path is
    /// compiled out). When off, `CommandPass` records no timestamps and does no
    /// readback, so a profiling-feature build runs at full speed until asked.
    pub fn set_profiling(&self, on: bool) {
        self.profiling_enabled
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    #[cfg_attr(not(feature = "profiling"), allow(dead_code))]
    pub(crate) fn profiling_on(&self) -> bool {
        self.profiling_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Whether cooperative-matrix ops are available — gates the coop-matrix GEMM
    /// kernel selection in the batched/prefill matmul path. False on adapters that
    /// don't advertise the feature (those use the scalar path).
    pub fn has_coop(&self) -> bool {
        self.coop_matrix
    }

    /// Whether the `_sg` subgroup GEMV kernels should be selected. False by
    /// default — they measured 3.9× slower than the tree on Qwen3-30B Q4_K, and
    /// requesting the device feature pessimizes all pipelines by a further 1.33×
    /// (see `new`). Opt in with `ARF_SUBGROUP=1`; the field is already gated
    /// on that env var at device creation, so this is just the accessor.
    pub fn has_subgroups(&self) -> bool {
        self.subgroups
    }

    /// Human-readable adapter description.
    pub fn info(&self) -> &str {
        &self.info
    }

    // --- Phase 2 foundation: resident buffers + generic compute dispatch -------
    //
    // These primitives let a kernel keep its inputs/outputs resident on the GPU
    // and run a compute pass without per-call allocation or readback. The
    // existing `matmul_nt`/`upload_weight` path above is unchanged.

    /// A resident `STORAGE` buffer initialised from `data` (readable by shaders;
    /// also a copy source so it can be read back in tests).
    pub fn storage_init<T: Pod>(&self, label: &str, data: &[T]) -> wgpu::Buffer {
        // NOTE: deliberately does NOT call note_upload. This path is used for per-forward scratch
        // (zeroed intermediates, small uniforms-adjacent buffers) as well as small f32 weights;
        // tripping the upload-flush threshold here would drain+poll the queue mid-forward and
        // serialize the GPU (a ~100×/step pathology). Large weight uploads — the ones wgpu drops —
        // go through storage_init_bf16 / storage_init_q8, which DO note_upload. A caller loading
        // big f32 weights via this path should flush_uploads() once at end of load.
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
    }

    /// A resident `STORAGE` buffer holding `data` as bf16-packed `u32`s (two
    /// weights per word), for the matmul kernels. bf16 is the high 16 bits of
    /// each f32 — exactly what the CPU `Bf16Matrix` stores — so this is lossless
    /// to the checkpoint, and the kernels unpack it bit-exactly. `data.len()`
    /// must be even (true for every Llama weight row: k ∈ {2048, 8192, 64}).
    pub fn storage_init_bf16(&self, label: &str, data: &[f32]) -> wgpu::Buffer {
        let packed = pack_bf16(data);
        let buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&packed),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        self.note_upload(std::mem::size_of_val(packed.as_slice()) as u64);
        buf
    }

    /// Like [`Self::storage_init_bf16`] but takes weights ALREADY rounded to bf16
    /// bits (e.g. `Bf16Matrix::bits()`), skipping the 5.6 GB f32 round-trip that
    /// `storage_init_bf16(&m.to_f32())` pays on a vocab×hidden table. The bf16 bits
    /// are uploaded UNCHANGED (two per `u32`, even index low), so this is
    /// bit-identical to the `to_f32()`-then-`pack_bf16` path (which truncates
    /// already-widened bf16, a no-op) and stays exact to the CPU `Bf16Matrix`.
    /// `bits.len()` must be even (true for every weight row: cols is even).
    pub fn storage_init_bf16_from_u16(&self, label: &str, bits: &[u16]) -> wgpu::Buffer {
        assert!(
            bits.len().is_multiple_of(2),
            "bf16-packed weights need even length; got {}",
            bits.len()
        );
        let packed: Vec<u32> = bits
            .chunks_exact(2)
            .map(|pair| u32::from(pair[0]) | (u32::from(pair[1]) << 16))
            .collect();
        let buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&packed),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        self.note_upload(std::mem::size_of_val(packed.as_slice()) as u64);
        buf
    }

    /// A resident `STORAGE` buffer holding per-row int8 weights packed four to a
    /// `u32` word, for the int8 matmul kernel. `data` is row-major `[n, k]` with
    /// `k % 4 == 0` (true for every Llama weight). The matmul reads this with the
    /// matching `scale[n]` buffer; dequant `w ≈ q · scale[row]` matches the CPU
    /// `matmul_nt_q8` path exactly.
    pub fn storage_init_q8(&self, label: &str, data: &[i8]) -> wgpu::Buffer {
        assert!(
            data.len().is_multiple_of(4),
            "int8 weights need len % 4 == 0"
        );
        let packed: Vec<u32> = data
            .chunks_exact(4)
            .map(|q| {
                (q[0] as u8 as u32)
                    | ((q[1] as u8 as u32) << 8)
                    | ((q[2] as u8 as u32) << 16)
                    | ((q[3] as u8 as u32) << 24)
            })
            .collect();
        let buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&packed),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        self.note_upload(std::mem::size_of_val(packed.as_slice()) as u64);
        buf
    }

    /// A resident `STORAGE` buffer of `u8` values packed four per `u32` (low byte
    /// first), for Q4_K_S sub-block scales. `data.len() % 4 == 0`.
    /// A resident `STORAGE` buffer of `u32` words that PARTICIPATES in the periodic
    /// upload-flush (unlike the bare `storage_init`). For the large quantized-weight
    /// code buffers — e.g. the FLUX DiT's 6.66 GB of Q4_K_S codes — where a single
    /// end-of-load flush is too late and wgpu silently drops the backlog (the
    /// silent-zero-weight bug). Tracks the byte count so the backlog flushes past the
    /// 1 GiB threshold mid-load, exactly like `storage_init_bf16` / `storage_init_q8`.
    pub fn storage_init_u32_tracked(&self, label: &str, data: &[u32]) -> wgpu::Buffer {
        let buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        self.note_upload(std::mem::size_of_val(data) as u64);
        buf
    }

    pub fn storage_init_u8(&self, label: &str, data: &[u8]) -> wgpu::Buffer {
        assert!(data.len().is_multiple_of(4), "u8 buffer needs len % 4 == 0");
        let packed: Vec<u32> = data
            .chunks_exact(4)
            .map(|q| {
                (q[0] as u32) | ((q[1] as u32) << 8) | ((q[2] as u32) << 16) | ((q[3] as u32) << 24)
            })
            .collect();
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&packed),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
    }

    /// A resident `STORAGE` buffer of Q4_0 block scales: ONE bf16 scale per `u32`
    /// word (in the low 16 bits, high 16 zero), in row-major block order. One word
    /// per block — NOT 2-packed — so the q4 GEMV indexes block `b` of row `r` as
    /// word `r*(k/32) + b` with no row-straddling and no truncation for any
    /// `k % 32 == 0` (a 2-per-word packing breaks when a row's block count is odd,
    /// e.g. k=32). Scales are k/32 per row vs k/2 codes, so the 2× over a packed
    /// layout is a few percent of the codes — negligible, and bug-free.
    pub fn storage_init_q4_scales(&self, label: &str, scales: &[u16]) -> wgpu::Buffer {
        let words: Vec<u32> = scales.iter().map(|&s| s as u32).collect();
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&words),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
    }

    /// A resident, zero-initialised `STORAGE` buffer of `len` `f32`s (scratch /
    /// kernel output); a copy source so it can be read back.
    pub fn storage_zeros(&self, label: &str, len: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Create a raw resident buffer of `size_bytes` with the given usage,
    /// uninitialised. This is the building block for
    /// [`crate::arena::GpuArena`], which allocates one large buffer and
    /// then hands out many aligned sub-regions instead of creating a fresh
    /// `wgpu::Buffer` per call.
    pub fn create_buffer(
        &self,
        label: &str,
        size_bytes: u64,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size_bytes,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Upload `data` into `buffer` starting at byte `offset` (the buffer must
    /// carry `COPY_DST`). Lets a caller fill an arena sub-region without
    /// allocating a new buffer for it. `offset` and `data` must satisfy wgpu's
    /// 4-byte copy alignment.
    pub fn write_at<T: Pod>(&self, buffer: &wgpu::Buffer, offset: u64, data: &[T]) {
        self.queue
            .write_buffer(buffer, offset, bytemuck::cast_slice(data));
    }

    /// A small `UNIFORM` buffer holding one `Pod` value (kernel parameters).
    pub fn uniform_of<T: Pod>(&self, label: &str, value: &T) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::bytes_of(value),
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }

    /// Read `len` `f32`s from `src` (whole buffer). Blocks until the copy completes.
    pub fn read_f32(&self, src: &wgpu::Buffer, len: usize) -> Vec<f32> {
        self.read_f32_at(src, 0, len)
    }

    /// Submit an empty command buffer and block until the queue drains. Flushes any
    /// `create_buffer_init` upload copies wgpu has queued but not yet submitted — call
    /// after a burst of resident-weight uploads so the data is actually on the GPU
    /// before the buffers are used (otherwise large uploads can be silently dropped
    /// under allocation churn). Cheap: one empty submit + poll.
    pub fn flush_uploads(&self) {
        let enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("flush"),
            });
        self.queue.submit(std::iter::once(enc.finish()));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// Read `len` `f32`s from `src` starting at byte `offset` (e.g. an arena
    /// sub-region). Blocks until the copy completes.
    pub fn read_f32_at(&self, src: &wgpu::Buffer, offset: u64, len: usize) -> Vec<f32> {
        let bytes = (len * std::mem::size_of::<f32>()) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(src, offset, &staging, 0, bytes);
        self.queue.submit(std::iter::once(enc.finish()));
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let view = staging.slice(..).get_mapped_range();
        let out = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        staging.unmap();
        out
    }

    /// Force a set of resident buffers to be GPU-resident by referencing each in a
    /// single submitted command (a 4-byte self-copy). On the wgpu Metal backend a
    /// large buffer created with `create_buffer_init` but never referenced in a
    /// submitted command can read back as ZERO (observed loading Qwen3-30B: the
    /// embedding, uploaded first and untouched until the forward, came back all
    /// zeros → zero logits). Touching every weight buffer once at end-of-build makes
    /// them resident before the first decode. `COPY_SRC | COPY_DST` is required;
    /// every weight/scratch buffer here carries both.
    pub fn make_resident(&self, buffers: &[&wgpu::Buffer]) {
        // A 4-byte sink each buffer copies into (distinct dst avoids same-buffer
        // overlap rules). The copy references the source buffer, forcing residency.
        let sink = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resident-sink"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("make_resident"),
            });
        for b in buffers {
            if b.size() >= 4 {
                enc.copy_buffer_to_buffer(b, 0, &sink, 0, 4);
            }
        }
        self.queue.submit(std::iter::once(enc.finish()));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// Read a resident `u32` storage buffer back to the CPU (the sampled token
    /// id is one `u32`). Blocks until the copy completes.
    pub fn read_u32(&self, src: &wgpu::Buffer, len: usize) -> Vec<u32> {
        self.read_u32_at(src, 0, len)
    }

    /// Read `len` u32s starting at byte `offset` in `src` (the offset variant of `read_u32`, for
    /// reading a Region inside an arena buffer — e.g. the batched-argmax token output).
    pub fn read_u32_at(&self, src: &wgpu::Buffer, offset: u64, len: usize) -> Vec<u32> {
        let bytes = (len * std::mem::size_of::<u32>()) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback-u32"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(src, offset, &staging, 0, bytes);
        self.queue.submit(std::iter::once(enc.finish()));
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let view = staging.slice(..).get_mapped_range();
        let out = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        staging.unmap();
        out
    }

    /// A resident, zero-initialised `STORAGE` buffer of `len` `u32`s (e.g. the
    /// sample output, or block-table indices).
    pub fn storage_zeros_u32(&self, label: &str, len: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (len * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Like [`storage_zeros_u32`] but also a `COPY_DST` — for buffers written by
    /// an encoder `copy_buffer_to_buffer` (e.g. the decode loop's per-step token
    /// collector) and read back at the end.
    ///
    /// [`storage_zeros_u32`]: GpuContext::storage_zeros_u32
    pub fn storage_io_u32(&self, label: &str, len: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (len * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// A zeroed f32 scratch buffer that is BOTH a copy source AND destination (STORAGE |
    /// COPY_SRC | COPY_DST) — for intermediates the FLUX DiT splits/scatters between via
    /// [`copy_range`](GpuContext::copy_range) (qkv split, joint-stream concat).
    pub fn storage_io_f32(&self, label: &str, len: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Copy a byte range `src[src_off..src_off+bytes]` → `dst[dst_off..]` on the GPU (one
    /// submitted `copy_buffer_to_buffer`). Offsets/len must be 4-byte aligned (always true for
    /// f32 ranges). `src` needs COPY_SRC, `dst` needs COPY_DST (use `storage_io_f32` for dst).
    pub fn copy_range(
        &self,
        src: &wgpu::Buffer,
        src_off: u64,
        dst: &wgpu::Buffer,
        dst_off: u64,
        bytes: u64,
    ) {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy_range"),
            });
        enc.copy_buffer_to_buffer(src, src_off, dst, dst_off, bytes);
        self.queue.submit(std::iter::once(enc.finish()));
    }

    /// Upload a resident weight buffer of shape `[n, k]` (row-major), bf16-packed.
    ///
    /// Each weight is truncated f32→bf16 (drop the low 16 mantissa bits — exactly
    /// what `Bf16Matrix` stores on the CPU, so this is lossless to the checkpoint,
    /// not a new approximation) and two consecutive weights are packed into one
    /// `u32` (lo = even index, hi = odd). The matmul kernels unpack with a 16-bit
    /// shift, which is bit-exact. This halves the bytes read per token — and
    /// decode is bandwidth-bound on exactly these weight reads.
    ///
    /// Decision: picked bf16-pack-as-u32 from {f32, f16-via-extension,
    /// bf16-as-u32} — tradeoff: a shift/unpack in the kernel vs. either 2× the
    /// bandwidth (f32) or a non-portable shader-f16 device feature. `k` is even
    /// for every Llama weight (hidden 2048, inter 8192, head_dim 64).
    pub fn upload_weight(self: &Arc<Self>, weight: &[f32], n: usize, k: usize) -> GpuWeight {
        debug_assert_eq!(weight.len(), n * k);
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("weight-bf16"),
                contents: bytemuck::cast_slice(&pack_bf16(weight)),
                usage: wgpu::BufferUsages::STORAGE,
            });
        GpuWeight {
            ctx: self.clone(),
            buffer,
            n,
            k,
        }
    }

    /// `a [m, k]` × `weightᵀ` (`weight [n, k]`) → `[m, n]`, on the GPU.
    pub(crate) fn matmul_nt(
        &self,
        a: &[f32],
        m: usize,
        k: usize,
        weight: &wgpu::Buffer,
        n: usize,
    ) -> Vec<f32> {
        let out_len = m * n;
        let out_bytes = (out_len * std::mem::size_of::<f32>()) as u64;

        let a_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("a"),
                contents: bytemuck::cast_slice(a),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let dims = Dims {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            _pad: 0,
        };
        let dims_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dims"),
                contents: bytemuck::bytes_of(&dims),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let c_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("c"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: out_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("matmul"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            let tile = TILE as usize;
            pass.dispatch_workgroups(m.div_ceil(tile) as u32, n.div_ceil(tile) as u32, 1);
        }
        encoder.copy_buffer_to_buffer(&c_buf, 0, &staging, 0, out_bytes);
        self.queue.submit(std::iter::once(encoder.finish()));

        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let view = staging.slice(..).get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        staging.unmap();
        out
    }
}
