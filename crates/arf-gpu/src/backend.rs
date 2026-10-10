//! [`arf_core::backend::BatchedBackend`] implementation for the wgpu engine.
//!
//! This thin wrapper is the ONLY type the serving actor needs from this crate
//! (via `Box<dyn BatchedBackend>`), keeping the server loop free of wgpu types
//! . `step` IS overridden here — the one change the design v1 anticipated:
//! all-plain-greedy batches sample on the GPU (batched argmax; only B token-ids
//! read back instead of the full [B × vocab] logits), and any non-greedy /
//! penalized / constrained row falls back to the trait's default CPU sampling.

use arf_core::backend::BatchedBackend;
use arf_core::error::Result;
use arf_core::model::batch::ForwardBatch;
use arf_core::sampling::SeqSampling;

#[cfg(target_os = "macos")]
use crate::gpu::BatchPending;
use crate::gpu::GpuModel;

/// The resident GPU model as a batched serving backend.
pub struct WgpuBatched(pub GpuModel);

/// DEPTH-2 PIPELINE API — the serve loop's window into the submit/read split, keeping the
/// raw Metal `MTLBuffer` bank references entirely inside this crate. Banks are addressed by index
/// (`0`/`1`) instead of raw pointers; the token relay between steps is GPU-side (`TokenSrc::GpuBank`),
/// so the loop never round-trips a token through the CPU on the overlapped path. macOS-only (the
/// island is Metal); the whole block is `#[cfg]`'d out elsewhere and the loop stays depth-1.
#[cfg(target_os = "macos")]
impl WgpuBatched {
    /// Whether THIS step can run the depth-2 overlap (island up, both banks allocated, MoE, KV not
    /// quantized, pure q_len==1 decode). The caller ALSO checks `all_plain_greedy` on the sampling
    /// params — greedy is required (the island argmax is what writes the bank). Non-eligible → the
    /// loop falls back to depth-1 `step`.
    /// L364 — macOS-only: this is the island's depth-2 / k-step submission API. Every method
    /// here calls into `GpuModel`'s Metal-gated internals (`island`, `out_bank`,
    /// `submit_batched_step`, `read_pending`), so off macOS they do not exist at all rather
    /// than existing and failing. Their only consumers are the island benchmarks
    /// (`examples/serve_loop_bench.rs`) and the actor's depth-2 path, both macOS-only.
    #[cfg(target_os = "macos")]
    pub fn depth2_eligible(&self, batch: &ForwardBatch) -> bool {
        self.0.depth2_eligible(batch)
    }

    /// L118 — the island, for the honest GPU-busy probe (`ARF_BMEGA_GPUBUSY=1`). Diagnostic
    /// only: the bench snapshots `take_gpubusy()` at the end of a run.
    #[cfg(target_os = "macos")]
    pub fn island_for_probe(
        &self,
    ) -> Option<&std::sync::Mutex<crate::gpu::concurrent_metal::MetalIsland>> {
        self.0.island.as_ref()
    }

    /// Submit ONE batched decode step (record + async-commit, NO wait/read), returning the pending
    /// handle. `prev_bank` selects the token source: `Some(i)` reads the previous step's sampled ids
    /// from bank `i` GPU-side (the overlapped path); `None` reads `ids` from the CPU (the priming
    /// step, byte-identical to depth-1's `TokenSrc::Cpu`). The step's sampled ids are written into
    /// `out_bank(cur_bank)`. `None` if the step is ineligible / a bank is missing / ARF_NO_BATCH_MEGA.
    #[cfg(target_os = "macos")]
    pub fn submit_depth2(
        &self,
        prev_bank: Option<usize>,
        cur_bank: usize,
        ids: &[u32],
        batch: &ForwardBatch,
    ) -> Option<BatchPending> {
        use crate::gpu::concurrent_metal::TokenSrc;
        let out = self.0.out_bank(cur_bank)?;
        let token_src = match prev_bank {
            Some(i) => TokenSrc::GpuBank(self.0.out_bank(i)?),
            None => TokenSrc::Cpu(ids),
        };
        self.0.submit_batched_step(token_src, batch, out)
    }

    /// K-STEP SUBMIT: chain `k` decode steps into ONE record (no CPU seam between them) so
    /// the GPU runs continuously → DVFS ramps the pinned clock. `batch` MUST already carry each seq's
    /// slots extended to `past_len+k` (the burst's k new contiguous slots, within one KV block — the
    /// caller caps k to the block boundary). Reads the first step's tokens from CPU `ids`
    /// (`TokenSrc::Cpu`); steps 1..k read the previous chained step's ids GPU-side. Writes into
    /// `out_bank(cur_bank)` step-major (out_bank[ks*b+r]). Returns a pending whose `read_depth2`
    /// yields `k*b` tokens. `None` if ineligible / bank missing.
    #[cfg(target_os = "macos")]
    pub fn submit_kstep(
        &self,
        cur_bank: usize,
        ids: &[u32],
        batch: &ForwardBatch,
        k: usize,
    ) -> Option<BatchPending> {
        use crate::gpu::concurrent_metal::TokenSrc;
        let out = self.0.out_bank(cur_bank)?;
        self.0
            .submit_batched_step_k(TokenSrc::Cpu(ids), batch, out, k)
    }

    /// Fence a submitted step and read its `b` sampled ids out of bank `bank` (MUST be the
    /// `cur_bank` the matching `submit_depth2` wrote). A light CPU spin on the completion fence —
    /// NOT a GPU drain — so the NEXT step (already submitted) keeps the GPU warm while this waits.
    #[cfg(target_os = "macos")]
    pub fn read_depth2(&self, pending: &BatchPending, bank: usize) -> Vec<u32> {
        // The bank must be present (submit_depth2 already validated it); fall back to an empty
        // read only on the impossible missing-bank case rather than panicking the serve loop.
        match self.0.out_bank(bank) {
            Some(out) => self.0.read_pending(pending, out),
            None => Vec::new(),
        }
    }
}

impl BatchedBackend for WgpuBatched {
    fn forward_batch(&self, input_ids: &[u32], batch: &ForwardBatch) -> Vec<Vec<f32>> {
        self.0.forward_batch(input_ids, batch)
    }

    fn kv_geometry(&self) -> (usize, usize) {
        (self.0.kv.num_blocks, self.0.kv.block_size)
    }

    fn quiesce(&self) {
        self.0.quiesce();
    }

    fn greedy_only(&self) -> Option<&'static str> {
        self.0.island_only_moe().then_some(
            "this MoE model's experts are served by the Metal island, which runs greedy steps \
             only on this build (ARF_MOE_WGPU_FULL=1 adds a second ~16 GB copy of the experts for \
             the other path; on a 36 GB Mac that swaps)",
        )
    }

    /// G1 SEAM: delegate to the B-row verify megakernel. `try_batched_megakernel_verify` already
    /// honours ARBITRARY paged block tables — L11 removed the old `identity &&` gate that had kept
    /// spec dead in the daemon (the LIFO allocator hands every post-warmup request recycled blocks,
    /// so `identity=false` from the first real request onward). It returns `None` on any decline
    /// (k out of range, non-MoE, KV quantized, …), which the caller must treat as "no spec this
    /// step" and fall back to ordinary decode.
    #[cfg(target_os = "macos")]
    fn mtp_draft(&self, tok: u32, past_len: usize, slots: &[u32]) -> Option<u32> {
        self.0.mtp_draft(tok, past_len, slots)
    }

    #[cfg(target_os = "macos")]
    fn state_save(&self, stream: u64, key: u64) -> (bool, Option<u64>) {
        self.0.state_save(stream, key)
    }

    #[cfg(target_os = "macos")]
    fn state_save_anchor(&self, stream: u64, key: u64) -> (bool, Option<u64>) {
        self.0.state_save_anchor(stream, key)
    }

    #[cfg(target_os = "macos")]
    fn state_save_junction(&self, stream: u64, key: u64) -> (bool, Option<u64>) {
        self.0.state_save_junction(stream, key)
    }

    #[cfg(target_os = "macos")]
    fn state_save_checkpoint(&self, stream: u64, key: u64) -> (bool, Option<u64>) {
        self.0.state_save_checkpoint(stream, key)
    }

    #[cfg(target_os = "macos")]
    fn state_restore(&self, stream: u64, key: u64) -> bool {
        self.0.state_restore(stream, key)
    }

    #[cfg(target_os = "macos")]
    fn prefix_export(&self, key: u64, slots: &[u32]) -> std::result::Result<Vec<u8>, String> {
        self.0.prefix_export(key, slots)
    }

    #[cfg(target_os = "macos")]
    fn prefix_import(
        &self,
        key: u64,
        slots: &[u32],
        blob: &[u8],
    ) -> std::result::Result<Option<u64>, String> {
        self.0.prefix_import(key, slots, blob)
    }

    fn block_draft(&self, tok: u32, past_len: usize, stream: u64) -> Vec<u32> {
        self.0.dflash_draft(tok, past_len, stream)
    }

    #[cfg(target_os = "macos")]
    fn release_stream(&self, stream: u64) {
        self.0.release_stream(stream)
    }

    #[cfg(target_os = "macos")]
    fn set_prefill_logits_needed(&self, needed: bool) {
        crate::gpu::PREFILL_LOGITS_NEEDED.store(needed, std::sync::atomic::Ordering::Relaxed);
    }

    #[cfg(target_os = "macos")]
    fn block_draft_sampled(
        &self,
        tok: u32,
        past_len: usize,
        stream: u64,
        temperature: f32,
        seed: u64,
    ) -> Option<(Vec<u32>, Vec<Vec<(u32, f32)>>)> {
        self.0
            .dflash_draft_sampled(tok, past_len, stream, temperature, seed)
    }

    #[cfg(target_os = "macos")]
    fn block_draft_ready(&self, stream: u64, past_len: usize) -> bool {
        self.0.dflash_ready(stream, past_len)
    }

    #[cfg(target_os = "macos")]
    fn block_draft_gpu(&self, tok: u32, past_len: usize, stream: u64) -> Option<usize> {
        self.0.dflash_draft_gpu(tok, past_len, stream)
    }

    #[cfg(target_os = "macos")]
    fn block_draft_gpu_sampled(
        &self,
        tok: u32,
        past_len: usize,
        stream: u64,
        temperature: f32,
        seed: u64,
    ) -> Option<usize> {
        self.0
            .dflash_draft_gpu_sampled(tok, past_len, stream, temperature, seed)
    }

    #[cfg(target_os = "macos")]
    fn block_draft_gpu_sampled_waited(
        &self,
        tok: u32,
        past_len: usize,
        stream: u64,
        temperature: f32,
        seed: u64,
    ) -> Option<(Vec<u32>, Vec<Vec<(u32, f32)>>)> {
        self.0
            .dflash_draft_gpu_sampled_waited(tok, past_len, stream, temperature, seed)
    }

    #[cfg(target_os = "macos")]
    fn verify_window_gpu_draft(
        &self,
        k: usize,
        prefix_len: usize,
        prefix_slots: &[u32],
        stream_id: u64,
    ) -> Option<(Vec<u32>, Vec<u32>)> {
        self.0
            .verify_window_gpu_draft(k, prefix_len, prefix_slots, stream_id, None)
    }

    #[cfg(target_os = "macos")]
    fn verify_window_gpu_draft_sampled(
        &self,
        k: usize,
        prefix_len: usize,
        prefix_slots: &[u32],
        stream_id: u64,
        sampler: &arf_core::sampling::RowSampler,
    ) -> Option<(Vec<u32>, Vec<u32>)> {
        self.0
            .verify_window_gpu_draft(k, prefix_len, prefix_slots, stream_id, Some(sampler))
    }

    #[cfg(target_os = "macos")]
    fn verify_window_sampled(
        &self,
        window: &[u32],
        prefix_len: usize,
        prefix_slots: &[u32],
        stream_id: u64,
        sampler: &arf_core::sampling::RowSampler,
    ) -> Option<Vec<u32>> {
        self.0
            .verify_sampled(window, prefix_len, prefix_slots, stream_id, sampler)
    }

    /// L337 — the chained MTP draft (see `GpuModel::mtp_draft_chain`).
    #[cfg(target_os = "macos")]
    fn mtp_draft_chain(&self, tok: u32, past_len: usize, slots: &[u32], k: usize) -> Vec<u32> {
        self.0.mtp_draft_chain(tok, past_len, slots, k)
    }

    fn verify_window(
        &self,
        window: &[u32],
        prefix_len: usize,
        prefix_slots: &[u32],
        stream_id: u64,
    ) -> Option<Vec<u32>> {
        // L364 — the verify record is an island path; off macOS the caller falls back to the
        // WGSL `forward_batch_all` oracle, which is what `None` means here on Mac too.
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (window, prefix_len, prefix_slots, stream_id);
            None
        }
        #[cfg(target_os = "macos")]
        self.0
            .try_batched_megakernel_verify(window, prefix_len, prefix_slots, Some(stream_id))
    }

    #[cfg(target_os = "macos")]
    fn verify_segments(&self, segs: &[arf_core::backend::SegReq<'_>]) -> Option<Vec<Vec<u32>>> {
        self.0.verify_segments(segs, &[])
    }

    /// Sampled segments' rows are drawn from the record's own logits (see
    /// `GpuModel::verify_segments`).
    #[cfg(target_os = "macos")]
    fn verify_segments_sampled(
        &self,
        segs: &[arf_core::backend::SegReq<'_>],
        samplers: &[Option<&arf_core::sampling::RowSampler>],
    ) -> Option<Vec<Vec<u32>>> {
        self.0.verify_segments(segs, samplers)
    }

    /// ON-GPU GREEDY FAST PATH: when EVERY sequence is plain greedy (temperature ≤ 0) with no
    /// repetition penalty (which would need the CPU logits to edit them pre-argmax), sample on the
    /// GPU via the batched argmax and read back ONLY B token-ids — instead of the default's full
    /// [B × vocab] logits readback + blocking poll, which stalls the GPU between tokens (the
    /// dominant single-stream gap vs llama.cpp). Any non-greedy / penalized / constrained row falls
    /// back to the default CPU-sampling path (full readback), preserving exact behavior there.
    fn step(
        &self,
        input_ids: &[u32],
        batch: &ForwardBatch,
        sampling: &[SeqSampling],
    ) -> Result<Vec<u32>> {
        // A draft may only condition on a hidden that THIS step's own record publishes.
        #[cfg(target_os = "macos")]
        self.0.mtp_h_src.set(None);
        let all_plain_greedy = !sampling.is_empty()
            && sampling.iter().all(|s| {
                s.params.is_greedy() && (s.params.repetition_penalty - 1.0).abs() < f32::EPSILON
            });
        // EMBEDDED ROWS WITHOUT AN M-RoPE LAYOUT (2026-09-27, Qwen3-Omni audio; also Gemma-3
        // images) leave the greedy path. Its megakernel entries (`try_m1_megakernel`,
        // `try_batched_megakernel`, which takes a single-sequence prefill chunk as B rows) embed
        // by TOKEN ID and never read `batch.image_embeds`, so an audio prompt would be answered
        // as if every `<|audio_pad|>` were text — silently, at full speed. `forward_batch`
        // (greedy = false) skips every one of them (they sit under `if greedy && ..`) and reaches
        // the generic loop, which splices the rows; the CPU sampler then takes the argmax, the
        // same token. Only a prefill chunk holding such rows is affected: decode steps carry
        // none, so they keep the fast path. A Qwen3.8 image batch (`mrope_positions: Some`)
        // keeps this path: its greedy-only route (`forward_mrope_batch`) does the splice.
        // NOT RUN ON A GPU when written: see the M3 gate list in the commit.
        // CORRECTED the same day, measured on the GPU: on Metal the generic loop's MoE experts are
        // an L109 placeholder (the Omni Thinker's first spoken question hit that panic guard), so
        // with a Metal island the greedy path splices the rows itself (`forward_batch_impl`,
        // "EMBEDDED ROWS WITHOUT M-RoPE"). Only a backend with no island still needs the loop.
        let embeds_need_generic = batch.image_embeds.is_some()
            && batch.mrope_positions.is_none()
            && (!self.0.has_island() || batch.seqs.iter().any(|s| !s.image_spans.is_empty()));
        if all_plain_greedy && !embeds_need_generic {
            // forward_batch is the fast single-stream path: at conc=1 steady decode it ALREADY uses
            // the batch_binds bind-group cache (batch.rs steady branch) AND the batched MoE kernel
            // (all routed experts in one dispatch), plus on-GPU greedy argmax (no full-vocab
            // readback). Routing to decode_token here was SLOWER — decode_token's MoE uses the
            // per-expert serial path (~1152 dispatches/token), the 404ms catastrophe. Kept simple.
            return Ok(self.0.forward_batch_greedy(input_ids, batch));
        }
        // Default: forward + CPU sample (logprobs / temperature / top-k/p / penalty / constraints).
        let logits = self.forward_batch(input_ids, batch);
        arf_core::sampling::sample_batch(&logits, sampling)
    }
}
