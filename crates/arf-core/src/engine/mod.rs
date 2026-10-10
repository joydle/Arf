//! The inference engine: glue between the scheduler, the model, and the cache.
//!
//! A [`LlmEngine::step`] does one scheduler step end to end — plan a batch, run
//! the forward pass, sample, and retire finished sequences.

use crate::cache::{slots_for, write_runs, LayerKvGeom, PagedKvCache};
use crate::config::EngineConfig;
use crate::error::{ArfError, Result};
use crate::model::batch::{ForwardBatch, SeqAttn};
use crate::model::Llama;
use crate::sampling::SamplingParams;
use crate::scheduler::scheduler::{BatchPlan, SeqPlan};
use crate::scheduler::{Request, RequestOutput, Scheduler};

/// Owns the model, the KV cache, and the scheduler.
#[derive(Debug)]
pub struct LlmEngine {
    model: Llama,
    cache: PagedKvCache,
    scheduler: Scheduler,
    next_id: u64,
}

/// What a streaming observer sees after each committed token.
///
/// Carries only data the engine already computes; the engine itself does no
/// IO, timing, or rendering. Tokens/sec is the caller's job (it owns the clock).
#[derive(Debug, Clone, Copy)]
pub struct StepInfo {
    /// The token just sampled and committed for the observed sequence.
    pub token: u32,
    /// `Some` once the sequence has finished (and why); `None` while running.
    pub finish: Option<crate::scheduler::FinishReason>,
    /// Physical KV blocks currently in use across the pool.
    pub kv_blocks_used: usize,
    /// Total physical KV blocks in the pool.
    pub kv_blocks_total: usize,
    /// Tokens of context the observed sequence has so far (prompt + generated).
    pub context_len: usize,
}

impl LlmEngine {
    /// Build an engine around a loaded `model` with runtime config `cfg`.
    pub fn new(model: Llama, cfg: EngineConfig) -> Result<Self> {
        let mc = model.config();
        // Per-layer KV geometry: Gemma 4's global layers (4 kv-heads × 512) differ
        // from its sliding layers (16 × 256), so each layer's KV buffer is sized to
        // its own width. `layer_geometry` returns the base values for every layer on
        // non-Gemma-4 models, so this is identical to the old uniform cache there.
        let geom: Vec<LayerKvGeom> = (0..mc.num_layers)
            .map(|i| {
                let (head_dim, kv_heads, _rotary) = mc.layer_geometry(i);
                LayerKvGeom { kv_heads, head_dim }
            })
            .collect();
        let cache =
            PagedKvCache::with_quant_per_layer(&geom, cfg.num_blocks, cfg.block_size, cfg.kv_quant);
        let scheduler = Scheduler::new(cfg);
        Ok(LlmEngine {
            model,
            cache,
            scheduler,
            next_id: 0,
        })
    }

    pub fn model(&self) -> &Llama {
        &self.model
    }

    /// Submit a request with an explicit id, after validating it.
    pub fn add_request(&mut self, req: Request) -> Result<()> {
        self.validate(&req.prompt, &req.params)?;
        self.scheduler.add(req);
        Ok(())
    }

    /// Submit a prompt, returning the assigned request id.
    pub fn submit(&mut self, prompt: Vec<u32>, params: SamplingParams) -> Result<u64> {
        let id = self.next_id;
        self.add_request(Request::new(id, prompt, params))?;
        self.next_id += 1;
        Ok(id)
    }

    /// Reject empty prompts, out-of-range token ids, and bad sampling params
    /// before they can reach (and panic) the model.
    fn validate(&self, prompt: &[u32], params: &SamplingParams) -> Result<()> {
        if prompt.is_empty() {
            return Err(ArfError::EmptyPrompt);
        }
        let vocab = self.model.config().vocab_size;
        if let Some(&token) = prompt.iter().find(|&&t| t as usize >= vocab) {
            return Err(ArfError::TokenOutOfRange { token, vocab });
        }
        params.validate()
    }

    /// Whether any request is still in flight.
    pub fn has_unfinished(&self) -> bool {
        self.scheduler.has_unfinished()
    }

    /// Run one scheduler step. Returns one output token per running sequence.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(level = "trace", name = "engine_step", skip_all)
    )]
    pub fn step(&mut self) -> Result<Vec<RequestOutput>> {
        let Some(plan) = self.scheduler.schedule()? else {
            return Ok(Vec::new());
        };
        let (input_ids, batch) = build_forward(&plan, self.cache.block_size());
        let hidden = self.model.forward(&input_ids, &batch, &mut self.cache);
        let logits = self.model.logits_last(&hidden, &batch);
        self.scheduler.commit(&logits)
    }

    /// Generate to completion for a single prompt, returning its output tokens.
    ///
    /// Thin wrapper over [`LlmEngine::generate_streaming`] that collects the
    /// streamed tokens into a `Vec`.
    pub fn generate(&mut self, prompt: Vec<u32>, params: SamplingParams) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        self.generate_streaming(prompt, params, &mut |info| out.push(info.token))?;
        Ok(out)
    }

    /// Generate to completion for a single prompt, invoking `on_token` for each
    /// committed token as it is produced.
    ///
    /// Drives the engine until all in-flight requests finish, reporting only the
    /// submitted request's tokens. A stop token ends generation and its token is
    /// not reported; a length-limited final token is reported with
    /// [`FinishReason::Length`](crate::scheduler::FinishReason::Length).
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(
            level = "info",
            name = "generate",
            skip_all,
            fields(prompt = prompt.len(), max_tokens = params.max_tokens)
        )
    )]
    pub fn generate_streaming(
        &mut self,
        prompt: Vec<u32>,
        params: SamplingParams,
        on_token: &mut dyn FnMut(StepInfo),
    ) -> Result<()> {
        use crate::scheduler::FinishReason;
        let prompt_len = prompt.len();
        let id = self.submit(prompt, params)?;
        let mut produced = 0usize;
        let mut done = false;
        while self.has_unfinished() && !done {
            for out in self.step()? {
                if out.id != id {
                    continue;
                }
                let kv_blocks_total = self.scheduler.num_total_blocks();
                let kv_blocks_used = kv_blocks_total - self.scheduler.num_free_blocks();
                match out.finish_reason {
                    None => {
                        produced += 1;
                        on_token(StepInfo {
                            token: out.token,
                            finish: None,
                            kv_blocks_used,
                            kv_blocks_total,
                            context_len: prompt_len + produced,
                        });
                    }
                    // A stop string's last token is output, like the budget's last token.
                    Some(reason @ (FinishReason::Length | FinishReason::StopString)) => {
                        produced += 1;
                        on_token(StepInfo {
                            token: out.token,
                            finish: Some(reason),
                            kv_blocks_used,
                            kv_blocks_total,
                            context_len: prompt_len + produced,
                        });
                        done = true;
                    }
                    Some(FinishReason::Stop) => done = true,
                    Some(FinishReason::Evicted) => done = true,
                }
            }
        }
        Ok(())
    }
}

/// Group a sequence plan's image rows into contiguous `(local_start, len)` spans (local to
/// the seq's query range). `image_rows[i].0` is a flat batch row `= q_start + local`; a break
/// in the consecutive-row sequence delimits separate images. Empty for text sequences.
fn image_spans_for(sp: &crate::scheduler::SeqPlan) -> Vec<(usize, usize)> {
    // Qwen3.8 (an M-RoPE layout) keeps image tokens CAUSAL: no bidirectional span. So does
    // Qwen3-Omni audio (`image_causal`, plain positions).
    if sp.image_rows.is_empty() || sp.mrope.is_some() || sp.image_causal {
        return Vec::new();
    }
    let mut rows: Vec<usize> = sp.image_rows.iter().map(|(r, _)| *r).collect();
    rows.sort_unstable();
    let mut spans = Vec::new();
    let mut start = rows[0];
    let mut len = 1usize;
    for w in rows.windows(2) {
        if w[1] == w[0] + 1 {
            len += 1;
        } else {
            spans.push((start - sp.q_start, len));
            start = w[1];
            len = 1;
        }
    }
    spans.push((start - sp.q_start, len));
    spans
}

/// Convert a logical [`BatchPlan`] into model inputs. Public because the
/// serving actor (arf-serve) drives the same scheduler loop against a GPU
/// backend  and needs the identical plan->tensor translation.
///
/// Preconditions: `block_size` MUST equal the `block_size()` of the scheduler
/// that produced `plan` (a mismatch silently computes wrong slots/write_runs),
/// and each plan entry's blocks must already be allocated to cover
/// `past_len + q_len` positions (true by construction for plans from
/// `Scheduler::schedule`).
pub fn build_forward(plan: &BatchPlan, block_size: usize) -> (Vec<u32>, ForwardBatch) {
    build_forward_cached(plan, block_size, &mut SlotsCache::default())
}

/// L3 (2026-08-04) — per-sequence slots cache for [`build_forward_cached`].
///
/// `build_forward` rebuilt every sequence's FULL `slots` vec (`slots_for`: an O(ctx)
/// allocate + div/mod fill) on EVERY decode step, and on the daemon's m=1 megakernel path the
/// consumer then only re-verifies block alignment and discards it. This cache retires that
/// per-token O(ctx) cost: the caller hands each step's vecs back via [`SlotsCache::reclaim`]
/// (O(1) `Vec` moves — no copy), and the next step extends the held vec by `q_len` new
/// positions instead of rebuilding.
///
/// CORRECTNESS (the produced vec is byte-identical to `slots_for`): a held vec is reused only
/// when (a) its length is exactly `past_len`, and (b) every covered block start matches the
/// CURRENT plan's `block_table` (`v[j*bs] == tbl[j]*bs`). Every vec the cache holds was built
/// by the `slots_for` formula (invariant: entries come only from `slots_for` or the extend
/// below), so within a block the offsets are affine — block-start equality implies full
/// equality. Any mismatch (preempted-and-reallocated seq, chunk-size change, prefix-cache
/// rewind) falls back to a fresh `slots_for` build. Multi-seq batches use the same per-seq
/// logic, so the general path stays byte-identical too.
#[derive(Debug, Default)]
pub struct SlotsCache(std::collections::HashMap<u64, Vec<u32>>);

impl SlotsCache {
    /// The slots vec for `sp` covering `[0, past_len + q_len)` — extending the held vec when
    /// provably identical (see type docs), else a fresh `slots_for` build.
    fn take_slots(&mut self, sp: &SeqPlan, bs: usize) -> Vec<u32> {
        let ctx = sp.past_len + sp.q_len;
        if let Some(mut v) = self.0.remove(&sp.id) {
            let ok = v.len() == sp.past_len
                && v.chunks(bs)
                    .zip(sp.block_table.iter())
                    .all(|(c, &b)| c[0] == b * bs as u32);
            if ok {
                v.extend(
                    (sp.past_len..ctx)
                        .map(|p| sp.block_table[p / bs] * bs as u32 + (p % bs) as u32),
                );
                return v;
            }
        }
        slots_for(&sp.block_table, bs, ctx)
    }

    /// Take this step's slots vecs back from `batch` (O(1) moves) so the next step can extend
    /// them. Call after the forward pass, once `batch` is no longer read. Entries for seqs not
    /// in `plan` are dropped first — a running seq appears in every plan (continuous batching),
    /// so this bounds the cache to the live batch; a preempted-then-resumed seq just rebuilds.
    pub fn reclaim(&mut self, plan: &BatchPlan, batch: &mut ForwardBatch) {
        self.0.clear();
        for (sp, seq) in plan.seqs.iter().zip(batch.seqs.iter_mut()) {
            self.0.insert(sp.id, std::mem::take(&mut seq.slots));
        }
    }
}

/// [`build_forward`] with a step-to-step [`SlotsCache`] (the serving actor's hot loop). Same
/// outputs, byte-identical; only the slots allocation strategy differs.
pub fn build_forward_cached(
    plan: &BatchPlan,
    block_size: usize,
    cache: &mut SlotsCache,
) -> (Vec<u32>, ForwardBatch) {
    let seqs = plan
        .seqs
        .iter()
        .map(|sp| {
            SeqAttn {
                // L162 — carry the scheduler's stable sequence id so a recurrent (gated-delta-net)
                // layer can follow the SEQUENCE across steps instead of the positional row slot.
                stream_id: Some(sp.id),
                q_start: sp.q_start,
                q_len: sp.q_len,
                past_len: sp.past_len,
                slots: cache.take_slots(sp, block_size),
                write_runs: write_runs(&sp.block_table, block_size, sp.past_len, sp.q_len),
                // Vision: group this seq's image rows (sorted ascending flat rows) into
                // contiguous spans `(local_start, len)`, where `local_start = flat_row -
                // q_start` is the offset within q_len. Each image's 256 tokens are contiguous,
                // so a gap in the flat-row sequence starts a new span (a second image).
                image_spans: image_spans_for(sp),
            }
        })
        .collect();
    // Vision: gather every sequence's image rows (flat_row, embedding) into one flat
    // ImageEmbeds for the GPU embed-override. None when no sequence has image tokens.
    let img_rows: Vec<&(usize, Vec<f32>)> = plan
        .seqs
        .iter()
        .flat_map(|sp| sp.image_rows.iter())
        .collect();
    let image_embeds = if img_rows.is_empty() {
        None
    } else {
        let hidden = img_rows[0].1.len();
        let mut embeds = Vec::with_capacity(img_rows.len() * hidden);
        let mut rows = Vec::with_capacity(img_rows.len());
        for (row, emb) in &img_rows {
            rows.push(*row);
            embeds.extend_from_slice(emb);
        }
        Some(crate::model::batch::ImageEmbeds {
            embeds,
            hidden,
            rows,
        })
    };
    (
        plan.input_ids.clone(),
        ForwardBatch {
            positions: plan.positions.clone(),
            seqs,
            image_embeds,
            mrope_positions: mrope_positions_for(plan),
        },
    )
}

/// Flatten the plan's per-sequence M-RoPE rows into one per-token table, or `None` when no
/// sequence has a layout (every text batch: byte-identical to before M-RoPE existed). Sequences
/// without a layout contribute `[p, p, p]` from their plain position.
fn mrope_positions_for(plan: &BatchPlan) -> Option<Vec<[u32; 3]>> {
    if plan.seqs.iter().all(|sp| sp.mrope.is_none()) {
        return None;
    }
    let mut out: Vec<[u32; 3]> = plan.positions.iter().map(|&p| [p, p, p]).collect();
    for sp in &plan.seqs {
        if let Some(rows) = &sp.mrope {
            out[sp.q_start..sp.q_start + sp.q_len].copy_from_slice(rows);
        }
    }
    Some(out)
}
