//! `arf-serve`: a PERSISTENT, resident-model, OpenAI-compatible inference
//! HTTP server for the from-scratch wgpu/Metal arf engine.
//!
//! The whole point: load the GPU model ONCE at startup and keep it resident
//! across requests. The CLI reloads the (up to 18 GB) model on every invocation
//! and thrashes under memory pressure; this server is the ollama-style persistent
//! daemon that fixes that.
//!
//! ## Architecture (single-owner model actor)
//!
//! [`arf_gpu::GpuModel`] is `Send` but **`!Sync`** (three `RefCell` fields
//! on the hot decode path), so it cannot be shared across async tasks. We never
//! share it: one dedicated OS thread owns it for the whole process and is the
//! only code that touches it (see [`actor`]). axum handlers (async, on tokio)
//! build a [`actor::Job`], send it to the actor over a channel, and stream the
//! generated tokens back to the HTTP client. Concurrent requests are continuously
//! batched: each step folds every live sequence into one batched forward pass
//! .

use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;

use crate::{actor, chat, http};
use arf_core::backend::{BatchedBackend, ImageEncoder};
use arf_core::Tokenizer;
use clap::Parser;

#[cfg(feature = "wgpu")]
use crate::model_loader;

/// The context default, chosen so that it never turns a model that loads into one that does not.
///
/// 32K was the ask (2026-09-20) and the 27B earns it: facts planted at the start, middle and end
/// of a 19,868-token prompt come back 3 of 3. But `--max-context` raises the KV pool to one full
/// sequence, Metal wires a buffer WHOLE the first time it is referenced (measured: 6.4 / 8.5 /
/// 13 GB idle at 1,024 / 2,048 / 4,096 blocks — no lazy pages, residency set or not), and the
/// pool's size per block varies ~8x across models: gemma-4-31B's 8K pool is already 22.5 GB. A
/// flat 32K would stop the dense models loading at all on a 36 GB Mac.
///
/// So: 32,768 if weights (the file's size — they are stored near the target quant) plus the
/// 2,048-block pool stay under 70% of RAM, where the loader's own thrash warning starts. Else
/// 16,384, which the default `--num-blocks 1024` pool ALREADY holds — the old 8,192 default was
/// leaving half of it unreachable. A smaller explicit `--num-blocks` keeps the old 8,192 floor.
#[cfg(feature = "wgpu")]
fn auto_max_context(args: &Args, cfg: &arf_core::config::ModelConfig) -> usize {
    const BLOCK: usize = 16;
    let ram = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok());
    let weights = std::fs::metadata(&args.model).map(|m| m.len()).unwrap_or(0);
    // SPARSE 8-bit KV (2026-09-23): memory is mapped as a sequence grows, so a context costs
    // nothing until it is used — the load-time residency bounds below (measured against pools
    // committed at load) do not apply to it. Serve the model's native context. MEASURED: a
    // committed 262K pool on the 36 GB box ran free-form at 7.4 tok/s (draft wall 130 ms vs GPU
    // 16 ms — its buffers no longer stayed resident); the same 262K as a sparse pool, 42.3 tok/s.
    // A sequence that really grows past the GPU working set still pages; the pool says so in the
    // log at the step it crosses.
    if arf_gpu::kv_sparse_for(cfg) {
        let chosen = cfg.max_position_embeddings.clamp(16_384, 262_144);
        eprintln!(
            "--max-context auto: {chosen} tokens (the model's native context: the 8-bit KV cache is \
             SPARSE, ~{:.0} KB a token mapped only as a sequence grows; ARF_NO_SPARSE_KV=1 for the \
             committed pool and its memory-sized ceiling) — pass --max-context to override",
            arf_gpu::weights::kv_pool_bytes(cfg, 1) as f64 / BLOCK as f64 / 1024.0
        );
        return chosen;
    }
    let want = if arf_gpu::kv_q8_for(cfg) {
        131_072usize
    } else {
        32_768
    };
    // WITH A BLOCK DRAFT the working set grows by the draft (its projections at 4 bits are ~1/3
    // of the published bf16 file, plus rings and scratch) and the margin is gone. MEASURED on the
    // 36 GB Mac, 2026-09-20: at a 32K pool (24.1 GB in all) every draft command buffer waited
    // ~16 ms to START — GPU 21 ms, wall 37-40 ms, verify 124 -> 150+ ms, decode 19 -> 11-14 tok/s
    // — while 24K (23.1 GB) and 16K ran wall == GPU. The buffers no longer all stay resident.
    // So with a draft the ladder is 32K / 24K / 16K against a tighter bound, placed between the
    // two measured points (this rule counts neither the 2.9 GB embedding/lm_head).
    // 2026-09-23 — THE EMBEDDING CREDIT. The bounds below were measured with the token embedding
    // held as bf16 (2.54 GB on Qwen3.8-27B) and never counted it. When the GGUF ships it as Q4_K
    // the engine now keeps it at 0.72 GB (weights.rs, `ARF_NO_EMBED_Q4K`), and the difference
    // is memory the KV pool can have: wired pages measured 1,849,319 vs 1,936,839 (16 KB) at 24K.
    let embed_credit: u64 =
        if std::env::var_os("ARF_NO_EMBED_Q4K").is_none() && !cfg.tie_word_embeddings {
            arf_core::model::gguf::LazyGguf::open_raw(&args.model)
                .ok()
                .and_then(|g| {
                    g.get_q4k_blocks_raw("token_embd.weight")
                        .map(|(b, r, c)| ((r * c * 2) as u64).saturating_sub(b.len() as u64))
                })
                .unwrap_or(0)
        } else {
            0
        };
    if let Some(dir) = &args.draft {
        let file = std::fs::metadata(dir.join("model.safetensors")).map_or(0, |m| m.len());
        let draft = file / 3 + (128 << 20);
        // With the 8-bit KV cache (default on this arch, `kv_q8_for`) the pool is ~4x smaller per token, so the SAME
        // measured bound admits ~4x the context (2026-09-23).
        let ladder: &[usize] = if arf_gpu::kv_q8_for(cfg) {
            &[131_072, 98_304, 65_536, 49_152, 32_768, 24_576, 16_384]
        } else {
            &[32_768, 24_576, 16_384]
        };
        let pick = ladder.iter().copied().find(|ctx| {
            let pool =
                arf_gpu::weights::kv_pool_bytes(cfg, ctx.div_ceil(BLOCK).max(args.num_blocks));
            ram.is_some_and(|r| {
                weights > 0
                    && (weights + pool + draft).saturating_sub(embed_credit) as f64
                        <= r as f64 * 0.57
            })
        });
        let chosen = pick.unwrap_or_else(|| (args.num_blocks * BLOCK).clamp(8_192, 16_384));
        eprintln!(
            "--max-context auto: {chosen} tokens (weights {:.1} GB + block draft {:.1} GB + its KV \
             pool must stay resident: under 57% of RAM; Q4_K embedding credit {:.1} GB) — pass \
             --max-context to override",
            weights as f64 / 2f64.powi(30),
            draft as f64 / 2f64.powi(30),
            embed_credit as f64 / 2f64.powi(30),
        );
        return chosen;
    }
    let pool = arf_gpu::weights::kv_pool_bytes(cfg, want.div_ceil(BLOCK).max(args.num_blocks));
    let fits = ram.is_some_and(|r| {
        weights > 0 && (weights + pool).saturating_sub(embed_credit) as f64 <= r as f64 * 0.70
    });
    let chosen = if fits {
        want
    } else {
        (args.num_blocks * BLOCK).clamp(8_192, 16_384)
    };
    eprintln!(
        "--max-context auto: {chosen} tokens (weights {:.1} GB + a {}K KV pool {:.1} GB {} 70% of RAM){}",
        weights as f64 / 2f64.powi(30),
        want / 1024,
        pool as f64 / 2f64.powi(30),
        if fits { "fits under" } else { "would exceed" },
        if fits { "" } else { " — pass --max-context to override" },
    );
    chosen
}

#[derive(Parser)]
#[command(
    name = "arf-serve",
    about = "Persistent, resident-model, OpenAI-compatible inference server for arf",
    version
)]
struct Args {
    /// Path to a safetensors file/dir, a single-file GGUF blob, or a BUNDLE directory
    /// (`model.gguf` + optional `draft/`, `mmproj.gguf`, `arf-bundle.txt` — what
    /// `arf pull qwen3.8:27b` writes). A bundle's draft and projector are attached and its
    /// `--arch` / `--quant` filled in unless given here; see `arf_core::bundle`.
    #[arg(long)]
    model: PathBuf,

    /// Model architecture when there is no sibling `config.json` (required for a
    /// GGUF blob): `qwen3.8-27b`, `muse-glimmer-30b`, `qwen3-coder-30b`, `gemma-4-31b`,
    /// `gemma-4-12b`, `gemma-3-4b`, `llama-3.2-1b` (`arf_core::config::ARCHITECTURES`;
    /// an unknown name is rejected with the full list).
    #[arg(long)]
    arch: Option<String>,

    /// Path to `tokenizer.json`. Optional for a single-file GGUF model (its embedded
    /// tokenizer is used when this is omitted). Required for safetensors models.
    #[arg(long)]
    tokenizer: Option<PathBuf>,

    /// Path to the multimodal vision projector GGUF (`mmproj-*.gguf`). Gemma-3: the SigLIP
    /// tower is loaded onto the GPU. Qwen3.8 (`clip.projector_type = qwen3vl_merger`, detected
    /// from the file): the vision encoder runs on the CPU in the HTTP layer and image prompts use
    /// M-RoPE positions. Either way chat requests may then carry `image_url` parts and
    /// `/v1/messages` `image` blocks. Auto-attached only from a bundle's `mmproj.gguf`
    /// (`--no-mmproj` to skip); otherwise omit for text-only serving.
    #[arg(long)]
    mmproj: Option<PathBuf>,

    /// Do not attach a bundle's `mmproj.gguf` (text-only serving, less memory).
    #[arg(long, conflicts_with = "mmproj")]
    no_mmproj: bool,

    /// Qwen3-Omni AUDIO input: the Thinker's audio encoder as safetensors (`thinker.audio_tower.*`
    /// — the checkpoint's shard 1, or the extract `scripts/ref/qwen3_omni/fetch_audio_tower.py`
    /// writes). Not an `--mmproj`: llama.cpp's Omni mmproj carries the tower re-encoded (convs
    /// f16; its Q8_0 file quantizes the layers), while the encoder's gates were run against the
    /// HF tensors. With it, chat requests may carry `input_audio` (OpenAI) and `audio_url`
    /// (vLLM) parts; the encoder runs on the GPU (`ARF_AUDIO_CPU=1` for the CPU one).
    #[arg(long)]
    audio_tower: Option<PathBuf>,

    /// Weight quantization: `none` (bf16), `int8`, `q4`, `q4k`, `q4ks`, or `q3k`. Unset =
    /// the bundle's `arf-bundle.txt` value, else `none`.
    #[arg(long)]
    quant: Option<String>,

    /// KV-cache quantization: `none` (f32) or TurboQuant `tq2`/`tq3`/`tq4`
    /// (2/3/4-bit packed KV; append `q` for the QJL residual, e.g. `tq3q`).
    /// Shrinks the KV cache and the attention read bandwidth that bounds
    /// long-context / large-batch decode.
    #[arg(long, default_value = "none")]
    kv_quant: String,

    /// TCP port to bind.
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Host/IP to bind.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Render the live dashboard (auto-on when stdout is a TTY; use --no-tty for
    /// plain logs under systemd/Docker).
    #[arg(long)]
    no_tty: bool,

    /// RoPE table size / max context. Also caps per-request `max_tokens`. `0` (the default) =
    /// AUTO: 32,768 when the KV pool that needs fits beside the weights, else 16,384 — which
    /// the default `--num-blocks 1024` pool already holds, so it costs nothing. See
    /// `auto_max_context`.
    #[arg(long, default_value_t = 0)]
    max_context: usize,

    /// Total KV-cache blocks in the pool (the concurrency ceiling).
    /// Bumped automatically to at least one full --max-context sequence.
    #[arg(long, default_value_t = 1024)]
    num_blocks: usize,

    /// Max sequences scheduled concurrently in a batch.
    #[arg(long, default_value_t = 32)]
    max_batch_size: usize,

    /// L341 — MTP speculative decoding: draft `k` tokens from the model's own multi-token
    /// prediction head, then verify them in one pass. The verify decides and the draft only
    /// proposes, so the output is greedy's — up to NEAR-TIES: the verify computes logits in
    /// another batch shape, and where greedy's top two tokens are a hair apart the pick can
    /// differ (CORRECTED 2026-10-05: this said "byte-identical by construction"; a 0.067-nat tie
    /// in a reasoning trace flipped, measured 2026-10-05). No quality cost either way.
    ///
    /// Unset = AUTO (L363o): k=1 when the model ships its own draft head (`nextn_layers > 0`,
    /// i.e. Qwen3.8's in-model MTP head), else 0. `--speculative 0` forces it off; 1..3 forces it
    /// on. 1 is the measured sweet spot on a hybrid; higher k is accepted but does not help
    /// (measured (L338) — the marginal verify row costs 0.52 of a full pass).
    ///
    /// HISTORY: this was OFF by default (L339 measured 1.14x on a quiet box, L341 1.04x under
    /// memory pressure — speculation adds two CPU sync points per token and degrades first when
    /// the CPU is starved). L348 then settled the range on a rebooted box with a 0.05% control:
    /// **1.09x, byte-identical output, and 1.04x is the worst case ever measured — still a win**.
    /// Models WITHOUT a draft head stay at 0: for them k>0 would engage the n-gram suffix
    /// drafter, whose verify rows cost the same 0.52 for a far lower acceptance rate.
    #[arg(long)]
    speculative: Option<usize>,

    /// Per-step token budget; long prompts prefill in chunks of this size
    /// mixed with decode .
    #[arg(long, default_value_t = 512)]
    max_prefill_tokens: usize,

    /// Override the model id reported by `/v1/models` and echoed in responses.
    /// Defaults to the model path's file name.
    #[arg(long)]
    served_model_name: Option<String>,

    /// Override the model's `vocab_size`. A GGUF blob may carry a slightly
    /// different vocab than the built-in `--arch` config (some gemma variants
    /// differ by a handful of tokens); set this to the GGUF's embedding rows.
    #[arg(long)]
    vocab: Option<usize>,

    /// DISABLE automatic prefix caching . Prefix caching reuses the KV
    /// of block-aligned shared prompt prefixes across requests (repeated system
    /// prompt / few-shot preamble prefilled once) — bit-identical outputs, a big
    /// TTFT + throughput win on shared-prefix workloads (measured 3.7× tok/s,
    /// ~11× lower TTFT @conc32). It is ON BY DEFAULT; pass this flag to turn it
    /// off (e.g. to isolate a measurement or under adversarial unique-prompt load).
    ///
    /// REFUSED on hybrid recurrent models (Qwen3.8-27B) regardless of this flag: a hit
    /// restores KV but not the recurrent state and returned wrong text (2026-09-19).
    #[arg(long, default_value_t = false)]
    no_prefix_cache: bool,

    /// A DFlash 2 block-draft checkpoint for this model (a directory holding `config.json` and
    /// `model.safetensors`, e.g. `incoai/Qwen3.8-27B-DFlash2`): one draft pass proposes the next
    /// 7 tokens and the model verifies them in one window. Single-stream speed; greedy text is
    /// unchanged up to near-ties (see `--speculative`; corrected 2026-10-05). macOS / Metal only. See the port plan.
    /// Auto-attached from a bundle's `draft/` on Metal builds (`--no-draft` to skip).
    #[arg(long)]
    draft: Option<PathBuf>,

    /// Do not attach a bundle's `draft/` (plain decode, less memory).
    #[arg(long, conflicts_with = "draft")]
    no_draft: bool,
}

/// Resolve a bundle `--model` (a directory holding `model.gguf`, or a GGUF inside one) into the
/// flags it implies: the weights file, `--draft`, `--mmproj`, `--arch`, `--quant`. Flags given on
/// the command line win; `--no-draft` / `--no-mmproj` opt out. Logs ONE line naming what was
/// attached, and nothing at all for a launch that involves no bundle — so the benchmark arms that
/// pass `--model <file>.gguf --draft <dir>` start exactly as before.
///
/// Runs BEFORE everything that reads `args`: `resolve_config` given the bundle DIRECTORY would find
/// no `config.json`, see no GGUF magic, and silently fall back to the Llama-3.2-1B config; and
/// `auto_max_context` sizes the KV pool from `args.draft`.
fn apply_bundle(args: &mut Args) -> Result<(), String> {
    let r = arf_core::bundle::resolve(&arf_core::bundle::Request {
        model: args.model.clone(),
        arch: args.arch.clone(),
        quant: args.quant.clone(),
        draft: args.draft.clone(),
        mmproj: args.mmproj.clone(),
        no_draft: args.no_draft,
        no_mmproj: args.no_mmproj,
        // The block draft runs on Metal only (`metal_backend`), and the Qwen3.8 encoder needs the
        // Metal backend (`load_qwen_vision`); elsewhere the bundle's parts are skipped with a note
        // rather than failing a load that works without them.
        draft_supported: cfg!(all(feature = "wgpu", target_os = "macos")),
        mmproj_supported: cfg!(all(feature = "wgpu", target_os = "macos")),
    })?;
    if let Some(line) = r.summary() {
        eprintln!("{line}");
    }
    if args.served_model_name.is_none() {
        args.served_model_name = r.served_name();
    }
    args.draft = r.draft.path().map(PathBuf::from);
    args.mmproj = r.mmproj.path().map(PathBuf::from);
    args.arch = r.arch.map(|(a, _)| a);
    args.quant = r.quant.map(|(q, _)| q);
    args.model = r.model;
    Ok(())
}

// ---------------------------------------------------------------------------
// Backend factory : the backend is chosen by the binary that calls `run` — this
// crate's own `main.rs` passes `metal_backend`; an embedding binary passes its own
// factory from its own binary, building this crate without the `wgpu` feature.
// The actor/scheduler/http above this seam are untouched — they accept
// `Box<dyn BatchedBackend>` and know nothing of the backend.
// ---------------------------------------------------------------------------

/// What a backend factory is given: the resolved model config, the KV geometry the scheduler
/// was built with, and the command-line paths it may need. The scheduler's block ids must stay
/// inside the pool the backend allocates from `engine_cfg` (`actor::spawn` asserts it).
pub struct BackendRequest<'a> {
    /// `--model`, after bundle resolution.
    pub model: &'a std::path::Path,
    /// `--mmproj`, when given (a Gemma-3 SigLIP projector or a Qwen3.8 one).
    pub mmproj: Option<&'a std::path::Path>,
    /// `--draft`, when given (the block draft; Metal only).
    pub draft: Option<&'a std::path::Path>,
    /// `--max-context`, after the auto rule.
    pub max_context: usize,
    /// The resolved model config.
    pub cfg: &'a arf_core::config::ModelConfig,
    /// The engine config the scheduler runs with.
    pub engine_cfg: arf_core::EngineConfig,
    /// Weight quantization.
    pub quant: arf_core::config::Quant,
    /// KV-cache quantization.
    pub kv_quant: arf_core::config::KvQuant,
}

/// The backend a factory returns, and its image encoder if it has one.
pub type Backends = (Box<dyn BatchedBackend>, Option<Box<dyn ImageEncoder>>);

/// Builds the resident backend once, at startup. [`metal_backend`] is this crate's own; an
/// embedding binary passes its own backend factory to [`run`] from its own binary.
pub type BackendFactory = fn(BackendRequest<'_>) -> Result<Backends, Box<dyn Error>>;

/// The wgpu/Metal backend: `arf-serve`'s own, and what `main.rs` runs.
#[cfg(feature = "wgpu")]
pub fn metal_backend(req: BackendRequest<'_>) -> Result<Backends, Box<dyn Error>> {
    let BackendRequest {
        cfg,
        engine_cfg,
        quant,
        kv_quant,
        ..
    } = req;
    use arf_gpu::GpuContext;
    let ctx = Arc::new(GpuContext::new()?);
    // Enable the GPU profiler so /profile serves REAL per-kernel data to the HUD. The
    // overhead is a per-dispatch timestamp; acceptable for the live-cockpit use case.
    ctx.set_profiling(true);
    eprintln!("loading model onto gpu: {} ...", ctx.info());
    let model = model_loader::load_resident(
        cfg,
        req.model,
        quant,
        kv_quant,
        req.max_context,
        &ctx,
        Some(engine_cfg.num_blocks),
    )?;
    if let Some(dir) = req.draft {
        // an OS without Metal 4 tensor ops (before macOS 26) cannot run the draft: serve the
        // model without it rather than refuse to start
        #[cfg(target_os = "macos")]
        if !model.dflash_supported() {
            eprintln!(
                "block draft skipped: {} needs the Metal 4 tensor ops of macOS 26; serving without it",
                dir.display()
            );
        } else {
            let t = std::time::Instant::now();
            model
                .dflash_attach(dir)
                .map_err(|e| format!("--draft {}: {e}", dir.display()))?;
            eprintln!(
                "block draft attached: {} ({:.1} s)",
                dir.display(),
                t.elapsed().as_secs_f64()
            );
        }
        #[cfg(not(target_os = "macos"))]
        return Err(format!(
            "--draft {}: the block draft runs on Metal only",
            dir.display()
        )
        .into());
    }
    // Vision (Gemma-3): when an mmproj is provided, build the SigLIP tower on the SAME
    // GpuContext. The projector output dim must match the LM hidden size to splice in.
    let vision: Option<Box<dyn ImageEncoder>> = match req.mmproj {
        // Qwen3.8's encoder is loaded by `main` (`load_qwen_vision`), GPU or CPU, not here.
        Some(path) if arf_core::model::qwen_vision::is_qwen3vl_mmproj(path) => None,
        Some(path) => {
            eprintln!("loading vision tower (mmproj): {} ...", path.display());
            let vp = arf_gpu::gpu::vision::VisionPipeline::load(&ctx, path, cfg.hidden_size)?;
            eprintln!(
                "vision ready: {} soft-tokens × {} dim per image",
                vp.num_tokens(),
                cfg.hidden_size
            );
            Some(Box::new(vp))
        }
        None => None,
    };
    Ok((Box::new(arf_gpu::WgpuBatched(model)), vision))
}

/// Minimal `.env` loader (no `dotenvy` dependency). Reads `./.env` if present and,
/// for each non-comment `KEY=VALUE` line, sets the env var ONLY when it is not
/// already set in the process environment (real env always wins). Surrounding
/// single/double quotes are stripped from the value. Best-effort: a missing or
/// unreadable file is silently ignored. SECURITY: the HF_TOKEN value is NEVER
/// printed — only whether it is present.
fn load_dotenv() {
    if let Ok(contents) = std::fs::read_to_string("./.env") {
        for line in contents.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((key, value)) = line.split_once('=') {
                let key = key.trim();
                let value = value.trim().trim_matches('"').trim_matches('\'');
                if key.is_empty() || std::env::var_os(key).is_some() {
                    continue;
                }
                std::env::set_var(key, value);
            }
        }
    }
    eprintln!(
        "HF_TOKEN: {} (HuggingFace model search)",
        if std::env::var_os("HF_TOKEN").is_some() {
            "set"
        } else {
            "unset"
        }
    );
}

/// The whole server: parse the command line, resolve the model, build the backend with
/// `build_backend`, and serve until the process exits. `main.rs` calls it with
/// [`metal_backend`].
pub fn run(build_backend: BackendFactory) -> Result<(), Box<dyn Error>> {
    // FIRST, before anything allocates: see the function for the 4.87 GB this is worth.
    reexec_without_malloc_large_cache();
    load_dotenv();
    // Before anything reads the sparse-memory switches: a recent GPU driver panic that named this
    // server turns grow-on-use memory off (`gpu_panic`, #60).
    if let Some(line) = crate::gpu_panic::apply() {
        eprintln!("{line}");
    }
    // L359 — name any ARF_* set in the environment that this binary does not read. A typo in
    // one of the 240 levers is otherwise a silent no-op that looks exactly like a feature which
    // did not help; that is precisely how L333's fabricated "+20%" survived to publication.
    arf_core::env_registry::warn_unknown_env_vars();
    // FAST PATH BY DEFAULT — the whole env-defaulting block (MSL_GEMV / MEGAKERNEL /
    // SINGLEQ+BLOCKING / the concurrent win config + conditional KV_F16) moved VERBATIM to
    // `arf_gpu::apply_fast_path_defaults` so `arf run` / `arf generate` (in-process
    // CLI chat) get the same measured-winner defaults instead of silently taking the slow
    // portable path. Same conditions, same order, same messages (prefixed `[serve]`); the
    // per-lever history and measurements are documented at the function. Feature-gated: on
    // a build without the `wgpu` feature arf-gpu is not linked, and these are Metal-island levers.
    // Applied once the config is resolved (below), so a model the island cannot run is declined
    // from its own config first — not only when `--arch` happens to be on the launch line.
    let mut args = Args::parse();
    apply_bundle(&mut args)?;

    // --- Resolve config + load the resident model ONCE (the heavy step). -------
    #[cfg(feature = "wgpu")]
    let mut cfg = model_loader::resolve_config(&args.model, args.arch.as_deref())?;
    #[cfg(feature = "wgpu")]
    {
        arf_gpu::decline_fast_path_for_model("serve", &cfg);
        arf_gpu::apply_fast_path_defaults("serve");
    }
    #[cfg(feature = "wgpu")]
    if args.max_context == 0 {
        args.max_context = auto_max_context(&args, &cfg);
    }
    // A build without the `wgpu` feature has no auto rule yet (it needs the wgpu crate's pool arithmetic): keep
    // the historical default rather than letting `0` reach the RoPE table and the token cap.
    #[cfg(not(feature = "wgpu"))]
    if args.max_context == 0 {
        args.max_context = 8192;
    }

    // Without the `wgpu` feature: model_loader is not compiled; inline the config resolution.
    #[cfg(not(feature = "wgpu"))]
    let mut cfg = {
        match args.arch.as_deref() {
            Some(arch) => {
                // ONE table, in arf-core. This match used to live here too and had drifted
                // from the CLI's copy — the server rejected qwen3.5/3.6/3.8 and Gemma-4, which
                // `arf generate` accepted.
                let c = arf_core::config::config_for_arch(arch)?;
                c.validate()?;
                c
            }
            None => {
                return Err(
                    "a backend built without the wgpu feature requires --arch (model_loader is not available)"
                        .into(),
                )
            }
        }
    };

    if let Some(v) = args.vocab {
        eprintln!("overriding vocab_size {} -> {v}", cfg.vocab_size);
        cfg.vocab_size = v;
    }

    #[cfg(feature = "wgpu")]
    let quant = model_loader::parse_quant(args.quant.as_deref().unwrap_or("none"))?;
    // CACHE-KEY FIX: the weight-cache flags_hash reads the QUANT *env var* (so the bench, which is
    // launched `QUANT=q4ks ...`, keys correctly). The daemon takes --quant as a CLI flag and never
    // set that env var, so its flags_hash saw QUANT="" → a DIFFERENT key than the bench built → the
    // daemon MISSED the cache and rebuilt the ~18GB sidecar (~3-4 min) on EVERY start (which looked
    // like "the chat hangs"). Mirror the flag into the env BEFORE the model load computes the key so
    // the daemon HITs the same cache the bench built. Only set if unset (don't clobber an explicit env).
    if std::env::var_os("QUANT").is_none() {
        std::env::set_var("QUANT", args.quant.as_deref().unwrap_or("none"));
    }
    // Qwen3.8 is served with ITS OWN chat template (reasoning-effort system message + a
    // pre-opened `<think>`), see `chat::qwen3_think_template`. Mirrored into the env the same
    // way as QUANT; `ARF_NO_QWEN_THINK_TEMPLATE=1` is the A/B opt-out.
    if args
        .arch
        .as_deref()
        .is_some_and(|a| a.starts_with("qwen3.8"))
        && std::env::var_os("ARF_QWEN_THINK_TEMPLATE").is_none()
    {
        std::env::set_var("ARF_QWEN_THINK_TEMPLATE", "1");
    }
    // L341 — mirror --speculative into ARF_SPEC_K the same way, and for the same reason:
    // `spec_k()` in actor.rs is a process-wide OnceLock read from the environment, so a CLI
    // flag alone would parse and then do nothing. Set BEFORE the actor's first read. An
    // explicit env var wins (don't clobber someone's ARF_SPEC_K=2 experiment).
    let spec_k = match args.speculative {
        Some(k) => k,
        None if cfg.nextn_layers > 0 => 1,
        None => 0,
    };
    if std::env::var_os("ARF_SPEC_K").is_none() {
        eprintln!(
            "[serve] speculation: k={spec_k} ({}; --speculative 0 disables, --speculative N forces)",
            match (args.speculative, cfg.nextn_layers > 0) {
                (Some(_), _) => "explicit",
                (None, true) => "AUTO — the model ships an MTP draft head (L348: 1.09x, byte-identical)",
                (None, false) => "AUTO — no draft head in this model",
            }
        );
        if spec_k > 0 {
            std::env::set_var("ARF_SPEC_K", spec_k.to_string());
        }
    }
    // L343 — size the GEMV accumulator stride to the REAL batch ceiling, which is what L249
    // said should happen and nothing ever did: `ACC_ROWS` defaulted to MAXB=64, so COLS=4 burned
    // 4*64 = 1 KB of registers per lane with 60 of every 64 slots dead. Sized properly, COLS=8
    // fits in the same registers COLS=4 uses today.
    //
    // ⚠️ CORRECTNESS (the L142/L156c failure mode): rows >= ACC_ROWS are NEVER WRITTEN, so this
    // MUST be >= the largest `d.m` ever DISPATCHED — which is not the same as the largest batch.
    // Prefill runs at b=256 but is ROW-CHUNKED to GEMV_Q4KS_MAXB(64) before dispatch
    // (concurrent_metal.rs:3133), so no single dispatch ever exceeds 64. Decode and verify
    // records are bounded by --max-batch-size. Hence: max(max_batch_size, 1) capped at 64, and
    // the cap is the SAME constant the chunker uses, not a coincidence.
    //
    // Measured (L343, b=1, 3 samples each): COLS=8 with ACC_ROWS=64 SPILLS at 0.92x; with
    // ACC_ROWS sized it is 1.019x and climbing. COLS stays at its shipped 4 here — this only
    // removes the dead register stride, it does not raise COLS.
    if std::env::var_os("ARF_GEMV_ACC_ROWS").is_none() {
        const GEMV_MAXB: usize = 64;
        let acc_rows = args.max_batch_size.clamp(1, GEMV_MAXB);
        std::env::set_var("ARF_GEMV_ACC_ROWS", acc_rows.to_string());
    }
    #[cfg(feature = "wgpu")]
    let kv_quant = model_loader::parse_kv_quant(&args.kv_quant)?;

    #[cfg(not(feature = "wgpu"))]
    let quant = arf_core::config::Quant::None;
    #[cfg(not(feature = "wgpu"))]
    let kv_quant = arf_core::config::KvQuant::None;

    let tokenizer = Arc::new(match &args.tokenizer {
        Some(path) => Tokenizer::from_file(path)?,
        None => {
            // No --tokenizer: a single-file GGUF carries its own, so read that.
            if !arf_core::model::gguf::is_gguf_file(&args.model) {
                // L349 — a safetensors model directory ships `tokenizer.json` NEXT TO the
                // weights. Requiring --tokenizer for it was a papercut with no upside: the
                // sibling file is unambiguous, and `arf serve <dir>` failing on a model
                // the CLI had just pulled is the worst possible first-run experience.
                // Look for the sibling; only error if it genuinely is not there.
                let sib = std::path::Path::new(&args.model);
                let cand = if sib.is_dir() {
                    sib.join("tokenizer.json")
                } else {
                    sib.parent()
                        .unwrap_or(std::path::Path::new("."))
                        .join("tokenizer.json")
                };
                if cand.is_file() {
                    eprintln!("no --tokenizer: using {}", cand.display());
                    Tokenizer::from_file(&cand)?
                } else {
                    return Err(format!(
                        "no --tokenizer given, the model is not a GGUF blob, and no \
                         tokenizer.json sits beside it (looked for {})",
                        cand.display()
                    )
                    .into());
                }
            } else {
                eprintln!("no --tokenizer: using the GGUF-embedded tokenizer");
                Tokenizer::from_gguf_path(&args.model)?
            }
        }
    });
    let stop_tokens = Arc::new(chat::chat_stop_tokens(&tokenizer));
    // The model's own sampling defaults (GGUF `general.sampling.*`), for sampled requests that
    // omit top_k / top_p (2026-09-26; see `http::build_sampling`).
    if let Ok(g) = arf_core::model::gguf::LazyGguf::open_raw(&args.model) {
        let top_k = g
            .get_metadata_u32("general.sampling.top_k")
            .map(|k| k as usize);
        let top_p = g.get_metadata_f32("general.sampling.top_p");
        if top_k.is_some() || top_p.is_some() {
            eprintln!("[serve] model sampling defaults (GGUF): top_k {top_k:?}, top_p {top_p:?}");
        }
        http::set_model_sampling_defaults(top_k, top_p);
    }
    // The model's OWN chat template (the GGUF's `tokenizer.chat_template`), compiled once: requests
    // with tools or tool history render through it when it teaches a tool format Arf parses.
    // Logged either way, so the server log says which tool protocol requests will get.
    let chat_template = crate::jinja_chat::load(&args.model, &tokenizer).map(Arc::new);
    // System One (`/v1/systemone`): a decision model's template, temperatures and labels.
    let systemone = http::systemone::DecisionSupport::load(&args.model, &tokenizer);
    match &systemone {
        http::systemone::DecisionSupport::Ready(_) => {
            eprintln!("[serve] decision model: /v1/systemone enabled (openjev)")
        }
        http::systemone::DecisionSupport::Unsupported(why) => {
            eprintln!("[serve] decision model, /v1/systemone unavailable: {why}")
        }
        http::systemone::DecisionSupport::None => {}
    }

    // One source of truth for the KV geometry: the scheduler config and the GPU
    // pool are BOTH built from `engine_cfg` (spec §3 memory-safety: a block id
    // the scheduler hands out must always be inside the GPU pool — actor::spawn
    // asserts it).
    const KV_BLOCK_SIZE: usize = 16; // the engine's fixed KV page size
    let min_blocks = args.max_context.div_ceil(KV_BLOCK_SIZE);
    let num_blocks = args.num_blocks.max(min_blocks);
    if num_blocks > args.num_blocks {
        eprintln!(
            "--num-blocks {} raised to {num_blocks} (one full --max-context sequence)",
            args.num_blocks
        );
    }
    // ⚠️ KV-POOL UNDERSIZE WARNING (2026-08-11) — the guard above only guarantees ONE full
    // sequence fits. It says nothing about `max_batch_size` sequences running CONCURRENTLY, and
    // when the pool cannot hold them the allocator evicts blocks that a RUNNING sequence still
    // needs. The symptom is not an error: the victim reads KV another request overwrote and emits
    // fluent GARBAGE ("Bug Bug Bug ..."), while every throughput number looks perfectly healthy.
    //
    // MEASURED: 16 concurrent agents, ~940-token shared prompt, gen 256, default --num-blocks 1024
    // → needs ~1200 blocks → 1/16 agents corrupted, and it was the agent owning the SHARED PREFIX
    // blocks. Found by a text gate on agent requests; invisible to tok/s. Passing --num-blocks 2048
    // made it clean (2026-08-11).
    //
    // Warn rather than auto-raise: the pool is real GPU memory, so silently allocating several GB
    // more than asked is its own hazard. Tell the operator the number they need.
    {
        // A conservative per-sequence estimate: the typical serving shape is a prompt of a few
        // hundred tokens plus `max_tokens` of generation, not a full `max_context`.
        let per_seq_blocks = 2048usize.div_ceil(KV_BLOCK_SIZE); // ~2k tokens/sequence
        let want = per_seq_blocks.saturating_mul(args.max_batch_size);
        // L146/L147 — the hardcoded KV_BLOCKS_KNOWN_GOOD_MAX=2048 that lived here is GONE.
        // The real guard is now in the engine (`wired_memory_guard`, weights.rs): it prices the
        // KV pool alongside the weights against 80% of physical RAM, refuses an over-commit, and
        // NAMES the largest --num-blocks that fits on this box. That scales with the actual
        // machine and model instead of being an empirical constant that would be wrong on any
        // other box. ARF_FORCE_LOAD=1 overrides it.
        if num_blocks < want {
            // The engine's wired_memory_guard will refuse a pool that does not fit and name the
            // size that does, so advise the ideal here and let that guard arbitrate.
            let advise = want;
            let tail = String::new();
            eprintln!(
                "⚠️  --num-blocks {num_blocks} may be UNDERSIZED for --max-batch-size {}: \
                 {} concurrent 2k-token sequences need ~{want} blocks. Under full load the \
                 allocator can evict blocks a RUNNING sequence still needs, which produces \
                 CORRUPT OUTPUT (fluent garbage), not an error. Consider --num-blocks {advise}.{tail}",
                args.max_batch_size, args.max_batch_size
            );
        }
    }
    if args.max_prefill_tokens < args.max_batch_size {
        eprintln!(
            "warning: --max-prefill-tokens {} < --max-batch-size {} — under full decode \
             load, sequences beyond the budget are starved FCFS each step",
            args.max_prefill_tokens, args.max_batch_size
        );
    }
    // 🔴 PREFIX CACHING IS REFUSED ON A HYBRID (RECURRENT) MODEL (2026-09-19).
    //
    // The prefix cache shares KV BLOCKS. On Qwen3.8-27B only 16 of 64 layers have KV; the other
    // 48 are GDN layers whose state is a recurrence over EVERY token so far. A cache hit skips
    // the cached tokens' prefill, which is right for the attention layers and leaves the
    // recurrent layers having never seen those tokens. Nothing in the scheduler or block
    // manager snapshots or restores recurrent state, so the hit is silently wrong.
    //
    // MEASURED (measured 2026-09-19): same daemon, same prompt, temperature 0,
    // five requests. Cache ON: request 0 correct (96 tokens), requests 1-4 return 14 tokens of
    // UNRELATED CHINESE TEXT, md5-identical to each other, 4 of 4. `--no-prefix-cache`: 5 of 5
    // identical and correct. It was on by default, so every repeated prompt and every shared
    // system prompt hit it — and it was invisible to tok/s, which IMPROVED (TTFT 0.85 s -> 0.1 s).
    //
    // TO LIFT THIS: snapshot the recurrent state (GDN state + conv ring) at cached block
    // boundaries and restore it on a hit, the way another engine's StateCache does. Real work, and not
    // needed for correctness today. Until then a wrong answer served fast is not a feature.
    //
    // LIFTED 2026-09-21 on the Metal build: `metal/state_snapshots.rs` snapshots one sequence's
    // recurrent bank row (and the block draft's context ring) at a block boundary the scheduler
    // picks near the end of each prompt, and a hit is only taken down to a boundary that has a
    // snapshot (`EngineConfig::state_snapshots`). The refusal stands wherever that is missing.
    let hybrid = matches!(cfg.attn, arf_core::config::AttnKind::HybridSsmAttn { .. });
    // ✅ LIFTED 2026-09-21, and the two bugs that kept it refused are both fixed and gated.
    // A hit on a hybrid model is valid only where the backend holds a recurrent-state SNAPSHOT
    // (`metal/state_snapshots.rs`), and that snapshot is sound only when BOTH hold:
    //   1. it is published at the same moment as the KV blocks covering the same tokens — i.e.
    //      when the sequence FINISHES (`Sequence::pending_snapshot`). Announcing it at capture
    //      time let a later turn attach to KV that was still being written.
    //   2. it sits on a multiple of the backend's PREFILL WINDOW (128). The recurrence advanced
    //      one window at a time, so a boundary inside a window could not be resumed from.
    // Both were MEASURED failures on turn 2 of a conversation, and both are covered by
    // `scripts/prefix_cache_hybrid.py` (facts recalled, repeats faster, every answer identical to
    // --no-prefix-cache or parting only at a near-tie). `--no-prefix-cache` still turns it off.
    // 2. NO LONGER HOLDS since 2026-10-05: boundaries are KV-block aligned (prompt windows run
    // through the row-stepped verify record; the scheduler's `build_plan` has the gate results).
    let state_snapshots = hybrid && cfg!(all(feature = "wgpu", target_os = "macos"));
    let prefix_cache = !args.no_prefix_cache && (!hybrid || state_snapshots);
    eprintln!(
        "automatic prefix caching: {} (shared prompt prefixes prefilled once)",
        if prefix_cache && state_snapshots {
            "ON — hybrid model: hits are taken only down to a block-aligned \
             recurrent-state snapshot"
        } else if prefix_cache {
            "ON (default)"
        } else if hybrid && !args.no_prefix_cache {
            "OFF — REFUSED on a hybrid recurrent model: a hit restores KV but not the \
             recurrent state, which returns wrong text (measured 2026-09-19)"
        } else {
            "OFF (--no-prefix-cache)"
        }
    );
    // PREFIX ANCHORS (2026-09-26): with snapshots on, a chat request's shared system + tools
    // prefix gets a snapshot of its own, so a NEW session of the same agent resumes there
    // (`crate::prefix_anchor`). Measured before it: Claude Code's first turn 88-90 s on every
    // new session. The effect is NOT yet measured; the log lines
    // `[state-snapshot] anchor snapshot at N tokens` / `... resumes from the anchor snapshot`
    // are the proof that it ran.
    let anchor_snapshots = prefix_cache && state_snapshots && crate::prefix_anchor::enabled();
    if prefix_cache && state_snapshots {
        eprintln!(
            "prefix anchors: {}",
            if anchor_snapshots {
                "ON — the end of each shared system + tools prefix is snapshotted for new \
                 sessions (ARF_NO_ANCHOR_SNAPSHOT=1 to turn off)"
            } else {
                "OFF (ARF_NO_ANCHOR_SNAPSHOT)"
            }
        );
        // TOOLS ANCHORS (2026-09-27): a second anchor at the end of the tools block, for a
        // template that renders the tools before the system text (Qwen3.8), so a new session
        // whose system text differs (Claude Code in another working directory) resumes past the
        // tool schemas. NOT measured when written; `[state-snapshot] tools anchor snapshot at N
        // tokens` / `... resumes from the tools anchor snapshot ... a NEW session` are the proof it
        // ran. Measured since (measured 2026-09-27): a new
        // Claude Code session in another directory, first turn 98.4 -> 27.5 s.
        eprintln!(
            "tools anchors: {}",
            if anchor_snapshots && crate::prefix_anchor::tools_enabled() {
                "ON — the end of a tools block rendered before the system text is snapshotted \
                 too, for new sessions with another system text (ARF_NO_TOOLS_ANCHOR=1 to turn off)"
            } else if anchor_snapshots {
                "OFF (ARF_NO_TOOLS_ANCHOR)"
            } else {
                "OFF (prefix anchors are off)"
            }
        );
        // JUNCTION SNAPSHOTS (2026-09-26): the scheduler also snapshots where a prompt leaves
        // cached history, so the next session sharing that longer prefix resumes there. NOT
        // measured; `[state-snapshot] junction snapshot at N tokens` / `... resumes from the
        // junction snapshot` are the proof that it ran. On Claude Code it does not fire, by
        // design: the history it shares past the anchor is < 512 tokens (measured
        // 2026-09-26, "Claude Code's billing-header system block").
        eprintln!(
            "junction snapshots: {}",
            if arf_core::scheduler::junction_snapshots_enabled() {
                "ON — where a prompt leaves cached history is snapshotted for the next session \
                 sharing that prefix (ARF_NO_JUNCTION_SNAPSHOT=1 to turn off)"
            } else {
                "OFF (ARF_NO_JUNCTION_SNAPSHOT)"
            }
        );
        // SESSION-START SNAPSHOTS (2026-09-26): a conversation's first end-of-prompt snapshot is
        // filed in the junction pool, where its own later turns cannot evict it. NOT measured
        // when written; `[state-snapshot] session-start snapshot at N tokens` / `... resumes from
        // the session-start snapshot ... a NEW session` are the proof that it ran. Measured since
        // (measured 2026-09-26): first turn 14.4 -> 3.9 s.
        eprintln!(
            "session-start snapshots: {}",
            if arf_core::scheduler::session_start_snapshots_enabled() {
                "ON — a conversation's first prompt end is kept in the junction pool for the \
                 next session that opens the same way (ARF_NO_SESSION_START_SNAPSHOT=1 to turn off)"
            } else {
                "OFF (ARF_NO_SESSION_START_SNAPSHOT)"
            }
        );
    }
    // 🔴 A HYBRID MODEL'S BATCH IS CAPPED BY ITS RECURRENT BANK (2026-09-19).
    //
    // Each decoding sequence of a hybrid model owns one row of the recurrent state bank; the bank
    // has ARF_GDN_ROWS rows (default 8 WHEN THIS WAS MEASURED, 17 since — 16 sequences) and row 0
    // is reserved, so at 8 it held 7. `--max-batch-size`
    // defaults to 32 and nothing connected the two. MEASURED on Qwen3.8-27B, shipped path, no
    // experimental switch: 7 concurrent requests 7/7 sane, **8 concurrent 0/8 sane** — fluent
    // garbage in unrelated scripts — and with ARF_GDN_ROWS=16 the same 8 are 8/8 sane; with
    // ARF_GDN_ROWS=4 it breaks at 4. The bank refuses the 8th sequence CORRECTLY; the refusal
    // returned `None`, the caller fell back to the serial path, and that path's recurrent state is
    // not the bank's. tok/s looked fine throughout.
    //
    // The bank is ~157 MB a row on the 27B, so sizing it for 32 is 5 GB nobody asked for. Clamp the
    // scheduler instead: requests past the cap QUEUE, which is what a full batch always meant.
    // Raise ARF_GDN_ROWS to serve more at once, knowingly.
    #[allow(unused_mut)]
    let mut max_batch_size = args.max_batch_size;
    // ...and the other direction (2026-09-24): a server asked for FEWER streams than the default
    // bank holds gets a bank that size. 17 rows is 2.45 GB on the 27B (vmmap: 48 x 51 MB); a
    // `--max-batch-size 1` long-context server uses two of them and the rest is working set the
    // 8-bit KV pool could have had (~65K tokens of it). Must run before the first
    // `gdn_bank_rows()` read, which freezes the count; the bank itself is allocated at the first
    // step. An explicit ARF_GDN_ROWS still wins.
    #[cfg(feature = "wgpu")]
    if hybrid && std::env::var_os("ARF_GDN_ROWS").is_none() && args.max_batch_size < 16 {
        let rows = args.max_batch_size.max(1) + 1;
        std::env::set_var("ARF_GDN_ROWS", rows.to_string());
        eprintln!(
            "[serve] recurrent bank sized to --max-batch-size {}: {rows} rows (~{:.2} GB instead of \
             ~2.45 GB for the default 17)",
            args.max_batch_size,
            rows as f64 * 2.45 / 17.0 // measured: 48 x 51 MB for 17 rows
        );
    }
    #[cfg(feature = "wgpu")]
    if hybrid {
        let cap = arf_gpu::recurrent_stream_capacity();
        if max_batch_size > cap {
            eprintln!(
                "--max-batch-size {max_batch_size} lowered to {cap}: a hybrid recurrent model decodes \
                 one sequence per recurrent-bank row and the bank holds {cap} (ARF_GDN_ROWS - 1). \
                 More than that returned wrong text, not an error (measured 2026-09-19). Further \
                 requests queue. Raise ARF_GDN_ROWS to serve more at once (~157 MB a row on the 27B)."
            );
            max_batch_size = cap;
        }
    }
    // 🔴 f16 KV IS REFUSED ON A HYBRID MODEL UNLESS THE SCHEDULER RUNS ONE SEQUENCE AT A TIME
    // (2026-09-19).
    //
    // MEASURED on Qwen3.8-27B with `scripts/conc_prompt_fidelity.py` — 12 concurrent requests per
    // burst, six bursts per daemon, scoring whether each answer addresses ITS OWN prompt:
    //     pure defaults (f16 KV on, batch 7)   [12, 8, 8, 8, 8, 8]   1 clean burst of 6
    //     --max-batch-size 4                   [12, 8, 8, 8, 8, 8]
    //     --max-batch-size 1, and 2            [12, 12, 12]          clean
    //     defaults + ARF_NO_KV_F16=1        [12, 12, 12, 12, 12, 12]
    //     defaults + ARF_NO_WIN_CONFIG=1    [12, 12, 12, 12, 12, 12]
    // The wrong answers are FLUENT: "What is 17 times 23?" is answered as "the product of 17 and
    // 13". A sequence reads a prompt that is partly not its own. Identity gates never saw it (each
    // sends a prompt once, one at a time) and tok/s does not move.
    //
    // CLEARED BY MEASUREMENT, so nobody re-chases them: the recurrent bank (no row moved, stolen or
    // shared across 1,148 traced steps), token accounting (every sequence takes exactly
    // prompt+generated-1 recurrent steps), bank-row reuse (10/10 sequential on a 3-row bank), prompt
    // chunking (8/8 at --max-prefill-tokens 8), batch width (capped at 8 it still fails), and the
    // async ring (ARF_BATCH_MEGA_CHUNKWAIT changes nothing). NOT established: the mechanism.
    // "First burst clean, later bursts wrong" suggests reused KV blocks serving stale f16 data;
    // untested. The dense gemma-4-12B shows no f16-dependent difference.
    //
    // ALSO WRONG, and left alone for non-hybrid models where no fault is shown: `fast_path` turns
    // f16 KV on when --max-batch-size <= 8, read from argv, and ASSUMES 1 WHEN THE FLAG IS ABSENT —
    // the real default is 32. So a default daemon runs f16 KV at exactly the batch sizes its own
    // comment says it avoids.
    #[cfg(feature = "wgpu")]
    if hybrid && max_batch_size > 1 && std::env::var_os("ARF_KV_F16").is_some() {
        std::env::remove_var("ARF_KV_F16");
        std::env::set_var("ARF_NO_KV_F16", "1");
        eprintln!(
            "f16 KV turned OFF: on a hybrid recurrent model with --max-batch-size {max_batch_size} it \
             returned fluent WRONG answers under concurrency (a sequence reading part of another's \
             prompt — measured 2026-09-19). It stays on only at --max-batch-size 1. Ignore the \
             'f16 KV ON' in the win-config line above."
        );
    }
    let engine_cfg = arf_core::EngineConfig {
        block_size: KV_BLOCK_SIZE,
        num_blocks,
        max_batch_size,
        max_prefill_tokens: args.max_prefill_tokens,
        kv_quant,
        enable_prefix_cache: prefix_cache,
        // A verify window is up to 8 rows (prompt lookup, the block draft): keep them backed.
        decode_lookahead_tokens: 8,
        state_snapshots: prefix_cache && state_snapshots,
    };

    let qwen_vision = load_qwen_vision(&args, &cfg)?;
    let qwen_audio = load_audio_tower(&args, &cfg, &tokenizer)?;
    let (backend, vision) = build_backend(BackendRequest {
        model: &args.model,
        mmproj: args.mmproj.as_deref(),
        draft: args.draft.as_deref(),
        max_context: args.max_context,
        cfg: &cfg,
        engine_cfg: engine_cfg.clone(),
        quant,
        kv_quant,
    })?;
    let vision_enabled = vision.is_some();
    let vision_tokens = vision.as_ref().map(|v| v.num_tokens()).unwrap_or(0);

    let model_id = args.served_model_name.clone().unwrap_or_else(|| {
        args.model
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("arf")
            .to_string()
    });

    // --- Build the token→piece map for grammar-constrained sampling. -----------
    // This is used by JsonConstraint (response_format: json_object / json_schema)
    // to decode token ids to their string pieces during masking.  Built once here
    // and shared via Arc — cheap: one Vec<String> of ~32K–152K entries.
    let piece_map = {
        use arf_core::sampling::TokenPieceMap;
        let vocab = cfg.vocab_size;
        let mut pieces: Vec<String> = Vec::with_capacity(vocab);
        for id in 0..vocab as u32 {
            let piece = tokenizer.decode(&[id], false).unwrap_or_default();
            pieces.push(piece);
        }
        Arc::new(TokenPieceMap::new(pieces))
    };

    // --- Hand the model to its dedicated owner thread (the actor). -------------
    // `backend` is MOVED here and never leaves that thread — this is what keeps the
    // `!Sync` model sound. The handle we get back is just a channel sender.
    // The `metrics` Arc is shared with the actor (sole writer) and the HTTP
    // layer (reader via /metrics). No lock; relaxed atomics only.
    // ON-DISK PREFIX CACHE (`prefix_disk`): set up BEFORE the actor starts, so the saved states
    // load at its first idle moment — right after the warm-up, before any request. Set up after
    // the warm-up instead, the actor was already blocked waiting for a job and loaded them only
    // once the first request had finished (measured 2026-10-06: that request ran cold).
    if anchor_snapshots && crate::anchor_replay::enabled() && crate::prefix_disk::enabled() {
        let n = crate::prefix_disk::init(std::path::Path::new(&args.model));
        if n > 0 {
            eprintln!("[prefix-disk] {n} saved prefix state(s) load after the warm-up");
        }
    }
    let greedy_only = backend.greedy_only();
    if let Some(why) = greedy_only {
        eprintln!("[serve] greedy only: {why}; sampled requests are served greedily");
    }
    let (handle, metrics, _actor_thread) = actor::spawn(backend, engine_cfg, piece_map, vision);
    watch_power(metrics.clone());

    // --- WARM-UP REQUEST -------------------------------------------
    // Fire ONE synthetic decode through the real actor path before the listener
    // binds, so any first-request-only cost lands HERE (next to the pipeline
    // warm-up that already logs ~14s) instead of on a user's first API call.
    //
    // WHY: MEASURED 2026-08-03 — a warmed daemon served 16 tokens in 48-55s, and
    // the per-step profile (ARF_ACTOR_PROF) showed ONE step eating 54.4s while
    // every later token ran at 0-150ms. That is a first-request-only cost on the
    // real decode path; load-time `warm_pipelines` (which dispatches all 54 island
    // kernels with dummy bindings) does NOT cover it. A synthetic request does,
    // because it exercises exactly what real traffic exercises.
    //
    // Skipped by ARF_NO_WARMUP_REQUEST=1. Failures are non-fatal: this is an
    // optimisation, never a reason to refuse to serve.
    if std::env::var_os("ARF_NO_WARMUP_REQUEST").is_none() {
        let t_warm = std::time::Instant::now();
        let (wtx, mut wrx) = tokio::sync::mpsc::channel(64);
        let warm_job = actor::Job {
            id: u64::MAX, // reserved id; never collides with HTTP-assigned ids
            // MULTI-TOKEN prompt: this is the whole point. MEASURED 2026-08-03 on one server,
            // back to back: a 1-WORD prompt returned in 0.20s while an 8-WORD prompt took 46.16s.
            // A 1-token prompt goes straight to decode (q_len==1); a multi-token prompt takes the
            // PREFILL path, and THAT is where the ~46s first-time cost lives (the per-step profile
            // shows every decode step at 0-153ms, so the cost is entirely pre-decode). A 1-token
            // warm-up therefore misses it completely — the first version of this warm-up did
            // exactly that and the stall survived. 32 tokens also crosses any chunked-prefill
            // threshold on the way.
            // L60 (2026-08-06) — 32 tokens is NOT ENOUGH. MEASURED cold-TTFT with unique
            // prompts, 3 samples: 399ms / 91ms / 89ms. The FIRST real request paid +308ms over
            // steady state (~9x llama's +38ms first-request overhead) because a 32-token warm-up
            // never exercises the ~250-token prefill SHAPE, so the first real prompt still pays
            // the lazy buffer growth in `ensure_mega_bufs_batched` (scratch is sized on demand
            // from the observed b/ctx high-water). Sending a 256-token warm-up instead measured
            // 100ms / 89ms / 91ms — the penalty is GONE and steady state is unchanged.
            // Cost: a few hundred ms of extra one-time warm-up, paid before the daemon reports
            // ready, in exchange for the first user-visible request being 4x faster.
            prompt_ids: (0..256u32).map(|i| (i % 97) + 3).collect(),
            params: arf_core::sampling::SamplingParams::greedy(8),
            token_tx: wtx,
            want_logprobs: false,
            top_logprobs: 0,
            response_format: None,
            images: Vec::new(),
            image_positions: Vec::new(),
            image_prompt: None,
            prefix_anchor: None,
            tools_anchor: None,
            header_tail: None,
            background: false,
            stop: None,
        };
        if handle.submit(warm_job).is_ok() {
            // Drain to completion so the cost is paid before we accept traffic.
            while wrx.blocking_recv().is_some() {}
            // L61 — SECOND WARM-UP AT THE *SHORT* SHAPE. The 256-token pass above sets the
            // buffer high-water (L60: first-request 399ms -> 162ms), but a real chat prompt
            // arrives as a SMALL prefill chunk (measured q_len=16-17), and that shape still paid
            // a one-time cost: first prefill step 133.7ms vs 78-84ms steady. Sending a ~12-word
            // prompt by hand after the long warm measured 106ms vs 152ms first-TTFT, so the two
            // shapes exercise different work. Warm BOTH: long for the high-water, short for the
            // chunk shape real traffic actually uses.
            let (wtx2, mut wrx2) = tokio::sync::mpsc::channel(64);
            let warm_job_short = actor::Job {
                id: u64::MAX - 1, // reserved; never collides with HTTP-assigned ids
                prompt_ids: (0..17u32).map(|i| (i % 89) + 5).collect(),
                params: arf_core::sampling::SamplingParams::greedy(4),
                token_tx: wtx2,
                want_logprobs: false,
                top_logprobs: 0,
                response_format: None,
                images: Vec::new(),
                image_positions: Vec::new(),
                image_prompt: None,
                prefix_anchor: None,
                tools_anchor: None,
                header_tail: None,
                background: false,
                stop: None,
            };
            if handle.submit(warm_job_short).is_ok() {
                while wrx2.blocking_recv().is_some() {}
            }
            eprintln!(
                "[serve] warm-up requests done in {:.1}s (256-tok + 17-tok shapes) — first \
real request pays no JIT (ARF_NO_WARMUP_REQUEST=1 to skip)",
                t_warm.elapsed().as_secs_f64()
            );
        }
    }

    // --- ANCHOR REPLAY (issue #17, `crate::anchor_replay`) -----------------------
    // The prefix anchors recorded before the last restart are re-taken on a thread of their own,
    // shortest first, while the server already serves: each is an ordinary one-token request
    // carrying the anchor hint it was taken with, so the scheduler snapshots the same key.
    if anchor_snapshots && crate::anchor_replay::enabled() {
        let model = crate::anchor_replay::model_identity(std::path::Path::new(&args.model));
        if let Some(path) = crate::anchor_replay::default_path(&model) {
            let entries =
                crate::anchor_replay::replay_order(crate::anchor_replay::init(path.clone(), model));
            // Anchors whose STATE is on disk load in about a second at the first idle moment
            // (`prefix_disk`); only the others are prefilled again.
            // With the prompt STATES on disk (`prefix_disk`, loaded in seconds at the first idle
            // moment), the token re-read is not run at all: measured 2026-10-06 on a teammate's
            // M5 Max, it re-read ~100K tokens (~8 min of GPU) of anchors next to the two states
            // just loaded, and the first message after the restart waited behind it. Recording
            // goes on (the token file stays the fallback for ARF_PREFIX_DISK=0).
            let entries: Vec<_> = if crate::prefix_disk::enabled() {
                if !entries.is_empty() {
                    eprintln!(
                        "[anchor-replay] not re-reading {} recorded prompt(s): saved prompt states \
                         load from disk instead (ARF_PREFIX_DISK=0 re-reads them)",
                        entries.len()
                    );
                }
                Vec::new()
            } else {
                entries
            };
            if entries.is_empty() {
                eprintln!(
                    "[anchor-replay] recording prefix anchors to {} (ARF_ANCHOR_REPLAY is on)",
                    path.display()
                );
            } else {
                let tokens: usize = entries.iter().map(|e| e.at).sum();
                eprintln!(
                    "[anchor-replay] re-taking {} prefix anchor(s) from {} ({tokens} prefix \
                     tokens) in the background; a new session that starts first prefills as \
                     usual (ARF_ANCHOR_REPLAY is on)",
                    entries.len(),
                    path.display()
                );
                let h = handle.clone();
                std::thread::spawn(move || {
                    let t = std::time::Instant::now();
                    let n = entries.len();
                    for (i, e) in entries.into_iter().enumerate() {
                        let (prompt_ids, prefix_anchor, tools_anchor) = e.replay();
                        let (tx, mut rx) = tokio::sync::mpsc::channel(64);
                        let job = actor::Job {
                            // Reserved, below the two warm-up ids; never an HTTP-assigned id.
                            id: u64::MAX - 16 - i as u64,
                            prompt_ids,
                            params: arf_core::sampling::SamplingParams::greedy(1),
                            token_tx: tx,
                            want_logprobs: false,
                            top_logprobs: 0,
                            response_format: None,
                            images: Vec::new(),
                            image_positions: Vec::new(),
                            image_prompt: None,
                            prefix_anchor,
                            tools_anchor,
                            header_tail: None,
                            background: false,
                            stop: None,
                        };
                        if h.submit(job).is_err() {
                            return;
                        }
                        while rx.blocking_recv().is_some() {}
                    }
                    eprintln!(
                        "[anchor-replay] {n} prefix anchor(s) re-taken in {:.1}s",
                        t.elapsed().as_secs_f64()
                    );
                });
            }
        }
    }

    // --- Resident dashboard (TTY only). ----------------------------------------
    // Clone the metrics Arc BEFORE `metrics` is moved into AppState. The dashboard
    // thread is dedicated and only ever READS these lock-free atomics — it never
    // touches the actor or tokio, so it cannot perturb the hot decode path. When
    // stdout is not a TTY (systemd/Docker) or `--no-tty` is passed, we DO NOT spawn
    // it and keep the existing plain `eprintln!` logging unchanged. We decide
    // whether to render it here, but only ENTER the alternate screen after the
    // listener binds (below) — so a bind failure (e.g. port in use) surfaces on
    // the normal screen instead of leaving the terminal in the alternate screen.
    let want_dashboard = {
        use std::io::IsTerminal;
        std::io::stdout().is_terminal() && !args.no_tty
    };
    let metrics_dash = Arc::clone(&metrics);

    // --- Start the async HTTP runtime. -----------------------------------------
    // Built BEFORE state so the optional Postgres pool can be initialised inside an
    // async context (sqlx connect/migrate are async). The `block_on` below owns the
    // server for the rest of the process.
    let addr = format!("{}:{}", args.host, args.port);
    let dash_endpoint = addr.clone();
    let dash_model_id = model_id.clone();
    let kv_total = num_blocks;
    let batch_max = args.max_batch_size;
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    // --- Optional Postgres chat persistence (feature = "postgres"). ------------
    // Zero-DB by default: with the feature off there is no field; with it on, the
    // pool is `None` unless DATABASE_URL is set AND reachable. A connect failure
    // logs a warning and disables persistence — it never blocks startup. Run on the
    // tokio runtime since sqlx is async.
    #[cfg(feature = "postgres")]
    let db = rt.block_on(async {
        let pool = crate::persist::init_pool().await;
        if let Some(pool) = pool.as_ref() {
            if let Err(e) = crate::persist::run_migrations(pool).await {
                eprintln!("persistence: migrations failed ({e}) — chat persistence DISABLED");
            }
        }
        pool
    });

    let state = http::AppState {
        greedy_only,
        model: handle,
        tokenizer,
        model_id: model_id.clone(),
        stop_tokens,
        max_tokens_cap: args.max_context,
        chat_template,
        next_id: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        metrics,
        vision_enabled,
        vision_tokens,
        qwen_vision,
        qwen_audio,
        prefix_anchors: Arc::new(crate::prefix_anchor::AnchorCache::new(anchor_snapshots)),
        systemone: Arc::new(systemone),
        #[cfg(feature = "postgres")]
        db,
    };

    rt.block_on(async move {
        // Bind FIRST: a bind error returns here before the dashboard takes over
        // the screen.
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        eprintln!("model resident ({model_id}), serving on :{}", args.port);
        #[cfg(feature = "wgpu")]
        if let Some(b) = arf_gpu::phys_footprint_bytes() {
            eprintln!(
                "[mem] this process: {:.1} GB of its own memory (`arf status` shows it live); the \
                 weights are beside it, paged from the model's cache file",
                b as f64 / 1e9
            );
        }
        // Now that we hold the port, start the live dashboard (TTY only). The
        // thread loops until the process exits; the OS reclaims it.
        if want_dashboard {
            let _dash_thread = std::thread::Builder::new()
                .name("arf-dashboard".into())
                .spawn(move || {
                    run_dashboard(metrics_dash, dash_model_id, dash_endpoint, kv_total, batch_max);
                });
        }
        // L153 — GRACEFUL SHUTDOWN ON SIGTERM/SIGINT. Without this, `pkill` (or any supervisor
        // stop) killed the process WITHOUT unwinding, so no `Drop` ran — including
        // `ResidencySet::drop`, which is what calls `endResidency` to release ~20 GB of WIRED
        // memory. Every stopped daemon leaked its wired weight set, and a per-layer bisect that
        // launches the daemon repeatedly accumulated them until the box swapped, the AGX wedged,
        // and the machine froze (twice in one session).
        //
        // llama.cpp releases residency on buffer free (ggml-metal-device.m:1600 `endResidency`,
        // :1797 `rsets_rm`), which is why it can load the same 20 GB model repeatedly without
        // taking the machine down. This is the missing half of that contract on our side.
        // CORRECTED 2026-10-08: the graceful shutdown alone ran no such `Drop` — the residency
        // set belongs to the backend, which lives on the model thread, and the process exited
        // around it. The release happens after `serve` returns (`actor::SHUTDOWN`, below).
        // BOUNDED (2026-10-08): the graceful shutdown waits for every open request to finish, and
        // a client that keeps one open kept a stopped server running with it — an agent the
        // release benchmark had left behind held a 15,590-token read, and the server did not exit
        // until that process was killed. Open requests now get `STOP_GRACE` after the signal.
        const STOP_GRACE: std::time::Duration = std::time::Duration::from_secs(10);
        let (stopped_tx, mut stopped) = tokio::sync::watch::channel(false);
        let shutdown = async move {
            let mut term = tokio::signal::unix::signal(
                tokio::signal::unix::SignalKind::terminate(),
            )
            .expect("install SIGTERM handler");
            tokio::select! {
                _ = tokio::signal::ctrl_c() => eprintln!("[serve] SIGINT — releasing GPU residency and exiting"),
                _ = term.recv()             => eprintln!("[serve] SIGTERM — releasing GPU residency and exiting"),
            }
            let _ = stopped_tx.send(true);
        };
        let serve = std::future::IntoFuture::into_future(
            axum::serve(listener, http::router(state)).with_graceful_shutdown(shutdown),
        );
        let grace = async move {
            let _ = stopped.wait_for(|s| *s).await;
            tokio::time::sleep(STOP_GRACE).await;
        };
        tokio::select! {
            r = serve => r?,
            _ = grace => eprintln!(
                "[serve] requests still open {} s after the stop: not waiting for them",
                STOP_GRACE.as_secs()
            ),
        }
        Ok::<(), Box<dyn Error>>(())
    })?;
    // CLEAN SHUTDOWN (`actor::SHUTDOWN`): let the model thread finish its step, wait for the GPU
    // and release the model's memory itself before this process exits. Bounded: a supervisor's
    // SIGKILL follows a stop by tens of seconds, and a hung GPU must not keep the process alive.
    actor::SHUTDOWN.store(true, std::sync::atomic::Ordering::Relaxed);
    let t0 = std::time::Instant::now();
    while !_actor_thread.is_finished() && t0.elapsed() < std::time::Duration::from_secs(20) {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    // A STOP IS NOT A CRASH (2026-10-08): the background server's watcher (`arf watch`) restarts
    // an `arf-serve` that ends while its state file still names it. A clean stop removes that
    // file, so only an ending without one — a panic, a SIGKILL, the memory killer — is restarted.
    let state = std::env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .map(|h| h.join(".arf/run").join(format!("serve-{}.json", args.port)));
    if let Some(state) = state {
        let ours = std::fs::read_to_string(&state)
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|v| v["pid"].as_u64())
            == Some(std::process::id() as u64);
        if ours {
            let _ = std::fs::remove_file(&state);
        }
    }
    if _actor_thread.is_finished() {
        eprintln!(
            "[serve] stopped cleanly in {:.1} s",
            t0.elapsed().as_secs_f64()
        );
    } else {
        eprintln!("[serve] the model thread did not stop within 20 s; exiting anyway");
    }

    Ok(())
}

/// The resident dashboard loop. Runs on a dedicated OS thread; reads only the
/// lock-free `Metrics` atomics every ~250 ms, pushes them into history rings, and
/// renders sparklines + gauges with ratatui. Never touches the actor or tokio.
///
/// On entry it enters raw mode + the alternate screen via [`TermGuard`] (restored
/// on drop). If the terminal can't be set up, it falls back to doing nothing so
/// the server still runs with plain logging. The loop has no clean exit signal
/// from here (the server owns the main thread in `block_on`), so it loops until
/// the process exits and the OS reclaims it. NOTE: a SIGINT/Ctrl-C tears the
/// process down without unwinding, so the `TermGuard` Drop will NOT run on Ctrl-C
/// — the terminal may be left in the alternate screen. Acceptable for v1.
fn run_dashboard(
    metrics: Arc<crate::metrics::Metrics>,
    model_id: String,
    endpoint: String,
    kv_total: usize,
    batch_max: usize,
) {
    use crate::dashboard::{self, DashSnapshot, Ring, TermGuard};
    use crossterm::terminal::disable_raw_mode;
    use ratatui::{backend::CrosstermBackend, Terminal};
    use std::io::stdout;
    use std::sync::atomic::Ordering::Relaxed;
    use std::time::{Duration, Instant};

    // Log lines go to the log file while the panels are up (#57); the terminal gets them again
    // when the dashboard closes. Declared before `guard`, so it is dropped after it.
    #[cfg(unix)]
    let _log_lines = {
        let path = std::env::var_os("HOME").map(|h| {
            std::path::Path::new(&h).join(".arf/logs").join(format!(
                "serve-{}.log",
                endpoint.rsplit(':').next().unwrap_or("0")
            ))
        });
        if let Some(p) = path.as_ref() {
            eprintln!(
                "[serve] log lines go to {} while the dashboard is on",
                p.display()
            );
        }
        path.and_then(|p| dashboard::StderrToFile::open(&p))
    };
    let guard = TermGuard::enter();
    if !guard.active() {
        // Could not enter the alternate screen; restore whatever changed and bail.
        let _ = disable_raw_mode();
        return;
    }
    let mut term = match Terminal::new(CrosstermBackend::new(stdout())) {
        Ok(t) => t,
        Err(_) => return, // `guard` drops here and restores the terminal.
    };

    const CAP: usize = 60;
    let mut tok_ring = Ring::new(CAP);
    let mut req_ring = Ring::new(CAP);
    let mut tok_peak: u64 = 0;
    let mut prev_tokens = metrics.tokens_generated.load(Relaxed);
    let started = Instant::now();
    let mut prev_tick = Instant::now();

    loop {
        // Poll for input instead of a blind sleep: in raw mode Ctrl-C arrives as a
        // KeyEvent (NOT SIGINT), so this is how we catch it — plus `q`/Esc — and
        // break CLEANLY so `TermGuard`'s Drop restores the terminal. (This is the
        // fix for the "serve panel won't close" bug: there was no key handler and
        // Drop doesn't run on a true SIGINT.) 250ms poll = the old refresh cadence.
        if let Ok(true) = crossterm::event::poll(Duration::from_millis(250)) {
            if let Ok(crossterm::event::Event::Key(k)) = crossterm::event::read() {
                use crossterm::event::{KeyCode, KeyModifiers};
                let quit = matches!(k.code, KeyCode::Char('q') | KeyCode::Esc)
                    || (k.code == KeyCode::Char('c')
                        && k.modifiers.contains(KeyModifiers::CONTROL));
                if quit {
                    break; // guard drops → terminal restored
                }
            }
        }

        let now = Instant::now();
        let dt = now.duration_since(prev_tick).as_secs_f64();
        prev_tick = now;

        // Read the lock-free atomics (no actor / tokio access).
        let now_tokens = metrics.tokens_generated.load(Relaxed);
        let running = metrics.running.load(Relaxed);
        let kv_used = metrics.kv_blocks_used.load(Relaxed);
        let served = metrics.requests_finished.load(Relaxed);
        let reused = metrics.prefix_cache_tokens_reused.load(Relaxed);

        let tok_now = dashboard::rate(prev_tokens, now_tokens, dt);
        prev_tokens = now_tokens;
        tok_peak = tok_peak.max(tok_now);
        tok_ring.push(tok_now);
        // `Metrics` tracks no distinct "active request" count, so the requests
        // sparkline (and `req_active`/`batch_running` below) all reflect the same
        // `running` gauge — sequences currently in the batch.
        req_ring.push(running as u64);

        // Approximation: we don't track total prompt tokens, so the prefix-cache
        // hit % is reused / (reused + tokens_generated) — a rough indicator of how
        // much prompt KV was reused vs. computed, not a true hit ratio.
        let prefix_hit = dashboard::prefix_hit_pct(reused, reused + now_tokens);

        let uptime = fmt_uptime(started.elapsed());

        let snap = DashSnapshot {
            model_id: model_id.clone(),
            endpoint: endpoint.clone(),
            uptime,
            tok_series: tok_ring.series(),
            tok_now,
            tok_peak,
            req_series: req_ring.series(),
            req_active: running,
            req_served: served,
            kv_used,
            kv_total,
            batch_running: running,
            batch_max,
            prefix_hit,
            kernels: Vec::new(),
        };

        if term.draw(|f| dashboard::draw(f, &snap)).is_err() {
            break; // `guard` drops and restores the terminal.
        }
    }
}

/// Format a duration as a compact uptime string, e.g. `4m 12s` or `42s`.
fn fmt_uptime(d: std::time::Duration) -> String {
    let secs = d.as_secs();
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    if h > 0 {
        format!("{h}h {m}m")
    } else if m > 0 {
        format!("{m}m {s}s")
    } else {
        format!("{s}s")
    }
}

/// Re-exec once with `MallocLargeCache=0`, so the loader's freed scratch goes back to the OS.
///
/// MEASURED 2026-09-19 (measured), Qwen3.8-27B on a 36 GB M4 Max. After load,
/// `footprint` read **9,628 MB** dirty, of which 4,917 MB was `MALLOC_LARGE`: forty 128 MB
/// regions that `vmmap` labels "(empty)" — no live allocation — yet fully dirty, 2.36 GiB of it
/// already swapped out. Their total, 4.75 GiB, is the embedding table's two load-time
/// transients: the bf16 matrix (2.37 GiB) and its u32 repack (2.37 GiB). We free both; macOS's
/// large-allocation cache keeps the regions. Live heap was 0.06 GiB.
///
/// On top of ~19 GB of weights held resident for the GPU that dead 5 GB is what put this Mac
/// into swap: each daemon start made the OS evict it WHILE REQUESTS WERE BEING TIMED. In the
/// interleaved another engine comparison every Arf arm moved swap pages inside its timed section and
/// greedy fell 21.8 -> 18.0 tok/s between rounds, where another engine held steady.
///
/// With `MallocLargeCache=0`: **4,755 MB**, `MALLOC_LARGE` 58 MB, zero empty regions.
///
/// TRIED FIRST AND MEASURED-OUT: `malloc_zone_pressure_relief(NULL, 0)` after load — returned
/// 0.00 GB and the footprint did not move. DO NOT RE-CHASE it. The allocator reads this variable
/// at process start, so it cannot be set from inside a running process; hence the re-exec.
/// To opt out, set `MallocLargeCache` yourself (any value): the re-exec only happens when it is
/// unset. No `ARF_*` flag for this — the env surface guard is right that a lever which won
/// becomes the default and does not get a knob.
fn reexec_without_malloc_large_cache() {
    #[cfg(target_os = "macos")]
    {
        use std::os::unix::process::CommandExt;
        if std::env::var_os("MallocLargeCache").is_some() {
            return;
        }
        let Ok(exe) = std::env::current_exe() else {
            return;
        };
        // `exec` only returns on failure; carry on in this process if it does.
        let err = std::process::Command::new(exe)
            .args(std::env::args_os().skip(1))
            .env("MallocLargeCache", "0")
            .exec();
        eprintln!("[mem] could not re-exec with MallocLargeCache=0 ({err}); continuing without it");
    }
}

/// Qwen3.8 vision: when `--mmproj` names a `qwen3vl_merger` projector, load its encoder — on the
/// GPU when it comes up (`ARF_VISION_CPU=1` keeps the CPU one; see `qwen_image::QwenVision`).
/// `None` for no mmproj or a Gemma-3 one. Refuses a projector whose output width is not the text
/// model's hidden size, and any backend other than Metal-wgpu (the only one whose attention
/// honours `ForwardBatch::mrope_positions`).
fn load_qwen_vision(
    args: &Args,
    cfg: &arf_core::config::ModelConfig,
) -> Result<Option<Arc<crate::qwen_image::QwenVision>>, Box<dyn Error>> {
    let Some(path) = args.mmproj.as_ref() else {
        return Ok(None);
    };
    if !arf_core::model::qwen_vision::is_qwen3vl_mmproj(path) {
        return Ok(None);
    }
    if !cfg!(all(feature = "wgpu", target_os = "macos")) {
        return Err(
            "Qwen3.8 image input needs the Metal backend (M-RoPE is wired only there)".into(),
        );
    }
    // The M-RoPE image path (`forward_mrope_batch`) serves the qwen35 hybrid only, with its
    // sections hard-coded; any other model PANICKED on the first image batch. Refuse at start
    // instead. Qwen3-Omni's llama.cpp mmproj is a `qwen3vl_merger` too, but its images need
    // DeepStack and [24,20,20] sections with a T fallback (the port plan, M7).
    if !matches!(cfg.attn, arf_core::config::AttnKind::HybridSsmAttn { .. }) {
        return Err(format!(
            "--mmproj {}: a qwen3vl_merger projector is wired only for Qwen3.8 (the qwen35 hybrid); \
             this model's image input is not implemented (Qwen3-Omni images are M7)",
            path.display()
        )
        .into());
    }
    eprintln!("loading Qwen vision encoder: {} ...", path.display());
    let t0 = std::time::Instant::now();
    let enc = crate::qwen_image::QwenVision::load(path)?;
    if enc.cfg.proj_dim != cfg.hidden_size {
        return Err(format!(
            "mmproj projects to {} but the model's hidden size is {}",
            enc.cfg.proj_dim, cfg.hidden_size
        )
        .into());
    }
    eprintln!(
        "Qwen vision ready in {:.1}s ({}): {} ViT layers, {}..={} pixels per image",
        t0.elapsed().as_secs_f64(),
        enc.backend(),
        enc.cfg.layers,
        enc.cfg.min_pixels,
        enc.cfg.max_pixels
    );
    Ok(Some(Arc::new(enc)))
}

/// Qwen3-Omni audio (M3): when `--audio-tower` names the Thinker's `thinker.audio_tower.*`
/// safetensors, load its encoder (GPU on Metal, `ARF_AUDIO_CPU=1` for the CPU one). Refuses a
/// tower whose output width is not the text model's hidden size, and a tokenizer without the
/// `<|audio_start|>` / `<|audio_pad|>` / `<|audio_end|>` tokens the prompt layout needs.
fn load_audio_tower(
    args: &Args,
    cfg: &arf_core::config::ModelConfig,
    tokenizer: &Tokenizer,
) -> Result<Option<Arc<crate::qwen_audio::AudioTower>>, Box<dyn Error>> {
    use crate::qwen_audio::{AudioTower, AUDIO_END, AUDIO_PAD, AUDIO_START};
    let Some(path) = args.audio_tower.as_ref() else {
        return Ok(None);
    };
    // The splice is `ForwardBatch::image_embeds`, which the wgpu backend's generic loop honours;
    // another GPU backend may have no such path, and ignoring the rows would answer without the audio.
    if !cfg!(feature = "wgpu") {
        return Err("--audio-tower needs the wgpu/Metal backend".into());
    }
    for t in [AUDIO_START, AUDIO_PAD, AUDIO_END] {
        if tokenizer.token_to_id(t).is_none() {
            return Err(format!(
                "--audio-tower: the model's tokenizer has no {t} token (a Qwen3-Omni Thinker has)"
            )
            .into());
        }
    }
    eprintln!("loading Qwen3-Omni audio tower: {} ...", path.display());
    let t0 = std::time::Instant::now();
    let tower = AudioTower::load(path).map_err(|e| format!("--audio-tower: {e}"))?;
    if tower.output_dim() != cfg.hidden_size {
        return Err(format!(
            "--audio-tower projects to {} but the model's hidden size is {}",
            tower.output_dim(),
            cfg.hidden_size
        )
        .into());
    }
    eprintln!(
        "Qwen3-Omni audio ready in {:.1}s ({}): {} encoder layers -> {} wide rows, 13 tokens per \
         second of audio",
        t0.elapsed().as_secs_f64(),
        tower.backend(),
        tower.cfg.layers,
        tower.output_dim()
    );
    Ok(Some(Arc::new(tower)))
}

/// POWER STATE (2026-10-06): Low Power Mode lowers the GPU's clocks and this workload is GPU-bound —
/// a teammate's M5 Max in Low Power Mode on battery read a prompt with the GPU at 35-41 % and
/// decoded at 3-8 tok/s, which looked like a stall. Logged at start and on every change (checked
/// every 30 s), and kept in the metrics for `/v1/arf/status` and `arf status`.
fn watch_power(metrics: Arc<crate::metrics::Metrics>) {
    #[cfg(all(feature = "wgpu", target_os = "macos"))]
    std::thread::spawn(move || {
        use std::sync::atomic::Ordering::Relaxed;
        let mut last: Option<(bool, u8)> = None;
        loop {
            let now = arf_gpu::power_state();
            if last != Some(now) {
                let (low, thermal) = now;
                metrics.low_power.store(low, Relaxed);
                metrics.thermal.store(thermal, Relaxed);
                let slow = if low {
                    " — the GPU runs at lower clocks: prompts read and answers come slower \
                     (System Settings > Battery > Low Power Mode)"
                } else if thermal >= 2 {
                    " — the Mac is hot: the GPU may be throttled"
                } else {
                    ""
                };
                eprintln!(
                    "[power] Low Power Mode: {} · thermal: {}{slow}",
                    if low { "ON" } else { "off" },
                    crate::metrics::thermal_name(thermal)
                );
                last = Some(now);
            }
            std::thread::sleep(std::time::Duration::from_secs(30));
        }
    });
    #[cfg(not(all(feature = "wgpu", target_os = "macos")))]
    let _ = metrics;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn parse(extra: &[&str]) -> Result<Args, clap::Error> {
        let mut argv = vec!["arf-serve"];
        argv.extend_from_slice(extra);
        Args::try_parse_from(argv)
    }

    fn scratch(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("arf_serve_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A bundle directory with a draft and a projector (fake files: only names and the GGUF
    /// magic are looked at, nothing is loaded).
    fn fake_bundle(tag: &str) -> PathBuf {
        let d = scratch(tag);
        std::fs::create_dir_all(d.join("draft")).unwrap();
        std::fs::write(d.join("model.gguf"), b"GGUF\x03\0\0\0").unwrap();
        std::fs::write(d.join("mmproj.gguf"), b"GGUF\x03\0\0\0").unwrap();
        std::fs::write(d.join("draft/config.json"), b"{}").unwrap();
        std::fs::write(d.join("draft/model.safetensors"), b"x").unwrap();
        std::fs::write(d.join("arf-bundle.txt"), b"arch=qwen3.8-27b\nquant=q4ks\n").unwrap();
        d
    }

    #[test]
    fn opt_out_flags_conflict_with_the_explicit_ones() {
        assert!(parse(&["--model", "m", "--draft", "d", "--no-draft"]).is_err());
        assert!(parse(&["--model", "m", "--mmproj", "p", "--no-mmproj"]).is_err());
        let a = parse(&["--model", "m", "--no-draft", "--no-mmproj"]).unwrap();
        assert!(a.no_draft && a.no_mmproj);
        // --quant is optional now: unset stays None until a bundle or the "none" default fills it
        assert_eq!(parse(&["--model", "m"]).unwrap().quant, None);
    }

    #[test]
    fn apply_bundle_rewrites_args_from_the_bundle() {
        let d = fake_bundle("bundle_apply");
        let mut a = parse(&["--model", d.to_str().unwrap()]).unwrap();
        apply_bundle(&mut a).unwrap();
        assert_eq!(a.model, d.join("model.gguf"));
        assert_eq!(a.arch.as_deref(), Some("qwen3.8-27b"));
        assert_eq!(a.quant.as_deref(), Some("q4ks"));
        let on_metal = cfg!(all(feature = "wgpu", target_os = "macos"));
        assert_eq!(a.draft.is_some(), on_metal);
        assert_eq!(a.mmproj.is_some(), on_metal);
        assert_eq!(
            a.served_model_name.as_deref(),
            d.file_name().and_then(|n| n.to_str())
        );

        // explicit flags and opt-outs survive
        let mut a = parse(&[
            "--model",
            d.to_str().unwrap(),
            "--arch",
            "qwen3.6",
            "--quant",
            "q4k",
            "--no-draft",
            "--mmproj",
            "/x/mm.gguf",
            "--served-model-name",
            "mine",
        ])
        .unwrap();
        apply_bundle(&mut a).unwrap();
        assert_eq!(a.arch.as_deref(), Some("qwen3.6"));
        assert_eq!(a.quant.as_deref(), Some("q4k"));
        assert_eq!(a.draft, None);
        assert_eq!(a.mmproj.as_deref(), Some(Path::new("/x/mm.gguf")));
        assert_eq!(a.served_model_name.as_deref(), Some("mine"));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_plain_gguf_launch_is_unchanged() {
        // The benchmark arms: `--model <file>.gguf --arch .. --quant .. --draft <dir>`.
        let d = scratch("plain_gguf");
        let g = d.join("Qwen3.8-27B-arf-g64-imat-Q4_1.gguf");
        std::fs::write(&g, b"GGUF\x03\0\0\0").unwrap();
        let mut a = parse(&[
            "--model",
            g.to_str().unwrap(),
            "--arch",
            "qwen3.8-27b",
            "--quant",
            "q4ks",
            "--draft",
            "models/qwen3.8-27b-dflash2",
        ])
        .unwrap();
        apply_bundle(&mut a).unwrap();
        assert_eq!(a.model, g);
        assert_eq!(a.arch.as_deref(), Some("qwen3.8-27b"));
        assert_eq!(a.quant.as_deref(), Some("q4ks"));
        assert_eq!(
            a.draft.as_deref(),
            Some(Path::new("models/qwen3.8-27b-dflash2"))
        );
        assert_eq!(a.mmproj, None);
        assert_eq!(a.served_model_name, None);
        let _ = std::fs::remove_dir_all(&d);
    }
}
