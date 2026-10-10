//! `arf` command-line interface.

use std::error::Error;
use std::path::{Path, PathBuf};

use arf_core::config::{EngineConfig, ModelConfig};
use arf_core::engine::LlmEngine;
use arf_core::model::weights;
use arf_core::sampling::SamplingParams;
use arf_core::Tokenizer;
use clap::{Parser, Subcommand};

mod completions;
mod daemon;
mod doctor;
mod dump_logits;
mod hf_search;
mod inspect;
mod list;
mod perplexity;
mod pull;
mod registry;
mod rm;
mod run;
mod serve_shim;
mod ui;
mod warm;
mod watch;
mod welcome;

/// The full version line: pkg version + optional git hash + build date.
/// Computed from build.rs env vars (`option_env!` → known at compile time);
/// falls back to the bare Cargo version when git/date were unavailable at build.
fn version_string() -> &'static str {
    // Built once, leaked to a 'static str so clap can hold it. Called once at startup.
    use std::sync::OnceLock;
    static V: OnceLock<String> = OnceLock::new();
    V.get_or_init(|| {
        let base = env!("CARGO_PKG_VERSION");
        match (option_env!("ARF_GIT_HASH"), option_env!("ARF_BUILD_DATE")) {
            (Some(h), Some(d)) => format!("{base} ({h} {d})"),
            (Some(h), None) => format!("{base} ({h})"),
            (None, Some(d)) => format!("{base} ({d})"),
            (None, None) => base.to_string(),
        }
    })
    .as_str()
}

#[derive(Parser)]
#[command(
    name = "arf",
    about = "Fast local models on Apple silicon: chat, serve an API, run coding agents",
    version = version_string(),
    after_help = "Start here:\n  arf                         what this Mac can run, and what to type next\n  \
arf run qwen3.8:27b         chat (downloads the model the first time, after asking)\n  \
arf launch claude           Claude Code on a local model (also: opencode)\n  \
arf serve qwen3.8:27b       OpenAI + Anthropic API on localhost:8080"
)]
pub(crate) struct Cli {
    /// No command: show what this Mac can run and what to type next.
    #[command(subcommand)]
    command: Option<Command>,

    /// Disable ANSI colors/box-drawing (also honored via NO_COLOR / non-TTY).
    #[arg(long, global = true)]
    no_color: bool,
}

#[derive(Subcommand)]
enum Command {
    /// Chat in the terminal (starts the model in the background, or reuses it).
    Run(RunArgs),
    /// Start a coding agent (Claude Code, OpenCode) on a local model.
    Launch(LaunchArgs),
    /// Serve the OpenAI and Anthropic APIs in the foreground (execs arf-serve).
    Serve(ServeArgs),
    /// What the background model is doing: requests running and waiting, and how far a long
    /// prompt has been read (an agent's first message can take minutes).
    Status,
    /// Stop the background model that `arf run` / `arf launch` started.
    Stop,
    /// Download a model: a shorthand (`qwen3.8:27b`) or any HuggingFace repo.
    Pull(PullArgs),
    /// List models fetched into the models directory.
    #[command(visible_alias = "ls")]
    List(ListArgs),
    /// Remove a local model (directory + registry entry).
    Rm(RmArgs),
    /// Check this Mac: GPU, memory, models, arf-serve, HuggingFace.
    Doctor(DoctorArgs),
    /// Print a shell completion script (zsh|bash|fish|elvish|powershell).
    Completions(CompletionsArgs),
    /// One prompt, no chat: for scripts and benchmarks (loads the model in-process).
    #[command(hide = true)]
    Generate(GenerateArgs),
    /// (developer) Teacher-forced per-position logit dump (Correctness Ladder Layer 4).
    #[command(hide = true)]
    DumpLogits(dump_logits::DumpLogitsArgs),
    /// (developer) Perplexity over a fixed text corpus (Correctness Ladder Layer 5).
    #[command(hide = true)]
    Perplexity(perplexity::PerplexityArgs),
    /// (hidden) Print local model names, one per line — for dynamic completion.
    #[command(hide = true)]
    ModelNames(ModelNamesArgs),
    /// (internal) Watch the background server started on `--port` as `--pid`, and start it again
    /// if it ends without being stopped (see `daemon::watch`).
    #[command(hide = true)]
    Watch(WatchArgs),
    /// (internal) Have the server on `--port` read Claude Code's safety-check prompts in the
    /// background, ahead of the first session (see `warm`).
    #[command(hide = true)]
    WarmClaude(WarmArgs),
    /// (hidden) Search HuggingFace model ids for `<query>` — for dynamic completion.
    #[command(hide = true)]
    HfSearch(HfSearchArgs),
    /// (developer) List the tensors in a PyTorch checkpoint (.pt/.pth/.bin/.ckpt), read by a
    /// restricted unpickler that never executes the file.
    #[command(hide = true)]
    Inspect(InspectArgs),
}

/// The coding agents `arf launch` can start.
#[derive(clap::ValueEnum, Clone, Copy, Debug)]
enum Agent {
    /// Claude Code (`claude`), through the Anthropic Messages API.
    Claude,
    /// OpenCode (`opencode`), through the OpenAI chat API.
    Opencode,
}

#[derive(Parser)]
struct LaunchArgs {
    /// The agent to start.
    #[arg(value_enum)]
    agent: Agent,
    /// Model to run it on (default: a downloaded one, else the one suggested for this Mac).
    #[arg(long)]
    model: Option<String>,
    /// Arguments passed to the agent, after `--` (e.g. `arf launch claude -- --resume`).
    #[arg(last = true)]
    rest: Vec<String>,
}

#[derive(Parser)]
struct InspectArgs {
    /// Path to a `torch.save` file (zip or legacy format).
    file: PathBuf,
    /// Also print the non-tensor values (step, hyper-parameters, dtypes, ...).
    #[arg(long)]
    all: bool,
}

#[derive(Parser)]
struct CompletionsArgs {
    /// Target shell.
    #[arg(value_enum)]
    shell: clap_complete::Shell,
}

#[derive(Parser)]
struct ModelNamesArgs {
    /// Models directory (default: $ARF_MODELS_DIR, else ./models if it exists, else ~/.arf/models).
    #[arg(long, default_value_os_t = pull::default_models_dir())]
    dir: PathBuf,
}

#[derive(Parser)]
struct HfSearchArgs {
    /// Partial model name to search for.
    query: String,
}

#[derive(Parser)]
struct DoctorArgs {
    /// Models directory to check (default: $ARF_MODELS_DIR, else ./models if it exists, else ~/.arf/models).
    #[arg(long, default_value_os_t = pull::default_models_dir())]
    dir: PathBuf,
}

#[derive(Parser)]
struct ServeArgs {
    /// Model alias (`gemma3`), a local model dir, or an `org/repo`.
    model: String,
    /// Everything after the model is forwarded to arf-serve verbatim
    /// (e.g. `--port 9090 --quant q4k`).
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    rest: Vec<String>,
}

#[derive(Parser)]
struct WarmArgs {
    #[arg(long)]
    port: u16,
    #[arg(long)]
    model_id: String,
    #[arg(long)]
    claude: PathBuf,
}

#[derive(Parser)]
struct WatchArgs {
    #[arg(long)]
    port: u16,
    #[arg(long)]
    pid: u32,
}

#[derive(Parser)]
struct RunArgs {
    /// Model shorthand (`qwen3.8:27b`), a downloaded model's name, or a local path. Default: a
    /// downloaded model, else the one suggested for this Mac.
    model: Option<String>,
    /// Max new tokens per reply (default: the server's).
    #[arg(long)]
    max_tokens: Option<usize>,
    /// Ask the model to answer without reasoning first.
    #[arg(long)]
    no_think: bool,
    /// The old in-process chat (no server, bf16 weights, no chat template, no history): for
    /// debugging the engine without the server. The flags below apply to it only.
    #[arg(long)]
    in_process: bool,
    /// (--in-process) Path to tokenizer.json.
    #[arg(long, requires = "in_process")]
    tokenizer: Option<PathBuf>,
    /// The model's architecture, for a GGUF whose header does not match a built-in one
    /// (`arf serve --help` lists them). Passed to the background server; usually not needed.
    #[arg(long)]
    arch: Option<String>,
    /// (--in-process) Device to run on.
    #[arg(long, default_value = "gpu")]
    device: String,
    /// (--in-process) RoPE table size / max context.
    #[arg(long, default_value_t = 8192)]
    max_context: usize,
}

/// Sort order for `arf ls` (clap-facing; maps to `list::SortKey`).
#[derive(clap::ValueEnum, Clone, Copy)]
enum SortArg {
    Name,
    Size,
    State,
}

/// State filter for `arf ls` (clap-facing; maps to `list::State`).
#[derive(clap::ValueEnum, Clone, Copy)]
enum StateFilter {
    Ready,
    Downloading,
    Incomplete,
}

#[derive(Parser)]
struct ListArgs {
    /// Directory to scan for models (default: $ARF_MODELS_DIR, else ./models if it exists, else ~/.arf/models).
    #[arg(long, default_value_os_t = pull::default_models_dir())]
    dir: PathBuf,

    /// Emit a JSON array instead of the table (for scripts / pipes).
    #[arg(long)]
    json: bool,

    /// Sort order: name (A→Z), size (largest first), or state (ready first).
    #[arg(long, value_enum, default_value = "name")]
    sort: SortArg,

    /// Show only models in this state.
    #[arg(long, value_enum)]
    state: Option<StateFilter>,
}

#[derive(Parser)]
struct RmArgs {
    /// Local model name (directory under the models dir).
    model: String,
    /// Models directory (default: $ARF_MODELS_DIR, else ./models if it exists, else ~/.arf/models).
    #[arg(long, default_value_os_t = pull::default_models_dir())]
    dir: PathBuf,
}

#[derive(Parser)]
struct PullArgs {
    /// Model to pull. A known shorthand (`llama3.2:1b`), a HuggingFace repo
    /// (`org/model`), or a pinned one (`org/model@<revision>`). Omit (with no
    /// `--repo`/`--file`) to list the known shorthands. Overridden by `--repo`.
    name: Option<String>,

    /// HuggingFace repo id, explicit form (overrides the positional name).
    #[arg(long)]
    repo: Option<String>,

    /// Branch, tag, or commit to pin (default: main). `name@<rev>` sets this too.
    #[arg(long, default_value = "main")]
    revision: String,

    /// Directory to download into (default: derived from the model name).
    #[arg(long)]
    out: Option<PathBuf>,

    /// HuggingFace token (only needed for gated repos). Prefer the `HF_TOKEN` env
    /// var: passing `--token` puts the secret in argv, visible via `ps`/`pgrep`.
    #[arg(long)]
    token: Option<String>,

    /// Fetch EXACTLY these repo files instead of auto-resolving `model.safetensors`
    /// and config/tokenizer. Repeatable. Use for repos shipping a named single-file
    /// artifact, e.g. `--file gemma-4-31B_q4_0-it.gguf` for a QAT GGUF.
    #[arg(long = "file")]
    files: Vec<String>,

    /// Render download progress as a full-screen ratatui frame (TTY only).
    #[arg(long)]
    tui: bool,

    /// Download even when the weights alone exceed what arf will load on this machine
    /// (90% of RAM). Without it, `pull` checks the size before downloading and refuses a
    /// model `arf serve` would refuse to load here.
    #[arg(long)]
    force: bool,
}

#[derive(Parser)]
struct GenerateArgs {
    /// Path to a safetensors file/dir, or a single-file GGUF (detected by its
    /// `GGUF` magic — e.g. an ollama blob under `~/.ollama/models/blobs/`).
    #[arg(long)]
    model: PathBuf,

    /// Model architecture/config to use when it can't be read from a sibling
    /// `config.json` — required for a GGUF blob (which has none). Currently:
    /// `qwen3-coder-30b`. Ignored when a `config.json` is present.
    #[arg(long)]
    arch: Option<String>,

    /// Path to tokenizer.json. Defaults to `tokenizer.json` beside the model (in the
    /// model directory, or next to a safetensors file) when `--prompt` is used; a GGUF
    /// uses its embedded tokenizer. Not needed with --tokens.
    #[arg(long)]
    tokenizer: Option<PathBuf>,

    /// Prompt text (needs a tokenizer: --tokenizer, the model's own tokenizer.json, or a
    /// GGUF's embedded one).
    #[arg(long)]
    prompt: Option<String>,

    /// Force-wrap `--prompt` in the model's chat template. Picked from the tokenizer's
    /// special tokens by name (Gemma 4 `<|turn>`/`<turn|>`, Gemma 3 `<start_of_turn>`,
    /// Qwen/ChatML `<|im_start|>`). NOTE: the template is now applied AUTOMATICALLY
    /// whenever the tokenizer carries those turn markers (instruct models), so this
    /// flag is only needed to force it on; use --raw to force it OFF.
    #[arg(long)]
    chat: bool,

    /// Feed `--prompt` to the model RAW (a genuine base-model completion), skipping the
    /// automatic chat-template wrapping that instruct models otherwise get. Ignored for
    /// `--tokens` (already raw) and base models with no turn markers.
    #[arg(long)]
    raw: bool,

    /// Raw prompt token ids, comma-separated (bypasses the tokenizer).
    #[arg(long, value_delimiter = ',')]
    tokens: Option<Vec<u32>>,

    /// Maximum new tokens to generate.
    #[arg(long, default_value_t = 64)]
    max_tokens: usize,

    /// Sampling temperature (0 = greedy).
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    #[arg(long)]
    top_k: Option<usize>,

    #[arg(long)]
    top_p: Option<f32>,

    /// RNG seed for sampling.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Repetition penalty (`1.0` = off; ~1.1–1.3 suppresses repetition loops).
    /// Penalizes already-generated tokens on both greedy and stochastic decoding.
    #[arg(long, default_value_t = 1.0)]
    repetition_penalty: f32,

    /// Penalize only the last N generated tokens (`0` = the full history).
    #[arg(long, default_value_t = 0)]
    repeat_last_n: usize,

    /// Backend: `cpu` (SIMD) or `gpu` (requires building with `--features gpu`).
    #[arg(long, default_value = "cpu")]
    device: String,

    /// Size of the RoPE table / max context.
    #[arg(long, default_value_t = 8192)]
    max_context: usize,

    /// Render a live view (streaming text + KV-cache bar + tok/s) while decoding.
    #[arg(long)]
    watch: bool,

    /// GPU only: decode this many copies of the prompt concurrently (continuous
    /// batching) and report aggregate tok/s. Measures batched throughput.
    #[arg(long, default_value_t = 1)]
    batch: usize,

    /// Override the model's `vocab_size` (e.g. a padded GGUF embedding like
    /// gemma3's 262208). Needed when the GGUF tensor rows differ from the arch
    /// config's logical vocab.
    #[arg(long)]
    vocab: Option<usize>,

    /// Weight quantization: `none` (bf16, default), `int8` (per-row), or `q4`
    /// (Q4_0 block-32 4-bit — half the int8 bytes on the matmuls).
    #[arg(long, default_value = "none")]
    quant: String,

    /// KV-cache quantization (GPU only): `none` (f32, default) or TurboQuant
    /// `tq2`/`tq3`/`tq4` (2/3/4-bit packed KV — shrinks the cache and the
    /// attention read bandwidth that bounds long-context decode). Append `q` for
    /// the 1-bit QJL residual estimator, e.g. `tq3q`.
    #[arg(long, default_value = "none")]
    kv_quant: String,

    /// Emit perf instrumentation (CPU spans + per-kernel GPU timings) to stderr.
    /// Requires building with `--features profiling`. Override verbosity with
    /// `RUST_LOG` (e.g. `RUST_LOG=arf_gpu::gpu=trace`).
    #[arg(long)]
    profile: bool,
}

/// Install a `tracing` subscriber that prints span durations and GPU-timing
/// events to stderr. No-op when built without the `profiling` feature.
#[cfg(feature = "profiling")]
fn init_profiling() {
    use tracing_subscriber::fmt::format::FmtSpan;
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,arf_gpu::gpu=debug"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_span_events(FmtSpan::CLOSE)
        .with_writer(std::io::stderr)
        .try_init();
}

/// RAII: on drop (any return path out of `generate`) prints the by-time per-kernel
/// summary and writes `arf-profile.json` — the shareable profiler artifact that
/// also feeds the HUD. Only built under the `profiling` feature.
#[cfg(feature = "profiling")]
struct ProfileDump;
#[cfg(feature = "profiling")]
impl Drop for ProfileDump {
    fn drop(&mut self) {
        // M4 Max peak memory bandwidth; the roofline the GB/s figures compare against.
        const ROOFLINE_GBPS: f64 = 546.0;
        let snap = arf_gpu::gpu::profile::snapshot();
        if snap.steps == 0 {
            return;
        }
        let by_time = snap.by_time();
        let total = snap.total_us.max(1e-9);
        eprintln!(
            "\n=== GPU profile ({} steps, {:.1}ms total) ===",
            snap.steps,
            snap.total_us / 1000.0
        );
        eprintln!(
            "  {:<22} {:>6} {:>10} {:>7} {:>9} {:>8}",
            "kernel", "calls", "total_us", "%frame", "GB/s", "roof%"
        );
        for (name, st) in by_time.iter().take(20) {
            let g = st.mean_gbps();
            eprintln!(
                "  {:<22} {:>6} {:>10.1} {:>6.1}% {:>9.1} {:>7.0}%",
                name,
                st.calls,
                st.total_us,
                100.0 * st.total_us / total,
                g,
                if ROOFLINE_GBPS > 0.0 {
                    100.0 * g / ROOFLINE_GBPS
                } else {
                    0.0
                }
            );
        }
        let json = arf_gpu::gpu::profile::to_json(&snap, ROOFLINE_GBPS);
        if std::fs::write("arf-profile.json", &json).is_ok() {
            eprintln!("  → wrote arf-profile.json");
        }
    }
}

/// The chosen backend. `Cpu` runs core's SIMD engine; `Gpu` holds a resident
/// `wgpu` context for the fully-on-device decode path (a `GpuModel`).
pub(crate) enum DeviceChoice {
    Cpu,
    Gpu(std::sync::Arc<arf_gpu::GpuContext>),
}

impl DeviceChoice {
    pub(crate) fn describe(&self) -> String {
        match self {
            DeviceChoice::Cpu => "cpu (simd + threads)".to_string(),
            DeviceChoice::Gpu(ctx) => format!("gpu: {}", ctx.info()),
        }
    }
}

/// Resolve the `--device` flag into a backend.
pub(crate) fn resolve_device(name: &str) -> Result<DeviceChoice, Box<dyn Error>> {
    match name.to_ascii_lowercase().as_str() {
        "cpu" => Ok(DeviceChoice::Cpu),
        "gpu" => Ok(DeviceChoice::Gpu(std::sync::Arc::new(
            arf_gpu::GpuContext::new()?,
        ))),
        other => Err(format!("unknown device {other:?} (expected cpu or gpu)").into()),
    }
}

fn main() {
    // Print our errors cleanly: real line breaks (friendly multi-line "did you
    // mean?" messages read properly) and an accent-red `error:` prefix, instead of
    // the default `Error: "...\n..."` Debug rendering of `main() -> Result`.
    if let Err(e) = run() {
        let ui = ui::Ui::from_env(false);
        eprintln!("{} {e}", ui.red("error:"));
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let no_color = cli.no_color;
    let Some(command) = cli.command else {
        welcome::show(
            ui::Ui::from_env(no_color),
            &pull::default_models_dir(),
            version_string(),
        );
        return Ok(());
    };
    match command {
        Command::Generate(args) => generate(args),
        Command::Status => daemon::status(ui::Ui::from_env(no_color)),
        Command::Stop => daemon::stop(ui::Ui::from_env(no_color)),
        Command::Watch(a) => daemon::watch(a.port, a.pid),
        Command::WarmClaude(a) => warm::claude(a.port, &a.model_id, &a.claude),
        Command::Launch(a) => launch(a, no_color),
        Command::Pull(args) => run_pull(args),
        Command::List(a) => {
            let ui = ui::Ui::from_env(no_color);
            let sort = match a.sort {
                SortArg::Name => list::SortKey::Name,
                SortArg::Size => list::SortKey::Size,
                SortArg::State => list::SortKey::State,
            };
            let state = a.state.map(|s| match s {
                StateFilter::Ready => list::State::Ready,
                StateFilter::Downloading => list::State::Downloading,
                StateFilter::Incomplete => list::State::Incomplete,
            });
            list::list(&a.dir, ui, a.json, sort, state)
        }
        Command::Rm(a) => {
            let ui = ui::Ui::from_env(no_color);
            rm::remove(&a.dir, &a.model, ui)?;
            Ok(())
        }
        Command::DumpLogits(args) => dump_logits::dump_logits(args),
        Command::Perplexity(args) => perplexity::perplexity(args),
        Command::Serve(a) => {
            ensure_local(&a.model, no_color)?;
            serve_shim::run(&a.model, &a.rest)
        }
        Command::Run(a) if a.in_process => run_chat(a, no_color),
        Command::Run(a) => run_http(a, no_color),
        Command::Completions(a) => {
            print!(
                "{}{}{}",
                completions::install_hint(a.shell),
                completions::generate_to_string::<Cli>(a.shell, "arf"),
                completions::dynamic_pull_snippet(a.shell)
            );
            Ok(())
        }
        Command::ModelNames(a) => {
            use std::collections::BTreeSet;
            let mut names: BTreeSet<String> = BTreeSet::new();
            for e in registry::load(&a.dir) {
                names.insert(e.slug);
            }
            if let Ok(rd) = std::fs::read_dir(&a.dir) {
                for d in rd.flatten() {
                    if d.path().is_dir() {
                        if let Some(n) = d.file_name().to_str() {
                            names.insert(n.to_string());
                        }
                    }
                }
            }
            for n in names {
                println!("{n}");
            }
            Ok(())
        }
        Command::HfSearch(a) => {
            // Best-effort: `search` returns empty on every failure path, so this
            // prints nothing and exits 0 — a completion callout can never hang or
            // error the user's shell.
            for id in hf_search::search(&a.query) {
                println!("{id}");
            }
            Ok(())
        }
        Command::Doctor(a) => {
            let ui = ui::Ui::from_env(no_color);
            doctor::doctor(ui, &a.dir)
        }
        Command::Inspect(a) => inspect::inspect(&a.file, a.all),
    }
}

/// The local model directory for `model`. When it is a known shorthand that is not downloaded yet
/// and a person is at the terminal, offer to download it first; otherwise the "pull it first"
/// error, as before.
fn ensure_local(model: &str, no_color: bool) -> Result<PathBuf, Box<dyn Error>> {
    use std::io::IsTerminal;
    let dir = pull::default_models_dir();
    match serve_shim::resolve_local_model(model, &dir.to_string_lossy()) {
        Ok(p) => Ok(p),
        Err(e) => {
            let interactive = std::io::stdin().is_terminal() && std::io::stderr().is_terminal();
            if !interactive || !pull::is_known_alias(model) {
                return Err(e);
            }
            let ui = ui::Ui::from_env(no_color);
            eprint!(
                "{} isn't downloaded yet. Download it now into {}? [Y/n] ",
                ui.bold(model),
                dir.display()
            );
            std::io::Write::flush(&mut std::io::stderr())?;
            let mut answer = String::new();
            std::io::stdin().read_line(&mut answer)?;
            if !matches!(
                answer.trim().to_ascii_lowercase().as_str(),
                "" | "y" | "yes"
            ) {
                return Err(e);
            }
            run_pull(PullArgs {
                name: Some(model.to_string()),
                repo: None,
                revision: "main".into(),
                out: None,
                token: None,
                files: Vec::new(),
                tui: false,
                force: false,
            })?;
            serve_shim::resolve_local_model(model, &dir.to_string_lossy())
        }
    }
}

/// `arf run` — chat with the model through a background `arf-serve` (the fast path: the model's
/// own chat template, its draft, the prefix cache across turns). See `daemon`.
fn run_http(a: RunArgs, no_color: bool) -> Result<(), Box<dyn Error>> {
    let ui = ui::Ui::from_env(no_color);
    let name = a
        .model
        .clone()
        .unwrap_or_else(|| welcome::default_model(&pull::default_models_dir()));
    let path = ensure_local(&name, no_color)?;
    let server = daemon::ensure(&path, a.arch.as_deref(), ui)?;
    let label = format!("{name} · localhost:{}", server.port);
    run::chat_http(
        ui,
        &server,
        &label,
        &run::ChatOpts {
            max_tokens: a.max_tokens,
            think: !a.no_think,
        },
    )
}

/// `arf launch <agent>` — make sure the model is served, then start the agent pointed at it.
/// The agent's own configuration is left alone: everything is passed in its environment.
fn launch(a: LaunchArgs, no_color: bool) -> Result<(), Box<dyn Error>> {
    let ui = ui::Ui::from_env(no_color);
    let name = a
        .model
        .clone()
        .unwrap_or_else(|| welcome::default_model(&pull::default_models_dir()));
    let path = ensure_local(&name, no_color)?;
    let server = daemon::ensure(&path, None, ui)?;
    let base = format!("http://127.0.0.1:{}", server.port);
    let id = server.model_id.clone();
    let mut warming = false;
    let (bin, mut cmd) = match a.agent {
        Agent::Claude => {
            let bin = find_agent("claude", &[".local/bin/claude", ".claude/local/claude"]).ok_or(
                "Claude Code is not installed (https://docs.claude.com/en/docs/claude-code)",
            )?;
            let mut c = std::process::Command::new(&bin);
            c.env_remove("ANTHROPIC_API_KEY");
            // Claude Code's own background model calls go to this same local model and compete
            // with the user's message for the GPU (measured 2026-10-06: a ~5K- and a ~14K-token
            // side prompt beside a 44-54K-token first message). Off unless the user set them:
            // the session-title request, and the next-prompt suggestions sent after every answer
            // (code.claude.com/docs/en/env-vars). The list is shared with `warm`, whose capture
            // must see the same prompts a launched session does.
            for (k, v) in warm::claude_env(&base, &id) {
                if !warm::USER_MAY_SET.contains(&k) || std::env::var_os(k).is_none() {
                    c.env(k, v);
                }
            }
            // The model's real window, so auto-compact neither assumes 200K nor cuts early.
            if let Some(n) = server.context {
                c.env("CLAUDE_CODE_MAX_CONTEXT_TOKENS", n.to_string());
            }
            // Its auto-mode safety check (~30,500 tokens), read in the background now rather
            // than at the first tool call (`warm`).
            if warm::auto_mode(&a.rest) {
                warm::spawn(server.port, &id, &bin);
                warming = std::env::var_os("ARF_NO_WARM").is_none();
            }
            (bin, c)
        }
        Agent::Opencode => {
            let bin = find_agent("opencode", &[".opencode/bin/opencode"])
                .ok_or("OpenCode is not installed (https://opencode.ai)")?;
            let cfg = serde_json::json!({
                "$schema": "https://opencode.ai/config.json",
                "model": format!("arf/{id}"),
                "small_model": format!("arf/{id}"),
                "provider": {"arf": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "arf (local)",
                    "options": {"baseURL": format!("{base}/v1"), "apiKey": "arf", "timeout": false},
                    "models": {id.clone(): {"name": format!("{id} (arf)"), "tool_call": true}}
                }}
            });
            let mut c = std::process::Command::new(&bin);
            c.env("OPENCODE_CONFIG_CONTENT", cfg.to_string());
            (bin, c)
        }
    };
    cmd.args(&a.rest);
    eprintln!(
        "{} {} on {} {}",
        ui.accent("●"),
        bin.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
        ui.bold(&name),
        ui.faint(&format!("({base})"))
    );
    // An agent's first message carries its whole system prompt and tool definitions (Claude
    // Code's: ~23K tokens, measured 2026-10-06), and a Mac reads a prompt at ~170 tokens/s on an
    // M4 Max: minutes of silence that look like a hang unless said here.
    eprintln!(
        "  {}",
        ui.faint(
            "the first message also sends the agent's instructions and tools (~20K tokens): the \
             model reads them first, a few minutes on a Mac (~2-3 on an M4 Max). Later sessions \
             reuse them, and a restarted server loads them from disk."
        )
    );
    if warming {
        eprintln!(
            "  {}",
            ui.faint(
                "Claude Code's auto-mode safety check (~30K tokens) is being read in the \
                 background now, while the GPU is free, so tool calls need not wait for it."
            )
        );
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let err = cmd.exec();
        Err(format!("could not start {}: {err}", bin.display()).into())
    }
    #[cfg(not(unix))]
    {
        let status = cmd.status()?;
        std::process::exit(status.code().unwrap_or(1));
    }
}

/// An agent CLI on PATH, or at one of its usual install places under the home directory.
fn find_agent(name: &str, home_paths: &[&str]) -> Option<PathBuf> {
    if let Some(paths) = std::env::var_os("PATH") {
        for d in std::env::split_paths(&paths) {
            let p = d.join(name);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    let home = PathBuf::from(std::env::var_os("HOME")?);
    home_paths
        .iter()
        .map(|p| home.join(p))
        .find(|p| p.is_file())
}

/// `arf run <model>` — load the GPU model once (resident), then chat turn by
/// turn. Each turn is an independent `generate_stop` from the freshly-encoded
/// prompt: there is NO multi-turn KV reuse yet (acceptable for v1 — the model is
/// resident so only the per-turn decode is paid). GPU-only: the resident-decode
/// path lives behind `DeviceChoice::Gpu`.
fn run_chat(a: RunArgs, no_color: bool) -> Result<(), Box<dyn Error>> {
    use arf_core::config::{KvQuant, Quant};
    use std::time::Instant;

    let ui = ui::Ui::from_env(no_color);

    // Resolve the model dir with a friendly error FIRST (before the GPU gate) so a
    // typo'd alias (`gema3`) surfaces "did you mean 'gemma3'?" instead of the
    // device error. Mirrors serve_shim's front-door resolution.
    let model_name = a
        .model
        .clone()
        .unwrap_or_else(|| welcome::default_model(&pull::default_models_dir()));
    let model_path: PathBuf = serve_shim::resolve_local_model(
        &model_name,
        &pull::default_models_dir().to_string_lossy(),
    )?;

    let cfg = resolve_config(&model_path, a.arch.as_deref())?;
    // FAST PATH BY DEFAULT — the same measured-winner env levers `arf serve` defaults
    // (see `arf_gpu::apply_fast_path_defaults`). Without this, in-process chat silently
    // ran the slow portable path: the fast path is env-gated and only the daemon defaulted
    // it. Must run before the GPU model load below (where ARF_MSL_GEMV is read), and after the
    // config is resolved: a model the island cannot run is declined first, from its config.
    arf_gpu::decline_fast_path_for_model("cli", &cfg);
    arf_gpu::apply_fast_path_defaults("cli");
    let device = resolve_device(&a.device)?;
    let DeviceChoice::Gpu(ctx) = &device else {
        return Err("arf run currently requires --device gpu".into());
    };

    let tok_path = a
        .tokenizer
        .clone()
        .or_else(|| sibling_tokenizer(&model_path));
    let tokenizer = resolve_tokenizer(tok_path.as_ref(), &model_path)?;
    let tok = tokenizer.ok_or_else(|| {
        format!(
            "arf run requires a tokenizer: {}",
            tokenizer_hint(&model_path)
        )
    })?;

    // Load the GpuModel ONCE; it stays resident across turns. Default quant
    // (bf16 weights, f32 KV) — `run` is the simple resident-chat path.
    eprintln!(
        "loading onto {} for resident GPU chat...",
        device.describe()
    );
    let model = if is_gguf(&model_path) {
        arf_gpu::weights::load_gguf_gpu_kv_quant(
            &cfg,
            &model_path,
            a.max_context,
            ctx,
            Quant::None,
            KvQuant::None,
        )?
    } else {
        let paths = collect_safetensors(&model_path)?;
        arf_gpu::weights::load_safetensors_gpu_kv_quant(
            &cfg,
            &paths,
            a.max_context,
            ctx,
            Quant::None,
            KvQuant::None,
        )?
    };

    // End-of-turn / EOS ids resolved by NAME (no per-model hardcoding) so a chat
    // turn ends instead of looping to --max-tokens. Computed ONCE.
    let stop = chat_stop_tokens(&tok);
    // Owned label so the &str handed to `repl` does not borrow anything the
    // closure moves.
    let label = format!("arf run · {} · {}", model_path.display(), device.describe());

    let max_tokens = a.max_tokens.unwrap_or(256);
    run::repl(ui, &label, move |prompt| {
        // Mirror resolve_prompt's BOS logic: let the tokenizer insert its own
        // special tokens, then prepend the model's BOS if it is not already there.
        let mut ids = tok.encode(prompt, true)?;
        if let Some(b) = bos_token(&tok) {
            if ids.first() != Some(&b) {
                ids.insert(0, b);
            }
        }
        let t0 = Instant::now();
        let out = model.generate_stop(&ids, max_tokens, &stop);
        let secs = t0.elapsed().as_secs_f64();
        let text = tok.decode(&out, true)?;
        let stats = run::Stats {
            tokens: out.len(),
            tok_per_s: out.len() as f64 / secs.max(1e-6),
            // ttft is not measurable without per-token streaming hooks; v1 sets 0.
            ttft_ms: 0,
        };
        Ok((text, stats))
    })
}

/// Resolve a pull request — shorthand/`org/repo[@rev]`/`--repo` → (repo,
/// revision, out dir) — then download.
fn run_pull(args: PullArgs) -> Result<(), Box<dyn Error>> {
    // Bare `arf pull` (no name, no --repo, no --file): list the known shorthands
    // as a discovery aid instead of silently defaulting to one mirror.
    if args.name.is_none() && args.repo.is_none() && args.files.is_empty() {
        let ui = ui::Ui::from_env(false); // pull doesn't take --no-color here; from_env still honors NO_COLOR / non-TTY.
        println!("{}", ui.dim("known models — pull one by its shorthand:"));
        for b in pull::BUNDLES {
            let alias = b.aliases[0];
            println!("  {}  {}", ui.bold(&format!("{alias:<14}")), ui.dim(b.what));
        }
        for (alias, repo) in pull::known_aliases() {
            println!("  {}  {}", ui.bold(&format!("{alias:<14}")), ui.dim(repo));
        }
        println!(
            "\n{}",
            ui.dim("or pull any HuggingFace repo: arf pull <org>/<model>[@rev]")
        );
        return Ok(());
    }

    // Precedence: explicit --repo wins; else the positional name; else default.
    let spec = args
        .repo
        .clone()
        .or_else(|| args.name.clone())
        .unwrap_or_else(|| pull::DEFAULT_REPO.to_string());

    // A bare positional name with no '/' that isn't a known shorthand looks like a
    // typo of one (`gema3`). Hint the closest alias but DON'T block — the user may
    // genuinely want a raw repo. Only when --repo wasn't given (that's explicit).
    // A RETIRED shorthand is different: it does block, because proceeding would fetch
    // nothing and say nothing useful.
    if args.repo.is_none() {
        if let Some(name) = &args.name {
            if let Some(why) = pull::retired_alias(name) {
                return Err(why.into());
            }
            let is_known = pull::is_known_alias(name);
            if !name.contains('/') && !is_known {
                if let Some(s) = pull::suggest_alias(name) {
                    eprintln!(
                        "note: '{name}' isn't a known shorthand — did you mean '{s}'? (proceeding as a raw repo)"
                    );
                }
            }
        }
    }

    let resolved = pull::resolve_model(&spec);
    // A `@rev` in the spec sets the revision unless --revision was passed.
    let revision = if args.revision != "main" {
        args.revision.clone()
    } else {
        resolved.revision.unwrap_or_else(|| "main".to_string())
    };
    let out = args
        .out
        .unwrap_or_else(|| pull::default_models_dir().join(&resolved.out_slug));

    // Record provenance up front — repo + revision are known before the (slow)
    // weight download, so `list` shows the source even mid-pull. State (ready /
    // downloading) is reconciled from disk by `list`, so an interrupted pull
    // still reads correctly. Best-effort: a registry write never fails the pull.
    let slug = out
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(&resolved.out_slug)
        .to_string();
    let models_dir = out.parent().unwrap_or(Path::new(".")).to_path_buf();
    if let Err(e) = registry::record(
        &models_dir,
        registry::Entry {
            slug: slug.clone(),
            repo: resolved.repo.clone(),
            revision: revision.clone(),
            pulled_at: registry::now_unix(),
        },
    ) {
        eprintln!("note: could not update model registry: {e}");
    }

    // Prefer the explicit `--token`, but fall back to the `HF_TOKEN` env var so the
    // secret never has to appear in argv (visible in `ps`/`pgrep`). Env is the safer
    // default for gated pulls.
    let token = args
        .token
        .clone()
        .or_else(|| std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty()));

    // A BUNDLE shorthand (`qwen3.8:27b`) assembles one directory from several repos: the
    // weights, a block draft and a vision projector, plus the `--arch` / `--quant` arf-serve
    // needs — so `arf serve <alias>` takes no flags. See `arf_core::bundle`.
    if let Some(spec) = resolved.bundle {
        if !args.files.is_empty() {
            return Err(format!(
                "--file does not apply to the bundle shorthand '{}' (it fetches a fixed set of files)",
                spec.aliases[0]
            )
            .into());
        }
        if let Err(e) = pull::pull_bundle(
            spec,
            &revision,
            &out,
            token.as_deref(),
            args.tui,
            args.force,
        ) {
            // Refused before downloading (or unreachable): no directory was made, so drop the
            // provenance row recorded above rather than leave it pointing at nothing.
            if !out.exists() {
                let _ = registry::remove(&models_dir, &slug);
            }
            return Err(e);
        }
        let ui = ui::Ui::from_env(false);
        let serve = if out == pull::default_models_dir().join(spec.dir) {
            format!("arf serve {}", spec.aliases[0])
        } else {
            format!("arf serve {}", out.display())
        };
        println!("\n{} {}", ui.dim("serve it:"), ui.accent(&serve));
        return Ok(());
    }

    let only = (!args.files.is_empty()).then_some(args.files.as_slice());
    if let Err(e) = pull::pull(
        &resolved.repo,
        &revision,
        &out,
        token.as_deref(),
        only,
        resolved.quant.as_deref(),
        args.tui,
        args.force,
    ) {
        // A pull refused before downloading (the memory fit check) removes the directory it
        // made; drop the provenance row recorded above with it, so no entry points at nothing.
        if !out.exists() {
            let _ = registry::remove(&models_dir, &slug);
        }
        return Err(e);
    }

    // A GGUF pull has no config.json, so running it needs an explicit --arch. Print
    // an actionable hint using a token `config_for_arch` accepts (best-effort guess
    // from the slug; the user can override).
    if resolved.quant.is_some() {
        let ui = ui::Ui::from_env(false);
        let arch_hint = guess_arch(&resolved.out_slug).unwrap_or("<arch>");
        println!(
            "\n{} GGUF pulled. Run it with an explicit arch:\n  {}",
            ui.dim("note:"),
            ui.accent(&format!("arf run {} --arch {arch_hint}", resolved.out_slug)),
        );
    }
    Ok(())
}

/// Best-effort architecture token (one `config_for_arch` accepts) guessed from a
/// model slug, for the GGUF `--arch` run-hint. `None` when nothing matches.
fn guess_arch(slug: &str) -> Option<&'static str> {
    let s = slug.to_ascii_lowercase();
    if s.contains("gemma-4-31b") || s.contains("gemma4-31b") {
        Some("gemma-4-31b")
    } else if s.contains("gemma-4-12b") || s.contains("gemma4-12b") || s.contains("gemma-4") {
        Some("gemma-4-12b")
    } else if s.contains("gemma-3") || s.contains("gemma3") {
        Some("gemma3-4b")
    } else if s.contains("qwen3") {
        Some("qwen3-coder-30b")
    } else if s.contains("llama-3.2") || s.contains("llama3.2") {
        Some("llama-3.2-1b")
    } else {
        None
    }
}

fn generate(mut args: GenerateArgs) -> Result<(), Box<dyn Error>> {
    use arf_core::config::{KvQuant, Quant};
    // `--prompt` without `--tokenizer`: use the model's own `tokenizer.json` (the Makefile
    // quickstart used to pass it explicitly). Raw `--tokens` runs keep their id-only path —
    // loading a tokenizer there would change their stop tokens and their printed output.
    if args.tokenizer.is_none() && args.tokens.is_none() {
        args.tokenizer = sibling_tokenizer(&args.model);
    }
    #[cfg(feature = "profiling")]
    let _profile_dump = if args.profile {
        init_profiling();
        Some(ProfileDump)
    } else {
        None
    };
    #[cfg(not(feature = "profiling"))]
    if args.profile {
        eprintln!("note: --profile has no effect; rebuild with `--features profiling`");
    }
    // Architecture from the model dir's config.json when present (Llama / Qwen3-MoE
    // / Gemma 3); fall back to the hand-built Llama-3.2-1B config for a bare
    // safetensors file with no sibling config.json.
    let mut cfg = resolve_config(&args.model, args.arch.as_deref())?;
    // Override vocab_size to match a GGUF whose embedding is padded (e.g. gemma3's
    // 262208 = 262144 logical + 64 pad rows); the tensor shape check needs the
    // physical row count. Mirrors arf-serve's --vocab.
    if let Some(v) = args.vocab {
        cfg.vocab_size = v;
    }
    // FAST PATH BY DEFAULT — same as run_chat above: decided from the resolved config, then the
    // levers defaulted; must precede the GPU model load.
    arf_gpu::decline_fast_path_for_model("cli", &cfg);
    arf_gpu::apply_fast_path_defaults("cli");
    let device = resolve_device(&args.device)?;
    // Per-run opt-in: only `--profile` turns on the timestamp/readback path, so a
    // profiling-feature build runs at full speed otherwise.
    if args.profile {
        if let DeviceChoice::Gpu(ctx) = &device {
            ctx.set_profiling(true);
        }
    }
    let model_is_gguf = is_gguf(&args.model);
    let quant: Quant = args.quant.to_ascii_lowercase().parse()?;
    let kv_quant = match args.kv_quant.to_ascii_lowercase().as_str() {
        "none" => KvQuant::None,
        "tq2" => KvQuant::Tq { bits: 2, qjl: false },
        "tq3" => KvQuant::Tq { bits: 3, qjl: false },
        "tq4" => KvQuant::Tq { bits: 4, qjl: false },
        "tq2q" => KvQuant::Tq { bits: 2, qjl: true },
        "tq3q" => KvQuant::Tq { bits: 3, qjl: true },
        "tq4q" => KvQuant::Tq { bits: 4, qjl: true },
        other => {
            return Err(format!(
                "unknown --kv-quant {other:?} (expected none, tq2, tq3, tq4, or the qjl variants tq2q/tq3q/tq4q)"
            )
            .into())
        }
    };
    if kv_quant.is_quantized() && !matches!(device, DeviceChoice::Gpu(_)) {
        return Err("--kv-quant is GPU-only; pass --device gpu".into());
    }
    // GGUF is a single file loaded directly; safetensors collects shards.
    let paths = if model_is_gguf {
        vec![args.model.clone()]
    } else {
        collect_safetensors(&args.model)?
    };

    // GPU path: the whole forward pass runs on the device via GpuModel.
    if let DeviceChoice::Gpu(ctx) = &device {
        let tokenizer = resolve_tokenizer(args.tokenizer.as_ref(), &args.model)?;
        let prompt_ids = resolve_prompt(&args, tokenizer.as_ref())?;
        eprintln!(
            "loading onto {} for resident GPU decode...",
            device.describe()
        );
        let model = if model_is_gguf {
            arf_gpu::weights::load_gguf_gpu_kv_quant(
                &cfg,
                &args.model,
                args.max_context,
                ctx,
                quant,
                kv_quant,
            )?
        } else {
            arf_gpu::weights::load_safetensors_gpu_kv_quant(
                &cfg,
                &paths,
                args.max_context,
                ctx,
                quant,
                kv_quant,
            )?
        };
        if kv_quant.is_quantized() {
            eprintln!(
                "kv-cache: {} (TurboQuant)",
                args.kv_quant.to_ascii_lowercase()
            );
        }

        if args.batch > 1 {
            // Continuous-batching throughput: decode `batch` copies of the prompt
            // concurrently and report aggregate tok/s (tokens across all sequences
            // per wall-second) — the metric a server cares about.
            let prompts = vec![prompt_ids.clone(); args.batch];
            let start = std::time::Instant::now();
            let outs = model.generate_batch(&prompts, args.max_tokens);
            let secs = start.elapsed().as_secs_f64();
            let total: usize = outs.iter().map(|o| o.len()).sum();
            eprintln!(
                "gpu batch={}: {} tokens ({}/seq) in {:.2}s = {:.1} tok/s aggregate, {:.1} tok/s/seq",
                args.batch,
                total,
                args.max_tokens,
                secs,
                total as f64 / secs,
                total as f64 / secs / args.batch as f64,
            );
            if let Some(tok) = &tokenizer {
                println!("{}", tok.decode(&outs[0], true)?);
            }
            return Ok(());
        }

        // Stop at the model's end-of-turn / EOS markers so a chat prompt ends a
        // single turn instead of looping to --max-tokens. Resolved by NAME from the
        // tokenizer (no per-model hardcoded ids), so it works across Gemma / Qwen /
        // Llama; empty when no tokenizer (raw --tokens) → the fused fast path.
        let stop = tokenizer.as_ref().map(chat_stop_tokens).unwrap_or_default();
        let start = std::time::Instant::now();
        let out = model.generate_stop(&prompt_ids, args.max_tokens, &stop);
        let secs = start.elapsed().as_secs_f64();
        eprintln!(
            "gpu: {} tokens in {:.2}s = {:.2} tok/s",
            out.len(),
            secs,
            out.len() as f64 / secs.max(1e-6)
        );
        // Debug: surface the raw token ids so an empty/garbage decode is diagnosable.
        if std::env::var("ARF_SHOW_IDS").is_ok() {
            eprintln!("token ids: {out:?}");
        }
        match &tokenizer {
            Some(tok) => println!("{}", tok.decode(&out, true)?),
            None => println!("{out:?}"),
        }
        return Ok(());
    }

    if model_is_gguf {
        return Err(
            "GGUF load is only wired for the GPU path; pass --device gpu \
             (and build with --features gpu)"
                .into(),
        );
    }

    eprintln!(
        "loading {} safetensors shard(s) onto {} ({quant})...",
        paths.len(),
        device.describe(),
    );
    let model = weights::load_safetensors_quant(
        &cfg,
        &paths,
        args.max_context,
        &arf_core::Device::Cpu,
        quant,
    )?;

    let tokenizer = resolve_tokenizer(args.tokenizer.as_ref(), &args.model)?;
    let prompt_ids = resolve_prompt(&args, tokenizer.as_ref())?;

    // Stop at the model's end-of-turn / EOS markers, resolved by NAME from the
    // tokenizer (mirrors the GPU path) so a chat prompt ends one turn instead of
    // looping to --max-tokens; empty when no tokenizer (raw --tokens).
    let stop_tokens = tokenizer.as_ref().map(chat_stop_tokens).unwrap_or_default();
    let params = SamplingParams {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        max_tokens: args.max_tokens,
        stop_tokens,
        seed: args.seed,
        repetition_penalty: args.repetition_penalty,
        repeat_last_n: args.repeat_last_n,
    };
    params.validate()?;

    let mut engine = LlmEngine::new(model, EngineConfig::default())?;

    if args.watch {
        let tok = tokenizer
            .as_ref()
            .ok_or("--watch requires --tokenizer (to show decoded text)")?;
        let header = format!("arf · llama-3.2-1b · {}", device.describe());
        watch::watch_generate(&mut engine, tok, prompt_ids, params, &header)?;
        return Ok(());
    }

    let prompt_len = prompt_ids.len();
    let start = std::time::Instant::now();
    let output = engine.generate(prompt_ids, params)?;
    let secs = start.elapsed().as_secs_f64();
    // Symmetric with the GPU path: report decode rate over generated tokens
    // (prompt prefill folded in — the same convention the GPU branch prints).
    let gen = output.len().saturating_sub(prompt_len).max(1);
    eprintln!(
        "cpu: {} tokens in {:.2}s = {:.2} tok/s",
        gen,
        secs,
        gen as f64 / secs.max(1e-6),
    );
    match &tokenizer {
        Some(tok) => println!("{}", tok.decode(&output, true)?),
        None => println!("{output:?}"),
    }
    Ok(())
}

/// Resolve the prompt into token ids from either `--tokens` or `--prompt`.
fn resolve_prompt(
    args: &GenerateArgs,
    tokenizer: Option<&Tokenizer>,
) -> Result<Vec<u32>, Box<dyn Error>> {
    if let Some(tokens) = &args.tokens {
        return Ok(tokens.clone());
    }
    let prompt = args
        .prompt
        .as_ref()
        .ok_or("provide --prompt (with --tokenizer) or --tokens")?;
    let tok = tokenizer.ok_or_else(|| {
        format!(
            "--prompt needs a tokenizer: {}",
            tokenizer_hint(&args.model)
        )
    })?;
    // Chat mode: wrap the prompt in the model's turn template (markers are special
    // tokens, so build the id sequence directly around the encoded prompt body).
    // AUTO: an instruction-tuned model (its tokenizer carries turn markers) emits
    // degenerate output (e.g. gemma-4-12B-it loops digit tokens) when fed a RAW
    // completion prompt — it was trained to answer only inside `<|turn>…<turn|>`
    // framing. So apply the template by DEFAULT when those markers exist; opt OUT
    // with --raw for a genuine base-model completion. (Raw `--tokens` is untouched.)
    // Only the schemes `chat_prompt_ids` can actually build (Gemma 4 / Gemma 3 / ChatML).
    // Llama-3's header format isn't built here, so don't auto-trigger on it (its base
    // 1B has no turn markers anyway, so this is moot in practice — but keep it honest).
    let has_turn_markers = tok.token_to_id("<|turn>").is_some()
        || tok.token_to_id("<start_of_turn>").is_some()
        || tok.token_to_id("<|im_start|>").is_some();
    if args.chat || (has_turn_markers && !args.raw) {
        return chat_prompt_ids(tok, prompt);
    }
    // `add_special_tokens = true` lets the tokenizer's own post-processor insert
    // BOS/template tokens. Some `tokenizer.json`s lack a BOS post-processor though,
    // and models like Gemma decode into a structural-token loop without a leading
    // BOS — so if the encoding does not already start with the model's BOS, prepend
    // it. (Raw `--tokens` is left untouched: that path is the caller's explicit ids.)
    let mut ids = tok.encode(prompt, true)?;
    if let Some(bos) = bos_token(tok) {
        if ids.first() != Some(&bos) {
            ids.insert(0, bos);
        }
    }
    Ok(ids)
}

/// Build a single-user-turn chat prompt as token ids, picking the model's turn
/// template by NAME from the tokenizer's special tokens (no per-model flag):
///   Gemma 4: `<bos> <|turn>user\n {body} <turn|>\n <|turn>model\n`
///   Gemma 3: `<bos> <start_of_turn>user\n {body} <end_of_turn>\n <start_of_turn>model\n`
///   ChatML (Qwen): `<|im_start|>user\n {body} <|im_end|>\n <|im_start|>assistant\n`
/// Falls back to a bare BOS + body if none of the schemes' tokens are present.
use arf_core::chat_template::muse_glimmer_system_body;

fn chat_prompt_ids(tok: &Tokenizer, prompt: &str) -> Result<Vec<u32>, Box<dyn Error>> {
    let id = |s: &str| tok.token_to_id(s);
    let nl = tok
        .token_to_id("\n")
        .map(|n| vec![n])
        .unwrap_or_else(|| tok.encode("\n", false).unwrap_or_default());
    let body = tok.encode(prompt, false)?;
    let bos = bos_token(tok);

    // Muse Glimmer: `<|start|>user<|message|>{body}<|eot|><|start|>assistant`. It does NOT fit
    // the (turn_start, role, turn_end) tuple below — the role is followed by a `<|message|>`
    // separator rather than a newline, and the generation cue is a bare `<|start|>assistant`
    // with no trailing separator (the model emits `<|message|>` itself, since the assistant
    // header is where it picks a recipient for tool calls). So it gets its own branch instead
    // of being bent into a shape it does not have.
    if let (Some(st), Some(msg), Some(eot)) = (id("<|start|>"), id("<|message|>"), id("<|eot|>")) {
        let role_ids = |r: &str| tok.encode(r, false).unwrap_or_default();
        let mut ids: Vec<u32> = bos.into_iter().collect();
        // The template injects a system turn when the caller gives none, and it is REQUIRED:
        // its "# Valid recipients" line is what tells the model which channels exist. Without
        // it the first token after `<|start|>assistant` is a bare ` to` (the model addressing
        // a recipient it was never told about) and it repeats forever.
        ids.push(st);
        ids.extend(role_ids("system"));
        ids.push(msg);
        ids.extend(tok.encode(&muse_glimmer_system_body(), false)?);
        ids.push(eot);
        ids.push(st);
        ids.extend(role_ids("user"));
        ids.push(msg);
        ids.extend(&body);
        ids.push(eot);
        ids.push(st);
        ids.extend(role_ids("assistant"));
        return Ok(ids);
    }

    // (turn_start, role-word, turn_end, gen-role) schemes, by special-token name.
    let scheme = if let (Some(s), Some(e)) = (id("<|turn>"), id("<turn|>")) {
        Some((s, "user", e, "model")) // Gemma 4
    } else if let (Some(s), Some(e)) = (id("<start_of_turn>"), id("<end_of_turn>")) {
        Some((s, "user", e, "model")) // Gemma 3
    } else if let (Some(s), Some(e)) = (id("<|im_start|>"), id("<|im_end|>")) {
        Some((s, "user", e, "assistant")) // ChatML / Qwen
    } else {
        None
    };

    let Some((turn_start, user_role, turn_end, gen_role)) = scheme else {
        eprintln!("note: --chat: no known turn markers in this tokenizer; using a bare prompt");
        let mut ids = bos.into_iter().collect::<Vec<_>>();
        ids.extend(body);
        return Ok(ids);
    };

    let role_ids = |r: &str| tok.encode(r, false).unwrap_or_default();
    let mut ids: Vec<u32> = bos.into_iter().collect();
    // <turn_start> user \n {body} <turn_end> \n <turn_start> model \n
    ids.push(turn_start);
    ids.extend(role_ids(user_role));
    ids.extend(&nl);
    ids.extend(&body);
    ids.push(turn_end);
    ids.extend(&nl);
    ids.push(turn_start);
    ids.extend(role_ids(gen_role));
    ids.extend(&nl);
    Ok(ids)
}

/// The model's begin-of-sequence id, resolved by NAME from the tokenizer (no
/// per-model hardcoding). Covers SentencePiece `<bos>` (Gemma/Llama) and the
/// GPT-style `<|begin_of_text|>`. `None` if the tokenizer has no BOS.
/// The model's BOS token id, resolved by NAME (SentencePiece `<bos>` or the
/// GPT-style `<|begin_of_text|>`), so there's no per-model hardcoding. `None` when
/// the tokenizer has neither.
pub(crate) fn bos_token(tok: &Tokenizer) -> Option<u32> {
    ["<bos>", "<|begin_of_text|>"]
        .iter()
        .find_map(|t| tok.token_to_id(t))
}

/// End-of-turn / EOS token ids to stop decode on, resolved by NAME from the
/// tokenizer (so no per-model hardcoding). Covers the common chat terminators:
/// Gemma 4's `<turn|>`, Gemma 3's `<end_of_turn>`, the SentencePiece `<eos>`, and
/// the GPT-style `<|endoftext|>` / `<|im_end|>` (Qwen). Only those present are kept.
fn chat_stop_tokens(tok: &Tokenizer) -> Vec<u32> {
    // `<|eot|>` is Muse Glimmer's TURN terminator. `<|eom|>` deliberately is NOT here: it ends
    // a MESSAGE inside a turn, and this model routinely emits it to close its `to=self` chain of
    // thought before re-heading into `to=user` with the actual answer. Stopping on it truncates
    // the reply to the reasoning alone — measured: 36 tokens of thinking and no answer.
    [
        "<turn|>",
        "<end_of_turn>",
        "<eos>",
        "<|endoftext|>",
        "<|im_end|>",
        "<|eot|>",
    ]
    .iter()
    .filter_map(|t| tok.token_to_id(t))
    .collect()
}

/// Where a model's own `tokenizer.json` would be: inside the model directory, or beside a
/// single weights file. `None` for a GGUF, whose tokenizer is embedded
/// ([`resolve_tokenizer`] reads it from the file itself).
fn tokenizer_json_candidate(model: &Path) -> Option<PathBuf> {
    if model.is_dir() {
        return Some(model.join("tokenizer.json"));
    }
    if is_gguf(model) {
        return None;
    }
    model.parent().map(|d| d.join("tokenizer.json"))
}

/// The model's own `tokenizer.json` when it exists — the default for `--tokenizer`.
pub(crate) fn sibling_tokenizer(model: &Path) -> Option<PathBuf> {
    tokenizer_json_candidate(model).filter(|p| p.is_file())
}

/// What to do when no tokenizer was found, naming the path that was looked for.
fn tokenizer_hint(model: &Path) -> String {
    match tokenizer_json_candidate(model) {
        Some(p) => format!(
            "pass --tokenizer, or put tokenizer.json beside the model (looked for {})",
            p.display()
        ),
        None => "pass --tokenizer".to_string(),
    }
}

/// Resolve the tokenizer for a run: explicit `--tokenizer`, else the GGUF-embedded
/// tokenizer when the model is a GGUF blob, else `None` (the `--tokens` id-only
/// paths still work without a tokenizer).
pub(crate) fn resolve_tokenizer(
    tokenizer: Option<&PathBuf>,
    model: &Path,
) -> Result<Option<Tokenizer>, Box<dyn Error>> {
    if let Some(p) = tokenizer {
        return Ok(Some(Tokenizer::from_file(p)?));
    }
    if is_gguf(model) {
        eprintln!("note: using the GGUF-embedded tokenizer (no --tokenizer given)");
        return Ok(Some(Tokenizer::from_gguf_path(model)?));
    }
    Ok(None)
}

/// True if `path` is a single file whose first four bytes are the `GGUF` magic.
/// Delegates to the shared arf-core implementation.
pub(crate) fn is_gguf(path: &Path) -> bool {
    arf_core::model::gguf::is_gguf_file(path)
}

/// Map an `--arch` name to a built-in [`ModelConfig`].
///
/// Delegates to `arf_core::config::config_for_arch` — the ONE table. This function used to
/// carry its own `match`, and arf-serve carried a second one; they drifted until the server
/// rejected models the CLI accepted.
fn config_for_arch(arch: &str) -> Result<ModelConfig, Box<dyn Error>> {
    arf_core::config::config_for_arch(arch).map_err(|e| e.into())
}

/// Resolve the model config: parse a sibling `config.json` (Llama / Qwen3-MoE /
/// Gemma 3) if the model path is a directory containing one, else default to the
/// hand-built Llama-3.2-1B config (for a bare `.safetensors` file).
pub(crate) fn resolve_config(
    model: &Path,
    arch: Option<&str>,
) -> Result<ModelConfig, Box<dyn Error>> {
    // An explicit --arch wins (required for a GGUF blob, which has no config.json).
    if let Some(arch) = arch {
        let cfg = config_for_arch(arch)?;
        cfg.validate()?;
        eprintln!("using architecture --arch {arch}");
        return Ok(cfg);
    }
    let dir = if model.is_dir() {
        Some(model.to_path_buf())
    } else {
        model.parent().map(|p| p.to_path_buf())
    };
    if let Some(dir) = dir {
        let cfg_path = dir.join("config.json");
        if cfg_path.is_file() {
            let cfg = arf_core::model::hf_config::load_config_json(&cfg_path)?;
            cfg.validate()?;
            eprintln!("loaded architecture from {}", cfg_path.display());
            return Ok(cfg);
        }
    }
    if is_gguf(model) {
        // The file's own header (2026-10-07): a GGUF pulled from any repository names its
        // architecture and geometry, and when both are a built-in config's, that is the config.
        let g = arf_core::model::gguf::LazyGguf::open_raw(model)?;
        if let Some(arch) = arf_core::config::arch_from_gguf(&g) {
            let cfg = config_for_arch(arch)?;
            cfg.validate()?;
            eprintln!("architecture from the GGUF header: {arch}");
            return Ok(cfg);
        }
        return Err(arf_core::config::gguf_header_mismatch(&g).into());
    }
    // A DIRECTORY WITH NOTHING TO LOAD IS AN ERROR, NOT A LLAMA (2026-10-07): a folder holding a
    // GGUF under another name fell through to the Llama-3.2-1B config below and failed later on
    // its tokenizer, a message about a file that was never the problem.
    if model.is_dir()
        && !std::fs::read_dir(model).is_ok_and(|d| {
            d.flatten()
                .any(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
        })
    {
        return Err(format!(
            "{} holds no model: looked for model.gguf, a single *.gguf, or config.json with \
             *.safetensors",
            model.display()
        )
        .into());
    }
    Ok(ModelConfig::llama_3_2_1b())
}

/// Collect safetensors shards from a file or directory path.
pub(crate) fn collect_safetensors(path: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    if path.is_dir() {
        let mut shards: Vec<PathBuf> = std::fs::read_dir(path)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        shards.sort();
        if shards.is_empty() {
            return Err(format!("no .safetensors files in {}", path.display()).into());
        }
        return Ok(shards);
    }
    Err(format!("model path not found: {}", path.display()).into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_defaults_to_the_model_directory() {
        let dir = std::env::temp_dir().join(format!("arf_tok_default_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let want = dir.join("tokenizer.json");

        // Nothing there yet: no default, and the error names the path it looked for.
        assert_eq!(sibling_tokenizer(&dir), None);
        let err = tokenizer_hint(&dir);
        assert!(err.contains(&want.display().to_string()), "{err}");

        // A directory holding tokenizer.json: that is the default.
        std::fs::write(&want, "{}").unwrap();
        assert_eq!(sibling_tokenizer(&dir), Some(want.clone()));

        // A single safetensors file: the tokenizer beside it.
        let st = dir.join("model.safetensors");
        std::fs::write(&st, b"not gguf").unwrap();
        assert_eq!(sibling_tokenizer(&st), Some(want));

        // A GGUF: no sibling default — its embedded tokenizer is used instead.
        let gguf = dir.join("model.gguf");
        std::fs::write(&gguf, b"GGUF\x03\0\0\0").unwrap();
        assert_eq!(sibling_tokenizer(&gguf), None);
        assert!(!tokenizer_hint(&gguf).contains("looked for"));

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
