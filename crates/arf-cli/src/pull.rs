//! Download a model's files from a HuggingFace repo over HTTPS.
//!
//! Handles the things real models need: **sharded** weights
//! (`model-00001-of-0000N.safetensors` + a `model.safetensors.index.json`),
//! **sha256 verification** against HuggingFace's content hash, **resume** of a
//! partial download via HTTP range requests, and **retry** with backoff.
//!
//! Pure URL / shard-list / parse logic is unit-tested; the network fetch is thin
//! and behind an `#[ignore]`d integration test (CI is network-free).

use std::error::Error;
use std::fs;
use std::io::{IsTerminal, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use arf_core::config::Quant;
use sha2::{Digest, Sha256};

/// The non-weight files arf needs to run a Llama-arch model.
pub const SUPPORT_FILES: [&str; 2] = ["config.json", "tokenizer.json"];

/// The single-file weights name; sharded repos replace this with N shards
/// listed in [`WEIGHTS_INDEX`].
pub const WEIGHTS_FILE: &str = "model.safetensors";

/// The shard index emitted by sharded repos. Its `weight_map` values are the
/// shard filenames to download.
pub const WEIGHTS_INDEX: &str = "model.safetensors.index.json";

/// Default ungated repo whose config matches `ModelConfig::llama_3_2_1b()`.
pub const DEFAULT_REPO: &str = "unsloth/Llama-3.2-1B-Instruct";

/// The models directory of a source checkout, and where `scripts/get_qwen38.sh` builds by default.
pub const SOURCE_MODELS_DIR: &str = "./models";

/// Where `pull` downloads to and `ls` / `rm` / `doctor` / `run` / `serve` look, unless `--dir` says
/// otherwise: `ARF_MODELS_DIR` when set; else `./models` when it exists (a source checkout, or anyone
/// who already pulled there — nothing moves); else `~/.arf/models`, so an installed `arf` finds its
/// models from any directory. Before this, the default was `./models` relative to wherever `arf` ran,
/// so a Homebrew user who pulled in one directory could not serve from another.
pub fn default_models_dir() -> std::path::PathBuf {
    if let Some(d) = std::env::var_os("ARF_MODELS_DIR").filter(|d| !d.is_empty()) {
        return std::path::PathBuf::from(d);
    }
    let local = Path::new(SOURCE_MODELS_DIR);
    if local.is_dir() {
        return local.to_path_buf();
    }
    match std::env::var_os("HOME").filter(|h| !h.is_empty()) {
        Some(home) => std::path::PathBuf::from(home).join(".arf").join("models"),
        None => local.to_path_buf(),
    }
}

/// Friendly shorthands → ungated HuggingFace repos arf can run today.
/// Anything not in this table is treated as a raw `org/repo`.
const ALIASES: &[(&str, &str)] = &[
    ("llama3.2", "unsloth/Llama-3.2-1B-Instruct"),
    ("llama3.2:1b", "unsloth/Llama-3.2-1B-Instruct"),
    ("llama3.2:3b", "unsloth/Llama-3.2-3B-Instruct"),
    // Gemma 3 (ungated mirrors arf runs today). google/gemma-3-4b-it is gated
    // (HTTP 401 without auth), so we point at the ungated unsloth mirror.
    ("gemma3", "unsloth/gemma-3-4b-it"),
    ("gemma3:4b", "unsloth/gemma-3-4b-it"),
    // NO "gemma4" HERE, ON PURPOSE. It was the pre-launch shorthand for this SAME Gemma 3 4B
    // build (the Gemma 4 QAT GGUFs are not on a public mirror yet) — so `arf pull gemma4`
    // downloaded a model that is not Gemma 4. Retired 2026-09-20, before the public release:
    // see `retired_alias`, which turns it into an error that says what to type instead. Do not
    // re-add it until it can point at a real Gemma 4 build.
    ("gemma2", "unsloth/gemma-2-2b-it"),
    // Qwen3 MoE (the 30B routing target).
    ("qwen3:30b", "Qwen/Qwen3-Coder-30B-A3B-Instruct"),
];

/// Our group-64 Qwen3.8-27B GGUF (`scripts/g64/quantize_g64.py`). PUBLISHED 2026-09-27: public,
/// ungated, sha256-exact against the file the README numbers were measured on; an anonymous
/// `arf pull qwen3.8:27b` served from the pulled folder answers byte-identically to the local
/// bundle (measured 2026-09-27). (Until that day this
/// line was a placeholder for a repo name not yet decided.) `ARF_QWEN38_REPO=<org>/<repo>`
/// overrides it, and a pull that cannot reach it stops before downloading anything else and says
/// how to build the file locally (`scripts/get_qwen38.sh`).
pub const ARF_QWEN38_REPO: &str = "joydle/Qwen3.8-27B-Arf-GGUF";

/// The group-64 model repo: `ARF_QWEN38_REPO` from the environment, else the constant above.
fn qwen38_repo() -> String {
    repo_or_default(std::env::var("ARF_QWEN38_REPO").ok(), ARF_QWEN38_REPO)
}

/// An env override if it is set and not blank, else the default (split out to test without
/// mutating the process environment, which other tests read concurrently).
fn repo_or_default(env: Option<String>, default: &str) -> String {
    env.map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| default.to_string())
}

/// A shorthand that assembles a whole BUNDLE directory (`arf_core::bundle`): the weights plus the
/// parts that make them fast and multimodal, from up to three repos, under one name. `arf serve
/// <alias>` then attaches the parts and the `--arch` / `--quant` the weights need, with no flags.
pub struct BundleSpec {
    /// Shorthands that select it (the first is the one printed).
    pub aliases: &'static [&'static str],
    /// Directory under the models dir.
    pub dir: &'static str,
    /// One line for `arf pull` with no arguments.
    pub what: &'static str,
    /// The repo holding the weights GGUF (a function: it reads an env override).
    pub model_repo: fn() -> String,
    /// The block-draft repo (`config.json` + `model.safetensors`) -> `draft/`.
    pub draft_repo: &'static str,
    /// The repo and file of the vision projector -> `mmproj.gguf`.
    pub mmproj_repo: &'static str,
    pub mmproj_file: &'static str,
    /// Written to `arf-bundle.txt` for arf-serve.
    pub arch: &'static str,
    pub quant: &'static str,
    /// How to build the weights locally when the model repo cannot be reached.
    pub build_script: &'static str,
}

/// The bundles. `qwen3.8:27b` is the main model: our group-64 weights (the file the head-to-head
/// numbers in the README were measured with), the DFlash 2 draft, and unsloth's BF16 projector.
pub const BUNDLES: &[BundleSpec] = &[BundleSpec {
    aliases: &["qwen3.8:27b", "qwen3.8"],
    dir: "qwen3.8-27b-arf",
    what: "Qwen3.8-27B group-64 + DFlash 2 draft + vision (bundle)",
    model_repo: qwen38_repo,
    draft_repo: "incoai/Qwen3.8-27B-DFlash2",
    mmproj_repo: "unsloth/Qwen3.8-27B-GGUF",
    mmproj_file: "mmproj-BF16.gguf",
    arch: "qwen3.8-27b",
    quant: "q4ks",
    build_script: "scripts/get_qwen38.sh",
}];

/// The files a draft directory needs (what `dflash_attach` and `auto_max_context` read).
const DRAFT_FILES: [&str; 2] = ["config.json", "model.safetensors"];

/// The bundle a shorthand names, if any (case-insensitive, `@rev` ignored).
pub fn bundle_for(spec: &str) -> Option<&'static BundleSpec> {
    let head = spec.split('@').next().unwrap_or(spec);
    BUNDLES
        .iter()
        .find(|b| b.aliases.iter().any(|a| a.eq_ignore_ascii_case(head)))
}

/// Is `name` a known shorthand — a plain alias or a bundle?
pub fn is_known_alias(name: &str) -> bool {
    bundle_for(name).is_some() || ALIASES.iter().any(|(k, _)| k.eq_ignore_ascii_case(name))
}

/// Pick the weights GGUF from a bundle's model repo: the one `.gguf` that is not a projector or
/// an imatrix. With several, one whose name says `g64` wins; otherwise the choice is refused and
/// the candidates listed, never guessed.
pub fn pick_bundle_gguf(files: &[String]) -> Result<String, Box<dyn Error>> {
    let weights: Vec<String> = files
        .iter()
        .filter(|f| {
            let l = f.to_ascii_lowercase();
            l.ends_with(".gguf") && !l.starts_with("mmproj") && !l.contains("imatrix")
        })
        .cloned()
        .collect();
    let g64: Vec<&String> = weights
        .iter()
        .filter(|f| f.to_ascii_lowercase().contains("g64"))
        .collect();
    if weights.len() > 1 && g64.len() == 1 {
        return Ok(g64[0].clone());
    }
    pick_gguf(&weights, None)
}

/// The error for a model repo that cannot be reached or holds no weights GGUF: what failed, and
/// the exact commands that build the same file locally from public inputs.
fn bundle_unreachable(spec: &BundleSpec, repo: &str, why: &str, out_dir: &Path) -> String {
    // The script builds into models/<dir> under the repo root; say OUT= only when the pull was
    // pointed somewhere else, so the common case is one copy-pasteable word.
    let default_out = Path::new(SOURCE_MODELS_DIR).join(spec.dir);
    let (cmd, serve) = if out_dir == default_out {
        (spec.build_script.to_string(), spec.aliases[0].to_string())
    } else {
        (
            format!("OUT={} {}", out_dir.display(), spec.build_script),
            out_dir.display().to_string(),
        )
    };
    format!(
        "cannot fetch the {} weights from {repo}: {why}\n\
         \n  That repo may not be published yet. Build the same file locally from unsloth's \
         public\n  files instead (downloads ~51 GB, needs ~64 GB free while converting, \
         Apple silicon + Python\n  with numpy, gguf and mlx), from the repository root:\n\
         \n      {cmd}\n\
         \n  It writes the whole bundle (weights, draft, projector) to {}, then:\n\
         \n      arf serve {serve}\n\
         \n  Or point ARF_QWEN38_REPO at a repo that has the file and run `arf pull {}` again.",
        spec.aliases[0],
        out_dir.display(),
        spec.aliases[0],
    )
}

/// Download one file to `dest` unless it is already there (resumable, sha256-verified).
fn fetch_named(
    repo: &str,
    revision: &str,
    file: &str,
    dest: &Path,
    token: Option<&str>,
    label: &str,
    tui: bool,
) -> Result<bool, Box<dyn Error>> {
    if should_skip(dest) {
        println!("  {label:<34} present");
        return Ok(false);
    }
    fetch_to_file(&resolve_url(repo, revision, file), dest, token, label, tui)?;
    Ok(true)
}

/// Assemble a bundle directory: `model.gguf`, `draft/`, `mmproj.gguf`, `arf-bundle.txt`. Each
/// file is skipped when present, so an interrupted pull resumes and a finished one is a no-op.
/// The weights come FIRST: if their repo cannot be reached the pull stops before spending 4.8 GB
/// on parts that are useless without them. `revision` pins the weights repo; the draft and the
/// projector come from `main`.
pub fn pull_bundle(
    spec: &BundleSpec,
    revision: &str,
    out_dir: &Path,
    token: Option<&str>,
    tui: bool,
    force: bool,
) -> Result<(), Box<dyn Error>> {
    use arf_core::bundle::{Manifest, DRAFT_DIR, MANIFEST_FILE, MMPROJ_FILE, MODEL_FILE};
    let model_repo = (spec.model_repo)();
    println!(
        "pulling bundle {} -> {}",
        spec.aliases[0],
        out_dir.display()
    );
    let mut fetched = 0;

    // 1. the weights. The directory is created only once they are present or on their way: an
    //    empty models/<dir> left by a failed pull would make `arf serve <alias>` find it.
    let model_dest = out_dir.join(MODEL_FILE);
    if should_skip(&model_dest) {
        println!("  {:<34} present", format!("[1/4] {MODEL_FILE}"));
    } else {
        let files = list_repo_files(&model_repo, token)
            .map_err(|e| bundle_unreachable(spec, &model_repo, &e.to_string(), out_dir))?;
        let file = pick_bundle_gguf(&files)
            .map_err(|e| bundle_unreachable(spec, &model_repo, &e.to_string(), out_dir))?;
        // Fit check BEFORE any download: the weights must fit on their own; the draft and the
        // projector count toward the warning only, since `arf serve` can leave them out.
        let weights = remote_sizes(&model_repo, revision, &[file.as_str()], token);
        let draft = remote_sizes(spec.draft_repo, "main", &["model.safetensors"], token);
        let mmproj = remote_sizes(spec.mmproj_repo, "main", &[spec.mmproj_file], token);
        let size = weights
            .zip(draft.zip(mmproj))
            .map(|(w, (d, m))| pull_size(&w, d.iter().chain(&m).map(|(_, b)| b).sum()));
        enforce_fit(spec.aliases[0], size, force)?;
        fs::create_dir_all(out_dir)?;
        println!("  weights: {model_repo}/{file} -> {MODEL_FILE}");
        let label = format!("[1/4] {MODEL_FILE}");
        fetched += fetch_named(
            &model_repo,
            revision,
            &file,
            &model_dest,
            token,
            &label,
            tui,
        )? as usize;
    }

    // 2. the block draft
    let draft_dir = out_dir.join(DRAFT_DIR);
    fs::create_dir_all(&draft_dir)?;
    for f in DRAFT_FILES {
        let label = format!("[2/4] {DRAFT_DIR}/{f}");
        fetched += fetch_named(
            spec.draft_repo,
            "main",
            f,
            &draft_dir.join(f),
            token,
            &label,
            tui,
        )? as usize;
    }

    // 3. the vision projector
    let label = format!("[3/4] {MMPROJ_FILE}");
    fetched += fetch_named(
        spec.mmproj_repo,
        "main",
        spec.mmproj_file,
        &out_dir.join(MMPROJ_FILE),
        token,
        &label,
        tui,
    )? as usize;

    // 4. what arf-serve needs to know about the weights
    let manifest = Manifest {
        arch: Some(spec.arch.to_string()),
        quant: Some(spec.quant.to_string()),
    };
    fs::write(
        out_dir.join(MANIFEST_FILE),
        manifest.render(&format!("`arf pull {}`", spec.aliases[0])),
    )?;
    println!("  {:<34} written", format!("[4/4] {MANIFEST_FILE}"));

    if fetched == 0 {
        println!("up to date -> {}", out_dir.display());
    } else {
        println!("done ({fetched} fetched) -> {}", out_dir.display());
    }
    Ok(())
}

/// A resolved pull target.
pub struct Resolved {
    /// The HuggingFace `org/repo` to fetch from.
    pub repo: String,
    /// A revision pinned in the spec (`@rev`), if any.
    pub revision: Option<String>,
    /// A quant tag (`:Q4_K_M`) selecting a GGUF file, if any. Only set for raw
    /// `org/repo` specs — never for a known alias (whose `:` is part of its key).
    pub quant: Option<String>,
    /// A filesystem-safe directory slug to download into by default.
    pub out_slug: String,
    /// Set when the spec names a bundle: `repo` is then its weights repo and `out_slug` its dir.
    pub bundle: Option<&'static BundleSpec>,
}

/// Resolve a model spec into a repo + optional revision + quant + output slug.
///
/// Accepts a known shorthand (`llama3.2:1b`), a raw repo (`org/model`), a pinned
/// repo (`org/model@<revision>`), an ollama-style `hf.co/org/model` ref, and a
/// GGUF quant selector (`org/model:Q4_K_M`).
///
/// Parse order is deliberate (the `:` is overloaded between alias keys like
/// `qwen3:30b` and quant tags like `repo:Q4_K_M`):
/// 1. strip a `hf.co/` / `huggingface.co/` prefix;
/// 2. split off a trailing `@<revision>`;
/// 3. look the WHOLE remaining head up in the alias table — if it IS an alias, the
///    `:` belongs to the alias key and NO quant is extracted;
/// 4. only when the head is NOT an alias may a trailing `:TAG` on the last path
///    segment be peeled off as a GGUF quant selector.
///
/// This keeps `llama3.2:1b` / `qwen3:30b` intact while enabling `:Q4_K_M` on raw
/// repos. The quant (and prefix) are stripped before the slug, so `out_slug` stays
/// `repo`-shaped (not `repo-q4_k_m`) and matches how `arf run <slug>` refers to it.
pub fn resolve_model(spec: &str) -> Resolved {
    // 1. strip the ollama-style HuggingFace prefix.
    let spec = spec
        .strip_prefix("hf.co/")
        .or_else(|| spec.strip_prefix("huggingface.co/"))
        .unwrap_or(spec);

    // 2. split off `@revision`.
    let (head, mut revision) = match spec.split_once('@') {
        Some((h, r)) if !r.is_empty() => (h, Some(r.to_string())),
        _ => (spec, None),
    };

    // 3. full-head alias lookup FIRST (so an alias's `:` is never read as a quant). A bundle
    //    shorthand resolves to its weights repo and its OWN directory name (not the repo's slug:
    //    the directory holds three repos' files).
    if let Some(b) = bundle_for(head) {
        return Resolved {
            repo: (b.model_repo)(),
            revision,
            quant: None,
            out_slug: b.dir.to_string(),
            bundle: Some(b),
        };
    }
    if let Some((_, repo)) = ALIASES.iter().find(|(k, _)| k.eq_ignore_ascii_case(head)) {
        return Resolved {
            repo: (*repo).to_string(),
            revision,
            quant: None,
            out_slug: slug_of(repo),
            bundle: None,
        };
    }

    // 4. not an alias → a trailing `:TAG` is a GGUF quant. It may sit on the repo
    //    segment (`org/repo:Q4_K_M`, the ollama form) OR after `@rev`
    //    (`org/repo@main:Q4_K_M`). Only look for `:` AFTER the last `/` of the repo,
    //    so an `org` name is never mistaken for a quant.
    let mut quant: Option<String> = None;
    let repo = match head.rfind('/') {
        Some(slash) => match head[slash..].split_once(':') {
            Some((seg, tag)) if !tag.is_empty() => {
                quant = Some(tag.to_string());
                format!("{}{seg}", &head[..slash])
            }
            _ => head.to_string(),
        },
        None => head.to_string(),
    };
    // Quant trailing the revision (`@main:Q4_K_M`): peel it off the revision when the
    // repo segment didn't already carry one.
    if quant.is_none() {
        if let Some(rev) = revision.take() {
            match rev.split_once(':') {
                Some((r, tag)) if !tag.is_empty() => {
                    revision = Some(r.to_string());
                    quant = Some(tag.to_string());
                }
                _ => revision = Some(rev),
            }
        }
    }

    let out_slug = slug_of(&repo);
    Resolved {
        repo,
        revision,
        quant,
        out_slug,
        bundle: None,
    }
}

/// A filesystem-safe slug: last path segment, lowercased, `:` flattened to `-`.
fn slug_of(repo: &str) -> String {
    repo.rsplit('/')
        .next()
        .unwrap_or(repo)
        .replace(':', "-")
        .to_ascii_lowercase()
}

/// Pick a single `.gguf` file from a repo's file list, given an optional quant tag.
///
/// Rules:
/// - no quant + exactly one `.gguf` → take it;
/// - no quant + several `.gguf` → `Err` listing them (the user must pick a quant);
/// - quant matching exactly one `.gguf` (case-insensitive substring) → take it;
/// - quant matching zero or more than one → `Err` listing the available `.gguf`s.
///
/// Never silently grabs the first of several matches.
pub fn pick_gguf(files: &[String], quant: Option<&str>) -> Result<String, Box<dyn Error>> {
    let ggufs: Vec<&String> = files
        .iter()
        .filter(|f| f.to_ascii_lowercase().ends_with(".gguf"))
        .collect();
    if ggufs.is_empty() {
        return Err("repo has no .gguf files (and no safetensors weights)".into());
    }
    let available = || {
        ggufs
            .iter()
            .map(|f| f.as_str())
            .collect::<Vec<_>>()
            .join("\n  ")
    };
    match quant {
        None => {
            if ggufs.len() == 1 {
                Ok(ggufs[0].clone())
            } else {
                Err(format!(
                    "multiple .gguf files — pick a quant with `:TAG`, e.g. `:Q4_K_M`. available:\n  {}",
                    available()
                )
                .into())
            }
        }
        Some(tag) => {
            let tag_l = tag.to_ascii_lowercase();
            let hits: Vec<&&String> = ggufs
                .iter()
                .filter(|f| f.to_ascii_lowercase().contains(&tag_l))
                .collect();
            match hits.len() {
                1 => Ok((*hits[0]).clone()),
                0 => Err(format!(
                    "no .gguf matches quant `{tag}`. available:\n  {}",
                    available()
                )
                .into()),
                _ => Err(format!(
                    "quant `{tag}` is ambiguous (matches {} files) — be more specific. available:\n  {}",
                    hits.len(),
                    available()
                )
                .into()),
            }
        }
    }
}

/// All known shorthands → repos (for discovery / suggestions).
pub fn known_aliases() -> &'static [(&'static str, &'static str)] {
    ALIASES
}

/// Levenshtein edit distance between two byte strings (small inputs).
fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        curr[0] = i;
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

/// The closest known alias to `input`, if it's within a small edit distance
/// (≤2, or ≤ a third of the input length — whichever is larger). Case-insensitive.
/// Returns None when nothing is close enough (likely a real org/repo, not a typo).
/// Drives the serve/run/pull "did you mean?" hints.
/// A shorthand that used to exist and was withdrawn because it was misleading. Returns what
/// to tell the user, or `None` for anything else. Callers must treat `Some` as an ERROR, not a
/// hint: falling through would send the bare name to HuggingFace as a repo and fail with a 404
/// that explains nothing.
pub fn retired_alias(input: &str) -> Option<&'static str> {
    let head = input
        .split('@')
        .next()
        .unwrap_or(input)
        .to_ascii_lowercase();
    if head == "gemma4" || head.starts_with("gemma4:") {
        return Some(
            "'gemma4' is not a shorthand: it used to download the Gemma 3 4B build, which is not \
Gemma 4, and no Gemma 4 GGUF is on a public mirror yet.\n  \
→ for that Gemma 3 build:        arf pull gemma3:4b\n  \
→ for a Gemma 4 GGUF you have:   arf pull <org>/<repo>:<QUANT>",
        );
    }
    None
}

pub fn suggest_alias(input: &str) -> Option<&'static str> {
    let input_l = input.to_ascii_lowercase();
    let threshold = 2.max(input_l.chars().count() / 3);
    let mut best: Option<(&'static str, usize)> = None;
    let bundle_aliases = BUNDLES.iter().flat_map(|b| b.aliases.iter().copied());
    for alias in bundle_aliases.chain(ALIASES.iter().map(|(a, _)| *a)) {
        let d = edit_distance(&input_l, &alias.to_ascii_lowercase());
        if d <= threshold && best.map(|(_, bd)| d < bd).unwrap_or(true) {
            best = Some((alias, d));
        }
    }
    best.map(|(a, _)| a)
}

/// Per-file transient-failure retries (TLS timeouts, unexpected EOF).
const MAX_RETRIES: u32 = 3;

/// HuggingFace `resolve/{revision}` URL for a file in a repo.
fn resolve_url(repo: &str, revision: &str, file: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/{revision}/{file}")
}

/// List a repo's file names via the HuggingFace API (`siblings[].rfilename`).
///
/// Bounded (explicit connect + overall timeouts, can't hang) AND honest: a
/// transport/timeout failure is `Err` ("could not reach HuggingFace…"), distinct
/// from a successful-but-empty list — so the caller can tell a network problem from
/// "this repo has no such files" (the network-down ambiguity).
/// Unlike `hf_search::search`, this does NOT swallow errors to an empty Vec.
fn list_repo_files(repo: &str, token: Option<&str>) -> Result<Vec<String>, Box<dyn Error>> {
    use std::time::Duration;
    let url = format!("https://huggingface.co/api/models/{repo}");
    let agent = ureq::builder()
        .timeout_connect(Duration::from_secs(5))
        .timeout(Duration::from_secs(10))
        .build();
    let mut req = agent.get(&url);
    if let Some(t) = token {
        req = req.set("Authorization", &format!("Bearer {t}"));
    }
    let body = match req.call() {
        Ok(resp) => resp
            .into_string()
            .map_err(|e| format!("could not read HuggingFace response for {repo}: {e}"))?,
        Err(ureq::Error::Status(404, _)) => {
            return Err(format!("repo {repo} not found on HuggingFace (404)").into())
        }
        Err(ureq::Error::Status(401, _)) | Err(ureq::Error::Status(403, _)) => {
            return Err(format!(
                "access to {repo} denied (gated/private?) — pass --token or set HF_TOKEN"
            )
            .into())
        }
        Err(e) => {
            return Err(format!("could not reach HuggingFace to list files for {repo}: {e}").into())
        }
    };
    Ok(extract_rfilenames(&body))
}

/// Pull `"rfilename":"<name>"` values out of the HF model API JSON. Dependency-free
/// byte scan (same spirit as `parse_shard_list` / `hf_search`).
fn extract_rfilenames(json: &str) -> Vec<String> {
    let mut out = Vec::new();
    let pat = "\"rfilename\"";
    let mut rest = json;
    while let Some(pos) = rest.find(pat) {
        rest = &rest[pos + pat.len()..];
        // skip spaces + the colon
        rest = rest.trim_start();
        rest = rest.strip_prefix(':').unwrap_or(rest).trim_start();
        if let Some(after_q) = rest.strip_prefix('"') {
            if let Some(end) = after_q.find('"') {
                out.push(after_q[..end].to_string());
                rest = &after_q[end + 1..];
            }
        }
    }
    out
}

/// Every file's size at `revision`, from ONE call to HuggingFace's tree API
/// (`/api/models/{repo}/tree/{revision}?recursive=true`, entries `{"type","oid","size","path",…}`),
/// so a 118-shard repo is sized without 118 requests and without downloading anything. For an LFS
/// file (every weight file) the top-level `size` is the file's real size — the same number as its
/// `lfs.size` and as the `X-Linked-Size` header on its `resolve/` redirect — not the pointer's.
fn repo_file_sizes(
    repo: &str,
    revision: &str,
    token: Option<&str>,
) -> Result<Vec<(String, u64)>, Box<dyn Error>> {
    let url = format!("https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true");
    let agent = ureq::builder()
        .timeout_connect(Duration::from_secs(5))
        .timeout(Duration::from_secs(15))
        .build();
    let mut req = agent.get(&url);
    if let Some(t) = token {
        req = req.set("Authorization", &format!("Bearer {t}"));
    }
    Ok(parse_tree_sizes(&req.call()?.into_string()?))
}

/// `(path, size)` for each entry of a tree-API response. Dependency-free scan, like
/// [`extract_rfilenames`]: each entry opens with `{"type":`, and the nested `lfs` object (which
/// carries its own, equal, `size`) has no `type` key, so splitting on it yields one entry per file.
fn parse_tree_sizes(json: &str) -> Vec<(String, u64)> {
    json.split("{\"type\":")
        .skip(1)
        .filter_map(|entry| {
            let size = entry.split("\"size\":").nth(1)?;
            let size: u64 = size
                .trim_start()
                .split(|c: char| !c.is_ascii_digit())
                .next()?
                .parse()
                .ok()?;
            let path = entry
                .split("\"path\":")
                .nth(1)?
                .trim_start()
                .strip_prefix('"')?;
            Some((path[..path.find('"')?].to_string(), size))
        })
        .collect()
}

/// The size of each of `files`, paired with its name, from [`repo_file_sizes`]. `None` if the
/// listing fails or lacks any one of them (a partial list would understate the model and pass a
/// check it should fail) — the caller then skips the fit check rather than block the pull.
fn remote_sizes<'a>(
    repo: &str,
    revision: &str,
    files: &[&'a str],
    token: Option<&str>,
) -> Option<Vec<(&'a str, u64)>> {
    let tree = repo_file_sizes(repo, revision, token).ok()?;
    files
        .iter()
        .map(|f| tree.iter().find(|(p, _)| p == f).map(|(_, n)| (*f, *n)))
        .collect()
}

/// A weight file, as opposed to the config / tokenizer / index JSON that rides along with it.
fn is_weight_file(name: &str) -> bool {
    let l = name.to_ascii_lowercase();
    l.ends_with(".safetensors") || l.ends_with(".gguf")
}

/// The fraction of RAM above which `arf pull` warns that a model will be tight. Derived from the
/// load guard's cap (`arf_gpu::weights::MEM_CAP_FRACTION`, 0.90) so the two cannot drift: 0.15 of
/// RAM below it, i.e. 0.75. The 0.15 is a judgement, not a measurement — it is room for the KV pool,
/// the bf16 embedding table and the rest of the machine, which the weights' file size does not
/// count — and the result is only ever a warning.
pub const PULL_WARN_FRACTION: f64 = arf_gpu::weights::MEM_CAP_FRACTION - 0.15;

/// The smallest model `arf pull` knows, suggested when a model does not fit. Its weights are
/// 2,471,645,608 bytes (HuggingFace `X-Linked-Size` for `unsloth/Llama-3.2-1B-Instruct`
/// `model.safetensors`).
const SMALL_MODEL_HINT: &str = "llama3.2:1b";

/// The `--quant` the load guard prices smallest, and its bytes per parameter. Every variant is
/// listed so the refusal below is never stricter than some quant the loader accepts.
fn smallest_quant() -> (Quant, f64) {
    [
        Quant::None,
        Quant::Int8,
        Quant::Q4,
        Quant::Q4K,
        Quant::Q4KS,
        Quant::Q3K,
    ]
    .into_iter()
    .map(|q| (q, q.approx_bytes_per_param()))
    .min_by(|a, b| a.1.total_cmp(&b.1))
    .expect("non-empty")
}

/// What a pull is about to fetch, in the terms the fit check needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PullSize {
    /// Bytes of the weight files, as HuggingFace reports them (= bytes on disk).
    pub weights: u64,
    /// The smallest the load guard can price those weights at any `--quant`: a safetensors file
    /// is re-quantized at load (the guard prices it as bytes ÷ 2.0 × the target quant's bytes per
    /// parameter), so its floor is the smallest quant's; a GGUF is counted at its file size.
    pub min_resident: u64,
    /// Bytes `arf serve` loads by default on top of the weights — a bundle's draft and vision
    /// projector, which `--no-draft` / `--no-mmproj` leave out. 0 for a plain repo.
    pub extras: u64,
}

/// Price `(file, bytes)` weight files the way the load guard does (see [`PullSize`]).
pub fn pull_size(weight_files: &[(&str, u64)], extras: u64) -> PullSize {
    let (_, min_bpp) = smallest_quant();
    let weights = weight_files.iter().map(|(_, b)| b).sum();
    let min_resident = weight_files
        .iter()
        .map(|(f, b)| {
            if f.to_ascii_lowercase().ends_with(".safetensors") {
                // The guard's own pricing for safetensors: bf16 (2.0 bytes/param) re-quantized.
                (*b as f64 / Quant::None.approx_bytes_per_param() * min_bpp) as u64
            } else {
                *b
            }
        })
        .sum();
    PullSize {
        weights,
        min_resident,
        extras,
    }
}

/// Whether a model about to be pulled fits this machine — decided BEFORE any weight byte is
/// downloaded (issue #3). Before this, `arf pull qwen3.8:27b` on a 16 GB Mac downloaded ~17 GB and only
/// `arf serve` refused it, at load.
///
/// * REFUSED when even `size.min_resident` exceeds `MEM_CAP_FRACTION` of RAM: the load guard
///   refuses that before it adds the KV pool, so the download could never run here. `force` turns
///   the refusal into a warning.
/// * WARNED when what `arf serve` loads by default (`weights + extras`) exceeds
///   `PULL_WARN_FRACTION` of RAM — naming the levers that apply (a smaller `--quant`, or
///   `--no-draft` / `--no-mmproj`).
///
/// ⚠️ File bytes are a LOWER BOUND on resident memory at the precision they are loaded at, and no
/// multiplier is applied to them here because none has been measured for this purpose: the load
/// guard adds the KV pool and the bf16 embedding table on top (`wired_memory_guard`). The one way
/// weights get SMALLER than their file is a lower `--quant` at load, which [`pull_size`] prices
/// for safetensors the way the guard does. A GGUF is taken at its file size; re-quantizing one
/// down at load is the case `--force` is for.
///
/// 🔴 The refusal must not be stricter than the load guard. Its first version compared the bf16
/// file size and refused `arf pull qwen3:30b` (61,066,575,656 bytes of bf16 safetensors) on a
/// 36 GB box — a shipped alias that loads there at `--quant q4ks`.
///
/// `Ok(None)` fits, `Ok(Some(warning))` proceeds with a warning, `Err(refusal)` stops the pull.
pub fn fit_check(
    model: &str,
    size: PullSize,
    ram: u64,
    force: bool,
) -> Result<Option<String>, String> {
    let gb = |b: u64| b as f64 / 2f64.powi(30);
    let cap_frac = arf_gpu::weights::MEM_CAP_FRACTION;
    let cap = (ram as f64 * cap_frac) as u64;
    let (min_quant, _) = smallest_quant();
    let PullSize {
        weights,
        min_resident,
        extras,
    } = size;
    if min_resident > cap {
        let need = if min_resident < weights {
            format!(
                "at least {min_resident} bytes ({:.1} GB) for its weights alone, even at --quant \
                 {} (the files are {weights} bytes)",
                gb(min_resident),
                min_quant.as_str(),
            )
        } else {
            format!(
                "{weights} bytes ({:.1} GB) for its weights alone",
                gb(weights)
            )
        };
        let facts = format!(
            "{model} needs {need}; this Mac has {ram} bytes ({:.1} GB) of RAM, and arf will not \
             load more than {:.0}% of it ({cap} bytes, {:.1} GB) — before the KV cache is even \
             counted.",
            gb(ram),
            cap_frac * 100.0,
            gb(cap),
        );
        if force {
            return Ok(Some(format!(
                "⚠ --force: downloading anyway. {facts} `arf serve` / `arf run` will refuse to \
                 load it on this machine (ARF_FORCE_LOAD=1 overrides at your own risk)."
            )));
        }
        return Err(format!(
            "refusing to download: {facts}\n\
             \n  The download would finish and then `arf serve` would refuse to load it.\n\
             \n  A smaller model that fits:  arf pull {SMALL_MODEL_HINT}\
             \n  Download anyway (e.g. for another machine):  arf pull {model} --force"
        ));
    }
    let total = weights + extras;
    let warn_at = (ram as f64 * PULL_WARN_FRACTION) as u64;
    if total <= warn_at {
        return Ok(None);
    }
    let mut levers = Vec::new();
    if weights > cap {
        levers.push(format!(
            "at the precision it is stored in, its {weights} bytes of weights are over arf's \
             {cap}-byte load cap ({:.0}% of RAM), so it loads only with a smaller --quant (the \
             smallest, --quant {}, is priced at ~{min_resident} bytes; the KV cache comes on top)",
            cap_frac * 100.0,
            min_quant.as_str(),
        ));
    }
    if extras > 0 {
        levers.push(format!(
            "{extras} of those bytes are the draft and vision projector `arf serve` attaches by \
             default; --no-draft / --no-mmproj leave them out"
        ));
    }
    let levers = if levers.is_empty() {
        String::new()
    } else {
        format!(" Note: {}.", levers.join("; "))
    };
    Ok(Some(format!(
        "⚠ {model} is {total} bytes ({:.1} GB), over {:.0}% of this Mac's {ram} bytes ({:.1} GB) \
         of RAM. It may load, but leaves little room: the KV cache comes on top, and other apps \
         will page while it runs.{levers} A smaller model: arf pull {SMALL_MODEL_HINT}",
        gb(total),
        PULL_WARN_FRACTION * 100.0,
        gb(ram),
    )))
}

/// Run [`fit_check`] against this machine and print its warning. A file that cannot be sized (a
/// header missing, the network flaky) or an unreadable RAM size skips the check with a note: it
/// guards a download, and must not be the reason a pull fails.
fn enforce_fit(model: &str, size: Option<PullSize>, force: bool) -> Result<(), Box<dyn Error>> {
    let Some(ram) = arf_gpu::weights::physical_ram_bytes() else {
        return Ok(()); // the load guard is macOS-only too; nothing to compare against
    };
    let Some(size) = size else {
        eprintln!(
            "note: could not read the weights' size from HuggingFace; skipping the memory fit check"
        );
        return Ok(());
    };
    if let Some(warning) = fit_check(model, size, ram, force)? {
        eprintln!("{warning}");
    }
    Ok(())
}

/// Whether `path` already exists and is non-empty (so we can skip it).
fn should_skip(path: &Path) -> bool {
    fs::metadata(path).map(|m| m.len() > 0).unwrap_or(false)
}

/// Every file a safetensors model needs is already in `dir`: the support files and the weights —
/// the single file, or an index whose every shard is present. `Some(file count)`, else `None`.
fn local_complete(dir: &Path) -> Option<usize> {
    if !SUPPORT_FILES.iter().all(|f| should_skip(&dir.join(f))) {
        return None;
    }
    if should_skip(&dir.join(WEIGHTS_FILE)) {
        return Some(SUPPORT_FILES.len() + 1);
    }
    let index = fs::read_to_string(dir.join(WEIGHTS_INDEX)).ok()?;
    let shards = parse_shard_list(&index).ok()?;
    shards
        .iter()
        .all(|s| should_skip(&dir.join(s)))
        .then(|| SUPPORT_FILES.len() + 1 + shards.len())
}

/// Download every needed file for `repo` into `out_dir`, skipping ones already
/// present. `token` (optional) authenticates gated repos. When `only` is `Some`,
/// fetch EXACTLY those files (a specific GGUF, the tokenizer, …) instead of
/// auto-resolving the safetensors weights + support files — for repos that ship a
/// named single-file artifact (e.g. a QAT GGUF) rather than `model.safetensors`.
pub fn pull(
    repo: &str,
    revision: &str,
    out_dir: &Path,
    token: Option<&str>,
    only: Option<&[String]>,
    quant: Option<&str>,
    tui: bool,
    force: bool,
) -> Result<(), Box<dyn Error>> {
    let created = !out_dir.exists();
    fs::create_dir_all(out_dir)?;
    println!("pulling {repo} @ {revision}");

    // Already complete on disk (support files + the whole weight set): done, with NO network call.
    // 2026-09-27: `make quickstart` failed on a machine whose model was fully downloaded, because
    // resolving the weight set asked Hugging Face first (a HEAD) and a DNS failure aborted the pull.
    if only.is_none() {
        if let Some(n) = local_complete(out_dir) {
            println!(
                "up to date ({n} files on disk, not re-checked online) -> {}",
                out_dir.display()
            );
            return Ok(());
        }
    }

    // Explicit file list, or auto-resolve.
    let owned_files: Vec<String>;
    let files: Vec<&str> = if let Some(list) = only {
        list.iter().map(String::as_str).collect()
    } else {
        // Try safetensors (+ support files). If the repo has no safetensors at all,
        // fall back to GGUF discovery: list the repo, pick the .gguf by quant, and
        // fetch ONLY that file — GGUF repos have no config.json/tokenizer.json.
        match resolve_weight_files(repo, revision, out_dir, token) {
            Ok(weights) => {
                owned_files = SUPPORT_FILES
                    .iter()
                    .map(|s| s.to_string())
                    .chain(weights)
                    .collect();
                owned_files.iter().map(String::as_str).collect()
            }
            Err(safetensors_err) => {
                let all = list_repo_files(repo, token).map_err(|list_err| {
                    // Surface BOTH the safetensors miss and the list failure honestly.
                    format!("{safetensors_err}; also {list_err}")
                })?;
                let gguf = pick_gguf(&all, quant)?;
                println!("  gguf: {gguf}");
                owned_files = vec![gguf];
                owned_files.iter().map(String::as_str).collect()
            }
        }
    };

    // Fit check BEFORE any weight download (the index JSON above is the only thing fetched so far).
    // Sized whether or not a file is already on disk: the question is whether the MODEL fits.
    let weight_files: Vec<&str> = files
        .iter()
        .copied()
        .filter(|f| is_weight_file(f))
        .collect();
    if !weight_files.is_empty() && !weight_files.iter().all(|f| should_skip(&out_dir.join(f))) {
        let size = remote_sizes(repo, revision, &weight_files, token).map(|w| pull_size(&w, 0));
        // Name it the way the user can paste it back with --force (quant selector, pinned rev).
        let mut name = repo.to_string();
        if let Some(q) = quant {
            name = format!("{name}:{q}");
        }
        if revision != "main" {
            name = format!("{name}@{revision}");
        }
        if let Err(refusal) = enforce_fit(&name, size, force) {
            // Leave nothing behind for a pull that never started: the directory this call made
            // holds at most the shard index fetched above.
            if created {
                let _ = fs::remove_dir_all(out_dir);
            }
            return Err(refusal);
        }
    }

    let n = files.len();
    let mut fetched = 0;
    for (i, file) in files.iter().enumerate() {
        let dest = out_dir.join(file);
        let tag = format!("[{}/{n}] {file}", i + 1);
        if should_skip(&dest) {
            println!("  {tag:<34} present");
            continue;
        }
        fetch_to_file(&resolve_url(repo, revision, file), &dest, token, &tag, tui)?;
        fetched += 1;
    }

    if fetched == 0 {
        println!("up to date -> {}", out_dir.display());
    } else {
        println!("done ({fetched} fetched) -> {}", out_dir.display());
    }
    Ok(())
}

/// Decide which weight file(s) to fetch. Tries the single-file name; if the repo
/// is sharded (single file missing, index present) downloads + parses the index
/// and returns the shard list. The index itself is saved alongside the shards.
fn resolve_weight_files(
    repo: &str,
    revision: &str,
    out_dir: &Path,
    token: Option<&str>,
) -> Result<Vec<String>, Box<dyn Error>> {
    // A single-file model: the common 1B/3B case. Probe with a HEAD.
    if url_exists(&resolve_url(repo, revision, WEIGHTS_FILE), token)? {
        return Ok(vec![WEIGHTS_FILE.to_string()]);
    }

    // Sharded: fetch the index, parse its shard list.
    let index_url = resolve_url(repo, revision, WEIGHTS_INDEX);
    if !url_exists(&index_url, token)? {
        return Err(
            format!("{repo}@{revision}: neither {WEIGHTS_FILE} nor {WEIGHTS_INDEX} found").into(),
        );
    }
    let index_dest = out_dir.join(WEIGHTS_INDEX);
    if !should_skip(&index_dest) {
        // The index is a small JSON file — always use the plain printer for it.
        fetch_to_file(&index_url, &index_dest, token, WEIGHTS_INDEX, false)?;
    }
    let index_json = fs::read_to_string(&index_dest)?;
    let shards = parse_shard_list(&index_json)?;
    println!("  sharded model: {} shard(s)", shards.len());
    Ok(shards)
}

/// Extract the unique, sorted shard filenames from a `model.safetensors.index.json`.
///
/// The index's `weight_map` maps tensor names to shard filenames like
/// `model-00001-of-00004.safetensors`; we only need the distinct values. Parsed
/// with a small dependency-free scan (no JSON crate) — we look for the distinct
/// `*.safetensors` string values in the file.
fn parse_shard_list(index_json: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let mut shards: Vec<String> = Vec::new();
    // Walk every double-quoted string; keep the ones ending in `.safetensors`
    // that look like a filename (no path separators, no spaces).
    let bytes = index_json.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'"' {
            // find the closing quote (filenames have no escapes)
            if let Some(rel) = bytes[i + 1..].iter().position(|&b| b == b'"') {
                let s = &index_json[i + 1..i + 1 + rel];
                if s.ends_with(".safetensors")
                    && !s.contains('/')
                    && !s.contains('\\')
                    && !s.contains(char::is_whitespace)
                    && !shards.iter().any(|e| e == s)
                {
                    shards.push(s.to_string());
                }
                i += rel + 2;
                continue;
            }
        }
        i += 1;
    }
    shards.sort();
    if shards.is_empty() {
        return Err("no shard filenames found in index json".into());
    }
    Ok(shards)
}

/// HEAD a URL to see if it exists (200) without downloading it. Treats 404 as
/// "no" and other 4xx/5xx as a hard error (so auth problems still surface).
fn url_exists(url: &str, token: Option<&str>) -> Result<bool, Box<dyn Error>> {
    let mut req = ureq::head(url);
    if let Some(t) = token {
        req = req.set("Authorization", &format!("Bearer {t}"));
    }
    match req.call() {
        Ok(_) => Ok(true),
        Err(ureq::Error::Status(404, _)) => Ok(false),
        Err(ureq::Error::Status(401, _)) | Err(ureq::Error::Status(403, _)) => {
            Err("access denied (gated repo?). Pass --token <hf_token> for gated models.".into())
        }
        Err(e) => Err(format!("{url}: {e}").into()),
    }
}

/// Pull a 64-hex content sha256 out of an `X-Linked-Etag` header value.
///
/// **Only `X-Linked-Etag` is the content sha256.** The plain `ETag` is the LFS
/// *pointer* OID — also a 64-hex value, but a hash of different bytes, so using
/// it would fail verification on a correct file. A git-stored (non-LFS) file's
/// `X-Linked-Etag` is a 40-char sha1, which we reject here (we only verify the
/// big LFS weights).
fn parse_linked_etag(raw: &str) -> Option<String> {
    let hex = raw.trim_matches('"');
    (hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit())).then(|| hex.to_lowercase())
}

/// Capture the content sha256 HuggingFace advertises for a file, via a HEAD with
/// **redirects disabled** — `X-Linked-Etag` lives on the 302/307 to the CDN, not
/// on the final CDN response (which only carries the LFS-pointer `ETag`). Returns
/// `None` (skip verification) for non-LFS files or if the header is absent.
fn head_content_sha256(url: &str, token: Option<&str>) -> Option<String> {
    let agent = ureq::builder().redirects(0).build();
    let mut req = agent.head(url);
    if let Some(t) = token {
        req = req.set("Authorization", &format!("Bearer {t}"));
    }
    // A 3xx is returned as Err(Status) when redirects are disabled; the header is
    // on that response either way, so inspect both arms.
    let resp = match req.call() {
        Ok(r) => r,
        Err(ureq::Error::Status(_, r)) => r,
        Err(_) => return None,
    };
    resp.header("X-Linked-Etag").and_then(parse_linked_etag)
}

/// Resolve a HuggingFace `resolve/` URL to its final (post-redirect) CDN URL.
///
/// HF answers `resolve/<file>` with a 302/307 to a CDN host. `ureq` follows the
/// redirect automatically, BUT it strips the `Range` (and `Authorization`) header
/// across the cross-host hop — so a resume request lands at the CDN with NO range
/// and the server returns the full body (200) from offset 0, defeating the resume
/// (and, in practice, hanging). We follow the redirect OURSELVES (redirects(0),
/// read `Location`) and then issue the ranged GET directly against the final URL,
/// where the `Range` header survives. Returns the original URL unchanged if there
/// is no redirect (already a direct URL).
fn resolve_redirect(url: &str, token: Option<&str>) -> Result<String, Box<dyn Error>> {
    // HEAD (not GET) so resolving a non-redirecting URL doesn't pull the whole body.
    let agent = ureq::builder().redirects(0).build();
    let mut req = agent.head(url);
    if let Some(t) = token {
        req = req.set("Authorization", &format!("Bearer {t}"));
    }
    // With redirects disabled, a 3xx comes back as Err(Status) carrying the response.
    match req.call() {
        // No redirect (200 directly): the URL is already final.
        Ok(_) => Ok(url.to_string()),
        Err(ureq::Error::Status(301..=308, r)) => match r.header("Location") {
            Some(loc) => Ok(loc.to_string()),
            None => Ok(url.to_string()),
        },
        Err(ureq::Error::Status(401, _)) | Err(ureq::Error::Status(403, _)) => {
            Err("access denied (gated repo?). Set HF_TOKEN or pass --token.".into())
        }
        Err(e) => Err(format!("{url}: {e}").into()),
    }
}

/// Download `url` to `dest` with resume, retry, progress, and sha256 verify.
fn fetch_to_file(
    url: &str,
    dest: &Path,
    token: Option<&str>,
    label: &str,
    tui: bool,
) -> Result<(), Box<dyn Error>> {
    // Capture the expected content hash once, up front (before any redirect).
    let expected = head_content_sha256(url, token);
    let mut attempt = 0;
    loop {
        match fetch_once(url, dest, token, label, expected.as_deref(), tui) {
            Ok(()) => return Ok(()),
            Err(e) if attempt + 1 < MAX_RETRIES => {
                attempt += 1;
                let backoff = Duration::from_millis(500 * (1 << attempt));
                eprintln!(
                    "\n  {label}: {e} — retry {attempt}/{} in {backoff:?}",
                    MAX_RETRIES - 1
                );
                std::thread::sleep(backoff);
            }
            Err(e) => {
                return Err(format!("{label}: {e} (gave up after {MAX_RETRIES} tries)").into())
            }
        }
    }
}

/// One download attempt. Resumes from an existing `.part` via a range request,
/// streams to the part file while hashing, then verifies and atomically renames.
fn fetch_once(
    url: &str,
    dest: &Path,
    token: Option<&str>,
    label: &str,
    expected: Option<&str>,
    tui: bool,
) -> Result<(), Box<dyn Error>> {
    let tmp: PathBuf = dest.with_extension("part");
    let resume_from = fs::metadata(&tmp).map(|m| m.len()).unwrap_or(0);

    // Resolve the HF redirect to the final CDN URL FIRST, then range-request it
    // directly — otherwise ureq strips the `Range` header on the cross-host hop and
    // the resume silently restarts from 0 (or hangs). See `resolve_redirect`.
    let final_url = resolve_redirect(url, token)?;

    let mut req = ureq::get(&final_url);
    if let Some(t) = token {
        req = req.set("Authorization", &format!("Bearer {t}"));
    }
    if resume_from > 0 {
        req = req.set("Range", &format!("bytes={resume_from}-"));
    }

    let resp = req.call().map_err(|e| -> Box<dyn Error> {
        match e {
            ureq::Error::Status(401, _) | ureq::Error::Status(403, _) => {
                "access denied (gated repo?). Pass --token <hf_token> for gated models.".into()
            }
            other => format!("{other}").into(),
        }
    })?;

    // 206 = server honored the range (resume); 200 = full body (range ignored,
    // or no resume requested). On a plain 200 after a partial, start over.
    let resuming = resp.status() == 206 && resume_from > 0;
    let already = if resuming { resume_from } else { 0 };

    // Total file size: Content-Length is the *remaining* bytes on a 206.
    let remaining: u64 = resp
        .header("Content-Length")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let total = if remaining > 0 {
        already + remaining
    } else {
        0
    };

    // Open the part file at the right offset and seed the hasher with any bytes
    // we already have on disk (so a resumed download still verifies end-to-end).
    let mut hasher = Sha256::new();
    let mut file = if resuming {
        let mut f = fs::OpenOptions::new().read(true).write(true).open(&tmp)?;
        let mut existing = Vec::new();
        f.read_to_end(&mut existing)?;
        hasher.update(&existing);
        f.seek(SeekFrom::Start(already))?;
        f
    } else {
        fs::File::create(&tmp)?
    };

    // Optional full-screen ratatui renderer (`--tui`). We only engage it on a real
    // color TTY; otherwise we fall through to the plain `print_progress` printer.
    // The `TermGuard` is RAII: it restores raw-mode/alternate-screen on every exit
    // path (normal, error mid-download, panic). Built once per file's stream.
    let mut tui_term: Option<(crate::ui::TermGuard, Terminal<CrosstermBackend<Stdout>>)> = None;
    if tui && crate::ui::Ui::from_env(false).color() && std::io::stdout().is_terminal() {
        let guard = crate::ui::TermGuard::enter();
        if guard.active() {
            let term = Terminal::new(CrosstermBackend::new(std::io::stdout()))?;
            tui_term = Some((guard, term));
        }
        // If the guard failed to arm we leave `tui_term` None and fall back to plain.
    }

    let mut reader = resp.into_reader();
    let mut buf = [0u8; 64 * 1024];
    let mut got = already;
    let start = Instant::now();
    let mut last_tick = start;
    // BANDWIDTH CAP (ARF_PULL_MAX_MBPS). A 17 GB model pull saturates the link and makes the
    // machine unusable for anything else — including working on another box on the same
    // connection. Cap it: sleep whenever we are ahead of the target byte budget for the elapsed
    // time. Unset = no limit (previous behaviour).
    let max_bps: Option<f64> = std::env::var("ARF_PULL_MAX_MBPS")
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|v| *v > 0.0)
        .map(|mbps| mbps * 1_000_000.0);
    let throttle_from = already;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        hasher.update(&buf[..n]);
        got += n as u64;
        if let Some(bps) = max_bps {
            // Bytes fetched THIS session (resumed bytes cost no time), vs. what the cap allows
            // by now. Sleep off the excess.
            let session = (got - throttle_from) as f64;
            let allowed_secs = session / bps;
            let spent = start.elapsed().as_secs_f64();
            if allowed_secs > spent {
                std::thread::sleep(Duration::from_secs_f64(
                    (allowed_secs - spent).min(1.0), // cap a single nap so progress stays live
                ));
            }
        }
        // Throttle redraws to ~10 Hz so we don't spam the terminal.
        if last_tick.elapsed() >= Duration::from_millis(100) {
            draw_progress(
                tui_term.as_mut(),
                label,
                got,
                total,
                already,
                start.elapsed(),
            )?;
            last_tick = Instant::now();
        }
    }
    file.flush()?;
    draw_progress(
        tui_term.as_mut(),
        label,
        got,
        total,
        already,
        start.elapsed(),
    )?;
    // Whether the full-screen renderer was actually engaged (vs. plain fallback).
    let used_tui = tui_term.is_some();
    // Restore the terminal before we print the verify/rename lines below: drop the
    // guard (and Terminal) now so plain `println!`s land on the normal screen.
    drop(tui_term);
    if !used_tui {
        // The plain printer leaves the cursor on a `\r` progress line; terminate it
        // with a newline (the TUI path already restored the screen on guard drop).
        println!();
    }

    // Verify the content hash before trusting the file.
    if let Some(want) = expected {
        let have = format!("{:x}", hasher.finalize());
        if have.as_str() != want {
            fs::remove_file(&tmp).ok();
            return Err(format!("sha256 mismatch (got {have}, want {want})").into());
        }
        println!("  {label:<34} verified sha256");
    }

    fs::rename(&tmp, dest)?;
    Ok(())
}

/// Human-readable byte size, e.g. `2.0 GB`, `1.4 KB`, `96 B`.
/// Pure, testable progress facts for one file — shared by the plain printer and
/// the ratatui renderer. Built + tested here; wired into the TTY path by the
/// `--tui` integration that consumes `draw_pull_frame`.
pub(crate) struct ProgressLine {
    pub label: String,
    /// A pre-rendered text bar. The ratatui gauge renders from `pct`/`ratio`
    /// instead, so this is unread by `draw_pull_frame` (it backs the plain-bar
    /// shape + the `ProgressLine` unit test). Kept for parity with the plain path.
    #[allow(dead_code)]
    pub bar: String,
    pub pct: u16,
    pub got: String,
    pub total: String,
    pub speed: String,
    pub eta: String,
}

impl ProgressLine {
    pub fn new(label: &str, got: u64, total: u64, speed_bps: f64, _elapsed: Duration) -> Self {
        let (frac, eta) = if total > 0 {
            let f = (got as f64 / total as f64).min(1.0);
            let eta = if speed_bps > 0.0 {
                format!("{:>3.0}s", total.saturating_sub(got) as f64 / speed_bps)
            } else {
                "  ?".to_string()
            };
            (f, eta)
        } else {
            (0.0, "  ?".to_string())
        };
        let filled = ((frac * 20.0) as usize).min(20);
        Self {
            label: label.to_string(),
            bar: "█".repeat(filled) + &"░".repeat(20 - filled),
            pct: (frac * 100.0) as u16,
            got: crate::ui::human_size(got),
            total: if total > 0 {
                crate::ui::human_size(total)
            } else {
                "?".into()
            },
            speed: format!("{}/s", crate::ui::human_size(speed_bps as u64)),
            eta,
        }
    }
}

use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    widgets::{Block, Borders, Gauge},
    Terminal,
};
use std::io::Stdout;

/// Draw a single-file progress frame (one bordered gauge) for `pl`.
pub(crate) fn draw_pull_frame(
    term: &mut Terminal<CrosstermBackend<Stdout>>,
    pl: &ProgressLine,
) -> std::io::Result<()> {
    let accent = Color::Rgb(55, 224, 176);
    term.draw(|f| {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3)])
            .split(f.area());
        let g = Gauge::default()
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(format!(" {} ", pl.label)),
            )
            .gauge_style(Style::default().fg(accent))
            .ratio(pl.pct as f64 / 100.0)
            .label(format!(
                "{:>3}%  {}/{}  {}  eta {}",
                pl.pct, pl.got, pl.total, pl.speed, pl.eta
            ));
        f.render_widget(g, chunks[0]);
    })?;
    Ok(())
}

/// Render one progress update: a full-screen ratatui frame when `tui` is `Some`
/// (the `--tui` path), else the plain carriage-return printer. `already` is the
/// bytes present before this run (resumed); session bytes drive the speed/ETA so
/// both renderers agree. Errors only from the ratatui draw (I/O on the terminal).
fn draw_progress(
    tui: Option<&mut (crate::ui::TermGuard, Terminal<CrosstermBackend<Stdout>>)>,
    label: &str,
    got: u64,
    total: u64,
    already: u64,
    elapsed: Duration,
) -> std::io::Result<()> {
    match tui {
        Some((_guard, term)) => {
            let secs = elapsed.as_secs_f64().max(1e-3);
            let speed = (got - already) as f64 / secs; // bytes/s this session
            let pl = ProgressLine::new(label, got, total, speed, elapsed);
            draw_pull_frame(term, &pl)
        }
        None => {
            print_progress(label, got, total, got - already, elapsed);
            Ok(())
        }
    }
}

/// Print a single carriage-return progress line: a bar (if total is known),
/// human sizes, speed, and ETA. `session` is bytes fetched this run (for speed).
fn print_progress(label: &str, got: u64, total: u64, session: u64, elapsed: Duration) {
    let secs = elapsed.as_secs_f64().max(1e-3);
    let speed = session as f64 / secs; // bytes/s this session
    let speed_s = format!("{}/s", crate::ui::human_size(speed as u64));
    if total > 0 {
        let frac = (got as f64 / total as f64).min(1.0);
        let filled = ((frac * 20.0) as usize).min(20);
        let bar: String = "█".repeat(filled) + &"░".repeat(20 - filled);
        let eta = if speed > 0.0 {
            let rem = total.saturating_sub(got) as f64 / speed;
            format!("ETA {:>3.0}s", rem)
        } else {
            "ETA   ?".to_string()
        };
        print!(
            "\r  {label:<34} {bar} {:>3.0}% {:>9}/{:<9} {speed_s:>10} {eta}",
            frac * 100.0,
            crate::ui::human_size(got),
            crate::ui::human_size(total),
        );
    } else {
        print!(
            "\r  {label:<34} {:>9} {speed_s:>10}",
            crate::ui::human_size(got)
        );
    }
    let _ = std::io::stdout().flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-09-27: a complete model dir pulls with no network at all (the quickstart failed offline).
    #[test]
    fn a_complete_dir_is_up_to_date_without_the_network() {
        let d = std::env::temp_dir().join(format!("arf_pull_local_{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        for f in SUPPORT_FILES {
            fs::write(d.join(f), b"{}").unwrap();
        }
        assert_eq!(local_complete(&d), None, "no weights yet");
        fs::write(d.join(WEIGHTS_FILE), b"w").unwrap();
        assert_eq!(local_complete(&d), Some(3));
        // a repo name that cannot resolve: any network call would fail the pull
        pull(
            "nonexistent-org/never",
            "main",
            &d,
            None,
            None,
            None,
            false,
            false,
        )
        .unwrap();
        // sharded: complete only when every shard in the index is present
        fs::remove_file(d.join(WEIGHTS_FILE)).unwrap();
        fs::write(
            d.join(WEIGHTS_INDEX),
            br#"{"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();
        fs::write(d.join("model-00001-of-00002.safetensors"), b"w").unwrap();
        assert_eq!(local_complete(&d), None, "a shard is missing");
        fs::write(d.join("model-00002-of-00002.safetensors"), b"w").unwrap();
        assert_eq!(local_complete(&d), Some(5));
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn progress_line_formats_bar_and_pct() {
        let pl = ProgressLine::new("model-1", 5, 10, 1024.0, std::time::Duration::from_secs(1));
        assert_eq!(pl.pct, 50);
        assert!(pl.bar.contains('█'));
        assert!(pl.bar.contains('░'));
        assert!(pl.eta.contains('s'));
    }

    #[test]
    fn pull_frame_draws_into_test_backend() {
        use ratatui::{backend::TestBackend, Terminal};
        let backend = TestBackend::new(60, 4);
        let mut term = Terminal::new(backend).unwrap();
        let pl = ProgressLine::new("model-1", 5, 10, 1024.0, std::time::Duration::from_secs(1));
        // Render via the same drawing closure draw_pull_frame uses, but against TestBackend.
        term.draw(|f| {
            use ratatui::{
                layout::{Constraint, Direction, Layout},
                style::{Color, Style},
                widgets::{Block, Borders, Gauge},
            };
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(3)])
                .split(f.area());
            let g = Gauge::default()
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(format!(" {} ", pl.label)),
                )
                .gauge_style(Style::default().fg(Color::Rgb(55, 224, 176)))
                .ratio(pl.pct as f64 / 100.0)
                .label(format!("{:>3}%", pl.pct));
            f.render_widget(g, chunks[0]);
        })
        .unwrap();
        let buf = term.backend().buffer().clone();
        let text: String = buf.content().iter().map(|c| c.symbol()).collect();
        assert!(text.contains("model-1"));
        assert!(text.contains("50%"));
    }

    #[test]
    fn builds_resolve_url_with_revision() {
        assert_eq!(
            resolve_url("unsloth/Llama-3.2-1B-Instruct", "main", "config.json"),
            "https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/resolve/main/config.json"
        );
        // a pinned commit revision flows straight into the path
        assert_eq!(
            resolve_url("org/model", "abc123", "model.safetensors"),
            "https://huggingface.co/org/model/resolve/abc123/model.safetensors"
        );
    }

    #[test]
    fn resolves_shorthand_repo_and_revision() {
        // a known shorthand maps to its repo, default slug, no revision
        let r = resolve_model("llama3.2:1b");
        assert_eq!(r.repo, "unsloth/Llama-3.2-1B-Instruct");
        assert_eq!(r.revision, None);
        assert_eq!(r.out_slug, "llama-3.2-1b-instruct");

        // a raw org/repo passes through untouched
        let r = resolve_model("meta-llama/Llama-3.2-1B");
        assert_eq!(r.repo, "meta-llama/Llama-3.2-1B");
        assert_eq!(r.out_slug, "llama-3.2-1b");

        // `@rev` is split off for both shorthands and raw repos
        let r = resolve_model("org/model@abc123");
        assert_eq!(r.repo, "org/model");
        assert_eq!(r.revision.as_deref(), Some("abc123"));
        let r = resolve_model("llama3.2:3b@v2");
        assert_eq!(r.repo, "unsloth/Llama-3.2-3B-Instruct");
        assert_eq!(r.revision.as_deref(), Some("v2"));
    }

    // Edge cases for the `:` quant tag vs alias `:`,
    // the `hf.co/` prefix, and `@rev` ↔ `:quant` interaction.
    #[test]
    fn alias_colon_is_not_a_quant() {
        // The central break: `qwen3:30b` is an ALIAS key; its `:` must NOT be peeled
        // as a quant (which would leave a bogus raw repo `qwen3`).
        let r = resolve_model("qwen3:30b");
        assert_eq!(r.repo, "Qwen/Qwen3-Coder-30B-A3B-Instruct");
        assert_eq!(r.quant, None);
        // `llama3.2:1b` likewise keeps its alias and gets NO phantom quant.
        let r = resolve_model("llama3.2:1b");
        assert_eq!(r.repo, "unsloth/Llama-3.2-1B-Instruct");
        assert_eq!(r.quant, None);
    }

    #[test]
    fn strips_hf_prefix_and_parses_quant() {
        let r = resolve_model("hf.co/org/repo-GGUF:Q4_K_M");
        assert_eq!(r.repo, "org/repo-GGUF");
        assert_eq!(r.quant.as_deref(), Some("Q4_K_M"));
        // slug excludes the quant tag
        assert_eq!(r.out_slug, "repo-gguf");
        // huggingface.co/ prefix too
        let r = resolve_model("huggingface.co/org/repo");
        assert_eq!(r.repo, "org/repo");
        assert_eq!(r.quant, None);
    }

    #[test]
    fn raw_repo_quant_without_prefix() {
        let r = resolve_model("org/repo-GGUF:Q6_K");
        assert_eq!(r.repo, "org/repo-GGUF");
        assert_eq!(r.quant.as_deref(), Some("Q6_K"));
    }

    #[test]
    fn rev_and_quant_together() {
        // `org/repo@rev:Q4_K_M` → repo=org/repo, revision=rev, quant=Q4_K_M.
        // (@ is split first; the quant lives on the repo segment after @rev removal.)
        let r = resolve_model("org/repo@main:Q4_K_M");
        assert_eq!(r.repo, "org/repo");
        assert_eq!(r.revision.as_deref(), Some("main"));
        assert_eq!(r.quant.as_deref(), Some("Q4_K_M"));
    }

    #[test]
    fn plain_safetensors_repo_has_no_quant() {
        let r = resolve_model("org/model");
        assert_eq!(r.repo, "org/model");
        assert_eq!(r.quant, None);
        assert_eq!(r.revision, None);
    }

    #[test]
    fn extract_rfilenames_from_hf_json() {
        let json = r#"{"id":"x","siblings":[{"rfilename":".gitattributes"},{"rfilename":"gemma4-coding-Q4_K_M.gguf"},{"rfilename":"README.md"}]}"#;
        let files = extract_rfilenames(json);
        assert_eq!(
            files,
            vec![".gitattributes", "gemma4-coding-Q4_K_M.gguf", "README.md"]
        );
        assert!(extract_rfilenames("{}").is_empty());
    }

    #[test]
    fn pick_gguf_outcomes() {
        let files = vec![
            "README.md".to_string(),
            "gemma4-coding-Q2_K.gguf".to_string(),
            "gemma4-coding-Q4_K_M.gguf".to_string(),
            "gemma4-coding-Q6_K.gguf".to_string(),
        ];
        // exact quant match → that file
        assert_eq!(
            pick_gguf(&files, Some("Q4_K_M")).unwrap(),
            "gemma4-coding-Q4_K_M.gguf"
        );
        // case-insensitive
        assert_eq!(
            pick_gguf(&files, Some("q6_k")).unwrap(),
            "gemma4-coding-Q6_K.gguf"
        );
        // no quant + multiple → error listing them
        let e = pick_gguf(&files, None).unwrap_err().to_string();
        assert!(e.contains("multiple"));
        assert!(e.contains("Q4_K_M"));
        // quant matches none → error
        let e = pick_gguf(&files, Some("Q9_NOPE")).unwrap_err().to_string();
        assert!(e.contains("no .gguf matches"));
        // ambiguous (`Q` substring hits several) → error, never silently first
        let e = pick_gguf(&files, Some("Q")).unwrap_err().to_string();
        assert!(e.contains("ambiguous"));
        // single .gguf + no quant → take it
        let one = vec!["only-Q8_0.gguf".to_string()];
        assert_eq!(pick_gguf(&one, None).unwrap(), "only-Q8_0.gguf");
        // no gguf at all → error
        assert!(pick_gguf(&["model.safetensors".to_string()], None).is_err());
    }

    #[test]
    fn resolves_gemma3_alias() {
        let r = resolve_model("gemma3");
        // google/gemma-3-4b-it is gated; we resolve to the ungated unsloth mirror.
        assert_eq!(r.repo, "unsloth/gemma-3-4b-it");
        assert!(r.revision.is_none());
    }
    #[test]
    fn gemma4_is_retired_not_an_alias_for_gemma3() {
        // It used to resolve to the Gemma 3 4B repo — a shorthand that fetched a model it did
        // not name. It must never silently do that again.
        assert_ne!(resolve_model("gemma4").repo, "unsloth/gemma-3-4b-it");
        assert_ne!(resolve_model("gemma4:4b").repo, "unsloth/gemma-3-4b-it");
        assert!(!known_aliases()
            .iter()
            .any(|(k, _)| k.to_ascii_lowercase().starts_with("gemma4")));
        for spec in ["gemma4", "gemma4:4b", "GEMMA4", "gemma4@main"] {
            let msg = retired_alias(spec).unwrap_or_else(|| panic!("{spec} must be retired"));
            assert!(msg.contains("arf pull gemma3:4b")); // says what to type instead
        }
        // a real Gemma 4 repo, and the live shorthands, are untouched
        assert_eq!(retired_alias("org/gemma4-coding:Q4_K_M"), None);
        assert_eq!(retired_alias("gemma3"), None);
        assert_eq!(retired_alias("gemma3:4b"), None);
    }
    #[test]
    fn passes_through_raw_repo_with_rev() {
        let r = resolve_model("org/model@abc123");
        assert_eq!(r.repo, "org/model");
        assert_eq!(r.revision.as_deref(), Some("abc123"));
    }
    #[test]
    fn parses_shard_list_unique_and_sorted() {
        // weight_map values point at shards, often out of order and repeated.
        let index = r#"{
            "metadata": {"total_size": 16060522496},
            "weight_map": {
                "lm_head.weight": "model-00004-of-00004.safetensors",
                "model.embed_tokens.weight": "model-00001-of-00004.safetensors",
                "model.layers.0.mlp.down_proj.weight": "model-00001-of-00004.safetensors",
                "model.layers.20.mlp.up_proj.weight": "model-00003-of-00004.safetensors",
                "model.norm.weight": "model-00002-of-00004.safetensors"
            }
        }"#;
        let shards = parse_shard_list(index).unwrap();
        assert_eq!(
            shards,
            vec![
                "model-00001-of-00004.safetensors",
                "model-00002-of-00004.safetensors",
                "model-00003-of-00004.safetensors",
                "model-00004-of-00004.safetensors",
            ]
        );
    }

    #[test]
    fn shard_parse_ignores_non_shard_strings_and_errors_when_empty() {
        // The index name itself ends in .json, not .safetensors, and keys that
        // merely mention safetensors-like text shouldn't be mistaken for files.
        let no_shards = r#"{"weight_map": {"a": "model.bin", "b": "tokenizer.json"}}"#;
        assert!(parse_shard_list(no_shards).is_err());
    }

    #[test]
    fn linked_etag_accepts_only_64_hex_sha256() {
        // A 64-char LFS content sha256 is kept (quotes stripped, lowercased)...
        let sha256 = "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855";
        assert_eq!(
            parse_linked_etag(&format!("\"{sha256}\"")).as_deref(),
            Some(sha256.to_ascii_lowercase().as_str())
        );
        // ...but a 40-char git-blob sha1 (non-LFS files) is rejected — we only
        // verify the big LFS weights, whose linked-etag is a sha256.
        assert_eq!(
            parse_linked_etag("\"0123456789abcdef0123456789abcdef01234567\""),
            None
        );
        // garbage rejected
        assert_eq!(parse_linked_etag("\"not-a-hash\""), None);
    }

    #[test]
    fn qwen38_shorthand_resolves_to_the_bundle() {
        for spec in ["qwen3.8:27b", "QWEN3.8:27B", "qwen3.8", "hf.co/qwen3.8:27b"] {
            let r = resolve_model(spec);
            let b = r
                .bundle
                .unwrap_or_else(|| panic!("{spec} must be the bundle"));
            assert_eq!(b.dir, "qwen3.8-27b-arf");
            assert_eq!(r.out_slug, "qwen3.8-27b-arf"); // its own dir, not the repo's slug
            assert_eq!(r.repo, qwen38_repo());
            assert_eq!(r.quant, None); // the `:27b` is the alias key, never a quant
            assert_eq!((b.arch, b.quant), ("qwen3.8-27b", "q4ks"));
            assert_eq!(b.draft_repo, "incoai/Qwen3.8-27B-DFlash2");
            assert_eq!(
                (b.mmproj_repo, b.mmproj_file),
                ("unsloth/Qwen3.8-27B-GGUF", "mmproj-BF16.gguf")
            );
        }
        // a pinned revision applies to the weights repo
        let r = resolve_model("qwen3.8:27b@abc123");
        assert!(r.bundle.is_some());
        assert_eq!(r.revision.as_deref(), Some("abc123"));
        // known for the typo check, and suggested for near misses
        assert!(is_known_alias("qwen3.8:27b"));
        assert_eq!(suggest_alias("qwen38:27b"), Some("qwen3.8:27b"));
        // the other shorthands are not bundles
        assert!(resolve_model("qwen3:30b").bundle.is_none());
        assert!(resolve_model("org/repo:Q4_K_M").bundle.is_none());
    }

    #[test]
    fn qwen38_repo_env_override() {
        assert_eq!(repo_or_default(None, ARF_QWEN38_REPO), ARF_QWEN38_REPO);
        assert_eq!(repo_or_default(Some("  ".into()), "d/x"), "d/x");
        assert_eq!(
            repo_or_default(Some(" me/Arf-GGUF ".into()), "d/x"),
            "me/Arf-GGUF"
        );
    }

    #[test]
    fn pick_bundle_gguf_skips_projectors_and_refuses_to_guess() {
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        let files = s(&[
            "README.md",
            "mmproj-BF16.gguf",
            "imatrix_unsloth.gguf",
            "Qwen3.8-27B-arf-g64-imat-Q4_1.gguf",
        ]);
        assert_eq!(
            pick_bundle_gguf(&files).unwrap(),
            "Qwen3.8-27B-arf-g64-imat-Q4_1.gguf"
        );
        // several weights files, one says g64 -> that one
        let files = s(&["a-Q8_0.gguf", "b-g64-Q4_1.gguf"]);
        assert_eq!(pick_bundle_gguf(&files).unwrap(), "b-g64-Q4_1.gguf");
        // several and none says g64 -> an error listing them, never the first
        let e = pick_bundle_gguf(&s(&["a.gguf", "b.gguf"]))
            .unwrap_err()
            .to_string();
        assert!(e.contains("a.gguf") && e.contains("b.gguf"), "{e}");
        // nothing but a projector -> an error (the pull then prints how to build it)
        assert!(pick_bundle_gguf(&s(&["mmproj-BF16.gguf"])).is_err());
    }

    /// The models directory: `ARF_MODELS_DIR` wins; with no `./models` beside the process (the test
    /// binary runs in the crate directory, which has none) an installed `arf` lands in
    /// `~/.arf/models`. One test, because both arms set process env.
    #[test]
    fn models_dir_prefers_the_env_then_home() {
        let saved = (std::env::var_os("ARF_MODELS_DIR"), std::env::var_os("HOME"));
        assert!(
            !Path::new(SOURCE_MODELS_DIR).is_dir(),
            "the crate dir must not have ./models"
        );
        std::env::set_var("ARF_MODELS_DIR", "/tmp/arf-models-test");
        assert_eq!(
            default_models_dir(),
            std::path::PathBuf::from("/tmp/arf-models-test")
        );
        std::env::remove_var("ARF_MODELS_DIR");
        std::env::set_var("HOME", "/tmp/arf-home-test");
        assert_eq!(
            default_models_dir(),
            std::path::PathBuf::from("/tmp/arf-home-test/.arf/models")
        );
        match saved.0 {
            Some(v) => std::env::set_var("ARF_MODELS_DIR", v),
            None => std::env::remove_var("ARF_MODELS_DIR"),
        }
        match saved.1 {
            Some(v) => std::env::set_var("HOME", v),
            None => std::env::remove_var("HOME"),
        }
    }

    #[test]
    fn unreachable_message_says_how_to_build_it() {
        let b = &BUNDLES[0];
        let default_out = Path::new(SOURCE_MODELS_DIR).join(b.dir);
        let m = bundle_unreachable(
            b,
            "me/x",
            "repo me/x not found on HuggingFace (404)",
            &default_out,
        );
        assert!(m.contains("404"), "{m}");
        assert!(m.contains("\n      scripts/get_qwen38.sh\n"), "{m}");
        assert!(m.contains("arf serve qwen3.8:27b"), "{m}");
        assert!(m.contains("ARF_QWEN38_REPO"), "{m}");
        // a non-default --out is carried into the command
        let m = bundle_unreachable(b, "me/x", "down", Path::new("/big/disk/q"));
        assert!(m.contains("OUT=/big/disk/q scripts/get_qwen38.sh"), "{m}");
        assert!(m.contains("arf serve /big/disk/q"), "{m}");
    }

    #[test]
    fn complete_bundle_pull_is_offline_and_writes_the_manifest() {
        // Every file present: nothing is fetched (no network in CI), and the manifest arf-serve
        // reads is written from the spec.
        let dir = std::env::temp_dir().join(format!("arf_bundle_pull_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("draft")).unwrap();
        for f in [
            "model.gguf",
            "mmproj.gguf",
            "draft/config.json",
            "draft/model.safetensors",
        ] {
            fs::write(dir.join(f), b"GGUF").unwrap();
        }
        pull_bundle(&BUNDLES[0], "main", &dir, None, false, false).unwrap();
        let m = arf_core::bundle::Manifest::read(&dir).unwrap();
        assert_eq!(m.arch.as_deref(), Some("qwen3.8-27b"));
        assert_eq!(m.quant.as_deref(), Some("q4ks"));
        // and the result is a bundle that arf-serve resolves with every part attached
        let r = arf_core::bundle::resolve(&arf_core::bundle::Request {
            model: dir.clone(),
            draft_supported: true,
            mmproj_supported: true,
            ..Default::default()
        })
        .unwrap();
        assert!(r.draft.path().is_some() && r.mmproj.path().is_some());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_build_script_matches_the_bundle_spec() {
        // scripts/get_qwen38.sh writes the same bundle by hand; if the two drift, `arf pull`
        // and the script produce directories that serve differently.
        let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/get_qwen38.sh");
        let text =
            fs::read_to_string(&script).unwrap_or_else(|e| panic!("{}: {e}", script.display()));
        let b = &BUNDLES[0];
        assert_eq!(b.build_script, "scripts/get_qwen38.sh");
        for needle in [
            format!("models/{}", b.dir),
            format!("arch={}", b.arch),
            format!("quant={}", b.quant),
            b.draft_repo.to_string(),
            b.mmproj_repo.to_string(),
            b.mmproj_file.to_string(),
            arf_core::bundle::MANIFEST_FILE.to_string(),
            arf_core::bundle::MODEL_FILE.to_string(),
            arf_core::bundle::MMPROJ_FILE.to_string(),
        ] {
            assert!(
                text.contains(&needle),
                "{} lacks {needle:?}",
                script.display()
            );
        }
    }

    #[test]
    fn suggests_close_alias() {
        assert_eq!(suggest_alias("gema3"), Some("gemma3")); // 1 deletion
        assert_eq!(suggest_alias("llama3.2:1"), Some("llama3.2:1b")); // close
    }
    #[test]
    fn no_suggestion_for_far_input() {
        assert_eq!(suggest_alias("totally-different-xyz"), None);
        // An exact alias is not a "suggestion" (caller handles exact match separately)
        // but returning Some(exact) is acceptable; assert it doesn't suggest nonsense:
        assert!(suggest_alias("zzzzzzzz").is_none());
    }

    #[test]
    fn skips_existing_nonempty_files() {
        let dir = std::env::temp_dir().join(format!("arf_pull_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let present = dir.join("config.json");
        std::fs::write(&present, b"{}").unwrap();

        assert!(should_skip(&present));
        assert!(!should_skip(&dir.join("model.safetensors")));

        // An empty (truncated) file is NOT skipped — we re-fetch it.
        let empty = dir.join("tokenizer.json");
        std::fs::write(&empty, b"").unwrap();
        assert!(!should_skip(&empty));

        std::fs::remove_dir_all(&dir).ok();
    }

    // The qwen3.8:27b bundle as HuggingFace sizes it (tree API and `X-Linked-Size`, 2026-10-04): the weights
    // GGUF, then the draft `model.safetensors` + the BF16 projector `arf serve` attaches by default.
    const QWEN_GGUF: &str = "Qwen3.8-27B-arf-g64-imat-Q4_1.gguf";
    const QWEN_WEIGHTS: u64 = 17_119_616_032;
    const QWEN_EXTRAS: u64 = 3_848_817_896 + 931_146_432;
    // `qwen3:30b` (Qwen/Qwen3-Coder-30B-A3B-Instruct): 16 bf16 safetensors shards, summed the same
    // way (the total the live refusal printed before this test existed).
    const CODER30_BF16: u64 = 61_066_575_656;
    const GIB: u64 = 1 << 30;

    fn bundle(extras: u64) -> PullSize {
        pull_size(&[(QWEN_GGUF, QWEN_WEIGHTS)], extras)
    }

    /// the 16 GB Mac is refused before downloading, with the exact byte counts, the cap, a
    /// smaller model and the `--force` escape; with `--force` the same check only warns.
    #[test]
    fn fit_check_refuses_a_model_over_the_load_cap() {
        let ram = 16 * GIB;
        let err = fit_check("qwen3.8:27b", bundle(QWEN_EXTRAS), ram, false)
            .expect_err("17.1 GB of weights on a 16 GB Mac must be refused");
        let cap = (ram as f64 * arf_gpu::weights::MEM_CAP_FRACTION) as u64;
        for needle in [
            "refusing to download",
            &QWEN_WEIGHTS.to_string(),
            &ram.to_string(),
            &cap.to_string(),
            "arf pull llama3.2:1b",
            "arf pull qwen3.8:27b --force",
        ] {
            assert!(err.contains(needle), "missing {needle:?} in: {err}");
        }

        let forced = fit_check("qwen3.8:27b", bundle(QWEN_EXTRAS), ram, true)
            .expect("--force must not refuse")
            .expect("--force over the cap must still warn");
        assert!(forced.contains("--force") && forced.contains(&QWEN_WEIGHTS.to_string()));
    }

    /// Exactly at the cap loads (the guard refuses only `est > cap`); one byte over is refused.
    #[test]
    fn fit_check_cap_boundary_matches_the_load_guard() {
        let ram = 32 * GIB;
        let cap = (ram as f64 * arf_gpu::weights::MEM_CAP_FRACTION) as u64;
        let gguf = |b| pull_size(&[("m.gguf", b)], 0);
        assert!(fit_check("m", gguf(cap), ram, false).is_ok());
        assert!(fit_check("m", gguf(cap + 1), ram, false).is_err());
    }

    /// Between the warn threshold and the cap: proceed, with a warning. A bundle's draft and
    /// projector count toward the warning (`arf serve` loads them by default) but never toward
    /// the refusal (`--no-draft` / `--no-mmproj` drop them).
    #[test]
    fn fit_check_warns_when_tight_and_counts_bundle_extras_only_for_the_warning() {
        let ram = 24 * GIB;
        let warn_at = (ram as f64 * PULL_WARN_FRACTION) as u64;
        assert!(QWEN_WEIGHTS < warn_at && QWEN_WEIGHTS + QWEN_EXTRAS > warn_at);
        assert_eq!(fit_check("qwen3.8:27b", bundle(0), ram, false), Ok(None));
        let w = fit_check("qwen3.8:27b", bundle(QWEN_EXTRAS), ram, false)
            .expect("fits under the cap")
            .expect("the default bundle is over 75% of 24 GB");
        assert!(
            w.contains("--no-draft") && w.contains(&ram.to_string()),
            "{w}"
        );

        let gguf = |b| pull_size(&[("m.gguf", b)], 0);
        assert_eq!(fit_check("m", gguf(warn_at), ram, false), Ok(None));
        let plain = fit_check("m", gguf(warn_at + 1), ram, false)
            .unwrap()
            .expect("one byte over the warn threshold warns");
        assert!(
            !plain.contains("--no-draft") && !plain.contains("--quant"),
            "{plain}"
        );
    }

    /// 🔴 The refusal must never be stricter than the load guard. `qwen3:30b` is 57 GB of bf16
    /// safetensors, over the cap on a 36 GB box at its stored precision — and it loads there at
    /// `--quant q4ks`, because the guard prices safetensors at the target quant. So it is a
    /// WARNING naming `--quant`, not a refusal. On an 8 GB box even q3k cannot fit: refused.
    #[test]
    fn fit_check_prices_safetensors_at_the_smallest_quant_like_the_guard() {
        let shards: Vec<(&str, u64)> = vec![
            ("model-00001-of-00002.safetensors", CODER30_BF16 / 2),
            (
                "model-00002-of-00002.safetensors",
                CODER30_BF16 - CODER30_BF16 / 2,
            ),
        ];
        let size = pull_size(&shards, 0);
        assert_eq!(size.weights, CODER30_BF16);
        assert!(size.min_resident < size.weights / 4);

        let w = fit_check("qwen3:30b", size, 36 * GIB, false)
            .expect("a shipped alias that loads at --quant q4ks must not be refused on 36 GB")
            .expect("but it is over the cap at bf16, so warn");
        assert!(w.contains("--quant"), "{w}");

        let err = fit_check("qwen3:30b", size, 8 * GIB, false).expect_err("8 GB cannot hold it");
        assert!(err.contains("even at --quant q3k"), "{err}");
    }

    /// The whole bundle sits comfortably on the 36 GB box the numbers were measured on.
    #[test]
    fn fit_check_passes_the_bundle_on_36_gb() {
        assert_eq!(
            fit_check("qwen3.8:27b", bundle(QWEN_EXTRAS), 36 * GIB, false),
            Ok(None)
        );
    }

    /// The warn threshold is derived from the load guard's cap, below it, and is the issue's 75%.
    #[test]
    fn pull_warn_fraction_is_below_the_load_cap() {
        let cap = arf_gpu::weights::MEM_CAP_FRACTION;
        assert!(PULL_WARN_FRACTION < cap);
        assert!((PULL_WARN_FRACTION - 0.75).abs() < 1e-9);
    }

    /// A real tree-API response (abridged): sizes come from the top-level `size`, LFS entries
    /// included, and every entry is found.
    #[test]
    fn parses_tree_api_sizes() {
        let json = r#"[{"type":"file","oid":"efa3","size":894,"path":"config.json"},{"type":"file","oid":"5daf","size":2471645608,"lfs":{"oid":"1ff7","size":2471645608,"pointerSize":135},"xetHash":"8d72","path":"model.safetensors"},{"type":"directory","oid":"aa","size":0,"path":"sub"},{"type":"file","oid":"bb","size":7,"lfs":{"oid":"cc","size":7,"pointerSize":1},"path":"sub/x.gguf"}]"#;
        assert_eq!(
            parse_tree_sizes(json),
            vec![
                ("config.json".to_string(), 894),
                ("model.safetensors".to_string(), 2_471_645_608),
                ("sub".to_string(), 0),
                ("sub/x.gguf".to_string(), 7),
            ]
        );
        assert!(parse_tree_sizes("not json").is_empty());
    }

    /// Only weight files are sized; the JSON that rides along is not.
    #[test]
    fn weight_files_are_recognised() {
        assert!(is_weight_file("model-00001-of-00004.safetensors"));
        assert!(is_weight_file(QWEN_GGUF));
        assert!(!is_weight_file("config.json"));
        assert!(!is_weight_file(WEIGHTS_INDEX));
    }

    /// The live half of the fit check: HuggingFace's tree API sizes the bundle's three files
    /// without downloading them, to the byte (the same numbers as their `X-Linked-Size` headers). Network-bound, so `#[ignore]`d like the other network test.
    #[test]
    #[ignore]
    fn sizes_the_bundle_without_downloading() {
        let spec = &BUNDLES[0];
        let w = remote_sizes(&(spec.model_repo)(), "main", &[QWEN_GGUF], None).expect("weights");
        let d = remote_sizes(spec.draft_repo, "main", &["model.safetensors"], None).expect("draft");
        let m = remote_sizes(spec.mmproj_repo, "main", &[spec.mmproj_file], None).expect("mmproj");
        assert_eq!(w[0].1, QWEN_WEIGHTS);
        assert_eq!(d[0].1 + m[0].1, QWEN_EXTRAS);
    }
}
