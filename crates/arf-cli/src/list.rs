//! `arf list` — show models fetched into the models directory.
//!
//! Reconciles the [`registry`] (provenance: repo, revision,
//! pull date) with what is actually on disk (size, completeness), so the output
//! reflects reality even if a model was deleted, moved, or is mid-download.

use std::error::Error;
use std::fs;
use std::path::Path;

use crate::pull::{SUPPORT_FILES, WEIGHTS_INDEX};
use crate::registry::{self, Entry};
use crate::ui::{self, Ui};

/// Whether a model directory is ready to run, still downloading, or incomplete.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Ready,
    Downloading,
    Incomplete,
}

impl State {
    fn label(self) -> &'static str {
        match self {
            State::Ready => "ready",
            State::Downloading => "downloading",
            State::Incomplete => "incomplete",
        }
    }

    /// Readiness rank for sorting: ready < downloading < incomplete.
    fn rank(self) -> u8 {
        match self {
            State::Ready => 0,
            State::Downloading => 1,
            State::Incomplete => 2,
        }
    }
}

/// How to order the listing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SortKey {
    Name,
    Size,
    State,
}

impl SortKey {
    /// Apply the ordering to rows in place.
    fn apply(self, rows: &mut [Row]) {
        match self {
            SortKey::Name => rows.sort_by(|a, b| a.name.cmp(&b.name)),
            SortKey::Size => {
                // largest first; ties broken by name for a stable, readable order.
                rows.sort_by(|a, b| b.size.cmp(&a.size).then(a.name.cmp(&b.name)))
            }
            SortKey::State => {
                // group by readiness (ready, downloading, incomplete); ties by name.
                rows.sort_by(|a, b| {
                    a.state
                        .rank()
                        .cmp(&b.state.rank())
                        .then(a.name.cmp(&b.name))
                })
            }
        }
    }
}

/// One row of the listing: registry provenance (if any) plus disk facts.
struct Row {
    name: String,
    source: String,
    revision: String,
    size: u64,
    state: State,
    /// Relative pull time from the registry, or `—` when unrecorded.
    pulled: String,
}

/// Entry point for the `list` / `ls` command.
pub fn list(
    models_dir: &Path,
    ui: Ui,
    json: bool,
    sort: SortKey,
    state_filter: Option<State>,
) -> Result<(), Box<dyn Error>> {
    if !models_dir.is_dir() {
        if json {
            println!("[]");
        } else {
            println!(
                "no models yet — pull one with `arf pull llama3.2:1b` (looked in {})",
                models_dir.display()
            );
        }
        return Ok(());
    }
    let registry = registry::load(models_dir);
    let mut rows = collect_rows(models_dir, &registry)?;
    // Filter first, then order — both the table and the JSON honor them.
    if let Some(want) = state_filter {
        rows.retain(|r| r.state == want);
    }
    sort.apply(&mut rows);
    if json {
        print!("{}", render_json(&rows));
        return Ok(());
    }
    if rows.is_empty() {
        println!(
            "no models in {} — pull one with `arf pull llama3.2:1b`",
            models_dir.display()
        );
        return Ok(());
    }
    print!("{}", render_table(ui, &rows));
    Ok(())
}

/// How many models `arf ls` would list in `models_dir`, and their bytes on disk — one rule for
/// `ls` and `doctor` (2026-09-27: `doctor` kept its own config.json-only scan and reported 3
/// models while `ls` listed 9).
pub(crate) fn summary(models_dir: &Path) -> (usize, u64) {
    let registry = registry::load(models_dir);
    collect_rows(models_dir, &registry)
        .map(|rows| (rows.len(), rows.iter().map(|r| r.size).sum()))
        .unwrap_or((0, 0))
}

/// The models `arf ls` shows as ready, by name — for the welcome screen and the default model.
pub(crate) fn ready_names(models_dir: &Path) -> Vec<String> {
    let registry = registry::load(models_dir);
    let mut names: Vec<String> = collect_rows(models_dir, &registry)
        .map(|rows| {
            rows.into_iter()
                .filter(|r| r.state == State::Ready)
                .map(|r| r.name)
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    names
}

/// Scan `models_dir`, building one row per model subdirectory, enriched by the
/// registry where a record exists.
fn collect_rows(models_dir: &Path, registry: &[Entry]) -> Result<Vec<Row>, Box<dyn Error>> {
    let mut rows = Vec::new();
    for dirent in fs::read_dir(models_dir)? {
        let path = dirent?.path();
        if !path.is_dir() {
            continue;
        }
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };
        // A model dir holds a config.json (safetensors), a bundle, or a GGUF; otherwise it's not
        // ours. (2026-09-27: GGUF-only dirs — every Qwen3.8 file, flux — and the `arf pull
        // qwen3.8:27b` bundle were skipped, so `arf ls` hid the main model.)
        if !path.join("config.json").exists()
            && !path.join(arf_core::bundle::MODEL_FILE).exists()
            && !has_top_level_gguf(&path)
        {
            continue;
        }

        let rec = registry.iter().find(|e| e.slug == name);
        rows.push(Row {
            source: rec.map(|e| e.repo.clone()).unwrap_or_else(|| "—".into()),
            revision: rec
                .map(|e| e.revision.clone())
                .unwrap_or_else(|| "—".into()),
            pulled: rec
                .map(|e| registry::relative_time(e.pulled_at))
                .unwrap_or_else(|| "—".into()),
            size: dir_size(&path),
            state: model_state(&path),
            name,
        });
    }
    // Ordering is owned by `list` (via `SortKey::apply`); no sort here.
    Ok(rows)
}

/// A finished `*.gguf` directly in `dir` (not a `.part`).
fn has_top_level_gguf(dir: &Path) -> bool {
    fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .any(|e| e.path().is_file() && e.path().extension().is_some_and(|x| x == "gguf"))
}

/// `config.json` names a speculative draft architecture (e.g. `DFlash2DraftModel`).
fn is_draft_config(dir: &Path) -> bool {
    fs::read_to_string(dir.join("config.json"))
        .map(|c| c.contains("DraftModel"))
        .unwrap_or(false)
}

/// Decide a model dir's state from the files present on disk.
fn model_state(dir: &Path) -> State {
    // A `*.part` anywhere means a download is in flight (or was interrupted).
    let has_part = fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .any(|e| e.path().extension().is_some_and(|x| x == "part"));
    if has_part {
        return State::Downloading;
    }

    let has = |f: &str| dir.join(f).exists();
    // A bundle (`arf pull qwen3.8:27b`): ready once its weights are; its draft downloads into
    // `draft/`, so a `.part` there still means in flight.
    if has(arf_core::bundle::MODEL_FILE) {
        let draft_part = fs::read_dir(dir.join(arf_core::bundle::DRAFT_DIR))
            .into_iter()
            .flatten()
            .flatten()
            .any(|e| e.path().extension().is_some_and(|x| x == "part"));
        return if draft_part {
            State::Downloading
        } else {
            State::Ready
        };
    }
    // GGUF weights (tokenizer embedded): ready when a finished .gguf is present.
    if has_top_level_gguf(dir) {
        return State::Ready;
    }
    // A speculative DRAFT (DFlash 2): config.json + weights, no tokenizer of its own — it uses the
    // target's. Ready when its weights are.
    if is_draft_config(dir) && has("model.safetensors") {
        return State::Ready;
    }
    let support_ok = SUPPORT_FILES.iter().all(|f| has(f));
    // Weights present as a single file, an index + shards, or any *.safetensors.
    let has_weights = has("model.safetensors")
        || has(WEIGHTS_INDEX)
        || fs::read_dir(dir)
            .into_iter()
            .flatten()
            .flatten()
            .any(|e| e.path().extension().is_some_and(|x| x == "safetensors"));

    if support_ok && has_weights {
        State::Ready
    } else {
        State::Incomplete
    }
}

/// Total bytes of regular files directly in `dir` (one level; models are flat).
fn dir_size(dir: &Path) -> u64 {
    fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| e.metadata().ok())
        .filter(|m| m.is_file())
        .map(|m| m.len())
        .sum()
}

/// Render rows as a stable JSON array (for `--json`; scripts/pipes).
fn render_json(rows: &[Row]) -> String {
    let mut out = String::from("[\n");
    for (i, r) in rows.iter().enumerate() {
        out.push_str("  {\n");
        out.push_str(&format!("    \"name\": {},\n", jstr(&r.name)));
        out.push_str(&format!("    \"source\": {},\n", jstr(&r.source)));
        out.push_str(&format!("    \"revision\": {},\n", jstr(&r.revision)));
        out.push_str(&format!("    \"bytes\": {},\n", r.size));
        out.push_str(&format!("    \"state\": {},\n", jstr(r.state.label())));
        out.push_str(&format!("    \"pulled\": {}\n", jstr(&r.pulled)));
        out.push_str(if i + 1 == rows.len() {
            "  }\n"
        } else {
            "  },\n"
        });
    }
    out.push_str("]\n");
    out
}

/// Minimal JSON string escaping (quotes + backslash).
fn jstr(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            _ => o.push(c),
        }
    }
    o.push('"');
    o
}

/// Render the colored, aligned table + a summary line. The state cell is painted
/// by [`Ui::badge`]; `Table` aligns it by visible width, so the ANSI codes never
/// throw off the columns and a model literally named `ready` can't be mis-colored.
fn render_table(ui: Ui, rows: &[Row]) -> String {
    let mut t = ui::Table::new(ui, &["NAME", "SIZE", "STATE", "SOURCE", "REV", "PULLED"]);
    for r in rows {
        t.row(vec![
            r.name.clone(),
            ui::human_size(r.size),
            ui.badge(r.state.label()),
            r.source.clone(),
            r.revision.clone(),
            r.pulled.clone(),
        ]);
    }
    let total: u64 = rows.iter().map(|r| r.size).sum();
    let summary = format!("{} models · {} on disk", rows.len(), ui::human_size(total));
    format!("{}{}\n", t.render(), ui.dim(&summary))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_rows() -> Vec<Row> {
        vec![
            Row {
                name: "gemma3".into(),
                source: "google/gemma-3-4b-it".into(),
                revision: "main".into(),
                size: 8_660_000_000,
                state: State::Ready,
                pulled: "2h ago".into(),
            },
            Row {
                name: "phi-mini".into(),
                source: "—".into(),
                revision: "—".into(),
                size: 2_100_000_000,
                state: State::Incomplete,
                pulled: "—".into(),
            },
        ]
    }

    #[test]
    fn bundle_dir_is_ready_once_its_weights_are() {
        let d = std::env::temp_dir().join(format!("arf_ls_bundle_{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(d.join("draft")).unwrap();
        fs::write(d.join("model.part"), b"x").unwrap();
        assert_eq!(model_state(&d), State::Downloading); // weights in flight
        fs::rename(d.join("model.part"), d.join("model.gguf")).unwrap();
        fs::write(d.join("draft/model.part"), b"x").unwrap();
        assert_eq!(model_state(&d), State::Downloading); // draft in flight
        fs::remove_file(d.join("draft/model.part")).unwrap();
        assert_eq!(model_state(&d), State::Ready);
        let _ = fs::remove_dir_all(&d);
    }

    /// 2026-09-27: `arf ls` must LIST a GGUF-only dir, a bundle and a draft, not only test their
    /// state — `collect_rows` filtered all three out before `model_state` saw them.
    #[test]
    fn gguf_dirs_bundles_and_drafts_are_listed_ready() {
        let root = std::env::temp_dir().join(format!("arf_ls_kinds_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("plain-gguf")).unwrap();
        fs::write(root.join("plain-gguf/Qwen3.8-27B-Q4_K_M.gguf"), b"GGUF").unwrap();
        fs::create_dir_all(root.join("bundle/draft")).unwrap();
        fs::write(root.join("bundle/model.gguf"), b"GGUF").unwrap();
        fs::create_dir_all(root.join("a-draft")).unwrap();
        fs::write(
            root.join("a-draft/config.json"),
            br#"{"architectures": ["DFlash2DraftModel"]}"#,
        )
        .unwrap();
        fs::write(root.join("a-draft/model.safetensors"), b"x").unwrap();
        fs::create_dir_all(root.join("not-a-model")).unwrap();
        fs::write(root.join("not-a-model/notes.txt"), b"x").unwrap();
        let mut rows = collect_rows(&root, &[]).unwrap();
        rows.sort_by(|a, b| a.name.cmp(&b.name));
        let got: Vec<(&str, State)> = rows.iter().map(|r| (r.name.as_str(), r.state)).collect();
        assert_eq!(
            got,
            vec![
                ("a-draft", State::Ready),
                ("bundle", State::Ready),
                ("plain-gguf", State::Ready)
            ]
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn json_is_well_formed_array() {
        let s = render_json(&sample_rows());
        assert!(s.trim_start().starts_with('['));
        assert!(s.trim_end().ends_with(']'));
        assert!(s.contains("\"name\": \"gemma3\""));
        assert!(s.contains("\"state\": \"incomplete\""));
        assert!(s.contains("\"bytes\": 8660000000"));
    }

    #[test]
    fn table_has_header_and_badge_labels() {
        let ui = crate::ui::Ui::new(crate::ui::ColorMode::Off);
        let s = render_table(ui, &sample_rows());
        assert!(s.contains("NAME"));
        assert!(s.contains("gemma3"));
        assert!(s.contains("ready"));
        assert!(s.contains("incomplete"));
        assert!(s.contains("2 models"));
    }

    #[test]
    fn sort_by_size_desc() {
        let mut rows = vec![
            Row {
                name: "a".into(),
                source: "—".into(),
                revision: "—".into(),
                size: 100,
                state: State::Ready,
                pulled: "—".into(),
            },
            Row {
                name: "b".into(),
                source: "—".into(),
                revision: "—".into(),
                size: 900,
                state: State::Ready,
                pulled: "—".into(),
            },
        ];
        SortKey::Size.apply(&mut rows);
        assert_eq!(rows[0].name, "b"); // largest first
    }

    #[test]
    fn sort_by_state_groups_ready_first() {
        let mut rows = vec![
            Row {
                name: "x".into(),
                source: "—".into(),
                revision: "—".into(),
                size: 1,
                state: State::Incomplete,
                pulled: "—".into(),
            },
            Row {
                name: "y".into(),
                source: "—".into(),
                revision: "—".into(),
                size: 1,
                state: State::Ready,
                pulled: "—".into(),
            },
        ];
        SortKey::State.apply(&mut rows);
        assert_eq!(rows[0].state, State::Ready);
    }
}
