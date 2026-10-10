//! A tiny local registry of pulled models — `<models-dir>/registry.json`.
//!
//! `pull` records provenance here (which repo + revision a directory came from,
//! and when); `list` reads it but reconciles every entry against the actual
//! files on disk, so a manually deleted or half-downloaded model is reported
//! truthfully rather than from a stale record. The registry *enriches* the
//! filesystem; the filesystem is the source of truth for what exists.
//!
//! The schema is small and fixed, so we serialize it by hand (a few escaped
//! string fields) rather than pull in a JSON crate — in keeping with the
//! project's minimal-dependency ethos.

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// The registry filename, kept inside the models directory.
pub const REGISTRY_FILE: &str = "registry.json";

/// One recorded model: where it came from and when.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    /// Directory name under the models root (e.g. `llama-3.2-1b-instruct`).
    pub slug: String,
    /// Source HuggingFace repo (`org/model`).
    pub repo: String,
    /// Revision pinned at pull time (branch, tag, or commit).
    pub revision: String,
    /// Unix seconds when the pull was initiated (recorded up front so `list`
    /// shows provenance even while the weights are still downloading).
    pub pulled_at: u64,
}

/// Current unix-epoch seconds (for `pulled_at`).
pub fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Path to the registry file inside `models_dir`.
pub fn path(models_dir: &Path) -> PathBuf {
    models_dir.join(REGISTRY_FILE)
}

/// Record a successful pull, replacing any prior entry for the same slug.
/// Best-effort: a registry write failure must not fail the pull itself.
pub fn record(models_dir: &Path, entry: Entry) -> Result<(), Box<dyn Error>> {
    let mut entries = load(models_dir);
    entries.retain(|e| e.slug != entry.slug);
    entries.push(entry);
    entries.sort_by(|a, b| a.slug.cmp(&b.slug));
    fs::create_dir_all(models_dir)?;
    fs::write(path(models_dir), serialize(&entries))?;
    Ok(())
}

/// Drop the entry for `slug` from the registry (best-effort, like `record`).
pub fn remove(models_dir: &Path, slug: &str) -> Result<(), Box<dyn Error>> {
    let mut entries = load(models_dir);
    let before = entries.len();
    entries.retain(|e| e.slug != slug);
    if entries.len() != before {
        fs::write(path(models_dir), serialize(&entries))?;
    }
    Ok(())
}

/// Load all entries. A missing or unparseable registry yields an empty list
/// (the directory scan in `list` still finds the models themselves).
pub fn load(models_dir: &Path) -> Vec<Entry> {
    fs::read_to_string(path(models_dir))
        .ok()
        .map(|s| parse(&s))
        .unwrap_or_default()
}

/// Serialize entries to a small, stable, pretty JSON array.
fn serialize(entries: &[Entry]) -> String {
    let mut out = String::from("[\n");
    for (i, e) in entries.iter().enumerate() {
        out.push_str("  {\n");
        out.push_str(&format!("    \"slug\": {},\n", json_str(&e.slug)));
        out.push_str(&format!("    \"repo\": {},\n", json_str(&e.repo)));
        out.push_str(&format!("    \"revision\": {},\n", json_str(&e.revision)));
        out.push_str(&format!("    \"pulled_at\": {}\n", e.pulled_at));
        out.push_str(if i + 1 == entries.len() {
            "  }\n"
        } else {
            "  },\n"
        });
    }
    out.push(']');
    out.push('\n');
    out
}

/// Quote + escape a string for JSON (only `"` and `\` matter for our values).
fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Parse our own registry format. Tolerant: pulls `slug`/`repo`/`revision`/
/// `pulled_at` out of each `{ ... }` object, ignoring layout. An object missing
/// a required string field is skipped (forward/loose compatibility).
fn parse(s: &str) -> Vec<Entry> {
    let mut entries = Vec::new();
    // Split on object boundaries; cheap and sufficient for our flat schema.
    for chunk in s.split('{').skip(1) {
        let obj = match chunk.split('}').next() {
            Some(o) => o,
            None => continue,
        };
        let slug = field_str(obj, "slug");
        let repo = field_str(obj, "repo");
        let revision = field_str(obj, "revision");
        if let (Some(slug), Some(repo), Some(revision)) = (slug, repo, revision) {
            entries.push(Entry {
                slug,
                repo,
                revision,
                pulled_at: field_num(obj, "pulled_at").unwrap_or(0),
            });
        }
    }
    entries
}

/// Extract `"key": "value"` from a flat object body, unescaping `\"` and `\\`.
fn field_str(obj: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let rest = &obj[obj.find(&pat)? + pat.len()..];
    let rest = &rest[rest.find(':')? + 1..];
    let start = rest.find('"')? + 1;
    let mut out = String::new();
    let mut chars = rest[start..].chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => return Some(out),
            '\\' => match chars.next() {
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some(other) => out.push(other),
                None => break,
            },
            _ => out.push(c),
        }
    }
    None
}

/// Extract `"key": <number>` from a flat object body.
fn field_num(obj: &str, key: &str) -> Option<u64> {
    let pat = format!("\"{key}\"");
    let rest = &obj[obj.find(&pat)? + pat.len()..];
    let rest = &rest[rest.find(':')? + 1..];
    let digits: String = rest
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

/// A coarse, human "2h ago" / "3d ago" string from a unix timestamp.
pub fn relative_time(then_unix: u64) -> String {
    let now = now_unix();
    if then_unix == 0 || then_unix > now {
        return "—".to_string();
    }
    let secs = now - then_unix;
    match secs {
        0..=59 => "just now".to_string(),
        60..=3599 => format!("{}m ago", secs / 60),
        3600..=86_399 => format!("{}h ago", secs / 3600),
        _ => format!("{}d ago", secs / 86_400),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_entries() {
        let entries = vec![
            Entry {
                slug: "llama-3.2-1b-instruct".into(),
                repo: "unsloth/Llama-3.2-1B-Instruct".into(),
                revision: "main".into(),
                pulled_at: 1_700_000_000,
            },
            Entry {
                slug: "gemma-3-1b-it".into(),
                repo: "unsloth/gemma-3-1b-it".into(),
                revision: "abc123".into(),
                pulled_at: 1_700_000_500,
            },
        ];
        let parsed = parse(&serialize(&entries));
        assert_eq!(parsed, entries);
    }

    #[test]
    fn parses_with_quotes_in_values() {
        // A backslash/quote in a field must survive the round trip.
        let e = Entry {
            slug: "weird".into(),
            repo: "org/a\"b\\c".into(),
            revision: "main".into(),
            pulled_at: 42,
        };
        let parsed = parse(&serialize(std::slice::from_ref(&e)));
        assert_eq!(parsed, vec![e]);
    }

    #[test]
    fn missing_registry_is_empty_not_error() {
        let dir = std::env::temp_dir().join(format!("arf_reg_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        assert!(load(&dir).is_empty());
    }

    #[test]
    fn record_replaces_same_slug() {
        let dir = std::env::temp_dir().join(format!("arf_reg2_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        record(
            &dir,
            Entry {
                slug: "m".into(),
                repo: "old/repo".into(),
                revision: "main".into(),
                pulled_at: 1,
            },
        )
        .unwrap();
        record(
            &dir,
            Entry {
                slug: "m".into(),
                repo: "new/repo".into(),
                revision: "v2".into(),
                pulled_at: 2,
            },
        )
        .unwrap();
        let got = load(&dir);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].repo, "new/repo");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn remove_drops_entry() {
        let dir = std::env::temp_dir().join(format!("arf_reg_rm_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        record(
            &dir,
            Entry {
                slug: "a".into(),
                repo: "x/a".into(),
                revision: "main".into(),
                pulled_at: 1,
            },
        )
        .unwrap();
        record(
            &dir,
            Entry {
                slug: "b".into(),
                repo: "x/b".into(),
                revision: "main".into(),
                pulled_at: 2,
            },
        )
        .unwrap();
        remove(&dir, "a").unwrap();
        let got = load(&dir);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].slug, "b");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn relative_time_buckets() {
        let now = now_unix();
        assert_eq!(relative_time(0), "—");
        assert_eq!(relative_time(now.saturating_sub(30)), "just now");
        assert_eq!(relative_time(now.saturating_sub(120)), "2m ago");
        assert_eq!(relative_time(now.saturating_sub(7200)), "2h ago");
        assert_eq!(relative_time(now.saturating_sub(172_800)), "2d ago");
    }
}
