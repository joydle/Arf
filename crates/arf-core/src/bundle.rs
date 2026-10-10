//! A model BUNDLE: one directory holding everything a model needs to serve at its best, so a new
//! user types one name instead of four flags.
//!
//! ```text
//! models/qwen3.8-27b-arf/
//!   model.gguf         the weights                          (--model)
//!   draft/             a DFlash 2 block draft: config.json + model.safetensors   (--draft)
//!   mmproj.gguf        the vision projector                 (--mmproj)
//!   arf-bundle.txt     `arch=` / `quant=` the weights need  (--arch, --quant)
//! ```
//!
//! `arf pull qwen3.8:27b` and `scripts/get_qwen38.sh` write this layout; `arf-serve --model <dir>`
//! (or `--model <dir>/model.gguf`) reads it back through [`resolve`]. Every part is optional, and
//! every flag given on the command line wins over what the bundle holds. `--no-draft` /
//! `--no-mmproj` turn the automatic attach off.
//!
//! Pure path logic, no GPU and no network, so every decision here is unit-tested.

use std::path::{Path, PathBuf};

/// The weights inside a bundle directory.
pub const MODEL_FILE: &str = "model.gguf";
/// The block-draft checkpoint directory inside a bundle.
pub const DRAFT_DIR: &str = "draft";
/// The vision projector inside a bundle.
pub const MMPROJ_FILE: &str = "mmproj.gguf";
/// The small `key=value` file naming the `--arch` / `--quant` the weights need.
pub const MANIFEST_FILE: &str = "arf-bundle.txt";

/// The files a draft directory must hold before it is attached. A pull writes `*.part` until a
/// file is complete and verified, so a half-downloaded draft has no `model.safetensors` yet.
const DRAFT_FILES: [&str; 2] = ["config.json", "model.safetensors"];

/// What `arf-bundle.txt` says. Unknown keys are ignored (forward compatibility).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Manifest {
    /// `--arch` for the weights, e.g. `qwen3.8-27b`.
    pub arch: Option<String>,
    /// `--quant` for the weights, e.g. `q4ks`.
    pub quant: Option<String>,
}

impl Manifest {
    /// Parse `key=value` lines (`key = value` too). `#` starts a comment; blank lines are skipped.
    pub fn parse(text: &str) -> Manifest {
        let mut m = Manifest::default();
        for line in text.lines() {
            let line = line.split('#').next().unwrap_or("").trim();
            let Some((k, v)) = line.split_once('=') else {
                continue;
            };
            let v = v.trim();
            if v.is_empty() {
                continue;
            }
            match k.trim() {
                "arch" => m.arch = Some(v.to_string()),
                "quant" => m.quant = Some(v.to_string()),
                _ => {}
            }
        }
        m
    }

    /// The file text, with a header naming who wrote it. [`Manifest::parse`] reads it back.
    pub fn render(&self, written_by: &str) -> String {
        let mut s = format!(
            "# written by {written_by}; read by arf-serve when --model points at this directory.\n\
             # A flag given on the command line wins over the value here.\n"
        );
        if let Some(a) = &self.arch {
            s.push_str(&format!("arch={a}\n"));
        }
        if let Some(q) = &self.quant {
            s.push_str(&format!("quant={q}\n"));
        }
        s
    }

    /// Read `<dir>/arf-bundle.txt`; `None` when it is missing or unreadable.
    pub fn read(dir: &Path) -> Option<Manifest> {
        std::fs::read_to_string(dir.join(MANIFEST_FILE))
            .ok()
            .map(|s| Manifest::parse(&s))
    }
}

/// What the command line asked for.
#[derive(Debug, Default, Clone)]
pub struct Request {
    /// `--model`: a GGUF, a safetensors file or directory, or a bundle directory.
    pub model: PathBuf,
    /// `--arch`, if given.
    pub arch: Option<String>,
    /// `--quant`, if given.
    pub quant: Option<String>,
    /// `--draft`, if given.
    pub draft: Option<PathBuf>,
    /// `--mmproj`, if given.
    pub mmproj: Option<PathBuf>,
    /// `--no-draft`: never attach the bundle's draft.
    pub no_draft: bool,
    /// `--no-mmproj`: never attach the bundle's projector.
    pub no_mmproj: bool,
    /// This build can run a block draft (Metal only today). When false a bundle's draft is left
    /// off with a note instead of failing the load.
    pub draft_supported: bool,
    /// This build can run the bundle's projector. Same rule.
    pub mmproj_supported: bool,
}

/// How one optional part (the draft, the projector) was decided.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Part {
    /// Given on the command line; used as is.
    Explicit(PathBuf),
    /// Found in the bundle and attached.
    Auto(PathBuf),
    /// Found in the bundle, left off by `--no-draft` / `--no-mmproj`.
    OptedOut,
    /// Found in the bundle, left off because this build cannot run it.
    Unsupported,
    /// Present in the bundle but not finished (a pull still running, or interrupted).
    Incomplete(PathBuf),
    /// Nothing to attach.
    Absent,
}

impl Part {
    /// The path to pass on, if any.
    pub fn path(&self) -> Option<&Path> {
        match self {
            Part::Explicit(p) | Part::Auto(p) => Some(p),
            _ => None,
        }
    }

    fn describe(&self, flag: &str, opt_out: &str) -> String {
        match self {
            Part::Explicit(p) => format!("{flag} {} (given)", p.display()),
            Part::Auto(p) => format!("{flag} {} (auto)", p.display()),
            Part::OptedOut => format!("no {flag} ({opt_out})"),
            Part::Unsupported => format!("no {flag} (not supported by this build)"),
            Part::Incomplete(p) => format!("no {flag} ({} is incomplete)", p.display()),
            Part::Absent => format!("no {flag}"),
        }
    }
}

/// Where a value came from, for the log line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// The command line.
    Flag,
    /// The bundle's `arf-bundle.txt`.
    Manifest,
}

/// The resolved launch: what to load, and what to attach.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolved {
    /// The weights to load (a bundle directory is replaced by its `model.gguf`).
    pub model: PathBuf,
    /// The bundle directory, when the model is (or sits in) one. `None` = no bundle involved,
    /// and nothing below was changed from the request.
    pub bundle_dir: Option<PathBuf>,
    /// `--arch` to use, and where it came from.
    pub arch: Option<(String, Source)>,
    /// `--quant` to use, and where it came from.
    pub quant: Option<(String, Source)>,
    /// The block draft.
    pub draft: Part,
    /// The vision projector.
    pub mmproj: Part,
}

impl Resolved {
    /// One line naming what was loaded and attached — `None` when no bundle was involved, so a
    /// plain `--model x.gguf` launch prints nothing new.
    pub fn summary(&self) -> Option<String> {
        let dir = self.bundle_dir.as_ref()?;
        let mut parts = vec![format!("model {}", self.model.display())];
        let src = |s: Source| match s {
            Source::Flag => "given",
            Source::Manifest => MANIFEST_FILE,
        };
        if let Some((a, s)) = &self.arch {
            parts.push(format!("--arch {a} ({})", src(*s)));
        }
        if let Some((q, s)) = &self.quant {
            parts.push(format!("--quant {q} ({})", src(*s)));
        }
        parts.push(self.draft.describe("--draft", "--no-draft"));
        parts.push(self.mmproj.describe("--mmproj", "--no-mmproj"));
        Some(format!("bundle {}: {}", dir.display(), parts.join(", ")))
    }

    /// The name a bundle is served under by default: its directory's name (`model.gguf` says
    /// nothing). `None` when no bundle was involved.
    pub fn served_name(&self) -> Option<String> {
        self.bundle_dir
            .as_ref()?
            .file_name()
            .and_then(|n| n.to_str())
            .map(str::to_string)
    }
}

/// Is `dir` a draft directory ready to attach? `Err(())` = it exists but is incomplete.
fn draft_state(dir: &Path) -> Option<Result<(), ()>> {
    if !dir.join(DRAFT_FILES[0]).is_file() {
        return None;
    }
    Some(if DRAFT_FILES.iter().all(|f| non_empty(&dir.join(f))) {
        Ok(())
    } else {
        Err(())
    })
}

fn non_empty(p: &Path) -> bool {
    std::fs::metadata(p).is_ok_and(|m| m.is_file() && m.len() > 0)
}

/// Does `dir` look like a bundle (for a GGUF inside it)? Only the bundle's own names count, so a
/// GGUF that merely sits next to other files is left alone.
fn has_bundle_parts(dir: &Path) -> bool {
    dir.join(DRAFT_DIR).join(DRAFT_FILES[0]).is_file()
        || dir.join(MMPROJ_FILE).is_file()
        || dir.join(MANIFEST_FILE).is_file()
}

/// The one GGUF in `dir` that can be its weights: a single `*.gguf` file once projectors
/// (`mmproj*`) are set aside. `None` for none or for several — a guess between two models would
/// be worse than the error.
fn single_gguf(dir: &Path) -> Option<PathBuf> {
    let mut found = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|x| x == "gguf")
                && !p
                    .file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with("mmproj"))
                && crate::model::gguf::is_gguf_file(p)
        });
    let first = found.next()?;
    found.next().is_none().then_some(first)
}

/// Decide what to load and attach. Explicit flags always win; the bundle only fills gaps.
///
/// `Err` only for a directory that is plainly a bundle (it has a draft, a projector or a
/// manifest) but holds no `model.gguf` — left alone it would fall through to the safetensors
/// loader and fail with a message about safetensors.
pub fn resolve(req: &Request) -> Result<Resolved, String> {
    // 1. Which file are the weights, and is there a bundle around them?
    let (model, bundle_dir) = if req.model.is_dir() {
        let inner = req.model.join(MODEL_FILE);
        if inner.is_file() {
            (inner, Some(req.model.clone()))
        } else if has_bundle_parts(&req.model) {
            return Err(format!(
                "{} is a model bundle without its weights ({MODEL_FILE} is missing — a pull or \
                 scripts/get_qwen38.sh that did not finish?). Re-run it; finished files are skipped.",
                req.model.display()
            ));
        } else if let Some(only) = single_gguf(&req.model) {
            // A FOLDER HOLDING ONE GGUF UNDER ITS OWN NAME (2026-10-07): what `arf pull org/repo
            // --file x.gguf` writes, or a folder made by hand. It was taken for a safetensors
            // directory, so `arf run <name>` could not start it.
            (only, None)
        } else {
            // A safetensors directory, not a bundle.
            (req.model.clone(), None)
        }
    } else if crate::model::gguf::is_gguf_file(&req.model) {
        let parent = req
            .model
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
        let bundle = has_bundle_parts(&parent).then_some(parent);
        (req.model.clone(), bundle)
    } else {
        (req.model.clone(), None)
    };

    let manifest = bundle_dir
        .as_deref()
        .and_then(Manifest::read)
        .unwrap_or_default();
    let pick = |flag: &Option<String>, from_file: Option<String>| match flag {
        Some(v) => Some((v.clone(), Source::Flag)),
        None => from_file.map(|v| (v, Source::Manifest)),
    };
    let arch = pick(&req.arch, manifest.arch);
    let quant = pick(&req.quant, manifest.quant);

    // 2. The draft.
    let draft = match (&req.draft, &bundle_dir) {
        (Some(p), _) => Part::Explicit(p.clone()),
        (None, None) => Part::Absent,
        (None, Some(dir)) => {
            let d = dir.join(DRAFT_DIR);
            match draft_state(&d) {
                None => Part::Absent,
                Some(Err(())) => Part::Incomplete(d),
                Some(Ok(())) if req.no_draft => Part::OptedOut,
                Some(Ok(())) if !req.draft_supported => Part::Unsupported,
                Some(Ok(())) => Part::Auto(d),
            }
        }
    };

    // 3. The projector.
    let mmproj = match (&req.mmproj, &bundle_dir) {
        (Some(p), _) => Part::Explicit(p.clone()),
        (None, None) => Part::Absent,
        (None, Some(dir)) => {
            let m = dir.join(MMPROJ_FILE);
            if !m.is_file() {
                Part::Absent
            } else if !crate::model::gguf::is_gguf_file(&m) {
                Part::Incomplete(m)
            } else if req.no_mmproj {
                Part::OptedOut
            } else if !req.mmproj_supported {
                Part::Unsupported
            } else {
                Part::Auto(m)
            }
        }
    };

    Ok(Resolved {
        model,
        bundle_dir,
        arch,
        quant,
        draft,
        mmproj,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// A scratch directory unique to one test.
    struct Tmp(PathBuf);
    impl Tmp {
        fn new(tag: &str) -> Tmp {
            let p = std::env::temp_dir().join(format!("arf_bundle_{tag}_{}", std::process::id()));
            let _ = fs::remove_dir_all(&p);
            fs::create_dir_all(&p).unwrap();
            Tmp(p)
        }
    }
    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn gguf(p: &Path) {
        fs::write(p, b"GGUF\x03\0\0\0rest").unwrap();
    }

    /// A bundle with the parts asked for.
    fn bundle(tag: &str, draft: bool, mmproj: bool, manifest: bool) -> Tmp {
        let t = Tmp::new(tag);
        gguf(&t.0.join(MODEL_FILE));
        if draft {
            fs::create_dir_all(t.0.join(DRAFT_DIR)).unwrap();
            fs::write(t.0.join(DRAFT_DIR).join("config.json"), b"{}").unwrap();
            fs::write(t.0.join(DRAFT_DIR).join("model.safetensors"), b"x").unwrap();
        }
        if mmproj {
            gguf(&t.0.join(MMPROJ_FILE));
        }
        if manifest {
            let m = Manifest {
                arch: Some("qwen3.8-27b".into()),
                quant: Some("q4ks".into()),
            };
            fs::write(t.0.join(MANIFEST_FILE), m.render("a test")).unwrap();
        }
        t
    }

    fn req(model: &Path) -> Request {
        Request {
            model: model.to_path_buf(),
            draft_supported: true,
            mmproj_supported: true,
            ..Request::default()
        }
    }

    /// A folder holding one GGUF under the repository's own name is that model; with two, or
    /// with only a projector, it is left to the safetensors path as before.
    #[test]
    fn a_folder_with_one_gguf_under_another_name_is_that_model() {
        let t = Tmp::new("single_gguf");
        let file = t.0.join("OpenJev-Q4_K_M.gguf");
        gguf(&file);
        gguf(&t.0.join("mmproj-F16.gguf")); // a projector beside it does not count as weights
        let r = resolve(&req(&t.0)).unwrap();
        assert_eq!(r.model, file);

        gguf(&t.0.join("Other-Q8_0.gguf"));
        assert_eq!(
            resolve(&req(&t.0)).unwrap().model,
            t.0,
            "two candidates: no guess"
        );

        let p = Tmp::new("only_mmproj");
        gguf(&p.0.join("mmproj-F16.gguf"));
        assert_eq!(resolve(&req(&p.0)).unwrap().model, p.0);
    }

    #[test]
    fn manifest_round_trips_and_ignores_noise() {
        let m = Manifest {
            arch: Some("qwen3.8-27b".into()),
            quant: Some("q4ks".into()),
        };
        assert_eq!(Manifest::parse(&m.render("x")), m);
        let loose = "# c\n\n arch = qwen3.8-27b  # trailing\nfuture=1\nquant=\n";
        assert_eq!(
            Manifest::parse(loose),
            Manifest {
                arch: Some("qwen3.8-27b".into()),
                quant: None
            }
        );
    }

    #[test]
    fn bundle_dir_attaches_everything_and_fills_arch_quant() {
        let t = bundle("full", true, true, true);
        let r = resolve(&req(&t.0)).unwrap();
        assert_eq!(r.model, t.0.join(MODEL_FILE));
        assert_eq!(r.bundle_dir.as_deref(), Some(t.0.as_path()));
        assert_eq!(r.draft, Part::Auto(t.0.join(DRAFT_DIR)));
        assert_eq!(r.mmproj, Part::Auto(t.0.join(MMPROJ_FILE)));
        assert_eq!(r.arch, Some(("qwen3.8-27b".into(), Source::Manifest)));
        assert_eq!(r.quant, Some(("q4ks".into(), Source::Manifest)));
        let line = r.summary().unwrap();
        assert!(
            line.contains("--draft") && line.contains("(auto)"),
            "{line}"
        );
        assert!(line.contains("--arch qwen3.8-27b"), "{line}");
        assert_eq!(
            r.served_name().as_deref(),
            t.0.file_name().and_then(|n| n.to_str())
        );
    }

    #[test]
    fn gguf_inside_a_bundle_counts_as_the_bundle() {
        let t = bundle("file", true, true, true);
        let r = resolve(&req(&t.0.join(MODEL_FILE))).unwrap();
        assert_eq!(r.bundle_dir.as_deref(), Some(t.0.as_path()));
        assert!(matches!(r.draft, Part::Auto(_)));
        assert!(matches!(r.mmproj, Part::Auto(_)));
        assert_eq!(r.arch.unwrap().1, Source::Manifest);
    }

    #[test]
    fn without_draft_dir_only_mmproj_attaches() {
        let t = bundle("nodraft", false, true, true);
        let r = resolve(&req(&t.0)).unwrap();
        assert_eq!(r.draft, Part::Absent);
        assert!(matches!(r.mmproj, Part::Auto(_)));
    }

    #[test]
    fn without_mmproj_only_draft_attaches() {
        let t = bundle("nommproj", true, false, false);
        let r = resolve(&req(&t.0)).unwrap();
        assert!(matches!(r.draft, Part::Auto(_)));
        assert_eq!(r.mmproj, Part::Absent);
        // no manifest: nothing invented for arch / quant
        assert_eq!(r.arch, None);
        assert_eq!(r.quant, None);
    }

    #[test]
    fn explicit_flags_win_over_the_bundle() {
        let t = bundle("explicit", true, true, true);
        let mut q = req(&t.0);
        q.draft = Some(PathBuf::from("/elsewhere/draft"));
        q.mmproj = Some(PathBuf::from("/elsewhere/mm.gguf"));
        q.arch = Some("qwen3.6".into());
        q.quant = Some("q4k".into());
        let r = resolve(&q).unwrap();
        assert_eq!(r.draft, Part::Explicit(PathBuf::from("/elsewhere/draft")));
        assert_eq!(
            r.mmproj,
            Part::Explicit(PathBuf::from("/elsewhere/mm.gguf"))
        );
        assert_eq!(r.arch, Some(("qwen3.6".into(), Source::Flag)));
        assert_eq!(r.quant, Some(("q4k".into(), Source::Flag)));
        // the weights still come from the bundle
        assert_eq!(r.model, t.0.join(MODEL_FILE));
    }

    #[test]
    fn opt_outs_leave_the_parts_off() {
        let t = bundle("optout", true, true, true);
        let mut q = req(&t.0);
        q.no_draft = true;
        q.no_mmproj = true;
        let r = resolve(&q).unwrap();
        assert_eq!(r.draft, Part::OptedOut);
        assert_eq!(r.mmproj, Part::OptedOut);
        assert_eq!(r.draft.path(), None);
        let line = r.summary().unwrap();
        assert!(
            line.contains("--no-draft") && line.contains("--no-mmproj"),
            "{line}"
        );
        // an opt-out never touches arch / quant
        assert!(r.arch.is_some());
    }

    #[test]
    fn unsupported_builds_skip_instead_of_failing() {
        let t = bundle("unsup", true, true, false);
        let mut q = req(&t.0);
        q.draft_supported = false;
        q.mmproj_supported = false;
        let r = resolve(&q).unwrap();
        assert_eq!(r.draft, Part::Unsupported);
        assert_eq!(r.mmproj, Part::Unsupported);
    }

    #[test]
    fn half_downloaded_draft_is_not_attached() {
        let t = bundle("partial", false, false, false);
        fs::create_dir_all(t.0.join(DRAFT_DIR)).unwrap();
        fs::write(t.0.join(DRAFT_DIR).join("config.json"), b"{}").unwrap();
        fs::write(t.0.join(DRAFT_DIR).join("model.part"), b"xx").unwrap();
        let r = resolve(&req(&t.0)).unwrap();
        assert_eq!(r.draft, Part::Incomplete(t.0.join(DRAFT_DIR)));
    }

    #[test]
    fn a_bundle_without_its_weights_is_an_error_not_a_fallthrough() {
        let t = bundle("noweights", true, true, true);
        fs::remove_file(t.0.join(MODEL_FILE)).unwrap();
        let e = resolve(&req(&t.0)).unwrap_err();
        assert!(e.contains("model.gguf is missing"), "{e}");
    }

    #[test]
    fn plain_gguf_and_safetensors_dir_are_untouched() {
        // A GGUF with nothing of a bundle beside it: no bundle, no summary, nothing attached.
        let t = Tmp::new("plain");
        let g = t.0.join("Qwen3.8-27B-Q4_K_M.gguf");
        gguf(&g);
        fs::write(t.0.join("mmproj-Qwen3.8-27B-BF16.gguf"), b"GGUF").unwrap(); // not the bundle name
        let r = resolve(&req(&g)).unwrap();
        assert_eq!(r.model, g);
        assert_eq!(r.bundle_dir, None);
        assert_eq!(r.draft, Part::Absent);
        assert_eq!(r.mmproj, Part::Absent);
        assert_eq!(r.summary(), None);

        // A safetensors directory (no model.gguf) is passed through unchanged.
        let s = Tmp::new("st");
        fs::write(s.0.join("model.safetensors"), b"x").unwrap();
        fs::write(s.0.join("config.json"), b"{}").unwrap();
        let r = resolve(&req(&s.0)).unwrap();
        assert_eq!(r.model, s.0);
        assert_eq!(r.bundle_dir, None);
    }
}
