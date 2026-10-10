//! `arf serve <model> [args…]` — a thin shim that resolves the model alias
//! and `exec`s the separate `arf-serve` daemon binary. The binaries stay
//! split; this is just a friendly front door.

use std::error::Error;
use std::path::{Path, PathBuf};

/// Find the `arf-serve` binary: prefer a sibling of `current_exe`, else rely
/// on `$PATH`. `exe_dir` is injected for testability.
pub fn locate_serve(exe_dir: Option<&Path>) -> Option<PathBuf> {
    let name = if cfg!(windows) {
        "arf-serve.exe"
    } else {
        "arf-serve"
    };
    if let Some(dir) = exe_dir {
        let sibling = dir.join(name);
        if sibling.is_file() {
            return Some(sibling);
        }
    }
    // Fall back to PATH (returned as a bare name; the OS resolves it on exec).
    Some(PathBuf::from(name))
}

use crate::pull;

/// Resolve a user-supplied model token to a local model directory, with friendly
/// errors. Order: an existing path is used verbatim; else a known alias / pulled
/// model under `models_dir`; else a helpful error that suggests the closest alias
/// (typo) and tells the user to pull it.
pub fn resolve_local_model(model: &str, models_dir: &str) -> Result<PathBuf, Box<dyn Error>> {
    // 1. explicit existing path
    let p = Path::new(model);
    if p.exists() {
        return Ok(p.to_path_buf());
    }
    // 2. a retired shorthand is an error that says what to type, never a lookup
    if let Some(why) = pull::retired_alias(model) {
        return Err(why.into());
    }
    // 3. alias / pulled-model dir under models_dir
    let resolved = pull::resolve_model(model);
    let dir = Path::new(models_dir).join(&resolved.out_slug);
    if dir.is_dir() {
        return Ok(dir);
    }
    // 4. not found — build a helpful error
    let mut msg = format!("model '{model}' isn't available locally");
    if let Some(s) = pull::suggest_alias(model) {
        if s != model {
            msg.push_str(&format!("\n  did you mean '{s}'?"));
        }
    }
    // Only a name `arf pull` knows, or an `org/repo`, is worth suggesting: for any other bare
    // name `arf pull` had nothing to fetch and left an empty folder and a registry entry
    // (reported 2026-10-07 for `arf run openjev`).
    if pull::is_known_alias(model) || model.contains('/') {
        msg.push_str(&format!("\n  → pull it first:  arf pull {model}"));
    } else {
        msg.push_str(&format!(
            "\n  → `arf ls` lists the downloaded models; `arf pull <org/repo> --file <x.gguf>` \
             downloads a GGUF from Hugging Face. `{model}` is not a name `arf pull` knows."
        ));
    }
    Err(msg.into())
}

/// Resolve the model alias, then run `arf-serve --model <path-or-repo> <rest…>`.
/// On Unix this `exec`s (replaces the process); elsewhere it spawns and waits.
pub fn run(model: &str, passthrough: &[String]) -> Result<(), Box<dyn Error>> {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf));
    let serve = locate_serve(exe_dir.as_deref()).ok_or("could not determine arf-serve location")?;

    // Resolve the model to a local dir with a friendly error BEFORE handing off:
    // a typo'd alias (`gema3`) surfaces "did you mean 'gemma3'?" instead of a
    // cryptic arf-serve "no such file" downstream. `arf-serve --model` wants
    // a local path; we pass the resolved one.
    let model_path = resolve_local_model(model, &pull::default_models_dir().to_string_lossy())?;
    let model_arg = model_path.to_string_lossy().into_owned();

    let mut cmd = std::process::Command::new(&serve);
    cmd.arg("--model").arg(&model_arg);
    cmd.args(passthrough);

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let err = cmd.exec(); // only returns on failure
        Err(format!(
            "failed to exec {} — is arf-serve built? (`cargo build -p arf-serve`): {err}",
            serve.display()
        )
        .into())
    }
    #[cfg(not(unix))]
    {
        let status = cmd.status().map_err(|e| {
            format!(
                "failed to launch {} — is arf-serve built? (`cargo build -p arf-serve`): {e}",
                serve.display()
            )
        })?;
        std::process::exit(status.code().unwrap_or(1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefers_sibling_when_present() {
        let dir = std::env::temp_dir().join(format!("arf_shim_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let name = if cfg!(windows) {
            "arf-serve.exe"
        } else {
            "arf-serve"
        };
        let bin = dir.join(name);
        std::fs::write(&bin, b"#!/bin/sh\n").unwrap();
        assert_eq!(locate_serve(Some(&dir)), Some(bin));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn falls_back_to_bare_name() {
        let empty = std::env::temp_dir().join(format!("arf_shim_empty_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&empty);
        let got = locate_serve(Some(&empty)).unwrap();
        assert_eq!(
            got.file_name().unwrap().to_str().unwrap(),
            if cfg!(windows) {
                "arf-serve.exe"
            } else {
                "arf-serve"
            }
        );
        let _ = std::fs::remove_dir_all(&empty);
    }

    #[test]
    fn resolve_local_errors_with_suggestion_for_typo() {
        // a temp empty models dir so nothing resolves
        let dir = std::env::temp_dir().join(format!("arf_rlm_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let err = resolve_local_model("gema3", dir.to_str().unwrap())
            .unwrap_err()
            .to_string();
        assert!(err.contains("isn't available"));
        assert!(err.contains("gemma3")); // suggestion
        assert!(err.contains("arf pull")); // actionable
        let _ = std::fs::remove_dir_all(&dir);
    }
    /// A bare name `arf pull` does not know is not suggested to it: that pull made an empty folder.
    #[test]
    fn resolve_local_does_not_suggest_pulling_an_unknown_name() {
        let dir = std::env::temp_dir().join(format!("arf_rlm4_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let err = resolve_local_model("openjev", dir.to_str().unwrap())
            .unwrap_err()
            .to_string();
        assert!(err.contains("isn't available"));
        assert!(!err.contains("arf pull openjev"), "{err}");
        assert!(err.contains("arf ls"), "{err}");
        let err = resolve_local_model("org/repo", dir.to_str().unwrap())
            .unwrap_err()
            .to_string();
        assert!(err.contains("arf pull org/repo"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
    #[test]
    fn resolve_local_refuses_the_retired_gemma4_shorthand() {
        let dir = std::env::temp_dir().join(format!("arf_rlm3_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let err = resolve_local_model("gemma4", dir.to_str().unwrap())
            .unwrap_err()
            .to_string();
        assert!(err.contains("not a shorthand"));
        assert!(err.contains("arf pull gemma3:4b")); // what to type instead
        assert!(!err.contains("arf pull gemma4")); // never tells the user to pull it
        let _ = std::fs::remove_dir_all(&dir);
    }
    #[test]
    fn qwen38_shorthand_serves_the_bundle_dir() {
        // `arf serve qwen3.8:27b` -> `arf-serve --model models/qwen3.8-27b-arf`, and arf-serve
        // reads the bundle (weights, draft, projector, --arch / --quant from arf-bundle.txt).
        let dir = std::env::temp_dir().join(format!("arf_rlm_bundle_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let models = dir.to_str().unwrap();
        let err = resolve_local_model("qwen3.8:27b", models)
            .unwrap_err()
            .to_string();
        assert!(err.contains("arf pull qwen3.8:27b"), "{err}");
        let bundle = dir.join("qwen3.8-27b-arf");
        std::fs::create_dir_all(&bundle).unwrap();
        assert_eq!(resolve_local_model("qwen3.8:27b", models).unwrap(), bundle);
        assert_eq!(resolve_local_model("qwen3.8", models).unwrap(), bundle);
        let _ = std::fs::remove_dir_all(&dir);
    }
    #[test]
    fn resolve_local_uses_existing_path() {
        let dir = std::env::temp_dir().join(format!("arf_rlm2_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let got = resolve_local_model(dir.to_str().unwrap(), "./models").unwrap();
        assert_eq!(got, dir);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
