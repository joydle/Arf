//! `arf doctor` — quick environment health check (GPU, models, serve binary,
//! HuggingFace reachability). Exit 0 if nothing FAILED (warnings are fine), 1 if
//! any check failed — so it's usable in CI/setup scripts.

use std::error::Error;
use std::path::Path;

use crate::ui::{self, Ui};

#[derive(Clone, Copy, PartialEq)]
enum Status {
    Ok,
    Warn,
    Fail,
}

impl Status {
    /// Colored glyph (TTY) or plain tag (piped / NO_COLOR).
    fn render(self, ui: Ui) -> String {
        match self {
            Status::Ok => {
                if ui.color() {
                    ui.accent("✓")
                } else {
                    "[ok]  ".into()
                }
            }
            Status::Warn => {
                if ui.color() {
                    ui.amber("⚠")
                } else {
                    "[warn]".into()
                }
            }
            Status::Fail => {
                if ui.color() {
                    ui.red("✗")
                } else {
                    "[fail]".into()
                }
            }
        }
    }
}

fn line(ui: Ui, status: Status, label: &str, detail: &str) {
    println!(
        "  {}  {}  {}",
        status.render(ui),
        ui.bold(&format!("{label:<13}")),
        ui.dim(detail)
    );
}

/// Run all checks; return Ok(()) if none FAILED, else an Err so main exits 1.
pub fn doctor(ui: Ui, models_dir: &Path) -> Result<(), Box<dyn Error>> {
    println!("{}", ui.dim("arf doctor — environment check"));
    println!();
    let mut any_fail = false;

    // 1. GPU
    match arf_gpu::GpuContext::new() {
        Ok(ctx) => line(
            ui,
            Status::Ok,
            "GPU",
            &format!("{} (Metal/wgpu)", ctx.info()),
        ),
        Err(e) => {
            any_fail = true;
            line(
                ui,
                Status::Fail,
                "GPU",
                &format!("no GPU adapter ({e}); serve/run need a GPU"),
            );
        }
    }

    // 2. memory, and what fits in it
    let ram = arf_gpu::weights::physical_ram_bytes();
    let (suggest, why) = crate::welcome::recommended(ram);
    match ram {
        Some(b) => line(
            ui,
            Status::Ok,
            "memory",
            &format!("{} — suggested model: {suggest} ({why})", ui::human_size(b)),
        ),
        None => line(ui, Status::Warn, "memory", "could not read the memory size"),
    }

    // 3. models dir
    let (count, total) = scan_models(models_dir);
    if count > 0 {
        line(
            ui,
            Status::Ok,
            "models",
            &format!(
                "{count} model(s), {} in {}",
                ui::human_size(total),
                models_dir.display()
            ),
        );
    } else {
        line(
            ui,
            Status::Warn,
            "models",
            &format!(
                "none yet in {} — try `arf run {suggest}` (it offers to download)",
                models_dir.display()
            ),
        );
    }

    // 4. arf-serve sibling
    match sibling_serve() {
        Some(p) => line(
            ui,
            Status::Ok,
            "arf-serve",
            &format!("found ({})", p.display()),
        ),
        None => line(
            ui,
            Status::Warn,
            "arf-serve",
            "not built — `cargo build -p arf-serve` (for `arf serve`)",
        ),
    }

    // 5. HuggingFace reachability (best-effort, 2s timeout inside search)
    if crate::hf_search::search("llama").is_empty() {
        line(
            ui,
            Status::Warn,
            "HuggingFace",
            "unreachable or slow — pulls/autocomplete may lag",
        );
    } else {
        line(ui, Status::Ok, "HuggingFace", "reachable");
    }

    println!();
    if any_fail {
        println!("{}", ui.red("some checks failed — see ✗ above"));
        Err("doctor: one or more checks failed".into())
    } else {
        println!("{}", ui.accent("all good"));
        Ok(())
    }
}

/// Count model subdirs (those with a config.json) and sum their byte sizes.
fn scan_models(dir: &Path) -> (usize, u64) {
    crate::list::summary(dir)
}

/// The arf-serve binary as a sibling of the current exe, if it's a real file.
fn sibling_serve() -> Option<std::path::PathBuf> {
    let name = if cfg!(windows) {
        "arf-serve.exe"
    } else {
        "arf-serve"
    };
    let dir = std::env::current_exe().ok()?.parent()?.to_path_buf();
    let p = dir.join(name);
    p.is_file().then_some(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::{ColorMode, Ui};

    #[test]
    fn status_render_plain_when_off() {
        let ui = Ui::new(ColorMode::Off);
        assert_eq!(Status::Ok.render(ui).trim(), "[ok]");
        assert_eq!(Status::Warn.render(ui).trim(), "[warn]");
        assert_eq!(Status::Fail.render(ui).trim(), "[fail]");
    }

    #[test]
    fn scan_models_counts_dirs_with_config() {
        let dir = std::env::temp_dir().join(format!("arf_doc_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("m1")).unwrap();
        std::fs::write(dir.join("m1/config.json"), b"{}").unwrap();
        std::fs::write(dir.join("m1/model.safetensors"), b"abcd").unwrap();
        std::fs::create_dir_all(dir.join("not-a-model")).unwrap(); // no config.json
        let (count, total) = scan_models(&dir);
        assert_eq!(count, 1);
        assert_eq!(total, 6); // 2 (config) + 4 (weights)
        let _ = std::fs::remove_dir_all(&dir);
    }
}
