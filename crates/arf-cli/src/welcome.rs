//! `arf` with no command: what this Mac can run, what is already here, and what to type next.
//! A first command should never be an error.

use std::path::Path;

use crate::ui::Ui;

/// The shorthand to suggest for a Mac with `ram` bytes of memory, and why. The 27B needs ~27 GB
/// free while serving (the formula's caveat), so it is offered from 32 GB up.
pub fn recommended(ram: Option<u64>) -> (&'static str, &'static str) {
    let gb = ram.unwrap_or(0) as f64 / (1u64 << 30) as f64;
    if gb >= 32.0 {
        (
            "qwen3.8:27b",
            "reasoning, tools, images and video — the main model",
        )
    } else if gb >= 12.0 {
        ("gemma3:4b", "small and quick, with image input")
    } else {
        ("llama3.2:1b", "the smallest — a quick first try")
    }
}

/// The model `arf run` / `arf launch` use when none is named: a downloaded one (the main bundle
/// first), else the recommendation for this Mac.
pub fn default_model(models_dir: &Path) -> String {
    let ready = crate::list::ready_names(models_dir);
    if ready.iter().any(|n| n == "qwen3.8-27b-arf") {
        return "qwen3.8:27b".into();
    }
    if let Some(n) = ready.first() {
        return n.clone();
    }
    recommended(arf_gpu::weights::physical_ram_bytes()).0.into()
}

fn chip() -> String {
    std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "this machine".into())
}

pub fn show(ui: Ui, models_dir: &Path, version: &str) {
    let ram = arf_gpu::weights::physical_ram_bytes();
    let ready = crate::list::ready_names(models_dir);
    println!(
        "{}  {}",
        ui.bold("arf"),
        ui.dim(&format!("{version} — fast local models on Apple silicon"))
    );
    println!();
    let mem = ram.map_or(String::new(), |b| {
        format!(" · {} memory", crate::ui::human_size(b))
    });
    println!("  {}  {}{}", ui.dim("this Mac "), chip(), mem);
    if ready.is_empty() {
        println!(
            "  {}  none yet {}",
            ui.dim("models   "),
            ui.faint(&format!("({})", models_dir.display()))
        );
    } else {
        let mut shown = ready.iter().take(4).cloned().collect::<Vec<_>>().join(", ");
        if ready.len() > 4 {
            shown.push_str(&format!(" and {} more", ready.len() - 4));
        }
        println!(
            "  {}  {} {}",
            ui.dim("models   "),
            shown,
            ui.faint(&format!("({}; arf ls)", models_dir.display()))
        );
    }
    println!();
    let model = default_model(models_dir);
    if ready.is_empty() {
        let (alias, why) = recommended(ram);
        println!("  {} {}", ui.dim("Suggested for this Mac:"), ui.bold(alias));
        println!("  {}", ui.faint(why));
        println!();
    }
    let rows = [
        (format!("arf run {model}"), "chat in the terminal"),
        (
            "arf launch claude".to_string(),
            "Claude Code on this model (also: opencode)",
        ),
        (
            format!("arf serve {model}"),
            "OpenAI + Anthropic API on localhost:8080",
        ),
        ("arf doctor".to_string(), "check this Mac"),
    ];
    let w = rows.iter().map(|r| r.0.len()).max().unwrap_or(0);
    for (cmd, what) in rows {
        println!("  {}  {}", ui.accent(&format!("{cmd:<w$}")), ui.dim(what));
    }
    if ready.is_empty() {
        println!();
        println!(
            "  {}",
            ui.faint("The first run downloads the model and asks before it does.")
        );
    }
    println!();
    println!("  {}", ui.faint("All commands: arf --help"));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recommends_by_memory() {
        let gb = |n: u64| Some(n << 30);
        assert_eq!(recommended(gb(36)).0, "qwen3.8:27b");
        assert_eq!(recommended(gb(32)).0, "qwen3.8:27b");
        assert_eq!(recommended(gb(24)).0, "gemma3:4b");
        assert_eq!(recommended(gb(8)).0, "llama3.2:1b");
        assert_eq!(recommended(None).0, "llama3.2:1b");
    }

    #[test]
    fn default_model_falls_back_to_the_recommendation() {
        let dir = std::env::temp_dir().join(format!("arf_welcome_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let m = default_model(&dir);
        assert_eq!(m, recommended(arf_gpu::weights::physical_ram_bytes()).0);
    }
}
