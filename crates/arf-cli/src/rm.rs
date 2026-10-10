//! `arf rm <model>` — delete a local model directory and its registry entry.

use std::error::Error;
use std::fs;
use std::path::Path;

use crate::registry;
use crate::ui::Ui;

/// Remove the model named `slug` from `models_dir`. Refuses if the dir is missing.
pub fn remove(models_dir: &Path, slug: &str, ui: Ui) -> Result<(), Box<dyn Error>> {
    let dir = models_dir.join(slug);
    if !dir.is_dir() {
        return Err(format!("no local model `{slug}` in {}", models_dir.display()).into());
    }
    fs::remove_dir_all(&dir)?;
    let _ = registry::remove(models_dir, slug);
    println!("{} removed {}", ui.accent("✓"), ui.bold(slug));
    Ok(())
}
