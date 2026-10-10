//! Build script: capture the git short-hash and build date so `--version` can
//! report the exact build. Best-effort — a missing git / non-repo build still
//! compiles; the values just fall back to "unknown" at runtime.

use std::process::Command;

fn main() {
    // Re-run if HEAD moves (best-effort; harmless if the path doesn't exist).
    println!("cargo:rerun-if-changed=../../.git/HEAD");

    let git_hash = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    if let Some(h) = git_hash {
        println!("cargo:rustc-env=ARF_GIT_HASH={h}");
    }

    // Build date (UTC date only). Use `date` — available on the unix/mac dev hosts
    // this builds on; if it fails, the env var is simply unset and --version omits it.
    let date = Command::new("date")
        .args(["-u", "+%Y-%m-%d"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    if let Some(d) = date {
        println!("cargo:rustc-env=ARF_BUILD_DATE={d}");
    }
}
