//! A shader's `@group(0)` binding list is a contract with every call site. This test freezes it.
//!
//! WHY THIS EXISTS: `ComputeKernel::dispatch` binds `buffers[i]` to `@binding(i)` positionally and
//! validates nothing. Change a shader's bindings without changing its callers and there is no
//! panic and no validation error — a wrong buffer is read at the shifted index and the numbers come
//! out wrong.
//!
//! That is not hypothetical. `rope_interleaved.wgsl` held FLUX's 3-axis fused-q/k RoPE
//! (`q, k, cos, sin, d`). A commit reused the FILENAME for the LLM decode path's single-tensor
//! variant (`x, cos, sin, positions, d`) and did not touch FLUX's call site. Both declare FIVE
//! bindings, so an arity check would NOT have caught it — FLUX's `k` simply landed on the shader's
//! `cos`, and image generation returned solid black for two months.
//!
//! The names are what changed, so the names are what is frozen. `shader_bindings.json` records
//! every shader's binding list in order. If a shader's contract changes, this test fails and the
//! author must update the snapshot deliberately — which is the moment to go look for other callers.
//!
//! No GPU and no model weights required: this reads shader source as text.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

/// `@group(0) @binding(N) ... name:` → binding names ordered by N.
fn binding_names(src: &str) -> Vec<String> {
    let mut pairs: Vec<(usize, String)> = Vec::new();
    for chunk in src.split("@group(0)").skip(1) {
        let Some(rest) = chunk.split_once("@binding(") else {
            continue;
        };
        let Some((idx_s, after)) = rest.1.split_once(')') else {
            continue;
        };
        let Ok(idx) = idx_s.trim().parse::<usize>() else {
            continue;
        };
        // the declared name is the identifier immediately before the first ':'
        let Some((head, _)) = after.split_once(':') else {
            continue;
        };
        let Some(name) = head.split_whitespace().last() else {
            continue;
        };
        let name = name.trim_end_matches(':').trim();
        if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            continue;
        }
        pairs.push((idx, name.to_string()));
    }
    pairs.sort_by_key(|(i, _)| *i);
    pairs.dedup_by_key(|(i, _)| *i);
    pairs.into_iter().map(|(_, n)| n).collect()
}

#[test]
fn shader_binding_contracts_are_unchanged() {
    let root = repo_root();
    // L364b — `src/shaders/wgsl`: the shader directory was split by language (145 files flat).
    // This test's own panic message ("the shader layout changed") is what caught the move.
    let dir = root.join("src/shaders/wgsl");

    let mut current: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for e in std::fs::read_dir(&dir).expect("read shaders dir") {
        let p = e.expect("dir entry").path();
        if p.extension().and_then(|s| s.to_str()) != Some("wgsl") {
            continue;
        }
        let names = binding_names(&std::fs::read_to_string(&p).expect("read shader"));
        if names.is_empty() {
            continue;
        }
        current.insert(p.file_stem().unwrap().to_string_lossy().into_owned(), names);
    }

    let snap_path = root.join("tests/shader_bindings.json");
    let snap: BTreeMap<String, Vec<String>> =
        serde_json::from_str(&std::fs::read_to_string(&snap_path).expect("read snapshot"))
            .expect("parse snapshot");

    assert!(
        current.len() > 100,
        "parsed only {} shaders — the parser or the shader layout changed",
        current.len()
    );

    let mut problems: Vec<String> = Vec::new();
    for (name, want) in &snap {
        match current.get(name) {
            None => problems.push(format!(
                "{name}.wgsl is gone, but the snapshot still lists it ({want:?}). If it was renamed, \
                 check every dispatch site that referenced the OLD name before updating the snapshot."
            )),
            Some(got) if got != want => problems.push(format!(
                "{name}.wgsl bindings changed\n      was: {want:?}\n      now: {got:?}\n      \
                 Every caller binds positionally — find them all, then update the snapshot."
            )),
            _ => {}
        }
    }
    for name in current.keys() {
        if !snap.contains_key(name) {
            problems.push(format!(
                "{name}.wgsl is new. Add it to tests/shader_bindings.json to freeze its contract."
            ));
        }
    }

    assert!(
        problems.is_empty(),
        "shader binding contracts changed:\n  - {}\n\nRegenerate deliberately, not reflexively.",
        problems.join("\n  - ")
    );
}
