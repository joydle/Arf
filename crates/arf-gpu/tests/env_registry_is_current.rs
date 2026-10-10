//! L359 — the registry must not drift, or it becomes a lie that reads like documentation.
//!
//! `arf_core::env_registry::KNOWN_VARS` lists every `ARF_*` variable the workspace reads.
//! A stale list is worse than none: `warn_unknown_env_vars` would then name a *real* variable as
//! unknown, training people to ignore the warning — which is exactly the failure it exists to
//! prevent.
//!
//! This test lives in `arf-gpu` rather than `arf-core` because `arf-gpu` holds 254 of
//! the 240 read sites' home crate... more precisely, it holds the overwhelming majority, and the
//! test needs to walk the whole workspace source tree either way. It re-derives the list from
//! source and diffs.

use std::collections::BTreeSet;
use std::path::Path;

/// Walk `crates/` and collect every `ARF_*` name passed to `env::var` or `var_os`.
fn read_vars_from_source(root: &Path) -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                // Skip build output; it contains generated copies that would double-count.
                if p.file_name().is_some_and(|n| n == "target") {
                    continue;
                }
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "rs") {
                let Ok(src) = std::fs::read_to_string(&p) else {
                    continue;
                };
                collect(&src, &mut found);
            }
        }
    }
    found
}

/// Match `env::var("ARF_X")` / `var_os("ARF_X")` without pulling in a regex dependency.
/// Deliberately literal: a variable built at runtime from a format string would not be found
/// here, and would not be findable by the grep that generated the registry either — if one is
/// ever added, this test failing is the correct outcome.
fn collect(src: &str, out: &mut BTreeSet<String>) {
    // Strip line comments FIRST. The scanner found `ARF_X` on its own first run — from the
    // `env::var("ARF_X")` example in this file's own doc comment. A name written in prose is
    // documentation, not a call site, and counting it would make the registry chase its tail.
    // (Only `//` forms matter here: no source in this tree puts a call site inside a `/* */`.)
    let stripped: String = src
        .lines()
        .map(|l| match l.find("//") {
            Some(i) => &l[..i],
            None => l,
        })
        .collect::<Vec<_>>()
        .join("\n");
    let src = stripped.as_str();

    // `env_once!` / `env_once_os!` (island.rs, M1) are process-cached `env::var` / `var_os` reads.
    for (pat, _) in [
        ("env::var(", 0),
        ("var_os(", 0),
        ("env_once!(", 0),
        ("env_once_os!(", 0),
    ] {
        let mut rest = src;
        while let Some(i) = rest.find(pat) {
            rest = &rest[i + pat.len()..];
            let head = rest.trim_start();
            if let Some(q) = head.strip_prefix('"') {
                if let Some(end) = q.find('"') {
                    let name = &q[..end];
                    if name.starts_with("ARF_") {
                        out.insert(name.to_string());
                    }
                }
            }
        }
    }
}

#[test]
fn registry_matches_the_source_tree() {
    // CARGO_MANIFEST_DIR is crates/arf-gpu; the workspace crates dir is one level up.
    let crates_dir = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let from_source = read_vars_from_source(crates_dir);

    // If the source tree is unreadable, say so rather than passing vacuously — a test that goes
    // green because it could not run is the L358 failure mode.
    assert!(
        from_source.len() > 100,
        "found only {} ARF_* vars in {} — the source walk failed, so this test verified \
         NOTHING. Do not treat this as a pass.",
        from_source.len(),
        crates_dir.display()
    );

    let registered: BTreeSet<String> = arf_core::env_registry::KNOWN_VARS
        .iter()
        .map(|s| s.to_string())
        .collect();

    let missing: Vec<&String> = from_source.difference(&registered).collect();
    let stale: Vec<&String> = registered.difference(&from_source).collect();

    assert!(
        missing.is_empty(),
        "these ARF_* vars are read in source but NOT registered — `warn_unknown_env_vars` \
         would call each of them unknown, which trains people to ignore the warning:\n  {missing:#?}"
    );
    assert!(
        stale.is_empty(),
        "these ARF_* vars are registered but no longer read anywhere — remove them so the \
         registry stays a description of what exists:\n  {stale:#?}"
    );
}

/// `KNOWN_VARS` must stay sorted: `warn_unknown_env_vars` uses `binary_search`, which silently
/// returns wrong answers on unsorted input rather than failing.
#[test]
fn registry_is_sorted_because_lookup_uses_binary_search() {
    let v = arf_core::env_registry::KNOWN_VARS;
    for w in v.windows(2) {
        assert!(
            w[0] < w[1],
            "KNOWN_VARS must be sorted ({} >= {}) — binary_search returns wrong answers on \
             unsorted input WITHOUT failing, so a real variable would be reported unknown",
            w[0],
            w[1]
        );
    }
}
