//! `arf inspect <file>` — list the tensors in a PyTorch checkpoint without executing it.
//!
//! This is how a port learns a checkpoint's tensor names, dtypes and shapes. The file is read by
//! `arf_core::model::torch_ckpt`, a restricted unpickler: a checkpoint that references any
//! global outside its allow-list is refused with the global's name, never run.

use std::error::Error;
use std::path::Path;

use arf_core::model::torch_ckpt::{Object, TorchCheckpoint};

fn fmt_dims(d: &[usize]) -> String {
    let parts: Vec<String> = d.iter().map(|x| x.to_string()).collect();
    format!("[{}]", parts.join(", "))
}

/// A one-line rendering of a non-tensor leaf, for `--all`.
fn leaf(o: &Object) -> Option<String> {
    Some(match o {
        Object::None => "None".into(),
        Object::Bool(b) => b.to_string(),
        Object::Int(i) => i.to_string(),
        Object::BigInt(b) => format!("<int, {} bytes>", b.len()),
        Object::Float(f) => f.to_string(),
        Object::Str(s) if s.len() <= 60 => format!("{s:?}"),
        Object::Str(s) => format!("<str, {} bytes>", s.len()),
        Object::Bytes(b) => format!("<bytes, {} bytes>", b.len()),
        Object::Dtype(d) => format!("torch.{d}"),
        Object::Class(c) => format!("<class {c}>"),
        Object::Storage(i) => format!("<storage #{i}>"),
        _ => return None,
    })
}

fn walk_leaves(o: &Object, path: &mut String, out: &mut Vec<(String, String)>) {
    let mut child = |seg: String, v: &Object, path: &mut String| {
        let keep = path.len();
        if !path.is_empty() {
            path.push('.');
        }
        path.push_str(&seg);
        walk_leaves(v, path, out);
        path.truncate(keep);
    };
    match o {
        Object::Dict(items) => {
            for (k, v) in items {
                let seg = match k {
                    Object::Str(s) => s.clone(),
                    other => leaf(other).unwrap_or_else(|| "?".into()),
                };
                child(seg, v, path);
            }
        }
        // A tuple of plain numbers (a `torch.Size`, Adam betas) reads better on one line.
        Object::Tuple(xs)
            if xs
                .iter()
                .all(|x| matches!(x, Object::Int(_) | Object::Float(_))) =>
        {
            let parts: Vec<String> = xs.iter().filter_map(leaf).collect();
            out.push((path.clone(), format!("({})", parts.join(", "))));
        }
        Object::List(xs) | Object::Tuple(xs) => {
            for (i, v) in xs.iter().enumerate() {
                child(i.to_string(), v, path);
            }
        }
        Object::Tensor(_) => {}
        other => {
            if let Some(s) = leaf(other) {
                out.push((path.clone(), s));
            }
        }
    }
}

/// Print `name  dtype  shape` for every tensor; `all` also prints the non-tensor values
/// (step counters, hyper-parameters) of a training checkpoint.
pub(crate) fn inspect(path: &Path, all: bool) -> Result<(), Box<dyn Error>> {
    let ck = TorchCheckpoint::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let rows: Vec<(&str, String, String, String)> = ck
        .tensors()
        .map(|(n, t)| {
            let note = if t.is_contiguous() {
                String::new()
            } else {
                format!(
                    "  (view: stride {}, offset {})",
                    fmt_dims(&t.stride),
                    t.storage_offset
                )
            };
            (n, t.dtype.to_string(), fmt_dims(&t.shape), note)
        })
        .collect();
    let w0 = rows.iter().map(|r| r.0.len()).max().unwrap_or(0);
    let w1 = rows.iter().map(|r| r.1.len()).max().unwrap_or(0);
    for (n, d, s, note) in &rows {
        println!("{n:<w0$}  {d:<w1$}  {s}{note}");
    }
    let bytes: usize = ck.storages().iter().map(|s| s.numel * s.dtype.size()).sum();
    let params: usize = ck.tensors().map(|(_, t)| t.numel()).sum();
    let size = if bytes >= 1_000_000 {
        format!("{:.2} MB", bytes as f64 / 1e6)
    } else {
        format!("{bytes} bytes")
    };
    eprintln!(
        "{} tensors ({params} elements) in {} storages, {size}, {:?} format",
        rows.len(),
        ck.storages().len(),
        ck.format()
    );
    if all {
        let mut leaves = Vec::new();
        walk_leaves(ck.root(), &mut String::new(), &mut leaves);
        if !leaves.is_empty() {
            println!();
            let w = leaves.iter().map(|l| l.0.len()).max().unwrap_or(0);
            for (k, v) in leaves {
                println!("{k:<w$}  = {v}");
            }
        }
    }
    Ok(())
}
