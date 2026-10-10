//! FLUX P1 parity gate: run GpuDiT.forward on the EXACT inputs from the diffusers reference
//! (/tmp/flux_oracle/dit_reference.json) and compare the output velocity (cosine + relative L2).
//! Same gate that caught the vision projector + T5 rel-bias bugs.
//!
//! Usage: cargo run --release -p arf-gpu --example flux_dit_parity

use std::sync::Arc;

use arf_gpu::gpu::flux::{DitConfig, GpuDiT};
use arf_gpu::gpu::GpuContext;
use serde_json::Value;

fn flat(v: &Value) -> Vec<f32> {
    fn rec(v: &Value, out: &mut Vec<f32>) {
        match v {
            Value::Array(a) => a.iter().for_each(|x| rec(x, out)),
            // A reference dump may carry metadata beside its tensor; skip anything non-numeric
            // rather than erroring, and skip a number that will not fit f64 rather than panicking.
            Value::Number(n) => {
                if let Some(f) = n.as_f64() {
                    out.push(f as f32);
                }
            }
            _ => {}
        }
    }
    let mut o = Vec::new();
    rec(v, &mut o);
    o
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/flux_oracle/dit_reference.json".into());
    let r: Value =
        serde_json::from_reader(std::fs::File::open(&path).expect("dit_reference.json")).unwrap();
    let grid = r["grid"].as_u64().unwrap() as usize;
    let txt = r["txt"].as_u64().unwrap() as usize;
    let gguf = std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../models/flux-schnell/flux1-schnell-Q4_K_S.gguf"
    ));

    let img = flat(&r["img"]); // [grid*grid, 64]
    let t5 = flat(&r["t5"]); // [txt, 4096]
    let pooled = flat(&r["pooled"]); // [768]
    let timestep = r["timestep"].as_f64().unwrap() as f32;
    let refv = flat(&r["velocity"]); // [grid*grid, 64]

    let ctx = Arc::new(GpuContext::new().expect("gpu"));
    let cfg = DitConfig::schnell(grid, txt);
    let dit = GpuDiT::load(&ctx, gguf, cfg).expect("load dit");
    let ours = dit.forward(&img, &t5, &pooled, timestep);

    assert_eq!(
        ours.len(),
        refv.len(),
        "len {} vs {}",
        ours.len(),
        refv.len()
    );
    let (mut dot, mut na, mut nb, mut diff2, mut ref2) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
    for (a, b) in ours.iter().zip(refv.iter()) {
        let (a, b) = (*a as f64, *b as f64);
        dot += a * b;
        na += a * a;
        nb += b * b;
        diff2 += (a - b) * (a - b);
        ref2 += b * b;
    }
    let cos = dot / (na.sqrt() * nb.sqrt());
    let rel = (diff2 / ref2).sqrt();
    let ok = cos > 0.99 && rel < 0.1; // Q4 quant vs f32-on-bf16 ref → looser bar than the f16 encoders
    println!(
        "DiT velocity: cos={cos:.6} rel_l2={rel:.4}  {}",
        if ok { "PASS" } else { "FAIL" }
    );
    println!("FLUX P1 parity: {}", if ok { "GREEN ✓" } else { "RED ✗" });
    if !ok {
        std::process::exit(1);
    }
}
