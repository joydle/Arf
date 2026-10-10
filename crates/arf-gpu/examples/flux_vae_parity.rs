//! FLUX P2 parity gate: run GpuVae.decode on the EXACT latent from the diffusers reference
//! (/tmp/flux_oracle/vae_reference.json) and compare the decoded RGB (cosine + relative L2).
//! Same gate that caught the DiT split + T5 rel-bias bugs.
//!
//! Usage: cargo run --release -p arf-gpu --example flux_vae_parity

use std::sync::Arc;

use arf_gpu::gpu::flux::{GpuVae, VaeConfig};
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
        .unwrap_or_else(|| "/tmp/flux_oracle/vae_reference.json".into());
    let r: Value =
        serde_json::from_reader(std::fs::File::open(&path).expect("vae_reference.json")).unwrap();
    let h = r["H"].as_u64().unwrap() as usize;
    let w = r["W"].as_u64().unwrap() as usize;
    let z = flat(&r["z"]); // raw latent [16, h, w]
    let refrgb = flat(&r["rgb"]); // [3, 8h, 8w]
    let ae = std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../models/flux-schnell/ae.safetensors"
    ));

    let ctx = Arc::new(GpuContext::new().expect("gpu"));
    let vae = GpuVae::load(&ctx, ae, VaeConfig::default()).expect("load vae");
    let ours = vae.decode(&z, h, w); // GpuVae applies z/scaling+shift internally

    assert_eq!(
        ours.len(),
        refrgb.len(),
        "len {} vs {}",
        ours.len(),
        refrgb.len()
    );
    let (mut dot, mut na, mut nb, mut diff2, mut ref2) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
    for (a, b) in ours.iter().zip(refrgb.iter()) {
        let (a, b) = (*a as f64, *b as f64);
        dot += a * b;
        na += a * a;
        nb += b * b;
        diff2 += (a - b) * (a - b);
        ref2 += b * b;
    }
    let cos = dot / (na.sqrt() * nb.sqrt());
    let rel = (diff2 / ref2).sqrt();
    let ok = cos > 0.99 && rel < 0.1;
    println!(
        "VAE decode: cos={cos:.6} rel_l2={rel:.4}  {}",
        if ok { "PASS" } else { "FAIL" }
    );
    println!("FLUX P2 parity: {}", if ok { "GREEN ✓" } else { "RED ✗" });
    if !ok {
        std::process::exit(1);
    }
}
