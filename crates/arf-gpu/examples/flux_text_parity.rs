//! FLUX P0 parity gate: run CLIP-L + T5-XXL on the EXACT token ids from the transformers
//! reference (`/tmp/flux_oracle/reference.json`) and compare our GPU embeddings to the HF
//! reference (cosine + relative L2), the same gate that caught the vision projector-transpose.
//!
//! Usage: cargo run --release --example flux_text_parity [reference.json]

use std::sync::Arc;

use arf_gpu::gpu::flux::{ClipConfig, GpuClipText, GpuT5, T5Config};
use arf_gpu::gpu::GpuContext;
use serde_json::Value;

fn cos_rel(ours: &[f32], reference: &[f32]) -> (f64, f64) {
    assert_eq!(
        ours.len(),
        reference.len(),
        "len {} vs {}",
        ours.len(),
        reference.len()
    );
    let (mut dot, mut na, mut nb, mut diff2, mut ref2) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (a, b) in ours.iter().zip(reference.iter()) {
        let (a, b) = (*a as f64, *b as f64);
        dot += a * b;
        na += a * a;
        nb += b * b;
        diff2 += (a - b) * (a - b);
        ref2 += b * b;
    }
    (dot / (na.sqrt() * nb.sqrt()), (diff2 / ref2).sqrt())
}

fn main() {
    let ref_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/flux_oracle/reference.json".into());
    let r: Value =
        serde_json::from_reader(std::fs::File::open(&ref_path).expect("open reference.json"))
            .unwrap();
    let base = std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../models/flux-schnell"
    ));
    let ctx = Arc::new(GpuContext::new().expect("gpu"));
    let mut pass = true;

    // ---- CLIP-L pooled [768] ----
    {
        let c = &r["clip"];
        let toks: Vec<u32> = c["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let eos = c["eos_pos"].as_u64().unwrap() as usize;
        let refv: Vec<f32> = c["pooled"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let clip = GpuClipText::load(
            &ctx,
            &base.join("clip_l.safetensors"),
            ClipConfig::default(),
        )
        .expect("clip");
        let ours = clip.encode(&toks, eos);
        let (cos, rel) = cos_rel(&ours, &refv);
        let ok = cos > 0.999 && rel < 0.02;
        pass &= ok;
        println!(
            "CLIP pooled: cos={cos:.6} rel_l2={rel:.4}  {}",
            if ok { "PASS" } else { "FAIL" }
        );
    }

    // ---- T5-XXL sequence [seq, 4096] ----
    {
        let t = &r["t5"];
        let toks: Vec<u32> = t["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let rows = t["seq"].as_array().unwrap();
        let refv: Vec<f32> = rows
            .iter()
            .flat_map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
            })
            .collect();
        let t5 = GpuT5::load(&ctx, &base.join("t5xxl-Q8_0.gguf"), T5Config::default()).expect("t5");
        let ours = t5.encode(&toks);
        let (cos, rel) = cos_rel(&ours, &refv);
        // T5 is Q8 vs f16 reference → allow a slightly looser bar than CLIP (still tight).
        let ok = cos > 0.998 && rel < 0.05;
        pass &= ok;
        println!(
            "T5 seq:      cos={cos:.6} rel_l2={rel:.4}  {}",
            if ok { "PASS" } else { "FAIL" }
        );
    }

    println!(
        "\nFLUX P0 parity: {}",
        if pass { "GREEN ✓" } else { "RED ✗" }
    );
    if !pass {
        std::process::exit(1);
    }
}
