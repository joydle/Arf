//! Verify the T5 Q8_0 loader split is bit-correct: for one real T5 weight, compare
//! (a) get_f32_raw (full dequant, what the bf16 path uses) against
//! (b) the per-block dequant reconstructed from get_q8_0_blocks_raw the way the GPU loader
//!     splits it (codes i8 + per-block f16 scale → w = d·q).
//! If these match, the loader split + f16 read are correct and any A/B cosine gap is pure
//! Q8_0-vs-bf16 precision (the bf16 path ALSO quantizes, to a different scheme).
//!
//! Run: cargo run --release -p arf-core --example t5_q8_split_check

use arf_core::model::gguf::{f16_to_f32, LazyGguf};

const T5: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/flux-schnell/t5xxl-Q8_0.gguf"
);

fn main() {
    let g = LazyGguf::open_raw(std::path::Path::new(T5)).expect("open t5");
    let name = "enc.blk.0.attn_q.weight";

    let f32_ref = g.get_f32_raw(name).expect("f32 raw"); // ggml dequant_q8_0
    let (blocks, rows, cols) = g.get_q8_0_blocks_raw(name).expect("q8_0 blocks");
    println!(
        "weight {name}  [{rows} x {cols}]  ({} f32 elems)",
        f32_ref.len()
    );
    assert_eq!(rows * cols, f32_ref.len());

    // Reconstruct exactly as the GPU loader + kernel do: w[r,c] = dscale[block] * code.
    let bpr = cols / 32;
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for r in 0..rows {
        for blk in 0..bpr {
            let p = (r * bpr + blk) * 34;
            let d = f16_to_f32(u16::from_le_bytes([blocks[p], blocks[p + 1]]));
            for t in 0..32 {
                let code = blocks[p + 2 + t] as i8 as f32;
                let recon = d * code;
                let idx = r * cols + blk * 32 + t;
                let refv = f32_ref[idx];
                let ae = (recon - refv).abs();
                max_abs = max_abs.max(ae);
                max_rel = max_rel.max(ae / refv.abs().max(1e-6));
            }
        }
    }
    println!("split-vs-ggml-dequant  max_abs_err={max_abs:.3e}  max_rel_err={max_rel:.3e}");
    if max_abs < 1e-4 {
        println!("PASS — loader split is bit-correct; A/B gap is pure Q8_0-vs-bf16 precision.");
    } else {
        println!(
            "FAIL — loader split disagrees with ggml dequant → real bug in the split/f16 read."
        );
        std::process::exit(1);
    }
}
