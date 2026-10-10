//! Scratch: dump qwen35 GGUF metadata keys + values to ground-truth M4 wiring.
use arf_core::model::gguf::LazyGguf;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        format!(
            "{}/models/qwen3.6-27b-mtp/Qwen3.6-27B-Q4_K_S.gguf",
            std::env::var("HOME").unwrap()
        )
    });
    let g = LazyGguf::open_raw(std::path::Path::new(&path)).expect("open");
    let mut keys = g.metadata_keys();
    keys.sort();
    for k in &keys {
        if let Some(v) = g.get_metadata_u32(k) {
            println!("{k} = u32 {v}");
        } else if let Some(v) = g.get_metadata_f32(k) {
            println!("{k} = f32 {v}");
        } else {
            println!("{k} = (other)");
        }
    }
    // a few raw tensor shapes
    for t in [
        "blk.0.attn_qkv.weight",
        "blk.0.attn_gate.weight",
        "blk.0.ssm_conv1d.weight",
        "blk.0.ssm_beta.weight",
        "blk.0.ssm_alpha.weight",
        "blk.0.ssm_dt.bias",
        "blk.0.ssm_a",
        "blk.0.ssm_norm.weight",
        "blk.0.ssm_out.weight",
        "blk.3.attn_q.weight",
        "blk.3.attn_k.weight",
        "blk.3.attn_v.weight",
        "blk.3.attn_output.weight",
        "blk.3.attn_q_norm.weight",
        "blk.3.attn_k_norm.weight",
        "token_embd.weight",
        "output.weight",
        "output_norm.weight",
    ] {
        println!("SHAPE {t} = {:?}", g.shape_raw(t));
    }
    println!("--- blk.0 + blk.3 tensor names ---");
    let mut names: Vec<&str> = g
        .tensor_names()
        .filter(|n| n.starts_with("blk.0.") || n.starts_with("blk.3."))
        .collect();
    names.sort();
    for n in names {
        println!("  {n} {:?}", g.shape_raw(n));
    }
}
