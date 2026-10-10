//! Ground-truth probe: load a real DFlash draft-model GGUF (converted via llama.cpp's
//! own `convert_hf_to_gguf.py --target-model-dir <target>`) and print its DFlash-specific
//! metadata using the int-array readers added in the previous phase
//! (`get_metadata_i32_array` / `get_metadata_u32`). No forward pass — this is Phase 2b's
//! "does our GGUF reader correctly parse a real DFlash checkpoint's metadata" gate.
//!
//! Per the port plan, the GGUF conversion writes:
//!  - `dflash.target_layers`      (i32 array) — target-model tap layer indices, already
//!    `+1`-adjusted by the converter from the checkpoint's
//!    0-based `dflash_config.target_layer_ids`.
//!  - `dflash.block_size`         (u32) — draft block length (proposes up to block_size-1
//!    tokens per step).
//!  - `tokenizer.ggml.mask_token_id` (u32) — the MASK token id filled into un-drafted
//!    block slots.
//!
//! The design doc's own worked example was for the 30B-A3B target (48 layers, taps
//! [1,12,23,34,45]) — this probe intentionally runs against the much smaller
//! `z-lab/Qwen3-4B-DFlash-b16` / `Qwen/Qwen3-4B` pair (36-layer target), so the exact tap
//! indices are expected to differ. We assert only general sanity (some int-array metadata
//! parses, values are in-range for the target's layer count), not the literal 30B numbers.
//!
//! Run: cargo run --release -p arf-core --example dflash_probe -- <draft.gguf>

use arf_core::model::gguf::LazyGguf;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        std::env::temp_dir()
            .join("dflash-ckpt-draft")
            .join("Qwen3-4B-DFlash-b16.gguf")
            .to_string_lossy()
            .into_owned()
    });
    let g = LazyGguf::open_raw(std::path::Path::new(&path)).expect("open DFlash draft gguf");

    println!("=== DFlash draft GGUF probe ({path}) ===\n");

    // --- DFlash-specific metadata (the whole point of this phase) ---
    let target_layers = g.get_metadata_i32_array("dflash.target_layers");
    let block_size = g.get_metadata_u32("dflash.block_size");
    let mask_token_id = g.get_metadata_u32("tokenizer.ggml.mask_token_id");

    println!("dflash.target_layers      = {target_layers:?}");
    println!("dflash.block_size         = {block_size:?}");
    println!("tokenizer.ggml.mask_token_id = {mask_token_id:?}");

    // --- generic architecture metadata, for sanity/context ---
    let n_embd = g.get_metadata_u32("dflash.embedding_length");
    let n_layer = g.get_metadata_u32("dflash.block_count");
    let n_head = g.get_metadata_u32("dflash.attention.head_count");
    let n_head_kv = g.get_metadata_u32("dflash.attention.head_count_kv");
    let n_ff = g.get_metadata_u32("dflash.feed_forward_length");
    let rope_theta = g.get_metadata_f32("dflash.rope.freq_base");
    let rms_eps = g.get_metadata_f32("dflash.attention.layer_norm_rms_epsilon");

    println!("\ndflash.embedding_length         = {n_embd:?}");
    println!("dflash.block_count              = {n_layer:?}");
    println!("dflash.attention.head_count     = {n_head:?}");
    println!("dflash.attention.head_count_kv  = {n_head_kv:?}");
    println!("dflash.feed_forward_length      = {n_ff:?}");
    println!("dflash.rope.freq_base           = {rope_theta:?}");
    println!("dflash.attention.layer_norm_rms_epsilon = {rms_eps:?}");

    // --- tensor inventory, to confirm the encoder-fuse tensors (fc / hidden_norm) exist
    // alongside the ordinary per-layer dense Qwen3 stack ---
    let names: Vec<String> = g.tensor_names().map(|s| s.to_string()).collect();
    println!("\ntotal tensors: {}", names.len());
    let fc_tensors: Vec<&String> = names.iter().filter(|n| n.contains("fc")).collect();
    let enc_norm_tensors: Vec<&String> = names
        .iter()
        .filter(|n| n.contains("hidden_norm") || n.contains("enc"))
        .collect();
    println!("fc-like tensors:        {fc_tensors:?}");
    println!("encoder-norm tensors:   {enc_norm_tensors:?}");

    let blk0: Vec<&String> = names.iter().filter(|n| n.starts_with("blk.0.")).collect();
    println!("\nlayer-0 tensors ({}):", blk0.len());
    for n in &blk0 {
        if let Some(shape) = g.shape_raw(n) {
            println!("  {n}  {shape:?}");
        }
    }

    // --- sanity assertions: SOME sane int-array metadata parses, values plausible.
    // Not asserting the design doc's literal 30B numbers ([1,12,23,34,45] for a 48-layer
    // target) — this pair's target (Qwen3-4B) has a different layer count, so tap indices
    // are expected to differ. We only assert shape-level sanity.
    let target_layers = target_layers.expect("dflash.target_layers must parse as an i32 array");
    assert!(
        !target_layers.is_empty(),
        "target_layers must be non-empty (checkpoint taps at least one target layer)"
    );
    assert!(
        target_layers.windows(2).all(|w| w[0] < w[1]),
        "target_layers should be strictly increasing tap depths, got {target_layers:?}"
    );
    // Converter does target_layer_ids[i] + 1, so raw stored values are >= 1.
    assert!(
        target_layers.iter().all(|&x| x >= 1),
        "target_layers should all be >=1 after the converter's +1 adjustment, got {target_layers:?}"
    );

    let block_size = block_size.expect("dflash.block_size must parse as a u32");
    assert!(
        block_size >= 2,
        "block_size must allow at least [id_last, MASK] (>=2), got {block_size}"
    );

    let mask_token_id = mask_token_id.expect("tokenizer.ggml.mask_token_id must parse as a u32");
    println!("\nOK: parsed {} target-layer taps, block_size={}, mask_token_id={}, {} total tensors ({} layer-0 tensors, {} fc-like, {} encoder-norm)",
        target_layers.len(), block_size, mask_token_id, names.len(), blk0.len(), fc_tensors.len(), enc_norm_tensors.len());
}
