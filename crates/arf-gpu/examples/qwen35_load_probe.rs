//! M1 GATE: load the Qwen3.6-27B (`qwen35`) HYBRID gated-delta-net trunk from its GGUF and
//! verify it loaded resident at the correct shapes — NO forward (the gated-delta-net kernels
//! are M2-M4). Prints the hparams + the per-layer-type counts (linear vs full GQA) and asserts
//! a few representative tensor shapes are present + correctly shaped + finite.
//!
//! Run: cargo run --release -p arf-gpu --example qwen35_load_probe -- [gguf]

use std::sync::Arc;

use arf_core::config::ModelConfig;
use arf_core::model::gguf::LazyGguf;
use arf_gpu::gpu::ssm_qwen35::{is_recurrent, load_qwen35, HybridLayer};
use arf_gpu::gpu::GpuContext;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        format!(
            "{}/models/qwen3.6-27b-mtp/Qwen3.6-27B-Q4_K_S.gguf",
            std::env::var("HOME").unwrap()
        )
    });
    println!("opening GGUF: {path}");

    let g = LazyGguf::open_raw(std::path::Path::new(&path)).expect("open_raw");
    let cfg = ModelConfig::qwen35_27b();
    let ctx = Arc::new(GpuContext::new().expect("gpu"));

    let t0 = std::time::Instant::now();
    let trunk = load_qwen35(&ctx, &g, &cfg).expect("load_qwen35");
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let hp = &trunk.hparams;
    println!("\n=== HPARAMS (read from GGUF metadata) ===");
    println!("  n_embd                  = {}", hp.n_embd);
    println!("  n_ff                    = {}", hp.n_ff);
    println!("  n_head                  = {}", hp.n_head);
    println!("  n_head_kv               = {}", hp.n_head_kv);
    println!("  n_embd_head_k           = {}", hp.n_embd_head_k);
    println!("  n_embd_head_v           = {}", hp.n_embd_head_v);
    println!("  n_layer (block_count)   = {}", hp.n_layer);
    println!("  n_layer_nextn           = {}", hp.n_layer_nextn);
    println!("  n_vocab                 = {}", hp.n_vocab);
    println!("  rms_eps                 = {:e}", hp.rms_eps);
    println!("  full_attention_interval = {}", hp.full_attention_interval);
    println!("  ssm_d_conv              = {}", hp.ssm_d_conv);
    println!("  ssm_d_inner             = {}", hp.ssm_d_inner);
    println!("  ssm_d_state             = {}", hp.ssm_d_state);
    println!("  ssm_dt_rank             = {}", hp.ssm_dt_rank);
    println!("  ssm_n_group             = {}", hp.ssm_n_group);

    let geom = hp.linear_geom();
    println!("\n=== DERIVED gated-delta-net geometry ===");
    println!(
        "  head_k_dim = {}, head_v_dim = {}",
        geom.head_k_dim, geom.head_v_dim
    );
    println!(
        "  num_k_heads = {}, num_v_heads = {}",
        geom.num_k_heads, geom.num_v_heads
    );
    println!(
        "  key_dim = {}, value_dim = {}, conv_dim = {}, d_conv = {}",
        geom.key_dim, geom.value_dim, geom.conv_dim, geom.d_conv
    );

    // ---- per-layer-type counts ----
    let n_linear = trunk.layers.iter().filter(|l| l.is_linear()).count();
    let n_full = trunk.layers.len() - n_linear;
    let n_states = trunk.states.iter().filter(|s| s.is_some()).count();
    println!("\n=== TRUNK LAYER COUNTS ===");
    println!(
        "  total trunk layers = {} (n_layer {} - nextn {})",
        trunk.layers.len(),
        hp.n_layer,
        hp.n_layer_nextn
    );
    println!("  LINEAR (gated-delta-net) = {n_linear}");
    println!("  FULL   (GQA)             = {n_full}");
    println!(
        "  recurrent states alloc'd = {n_states} (== n_linear: {})",
        n_states == n_linear
    );

    // Cross-check the predicate produced the expected pattern.
    let expected_full: usize = (0..trunk.layers.len())
        .filter(|&il| !is_recurrent(il, hp.full_attention_interval))
        .count();
    assert_eq!(
        n_full, expected_full,
        "full-layer count mismatch vs is_recurrent"
    );
    assert_eq!(
        n_states, n_linear,
        "state alloc must be 1:1 with linear layers"
    );
    // Layer 0,1,2 linear, 3 full (interval 4).
    assert!(
        matches!(trunk.layers.first(), Some(HybridLayer::Linear(_))),
        "layer 0 must be linear"
    );
    assert!(
        matches!(trunk.layers.get(3), Some(HybridLayer::Gqa(_))),
        "layer 3 must be full GQA"
    );

    // ---- assert representative tensor SHAPES (raw GGUF, non-zero + correct) ----
    println!("\n=== TENSOR SHAPE ASSERTS (raw GGUF dims [in, out]) ===");
    let check = |name: &str, want: &[usize]| {
        let shape = g
            .shape_raw(name)
            .unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(shape, want, "{name}: shape {shape:?} != expected {want:?}");
        let buflen: usize = shape.iter().product();
        assert!(buflen > 0, "{name} empty");
        println!("  {name:35} shape={shape:?} OK");
    };
    // linear layer 0
    check(
        "blk.0.attn_qkv.weight",
        &[hp.n_embd, geom.key_dim * 2 + geom.value_dim],
    );
    check("blk.0.attn_gate.weight", &[hp.n_embd, geom.value_dim]);
    check("blk.0.ssm_conv1d.weight", &[geom.d_conv, geom.conv_dim]);
    check("blk.0.ssm_out.weight", &[geom.value_dim, hp.n_embd]);
    check("blk.0.ssm_a", &[geom.num_v_heads]);
    check("blk.0.ssm_norm.weight", &[geom.head_v_dim]);
    // full layer 3
    check(
        "blk.3.attn_q.weight",
        &[hp.n_embd, hp.n_embd_head_k * hp.n_head * 2],
    );
    check(
        "blk.3.attn_k.weight",
        &[hp.n_embd, hp.n_embd_head_k * hp.n_head_kv],
    );
    check(
        "blk.3.attn_output.weight",
        &[hp.n_embd_head_k * hp.n_head, hp.n_embd],
    );
    // dense FFN (both layer kinds)
    check("blk.0.ffn_gate.weight", &[hp.n_embd, hp.n_ff]);
    check("blk.0.ffn_down.weight", &[hp.n_ff, hp.n_embd]);

    println!("\nM1 GATE PASS: trunk loaded resident, shapes verified, no panic/NaN. (load {load_ms:.0} ms)");
}
