//! Exercise the real `load_safetensors` path: serialize a model to a
//! safetensors file, load it back, and assert it matches the in-memory build.

use std::collections::HashMap;

use arf_core::cache::{slots_for, write_runs, PagedKvCache};
use arf_core::config::{ModelConfig, RopeScaling};
use arf_core::model::weights::{build, load_safetensors, Weights};
use arf_core::model::{ForwardBatch, Llama, SeqAttn};
use arf_core::Tensor;
use safetensors::tensor::{Dtype, TensorView};

fn cfg() -> ModelConfig {
    ModelConfig {
        nextn_layers: 0,
        vocab_size: 32,
        hidden_size: 8,
        intermediate_size: 16,
        num_layers: 1,
        num_attention_heads: 2,
        num_kv_heads: 1,
        head_dim: 4,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        rope_scaling: RopeScaling::None,
        max_position_embeddings: 64,
        tie_word_embeddings: true,
        mlp: arf_core::config::MlpKind::Dense,
        qk_norm: false,
        norm_style: arf_core::config::NormStyle::Llama,
        attn: arf_core::config::AttnKind::Causal,
        embedding_scale: None,
        query_pre_attn_scalar: None,
        final_logit_softcap: None,
        value_norm: false,
        gate_act: arf_core::config::GateAct::Silu,
    }
}

/// Named (shape, f32-data) pairs for a complete tiny checkpoint.
fn raw_weights(cfg: &ModelConfig) -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let h = cfg.hidden_size;
    let q = cfg.num_attention_heads * cfg.head_dim;
    let kv = cfg.num_kv_heads * cfg.head_dim;
    let mut seed = 99u64;
    let mut gen = |shape: Vec<usize>| {
        let n: usize = shape.iter().product();
        let data = (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                (seed >> 40) as f32 / (1u64 << 23) as f32 - 0.5
            })
            .collect();
        (shape, data)
    };
    let mut out = Vec::new();
    let push = |out: &mut Vec<(String, Vec<usize>, Vec<f32>)>,
                name: String,
                sd: (Vec<usize>, Vec<f32>)| {
        out.push((name, sd.0, sd.1));
    };
    push(
        &mut out,
        "model.embed_tokens.weight".into(),
        gen(vec![cfg.vocab_size, h]),
    );
    let p = "model.layers.0";
    push(
        &mut out,
        format!("{p}.input_layernorm.weight"),
        gen(vec![h]),
    );
    push(
        &mut out,
        format!("{p}.post_attention_layernorm.weight"),
        gen(vec![h]),
    );
    push(
        &mut out,
        format!("{p}.self_attn.q_proj.weight"),
        gen(vec![q, h]),
    );
    push(
        &mut out,
        format!("{p}.self_attn.k_proj.weight"),
        gen(vec![kv, h]),
    );
    push(
        &mut out,
        format!("{p}.self_attn.v_proj.weight"),
        gen(vec![kv, h]),
    );
    push(
        &mut out,
        format!("{p}.self_attn.o_proj.weight"),
        gen(vec![h, q]),
    );
    push(
        &mut out,
        format!("{p}.mlp.gate_proj.weight"),
        gen(vec![cfg.intermediate_size, h]),
    );
    push(
        &mut out,
        format!("{p}.mlp.up_proj.weight"),
        gen(vec![cfg.intermediate_size, h]),
    );
    push(
        &mut out,
        format!("{p}.mlp.down_proj.weight"),
        gen(vec![h, cfg.intermediate_size]),
    );
    push(&mut out, "model.norm.weight".into(), gen(vec![h]));
    out
}

fn last_logits(model: &Llama, tokens: &[u32]) -> Vec<f32> {
    let mut cache = PagedKvCache::new(
        model.num_layers(),
        4,
        4,
        model.config().num_kv_heads,
        model.config().head_dim,
    );
    let q_len = tokens.len();
    let block_table = [0u32, 1];
    let batch = ForwardBatch {
        positions: (0..q_len as u32).collect(),
        seqs: vec![SeqAttn {
            stream_id: None,
            q_start: 0,
            q_len,
            past_len: 0,
            slots: slots_for(&block_table, 4, q_len),
            write_runs: write_runs(&block_table, 4, 0, q_len),
            image_spans: Vec::new(),
        }],
        image_embeds: None,
        mrope_positions: None,
    };
    let hidden = model.forward(tokens, &batch, &mut cache);
    model.logits_last(&hidden, &batch).into_vec()
}

#[test]
fn safetensors_roundtrip_matches_in_memory() {
    let cfg = cfg();
    let raw = raw_weights(&cfg);

    // In-memory model.
    let map: HashMap<String, Tensor> = raw
        .iter()
        .map(|(n, shape, data)| (n.clone(), Tensor::from_vec(data.clone(), shape.clone())))
        .collect();
    let mem = build(&cfg, &Weights::from_map(map), cfg.max_position_embeddings).unwrap();

    // Serialize the same weights to a safetensors file (f32, little-endian).
    let byte_store: Vec<(String, Vec<usize>, Vec<u8>)> = raw
        .iter()
        .map(|(n, shape, data)| {
            let bytes = data.iter().flat_map(|x| x.to_le_bytes()).collect();
            (n.clone(), shape.clone(), bytes)
        })
        .collect();
    let views: Vec<(String, TensorView)> = byte_store
        .iter()
        .map(|(n, shape, bytes)| {
            (
                n.clone(),
                TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
            )
        })
        .collect();
    let blob = safetensors::serialize(views, None).unwrap();

    let path = std::env::temp_dir().join(format!("arf_test_{}.safetensors", std::process::id()));
    std::fs::write(&path, &blob).unwrap();
    let loaded = load_safetensors(&cfg, &[&path], cfg.max_position_embeddings).unwrap();
    std::fs::remove_file(&path).ok();

    let tokens = [3u32, 1, 4, 1, 5];
    assert_eq!(
        last_logits(&mem, &tokens),
        last_logits(&loaded, &tokens),
        "loaded model must match the in-memory build bit-for-bit"
    );
}
