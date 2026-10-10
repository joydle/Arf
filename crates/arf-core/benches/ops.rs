//! Micro-benchmarks for the hot ops.

use arf_core::config::ModelConfig;
use arf_core::model::rms_norm::RmsNorm;
use arf_core::model::Rope;
use arf_core::tensor::Bf16Matrix;
use arf_core::Tensor;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

fn filled(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i % 97) as f32) * 0.01 - 0.5).collect()
}

fn bench_rms_norm(c: &mut Criterion) {
    let hidden = 2048;
    let norm = RmsNorm::new(Tensor::ones(vec![hidden]), 1e-5);
    let mut group = c.benchmark_group("rms_norm");
    for &tokens in &[1usize, 128, 512] {
        let x = Tensor::from_vec(filled(tokens * hidden), vec![tokens, hidden]);
        group.bench_with_input(BenchmarkId::from_parameter(tokens), &tokens, |b, _| {
            b.iter(|| norm.forward(black_box(&x)));
        });
    }
    group.finish();
}

fn bench_rope(c: &mut Criterion) {
    let cfg = ModelConfig::llama_3_2_1b();
    let rope = Rope::new(&cfg, 4096);
    let heads = cfg.num_attention_heads;
    let mut group = c.benchmark_group("rope");
    for &tokens in &[1usize, 128, 512] {
        let x = Tensor::from_vec(
            filled(tokens * heads * cfg.head_dim),
            vec![tokens, heads, cfg.head_dim],
        );
        let pos: Vec<u32> = (0..tokens as u32).collect();
        group.bench_with_input(BenchmarkId::from_parameter(tokens), &tokens, |b, _| {
            b.iter(|| rope.apply(black_box(&x), black_box(&pos)));
        });
    }
    group.finish();
}

fn bench_linear(c: &mut Criterion) {
    let hidden = 2048;
    // bf16-resident weight, widened to f32 in the matmul (the real Linear path).
    let w = Bf16Matrix::from_f32(&filled(hidden * hidden), hidden, hidden);
    let mut group = c.benchmark_group("linear_proj");
    for &tokens in &[1usize, 128, 512] {
        let x = Tensor::from_vec(filled(tokens * hidden), vec![tokens, hidden]);
        group.bench_with_input(BenchmarkId::from_parameter(tokens), &tokens, |b, _| {
            b.iter(|| w.linear(black_box(&x)));
        });
    }
    group.finish();
}

fn bench_linear_q8(c: &mut Criterion) {
    use arf_core::tensor::Q8Matrix;
    let hidden = 2048;
    let w = Q8Matrix::from_f32(&filled(hidden * hidden), hidden, hidden);
    let mut group = c.benchmark_group("linear_proj_q8");
    for &tokens in &[1usize, 128, 512] {
        let x = Tensor::from_vec(filled(tokens * hidden), vec![tokens, hidden]);
        group.bench_with_input(BenchmarkId::from_parameter(tokens), &tokens, |b, _| {
            b.iter(|| w.linear(black_box(&x)));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_rms_norm,
    bench_rope,
    bench_linear,
    bench_linear_q8
);
criterion_main!(benches);
