use arf_core::tensor::{Q3KMatrix, Q4KMatrix, Q4KSMatrix, Q4Matrix};
use arf_gpu::weights::{
    qkv_concat_bf16, qkv_concat_q3k, qkv_concat_q4, qkv_concat_q4k, qkv_concat_q4ks,
};

#[test]
fn qkv_concat_bf16_stacks_rows() {
    // q: 2 rows x 4 cols, k: 1 row x 4, v: 1 row x 4 — bf16 = 1 u16/weight.
    let q: Vec<u16> = (0..8).collect(); // rows 0,1
    let k: Vec<u16> = (100..104).collect(); // row 0
    let v: Vec<u16> = (200..204).collect(); // row 0
    let fused = qkv_concat_bf16(&q, &k, &v);
    assert_eq!(fused.len(), 16);
    assert_eq!(&fused[0..8], &q[..]); // q rows first
    assert_eq!(&fused[8..12], &k[..]); // then k
    assert_eq!(&fused[12..16], &v[..]); // then v
}

#[test]
fn qkv_concat_q4_stacks_subbuffers() {
    // cols % 32 == 0 for Q4_0 blocks; GQA dims (q_rows != kv_rows).
    let (q_rows, kv_rows, in_) = (4usize, 2usize, 64usize);
    let qf: Vec<f32> = (0..q_rows * in_).map(|i| (i as f32 % 7.0) - 3.0).collect();
    let kf: Vec<f32> = (0..kv_rows * in_).map(|i| (i as f32 % 5.0) - 2.0).collect();
    let vf: Vec<f32> = (0..kv_rows * in_).map(|i| (i as f32 % 3.0) - 1.0).collect();
    let q = Q4Matrix::from_f32(&qf, q_rows, in_);
    let k = Q4Matrix::from_f32(&kf, kv_rows, in_);
    let v = Q4Matrix::from_f32(&vf, kv_rows, in_);
    let (codes, scales) = qkv_concat_q4(&q, &k, &v);
    assert_eq!(codes, [q.codes(), k.codes(), v.codes()].concat());
    assert_eq!(scales, [q.scales(), k.scales(), v.scales()].concat());
}

#[test]
fn qkv_concat_q4k_stacks_subbuffers() {
    // cols % 256 == 0 for Q4_K super-blocks; GQA dims.
    let (q_rows, kv_rows, in_) = (4usize, 2usize, 256usize);
    let qf: Vec<f32> = (0..q_rows * in_).map(|i| (i as f32 % 7.0) - 3.0).collect();
    let kf: Vec<f32> = (0..kv_rows * in_).map(|i| (i as f32 % 5.0) - 2.0).collect();
    let vf: Vec<f32> = (0..kv_rows * in_).map(|i| (i as f32 % 3.0) - 1.0).collect();
    let q = Q4KMatrix::from_f32(&qf, q_rows, in_);
    let k = Q4KMatrix::from_f32(&kf, kv_rows, in_);
    let v = Q4KMatrix::from_f32(&vf, kv_rows, in_);
    let (codes, scales, mins) = qkv_concat_q4k(&q, &k, &v);
    assert_eq!(codes, [q.codes(), k.codes(), v.codes()].concat());
    assert_eq!(scales, [q.scales(), k.scales(), v.scales()].concat());
    assert_eq!(mins, [q.mins(), k.mins(), v.mins()].concat());
}

#[test]
fn qkv_concat_q4ks_stacks_subbuffers() {
    // in_ % 256 == 0 for Q4_K_S super-blocks; q_rows != kv_rows (GQA).
    let (q_rows, kv_rows, in_) = (4usize, 2usize, 256usize);
    let qf: Vec<f32> = (0..q_rows * in_).map(|i| (i as f32 % 7.0) - 3.0).collect();
    let kf: Vec<f32> = (0..kv_rows * in_).map(|i| (i as f32 % 5.0) - 2.0).collect();
    let vf: Vec<f32> = (0..kv_rows * in_).map(|i| (i as f32 % 3.0) - 1.0).collect();
    let q = Q4KSMatrix::from_f32(&qf, q_rows, in_);
    let k = Q4KSMatrix::from_f32(&kf, kv_rows, in_);
    let v = Q4KSMatrix::from_f32(&vf, kv_rows, in_);
    let (codes, scales, mins, dd) = qkv_concat_q4ks(&q, &k, &v);
    assert_eq!(codes, [q.codes(), k.codes(), v.codes()].concat());
    assert_eq!(scales, [q.scales(), k.scales(), v.scales()].concat());
    assert_eq!(mins, [q.mins(), k.mins(), v.mins()].concat());
    let interleave = |m: &Q4KSMatrix| {
        let mut d = Vec::new();
        for (a, b) in m.d().iter().zip(m.dmin()) {
            d.push(*a);
            d.push(*b);
        }
        d
    };
    assert_eq!(
        dd,
        [interleave(&q), interleave(&k), interleave(&v)].concat()
    );
}

#[test]
fn qkv_concat_q3k_stacks_subbuffers() {
    // in_ % 256 == 0 for Q3_K super-blocks; q_rows != kv_rows (GQA).
    let (q_rows, kv_rows, in_) = (4usize, 2usize, 256usize);
    let qf: Vec<f32> = (0..q_rows * in_).map(|i| (i as f32 % 7.0) - 3.0).collect();
    let kf: Vec<f32> = (0..kv_rows * in_).map(|i| (i as f32 % 5.0) - 2.0).collect();
    let vf: Vec<f32> = (0..kv_rows * in_).map(|i| (i as f32 % 3.0) - 1.0).collect();
    let q = Q3KMatrix::from_f32(&qf, q_rows, in_);
    let k = Q3KMatrix::from_f32(&kf, kv_rows, in_);
    let v = Q3KMatrix::from_f32(&vf, kv_rows, in_);
    let (ql, qh, scales, mins, dd) = qkv_concat_q3k(&q, &k, &v);
    assert_eq!(ql, [q.ql(), k.ql(), v.ql()].concat());
    assert_eq!(qh, [q.qh(), k.qh(), v.qh()].concat());
    assert_eq!(scales, [q.scales(), k.scales(), v.scales()].concat());
    assert_eq!(mins, [q.mins(), k.mins(), v.mins()].concat());
    let interleave = |m: &Q3KMatrix| {
        let mut d = Vec::new();
        for (a, b) in m.d().iter().zip(m.dmin()) {
            d.push(*a);
            d.push(*b);
        }
        d
    };
    assert_eq!(
        dd,
        [interleave(&q), interleave(&k), interleave(&v)].concat()
    );
}
