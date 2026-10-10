//! Cache-aware, row-parallel, SIMD matmul kernels in safe Rust.
//!
//! Output rows are independent, so we split them across scoped threads writing
//! disjoint `chunks_mut` — no locking, no `unsafe`. Inner loops use `wide`'s
//! `f32x8` with fused multiply-add; build with `RUSTFLAGS="-C target-cpu=native"`
//! to widen those lanes to AVX2/AVX-512 on capable hardware.

use wide::f32x8;

use crate::dtype::bf16_to_f32;

const LANES: usize = 8;

/// Run `f(row_index, row_slice)` for each of `n_rows` output rows of length
/// `row_len`, in parallel when the workload is large enough.
fn par_rows<F>(out: &mut [f32], row_len: usize, n_rows: usize, f: F)
where
    F: Fn(usize, &mut [f32]) + Sync,
{
    debug_assert_eq!(out.len(), row_len * n_rows);
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    // Heuristic: only parallelize when there is enough work to amortize threads.
    if threads <= 1 || n_rows < 2 || row_len * n_rows < 1 << 14 {
        for (i, row) in out.chunks_mut(row_len).enumerate() {
            f(i, row);
        }
        return;
    }

    let rows_per = n_rows.div_ceil(threads);
    std::thread::scope(|scope| {
        for (t, block) in out.chunks_mut(rows_per * row_len).enumerate() {
            let base = t * rows_per;
            let f = &f;
            scope.spawn(move || {
                for (i, row) in block.chunks_mut(row_len).enumerate() {
                    f(base + i, row);
                }
            });
        }
    });
}

/// SIMD dot product of two equal-length slices.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = f32x8::ZERO;
    let mut ac = a.chunks_exact(LANES);
    let mut bc = b.chunks_exact(LANES);
    for (av, bv) in ac.by_ref().zip(bc.by_ref()) {
        let va = f32x8::from(<[f32; LANES]>::try_from(av).unwrap());
        let vb = f32x8::from(<[f32; LANES]>::try_from(bv).unwrap());
        acc = va.mul_add(vb, acc);
    }
    let mut sum = acc.reduce_add();
    for (x, y) in ac.remainder().iter().zip(bc.remainder()) {
        sum += x * y;
    }
    sum
}

/// SIMD `row += scalar * b_row`, both equal length.
#[inline]
fn axpy(row: &mut [f32], scalar: f32, b_row: &[f32]) {
    debug_assert_eq!(row.len(), b_row.len());
    let vs = f32x8::splat(scalar);
    let mut bc = b_row.chunks_exact(LANES);
    let mut rc = row.chunks_exact_mut(LANES);
    for (r, b) in rc.by_ref().zip(bc.by_ref()) {
        let vb = f32x8::from(<[f32; LANES]>::try_from(b).unwrap());
        let vr = f32x8::from(<[f32; LANES]>::try_from(&r[..]).unwrap());
        r.copy_from_slice(&vb.mul_add(vs, vr).to_array());
    }
    let tail = bc.remainder();
    for (r, &b) in rc.into_remainder().iter_mut().zip(tail) {
        *r += scalar * b;
    }
}

/// `a [m, k]` × `bᵀ` where `b` is `[n, k]` → `[m, n]`.
///
/// Both `a`'s and `b`'s rows are contiguous, so each output element is a SIMD dot
/// of two contiguous slices — the friendly memory pattern for a `Linear`.
pub fn matmul_nt(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "matmul_nt: a shape");
    assert_eq!(b.len(), n * k, "matmul_nt: b shape");
    let mut out = vec![0.0f32; m * n];
    par_rows(&mut out, n, m, |i, row| {
        let a_row = &a[i * k..(i + 1) * k];
        for (j, slot) in row.iter_mut().enumerate() {
            *slot = dot(a_row, &b[j * k..(j + 1) * k]);
        }
    });
    out
}

/// SIMD dot of an f32 slice with a bf16 slice (the weight), widening bf16 → f32
/// on the fly. Same length.
#[inline]
fn dot_bf16(x: &[f32], w: &[u16]) -> f32 {
    debug_assert_eq!(x.len(), w.len());
    let mut acc = f32x8::ZERO;
    let mut xc = x.chunks_exact(LANES);
    let mut wc = w.chunks_exact(LANES);
    for (xv, wv) in xc.by_ref().zip(wc.by_ref()) {
        let wf = crate::dtype::widen8(<[u16; LANES]>::try_from(wv).unwrap());
        let vx = f32x8::from(<[f32; LANES]>::try_from(xv).unwrap());
        acc = vx.mul_add(wf, acc);
    }
    let mut sum = acc.reduce_add();
    for (x, &w) in xc.remainder().iter().zip(wc.remainder()) {
        sum += x * bf16_to_f32(w);
    }
    sum
}

/// `a [m, k]` (f32) × `wᵀ` where `w` is `[n, k]` bf16 → `[m, n]` f32.
///
/// This is the `Linear` kernel for bf16-resident weights: weights stay at 2
/// bytes and are widened to f32 inside the dot product.
pub fn matmul_nt_bf16(a: &[f32], m: usize, k: usize, w: &[u16], n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "matmul_nt_bf16: a shape");
    assert_eq!(w.len(), n * k, "matmul_nt_bf16: w shape");
    let mut out = vec![0.0f32; m * n];
    par_rows(&mut out, n, m, |i, row| {
        let a_row = &a[i * k..(i + 1) * k];
        for (j, slot) in row.iter_mut().enumerate() {
            *slot = dot_bf16(a_row, &w[j * k..(j + 1) * k]);
        }
    });
    out
}

/// SIMD dot of an f32 activation slice with an int8 weight row, before scaling:
/// `Σ x[c] * (q[c] as f32)`. The caller multiplies by the row's scale.
#[inline]
fn dot_q8(x: &[f32], q: &[i8]) -> f32 {
    debug_assert_eq!(x.len(), q.len());
    let mut acc = f32x8::ZERO;
    let mut xc = x.chunks_exact(LANES);
    let mut qc = q.chunks_exact(LANES);
    for (xv, qv) in xc.by_ref().zip(qc.by_ref()) {
        let qf = f32x8::from(std::array::from_fn::<f32, LANES, _>(|i| f32::from(qv[i])));
        let vx = f32x8::from(<[f32; LANES]>::try_from(xv).unwrap());
        acc = vx.mul_add(qf, acc);
    }
    let mut sum = acc.reduce_add();
    for (x, &q) in xc.remainder().iter().zip(qc.remainder()) {
        sum += x * f32::from(q);
    }
    sum
}

/// `a [m, k]` (f32) × `wᵀ` where `w` is `[n, k]` per-row int8 with `scale [n]` →
/// `[m, n]` f32. The row scale factors out of the dot: `scale[j] · Σ a·q`.
pub fn matmul_nt_q8(a: &[f32], m: usize, k: usize, q: &[i8], scale: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "matmul_nt_q8: a shape");
    assert_eq!(q.len(), n * k, "matmul_nt_q8: q shape");
    assert_eq!(scale.len(), n, "matmul_nt_q8: scale len");
    let mut out = vec![0.0f32; m * n];
    par_rows(&mut out, n, m, |i, row| {
        let a_row = &a[i * k..(i + 1) * k];
        for (j, slot) in row.iter_mut().enumerate() {
            *slot = scale[j] * dot_q8(a_row, &q[j * k..(j + 1) * k]);
        }
    });
    out
}

/// `C[m,n] = A[m,k] · dequant(Q8_0)ᵀ` from RAW ggml **Q8_0** block bytes — the
/// per-32-block analog of [`matmul_nt_q8`] (which is per-row). `blocks` is the weight
/// `[n, k]` stored as native Q8_0: 34 bytes per 32 weights (f16 `d` then 32 i8 codes),
/// `w = d·q`. The bit-identical CPU oracle for the GPU `matmul_coop_q8_0` kernel.
/// `k % 32 == 0`.
pub fn matmul_nt_q8_0(a: &[f32], m: usize, k: usize, blocks: &[u8], n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "matmul_nt_q8_0: a shape");
    assert_eq!(k % 32, 0, "matmul_nt_q8_0: k must be a multiple of 32");
    let bpr = k / 32; // blocks per row
    assert_eq!(blocks.len(), n * bpr * 34, "matmul_nt_q8_0: blocks length");
    let mut out = vec![0.0f32; m * n];
    par_rows(&mut out, n, m, |i, row| {
        let a_row = &a[i * k..(i + 1) * k];
        for (j, slot) in row.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for blk in 0..bpr {
                let p = (j * bpr + blk) * 34;
                let d = crate::dtype::f16_to_f32(u16::from_le_bytes([blocks[p], blocks[p + 1]]));
                let qs = &blocks[p + 2..p + 34];
                let xb = blk * 32;
                let mut bsum = 0.0f32;
                for t in 0..32 {
                    bsum += a_row[xb + t] * (qs[t] as i8) as f32;
                }
                acc += d * bsum;
            }
            *slot = acc;
        }
    });
    out
}

/// Dot one Q4_0 weight row (`codes` packed 8 nibbles/u32, `scales` one bf16 per
/// 32-weight block) with an f32 activation row of length `k`. Each block's integer
/// sub-dot is scaled by that block's scale, then accumulated — the bit-identical
/// CPU oracle for the GPU `matmul_vec_q4` kernel. `k % 32 == 0`.
fn dot_q4(x: &[f32], codes: &[u32], scales: &[u16], k: usize) -> f32 {
    use crate::dtype::bf16_to_f32;
    use crate::weight::Q4Matrix;
    let blocks = k / 32;
    let mut sum = 0.0f32;
    for b in 0..blocks {
        let s = bf16_to_f32(scales[b]);
        let mut bsum = 0.0f32;
        // 32 weights = 4 u32 words; nibble j of word w is weight (w*8 + j).
        for w in 0..4 {
            let word = codes[b * 4 + w];
            let xb = b * 32 + w * 8;
            for j in 0..8 {
                let nibble = (word >> (j * 4)) & 0xF;
                bsum += x[xb + j] * Q4Matrix::decode_nibble(nibble) as f32;
            }
        }
        sum += s * bsum;
    }
    sum
}

/// `a [m, k]` (f32) × `wᵀ` where `w` is `[n, k]` Q4_0 (block-32 4-bit), `codes`
/// packed 8/u32 and `scales` one bf16 per block → `[m, n]` f32. The scale is
/// applied per block (it does not factor out of the whole row like int8's does).
pub fn matmul_nt_q4(
    a: &[f32],
    m: usize,
    k: usize,
    codes: &[u32],
    scales: &[u16],
    n: usize,
) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "matmul_nt_q4: a shape");
    assert_eq!(k % 32, 0, "matmul_nt_q4: k must be a multiple of 32");
    let words_per_row = k / 8;
    let blocks_per_row = k / 32;
    assert_eq!(codes.len(), n * words_per_row, "matmul_nt_q4: codes shape");
    assert_eq!(scales.len(), n * blocks_per_row, "matmul_nt_q4: scales len");
    let mut out = vec![0.0f32; m * n];
    par_rows(&mut out, n, m, |i, row| {
        let a_row = &a[i * k..(i + 1) * k];
        for (j, slot) in row.iter_mut().enumerate() {
            let cr = &codes[j * words_per_row..(j + 1) * words_per_row];
            let sr = &scales[j * blocks_per_row..(j + 1) * blocks_per_row];
            *slot = dot_q4(a_row, cr, sr, k);
        }
    });
    out
}

/// Dot one Q4_K-lite weight row with an f32 activation row of length `k`. Each
/// 32-weight sub-block has an asymmetric `(scale, min)` (bf16); a weight is
/// `code*scale + min` (code 0..15 UNSIGNED), so the sub-block contributes
/// `scale*Σ(code*x) + min*Σ(x)`. Bit-identical CPU oracle for `matmul_vec_q4k`.
/// `k % 32 == 0`.
fn dot_q4k(x: &[f32], codes: &[u32], scales: &[u16], mins: &[u16], k: usize) -> f32 {
    use crate::dtype::bf16_to_f32;
    let subs = k / 32;
    let mut sum = 0.0f32;
    for b in 0..subs {
        let s = bf16_to_f32(scales[b]);
        let lo = bf16_to_f32(mins[b]);
        let mut code_sum = 0.0f32; // Σ code*x
        let mut x_sum = 0.0f32; // Σ x  (carries the min offset)
        for w in 0..4 {
            let word = codes[b * 4 + w];
            let xb = b * 32 + w * 8;
            for j in 0..8 {
                let code = ((word >> (j * 4)) & 0xF) as f32;
                let xv = x[xb + j];
                code_sum += xv * code;
                x_sum += xv;
            }
        }
        sum += s * code_sum + lo * x_sum;
    }
    sum
}

/// `a [m, k]` (f32) × `wᵀ` where `w` is `[n, k]` Q4_K-lite (super-block-256, 4-bit
/// unsigned, per-32 asymmetric scale+min) → `[m, n]` f32.
#[allow(clippy::too_many_arguments)]
pub fn matmul_nt_q4k(
    a: &[f32],
    m: usize,
    k: usize,
    codes: &[u32],
    scales: &[u16],
    mins: &[u16],
    n: usize,
) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "matmul_nt_q4k: a shape");
    assert_eq!(k % 256, 0, "matmul_nt_q4k: k must be a multiple of 256");
    let words_per_row = k / 8;
    let subs_per_row = k / 32;
    assert_eq!(codes.len(), n * words_per_row, "matmul_nt_q4k: codes shape");
    assert_eq!(scales.len(), n * subs_per_row, "matmul_nt_q4k: scales len");
    assert_eq!(mins.len(), n * subs_per_row, "matmul_nt_q4k: mins len");
    let mut out = vec![0.0f32; m * n];
    par_rows(&mut out, n, m, |i, row| {
        let a_row = &a[i * k..(i + 1) * k];
        for (j, slot) in row.iter_mut().enumerate() {
            let cr = &codes[j * words_per_row..(j + 1) * words_per_row];
            let sr = &scales[j * subs_per_row..(j + 1) * subs_per_row];
            let mr = &mins[j * subs_per_row..(j + 1) * subs_per_row];
            *slot = dot_q4k(a_row, cr, sr, mr, k);
        }
    });
    out
}

/// Dot one Q4_K_S weight row with an f32 activation row. Sub-block `s` (32 weights)
/// reconstructs `scale = d·scale_u8`, `min = dmin·min_i8` from the super-block's
/// `d`/`dmin`, contributing `scale·Σ(code·x) + min·Σ(x)`. `k % 256 == 0`. Bit-
/// identical CPU oracle for `matmul_vec_q4ks`.
#[allow(clippy::too_many_arguments)]
fn dot_q4ks(
    x: &[f32],
    codes: &[u32],
    scales: &[u8],
    mins: &[i8],
    d: &[f32],
    dmin: &[f32],
    k: usize,
) -> f32 {
    let subs = k / 32;
    let subs_per_super = 8usize;
    let mut sum = 0.0f32;
    for sb in 0..subs {
        let sblk = sb / subs_per_super;
        let s = d[sblk] * scales[sb] as f32;
        let lo = dmin[sblk] * mins[sb] as f32;
        let mut code_sum = 0.0f32;
        let mut x_sum = 0.0f32;
        for w in 0..4 {
            let word = codes[sb * 4 + w];
            let xb = sb * 32 + w * 8;
            for j in 0..8 {
                let code = ((word >> (j * 4)) & 0xF) as f32;
                let xv = x[xb + j];
                code_sum += xv * code;
                x_sum += xv;
            }
        }
        sum += s * code_sum + lo * x_sum;
    }
    sum
}

/// `a [m, k]` × `wᵀ` where `w` is `[n, k]` Q4_K_S (super-block-256, u8/i8 two-level
/// scales) → `[m, n]` f32.
#[allow(clippy::too_many_arguments)]
pub fn matmul_nt_q4ks(
    a: &[f32],
    m: usize,
    k: usize,
    codes: &[u32],
    scales: &[u8],
    mins: &[i8],
    d: &[f32],
    dmin: &[f32],
    n: usize,
) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "matmul_nt_q4ks: a shape");
    assert_eq!(k % 256, 0, "matmul_nt_q4ks: k must be a multiple of 256");
    let words_per_row = k / 8;
    let subs_per_row = k / 32;
    let supers_per_row = k / 256;
    assert_eq!(
        codes.len(),
        n * words_per_row,
        "matmul_nt_q4ks: codes shape"
    );
    assert_eq!(scales.len(), n * subs_per_row, "matmul_nt_q4ks: scales len");
    assert_eq!(mins.len(), n * subs_per_row, "matmul_nt_q4ks: mins len");
    assert_eq!(d.len(), n * supers_per_row, "matmul_nt_q4ks: d len");
    let mut out = vec![0.0f32; m * n];
    par_rows(&mut out, n, m, |i, row| {
        let a_row = &a[i * k..(i + 1) * k];
        for (j, slot) in row.iter_mut().enumerate() {
            let cr = &codes[j * words_per_row..(j + 1) * words_per_row];
            let sr = &scales[j * subs_per_row..(j + 1) * subs_per_row];
            let mr = &mins[j * subs_per_row..(j + 1) * subs_per_row];
            let dr = &d[j * supers_per_row..(j + 1) * supers_per_row];
            let dmr = &dmin[j * supers_per_row..(j + 1) * supers_per_row];
            *slot = dot_q4ks(a_row, cr, sr, mr, dr, dmr, k);
        }
    });
    out
}

/// Dot one Q3_K weight row with an f32 activation row. 3-bit codes split into `ql`
/// (2 low bits, 16/u32) + `qh` (1 high bit, 32/u32), reconstructed as
/// `(qh<<2)|ql`; sub-block `s` (32 weights) reconstructs `scale = d·scale_u8`,
/// `min = dmin·min_i8` from the super-block, contributing
/// `scale·Σ(code·x) + min·Σ(x)`. `k % 256 == 0`. Bit-identical oracle for
/// `matmul_vec_q3k`. `ql`/`qh` are this ROW's slices (indices are row-local here).
#[allow(clippy::too_many_arguments)]
fn dot_q3k(
    x: &[f32],
    ql: &[u32],
    qh: &[u32],
    scales: &[u8],
    mins: &[i8],
    d: &[f32],
    dmin: &[f32],
    k: usize,
) -> f32 {
    let subs = k / 32;
    let subs_per_super = 8usize;
    let mut sum = 0.0f32;
    for sb in 0..subs {
        let sblk = sb / subs_per_super;
        let s = d[sblk] * scales[sb] as f32;
        let lo = dmin[sblk] * mins[sb] as f32;
        let mut code_sum = 0.0f32;
        let mut x_sum = 0.0f32;
        for j in 0..32 {
            let widx = sb * 32 + j; // row-local weight index
            let lo2 = (ql[widx / 16] >> ((widx % 16) * 2)) & 0x3;
            let hi1 = (qh[widx / 32] >> (widx % 32)) & 0x1;
            let code = ((hi1 << 2) | lo2) as f32;
            let xv = x[widx];
            code_sum += xv * code;
            x_sum += xv;
        }
        sum += s * code_sum + lo * x_sum;
    }
    sum
}

/// `a [m, k]` × `wᵀ` where `w` is `[n, k]` Q3_K (super-block-256, 3-bit split-plane,
/// u8/i8 two-level scales) → `[m, n]` f32.
#[allow(clippy::too_many_arguments)]
pub fn matmul_nt_q3k(
    a: &[f32],
    m: usize,
    k: usize,
    ql: &[u32],
    qh: &[u32],
    scales: &[u8],
    mins: &[i8],
    d: &[f32],
    dmin: &[f32],
    n: usize,
) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "matmul_nt_q3k: a shape");
    assert_eq!(k % 256, 0, "matmul_nt_q3k: k must be a multiple of 256");
    let ql_per_row = k / 16;
    let qh_per_row = k / 32;
    let subs_per_row = k / 32;
    let supers_per_row = k / 256;
    assert_eq!(ql.len(), n * ql_per_row, "matmul_nt_q3k: ql shape");
    assert_eq!(qh.len(), n * qh_per_row, "matmul_nt_q3k: qh shape");
    assert_eq!(scales.len(), n * subs_per_row, "matmul_nt_q3k: scales len");
    let mut out = vec![0.0f32; m * n];
    par_rows(&mut out, n, m, |i, row| {
        let a_row = &a[i * k..(i + 1) * k];
        for (j, slot) in row.iter_mut().enumerate() {
            let qlr = &ql[j * ql_per_row..(j + 1) * ql_per_row];
            let qhr = &qh[j * qh_per_row..(j + 1) * qh_per_row];
            let sr = &scales[j * subs_per_row..(j + 1) * subs_per_row];
            let mr = &mins[j * subs_per_row..(j + 1) * subs_per_row];
            let dr = &d[j * supers_per_row..(j + 1) * supers_per_row];
            let dmr = &dmin[j * supers_per_row..(j + 1) * supers_per_row];
            *slot = dot_q3k(a_row, qlr, qhr, sr, mr, dr, dmr, k);
        }
    });
    out
}

/// `a [m, k]` × `b [k, n]` → `[m, n]`.
///
/// Uses `i,l,j` loop order so the innermost loop is a SIMD axpy over a contiguous
/// row of `b` accumulating into a contiguous output row.
pub fn matmul(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "matmul: a shape");
    assert_eq!(b.len(), k * n, "matmul: b shape");
    let mut out = vec![0.0f32; m * n];
    par_rows(&mut out, n, m, |i, row| {
        let a_row = &a[i * k..(i + 1) * k];
        for (l, &av) in a_row.iter().enumerate() {
            axpy(row, av, &b[l * n..(l + 1) * n]);
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(&x, &y)| x * y).sum()
    }

    #[test]
    fn simd_dot_matches_scalar_including_tail() {
        for len in [0usize, 1, 7, 8, 9, 33, 100] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32).sin()).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.3).cos()).collect();
            assert!((dot(&a, &b) - naive_dot(&a, &b)).abs() < 1e-4, "len={len}");
        }
    }

    #[test]
    fn nt_and_standard_agree() {
        let (m, k, n) = (3, 13, 5); // k not a multiple of 8 -> exercises the tail
        let a: Vec<f32> = (0..m * k).map(|x| x as f32 * 0.5 - 1.0).collect();
        let b: Vec<f32> = (0..n * k).map(|x| (x as f32).sin()).collect();
        let nt = matmul_nt(&a, m, k, &b, n);

        let mut bt = vec![0.0f32; k * n];
        for r in 0..n {
            for c in 0..k {
                bt[c * n + r] = b[r * k + c];
            }
        }
        let std = matmul(&a, m, k, &bt, n);
        for (x, y) in nt.iter().zip(&std) {
            assert!((x - y).abs() < 1e-4);
        }
    }

    #[test]
    fn bf16_matmul_matches_f32_within_tolerance() {
        use crate::dtype::{bf16_to_f32, f32_to_bf16};
        let (m, k, n) = (4, 20, 6);
        let a: Vec<f32> = (0..m * k).map(|x| (x as f32 * 0.1).sin()).collect();
        let wf: Vec<f32> = (0..n * k).map(|x| (x as f32 * 0.2).cos()).collect();
        let wb: Vec<u16> = wf.iter().map(|&x| f32_to_bf16(x)).collect();
        // Reference uses the *rounded* weights so we isolate matmul correctness.
        let wf_rounded: Vec<f32> = wb.iter().map(|&b| bf16_to_f32(b)).collect();
        let got = matmul_nt_bf16(&a, m, k, &wb, n);
        let want = matmul_nt(&a, m, k, &wf_rounded, n);
        for (x, y) in got.iter().zip(&want) {
            assert!((x - y).abs() < 1e-4, "{x} vs {y}");
        }
    }

    #[test]
    fn parallel_path_matches_serial() {
        let (m, k, n) = (64, 40, 64);
        let a: Vec<f32> = (0..m * k).map(|x| (x % 7) as f32).collect();
        let b: Vec<f32> = (0..n * k).map(|x| (x % 5) as f32).collect();
        let par = matmul_nt(&a, m, k, &b, n);
        let mut serial = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                serial[i * n + j] = naive_dot(&a[i * k..(i + 1) * k], &b[j * k..(j + 1) * k]);
            }
        }
        for (x, y) in par.iter().zip(&serial) {
            assert!((x - y).abs() < 1e-3);
        }
    }
}
