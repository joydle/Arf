//! DFlash 2 draft, stage 4 — the BLOCK FORWARD.
//!
//! One pass over the draft's 5 layers proposes a block: row 0 is the anchor (the pending token,
//! at position `ctx_len`), rows 1..8 are `mask_token_id`; row i's logits predict the token AT
//! position `ctx_len + i`. The block attends — not causally — over its own 8 K/V plus the context
//! ring stage 3 maintains. Embedding and lm_head are the TARGET's.
//!
//! All nine projections of a layer run on the target's 8-row MPP Q4 unit; siblings that read the
//! same input share one narrowing (q/k/v, gate/up). One serial command buffer, committed and
//! WAITED on: the proposals are needed on the host to build the verify window, so the draft is
//! synchronous by nature and no in-flight uniform hazard exists here.

use super::dflash2::Dflash2Config;
use super::dflash2_ctx::Dflash2State;
use super::island::{IslandLmHead, MetalIsland, MtlBuf, Q4ksMtl, Q4TILE_ROWS};
use crate::gpu::GpuContext;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer as _, MTLCommandEncoder as _, MTLCommandQueue as _,
    MTLComputeCommandEncoder as _, MTLDispatchType, MTLResourceUsage, MTLSize,
};

/// Rows of a block. The MPP unit and the published `block_size` are both 8; attach refuses a
/// draft whose block is anything else.
pub const DFLASH_BLOCK_ROWS: usize = Q4TILE_ROWS;

/// THE DRAFT'S CANDIDATE RANGE (2026-09-25). The draft lm_head is its largest stage (~3.3 ms of a
/// ~7.6 ms block at 248,320 rows), and the target verifies every proposal — so narrowing what the
/// draft may PROPOSE can cost acceptance, never correctness. MEASURED (bicycle / code edit /
/// Chinese, 8-row verifies for the same output): a fixed 65,536-id range kept acceptance on English
/// (153 = 153) and code (76 vs 75) and cut the draft 7.66 -> 5.76 ms, but COLLAPSED Chinese
/// (108 -> 163 verifies, 9.3 -> 13.6 s). So the range ADAPTS: it covers every token id this
/// sequence has committed (prompt and output), floored at 65,536, stepped by DFLASH_VOCAB_STEPS.
/// `ARF_NO_DFLASH_VOCAB_ADAPT=1` = the full vocabulary always.
///
/// 2026-09-28 — THE RANGE COULD NEVER PROPOSE THE CHAT'S CONTROL TOKENS. `dflash_note_tokens`
/// leaves them out of the range accounting (on purpose: the template alone would pin it at the
/// full 248,320 rows), so `</think>` (248069) and `<|im_end|>` (248046) sat above every range
/// short of the full vocabulary. Teacher-forced on the draft's training-style texts (measured
/// 2026-09-28, six README prompts), windows died exactly there — p3 j=23-27 proposes ordinary
/// tokens where the text and another engine's full-vocabulary draft have `</think>` — and the
/// full-vocabulary arm recovered 3 of 120 windows on p2-p5, at ~1.9 ms more draft lm_head a
/// window. So the range stays small and the control block is scored ON ITS OWN: a second lm_head
/// matmul over just the rows `[tail_start, vocab)` (see [`DflashTail`]; `tail_start` =
/// `special_floor` rounded down to the 256-row weight tile), a few hundred rows instead of ~50k.
/// Every selector (the plain argmax, the GPU top-16, the CPU `dflash_select`) sees candidates
/// from `[0, vocab_eff) ∪ [max(special_floor, vocab_eff), vocab)`, emitting TRUE token ids.
/// `ARF_NO_DFLASH_SPECIAL_TAIL=1` = the range alone, as before. NOT MEASURED yet on the GPU.
pub const DFLASH_VOCAB_STEPS: [usize; 5] = [65536, 98304, 131072, 163840, 196608];
pub fn dflash_vocab_eff(vocab: usize, max_tok: u32) -> usize {
    if std::env::var_os("ARF_NO_DFLASH_VOCAB_ADAPT").is_some() {
        return vocab;
    }
    let need = max_tok as usize + 1;
    DFLASH_VOCAB_STEPS
        .iter()
        .copied()
        .find(|&n| n >= need && n < vocab)
        .unwrap_or(vocab)
}

/// Row granularity of the lm_head's storage a tail may start at: the Q4_K_S row-major kernels
/// walk 256-row tiles (`q4rm_dims_for`), and the group-64 layout is StorageN=256 (a 256-row tile
/// is contiguous), so a tail starting on a 256 boundary is a plain byte range of either.
pub const DFLASH_TAIL_ALIGN: usize = 256;

/// THE SPECIAL-TOKEN TAIL's geometry (2026-09-28, see [`dflash_vocab_eff`]): logits for the
/// lm_head rows `[start, start + stride)` live in their own `[rows, stride]` buffer, and tail
/// index `i` IS token id `start + i`. Only indices `lo..hi` are candidates — `lo` skips what the
/// main range already scored (no double scoring) and anything below `special_floor`; `hi` stops at
/// the vocabulary (never a padding row). Mirrors the shader's `DflashTailDims`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DflashTail {
    pub start: usize,
    pub stride: usize,
    pub lo: usize,
    pub hi: usize,
}

impl DflashTail {
    /// The tail's STORAGE at attach: `None` when opted out (`ARF_NO_DFLASH_SPECIAL_TAIL`, passed
    /// in as `off`), when there is no special block below the vocabulary, or when the vocabulary is
    /// not whole weight tiles (the copied tail would need a partial tile). `lo`/`hi` are the
    /// widest window; [`Self::for_range`] narrows it per block.
    pub fn plan(vocab: usize, special_floor: u32, off: bool) -> Option<Self> {
        let floor = special_floor as usize;
        if off || floor >= vocab || !vocab.is_multiple_of(DFLASH_TAIL_ALIGN) {
            return None;
        }
        let start = floor / DFLASH_TAIL_ALIGN * DFLASH_TAIL_ALIGN;
        let stride = vocab - start; // whole tiles: both ends are multiples of the alignment
        Some(Self {
            start,
            stride,
            lo: floor - start,
            hi: vocab - start,
        })
    }

    /// This block's tail given its main range `[0, vocab_eff)`: `None` when the range already
    /// covers every candidate the tail would add (vocab_eff at or past the vocabulary — the
    /// full-vocabulary arm — skips the tail matmul entirely).
    pub fn for_range(self, vocab_eff: usize) -> Option<Self> {
        let lo = self.lo.max(vocab_eff.saturating_sub(self.start));
        (lo < self.hi).then_some(Self { lo, ..self })
    }

    /// The token id tail index `i` stands for.
    pub fn token(&self, i: usize) -> usize {
        self.start + i
    }

    /// The shader's `DflashTailDims` (`start, stride, lo, hi`); all zero = no tail.
    pub fn dims(tail: Option<Self>) -> [u32; 4] {
        tail.map_or([0; 4], |t| {
            [t.start as u32, t.stride as u32, t.lo as u32, t.hi as u32]
        })
    }
}

/// Every candidate of one block row as `(logit, TRUE token id)`, main range first, then the tail's
/// live window — the ONE enumeration the CPU selector scans. With `tail == None` it is exactly
/// the main row, index = id, as before the tail existed.
pub fn dflash_row_candidates<'a>(
    main: &'a [f32],
    tail: Option<(&'a [f32], DflashTail)>,
) -> impl Iterator<Item = (f32, usize)> + 'a {
    let t = tail.map(|(row, t)| (&row[t.lo..t.hi], t));
    main.iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .chain(t.into_iter().flat_map(|(row, t)| {
            row.iter()
                .enumerate()
                .map(move |(i, &v)| (v, t.token(t.lo + i)))
        }))
}

/// CPU reference of the plain top-1 with the tail: `argmax_b` over the main row (the first
/// maximum), then `dflash_tail_argmax`'s merge — a tail entry wins only if STRICTLY greater
/// (ties -> the lower id, and every tail id is above the main range).
pub fn dflash_top1_ref(main: &[f32], tail: Option<(&[f32], DflashTail)>) -> u32 {
    let (mut bv, mut bi) = (-3.4e38f32, 0usize);
    for (i, &v) in main.iter().enumerate() {
        if v > bv {
            (bv, bi) = (v, i);
        }
    }
    if let Some((row, t)) = tail {
        let (mut tv, mut ti) = (-3.4e38f32, usize::MAX);
        for (i, &v) in row.iter().enumerate().take(t.hi).skip(t.lo) {
            if v > tv {
                (tv, ti) = (v, i);
            }
        }
        if ti != usize::MAX && tv > main[bi] {
            bi = t.token(ti);
        }
    }
    bi as u32
}

/// Candidates per position of the GPU selector — the shader's `DFLASH_TOPK`.
pub const DFLASH_SEL_TOPK: usize = 16;

/// The SAMPLED draft's uniform for window position `pos` (1..rows) of a block anchored at
/// `past_len`: `keyed_uniform(seed, past_len + pos, DRAFT)`. The ONE keying both selectors use —
/// the CPU sampled branch of `dflash_select` and the host upload for `dflash_chain_sampled`.
pub fn dflash_draft_uniform(seed: u64, past_len: usize, pos: usize) -> f32 {
    arf_core::sampling::keyed_uniform(seed, past_len + pos, arf_core::sampling::purpose::DRAFT)
}

/// `dflash_chain_sampled`'s uniforms buffer, `[rows]`: entry 0 (the anchor) unused.
pub fn dflash_chain_uniforms(seed: u64, past_len: usize) -> [f32; DFLASH_BLOCK_ROWS] {
    let mut u = [0.0f32; DFLASH_BLOCK_ROWS];
    for (pos, x) in u.iter_mut().enumerate().skip(1) {
        *x = dflash_draft_uniform(seed, past_len, pos);
    }
    u
}

/// `dflash_chain_sampled`'s q output (`[rows, 16]` ids, 0xFFFFFFFF = no entry, and `[rows, 16]`
/// probabilities) as `RowSampler::draft_q`: one entry per DRAFTED position (rows 1..), each in the
/// kernel's order (token id ascending) — the shape the CPU sampled selector produces.
pub fn dflash_q_rows(q_id: &[u32], q_p: &[f32], rows: usize) -> Vec<Vec<(u32, f32)>> {
    (1..rows)
        .map(|pos| {
            (0..DFLASH_SEL_TOPK)
                .map(|i| pos * DFLASH_SEL_TOPK + i)
                .filter(|&i| q_id[i] != u32::MAX)
                .map(|i| (q_id[i], q_p[i]))
                .collect()
        })
        .collect()
}

/// CPU REFERENCE of the `dflash_chain_sampled` kernel, statement for statement (its loops, its
/// insertion sort, its fallbacks — NOT a call into `sampled_draft_pick`, so a test comparing the
/// two compares two implementations). Inputs are the selector's tables as the GPU holds them:
/// `cands`/`unary` `[rows, 16]`, `edges` `[rows, 16, 16]`, `unif` from [`dflash_chain_uniforms`].
/// Returns the verify window `[anchor, t1..]` and the q the kernel writes (see [`dflash_q_rows`]).
/// The GPU probe (`examples/dflash_chain_sampled_probe.rs`) gates the kernel against this.
#[allow(clippy::needless_range_loop)] // written as the kernel is, index for index
pub fn dflash_chain_sampled_ref(
    cands: &[u32],
    unary: &[f32],
    edges: &[f32],
    anchor: u32,
    unif: &[f32],
    temp: f32,
) -> (Vec<u32>, Vec<Vec<(u32, f32)>>) {
    const K: usize = DFLASH_SEL_TOPK;
    let rows = unif.len();
    let mut tok_win = vec![0u32; rows];
    let (mut q_id, mut q_p) = (vec![u32::MAX; rows * K], vec![0.0f32; rows * K]);
    tok_win[0] = anchor;
    let tdiv = temp.max(1e-6);
    let mut prev = 0usize;
    for pos in 1..rows {
        let c = &cands[pos * K..(pos + 1) * K];
        let mut s = [0.0f32; K];
        let mut ord = [0usize; K];
        let mut n = 0usize;
        let (mut best, mut bs, mut bt) = (0usize, f32::NEG_INFINITY, u32::MAX);
        for j in 0..K {
            let t = c[j];
            if t == u32::MAX {
                continue;
            }
            s[j] = unary[pos * K + j] + edges[(pos * K + prev) * K + j];
            if s[j] > bs || (s[j] == bs && t < bt) {
                (bs, bt, best) = (s[j], t, j);
            }
            let mut p = n;
            while p > 0 && c[ord[p - 1]] > t {
                ord[p] = ord[p - 1];
                p -= 1;
            }
            ord[p] = j;
            n += 1;
        }
        let mut m = f32::NEG_INFINITY;
        for i in 0..n {
            let x = s[ord[i]];
            if !x.is_nan() && x > m {
                m = x;
            }
        }
        let mut w = [0.0f32; K];
        let mut z = 0.0f32;
        for i in 0..n {
            w[i] = ((s[ord[i]] - m) / tdiv).exp();
            z += w[i];
        }
        let (mut qt, mut qv, mut qj) = ([0u32; K], [0.0f32; K], [0usize; K]);
        let nq;
        if z.is_finite() && z > 0.0 {
            for i in 0..n {
                (qt[i], qv[i], qj[i]) = (c[ord[i]], w[i] / z, ord[i]);
            }
            nq = n;
        } else {
            (qt[0], qv[0], qj[0]) = (bt, 1.0, best);
            nq = 1;
        }
        let (mut pick, mut pick_j) = (bt, best);
        let mut total = 0.0f32;
        for i in 0..nq {
            total += qv[i];
        }
        if !(total.is_nan() || total <= 0.0) {
            let target = unif[pos] * total;
            let mut cum = 0.0f32;
            let mut hit = false;
            for i in 0..nq {
                cum += qv[i];
                if cum > target {
                    (pick, pick_j, hit) = (qt[i], qj[i], true);
                    break;
                }
            }
            if !hit {
                for i in (1..=nq).rev() {
                    if qv[i - 1] > 0.0 {
                        (pick, pick_j) = (qt[i - 1], qj[i - 1]);
                        break;
                    }
                }
            }
        }
        for i in 0..K {
            q_id[pos * K + i] = if i < nq { qt[i] } else { u32::MAX };
            q_p[pos * K + i] = if i < nq { qv[i] } else { 0.0 };
        }
        tok_win[pos] = pick;
        prev = pick_j;
    }
    let q = dflash_q_rows(&q_id, &q_p, rows);
    (tok_win, q)
}

/// The CPU SAMPLED SELECTOR's chain (`dflash_select`'s sampled branch) over the same tables the
/// GPU chain reads: per position score = `unary + edges[prev]`, the argmax seeded as
/// `dflash_select` seeds it (the first candidate, ties -> the lower id), then
/// `sampled_draft_pick` with [`dflash_draft_uniform`], and `prev` = the drawn candidate's index.
/// What the kernel (and [`dflash_chain_sampled_ref`]) must reproduce, draft for draft and q for q.
#[allow(clippy::too_many_arguments)]
pub fn dflash_chain_sampled_cpu(
    cands: &[u32],
    unary: &[f32],
    edges: &[f32],
    anchor: u32,
    rows: usize,
    temp: f32,
    seed: u64,
    past_len: usize,
) -> (Vec<u32>, Vec<Vec<(u32, f32)>>) {
    const K: usize = DFLASH_SEL_TOPK;
    let mut out = vec![anchor];
    let mut qs = Vec::with_capacity(rows - 1);
    let mut prev = 0usize;
    for pos in 1..rows {
        let top: Vec<(usize, u32)> = (0..K)
            .map(|j| (j, cands[pos * K + j]))
            .filter(|t| t.1 != u32::MAX)
            .collect();
        let mut scored: Vec<(f32, u32)> = Vec::with_capacity(top.len());
        let mut best = (f32::NEG_INFINITY, top[0].1);
        for &(j, tok) in &top {
            let score = unary[pos * K + j] + edges[(pos * K + prev) * K + j];
            scored.push((score, tok));
            if score > best.0 || (score == best.0 && tok < best.1) {
                best = (score, tok);
            }
        }
        let u = dflash_draft_uniform(seed, past_len, pos);
        let (pick, dist) = arf_core::sampling::sampled_draft_pick(&scored, best.1, temp, u);
        prev = top.iter().find(|t| t.1 == pick).map_or(0, |t| t.0);
        out.push(pick);
        qs.push(dist);
    }
    (out, qs)
}

/// `dflash_chain` (the GREEDY chain) in Rust — the point-mass draft the sampled chain must leave
/// at T > 0 (a sampled chain that never leaves it is not drawing).
pub fn dflash_chain_greedy_ref(
    cands: &[u32],
    unary: &[f32],
    edges: &[f32],
    anchor: u32,
) -> Vec<u32> {
    const K: usize = DFLASH_SEL_TOPK;
    let rows = cands.len() / K;
    let mut out = vec![anchor];
    let mut prev = 0usize;
    for pos in 1..rows {
        let (mut best, mut bs, mut bt) = (0usize, f32::NEG_INFINITY, u32::MAX);
        for j in 0..K {
            let t = cands[pos * K + j];
            if t == u32::MAX {
                continue;
            }
            let s = unary[pos * K + j] + edges[(pos * K + prev) * K + j];
            if s > bs || (s == bs && t < bt) {
                (bs, bt, best) = (s, t, j);
            }
        }
        out.push(bt);
        prev = best;
    }
    out
}

/// Deterministic synthetic selector tables for the chain gates (the CPU tests and the GPU probe):
/// `[rows, 16]` distinct candidate ids in a scan order that is NOT id order, unary logits, and
/// `[rows, 16, 16]` edges. `coarse` rounds scores to 0.5 steps so exact ties occur.
pub fn dflash_chain_synthetic_tables(seed: u64, coarse: bool) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    const K: usize = DFLASH_SEL_TOPK;
    const R: usize = DFLASH_BLOCK_ROWS;
    let mut z = seed;
    let mut next = || {
        // splitmix64
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut x = z;
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    };
    let mut cands = vec![u32::MAX; R * K];
    for pos in 1..R {
        let mut ids: Vec<u32> = Vec::with_capacity(K);
        while ids.len() < K {
            let t = (next() % 50_000) as u32;
            if !ids.contains(&t) {
                ids.push(t);
            }
        }
        cands[pos * K..(pos + 1) * K].copy_from_slice(&ids);
    }
    let mut f = |scale: f32, off: f32| {
        let x = (next() >> 40) as f32 / (1u64 << 24) as f32 * scale - off;
        if coarse {
            (x * 2.0).round() / 2.0
        } else {
            x
        }
    };
    let unary: Vec<f32> = (0..R * K).map(|_| f(12.0, 2.0)).collect();
    let edges: Vec<f32> = (0..R * K * K).map(|_| f(6.0, 3.0)).collect();
    (cands, unary, edges)
}

/// Keys per threadgroup of `dflash_attn_gqa` — the shader's `DFLASH_GQA_CHUNK`.
const DFLASH_GQA_CHUNK: usize = 64;

pub struct Dflash2BlockBufs {
    hid: [MtlBuf; 2],
    normed: MtlBuf,
    dynw: MtlBuf,
    conv: MtlBuf,
    q: MtlBuf,
    k: MtlBuf,
    v: MtlBuf,
    attn: MtlBuf,
    /// `[rows * heads, window + rows]` — every query's scores over every key it can see.
    scores: MtlBuf,
    /// `dflash_attn_gqa`'s per-chunk partials: `[chunks, rows * heads, head_dim]` outputs and
    /// `[chunks, rows * heads]` (max, sum) pairs, chunks = ceil((window + rows) / 64).
    gqa_o: MtlBuf,
    gqa_ml: MtlBuf,
    proj: MtlBuf,
    resid: MtlBuf,
    gate: MtlBuf,
    up: MtlBuf,
    inter: MtlBuf,
    logits: MtlBuf,
    /// `[rows, selector_rank]` — `candidate_selector.hidden_projection(final hidden)`.
    sel: MtlBuf,
    tok_in: MtlBuf,
    tok_out: MtlBuf,
    /// The GPU candidate selector's scratch (`dflash_topk` / `dflash_edges` / `dflash_chain`):
    /// `[rows, 16]` candidate ids and their logits, the `[rows, 16, 16]` edge table, the sorted
    /// suppression list, and THE VERIFY WINDOW `[anchor, t1..t7]` the record embeds from.
    cands: MtlBuf,
    unary: MtlBuf,
    edges: MtlBuf,
    mask: MtlBuf,
    tok_win: MtlBuf,
    /// `dflash_chain_sampled`'s q: `[rows, 16]` candidate ids (0xFFFFFFFF = no entry) and their
    /// probabilities, read back after the verify record (see `dflash_gpu_q`).
    q_id: MtlBuf,
    q_p: MtlBuf,
    /// `q4rm` dims per projection shape, fetched at attach (they allocate on first use).
    d_dyn: MtlBuf,
    d_q: MtlBuf,
    d_kv: MtlBuf,
    d_o: MtlBuf,
    d_gate: MtlBuf,
    d_down: MtlBuf,
    d_lm: MtlBuf,
    d_sel: MtlBuf,
    /// `d_lm` at each of `DFLASH_VOCAB_STEPS` below the vocabulary (the adaptive candidate range).
    d_lm_steps: Vec<(usize, MtlBuf)>,
    /// THE SPECIAL-TOKEN TAIL (2026-09-28): its storage plan, `q4rm` dims for its `stride` rows,
    /// and its `[rows, stride]` logits. `None` = opted out (`ARF_NO_DFLASH_SPECIAL_TAIL=1`) or no
    /// special block below the vocabulary.
    tail: Option<(DflashTail, MtlBuf, MtlBuf)>,
    /// The tail's lm_head rows, COPIED once from the target's lm_head on the first draft (a few
    /// hundred rows — ~1.5 MB at hidden 5120 — never the 715 MB whole): keyed by the source
    /// weight's buffer address, so a different lm_head is re-copied rather than silently reused.
    /// `(key, None)` = the copy failed for that lm_head (logged once): the tail stays off.
    tail_w: std::cell::RefCell<Option<(usize, Option<Q4ksMtl>)>>,
    /// The tail the last block used (`None`: none) — with `block_vocab`, the logits' layout the
    /// CPU selector reads.
    block_tail: std::cell::Cell<Option<DflashTail>>,
}

/// `(tail plan, its q4rm dims)` — computed with the other dims at attach.
pub(super) type TailDims = Option<(DflashTail, MtlBuf)>;

impl Dflash2BlockBufs {
    /// The `q4rm` dims — call BEFORE taking the device guard (`q4rm_dims_for` takes its own).
    #[allow(clippy::type_complexity)]
    pub(super) fn dims(
        isl: &MetalIsland,
        ctx: &GpuContext,
        c: &Dflash2Config,
    ) -> Result<([MtlBuf; 8], Vec<(usize, MtlBuf)>, TailDims), String> {
        if c.block_size != DFLASH_BLOCK_ROWS {
            return Err(format!(
                "dflash: block_size {} is not the {DFLASH_BLOCK_ROWS}-row MPP unit",
                c.block_size
            ));
        }
        let (h, qd, kvd) = (c.hidden, c.heads * c.head_dim, c.kv_heads * c.head_dim);
        let d = |n: usize, k: usize| {
            isl.q4rm_dims_for(ctx, n, k)
                .ok_or_else(|| format!("dflash: [{n}, {k}] is not whole MPP tiles"))
        };
        let steps = DFLASH_VOCAB_STEPS
            .iter()
            .filter(|&&n| n < c.vocab)
            .map(|&n| Ok((n, d(n, h)?)))
            .collect::<Result<Vec<_>, String>>()?;
        let tail = DflashTail::plan(
            c.vocab,
            c.special_floor,
            std::env::var_os("ARF_NO_DFLASH_SPECIAL_TAIL").is_some(),
        );
        // rule 7: say whether the tail exists, and what it covers
        match tail {
            Some(t) => eprintln!(
                "[dflash] special-token tail: ids [{}, {}) scored every block ({} lm_head rows from {})",
                t.start + t.lo,
                t.start + t.hi,
                t.stride,
                t.start
            ),
            None => eprintln!("[dflash] special-token tail: off"),
        }
        let tail = match tail {
            Some(t) => Some((t, d(t.stride, h)?)),
            None => None,
        };
        Ok((
            [
                d(c.dynamic_size(), h)?,
                d(qd, h)?,
                d(kvd, h)?,
                d(h, qd)?,
                d(c.intermediate, h)?,
                d(h, c.intermediate)?,
                d(c.vocab, h)?,
                d(c.selector_rank, h)?,
            ],
            steps,
            tail,
        ))
    }

    pub(super) fn alloc(
        c: &Dflash2Config,
        dims: ([MtlBuf; 8], Vec<(usize, MtlBuf)>, TailDims),
        mk: &dyn Fn(usize) -> Result<MtlBuf, String>,
    ) -> Result<Self, String> {
        let r = DFLASH_BLOCK_ROWS;
        let (h, qd, kvd) = (c.hidden, c.heads * c.head_dim, c.kv_heads * c.head_dim);
        let f = |n: usize| mk(r * n * 4);
        let ([d_dyn, d_q, d_kv, d_o, d_gate, d_down, d_lm, d_sel], d_lm_steps, tail) = dims;
        let tail = match tail {
            Some((t, d)) => Some((t, d, f(t.stride)?)),
            None => None,
        };
        Ok(Self {
            hid: [f(h)?, f(h)?],
            normed: f(h)?,
            dynw: f(c.dynamic_size())?,
            conv: f(h)?,
            q: f(qd)?,
            k: f(kvd)?,
            v: f(kvd)?,
            attn: f(qd)?,
            scores: f(c.heads * (c.sliding_window + r))?,
            gqa_o: mk((c.sliding_window + r).div_ceil(DFLASH_GQA_CHUNK) * r * qd * 4)?,
            gqa_ml: mk((c.sliding_window + r).div_ceil(DFLASH_GQA_CHUNK) * r * c.heads * 8)?,
            proj: f(h)?,
            resid: f(h)?,
            gate: f(c.intermediate)?,
            up: f(c.intermediate)?,
            inter: f(c.intermediate)?,
            logits: f(c.vocab)?,
            sel: f(c.selector_rank)?,
            tok_in: mk(r * 4)?,
            tok_out: mk(r * 4)?,
            cands: mk(r * 16 * 4)?,
            unary: mk(r * 16 * 4)?,
            edges: mk(r * 16 * 16 * 4)?,
            mask: mk(256 * 4)?,
            tok_win: mk(r * 4)?,
            q_id: mk(r * 16 * 4)?,
            q_p: mk(r * 16 * 4)?,
            d_dyn,
            d_q,
            d_kv,
            d_o,
            d_gate,
            d_down,
            d_lm,
            d_sel,
            d_lm_steps,
            tail,
            tail_w: std::cell::RefCell::new(None),
            block_tail: std::cell::Cell::new(None),
        })
    }
}

/// DIAGNOSTIC cuts for `dflash_draft_block_cut` — skip a stage to attribute the block's cost by
/// subtraction (the proposals are garbage under any cut). The serving path passes 0.
pub const DFLASH_CUT_LM_HEAD: u32 = 1;
pub const DFLASH_CUT_ATTN_KERNEL: u32 = 2;
pub const DFLASH_CUT_MLP: u32 = 4;
pub const DFLASH_CUT_ATTN_BLOCK: u32 = 8;
/// Skip only the MLP sub-block's MATMULS (dynamic conv, gate, up, down) and keep its norm, conv
/// and SwiGLU kernels: splits the sub-block's cost into matmul vs everything else (2026-09-23 —
/// the MLP measured ~9.5 ms whichever matmul kernel ran, so it was not obviously the matmuls).
pub const DFLASH_CUT_MLP_MATMULS: u32 = 16;
/// Same split for the attention sub-block: skip conv-dynamic/q/k/v/o matmuls only.
pub const DFLASH_CUT_ATTN_MATMULS: u32 = 32;
/// NOT a diagnostic cut: the GPU-SELECT mode of `dflash_draft_block_cut` (2026-09-22). The
/// candidate selector runs on the GPU and the chosen window is left in `tok_win`; the call
/// returns WITHOUT waiting and its `Vec` is empty. Carried in `cut` so the one function body
/// serves both modes; it never combines with a real cut.
pub const DFLASH_CUT_GPU_SELECT: u32 = 1 << 30;
/// NOT a diagnostic cut either: with [`DFLASH_CUT_GPU_SELECT`], WAIT for the block and its GPU
/// selector before returning (2026-09-27, `dflash_draft_block_gpu_sampled_waited`) — for a caller
/// that needs the chosen tokens and q on the host before the next draft reuses the one set of
/// block buffers (the multi-stream cycle drafts its streams one after another). Waits on the
/// draft's OWN command buffer, not a queue fence.
pub const DFLASH_GPU_SELECT_WAIT: u32 = 1 << 29;

impl MetalIsland {
    /// Propose the `block_size - 1` tokens after `anchor`, the pending token of `stream` at
    /// position `past_len`. `Ok(None)` = no draft this step, for a stated reason the caller may
    /// print: no draft attached, the ring is another stream's / has a hole / is not at
    /// `past_len`, or the lm_head is not the native Q4_K one the MPP unit needs.
    /// Top-1 per position (stage 4); the selector is stage 5.
    pub fn dflash_draft_block(
        &self,
        embed_table: &ProtocolObject<dyn MTLBuffer>,
        embed_scale: f32,
        lm_head: &IslandLmHead,
        stream: Option<u64>,
        anchor: u32,
        past_len: usize,
    ) -> Result<Option<Vec<u32>>, String> {
        self.dflash_draft_block_cut(
            embed_table,
            embed_scale,
            lm_head,
            stream,
            anchor,
            past_len,
            0,
        )
    }

    /// [`Self::dflash_draft_block`], GPU-selected: the selector runs on the GPU and the verify
    /// window is left in a GPU buffer (`dflash_token_window`) with NO wait — the verify record
    /// encoded next on the same queue is ordered behind it. `Ok(Some(n))` = launched, proposing
    /// `n` tokens; `Ok(None)` = no draft this step, same reasons as `dflash_draft_block`.
    pub fn dflash_draft_block_gpu(
        &self,
        embed_table: &ProtocolObject<dyn MTLBuffer>,
        embed_scale: f32,
        lm_head: &IslandLmHead,
        stream: Option<u64>,
        anchor: u32,
        past_len: usize,
    ) -> Result<Option<usize>, String> {
        Ok(self
            .dflash_draft_block_cut(
                embed_table,
                embed_scale,
                lm_head,
                stream,
                anchor,
                past_len,
                DFLASH_CUT_GPU_SELECT,
            )?
            .map(|_| DFLASH_BLOCK_ROWS - 1))
    }

    /// GATE for the GPU selector (rule 4: text identity cannot validate a selector — a wrong
    /// draft still yields identical text, it is just rejected). Runs the CPU selector over the
    /// block's logits, which are untouched until the next draft, and returns `(cpu, gpu)` token
    /// lists for the caller to compare. Call only after the verify that consumed `tok_win`
    /// completed, so the GPU side is final.
    ///
    /// 2026-09-26: after a SAMPLED GPU draft (`dflash_chain_sampled`) the CPU side is the CPU
    /// SAMPLED selector under the same (temperature, seed, past_len) key, and the third value is
    /// the largest |q_cpu - q_gpu| over the drafted positions (`f32::INFINITY` when a position's
    /// candidate ids differ); 0.0 after a greedy chain.
    pub fn dflash_gpu_select_check(&self) -> Option<(Vec<u32>, Vec<u32>, f32)> {
        let state = self.dflash_state.borrow();
        let st = state.as_ref()?;
        let rows = DFLASH_BLOCK_ROWS;
        let win = unsafe {
            std::slice::from_raw_parts(st.block.tok_win.0.contents().as_ptr() as *const u32, rows)
        };
        let Some(key) = st.gpu_sampled.get() else {
            let cpu = self.dflash_select(st, win[0]);
            return Some((cpu, win[1..].to_vec(), 0.0));
        };
        st.draft_sampling.set(Some(key));
        st.last_q.borrow_mut().clear();
        let cpu = self.dflash_select(st, win[0]);
        st.draft_sampling.set(None);
        let q_cpu = std::mem::take(&mut *st.last_q.borrow_mut());
        let q_gpu = Self::dflash_q_of(st);
        let mut dq = 0.0f32;
        for (a, b) in q_cpu.iter().zip(&q_gpu) {
            if a.len() != b.len() || a.iter().zip(b).any(|(x, y)| x.0 != y.0) {
                dq = f32::INFINITY;
                break;
            }
            for (x, y) in a.iter().zip(b) {
                dq = dq.max((x.1 - y.1).abs());
            }
        }
        Some((cpu, win[1..].to_vec(), dq))
    }

    fn dflash_q_of(st: &Dflash2State) -> Vec<Vec<(u32, f32)>> {
        let n = DFLASH_BLOCK_ROWS * DFLASH_SEL_TOPK;
        let (q_id, q_p) = unsafe {
            (
                std::slice::from_raw_parts(st.block.q_id.0.contents().as_ptr() as *const u32, n),
                std::slice::from_raw_parts(st.block.q_p.0.contents().as_ptr() as *const f32, n),
            )
        };
        dflash_q_rows(q_id, q_p, DFLASH_BLOCK_ROWS)
    }

    /// The q of the last GPU-select draft when it was SAMPLED (`dflash_chain_sampled`), one entry
    /// per drafted position, token id ascending — `RowSampler::draft_q`'s shape. `None` after a
    /// greedy chain (a point-mass draft). Call only after the verify that consumed `tok_win`
    /// completed: the same kernel wrote both, so q is final exactly when the window is.
    pub fn dflash_gpu_q(&self) -> Option<Vec<Vec<(u32, f32)>>> {
        let state = self.dflash_state.borrow();
        let st = state.as_ref()?;
        st.gpu_sampled.get()?;
        Some(Self::dflash_q_of(st))
    }

    /// [`Self::dflash_draft_block_gpu`] for a SAMPLED request (speculative sampling, step 2, on
    /// the GPU selector — 2026-09-26): the chain is `dflash_chain_sampled`, each position drawn
    /// from softmax(score / `temperature`) keyed by (`seed`, absolute position) exactly as the CPU
    /// sampled selector keys it; q stays on the GPU until [`Self::dflash_gpu_q`]. No wait.
    /// `Ok(None)` = no draft this step (the reasons of `dflash_draft_block`), or the kernel is
    /// not compiled — the caller then takes the CPU sampled selector.
    #[allow(clippy::too_many_arguments)]
    pub fn dflash_draft_block_gpu_sampled(
        &self,
        embed_table: &ProtocolObject<dyn MTLBuffer>,
        embed_scale: f32,
        lm_head: &IslandLmHead,
        stream: Option<u64>,
        anchor: u32,
        past_len: usize,
        temperature: f32,
        seed: u64,
    ) -> Result<Option<usize>, String> {
        if !self.pipelines.contains_key("dflash_chain_sampled") {
            return Ok(None);
        }
        {
            let state = self.dflash_state.borrow();
            let Some(st) = state.as_ref() else {
                return Ok(None);
            };
            st.draft_sampling.set(Some((temperature, seed, past_len)));
        }
        let r = self.dflash_draft_block_gpu(
            embed_table,
            embed_scale,
            lm_head,
            stream,
            anchor,
            past_len,
        );
        if let Some(st) = self.dflash_state.borrow().as_ref() {
            st.draft_sampling.set(None);
        }
        r
    }

    /// [`Self::dflash_draft_block_gpu_sampled`], WAITED and READ BACK (2026-09-27): the same
    /// block and the same `dflash_chain_sampled` draws, but the call waits for the draft's command
    /// buffer and returns the drafted tokens (window positions 1..) and the q each was drawn from
    /// (`dflash_q_rows`' shape — `RowSampler::draft_q`). For the MULTI-STREAM cycle, whose record
    /// takes CPU token windows and whose streams draft one after another into the ONE set of block
    /// buffers (`tok_win`, `q_id`, `q_p`): each stream's draft must leave the GPU before the next
    /// one overwrites them. The wait replaces the point-mass draft's wait + CPU selector, which
    /// that path already paid per stream. `Ok(None)` = no draft this step (the reasons of
    /// `dflash_draft_block`), or the sampled chain is not compiled — the caller takes the CPU
    /// sampled selector.
    #[allow(clippy::too_many_arguments)]
    pub fn dflash_draft_block_gpu_sampled_waited(
        &self,
        embed_table: &ProtocolObject<dyn MTLBuffer>,
        embed_scale: f32,
        lm_head: &IslandLmHead,
        stream: Option<u64>,
        anchor: u32,
        past_len: usize,
        temperature: f32,
        seed: u64,
    ) -> Result<Option<(Vec<u32>, Vec<Vec<(u32, f32)>>)>, String> {
        if !self.pipelines.contains_key("dflash_chain_sampled") {
            return Ok(None);
        }
        {
            let state = self.dflash_state.borrow();
            let Some(st) = state.as_ref() else {
                return Ok(None);
            };
            st.draft_sampling.set(Some((temperature, seed, past_len)));
        }
        let r = self.dflash_draft_block_cut(
            embed_table,
            embed_scale,
            lm_head,
            stream,
            anchor,
            past_len,
            DFLASH_CUT_GPU_SELECT | DFLASH_GPU_SELECT_WAIT,
        );
        let state = self.dflash_state.borrow();
        let st = state.as_ref().ok_or("dflash: no draft attached")?;
        st.draft_sampling.set(None);
        if r?.is_none() {
            return Ok(None);
        }
        // The chain that ran must be the SAMPLED one: its q is what the verify divides by. A
        // greedy chain here would hand the acceptor a q the draft never drew from.
        if st.gpu_sampled.get() != Some((temperature, seed, past_len)) {
            return Err("dflash: the waited GPU draft did not run the sampled chain".into());
        }
        let rows = DFLASH_BLOCK_ROWS;
        let win = unsafe {
            std::slice::from_raw_parts(st.block.tok_win.0.contents().as_ptr() as *const u32, rows)
        };
        if win[0] != anchor {
            return Err(format!(
                "dflash: the waited GPU draft's window starts at {} (anchor {anchor})",
                win[0]
            ));
        }
        Ok(Some((win[1..].to_vec(), Self::dflash_q_of(st))))
    }

    /// The GPU-selected draft's verify window buffer, `[rows]` u32: `[anchor, t1..t7]`.
    pub fn dflash_token_window(&self) -> Option<MtlBuf> {
        self.dflash_state
            .borrow()
            .as_ref()
            .map(|s| s.block.tok_win.clone())
    }

    /// [`Self::dflash_draft_block`] with `DFLASH_CUT_*` stages skipped — timing harnesses only.
    #[allow(clippy::too_many_arguments)]
    pub fn dflash_draft_block_cut(
        &self,
        embed_table: &ProtocolObject<dyn MTLBuffer>,
        embed_scale: f32,
        lm_head: &IslandLmHead,
        stream: Option<u64>,
        anchor: u32,
        past_len: usize,
        cut: u32,
    ) -> Result<Option<Vec<u32>>, String> {
        let state = self.dflash_state.borrow();
        let Some(st) = state.as_ref() else {
            return Ok(None);
        };
        let st: &Dflash2State = st;
        let Some(ring) = st.ring_of(stream) else {
            return Ok(None);
        };
        if !ring.valid.get() || ring.committed.get() != past_len {
            return Ok(None);
        }
        let IslandLmHead::Q4ks(lm) = lm_head else {
            return Ok(None);
        };
        let c = &st.draft.cfg;
        let vocab_eff = dflash_vocab_eff(c.vocab, ring.max_tok.get());
        if st.block_vocab.get() != vocab_eff {
            // rule 7: show the range actually in use, and what set it
            eprintln!(
                "[dflash] draft candidate range -> {vocab_eff} ids (largest committed token {})",
                ring.max_tok.get()
            );
        }
        st.block_vocab.set(vocab_eff);
        if lm.n != c.vocab {
            return Err(format!(
                "dflash: lm_head is {} wide, the draft expects {}",
                lm.n, c.vocab
            ));
        }
        // The special-token tail this block scores (None: opted out, or the range covers it).
        let tail = self.dflash_tail_ready(st, lm, vocab_eff);
        st.block.block_tail.set(tail);
        // Under the profiler, drain the queue first and time that: it separates "the draft is slow"
        // from "the draft waited behind the context commit".
        let prof = std::env::var_os("ARF_SPEC_PROF").is_some();
        let t_fence = std::time::Instant::now();
        // EXPERIMENT (ARF_DFLASH_FENCE=1): drain the queue before encoding the draft. If the
        // draft's 11-99 ms `launch+drain` is its buffer QUEUING BEHIND decode work on the shared
        // `self.queue`, draining first moves that time into `fence_ms` and leaves launch small.
        // If launch stays large, the stall is not queue contention and this says so cheaply.
        if (prof || std::env::var_os("ARF_DFLASH_FENCE").is_some())
            && std::env::var_os("ARF_NO_DFLASH_FENCE").is_none()
        {
            self.dflash_fence();
        }
        let fence_ms = t_fence.elapsed().as_secs_f64() * 1e3;
        let t_start = std::time::Instant::now();
        let b = &st.block;
        let rows = DFLASH_BLOCK_ROWS;
        let (h, qd, kvd, inter) = (
            c.hidden,
            c.heads * c.head_dim,
            c.kv_heads * c.head_dim,
            c.intermediate,
        );
        unsafe {
            let p = b.tok_in.0.contents().as_ptr() as *mut u32;
            *p = anchor;
            for i in 1..rows {
                *p.add(i) = c.mask_token_id;
            }
        }
        let pso = |k: &str| {
            self.pipelines
                .get(k)
                .map(|p| p.pso.clone())
                .ok_or_else(|| format!("pso {k}"))
        };
        // ARF_NO_DFLASH_FUSED_ATTN=1 falls back to the three-kernel path for A/B.
        let p_fused = (std::env::var_os("ARF_NO_DFLASH_FUSED_ATTN").is_none())
            .then(|| {
                self.pipelines
                    .get("dflash_attn_fused")
                    .map(|p| p.pso.clone())
            })
            .flatten();
        // The split-K GQA draft attention (2026-09-24), when it compiled and the geometry is the
        // one it is written for; ARF_NO_DFLASH_GQA_ATTN=1 = the per-(row, head) kernel above.
        let p_gqa = (std::env::var_os("ARF_NO_DFLASH_GQA_ATTN").is_none()
            && rows == 8
            && c.heads == 4 * c.kv_heads
            && c.head_dim == 128)
            .then(|| {
                Some((
                    self.pipelines.get("dflash_attn_gqa")?.pso.clone(),
                    self.pipelines.get("dflash_attn_gqa_combine")?.pso.clone(),
                ))
            })
            .flatten();
        let (p_embed, p_norm, p_conv, p_head, p_scores, p_softmax, p_mix, p_swiglu, p_argmax) = (
            pso("embed_tok")?,
            pso("dflash_rmsnorm_tg")?,
            pso("dflash_conv")?,
            pso("dflash_head_norm_rope")?,
            pso("dflash_attn_scores")?,
            pso("dflash_attn_softmax")?,
            pso("dflash_attn_mix")?,
            pso("dflash_swiglu")?,
            pso("argmax_b")?,
        );
        let cmd = self
            .queue
            .commandBuffer()
            .ok_or("dflash: no command buffer")?;
        super::island::label_cb(&cmd, "arf-draft");
        let enc = cmd
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Serial)
            .ok_or("dflash: no encoder")?;
        let bind = |bufs: &[(&ProtocolObject<dyn MTLBuffer>, usize)]| {
            for (i, (bf, off)) in bufs.iter().enumerate() {
                enc.useResource_usage(
                    ProtocolObject::from_ref(*bf),
                    MTLResourceUsage::Read | MTLResourceUsage::Write,
                );
                unsafe { enc.setBuffer_offset_atIndex(Some(*bf), *off, i) };
            }
        };
        let bytes = |v: &[u32], idx: usize| unsafe {
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(v.as_ptr() as *mut std::ffi::c_void).unwrap(),
                std::mem::size_of_val(v),
                idx,
            );
        };
        // `tw` threads per threadgroup along x. NEVER 1 for a many-thread kernel: a threadgroup is
        // the unit the GPU fills its SIMD width from, so (heads x rows) threadgroups of ONE thread
        // ran each thread alone — the first build's draft cost 50-65 ms and grew with context
        // (2026-09-20), most of it this.
        let grid = |w: usize, hgt: usize, tw: usize| {
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: w.div_ceil(tw),
                    height: hgt,
                    depth: 1,
                },
                MTLSize {
                    width: tw,
                    height: 1,
                    depth: 1,
                },
            );
        };
        // Which input the Q4_K_S narrowing scratch holds. A caller's `reuse` is only honoured when
        // it matches: 2026-09-25, a group-64 lm_head (which never touches that scratch) followed by
        // the selector's `reuse = true` made the selector read a PREVIOUS layer's activations —
        // acceptance fell ~35% with no error (190 -> 291 verifies for 600 tokens).
        let narrowed_in: std::cell::Cell<Option<usize>> = std::cell::Cell::new(None);
        let matmul = |w: &Q4ksMtl,
                      dims: &MtlBuf,
                      n: usize,
                      k: usize,
                      a: &MtlBuf,
                      out: &MtlBuf,
                      reuse: bool,
                      split: bool|
         -> Result<(), String> {
            unsafe {
                // a group-64 weight (the target's lm_head when the model carries one, 2026-09-25)
                // has only placeholder Q4_K_S buffers: its own kernel, table re-prepared each time
                if let Some(g) = w.g64.as_ref() {
                    return self.g64_encode(
                        &enc,
                        rows,
                        g,
                        n,
                        &a.0,
                        &out.0,
                        &|_, _| {},
                        false,
                        None,
                    );
                }
                let addr = &*a.0 as *const _ as *const () as usize;
                let reuse = reuse && narrowed_in.get() == Some(addr);
                if !reuse {
                    narrowed_in.set(Some(addr));
                }
                self.q4rm_encode(
                    &enc,
                    rows,
                    w,
                    &dims.0,
                    n,
                    k,
                    &st.scratch,
                    &a.0,
                    &out.0,
                    &|_, _| {},
                    reuse,
                    st.parts.as_ref().filter(|_| split).map(|p| &*p.0),
                )
            }
        };
        // One 256-thread threadgroup per row (`dflash_rmsnorm_tg`); the one-thread-per-row
        // kernel it replaces cost most of the MLP sub-block's non-matmul ~3.2 ms.
        let rms = |x: &MtlBuf, w: &MtlBuf, y: &MtlBuf| {
            enc.setComputePipelineState(&p_norm);
            bind(&[(&x.0, 0), (&w.0, 0), (&y.0, 0)]);
            bytes(&[rows as u32, h as u32, c.rms_eps.to_bits(), 0], 3);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: rows,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        };
        let conv = |x: &MtlBuf, base: &MtlBuf, residual: &MtlBuf, y: &MtlBuf, stage: u32| {
            enc.setComputePipelineState(&p_conv);
            bind(&[
                (&x.0, 0),
                (&b.dynw.0, 0),
                (&base.0, 0),
                (&residual.0, 0),
                (&y.0, 0),
            ]);
            bytes(&[rows as u32, h as u32, c.conv_group_size as u32, stage], 5);
            grid(h, rows, 256);
        };
        let head_rope = |x: &MtlBuf, w: &MtlBuf, heads: usize| {
            enc.setComputePipelineState(&p_head);
            bind(&[(&x.0, 0), (&w.0, 0), (&st.inv_freq.0, 0)]);
            bytes(
                &[
                    rows as u32,
                    heads as u32,
                    c.head_dim as u32,
                    past_len as u32,
                    c.rms_eps.to_bits(),
                    0,
                    0,
                    0,
                ],
                3,
            );
            grid(heads, rows, heads);
        };

        // ---- embed: the anchor + 7 masks, through the target's table ----
        enc.setComputePipelineState(&p_embed);
        for r in 0..rows {
            bind(&[
                (embed_table, 0),
                (&b.hid[0].0, r * h * 4),
                (&b.tok_in.0, r * 4),
            ]);
            bytes(&[h as u32, embed_scale.to_bits(), 0, 0], 3);
            grid(h, 1, 256);
        }

        for (l, ly) in st.draft.layers.iter().enumerate() {
            let (x, y) = (&b.hid[l & 1], &b.hid[(l + 1) & 1]);
            if cut & DFLASH_CUT_ATTN_BLOCK == 0 {
                // ---- attention sub-block ----
                rms(x, &ly.input_norm, &b.normed);
                if cut & DFLASH_CUT_ATTN_MATMULS == 0 {
                    matmul(
                        &ly.attention_conv_dynamic,
                        &b.d_dyn,
                        c.dynamic_size(),
                        h,
                        &b.normed,
                        &b.dynw,
                        false,
                        true,
                    )?;
                }
                conv(&b.normed, &ly.attention_conv_base, x, &b.conv, 0);
                if cut & DFLASH_CUT_ATTN_MATMULS == 0 {
                    matmul(&ly.q_proj, &b.d_q, qd, h, &b.conv, &b.q, false, true)?;
                }
                if cut & DFLASH_CUT_ATTN_MATMULS == 0 {
                    matmul(&ly.k_proj, &b.d_kv, kvd, h, &b.conv, &b.k, true, true)?;
                }
                if cut & DFLASH_CUT_ATTN_MATMULS == 0 {
                    matmul(&ly.v_proj, &b.d_kv, kvd, h, &b.conv, &b.v, true, true)?;
                }
                head_rope(&b.q, &ly.q_norm, c.heads);
                head_rope(&b.k, &ly.k_norm, c.kv_heads);
                if cut & DFLASH_CUT_ATTN_KERNEL == 0 {
                    // Context positions ANY row sees: row 0's window reaches furthest back.
                    // The draft attends over its full sliding window. MEASURED 2026-09-21:
                    // capping it lowers acceptance monotonically (2.13 of 7 at the full window,
                    // 2.01 at 128, 1.82 at 32, 1.58 at 8), so the long ring earns its keep even
                    // though the reference implementation uses a short context plus a KV cache.
                    // DO NOT RE-CHASE (measured, same date).
                    let lo0 = (past_len + 1).saturating_sub(c.sliding_window);
                    let n_ctx = past_len - lo0;
                    let n_keys = n_ctx + rows;
                    let ad = [
                        rows as u32,
                        c.heads as u32,
                        c.kv_heads as u32,
                        c.head_dim as u32,
                        past_len as u32,
                        c.sliding_window as u32,
                        n_keys as u32,
                        n_ctx as u32,
                    ];
                    let queries = rows * c.heads;
                    if let Some((p_g, p_comb)) = p_gqa.as_ref() {
                        // one threadgroup per (kv head, 64-key chunk), 8 simdgroups = the 8 rows
                        let chunks = n_keys.div_ceil(DFLASH_GQA_CHUNK);
                        {
                            let n = self.dflash_gqa_dispatches.get() + 1;
                            self.dflash_gqa_dispatches.set(n);
                            if n == 1 {
                                eprintln!("[dflash] draft attention -> dflash_attn_gqa (split-K, GQA-shared) + combine");
                            }
                        }
                        enc.setComputePipelineState(p_g);
                        bind(&[
                            (&b.q.0, 0),
                            (&b.k.0, 0),
                            (&b.v.0, 0),
                            (&ring.ring_k[l].0, 0),
                            (&ring.ring_v[l].0, 0),
                            (&b.gqa_o.0, 0),
                            (&b.gqa_ml.0, 0),
                        ]);
                        bytes(&ad, 7);
                        enc.dispatchThreadgroups_threadsPerThreadgroup(
                            MTLSize {
                                width: c.kv_heads,
                                height: chunks,
                                depth: 1,
                            },
                            MTLSize {
                                width: 256,
                                height: 1,
                                depth: 1,
                            },
                        );
                        enc.setComputePipelineState(p_comb);
                        bind(&[(&b.gqa_o.0, 0), (&b.gqa_ml.0, 0), (&b.attn.0, 0)]);
                        bytes(&ad, 3);
                        bytes(&[chunks as u32], 4);
                        enc.dispatchThreadgroups_threadsPerThreadgroup(
                            MTLSize {
                                width: queries,
                                height: 1,
                                depth: 1,
                            },
                            MTLSize {
                                width: 128,
                                height: 1,
                                depth: 1,
                            },
                        );
                    } else if let Some(p_fused) = p_fused.as_ref() {
                        // ONE threadgroup per (row, head), 128 threads striding the keys with an
                        // online softmax — replaces the scores/softmax/mix trio, which measured
                        // 59x its byte floor because 8 queries gave it no parallelism.
                        enc.setComputePipelineState(p_fused);
                        bind(&[
                            (&b.q.0, 0),
                            (&b.k.0, 0),
                            (&b.v.0, 0),
                            (&ring.ring_k[l].0, 0),
                            (&ring.ring_v[l].0, 0),
                            (&b.attn.0, 0),
                        ]);
                        bytes(&ad, 6);
                        enc.dispatchThreadgroups_threadsPerThreadgroup(
                            MTLSize {
                                width: queries,
                                height: 1,
                                depth: 1,
                            },
                            MTLSize {
                                width: 128,
                                height: 1,
                                depth: 1,
                            },
                        );
                    } else {
                        enc.setComputePipelineState(&p_scores);
                        bind(&[
                            (&b.q.0, 0),
                            (&b.k.0, 0),
                            (&ring.ring_k[l].0, 0),
                            (&b.scores.0, 0),
                        ]);
                        bytes(&ad, 4);
                        grid(n_keys, queries, 64);
                        enc.setComputePipelineState(&p_softmax);
                        bind(&[(&b.scores.0, 0)]);
                        bytes(&ad, 1);
                        grid(queries, 1, 64);
                        enc.setComputePipelineState(&p_mix);
                        bind(&[
                            (&b.scores.0, 0),
                            (&b.v.0, 0),
                            (&ring.ring_v[l].0, 0),
                            (&b.attn.0, 0),
                        ]);
                        bytes(&ad, 4);
                        grid(c.head_dim, queries, 64);
                    }
                }
                if cut & DFLASH_CUT_ATTN_MATMULS == 0 {
                    matmul(&ly.o_proj, &b.d_o, h, qd, &b.attn, &b.proj, false, true)?;
                }
                conv(&b.proj, &ly.attention_conv_base, x, &b.resid, 1);
            }
            if cut & DFLASH_CUT_MLP == 0 {
                // ---- MLP sub-block ----
                rms(&b.resid, &ly.post_attention_norm, &b.normed);
                if cut & DFLASH_CUT_MLP_MATMULS == 0 {
                    matmul(
                        &ly.mlp_conv_dynamic,
                        &b.d_dyn,
                        c.dynamic_size(),
                        h,
                        &b.normed,
                        &b.dynw,
                        false,
                        true,
                    )?;
                }
                conv(&b.normed, &ly.mlp_conv_base, &b.resid, &b.conv, 0);
                if cut & DFLASH_CUT_MLP_MATMULS == 0 {
                    matmul(
                        &ly.gate_proj,
                        &b.d_gate,
                        inter,
                        h,
                        &b.conv,
                        &b.gate,
                        false,
                        true,
                    )?;
                }
                if cut & DFLASH_CUT_MLP_MATMULS == 0 {
                    matmul(&ly.up_proj, &b.d_gate, inter, h, &b.conv, &b.up, true, true)?;
                }
                enc.setComputePipelineState(&p_swiglu);
                bind(&[(&b.gate.0, 0), (&b.up.0, 0), (&b.inter.0, 0)]);
                bytes(&[(rows * inter) as u32], 3);
                grid(rows * inter, 1, 256);
                if cut & DFLASH_CUT_MLP_MATMULS == 0 {
                    matmul(
                        &ly.down_proj,
                        &b.d_down,
                        h,
                        inter,
                        &b.inter,
                        &b.proj,
                        false,
                        true,
                    )?;
                }
                conv(&b.proj, &ly.mlp_conv_base, &b.resid, y, 1);
            }
        }

        // ---- final norm -> the TARGET's lm_head (unsplit: 970 tiles saturate the GPU) -> top-1 ----
        let last = &b.hid[st.draft.layers.len() & 1];
        rms(last, &st.draft.norm, &b.normed);
        if cut & DFLASH_CUT_LM_HEAD == 0 {
            let d_lm_eff = b
                .d_lm_steps
                .iter()
                .find(|(n, _)| *n == vocab_eff)
                .map_or(&b.d_lm, |(_, d)| d);
            matmul(
                lm, d_lm_eff, vocab_eff, h, &b.normed, &b.logits, false, false,
            )?;
            // THE SPECIAL-TOKEN TAIL (2026-09-28): the control block's few hundred rows, same
            // input — it reuses the narrowing the lm_head just did.
            if let (Some(t), Some((_, d_tail, lg_tail)), Some((_, Some(w)))) =
                (tail, b.tail.as_ref(), b.tail_w.borrow().as_ref())
            {
                let split = t.stride <= h.max(c.intermediate);
                matmul(w, d_tail, t.stride, h, &b.normed, lg_tail, true, split)?;
            }
            // The selector's view of each position — same input as the lm_head, so it reuses
            // the narrowing the lm_head just did.
            matmul(
                &st.draft.selector_hidden,
                &b.d_sel,
                c.selector_rank,
                h,
                &b.normed,
                &b.sel,
                true,
                true,
            )?;
        }
        enc.setComputePipelineState(&p_argmax);
        bind(&[(&b.logits.0, 0), (&b.tok_out.0, 0)]);
        bytes(&[vocab_eff as u32, 0, 0, 0], 2);
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: rows,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        // ...and the tail's best replaces it only when strictly greater (`dflash_top1_ref`).
        if let (Some(t), Some((_, _, lg_tail))) = (tail, b.tail.as_ref()) {
            let p_tail = pso("dflash_tail_argmax")?;
            enc.setComputePipelineState(&p_tail);
            bind(&[(&b.logits.0, 0), (&lg_tail.0, 0), (&b.tok_out.0, 0)]);
            bytes(&DflashTail::dims(Some(t)), 3);
            bytes(&[vocab_eff as u32], 4);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: rows,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        }
        if cut & DFLASH_CUT_GPU_SELECT != 0 {
            // THE GPU SELECTOR. Same maths as `dflash_select` below (top-16 per position, the
            // pairwise edge table, the greedy chain), on the GPU, into `tok_win` — and then NO
            // WAIT. The verify record is encoded next on this same queue and reads `tok_win`
            // through TokenSrc::GpuBank; the CPU reads it back only after THAT completes.
            let n_mask = st.no_propose.len().min(256);
            unsafe {
                let p = b.mask.0.contents().as_ptr() as *mut u32;
                for (i, &t) in st.no_propose.iter().take(n_mask).enumerate() {
                    *p.add(i) = t;
                }
            }
            let dims: [u32; 4] = [
                vocab_eff as u32,
                c.selector_rank as u32,
                rows as u32,
                n_mask as u32,
            ];
            // A SAMPLED request's draft (2026-09-26, `dflash_draft_block_gpu_sampled`): the chain
            // DRAWS each position from softmax(score / T) with host-keyed uniforms and writes q.
            // Recorded for EVERY GPU-select encode, so a greedy chain never leaves a stale q.
            let sampled = st
                .draft_sampling
                .get()
                .filter(|_| self.pipelines.contains_key("dflash_chain_sampled"));
            st.gpu_sampled.set(sampled);
            let (p_topk, p_edges, p_chain) = (
                pso("dflash_topk")?,
                pso("dflash_edges")?,
                pso(if sampled.is_some() {
                    "dflash_chain_sampled"
                } else {
                    "dflash_chain"
                })?,
            );
            let setb = |bufs: &[&ProtocolObject<dyn MTLBuffer>]| {
                for (i, bf) in bufs.iter().enumerate() {
                    enc.useResource_usage(
                        ProtocolObject::from_ref(*bf),
                        MTLResourceUsage::Read | MTLResourceUsage::Write,
                    );
                    unsafe { enc.setBuffer_offset_atIndex(Some(*bf), 0, i) };
                }
            };
            let tgs = |w: usize, t: usize| {
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: w,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: t,
                        height: 1,
                        depth: 1,
                    },
                )
            };
            enc.setComputePipelineState(&p_topk);
            setb(&[&b.logits.0, &b.mask.0, &b.cands.0, &b.unary.0]);
            unsafe {
                enc.setBytes_length_atIndex(
                    std::ptr::NonNull::new(dims.as_ptr() as *mut _).unwrap(),
                    16,
                    4,
                )
            };
            // the tail's logits and window (all-zero dims = no tail: the loop is empty and the
            // main logits stand in as a bound-but-unread buffer)
            let tail_dims = DflashTail::dims(tail);
            let tail_buf = match (tail, b.tail.as_ref()) {
                (Some(_), Some((_, _, lg))) => &lg.0,
                _ => &b.logits.0,
            };
            enc.useResource_usage(
                ProtocolObject::from_ref(&**tail_buf),
                MTLResourceUsage::Read,
            );
            unsafe {
                enc.setBuffer_offset_atIndex(Some(tail_buf), 0, 5);
                enc.setBytes_length_atIndex(
                    std::ptr::NonNull::new(tail_dims.as_ptr() as *mut _).unwrap(),
                    16,
                    6,
                )
            };
            tgs(rows, 128);
            enc.setComputePipelineState(&p_edges);
            setb(&[
                &b.cands.0,
                &b.sel.0,
                &st.draft.predecessor_codebook.0,
                &st.draft.successor_codebook.0,
                &b.tok_in.0,
                &b.edges.0,
            ]);
            unsafe {
                enc.setBytes_length_atIndex(
                    std::ptr::NonNull::new(dims.as_ptr() as *mut _).unwrap(),
                    16,
                    6,
                )
            };
            tgs(rows * 16 * 16 / 256, 256);
            enc.setComputePipelineState(&p_chain);
            setb(&[
                &b.cands.0,
                &b.unary.0,
                &b.edges.0,
                &b.tok_in.0,
                &b.tok_win.0,
            ]);
            unsafe {
                enc.setBytes_length_atIndex(
                    std::ptr::NonNull::new(dims.as_ptr() as *mut _).unwrap(),
                    16,
                    5,
                )
            };
            if let Some((temp, seed, key_past)) = sampled {
                // Uniforms and temperature by `setBytes` (copied at ENCODE time, so no later
                // write can race an in-flight read); keyed exactly as the CPU sampled branch keys
                // them (`dflash_draft_uniform`).
                let unif = dflash_chain_uniforms(seed, key_past);
                unsafe {
                    enc.setBytes_length_atIndex(
                        std::ptr::NonNull::new(unif.as_ptr() as *mut _).unwrap(),
                        std::mem::size_of_val(&unif),
                        6,
                    );
                    enc.setBytes_length_atIndex(
                        std::ptr::NonNull::new(&temp as *const f32 as *mut _).unwrap(),
                        4,
                        7,
                    );
                }
                for (i, bf) in [&b.q_id.0, &b.q_p.0].into_iter().enumerate() {
                    enc.useResource_usage(
                        ProtocolObject::from_ref(&**bf),
                        MTLResourceUsage::Read | MTLResourceUsage::Write,
                    );
                    unsafe { enc.setBuffer_offset_atIndex(Some(bf), 0, 8 + i) };
                }
            }
            tgs(1, 1);
            enc.endEncoding();
            super::island::seam_tag(&cmd, "draft");
            cmd.commit();
            if cut & DFLASH_GPU_SELECT_WAIT != 0 {
                cmd.waitUntilCompleted();
                if prof {
                    eprintln!(
                        "[dflash-prof] gpu-select: waited, {:.2} ms from encode start",
                        t_start.elapsed().as_secs_f64() * 1e3
                    );
                }
                return Ok(Some(Vec::new()));
            }
            if prof {
                eprintln!(
                    "[dflash-prof] gpu-select: encode {:.2} ms, committed, NOT waited",
                    t_start.elapsed().as_secs_f64() * 1e3
                );
            }
            return Ok(Some(Vec::new()));
        }
        enc.endEncoding();
        let t_enc = t_start.elapsed().as_secs_f64() * 1e3;
        let t_commit = std::time::Instant::now();
        super::island::seam_tag(&cmd, "draft");
        cmd.commit();
        let commit_ms = t_commit.elapsed().as_secs_f64() * 1e3;
        let t_wait = std::time::Instant::now();
        cmd.waitUntilCompleted();
        let wait_ms = t_wait.elapsed().as_secs_f64() * 1e3;
        if prof {
            // SCHEDULING LATENCY: how long after commit() the GPU actually STARTS, and how long
            // after it ENDS before the CPU resumes. Both are dead time — the draft block's wall
            // is ~35 ms against ~21 ms of GPU and ~1 ms of selector (MEASURED 2026-09-22), and
            // this is the ~12 ms that was unexplained.
            let (st_t, en_t) = (cmd.GPUStartTime(), cmd.GPUEndTime());
            let gpu_ms = (en_t - st_t) * 1e3;
            eprintln!(
                "[dflash-prof] commit {commit_ms:.2} ms | wait {wait_ms:.2} ms (gpu {gpu_ms:.2}) | launch+drain {:.2} ms",
                wait_ms - gpu_ms
            );
        }
        if prof {
            // Where a draft's wall time goes: host encode, GPU execution, and the rest (waiting
            // behind work already queued — the context commit — plus scheduling).
            let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            let wall = t_start.elapsed().as_secs_f64() * 1e3;
            eprintln!(
                "[dflash-prof] ctx={past_len} queue-drain {fence_ms:.2} ms | encode {t_enc:.2} ms | gpu {gpu:.2} ms | wall {wall:.2} ms"
            );
        }
        let toks = unsafe {
            std::slice::from_raw_parts(b.tok_out.0.contents().as_ptr() as *const u32, rows)
        };
        if cut != 0 || std::env::var_os("ARF_NO_DFLASH_SELECTOR").is_some() {
            return Ok(Some(toks[1..].to_vec())); // top-1 per position (stage 4)
        }
        // The CPU selector runs AFTER `cmd.waitUntilCompleted()` above, i.e. with the GPU idle,
        // so every millisecond here is dead time in the speculation cycle. MEASURED 2026-09-22:
        // the draft block's wall is ~2-6x its GPU time on some cycles; this attributes it.
        let t_sel = std::time::Instant::now();
        let sel = self.dflash_select(st, anchor);
        if prof {
            eprintln!(
                "[dflash-prof] selector {:.2} ms (GPU idle throughout)",
                t_sel.elapsed().as_secs_f64() * 1e3
            );
        }
        Ok(Some(sel))
    }

    /// The special-token tail this block scores: `None` when there is none (opted out), when
    /// the candidate range already covers it, or when its rows could not be copied from `lm`.
    /// Copies the rows on first use for this lm_head (a one-off blit, waited — before the
    /// draft's own command buffer is encoded, so nothing of the draft is in flight).
    fn dflash_tail_ready(
        &self,
        st: &Dflash2State,
        lm: &Q4ksMtl,
        vocab_eff: usize,
    ) -> Option<DflashTail> {
        let (plan, _, _) = st.block.tail.as_ref()?;
        let t = plan.for_range(vocab_eff)?;
        let key = &*lm.codes as *const _ as *const () as usize
            ^ lm.g64
                .as_ref()
                .map_or(0, |g| &*g.w as *const _ as *const () as usize);
        let mut w = st.block.tail_w.borrow_mut();
        if w.as_ref().map(|x| x.0) != Some(key) {
            let built = self.dflash_tail_weight(lm, *plan, st.draft.cfg.hidden);
            match &built {
                // rule 7: the copy ran, and what it holds
                Ok(q) => eprintln!(
                    "[dflash] special-token tail: copied lm_head rows [{}, {}) ({} bytes, {})",
                    plan.start,
                    plan.start + plan.stride,
                    q.bytes(),
                    if q.g64.is_some() {
                        "group-64"
                    } else {
                        "Q4_K_S"
                    }
                ),
                Err(e) => eprintln!("[dflash] special-token tail DISABLED: {e}"),
            }
            *w = Some((key, built.ok()));
        }
        w.as_ref()?.1.as_ref()?;
        Some(t)
    }

    /// Copy the lm_head rows `[plan.start, plan.start + plan.stride)` into a weight of their own.
    /// Both storages hold a 256-row-aligned run of rows as ONE byte range per buffer: Q4_K_S is
    /// row-major (codes k/2, scales k/32, mins k/32, `ddh` k/256 x 4 bytes a row); group-64 is
    /// StorageN=256 tiled, so whole tiles are contiguous (w groups x 32 bytes a row, 24 for the
    /// 3-bit records; s and b groups x 2).
    fn dflash_tail_weight(
        &self,
        lm: &Q4ksMtl,
        plan: DflashTail,
        k: usize,
    ) -> Result<Q4ksMtl, String> {
        use objc2_metal::{MTLBlitCommandEncoder as _, MTLDevice as _};
        if !plan.start.is_multiple_of(DFLASH_TAIL_ALIGN)
            || !plan.stride.is_multiple_of(DFLASH_TAIL_ALIGN)
            || plan.start + plan.stride > lm.n
            || !k.is_multiple_of(256)
        {
            return Err(format!(
                "rows [{}, +{}) of a {}-row lm_head are not whole 256-row tiles",
                plan.start, plan.stride, lm.n
            ));
        }
        let dev = self.queue.device();
        let cb = self.queue.commandBuffer().ok_or("tail: command buffer")?;
        let blit = cb.blitCommandEncoder().ok_or("tail: blit")?;
        let copy = |src: &ProtocolObject<dyn MTLBuffer>, per_row: usize| {
            let (off, len) = (plan.start * per_row, plan.stride * per_row);
            if src.length() < off + len {
                return Err(format!(
                    "a {}-byte buffer is shorter than rows [{}, +{}) at {per_row} bytes a row",
                    src.length(),
                    plan.start,
                    plan.stride
                ));
            }
            let dst = dev
                .newBufferWithLength_options(
                    len.max(16),
                    objc2_metal::MTLResourceOptions::StorageModeShared,
                )
                .ok_or("tail: alloc")?;
            unsafe {
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    src, off, &dst, 0, len,
                )
            };
            Ok(dst)
        };
        let dims_word3 = lm.g64.as_ref().map_or(0u32, |g| u32::from(g.bits == 3) * 3);
        let built = match lm.g64.as_ref() {
            Some(g) => {
                let groups = k / 64;
                let rec = if g.bits == 3 { 24 } else { 32 };
                (|| -> Result<Q4ksMtl, String> {
                    Ok(Q4ksMtl {
                        codes: lm.codes.clone(), // placeholders, as the source's: never read
                        scales: lm.scales.clone(),
                        mins: lm.mins.clone(),
                        dd: lm.dd.clone(),
                        dims: lm.dims.clone(), // replaced below
                        n: plan.stride,
                        g64: Some(super::island::G64Mtl {
                            w: copy(&g.w, groups * rec)?,
                            s: copy(&g.s, groups * 2)?,
                            b: copy(&g.b, groups * 2)?,
                            k: g.k,
                            bits: g.bits,
                        }),
                    })
                })()
            }
            None => (|| -> Result<Q4ksMtl, String> {
                Ok(Q4ksMtl {
                    codes: copy(&lm.codes, k / 2)?,
                    scales: copy(&lm.scales, k / 32)?,
                    mins: copy(&lm.mins, k / 32)?,
                    dd: copy(&lm.dd, k / 256 * 4)?,
                    dims: lm.dims.clone(), // replaced below
                    n: plan.stride,
                    g64: None,
                })
            })(),
        };
        blit.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        let mut w = built?;
        let dims = dev
            .newBufferWithLength_options(16, objc2_metal::MTLResourceOptions::StorageModeShared)
            .ok_or("tail: alloc")?;
        unsafe {
            let p = dims.contents().as_ptr() as *mut u32;
            for (i, v) in [1u32, k as u32, plan.stride as u32, dims_word3]
                .into_iter()
                .enumerate()
            {
                *p.add(i) = v;
            }
        }
        w.dims = dims;
        Ok(w)
    }

    /// STAGE 5 — the candidate selector, as the reference implements it (z-lab/dflash,
    /// `CandidateSelector.select`, greedy): per position keep the top-`selector_top_k` logits,
    /// then walk the block left to right choosing
    ///   argmax_k  logit_k + < predecessor_codebook[prev] * selector_hidden[pos],
    ///                         successor_codebook[candidate_k] >
    /// with `prev` the anchor for the first position and the CHOSEN token after that. It is a
    /// greedy chain, not a path search: one pass, 7 x 16 dot products of rank 256. Top-1 per
    /// position picks each position's marginal favourite and the block decays toward its end
    /// ("the the the"); the pairwise term is what makes position i agree with position i-1.
    /// Runs on the HOST over the shared logits buffer: ~1.7M compares for the top-16s.
    fn dflash_select(&self, st: &Dflash2State, anchor: u32) -> Vec<u32> {
        let c = &st.draft.cfg;
        let (rows, vocab, rank, top_k) = (
            DFLASH_BLOCK_ROWS,
            st.block_vocab.get(), // the logits' row width: the block's candidate range
            c.selector_rank,
            c.selector_top_k,
        );
        let f32s = |b: &MtlBuf, n: usize| unsafe {
            std::slice::from_raw_parts(b.0.contents().as_ptr() as *const f32, n)
        };
        let bits = |b: &MtlBuf| unsafe {
            std::slice::from_raw_parts(b.0.contents().as_ptr() as *const u16, c.vocab * rank)
        };
        let bf = |x: u16| f32::from_bits((x as u32) << 16);
        let (logits, sel) = (
            f32s(&st.block.logits, rows * vocab),
            f32s(&st.block.sel, rows * rank),
        );
        // the special-token tail this block scored, if any (its `[rows, stride]` logits)
        let tail = st
            .block
            .block_tail
            .get()
            .zip(st.block.tail.as_ref())
            .map(|(t, (_, _, lg))| (f32s(lg, rows * t.stride), t));
        let (pred_cb, succ_cb) = (
            bits(&st.draft.predecessor_codebook),
            bits(&st.draft.successor_codebook),
        );
        let mut prev = anchor as usize;
        let mut out = Vec::with_capacity(rows - 1);
        let mut conf: Vec<f32> = Vec::with_capacity(rows - 1);
        let sampling = st.draft_sampling.get();
        let mut qs: Vec<Vec<(u32, f32)>> = Vec::new();
        let mut top: Vec<(f32, usize)> = Vec::with_capacity(top_k);
        for pos in 1..rows {
            let row = &logits[pos * vocab..][..vocab];
            let tail_row = tail.map(|(l, t)| (&l[pos * t.stride..][..t.stride], t));
            // top-k by threshold scan: `top` stays small and its minimum is the bar to clear.
            top.clear();
            let (mut bar, mut bar_at) = (f32::NEG_INFINITY, 0usize);
            // `tok` is the TRUE id: the main range's index, then the tail's `start + i`
            for (v, tok) in dflash_row_candidates(row, tail_row) {
                // Control tokens are never a legal DRAFT proposal mid-generation (see
                // `Dflash2State::no_propose`). Excluding them from candidacy is exact — the
                // target still emits them whenever it wants, because every drafted token is
                // verified and a rejected draft costs only the rest of the block.
                if st.no_propose.binary_search(&(tok as u32)).is_ok() {
                    continue;
                }
                if top.len() < top_k {
                    top.push((v, tok));
                    if top.len() == top_k {
                        (bar_at, bar) = top
                            .iter()
                            .enumerate()
                            .map(|(i, t)| (i, t.0))
                            .fold((0, f32::INFINITY), |m, t| if t.1 < m.1 { t } else { m });
                    }
                } else if v > bar {
                    top[bar_at] = (v, tok);
                    (bar_at, bar) = top
                        .iter()
                        .enumerate()
                        .map(|(i, t)| (i, t.0))
                        .fold((0, f32::INFINITY), |m, t| if t.1 < m.1 { t } else { m });
                }
            }
            // q = predecessor_codebook[prev] * selector_hidden[pos]
            let hs = &sel[pos * rank..][..rank];
            let q: Vec<f32> = (0..rank)
                .map(|r| bf(pred_cb[prev * rank + r]) * hs[r])
                .collect();
            let mut best = (f32::NEG_INFINITY, top[0].1);
            let mut scored: Vec<(f32, usize)> = Vec::with_capacity(top.len());
            for &(unary, tok) in &top {
                let pair: f32 = (0..rank).map(|r| q[r] * bf(succ_cb[tok * rank + r])).sum();
                let score = unary + pair;
                scored.push((score, tok));
                // ties -> the lower token id, so the choice does not depend on scan order
                if score > best.0 || (score == best.0 && tok < best.1) {
                    best = (score, tok);
                }
            }
            // SAMPLED draft (speculative sampling, step 2): draw from softmax(score / T) over the
            // candidates — another engine's form (its draft_select_dflash) — and record that q for the
            // min(1, p/q) acceptance. Candidates in id order, so the draw is reproducible.
            // 2026-09-26: the arithmetic moved, unchanged, into `arf_core::sampling::
            // sampled_draft_pick`, so the GPU chain's reference (`dflash_chain_sampled_ref`) and
            // its test share it; a degenerate softmax (NaN or zero mass) still records the point
            // mass it falls back to — the acceptor then uses the q the draft really drew from.
            if let Some((temp, seed, past_len)) = sampling {
                let cand: Vec<(f32, u32)> = scored.iter().map(|t| (t.0, t.1 as u32)).collect();
                let (pick, dist) = arf_core::sampling::sampled_draft_pick(
                    &cand,
                    best.1 as u32,
                    temp,
                    dflash_draft_uniform(seed, past_len, pos),
                );
                let pick = pick as usize;
                best = (
                    scored.iter().find(|t| t.1 == pick).map_or(best.0, |t| t.0),
                    pick,
                );
                qs.push(dist);
            }
            prev = best.1;
            out.push(best.1 as u32);
            // The draft's own confidence in this choice: its logit's softmax over the top-k
            // (an over-estimate of the true probability, monotone in it — a signal, not a
            // calibrated number).
            let chosen = top
                .iter()
                .find(|t| t.1 == best.1)
                .map_or(f32::NEG_INFINITY, |t| t.0);
            let z: f32 = top.iter().map(|t| (t.0 - chosen).exp()).sum();
            conf.push(1.0 / z);
        }
        if std::env::var_os("ARF_SPEC_DEBUG").is_some() {
            eprintln!("[dflash-conf] {conf:.3?}");
        }
        if sampling.is_some() {
            *st.last_q.borrow_mut() = qs;
        }
        out
    }

    /// [`Self::dflash_draft_block`] with the SAMPLED selector (speculative sampling, step 2): the
    /// same block forward, then each position drawn from softmax(score / `temperature`) over its
    /// candidates, keyed by (`seed`, absolute position). Returns the drafts and, per drafted
    /// position, the distribution each was drawn from — the verify accepts with min(1, p/q).
    #[allow(clippy::too_many_arguments)]
    pub fn dflash_draft_block_sampled(
        &self,
        embed_table: &ProtocolObject<dyn MTLBuffer>,
        embed_scale: f32,
        lm_head: &IslandLmHead,
        stream: Option<u64>,
        anchor: u32,
        past_len: usize,
        temperature: f32,
        seed: u64,
    ) -> Result<Option<(Vec<u32>, Vec<Vec<(u32, f32)>>)>, String> {
        {
            let state = self.dflash_state.borrow();
            let Some(st) = state.as_ref() else {
                return Ok(None);
            };
            st.draft_sampling.set(Some((temperature, seed, past_len)));
            st.last_q.borrow_mut().clear();
        }
        let r = self.dflash_draft_block_cut(
            embed_table,
            embed_scale,
            lm_head,
            stream,
            anchor,
            past_len,
            0,
        );
        let state = self.dflash_state.borrow();
        let st = state.as_ref().ok_or("dflash: no draft attached")?;
        st.draft_sampling.set(None);
        let q = std::mem::take(&mut *st.last_q.borrow_mut());
        Ok(r?.map(|d| (d, q)))
    }

    /// EXPERIMENTS ONLY: propose from a final normed hidden `[8, hidden]` computed ELSEWHERE (the
    /// CPU reference at another precision) — the target's lm_head, the selector projection and
    /// the selection run here exactly as in `dflash_draft_block`.
    pub fn dflash_propose_from_hidden(
        &self,
        lm_head: &IslandLmHead,
        hidden: &[f32],
        anchor: u32,
    ) -> Result<Vec<u32>, String> {
        let state = self.dflash_state.borrow();
        let st = state.as_ref().ok_or("dflash: no draft attached")?;
        let IslandLmHead::Q4ks(lm) = lm_head else {
            return Err("dflash: lm_head is not native Q4_K".into());
        };
        let c = &st.draft.cfg;
        let (b, rows, h) = (&st.block, DFLASH_BLOCK_ROWS, c.hidden);
        assert_eq!(hidden.len(), rows * h);
        // This path scores the FULL vocabulary into `logits`, so the CPU selector must read it
        // as such: the last draft's range and tail would have it read the wrong row width and
        // stale tail logits (2026-09-28 — the width half predates the tail).
        st.block_vocab.set(c.vocab);
        b.block_tail.set(None);
        self.dflash_fence();
        unsafe {
            std::ptr::copy_nonoverlapping(
                hidden.as_ptr(),
                b.normed.0.contents().as_ptr() as *mut f32,
                rows * h,
            );
        }
        let cmd = self
            .queue
            .commandBuffer()
            .ok_or("dflash: no command buffer")?;
        super::island::label_cb(&cmd, "arf-draft-propose");
        let enc = cmd
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Serial)
            .ok_or("dflash: no encoder")?;
        unsafe {
            if let Some(g) = lm.g64.as_ref() {
                self.g64_encode(
                    &enc,
                    rows,
                    g,
                    c.vocab,
                    &b.normed.0,
                    &b.logits.0,
                    &|_, _| {},
                    false,
                    None,
                )?;
            } else {
                self.q4rm_encode(
                    &enc,
                    rows,
                    lm,
                    &b.d_lm.0,
                    c.vocab,
                    h,
                    &st.scratch,
                    &b.normed.0,
                    &b.logits.0,
                    &|_, _| {},
                    false,
                    None,
                )?;
            }
            if let Some(g) = st.draft.selector_hidden.g64.as_ref() {
                self.g64_encode(
                    &enc,
                    rows,
                    g,
                    c.selector_rank,
                    &b.normed.0,
                    &b.sel.0,
                    &|_, _| {},
                    false,
                    None,
                )?;
            } else {
                self.q4rm_encode(
                    &enc,
                    rows,
                    &st.draft.selector_hidden,
                    &b.d_sel.0,
                    c.selector_rank,
                    h,
                    &st.scratch,
                    &b.normed.0,
                    &b.sel.0,
                    &|_, _| {},
                    // the narrowing is only there if the lm_head above took the Q4_K_S path
                    lm.g64.is_none(),
                    st.parts.as_ref().map(|p| &*p.0),
                )?;
            }
        }
        enc.endEncoding();
        super::island::seam_tag(&cmd, "draft");
        cmd.commit();
        // Split the blocking wait from the CPU selector: both happen with the GPU otherwise idle,
        // and the draft block's wall is ~2-5x its GPU time on some cycles (MEASURED 2026-09-22).
        // This says which of the two it is.
        let t_wait = std::time::Instant::now();
        cmd.waitUntilCompleted();
        let wait_ms = t_wait.elapsed().as_secs_f64() * 1e3;
        let t_sel = std::time::Instant::now();
        let sel = self.dflash_select(st, anchor);
        if std::env::var_os("ARF_SPEC_PROF").is_some() {
            let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            eprintln!(
                "[dflash-prof] propose: wait {wait_ms:.2} ms (gpu {gpu:.2}) | selector {:.2} ms",
                t_sel.elapsed().as_secs_f64() * 1e3
            );
        }
        Ok(sel)
    }

    /// GATES ONLY: the last block's logits `[8, vocab]` and final hidden `[8, hidden]`.
    pub fn dflash_block_debug(&self) -> Option<(Vec<f32>, Vec<f32>)> {
        let state = self.dflash_state.borrow();
        let st = state.as_ref()?;
        let c = &st.draft.cfg;
        let rd = |b: &MtlBuf, n: usize| unsafe {
            std::slice::from_raw_parts(b.0.contents().as_ptr() as *const f32, n).to_vec()
        };
        Some((
            rd(&st.block.logits, DFLASH_BLOCK_ROWS * c.vocab),
            rd(&st.block.normed, DFLASH_BLOCK_ROWS * c.hidden),
        ))
    }
}

#[cfg(test)]
mod tests {
    //! The SAMPLED GPU chain's host side, CPU-only (2026-09-26): the uniforms the host uploads,
    //! the q order it reads back, and a statement-for-statement reference of the kernel against
    //! the CPU sampled selector's logic. The kernel itself is gated on the GPU by
    //! `examples/dflash_chain_sampled_probe.rs` against the same reference.
    use super::*;
    use arf_core::sampling::{keyed_uniform, purpose};

    const K: usize = DFLASH_SEL_TOPK;
    const R: usize = DFLASH_BLOCK_ROWS;

    /// `[rows, 16]` distinct candidate ids (in a scan order that is NOT id order), unary logits,
    /// `[rows, 16, 16]` edges. `coarse` rounds scores to 0.5 steps so exact ties occur; `holes`
    /// leaves position 3 with 14 legal candidates.
    fn tables(seed: u64, coarse: bool, holes: bool) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
        let (c, u, e) = dflash_chain_synthetic_tables(seed, coarse);
        let mut c = c;
        if holes {
            c[3 * K + 5] = u32::MAX;
            c[3 * K + 11] = u32::MAX;
        }
        (c, u, e)
    }

    // ---- the special-token tail (2026-09-28): geometry, remap, bounds, opt-out ----

    /// Qwen3.x's geometry: vocab 248,320 (970 x 256), eos `<|im_end|>` 248046 as the floor.
    const QV: usize = 248_320;
    const QF: u32 = 248_046;

    #[test]
    fn tail_plan_is_tile_aligned_and_stops_at_the_vocabulary() {
        let t = DflashTail::plan(QV, QF, false).unwrap();
        assert_eq!(t.start, 247_808, "floor rounded DOWN to the 256-row tile");
        assert_eq!(t.stride, 512);
        assert_eq!(
            (t.token(t.lo), t.token(t.hi)),
            (QF as usize, QV),
            "live window = [floor, vocab)"
        );
        assert!(
            t.start.is_multiple_of(DFLASH_TAIL_ALIGN) && t.stride.is_multiple_of(DFLASH_TAIL_ALIGN)
        );
        // opted out, no special block, a floor at/above the vocabulary, partial tiles -> none
        assert_eq!(DflashTail::plan(QV, QF, true), None);
        assert_eq!(DflashTail::plan(QV, u32::MAX, false), None);
        assert_eq!(DflashTail::plan(QV, QV as u32, false), None);
        assert_eq!(DflashTail::plan(QV + 8, QF, false), None);
        // a floor exactly on a tile boundary: lo = 0
        let t = DflashTail::plan(QV, 247_808, false).unwrap();
        assert_eq!((t.start, t.lo, t.hi), (247_808, 0, 512));
    }

    #[test]
    fn tail_skips_what_the_range_scored_and_vanishes_at_full_vocabulary() {
        let t = DflashTail::plan(QV, QF, false).unwrap();
        for &eff in &DFLASH_VOCAB_STEPS {
            assert_eq!(
                t.for_range(eff),
                Some(t),
                "every adaptive step is below the floor"
            );
        }
        // a range reaching into the tail's tile: no id scored twice
        let r = t.for_range(248_100).unwrap();
        assert_eq!(r.token(r.lo), 248_100);
        // the full-vocabulary arm: the tail is skipped entirely
        assert_eq!(t.for_range(QV), None);
        assert_eq!(t.for_range(QV + 1000), None);
        assert_eq!(
            DflashTail::dims(None),
            [0; 4],
            "no tail = all-zero dims, an empty shader loop"
        );
        assert_eq!(
            DflashTail::dims(Some(t)),
            [247_808, 512, (QF - 247_808), 512],
            "the shader's DflashTailDims order: start, stride, lo, hi"
        );
    }

    /// Every candidate the tail adds is a TRUE id in [max(floor, range), vocab): never a padding
    /// row, never a sub-floor row of the copied tile, never an id the main range also scored.
    #[test]
    fn tail_candidates_are_true_ids_within_bounds() {
        let t = DflashTail::plan(QV, QF, false).unwrap();
        for &eff in &[65_536usize, 196_608, 248_000, 248_100, 248_319] {
            let tt = t.for_range(eff).unwrap();
            let main: Vec<f32> = (0..eff).map(|i| (i % 97) as f32).collect();
            let tail: Vec<f32> = (0..tt.stride).map(|i| 1000.0 + i as f32).collect();
            let c: Vec<(f32, usize)> = dflash_row_candidates(&main, Some((&tail, tt))).collect();
            let ids: Vec<usize> = c.iter().map(|x| x.1).collect();
            assert!(
                ids[..eff].iter().enumerate().all(|(i, &id)| i == id),
                "main index = id"
            );
            let tail_ids = &ids[eff..];
            assert_eq!(tail_ids.len(), QV - eff.max(QF as usize));
            assert!(tail_ids
                .iter()
                .all(|&id| id >= QF as usize && id >= eff && id < QV));
            // the remap: the logit attached to id is tail[id - start]
            for &(v, id) in &c[eff..] {
                assert_eq!(v, tail[id - tt.start]);
            }
            let mut sorted = ids.clone();
            sorted.dedup();
            assert_eq!(sorted.len(), ids.len(), "no id twice");
        }
    }

    /// Opted out, the CPU selector's enumeration is the main row, index for index — the old loop
    /// `row.iter().enumerate()` exactly.
    #[test]
    fn opt_out_enumeration_is_the_old_row_scan() {
        let main: Vec<f32> = (0..4096)
            .map(|i| ((i * 7919) % 1013) as f32 * 0.01 - 3.0)
            .collect();
        let old: Vec<(f32, usize)> = main.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        let new: Vec<(f32, usize)> = dflash_row_candidates(&main, None).collect();
        assert_eq!(old.len(), new.len());
        for (a, b) in old.iter().zip(&new) {
            assert_eq!((a.0.to_bits(), a.1), (b.0.to_bits(), b.1));
        }
        assert_eq!(
            DflashTail::plan(QV, QF, true),
            None,
            "the opt-out yields no tail at all"
        );
        // and the top-1 without a tail is argmax_b's first maximum
        let mut m = main.clone();
        m[100] = 50.0;
        m[200] = 50.0;
        assert_eq!(dflash_top1_ref(&m, None), 100);
    }

    #[test]
    fn top1_takes_the_tail_only_when_strictly_greater() {
        let t = DflashTail::plan(QV, QF, false)
            .unwrap()
            .for_range(65_536)
            .unwrap();
        let mut main = vec![0.0f32; 65_536];
        main[42] = 5.0;
        let mut tail = vec![-1.0f32; t.stride];
        // a huge logit in the copied tile BELOW the floor (a padding-like row) never wins
        tail[0] = 1e9;
        assert_eq!(dflash_top1_ref(&main, Some((&tail, t))), 42);
        // a tie with the main winner -> the main (lower) id
        tail[248_069 - t.start] = 5.0;
        assert_eq!(dflash_top1_ref(&main, Some((&tail, t))), 42);
        // strictly greater -> the TRUE id of `</think>`
        tail[248_069 - t.start] = 5.5;
        assert_eq!(dflash_top1_ref(&main, Some((&tail, t))), 248_069);
        // two tail entries tied -> the lower id
        tail[248_046 - t.start] = 5.5;
        assert_eq!(dflash_top1_ref(&main, Some((&tail, t))), 248_046);
    }

    #[test]
    fn chain_uniforms_are_the_cpu_selectors_keying() {
        for &(seed, past) in &[(0u64, 0usize), (1234, 77), (u64::MAX, 1 << 20)] {
            let u = dflash_chain_uniforms(seed, past);
            assert_eq!(u[0], 0.0, "row 0 is the anchor, never drawn");
            for (pos, &x) in u.iter().enumerate().skip(1) {
                // exactly the key `dflash_select`'s sampled branch draws position `pos` with
                assert_eq!(x, keyed_uniform(seed, past + pos, purpose::DRAFT));
                assert_eq!(x, dflash_draft_uniform(seed, past, pos));
            }
        }
    }

    /// The kernel's arithmetic (its reference) == the CPU sampled selector's, draft for draft and
    /// q for q, bit for bit — over many synthetic blocks, temperatures from near-greedy to hot,
    /// exact score ties, and candidate holes. Both run in Rust here, so any difference is a
    /// difference in ORDER, KEYING, PREV-INDEXING or FALLBACK, not in float libraries.
    #[test]
    fn chain_sampled_reference_matches_the_cpu_sampled_selector() {
        let mut n = 0usize;
        let mut not_argmax = 0usize;
        for case in 0..400u64 {
            let (cands, unary, edges) = tables(case, case % 3 == 0, case % 5 == 0);
            let temp = [0.7f32, 1.0, 0.3, 1.5, 1e-9][case as usize % 5];
            let (seed, past) = (case.wrapping_mul(7919) + 3, 100 + case as usize * 13);
            let anchor = 42_000 + case as u32;
            let unif = dflash_chain_uniforms(seed, past);
            let (win, q) = dflash_chain_sampled_ref(&cands, &unary, &edges, anchor, &unif, temp);
            let (cpu, q_cpu) =
                dflash_chain_sampled_cpu(&cands, &unary, &edges, anchor, R, temp, seed, past);
            assert_eq!(win, cpu, "case {case}: drafts differ");
            assert_eq!(q.len(), R - 1);
            assert_eq!(q.len(), q_cpu.len());
            for (i, (a, b)) in q.iter().zip(&q_cpu).enumerate() {
                let pos = i + 1;
                assert_eq!(a.len(), b.len(), "case {case} pos {pos}: q sizes");
                for (x, y) in a.iter().zip(b) {
                    assert_eq!(x.0, y.0, "case {case} pos {pos}: q order");
                    assert_eq!(x.1.to_bits(), y.1.to_bits(), "case {case}: q values");
                }
                // q is over the legal candidates, in token-id order, and a distribution
                assert!(a.windows(2).all(|w| w[0].0 < w[1].0), "q not id-sorted");
                let legal = (0..K).filter(|&j| cands[pos * K + j] != u32::MAX).count();
                assert_eq!(a.len(), legal, "case {case} pos {pos}: q support");
                let s: f32 = a.iter().map(|t| t.1).sum();
                assert!((s - 1.0).abs() < 1e-5, "q sums to {s}");
                // the drafted token is one q gives mass to (min(1, p/q) needs q(x) > 0)
                let x = win[pos];
                assert!(a.iter().any(|t| t.0 == x && t.1 > 0.0), "draft outside q");
            }
            n += R - 1;
            // at T > 0.1 the chain must actually SAMPLE (not collapse to the greedy chain)
            if temp > 0.1 {
                let greedy = dflash_chain_greedy_ref(&cands, &unary, &edges, anchor);
                not_argmax += win.iter().zip(&greedy).filter(|(a, b)| a != b).count();
            }
        }
        assert!(n > 2000);
        assert!(
            not_argmax > 100,
            "only {not_argmax} positions left the greedy chain — the draw is not drawing"
        );
    }

    /// A degenerate softmax (a NaN score, or every score -inf) records the point mass at the
    /// argmax — on both sides — and the chain continues from that candidate.
    #[test]
    fn degenerate_softmax_falls_back_to_the_argmax_point_mass() {
        let (cands, mut unary, edges) = tables(9, false, false);
        unary[2 * K + 4] = f32::NAN; // position 2: a NaN score poisons z
        for j in 0..K {
            unary[5 * K + j] = f32::NEG_INFINITY; // position 5: no finite score at all
        }
        let unif = dflash_chain_uniforms(5, 50);
        let (win, q) = dflash_chain_sampled_ref(&cands, &unary, &edges, 1, &unif, 0.8);
        let (cpu, q_cpu) = dflash_chain_sampled_cpu(&cands, &unary, &edges, 1, R, 0.8, 5, 50);
        assert_eq!(win, cpu);
        assert_eq!(q, q_cpu);
        // position 2: the argmax over the finite scores, given the candidate drawn at 1
        let prev = (0..K).find(|&j| cands[K + j] == win[1]).unwrap();
        let best = (0..K)
            .filter(|&j| j != 4)
            .map(|j| {
                (
                    unary[2 * K + j] + edges[(2 * K + prev) * K + j],
                    cands[2 * K + j],
                )
            })
            .fold((f32::NEG_INFINITY, u32::MAX), |b, t| {
                if t.0 > b.0 || (t.0 == b.0 && t.1 < b.1) {
                    t
                } else {
                    b
                }
            })
            .1;
        assert_eq!(
            q[1],
            vec![(best, 1.0)],
            "position 2: point mass at the argmax"
        );
        assert_eq!(win[2], best);
        let lowest = (0..K).map(|j| cands[5 * K + j]).min().unwrap();
        assert_eq!(q[4], vec![(lowest, 1.0)], "all -inf: ties -> the lower id");
        assert_eq!(win[5], lowest);
    }

    /// The readback: `[rows, 16]` ids with 0xFFFFFFFF holes -> one q per drafted position, row 0
    /// skipped, order kept.
    #[test]
    fn q_rows_decode_the_kernels_layout() {
        let mut id = vec![u32::MAX; R * K];
        let mut p = vec![0.0f32; R * K];
        id[0] = 7; // row 0 is never a drafted position
        (id[K], p[K]) = (3, 0.25);
        (id[K + 1], p[K + 1]) = (9, 0.75);
        (id[2 * K], p[2 * K]) = (5, 1.0);
        let q = dflash_q_rows(&id, &p, R);
        assert_eq!(q.len(), R - 1);
        assert_eq!(q[0], vec![(3, 0.25), (9, 0.75)]);
        assert_eq!(q[1], vec![(5, 1.0)]);
        assert!(q[2..].iter().all(|r| r.is_empty()));
    }
}
