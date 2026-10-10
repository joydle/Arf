//! Paged KV cache.
//!
//! Per layer we keep two flat `f32` buffers of `num_blocks * block_size` rows,
//! each row holding `kv_heads * head_dim` values — one buffer for keys, one for
//! values. A sequence owns a list of physical block ids (its *block table*);
//! logical token position `p` lives at physical slot
//! `block_table[p / block_size] * block_size + (p % block_size)`.
//!
//! Writes copy contiguous runs (a decode step is one slot; a prefill spans a few
//! blocks). Reads gather a sequence's whole context by slot.
//!
//! The pool can be **exact `f32`** (`KvStore::F32`, the default and the parity
//! oracle) or **TurboQuant-compressed** (`KvStore::Tq`, selected via
//! [`KvQuant::Tq`]). Under Tq each head-vector (`head_dim` wide) is stored as a
//! packed [`TqKvBlock`] code + bf16 norm; `write` quantizes per head-vector and
//! `gather` reconstructs to `f32` (rotate-back), so the attention layer is
//! unchanged — it sees a slightly lossy `f32` cache, exactly as Q4 weights are a
//! lossy view of bf16. (The R-space attention fast path that skips the per-read
//! rotate-back is a later optimization.)

use crate::config::KvQuant;
use crate::tensor::turboquant::{Codebook, QjlProjection, Rotation, TqKvBlock};

/// Seed for the shared TurboQuant rotation (only used by `DenseRotation`; the
/// Hadamard path is seedless). Fixed so a run is bit-reproducible.
const TQ_ROTATION_SEED: u64 = 0x7175_726f_5f74_7100; // "quro_tq"
/// Seed for the QJL projection matrix.
const TQ_QJL_SEED: u64 = 0x716a_6c5f_7475_7262; // "qjl_turb"

/// A contiguous write into slot space: copy `count` rows from
/// `source[src_off..]` to cache rows `[phys_start..phys_start+count)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WriteRun {
    pub src_off: usize,
    pub phys_start: usize,
    pub count: usize,
}

/// Backing storage for the pool: exact `f32` or TurboQuant-packed.
#[derive(Debug, Clone)]
enum KvStore {
    /// Per layer: `(num_blocks*block_size)` rows × `row_floats` — the exact path.
    F32 {
        keys: Vec<Vec<f32>>,
        values: Vec<Vec<f32>>,
    },
    /// Per layer: one packed block of `total_slots * kv_heads` head-vectors for
    /// keys and one for values, sharing a codebook + rotation (+ QJL projection).
    Tq {
        keys: Vec<TqKvBlock>,
        values: Vec<TqKvBlock>,
        bits: u8,
        cb: Codebook,
        rot: Rotation,
        qjl: Option<QjlProjection>,
    },
}

/// Keys and values for every layer.
/// Per-layer KV geometry. Uniform across layers for Llama/Qwen/Gemma-3, but Gemma 4
/// interleaves SLIDING layers (16 kv-heads × 256 head_dim = 4096-wide rows) with
/// GLOBAL layers (4 × 512 = 2048-wide) — so the cache must size and stride each
/// layer's K/V buffer by its OWN geometry, not one shared `row_floats`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerKvGeom {
    pub kv_heads: usize,
    pub head_dim: usize,
}

impl LayerKvGeom {
    #[inline]
    pub fn row_floats(&self) -> usize {
        self.kv_heads * self.head_dim
    }
}

#[derive(Debug, Clone)]
pub struct PagedKvCache {
    store: KvStore,
    block_size: usize,
    num_blocks: usize,
    /// Per-layer geometry (one entry per layer). All entries are equal for
    /// non-Gemma-4 models — then this behaves exactly like the old uniform cache.
    geom: Vec<LayerKvGeom>,
}

impl PagedKvCache {
    /// Allocate a zeroed **exact `f32`** cache for `num_layers` layers.
    pub fn new(
        num_layers: usize,
        num_blocks: usize,
        block_size: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self::with_quant(
            num_layers,
            num_blocks,
            block_size,
            kv_heads,
            head_dim,
            KvQuant::None,
        )
    }

    /// Allocate a zeroed cache with an explicit [`KvQuant`] mode. `KvQuant::None`
    /// is identical to [`PagedKvCache::new`]; `KvQuant::Tq` builds the packed
    /// pools plus the shared codebook/rotation.
    pub fn with_quant(
        num_layers: usize,
        num_blocks: usize,
        block_size: usize,
        kv_heads: usize,
        head_dim: usize,
        kv_quant: KvQuant,
    ) -> Self {
        // Uniform geometry: every layer the same — the Llama/Qwen/Gemma-3 path.
        let geom = vec![LayerKvGeom { kv_heads, head_dim }; num_layers];
        Self::with_quant_per_layer(&geom, num_blocks, block_size, kv_quant)
    }

    /// Allocate a cache where each layer can have its OWN KV geometry (Gemma 4). The
    /// Tq codebook/rotation are sized to the MAX head_dim across layers (a per-vector
    /// reconstruction reads only its own head_dim, so over-sizing is safe and the
    /// shared tables stay valid). Panics if `geom` is empty.
    pub fn with_quant_per_layer(
        geom: &[LayerKvGeom],
        num_blocks: usize,
        block_size: usize,
        kv_quant: KvQuant,
    ) -> Self {
        assert!(!geom.is_empty(), "kv cache needs at least one layer");
        let total_slots = num_blocks * block_size;
        let store = match kv_quant {
            KvQuant::None => {
                // Each layer's buffer sized to ITS OWN row width.
                let keys = geom
                    .iter()
                    .map(|g| vec![0.0; total_slots * g.row_floats()])
                    .collect();
                let values = geom
                    .iter()
                    .map(|g| vec![0.0; total_slots * g.row_floats()])
                    .collect();
                KvStore::F32 { keys, values }
            }
            KvQuant::Tq { bits, qjl } => {
                // Tq head_dim must be uniform (the codebook/rotation are per-head_dim);
                // assert it until per-layer-head_dim Tq is built. Most Gemma-4 routes
                // use F32 KV on this path; Tq + varying head_dim is a follow-up.
                let hd = geom[0].head_dim;
                assert!(
                    geom.iter().all(|g| g.head_dim == hd),
                    "Tq KV requires a uniform head_dim across layers (Gemma-4 global \
                     layers vary head_dim; use F32 KV for now — per-layer-head_dim Tq \
                     is a follow-up)"
                );
                let cb = Codebook::new(bits, hd);
                let rot = Rotation::default_for(hd, TQ_ROTATION_SEED);
                let qjl = qjl.then(|| QjlProjection::new(hd, hd, TQ_QJL_SEED));
                let keys = geom
                    .iter()
                    .map(|g| TqKvBlock::zeros(total_slots * g.kv_heads, hd, bits))
                    .collect();
                let values = geom
                    .iter()
                    .map(|g| TqKvBlock::zeros(total_slots * g.kv_heads, hd, bits))
                    .collect();
                KvStore::Tq {
                    keys,
                    values,
                    bits,
                    cb,
                    rot,
                    qjl,
                }
            }
        };
        PagedKvCache {
            store,
            block_size,
            num_blocks,
            geom: geom.to_vec(),
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    pub fn num_layers(&self) -> usize {
        match &self.store {
            KvStore::F32 { keys, .. } => keys.len(),
            KvStore::Tq { keys, .. } => keys.len(),
        }
    }

    /// Whether this pool is TurboQuant-compressed.
    pub fn is_quantized(&self) -> bool {
        matches!(self.store, KvStore::Tq { .. })
    }

    /// The TurboQuant parameters, if this pool is quantized: `(bits, codebook,
    /// rotation, optional QJL projection)`. The attention R-space fast path and
    /// the GPU upload read these; `None` for the `f32` pool.
    pub fn tq_params(&self) -> Option<(u8, &Codebook, &Rotation, Option<&QjlProjection>)> {
        match &self.store {
            KvStore::F32 { .. } => None,
            KvStore::Tq {
                bits, cb, rot, qjl, ..
            } => Some((*bits, cb, rot, qjl.as_ref())),
        }
    }

    /// Write new keys/values for one sequence into `layer`. `new_k`/`new_v` hold
    /// `q_len` rows (`q_len * row_floats` values); `runs` say where they land.
    pub fn write(&mut self, layer: usize, new_k: &[f32], new_v: &[f32], runs: &[WriteRun]) {
        let rf = self.geom[layer].row_floats();
        match &mut self.store {
            KvStore::F32 { keys, values } => {
                let k = &mut keys[layer];
                let v = &mut values[layer];
                for run in runs {
                    let (s0, s1) = (run.src_off * rf, (run.src_off + run.count) * rf);
                    let (d0, d1) = (run.phys_start * rf, (run.phys_start + run.count) * rf);
                    k[d0..d1].copy_from_slice(&new_k[s0..s1]);
                    v[d0..d1].copy_from_slice(&new_v[s0..s1]);
                }
            }
            KvStore::Tq {
                keys,
                values,
                cb,
                rot,
                ..
            } => {
                let (kb, vb) = (&mut keys[layer], &mut values[layer]);
                let (hd, heads) = (self.geom[layer].head_dim, self.geom[layer].kv_heads);
                for run in runs {
                    for local in 0..run.count {
                        let slot = run.phys_start + local;
                        let srow = (run.src_off + local) * rf;
                        for h in 0..heads {
                            let off = srow + h * hd;
                            let vidx = slot * heads + h;
                            kb.set_vector(vidx, &new_k[off..off + hd], cb, rot);
                            vb.set_vector(vidx, &new_v[off..off + hd], cb, rot);
                        }
                    }
                }
            }
        }
    }

    /// Gather a sequence's full context for `layer`. Returns `(keys, values)`,
    /// each a flat `[ctx * row_floats]` buffer in slot order. Under Tq the rows
    /// are reconstructed to `f32` (dequant + inverse-rotate).
    pub fn gather(&self, layer: usize, slots: &[u32]) -> (Vec<f32>, Vec<f32>) {
        let rf = self.geom[layer].row_floats();
        match &self.store {
            KvStore::F32 { keys, values } => {
                let k = &keys[layer];
                let v = &values[layer];
                let mut gk = Vec::with_capacity(slots.len() * rf);
                let mut gv = Vec::with_capacity(slots.len() * rf);
                for &slot in slots {
                    let o = slot as usize * rf;
                    gk.extend_from_slice(&k[o..o + rf]);
                    gv.extend_from_slice(&v[o..o + rf]);
                }
                (gk, gv)
            }
            KvStore::Tq {
                keys,
                values,
                cb,
                rot,
                ..
            } => {
                let (kb, vb) = (&keys[layer], &values[layer]);
                let (hd, heads) = (self.geom[layer].head_dim, self.geom[layer].kv_heads);
                let mut gk = vec![0.0f32; slots.len() * rf];
                let mut gv = vec![0.0f32; slots.len() * rf];
                let mut tmp = vec![0.0f32; hd];
                for (si, &slot) in slots.iter().enumerate() {
                    for h in 0..heads {
                        let vidx = slot as usize * heads + h;
                        let out = si * rf + h * hd;
                        kb.dequant_rotated_into(vidx, cb, &mut tmp);
                        rot.apply_transpose(&mut tmp);
                        gk[out..out + hd].copy_from_slice(&tmp);
                        vb.dequant_rotated_into(vidx, cb, &mut tmp);
                        rot.apply_transpose(&mut tmp);
                        gv[out..out + hd].copy_from_slice(&tmp);
                    }
                }
                (gk, gv)
            }
        }
    }
}

/// Physical slot ids for logical positions `[0, len)` given a `block_table`.
pub fn slots_for(block_table: &[u32], block_size: usize, len: usize) -> Vec<u32> {
    (0..len)
        .map(|p| block_table[p / block_size] * block_size as u32 + (p % block_size) as u32)
        .collect()
}

/// Contiguous write runs for `q_len` new tokens appended starting at logical
/// position `past_len`. Splits at block boundaries.
pub fn write_runs(
    block_table: &[u32],
    block_size: usize,
    past_len: usize,
    q_len: usize,
) -> Vec<WriteRun> {
    let mut runs = Vec::new();
    let mut pos = past_len;
    let mut remaining = q_len;
    let mut src_off = 0;
    while remaining > 0 {
        let block = block_table[pos / block_size] as usize;
        let off = pos % block_size;
        let count = remaining.min(block_size - off);
        runs.push(WriteRun {
            src_off,
            phys_start: block * block_size + off,
            count,
        });
        pos += count;
        remaining -= count;
        src_off += count;
    }
    runs
}

/// Number of blocks needed to hold `tokens` positions.
pub fn blocks_needed(tokens: usize, block_size: usize) -> usize {
    tokens.div_ceil(block_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_map_across_blocks() {
        let slots = slots_for(&[2, 0], 4, 6);
        assert_eq!(slots, vec![8, 9, 10, 11, 0, 1]);
    }

    #[test]
    fn write_runs_split_at_block_boundary() {
        let runs = write_runs(&[2, 0, 5], 4, 3, 5);
        assert_eq!(
            runs,
            vec![
                WriteRun {
                    src_off: 0,
                    phys_start: 11,
                    count: 1
                },
                WriteRun {
                    src_off: 1,
                    phys_start: 0,
                    count: 4
                },
            ]
        );
    }

    #[test]
    fn write_then_gather_roundtrips() {
        let (block_size, kv_heads, head_dim) = (4, 2, 3);
        let rf = kv_heads * head_dim;
        let mut cache = PagedKvCache::new(1, 4, block_size, kv_heads, head_dim);
        let block_table = vec![1u32, 3]; // non-contiguous physical blocks
        let n = 6;
        let k: Vec<f32> = (0..n * rf).map(|i| i as f32).collect();
        let v: Vec<f32> = k.iter().map(|x| x + 100.0).collect();

        let runs = write_runs(&block_table, block_size, 0, n);
        cache.write(0, &k, &v, &runs);

        let slots = slots_for(&block_table, block_size, n);
        let (gk, gv) = cache.gather(0, &slots);
        assert_eq!(gk, k);
        assert_eq!(gv, v);
    }

    #[test]
    fn per_layer_geometry_roundtrips_independently() {
        // Gemma-4 shape: layer 0 = GLOBAL (4 heads × 8 = row 32), layer 1 = SLIDING
        // (2 heads × 4 = row 16). Each layer's buffer is sized to its OWN width; a
        // write+gather on one must not be confused by the other's stride.
        let block_size = 4;
        let geom = [
            LayerKvGeom {
                kv_heads: 4,
                head_dim: 8,
            }, // row_floats 32
            LayerKvGeom {
                kv_heads: 2,
                head_dim: 4,
            }, // row_floats 16
        ];
        let mut cache = PagedKvCache::with_quant_per_layer(&geom, 4, block_size, KvQuant::None);
        let block_table = vec![1u32, 3];
        let n = 6;

        // Layer 0 (row 32) and layer 1 (row 16) get DISTINCT data at their own widths.
        let rf0 = geom[0].row_floats();
        let k0: Vec<f32> = (0..n * rf0).map(|i| i as f32).collect();
        let v0: Vec<f32> = k0.iter().map(|x| x + 1000.0).collect();
        let rf1 = geom[1].row_floats();
        let k1: Vec<f32> = (0..n * rf1).map(|i| (i as f32) + 0.5).collect();
        let v1: Vec<f32> = k1.iter().map(|x| x + 2000.0).collect();

        let runs = write_runs(&block_table, block_size, 0, n);
        cache.write(0, &k0, &v0, &runs);
        cache.write(1, &k1, &v1, &runs);

        let slots = slots_for(&block_table, block_size, n);
        let (g0k, g0v) = cache.gather(0, &slots);
        let (g1k, g1v) = cache.gather(1, &slots);
        // Each layer round-trips at its OWN width — no cross-stride contamination.
        assert_eq!(g0k, k0, "layer 0 (row 32) keys");
        assert_eq!(g0v, v0, "layer 0 (row 32) values");
        assert_eq!(g1k, k1, "layer 1 (row 16) keys");
        assert_eq!(g1v, v1, "layer 1 (row 16) values");
    }

    use crate::config::KvQuant;

    fn gaussian(n: usize, seed: u64) -> Vec<f32> {
        use rand::rngs::StdRng;
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                let u1: f64 = rng.random::<f64>().max(1e-12);
                let u2: f64 = rng.random::<f64>();
                ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
            })
            .collect()
    }

    fn rel_mse(a: &[f32], b: &[f32]) -> f64 {
        let mut err = 0.0f64;
        let mut sig = 0.0f64;
        for (x, y) in a.iter().zip(b) {
            err += ((x - y) as f64).powi(2);
            sig += (*y as f64).powi(2);
        }
        err / sig
    }

    #[test]
    fn tq_write_gather_reconstructs_within_bound() {
        // head_dim 64 (power of two → Hadamard); kv_heads 8 like Llama-1B.
        let (block_size, kv_heads, head_dim) = (4, 8, 64);
        let rf = kv_heads * head_dim;
        let n = 6usize;
        let k = gaussian(n * rf, 1);
        let v = gaussian(n * rf, 2);
        let block_table = vec![1u32, 3];
        let runs = write_runs(&block_table, block_size, 0, n);
        let slots = slots_for(&block_table, block_size, n);
        let bound = |bits: u8| match bits {
            4 => 0.009,
            3 => 0.036,
            2 => 0.117,
            _ => unreachable!(),
        };
        for bits in [2u8, 3, 4] {
            let mut cache = PagedKvCache::with_quant(
                1,
                4,
                block_size,
                kv_heads,
                head_dim,
                KvQuant::Tq { bits, qjl: false },
            );
            assert!(cache.is_quantized());
            cache.write(0, &k, &v, &runs);
            let (gk, gv) = cache.gather(0, &slots);
            let rk = rel_mse(&gk, &k);
            let rv = rel_mse(&gv, &v);
            assert!(rk <= bound(bits) * 2.0, "{bits}-bit key rel MSE {rk}");
            assert!(rv <= bound(bits) * 2.0, "{bits}-bit value rel MSE {rv}");
        }
    }

    #[test]
    fn tq_paged_equals_contiguous() {
        // Same invariant the f32 path must hold: gather order is slot order,
        // independent of physical block layout.
        let (block_size, kv_heads, head_dim) = (4, 4, 64);
        let rf = kv_heads * head_dim;
        let n = 6usize;
        let k = gaussian(n * rf, 7);
        let v = gaussian(n * rf, 8);
        let q = KvQuant::Tq {
            bits: 4,
            qjl: false,
        };

        let mut contiguous = PagedKvCache::with_quant(1, 4, block_size, kv_heads, head_dim, q);
        let bt_c = vec![0u32, 1];
        contiguous.write(0, &k, &v, &write_runs(&bt_c, block_size, 0, n));
        let (ck, cv) = contiguous.gather(0, &slots_for(&bt_c, block_size, n));

        let mut paged = PagedKvCache::with_quant(1, 4, block_size, kv_heads, head_dim, q);
        let bt_p = vec![3u32, 1]; // scrambled physical blocks
        paged.write(0, &k, &v, &write_runs(&bt_p, block_size, 0, n));
        let (pk, pv) = paged.gather(0, &slots_for(&bt_p, block_size, n));

        assert_eq!(ck, pk, "paged keys differ from contiguous");
        assert_eq!(cv, pv, "paged values differ from contiguous");
    }

    #[test]
    fn tq_unwritten_slots_are_zero() {
        let q = KvQuant::Tq {
            bits: 3,
            qjl: false,
        };
        let cache = PagedKvCache::with_quant(1, 2, 4, 4, 64, q);
        let slots: Vec<u32> = (0..8).collect();
        let (gk, gv) = cache.gather(0, &slots);
        assert!(gk.iter().all(|&x| x == 0.0));
        assert!(gv.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn tq_params_exposed_only_when_quantized() {
        let f32_cache = PagedKvCache::new(1, 2, 4, 4, 64);
        assert!(f32_cache.tq_params().is_none());
        let q = KvQuant::Tq { bits: 4, qjl: true };
        let tq = PagedKvCache::with_quant(1, 2, 4, 4, 64, q);
        let (bits, _cb, _rot, qjl) = tq.tq_params().expect("tq params");
        assert_eq!(bits, 4);
        assert!(
            qjl.is_some(),
            "qjl projection should be built when requested"
        );
    }
}
