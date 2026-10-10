//! Load weights from safetensors and map HuggingFace Llama parameter names onto
//! our modules.
//!
//! Files are memory-mapped; matrices are kept resident as **bf16** (half the
//! memory of f32) and norm vectors as f32. The mmap is read-only and its bytes
//! are copied into owned buffers before it is dropped.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;
use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensors;

use crate::config::{AttnKind, MlpKind, ModelConfig, NormStyle, Quant};
use crate::device::Device;
use crate::error::{ArfError, Result};
use crate::model::attention::Attention;
use crate::model::layer::{DecoderLayer, MlpBlock};
use crate::model::llama::Llama;
use crate::model::mlp::Mlp;
use crate::model::nn::{Embedding, Linear};
use crate::model::rms_norm::RmsNorm;
use crate::model::rope::Rope;
use crate::tensor::dtype::bf16_to_f32;
use crate::tensor::{Bf16Matrix, Q8Matrix, Tensor};

/// Raw tensor data as loaded, before it is committed to a resident layout.
enum RawData {
    F32(Vec<f32>),
    Bf16(Vec<u16>),
}

struct RawTensor {
    shape: Vec<usize>,
    data: RawData,
}

/// A named collection of weight tensors. Either fully resident in RAM (safetensors
/// and synthetic test weights) or lazily backed by an mmap'd GGUF that dequantizes
/// each tensor on demand (so a model larger than RAM-as-f32 still loads).
pub struct Weights {
    backend: WeightsBackend,
}

/// Lazy safetensors backend: the shard mmaps stay alive, and each tensor is decoded
/// to f32/bf16 from its mmap on demand (one tensor at a time) — so a 31B bf16 model
/// (61 GB) loads with a peak working set of ONE tensor, not the whole model. `index`
/// maps a tensor name to the shard that holds it; the per-access cost is re-parsing
/// that shard's small safetensors header, negligible vs. copying 61 GB into RAM.
struct LazySafetensors {
    mmaps: Vec<Mmap>,
    /// tensor name → index into `mmaps`.
    index: HashMap<String, usize>,
}

enum WeightsBackend {
    /// All tensors owned in RAM (f32 or bf16). Built by `from_map` / `read_weights`.
    Eager(HashMap<String, RawTensor>),
    /// mmap'd GGUF; tensors dequantized to f32 one at a time on access.
    LazyGguf(crate::model::gguf::LazyGguf),
    /// mmap'd safetensors; each tensor decoded to f32/bf16 one at a time on access
    /// (so a model larger than RAM-as-bf16 still loads — peak = one tensor).
    LazySafetensors(LazySafetensors),
}

impl LazySafetensors {
    /// Resolve the loader's HF-Llama-style tensor name to the actual stored name.
    /// Gemma 4 ships as `Gemma4ForConditionalGeneration` (a multimodal wrapper), so
    /// its TEXT weights are nested under `model.language_model.` — e.g. the loader
    /// asks for `model.embed_tokens.weight` but the file stores
    /// `model.language_model.embed_tokens.weight`. Try the literal name first, then
    /// the gemma-4 rewrite. Returns the resolved name actually present in the index.
    fn resolve_name(&self, name: &str) -> Option<String> {
        if self.index.contains_key(name) {
            return Some(name.to_string());
        }
        // `model.<rest>` → `model.language_model.<rest>` (the gemma-4 text-stack nest).
        if let Some(rest) = name.strip_prefix("model.") {
            let gemma4 = format!("model.language_model.{rest}");
            if self.index.contains_key(&gemma4) {
                return Some(gemma4);
            }
        }
        // `<name>` → `language_model.<name>`: Gemma 3's multimodal checkpoint
        // (`Gemma3ForConditionalGeneration`, e.g. unsloth/gemma-3-4b-it, what `arf pull gemma3:4b`
        // fetches) nests the text stack the other way round — `language_model.model.layers.*`,
        // `language_model.lm_head.weight` — beside `vision_tower.*` and `multi_modal_projector.*`,
        // which a text load never asks for. Missing until 2026-10-06: the load failed with
        // `missing tensor "model.embed_tokens.weight"` (issue: a teammate's `arf run gemma3:4b`).
        let gemma3 = format!("language_model.{name}");
        if self.index.contains_key(&gemma3) {
            return Some(gemma3);
        }
        None
    }

    /// Decode tensor `name` from its mmap to an owned f32 `Vec` — the SAME bf16/f32/f16
    /// conversion as the eager `raw_from_view` path, so the loaded values are
    /// byte-identical to the eager backend (the parity contract). `None` if absent.
    ///
    /// After decoding, `madvise(MADV_DONTNEED)` releases this tensor's source pages
    /// from the OS page cache. At load we touch each tensor exactly once, so without
    /// this the whole file (61 GB bf16 for a 31B) accumulates in the unified-memory
    /// page cache and competes with the resident GPU buffers — the swap-wedge. With
    /// it, the resident page footprint stays bounded to the working set (the lean
    /// behavior ollama gets for free from its smaller pre-quantized file). Pages are
    /// re-read from disk if ever touched again (they aren't, post-load).
    fn get_f32(&self, name: &str) -> Option<Vec<f32>> {
        let resolved = self.resolve_name(name)?;
        let shard = *self.index.get(&resolved)?;
        let mmap = &self.mmaps[shard];
        // Re-parse the (small) header; slice this one tensor's bytes from the mmap.
        let st = SafeTensors::deserialize(mmap).ok()?;
        let view = st.tensor(&resolved).ok()?;
        let out = decode_view_f32(&view);
        // Release this tensor's source pages from the page cache (Unix only;
        // best-effort — ignore errors / unsupported platforms).
        #[cfg(unix)]
        {
            let bytes = view.data();
            if let Some(off) = byte_offset_in(mmap, bytes) {
                // SAFETY: the mmap is READ-ONLY (Mmap, not MmapMut), so MADV_DONTNEED
                // discards no un-synced writes — it only drops clean cached pages,
                // which are transparently re-read from the file if touched again. We
                // touch each tensor exactly once at load, so they never are.
                let _ = unsafe {
                    mmap.unchecked_advise_range(
                        memmap2::UncheckedAdvice::DontNeed,
                        off,
                        bytes.len(),
                    )
                };
            }
        }
        Some(out)
    }

    fn shape(&self, name: &str) -> Option<Vec<usize>> {
        let resolved = self.resolve_name(name)?;
        let shard = *self.index.get(&resolved)?;
        let st = SafeTensors::deserialize(&self.mmaps[shard]).ok()?;
        Some(st.tensor(&resolved).ok()?.shape().to_vec())
    }
}

/// Byte offset of slice `inner` within `outer` (the mmap), if `inner` lies inside
/// it. Used to `madvise` exactly a tensor's page range. `None` if `inner` is not a
/// sub-slice of `outer` (then we skip the advise rather than madvise a wrong range).
#[cfg(unix)]
fn byte_offset_in(outer: &[u8], inner: &[u8]) -> Option<usize> {
    let (base, ip) = (outer.as_ptr() as usize, inner.as_ptr() as usize);
    let end = base.checked_add(outer.len())?;
    if ip >= base && ip.checked_add(inner.len())? <= end {
        Some(ip - base)
    } else {
        None
    }
}

/// Decode a safetensors view's raw bytes to f32 — bf16/f16 widen, f32 passthrough.
/// Mirrors `raw_from_view` exactly so the lazy and eager backends agree bit-for-bit.
fn decode_view_f32(view: &TensorView) -> Vec<f32> {
    let bytes = view.data();
    match view.dtype() {
        Dtype::BF16 => bytes
            .chunks_exact(2)
            .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
            .collect(),
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        Dtype::F16 => bytes
            .chunks_exact(2)
            .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
            .collect(),
        _ => Vec::new(),
    }
}

impl Default for Weights {
    fn default() -> Self {
        Weights {
            backend: WeightsBackend::Eager(HashMap::new()),
        }
    }
}

impl Weights {
    /// Build from f32 tensors (used by tests and synthetic models).
    pub fn from_map(map: HashMap<String, Tensor>) -> Self {
        let tensors = map
            .into_iter()
            .map(|(k, t)| {
                let shape = t.shape().to_vec();
                (
                    k,
                    RawTensor {
                        shape,
                        data: RawData::F32(t.into_vec()),
                    },
                )
            })
            .collect();
        Weights {
            backend: WeightsBackend::Eager(tensors),
        }
    }

    /// Build from a lazily-dequantizing GGUF backend (the production 30B path).
    pub(crate) fn from_lazy_gguf(lazy: crate::model::gguf::LazyGguf) -> Self {
        Weights {
            backend: WeightsBackend::LazyGguf(lazy),
        }
    }

    /// Build from mmap'd safetensors, decoding each tensor on demand (the large-model
    /// path: a 31B bf16 checkpoint is 61 GB and won't fit RAM as owned copies, but
    /// quantizes one tensor at a time off the mmap). See [`read_weights_lazy`].
    fn from_lazy_safetensors(lazy: LazySafetensors) -> Self {
        Weights {
            backend: WeightsBackend::LazySafetensors(lazy),
        }
    }

    /// True when the weights came from a GGUF file. Matters for Gemma: llama.cpp's
    /// HF→GGUF converter bakes the `(1 + w)` RMSNorm shift into every `*norm.weight`
    /// at conversion time (the "layernorm1p" trick), so a GGUF already stores the
    /// shifted weight. Raw HF/safetensors weights are unshifted and need the `+1`
    /// folded in at load. Folding twice corrupts every norm → finite garbage output.
    pub fn is_gguf(&self) -> bool {
        matches!(self.backend, WeightsBackend::LazyGguf(_))
    }

    /// The native [`Quant`] this checkpoint already stores its bulk weights in, so the
    /// loader can default a GGUF to its OWN lossless block path instead of bf16-expanding
    /// it (the load-time OOM driver: a 6.5 GB Q4_0 GGUF becomes ~24 GB bf16 under the old
    /// `Quant::None` default). `None` for safetensors / full-precision GGUFs (keep bf16).
    /// See `LazyGguf::dominant_quant`.
    pub fn native_quant_hint(&self) -> Option<Quant> {
        match &self.backend {
            WeightsBackend::LazyGguf(g) => g.dominant_quant(),
            _ => None,
        }
    }

    /// The raw ggml type a tensor is STORED as (12 = Q4_K, 13 = Q5_K,
    /// 14 = Q6_K, 8 = Q8_0, 23 = IQ4_XS, 0 = F32), by HF name.
    ///
    /// Lets a backend avoid re-quantising DOWN: see
    /// `LazyGguf::stored_type_by_hf_name`.
    pub fn stored_ggml_type(&self, name: &str) -> Option<u32> {
        match &self.backend {
            WeightsBackend::LazyGguf(g) => g.stored_type_by_hf_name(name),
            _ => None,
        }
    }

    /// The byte length of the checkpoint backing this `Weights` (mmap length for the
    /// lazy backends, owned-data sum for eager). Feeds the load-time wired-memory
    /// guard: combined with [`Self::native_quant_hint`] it estimates the parameter
    /// count, and hence what the checkpoint will occupy at any target [`Quant`].
    pub fn backing_bytes(&self) -> u64 {
        match &self.backend {
            WeightsBackend::Eager(map) => map
                .values()
                .map(|t| match &t.data {
                    RawData::F32(v) => v.len() as u64 * 4,
                    RawData::Bf16(v) => v.len() as u64 * 2,
                })
                .sum(),
            WeightsBackend::LazyGguf(g) => g.backing_len(),
            WeightsBackend::LazySafetensors(s) => s.mmaps.iter().map(|m| m.len() as u64).sum(),
        }
    }

    /// Read a GGUF metadata scalar by key as `u32` (`None` for safetensors or absent).
    /// Used by the MTP draft-head loader to read its `gemma4-assistant.*` config dims
    /// (block_count, embedding_length) from the draft GGUF's own metadata.
    pub fn metadata_u32(&self, key: &str) -> Option<u32> {
        match &self.backend {
            WeightsBackend::LazyGguf(g) => g.get_metadata_u32(key),
            _ => None,
        }
    }

    /// The stored shape of a tensor, or `None` if absent. Used at load time to
    /// reconcile a hardcoded `--arch` config against the actual checkpoint (e.g.
    /// Gemma's vocab is padded differently across GGUF exports: 262144 in the
    /// ollama blob vs 262208 from the HF/Unsloth converter).
    pub fn shape(&self, name: &str) -> Option<Vec<usize>> {
        match &self.backend {
            WeightsBackend::Eager(map) => map.get(name).map(|t| t.shape.clone()),
            WeightsBackend::LazyGguf(g) => g.shape(name),
            WeightsBackend::LazySafetensors(s) => s.shape(name),
        }
    }

    /// Fetch a tensor's values as f32 (widening bf16 / dequantizing GGUF blocks),
    /// ignoring its shape. For inspection/probing (e.g. validating a GGUF load);
    /// `None` if absent.
    pub fn get_f32(&self, name: &str) -> Option<Vec<f32>> {
        match &self.backend {
            WeightsBackend::Eager(t) => t.get(name).map(|t| match &t.data {
                RawData::F32(v) => v.clone(),
                RawData::Bf16(v) => v.iter().map(|&b| bf16_to_f32(b)).collect(),
            }),
            WeightsBackend::LazyGguf(g) => g.get_f32(name),
            WeightsBackend::LazySafetensors(s) => s.get_f32(name),
        }
    }

    /// Fetch a tensor as an owned f32 `Vec`, validating presence and exact shape.
    /// Both backends return owned data — the lazy GGUF backend cannot hand out a
    /// borrow into freshly-dequantized data, so this is owned-returning (a clone on
    /// the eager path; build-time only). The single internal chokepoint every
    /// accessor below is built on. Missing/wrong-shape → [`ArfError::ModelLoad`].
    pub fn get_f32_shaped(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        match &self.backend {
            WeightsBackend::Eager(map) => {
                let t = map
                    .get(name)
                    .ok_or_else(|| ArfError::model_load(format!("missing tensor {name:?}")))?;
                if t.shape != shape {
                    return Err(ArfError::model_load(format!(
                        "tensor {name:?} has shape {:?}, expected {shape:?}",
                        t.shape
                    )));
                }
                Ok(match &t.data {
                    RawData::F32(v) => v.clone(),
                    RawData::Bf16(v) => v.iter().map(|&b| bf16_to_f32(b)).collect(),
                })
            }
            WeightsBackend::LazyGguf(g) => {
                let got = g
                    .shape(name)
                    .ok_or_else(|| ArfError::model_load(format!("missing tensor {name:?}")))?;
                if got != shape {
                    return Err(ArfError::model_load(format!(
                        "tensor {name:?} has shape {got:?}, expected {shape:?}"
                    )));
                }
                g.get_f32(name)
                    .ok_or_else(|| ArfError::model_load(format!("missing tensor {name:?}")))
            }
            WeightsBackend::LazySafetensors(s) => {
                let got = s
                    .shape(name)
                    .ok_or_else(|| ArfError::model_load(format!("missing tensor {name:?}")))?;
                if got != shape {
                    return Err(ArfError::model_load(format!(
                        "tensor {name:?} has shape {got:?}, expected {shape:?}"
                    )));
                }
                s.get_f32(name)
                    .ok_or_else(|| ArfError::model_load(format!("missing tensor {name:?}")))
            }
        }
    }

    /// Fetch a vector as f32 (widening/dequantizing if needed). For norm weights.
    pub fn f32_vec(&self, name: &str, len: usize) -> Result<Vec<f32>> {
        self.get_f32_shaped(name, &[len])
    }

    /// Fetch a matrix as a resident bf16 matrix (rounding the f32 source).
    pub fn bf16_matrix(&self, name: &str, rows: usize, cols: usize) -> Result<Bf16Matrix> {
        let v = self.get_f32_shaped(name, &[rows, cols])?;
        Ok(Bf16Matrix::from_f32(&v, rows, cols))
    }

    /// Fetch a matrix quantized to per-row int8.
    ///
    /// The f32 source is rounded through bf16 FIRST so the quantization input is
    /// identical to the GPU build path (`bf16_matrix().to_f32()`) and to real bf16
    /// checkpoints — CPU and GPU then quantize the same values, a requirement for
    /// the GPU==CPU parity oracle. (bf16 sources widen to f32 then round back, which
    /// is lossless; f32 test sources round once — so the oracle is exact.)
    pub fn q8_matrix(&self, name: &str, rows: usize, cols: usize) -> Result<Q8Matrix> {
        let v = self.get_f32_shaped(name, &[rows, cols])?;
        Ok(Q8Matrix::from_bf16(&to_bf16(&v), rows, cols))
    }

    /// Fetch a matrix quantized to Q4_0 block-32 4-bit. Like [`Self::q8_matrix`],
    /// the f32 source is rounded through bf16 first to match the GPU/checkpoint path.
    pub fn q4_matrix(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<crate::tensor::Q4Matrix> {
        let v = self.get_f32_shaped(name, &[rows, cols])?;
        Ok(crate::tensor::Q4Matrix::from_bf16(&to_bf16(&v), rows, cols))
    }

    /// Fetch a matrix quantized to Q4_K-lite (super-block-256). f32 source rounded
    /// through bf16 first to match the GPU/checkpoint path (see q8_matrix).
    pub fn q4k_matrix(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<crate::tensor::Q4KMatrix> {
        let v = self.get_f32_shaped(name, &[rows, cols])?;
        Ok(crate::tensor::Q4KMatrix::from_bf16(
            &to_bf16(&v),
            rows,
            cols,
        ))
    }

    /// Fetch a matrix quantized to Q4_K_S (u8/i8 two-level scales).
    pub fn q4ks_matrix(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<crate::tensor::Q4KSMatrix> {
        let v = self.get_f32_shaped(name, &[rows, cols])?;
        Ok(crate::tensor::Q4KSMatrix::from_bf16(
            &to_bf16(&v),
            rows,
            cols,
        ))
    }

    /// Fetch a matrix quantized to Q3_K (3-bit split-plane + two-level scales).
    pub fn q3k_matrix(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<crate::tensor::Q3KMatrix> {
        let v = self.get_f32_shaped(name, &[rows, cols])?;
        Ok(crate::tensor::Q3KMatrix::from_bf16(
            &to_bf16(&v),
            rows,
            cols,
        ))
    }
    /// Raw native **IQ4_XS** block bytes, for a backend that packs them itself.
    ///
    /// Returns None unless the tensor is stored as IQ4_XS in a lazy GGUF, so the
    /// caller falls back rather than guessing. A backend's loader can use this to
    /// hold IQ4_XS at its own 4.25 bpw instead of dequantizing to f32 and
    /// re-packing to 8.
    /// Raw native **Q4_K** block bytes by HF name (144 bytes per 256 weights), when the tensor
    /// is stored Q4_K in a GGUF and carries no head permute. The token embedding uses it
    /// (2026-09-23): held at its own 4.5 bpw instead of widened to bf16.
    pub fn q4k_blocks(&self, name: &str) -> Option<(&[u8], usize, usize)> {
        match &self.backend {
            WeightsBackend::LazyGguf(g) => g.get_q4k_blocks(name),
            _ => None,
        }
    }

    /// Raw native **Q4_1** block bytes by HF name (the group-64 carrier, see `g64_from_q4_1`).
    pub fn q4_1_blocks(&self, name: &str) -> Option<(&[u8], usize, usize)> {
        match &self.backend {
            WeightsBackend::LazyGguf(g) => g.get_q4_1_blocks(name),
            _ => None,
        }
    }

    pub fn iq4_xs_blocks(&self, name: &str) -> Option<(&[u8], usize, usize)> {
        match &self.backend {
            WeightsBackend::LazyGguf(g) => g.get_iq4_xs_blocks(name),
            _ => None,
        }
    }

    /// Transcode a GGUF-native Q4_K tensor LOSSLESSLY into Q4KS . Returns
    /// `None` if this `Weights` is not GGUF-backed or the tensor is not native Q4_K
    /// (caller falls back to the self-quantized from_f32 path).
    pub fn q4ks_matrix_from_gguf_q4k(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Option<crate::tensor::Q4KSMatrix> {
        match &self.backend {
            WeightsBackend::LazyGguf(g) => {
                let (blocks, r, c) = g.get_q4k_blocks(name)?;
                // ⚠️ HARD check, not debug_assert. This was a debug_assert_eq! and therefore
                // COMPILED OUT IN RELEASE — a shape mismatch sailed past it and blew up two
                // frames later inside from_ggml_q4k as an opaque "blocks length mismatch",
                // which cost real debugging time on the qwen3.8 port. A caller asking for the
                // wrong shape must fail HERE, where the name is still in hand.
                if (r, c) != (rows, cols) {
                    panic!(
                        "q4ks_matrix_from_gguf_q4k: shape mismatch for {name}: \
                         file has ({r},{c}), caller asked ({rows},{cols})"
                    );
                }
                Some(crate::tensor::Q4KSMatrix::from_ggml_q4k(blocks, rows, cols))
            }
            _ => None,
        }
    }

    /// L212 — Q3_K from a GGUF-native **Q4_K** tensor: dequantize the native blocks to f32,
    /// then requantize to Q3_K. This is the "genuine piece of work" the `q3k_mtl` comment in
    /// arf-gpu/src/weights.rs calls for.
    ///
    /// WHY IT CANNOT BE A SOURCE SWAP. `ARF_Q3K_FFN` reads `bf16_matrix`, which has nothing
    /// to return on a native-Q4_K GGUF, so the flag has been a SILENT NO-OP on every such model
    /// (L159). Routing it through `q4k_matrix` does not help — that helper is also built on
    /// `get_f32_shaped`, i.e. it also wants an f32/bf16 source. Wiring it that way produced
    /// 92.75 tok/s of pure garbage: 4.3x past the bandwidth roofline, the tell that the FFN was
    /// reading NOTHING rather than reading less.
    ///
    /// This goes through the bytes instead: `get_q4k_blocks` → `dequant_q4_k` → `Q3KMatrix`.
    /// It IS double quantization (Q4_K → f32 → Q3_K) and therefore lossy on top of lossy; the
    /// caller must gate it on measured output quality, not just on tok/s.
    pub fn q3k_matrix_from_gguf_q4k(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Option<crate::tensor::Q3KMatrix> {
        match &self.backend {
            WeightsBackend::LazyGguf(g) => {
                let (blocks, r, c) = g.get_q4k_blocks(name)?;
                if (r, c) != (rows, cols) {
                    panic!(
                        "q3k_matrix_from_gguf_q4k: shape mismatch for {name}: \
                         file has ({r},{c}), caller asked ({rows},{cols})"
                    );
                }
                let f = crate::model::gguf::dequant_q4_k(blocks, rows * cols).ok()?;
                Some(crate::tensor::Q3KMatrix::from_f32(&f, rows, cols))
            }
            _ => None,
        }
    }

    /// Transcode a GGUF-native **Q4_0** tensor LOSSLESSLY into `Q4Matrix` (the Q4_0
    /// analog of [`Self::q4ks_matrix_from_gguf_q4k`]). Avoids the dequant→re-quant
    /// double-quantization that adds sampling-visible logit noise on Q4_0 GGUFs. `None`
    /// if not GGUF-backed or the tensor is not native Q4_0 (caller falls back to from_f32).
    pub fn q4_matrix_from_gguf_q4_0(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Option<crate::tensor::Q4Matrix> {
        match &self.backend {
            WeightsBackend::LazyGguf(g) => {
                let (blocks, r, c) = g.get_q4_0_blocks(name)?;
                debug_assert_eq!(
                    (r, c),
                    (rows, cols),
                    "q4_matrix_from_gguf_q4_0: shape mismatch for {name}: got ({r},{c}), expected ({rows},{cols})"
                );
                Some(crate::tensor::Q4Matrix::from_ggml_q4_0(blocks, rows, cols))
            }
            _ => None,
        }
    }
}

/// Memory-map a model file read-only. One of the crate's only two `unsafe` sites
/// (the other is the MADV_DONTNEED advise above): the mapping is
/// read-only and never aliased mutably; callers copy bytes into owned buffers and
/// drop the mapping. The file must not be modified concurrently (the standard mmap
/// contract). Shared by the safetensors and GGUF loaders.
pub(crate) fn mmap_file(path: &Path) -> Result<Mmap> {
    let file = std::fs::File::open(path)
        .map_err(|e| ArfError::model_load(format!("open {path:?}: {e}")))?;
    // SAFETY: see the doc comment above — read-only, single mmap contract.
    unsafe { Mmap::map(&file) }.map_err(|e| ArfError::model_load(format!("mmap {path:?}: {e}")))
}

/// Round an f32 slice to bf16 bit patterns (the precision real checkpoints and the
/// GPU build path both use before quantizing).
fn to_bf16(v: &[f32]) -> Vec<u16> {
    v.iter()
        .copied()
        .map(crate::tensor::dtype::f32_to_bf16)
        .collect()
}

/// Build a CPU [`Llama`] from a [`Weights`] store. `max_positions` sizes RoPE.
pub fn build(cfg: &ModelConfig, w: &Weights, max_positions: usize) -> Result<Llama> {
    build_quant(cfg, w, max_positions, &Device::Cpu, Quant::None)
}

/// Build a [`Llama`] on `device` (uploads `Linear` weights to the GPU if so).
pub fn build_on(
    cfg: &ModelConfig,
    w: &Weights,
    max_positions: usize,
    device: &Device,
) -> Result<Llama> {
    build_quant(cfg, w, max_positions, device, Quant::None)
}

/// Build a [`Llama`] on `device` with weight quantization `quant`. `Int8`
/// quantizes the projections and the lm_head to per-row int8 (the embedding
/// table stays bf16 for the gather); `None` keeps everything bf16.
pub fn build_quant(
    cfg: &ModelConfig,
    w: &Weights,
    max_positions: usize,
    device: &Device,
    quant: Quant,
) -> Result<Llama> {
    cfg.validate()?;
    let h = cfg.hidden_size;
    // q/k/v projection out-dims are per-layer (Gemma 4 global layers differ) — see
    // `cfg.layer_geometry(i)` inside the layer loop below.
    // A projection: int8 on the CPU when requested, else bf16 (and GPU-uploaded
    // on a GPU device). int8 is CPU-resident in this phase.
    let lin = |name: &str, out: usize, in_: usize| -> Result<Linear> {
        Ok(match quant {
            Quant::Int8 => Linear::from_q8(Arc::new(w.q8_matrix(name, out, in_)?)),
            Quant::Q4 => Linear::from_q4(Arc::new(w.q4_matrix(name, out, in_)?)),
            Quant::Q4K => Linear::from_q4k(Arc::new(w.q4k_matrix(name, out, in_)?)),
            Quant::Q4KS => Linear::from_q4ks(Arc::new(w.q4ks_matrix(name, out, in_)?)),
            Quant::Q3K => Linear::from_q3k(Arc::new(w.q3k_matrix(name, out, in_)?)),
            Quant::None => Linear::on_device(Arc::new(w.bf16_matrix(name, out, in_)?), device),
        })
    };
    // Gemma's RMSNorm scales by `(1 + weight)`; folding the `+1` into the loaded
    // weight reproduces it exactly with the standard `weight · x` norm (and keeps
    // the GPU kernel unchanged). Llama/Qwen use the weight as-is.
    //
    // BUT only for *raw* HF weights: llama.cpp's HF→GGUF converter already bakes the
    // `+1` into every `*norm.weight` (the "layernorm1p" trick), so a GGUF arrives
    // pre-shifted. Folding again would store `(2 + w)` and corrupt every norm.
    let is_gemma = cfg.norm_style == NormStyle::Gemma;
    let fold_gemma_norm = is_gemma && !w.is_gguf();
    let norm_offset = |mut v: Vec<f32>| -> Vec<f32> {
        if fold_gemma_norm {
            for x in &mut v {
                *x += 1.0;
            }
        }
        v
    };
    let norm = |name: &str| -> Result<RmsNorm> {
        Ok(RmsNorm::new(
            Tensor::from_vec(norm_offset(w.f32_vec(name, h)?), vec![h]),
            cfg.rms_norm_eps,
        ))
    };

    let embed_mat = Arc::new(w.bf16_matrix("model.embed_tokens.weight", cfg.vocab_size, h)?);
    let embed = Embedding::new(embed_mat.clone());

    let mut layers = Vec::with_capacity(cfg.num_layers);
    for i in 0..cfg.num_layers {
        let p = format!("model.layers.{i}");
        // Per-layer geometry: Gemma 4's GLOBAL layers use a different head_dim /
        // kv_heads / rotary_dim than the sliding ones (cfg.layer_geometry). Every
        // other model returns the base values for all layers, so this is identical to
        // the old uniform q_dim/kv_dim there.
        let (l_head_dim, l_kv_heads, l_rotary_dim) = cfg.layer_geometry(i);
        let l_q_dim = cfg.num_attention_heads * l_head_dim;
        let l_kv_dim = l_kv_heads * l_head_dim;
        let k_proj = lin(&format!("{p}.self_attn.k_proj.weight"), l_kv_dim, h)?;
        // Gemma 4's GLOBAL layers share K as V (no `attn_v.weight` in the GGUF —
        // `shared_kv_layers`); sliding layers carry a distinct V. Load V if present,
        // else clone K (`Linear` is Arc-backed). The weightless V-norm (`cfg.value_norm`)
        // is a separate post-projection step.
        let v_name = format!("{p}.self_attn.v_proj.weight");
        let v_proj = if w.shape(&v_name).is_some() {
            lin(&v_name, l_kv_dim, h)?
        } else {
            k_proj.clone()
        };
        let mut attn = Attention::new(
            lin(&format!("{p}.self_attn.q_proj.weight"), l_q_dim, h)?,
            k_proj,
            v_proj,
            lin(&format!("{p}.self_attn.o_proj.weight"), h, l_q_dim)?,
            cfg,
        )
        .with_geometry(l_head_dim, l_kv_heads, l_rotary_dim);
        // Per-head q/k RMSNorm before RoPE (Qwen3/Gemma); Llama has none. Gemma's
        // q/k norms are also `(1 + weight)`, so fold the `+1` the same way — and,
        // identically, only for raw HF weights (GGUF already baked it in).
        if cfg.qk_norm {
            // q/k RMSNorm weights are `[head_dim]` — per-layer (Gemma 4 global layers
            // are 512-wide vs the sliding 256).
            let qk = |name: &str| -> Result<crate::model::qk_norm::QkNorm> {
                let mut v = w.f32_vec(name, l_head_dim)?;
                if fold_gemma_norm {
                    for x in &mut v {
                        *x += 1.0;
                    }
                }
                Ok(crate::model::qk_norm::QkNorm::new(v, cfg.rms_norm_eps))
            };
            attn = attn.with_qk_norm(
                qk(&format!("{p}.self_attn.q_norm.weight"))?,
                qk(&format!("{p}.self_attn.k_norm.weight"))?,
            );
        }
        // Gemma interleaves sliding-window *local* layers with periodic *global*
        // ones (every `global_every`-th layer is global); local layers also use
        // the local RoPE base. Llama/Qwen are global/causal throughout.
        if let AttnKind::HybridLocalGlobal {
            window,
            global_every,
            ..
        } = &cfg.attn
        {
            let is_global = (i + 1) % global_every.max(&1) == 0;
            attn = attn.with_window(if is_global { None } else { Some(*window) });
        }
        // Feed-forward: dense SwiGLU (Llama/Gemma) or MoE (Qwen3). A small SwiGLU
        // builder reused for the dense block, each expert, and the shared expert.
        let swiglu = |prefix: &str, inter: usize| -> Result<Mlp> {
            Ok(Mlp::new(
                lin(&format!("{prefix}.gate_proj.weight"), inter, h)?,
                lin(&format!("{prefix}.up_proj.weight"), inter, h)?,
                lin(&format!("{prefix}.down_proj.weight"), h, inter)?,
            ))
        };
        let mlp = match &cfg.mlp {
            MlpKind::Dense => MlpBlock::Dense(swiglu(&format!("{p}.mlp"), cfg.intermediate_size)?),
            MlpKind::Moe {
                num_experts,
                top_k,
                shared_experts,
                moe_intermediate,
                norm_topk,
            } => {
                let router = lin(&format!("{p}.mlp.gate.weight"), *num_experts, h)?;
                let experts = (0..*num_experts)
                    .map(|e| swiglu(&format!("{p}.mlp.experts.{e}"), *moe_intermediate))
                    .collect::<Result<Vec<_>>>()?;
                // Qwen3 has 0 or 1 shared expert (`mlp.shared_expert.*`).
                let shared = (0..*shared_experts)
                    .map(|_| swiglu(&format!("{p}.mlp.shared_expert"), *moe_intermediate))
                    .collect::<Result<Vec<_>>>()?;
                MlpBlock::Moe(crate::model::moe::MoeMlp::new(
                    router, experts, shared, *top_k, *norm_topk,
                ))
            }
        };
        // Gemma has four norms (a post-norm around each sub-block, applied before
        // the residual add); Llama/Qwen have two. For Llama the pre-MLP norm is
        // `post_attention_layernorm`; for Gemma it is `pre_feedforward_layernorm`.
        let (post_attn_norm, pre_ffn_norm, post_ffn_norm) = if is_gemma {
            (
                Some(norm(&format!("{p}.post_attention_layernorm.weight"))?),
                norm(&format!("{p}.pre_feedforward_layernorm.weight"))?,
                Some(norm(&format!("{p}.post_feedforward_layernorm.weight"))?),
            )
        } else {
            (
                None,
                norm(&format!("{p}.post_attention_layernorm.weight"))?,
                None,
            )
        };
        layers.push(DecoderLayer::new(
            norm(&format!("{p}.input_layernorm.weight"))?,
            attn,
            post_attn_norm,
            pre_ffn_norm,
            post_ffn_norm,
            mlp,
        ));
    }

    let final_norm = norm("model.norm.weight")?;
    let lm_head = match (cfg.tie_word_embeddings, quant) {
        // TIED lm_head stays bf16 at EVERY quant — it reuses the embedding matrix.
        //
        // This used to quantize a separate lm_head from the same weights, which made the CPU
        // reference disagree with the GPU by 1.4-2.9% on logits and broke four parity tests
        // (L289/L290). The GPU aliases its bf16 `embed` buffer here, deliberately: it saves a
        // second multi-GB resident copy, and real GGUFs store the tied table as Q6_K/F16 so the
        // GPU's `up_w` lands on bf16 regardless of `--quant`. Quantizing only on the CPU side
        // made the ORACLE wrong, not the engine — measured 9.5x (Q4) to 33x (Q4_K_S) further
        // from the bf16 ground truth than the GPU.
        //
        // Int8 is the one exception, kept as-is: its per-row scales are a different tradeoff and
        // its parity path is separate.
        (true, Quant::Int8) => Linear::from_q8(Arc::new(w.q8_matrix(
            "model.embed_tokens.weight",
            cfg.vocab_size,
            h,
        )?)),
        (true, _) => Linear::on_device(embed_mat, device),
        // Untied: load the dedicated lm_head weight (honors `quant`).
        (false, _) => lin("lm_head.weight", cfg.vocab_size, h)?,
    };
    // Global RoPE. For Gemma 4 the GLOBAL (full-attention) layers have their own
    // head_dim (512) and PARTIAL rotary (rotary_dim 128 of 512) over the global θ
    // (1M) — distinct from the base/sliding geometry — so the global table is built
    // with `with_theta_rotary`. Every other model (and Gemma 3, whose global layers
    // match the base head_dim with full rotary) uses the plain base table.
    let rope = match &cfg.attn {
        AttnKind::HybridLocalGlobal {
            global_theta,
            global_head_dim,
            partial_rotary_factor,
            ..
        } if *global_head_dim != cfg.head_dim || *partial_rotary_factor != 1.0 => {
            // rotary_dim = round(global_head_dim * factor), even (rotates pairs).
            let rot = (*global_head_dim as f32 * partial_rotary_factor).round() as usize;
            let rot = rot - (rot % 2);
            Rope::with_theta_rotary(max_positions, *global_theta, *global_head_dim, rot)
        }
        _ => Rope::new(cfg, max_positions),
    };
    // Gemma builds a local base for its sliding-window layers (rope_local_base_freq);
    // sliding layers keep the base head_dim with full rotary.
    let rope_local = match &cfg.attn {
        AttnKind::HybridLocalGlobal { local_theta, .. } => {
            Some(Rope::with_theta(cfg, max_positions, *local_theta))
        }
        // qwen35: one rope base for the attention layers; the GDN layers do not rope at all.
        AttnKind::HybridSsmAttn { .. } | AttnKind::Causal => None,
    };

    Ok(Llama::new(
        embed,
        layers,
        final_norm,
        lm_head,
        rope,
        rope_local,
        cfg.clone(),
    ))
}

/// Load real weights from safetensors onto the CPU.
pub fn load_safetensors<P: AsRef<Path>>(
    cfg: &ModelConfig,
    paths: &[P],
    max_positions: usize,
) -> Result<Llama> {
    load_safetensors_on(cfg, paths, max_positions, &Device::Cpu)
}

/// Load real weights from safetensors onto `device`, via memory-mapped files.
pub fn load_safetensors_on<P: AsRef<Path>>(
    cfg: &ModelConfig,
    paths: &[P],
    max_positions: usize,
    device: &Device,
) -> Result<Llama> {
    load_safetensors_quant(cfg, paths, max_positions, device, Quant::None)
}

/// Read every tensor from the safetensors `paths` into an owned [`Weights`]
/// store via memory-mapping. The shared front end of the loaders below, and of
/// the GPU loaders in `arf-gpu`.
pub fn read_weights<P: AsRef<Path>>(paths: &[P]) -> Result<Weights> {
    if paths.is_empty() {
        return Err(ArfError::model_load("no safetensors files provided"));
    }
    let mut tensors = HashMap::new();
    for path in paths {
        let mmap = mmap_file(path.as_ref())?;
        let st = SafeTensors::deserialize(&mmap)?;
        for name in st.names() {
            let view = st.tensor(name)?;
            tensors.insert(name.to_string(), raw_from_view(name, &view)?);
        }
        // `mmap` is dropped here; owned copies live on in `tensors`.
    }
    Ok(Weights {
        backend: WeightsBackend::Eager(tensors),
    })
}

/// Like [`read_weights`] but LAZY: keep the shard mmaps alive and decode each tensor
/// on demand, instead of copying all tensors into owned RAM. Peak working set is one
/// tensor — so a 31B bf16 checkpoint (61 GB) loads on a machine with far less RAM
/// (the eager path would need the whole 61 GB resident at once). Values are
/// byte-identical to the eager path (`decode_view_f32` mirrors `raw_from_view`).
pub fn read_weights_lazy<P: AsRef<Path>>(paths: &[P]) -> Result<Weights> {
    if paths.is_empty() {
        return Err(ArfError::model_load("no safetensors files provided"));
    }
    let mut mmaps = Vec::with_capacity(paths.len());
    let mut index = HashMap::new();
    for (shard, path) in paths.iter().enumerate() {
        let mmap = mmap_file(path.as_ref())?;
        // Parse the header once to record which shard each tensor lives in; the
        // tensor bytes are sliced from the mmap later, on access.
        let st = SafeTensors::deserialize(&mmap)?;
        for name in st.names() {
            index.insert(name.to_string(), shard);
        }
        mmaps.push(mmap);
    }
    Ok(Weights::from_lazy_safetensors(LazySafetensors {
        mmaps,
        index,
    }))
}

/// Load real weights from safetensors onto `device` with quantization `quant`.
///
/// Uses the LAZY mmap path ([`read_weights_lazy`]) so large bf16 checkpoints load
/// with a one-tensor peak instead of materializing the whole model in RAM.
pub fn load_safetensors_quant<P: AsRef<Path>>(
    cfg: &ModelConfig,
    paths: &[P],
    max_positions: usize,
    device: &Device,
    quant: Quant,
) -> Result<Llama> {
    build_quant(
        cfg,
        &read_weights_lazy(paths)?,
        max_positions,
        device,
        quant,
    )
}

/// Copy a safetensors view into an owned [`RawTensor`], validating the byte
/// length against the declared shape/dtype (untrusted input).
fn raw_from_view(name: &str, view: &TensorView) -> Result<RawTensor> {
    let bytes = view.data();
    let shape = view.shape().to_vec();
    let elems: usize = shape.iter().product();

    let bytes_per = match view.dtype() {
        Dtype::F32 => 4,
        Dtype::BF16 | Dtype::F16 => 2,
        other => {
            return Err(ArfError::model_load(format!(
                "tensor {name:?}: unsupported dtype {other:?}"
            )))
        }
    };
    if bytes.len() != elems * bytes_per {
        return Err(ArfError::model_load(format!(
            "tensor {name:?}: {} bytes for shape {shape:?} (expected {})",
            bytes.len(),
            elems * bytes_per
        )));
    }

    let data = match view.dtype() {
        // bf16 is kept as-is (no widening) to halve resident memory.
        Dtype::BF16 => RawData::Bf16(
            bytes
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect(),
        ),
        Dtype::F32 => RawData::F32(
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
        ),
        Dtype::F16 => RawData::F32(
            bytes
                .chunks_exact(2)
                .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect(),
        ),
        _ => unreachable!("dtype matched above"),
    };
    Ok(RawTensor { shape, data })
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let frac = h & 0x3ff;
    let val = if exp == 0 {
        f32::from(frac) * 2f32.powi(-24)
    } else if exp == 0x1f {
        if frac == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        (1.0 + f32::from(frac) / 1024.0) * 2f32.powi(i32::from(exp) - 15)
    };
    if sign == 1 {
        -val
    } else {
        val
    }
}

/// GROUP-64 AFFINE Q4 FROM Q4_1 PAIRS (2026-09-25). A group-64 weight (MLX / another engine format:
/// `w = scale * q + bias`, one scale and bias per 64 inputs) is carried in a GGUF as Q4_1, two
/// blocks per group with identical `d` and `m`. This recombines them into the layout the
/// group-64 decode kernel reads — another engine's StorageN=256 tiling: the 32 bytes of (row n,
/// group g) at `((n / 256) * groups + g) * 256 + n % 256` (x 32 bytes), nibbles SEQUENTIAL low
/// first (input 2i in the low nibble of byte i); scales and biases (f16 bits) at the same
/// parameter index. Returns `None` — the caller keeps the tensor on its Q4_1 path — unless every
/// pair shares `d` and `m` (i.e. the tensor really is group-64), rows % 256 == 0 and cols % 64 == 0.
/// Lossless: the nibbles and the f16 d/m are copied, never re-quantised.
pub fn g64_from_q4_1(
    blocks: &[u8],
    rows: usize,
    cols: usize,
) -> Option<(Vec<u8>, Vec<u16>, Vec<u16>)> {
    if !rows.is_multiple_of(256)
        || !cols.is_multiple_of(64)
        || blocks.len() != rows * cols / 32 * 20
    {
        return None;
    }
    let groups = cols / 64;
    let mut w = vec![0u8; rows * cols / 2];
    let mut sc = vec![0u16; rows * groups];
    let mut bi = vec![0u16; rows * groups];
    let rd16 = |b: &[u8], o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
    for n in 0..rows {
        let row = &blocks[n * (cols / 32) * 20..(n + 1) * (cols / 32) * 20];
        for g in 0..groups {
            let (b0, b1) = (
                &row[(2 * g) * 20..(2 * g + 1) * 20],
                &row[(2 * g + 1) * 20..(2 * g + 2) * 20],
            );
            let (d0, m0, d1, m1) = (rd16(b0, 0), rd16(b0, 2), rd16(b1, 0), rd16(b1, 2));
            if d0 != d1 || m0 != m1 {
                return None; // not a group-64 tensor
            }
            let p = ((n / 256) * groups + g) * 256 + n % 256;
            sc[p] = d0;
            bi[p] = m0;
            // Q4_1 block: byte j = x[j] | x[j + 16] << 4 (j < 16). Unpack 64 codes, repack
            // sequentially: byte i = x[2i] | x[2i + 1] << 4.
            let mut x = [0u8; 64];
            for (bk, blk) in [b0, b1].iter().enumerate() {
                for j in 0..16 {
                    let v = blk[4 + j];
                    x[bk * 32 + j] = v & 0xF;
                    x[bk * 32 + j + 16] = v >> 4;
                }
            }
            let dst = &mut w[p * 32..p * 32 + 32];
            for i in 0..32 {
                dst[i] = x[2 * i] | (x[2 * i + 1] << 4);
            }
        }
    }
    Some((w, sc, bi))
}

/// GROUP-64 AT 3 BITS (2026-09-26): the tiled 4-bit layout of `g64_from_q4_1` (32-byte records per
/// (column, group), sequential nibbles) repacked IF every code is below 8 — a 3-bit tensor carried in
/// the 4-bit format (`scripts/g64/quantize_g64.py` with `G64_3BIT`). GROUP PAIRS: per (tile, groups 2p
/// and 2p+1, column) a 48-byte record; lane-quarter c (inputs 16c..16c+15 of each group) holds 12
/// contiguous bytes at \[12c\]: the even group's u32 of 2-bit lows, the odd group's, then their u16s of
/// high bits. Input o of a quarter sits at bits 2i+16h (lows) / i+8h (highs), i = (o&3) + 4(o>>3),
/// h = (o>>2)&1 — what the kernels' one-shift pair extraction reads (matmul_g64_msl.metal,
/// `g64_mm_impl` BITS 3). `cols / 64` must be even. `None` (keep 4 bits) as soon as one code is 8+.
pub fn g64_pack3(w4: &[u8], cols: usize) -> Option<Vec<u8>> {
    let groups = cols / 64;
    if !cols.is_multiple_of(128) || !w4.len().is_multiple_of(32 * 256 * groups) {
        return None;
    }
    let recs = w4.len() / 32;
    let mut w3 = vec![0u8; recs * 24];
    for (p, src) in w4.chunks_exact(32).enumerate() {
        let (t, g, col) = (p / (groups * 256), (p / 256) % groups, p % 256);
        let rec = ((t * groups + (g & !1)) * 256 + col * 2) * 24;
        let odd = g & 1;
        let mut lo = [0u32; 4];
        let mut hi = [0u32; 4];
        for (byte_i, &b) in src.iter().enumerate() {
            for (half, v) in [(0usize, b & 0xF), (1, b >> 4)] {
                if v > 7 {
                    return None;
                }
                let k = 2 * byte_i + half; // input within the group
                let (c, o) = (k / 16, k % 16);
                let (i, h) = ((o & 3) + 4 * (o >> 3), (o >> 2) & 1);
                lo[c] |= u32::from(v & 3) << (2 * i + 16 * h);
                hi[c] |= u32::from((v >> 2) & 1) << (i + 8 * h);
            }
        }
        for c in 0..4 {
            let q = rec + 12 * c;
            w3[q + 4 * odd..q + 4 * odd + 4].copy_from_slice(&lo[c].to_le_bytes());
            w3[q + 8 + 2 * odd..q + 10 + 2 * odd].copy_from_slice(&(hi[c] as u16).to_le_bytes());
        }
    }
    Some(w3)
}

/// f32 -> IEEE half bits, round to nearest even, subnormals and overflow (to inf) handled.
pub fn f32_to_f16_rne(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7f_ffff;
    if e == 0xff {
        return sign | 0x7c00 | if m != 0 { 0x200 } else { 0 };
    }
    let exp = e - 127 + 15;
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        // half subnormal: frac = 1.m * 2^(exp - 14) in units of 2^-24
        let shift = (14 - exp) as u32;
        if shift > 24 {
            return sign;
        }
        let mf = m | 0x80_0000;
        let mut h = mf >> shift;
        let rem = mf & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        if rem > half || (rem == half && h & 1 == 1) {
            h += 1;
        }
        return sign | h as u16;
    }
    let mut h = ((exp as u32) << 10) | (m >> 13);
    let rem = m & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) {
        h += 1; // a carry into the exponent is the correct rounding (up to inf)
    }
    sign | h as u16
}

/// One group of 64: the (scale, bias) with the least squared error over llama.cpp's
/// `make_qkx3_quants` grid (iscale = (14.1 + 0.05 j) / range, j = 0..=36) plus plain min/max, each
/// refit by least squares for its codes; both rounded to f16 and the codes re-rounded against them.
fn g64_fit(xs: &[f32]) -> ([u8; 64], u16, u16) {
    let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in xs {
        mn = mn.min(v);
        mx = mx.max(v);
    }
    let rng = mx - mn;
    let (mut bd, mut bm) = (rng / 15.0, mn);
    if rng > 0.0 {
        let sx: f32 = xs.iter().sum();
        let mut be = f32::INFINITY;
        let mut l = [0f32; 64];
        for j in 0..38 {
            let isc = if j == 37 {
                15.0 / rng
            } else {
                (14.1 + 0.05 * j as f32) / rng
            };
            let (mut sl, mut sl2, mut sxl) = (0f32, 0f32, 0f32);
            for (i, &v) in xs.iter().enumerate() {
                let q = (isc * (v - mn)).round().clamp(0.0, 15.0);
                l[i] = q;
                sl += q;
                sl2 += q * q;
                sxl += q * v;
            }
            let dd = 64.0 * sl2 - sl * sl;
            if dd <= 0.0 {
                continue;
            }
            let d = (64.0 * sxl - sx * sl) / dd;
            let m = (sl2 * sx - sl * sxl) / dd;
            let e: f32 = xs
                .iter()
                .zip(l.iter())
                .map(|(&v, &q)| (d * q + m - v).powi(2))
                .sum();
            if e < be {
                (be, bd, bm) = (e, d, m);
            }
        }
    }
    let (d16, m16) = (f32_to_f16_rne(bd), f32_to_f16_rne(bm));
    let (df, mf) = (f16_to_f32(d16), f16_to_f32(m16));
    let mut q = [0u8; 64];
    if df != 0.0 {
        for (i, &v) in xs.iter().enumerate() {
            q[i] = ((v - mf) / df).round().clamp(0.0, 15.0) as u8;
        }
    }
    (q, d16, m16)
}

/// GROUP-64 AFFINE Q4 FROM f32, SEARCHED (2026-09-25) — for weights that arrive unquantized (the
/// DFlash 2 draft's bf16 checkpoint): `g64_fit` per group, written in the same StorageN=256 tiling
/// as `g64_from_q4_1`. Threads split the 256-row tiles (each owns contiguous output). `None` unless
/// rows % 256 == 0 and cols % 64 == 0.
pub fn g64_quantize_searched(
    x: &[f32],
    rows: usize,
    cols: usize,
) -> Option<(Vec<u8>, Vec<u16>, Vec<u16>)> {
    if !rows.is_multiple_of(256) || !cols.is_multiple_of(64) || x.len() != rows * cols {
        return None;
    }
    let (groups, tiles) = (cols / 64, rows / 256);
    let mut w = vec![0u8; rows * cols / 2];
    let mut sc = vec![0u16; rows * groups];
    let mut bi = vec![0u16; rows * groups];
    let (tw, tp) = (groups * 256 * 32, groups * 256); // bytes / parameters per tile
    let threads = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(tiles)
        .max(1);
    let per = tiles.div_ceil(threads);
    std::thread::scope(|s| {
        for (k, ((wc, scc), bic)) in w
            .chunks_mut(per * tw)
            .zip(sc.chunks_mut(per * tp))
            .zip(bi.chunks_mut(per * tp))
            .enumerate()
        {
            s.spawn(move || {
                for (lt, ((wt, st), bt)) in wc
                    .chunks_mut(tw)
                    .zip(scc.chunks_mut(tp))
                    .zip(bic.chunks_mut(tp))
                    .enumerate()
                {
                    let t = k * per + lt;
                    for col in 0..256 {
                        let n = t * 256 + col;
                        for g in 0..groups {
                            let (q, d, m) = g64_fit(&x[n * cols + g * 64..n * cols + g * 64 + 64]);
                            let lp = g * 256 + col;
                            st[lp] = d;
                            bt[lp] = m;
                            for i in 0..32 {
                                wt[lp * 32 + i] = q[2 * i] | (q[2 * i + 1] << 4);
                            }
                        }
                    }
                }
            });
        }
    });
    Some((w, sc, bi))
}

#[cfg(test)]
mod tests {
    #[test]
    fn g64_pack3_places_every_code_where_the_kernel_reads_it() {
        // one tile (256 columns) x 2 groups (128 inputs), codes 0..7; the 4-bit tiled layout holds
        // them as sequential nibbles, record ((0 * groups + g) * 256 + col)
        let (groups, cols) = (2usize, 128usize);
        let code = |col: usize, g: usize, k: usize| ((col * 7 + g * 3 + k * 5) % 8) as u8;
        let mut w4 = vec![0u8; 256 * groups * 32];
        for col in 0..256 {
            for g in 0..groups {
                let p = g * 256 + col;
                for i in 0..32 {
                    w4[p * 32 + i] = code(col, g, 2 * i) | (code(col, g, 2 * i + 1) << 4);
                }
            }
        }
        let w3 = super::g64_pack3(&w4, cols).unwrap();
        assert_eq!(w3.len(), 256 * groups * 24);
        // the kernel's extraction: pair record (col * 48), quarter c at [12c], group g's lows at
        // +4(g&1), highs at +8+2(g&1); step j: inputs o0 = (j&3)+8(j>>2) and o0+4
        for col in [0usize, 17, 255] {
            for g in 0..groups {
                for c in 0..4 {
                    let q = col * 48 + 12 * c;
                    let lo = u32::from_le_bytes(
                        w3[q + 4 * (g & 1)..q + 4 * (g & 1) + 4].try_into().unwrap(),
                    );
                    let hi = u32::from(u16::from_le_bytes(
                        w3[q + 8 + 2 * (g & 1)..q + 10 + 2 * (g & 1)]
                            .try_into()
                            .unwrap(),
                    ));
                    let h32 = (hi & 0xFF) | ((hi & 0xFF00) << 8);
                    for j in 0..8u32 {
                        let pair =
                            ((lo >> (2 * j)) & 0x0003_0003) | (((h32 >> j) & 0x0001_0001) << 2);
                        let o0 = ((j & 3) + 8 * (j >> 2)) as usize;
                        assert_eq!(
                            pair & 0xFFFF,
                            u32::from(code(col, g, 16 * c + o0)),
                            "{col} {g} {c} {j}"
                        );
                        assert_eq!(
                            pair >> 16,
                            u32::from(code(col, g, 16 * c + o0 + 4)),
                            "{col} {g} {c} {j}"
                        );
                    }
                }
            }
        }
        // a code of 8 or more keeps the tensor at 4 bits
        let mut w4b = w4.clone();
        w4b[3] = 0x80;
        assert!(super::g64_pack3(&w4b, cols).is_none());
    }

    #[test]
    fn f32_to_f16_rne_matches_the_obvious_values() {
        use super::{f16_to_f32, f32_to_f16_rne};
        for v in [
            0.0f32,
            1.0,
            -2.5,
            65504.0,
            6.1035156e-5,
            5.9604645e-8,
            1.0e-7,
            0.33333334,
        ] {
            let h = f32_to_f16_rne(v);
            let back = f16_to_f32(h);
            // within half an f16 ulp (relative 2^-11 for normals, 2^-25 absolute for subnormals)
            assert!(
                (back - v).abs() <= (v.abs() * 4.9e-4).max(3.0e-8),
                "{v} -> {h:#06x} -> {back}"
            );
        }
        assert_eq!(f32_to_f16_rne(1.0e6), 0x7c00);
    }

    #[test]
    fn g64_quantize_searched_beats_plain_min_max_and_decodes() {
        let (rows, cols) = (256usize, 128usize);
        let mut s = 0x9E3779B97F4A7C15u64;
        let x: Vec<f32> = (0..rows * cols)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s % 20001) as f32 / 10000.0 - 1.0) * if s.is_multiple_of(13) { 4.0 } else { 1.0 }
            })
            .collect();
        let (w, sc, bi) = super::g64_quantize_searched(&x, rows, cols).unwrap();
        let groups = cols / 64;
        let (mut e_s, mut e_p) = (0f64, 0f64);
        for n in 0..rows {
            for g in 0..groups {
                let p = ((n / 256) * groups + g) * 256 + n % 256;
                let (d, m) = (super::f16_to_f32(sc[p]), super::f16_to_f32(bi[p]));
                let xs = &x[n * cols + g * 64..n * cols + g * 64 + 64];
                let (mn, mx) = xs
                    .iter()
                    .fold((f32::MAX, f32::MIN), |a, &v| (a.0.min(v), a.1.max(v)));
                for (i, &v) in xs.iter().enumerate() {
                    let q = (w[p * 32 + i / 2] >> (4 * (i % 2))) & 15;
                    e_s += ((d * q as f32 + m - v) as f64).powi(2);
                    let dp = (mx - mn) / 15.0;
                    let qp = ((v - mn) / dp).round().clamp(0.0, 15.0);
                    e_p += ((dp * qp + mn - v) as f64).powi(2);
                }
            }
        }
        assert!(e_s < e_p, "searched {e_s} vs plain {e_p}");
    }

    use super::*;

    /// The load-bearing contract for the lazy-safetensors backend: a tensor decoded
    /// LAZILY (mmap, on demand) must be byte-identical to the EAGER path — same
    /// values feed the quantizer, so the GPU==CPU parity oracle is unaffected.
    #[test]
    fn lazy_safetensors_is_byte_identical_to_eager() {
        use safetensors::tensor::{serialize_to_file, Dtype as StDtype, TensorView};
        // Two tensors: one bf16 [2,3], one f32 [4].
        let bf16_vals: [u16; 6] = [0x3F80, 0x4000, 0xBF80, 0x4040, 0x0000, 0x3C00];
        let mut bf16_bytes = Vec::new();
        for v in bf16_vals {
            bf16_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let f32_vals: [f32; 4] = [1.5, -2.25, 0.0, 7.0];
        let mut f32_bytes = Vec::new();
        for v in f32_vals {
            f32_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let views = vec![
            (
                "mat".to_string(),
                TensorView::new(StDtype::BF16, vec![2, 3], &bf16_bytes).unwrap(),
            ),
            (
                "vec".to_string(),
                TensorView::new(StDtype::F32, vec![4], &f32_bytes).unwrap(),
            ),
        ];
        let path = std::env::temp_dir().join(format!(
            "arf_lazy_st_test_{}.safetensors",
            std::process::id()
        ));
        serialize_to_file(views, None, &path).unwrap();

        let eager = read_weights(&[&path]).unwrap();
        let lazy = read_weights_lazy(&[&path]).unwrap();

        // shapes agree
        assert_eq!(eager.shape("mat"), lazy.shape("mat"));
        assert_eq!(eager.shape("vec"), lazy.shape("vec"));
        // values byte-identical (the contract)
        assert_eq!(
            eager.get_f32_shaped("mat", &[2, 3]).unwrap(),
            lazy.get_f32_shaped("mat", &[2, 3]).unwrap(),
            "bf16 tensor must decode identically lazy vs eager"
        );
        assert_eq!(
            eager.get_f32_shaped("vec", &[4]).unwrap(),
            lazy.get_f32_shaped("vec", &[4]).unwrap(),
            "f32 tensor must decode identically lazy vs eager"
        );
        // a missing tensor errors the same way
        assert!(lazy.get_f32_shaped("nope", &[1]).is_err());
        let _ = std::fs::remove_file(&path);
    }

    /// Gemma 3's multimodal checkpoint (`Gemma3ForConditionalGeneration`) stores the text stack as
    /// `language_model.model.*`: the loader's `model.*` names resolve to it, and the vision
    /// tensors beside it are simply never asked for.
    #[test]
    fn gemma3_multimodal_names_resolve_to_the_text_stack() {
        use safetensors::tensor::{serialize_to_file, Dtype as StDtype, TensorView};
        let vals: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let views = vec![
            (
                "language_model.model.embed_tokens.weight".to_string(),
                TensorView::new(StDtype::F32, vec![2, 2], &bytes).unwrap(),
            ),
            (
                "language_model.model.norm.weight".to_string(),
                TensorView::new(StDtype::F32, vec![4], &bytes).unwrap(),
            ),
            (
                "vision_tower.vision_model.post_layernorm.weight".to_string(),
                TensorView::new(StDtype::F32, vec![4], &bytes).unwrap(),
            ),
        ];
        let path = std::env::temp_dir().join(format!(
            "arf_gemma3_mm_test_{}.safetensors",
            std::process::id()
        ));
        serialize_to_file(views, None, &path).unwrap();
        let w = read_weights_lazy(&[&path]).unwrap();
        assert_eq!(w.shape("model.embed_tokens.weight"), Some(vec![2, 2]));
        assert_eq!(
            w.get_f32_shaped("model.embed_tokens.weight", &[2, 2])
                .unwrap(),
            vals.to_vec()
        );
        assert_eq!(w.f32_vec("model.norm.weight", 4).unwrap(), vals.to_vec());
        assert!(w.shape("model.layers.0.self_attn.q_proj.weight").is_none());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn accessors_validate_presence_and_shape() {
        let mut m = HashMap::new();
        m.insert("mat".to_string(), Tensor::zeros(vec![2, 3]));
        m.insert("vec".to_string(), Tensor::zeros(vec![4]));
        let w = Weights::from_map(m);

        assert!(w.bf16_matrix("mat", 2, 3).is_ok());
        assert!(w.f32_vec("vec", 4).is_ok());
        assert!(matches!(
            w.bf16_matrix("mat", 3, 2),
            Err(ArfError::ModelLoad(_))
        ));
        assert!(matches!(w.f32_vec("vec", 3), Err(ArfError::ModelLoad(_))));
        assert!(matches!(
            w.bf16_matrix("missing", 1, 1),
            Err(ArfError::ModelLoad(_))
        ));
    }

    #[test]
    fn f16_known_values() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
    }
}
