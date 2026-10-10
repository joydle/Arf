//! Minimal **lazy** GGUF reader: parse a `.gguf` model file (the container
//! llama.cpp/ollama use) and serve each tensor as f32 ON DEMAND, dequantizing the
//! ggml block formats we encounter (F32, F16, BF16, Q4_K, Q5_K, Q6_K). The engine then
//! re-quantizes on load (Q4 / Q4_K-lite) exactly as it does for safetensors, so the
//! whole MoE/QK-norm/Q4 pipeline is reused.
//!
//! Laziness is not an optimization here, it's a requirement: dequantizing the whole
//! Qwen3-Coder-30B to f32 at once is ~112 GiB. [`LazyGguf`] owns the mmap, parses
//! the header + tensor table once into a name→`Entry` index, and materializes one
//! tensor (or one expert's sub-slice) per access — peak resident f32 is ONE tensor.
//!
//! Scope: read-only, single-file GGUF v3 (ollama's qwen3-coder:30b). We map ggml
//! tensor names (`blk.N.*`, `ffn_*_exps`, `ffn_gate_inp`, `attn_*_norm`,
//! `token_embd`, `output`) onto our HuggingFace-style names. The 3D packed expert
//! tensors (`[in, ff, n_expert]`) are exposed as per-expert 2D matrices
//! (`mlp.experts.E.{gate,up,down}_proj.weight`) via a per-expert reverse index —
//! each expert is a whole number of QK_K super-blocks, so its block-aligned
//! sub-slice dequantizes bit-identically to splitting the whole tensor.
//!
//! Only the formats this checkpoint uses are implemented; an unknown ggml type is a
//! clear error rather than silent corruption.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;

use crate::config::{ModelConfig, Quant};
use crate::error::{ArfError, Result};
use crate::model::gguf_tokenizer::GgufTokenizerMeta;
use crate::model::weights::{mmap_file, Weights};

/// ggml block quantization type ids (subset we read).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GgmlType {
    F32,
    F16,
    BF16,
    /// Q4_0: 32-element blocks, one f16 scale + 16 bytes of 4-bit codes; symmetric
    /// (`y = d·(q-8)`). The simplest ggml quant; gemma-4's QAT GGUF ships in it.
    Q40,
    /// Q4_1: 32-element blocks, f16 scale + f16 min + 16 bytes of 4-bit codes; asymmetric
    /// (`y = d·q + m`, q in 0..15). Some Gemma-3 Q4_0 GGUFs store a few sensitive tensors
    /// (e.g. early `ffn_down`) in Q4_1 for quality.
    Q41,
    /// Q8_0: 32-element blocks, one f16 scale + 32 int8 codes (34 B); symmetric (`y = d·q`).
    /// FLUX's T5-XXL text encoder ships in this; near-lossless 8-bit.
    Q80,
    Q4K,
    Q5K,
    Q6K,
    Iq4Nl,
    Iq4Xs,
    Q3K,
    Iq3S,
}

impl GgmlType {
    fn from_id(id: u32) -> Result<Self> {
        Ok(match id {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            // Q4_0 (id 2): the QAT GGUFs (e.g. gemma-4 q4_0) and many ollama blobs.
            2 => GgmlType::Q40,
            // Q4_1 (id 3): a few sensitive tensors in some "Q4_0" Gemma-3 GGUFs.
            3 => GgmlType::Q41,
            // Q8_0 (id 8): FLUX T5-XXL encoder, and the high-precision tensors in many GGUFs.
            8 => GgmlType::Q80,
            // bf16 is ggml type 30 (e.g. Unsloth's `*-BF16.gguf` full-precision files).
            30 => GgmlType::BF16,
            12 => GgmlType::Q4K,
            // Q5_K (id 13): even "Q4_K_S" GGUFs store token_embd/output in Q5_K.
            13 => GgmlType::Q5K,
            14 => GgmlType::Q6K,
            // IQ4_NL (20) / IQ4_XS (23): the non-linear 4-bit codebook quants that
            // Unsloth's "UD" (dynamic) GGUFs mix into an otherwise-Q4_K file.
            // Q3_K (11): UD files drop a few tensors to 3 bits.
            11 => GgmlType::Q3K,
            21 => GgmlType::Iq3S,
            20 => GgmlType::Iq4Nl,
            23 => GgmlType::Iq4Xs,
            other => {
                return Err(ArfError::model_load(format!(
                    "unsupported ggml tensor type {other} (only F32/F16/BF16/Q4_0/Q4_1/Q8_0/Q3_K/Q4_K/Q5_K/Q6_K/IQ3_S/IQ4_NL/IQ4_XS read)"
                )))
            }
        })
    }
}

/// A parsed tensor descriptor: ggml dims (fastest-varying first), type, and the
/// absolute byte offset of its data in the file.
struct TensorDesc {
    name: String,
    dims: Vec<u64>,
    ty: GgmlType,
    offset: u64,
}

/// A resolved HF name points either at a whole ggml tensor or at one expert's
/// block-aligned sub-slice of a packed 3D `*_exps` tensor. Built once at open so
/// every accessor is an O(1) lookup, never a re-scan of the tensor table.
enum Entry {
    /// A whole tensor. `vec_len = Some(n)` is a 1-D vector returning shape `[n]`;
    /// `None` is a matrix returning `[rows, cols]` (the ggml→HF `[ne1, ne0]` label).
    ///
    /// `permute_head_dim = Some(hd)` flags an attention Q/K matrix that llama.cpp
    /// row-permuted at HF→GGUF conversion: its `hd` rows per head are interleaved
    /// `[hd/2, 2]` (ggml RoPE layout) instead of HF's split-half `[2, hd/2]`. The
    /// dequant path un-permutes the rows so Arf's split-half RoPE reads them
    /// correctly. (`None` for every other tensor — they are not permuted.)
    Whole {
        desc_idx: usize,
        rows: usize,
        cols: usize,
        vec_len: Option<usize>,
        permute_head_dim: Option<usize>,
    },
    /// One expert `E` of a packed 3D tensor: a block-aligned sub-slice of `per_elems`
    /// elements at byte offset `E * ggml_byte_len(ty, per_elems)` from the tensor base.
    Expert {
        desc_idx: usize,
        expert: usize,
        per_elems: usize,
        rows: usize,
        cols: usize,
    },
    /// A ROW RANGE of a fused 2-D tensor — the qwen35 family packs q/k/v into one
    /// `attn_qkv.weight` of `[q_dim + 2*kv_dim, hidden]`, and the loader wants three separate
    /// matrices. Byte offset is `row0 * cols` elements from the base.
    ///
    /// BLOCK ALIGNMENT IS FREE HERE: the slice is along ROWS while quantization blocks run
    /// along COLS, and cols is a multiple of the 256-element super-block (qwen35: 5120 = 20
    /// super-blocks). So every row boundary is also a block boundary and no re-quantization is
    /// needed — the same property `Expert` relies on.
    RowSlice {
        desc_idx: usize,
        row0: usize,
        rows: usize,
        cols: usize,
    },
}

/// A cursor over the mmap'd file for the little-endian GGUF primitives.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }
    fn u32(&mut self) -> Result<u32> {
        let v = u32::from_le_bytes(self.take(4)?.try_into().unwrap());
        Ok(v)
    }
    fn u64(&mut self) -> Result<u64> {
        let v = u64::from_le_bytes(self.take(8)?.try_into().unwrap());
        Ok(v)
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| ArfError::model_load("GGUF truncated"))?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    /// A GGUF string: u64 length + UTF-8 bytes.
    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        let s = self.take(n)?;
        Ok(String::from_utf8_lossy(s).into_owned())
    }
    /// Read a string VALUE given its already-consumed value-type `vty` (must be 8).
    fn string_after(&mut self, vty: u32) -> Result<String> {
        if vty != 8 {
            return Err(ArfError::model_load(format!(
                "GGUF string field had value type {vty}, expected 8"
            )));
        }
        self.string()
    }
    /// Read an `array<string>` value: outer value-type(must be 9) + element-type(must
    /// be 8) + count(u64) + each as a u64-len-prefixed UTF-8 string. Caller passes the
    /// already-consumed outer value-type `vty`.
    fn string_array(&mut self, vty: u32) -> Result<Vec<String>> {
        if vty != 9 {
            return Err(ArfError::model_load(format!(
                "GGUF expected array (vty 9), got {vty}"
            )));
        }
        let ety = self.u32()?;
        if ety != 8 {
            return Err(ArfError::model_load(format!(
                "GGUF array expected strings (ety 8), got {ety}"
            )));
        }
        let n = self.u64()? as usize;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.string()?);
        }
        Ok(out)
    }

    /// Read an `array<i32>` value: outer value-type(must be 9) + element-type(must be
    /// 5) + count + elements. Caller passes the already-consumed outer value-type.
    fn i32_array(&mut self, vty: u32) -> Result<Vec<i32>> {
        if vty != 9 {
            return Err(ArfError::model_load(format!(
                "GGUF expected array (vty 9), got {vty}"
            )));
        }
        let ety = self.u32()?;
        if ety != 5 {
            return Err(ArfError::model_load(format!(
                "GGUF array expected i32 (ety 5), got {ety}"
            )));
        }
        let n = self.u64()? as usize;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(i32::from_le_bytes(self.take(4)?.try_into().unwrap()));
        }
        Ok(out)
    }

    /// Read an `array<f32>` value: outer value-type(must be 9) + element-type(must be
    /// 6) + count + elements. Caller passes the already-consumed outer value-type.
    /// Used for `clip.vision.image_mean` / `image_std` in the mmproj.
    fn f32_array(&mut self, vty: u32) -> Result<Vec<f32>> {
        if vty != 9 {
            return Err(ArfError::model_load(format!(
                "GGUF expected array (vty 9), got {vty}"
            )));
        }
        let ety = self.u32()?;
        if ety != 6 {
            return Err(ArfError::model_load(format!(
                "GGUF array expected f32 (ety 6), got {ety}"
            )));
        }
        let n = self.u64()? as usize;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(f32::from_le_bytes(self.take(4)?.try_into().unwrap()));
        }
        Ok(out)
    }

    /// Read an `array<i64>` value: outer value-type(must be 9) + element-type(must be
    /// 11) + count + elements. Caller passes the already-consumed outer value-type.
    /// Used for DFlash's `dflash.target_layers`-style tap-index metadata when a
    /// checkpoint's converter widens the tap indices to i64 (GGUF's `i32_array` above
    /// covers the narrower/common case).
    fn i64_array(&mut self, vty: u32) -> Result<Vec<i64>> {
        if vty != 9 {
            return Err(ArfError::model_load(format!(
                "GGUF expected array (vty 9), got {vty}"
            )));
        }
        let ety = self.u32()?;
        if ety != 11 {
            return Err(ArfError::model_load(format!(
                "GGUF array expected i64 (ety 11), got {ety}"
            )));
        }
        let n = self.u64()? as usize;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(i64::from_le_bytes(self.take(8)?.try_into().unwrap()));
        }
        Ok(out)
    }

    /// Read a scalar f32 (vty 6) or f64 (vty 12) metadata value into f32. Errors on any
    /// other value-type. Used for the per-tensor RMS epsilon, which some GGUFs store as f32
    /// and others as f64.
    fn scalar_f32(&mut self, vty: u32) -> Result<f32> {
        Ok(match vty {
            6 => f32::from_le_bytes(self.take(4)?.try_into().unwrap()),
            12 => f64::from_le_bytes(self.take(8)?.try_into().unwrap()) as f32,
            other => {
                return Err(ArfError::model_load(format!(
                    "GGUF scalar: non-float value type {other}"
                )))
            }
        })
    }

    /// Read a scalar value of GGUF int value-type into i64 (u8=0,i8=1,u16=2,i16=3,
    /// u32=4,i32=5,u64=10,i64=11). Caller has consumed the value-type; `vty` is passed in.
    /// Errors on a non-int `vty` (e.g. f32=6, bool=7) — callers must check the value is
    /// an integer type before dispatching here; this cannot tell wrong-dispatch from a
    /// corrupt file.
    fn scalar_int(&mut self, vty: u32) -> Result<i64> {
        Ok(match vty {
            // 0 = uint8, 1 = int8, 7 = bool — all one byte on the wire.
            0 | 7 => self.take(1)?[0] as i64,
            1 => self.take(1)?[0] as i8 as i64,
            2 => u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64,
            3 => i16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64,
            4 => self.u32()? as i64,
            5 => i32::from_le_bytes(self.take(4)?.try_into().unwrap()) as i64,
            10 => i64::try_from(self.u64()?)
                .map_err(|_| ArfError::model_load("GGUF u64 metadata value overflows i64"))?,
            11 => i64::from_le_bytes(self.take(8)?.try_into().unwrap()),
            other => {
                return Err(ArfError::model_load(format!(
                    "GGUF scalar: non-int value type {other}"
                )))
            }
        })
    }

    /// Skip one metadata value of GGUF value-type `vty` (we only need the tensor
    /// data; the architecture comes from our `ModelConfig`).
    fn skip_value(&mut self, vty: u32) -> Result<()> {
        match vty {
            0 | 1 | 7 => {
                self.take(1)?;
            } // u8/i8/bool
            2 | 3 => {
                self.take(2)?;
            } // u16/i16
            4..=6 => {
                self.take(4)?;
            } // u32/i32/f32
            10..=12 => {
                self.take(8)?;
            } // u64/i64/f64
            8 => {
                let n = self.u64()? as usize;
                self.take(n)?;
            } // string
            9 => {
                // array: element type (u32) + count (u64) + elements
                let ety = self.u32()?;
                let n = self.u64()?;
                for _ in 0..n {
                    self.skip_value(ety)?;
                }
            }
            other => {
                return Err(ArfError::model_load(format!(
                    "unknown GGUF value type {other}"
                )))
            }
        }
        Ok(())
    }
}

/// Parse a GGUF file into a [`Weights`] store backed by [`LazyGguf`]: the header
/// and tensor table are read up front, but each tensor is dequantized to f32 only
/// when an accessor asks for it. This is the only way the 30B (which would be
/// ~112 GiB if every tensor were dequantized to f32 at once) loads on a machine
/// with less RAM than that. `cfg` supplies `tie_word_embeddings` (whether the
/// GGUF's `output.weight` becomes our `lm_head`).
pub fn load_gguf(path: &Path, cfg: &ModelConfig) -> Result<Weights> {
    Ok(Weights::from_lazy_gguf(LazyGguf::open(path, cfg)?))
}

/// Capture one metadata KV into `alignment`/`tok_meta` if it's a key we care about.
/// Returns `true` if the value was consumed here, `false` if the caller must
/// `skip_value(vty)`. Shared by `LazyGguf::open` and `read_tokenizer_meta`.
///
/// `tok_meta` is `None` for callers that only need `alignment` (the model-loading
/// `open` path): the tokenizer.ggml.* keys then fall through to `false` so the
/// caller skips them — avoiding a redundant parse of the ~151k-token vocab on every
/// model load (the tokenizer is built separately via `read_tokenizer_meta`).
fn capture_meta_kv(
    r: &mut Reader<'_>,
    key: &str,
    vty: u32,
    alignment: &mut u64,
    tok_meta: Option<&mut GgufTokenizerMeta>,
) -> Result<bool> {
    if key == "general.alignment" {
        debug_assert_eq!(vty, 4);
        *alignment = r.u32()? as u64;
        return Ok(true);
    }
    let Some(tok_meta) = tok_meta else {
        return Ok(false); // alignment-only caller: skip tokenizer keys
    };
    match key {
        "tokenizer.ggml.model" => tok_meta.model = Some(r.string_after(vty)?),
        "tokenizer.ggml.pre" => tok_meta.pre = Some(r.string_after(vty)?),
        "tokenizer.ggml.tokens" => tok_meta.tokens = r.string_array(vty)?,
        "tokenizer.ggml.merges" => tok_meta.merges = r.string_array(vty)?,
        "tokenizer.ggml.token_type" => tok_meta.token_type = r.i32_array(vty)?,
        // L320 — SentencePiece (Gemma) only: whether Metaspace prepends a leading
        // space. GGUF stores bool as a single byte, which `scalar_int` reads.
        "tokenizer.ggml.add_space_prefix" => {
            tok_meta.add_space_prefix = Some(r.scalar_int(vty)? != 0)
        }
        "tokenizer.ggml.bos_token_id" => tok_meta.bos_token_id = Some(r.scalar_int(vty)?),
        "tokenizer.ggml.eos_token_id" => tok_meta.eos_token_id = Some(r.scalar_int(vty)?),
        "tokenizer.ggml.padding_token_id" => tok_meta.padding_token_id = Some(r.scalar_int(vty)?),
        _ => return Ok(false),
    }
    Ok(true)
}

/// A lazily-dequantizing GGUF backend. Owns the mmap and a name→`Entry` index
/// built once at [`open`](LazyGguf::open). Each accessor dequantizes exactly one
/// tensor (or one expert's block-aligned sub-slice) on demand, so peak resident
/// f32 is ONE tensor — never the whole model. The index reproduces the same
/// ggml→HF name mapping the eager loader used, so consumers are unchanged.
pub struct LazyGguf {
    /// Owns the mapping for the backend's lifetime (every deferred read borrows it).
    /// `Arc<Mmap>` is `Send + Sync` (so the backend can move across a parallel
    /// quantization pass) at no cost over a bare `Mmap`.
    mmap: Arc<Mmap>,
    descs: Vec<TensorDesc>,
    data_start: u64,
    index: HashMap<String, Entry>,
}

impl LazyGguf {
    /// The mapped GGUF's byte length (the wired-memory guard's size input).
    pub(crate) fn backing_len(&self) -> u64 {
        self.mmap.len() as u64
    }

    /// Parse the GGUF header + tensor table and build the HF-name index. Does NOT
    /// touch the data section — no dequantization happens here.
    pub fn open(path: &Path, cfg: &ModelConfig) -> Result<Self> {
        let mmap = Arc::new(mmap_file(path)?);
        let buf: &[u8] = &mmap;

        let mut r = Reader::new(buf);
        if r.take(4)? != b"GGUF" {
            return Err(ArfError::model_load("not a GGUF file (bad magic)"));
        }
        let version = r.u32()?;
        if version != 3 {
            return Err(ArfError::model_load(format!(
                "GGUF version {version} unsupported (only v3)"
            )));
        }
        let n_tensors = r.u64()? as usize;
        let n_kv = r.u64()? as usize;

        // Metadata: capture the alignment and `general.architecture`; skip the rest
        // (the rest of the config comes from `cfg`, and the tokenizer is built
        // separately via `read_tokenizer_meta` — `None` here skips the
        // tokenizer.ggml.* keys rather than parsing the vocab). The architecture
        // string selects the ggml→HF name/layout mapping (only `llama` row-permutes
        // its attention Q/K weights — see `build_index_entry`).
        let mut alignment: u64 = 32;
        let mut architecture = String::new();
        for _ in 0..n_kv {
            let key = r.string()?;
            let vty = r.u32()?;
            if key == "general.architecture" {
                architecture = r.string_after(vty)?;
            } else if !capture_meta_kv(&mut r, &key, vty, &mut alignment, None)? {
                r.skip_value(vty)?;
            }
        }

        // Tensor table.
        let mut descs = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = r.string()?;
            let n_dims = r.u32()? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(r.u64()?);
            }
            let ty = GgmlType::from_id(r.u32()?)?;
            let offset = r.u64()?;
            descs.push(TensorDesc {
                name,
                dims,
                ty,
                offset,
            });
        }

        // Data section starts after the header, padded to `alignment`.
        let data_start = (r.pos as u64).div_ceil(alignment) * alignment;

        // Build the HF-name index WITHOUT touching the data section.
        let mut index = HashMap::new();
        for (i, d) in descs.iter().enumerate() {
            build_index_entry(i, d, cfg, &architecture, &mut index)?;
        }

        Ok(LazyGguf {
            mmap,
            descs,
            data_start,
            index,
        })
    }

    /// Open a GGUF for RAW-name reads only: parses the header + tensor table but does NOT build
    /// the HF-name index, so a foreign architecture (e.g. T5's `t5encoder`, CLIP, a DiT) loads
    /// without the text indexer rejecting its tensor names. Use with `get_f32_raw` /
    /// `shape_raw` / `tensor_names`; the HF-mapped `get_f32` returns `None` here.
    pub fn open_raw(path: &Path) -> Result<Self> {
        let mmap = Arc::new(mmap_file(path)?);
        let buf: &[u8] = &mmap;
        let mut r = Reader::new(buf);
        if r.take(4)? != b"GGUF" {
            return Err(ArfError::model_load("not a GGUF file (bad magic)"));
        }
        let version = r.u32()?;
        if version != 3 {
            return Err(ArfError::model_load(format!(
                "GGUF version {version} unsupported (only v3)"
            )));
        }
        let n_tensors = r.u64()? as usize;
        let n_kv = r.u64()? as usize;
        let mut alignment: u64 = 32;
        for _ in 0..n_kv {
            let key = r.string()?;
            let vty = r.u32()?;
            if key == "general.alignment" {
                debug_assert_eq!(vty, 4);
                alignment = r.u32()? as u64;
            } else {
                r.skip_value(vty)?;
            }
        }
        let mut descs = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = r.string()?;
            let n_dims = r.u32()? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(r.u64()?);
            }
            let ty = GgmlType::from_id(r.u32()?)?;
            let offset = r.u64()?;
            descs.push(TensorDesc {
                name,
                dims,
                ty,
                offset,
            });
        }
        let data_start = (r.pos as u64).div_ceil(alignment) * alignment;
        Ok(LazyGguf {
            mmap,
            descs,
            data_start,
            index: HashMap::new(), // raw-only: HF-name index intentionally empty
        })
    }

    /// Read a metadata `array<f32>` value by key (e.g. `clip.vision.image_mean`), or
    /// `None` if the key is absent or not an f32 array. Re-scans the KV section on demand
    /// (the mmap is retained); the open path discards most KVs, and the vision
    /// preprocessing needs only these two small arrays — re-scanning is cheaper than
    /// keeping every KV resident for the whole model's lifetime.
    pub fn get_metadata_f32_array(&self, key: &str) -> Option<Vec<f32>> {
        self.with_meta_value(key, |r, vty| r.f32_array(vty))
    }

    /// Read a metadata `array<i32>` value by key (e.g. a DFlash checkpoint's
    /// `dflash.target_layers` tap indices), or `None` if the key is absent or not an
    /// i32 array. Mirrors `get_metadata_f32_array`; re-scans the KV section on demand.
    pub fn get_metadata_i32_array(&self, key: &str) -> Option<Vec<i32>> {
        self.with_meta_value(key, |r, vty| r.i32_array(vty))
    }

    /// Read a metadata `array<i64>` value by key, or `None` if the key is absent or
    /// not an i64 array. Mirrors `get_metadata_f32_array`; some converters widen
    /// small integer-array metadata (e.g. layer-index lists) to i64 rather than i32 —
    /// this covers that case without forcing every caller through i32 first.
    pub fn get_metadata_i64_array(&self, key: &str) -> Option<Vec<i64>> {
        self.with_meta_value(key, |r, vty| r.i64_array(vty))
    }

    /// Read a metadata scalar (any int value-type) by key as `u32`, or `None` if absent /
    /// non-integer / out of range. Used for `clip.vision.image_size` / `patch_size`.
    pub fn get_metadata_u32(&self, key: &str) -> Option<u32> {
        self.with_meta_value(key, |r, vty| r.scalar_int(vty))
            .and_then(|v| u32::try_from(v).ok())
    }

    /// Read a metadata scalar `f32`/`f64` value by key as `f32`, or `None` if absent /
    /// non-float. Used for `*.attention.layer_norm_rms_epsilon` (the RMS eps), which a
    /// GGUF may store as either f32 or f64. Mirrors `get_metadata_u32`.
    pub fn get_metadata_f32(&self, key: &str) -> Option<f32> {
        self.with_meta_value(key, |r, vty| r.scalar_f32(vty))
    }

    /// Read a metadata `string` value (value type 8) by key, or `None` if absent / not a
    /// string. Used for `tokenizer.chat_template` — the model's own Jinja chat template, which
    /// arf-serve renders for tool-calling requests. Mirrors `get_metadata_u32`.
    pub fn get_metadata_string(&self, key: &str) -> Option<String> {
        self.with_meta_value(key, |r, vty| r.string_after(vty))
    }

    /// Enumerate every metadata KV key in the file (re-walks the KV section). Used to
    /// ground-truth an unknown architecture's metadata-key spelling at probe time.
    pub fn metadata_keys(&self) -> Vec<String> {
        let buf: &[u8] = &self.mmap;
        let mut r = Reader::new(buf);
        let mut out = Vec::new();
        if r.take(4).map(|m| m == b"GGUF").unwrap_or(false) {
            let _version = r.u32().ok();
            let _n_tensors = r.u64().ok();
            if let Ok(n_kv) = r.u64() {
                for _ in 0..n_kv {
                    let Ok(k) = r.string() else { break };
                    let Ok(vty) = r.u32() else { break };
                    out.push(k);
                    if r.skip_value(vty).is_err() {
                        break;
                    }
                }
            }
        }
        out
    }

    /// Re-walk the metadata KV section, invoke `read` on the value when `key` matches.
    /// Returns `None` if the key is absent or `read` errors (wrong type / corrupt).
    fn with_meta_value<T>(
        &self,
        key: &str,
        read: impl FnOnce(&mut Reader<'_>, u32) -> Result<T>,
    ) -> Option<T> {
        let buf: &[u8] = &self.mmap;
        let mut r = Reader::new(buf);
        // header: magic(4) + version(u32) + n_tensors(u64) + n_kv(u64)
        if r.take(4).ok()? != b"GGUF" {
            return None;
        }
        let _version = r.u32().ok()?;
        let _n_tensors = r.u64().ok()?;
        let n_kv = r.u64().ok()? as usize;
        for _ in 0..n_kv {
            let k = r.string().ok()?;
            let vty = r.u32().ok()?;
            if k == key {
                return read(&mut r, vty).ok();
            }
            r.skip_value(vty).ok()?;
        }
        None
    }

    /// Dequantize one tensor (or one expert sub-slice) to f32; `None` if absent.
    pub fn get_f32(&self, name: &str) -> Option<Vec<f32>> {
        match self.index.get(name)? {
            Entry::Whole {
                desc_idx,
                rows,
                cols,
                vec_len,
                permute_head_dim,
            } => {
                let d = &self.descs[*desc_idx];
                let n = vec_len.unwrap_or(rows * cols);
                let base = (self.data_start + d.offset) as usize;
                let data = dequantize(d.ty, &self.mmap[base..], n).ok()?;
                Some(match permute_head_dim {
                    Some(hd) => unpermute_qk_rows(&data, *rows, *cols, *hd),
                    None => data,
                })
            }
            Entry::Expert {
                desc_idx,
                expert,
                per_elems,
                ..
            } => {
                let d = &self.descs[*desc_idx];
                // Expert E begins exactly on a block boundary because `per_elems`
                // is a multiple of QK_K (asserted in `index_experts`), so no block
                // straddles an expert boundary: dequantizing the sub-slice is
                // bit-identical to dequantizing the whole tensor and slicing.
                let base = (self.data_start + d.offset) as usize
                    + expert * ggml_byte_len(d.ty, *per_elems);
                dequantize(d.ty, &self.mmap[base..], *per_elems).ok()
            }
            Entry::RowSlice {
                desc_idx,
                row0,
                rows,
                cols,
            } => {
                // Same primitive as Expert: a block-aligned sub-slice. The offset is
                // `row0 * cols` elements because the slice runs along ROWS while blocks run
                // along COLS, and cols is a multiple of QK_K — so the boundary is free.
                let d = &self.descs[*desc_idx];
                let base = (self.data_start + d.offset) as usize + ggml_byte_len(d.ty, row0 * cols);
                dequantize(d.ty, &self.mmap[base..], rows * cols).ok()
            }
        }
    }

    /// Dequantize a tensor to f32 by its RAW ggml name (the name as stored in the file,
    /// e.g. `enc.blk.0.attn_q.weight`), bypassing the HF-name index. This is the primitive a
    /// self-contained module (T5/CLIP/DiT/VAE) uses to read its own weights without teaching
    /// the gemma/llama indexer about a foreign architecture — keeping each model's tensor
    /// knowledge local to its module. Returns `None` if the name is absent.
    pub fn get_f32_raw(&self, ggml_name: &str) -> Option<Vec<f32>> {
        let d = self.descs.iter().find(|d| d.name == ggml_name)?;
        let n: usize = d.dims.iter().product::<u64>() as usize;
        let base = (self.data_start + d.offset) as usize;
        dequantize(d.ty, &self.mmap[base..], n).ok()
    }

    /// The ggml dims (fastest-varying first, e.g. `[in, out]` for a 2D weight) of a tensor by
    /// its raw ggml name, or `None` if absent. Companion to `get_f32_raw`.
    pub fn shape_raw(&self, ggml_name: &str) -> Option<Vec<usize>> {
        let d = self.descs.iter().find(|d| d.name == ggml_name)?;
        Some(d.dims.iter().map(|&x| x as usize).collect())
    }

    /// The raw ggml type id of a tensor by its raw ggml name (e.g. 12 = Q4_K, 13 = Q5_K,
    /// 14 = Q6_K, 30 = BF16, 0 = F32), or `None` if absent. Lets a self-contained module
    /// branch its residency/dispatch on the actual stored quant without re-reading bytes.
    /// The raw ggml type id a tensor is STORED as, looked up by the HF name
    /// the engine uses (not the raw GGUF name).
    ///
    /// A consumer needs this to avoid re-quantising DOWN. A mixed-quant file
    /// keeps its sensitive tensors at 5-8 bits - Qwen3.8-27B-UD-Q4_K_M is 131
    /// Q5_K, 117 IQ4_XS, 106 Q8_0 against 104 Q4_K - and treating them all as
    /// Q4_K measured rel 1.1e-1 on a projection where the stored precision
    /// gives 1.5e-6.
    pub fn stored_type_by_hf_name(&self, name: &str) -> Option<u32> {
        let desc_idx = match self.index.get(name)? {
            Entry::Whole { desc_idx, .. } => *desc_idx,
            Entry::RowSlice { desc_idx, .. } => *desc_idx,
            _ => return None,
        };
        let raw = self.descs.get(desc_idx)?.name.clone();
        self.raw_ggml_type_id(&raw)
    }

    pub fn raw_ggml_type_id(&self, ggml_name: &str) -> Option<u32> {
        let d = self.descs.iter().find(|d| d.name == ggml_name)?;
        Some(match d.ty {
            GgmlType::F32 => 0,
            GgmlType::F16 => 1,
            GgmlType::Q40 => 2,
            GgmlType::Q41 => 3,
            GgmlType::Q80 => 8,
            GgmlType::Q4K => 12,
            GgmlType::Q5K => 13,
            GgmlType::Q6K => 14,
            GgmlType::Q3K => 11,
            GgmlType::Iq3S => 21,
            GgmlType::Iq4Nl => 20,
            GgmlType::Iq4Xs => 23,
            GgmlType::BF16 => 30,
        })
    }

    /// Raw native **Q4_K** block bytes for a tensor by its RAW ggml name (the raw-name
    /// analog of `get_q4k_blocks`, bypassing the HF-name index — the primitive a
    /// self-contained module like the FLUX DiT uses to transcode its Q4_K_S weights
    /// straight to GPU residency without ever materialising f32). Returns
    /// `Some((blocks, rows, cols))` only when the tensor is present and stored as native
    /// Q4_K (ggml type 12; "Q4_K_S" is a mix name — its per-tensor block format is Q4_K).
    /// `None` otherwise → the caller falls back to the `get_f32_raw` dequant path. FLUX
    /// tensors carry no Q/K head-permute, so there is no permute caveat here. Feed the
    /// bytes to [`Q4KSMatrix::from_ggml_q4k`](crate::tensor::Q4KSMatrix::from_ggml_q4k).
    pub fn get_q4k_blocks_raw(&self, ggml_name: &str) -> Option<(&[u8], usize, usize)> {
        let d = self.descs.iter().find(|d| d.name == ggml_name)?;
        if d.ty != GgmlType::Q4K {
            return None;
        }
        let n: usize = d.dims.iter().product::<u64>() as usize;
        // ggml dims are fastest-varying first → [cols, rows] for a 2-D weight.
        let cols = d.dims.first().copied().unwrap_or(1) as usize;
        let rows = if d.dims.len() >= 2 { n / cols } else { 1 };
        let base = (self.data_start + d.offset) as usize;
        let byte_len = ggml_byte_len(GgmlType::Q4K, n);
        Some((&self.mmap[base..base + byte_len], rows, cols))
    }

    /// Raw native **IQ4_XS** block bytes by RAW ggml name — the IQ4_XS analog of
    /// `get_q4k_blocks_raw`. 136 bytes per 256 weights.
    ///
    /// WHY IT EXISTS: IQ4_XS is the largest tensor group in Qwen3.8-27B (117
    /// tensors, 8.96e9 weights) and without a native reader every one of them is
    /// held at 8.5 bpw instead of 4.25 — **4.76 GB** wasted, which is precisely
    /// what keeps 32K context out of reach (measured, the GGUF
    /// census and the measured 24K tq4 ceiling).
    pub fn get_iq4_xs_blocks_raw(&self, ggml_name: &str) -> Option<(&[u8], usize, usize)> {
        let d = self.descs.iter().find(|d| d.name == ggml_name)?;
        if d.ty != GgmlType::Iq4Xs {
            return None;
        }
        let n: usize = d.dims.iter().product::<u64>() as usize;
        let cols = d.dims.first().copied().unwrap_or(1) as usize;
        let rows = if d.dims.len() >= 2 { n / cols } else { 1 };
        let base = (self.data_start + d.offset) as usize;
        let byte_len = ggml_byte_len(GgmlType::Iq4Xs, n);
        Some((&self.mmap[base..base + byte_len], rows, cols))
    }

    /// Raw native **Q8_0** block bytes for a tensor by its RAW ggml name (the Q8_0 analog
    /// of `get_q4k_blocks_raw`). Returns `Some((blocks, rows, cols))` only when the tensor
    /// is present and stored as native Q8_0 (ggml type 8: 34 bytes/32 weights — f16 `d` then
    /// 32 i8 codes, `w = d·q`). `None` otherwise → the caller falls back to `get_f32_raw`.
    /// The primitive a self-contained module (FLUX T5-XXL) uses to keep its Q8_0 weights
    /// quantized-resident without ever materialising f32. Feed to a Q8_0-block GEMM kernel /
    /// [`matmul_nt_q8_0`](crate::tensor::matmul_nt_q8_0).
    pub fn get_q8_0_blocks_raw(&self, ggml_name: &str) -> Option<(&[u8], usize, usize)> {
        let d = self.descs.iter().find(|d| d.name == ggml_name)?;
        if d.ty != GgmlType::Q80 {
            return None;
        }
        let n: usize = d.dims.iter().product::<u64>() as usize;
        let cols = d.dims.first().copied().unwrap_or(1) as usize;
        let rows = if d.dims.len() >= 2 { n / cols } else { 1 };
        let base = (self.data_start + d.offset) as usize;
        let byte_len = ggml_byte_len(GgmlType::Q80, n);
        Some((&self.mmap[base..base + byte_len], rows, cols))
    }

    /// Iterate every tensor's raw ggml name (for discovering an unknown model's layout).
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.descs.iter().map(|d| d.name.as_str())
    }

    /// The [`Quant`] whose native lossless transcode path matches how this GGUF stores
    /// its bulk weights — so a GGUF loads via its OWN block format (≈ on-disk footprint,
    /// the way ggml/ollama read it) instead of being dequant-expanded to bf16. Picked from
    /// the MAJORITY ggml type across the `blk.*` (layer projection / FFN) tensors — the
    /// embedding/output tensors are deliberately ignored because even a "Q4_0"/"Q4_K_S"
    /// GGUF stores those in a higher-precision type (Q5_K/Q6_K), which would skew the vote.
    ///
    /// `None` when the dominant type has no native quant path (F32/F16/BF16 full-precision
    /// GGUFs, or a mixed type we don't transcode) — the caller then keeps its bf16 default.
    pub fn dominant_quant(&self) -> Option<Quant> {
        let mut q40 = 0usize;
        let mut q4k = 0usize;
        let mut other = 0usize;
        for d in &self.descs {
            // Vote only on the repeating per-layer weight matrices; skip embeddings,
            // norms, and output head whose types are not representative of the bulk.
            if !d.name.starts_with("blk.") {
                continue;
            }
            // 2-D matmul weights only (norms are 1-D vectors stored f32).
            if d.dims.len() < 2 {
                continue;
            }
            match d.ty {
                GgmlType::Q40 | GgmlType::Q41 => q40 += 1,
                GgmlType::Q4K => q4k += 1,
                _ => other += 1,
            }
        }
        // Require a clear majority of the layer weights to be in the chosen format; a
        // GGUF whose layers are mostly f32/f16/Q6_K has no lossless 4-bit path here.
        let total = q40 + q4k + other;
        if total == 0 {
            return None;
        }
        if q40 * 2 > total {
            Some(Quant::Q4)
        } else if q4k * 2 > total {
            Some(Quant::Q4KS)
        } else {
            None
        }
    }

    /// Return the raw ggml Q4_K block bytes (no dequantization) for a tensor, along
    /// with its `(rows, cols)`. Returns `None` if the tensor is absent, is not Q4_K,
    /// or needs a row-permute (Q/K) — the latter can only be applied in f32 space, so
    /// such tensors must take the dequant→re-quant path instead of the raw transcode.
    ///
    /// The caller may pass these bytes directly to
    /// [`Q4KSMatrix::from_ggml_q4k`](crate::tensor::Q4KSMatrix::from_ggml_q4k)
    /// to transcode without ever materialising f32 weights.
    /// Name-mapped IQ4_XS blocks — the IQ4_XS analog of `get_q4k_blocks`.
    ///
    /// Deliberately handles only the `Whole` entry shape. An IQ4_XS tensor that
    /// is an expert slice or a row slice returns None and the caller falls back,
    /// rather than this guessing at an offset arithmetic that has not been
    /// verified for this format. On Qwen3.8-27B every IQ4_XS tensor is Whole.
    pub fn get_iq4_xs_blocks(&self, name: &str) -> Option<(&[u8], usize, usize)> {
        match self.index.get(name)? {
            Entry::Whole {
                desc_idx,
                rows,
                cols,
                vec_len,
                permute_head_dim,
            } => {
                let d = &self.descs[*desc_idx];
                // permute_head_dim would require re-ordering ROWS, which cannot
                // be done on packed 256-weight super-blocks without unpacking.
                if d.ty != GgmlType::Iq4Xs || permute_head_dim.is_some() {
                    return None;
                }
                let n = vec_len.unwrap_or(rows * cols);
                let base = (self.data_start + d.offset) as usize;
                let byte_len = ggml_byte_len(GgmlType::Iq4Xs, n);
                Some((&self.mmap[base..base + byte_len], *rows, *cols))
            }
            _ => None,
        }
    }

    pub fn get_q4k_blocks(&self, name: &str) -> Option<(&[u8], usize, usize)> {
        match self.index.get(name)? {
            Entry::Whole {
                desc_idx,
                rows,
                cols,
                vec_len,
                permute_head_dim,
            } => {
                let d = &self.descs[*desc_idx];
                if d.ty != GgmlType::Q4K || permute_head_dim.is_some() {
                    return None;
                }
                let n = vec_len.unwrap_or(rows * cols);
                let base = (self.data_start + d.offset) as usize;
                let byte_len = ggml_byte_len(GgmlType::Q4K, n);
                Some((&self.mmap[base..base + byte_len], *rows, *cols))
            }
            Entry::Expert {
                desc_idx,
                expert,
                per_elems,
                rows,
                cols,
            } => {
                let d = &self.descs[*desc_idx];
                if d.ty != GgmlType::Q4K {
                    return None;
                }
                let byte_len = ggml_byte_len(GgmlType::Q4K, *per_elems);
                let base = (self.data_start + d.offset) as usize + expert * byte_len;
                Some((&self.mmap[base..base + byte_len], *rows, *cols))
            }
            Entry::RowSlice {
                desc_idx,
                row0,
                rows,
                cols,
            } => {
                // NATIVE Q4_K sub-slice — lossless, no dequant. Block-aligned because the cut
                // is along ROWS and cols is a multiple of QK_K.
                let d = &self.descs[*desc_idx];
                if d.ty != GgmlType::Q4K {
                    return None;
                }
                let base = (self.data_start + d.offset) as usize
                    + ggml_byte_len(GgmlType::Q4K, row0 * cols);
                let byte_len = ggml_byte_len(GgmlType::Q4K, rows * cols);
                Some((&self.mmap[base..base + byte_len], *rows, *cols))
            }
        }
    }

    /// Raw native **Q4_1** block bytes for `name` (2026-09-25): the carrier for group-64 affine
    /// weights (a group of 64 with one scale + bias is EXACTLY two Q4_1 blocks sharing d and m —
    /// see `scripts/g64/mlx_to_q4_1_gguf.py`). Whole, un-permuted tensors only.
    pub fn get_q4_1_blocks(&self, name: &str) -> Option<(&[u8], usize, usize)> {
        match self.index.get(name)? {
            Entry::Whole {
                desc_idx,
                rows,
                cols,
                vec_len,
                permute_head_dim,
            } => {
                let d = &self.descs[*desc_idx];
                if d.ty != GgmlType::Q41 || permute_head_dim.is_some() {
                    return None;
                }
                let n = vec_len.unwrap_or(rows * cols);
                let base = (self.data_start + d.offset) as usize;
                let byte_len = ggml_byte_len(GgmlType::Q41, n);
                Some((&self.mmap[base..base + byte_len], *rows, *cols))
            }
            // the GDN layers' q/k/v row ranges of `attn_qkv` — block-aligned for the same reason
            // as the Q4_K arm (the cut is along rows); without this they took the bf16 fallback
            Entry::RowSlice {
                desc_idx,
                row0,
                rows,
                cols,
            } => {
                let d = &self.descs[*desc_idx];
                if d.ty != GgmlType::Q41 {
                    return None;
                }
                let base = (self.data_start + d.offset) as usize
                    + ggml_byte_len(GgmlType::Q41, row0 * cols);
                let byte_len = ggml_byte_len(GgmlType::Q41, rows * cols);
                Some((&self.mmap[base..base + byte_len], *rows, *cols))
            }
            _ => None,
        }
    }

    /// Raw native **Q4_0** block bytes for `name` (the Q4_0 analog of
    /// `get_q4k_blocks`), for the lossless `Q4Matrix::from_ggml_q4_0` transcode.
    /// `None` unless the tensor is stored as native Q4_0 with no head-permute (permuted
    /// tensors must take the dequant→re-quant path).
    pub fn get_q4_0_blocks(&self, name: &str) -> Option<(&[u8], usize, usize)> {
        match self.index.get(name)? {
            Entry::Whole {
                desc_idx,
                rows,
                cols,
                vec_len,
                permute_head_dim,
            } => {
                let d = &self.descs[*desc_idx];
                if d.ty != GgmlType::Q40 || permute_head_dim.is_some() {
                    return None;
                }
                let n = vec_len.unwrap_or(rows * cols);
                let base = (self.data_start + d.offset) as usize;
                let byte_len = ggml_byte_len(GgmlType::Q40, n);
                Some((&self.mmap[base..base + byte_len], *rows, *cols))
            }
            Entry::Expert {
                desc_idx,
                expert,
                per_elems,
                rows,
                cols,
            } => {
                let d = &self.descs[*desc_idx];
                if d.ty != GgmlType::Q40 {
                    return None;
                }
                let byte_len = ggml_byte_len(GgmlType::Q40, *per_elems);
                let base = (self.data_start + d.offset) as usize + expert * byte_len;
                Some((&self.mmap[base..base + byte_len], *rows, *cols))
            }
            Entry::RowSlice {
                desc_idx,
                row0,
                rows,
                cols,
            } => {
                // Q4_0 native slice. qwen35 ships Q4_K so this arm is unused today, but the
                // primitive is identical and leaving it unimplemented would be a landmine.
                let d = &self.descs[*desc_idx];
                if d.ty != GgmlType::Q40 {
                    return None;
                }
                let base = (self.data_start + d.offset) as usize
                    + ggml_byte_len(GgmlType::Q40, row0 * cols);
                let byte_len = ggml_byte_len(GgmlType::Q40, rows * cols);
                Some((&self.mmap[base..base + byte_len], *rows, *cols))
            }
        }
    }

    /// Reconstructed shape for a name (`[len]` for a vector, `[rows, cols]` for a
    /// matrix), for the shape-validating accessor path; `None` if absent.
    pub fn shape(&self, name: &str) -> Option<Vec<usize>> {
        match self.index.get(name)? {
            Entry::Whole {
                rows,
                cols,
                vec_len,
                ..
            } => Some(vec_len.map_or_else(|| vec![*rows, *cols], |n| vec![n])),
            Entry::Expert { rows, cols, .. } => Some(vec![*rows, *cols]),
            Entry::RowSlice { rows, cols, .. } => Some(vec![*rows, *cols]),
        }
    }
}

/// Read JUST the embedded tokenizer metadata from a GGUF header — no tensor table,
/// no `ModelConfig` needed. Used to build a tokenizer without loading the model.
pub fn read_tokenizer_meta(path: &Path) -> Result<GgufTokenizerMeta> {
    let mmap = mmap_file(path)?;
    let buf: &[u8] = &mmap;
    let mut r = Reader::new(buf);
    if r.take(4)? != b"GGUF" {
        return Err(ArfError::model_load("not a GGUF file (bad magic)"));
    }
    let version = r.u32()?;
    if version != 3 {
        return Err(ArfError::model_load(format!(
            "GGUF version {version} unsupported (only v3)"
        )));
    }
    let _n_tensors = r.u64()?;
    let n_kv = r.u64()? as usize;
    // `capture_meta_kv` writes alignment, but a tokenizer-only read doesn't need it.
    let mut _alignment: u64 = 32;
    let mut tok_meta = GgufTokenizerMeta::default();
    for _ in 0..n_kv {
        let key = r.string()?;
        let vty = r.u32()?;
        if !capture_meta_kv(&mut r, &key, vty, &mut _alignment, Some(&mut tok_meta))? {
            r.skip_value(vty)?;
        }
    }
    Ok(tok_meta)
}

/// True if `path` is an existing file whose first 4 bytes are the GGUF magic.
pub fn is_gguf_file(path: &Path) -> bool {
    use std::io::Read;
    if !path.is_file() {
        return false;
    }
    let mut magic = [0u8; 4];
    std::fs::File::open(path)
        .and_then(|mut f| f.read_exact(&mut magic))
        .map(|_| &magic == b"GGUF")
        .unwrap_or(false)
}

/// Index a SigLIP vision-tower (`v.*`) or multimodal-projector (`mm.*`) tensor under a
/// stable `vision.*` key. The names pass through nearly 1:1 (we just prefix `vision.`),
/// preserving the layer index. 2D weights → matrix entry `[out,in]=[ne1,ne0]`; 1D
/// (biases, norms, conv-bias) → vector entry. The 4D patch-embed conv weight
/// `[14,14,3,1152]` is stored as a flat vector (the encoder reshapes it).
fn index_vision_tensor(i: usize, d: &TensorDesc, index: &mut HashMap<String, Entry>) -> Result<()> {
    let key = format!("vision.{}", d.name); // e.g. vision.v.blk.0.attn_q.weight
    let dims = &d.dims;
    if dims.len() == 2 {
        // ggml [ne0=in, ne1=out] → our [out, in].
        index.insert(
            key,
            Entry::Whole {
                desc_idx: i,
                rows: dims[1] as usize,
                cols: dims[0] as usize,
                vec_len: None,
                permute_head_dim: None,
            },
        );
    } else {
        // 1D (bias/norm) or 4D (patch-embed conv) → a flat vector of all elements.
        let n: usize = dims.iter().product::<u64>() as usize;
        index.insert(
            key,
            Entry::Whole {
                desc_idx: i,
                rows: 0,
                cols: 0,
                vec_len: Some(n),
                permute_head_dim: None,
            },
        );
    }
    Ok(())
}

/// Insert the index entry (or entries, for packed experts) for one ggml tensor,
/// applying the same ggml→HF name mapping the eager loader used. NOTE: a 2D weight
/// is *labeled* `[out, in] = [ne1, ne0]` but its f32 data is NOT permuted — the
/// eager path stored the raw dequantized stream under that label, and so must this
/// one. Implementing a transpose here would be a bug.
fn build_index_entry(
    i: usize,
    d: &TensorDesc,
    cfg: &ModelConfig,
    architecture: &str,
    index: &mut HashMap<String, Entry>,
) -> Result<()> {
    let name = &d.name;
    // Multimodal GGUFs / mmproj files (e.g. gemma-3 4B) ship a SigLIP vision tower (`v.*`)
    // and the multimodal projector (`mm.*`). Route them into the index under stable
    // `vision.*` names so the vision encoder can fetch them; the text path simply never
    // asks for these keys. (Was: skipped — text-only.)
    if name.starts_with("v.") || name.starts_with("mm.") {
        return index_vision_tensor(i, d, index);
    }
    // L155 — key the four-norm NAMING on placement, not on Gemma. Any four-norm model puts its
    // pre-FFN norm in ggml's `ffn_norm` slot and its around-sublayer norms in
    // `post_attention_norm` / `post_ffw_norm`. Keying on `is_gemma` meant a non-Gemma four-norm
    // model (muse-glimmer) mapped BOTH `ffn_norm` and `post_attention_norm` onto
    // `post_attention_layernorm` — a silent overwrite, and exactly the kind of bug that shows up
    // as garbage tokens rather than an error.
    let four_norm = cfg.has_post_norms();
    // ggml dims are fastest-varying first: [ne0=in, ne1=out] → our [out, in] label.
    let mat_shape = |dims: &[u64]| (dims[1] as usize, dims[0] as usize);
    let whole_mat = |index: &mut HashMap<String, Entry>, hf: String, dims: &[u64]| {
        let (rows, cols) = mat_shape(dims);
        index.insert(
            hf,
            Entry::Whole {
                desc_idx: i,
                rows,
                cols,
                vec_len: None,
                permute_head_dim: None,
            },
        );
    };
    // Attention Q/K matrices. llama.cpp's `LlamaModel` HF→GGUF converter row-permutes
    // these from HF's split-half RoPE layout into ggml's interleaved layout; Arf's
    // RoPE uses the split-half layout, so for the `llama` architecture we flag them to
    // be un-permuted on dequant. Other architectures (gemma/qwen converters) store Q/K
    // un-permuted, so there we treat them as a plain matrix.
    let qk_permutes = architecture == "llama";
    let whole_mat_qk = |index: &mut HashMap<String, Entry>, hf: String, dims: &[u64]| {
        let (rows, cols) = mat_shape(dims);
        index.insert(
            hf,
            Entry::Whole {
                desc_idx: i,
                rows,
                cols,
                vec_len: None,
                permute_head_dim: qk_permutes.then_some(cfg.head_dim),
            },
        );
    };
    let whole_vec = |index: &mut HashMap<String, Entry>, hf: String, dims: &[u64]| {
        let n: usize = dims.iter().product::<u64>() as usize;
        index.insert(
            hf,
            Entry::Whole {
                desc_idx: i,
                rows: 0,
                cols: 0,
                vec_len: Some(n),
                permute_head_dim: None,
            },
        );
    };

    // Layer-indexed names: blk.N.*
    if let Some(rest) = name.strip_prefix("blk.") {
        let (idx, field) = rest
            .split_once('.')
            .ok_or_else(|| ArfError::model_load(format!("bad ggml name {name:?}")))?;
        let p = format!("model.layers.{idx}");
        match field {
            "attn_norm.weight" => whole_vec(index, format!("{p}.input_layernorm.weight"), &d.dims),
            // ggml `ffn_norm` is the pre-MLP norm. For Llama/Qwen that role IS
            // `post_attention_layernorm`; Gemma instead names it
            // `pre_feedforward_layernorm` (and uses `post_attention_norm` /
            // `post_ffw_norm` for the two extra around-sublayer norms below).
            "ffn_norm.weight" if four_norm => whole_vec(
                index,
                format!("{p}.pre_feedforward_layernorm.weight"),
                &d.dims,
            ),
            "ffn_norm.weight" => whole_vec(
                index,
                format!("{p}.post_attention_layernorm.weight"),
                &d.dims,
            ),
            // Gemma's two extra norms (around attention output / MLP output).
            "post_attention_norm.weight" => whole_vec(
                index,
                format!("{p}.post_attention_layernorm.weight"),
                &d.dims,
            ),
            "post_ffw_norm.weight" => whole_vec(
                index,
                format!("{p}.post_feedforward_layernorm.weight"),
                &d.dims,
            ),
            // Dense FFN (Llama/Gemma); the MoE `ffn_*_exps` variants are below.
            "ffn_gate.weight" => whole_mat(index, format!("{p}.mlp.gate_proj.weight"), &d.dims),
            "ffn_up.weight" => whole_mat(index, format!("{p}.mlp.up_proj.weight"), &d.dims),
            "ffn_down.weight" => whole_mat(index, format!("{p}.mlp.down_proj.weight"), &d.dims),
            // FUSED QKV (qwen35 family: Qwen3.5/3.6/3.8-27B). One `[q_dim + 2*kv_dim, hidden]`
            // tensor where the loader wants three matrices. Split by ROW RANGE — free of
            // re-quantization because blocks run along cols and cols is a multiple of QK_K.
            //
            // Row counts come from the CONFIG, not the tensor, and are ASSERTED against it so a
            // shape mismatch fails loudly rather than silently mis-slicing. For the hybrid arch
            // the split is the SSM input projection (ssm_inner + 2*groups*state), NOT attention
            // q/k/v — see the correction note in the arm below.
            "attn_qkv.weight" => {
                // CORRECTED (and the correction matters): this is the GDN layers' SSM INPUT
                // projection, not attention q/k/v. Measured from Qwen3.8-27B:
                //     dims=[5120, 10240] -> rows=10240, cols=5120(hidden)
                //     10240 = ssm_inner(6144) + 2 * groups(16)*state(128)=2048
                // i.e. [q 6144 | k 2048 | v 2048].
                //
                // My first pass asserted rows == nh*hd + 2*nkv*hd = 8192 and the split SILENTLY
                // NEVER FIRED — the arithmetic looked plausible because 8192 is also a sum of
                // this model's dims, and GGUF stores dims fastest-first so I had rows/cols
                // transposed. Hence the assert-and-skip is now an assert-and-FAIL below.
                let (rows, cols) = mat_shape(&d.dims);
                let (q_rows, kv_rows) = match &cfg.attn {
                    crate::config::AttnKind::HybridSsmAttn {
                        ssm_inner,
                        ssm_state,
                        groups,
                        ..
                    } => (*ssm_inner, groups * ssm_state),
                    _ => (
                        cfg.num_attention_heads * cfg.head_dim,
                        cfg.num_kv_heads * cfg.head_dim,
                    ),
                };
                if rows != q_rows + 2 * kv_rows {
                    return Err(ArfError::model_load(format!(
                        "attn_qkv rows {rows} != q {q_rows} + 2*kv {kv_rows} — fused-QKV layout \
                         does not match the config; refusing to mis-slice"
                    )));
                }
                {
                    for (hf, row0, n) in [
                        (format!("{p}.self_attn.q_proj.weight"), 0, q_rows),
                        (format!("{p}.self_attn.k_proj.weight"), q_rows, kv_rows),
                        (
                            format!("{p}.self_attn.v_proj.weight"),
                            q_rows + kv_rows,
                            kv_rows,
                        ),
                    ] {
                        index.insert(
                            hf,
                            Entry::RowSlice {
                                desc_idx: i,
                                row0,
                                rows: n,
                                cols,
                            },
                        );
                    }
                    // ALSO expose the tensor WHOLE, under its own name. The gated-delta-net
                    // forward needs the packed projection as ONE matrix.
                    //
                    // RETRACTED: this comment used to say the GGUF packs [6144 | 2048 | 2048]
                    // and that "the slice NAMED q_proj is the 6144 block, which is the SSM's
                    // v". Wrong. The row order is [q 2048 | k 2048 | v 6144] — llama.cpp
                    // qwen35.cpp:75-91 views q at 0, k at key_dim, v at 2*key_dim, and the
                    // Metal offsets this very comment cited say the same. The `rows ==
                    // q_rows + 2*kv_rows` assert above is satisfied by BOTH orders (6144+
                    // 2048+2048 == 2048+2048+6144), so it never could tell them apart; the
                    // "[6144 first]" came from q_rows = ssm_inner, a NAMING choice, not a
                    // measurement of the file. That prose then licensed a backend's tile to
                    // read [v|q|k], which scrambled every recurrent layer with finite,
                    // plausible values while every parity check stayed green.
                    //
                    // Consequence for the three slices above: they are MISLABELLED (the one
                    // named q_proj is rows 0..6144 = [q|k|v[0..2048]]) but harmless as long
                    // as a consumer reassembles them in row order, which a backend's loader
                    // does. A consumer that uses q_proj AS q (gdn_layer_parity's front
                    // half, ARF_GDN_F32_QKV) is comparing the wrong thing under the
                    // right name — pending cleanup, deliberately not bundled with the fix.
                    // Reading this WHOLE entry and tiling [q|k|v] is what Metal does and is
                    // the right shape.
                    index.insert(
                        format!("{p}.linear_attn.qkv_proj.weight"),
                        Entry::Whole {
                            desc_idx: i,
                            rows,
                            cols,
                            vec_len: None,
                            permute_head_dim: None,
                        },
                    );
                }
            }
            "attn_q.weight" => whole_mat_qk(index, format!("{p}.self_attn.q_proj.weight"), &d.dims),
            "attn_k.weight" => whole_mat_qk(index, format!("{p}.self_attn.k_proj.weight"), &d.dims),
            "attn_v.weight" => whole_mat(index, format!("{p}.self_attn.v_proj.weight"), &d.dims),
            "attn_output.weight" => {
                whole_mat(index, format!("{p}.self_attn.o_proj.weight"), &d.dims)
            }
            "attn_q_norm.weight" => {
                whole_vec(index, format!("{p}.self_attn.q_norm.weight"), &d.dims)
            }
            "attn_k_norm.weight" => {
                whole_vec(index, format!("{p}.self_attn.k_norm.weight"), &d.dims)
            }
            // Gemma 4: a learned per-layer scalar ([1]) multiplying the WHOLE layer
            // output at the very end (`hidden_states *= layer_scalar`, after both
            // sublayers + residuals + post-norms — verified vs HF modeling_gemma4).
            "layer_output_scale.weight" => whole_vec(index, format!("{p}.layer_scalar"), &d.dims),
            // Muse Glimmer: a sigmoid gate on the ATTENTION output, computed from the
            // pre-attention hidden state and applied before o_proj
            // (`attn_out *= sigmoid(attn_gate @ attn_input)`). No other arch we load has this.
            "attn_gate.weight" => {
                whole_mat(index, format!("{p}.self_attn.gate_proj.weight"), &d.dims)
            }
            // ── GATED-DELTA-NET (qwen35 family: Qwen3.5/3.6/3.8-27B) ────────────────────────
            // These layers replace attention on 3 of every 4 blocks (full_attention_interval=4).
            // Shapes read from the Qwen3.8-27B GGUF, with the config values they correspond to:
            //   ssm_a           [48]          time_step_rank — per-head decay log
            //   ssm_dt.bias     [48]          time_step_rank — delta bias
            //   ssm_alpha       [5120, 48]    hidden x rank  — delta projection
            //   ssm_beta        [5120, 48]    hidden x rank  — beta projection
            //   ssm_conv1d      [4, 10240]    conv_kernel x (2*inner) — causal depthwise conv
            //   ssm_norm        [128]         state_size     — per-state RMSNorm
            //   ssm_out         [6144, 5120]  inner x hidden — output projection
            // Named under `.linear_attn.` on the HF side, which is what the GDN kernels
            // (gdn_conv1d / gdn_l2_norm / gdn_g_decay / gdn_tile / gdn_recurrence) will look up.
            "ssm_a" => whole_vec(index, format!("{p}.linear_attn.A_log"), &d.dims),
            "ssm_dt.bias" => whole_vec(index, format!("{p}.linear_attn.dt_bias"), &d.dims),
            "ssm_norm.weight" => whole_vec(index, format!("{p}.linear_attn.norm.weight"), &d.dims),
            "ssm_alpha.weight" => {
                whole_mat(index, format!("{p}.linear_attn.alpha_proj.weight"), &d.dims)
            }
            "ssm_beta.weight" => {
                whole_mat(index, format!("{p}.linear_attn.beta_proj.weight"), &d.dims)
            }
            "ssm_conv1d.weight" => {
                whole_mat(index, format!("{p}.linear_attn.conv1d.weight"), &d.dims)
            }
            "ssm_out.weight" => {
                whole_mat(index, format!("{p}.linear_attn.out_proj.weight"), &d.dims)
            }
            "ffn_gate_inp.weight" => whole_mat(index, format!("{p}.mlp.gate.weight"), &d.dims),
            "ffn_gate_exps.weight" => index_experts(i, d, &p, "gate_proj", index)?,
            "ffn_up_exps.weight" => index_experts(i, d, &p, "up_proj", index)?,
            "ffn_down_exps.weight" => index_experts(i, d, &p, "down_proj", index)?,
            // L240 — MULTI-TOKEN-PREDICTION HEAD (qwen35: nextn_predict_layers=1). A trained
            // speculative-decoding head that sits OUTSIDE the forward pass: the model is correct
            // without it. It used to be SKIPPED, on the reasoning that Arf drives speculation
            // from its own drafter — but L238 measured that drafter proposing on only 12% of
            // steps, which dilutes a real 1.16x per-window win to ~1.02x overall. This head
            // proposes on EVERY step by construction, so it is now MAPPED.
            //
            // The rest of blk.64 (attn_q/k/v/output, both norms, ffn_gate/up/down) needs no
            // special case — it is named exactly like a trunk block and already maps above.
            "nextn.eh_proj.weight" => {
                whole_mat(index, format!("{p}.nextn.eh_proj.weight"), &d.dims)
            }
            "nextn.enorm.weight" => whole_vec(index, format!("{p}.nextn.enorm.weight"), &d.dims),
            "nextn.hnorm.weight" => whole_vec(index, format!("{p}.nextn.hnorm.weight"), &d.dims),
            "nextn.shared_head_norm.weight" => {
                whole_vec(index, format!("{p}.nextn.shared_head_norm.weight"), &d.dims)
            }
            // Any other nextn.* (e.g. embed_tokens on other architectures) stays skipped rather
            // than failing the load — the head is optional by design.
            n if n.starts_with("nextn") => {}
            other => {
                return Err(ArfError::model_load(format!(
                    "unmapped ggml layer tensor {other:?}"
                )))
            }
        }
        return Ok(());
    }

    // Top-level tensors.
    match name.as_str() {
        "token_embd.weight" => whole_mat(index, "model.embed_tokens.weight".into(), &d.dims),
        "output_norm.weight" => whole_vec(index, "model.norm.weight".into(), &d.dims),
        // Tied models reuse the embedding as the lm_head; the GGUF's `output.weight`
        // (when present) is ignored in that case, matching the eager loader.
        "output.weight" => {
            if !cfg.tie_word_embeddings {
                whole_mat(index, "lm_head.weight".into(), &d.dims);
            }
        }
        // llama.cpp ships precomputed RoPE inverse-frequencies as a top-level tensor;
        // Arf computes RoPE itself, so this is not a model weight we consume.
        "rope_freqs.weight" => {}
        other => {
            return Err(ArfError::model_load(format!(
                "unmapped ggml tensor {other:?}"
            )))
        }
    }
    Ok(())
}

/// Insert one [`Entry::Expert`] per expert for a packed 3D expert tensor. ggml dims
/// are `[ne0, ne1, ne2]` = `[in, ff, n_expert]` (gate/up) or `[ff, in, n_expert]`
/// (down); expert `E` occupies the contiguous `per = ne0*ne1` elements at element
/// offset `E*per`. Each expert is labeled `[out, in] = [ne1, ne0]` — the same rule
/// the eager `split_experts` used.
fn index_experts(
    desc_idx: usize,
    d: &TensorDesc,
    layer_prefix: &str,
    proj: &str,
    index: &mut HashMap<String, Entry>,
) -> Result<()> {
    let ne0 = d.dims[0] as usize; // in (gate/up) or ff (down)
    let ne1 = d.dims[1] as usize; // out
    let n_expert = d.dims[2] as usize;
    let per = ne0 * ne1;
    // Block types require per-expert weights to be a whole number of QK_K blocks so
    // expert boundaries land on super-block boundaries (else a block would straddle
    // two experts and the sub-slice dequant would corrupt). F32/F16 are per-element
    // and exempt.
    if matches!(d.ty, GgmlType::Q4K | GgmlType::Q6K) && !per.is_multiple_of(QK_K) {
        return Err(ArfError::model_load(format!(
            "expert tensor {:?}: per-expert element count {per} is not a multiple of {QK_K}",
            d.name
        )));
    }
    for e in 0..n_expert {
        index.insert(
            format!("{layer_prefix}.mlp.experts.{e}.{proj}.weight"),
            Entry::Expert {
                desc_idx,
                expert: e,
                per_elems: per,
                rows: ne1,
                cols: ne0,
            },
        );
    }
    Ok(())
}

/// Un-permute the rows of a llama.cpp-converted attention Q/K matrix.
///
/// llama.cpp's `LlamaModel` HF→GGUF converter applies, per head, the permute
/// `w.reshape(2, head_dim/2, cols).swapaxes(0, 1)` — reordering each head's
/// `head_dim` rows from HF's split-half RoPE layout (first `head_dim/2` rows pair
/// with the last `head_dim/2`) into ggml's interleaved layout (adjacent rows form a
/// rotary pair). Arf applies RoPE in the split-half convention, so we invert the
/// permute on load: ggml row `k` within a head came from HF row `(k % 2) * half +
/// (k / 2)`. Equivalently, HF row `j = g*half + h` (g∈{0,1}) sits at ggml row
/// `2*h + g`. Empirically validated: q/k cosine-to-safetensors jumps 0.02 → 0.997.
fn unpermute_qk_rows(data: &[f32], rows: usize, cols: usize, head_dim: usize) -> Vec<f32> {
    debug_assert_eq!(data.len(), rows * cols);
    debug_assert_eq!(
        rows % head_dim,
        0,
        "rows {rows} not a multiple of head_dim {head_dim}"
    );
    let half = head_dim / 2;
    let n_head = rows / head_dim;
    let mut out = vec![0.0f32; data.len()];
    for h in 0..n_head {
        for k in 0..head_dim {
            let hf_i = (k % 2) * half + (k / 2);
            let src_row = h * head_dim + k; // ggml row
            let dst_row = h * head_dim + hf_i; // HF row
            out[dst_row * cols..dst_row * cols + cols]
                .copy_from_slice(&data[src_row * cols..src_row * cols + cols]);
        }
    }
    out
}

/// Dequantize `n` elements of ggml type `ty` from `bytes` to f32.
fn dequantize(ty: GgmlType, bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    match ty {
        GgmlType::F32 => {
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                out.push(f32::from_le_bytes(
                    bytes[i * 4..i * 4 + 4].try_into().unwrap(),
                ));
            }
            Ok(out)
        }
        GgmlType::F16 => {
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let bits = u16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap());
                out.push(f16_to_f32(bits));
            }
            Ok(out)
        }
        GgmlType::BF16 => {
            // bf16 is the high 16 bits of an f32; widen by left-shifting back.
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let bits = u16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap());
                out.push(f32::from_bits((bits as u32) << 16));
            }
            Ok(out)
        }
        GgmlType::Q40 => dequant_q4_0(bytes, n),
        GgmlType::Q41 => dequant_q4_1(bytes, n),
        GgmlType::Q80 => dequant_q8_0(bytes, n),
        GgmlType::Q4K => dequant_q4_k(bytes, n),
        GgmlType::Q5K => dequant_q5_k(bytes, n),
        GgmlType::Q6K => dequant_q6_k(bytes, n),
        GgmlType::Q3K => dequant_q3_k(bytes, n),
        GgmlType::Iq3S => dequant_iq3_s(bytes, n),
        GgmlType::Iq4Nl => dequant_iq4_nl(bytes, n),
        GgmlType::Iq4Xs => dequant_iq4_xs(bytes, n),
    }
}

/// ggml Q4_0 block: 18 bytes per 32 weights — `d` (f16) then 16 bytes of codes.
/// Byte `k` (0..16) packs element `k` in its LOW nibble and element `k+16` in its
/// HIGH nibble; `y = d·(nibble - 8)` (symmetric). Matches ggml `dequantize_row_q4_0`.
pub(crate) fn dequant_q4_0(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    let nblocks = n / QK4_0;
    let block = 2 + 16; // 18
    let mut out = vec![0.0f32; n];
    for b in 0..nblocks {
        let p = b * block;
        let d = f16_to_f32(u16::from_le_bytes([bytes[p], bytes[p + 1]]));
        let qs = &bytes[p + 2..p + 18]; // 16 bytes → 32 nibbles
        let o = b * QK4_0;
        for k in 0..16 {
            let byte = qs[k];
            out[o + k] = d * ((byte & 0x0f) as f32 - 8.0);
            out[o + k + 16] = d * ((byte >> 4) as f32 - 8.0);
        }
    }
    Ok(out)
}

/// ggml Q4_1 block: 20 bytes per 32 weights — `d` (f16), `m` (f16), then 16 bytes of
/// codes. Asymmetric: `y = d·q + m` with `q` the raw 0..15 nibble (no -8 offset). Byte `k`
/// packs element `k` in its LOW nibble and element `k+16` in its HIGH nibble. Matches ggml
/// `dequantize_row_q4_1`.
pub(crate) fn dequant_q4_1(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    let nblocks = n / QK4_0; // Q4_1 uses the same 32-element block as Q4_0
    let block = 2 + 2 + 16; // 20
    let mut out = vec![0.0f32; n];
    for b in 0..nblocks {
        let p = b * block;
        let d = f16_to_f32(u16::from_le_bytes([bytes[p], bytes[p + 1]]));
        let m = f16_to_f32(u16::from_le_bytes([bytes[p + 2], bytes[p + 3]]));
        let qs = &bytes[p + 4..p + 20]; // 16 bytes → 32 nibbles
        let o = b * QK4_0;
        for k in 0..16 {
            let byte = qs[k];
            out[o + k] = d * (byte & 0x0f) as f32 + m;
            out[o + k + 16] = d * (byte >> 4) as f32 + m;
        }
    }
    Ok(out)
}

/// ggml Q8_0 block: 34 bytes per 32 weights — `d` (f16) then 32 signed int8 codes.
/// Symmetric: `y = d · q` (q ∈ -128..127). Matches ggml `dequantize_row_q8_0`.
pub(crate) fn dequant_q8_0(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    let nblocks = n / QK4_0; // Q8_0 uses the same 32-element block as Q4_0
    let block = 2 + 32; // 34
    let mut out = vec![0.0f32; n];
    for b in 0..nblocks {
        let p = b * block;
        let d = f16_to_f32(u16::from_le_bytes([bytes[p], bytes[p + 1]]));
        let qs = &bytes[p + 2..p + 34];
        let o = b * QK4_0;
        for k in 0..32 {
            out[o + k] = d * (qs[k] as i8) as f32;
        }
    }
    Ok(out)
}

pub use arf_ml::dtype::f16_to_f32;

const QK_K: usize = 256;

/// Bytes per ggml block (block types) or per element (F32/F16). Block types pack
/// `QK_K` (256) weights per block.
fn ggml_block_bytes(ty: GgmlType) -> usize {
    match ty {
        GgmlType::F32 => 4,     // per element
        GgmlType::F16 => 2,     // per element
        GgmlType::BF16 => 2,    // per element
        GgmlType::Q40 => 18,    // per 32 weights (f16 d + 16B codes)
        GgmlType::Q41 => 20,    // per 32 weights (f16 d + f16 m + 16B codes)
        GgmlType::Q80 => 34,    // per 32 weights (f16 d + 32 int8 codes)
        GgmlType::Q4K => 144,   // per 256 weights
        GgmlType::Q5K => 176,   // per 256 weights
        GgmlType::Q6K => 210,   // per 256 weights
        GgmlType::Q3K => 110,   // per 256 weights
        GgmlType::Iq3S => 110,  // per 256 weights
        GgmlType::Iq4Xs => 136, // per 256 weights
        GgmlType::Iq4Nl => 18,  // per 32 weights
    }
}

/// Byte length on disk of `n` elements of `ty`. For block types `n` must be a
/// multiple of `QK_K` (the `Expert` index builder asserts this; whole tensors in
/// this checkpoint already satisfy it).
fn ggml_byte_len(ty: GgmlType, n: usize) -> usize {
    match ty {
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => n * ggml_block_bytes(ty),
        // IQ4_NL joins the 32-weight block types; IQ4_XS is a 256-weight superblock.
        GgmlType::Q40 | GgmlType::Q41 | GgmlType::Q80 | GgmlType::Iq4Nl => {
            (n / QK4_0) * ggml_block_bytes(ty)
        }
        GgmlType::Q4K
        | GgmlType::Q5K
        | GgmlType::Q6K
        | GgmlType::Iq4Xs
        | GgmlType::Q3K
        | GgmlType::Iq3S => (n / QK_K) * ggml_block_bytes(ty),
    }
}

/// Q4_0 block size (elements per block); ggml `QK4_0`.
const QK4_0: usize = 32;

/// Q3_K (ggml type 11) — 110 bytes per 256 weights.
///
/// Layout (`block_q3_K`, note d is LAST): `hmask[32]`, `qs[64]`, `scales[12]`,
/// `d` f16. Sixteen 16-weight sub-blocks. Each weight is 3 bits: the low 2 from
/// `qs`, and an inverted high bit from `hmask` — set means "do not subtract 4",
/// clear means subtract, so the value lands in [-4, 3]. The sixteen 6-bit scales
/// are packed into 12 bytes and biased by -32.
///
/// Ported from ggml's `dequantize_row_q3_K` (llama.cpp 41fc7584f), including its
/// scale-unpacking shuffle, rather than re-derived.
pub(crate) fn dequant_q3_k(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    let nblocks = n / QK_K;
    let block = 32 + 64 + 12 + 2; // 110
    if bytes.len() < nblocks * block {
        return Err(ArfError::model_load(format!(
            "q3_k: need {} bytes for {n} weights, got {}",
            nblocks * block,
            bytes.len()
        )));
    }
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    let mut out = vec![0.0f32; n];
    for b in 0..nblocks {
        let p = b * block;
        let hmask = &bytes[p..p + 32];
        let qs = &bytes[p + 32..p + 96];
        let sc = &bytes[p + 96..p + 108];
        let d_all = f16_to_f32(u16::from_le_bytes([bytes[p + 108], bytes[p + 109]]));

        // The 12 scale bytes hold sixteen 6-bit values in ggml's interleaved
        // packing; this reproduces its `aux` shuffle exactly.
        let mut aux = [0u32; 4];
        for (k, a) in aux.iter_mut().enumerate().take(3) {
            *a = u32::from_le_bytes([sc[4 * k], sc[4 * k + 1], sc[4 * k + 2], sc[4 * k + 3]]);
        }
        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        aux[0] = (aux[0] & KMASK2) | (((tmp) & KMASK1) << 4);
        aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
        let scales: Vec<i8> = aux
            .iter()
            .flat_map(|a| a.to_le_bytes())
            .map(|x| x as i8)
            .collect();

        let o = b * QK_K;
        let mut y = 0usize;
        let mut is = 0usize;
        let mut m: u8 = 1;
        for nn in (0..QK_K).step_by(128) {
            let qbase = nn / 4; // q advances 32 bytes per 128 weights
            let mut shift = 0u32;
            for _j in 0..4 {
                for half in 0..2 {
                    let dl = d_all * (scales[is] as i32 - 32) as f32;
                    is += 1;
                    let qoff = qbase + half * 16;
                    let hoff = half * 16;
                    for l in 0..16 {
                        let q2 = ((qs[qoff + l] >> shift) & 3) as i32;
                        let sub = if hmask[hoff + l] & m != 0 { 0 } else { 4 };
                        out[o + y] = dl * (q2 - sub) as f32;
                        y += 1;
                    }
                }
                shift += 2;
                m <<= 1;
            }
        }
    }
    Ok(out)
}

/// ggml's `iq3s_grid`: 512 codebook entries, each packing four u8 grid values
/// into a u32. Copied verbatim from llama.cpp 41fc7584f ggml-common.h — a
/// codebook cannot be derived, only transcribed, so it is machine-extracted
/// rather than retyped.
#[rustfmt::skip]
const IQ3S_GRID: [u32; 512] = [
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
];

/// IQ3_S (ggml type 21) — 110 bytes per 256 weights.
///
/// Layout (`block_iq3_s`): `d` f16, `qs[64]`, `qh[8]`, `signs[32]`, `scales[4]`.
/// Each 32-weight group takes a 4-bit scale (nibble-packed, two groups per byte)
/// used as `d * (1 + 2*s)`. Every 8 weights come from two grid entries selected
/// by a `qs` byte plus one extra high bit out of `qh`, and each of those 8 gets
/// a sign flip from the corresponding bit of `signs`.
///
/// Ported from `dequantize_row_iq3_s` (llama.cpp 41fc7584f), keeping its exact
/// shift arithmetic for the 9th index bit.
pub(crate) fn dequant_iq3_s(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    let nblocks = n / QK_K;
    let block = 2 + 64 + 8 + 32 + 4; // 110
    if bytes.len() < nblocks * block {
        return Err(ArfError::model_load(format!(
            "iq3_s: need {} bytes for {n} weights, got {}",
            nblocks * block,
            bytes.len()
        )));
    }
    let mut out = vec![0.0f32; n];
    for b in 0..nblocks {
        let p = b * block;
        let d = f16_to_f32(u16::from_le_bytes([bytes[p], bytes[p + 1]]));
        let qs = &bytes[p + 2..p + 66];
        let qh = &bytes[p + 66..p + 74];
        let signs = &bytes[p + 74..p + 106];
        let scales = &bytes[p + 106..p + 110];

        let o = b * QK_K;
        let mut y = 0usize;
        // Two 32-weight groups per iteration; they share one scales byte.
        for ib32 in (0..QK_K / 32).step_by(2) {
            let sc = scales[ib32 / 2];
            let db = [
                d * (1 + 2 * (sc & 0xf) as u32) as f32,
                d * (1 + 2 * (sc >> 4) as u32) as f32,
            ];
            for (half, &dbx) in db.iter().enumerate() {
                let qs_off = (ib32 + half) * 8;
                let sg_off = (ib32 + half) * 4;
                let qhb = qh[ib32 + half];
                for l in 0..4 {
                    // The 9th index bit comes out of qh with these exact shifts.
                    let i1 = qs[qs_off + 2 * l] as usize | (((qhb as usize) << (8 - 2 * l)) & 256);
                    let i2 =
                        qs[qs_off + 2 * l + 1] as usize | (((qhb as usize) << (7 - 2 * l)) & 256);
                    let g1 = IQ3S_GRID[i1].to_le_bytes();
                    let g2 = IQ3S_GRID[i2].to_le_bytes();
                    let sgn = signs[sg_off + l];
                    for j in 0..4 {
                        let s1 = if sgn & (1 << j) != 0 { -1.0 } else { 1.0 };
                        let s2 = if sgn & (1 << (j + 4)) != 0 { -1.0 } else { 1.0 };
                        out[o + y + j] = dbx * g1[j] as f32 * s1;
                        out[o + y + j + 4] = dbx * g2[j] as f32 * s2;
                    }
                    y += 8;
                }
            }
        }
    }
    Ok(out)
}

/// The 16-entry non-linear codebook shared by IQ4_NL and IQ4_XS. A 4-bit index
/// selects one of these int8 values (ggml's `kvalues_iq4nl`); the block scale
/// then maps them to f32. This is what makes the "I" quants non-uniform: the
/// levels are spaced tighter near zero, where weights actually cluster.
const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// IQ4_XS (ggml type 23) — 256 weights in 136 bytes.
///
/// Layout (`block_iq4_xs` in ggml-common.h): `d` f16, `scales_h` u16,
/// `scales_l[4]` u8, `qs[128]` u8. Eight 32-weight sub-blocks each carry a
/// 6-bit scale split across the two scale arrays: 4 low bits in `scales_l`
/// (nibble-packed, two sub-blocks per byte) and 2 high bits in `scales_h`
/// (2 bits per sub-block). The scale is biased by -32, then multiplied by `d`.
/// IQ4_XS dequantization, public so a backend's parity gate can check its GEMV
/// against a reference that is itself unit-tested (the -32 bias, per-sub-block
/// scales, the non-linear codebook). Exposed for that gate rather than for
/// f16 -> f32, public for backends that split raw GGUF blocks themselves (the
/// an IQ4_XS loader that reads the file's f16 super-scales directly).
pub fn f16_to_f32_pub(h: u16) -> f32 {
    f16_to_f32(h)
}

/// general use — the loader goes through `get_f32_shaped`.
pub fn dequant_iq4_xs_pub(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    dequant_iq4_xs(bytes, n)
}

pub(crate) fn dequant_iq4_xs(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    let nblocks = n / QK_K;
    let block = 2 + 2 + 4 + 128; // 136
    if bytes.len() < nblocks * block {
        return Err(ArfError::model_load(format!(
            "iq4_xs: need {} bytes for {n} weights, got {}",
            nblocks * block,
            bytes.len()
        )));
    }
    let mut out = vec![0.0f32; n];
    for b in 0..nblocks {
        let p = b * block;
        let d = f16_to_f32(u16::from_le_bytes([bytes[p], bytes[p + 1]]));
        let scales_h = u16::from_le_bytes([bytes[p + 2], bytes[p + 3]]);
        let scales_l = &bytes[p + 4..p + 8];
        let qs = &bytes[p + 8..p + 136];
        let o = b * QK_K;
        for j in 0..8 {
            // 4 low bits: two sub-blocks share a byte, even in the low nibble.
            let lo = (scales_l[j / 2] >> (4 * (j % 2))) & 0xF;
            // 2 high bits: packed 2-per-sub-block across the u16.
            let hi = ((scales_h >> (2 * j)) & 0x3) as u8;
            let sc = (((hi << 4) | lo) as i32) - 32;
            let dl = d * sc as f32;
            // qs is nibble-packed: sub-block j owns 16 bytes, low nibbles for the
            // first 16 weights and high nibbles for the next 16.
            let qbase = j * 16;
            for i in 0..16 {
                let byte = qs[qbase + i];
                out[o + j * 32 + i] = dl * KVALUES_IQ4NL[(byte & 0xF) as usize] as f32;
                out[o + j * 32 + i + 16] = dl * KVALUES_IQ4NL[(byte >> 4) as usize] as f32;
            }
        }
    }
    Ok(out)
}

/// IQ4_NL (ggml type 20) — 32 weights in 18 bytes: `d` f16 + `qs[16]`.
/// The same codebook as IQ4_XS, but a single scale per 32-weight block and no
/// sub-block scales.
pub(crate) fn dequant_iq4_nl(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    const QK4_NL: usize = 32;
    let nblocks = n / QK4_NL;
    let block = 2 + 16; // 18
    if bytes.len() < nblocks * block {
        return Err(ArfError::model_load(format!(
            "iq4_nl: need {} bytes for {n} weights, got {}",
            nblocks * block,
            bytes.len()
        )));
    }
    let mut out = vec![0.0f32; n];
    for b in 0..nblocks {
        let p = b * block;
        let d = f16_to_f32(u16::from_le_bytes([bytes[p], bytes[p + 1]]));
        let qs = &bytes[p + 2..p + 18];
        let o = b * QK4_NL;
        for i in 0..16 {
            let byte = qs[i];
            out[o + i] = d * KVALUES_IQ4NL[(byte & 0xF) as usize] as f32;
            out[o + i + 16] = d * KVALUES_IQ4NL[(byte >> 4) as usize] as f32;
        }
    }
    Ok(out)
}

/// ggml Q4_K block: 144 bytes per 256 weights.
///   d (f16) | dmin (f16) | scales[12] (6-bit packed: 8 scales + 8 mins) | qs[128]
/// y = d·sc·(q) - dmin·m, per 32-weight sub-block (8 sub-blocks). Matches
/// ggml `dequantize_row_q4_K`.
pub(crate) fn dequant_q4_k(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    let nblocks = n / QK_K;
    let block = 2 + 2 + 12 + 128; // 144
    let mut out = vec![0.0f32; n];
    for b in 0..nblocks {
        let p = b * block;
        let d = f16_to_f32(u16::from_le_bytes([bytes[p], bytes[p + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([bytes[p + 2], bytes[p + 3]]));
        let scales = &bytes[p + 4..p + 16]; // 12 bytes
        let qs = &bytes[p + 16..p + 144]; // 128 bytes (256 nibbles)
        let o = b * QK_K;
        for j in 0..8 {
            let (sc, m) = q4k_get_scale_min(scales, j);
            let d1 = d * sc as f32;
            let m1 = dmin * m as f32;
            // sub-block j covers 32 weights: low nibbles for even j, high for odd —
            // ggml packs two sub-blocks per 32 qs bytes.
            let half = j / 2;
            let qbase = half * 32; // 32 bytes per pair of sub-blocks
            let shift = if j % 2 == 0 { 0 } else { 4 };
            for i in 0..32 {
                let q = ((qs[qbase + i] >> shift) & 0xF) as f32;
                out[o + j * 32 + i] = d1 * q - m1;
            }
        }
    }
    Ok(out)
}

pub(crate) use arf_ml::weight::q4k_get_scale_min;

/// ggml Q5_K block: 176 bytes per 256 weights.
///   d (f16) | dmin (f16) | scales\[12\] (6-bit packed, same as Q4_K) | qh\[32\] (the
///   5th bit of each weight) | qs\[128\] (low 4 bits)
/// y = d·sc·q - dmin·m, where q is the 5-bit value `(qs nibble) + (qh bit ? 16 : 0)`.
/// Matches ggml `dequantize_row_q5_K`: 64 weights per outer step (two 32-weight
/// sub-blocks), the qh bit-selector shifting left by 2 each step so all 8 qh bits
/// are consumed across the four steps.
/// The reference Q5_K dequantizer, exposed so a native reader can be checked
/// against it. A native unpack that transposes the high bit produces
/// plausible magnitudes and wrong weights - silent without this comparison.
pub fn dequant_q5_k_for_test(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    dequant_q5_k(bytes, n)
}

fn dequant_q5_k(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    let nblocks = n / QK_K;
    let block = 2 + 2 + 12 + 32 + 128; // 176
    let mut out = vec![0.0f32; n];
    for b in 0..nblocks {
        let p = b * block;
        let d = f16_to_f32(u16::from_le_bytes([bytes[p], bytes[p + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([bytes[p + 2], bytes[p + 3]]));
        let scales = &bytes[p + 4..p + 16]; // 12 bytes (Q4_K-style packing)
        let qh = &bytes[p + 16..p + 48]; // 32 bytes: the high (5th) bit per weight
        let qs = &bytes[p + 48..p + 176]; // 128 bytes: low nibbles
        let o = b * QK_K;
        for g in 0..4 {
            let is = 2 * g; // scale sub-block indices is and is+1
            let (sc1, m1) = q4k_get_scale_min(scales, is);
            let (sc2, m2) = q4k_get_scale_min(scales, is + 1);
            let d1 = d * sc1 as f32;
            let mm1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let mm2 = dmin * m2 as f32;
            let qbase = g * 32;
            let u1 = 1u8 << (2 * g); // qh bit for the low-nibble sub-block
            let u2 = 2u8 << (2 * g); // qh bit for the high-nibble sub-block
            let obase = o + g * 64;
            for l in 0..32 {
                let hi = if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
                let q = (qs[qbase + l] & 0xF) as f32 + hi;
                out[obase + l] = d1 * q - mm1;
            }
            for l in 0..32 {
                let hi = if qh[l] & u2 != 0 { 16.0 } else { 0.0 };
                let q = (qs[qbase + l] >> 4) as f32 + hi;
                out[obase + 32 + l] = d2 * q - mm2;
            }
        }
    }
    Ok(out)
}

/// ggml Q6_K block: 210 bytes per 256 weights.
///   ql[128] (low 4 bits) | qh[64] (high 2 bits) | scales[16] (i8) | d (f16)
/// y = d · scale · q, q in [-32,31]. Matches ggml `dequantize_row_q6_K`.
fn dequant_q6_k(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    let nblocks = n / QK_K;
    let block = 128 + 64 + 16 + 2; // 210
    let mut out = vec![0.0f32; n];
    for b in 0..nblocks {
        let p = b * block;
        let ql = &bytes[p..p + 128];
        let qh = &bytes[p + 128..p + 192];
        let scales = &bytes[p + 192..p + 208]; // 16 i8
        let d = f16_to_f32(u16::from_le_bytes([bytes[p + 208], bytes[p + 209]]));
        let o = b * QK_K;
        // ggml processes the 256 weights in two halves of 128, each with its own
        // ql/qh sub-indexing; we follow its exact loop.
        for n2 in 0..2 {
            let base_ql = n2 * 64;
            let base_qh = n2 * 32;
            let base_sc = n2 * 8;
            let base_out = o + n2 * 128;
            for l in 0..32 {
                let is = l / 16; // 0 or 1 within this 64-block group of scales
                let q1 = ((ql[base_ql + l] & 0xF) as i32) | (((qh[base_qh + l] & 3) as i32) << 4);
                let q2 = ((ql[base_ql + l + 32] & 0xF) as i32)
                    | ((((qh[base_qh + l] >> 2) & 3) as i32) << 4);
                let q3 = (((ql[base_ql + l] >> 4) & 0xF) as i32)
                    | ((((qh[base_qh + l] >> 4) & 3) as i32) << 4);
                let q4 = (((ql[base_ql + l + 32] >> 4) & 0xF) as i32)
                    | ((((qh[base_qh + l] >> 6) & 3) as i32) << 4);
                let sc = |k: usize| scales[base_sc + k] as i8 as i32;
                out[base_out + l] = d * sc(is) as f32 * (q1 - 32) as f32;
                out[base_out + l + 32] = d * sc(is + 2) as f32 * (q2 - 32) as f32;
                out[base_out + l + 64] = d * sc(is + 4) as f32 * (q3 - 32) as f32;
                out[base_out + l + 96] = d * sc(is + 6) as f32 * (q4 - 32) as f32;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod gguf_meta_tests {
    use super::*;

    // Build a minimal GGUF KV blob: one key "tokenizer.ggml.tokens" as array<string>.
    fn gguf_string_array_kv(key: &str, items: &[&str]) -> Vec<u8> {
        let mut b = Vec::new();
        // key string: u64 len + bytes
        b.extend_from_slice(&(key.len() as u64).to_le_bytes());
        b.extend_from_slice(key.as_bytes());
        b.extend_from_slice(&9u32.to_le_bytes()); // value type 9 = array
        b.extend_from_slice(&8u32.to_le_bytes()); // element type 8 = string
        b.extend_from_slice(&(items.len() as u64).to_le_bytes());
        for it in items {
            b.extend_from_slice(&(it.len() as u64).to_le_bytes());
            b.extend_from_slice(it.as_bytes());
        }
        b
    }

    #[test]
    fn reads_string_array() {
        let blob = gguf_string_array_kv("tokenizer.ggml.tokens", &["hello", "world"]);
        let mut r = Reader::new(&blob);
        let key = r.string().unwrap();
        assert_eq!(key, "tokenizer.ggml.tokens");
        let vty = r.u32().unwrap();
        assert_eq!(vty, 9);
        let got = r.string_array(vty).unwrap();
        assert_eq!(got, vec!["hello".to_string(), "world".to_string()]);
    }

    // Build a minimal GGUF KV blob: one key as array<i32> (ety 5, GGUF value type 9
    // wrapping element type 5). Mirrors `gguf_string_array_kv`.
    fn gguf_i32_array_kv(key: &str, items: &[i32]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&(key.len() as u64).to_le_bytes());
        b.extend_from_slice(key.as_bytes());
        b.extend_from_slice(&9u32.to_le_bytes()); // value type 9 = array
        b.extend_from_slice(&5u32.to_le_bytes()); // element type 5 = i32
        b.extend_from_slice(&(items.len() as u64).to_le_bytes());
        for it in items {
            b.extend_from_slice(&it.to_le_bytes());
        }
        b
    }

    // Same shape but element type 11 = i64.
    fn gguf_i64_array_kv(key: &str, items: &[i64]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&(key.len() as u64).to_le_bytes());
        b.extend_from_slice(key.as_bytes());
        b.extend_from_slice(&9u32.to_le_bytes()); // value type 9 = array
        b.extend_from_slice(&11u32.to_le_bytes()); // element type 11 = i64
        b.extend_from_slice(&(items.len() as u64).to_le_bytes());
        for it in items {
            b.extend_from_slice(&it.to_le_bytes());
        }
        b
    }

    #[test]
    fn reads_i32_array() {
        // DFlash's actual tap-index shape: [1, 12, 23, 34, 45].
        let blob = gguf_i32_array_kv("dflash.target_layers", &[1, 12, 23, 34, 45]);
        let mut r = Reader::new(&blob);
        let key = r.string().unwrap();
        assert_eq!(key, "dflash.target_layers");
        let vty = r.u32().unwrap();
        assert_eq!(vty, 9);
        let got = r.i32_array(vty).unwrap();
        assert_eq!(got, vec![1, 12, 23, 34, 45]);
    }

    #[test]
    fn reads_i32_array_empty() {
        let blob = gguf_i32_array_kv("dflash.empty", &[]);
        let mut r = Reader::new(&blob);
        let _key = r.string().unwrap();
        let vty = r.u32().unwrap();
        let got = r.i32_array(vty).unwrap();
        assert!(got.is_empty());
    }

    #[test]
    fn reads_i32_array_negative() {
        let blob = gguf_i32_array_kv("dflash.signed", &[-1, 0, i32::MIN, i32::MAX]);
        let mut r = Reader::new(&blob);
        let _key = r.string().unwrap();
        let vty = r.u32().unwrap();
        let got = r.i32_array(vty).unwrap();
        assert_eq!(got, vec![-1, 0, i32::MIN, i32::MAX]);
    }

    #[test]
    fn reads_i64_array() {
        let blob = gguf_i64_array_kv("dflash.target_layers_wide", &[1, 12, 23, 34, 45]);
        let mut r = Reader::new(&blob);
        let _key = r.string().unwrap();
        let vty = r.u32().unwrap();
        let got = r.i64_array(vty).unwrap();
        assert_eq!(got, vec![1i64, 12, 23, 34, 45]);
    }

    #[test]
    fn reads_i64_array_empty() {
        let blob = gguf_i64_array_kv("dflash.empty_wide", &[]);
        let mut r = Reader::new(&blob);
        let _key = r.string().unwrap();
        let vty = r.u32().unwrap();
        let got = r.i64_array(vty).unwrap();
        assert!(got.is_empty());
    }

    #[test]
    fn reads_i64_array_negative() {
        let blob = gguf_i64_array_kv("dflash.signed_wide", &[-1, 0, i64::MIN, i64::MAX]);
        let mut r = Reader::new(&blob);
        let _key = r.string().unwrap();
        let vty = r.u32().unwrap();
        let got = r.i64_array(vty).unwrap();
        assert_eq!(got, vec![-1i64, 0, i64::MIN, i64::MAX]);
    }

    #[test]
    fn i32_array_rejects_wrong_element_type() {
        // ety 6 = f32, not 5 = i32 → must error, not silently misparse.
        let mut b = Vec::new();
        b.extend_from_slice(&6u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        let mut r = Reader::new(&b);
        assert!(r.i32_array(9).is_err());
    }

    #[test]
    fn i32_array_rejects_wrong_outer_value_type() {
        // vty must be 9 (array); passing a scalar type must error.
        let b: Vec<u8> = Vec::new();
        let mut r = Reader::new(&b);
        assert!(r.i32_array(5).is_err());
    }

    #[test]
    fn i64_array_rejects_wrong_element_type() {
        // ety 5 = i32, not 11 = i64 → must error.
        let mut b = Vec::new();
        b.extend_from_slice(&5u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        let mut r = Reader::new(&b);
        assert!(r.i64_array(9).is_err());
    }

    /// Wrap a raw KV-section blob (as produced by `gguf_i32_array_kv` etc.) in a full
    /// GGUF header (magic + version 3 + n_tensors=0 + n_kv=1) so it can be opened via
    /// [`LazyGguf::open_raw`] and read back through the public metadata accessors —
    /// this exercises `with_meta_value`'s real re-scan path, not just the `Reader`
    /// primitive directly.
    fn wrap_gguf_file(kv_blob: &[u8]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes()); // version 3
        b.extend_from_slice(&0u64.to_le_bytes()); // n_tensors = 0
        b.extend_from_slice(&1u64.to_le_bytes()); // n_kv = 1
        b.extend_from_slice(kv_blob);
        b
    }

    #[test]
    fn lazy_gguf_get_metadata_i32_array_roundtrip() {
        let kv = gguf_i32_array_kv("dflash.target_layers", &[1, 12, 23, 34, 45]);
        let file_bytes = wrap_gguf_file(&kv);
        let dir = std::env::temp_dir();
        let path = dir.join("arf_test_i32_array.gguf");
        std::fs::write(&path, &file_bytes).unwrap();

        let g = LazyGguf::open_raw(&path).expect("open_raw synthetic gguf");
        assert_eq!(
            g.get_metadata_i32_array("dflash.target_layers"),
            Some(vec![1, 12, 23, 34, 45])
        );
        // Absent key → None, not a panic.
        assert_eq!(g.get_metadata_i32_array("dflash.missing"), None);
        // Wrong-type accessor on a present-but-different-typed key → None.
        assert_eq!(g.get_metadata_f32_array("dflash.target_layers"), None);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn lazy_gguf_get_metadata_i32_array_empty_roundtrip() {
        let kv = gguf_i32_array_kv("dflash.empty", &[]);
        let file_bytes = wrap_gguf_file(&kv);
        let dir = std::env::temp_dir();
        let path = dir.join("arf_test_i32_array_empty.gguf");
        std::fs::write(&path, &file_bytes).unwrap();

        let g = LazyGguf::open_raw(&path).expect("open_raw synthetic gguf");
        assert_eq!(g.get_metadata_i32_array("dflash.empty"), Some(vec![]));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn lazy_gguf_get_metadata_i64_array_roundtrip() {
        // Negative + boundary values to prove i64 (not truncated to i32) round-trips.
        let kv = gguf_i64_array_kv(
            "dflash.target_layers_wide",
            &[-1, 0, 45, i64::MIN, i64::MAX],
        );
        let file_bytes = wrap_gguf_file(&kv);
        let dir = std::env::temp_dir();
        let path = dir.join("arf_test_i64_array.gguf");
        std::fs::write(&path, &file_bytes).unwrap();

        let g = LazyGguf::open_raw(&path).expect("open_raw synthetic gguf");
        assert_eq!(
            g.get_metadata_i64_array("dflash.target_layers_wide"),
            Some(vec![-1i64, 0, 45, i64::MIN, i64::MAX])
        );
        assert_eq!(g.get_metadata_i64_array("dflash.missing"), None);

        let _ = std::fs::remove_file(&path);
    }

    /// One `string` KV (value type 8): u64-len key, vty, u64-len UTF-8 value.
    fn gguf_string_kv(key: &str, value: &str) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&(key.len() as u64).to_le_bytes());
        b.extend_from_slice(key.as_bytes());
        b.extend_from_slice(&8u32.to_le_bytes()); // value type 8 = string
        b.extend_from_slice(&(value.len() as u64).to_le_bytes());
        b.extend_from_slice(value.as_bytes());
        b
    }

    /// `tokenizer.chat_template` round-trips byte for byte — newlines, quotes, Jinja braces and
    /// non-ASCII included — and a KV of another type or an absent key reads as `None`.
    #[test]
    fn lazy_gguf_get_metadata_string_roundtrip() {
        let tmpl = "{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}\"é\"";
        let mut file_bytes = Vec::new();
        file_bytes.extend_from_slice(b"GGUF");
        file_bytes.extend_from_slice(&3u32.to_le_bytes());
        file_bytes.extend_from_slice(&0u64.to_le_bytes()); // n_tensors
        file_bytes.extend_from_slice(&2u64.to_le_bytes()); // n_kv = 2: the string AFTER an array
        file_bytes.extend_from_slice(&gguf_i32_array_kv("dflash.target_layers", &[1, 2]));
        file_bytes.extend_from_slice(&gguf_string_kv("tokenizer.chat_template", tmpl));
        let path = std::env::temp_dir().join("arf_test_string_meta.gguf");
        std::fs::write(&path, &file_bytes).unwrap();

        let g = LazyGguf::open_raw(&path).expect("open_raw synthetic gguf");
        assert_eq!(
            g.get_metadata_string("tokenizer.chat_template").as_deref(),
            Some(tmpl)
        );
        assert_eq!(g.get_metadata_string("tokenizer.missing"), None);
        // A present key of another type is not a string.
        assert_eq!(g.get_metadata_string("dflash.target_layers"), None);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn is_gguf_file_detects_magic() {
        use std::io::Write;
        // A non-file path is not a gguf.
        assert!(!is_gguf_file(std::path::Path::new("/nonexistent/path/xyz")));

        // A file starting with the GGUF magic IS detected; wrong magic is not.
        let dir = std::env::temp_dir();
        let good = dir.join("arf_test_good.gguf");
        let bad = dir.join("arf_test_bad.bin");
        std::fs::File::create(&good)
            .unwrap()
            .write_all(b"GGUF\x03\x00\x00\x00")
            .unwrap();
        std::fs::File::create(&bad)
            .unwrap()
            .write_all(b"XXXXdata")
            .unwrap();
        assert!(is_gguf_file(&good));
        assert!(!is_gguf_file(&bad));
        // A file shorter than 4 bytes is not a gguf.
        let tiny = dir.join("arf_test_tiny.bin");
        std::fs::File::create(&tiny)
            .unwrap()
            .write_all(b"GG")
            .unwrap();
        assert!(!is_gguf_file(&tiny));
        // Cleanup (best-effort).
        let _ = std::fs::remove_file(&good);
        let _ = std::fs::remove_file(&bad);
        let _ = std::fs::remove_file(&tiny);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The mmproj GGUF (Gemma-3 vision tower + projector) is indexed under `vision.*`
    /// keys with correct shapes, and a representative tensor dequantizes. Skips if the
    /// file isn't present (it's ~851MB, not committed).
    #[test]
    fn vision_tensors_indexed_and_dequant() {
        let path = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../models/gemma-3-4b-vision/mmproj-F16.gguf"
        ));
        if !path.exists() {
            eprintln!("SKIP: mmproj GGUF not found at {path:?}");
            return;
        }
        let cfg = crate::config::ModelConfig::gemma3_4b();
        let g = LazyGguf::open(path, &cfg).expect("open mmproj");

        // projector input_projection: ggml dims [2560,1152] → loader [rows=ne1, cols=ne0]
        // = [1152, 2560] (same [out,in]=[ne1,ne0] convention as every other matrix).
        // soft_emb_norm is a [1152] vector.
        assert_eq!(
            g.shape("vision.mm.input_projection.weight"),
            Some(vec![1152, 2560]),
            "projector matrix shape"
        );
        assert_eq!(
            g.shape("vision.mm.soft_emb_norm.weight"),
            Some(vec![1152]),
            "soft_emb_norm vector"
        );
        // vision layer 0 attention q weight [1152,1152]; ln1 [1152]; patch-embed conv
        // is a flat 14*14*3*1152 vector.
        assert_eq!(
            g.shape("vision.v.blk.0.attn_q.weight"),
            Some(vec![1152, 1152])
        );
        assert_eq!(g.shape("vision.v.blk.0.ln1.weight"), Some(vec![1152]));
        assert_eq!(
            g.shape("vision.v.patch_embd.weight"),
            Some(vec![14 * 14 * 3 * 1152]),
            "patch-embed conv flattened"
        );

        // a representative tensor must dequantize to finite values of the right length.
        let w = g
            .get_f32("vision.mm.input_projection.weight")
            .expect("dequant input_projection");
        assert_eq!(w.len(), 1152 * 2560);
        assert!(w.iter().all(|v| v.is_finite()));
        assert!(w.iter().any(|&v| v != 0.0), "not all zero");
    }

    #[test]
    fn f16_roundtrip_known_values() {
        assert_eq!(f16_to_f32(0x3C00), 1.0); // 1.0
        assert_eq!(f16_to_f32(0x4000), 2.0); // 2.0
        assert_eq!(f16_to_f32(0xC000), -2.0); // -2.0
        assert_eq!(f16_to_f32(0x0000), 0.0); // +0
        assert!((f16_to_f32(0x3555) - 0.333).abs() < 1e-3); // ~1/3
    }

    #[test]
    fn q4k_scale_min_extract_matches_ggml() {
        // All-zero scales → all zero (degenerate but exercises the bit logic).
        let scales = [0u8; 12];
        for j in 0..8 {
            assert_eq!(q4k_get_scale_min(&scales, j), (0, 0));
        }
        // j<4 path: low 6 bits of scales[j]/scales[j+4].
        let mut s = [0u8; 12];
        s[0] = 0b0010_1010; // 42 & 63 = 42
        s[4] = 0b0001_0101; // 21
        assert_eq!(q4k_get_scale_min(&s, 0), (42, 21));
    }

    fn f32_to_f16_bits(x: f32) -> u16 {
        // Minimal f32→f16 for building test blocks (round-to-nearest-even not
        // needed; the values we pick are exactly representable).
        let b = x.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
        let mant = (b >> 13) & 0x3ff;
        if exp <= 0 {
            sign // flush tiny to signed zero (only used for 0.0 here)
        } else {
            sign | ((exp as u16) << 10) | mant as u16
        }
    }

    #[test]
    fn dequant_q4_k_known_block() {
        // One 144-byte Q4_K block (256 weights). Sub-block 0 uses scale index 0;
        // set d=2.0, dmin=1.0, scales[0]=3 (sc), scales[4]=5 (m). Then weight =
        // d*sc*q - dmin*m = 2*3*q - 1*5 = 6q - 5. With nibble q=4 → 19.0.
        let mut bytes = vec![0u8; 144];
        bytes[0..2].copy_from_slice(&f32_to_f16_bits(2.0).to_le_bytes()); // d
        bytes[2..4].copy_from_slice(&f32_to_f16_bits(1.0).to_le_bytes()); // dmin
        bytes[4] = 3; // scales[0] → sc for sub-block 0
        bytes[8] = 5; // scales[4] → m for sub-block 0
                      // qs: first nibble (weight 0) = 4. qs starts at byte 16.
        bytes[16] = 0x04;
        let out = dequant_q4_k(&bytes, QK_K).unwrap();
        assert_eq!(out.len(), QK_K);
        assert!((out[0] - 19.0).abs() < 1e-3, "got {}", out[0]);
        // weight 1 = high nibble of byte 16 = 0 → 6*0 - 5 = -5.
        assert!((out[1] - (-5.0)).abs() < 1e-3, "got {}", out[1]);
    }

    #[test]
    fn dequant_q5_k_known_block() {
        // One 176-byte Q5_K block (256 weights). Layout: d|dmin|scales[12]|qh[32]|qs[128].
        // Sub-block 0 uses scale index 0: d=2.0, dmin=1.0, scales[0]=3 (sc),
        // scales[4]=5 (m) → weight = d*sc*q - dmin*m = 6q - 5.
        let mut bytes = vec![0u8; 176];
        bytes[0..2].copy_from_slice(&f32_to_f16_bits(2.0).to_le_bytes()); // d
        bytes[2..4].copy_from_slice(&f32_to_f16_bits(1.0).to_le_bytes()); // dmin
        bytes[4] = 3; // scales[0] → sc for sub-block 0
        bytes[8] = 5; // scales[4] → m for sub-block 0
                      // qh starts at byte 16, qs at byte 48. weight 0 low nibble = 4,
                      // its high (5th) bit (qh[0] & 1) = 0 → q = 4 → 6*4 - 5 = 19.
        bytes[48] = 0x04;
        let out = dequant_q5_k(&bytes, QK_K).unwrap();
        assert_eq!(out.len(), QK_K);
        assert!((out[0] - 19.0).abs() < 1e-3, "got {}", out[0]);

        // Now set weight 0's high bit (qh[0] bit 0) → q = 4 + 16 = 20 → 6*20 - 5 = 115.
        bytes[16] = 0x01;
        let out = dequant_q5_k(&bytes, QK_K).unwrap();
        assert!((out[0] - 115.0).abs() < 1e-3, "got {}", out[0]);
    }

    #[test]
    fn dequant_q6_k_known_block() {
        // One 210-byte Q6_K block. d at byte 208. scales (16 i8) at byte 192.
        // q = (ql_lo | qh<<4); weight = d * scale * (q - 32). Set d=0.5,
        // scales[0]=4, ql[0]=10, qh bits 0 → q=10, weight = 0.5*4*(10-32) = -44.
        let mut bytes = vec![0u8; 210];
        bytes[192] = 4; // scales[0] (i8)
        bytes[208..210].copy_from_slice(&f32_to_f16_bits(0.5).to_le_bytes()); // d
        bytes[0] = 10; // ql[0] low nibble = 10, high nibble (weight 64) = 0
        let out = dequant_q6_k(&bytes, QK_K).unwrap();
        assert_eq!(out.len(), QK_K);
        assert!((out[0] - (-44.0)).abs() < 1e-3, "got {}", out[0]);
    }

    #[test]
    fn ggml_byte_len_matches_block_layout() {
        // 256 weights = 1 super-block. Q4_K = 144 B, Q6_K = 210 B, F32 = 1024 B.
        assert_eq!(ggml_byte_len(GgmlType::Q4K, 256), 144);
        assert_eq!(ggml_byte_len(GgmlType::Q5K, 256), 176);
        assert_eq!(ggml_byte_len(GgmlType::Q6K, 256), 210);
        assert_eq!(ggml_byte_len(GgmlType::F32, 256), 256 * 4);
        assert_eq!(ggml_byte_len(GgmlType::F16, 256), 256 * 2);
        // A whole expert (1,572,864 = 6144 blocks): exact integer multiple.
        assert_eq!(ggml_byte_len(GgmlType::Q4K, 1_572_864), 6144 * 144);
        assert_eq!(ggml_byte_len(GgmlType::Q5K, 1_572_864), 6144 * 176);
        assert_eq!(ggml_byte_len(GgmlType::Q6K, 1_572_864), 6144 * 210);
        // Q4_0: 32 weights = 1 block = 18 B.
        assert_eq!(ggml_byte_len(GgmlType::Q40, 32), 18);
        assert_eq!(ggml_byte_len(GgmlType::Q40, 32 * 100), 18 * 100);
        assert_eq!(GgmlType::from_id(2).unwrap() as u8, GgmlType::Q40 as u8);
    }

    #[test]
    fn dequant_q4_0_matches_ggml_formula() {
        // One 32-weight Q4_0 block: d = 2.0 (f16 bits 0x4000), 16 code bytes.
        // Byte k packs element k (low nibble) and element k+16 (high nibble);
        // y = d·(nibble - 8). Build a known block and check every element.
        let d_bits: u16 = 0x4000; // f16 2.0
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&d_bits.to_le_bytes());
        // codes: byte k = low nibble (k%16), high nibble ((k+3)%16) — arbitrary but known.
        for k in 0..16u8 {
            let lo = k % 16;
            let hi = (k + 3) % 16;
            bytes.push((hi << 4) | lo);
        }
        let out = dequant_q4_0(&bytes, 32).unwrap();
        let d = f16_to_f32(d_bits);
        assert!((d - 2.0).abs() < 1e-3);
        for k in 0..16usize {
            let lo = (k as u8 % 16) as f32;
            let hi = ((k as u8 + 3) % 16) as f32;
            assert!((out[k] - d * (lo - 8.0)).abs() < 1e-5, "elem {k}");
            assert!(
                (out[k + 16] - d * (hi - 8.0)).abs() < 1e-5,
                "elem {}",
                k + 16
            );
        }
    }

    #[test]
    fn dequant_q4_1_matches_ggml_formula() {
        // One 32-weight Q4_1 block: d = 2.0, m = -3.0 (f16), 16 code bytes.
        // Byte k packs element k (low nibble) and k+16 (high nibble); y = d·q + m.
        let d_bits: u16 = 0x4000; // f16 2.0
        let m_bits: u16 = f32_to_f16_bits(-3.0);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&d_bits.to_le_bytes());
        bytes.extend_from_slice(&m_bits.to_le_bytes());
        for k in 0..16u8 {
            let lo = k % 16;
            let hi = (k + 5) % 16;
            bytes.push((hi << 4) | lo);
        }
        let out = dequant_q4_1(&bytes, 32).unwrap();
        let d = f16_to_f32(d_bits);
        let m = f16_to_f32(m_bits);
        for k in 0..16usize {
            let lo = (k as u8 % 16) as f32;
            let hi = ((k as u8 + 5) % 16) as f32;
            assert!((out[k] - (d * lo + m)).abs() < 1e-3, "elem {k}");
            assert!((out[k + 16] - (d * hi + m)).abs() < 1e-3, "elem {}", k + 16);
        }
        // byte length + id mapping.
        assert_eq!(ggml_byte_len(GgmlType::Q41, 32), 20);
        assert_eq!(GgmlType::from_id(3).unwrap() as u8, GgmlType::Q41 as u8);
    }

    #[test]
    fn expert_subslice_dequant_equals_whole_then_slice() {
        // THE lazy-loader invariant: dequantizing expert E from its block-aligned
        // byte sub-slice equals dequantizing the whole packed tensor and slicing
        // [E*per .. (E+1)*per]. Use Q4_K with per = 256 (one block/expert) and 3
        // experts, each block with a distinct d so the slices differ.
        let per = QK_K; // 256 weights = 1 Q4_K block per expert
        let n_expert = 3usize;
        let block = 144usize;
        let mut bytes = vec![0u8; n_expert * block];
        for e in 0..n_expert {
            let p = e * block;
            bytes[p..p + 2].copy_from_slice(&f32_to_f16_bits((e + 1) as f32).to_le_bytes()); // d
            bytes[p + 4] = 2; // sc
            bytes[p + 16] = 0x03; // weight 0 nibble = 3
        }
        // Whole then slice.
        let whole = dequant_q4_k(&bytes, n_expert * per).unwrap();
        for e in 0..n_expert {
            let want = &whole[e * per..(e + 1) * per];
            // Sub-slice path: start at the expert's block-aligned byte offset.
            let base = e * ggml_byte_len(GgmlType::Q4K, per);
            let got = dequant_q4_k(&bytes[base..], per).unwrap();
            assert_eq!(
                got, want,
                "expert {e} sub-slice diverged from whole-then-slice"
            );
            // weight 0 = d*sc*3 - 0 = (e+1)*2*3 = 6*(e+1).
            assert!(
                (got[0] - (6 * (e + 1)) as f32).abs() < 1e-3,
                "expert {e} w0 = {}",
                got[0]
            );
        }
    }

    #[test]
    fn unpermute_qk_rows_is_an_involution_pair_with_known_mapping() {
        // 2 heads, head_dim 4 (half 2), cols 1: rows hold their own index as the
        // value so we can read the permutation off directly.
        let head_dim = 4usize;
        let rows = 2 * head_dim;
        let cols = 1usize;
        let data: Vec<f32> = (0..rows).map(|r| r as f32).collect();
        let out = unpermute_qk_rows(&data, rows, cols, head_dim);
        // Within head h, ggml row k (k%2)*half + (k/2): 0->0,1->2,2->1,3->3.
        // So out[hf_row] = ggml_row, i.e. dst rows [0,2,1,3] receive [0,1,2,3].
        assert_eq!(out, vec![0.0, 2.0, 1.0, 3.0, 4.0, 6.0, 5.0, 7.0]);
    }

    /// Regression guard for the llama.cpp Q/K row-permute fix. Dequantize a few named
    /// tensors from the llama-3.2-1B GGUF and compare them, value by value, to the
    /// safetensors checkpoint. After the fix EVERY tensor (including q/k) must align
    /// closely; before it, q/k aligned-cosine was ~0.02. Gated on both files existing
    /// so it is a no-op where the blobs are absent (CI).
    #[test]
    fn gguf_qk_rows_match_safetensors_after_unpermute() {
        use crate::config::ModelConfig;
        // L264 — paths come from the environment so this runs for anyone; it already
        // self-skips when the files are absent, which is what keeps CI green.
        //   ARF_TEST_GGUF_BLOB  a llama-3.2-1b GGUF (e.g. an ollama blob)
        //   ARF_TEST_SAFETENSORS  the matching model.safetensors
        let blob_p = std::env::var("ARF_TEST_GGUF_BLOB").unwrap_or_default();
        let st_p = std::env::var("ARF_TEST_SAFETENSORS").unwrap_or_default();
        let blob = std::path::Path::new(&blob_p);
        let st = std::path::Path::new(&st_p);
        if !blob.exists() || !st.exists() {
            eprintln!("gguf<->safetensors probe skipped: blob or safetensors absent");
            return;
        }
        let cfg = ModelConfig::llama_3_2_1b();
        let g = crate::model::gguf::load_gguf(blob, &cfg).expect("load gguf");
        let s = crate::model::weights::read_weights(&[st]).expect("read safetensors");

        // (name, is_q_or_k). q/k were the permuted tensors; the others guard against a
        // regression that would over-permute a non-q/k matrix.
        let cases: &[&str] = &[
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
        ];
        for name in cases {
            let gv = g
                .get_f32(name)
                .unwrap_or_else(|| panic!("gguf missing {name}"));
            let sv = s
                .get_f32(name)
                .unwrap_or_else(|| panic!("st missing {name}"));
            assert_eq!(
                gv.len(),
                sv.len(),
                "{name}: len {} vs {}",
                gv.len(),
                sv.len()
            );
            let (mut dot, mut ng, mut ns) = (0.0f64, 0.0f64, 0.0f64);
            for (&a, &b) in gv.iter().zip(sv.iter()) {
                dot += a as f64 * b as f64;
                ng += a as f64 * a as f64;
                ns += b as f64 * b as f64;
            }
            let cos = dot / (ng.sqrt() * ns.sqrt());
            // Q4_K dequant vs HF bf16 leaves a small gap; 0.99 is comfortably above
            // the ~0.02 a wrong (permuted) q/k would score.
            assert!(
                cos > 0.99,
                "{name}: aligned cosine {cos:.4} — GGUF weights diverge from safetensors"
            );
        }
    }

    use arf_ml::weight::{Q4KSMatrix, Q4Matrix};

    fn f32_to_f16_bits_test(x: f32) -> u16 {
        let b = x.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
        let mant = (b >> 13) & 0x3ff;
        if exp <= 0 {
            sign
        } else {
            sign | ((exp as u16) << 10) | mant as u16
        }
    }

    fn make_q4k_block(d: f32, dmin: f32, scales12: [u8; 12], qs128: [u8; 128]) -> [u8; 144] {
        let mut block = [0u8; 144];
        block[0..2].copy_from_slice(&f32_to_f16_bits_test(d).to_le_bytes());
        block[2..4].copy_from_slice(&f32_to_f16_bits_test(dmin).to_le_bytes());
        block[4..16].copy_from_slice(&scales12);
        block[16..144].copy_from_slice(&qs128);
        block
    }

    fn assert_bit_exact(got: &[f32], want: &[f32], label: &str) {
        assert_eq!(got.len(), want.len(), "{label}: length mismatch");
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "{label}: diverged at index {i}: got {g} (bits {:08x}) want {w} (bits {:08x})",
                g.to_bits(),
                w.to_bits()
            );
        }
    }

    // ---- ggml-parity for arf-ml's quantized matrices --------------------------------
    // These live here rather than in arf-ml because they cross-check its Q4 decoders
    // against THIS module's independent ggml block decoders. arf-ml cannot depend on
    // arf-core (that is the whole point of the split), so the test sits on the side
    // that can see both.

    #[test]
    fn from_ggml_q4_0_matches_ggml_within_bf16_scale() {
        // Build synthetic ggml Q4_0 blocks (18 bytes each: f16 d + 16 code bytes) for a
        // 2-row × 64-col tensor and compare the native transcode's dequant against ggml's
        // dequant_q4_0. The CODES are preserved EXACTLY (this is the whole point — no
        // dequant→re-quant of the codes, which was the corruption source). The only
        // residual is that Q4Matrix stores the per-block scale as bf16 vs ggml's f16, so
        // values match within bf16-scale rounding (≪ the double-quant error we removed).
        // minimal f32→f16 (round-to-nearest-even, normals only — fine for these scales).
        fn f32_to_f16(x: f32) -> u16 {
            let bits = x.to_bits();
            let sign = ((bits >> 16) & 0x8000) as u16;
            let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
            let mant = bits & 0x7f_ffff;
            if exp <= 0 {
                return sign; // flush tiny to ±0 (scales here are well within normal range)
            }
            let mut h_mant = (mant >> 13) as u16;
            // round-to-nearest-even on the dropped low 13 bits.
            if (mant & 0x1000) != 0 && ((mant & 0x0fff) != 0 || (h_mant & 1) != 0) {
                h_mant += 1;
            }
            sign | ((exp as u16) << 10) | h_mant
        }
        let (rows, cols) = (2usize, 64usize);
        let blocks_per_row = cols / 32;
        let nblocks = rows * blocks_per_row;
        let mut raw = Vec::with_capacity(nblocks * 18);
        for b in 0..nblocks {
            // vary the scale per block (f16), and the 32 nibble codes 0..15.
            let d_f16 = f32_to_f16(0.05 + 0.01 * b as f32);
            raw.extend_from_slice(&d_f16.to_le_bytes());
            for k in 0..16u8 {
                let lo = (k as u32 * 7 + b as u32) % 16; // element k
                let hi = (k as u32 * 11 + 3 + b as u32) % 16; // element k+16
                raw.push((lo | (hi << 4)) as u8);
            }
        }
        let n = rows * cols;
        let reference = crate::model::gguf::dequant_q4_0(&raw, n).expect("dequant_q4_0");
        let ours = Q4Matrix::from_ggml_q4_0(&raw, rows, cols).to_f32();
        assert_eq!(ours.len(), reference.len());
        for (i, (g, w)) in ours.iter().zip(&reference).enumerate() {
            // sign + relative magnitude must match (codes are exact); abs diff is bounded
            // by the bf16-vs-f16 scale rounding (~2^-8 relative), far below double-quant.
            let rel = (g - w).abs() / w.abs().max(1e-6);
            assert!(
                rel < 0.01,
                "from_ggml_q4_0 diverged at {i}: got {g} want {w} (rel {rel})"
            );
        }
    }

    #[test]
    fn from_ggml_q4k_is_bit_exact_to_ggml_dequant() {
        // (a) Known-block values — mirrors the gguf.rs dequant_q4_k_known_block test.
        // d=2.0, dmin=1.0, scales[0]=3 (sc0), scales[4]=5 (m0).
        // Sub-block 0: w = 2*3*q - 1*5 = 6q - 5. Weight 0 (nibble=4) → 19.0.
        {
            let mut scales = [0u8; 12];
            scales[0] = 3;
            scales[4] = 5;
            let mut qs = [0u8; 128];
            qs[0] = 0x04; // weight 0 low nibble = 4
            let block = make_q4k_block(2.0, 1.0, scales, qs);
            let rows = 1;
            let cols = 256;
            let ggml_out = crate::model::gguf::dequant_q4_k(&block, rows * cols).unwrap();
            let arf_out = Q4KSMatrix::from_ggml_q4k(&block, rows, cols).to_f32();
            assert_bit_exact(&arf_out, &ggml_out, "known_block");
        }

        // (b) Block where ggml min m > 0 so the sign-fold matters.
        // d=1.0, dmin=0.5, sub-block 1 (j=1): scales[1]=10 (sc), scales[5]=63 (m).
        // The m=63 → min_i8 = -63 must reconstruct -(dmin*m) = -0.5*63 = -31.5.
        {
            let mut scales = [0u8; 12];
            scales[1] = 10; // sc for sub-block 1
            scales[5] = 63; // m  for sub-block 1 (maximum 6-bit value)
            let mut qs = [0u8; 128];
            // Sub-block 1 is odd: half=0, shift=4. qs[0] high nibble = 7.
            qs[0] = 0x70;
            let block = make_q4k_block(1.0, 0.5, scales, qs);
            let rows = 1;
            let cols = 256;
            let ggml_out = crate::model::gguf::dequant_q4_k(&block, rows * cols).unwrap();
            let arf_out = Q4KSMatrix::from_ggml_q4k(&block, rows, cols).to_f32();
            assert_bit_exact(&arf_out, &ggml_out, "sign_fold_m63");
        }

        // (c) Pseudo-random blocks — fixed-seed LCG, no rand dependency.
        // Tests that ALL nibble positions map correctly (catches any offset bug).
        {
            // LCG: x_{n+1} = (1664525 * x_n + 1013904223) mod 2^32
            let mut lcg = 0x12345678u32;
            let mut next_byte = || -> u8 {
                lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (lcg >> 24) as u8
            };

            // 3 blocks, 1 row (cols=768 = 3 super-blocks).
            const N_BLOCKS: usize = 3;
            let rows = 1usize;
            let cols = N_BLOCKS * 256;
            let mut raw = vec![0u8; N_BLOCKS * 144];
            for b in 0..N_BLOCKS {
                let p = b * 144;
                // d: small positive (f16-representable, use 1.0..4.0 range).
                let d_val = 1.0 + (next_byte() as f32 / 64.0);
                let dmin_val = 0.5 + (next_byte() as f32 / 128.0);
                raw[p..p + 2].copy_from_slice(&f32_to_f16_bits_test(d_val).to_le_bytes());
                raw[p + 2..p + 4].copy_from_slice(&f32_to_f16_bits_test(dmin_val).to_le_bytes());
                // scales[0..12]: keep values in 6-bit range (0..63).
                for i in 0..12usize {
                    raw[p + 4 + i] = next_byte() & 63;
                }
                // qs[0..128]: random nibble data.
                for i in 0..128usize {
                    raw[p + 16 + i] = next_byte();
                }
            }
            let ggml_out = crate::model::gguf::dequant_q4_k(&raw, rows * cols).unwrap();
            let arf_out = Q4KSMatrix::from_ggml_q4k(&raw, rows, cols).to_f32();
            assert_bit_exact(&arf_out, &ggml_out, "pseudo_random_1row_3blocks");
        }

        // (d) Multi-row: rows=2, cols=256 (2 blocks) — catches row-stride indexing bugs.
        {
            let rows = 2usize;
            let cols = 256;
            let mut lcg = 0xDEADBEEFu32;
            let mut next_byte = || -> u8 {
                lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (lcg >> 24) as u8
            };
            let mut raw = vec![0u8; rows * 144];
            for b in 0..rows {
                let p = b * 144;
                let d_val = 2.0 + (next_byte() as f32 / 128.0);
                let dmin_val = 1.0 + (next_byte() as f32 / 256.0);
                raw[p..p + 2].copy_from_slice(&f32_to_f16_bits_test(d_val).to_le_bytes());
                raw[p + 2..p + 4].copy_from_slice(&f32_to_f16_bits_test(dmin_val).to_le_bytes());
                for i in 0..12usize {
                    raw[p + 4 + i] = next_byte() & 63;
                }
                for i in 0..128usize {
                    raw[p + 16 + i] = next_byte();
                }
            }
            let ggml_out = crate::model::gguf::dequant_q4_k(&raw, rows * cols).unwrap();
            let arf_out = Q4KSMatrix::from_ggml_q4k(&raw, rows, cols).to_f32();
            assert_bit_exact(&arf_out, &ggml_out, "multi_row_2x256");
        }

        // (e) All sub-blocks with high (j>=4) scale path — the (scales[j+4]&0xF)|... branch.
        {
            let mut scales = [0u8; 12];
            // Set bits in both the lower and upper halves so j>=4 path exercises bit-OR.
            // j=4: sc = (scales[8] & 0xF) | ((scales[0] >> 6) << 4)
            //       m  = (scales[8] >> 4)  | ((scales[4] >> 6) << 4)
            scales[0] = 0b1100_0101; // bits[7:6]=0b11 → contributes 0b11<<4=48 to sc for j=4
            scales[4] = 0b0100_0111; // bits[7:6]=0b01 → contributes 16 to m for j=4
            scales[8] = 0b1010_0011; // low nibble=3 → sc base; high nibble=10 → m base
                                     // sc = (3) | (3<<4) = 3|48 = 51; m = (10) | (1<<4) = 10|16 = 26
            let mut qs = [0u8; 128];
            qs[64] = 0x0F; // sub-block 4: half=2, qbase=64, shift=0 → first nibble=15
            let block = make_q4k_block(0.25, 0.125, scales, qs);
            let rows = 1;
            let cols = 256;
            let ggml_out = crate::model::gguf::dequant_q4_k(&block, rows * cols).unwrap();
            let arf_out = Q4KSMatrix::from_ggml_q4k(&block, rows, cols).to_f32();
            assert_bit_exact(&arf_out, &ggml_out, "high_scale_path_j4");
        }
    }
}

#[cfg(test)]
mod iq4_tests {
    use super::*;

    /// IQ4_XS, hand-computed from the ggml layout. d=1.0 (f16 0x3C00),
    /// scales_h=0 and scales_l all zero → every sub-block scale is (0 - 32) = -32,
    /// so weight = 1.0 * -32 * kvalues[q]. With qs bytes 0x10 the first weight
    /// takes nibble 0 (kvalues[0] = -127) and the 17th takes nibble 1
    /// (kvalues[1] = -104).
    #[test]
    fn dequant_iq4_xs_known_block() {
        let mut b = vec![0u8; 136];
        // d = 1.0
        b[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        b[8] = 0x10; // qs[0]: low nibble 0, high nibble 1
        let out = dequant_iq4_xs(&b, 256).unwrap();
        assert_eq!(out[0], -32.0 * -127.0);
        assert_eq!(out[16], -32.0 * -104.0);
        // qs[1..] are zero → nibble 0 → kvalues[0]
        assert_eq!(out[1], -32.0 * -127.0);
    }

    /// THE -32 SCALE BIAS, pinned by construction: a sub-scale of exactly 32
    /// gives dl = 0, so the sub-block decodes to ZEROS regardless of its codes.
    /// The existing `dequant_iq4_xs_known_block` uses ls=0 (dl = -32) and so
    /// would still pass if the bias were dropped and the scale read unsigned.
    #[test]
    fn iq4_xs_sub_scale_32_is_the_zero_point() {
        let mut b = vec![0u8; 136];
        // d = 1.0
        b[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        // ls = 32 for all 8 sub-blocks: low nibble 0, high bits 0b10.
        for j in 0..8 {
            b[2 + j / 8] = 0;
        }
        let scales_h: u16 = (0..8).map(|j| 2u16 << (2 * j)).sum();
        b[2..4].copy_from_slice(&scales_h.to_le_bytes());
        for q in b[8..136].iter_mut() {
            *q = 0xff; // every code 15 (+113) — a missing bias would be loud
        }
        let out = dequant_iq4_xs(&b, 256).unwrap();
        assert!(
            out.iter().all(|&x| x == 0.0),
            "ls=32 must give dl=0 (the -32 bias); got nonzero"
        );
    }

    /// EACH SUB-BLOCK USES ITS OWN SCALE. A reader that applies sub-scale 0 to
    /// the whole super-block passes every single-scale test and fails this one.
    #[test]
    fn iq4_xs_each_sub_block_uses_its_own_scale() {
        let mut b = vec![0u8; 136];
        // d = 1.0
        b[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        // ls = 33..40  ->  dl = 1..8
        let mut scales_h = 0u16;
        for j in 0..8usize {
            let ls = 33u8 + j as u8;
            b[4 + j / 2] |= (ls & 0xf) << (4 * (j % 2));
            scales_h |= ((ls >> 4) as u16 & 3) << (2 * j);
        }
        b[2..4].copy_from_slice(&scales_h.to_le_bytes());
        for q in b[8..136].iter_mut() {
            *q = 0x0f; // low nibble code 15 -> +113
        }
        let out = dequant_iq4_xs(&b, 256).unwrap();
        for j in 0..8 {
            let want = (j as f32 + 1.0) * 113.0;
            assert_eq!(
                out[j * 32],
                want,
                "sub-block {j} must use its OWN scale, not sub-block 0's"
            );
        }
    }

    /// The codebook is NON-LINEAR. Reading the nibble as a signed integer
    /// decodes every IQ4 tensor wrong — plausibly, not loudly.
    #[test]
    fn iq4_xs_codebook_is_not_a_linear_ramp() {
        let linear: Vec<i8> = (0..16).map(|c| (c as i8 - 8) * 16).collect();
        assert_ne!(KVALUES_IQ4NL.to_vec(), linear);
        assert_eq!(KVALUES_IQ4NL[0], -127);
        assert_eq!(KVALUES_IQ4NL[15], 113);
    }

    /// IQ4_NL: 32 weights, one f16 scale, no sub-block scales.
    #[test]
    fn dequant_iq4_nl_known_block() {
        let mut b = vec![0u8; 18];
        b[0..2].copy_from_slice(&0x4000u16.to_le_bytes()); // d = 2.0
        b[2] = 0xF0; // low nibble 0 -> kvalues[0]=-127; high nibble 15 -> 113
        let out = dequant_iq4_nl(&b, 32).unwrap();
        assert_eq!(out[0], 2.0 * -127.0);
        assert_eq!(out[16], 2.0 * 113.0);
    }

    /// The scale is 6 bits split across two arrays and biased by -32. Pin that
    /// assembly: low nibble 5 in scales_l[0], high bits 2 in scales_h → 0x25 = 37,
    /// minus 32 = 5.
    #[test]
    fn iq4_xs_scale_is_six_bits_biased() {
        let mut b = vec![0u8; 136];
        // d = 1.0
        b[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        b[2..4].copy_from_slice(&0x0002u16.to_le_bytes()); // scales_h: sub-block 0 hi = 2
        b[4] = 0x05; // scales_l[0]: sub-block 0 lo = 5
        let out = dequant_iq4_xs(&b, 256).unwrap();
        assert_eq!(out[0], 5.0 * -127.0);
    }

    /// Q3_K, hand-computed. d=1.0; scales all zero → every sub-block scale is
    /// (0 - 32) = -32. qs all zero → low 2 bits are 0. hmask all zero → the high
    /// bit is CLEAR, which in ggml means SUBTRACT 4 (the bit is inverted). So
    /// every weight is -32 * (0 - 4) = 128.
    #[test]
    fn dequant_q3_k_known_block() {
        let mut b = vec![0u8; 110];
        b[108..110].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0 (last field!)
        let out = dequant_q3_k(&b, 256).unwrap();
        assert_eq!(out[0], 128.0);
        assert_eq!(out[255], 128.0);
    }

    /// Setting a hmask bit CLEARS the -4 subtraction for that weight.
    #[test]
    fn q3_k_hmask_bit_is_inverted() {
        let mut b = vec![0u8; 110];
        b[108..110].copy_from_slice(&0x3C00u16.to_le_bytes());
        b[0] = 0x01; // weight 0: high bit set -> no subtraction -> -32 * 0 = 0
        let out = dequant_q3_k(&b, 256).unwrap();
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 128.0); // neighbour unaffected
    }
}
