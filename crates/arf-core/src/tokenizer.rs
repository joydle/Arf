//! Thin wrapper over the HuggingFace `tokenizers` crate.

use std::path::Path;

use crate::error::{ArfError, Result};
use crate::model::gguf_tokenizer::{build_tokenizer, GgufTokenizerMeta};

/// A loaded tokenizer (e.g. from a `tokenizer.json`).
#[derive(Debug, Clone)]
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl Tokenizer {
    /// Build from a GGUF's embedded tokenizer metadata (gpt2 byte-level BPE).
    pub fn from_gguf_meta(meta: &GgufTokenizerMeta) -> Result<Self> {
        Ok(Tokenizer {
            inner: build_tokenizer(meta)?,
        })
    }

    /// Open a GGUF and build its embedded tokenizer. Errors (with a "pass
    /// --tokenizer" hint) if the GGUF has no usable gpt2-BPE tokenizer.
    pub fn from_gguf_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let meta = crate::model::gguf::read_tokenizer_meta(path.as_ref())?;
        Self::from_gguf_meta(&meta)
    }

    /// Load from a `tokenizer.json` file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| ArfError::Tokenizer(e.to_string()))?;
        Ok(Tokenizer { inner })
    }

    /// Encode `text` into token ids.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| ArfError::Tokenizer(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode token ids back into text.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| ArfError::Tokenizer(e.to_string()))
    }

    /// Number of tokens in the vocabulary (including added tokens).
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// The id of a literal token string (e.g. `"<end_of_turn>"`, `"<eos>"`), or
    /// `None` if it isn't in the vocabulary. Used to resolve stop tokens for chat
    /// generation without hardcoding numeric ids per model.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
}
