//! Model and engine configuration.
//!
//! [`ModelConfig`] describes the transformer architecture (mirrors the relevant
//! fields of a HuggingFace `config.json`). [`EngineConfig`] describes runtime
//! knobs — block size, number of KV blocks, batch limits.

use crate::error::{ArfError, Result};

/// RoPE frequency scaling. Llama 3.x uses the `llama3` scheme; older models use
/// none. We keep the fields explicit so a loaded `config.json` maps cleanly.
#[derive(Debug, Clone, PartialEq)]
pub enum RopeScaling {
    /// No scaling: frequencies are `theta^(-2i/d)`.
    None,
    /// The Llama 3 piecewise wavelength scaling.
    Llama3 {
        factor: f64,
        low_freq_factor: f64,
        high_freq_factor: f64,
        original_max_position: usize,
    },
}

impl RopeScaling {
    /// The scaling used by Llama 3.2 1B/3B Instruct.
    pub fn llama3_default() -> Self {
        RopeScaling::Llama3 {
            factor: 32.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position: 8192,
        }
    }
}

/// The FFN block kind. Llama/Gemma are dense SwiGLU; Qwen3-MoE routes each token to
/// a few of many experts (+ optional always-on shared experts).
#[derive(Debug, Clone, PartialEq)]
pub enum MlpKind {
    /// Dense SwiGLU (`down(silu(gate(x)) * up(x))`), `intermediate_size` wide.
    Dense,
    /// Mixture-of-experts: a router picks `top_k` of `num_experts`, each a SwiGLU of
    /// width `moe_intermediate`, plus `shared_experts` always-on experts.
    Moe {
        num_experts: usize,
        top_k: usize,
        shared_experts: usize,
        moe_intermediate: usize,
        /// Renormalize the top-k router weights to sum to 1 (Qwen3 `norm_topk_prob`).
        norm_topk: bool,
    },
}

/// The FFN gate activation. Llama/Qwen use SwiGLU (SiLU gate); Gemma uses GeGLU
/// (the `gelu_pytorch_tanh` gate). Applied to the gate projection before the
/// elementwise multiply with the up projection: `out = act(gate) * up`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GateAct {
    /// SiLU: `x · σ(x)`. Llama, Qwen3.
    Silu,
    /// GELU-tanh approximation (`gelu_pytorch_tanh`): Gemma 2/3/4.
    GeluTanh,
}

/// RMSNorm placement + formula. Llama: 2 pre-norms, `w·x`. Gemma: 4 norms
/// (pre+post around attn and mlp), `(1+w)·x`, embedding scaled by √hidden.
/// Both are live.
#[derive(Debug, Clone, PartialEq)]
pub enum NormStyle {
    Llama,
    Gemma,
}

/// Attention masking. Llama/Qwen are fully causal; Gemma 3/4 interleave local
/// (sliding-window) and global layers with two RoPE thetas.
///
/// The base `ModelConfig.head_dim` / `num_kv_heads` describe the **sliding
/// (local)** layers. Gemma 3 uses the *same* geometry on its global layers, but
/// Gemma 4 gives the global layers a *different* head_dim / KV-head count / RoPE
/// rotary fraction — so `HybridLocalGlobal` carries that global-layer profile
/// alongside the windowing params. For Gemma 3 the global fields are set equal to
/// the sliding geometry (`partial_rotary_factor = 1.0`), so nothing changes.
#[derive(Debug, Clone, PartialEq)]
pub enum AttnKind {
    Causal,
    /// HYBRID SSM + ATTENTION (qwen35 family: Qwen3.5/3.6/3.8-27B). Full causal attention on
    /// every `attn_every`-th layer; GATED-DELTA-NET on all the others. Read straight from the
    /// GGUF: `full_attention_interval = 4`, and the tensor table confirms it — attention layers
    /// (3, 7, 11, ...) carry attn_q/k/v + q_norm/k_norm + attn_output (11 tensors), while GDN
    /// layers carry a FUSED attn_qkv + the seven ssm_* + attn_gate (14 tensors).
    ///
    /// This is NOT HybridLocalGlobal: that one switches between two flavours of ATTENTION
    /// (sliding vs full). Here 3 of every 4 layers have no attention at all — which is exactly
    /// why the architecture is cheap: SSM layers are O(1) in context length.
    HybridSsmAttn {
        /// Every Nth layer is full attention; the rest are GDN. `4` for qwen35.
        attn_every: usize,
        /// GDN inner width (`ssm.inner_size`, 6144 for qwen35).
        ssm_inner: usize,
        /// GDN recurrent state width (`ssm.state_size`, 128).
        ssm_state: usize,
        /// Depthwise causal conv width (`ssm.conv_kernel`, 4).
        conv_kernel: usize,
        /// Rank of the delta/beta projections (`ssm.time_step_rank`, 48).
        dt_rank: usize,
        /// Number of GDN groups (`ssm.group_count`, 16).
        groups: usize,
    },
    HybridLocalGlobal {
        window: usize,
        /// e.g. 6 → every 6th layer is global, the rest local (Gemma 3 = 5 local:1 global).
        global_every: usize,
        local_theta: f64,
        global_theta: f64,
        /// Head dimension of the GLOBAL (full-attention) layers. For Gemma 3 this
        /// equals the base (sliding) `head_dim`; Gemma 4 uses 512 vs 256 sliding.
        global_head_dim: usize,
        /// Number of KV heads on the GLOBAL layers. For Gemma 3 == base
        /// `num_kv_heads`; Gemma 4 uses 4 vs 16 sliding.
        global_kv_heads: usize,
        /// Fraction of each GLOBAL-layer head that gets rotated by RoPE (the rest
        /// pass through un-rotated). Gemma 4 global = 0.25 (only the first ¼ of the
        /// 512-dim head rotates). `1.0` = full rotation (Gemma 3 / sliding layers).
        partial_rotary_factor: f32,
    },
}

/// Transformer architecture parameters. The base fields mirror a HuggingFace
/// `config.json`; the `mlp`/`norm`/`attn`/`qk_norm`/scaling fields make it
/// multi-architecture (Llama, Qwen3-MoE, Gemma 3/4).
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    /// L240 — trained multi-token-prediction (MTP / "nextn") draft layers that sit AFTER the
    /// trunk. Qwen3.8-27B ships exactly one (`blk.64`, `nextn_predict_layers=1`): a full
    /// attention+FFN block plus eh_proj/enorm/hnorm/shared_head_norm. It is NOT part of the
    /// forward pass — the model is correct without it — but it is a 100%-hit-rate drafter,
    /// which is the one thing `SuffixDrafter` cannot be (L238: it proposes on 12% of steps).
    /// 0 = no head / do not load it.
    pub nextn_layers: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rope_scaling: RopeScaling,
    pub max_position_embeddings: usize,
    /// Llama 1B/3B tie the LM head to the input embedding matrix.
    pub tie_word_embeddings: bool,
    /// FFN kind: dense (Llama/Gemma) or MoE (Qwen3). Default `Dense`.
    pub mlp: MlpKind,
    /// Per-head RMSNorm on q,k before RoPE (Qwen3 + Gemma have it; Llama doesn't).
    pub qk_norm: bool,
    /// RMSNorm placement/formula. Default `Llama`.
    pub norm_style: NormStyle,
    /// Attention masking. Default `Causal`.
    pub attn: AttnKind,
    /// Scale embeddings by this after lookup (Gemma uses √hidden). `None` = no scale.
    pub embedding_scale: Option<f32>,
    /// Custom pre-softmax query scale (Gemma `query_pre_attn_scalar`); `None` =
    /// the default `1/√head_dim`.
    pub query_pre_attn_scalar: Option<f32>,
    /// Final-logit soft-capping (Gemma 4 `final_logit_softcapping = 30.0`): after
    /// the lm_head projection, apply `cap · tanh(logit / cap)` element-wise to the
    /// vocab logits. `None` = no capping (Gemma 3 / Llama / Qwen).
    pub final_logit_softcap: Option<f32>,
    /// Gemma 4 applies a per-head WEIGHTLESS RMSNorm to the value projection (V),
    /// exactly like its Q/K norms but with no learned gain — llama.cpp's
    /// `Vcur_normed = RMS_NORM(Vcur)`. Q/K/V are still INDEPENDENT projections, except
    /// Gemma 4's GLOBAL layers ship NO `attn_v.weight` and reuse K as V (the loaders
    /// clone K when the V tensor is absent). This flag only adds the V normalize.
    /// `false` for Gemma 3 / Llama / Qwen (no V-norm).
    pub value_norm: bool,
    /// FFN gate activation. `Silu` (SwiGLU) for Llama/Qwen; `GeluTanh` (GeGLU,
    /// `gelu_pytorch_tanh`) for Gemma — Gemma's FFN is GELU-gated, NOT SiLU.
    pub gate_act: GateAct,
}

/// Can the Metal island's decode attention kernels address this `head_dim`?
///
/// 🔴 ONE RULE, ONE PLACE. The island's decode attention kernels load KV as 32 lanes ×
/// float4/half4 and state the requirement in their own headers ("Requires hd % 128 == 0",
/// `attention_msl.metal`, and the same for `attention_decode_coalesced_b_f16`). A `head_dim` that
/// violates it reads past the end of a KV row and the engine emits `!!!!` AT FULL SPEED with no
/// error — llama-3.2-1B (head_dim 64) ships exactly that way, because the host guards for the
/// rule are spread across `island.rs` and `selftest.rs` and none of them runs early enough to keep
/// the island from being enabled in the first place.
///
/// Those scattered sites each re-spell `hd.is_multiple_of(128)` inline. Callers may keep their own
/// PHRASING — each reports in its own voice, at its own stage — but must not keep their own RULE,
/// because a rule that decides whether a model emits text or exclamation marks is one that gets
/// fixed in one copy and missed in the others. That is precisely the llama-3.2-1B shape.
///
/// The kernels are the source of truth: change this only alongside them.
pub const fn island_head_dim_supported(head_dim: usize) -> bool {
    head_dim.is_multiple_of(128) && head_dim <= 512
}

impl ModelConfig {
    /// Does each block carry FOUR norms (pre+post around attention and MLP) rather than Llama's
    /// two? This is a question about norm PLACEMENT, and it is deliberately separate from
    /// `norm_style`, which is the RMSNorm FORMULA (`w·x` vs Gemma's `(1+w)·x`).
    ///
    /// Those two were the same question while Gemma was the only four-norm model, so the code
    /// spelled both `is_gemma`. Muse Glimmer separates them: its GGUF ships
    /// `post_attention_norm` + `post_ffw_norm` (four-norm placement) but plain `w·x` Llama norms.
    /// Ask this when dispatching the post-norms; keep using `norm_style` for the formula.
    pub fn has_post_norms(&self) -> bool {
        // Gemma 3/4, and Muse Glimmer — which pairs the four-norm PLACEMENT with plain Llama
        // `w·x` norms, the case this predicate exists to express. Identified by its geometry
        // (52 layers x hidden 6656) rather than a new enum variant: adding NormStyle::MuseGlimmer
        // would force every `norm_style` match into a 3-way and re-conflate placement with
        // formula, which is the thing L152 just separated.
        self.norm_style == NormStyle::Gemma || self.is_muse_glimmer()
    }

    /// Does this model's GGUF store q/k in INTERLEAVED (ggml "NORM") rope order rather than
    /// split-half ("NEOX")? llama.cpp decides this per arch (src/llama-model.cpp:2624); every
    /// model we ship is NEOX except muse-glimmer, whose converter permutes q/k on the way in
    /// (conversion/muse_glimmer.py:14-16). Applying the wrong pairing scrambles every q and k
    /// without erroring, so this must be explicit per arch, never inferred.
    pub fn rope_interleaved(&self) -> bool {
        self.is_muse_glimmer()
    }

    /// Muse Glimmer's distinguishing geometry. Used only to select block STRUCTURE (four-norm
    /// placement, gated attention) — never math, which comes from the fields.
    pub fn is_muse_glimmer(&self) -> bool {
        self.num_layers == 52 && self.hidden_size == 6656 && self.num_kv_heads == 2
    }

    /// The first `head_dim` in this model the Metal island's decode kernels cannot address, if
    /// any. `None` means every attention layer is addressable and the island may be enabled.
    ///
    /// Checks EVERY layer rather than `self.head_dim`, because Gemma 4's global layers use 512
    /// where its sliding layers use 256 — a model is only eligible if all of them are.
    pub fn island_unsupported_head_dim(&self) -> Option<usize> {
        (0..self.num_layers)
            .map(|i| self.layer_geometry(i).0)
            .find(|&hd| !island_head_dim_supported(hd))
    }

    /// The exact configuration of `meta-llama/Llama-3.2-1B-Instruct`.
    pub fn llama_3_2_1b() -> Self {
        ModelConfig {
            nextn_layers: 0,
            vocab_size: 128_256,
            hidden_size: 2048,
            intermediate_size: 8192,
            num_layers: 16,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dim: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            rope_scaling: RopeScaling::llama3_default(),
            max_position_embeddings: 131_072,
            tie_word_embeddings: true,
            mlp: MlpKind::Dense,
            qk_norm: false,
            norm_style: NormStyle::Llama,
            attn: AttnKind::Causal,
            embedding_scale: None,
            query_pre_attn_scalar: None,
            final_logit_softcap: None,
            value_norm: false,
            gate_act: GateAct::Silu,
        }
    }

    /// `Qwen/Qwen3-Coder-30B-A3B-Instruct` (MoE: 128 experts, top-8, no shared;
    /// QK-norm; GQA causal). The ollama `qwen3-coder:30b` target. Values verified
    /// against the GGUF metadata: head_dim 128 (q packs to 32·128=4096 ≠ hidden),
    /// rope_theta 1e7, expert_ff 768, shared_ff 0, eps 1e-6.
    pub fn qwen3_coder_30b() -> Self {
        ModelConfig {
            nextn_layers: 0,
            vocab_size: 151_936,
            hidden_size: 2048,
            intermediate_size: 768, // unused for MoE; kept = moe_intermediate for sanity
            num_layers: 48,
            num_attention_heads: 32,
            num_kv_heads: 4,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000_000.0,
            rope_scaling: RopeScaling::None,
            max_position_embeddings: 262_144,
            // Qwen3-30B-A3B ships a dedicated `output.weight` lm_head (Q6_K),
            // distinct from `token_embd.weight` — it is NOT tied. (Only the small
            // dense Qwen3 0.6B/1.7B/4B tie.) Treating it as tied projects the final
            // hidden state with the embedding matrix → garbage logits.
            tie_word_embeddings: false,
            mlp: MlpKind::Moe {
                num_experts: 128,
                // ⚠️ L28 MEASURED THIS AND REJECTED LOWERING IT — do not "optimize" it.
                // Lowering top_k DOES speed up every tier (8→6 = +8.3% at conc16, +7.6-12.3%
                // across the ladder) because it cuts the active-expert count and hence the
                // expert-weight traffic. It is not free: the measured perplexity curve is
                //     top_k  8: 16.3445 (shipped)   7: 16.7532 (+2.50%)
                //            6: 17.4256 (+6.61%)    5: 18.7904 (+14.96%)   4: 22.9456 (+40.39%)
                // i.e. ~1.26% throughput per 1% perplexity — a dial between two products, not
                // an engineering win. The temporary env knob that produced this curve was
                // DELETED per the env-surface ratchet (the lever lost; the curve is
                // published). Re-adding it re-runs a closed experiment.
                // (Measured 2026-08-05.)
                top_k: 8,
                shared_experts: 0,
                moe_intermediate: 768,
                norm_topk: true,
            },
            qk_norm: true,
            norm_style: NormStyle::Llama,
            attn: AttnKind::Causal,
            embedding_scale: None,
            query_pre_attn_scalar: None,
            final_logit_softcap: None,
            value_norm: false,
            gate_act: GateAct::Silu,
        }
    }

    /// `Qwen/Qwen3-Omni-30B-A3B-Instruct`'s THINKER (M3 of the port,
    /// 2026-09-27), as llama.cpp converts it: GGUF arch `qwen3vlmoe`, `qwen3moe` tensor names.
    /// Field for field the Coder's text shape (48 layers, 2048 hidden, 32 q / 4 kv x 128, QK-norm,
    /// 128 experts top-8 of width 768, no shared expert, renormalised top-k, untied lm_head);
    /// what differs is read from `ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF`'s Q4_K_M header:
    /// vocab 152,064 (`token_embd` / `output` rows), `rope.freq_base` 1e6 (the Coder's is 1e7),
    /// `context_length` 65,536, eps 1e-6. A server started on that GGUF takes these values
    /// from the file itself ([`qwen3moe_family_from_gguf`]); this built-in exists so `--arch
    /// qwen3-omni` resolves where no header is at hand, and a test pins the two to each other.
    ///
    /// M-RoPE: the file declares `rope.dimension_sections [24, 20, 20, 0]`, and this config
    /// carries NO sections on purpose. For text and audio every position triple is `(p, p, p)`
    /// (HF `get_rope_index`: an audio span is `arange(n)` on all three axes), and with equal axes
    /// every pair rotates at `p` by its own frequency, which IS the plain NEOX rope this path runs.
    /// Only image and video rows make the axes differ; that is M7, and a `qwen3vl_merger`
    /// projector is refused for this model until then (`arf-serve`'s `load_qwen_vision`).
    pub fn qwen3_omni_30b() -> Self {
        ModelConfig {
            vocab_size: 152_064,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 65_536,
            ..Self::qwen3_coder_30b()
        }
    }

    /// `Qwen3.6-27B` (`qwen35` arch) — the HYBRID gated-delta-net / full-GQA model. Ground-
    /// truthed against `Qwen3.6-27B-Q4_K_S.gguf`: hidden 5120, 65 trunk blocks (1 nextn/MTP),
    /// 24 q / 4 kv heads, key/value_length 256, ffn 17408, vocab 248320, eps 1e-6, rope_theta
    /// 1e7, NOT tied (dedicated `output.weight`). DENSE SwiGLU FFN (NO experts in the file).
    ///
    /// NOTE: this `ModelConfig` only describes the standard-transformer face of the model; the
    /// HYBRID structure (3/4 layers are linear gated-delta-net) + the SSM hparams are read from
    /// the GGUF metadata by [`crate`]'s `ssm_qwen35::load_qwen35`, NOT from here. The standard
    /// GQA decode path can NOT run this model as-is (the linear layers need new kernels, M2-M4)
    /// — this config exists so `--arch qwen35` resolves and the loader can be driven.
    pub fn qwen35_27b() -> Self {
        ModelConfig {
            // blk.64 — measured present in Qwen3.8-27B-Q4_K_M.gguf with
            // qwen35.nextn_predict_layers = 1.
            nextn_layers: 1,
            vocab_size: 248_320,
            hidden_size: 5120,
            intermediate_size: 17_408,
            // 64 TRUNK layers, not 65. The GGUF says block_count=65 because blk.64 is the MTP
            // DRAFT LAYER: it is a full attention block PLUS the four nextn.* tensors
            // (eh_proj / enorm / hnorm / shared_head_norm) and nextn_predict_layers=1.
            // Measured: blk.63 has 11 tensors, blk.64 has those same 11 plus the four nextn ones.
            // Loading it as trunk layer 65 both breaks the 4-layer attention schedule and runs a
            // draft head as if it were part of the main forward pass.
            num_layers: 64,
            num_attention_heads: 24,
            num_kv_heads: 4,
            head_dim: 256, // key_length/value_length; full-attn q packs 2× (q_dim = 256·24·2)
            rms_norm_eps: 1e-6,
            rope_theta: 10_000_000.0,
            rope_scaling: RopeScaling::None,
            max_position_embeddings: 262_144,
            tie_word_embeddings: false,
            mlp: MlpKind::Dense,
            qk_norm: true,
            norm_style: NormStyle::Llama,
            attn: AttnKind::HybridSsmAttn {
                // full_attention_interval=4 — attention on layers 3,7,11,...; GDN on the rest.
                attn_every: 4,
                ssm_inner: 6144,
                ssm_state: 128,
                conv_kernel: 4,
                dt_rank: 48,
                groups: 16,
            },
            embedding_scale: None,
            query_pre_attn_scalar: None,
            final_logit_softcap: None,
            value_norm: false,
            gate_act: GateAct::Silu,
        }
    }

    /// `meta-models/Muse-Glimmer-30B` (`muse-glimmer`) — DENSE 27.9B. Ground-truthed by reading
    /// the GGUF KV header (range request, no download): block_count 52, embedding_length 6656,
    /// feed_forward_length 19968, head_count 32, head_count_kv 2, key/value_length 128,
    /// rms_eps 1e-5, rope.freq_base 500000, context_length 131072.
    ///
    /// DENSE despite being 30B-class: 731 tensors / 52 layers = 14.0 per layer (dense shape), a
    /// dense param estimate at these dims gives 29.9B vs the file's 27.9B, and there is no
    /// `expert_count` key. So: plain SwiGLU, no expert routing.
    ///
    /// 🔴 **THIS CONFIG IS NOT YET RUNNABLE.** The tensor table (read from the partial download)
    /// shows two structures we do not implement:
    ///
    /// ```text
    /// blk.0.attn_q_norm.weight            [128]           <- qk_norm: CONFIRMED, verified
    /// blk.0.attn_k_norm.weight            [128]
    /// blk.0.attn_norm.weight              [6656]          <- FOUR norms per layer (Gemma shape),
    /// blk.0.post_attention_norm.weight    [6656]             not Llama's two
    /// blk.0.ffn_norm.weight               [6656]
    /// blk.0.post_ffw_norm.weight          [6656]
    /// blk.0.attn_gate.weight              [6656, 4096]    <- GATED ATTENTION OUTPUT: no arch we
    ///                                                        support has this tensor at all
    /// blk.0.attn_q.weight                 [6656, 4096]    q packs to 32*128, not hidden
    /// blk.0.attn_k/v.weight               [6656, 256]     2 kv heads * 128
    /// ```
    ///
    /// So the remaining work is real, not cosmetic: a four-norm Llama-style block (we have that
    /// shape only under `NormStyle::Gemma`, which also changes the RMSNorm formula to `(1+w)·x`
    /// — wrong here), and a new gated-attention path for `attn_gate`. Wiring `--arch
    /// muse-glimmer` without those produces wrong logits, not an error.
    ///
    /// Verified-good: dense SwiGLU, 16x GQA (our kernels index `head / group` uncapped), hidden
    /// 6656 (no hard-coded hidden size in the kernels; ATTN_MAXHD is per-model), dedicated
    /// lm_head. Text backbone only — the vision tower (`mmproj-*.gguf`) and the ATEM chat
    /// template are separate jobs.
    pub fn muse_glimmer_30b() -> Self {
        ModelConfig {
            nextn_layers: 0,
            vocab_size: 202_048,
            hidden_size: 6656,
            intermediate_size: 19_968,
            num_layers: 52,
            num_attention_heads: 32,
            num_kv_heads: 2,
            // 32 x 128 = 4096 != hidden 6656: the q/k/v projections do NOT pack to hidden, the
            // same shape Gemma has. `validate()` allows this for any model with qk_norm or Gemma
            // norms; see the note there for why this one also qualifies.
            head_dim: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            rope_scaling: RopeScaling::None,
            max_position_embeddings: 131_072,
            // 731 = 52*14 + 3, the +3 being token_embd / output_norm / output — a dedicated
            // lm_head, so not tied.
            tie_word_embeddings: false,
            mlp: MlpKind::Dense,
            // Synthesised at conversion, NOT learned: llama's converter emits
            // q_norm = full(head_dim, qk_scale_factor) and k_norm = ones(head_dim), so the
            // per-head RMSNorm absorbs the model's qk scale. Weight values carry it; we just
            // have to APPLY the norm (conversion/muse_glimmer.py:63-75).
            qk_norm: true,
            // `w·x`. The converter already folded the +1 into every layer norm
            // (`norm_shift` = 1.0 for *layernorm.weight, 0.0 for the final norm), exactly like
            // our fold_gemma_norm does for safetensors — so the GGUF weights are pre-shifted and
            // the Gemma FORMULA must NOT be applied again here.
            norm_style: NormStyle::Llama,
            // 🔴 3 SLIDING : 1 GLOBAL, window 2048. Read from the GGUF:
            //   attention.sliding_window          = 2048
            //   attention.sliding_window_pattern  = [1,1,1,0, 1,1,1,0, ...] x52
            // Running these as plain Causal lets every layer attend to context it must not see,
            // which is on its own enough to produce degenerate output. Geometry is uniform
            // across sliding and global layers (unlike Gemma 4), so the global_* fields mirror
            // the base and partial_rotary_factor stays 1.0.
            attn: AttnKind::HybridLocalGlobal {
                window: 2048,
                global_every: 4,
                local_theta: 500_000.0,
                global_theta: 500_000.0,
                global_head_dim: 128,
                global_kv_heads: 2,
                partial_rotary_factor: 1.0,
            },
            embedding_scale: None,
            query_pre_attn_scalar: None,
            // GGUF final_logit_softcapping = 20.0: `cap * tanh(logit/cap)` on the vocab logits.
            final_logit_softcap: Some(20.0),
            value_norm: false,
            gate_act: GateAct::Silu,
        }
    }

    /// `google/gemma-3-4b` text backbone (the ollama `gemma3:4b` GGUF target).
    /// Verified against the GGUF metadata: hidden 2560, 34 layers, 8 q / 4 kv
    /// heads, head_dim 256 (q packs to 8·256=2048 ≠ hidden), ffn 10240, vocab
    /// 262144, tied embeddings, QK-norm, four-norm Gemma blocks, sliding window
    /// 1024 (global every 6th layer), dual RoPE (local 10k / global 1M), eps 1e-6.
    /// rope_theta / local_theta / global_every / window are llama.cpp's gemma3
    /// defaults (absent from the GGUF metadata); query_pre_attn_scalar = head_dim
    /// so the default 1/√head_dim scale already matches.
    pub fn gemma3_4b() -> Self {
        ModelConfig {
            nextn_layers: 0,
            // L363n — 262_208, not 262_144: the ollama / HF `gemma-3-4b-it` GGUF's token_embd is
            // [262208, 2560] (Gemma 3's text vocab plus its 64 image/control ids), and the loader
            // refused the tensor shape with 262144 here. llama.cpp uses 262208 for gemma3.
            vocab_size: 262_208,
            hidden_size: 2560,
            intermediate_size: 10_240,
            num_layers: 34,
            num_attention_heads: 8,
            num_kv_heads: 4,
            head_dim: 256,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            rope_scaling: RopeScaling::None,
            max_position_embeddings: 131_072,
            tie_word_embeddings: true,
            mlp: MlpKind::Dense,
            qk_norm: true,
            norm_style: NormStyle::Gemma,
            attn: AttnKind::HybridLocalGlobal {
                window: 1024,
                global_every: 6,
                local_theta: 10_000.0,
                global_theta: 1_000_000.0,
                // Gemma 3: global layers share the sliding geometry, fully rotated.
                global_head_dim: 256,
                global_kv_heads: 4,
                partial_rotary_factor: 1.0,
            },
            embedding_scale: Some((2560.0f32).sqrt()),
            query_pre_attn_scalar: None,
            final_logit_softcap: None,
            value_norm: false,
            gate_act: GateAct::GeluTanh,
        }
    }

    /// `google/gemma-4-31b-it` text backbone (the multimodal Gemma-4-31B's text
    /// tower). Verified against `models/gemma-4-31b-it/config.json` (`text_config`):
    /// hidden 5376, 60 layers, 32 q-heads, vocab 262144, ffn 21504, eps 1e-6, tied
    /// embeddings, QK-norm, four-norm Gemma blocks. Three Gemma-4-only deltas vs
    /// Gemma 3: (1) final-logit soft-cap 30.0; (2) K==V shared projection; (3)
    /// per-layer-type geometry — SLIDING (local) layers are head_dim 256 / 16 KV /
    /// full rotary (the base fields), GLOBAL (full-attention) layers are head_dim
    /// 512 / 4 KV / partial_rotary 0.25, every 6th layer global. Dual RoPE: local
    /// θ 10k / global θ 1M, window 1024.
    pub fn gemma4_31b() -> Self {
        ModelConfig {
            nextn_layers: 0,
            vocab_size: 262_144,
            hidden_size: 5376,
            intermediate_size: 21_504,
            num_layers: 60,
            num_attention_heads: 32,
            // Base (sliding/local-layer) geometry: 16 KV heads, head_dim 256.
            num_kv_heads: 16,
            head_dim: 256,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            rope_scaling: RopeScaling::None,
            max_position_embeddings: 262_144,
            tie_word_embeddings: true,
            mlp: MlpKind::Dense,
            qk_norm: true,
            norm_style: NormStyle::Gemma,
            attn: AttnKind::HybridLocalGlobal {
                window: 1024,
                global_every: 6,
                local_theta: 10_000.0,
                global_theta: 1_000_000.0,
                // Gemma 4 global (full-attention) layers differ from sliding ones.
                global_head_dim: 512,
                global_kv_heads: 4,
                partial_rotary_factor: 0.25,
            },
            embedding_scale: Some((5376.0f32).sqrt()),
            // Gemma 4 attention scale is 1.0 — the per-head qk_norm already normalizes
            // Q/K, so no 1/√d pre-attention scaling (llama.cpp gemma4.cpp f_attention_scale
            // = 1.0f). See gemma4_12b() for the full derivation. (Verified end-to-end on
            // 12B; 31B shares the arch but is unverified here — 31B OOMs on this Mac.)
            query_pre_attn_scalar: Some(1.0),
            final_logit_softcap: Some(30.0),
            value_norm: true,
            gate_act: GateAct::GeluTanh,
        }
    }

    /// Gemma-4-12B (text). Same arch family as the 31B, smaller + its own per-layer
    /// geometry — all values read from the QAT GGUF metadata: hidden 3840, 48 layers,
    /// 16 q-heads, ffn 15360, vocab 262144, eps 1e-6, tied embeddings, window 1024,
    /// global every 6th. SLIDING layers: head_dim 256, 8 kv-heads, θ 10k, FULL rotary
    /// (rope dim_count_swa 256 == head_dim). GLOBAL layers: head_dim 512, 1 kv-head,
    /// θ 1M, FULL rotary (rope dimension_count 512 == 512, so partial_rotary_factor
    /// 1.0 — unlike the 31B's 0.25). Softcap 30, k_eq_v true.
    pub fn gemma4_12b() -> Self {
        ModelConfig {
            nextn_layers: 0,
            vocab_size: 262_144,
            hidden_size: 3840,
            intermediate_size: 15_360,
            num_layers: 48,
            num_attention_heads: 16,
            // Base = SLIDING geometry: 8 kv-heads, head_dim 256.
            num_kv_heads: 8,
            head_dim: 256,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            rope_scaling: RopeScaling::None,
            max_position_embeddings: 262_144,
            tie_word_embeddings: true,
            mlp: MlpKind::Dense,
            qk_norm: true,
            norm_style: NormStyle::Gemma,
            attn: AttnKind::HybridLocalGlobal {
                window: 1024,
                global_every: 6,
                local_theta: 10_000.0,
                global_theta: 1_000_000.0,
                // 12B global layers: head_dim 512, 1 kv-head, FULL rotary (factor 1.0).
                global_head_dim: 512,
                global_kv_heads: 1,
                partial_rotary_factor: 1.0,
            },
            embedding_scale: Some((3840.0f32).sqrt()),
            // Gemma 4 attention scale is 1.0 — NOT 1/√query_pre_attn_scalar. The
            // per-head qk_norm (RMS_NORM + learned weight on Q and K, applied before
            // RoPE) already normalizes the queries/keys, so Gemma 4 sets
            // `self.scaling = 1.0` (verified vs llama.cpp gemma4.cpp: f_attention_scale
            // = 1.0f, "Gemma4 uses self.scaling = 1.0 (no pre-attn scalar)"). The old
            // 1/√256 = 0.0625 shrank every score 16×, flattening the softmax so tokens
            // could not form the BOS attention sink → wrong attention from the 2nd
            // token on → garbage output. Single-token (BOS-only) was unaffected because
            // softmax over one key is [1.0] regardless of scale.
            query_pre_attn_scalar: Some(1.0),
            final_logit_softcap: Some(30.0),
            value_norm: true,
            gate_act: GateAct::GeluTanh,
        }
    }

    /// Number of query heads that share each KV head (the GQA group size).
    pub fn gqa_group_size(&self) -> usize {
        self.num_attention_heads / self.num_kv_heads
    }

    /// Per-layer attention geometry `(head_dim, kv_heads, rotary_dim)` for layer
    /// index `i`. The base `head_dim`/`num_kv_heads` describe the SLIDING (local)
    /// layers; an [`AttnKind::HybridLocalGlobal`] gives the GLOBAL (full-attention,
    /// every `global_every`-th) layers a distinct `global_head_dim`/`global_kv_heads`
    /// and partial RoPE (`rotary_dim = round(global_head_dim · partial_rotary_factor)`).
    /// `rotary_dim` is the count of leading head dims RoPE rotates (the rest pass
    /// through). For [`AttnKind::Causal`] and Gemma 3 (global == sliding,
    /// `partial_rotary_factor == 1.0`) every layer returns the base geometry with
    /// `rotary_dim == head_dim` (full rotation). The global predicate is
    /// `(i + 1) % global_every == 0` — the SAME one the loader uses for `window`.
    /// Rows in this layer's q projection. Usually `num_attention_heads * head_dim`, but
    /// qwen35's FULL-ATTENTION layers pack q at TWICE head_dim — measured from the file:
    /// `layers.3.self_attn.q_proj` is [12288, 5120] = 48 * 256 = 2 * 24 heads. On its GDN
    /// layers the "q" slot is the SSM input projection instead, width `ssm_inner`.
    pub fn layer_q_dim(&self, i: usize) -> usize {
        let (hd, _, _) = self.layer_geometry(i);
        match &self.attn {
            AttnKind::HybridSsmAttn {
                attn_every,
                ssm_inner,
                ..
            } => {
                if (i + 1).is_multiple_of(*attn_every.max(&1)) {
                    2 * self.num_attention_heads * hd // attention layer: q packs 2x
                } else {
                    *ssm_inner // GDN layer: the q slot is the SSM input projection
                }
            }
            _ => self.num_attention_heads * hd,
        }
    }

    pub fn layer_geometry(&self, i: usize) -> (usize, usize, usize) {
        match &self.attn {
            AttnKind::HybridLocalGlobal {
                global_every,
                global_head_dim,
                global_kv_heads,
                partial_rotary_factor,
                ..
            } if (i + 1).is_multiple_of(*global_every.max(&1)) => {
                // round() matches HF: rotary_dim = round(head_dim * factor).
                let rot = (*global_head_dim as f32 * partial_rotary_factor).round() as usize;
                // RoPE rotates pairs, so the rotary dim must be even.
                let rot = rot - (rot % 2);
                (*global_head_dim, *global_kv_heads, rot)
            }
            // qwen35 GDN layers: the q/k/v widths come from the SSM geometry, not the attention
            // head counts. The fused attn_qkv is [q ssm_inner | k groups*state | v groups*state],
            // so kv_heads is reported such that kv_heads * head_dim == groups * state — that is
            // what the loader's `l_kv_dim` needs, and it keeps every downstream size correct
            // without a second code path.
            AttnKind::HybridSsmAttn {
                attn_every,
                ssm_state,
                groups,
                ..
            } if !(i + 1).is_multiple_of(*attn_every.max(&1)) => (
                self.head_dim,
                (groups * ssm_state) / self.head_dim,
                self.head_dim,
            ),
            // L155 — qwen35 FULL-ATTENTION layers use PARTIAL RoPE. The GGUF declares
            // `qwen35.rope.dimension_count = 64` while `attention.key_length = 256`, so only the
            // leading 64 of each 256-wide head is rotated and the remaining 192 pass through
            // unchanged. Reporting `rotary_dim = head_dim` (the old `_` arm) rotated ALL 256 —
            // wrong angles on 3/4 of every head vector, on every attention layer.
            // (The GGUF also declares `rope.dimension_sections = [11,11,10,0]`, i.e. llama.cpp's
            // mRoPE `ggml_rope_multi` — 32 pairs = the same 64 dims. The sectioning matters for
            // multimodal position ids; for text-only decode all sections advance with the same
            // position, so a plain partial RoPE over 64 dims is equivalent. Revisit if vision
            // inputs are ever wired for this arch.)
            // (2026-09-27: they are. Text keeps this plain partial rope untouched; an IMAGE batch
            // carries `ForwardBatch::mrope_positions` and the island rotates it with
            // `rope_qk_b_mrope` over the same 64 dims — see arf_core::model::mrope.)
            AttnKind::HybridSsmAttn { .. } => (self.head_dim, self.num_kv_heads, 64),
            _ => (self.head_dim, self.num_kv_heads, self.head_dim),
        }
    }

    /// The maximum per-layer `head_dim` and `kv_dim` (`kv_heads · head_dim`) over
    /// all layers — the size the shared q/k/v decode scratch and the per-layer KV
    /// pool buffers must be allocated to so a layer of either type fits. For Gemma 4
    /// the sliding `kv_dim` (16·256 = 4096) exceeds the global (4·512 = 2048) while
    /// the global `head_dim` (512) exceeds the sliding (256), so both maxima matter.
    pub fn max_attn_dims(&self) -> (usize, usize) {
        let mut max_hd = self.head_dim;
        let mut max_kv = self.num_kv_heads * self.head_dim;
        if let AttnKind::HybridLocalGlobal {
            global_head_dim,
            global_kv_heads,
            ..
        } = &self.attn
        {
            max_hd = max_hd.max(*global_head_dim);
            max_kv = max_kv.max(global_kv_heads * global_head_dim);
        }
        (max_hd, max_kv)
    }

    /// L154 — the widest `q_proj` OUTPUT over all layers, in floats. **This is what a q scratch
    /// buffer must be sized by — not `num_attention_heads * max_attn_dims().0`.**
    ///
    /// Those two agree for every arch EXCEPT the hybrid SSM family, where an attention layer's q
    /// is packed 2× (`layer_q_dim`: `2 * num_attention_heads * hd`, ground-truthed against
    /// Qwen3.8's `blk.3.attn_q` and noted at weights.rs:1490). For Qwen3.8 that is 12288, while
    /// `nh * max_hd` is 6144 — so a q buffer sized the old way is **half** what the very first
    /// attention layer writes into it.
    ///
    /// That undersizing is a silent 2× heap overflow past the end of `q`, not a bounds error:
    /// the GEMV writes `n` rows because its dims say so. The GDN layers (q slot = `ssm_inner` =
    /// 6144) fit exactly, so layers 0-2 are clean and **layer 3 — the first full-attention layer
    /// — is where it first corrupts**, which is precisely where the NaN was bisected to.
    ///
    /// Returns floats per row (multiply by the row count for a byte size × 4).
    pub fn max_q_dim(&self) -> usize {
        let (max_hd, _) = self.max_attn_dims();
        let base = self.num_attention_heads * max_hd;
        match &self.attn {
            // Take the max over BOTH layer kinds: attention layers pack 2×, GDN layers use
            // ssm_inner in the same slot. Neither is always the larger.
            AttnKind::HybridSsmAttn { ssm_inner, .. } => (2 * base).max(*ssm_inner),
            _ => base,
        }
    }

    /// Validate internal consistency. Cheap; call after loading from JSON.
    pub fn validate(&self) -> Result<()> {
        if self.num_layers == 0 {
            return Err(ArfError::model_load(
                "num_layers is 0; a model needs at least one transformer layer",
            ));
        }
        if !self.num_attention_heads.is_multiple_of(self.num_kv_heads) {
            return Err(ArfError::model_load(format!(
                "num_attention_heads ({}) not divisible by num_kv_heads ({})",
                self.num_attention_heads, self.num_kv_heads
            )));
        }
        if !self.head_dim.is_multiple_of(2) {
            return Err(ArfError::model_load(format!(
                "head_dim ({}) must be even for RoPE",
                self.head_dim
            )));
        }
        // Plain Llama packs q heads to exactly hidden_size. Models with an explicit
        // head_dim — Gemma, and Qwen3 (which carries q/k norm, head_dim 128 ≠
        // hidden/heads) — break that, so only enforce it when neither applies.
        let explicit_head_dim = self.norm_style == NormStyle::Gemma || self.qk_norm;
        if !explicit_head_dim && self.head_dim * self.num_attention_heads != self.hidden_size {
            return Err(ArfError::model_load(format!(
                "head_dim*num_heads ({}) != hidden_size ({})",
                self.head_dim * self.num_attention_heads,
                self.hidden_size
            )));
        }
        Ok(())
    }
}

/// Weight quantization mode, chosen at load time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Quant {
    /// Keep weights bf16 — the default, lossless-to-checkpoint path.
    #[default]
    None,
    /// Per-row symmetric int8 for the projections and the lm_head.
    Int8,
    /// Q4_0-style block-32 4-bit for the projections and the lm_head — half the
    /// bytes of int8 on the bandwidth-bound matmuls (the path to beat ollama).
    Q4,
    /// Q4_K-lite: super-block-256, asymmetric per-32 scale+min 4-bit — ollama's
    /// Q4_K accuracy class at ~Q4_0 speed (the min-offset fits skewed weights better).
    Q4K,
    /// Q4_K_S: Q4_K with u8/i8 two-level sub-scales (vs Q4_K-lite's bf16) — ~5%
    /// fewer bytes (the ggml Q4_K_S idea), ~Q4_K accuracy, above the 4-bit cliff.
    Q4KS,
    /// Q3_K: 3-bit split-plane codes + Q4_K_S two-level scales — ~25% fewer code
    /// bytes than Q4 (the real decode-speed cut), at a 3-bit accuracy cost.
    Q3K,
}

impl Quant {
    /// Approximate resident bytes per weight parameter, scales included. Used by the
    /// load-time wired-memory guard to estimate a model's GPU footprint BEFORE any
    /// buffer is allocated — Metal weight buffers are WIRED (unswappable), so an
    /// oversized load doesn't degrade, it freezes the machine. Estimates only need to
    /// separate the ~4× spread between quant classes, not be exact.
    pub fn approx_bytes_per_param(self) -> f64 {
        match self {
            Quant::None => 2.0,                           // bf16
            Quant::Int8 => 1.06,                          // 1B code + per-row scale
            Quant::Q4 | Quant::Q4K | Quant::Q4KS => 0.58, // ~4.5 bits + scales
            Quant::Q3K => 0.44,
        }
    }

    /// The canonical lowercase name (matches the CLI flag value + `FromStr`).
    pub fn as_str(self) -> &'static str {
        match self {
            Quant::None => "none",
            Quant::Int8 => "int8",
            Quant::Q4 => "q4",
            Quant::Q4K => "q4k",
            Quant::Q4KS => "q4ks",
            Quant::Q3K => "q3k",
        }
    }
}

impl std::str::FromStr for Quant {
    type Err = String;
    /// Parse a `--quant` flag value. Accepts the canonical names plus the ggml-style
    /// underscored aliases (`q4_k`, `q4_k_s`, `q3_k`).
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        Ok(match s {
            "none" | "bf16" => Quant::None,
            "int8" => Quant::Int8,
            "q4" | "q4_0" => Quant::Q4,
            "q4k" | "q4_k" => Quant::Q4K,
            "q4ks" | "q4_k_s" => Quant::Q4KS,
            "q3k" | "q3_k" => Quant::Q3K,
            other => {
                return Err(format!(
                    "unknown quant '{other}' (none|int8|q4|q4k|q4ks|q3k)"
                ))
            }
        })
    }
}

impl std::fmt::Display for Quant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// KV-cache quantization mode, chosen at runtime — independent of the weight
/// [`Quant`]. `--quant q4 --kv-quant tq3` is a valid pairing: weights and KV are
/// different bottlenecks (matmul bandwidth vs cache bandwidth/footprint).
///
/// `None` is today's exact `f32` cache (the correctness oracle, like bf16 is for
/// weights). `Tq` is TurboQuant ([arXiv:2504.19874]): per-head-vector random
/// rotation + Lloyd-Max scalar quantization at `bits ∈ {2,3,4}`, optionally with
/// the 1-bit QJL inner-product residual (`qjl`, default off).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvQuant {
    /// Exact `f32` KV cache — the default and the parity oracle.
    #[default]
    None,
    /// TurboQuant at `bits` per channel; `qjl` enables the 1-bit residual stage.
    Tq { bits: u8, qjl: bool },
}

impl KvQuant {
    /// Bits per channel for a Tq mode (4 for the f32 oracle's sake of sizing).
    pub fn bits(&self) -> u8 {
        match self {
            KvQuant::None => 32,
            KvQuant::Tq { bits, .. } => *bits,
        }
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self, KvQuant::Tq { .. })
    }
}

/// Runtime configuration for the engine's memory and scheduling.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineConfig {
    /// Tokens per KV block (paged attention page size).
    pub block_size: usize,
    /// Total number of physical KV blocks in the pool (per layer).
    pub num_blocks: usize,
    /// Max sequences scheduled concurrently in a batch.
    pub max_batch_size: usize,
    /// Max tokens processed in a single prefill step (across the batch).
    pub max_prefill_tokens: usize,
    /// KV-cache quantization mode (default: exact `f32`).
    pub kv_quant: KvQuant,
    /// Automatic prefix caching : reuse the KV of block-aligned shared
    /// prompt prefixes across requests so a repeated system prompt is prefilled
    /// once. Off by default (the parity oracle is "every prompt recomputed"); the
    /// serve CLI / loadgen flip it on. Bit-identical outputs either way.
    pub enable_prefix_cache: bool,
    /// Positions past a sequence's end to keep backed by KV blocks, best effort, so a
    /// speculative verify window (up to 8 rows) is not cut at a 16-slot block boundary. 0 = off,
    /// the default: only a server that speculates sets it. See
    /// `BlockManager::set_decode_lookahead`.
    pub decode_lookahead_tokens: usize,
    /// The model has RECURRENT layers: a prefix-cache hit is only valid where the backend holds
    /// a snapshot of the recurrent state. With this on (and `enable_prefix_cache`), the
    /// scheduler ends one prefill chunk of every long-enough prompt on a block boundary and asks
    /// for a snapshot there (`SeqPlan::snapshot_key`), and only matches prefixes down to a
    /// boundary that has one (`SeqPlan::restore_key`). Off by default.
    pub state_snapshots: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            block_size: 16,
            num_blocks: 1024,
            max_batch_size: 32,
            max_prefill_tokens: 4096,
            kv_quant: KvQuant::None,
            enable_prefix_cache: false,
            decode_lookahead_tokens: 0,
            state_snapshots: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Quant;

    #[test]
    fn quant_parse_display_roundtrip() {
        for q in [
            Quant::None,
            Quant::Int8,
            Quant::Q4,
            Quant::Q4K,
            Quant::Q4KS,
            Quant::Q3K,
        ] {
            assert_eq!(q.as_str().parse::<Quant>().unwrap(), q);
            assert_eq!(q.to_string(), q.as_str());
        }
        // ggml-style aliases parse to the canonical variant.
        assert_eq!("q4_k".parse::<Quant>().unwrap(), Quant::Q4K);
        assert_eq!("q4_k_s".parse::<Quant>().unwrap(), Quant::Q4KS);
        assert!("bogus".parse::<Quant>().is_err());
    }
}

#[cfg(test)]
mod muse_glimmer_config_tests {
    use super::*;

    /// The predicates that select muse-glimmer's block STRUCTURE. If either of these is false the
    /// model silently runs as a two-norm Llama block and produces degenerate text, so pin them.
    #[test]
    fn muse_glimmer_selects_four_norm_placement() {
        let c = ModelConfig::muse_glimmer_30b();
        assert!(
            c.is_muse_glimmer(),
            "geometry predicate must match its own config"
        );
        assert!(c.has_post_norms(), "must take the four-norm placement path");
        assert!(c.qk_norm, "attn_q_norm/attn_k_norm are present in the GGUF");
        assert_eq!(
            c.norm_style,
            NormStyle::Llama,
            "formula is w*x, NOT Gemma's (1+w)*x"
        );
        // The non-square q projection that validate() must tolerate.
        assert_ne!(c.head_dim * c.num_attention_heads, c.hidden_size);
        assert!(c.validate().is_ok(), "config must validate");
        // Values read from the GGUF header — regressing any of these silently produces
        // degenerate text rather than an error, which is exactly how they were missed.
        assert_eq!(
            c.final_logit_softcap,
            Some(20.0),
            "GGUF final_logit_softcapping"
        );
        match c.attn {
            AttnKind::HybridLocalGlobal {
                window,
                global_every,
                ..
            } => {
                assert_eq!(window, 2048, "GGUF attention.sliding_window");
                // sliding_window_pattern = [1,1,1,0] repeating: 3 sliding : 1 global.
                assert_eq!(global_every, 4);
            }
            _ => panic!("muse-glimmer is 3-sliding:1-global, NOT plain Causal"),
        }
    }
}

/// The supported architectures, and every `--arch` name that selects one.
///
/// THIS IS THE ONE TABLE. It used to be two — `arf-cli/src/main.rs` and
/// `arf-serve/src/main.rs` each carried a `match` over arch names, and they had drifted: the
/// server accepted neither the qwen3.5/3.6/3.8 family nor Gemma-4, so a model that ran fine under
/// `arf generate` was rejected by `arf serve`. Both error messages also listed names their
/// own arms did not accept. One table, one error message, no drift.
///
/// Adding an architecture means adding one row here.
pub const ARCHITECTURES: &[ArchEntry] = &[
    ArchEntry {
        canonical: "qwen3-coder-30b",
        aliases: &["qwen3-coder", "qwen3moe"],
        note: "Qwen3-Coder-30B-A3B — MoE, top-k router",
    },
    ArchEntry {
        canonical: "qwen3-omni-30b",
        // `qwen3vlmoe` is llama.cpp's arch string for this file, but Qwen3-VL-30B-A3B shares it
        // with a different vocab and rope base; a GGUF's own header wins over this row
        // (`qwen3moe_family_from_gguf`), so the alias only matters without one.
        aliases: &["qwen3-omni", "qwen3vlmoe"],
        note: "Qwen3-Omni-30B-A3B Thinker — MoE, text + audio in (--audio-tower), text out",
    },
    ArchEntry {
        canonical: "muse-glimmer-30b",
        aliases: &["muse-glimmer"],
        note: "Muse-Glimmer-30B — dense SwiGLU, 3:1 sliding-window (2048) / global attention, gated attention output",
    },
    ArchEntry {
        canonical: "qwen3.8-27b",
        // One geometry serves the whole 3.5/3.6/3.8 27B family: they share the hybrid SSM+attention
        // block and the 5120/17408/65L/24q/4kv/256hd shape, verified field-by-field against the
        // Qwen3.8-27B GGUF header.
        aliases: &[
            "qwen35",
            "qwen3.5",
            "qwen3.6",
            "qwen3.6-27b",
            "qwen35-27b",
            "qwen3.8",
        ],
        note: "Qwen3.5/3.6/3.8-27B — 48 gated-delta-net + 16 full-attention layers",
    },
    ArchEntry {
        canonical: "gemma-4-31b",
        aliases: &["gemma4-31b", "gemma4:31b", "gemma-4-31b-it"],
        note: "Gemma 4 31B",
    },
    ArchEntry {
        canonical: "gemma-4-12b",
        aliases: &["gemma4-12b", "gemma4:12b", "gemma-4-12b-it"],
        note: "Gemma 4 12B",
    },
    ArchEntry {
        canonical: "gemma-3-4b",
        aliases: &["gemma3", "gemma3-4b", "gemma3:4b"],
        note: "Gemma 3 4B (vision-capable)",
    },
    ArchEntry {
        canonical: "llama-3.2-1b",
        aliases: &["llama3.2-1b"],
        note: "Llama 3.2 1B — the small correctness model",
    },
];

/// One row of [`ARCHITECTURES`].
pub struct ArchEntry {
    /// The name to print and document.
    pub canonical: &'static str,
    /// Other accepted spellings. Matching is case-insensitive with `_` normalised to `-`.
    pub aliases: &'static [&'static str],
    /// One line for `--help` and error messages.
    pub note: &'static str,
}

impl ArchEntry {
    /// Does `name` (already normalised) select this entry?
    fn matches(&self, name: &str) -> bool {
        self.canonical == name || self.aliases.contains(&name)
    }
}

/// The canonical [`ARCHITECTURES`] name an `--arch` spelling selects, if any.
pub fn canonical_arch(arch: &str) -> Option<&'static str> {
    let key = arch.to_ascii_lowercase().replace('_', "-");
    ARCHITECTURES
        .iter()
        .find(|e| e.matches(&key))
        .map(|e| e.canonical)
}

/// llama.cpp's `general.architecture` string for the built-in architectures a GGUF header can
/// select by itself (2026-10-07). Only pairs seen in a real file or already an alias above; an
/// architecture missing here still needs `--arch`.
const GGUF_ARCHITECTURES: &[(&str, &str)] = &[
    ("qwen35", "qwen3.8-27b"),
    ("gemma3", "gemma-3-4b"),
    ("llama", "llama-3.2-1b"),
    ("qwen3moe", "qwen3-coder-30b"),
];

/// The built-in architecture a GGUF's own header names, when it names exactly one: its
/// `general.architecture` family AND its geometry (layers, hidden size, attention heads) must be
/// a built-in config's. The family string alone does not pin the geometry — `qwen35` is every
/// Qwen3.5/3.8 size — so a file of a size we have no config for answers `None`, and the caller
/// asks for `--arch` as before. `block_count` includes the next-token prediction layers a file
/// carries (`nextn_predict_layers`: the 27B says 65 blocks, 1 of them the MTP head).
pub fn arch_from_gguf_header(
    family: &str,
    block_count: usize,
    nextn_layers: usize,
    hidden: usize,
    heads: usize,
) -> Option<&'static str> {
    let layers = block_count.checked_sub(nextn_layers)?;
    let mut hits = GGUF_ARCHITECTURES
        .iter()
        .filter(|(f, _)| *f == family)
        .filter_map(|(_, canonical)| {
            let cfg = config_for_arch(canonical).ok()?;
            (cfg.num_layers == layers
                && cfg.hidden_size == hidden
                && cfg.num_attention_heads == heads)
                .then_some(*canonical)
        });
    let first = hits.next()?;
    hits.next().is_none().then_some(first)
}

/// Why a GGUF's header selected no built-in architecture, for the error that asks for one: the
/// family and geometry it read, the built-in configs of that family with theirs, and whether
/// `--arch` can help at all — it cannot when the family has a config but not this size (the load
/// then refuses on the first tensor whose shape differs).
pub fn gguf_header_mismatch(g: &crate::model::gguf::LazyGguf) -> String {
    let family = g
        .get_metadata_string("general.architecture")
        .unwrap_or_else(|| "none".into());
    let n = |k: &str| {
        g.get_metadata_u32(&format!("{family}.{k}"))
            .map(|v| v as usize)
    };
    header_mismatch_text(
        &family,
        n("block_count"),
        n("nextn_predict_layers"),
        n("embedding_length"),
        n("attention.head_count"),
    )
}

/// [`gguf_header_mismatch`] on the values it reads.
fn header_mismatch_text(
    family: &str,
    blocks: Option<usize>,
    nextn: Option<usize>,
    hidden: Option<usize>,
    heads: Option<usize>,
) -> String {
    let num = |v: Option<usize>| v.map_or("?".into(), |v| v.to_string());
    let read = format!(
        "{} blocks{}, hidden {}, {} attention heads",
        num(blocks),
        nextn
            .filter(|&v| v > 0)
            .map_or(String::new(), |v| format!(" ({v} next-token)")),
        num(hidden),
        num(heads),
    );
    let same: Vec<String> = GGUF_ARCHITECTURES
        .iter()
        .filter(|(f, _)| *f == family)
        .filter_map(|(_, c)| {
            let cfg = config_for_arch(c).ok()?;
            Some(format!(
                "{c} ({} layers, hidden {}, {} heads)",
                cfg.num_layers, cfg.hidden_size, cfg.num_attention_heads
            ))
        })
        .collect();
    if same.is_empty() {
        let known = ARCHITECTURES
            .iter()
            .map(|e| e.canonical)
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "this GGUF's header says general.architecture = {family} ({read}), which no built-in \
             architecture is read from; if it is one of them, pass --arch (one of: {known})"
        )
    } else {
        format!(
            "this GGUF's header says general.architecture = {family} ({read}); the built-in \
             {family} architecture{} {} {}, so this size is not supported yet and --arch will \
             not start it",
            if same.len() > 1 { "s" } else { "" },
            if same.len() > 1 { "are" } else { "is" },
            same.join(", ")
        )
    }
}

/// [`arch_from_gguf_header`] for an open file.
pub fn arch_from_gguf(g: &crate::model::gguf::LazyGguf) -> Option<&'static str> {
    let family = g.get_metadata_string("general.architecture")?;
    let n = |k: &str| {
        g.get_metadata_u32(&format!("{family}.{k}"))
            .map(|v| v as usize)
    };
    arch_from_gguf_header(
        &family,
        n("block_count")?,
        n("nextn_predict_layers").unwrap_or(0),
        n("embedding_length")?,
        n("attention.head_count")?,
    )
}

/// Resolve an `--arch` name to its [`ModelConfig`], or `Err` with the supported list.
///
/// Callers must not keep their own table — see [`ARCHITECTURES`].
pub fn config_for_arch(arch: &str) -> std::result::Result<ModelConfig, String> {
    let key = arch.to_ascii_lowercase().replace('_', "-");
    let entry = ARCHITECTURES
        .iter()
        .find(|e| e.matches(&key))
        .ok_or_else(|| {
            // ONE LINE on purpose: this error travels through `Box<dyn Error>` out of `main`,
            // where Rust formats it with Debug — a multi-line message comes out as literal `\n`.
            let known = ARCHITECTURES
                .iter()
                .map(|e| e.canonical)
                .collect::<Vec<_>>()
                .join(", ");
            format!("unknown --arch {arch:?}. Supported: {known}")
        })?;
    Ok(match entry.canonical {
        "qwen3-coder-30b" => ModelConfig::qwen3_coder_30b(),
        "qwen3-omni-30b" => ModelConfig::qwen3_omni_30b(),
        "muse-glimmer-30b" => ModelConfig::muse_glimmer_30b(),
        "qwen3.8-27b" => ModelConfig::qwen35_27b(),
        "gemma-4-31b" => ModelConfig::gemma4_31b(),
        "gemma-4-12b" => ModelConfig::gemma4_12b(),
        "gemma-3-4b" => ModelConfig::gemma3_4b(),
        "llama-3.2-1b" => ModelConfig::llama_3_2_1b(),
        // ARCHITECTURES and this match are the same list; a row without an arm is a bug here,
        // not a user error.
        other => unreachable!("ARCHITECTURES lists {other} with no ModelConfig arm"),
    })
}

/// What a `qwen3moe`-family GGUF header says about its text model: the [`ModelConfig`] built from
/// the file's own keys, plus the two keys `ModelConfig` has no field for.
#[derive(Debug, Clone, PartialEq)]
pub struct GgufMoeHeader {
    /// `general.architecture`: `qwen3vlmoe` (Qwen3-Omni's Thinker, Qwen3-VL-MoE) or `qwen3moe`.
    pub arch: String,
    pub cfg: ModelConfig,
    /// `{arch}.rope.dimension_sections` (`[24, 20, 20, 0]` for Omni). Not used by the text and
    /// audio path, where every position triple is `(p, p, p)` (see [`ModelConfig::qwen3_omni_30b`]).
    pub mrope_sections: Option<[u32; 4]>,
    /// `{arch}.n_deepstack_layers` (0 in the Omni text GGUF; image input only).
    pub deepstack_layers: u32,
}

/// Build a `qwen3vlmoe` GGUF's config from its header alone (M3 of the port).
/// `Ok(None)` for any other architecture, so a caller can try it on every GGUF. The Coder's
/// `qwen3moe` files keep their `--arch` row (this returns `None` for them): nothing about how
/// they load changes.
///
/// Every geometry field is read, none defaulted: a missing key is an error naming it. The
/// structure the file does NOT spell is what llama.cpp's `qwen3vlmoe` graph hard-codes, and it
/// is Arf's `qwen3moe` shape: NEOX rope, QK-norm, SwiGLU experts with the router's top-k
/// renormalised (`norm_w = true`), Llama-placed RMSNorms. A shared expert, rope scaling or a
/// value length other than the key length is refused rather than silently ignored.
pub fn qwen3moe_family_from_gguf(
    g: &crate::model::gguf::LazyGguf,
) -> std::result::Result<Option<GgufMoeHeader>, String> {
    let arch = g.get_metadata_string("general.architecture");
    if arch.as_deref() != Some("qwen3vlmoe") {
        return Ok(None);
    }
    let arch = arch.unwrap_or_default();
    let key = |k: &str| format!("{arch}.{k}");
    let u = |k: &str| -> std::result::Result<usize, String> {
        g.get_metadata_u32(&key(k))
            .map(|v| v as usize)
            .ok_or_else(|| format!("{arch} GGUF has no {}", key(k)))
    };
    // A GGUF stores these as f32; `1e-6` comes back as 9.99999997e-7. Widen through the f32's
    // shortest decimal (`1e-6`), the value the converter was given rather than its f32 rounding.
    let f = |k: &str| -> std::result::Result<f64, String> {
        g.get_metadata_f32(&key(k))
            .map(|v| v.to_string().parse::<f64>().unwrap_or(v as f64))
            .ok_or_else(|| format!("{arch} GGUF has no {}", key(k)))
    };
    let embd = g
        .shape_raw("token_embd.weight")
        .ok_or("GGUF has no token_embd.weight")?;
    let vocab = *embd.get(1).ok_or("token_embd.weight is not 2-D")?;
    let hidden = u("embedding_length")?;
    if embd[0] != hidden {
        return Err(format!(
            "token_embd.weight is {embd:?} but embedding_length is {hidden}"
        ));
    }
    let head_dim = u("attention.key_length")?;
    let v_len = u("attention.value_length")?;
    if v_len != head_dim {
        return Err(format!(
            "attention.value_length {v_len} != key_length {head_dim} (not supported)"
        ));
    }
    let shared = g
        .get_metadata_u32(&key("expert_shared_feed_forward_length"))
        .unwrap_or(0);
    if shared != 0 {
        return Err(format!(
            "{arch} GGUF declares a shared expert of width {shared}; this path has none"
        ));
    }
    if let Some(k) = g
        .metadata_keys()
        .into_iter()
        .find(|k| k.starts_with(&key("rope.scaling")))
    {
        return Err(format!(
            "{arch} GGUF sets {k}: rope scaling is not supported here"
        ));
    }
    let moe_intermediate = u("expert_feed_forward_length")?;
    let cfg = ModelConfig {
        nextn_layers: 0,
        vocab_size: vocab,
        hidden_size: hidden,
        intermediate_size: u("feed_forward_length")?,
        num_layers: u("block_count")?,
        num_attention_heads: u("attention.head_count")?,
        num_kv_heads: u("attention.head_count_kv")?,
        head_dim,
        rms_norm_eps: f("attention.layer_norm_rms_epsilon")?,
        rope_theta: f("rope.freq_base")?,
        rope_scaling: RopeScaling::None,
        max_position_embeddings: u("context_length")?,
        tie_word_embeddings: g.shape_raw("output.weight").is_none(),
        mlp: MlpKind::Moe {
            num_experts: u("expert_count")?,
            top_k: u("expert_used_count")?,
            shared_experts: 0,
            moe_intermediate,
            norm_topk: true,
        },
        qk_norm: g.shape_raw("blk.0.attn_q_norm.weight").is_some(),
        norm_style: NormStyle::Llama,
        attn: AttnKind::Causal,
        embedding_scale: None,
        query_pre_attn_scalar: None,
        final_logit_softcap: None,
        value_norm: false,
        gate_act: GateAct::Silu,
    };
    cfg.validate().map_err(|e| e.to_string())?;
    let mrope_sections = g
        .get_metadata_i32_array(&key("rope.dimension_sections"))
        .and_then(|v| <[i32; 4]>::try_from(v).ok())
        .map(|v| v.map(|x| x.max(0) as u32));
    Ok(Some(GgufMoeHeader {
        deepstack_layers: g.get_metadata_u32(&key("n_deepstack_layers")).unwrap_or(0),
        arch,
        cfg,
        mrope_sections,
    }))
}

#[cfg(test)]
mod qwen3_omni_config_tests {
    use super::*;

    /// The real header (read-only, header bytes only), when the file is on this disk
    /// (`ARF_QWEN_OMNI_GGUF=<path>`, else `models/qwen3-omni/` under the checkout). The test
    /// SKIPS LOUDLY without it: then it verified nothing about the file.
    #[test]
    fn omni_thinker_header_matches_the_built_in_row() {
        let p = std::env::var("ARF_QWEN_OMNI_GGUF")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                std::path::PathBuf::from(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../models/qwen3-omni/Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf"
                ))
            });
        let Ok(g) = crate::model::gguf::LazyGguf::open_raw(&p) else {
            eprintln!(
                "SKIP: {} not present -- this run verified NOTHING",
                p.display()
            );
            return;
        };
        let h = qwen3moe_family_from_gguf(&g)
            .expect("header parses")
            .expect("qwen3vlmoe");
        assert_eq!(h.arch, "qwen3vlmoe");
        assert_eq!(h.cfg, ModelConfig::qwen3_omni_30b());
        assert_eq!(h.mrope_sections, Some([24, 20, 20, 0]));
        assert_eq!(h.deepstack_layers, 0);
        assert_eq!(h.cfg.vocab_size, 152_064);
        assert_eq!(h.cfg.rope_theta, 1e6);
    }

    #[test]
    fn omni_differs_from_the_coder_only_where_the_file_says() {
        let (o, c) = (
            ModelConfig::qwen3_omni_30b(),
            ModelConfig::qwen3_coder_30b(),
        );
        assert_eq!((o.vocab_size, c.vocab_size), (152_064, 151_936));
        assert_eq!((o.rope_theta, c.rope_theta), (1e6, 1e7));
        assert_eq!(
            ModelConfig {
                vocab_size: c.vocab_size,
                rope_theta: c.rope_theta,
                max_position_embeddings: c.max_position_embeddings,
                ..o.clone()
            },
            c
        );
        assert_eq!(config_for_arch("qwen3vlmoe").unwrap(), o);
        assert_eq!(config_for_arch("qwen3-omni").unwrap(), o);
        // the Coder's names still select the Coder
        assert_eq!(config_for_arch("qwen3moe").unwrap(), c);
    }
}

#[cfg(test)]
mod arch_registry_tests {
    use super::*;

    /// The error a GGUF header gets when it selects no built-in architecture: what it read, and
    /// whether `--arch` can help.
    #[test]
    fn a_header_that_selects_nothing_says_what_it_read_and_whether_arch_helps() {
        // A family we have, another size: --arch cannot help.
        let e = header_mismatch_text("qwen35", Some(4), None, Some(64), Some(2));
        assert!(
            e.contains("qwen35 (4 blocks, hidden 64, 2 attention heads)"),
            "{e}"
        );
        assert!(
            e.contains("qwen3.8-27b (64 layers, hidden 5120, 24 heads)"),
            "{e}"
        );
        assert!(e.contains("--arch will not start it"), "{e}");
        // A family no built-in is read from: the list, and --arch may help.
        let e = header_mismatch_text("mamba", Some(48), Some(1), None, Some(8));
        assert!(
            e.contains("mamba (48 blocks (1 next-token), hidden ?, 8 attention heads)"),
            "{e}"
        );
        assert!(
            e.contains("pass --arch (one of: ") && e.contains("qwen3.8-27b"),
            "{e}"
        );
    }

    /// Header values read from real files (2026-10-07): the 27B says 65 blocks with 1 next-token
    /// layer, the decision model built on it 64 with none, Gemma 3 4B 34 blocks.
    #[test]
    fn a_gguf_header_selects_a_built_in_architecture_by_family_and_geometry() {
        assert_eq!(
            arch_from_gguf_header("qwen35", 65, 1, 5120, 24),
            Some("qwen3.8-27b")
        );
        assert_eq!(
            arch_from_gguf_header("qwen35", 64, 0, 5120, 24),
            Some("qwen3.8-27b")
        );
        assert_eq!(
            arch_from_gguf_header("gemma3", 34, 0, 2560, 8),
            Some("gemma-3-4b")
        );
        // The family alone is not enough: another size of it has no built-in config.
        assert_eq!(arch_from_gguf_header("qwen35", 32, 0, 2560, 16), None);
        // Nor is the geometry under another family's name.
        assert_eq!(arch_from_gguf_header("llama", 64, 0, 5120, 24), None);
        assert_eq!(arch_from_gguf_header("unknown", 64, 0, 5120, 24), None);
        // Every pair in the table resolves to a real architecture.
        for (_, canonical) in GGUF_ARCHITECTURES {
            assert!(config_for_arch(canonical).is_ok(), "{canonical}");
        }
    }

    #[test]
    fn every_listed_architecture_resolves() {
        for e in ARCHITECTURES {
            config_for_arch(e.canonical)
                .unwrap_or_else(|_| panic!("{} does not resolve", e.canonical));
            for a in e.aliases {
                config_for_arch(a).unwrap_or_else(|_| panic!("alias {a} does not resolve"));
            }
        }
    }

    #[test]
    fn names_are_unique_across_entries() {
        let mut seen = std::collections::HashSet::new();
        for e in ARCHITECTURES {
            for n in std::iter::once(&e.canonical).chain(e.aliases.iter()) {
                assert!(seen.insert(*n), "{n} is listed twice");
            }
        }
    }

    #[test]
    fn lookup_normalises_case_and_underscores() {
        assert!(config_for_arch("MUSE_GLIMMER_30B").is_ok());
        assert!(config_for_arch("Qwen3.8").is_ok());
    }

    #[test]
    fn unknown_arch_lists_what_is_supported() {
        let e = config_for_arch("gpt-5").unwrap_err();
        assert!(
            e.contains("muse-glimmer-30b"),
            "error must list the supported set: {e}"
        );
    }
}
