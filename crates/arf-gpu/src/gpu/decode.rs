//! Single-stream (m=1) decode path for [`GpuModel`]: per-token forward
//! (`decode_token_impl`), the MoE block, and the sort-by-expert test hook.
//! Split out of the original monolithic gpu.rs; behavior unchanged.

use super::*;

/// The MoE expert binding list for the **wts-present** layout, shared by `expert_gemv`
/// and the fused `down_reduce` block (both bind weights AND routing weights). The slot
/// order is a HARD contract with the hand-written `matmul_vec_moe*` WGSL:
///
/// ```text
///   0:a  1:weights-buf  2:c  3:dims  [4:scales]  ids  wts  [mins]  [dd]
/// ```
///
/// where `scales` is present for Q4/Q4K/Q4KS (inserted after `dims`), and `mins`/`dd`
/// are appended after `wts` for Q4K (mins) / Q4KS (mins+dd). The caller supplies `a`,
/// `c`, `dims`, `ids`, `wts` and picks the matching kernel; this builds ONLY the
/// variant-derived binding vector so the contract lives in one audited place.
///
/// NOTE: `batched_gu` deliberately does NOT use this — it has a DIFFERENT layout (no
/// `wts`; `mins`/`dd` follow `ids` directly) plus per-arm grid math. Unifying the two
/// would put `mins`/`dd` at the wrong slot. The `moe_binding_order_is_contract` test
/// pins BOTH layouts against silent regression.
fn moe_wts_bindings<'a>(
    experts: &'a PackedExperts,
    a: wgpu::BindingResource<'a>,
    c: wgpu::BindingResource<'a>,
    dims: wgpu::BindingResource<'a>,
    ids: wgpu::BindingResource<'a>,
    wts: wgpu::BindingResource<'a>,
) -> Vec<wgpu::BindingResource<'a>> {
    let binds = moe_wts_bindings_inner(experts, a, c, dims, ids, wts);
    // Pin the slot count to the role contract — catches an arm that drops/adds a binding.
    debug_assert_eq!(
        binds.len(),
        moe_wts_roles(experts.tag()).len(),
        "moe_wts_bindings slot count must match moe_wts_roles"
    );
    binds
}

fn moe_wts_bindings_inner<'a>(
    experts: &'a PackedExperts,
    a: wgpu::BindingResource<'a>,
    c: wgpu::BindingResource<'a>,
    dims: wgpu::BindingResource<'a>,
    ids: wgpu::BindingResource<'a>,
    wts: wgpu::BindingResource<'a>,
) -> Vec<wgpu::BindingResource<'a>> {
    match experts {
        PackedExperts::Bf16(b) => vec![a, b.as_entire_binding(), c, dims, ids, wts],
        PackedExperts::Q4 { codes, scales } => vec![
            a,
            codes.as_entire_binding(),
            c,
            dims,
            scales.as_entire_binding(),
            ids,
            wts,
        ],
        PackedExperts::Q4K {
            codes,
            scales,
            mins,
        } => vec![
            a,
            codes.as_entire_binding(),
            c,
            dims,
            scales.as_entire_binding(),
            ids,
            wts,
            mins.as_entire_binding(),
        ],
        PackedExperts::Q4KS {
            codes,
            scales,
            mins,
            dd,
        } => vec![
            a,
            codes.as_entire_binding(),
            c,
            dims,
            scales.as_entire_binding(),
            ids,
            wts,
            mins.as_entire_binding(),
            dd.as_entire_binding(),
        ],
    }
}

/// What one `decode_token_impl` call should do.
///
/// Eight positional parameters read as `(Some(tok), past_len + i, true, None, Some(i as u64),
/// None, 1)` at the call site — five of which are `None`/`true`/`1` and say nothing about their
/// own meaning. Named fields plus `Default` let each caller state only what it is changing.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct DecodeStep {
    /// The token to embed, or `None` to read the previous step's on-GPU output.
    pub token: Option<u32>,
    /// Position in the sequence (drives RoPE and the KV write slot).
    pub pos: usize,
    /// Run the final norm + lm_head. `false` skips the tail for a prefill-only step.
    pub want_logits: bool,
    /// Write the sampled id to `out_tokens[slot]` instead of slot 0.
    pub collect_slot: Option<u64>,
    /// Write to `out_tokens[collect_after]` — the speculative-verify sequential path.
    pub collect_after: Option<u64>,
    /// On-GPU sampling parameters; `None` is greedy argmax.
    pub sampling: Option<GpuSampling>,
    /// How many consecutive tokens to chain into ONE command buffer. `0` and `1` both mean one.
    pub k: usize,
}

impl GpuModel {
    /// The FFN gate-activation kernel for this model: GeGLU (`gelu_pytorch_tanh`)
    /// for Gemma, SwiGLU (SiLU) for Llama/Qwen — selected by `cfg.gate_act`. Both
    /// share the swiglu binding layout, so it's a drop-in kernel swap.
    pub(super) fn gate_kernel<'a>(&self, k: &'a GpuKernels) -> &'a ComputeKernel {
        match self.cfg.gate_act {
            arf_core::config::GateAct::GeluTanh => &k.geglu,
            arf_core::config::GateAct::Silu => &k.swiglu,
        }
    }

    /// Try to run the Dense FFN (gate+up+geglu+down) on the native Metal island over the
    /// shared scratch + Q4ksMtl weights — returns `true` if it ran (mlp_down written), `false`
    /// to fall through to the portable wgpu path. Gated on: the island exists, the 3 FFN
    /// weights are native Q4KS (carry `.mtl`), and the scratch is dual-view (`.mtl` present).
    /// Flushes the wgpu CommandPass + poll-fences so `normed` is GPU-resident before the
    /// island reads it (separate queues → no auto-ordering). The post-FFN rmsnorm_add stays
    /// on the wgpu path (small norm weight; reads the shared mlp_down the island wrote).
    #[cfg(target_os = "macos")]
    fn try_island_ffn(
        &self,
        gate_w: &GpuMatWeight,
        up_w: &GpuMatWeight,
        down_w: &GpuMatWeight,
        cp: &mut CommandPass,
        cursor: &mut Option<BindCursor>,
    ) -> bool {
        use crate::gpu::types::GpuMatWeight::Q4KS;
        // Need the island + all 3 weights' Q4ksMtl + the scratch dual-views.
        let Some(island) = self.island.as_ref() else {
            return false;
        };
        let (Q4KS { mtl: Some(g), .. }, Q4KS { mtl: Some(u), .. }, Q4KS { mtl: Some(d), .. }) =
            (gate_w, up_w, down_w)
        else {
            return false;
        };
        let (Some(normed), Some(gate), Some(up), Some(mlp_down)) = (
            self.scratch.normed_mtl.as_ref(),
            self.scratch.gate_mtl.as_ref(),
            self.scratch.up_mtl.as_ref(),
            self.scratch.mlp_down_mtl.as_ref(),
        ) else {
            return false;
        };

        // Flush wgpu so `normed` (the in_norm output recorded into cp) is GPU-resident, then
        // poll-FENCE: flush submits but does NOT block, and the island runs on a separate
        // MTLCommandQueue with no auto-ordering → without the poll the island reads stale
        // `normed`. The steady-decode bind cursor cannot span the flush (clears binds).
        // BOUNDED wait (was wait_indefinitely): the wgpu<->Metal boundary can DEADLOCK on the
        // B=1 MoE first token (wgpu poll never returns while the island holds the GPU) — the
        // all-night hang, 2026-07-08. A 10s timeout breaks the deadlock (returns Timeout) instead
        // of hanging forever; the normal case completes in <<10s so behavior is unchanged.
        *cursor = None;
        cp.flush();
        let _ = self.ctx.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(10)),
        });

        let inter = self.cfg.intermediate_size;
        let h = self.cfg.hidden_size;
        let mut isl = island.lock().unwrap();
        let r = isl.ffn_dispatch(&self.ctx, g, u, d, normed, gate, up, mlp_down, inter, h);
        r.is_ok()
    }

    /// Whole-token megakernel driver: assemble the per-layer descriptors + invoke the island
    /// encoder. Returns true if it ran (all preconditions met: island up, gemma-4 Dense Q4KS
    /// with .mtl everywhere, KvQuant::None, scratch+norm+pool+rope dual-views present), else
    /// false to fall back to the wgpu layer loop. `hidden` must already hold the embedded token.
    #[cfg(target_os = "macos")]
    fn run_megakernel(
        &self,
        pos: usize,
        ctx_len: usize,
        n_layers: usize,
        do_final_norm: bool,
        island_embed: bool,
        island_sample: bool,
        out_slot: Option<u64>,
        k: usize,
    ) -> bool {
        use crate::gpu::concurrent_metal::{MegaIo, MegaLayer, MegaScratch};
        use crate::gpu::types::{GpuMatWeight::Q4KS, GpuMlp};
        let Some(island) = self.island.as_ref() else {
            return false;
        };
        let mc = &self.cfg;
        let h = mc.hidden_size;
        let inter = mc.intermediate_size;
        let s = &self.scratch;
        let dbg = std::env::var("ARF_MEGAKERNEL_DEBUG").is_ok();
        macro_rules! fail {
            ($w:expr) => {{
                if dbg {
                    eprintln!("[mega] precondition FAIL: {}", $w);
                }
                return false;
            }};
        }
        // All scratch dual-views must be present.
        let (
            Some(hidden),
            Some(normed),
            Some(qb),
            Some(kb),
            Some(vb),
            Some(aout),
            Some(gateb),
            Some(upb),
            Some(mdb),
        ) = (
            s.hidden_mtl.as_ref(),
            s.normed_mtl.as_ref(),
            s.q_mtl.as_ref(),
            s.k_mtl.as_ref(),
            s.v_mtl.as_ref(),
            s.attn_out_mtl.as_ref(),
            s.gate_mtl.as_ref(),
            s.up_mtl.as_ref(),
            s.mlp_down_mtl.as_ref(),
        )
        else {
            fail!("scratch dual-views (hidden/normed/q/k/v/attn_out/gate/up/mlp_down)")
        };
        let Some(final_norm) = self.mega.final_norm.as_ref() else {
            fail!("final_norm mtl")
        };
        if self.kv.keys_mtl.is_empty() {
            fail!("kv keys_mtl empty")
        }

        let mut isl = island.lock().unwrap();
        // NOWAIT pipeline: wait on the token MEGA_DEPTH-ago (which owns this token's ring-slot
        // uniform buffers) BEFORE we overwrite those uniforms — so the CPU never races an
        // in-flight GPU read. Bounds CPU to MEGA_DEPTH-ahead; GPU stays pipelined.
        isl.wait_ring_slot(pos);
        // Build/refresh the per-layer dims uniforms (geometry is fixed per layer → build once).
        if !isl.mega_dims_ready(self.layers.len()) {
            let descs: Vec<_> = self
                .layers
                .iter()
                .map(|ly| {
                    let nh = mc.num_attention_heads;
                    crate::gpu::concurrent_metal::MegaDimDesc {
                        nh,
                        nkv: ly.kv_heads,
                        hd: ly.head_dim,
                        rotary_dim: ly.rotary_dim,
                        eps: mc.rms_norm_eps as f32,
                        scale: mc
                            .query_pre_attn_scalar
                            .unwrap_or(1.0 / (ly.head_dim as f32).sqrt()),
                        window: ly.window.unwrap_or(0),
                        layer_scalar: ly.layer_scalar,
                    }
                })
                .collect();
            if !isl.build_mega_dims(&self.ctx, &descs, ctx_len, h) {
                fail!("build_mega_dims")
            }
        }
        // Ensure per-token index buffers + uniforms exist (mutable) BEFORE the immutable
        // descriptor borrows (megakernel_record is &self). With K>1 the island identity slots
        // must cover up to the last sub-token's pos (pos+k-1) → size for ctx_len+k.
        let k = k.max(1);
        if isl
            .ensure_mega_bufs(&self.ctx, h, inter, ctx_len + k)
            .is_err()
        {
            fail!("ensure_mega_bufs")
        }
        // MoE (Qwen3) intermediates + dims (mutable) BEFORE the immutable descriptor borrows.
        // Sized for the model's (top_k, moe_inter, num_experts) — uniform across MoE layers.
        if let arf_core::config::MlpKind::Moe {
            num_experts,
            top_k,
            moe_intermediate,
            norm_topk,
            ..
        } = &mc.mlp
        {
            if isl
                .ensure_moe_bufs(&self.ctx, *top_k, *moe_intermediate, *num_experts)
                .is_err()
            {
                fail!("ensure_moe_bufs")
            }
            // qk_norm dims need nh/nkv/hd/eps (uniform across qwen MoE layers — use layer 0).
            let (nh0, nkv0, hd0) = (
                mc.num_attention_heads,
                self.layers[0].kv_heads,
                self.layers[0].head_dim,
            );
            if isl
                .ensure_moe_dims(
                    &self.ctx,
                    h,
                    *moe_intermediate,
                    *top_k,
                    *num_experts,
                    *norm_topk,
                    nh0,
                    nkv0,
                    hd0,
                    mc.rms_norm_eps as f32,
                )
                .is_err()
            {
                fail!("ensure_moe_dims")
            }
        } else if mc.qk_norm && !mc.value_norm {
            // DENSE model with q/k-only per-head norm (gemma3-4b): the megakernel's qk_norm branch
            // needs the moe_qk_dims uniform {nh,nkv,hd,eps} too. Model-agnostic (no MoE params) — build
            // it so the greedy megakernel doesn't panic on moe_qk_dims_buf(). (Gemma-4 has value_norm →
            // the fused qkv_norm branch, which doesn't touch moe_qk_dims.)
            let (nh0, nkv0, hd0) = (
                mc.num_attention_heads,
                self.layers[0].kv_heads,
                self.layers[0].head_dim,
            );
            if isl
                .ensure_qk_dims(&self.ctx, nh0, nkv0, hd0, mc.rms_norm_eps as f32)
                .is_err()
            {
                fail!("ensure_qk_dims")
            }
        }
        // K-deep per-sub-token uniforms (pos/attn_dims/out_slot) for multi-token-per-cmdbuf.
        // Allocated after build_mega_dims (which seeds the per-layer attn words they copy).
        if k > 1 && isl.ensure_ktok_bufs(&self.ctx, k, n_layers).is_err() {
            fail!("ensure_ktok_bufs")
        }
        // Single-queue I/O dims (mutable) — also before the immutable borrows.
        let embed_scale = mc.embedding_scale.unwrap_or(1.0);
        isl.set_mega_io_dims(
            &self.ctx,
            h,
            embed_scale,
            mc.vocab_size,
            mc.final_logit_softcap,
        );
        // out_tokens store slot (NOWAIT double-buffer by pos parity) — write now (mutable).
        let do_store = island_sample && out_slot.is_some() && self.scratch.out_tokens_mtl.is_some();
        if do_store {
            isl.set_mega_out_slot(&self.ctx, out_slot.unwrap() as u32, pos);
        }
        // o_proj reads `attn` (q_dim-sized), which needs its own buffer distinct from the input
        // `q` — a dedicated island attn buffer, built in build_mega_dims. Assemble the borrowed
        // per-layer descriptors.
        let mut megas: Vec<MegaLayer> = Vec::with_capacity(n_layers);
        for (li, ly) in self.layers.iter().enumerate().take(n_layers) {
            let extract = |w: &crate::gpu::types::GpuMatWeight| -> Option<*const crate::gpu::concurrent_metal::Q4ksMtl> {
                if let Q4KS { mtl: Some(m), .. } = w { Some(std::sync::Arc::as_ptr(m)) } else { None }
            };
            if dbg && li == 0 {
                let mut names: Vec<(&str, &crate::gpu::types::GpuMatWeight)> = vec![
                    ("q", &ly.q_proj),
                    ("k", &ly.k_proj),
                    ("v", &ly.v_proj),
                    ("o", &ly.o_proj),
                ];
                match &ly.mlp {
                    GpuMlp::Dense(d) => names.extend([
                        ("gate", &d.gate_proj),
                        ("up", &d.up_proj),
                        ("down", &d.down_proj),
                    ]),
                    GpuMlp::Moe(m) => names.push(("router", &m.router)),
                }
                for (n, w) in names {
                    let kind = match w {
                        Q4KS { mtl: Some(_), .. } => "Q4KS+mtl",
                        Q4KS { mtl: None, .. } => "Q4KS NO-mtl",
                        crate::gpu::types::GpuMatWeight::Bf16(_) => "Bf16",
                        _ => "other",
                    };
                    eprintln!("[mega] L0 {n}_proj = {kind}");
                }
            }
            // q/k/v/o attention projections — required for BOTH dense and MoE layers.
            let (Some(q), Some(k), Some(v), Some(o)) = (
                extract(&ly.q_proj),
                extract(&ly.k_proj),
                extract(&ly.v_proj),
                extract(&ly.o_proj),
            ) else {
                fail!(format!("L{li} q/k/v/o Q4ksMtl (some weight lacks .mtl)"))
            };
            // Muse Glimmer's attention gate. A layer that HAS a gate but whose gate did not land
            // on the native path must REFUSE the island, not run without it — dropping the gate
            // is a correctness bug (fluent output that ignores the prompt), not a slow path.
            let attn_gate_mtl = match ly.attn_gate.as_ref() {
                None => None,
                Some(g) => match extract(g) {
                    Some(p) => Some(unsafe { &*p }),
                    None => fail!(format!(
                        "L{li} attn_gate Q4ksMtl (island would drop the gate)"
                    )),
                },
            };
            // FFN: Dense (gemma/llama gate/up/down Q4ksMtl) OR MoE (router .mtl + all-experts MoeMtl
            // + per-model dims on the island). On MoE, the dense gate/up/down fields are None.
            let (gate, up, down, gate_q3k, up_q3k, down_q3k, moe): (
                Option<*const _>,
                Option<*const _>,
                Option<*const _>,
                Option<&crate::gpu::concurrent_metal::Q3ksMtl>,
                Option<&_>,
                Option<&_>,
                Option<crate::gpu::concurrent_metal::MegaMoe>,
            ) = match &ly.mlp {
                GpuMlp::Dense(d) => {
                    let (Some(g), Some(u), Some(dn)) = (
                        extract(&d.gate_proj),
                        extract(&d.up_proj),
                        extract(&d.down_proj),
                    ) else {
                        fail!(format!(
                            "L{li} dense gate/up/down Q4ksMtl (some weight lacks .mtl)"
                        ))
                    };
                    let (gq, uq, dq) = match d.q3k.as_ref() {
                        Some(t) => (Some(&t.0), Some(&t.1), Some(&t.2)),
                        None => (None, None, None),
                    };
                    (Some(g), Some(u), Some(dn), gq, uq, dq, None)
                }
                GpuMlp::Moe(m) => {
                    // Router (.mtl Q4ksMtl) + all-experts gate/up/down MoeMtl. If the loader didn't
                    // build any of them (not Q4KS, or MSL off), FALL BACK to the wgpu path (fail).
                    let Some(router) = extract(&m.router) else {
                        fail!(format!("L{li} MoE router .mtl"))
                    };
                    let (Some(gm), Some(um), Some(dm)) =
                        (m.gate_mtl.as_ref(), m.up_mtl.as_ref(), m.down_mtl.as_ref())
                    else {
                        fail!(format!(
                            "L{li} MoE gate/up/down all-experts .mtl (loader off)"
                        ))
                    };
                    let mm = crate::gpu::concurrent_metal::MegaMoe {
                        router: unsafe { &*router },
                        gate: gm.as_ref(),
                        up: um.as_ref(),
                        down: dm.as_ref(),
                        gu_dims: isl.moe_gu_dims_buf(),
                        down_dims: isl.moe_down_dims_buf(),
                        route_dims: isl.moe_route_dims_buf(),
                        swiglu_dims: isl.moe_swiglu_dims_buf(),
                        num_experts: m.num_experts,
                        top_k: m.top_k,
                        inter: m.moe_inter,
                    };
                    (None, None, None, None, None, None, Some(mm))
                }
            };
            let nm = &ly.norm_mtls;
            // pre/post-ffn norms = gemma only (None on Qwen MoE — the MoE FFN flow uses add_norm
            // (post_norm) + plain residual). input/post/q/k norms required for both.
            let (Some(input_norm), Some(post_norm), Some(qn), Some(kn)) = (
                nm.input_norm.as_ref(),
                nm.post_norm.as_ref(),
                nm.q_norm.as_ref(),
                nm.k_norm.as_ref(),
            ) else {
                fail!(format!("L{li} norm_mtls (input/post/q/k)"))
            };
            let pre_ffn = nm.pre_ffn_norm.as_deref();
            let post_ffn = nm.post_ffn_norm.as_deref();
            // Dense layers MUST have pre/post-ffn norms (gemma); MoE must NOT need them.
            if moe.is_none() && (pre_ffn.is_none() || post_ffn.is_none()) {
                fail!(format!("L{li} dense layer missing pre/post-ffn norm_mtls"));
            }
            let (Some(kpool), Some(vpool)) = (
                self.kv.keys_mtl.get(li).and_then(|o| o.as_ref()),
                self.kv.values_mtl.get(li).and_then(|o| o.as_ref()),
            ) else {
                fail!(format!("L{li} kv pool mtl"))
            };
            // RoPE table per layer: global (window None) → global table, else local.
            let (cos, sin) = if ly.window.is_none() {
                (
                    self.mega
                        .rope_cos_global
                        .as_ref()
                        .or(self.mega.rope_cos.as_ref()),
                    self.mega
                        .rope_sin_global
                        .as_ref()
                        .or(self.mega.rope_sin.as_ref()),
                )
            } else {
                (
                    self.mega
                        .rope_cos_local
                        .as_ref()
                        .or(self.mega.rope_cos.as_ref()),
                    self.mega
                        .rope_sin_local
                        .as_ref()
                        .or(self.mega.rope_sin.as_ref()),
                )
            };
            let (Some(cos), Some(sin)) = (cos, sin) else {
                fail!(format!("L{li} rope table mtl"))
            };
            // The attn_dims double-buffer is ringed by `pos % MEGA_DEPTH` (the SAME parity the
            // island's per-token patch uses for ctx/past_len in megakernel_record). Using `pos & 1`
            // here desynced the bind (mod 2) from the patch (mod MEGA_DEPTH=4): at pos≥2 the bound
            // attn_dims held a STALE past_len (e.g. pos=2 bound db[0] = past_len 0), so attention
            // silently attended only slot 0 → incoherent output that compounds every layer.
            let dims = isl.mega_layer_dims(li, pos % crate::gpu::concurrent_metal::MEGA_DEPTH);
            megas.push(MegaLayer {
                q: unsafe { &*q },
                k: unsafe { &*k },
                v: unsafe { &*v },
                o: unsafe { &*o },
                gate: gate.map(|p| unsafe { &*p }),
                up: up.map(|p| unsafe { &*p }),
                down: down.map(|p| unsafe { &*p }),
                gate_q3k,
                up_q3k,
                down_q3k,
                // ARF_Q8_DOWN is a hybrid-27B experiment; this path refuses hybrids (below).
                down_q8: None,
                moe,
                gdn: None, // L136: GDN recurrent branch — Qwen3.8 hybrid only.
                // L155: false for the same reason `gdn` is None — this m=1 path REFUSES hybrid
                // models outright (try_m1_megakernel bails on any layer with `gdn`, L140), so no
                // joint query+gate layer can reach it.
                q_joint_gate: false,
                input_norm,
                post_norm,
                pre_ffn_norm: pre_ffn,
                post_ffn_norm: post_ffn,
                // f16 KV on the m=1 (chat) path. The f16 pools are allocated by
                // weights.rs whenever ARF_KV_F16 is set (path-independent), and this path
                // used to hard-code None, which is WHY chat never got the batched path's KV
                // bandwidth work: MEASURED 2026-08-03, single-stream falls 89.9 -> 21.8 tok/s
                // from depth 0 -> 2048 while llama stays FLAT at ~93-96 (ratio 0.94x -> 0.23x).
                // Depth is a BANDWIDTH problem; f16 halves the KV bytes read per token.
                attn_gate: attn_gate_mtl,
                rope_interleaved: mc.rope_interleaved(),
                gate_silu: matches!(mc.gate_act, arf_core::config::GateAct::Silu),
                q_norm: qn,
                k_norm: kn,
                value_norm: mc.value_norm,
                kpool,
                vpool,
                kv_q8: None,
                kpool_f16: self
                    .kv
                    .keys_mtl_f16
                    .get(li)
                    .and_then(|o| o.as_ref())
                    .map(|b| b.0.as_ref()),
                vpool_f16: self
                    .kv
                    .values_mtl_f16
                    .get(li)
                    .and_then(|o| o.as_ref())
                    .map(|b| b.0.as_ref()),
                rope_cos: cos,
                rope_sin: sin,
                qkv_dims: dims.0,
                rope_dims: dims.1,
                scatter_dims: dims.2,
                attn_dims: dims.3,
                scale_dims: dims.4,
                nh: mc.num_attention_heads,
                nkv: ly.kv_heads,
                hd: ly.head_dim,
            });
        }
        let silu = isl.mega_silu_buf();
        let attn = isl.mega_attn_buf();
        let scratch = MegaScratch {
            hidden,
            normed,
            q: qb,
            k: kb,
            v: vb,
            attn,
            attn_out: aout,
            gate: gateb,
            up: upb,
            silu,
            mlp_down: mdb,
        };
        // lm_head ON the island (Q8) when transcoded + logits is shared — folds the bf16
        // tail GEMV into the one command buffer. Only when this token wants logits (final norm).
        let (lm_head, logits) = if do_final_norm {
            match (self.mega.lm_head.as_ref(), self.scratch.logits_mtl.as_ref()) {
                (Some(lm), Some(lg)) => (Some(lm.as_ref()), Some(&**lg)),
                _ => (None, None),
            }
        } else {
            (None, None)
        };
        // SINGLE-QUEUE I/O: embed (first op) + softcap/argmax (last ops) on the island, so the
        // whole token is one queue. embed only on the steady token-feedback path; argmax only
        // for greedy. (Dims built above before the immutable borrows.)
        let io = MegaIo {
            embed_table: if island_embed {
                self.mega.embed.as_deref()
            } else {
                None
            },
            next_token: self.scratch.next_token_mtl.as_deref(),
            embed_dims: if island_embed {
                isl.mega_embed_dims()
            } else {
                None
            },
            // Pre-trunk embedding norm — ONLY when the island runs the embed itself; otherwise
            // the wgpu dispatch already applied it and doing it twice would double-normalize.
            embed_norm: if island_embed {
                self.mega.embed_norm.as_deref()
            } else {
                None
            },
            softcap_dims: if island_sample {
                isl.mega_softcap_dims()
            } else {
                None
            },
            argmax_dims: if island_sample {
                isl.mega_argmax_dims()
            } else {
                None
            },
            out_tokens: if do_store {
                self.scratch.out_tokens_mtl.as_deref()
            } else {
                None
            },
            out_slot_dims: if do_store {
                isl.mega_out_slot_buf(pos)
            } else {
                None
            },
            // MULTI-TOKEN: base out_tokens slot for sub-token kt=0 (= caller's collect_slot+1);
            // sub-token kt stores to base+kt via the island-owned kt-deep out-slot uniforms.
            out_slot_base: if do_store && k > 1 {
                out_slot.map(|s| s as u32)
            } else {
                None
            },
        };
        isl.megakernel_record(
            &self.ctx,
            &megas,
            &scratch,
            final_norm,
            h,
            inter,
            mc.rms_norm_eps as f32,
            pos,
            ctx_len,
            do_final_norm,
            lm_head,
            logits,
            &io,
            k,
        )
        .is_ok()
    }

    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(
            level = "debug",
            name = "gpu_decode_token",
            skip_all,
            fields(pos = pos, want_logits = want_logits)
        )
    )]
    pub(super) fn decode_token(
        &self,
        token: Option<u32>,
        pos: usize,
        want_logits: bool,
        collect_slot: Option<u64>,
        sampling: Option<GpuSampling>,
    ) {
        self.decode_token_impl(DecodeStep {
            token,
            pos,
            want_logits,
            collect_slot,
            sampling,
            k: 1,
            ..Default::default()
        })
    }

    /// MULTI-TOKEN-PER-CMDBUF (ARF_MEGA_KTOK): decode `k` consecutive greedy tokens starting
    /// at `pos` into ONE island command buffer (one commit / k tokens) — the architecture lever
    /// that kills the per-buffer ~20ms inter-buffer GPU idle gap. Sub-token kt runs at pos+kt and
    /// stores to out_tokens[collect_slot+1+kt]; the k sub-tokens chain on-GPU via next_token. Only
    /// valid on the greedy island_sample path (token==None, full layers, want_logits, no sampling)
    /// — falls back to k=1 otherwise. Returns the number of tokens actually decoded (k on the
    /// island path, 1 if it fell back).
    #[cfg(target_os = "macos")]
    pub(super) fn decode_tokens_k(&self, pos: usize, collect_slot: u64, k: usize) -> usize {
        self.decode_token_impl(DecodeStep {
            pos,
            want_logits: true,
            collect_slot: Some(collect_slot),
            k: k.max(1),
            ..Default::default()
        });
        // The island advanced `k` positions only if the megakernel actually ran k tokens; it
        // does so whenever the island_sample path is taken (the generate loop guards that with
        // KTOK>1 + greedy). We return k; the loop advances by k. (A fallback to k=1 would mean
        // the island didn't run — but the generate loop only calls this when KTOK is engaged and
        // the model is the megakernel-eligible greedy path, asserted by the warmup.)
        k.max(1)
    }

    /// `decode_token`, plus an optional copy of THIS step's freshly-sampled argmax
    /// into `out_tokens[collect_after]` (after the sample dispatch). The spec-decode
    /// verify uses it to capture every window position's prediction in one submit
    /// each, then drains `out_tokens` once — the cheap single-stream decode path
    /// (m==1 GEMV, so Q4 works), not the heavy `forward_batch`.
    pub(super) fn decode_token_impl(&self, step: DecodeStep) {
        let DecodeStep {
            token,
            pos,
            want_logits,
            collect_slot,
            collect_after,
            sampling,
            k: decode_k,
        } = step;
        // L364 — `decode_k` drives the island's k-step submission, which is macOS-only; off
        // macOS nothing reads it. Discarded explicitly so the binding above stays identical on
        // both platforms (renaming it to `_decode_k` would break every macOS use below).
        #[cfg(not(target_os = "macos"))]
        let _ = decode_k;
        // L8 — the f32 KV pool may be VOLATILE after a batched-prefill export, and every
        // single-stream path below reads/writes it directly (fused m=1 megakernel, WGSL
        // fallback, debug dumps). Re-arm + reclaim-recover BEFORE any encoding; no-op in the
        // common resident case. See kv_f32_rearm_for_single (batch.rs).
        #[cfg(target_os = "macos")]
        self.kv_f32_rearm_for_single(pos);
        let mc = &self.cfg;
        let h = mc.hidden_size;
        // `nh` (query-head count) is uniform across layers; head_dim/kv_heads/q_dim/
        // kv_dim/group/scale are PER-LAYER for Gemma 4 (global layers differ) and are
        // recomputed inside the layer loop from `GpuLayer`. The base values here are
        // only used for the final-norm/lm_head tail (which is layer-independent).
        let nh = mc.num_attention_heads;
        let bs = self.kv.block_size;
        // Gemma scales embeddings by √hidden after lookup; 1.0 (identity) otherwise.
        let embed_scale = mc.embedding_scale.unwrap_or(1.0);
        // PLACEMENT, not formula: gates the post_attn/post_ffn rmsnorm_add dispatches below.
        let is_gemma = mc.has_post_norms();
        let k = &self.kernels;
        let wg = |n: usize| [(n as u32).div_ceil(256), 1u32, 1u32];

        let ctx_len = pos + 1; // context is 0..=pos (all of it already in the pool)
                               // Slot mapping: identity (`slots[p]=p`, the single-stream default — sequence
                               // owns blocks 0,1,2,…) OR, when `block_table` is given (batched serving — each
                               // sequence owns a DISJOINT block range), `block_table[p/bs]*bs + p%bs`. This
                               // lets `generate_batch` prefill Gemma/MoE token-by-token through THIS correct
                               // single-stream path into each sequence's own slots (forward_batch is the
                               // Llama/Qwen-shaped path and diverges for Gemma's four-norm prefill).
                               // GAME-ENGINE FIX: on the full single-queue megakernel path the island uses its OWN
                               // resident ring (mega_slots_db/mega_pos_db) — the wgpu `slots` Vec here is UNUSED. Building
                               // it `(0..ctx_len)` every token = a fresh, GROWING heap Vec + a wgpu alloc_write into a
                               // GROWING uniform buffer → as ctx_len grows the arena periodically reallocs → wgpu
                               // create_buffer → staging-belt flush = the periodic ~18ms decode SPIKE. Skip it; the island
                               // only needs THIS token's slot (= pos for the single-stream identity map).
        let full_singleq = cfg!(target_os = "macos")
            && crate::weights::megakernel_on()
            && std::env::var_os("ARF_MEGA_SINGLEQ").is_some()
            && self.decode_block_table.borrow().is_none();
        let slots: Vec<u32> = if full_singleq {
            // only slot[pos] is read (by the island scatter via pos_b); avoid the O(ctx) Vec.
            let p = pos as u32;
            vec![(p / bs as u32) * bs as u32 + (p % bs as u32)]
        } else {
            (0..ctx_len as u32)
                .map(|p| match self.decode_block_table.borrow().as_deref() {
                    Some(tbl) => tbl[(p / bs as u32) as usize] * bs as u32 + (p % bs as u32),
                    None => (p / bs as u32) * bs as u32 + (p % bs as u32),
                })
                .collect()
        };
        let new_slot = [*slots.last().unwrap()];
        let positions = [pos as u32];

        let f32b = std::mem::size_of::<f32>() as u64;
        // Per-token transients are sub-allocated from arenas split so no single
        // dispatch reads and writes the same physical buffer (wgpu forbids
        // STORAGE_READ_ONLY + STORAGE_READ_WRITE on one buffer in a pass):
        //   idx — index buffers read by kv_scatter/rope/attention (slots, pos)
        //   out — attn/silu: written by attention/swiglu, read by the next matmul
        //   uni — the per-pass uniform blocks
        // The `kv` arena (formerly k_all/v_all scratch for kv_gather) is gone:
        // attention now reads the paged KV pool directly by slot (FlashInfer-style
        // in-kernel gather), so the gather copy + scratch + a dispatch/barrier are
        // eliminated. The arenas are allocated once for the model (sized for max
        // context) and reset() here each step — recreating them per token was
        // device allocations on the CPU critical path. They are dropped (reset) at
        // the start of each token, so there is no cross-token aliasing.
        let mut arenas = self.decode_arenas.borrow_mut();
        arenas.reset();
        let DecodeArenas { idx, out, uni } = &mut *arenas;

        // out: written by attention/swiglu; idx: index buffers uploaded for
        // scatter/rope/attention. `attn` holds the attention output before o_proj;
        // it is sized to the LARGEST layer's q_dim (Gemma 4 global layers have a
        // bigger head_dim → bigger q_dim) so any layer's write fits the region.
        // L154 — `max_q_dim()` covers the hybrid SSM family's 2×-packed attention q; `nh *
        // max_attn_dims().0` undersizes it by half and overflows on the first attention layer.
        let max_q_dim = mc.max_q_dim();
        let attn = out.alloc(max_q_dim as u64 * f32b).expect("attn");
        // Split-KV (flash-decoding) attention engages at long ctx to dodge the Metal
        // per-dispatch GPU watchdog (a single decode attention over ~8K keys in one
        // workgroup exceeds it). SPLITK workgroups each cover ctx/SPLITK keys and write
        // a partial (acc[hd], m, l) to `splitk_partials`; the combine kernel merges
        // them. Below SPLITK_CTX the single-dispatch kernel is cheaper (no combine).
        const SPLITK: u32 = 8;
        const SPLITK_CTX: usize = 2048;
        let max_hd = mc.max_attn_dims().0;
        let use_splitk = std::env::var("ARF_SPLITK").is_ok()
            && ctx_len > SPLITK_CTX
            && !self.kv.kv_quant.is_quantized();
        let splitk_partials = if use_splitk {
            if std::env::var("ARF_SPLITK_DEBUG").is_ok() && ctx_len.is_multiple_of(512) {
                eprintln!(
                    "[splitk engaged] ctx_len={ctx_len} splitk={SPLITK} chunk={}",
                    ctx_len.div_ceil(SPLITK as usize)
                );
            }
            Some(
                out.alloc(nh as u64 * SPLITK as u64 * (max_hd as u64 + 2) * f32b)
                    .expect("splitk_partials"),
            )
        } else {
            None
        };
        // silu holds [intermediate_size] (dense) or [top_k*moe_inter] (fused MoE).
        // Must match the `out` arena sizing in DecodeArenas::new (silu_len).
        let silu_len = match &mc.mlp {
            arf_core::config::MlpKind::Moe {
                top_k,
                moe_intermediate,
                ..
            } => (top_k * moe_intermediate).max(mc.intermediate_size),
            arf_core::config::MlpKind::Dense => mc.intermediate_size,
        };
        let silu = out.alloc(silu_len as u64 * f32b).expect("silu");
        // pos_buf and new_slot are FIXED size (1 elem); slots GROWS with ctx_len.
        // Allocate the fixed ones FIRST so their offsets are invariant across
        // tokens (pos_buf -> 0, new_slot -> 256) — this lets rope_qk's bind group
        // (which captures pos_buf) be cached (P15). If slots were first, its
        // ctx_len-growing size would shift pos_buf's offset past ctx_len~64 and a
        // cached rope_qk would silently read the wrong granule.
        let pos_buf = idx.alloc_write(&positions).expect("pos");
        let new_slot_buf = idx.alloc_write(&new_slot).expect("new_slot");
        let slots_buf = idx.alloc_write(&slots).expect("slots");

        let mut cp = CommandPass::new(&self.ctx);

        // Stash the id this step emits (the one sitting in next_token now, before
        // this step's sample overwrites it) into the output collector, so the
        // whole generation drains to the CPU once instead of once per token.
        // The island store_word writes out_tokens directly (collect_slot+1) on the single-queue
        // greedy path → skip this per-token wgpu copy (the remaining cross-queue stall).
        let skip_wgpu_collect = cfg!(target_os = "macos")
            && crate::weights::megakernel_on()
            && std::env::var_os("ARF_MEGA_SINGLEQ").is_some()
            && sampling.is_none();
        if let Some(slot) = collect_slot {
            if !skip_wgpu_collect {
                cp.copy_word(&self.scratch.next_token, &self.scratch.out_tokens, slot);
            }
        }

        // embed: hidden <- embed_table[token]. Known id → from a uniform;
        // None → from the resident next_token buffer (on-GPU feedback).
        match token {
            Some(id) => {
                // embed dims: [src_row, hidden, dst_row, scale_bits]
                let edims = uni
                    .alloc_write(&[id, h as u32, 0u32, embed_scale.to_bits()])
                    .expect("e");
                cp.add_bound(
                    // bf16 table → bf16 gather kernel (self.embed is uploaded bf16).
                    if self.embed_q4k {
                        &k.embed_q4k
                    } else {
                        &k.embed_bf16
                    },
                    "embed",
                    &[
                        self.embed.as_entire_binding(),
                        self.scratch.hidden.as_entire_binding(),
                        uni.resource(&edims),
                    ],
                    wg(h),
                );
            }
            None => {
                // SINGLE-QUEUE: skip the wgpu embed — the island runs it as op 0 (token==None
                // feedback path), keeping the token single-queue. Otherwise the wgpu embed runs.
                let skip_wgpu_embed = cfg!(target_os = "macos")
                    && crate::weights::megakernel_on()
                    && std::env::var_os("ARF_MEGA_SINGLEQ").is_some();
                if !skip_wgpu_embed {
                    // embed_tok dims: [hidden, scale_bits, _, _]
                    let edims = uni
                        .alloc_write(&[h as u32, embed_scale.to_bits(), 0u32, 0u32])
                        .expect("e");
                    cp.add_bound(
                        if self.embed_q4k {
                            &k.embed_tok_q4k
                        } else {
                            &k.embed_tok_bf16
                        },
                        "embed",
                        &[
                            self.embed.as_entire_binding(),
                            self.scratch.hidden.as_entire_binding(),
                            self.scratch.next_token.as_entire_binding(),
                            uni.resource(&edims),
                        ],
                        wg(h),
                    );
                }
            }
        }

        // Muse Glimmer: WEIGHTLESS pre-trunk RMSNorm on the embedding, before layer 0 — the twin
        // of the batch.rs site (see `GpuModel::embed_norm` for llama's op order and why omitting
        // it mis-scales all 52 layers). `embed_norm` is all-ones, so the standard kernel is a
        // weightless norm. `normed` is dead here (layer 0's `in_norm` rewrites it), so it serves
        // as the scratch for the hidden→normed→hidden round-trip; `k.embed` with scale 1.0 is
        // the plain row copy back. One row here, so it is two dispatches, not a loop.
        if let Some(embed_norm) = self.embed_norm.as_ref() {
            let end = uni
                .alloc_write(&dims_norm(h, mc.rms_norm_eps))
                .expect("end");
            cp.add_bound(
                &k.rmsnorm,
                "embed_norm",
                &[
                    self.scratch.hidden.as_entire_binding(),
                    embed_norm.as_entire_binding(),
                    self.scratch.normed.as_entire_binding(),
                    uni.resource(&end),
                ],
                [1, 1, 1],
            );
            let cd = uni
                .alloc_write(&[0u32, h as u32, 0u32, 1.0f32.to_bits()])
                .expect("embed_norm_cd");
            cp.add_bound(
                &k.embed,
                "embed_norm_copy",
                &[
                    self.scratch.normed.as_entire_binding(),
                    self.scratch.hidden.as_entire_binding(),
                    uni.resource(&cd),
                ],
                wg(h),
            );
        }

        // Post-embed hidden (the layer-loop input) for the megakernel embed-handoff diff.
        if std::env::var("ARF_EMBED_DUMP").ok().as_deref() == Some(&pos.to_string()) {
            cp.submit();
            cp = CommandPass::new(&self.ctx);
            let hv = self.ctx.read_f32(&self.scratch.hidden, h);
            eprintln!(
                "[post-embed pos={pos}] hidden ||={:.3} first4={:?}",
                hv.iter().map(|x| x * x).sum::<f32>().sqrt(),
                &hv[..4.min(hv.len())]
            );
        }

        // P15 bind-group cache: only in the steady decode shape (token == None,
        // m == 1). The matmul dispatch sequence is then identical every step, so
        // each call-site's bind group caches at a stable slot. Prefill (token =
        // Some, m may be > 1) and any other shape pass `None` and rebuild — their
        // slots would not line up with decode's. Held across the layer loop.
        let mut binds_guard = self.decode_binds.borrow_mut();
        let mut cursor = if token.is_none() {
            Some(BindCursor::new(&mut binds_guard))
        } else {
            None
        };

        // WHOLE-TOKEN MEGAKERNEL (macOS + ARF_MEGAKERNEL): run ALL layers + final_norm in
        // ONE island command buffer (1 cross-queue fence vs the per-layer FFN's 48). The embed
        // is already in the shared `hidden`. On success, jump straight to the want_logits tail
        // with `normed` holding the post-final-norm vector (skip the tail's own final_norm).
        #[cfg(target_os = "macos")]
        let mut mega_did_final_norm = false;
        #[cfg(target_os = "macos")]
        let mut mega_did_lm_head = false;
        #[cfg(target_os = "macos")]
        let mut mega_did_sample = false;
        // ARF_MEGAKERNEL_LAYERS=N caps the megakernel to the first N layers (debug binary
        // search); the wgpu loop resumes at layer N. 0 / unset = all 48 + final_norm.
        //
        // ⚠️ NOT A PERF PROBE (measured 2026-08-03, burned a run). Capping sets `full=false`,
        // which DISABLES island_sample (and pushes the remaining layers onto the slow wgpu
        // loop), so a capped run measures a DIFFERENT ENGINE, not fewer layers. Observed
        // nonsense: 48L=91.7ms, 24L=14.7ms, 12L=12196.8ms per 8-token burst.
        // For a per-layer/fixed-cost split at B=1 there is currently NO valid knob — the
        // batched path's ARF_BATCH_MEGA_NLAYER is the one that cleanly caps work, and it
        // does not apply here.
        #[cfg(target_os = "macos")]
        let mut mega_layers_done = 0usize;
        #[cfg(target_os = "macos")]
        if crate::weights::megakernel_on() {
            let prof = std::env::var_os("ARF_MEGAKERNEL_PROF").is_some();
            let t_flush = std::time::Instant::now();
            cursor = None;
            // On the FULLY single-queue feedback path (SINGLEQ + token==None = the island runs
            // embed from the prev on-GPU sample), there is NO wgpu work queued before the island
            // → the flush + blocking poll(wait_indefinitely) is pure waste, and worse: it
            // PERIODICALLY hits accumulated wgpu deferred-upload work → ~15ms spikes every few
            // tokens (the profnw spikes). Skip it on that path. The wgpu embed path still drains.
            let sq_feedback = std::env::var_os("ARF_MEGA_SINGLEQ").is_some() && token.is_none();
            if !sq_feedback {
                cp.flush();
                // BOUNDED wait — see the ffn_dispatch fence above: wait_indefinitely here deadlocks
                // the B=1 MoE first token at the wgpu<->Metal boundary. 10s timeout breaks it.
                let _ = self.ctx.device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(std::time::Duration::from_secs(10)),
                });
            }
            if prof {
                eprintln!(
                    "[mega phase] flush+poll(embed drain) {:.3} ms",
                    t_flush.elapsed().as_secs_f64() * 1000.0
                );
            }
            let t_mega = std::time::Instant::now();
            // Pre-layers hidden = the embed output the megakernel consumes (isolates the embed
            // handoff on tokens ≥1, where the input is the prev sample via on-GPU feedback).
            if std::env::var("ARF_MEGA_EMBED_DUMP").ok().as_deref() == Some(&pos.to_string()) {
                let hv = self.ctx.read_f32(&self.scratch.hidden, h);
                eprintln!(
                    "[mega pre-layers pos={pos}] hidden ||={:.3} first4={:?}",
                    hv.iter().map(|x| x * x).sum::<f32>().sqrt(),
                    &hv[..4.min(hv.len())]
                );
            }
            let cap = std::env::var("ARF_MEGAKERNEL_LAYERS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0usize);
            let n = if cap == 0 {
                self.layers.len()
            } else {
                cap.min(self.layers.len())
            };
            let full = n == self.layers.len();
            // SINGLE-QUEUE (ARF_MEGA_SINGLEQ): island runs embed (steady feedback: token==None)
            // + greedy argmax (want_logits + sampling==None) so the whole token is one queue.
            // The wgpu embed (above) + lm_head/softcap/sample (tail) are skipped accordingly.
            let singleq = std::env::var_os("ARF_MEGA_SINGLEQ").is_some();
            let island_embed = singleq && token.is_none();
            let island_sample = singleq && full && want_logits && sampling.is_none();
            #[cfg(target_os = "macos")]
            if singleq && std::env::var_os("ARF_SINGLEQ_DEBUG").is_some() {
                eprintln!("[singleq flags pos={pos}] token.is_none={} full={full} want_logits={want_logits} sampling.is_none={} → embed={island_embed} sample={island_sample}",
                    token.is_none(), sampling.is_none());
            }
            // out_tokens store slot (the +1 off-by-one): the island argmax PRODUCES the
            // next-step token. Steady decode (token==None, collect_slot=i) → out_tokens[i+1].
            // Prefill last token (token=Some, want_logits) → seeds out_tokens[0] = T_0.
            // out_tokens has max_positions entries (> max_tokens) so slot i+1 never overflows;
            // the unread tail slot is harmless. Only on the island_sample path.
            let out_slot: Option<u64> = if island_sample {
                match token {
                    None => collect_slot.map(|i| i + 1),
                    Some(_) => Some(0),
                }
            } else {
                None
            };
            // MULTI-TOKEN (ARF_MEGA_KTOK): K>1 only on the greedy island_sample feedback path
            // (token==None, full, want_logits, no sampling). Anything else stays K=1 (byte-
            // identical). The caller (decode_tokens_k) clamps K to the tokens remaining.
            let mega_k = if island_sample && token.is_none() {
                decode_k.max(1)
            } else {
                1
            };
            if self.run_megakernel(
                pos,
                pos + 1,
                n,
                full,
                island_embed,
                island_sample,
                out_slot,
                mega_k,
            ) {
                mega_layers_done = n;
                mega_did_final_norm = full;
                #[cfg(target_os = "macos")]
                if island_sample && std::env::var_os("ARF_SINGLEQ_DEBUG").is_some() {
                    let wv = self.ctx.read_u32(&self.scratch.next_token, 1)[0];
                    let mv = self.scratch.next_token_mtl.as_ref().map(|m| {
                        use objc2_metal::MTLBuffer as _;
                        unsafe { *(m.contents().as_ptr() as *const u32) }
                    });
                    eprintln!("[singleq pos={pos}] next_token wgpu={wv} mtl={mv:?} (island argmax wrote it)");
                }
                #[cfg(target_os = "macos")]
                {
                    mega_did_sample = island_sample;
                }
                // The island ran lm_head (Q4KS→logits) iff it did final_norm + the transcode +
                // shared logits exist → the tail skips its bf16 lm_head GEMV.
                #[cfg(target_os = "macos")]
                {
                    mega_did_lm_head =
                        full && self.mega.lm_head.is_some() && self.scratch.logits_mtl.is_some();
                }
                if prof {
                    eprintln!(
                        "[mega phase] island(48L+wait) {:.3} ms",
                        t_mega.elapsed().as_secs_f64() * 1000.0
                    );
                }
                // KV the megakernel just scattered at THIS pos's slot, layer 0 — the smoking
                // gun for whether the megakernel writes correct K/V (pos=10 attends slot 9).
                if std::env::var("ARF_MEGA_KV_DUMP").ok().as_deref() == Some(&pos.to_string()) {
                    let kvd = self.layers[0].kv_dim();
                    let kr = self.ctx.read_f32_at(
                        &self.kv.keys[0],
                        pos as u64 * kvd as u64 * 4,
                        kvd.min(6),
                    );
                    eprintln!(
                        "[mega KV pos={pos} L0 slot{pos} via WGPU-view] K6={:?}",
                        &kr[..6.min(kr.len())]
                    );
                    // Direct MTL-view read (bypasses wgpu readback) — distinguishes "scatter
                    // wrote zeros" from "wgpu read doesn't see the MTL write".
                    use objc2_metal::MTLBuffer as _;
                    if let Some(km) = self.kv.keys_mtl.first().and_then(|o| o.as_ref()) {
                        unsafe {
                            let p = km.contents().as_ptr() as *const f32;
                            let off = pos * kvd;
                            let direct: Vec<f32> = (0..6).map(|i| *p.add(off + i)).collect();
                            // also slot 0 (where the OLD buggy full-slots bind would have written).
                            let s0: Vec<f32> = (0..6).map(|i| *p.add(i)).collect();
                            eprintln!("[mega KV pos={pos} L0 slot{pos} via MTL-view] K6={:?} | slot0 K6={:?}", direct, s0);
                        }
                    }
                    // Is scratch.k (the scatter INPUT) even non-zero?
                    if let Some(km) = self.scratch.k_mtl.as_ref() {
                        unsafe {
                            let p = km.contents().as_ptr() as *const f32;
                            let kin: Vec<f32> = (0..6).map(|i| *p.add(i)).collect();
                            eprintln!("[mega scratch.k (scatter input)] K6={:?}", kin);
                        }
                    }
                }
                let dump = std::env::var("ARF_MEGAKERNEL_DUMP").ok();
                if dump.as_deref() == Some("1") || dump.as_deref() == Some(&pos.to_string()) {
                    let hv = self.ctx.read_f32(&self.scratch.hidden, h);
                    eprintln!(
                        "[mega pos={pos} after {n}L] ||h||={:.3} first3={:?}",
                        hv.iter().map(|x| x * x).sum::<f32>().sqrt(),
                        &hv[..3.min(hv.len())]
                    );
                    // per-op intermediates of the LAST megakernel layer (for op-localization).
                    let nv = self.ctx.read_f32(&self.scratch.normed, h);
                    let qv = self.ctx.read_f32(&self.scratch.q, 8);
                    let av = self.ctx.read_f32(&self.scratch.attn_out, h);
                    let mdv = self.ctx.read_f32(&self.scratch.mlp_down, h);
                    eprintln!(
                        "[mega ops] normed4={:?} q4(post-rope)={:?} attn_out4={:?} mlp_down4={:?}",
                        &nv[..4],
                        &qv[..4],
                        &av[..4],
                        &mdv[..4]
                    );
                }
            } else if std::env::var("ARF_MEGAKERNEL_STRICT").is_ok() {
                panic!("megakernel requested but preconditions not met (need ARF_MSL_GEMV + gemma-4 Q4KS)");
            }
        }

        // 8-BIT KV (`kv_q8_for`): the f32 pool this generic loop binds is a 16-byte PLACEHOLDER —
        // the cache lives in `kv.q8`, which only the island record reads. Reaching here means an
        // island path bailed; attending the placeholder is fluent garbage or a hung queue
        // (2026-09-23: an Err raised mid-record by the shared-prefix verify hung a request for
        // 900 s). Say why, loudly, instead.
        #[cfg(target_os = "macos")]
        if self.kv.q8.iter().any(Option::is_some) {
            panic!(
                "[kv] 8-bit KV cache is live but this step fell through to the generic f32 layer \
                 loop, which cannot read it (b={}, q_lens={:?}; single-token decode loop). Rerun with \
                 ARF_BATCH_MEGA_DEBUG=1 for the bail reason; ARF_NO_KV_Q8=1 serves f32.",
                1usize,
                [1usize]
            );
        }
        for (li, layer) in self.layers.iter().enumerate() {
            // The megakernel already ran layers [0, mega_layers_done); wgpu does the rest.
            #[cfg(target_os = "macos")]
            if li < mega_layers_done {
                continue;
            }
            // ARF_DUMP_NORM=1: per-layer ||hidden|| trajectory (one line/layer) to
            // pinpoint where a bad model diverges — a global layer (head_dim 512) that
            // blows up or collapses the norm localizes the gemma-4-12B garbage bug.
            if std::env::var("ARF_DUMP_NORM").is_ok() {
                cursor = None;
                cp.submit();
                cp = CommandPass::new(&self.ctx);
                let hv = self.ctx.read_f32(&self.scratch.hidden, h);
                let n2: f32 = hv.iter().map(|x| x * x).sum::<f32>().sqrt();
                let is_global = layer.window.is_none();
                eprintln!(
                    "[L{li}{}] ||hidden||={n2:.3} hd={} first3={:?}",
                    if is_global { " GLOBAL" } else { "" },
                    layer.head_dim,
                    &hv[..3.min(hv.len())]
                );
            }
            // Per-layer attention geometry (Gemma 4 global layers differ from the
            // base/sliding ones). All q/k/v/o projection dims, the qk_norm length,
            // the rope rotary dim, the kv_scatter stride and the attention uniforms
            // derive from these. For every other model these equal the cfg scalars.
            let hd = layer.head_dim;
            let nkv = layer.kv_heads;
            let q_dim = layer.q_dim(nh);
            let kv_dim = layer.kv_dim();
            let group = nh / nkv;
            // Gemma overrides the pre-softmax query scale (`query_pre_attn_scalar`);
            // everyone else uses 1/√head_dim (per-layer head_dim). Matches the CPU.
            let scale = mc.query_pre_attn_scalar.unwrap_or(1.0 / (hd as f32).sqrt());
            // ARF_DUMP_FULL=<layer>: pre-input_norm hidden (= embedding for L0),
            // first-4 — to diff vs batched FB_OPS "PRE-norm hidden4".
            if std::env::var("ARF_DUMP_FULL").ok().as_deref() == Some(&li.to_string()) {
                cursor = None;
                cp.submit();
                cp = CommandPass::new(&self.ctx);
                let hv = self.ctx.read_f32(&self.scratch.hidden, h);
                eprintln!("[L{li} PRE-norm hidden4]={:?}", &hv[..4.min(hv.len())]);
            }
            // --- attention block: normed = rmsnorm(hidden) ---
            // `nd` (norm dims) is identical for the in- and post-attention norms,
            // so it is allocated once and bound by both passes.
            let nd = uni.alloc_write(&dims_norm(h, mc.rms_norm_eps)).expect("nd");
            // Cacheable: binds only resident buffers (hidden/normed scratch, the
            // layer norm weight) + a fixed-offset uniform — all byte-identical every
            // token. (kv_scatter/attention below are NOT cached: they bind
            // ctx_len-growing arena regions.)
            cp.add_maybe_cached(
                cursor.as_mut(),
                &k.rmsnorm,
                "in_norm",
                &[
                    self.scratch.hidden.as_entire_binding(),
                    layer.input_norm.as_entire_binding(),
                    self.scratch.normed.as_entire_binding(),
                    uni.resource(&nd),
                ],
                [1, 1, 1],
            );

            // ARF_DUMP_FULL=<layer>: post-input_norm `normed` (input to q_proj),
            // first-4 — to diff vs batched FB_OPS "normed4".
            if std::env::var("ARF_DUMP_FULL").ok().as_deref() == Some(&li.to_string()) {
                cursor = None;
                cp.submit();
                cp = CommandPass::new(&self.ctx);
                let nv = self.ctx.read_f32(&self.scratch.normed, h);
                eprintln!("[L{li} normed4]={:?}", &nv[..4.min(nv.len())]);
            }
            // q/k/v = normed · Wᵀ
            add_matmul(
                &mut cp,
                k,
                &mut *uni,
                self.scratch.normed.as_entire_binding(),
                &layer.q_proj,
                self.scratch.q.as_entire_binding(),
                q_dim,
                h,
                cursor.as_mut(),
            );
            add_matmul(
                &mut cp,
                k,
                &mut *uni,
                self.scratch.normed.as_entire_binding(),
                &layer.k_proj,
                self.scratch.k.as_entire_binding(),
                kv_dim,
                h,
                cursor.as_mut(),
            );
            add_matmul(
                &mut cp,
                k,
                &mut *uni,
                self.scratch.normed.as_entire_binding(),
                &layer.v_proj,
                self.scratch.v.as_entire_binding(),
                kv_dim,
                h,
                cursor.as_mut(),
            );

            // Global-layer RAW K + V (pre-qk_norm/v_norm) — V=K-clone so they must be
            // identical; compare both to llama Kcur-5/Vcur-5=[0.5143,-0.0711,-0.6318].
            if li == 5 && std::env::var("ARF_DUMP_GLOBAL").is_ok() {
                cursor = None;
                cp.submit();
                cp = CommandPass::new(&self.ctx);
                let kv = self.ctx.read_f32(&self.scratch.k, kv_dim);
                let vv = self.ctx.read_f32(&self.scratch.v, kv_dim);
                eprintln!(
                    "[L5 raw K/V] k4={:?}\n             v4={:?}",
                    &kv[..4.min(kv.len())],
                    &vv[..4.min(vv.len())]
                );
            }

            // FUSED q/k/v per-head RMSNorm (one dispatch) when the model has q/k norm
            // AND weightless v-norm (Gemma 4) and ARF_QKV_NORM is on — replaces the
            // qk_norm + v_norm pair below, cutting one dispatch/layer. Bit-identical:
            // q/k rows weighted, v rows weightless, same per-row RMS as the split path.
            let fuse_qkv =
                qkv_norm() && mc.value_norm && layer.q_norm.is_some() && layer.k_norm.is_some();
            if fuse_qkv {
                let (qn, kn) = (
                    layer.q_norm.as_ref().unwrap(),
                    layer.k_norm.as_ref().unwrap(),
                );
                let qkvd = uni
                    .alloc_write(&[
                        nh as u32,  // q_rows
                        nkv as u32, // k_rows
                        nkv as u32, // v_rows
                        hd as u32,
                        (mc.rms_norm_eps as f32).to_bits(),
                        0,
                        0,
                        0,
                    ])
                    .expect("qkvd");
                cp.add_maybe_cached(
                    cursor.as_mut(),
                    &k.qkv_norm,
                    "qkv_norm",
                    &[
                        self.scratch.q.as_entire_binding(),
                        self.scratch.k.as_entire_binding(),
                        self.scratch.v.as_entire_binding(),
                        qn.as_entire_binding(),
                        kn.as_entire_binding(),
                        uni.resource(&qkvd),
                    ],
                    [(nh + 2 * nkv) as u32, 1, 1],
                );
            }
            // Per-head q/k RMSNorm before RoPE (Qwen3/Gemma); Llama skips it.
            if !fuse_qkv {
                if let (Some(qn), Some(kn)) = (&layer.q_norm, &layer.k_norm) {
                    // q_rows = nh, k_rows = nkv (one token in decode); eps as f32 bits.
                    let qkd = uni
                        .alloc_write(&[
                            nh as u32,
                            nkv as u32,
                            hd as u32,
                            (mc.rms_norm_eps as f32).to_bits(),
                        ])
                        .expect("qkd");
                    cp.add_maybe_cached(
                        cursor.as_mut(),
                        &k.qk_norm,
                        "qk_norm",
                        &[
                            self.scratch.q.as_entire_binding(),
                            self.scratch.k.as_entire_binding(),
                            qn.as_entire_binding(),
                            kn.as_entire_binding(),
                            uni.resource(&qkd),
                        ],
                        [(nh + nkv) as u32, 1, 1],
                    );
                }
            }

            // Global-layer RAW V (pre-v_norm) probe — compare to llama Vcur-5.
            if li == 5 && std::env::var("ARF_DUMP_GLOBAL").is_ok() {
                cursor = None;
                cp.submit();
                cp = CommandPass::new(&self.ctx);
                let vv = self.ctx.read_f32(&self.scratch.v, kv_dim);
                eprintln!("[L5 raw V] v4={:?}", &vv[..4.min(vv.len())]);
            }
            // Gemma 4 ALSO normalizes V per head — weightless RMSNorm (llama.cpp
            // `Vcur_normed = RMS_NORM(Vcur)`, no learned gain, no RoPE). Runs after
            // v_proj, before attention. `nkv` rows of `hd` (one token in decode).
            // Skipped when the fused qkv_norm above already handled v.
            if mc.value_norm && !fuse_qkv {
                let vnd = uni
                    .alloc_write(&[nkv as u32, hd as u32, (mc.rms_norm_eps as f32).to_bits(), 0])
                    .expect("vnd");
                cp.add_maybe_cached(
                    cursor.as_mut(),
                    &k.v_norm,
                    "v_norm",
                    &[self.scratch.v.as_entire_binding(), uni.resource(&vnd)],
                    [nkv as u32, 1, 1],
                );
            }

            // Global-layer attention probe (ARF_DUMP_GLOBAL=1, layer 5 = 1st global):
            // dump q (post-proj, pre-RoPE) and v (post-v_norm) to diff vs llama.cpp
            // Qcur-5 / Vcur_normed-5. Slow (submit+readback); diagnostic only.
            if li == 5 && std::env::var("ARF_DUMP_GLOBAL").is_ok() {
                cursor = None;
                cp.submit();
                cp = CommandPass::new(&self.ctx);
                let qv = self.ctx.read_f32(&self.scratch.q, q_dim);
                let vv = self.ctx.read_f32(&self.scratch.v, kv_dim);
                eprintln!(
                    "[L5 global] q_dim={q_dim} kv_dim={kv_dim} hd={hd} nkv={nkv}\n  q4={:?}\n  v_normed4={:?}",
                    &qv[..4.min(qv.len())],
                    &vv[..4.min(vv.len())]
                );
            }

            // Fused RoPE on q and k (in place), one dispatch instead of two. Table
            // selection per layer: LOCAL (sliding-window) layers rotate with the
            // local base; GLOBAL layers with per-layer partial rotary (Gemma 4) use
            // the dedicated global table; everything else uses the default tables.
            // The per-layer window/geometry is fixed across tokens, so each layer's
            // rope_qk site captures the same table every step — the P15 cache stays
            // valid. `rotary_dim` (= layer.rotary_dim) gates partial rotation: only
            // the first rotary_dim head dims rotate (Gemma 4 global = 128 of 512).
            let (rope_cos, rope_sin) = match (
                layer.window,
                &self.rope_cos_local,
                &self.rope_sin_local,
                &self.rope_cos_global,
                &self.rope_sin_global,
            ) {
                // Local/sliding layer with a local base.
                (Some(_), Some(c), Some(s), _, _) => (c, s),
                // Global layer (window == None) with a dedicated global partial table.
                (None, _, _, Some(c), Some(s)) => (c, s),
                _ => (&self.rope_cos, &self.rope_sin),
            };
            let rd = uni
                .alloc_write(&[
                    1u32,
                    nh as u32,
                    nkv as u32,
                    hd as u32,
                    layer.rotary_dim as u32,
                    0,
                    0,
                    0,
                ])
                .expect("rd");
            // ARF_DUMP_FULL=<layer>: PRE-RoPE q/k (post q_proj), first-4 — to diff
            // vs batched FB_OPS "PRE-rope". normed matches both paths, so a divergence
            // here = the q_proj/k_proj matmul (m=1 vs m>1) on the real GGUF weights.
            if std::env::var("ARF_DUMP_FULL").ok().as_deref() == Some(&li.to_string()) {
                cursor = None;
                cp.submit();
                cp = CommandPass::new(&self.ctx);
                let qv = self.ctx.read_f32(&self.scratch.q, q_dim);
                let kv = self.ctx.read_f32(&self.scratch.k, kv_dim);
                eprintln!(
                    "[L{li} PRE-rope q4]={:?} k4={:?}",
                    &qv[..4.min(qv.len())],
                    &kv[..4.min(kv.len())]
                );
            }
            // Cacheable: pos_buf is allocated FIRST in idx (fixed offset 0, fixed
            // size) precisely so this dispatch's bind group can be cached — see the
            // arena alloc-order note where pos_buf/new_slot_buf precede slots_buf.
            // ROPE PAIRING IS PER-ARCH. muse-glimmer's GGUF stores q/k in INTERLEAVED (ggml
            // NORM) order; every other model we ship is split-half (NEOX). Wrong pairing does
            // not error — it scrambles every q and k, which is why this must be explicit.
            let (rope_pso, rope_label) = if mc.rope_interleaved() {
                (&k.rope_qk_interleaved, "rope_qk_interleaved")
            } else {
                (&k.rope_qk, "rope_qk")
            };
            cp.add_maybe_cached(
                cursor.as_mut(),
                rope_pso,
                rope_label,
                &[
                    self.scratch.q.as_entire_binding(),
                    self.scratch.k.as_entire_binding(),
                    rope_cos.as_entire_binding(),
                    rope_sin.as_entire_binding(),
                    idx.resource(&pos_buf),
                    uni.resource(&rd),
                ],
                wg(q_dim + kv_dim),
            );

            // Post-RoPE Q/K dump for the per-channel diff vs llama Qcur_pos-N/Kcur_pos-N.
            // Holds the LAST prefilled token row. Isolates proj+qknorm+rope from combine.
            if std::env::var("ARF_DUMP_FULL").ok().as_deref() == Some(&li.to_string()) {
                cursor = None;
                cp.submit();
                cp = CommandPass::new(&self.ctx);
                let qv = self.ctx.read_f32(&self.scratch.q, q_dim);
                let kvv = self.ctx.read_f32(&self.scratch.k, kv_dim);
                std::fs::write(
                    format!("/tmp/ours_q_L{li}.txt"),
                    qv.iter().map(|v| format!("{v:.6}\n")).collect::<String>(),
                )
                .ok();
                std::fs::write(
                    format!("/tmp/ours_k_L{li}.txt"),
                    kvv.iter().map(|v| format!("{v:.6}\n")).collect::<String>(),
                )
                .ok();
            }
            // scatter new k/v into the pool; attention then reads the pool directly
            // by slot (no kv_gather copy into k_all/v_all). The scatter (write) and
            // attention (read) touch the same pool buffers but in DISTINCT dispatches
            // of this pass — a read-after-write, which wgpu allows (the forbidden case
            // is read+read_write of one buffer within a SINGLE dispatch).
            if self.kv.kv_quant.is_quantized() {
                let bits = self.kv.kv_quant.bits() as u32;
                let sd = uni
                    .alloc_write(&[1u32, nkv as u32, hd as u32, 0, bits, 0, 0, 0])
                    .expect("sd_tq");
                cp.add_bound(
                    &k.kv_scatter_tq,
                    "kv_scatter_tq",
                    &[
                        self.scratch.k.as_entire_binding(),
                        self.scratch.v.as_entire_binding(),
                        idx.resource(&new_slot_buf),
                        self.kv.keys[li].as_entire_binding(),
                        self.kv.values[li].as_entire_binding(),
                        self.kv.key_norms[li].as_entire_binding(),
                        self.kv.value_norms[li].as_entire_binding(),
                        self.kv_levels.as_ref().unwrap().as_entire_binding(),
                        uni.resource(&sd),
                    ],
                    [(nkv as u32).div_ceil(64), 1, 1], // q_len(=1)·kv_heads vectors
                );
            } else {
                let sd = uni.alloc_write(&[1u32, kv_dim as u32, 0, 0]).expect("sd");
                cp.add_bound(
                    &k.kv_scatter,
                    "kv_scatter",
                    &[
                        self.scratch.k.as_entire_binding(),
                        self.scratch.v.as_entire_binding(),
                        idx.resource(&new_slot_buf),
                        self.kv.keys[li].as_entire_binding(),
                        self.kv.values[li].as_entire_binding(),
                        uni.resource(&sd),
                    ],
                    wg(kv_dim),
                );
            }
            // Decode-time WGSL KV at the just-scattered slot (layer 0) — reference for the
            // megakernel KV diff. Fires for both this WGSL path and via the same env.
            if li == 0
                && std::env::var("ARF_MEGA_KV_DUMP").ok().as_deref() == Some(&pos.to_string())
            {
                cursor = None;
                cp.submit();
                cp = CommandPass::new(&self.ctx);
                let kvd = kv_dim;
                let kr =
                    self.ctx
                        .read_f32_at(&self.kv.keys[0], pos as u64 * kvd as u64 * 4, kvd.min(6));
                let vr = self.ctx.read_f32_at(
                    &self.kv.values[0],
                    pos as u64 * kvd as u64 * 4,
                    kvd.min(6),
                );
                eprintln!(
                    "[WGSL KV pos={pos} L0 slot{pos}] K6={:?} V6={:?}",
                    &kr[..6.min(kr.len())],
                    &vr[..6.min(vr.len())]
                );
            }

            // KV-pool slot probe: at the LAST prefilled token, read back each cached
            // slot's K/V per-kv-head norm to check for cross-token pool corruption.
            // Expected (gemma L0): every K row per-head ||1.953||, V per-head ||16.000||.
            if std::env::var("ARF_DUMP_FULL").ok().as_deref() == Some(&li.to_string()) {
                cursor = None;
                cp.submit();
                cp = CommandPass::new(&self.ctx);
                let rf = (nkv * hd) as u64;
                for s in 0..ctx_len.min(4) {
                    let kr = self
                        .ctx
                        .read_f32_at(&self.kv.keys[li], s as u64 * rf * 4, nkv * hd);
                    let vr = self
                        .ctx
                        .read_f32_at(&self.kv.values[li], s as u64 * rf * 4, nkv * hd);
                    let l2 = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / nkv as f32).sqrt();
                    eprintln!(
                        "[L{li} pool slot{s}] K/head={:.3} V/head={:.3}",
                        l2(&kr),
                        l2(&vr)
                    );
                    std::fs::write(
                        format!("/tmp/ours_poolK_L{li}_s{s}.txt"),
                        kr.iter().map(|v| format!("{v:.6}\n")).collect::<String>(),
                    )
                    .ok();
                    std::fs::write(
                        format!("/tmp/ours_poolV_L{li}_s{s}.txt"),
                        vr.iter().map(|v| format!("{v:.6}\n")).collect::<String>(),
                    )
                    .ok();
                }
                eprintln!("[L{li} attn dims] scale_bits={:08x} scale={} nh={nh} nkv={nkv} hd={hd} group={group} ctx_len={ctx_len} pos={pos} window={}", scale.to_bits(), scale, layer.window.unwrap_or(0));
            }
            // fused attention -> attn (Gemma local layers mask to the last `window`).
            // Reads the paged KV pool in place by slot (FlashInfer-style in-kernel
            // gather): binding 1 = key_pool, 2 = value_pool, 4 = slots.
            let mut adims = dims_attn(
                ctx_len,
                nh,
                nkv,
                hd,
                pos,
                scale,
                group,
                layer.window.unwrap_or(0),
            );
            if self.kv.kv_quant.is_quantized() {
                adims[10] = self.kv.kv_quant.bits() as u32; // bits slot (f32 path: pad)
                let ad = uni.alloc_write(&adims).expect("ad_tq");
                cp.add_bound(
                    &k.attention_tq,
                    "attention_tq",
                    &[
                        self.scratch.q.as_entire_binding(),
                        self.kv.keys[li].as_entire_binding(),
                        self.kv.values[li].as_entire_binding(),
                        out.resource(&attn),
                        idx.resource(&slots_buf),
                        uni.resource(&ad),
                        self.kv.key_norms[li].as_entire_binding(),
                        self.kv.value_norms[li].as_entire_binding(),
                        self.kv_levels.as_ref().unwrap().as_entire_binding(),
                    ],
                    [nh as u32, 1, 1],
                );
            } else if let Some(parts) = &splitk_partials {
                // Long ctx, f32 KV: split-KV (flash-decoding) to stay under the GPU
                // watchdog. adims[10] carries `splitk` (the f32 path pads it). Pass 1:
                // SPLITK workgroups write partials; pass 2 combines → attn.
                let mut sd = adims;
                sd[10] = SPLITK;
                let ad = uni.alloc_write(&sd).expect("ad_splitk");
                cp.add_bound(
                    &k.attention_splitk,
                    "attention_splitk",
                    &[
                        self.scratch.q.as_entire_binding(),
                        self.kv.keys[li].as_entire_binding(),
                        self.kv.values[li].as_entire_binding(),
                        out.resource(parts),
                        idx.resource(&slots_buf),
                        uni.resource(&ad),
                    ],
                    [nh as u32, SPLITK, 1],
                );
                cp.add_bound(
                    &k.attention_splitk_combine,
                    "attention_splitk_combine",
                    &[out.resource(parts), out.resource(&attn), uni.resource(&ad)],
                    [nh as u32, 1, 1],
                );
            } else {
                let ad = uni.alloc_write(&adims).expect("ad");
                cp.add_bound(
                    &k.attention,
                    "attention",
                    &[
                        self.scratch.q.as_entire_binding(),
                        self.kv.keys[li].as_entire_binding(),
                        self.kv.values[li].as_entire_binding(),
                        out.resource(&attn),
                        idx.resource(&slots_buf),
                        uni.resource(&ad),
                    ],
                    [nh as u32, 1, 1],
                );
            }

            // Raw attention output (pre o_proj) full dump for the diff vs llama
            // kqv_out-N. Holds the LAST prefilled token row. q_dim channels.
            if std::env::var("ARF_DUMP_FULL").ok().as_deref() == Some(&li.to_string()) {
                cursor = None;
                cp.submit();
                cp = CommandPass::new(&self.ctx);
                let av = self.ctx.read_f32_at(out.buffer(), attn.offset(), q_dim);
                std::fs::write(
                    format!("/tmp/ours_kqv_L{li}.txt"),
                    av.iter().map(|v| format!("{v:.6}\n")).collect::<String>(),
                )
                .ok();
            }

            // MUSE GLIMMER ATTENTION GATE — `attn *= sigmoid(attn_gate @ normed)`, between SDPA
            // and o_proj (llama.cpp src/models/muse-glimmer.cpp:107,137-139). The gate projects
            // the PRE-attention hidden state (`normed`, the same input q/k/v saw), NOT the
            // attention output. `None` for every other arch, so this is a no-op there.
            //
            // Reuses `scratch.gate`, which is sized for max(intermediate, moe_inter) — 19968 for
            // this model vs a q_dim of 4096, so it comfortably holds the gate projection.
            if let Some(gate_w) = layer.attn_gate.as_ref() {
                add_matmul(
                    &mut cp,
                    k,
                    &mut *uni,
                    self.scratch.normed.as_entire_binding(),
                    gate_w,
                    self.scratch.gate.as_entire_binding(),
                    q_dim,
                    h,
                    cursor.as_mut(),
                );
                let gd = uni
                    .alloc_write(&[q_dim as u32, 0, 0, 0])
                    .expect("attn_gate dims");
                cp.add_maybe_cached(
                    cursor.as_mut(),
                    &k.attn_gate_mul,
                    "attn_gate_mul",
                    &[
                        out.resource(&attn),
                        self.scratch.gate.as_entire_binding(),
                        uni.resource(&gd),
                    ],
                    [(q_dim as u32).div_ceil(256), 1, 1],
                );
            }
            // o_proj then residual: hidden += o_proj(attn)
            add_matmul(
                &mut cp,
                k,
                &mut *uni,
                out.resource(&attn),
                &layer.o_proj,
                self.scratch.attn_out.as_entire_binding(),
                h,
                q_dim,
                cursor.as_mut(),
            );
            // Per-layer attention CONTRIBUTION (o_proj output) for the wash-out probe:
            // is the global layer's attention output input-DEPENDENT? (ARF_DUMP_ATTNOUT)
            // Per-layer attention output (post o_proj, pre post-norm/residual) full dump
            // for the per-channel diff vs llama. Holds the LAST prefilled token row.
            if std::env::var("ARF_DUMP_FULL").ok().as_deref() == Some(&li.to_string()) {
                cursor = None;
                cp.submit();
                cp = CommandPass::new(&self.ctx);
                let av = self.ctx.read_f32(&self.scratch.attn_out, h);
                let nv = self.ctx.read_f32(&self.scratch.normed, h);
                std::fs::write(
                    format!("/tmp/ours_oproj_L{li}.txt"),
                    av.iter().map(|v| format!("{v:.6}\n")).collect::<String>(),
                )
                .ok();
                std::fs::write(
                    format!("/tmp/ours_attnnorm_L{li}.txt"),
                    nv.iter().map(|v| format!("{v:.6}\n")).collect::<String>(),
                )
                .ok();
            }
            if std::env::var("ARF_DUMP_ATTNOUT").is_ok() && (li == 5 || li == 10 || li == 11) {
                cursor = None;
                cp.submit();
                cp = CommandPass::new(&self.ctx);
                // attn_out = o_proj(attn); hidden still holds the pre-attn residual.
                let av = self.ctx.read_f32(&self.scratch.attn_out, h);
                let hv = self.ctx.read_f32(&self.scratch.hidden, h);
                let (ai, avv) = av
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
                    .map(|(i, &v)| (i, v))
                    .unwrap_or((0, 0.0));
                eprintln!(
                    "[L{li} attn_out] argmax=ch{ai}={avv:.3} ch1750={:.4} | hidden(pre-attn) ch1750={:.4}",
                    av.get(1750).copied().unwrap_or(0.0),
                    hv.get(1750).copied().unwrap_or(0.0)
                );
            }
            // --- MLP block ---
            if is_gemma {
                // Gemma's four-norm block: a post-norm on the attention output
                // applied BEFORE the residual add, then a separate pre-MLP norm.
                //   hidden += rmsnorm(attn_out, post_attention_layernorm)  [fused]
                //   normed = rmsnorm(hidden, pre_feedforward_layernorm)
                cp.add_maybe_cached(
                    cursor.as_mut(),
                    &k.rmsnorm_add,
                    "post_attn_norm",
                    &[
                        self.scratch.attn_out.as_entire_binding(),
                        layer.post_norm.as_entire_binding(),
                        self.scratch.hidden.as_entire_binding(),
                        uni.resource(&nd),
                    ],
                    [1, 1, 1],
                );
                // ARF_DUMP_FULL=<layer>: write post-attn hidden (= llama attn_out-N).
                if std::env::var("ARF_DUMP_FULL").ok().as_deref() == Some(&li.to_string()) {
                    cursor = None;
                    cp.submit();
                    cp = CommandPass::new(&self.ctx);
                    let hv = self.ctx.read_f32(&self.scratch.hidden, h);
                    std::fs::write(
                        format!("/tmp/ours_attn_out_L{li}.txt"),
                        hv.iter().map(|v| format!("{v:.6}\n")).collect::<String>(),
                    )
                    .ok();
                }
                let pre = layer.pre_ffn_norm.as_ref().expect("gemma pre_ffn_norm");
                cp.add_maybe_cached(
                    cursor.as_mut(),
                    &k.rmsnorm,
                    "pre_ffn_norm",
                    &[
                        self.scratch.hidden.as_entire_binding(),
                        pre.as_entire_binding(),
                        self.scratch.normed.as_entire_binding(),
                        uni.resource(&nd),
                    ],
                    [1, 1, 1],
                );
            } else {
                // Llama/Qwen: fused residual-add + pre-MLP norm in one dispatch
                // (hidden += attn_out; normed = rmsnorm(hidden)).
                cp.add_maybe_cached(
                    cursor.as_mut(),
                    &k.add_norm,
                    "post_norm",
                    &[
                        self.scratch.hidden.as_entire_binding(),
                        self.scratch.attn_out.as_entire_binding(),
                        layer.post_norm.as_entire_binding(),
                        self.scratch.normed.as_entire_binding(),
                        uni.resource(&nd),
                    ],
                    [1, 1, 1],
                );
            }
            match &layer.mlp {
                GpuMlp::Dense(dense) => {
                    let GpuDenseMlp {
                        gate_proj,
                        up_proj,
                        down_proj,
                        ..
                    } = dense.as_ref();
                    // DEBUG: ARF_GATE_USES_UP makes the gate matmul use the up_proj
                    // weight. If gate-output then == up-output, the gate_proj weight
                    // OBJECT is the problem (kernel/input/buffer all fine).
                    let gate_w = if std::env::var("ARF_GATE_USES_UP").is_ok() {
                        up_proj
                    } else {
                        gate_proj
                    };

                    // NATIVE-METAL FFN ISLAND (macOS + ARF_MSL_GEMV): run gate+up+geglu+
                    // down (the frame-dominant Q4_K_S GEMVs at 86-97% roofline) in ONE island
                    // command buffer over the shared scratch, then resume wgpu for the small
                    // post-FFN rmsnorm_add (reads the shared mlp_down). Only when the island
                    // exists, the weights are native Q4KS (carry .mtl), and the scratch is
                    // dual-view. Else falls through to the portable wgpu path (the oracle).
                    #[cfg(target_os = "macos")]
                    let did_island_ffn =
                        self.try_island_ffn(gate_w, up_proj, down_proj, &mut cp, &mut cursor);
                    #[cfg(not(target_os = "macos"))]
                    let did_island_ffn = false;

                    if !did_island_ffn {
                        add_matmul(
                            &mut cp,
                            k,
                            &mut *uni,
                            self.scratch.normed.as_entire_binding(),
                            gate_w,
                            self.scratch.gate.as_entire_binding(),
                            mc.intermediate_size,
                            h,
                            cursor.as_mut(),
                        );
                        add_matmul(
                            &mut cp,
                            k,
                            &mut *uni,
                            self.scratch.normed.as_entire_binding(),
                            up_proj,
                            self.scratch.up.as_entire_binding(),
                            mc.intermediate_size,
                            h,
                            cursor.as_mut(),
                        );
                        // L0 gate/up magnitude probe vs llama (gate 542, up 456, geglu 1002).
                        if std::env::var("ARF_DUMP_FULL").ok().as_deref() == Some(&li.to_string()) {
                            cursor = None;
                            cp.submit();
                            cp = CommandPass::new(&self.ctx);
                            let n = mc.intermediate_size;
                            let gv = self.ctx.read_f32(&self.scratch.gate, n);
                            let uv = self.ctx.read_f32(&self.scratch.up, n);
                            let l2 = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
                            eprintln!(
                                "[L{li} gate/up] ||gate||={:.1} ||up||={:.1} gate4={:?}",
                                l2(&gv),
                                l2(&uv),
                                &gv[..4.min(gv.len())]
                            );
                            // Per-channel dump for the gate/up diff vs llama ffn_gate-N/ffn_up-N.
                            // Overwrites each token row; the FILE holds the LAST prefilled token.
                            std::fs::write(
                                format!("/tmp/ours_ffn_gate_L{li}.txt"),
                                gv.iter().map(|v| format!("{v:.6}\n")).collect::<String>(),
                            )
                            .ok();
                            std::fs::write(
                                format!("/tmp/ours_ffn_up_L{li}.txt"),
                                uv.iter().map(|v| format!("{v:.6}\n")).collect::<String>(),
                            )
                            .ok();
                        }
                        let swd = uni
                            .alloc_write(&[mc.intermediate_size as u32, 0, 0, 0])
                            .expect("swd");
                        cp.add_maybe_cached(
                            cursor.as_mut(),
                            // Gemma's FFN gate is GELU-tanh (GeGLU), not SiLU (SwiGLU).
                            self.gate_kernel(k),
                            "swiglu",
                            &[
                                self.scratch.gate.as_entire_binding(),
                                self.scratch.up.as_entire_binding(),
                                out.resource(&silu),
                                uni.resource(&swd),
                            ],
                            wg(mc.intermediate_size),
                        );
                        add_matmul(
                            &mut cp,
                            k,
                            &mut *uni,
                            out.resource(&silu),
                            down_proj,
                            self.scratch.mlp_down.as_entire_binding(),
                            h,
                            mc.intermediate_size,
                            cursor.as_mut(),
                        );
                    } // end `if !did_island_ffn` — the island path already wrote mlp_down.
                      // ARF_DUMP_FULL=<layer>: write down_proj output (= llama ffn_out-N).
                    if std::env::var("ARF_DUMP_FULL").ok().as_deref() == Some(&li.to_string()) {
                        cursor = None;
                        cp.submit();
                        cp = CommandPass::new(&self.ctx);
                        let dv = self.ctx.read_f32(&self.scratch.mlp_down, h);
                        std::fs::write(
                            format!("/tmp/ours_ffn_out_L{li}.txt"),
                            dv.iter().map(|v| format!("{v:.6}\n")).collect::<String>(),
                        )
                        .ok();
                    }
                }
                GpuMlp::Moe(moe) => {
                    self.add_moe_block(
                        &mut cp,
                        k,
                        &mut *uni,
                        out,
                        &silu,
                        moe,
                        h,
                        wg,
                        cursor.as_mut(),
                    );
                }
            }
            if is_gemma {
                // Gemma: post-FFN norm on the MLP output before the residual add.
                //   hidden += rmsnorm(mlp_down, post_feedforward_layernorm)  [fused]
                let post = layer.post_ffn_norm.as_ref().expect("gemma post_ffn_norm");
                cp.add_maybe_cached(
                    cursor.as_mut(),
                    &k.rmsnorm_add,
                    "post_ffn_norm",
                    &[
                        self.scratch.mlp_down.as_entire_binding(),
                        post.as_entire_binding(),
                        self.scratch.hidden.as_entire_binding(),
                        uni.resource(&nd),
                    ],
                    [1, 1, 1],
                );
            } else {
                add_residual(
                    &mut cp,
                    k,
                    &mut *uni,
                    &self.scratch.hidden,
                    &self.scratch.mlp_down,
                    h,
                );
            }

            // PRE-layer_scalar hidden (ARF_DUMP_PRESCALE): is the input still present
            // BEFORE the tiny L11 scalar? Compare two prompts to see if the sublayers or
            // the scalar destroy the input-dependence.
            if std::env::var("ARF_DUMP_PRESCALE").is_ok() && (li == 10 || li == 11) {
                cursor = None;
                cp.submit();
                cp = CommandPass::new(&self.ctx);
                let hv = self.ctx.read_f32(&self.scratch.hidden, h);
                let l2 = hv.iter().map(|x| x * x).sum::<f32>().sqrt();
                eprintln!(
                    "[L{li} PRE-scalar] ||hidden||={l2:.3} scalar={:?} first4={:?}",
                    layer.layer_scalar,
                    &hv[..4.min(hv.len())]
                );
            }
            // ch1750 source probe (ARF_DUMP_CH1750): llama keeps ch1750 ≡ 0; ours
            // grows a spurious spike. Read each scratch buffer's ch1750 to see which
            // sub-block output (attn_out via o_proj, mlp_down via down_proj) injects it.
            if std::env::var("ARF_DUMP_CH1750").is_ok() {
                cursor = None;
                cp.submit();
                cp = CommandPass::new(&self.ctx);
                let at =
                    |b: &wgpu::Buffer| self.ctx.read_f32(b, h).get(1750).copied().unwrap_or(0.0);
                // RMS of mlp_down (tiny RMS => huge inv_rms in post_ffn rmsnorm_add).
                let md = self.ctx.read_f32(&self.scratch.mlp_down, h);
                let md_rms = (md.iter().map(|x| x * x).sum::<f32>() / h as f32).sqrt();
                let post_w = self
                    .ctx
                    .read_f32(&layer.post_norm, h)
                    .get(1750)
                    .copied()
                    .unwrap_or(0.0);
                let pffw = layer
                    .post_ffn_norm
                    .as_ref()
                    .map(|b| self.ctx.read_f32(b, h).get(1750).copied().unwrap_or(0.0));
                eprintln!(
                    "[L{li} ch1750] hidden={:.4} normed={:.4} attn_out={:.4} mlp_down={:.4} | mlp_down_RMS={md_rms:.4} post_attn_w[1750]={post_w:.4} post_ffn_w[1750]={:?}",
                    at(&self.scratch.hidden),
                    at(&self.scratch.normed),
                    at(&self.scratch.attn_out),
                    at(&self.scratch.mlp_down),
                    pffw
                );
            }
            // Gemma 4: scale the WHOLE layer output by the learned per-layer scalar,
            // at the very end of the layer (after both sublayers + residuals +
            // post-norms — matches HF Gemma4TextDecoderLayer: `hidden *= layer_scalar`).
            if let Some(scale) = layer.layer_scalar {
                let sd = uni
                    .alloc_write(&[h as u32, scale.to_bits(), 0, 0])
                    .expect("layer_scalar dims");
                cp.add_maybe_cached(
                    cursor.as_mut(),
                    &k.scale_inplace,
                    "layer_scalar",
                    &[self.scratch.hidden.as_entire_binding(), uni.resource(&sd)],
                    wg(h),
                );
            }

            // Full-hidden file dump (ARF_DUMP_FULL=<layer>): write all `h` floats of
            // this layer's output to /tmp/ours_L<li>.txt for a per-channel diff vs llama.
            if std::env::var("ARF_DUMP_FULL").ok().as_deref() == Some(&li.to_string()) {
                cursor = None;
                cp.submit();
                cp = CommandPass::new(&self.ctx);
                let hv = self.ctx.read_f32(&self.scratch.hidden, h);
                let body: String = hv.iter().map(|v| format!("{v:.6}\n")).collect();
                std::fs::write(format!("/tmp/ours_L{li}.txt"), body).ok();
                eprintln!("[L{li} FULL] wrote {h} channels to /tmp/ours_L{li}.txt");
            }
            // Env-gated per-layer dump (ARF_DUMP_LAYERS=1): flush + read ||hidden||
            // after THIS layer to localize where the residual stream degenerates (vs a
            // llama.cpp eval-callback per-layer norm). Slow (submit + readback/layer) —
            // diagnostic only; resets the cursor since a mid-loop submit ends caching.
            if std::env::var("ARF_DUMP_LAYERS").is_ok() {
                cursor = None;
                cp.submit();
                cp = CommandPass::new(&self.ctx);
                let hv = self.ctx.read_f32(&self.scratch.hidden, h);
                let l2 = hv.iter().map(|x| x * x).sum::<f32>().sqrt();
                let (amax_i, amax_v) = hv
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
                    .map(|(i, &v)| (i, v))
                    .unwrap_or((0, 0.0));
                eprintln!(
                    "[LAYER {li}] ||hidden||={l2:.3} max_abs=ch{amax_i}={amax_v:.3} nan={} first4={:?}",
                    hv.iter().any(|x| !x.is_finite()),
                    &hv[..4.min(hv.len())]
                );
            }

            // Submit-SPLIT the decode every FLUSH_LAYERS layers. A single token packs
            // ~15·num_layers dispatches (≈720 for the 48-layer 12B) into ONE Metal
            // command buffer; flushing per-band lets the GPU start executing committed
            // work while the host records the next band — MEASURED +8.7% single-stream
            // on gemma-4-12B (24.0 → 26.1 tok/s), the same digestible-submit lesson as
            // the batched path's flush. SAFE: every buffer is a resident arena/weight,
            // and the flush lands at a LAYER BOUNDARY (the residual `hidden` is fully
            // written, nothing in flight). The steady-decode bind cache cannot span a
            // flush (it reopens the encoder + clears `binds`), so disable the cursor
            // when flushing — the submit win outweighs the lost cache (measured).
            let flush_n = decode_flush_layers();
            if flush_n > 0 && (li + 1) % flush_n == 0 && li + 1 < self.layers.len() {
                cursor = None;
                cp.flush();
            }
        }

        // final norm + lm_head only when logits are needed (the last token).
        if want_logits {
            // The megakernel already ran final_norm (hidden→normed); skip the duplicate.
            #[cfg(target_os = "macos")]
            let skip_final_norm = mega_did_final_norm;
            #[cfg(not(target_os = "macos"))]
            let skip_final_norm = false;
            if !skip_final_norm {
                let nd = uni
                    .alloc_write(&dims_norm(h, mc.rms_norm_eps))
                    .expect("final nd");
                cp.add_bound(
                    &k.rmsnorm,
                    "final_norm",
                    &[
                        self.scratch.hidden.as_entire_binding(),
                        self.final_norm.as_entire_binding(),
                        self.scratch.normed.as_entire_binding(),
                        uni.resource(&nd),
                    ],
                    [1, 1, 1],
                );
            }
            // Apples-to-apples decode-time normed (post-final-norm, pre-lm_head) for the
            // megakernel-vs-WGSL parity diff — fires at the same `pos` for BOTH paths.
            if std::env::var("ARF_DECODE_DUMP").ok().as_deref() == Some(&pos.to_string()) {
                cursor = None;
                cp.submit();
                cp = CommandPass::new(&self.ctx);
                let nv = self.ctx.read_f32(&self.scratch.normed, h);
                let l2 = nv.iter().map(|x| x * x).sum::<f32>().sqrt();
                eprintln!(
                    "[decode pos={pos}] normed ||={l2:.4} first6={:?}",
                    &nv[..6.min(nv.len())]
                );
            }
            // The island already ran lm_head (Q4KS) into the shared `logits`; skip the bf16
            // GEMV. softcap+sample below read `logits` regardless (shared buffer).
            #[cfg(target_os = "macos")]
            let skip_lm_head = mega_did_lm_head;
            #[cfg(not(target_os = "macos"))]
            let skip_lm_head = false;
            if !skip_lm_head {
                add_matmul(
                    &mut cp,
                    k,
                    &mut *uni,
                    self.scratch.normed.as_entire_binding(),
                    &self.lm_head,
                    self.scratch.logits.as_entire_binding(),
                    mc.vocab_size,
                    h,
                    cursor.as_mut(),
                );
            }
            // Gemma-4 final-logit soft-cap: logit = cap·tanh(logit/cap) over the
            // [vocab] logits, in place, BEFORE sampling/argmax. `None` (Gemma 3 /
            // Llama / Qwen) skips the dispatch entirely (no-op).
            // SINGLE-QUEUE: the island already ran softcap + greedy argmax into next_token;
            // skip the wgpu softcap + sample. Else the wgpu path runs them.
            #[cfg(target_os = "macos")]
            let skip_sample = mega_did_sample;
            #[cfg(not(target_os = "macos"))]
            let skip_sample = false;
            if !skip_sample {
                if let Some(cap) = mc.final_logit_softcap {
                    let scd = uni
                        .alloc_write(&[mc.vocab_size as u32, cap.to_bits(), 0, 0])
                        .expect("softcap dims");
                    cp.add_bound(
                        &k.softcap,
                        "softcap",
                        &[self.scratch.logits.as_entire_binding(), uni.resource(&scd)],
                        wg(mc.vocab_size),
                    );
                }
            }
            // Collapse [vocab] logits to one token id in the resident next_token
            // buffer, so the decode loop reads back a single u32 instead of
            // shipping the whole 128k-vocab logits to the CPU every step. Greedy
            // (sampling == None) takes the `sample` argmax fast path — zero extra
            // work, on-GPU feedback intact. Stochastic (temperature > 0) takes
            // `sample_stochastic`, which ALSO writes next_token on-GPU (no logits
            // readback): temperature + top-k + top-p + a deterministic
            // (seed, position) RNG, faithful to Sampler::sample / filtered_probs.
            // Decision: picked GPU-argmax/GPU-stochastic from {CPU-argmax,
            // GPU-argmax, on-GPU-feedback} — tradeoff: one tiny dispatch + 4-byte
            // readback vs. a 0.5 MB readback/token; keeps the fused feedback loop.
            // Top-k logits dump (ARF_DUMP_LOGITS): the argmax + top-5 token ids to
            // compare the predicted token against llama's greedy choice.
            if std::env::var("ARF_DUMP_LOGITS").is_ok() {
                cursor = None; // end caching for the mid-pass readback
                let _ = &cursor;
                cp.submit();
                cp = CommandPass::new(&self.ctx);
                let lg = self.ctx.read_f32(&self.scratch.logits, mc.vocab_size);
                let mut idx: Vec<usize> = (0..lg.len()).collect();
                idx.sort_by(|&a, &b| lg[b].total_cmp(&lg[a]));
                let top: Vec<(usize, f32)> = idx.iter().take(8).map(|&i| (i, lg[i])).collect();
                eprintln!("[LOGITS] top8={top:?}");
            }
            match sampling {
                None if skip_sample => { /* island ran the greedy argmax into next_token */ }
                None => {
                    let sd = uni
                        .alloc_write(&[mc.vocab_size as u32, 0, 0, 0])
                        .expect("sample dims");
                    cp.add_bound(
                        &k.sample,
                        "sample",
                        &[
                            self.scratch.logits.as_entire_binding(),
                            self.scratch.next_token.as_entire_binding(),
                            uni.resource(&sd),
                        ],
                        [1, 1, 1],
                    );
                }
                Some(s) => {
                    // `position` feeds the RNG counter so every decode step draws
                    // a fresh uniform from the same (seed, position) stream. The
                    // bind group is rebuilt each step (not P15-cached), so the
                    // per-step position lands in the uniform the kernel reads.
                    let sd = uni
                        .alloc_write(&s.dims(mc.vocab_size, pos))
                        .expect("sample_stochastic dims");
                    cp.add_bound(
                        &k.sample_stochastic,
                        "sample_stochastic",
                        &[
                            self.scratch.logits.as_entire_binding(),
                            self.scratch.next_token.as_entire_binding(),
                            uni.resource(&sd),
                        ],
                        [1, 1, 1],
                    );
                }
            }
            // Spec-decode verify: stash THIS step's argmax (just written into
            // next_token) into out_tokens[collect_after], so a k+1-token window's
            // predictions all drain in one readback after the loop.
            if let Some(slot) = collect_after {
                cp.copy_word(&self.scratch.next_token, &self.scratch.out_tokens, slot);
            }
        }
        // Skip submitting an EMPTY wgpu pass — on the full-single-queue megakernel path the
        // island owns the whole token, so `cp` has zero dispatches; submitting it anyway creates
        // + commits an empty command buffer that periodically triggers wgpu queue maintenance
        // (the ~18ms per-token spike). Empty → nothing to submit.
        if !cp.is_empty() {
            cp.submit();
        }

        // Env-gated one-shot diagnostic: dump activation norms + top-5 logits per
        // decode step to localize WHERE a decode degenerates (ARF_DUMP_LOGITS=1).
        if want_logits && std::env::var("ARF_DUMP_LOGITS").is_ok() {
            let hidden = self.ctx.read_f32(&self.scratch.hidden, h);
            let normed = self.ctx.read_f32(&self.scratch.normed, h);
            let logits = self.ctx.read_f32(&self.scratch.logits, mc.vocab_size);
            let l2 = |v: &[f32]| -> f32 { v.iter().map(|x| x * x).sum::<f32>().sqrt() };
            let nan = |v: &[f32]| v.iter().any(|x| !x.is_finite());
            let mut idx: Vec<usize> = (0..logits.len()).collect();
            // `total_cmp`, not `partial_cmp().unwrap()`: this diagnostic exists to
            // catch NaN/degenerate decodes (see `nan` above), so it must not itself
            // panic when a logit is NaN. total_cmp gives a total order over all f32.
            idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
            let top5: Vec<(usize, f32)> = idx.iter().take(5).map(|&i| (i, logits[i])).collect();
            // Dominant channel of the final hidden — a constant massive-activation
            // channel that swamps the input variation would explain input-independent
            // logits (Gemma has attention-sink channels).
            let (hmax_i, hmax_v) = hidden
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
                .map(|(i, &v)| (i, v))
                .unwrap_or((0, 0.0));
            eprintln!(
                "[DUMP] pos={pos} ||hidden||={:.3} ||normed||={:.3} hidden_nan={} logits_nan={} top5={:?}\n  hidden4={:?} normed4={:?} hidden_argmax=ch{hmax_i}={hmax_v:.3} logits[0..4]={:?}",
                l2(&hidden), l2(&normed), nan(&hidden), nan(&logits), top5,
                &hidden[..4.min(hidden.len())], &normed[..4.min(normed.len())],
                &logits[..4.min(logits.len())]
            );
        }

        // Env-gated routing-skew probe (ARF_DUMP_ROUTING=1): dump the routed
        // expert ids for THIS step. `moe_ids` holds the LAST MoE layer's top-k
        // selection (it is overwritten per layer in the merged pass), so each step
        // yields one sample of that layer's routing. Across many steps the
        // distribution reveals whether Qwen3 routing is near-uniform (sort-by-expert
        // only pays at large B) or skewed toward hot experts (it pays even at B=8).
        // A downstream script aggregates the `[ROUTE]` lines into a histogram. Zero
        // cost when off; no hot-path/parity impact (post-`submit` readback only).
        if std::env::var("ARF_DUMP_ROUTING").is_ok() {
            if let arf_core::config::MlpKind::Moe { top_k, .. } = &mc.mlp {
                if let Some(ids_buf) = self.scratch.moe_ids.as_ref() {
                    let ids = self.ctx.read_u32(ids_buf, *top_k);
                    eprintln!("[ROUTE] pos={pos} ids={ids:?}");
                }
            }
        }
    }

    /// The MoE feed-forward block (batch-1 decode): router GEMV → on-GPU top-k →
    /// for each routed expert run gate/up (per-expert scratch) → swiglu → down
    /// (weight-summed into `mlp_down`), then the shared expert(s). All dispatches
    /// stay in the merged compute pass `cp`; only the routed experts' weights are
    /// streamed. Reads/writes `self.scratch.{normed,gate,up,mlp_down,router_logits,
    /// moe_ids,moe_wts}` and `silu` from the `out` arena. See
    /// [`GpuMoe`]/matmul_vec_moe.wgsl. `wg(n)` = workgroups for `n` outputs.
    pub(super) fn add_moe_block(
        &self,
        cp: &mut CommandPass,
        k: &GpuKernels,
        uni: &mut GpuArena,
        out: &GpuArena,
        silu: &Region,
        moe: &GpuMoe,
        h: usize,
        wg: impl Fn(usize) -> [u32; 3],
        mut cursor: Option<&mut BindCursor<'_>>,
    ) {
        // Every dispatch here binds only resident buffers (packed experts, router),
        // fixed decode scratch (normed/gate/up/mlp_down/router_logits/ids/wts), the
        // fixed-offset `silu` arena region, and fixed-offset uniforms — all
        // byte-identical every token — so the WHOLE block participates in the P15
        // bind-group cache when `cursor` is Some (steady decode). This is the bulk
        // of the per-token dispatches (34/layer × 48), so caching it removes the
        // dominant per-token create_bind_group cost.
        let mi = moe.moe_inter;
        let router_logits = self.scratch.router_logits.as_ref().expect("router_logits");
        let ids = self.scratch.moe_ids.as_ref().expect("moe_ids");
        let wts = self.scratch.moe_wts.as_ref().expect("moe_wts");

        // High-occupancy launch for the expert GEMVs. The kernel grid-strides over
        // output columns (`col += num_workgroups.x`), so launching MORE workgroups
        // than `ceil(n/256)` is correct — they just split the columns finer. The
        // default `wg(n)` gives only ~3 workgroups for an expert (n=768) = ~5%
        // occupancy on this GPU, and there are 8×3×layers of these serialized per
        // token (measured ~404 ms/token). Saturating the device (more workgroups
        // per dispatch) is a ~50× win on the MoE path with no kernel/scratch change.
        // Cap so we never launch more workgroups than there are columns to cover.
        let wg_gemv = |n: usize| [(n.min(512) as u32).max(1), 1u32, 1u32];

        // 1) router logits = normed · routerᵀ  → [num_experts] (standard GEMV path).
        add_matmul(
            cp,
            k,
            uni,
            self.scratch.normed.as_entire_binding(),
            &moe.router,
            router_logits.as_entire_binding(),
            moe.num_experts,
            h,
            cursor.as_deref_mut(),
        );

        // 2) on-GPU top-k → ids[k], wts[k]
        let routed = uni
            .alloc_write(&[
                moe.num_experts as u32,
                moe.top_k as u32,
                moe.norm_topk as u32,
                0,
            ])
            .expect("moe route dims");
        cp.add_maybe_cached(
            cursor.as_deref_mut(),
            &k.moe_route,
            "moe_route",
            &[
                router_logits.as_entire_binding(),
                ids.as_entire_binding(),
                wts.as_entire_binding(),
                uni.resource(&routed),
            ],
            [1, 1, 1],
        );

        // One expert GEMV: dispatch the bf16 or Q4 kernel with the matching bindings.
        // experts = packed buffer (selected on-GPU via ids[slot]); a=activation,
        // c=output; mode 0 raw / 1 write-scaled / 2 accumulate-scaled.
        fn expert_gemv(
            cp: &mut CommandPass,
            k: &GpuKernels,
            uni: &mut GpuArena,
            experts: &PackedExperts,
            a: wgpu::BindingResource<'_>,
            c: wgpu::BindingResource<'_>,
            ids: wgpu::BindingResource<'_>,
            wts: wgpu::BindingResource<'_>,
            kdim: usize,
            n: usize,
            slot: usize,
            mode: u32,
            label: &str,
            groups: [u32; 3],
            cursor: Option<&mut BindCursor<'_>>,
        ) {
            // L109 SAFETY GUARD — mirrors batch.rs: the wgpu expert set is a placeholder when the
            // island owns MoE, so a dispatch here must fail loudly rather than compute garbage.
            #[cfg(target_os = "macos")]
            if std::env::var_os("ARF_MSL_GEMV").is_some()
                && std::env::var_os("ARF_MOE_WGPU_FULL").is_none()
            {
                panic!(
                    "wgpu m=1 MoE dispatch ({label}) reached while the island owns the MoE path — \
                     the wgpu expert set is an L109 placeholder and would compute garbage. \
                     Re-run with ARF_MOE_WGPU_FULL=1 to restore the full expert allocation."
                );
            }
            let dims = uni
                .alloc_write(&[1u32, kdim as u32, n as u32, slot as u32, mode, 0, 0, 0])
                .expect("moe dims");
            // Kernel per variant; the wts-present binding list is built once (see
            // `moe_wts_bindings` for the slot contract).
            let kernel = match experts {
                PackedExperts::Bf16(_) => &k.matmul_vec_moe,
                PackedExperts::Q4 { .. } => &k.matmul_vec_moe_q4,
                PackedExperts::Q4K { .. } => &k.matmul_vec_moe_q4k,
                PackedExperts::Q4KS { .. } => &k.matmul_vec_moe_q4ks,
            };
            let binds = moe_wts_bindings(experts, a, c, uni.resource(&dims), ids, wts);
            cp.add_maybe_cached(cursor, kernel, label, &binds, groups);
        }

        // One SwiGLU expert: gate/up (raw, per-expert scratch) → swiglu → down
        // (weight-summed into mlp_down). `mode` controls the down write.
        let run_expert = |cp: &mut CommandPass,
                          uni: &mut GpuArena,
                          gate: &PackedExperts,
                          up: &PackedExperts,
                          down: &PackedExperts,
                          ids: wgpu::BindingResource<'_>,
                          wts: wgpu::BindingResource<'_>,
                          slot: usize,
                          down_mode: u32,
                          mut cursor: Option<&mut BindCursor<'_>>| {
            expert_gemv(
                cp,
                k,
                uni,
                gate,
                self.scratch.normed.as_entire_binding(),
                self.scratch.gate.as_entire_binding(),
                ids.clone(),
                wts.clone(),
                h,
                mi,
                slot,
                0,
                "moe_gate",
                wg_gemv(mi),
                cursor.as_deref_mut(),
            );
            expert_gemv(
                cp,
                k,
                uni,
                up,
                self.scratch.normed.as_entire_binding(),
                self.scratch.up.as_entire_binding(),
                ids.clone(),
                wts.clone(),
                h,
                mi,
                slot,
                0,
                "moe_up",
                wg_gemv(mi),
                cursor.as_deref_mut(),
            );
            let swd = uni.alloc_write(&[mi as u32, 0, 0, 0]).expect("moe swd");
            cp.add_maybe_cached(
                cursor.as_deref_mut(),
                &k.swiglu,
                "moe_swiglu",
                &[
                    self.scratch.gate.as_entire_binding(),
                    self.scratch.up.as_entire_binding(),
                    out.resource(silu),
                    uni.resource(&swd),
                ],
                wg(mi),
            );
            expert_gemv(
                cp,
                k,
                uni,
                down,
                out.resource(silu),
                self.scratch.mlp_down.as_entire_binding(),
                ids,
                wts,
                mi,
                h,
                slot,
                down_mode,
                "moe_down",
                wg_gemv(h),
                cursor.as_deref_mut(),
            );
        };

        // 3) FUSED routed experts (mul_mat_id-style): instead of top_k×{gate,up,
        // swiglu,down} serialized dispatches (the per-expert dependency chain +
        // shared-mlp_down accumulate that serialized ~32 dispatches/layer and made
        // MoE ~90% of decode), do ONE dispatch each: gate_all/up_all over all
        // top_k experts → [top_k,mi], swiglu over [top_k*mi], then ONE down that
        // reduces over experts (slot-ascending, bit-exact) into mlp_down. ~32→4
        // dispatches, each high-occupancy and independent (only the natural
        // gate/up→swiglu→down chain remains, depth 3 not 32).
        let gate_all = self.scratch.moe_gate_all.as_ref().expect("moe_gate_all");
        let up_all = self.scratch.moe_up_all.as_ref().expect("moe_up_all");
        let tk = moe.top_k;
        // Workgroups: grid-stride, cap at 512 (saturates the device; kernel strides).
        let wg_cap = |n: usize| [(n.min(512) as u32).max(1), 1u32, 1u32];

        // gate_all / up_all: batched expert GEMV, output [top_k, mi]. dims word[5]=top_k.
        // One helper, invoked for gate then up (same kernel, different packed buf + out).
        // gate/up don't apply the router weight (the fused down does), so these
        // kernels have NO `wts` binding (an unused binding is stripped from the
        // layout by naga, causing a bind-count mismatch). Binding order matches the
        // *_batch_gu shaders exactly.
        fn batched_gu(
            cp: &mut CommandPass,
            k: &GpuKernels,
            uni: &mut GpuArena,
            experts: &PackedExperts,
            a: wgpu::BindingResource<'_>,
            c: wgpu::BindingResource<'_>,
            ids: wgpu::BindingResource<'_>,
            kdim: usize,
            n: usize,
            top_k: usize,
            label: &str,
            groups: [u32; 3],
            cursor: Option<&mut BindCursor<'_>>,
        ) {
            let dims = uni
                .alloc_write(&[1u32, kdim as u32, n as u32, 0, 0, top_k as u32, 0, 0])
                .expect("moe gu dims");
            // Each arm builds the no-wts binding list inline (it also carries per-arm
            // grid math / `_sg` selection, so it is NOT routed through `moe_wts_bindings`).
            // The slot contract for this layout is `moe_gu_roles`, pinned by
            // `moe_binding_order_is_contract`.
            match experts {
                PackedExperts::Bf16(b) => cp.add_maybe_cached(
                    cursor,
                    &k.matmul_vec_moe_batch_gu,
                    label,
                    &[a, b.as_entire_binding(), c, uni.resource(&dims), ids],
                    groups,
                ),
                PackedExperts::Q4 { codes, scales } => cp.add_maybe_cached(
                    cursor,
                    &k.matmul_vec_moe_q4_batch_gu,
                    label,
                    &[
                        a,
                        codes.as_entire_binding(),
                        c,
                        uni.resource(&dims),
                        scales.as_entire_binding(),
                        ids,
                    ],
                    groups,
                ),
                PackedExperts::Q4K {
                    codes,
                    scales,
                    mins,
                } => {
                    // The non-_sg variant is multi-row (ROWS_PER_WG=2, reusing each loaded
                    // activation sub-block across 2 weight columns) and grid-strides over
                    // GROUPS of 2, so halve the count. Grid-stride, so a clamped/rounded
                    // count stays correct.
                    const ROWS_PER_WG: u32 = 2;
                    let kernel = &k.matmul_vec_moe_q4k_batch_gu;
                    let g = [groups[0].div_ceil(ROWS_PER_WG).max(1), groups[1], groups[2]];
                    cp.add_maybe_cached(
                        cursor,
                        kernel,
                        label,
                        &[
                            a,
                            codes.as_entire_binding(),
                            c,
                            uni.resource(&dims),
                            scales.as_entire_binding(),
                            ids,
                            mins.as_entire_binding(),
                        ],
                        g,
                    )
                }
                PackedExperts::Q4KS {
                    codes,
                    scales,
                    mins,
                    dd,
                } => {
                    // Multi-row (lever B): the q4ks batched gate/up kernel computes
                    // ROWS_PER_WG=2 output columns per workgroup (reusing each loaded
                    // activation sub-block across them), so it grid-strides over
                    // GROUPS of 2. Halve the group count to match (n=mi is a multiple
                    // of 2, so total = top_k*n is even and divides cleanly). The
                    // kernel grid-strides, so a clamped/rounded count stays correct.
                    const ROWS_PER_WG: u32 = 2;
                    let g = [groups[0].div_ceil(ROWS_PER_WG).max(1), groups[1], groups[2]];
                    cp.add_maybe_cached(
                        cursor,
                        &k.matmul_vec_moe_q4ks_batch_gu,
                        label,
                        &[
                            a,
                            codes.as_entire_binding(),
                            c,
                            uni.resource(&dims),
                            scales.as_entire_binding(),
                            ids,
                            mins.as_entire_binding(),
                            dd.as_entire_binding(),
                        ],
                        g,
                    )
                }
            }
        }
        batched_gu(
            cp,
            k,
            uni,
            &moe.gate,
            self.scratch.normed.as_entire_binding(),
            gate_all.as_entire_binding(),
            ids.as_entire_binding(),
            h,
            mi,
            tk,
            "moe_gate_all",
            wg_cap(tk * mi),
            cursor.as_deref_mut(),
        );
        batched_gu(
            cp,
            k,
            uni,
            &moe.up,
            self.scratch.normed.as_entire_binding(),
            up_all.as_entire_binding(),
            ids.as_entire_binding(),
            h,
            mi,
            tk,
            "moe_up_all",
            wg_cap(tk * mi),
            cursor.as_deref_mut(),
        );
        // swiglu over all top_k*mi elements (elementwise; existing kernel unchanged).
        let swd = uni
            .alloc_write(&[(tk * mi) as u32, 0, 0, 0])
            .expect("moe swd");
        cp.add_maybe_cached(
            cursor.as_deref_mut(),
            &k.swiglu,
            "moe_swiglu_all",
            &[
                gate_all.as_entire_binding(),
                up_all.as_entire_binding(),
                out.resource(silu),
                uni.resource(&swd),
            ],
            wg(tk * mi),
        );
        // Fused down: reduce over all top_k experts into mlp_down (mode 1 = write).
        // dims: k=mi, n=h, mode=1, word[5]=top_k. One workgroup per output row h_j.
        let dd = uni
            .alloc_write(&[1u32, mi as u32, h as u32, 0, 1u32, tk as u32, 0, 0])
            .expect("moe down dims");
        {
            let a = out.resource(silu);
            let c = self.scratch.mlp_down.as_entire_binding();
            let idsb = ids.as_entire_binding();
            let wtsb = wts.as_entire_binding();
            // Same wts-present layout as `expert_gemv`; only the kernel differs (the
            // `*_down_reduce` family). Binding list via the shared helper so the slot
            // contract stays in one place.
            let kernel = match &moe.down {
                PackedExperts::Bf16(_) => &k.matmul_vec_moe_down_reduce,
                PackedExperts::Q4 { .. } => &k.matmul_vec_moe_q4_down_reduce,
                PackedExperts::Q4K { .. } => &k.matmul_vec_moe_q4k_down_reduce,
                PackedExperts::Q4KS { .. } => &k.matmul_vec_moe_q4ks_down_reduce,
            };
            let binds = moe_wts_bindings(&moe.down, a, c, uni.resource(&dd), idsb, wtsb);
            cp.add_maybe_cached(
                cursor.as_deref_mut(),
                kernel,
                "moe_down_reduce",
                &binds,
                wg_cap(h),
            );
        }

        // 4) shared expert(s): weight 1, always accumulate (top_k≥1 so mlp_down is
        // already initialized by the routed experts above).
        if moe.shared_experts > 0 {
            let sids = self.scratch.shared_ids.as_ref().expect("shared_ids");
            let swts = self.scratch.shared_wts.as_ref().expect("shared_wts");
            let sgate = moe.shared_gate.as_ref().expect("shared_gate");
            let sup = moe.shared_up.as_ref().expect("shared_up");
            let sdown = moe.shared_down.as_ref().expect("shared_down");
            for slot in 0..moe.shared_experts {
                run_expert(
                    cp,
                    uni,
                    sgate,
                    sup,
                    sdown,
                    sids.as_entire_binding(),
                    swts.as_entire_binding(),
                    slot,
                    2,
                    cursor.as_deref_mut(),
                );
            }
        }
    }
}
