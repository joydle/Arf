//! The MoE expert-kernel binding-order CONTRACT — one auditable source of truth.
//!
//! Every `matmul_vec_moe*` WGSL kernel reads its storage bindings at fixed `@binding(N)`
//! slots, so the host MUST bind buffers in the exact order the shader expects; a reorder
//! silently corrupts MoE output. The GPU parity tests that would catch that SKIP without
//! an adapter (CI has none), so these role tables + the `tests` below are the GPU-less
//! guard. Changing a sequence here means you changed the contract — update the matching
//! `.wgsl` `@binding(N)` AND the dispatch site too.
//!
//! Enforcement strength differs by site: `decode.rs`'s `moe_wts_bindings` builds its list
//! through this contract and `debug_assert`s its length against [`moe_wts_roles`], so the
//! produced bindings are checked at runtime. The other sites (decode `batched_gu`, all of
//! `batch.rs`'s prefill dispatch) build their `&[..]` inline with per-arm grid
//! math; for those these tables are the WRITTEN contract a reviewer must keep in sync —
//! a slot reorder here fails CI, flagging the divergence.
//!
//! There are TWO layouts (do not conflate them):
//! - **wts-present** (`moe_wts_roles`): `expert_gemv`, `down_reduce`, batched `down_reduce_b`.
//! - **no-wts gate/up** (`moe_gu_roles`): `batched_gu`, batched `batched_gu_b`. `mins`/`dd`
//!   follow `ids` directly (no `wts` slot).

/// A binding's logical role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BindRole {
    /// Activation input.
    A,
    /// The packed expert weight buffer (`buf` for bf16, `codes` for quant).
    Weights,
    /// Output buffer (`c` / `dst` / a scratch arena).
    C,
    /// The `dims` uniform.
    Dims,
    /// Quant scales (Q4/Q4K/Q4KS).
    Scales,
    /// Routed expert ids.
    Ids,
    /// Router weights.
    Wts,
    /// Q4K/Q4KS per-block mins.
    Mins,
    /// Q4KS interleaved [d, dmin].
    Dd,
}

/// The quant discriminant of a [`PackedExperts`](super::PackedExperts), independent of its
/// GPU buffers — so the contract can be exercised without a GPU adapter (real
/// `PackedExperts` hold `wgpu::Buffer`s, which can't be built GPU-less).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum QuantTag {
    Bf16,
    Q4,
    Q4K,
    Q4KS,
}

impl super::PackedExperts {
    pub(super) fn tag(&self) -> QuantTag {
        match self {
            super::PackedExperts::Bf16(_) => QuantTag::Bf16,
            super::PackedExperts::Q4 { .. } => QuantTag::Q4,
            super::PackedExperts::Q4K { .. } => QuantTag::Q4K,
            super::PackedExperts::Q4KS { .. } => QuantTag::Q4KS,
        }
    }
}

/// **wts-present** layout: `a, weights, c, dims, [scales], ids, wts, [mins], [dd]`.
/// Used by decode `expert_gemv` + `down_reduce` and batched `down_reduce_b`.
pub(super) fn moe_wts_roles(tag: QuantTag) -> &'static [BindRole] {
    use BindRole::*;
    match tag {
        QuantTag::Bf16 => &[A, Weights, C, Dims, Ids, Wts],
        QuantTag::Q4 => &[A, Weights, C, Dims, Scales, Ids, Wts],
        QuantTag::Q4K => &[A, Weights, C, Dims, Scales, Ids, Wts, Mins],
        QuantTag::Q4KS => &[A, Weights, C, Dims, Scales, Ids, Wts, Mins, Dd],
    }
}

/// **no-wts gate/up** layout: `a, weights, c, dims, [scales], ids, [mins], [dd]`.
/// `mins`/`dd` follow `ids` directly (no `wts` slot). Used by decode `batched_gu` and
/// batched `batched_gu_b`, which build their lists inline (they also carry per-arm grid
/// math), so this has no production caller — it is the written contract the tests pin.
#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn moe_gu_roles(tag: QuantTag) -> &'static [BindRole] {
    use BindRole::*;
    match tag {
        QuantTag::Bf16 => &[A, Weights, C, Dims, Ids],
        QuantTag::Q4 => &[A, Weights, C, Dims, Scales, Ids],
        QuantTag::Q4K => &[A, Weights, C, Dims, Scales, Ids, Mins],
        QuantTag::Q4KS => &[A, Weights, C, Dims, Scales, Ids, Mins, Dd],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use BindRole::*;

    // GOLDEN TABLE — the wts-present and no-wts layouts (decode + batched non-grouped).
    #[test]
    fn moe_binding_order_is_contract() {
        // wts-present.
        assert_eq!(
            moe_wts_roles(QuantTag::Bf16),
            &[A, Weights, C, Dims, Ids, Wts]
        );
        assert_eq!(
            moe_wts_roles(QuantTag::Q4),
            &[A, Weights, C, Dims, Scales, Ids, Wts]
        );
        assert_eq!(
            moe_wts_roles(QuantTag::Q4K),
            &[A, Weights, C, Dims, Scales, Ids, Wts, Mins]
        );
        assert_eq!(
            moe_wts_roles(QuantTag::Q4KS),
            &[A, Weights, C, Dims, Scales, Ids, Wts, Mins, Dd]
        );
        // no-wts gate/up: mins/dd follow ids directly.
        assert_eq!(moe_gu_roles(QuantTag::Bf16), &[A, Weights, C, Dims, Ids]);
        assert_eq!(
            moe_gu_roles(QuantTag::Q4),
            &[A, Weights, C, Dims, Scales, Ids]
        );
        assert_eq!(
            moe_gu_roles(QuantTag::Q4K),
            &[A, Weights, C, Dims, Scales, Ids, Mins]
        );
        assert_eq!(
            moe_gu_roles(QuantTag::Q4KS),
            &[A, Weights, C, Dims, Scales, Ids, Mins, Dd]
        );
    }

    // The non-grouped wts layout is the gu layout with `Wts` inserted right after `Ids`.
    // Pins the one-slot relationship so the two can't silently diverge another way.
    #[test]
    fn wts_layout_is_gu_layout_plus_wts_after_ids() {
        for tag in [QuantTag::Bf16, QuantTag::Q4, QuantTag::Q4K, QuantTag::Q4KS] {
            let gu = moe_gu_roles(tag);
            let wts = moe_wts_roles(tag);
            let ids_pos = gu.iter().position(|r| *r == Ids).expect("gu has Ids");
            let mut expected: Vec<BindRole> = gu.to_vec();
            expected.insert(ids_pos + 1, Wts);
            assert_eq!(wts, expected.as_slice(), "wts layout for {tag:?}");
        }
    }
}
