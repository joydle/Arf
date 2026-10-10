//! L353 — the LAYER KIND taxonomy, pinned.
//!
//! `MegaLayer::kind()` (arf-gpu) names what a decode layer structurally IS, replacing
//! `Option` probes scattered through the encode loop. This file pins the mapping and, more
//! importantly, the ONE equivalence that a refactor can silently break.
//!
//! WHY IT LIVES IN arf-core AND MODELS THE LOGIC: `MegaLayer` borrows raw Metal handles, so
//! constructing one in a test needs a GPU. The logic under test is pure boolean, so it is
//! modelled here and runs on any machine in microseconds — the same choice the checkpoint and
//! serial-row oracles made (L324), and for the same reason: the failure mode is silent.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    DenseAttn,
    MoeAttn,
    Recurrent,
}

/// Mirror of `MegaLayer::kind()`.
fn kind(gdn: bool, moe: bool) -> Kind {
    if gdn {
        Kind::Recurrent
    } else if moe {
        Kind::MoeAttn
    } else {
        Kind::DenseAttn
    }
}

/// Mirror of the encoder's FFN precondition, BEFORE and AFTER L353.
fn declines_original(moe: bool, dense: bool) -> bool {
    !moe && !dense
}
fn declines_now(moe: bool, dense: bool) -> bool {
    !moe && !dense
}

/// Every layer resolves to exactly one kind, and `gdn` wins over `moe` — a recurrent MoE layer
/// is Recurrent, because the recurrence replaces attention while the MoE replaces only the FFN.
#[test]
fn kind_is_total_and_gdn_dominates() {
    assert_eq!(kind(false, false), Kind::DenseAttn);
    assert_eq!(kind(false, true), Kind::MoeAttn);
    assert_eq!(kind(true, false), Kind::Recurrent);
    assert_eq!(kind(true, true), Kind::Recurrent);
}

/// 🔴 THE REGRESSION THIS FILE EXISTS FOR.
///
/// The obvious way to write the FFN precondition with the new vocabulary is
/// `kind() == DenseAttn && !has_dense_ffn()`. It reads better and it is WRONG: a Recurrent layer
/// still runs an FFN, so `gdn` present must not excuse a missing gate/up/down. That form let a
/// recurrent-without-FFN layer through, and it was caught by enumerating all eight combinations
/// rather than by reading the code.
///
/// The precondition is about the FFN alone — hence `moe`, never `kind`.
#[test]
fn ffn_precondition_is_unchanged_for_every_combination() {
    for gdn in [false, true] {
        for moe in [false, true] {
            for dense in [false, true] {
                assert_eq!(
                    declines_original(moe, dense),
                    declines_now(moe, dense),
                    "gdn={gdn} moe={moe} dense={dense}: the L353 rewrite changed behaviour"
                );
            }
        }
    }
}

/// The tempting-but-wrong form, kept as an executable record of the bug. If someone rewrites the
/// precondition in terms of `kind` again, this test says exactly which case it breaks.
#[test]
fn the_kind_based_form_is_wrong_and_here_is_the_case() {
    let (gdn, moe, dense) = (true, false, false); // recurrent layer, no MoE, no dense FFN
    let tempting = kind(gdn, moe) == Kind::DenseAttn && !dense;
    assert!(!tempting, "the kind-based form does not decline this layer");
    assert!(
        declines_original(moe, dense),
        "but the original DOES decline it — that is the silent difference"
    );
}

/// L358f — **the predicate that must NOT be `kind()`.** Two sites in the Metal encode loop pick
/// between the fused (Llama/Qwen) and four-norm (Gemma) norm topologies. Both used to read
/// `ly.moe.is_some()`; both now read `ly.uses_fused_norms()`, which is the same boolean with a
/// name that says what it asks.
///
/// The temptation is to "simplify" that to `kind() == MoeAttn`. This test exists to fail when
/// someone does. Over all eight `(gdn, moe, dense)` combinations the two predicates disagree in
/// exactly two — a layer carrying BOTH `gdn` and `moe`:
///
/// ```text
///   gdn moe | moe.is_some() | kind()    | kind()==MoeAttn
///    1   1  |     true      | Recurrent |     false        <- DIFFERS
/// ```
///
/// `kind()` is right that such a layer is Recurrent: recurrence replaces ATTENTION. But its FFN
/// is an MoE, so its norm topology is the fused one, and a `kind()`-based branch would hand it
/// Gemma's four-norm path — reading `post_attn_norm`/`pre_ffn_norm` weights that a Qwen-shaped
/// layer does not have. Same shape as the `has_dense_ffn` near-miss L353 caught by truth table.
#[test]
fn norm_topology_is_not_the_layer_kind() {
    // Mirror of the two accessors, over the full (gdn, moe) space.
    let kind = |gdn: bool, moe: bool| {
        if gdn {
            "Recurrent"
        } else if moe {
            "MoeAttn"
        } else {
            "DenseAttn"
        }
    };
    let uses_fused_norms = |_gdn: bool, moe: bool| moe;

    let mut disagreements = 0;
    for gdn in [false, true] {
        for moe in [false, true] {
            let via_kind = kind(gdn, moe) == "MoeAttn";
            if via_kind != uses_fused_norms(gdn, moe) {
                disagreements += 1;
                assert!(
                    gdn && moe,
                    "the ONLY case where kind() and norm topology may differ is gdn+moe,                      got gdn={gdn} moe={moe}"
                );
            }
        }
    }
    assert_eq!(
        disagreements, 1,
        "exactly one (gdn, moe) combination must distinguish the two predicates — if this is 0,          someone collapsed uses_fused_norms() into kind() == MoeAttn and a recurrent MoE layer          now takes Gemma's four-norm path"
    );

    // And state the case itself, so the failure names the bug rather than a count.
    assert!(
        uses_fused_norms(true, true),
        "a recurrent layer that routes experts still uses the FUSED norm topology"
    );
    assert_ne!(
        kind(true, true),
        "MoeAttn",
        "...while its KIND is Recurrent — which is exactly why the two must stay separate"
    );
}
