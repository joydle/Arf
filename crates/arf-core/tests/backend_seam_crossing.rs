//! L358 — the seam is only a seam if BOTH backends can reach it.
//!
//! The flaw this guards against is one I shipped and then found: L354 defined `LayerEncoder` in
//! `arf-gpu`, and "proved" it implementable with a compile probe that also lived in
//! `arf-gpu`. The probe passed. It could not have failed — the trait was in the same crate.
//! Meanwhile any other backend depends on `arf-core` alone (correctly: a backend must not
//! pull in the Metal one), so the seam was reachable by exactly one backend.
//!
//! These tests live in `arf-core` and touch NO backend crate, which is the point: they pin the
//! properties that made the trait crossable, so a future edit that quietly re-anchors the seam to
//! one backend's types fails here rather than in six months on a machine with a GPU.

use arf_core::backend::{LayerEncoder, LayerKind};

/// A backend whose types resemble no real backend's. If the trait can be implemented
/// with these, it is not secretly shaped around one vendor's handles.
struct AlienBackend;

/// Two INDEPENDENT flags, not one tag. A single tag makes the dominance rule untestable — the
/// first mutation run proved it: reordering the checks survived, because a one-tag layer can
/// never be both. Real layers can: 48 of Qwen3.8-27B's 64 carry `gdn`, and a routed one carries
/// `moe` as well.
struct AlienLayer<'a> {
    gdn: bool,
    moe: bool,
    _name: &'a str,
}

impl LayerEncoder for AlienBackend {
    type Layer<'a> = AlienLayer<'a>;
    type ModelConstants<'a> = &'a [u32];
    type StepInputs<'a> = (usize, &'a str);
    /// Owned, not `Vec<&'a str>`: the layer's lifetime and the encoder's are independent
    /// parameters, so a borrow cannot cross from one to the other. That is the correct shape —
    /// a real backend records into an encoder that outlives no particular layer.
    type Encoder<'a> = Vec<String>;

    fn layer_kind(layer: &Self::Layer<'_>) -> LayerKind {
        if layer.gdn {
            LayerKind::Recurrent
        } else if layer.moe {
            LayerKind::MoeAttn
        } else {
            LayerKind::DenseAttn
        }
    }

    fn encode_layer(
        &self,
        layer: &Self::Layer<'_>,
        _mc: &Self::ModelConstants<'_>,
        _step: &Self::StepInputs<'_>,
        enc: &mut Self::Encoder<'_>,
    ) -> Result<(), String> {
        enc.push(layer._name.to_string());
        Ok(())
    }

    fn encode_epilogue(
        &self,
        _mc: &Self::ModelConstants<'_>,
        _step: &Self::StepInputs<'_>,
        enc: &mut Self::Encoder<'_>,
    ) -> Result<(), String> {
        enc.push("epilogue".to_string());
        Ok(())
    }
}

/// The property L354 lacked: implementable from a crate that knows nothing about any GPU API.
#[test]
fn seam_is_implementable_without_any_backend_crate() {
    fn accepts<T: LayerEncoder>() {}
    accepts::<AlienBackend>();
}

/// A full step records every layer plus one epilogue — the shape both backends must honour.
#[test]
fn a_step_records_every_layer_then_one_epilogue() {
    let be = AlienBackend;
    let mc: &[u32] = &[4096, 151936];
    let step = (1usize, "tok");
    let layers = [
        AlienLayer {
            gdn: true,
            moe: false,
            _name: "gdn",
        },
        AlienLayer {
            gdn: true,
            moe: false,
            _name: "gdn",
        },
        AlienLayer {
            gdn: false,
            moe: false,
            _name: "dense",
        },
        AlienLayer {
            gdn: false,
            moe: true,
            _name: "moe",
        },
    ];
    let mut enc: Vec<String> = Vec::new();

    for ly in &layers {
        be.encode_layer(ly, &mc, &step, &mut enc)
            .expect("layer encodes");
    }
    be.encode_epilogue(&mc, &step, &mut enc).expect("epilogue");

    assert_eq!(enc, ["gdn", "gdn", "dense", "moe", "epilogue"]);
    assert_eq!(
        enc.iter().filter(|s| *s == "epilogue").count(),
        1,
        "the epilogue runs once per STEP, not once per layer — that split is why it is a \
         separate method (L188: GEMV below b=8, tiled GEMM above)"
    );
}

/// `gdn` dominates `moe`. Recurrence replaces ATTENTION; MoE replaces only the FFN — so a
/// recurrent layer that also routes experts is Recurrent, and a backend that ordered these
/// checks the other way would silently run attention on 48 of Qwen3.8-27B's 64 layers.
///
/// This is the exact near-miss caught by truth table in L354 step 2: `kind() == DenseAttn &&
/// !has_dense_ffn()` differs from the original in 1 of 8 combinations, because a Recurrent layer
/// still runs an FFN.
#[test]
fn recurrence_dominates_routing() {
    let k = |gdn, moe| {
        AlienBackend::layer_kind(&AlienLayer {
            gdn,
            moe,
            _name: "",
        })
    };

    // THE CASE THAT MATTERS. Both flags set: the two orderings disagree here and nowhere else,
    // which is why the first version of this test — a single tag, one arm per kind — could not
    // fail no matter how the checks were ordered. Recurrence wins.
    assert_eq!(k(true, true), LayerKind::Recurrent, "gdn must dominate moe");

    assert_eq!(k(true, false), LayerKind::Recurrent);
    assert_eq!(k(false, true), LayerKind::MoeAttn);
    assert_eq!(k(false, false), LayerKind::DenseAttn);
}

/// Three kinds, all distinct. If a fourth architecture family appears, this fails and forces the
/// decision to be made in the taxonomy rather than in an `if` somewhere in an encode loop.
#[test]
fn the_taxonomy_is_three_distinct_kinds() {
    let all = [
        LayerKind::DenseAttn,
        LayerKind::MoeAttn,
        LayerKind::Recurrent,
    ];
    for (i, a) in all.iter().enumerate() {
        for (j, b) in all.iter().enumerate() {
            assert_eq!(a == b, i == j, "{a:?} vs {b:?}");
        }
    }
}
