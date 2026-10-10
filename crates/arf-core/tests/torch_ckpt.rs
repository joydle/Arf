//! The safe PyTorch-checkpoint reader against checkpoints written by torch itself.
//!
//! Fixtures and `expected.json` come from `tests/fixtures/torch_ckpt/gen_fixtures.py` (CPU torch
//! 2.14.0). `expected.json` is torch's own account of each tensor — dtype, shape, stride, storage
//! offset and the bytes of `t.contiguous()` — taken from the in-memory tensors, so every byte
//! comparison here is reader-vs-torch, not reader-vs-reader.

use std::borrow::Cow;
use std::collections::BTreeSet;
use std::path::PathBuf;

use arf_core::model::torch_ckpt::{DType, Format, Object, TorchCheckpoint, TorchCkptError};

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/torch_ckpt")
}

fn expected() -> serde_json::Value {
    let s = std::fs::read_to_string(dir().join("expected.json")).unwrap();
    serde_json::from_str(&s).unwrap()
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn usizes(v: &serde_json::Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect()
}

/// Every tensor torch saved is found under the same name with the same dtype, shape, stride,
/// offset and bytes — and no extra tensors are reported.
fn check_against_torch(file: &str, want_format: Format) {
    let exp = expected();
    let exp = exp[file].as_object().unwrap();
    let ck = TorchCheckpoint::open(dir().join(file)).unwrap();
    assert_eq!(ck.format(), want_format, "{file}");
    let got: BTreeSet<&str> = ck.tensors().map(|(n, _)| n).collect();
    let want: BTreeSet<&str> = exp.keys().map(|s| s.as_str()).collect();
    assert_eq!(got, want, "{file}: tensor names");
    for (name, e) in exp {
        let t = ck.tensor(name).unwrap();
        assert_eq!(
            t.dtype.name(),
            e["dtype"].as_str().unwrap(),
            "{file}:{name} dtype"
        );
        assert_eq!(t.shape, usizes(&e["shape"]), "{file}:{name} shape");
        assert_eq!(t.stride, usizes(&e["stride"]), "{file}:{name} stride");
        assert_eq!(
            t.storage_offset,
            e["offset"].as_u64().unwrap() as usize,
            "{file}:{name} offset"
        );
        let bytes = ck.tensor_bytes(t).unwrap();
        assert_eq!(
            hex(&bytes),
            e["hex"].as_str().unwrap(),
            "{file}:{name} bytes"
        );
        assert_eq!(bytes.len(), t.nbytes());
    }
}

#[test]
fn zip_state_dict_every_dtype_matches_torch() {
    check_against_torch("state_dict.pt", Format::Zip);
    let ck = TorchCheckpoint::open(dir().join("state_dict.pt")).unwrap();
    let dtypes: BTreeSet<&str> = ck.tensors().map(|(_, t)| t.dtype.name()).collect();
    for d in [
        "float32",
        "float64",
        "float16",
        "bfloat16",
        "int64",
        "int32",
        "int16",
        "int8",
        "uint8",
        "bool",
        "complex64",
    ] {
        assert!(dtypes.contains(d), "fixture lacks {d}");
    }
    // The nn.Module buffer and a 0-dim scalar survive with their exact shapes.
    assert_eq!(
        ck.tensor("norm.num_batches_tracked").unwrap().dtype,
        DType::I64
    );
    assert!(ck.tensor("scalar").unwrap().shape.is_empty());
    assert_eq!(ck.tensor("empty").unwrap().shape, vec![0, 3]);
    ck.verify_checksums().unwrap();
}

#[test]
fn zip_views_and_shared_storage_match_torch() {
    check_against_torch("views.pt", Format::Zip);
    let ck = TorchCheckpoint::open(dir().join("views.pt")).unwrap();
    let base = ck.tensor("base").unwrap();
    // Five views of one storage: torch writes it once, and every view points at it.
    for n in ["transposed", "row2", "col_slice", "narrow"] {
        assert_eq!(ck.tensor(n).unwrap().storage, base.storage, "{n}");
    }
    assert_eq!(
        ck.storages().len(),
        2,
        "base's storage + the expanded int32 one"
    );
    // Contiguous views are borrowed from the map (zero-copy) at the right offset...
    let storage = ck.storage_bytes(base.storage);
    let row2 = ck.tensor("row2").unwrap();
    assert!(row2.is_contiguous());
    match ck.tensor_bytes(row2).unwrap() {
        Cow::Borrowed(b) => assert_eq!(b.as_ptr(), storage[12 * 4..].as_ptr()),
        Cow::Owned(_) => panic!("contiguous view was copied"),
    }
    // ...non-contiguous ones (transpose, strided slice, stride-0 expand) are materialized.
    for n in ["transposed", "col_slice", "expanded"] {
        let t = ck.tensor(n).unwrap();
        assert!(!t.is_contiguous(), "{n}");
        assert!(matches!(ck.tensor_bytes(t).unwrap(), Cow::Owned(_)), "{n}");
    }
}

#[test]
fn legacy_format_matches_torch() {
    check_against_torch("legacy_views.pt", Format::Legacy);
    check_against_torch("legacy_state_dict.pt", Format::Legacy);
}

#[test]
fn nested_checkpoint_is_walkable() {
    check_against_torch("nested.pt", Format::Zip);
    let ck = TorchCheckpoint::open(dir().join("nested.pt")).unwrap();
    let root = ck.root();
    assert!(matches!(root.get("model"), Some(Object::Dict(_))));
    assert!(matches!(root.get("ema"), Some(Object::Dict(_))));
    assert_eq!(root.get("step"), Some(&Object::Int(1234)));
    assert_eq!(root.get("neg"), Some(&Object::Int(-(1 << 40))));
    let mut big = vec![0u8; 9];
    big[8] = 0x40; // 2**70, little-endian two's complement
    assert_eq!(root.get("big"), Some(&Object::BigInt(big)));
    assert_eq!(root.get("lr"), Some(&Object::Float(3.5e-4)));
    assert_eq!(root.get("name"), Some(&Object::Str("tiny".into())));
    assert_eq!(root.get("done"), Some(&Object::Bool(false)));
    assert_eq!(root.get("nothing"), Some(&Object::None));
    assert_eq!(root.get("dtype"), Some(&Object::Dtype(DType::BF16)));
    assert_eq!(
        root.get("size"),
        Some(&Object::Tuple(vec![Object::Int(2), Object::Int(3)]))
    );
    assert_eq!(
        root.get("blob"),
        Some(&Object::Bytes(b"\x00\x01ab".to_vec()))
    );
    let Some(Object::List(tags)) = root.get("tags") else {
        panic!("tags")
    };
    let tags: BTreeSet<&str> = tags
        .iter()
        .map(|t| match t {
            Object::Str(s) => s.as_str(),
            other => panic!("tag {other:?}"),
        })
        .collect();
    assert_eq!(tags, BTreeSet::from(["x", "y"]));
    let lr = root
        .get("optimizer")
        .and_then(|o| match o.get("param_groups") {
            Some(Object::List(g)) => g.first(),
            _ => None,
        })
        .and_then(|g| g.get("lr"));
    assert_eq!(lr, Some(&Object::Float(0.001)));
    // The same Python object saved under two keys is one tensor record under two names.
    let (Some(Object::Tensor(a)), Some(Object::Tensor(b))) =
        (root.get("tied_a"), root.get("tied_b"))
    else {
        panic!("tied")
    };
    assert_eq!(a, b);
    assert!(ck.tensor("param").unwrap().requires_grad);
    assert!(!ck.tensor("tied_a").unwrap().requires_grad);
    // `list.1` is shared[1]: same storage as `tied_a`, offset 3.
    let l1 = ck.tensor("list.1").unwrap();
    assert_eq!(l1.storage, ck.tensor("tied_a").unwrap().storage);
    assert_eq!(l1.storage_offset, 3);
}

#[test]
fn pickle_protocol_4_matches_torch() {
    // FRAME, MEMOIZE, SHORT_BINUNICODE, STACK_GLOBAL, EMPTY_SET/ADDITEMS, SHORT_BINBYTES.
    check_against_torch("nested_proto4.pt", Format::Zip);
    let a = TorchCheckpoint::open(dir().join("nested.pt")).unwrap();
    let b = TorchCheckpoint::open(dir().join("nested_proto4.pt")).unwrap();
    assert_eq!(
        a.root(),
        b.root(),
        "protocol 2 and 4 decode to the same object"
    );
}

#[test]
fn malicious_global_is_rejected_and_named() {
    // data.pkl = GLOBAL 'os system' ... REDUCE — hand-assembled bytes, never executed anywhere.
    match TorchCheckpoint::open(dir().join("malicious.pt")) {
        Err(TorchCkptError::DisallowedGlobal(g)) => assert_eq!(g, "os.system"),
        other => panic!("expected DisallowedGlobal(os.system), got {other:?}"),
    }
}

#[test]
fn compressed_zip_entries_are_refused_clearly() {
    match TorchCheckpoint::open(dir().join("compressed.pt")) {
        Err(TorchCkptError::Unsupported(m)) => assert!(m.contains("compressed"), "{m}"),
        other => panic!("expected Unsupported(compressed), got {other:?}"),
    }
}

const ALL: [&str; 8] = [
    "state_dict.pt",
    "views.pt",
    "nested.pt",
    "nested_proto4.pt",
    "legacy_views.pt",
    "legacy_state_dict.pt",
    "malicious.pt",
    "compressed.pt",
];

#[test]
fn every_truncation_errors_without_panicking() {
    for f in ALL {
        let full = std::fs::read(dir().join(f)).unwrap();
        for len in 0..full.len() {
            let r = TorchCheckpoint::from_bytes(full[..len].to_vec());
            assert!(r.is_err(), "{f} truncated to {len} bytes parsed OK");
        }
    }
}

#[test]
fn every_single_byte_corruption_is_handled_without_panicking() {
    // Each byte flipped in turn (three masks each). The zip formats mostly fail on CRC/structure; the legacy
    // format has no checksum, so this drives the unpickler through thousands of malformed
    // opcode streams. The assertion is "returns", Ok or Err; a panic fails the test.
    let mut parsed_ok = 0usize;
    for f in ALL {
        let full = std::fs::read(dir().join(f)).unwrap();
        for (i, mask) in (0..full.len()).flat_map(|i| [0xffu8, 0x01, 0x80].map(|m| (i, m))) {
            let mut b = full.clone();
            b[i] ^= mask;
            if let Ok(ck) = TorchCheckpoint::from_bytes(b) {
                parsed_ok += 1;
                for (_, t) in ck.tensors() {
                    let _ = ck.tensor_bytes(t);
                }
                let _ = ck.verify_checksums();
            }
        }
    }
    assert!(
        parsed_ok > 0,
        "no corruption landed in tensor data — test is not exercising reads"
    );
}

#[test]
fn corrupt_storage_bytes_fail_the_checksum() {
    let full = std::fs::read(dir().join("state_dict.pt")).unwrap();
    let ck = TorchCheckpoint::from_bytes(full.clone()).unwrap();
    let t = ck.tensor("fc.weight").unwrap();
    let st = ck.storage_bytes(t.storage);
    let off = full
        .windows(st.len())
        .position(|w| w == st)
        .expect("storage bytes appear in the file");
    let mut b = full;
    b[off] ^= 1;
    let ck = TorchCheckpoint::from_bytes(b).unwrap(); // opening does not read storages
    assert!(ck.verify_checksums().is_err());
}

#[test]
fn not_a_checkpoint() {
    assert!(TorchCheckpoint::from_bytes(b"hello world".to_vec()).is_err());
    assert!(TorchCheckpoint::from_bytes(Vec::new()).is_err());
    // A plain pickle (no torch magic) is not a legacy checkpoint.
    assert!(TorchCheckpoint::from_bytes(b"\x80\x02K\x01.".to_vec()).is_err());
}
