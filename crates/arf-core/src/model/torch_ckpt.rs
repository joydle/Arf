//! A SAFE reader for PyTorch checkpoint files (`torch.save` output) — no code execution.
//!
//! `torch.save` writes a pickle. Loading a pickle in Python means *running* it: a `GLOBAL`
//! opcode imports any callable and `REDUCE` calls it, which is why a downloaded `.pt` is an
//! arbitrary-code-execution vector. This module never calls anything. It is a restricted
//! unpickler: a small stack machine that understands the opcodes `torch.save` emits for tensors
//! and plain containers, and resolves `GLOBAL`/`STACK_GLOBAL` against a fixed allow-list of
//! names whose effect it re-implements as data (`OrderedDict()` → an empty dict,
//! `_rebuild_tensor_v2(...)` → a tensor record, `torch.float16` → a dtype). Every other global is
//! refused with an error naming it ([`TorchCkptError::DisallowedGlobal`]) — the file is
//! rejected, not partially loaded.
//!
//! # Formats
//!
//! - **Zip** (torch >= 1.6, the default): `<prefix>/data.pkl` plus one `<prefix>/data/<key>` entry
//!   per storage, `<prefix>/byteorder`, `<prefix>/version`. torch writes every entry STORED
//!   (uncompressed, 64-byte aligned) with ZIP64 records; the zip is mmap'd and tensor bytes are
//!   borrowed straight from the map. A compressed or encrypted entry is an error, not a
//!   decompression.
//! - **Legacy** (`_use_new_zipfile_serialization=False`, torch < 1.6): five pickles back to back
//!   (magic number, protocol version 1001, sys_info, the object, the sorted storage keys) and then
//!   each storage as `i64 numel` + raw bytes. Also mmap'd and zero-copy. Legacy *storage views*
//!   (`view_metadata`, not written since torch 1.x) and the pre-0.4 tar format are refused.
//!
//! Only little-endian checkpoints are accepted (`byteorder` = `little`, or absent as in old torch).
//!
//! # Limits that are errors, not panics
//!
//! Truncated or corrupt input returns `Err`: every length is bounds-checked before slicing, no
//! allocation is sized from a length field without the bytes being present, containers live in a
//! flat arena (so a hostile file cannot build a structure whose recursive drop overflows the
//! stack), and the public [`Object`] tree is built with a depth cap and a node budget (so memo
//! references cannot fan a small file out exponentially).
//!
//! Clean-room: written from PyTorch's documented save format (`torch/serialization.py`) and
//! Python's `pickletools` opcode table. Fixtures are generated with CPU torch by
//! `tests/fixtures/torch_ckpt/gen_fixtures.py`.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use std::rc::Rc;

use memmap2::Mmap;

/// Errors from reading a PyTorch checkpoint.
#[derive(Debug, thiserror::Error)]
pub enum TorchCkptError {
    /// Opening or mapping the file failed.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// The file is truncated, corrupt, or not a checkpoint.
    #[error("malformed checkpoint: {0}")]
    Malformed(String),
    /// A well-formed feature this reader deliberately does not support.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// The pickle references a global outside the allow-list. The file is rejected.
    #[error("disallowed pickle global `{0}` — refusing to load (this reader never executes pickled callables)")]
    DisallowedGlobal(String),
}

type Result<T> = std::result::Result<T, TorchCkptError>;

fn bad<T>(msg: impl Into<String>) -> Result<T> {
    Err(TorchCkptError::Malformed(msg.into()))
}

/// Element type of a stored tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    /// `torch.float64`
    F64,
    /// `torch.float32`
    F32,
    /// `torch.float16`
    F16,
    /// `torch.bfloat16`
    BF16,
    /// `torch.int64`
    I64,
    /// `torch.int32`
    I32,
    /// `torch.int16`
    I16,
    /// `torch.int8`
    I8,
    /// `torch.uint8` (also the element type of an `UntypedStorage`)
    U8,
    /// `torch.bool` (one byte per element)
    Bool,
    /// `torch.complex64` (two f32)
    C64,
    /// `torch.complex128` (two f64)
    C128,
}

impl DType {
    /// Bytes per element.
    pub fn size(self) -> usize {
        match self {
            DType::F64 | DType::I64 | DType::C64 => 8,
            DType::C128 => 16,
            DType::F32 | DType::I32 => 4,
            DType::F16 | DType::BF16 | DType::I16 => 2,
            DType::I8 | DType::U8 | DType::Bool => 1,
        }
    }

    /// The torch name without the `torch.` prefix (`float32`, `bfloat16`, ...).
    pub fn name(self) -> &'static str {
        match self {
            DType::F64 => "float64",
            DType::F32 => "float32",
            DType::F16 => "float16",
            DType::BF16 => "bfloat16",
            DType::I64 => "int64",
            DType::I32 => "int32",
            DType::I16 => "int16",
            DType::I8 => "int8",
            DType::U8 => "uint8",
            DType::Bool => "bool",
            DType::C64 => "complex64",
            DType::C128 => "complex128",
        }
    }

    /// The legacy typed-storage class torch pickles for this dtype (`torch.FloatStorage` ...).
    fn from_storage_class(name: &str) -> Option<DType> {
        Some(match name {
            "DoubleStorage" => DType::F64,
            "FloatStorage" => DType::F32,
            "HalfStorage" => DType::F16,
            "BFloat16Storage" => DType::BF16,
            "LongStorage" => DType::I64,
            "IntStorage" => DType::I32,
            "ShortStorage" => DType::I16,
            "CharStorage" => DType::I8,
            "ByteStorage" | "UntypedStorage" => DType::U8,
            "BoolStorage" => DType::Bool,
            "ComplexFloatStorage" => DType::C64,
            "ComplexDoubleStorage" => DType::C128,
            _ => return None,
        })
    }

    /// A `torch.<dtype>` attribute, as a dtype object pickles (`GLOBAL 'torch float16'`).
    fn from_torch_attr(name: &str) -> Option<DType> {
        Some(match name {
            "float64" | "double" => DType::F64,
            "float32" | "float" => DType::F32,
            "float16" | "half" => DType::F16,
            "bfloat16" => DType::BF16,
            "int64" | "long" => DType::I64,
            "int32" | "int" => DType::I32,
            "int16" | "short" => DType::I16,
            "int8" => DType::I8,
            "uint8" => DType::U8,
            "bool" => DType::Bool,
            "complex64" | "cfloat" => DType::C64,
            "complex128" | "cdouble" => DType::C128,
            _ => return None,
        })
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Which on-disk container the checkpoint used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// torch >= 1.6 zip archive.
    Zip,
    /// Legacy back-to-back pickles followed by raw storages.
    Legacy,
}

/// One storage: a contiguous byte range of the file that one or more tensors view.
#[derive(Clone, Debug)]
pub struct StorageInfo {
    /// The storage key (`data/<key>` in a zip).
    pub key: String,
    /// Element type the storage was saved as.
    pub dtype: DType,
    /// Number of elements.
    pub numel: usize,
    /// Byte range within the file.
    range: std::ops::Range<usize>,
}

/// A tensor record: a strided view into one storage (torch's `_rebuild_tensor_v2` arguments).
#[derive(Clone, Debug)]
pub struct TensorInfo {
    /// Element type (always the storage's dtype).
    pub dtype: DType,
    /// Sizes, outermost first. Empty for a 0-dim scalar.
    pub shape: Vec<usize>,
    /// Strides in ELEMENTS, as torch reports them.
    pub stride: Vec<usize>,
    /// Offset of element 0 into the storage, in elements.
    pub storage_offset: usize,
    /// Index into [`TorchCheckpoint::storages`].
    pub storage: usize,
    /// Whether it was saved as a `Parameter`/with `requires_grad`.
    pub requires_grad: bool,
}

impl TensorInfo {
    /// Number of elements (1 for a scalar, 0 if any dim is 0).
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Bytes of the tensor's own elements (numel × element size).
    pub fn nbytes(&self) -> usize {
        self.numel() * self.dtype.size()
    }

    /// Row-major contiguous in the C sense (dims of size 1 may have any stride).
    pub fn is_contiguous(&self) -> bool {
        if self.numel() == 0 {
            return true;
        }
        let mut expected = 1usize;
        for (&n, &s) in self.shape.iter().zip(&self.stride).rev() {
            if n != 1 {
                if s != expected {
                    return false;
                }
                expected *= n;
            }
        }
        true
    }
}

/// A value from the checkpoint's pickle, with tensors replaced by indices.
///
/// This is the walkable form of whatever was saved: a bare state_dict is a `Dict` of
/// `Str → Tensor`; a training checkpoint is typically a `Dict` whose values include nested
/// state_dicts, numbers and strings.
#[derive(Clone, Debug, PartialEq)]
pub enum Object {
    /// `None`
    None,
    /// `True`/`False`
    Bool(bool),
    /// An int that fits in i64.
    Int(i64),
    /// A larger int, as little-endian two's-complement bytes (pickle `LONG1`/`LONG4`).
    BigInt(Vec<u8>),
    /// A Python float.
    Float(f64),
    /// A Python str.
    Str(String),
    /// A Python bytes.
    Bytes(Vec<u8>),
    /// A tuple (also `torch.Size`).
    Tuple(Vec<Object>),
    /// A list (also a `set`, in pickle order).
    List(Vec<Object>),
    /// A dict or `OrderedDict`, in insertion order.
    Dict(Vec<(Object, Object)>),
    /// A tensor: index into [`TorchCheckpoint::tensor_infos`].
    Tensor(usize),
    /// A `torch.dtype` value.
    Dtype(DType),
    /// A bare storage object (not wrapped in a tensor): index into [`TorchCheckpoint::storages`].
    Storage(usize),
    /// An allowed class object that appeared as a value rather than being called.
    Class(String),
}

impl Object {
    /// Look up `key` in a `Dict` with string keys.
    pub fn get(&self, key: &str) -> Option<&Object> {
        match self {
            Object::Dict(items) => items
                .iter()
                .find(|(k, _)| matches!(k, Object::Str(s) if s == key))
                .map(|(_, v)| v),
            _ => None,
        }
    }
}

// ─────────────────────────────── restricted unpickler ───────────────────────────────

/// An allow-listed global. Each has a fixed, data-only meaning; none is ever "called".
#[derive(Clone, Copy, Debug, PartialEq)]
enum Global {
    OrderedDict,
    Set,
    TorchSize,
    RebuildTensorV2,
    RebuildParameter,
    RebuildParameterWithState,
    CodecsEncode,
    StorageClass(DType),
}

impl Global {
    fn name(self) -> String {
        match self {
            Global::OrderedDict => "collections.OrderedDict".into(),
            Global::Set => "builtins.set".into(),
            Global::TorchSize => "torch.Size".into(),
            Global::RebuildTensorV2 => "torch._utils._rebuild_tensor_v2".into(),
            Global::RebuildParameter => "torch._utils._rebuild_parameter".into(),
            Global::RebuildParameterWithState => {
                "torch._utils._rebuild_parameter_with_state".into()
            }
            Global::CodecsEncode => "_codecs.encode".into(),
            Global::StorageClass(d) => format!("torch.<{d} storage>"),
        }
    }
}

/// The allow-list. Anything not matched here rejects the whole file.
fn resolve_global(module: &str, name: &str) -> Result<V> {
    let g = match (module, name) {
        ("collections", "OrderedDict") => Global::OrderedDict,
        ("builtins" | "__builtin__", "set") => Global::Set,
        ("torch", "Size") => Global::TorchSize,
        ("torch._utils", "_rebuild_tensor_v2") => Global::RebuildTensorV2,
        ("torch._utils", "_rebuild_parameter") => Global::RebuildParameter,
        ("torch._utils", "_rebuild_parameter_with_state") => Global::RebuildParameterWithState,
        // How protocol 2 pickles a `bytes` value: `_codecs.encode(<latin-1 str>, 'latin1')`.
        // Emulated only for exactly that argument shape (a pure data conversion).
        ("_codecs", "encode") => Global::CodecsEncode,
        ("torch.storage", "UntypedStorage") => Global::StorageClass(DType::U8),
        ("torch", n) => {
            if let Some(d) = DType::from_storage_class(n) {
                Global::StorageClass(d)
            } else if let Some(d) = DType::from_torch_attr(n) {
                return Ok(V::Dtype(d));
            } else {
                return Err(TorchCkptError::DisallowedGlobal(format!("{module}.{name}")));
            }
        }
        _ => return Err(TorchCkptError::DisallowedGlobal(format!("{module}.{name}"))),
    };
    Ok(V::Global(g))
}

/// A stack value. Shallow by construction: containers are arena indices, so values clone in
/// O(1) (memo `GET` shares like Python does) and nothing here drops recursively.
#[derive(Clone, Debug)]
enum V {
    None,
    Bool(bool),
    Int(i64),
    BigInt(Rc<[u8]>),
    Float(f64),
    Str(Rc<str>),
    Bytes(Rc<[u8]>),
    /// Arena index of a Tuple/List/Dict node.
    Node(usize),
    Global(Global),
    Dtype(DType),
    /// Index into the storage registry.
    Storage(usize),
    /// Index into the tensor records.
    Tensor(usize),
}

#[derive(Debug)]
enum Node {
    Tuple(Vec<V>),
    List(Vec<V>),
    Dict {
        items: Vec<(V, V)>,
        /// Str/Int key → position, so `SETITEM` on an existing key replaces (Python semantics)
        /// without an O(n) scan.
        index: HashMap<Key, usize>,
    },
}

#[derive(Hash, PartialEq, Eq, Debug)]
enum Key {
    Str(Rc<str>),
    Int(i64),
}

/// A storage referenced by a persistent id, before its bytes are located.
#[derive(Clone, Debug)]
struct StorageRef {
    key: String,
    dtype: DType,
    numel: usize,
}

/// Shared state across the pickles of one file (the legacy format has five).
#[derive(Default)]
struct Heap {
    nodes: Vec<Node>,
    storages: Vec<StorageRef>,
    storage_by_key: HashMap<String, usize>,
    tensors: Vec<TensorInfo>,
}

impl Heap {
    fn alloc(&mut self, n: Node) -> V {
        self.nodes.push(n);
        V::Node(self.nodes.len() - 1)
    }

    fn tuple(&self, v: &V) -> Option<&[V]> {
        match v {
            V::Node(i) => match &self.nodes[*i] {
                Node::Tuple(t) => Some(t),
                _ => None,
            },
            _ => None,
        }
    }

    fn usize_tuple(&self, v: &V, what: &str) -> Result<Vec<usize>> {
        let Some(items) = self.tuple(v) else {
            return bad(format!("{what}: expected a tuple"));
        };
        items.iter().map(|x| as_usize(x, what)).collect()
    }
}

fn as_usize(v: &V, what: &str) -> Result<usize> {
    match v {
        V::Int(i) if *i >= 0 => usize::try_from(*i).or_else(|_| bad(format!("{what}: too large"))),
        _ => bad(format!("{what}: expected a non-negative int, got {v:?}")),
    }
}

/// Pickle opcode reader over `data[pos..]`.
struct Unpickler<'a, 'h> {
    data: &'a [u8],
    pos: usize,
    stack: Vec<V>,
    marks: Vec<usize>,
    memo: HashMap<u64, V>,
    heap: &'h mut Heap,
    /// Whether BINPERSID (a storage reference) is legal in this pickle.
    allow_persid: bool,
}

impl<'a, 'h> Unpickler<'a, 'h> {
    fn new(data: &'a [u8], pos: usize, heap: &'h mut Heap, allow_persid: bool) -> Self {
        Unpickler {
            data,
            pos,
            stack: Vec::new(),
            marks: Vec::new(),
            memo: HashMap::new(),
            heap,
            allow_persid,
        }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.data.len());
        match end {
            Some(end) => {
                let s = &self.data[self.pos..end];
                self.pos = end;
                Ok(s)
            }
            None => bad(format!(
                "pickle truncated: need {n} bytes at offset {}",
                self.pos
            )),
        }
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn len_u64(&mut self) -> Result<usize> {
        let n = self.u64()?;
        usize::try_from(n).or_else(|_| bad("pickle length does not fit usize"))
    }
    fn line(&mut self) -> Result<&'a str> {
        let rest = &self.data[self.pos..];
        let Some(nl) = rest.iter().position(|&b| b == b'\n') else {
            return bad("pickle truncated: unterminated GLOBAL line");
        };
        let s = std::str::from_utf8(&rest[..nl]).or_else(|_| bad("non-UTF-8 GLOBAL name"))?;
        self.pos += nl + 1;
        Ok(s)
    }

    fn pop(&mut self) -> Result<V> {
        let floor = self.marks.last().copied().unwrap_or(0);
        if self.stack.len() <= floor {
            return bad("pickle stack underflow");
        }
        Ok(self.stack.pop().unwrap())
    }
    fn top(&self) -> Result<&V> {
        let floor = self.marks.last().copied().unwrap_or(0);
        if self.stack.len() <= floor {
            return bad("pickle stack underflow");
        }
        Ok(self.stack.last().unwrap())
    }
    fn pop_mark(&mut self) -> Result<Vec<V>> {
        let Some(m) = self.marks.pop() else {
            return bad("pickle: no MARK on the stack");
        };
        Ok(self.stack.split_off(m))
    }

    fn string(bytes: &[u8]) -> Result<V> {
        match std::str::from_utf8(bytes) {
            Ok(s) => Ok(V::Str(Rc::from(s))),
            Err(_) => bad("pickle: invalid UTF-8 in a str"),
        }
    }

    fn long(bytes: &[u8]) -> V {
        if bytes.is_empty() {
            return V::Int(0);
        }
        if bytes.len() <= 8 {
            let neg = bytes[bytes.len() - 1] & 0x80 != 0;
            let mut buf = if neg { [0xffu8; 8] } else { [0u8; 8] };
            buf[..bytes.len()].copy_from_slice(bytes);
            return V::Int(i64::from_le_bytes(buf));
        }
        V::BigInt(Rc::from(bytes))
    }

    /// Run until STOP; returns the unpickled value. `self.pos` is left just past STOP.
    fn run(&mut self) -> Result<V> {
        loop {
            let op = self.u8()?;
            match op {
                0x80 => {
                    // PROTO
                    let p = self.u8()?;
                    if p > 5 {
                        return Err(TorchCkptError::Unsupported(format!("pickle protocol {p}")));
                    }
                }
                0x95 => {
                    // FRAME: a length hint only; the opcodes follow inline.
                    self.u64()?;
                }
                b'.' => return self.pop(),                 // STOP
                b'(' => self.marks.push(self.stack.len()), // MARK
                b'0' => {
                    self.pop()?; // POP
                }
                b'1' => {
                    self.pop_mark()?; // POP_MARK
                }
                b'2' => {
                    let v = self.top()?.clone(); // DUP
                    self.stack.push(v);
                }
                b'N' => self.stack.push(V::None),
                0x88 => self.stack.push(V::Bool(true)),
                0x89 => self.stack.push(V::Bool(false)),
                b'J' => {
                    let v = i32::from_le_bytes(self.take(4)?.try_into().unwrap());
                    self.stack.push(V::Int(v as i64));
                }
                b'K' => {
                    let v = self.u8()?;
                    self.stack.push(V::Int(v as i64));
                }
                b'M' => {
                    let v = self.u16()?;
                    self.stack.push(V::Int(v as i64));
                }
                0x8a => {
                    // LONG1
                    let n = self.u8()? as usize;
                    let b = self.take(n)?;
                    self.stack.push(Self::long(b));
                }
                0x8b => {
                    // LONG4
                    let n = self.u32()? as usize;
                    let b = self.take(n)?;
                    self.stack.push(Self::long(b));
                }
                b'G' => {
                    let v = f64::from_be_bytes(self.take(8)?.try_into().unwrap());
                    self.stack.push(V::Float(v));
                }
                b'X' => {
                    let n = self.u32()? as usize;
                    let b = self.take(n)?;
                    self.stack.push(Self::string(b)?);
                }
                0x8c => {
                    let n = self.u8()? as usize;
                    let b = self.take(n)?;
                    self.stack.push(Self::string(b)?);
                }
                0x8d => {
                    let n = self.len_u64()?;
                    let b = self.take(n)?;
                    self.stack.push(Self::string(b)?);
                }
                b'T' | b'U' => {
                    // BINSTRING / SHORT_BINSTRING (py2 str): decode as latin-1.
                    let n = if op == b'T' {
                        self.u32()? as usize
                    } else {
                        self.u8()? as usize
                    };
                    let b = self.take(n)?;
                    let s: String = b.iter().map(|&c| c as char).collect();
                    self.stack.push(V::Str(Rc::from(s)));
                }
                b'B' | b'C' | 0x8e => {
                    let n = match op {
                        b'B' => self.u32()? as usize,
                        b'C' => self.u8()? as usize,
                        _ => self.len_u64()?,
                    };
                    let b = self.take(n)?;
                    self.stack.push(V::Bytes(Rc::from(b)));
                }
                b'}' => {
                    let v = self.heap.alloc(Node::Dict {
                        items: Vec::new(),
                        index: HashMap::new(),
                    });
                    self.stack.push(v);
                }
                b']' => {
                    let v = self.heap.alloc(Node::List(Vec::new()));
                    self.stack.push(v);
                }
                b')' => {
                    let v = self.heap.alloc(Node::Tuple(Vec::new()));
                    self.stack.push(v);
                }
                0x8f => {
                    // EMPTY_SET → list
                    let v = self.heap.alloc(Node::List(Vec::new()));
                    self.stack.push(v);
                }
                b't' | 0x91 | b'l' => {
                    // TUPLE / FROZENSET / LIST from the items since MARK
                    let items = self.pop_mark()?;
                    let node = if op == b'l' {
                        Node::List(items)
                    } else {
                        Node::Tuple(items)
                    };
                    let v = self.heap.alloc(node);
                    self.stack.push(v);
                }
                0x85..=0x87 => {
                    // TUPLE1..3
                    let n = (op - 0x84) as usize;
                    let mut items = Vec::with_capacity(n);
                    for _ in 0..n {
                        items.push(self.pop()?);
                    }
                    items.reverse();
                    let v = self.heap.alloc(Node::Tuple(items));
                    self.stack.push(v);
                }
                b'd' => {
                    // DICT from key/value pairs since MARK
                    let items = self.pop_mark()?;
                    let v = self.heap.alloc(Node::Dict {
                        items: Vec::new(),
                        index: HashMap::new(),
                    });
                    self.set_items(&v, items)?;
                    self.stack.push(v);
                }
                b'a' => {
                    let item = self.pop()?;
                    let list = self.top()?.clone();
                    self.append(&list, vec![item])?;
                }
                b'e' | 0x90 => {
                    // APPENDS / ADDITEMS
                    let items = self.pop_mark()?;
                    let list = self.top()?.clone();
                    self.append(&list, items)?;
                }
                b's' => {
                    let v = self.pop()?;
                    let k = self.pop()?;
                    let d = self.top()?.clone();
                    self.set_items(&d, vec![k, v])?;
                }
                b'u' => {
                    let items = self.pop_mark()?;
                    let d = self.top()?.clone();
                    self.set_items(&d, items)?;
                }
                b'c' => {
                    let module = self.line()?;
                    let name = self.line()?;
                    let g = resolve_global(module, name)?;
                    self.stack.push(g);
                }
                0x93 => {
                    // STACK_GLOBAL
                    let name = self.pop()?;
                    let module = self.pop()?;
                    let (V::Str(m), V::Str(n)) = (&module, &name) else {
                        return bad("STACK_GLOBAL: module and name must be str");
                    };
                    let g = resolve_global(m, n)?;
                    self.stack.push(g);
                }
                b'R' => {
                    let args = self.pop()?;
                    let func = self.pop()?;
                    let v = self.reduce(func, args)?;
                    self.stack.push(v);
                }
                b'b' => {
                    // BUILD: set state on the object below. The only objects torch BUILDs are
                    // OrderedDicts (the state_dict `_metadata` attribute) and tensors; the state
                    // is not tensor data, so it is dropped.
                    let _state = self.pop()?;
                    match self.top()? {
                        V::Node(_) | V::Tensor(_) => {}
                        other => return bad(format!("BUILD on unsupported object {other:?}")),
                    }
                }
                b'Q' => {
                    let pid = self.pop()?;
                    if !self.allow_persid {
                        return bad("persistent id outside the object pickle");
                    }
                    let v = self.persistent_load(&pid)?;
                    self.stack.push(v);
                }
                b'q' => {
                    let i = self.u8()? as u64;
                    let v = self.top()?.clone();
                    self.memo.insert(i, v);
                }
                b'r' => {
                    let i = self.u32()? as u64;
                    let v = self.top()?.clone();
                    self.memo.insert(i, v);
                }
                0x94 => {
                    // MEMOIZE
                    let i = self.memo.len() as u64;
                    let v = self.top()?.clone();
                    self.memo.insert(i, v);
                }
                b'h' | b'j' => {
                    let i = if op == b'h' {
                        self.u8()? as u64
                    } else {
                        self.u32()? as u64
                    };
                    let Some(v) = self.memo.get(&i) else {
                        return bad(format!("pickle memo key {i} not found"));
                    };
                    self.stack.push(v.clone());
                }
                other => {
                    return Err(TorchCkptError::Unsupported(format!(
                        "pickle opcode 0x{other:02x} at offset {} (this reader implements the \
                         protocol 2-5 subset torch.save emits; NEWOBJ/INST/OBJ/EXT* and \
                         protocol-0 text opcodes are refused)",
                        self.pos - 1
                    )))
                }
            }
        }
    }

    fn append(&mut self, list: &V, items: Vec<V>) -> Result<()> {
        match list {
            V::Node(i) => match &mut self.heap.nodes[*i] {
                Node::List(l) => {
                    l.extend(items);
                    Ok(())
                }
                _ => bad("APPEND to a non-list"),
            },
            _ => bad("APPEND to a non-list"),
        }
    }

    fn set_items(&mut self, dict: &V, items: Vec<V>) -> Result<()> {
        if !items.len().is_multiple_of(2) {
            return bad("SETITEMS with an odd number of items");
        }
        let V::Node(i) = dict else {
            return bad("SETITEM on a non-dict");
        };
        let Node::Dict { items: d, index } = &mut self.heap.nodes[*i] else {
            return bad("SETITEM on a non-dict");
        };
        let mut it = items.into_iter();
        while let (Some(k), Some(v)) = (it.next(), it.next()) {
            let key = match &k {
                V::Str(s) => Some(Key::Str(s.clone())),
                V::Int(n) => Some(Key::Int(*n)),
                _ => None,
            };
            match key.as_ref().and_then(|key| index.get(key)) {
                Some(&pos) => d[pos].1 = v,
                None => {
                    if let Some(key) = key {
                        index.insert(key, d.len());
                    }
                    d.push((k, v));
                }
            }
        }
        Ok(())
    }

    fn reduce(&mut self, func: V, args: V) -> Result<V> {
        let V::Global(g) = func else {
            return bad(format!("REDUCE on a non-callable {func:?}"));
        };
        let Some(a) = self.heap.tuple(&args).map(|a| a.to_vec()) else {
            return bad(format!("REDUCE {}: args are not a tuple", g.name()));
        };
        match g {
            Global::OrderedDict => {
                if !a.is_empty() {
                    return Err(TorchCkptError::Unsupported(
                        "OrderedDict(...) with constructor arguments".into(),
                    ));
                }
                Ok(self.heap.alloc(Node::Dict {
                    items: Vec::new(),
                    index: HashMap::new(),
                }))
            }
            Global::Set => {
                let items = match a.as_slice() {
                    [] => Vec::new(),
                    [V::Node(i)] => match &self.heap.nodes[*i] {
                        Node::List(l) | Node::Tuple(l) => l.clone(),
                        _ => return bad("set(...) of a non-sequence"),
                    },
                    _ => return bad("set(...) with unexpected arguments"),
                };
                Ok(self.heap.alloc(Node::List(items)))
            }
            Global::TorchSize => match a.as_slice() {
                [t] if self.heap.tuple(t).is_some() => Ok(t.clone()),
                _ => bad("torch.Size(...) expects one tuple"),
            },
            Global::CodecsEncode => match a.as_slice() {
                [V::Str(s), V::Str(enc)] if &**enc == "latin1" || &**enc == "latin-1" => {
                    let mut out = Vec::with_capacity(s.len());
                    for c in s.chars() {
                        let c = c as u32;
                        if c > 0xff {
                            return bad("_codecs.encode: char outside latin-1");
                        }
                        out.push(c as u8);
                    }
                    Ok(V::Bytes(Rc::from(out)))
                }
                _ => Err(TorchCkptError::DisallowedGlobal(
                    "_codecs.encode (only the (str, 'latin1') form torch uses for bytes is allowed)"
                        .into(),
                )),
            },
            Global::RebuildTensorV2 => {
                // (storage, storage_offset, size, stride, requires_grad, backward_hooks[, metadata])
                if a.len() != 6 && a.len() != 7 {
                    return bad(format!("_rebuild_tensor_v2 with {} args", a.len()));
                }
                let V::Storage(storage) = a[0] else {
                    return bad("_rebuild_tensor_v2: first argument is not a storage");
                };
                let storage_offset = as_usize(&a[1], "storage_offset")?;
                let shape = self.heap.usize_tuple(&a[2], "size")?;
                let stride = self.heap.usize_tuple(&a[3], "stride")?;
                if shape.len() != stride.len() {
                    return bad("tensor size and stride have different ranks");
                }
                let requires_grad = matches!(a[4], V::Bool(true));
                let dtype = self.heap.storages[storage].dtype;
                self.heap.tensors.push(TensorInfo {
                    dtype,
                    shape,
                    stride,
                    storage_offset,
                    storage,
                    requires_grad,
                });
                Ok(V::Tensor(self.heap.tensors.len() - 1))
            }
            Global::RebuildParameter | Global::RebuildParameterWithState => {
                // (data, requires_grad, backward_hooks[, state])
                let want = if g == Global::RebuildParameter { 3 } else { 4 };
                if a.len() != want {
                    return bad(format!("{} with {} args", g.name(), a.len()));
                }
                let V::Tensor(t) = a[0] else {
                    return bad(format!("{}: first argument is not a tensor", g.name()));
                };
                self.heap.tensors[t].requires_grad = matches!(a[1], V::Bool(true));
                Ok(V::Tensor(t))
            }
            Global::StorageClass(_) => bad(format!(
                "REDUCE on {} — storage classes are only valid inside a persistent id",
                g.name()
            )),
        }
    }

    /// `('storage', storage_type, key, location, numel)` (zip) or the legacy 6-tuple with a
    /// trailing `view_metadata`.
    fn persistent_load(&mut self, pid: &V) -> Result<V> {
        let Some(t) = self.heap.tuple(pid).map(|t| t.to_vec()) else {
            return bad("persistent id is not a tuple");
        };
        let tag_ok = matches!(t.first(), Some(V::Str(s)) if &**s == "storage");
        if !tag_ok || !(t.len() == 5 || t.len() == 6) {
            return bad("persistent id is not a ('storage', ...) tuple");
        }
        let V::Global(Global::StorageClass(dtype)) = t[1] else {
            return bad(format!("persistent id storage type is {:?}", t[1]));
        };
        let key = match &t[2] {
            V::Str(s) => s.to_string(),
            _ => return bad("persistent id storage key is not a str"),
        };
        let numel = as_usize(&t[4], "storage numel")?;
        if t.len() == 6 && !matches!(t[5], V::None) {
            return Err(TorchCkptError::Unsupported(
                "legacy storage views (view_metadata) — not written since torch 1.x".into(),
            ));
        }
        if let Some(&i) = self.heap.storage_by_key.get(&key) {
            let s = &self.heap.storages[i];
            if s.dtype != dtype || s.numel != numel {
                return bad(format!(
                    "storage {key} referenced as both {}×{} and {dtype}×{numel}",
                    s.dtype, s.numel
                ));
            }
            return Ok(V::Storage(i));
        }
        self.heap.storages.push(StorageRef {
            key: key.clone(),
            dtype,
            numel,
        });
        let i = self.heap.storages.len() - 1;
        self.heap.storage_by_key.insert(key, i);
        Ok(V::Storage(i))
    }
}

/// Deepest nesting accepted in the object tree (a real checkpoint is < 10 deep).
const MAX_DEPTH: usize = 128;

/// Convert the arena graph into an owned tree, depth-capped and node-budgeted.
fn to_object(heap: &Heap, v: &V, depth: usize, budget: &mut usize) -> Result<Object> {
    if depth > MAX_DEPTH {
        return bad(format!(
            "object nesting deeper than {MAX_DEPTH} (or a reference cycle)"
        ));
    }
    if *budget == 0 {
        return bad("object tree too large (shared references expand beyond the node budget)");
    }
    *budget -= 1;
    Ok(match v {
        V::None => Object::None,
        V::Bool(b) => Object::Bool(*b),
        V::Int(i) => Object::Int(*i),
        V::BigInt(b) => Object::BigInt(b.to_vec()),
        V::Float(f) => Object::Float(*f),
        V::Str(s) => Object::Str(s.to_string()),
        V::Bytes(b) => Object::Bytes(b.to_vec()),
        V::Dtype(d) => Object::Dtype(*d),
        V::Storage(i) => Object::Storage(*i),
        V::Tensor(i) => Object::Tensor(*i),
        V::Global(g) => Object::Class(g.name()),
        V::Node(i) => {
            let conv = |xs: &[V], budget: &mut usize| -> Result<Vec<Object>> {
                xs.iter()
                    .map(|x| to_object(heap, x, depth + 1, budget))
                    .collect()
            };
            match &heap.nodes[*i] {
                Node::Tuple(xs) => Object::Tuple(conv(xs, budget)?),
                Node::List(xs) => Object::List(conv(xs, budget)?),
                Node::Dict { items, .. } => {
                    let mut out = Vec::with_capacity(items.len());
                    for (k, v) in items {
                        out.push((
                            to_object(heap, k, depth + 1, budget)?,
                            to_object(heap, v, depth + 1, budget)?,
                        ));
                    }
                    Object::Dict(out)
                }
            }
        }
    })
}

// ─────────────────────────────── zip (stored entries only) ───────────────────────────────

fn rd_u16(b: &[u8], at: usize) -> Result<u16> {
    b.get(at..at + 2)
        .map(|s| u16::from_le_bytes(s.try_into().unwrap()))
        .ok_or_else(|| TorchCkptError::Malformed(format!("zip truncated at {at}")))
}
fn rd_u32(b: &[u8], at: usize) -> Result<u32> {
    b.get(at..at + 4)
        .map(|s| u32::from_le_bytes(s.try_into().unwrap()))
        .ok_or_else(|| TorchCkptError::Malformed(format!("zip truncated at {at}")))
}
fn rd_u64(b: &[u8], at: usize) -> Result<u64> {
    b.get(at..at + 8)
        .map(|s| u64::from_le_bytes(s.try_into().unwrap()))
        .ok_or_else(|| TorchCkptError::Malformed(format!("zip truncated at {at}")))
}
fn to_usize(v: u64) -> Result<usize> {
    usize::try_from(v).or_else(|_| bad("zip offset does not fit usize"))
}

#[derive(Debug)]
struct ZipEntry {
    name: String,
    /// Data range in the file (stored entries: compressed == uncompressed).
    range: std::ops::Range<usize>,
    crc32: u32,
}

/// Parse the central directory. Every entry must be STORED and unencrypted.
fn zip_entries(b: &[u8]) -> Result<Vec<ZipEntry>> {
    const EOCD: u32 = 0x0605_4b50;
    const EOCD64: u32 = 0x0606_4b50;
    const LOC64: u32 = 0x0706_4b50;
    const CEN: u32 = 0x0201_4b50;
    const LOC: u32 = 0x0403_4b50;
    if b.len() < 22 {
        return bad("zip too short for an end-of-central-directory record");
    }
    // The EOCD is 22 bytes + an up-to-64 KiB comment; scan back for its signature.
    let lo = b.len().saturating_sub(22 + 0xffff);
    let eocd = (lo..=b.len() - 22)
        .rev()
        .find(|&i| rd_u32(b, i).ok() == Some(EOCD))
        .ok_or_else(|| {
            TorchCkptError::Malformed("zip end-of-central-directory not found".into())
        })?;
    let mut count = rd_u16(b, eocd + 10)? as u64;
    let mut cd_size = rd_u32(b, eocd + 12)? as u64;
    let mut cd_off = rd_u32(b, eocd + 16)? as u64;
    // ZIP64: torch always writes the locator; prefer its values when present.
    if eocd >= 20 && rd_u32(b, eocd - 20)? == LOC64 {
        let e64 = to_usize(rd_u64(b, eocd - 20 + 8)?)?;
        if rd_u32(b, e64)? != EOCD64 {
            return bad("zip64 end-of-central-directory signature mismatch");
        }
        count = rd_u64(b, e64 + 32)?;
        cd_size = rd_u64(b, e64 + 40)?;
        cd_off = rd_u64(b, e64 + 48)?;
    }
    let cd_off = to_usize(cd_off)?;
    let cd_end = cd_off
        .checked_add(to_usize(cd_size)?)
        .filter(|&e| e <= b.len())
        .ok_or_else(|| TorchCkptError::Malformed("zip central directory out of bounds".into()))?;
    // Each central record is >= 46 bytes, so `count` cannot demand more than the bytes hold.
    if count > (cd_end - cd_off) as u64 / 46 {
        return bad("zip entry count exceeds the central directory size");
    }
    let mut out = Vec::with_capacity(count as usize);
    let mut p = cd_off;
    for _ in 0..count {
        if rd_u32(b, p)? != CEN {
            return bad(format!("zip central directory record signature at {p}"));
        }
        let flags = rd_u16(b, p + 8)?;
        let method = rd_u16(b, p + 10)?;
        let crc32 = rd_u32(b, p + 16)?;
        let mut csize = rd_u32(b, p + 20)? as u64;
        let mut usize_ = rd_u32(b, p + 24)? as u64;
        let nlen = rd_u16(b, p + 28)? as usize;
        let xlen = rd_u16(b, p + 30)? as usize;
        let clen = rd_u16(b, p + 32)? as usize;
        let mut loc = rd_u32(b, p + 42)? as u64;
        let name_bytes = b
            .get(p + 46..p + 46 + nlen)
            .ok_or_else(|| TorchCkptError::Malformed("zip entry name out of bounds".into()))?;
        let name = String::from_utf8_lossy(name_bytes).into_owned();
        // ZIP64 extra field (id 1): the 64-bit values, in order, for each 0xFFFFFFFF field.
        let extra = b
            .get(p + 46 + nlen..p + 46 + nlen + xlen)
            .ok_or_else(|| TorchCkptError::Malformed("zip extra field out of bounds".into()))?;
        let mut x = 0;
        while x + 4 <= extra.len() {
            let id = rd_u16(extra, x)?;
            let sz = rd_u16(extra, x + 2)? as usize;
            if id == 1 {
                let mut q = x + 4;
                for field in [&mut usize_, &mut csize, &mut loc] {
                    if *field == 0xffff_ffff {
                        *field = rd_u64(extra, q)?;
                        q += 8;
                    }
                }
            }
            x += 4 + sz;
        }
        if flags & 1 != 0 {
            return Err(TorchCkptError::Unsupported(format!(
                "zip entry {name} is encrypted"
            )));
        }
        if method != 0 {
            return Err(TorchCkptError::Unsupported(format!(
                "zip entry {name} is compressed (method {method}); torch.save writes STORED \
                 entries — re-save the checkpoint with torch.save, do not re-zip it"
            )));
        }
        if csize != usize_ {
            return bad(format!("stored zip entry {name} has csize != size"));
        }
        let loc = to_usize(loc)?;
        if rd_u32(b, loc)? != LOC {
            return bad(format!("zip local header signature for {name}"));
        }
        let data = loc + 30 + rd_u16(b, loc + 26)? as usize + rd_u16(b, loc + 28)? as usize;
        let end = data
            .checked_add(to_usize(usize_)?)
            .filter(|&e| e <= b.len())
            .ok_or_else(|| TorchCkptError::Malformed(format!("zip entry {name} out of bounds")))?;
        out.push(ZipEntry {
            name,
            range: data..end,
            crc32,
        });
        p += 46 + nlen + xlen + clen;
    }
    Ok(out)
}

/// CRC-32 (IEEE, reflected), as zip uses.
fn crc32(data: &[u8]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let t = TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for (i, e) in t.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 {
                    0xedb8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
            }
            *e = c;
        }
        t
    });
    let mut c = !0u32;
    for &b in data {
        c = t[((c ^ b as u32) & 0xff) as usize] ^ (c >> 8);
    }
    !c
}

// ─────────────────────────────── the checkpoint ───────────────────────────────

enum Backing {
    Mmap(Mmap),
    Owned(Vec<u8>),
}

impl Backing {
    fn bytes(&self) -> &[u8] {
        match self {
            Backing::Mmap(m) => m,
            Backing::Owned(v) => v,
        }
    }
}

/// A parsed PyTorch checkpoint. Tensor bytes are borrowed from the mapped file.
pub struct TorchCheckpoint {
    backing: Backing,
    format: Format,
    root: Object,
    storages: Vec<StorageInfo>,
    tensors: Vec<TensorInfo>,
    names: Vec<(String, usize)>,
    /// Zip only: (range, crc) of each storage entry, for [`Self::verify_checksums`].
    crcs: Vec<(std::ops::Range<usize>, u32)>,
}

impl fmt::Debug for TorchCheckpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TorchCheckpoint")
            .field("format", &self.format)
            .field("tensors", &self.names.len())
            .field("storages", &self.storages.len())
            .finish()
    }
}

/// Legacy-format magic number (`torch.serialization.MAGIC_NUMBER`), little-endian two's
/// complement as pickled by LONG1.
const LEGACY_MAGIC: u128 = 0x1950a86a20f9469cfc6c;
const LEGACY_PROTOCOL: i64 = 1001;

impl TorchCheckpoint {
    /// Map and parse the checkpoint at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = std::fs::File::open(path.as_ref())?;
        // SAFETY: read-only private map of a file we do not write. As with every mmap'd
        // weight file in this crate, a concurrent writer truncating it would be a bug outside
        // this process's control.
        let mmap = unsafe { Mmap::map(&file)? };
        Self::parse(Backing::Mmap(mmap))
    }

    /// Parse a checkpoint held in memory.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        Self::parse(Backing::Owned(bytes))
    }

    fn parse(backing: Backing) -> Result<Self> {
        let b = backing.bytes();
        let mut heap = Heap::default();
        let (format, root, ranges, crcs) = if b.starts_with(b"PK\x03\x04") {
            let (root, ranges, crcs) = parse_zip(b, &mut heap)?;
            (Format::Zip, root, ranges, crcs)
        } else if b.starts_with(&[0x80]) {
            let (root, ranges) = parse_legacy(b, &mut heap)?;
            (Format::Legacy, root, ranges, Vec::new())
        } else if b.len() > 262 && &b[257..262] == b"ustar" {
            return Err(TorchCkptError::Unsupported(
                "the pre-0.4 tar checkpoint format".into(),
            ));
        } else {
            return bad("not a PyTorch checkpoint (neither a zip nor a pickle)");
        };

        let storages: Vec<StorageInfo> = heap
            .storages
            .iter()
            .zip(ranges)
            .map(|(s, range)| StorageInfo {
                key: s.key.clone(),
                dtype: s.dtype,
                numel: s.numel,
                range,
            })
            .collect();
        for t in &heap.tensors {
            check_bounds(t, &storages[t.storage])?;
        }
        let mut budget = 8 * b.len() + 1024;
        let root = to_object(&heap, &root, 0, &mut budget)?;
        let mut names = Vec::new();
        collect_names(&root, &mut String::new(), &mut names);
        let tensors = std::mem::take(&mut heap.tensors);
        Ok(TorchCheckpoint {
            backing,
            format,
            root,
            storages,
            tensors,
            names,
            crcs,
        })
    }

    /// Which container format the file used.
    pub fn format(&self) -> Format {
        self.format
    }

    /// The unpickled object, for walking nested checkpoints (`{"model": ..., "ema": ...}`).
    pub fn root(&self) -> &Object {
        &self.root
    }

    /// Every tensor reachable from the root, as (dotted path, info), in pickle order. Nested
    /// dict keys are joined with `.` and list/tuple positions appear as their index, so
    /// `{"model": {"fc.weight": t}}` yields `model.fc.weight`. A tensor saved under two names
    /// (tied weights) appears under both.
    pub fn tensors(&self) -> impl Iterator<Item = (&str, &TensorInfo)> {
        self.names
            .iter()
            .map(|(n, i)| (n.as_str(), &self.tensors[*i]))
    }

    /// Look up one tensor by its dotted path.
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.names
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, i)| &self.tensors[*i])
    }

    /// All tensor records, indexed by [`Object::Tensor`].
    pub fn tensor_infos(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// All storages, indexed by [`TensorInfo::storage`] / [`Object::Storage`].
    pub fn storages(&self) -> &[StorageInfo] {
        &self.storages
    }

    /// The whole storage's bytes (zero-copy).
    pub fn storage_bytes(&self, storage: usize) -> &[u8] {
        &self.backing.bytes()[self.storages[storage].range.clone()]
    }

    /// The tensor's elements in row-major order. Borrowed from the map when the view is
    /// contiguous; materialized (gathered through the strides) when it is not. Errors only if
    /// the materialized copy cannot be allocated (a stride-0 view may expand far past its
    /// storage).
    pub fn tensor_bytes(&self, t: &TensorInfo) -> Result<Cow<'_, [u8]>> {
        let es = t.dtype.size();
        let storage = self.storage_bytes(t.storage);
        if t.numel() == 0 {
            // An empty view may carry any offset (bounds are only checked for addressed
            // elements) — found by the byte-flip test, which set one past an empty storage.
            return Ok(Cow::Borrowed(&[]));
        }
        if t.is_contiguous() {
            let start = t.storage_offset * es;
            return Ok(Cow::Borrowed(&storage[start..start + t.nbytes()]));
        }
        let n = t.numel();
        let mut out = Vec::new();
        n.checked_mul(es)
            .and_then(|total| out.try_reserve_exact(total).ok())
            .ok_or_else(|| {
                TorchCkptError::Unsupported(format!(
                    "materializing a {n}-element non-contiguous view does not fit in memory"
                ))
            })?;
        let rank = t.shape.len();
        let mut idx = vec![0usize; rank];
        for _ in 0..n {
            let off =
                t.storage_offset + idx.iter().zip(&t.stride).map(|(i, s)| i * s).sum::<usize>();
            out.extend_from_slice(&storage[off * es..off * es + es]);
            for d in (0..rank).rev() {
                idx[d] += 1;
                if idx[d] < t.shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        Ok(Cow::Owned(out))
    }

    /// Check every zip storage entry's CRC-32 (reads every byte; opening does not). The legacy
    /// format has no checksums, so this is a no-op there.
    pub fn verify_checksums(&self) -> Result<()> {
        for (range, want) in &self.crcs {
            let got = crc32(&self.backing.bytes()[range.clone()]);
            if got != *want {
                return bad(format!(
                    "CRC mismatch in storage at {range:?}: {got:08x} != {want:08x}"
                ));
            }
        }
        Ok(())
    }
}

/// Every element a tensor can address must lie inside its storage.
fn check_bounds(t: &TensorInfo, s: &StorageInfo) -> Result<()> {
    let overflow = || TorchCkptError::Malformed("tensor extent overflows".into());
    let numel = t
        .shape
        .iter()
        .try_fold(1usize, |a, &n| a.checked_mul(n))
        .ok_or_else(overflow)?;
    if numel == 0 {
        return Ok(());
    }
    let mut last = t.storage_offset;
    for (&n, &st) in t.shape.iter().zip(&t.stride) {
        last = (n - 1)
            .checked_mul(st)
            .and_then(|x| last.checked_add(x))
            .ok_or_else(overflow)?;
    }
    if last >= s.numel {
        return bad(format!(
            "tensor view reaches element {last} of storage {} which has {}",
            s.key, s.numel
        ));
    }
    Ok(())
}

fn collect_names(o: &Object, path: &mut String, out: &mut Vec<(String, usize)>) {
    let mut child = |seg: &str, v: &Object, path: &mut String| {
        let keep = path.len();
        if !path.is_empty() {
            path.push('.');
        }
        path.push_str(seg);
        collect_names(v, path, out);
        path.truncate(keep);
    };
    match o {
        Object::Tensor(i) => out.push((path.clone(), *i)),
        Object::Dict(items) => {
            for (k, v) in items {
                let seg = match k {
                    Object::Str(s) => s.clone(),
                    Object::Int(n) => n.to_string(),
                    other => format!("{other:?}"),
                };
                child(&seg, v, path);
            }
        }
        Object::List(xs) | Object::Tuple(xs) => {
            for (i, v) in xs.iter().enumerate() {
                child(&i.to_string(), v, path);
            }
        }
        _ => {}
    }
}

type Ranges = Vec<std::ops::Range<usize>>;
type Crcs = Vec<(std::ops::Range<usize>, u32)>;

fn parse_zip(b: &[u8], heap: &mut Heap) -> Result<(V, Ranges, Crcs)> {
    let entries = zip_entries(b)?;
    let pkl = entries
        .iter()
        .filter(|e| e.name.ends_with("data.pkl"))
        .min_by_key(|e| e.name.len())
        .ok_or_else(|| TorchCkptError::Malformed("zip has no data.pkl".into()))?;
    let prefix = &pkl.name[..pkl.name.len() - "data.pkl".len()];
    let find = |rel: &str| {
        entries.iter().find(|e| {
            e.name.len() == prefix.len() + rel.len()
                && e.name.starts_with(prefix)
                && e.name.ends_with(rel)
        })
    };
    if let Some(bo) = find("byteorder") {
        let order = &b[bo.range.clone()];
        if order != b"little" {
            return Err(TorchCkptError::Unsupported(format!(
                "byteorder {:?} (only little-endian checkpoints are read)",
                String::from_utf8_lossy(order)
            )));
        }
    }
    let pkl_bytes = &b[pkl.range.clone()];
    if crc32(pkl_bytes) != pkl.crc32 {
        return bad("data.pkl CRC mismatch (corrupt file)");
    }
    let mut up = Unpickler::new(pkl_bytes, 0, heap, true);
    let root = up.run()?;
    let mut ranges = Vec::new();
    let mut crcs = Vec::new();
    for s in &heap.storages {
        let rel = format!("data/{}", s.key);
        let e = find(&rel).ok_or_else(|| {
            TorchCkptError::Malformed(format!("storage entry {prefix}{rel} missing"))
        })?;
        let need = s
            .numel
            .checked_mul(s.dtype.size())
            .ok_or_else(|| TorchCkptError::Malformed("storage size overflows".into()))?;
        if e.range.len() < need {
            return bad(format!(
                "storage {rel} holds {} bytes, needs {need}",
                e.range.len()
            ));
        }
        ranges.push(e.range.start..e.range.start + need);
        crcs.push((e.range.clone(), e.crc32));
    }
    Ok((root, ranges, crcs))
}

fn parse_legacy(b: &[u8], heap: &mut Heap) -> Result<(V, Ranges)> {
    let mut pos = 0;
    let next = |heap: &mut Heap, persid: bool, pos: &mut usize| -> Result<V> {
        let mut up = Unpickler::new(b, *pos, heap, persid);
        let v = up.run()?;
        *pos = up.pos;
        Ok(v)
    };
    let magic = next(heap, false, &mut pos)?;
    let magic_ok = match &magic {
        V::BigInt(bytes) if bytes.len() <= 16 => {
            let mut buf = [0u8; 16];
            buf[..bytes.len()].copy_from_slice(bytes);
            u128::from_le_bytes(buf) == LEGACY_MAGIC
        }
        _ => false,
    };
    if !magic_ok {
        return bad("pickle file without the torch legacy magic number (not a torch.save file)");
    }
    match next(heap, false, &mut pos)? {
        V::Int(LEGACY_PROTOCOL) => {}
        other => {
            return Err(TorchCkptError::Unsupported(format!(
                "legacy protocol version {other:?}"
            )))
        }
    }
    let sys_info = next(heap, false, &mut pos)?;
    let sys_info = to_object(heap, &sys_info, 0, &mut 4096)?;
    if let Some(Object::Bool(false)) = sys_info.get("little_endian") {
        return Err(TorchCkptError::Unsupported(
            "big-endian legacy checkpoint".into(),
        ));
    }
    let root = next(heap, true, &mut pos)?;
    let keys = next(heap, false, &mut pos)?;
    let keys: Vec<String> = match &keys {
        V::Node(i) => match &heap.nodes[*i] {
            Node::List(xs) => xs
                .iter()
                .map(|x| match x {
                    V::Str(s) => Ok(s.to_string()),
                    _ => bad("legacy storage key is not a str"),
                })
                .collect::<Result<_>>()?,
            _ => return bad("legacy storage key list is not a list"),
        },
        _ => return bad("legacy storage key list is not a list"),
    };
    let mut ranges: Vec<Option<std::ops::Range<usize>>> = vec![None; heap.storages.len()];
    for key in keys {
        let Some(&i) = heap.storage_by_key.get(&key) else {
            return bad(format!(
                "legacy storage {key} is not referenced by the object"
            ));
        };
        let s = &heap.storages[i];
        let numel = rd_u64(b, pos)
            .map_err(|_| TorchCkptError::Malformed("legacy storage header truncated".into()))?;
        pos += 8;
        if numel != s.numel as u64 {
            return bad(format!(
                "legacy storage {key}: header says {numel} elements, pickle says {}",
                s.numel
            ));
        }
        let need = s
            .numel
            .checked_mul(s.dtype.size())
            .ok_or_else(|| TorchCkptError::Malformed("storage size overflows".into()))?;
        let end = pos
            .checked_add(need)
            .filter(|&e| e <= b.len())
            .ok_or_else(|| TorchCkptError::Malformed(format!("legacy storage {key} truncated")))?;
        ranges[i] = Some(pos..end);
        pos = end;
    }
    let ranges = ranges
        .into_iter()
        .zip(&heap.storages)
        .map(|(r, s)| {
            r.ok_or_else(|| TorchCkptError::Malformed(format!("legacy storage {} missing", s.key)))
        })
        .collect::<Result<_>>()?;
    Ok((root, ranges))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(bytes: &[u8]) -> Result<Object> {
        let mut heap = Heap::default();
        let v = Unpickler::new(bytes, 0, &mut heap, true).run()?;
        to_object(&heap, &v, 0, &mut 1_000_000)
    }

    #[test]
    fn crc32_known_value() {
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
    }

    #[test]
    fn plain_values() {
        // {'a': [1, -2, 3.5, None, True], 'b': (b'\x01',)}
        let p = b"\x80\x02}q\x00(X\x01\x00\x00\x00a]q\x01(K\x01J\xfe\xff\xff\xffG@\x0c\x00\x00\x00\x00\x00\x00N\x88eX\x01\x00\x00\x00bC\x01\x01\x85u.";
        let o = run(p).unwrap();
        assert_eq!(
            o.get("a"),
            Some(&Object::List(vec![
                Object::Int(1),
                Object::Int(-2),
                Object::Float(3.5),
                Object::None,
                Object::Bool(true)
            ]))
        );
        assert_eq!(
            o.get("b"),
            Some(&Object::Tuple(vec![Object::Bytes(vec![1])]))
        );
    }

    #[test]
    fn disallowed_globals_are_named() {
        for (p, name) in [
            (
                &b"\x80\x02cos\nsystem\nX\x02\x00\x00\x00ls\x85R."[..],
                "os.system",
            ),
            (&b"cbuiltins\neval\n."[..], "builtins.eval"),
            (&b"cposix\nsystem\n."[..], "posix.system"),
            (&b"ctorch\nload\n."[..], "torch.load"),
            (
                &b"ctorch._utils\n_rebuild_tensor\n."[..],
                "torch._utils._rebuild_tensor",
            ),
            // STACK_GLOBAL, protocol 4 style
            (
                &b"\x80\x04\x8c\x08builtins\x8c\x04exec\x93."[..],
                "builtins.exec",
            ),
        ] {
            match run(p) {
                Err(TorchCkptError::DisallowedGlobal(g)) => assert_eq!(g, name),
                other => panic!("{name}: expected DisallowedGlobal, got {other:?}"),
            }
        }
    }

    #[test]
    fn deep_nesting_errors_without_overflow() {
        // 200k nested TUPLE1s: stack depth in Rust stays flat; conversion refuses the depth.
        let mut p = vec![0x80, 2, b'N'];
        p.extend(std::iter::repeat_n(0x85u8, 200_000));
        p.push(b'.');
        assert!(matches!(run(&p), Err(TorchCkptError::Malformed(_))));
    }

    #[test]
    fn memo_fanout_hits_budget() {
        // x = []; then 60 times: x = (x, x) — 2^60 nodes if expanded naively.
        let mut p = vec![0x80, 2, b']'];
        for _ in 0..60 {
            p.extend_from_slice(&[b'q', 0, b'h', 0, 0x86]);
        }
        p.push(b'.');
        let mut heap = Heap::default();
        let v = Unpickler::new(&p, 0, &mut heap, true).run().unwrap();
        assert!(to_object(&heap, &v, 0, &mut 10_000).is_err());
    }

    #[test]
    fn reference_cycle_errors() {
        // l = []; l.append(l)
        let p = b"\x80\x02]q\x00h\x00a.";
        assert!(run(p).is_err());
    }
}
