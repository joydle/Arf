//! Dump one GGUF tensor as raw little-endian f32 (dequantised) — for cross-engine weight
//! comparisons (2026-09-22: another engine's target vs ours, same tensor).
//! usage: dump_tensor_f32 <model.gguf> <tensor name> <out.f32>
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (path, name, out) = (&a[0], &a[1], &a[2]);
    let g =
        arf_core::model::gguf::LazyGguf::open_raw(std::path::Path::new(path)).expect("open_raw");
    let f = g.get_f32_raw(name).expect("tensor");
    let bytes: Vec<u8> = f.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(out, &bytes).expect("write");
    eprintln!("{name}: {} floats -> {out}", f.len());
}
