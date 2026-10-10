//! Ground-truth probe: how is the FLUX DiT GGUF actually quantized, tensor by tensor?
//! Tells F1 which weights can take the native Q4_K transcode (get_q4k_blocks_raw → Some)
//! vs which fall back to the f32 dequant path (norms/biases/non-Q4K). No model inference.
//!
//! Run: cargo run --release -p arf-core --example flux_dit_quant_probe -- models/flux-schnell/flux1-schnell-Q4_K_S.gguf

use arf_core::model::gguf::LazyGguf;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/flux-schnell/flux1-schnell-Q4_K_S.gguf".into());
    let g = LazyGguf::open_raw(std::path::Path::new(&path)).expect("open gguf");

    let names: Vec<String> = g.tensor_names().map(|s| s.to_string()).collect();
    let total = names.len();
    let mut q4k = 0usize;
    let mut q4k_bytes = 0u64;
    let mut other = 0usize;
    let mut other_examples: Vec<String> = Vec::new();
    let mut weight_tensors = 0usize;
    let mut weight_q4k = 0usize;

    for n in &names {
        let is_weight = n.ends_with(".weight");
        if is_weight {
            weight_tensors += 1;
        }
        match g.get_q4k_blocks_raw(n) {
            Some((blocks, rows, cols)) => {
                q4k += 1;
                q4k_bytes += blocks.len() as u64;
                if is_weight {
                    weight_q4k += 1;
                }
                if q4k <= 3 {
                    println!(
                        "  Q4K  {n}  [{rows} x {cols}]  {} blocks-bytes",
                        blocks.len()
                    );
                }
            }
            None => {
                other += 1;
                if other_examples.len() < 8 {
                    other_examples.push(n.clone());
                }
            }
        }
    }

    println!("\n=== FLUX DiT GGUF quant breakdown ({path}) ===");
    println!("total tensors:        {total}");
    println!(
        "native Q4_K (transcodable): {q4k}   ({:.2} GB of Q4K blocks)",
        q4k_bytes as f64 / 1e9
    );
    println!("non-Q4K (f32 fallback):     {other}");
    println!(
        ".weight tensors:      {weight_tensors}  ({weight_q4k} are Q4_K → the big GEMM weights)"
    );
    println!("\nnon-Q4K examples (norms/biases — expected to stay f32):");
    for e in &other_examples {
        println!("  {e}");
    }
    println!(
        "\nIf the big .weight GEMM tensors are Q4_K, F1 keeps them quantized-resident (~6.5 GB)"
    );
    println!("instead of expanding to bf16 (~24 GB). Norms/biases stay f32 (tiny).");
}
