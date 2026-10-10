//! M2 GATE: standalone parity self-test of the `ssm_conv1d` MSL kernel (the Qwen3.6-27B
//! gated-delta-net causal depthwise conv1d). NO MODEL — synthetic, deterministic buffers,
//! runs in milliseconds. Compares the GPU kernel against a CPU oracle of the EXACT ggml
//! conv math (the d_conv-tap dot + silu + ring advance), and verifies the conv_state ring
//! was advanced correctly so the next-token window is right.
//!
//! Run: cargo run --release -p arf-gpu --example ssm_conv1d_probe

#[cfg(target_os = "macos")]
fn main() {
    use arf_gpu::gpu::concurrent_metal::MetalIsland;
    use arf_gpu::gpu::GpuContext;

    // The beast's real geometry (M1): conv_dim = 10240 channels, d_conv = 4 taps.
    let conv_dim: usize = 10240;
    let d_conv: usize = 4;
    let cs = d_conv - 1; // cached cols per channel (3)

    // ---- SYNTHETIC, deterministic inputs (small ramps so f32 stays exact-ish) ----
    // conv_in[i1] = the new token's value for channel i1.
    let conv_in: Vec<f32> = (0..conv_dim)
        .map(|i| ((i % 17) as f32) * 0.013 - 0.1)
        .collect();
    // conv_kernel[i0 + i1*d_conv] = tap i0 of channel i1.
    let conv_kernel: Vec<f32> = (0..d_conv * conv_dim)
        .map(|j| {
            let i0 = j % d_conv;
            let i1 = j / d_conv;
            // distinct per-(tap,channel); kept small.
            ((i0 as f32) * 0.07 + ((i1 % 23) as f32) * 0.0031 - 0.05) * 0.5
        })
        .collect();
    // conv_state[i1*cs + c] = the (oldest..newest) prior values for channel i1.
    let conv_state: Vec<f32> = (0..conv_dim * cs)
        .map(|j| {
            let c = j % cs;
            let i1 = j / cs;
            ((c as f32) * 0.021 + ((i1 % 11) as f32) * 0.0042 - 0.03) * 0.5
        })
        .collect();

    // ---- CPU ORACLE: the EXACT ggml_ssm_conv decode math + silu + ring advance ----
    let silu = |x: f32| -> f32 {
        // EXACT match to silu.wgsl / the kernel: x / (1 + exp(-clamp(x,-30,30))).
        let z = x.clamp(-30.0, 30.0);
        x / (1.0 + (-z).exp())
    };
    let mut cpu_out = vec![0.0f32; conv_dim];
    let mut cpu_new_state = vec![0.0f32; conv_dim * cs];
    for i1 in 0..conv_dim {
        // window = [state[0], state[1], state[2], conv_in[i1]]  (oldest..newest, new last)
        let mut window = [0.0f32; 8];
        for c in 0..cs {
            window[c] = conv_state[i1 * cs + c];
        }
        window[cs] = conv_in[i1];
        // d_conv-tap dot with kernel[i0 + i1*d_conv].
        let mut sumf = 0.0f32;
        for i0 in 0..d_conv {
            sumf += window[i0] * conv_kernel[i0 + i1 * d_conv];
        }
        cpu_out[i1] = silu(sumf);
        // advance ring: drop oldest, append new input → window[1..d_conv].
        for c in 0..cs {
            cpu_new_state[i1 * cs + c] = window[c + 1];
        }
    }

    // ---- GPU: compile + dispatch the ssm_conv1d MSL kernel (no model) ----
    let ctx = GpuContext::new().expect("gpu context");
    let mut island = MetalIsland::new(&ctx).expect("metal island (need a Metal backend)");
    let (gpu_out, gpu_new_state) = island
        .ssm_conv1d_selftest(&ctx, &conv_in, &conv_kernel, &conv_state, conv_dim, d_conv)
        .expect("ssm_conv1d dispatch");

    // ---- PARITY: conv_out (post-silu) ----
    let mut max_abs_out = 0.0f32;
    let mut argmax_out = 0usize;
    for i in 0..conv_dim {
        let a = (gpu_out[i] - cpu_out[i]).abs();
        if a > max_abs_out {
            max_abs_out = a;
            argmax_out = i;
        }
    }

    // ---- STATE-SHIFT check: the advanced ring must match the CPU's shifted+appended ring ----
    let mut max_abs_state = 0.0f32;
    let mut argmax_state = 0usize;
    for i in 0..conv_state.len() {
        let a = (gpu_new_state[i] - cpu_new_state[i]).abs();
        if a > max_abs_state {
            max_abs_state = a;
            argmax_state = i;
        }
    }
    // Sanity: the new ring's LAST col per channel must equal that channel's new input
    // (the appended value), and the first cs-1 cols must equal the old ring shifted left.
    let mut shift_struct_ok = true;
    for i1 in 0..conv_dim {
        // last col == conv_in[i1]
        if (gpu_new_state[i1 * cs + (cs - 1)] - conv_in[i1]).abs() > 1e-6 {
            shift_struct_ok = false;
            break;
        }
        // cols [0..cs-1] == old state cols [1..cs]
        for c in 0..(cs - 1) {
            if (gpu_new_state[i1 * cs + c] - conv_state[i1 * cs + c + 1]).abs() > 1e-6 {
                shift_struct_ok = false;
                break;
            }
        }
        if !shift_struct_ok {
            break;
        }
    }

    println!("=== ssm_conv1d M2 PARITY (synthetic, no model) ===");
    println!("  geometry: conv_dim = {conv_dim}, d_conv = {d_conv} (cached cols = {cs})");
    println!(
        "  conv_out (post-silu) max_abs = {max_abs_out:.3e}  @channel {argmax_out}  (gpu {:.6} vs cpu {:.6})",
        gpu_out[argmax_out], cpu_out[argmax_out]
    );
    println!(
        "  conv_state (advanced ring) max_abs = {max_abs_state:.3e}  @idx {argmax_state}  (gpu {:.6} vs cpu {:.6})",
        gpu_new_state[argmax_state], cpu_new_state[argmax_state]
    );
    println!(
        "  state-shift structure (drop-oldest + append-new) = {}",
        if shift_struct_ok { "OK" } else { "WRONG" }
    );

    let tol = 1e-5f32;
    assert!(
        max_abs_out < tol,
        "conv_out parity {max_abs_out:.3e} >= {tol:.0e}"
    );
    assert!(
        max_abs_state < tol,
        "conv_state parity {max_abs_state:.3e} >= {tol:.0e}"
    );
    assert!(
        shift_struct_ok,
        "conv_state ring was not advanced correctly (shift structure WRONG)"
    );

    println!("\nM2 GATE PASS: ssm_conv1d GPU == CPU oracle (out + advanced ring), tol {tol:.0e}.");
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("ssm_conv1d_probe: the MSL ssm_conv1d kernel is macOS-only (Metal island). Skipped.");
}
