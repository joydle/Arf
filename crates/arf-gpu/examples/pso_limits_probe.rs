//! COMPILE-ONLY probe: prints each attention kernel's PSO launch limits. NO dispatches — safe
//! to run even while chasing a GPU-wedge (a threadsPerThreadgroup > maxTotalThreadsPerThreadgroup
//! dispatch is undefined behavior and a known silent-hang; this checks that legality statically).

#[cfg(target_os = "macos")]
fn main() {
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLCompileOptions, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
        MTLLibrary,
    };

    let dev = MTLCreateSystemDefaultDevice().expect("no Metal device");
    let src = NSString::from_str(include_str!("../src/shaders/metal/attention_msl.metal"));
    let opts = MTLCompileOptions::new();
    let lib = dev
        .newLibraryWithSource_options_error(&src, Some(&opts))
        .expect("MSL compile");
    for entry in [
        "attention_decode",
        "attention_decode_b",
        "kv_scatter_b",
        "attention_verify_shared_prefix",
    ] {
        let f = lib
            .newFunctionWithName(&NSString::from_str(entry))
            .unwrap_or_else(|| panic!("entry {entry} not found"));
        let pso = dev
            .newComputePipelineStateWithFunction_error(&f)
            .unwrap_or_else(|e| panic!("PSO {entry}: {e:?}"));
        println!(
            "{entry}: maxTotalThreadsPerThreadgroup={} threadExecutionWidth={} staticThreadgroupMemoryLength={}B",
            pso.maxTotalThreadsPerThreadgroup(),
            pso.threadExecutionWidth(),
            pso.staticThreadgroupMemoryLength()
        );
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("macOS only");
}
