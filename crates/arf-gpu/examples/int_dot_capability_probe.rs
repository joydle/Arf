//! EMPIRICAL Metal integer-dot capability probe for T1a (MoE q8_1 x int4 int-dot).
//! Compiles a battery of candidate integer-dot MSL idioms on THIS device (M4 Max / Metal3 /
//! macOS26) via newLibraryWithSource + newComputePipelineState — the SAME path the island uses.
//! Reports, per idiom, whether the MSL front-end compiled AND whether a PSO built (back-end).
//!
//! Run: cargo run --release -p arf-gpu --example int_dot_capability_probe

#[cfg(target_os = "macos")]
fn main() {
    use arf_gpu::gpu::concurrent_metal::MetalIsland;
    use arf_gpu::gpu::GpuContext;
    use std::sync::Arc;

    let ctx = Arc::new(GpuContext::new().expect("gpu"));
    eprintln!("device: {}\n", ctx.info());
    let island = MetalIsland::new(&ctx).expect("metal island");

    let cases: &[(&str, &str, &str)] = &[
        (
            "dot(char4,char4)->int",
            r#"
#include <metal_stdlib>
using namespace metal;
kernel void k(device const char4* a[[buffer(0)]], device const char4* b[[buffer(1)]],
              device int* o[[buffer(2)]], uint i[[thread_position_in_grid]]) {
    o[i] = dot(a[i], b[i]);
}"#,
            "k",
        ),
        (
            "explicit char4 int MAC",
            r#"
#include <metal_stdlib>
using namespace metal;
kernel void k(device const char4* a[[buffer(0)]], device const char4* b[[buffer(1)]],
              device int* o[[buffer(2)]], uint i[[thread_position_in_grid]]) {
    char4 x=a[i], y=b[i];
    o[i] = int(x.x)*int(y.x)+int(x.y)*int(y.y)+int(x.z)*int(y.z)+int(x.w)*int(y.w);
}"#,
            "k",
        ),
        (
            "simdgroup_matrix<char,8,8>",
            r#"
#include <metal_stdlib>
using namespace metal;
kernel void k(device int* o[[buffer(1)]], uint i[[thread_position_in_grid]]) {
    simdgroup_matrix<char,8,8> m = make_filled_simdgroup_matrix<char,8,8>(char(0));
    (void)m; o[i]=0;
}"#,
            "k",
        ),
        (
            "simdgroup_matrix<int,8,8>",
            r#"
#include <metal_stdlib>
using namespace metal;
kernel void k(device int* o[[buffer(1)]], uint i[[thread_position_in_grid]]) {
    simdgroup_matrix<int,8,8> m = make_filled_simdgroup_matrix<int,8,8>(0);
    (void)m; o[i]=0;
}"#,
            "k",
        ),
        (
            "simdgroup int8*int8->int32 MMA",
            r#"
#include <metal_stdlib>
using namespace metal;
kernel void k(threadgroup char* sa[[threadgroup(0)]],
              device int* o[[buffer(1)]], uint i[[thread_position_in_grid]]) {
    simdgroup_matrix<char,8,8> ma = make_filled_simdgroup_matrix<char,8,8>(char(1));
    simdgroup_matrix<char,8,8> mb = make_filled_simdgroup_matrix<char,8,8>(char(1));
    simdgroup_matrix<int,8,8>  mc = make_filled_simdgroup_matrix<int,8,8>(0);
    simdgroup_multiply_accumulate(mc, ma, mb, mc);
    simdgroup_store(mc, (threadgroup int*)sa, 8);
    o[i]=0;
}"#,
            "k",
        ),
        (
            "simd int reduce (per-lane)",
            r#"
#include <metal_stdlib>
using namespace metal;
kernel void k(device const char* qa[[buffer(0)]], device const uchar* qn[[buffer(1)]],
              device int* o[[buffer(2)]], uint tg[[threadgroup_position_in_grid]],
              ushort lane[[thread_index_in_simdgroup]]) {
    int acc = 0;
    for (uint j = lane; j < 256u; j += 32u) { acc += int(qn[j]) * int(qa[j]); }
    acc = simd_sum(acc);
    if (lane==0) o[tg]=acc;
}"#,
            "k",
        ),
        (
            "dot(uchar4,char4) mixed",
            r#"
#include <metal_stdlib>
using namespace metal;
kernel void k(device const uchar4* n[[buffer(0)]], device const char4* a[[buffer(1)]],
              device int* o[[buffer(2)]], uint i[[thread_position_in_grid]]) {
    uchar4 u=n[i]; char4 s=a[i];
    o[i]=int(u.x)*int(s.x)+int(u.y)*int(s.y)+int(u.z)*int(s.z)+int(u.w)*int(s.w);
}"#,
            "k",
        ),
    ];

    let res = island.probe_msl_compile(&ctx, cases).expect("probe");
    for (name, lib_ok, pso_ok, err) in &res {
        let status = if *pso_ok {
            "PSO OK "
        } else if *lib_ok {
            "PSO ERR"
        } else {
            "LIB ERR"
        };
        let e = if err.is_empty() {
            String::new()
        } else {
            let s = err.replace('\n', " ");
            format!("  :: {}", &s[..s.len().min(400)])
        };
        eprintln!("[{status}] {name}{e}");
    }
    eprintln!("\ndone");
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("macOS/Metal-only");
}
