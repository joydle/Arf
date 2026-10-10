//! Probe: does this adapter advertise cooperative-matrix (Apple simdgroup_matrix /
//! NVIDIA tensor-core) configurations? Landed in wgpu 29.0.1 (Metal output path).
//! If this lists configs, the P8 "WGSL can't reach matrix hardware" ceiling is
//! liftable here. Run: `cargo run -p arf-gpu --example coop_probe`.

fn main() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .expect("no GPU adapter");

    let info = adapter.get_info();
    println!("adapter: {} ({:?})", info.name, info.backend);

    let feats = adapter.features();
    println!(
        "EXPERIMENTAL_COOPERATIVE_MATRIX feature present: {}",
        feats.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
    );

    let props = adapter.cooperative_matrix_properties();
    if props.is_empty() {
        println!("cooperative_matrix_properties: NONE advertised");
    } else {
        println!("cooperative_matrix_properties ({} configs):", props.len());
        for p in props {
            println!(
                "  M{} N{} K{}  AB={:?}  CR={:?}  sat={}",
                p.m_size, p.n_size, p.k_size, p.ab_type, p.cr_type, p.saturating_accumulation
            );
        }
    }
}
