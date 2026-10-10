//! L364 — PORTABLE STAND-INS for the Metal island's handle types, compiled ONLY off macOS.
//!
//! WHY THIS EXISTS, and why it is not a second backend. `GpuLayer`, `GpuMatWeight` and friends
//! carry `Option<MtlBuf>` / `Option<Q4ksMtl>` views next to their portable `wgpu::Buffer`s — the
//! island borrows wgpu's device and reads the SAME allocations through raw Metal handles. Those
//! fields are `Option`, and off macOS they are always `None`: nothing constructs them, nothing
//! reads them, because every code path that would is behind `cfg(target_os = "macos")`.
//!
//! The alternative was gating each of the ~88 field declarations and every constructor that
//! names them, which is 88 chances to miss one and a diff nobody can review. This is the
//! contiguous move: four uninhabitable types in one file, and the field declarations stay
//! byte-identical across platforms.
//!
//! These are UNINHABITABLE on purpose. Each wraps `Infallible`, so the compiler proves no value
//! can exist — a `Some(..)` off macOS is not merely unused, it is unconstructible. If a future
//! change tries to build one on Linux it is a compile error at the construction site, which is
//! exactly where the mistake would be, rather than a silent `None` at runtime.

use std::convert::Infallible;

macro_rules! uninhabited {
    ($($(#[$m:meta])* $name:ident),* $(,)?) => {$(
        $(#[$m])*
        #[derive(Debug, Clone)]
        pub struct $name(Infallible);
    )*};
}

uninhabited! {
    /// Off-macOS twin of the raw `MTLBuffer` handle wrapper.
    MtlBuf,
    /// Off-macOS twin of the Q4_K_S raw-Metal weight views (codes/scales/mins/dd/dims).
    Q4ksMtl,
    /// Off-macOS twin of the shared host/device buffer used by the island's rings.
    SharedBuffer,
    /// Off-macOS twin of the all-experts MoE projection views.
    MoeMtl,
}
