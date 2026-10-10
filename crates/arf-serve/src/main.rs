//! `arf-serve`: the persistent, resident-model, OpenAI-compatible inference server on the
//! wgpu/Metal backend. The server itself lives in the library ([`arf_serve::server`]) so an
//! embedding binary can run the same server with its own backend; this binary passes
//! the Metal backend.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    arf_serve::server::run(arf_serve::server::metal_backend)
}
