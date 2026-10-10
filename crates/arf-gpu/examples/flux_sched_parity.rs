//! FLUX P3 loop-parity gate: verify the NEW host-side pipeline pieces — the flow-match Euler
//! SCHEDULE (sigmas/timesteps), the 2×2 latent PACK/UNPACK, and the Euler UPDATE arithmetic —
//! against the diffusers reference (/tmp/flux_oracle/sched_reference.json), isolated from the
//! (already-parity-proven) DiT by using the SAME deterministic fake velocity vel = sin(x)·(1+t/1000).
//!
//! Usage: cargo run --release -p arf-gpu --example flux_sched_parity

use arf_gpu::gpu::flux::{pack, FlowMatchSchedule};
use serde_json::Value;

fn flat(v: &Value) -> Vec<f32> {
    fn rec(v: &Value, out: &mut Vec<f32>) {
        match v {
            Value::Array(a) => a.iter().for_each(|x| rec(x, out)),
            // A reference dump may carry metadata beside its tensor; skip anything non-numeric
            // rather than erroring, and skip a number that will not fit f64 rather than panicking.
            Value::Number(n) => {
                if let Some(f) = n.as_f64() {
                    out.push(f as f32);
                }
            }
            _ => {}
        }
    }
    let mut o = Vec::new();
    rec(v, &mut o);
    o
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut r) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        d += ((*x - *y) as f64).powi(2);
        r += (*y as f64).powi(2);
    }
    (d / r).sqrt()
}

fn main() {
    let r: Value = serde_json::from_reader(
        std::fs::File::open("/tmp/flux_oracle/sched_reference.json").expect("sched_reference.json"),
    )
    .unwrap();
    let (h, w, n) = (
        r["H"].as_u64().unwrap() as usize,
        r["W"].as_u64().unwrap() as usize,
        r["N"].as_u64().unwrap() as usize,
    );
    let seq = r["seq"].as_u64().unwrap() as usize;
    let ref_sigmas = flat(&r["sigmas"]);
    let ref_ts = flat(&r["timesteps"]);
    let init_packed = flat(&r["init_packed"]); // [seq, 64]
    let ref_final_packed = flat(&r["final_packed"]);
    let ref_final_latent = flat(&r["final_latent"]); // [16, h, w]

    let mut ok = true;

    // 1) schedule
    let sched = FlowMatchSchedule::schnell(n, seq);
    let s_err = rel_l2(&sched.sigmas, &ref_sigmas);
    let t_err = rel_l2(&sched.timesteps, &ref_ts);
    println!("schedule:  sigmas rel={s_err:.2e}  timesteps rel={t_err:.2e}");
    ok &= s_err < 1e-5 && t_err < 1e-5;

    // 2) pack/unpack roundtrip on the reference's unpacked final latent
    let repacked = pack::pack_latents(&ref_final_latent, 16, h, w);
    let reunpacked = pack::unpack_latents(&repacked, 16, h, w);
    let rt_err = rel_l2(&reunpacked, &ref_final_latent);
    println!("pack roundtrip: rel={rt_err:.2e}");
    ok &= rt_err < 1e-6;

    // 3) Euler loop with the SAME deterministic fake velocity, from the reference's init_packed
    let mut x = init_packed.clone();
    for i in 0..n {
        let t = sched.timesteps[i];
        let vel: Vec<f32> = x.iter().map(|&xi| xi.sin() * (1.0 + t / 1000.0)).collect();
        sched.step(i, &mut x, &vel);
    }
    let loop_err = rel_l2(&x, &ref_final_packed);
    println!("euler loop: final_packed rel={loop_err:.2e}");
    ok &= loop_err < 1e-4;

    // 4) unpack our loop output → latent, compare to reference unpacked latent
    let our_latent = pack::unpack_latents(&x, 16, h, w);
    let lat_err = rel_l2(&our_latent, &ref_final_latent);
    println!("unpacked latent: rel={lat_err:.2e}");
    ok &= lat_err < 1e-4;

    println!(
        "FLUX P3 loop parity: {}",
        if ok { "GREEN ✓" } else { "RED ✗" }
    );
    if !ok {
        std::process::exit(1);
    }
}
