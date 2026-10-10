//! Compare two raw f32 dumps of the same [n, k] matrix: relative RMS difference, correlation,
//! and the same with the second TRANSPOSED (in case the layouts differ).
//! usage: compare_f32 <a.f32> <b.f32> <n> <k>
fn load(p: &str) -> Vec<f32> {
    std::fs::read(p)
        .expect("read")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
fn stats(a: &[f32], b: impl Fn(usize) -> f32, n: usize) -> (f64, f64) {
    let (mut sab, mut saa, mut sbb, mut sdd) = (0f64, 0f64, 0f64, 0f64);
    for i in 0..n {
        let (x, y) = (a[i] as f64, b(i) as f64);
        sab += x * y;
        saa += x * x;
        sbb += y * y;
        sdd += (x - y) * (x - y);
    }
    ((sdd / saa).sqrt(), sab / (saa * sbb).sqrt())
}
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (x, y) = (load(&a[0]), load(&a[1]));
    let (n, k) = (
        a[2].parse::<usize>().unwrap(),
        a[3].parse::<usize>().unwrap(),
    );
    assert_eq!(x.len(), n * k);
    assert_eq!(y.len(), n * k);
    let (rel, corr) = stats(&x, |i| y[i], n * k);
    let (relt, corrt) = stats(
        &x,
        |i| {
            let (r, c) = (i / k, i % k);
            y[c * n + r]
        },
        n * k,
    );
    let ra: f64 = (x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / (n * k) as f64).sqrt();
    let rb: f64 = (y.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / (n * k) as f64).sqrt();
    println!("rms a={ra:.5e} b={rb:.5e} | same layout: rel RMS diff {rel:.4} corr {corr:.5} | b transposed: rel {relt:.4} corr {corrt:.5}");
    println!("first 6: a {:?} b {:?}", &x[..6], &y[..6]);
}
