// Cooperative-matrix GEMM for Q4_0 (block-32 4-bit), M register-blocked RB=2: one
// workgroup computes two stacked 8×8 output tiles and unpacks+dequantizes the shared
// 8×8 weight tile ONCE per K-step, reusing it for both row accumulators. The RB=2
// analog of matmul_coop_q4. Doubly amortizing for quants: the weight read AND the
// nibble-unpack+scale ALU are now shared across 16 output rows instead of 8. Same
// math + boundary contract as matmul_coop_q4 (block scale folded per element);
// matmul_nt_q4 oracle. Caller dispatches for m ≥ 16 (ceil(m/16) row-groups).

enable wgpu_cooperative_matrix;

struct Dims { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       codes: array<u32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;
@group(0) @binding(4) var<storage, read>       scales: array<u32>;

const T: u32 = 8u;
var<workgroup> as0: array<f32, 64>;
var<workgroup> as1: array<f32, 64>;
var<workgroup> bs_t: array<f32, 64>;
var<workgroup> cs_t: array<f32, 64>;

fn bf16_to_f32(bits: u32) -> f32 {
    return bitcast<f32>((bits & 0xFFFFu) << 16u);
}

fn qweight(col: u32, kk: u32, k: u32) -> f32 {
    let blocks = k / 32u;
    let b = kk / 32u;
    let j = kk % 32u;
    let word = codes[(col * blocks + b) * 4u + j / 8u];
    let nib = (word >> ((j % 8u) * 4u)) & 0xFu;
    let val = f32(nib) - select(0.0, 16.0, nib >= 8u);
    return val * bf16_to_f32(scales[col * blocks + b]);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let M = d.m;
    let K = d.k;
    let N = d.n;
    let row0 = wid.x * (T * 2u);
    let col0 = wid.y * T;
    let lr = lid.x;
    let lc = lid.y;

    cs_t[lr * T + lc] = 0.0;
    workgroupBarrier();
    var acc0 = coopLoad<coop_mat8x8<f32, C>>(&cs_t[0], T);
    var acc1 = coopLoad<coop_mat8x8<f32, C>>(&cs_t[0], T);

    let num_tiles = (K + T - 1u) / T;
    for (var t = 0u; t < num_tiles; t = t + 1u) {
        let ka = t * T + lc;
        let r0 = row0 + lr;
        let r1 = row0 + T + lr;
        as0[lr * T + lc] = select(0.0, a[r0 * K + ka], r0 < M && ka < K);
        as1[lr * T + lc] = select(0.0, a[r1 * K + ka], r1 < M && ka < K);
        let kb = t * T + lr;
        bs_t[lr * T + lc] = select(0.0, qweight(col0 + lc, kb, K), col0 + lc < N && kb < K);
        workgroupBarrier();

        let bt = coopLoad<coop_mat8x8<f32, B>>(&bs_t[0], T);
        let at0 = coopLoad<coop_mat8x8<f32, A>>(&as0[0], T);
        let at1 = coopLoad<coop_mat8x8<f32, A>>(&as1[0], T);
        acc0 = coopMultiplyAdd(at0, bt, acc0);
        acc1 = coopMultiplyAdd(at1, bt, acc1);
        workgroupBarrier();
    }

    coopStore(acc0, &cs_t[0], T);
    workgroupBarrier();
    if (row0 + lr < M && col0 + lc < N) {
        c[(row0 + lr) * N + col0 + lc] = cs_t[lr * T + lc];
    }
    workgroupBarrier();
    coopStore(acc1, &cs_t[0], T);
    workgroupBarrier();
    if (row0 + T + lr < M && col0 + lc < N) {
        c[(row0 + T + lr) * N + col0 + lc] = cs_t[lr * T + lc];
    }
}
