// Tiled matmul: C[m, n] = A[m, k] · Bᵀ  (B stored [n, k], i.e. Linearᵀ).
//
// Each 16×16 workgroup computes a 16×16 tile of C, streaming K in 16-wide tiles
// through workgroup-shared memory so each input element is read from global
// memory once per tile rather than once per output element. This is the
// resident-weight Linear kernel: A is the activation, B the resident weight.

struct Dims { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<u32>;   // bf16-packed: 2 weights per word
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;

// bf16 -> f32 (bit-exact: bf16 is the high 16 bits of the f32). Weight (col, kk)
// of the row-major [n, k] matrix is packed two-per-word: word col*(k/2)+kk/2,
// low half when kk is even.
fn weight_at(col: u32, kk: u32, k: u32) -> f32 {
    let word = b[col * (k / 2u) + kk / 2u];
    let half = select(word >> 16u, word & 0xffffu, (kk & 1u) == 0u);
    return bitcast<f32>(half << 16u);
}

const TILE: u32 = 16u;
var<workgroup> as_tile: array<array<f32, 16>, 16>; // [rowInTile][kInTile]
var<workgroup> bs_tile: array<array<f32, 16>, 16>; // [colInTile][kInTile]

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let lx = lid.x; // row within tile
    let ly = lid.y; // col within tile
    let row = wid.x * TILE + lx;
    let col = wid.y * TILE + ly;

    var acc = 0.0;
    let num_tiles = (d.k + TILE - 1u) / TILE;
    for (var t = 0u; t < num_tiles; t = t + 1u) {
        let k_a = t * TILE + ly;
        if (row < d.m && k_a < d.k) {
            as_tile[lx][ly] = a[row * d.k + k_a];
        } else {
            as_tile[lx][ly] = 0.0;
        }
        let k_b = t * TILE + lx;
        if (col < d.n && k_b < d.k) {
            bs_tile[ly][lx] = weight_at(col, k_b, d.k);
        } else {
            bs_tile[ly][lx] = 0.0;
        }
        workgroupBarrier();

        for (var p = 0u; p < TILE; p = p + 1u) {
            acc = acc + as_tile[lx][p] * bs_tile[ly][p];
        }
        workgroupBarrier();
    }

    if (row < d.m && col < d.n) {
        c[row * d.n + col] = acc;
    }
}
