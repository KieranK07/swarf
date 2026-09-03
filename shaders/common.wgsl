// ---------------------------------------------------------------------------
// swarf :: common.wgsl
// Shared definitions, prepended to every other shader at load time (WGSL has
// no #include). Anything here must compile inside a compute *and* a render
// module, so keep it to constants, types and pure functions.
// ---------------------------------------------------------------------------

// --- cell packing ----------------------------------------------------------
//
// One cell is exactly one u32. Four bytes per cell is the whole performance
// story: an 4096x2048 world is 33 MB, so a full sim tick touches ~67 MB of
// bandwidth. On M4's ~120 GB/s that is a few percent, and chunk sleeping takes
// it far below even that.
//
//   bits  0..7   material id      (256 materials)
//   bits  8..15  variant          (per-cell colour noise, fixed at spawn)
//   bits 16..31  temperature      (Kelvin, 0..65535)
//
fn cell_mat(c: u32) -> u32 { return c & 0xFFu; }
fn cell_var(c: u32) -> u32 { return (c >> 8u) & 0xFFu; }
fn cell_temp(c: u32) -> u32 { return c >> 16u; }

fn make_cell(mat: u32, variant: u32, temp: u32) -> u32 {
    return (mat & 0xFFu) | ((variant & 0xFFu) << 8u) | ((temp & 0xFFFFu) << 16u);
}

// --- material table --------------------------------------------------------

const KIND_STATIC: u32 = 0u;  // never moves (stone, machine housings)
const KIND_POWDER: u32 = 1u;  // falls, slumps diagonally, holds an angle of repose
const KIND_FLUID:  u32 = 2u;  // falls, slumps, and spreads sideways under pressure

// Density of the ambient medium. Materials lighter than this rise, heavier
// ones sink -- gases need no special case, they fall *upwards* for free.
const AIR_DENSITY: f32 = 10.0;

const MAT_AIR: u32 = 0u;

// Simulation chunk edge in cells. One workgroup covers one chunk:
// 32x32 cells == 16x16 Margolus blocks == 256 threads. Must match
// `world::CHUNK`.
const CHUNK: u32 = 32u;

// Per-chunk wake bits.
//   bit 0  simulate this chunk this tick
//   bit 1  simulate it next tick
// `rotate_wake` shifts bit 1 down into bit 0 between ticks, so a chunk stays
// awake exactly as long as something keeps happening in it.
const WAKE_NOW: u32 = 1u;
const WAKE_NEXT: u32 = 2u;

struct Material {
    colour: vec4<f32>,   // rgb + emissive strength
    density: f32,
    kind: u32,
    variance: f32,       // how much per-cell colour noise to apply
    mobility: f32,       // chance per tick of taking the gravity step
}

// --- hashing ---------------------------------------------------------------
// Cheap integer hash. Falling-sand needs symmetry breaking everywhere: without
// randomness sand piles grow into perfect pyramids and water splits evenly
// down both sides of every obstacle forever.

fn hash_u32(x0: u32) -> u32 {
    var x = x0;
    x ^= x >> 16u; x *= 0x7feb352du;
    x ^= x >> 15u; x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

fn hash2(a: u32, b: u32, c: u32) -> u32 {
    return hash_u32(a * 73856093u ^ b * 19349663u ^ c * 83492791u);
}

fn rand_f32(h: u32) -> f32 {
    return f32(h & 0xFFFFFFu) / 16777215.0;
}
