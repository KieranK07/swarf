//! World dimensions, cell packing, and terrain generation.

use crate::materials::Material;

/// Which world to build.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Scene {
    /// The real thing: terrain, caves, ore, flooded caverns.
    Terrain,
    /// A bare rig for judging the physics. Terrain noise makes it impossible to
    /// tell a rule bug from a lumpy cave, so rules get tuned against flat
    /// floors, a step, and a sealed vessel with known-good answers: a heap
    /// should hold a stable angle, and a vessel should end up level.
    Lab,
}

/// World size in cells.
///
/// At 4 bytes a cell this is 32 MB -- large enough that the whole thing does
/// not fit on screen at 1:1, small enough to stay comfortably inside the
/// default 128 MB storage-buffer binding limit.
pub const WIDTH: u32 = 4096;
pub const HEIGHT: u32 = 2048;

/// Simulation chunk edge, in cells. One compute workgroup covers one chunk:
/// 32x32 cells == 16x16 Margolus blocks == 256 threads.
pub const CHUNK: u32 = 32;

pub const CELL_COUNT: usize = (WIDTH * HEIGHT) as usize;

/// Ambient temperature, Kelvin. Roughly 20 C.
pub const AMBIENT_K: u16 = 293;

#[inline]
pub fn pack(material: u8, variant: u8, temperature: u16) -> u32 {
    material as u32 | ((variant as u32) << 8) | ((temperature as u32) << 16)
}

// --- value noise -----------------------------------------------------------

fn hash_f(x: i32, y: i32, seed: u32) -> f32 {
    let mut h = (x as u32)
        .wrapping_mul(73_856_093)
        ^ (y as u32).wrapping_mul(19_349_663)
        ^ seed.wrapping_mul(83_492_791);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^= h >> 16;
    (h & 0x00FF_FFFF) as f32 / 16_777_215.0
}

fn value_noise(x: f32, y: f32, seed: u32) -> f32 {
    let (xi, yi) = (x.floor(), y.floor());
    let (xf, yf) = (x - xi, y - yi);
    // Smoothstep the interpolant, otherwise the lattice shows as visible creases.
    let u = xf * xf * (3.0 - 2.0 * xf);
    let v = yf * yf * (3.0 - 2.0 * yf);
    let (x0, y0) = (xi as i32, yi as i32);
    let a = hash_f(x0, y0, seed);
    let b = hash_f(x0 + 1, y0, seed);
    let c = hash_f(x0, y0 + 1, seed);
    let d = hash_f(x0 + 1, y0 + 1, seed);
    (a * (1.0 - u) + b * u) * (1.0 - v) + (c * (1.0 - u) + d * u) * v
}

/// Fractal noise in 0..1.
fn fbm(x: f32, y: f32, octaves: u32, seed: u32) -> f32 {
    let mut sum = 0.0;
    let mut amp = 0.5;
    let mut freq = 1.0;
    let mut norm = 0.0;
    for i in 0..octaves {
        sum += amp * value_noise(x * freq, y * freq, seed.wrapping_add(i * 17));
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    sum / norm
}

// --- generation ------------------------------------------------------------

const SEA_LEVEL: f32 = 640.0;
const BEDROCK_MARGIN: i32 = 3;

/// Build a fresh world. Rows are generated in parallel bands -- the work is
/// perfectly independent, and it keeps a 8.4M-cell regenerate under a blink.
pub fn generate(seed: u32, scene: Scene) -> Vec<u32> {
    let mut cells = vec![0u32; CELL_COUNT];

    const BAND: usize = 64;
    let row_bytes = WIDTH as usize;

    std::thread::scope(|scope| {
        for (band_index, band) in cells.chunks_mut(row_bytes * BAND).enumerate() {
            scope.spawn(move || {
                let y0 = band_index * BAND;
                for (local_y, row) in band.chunks_mut(row_bytes).enumerate() {
                    let y = (y0 + local_y) as u32;
                    match scene {
                        Scene::Terrain => fill_row(row, y, seed),
                        Scene::Lab => fill_row_lab(row, y, seed),
                    }
                }
            });
        }
    });

    cells
}

/// Floor height for [`Scene::Lab`].
pub const LAB_FLOOR: u32 = 1200;

fn fill_row_lab(row: &mut [u32], y: u32, seed: u32) {
    let yi = y as i32;
    let floor = LAB_FLOOR as i32;

    // Sealed vessel: two walls and a base, open at the top.
    const VESSEL_LEFT: i32 = 900;
    const VESSEL_RIGHT: i32 = 1700;
    const WALL: i32 = 8;
    let vessel_top = floor - 400;

    for (x, cell) in row.iter_mut().enumerate() {
        let xi = x as i32;
        let variant = (hash_f(xi, yi, seed ^ 0xABCD) * 255.0) as u8;

        let border = xi < BEDROCK_MARGIN
            || xi >= WIDTH as i32 - BEDROCK_MARGIN
            || yi < BEDROCK_MARGIN
            || yi >= HEIGHT as i32 - BEDROCK_MARGIN;

        // A step, so heaps have an edge to slump over.
        let step = (2600..3200).contains(&xi) && yi >= floor - 120;

        let in_vessel_span = yi >= vessel_top && yi < floor;
        let on_vessel_wall = (VESSEL_LEFT - WALL..=VESSEL_LEFT).contains(&xi)
            || (VESSEL_RIGHT..=VESSEL_RIGHT + WALL).contains(&xi);
        let on_vessel_base =
            (VESSEL_LEFT..=VESSEL_RIGHT).contains(&xi) && yi >= floor - WALL;
        let vessel = in_vessel_span && (on_vessel_wall || on_vessel_base);

        let material = if border {
            Material::BEDROCK
        } else if yi >= floor || step || vessel {
            2 // stone
        } else {
            Material::AIR
        };

        *cell = pack(material, variant, AMBIENT_K);
    }
}

fn fill_row(row: &mut [u32], y: u32, seed: u32) {
    let yf = y as f32;

    for (x, cell) in row.iter_mut().enumerate() {
        let xf = x as f32;
        let variant = (hash_f(x as i32, y as i32, seed ^ 0xABCD) * 255.0) as u8;

        // Indestructible shell, so nothing can ever leave the simulation.
        let border = (x as i32) < BEDROCK_MARGIN
            || (x as i32) >= WIDTH as i32 - BEDROCK_MARGIN
            || (y as i32) < BEDROCK_MARGIN
            || (y as i32) >= HEIGHT as i32 - BEDROCK_MARGIN;
        if border {
            *cell = pack(Material::BEDROCK, variant, AMBIENT_K);
            continue;
        }

        // Rolling terrain: one low-frequency ridge plus finer detail on top.
        let surface = 380.0 + fbm(xf / 420.0, 0.0, 5, seed) * 420.0
            + (fbm(xf / 70.0, 0.0, 3, seed ^ 0x55) - 0.5) * 26.0;

        let depth = yf - surface;

        let mut material = if depth < 0.0 {
            // Above ground: air, or sea water where the land dips below sea level.
            if yf >= SEA_LEVEL { 8 } else { Material::AIR }
        } else {
            // Soil grades into rock.
            let soil = 7.0 + fbm(xf / 130.0, 0.0, 3, seed ^ 0x77) * 20.0;
            if depth < soil {
                // Beaches and deserts: sand instead of dirt where the surface
                // noise says so, and always at the waterline.
                let sandy = fbm(xf / 260.0, 11.0, 3, seed ^ 0x99) > 0.58
                    || (surface - SEA_LEVEL).abs() < 26.0;
                if sandy { 4 } else { 3 }
            } else {
                2
            }
        };

        // Caves. Carved only below the soil so the surface stays intact, and
        // flooded below sea level -- which gives the underground water to work
        // with without hand-placing a single pool.
        if depth > 14.0 && yf < HEIGHT as f32 - 90.0 {
            let cave = fbm(xf / 95.0, yf / 95.0, 4, seed ^ 0x1234);
            if cave > 0.615 {
                material = if yf > SEA_LEVEL + 220.0 { 8 } else { Material::AIR };
            }
        }

        // Ore veins, only in rock. Coal is shallow and common, iron deeper and
        // scarcer -- so the early factory sits near the surface and has to dig
        // for the second resource.
        if material == 2 {
            if fbm(xf / 44.0, yf / 30.0, 3, seed ^ 0x2222) > 0.70 && depth > 40.0 {
                material = 6;
            } else if fbm(xf / 36.0, yf / 26.0, 3, seed ^ 0x3333) > 0.725 && depth > 150.0 {
                material = 7;
            } else if fbm(xf / 90.0, yf / 60.0, 3, seed ^ 0x4444) > 0.74 && depth > 90.0 {
                material = 5;
            }
        }

        // Deep lava, well below anything reachable by accident.
        if yf > HEIGHT as f32 - 190.0 && fbm(xf / 120.0, yf / 70.0, 3, seed ^ 0x5555) > 0.52 {
            material = 13;
        }

        let temperature = if material == 13 { 1500 } else { AMBIENT_K };
        *cell = pack(material, variant, temperature);
    }
}
