//! Headless physics tests. Each one builds the `lab` scene on the GPU, paints
//! material into it, runs the real shaders and reads the cell buffer back.
//!
//! These need a GPU that wgpu can reach, the same as `--shot`.

use crate::gpu::Gpu;
use crate::materials::MATERIALS;
use crate::sim::{Sim, Stroke};
use crate::world::{self, Scene, LAB_FLOOR};

const STONE: u8 = 2;
const SAND: u8 = 4;
const GRAVEL: u8 = 5;
const WATER: u8 = 8;
const OIL: u8 = 9;
const STEAM: u8 = 10;
const LAVA: u8 = 13;

fn lab() -> (Gpu, Sim) {
    let gpu = Gpu::headless(64, 64);
    let sim = Sim::new(&gpu, 1, Scene::Lab);
    (gpu, sim)
}

fn disc(material: u8, x: f32, y: f32, radius: f32) -> Stroke {
    Stroke { from: [x, y], to: [x, y], radius, material }
}

fn wall(x: f32, top: f32, bottom: f32, half_width: f32) -> Stroke {
    Stroke { from: [x, top], to: [x, bottom], radius: half_width, material: STONE }
}

fn material_at(cells: &[u32], x: u32, y: u32) -> u8 {
    (cells[(y * world::WIDTH + x) as usize] & 0xFF) as u8
}

fn histogram(cells: &[u32]) -> Vec<usize> {
    let mut counts = vec![0; MATERIALS.len()];
    for &c in cells {
        counts[(c & 0xFF) as usize] += 1;
    }
    counts
}

/// The block automaton only ever permutes cells, so every material's cell
/// count must survive any number of ticks exactly, however violent the mixing.
#[test]
fn mass_is_conserved_per_material() {
    let (gpu, mut sim) = lab();
    sim.paint_all(
        &gpu,
        &[
            disc(SAND, 1300.0, 500.0, 90.0),   // into the vessel, through the water
            disc(WATER, 1300.0, 700.0, 80.0),
            disc(OIL, 1150.0, 650.0, 50.0),
            disc(LAVA, 2900.0, 850.0, 70.0),   // onto the step, spilling off it
            disc(GRAVEL, 2600.0, 900.0, 60.0),
            disc(STEAM, 2000.0, 1150.0, 40.0), // rises
        ],
    );
    let before = histogram(&sim.read_cells(&gpu));
    for id in [SAND, WATER, OIL, LAVA, GRAVEL, STEAM] {
        assert!(before[id as usize] > 1000, "{} was not painted", MATERIALS[id as usize].name);
    }

    for _ in 0..4 {
        sim.run(&gpu, 1000);
        let after = histogram(&sim.read_cells(&gpu));
        for (id, (b, a)) in before.iter().zip(&after).enumerate() {
            assert_eq!(b, a, "{} count changed by tick {}", MATERIALS[id].name, sim.tick);
        }
    }
}

/// Water poured into a basin must settle flat: every column's surface within
/// one cell of every other.
#[test]
fn water_settles_level() {
    let (gpu, mut sim) = lab();
    let floor = LAB_FLOOR as f32;
    let (left, right) = (2000.0, 2100.0);
    sim.paint_all(
        &gpu,
        &[
            wall(left, floor - 120.0, floor - 1.0, 3.0),
            wall(right, floor - 120.0, floor - 1.0, 3.0),
            disc(WATER, 2030.0, floor - 170.0, 30.0), // off-centre, dropped from above
        ],
    );
    sim.run(&gpu, 40_000);
    let cells = sim.read_cells(&gpu);

    let mut surface = Vec::new();
    for x in (left as u32 + 4)..(right as u32 - 3) {
        let top = (LAB_FLOOR - 200..LAB_FLOOR)
            .find(|&y| material_at(&cells, x, y) == WATER)
            .unwrap_or_else(|| panic!("column {x} has no water"));
        surface.push(top);
    }
    let (lo, hi) = (surface.iter().min().unwrap(), surface.iter().max().unwrap());
    assert!(hi - lo <= 1, "water surface spans rows {lo}..={hi}: {surface:?}");
}

/// Sand heaps no steeper than the 2x2 stencil allows (45 degrees): adjacent
/// columns of a settled pile differ by at most one cell of height.
#[test]
fn sand_heap_holds_45_degrees() {
    let (gpu, mut sim) = lab();
    let (cx, cy, r) = (2200.0, LAB_FLOOR as f32 - 300.0, 60.0);
    sim.paint_all(&gpu, &[disc(SAND, cx, cy, r)]);
    sim.run(&gpu, 20_000);
    let cells = sim.read_cells(&gpu);

    let height = |x: u32| {
        (LAB_FLOOR - 400..LAB_FLOOR)
            .find(|&y| material_at(&cells, x, y) == SAND)
            .map_or(0, |y| LAB_FLOOR - y)
    };
    let heights: Vec<u32> = (1900..2500).map(height).collect();
    let peak = *heights.iter().max().unwrap();
    assert!(peak > 20, "no heap formed (peak {peak})");
    for (i, pair) in heights.windows(2).enumerate() {
        let step = pair[0].abs_diff(pair[1]);
        assert!(step <= 1, "slope of {step} cells at x={}: {:?}", 1900 + i, pair);
    }
}
