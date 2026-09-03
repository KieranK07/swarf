//! The material table.
//!
//! Everything the simulation knows about a substance lives here as plain data
//! and is uploaded once as a storage buffer. The shaders never special-case a
//! material by name -- behaviour falls out of `kind` and `density` alone, so
//! adding a new substance is a matter of adding a row.

/// Movement class. Must match the `KIND_*` constants in `shaders/common.wgsl`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Kind {
    /// Never moves. Bedrock, machine housings, structural metal.
    Static = 0,
    /// Falls and slumps diagonally, so it heaps up at an angle of repose.
    Powder = 1,
    /// Falls, slumps, and spreads sideways when something is pressing on it.
    Fluid = 2,
}

/// Density of the ambient medium. Anything lighter rises, anything heavier
/// sinks, and gases need no special case at all.
///
/// Must match `AIR_DENSITY` in `shaders/common.wgsl`; [`Material::AIR`] is
/// asserted against it at startup.
pub const AIR_DENSITY: f32 = 10.0;

/// GPU-side material record. 32 bytes, `std430`-compatible.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMaterial {
    /// Linear rgb, with emissive strength in `a`.
    pub colour: [f32; 4],
    pub density: f32,
    pub kind: u32,
    /// How much per-cell brightness jitter to apply when drawing.
    pub variance: f32,
    /// Chance per tick that this material takes its gravity step. See
    /// [`Material::mobility`].
    pub mobility: f32,
}

pub struct Material {
    pub name: &'static str,
    pub kind: Kind,
    pub density: f32,
    /// Base colour as sRGB hex, the way a colour picker reports it.
    pub rgb: u32,
    pub emissive: f32,
    pub variance: f32,
    /// Probability per tick of taking the downward step, in 0..1.
    ///
    /// At 1.0 every cell in a falling mass tries to move on exactly the same
    /// tick. Only the bottom one can -- the rest have their own kind directly
    /// below them -- so a gap opens under the mass and walks upward one row per
    /// tick, in perfect lockstep. The mass falls as a barcode of alternating
    /// full and empty rows.
    ///
    /// The dilation is real (a poured stream genuinely thins as it falls); the
    /// regularity is not. Below 1.0 the gaps open at irregular heights and the
    /// stream reads as scattered grains. It doubles as a viscosity knob: lava
    /// oozes, water runs.
    pub mobility: f32,
}

impl Material {
    pub const AIR: u8 = 0;
    pub const BEDROCK: u8 = 1;
}

/// sRGB byte -> linear float.
///
/// The surface is an sRGB format, so the GPU applies the encoding transfer
/// function on write. Handing it sRGB values directly would double-encode and
/// wash everything out.
fn to_linear(c: u8) -> f32 {
    let s = c as f32 / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

/// The table. Index is the material id stored in the low byte of every cell,
/// so **order is load-bearing** -- append, never reorder.
pub const MATERIALS: &[Material] = &[
    Material { name: "Air",         kind: Kind::Fluid,  density: AIR_DENSITY, rgb: 0x0d1014, emissive: 0.0,  variance: 0.15, mobility: 1.00 },
    Material { name: "Bedrock",     kind: Kind::Static, density: 4000.0,      rgb: 0x1a1c22, emissive: 0.0,  variance: 0.25, mobility: 0.00 },
    Material { name: "Stone",       kind: Kind::Static, density: 250.0,       rgb: 0x5f646e, emissive: 0.0,  variance: 0.30, mobility: 0.00 },
    Material { name: "Dirt",        kind: Kind::Powder, density: 160.0,       rgb: 0x4a3728, emissive: 0.0,  variance: 0.35, mobility: 0.70 },
    Material { name: "Sand",        kind: Kind::Powder, density: 150.0,       rgb: 0xd4a95e, emissive: 0.0,  variance: 0.22, mobility: 0.78 },
    Material { name: "Gravel",      kind: Kind::Powder, density: 175.0,       rgb: 0x7c8088, emissive: 0.0,  variance: 0.40, mobility: 0.68 },
    Material { name: "Coal",        kind: Kind::Powder, density: 140.0,       rgb: 0x24262c, emissive: 0.0,  variance: 0.45, mobility: 0.74 },
    Material { name: "Iron ore",    kind: Kind::Powder, density: 185.0,       rgb: 0x8a6a52, emissive: 0.0,  variance: 0.40, mobility: 0.72 },
    Material { name: "Water",       kind: Kind::Fluid,  density: 100.0,       rgb: 0x2b6cb0, emissive: 0.0,  variance: 0.12, mobility: 0.92 },
    Material { name: "Oil",         kind: Kind::Fluid,  density: 80.0,        rgb: 0x231a12, emissive: 0.0,  variance: 0.20, mobility: 0.80 },
    Material { name: "Steam",       kind: Kind::Fluid,  density: 4.0,         rgb: 0xa8b8c4, emissive: 0.0,  variance: 0.25, mobility: 0.90 },
    Material { name: "Smoke",       kind: Kind::Fluid,  density: 6.0,         rgb: 0x2e3138, emissive: 0.0,  variance: 0.35, mobility: 0.85 },
    Material { name: "Molten iron", kind: Kind::Fluid,  density: 190.0,       rgb: 0xff7a24, emissive: 0.55, variance: 0.18, mobility: 0.62 },
    Material { name: "Lava",        kind: Kind::Fluid,  density: 205.0,       rgb: 0xe8431a, emissive: 0.45, variance: 0.22, mobility: 0.45 },
    Material { name: "Iron",        kind: Kind::Static, density: 210.0,       rgb: 0x9aa4b0, emissive: 0.0,  variance: 0.18, mobility: 0.00 },
    Material { name: "Wood",        kind: Kind::Static, density: 155.0,       rgb: 0x6b4a2c, emissive: 0.0,  variance: 0.30, mobility: 0.00 },
];

/// Keys 1-9 then 0.
pub const HOTBAR: [u8; 10] = [4, 8, 2, 3, 5, 6, 7, 9, 13, 0];

pub fn name(id: u8) -> &'static str {
    MATERIALS.get(id as usize).map_or("?", |m| m.name)
}

/// Flatten the table into the buffer the shaders bind.
///
/// Always emits a full 256 entries. A corrupt or out-of-range material id then
/// reads a defined (inert, black) row instead of running off the end of the
/// buffer, which on some backends is a hang rather than a visible glitch.
pub fn upload_table() -> Vec<GpuMaterial> {
    debug_assert_eq!(
        MATERIALS[Material::AIR as usize].density,
        AIR_DENSITY,
        "air's row must match AIR_DENSITY, or the shader's rise/sink test flips"
    );

    let mut table = vec![
        GpuMaterial {
            colour: [0.0; 4],
            density: AIR_DENSITY,
            kind: Kind::Static as u32,
            variance: 0.0,
            mobility: 0.0,
        };
        256
    ];

    for (i, m) in MATERIALS.iter().enumerate() {
        table[i] = GpuMaterial {
            colour: [
                to_linear((m.rgb >> 16) as u8),
                to_linear((m.rgb >> 8) as u8),
                to_linear(m.rgb as u8),
                m.emissive,
            ],
            density: m.density,
            kind: m.kind as u32,
            variance: m.variance,
            mobility: m.mobility,
        };
    }
    table
}
