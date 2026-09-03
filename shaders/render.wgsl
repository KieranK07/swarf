// ---------------------------------------------------------------------------
// swarf :: render.wgsl
//
// One fullscreen triangle. The fragment shader maps each screen pixel straight
// into the cell buffer -- there is no intermediate texture and no upload,
// because on Apple silicon the buffer the compute pass just wrote is already
// in the same physical memory the fragment shader reads. Zero copies per frame.
// ---------------------------------------------------------------------------

struct View {
    size:         vec2<u32>,   // world size in cells
    screen:       vec2<f32>,   // surface size in physical pixels
    centre:       vec2<f32>,   // world cell at the centre of the screen
    zoom:         f32,         // screen pixels per cell
    brush_radius: f32,
    brush_pos:    vec2<f32>,
    brush_show:   u32,
    _pad:         u32,
}

@group(0) @binding(0) var<storage, read> cells: array<u32>;
@group(0) @binding(1) var<storage, read> materials: array<Material>;
@group(0) @binding(2) var<uniform>       view: View;

struct VOut {
    @builtin(position) clip: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {
    // Oversized triangle covering the viewport: (-1,-1), (3,-1), (-1,3).
    var out: VOut;
    let x = f32((vi & 1u) << 2u) - 1.0;
    let y = f32((vi & 2u) << 1u) - 1.0;
    out.clip = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

// Colour of a single cell, before any screen-space effects.
fn shade_cell(p: vec2<i32>) -> vec3<f32> {
    if (p.x < 0 || p.y < 0 || p.x >= i32(view.size.x) || p.y >= i32(view.size.y)) {
        return vec3<f32>(0.02, 0.02, 0.03);           // outside the world
    }

    let c = cells[u32(p.y) * view.size.x + u32(p.x)];
    let m = materials[cell_mat(c)];

    // Per-cell brightness jitter, fixed at spawn time. This single line is the
    // difference between "flat blocks of colour" and something that reads as
    // granular material.
    let jitter = (f32(cell_var(c)) / 255.0 - 0.5) * m.variance;
    var col = m.colour.rgb * (1.0 + jitter);

    // Emissive materials (lava, molten metal) push past 1.0 so they still read
    // as hot against bright surroundings.
    col += m.colour.rgb * m.colour.a;

    return col;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let screen_offset = in.clip.xy - view.screen * 0.5;
    let world = view.centre + screen_offset / view.zoom;

    var col: vec3<f32>;

    if (view.zoom < 1.0) {
        // Zoomed out past 1:1 a single tap aliases badly -- fine detail
        // crawls and shimmers as the camera moves. Four taps on a rotated
        // grid is enough to settle it down.
        let s = 0.5 / view.zoom;
        col  = shade_cell(vec2<i32>(floor(world + vec2<f32>(-s, -s * 0.5))));
        col += shade_cell(vec2<i32>(floor(world + vec2<f32>( s * 0.5, -s))));
        col += shade_cell(vec2<i32>(floor(world + vec2<f32>( s,  s * 0.5))));
        col += shade_cell(vec2<i32>(floor(world + vec2<f32>(-s * 0.5,  s))));
        col *= 0.25;
    } else {
        col = shade_cell(vec2<i32>(floor(world)));
    }

    // Brush outline, drawn as a ring of constant *screen* thickness so it stays
    // readable at every zoom level.
    if (view.brush_show != 0u) {
        let d = length(world - view.brush_pos) - view.brush_radius;
        let edge = abs(d) * view.zoom;
        if (edge < 1.5) {
            col = mix(vec3<f32>(1.0, 1.0, 1.0), col, edge / 1.5);
        }
    }

    return vec4<f32>(col, 1.0);
}
