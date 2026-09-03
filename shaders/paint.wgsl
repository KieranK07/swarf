// ---------------------------------------------------------------------------
// swarf :: paint.wgsl
//
// Stamps material into the world. Takes a *segment* rather than a point: the
// mouse moves far more than one cell between frames, so painting a disc at the
// cursor leaves a dotted trail. Stamping the capsule swept from last frame's
// position to this one gives a continuous stroke at any speed.
//
// Runs on the GPU so the cell buffer never has to travel back to the CPU.
// ---------------------------------------------------------------------------

struct PaintParams {
    size:        vec2<u32>,
    chunks:      vec2<u32>,
    origin:      vec2<i32>,    // top-left cell of the dispatched region
    p0:          vec2<f32>,    // stroke start, world cells
    p1:          vec2<f32>,    // stroke end
    radius:      f32,
    material:    u32,
    temperature: u32,
    tick:        u32,
}

@group(0) @binding(0) var<storage, read_write> cells: array<u32>;
@group(0) @binding(1) var<storage, read>       materials: array<Material>;
@group(0) @binding(2) var<uniform>             pp: PaintParams;
@group(0) @binding(3) var<storage, read_write> wake: array<atomic<u32>>;

fn dist_to_segment(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let len2 = dot(ab, ab);
    if (len2 < 1e-6) { return length(p - a); }
    let t = clamp(dot(p - a, ab) / len2, 0.0, 1.0);
    return length(p - (a + ab * t));
}

@compute @workgroup_size(8, 8, 1)
fn paint_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = pp.origin + vec2<i32>(i32(gid.x), i32(gid.y));
    if (p.x < 0 || p.y < 0 || p.x >= i32(pp.size.x) || p.y >= i32(pp.size.y)) { return; }

    let centre = vec2<f32>(f32(p.x) + 0.5, f32(p.y) + 0.5);
    if (dist_to_segment(centre, pp.p0, pp.p1) > pp.radius) { return; }

    let i = u32(p.y) * pp.size.x + u32(p.x);

    // Bedrock is the world's container. Letting the brush chew through it
    // would let material fall out of the simulation entirely.
    if (cell_mat(cells[i]) == 1u) { return; }

    let variant = hash2(u32(p.x), u32(p.y), pp.tick) & 0xFFu;
    cells[i] = make_cell(pp.material, variant, pp.temperature);

    // Wake this chunk and its neighbours, both now and next tick. Painted
    // material has to start moving immediately, and it may well have landed in
    // a region that has been asleep for minutes.
    let cx = i32(u32(p.x) / CHUNK);
    let cy = i32(u32(p.y) / CHUNK);
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let nx = cx + dx;
            let ny = cy + dy;
            if (nx >= 0 && ny >= 0 && nx < i32(pp.chunks.x) && ny < i32(pp.chunks.y)) {
                atomicOr(&wake[u32(ny) * pp.chunks.x + u32(nx)], WAKE_NOW | WAKE_NEXT);
            }
        }
    }
}
