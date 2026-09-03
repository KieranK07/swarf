// ---------------------------------------------------------------------------
// swarf :: sim.wgsl  --  the falling-sand core
//
// This is a *Margolus block cellular automaton*. The grid is carved into 2x2
// blocks and one GPU thread owns one block outright: it reads its four cells,
// permutes them, and writes those same four cells back. Nothing else touches
// them.
//
// Why this and not the usual per-cell "look down, move if empty" rule:
//
//   * No write conflicts, so no atomics and no arbitration pass. Two grains
//     can never both claim the same destination, because destinations never
//     span two threads.
//   * Mass is conserved *exactly*. Cells are only ever permuted, never
//     created or destroyed. For a factory game that is not a nicety -- it is
//     the difference between a throughput number you can trust and one that
//     quietly drifts. Ore in really does equal ingots out.
//   * It maps perfectly onto the GPU. Every thread does identical work with
//     zero divergent memory traffic.
//
// The cost is that a 2x2 block only sees itself, so material moves at most one
// cell per tick and the partition seam is visible if you hold it still. Both
// are fixed by shifting the block origin every tick (params.offset cycles
// through the four 2x2 phases), which is why the offset exists.
//
// IMPORTANT -- the one rule you must not break:
//   A thread may READ any cell in the world, but may only WRITE its own four.
//   Neighbour reads are racy by design: another workgroup may be mid-write, so
//   you get either the pre- or post-tick value. That is fine, because those
//   reads only ever feed a *decision* (is this cell under pressure?), never a
//   write. Worst case a drop of water spreads one tick early. It cannot
//   corrupt state and it cannot lose mass. Reading a neighbour to decide what
//   to *write there* would break everything -- don't.
// ---------------------------------------------------------------------------

struct SimParams {
    size: vec2<u32>,       // world size in cells
    offset: vec2<i32>,     // Margolus partition phase, each component 0 or 1
    chunks: vec2<u32>,     // world size in chunks
    tick: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> cells: array<u32>;
@group(0) @binding(1) var<storage, read>       materials: array<Material>;
@group(0) @binding(2) var<uniform>             params: SimParams;
@group(0) @binding(3) var<storage, read_write> wake: array<atomic<u32>>;

// Set by any thread that permutes its block; reduced once per workgroup.
var<workgroup> wg_moved: atomic<u32>;

// --- the block this thread owns -------------------------------------------
// Held in private storage so the helper functions below can mutate it without
// WGSL references. Index order:  0 = top-left, 1 = top-right,
//                                2 = bottom-left, 3 = bottom-right.
// +y is down, so gravity pulls 0->2 and 1->3.

var<private> blk:  array<u32, 4>;
var<private> lock: array<bool, 4>;   // a cell may take part in one swap per tick
var<private> pos:  array<vec2<i32>, 4>;
var<private> rng:  u32;

fn next_rand() -> u32 {
    rng = hash_u32(rng);
    return rng;
}

// --- material queries ------------------------------------------------------

fn dens(c: u32) -> f32   { return materials[cell_mat(c)].density; }
fn kind(c: u32) -> u32   { return materials[cell_mat(c)].kind; }
fn movable(c: u32) -> bool { return materials[cell_mat(c)].kind != KIND_STATIC; }

// Can `top` displace `bot` by falling into it? Purely a density comparison, so
// sand sinks through water, water sinks through steam, and steam "sinks"
// upward through air -- all from one rule.
fn can_sink(top: u32, bot: u32) -> bool {
    return movable(top) && movable(bot) && dens(top) > dens(bot) + 0.5;
}

// Sideways flow. Only fluids spread, and only into something they outrank in
// the direction gravity is pushing them: heavy fluids shoulder aside lighter
// cells, gases shoulder aside heavier ones.
fn can_push_h(src: u32, dst: u32) -> bool {
    let ms = cell_mat(src);
    if (ms == MAT_AIR) { return false; }          // air is the medium, not a flow
    if (materials[ms].kind != KIND_FLUID) { return false; }
    if (!movable(dst)) { return false; }
    let ds = dens(src);
    let dd = dens(dst);
    if (ds > AIR_DENSITY) { return ds > dd + 0.5; }
    return ds < dd - 0.5;
}

fn in_bounds(p: vec2<i32>) -> bool {
    return p.x >= 0 && p.y >= 0 && p.x < i32(params.size.x) && p.y < i32(params.size.y);
}

fn idx(p: vec2<i32>) -> u32 {
    return u32(p.y) * params.size.x + u32(p.x);
}

// Is this cell being squeezed from the direction it fell from?
//
// This is what makes water behave, and it is worth being precise about.
//
// Without a pressure test, a fluid free to move sideways random-walks forever:
// a flat pond shimmers, never settles, and its chunks never sleep. A plain
// sideways swap moves no material up or down, so nothing in the rule set
// distinguishes it from its own reverse.
//
// Requiring weight overhead breaks that symmetry. But it must include the two
// *diagonal* neighbours, not just the cell directly above. Check only straight
// up and a heap of water is stable by exactly one cell: every cell with an air
// neighbour to flow into is a surface cell, and every surface cell has air
// directly above it. The mound sits there as a dome and never levels. Its
// outermost cells do have fluid diagonally above, which is precisely the weight
// that ought to be pushing them out.
//
// The result is that fluid spreads until it is a single layer and then stops
// dead, because a lone layer has nothing above it anywhere. Settled water is
// genuinely free.
//
// Heavy materials are pressed from above; gases are pressed from below.
fn is_pressured(p: vec2<i32>, c: u32) -> bool {
    var dy = -1;
    if (dens(c) <= AIR_DENSITY) { dy = 1; }

    for (var dx = -1; dx <= 1; dx++) {
        let q = p + vec2<i32>(dx, dy);
        if (!in_bounds(q)) { return true; }        // world edge acts as a lid
        let n = cells[idx(q)];                      // racy neighbour read -- see header
        if (!movable(n)) {
            // Solid directly overhead is a lid and does press down. Solid off to
            // one side is just a wall, and holds nothing up.
            if (dx == 0) { return true; }
            continue;
        }
        if (dens(n) >= dens(c) - 0.5) { return true; }
    }
    return false;
}

// --- swap primitives -------------------------------------------------------

fn swap_cells(i: u32, j: u32) {
    let t = blk[i]; blk[i] = blk[j]; blk[j] = t;
    lock[i] = true;
    lock[j] = true;
    atomicStore(&wg_moved, 1u);
}

// Gravity-driven: cell i falls into cell j.
fn attempt_fall(i: u32, j: u32) {
    if (lock[i] || lock[j]) { return; }
    if (!can_sink(blk[i], blk[j])) { return; }
    swap_cells(i, j);
}

// Is this cell being shoved from behind by more of the same fluid?
//
// Weight from above alone is not enough to make water level. A wedge of water
// is stable under an above-only test: the cells that could move are the ones at
// the end of each row, and those are exactly the cells with nothing above them.
// The water at the toe of a slope is not pushed down by anything -- it is
// pushed *sideways*, by the water behind it.
//
// One cell of look-behind is all it takes. Pressure then walks along a row one
// cell per tick, the toe advances, and the surface flattens properly. It also
// keeps isolated droplets still: a drop with air on both sides has nothing
// behind it and no weight on top, so it stays where it landed instead of
// wandering the floor forever and holding its chunk awake.
fn has_backing(p: vec2<i32>, c: u32, dir: i32) -> bool {
    let q = p - vec2<i32>(dir, 0);
    if (!in_bounds(q)) { return false; }        // a wall behind pushes nothing
    let n = cells[idx(q)];                       // racy neighbour read -- see header
    return movable(n) && dens(n) >= dens(c) - 0.5;
}

// Sideways flow between horizontal neighbours i and j. A fluid moves if it has
// either weight above it or fluid behind it.
fn attempt_flow(i: u32, j: u32) {
    if (lock[i] || lock[j]) { return; }
    let dir = pos[j].x - pos[i].x;               // +1 when j lies to the right of i
    let i_pushes = can_push_h(blk[i], blk[j])
        && (is_pressured(pos[i], blk[i]) || has_backing(pos[i], blk[i], dir));
    let j_pushes = can_push_h(blk[j], blk[i])
        && (is_pressured(pos[j], blk[j]) || has_backing(pos[j], blk[j], -dir));
    if (!i_pushes && !j_pushes) { return; }
    swap_cells(i, j);
}

// --- entry point -----------------------------------------------------------
//
// One workgroup per 32x32 chunk. A chunk whose contents have stopped moving is
// skipped entirely -- that is the whole energy story of this simulation.
// Simulating settled rock, still air and level water costs the same as
// simulating an avalanche, and almost all of a mature world is settled. Sleeping
// means an idle factory draws close to nothing, which on a fanless machine is
// the difference between silence and a hot lap.
//
// A chunk wakes when anything moves in it, and wakes its eight neighbours too,
// since a block straddles the chunk edge on odd partition phases and material
// crossing a boundary must not land in a chunk that is not looking.

fn wake_chunk(cx: i32, cy: i32) {
    if (cx < 0 || cy < 0 || cx >= i32(params.chunks.x) || cy >= i32(params.chunks.y)) {
        return;
    }
    atomicOr(&wake[u32(cy) * params.chunks.x + u32(cx)], WAKE_NEXT);
}

@compute @workgroup_size(16, 16, 1)
fn sim_main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let chunk = wg.y * params.chunks.x + wg.x;
    // Uniform across the workgroup, so the barriers below stay in uniform
    // control flow -- which is why this is a guard rather than an early return.
    let awake = (atomicLoad(&wake[chunk]) & WAKE_NOW) != 0u;

    if (li == 0u) { atomicStore(&wg_moved, 0u); }
    workgroupBarrier();

    if (awake) {
        let chunk_origin = vec2<i32>(vec2<u32>(wg.x, wg.y) * CHUNK);
        let origin = chunk_origin + vec2<i32>(lid.xy) * 2 + params.offset;
        simulate_block(origin);
    }

    workgroupBarrier();

    if (li == 0u && atomicLoad(&wg_moved) != 0u) {
        let cx = i32(wg.x);
        let cy = i32(wg.y);
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                wake_chunk(cx + dx, cy + dy);
            }
        }
    }
}

fn simulate_block(origin: vec2<i32>) {
    // Blocks that hang off the edge of the world are skipped rather than
    // clamped. The world carries a bedrock border, so nothing is lost.
    if (origin.x < 0 || origin.y < 0) { return; }
    if (origin.x + 1 >= i32(params.size.x) || origin.y + 1 >= i32(params.size.y)) { return; }

    pos[0] = origin;
    pos[1] = origin + vec2<i32>(1, 0);
    pos[2] = origin + vec2<i32>(0, 1);
    pos[3] = origin + vec2<i32>(1, 1);

    for (var i = 0u; i < 4u; i++) {
        blk[i]  = cells[idx(pos[i])];
        lock[i] = false;
    }

    // Early out on blocks that physically cannot do anything. Solid rock is
    // most of a mature world, and this check is four table lookups against
    // eight bytes of writes.
    if (!movable(blk[0]) && !movable(blk[1]) && !movable(blk[2]) && !movable(blk[3])) {
        return;
    }

    rng = hash2(u32(origin.x), u32(origin.y), params.tick);

    // 1. Straight down. Both columns fall independently.
    attempt_fall(0u, 2u);
    attempt_fall(1u, 3u);

    // 2. Diagonal slump. A grain only reaches here if the cell directly below
    //    it was blocked (otherwise step 1 locked it), which is exactly when a
    //    real pile slides. Order is randomised so heaps do not lean.
    if ((next_rand() & 1u) == 0u) {
        attempt_fall(0u, 3u);
        attempt_fall(1u, 2u);
    } else {
        attempt_fall(1u, 2u);
        attempt_fall(0u, 3u);
    }

    // 3. Sideways flow, bottom row first so fluid prefers the floor.
    attempt_flow(2u, 3u);
    attempt_flow(0u, 1u);

    for (var i = 0u; i < 4u; i++) {
        cells[idx(pos[i])] = blk[i];
    }
}
