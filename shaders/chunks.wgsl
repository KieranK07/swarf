// ---------------------------------------------------------------------------
// swarf :: chunks.wgsl
//
// Between ticks, advance every chunk's wake state: whatever asked to run next
// tick now runs this tick, and everything else goes back to sleep.
//
// One thread per chunk. At 32x32 cells that is 8192 threads for a 4096x2048
// world -- immaterial next to the pass it gates.
// ---------------------------------------------------------------------------

struct ChunkParams {
    chunk_count: u32,
    // Three scalars rather than a vec3: a vec3<u32> is 16-byte aligned, which
    // would round the struct up to 32 bytes and no longer match the Rust side.
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> wake: array<u32>;
@group(0) @binding(1) var<uniform>             params: ChunkParams;

@compute @workgroup_size(256, 1, 1)
fn rotate_wake(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.chunk_count) { return; }
    // Shift the pending ticks down by one, so a chunk stays awake only while
    // something keeps asking for it. See WAKE_NEXT in common.wgsl.
    wake[i] = wake[i] >> 1u;
}
