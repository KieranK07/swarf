# swarf

A granular-material sandbox: every pixel is a cell of real material with a
density and a temperature, falling, flowing and piling under its own rules.
The intended destination is a factory game where machines crush, melt and cast
actual material rather than incrementing a counter.

Rust + wgpu (Metal), simulation on the GPU, 2D.

```
cargo run --release
```

| | |
|---|---|
| LMB / RMB | paint / erase |
| MMB or space + drag | pan |
| scroll | zoom at cursor |
| `1`-`0` | material hotbar |
| `Q` / `E` | cycle all materials |
| `[` / `]` | brush size |
| `P` / `N` | pause / single step |
| `R` | regenerate world |

## How it works

The world is 4096x2048 cells. One cell is one `u32` — material id, a colour
variant, and a temperature in Kelvin — so the whole world is 32 MB and lives in
a single storage buffer.

**The simulation is a Margolus block cellular automaton.** The grid is carved
into 2x2 blocks and one GPU thread owns one block outright: it reads its four
cells, permutes them, and writes those same four back. Nothing else touches
them. This buys three things:

- **No write conflicts**, so no atomics and no arbitration pass. Two grains can
  never both claim one destination, because destinations never span two threads.
- **Exact mass conservation.** Cells are permuted, never created or destroyed.
  For a factory game that is not a nicety — it is the difference between a
  throughput number you can trust and one that quietly drifts.
- **A uniform GPU workload**, every thread doing identical work.

The cost is that a block only sees itself, so material moves at most one cell
per tick. The partition origin shifts every tick to hide the seam; the y
component *must* flip every tick, or everything falls at half speed.

Because one cell per tick is the hard ceiling on every transport rate, the
simulation runs at **240 Hz**, decoupled from the display.

**Chunk sleeping is what makes that affordable.** A 32x32 chunk whose contents
have stopped moving is not dispatched at all, and almost all of a mature world
is settled at any moment. On a 4096x2048 world:

| | ms per tick |
|---|---|
| simulating everything | 0.72 |
| with chunk sleeping, settled | 0.11 |

Rendering is one fullscreen triangle whose fragment shader samples the cell
buffer directly — no intermediate texture and no upload, because on unified
memory the buffer the compute pass just wrote is already where the fragment
shader reads. Zero copies per frame.

## Rules worth knowing

Movement is driven by density alone: denser sinks. Gases need no special case —
steam is lighter than air, so it "sinks" upward for free.

Two rules took real work to get right, and both are easy to get subtly wrong:

- **Sand heaps at 45 degrees**, the theoretical maximum for a 2x2 stencil.
- **Water needs pressure from above *and* from behind.** Weight alone is not
  enough: the cells free to move sideways are exactly the ones with nothing on
  top of them, so water settles into a stable 26-degree wedge and never levels.
  One cell of look-behind fixes it, and keeps isolated droplets from wandering
  the floor forever holding their chunk awake.
- **Mobility below 1.0.** With every cell trying to fall on the same tick, only
  the bottom of a falling mass can move, so a gap walks up it in lockstep and it
  descends as a barcode of alternating full and empty rows. The dilation is real;
  the regularity is not.

## Developing

The physics is tuned by *looking* at it, so there is a headless renderer:

```
swarf --shot out.png --scene lab --ticks 6000 --zoom 1 \
      --centre 2100,1090 --drop sand@2100,850,60
```

`--scene lab` is a bare rig — flat floor, a step, a sealed vessel — because
terrain noise makes it impossible to tell a rule bug from a lumpy cave. A heap
should hold a stable angle; a vessel should end up level. `swarf --help` lists
the rest.

## Layout

```
shaders/common.wgsl   cell packing, material struct, hashing (prepended to the rest)
shaders/sim.wgsl      the block automaton and chunk sleeping
shaders/chunks.wgsl   advances wake state between ticks
shaders/paint.wgsl    brush strokes, stamped as swept capsules
shaders/render.wgsl   fullscreen triangle, palette, brush ring
src/materials.rs      the material table — behaviour is data, not code
src/world.rs          cell packing and terrain generation
src/sim.rs            buffers, compute pipelines, the tick
```

Adding a material is a row in `MATERIALS`. Nothing in the shaders knows any
material by name.
