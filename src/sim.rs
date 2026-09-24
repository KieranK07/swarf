//! GPU simulation resources: the cell buffer, the sim pass, and the brush pass.

use std::num::NonZeroU64;
use wgpu::util::DeviceExt;

use crate::gpu::Gpu;
use crate::materials;
use crate::world;

/// The four Margolus partition phases.
///
/// The **y component must flip on every tick**. A cell can only fall when it
/// sits in the top row of its block, so if y holds still for two ticks in a row
/// everything falls at half speed: it drops one cell, lands in a bottom row,
/// and is stuck until the partition moves again. Cycling y as 0,1,0,1 keeps
/// every falling cell in a top row every tick.
///
/// x cycles on a period of four instead, which varies which neighbour a grain
/// can slump into without disturbing the vertical cadence.
const OFFSETS: [[i32; 2]; 4] = [[0, 0], [1, 1], [1, 0], [0, 1]];

/// Upper bound on catch-up ticks in one frame. Past this the sim deliberately
/// runs slow rather than spiralling: trying to catch up after a stall just
/// makes the next frame later still.
///
/// Steady state at 240 Hz on a 60 Hz display is four; the headroom absorbs a
/// dropped frame or two without the world visibly lurching.
pub const MAX_TICKS_PER_FRAME: usize = 12;

/// Wake bits, mirroring `WAKE_NOW` / `WAKE_NEXT` in `shaders/common.wgsl`.
const WAKE_NOW: u32 = 1;
const WAKE_NEXT: u32 = 2;

/// Dynamic uniform offsets must be a multiple of
/// `min_uniform_buffer_offset_alignment`, which is 256 in the default limits.
const UNIFORM_STRIDE: u64 = 256;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SimParams {
    size: [u32; 2],
    offset: [i32; 2],
    chunks: [u32; 2],
    tick: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ChunkParams {
    chunk_count: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PaintParams {
    size: [u32; 2],
    chunks: [u32; 2],
    origin: [i32; 2],
    p0: [f32; 2],
    p1: [f32; 2],
    radius: f32,
    material: u32,
    temperature: u32,
    tick: u32,
}

/// A brush stroke, as a swept segment in cell coordinates.
pub struct Stroke {
    pub from: [f32; 2],
    pub to: [f32; 2],
    pub radius: f32,
    pub material: u8,
}

pub struct Sim {
    pub cells: wgpu::Buffer,
    pub materials: wgpu::Buffer,
    /// One `u32` per chunk holding the wake bits. See `WAKE_*` in common.wgsl.
    wake: wgpu::Buffer,
    sim_params: wgpu::Buffer,
    paint_params: wgpu::Buffer,
    /// Held only to keep the buffer alive for `chunk_bind`.
    _chunk_params: wgpu::Buffer,
    sim_bind: wgpu::BindGroup,
    paint_bind: wgpu::BindGroup,
    chunk_bind: wgpu::BindGroup,
    sim_pipeline: wgpu::ComputePipeline,
    paint_pipeline: wgpu::ComputePipeline,
    rotate_pipeline: wgpu::ComputePipeline,
    pub tick: u32,
    /// World size in chunks; also the sim dispatch dimensions.
    chunks: [u32; 2],
}

impl Sim {
    pub fn new(gpu: &Gpu, seed: u32, scene: world::Scene) -> Self {
        let device = &gpu.device;

        let chunks = [
            world::WIDTH.div_ceil(world::CHUNK),
            world::HEIGHT.div_ceil(world::CHUNK),
        ];
        let chunk_count = chunks[0] * chunks[1];

        let initial = world::generate(seed, scene);
        let cells = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cells"),
            contents: bytemuck::cast_slice(&initial),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let table = materials::upload_table();
        let materials_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("materials"),
            contents: bytemuck::cast_slice(&table),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Everything starts awake; the first few ticks put to sleep whatever
        // was already settled at generation time.
        let wake = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk wake"),
            contents: bytemuck::cast_slice(&vec![WAKE_NOW | WAKE_NEXT; chunk_count as usize]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let sim_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim params"),
            size: UNIFORM_STRIDE * MAX_TICKS_PER_FRAME as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let paint_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("paint params"),
            size: size_of::<PaintParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let chunk_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk params"),
            contents: bytemuck::bytes_of(&ChunkParams { chunk_count, _pad: [0; 3] }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // --- bind group layouts ---------------------------------------------
        let entry = |binding: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let storage = |read_only: bool| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let uniform = |dynamic: bool, size: u64| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: dynamic,
            min_binding_size: NonZeroU64::new(size),
        };

        // The sim and paint passes share a shape: cells, materials, params,
        // wake. Only the params buffer differs.
        let world_layout = |label: &str, dynamic: bool, size: u64| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &[
                    entry(0, storage(false)),
                    entry(1, storage(true)),
                    entry(2, uniform(dynamic, size)),
                    entry(3, storage(false)),
                ],
            })
        };
        let sim_layout = world_layout("sim bgl", true, size_of::<SimParams>() as u64);
        let paint_layout = world_layout("paint bgl", false, size_of::<PaintParams>() as u64);

        let chunk_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("chunk bgl"),
            entries: &[
                entry(0, storage(false)),
                entry(1, uniform(false, size_of::<ChunkParams>() as u64)),
            ],
        });

        // --- bind groups -----------------------------------------------------
        let world_bind = |label: &str, bgl: &wgpu::BindGroupLayout, params: &wgpu::Buffer, size: u64| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: cells.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: materials_buf.as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: params,
                            offset: 0,
                            size: NonZeroU64::new(size),
                        }),
                    },
                    wgpu::BindGroupEntry { binding: 3, resource: wake.as_entire_binding() },
                ],
            })
        };

        let sim_bind =
            world_bind("sim bg", &sim_layout, &sim_params, size_of::<SimParams>() as u64);
        let paint_bind =
            world_bind("paint bg", &paint_layout, &paint_params, size_of::<PaintParams>() as u64);

        let chunk_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("chunk bg"),
            layout: &chunk_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wake.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: chunk_params.as_entire_binding() },
            ],
        });

        // --- pipelines --------------------------------------------------------
        let compute = |label: &str, bgl: &wgpu::BindGroupLayout, src: &str, entry_point: &str| {
            let module = gpu.shader(label, src);
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(bgl)],
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let sim_pipeline =
            compute("sim", &sim_layout, include_str!("../shaders/sim.wgsl"), "sim_main");
        let paint_pipeline =
            compute("paint", &paint_layout, include_str!("../shaders/paint.wgsl"), "paint_main");
        let rotate_pipeline = compute(
            "rotate wake",
            &chunk_layout,
            include_str!("../shaders/chunks.wgsl"),
            "rotate_wake",
        );

        Self {
            cells,
            materials: materials_buf,
            wake,
            sim_params,
            paint_params,
            _chunk_params: chunk_params,
            sim_bind,
            paint_bind,
            chunk_bind,
            sim_pipeline,
            paint_pipeline,
            rotate_pipeline,
            tick: 0,
            chunks,
        }
    }

    pub fn chunk_count(&self) -> u32 {
        self.chunks[0] * self.chunks[1]
    }

    /// Stage the uniforms for every tick this frame will run.
    ///
    /// `queue.write_buffer` is ordered against *submission*, not against passes
    /// inside an encoder -- writing between two encoded passes would leave both
    /// reading the last value written. Staging all the ticks up front and
    /// selecting between them with a dynamic offset keeps the whole frame to a
    /// single write and a single submit.
    pub fn prepare(&self, queue: &wgpu::Queue, ticks: usize) {
        debug_assert!(ticks <= MAX_TICKS_PER_FRAME);
        let mut staging = vec![0u8; ticks * UNIFORM_STRIDE as usize];
        for i in 0..ticks {
            let tick = self.tick.wrapping_add(i as u32);
            let params = SimParams {
                size: [world::WIDTH, world::HEIGHT],
                offset: OFFSETS[(tick % 4) as usize],
                chunks: self.chunks,
                tick,
                _pad: 0,
            };
            let at = i * UNIFORM_STRIDE as usize;
            staging[at..at + size_of::<SimParams>()].copy_from_slice(bytemuck::bytes_of(&params));
        }
        queue.write_buffer(&self.sim_params, 0, &staging);
    }

    pub fn encode_tick(&self, encoder: &mut wgpu::CommandEncoder, slot: usize) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sim tick"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sim_pipeline);
            pass.set_bind_group(0, &self.sim_bind, &[(slot as u64 * UNIFORM_STRIDE) as u32]);
            pass.dispatch_workgroups(self.chunks[0], self.chunks[1], 1);
        }

        // Separate pass so it observes every wake bit the tick just set.
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rotate wake"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.rotate_pipeline);
        pass.set_bind_group(0, &self.chunk_bind, &[]);
        pass.dispatch_workgroups(self.chunk_count().div_ceil(256), 1, 1);
    }

    /// Stamp a stroke into the world. Dispatches only over the segment's
    /// bounding box, so brush cost scales with the brush, not the world.
    pub fn paint(&self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, s: &Stroke) {
        let pad = s.radius + 2.0;
        let min_x = (s.from[0].min(s.to[0]) - pad).floor().max(0.0) as i32;
        let min_y = (s.from[1].min(s.to[1]) - pad).floor().max(0.0) as i32;
        let max_x = ((s.from[0].max(s.to[0]) + pad).ceil() as i32).min(world::WIDTH as i32 - 1);
        let max_y = ((s.from[1].max(s.to[1]) + pad).ceil() as i32).min(world::HEIGHT as i32 - 1);
        if max_x < min_x || max_y < min_y {
            return;
        }

        let temperature = if s.material == 13 { 1500 } else { world::AMBIENT_K };
        queue.write_buffer(
            &self.paint_params,
            0,
            bytemuck::bytes_of(&PaintParams {
                size: [world::WIDTH, world::HEIGHT],
                chunks: self.chunks,
                origin: [min_x, min_y],
                p0: s.from,
                p1: s.to,
                radius: s.radius,
                material: s.material as u32,
                temperature: temperature as u32,
                tick: self.tick,
            }),
        );

        let w = (max_x - min_x + 1) as u32;
        let h = (max_y - min_y + 1) as u32;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("paint"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.paint_pipeline);
        pass.set_bind_group(0, &self.paint_bind, &[]);
        pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
    }

    /// Stamp a list of strokes, one submit each: the strokes share a uniform
    /// buffer, and `write_buffer` is ordered against submission, not passes.
    pub fn paint_all(&self, gpu: &Gpu, strokes: &[Stroke]) {
        for stroke in strokes {
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("paint") });
            self.paint(&gpu.queue, &mut encoder, stroke);
            gpu.queue.submit([encoder.finish()]);
        }
    }

    /// Run `ticks` ticks back to back, in batches of [`MAX_TICKS_PER_FRAME`].
    pub fn run(&mut self, gpu: &Gpu, ticks: u32) {
        let mut remaining = ticks as usize;
        while remaining > 0 {
            let batch = remaining.min(MAX_TICKS_PER_FRAME);
            self.prepare(&gpu.queue, batch);
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("run") });
            for slot in 0..batch {
                self.encode_tick(&mut encoder, slot);
            }
            gpu.queue.submit([encoder.finish()]);
            self.tick = self.tick.wrapping_add(batch as u32);
            remaining -= batch;
        }
    }

    /// Copy the whole cell buffer back to the CPU. Blocks until the GPU is done.
    #[cfg(test)]
    pub fn read_cells(&self, gpu: &Gpu) -> Vec<u32> {
        let size = (world::CELL_COUNT * 4) as u64;
        let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cells readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
        encoder.copy_buffer_to_buffer(&self.cells, 0, &readback, 0, size);
        gpu.queue.submit([encoder.finish()]);

        let (tx, rx) = std::sync::mpsc::channel();
        readback.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r);
        });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll failed");
        rx.recv().expect("map channel closed").expect("buffer map failed");
        let cells = bytemuck::cast_slice(
            &readback.slice(..).get_mapped_range().expect("failed to read mapped buffer"),
        )
        .to_vec();
        readback.unmap();
        cells
    }

    pub fn reset(&mut self, queue: &wgpu::Queue, seed: u32, scene: world::Scene) {
        let fresh = world::generate(seed, scene);
        queue.write_buffer(&self.cells, 0, bytemuck::cast_slice(&fresh));
        queue.write_buffer(
            &self.wake,
            0,
            bytemuck::cast_slice(&vec![WAKE_NOW | WAKE_NEXT; self.chunk_count() as usize]),
        );
        self.tick = 0;
    }
}
