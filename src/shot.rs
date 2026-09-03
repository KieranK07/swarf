//! Headless capture: run the simulation for N ticks and write a PNG.
//!
//! Not a toy. A falling-sand ruleset is tuned by *looking* at it -- whether
//! sand heaps at a believable angle, whether water actually levels, whether a
//! pile settles instead of shimmering forever. This makes those questions
//! answerable in one command, without a window, and makes them diffable across
//! rule changes.

use std::path::Path;

use crate::camera::Camera;
use crate::gpu::Gpu;
use crate::render::Renderer;
use crate::sim::{self, Sim, Stroke};

pub struct ShotOptions {
    pub width: u32,
    pub height: u32,
    pub ticks: u32,
    pub zoom: f32,
    pub centre: Option<[f32; 2]>,
    pub seed: u32,
    pub strokes: Vec<Stroke>,
    pub out: String,
    pub scene: crate::world::Scene,
}

pub fn capture(opts: ShotOptions) {
    let started = std::time::Instant::now();
    let gpu = Gpu::headless(opts.width, opts.height);
    let mut sim = Sim::new(&gpu, opts.seed, opts.scene);
    let renderer = Renderer::new(&gpu, &sim);

    let mut camera = Camera::new(1.0);
    camera.zoom = opts.zoom;
    if let Some(c) = opts.centre {
        camera.centre = c;
    }

    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shot target"),
        size: wgpu::Extent3d {
            width: opts.width,
            height: opts.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: Gpu::HEADLESS_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());

    // `copy_texture_to_buffer` requires rows padded to 256 bytes.
    let unpadded = opts.width * 4;
    let padded = unpadded.div_ceil(256) * 256;
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("shot readback"),
        size: (padded * opts.height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Apply the requested strokes before simulating, so they have time to fall.
    if !opts.strokes.is_empty() {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("shot paint") });
        for stroke in &opts.strokes {
            sim.paint(&gpu.queue, &mut encoder, stroke);
            // One submit per stroke: the strokes share a uniform buffer, and
            // `write_buffer` is ordered against submission, not against passes.
            gpu.queue.submit([encoder.finish()]);
            encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("shot paint"),
            });
        }
        gpu.queue.submit([encoder.finish()]);
    }

    let mut remaining = opts.ticks as usize;
    while remaining > 0 {
        let batch = remaining.min(sim::MAX_TICKS_PER_FRAME);
        sim.prepare(&gpu.queue, batch);
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("shot sim") });
        for slot in 0..batch {
            sim.encode_tick(&mut encoder, slot);
        }
        gpu.queue.submit([encoder.finish()]);
        sim.tick = sim.tick.wrapping_add(batch as u32);
        remaining -= batch;
    }

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("shot draw") });
    renderer.draw(&gpu, &mut encoder, &view, &camera, [0.0, 0.0], 0.0, false);
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(opts.height),
            },
        },
        wgpu::Extent3d {
            width: opts.width,
            height: opts.height,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit([encoder.finish()]);

    let (tx, rx) = std::sync::mpsc::channel();
    readback.map_async(wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r);
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll failed");
    rx.recv().expect("map channel closed").expect("buffer map failed");

    let mapped = readback
        .slice(..)
        .get_mapped_range()
        .expect("failed to read back mapped buffer");
    let mut pixels = Vec::with_capacity((unpadded * opts.height) as usize);
    for row in 0..opts.height {
        let start = (row * padded) as usize;
        pixels.extend_from_slice(&mapped[start..start + unpadded as usize]);
    }
    drop(mapped);
    readback.unmap();

    write_png(Path::new(&opts.out), opts.width, opts.height, &pixels);
    println!(
        "wrote {} ({}x{}, {} ticks) in {:.0} ms",
        opts.out,
        opts.width,
        opts.height,
        opts.ticks,
        started.elapsed().as_secs_f32() * 1000.0
    );
}

fn write_png(path: &Path, width: u32, height: u32, rgba: &[u8]) {
    let file = std::fs::File::create(path).expect("failed to create output file");
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    // The surface format is sRGB, so the bytes coming back are already encoded.
    encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
    encoder
        .write_header()
        .expect("failed to write png header")
        .write_image_data(rgba)
        .expect("failed to write png data");
}
