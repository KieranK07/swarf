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
use crate::sim::{Sim, Stroke};

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
    /// Also write a frame every this many ticks, numbered into the file name.
    pub every: Option<u32>,
}

/// `out.png` -> `out-0007.png`.
fn frame_path(out: &str, frame: u32) -> String {
    let path = Path::new(out);
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("shot");
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("png");
    path.with_file_name(format!("{stem}-{frame:04}.{ext}"))
        .to_string_lossy()
        .into_owned()
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
    sim.paint_all(&gpu, &opts.strokes);

    let draw = |out: &str| {
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

        write_png(Path::new(out), opts.width, opts.height, &pixels);
    };

    match opts.every {
        // One continuous run, so consecutive frames show the same grains
        // moving rather than independent runs that each diverge slightly.
        Some(every) => {
            let every = every.max(1);
            let mut frame = 0;
            loop {
                draw(&frame_path(&opts.out, frame));
                frame += 1;
                if sim.tick >= opts.ticks {
                    break;
                }
                sim.run(&gpu, every.min(opts.ticks - sim.tick));
            }
            print!("wrote {frame} frames of ");
        }
        None => {
            sim.run(&gpu, opts.ticks);
            draw(&opts.out);
            print!("wrote ");
        }
    }
    println!(
        "{} ({}x{}, {} ticks) in {:.0} ms",
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
