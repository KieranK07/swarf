//! Screen rendering. One fullscreen triangle sampling the cell buffer directly.

use crate::camera::Camera;
use crate::gpu::Gpu;
use crate::sim::Sim;
use crate::world;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ViewUniform {
    size: [u32; 2],
    screen: [f32; 2],
    centre: [f32; 2],
    zoom: f32,
    brush_radius: f32,
    brush_pos: [f32; 2],
    brush_show: u32,
    _pad: u32,
}

pub struct Renderer {
    pipeline: wgpu::RenderPipeline,
    bind: wgpu::BindGroup,
    view: wgpu::Buffer,
}

impl Renderer {
    pub fn new(gpu: &Gpu, sim: &Sim) -> Self {
        let device = &gpu.device;

        let view = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view"),
            size: size_of::<ViewUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let storage = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("render bgl"),
            entries: &[
                storage(0),
                storage(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sim.cells.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: sim.materials.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: view.as_entire_binding() },
            ],
        });

        let module = gpu.shader("render", include_str!("../shaders/render.wgsl"));
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: gpu.config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self { pipeline, bind, view }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        camera: &Camera,
        brush_pos: [f32; 2],
        brush_radius: f32,
        brush_visible: bool,
    ) {
        gpu.queue.write_buffer(
            &self.view,
            0,
            bytemuck::bytes_of(&ViewUniform {
                size: [world::WIDTH, world::HEIGHT],
                screen: [gpu.config.width as f32, gpu.config.height as f32],
                centre: camera.centre,
                zoom: camera.zoom,
                brush_radius,
                brush_pos,
                brush_show: brush_visible as u32,
                _pad: 0,
            }),
        );

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("present"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    // The triangle covers every pixel, so there is nothing to
                    // clear -- discarding the previous contents is free.
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind, &[]);
        pass.draw(0..3, 0..1);
    }
}
