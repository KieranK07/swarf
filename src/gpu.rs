//! wgpu device, queue and surface setup.

use std::sync::Arc;
use winit::window::Window;

pub struct Gpu {
    /// Absent in headless mode, where rendering targets an offscreen texture.
    surface: Option<wgpu::Surface<'static>>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// Carries the render target's size and format. In headless mode it is
    /// synthesised, so the renderer needs no knowledge of which mode it is in.
    pub config: wgpu::SurfaceConfiguration,
    pub adapter_name: String,
}

impl Gpu {
    /// Shared device setup. `compatible` steers adapter selection towards a
    /// surface when there is one.
    fn device_for(
        instance: &wgpu::Instance,
        compatible: Option<&wgpu::Surface<'static>>,
    ) -> (wgpu::Adapter, wgpu::Device, wgpu::Queue, String) {
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: compatible,
            force_fallback_adapter: false,
            ..Default::default()
        }))
        .expect("no suitable GPU adapter");

        let info = adapter.get_info();
        let name = format!("{} ({:?})", info.name, info.backend);

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("swarf device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .expect("failed to create device");

        (adapter, device, queue, name)
    }

    fn instance() -> wgpu::Instance {
        // `InstanceDescriptor` carries a boxed display handle, so it has no
        // `Default`; build it through the constructor instead.
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::from_env().unwrap_or(wgpu::Backends::METAL);
        wgpu::Instance::new(desc)
    }

    /// Offscreen context for `--shot`. No window, no surface, no event loop.
    pub fn headless(width: u32, height: u32) -> Self {
        let instance = Self::instance();
        let (_adapter, device, queue, adapter_name) = Self::device_for(&instance, None);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: Self::HEADLESS_FORMAT,
            color_space: wgpu::SurfaceColorSpace::Srgb,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
        };

        Self { surface: None, device, queue, config, adapter_name }
    }

    pub const HEADLESS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

    pub fn surface(&self) -> &wgpu::Surface<'static> {
        self.surface.as_ref().expect("no surface in headless mode")
    }

    pub fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        let instance = Self::instance();

        let surface = instance
            .create_surface(window.clone())
            .expect("failed to create surface");

        let (adapter, device, queue, adapter_name) = Self::device_for(&instance, Some(&surface));

        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("surface is not supported by this adapter");

        // Prefer an sRGB surface and let the hardware do the encode. Material
        // colours are stored linear so blending and the emissive boost behave.
        let caps = surface.get_capabilities(&adapter);
        if let Some(srgb) = caps.formats.iter().copied().find(|f| f.is_srgb()) {
            config.format = srgb;
        }
        config.present_mode = wgpu::PresentMode::AutoVsync;
        config.desired_maximum_frame_latency = 2;

        surface.configure(&device, &config);

        Self { surface: Some(surface), device, queue, config, adapter_name }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        if let Some(surface) = &self.surface {
            surface.configure(&self.device, &self.config);
        }
    }

    /// Compile a shader, prepending the shared prelude. WGSL has no `#include`,
    /// so the concatenation *is* the include.
    pub fn shader(&self, label: &str, body: &str) -> wgpu::ShaderModule {
        const COMMON: &str = include_str!("../shaders/common.wgsl");
        self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(format!("{COMMON}\n{body}").into()),
        })
    }
}
