//! swarf -- a granular-material sandbox.

mod camera;
mod gpu;
mod materials;
mod render;
mod shot;
mod sim;
mod world;

#[cfg(test)]
mod tests;

use std::sync::Arc;
use std::time::{Duration, Instant};

use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use camera::Camera;
use gpu::Gpu;
use render::Renderer;
use sim::{Sim, Stroke};

/// Fixed simulation rate, in ticks per second. Decoupled from the display: a
/// 120 Hz screen renders twice per tick rather than simulating twice as fast,
/// so the physics runs at the same speed on every machine.
///
/// This constant sets the speed of *everything*. A 2x2 block can only ever move
/// material one cell per tick, so fall speed, avalanche speed and how fast
/// water levels are all exactly `SIM_HZ` cells per second. Running at display
/// rate makes sand crawl.
///
/// 240 Hz is four ticks per 60 Hz frame, which costs under 3 ms of GPU on an
/// M4 even with the world in motion -- and next to nothing once it settles,
/// because sleeping chunks are not dispatched at all. Paying only for what is
/// actually moving is what makes a rate this high affordable on a fanless
/// machine.
const SIM_HZ: u32 = 240;
const TICK: Duration = Duration::from_nanos(1_000_000_000 / SIM_HZ as u64);

/// If we fall further behind than this, drop the backlog instead of trying to
/// catch up. Catching up after a stall only makes the next frame later still.
const MAX_LAG: Duration = Duration::from_millis(250);

const PAN_CELLS_PER_SEC: f32 = 600.0;

struct State {
    window: Arc<Window>,
    gpu: Gpu,
    sim: Sim,
    renderer: Renderer,
    camera: Camera,

    cursor: [f32; 2],
    cursor_in_window: bool,
    stroke_from: Option<[f32; 2]>,
    painting: Option<u8>,
    space_pan: bool,
    middle_pan: bool,
    pan_keys: [bool; 4], // left, right, up, down
    material: u8,
    brush: f32,

    paused: bool,
    step_once: bool,

    accumulator: Duration,
    last_frame: Instant,
    seed: u32,

    frame_accum: Duration,
    frames: u32,
    title_timer: Duration,
}

impl State {
    fn new(window: Arc<Window>) -> Self {
        let gpu = Gpu::new(window.clone());
        let seed = seed_from_clock();
        let started = Instant::now();
        let sim = Sim::new(&gpu, seed, world::Scene::Terrain);
        let renderer = Renderer::new(&gpu, &sim);
        let camera = Camera::new(window.scale_factor() as f32);

        println!("swarf");
        println!("  gpu    {}", gpu.adapter_name);
        println!(
            "  world  {} x {} cells ({:.1} MB)",
            world::WIDTH,
            world::HEIGHT,
            (world::CELL_COUNT * 4) as f32 / 1e6
        );
        println!("  gen    {:.0} ms", started.elapsed().as_secs_f32() * 1000.0);
        println!();
        println!("  LMB paint   RMB erase   MMB/space+drag pan   scroll zoom");
        println!("  1-0 material   Q/E cycle   [ ] brush   P pause   N step   R regenerate");

        Self {
            window,
            gpu,
            sim,
            renderer,
            camera,
            cursor: [0.0, 0.0],
            cursor_in_window: false,
            stroke_from: None,
            painting: None,
            space_pan: false,
            middle_pan: false,
            pan_keys: [false; 4],
            material: materials::HOTBAR[0],
            brush: 12.0,
            paused: false,
            step_once: false,
            accumulator: Duration::ZERO,
            last_frame: Instant::now(),
            seed,
            frame_accum: Duration::ZERO,
            frames: 0,
            title_timer: Duration::ZERO,
        }
    }

    fn surface_size(&self) -> [f32; 2] {
        [self.gpu.config.width as f32, self.gpu.config.height as f32]
    }

    fn cursor_world(&self) -> [f32; 2] {
        self.camera.screen_to_world(self.cursor, self.surface_size())
    }

    fn frame(&mut self) {
        let now = Instant::now();
        let dt = now - self.last_frame;
        self.last_frame = now;

        // --- camera ---------------------------------------------------------
        let pan = PAN_CELLS_PER_SEC * dt.as_secs_f32();
        let dx = (self.pan_keys[1] as i32 - self.pan_keys[0] as i32) as f32 * pan;
        let dy = (self.pan_keys[3] as i32 - self.pan_keys[2] as i32) as f32 * pan;
        if dx != 0.0 || dy != 0.0 {
            self.camera.centre[0] += dx;
            self.camera.centre[1] += dy;
            self.camera.clamp();
        }

        // --- how many sim ticks this frame ----------------------------------
        let mut ticks = 0usize;
        if self.paused {
            if self.step_once {
                self.step_once = false;
                ticks = 1;
            }
        } else {
            self.accumulator += dt.min(MAX_LAG);
            ticks = (self.accumulator.as_nanos() / TICK.as_nanos()) as usize;
            if ticks > sim::MAX_TICKS_PER_FRAME {
                ticks = sim::MAX_TICKS_PER_FRAME;
                self.accumulator = Duration::ZERO;
            } else {
                self.accumulator -= TICK * ticks as u32;
            }
        }

        // --- brush ----------------------------------------------------------
        let world_cursor = self.cursor_world();
        let stroke = self.painting.map(|material| {
            let from = self.stroke_from.unwrap_or(world_cursor);
            Stroke { from, to: world_cursor, radius: self.brush, material }
        });
        if self.painting.is_some() {
            self.stroke_from = Some(world_cursor);
        }

        // --- submit ---------------------------------------------------------
        use wgpu::CurrentSurfaceTexture as Acquired;
        let frame = match self.gpu.surface().get_current_texture() {
            // Suboptimal still presents; it just wants reconfiguring eventually,
            // which the next resize will do anyway.
            Acquired::Success(f) | Acquired::Suboptimal(f) => f,
            Acquired::Outdated | Acquired::Lost => {
                let (w, h) = (self.gpu.config.width, self.gpu.config.height);
                self.gpu.resize(w, h);
                return;
            }
            // Occluded means nothing is visible, so skipping the frame is the
            // correct (and coolest-running) response.
            Acquired::Timeout | Acquired::Occluded => return,
            Acquired::Validation => {
                eprintln!("surface acquire failed validation");
                return;
            }
        };

        let target = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });

        if let Some(s) = &stroke {
            self.sim.paint(&self.gpu.queue, &mut encoder, s);
        }

        if ticks > 0 {
            self.sim.prepare(&self.gpu.queue, ticks);
            for slot in 0..ticks {
                self.sim.encode_tick(&mut encoder, slot);
            }
        }

        self.renderer.draw(
            &self.gpu,
            &mut encoder,
            &target,
            &self.camera,
            world_cursor,
            self.brush,
            self.cursor_in_window,
        );

        self.gpu.queue.submit([encoder.finish()]);
        // wgpu 30 presents through the queue rather than the texture.
        self.gpu.queue.present(frame);

        self.sim.tick = self.sim.tick.wrapping_add(ticks as u32);

        // --- stats ----------------------------------------------------------
        self.frame_accum += dt;
        self.frames += 1;
        self.title_timer += dt;
        if self.title_timer >= Duration::from_millis(250) {
            let fps = self.frames as f32 / self.frame_accum.as_secs_f32().max(1e-6);
            let ms = self.frame_accum.as_secs_f32() * 1000.0 / self.frames as f32;
            self.window.set_title(&format!(
                "swarf  |  {}  brush {:.0}  |  {:.0}x  |  {fps:.0} fps  {ms:.2} ms{}",
                materials::name(self.material),
                self.brush,
                self.camera.zoom,
                if self.paused { "  [paused]" } else { "" },
            ));
            self.frame_accum = Duration::ZERO;
            self.frames = 0;
            self.title_timer = Duration::ZERO;
        }
    }

    fn key(&mut self, code: KeyCode, pressed: bool) {
        match code {
            KeyCode::KeyA | KeyCode::ArrowLeft => self.pan_keys[0] = pressed,
            KeyCode::KeyD | KeyCode::ArrowRight => self.pan_keys[1] = pressed,
            KeyCode::KeyW | KeyCode::ArrowUp => self.pan_keys[2] = pressed,
            KeyCode::KeyS | KeyCode::ArrowDown => self.pan_keys[3] = pressed,
            KeyCode::Space => self.space_pan = pressed,
            _ => {}
        }
        if !pressed {
            return;
        }

        let hotbar = [
            KeyCode::Digit1, KeyCode::Digit2, KeyCode::Digit3, KeyCode::Digit4, KeyCode::Digit5,
            KeyCode::Digit6, KeyCode::Digit7, KeyCode::Digit8, KeyCode::Digit9, KeyCode::Digit0,
        ];
        if let Some(slot) = hotbar.iter().position(|k| *k == code) {
            self.material = materials::HOTBAR[slot];
            return;
        }

        let count = materials::MATERIALS.len() as u8;
        match code {
            KeyCode::KeyE => self.material = (self.material + 1) % count,
            KeyCode::KeyQ => self.material = (self.material + count - 1) % count,
            KeyCode::BracketLeft => self.brush = (self.brush / 1.4).max(1.0),
            KeyCode::BracketRight => self.brush = (self.brush * 1.4).min(400.0),
            KeyCode::KeyP => self.paused = !self.paused,
            KeyCode::KeyN => self.step_once = true,
            KeyCode::KeyR => {
                self.seed = self.seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                self.sim.reset(&self.gpu.queue, self.seed, world::Scene::Terrain);
            }
            _ => {}
        }
    }
}

fn seed_from_clock() -> u32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos() ^ d.as_secs() as u32)
        .unwrap_or(0x5EED)
}

#[derive(Default)]
struct App {
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("swarf")
            .with_inner_size(LogicalSize::new(1600.0, 950.0));
        let window = Arc::new(event_loop.create_window(attrs).expect("failed to create window"));
        self.state = Some(State::new(window));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else { return };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => state.gpu.resize(size.width, size.height),

            WindowEvent::RedrawRequested => state.frame(),

            WindowEvent::CursorMoved { position, .. } => {
                let next = [position.x as f32, position.y as f32];
                if state.space_pan || state.middle_pan {
                    state
                        .camera
                        .pan_pixels(next[0] - state.cursor[0], next[1] - state.cursor[1]);
                    state.camera.clamp();
                }
                state.cursor = next;
                state.cursor_in_window = true;
            }

            WindowEvent::CursorLeft { .. } => state.cursor_in_window = false,

            WindowEvent::MouseWheel { delta, .. } => {
                let steps = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                };
                let surface = state.surface_size();
                state.camera.zoom_at(state.cursor, surface, steps);
                state.camera.clamp();
            }

            WindowEvent::MouseInput { state: element, button, .. } => {
                let down = element == ElementState::Pressed;
                match button {
                    MouseButton::Left => {
                        state.painting = if down { Some(state.material) } else { None };
                        state.stroke_from = None;
                    }
                    MouseButton::Right => {
                        state.painting =
                            if down { Some(materials::Material::AIR) } else { None };
                        state.stroke_from = None;
                    }
                    MouseButton::Middle => state.middle_pan = down,
                    _ => {}
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    if code == KeyCode::Escape {
                        event_loop.exit();
                    }
                    state.key(code, event.state == ElementState::Pressed);
                }
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}

const USAGE: &str = "\
swarf -- granular material sandbox

  swarf                              run interactively
  swarf --shot FILE [options]        render one frame headless and exit

shot options:
  --ticks N          simulate N ticks before capturing   (default 0)
  --size WxH         output resolution                   (default 1280x720)
  --zoom Z           screen pixels per cell              (default 1)
  --centre X,Y       world cell at the centre of frame
  --seed S           world seed                          (default random)
  --scene NAME       terrain | lab (bare rig for judging physics)
  --drop MAT@X,Y,R   stamp a disc of material MAT before simulating (repeatable)
  --every K          also write a frame every K ticks, as FILE-0000.png, FILE-0001.png, ...
";

/// Minimal flag parsing. A CLI arg crate would be a dependency earning its keep
/// only in `--shot`, which is a development tool.
fn parse_shot(args: &[String]) -> Option<shot::ShotOptions> {
    if !args.iter().any(|a| a == "--shot") {
        if args.iter().any(|a| a == "--help" || a == "-h") {
            print!("{USAGE}");
            std::process::exit(0);
        }
        return None;
    }

    let mut opts = shot::ShotOptions {
        width: 1280,
        height: 720,
        ticks: 0,
        zoom: 1.0,
        centre: None,
        seed: seed_from_clock(),
        strokes: Vec::new(),
        out: "shot.png".into(),
        scene: world::Scene::Terrain,
        every: None,
    };

    /// Consume the value that follows a flag.
    fn value(args: &[String], i: &mut usize) -> String {
        *i += 1;
        args.get(*i)
            .unwrap_or_else(|| panic!("{} needs a value", args[*i - 1]))
            .clone()
    }

    let mut i = 0;
    while i < args.len() {
        macro_rules! value {
            () => {
                value(args, &mut i)
            };
        }
        match args[i].as_str() {
            "--shot" => opts.out = value!(),
            "--ticks" => opts.ticks = value!().parse().expect("--ticks expects an integer"),
            "--zoom" => opts.zoom = value!().parse().expect("--zoom expects a number"),
            "--every" => opts.every = Some(value!().parse().expect("--every expects an integer")),
            "--seed" => opts.seed = value!().parse().expect("--seed expects an integer"),
            "--scene" => {
                opts.scene = match value!().as_str() {
                    "terrain" => world::Scene::Terrain,
                    "lab" => world::Scene::Lab,
                    other => panic!("unknown scene {other:?} (terrain | lab)"),
                }
            }
            "--size" => {
                let v = value!();
                let (w, h) = v.split_once('x').expect("--size expects WxH");
                opts.width = w.parse().expect("bad width");
                opts.height = h.parse().expect("bad height");
            }
            "--centre" | "--center" => {
                let v = value!();
                let (x, y) = v.split_once(',').expect("--centre expects X,Y");
                opts.centre = Some([x.parse().expect("bad x"), y.parse().expect("bad y")]);
            }
            "--drop" => {
                let v = value!();
                let (mat, rest) = v.split_once('@').expect("--drop expects MAT@X,Y,R");
                let parts: Vec<&str> = rest.split(',').collect();
                assert_eq!(parts.len(), 3, "--drop expects MAT@X,Y,R");
                let at = [
                    parts[0].parse().expect("bad x"),
                    parts[1].parse().expect("bad y"),
                ];
                opts.strokes.push(Stroke {
                    from: at,
                    to: at,
                    radius: parts[2].parse().expect("bad radius"),
                    material: parse_material(mat),
                });
            }
            other => panic!("unknown argument {other}\n\n{USAGE}"),
        }
        i += 1;
    }
    Some(opts)
}

/// Accept either a material index or a (case-insensitive) name.
fn parse_material(spec: &str) -> u8 {
    if let Ok(id) = spec.parse::<u8>() {
        return id;
    }
    materials::MATERIALS
        .iter()
        .position(|m| m.name.eq_ignore_ascii_case(spec))
        .unwrap_or_else(|| panic!("unknown material {spec:?}")) as u8
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(opts) = parse_shot(&args) {
        shot::capture(opts);
        return;
    }

    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::default()).expect("event loop failed");
}
