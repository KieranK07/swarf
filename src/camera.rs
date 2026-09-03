//! Pan/zoom camera. Works entirely in cell coordinates; `zoom` is the number
//! of screen pixels one cell occupies.

use crate::world;

pub struct Camera {
    pub centre: [f32; 2],
    pub zoom: f32,
}

/// Below 1:1 the four-tap filter in the fragment shader stops keeping up, and
/// above 32:1 there is nothing left to see.
const MIN_ZOOM: f32 = 0.25;
const MAX_ZOOM: f32 = 32.0;

impl Camera {
    pub fn new(scale_factor: f32) -> Self {
        Self {
            // Start looking at the surface, not the middle of the rock.
            centre: [world::WIDTH as f32 * 0.5, 620.0],
            zoom: scale_factor.max(1.0),
        }
    }

    pub fn screen_to_world(&self, screen: [f32; 2], surface: [f32; 2]) -> [f32; 2] {
        [
            self.centre[0] + (screen[0] - surface[0] * 0.5) / self.zoom,
            self.centre[1] + (screen[1] - surface[1] * 0.5) / self.zoom,
        ]
    }

    /// Zoom about a fixed screen point, so the cell under the cursor stays put.
    pub fn zoom_at(&mut self, screen: [f32; 2], surface: [f32; 2], steps: f32) {
        let before = self.screen_to_world(screen, surface);
        self.zoom = (self.zoom * 1.15f32.powf(steps)).clamp(MIN_ZOOM, MAX_ZOOM);
        let after = self.screen_to_world(screen, surface);
        self.centre[0] += before[0] - after[0];
        self.centre[1] += before[1] - after[1];
    }

    pub fn pan_pixels(&mut self, dx: f32, dy: f32) {
        self.centre[0] -= dx / self.zoom;
        self.centre[1] -= dy / self.zoom;
    }

    /// Keep the view over the world. Clamping the centre rather than the edges
    /// means small worlds (or heavy zoom-out) stay centred instead of jamming
    /// into a corner.
    pub fn clamp(&mut self) {
        self.centre[0] = self.centre[0].clamp(0.0, world::WIDTH as f32);
        self.centre[1] = self.centre[1].clamp(0.0, world::HEIGHT as f32);
    }
}
