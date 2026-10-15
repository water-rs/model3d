//! Interactive viewer hosting `Model3D`'s `GpuContentView` in a winit window.
//!
//! ```sh
//! cargo run --example viewer -- --model assets/models/Fox.glb --clip Survey
//! ```
//!
//! The window hosts the content through [`GpuContentRenderer`]: the content
//! renders into the engine's retained offscreen layer (premultiplied linear
//! Display-P3) and `present` composites it into the swapchain — the same
//! presentation conversion a `WaterUI` backend applies. Input arrives as
//! [`SurfaceInputEvent`]s and rendering is event-driven: a redraw is
//! requested only when a watcher fires or a frame asks for another one.

mod shared;

use std::error::Error;
use std::sync::Arc;

use clap::Parser;
use nami::{Signal, binding};
use waterui_graphics::cherenkov::{Display, FrameTime};
use waterui_graphics::gpu::runtime::GpuContentRenderer;
use waterui_graphics::gpu::{GpuContentView, RedrawHandle};
use waterui_graphics::input::{
    Key, Modifiers, NamedKey, ScrollUnit, SurfaceInputEvent, SurfacePointerButton,
};
use waterui_graphics::{GpuRuntime, OffscreenSize, wgpu};
use waterui_model3d::{Model, Model3D, Playback};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::{Key as WinitKey, NamedKey as WinitNamedKey};
use winit::window::{Window, WindowId};

use shared::CommonArgs;

/// Host a glTF model in an interactive desktop window.
#[derive(Parser, Debug)]
#[command(version, about)]
struct ViewerArgs {
    #[command(flatten)]
    common: CommonArgs,
}

/// Events the event loop can be woken with.
enum Wake {
    /// A `RedrawHandle` waker fired — reactive state changed.
    Redraw,
}

struct Host {
    runtime: GpuRuntime,
    view: GpuContentView,
    /// Kept for validated clip cycling/seek — `select_clip`/`seek` reject
    /// handles minted by another asset.
    model: Model,
    playback: nami::Binding<Playback>,
    window: Option<Arc<Window>>,
    surface: Option<wgpu::Surface<'static>>,
    renderer: Option<GpuContentRenderer>,
    proxy: EventLoopProxy<Wake>,
    scale: f32,
    mods: Modifiers,
}

impl Host {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "display scale factors are a few units (1.0–4.0 on real monitors)"
    )]
    fn install_window(&mut self, event_loop: &ActiveEventLoop) -> Result<(), Box<dyn Error>> {
        let size = LogicalSize::new(960.0, 600.0);
        let window = Arc::new(
            event_loop.create_window(
                Window::default_attributes()
                    .with_title("waterui-model3d viewer")
                    .with_inner_size(size),
            )?,
        );
        let context = self.runtime.context();
        // The surface must live on the engine's own instance/device.
        let surface = context.instance().create_surface(window.clone())?;
        let caps = surface.get_capabilities(context.adapter());
        let format = caps
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .unwrap_or(caps.formats[0]);
        let size_px = window.inner_size();
        surface.configure(
            context.device(),
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: size_px.width.max(1),
                height: size_px.height.max(1),
                present_mode: wgpu::PresentMode::Fifo,
                color_space: wgpu::SurfaceColorSpace::Srgb,
                desired_maximum_frame_latency: 2,
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
            },
        );
        let proxy = self.proxy.clone();
        let redraw = RedrawHandle::new(move || {
            let _ = proxy.send_event(Wake::Redraw);
        });
        let size = OffscreenSize::try_from_pixels(size_px.width, size_px.height)
            .ok_or("zero-size window")?;
        self.renderer = Some(GpuContentRenderer::new(
            &self.runtime,
            self.runtime.context(),
            self.view.take_engine_content(),
            size,
            redraw,
        )?);
        self.scale = window.scale_factor() as f32;
        self.window = Some(window);
        self.surface = Some(surface);
        Ok(())
    }

    fn to_input(&self, event: &WindowEvent) -> Option<SurfaceInputEvent> {
        match event {
            WindowEvent::CursorMoved { position, .. } => Some(SurfaceInputEvent::PointerMove {
                position: waterui_graphics::draw::kurbo::Point::new(
                    position.x / f64::from(self.scale),
                    position.y / f64::from(self.scale),
                ),
            }),
            WindowEvent::MouseInput { state, button, .. } => {
                let button = match button {
                    MouseButton::Left => SurfacePointerButton::Primary,
                    MouseButton::Right => SurfacePointerButton::Secondary,
                    MouseButton::Middle => SurfacePointerButton::Middle,
                    MouseButton::Back => SurfacePointerButton::Back,
                    MouseButton::Forward => SurfacePointerButton::Forward,
                    MouseButton::Other(_) => return None,
                };
                Some(SurfaceInputEvent::PointerButton {
                    pressed: *state == ElementState::Pressed,
                    button,
                    position: waterui_graphics::draw::kurbo::Point::new(0.0, 0.0),
                })
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let (dx, dy, unit) = match delta {
                    MouseScrollDelta::LineDelta(x, y) => {
                        (f64::from(*x), f64::from(*y), ScrollUnit::Line)
                    }
                    MouseScrollDelta::PixelDelta(p) => (p.x, p.y, ScrollUnit::Pixel),
                };
                Some(SurfaceInputEvent::Scroll {
                    position: waterui_graphics::draw::kurbo::Point::new(0.0, 0.0),
                    delta_x: dx,
                    delta_y: dy,
                    unit,
                    finished: true,
                })
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key,
                        state,
                        repeat,
                        ..
                    },
                ..
            } => Some(SurfaceInputEvent::Key {
                pressed: *state == ElementState::Pressed,
                key: match logical_key {
                    WinitKey::Named(WinitNamedKey::ArrowLeft) => Key::Named(NamedKey::ArrowLeft),
                    WinitKey::Named(WinitNamedKey::ArrowRight) => Key::Named(NamedKey::ArrowRight),
                    WinitKey::Named(WinitNamedKey::ArrowUp) => Key::Named(NamedKey::ArrowUp),
                    WinitKey::Named(WinitNamedKey::ArrowDown) => Key::Named(NamedKey::ArrowDown),
                    WinitKey::Character(s) => Key::Character(s.to_string()),
                    _ => return None,
                },
                code: waterui_graphics::input::Code::Unidentified,
                modifiers: self.mods,
                repeat: *repeat,
            }),
            WindowEvent::Focused(focused) => Some(SurfaceInputEvent::Focus(*focused)),
            _ => None,
        }
    }

    fn on_modifiers(&mut self, state: winit::keyboard::ModifiersState) {
        self.mods = Modifiers::empty()
            | if state.shift_key() {
                Modifiers::SHIFT
            } else {
                Modifiers::empty()
            }
            | if state.control_key() {
                Modifiers::CONTROL
            } else {
                Modifiers::empty()
            }
            | if state.alt_key() {
                Modifiers::ALT
            } else {
                Modifiers::empty()
            }
            | if state.super_key() {
                Modifiers::META
            } else {
                Modifiers::empty()
            };
    }

    fn draw(&mut self) -> Result<(), Box<dyn Error>> {
        let (Some(renderer), Some(window), Some(surface)) =
            (self.renderer.as_mut(), &self.window, &self.surface)
        else {
            return Ok(());
        };
        // Per-frame UI hook: playhead write-back + autofit bookkeeping.
        self.view.frame();
        // The fork returns the frame as an enum — terminal states are not
        // errors, they ask the host to skip or reconfigure this frame.
        let output = match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                window.request_redraw();
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                return Err("swapchain lost — surface needs recreation".into());
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err("surface texture acquisition failed validation".into());
            }
        };
        renderer.present(
            &output.texture,
            Display {
                scale: f64::from(self.scale),
                headroom: 1.0,
            },
            FrameTime::now(),
        )?;
        window.pre_present_notify();
        // The fork presents through the queue (`SurfaceTexture::present` is
        // gone); dropping an unpresented texture discards the frame.
        self.runtime.context().queue().present(output);
        Ok(())
    }
}

impl ApplicationHandler<Wake> for Host {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none()
            && let Err(e) = self.install_window(event_loop)
        {
            tracing::error!("window setup failed: {e}");
            event_loop.exit();
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: Wake) {
        match event {
            Wake::Redraw => {
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match &event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
                return;
            }
            WindowEvent::Resized(size) => {
                if let Some(surface) = &self.surface {
                    let context = self.runtime.context();
                    surface.configure(
                        context.device(),
                        &wgpu::SurfaceConfiguration {
                            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                            format: surface.get_capabilities(context.adapter()).formats[0],
                            width: size.width.max(1),
                            height: size.height.max(1),
                            present_mode: wgpu::PresentMode::Fifo,
                            color_space: wgpu::SurfaceColorSpace::Srgb,
                            desired_maximum_frame_latency: 2,
                            alpha_mode: wgpu::CompositeAlphaMode::Auto,
                            view_formats: vec![],
                        },
                    );
                }
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
                return;
            }
            WindowEvent::RedrawRequested => {
                if let Err(e) = self.draw() {
                    tracing::error!("frame failed: {e}");
                }
                return;
            }
            WindowEvent::ModifiersChanged(m) => {
                self.on_modifiers(m.state());
                return;
            }
            _ => {}
        }
        if let WindowEvent::KeyboardInput {
            event:
                KeyEvent {
                    logical_key,
                    state: ElementState::Pressed,
                    ..
                },
            ..
        } = &event
        {
            let mut playback = self.playback.snapshot();
            match logical_key {
                // Space toggles play/pause — the component's renderer clock
                // advances; a fresh resume must not jump by the paused gap.
                WinitKey::Named(WinitNamedKey::Space) => {
                    if playback.is_playing() {
                        playback.pause();
                    } else if playback.clip().is_some() {
                        playback.play();
                    }
                    self.playback.set(playback);
                    return;
                }
                // C cycles clips on this model (asset-bound handles).
                WinitKey::Character(c) if c.eq_ignore_ascii_case("c") => {
                    let count = self.model.clip_count();
                    if count > 0 {
                        let next = playback.clip().map_or(0, |h| (h.index() + 1) % count);
                        playback
                            .select_clip(&self.model, self.model.clip_handle(next).unwrap())
                            .expect("clip handle minted from this model");
                        self.playback.set(playback);
                    }
                    return;
                }
                // Home seeks to t=0.
                WinitKey::Named(WinitNamedKey::Home) => {
                    playback
                        .seek(&self.model, std::time::Duration::ZERO)
                        .expect("seek within clip duration");
                    self.playback.set(playback);
                    return;
                }
                _ => {}
            }
        }
        if let Some(event) = self.to_input(&event) {
            self.view.input(&event);
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = ViewerArgs::parse();
    shared::init_tracing();

    let model = pollster::block_on(Model::load(args.common.model.clone()))?;
    let workload = shared::workload_stats(&std::fs::read(&args.common.model)?)?;
    tracing::info!(
        "loaded {}: {} primitives, {} clips",
        args.common.model.display(),
        workload.rendered_primitives,
        workload.animation_clips
    );

    let runtime = pollster::block_on(GpuRuntime::new())?;
    shared::require_hardware_adapter(runtime.context().adapter())?;
    tracing::info!(
        "adapter: {}",
        shared::adapter_report(runtime.context().adapter())
    );

    let aspect = 960.0f32 / 600.0;
    let camera = shared::camera(&model, &args.common, aspect)?;
    let playback = shared::playback(&model, args.common.clip.as_deref(), args.common.time, false)?;
    let playback_binding = binding(playback);

    let view = Model3D::new(model.clone())
        .camera(binding(camera))
        .playback(playback_binding.clone())
        .accessibility_label("3D model viewer")
        .into_surface();

    let event_loop = EventLoop::<Wake>::with_user_event().build()?;
    event_loop.set_control_flow(ControlFlow::Wait);
    let proxy = event_loop.create_proxy();
    let mut host = Host {
        runtime,
        view,
        model,
        playback: playback_binding,
        window: None,
        surface: None,
        renderer: None,
        proxy,
        scale: 1.0,
        mods: Modifiers::empty(),
    };
    event_loop.run_app(&mut host)?;
    Ok(())
}
