//! [`Model3D`]: the `WaterUI` view, its reactive inputs, and its input handling.
//!
//! Architecture: all reactive inputs (model, camera, playback, lighting,
//! background) are snapshotted into an immutable [`RenderInputs`] published
//! through `ArcSwap` to the render-thread-owned [`ModelContent`]
//! ([`GpuContent`]). Watchers live UI-side in the view's input/frame
//! closures; a published [`RedrawHandle`] slot (`ArcSwapOption`) lets them
//! wake the host without owning it. The animation clock is renderer-owned:
//! the UI publishes the authored [`Playback`] plus a revision, the renderer
//! advances by `Frame::delta`, and per-frame feedback writes the playhead
//! back into the binding only while the revision is still current.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use arc_swap::{ArcSwap, ArcSwapOption};
use glam::Vec3;
use nami::Signal;
use nami::signal::IntoComputed;
use waterui_core::{Binding, Computed, Environment, View};
use waterui_graphics::WorkingColor;
use waterui_graphics::color::working;
use waterui_graphics::color::{Srgb, srgb_to_linear};
use waterui_graphics::draw::kurbo::Point;
use waterui_graphics::gpu::{Context, Frame, GpuContent, GpuContentView, RedrawHandle};
use waterui_graphics::input::{
    Key, Modifiers, NamedKey, ScrollUnit, SurfaceInputEvent, SurfacePointerButton,
};

use crate::camera::OrbitCamera;
use crate::model::Model;
use crate::playback::Playback;
use crate::render::{FrameParams, Renderer};

// `Model`, `Lighting` and `OrbitCamera`/`Playback` are plain values in
// signals — constant `Signal` impls let them flow through `IntoComputed`.
nami::impl_constant!(Model);
nami::impl_constant!(Lighting);

/// Which pointer/keyboard gestures drive the camera.
///
/// A flags set — combine the associated constants with `|`:
/// `Interaction::ORBIT | Interaction::ZOOM`. [`Interaction::ALL`] (the
/// default) enables every gesture; [`Interaction::NONE`] makes the view
/// passive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Interaction(u8);

impl Interaction {
    /// Primary-button drag orbits.
    pub const ORBIT: Self = Self(1 << 0);
    /// Secondary/middle-button drag pans.
    pub const PAN: Self = Self(1 << 1);
    /// Wheel/trackpad scroll zooms.
    pub const ZOOM: Self = Self(1 << 2);
    /// Arrow-key orbit, `+`/`-` zoom, `R` reset.
    pub const KEYBOARD: Self = Self(1 << 3);
    /// Every gesture enabled — the default.
    pub const ALL: Self = Self(Self::ORBIT.0 | Self::PAN.0 | Self::ZOOM.0 | Self::KEYBOARD.0);
    /// No gestures: a passive view.
    pub const NONE: Self = Self(0);

    /// Whether every gesture in `flag` is enabled.
    #[must_use]
    pub const fn contains(self, flag: Self) -> bool {
        self.0 & flag.0 == flag.0
    }
}

impl std::ops::BitOr for Interaction {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for Interaction {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl Default for Interaction {
    fn default() -> Self {
        Self::ALL
    }
}

/// Lighting/exposure controls for the viewer, in sRGB authoring colors —
/// conversion to linear radiance happens at the rendering boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lighting {
    /// Direction the key light shines *toward*, in model space.
    pub light_direction: Vec3,
    /// Key light color (sRGB).
    pub light_color: Srgb,
    /// Key light intensity.
    pub light_intensity: f32,
    /// Fill light direction; `Vec3::ZERO` disables it.
    pub fill_direction: Vec3,
    /// Fill light color (sRGB).
    pub fill_color: Srgb,
    /// Fill light intensity.
    pub fill_intensity: f32,
    /// Ambient light color (sRGB).
    pub ambient_color: Srgb,
    /// Ambient intensity.
    pub ambient_intensity: f32,
    /// Linear exposure multiplier applied after shading.
    pub exposure: f32,
}

impl Default for Lighting {
    fn default() -> Self {
        Self {
            light_direction: Vec3::new(-0.4, -1.0, -0.6).normalize(),
            light_color: Srgb::WHITE,
            light_intensity: 3.0,
            fill_direction: Vec3::ZERO,
            fill_color: Srgb::WHITE,
            fill_intensity: 0.0,
            ambient_color: Srgb::WHITE,
            ambient_intensity: 0.4,
            exposure: 1.0,
        }
    }
}

/// Immutable UI→renderer snapshot. Published whole through `ArcSwap`.
#[derive(Debug, Clone)]
struct RenderInputs {
    model: Model,
    camera: OrbitCamera,
    /// Authored playback (clip selection, play state, user seeks).
    playback: Playback,
    /// Bumped on every *user* playback write; renderer clock resets on change.
    authored_revision: u64,
    /// `true` while the camera binding still holds auto-fitted values — the
    /// renderer then fits the model's bounds at the real viewport aspect.
    auto_camera: bool,
    lighting: Lighting,
    /// Linear Display-P3 straight-alpha clear color.
    clear: [f32; 4],
}

/// Shared UI↔renderer channels and flags built once at surface creation:
/// the published input snapshots, the wake slot, the feedback channel, the
/// latest aspect, and the write-back bookkeeping cells the watchers and
/// frame hook share.
#[derive(Clone)]
struct SurfaceShared {
    inputs: Arc<ArcSwap<RenderInputs>>,
    redraw_slot: Arc<ArcSwapOption<RedrawHandle>>,
    feedback: Arc<ArcSwapOption<Feedback>>,
    aspect_bits: Arc<AtomicU32>,
    /// Bumped on every *user* playback write; renderer clock resets on
    /// change and stale feedback revisions are dropped.
    authored_revision: Rc<Cell<u64>>,
    /// `true` while a feedback write-back is in flight — watchers must not
    /// treat it as authored input.
    suppress: Rc<Cell<bool>>,
    /// `true` while the camera binding holds auto-fitted values.
    auto_camera: Rc<Cell<bool>>,
}

/// Render→UI feedback: the playback value advanced by frame deltas for
/// `revision`, plus the *effective* camera and viewport the frame used. A
/// stale `revision` (older than the newest authored write) is discarded
/// UI-side, so renderer time never overwrites a user's seek; the camera
/// write-back lets drags start from the fitted pose without a wrong first
/// frame.
#[derive(Debug, Clone)]
struct Feedback {
    revision: u64,
    playback: Playback,
    camera: OrbitCamera,
    /// Logical viewport height (physical pixels / scale), for pan math.
    logical_height: f32,
    aspect: f32,
}

/// The render-thread-owned [`GpuContent`]. Everything framework-reactive is
/// behind the shared atomics; the struct itself is plain `Send`.
struct ModelContent {
    inputs: Arc<ArcSwap<RenderInputs>>,
    redraw_slot: Arc<ArcSwapOption<RedrawHandle>>,
    feedback: Arc<ArcSwapOption<Feedback>>,
    /// Latest viewport aspect (width/height), for UI-side camera fitting.
    aspect_bits: Arc<AtomicU32>,
    renderer: Option<Renderer>,
    /// Renderer-owned clock: latest authored playback, advanced by
    /// `Frame::delta` while playing.
    playback: Playback,
    seen_revision: u64,
    msaa_samples: u32,
}

impl std::fmt::Debug for ModelContent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelContent").finish_non_exhaustive()
    }
}

impl GpuContent for ModelContent {
    fn setup(&mut self, gpu: &Context<'_>) {
        // Install the wake handle for UI watchers, then (re)build every
        // device-bound resource. A device loss lands here again with fresh
        // handles; the slot is *replaced*, never accumulated.
        self.redraw_slot.store(Some(Arc::new(gpu.redraw.clone())));
        let mut renderer = Renderer::new(gpu.device, gpu.queue, gpu.format, self.msaa_samples);
        let model = self.inputs.load_full().model.clone();
        renderer.upload(&model);
        self.renderer = Some(renderer);
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "framebuffer dimensions are bounded by the surface — far below 2^24 texels"
    )]
    fn render(&mut self, frame: &mut Frame<'_>) {
        // Snapshot first; a whole-`self` immutable borrow cannot coexist
        // with the mutable renderer borrow below.
        let snap = self.inputs.load_full();
        let renderer = self
            .renderer
            .as_mut()
            .expect("GpuContent::setup must run before render");

        if !renderer.contains(&snap.model) {
            renderer.upload(&snap.model);
        }

        let aspect = frame.width as f32 / frame.height.max(1) as f32;
        self.aspect_bits.store(aspect.to_bits(), Ordering::Relaxed);

        // Clock: an authored revision change adopts the published state
        // verbatim (advance zero this frame — a resume must not jump by the
        // paused gap); otherwise the clock advances by the frame delta only
        // while playing.
        if snap.authored_revision != self.seen_revision {
            self.playback = snap.playback.clone();
            self.seen_revision = snap.authored_revision;
        } else if let Some(handle) = self.playback.clip().cloned() {
            let clip = snap
                .model
                .clip(&handle)
                .expect("playback clip must resolve against the uploaded model");
            self.playback.advance(frame.delta, clip.duration());
        }

        // Effective camera: while auto-fit is active the renderer fits the
        // model at this frame's real aspect — the first frame is correct.
        let camera = if snap.auto_camera {
            OrbitCamera::fit(&snap.model.bounds(), aspect).unwrap_or(snap.camera)
        } else {
            snap.camera
        };
        self.feedback.store(Some(Arc::new(Feedback {
            revision: self.seen_revision,
            playback: self.playback.clone(),
            camera,
            logical_height: frame.height as f32 / frame.scale.max(1e-6),
            aspect,
        })));
        let camera = &camera;
        let lighting = snap.lighting;
        let lin = |c: Srgb| {
            Vec3::new(
                srgb_to_linear(c.red),
                srgb_to_linear(c.green),
                srgb_to_linear(c.blue),
            )
        };
        let params = FrameParams {
            view: camera.view(),
            proj: camera.projection(aspect),
            camera_pos: camera.eye(),
            light_dir: lighting.light_direction.normalize_or(Vec3::NEG_Y),
            light_color: lin(lighting.light_color),
            light_intensity: lighting.light_intensity,
            fill_dir: lighting.fill_direction,
            fill_color: lin(lighting.fill_color) * lighting.fill_intensity,
            ambient: {
                let a = lin(lighting.ambient_color);
                [a.x, a.y, a.z, lighting.ambient_intensity]
            },
            exposure: lighting.exposure,
            clear: snap.clear,
        };
        let mut encoder = frame
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("model3d frame"),
            });
        renderer
            .draw(
                &mut encoder,
                frame.view,
                (frame.width, frame.height),
                &params,
                &self.playback,
            )
            .expect("playback clip must resolve against the uploaded model");
        frame.queue.submit([encoder.finish()]);

        // Keep frames coming only while the clip is actually playing.
        if self.playback.is_playing() && self.playback.clip().is_some() {
            frame.request_redraw();
        }
    }

    fn is_opaque(&self) -> bool {
        // Default background is transparent; the view must composite.
        false
    }
}

/// A glTF 3D model viewer.
///
/// `Model3D` is a `WaterUI` [`View`] hosting a [`GpuContentView`]. The model is
/// a signal: swapping the [`Binding`]/[`Computed`] value uploads the new
/// asset once and redraws — the view tree is never rebuilt. Camera and
/// playback are explicit `Binding`s the application owns; pointer and
/// keyboard input write back into them.
///
/// ```no_run
/// use waterui_model3d::{Model, Model3D, OrbitCamera, Playback};
///
/// # async fn example() -> Result<(), waterui_model3d::LoadError> {
/// let model = Model::load("assets/model.glb").await?;
/// let camera = nami::binding(
///     OrbitCamera::fit(&model.bounds(), 16.0 / 9.0).expect("model has finite bounds"),
/// );
/// let playback = nami::binding(Playback::new());
/// let view = Model3D::new(model).camera(camera).playback(playback);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct Model3D {
    model: Computed<Model>,
    camera: Binding<OrbitCamera>,
    playback: Binding<Playback>,
    camera_supplied: bool,
    background: Computed<WorkingColor>,
    lighting: Computed<Lighting>,
    interaction: Interaction,
    accessibility_label: String,
    msaa_samples: u32,
}

impl Model3D {
    /// A viewer for `model` — a loaded [`Model`], a `Binding<Model>`, or a
    /// `Computed<Model>`; all are accepted via [`IntoComputed`].
    pub fn new(model: impl IntoComputed<Model>) -> Self {
        Self {
            model: model.into_computed(),
            camera: nami::binding(OrbitCamera::new()),
            playback: nami::binding(Playback::new()),
            camera_supplied: false,
            background: nami::constant(working::from_linear_srgb([0.0; 3], 0.0)).into_computed(),
            lighting: nami::constant(Lighting::default()).into_computed(),
            interaction: Interaction::default(),
            accessibility_label: "3D model".to_owned(),
            msaa_samples: 4,
        }
    }

    /// Supplies the camera binding the view orbits/pans/zooms into.
    ///
    /// A supplied camera is authoritative: model swaps do not re-fit it.
    /// When omitted, the view auto-fits each new model's bounds and each
    /// viewport change while the user has not touched the camera.
    #[must_use]
    pub fn camera(mut self, camera: Binding<OrbitCamera>) -> Self {
        self.camera = camera;
        self.camera_supplied = true;
        self
    }

    /// Supplies the playback binding (clip selection, play/pause, loop,
    /// seek).
    #[must_use]
    pub fn playback(mut self, playback: Binding<Playback>) -> Self {
        self.playback = playback;
        self
    }

    /// Background cleared behind the model, as a linear-P3
    /// [`WorkingColor`] (straight alpha; default transparent). Author with
    /// `waterui_graphics::color::working::from_linear_srgb` or resolve a
    /// [`waterui_graphics::Color`] and feed its components.
    #[must_use]
    pub fn background(mut self, color: impl IntoComputed<WorkingColor>) -> Self {
        self.background = color.into_computed();
        self
    }

    /// Lighting and exposure parameters (reactive).
    #[must_use]
    pub fn lighting(mut self, lighting: impl IntoComputed<Lighting>) -> Self {
        self.lighting = lighting.into_computed();
        self
    }

    /// Which camera gestures are enabled.
    #[must_use]
    pub const fn interaction(mut self, interaction: Interaction) -> Self {
        self.interaction = interaction;
        self
    }

    /// MSAA sample count for the model pass (1 disables).
    ///
    /// # Panics
    /// When `samples` is not 1, 2, 4 or 8 — wgpu's supported MSAA counts.
    #[must_use]
    pub fn msaa_samples(mut self, samples: u32) -> Self {
        assert!(
            matches!(samples, 1 | 2 | 4 | 8),
            "MSAA sample count must be 1, 2, 4 or 8"
        );
        self.msaa_samples = samples;
        self
    }

    /// Accessibility label describing the view (static — the GPU layer has
    /// no reactive accessibility channel).
    #[must_use]
    pub fn accessibility_label(mut self, label: impl Into<String>) -> Self {
        self.accessibility_label = label.into();
        self
    }

    /// Consumes the view into the [`GpuContentView`] a GPU backend hosts.
    ///
    /// Watches on the reactive inputs are installed here once — they publish
    /// snapshots and wake the host through the slot the content installs at
    /// `setup`; their guards are retained by the returned view's input and
    /// frame closures for the view's whole lifetime.
    #[must_use]
    pub fn into_surface(self) -> GpuContentView {
        let shared = SurfaceShared {
            inputs: Arc::new(ArcSwap::from_pointee(self.render_inputs())),
            redraw_slot: Arc::new(ArcSwapOption::empty()),
            feedback: Arc::new(ArcSwapOption::empty()),
            aspect_bits: Arc::new(AtomicU32::new(1.0f32.to_bits())),
            authored_revision: Rc::new(Cell::new(0)),
            suppress: Rc::new(Cell::new(false)),
            auto_camera: Rc::new(Cell::new(!self.camera_supplied)),
        };

        let content = ModelContent {
            inputs: shared.inputs.clone(),
            redraw_slot: shared.redraw_slot.clone(),
            feedback: shared.feedback.clone(),
            aspect_bits: shared.aspect_bits.clone(),
            renderer: None,
            playback: self.playback.snapshot(),
            seen_revision: 0,
            msaa_samples: self.msaa_samples,
        };

        let guards = self.install_watchers(&shared);
        let frame_hook = self.frame_hook(&shared, guards);
        let input_handler = self.input_handler(&shared);

        let label = self.accessibility_label.clone();
        GpuContentView::new(content)
            .on_input(input_handler)
            .on_frame(frame_hook)
            .labeled(label)
    }

    /// The immutable UI→renderer snapshot the watchers keep publishing.
    fn render_inputs(&self) -> RenderInputs {
        RenderInputs {
            model: self.model.snapshot(),
            camera: self.camera.snapshot(),
            playback: self.playback.snapshot(),
            authored_revision: 0,
            auto_camera: !self.camera_supplied,
            lighting: self.lighting.snapshot(),
            clear: {
                let c = self.background.snapshot().components;
                [c[0], c[1], c[2], c[3]]
            },
        }
    }

    /// Installs the publish-and-wake watchers on every reactive input. The
    /// returned guards must live as long as the view — the frame hook owns
    /// them.
    fn install_watchers(&self, shared: &SurfaceShared) -> Vec<nami::watcher::BoxWatcherGuard> {
        let mut guards = Vec::new();
        {
            let inputs = shared.inputs.clone();
            let slot = shared.redraw_slot.clone();
            guards.push(self.model.watch(move |ctx| {
                let model = ctx.value().clone();
                publish_inputs(&inputs, &slot, |i| i.model = model.clone());
            }));
        }
        {
            let inputs = shared.inputs.clone();
            let slot = shared.redraw_slot.clone();
            let auto_camera = shared.auto_camera.clone();
            let suppress_camera = shared.suppress.clone();
            guards.push(self.camera.watch(move |ctx| {
                if suppress_camera.get() {
                    // Feedback write-back of the renderer's fitted pose — not
                    // authored input: publish nothing, never wake.
                    return;
                }
                auto_camera.set(false);
                let camera = *ctx.value();
                publish_inputs(&inputs, &slot, |i| {
                    i.camera = camera;
                    i.auto_camera = false;
                });
            }));
        }
        {
            let inputs = shared.inputs.clone();
            let slot = shared.redraw_slot.clone();
            let authored_revision = shared.authored_revision.clone();
            let suppress = shared.suppress.clone();
            guards.push(self.playback.watch(move |ctx| {
                if suppress.get() {
                    // Feedback write-back is not authored input: the
                    // renderer owns the clock while the revision is
                    // unchanged, so publish nothing and never wake — a paused
                    // or static scene must schedule no frames.
                    return;
                }
                authored_revision.set(authored_revision.get() + 1);
                let playback = ctx.value().clone();
                let rev = authored_revision.get();
                publish_inputs(&inputs, &slot, |i| {
                    i.playback = playback.clone();
                    i.authored_revision = rev;
                });
            }));
        }
        {
            let inputs = shared.inputs.clone();
            let slot = shared.redraw_slot.clone();
            guards.push(self.lighting.watch(move |ctx| {
                let lighting = *ctx.value();
                publish_inputs(&inputs, &slot, |i| i.lighting = lighting);
            }));
        }
        {
            let inputs = shared.inputs.clone();
            let slot = shared.redraw_slot.clone();
            guards.push(self.background.watch(move |ctx| {
                let c = ctx.value().components;
                publish_inputs(&inputs, &slot, |i| i.clear = [c[0], c[1], c[2], c[3]]);
            }));
        }
        guards
    }

    /// The per-frame UI hook: writes the renderer's playhead and effective
    /// (auto-fitted) camera back into the bindings while the authored
    /// revision is current. Owns the watch guards for the view's lifetime.
    fn frame_hook(
        &self,
        shared: &SurfaceShared,
        guards: Vec<nami::watcher::BoxWatcherGuard>,
    ) -> impl Fn() + 'static {
        let frame_playback = self.playback.clone();
        let frame_camera = self.camera.clone();
        let feedback = shared.feedback.clone();
        let authored_revision = shared.authored_revision.clone();
        let suppress = shared.suppress.clone();
        let auto_camera = shared.auto_camera.clone();
        move || {
            if let Some(fb) = feedback.load().as_deref().cloned() {
                // Apply the renderer's playhead only while the authored
                // revision is still current — a stale frame must not
                // overwrite a seek or a model swap.
                if fb.revision == authored_revision.get() {
                    let current = frame_playback.snapshot();
                    if fb.playback != current {
                        suppress.set(true);
                        frame_playback.set(fb.playback);
                        suppress.set(false);
                    }
                }
                // Mirror the effective (auto-fitted) camera into the
                // binding so drags start from the visible pose; the
                // suppressed write keeps `auto_camera` set.
                if auto_camera.get() && frame_camera.snapshot() != fb.camera {
                    suppress.set(true);
                    frame_camera.set(fb.camera);
                    suppress.set(false);
                }
            }
            // Keep the watch guards alive for the view's lifetime.
            let _ = &guards;
        }
    }

    /// The input handler: pointer drags orbit/pan, scroll zooms, keys orbit
    /// and reset — each gated by its [`Interaction`] flag.
    fn input_handler(&self, shared: &SurfaceShared) -> impl Fn(&SurfaceInputEvent) + 'static {
        let input_state = Rc::new(RefCell::new(InteractionState::default()));
        let camera = self.camera.clone();
        let model = self.model.clone();
        let shared = shared.clone();
        let interaction = self.interaction;
        move |event: &SurfaceInputEvent| {
            let mut st = input_state.borrow_mut();
            match event {
                SurfaceInputEvent::Focus(false) => {
                    st.orbit = false;
                    st.pan_secondary = false;
                    st.pan_middle = false;
                    st.last = None;
                }
                SurfaceInputEvent::PointerButton {
                    pressed,
                    button,
                    position,
                } => {
                    on_pointer_button(&mut st, *button, *pressed, *position, interaction);
                }
                SurfaceInputEvent::PointerMove { position } => {
                    on_pointer_move(&mut st, *position, &camera, &shared);
                }
                SurfaceInputEvent::Scroll { delta_y, unit, .. }
                    if interaction.contains(Interaction::ZOOM) =>
                {
                    on_scroll(*delta_y, *unit, &camera, &shared);
                }
                SurfaceInputEvent::Key {
                    pressed: true,
                    key,
                    modifiers,
                    ..
                } if interaction.contains(Interaction::KEYBOARD) => {
                    on_key(key, *modifiers, &camera, &model, &shared);
                }
                _ => {}
            }
        }
    }
}

/// Wakes the host through the slot installed at `setup`.
fn wake_host(slot: &Arc<ArcSwapOption<RedrawHandle>>) {
    if let Some(rh) = slot.load().as_deref() {
        rh.request_redraw();
    }
}

/// Publishes `f`'s edit as a fresh immutable snapshot, then wakes the host.
fn publish_inputs(
    inputs: &Arc<ArcSwap<RenderInputs>>,
    slot: &Arc<ArcSwapOption<RedrawHandle>>,
    f: impl Fn(&mut RenderInputs),
) {
    let mut next = (**inputs.load()).clone();
    f(&mut next);
    inputs.store(Arc::new(next));
    wake_host(slot);
}

/// The camera the user currently sees: the renderer's effective (possibly
/// auto-fitted) camera from the latest feedback, else the binding's value.
fn visible_camera(
    feedback: &Arc<ArcSwapOption<Feedback>>,
    camera: &Binding<OrbitCamera>,
) -> OrbitCamera {
    feedback
        .load()
        .as_deref()
        .map_or_else(|| camera.snapshot(), |f| f.camera)
}

/// Screen-space deltas never approach f32's range — pointer positions are
/// bounded by surface coordinates and scroll deltas by device steps.
#[expect(
    clippy::cast_possible_truncation,
    reason = "pointer/scroll deltas are screen-space magnitudes, far below f32 overflow"
)]
const fn f32_delta(v: f64) -> f32 {
    v as f32
}

/// Pointer button edge: toggles the matching drag flag and re-anchors
/// `last` so a mid-gesture release never jumps the view. Each button has
/// its own flag: releasing one keeps the others' drags.
fn on_pointer_button(
    st: &mut InteractionState,
    button: SurfacePointerButton,
    pressed: bool,
    position: Point,
    interaction: Interaction,
) {
    match button {
        SurfacePointerButton::Primary if interaction.contains(Interaction::ORBIT) => {
            st.orbit = pressed;
        }
        SurfacePointerButton::Secondary if interaction.contains(Interaction::PAN) => {
            st.pan_secondary = pressed;
        }
        SurfacePointerButton::Middle if interaction.contains(Interaction::PAN) => {
            st.pan_middle = pressed;
        }
        _ => {}
    }
    st.last = st.any_drag().then_some(position);
}

/// Pointer drag: orbits while the primary flag is down, pans while a
/// secondary/middle flag is — world units per logical pixel derive from
/// the feedback viewport height.
fn on_pointer_move(
    st: &mut InteractionState,
    position: Point,
    camera: &Binding<OrbitCamera>,
    shared: &SurfaceShared,
) {
    if !position.x.is_finite() || !position.y.is_finite() {
        return;
    }
    let Some(last) = st.last else { return };
    let dx = f32_delta(position.x - last.x);
    let dy = f32_delta(position.y - last.y);
    st.last = Some(position);
    let mut cam = visible_camera(&shared.feedback, camera);
    let changed = if st.orbit {
        cam.orbit_by(-dx * 0.01, dy * 0.01)
            .expect("orbit deltas are finite");
        true
    } else if st.pan_secondary || st.pan_middle {
        // World units per logical pixel at the target: the viewport height
        // spans 2·d·tan(fov/2).
        let logical_h = shared
            .feedback
            .load()
            .as_deref()
            .map_or(600.0, |f| f.logical_height);
        let scale = 2.0 * cam.distance() * (cam.fov_y() * 0.5).tan() / logical_h;
        cam.pan_by(-dx * scale, dy * scale)
            .expect("pan deltas are finite");
        true
    } else {
        false
    };
    if changed {
        // Authored edit: the camera watcher publishes and clears
        // auto_camera — no suppression.
        camera.set(cam);
    }
}

/// Scroll zoom: W3C deltaY positive = scroll down = zoom out, so the
/// distance factor is `exp(+step)`.
fn on_scroll(
    delta_y: f64,
    unit: ScrollUnit,
    camera: &Binding<OrbitCamera>,
    shared: &SurfaceShared,
) {
    let step = match unit {
        ScrollUnit::Line => f32_delta(delta_y) * 0.08,
        ScrollUnit::Pixel => f32_delta(delta_y) * 0.0015,
    };
    if !step.is_finite() {
        return;
    }
    let mut cam = visible_camera(&shared.feedback, camera);
    cam.zoom_by(step.exp())
        .expect("exp of a finite scroll step is positive and finite");
    camera.set(cam);
}

/// Key press: arrows orbit (faster with shift), `+`/`-` zoom, `R` resets to
/// the fitted pose and restores auto-fit.
fn on_key(
    key: &Key,
    modifiers: Modifiers,
    camera: &Binding<OrbitCamera>,
    model: &Computed<Model>,
    shared: &SurfaceShared,
) {
    let mut cam = visible_camera(&shared.feedback, camera);
    let speed = if modifiers.contains(Modifiers::SHIFT) {
        0.3
    } else {
        0.1
    };
    let (handled, is_reset) = match key {
        Key::Named(NamedKey::ArrowLeft) => {
            cam.orbit_by(-speed, 0.0).expect("finite orbit");
            (true, false)
        }
        Key::Named(NamedKey::ArrowRight) => {
            cam.orbit_by(speed, 0.0).expect("finite orbit");
            (true, false)
        }
        Key::Named(NamedKey::ArrowUp) => {
            cam.orbit_by(0.0, speed).expect("finite orbit");
            (true, false)
        }
        Key::Named(NamedKey::ArrowDown) => {
            cam.orbit_by(0.0, -speed).expect("finite orbit");
            (true, false)
        }
        Key::Character(k) if k.as_str() == "+" || k.as_str() == "=" => {
            cam.zoom_by(0.9).expect("finite zoom");
            (true, false)
        }
        Key::Character(k) if k.as_str() == "-" || k.as_str() == "_" => {
            cam.zoom_by(1.1).expect("finite zoom");
            (true, false)
        }
        Key::Character(k) if k.eq_ignore_ascii_case("r") => {
            let fb = shared.feedback.load();
            let aspect = fb.as_deref().map_or(1.0, |f| f.aspect);
            OrbitCamera::fit(&model.snapshot().bounds(), aspect).map_or((false, true), |fit| {
                cam = fit;
                (true, true)
            })
        }
        _ => (false, false),
    };
    if handled {
        camera.set(cam);
        // A reset restores the fitted pose *and* the auto-fit behavior;
        // other edits disable it via the camera watcher.
        if is_reset {
            shared.auto_camera.set(true);
            publish_inputs(&shared.inputs, &shared.redraw_slot, |i| {
                i.auto_camera = true;
            });
        }
    }
}

/// Pointer drag bookkeeping: separate flags per gesture so a release of one
/// button does not cancel the other's in-flight drag.
#[derive(Debug, Default)]
struct InteractionState {
    orbit: bool,
    pan_secondary: bool,
    pan_middle: bool,
    last: Option<Point>,
}

impl InteractionState {
    const fn any_drag(&self) -> bool {
        self.orbit || self.pan_secondary || self.pan_middle
    }
}

impl View for Model3D {
    fn body(self, _env: &Environment) -> impl View {
        self.into_surface()
    }
}
