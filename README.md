# waterui-model3d

A [WaterUI](https://github.com/water-rs/waterui) component that loads and renders
glTF 2.0 models on GPU. Built on `waterui-graphics`' shared GPU runtime
(wgpu 30): `Model3D` produces a `GpuContentView` whose `GpuContent` renders
on the engine's render thread into a premultiplied linear Display-P3 target.
This crate always consumes wgpu through the `waterui_graphics::wgpu`
re-export so the wgpu pin lives in exactly one place.

## Status

Early development. Public API is fixed; internals are actively changing.
No performance numbers are published yet — benchmarks are collected on
maintainer hardware (Apple Silicon) and will be documented once measured.

## Features

- glTF 2.0 loading: `Model::load` reads `.gltf`/`.glb` from disk including
  sidecar buffers and images; `Model::from_glb` decodes self-contained bytes
  (base64 `data:` URIs allowed) and rejects external file URIs
- PBR metallic-roughness materials with `KHR_materials_unlit`,
  `KHR_materials_emissive_strength`, and `KHR_texture_transform` extension
  support
- Skeletal (skin) and morph-target animation clips
- Orbit camera (`OrbitCamera`) with validated mutators and automatic
  bounds fitting (`OrbitCamera::fit(&bounds, aspect)`)
- Reactive `Playback` control (clip selection, seek, pause/play) through
  `nami` bindings — state changes drive redraws automatically
- `GpuContentView` integration for GPU-content-hosting backends, plus
  headless offscreen rendering (`GpuRuntime::render_content`) for snapshots
  and tests

## Usage

```rust,no_run
use waterui_model3d::{Model, Model3D, OrbitCamera, Playback};

# async fn example() -> Result<(), Box<dyn std::error::Error>> {
let model = Model::load("assets/models/Fox.glb").await?;

// `fit` frames the model in the viewport; returns `None` for empty scenes.
let aspect = 16.0 / 9.0;
let camera = OrbitCamera::fit(&model.bounds(), aspect)
    .ok_or("model has no finite bounds")?;

// Playback starts paused; select a clip and seek before playing. The Fox
// sample animates: `clip_named("Survey")`, `clip_handle(i)` by index.
let mut playback = Playback::new();
let clip = model.clip_named("Survey")?;
playback.select_clip(&model, clip)?;
playback.seek(&model, std::time::Duration::from_secs(1))?;
playback.play();

// `Model3D` is a WaterUI view. `into_surface` gives the `GpuContentView`
// a GPU-content-hosting backend presents.
let surface = Model3D::new(model)
    .camera(nami::binding(camera))
    .playback(nami::binding(playback))
    .into_surface();
let _ = surface;
# Ok(())
# }
```

`Model` is cheap to clone and is a `nami` constant, so it drops directly into
`Model3D::new`. Camera and playback are live `Binding`s — mutating them (e.g.
`playback` play/pause from a button) re-renders through the `RedrawHandle`
waker; no manual invalidation is needed.

## Examples

Fetch the pinned glTF sample assets first (sha256-verified, ~6.6 MB total,
into the gitignored `assets/models/`):

```sh
python3 scripts/fetch_models.py            # all five models
python3 scripts/fetch_models.py --model Fox --model DamagedHelmet
```

### `offscreen` — headless PNG snapshot

```sh
cargo run --example offscreen -- \
    --model assets/models/DamagedHelmet.glb \
    --output renders/helmet.png --width 1024 --height 768 \
    --yaw 0.6 --pitch 0.25
```

Always paused — `--clip <index|name>` and `--time <seconds>` select an exact
animation pose for deterministic snapshots. The PNG preserves the alpha
channel. Adapter facts print as JSON on stdout; diagnostics go through
`tracing` (`RUST_LOG`).

### `benchmark` — persistent-context frame timing

```sh
cargo run --release --example benchmark -- \
    --model assets/models/DamagedHelmet.glb --frames 600 --warmup 60
```

Sets up one `GpuRuntime`, `GpuSurface` and render target, then times CPU
submit and completed frames (via `wgpu::Device::poll`) separately, reporting
p50/p95 as JSON together with adapter and workload facts from the glTF
metadata. `render_offscreen` is deliberately not used — it recreates setup
and readback per call. Refuses to run on a software CPU adapter. A no-clip
run additionally reports whether the static model requested further redraws
after initial readiness (`checks.static_model_idle`).

### `viewer` — interactive winit host

```sh
cargo run --example viewer -- --model assets/models/Fox.glb --clip Survey
```

Viewer controls: drag orbits, secondary/middle drag pans, scroll zooms,
arrow keys orbit, `+`/`-` zoom, `R` resets the camera; `Space` toggles
play/pause, `C` cycles animation clips, `Home` seeks to the start.
```

Hosts the real `Model3D` `GpuSurface` in a transparent winit window.
Event-driven rendering (`ControlFlow::Wait` + `RedrawHandle` waker), no busy
polling. Input is forwarded as platform-neutral `SurfaceInputEvent`s.

Local keys: `Space` pause/play, `R` restart clip, `C` cycle clips.

## Design rationale

glTF rendering was built on `waterui-graphics` rather than an existing
ecosystem renderer because:

- **rend3** is archived and pinned to an old wgpu generation.
- **three-d** targets glow/OpenGL, not wgpu.
- **Bevy** couples rendering to its ECS/application model — unusable as a
  component inside a WaterUI view tree.

In-tree rendering keeps the whole pipeline behind WaterUI's own reactive
surface contract (`GpuView`/`GpuSurface`), which is what makes the component
embeddable.

## License

MIT OR Apache-2.0, at your option.
