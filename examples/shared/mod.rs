//! Helpers shared by the `waterui-model3d` examples.
//!
//! Not a cargo target: `mod shared;` inside an `examples/*.rs` file resolves
//! to this directory, which is how the three binaries share their common
//! argument parsing, adapter reporting, camera/playback construction and glTF
//! workload accounting.

use std::error::Error;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::time::Duration;

use clap::Args;
use serde::Serialize;
use waterui_graphics::wgpu;
use waterui_model3d::{Model, OrbitCamera, Playback};

/// Model, viewport and pose arguments common to every example.
#[derive(Args, Debug, Clone)]
pub struct CommonArgs {
    /// Path to a self-contained glTF asset (`.glb`, or `.gltf` with its
    /// sidecar resources).
    #[arg(long, value_name = "PATH")]
    pub model: PathBuf,

    /// Render width in physical pixels; must be nonzero.
    #[arg(long, default_value = "1024")]
    pub width: NonZeroU32,

    /// Render height in physical pixels; must be nonzero.
    #[arg(long, default_value = "768")]
    pub height: NonZeroU32,

    /// Orbit angle around the +Y axis, radians (keeps the fitted default when
    /// omitted).
    #[arg(long, value_name = "RADIANS")]
    pub yaw: Option<f32>,

    /// Orbit elevation above the horizon plane, radians.
    #[arg(long, value_name = "RADIANS")]
    pub pitch: Option<f32>,

    /// Eye distance from the camera target.
    #[arg(long, value_name = "UNITS")]
    pub distance: Option<f32>,

    /// Animation clip to select, by index or name.
    #[arg(long, value_name = "INDEX_OR_NAME")]
    pub clip: Option<String>,

    /// Animation playback position, seconds.
    #[arg(long, value_name = "SECONDS")]
    pub time: Option<f64>,
}

/// Installs a `tracing` subscriber honouring `RUST_LOG`, defaulting to `info`.
pub fn init_tracing() {
    // `env-filter` would pull new crates into the frozen lockfile, so
    // RUST_LOG selects a coarse level; per-target filters are not supported.
    let level = std::env::var("RUST_LOG")
        .ok()
        .and_then(|v| v.parse::<tracing_subscriber::filter::LevelFilter>().ok())
        .unwrap_or(tracing_subscriber::filter::LevelFilter::INFO);
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_max_level(level)
        .init();
}

/// Structured facts about the selected GPU adapter, for JSON output.
pub fn adapter_report(adapter: &wgpu::Adapter) -> serde_json::Value {
    let info = adapter.get_info();
    serde_json::json!({
        "name": info.name,
        "backend": format!("{:?}", info.backend),
        "device_type": format!("{:?}", info.device_type),
        "vendor": info.vendor,
        "device": info.device,
        "driver": info.driver,
        "driver_info": info.driver_info,
    })
}

/// Rejects software CPU adapters: every example executes real GPU work and a
/// fallback rasterizer would silently produce unrepresentative results.
pub fn require_hardware_adapter(adapter: &wgpu::Adapter) -> Result<(), Box<dyn Error>> {
    let info = adapter.get_info();
    if info.device_type == wgpu::DeviceType::Cpu {
        return Err(format!(
            "refusing to run on software CPU adapter '{}'; a hardware GPU is required",
            info.name
        )
        .into());
    }
    Ok(())
}

/// Builds the orbit camera for `model` at `aspect`, applying pose overrides.
///
/// `aspect` must be positive and finite; callers pass the real viewport ratio
/// so `OrbitCamera::fit` frames portrait and landscape content correctly. A
/// `None` fit means the model has no finite bounds (an empty scene), which the
/// caller surfaces as an error rather than hiding behind an epsilon.
///
/// # Errors
/// Returns an error when the aspect is invalid, the model cannot be framed,
/// or a validated camera mutator rejects an override.
pub fn camera(
    model: &Model,
    args: &CommonArgs,
    aspect: f32,
) -> Result<OrbitCamera, Box<dyn Error>> {
    if !(aspect.is_finite() && aspect > 0.0) {
        return Err(
            format!("invalid viewport aspect {aspect}; expected a positive finite value").into(),
        );
    }
    let mut camera = OrbitCamera::fit(&model.bounds(), aspect)
        .ok_or("model bounds are empty or non-finite; cannot fit a camera")?;
    if let Some(yaw) = args.yaw {
        camera.set_yaw(yaw)?;
    }
    if let Some(pitch) = args.pitch {
        camera.set_pitch(pitch)?;
    }
    if let Some(distance) = args.distance {
        camera.set_distance(distance)?;
    }
    Ok(camera)
}

/// Builds playback state: explicit clip selection, a validated seek position
/// and the requested pause state.
///
/// `Playback::new` starts paused with no clip. `clip` selects by index
/// (`usize`) or by name; `time` seeks to a non-negative finite position in
/// seconds. `paused == false` calls `play` last, so a selected clip starts
/// running and a clip-less static model simply stays a no-op.
///
/// # Errors
/// Returns the loader's `InvalidClipError` when the clip does not exist, or an
/// error for a non-finite/negative `--time`.
pub fn playback(
    model: &Model,
    clip: Option<&str>,
    time: Option<f64>,
    paused: bool,
) -> Result<Playback, Box<dyn Error>> {
    let mut playback = Playback::new();
    if let Some(selector) = clip {
        let handle = match selector.parse::<usize>() {
            Ok(index) => model.clip_handle(index)?,
            Err(_) => model.clip_named(selector)?,
        };
        playback.select_clip(model, handle)?;
    }
    if let Some(seconds) = time {
        if !(seconds.is_finite() && seconds >= 0.0) {
            return Err(format!(
                "invalid --time {seconds}; expected a non-negative finite seconds value"
            )
            .into());
        }
        playback.seek(model, Duration::from_secs_f64(seconds))?;
    }
    if !paused {
        playback.play();
    }
    Ok(playback)
}

/// glTF workload accounting, separated into asset inventory and the geometry
/// the renderer is actually asked to draw (nodes of the default scene that
/// carry a mesh).
#[derive(Serialize, Debug)]
pub struct WorkloadStats {
    /// `mesh.primitives` declared anywhere in the asset.
    pub asset_primitives: usize,
    /// Total `POSITION` accessor vertices across all asset meshes.
    pub asset_vertices: usize,
    /// Total index-accessor entries across all asset meshes.
    pub asset_indices: usize,
    /// Nodes in the default scene subtree that instantiate a mesh.
    pub rendered_mesh_instances: usize,
    /// `mesh.primitives` reachable from the default scene's mesh nodes.
    pub rendered_primitives: usize,
    /// Number of animation clips in the asset.
    pub animation_clips: usize,
}

/// Reads an asset's bytes on a blocking executor thread.
///
/// Metadata inspection needs the file contents even though `Model::load`
/// already performs its own async-safe read; duplicating a few megabytes of
/// I/O is honest, duplicating the parse is not — this only feeds `gltf`
/// metadata counting.
///
/// # Errors
/// Propagates the file read error.
/// Counts workload facts from a glTF document's binary payload.
///
/// `rendered_*` numbers walk only the default scene's node tree, so extras in
/// the file that never reach the renderer stay out of the figure that matters.
///
/// # Errors
/// Returns an error when the bytes are not a parseable glTF document.
pub fn workload_stats(bytes: &[u8]) -> Result<WorkloadStats, Box<dyn Error>> {
    let gltf = gltf::Gltf::from_slice(bytes)?;
    let document = &gltf.document;

    let mut asset_primitives = 0usize;
    let mut asset_vertices = 0usize;
    let mut asset_indices = 0usize;
    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            asset_primitives += 1;
            if let Some(accessor) = primitive.get(&gltf::Semantic::Positions) {
                asset_vertices += accessor.count();
            }
            if let Some(indices) = primitive.indices() {
                asset_indices += indices.count();
            }
        }
    }

    let mut rendered_mesh_instances = 0usize;
    let mut rendered_primitives = 0usize;
    let mut stack: Vec<gltf::Node<'_>> = document
        .default_scene()
        .or_else(|| document.scenes().next())
        .map(|scene| scene.nodes().collect())
        .unwrap_or_default();
    while let Some(node) = stack.pop() {
        if let Some(mesh) = node.mesh() {
            rendered_mesh_instances += 1;
            rendered_primitives += mesh.primitives().count();
        }
        stack.extend(node.children());
    }

    Ok(WorkloadStats {
        asset_primitives,
        asset_vertices,
        asset_indices,
        rendered_mesh_instances,
        rendered_primitives,
        animation_clips: document.animations().count(),
    })
}
