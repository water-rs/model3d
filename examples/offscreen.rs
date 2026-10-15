//! Headless snapshot: renders a glTF model to a PNG without opening a window.
//!
//! ```sh
//! cargo run --example offscreen -- \
//!     --model assets/models/Fox.glb \
//!     --output renders/fox.png --width 1024 --height 768 \
//!     --clip Survey --time 0.5 --yaw 0.6 --pitch 0.25
//! ```
//!
//! Playback is always paused: pair `--clip` with `--time` to snapshot an exact
//! animation pose deterministically. The output PNG keeps the renderer's
//! alpha channel so snapshots composite like the live view.
//!
//! Only the asset load and adapter bring-up are async — they are independent
//! and joined. The GPU render/readback and the PNG write run synchronously on
//! `main` after that; no blocking work sits inside an executor.

mod shared;

use std::error::Error;
use std::io::stdout;

use clap::Parser;
use waterui_graphics::gpu::{Context, Frame, GpuContent, GpuContentView};
use waterui_graphics::{GpuRuntime, OffscreenSize};
use waterui_model3d::{Model, Model3D};

use shared::CommonArgs;

/// Render a glTF model to a PNG without opening a window.
#[derive(Parser, Debug)]
#[command(version, about)]
struct OffscreenArgs {
    #[command(flatten)]
    common: CommonArgs,

    /// Destination PNG path; missing parent directories are created.
    #[arg(long, value_name = "PATH")]
    output: std::path::PathBuf,
}

/// `GpuRuntime::render_content` takes `impl GpuContent` while a view hands
/// back `Box<dyn GpuContent>` (the trait has no blanket impl), so forward the
/// three methods explicitly — a private headless host wrapper, not a shim.
struct Headless(Box<dyn GpuContent>);

impl GpuContent for Headless {
    fn setup(&mut self, gpu: &Context<'_>) {
        self.0.setup(gpu);
    }

    fn render(&mut self, frame: &mut Frame<'_>) {
        self.0.render(frame);
    }

    fn is_opaque(&self) -> bool {
        self.0.is_opaque()
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = OffscreenArgs::parse();
    shared::init_tracing();

    // The two async steps are independent — join them.
    let path = args.common.model.clone();
    let (model, runtime) =
        pollster::block_on(async { futures_util::join!(Model::load(path), GpuRuntime::new()) });
    let model = model?;
    let runtime = runtime?;
    shared::require_hardware_adapter(runtime.context().adapter())?;

    let bytes = std::fs::read(&args.common.model)?;
    let workload = shared::workload_stats(&bytes)?;

    let width = args.common.width.get();
    let height = args.common.height.get();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "render sizes are bounded by GPU limits, far below f32's range"
    )]
    let aspect = (f64::from(width) / f64::from(height)) as f32;
    let camera = shared::camera(&model, &args.common, aspect)?;
    // Snapshots are always paused; --time selects the pose.
    let playback = shared::playback(&model, args.common.clip.as_deref(), args.common.time, true)?;

    // Transparent background, matching the live view's default so the PNG
    // alpha channel composites identically.
    let mut surface: GpuContentView = Model3D::new(model)
        .camera(nami::binding(camera))
        .playback(nami::binding(playback))
        .into_surface();
    let content = Headless(surface.take_content());

    let size = OffscreenSize::try_from_pixels(width, height).ok_or("zero-size render requested")?;
    // `render_content` runs the real GpuContent through the engine's
    // offscreen surface — the same premultiplied linear Display-P3 pipeline a
    // hosted view renders — then reads it back in one synchronous call.
    let image = runtime.render_content(content, size, 1.0);

    // Blocking filesystem + PNG encode on main, outside any executor.
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    image.save_png(&args.output)?;

    // Facts on stdout, diagnostics on stderr.
    serde_json::to_writer_pretty(
        stdout(),
        &serde_json::json!({
            "adapter": shared::adapter_report(runtime.context().adapter()),
            "workload": workload,
            "output": args.output,
            "width": width,
            "height": height,
        }),
    )?;
    println!();
    Ok(())
}
