//! Measures persistent GPU content, excluding load, setup, and readback.
//! Run release builds on a physical device. Completion samples include CPU
//! submission, driver scheduling, and GPU execution; they are not GPU timestamps.

mod shared;

use std::error::Error;
use std::io::stdout;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use clap::Parser;
use nami::{Binding, Signal};
use waterui_graphics::gpu::{Context, Frame, GpuContent, GpuContentView, RedrawHandle};
use waterui_graphics::wgpu;
use waterui_model3d::{Model, Model3D, OrbitCamera};

use shared::CommonArgs;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    #[command(flatten)]
    common: CommonArgs,
    /// Number of measured frames.
    #[arg(long, default_value = "300")]
    frames: NonZeroU32,
    /// Additional unmeasured frames after the explicit readiness frame.
    #[arg(long, default_value_t = 60)]
    warmup: u32,
    /// Change the camera by this many radians per frame; zero tests static idle.
    #[arg(long, default_value_t = 0.0)]
    orbit_step: f32,
}

struct Gpu {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

/// Requests a hardware adapter and device. Async initialization ends before
/// the synchronous device polling loop.
fn init_gpu() -> Result<Gpu> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
        compatible_surface: None,
    }))?;
    shared::require_hardware_adapter(&adapter)?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("model3d-benchmark"),
        ..Default::default()
    }))?;
    Ok(Gpu {
        adapter,
        device,
        queue,
    })
}

fn completed(device: &wgpu::Device) -> Result<()> {
    device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(Duration::from_secs(10)),
    })?;
    Ok(())
}

fn summary(samples: &mut [f64]) -> serde_json::Value {
    samples.sort_by(f64::total_cmp);
    let rank = |percent: usize| (samples.len() * percent).div_ceil(100) - 1;
    serde_json::json!({
        "min": samples[0], "p50": samples[rank(50)],
        "p95": samples[rank(95)], "max": samples[samples.len() - 1],
    })
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

#[expect(
    clippy::cast_precision_loss,
    reason = "render sizes are bounded by GPU texture limits, far below 2^24"
)]
fn aspect(width: u32, height: u32) -> f32 {
    width as f32 / height as f32
}

/// One persistent render target driven frame by frame, the way a host does.
struct Driver {
    surface: GpuContentView,
    content: Box<dyn GpuContent>,
    camera: Binding<OrbitCamera>,
    orbit_step: f32,
    target: wgpu::Texture,
    view: wgpu::TextureView,
    size: (u32, u32),
    elapsed: Duration,
}

impl Driver {
    const DELTA: Duration = Duration::from_nanos(16_666_667);

    /// Advances the camera and the clock, renders, and reports whether the
    /// content asked for another frame.
    fn frame(&mut self, gpu: &Gpu) -> Result<bool> {
        if self.orbit_step != 0.0 {
            let mut next = self.camera.snapshot();
            next.set_yaw(next.yaw() + self.orbit_step)?;
            self.camera.set(next);
        }
        self.surface.frame();
        self.elapsed += Self::DELTA;
        let mut frame = Frame::new(
            &gpu.device,
            &gpu.queue,
            &self.target,
            &self.view,
            FORMAT,
            self.size,
            1.0,
            (self.elapsed, Self::DELTA),
        );
        self.content.render(&mut frame);
        Ok(frame.redraw_requested())
    }
}

fn render_target(device: &wgpu::Device, (width, height): (u32, u32)) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("model3d-benchmark-target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    })
}

struct Measurement {
    frame_requests: u64,
    wake_requests: u64,
    submit_ms: Vec<f64>,
    completed_ms: Vec<f64>,
}

/// Runs the warmup and measured frames after readiness; every frame waits
/// for GPU completion before the next one starts.
fn measure(driver: &mut Driver, gpu: &Gpu, args: &Args, wakes: &AtomicU64) -> Result<Measurement> {
    let mut frame_requests = 0u64;
    for _ in 0..args.warmup {
        frame_requests += u64::from(driver.frame(gpu)?);
        completed(&gpu.device)?;
    }
    let samples = args.frames.get() as usize;
    let mut submit_ms = Vec::with_capacity(samples);
    let mut completed_ms = Vec::with_capacity(samples);
    for _ in 0..samples {
        let start = Instant::now();
        frame_requests += u64::from(driver.frame(gpu)?);
        let submitted = Instant::now();
        completed(&gpu.device)?;
        let done = Instant::now();
        submit_ms.push(milliseconds(submitted.duration_since(start)));
        completed_ms.push(milliseconds(done.duration_since(start)));
    }
    Ok(Measurement {
        frame_requests,
        wake_requests: wakes.load(Ordering::Relaxed),
        submit_ms,
        completed_ms,
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    shared::init_tracing();
    if !args.orbit_step.is_finite() {
        return Err("--orbit-step must be finite".into());
    }
    let gpu = init_gpu()?;
    let load_started = Instant::now();
    let model = pollster::block_on(Model::load(args.common.model.clone()))?;
    let asset_load_ms = milliseconds(load_started.elapsed());
    let workload = shared::workload_stats(&std::fs::read(&args.common.model)?)?;
    let size = (args.common.width.get(), args.common.height.get());
    let camera = nami::binding(shared::camera(
        &model,
        &args.common,
        aspect(size.0, size.1),
    )?);
    let playback = shared::playback(&model, args.common.clip.as_deref(), args.common.time, false)?;
    let animated = args.common.clip.is_some();
    let static_scene = !animated && args.orbit_step == 0.0;
    let mut surface = Model3D::new(model)
        .camera(camera.clone())
        .playback(nami::binding(playback))
        .into_surface();
    let content = surface.take_content();
    let target = render_target(&gpu.device, size);
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let mut driver = Driver {
        surface,
        content,
        camera,
        orbit_step: args.orbit_step,
        target,
        view,
        size,
        elapsed: Duration::ZERO,
    };
    let wakes = Arc::new(AtomicU64::new(0));
    let count = Arc::clone(&wakes);
    let setup_started = Instant::now();
    driver.content.setup(&Context {
        adapter: &gpu.adapter,
        device: &gpu.device,
        queue: &gpu.queue,
        format: FORMAT,
        redraw: RedrawHandle::new(move || {
            count.fetch_add(1, Ordering::Relaxed);
        }),
    });
    let setup_cpu_ms = milliseconds(setup_started.elapsed());
    // Always initialize fully, including --warmup=0. Setup uploads and the
    // first frame's requests are excluded from idle and timing accounting.
    driver.frame(&gpu)?;
    completed(&gpu.device)?;
    let setup_and_first_frame_ms = milliseconds(setup_started.elapsed());
    wakes.store(0, Ordering::Relaxed);
    let mut measured = measure(&mut driver, &gpu, &args, &wakes)?;
    let idle = measured.frame_requests == 0 && measured.wake_requests == 0;
    let output = serde_json::json!({
        "adapter": shared::adapter_report(&gpu.adapter), "workload": workload,
        "config": {
            "model": args.common.model, "width": size.0, "height": size.1,
            "format": format!("{FORMAT:?}"), "samples": args.frames.get(),
            "warmup": args.warmup, "readiness_frames": 1,
            "clip": args.common.clip, "orbit_step": args.orbit_step,
            "animation_delta_seconds": Driver::DELTA.as_secs_f64(),
        },
        "checks": {
            "frame_redraw_requests": measured.frame_requests,
            "host_wake_requests": measured.wake_requests,
            "static_idle": static_scene.then_some(idle),
        },
        "timing": {
            "cpu_submit_ms": summary(&mut measured.submit_ms),
            "completed_frame_ms": summary(&mut measured.completed_ms),
        },
        "initialization": {
            "asset_load_ms": asset_load_ms,
            "setup_cpu_ms": setup_cpu_ms,
            "setup_and_first_completed_frame_ms": setup_and_first_frame_ms,
        },
    });
    serde_json::to_writer_pretty(stdout(), &output)?;
    if static_scene && !idle {
        return Err("static content kept requesting redraws after readiness".into());
    }
    if animated && measured.frame_requests == 0 {
        return Err("playing animation failed to request another frame".into());
    }
    Ok(())
}
