//! Opt-in hardware GPU validation. Runs on a physical device or Metal CI.

mod support;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use waterui_graphics::gpu::{Context, Frame, RedrawHandle};
use waterui_graphics::wgpu;
use waterui_model3d::{Model, Model3D};

#[test]
#[ignore = "requires a hardware GPU; run explicitly with --ignored"]
fn static_content_stays_idle_and_recreates_all_device_resources() {
    let mut document = support::document();
    document["materials"] = serde_json::json!([
        {"alphaMode": "OPAQUE"},
        {"alphaMode": "MASK", "doubleSided": true},
        {"alphaMode": "BLEND", "pbrMetallicRoughness": {"baseColorFactor": [1, 0, 0, 0.5]}}
    ]);
    document["meshes"][0]["primitives"] = serde_json::json!([
        {"attributes": {"POSITION": 0}, "material": 0},
        {"attributes": {"POSITION": 0}, "material": 1},
        {"attributes": {"POSITION": 0}, "material": 2}
    ]);
    let model = pollster::block_on(Model::from_glb(support::glb(&document))).unwrap();
    let mut surface = Model3D::new(model).into_surface();
    let mut content = surface.take_content();
    let wakes = Arc::new(AtomicU64::new(0));
    for _ in 0..2 {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
            compatible_surface: None,
        }))
        .expect("hardware adapter");
        assert_ne!(adapter.get_info().device_type, wgpu::DeviceType::Cpu);
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
        let format = wgpu::TextureFormat::Rgba16Float;
        let count = Arc::clone(&wakes);
        content.setup(&Context {
            adapter: &adapter,
            device: &device,
            queue: &queue,
            format,
            redraw: RedrawHandle::new(move || {
                count.fetch_add(1, Ordering::Relaxed);
            }),
        });
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("model3d-recreation-test"),
            size: wgpu::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        for index in 0..3 {
            surface.frame();
            let mut frame = Frame::new(
                &device,
                &queue,
                &texture,
                &view,
                format,
                (256, 256),
                1.0,
                (Duration::from_millis(index * 16), Duration::from_millis(16)),
            );
            content.render(&mut frame);
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(Duration::from_secs(10)),
                })
                .unwrap();
            if index > 0 {
                assert!(
                    !frame.redraw_requested(),
                    "paused static scene requested another frame"
                );
                assert_eq!(
                    wakes.load(Ordering::Relaxed),
                    0,
                    "static scene woke the host"
                );
            }
            wakes.store(0, Ordering::Relaxed);
        }
    }
}
