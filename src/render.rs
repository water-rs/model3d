//! wgpu 30 renderer for [`Model`]: PBR pipelines, scene upload, draw
//! dispatch with instancing, frustum culling, skinned/morphed bounds.
//!
//! Buffer ABI (all `#[repr(C)]`, mirrored in `src/shaders/model.wgsl`):
//!
//! - `InstanceData` — per-draw storage record: world + normal matrices, the
//!   draw's joint-palette base, morph-weight base, morph-delta range and the
//!   primitive's vertex window inside the shared buffers.
//! - `GlobalsData` — camera matrices, position, exposure, key/fill lights and
//!   ambient.
//! - `MaterialUniform` — factors plus per-texture-slot uv transform:
//!   `slots_a[i] = (uv_set, offset.x, offset.y, rotation)`,
//!   `slots_b[i] = (scale.x, scale.y, _, _)`.
//! - `MorphDelta` — storage vec4 stride (48 bytes), indexed
//!   `morph_offset + target * vertex_count + local_vertex`.
//!
//! The render target is composited as premultiplied-alpha linear color:
//! blend draws output straight alpha through `ALPHA_BLENDING` (whose source
//! term multiplies by src-alpha), opaque/mask draws force alpha to 1, and
//! the clear color is premultiplied before it is written.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3, Vec4};

// `wgpu` crosses the public `GpuContent` boundary (`Context`/`Frame` hand us
// the engine's device, queue and texture), so it comes through
// `waterui-graphics`' re-export — one GPU type universe, no version drift.
use waterui_graphics::wgpu;

use crate::model::{
    AlphaMode, Model, ModelData, MorphDelta, SkinVertex, Trs, Vertex, anim,
    compute_global_transforms,
};

/// Shader module label.
const SHADER_LABEL: &str = "waterui-model3d/model.wgsl";
/// Depth buffer format.
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Texture slots in material bind group order.
const TEXTURE_SLOTS: usize = 5;
/// Which material slots are authored in sRGB.
const SLOT_SRGB: [bool; TEXTURE_SLOTS] = [true, false, false, false, true];

/// Per-draw instance record (uploaded as one storage buffer).
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct InstanceData {
    world: [[f32; 4]; 4],
    normal: [[f32; 4]; 4],
    joint_base: u32,
    weight_base: u32,
    morph_offset: u32,
    morph_count: u32,
    vertex_base: u32,
    vertex_count: u32,
    joint_count: u32,
    _pad: u32,
}
const _: () = assert!(std::mem::size_of::<InstanceData>() == 160);

/// Frame-global uniform: camera matrices, position, exposure, key and fill
/// lights, ambient. Sized 208 bytes, 16-byte aligned for `uniform`.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct GlobalsData {
    view_proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    light_count: u32,
    light_dir: [f32; 3],
    exposure: f32,
    light_color: [f32; 3],
    _pad0: f32,
    fill_dir: [f32; 3],
    _pad1: f32,
    fill_color: [f32; 3],
    _pad2: f32,
    ambient: [f32; 4],
}
const _: () = assert!(std::mem::size_of::<GlobalsData>() == 224);

/// Per-material uniform block; `slots_a[i]` packs
/// `(uv_set_index, offset.x, offset.y, rotation)` and `slots_b[i]` packs
/// `(scale.x, scale.y, _, _)` for `KHR_texture_transform`.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct MaterialUniform {
    base_color_factor: [f32; 4],
    emissive_factor: [f32; 4],
    metallic_factor: f32,
    roughness_factor: f32,
    normal_scale: f32,
    occlusion_strength: f32,
    alpha_cutoff: f32,
    unlit: u32,
    alpha_mode: u32,
    double_sided: u32,
    slots_a: [[f32; 4]; TEXTURE_SLOTS],
    slots_b: [[f32; 4]; TEXTURE_SLOTS],
}
const _: () = assert!(std::mem::size_of::<MaterialUniform>() == 224);

/// Parameters the caller controls per frame. Colors are linear; `clear` is
/// straight-alpha (premultiplied before it reaches the target).
#[derive(Debug, Clone)]
pub struct FrameParams {
    /// View matrix.
    pub view: Mat4,
    /// Projection matrix.
    pub proj: Mat4,
    /// Camera world position.
    pub camera_pos: Vec3,
    /// Direction the key light shines *toward*, normalized.
    pub light_dir: Vec3,
    /// Key-light color (linear) and intensity carried in the draw call.
    pub light_color: Vec3,
    /// Key-light intensity.
    pub light_intensity: f32,
    /// Fill light direction; `Vec3::ZERO` disables it.
    pub fill_dir: Vec3,
    /// Fill light color (linear).
    pub fill_color: Vec3,
    /// Ambient light: linear color premultiplied by intensity in `.w`-less
    /// form — `ambient.rgb * ambient.a` is the irradiance applied.
    pub ambient: [f32; 4],
    /// Linear exposure multiplier applied after shading.
    pub exposure: f32,
    /// Linear straight-alpha clear color.
    pub clear: [f32; 4],
}

/// Result of one [`Renderer::draw`] — what was submitted.
#[derive(Debug, Clone, Copy, Default)]
pub struct DrawStats {
    /// Draw calls issued.
    pub draws: u32,
    /// Instances across calls.
    pub instances: u32,
    /// Draws culled by the frustum.
    pub culled: u32,
}

#[derive(Debug)]
struct GpuTexture {
    #[expect(dead_code, reason = "the view keeps the texture alive")]
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

/// One uploaded glTF primitive: byte ranges into the shared buffers.
#[derive(Debug)]
struct GpuPrimitive {
    vertex_offset: u64,
    vertex_count: u32,
    index_offset: u64,
    index_count: u32,
    morph_offset: u32,
    morph_count: u32,
    material: usize,
    alpha: AlphaMode,
    bounds_center: Vec3,
    bounds_radius: f32,
    /// Per-target max |position delta|; the culling expansion is
    /// Σ|weight|·delta from the evaluated node weights.
    morph_max_displacement: Vec<f32>,
}

/// A draw: (node, primitive) resolved to GPU ranges.
#[derive(Debug, Clone, Copy)]
struct DrawCall {
    node: usize,
    primitive: usize,
}

/// One instanced opaque call: `members` are indices into `draws`.
#[derive(Debug)]
struct OpaqueBucket {
    primitive: usize,
    key: PipelineKey,
    material: usize,
    members: Vec<u32>,
}

/// Pending blend draw: (view depth, draw index, material index).
type BlendSpan = (f32, u32, u32);
/// Sorted blend draw: (draw index, first instance, material index).
type BlendDraw = (u32, u32, u32);

/// One recorded instanced call, retaining its primitive identity.
#[derive(Debug, Clone, Copy)]
struct DrawSpan {
    first_instance: u32,
    count: u32,
    key: PipelineKey,
    material: usize,
    primitive: usize,
}

/// All device-bound resources for one uploaded model.
#[derive(Debug)]
struct GpuScene {
    // Uploaded textures kept alive for the material bind groups' views.
    #[expect(dead_code, reason = "views hold no owning reference to textures")]
    textures: Vec<GpuTexture>,
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    skin_buf: wgpu::Buffer,
    /// Retains the morph-delta storage buffer for the scene bind group —
    /// views hold no owning reference.
    _morph_buf: Option<wgpu::Buffer>,
    joint_buf: wgpu::Buffer,
    weight_buf: wgpu::Buffer,
    instance_buf: wgpu::Buffer,
    scene_bind_group: wgpu::BindGroup,
    primitives: Vec<GpuPrimitive>,
    draws: Vec<DrawCall>,
    materials: Vec<wgpu::BindGroup>,
    // Per-frame scratch — no allocation during draw.
    pose_globals: Vec<Mat4>,
    joint_scratch: Vec<[f32; 16]>,
    weights_uploaded: Vec<f32>,
    draw_order: Vec<u32>,
    draw_depths: Vec<f32>,
    buckets: Vec<OpaqueBucket>,
    bucket_pool: Vec<Vec<u32>>,
    draw_spans: Vec<DrawSpan>,
    blend_spans: Vec<BlendSpan>,
    blend_draws: Vec<BlendDraw>,
    instance_scratch: Vec<InstanceData>,
}

#[derive(Debug)]
struct UploadedScene {
    model: Model,
    scene: GpuScene,
}

/// Shared geometry buffers and per-primitive GPU ranges of one model.
#[derive(Debug)]
struct GpuGeometry {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    skin_buf: wgpu::Buffer,
    morph_buf: Option<wgpu::Buffer>,
    primitives: Vec<GpuPrimitive>,
}

/// Per-frame deformation storage buffers and the element counts they hold.
#[derive(Debug)]
struct DeformBuffers {
    joint_buf: wgpu::Buffer,
    weight_buf: wgpu::Buffer,
    instance_buf: wgpu::Buffer,
    joint_count: usize,
    weight_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PipelineKey {
    blend: bool,
    double_sided: bool,
    flipped_winding: bool,
}

/// The wgpu-side renderer. Owns shared GPU state; one `GpuScene` per uploaded
/// [`Model`]. Reused across frames and across model swaps.
pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    format: wgpu::TextureFormat,
    msaa_samples: u32,
    globals_bgl: wgpu::BindGroupLayout,
    scene_bgl: wgpu::BindGroupLayout,
    material_bgl: wgpu::BindGroupLayout,
    globals_buf: wgpu::Buffer,
    globals_bg: wgpu::BindGroup,
    shader: wgpu::ShaderModule,
    mip_pipeline_lin: wgpu::RenderPipeline,
    mip_pipeline_srgb: wgpu::RenderPipeline,
    default_sampler: wgpu::Sampler,
    white_lin: wgpu::TextureView,
    white_srgb: wgpu::TextureView,
    neutral_normal: wgpu::TextureView,
    depth_view: Option<wgpu::TextureView>,
    msaa_view: Option<wgpu::TextureView>,
    pass_size: (u32, u32),
    pipelines: HashMap<PipelineKey, wgpu::RenderPipeline>,
    scene: Option<UploadedScene>,
    // Scratch kept across frames.
    pose_scratch: Vec<Trs>,
    morph_weight_scratch: Vec<Vec<f32>>,
}

impl Renderer {
    /// Creates the renderer bound to `device`/`queue`, rendering into targets
    /// of `format` with `msaa_samples` multisampling (1 = off).
    ///
    /// The `GpuContent` adapter passes its framework-provided device/queue/
    /// format through here so there is exactly one GPU type universe.
    #[must_use]
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        msaa_samples: u32,
    ) -> Self {
        let device = device.clone();
        let queue = queue.clone();

        // `shader_module` prepends cherenkov's shared WGSL helpers — the
        // fragment shader converts linear sRGB into the target's
        // premultiplied linear Display-P3 through `cherenkov_linear_srgb`.
        let shader = waterui_graphics::cherenkov_gpu::interop::shader_module(
            &device,
            SHADER_LABEL,
            include_str!("shaders/model.wgsl"),
        );

        let (globals_bgl, scene_bgl, material_bgl) = bind_group_layouts(&device);
        let (globals_buf, globals_bg) = globals_binding(&device, &globals_bgl);
        // Fallback textures: flat-sampled stand-ins for absent slots.
        let white_lin = fallback_texture(&device, &queue, "white lin", [255; 4], false);
        let white_srgb = fallback_texture(&device, &queue, "white srgb", [255; 4], true);
        let neutral_normal = fallback_texture(
            &device,
            &queue,
            "neutral normal",
            [128, 128, 255, 255],
            false,
        );

        let default_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("model3d default"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let (mip_pipeline_lin, mip_pipeline_srgb) = mip_pipelines(&device);

        Self {
            device,
            queue,
            format,
            msaa_samples,
            globals_bgl,
            scene_bgl,
            material_bgl,
            globals_buf,
            globals_bg,
            shader,
            mip_pipeline_lin,
            mip_pipeline_srgb,
            default_sampler,
            white_lin,
            white_srgb,
            neutral_normal,
            depth_view: None,
            msaa_view: None,
            pass_size: (0, 0),
            pipelines: HashMap::new(),
            scene: None,
            pose_scratch: Vec::new(),
            morph_weight_scratch: Vec::new(),
        }
    }

    /// Whether `model` is the currently uploaded scene (Arc identity).
    #[must_use]
    pub fn contains(&self, model: &Model) -> bool {
        self.scene
            .as_ref()
            .is_some_and(|s| Model::ptr_eq(&s.model, model))
    }

    /// Uploads `model` into a fresh `GpuScene`, replacing any previous scene.
    /// Texture/pipeline/buffer resources are rebuilt against the live device.
    pub fn upload(&mut self, model: &Model) {
        let scene = self.upload_scene(model.data());
        self.scene = Some(UploadedScene {
            model: model.clone(),
            scene,
        });
    }

    fn ensure_targets(&mut self, width: u32, height: u32) {
        if self.pass_size == (width, height) {
            return;
        }
        self.pass_size = (width, height);
        let size = wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        };
        let depth = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("model3d depth"),
            size,
            mip_level_count: 1,
            sample_count: self.msaa_samples,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        self.depth_view = Some(depth.create_view(&wgpu::TextureViewDescriptor::default()));
        self.msaa_view = if self.msaa_samples > 1 {
            let t = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("model3d msaa"),
                size,
                mip_level_count: 1,
                sample_count: self.msaa_samples,
                dimension: wgpu::TextureDimension::D2,
                format: self.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            Some(t.create_view(&wgpu::TextureViewDescriptor::default()))
        } else {
            None
        };
    }

    fn pipeline(&mut self, key: PipelineKey) -> &wgpu::RenderPipeline {
        if !self.pipelines.contains_key(&key) {
            let layout = self
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("model3d layout"),
                    bind_group_layouts: &[
                        Some(&self.globals_bgl),
                        Some(&self.scene_bgl),
                        Some(&self.material_bgl),
                    ],
                    immediate_size: 0,
                });
            let vertex_buffers = [
                Some(wgpu::VertexBufferLayout {
                    array_stride: u64::try_from(std::mem::size_of::<Vertex>())
                        .expect("size fits u64"),
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x3, // position
                        1 => Float32x3, // normal
                        2 => Float32x4, // tangent (xyz + w sign)
                        3 => Float32x4, // color
                        4 => Float32x2, // uv0
                        5 => Float32x2, // uv1
                    ],
                }),
                Some(wgpu::VertexBufferLayout {
                    array_stride: u64::try_from(std::mem::size_of::<SkinVertex>())
                        .expect("size fits u64"),
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![
                        6 => Uint16x4,  // joints
                        7 => Float32x4, // weights
                    ],
                }),
            ];
            let blend = if key.blend {
                // Straight-alpha fragment output × src-alpha produces a
                // premultiplied target for the compositor.
                Some(wgpu::BlendState::ALPHA_BLENDING)
            } else {
                None
            };
            let p = self
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("model3d pipeline"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &self.shader,
                        entry_point: Some("vs_main"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &vertex_buffers,
                    },
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        front_face: if key.flipped_winding {
                            wgpu::FrontFace::Cw
                        } else {
                            wgpu::FrontFace::Ccw
                        },
                        cull_mode: if key.double_sided {
                            None
                        } else {
                            Some(wgpu::Face::Back)
                        },
                        ..Default::default()
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: DEPTH_FORMAT,
                        depth_write_enabled: Some(!key.blend),
                        depth_compare: Some(wgpu::CompareFunction::LessEqual),
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState {
                        count: self.msaa_samples,
                        ..Default::default()
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &self.shader,
                        entry_point: Some("fs_main"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: self.format,
                            blend,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                });
            self.pipelines.insert(key, p);
        }
        self.pipelines.get(&key).unwrap()
    }

    /// Records one frame into `encoder` covering `target` at `size`, drawing
    /// `params`' camera/lighting. The pose is sampled at
    /// `playback.time()` — the clock is owned UI-side; a handle that does not
    /// resolve against the uploaded model surfaces as
    /// [`crate::InvalidClipError`], never a silent retarget.
    ///
    /// # Errors
    /// [`crate::InvalidClipError`] if the playback clip does not belong to the
    /// uploaded model.
    pub fn draw(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        size: (u32, u32),
        params: &FrameParams,
        playback: &crate::playback::Playback,
    ) -> Result<DrawStats, crate::InvalidClipError> {
        self.ensure_targets(size.0, size.1);
        self.write_globals(params);

        // The scene moves out so its buffers can be borrowed alongside the
        // renderer's scratch, and moves back on every path: a scene that is
        // not restored reads as "not uploaded" and the caller re-uploads the
        // whole model on the next frame.
        let Some(mut uploaded) = self.scene.take() else {
            Self::record_clear_only(encoder, target, params.clear);
            return Ok(DrawStats::default());
        };
        let result = self.draw_uploaded(&mut uploaded, encoder, target, params, playback);
        self.scene = Some(uploaded);
        result
    }

    /// Draws the uploaded scene; [`Self::draw`] owns taking and restoring it.
    fn draw_uploaded(
        &mut self,
        uploaded: &mut UploadedScene,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        params: &FrameParams,
        playback: &crate::playback::Playback,
    ) -> Result<DrawStats, crate::InvalidClipError> {
        let mut stats = DrawStats::default();
        let view_proj = params.proj * params.view;
        let data = uploaded.model.data();
        let scene = &mut uploaded.scene;

        // Evaluate the pose at the UI-supplied playhead.
        scene.evaluate_pose(
            data,
            &uploaded.model,
            playback,
            &mut self.pose_scratch,
            &mut self.morph_weight_scratch,
        )?;

        // Cull, bucket, emit instances, upload scratch buffers, record.
        cull_draws(
            scene,
            data,
            Frustum::from(view_proj),
            params.view,
            &self.morph_weight_scratch,
            &mut stats,
        );
        bucket_draws(scene, data);
        emit_instances(scene, data, &mut stats);
        self.upload_frame_data(data, scene);
        self.record_pass(encoder, target, params, data, scene);
        Ok(stats)
    }

    /// Writes the frame-global uniform (camera matrices, lights, exposure).
    fn write_globals(&self, params: &FrameParams) {
        let view_proj = params.proj * params.view;
        let globals = GlobalsData {
            view_proj: view_proj.to_cols_array_2d(),
            view: params.view.to_cols_array_2d(),
            camera_pos: params.camera_pos.to_array(),
            light_count: if params.fill_dir == Vec3::ZERO { 1 } else { 2 },
            light_dir: params.light_dir.to_array(),
            exposure: params.exposure,
            light_color: (params.light_color * params.light_intensity).to_array(),
            _pad0: 0.0,
            fill_dir: params.fill_dir.to_array(),
            _pad1: 0.0,
            fill_color: params.fill_color.to_array(),
            _pad2: 0.0,
            ambient: params.ambient,
        };
        self.queue
            .write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&globals));
    }

    /// Uploads this frame's instance/joint/weight buffers.
    fn upload_frame_data(&self, data: &ModelData, scene: &mut GpuScene) {
        if !scene.instance_scratch.is_empty() {
            self.queue.write_buffer(
                &scene.instance_buf,
                0,
                bytemuck::cast_slice(&scene.instance_scratch),
            );
        }
        if !scene.joint_scratch.is_empty() {
            self.queue.write_buffer(
                &scene.joint_buf,
                0,
                bytemuck::cast_slice(&scene.joint_scratch),
            );
        }
        self.flush_weights(data, scene);
    }

    /// Records the render pass: shared bindings, instanced opaque spans,
    /// then sorted blend draws.
    fn record_pass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        params: &FrameParams,
        data: &ModelData,
        scene: &GpuScene,
    ) {
        let view_att: &wgpu::TextureView = self.msaa_view.as_ref().map_or(target, |ms| ms);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("model3d pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: view_att,
                resolve_target: if self.msaa_view.is_some() {
                    Some(target)
                } else {
                    None
                },
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(premultiplied(params.clear)),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: self.depth_view.as_ref().unwrap(),
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_bind_group(0, &self.globals_bg, &[]);
        pass.set_bind_group(1, &scene.scene_bind_group, &[]);
        pass.set_vertex_buffer(0, scene.vertex_buf.slice(..));
        pass.set_vertex_buffer(1, scene.skin_buf.slice(..));
        pass.set_index_buffer(scene.index_buf.slice(..), wgpu::IndexFormat::Uint32);

        for span in &scene.draw_spans {
            let pipeline = self.pipeline(span.key).clone();
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(2, &scene.materials[span.material], &[]);
            let (indices, base_vertex) = draw_range(&scene.primitives[span.primitive]);
            pass.draw_indexed(
                indices,
                base_vertex,
                span.first_instance..span.first_instance + span.count,
            );
        }
        for &(di, first, material) in &scene.blend_draws {
            let call = scene.draws[di as usize];
            let material = usize::try_from(material).expect("material index was u32");
            let world = scene.pose_globals[call.node];
            let key = PipelineKey {
                blend: true,
                double_sided: data.materials[material].double_sided,
                flipped_winding: Mat3::from_mat4(world).determinant() < 0.0,
            };
            let pipeline = self.pipeline(key).clone();
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(2, &scene.materials[material], &[]);
            let (indices, base_vertex) = draw_range(&scene.primitives[call.primitive]);
            pass.draw_indexed(indices, base_vertex, first..first + 1);
        }
    }

    fn record_clear_only(
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clear: [f32; 4],
    ) {
        let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("model3d clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(premultiplied(clear)),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    }

    fn flush_weights(&self, data: &ModelData, scene: &mut GpuScene) {
        // Write morph weights only for nodes whose current weights differ from
        // what we last uploaded (weights_uploaded seeded with NaN so the first
        // frame always writes).
        for (index, node) in data.nodes.iter().enumerate() {
            let Some(base) = node.weight_base(&data.meshes) else {
                continue;
            };
            let current = &self.morph_weight_scratch[index][..];
            let prior = &scene.weights_uploaded[base..base + current.len()];
            if prior
                .iter()
                .zip(current)
                .all(|(a, b)| a.to_bits() == b.to_bits())
            {
                continue;
            }
            self.queue.write_buffer(
                &scene.weight_buf,
                base as u64 * 4,
                bytemuck::cast_slice(current),
            );
            scene.weights_uploaded[base..base + current.len()].copy_from_slice(current);
        }
    }

    fn upload_scene(&self, data: &ModelData) -> GpuScene {
        let geometry = self.upload_geometry(data);
        let draws = collect_draws(data);
        let deform = self.deform_buffers(data, draws.len());
        let scene_bind_group = self.scene_bind_group(
            &deform.instance_buf,
            &deform.joint_buf,
            &deform.weight_buf,
            geometry.morph_buf.as_ref(),
        );
        let mut tex_cache: HashMap<(usize, bool), wgpu::TextureView> = HashMap::new();
        let mut sampler_cache: HashMap<usize, wgpu::Sampler> = HashMap::new();
        let gpu_textures = self.upload_textures(data, &mut tex_cache, &mut sampler_cache);
        let materials = data
            .materials
            .iter()
            .map(|mat| self.upload_material(data, mat, &tex_cache, &sampler_cache))
            .collect();

        GpuScene {
            textures: gpu_textures,
            vertex_buf: geometry.vertex_buf,
            index_buf: geometry.index_buf,
            skin_buf: geometry.skin_buf,
            _morph_buf: geometry.morph_buf,
            joint_buf: deform.joint_buf,
            weight_buf: deform.weight_buf,
            instance_buf: deform.instance_buf,
            scene_bind_group,
            primitives: geometry.primitives,
            draws,
            materials,
            pose_globals: vec![Mat4::IDENTITY; data.nodes.len()],
            joint_scratch: vec![[0.0; 16]; deform.joint_count],
            weights_uploaded: vec![f32::NAN; deform.weight_count.max(1)],
            draw_order: Vec::new(),
            draw_depths: Vec::new(),
            buckets: Vec::new(),
            bucket_pool: Vec::new(),
            draw_spans: Vec::new(),
            blend_spans: Vec::new(),
            blend_draws: Vec::new(),
            instance_scratch: Vec::new(),
        }
    }

    /// Packs every primitive's vertices/indices/skin attributes/morph deltas
    /// into shared device buffers and records each primitive's ranges.
    fn upload_geometry(&self, data: &ModelData) -> GpuGeometry {
        let (vertex_len, index_len, skin_len, morph_len) = geometry_sizes(data);

        let vertex_buf = make_buf(
            &self.device,
            "vertices",
            u64::from(u32::try_from(vertex_len).expect("vertex total fits u32"))
                * u64::try_from(std::mem::size_of::<Vertex>()).expect("size fits u64"),
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        );
        let index_buf = make_buf(
            &self.device,
            "indices",
            u64::from(u32::try_from(index_len).expect("index total fits u32")) * 4,
            wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        );
        // The skin vertex buffer is always full-size: unskinned vertices
        // carry the [0;4]/[1,0,0,0] identity attributes the loader emits.
        let skin_buf = make_buf(
            &self.device,
            "skin",
            u64::from(u32::try_from(skin_len).expect("skin total fits u32"))
                * u64::try_from(std::mem::size_of::<SkinVertex>()).expect("size fits u64"),
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        );
        let morph_buf = if data.has_morphs {
            Some(make_buf(
                &self.device,
                "morph",
                u64::from(u32::try_from(morph_len).expect("morph total fits u32"))
                    * std::mem::size_of::<MorphDelta>() as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ))
        } else {
            None
        };

        let mut primitives = Vec::with_capacity(data.primitives.len());
        let mut vertex_cursor = 0u64;
        let mut index_cursor = 0u64;
        let mut skin_cursor = 0u64;
        let mut morph_cursor = 0u32;
        for prim in &data.primitives {
            self.queue.write_buffer(
                &vertex_buf,
                vertex_cursor,
                bytemuck::cast_slice(&prim.vertices),
            );
            self.queue.write_buffer(
                &index_buf,
                index_cursor,
                bytemuck::cast_slice(&prim.indices),
            );
            self.queue
                .write_buffer(&skin_buf, skin_cursor, bytemuck::cast_slice(&prim.skin));
            if let Some(morph_buf) = &morph_buf {
                for deltas in &prim.morph_deltas {
                    self.queue.write_buffer(
                        morph_buf,
                        u64::from(morph_cursor) * std::mem::size_of::<MorphDelta>() as u64,
                        bytemuck::cast_slice(deltas),
                    );
                    morph_cursor += u32::try_from(deltas.len()).expect("target count fits u32");
                }
            }
            let prim_morph_offset = morph_cursor
                - prim
                    .morph_deltas
                    .iter()
                    .map(|t| u32::try_from(t.len()).expect("morph delta count fits u32"))
                    .sum::<u32>();
            primitives.push(GpuPrimitive {
                vertex_offset: vertex_cursor,
                vertex_count: u32::try_from(prim.vertices.len()).expect("vertex count fits u32"),
                index_offset: index_cursor,
                index_count: u32::try_from(prim.indices.len()).expect("index count fits u32"),
                morph_offset: prim_morph_offset,
                morph_count: u32::try_from(prim.morph_deltas.len()).expect("morph count fits u32"),
                material: prim.material,
                alpha: data.materials[prim.material].alpha,
                bounds_center: prim.bounds.center(),
                bounds_radius: prim.bounds.radius(),
                morph_max_displacement: prim.morph_max_displacement.clone(),
            });
            vertex_cursor += (prim.vertices.len() * std::mem::size_of::<Vertex>()) as u64;
            index_cursor += (prim.indices.len() * 4) as u64;
            skin_cursor += (prim.skin.len() * std::mem::size_of::<SkinVertex>()) as u64;
        }
        GpuGeometry {
            vertex_buf,
            index_buf,
            skin_buf,
            morph_buf,
            primitives,
        }
    }

    /// Joint-palette, morph-weight and instance storage buffers sized for
    /// the scene's deformation data.
    fn deform_buffers(&self, data: &ModelData, draw_count: usize) -> DeformBuffers {
        let joint_count: usize = data
            .nodes
            .iter()
            .filter_map(|n| n.skin.map(|s| data.skins[s].joints.len()))
            .sum();
        let joint_buf = storage_buf(&self.device, "joints", joint_count.max(1) as u64 * 64);
        let weight_count: usize = data.nodes.iter().map(|n| n.morph_count(&data.meshes)).sum();
        let weight_buf = storage_buf(&self.device, "weights", weight_count.max(1) as u64 * 4);
        let instance_buf = storage_buf(
            &self.device,
            "instances",
            u64::from(u32::try_from(draw_count.max(1)).expect("draw count fits u32"))
                * u64::try_from(std::mem::size_of::<InstanceData>()).expect("size"),
        );
        DeformBuffers {
            joint_buf,
            weight_buf,
            instance_buf,
            joint_count,
            weight_count,
        }
    }

    /// The scene bind group. Morph deltas need a bound buffer even for
    /// morph-less scenes; point the entry at the joint buffer (a valid
    /// storage buffer) — the shader only reads morphs when `morph_count > 0`.
    fn scene_bind_group(
        &self,
        instance_buf: &wgpu::Buffer,
        joint_buf: &wgpu::Buffer,
        weight_buf: &wgpu::Buffer,
        morph_buf: Option<&wgpu::Buffer>,
    ) -> wgpu::BindGroup {
        let morph_binding: wgpu::BindingResource<'_> =
            morph_buf.map_or_else(|| joint_buf.as_entire_binding(), |b| b.as_entire_binding());
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("model3d scene bg"),
            layout: &self.scene_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: instance_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: joint_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weight_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: morph_binding,
                },
            ],
        })
    }

    /// Uploads every texture referenced by a material slot, deduping views
    /// by (image, sRGB) and collecting the samplers each texture requests.
    fn upload_textures(
        &self,
        data: &ModelData,
        tex_cache: &mut HashMap<(usize, bool), wgpu::TextureView>,
        sampler_cache: &mut HashMap<usize, wgpu::Sampler>,
    ) -> Vec<GpuTexture> {
        let mut gpu_textures = Vec::new();
        for mat in &data.materials {
            for (slot, tex) in mat.texture_refs().iter().enumerate() {
                if let Some(t) = tex {
                    let key = (t.texture, SLOT_SRGB[slot]);
                    tex_cache.entry(key).or_insert_with(|| {
                        gpu_textures.push(self.upload_texture(data, t.texture, SLOT_SRGB[slot]));

                        gpu_textures.last().unwrap().view.clone()
                    });
                    let tex_data = &data.textures[t.texture];
                    if let Some(si) = tex_data.sampler {
                        sampler_cache
                            .entry(si)
                            .or_insert_with(|| make_sampler(&self.device, data.samplers[si]));
                    }
                }
            }
        }
        gpu_textures
    }

    fn upload_texture(&self, data: &ModelData, index: usize, srgb: bool) -> GpuTexture {
        let t = &data.textures[index];
        let mips = 32 - t.width.max(t.height).leading_zeros();
        let format = if srgb {
            wgpu::TextureFormat::Rgba8UnormSrgb
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("model3d tex"),
            size: wgpu::Extent3d {
                width: t.width,
                height: t.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &t.pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(t.width * 4),
                rows_per_image: Some(t.height),
            },
            wgpu::Extent3d {
                width: t.width,
                height: t.height,
                depth_or_array_layers: 1,
            },
        );
        // Generate mips by successive half-size blits. Views use the texture's
        // native format: for sRGB textures the sample decodes to linear and
        // the store encodes back; for linear textures both are raw. The
        // per-encoding pipeline matches the render-target format.
        let mip_pipeline = if srgb {
            &self.mip_pipeline_srgb
        } else {
            &self.mip_pipeline_lin
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        for level in 1..mips {
            let src = texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: level - 1,
                mip_level_count: Some(1),
                format: Some(format),
                ..Default::default()
            });
            let dst = texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: level,
                mip_level_count: Some(1),
                format: Some(format),
                ..Default::default()
            });
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mip bg"),
                layout: &mip_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&src),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.default_sampler),
                    },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mip blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dst,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(mip_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([encoder.finish()]);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        GpuTexture { texture, view }
    }

    fn upload_material(
        &self,
        data: &ModelData,
        mat: &crate::model::MaterialData,
        tex_cache: &HashMap<(usize, bool), wgpu::TextureView>,
        sampler_cache: &HashMap<usize, wgpu::Sampler>,
    ) -> wgpu::BindGroup {
        let mut slots_a = [[0.0f32; 4]; TEXTURE_SLOTS];
        let mut slots_b = [[0.0f32; 4]; TEXTURE_SLOTS];
        for (i, t) in mat.texture_refs().iter().enumerate() {
            if let Some(t) = t {
                slots_a[i] = [
                    f32::from(u8::try_from(t.uv_index).expect("uv_index ≤ 1")),
                    t.offset[0],
                    t.offset[1],
                    t.rotation,
                ];
                slots_b[i] = [t.scale[0], t.scale[1], 0.0, 0.0];
            }
        }
        // Slots with no texture still need an identity transform so the
        // fallback sample is undisturbed.
        for (i, t) in mat.texture_refs().iter().enumerate() {
            if t.is_none() {
                slots_a[i] = [0.0, 0.0, 0.0, 0.0];
                slots_b[i] = [1.0, 1.0, 0.0, 0.0];
            }
        }
        let uniform = MaterialUniform {
            base_color_factor: mat.base_color,
            emissive_factor: [
                mat.emissive_factor[0] * mat.emissive_strength,
                mat.emissive_factor[1] * mat.emissive_strength,
                mat.emissive_factor[2] * mat.emissive_strength,
                1.0,
            ],
            metallic_factor: mat.metallic,
            roughness_factor: mat.roughness,
            normal_scale: mat.normal_scale,
            occlusion_strength: mat.occlusion_strength,
            alpha_cutoff: if mat.alpha == AlphaMode::Mask {
                mat.alpha_cutoff
            } else {
                0.0
            },
            unlit: u32::from(mat.unlit),
            alpha_mode: match mat.alpha {
                AlphaMode::Opaque => 0,
                AlphaMode::Mask => 1,
                AlphaMode::Blend => 2,
            },
            double_sided: u32::from(mat.double_sided),
            slots_a,
            slots_b,
        };
        let buf = make_buf(
            &self.device,
            "mat",
            std::mem::size_of::<MaterialUniform>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        self.queue
            .write_buffer(&buf, 0, bytemuck::bytes_of(&uniform));

        let fallbacks: [&wgpu::TextureView; TEXTURE_SLOTS] = [
            &self.white_srgb,
            &self.white_lin,
            &self.neutral_normal,
            &self.white_lin,
            // No emissive texture means "sample is 1" — the factor (default
            // zero) alone decides emission.
            &self.white_srgb,
        ];
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: buf.as_entire_binding(),
        }];
        for (slot, t) in mat.texture_refs().iter().enumerate() {
            let view = t
                .and_then(|t| tex_cache.get(&(t.texture, SLOT_SRGB[slot])))
                .unwrap_or(fallbacks[slot]);
            let sampler = t
                .and_then(|t| data.textures[t.texture].sampler)
                .and_then(|si| sampler_cache.get(&si))
                .unwrap_or(&self.default_sampler);
            entries.push(wgpu::BindGroupEntry {
                binding: u32::try_from(slot * 2 + 1).expect("binding index fits u32"),
                resource: wgpu::BindingResource::TextureView(view),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: u32::try_from(slot * 2 + 2).expect("binding index fits u32"),
                resource: wgpu::BindingResource::Sampler(sampler),
            });
        }
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("model3d material bg"),
            layout: &self.material_bgl,
            entries: &entries,
        })
    }
}

// ---------------------------------------------------------------------------

/// Bind-group layouts in pipeline slot order: globals, scene, material.
fn bind_group_layouts(
    device: &wgpu::Device,
) -> (
    wgpu::BindGroupLayout,
    wgpu::BindGroupLayout,
    wgpu::BindGroupLayout,
) {
    let globals_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("model3d globals"),
        entries: &[uniform_entry(0)],
    });
    let scene_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("model3d scene"),
        entries: &[
            storage_entry(0), // instances
            storage_entry(1), // joint palettes
            storage_entry(2), // morph weights
            storage_entry(3), // morph deltas
        ],
    });
    let material_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("model3d material"),
        entries: &[
            uniform_entry(0),
            texture_entry(1),
            sampler_entry(2),
            texture_entry(3),
            sampler_entry(4),
            texture_entry(5),
            sampler_entry(6),
            texture_entry(7),
            sampler_entry(8),
            texture_entry(9),
            sampler_entry(10),
        ],
    });
    (globals_bgl, scene_bgl, material_bgl)
}

/// The frame-global uniform buffer and its bind group.
fn globals_binding(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> (wgpu::Buffer, wgpu::BindGroup) {
    let buf = make_buf(
        device,
        "globals",
        u64::try_from(std::mem::size_of::<GlobalsData>()).expect("size fits u64"),
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("model3d globals bg"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buf.as_entire_binding(),
        }],
    });
    (buf, bg)
}

/// A 1×1 flat-colored texture standing in for an absent material slot. The
/// returned view keeps the texture alive.
fn fallback_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    rgba: [u8; 4],
    srgb: bool,
) -> wgpu::TextureView {
    let t = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: if srgb {
            wgpu::TextureFormat::Rgba8UnormSrgb
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        },
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &t,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    t.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Mipmap generation blit pipelines, (linear, sRGB): sampling an sRGB-format
/// view decodes to linear in the shader and rendering into an sRGB-format
/// view encodes on store, so the chain filters encoded values correctly end
/// to end.
fn mip_pipelines(device: &wgpu::Device) -> (wgpu::RenderPipeline, wgpu::RenderPipeline) {
    let mip_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("model3d mip blit"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mip_blit.wgsl").into()),
    });
    let mip_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("model3d mip"),
        entries: &[texture_entry(0), sampler_entry(1)],
    });
    let mip_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("model3d mip layout"),
        bind_group_layouts: &[Some(&mip_bgl)],
        immediate_size: 0,
    });
    let mip_pipeline_for = |format: wgpu::TextureFormat| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("model3d mip pipe"),
            layout: Some(&mip_layout),
            vertex: wgpu::VertexState {
                module: &mip_shader,
                entry_point: Some("vs_mip"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &mip_shader,
                entry_point: Some("fs_mip"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        })
    };
    (
        mip_pipeline_for(wgpu::TextureFormat::Rgba8Unorm),
        mip_pipeline_for(wgpu::TextureFormat::Rgba8UnormSrgb),
    )
}

/// Premultiplies a straight-alpha linear color for the composited target.
fn premultiplied(clear: [f32; 4]) -> wgpu::Color {
    wgpu::Color {
        r: f64::from(clear[0] * clear[3]),
        g: f64::from(clear[1] * clear[3]),
        b: f64::from(clear[2] * clear[3]),
        a: f64::from(clear[3]),
    }
}

const fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
const fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
const fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}
const fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn make_buf(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(4),
        usage,
        mapped_at_creation: false,
    })
}
fn storage_buf(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    make_buf(
        device,
        label,
        size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    )
}

fn make_sampler(device: &wgpu::Device, s: crate::model::SamplerData) -> wgpu::Sampler {
    let addr = |m: crate::model::WrapModeData| match m {
        crate::model::WrapModeData::Clamp => wgpu::AddressMode::ClampToEdge,
        crate::model::WrapModeData::Repeat => wgpu::AddressMode::Repeat,
        crate::model::WrapModeData::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
    };
    let filt = |m: Option<crate::model::FilterModeData>| match m {
        Some(crate::model::FilterModeData::Nearest) => wgpu::FilterMode::Nearest,
        _ => wgpu::FilterMode::Linear,
    };
    let mip = match s.mipmap {
        crate::model::FilterModeData::Nearest => wgpu::MipmapFilterMode::Nearest,
        crate::model::FilterModeData::Linear => wgpu::MipmapFilterMode::Linear,
    };
    let has_mips = s.has_mipmaps;
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("model3d sampler"),
        address_mode_u: addr(s.wrap_s),
        address_mode_v: addr(s.wrap_t),
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: filt(s.mag),
        min_filter: filt(s.min),
        mipmap_filter: mip,
        lod_min_clamp: 0.0,
        lod_max_clamp: if has_mips { f32::MAX } else { 0.0 },
        ..Default::default()
    })
}

/// Element totals across all primitives: (vertices, indices, skin, morph).
fn geometry_sizes(data: &ModelData) -> (usize, usize, usize, usize) {
    (
        data.primitives.iter().map(|p| p.vertices.len()).sum(),
        data.primitives.iter().map(|p| p.indices.len()).sum(),
        data.primitives.iter().map(|p| p.skin.len()).sum(),
        data.primitives
            .iter()
            .flat_map(|p| p.morph_deltas.iter())
            .map(Vec::len)
            .sum(),
    )
}

/// One [`DrawCall`] per (node, primitive) of the selected scene.
fn collect_draws(data: &ModelData) -> Vec<DrawCall> {
    let mut draws = Vec::new();
    for (i, node) in data.nodes.iter().enumerate() {
        for &p in &node.draws {
            draws.push(DrawCall {
                node: i,
                primitive: p,
            });
        }
    }
    draws
}

/// Frustum-culls every draw, recording surviving draw indices and their
/// view-space depths for blend sorting.
fn cull_draws(
    scene: &mut GpuScene,
    data: &ModelData,
    frustum: Frustum,
    view: Mat4,
    morph_weights: &[Vec<f32>],
    stats: &mut DrawStats,
) {
    scene.draw_order.clear();
    scene.draw_depths.clear();
    for (draw_index, call) in scene.draws.iter().enumerate() {
        let prim = &scene.primitives[call.primitive];
        let world = scene.pose_globals[call.node];
        let bound = draw_bound(
            prim,
            world,
            data,
            call.node,
            &scene.pose_globals,
            &morph_weights[call.node],
        );
        let Some((center, radius)) = bound else {
            continue;
        };
        if frustum.contains_sphere(center, radius) {
            scene
                .draw_order
                .push(u32::try_from(draw_index).expect("draw index fits u32"));
            scene.draw_depths.push((view * center.extend(1.0)).z);
        } else {
            stats.culled += 1;
        }
    }
}

/// Opaque/mask draws bucket by (primitive, pipeline key) for instancing;
/// blend draws sort far-to-near (view z is negative-forward, so ascending
/// is farthest first).
fn bucket_draws(scene: &mut GpuScene, data: &ModelData) {
    for mut bucket in scene.buckets.drain(..) {
        bucket.members.clear();
        scene.bucket_pool.push(bucket.members);
    }
    scene.blend_spans.clear();
    for (slot, &di) in scene.draw_order.iter().enumerate() {
        let call = scene.draws[di as usize];
        let prim = &scene.primitives[call.primitive];
        let world = scene.pose_globals[call.node];
        let key = PipelineKey {
            blend: prim.alpha == AlphaMode::Blend,
            double_sided: data.materials[prim.material].double_sided,
            flipped_winding: Mat3::from_mat4(world).determinant() < 0.0,
        };
        if key.blend {
            scene.blend_spans.push((
                scene.draw_depths[slot],
                di,
                u32::try_from(prim.material).expect("material index fits u32"),
            ));
        } else {
            let bucket = scene
                .buckets
                .iter_mut()
                .find(|b| b.primitive == call.primitive && b.key == key);
            if let Some(b) = bucket {
                b.members.push(di);
            } else {
                // A pooled list is empty; the draw that opens the bucket is
                // its first member.
                let mut members = scene.bucket_pool.pop().unwrap_or_default();
                members.push(di);
                scene.buckets.push(OpaqueBucket {
                    primitive: call.primitive,
                    key,
                    material: prim.material,
                    members,
                });
            }
        }
    }
    // View-space z is negative-forward: ascending = farthest first.
    scene
        .blend_spans
        .sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
}

/// Emits instance records: opaque groups first (instanced), then one record
/// per blend draw in sorted order.
fn emit_instances(scene: &mut GpuScene, data: &ModelData, stats: &mut DrawStats) {
    scene.instance_scratch.clear();
    scene.draw_spans.clear();
    for bucket in &scene.buckets {
        let first = u32::try_from(scene.instance_scratch.len()).expect("instance count fits u32");
        for &di in &bucket.members {
            let call = scene.draws[di as usize];
            scene
                .instance_scratch
                .push(make_instance(data, scene, call));
        }
        scene.draw_spans.push(DrawSpan {
            first_instance: first,
            count: u32::try_from(bucket.members.len()).expect("bucket size fits u32"),
            key: bucket.key,
            material: bucket.material,
            primitive: bucket.primitive,
        });
        stats.draws += 1;
        stats.instances += u32::try_from(bucket.members.len()).expect("bucket size fits u32");
    }
    scene.blend_draws.clear();
    for &(_, di, material) in &scene.blend_spans {
        let call = scene.draws[di as usize];
        let first = u32::try_from(scene.instance_scratch.len()).expect("instance count fits u32");
        scene
            .instance_scratch
            .push(make_instance(data, scene, call));
        scene.blend_draws.push((di, first, material));
        stats.draws += 1;
        stats.instances += 1;
    }
}

/// Indexed-draw parameters for `prim`: index range and base vertex.
fn draw_range(prim: &GpuPrimitive) -> (std::ops::Range<u32>, i32) {
    let index_start = u32::try_from(prim.index_offset / 4).expect("index offset fits u32");
    let base_vertex = i32::try_from(
        prim.vertex_offset / u64::try_from(std::mem::size_of::<Vertex>()).expect("size fits u64"),
    )
    .expect("vertex offset fits i32");
    (index_start..index_start + prim.index_count, base_vertex)
}

fn make_instance(data: &ModelData, scene: &GpuScene, call: DrawCall) -> InstanceData {
    let prim = &scene.primitives[call.primitive];
    let node = &data.nodes[call.node];
    let world = scene.pose_globals[call.node];
    InstanceData {
        world: world.to_cols_array_2d(),
        normal: normal_matrix(world).to_cols_array_2d(),
        joint_base: node.joint_base,
        weight_base: u32::try_from(node.weight_base(&data.meshes).unwrap_or(0))
            .expect("weight base fits u32"),
        morph_offset: prim.morph_offset,
        morph_count: prim.morph_count,
        vertex_base: u32::try_from(
            prim.vertex_offset
                / u64::try_from(std::mem::size_of::<Vertex>()).expect("size fits u64"),
        )
        .expect("vertex base fits u32"),
        vertex_count: prim.vertex_count,
        joint_count: node.joint_count(data),
        _pad: 0,
    }
}

/// Inverse-transpose of the world matrix's linear part, for normals.
fn normal_matrix(world: Mat4) -> Mat4 {
    Mat4::from_mat3(Mat3::from_mat4(world).inverse().transpose())
}

/// Conservative spectral-norm bound of a matrix's linear part:
/// `sqrt(‖A‖₁ · ‖A‖∞)` — max column absolute-sum × max row absolute-sum —
/// which stays sound under shear (nested non-uniform scale + rotation
/// produces shear even when every local node transform is TRS).
fn linear_extent(world: Mat4) -> f32 {
    let m = Mat3::from_mat4(world);
    let r = m.to_cols_array();
    // r is column-major: r[0..3] column 0, r[3..6] column 1, r[6..9] column 2.
    let col = |i: usize| r[i * 3].abs() + r[i * 3 + 1].abs() + r[i * 3 + 2].abs();
    let row = |i: usize| r[i].abs() + r[i + 3].abs() + r[i + 6].abs();
    let norm1 = col(0).max(col(1)).max(col(2));
    let norm_inf = row(0).max(row(1)).max(row(2));
    (norm1 * norm_inf).sqrt()
}

/// Conservative world-space bound for a draw. Unskinned: the primitive's
/// local sphere transformed by the node matrix. Skinned: a sphere covering
/// the per-joint transformed centers plus the skinned radius. Both expand by
/// the morph displacement budget.
fn draw_bound(
    prim: &GpuPrimitive,
    world: Mat4,
    data: &ModelData,
    node_index: usize,
    pose_globals: &[Mat4],
    morph_weights: &[f32],
) -> Option<(Vec3, f32)> {
    if prim.vertex_count == 0 {
        return None;
    }
    // Mesh-space morph expansion: Σ |weight_i| · maxdelta_i — weights may
    // exceed 1 or go negative, so the sum of absolutes is the bound.
    let morph_expand: f32 = morph_weights
        .iter()
        .zip(prim.morph_max_displacement.iter())
        .map(|(w, d)| w.abs() * d)
        .sum();
    let node = &data.nodes[node_index];
    if let Some(skin_index) = node.skin {
        let skin = &data.skins[skin_index];
        // Each joint maps the mesh-space bound through `joint_world * ibm`.
        // The drawn sphere covers the spread of joint-transformed centers
        // plus the primitive radius at the largest joint scale.
        let mut max_scale = 0.0f32;
        let mut joint_centers = Vec3::ZERO;
        let mut count = 0u32;
        for (i, &j) in skin.joints.iter().enumerate() {
            let ibm = skin
                .inverse_bind_matrices
                .get(i)
                .copied()
                .unwrap_or(Mat4::IDENTITY);
            let t = pose_globals[j] * ibm;
            joint_centers += t.transform_point3(prim.bounds_center);
            max_scale = max_scale.max(linear_extent(t));
            count += 1;
        }
        if count > 0 {
            let center =
                joint_centers / f32::from(u16::try_from(count).expect("joint count fits u16"));
            let mut radius = prim.bounds_radius * max_scale;
            for (i, &j) in skin.joints.iter().enumerate() {
                let ibm = skin
                    .inverse_bind_matrices
                    .get(i)
                    .copied()
                    .unwrap_or(Mat4::IDENTITY);
                let t = pose_globals[j] * ibm;
                radius = radius.max(prim.bounds_radius.mul_add(
                    linear_extent(t),
                    t.transform_point3(prim.bounds_center).distance(center),
                ));
            }
            return Some((center, morph_expand.mul_add(max_scale.max(1.0), radius)));
        }
    }
    let center = world.transform_point3(prim.bounds_center);
    let extent = linear_extent(world);
    let radius = (prim.bounds_radius + morph_expand) * extent;
    Some((center, radius.max(0.0)))
}

/// Six-plane frustum extracted from a view-projection matrix
/// (Gribb/Hartmann), with planes normalized so `contains_sphere` compares
/// true distances.
#[derive(Debug, Clone, Copy)]
struct Frustum {
    planes: [Vec4; 6],
}

impl Frustum {
    fn from(m: Mat4) -> Self {
        let r = m.to_cols_array_2d();
        let row = |i: usize| Vec4::new(r[0][i], r[1][i], r[2][i], r[3][i]);
        let r0 = row(0);
        let r1 = row(1);
        let r2 = row(2);
        let r3 = row(3);
        let normalize = |p: Vec4| {
            let n = p.truncate().length();
            if n > 0.0 { p / n } else { p }
        };
        Self {
            planes: [
                normalize(r3 + r0),
                normalize(r3 - r0),
                normalize(r3 + r1),
                normalize(r3 - r1),
                normalize(r2),      // near: clip z in [0, w] for wgpu
                normalize(r3 - r2), // far
            ],
        }
    }
    fn contains_sphere(&self, center: Vec3, radius: f32) -> bool {
        self.planes
            .iter()
            .all(|p| p.truncate().dot(center) + p.w >= -radius)
    }
}

impl GpuScene {
    /// Evaluates the pose at `playback`'s current time: resets locals to the
    /// rest TRS and morph weights to the authored defaults — in place, no
    /// per-frame allocation — applies the selected clip when it validates
    /// against `model`, then recomputes every node's world transform (all
    /// loaded roots, not only the selected scene, so out-of-scene skeleton
    /// joints resolve) and each skinned node's joint palette
    /// `inv(meshWorld) * jointWorld * ibm`.
    ///
    /// # Errors
    /// [`crate::InvalidClipError`] when the selected clip doesn't resolve
    /// against this model.
    fn evaluate_pose(
        &mut self,
        data: &ModelData,
        model: &Model,
        playback: &crate::playback::Playback,
        pose: &mut Vec<Trs>,
        weights: &mut Vec<Vec<f32>>,
    ) -> Result<(), crate::InvalidClipError> {
        pose.resize(data.nodes.len(), Trs::IDENTITY);
        for (p, node) in pose.iter_mut().zip(data.nodes.iter()) {
            *p = Trs {
                translation: node.translation,
                rotation: node.rotation,
                scale: node.scale,
            };
        }
        weights.resize(data.nodes.len(), Vec::new());
        for (w, node) in weights.iter_mut().zip(data.nodes.iter()) {
            w.clone_from(&node.morph_weights);
        }
        if let Some(handle) = playback.clip() {
            let clip = model.clip(handle)?;
            let t = playback.time().min(clip.duration());
            anim::evaluate_clip(clip, t.as_secs_f32(), pose, weights);
        }
        compute_global_transforms(&data.nodes, pose, &mut self.pose_globals);

        // Joint palettes: inv(meshWorld) * jointWorld * IBM, one palette per
        // skinned node (a shared skin still needs a distinct palette per
        // mesh node because inv(meshWorld) differs).
        self.joint_scratch.fill([0.0; 16]);
        for (index, node) in data.nodes.iter().enumerate() {
            let Some(skin_index) = node.skin else {
                continue;
            };
            let skin = &data.skins[skin_index];
            let inv_mesh = self.pose_globals[index].inverse();
            let base = node.joint_base as usize;
            for (i, &j) in skin.joints.iter().enumerate() {
                let palette = inv_mesh
                    * self.pose_globals[j]
                    * skin
                        .inverse_bind_matrices
                        .get(i)
                        .copied()
                        .unwrap_or(Mat4::IDENTITY);
                self.joint_scratch[base + i] = palette.to_cols_array();
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use glam::Vec3;
    use waterui_graphics::wgpu;

    use super::{FrameParams, Renderer};
    use crate::{Model, OrbitCamera, Playback};

    /// A 10×10×10 grid of nodes that all reference mesh 0 (one triangle).
    fn shared_mesh_grid() -> Vec<u8> {
        let nodes: Vec<_> = (0..1000)
            .map(|index| {
                let (x, y, z) = (index % 10, index / 10 % 10, index / 100);
                serde_json::json!({"mesh": 0, "translation": [x * 2, y * 2, z * 2]})
            })
            .collect();
        let document = serde_json::json!({
            "asset": {"version": "2.0"}, "scene": 0,
            "scenes": [{"nodes": (0..1000).collect::<Vec<_>>()}],
            "nodes": nodes,
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
            "buffers": [{"byteLength": 36}],
            "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36}],
            "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3,
                           "type": "VEC3", "min": [0, 0, 0], "max": [1, 1, 0]}]
        });
        let mut json = serde_json::to_vec(&document).expect("serialize fixture");
        while !json.len().is_multiple_of(4) {
            json.push(b' ');
        }
        let binary: Vec<u8> = [0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let chunk = |bytes: &[u8], kind: u32| {
            let mut out = u32::try_from(bytes.len()).unwrap().to_le_bytes().to_vec();
            out.extend_from_slice(&kind.to_le_bytes());
            out.extend_from_slice(bytes);
            out
        };
        let body = [chunk(&json, 0x4e4f_534a), chunk(&binary, 0x004e_4942)].concat();
        let total = u32::try_from(12 + body.len()).unwrap();
        [0x4654_6c67_u32, 2, total]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .chain(body)
            .collect()
    }

    #[test]
    #[ignore = "requires a hardware GPU; run explicitly with --ignored"]
    fn shared_mesh_instances_upload_once_and_batch_into_one_draw() {
        let model = pollster::block_on(Model::from_glb(shared_mesh_grid())).unwrap();
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
        let mut renderer = Renderer::new(&device, &queue, format, 1);
        renderer.upload(&model);
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("model3d-instancing-test"),
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
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let camera = OrbitCamera::fit(&model.bounds(), 1.0).expect("grid has bounds");
        let params = FrameParams {
            view: camera.view(),
            proj: camera.projection(1.0),
            camera_pos: camera.eye(),
            light_dir: Vec3::NEG_Y,
            light_color: Vec3::ONE,
            light_intensity: 1.0,
            fill_dir: Vec3::ZERO,
            fill_color: Vec3::ZERO,
            ambient: [1.0, 1.0, 1.0, 0.1],
            exposure: 1.0,
            clear: [0.0; 4],
        };
        let mut frame = || {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            let stats = renderer
                .draw(&mut encoder, &view, (256, 256), &params, &Playback::new())
                .unwrap();
            queue.submit([encoder.finish()]);
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(Duration::from_secs(10)),
                })
                .unwrap();
            stats
        };
        frame();
        let stats = frame();
        assert!(
            renderer.contains(&model),
            "drawing keeps the uploaded scene; losing it re-uploads every frame"
        );
        let scene = &renderer.scene.as_ref().expect("uploaded").scene;
        assert_eq!(scene.primitives.len(), 1, "shared mesh uploaded once");
        assert_eq!(stats.culled, 0, "fitted camera sees every instance");
        assert_eq!(stats.draws, 1, "shared opaque mesh batched");
        assert_eq!(stats.instances, 1000);
    }
}
