//! [`Model`]: the immutable, cheaply shareable loaded glTF asset.

pub mod anim;
pub mod loader;

use std::path::Path;
use std::sync::{Arc, Weak};
use std::time::Duration;

use glam::{Mat4, Quat, Vec3};

use crate::error::{InvalidClipError, LoadError};

pub use anim::Trs;
pub use loader::{compute_global_transforms, load_from_parts};

/// Axis-aligned bounding box of static geometry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    /// Minimum corner.
    pub min: Vec3,
    /// Maximum corner.
    pub max: Vec3,
}

impl Bounds {
    /// Empty (inverted) bounds; `is_valid` is `false` until points are added.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            min: Vec3::INFINITY,
            max: Vec3::NEG_INFINITY,
        }
    }

    /// Expands to include `point`.
    pub fn include(&mut self, point: Vec3) {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
    }

    /// Expands to include `other`.
    pub fn union(&mut self, other: &Self) {
        if other.is_valid() {
            self.include(other.min);
            self.include(other.max);
        }
    }

    /// This box transformed by `m` (re-bounded axis-aligned).
    #[must_use]
    pub fn transformed(&self, m: &Mat4) -> Self {
        if !self.is_valid() {
            return Self::empty();
        }
        let mut out = Self::empty();
        for i in 0..8 {
            let corner = Vec3::new(
                if i & 1 == 0 { self.min.x } else { self.max.x },
                if i & 2 == 0 { self.min.y } else { self.max.y },
                if i & 4 == 0 { self.min.z } else { self.max.z },
            );
            out.include(m.transform_point3(corner));
        }
        out
    }

    /// Box center; `Vec3::ZERO` for empty bounds.
    #[must_use]
    pub fn center(&self) -> Vec3 {
        if self.is_valid() {
            (self.min + self.max) * 0.5
        } else {
            Vec3::ZERO
        }
    }

    /// Radius of the smallest sphere containing the box; 0 when empty.
    #[must_use]
    pub fn radius(&self) -> f32 {
        if self.is_valid() {
            (self.max - self.min).length() * 0.5
        } else {
            0.0
        }
    }

    /// Whether any point has been included.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.min.x <= self.max.x
            && self.min.y <= self.max.y
            && self.min.z <= self.max.z
            && self.min.is_finite()
            && self.max.is_finite()
    }
}

/// Interleaved position/normal/tangent/color/uv vertex (72 bytes).
///
/// Colors keep full `f32` precision: glTF `COLOR_0` may be `VEC3`, `VEC4`,
/// normalized u8/u16, or float, and values above 1.0 are meaningful for HDR
/// content.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    /// glTF `POSITION`.
    pub position: [f32; 3],
    /// `NORMAL` (flat-shaded normal synthesized when absent).
    pub normal: [f32; 3],
    /// `TANGENT` or mikktspace-generated `VEC4` with handedness in `.w`.
    pub tangent: [f32; 4],
    /// `COLOR_0` in float RGBA; defaults to white.
    pub color: [f32; 4],
    /// `TEXCOORD_0`.
    pub uv0: [f32; 2],
    /// `TEXCOORD_1`.
    pub uv1: [f32; 2],
}

/// Per-vertex skinning attributes (24 bytes). `u16` joint indices cover the
/// full glTF `JOINTS_0` component range — no clamping.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkinVertex {
    /// `JOINTS_0` (u16 components, zeroed when unskinned).
    pub joints: [u16; 4],
    /// `WEIGHTS_0` normalized to sum to 1; `[1,0,0,0]` when unskinned so the
    /// shader's skinning path degenerates to identity.
    pub weights: [f32; 4],
}

/// Position/normal/tangent deltas for one morph target (36 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MorphDelta {
    /// Position delta (xyz; w unused — storage vec3 alignment).
    pub position: [f32; 4],
    /// Normal delta (xyz; w unused).
    pub normal: [f32; 4],
    /// Tangent (xyz) delta (w unused).
    pub tangent: [f32; 4],
}

/// glTF alpha modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlphaMode {
    /// Depth-tested opaque.
    Opaque,
    /// Depth-tested with alpha discard at the material's cutoff.
    Mask,
    /// Alpha-blended, sorted back-to-front.
    Blend,
}

/// A decoded texture image ready for upload.
#[derive(Debug)]
pub struct TextureData {
    /// RGBA8 pixels.
    pub pixels: Vec<u8>,
    /// Pixel width.
    pub width: u32,
    /// Pixel height.
    pub height: u32,
    /// Sampler index into `samplers`, or `None` for the default.
    pub sampler: Option<usize>,
}

/// Sampler wrap modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WrapModeData {
    /// Clamp to edge (33071).
    Clamp,
    /// Repeat (10497).
    Repeat,
    /// Mirrored repeat (33648).
    MirroredRepeat,
}

/// Sampler min/mag filter modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterModeData {
    /// Nearest texel (9728).
    Nearest,
    /// Bilinear (9729).
    Linear,
}

/// A texture sampler (wrap/filter).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplerData {
    /// S wrap.
    pub wrap_s: WrapModeData,
    /// T wrap.
    pub wrap_t: WrapModeData,
    /// Magnification filter; `None` = implementation default (linear).
    pub mag: Option<FilterModeData>,
    /// Minification filter; `None` = implementation default (linear + mipmap).
    pub min: Option<FilterModeData>,
    /// Mipmap filter implied by the min filter.
    pub mipmap: FilterModeData,
    /// Whether the min filter samples mipmaps (gltf 9728/9729 do not).
    pub has_mipmaps: bool,
}

/// A texture reference: which texture, which UV set, which transform.
#[derive(Debug, Clone, Copy)]
pub struct TextureRef {
    /// Index into `ModelData::textures`.
    pub texture: usize,
    /// `texCoord` UV set: 0 or 1.
    pub uv_index: u32,
    /// `KHR_texture_transform` `offset` (uv).
    pub offset: [f32; 2],
    /// `KHR_texture_transform` `rotation` (radians).
    pub rotation: f32,
    /// `KHR_texture_transform` `scale`.
    pub scale: [f32; 2],
}

/// Metallic-roughness material with the optional texture set.
#[derive(Debug)]
pub struct MaterialData {
    /// `baseColorFactor`.
    pub base_color: [f32; 4],
    /// `metallicFactor`.
    pub metallic: f32,
    /// `roughnessFactor`.
    pub roughness: f32,
    /// Base color texture (sRGB when uploaded).
    pub base_color_tex: Option<TextureRef>,
    /// Metallic-roughness texture (linear; B=metallic, G=roughness).
    pub metallic_roughness_tex: Option<TextureRef>,
    /// Normal map (linear).
    pub normal_tex: Option<TextureRef>,
    /// `NormalTexture.scale`.
    pub normal_scale: f32,
    /// Occlusion texture (linear; R channel).
    pub occlusion_tex: Option<TextureRef>,
    /// `OcclusionTexture.strength`.
    pub occlusion_strength: f32,
    /// Emissive texture (sRGB).
    pub emissive_tex: Option<TextureRef>,
    /// `emissiveFactor`.
    pub emissive_factor: [f32; 3],
    /// `KHR_materials_emissive_strength`.
    pub emissive_strength: f32,
    /// `alphaMode`.
    pub alpha: AlphaMode,
    /// `alphaCutoff` (only meaningful for `Mask`).
    pub alpha_cutoff: f32,
    /// `doubleSided`.
    pub double_sided: bool,
    /// `KHR_materials_unlit`.
    pub unlit: bool,
}

impl MaterialData {
    /// The glTF default material: opaque, white, metallic 1, roughness 1.
    pub const fn default_gltf() -> Self {
        Self {
            base_color: [1.0; 4],
            metallic: 1.0,
            roughness: 1.0,
            base_color_tex: None,
            metallic_roughness_tex: None,
            normal_tex: None,
            normal_scale: 1.0,
            occlusion_tex: None,
            occlusion_strength: 1.0,
            emissive_tex: None,
            emissive_factor: [0.0; 3],
            emissive_strength: 1.0,
            alpha: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            double_sided: false,
            unlit: false,
        }
    }

    /// Texture references in shader slot order: base color, metallic-
    /// roughness, normal, occlusion, emissive.
    pub const fn texture_refs(&self) -> [Option<TextureRef>; 5] {
        [
            self.base_color_tex,
            self.metallic_roughness_tex,
            self.normal_tex,
            self.occlusion_tex,
            self.emissive_tex,
        ]
    }
}

/// Where an animation channel writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimPath {
    /// `translation`.
    Translation,
    /// `rotation`.
    Rotation,
    /// `scale`.
    Scale,
    /// `weights` (morph targets).
    MorphWeights,
}

/// Interpolation between keyframes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationData {
    /// Hold the key value.
    Step,
    /// Linear (slerp for rotations).
    Linear,
    /// Cubic Hermite spline (in-tangent, value, out-tangent triples).
    CubicSpline,
}

/// Keyframe channel payload.
#[derive(Debug)]
pub struct ChannelData {
    /// Target node index.
    pub node: usize,
    /// Target property.
    pub path: AnimPath,
    /// Interpolation.
    pub interpolation: InterpolationData,
    /// Keyframe times in seconds.
    pub times: Vec<f32>,
    /// Keyframe values (flat; rotations are xyzw, morph weights expand
    /// per-target).
    pub values: AnimValues,
}

/// Channel value storage.
#[derive(Debug)]
pub enum AnimValues {
    /// Vec3 keyframes (translation/scale).
    Vec3(Vec<Vec3>),
    /// Quaternion keyframes (rotation).
    Quat(Vec<glam::Quat>),
    /// Morph weight keyframes: `morph_count` floats per output element
    /// (per key, or per tangent triple for cubic-spline channels).
    Weights {
        /// Weight stride — the target mesh's morph target count.
        morph_count: usize,
        /// Flat `times.len() * morph_count` (`*3` for cubic spline) values.
        values: Vec<f32>,
    },
}

/// A named glTF animation clip.
#[derive(Debug)]
pub struct AnimationClip {
    /// Clip name, if the asset provides one.
    pub name: Option<String>,
    /// Duration of the last keyframe.
    pub duration: Duration,
    /// Channels.
    pub channels: Vec<ChannelData>,
}

impl AnimationClip {
    /// Clip duration.
    #[must_use]
    pub const fn duration(&self) -> Duration {
        self.duration
    }
}

/// Per-node hierarchy data.
#[derive(Debug)]
pub struct NodeData {
    /// Parent index, or `usize::MAX` for roots.
    pub parent: usize,
    /// Child indices.
    pub children: Vec<usize>,
    /// Local TRS.
    pub translation: Vec3,
    /// Local rotation quaternion.
    pub rotation: Quat,
    /// Local scale.
    pub scale: Vec3,
    /// Indices into `ModelData::primitives` drawn by this node.
    pub draws: Vec<usize>,
    /// Skin index, if skinned.
    pub skin: Option<usize>,
    /// Mesh index (for morph counts).
    pub mesh: Option<usize>,
    /// Authored default morph weights (node `weights` overriding the mesh's,
    /// or zeros when absent). Length = the mesh's morph target count.
    pub morph_weights: Vec<f32>,
    /// Offset of this node's joint palette in the flat joint buffer (0 when
    /// unskinned).
    pub joint_base: u32,
    /// Offset of this node's morph weights in the flat weight buffer (0 when
    /// it has no morph targets).
    pub weight_base: u32,
}

impl NodeData {
    /// Morph target count of this node's mesh (0 when unmeshed).
    pub fn morph_count(&self, meshes: &[MeshData]) -> usize {
        self.mesh.map_or(0, |m| meshes[m].morph_count)
    }

    /// This node's offset into the flat morph-weight buffer, when it owns a
    /// range (`morph_count > 0`).
    pub fn weight_base(&self, meshes: &[MeshData]) -> Option<usize> {
        (self.morph_count(meshes) > 0).then_some(self.weight_base as usize)
    }

    /// Number of joints in this node's skin palette (0 when unskinned).
    ///
    /// # Panics
    /// When a skin's joint count exceeds `u32::MAX` — impossible for any
    /// [`ModelData`] the loader produced (it bounds the same count when
    /// assigning `joint_base`s).
    pub fn joint_count(&self, data: &ModelData) -> u32 {
        self.skin.map_or(0, |s| {
            u32::try_from(data.skins[s].joints.len()).expect("joint count fits u32")
        })
    }
}

/// One shared primitive: a mesh primitive's geometry, stored once per
/// (mesh, primitive) pair so instanced nodes share GPU data.
#[derive(Debug)]
pub struct PrimitiveData {
    /// Interleaved base vertices.
    pub vertices: Vec<Vertex>,
    /// Skinning attributes parallel to `vertices`.
    pub skin: Vec<SkinVertex>,
    /// Triangle indices.
    pub indices: Vec<u32>,
    /// Morph target deltas, `morph_count` entries each `vertices.len()` long.
    pub morph_deltas: Vec<Vec<MorphDelta>>,
    /// Material index, or `usize::MAX` for the glTF default material.
    pub material: usize,
    /// Static local-space bounds.
    pub bounds: Bounds,
    /// Per morph target: max |position delta| — the culling bound expansion
    /// is the weight-weighted sum over these, evaluated per node per frame.
    pub morph_max_displacement: Vec<f32>,
}

/// A mesh record (just its morph target count for weight scratch sizing).
#[derive(Debug)]
pub struct MeshData {
    /// Number of morph targets across the mesh's primitives.
    pub morph_count: usize,
}

/// A skin: joint node list + inverse bind matrices.
#[derive(Debug)]
pub struct SkinData {
    /// Joint node indices.
    pub joints: Vec<usize>,
    /// Inverse bind matrices (one per joint; identity-padded).
    pub inverse_bind_matrices: Vec<Mat4>,
}

/// Decoded model internals, shared by every clone of a [`Model`].
#[derive(Debug)]
pub struct ModelData {
    /// All nodes.
    pub nodes: Vec<NodeData>,
    /// Shared primitive geometry.
    pub primitives: Vec<PrimitiveData>,
    /// Materials (index `usize::MAX` denotes the default).
    pub materials: Vec<MaterialData>,
    /// Textures.
    pub textures: Vec<TextureData>,
    /// Samplers.
    pub samplers: Vec<SamplerData>,
    /// Skins.
    pub skins: Vec<SkinData>,
    /// Meshes.
    pub meshes: Vec<MeshData>,
    /// Rest-pose union of transformed primitive bounds.
    pub bounds: Bounds,
    /// Whether any primitive has morph targets.
    pub has_morphs: bool,
    /// Animation clips.
    pub clips: Vec<AnimationClip>,
}

/// An opaque handle to one clip of one specific model.
///
/// Handles carry the identity of the [`Model`] they were minted from:
/// resolving a handle against a different model — including after a model
/// swap — returns [`InvalidClipError::WrongModel`], never a same-index clip of
/// the new asset.
#[derive(Debug, Clone)]
pub struct ClipHandle {
    data: Weak<ModelData>,
    index: usize,
}

impl PartialEq for ClipHandle {
    /// Identity equality: same model allocation and same clip index.
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.data.ptr_eq(&other.data)
    }
}

impl Eq for ClipHandle {}

impl ClipHandle {
    /// Clip index within the owning model.
    #[must_use]
    pub const fn index(&self) -> usize {
        self.index
    }
}

/// Immutable, cheaply cloneable handle to a decoded glTF model.
///
/// Clones share the same `Arc<ModelData>`; resource identity comparisons use
/// `Arc` pointer identity, so uploading an already-uploaded model is a no-op.
#[derive(Debug, Clone)]
pub struct Model {
    /// Shared decoded asset data — clones point at the same allocation.
    pub data: Arc<ModelData>,
}

impl PartialEq for Model {
    /// `Arc` identity: two clones of one asset compare equal.
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.data, &other.data)
    }
}

impl Eq for Model {}

impl Model {
    /// Shared asset data.
    #[must_use]
    pub const fn data(&self) -> &Arc<ModelData> {
        &self.data
    }

    /// Loads a glTF asset (`.gltf`, `.glb`) from `path`, resolving relative
    /// sidecar buffers and images on a worker thread.
    ///
    /// # Errors
    /// [`LoadError::Io`], [`LoadError::Parse`],
    /// [`LoadError::UnsupportedRequiredExtension`], or
    /// [`LoadError::InvalidData`].
    pub async fn load(path: impl AsRef<Path> + Send + 'static) -> Result<Self, LoadError> {
        blocking::unblock(move || -> Result<Self, LoadError> {
            let path = path.as_ref();
            let (document, buffers, images) = gltf::import(path).map_err(|e| match e {
                gltf::Error::Io(source) => LoadError::Io {
                    path: path.to_path_buf(),
                    source,
                },
                other => LoadError::Parse(other),
            })?;
            loader::check_required_extensions(&document)?;
            let data = load_from_parts(&document, &buffers, &images)?;
            Ok(Self {
                data: Arc::new(data),
            })
        })
        .await
    }

    /// Decodes an in-memory `.glb` (or `.gltf` JSON) document.
    ///
    /// External buffer/image URIs are rejected with
    /// [`LoadError::ExternalUri`] — an in-memory document has no resolver for
    /// sidecar files (embedded `data:` URIs are supported).
    ///
    /// # Errors
    /// [`LoadError::Parse`], [`LoadError::ExternalUri`],
    /// [`LoadError::UnsupportedRequiredExtension`], or
    /// [`LoadError::InvalidData`].
    pub async fn from_glb(bytes: Vec<u8>) -> Result<Self, LoadError> {
        blocking::unblock(move || -> Result<Self, LoadError> {
            let gltf = gltf::Gltf::from_slice(&bytes)?;
            loader::check_required_extensions(&gltf.document)?;
            loader::check_no_external_uris(&gltf.document)?;
            let (document, buffers, images) = gltf::import_slice(&bytes)?;
            let data = load_from_parts(&document, &buffers, &images)?;
            Ok(Self {
                data: Arc::new(data),
            })
        })
        .await
    }

    /// Number of animation clips.
    #[must_use]
    pub fn clip_count(&self) -> usize {
        self.data.clips.len()
    }

    /// Clip names in order.
    #[must_use]
    pub fn clip_names(&self) -> Vec<&str> {
        self.data
            .clips
            .iter()
            .filter_map(|c| c.name.as_deref())
            .collect()
    }

    /// A validated handle for clip `index`.
    ///
    /// # Errors
    /// [`InvalidClipError::OutOfRange`] when `index >= clip_count()`.
    pub fn clip_handle(&self, index: usize) -> Result<ClipHandle, InvalidClipError> {
        if index >= self.data.clips.len() {
            return Err(InvalidClipError::OutOfRange {
                index,
                clip_count: self.data.clips.len(),
            });
        }
        Ok(ClipHandle {
            data: Arc::downgrade(&self.data),
            index,
        })
    }

    /// A validated handle for the clip named `name`.
    ///
    /// # Errors
    /// [`InvalidClipError::OutOfRange`] when no clip carries that name.
    pub fn clip_named(&self, name: &str) -> Result<ClipHandle, InvalidClipError> {
        let index = self
            .data
            .clips
            .iter()
            .position(|c| c.name.as_deref() == Some(name))
            .ok_or(InvalidClipError::OutOfRange {
                index: usize::MAX,
                clip_count: self.data.clips.len(),
            })?;
        self.clip_handle(index)
    }

    /// Resolves `handle` against this model.
    ///
    /// # Errors
    /// [`InvalidClipError::WrongModel`] when the handle was minted from a
    /// different model; [`InvalidClipError::DroppedModel`] when the origin
    /// model no longer exists; [`InvalidClipError::OutOfRange`] for a stale
    /// index.
    pub fn clip(&self, handle: &ClipHandle) -> Result<&AnimationClip, InvalidClipError> {
        let data = handle
            .data
            .upgrade()
            .ok_or(InvalidClipError::DroppedModel)?;
        if !Arc::ptr_eq(&data, &self.data) {
            return Err(InvalidClipError::WrongModel);
        }
        self.data
            .clips
            .get(handle.index)
            .ok_or(InvalidClipError::OutOfRange {
                index: handle.index,
                clip_count: self.data.clips.len(),
            })
    }

    /// Union of the rest-pose geometry bounds.
    #[must_use]
    pub fn bounds(&self) -> Bounds {
        self.data.bounds
    }

    /// `Arc` identity check for resource reuse.
    #[must_use]
    pub fn ptr_eq(a: &Self, b: &Self) -> bool {
        Arc::ptr_eq(&a.data, &b.data)
    }
}

// `nami::impl_constant!(Model)` is applied by the `Model3D` adapter module
// (src/viewer.rs) — the engine itself carries no framework dependencies.
