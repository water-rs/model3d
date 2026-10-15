//! glTF → [`ModelData`] decoding.
//!
//! Geometry is built through a corner-soup pipeline: each triangle corner is
//! materialized, normals are synthesized flat when the asset omits them,
//! mikktspace computes tangents per corner when a normal map needs them, and a
//! `(source vertex, normal, tangent)` dedup key re-indexes — preserving the
//! seam splits glTF requires for both normal mapping and flat shading.

use std::collections::HashMap;

use glam::{Mat4, Quat, Vec3};

use crate::error::LoadError;
use crate::model::{
    AlphaMode, AnimPath, AnimValues, AnimationClip, Bounds, ChannelData, FilterModeData,
    InterpolationData, MaterialData, MeshData, ModelData, MorphDelta, NodeData, PrimitiveData,
    SamplerData, SkinData, SkinVertex, TextureData, TextureRef, Vertex, WrapModeData,
};

mod texture_ref_ext {
    use super::{LoadError, TextureRef};
    /// Raw `extensions` JSON of a glTF texture-info object.
    pub type ExtensionsMap = serde_json::Map<String, serde_json::Value>;
    /// Builds a [`TextureRef`] from a glTF texture-info type, applying
    /// `KHR_texture_transform` when present.
    pub trait TextureRefExt: Sized {
        /// Texture index into `Document::textures`.
        fn texture_index(&self) -> usize;
        /// Base `texCoord`.
        fn base_uv(&self) -> u32;
        /// Transform extension, when present.
        fn transform_parts(&self) -> super::TextureTransformParts;
    }
    pub fn make<E: TextureRefExt>(info: &E) -> Result<TextureRef, LoadError> {
        let (offset, rotation, scale, tex_coord) = info.transform_parts();
        let uv_index = tex_coord.unwrap_or_else(|| info.base_uv());
        if uv_index > 1 {
            return Err(LoadError::InvalidData(format!(
                "texCoord {uv_index} exceeds the two supported UV sets"
            )));
        }
        Ok(TextureRef {
            texture: info.texture_index(),
            uv_index,
            offset: offset.unwrap_or([0.0, 0.0]),
            rotation: rotation.unwrap_or(0.0),
            scale: scale.unwrap_or([1.0, 1.0]),
        })
    }
}

pub use texture_ref_ext::TextureRefExt;

impl TextureRefExt for gltf::texture::Info<'_> {
    fn texture_index(&self) -> usize {
        self.texture().index()
    }
    fn base_uv(&self) -> u32 {
        self.tex_coord()
    }
    fn transform_parts(&self) -> TextureTransformParts {
        self.texture_transform()
            .map(|t| {
                (
                    Some(t.offset()),
                    Some(t.rotation()),
                    Some(t.scale()),
                    t.tex_coord(),
                )
            })
            .unwrap_or_default()
    }
}

/// Reads `KHR_texture_transform` from a raw extensions map — the accessor on
/// normal/occlusion texture info is not typed by the gltf crate, so the JSON
/// `(offset, rotation, scale, tex_coord)` from a `KHR_texture_transform` block.
type TextureTransformParts = (Option<[f32; 2]>, Option<f32>, Option<[f32; 2]>, Option<u32>);

/// is read directly.
///
/// f64→f32 narrowing is intentional: `KHR_texture_transform` stores JSON
/// doubles whose authored range is small (offsets, radians, scales).
#[expect(
    clippy::cast_possible_truncation,
    reason = "KHR UV transform values are small"
)]
fn raw_texture_transform(
    extensions: Option<&texture_ref_ext::ExtensionsMap>,
) -> TextureTransformParts {
    let Some(map) = extensions else {
        return (None, None, None, None);
    };
    let Some(t) = map.get("KHR_texture_transform") else {
        return (None, None, None, None);
    };
    let array2 = |key: &str| -> Option<[f32; 2]> {
        let v = t.get(key)?.as_array()?;
        // KHR UV transform components are small authored values; f32 suffices.
        Some([v.first()?.as_f64()? as f32, v.get(1)?.as_f64()? as f32])
    };
    (
        array2("offset"),
        t.get("rotation")
            .and_then(serde_json::Value::as_f64)
            .map(|v| v as f32),
        array2("scale"),
        t.get("texCoord")
            .and_then(serde_json::Value::as_u64)
            .and_then(|v| u32::try_from(v).ok()),
    )
}

impl TextureRefExt for gltf::material::NormalTexture<'_> {
    fn texture_index(&self) -> usize {
        self.texture().index()
    }
    fn base_uv(&self) -> u32 {
        self.tex_coord()
    }
    fn transform_parts(&self) -> TextureTransformParts {
        raw_texture_transform(self.extensions())
    }
}

impl TextureRefExt for gltf::material::OcclusionTexture<'_> {
    fn texture_index(&self) -> usize {
        self.texture().index()
    }
    fn base_uv(&self) -> u32 {
        self.tex_coord()
    }
    fn transform_parts(&self) -> TextureTransformParts {
        raw_texture_transform(self.extensions())
    }
}

/// Extensions this loader actually implements. Anything in
/// `extensionsRequired` outside this set is a hard error — never a silent
/// drop.
const SUPPORTED_EXTENSIONS: &[&str] = &[
    "KHR_materials_unlit",
    "KHR_texture_transform",
    "KHR_materials_emissive_strength",
];

/// Rejects assets whose `extensionsRequired` names something unimplemented.
pub fn check_required_extensions(document: &gltf::Document) -> Result<(), LoadError> {
    for ext in document.extensions_required() {
        if !SUPPORTED_EXTENSIONS.contains(&ext) {
            return Err(LoadError::UnsupportedRequiredExtension(ext.to_owned()));
        }
    }
    Ok(())
}

/// Rejects external file URIs for in-memory loads (embedded `data:` URIs are
/// still allowed — the importer resolves them).
pub fn check_no_external_uris(document: &gltf::Document) -> Result<(), LoadError> {
    for buffer in document.buffers() {
        if let gltf::buffer::Source::Uri(uri) = buffer.source()
            && !uri.starts_with("data:")
        {
            return Err(LoadError::ExternalUri(uri.to_owned()));
        }
    }
    for image in document.images() {
        if let gltf::image::Source::Uri { uri, .. } = image.source()
            && !uri.starts_with("data:")
        {
            return Err(LoadError::ExternalUri(uri.to_owned()));
        }
    }
    Ok(())
}

fn invalid(msg: impl Into<String>) -> LoadError {
    LoadError::InvalidData(msg.into())
}

/// Decodes imported glTF parts into [`ModelData`].
///
/// # Errors
/// [`LoadError::InvalidData`] for structurally unusable assets (missing
/// positions, non-triangle primitives, mismatched attribute counts, >2 UV
/// sets, …).
pub fn load_from_parts(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    images: &[gltf::image::Data],
) -> Result<ModelData, LoadError> {
    let textures = load_textures(document, images)?;
    let samplers: Vec<SamplerData> = document.samplers().map(|s| load_sampler(&s)).collect();
    let materials = load_materials(document)?;

    // Primitives: one shared entry per (mesh, primitive) — the instancing key.
    let mut primitives: Vec<PrimitiveData> = Vec::new();
    let mut mesh_draws: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut meshes: Vec<MeshData> = Vec::new();
    for mesh in document.meshes() {
        let mut draws = Vec::new();
        let mut morph_count = 0usize;
        for primitive in mesh.primitives() {
            let material_index = primitive.material().index().unwrap_or(usize::MAX);
            let normal_tex = (material_index != usize::MAX)
                .then(|| materials[material_index].normal_tex)
                .flatten();
            let prim = load_primitive(&primitive, buffers, normal_tex)?;
            morph_count = morph_count.max(prim.morph_deltas.len());
            draws.push(primitives.len());
            primitives.push(prim);
        }
        mesh_draws.insert(mesh.index(), draws);
        meshes.push(MeshData { morph_count });
    }

    let skins = load_skins(document, buffers)?;
    let mut nodes = load_nodes(document, &mesh_draws, &meshes, &skins)?;
    let clips = load_animations(document, buffers, &nodes, &meshes)?;

    // Canonicalize the glTF default material: `usize::MAX` material indices
    // point at a real appended record so GPU indexing never sees the sentinel.
    let mut materials = materials;
    if primitives.iter().any(|p| p.material == usize::MAX) {
        let default_index = materials.len();
        materials.push(MaterialData::default_gltf());
        for prim in &mut primitives {
            if prim.material == usize::MAX {
                prim.material = default_index;
            }
        }
    }

    // Selected scene (default scene, else the first): only its subtree is
    // drawn and bounded. Skeleton joints or detached roots outside it keep
    // their global transforms evaluated — skins may reference them — but
    // they contribute no draws.
    let scene = document
        .default_scene()
        .or_else(|| document.scenes().next());
    let roots: Vec<usize> = scene.map_or_else(
        || {
            nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| n.parent == usize::MAX)
                .map(|(i, _)| i)
                .collect()
        },
        |s| s.nodes().map(|n| n.index()).collect(),
    );
    let mut in_scene = vec![false; nodes.len()];
    let mut stack: Vec<usize> = roots;
    while let Some(i) = stack.pop() {
        if in_scene[i] {
            continue;
        }
        in_scene[i] = true;
        stack.extend_from_slice(&nodes[i].children);
    }
    for (i, node) in nodes.iter_mut().enumerate() {
        if !in_scene[i] {
            node.draws.clear();
        }
    }

    let locals = rest_locals(&nodes);
    let mut globals = vec![Mat4::IDENTITY; nodes.len()];
    compute_global_transforms(&nodes, &locals, &mut globals);
    let mut bounds = Bounds::empty();
    for (i, node) in nodes.iter().enumerate() {
        for &prim_index in &node.draws {
            bounds.union(&primitives[prim_index].bounds.transformed(&globals[i]));
        }
    }

    let has_morphs = primitives.iter().any(|p| !p.morph_deltas.is_empty());

    Ok(ModelData {
        nodes,
        primitives,
        materials,
        textures,
        samplers,
        skins,
        meshes,
        bounds,
        has_morphs,
        clips,
    })
}

fn load_textures(
    document: &gltf::Document,
    images: &[gltf::image::Data],
) -> Result<Vec<TextureData>, LoadError> {
    let mut textures = Vec::new();
    for texture in document.textures() {
        let image_index = texture.source().index();
        let image = images
            .get(image_index)
            .ok_or_else(|| invalid(format!("image {image_index} missing")))?;
        let (width, height) = (image.width, image.height);
        let pixels = match image.format {
            gltf::image::Format::R8G8B8A8 => image.pixels.clone(),
            gltf::image::Format::R8G8B8 => {
                let mut out = Vec::with_capacity(image.pixels.len() / 3 * 4);
                for px in image.pixels.as_chunks::<3>().0 {
                    out.extend_from_slice(&[px[0], px[1], px[2], 255]);
                }
                out
            }
            gltf::image::Format::R8 => {
                let mut out = Vec::with_capacity(image.pixels.len() * 4);
                for &px in &image.pixels {
                    out.extend_from_slice(&[px, px, px, 255]);
                }
                out
            }
            gltf::image::Format::R8G8 => {
                let mut out = Vec::with_capacity(image.pixels.len() / 2 * 4);
                for px in image.pixels.as_chunks::<2>().0 {
                    out.extend_from_slice(&[px[0], px[1], 0, 255]);
                }
                out
            }
            gltf::image::Format::R16
            | gltf::image::Format::R16G16
            | gltf::image::Format::R16G16B16
            | gltf::image::Format::R16G16B16A16 => {
                return Err(invalid(format!(
                    "16-bit image format {:?} not supported",
                    image.format
                )));
            }
            fmt => {
                return Err(invalid(format!("image format {fmt:?} not supported")));
            }
        };
        textures.push(TextureData {
            pixels,
            width,
            height,
            sampler: texture.sampler().index(),
        });
    }
    Ok(textures)
}

fn load_sampler(sampler: &gltf::texture::Sampler<'_>) -> SamplerData {
    use gltf::texture::{MagFilter, MinFilter, WrappingMode};
    let wrap = |mode| match mode {
        WrappingMode::ClampToEdge => WrapModeData::Clamp,
        WrappingMode::MirroredRepeat => WrapModeData::MirroredRepeat,
        WrappingMode::Repeat => WrapModeData::Repeat,
    };
    let mag = sampler.mag_filter().map(|m| match m {
        MagFilter::Nearest => FilterModeData::Nearest,
        MagFilter::Linear => FilterModeData::Linear,
    });
    let min = sampler.min_filter();
    SamplerData {
        wrap_s: wrap(sampler.wrap_s()),
        wrap_t: wrap(sampler.wrap_t()),
        mag,
        // Nearest min filters (with or without mipmapping) sample the nearest
        // texel; every other min filter bilinear-filters.
        min: min.map(|m| match m {
            MinFilter::Nearest
            | MinFilter::NearestMipmapNearest
            | MinFilter::NearestMipmapLinear => FilterModeData::Nearest,
            _ => FilterModeData::Linear,
        }),
        // `*MipmapNearest` picks the nearest mip level; `*MipmapLinear`
        // interpolates between levels.
        mipmap: match min {
            Some(MinFilter::NearestMipmapNearest | MinFilter::LinearMipmapNearest) => {
                FilterModeData::Nearest
            }
            _ => FilterModeData::Linear,
        },
        has_mipmaps: matches!(
            min,
            Some(
                MinFilter::NearestMipmapNearest
                    | MinFilter::NearestMipmapLinear
                    | MinFilter::LinearMipmapNearest
                    | MinFilter::LinearMipmapLinear
            )
        ),
    }
}

fn load_materials(document: &gltf::Document) -> Result<Vec<MaterialData>, LoadError> {
    let mut materials = Vec::new();
    for material in document.materials() {
        let pbr = material.pbr_metallic_roughness();
        let alpha = match material.alpha_mode() {
            gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
            gltf::material::AlphaMode::Mask => AlphaMode::Mask,
            gltf::material::AlphaMode::Blend => AlphaMode::Blend,
        };
        materials.push(MaterialData {
            base_color: pbr.base_color_factor(),
            metallic: pbr.metallic_factor(),
            roughness: pbr.roughness_factor(),
            base_color_tex: pbr
                .base_color_texture()
                .map(|info| texture_ref_ext::make(&info))
                .transpose()?,
            metallic_roughness_tex: pbr
                .metallic_roughness_texture()
                .map(|info| texture_ref_ext::make(&info))
                .transpose()?,
            normal_tex: material
                .normal_texture()
                .map(|info| texture_ref_ext::make(&info))
                .transpose()?,
            normal_scale: material.normal_texture().map_or(1.0, |t| t.scale()),
            occlusion_tex: material
                .occlusion_texture()
                .map(|info| texture_ref_ext::make(&info))
                .transpose()?,
            occlusion_strength: material.occlusion_texture().map_or(1.0, |t| t.strength()),
            emissive_tex: material
                .emissive_texture()
                .map(|info| texture_ref_ext::make(&info))
                .transpose()?,
            emissive_factor: material.emissive_factor(),
            emissive_strength: material.emissive_strength().unwrap_or(1.0),
            alpha,
            alpha_cutoff: material.alpha_cutoff().unwrap_or(0.5),
            double_sided: material.double_sided(),
            unlit: material.unlit(),
        });
    }
    Ok(materials)
}

/// Per-corner working vertex before dedup.
#[derive(Clone, Copy)]
struct Corner {
    src: usize,
    position: [f32; 3],
    normal: [f32; 3],
    tangent: [f32; 4],
}

struct TangentSoup<'a> {
    corners: &'a mut [Corner],
    uv_src: &'a [[f32; 2]],
    uv_xform: &'a dyn Fn([f32; 2]) -> [f32; 2],
    undefined: bool,
}

impl bevy_mikktspace::Geometry for TangentSoup<'_> {
    fn num_faces(&self) -> usize {
        self.corners.len() / 3
    }
    fn num_vertices_of_face(&self, _face: usize) -> usize {
        3
    }
    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        self.corners[face * 3 + vert].position
    }
    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        self.corners[face * 3 + vert].normal
    }
    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        (self.uv_xform)(self.uv_src[self.corners[face * 3 + vert].src])
    }
    fn set_tangent(
        &mut self,
        tangent: Option<bevy_mikktspace::TangentSpace>,
        face: usize,
        vert: usize,
    ) {
        if let Some(t) = tangent {
            self.corners[face * 3 + vert].tangent = t.tangent_encoded();
        } else {
            // `None` is an expected algorithm result: mikktspace yields it
            // when a vertex's whole neighborhood is degenerate (e.g.
            // DamagedHelmet's 10 zero-UV-area triangles of 15452) with
            // nothing to borrow a basis from. We never invent an
            // arbitrary frame — the zero tangent is a defined sentinel
            // the fragment shader detects (`dot(tt, tt) < 1e-8`) and
            // falls back to the geometric normal.
            self.corners[face * 3 + vert].tangent = [0.0; 4];
            self.undefined = true;
        }
    }
}

fn load_primitive(
    primitive: &gltf::Primitive<'_>,
    buffers: &[gltf::buffer::Data],
    normal_tex: Option<TextureRef>,
) -> Result<PrimitiveData, LoadError> {
    if primitive.mode() != gltf::mesh::Mode::Triangles {
        return Err(invalid(format!(
            "primitive mode {:?} is not TRIANGLES",
            primitive.mode()
        )));
    }
    let attrs = read_attributes(primitive, buffers)?;
    let vertex_count = attrs.positions.len();
    let morph_deltas = read_morph_deltas(primitive, buffers, vertex_count)?;
    let indices = read_indices(primitive, buffers, vertex_count)?;

    let mut corners = corner_soup(&indices, &attrs)?;
    if attrs.normals.is_none() {
        apply_flat_normals(&mut corners);
    }
    if let Some(nref) = normal_tex
        && attrs.tangents.is_none()
    {
        mikktspace_tangents(&mut corners, &nref, &attrs)?;
    }

    let dedup = dedup_corners(&corners, &attrs);
    // Morph deltas follow the *source* vertex: every corner mapping to one
    // emitted vertex shares its source, so `emit_src[emit]` is well-defined.
    let morph_deltas = remap_morph_deltas(&morph_deltas, &dedup.emit_src);
    let morph_max_displacement: Vec<f32> = morph_deltas
        .iter()
        .map(|deltas| {
            deltas
                .iter()
                .map(|d| Vec3::new(d.position[0], d.position[1], d.position[2]).length())
                .fold(0.0f32, f32::max)
        })
        .collect();

    Ok(PrimitiveData {
        vertices: dedup.vertices,
        skin: dedup.skin,
        indices: dedup.indices,
        morph_deltas,
        material: primitive.material().index().unwrap_or(usize::MAX),
        bounds: dedup.bounds,
        morph_max_displacement,
    })
}

/// An attribute accessor's element count must match the vertex count.
fn check_len(len: usize, vertex_count: usize, attr: &str) -> Result<(), LoadError> {
    if len != vertex_count {
        return Err(invalid(format!(
            "attribute {attr} has {len} values, expected {vertex_count}"
        )));
    }
    Ok(())
}

/// One primitive's decoded attributes — every `Vec` is indexed by *source*
/// vertex (`positions.len()` entries).
struct PrimitiveAttributes {
    positions: Vec<Vec3>,
    normals: Option<Vec<Vec3>>,
    tangents: Option<Vec<[f32; 4]>>,
    colors: Vec<[f32; 4]>,
    uv0: Vec<[f32; 2]>,
    uv1: Vec<[f32; 2]>,
    joints: Vec<[u16; 4]>,
    weights: Vec<[f32; 4]>,
}

/// Reads every per-vertex attribute accessor of `primitive`.
fn read_attributes(
    primitive: &gltf::Primitive<'_>,
    buffers: &[gltf::buffer::Data],
) -> Result<PrimitiveAttributes, LoadError> {
    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
    let positions: Vec<Vec3> = reader
        .read_positions()
        .ok_or_else(|| invalid("primitive has no POSITION attribute"))?
        .map(Vec3::from)
        .collect();
    let vertex_count = positions.len();
    let normals: Option<Vec<Vec3>> = reader
        .read_normals()
        .map(|it| it.map(Vec3::from).collect::<Vec<_>>())
        .map(|v| check_len(v.len(), vertex_count, "NORMAL").map(|()| v))
        .transpose()?;
    let tangents: Option<Vec<[f32; 4]>> = reader
        .read_tangents()
        .map(std::iter::Iterator::collect::<Vec<_>>)
        .map(|v| check_len(v.len(), vertex_count, "TANGENT").map(|()| v))
        .transpose()?;
    Ok(PrimitiveAttributes {
        positions,
        normals,
        tangents,
        colors: read_colors(primitive, buffers, vertex_count)?,
        uv0: read_uvs(primitive, buffers, 0, vertex_count)?,
        uv1: read_uvs(primitive, buffers, 1, vertex_count)?,
        joints: read_joints(primitive, buffers, vertex_count)?,
        weights: read_weights(primitive, buffers, vertex_count)?,
    })
}

/// `COLOR_0` normalized to float RGBA, or opaque white when absent.
fn read_colors(
    primitive: &gltf::Primitive<'_>,
    buffers: &[gltf::buffer::Data],
    vertex_count: usize,
) -> Result<Vec<[f32; 4]>, LoadError> {
    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
    let Some(read) = reader.read_colors(0) else {
        return Ok(vec![[1.0; 4]; vertex_count]);
    };
    let v: Vec<[f32; 4]> = match read {
        gltf::mesh::util::ReadColors::RgbU8(it) => it
            .map(|c| {
                let [r, g, b] = c;
                [
                    f32::from(r) / 255.0,
                    f32::from(g) / 255.0,
                    f32::from(b) / 255.0,
                    1.0,
                ]
            })
            .collect(),
        gltf::mesh::util::ReadColors::RgbU16(it) => it
            .map(|c| {
                let [r, g, b] = c;
                [
                    f32::from(r) / 65535.0,
                    f32::from(g) / 65535.0,
                    f32::from(b) / 65535.0,
                    1.0,
                ]
            })
            .collect(),
        gltf::mesh::util::ReadColors::RgbF32(it) => it.map(|c| [c[0], c[1], c[2], 1.0]).collect(),
        gltf::mesh::util::ReadColors::RgbaU8(it) => {
            it.map(|c| c.map(|x| f32::from(x) / 255.0)).collect()
        }
        gltf::mesh::util::ReadColors::RgbaU16(it) => {
            it.map(|c| c.map(|x| f32::from(x) / 65535.0)).collect()
        }
        gltf::mesh::util::ReadColors::RgbaF32(it) => it.collect(),
    };
    check_len(v.len(), vertex_count, "COLOR_0")?;
    Ok(v)
}

/// `TEXCOORD_n` accessor, or zeroed UVs when the primitive lacks the set.
fn read_uvs(
    primitive: &gltf::Primitive<'_>,
    buffers: &[gltf::buffer::Data],
    set: u32,
    vertex_count: usize,
) -> Result<Vec<[f32; 2]>, LoadError> {
    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
    let Some(read) = reader.read_tex_coords(set) else {
        return Ok(vec![[0.0; 2]; vertex_count]);
    };
    let v: Vec<[f32; 2]> = read.into_f32().collect();
    check_len(v.len(), vertex_count, &format!("TEXCOORD_{set}"))?;
    Ok(v)
}

/// `JOINTS_0` accessor, or zeroed joints when the primitive is unskinned.
fn read_joints(
    primitive: &gltf::Primitive<'_>,
    buffers: &[gltf::buffer::Data],
    vertex_count: usize,
) -> Result<Vec<[u16; 4]>, LoadError> {
    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
    let Some(read) = reader.read_joints(0) else {
        return Ok(vec![[0; 4]; vertex_count]);
    };
    let v: Vec<[u16; 4]> = read.into_u16().collect();
    check_len(v.len(), vertex_count, "JOINTS_0")?;
    Ok(v)
}

/// `WEIGHTS_0` accessor renormalized to sum 1, or the identity weight —
/// the shader's skinning path then degenerates to a no-op.
fn read_weights(
    primitive: &gltf::Primitive<'_>,
    buffers: &[gltf::buffer::Data],
    vertex_count: usize,
) -> Result<Vec<[f32; 4]>, LoadError> {
    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
    let Some(read) = reader.read_weights(0) else {
        return Ok(vec![[1.0, 0.0, 0.0, 0.0]; vertex_count]);
    };
    let v: Vec<[f32; 4]> = read
        .into_f32()
        .map(|w| {
            let sum: f32 = w.iter().sum();
            if sum > f32::EPSILON {
                w.map(|x| x / sum)
            } else {
                [1.0, 0.0, 0.0, 0.0]
            }
        })
        .collect();
    check_len(v.len(), vertex_count, "WEIGHTS_0")?;
    Ok(v)
}

/// Morph target deltas (position/normal/tangent), per target; the gltf
/// reader yields (positions, normals, tangents) iterator tuples.
fn read_morph_deltas(
    primitive: &gltf::Primitive<'_>,
    buffers: &[gltf::buffer::Data],
    vertex_count: usize,
) -> Result<Vec<Vec<MorphDelta>>, LoadError> {
    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
    let mut morph_deltas: Vec<Vec<MorphDelta>> = Vec::new();
    for (positions, normals, tangents) in reader.read_morph_targets() {
        let pos: Vec<[f32; 3]> = positions.map_or_else(
            || vec![[0.0; 3]; vertex_count],
            std::iter::Iterator::collect,
        );
        let nrm: Vec<[f32; 3]> = normals.map_or_else(
            || vec![[0.0; 3]; vertex_count],
            std::iter::Iterator::collect,
        );
        let tan: Vec<[f32; 3]> = tangents.map_or_else(
            || vec![[0.0; 3]; vertex_count],
            std::iter::Iterator::collect,
        );
        check_len(pos.len(), vertex_count, "morph POSITION")?;
        check_len(nrm.len(), vertex_count, "morph NORMAL")?;
        check_len(tan.len(), vertex_count, "morph TANGENT")?;
        morph_deltas.push(
            (0..vertex_count)
                .map(|i| MorphDelta {
                    position: [pos[i][0], pos[i][1], pos[i][2], 0.0],
                    normal: [nrm[i][0], nrm[i][1], nrm[i][2], 0.0],
                    tangent: [tan[i][0], tan[i][1], tan[i][2], 0.0],
                })
                .collect(),
        );
    }
    Ok(morph_deltas)
}

/// Index buffer, or the identity `0..vertex_count` sequence when the
/// primitive is non-indexed.
fn read_indices(
    primitive: &gltf::Primitive<'_>,
    buffers: &[gltf::buffer::Data],
    vertex_count: usize,
) -> Result<Vec<u32>, LoadError> {
    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
    let indices: Vec<u32> = reader.read_indices().map_or_else(
        || (0..u32::try_from(vertex_count).expect("vertex count fits u32")).collect(),
        |read| read.into_u32().collect(),
    );
    if !indices.len().is_multiple_of(3) {
        return Err(invalid("index count is not a multiple of 3"));
    }
    Ok(indices)
}

/// Materializes one [`Corner`] per index — the un-deduplicated soup.
fn corner_soup(indices: &[u32], attrs: &PrimitiveAttributes) -> Result<Vec<Corner>, LoadError> {
    let vertex_count = attrs.positions.len();
    let mut corners: Vec<Corner> = Vec::with_capacity(indices.len());
    for &idx in indices {
        let src = idx as usize;
        let src_pos = attrs
            .positions
            .get(src)
            .ok_or_else(|| invalid(format!("index {src} out of {vertex_count} vertices")))?;
        let normal = attrs.normals.as_ref().map_or(Vec3::ZERO, |n| n[src]);
        corners.push(Corner {
            src,
            position: (*src_pos).into(),
            normal: normal.into(),
            tangent: attrs
                .tangents
                .as_ref()
                .map_or([1.0, 0.0, 0.0, 1.0], |t| t[src]),
        });
    }
    Ok(corners)
}

/// Flat shading: every corner takes its face normal — no shared-vertex
/// overwrite.
fn apply_flat_normals(corners: &mut [Corner]) {
    for face in corners.as_chunks_mut::<3>().0 {
        let a = Vec3::from(face[0].position);
        let n = (Vec3::from(face[1].position) - a)
            .cross(Vec3::from(face[2].position) - a)
            .normalize_or(Vec3::Y);
        for corner in face.iter_mut() {
            corner.normal = n.into();
        }
    }
}

/// mikktspace over the corner soup preserves UV-seam splits. The normal
/// map's own UV set and `KHR_texture_transform` select the tex coords — not
/// unconditionally uv0.
fn mikktspace_tangents(
    corners: &mut [Corner],
    nref: &TextureRef,
    attrs: &PrimitiveAttributes,
) -> Result<(), LoadError> {
    let uv_src: &[[f32; 2]] = if nref.uv_index == 0 {
        &attrs.uv0
    } else {
        &attrs.uv1
    };
    let (cr, sr) = (nref.rotation.cos(), nref.rotation.sin());
    let uv_xform = |uv: [f32; 2]| -> [f32; 2] {
        let (x, y) = (uv[0] * nref.scale[0], uv[1] * nref.scale[1]);
        [
            f32::mul_add(sr, -y, cr * x) + nref.offset[0],
            f32::mul_add(cr, y, sr * x) + nref.offset[1],
        ]
    };
    let mut soup = TangentSoup {
        corners,
        uv_src,
        uv_xform: &uv_xform,
        undefined: false,
    };
    bevy_mikktspace::generate_tangents(&mut soup)
        .map_err(|e| invalid(format!("tangent generation failed: {e}")))?;
    // Degenerate corners legitimately produced no basis; they carry the
    // zero-tangent sentinel — not a reason to reject the asset.
    let _ = soup.undefined;
    Ok(())
}

/// Re-indexed streams plus the emit-vertex → source-vertex map.
struct Deduped {
    vertices: Vec<Vertex>,
    skin: Vec<SkinVertex>,
    indices: Vec<u32>,
    bounds: Bounds,
    emit_src: Vec<usize>,
}

/// Re-indexes: identical (source vertex, normal, tangent) corners share one
/// vertex; differing corners split — which is exactly what seams need.
fn dedup_corners(corners: &[Corner], attrs: &PrimitiveAttributes) -> Deduped {
    let mut vertex_map: HashMap<(u32, [u32; 3], [u32; 4]), u32> = HashMap::new();
    let mut vertices: Vec<Vertex> = Vec::new();
    let mut skin: Vec<SkinVertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::with_capacity(corners.len());
    let mut bounds = Bounds::empty();
    let mut emit_src: Vec<usize> = Vec::new();
    for corner in corners {
        let key = (
            u32::try_from(corner.src).expect("vertex index fits u32"),
            corner.normal.map(f32::to_bits),
            corner.tangent.map(f32::to_bits),
        );
        let index = *vertex_map.entry(key).or_insert_with(|| {
            vertices.push(Vertex {
                position: corner.position,
                normal: corner.normal,
                tangent: corner.tangent,
                color: attrs.colors[corner.src],
                uv0: attrs.uv0[corner.src],
                uv1: attrs.uv1[corner.src],
            });
            skin.push(SkinVertex {
                joints: attrs.joints[corner.src],
                weights: attrs.weights[corner.src],
            });
            // The first corner to emit a vertex defines its source — all
            // later corners sharing the key carry the same `src`.
            emit_src.push(corner.src);
            u32::try_from(vertices.len() - 1).expect("vertex index fits u32")
        });
        indices.push(index);
        bounds.include(Vec3::from(corner.position));
    }
    Deduped {
        vertices,
        skin,
        indices,
        bounds,
        emit_src,
    }
}

/// Re-bases per-source-vertex morph deltas onto emitted vertices.
fn remap_morph_deltas(deltas: &[Vec<MorphDelta>], emit_src: &[usize]) -> Vec<Vec<MorphDelta>> {
    deltas
        .iter()
        .map(|d| emit_src.iter().map(|&src| d[src]).collect())
        .collect()
}

fn load_skins(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
) -> Result<Vec<SkinData>, LoadError> {
    let mut skins = Vec::new();
    for skin in document.skins() {
        let joints: Vec<usize> = skin.joints().map(|j| j.index()).collect();
        let reader = skin.reader(|buffer| Some(&buffers[buffer.index()]));
        let ibms: Vec<Mat4> = reader.read_inverse_bind_matrices().map_or_else(
            || vec![Mat4::IDENTITY; joints.len()],
            |it| it.map(|m| Mat4::from_cols_array_2d(&m)).collect(),
        );
        if ibms.len() != joints.len() {
            return Err(invalid(format!(
                "skin has {} inverse bind matrices for {} joints",
                ibms.len(),
                joints.len()
            )));
        }
        skins.push(SkinData {
            joints,
            inverse_bind_matrices: ibms,
        });
    }
    Ok(skins)
}

fn load_nodes(
    document: &gltf::Document,
    mesh_draws: &HashMap<usize, Vec<usize>>,
    meshes: &[MeshData],
    skins: &[SkinData],
) -> Result<Vec<NodeData>, LoadError> {
    let mut nodes: Vec<NodeData> = Vec::new();
    let mut parents: Vec<usize> = Vec::new();
    for node in document.nodes() {
        let (t, r, s) = match node.transform() {
            gltf::scene::Transform::Decomposed { .. } => node.transform().decomposed(),
            gltf::scene::Transform::Matrix { matrix } => {
                let m = Mat4::from_cols_array_2d(&matrix);
                let (ms, mr, mt) = m.to_scale_rotation_translation();
                let rebuilt = Mat4::from_scale_rotation_translation(ms, mr, mt);
                // TRS decomposition silently drops shear — validate the
                // authored matrix really is a TRS instead of losing it.
                let ok = m
                    .to_cols_array()
                    .iter()
                    .zip(rebuilt.to_cols_array())
                    .all(|(a, b)| (a - b).abs() < 1e-4);
                if !ok {
                    return Err(invalid(format!(
                        "node {} carries a shear/non-TRS matrix transform",
                        node.index()
                    )));
                }
                (mt.to_array(), mr.to_array(), ms.to_array())
            }
        };
        let mesh_index = node.mesh().map(|m| m.index());
        let draws = mesh_index.map_or_else(Vec::new, |m| {
            mesh_draws.get(&m).cloned().unwrap_or_default()
        });
        let morph_count = mesh_index.map_or(0, |m| meshes[m].morph_count);
        // Authored morph defaults: the node's `weights` override the mesh's;
        // absent both, all targets rest at 0.
        let mesh = node.mesh();
        let morph_weights = if morph_count == 0 {
            Vec::new()
        } else {
            node.weights()
                .or_else(|| mesh.as_ref().and_then(gltf::Mesh::weights))
                .map_or_else(|| vec![0.0; morph_count], Vec::from)
        };
        if morph_weights.len() != morph_count {
            return Err(invalid(format!(
                "node {} has {} authored morph weights, expected {morph_count}",
                node.index(),
                morph_weights.len()
            )));
        }
        nodes.push(NodeData {
            parent: usize::MAX,
            children: node.children().map(|c| c.index()).collect(),
            translation: Vec3::from(t),
            rotation: Quat::from_array(r),
            scale: Vec3::from(s),
            draws,
            skin: node.skin().map(|s| s.index()),
            mesh: mesh_index,
            morph_weights,
            joint_base: 0,
            weight_base: 0,
        });
        parents.push(usize::MAX);
    }
    for (i, node) in nodes.iter().enumerate() {
        for &child in &node.children {
            if child >= nodes.len() {
                return Err(invalid(format!("node {i} references child {child}")));
            }
            if parents[child] != usize::MAX {
                return Err(invalid(format!(
                    "node {child} has multiple parents ({} and {i})",
                    parents[child]
                )));
            }
            parents[child] = i;
        }
    }
    // Reject cycles anywhere in the graph — a cycle hangs transform walks.
    let mut state = vec![0u8; nodes.len()]; // 0 unvisited, 1 in-stack, 2 done
    for root in 0..nodes.len() {
        check_cycles(&nodes, &mut state, root)?;
    }
    for (i, node) in nodes.iter_mut().enumerate() {
        node.parent = parents[i];
    }
    // Flat deformation-buffer offsets: joint palettes run per *skinned node*
    // (each carries its own inverse mesh transform even for a shared skin),
    // morph weights per node with a morphing mesh.
    let mut joint_cursor = 0u32;
    let mut weight_cursor = 0u32;
    for node in &mut nodes {
        if let Some(skin) = node.skin {
            node.joint_base = joint_cursor;
            joint_cursor += u32::try_from(skins[skin].joints.len()).expect("joint count fits u32");
        }
        if node.morph_count(meshes) > 0 {
            node.weight_base = weight_cursor;
            weight_cursor += u32::try_from(node.morph_count(meshes)).expect("morph count fits u32");
        }
    }
    Ok(nodes)
}

fn check_cycles(nodes: &[NodeData], state: &mut [u8], index: usize) -> Result<(), LoadError> {
    if state[index] == 1 {
        return Err(invalid(format!("node {index} participates in a cycle")));
    }
    if state[index] == 2 {
        return Ok(());
    }
    state[index] = 1;
    for &child in &nodes[index].children {
        check_cycles(nodes, state, child)?;
    }
    state[index] = 2;
    Ok(())
}

fn load_animations(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    nodes: &[NodeData],
    meshes: &[MeshData],
) -> Result<Vec<AnimationClip>, LoadError> {
    let mut clips = Vec::new();
    for animation in document.animations() {
        let mut channels = Vec::new();
        let mut duration = 0.0f32;
        for channel in animation.channels() {
            let channel = load_channel(&channel, buffers, nodes, meshes)?;
            duration = duration.max(*channel.times.last().expect("channel times are non-empty"));
            channels.push(channel);
        }
        clips.push(AnimationClip {
            name: animation.name().map(str::to_owned),
            // A finite-but-huge f32 (e.g. 1e30) exceeds Duration's range —
            // `try_from_secs_f32` rejects instead of panicking.
            duration: std::time::Duration::try_from_secs_f32(duration)
                .map_err(|_| invalid("animation clip duration exceeds Duration range"))?,
            channels,
        });
    }
    Ok(clips)
}

/// One decoded channel: validated input times plus typed output values.
fn load_channel(
    channel: &gltf::animation::Channel<'_>,
    buffers: &[gltf::buffer::Data],
    nodes: &[NodeData],
    meshes: &[MeshData],
) -> Result<ChannelData, LoadError> {
    let reader = channel.reader(|buffer| Some(&buffers[buffer.index()]));
    let times: Vec<f32> = reader
        .read_inputs()
        .ok_or_else(|| invalid("animation channel has no input times"))?
        .collect();
    if times.is_empty() {
        return Err(invalid("animation channel has empty input times"));
    }
    if !times.iter().all(|t| t.is_finite() && *t >= 0.0) || !times.windows(2).all(|w| w[0] < w[1]) {
        return Err(invalid(
            "animation input times must be finite, non-negative and strictly increasing",
        ));
    }
    let interpolation = match channel.sampler().interpolation() {
        gltf::animation::Interpolation::Linear => InterpolationData::Linear,
        gltf::animation::Interpolation::Step => InterpolationData::Step,
        gltf::animation::Interpolation::CubicSpline => InterpolationData::CubicSpline,
    };
    let node_index = channel.target().node().index();
    let path = match channel.target().property() {
        gltf::animation::Property::Translation => AnimPath::Translation,
        gltf::animation::Property::Rotation => AnimPath::Rotation,
        gltf::animation::Property::Scale => AnimPath::Scale,
        gltf::animation::Property::MorphTargetWeights => AnimPath::MorphWeights,
    };
    let values = read_channel_values(
        channel,
        buffers,
        &times,
        interpolation,
        node_index,
        nodes,
        meshes,
    )?;
    Ok(ChannelData {
        node: node_index,
        path,
        interpolation,
        times,
        values,
    })
}

/// Reads and validates one channel's output accessor. Cubic-spline outputs
/// carry in-tangent/value/out-tangent triples per key; other interpolations
/// carry one element per key.
fn read_channel_values(
    channel: &gltf::animation::Channel<'_>,
    buffers: &[gltf::buffer::Data],
    times: &[f32],
    interpolation: InterpolationData,
    node_index: usize,
    nodes: &[NodeData],
    meshes: &[MeshData],
) -> Result<AnimValues, LoadError> {
    let reader = channel.reader(|buffer| Some(&buffers[buffer.index()]));
    let elements_per_key = if interpolation == InterpolationData::CubicSpline {
        3
    } else {
        1
    };
    let expect = |actual: usize| -> Result<(), LoadError> {
        let wanted = times.len() * elements_per_key;
        if actual != wanted {
            return Err(invalid(format!(
                "animation output has {actual} elements, expected {wanted} \
                 ({} keys x {elements_per_key})",
                times.len()
            )));
        }
        Ok(())
    };
    let values = match reader.read_outputs() {
        Some(gltf::animation::util::ReadOutputs::Translations(it)) => {
            let v: Vec<Vec3> = it.map(Vec3::from).collect();
            expect(v.len())?;
            if !v.iter().all(|q| q.is_finite()) {
                return Err(invalid("translation channel has non-finite values"));
            }
            AnimValues::Vec3(v)
        }
        Some(gltf::animation::util::ReadOutputs::Rotations(it)) => {
            let mut v: Vec<Quat> = it.into_f32().map(Quat::from_array).collect();
            expect(v.len())?;
            if !v.iter().all(|q| q.is_finite()) {
                return Err(invalid("rotation channel has non-finite values"));
            }
            // Key values (not cubic tangents) must be nonzero
            // quaternions; normalize them for clean slerp inputs.
            for (i, q) in v.iter_mut().enumerate() {
                let is_value = interpolation != InterpolationData::CubicSpline || i % 3 == 1;
                if is_value {
                    let len = q.length();
                    if !(len.is_finite() && len > 1e-6) {
                        return Err(invalid(
                            "rotation channel has a zero/non-finite quaternion key",
                        ));
                    }
                    *q /= len;
                }
            }
            AnimValues::Quat(v)
        }
        Some(gltf::animation::util::ReadOutputs::Scales(it)) => {
            let v: Vec<Vec3> = it.map(Vec3::from).collect();
            expect(v.len())?;
            if !v.iter().all(|q| q.is_finite()) {
                return Err(invalid("scale channel has non-finite values"));
            }
            AnimValues::Vec3(v)
        }
        Some(gltf::animation::util::ReadOutputs::MorphTargetWeights(it)) => {
            let morph_count = nodes.get(node_index).map_or(0, |n| n.morph_count(meshes));
            if morph_count == 0 {
                return Err(invalid(format!(
                    "weights channel targets node {node_index} with no morphing mesh"
                )));
            }
            let values: Vec<f32> = it.into_f32().collect();
            if !values.iter().all(|v| v.is_finite()) {
                return Err(invalid("weights channel has non-finite values"));
            }
            let wanted = times.len() * elements_per_key * morph_count;
            if values.len() != wanted {
                return Err(invalid(format!(
                    "weights channel has {} values, expected {wanted} \
                     ({} keys x {elements_per_key} x {morph_count} weights)",
                    values.len(),
                    times.len()
                )));
            }
            AnimValues::Weights {
                morph_count,
                values,
            }
        }
        None => return Err(invalid("animation channel has no outputs")),
    };
    Ok(values)
}

/// Rest-pose locals straight from each node's stored TRS.
pub fn rest_locals(nodes: &[NodeData]) -> Vec<super::Trs> {
    nodes
        .iter()
        .map(|n| super::Trs {
            translation: n.translation,
            rotation: n.rotation,
            scale: n.scale,
        })
        .collect()
}

/// Recomputes every node's global transform from `locals` (indexed by node).
pub fn compute_global_transforms(nodes: &[NodeData], locals: &[super::Trs], globals: &mut [Mat4]) {
    fn visit(
        nodes: &[NodeData],
        locals: &[super::Trs],
        globals: &mut [Mat4],
        index: usize,
        parent: Mat4,
    ) {
        let local = locals[index];
        let global = parent
            * Mat4::from_scale_rotation_translation(local.scale, local.rotation, local.translation);
        globals[index] = global;
        for &child in &nodes[index].children {
            visit(nodes, locals, globals, child, global);
        }
    }
    for index in 0..nodes.len() {
        if nodes[index].parent == usize::MAX {
            visit(nodes, locals, globals, index, Mat4::IDENTITY);
        }
    }
}
