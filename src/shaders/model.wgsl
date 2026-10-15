// Metallic-roughness PBR for waterui-model3d.
// Bind groups: 0 = globals, 1 = scene (instances, joints, weights, morphs),
// 2 = material. The module is compiled through cherenkov's shader_module,
// which prepends the shared color helpers (`cherenkov_linear_srgb`) —
// shading runs in linear sRGB and the fragment converts to the target's
// premultiplied linear Display-P3 encoding.
//
// The compositor consumes premultiplied alpha: BLEND draws emit straight
// color and let ALPHA_BLENDING's src-alpha term premultiply, OPAQUE/MASK
// force alpha to 1.

struct Globals {
    view_proj: mat4x4f,          // 0
    view: mat4x4f,               // 64
    camera_pos: vec3f,           // 128
    light_count: u32,            // 140
    light_dir: vec3f,            // 144 (direction light travels toward)
    exposure: f32,               // 156
    light_color: vec3f,          // 160 (already intensity-scaled)
    _pad0: f32,
    fill_dir: vec3f,             // 176
    _pad1: f32,
    fill_color: vec3f,           // 192 (already intensity-scaled)
    _pad2: f32,
    ambient: vec4f,              // 208 (rgb color, w intensity)
};

struct Instance {
    world: mat4x4f,
    normal: mat4x4f,
    joint_base: u32,
    weight_base: u32,
    morph_offset: u32,
    morph_count: u32,
    vertex_base: u32,
    vertex_count: u32,
    joint_count: u32,
    _pad: u32,
};

struct MorphDelta {
    position: vec4f,
    normal: vec4f,
    tangent: vec4f,
};

struct Material {
    base_color_factor: vec4f,
    emissive_factor: vec4f,   // rgb already carries emissive_strength
    metallic_factor: f32,
    roughness_factor: f32,
    normal_scale: f32,
    occlusion_strength: f32,
    alpha_cutoff: f32,
    unlit: u32,
    alpha_mode: u32,    // 0 opaque, 1 mask, 2 blend
    double_sided: u32,
    // Per-texture-slot KHR_texture_transform:
    // slots_a[i] = (uv set index, offset.x, offset.y, rotation)
    // slots_b[i] = (scale.x, scale.y, -, -)
    slots_a: array<vec4f, 5>,
    slots_b: array<vec4f, 5>,
};

@group(0) @binding(0) var<uniform> globals: Globals;

@group(1) @binding(0) var<storage, read> instances: array<Instance>;
@group(1) @binding(1) var<storage, read> joints: array<mat4x4f>;
@group(1) @binding(2) var<storage, read> weights: array<f32>;
@group(1) @binding(3) var<storage, read> morphs: array<MorphDelta>;

@group(2) @binding(0) var<uniform> material: Material;
@group(2) @binding(1) var base_color_tex: texture_2d<f32>;
@group(2) @binding(2) var base_color_samp: sampler;
@group(2) @binding(3) var mr_tex: texture_2d<f32>;
@group(2) @binding(4) var mr_samp: sampler;
@group(2) @binding(5) var normal_tex: texture_2d<f32>;
@group(2) @binding(6) var normal_samp: sampler;
@group(2) @binding(7) var occlusion_tex: texture_2d<f32>;
@group(2) @binding(8) var occlusion_samp: sampler;
@group(2) @binding(9) var emissive_tex: texture_2d<f32>;
@group(2) @binding(10) var emissive_samp: sampler;

struct VsIn {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) tangent: vec4f,
    @location(3) color: vec4f,
    @location(4) uv0: vec2f,
    @location(5) uv1: vec2f,
    @location(6) joints_in: vec4u,
    @location(7) weights_in: vec4f,
};

struct VsOut {
    @builtin(position) position: vec4f,
    @location(0) world_pos: vec3f,
    @location(1) normal: vec3f,
    @location(2) tangent: vec4f,
    @location(3) color: vec4f,
    @location(4) uv0: vec2f,
    @location(5) uv1: vec2f,
};

@vertex
fn vs_main(
    in: VsIn,
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VsOut {
    let inst = instances[instance_index];
    var pos = in.position;
    var nrm = in.normal;
    var tan4 = vec4f(in.tangent.xyz, in.tangent.w);

    // Morph targets first (glTF spec order: morph then skin). Weights are
    // unclamped — authored defaults and animations may exceed [0, 1].
    if inst.morph_count > 0u {
        let local_vertex = vertex_index - inst.vertex_base;
        for (var t = 0u; t < inst.morph_count; t++) {
            let w = weights[inst.weight_base + t];
            let d = morphs[inst.morph_offset + t * inst.vertex_count + local_vertex];
            pos += d.position.xyz * w;
            nrm += d.normal.xyz * w;
            tan4 = vec4f(tan4.xyz + d.tangent.xyz * w, tan4.w);
        }
    }

    if inst.joint_count > 0u {
        // Linear blend skinning over this node's joint palette.
        let j = in.joints_in;
        let w = in.weights_in;
        let skin_mat = joints[inst.joint_base + j.x] * w.x
            + joints[inst.joint_base + j.y] * w.y
            + joints[inst.joint_base + j.z] * w.z
            + joints[inst.joint_base + j.w] * w.w;
        pos = (skin_mat * vec4f(pos, 1.0)).xyz;
        let skin3 = mat3x3f(skin_mat[0].xyz, skin_mat[1].xyz, skin_mat[2].xyz);
        // Normals take the inverse-transpose of the blended linear part; the
        // cofactor matrix equals det * A^-T, so sign(det) divides the
        // determinant out — including the negative-determinate mirror flip.
        nrm = cofactor3(skin3) * nrm * sign(determinant(skin3));
        tan4 = vec4f(skin3 * tan4.xyz, tan4.w * sign(determinant(skin3)));
    }

    let world = inst.world * vec4f(pos, 1.0);
    var out: VsOut;
    out.position = globals.view_proj * world;
    out.world_pos = world.xyz;
    out.normal = (inst.normal * vec4f(nrm, 0.0)).xyz;
    // Tangents transform by the linear part of world (not the
    // inverse-transpose); the fragment re-orthogonalizes against the normal.
    let world3 = mat3x3f(inst.world[0].xyz, inst.world[1].xyz, inst.world[2].xyz);
    out.tangent = vec4f(
        (inst.world * vec4f(tan4.xyz, 0.0)).xyz,
        tan4.w * sign(determinant(world3)),
    );
    out.color = in.color;
    out.uv0 = in.uv0;
    out.uv1 = in.uv1;
    return out;
}

// UV set selection + KHR_texture_transform for texture slot `slot`:
// offset + rotation * (scale * uv).
fn tex_uv(slot: u32, uv0: vec2f, uv1: vec2f) -> vec2f {
    let a = material.slots_a[slot];
    let b = material.slots_b[slot];
    var uv = uv0;
    if a.x > 0.5 {
        uv = uv1;
    }
    let scaled = uv * b.xy;
    let c = cos(a.w);
    let s = sin(a.w);
    let rotated = vec2f(c * scaled.x - s * scaled.y, s * scaled.x + c * scaled.y);
    return rotated + a.yz;
}

fn normal_from_map(n: vec3f, t: vec4f, uv: vec2f, front: bool) -> vec3f {
    let sample_n = textureSample(normal_tex, normal_samp, uv).xyz * 2.0 - 1.0;
    let scaled = vec3f(sample_n.xy * material.normal_scale, sample_n.z);
    var nn = normalize(n);
    // Double-sided surfaces flip the whole frame on backfaces.
    if material.double_sided > 0u && !front {
        nn = -nn;
    }
    let tt = t.xyz - nn * dot(nn, t.xyz);
    if dot(tt, tt) < 1e-8 {
        return nn;
    }
    let tb = normalize(tt);
    let b = cross(nn, tb) * t.w;
    return normalize(mat3x3f(tb, b, nn) * scaled);
}

/// Cofactor matrix of a 3x3 — inverse-transpose without the 1/det factor.
fn cofactor3(m: mat3x3f) -> mat3x3f {
    let c0 = m[0].xyz;
    let c1 = m[1].xyz;
    let c2 = m[2].xyz;
    return mat3x3f(cross(c1, c2), cross(c2, c0), cross(c0, c1));
}

const PI: f32 = 3.141592653589793;

// GGX / Cook-Torrance specular + Lambertian diffuse (metallic-roughness).
// `v` and `l` point away from the surface; `light_dir` is the direction the
// light travels *toward*.
fn pbr(n: vec3f, v: vec3f, l: vec3f, base: vec3f, metallic: f32, roughness: f32) -> vec3f {
    let ndotl = max(dot(n, l), 0.0);
    if ndotl <= 0.0 {
        return vec3f(0.0);
    }
    let h = normalize(v + l);
    let ndotv = max(dot(n, v), 1e-4);
    let ndoth = max(dot(n, h), 0.0);
    let vdoth = max(dot(v, h), 0.0);

    let f0 = mix(vec3f(0.04), base, metallic);
    let alpha = max(roughness * roughness, 1e-3);
    let a2 = alpha * alpha;

    // Distribution GGX.
    let denom = ndoth * ndoth * (a2 - 1.0) + 1.0;
    let d = a2 / (PI * denom * denom);

    // Fresnel-Schlick.
    let f = f0 + (1.0 - f0) * pow(1.0 - vdoth, 5.0);

    // Height-correlated Smith visibility V = G / (4 NdotV NdotL).
    let gv = ndotl * sqrt(ndotv * ndotv * (1.0 - a2) + a2);
    let gl = ndotv * sqrt(ndotl * ndotl * (1.0 - a2) + a2);
    let vis = 0.5 / max(gv + gl, 1e-6);

    let spec = d * vis * f;
    let kd = (vec3f(1.0) - f) * (1.0 - metallic);
    let diffuse = kd * base / PI;
    return (diffuse + spec) * ndotl;
}

fn shade_light(n: vec3f, v: vec3f, l: vec3f, radiance: vec3f, base: vec3f, metallic: f32, roughness: f32) -> vec3f {
    return pbr(n, v, l, base, metallic, roughness) * radiance;
}

@fragment
fn fs_main(in: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
    let uv_base = tex_uv(0u, in.uv0, in.uv1);
    var base = textureSample(base_color_tex, base_color_samp, uv_base)
        * material.base_color_factor * in.color;
    if material.alpha_mode == 1u && base.a < material.alpha_cutoff {
        discard;
    }

    if material.unlit > 0u {
        // KHR_materials_unlit: base color only, no lighting or emissive add.
        let rgb = cherenkov_linear_srgb(base.rgb);
        // BLEND keeps authored alpha; OPAQUE/MASK write opaque.
        let alpha = select(1.0, base.a, material.alpha_mode == 2u);
        return vec4f(rgb, alpha);
    }

    let mr = textureSample(mr_tex, mr_samp, tex_uv(1u, in.uv0, in.uv1)).rgb;
    let metallic = clamp(mr.b * material.metallic_factor, 0.0, 1.0);
    let roughness = clamp(mr.g * material.roughness_factor, 0.0, 1.0);
    let occlusion = mix(
        1.0,
        textureSample(occlusion_tex, occlusion_samp, tex_uv(3u, in.uv0, in.uv1)).r,
        material.occlusion_strength,
    );

    let n = normal_from_map(in.normal, in.tangent, tex_uv(2u, in.uv0, in.uv1), front);
    let v = normalize(globals.camera_pos - in.world_pos);
    let l = normalize(-globals.light_dir);

    var color = shade_light(n, v, l, globals.light_color, base.rgb, metallic, roughness);
    if globals.light_count > 1u {
        let lf = normalize(-globals.fill_dir);
        color += shade_light(n, v, lf, globals.fill_color, base.rgb, metallic, roughness);
    }
    color += base.rgb * globals.ambient.rgb * globals.ambient.w * occlusion;

    let emissive = textureSample(emissive_tex, emissive_samp, tex_uv(4u, in.uv0, in.uv1)).rgb
        * material.emissive_factor.rgb;
    color += emissive;

    color *= globals.exposure;

    // OPAQUE and MASK writes are opaque on the premultiplied target; BLEND
    // keeps straight alpha and the blend state premultiplies.
    let alpha = select(1.0, base.a, material.alpha_mode == 2u);
    return vec4f(cherenkov_linear_srgb(color), alpha);
}
