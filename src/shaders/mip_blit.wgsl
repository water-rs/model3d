// Fullscreen blit for mipmap generation. Source and target views carry the
// texture's native format, so sRGB textures decode on sample and re-encode
// on store — the mip chain filters encoded values correctly.

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_smp: sampler;

struct Out {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs_mip(@builtin(vertex_index) vi: u32) -> Out {
    var o: Out;
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    o.pos = vec4f(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    o.uv = vec2f(x, y);
    return o;
}

@fragment
fn fs_mip(o: Out) -> @location(0) vec4f {
    return textureSample(src_tex, src_smp, o.uv);
}
