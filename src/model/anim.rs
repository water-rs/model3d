//! glTF animation channel evaluation (STEP / LINEAR / CUBICSPLINE).

use glam::{Quat, Vec3, Vec4};

use super::{AnimPath, AnimValues, AnimationClip, ChannelData, InterpolationData};

/// Local transform triple evaluated per node per frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Trs {
    /// Translation.
    pub translation: Vec3,
    /// Rotation quaternion.
    pub rotation: Quat,
    /// Scale.
    pub scale: Vec3,
}

impl Trs {
    /// Identity transform.
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };
}

/// Evaluates `clip` at `time` (seconds) into `locals` (indexed by node) and
/// `morph_weights` (per node, indexed by node, each `morph_count` long).
///
/// `locals` must be initialized with the nodes' rest TRS before the call so
/// untouched channels keep their static pose.
pub fn evaluate_clip(
    clip: &AnimationClip,
    time: f32,
    locals: &mut [Trs],
    morph_weights: &mut [Vec<f32>],
) {
    for channel in &clip.channels {
        let Some(local) = locals.get_mut(channel.node) else {
            continue;
        };
        match (&channel.path, &channel.values) {
            (AnimPath::Translation, AnimValues::Vec3(v)) => {
                local.translation = sample_vec3(channel, time, v);
            }
            (AnimPath::Rotation, AnimValues::Quat(v)) => {
                local.rotation = sample_quat(channel, time, v);
            }
            (AnimPath::Scale, AnimValues::Vec3(v)) => {
                local.scale = sample_vec3(channel, time, v);
            }
            (
                AnimPath::MorphWeights,
                AnimValues::Weights {
                    morph_count,
                    values,
                },
            ) => {
                if let Some(out) = morph_weights.get_mut(channel.node) {
                    sample_weights(channel, time, *morph_count, values, out);
                }
            }
            _ => {}
        }
    }
}

/// Index of the key `t` falls on or after: `i` such that `times[i] <= t`.
/// Returns `0` before the first key and `len-1` after the last.
fn key_index(times: &[f32], t: f32) -> usize {
    times.partition_point(|&k| k <= t).saturating_sub(1)
}

/// Normalized position of `t` inside `[times[i], times[i+1]]`.
fn key_fraction(times: &[f32], i: usize, t: f32) -> (usize, usize, f32) {
    let next = (i + 1).min(times.len() - 1);
    if next == i {
        return (i, i, 0.0);
    }
    let span = times[next] - times[i];
    let factor = if span > 0.0 {
        ((t - times[i]) / span).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (i, next, factor)
}

fn sample_vec3(channel: &ChannelData, time: f32, values: &[Vec3]) -> Vec3 {
    match channel.interpolation {
        InterpolationData::Step => values[key_index(&channel.times, time)],
        InterpolationData::Linear => {
            let (a, b, factor) =
                key_fraction(&channel.times, key_index(&channel.times, time), time);
            values[a].lerp(values[b], factor)
        }
        InterpolationData::CubicSpline => {
            // Layout: [in_tangent, value, out_tangent] per key.
            let i = key_index(&channel.times, time);
            let (a, b, factor) = key_fraction(&channel.times, i, time);
            if a == b {
                return values[a * 3 + 1];
            }
            let dt = channel.times[b] - channel.times[a];
            hermite(
                values[a * 3 + 1],
                values[a * 3 + 2] * dt,
                values[b * 3 + 1],
                values[b * 3] * dt,
                factor,
            )
        }
    }
}

fn sample_quat(channel: &ChannelData, time: f32, values: &[Quat]) -> Quat {
    match channel.interpolation {
        InterpolationData::Step => values[key_index(&channel.times, time)],
        InterpolationData::Linear => {
            let (a, b, factor) =
                key_fraction(&channel.times, key_index(&channel.times, time), time);
            values[a]
                .normalize()
                .slerp(values[b].normalize(), factor)
                .normalize()
        }
        InterpolationData::CubicSpline => {
            let i = key_index(&channel.times, time);
            let (a, b, factor) = key_fraction(&channel.times, i, time);
            if a == b {
                return values[a * 3 + 1].normalize();
            }
            let dt = channel.times[b] - channel.times[a];
            // Cubic-spline quaternions interpolate component-wise, then
            // normalize (glTF spec).
            let out = hermite4(
                Vec4::from(values[a * 3 + 1]),
                Vec4::from(values[a * 3 + 2]) * dt,
                Vec4::from(values[b * 3 + 1]),
                Vec4::from(values[b * 3]) * dt,
                factor,
            );
            Quat::from_vec4(out).normalize()
        }
    }
}

fn sample_weights(
    channel: &ChannelData,
    time: f32,
    morph_count: usize,
    values: &[f32],
    out: &mut [f32],
) {
    let count = morph_count.min(out.len());
    match channel.interpolation {
        InterpolationData::Step => {
            let i = key_index(&channel.times, time);
            out[..count].copy_from_slice(&values[i * morph_count..i * morph_count + count]);
        }
        InterpolationData::Linear => {
            let (a, b, factor) =
                key_fraction(&channel.times, key_index(&channel.times, time), time);
            for k in 0..count {
                let va = values[a * morph_count + k];
                let vb = values[b * morph_count + k];
                out[k] = (vb - va).mul_add(factor, va);
            }
        }
        InterpolationData::CubicSpline => {
            let i = key_index(&channel.times, time);
            let (a, b, factor) = key_fraction(&channel.times, i, time);
            if a == b {
                out[..count].copy_from_slice(
                    &values[(a * 3 + 1) * morph_count..(a * 3 + 1) * morph_count + count],
                );
                return;
            }
            let dt = channel.times[b] - channel.times[a];
            for k in 0..count {
                let p0 = values[(a * 3 + 1) * morph_count + k];
                let m0 = values[(a * 3 + 2) * morph_count + k] * dt;
                let p1 = values[(b * 3 + 1) * morph_count + k];
                let m1 = values[b * 3 * morph_count + k] * dt;
                out[k] = cubic(p0, m0, p1, m1, factor);
            }
        }
    }
}

/// Scalar cubic Hermite.
fn cubic(p0: f32, m0: f32, p1: f32, m1: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    m1.mul_add(
        t3 - t2,
        p1.mul_add(
            3.0f32.mul_add(t2, -2.0 * t3),
            m0.mul_add(
                2.0f32.mul_add(-t2, t3) + t,
                p0 * (3.0f32.mul_add(-t2, 2.0 * t3) + 1.0),
            ),
        ),
    )
}

fn hermite(p0: Vec3, m0: Vec3, p1: Vec3, m1: Vec3, t: f32) -> Vec3 {
    Vec3::new(
        cubic(p0.x, m0.x, p1.x, m1.x, t),
        cubic(p0.y, m0.y, p1.y, m1.y, t),
        cubic(p0.z, m0.z, p1.z, m1.z, t),
    )
}

fn hermite4(p0: Vec4, m0: Vec4, p1: Vec4, m1: Vec4, t: f32) -> Vec4 {
    let t2 = t * t;
    let t3 = t2 * t;
    p0 * (3.0f32.mul_add(-t2, 2.0 * t3) + 1.0)
        + m0 * (2.0f32.mul_add(-t2, t3) + t)
        + p1 * 3.0f32.mul_add(t2, -2.0 * t3)
        + m1 * (t3 - t2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn channel(times: Vec<f32>, interp: InterpolationData, values: AnimValues) -> ChannelData {
        ChannelData {
            node: 0,
            path: AnimPath::Translation,
            interpolation: interp,
            times,
            values,
        }
    }

    #[test]
    fn linear_vec3_interpolates() {
        let c = channel(
            vec![0.0, 1.0],
            InterpolationData::Linear,
            AnimValues::Vec3(vec![Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)]),
        );
        let v = match &c.values {
            AnimValues::Vec3(v) => v.clone(),
            _ => unreachable!(),
        };
        assert_eq!(sample_vec3(&c, 0.5, &v), Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn step_holds_previous_key() {
        let c = channel(
            vec![0.0, 1.0],
            InterpolationData::Step,
            AnimValues::Vec3(vec![Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)]),
        );
        let v = match &c.values {
            AnimValues::Vec3(v) => v.clone(),
            _ => unreachable!(),
        };
        assert_eq!(sample_vec3(&c, 0.9, &v), Vec3::ZERO);
        assert_eq!(sample_vec3(&c, 1.0, &v), Vec3::new(2.0, 0.0, 0.0));
    }

    #[test]
    fn linear_quat_slerps() {
        let start = Quat::IDENTITY;
        let end = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let c = ChannelData {
            node: 0,
            path: AnimPath::Rotation,
            interpolation: InterpolationData::Linear,
            times: vec![0.0, 1.0],
            values: AnimValues::Quat(vec![start, end]),
        };
        let v = match &c.values {
            AnimValues::Quat(v) => v.clone(),
            _ => unreachable!(),
        };
        let q = sample_quat(&c, 0.5, &v);
        let expected = Quat::from_rotation_z(std::f32::consts::FRAC_PI_4);
        assert!((q.dot(expected)).abs() > 0.999);
    }

    #[test]
    fn cubic_spline_vec3_smooth() {
        // Single segment, zero tangents -> smoothstep between keys.
        let c = channel(
            vec![0.0, 1.0],
            InterpolationData::CubicSpline,
            AnimValues::Vec3(vec![
                Vec3::ZERO,
                Vec3::ZERO,
                Vec3::ZERO, // key0: in, value, out
                Vec3::ZERO,
                Vec3::X,
                Vec3::ZERO, // key1
            ]),
        );
        let v = match &c.values {
            AnimValues::Vec3(v) => v.clone(),
            _ => unreachable!(),
        };
        let mid = sample_vec3(&c, 0.5, &v);
        assert!((mid.x - 0.5).abs() < 1e-6);
    }

    #[test]
    fn evaluate_updates_locals() {
        let clip = AnimationClip {
            name: None,
            duration: Duration::from_secs(1),
            channels: vec![ChannelData {
                node: 0,
                path: AnimPath::Translation,
                interpolation: InterpolationData::Linear,
                times: vec![0.0, 1.0],
                values: AnimValues::Vec3(vec![Vec3::ZERO, Vec3::new(1.0, 2.0, 3.0)]),
            }],
        };
        let mut locals = vec![Trs::IDENTITY];
        let mut morphs = vec![Vec::new()];
        evaluate_clip(&clip, 0.5, &mut locals, &mut morphs);
        assert_eq!(locals[0].translation, Vec3::new(0.5, 1.0, 1.5));
    }
}
