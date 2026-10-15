//! Validated orbit camera for the model viewer.
//!
//! [`OrbitCamera`] keeps its state in private fields; every mutation goes
//! through a validated mutator, so an `OrbitCamera` value always represents a
//! renderable pose — finite coordinates, positive distance, sane clip planes
//! and a valid field of view. Invalid input is rejected with [`CameraError`]
//! rather than stored.

use core::fmt;

use glam::{Mat4, Quat, Vec3};

use crate::model::Bounds;

/// Pitch is clamped just below the poles to keep the orbit basis stable.
const PITCH_LIMIT: f32 = core::f32::consts::FRAC_PI_2 - 0.02;
const DEFAULT_FOV_Y: f32 = 0.79; // ~45°
const DEFAULT_DISTANCE: f32 = 3.0;
const NEAR_RATIO: f32 = 0.01;
const FAR_RATIO: f32 = 100.0;

/// An invalid camera mutation was rejected.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum CameraError {
    /// A value was NaN or infinite.
    NotFinite,
    /// Distance must be positive.
    InvalidDistance,
    /// Requires `0 < near < far`.
    InvalidClipPlanes,
    /// Vertical FOV must be within (0, π).
    InvalidFov,
}

impl fmt::Display for CameraError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFinite => write!(f, "camera parameter is not finite"),
            Self::InvalidDistance => write!(f, "orbit distance must be positive"),
            Self::InvalidClipPlanes => write!(f, "requires 0 < near < far"),
            Self::InvalidFov => write!(f, "field of view must be in (0, pi)"),
        }
    }
}

impl std::error::Error for CameraError {}

const fn finite_vec3(v: Vec3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

/// An orbit camera looking at `target` from a spherical pose.
///
/// Construct with [`OrbitCamera::new`], [`OrbitCamera::fit`], or
/// [`OrbitCamera::default`]; read state via the accessors and move it via the
/// mutators (`set_*`, [`orbit_by`](Self::orbit_by),
/// [`zoom_by`](Self::zoom_by), [`pan_by`](Self::pan_by)).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbitCamera {
    target: Vec3,
    yaw: f32,
    pitch: f32,
    distance: f32,
    fov_y: f32,
    near: f32,
    far: f32,
    /// `true` once the caller supplied clip planes explicitly; `zoom_by`
    /// then keeps them instead of re-deriving them from the distance.
    clip_explicit: bool,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self::new()
    }
}

impl OrbitCamera {
    /// A camera at a neutral pose looking at the origin.
    #[must_use]
    pub fn new() -> Self {
        Self {
            target: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.3,
            distance: DEFAULT_DISTANCE,
            fov_y: DEFAULT_FOV_Y,
            near: DEFAULT_DISTANCE * NEAR_RATIO,
            far: DEFAULT_DISTANCE * FAR_RATIO,
            clip_explicit: false,
        }
    }

    /// A camera framing `bounds` at the given viewport `aspect` (width/height).
    ///
    /// The orbit distance fits the bounding sphere inside *both* the vertical
    /// and horizontal frustums, so narrow portrait viewports zoom out rather
    /// than cropping the model. Returns `None` when `bounds` is empty — there
    /// is nothing to frame, and inventing a radius would hide that.
    #[must_use]
    pub fn fit(bounds: &Bounds, aspect: f32) -> Option<Self> {
        if !bounds.is_valid() || !aspect.is_finite() || aspect <= 0.0 {
            return None;
        }
        let center = bounds.center();
        let radius = bounds.radius();
        let mut camera = Self::new();
        camera.target = center;
        let distance = camera.fit_distance(radius, aspect);
        camera.distance = distance.max(1e-4);
        camera.near = 2.0f32
            .mul_add(-radius, camera.distance)
            .max(camera.distance * NEAR_RATIO);
        camera.far = 4.0f32
            .mul_add(radius, camera.distance)
            .max(camera.distance * FAR_RATIO);
        Some(camera)
    }

    /// The look-at point.
    #[must_use]
    pub const fn target(&self) -> Vec3 {
        self.target
    }

    /// Orbit yaw around the vertical axis, radians.
    #[must_use]
    pub const fn yaw(&self) -> f32 {
        self.yaw
    }

    /// Orbit pitch, radians, clamped within ±(π/2 − 0.02).
    #[must_use]
    pub const fn pitch(&self) -> f32 {
        self.pitch
    }

    /// Distance from target to eye.
    #[must_use]
    pub const fn distance(&self) -> f32 {
        self.distance
    }

    /// Vertical field of view, radians.
    #[must_use]
    pub const fn fov_y(&self) -> f32 {
        self.fov_y
    }

    /// Near clip distance.
    #[must_use]
    pub const fn near(&self) -> f32 {
        self.near
    }

    /// Far clip distance.
    #[must_use]
    pub const fn far(&self) -> f32 {
        self.far
    }

    /// Sets the look-at point.
    ///
    /// # Errors
    /// [`CameraError::NotFinite`] when `target` contains NaN or infinity.
    pub const fn set_target(&mut self, target: Vec3) -> Result<(), CameraError> {
        if !finite_vec3(target) {
            return Err(CameraError::NotFinite);
        }
        self.target = target;
        Ok(())
    }

    /// Sets yaw (radians). Wrapped into (−π, π].
    ///
    /// # Errors
    /// [`CameraError::NotFinite`] when `yaw` is not finite.
    pub fn set_yaw(&mut self, yaw: f32) -> Result<(), CameraError> {
        if !yaw.is_finite() {
            return Err(CameraError::NotFinite);
        }
        self.yaw = yaw.rem_euclid(2.0 * core::f32::consts::PI);
        Ok(())
    }

    /// Sets pitch (radians), clamped within ±(π/2 − 0.02).
    ///
    /// # Errors
    /// [`CameraError::NotFinite`] when `pitch` is not finite.
    pub fn set_pitch(&mut self, pitch: f32) -> Result<(), CameraError> {
        if !pitch.is_finite() {
            return Err(CameraError::NotFinite);
        }
        self.pitch = pitch.clamp(-PITCH_LIMIT, PITCH_LIMIT);
        Ok(())
    }

    /// Sets orbit distance; must be positive.
    ///
    /// # Errors
    /// [`CameraError::InvalidDistance`] when `distance` is non-positive or not
    /// finite.
    pub fn set_distance(&mut self, distance: f32) -> Result<(), CameraError> {
        if !(distance.is_finite() && distance > 0.0) {
            return Err(CameraError::InvalidDistance);
        }
        self.distance = distance;
        Ok(())
    }

    /// Sets the clip planes together so `near < far` is preserved atomically.
    ///
    /// # Errors
    /// [`CameraError::InvalidClipPlanes`] when `0 < near < far` does not hold.
    pub fn set_clip_planes(&mut self, near: f32, far: f32) -> Result<(), CameraError> {
        if !(near.is_finite() && far.is_finite() && near > 0.0 && near < far) {
            return Err(CameraError::InvalidClipPlanes);
        }
        self.near = near;
        self.far = far;
        self.clip_explicit = true;
        Ok(())
    }

    /// Sets the vertical field of view (radians, in (0, π)).
    ///
    /// # Errors
    /// [`CameraError::InvalidFov`] when `fov_y` is out of range.
    pub fn set_fov_y(&mut self, fov_y: f32) -> Result<(), CameraError> {
        if !(fov_y.is_finite() && fov_y > 0.0 && fov_y < core::f32::consts::PI) {
            return Err(CameraError::InvalidFov);
        }
        self.fov_y = fov_y;
        Ok(())
    }

    /// Rotates the orbit by `(dyaw, dpitch)` radians; pitch clamps at the
    /// poles.
    ///
    /// # Errors
    /// [`CameraError::NotFinite`] when either delta is not finite.
    pub fn orbit_by(&mut self, dyaw: f32, dpitch: f32) -> Result<(), CameraError> {
        if !(dyaw.is_finite() && dpitch.is_finite()) {
            return Err(CameraError::NotFinite);
        }
        self.yaw = (self.yaw + dyaw).rem_euclid(2.0 * core::f32::consts::PI);
        self.pitch = (self.pitch + dpitch).clamp(-PITCH_LIMIT, PITCH_LIMIT);
        Ok(())
    }

    /// Multiplies the orbit distance by `factor` (must be positive and
    /// finite), then expands the clip planes to stay proportional.
    ///
    /// # Errors
    /// [`CameraError::InvalidDistance`] when the factor is non-positive or
    /// non-finite.
    pub fn zoom_by(&mut self, factor: f32) -> Result<(), CameraError> {
        if !(factor.is_finite() && factor > 0.0) {
            return Err(CameraError::InvalidDistance);
        }
        self.distance = (self.distance * factor).max(1e-4);
        // Caller-supplied clip planes stay untouched; the proportional
        // defaults track the distance so zooming out cannot clip the model.
        if !self.clip_explicit {
            self.near = self.distance * NEAR_RATIO;
            self.far = self.distance * FAR_RATIO;
        }
        Ok(())
    }

    /// Pans the target in the camera's right/up plane.
    ///
    /// # Errors
    /// [`CameraError::NotFinite`] when either delta is not finite.
    pub fn pan_by(&mut self, dx: f32, dy: f32) -> Result<(), CameraError> {
        if !(dx.is_finite() && dy.is_finite()) {
            return Err(CameraError::NotFinite);
        }
        let (right, up, _) = self.basis();
        self.target += right * dx + up * dy;
        Ok(())
    }

    /// Eye position in world space.
    #[must_use]
    pub fn eye(&self) -> Vec3 {
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let offset = Vec3::new(cos_pitch * sin_yaw, sin_pitch, cos_pitch * cos_yaw);
        self.target + offset * self.distance
    }

    /// Camera basis: (right, up, forward) unit vectors.
    #[must_use]
    pub fn basis(&self) -> (Vec3, Vec3, Vec3) {
        let forward = (self.target - self.eye()).normalize_or(Vec3::Z);
        let right = forward.cross(Vec3::Y).normalize_or(Vec3::X);
        let up = right.cross(forward);
        (right, up, forward)
    }

    /// Distance that fits a bounding `sphere_radius` at `aspect`.
    ///
    /// A sphere of radius `r` tangent to the frustum sits at `r / sin(θ)`
    /// for half-angle `θ`; the horizontal half-angle derives from the
    /// vertical one through `tan θx = tan θy * aspect`.
    fn fit_distance(&self, sphere_radius: f32, aspect: f32) -> f32 {
        let radius = sphere_radius.max(f32::EPSILON);
        let half_fov_y = self.fov_y * 0.5;
        let vertical = radius / half_fov_y.sin();
        let tan_x = half_fov_y.tan() * aspect;
        // sin θx = tan θx / sqrt(1 + tan² θx)
        let sin_x = tan_x / tan_x.mul_add(tan_x, 1.0).sqrt();
        let horizontal = radius / sin_x;
        (vertical.max(horizontal)) * 1.1 // small margin around the sphere
    }

    /// View matrix (world → camera).
    #[must_use]
    pub fn view(&self) -> Mat4 {
        glam::camera::rh::view::look_at_mat4(self.eye(), self.target, Vec3::Y)
    }

    /// Projection matrix (camera → clip) for the given viewport aspect, in
    /// wgpu/DirectX clip space (z ∈ [0, 1]).
    #[must_use]
    pub fn projection(&self, aspect: f32) -> Mat4 {
        glam::camera::rh::proj::directx::perspective(
            self.fov_y,
            aspect.max(1e-6),
            self.near,
            self.far,
        )
    }

    /// Combined `projection * view`.
    #[must_use]
    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        self.projection(aspect) * self.view()
    }
}

/// Rotation-only helper for reusing the camera's orbit orientation.
impl From<&OrbitCamera> for Quat {
    fn from(camera: &OrbitCamera) -> Self {
        Self::from_rotation_y(camera.yaw) * Self::from_rotation_x(camera.pitch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn fit_respects_horizontal_frustum() {
        // A wide model at portrait aspect must pull farther back than at
        // square aspect: the horizontal frustum becomes the constraint.
        let mut bounds = Bounds::empty();
        bounds.include(Vec3::new(-4.0, -0.5, -0.5));
        bounds.include(Vec3::new(4.0, 0.5, 0.5));
        let portrait = OrbitCamera::fit(&bounds, 0.25).unwrap();
        let square = OrbitCamera::fit(&bounds, 1.0).unwrap();
        assert!(portrait.distance() > square.distance());
    }

    #[test]
    fn fit_rejects_empty_bounds() {
        assert!(OrbitCamera::fit(&Bounds::empty(), 1.0).is_none());
    }

    #[test]
    fn invalid_mutations_are_rejected() {
        let mut camera = OrbitCamera::new();
        assert_eq!(camera.set_distance(-1.0), Err(CameraError::InvalidDistance));
        assert_eq!(
            camera.set_clip_planes(2.0, 1.0),
            Err(CameraError::InvalidClipPlanes)
        );
        assert_eq!(camera.set_fov_y(f32::NAN), Err(CameraError::InvalidFov));
        assert_eq!(camera.set_target(Vec3::NAN), Err(CameraError::NotFinite));
        assert_eq!(
            camera.orbit_by(f32::INFINITY, 0.0),
            Err(CameraError::NotFinite)
        );
        // Rejected mutations leave the prior state intact.
        assert_eq!(camera, OrbitCamera::new());
    }

    #[test]
    fn pitch_is_clamped() {
        let mut camera = OrbitCamera::new();
        camera.set_pitch(10.0).unwrap();
        assert!(camera.pitch() < core::f32::consts::FRAC_PI_2);
        camera.set_pitch(-10.0).unwrap();
        assert!(camera.pitch() > -core::f32::consts::FRAC_PI_2);
    }

    #[test]
    fn view_projection_is_finite() {
        let mut bounds = Bounds::empty();
        bounds.include(Vec3::new(-1.0, -1.0, -1.0));
        bounds.include(Vec3::new(1.0, 1.0, 1.0));
        let camera = OrbitCamera::fit(&bounds, 1.6).unwrap();
        let vp = camera.view_projection(1.6);
        assert!(vp.to_cols_array().iter().all(|v| v.is_finite()));
    }
}
