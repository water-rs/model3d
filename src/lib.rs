//! # waterui-model3d
//!
//! A glTF 2.0 / GLB 3D model viewer for `WaterUI`, rendered with wgpu through
//! `waterui-graphics`' [`GpuContent`]. [`Model`] is an immutable, cheap-clone
//! asset handle; [`Model3D`] is the view.
//!

mod camera;
mod error;
mod model;
mod playback;
mod render;
mod viewer;

pub use camera::{CameraError, OrbitCamera};
pub use error::{InvalidClipError, LoadError};
pub use model::{AlphaMode, AnimationClip, Bounds, ClipHandle, Model};
pub use playback::{PlayState, Playback};
pub use viewer::{Interaction, Lighting, Model3D};
