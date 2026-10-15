//! Error types returned by [`crate::Model`] loading and clip selection.

use core::fmt;
use std::path::PathBuf;

/// Failure to load a glTF asset into a [`crate::Model`].
#[derive(Debug)]
#[non_exhaustive]
pub enum LoadError {
    /// Filesystem I/O failed.
    Io {
        /// Path that could not be read.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// The glTF importer rejected the asset.
    Parse(gltf::Error),
    /// The asset lists a `extensionsRequired` entry the viewer does not
    /// implement.
    UnsupportedRequiredExtension(String),
    /// An in-memory load references an external URI; there is no resolver for
    /// sidecar files in [`crate::Model::from_glb`].
    ExternalUri(String),
    /// The asset is structurally invalid for this viewer.
    InvalidData(String),
    /// An image could not be decoded into RGBA8 pixels.
    Image(String),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(f, "failed to read {}: {source}", path.display())
            }
            Self::Parse(e) => write!(f, "glTF parse failed: {e}"),
            Self::UnsupportedRequiredExtension(ext) => {
                write!(f, "unsupported required glTF extension: {ext}")
            }
            Self::ExternalUri(uri) => {
                write!(f, "in-memory load references external URI: {uri}")
            }
            Self::InvalidData(msg) => write!(f, "invalid glTF data: {msg}"),
            Self::Image(msg) => write!(f, "image decode failed: {msg}"),
        }
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Parse(e) => Some(e),
            _ => None,
        }
    }
}

impl From<gltf::Error> for LoadError {
    fn from(e: gltf::Error) -> Self {
        Self::Parse(e)
    }
}

/// A [`crate::ClipHandle`] could not be resolved against a [`crate::Model`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InvalidClipError {
    /// The handle's clip index is outside the model's clip list.
    OutOfRange {
        /// Index that was requested.
        index: usize,
        /// Number of clips the model actually has.
        clip_count: usize,
    },
    /// The handle was created for a different model instance. Handles carry
    /// the identity of the model they were minted from and are never
    /// reinterpreted against another asset.
    WrongModel,
    /// The model the handle was minted from has been dropped.
    DroppedModel,
}

impl fmt::Display for InvalidClipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfRange { index, clip_count } => {
                write!(
                    f,
                    "clip index {index} out of range (model has {clip_count} clips)"
                )
            }
            Self::WrongModel => write!(f, "clip handle belongs to a different model"),
            Self::DroppedModel => write!(f, "clip handle's model has been dropped"),
        }
    }
}

impl std::error::Error for InvalidClipError {}
