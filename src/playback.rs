//! Animation playback state for [`crate::Model3D`].
//!
//! [`Playback`] is a value type held in a `Binding<Playback>`: the view
//! advances it by each frame's delta, and the application can select clips,
//! toggle playing/paused, loop, and seek. Clip selection takes a
//! [`ClipHandle`] minted by [`crate::Model::clip_handle`] or
//! [`crate::Model::clip_named`]; invalid handles surface as
//! [`InvalidClipError`] here — before render — never as silent clamping.

use std::time::Duration;

use crate::error::InvalidClipError;
use crate::model::{ClipHandle, Model};

/// Reconstructs a `Duration` from total nanoseconds without truncation.
fn duration_from_nanos(nanos: u128) -> Duration {
    Duration::new(
        u64::try_from(nanos / 1_000_000_000).unwrap_or(u64::MAX),
        u32::try_from(nanos % 1_000_000_000).unwrap_or(0),
    )
}

/// Whether the clock advances.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayState {
    /// Time advances by frame delta.
    Playing,
    /// Time is frozen.
    Paused,
}

/// Animation playback state: which clip, where, and whether it's running.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Playback {
    clip: Option<ClipHandle>,
    state: PlayState,
    looping: bool,
    /// Position within the clip.
    time: Duration,
}

impl Default for Playback {
    fn default() -> Self {
        Self::new()
    }
}

impl Playback {
    /// No clip selected, paused.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            clip: None,
            state: PlayState::Paused,
            looping: true,
            time: Duration::ZERO,
        }
    }

    /// The selected clip handle, if any.
    #[must_use]
    pub const fn clip(&self) -> Option<&ClipHandle> {
        self.clip.as_ref()
    }

    /// Selects a clip on `model`, validated immediately.
    ///
    /// The playback position resets to zero. A handle minted from a *different*
    /// model is rejected — after a model swap an old handle cannot silently
    /// retarget the new asset's same-index clip.
    ///
    /// # Errors
    /// [`InvalidClipError::WrongModel`], [`InvalidClipError::DroppedModel`],
    /// or [`InvalidClipError::OutOfRange`].
    pub fn select_clip(
        &mut self,
        model: &Model,
        handle: ClipHandle,
    ) -> Result<(), InvalidClipError> {
        // Validates identity + index up front.
        model.clip(&handle)?;
        self.clip = Some(handle);
        self.time = Duration::ZERO;
        Ok(())
    }

    /// Clears the clip selection; the model renders its rest pose.
    pub fn clear_clip(&mut self) {
        self.clip = None;
        self.time = Duration::ZERO;
    }

    /// Play state.
    #[must_use]
    pub const fn state(&self) -> PlayState {
        self.state
    }

    /// Whether a playing clip's clock is advancing.
    #[must_use]
    pub fn is_playing(&self) -> bool {
        self.state == PlayState::Playing
    }

    /// Starts the clock.
    pub const fn play(&mut self) {
        self.state = PlayState::Playing;
    }

    /// Freezes the clock.
    pub const fn pause(&mut self) {
        self.state = PlayState::Paused;
    }

    /// Whether the clip loops.
    #[must_use]
    pub const fn is_looping(&self) -> bool {
        self.looping
    }

    /// Sets looping.
    pub const fn set_looping(&mut self, looping: bool) {
        self.looping = looping;
    }

    /// Current position within the clip.
    #[must_use]
    pub const fn time(&self) -> Duration {
        self.time
    }

    /// Seeks to `time`, clamped to the clip's duration when a clip is
    /// selected on `model`.
    ///
    /// # Errors
    /// [`InvalidClipError`] when the stored handle no longer resolves on
    /// `model` (e.g. the model was replaced).
    pub fn seek(&mut self, model: &Model, time: Duration) -> Result<(), InvalidClipError> {
        match &self.clip {
            Some(handle) => {
                let clip = model.clip(handle)?;
                self.time = time.min(clip.duration());
            }
            None => self.time = time,
        }
        Ok(())
    }

    /// Advances the clock by `delta` (frame delta, not lifetime elapsed —
    /// resuming playback never jumps), returns the in-clip sample time.
    ///
    /// Looping clips wrap; non-looping clips clamp at the end and pause.
    pub(crate) fn advance(&mut self, delta: Duration, duration: Duration) -> Duration {
        if self.state != PlayState::Playing || duration.is_zero() {
            return self.time.min(duration);
        }
        // u128 total nanos cannot overflow Duration's range.
        let total = self.time.as_nanos() + delta.as_nanos();
        let clip_nanos = duration.as_nanos();
        if self.looping {
            let wrapped = total % clip_nanos.max(1);
            self.time = duration_from_nanos(wrapped);
        } else if total >= clip_nanos {
            self.time = duration;
            self.state = PlayState::Paused;
        } else {
            self.time = duration_from_nanos(total);
        }
        self.time
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn playback_with_dummy_clip() -> Playback {
        // `advance` exercises clock math only; a selected clip isn't needed.
        let mut p = Playback::new();
        p.play();
        p
    }

    #[test]
    fn resume_does_not_jump() {
        let mut p = playback_with_dummy_clip();
        p.advance(Duration::from_millis(500), Duration::from_secs(2));
        p.pause();
        // A long lifetime gap between frames must not move the clock.
        p.advance(Duration::from_secs(60), Duration::from_secs(2));
        assert_eq!(p.time(), Duration::from_millis(500));
        p.play();
        p.advance(Duration::from_millis(100), Duration::from_secs(2));
        assert_eq!(p.time(), Duration::from_millis(600));
    }

    #[test]
    fn looping_wraps() {
        let mut p = playback_with_dummy_clip();
        p.advance(Duration::from_millis(2500), Duration::from_secs(2));
        assert_eq!(p.time(), Duration::from_millis(500));
        assert!(p.is_playing());
    }

    #[test]
    fn non_looping_clamps_and_pauses() {
        let mut p = playback_with_dummy_clip();
        p.set_looping(false);
        p.advance(Duration::from_millis(2500), Duration::from_secs(2));
        assert_eq!(p.time(), Duration::from_secs(2));
        assert!(!p.is_playing());
    }
}
