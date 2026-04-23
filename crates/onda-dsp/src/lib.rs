//! DSP primitives: crossfade curves, envelopes, FFT utilities.

pub mod crossfade;
pub mod envelope;

pub use crossfade::{
    beat_aligned_crossfade, crossfade_equal_power, crossfade_linear, CrossfadeCurve,
};
pub use envelope::{apply_gain, rms};
