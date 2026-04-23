//! DSP primitives: crossfade curves, envelopes, FFT utilities.

pub mod crossfade;
pub mod envelope;

pub use crossfade::{crossfade_linear, crossfade_equal_power, CrossfadeCurve};
pub use envelope::{apply_gain, rms};
