//! DSP primitives: crossfade curves, envelopes, FFT utilities.

pub mod crossfade;
pub mod envelope;
pub mod filters;
pub mod stretch;

pub use crossfade::{
    beat_aligned_crossfade, crossfade_bass_swap, crossfade_equal_power, crossfade_linear,
    pick_a_fade_start, CrossfadeCurve,
};
pub use envelope::{apply_gain, rms};
pub use filters::Biquad;
pub use stretch::{bpm_match_ratio, turntable_stretch};
