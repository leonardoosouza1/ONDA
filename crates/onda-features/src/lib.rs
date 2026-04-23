//! Music information retrieval (MIR) features: spectrum, onsets, tempo, beats.
//!
//! Pure-Rust, no ML, no ONNX. Classical DSP algorithms sufficient for the
//! pipeline's first pass. Learnable models plug in later on top of these
//! features.

pub mod beats;
pub mod onsets;
pub mod spectrum;
pub mod tempo;

pub use beats::{beat_grid_from_tempo, BeatGrid};
pub use onsets::{onset_envelope, OnsetParams};
pub use spectrum::{stft_mono, Spectrogram, StftParams};
pub use tempo::{estimate_tempo, TempoEstimate};
