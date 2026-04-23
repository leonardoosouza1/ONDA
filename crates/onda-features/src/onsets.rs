//! Onset detection via spectral flux.
//!
//! For each STFT frame, sum the *positive* magnitude differences from the
//! previous frame across bins. This "half-wave rectified flux" spikes at
//! transients (drum hits, sharp attacks) and is smooth between them.

use crate::spectrum::Spectrogram;

#[derive(Debug, Clone, Copy)]
pub struct OnsetParams {
    /// Smoothing window size (frames). 0 disables.
    pub smooth_win: usize,
}

impl Default for OnsetParams {
    fn default() -> Self {
        Self { smooth_win: 3 }
    }
}

/// Onset-strength envelope, one value per STFT frame (same frame rate as the
/// input spectrogram). Normalized to [0.0, 1.0].
pub fn onset_envelope(spec: &Spectrogram, params: OnsetParams) -> Vec<f32> {
    if spec.n_frames() < 2 {
        return vec![0.0; spec.n_frames()];
    }
    let n_frames = spec.n_frames();
    let n_bins = spec.n_bins();
    let mut flux = vec![0.0f32; n_frames];

    for f in 1..n_frames {
        let mut sum = 0.0;
        for b in 0..n_bins {
            let d = spec.magnitudes[f][b] - spec.magnitudes[f - 1][b];
            if d > 0.0 {
                sum += d;
            }
        }
        flux[f] = sum;
    }

    if params.smooth_win > 1 {
        flux = moving_average(&flux, params.smooth_win);
    }

    normalize(&mut flux);
    flux
}

fn moving_average(x: &[f32], win: usize) -> Vec<f32> {
    if win <= 1 || x.is_empty() {
        return x.to_vec();
    }
    let n = x.len();
    let mut out = vec![0.0f32; n];
    let half = win / 2;
    for i in 0..n {
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(n);
        let slice = &x[lo..hi];
        out[i] = slice.iter().sum::<f32>() / slice.len() as f32;
    }
    out
}

fn normalize(x: &mut [f32]) {
    let max = x.iter().cloned().fold(0.0f32, f32::max);
    if max > 0.0 {
        for v in x.iter_mut() {
            *v /= max;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectrum::{stft_mono, StftParams};
    use onda_core::{AudioBuffer, SAMPLE_RATE};
    use std::f32::consts::TAU;

    fn synth_kicks(bpm: f32, secs: f32) -> AudioBuffer {
        let n = (secs * SAMPLE_RATE as f32) as usize;
        let beat_period = (60.0 / bpm * SAMPLE_RATE as f32) as usize;
        let mut samples = Vec::with_capacity(n * 2);
        for i in 0..n {
            let beat_phase = i % beat_period;
            let beat_t = beat_phase as f32 / SAMPLE_RATE as f32;
            let env = (-beat_t * 25.0).exp();
            let freq = 60.0 * (-beat_t * 8.0).exp() + 40.0;
            let kick = (beat_t * TAU * freq).sin() * env * 0.7;
            samples.push(kick);
            samples.push(kick);
        }
        AudioBuffer { samples, sample_rate: SAMPLE_RATE, channels: 2 }
    }

    #[test]
    fn onset_envelope_spikes_for_kick_drum_at_expected_rate() {
        let bpm = 120.0;
        let buf = synth_kicks(bpm, 8.0);
        let spec = stft_mono(&buf, StftParams::MIR_STANDARD).unwrap();
        let env = onset_envelope(&spec, OnsetParams::default());

        // Count peaks (values > 0.3 preceded by smaller).
        let mut peaks = 0;
        for i in 1..env.len() - 1 {
            if env[i] > 0.3 && env[i] > env[i - 1] && env[i] >= env[i + 1] {
                peaks += 1;
            }
        }
        // 120 BPM × 8s / 60s = 16 beats. Tolerate transient loss ±3.
        assert!(
            (12..=20).contains(&peaks),
            "expected ~16 peaks, got {peaks}"
        );
    }
}
