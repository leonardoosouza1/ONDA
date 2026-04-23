//! Tempo estimation via autocorrelation of the onset envelope.
//!
//! Classical MIR method (Ellis 2007 style, simplified): autocorrelate the
//! onset envelope, then pick the largest peak within a plausible BPM range.

use crate::onsets::{onset_envelope, OnsetParams};
use crate::spectrum::Spectrogram;

#[derive(Debug, Clone, Copy)]
pub struct TempoEstimate {
    pub bpm: f32,
    /// Peak autocorrelation value at the chosen lag — quality proxy.
    pub confidence: f32,
}

/// Estimate tempo in BPM from a spectrogram. Searches plausible range
/// [60, 200] BPM.
pub fn estimate_tempo(spec: &Spectrogram) -> TempoEstimate {
    let env = onset_envelope(spec, OnsetParams::default());
    estimate_tempo_from_env(&env, spec.frame_duration())
}

pub fn estimate_tempo_from_env(env: &[f32], frame_sec: f32) -> TempoEstimate {
    estimate_tempo_with_prior(env, frame_sec, 125.0, 0.9)
}

/// Estimate tempo with a log-Gaussian prior centered at `prior_bpm`.
///
/// The prior suppresses octave errors (locking on half/double tempo) and
/// reflects the genre we're targeting. For electronic music, center near
/// 125 BPM; for drum'n'bass raise to 170. `prior_width_octaves` controls
/// tolerance (0.9 is broad enough to allow 60-200 BPM at nontrivial weight).
pub fn estimate_tempo_with_prior(
    env: &[f32],
    frame_sec: f32,
    prior_bpm: f32,
    prior_width_octaves: f32,
) -> TempoEstimate {
    let n = env.len();
    if n < 32 {
        return TempoEstimate { bpm: 0.0, confidence: 0.0 };
    }

    let mean: f32 = env.iter().sum::<f32>() / n as f32;
    let centered: Vec<f32> = env.iter().map(|v| v - mean).collect();

    let min_lag = ((60.0 / 200.0) / frame_sec).round() as usize;
    let max_lag = ((60.0 / 60.0) / frame_sec).round() as usize;
    let max_lag = max_lag.min(n / 2);

    let mut best_lag = min_lag;
    let mut best_val = f32::MIN;
    for lag in min_lag..=max_lag {
        let mut acc = 0.0f32;
        let mut count = 0u32;
        for i in 0..(n - lag) {
            acc += centered[i] * centered[i + lag];
            count += 1;
        }
        let raw = if count > 0 { acc / count as f32 } else { 0.0 };

        // Log-Gaussian prior in octave space.
        let bpm_at_lag = 60.0 / (lag as f32 * frame_sec);
        let octave_delta = (bpm_at_lag / prior_bpm).log2();
        let prior = (-(octave_delta * octave_delta) / (2.0 * prior_width_octaves.powi(2))).exp();

        let val = raw * prior;
        if val > best_val {
            best_val = val;
            best_lag = lag;
        }
    }

    let bpm = 60.0 / (best_lag as f32 * frame_sec);

    // Normalize confidence by zero-lag autocorrelation (variance).
    let variance: f32 = centered.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let confidence = if variance > 0.0 {
        (best_val / variance).clamp(0.0, 1.0)
    } else {
        0.0
    };

    TempoEstimate { bpm, confidence }
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
            let bp = i % beat_period;
            let bt = bp as f32 / SAMPLE_RATE as f32;
            let env = (-bt * 25.0).exp();
            let f = 60.0 * (-bt * 8.0).exp() + 40.0;
            let kick = (bt * TAU * f).sin() * env * 0.7;
            samples.push(kick);
            samples.push(kick);
        }
        AudioBuffer { samples, sample_rate: SAMPLE_RATE, channels: 2 }
    }

    #[test]
    fn estimate_tempo_recovers_120_bpm_from_synthetic_kick() {
        let buf = synth_kicks(120.0, 16.0);
        let spec = stft_mono(&buf, StftParams::MIR_STANDARD).unwrap();
        let t = estimate_tempo(&spec);
        // Classical methods often lock onto half/double — accept either.
        let ok = (t.bpm - 120.0).abs() < 3.0
            || (t.bpm - 60.0).abs() < 3.0
            || (t.bpm - 240.0).abs() < 3.0;
        assert!(ok, "got BPM {}, expected 120 (or half/double)", t.bpm);
        assert!(t.confidence > 0.1, "low confidence {}", t.confidence);
    }

    #[test]
    fn estimate_tempo_recovers_128_bpm_from_synthetic_kick() {
        let buf = synth_kicks(128.0, 16.0);
        let spec = stft_mono(&buf, StftParams::MIR_STANDARD).unwrap();
        let t = estimate_tempo(&spec);
        let ok = (t.bpm - 128.0).abs() < 3.0
            || (t.bpm - 64.0).abs() < 3.0
            || (t.bpm - 256.0).abs() < 3.0;
        assert!(ok, "got BPM {}, expected 128 (or half/double)", t.bpm);
    }
}
