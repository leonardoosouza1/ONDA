//! Gain/envelope utilities.

use onda_core::AudioBuffer;

/// Apply a constant linear gain to every sample in-place. `1.0` = unity.
pub fn apply_gain(buf: &mut AudioBuffer, gain: f32) {
    for s in buf.samples.iter_mut() {
        *s *= gain;
    }
}

/// Root-mean-square over all samples. Useful for loudness normalization sanity.
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
    (sum_sq / samples.len() as f64).sqrt() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_of_constant_signal_equals_magnitude() {
        let sig = vec![0.5f32; 1024];
        assert!((rms(&sig) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn rms_of_silence_is_zero() {
        let sig = vec![0.0f32; 1024];
        assert_eq!(rms(&sig), 0.0);
    }
}
