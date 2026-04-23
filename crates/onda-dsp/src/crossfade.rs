//! Crossfade between two stereo interleaved buffers.
//!
//! Equal-power (cos/sin) is the default — preserves perceived loudness through
//! the transition. Linear is provided for debugging and A/B.

use onda_core::{AudioBuffer, OndaError, Result, SAMPLE_RATE};

#[derive(Debug, Clone, Copy)]
pub enum CrossfadeCurve {
    Linear,
    EqualPower,
}

/// Crossfade `tail_of_a` (end of track A) with `head_of_b` (start of track B)
/// over the entire length of both buffers. Requires equal length, same rate,
/// stereo. Returns a new stereo interleaved buffer.
pub fn crossfade(
    tail_of_a: &[f32],
    head_of_b: &[f32],
    curve: CrossfadeCurve,
) -> Result<Vec<f32>> {
    if tail_of_a.len() != head_of_b.len() {
        return Err(OndaError::Dsp(format!(
            "crossfade length mismatch: a={} b={}",
            tail_of_a.len(),
            head_of_b.len()
        )));
    }
    if tail_of_a.len() % 2 != 0 {
        return Err(OndaError::Dsp("expected stereo (even-length) buffer".into()));
    }

    let frames = tail_of_a.len() / 2;
    let mut out = Vec::with_capacity(tail_of_a.len());

    for i in 0..frames {
        let t = i as f32 / frames.max(1) as f32;
        let (ga, gb) = match curve {
            CrossfadeCurve::Linear => (1.0 - t, t),
            CrossfadeCurve::EqualPower => {
                // sin/cos law: ga² + gb² = 1 through the whole transition.
                let ang = t * std::f32::consts::FRAC_PI_2;
                (ang.cos(), ang.sin())
            }
        };
        let l = tail_of_a[i * 2] * ga + head_of_b[i * 2] * gb;
        let r = tail_of_a[i * 2 + 1] * ga + head_of_b[i * 2 + 1] * gb;
        out.push(l);
        out.push(r);
    }

    Ok(out)
}

/// Compose a full A→B mix: keep the pre-fade of A, crossfade the overlap,
/// then keep the post-fade of B. `fade_secs` sets the overlap length.
///
/// This is the simplest possible "DJ transition" — no beat match, no key
/// compatibility, no stems. Useful as a pipeline smoke test.
pub fn build_crossfade_mix(
    a: &AudioBuffer,
    b: &AudioBuffer,
    fade_secs: f32,
    curve: CrossfadeCurve,
) -> Result<AudioBuffer> {
    if a.sample_rate != SAMPLE_RATE || b.sample_rate != SAMPLE_RATE {
        return Err(OndaError::Dsp(format!(
            "expected {}Hz, got a={} b={}",
            SAMPLE_RATE, a.sample_rate, b.sample_rate
        )));
    }
    if a.channels != 2 || b.channels != 2 {
        return Err(OndaError::Dsp("expected stereo input".into()));
    }
    if fade_secs <= 0.0 {
        return Err(OndaError::InvalidArg("fade_secs must be > 0".into()));
    }

    let fade_frames = (fade_secs * SAMPLE_RATE as f32) as usize;
    let fade_samples = fade_frames * 2;

    if fade_samples > a.samples.len() || fade_samples > b.samples.len() {
        return Err(OndaError::InvalidArg(format!(
            "fade_secs={} too long; a={}s b={}s",
            fade_secs,
            a.duration_secs(),
            b.duration_secs()
        )));
    }

    let a_head_end = a.samples.len() - fade_samples;
    let a_tail = &a.samples[a_head_end..];
    let b_head = &b.samples[..fade_samples];
    let b_rest = &b.samples[fade_samples..];

    let crossfaded = crossfade(a_tail, b_head, curve)?;

    let total = a_head_end + crossfaded.len() + b_rest.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&a.samples[..a_head_end]);
    out.extend_from_slice(&crossfaded);
    out.extend_from_slice(b_rest);

    Ok(AudioBuffer {
        samples: out,
        sample_rate: SAMPLE_RATE,
        channels: 2,
    })
}

/// Convenience wrappers matching the shape exposed from lib.rs.
pub fn crossfade_linear(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    crossfade(a, b, CrossfadeCurve::Linear)
}
pub fn crossfade_equal_power(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    crossfade(a, b, CrossfadeCurve::EqualPower)
}

/// Beat-aligned crossfade: the overlap length is an integer number of bars
/// of track A (4 beats per bar assumed — 4/4 time), and the fade begins at
/// the latest beat of A that leaves room for `bars` complete bars.
///
/// Track B is aligned so that its first beat of the overlap lands on the
/// downbeat of A's fade-in bar. No tempo matching yet (that requires a
/// phase vocoder) — for two tracks with similar tempos (±3 BPM), this is
/// already audibly coherent; beyond that, a tempo mismatch is audible as
/// phasing. Returns the combined mix.
pub fn beat_aligned_crossfade(
    a: &AudioBuffer,
    b: &AudioBuffer,
    bars: u32,
    a_bpm: f32,
    b_first_beat_sec: f32,
    curve: CrossfadeCurve,
) -> Result<AudioBuffer> {
    if a.sample_rate != SAMPLE_RATE || b.sample_rate != SAMPLE_RATE {
        return Err(OndaError::Dsp(format!(
            "expected {}Hz",
            SAMPLE_RATE
        )));
    }
    if a.channels != 2 || b.channels != 2 {
        return Err(OndaError::Dsp("expected stereo input".into()));
    }
    if bars == 0 || a_bpm <= 0.0 {
        return Err(OndaError::InvalidArg(format!(
            "bars={bars} a_bpm={a_bpm} must be positive"
        )));
    }

    let beats_per_bar = 4u32;
    let total_beats = bars * beats_per_bar;
    let fade_secs = total_beats as f32 * 60.0 / a_bpm;
    let fade_frames = (fade_secs * SAMPLE_RATE as f32).round() as usize;
    let fade_samples = fade_frames * 2;

    if fade_samples > a.samples.len() {
        return Err(OndaError::InvalidArg(format!(
            "need {:.2}s fade but A is only {:.2}s",
            fade_secs,
            a.duration_secs()
        )));
    }

    // Trim track B so its first detected beat is the start of the overlap,
    // dropping anything before it (cut the intro silence / off-beat head).
    let b_skip_samples = ((b_first_beat_sec.max(0.0) * SAMPLE_RATE as f32) as usize) * 2;
    let b_skip_samples = b_skip_samples.min(b.samples.len());
    let b_trimmed = &b.samples[b_skip_samples..];
    if fade_samples > b_trimmed.len() {
        return Err(OndaError::InvalidArg(format!(
            "B too short after trimming head: need {}s",
            fade_secs
        )));
    }

    let a_head_end = a.samples.len() - fade_samples;
    let a_tail = &a.samples[a_head_end..];
    let b_head = &b_trimmed[..fade_samples];
    let b_rest = &b_trimmed[fade_samples..];

    let crossfaded = crossfade(a_tail, b_head, curve)?;

    let mut out = Vec::with_capacity(a_head_end + crossfaded.len() + b_rest.len());
    out.extend_from_slice(&a.samples[..a_head_end]);
    out.extend_from_slice(&crossfaded);
    out.extend_from_slice(b_rest);

    Ok(AudioBuffer {
        samples: out,
        sample_rate: SAMPLE_RATE,
        channels: 2,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_midpoint_is_average() {
        // 10 frames stereo = 20 samples.
        let a = vec![1.0; 20];
        let b = vec![-1.0; 20];
        let out = crossfade(&a, &b, CrossfadeCurve::Linear).unwrap();
        assert_eq!(out.len(), 20);
        // At t=0.5 (frame 5), ga=0.5, gb=0.5 → 0.5*1 + 0.5*(-1) = 0.
        assert!((out[10] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn equal_power_preserves_energy_of_constant_signals() {
        let a = vec![1.0; 200];
        let b = vec![1.0; 200];
        let out = crossfade(&a, &b, CrossfadeCurve::EqualPower).unwrap();
        // Both constant 1.0, equal-power sum: sin(θ)+cos(θ) which is >= 1, so
        // *perceived* power stays ~equal. Spot check one sample.
        let midpoint = out[out.len() / 2];
        let expected = (std::f32::consts::FRAC_PI_4).cos() + (std::f32::consts::FRAC_PI_4).sin();
        assert!((midpoint - expected).abs() < 1e-3);
    }

    #[test]
    fn crossfade_length_mismatch_errors() {
        let a = vec![0.0; 20];
        let b = vec![0.0; 22];
        assert!(crossfade(&a, &b, CrossfadeCurve::Linear).is_err());
    }

    #[test]
    fn build_mix_total_length_is_a_plus_b_minus_overlap() {
        let sr = SAMPLE_RATE;
        let a = AudioBuffer {
            samples: vec![0.5; sr as usize * 2 * 4], // 4s stereo
            sample_rate: sr,
            channels: 2,
        };
        let b = AudioBuffer {
            samples: vec![-0.5; sr as usize * 2 * 4], // 4s stereo
            sample_rate: sr,
            channels: 2,
        };
        let mix = build_crossfade_mix(&a, &b, 2.0, CrossfadeCurve::EqualPower).unwrap();
        // 4 + 4 - 2 = 6 seconds
        assert!((mix.duration_secs() - 6.0).abs() < 1e-3);
    }
}
