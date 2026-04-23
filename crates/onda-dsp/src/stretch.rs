//! Time-stretch operators.
//!
//! Current approach: **turntable mode** — pitch + tempo move together via
//! high-quality sinc resampling. This is exactly what a vinyl DJ does with
//! the pitch fader, and what CDJs/Serato/Traktor do when "keylock" is OFF.
//!
//! For small BPM differences (<8%, ~1.4 semitones) it sounds completely
//! natural on electronic music. For larger stretches we'd need a phase vocoder
//! or WSOLA (pitch-independent), which we can layer on later.

use onda_core::{AudioBuffer, OndaError, Result, SAMPLE_RATE};

/// De-interleave a stereo f32 buffer into (L, R).
fn deinterleave_stereo(samples: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let frames = samples.len() / 2;
    let mut l = Vec::with_capacity(frames);
    let mut r = Vec::with_capacity(frames);
    for i in 0..frames {
        l.push(samples[i * 2]);
        r.push(samples[i * 2 + 1]);
    }
    (l, r)
}

/// Re-interleave L and R back into stereo, truncating to the shorter channel.
fn interleave_stereo(l: &[f32], r: &[f32]) -> Vec<f32> {
    let n = l.len().min(r.len());
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        out.push(l[i]);
        out.push(r[i]);
    }
    out
}

/// Resample a stereo interleaved buffer, splitting channels so the resampler
/// treats them independently (identical ratio, so phase coherence is kept).
/// Uses `onda-audio`'s `resample_to` under the hood via a caller-provided fn
/// to avoid a dep cycle.
fn resample_stereo_with<F>(samples: &[f32], resample_mono: F) -> Vec<f32>
where
    F: Fn(&[f32]) -> Vec<f32>,
{
    let (l, r) = deinterleave_stereo(samples);
    let lo = resample_mono(&l);
    let ro = resample_mono(&r);
    interleave_stereo(&lo, &ro)
}

/// Stretch a track so its playback duration is multiplied by `stretch_ratio`.
/// `stretch_ratio > 1.0` slows down (lower pitch), `< 1.0` speeds up (higher pitch).
///
/// Implemented as: resample from `SAMPLE_RATE` to `SAMPLE_RATE * stretch_ratio`,
/// then label the result as `SAMPLE_RATE`. When played at the nominal rate,
/// it now takes `stretch_ratio` times longer to play.
///
/// Caller provides the resample function (must be `fn(signal, src, dst) -> Vec<f32>`).
pub fn turntable_stretch<R>(
    buf: &AudioBuffer,
    stretch_ratio: f32,
    resample_mono: R,
) -> Result<AudioBuffer>
where
    R: Fn(&[f32], u32, u32) -> Result<Vec<f32>>,
{
    if buf.channels != 2 {
        return Err(OndaError::Dsp("turntable_stretch expects stereo".into()));
    }
    if !(0.5..=2.0).contains(&stretch_ratio) {
        return Err(OndaError::InvalidArg(format!(
            "stretch_ratio {stretch_ratio} out of safe range [0.5, 2.0]"
        )));
    }
    if (stretch_ratio - 1.0).abs() < 1e-4 {
        return Ok(buf.clone());
    }

    let src_rate = SAMPLE_RATE;
    let dst_rate = (SAMPLE_RATE as f32 * stretch_ratio).round() as u32;

    let resample_at = |mono: &[f32]| -> Vec<f32> {
        resample_mono(mono, src_rate, dst_rate).unwrap_or_else(|_| mono.to_vec())
    };

    let out_samples = resample_stereo_with(&buf.samples, resample_at);

    Ok(AudioBuffer {
        samples: out_samples,
        sample_rate: SAMPLE_RATE,
        channels: 2,
    })
}

/// Convenience: stretch so `src_bpm` becomes `target_bpm`. If source is faster
/// than target, slows down (ratio > 1); if slower, speeds up (ratio < 1).
pub fn bpm_match_ratio(src_bpm: f32, target_bpm: f32) -> f32 {
    if src_bpm <= 0.0 || target_bpm <= 0.0 {
        return 1.0;
    }
    src_bpm / target_bpm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratio_matches_bpm_direction() {
        // Source faster than target → ratio > 1 (stretch longer → slower).
        assert!(bpm_match_ratio(150.0, 120.0) > 1.0);
        // Source slower than target → ratio < 1 (compress → faster).
        assert!(bpm_match_ratio(100.0, 120.0) < 1.0);
        // Equal → 1.0
        assert!((bpm_match_ratio(120.0, 120.0) - 1.0).abs() < 1e-6);
    }
}
