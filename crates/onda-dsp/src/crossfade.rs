//! Crossfade between two stereo interleaved buffers.
//!
//! Equal-power (cos/sin) is the default — preserves perceived loudness through
//! the transition. Linear is provided for debugging and A/B.

use onda_core::{AudioBuffer, OndaError, Result, SAMPLE_RATE};

#[derive(Debug, Clone, Copy)]
pub enum CrossfadeCurve {
    Linear,
    EqualPower,
    /// Split at `cutoff_hz` (typical 150); hard-switch lows at midpoint,
    /// equal-power highs. Avoids kick phase-cancellation.
    BassSwap { cutoff_hz: f32 },
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

    if let CrossfadeCurve::BassSwap { cutoff_hz } = curve {
        return crossfade_bass_swap(tail_of_a, head_of_b, SAMPLE_RATE as f32, cutoff_hz);
    }

    let frames = tail_of_a.len() / 2;
    let mut out = Vec::with_capacity(tail_of_a.len());

    for i in 0..frames {
        let t = i as f32 / frames.max(1) as f32;
        let (ga, gb) = match curve {
            CrossfadeCurve::Linear => (1.0 - t, t),
            CrossfadeCurve::EqualPower => {
                let ang = t * std::f32::consts::FRAC_PI_2;
                (ang.cos(), ang.sin())
            }
            CrossfadeCurve::BassSwap { .. } => unreachable!(),
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

/// Bass-swap crossfade: splits A and B into low/high bands at `cutoff_hz`
/// (typical 150Hz), hard-switches the low band at the midpoint (so only ONE
/// kick plays at a time), and equal-power crossfades the high band.
/// Short (50ms) linear ramp at the low-band switch avoids a click.
///
/// Inputs must be equal-length stereo interleaved at `sample_rate`.
pub fn crossfade_bass_swap(
    tail_of_a: &[f32],
    head_of_b: &[f32],
    sample_rate: f32,
    cutoff_hz: f32,
) -> Result<Vec<f32>> {
    use crate::filters::{apply_stereo, Biquad};

    if tail_of_a.len() != head_of_b.len() {
        return Err(OndaError::Dsp(format!(
            "bass-swap length mismatch: a={} b={}",
            tail_of_a.len(),
            head_of_b.len()
        )));
    }
    if tail_of_a.len() % 2 != 0 {
        return Err(OndaError::Dsp("expected stereo buffer".into()));
    }

    let frames = tail_of_a.len() / 2;
    if frames == 0 {
        return Ok(Vec::new());
    }

    // Compute each track's low band. Highs = original - lows (linear-phase
    // complement, so A_low+A_high == A exactly).
    let mut a_low = tail_of_a.to_vec();
    let mut b_low = head_of_b.to_vec();
    {
        let mut la = Biquad::lowpass(sample_rate, cutoff_hz);
        let mut ra = Biquad::lowpass(sample_rate, cutoff_hz);
        apply_stereo(&mut a_low, &mut la, &mut ra);
    }
    {
        let mut lb = Biquad::lowpass(sample_rate, cutoff_hz);
        let mut rb = Biquad::lowpass(sample_rate, cutoff_hz);
        apply_stereo(&mut b_low, &mut lb, &mut rb);
    }

    // Switch at midpoint with a 50ms linear ramp for click-free swap.
    let midpoint = frames / 2;
    let ramp_frames = ((0.050 * sample_rate) as usize).min(midpoint);
    let ramp_start = midpoint.saturating_sub(ramp_frames / 2);
    let ramp_end = (ramp_start + ramp_frames).min(frames);

    let mut out = Vec::with_capacity(tail_of_a.len());
    for i in 0..frames {
        let t = i as f32 / frames.max(1) as f32;
        // High band: equal-power cos/sin.
        let ang = t * std::f32::consts::FRAC_PI_2;
        let ga_hi = ang.cos();
        let gb_hi = ang.sin();

        // Low band: hard switch with small ramp.
        let (ga_lo, gb_lo) = if i < ramp_start {
            (1.0, 0.0)
        } else if i >= ramp_end {
            (0.0, 1.0)
        } else {
            let r = (i - ramp_start) as f32 / ramp_frames.max(1) as f32;
            (1.0 - r, r)
        };

        for ch in 0..2 {
            let idx = i * 2 + ch;
            let al = a_low[idx];
            let bl = b_low[idx];
            let ah = tail_of_a[idx] - al;
            let bh = head_of_b[idx] - bl;
            let mixed = (ga_hi * ah + gb_hi * bh) + (ga_lo * al + gb_lo * bl);
            out.push(mixed);
        }
    }
    Ok(out)
}

/// Beat-aligned crossfade.
///
/// Contract:
/// - `bpm` is the *shared* tempo during the overlap. Caller is responsible
///   for time-stretching B to match A's BPM beforehand (or vice versa).
/// - `a_fade_start_sec` is where A's fade-out begins — this should be an A
///   beat/downbeat chosen by the caller so bars align.
/// - `b_first_beat_sec` is where in B the overlap begins; typically the first
///   detected beat of B (drops silence/off-beat intro).
/// - Overlap length is `bars * 4 * 60 / bpm` seconds (4/4 assumed).
///
/// The function pads with silence if A is shorter than `a_fade_start_sec +
/// overlap`, so the caller can pick a fade-start near the end without exact
/// overlap length.
pub fn beat_aligned_crossfade(
    a: &AudioBuffer,
    b: &AudioBuffer,
    bars: u32,
    bpm: f32,
    a_fade_start_sec: f32,
    b_first_beat_sec: f32,
    curve: CrossfadeCurve,
) -> Result<AudioBuffer> {
    if a.sample_rate != SAMPLE_RATE || b.sample_rate != SAMPLE_RATE {
        return Err(OndaError::Dsp(format!("expected {}Hz", SAMPLE_RATE)));
    }
    if a.channels != 2 || b.channels != 2 {
        return Err(OndaError::Dsp("expected stereo input".into()));
    }
    if bars == 0 || bpm <= 0.0 {
        return Err(OndaError::InvalidArg(format!(
            "bars={bars} bpm={bpm} must be positive"
        )));
    }

    let beats_per_bar = 4u32;
    let total_beats = bars * beats_per_bar;
    let overlap_secs = total_beats as f32 * 60.0 / bpm;
    let overlap_frames = (overlap_secs * SAMPLE_RATE as f32).round() as usize;
    let overlap_samples = overlap_frames * 2;

    // A: pre-fade keep + overlap tail. Anchor overlap at `a_fade_start_sec`.
    let a_fade_start_frame = (a_fade_start_sec.max(0.0) * SAMPLE_RATE as f32).round() as usize;
    let a_pre_end = (a_fade_start_frame * 2).min(a.samples.len());
    let a_pre = &a.samples[..a_pre_end];

    // Take A's tail starting at a_pre_end, up to overlap_samples; pad silence
    // if we run off the end.
    let a_tail_avail = a.samples.len().saturating_sub(a_pre_end);
    let a_tail_take = a_tail_avail.min(overlap_samples);
    let mut a_tail_buf = vec![0.0f32; overlap_samples];
    a_tail_buf[..a_tail_take]
        .copy_from_slice(&a.samples[a_pre_end..a_pre_end + a_tail_take]);

    // B: skip head up to first beat, then take overlap + rest.
    let b_skip_frame = (b_first_beat_sec.max(0.0) * SAMPLE_RATE as f32).round() as usize;
    let b_skip_samples = (b_skip_frame * 2).min(b.samples.len());
    let b_trimmed = &b.samples[b_skip_samples..];
    if b_trimmed.len() < overlap_samples {
        return Err(OndaError::InvalidArg(format!(
            "B too short after trim: need {:.2}s, have {:.2}s",
            overlap_secs,
            b_trimmed.len() as f32 / 2.0 / SAMPLE_RATE as f32
        )));
    }
    let b_head = &b_trimmed[..overlap_samples];
    let b_rest = &b_trimmed[overlap_samples..];

    let crossfaded = crossfade(&a_tail_buf, b_head, curve)?;

    let mut out = Vec::with_capacity(a_pre.len() + crossfaded.len() + b_rest.len());
    out.extend_from_slice(a_pre);
    out.extend_from_slice(&crossfaded);
    out.extend_from_slice(b_rest);

    Ok(AudioBuffer {
        samples: out,
        sample_rate: SAMPLE_RATE,
        channels: 2,
    })
}

/// Pick the latest beat in `beats_sec` such that at least `overlap_secs` of A
/// remain after it. Returns the chosen beat time (seconds). Falls back to
/// `a_duration - overlap_secs` if no beat qualifies.
pub fn pick_a_fade_start(beats_sec: &[f32], a_duration_secs: f32, overlap_secs: f32) -> f32 {
    let deadline = (a_duration_secs - overlap_secs).max(0.0);
    // Prefer a downbeat-ish position (every 4th beat) near the deadline.
    let best = beats_sec
        .iter()
        .copied()
        .filter(|&t| t <= deadline)
        .enumerate()
        // Weight: prefer indices divisible by 4 (downbeats), and closeness to deadline.
        .map(|(i, t)| {
            let downbeat_bonus = if i % 4 == 0 { 0.25 } else { 0.0 };
            let score = t + downbeat_bonus;
            (score, t)
        })
        .fold(None::<(f32, f32)>, |acc, cur| match acc {
            None => Some(cur),
            Some(prev) if cur.0 > prev.0 => Some(cur),
            _ => acc,
        });
    best.map(|(_, t)| t).unwrap_or(deadline)
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
    fn bass_swap_low_end_switches_at_midpoint() {
        // Synthesize A = 60Hz sine (kick-range), B = 60Hz sine negative-phased.
        // After bass-swap: first half should track A's low, second half B's low.
        let sr = SAMPLE_RATE as f32;
        let frames = 44100; // 1s
        let mut a = Vec::with_capacity(frames * 2);
        let mut b = Vec::with_capacity(frames * 2);
        for n in 0..frames {
            let t = n as f32 / sr;
            // A: 60Hz sine. B: same frequency, inverted (would phase-cancel if mixed).
            let sa = (2.0 * std::f32::consts::PI * 60.0 * t).sin();
            let sb = -sa;
            a.push(sa);
            a.push(sa);
            b.push(sb);
            b.push(sb);
        }
        let out = crossfade_bass_swap(&a, &b, sr, 150.0).unwrap();

        // Measure low-band RMS of first and second quarter of output.
        let rms_range = |data: &[f32], start: usize, end: usize| -> f32 {
            let acc: f32 = data[start..end].iter().map(|x| x * x).sum();
            (acc / (end - start) as f32).sqrt()
        };
        let q1 = frames / 2; // 0.25s
        let q3 = frames + frames / 2; // 0.75s mark in stereo samples
        let rms_first = rms_range(&out, 0, q1);
        let rms_last = rms_range(&out, q3, out.len());
        // Both should be ~sine RMS 0.707 — meaning one kick is cleanly present
        // in each half. If bass swap failed, summing A+B≈0 → tiny RMS.
        assert!(
            rms_first > 0.4,
            "first-quarter low RMS too low: {rms_first}"
        );
        assert!(
            rms_last > 0.4,
            "last-quarter low RMS too low: {rms_last}"
        );
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
