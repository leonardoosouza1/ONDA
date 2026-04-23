//! Structural/section detection for picking musical cut points.
//!
//! Heuristic: a DJ fades out at an *outro* (energy drop) or at a breakdown
//! (bass drops, drums thin out). Full structural segmentation via SSM/novelty
//! is expensive; we use a cheap but effective proxy:
//!
//! 1. Compute short-window RMS envelope of the waveform.
//! 2. Smooth it (moving-average ~2 bars).
//! 3. Find local minima: points where RMS drops >= `drop_pct` below the
//!    rolling max of the preceding `memory_bars`.
//! 4. Score each candidate by: how deep the drop is, how close to a
//!    downbeat it is, and how late in the track it sits (bias toward outro).
//!
//! The picker returns a fade-start time that is simultaneously a beat
//! (from the grid) AND near a detected section boundary.

use onda_core::AudioBuffer;

/// RMS envelope of the full stereo buffer, windowed at `window_secs`.
/// Returns one value per window; timestamp of window `i` is `i*window_secs`.
pub fn rms_envelope(buf: &AudioBuffer, window_secs: f32) -> Vec<f32> {
    let win = ((window_secs * buf.sample_rate as f32) as usize).max(1);
    let ch = buf.channels.max(1) as usize;
    let frames = buf.samples.len() / ch;
    if frames == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(frames / win + 1);
    let mut i = 0;
    while i < frames {
        let end = (i + win).min(frames);
        let mut acc = 0.0f64;
        let mut n = 0usize;
        for f in i..end {
            for c in 0..ch {
                let v = buf.samples[f * ch + c] as f64;
                acc += v * v;
                n += 1;
            }
        }
        let rms = if n > 0 { (acc / n as f64).sqrt() as f32 } else { 0.0 };
        out.push(rms);
        i += win;
    }
    out
}

/// Smooth an envelope with a centered moving average of `radius` bins.
pub fn smooth(env: &[f32], radius: usize) -> Vec<f32> {
    if env.is_empty() || radius == 0 {
        return env.to_vec();
    }
    let mut out = Vec::with_capacity(env.len());
    for i in 0..env.len() {
        let lo = i.saturating_sub(radius);
        let hi = (i + radius + 1).min(env.len());
        let sum: f32 = env[lo..hi].iter().sum();
        out.push(sum / (hi - lo) as f32);
    }
    out
}

/// A candidate cut point.
#[derive(Debug, Clone, Copy)]
pub struct CutCandidate {
    /// Seconds into the track.
    pub time_sec: f32,
    /// 0..1 — how much energy dropped vs. the preceding peak.
    pub drop_ratio: f32,
    /// 0..1 — confidence this is a section boundary.
    pub confidence: f32,
}

/// Find cut candidates — timestamps where the smoothed RMS drops sharply
/// below the rolling max of the preceding `memory_secs`.
///
/// `min_drop_pct` = 25 means RMS fell to ≤75% of the preceding peak.
pub fn find_cut_candidates(
    env: &[f32],
    window_secs: f32,
    memory_secs: f32,
    min_drop_pct: f32,
) -> Vec<CutCandidate> {
    if env.len() < 4 {
        return Vec::new();
    }
    let memory_bins = (memory_secs / window_secs).max(1.0) as usize;
    let min_drop = min_drop_pct / 100.0;
    let mut candidates = Vec::new();
    for i in memory_bins..env.len() {
        // Rolling max over preceding `memory_bins` windows.
        let peak = env[(i - memory_bins)..i]
            .iter()
            .copied()
            .fold(0.0f32, f32::max);
        if peak <= 1e-6 {
            continue;
        }
        let here = env[i];
        let ratio = 1.0 - (here / peak);
        if ratio < min_drop {
            continue;
        }
        let time_sec = i as f32 * window_secs;
        candidates.push(CutCandidate {
            time_sec,
            drop_ratio: ratio,
            confidence: ratio.clamp(0.0, 1.0),
        });
    }
    // De-duplicate candidates within `memory_secs/2` — keep the deepest drop.
    let group_window = memory_secs * 0.5;
    let mut dedup: Vec<CutCandidate> = Vec::new();
    for c in candidates {
        match dedup.last_mut() {
            Some(last) if (c.time_sec - last.time_sec).abs() < group_window => {
                if c.drop_ratio > last.drop_ratio {
                    *last = c;
                }
            }
            _ => dedup.push(c),
        }
    }
    dedup
}

/// Pick the best musical fade-start time for A.
///
/// Scoring: among all beats that leave room for `overlap_secs` at the end,
/// prefer those within `tolerance_secs` of a cut candidate, weighted by the
/// candidate's drop ratio and by how late it is in the track (bias outro).
///
/// Falls back to the "latest beat" heuristic if no candidates match.
pub fn pick_musical_fade_start(
    beats_sec: &[f32],
    cuts: &[CutCandidate],
    track_duration: f32,
    overlap_secs: f32,
    tolerance_secs: f32,
) -> f32 {
    let deadline = (track_duration - overlap_secs).max(0.0);
    if cuts.is_empty() {
        // Fallback: latest downbeat-ish beat ≤ deadline.
        return beats_sec
            .iter()
            .copied()
            .enumerate()
            .filter(|&(_, t)| t <= deadline)
            .map(|(i, t)| {
                let db_bonus = if i % 4 == 0 { 0.25 } else { 0.0 };
                (t + db_bonus, t)
            })
            .fold(None::<(f32, f32)>, |a, c| match a {
                None => Some(c),
                Some(p) if c.0 > p.0 => Some(c),
                _ => a,
            })
            .map(|(_, t)| t)
            .unwrap_or(deadline);
    }

    let mut best: Option<(f32, f32)> = None;
    for (i, &t) in beats_sec.iter().enumerate() {
        if t > deadline {
            continue;
        }
        // Find nearest cut within tolerance.
        let near = cuts
            .iter()
            .map(|c| ((c.time_sec - t).abs(), c))
            .filter(|(d, _)| *d <= tolerance_secs)
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let base_score = if let Some((dist, c)) = near {
            let proximity = 1.0 - (dist / tolerance_secs);
            // weight: proximity to cut × drop depth
            2.0 * proximity * c.drop_ratio
        } else {
            0.0
        };

        let late_bonus = t / track_duration.max(1.0) * 0.5;
        let downbeat_bonus = if i % 4 == 0 { 0.15 } else { 0.0 };
        let score = base_score + late_bonus + downbeat_bonus;

        if best.map(|(s, _)| score > s).unwrap_or(true) {
            best = Some((score, t));
        }
    }

    best.map(|(_, t)| t).unwrap_or(deadline)
}

#[cfg(test)]
mod tests {
    use super::*;
    use onda_core::AudioBuffer;

    #[test]
    fn rms_envelope_detects_energy_drop() {
        let sr = 44100u32;
        // 4s of loud, 4s of quiet (stereo).
        let mut samples = Vec::with_capacity(sr as usize * 8 * 2);
        for _ in 0..(sr as usize * 4) {
            samples.push(0.8);
            samples.push(0.8);
        }
        for _ in 0..(sr as usize * 4) {
            samples.push(0.05);
            samples.push(0.05);
        }
        let buf = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: 2,
        };
        let env = rms_envelope(&buf, 0.5);
        // First 8 windows (4s) should be ~0.8, last 8 ~0.05.
        let first_mean: f32 = env[..8].iter().sum::<f32>() / 8.0;
        let last_mean: f32 = env[env.len() - 8..].iter().sum::<f32>() / 8.0;
        assert!(first_mean > 0.7, "first={first_mean}");
        assert!(last_mean < 0.1, "last={last_mean}");

        // Cut detector should find a candidate in the transition.
        let smoothed = smooth(&env, 2);
        let cuts = find_cut_candidates(&smoothed, 0.5, 3.0, 30.0);
        assert!(!cuts.is_empty(), "expected at least one cut candidate");
        // Drop should be near 4s mark.
        let first_cut = cuts[0].time_sec;
        assert!(first_cut >= 3.0 && first_cut <= 6.0, "got {first_cut}");
    }
}
