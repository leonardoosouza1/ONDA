//! Beat grid construction from a tempo estimate.
//!
//! Given a tempo, align a uniform beat grid to the onset envelope by sliding
//! the phase and picking the alignment that maximizes sum-of-onsets at beat
//! positions. Simple but sufficient for constant-tempo electronic music.

#[derive(Debug, Clone)]
pub struct BeatGrid {
    /// Beat times in seconds (monotonically increasing).
    pub beats: Vec<f32>,
    /// Period between beats in seconds (= 60/BPM).
    pub period_sec: f32,
    /// Alignment quality 0..1 (fraction of beats that land on local onset peaks).
    pub quality: f32,
}

/// Build a beat grid from tempo + onset envelope. `frame_sec` is the
/// duration of one envelope frame (hop_size / sample_rate).
pub fn beat_grid_from_tempo(env: &[f32], frame_sec: f32, bpm: f32) -> BeatGrid {
    let period_sec = 60.0 / bpm;
    let period_frames = (period_sec / frame_sec).round().max(1.0) as usize;
    let n = env.len();

    // Try each phase offset [0, period_frames), pick the one that maximizes
    // sum of envelope at beat positions.
    let mut best_offset = 0usize;
    let mut best_score = f32::MIN;
    for offset in 0..period_frames.min(n) {
        let mut sum = 0.0f32;
        let mut count = 0u32;
        let mut pos = offset;
        while pos < n {
            sum += env[pos];
            count += 1;
            pos += period_frames;
        }
        let score = if count > 0 { sum / count as f32 } else { 0.0 };
        if score > best_score {
            best_score = score;
            best_offset = offset;
        }
    }

    let mut beats = Vec::new();
    let mut pos = best_offset;
    while pos < n {
        beats.push(pos as f32 * frame_sec);
        pos += period_frames;
    }

    // Quality metric: how many beats land within ±1 frame of a local peak.
    let mut aligned = 0u32;
    for &t in &beats {
        let idx = (t / frame_sec) as usize;
        if idx == 0 || idx + 1 >= n {
            continue;
        }
        if env[idx] >= env[idx - 1] && env[idx] >= env[idx + 1] && env[idx] > 0.1 {
            aligned += 1;
        }
    }
    let quality = if beats.is_empty() {
        0.0
    } else {
        aligned as f32 / beats.len() as f32
    };

    BeatGrid { beats, period_sec, quality }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beat_grid_spacing_matches_bpm() {
        // Synthesize an onset envelope: spikes at every 0.5s (= 120 BPM),
        // spec frame hop = 10ms.
        let frame_sec = 0.01;
        let mut env = vec![0.0f32; 1000]; // 10s
        for beat_idx in 0..20 {
            let t = beat_idx as f32 * 0.5;
            let frame = (t / frame_sec) as usize;
            if frame < env.len() {
                env[frame] = 1.0;
            }
        }
        let grid = beat_grid_from_tempo(&env, frame_sec, 120.0);
        assert!((grid.period_sec - 0.5).abs() < 1e-4);
        assert!(grid.beats.len() >= 18);
        assert!(grid.quality > 0.8, "quality was {}", grid.quality);
    }
}
