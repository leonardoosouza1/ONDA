//! ONDA core types shared across the workspace.

use serde::{Deserialize, Serialize};

/// Canonical audio sample type. Stereo interleaved f32 at 44.1kHz is the
/// internal working format for everything that isn't explicit about a
/// different layout.
pub type Sample = f32;

pub const SAMPLE_RATE: u32 = 44_100;
pub const CHANNELS: u16 = 2;

/// Stereo audio buffer, interleaved `[L0, R0, L1, R1, ...]`.
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    pub samples: Vec<Sample>,
    pub sample_rate: u32,
    pub channels: u16,
}

impl AudioBuffer {
    pub fn frames(&self) -> usize {
        self.samples.len() / self.channels as usize
    }

    pub fn duration_secs(&self) -> f64 {
        self.frames() as f64 / self.sample_rate as f64
    }

    pub fn empty() -> Self {
        Self {
            samples: Vec::new(),
            sample_rate: SAMPLE_RATE,
            channels: CHANNELS,
        }
    }
}

/// 8-dimensional continuous vibe state (see spec §5).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct VibeState {
    pub energy: f32,
    pub darkness: f32,
    pub density: f32,
    pub tension: f32,
    pub groove_lock: f32,
    pub bass_pressure: f32,
    pub brightness: f32,
    pub novelty: f32,
}

impl VibeState {
    pub const ZERO: Self = Self {
        energy: 0.0,
        darkness: 0.0,
        density: 0.0,
        tension: 0.0,
        groove_lock: 0.0,
        bass_pressure: 0.0,
        brightness: 0.0,
        novelty: 0.0,
    };

    pub fn as_array(&self) -> [f32; 8] {
        [
            self.energy,
            self.darkness,
            self.density,
            self.tension,
            self.groove_lock,
            self.bass_pressure,
            self.brightness,
            self.novelty,
        ]
    }
}

#[derive(Debug, thiserror::Error)]
pub enum OndaError {
    #[error("audio I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("audio decode error: {0}")]
    Decode(String),

    #[error("audio encode error: {0}")]
    Encode(String),

    #[error("DSP error: {0}")]
    Dsp(String),

    #[error("invalid argument: {0}")]
    InvalidArg(String),
}

pub type Result<T> = std::result::Result<T, OndaError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_buffer_frames_counts_correctly_for_stereo() {
        let buf = AudioBuffer {
            samples: vec![0.0; 200],
            sample_rate: 44_100,
            channels: 2,
        };
        assert_eq!(buf.frames(), 100);
    }

    #[test]
    fn audio_buffer_duration_is_frames_over_rate() {
        let buf = AudioBuffer {
            samples: vec![0.0; 44_100 * 2], // 1 second of stereo
            sample_rate: 44_100,
            channels: 2,
        };
        assert!((buf.duration_secs() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn vibe_state_zero_is_all_zeros() {
        assert_eq!(VibeState::ZERO.as_array(), [0.0; 8]);
    }
}
