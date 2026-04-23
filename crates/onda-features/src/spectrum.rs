//! Short-Time Fourier Transform (STFT).
//!
//! Frames the input with a Hann window and FFTs each frame. Output is
//! magnitude-only (phase dropped for the MIR features we care about).

use onda_core::{AudioBuffer, OndaError, Result};
use realfft::RealFftPlanner;

#[derive(Debug, Clone, Copy)]
pub struct StftParams {
    pub win_size: usize, // samples per frame
    pub hop_size: usize, // samples between frame starts
}

impl StftParams {
    /// 2048 / 512 ≈ ~46ms window, ~11ms hop @ 44.1kHz — standard for MIR.
    pub const MIR_STANDARD: Self = Self {
        win_size: 2048,
        hop_size: 512,
    };
}

/// Magnitude spectrogram indexed as `[frame][bin]`, shape = (n_frames, win_size/2 + 1).
pub struct Spectrogram {
    pub magnitudes: Vec<Vec<f32>>,
    pub sample_rate: u32,
    pub hop_size: usize,
    pub win_size: usize,
}

impl Spectrogram {
    pub fn n_frames(&self) -> usize {
        self.magnitudes.len()
    }
    pub fn n_bins(&self) -> usize {
        self.magnitudes.first().map(|f| f.len()).unwrap_or(0)
    }
    /// Frame center time in seconds.
    pub fn frame_time(&self, frame_idx: usize) -> f32 {
        (frame_idx * self.hop_size) as f32 / self.sample_rate as f32
    }
    /// Seconds-per-frame (hop duration).
    pub fn frame_duration(&self) -> f32 {
        self.hop_size as f32 / self.sample_rate as f32
    }
}

/// Compute STFT magnitude of the mono downmix of a stereo buffer.
pub fn stft_mono(buf: &AudioBuffer, params: StftParams) -> Result<Spectrogram> {
    if buf.channels != 2 {
        return Err(OndaError::Dsp("stft_mono expects stereo input".into()));
    }
    if params.win_size == 0 || params.hop_size == 0 || params.hop_size > params.win_size {
        return Err(OndaError::InvalidArg(format!(
            "invalid stft params: {params:?}"
        )));
    }

    // Mono downmix.
    let n_frames_in = buf.frames();
    let mut mono = Vec::with_capacity(n_frames_in);
    for i in 0..n_frames_in {
        mono.push((buf.samples[i * 2] + buf.samples[i * 2 + 1]) * 0.5);
    }

    // Hann window.
    let hann: Vec<f32> = (0..params.win_size)
        .map(|i| {
            let x = i as f32 / (params.win_size - 1) as f32;
            0.5 - 0.5 * (2.0 * std::f32::consts::PI * x).cos()
        })
        .collect();

    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(params.win_size);
    let mut input = fft.make_input_vec();
    let mut output = fft.make_output_vec();

    let n_frames = if mono.len() < params.win_size {
        0
    } else {
        1 + (mono.len() - params.win_size) / params.hop_size
    };
    let n_bins = params.win_size / 2 + 1;
    let mut magnitudes = Vec::with_capacity(n_frames);

    for frame_idx in 0..n_frames {
        let start = frame_idx * params.hop_size;
        for i in 0..params.win_size {
            input[i] = mono[start + i] * hann[i];
        }
        fft.process(&mut input, &mut output)
            .map_err(|e| OndaError::Dsp(format!("stft fft: {e}")))?;
        let mag: Vec<f32> = output.iter().map(|c| (c.re * c.re + c.im * c.im).sqrt()).collect();
        debug_assert_eq!(mag.len(), n_bins);
        magnitudes.push(mag);
    }

    Ok(Spectrogram {
        magnitudes,
        sample_rate: buf.sample_rate,
        hop_size: params.hop_size,
        win_size: params.win_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use onda_core::{AudioBuffer, SAMPLE_RATE};
    use std::f32::consts::TAU;

    fn make_sine(freq: f32, secs: f32) -> AudioBuffer {
        let n = (secs * SAMPLE_RATE as f32) as usize;
        let mut samples = Vec::with_capacity(n * 2);
        for i in 0..n {
            let s = (i as f32 / SAMPLE_RATE as f32 * TAU * freq).sin() * 0.5;
            samples.push(s);
            samples.push(s);
        }
        AudioBuffer { samples, sample_rate: SAMPLE_RATE, channels: 2 }
    }

    #[test]
    fn stft_produces_expected_frame_count() {
        let buf = make_sine(440.0, 2.0); // 2s sine
        let spec = stft_mono(&buf, StftParams::MIR_STANDARD).unwrap();
        let expected_frames = 1 + (buf.frames() - 2048) / 512;
        assert_eq!(spec.n_frames(), expected_frames);
        assert_eq!(spec.n_bins(), 1025); // 2048/2 + 1
    }

    #[test]
    fn stft_peak_bin_matches_sine_frequency() {
        let freq = 1000.0;
        let buf = make_sine(freq, 1.0);
        let spec = stft_mono(&buf, StftParams::MIR_STANDARD).unwrap();
        // bin resolution = sample_rate / win_size = 44100/2048 ≈ 21.5 Hz
        // peak bin should be at freq / resolution ≈ 46
        let mid_frame = &spec.magnitudes[spec.n_frames() / 2];
        let peak_bin = mid_frame
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let bin_hz = peak_bin as f32 * SAMPLE_RATE as f32 / 2048.0;
        assert!(
            (bin_hz - freq).abs() < 25.0,
            "peak at {bin_hz} Hz, expected ~{freq}"
        );
    }
}
