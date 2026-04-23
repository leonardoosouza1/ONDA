//! IIR biquad filters (Butterworth 2nd-order).
//!
//! Used for band-splitting during bass-swap crossfades — we need to isolate
//! the low end (<~150Hz, i.e. kick fundamental + sub) so only ONE track's
//! kick plays at a time, avoiding phase cancellation between two bumping
//! tracks.

#[derive(Clone, Debug)]
pub struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    pub fn lowpass(sr: f32, fc: f32) -> Self {
        Self::make(sr, fc, false)
    }
    pub fn highpass(sr: f32, fc: f32) -> Self {
        Self::make(sr, fc, true)
    }
    fn make(sr: f32, fc: f32, hp: bool) -> Self {
        let w0 = 2.0 * std::f32::consts::PI * fc / sr;
        let cs = w0.cos();
        let sn = w0.sin();
        let q = std::f32::consts::FRAC_1_SQRT_2;
        let alpha = sn / (2.0 * q);
        let (b0, b1, b2) = if hp {
            let c = 1.0 + cs;
            (c * 0.5, -c, c * 0.5)
        } else {
            let c = 1.0 - cs;
            (c * 0.5, c, c * 0.5)
        };
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cs;
        let a2 = 1.0 - alpha;
        Biquad {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }
    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        // Transposed Direct Form II.
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
}

/// Filter a stereo interleaved buffer in place with per-channel state.
pub fn apply_stereo(samples: &mut [f32], left: &mut Biquad, right: &mut Biquad) {
    let frames = samples.len() / 2;
    for i in 0..frames {
        samples[i * 2] = left.process(samples[i * 2]);
        samples[i * 2 + 1] = right.process(samples[i * 2 + 1]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowpass_kills_highs() {
        // Feed a 10kHz sine at 44.1kHz through a 150Hz LPF.
        let sr = 44100.0;
        let fc = 150.0;
        let mut lp = Biquad::lowpass(sr, fc);
        let mut rms_hi = 0.0f32;
        for n in 0..44100 {
            let t = n as f32 / sr;
            let x = (2.0 * std::f32::consts::PI * 10000.0 * t).sin();
            let y = lp.process(x);
            rms_hi += y * y;
        }
        rms_hi = (rms_hi / 44100.0).sqrt();
        assert!(rms_hi < 0.01, "lpf leaked 10kHz sine: rms={rms_hi}");
    }

    #[test]
    fn lowpass_preserves_lows() {
        let sr = 44100.0;
        let fc = 150.0;
        let mut lp = Biquad::lowpass(sr, fc);
        // 60Hz sine (kick fundamental range).
        let mut acc = 0.0f32;
        for n in 0..44100 {
            let t = n as f32 / sr;
            let x = (2.0 * std::f32::consts::PI * 60.0 * t).sin();
            let y = lp.process(x);
            if n > 4410 {
                // skip warmup
                acc += y * y;
            }
        }
        let rms = (acc / (44100.0 - 4410.0)).sqrt();
        // Sine RMS = 0.707. At 60Hz through 150Hz LPF, minimal attenuation.
        assert!(rms > 0.6, "lpf ate 60Hz sine: rms={rms}");
    }
}
