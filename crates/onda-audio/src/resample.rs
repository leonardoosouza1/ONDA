//! High-quality resampling via `rubato` (sinc-based async resampler).

use onda_core::{OndaError, Result};
use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};

/// Resample a mono f32 signal from `src_rate` to `dst_rate`.
///
/// Uses a sinc resampler with a windowed kernel. For short buffers this is
/// overkill but correctness matters more than speed at the decode stage.
pub fn resample_to(signal: &[f32], src_rate: u32, dst_rate: u32) -> Result<Vec<f32>> {
    if src_rate == dst_rate {
        return Ok(signal.to_vec());
    }
    if signal.is_empty() {
        return Ok(Vec::new());
    }

    let ratio = dst_rate as f64 / src_rate as f64;
    let chunk_size = 4096.min(signal.len());

    let params = SincInterpolationParameters {
        sinc_len: 128,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 128,
        window: WindowFunction::BlackmanHarris2,
    };

    let mut resampler = SincFixedIn::<f32>::new(ratio, 2.0, params, chunk_size, 1)
        .map_err(|e| OndaError::Dsp(format!("resampler init: {e}")))?;

    let mut output = Vec::<f32>::with_capacity((signal.len() as f64 * ratio) as usize + chunk_size);
    let mut pos = 0;

    while pos + chunk_size <= signal.len() {
        let chunk = vec![signal[pos..pos + chunk_size].to_vec()];
        let out = resampler
            .process(&chunk, None)
            .map_err(|e| OndaError::Dsp(format!("resample: {e}")))?;
        output.extend_from_slice(&out[0]);
        pos += chunk_size;
    }

    // Flush remaining samples (pad with zeros to chunk_size, trim output).
    if pos < signal.len() {
        let remaining = signal.len() - pos;
        let mut last = vec![0.0f32; chunk_size];
        last[..remaining].copy_from_slice(&signal[pos..]);
        let out = resampler
            .process(&[last], None)
            .map_err(|e| OndaError::Dsp(format!("resample tail: {e}")))?;
        let trim = (remaining as f64 * ratio).round() as usize;
        output.extend_from_slice(&out[0][..trim.min(out[0].len())]);
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_identity_passes_through() {
        let sig = vec![0.1, -0.2, 0.3, -0.4, 0.5];
        let out = resample_to(&sig, 44_100, 44_100).unwrap();
        assert_eq!(out, sig);
    }

    #[test]
    fn resample_upsample_doubles_length_approx() {
        let sig: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.01).sin()).collect();
        let out = resample_to(&sig, 22_050, 44_100).unwrap();
        // ~2x the input length (±5% tolerance for sinc transient).
        assert!(
            out.len() >= sig.len() * 2 * 95 / 100 && out.len() <= sig.len() * 2 * 105 / 100,
            "got {} expected ~{}",
            out.len(),
            sig.len() * 2
        );
    }
}
