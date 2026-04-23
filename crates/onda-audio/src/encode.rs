//! Write interleaved stereo f32 to a 16-bit PCM WAV file.

use std::path::Path;

use hound::{SampleFormat, WavSpec, WavWriter};
use onda_core::{AudioBuffer, OndaError, Result};

/// Write an [`AudioBuffer`] as 16-bit PCM WAV. Samples outside [-1.0, 1.0] are
/// hard-clipped (caller is responsible for headroom).
pub fn write_wav<P: AsRef<Path>>(path: P, buf: &AudioBuffer) -> Result<()> {
    let spec = WavSpec {
        channels: buf.channels,
        sample_rate: buf.sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer =
        WavWriter::create(path, spec).map_err(|e| OndaError::Encode(format!("wav create: {e}")))?;
    for &s in &buf.samples {
        let clipped = s.clamp(-1.0, 1.0);
        let pcm = (clipped * i16::MAX as f32) as i16;
        writer
            .write_sample(pcm)
            .map_err(|e| OndaError::Encode(format!("wav write: {e}")))?;
    }
    writer
        .finalize()
        .map_err(|e| OndaError::Encode(format!("wav finalize: {e}")))?;
    Ok(())
}
