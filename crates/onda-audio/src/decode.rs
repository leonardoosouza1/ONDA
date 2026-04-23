//! Decode arbitrary audio formats (MP3/FLAC/WAV/OGG/...) to interleaved f32 stereo.
//!
//! We rely on symphonia for format/codec dispatch. Output is always normalized
//! to [-1.0, 1.0] and resampled/channel-mixed to the workspace canonical
//! (44.1kHz stereo) via `resample::resample_to` and a simple mono→stereo/
//! stereo→stereo path.

use std::fs::File;
use std::path::Path;

use onda_core::{AudioBuffer, OndaError, Result, SAMPLE_RATE};
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use tracing::{debug, warn};

use crate::resample::resample_to;

/// Decode a full audio file into an interleaved stereo f32 buffer at 44.1kHz.
///
/// Supported formats: whatever symphonia's default codec/format registry
/// handles (MP3, FLAC, WAV, OGG, AAC, AIFF, ...).
pub fn decode_file<P: AsRef<Path>>(path: P) -> Result<AudioBuffer> {
    let path = path.as_ref();
    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| OndaError::Decode(format!("probe: {e}")))?;

    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| OndaError::Decode("no default track".into()))?;

    let codec_params = track.codec_params.clone();
    let track_id = track.id;
    let src_rate = codec_params.sample_rate.unwrap_or(SAMPLE_RATE);
    let src_channels = codec_params
        .channels
        .ok_or_else(|| OndaError::Decode("no channel layout".into()))?
        .count() as u16;

    debug!(?path, src_rate, src_channels, "decoding");

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| OndaError::Decode(format!("codec: {e}")))?;

    // Decode into two buffers (one per channel) to make resample+interleave clean.
    let mut left = Vec::<f32>::new();
    let mut right = Vec::<f32>::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymphError::ResetRequired) => break,
            Err(e) => return Err(OndaError::Decode(format!("next_packet: {e}"))),
        };
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(decoded) => append_channels(decoded, &mut left, &mut right),
            Err(SymphError::DecodeError(e)) => warn!("decode err (skipped packet): {e}"),
            Err(e) => return Err(OndaError::Decode(format!("decode: {e}"))),
        }
    }

    // Resample per-channel if needed.
    let (left, right) = if src_rate != SAMPLE_RATE {
        (
            resample_to(&left, src_rate, SAMPLE_RATE)?,
            resample_to(&right, src_rate, SAMPLE_RATE)?,
        )
    } else {
        (left, right)
    };

    // Interleave to stereo.
    let n = left.len().min(right.len());
    let mut interleaved = Vec::with_capacity(n * 2);
    for i in 0..n {
        interleaved.push(left[i]);
        interleaved.push(right[i]);
    }

    Ok(AudioBuffer {
        samples: interleaved,
        sample_rate: SAMPLE_RATE,
        channels: 2,
    })
}

/// Append one decoded packet's samples to per-channel f32 buffers, converting
/// any symphonia sample format and mono→dup→stereo if needed.
fn append_channels(buf: AudioBufferRef<'_>, left: &mut Vec<f32>, right: &mut Vec<f32>) {
    match buf {
        AudioBufferRef::F32(b) => push_f32(&b, left, right),
        AudioBufferRef::S16(b) => {
            let mut tmp = b.make_equivalent::<f32>();
            b.convert(&mut tmp);
            push_f32(&tmp, left, right);
        }
        AudioBufferRef::S32(b) => {
            let mut tmp = b.make_equivalent::<f32>();
            b.convert(&mut tmp);
            push_f32(&tmp, left, right);
        }
        AudioBufferRef::U8(b) => {
            let mut tmp = b.make_equivalent::<f32>();
            b.convert(&mut tmp);
            push_f32(&tmp, left, right);
        }
        AudioBufferRef::F64(b) => {
            let mut tmp = b.make_equivalent::<f32>();
            b.convert(&mut tmp);
            push_f32(&tmp, left, right);
        }
        _ => warn!("unsupported sample format (skipped)"),
    }
}

fn push_f32(
    b: &symphonia::core::audio::AudioBuffer<f32>,
    left: &mut Vec<f32>,
    right: &mut Vec<f32>,
) {
    let spec = b.spec();
    let frames = b.frames();
    let n_chan = spec.channels.count();
    match n_chan {
        1 => {
            let ch = b.chan(0);
            left.extend_from_slice(&ch[..frames]);
            right.extend_from_slice(&ch[..frames]);
        }
        _ => {
            let l = b.chan(0);
            let r = b.chan(1);
            left.extend_from_slice(&l[..frames]);
            right.extend_from_slice(&r[..frames]);
        }
    }
}
