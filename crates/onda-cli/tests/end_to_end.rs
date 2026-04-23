//! End-to-end pipeline test: synthesize → decode → crossfade → encode → decode.
//! Locks in the Phase-0 smoke test so regressions break CI.

use std::f32::consts::TAU;

use onda_audio::{decode_file, write_wav};
use onda_core::{AudioBuffer, SAMPLE_RATE};
use onda_dsp::crossfade::{build_crossfade_mix, CrossfadeCurve};
use onda_dsp::envelope::rms;

fn synth(freq: f32, bpm: f32, secs: f32) -> AudioBuffer {
    let total_frames = (secs * SAMPLE_RATE as f32) as usize;
    let beat_period = (60.0 / bpm * SAMPLE_RATE as f32) as usize;
    let mut samples = Vec::with_capacity(total_frames * 2);
    for i in 0..total_frames {
        let t = i as f32 / SAMPLE_RATE as f32;
        let tone = (t * TAU * freq).sin() * 0.25;
        let beat_phase = i % beat_period;
        let beat_t = beat_phase as f32 / SAMPLE_RATE as f32;
        let kick = (beat_t * TAU * (60.0 * (-beat_t * 8.0).exp() + 40.0)).sin()
            * (-beat_t * 25.0).exp()
            * 0.7;
        let l = (tone + kick).clamp(-1.0, 1.0);
        let r = (tone + kick).clamp(-1.0, 1.0);
        samples.push(l);
        samples.push(r);
    }
    AudioBuffer {
        samples,
        sample_rate: SAMPLE_RATE,
        channels: 2,
    }
}

#[test]
fn synth_crossfade_decode_roundtrip_preserves_duration_and_avoids_clipping() {
    let tmp = std::env::temp_dir().join("onda_e2e_test");
    std::fs::create_dir_all(&tmp).unwrap();
    let a_path = tmp.join("a.wav");
    let b_path = tmp.join("b.wav");
    let mix_path = tmp.join("mix.wav");

    let a = synth(220.0, 120.0, 6.0);
    let b = synth(293.66, 128.0, 6.0);
    write_wav(&a_path, &a).unwrap();
    write_wav(&b_path, &b).unwrap();

    let a_decoded = decode_file(&a_path).unwrap();
    let b_decoded = decode_file(&b_path).unwrap();
    assert_eq!(a_decoded.channels, 2);
    assert!((a_decoded.duration_secs() - 6.0).abs() < 0.01);

    let mix = build_crossfade_mix(&a_decoded, &b_decoded, 2.0, CrossfadeCurve::EqualPower).unwrap();
    // 6 + 6 - 2 = 10 seconds
    assert!((mix.duration_secs() - 10.0).abs() < 0.01);

    // No clipping: every sample strictly inside [-1.0, 1.0].
    let max_abs = mix.samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    assert!(max_abs < 1.0, "clipped at {max_abs}");

    // Sanity on loudness — we expect non-silent output.
    let r = rms(&mix.samples);
    assert!(r > 0.01, "suspiciously quiet: rms={r}");

    // Roundtrip through WAV and confirm length preserved.
    write_wav(&mix_path, &mix).unwrap();
    let rt = decode_file(&mix_path).unwrap();
    assert!((rt.duration_secs() - 10.0).abs() < 0.01);

    // cleanup
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn crossfade_too_long_is_rejected() {
    let a = synth(220.0, 120.0, 2.0);
    let b = synth(250.0, 120.0, 2.0);
    // Fade 5s on 2s tracks → must error.
    let err = build_crossfade_mix(&a, &b, 5.0, CrossfadeCurve::Linear).unwrap_err();
    assert!(err.to_string().contains("too long"));
}
