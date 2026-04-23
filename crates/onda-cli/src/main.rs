//! ONDA — Organic Neural DJ Agent. CLI entry point.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use onda_audio::{decode_file, write_wav};
use onda_dsp::crossfade::{build_crossfade_mix, CrossfadeCurve};
use tracing::info;

#[derive(Parser, Debug)]
#[command(name = "onda", version, about = "Organic Neural DJ Agent")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Pipeline smoke test: crossfade two audio files with a given fade length.
    /// No AI yet — just validates decode → DSP → encode plumbing.
    Crossfade {
        /// Input A (played first, faded out at the tail).
        #[arg(long)]
        in_a: PathBuf,
        /// Input B (played second, faded in at the head).
        #[arg(long)]
        in_b: PathBuf,
        /// Output WAV path (16-bit PCM, 44.1kHz stereo).
        #[arg(long)]
        out: PathBuf,
        /// Crossfade duration in seconds.
        #[arg(long, default_value_t = 8.0)]
        fade_secs: f32,
        /// "linear" or "equal-power" (default).
        #[arg(long, default_value = "equal-power")]
        curve: String,
    },

    /// Print info about an audio file (duration, rate, channels).
    Info {
        #[arg(long)]
        input: PathBuf,
    },

    /// Generate synthetic test audio: sine + kick-sim for pipeline smoke tests.
    Synth {
        #[arg(long)]
        out: PathBuf,
        /// Duration in seconds.
        #[arg(long, default_value_t = 8.0)]
        secs: f32,
        /// Base tone frequency in Hz (default A3 = 220Hz).
        #[arg(long, default_value_t = 220.0)]
        freq: f32,
        /// Tempo in BPM (drives kick-drum simulation).
        #[arg(long, default_value_t = 120.0)]
        bpm: f32,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Crossfade {
            in_a,
            in_b,
            out,
            fade_secs,
            curve,
        } => run_crossfade(in_a, in_b, out, fade_secs, &curve),
        Cmd::Info { input } => run_info(input),
        Cmd::Synth {
            out,
            secs,
            freq,
            bpm,
        } => run_synth(out, secs, freq, bpm),
    }
}

fn run_synth(out: PathBuf, secs: f32, freq: f32, bpm: f32) -> Result<()> {
    use onda_core::{AudioBuffer, SAMPLE_RATE};
    use std::f32::consts::TAU;

    let total_frames = (secs * SAMPLE_RATE as f32) as usize;
    let beat_period = (60.0 / bpm * SAMPLE_RATE as f32) as usize;
    let mut samples = Vec::with_capacity(total_frames * 2);

    for i in 0..total_frames {
        let t = i as f32 / SAMPLE_RATE as f32;
        // Sustained tone (gently detuned stereo).
        let tone_l = (t * TAU * freq).sin() * 0.25;
        let tone_r = (t * TAU * freq * 1.003).sin() * 0.25;

        // Kick-drum simulation: exponentially-decayed sine on every beat.
        let beat_phase = i % beat_period;
        let beat_t = beat_phase as f32 / SAMPLE_RATE as f32;
        let kick_env = (-beat_t * 25.0).exp();
        let kick_freq = 60.0 * (-beat_t * 8.0).exp() + 40.0;
        let kick = (beat_t * TAU * kick_freq).sin() * kick_env * 0.7;

        let l = (tone_l + kick).clamp(-1.0, 1.0);
        let r = (tone_r + kick).clamp(-1.0, 1.0);
        samples.push(l);
        samples.push(r);
    }

    let buf = AudioBuffer {
        samples,
        sample_rate: SAMPLE_RATE,
        channels: 2,
    };
    info!(out=?out, secs, freq, bpm, "writing synthetic audio");
    onda_audio::write_wav(&out, &buf)?;
    Ok(())
}

fn run_crossfade(
    in_a: PathBuf,
    in_b: PathBuf,
    out: PathBuf,
    fade_secs: f32,
    curve: &str,
) -> Result<()> {
    let curve = match curve {
        "linear" => CrossfadeCurve::Linear,
        "equal-power" | "eq" | "eqp" => CrossfadeCurve::EqualPower,
        other => anyhow::bail!("unknown curve: {other}"),
    };

    info!(path=?in_a, "decoding A");
    let a = decode_file(&in_a).with_context(|| format!("decoding {}", in_a.display()))?;
    info!(path=?in_b, "decoding B");
    let b = decode_file(&in_b).with_context(|| format!("decoding {}", in_b.display()))?;

    info!(
        a_secs = a.duration_secs(),
        b_secs = b.duration_secs(),
        fade_secs,
        "building crossfade mix"
    );
    let mix = build_crossfade_mix(&a, &b, fade_secs, curve)
        .context("building crossfade mix")?;

    info!(out=?out, mix_secs = mix.duration_secs(), "writing WAV");
    write_wav(&out, &mix).with_context(|| format!("writing {}", out.display()))?;

    info!(mix_secs = mix.duration_secs(), "done");
    Ok(())
}

fn run_info(input: PathBuf) -> Result<()> {
    let buf = decode_file(&input).with_context(|| format!("decoding {}", input.display()))?;
    println!(
        "{}: {:.3}s @ {}Hz, {} channels, {} frames",
        input.display(),
        buf.duration_secs(),
        buf.sample_rate,
        buf.channels,
        buf.frames()
    );
    Ok(())
}
