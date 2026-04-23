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
        /// If set, ignore --fade-secs and run a beat-aligned crossfade of N bars
        /// (assumes 4/4). Uses detected tempo of track A.
        #[arg(long)]
        bars: Option<u32>,
    },

    /// Print info about an audio file (duration, rate, channels).
    Info {
        #[arg(long)]
        input: PathBuf,
    },

    /// Analyze an audio file: estimate tempo, count beats, print summary.
    Analyze {
        #[arg(long)]
        input: PathBuf,
    },

    /// Download audio via yt-dlp (Rust → uv run tools/download.py).
    /// Accepts YouTube, SoundCloud, Bandcamp, archive.org, direct MP3, etc.
    Download {
        /// Source URL.
        #[arg(long)]
        url: String,
        /// Output directory (filename chosen from track id/title).
        #[arg(long)]
        out_dir: PathBuf,
        /// Audio format: mp3 (default), m4a, or bestaudio (no transcode).
        #[arg(long, default_value = "mp3")]
        format: String,
        /// Skip tracks longer than N seconds (default 900 = 15min).
        #[arg(long, default_value_t = 900)]
        max_duration: u32,
    },

    /// Fetch tracks from YouTube via yt-dlp.
    /// `--query` accepts free text (`"house 128 bpm"`) OR a channel/playlist URL
    /// (e.g. `https://www.youtube.com/@NoCopyrightSounds/videos`).
    /// Downloads as m4a (native YouTube audio, no ffmpeg needed).
    YtFetch {
        /// Free-text search OR channel/playlist URL.
        #[arg(long)]
        query: String,
        /// Target successful downloads.
        #[arg(long, default_value_t = 10)]
        count: u32,
        /// Minimum track duration (seconds), skips shorts.
        #[arg(long, default_value_t = 120)]
        min_duration: u32,
        /// Maximum track duration (seconds), skips long DJ mixes.
        #[arg(long, default_value_t = 420)]
        max_duration: u32,
        /// Output library dir.
        #[arg(long, default_value = "library/")]
        out_dir: PathBuf,
    },

    /// Analyze every audio file in a directory, pick the best BPM-compatible pair,
    /// and render a beat-aligned mix. Fully mechanical (no model yet).
    PlanMix {
        /// Directory of audio files to scan.
        #[arg(long)]
        dir: PathBuf,
        /// Output WAV.
        #[arg(long)]
        out: PathBuf,
        /// Max BPM gap allowed in the pair (percent, default 8).
        #[arg(long, default_value_t = 8.0)]
        max_bpm_gap_pct: f32,
        /// Crossfade bars (4/4 assumed).
        #[arg(long, default_value_t = 16)]
        bars: u32,
    },

    /// Programmatic mix validation: slice a mix into pre/crossfade/post regions
    /// at given timestamps and report BPM/RMS/centroid stability.
    Diagnose {
        #[arg(long)]
        input: PathBuf,
        /// Fade-start time (A's tail starts here), seconds.
        #[arg(long)]
        fade_start: f32,
        /// Overlap duration, seconds.
        #[arg(long)]
        overlap: f32,
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
            bars,
        } => run_crossfade(in_a, in_b, out, fade_secs, &curve, bars),
        Cmd::Info { input } => run_info(input),
        Cmd::Analyze { input } => run_analyze(input),
        Cmd::Synth {
            out,
            secs,
            freq,
            bpm,
        } => run_synth(out, secs, freq, bpm),
        Cmd::Download {
            url,
            out_dir,
            format,
            max_duration,
        } => run_download(url, out_dir, format, max_duration),
        Cmd::YtFetch {
            query,
            count,
            min_duration,
            max_duration,
            out_dir,
        } => run_yt_fetch(query, count, min_duration, max_duration, out_dir),
        Cmd::PlanMix {
            dir,
            out,
            max_bpm_gap_pct,
            bars,
        } => run_plan_mix(dir, out, max_bpm_gap_pct, bars),
        Cmd::Diagnose {
            input,
            fade_start,
            overlap,
        } => run_diagnose(input, fade_start, overlap),
    }
}

/// Spectral centroid per frame of a spectrogram (Hz).
fn centroid_per_frame(s: &onda_features::spectrum::Spectrogram) -> Vec<f32> {
    use onda_features::spectrum::StftParams;
    let bin_hz = 44100.0 / (StftParams::MIR_STANDARD.win_size as f32);
    let n_frames = s.n_frames();
    let mut out = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        let frame = &s.magnitudes[f];
        let mut num = 0.0f32;
        let mut den = 0.0f32;
        for (k, &m) in frame.iter().enumerate() {
            num += m * (k as f32 * bin_hz);
            den += m;
        }
        out.push(if den > 1e-6 { num / den } else { 0.0 });
    }
    out
}

/// Like pick_musical_fade_start, plus a spectral-compatibility term that
/// rewards fade points where A's local centroid matches B's incoming centroid.
#[allow(clippy::too_many_arguments)]
fn pick_musical_fade_start_spectral(
    beats_sec: &[f32],
    cuts: &[onda_features::sections::CutCandidate],
    track_duration: f32,
    overlap_secs: f32,
    tolerance_secs: f32,
    a_centroid_per_frame: &[f32],
    frame_dur: f32,
    b_target_centroid: f32,
) -> f32 {
    let deadline = (track_duration - overlap_secs).max(0.0);
    let centroid_at = |t: f32| -> f32 {
        // Average centroid over a ±1.5s window around t.
        if a_centroid_per_frame.is_empty() || frame_dur <= 0.0 {
            return 0.0;
        }
        let half = 1.5;
        let lo = ((t - half) / frame_dur).max(0.0) as usize;
        let hi = (((t + half) / frame_dur) as usize + 1).min(a_centroid_per_frame.len());
        if hi <= lo {
            return a_centroid_per_frame.get(lo).copied().unwrap_or(0.0);
        }
        let sum: f32 = a_centroid_per_frame[lo..hi].iter().sum();
        sum / (hi - lo) as f32
    };

    let mut best: Option<(f32, f32)> = None;
    for (i, &t) in beats_sec.iter().enumerate() {
        if t > deadline {
            continue;
        }
        let near = cuts
            .iter()
            .map(|c| ((c.time_sec - t).abs(), c))
            .filter(|(d, _)| *d <= tolerance_secs)
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let cut_score = if let Some((dist, c)) = near {
            let proximity = 1.0 - (dist / tolerance_secs);
            2.0 * proximity * c.drop_ratio
        } else {
            0.0
        };
        let late_bonus = t / track_duration.max(1.0) * 0.5;
        let downbeat_bonus = if i % 4 == 0 { 0.15 } else { 0.0 };

        // Spectral match: exp(-|Δcentroid| / 800Hz). 0..1, where 1 = A's
        // local brightness matches B's incoming brightness.
        let spectral_match = if b_target_centroid > 1e-3 {
            let c = centroid_at(t);
            let delta = (c - b_target_centroid).abs();
            (-(delta / 800.0)).exp()
        } else {
            0.5
        };

        // Gate structural/lateness bonuses by spectral compatibility — a cut
        // into a spectrally-incompatible region is worse than no cut at all.
        let structural = cut_score + late_bonus + downbeat_bonus;
        let score = structural * (0.3 + 0.7 * spectral_match) + 1.5 * spectral_match;
        if best.map(|(s, _)| score > s).unwrap_or(true) {
            best = Some((score, t));
        }
    }
    best.map(|(_, t)| t).unwrap_or(deadline)
}

fn region_stats(buf: &onda_core::AudioBuffer) -> (f32, f32, f32) {
    use onda_features::{
        spectrum::{stft_mono, StftParams},
        tempo::estimate_tempo,
    };
    let rms = onda_dsp::envelope::rms(&buf.samples);
    let spec = stft_mono(buf, StftParams::MIR_STANDARD).ok();
    let (bpm, centroid) = if let Some(s) = spec {
        let t = estimate_tempo(&s);
        // spectral centroid averaged over frames (brightness proxy).
        let n_frames = s.n_frames();
        let bin_hz = 44100.0 / (StftParams::MIR_STANDARD.win_size as f32);
        let mut acc = 0.0f32;
        let mut n = 0u32;
        for f in 0..n_frames {
            let frame = &s.magnitudes[f];
            let mut num = 0.0f32;
            let mut den = 0.0f32;
            for (k, &m) in frame.iter().enumerate() {
                num += m * (k as f32 * bin_hz);
                den += m;
            }
            if den > 1e-6 {
                acc += num / den;
                n += 1;
            }
        }
        let centroid = if n > 0 { acc / n as f32 } else { 0.0 };
        (t.bpm, centroid)
    } else {
        (0.0, 0.0)
    };
    (bpm, rms, centroid)
}

fn run_diagnose(input: PathBuf, fade_start: f32, overlap: f32) -> Result<()> {
    use onda_core::{AudioBuffer, SAMPLE_RATE};
    let buf = decode_file(&input)?;
    let frame_of = |t: f32| ((t * SAMPLE_RATE as f32) as usize * 2).min(buf.samples.len());
    let pre_end = frame_of(fade_start);
    let cf_end = frame_of(fade_start + overlap);
    let pre = AudioBuffer {
        samples: buf.samples[..pre_end].to_vec(),
        sample_rate: SAMPLE_RATE,
        channels: 2,
    };
    let cf = AudioBuffer {
        samples: buf.samples[pre_end..cf_end].to_vec(),
        sample_rate: SAMPLE_RATE,
        channels: 2,
    };
    let post = AudioBuffer {
        samples: buf.samples[cf_end..].to_vec(),
        sample_rate: SAMPLE_RATE,
        channels: 2,
    };

    // Take a shorter slice of pre/post for tempo stability (first/last 30s).
    let trim_secs = 30.0;
    let trim_samples = ((trim_secs * SAMPLE_RATE as f32) as usize) * 2;
    let pre_tail_start = pre.samples.len().saturating_sub(trim_samples);
    let pre_tail = AudioBuffer {
        samples: pre.samples[pre_tail_start..].to_vec(),
        sample_rate: SAMPLE_RATE,
        channels: 2,
    };
    let post_head = AudioBuffer {
        samples: post.samples[..post.samples.len().min(trim_samples)].to_vec(),
        sample_rate: SAMPLE_RATE,
        channels: 2,
    };

    println!("=== ONDA mix diagnose: {} ===", input.display());
    println!(
        "  total      {:>7.2}s   (pre={:.2}s  xfade={:.2}s  post={:.2}s)",
        buf.duration_secs(),
        pre.duration_secs(),
        cf.duration_secs(),
        post.duration_secs()
    );

    let print_region = |name: &str, buf: &AudioBuffer| {
        if buf.samples.is_empty() {
            println!("  {name:<10} (empty)");
            return;
        }
        let (bpm, rms, centroid) = region_stats(buf);
        println!(
            "  {name:<10} dur={:>6.2}s  BPM={:>7.2}  RMS={:.4}  centroid={:>6.0}Hz",
            buf.duration_secs(),
            bpm,
            rms,
            centroid
        );
    };
    print_region("pre.tail30", &pre_tail);
    print_region("crossfade", &cf);
    print_region("post.head30", &post_head);

    // Spectral continuity — compare crossfade vs pre-tail and post-head.
    let (_, rms_pre, c_pre) = region_stats(&pre_tail);
    let (_, rms_cf, c_cf) = region_stats(&cf);
    let (_, rms_post, c_post) = region_stats(&post_head);
    let rms_jump_in = (rms_cf - rms_pre).abs() / rms_pre.max(1e-6);
    let rms_jump_out = (rms_post - rms_cf).abs() / rms_cf.max(1e-6);
    let c_jump_in = (c_cf - c_pre).abs() / c_pre.max(1.0);
    let c_jump_out = (c_post - c_cf).abs() / c_cf.max(1.0);
    println!("  loudness jump  pre→xfade {:>4.0}%   xfade→post {:>4.0}%",
        rms_jump_in * 100.0, rms_jump_out * 100.0);
    println!("  brightness jmp pre→xfade {:>4.0}%   xfade→post {:>4.0}%",
        c_jump_in * 100.0, c_jump_out * 100.0);
    Ok(())
}

fn run_yt_fetch(
    query: String,
    count: u32,
    min_duration: u32,
    max_duration: u32,
    out_dir: PathBuf,
) -> Result<()> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    rt.block_on(async move {
        info!(%query, count, "yt-dlp search");
        let hits = onda_download::yt_search(&query, count.saturating_mul(2), min_duration, max_duration)
            .await
            .map_err(|e| anyhow::anyhow!("yt search: {e}"))?;
        println!("=== YouTube hits: {} ===", hits.len());

        // m4a = native YouTube audio container, no ffmpeg postprocessing.
        let opts = onda_download::DownloadOptions {
            format: onda_download::Format::M4a,
            max_duration_secs: max_duration,
        };
        tokio::fs::create_dir_all(&out_dir).await?;

        let mut ok = 0u32;
        let mut fail = 0u32;
        for (i, hit) in hits.iter().enumerate() {
            if ok >= count {
                break;
            }
            let title = hit.title.as_deref().unwrap_or(&hit.url);
            println!("→ [{:>2}] {} ({:.0}s)", i + 1, title, hit.duration.unwrap_or(0.0));
            match onda_download::download(&hit.url, &out_dir, &opts).await {
                Ok(info) => {
                    ok += 1;
                    println!("    ✓ {}", info.path.file_name().unwrap().to_string_lossy());
                }
                Err(e) => {
                    fail += 1;
                    println!("    ✗ skip: {e}");
                }
            }
        }
        println!("=== done: {} ok, {} fail ===", ok, fail);
        if ok == 0 {
            anyhow::bail!("no tracks downloaded");
        }
        Ok::<_, anyhow::Error>(())
    })
}

fn run_download(url: String, out_dir: PathBuf, format: String, max_duration: u32) -> Result<()> {
    let fmt = match format.as_str() {
        "mp3" => onda_download::Format::Mp3,
        "m4a" => onda_download::Format::M4a,
        "bestaudio" => onda_download::Format::BestAudio,
        other => anyhow::bail!("unknown format '{other}' (use mp3|m4a|bestaudio)"),
    };
    let opts = onda_download::DownloadOptions {
        format: fmt,
        max_duration_secs: max_duration,
    };
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let info = rt
        .block_on(onda_download::download(&url, &out_dir, &opts))
        .map_err(|e| anyhow::anyhow!("download: {e}"))?;

    println!("=== ONDA download ===");
    println!("  path:     {}", info.path.display());
    if let Some(t) = &info.title {
        println!("  title:    {t}");
    }
    if let Some(u) = &info.uploader {
        println!("  uploader: {u}");
    }
    if let Some(d) = info.duration {
        println!("  duration: {d:.1}s");
    }
    if let Some(e) = &info.extractor {
        println!("  source:   {e}");
    }
    if let Some(w) = &info.webpage_url {
        println!("  url:      {w}");
    }
    Ok(())
}

fn run_analyze(input: PathBuf) -> Result<()> {
    use onda_features::{
        beats::beat_grid_from_tempo,
        onsets::{onset_envelope, OnsetParams},
        spectrum::{stft_mono, StftParams},
        tempo::estimate_tempo,
    };

    info!(path=?input, "decoding for analysis");
    let buf = decode_file(&input).with_context(|| format!("decoding {}", input.display()))?;
    let spec = stft_mono(&buf, StftParams::MIR_STANDARD)
        .map_err(|e| anyhow::anyhow!("stft: {e}"))?;
    let env = onset_envelope(&spec, OnsetParams::default());
    let t = estimate_tempo(&spec);
    let grid = beat_grid_from_tempo(&env, spec.frame_duration(), t.bpm);

    println!("=== ONDA analysis: {} ===", input.display());
    println!("  duration:       {:.3}s @ {}Hz stereo", buf.duration_secs(), buf.sample_rate);
    println!("  stft frames:    {} (hop {}ms, win {}ms)",
        spec.n_frames(),
        (spec.frame_duration() * 1000.0).round(),
        (StftParams::MIR_STANDARD.win_size as f32 / buf.sample_rate as f32 * 1000.0).round(),
    );
    println!("  tempo:          {:.2} BPM (confidence {:.3})", t.bpm, t.confidence);
    println!("  beat period:    {:.3}s", grid.period_sec);
    println!("  beats detected: {} (alignment quality {:.2})", grid.beats.len(), grid.quality);
    if grid.beats.len() >= 4 {
        print!("  first 4 beats:  ");
        for b in &grid.beats[..4] {
            print!("{:.3}s  ", b);
        }
        println!();
    }
    Ok(())
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

#[derive(Debug, Clone)]
struct TrackAnalysis {
    path: PathBuf,
    duration: f32,
    bpm: f32,
    confidence: f32,
    rms: f32,
    /// Average spectral centroid of the LAST 30s (what A sounds like at its outro).
    outro_centroid: f32,
    /// Average spectral centroid of the FIRST 30s (what B sounds like coming in).
    intro_centroid: f32,
}

fn analyze_track(path: &PathBuf) -> Result<TrackAnalysis> {
    use onda_features::{
        spectrum::{stft_mono, StftParams},
        tempo::estimate_tempo,
    };
    let buf = decode_file(path).with_context(|| format!("decoding {}", path.display()))?;
    let spec = stft_mono(&buf, StftParams::MIR_STANDARD)
        .map_err(|e| anyhow::anyhow!("stft: {e}"))?;
    let t = estimate_tempo(&spec);
    let rms = onda_dsp::envelope::rms(&buf.samples);
    let cs = centroid_per_frame(&spec);
    let fd = spec.frame_duration();
    let win = (30.0 / fd).ceil() as usize;
    let intro_centroid = if cs.is_empty() {
        0.0
    } else {
        let n = cs.len().min(win.max(1));
        cs[..n].iter().sum::<f32>() / n as f32
    };
    let outro_centroid = if cs.is_empty() {
        0.0
    } else {
        let n = cs.len().min(win.max(1));
        let start = cs.len() - n;
        cs[start..].iter().sum::<f32>() / n as f32
    };
    Ok(TrackAnalysis {
        path: path.clone(),
        duration: buf.duration_secs() as f32,
        bpm: t.bpm,
        confidence: t.confidence,
        rms,
        outro_centroid,
        intro_centroid,
    })
}

fn list_audio_files(dir: &PathBuf) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir)
        .with_context(|| format!("reading dir {}", dir.display()))?
    {
        let entry = entry?;
        let p = entry.path();
        if !p.is_file() {
            continue;
        }
        match p.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()) {
            Some(ref e) if matches!(e.as_str(), "mp3" | "wav" | "flac" | "ogg" | "m4a") => {
                out.push(p)
            }
            _ => {}
        }
    }
    out.sort();
    Ok(out)
}

fn run_plan_mix(
    dir: PathBuf,
    out: PathBuf,
    max_bpm_gap_pct: f32,
    bars: u32,
) -> Result<()> {
    use onda_audio::resample::resample_to;
    use onda_dsp::{
        crossfade::beat_aligned_crossfade,
        stretch::{bpm_match_ratio, turntable_stretch},
    };
    use onda_features::{
        beats::beat_grid_from_tempo,
        onsets::{onset_envelope, OnsetParams},
        sections::{find_cut_candidates, rms_envelope, smooth},
        spectrum::{stft_mono, StftParams},
        tempo::estimate_tempo,
    };

    let files = list_audio_files(&dir)?;
    if files.len() < 2 {
        anyhow::bail!("need at least 2 audio files in {}", dir.display());
    }
    println!("=== analyzing {} files ===", files.len());
    let mut analyses = Vec::with_capacity(files.len());
    for (i, f) in files.iter().enumerate() {
        match analyze_track(f) {
            Ok(a) => {
                println!(
                    "  [{:>2}] {:>7.2} BPM (conf {:.2}) dur {:>6.1}s rms {:.3} intro={:>4.0}Hz outro={:>4.0}Hz  {}",
                    i + 1,
                    a.bpm,
                    a.confidence,
                    a.duration,
                    a.rms,
                    a.intro_centroid,
                    a.outro_centroid,
                    f.file_name().unwrap().to_string_lossy()
                );
                analyses.push(a);
            }
            Err(e) => println!("  [{:>2}] ERROR {}: {e}", i + 1, f.display()),
        }
    }

    // Pick best pair: minimize BPM gap among valid candidates, break ties by
    // higher combined confidence and RMS similarity.
    let mut best: Option<(f32, usize, usize)> = None; // (score_lower_is_better, a, b)
    for i in 0..analyses.len() {
        for j in 0..analyses.len() {
            if i == j {
                continue;
            }
            let a = &analyses[i];
            let b = &analyses[j];
            if a.duration < 30.0 || b.duration < 30.0 {
                continue;
            }
            let gap_pct = (a.bpm - b.bpm).abs() / a.bpm * 100.0;
            if gap_pct > max_bpm_gap_pct {
                continue;
            }
            // Need room in A for overlap + at least 30s intro of A.
            let overlap_secs = (bars * 4) as f32 * 60.0 / a.bpm;
            if a.duration < overlap_secs + 30.0 {
                continue;
            }
            let conf_penalty = 2.0 - (a.confidence + b.confidence);
            let rms_penalty = (a.rms - b.rms).abs() * 5.0;
            // Spectral compat: A's outro should resemble B's intro brightness.
            // |Δcentroid|/1000 as a penalty (0.5 per 500Hz of mismatch).
            let spec_penalty =
                (a.outro_centroid - b.intro_centroid).abs() / 1000.0;
            let score = gap_pct + conf_penalty + rms_penalty + spec_penalty;
            match best {
                None => best = Some((score, i, j)),
                Some((sb, _, _)) if score < sb => best = Some((score, i, j)),
                _ => {}
            }
        }
    }

    let (score, ia, ib) = best.ok_or_else(|| {
        anyhow::anyhow!(
            "no BPM-compatible pair within {:.1}% — try a larger gap",
            max_bpm_gap_pct
        )
    })?;
    let a_meta = &analyses[ia];
    let b_meta = &analyses[ib];
    println!("\n=== chosen pair (score {:.2}) ===", score);
    println!(
        "  A: {:>7.2} BPM  outro={:>4.0}Hz  {}",
        a_meta.bpm,
        a_meta.outro_centroid,
        a_meta.path.file_name().unwrap().to_string_lossy()
    );
    println!(
        "  B: {:>7.2} BPM  intro={:>4.0}Hz  {}",
        b_meta.bpm,
        b_meta.intro_centroid,
        b_meta.path.file_name().unwrap().to_string_lossy()
    );

    // Run the same beat-aligned mix as `crossfade --bars`.
    let a = decode_file(&a_meta.path)?;
    let b = decode_file(&b_meta.path)?;
    let spec_a = stft_mono(&a, StftParams::MIR_STANDARD)
        .map_err(|e| anyhow::anyhow!("stft A: {e}"))?;
    let spec_b = stft_mono(&b, StftParams::MIR_STANDARD)
        .map_err(|e| anyhow::anyhow!("stft B: {e}"))?;
    let tempo_a = estimate_tempo(&spec_a);
    let tempo_b = estimate_tempo(&spec_b);
    let env_a = onset_envelope(&spec_a, OnsetParams::default());
    let grid_a = beat_grid_from_tempo(&env_a, spec_a.frame_duration(), tempo_a.bpm);

    let ratio = bpm_match_ratio(tempo_b.bpm, tempo_a.bpm);
    info!(ratio, "turntable stretch B");
    let b_stretched = turntable_stretch(&b, ratio, resample_to)?;

    let spec_b2 = stft_mono(&b_stretched, StftParams::MIR_STANDARD)
        .map_err(|e| anyhow::anyhow!("stft B2: {e}"))?;
    let env_b2 = onset_envelope(&spec_b2, OnsetParams::default());
    let grid_b2 = beat_grid_from_tempo(&env_b2, spec_b2.frame_duration(), tempo_a.bpm);
    let b_first = grid_b2.beats.first().copied().unwrap_or(0.0);

    let overlap_secs = (bars * 4) as f32 * 60.0 / tempo_a.bpm;
    // Musical cut-point: detect energy drops in A (outros/breakdowns) and
    // prefer beats near them. Window = 0.25s, memory = 8 bars, threshold 20%.
    let bar_secs = 4.0 * 60.0 / tempo_a.bpm;
    let rms_env = rms_envelope(&a, 0.25);
    let rms_smooth = smooth(&rms_env, 4);
    let cuts = find_cut_candidates(&rms_smooth, 0.25, 8.0 * bar_secs, 20.0);
    println!("  A cut candidates: {} (drops ≥20% over 8 bars)", cuts.len());
    for c in cuts.iter().take(5) {
        println!(
            "    @ {:>6.1}s  drop {:>4.0}%  conf {:.2}",
            c.time_sec,
            c.drop_ratio * 100.0,
            c.confidence
        );
    }

    // Spectral compatibility: compute centroid per frame on A, and a target
    // centroid = average of B's first 30s. Prefer A fade-start points where
    // A's local brightness matches B's incoming brightness.
    let centroid_series = centroid_per_frame(&spec_a);
    let frame_dur = spec_a.frame_duration();
    let b_target_centroid = {
        let win_frames = (30.0 / spec_b2.frame_duration()).ceil() as usize;
        let cs = centroid_per_frame(&spec_b2);
        if cs.is_empty() {
            0.0
        } else {
            let n = cs.len().min(win_frames.max(1));
            cs[..n].iter().sum::<f32>() / n as f32
        }
    };
    println!("  B target centroid ≈ {:.0} Hz", b_target_centroid);

    let a_fade_start = pick_musical_fade_start_spectral(
        &grid_a.beats,
        &cuts,
        a.duration_secs() as f32,
        overlap_secs,
        bar_secs * 2.0,
        &centroid_series,
        frame_dur,
        b_target_centroid,
    );

    // Debug: dump top-10 scored candidates in the final 90s for inspection.
    let deadline = (a.duration_secs() as f32 - overlap_secs).max(0.0);
    let mut scored: Vec<(f32, f32, f32)> = grid_a
        .beats
        .iter()
        .copied()
        .filter(|&t| t <= deadline && t >= deadline - 90.0)
        .map(|t| {
            let idx = (t / frame_dur) as usize;
            let half = (1.5 / frame_dur) as usize;
            let lo = idx.saturating_sub(half);
            let hi = (idx + half + 1).min(centroid_series.len());
            let c = if hi > lo {
                centroid_series[lo..hi].iter().sum::<f32>() / (hi - lo) as f32
            } else {
                0.0
            };
            let near = cuts
                .iter()
                .map(|c| ((c.time_sec - t).abs(), c))
                .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(d, c)| (d, c.drop_ratio))
                .unwrap_or((999.0, 0.0));
            (t, c, near.1)
        })
        .collect();
    scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    println!("  top beats in last 90s (time, centroid Hz, nearest cut drop):");
    for (t, c, d) in scored.iter().take(8) {
        println!("    t={:>6.1}s  centroid={:>5.0}Hz  drop={:>4.0}%", t, c, d * 100.0);
    }

    info!(overlap_secs, a_fade_start, b_first, "plan-mix crossfade (musical, bass-swap)");
    let mix = beat_aligned_crossfade(
        &a,
        &b_stretched,
        bars,
        tempo_a.bpm,
        a_fade_start,
        b_first,
        CrossfadeCurve::BassSwap { cutoff_hz: 150.0 },
    )?;
    write_wav(&out, &mix)?;
    println!("\n→ mix written: {} ({:.1}s)", out.display(), mix.duration_secs());
    Ok(())
}

fn run_crossfade(
    in_a: PathBuf,
    in_b: PathBuf,
    out: PathBuf,
    fade_secs: f32,
    curve: &str,
    bars: Option<u32>,
) -> Result<()> {
    use onda_audio::resample::resample_to;
    use onda_dsp::{
        crossfade::{beat_aligned_crossfade, pick_a_fade_start},
        stretch::{bpm_match_ratio, turntable_stretch},
    };
    use onda_features::{
        beats::beat_grid_from_tempo,
        onsets::{onset_envelope, OnsetParams},
        spectrum::{stft_mono, StftParams},
        tempo::estimate_tempo,
    };

    let curve = match curve {
        "linear" => CrossfadeCurve::Linear,
        "equal-power" | "eq" | "eqp" => CrossfadeCurve::EqualPower,
        "bass-swap" | "bs" => CrossfadeCurve::BassSwap { cutoff_hz: 150.0 },
        other => anyhow::bail!("unknown curve: {other}"),
    };

    info!(path=?in_a, "decoding A");
    let a = decode_file(&in_a).with_context(|| format!("decoding {}", in_a.display()))?;
    info!(path=?in_b, "decoding B");
    let b = decode_file(&in_b).with_context(|| format!("decoding {}", in_b.display()))?;

    let mix = if let Some(bars) = bars {
        // 1. Analyze both tracks: tempo + beat grid.
        let spec_a = stft_mono(&a, StftParams::MIR_STANDARD)
            .map_err(|e| anyhow::anyhow!("stft A: {e}"))?;
        let spec_b = stft_mono(&b, StftParams::MIR_STANDARD)
            .map_err(|e| anyhow::anyhow!("stft B: {e}"))?;
        let tempo_a = estimate_tempo(&spec_a);
        let tempo_b = estimate_tempo(&spec_b);
        let env_a = onset_envelope(&spec_a, OnsetParams::default());
        let grid_a = beat_grid_from_tempo(&env_a, spec_a.frame_duration(), tempo_a.bpm);

        // 2. Turntable-stretch B to match A's BPM (pitch + tempo, vinyl-style).
        let stretch_ratio = bpm_match_ratio(tempo_b.bpm, tempo_a.bpm);
        info!(
            tempo_a = tempo_a.bpm,
            tempo_b = tempo_b.bpm,
            stretch_ratio,
            "turntable match: stretching B"
        );
        let b_stretched = turntable_stretch(&b, stretch_ratio, resample_to)
            .context("turntable stretch B")?;

        // 3. Recompute B's first beat on the stretched version.
        let spec_b2 = stft_mono(&b_stretched, StftParams::MIR_STANDARD)
            .map_err(|e| anyhow::anyhow!("stft B2: {e}"))?;
        let env_b2 = onset_envelope(&spec_b2, OnsetParams::default());
        let grid_b2 = beat_grid_from_tempo(&env_b2, spec_b2.frame_duration(), tempo_a.bpm);
        let b_first_beat = grid_b2.beats.first().copied().unwrap_or(0.0);

        // 4. Pick A's fade start on its beat grid (prefers downbeats near end).
        let overlap_secs = (bars * 4) as f32 * 60.0 / tempo_a.bpm;
        let a_fade_start =
            pick_a_fade_start(&grid_a.beats, a.duration_secs() as f32, overlap_secs);

        info!(
            bars,
            overlap_secs,
            a_fade_start,
            b_first_beat,
            "beat-aligned crossfade"
        );
        beat_aligned_crossfade(
            &a,
            &b_stretched,
            bars,
            tempo_a.bpm,
            a_fade_start,
            b_first_beat,
            curve,
        )
        .context("beat-aligned crossfade")?
    } else {
        info!(
            a_secs = a.duration_secs(),
            b_secs = b.duration_secs(),
            fade_secs,
            "time-based crossfade"
        );
        build_crossfade_mix(&a, &b, fade_secs, curve).context("building crossfade mix")?
    };

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
