# ONDA: an AI DJ copilot in Rust (early research prototype)

ONDA (*Organic Neural DJ Agent*) is a pure-Rust audio workspace. It decodes and analyses tracks, plans
beat-aligned mixes, and composes and synthesises electronic music procedurally. One synthesis path runs as a
HIP kernel on an AMD GPU.

The long-term design (a liquid-network "vibe sensor", stem separation, a learned transition planner) is in
[`docs/superpowers/specs/2026-04-23-onda-design.md`](docs/superpowers/specs/2026-04-23-onda-design.md).
**Most of that design is not built yet.** This README covers only what exists and has been measured.

## Highlights

- **GPU synthesis at 370× real time.** A harmonic-synthesis HIP kernel (gfx1031) renders 16.4 s of audio in
  44 ms. A pad tuned to the same octave as a reference track lowered the spectral (Bark-band) MSE from 0.040
  (modal synthesis) to 0.025.
- **Profile "training" end to end.** Features were extracted from 36/36 downloaded tracks and fitted into a JSON
  profile (histograms + k-means rhythm archetypes); new music is then sampled from it. Sampled tempos follow
  the dataset distribution (129 BPM is the mode at 38.9 %). A track rendered with seed 42 was measured back by
  ONDA's own tempo detector at 129.20 BPM (confidence 0.947) against a sampled 128.5 BPM.
- **Day-0 research gates.** The gammatone modulation-spectrum features passed (36 tracks, 73.3k steps). An echo
  state network trained in 12.4 s with an overfit ratio of 1.47 (gate < 1.5). Kuramoto beat-sync oscillators
  reached phase coherence r = 0.72 at coupling K = 1.2 and r = 0.06 at K = 0.8.
- **Rule-based "Symphony" composer, v1.** Maestro → reactive bus (sidechain, risers) → instrument layers. It
  rendered a ~121 s track with 74 tests passing, and it was the first output the listener judged as good.
- **Stem dissection without ML.** Kick energy is 13× higher in its bands (0.415 vs 0.032) and bass 8× higher
  (0.322 vs 0.039), so both separate cleanly from band energy alone.

## How it works

```
audio file ─► onda-audio (symphonia decode, rubato resample)
           ─► onda-features (spectrum, onsets, tempo, beats, sections, chroma)
           ─► onda-cli plan-mix: pick a BPM-compatible pair → onda-dsp beat-aligned crossfade → WAV

dataset ─► onda-train fit (features → histograms + k-means profile.json)
        ─► onda-train gen / onda-compose (rhythm, harmony, Markov melody, synth) → WAV

onda-research: modspec · ESN · Kuramoto · dissect · HIP harmonic synth · compose-symphony
```

| Crate | What it does |
|---|---|
| `onda-core` | shared types |
| `onda-audio` | decode (mp3/m4a/flac/… via symphonia), WAV encode, resample |
| `onda-dsp` | crossfade curves (linear / equal-power), envelopes, filters, time-stretch |
| `onda-features` | classical MIR: spectrum, onsets, tempo, beat tracking, sections, chroma. No ML. |
| `onda-download` | wrapper around `yt-dlp` / archive.org through the scripts in `tools/` |
| `onda-compose` | procedural composition: rhythm, harmony, melody, synths *(work in progress)* |
| `onda-train` | dataset → profile fit / inspect / generate *(work in progress)* |
| `onda-research` | experiments: modspec, ESN, Kuramoto, dissection, HIP synth, Symphony |
| `onda-cli` | the `onda` binary |

## Results

| Experiment | Number | Verdict |
|---|---|---|
| HIP harmonic synth (gfx1031) | 44 ms for 16.4 s of audio (370× real time) | works |
| Spectral match to a reference | Bark MSE 0.040 → 0.025 | better, but the listener heard "few elements, no sense" |
| Profile fit + generation | 36/36 tracks; seed 42 rendered at 129.20 BPM (conf 0.947), sampled 128.5 | pipeline works; music "not good yet" |
| ESN on modspec | overfit ratio 1.47, 12.4 s training | passed the gate |
| Kuramoto beat-sync | r = 0.72 at K = 1.2; r = 0.06 at K = 0.8 | mixed (see below) |
| Symphony v1 | ~121 s, 74 tests | first track judged good |

The quality verdicts are listening tests by the author, not a formal metric.

## What didn't work (and why)

- **Symphony v3–v5 got worse, not better.** v3 sounded like "8-bit game music", v4 was "out of tune, odd and
  loud FX", and v5 had "the face of badly made AI music". Mechanism: hard-coded rules plus additive synthesis
  with no learning. Vibrato and detune created dissonance, and FX fired on every transition. Honest summary:
  this is rule-based composition, not AI.
- **Stem-to-pattern composition produced dense mush.** Kick and bass separate, but lead and pad gave false
  positives, and the extracted patterns were far too dense (kick on 100 of 128 steps). The result had "no
  coherence; the kick sounds like a pad".
- **Kuramoto metric was wrong.** A Pearson-based coherence score is the wrong measure for a 1:4 frequency ratio,
  so the mixed Kuramoto result reflects the metric as much as the oscillators.
- **Small, procedural dataset.** The profile was fitted on 36 tracks with procedural timbres and a fixed
  harmony, which is too little to learn style.

## Hardware & stack

- Rust 2024 edition (rust-version 1.90), workspace of 9 crates. symphonia, rubato, rustfft/realfft, rayon, clap,
  tokio.
- GPU: AMD RX 6750 XT (gfx1031, RDNA2) with ROCm. `onda-research/build.rs` compiles the HIP kernel with `hipcc`
  if `ROCM_PATH` (default `/opt/rocm`) is available. Otherwise it falls back silently to a Rayon CPU path.
- Optional: `yt-dlp` / `uv` for the download helpers in `tools/`.

## Build & run

```bash
cargo build --release

# pipeline smoke: equal-power crossfade of two files (or --bars N for a beat-aligned fade)
cargo run --release --bin onda -- crossfade --in-a a.mp3 --in-b b.mp3 --out mix.wav --fade-secs 12

# tempo / beat summary of one file
cargo run --release --bin onda -- analyze --input track.mp3

# scan a folder, pick the best BPM-compatible pair, render a 16-bar beat-aligned mix
cargo run --release --bin onda -- plan-mix --dir tracks/ --out mix.wav --bars 16

# research: Kuramoto → harmonic synth (HIP if available) and the Symphony composer
cargo run --release --bin onda-research -- synth --k 0.8 --out synth.wav
cargo run --release --bin onda-research -- compose-symphony --bpm 128 --out symphony.wav

# fit a style profile from a folder of tracks, then generate from it
cargo run --release --bin onda-train -- fit --dataset tracks/ --out profile.json
cargo run --release --bin onda-train -- gen --profile profile.json --out demo.wav --bars 32 --seed 42
```

Audio files (`*.wav`, `*.mp3`, …) and `tests_out/` are git-ignored on purpose. Bring your own tracks, and use
only music you have the rights to.

## Status / roadmap

Early prototype, paused since 2026-04. Built: decoding, classical MIR features, beat-aligned mixing,
procedural composition, the HIP synth, and the research gates. Not built: the learned vibe sensor (CfC/LNN),
stem separation (HTDemucs), a learned transition planner, the HTTP/stream API, and remote control.

License: TBD. `Cargo.toml` declares MIT, but there is no LICENSE file yet.
