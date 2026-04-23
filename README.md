# ONDA — Organic Neural DJ Agent

100% Rust AI DJ copilot. Stem-aware mixing, continuous-time vibe sensing via
liquid neural networks, Kuramoto beat-sync, NCP transition planning.

See [docs/superpowers/specs/2026-04-23-onda-design.md](docs/superpowers/specs/2026-04-23-onda-design.md).

## Quick start

```bash
# Build
cargo build --release

# Crossfade two tracks (MVP Phase 0 — no AI yet, just pipeline validation)
cargo run --release --bin onda -- crossfade --in-a track_a.mp3 --in-b track_b.mp3 --out mix.wav --fade-secs 12
```

## Workspace

| Crate | Status | Responsibility |
|---|---|---|
| `onda-core` | 🟡 scaffold | Shared types |
| `onda-audio` | 🟡 scaffold | Decode/encode, resample |
| `onda-dsp` | 🟡 scaffold | FFT, crossfade curves, filters |
| `onda-cli` | 🟡 scaffold | Main binary |
| `onda-features` | ⬜ TODO | Beat/key/tempo/CLAP (tract ONNX) |
| `onda-stems` | ⬜ TODO | HTDemucs ONNX inference |
| `onda-store` | ⬜ TODO | LRU feature cache |
| `onda-download` | ⬜ TODO | yt-dlp subprocess wrapper |
| `onda-brain-fast` | ⬜ TODO | LNN Vibe-Sensor + NCP Planner (Burn) |
| `onda-kuramoto` | ⬜ TODO | Coupled oscillator beat-sync |
| `onda-brain-deep` | ⬜ TODO | Qwen LLM strategic planning |
| `onda-render` | ⬜ TODO | Mixdown renderer |
| `onda-api` | ⬜ TODO | Axum HTTP/WS + Icecast stream |
| `onda-bot` | ⬜ TODO | Telegram remote control |
