# ONDA — Organic Neural DJ Agent

**Data:** 2026-04-23
**Autor:** Leonardo + Claude (brainstorm pair)
**Status:** Aprovado — pronto pra plano de implementação

## 1. Visão em 30 segundos

ONDA é um **DJ senciente em Rust puro** que escuta, sente e mistura música.
Entrega uma playlist → ele separa stems, detecta beat/key/vibe, planeja transições
e renderiza um mix coeso (offline) ou toca ao vivo (streaming) respondendo a intervenções
em linguagem natural ("mais pesado", "puxa o bass"). O coração é um **Vibe-Sensor LNN
(rede neural líquida contínua-tempo)** que evolui um estado latente de 8 dimensões
enquanto a música toca — essa é a contribuição original e inexplorada na literatura.

Três camadas (D da brainstorm): **A)** DJ (foco inicial), **B)** Remix, **C)** Criação.
Todas compartilham o mesmo tronco de análise. MVP = A, offline.

## 2. Filosofia e fit no ecossistema IARA

ONDA é coerente com a tese central do ecossistema: **inteligência como organismo
buscando homeostase**. Aqui a homeostase é musical — o organismo sente a "temperatura
vibracional" da pista e age pra manter/escalar/relaxar a energia. Herda padrões
diretos de outros projetos:

- **Dual Brain** (jarvis Architect+Explorer, saci-trader Trinity): Fast Brain LNN @ 50Hz
  decide batida-a-batida; Deep Brain LLM local (Qwen via llm-lab) faz curadoria estratégica
  e interpreta comando natural.
- **Kuramoto oscillators** (Universe, jarvis): rede de osciladores acoplados pra
  beat-matching/phase-lock — fit matemático perfeito, já implementado no ecossistema.
- **CfC/NCP** (saci-trader, iara-brain, enchente-zero): Vibe-Sensor é CfC, Transition-Planner
  é NCP com wiring esparso sensory→inter→command→motor.
- **KAN axiom filters** (Universe): safety gates que impedem transições clipping/dissonantes.
- **Fast-then-Deep** (IARA blueprint): Fase 3 reflexo + Fase 4 raciocínio.

## 3. Arquitetura

```
                      ┌─────────────────────────────────────────────┐
                      │                 ONDA core                   │
                      │                                             │
  input playlist ───▶ │  ┌────────────┐    ┌────────────────────┐   │
  (URLs/MP3s)         │  │ Download & │───▶│  Feature Extractor │   │
                      │  │  Decode    │    │  (stems/beat/key)  │   │
                      │  └────────────┘    └──────────┬─────────┘   │
                      │                               │             │
                      │                               ▼             │
                      │                 ┌──────────────────────┐    │
                      │                 │  Feature Store (LRU) │    │
                      │                 └──────────┬───────────┘    │
                      │                            │                │
                      │         ┌──────────────────┴────────────┐   │
                      │         ▼                               ▼   │
                      │  ┌──────────────┐              ┌───────────────┐
                      │  │ Deep Brain   │  plan ◀────▶ │ Fast Brain    │
                      │  │ (LLM Qwen)   │              │ (LNN Vibe-    │
                      │  │ strategic    │              │  Sensor + NCP │
                      │  │ & NL cmds    │              │  Planner)     │
                      │  └──────────────┘              └──────┬────────┘
                      │                                       │        │
                      │                                       ▼        │
                      │                              ┌────────────────┐│
                      │                              │ Kuramoto       ││
                      │                              │ Beat-Sync      ││
                      │                              └───────┬────────┘│
                      │                                      │         │
                      │                                      ▼         │
                      │                              ┌───────────────┐ │
                      │                              │ Mix Renderer  │ │
                      │                              │ (DSP/crossfade│ │
                      │                              │  filter, EQ)  │ │
                      │                              └───────┬───────┘ │
                      │                                      │         │
                      └──────────────────────────────────────┼─────────┘
                                                             │
                             offline: ▶ WAV + vibe-trajectory PNG
                             online: ▶ Icecast/WebRTC stream + Telegram bot
```

## 4. Componentes (crates do workspace)

Cargo workspace em `/home/leonardo/projects/LLM/ONDA`. Cada crate tem responsabilidade
única e interface testável.

| Crate | Responsabilidade | Dependências chave |
|---|---|---|
| `onda-core` | Tipos compartilhados: `Track`, `Features`, `VibeState`, `TransitionPlan`, `MixSession`. | serde, thiserror |
| `onda-audio` | Decode/encode (symphonia/hound), resample (rubato), playback (cpal). | symphonia, rubato, cpal |
| `onda-dsp` | FFT, mel-spec, filtros, EQ, crossfade curves, time-stretch, pitch-shift. | rustfft, realfft, fundsp |
| `onda-features` | Beat/tempo/downbeat, key, CLAP embedding via ONNX, energy curves. | tract (ou ort), ndarray |
| `onda-stems` | HTDemucs ONNX inference. 4 stems: drums/bass/vocals/other. | tract |
| `onda-store` | Feature store LRU em disco. Tensores comprimidos flatbuffers. **Nunca guarda áudio bruto long-term.** | flatbuffers, zstd |
| `onda-download` | yt-dlp binário via subprocess. Fallback rustube. | tokio::process |
| `onda-brain-fast` | LNN Vibe-Sensor (CfC) + NCP Transition-Planner. Treino via Burn. | burn, burn-wgpu, burn-rocm |
| `onda-kuramoto` | Rede de osciladores acoplados pra beat phase-lock. | nalgebra |
| `onda-brain-deep` | Cliente pra llm-lab engine local (Qwen). Strategic planning, NL command parsing. | reqwest (llm-lab API) |
| `onda-render` | Mix Renderer: aplica plano de transição nas stems, renderiza mixdown. | onda-dsp |
| `onda-api` | Axum HTTP/WebSocket API pra controle remoto + Icecast server. | axum, tokio-tungstenite |
| `onda-bot` | Telegram bot (teloxide) pra controle do celular. | teloxide |
| `onda-cli` | Binário CLI principal: `onda mix`, `onda train`, `onda serve`. | clap |

**Limite de tamanho por arquivo:** mirar 300-500 linhas. Se crescer, refatorar.

## 5. Estado Vibe (8 dimensões)

O Vibe-Sensor LNN emite um vetor contínuo R⁸ atualizado a 50-100Hz:

| Dim | Nome | Range | Semântica |
|---|---|---|---|
| 0 | energy | [0,1] | Intensidade total (RMS + perceptual loudness) |
| 1 | darkness | [0,1] | Minor/dissonant/low-brightness bias |
| 2 | density | [0,1] | Quão cheio o espectro (contagem de eventos por compasso) |
| 3 | tension | [0,1] | Build-up pressure (crescendos, filters opening) |
| 4 | groove_lock | [0,1] | Quão "in-the-pocket" está o ritmo |
| 5 | bass_pressure | [0,1] | Peso do sub/bass |
| 6 | brightness | [0,1] | Centroid espectral normalizado |
| 7 | novelty | [0,1] | Rate-of-change das features recentes |

"VIBE mode" = modo em que o Deep Brain **não impõe um target curve** — só observa a trajetória
do sensor e deixa o Fast Brain orbitar livremente. O modelo "sente" sem ser comandado.
Intervenção humana = override de um target curve sobre uma ou mais dimensões.

## 6. Fluxo de dados (MVP offline)

1. **CLI:** `onda mix --playlist urls.txt --out set.wav --graph vibe.png`
2. **Download:** yt-dlp binário baixa em cache temp (`/tmp/onda-dl/`).
3. **Decode:** symphonia → f32 stereo @ 44.1kHz.
4. **Stem separation:** HTDemucs ONNX → 4 stems f32.
5. **Feature extraction:** beats (BeatNet ONNX), tempo, downbeat, key (Basic-Pitch ONNX ou
   rule-based Essentia-port), CLAP embedding (512-d), mel-spec (128 bins 10ms hop),
   vibe snapshot @ beat granularity.
6. **Feature store:** salva tensores comprimidos em `~/.cache/onda/features/<hash>.fb.zst`.
   Deleta áudio bruto após extração.
7. **Deep Brain (Qwen local via llm-lab):** ordena playlist por compatibilidade
   (BPM gradient, Camelot wheel para keys, vibe progression desejada se informada).
8. **Fast Brain (NCP planner):** para cada par (A→B) consecutivo, decide tipo de
   transição (cut, filter-fade, echo-tail, double-drop, EQ-swap) + length em bars
   + stem-routing (kick de B entra antes de vocals de A saírem, etc.).
9. **Kuramoto Beat-Sync:** ajusta phase das stems de B pra alinhar com o grid de A.
10. **Mix Renderer:** aplica o plano, gera mixdown WAV 44.1kHz/16-bit.
11. **Vibe trajectory plotter:** desenha PNG da trajetória 8D projetada em 2D (t-SNE ou PCA)
    + curvas individuais por dimensão.

## 7. Estratégia de treino

### 7.1 Modelos pretrained (zero training custo nosso)
- **HTDemucs v4** (htdemucs_ft, ~320MB ONNX) — estado da arte.
- **BeatNet** (beat/downbeat, ~20MB ONNX).
- **CLAP** (laion/CLAP-general, ~500MB ONNX) — audio-text embedding.

Todos rodam via `tract` (Rust puro, zero C++). Carregamos safetensors/ONNX, rodamos
com kernels WGSL via WGPU ou HIP direto quando perf importar.

### 7.2 Modelos treinados por nós (em Burn, Rust puro)

| Modelo | Tamanho | Dataset | Método | VRAM |
|---|---|---|---|---|
| Vibe-Sensor CfC | ~200K params | features de ~5k tracks eletrônicas | Contrastive Predictive Coding + self-supervised (prediz próximo segmento) | <2GB |
| NCP Transition-Planner | ~50K params | self-play (gera mix, score via proxy metrics) + destilação de mixes humanos | Reinforcement-free bootstrapping | <1GB |
| Kuramoto coupling weights | ~1K params | SGD supervisionado em pares beat-aligned | MSE on phase-lock error | <100MB |

Orçamento total de treino: **~12-48h na 6750 XT**. Dataset fica em features (30GB dá e sobra).

### 7.3 Aquisição de dados com 30GB
Política **stream-and-distill**:
1. yt-dlp baixa em `/tmp` (delete-on-close).
2. Extrai features e stems comprimidos (stems só se track vai ser tocada).
3. Features salvas (~3MB/track) em `~/.cache/onda/features/`.
4. Áudio original deletado.
5. LRU cache de áudio renderizado para playback (5GB ceiling).

**Alvo Fase 1:** 500-1000 tracks eletrônicas de Jamendo (CC-license, API direta) +
playlists curadas via yt-dlp (uso pessoal, lícito sob fair-use brasileiro pra testes).
Breakdown 30GB: 10GB features (~3k tracks), 10GB stems cache, 5GB modelos, 5GB buffer.

## 8. MVP — o que precisa funcionar primeiro

**Critério de aceite Fase 1 (mixdown offline funcional):**

1. `onda mix --playlist lista.txt --out mix.wav --graph vibe.png` roda sem erro.
2. Input: 5 URLs/arquivos de tracks house/techno (120-135 BPM).
3. Output: `mix.wav` ~20min, stereo 44.1kHz 16-bit, sem clipping.
4. Todas as transições **beat-matched** (phase lock ±10ms).
5. Todas as transições **key-compatible** (Camelot adjacent ou relative minor).
6. `vibe.png` mostra trajetória 8D coerente (não é ruído) — subjetivo mas inspecionável.
7. Nenhum segmento tem gap > 50ms nem clique audível.
8. **Nota importante:** MVP usa **Vibe-Sensor rule-based + planner heurístico** como fallback
   quando os modelos LNN/NCP não estão treinados. Treino vem na Fase 2. Isso garante que
   a **pipeline end-to-end funcione antes de a mágica entrar**.

## 9. Tratamento de erro

Erros são explícitos (`thiserror`). Pontos chave:
- Track fails download → pula, loga, prossegue.
- Stem-sep VRAM OOM → fallback pra CPU (slow but robusto).
- Feature extraction falha → track descartada do set, loga.
- Vibe-Sensor NaN/inf → retorna último estado válido + loga (homeostase = resiliência).
- Kuramoto não converge em N iter → aceita phase atual + loga warning.

Sem retries infinitos, sem silenciar erros.

## 10. Testes

- **Unit tests** por crate — tratamento de casos limite (silêncio, clipping, mono vs stereo).
- **Integration test** fim-a-fim: playlist fixture de 3 tracks CC-license commitadas no repo
  (< 2MB cada). Gera mix, valida métricas (duração, no-clipping, BPM continuity).
- **Golden audio test**: hash SHA256 do mixdown determinístico (seed fixo) — regressão visual.
- **Property tests** (`proptest`): crossfade nunca clipping, beat-match sempre monotônico.

## 11. Streaming remoto (Fase 5)

- Backend Axum roda em casa (Linux, GPU).
- Icecast server embarcado (mountpoint `/onda.ogg`, Opus 128kbps).
- Cliente: qualquer player HTTP (VLC, browser) aponta pro endpoint.
- Controle: Telegram bot (teloxide) + mini web UI (HTMX-style single file, sem build).
- Latência alvo: <3s (Icecast buffer) entre comando e efeito audível.
- Segurança: token bearer pra API; Telegram whitelist de user_id.

## 12. Roadmap

| Fase | Semanas | Entrega |
|---|---|---|
| 0. Scaffolding | 1 | Workspace Cargo, CI, crates vazios, HTDemucs ONNX carregando |
| 1. MVP Offline | 2-3 | `onda mix` rodando end-to-end com heurísticas |
| 2. Vibe-Sensor | 4-5 | CfC treinado em Burn, substitui heurística |
| 3. Kuramoto | 6 | Beat-sync via osciladores acoplados |
| 4. NCP Planner | 7-8 | Transition planning neural (self-play) |
| 5. Streaming | 9-10 | Axum + Icecast + Telegram bot |
| 6. Polish | 11-12 | Adapters LoRA pra outros gêneros, perf tuning |

## 13. Out of scope (YAGNI)

- Não vamos treinar HTDemucs/CLAP do zero (orçamento insuficiente, estado-da-arte
  pretrained já excelente).
- Não vamos fazer UI gráfica no MVP. Terminal + PNG + WAV suficiente.
- Não vamos implementar VST plugin no MVP (Fase C futura).
- Não vamos suportar múltiplos usuários simultâneos no servidor (single-user, home).
- Não vamos persistir playlists em banco de dados (JSON file suficiente).

## 14. Decisões abertas (resolvidas neste doc)

- ✅ Nome: **ONDA** (Organic Neural DJ Agent)
- ✅ Stack: 100% Rust. Burn pra treino, tract pra inferência ONNX, HIP pra kernels custom.
- ✅ yt-dlp binário via subprocess (aceito como exceção pragmática).
- ✅ Primeiro MVP: mixdown offline 5 tracks + gráfico vibe (Fase 1).
- ✅ Gênero inicial: eletrônica (house/techno/DnB/trap). Outros gêneros via LoRA adapters depois.
- ✅ Vibe = vetor 8D contínuo (ver §5).

## 15. Referências técnicas

- HTDemucs v4 (2023) — Rouard et al., "Hybrid Transformers for Music Source Separation"
- Kuramoto model — Universe project crate `kuramoto-core`, lib/LNN-ROUTER.md
- CfC/LNN — saci-trader/fast_brain, enchente-zero/cfc_layer.rs
- NCP — iara-brain/ncp_cell.py → porta pra Burn
- Burn framework — https://burn.dev, ROCm backend confirmado em 2026
- CLAP — LAION CLAP-general, ONNX export disponível
- Camelot Wheel — harmonic mixing standard (https://mixedinkey.com/camelot-wheel/)
