# tests_out/ — áudio de dev

Tudo aqui é ignorado pelo git (ver `/.gitignore`).

```
tests_out/
├── tracks/           # fontes de música (input)
│   ├── ncs/          # YouTube / NoCopyrightSounds — m4a
│   └── netlabels/    # archive.org netlabels CC — mp3
└── mixes/            # mixes renderizados (output)
    ├── v4_spectral_pair.wav    # par escolhido por centroide, equal-power xfade
    └── v5_bass_swap.wav        # idem + bass-swap no crossfade (atual)
```

## Como regenerar

```bash
# baixar mais faixas (NCS via YouTube):
./target/debug/onda yt-fetch \
  --query "https://www.youtube.com/@NoCopyrightSounds/videos" \
  --count 10 --min-duration 120 --max-duration 400 \
  --out-dir tests_out/tracks/ncs

# baixar de archive.org:
./target/debug/onda analyze --input tests_out/tracks/netlabels/<file>

# renderizar um mix:
./target/debug/onda plan-mix \
  --dir tests_out/tracks/netlabels \
  --out tests_out/mixes/v6_next.wav \
  --bars 16 --max-bpm-gap-pct 8

# diagnosticar um mix:
./target/debug/onda diagnose \
  --input tests_out/mixes/v5_bass_swap.wav \
  --fade-start 42.91 --overlap 29.72
```

## Convenção de versão dos mixes

`v{N}_{mudança-principal}.wav` — nunca sobrescrever um mix antigo; criar
versão nova. Deletar versões antigas só quando houver baseline melhor
documentada no `MEMORY.md` do projeto.
