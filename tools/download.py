#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["yt-dlp>=2025.1.1"]
# ///
"""ONDA download helper. Invoked by the Rust `onda-download` crate.

Usage:
    uv run tools/download.py <url> <out_dir> [--format mp3|bestaudio]

Emits one JSON line to stdout on success:
    {"ok": true, "path": "...", "title": "...", "duration": 123.4, "uploader": "..."}

Emits a JSON line on error and exits 1:
    {"ok": false, "error": "..."}

yt-dlp supports ~1000 sources (YouTube, SoundCloud, Bandcamp, archive.org, direct URLs).
We prefer m4a/mp3 audio-only to keep disk usage low (~30GB budget for ONDA).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yt_dlp  # type: ignore


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="ONDA download helper")
    parser.add_argument("url", help="Source URL (YouTube, SoundCloud, direct MP3, etc.)")
    parser.add_argument("out_dir", help="Output directory (created if missing)")
    parser.add_argument(
        "--format",
        default="mp3",
        choices=["mp3", "bestaudio", "m4a"],
        help="Audio format preference (default: mp3)",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=900,
        help="Skip tracks longer than N seconds (default: 900 = 15min)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts: dict = {
        "outtmpl": str(out_dir / "%(id)s-%(title).80B.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "restrictfilenames": True,
    }

    if args.format == "mp3":
        ydl_opts["format"] = "bestaudio/best"
        ydl_opts["postprocessors"] = [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ]
    elif args.format == "m4a":
        ydl_opts["format"] = "bestaudio[ext=m4a]/bestaudio"
    else:
        ydl_opts["format"] = "bestaudio/best"

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(args.url, download=False)
            if info is None:
                emit({"ok": False, "error": "extract_info returned None"})
                return 1

            duration = info.get("duration") or 0
            if duration and duration > args.max_duration:
                emit({
                    "ok": False,
                    "error": f"duration {duration}s exceeds max {args.max_duration}s",
                })
                return 1

            ydl.download([args.url])

            final_path = ydl.prepare_filename(info)
            if args.format == "mp3":
                final_path = str(Path(final_path).with_suffix(".mp3"))

            if not Path(final_path).exists():
                candidates = list(out_dir.glob(f"{info.get('id', '')}*"))
                if candidates:
                    final_path = str(candidates[0])

            emit({
                "ok": True,
                "path": str(Path(final_path).resolve()),
                "title": info.get("title"),
                "duration": duration,
                "uploader": info.get("uploader"),
                "extractor": info.get("extractor"),
                "webpage_url": info.get("webpage_url"),
            })
            return 0
    except Exception as e:
        emit({"ok": False, "error": f"{type(e).__name__}: {e}"})
        return 1


if __name__ == "__main__":
    sys.exit(main())
