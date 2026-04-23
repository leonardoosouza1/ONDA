#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["yt-dlp>=2025.1.1"]
# ///
"""YouTube search via yt-dlp — emits JSON hits to stdout.

Usage:
    uv run tools/yt_search.py --query "electronic house 130 bpm mix" --limit 10
    uv run tools/yt_search.py --query "dnb mix no copyright" --limit 20 --min-sec 120 --max-sec 900

Filters out extremes (too short = shorts, too long = full 3-hour mixes that'd
blow the 30GB budget).
"""
from __future__ import annotations

import argparse
import json
import sys

import yt_dlp  # type: ignore


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--min-sec", type=int, default=60, help="Skip clips shorter than this")
    p.add_argument("--max-sec", type=int, default=600, help="Skip clips longer than this")
    args = p.parse_args()

    q = args.query
    if q.startswith("http://") or q.startswith("https://"):
        # Channel/playlist URL — yt-dlp flat-extracts all entries.
        search_url = q
        playlist_end = args.limit * 3
    else:
        overrequest = max(args.limit * 5, 50)
        search_url = f"ytsearch{overrequest}:{q}"
        playlist_end = overrequest

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": "in_playlist",
        "noprogress": True,
        "skip_download": True,
        "playlistend": playlist_end,
    }

    kept = 0
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_url, download=False)
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}), file=sys.stderr)
        return 1

    entries = (info or {}).get("entries") or []
    for e in entries:
        if kept >= args.limit:
            break
        if not e:
            continue
        dur = e.get("duration") or 0
        if dur and (dur < args.min_sec or dur > args.max_sec):
            continue
        vid = e.get("id") or e.get("url")
        if not vid:
            continue
        url = f"https://www.youtube.com/watch?v={vid}" if not str(vid).startswith("http") else vid
        out = {
            "url": url,
            "id": e.get("id"),
            "title": e.get("title"),
            "channel": e.get("channel") or e.get("uploader"),
            "duration": dur,
        }
        print(json.dumps(out))
        kept += 1

    print(json.dumps({"ok": True, "count": kept}), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
