#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests>=2.32"]
# ///
"""archive.org CC audio catalog search.

Queries the Internet Archive advanced-search API (no auth) for audio items in
the `netlabels` collection (Creative Commons electronic/indie netlabels).

Output: one JSON object per result to stdout, then a final summary line.

Usage:
    uv run tools/search_archive.py --query electronic --limit 20
    uv run tools/search_archive.py --subject "electronic OR techno OR house" --limit 30
"""
from __future__ import annotations

import argparse
import json
import sys

import requests

ARCHIVE_API = "https://archive.org/advancedsearch.php"


def build_query(subject: str, extra: str | None) -> str:
    parts = [
        "collection:(netlabels)",
        "mediatype:(audio)",
        f"subject:({subject})",
    ]
    if extra:
        parts.append(f"({extra})")
    return " AND ".join(parts)


def search(subject: str, limit: int, extra: str | None) -> list[dict]:
    q = build_query(subject, extra)
    params = {
        "q": q,
        "fl[]": ["identifier", "title", "creator", "year", "subject"],
        "rows": str(limit),
        "page": "1",
        "output": "json",
        "sort[]": "downloads desc",
    }
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            r = requests.get(
                ARCHIVE_API,
                params=params,
                timeout=(10, 90),  # (connect, read)
                headers={"User-Agent": "onda-download/0.1 (pure-rust pipeline)"},
            )
            r.raise_for_status()
            data = r.json()
            return data.get("response", {}).get("docs", [])
        except requests.RequestException as e:
            last_err = e
            print(json.dumps({"retry": attempt + 1, "error": str(e)[:200]}), file=sys.stderr)
    raise last_err if last_err else RuntimeError("search failed")


def main() -> int:
    p = argparse.ArgumentParser(description="archive.org CC audio search")
    p.add_argument("--subject", default="electronic",
                   help="Subject filter (e.g. 'electronic', 'techno OR house')")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--extra", default=None,
                   help="Additional query clause, e.g. 'creator:kahvi'")
    args = p.parse_args()

    try:
        docs = search(args.subject, args.limit, args.extra)
    except requests.RequestException as e:
        print(json.dumps({"ok": False, "error": f"search: {e}"}), file=sys.stderr)
        return 1

    for d in docs:
        ident = d.get("identifier")
        if not ident:
            continue
        url = f"https://archive.org/details/{ident}"
        out = {
            "url": url,
            "identifier": ident,
            "title": d.get("title"),
            "creator": d.get("creator"),
            "year": d.get("year"),
            "subject": d.get("subject"),
        }
        print(json.dumps(out))

    print(json.dumps({"ok": True, "count": len(docs)}), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
