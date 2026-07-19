"""Scrape pokepaste teams linked from the FIRST POST (OP) of Smogon threads.

Only the original post is considered (XenForo delimits posts by
``data-content="post-<id>"``); reply/RMT/changelog links are ignored because
those are not curated current samples and are likely illegal/outdated.

Each pokepaste is downloaded via its ``/raw`` endpoint (clean Showdown export)
and written as ``team_NNNNNN.<format>_team`` into the staging dir. Legality is
NOT checked here -- that is done downstream by the Showdown TeamValidator.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import urllib.request

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
POKEPASTE_RE = re.compile(r"https?://pokepast\.es/([A-Za-z0-9]+)")


def fetch(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def op_pokepaste_ids(thread_url: str) -> list[str]:
    """Unique pokepaste IDs linked from the thread's first post, in order."""
    html = fetch(thread_url)
    parts = html.split('data-content="post-')
    if len(parts) < 2:
        print(f"  WARN: could not locate any post block in {thread_url}")
        return []
    op = parts[1]
    ids: list[str] = []
    seen: set[str] = set()
    for m in POKEPASTE_RE.finditer(op):
        pid = m.group(1)
        if pid not in seen:
            seen.add(pid)
            ids.append(pid)
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", default="gen9ou")
    ap.add_argument("--out", required=True, help="staging dir for .<fmt>_team files")
    ap.add_argument("--threads", nargs="+", required=True, help="thread URLs")
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    suffix = f".{args.format.lower()}_team"

    # Collect OP pokepaste ids across all threads, deduped globally.
    all_ids: list[tuple[str, str]] = []  # (id, source_thread)
    seen: set[str] = set()
    for url in args.threads:
        ids = op_pokepaste_ids(url)
        print(f"OP of {url}\n  -> {len(ids)} unique pokepaste links")
        for pid in ids:
            if pid not in seen:
                seen.add(pid)
                all_ids.append((pid, url))
        time.sleep(args.sleep)

    print(f"\nTotal unique teams to download: {len(all_ids)}")

    manifest = open(os.path.join(args.out, "_manifest.tsv"), "w", encoding="utf-8")
    manifest.write("file\tpokepaste\tsource_thread\n")
    written = 0
    for i, (pid, src) in enumerate(all_ids):
        raw_url = f"https://pokepast.es/{pid}/raw"
        try:
            team = fetch(raw_url)
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {pid}: {e}")
            continue
        # normalize line endings; pokepaste uses trailing 2 spaces per line
        team = team.replace("\r\n", "\n").rstrip() + "\n"
        if not team.strip():
            print(f"  EMPTY {pid}")
            continue
        fname = f"team_{i:06d}{suffix}"
        with open(os.path.join(args.out, fname), "w", encoding="utf-8") as f:
            f.write(team)
        manifest.write(f"{fname}\t{raw_url}\t{src}\n")
        written += 1
        time.sleep(args.sleep)
    manifest.close()
    print(f"\nWrote {written} team files to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
