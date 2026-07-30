"""Recovery tool for the Phase A terminal-win benchmark.

The full G=128 / 80-root run can take many hours and is killable by a signal
(e.g. the Node host hitting its V8 heap limit) with no Python traceback. The
benchmark streams each root to ``terminal_win_roots.jsonl`` as it is produced
(flushed per record), so the first N roots always survive a crash. This script
re-aggregates the streamed JSONL into the summary + Gate A verdict + Markdown
report, so a partial run is never lost.

Usage::

    uv run python -m metamon.rl.experimental.test_time_search.recover_terminal_win \\
        --input_dir /tmp/tts_phaseA_run --derived_ks 4 16 64

Reads ``<input_dir>/terminal_win_roots.jsonl``, writes
``<input_dir>/terminal_win_summary.json`` + ``terminal_win_REPORT.md`` +
``run_manifest.json`` (if absent).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import List

from .terminal_win import (
    TerminalWinRootRecord,
    aggregate_terminal_win,
    terminal_win_gate,
    write_results,
)


def load_records(path: str) -> List[TerminalWinRootRecord]:
    records: List[TerminalWinRootRecord] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            records.append(TerminalWinRootRecord(**d))
    return records


def main() -> None:
    p = argparse.ArgumentParser(
        description="Re-aggregate a streamed Phase A terminal_win_roots.jsonl"
    )
    p.add_argument("--input_dir", required=True)
    p.add_argument("--derived_ks", type=int, nargs="+", default=[4, 16, 64])
    args = p.parse_args()

    roots_path = os.path.join(args.input_dir, "terminal_win_roots.jsonl")
    records = load_records(roots_path)
    print(f"loaded {len(records)} roots from {roots_path}", flush=True)
    if not records:
        print("no records; nothing to do")
        return

    derived_ks = sorted(k for k in args.derived_ks if k <= records[0].G)
    if not derived_ks:
        derived_ks = [4, 16, 64]

    summary = aggregate_terminal_win(records, derived_ks)
    gate = terminal_win_gate(summary, derived_ks)

    # reuse the streamed run_manifest if present, else a minimal stub
    rm_path = os.path.join(args.input_dir, "run_manifest.json")
    if os.path.exists(rm_path):
        with open(rm_path) as f:
            rm = json.load(f)
    else:
        rm = {
            "recovered": True,
            "n_roots": len(records),
            "derived_ks": derived_ks,
            "recovered_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    paths = write_results(records, [], summary, gate, rm, args.input_dir, derived_ks)
    print(
        json.dumps(
            {
                "verdict": gate["verdict"],
                "passed": gate["passed"],
                "total": gate["total"],
                "n_roots": len(records),
                "spearman_d0_k_ref": summary.get("spearman_shaped_vs_terminal", {})
                .get("d0_k_ref", {})
                .get("mean"),
                "actor_regret": summary.get("terminal_win_regret", {})
                .get("actor", {})
                .get("mean"),
                "d0_k_ref_regret": summary.get("terminal_win_regret", {})
                .get("d0_k_ref", {})
                .get("mean"),
                "phase_distribution": summary.get("phase_distribution", {}),
                "paths": paths,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
