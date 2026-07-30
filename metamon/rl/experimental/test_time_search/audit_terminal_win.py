"""Human-readable per-root audit for the Phase A terminal-win benchmark.

The expert's Phase A "minimum deliverable" includes a **manual audit of 20-30
branched roots**: for a sampled set of roots, render the turn, legal actions,
base probabilities, shaped-Q (root critic / D=0 K=ref / D=1) with SEM, terminal
win probability with SEM, each selector's pick + terminal-win regret, and whether
the shaped-search argmax *decreased* terminal win vs the actor (a "catastrophic"
decision) or *recovered* a better action than the actor (a "recovered" decision).

This is skill §33 "Diagnostics for action changes" applied to the terminal-win
ground truth: aggregate metrics hide systematic Gen1 errors that a per-root view
reveals (e.g. does shaped Q systematically mis-rank switches vs attacks at
imminent-KO roots?).

Usage::

    uv run python -m metamon.rl.experimental.test_time_search.audit_terminal_win \\
        --input_dir /tmp/tts_phaseA_run --n_audit 25 --output audit.md
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional, Tuple

import numpy as np

from .terminal_win import TerminalWinRootRecord, load_records  # type: ignore[attr-defined]


def _fmt(x, prec: int = 3) -> str:
    try:
        v = float(x)
    except Exception:
        return "nan"
    if np.isnan(v) or np.isinf(v):
        return "nan"
    return f"{v:.{prec}f}"


def _action_label(idx: int, legal: List[int]) -> str:
    """Compact label: the agent-action index in the full action space."""
    return f"a{int(legal[idx])}"


def _root_summary(r: TerminalWinRootRecord) -> Tuple[str, str, str]:
    """Render one root as (header, table, verdict) Markdown."""
    legal = r.legal_actions
    A = r.n_legal
    tw = np.asarray(r.terminal_win, dtype=np.float64)
    tw_sem = np.asarray(r.terminal_win_sem, dtype=np.float64)
    rc = np.asarray(r.root_critic_q, dtype=np.float64)
    d0 = np.asarray(r.d0_q, dtype=np.float64)
    d0_sem = np.asarray(r.d0_q_sem, dtype=np.float64)
    d1 = np.asarray(r.d1_q, dtype=np.float64) if r.d1_q is not None else None
    bp = np.asarray(r.base_probs, dtype=np.float64)

    best = int(np.nanargmax(tw))
    actor = int(np.nanargmax(bp))
    rc_pick = int(np.nanargmax(rc))
    d0_pick = int(np.nanargmax(d0))
    d1_pick = int(np.nanargmax(d1)) if d1 is not None else None

    def reg(idx):
        if np.isnan(tw[idx]):
            return float("nan")
        return float(tw[best] - tw[idx])

    header = (
        f"### {r.root_id}  —  phase={r.phase_band}  request={r.request_kind}  "
        f"tactical={r.tactical_category}  n_legal={A}  G={r.G}  "
        f"mean_steps={_fmt(r.mean_steps_to_terminal, 1)}"
    )
    lines = [
        "| action | base_p | root_critic_Q | D0_Q(Kref)±SEM | D1_Q | term_win±SEM | regret |",
        "|---|---|---|---|---|---|---|",
    ]
    for i in range(A):
        d1c = _fmt(d1[i], 0) if d1 is not None else "—"
        lines.append(
            f"| {_action_label(i, legal)} | {_fmt(bp[i], 3)} | {_fmt(rc[i], 0)} | "
            f"{_fmt(d0[i], 0)}±{_fmt(d0_sem[i], 0)} | {d1c} | "
            f"{_fmt(tw[i], 3)}±{_fmt(tw_sem[i], 4)} | {_fmt(reg(i), 3)} |"
        )
    table = "\n".join(lines)

    # selector verdicts
    sel = [
        ("actor", actor),
        ("root_critic", rc_pick),
        ("D0_Kref", d0_pick),
    ]
    if d1_pick is not None:
        sel.append(("D1", d1_pick))
    vlines = []
    actor_win = float(tw[actor]) if not np.isnan(tw[actor]) else float("nan")
    for name, idx in sel:
        w = float(tw[idx]) if not np.isnan(tw[idx]) else float("nan")
        tag = ""
        if idx == best:
            tag = " ← terminal-best"
        elif np.isfinite(w) and np.isfinite(actor_win) and name != "actor":
            if w < actor_win - 1e-9:
                tag = " ⚠ DECREASE vs actor"
            elif w > actor_win + 1e-9:
                tag = " ✓ recovered vs actor"
        vlines.append(
            f"- {name} → {_action_label(idx, legal)} (win={_fmt(w)}, regret={_fmt(reg(idx))}){tag}"
        )
    vlines.append(
        f"- terminal-best → {_action_label(best, legal)} (win={_fmt(tw[best])})"
    )
    verdict = "\n".join(vlines)
    return header, table, verdict


def audit_report(records: List[TerminalWinRootRecord], n_audit: int = 25) -> str:
    """Render a Markdown audit of up to ``n_audit`` roots, prioritizing the
    most diagnostic roots (catastrophic + recovered + highest-regret)."""
    if not records:
        return "# Terminal-win root audit\n\n(no roots)\n"

    usable = []
    for r in records:
        tw = np.asarray(r.terminal_win, dtype=np.float64)
        if r.n_legal < 2 or np.all(np.isnan(tw)):
            continue
        bp = np.asarray(r.base_probs, dtype=np.float64)
        d0 = np.asarray(r.d0_q, dtype=np.float64)
        actor = int(np.nanargmax(bp))
        d0_pick = int(np.nanargmax(d0))
        best = int(np.nanargmax(tw))
        actor_win = float(tw[actor])
        d0_win = float(tw[d0_pick])
        catastrophic = (
            np.isfinite(d0_win) and np.isfinite(actor_win) and d0_win < actor_win - 1e-9
        )
        recovered = (
            np.isfinite(d0_win) and np.isfinite(actor_win) and d0_win > actor_win + 1e-9
        )
        actor_regret = float(tw[best] - tw[actor])
        # priority: catastrophic first, then recovered, then high actor-regret
        priority = (0 if catastrophic else 1 if recovered else 2, -actor_regret)
        usable.append((priority, r))

    usable.sort(key=lambda x: x[0])
    sampled = [r for _, r in usable[:n_audit]]

    out = [
        f"# Terminal-win root audit — {len(sampled)} of {len(records)} roots",
        "",
        "Prioritized: catastrophic (D0 argmax decreases terminal win vs actor) "
        "→ recovered (D0 beats actor) → highest actor-regret.",
        "",
        "⚠ = D0 shaped-Q argmax picks an action that wins LESS than the actor. "
        "✓ = D0 beats the actor. ← = the terminal-win-best action.",
        "",
    ]
    for r in sampled:
        header, table, verdict = _root_summary(r)
        out += [header, "", table, "", verdict, ""]
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Human-readable per-root audit of a Phase A terminal-win run"
    )
    p.add_argument("--input_dir", required=True)
    p.add_argument("--n_audit", type=int, default=25)
    p.add_argument(
        "--output",
        default=None,
        help="write Markdown here (default: <input_dir>/audit.md",
    )
    args = p.parse_args()

    records = load_records(os.path.join(args.input_dir, "terminal_win_roots.jsonl"))
    print(f"loaded {len(records)} roots", flush=True)
    report = audit_report(records, n_audit=args.n_audit)
    out_path = args.output or os.path.join(args.input_dir, "audit.md")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
