"""Recompute the Phase 1 analysis from a streamed ``root_results.jsonl``.

The benchmark streams each root record to disk as it is produced (crash
safety), but writes ``summary.json`` / ``comparison.json`` / ``REPORT.md`` only
at the end. If a run is killed mid-way (signal, host stall, timeout), this tool
reloads the streamed roots and regenerates the full analysis (convergence gate
+ estimator head-to-head + search-justification gate + report) from whatever
roots survived.

Usage::

    uv run python -m metamon.rl.experimental.test_time_search.analyze_roots \
        --input_dir /tmp/tts_phase1_full \
        --derived_ks 4 16 64 --depths 0 1

It reads ``<input_dir>/root_results.jsonl`` and overwrites ``summary.json``,
``comparison.json``, ``REPORT.md`` (plus writes ``analysis_recovered.json``).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np

from metamon.rl.experimental.test_time_search.benchmark_roots import (
    RootResultRecord,
    aggregate_convergence,
    go_no_go_assessment,
    estimator_comparison,
    comparison_table,
    search_justification_gate,
    convergence_table,
)


def estimate_global_advantage_scale(
    records: List[RootResultRecord],
    reference_name: str = "d0",
) -> Dict[str, Any]:
    """Compute a frozen global advantage scale from the Phase 1 dev corpus.

    Skill §11: ``A_scaled(root, action) = A_raw(root, action) / global_scale``
    where ``global_scale`` is a robust scale (MAD-based std) of the raw
    per-root advantages, frozen once from the dev corpus so beta has a stable
    global interpretation across roots / runs / candidate counts.

    The advantage for each root is ``A(a) = Q(a) - sum_a' pi(a') Q(a')`` (the
    centered Q under the base policy). Returns the MAD-based robust std
    (1.4826 * MAD) plus recommended betas for a target median KL (skill §11:
    0.01-0.05; KL ~= Var(A_scaled)/(2 beta^2) => beta ~= sqrt(1/(2 KL)) when
    A is standardized to unit variance).
    """
    all_adv: List[float] = []
    for r in records:
        ref_cfg = r.configs.get(reference_name)
        if ref_cfg is None or ref_cfg.get("q_mean") is None:
            continue
        q = np.asarray(ref_cfg["q_mean"], dtype=np.float64)
        bp = np.asarray(r.base_probs, dtype=np.float64)
        if q.size == 0 or bp.size == 0 or q.size != bp.size:
            continue
        mean_q = float((bp * q).sum())
        adv = q - mean_q
        all_adv.extend(float(x) for x in adv)
    if not all_adv:
        return {"n_advantages": 0, "global_scale": None}
    arr = np.asarray(all_adv, dtype=np.float64)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    robust_std = 1.4826 * mad
    plain_std = float(arr.std())
    betas = {f"{kl:.2f}": float(np.sqrt(1.0 / (2.0 * kl))) for kl in (0.01, 0.02, 0.05)}
    return {
        "n_advantages": int(arr.size),
        "n_roots": len(records),
        "reference": reference_name,
        "raw_mean": float(arr.mean()),
        "raw_std": plain_std,
        "raw_median": median,
        "mad": mad,
        "global_scale_robust_std": robust_std,
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max_abs": float(np.abs(arr).max()),
        "recommended_beta_for_target_kl": betas,
        "note": (
            "Set search_value_scale_mode=global_standardized and "
            "search_global_advantage_scale=global_scale_robust_std. Pick beta "
            "from recommended_beta_for_target_kl; verify the actual median KL "
            "from Phase 2 search root logs and adjust."
        ),
    }


def _load_records(path: str) -> List[RootResultRecord]:
    recs: List[RootResultRecord] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # RootResultRecord is a dataclass whose to_dict() is __dict__; the
            # streamed JSON has exactly the dataclass fields. Reconstruct.
            recs.append(RootResultRecord(**d))
    return recs


def main() -> None:
    p = argparse.ArgumentParser(
        description="Recompute Phase 1 analysis from streamed JSONL"
    )
    p.add_argument("--input_dir", required=True)
    p.add_argument("--derived_ks", type=int, nargs="+", default=[4, 16, 64])
    p.add_argument("--depths", type=int, nargs="+", default=[0, 1])
    p.add_argument("--k_ref", type=int, default=256)
    args = p.parse_args()

    roots_path = os.path.join(args.input_dir, "root_results.jsonl")
    recs = _load_records(roots_path)
    print(f"Loaded {len(recs)} root records from {roots_path}")

    summary = aggregate_convergence(recs)
    assessment = go_no_go_assessment(summary, args.derived_ks)
    comparison = estimator_comparison(recs)
    justification = search_justification_gate(comparison)
    scale = estimate_global_advantage_scale(recs)

    summary_with = dict(summary)
    summary_with["_comparison"] = comparison
    summary_with["_justification_gate"] = justification

    with open(os.path.join(args.input_dir, "summary.json"), "w") as f:
        json.dump(summary_with, f, indent=2, default=lambda o: str(o))
    with open(os.path.join(args.input_dir, "comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2, default=lambda o: str(o))

    md = [
        "# Test-Time Search — Phase 1 Fixed-Root Estimator Benchmark (RECOVERED)",
        "",
        f"- roots: {summary.get('_n_roots', 0)}",
        f"- K_ref: {args.k_ref}",
        f"- derived K: {args.derived_ks}",
        f"- depths: {args.depths}",
        f"- §22 convergence verdict: **{assessment['verdict']}** ({assessment['passed']}/{assessment['total']} criteria)",
        f"- §23-precondition (search-justification) verdict: **{justification['verdict']}** ({justification['passed']}/{justification['total']} criteria)",
        "",
        convergence_table(summary, args.derived_ks),
        "",
        comparison_table(comparison),
        "",
        "## §22 convergence gate (estimator validity)",
        "",
        "```json",
        json.dumps(assessment, indent=2, default=lambda o: str(o)),
        "```",
        "",
        "## §23-precondition gate (is there a search signal?)",
        "",
        "```json",
        json.dumps(justification, indent=2, default=lambda o: str(o)),
        "```",
    ]
    with open(os.path.join(args.input_dir, "REPORT.md"), "w") as f:
        f.write("\n".join(md))

    out = {
        "n_roots": len(recs),
        "convergence_verdict": assessment["verdict"],
        "search_justification_verdict": justification["verdict"],
        "phase_distribution": comparison.get("phase_distribution", {}),
        "global_advantage_scale": scale,
    }
    with open(os.path.join(args.input_dir, "analysis_recovered.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
