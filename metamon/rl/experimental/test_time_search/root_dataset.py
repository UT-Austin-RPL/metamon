"""Fixed-root corpus capture / replay utilities (skill §22 Phase 1).

A *fixed root* is one settled evaluated-player decision in a real Showdown
battle, identified stably so the estimator can be evaluated at multiple
``(K, depth, leaf_mode)`` configs against the **same** trunk state. This module
defines the manifest schema and the stratification features used to subgroup
roots in the convergence analysis (skill §22 "Root corpus" / "Estimator
metrics" / §31).

Two capture modes are supported:

* **in-battle** (primary for the pilot): roots are captured *while* the
  benchmark battle runs, and the estimator grid is evaluated at each root
  before the trunk advances (``benchmark_roots.benchmark_roots``). The trunk
  state is identical across configs by construction -- search forks never
  advance the trunk -- so this is the cleanest "fixed root".
* **replay-from-history** (skill §31 truth source): a manifest records the env
  seed + per-step action history so a root can be reconstructed later. The
  replay helper is provided for later use; the pilot uses in-battle capture
  because it needs no opponent-action injection machinery and trivially
  guarantees the fixed-root property.

The manifest is JSONL (one entry per root). It records enough to (a) stratify
the convergence metrics and (b) reconstruct the root's public trajectory from
the env seed + action history. It does **not** store opaque policy pickles
(skill §31: "Never rely on opaque pickles without version and checkpoint
hashes"); the checkpoint hash is recorded once in the run-level manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Stratification features
# ---------------------------------------------------------------------------


def _safe_entropy(probs: np.ndarray) -> float:
    """Natural entropy of a probability vector over its support."""
    p = np.asarray(probs, dtype=np.float64)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def compute_root_features(
    base_probs: np.ndarray, legal_arr: np.ndarray, base_argmax: int
) -> Dict[str, Any]:
    """Pre-estimate stratification features (skill §22 "Stratify metrics by").

    Args:
        base_probs: (A_full,) frozen actor distribution.
        legal_arr: (A,) retained legal actions.
        base_argmax: the base actor argmax over legal actions.

    Returns a dict with ``n_legal``, ``base_entropy`` (over the retained legal
    actions), ``base_top1_prob``, ``base_top2_gap`` (top1 - top2 legal prob),
    and ``base_argmax_prob``. These do not require a Q estimate, so they can be
    computed before the rollout grid and used to pre-stratify / cap roots.
    """
    legal_probs = np.asarray(base_probs, dtype=np.float64)[legal_arr]
    n_legal = int(legal_arr.size)
    entropy = _safe_entropy(legal_probs)
    order = np.argsort(-legal_probs)
    top1 = float(legal_probs[order[0]]) if n_legal else 0.0
    top2 = float(legal_probs[order[1]]) if n_legal > 1 else 0.0
    base_argmax_prob = (
        float(legal_probs[int(np.argmax(legal_probs))]) if n_legal else 0.0
    )
    return {
        "n_legal": n_legal,
        "base_entropy": entropy,
        "base_top1_prob": top1,
        "base_top2_gap": top1 - top2,
        "base_argmax_prob": base_argmax_prob,
    }


def entropy_band(entropy: float) -> str:
    """Coarse actor-entropy band for subgroup analysis (skill §22)."""
    if entropy < 0.9:
        return "low"
    if entropy < 1.6:
        return "medium"
    return "high"


def top2_gap_band(gap: float) -> str:
    """Coarse actor top-2 gap band (skill §22: "large and small actor top-2 gaps")."""
    if gap < 0.15:
        return "small"
    if gap < 0.45:
        return "medium"
    return "large"


def ref_gap_band(gap: float) -> str:
    """Coarse reference-Q top-2 gap band (how clear the best action is)."""
    if abs(gap) < 20.0:
        return "near_tied"
    if abs(gap) < 100.0:
        return "medium"
    return "clear"


def phase_band(decision: int, typical_battle_len: int = 120) -> str:
    """Coarse battle-phase band from the decision index (skill §22: early/mid/late).

    ``typical_battle_len`` is a rough Gen1 OU self-play default; the analysis
    also records the raw decision index so bands can be recomputed post-hoc.
    """
    frac = decision / max(typical_battle_len, 1)
    if frac < 0.33:
        return "early"
    if frac < 0.66:
        return "mid"
    return "late"


# ---------------------------------------------------------------------------
# Manifest entry
# ---------------------------------------------------------------------------


@dataclass
class RootManifestEntry:
    """One fixed root (skill §31 capture format).

    The pre-estimate fields (``n_legal`` ... ``base_top2_gap``) are filled at
    capture time. The post-estimate fields (``ref_q_*``, ``critic_disagreement``,
    ``terminal_frac_d0``) are filled by the benchmark after the reference
    estimate, so the manifest doubles as the per-root result record.
    """

    root_id: str
    battle_id: str
    lane: int
    decision: int
    battle_seed: Optional[int]
    legal_actions: List[int]
    # pre-estimate stratification features
    n_legal: int
    base_probs: List[float]
    base_argmax: int
    base_entropy: float
    base_top1_prob: float
    base_top2_gap: float
    base_argmax_prob: float
    entropy_band: str
    top2_gap_band: str
    phase_band: str
    # action history up to (not including) this root (skill §31 replay source).
    # For in-battle capture this is the sequence of (eval_action, opp_action)
    # pairs submitted so far in this battle; replay re-seeds and re-submits them.
    action_history: List[List[int]] = field(default_factory=list)
    # post-estimate (filled by the benchmark):
    ref_q_mean: Optional[List[float]] = None
    ref_q_argmax: Optional[int] = None
    ref_q_gap: Optional[float] = None
    ref_q_top2_gap_band: Optional[str] = None
    critic_disagreement: Optional[float] = None
    terminal_frac_d0: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RootManifestEntry":
        return cls(**d)


def make_manifest_entry(
    battle_id: str,
    lane: int,
    decision: int,
    battle_seed: Optional[int],
    legal: List[int],
    base_probs: np.ndarray,
    legal_arr: np.ndarray,
    base_argmax: int,
    action_history: List[List[int]],
) -> RootManifestEntry:
    """Build a manifest entry from the base distribution at one root."""
    feats = compute_root_features(base_probs, legal_arr, base_argmax)
    return RootManifestEntry(
        root_id=f"{battle_id}:d{decision}",
        battle_id=battle_id,
        lane=lane,
        decision=decision,
        battle_seed=battle_seed,
        legal_actions=[int(x) for x in legal_arr],
        n_legal=feats["n_legal"],
        base_probs=[float(x) for x in base_probs[legal_arr]],
        base_argmax=int(base_argmax),
        base_entropy=feats["base_entropy"],
        base_top1_prob=feats["base_top1_prob"],
        base_top2_gap=feats["base_top2_gap"],
        base_argmax_prob=feats["base_argmax_prob"],
        entropy_band=entropy_band(feats["base_entropy"]),
        top2_gap_band=top2_gap_band(feats["base_top2_gap"]),
        phase_band=phase_band(decision),
        action_history=[[int(a), int(o)] for a, o in action_history],
    )


def fill_reference_fields(
    entry: RootManifestEntry,
    ref_q: np.ndarray,
    legal_arr: np.ndarray,
    critic_disagreement: float,
    terminal_frac_d0: float,
) -> None:
    """Fill the post-estimate reference fields on ``entry`` in place.

    ``ref_q`` is the high-K reference Q over the retained legal actions;
    ``legal_arr`` maps the entry's legal actions to the estimate's ordering
    (they are identical in the in-battle path, but passed explicitly for
    safety).
    """
    q = np.asarray(ref_q, dtype=np.float64)
    entry.ref_q_mean = [float(x) for x in q]
    entry.ref_q_argmax = int(legal_arr[int(np.argmax(q))]) if q.size else None
    order = np.argsort(-q)
    top1 = float(q[order[0]]) if q.size else 0.0
    top2 = float(q[order[1]]) if q.size > 1 else 0.0
    gap = top1 - top2
    entry.ref_q_gap = float(gap)
    entry.ref_q_top2_gap_band = ref_gap_band(gap)
    entry.critic_disagreement = float(critic_disagreement)
    entry.terminal_frac_d0 = float(terminal_frac_d0)


def write_manifest(entries: List[RootManifestEntry], path: str) -> None:
    """Write manifest entries as JSONL (one root per line)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for e in entries:
            f.write(e.to_json() + "\n")


def read_manifest(path: str) -> List[RootManifestEntry]:
    """Read a JSONL manifest written by :func:`write_manifest`."""
    out: List[RootManifestEntry] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(RootManifestEntry.from_dict(json.loads(line)))
    return out


# ---------------------------------------------------------------------------
# Run-level manifest (skill §20 / §37 reproducibility artifacts)
# ---------------------------------------------------------------------------


def checkpoint_hash(ckpt_path: str) -> str:
    """SHA256 of a checkpoint file (skill §37: checkpoint hash)."""
    h = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_manifest(
    *,
    agent: str,
    checkpoint: int,
    ckpt_path: str,
    battle_format: str,
    team_set: str,
    env_seed: Optional[int],
    search_seed: int,
    k_ref: int,
    derived_ks: List[int],
    depths: List[int],
    leaf_modes: List[str],
    chance_mode: str,
    n_roots: int,
    n_battles: int,
    git_sha: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the run-level manifest dict (skill §20)."""
    import time as _t

    return {
        "timestamp": _t.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "agent": agent,
        "checkpoint": checkpoint,
        "checkpoint_hash": (
            checkpoint_hash(ckpt_path) if os.path.exists(ckpt_path) else None
        ),
        "battle_format": battle_format,
        "team_set": team_set,
        "env_seed": env_seed,
        "search_seed": search_seed,
        "k_ref": k_ref,
        "derived_ks": list(derived_ks),
        "depths": list(depths),
        "leaf_modes": list(leaf_modes),
        "chance_mode": chance_mode,
        "n_roots": n_roots,
        "n_battles": n_battles,
        "git_sha": git_sha,
        "extra": extra or {},
    }
