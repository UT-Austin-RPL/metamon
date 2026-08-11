"""PSRO-Lite: prioritized opponent sampling for online RL (v1).

Reweights the opponent pool at collection time using a meta-distribution derived
from the learner's empirical win rate per opponent, computed from the win/loss
tags already embedded in the FIFO buffer's trajectory filenames.

This module is the stateless core (``compute_prioritized_weights``) plus a small
stateful updater (``PsroLite``) that the collector process drives once per
``update_interval`` epochs. The weights are transported to the env and the FIFO
sampler via a sidecar JSON file written atomically to
``{buffer_dir}/{format}/meta_weights.json``.

See ``docs/psro_lite_plan.md`` for the full design.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

# ``_save_lane_outcome`` writes:
#   metamon-{format}-{battle_id}_Unrated_{player}_vs_{opponent}_{ts}_{WIN|LOSS}.json.lz4
# where ``{opponent}`` is the active PolicySpec's ``short_label`` (which may
# itself contain ``-`` and ``_``, e.g. ``TaurosV0-ckpt40-t2.0-gl_05_26``) and
# ``{ts}`` is ``MM-DD-YYYY-HH:MM:SS``. The timestamp's fixed shape lets a
# non-greedy capture of the opponent token terminate unambiguously.
#
# An optional ``_ts-{teamset}`` token may appear between the timestamp and the
# result; it records the concrete team set the *learner* (player) drew for that
# battle (e.g. ``gl_05_26``, ``smogon_pass2_selected``). It is emitted by
# ``VectorizedShowdownEnv._save_lane_outcome`` when the player's team file is
# known, so PSRO win rates can be broken down per team set. Old files (and
# random-team formats) omit it; ``parse_trajectory_filename`` returns ``None``
# for the teamset in that case.
_TRAJ_FILENAME_RE = re.compile(
    r"_vs_(?P<opponent>.+?)_"
    r"(?P<ts>\d{2}-\d{2}-\d{4}-\d{2}:\d{2}:\d{2})"
    r"(?:_ts-(?P<teamset>[A-Za-z0-9_.]+))?_"
    r"(?P<result>WIN|LOSS)"
    r"\.json(?:\.lz4)?$"
)


def parse_trajectory_filename(
    filename: str,
) -> Optional[Tuple[str, Optional[str], str]]:
    """Return ``(opponent_short_label, teamset, result)`` from a trajectory filename.

    ``result`` is ``"WIN"`` or ``"LOSS"``. ``teamset`` is the concrete team set
    the learner drew for that battle (e.g. ``"gl_05_26"``), or ``None`` when the
    filename carries no ``_ts-{teamset}`` token (old files, random formats).
    Returns ``None`` entirely if the filename does not match the metamon
    collection format (e.g. human replay filenames).
    """
    base = os.path.basename(filename)
    m = _TRAJ_FILENAME_RE.search(base)
    if m is None:
        return None
    return m.group("opponent"), m.group("teamset"), m.group("result")


def match_agent_name(short_label: str, agent_names: List[str]) -> Optional[str]:
    """Map a ``short_label`` (``name-ckptN-tN-team_set``) back to its row name.

    ``short_label`` starts with the agent row name (the ``name`` field of the
    pool row, possibly ``base_name-N`` from ``num_agents`` expansion), followed
    by ``-ckptN`` / ``-tN`` / ``-team_set`` suffixes. We return the longest
    ``agent_name`` that is a prefix of ``short_label`` and is followed by either
    ``-`` or the end of the string (so ``TaurosV0`` does not steal
    ``TaurosV0-1-ckpt40-...`` from the ``TaurosV0-1`` row).
    """
    # Try longest first so ``TaurosV0-1`` wins over ``TaurosV0`` when both match.
    for name in sorted(agent_names, key=len, reverse=True):
        if short_label == name:
            return name
        if short_label.startswith(name) and short_label[len(name) :].startswith("-"):
            return name
    return None


# ---------------------------------------------------------------------------
# Core weight computation
# ---------------------------------------------------------------------------


@dataclass
class PsroConfig:
    """All knobs for PSRO-Lite, passed through from the CLI."""

    buffer_dir: str
    battle_format: str
    agent_names: List[str]
    start_epoch: int = 0
    update_interval: int = 5
    window: int = 50_000
    min_games: int = 20
    temp: float = 1.0
    floor: float = 0.05
    ema: float = 0.7
    solver: str = "prioritized"  # "nash" reserved for v3
    fifo_reweight: bool = False
    buffer_trim: Optional[int] = None
    # Cold-fallback / safety-net knobs (see compute_prioritized_weights).
    novelty_gamma: float = 0.0
    novelty_gamma0: float = 5.0
    cap_ratio: Optional[float] = None

    @property
    def sidecar_path(self) -> str:
        return os.path.join(
            os.path.abspath(self.buffer_dir), self.battle_format, "meta_weights.json"
        )


def _scan_recent_files(fmt_dir: str, window: int) -> List[Tuple[float, str]]:
    """Return the ``window`` most-recently-modified ``.json(.lz4)`` files.

    Falls back to *all* files if fewer than ``window`` exist. One ``scandir``
    pass with a stat per entry (matters on NFS with hundreds of thousands of
    files).
    """
    entries: List[Tuple[float, str]] = []
    try:
        with os.scandir(fmt_dir) as it:
            for entry in it:
                name = entry.name
                if not name.endswith((".json", ".json.lz4")):
                    continue
                if name == "meta_weights.json":
                    continue
                try:
                    entries.append((entry.stat().st_mtime, entry.path))
                except OSError:
                    continue
    except (OSError, FileNotFoundError):
        return []
    entries.sort(key=lambda x: x[0], reverse=True)
    if window > 0 and len(entries) > window:
        entries = entries[:window]
    return entries


def compute_prioritized_weights(
    *,
    buffer_dir: str,
    battle_format: str,
    agent_names: List[str],
    window: int = 50_000,
    min_games: int = 20,
    temp: float = 1.0,
    floor: float = 0.05,
    ema: float = 0.7,
    prev_weights: Optional[Dict[str, float]] = None,
    laplace_alpha: float = 1.0,
    novelty_gamma: float = 0.0,
    novelty_gamma0: float = 5.0,
    cap_ratio: Optional[float] = None,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
    """Compute per-agent prioritized weights from the FIFO buffer filenames.

    Returns ``(weights, diagnostics)`` where ``weights`` maps each agent name in
    ``agent_names`` to a normalized weight (summing to 1), and ``diagnostics``
    maps each agent name to ``{n, win_rate, raw_weight, weight}``.

    The transform (PFSP-style with a confidence-weighted cold fallback):

    .. code-block:: python

        p_o      = (wins_o + α) / (n_o + 2α)     # Laplace-smoothed learner win rate
        score_o  = max(0, 0.5 - p_o)             # high when learner loses
        conf_o   = n_o / (n_o + k)               # 0 at n=0 → 1; k = min_games
        w_o      = (score_o ** (1/τ)) * conf_o   # cold → 0, not uniform
        w_o     += γ / (n_o + γ0)                # decaying novelty bonus (γ>0)
        w_o      = max(floor, w_o)               # diversity floor
        w_o      = min(w_o, R*floor)             # optional weight-ratio cap (R)
        W        = w / sum(w)

    Cold opponents (``n_o`` small) get ``conf_o ≈ 0`` so their raw weight
    collapses to the ``floor`` rather than snapping to the uniform share — this
    is the fix for the oscillation where a dominated opponent's game count dips
    below ``min_games`` and its weight spikes to ``1/n_agents``. A never-played
    opponent still gets a Laplace-neutral ``score_o = 0`` (``p_o = 0.5``), so it
    also rests at the floor; set ``novelty_gamma > 0`` to add a small,
    ``n``-decaying exploration bump on top of the floor for genuinely novel
    opponents. ``cap_ratio`` (e.g. 20) hard-bounds any raw weight to ``R*floor``
    as a safety net against solver spikes.

    If every agent is cold or all raw weights are zero/``floor``-only, the
    normalized result is uniform. EMA smoothing blends the new weights with
    ``prev_weights`` (``W_t = β·W_t + (1-β)·W_{t-1}``, where ``β = ema``);
    ``prev_weights=None`` initializes to uniform.
    """
    n_agents = len(agent_names)
    if n_agents == 0:
        return {}, {}

    fmt_dir = os.path.join(os.path.abspath(buffer_dir), battle_format)
    files = _scan_recent_files(fmt_dir, window)

    # Aggregate per-agent counts / wins (overall + per learner team set).
    n_games: Dict[str, int] = {name: 0 for name in agent_names}
    wins: Dict[str, int] = {name: 0 for name in agent_names}
    # per_teamset[agent][teamset] = [n, wins] — teamset None buckets as "_unknown".
    per_teamset: Dict[str, Dict[str, List[int]]] = {name: {} for name in agent_names}
    unmatched = 0
    for _, path in files:
        parsed = parse_trajectory_filename(path)
        if parsed is None:
            continue
        opp_label, teamset, result = parsed
        agent = match_agent_name(opp_label, agent_names)
        if agent is None:
            unmatched += 1
            continue
        n_games[agent] += 1
        if result == "WIN":
            wins[agent] += 1
        ts_key = teamset if teamset is not None else "_unknown"
        bucket = per_teamset[agent].setdefault(ts_key, [0, 0])
        bucket[0] += 1
        if result == "WIN":
            bucket[1] += 1

    # Raw per-agent weights via the confidence-weighted prioritized transform.
    # ``min_games`` is now a soft concentration parameter (k), not a hard gate:
    # conf_o = n_o / (n_o + k) smoothly damps cold opponents toward the floor
    # instead of snapping them to the uniform share (which caused dominated
    # opponents to spike when their rolling-window count dipped below the gate).
    k = max(float(min_games), 1e-9)
    raw: Dict[str, float] = {}
    for name in agent_names:
        n_o = n_games[name]
        p_o = (wins[name] + laplace_alpha) / (n_o + 2.0 * laplace_alpha)
        score = max(0.0, 0.5 - p_o)
        conf = n_o / (n_o + k)
        w = (score ** (1.0 / max(temp, 1e-9))) * conf
        if novelty_gamma > 0.0:
            # Decaying exploration bump for genuinely novel opponents; fades to
            # 0 as games accrue. Additive on top of the score (not the floor),
            # so a never-played opponent starts at floor + γ/γ0, not 1/n_agents.
            w += novelty_gamma / (n_o + max(novelty_gamma0, 1e-9))
        w = max(floor, w)
        if cap_ratio is not None and cap_ratio > 0.0:
            w = min(w, cap_ratio * floor)
        raw[name] = w

    # If every raw weight is equal (all floor → all opponents cold/dominated),
    # normalization yields uniform; detect that explicitly to short-circuit.
    total = sum(raw.values())
    if total <= 0.0 or len(set(raw.values())) == 1:
        uniform = 1.0 / n_agents
        weights = {name: uniform for name in agent_names}
    else:
        weights = {name: raw[name] / total for name in agent_names}

    # EMA smoothing with prev_weights.
    if prev_weights is not None and 0.0 < ema < 1.0:
        beta = ema
        ema_w: Dict[str, float] = {}
        prev_total = 0.0
        for name in agent_names:
            pv = prev_weights.get(name)
            if pv is None:
                pv = 1.0 / n_agents
            prev_total += pv
        for name in agent_names:
            pv = prev_weights.get(name)
            if pv is None or prev_total <= 0.0:
                pv = 1.0 / n_agents
            else:
                pv = pv / prev_total
            ema_w[name] = beta * weights[name] + (1.0 - beta) * pv
        t = sum(ema_w.values())
        if t > 0.0:
            weights = {name: ema_w[name] / t for name in agent_names}

    diag = {}
    for name in agent_names:
        n_o = n_games[name]
        if n_o > 0:
            win_rate = wins[name] / n_o
        else:
            win_rate = None
        # Per learner-teamset win rates — diagnostic only (the solver weights
        # use the overall aggregate above). ``_unknown`` groups files written
        # before the ``_ts-`` filename token existed (or random-team formats).
        ts_diag: Dict[str, Dict[str, Any]] = {}
        for ts_key, (ts_n, ts_w) in per_teamset[name].items():
            ts_diag[ts_key] = {
                "n": ts_n,
                "win_rate": (ts_w / ts_n) if ts_n > 0 else None,
            }
        diag[name] = {
            "n": n_o,
            "win_rate": win_rate,
            "raw_weight": raw[name],
            "weight": weights[name],
            "per_teamset": ts_diag,
        }
    if unmatched:
        diag["_unmatched_files"] = unmatched  # type: ignore[assignment]
    return weights, diag


def weight_entropy(weights: Dict[str, float]) -> float:
    """Shannon entropy of the weight distribution (natural log)."""
    h = 0.0
    for w in weights.values():
        if w > 0.0:
            h -= w * math.log(w)
    return h


# ---------------------------------------------------------------------------
# Stateful updater (driven by the collector process)
# ---------------------------------------------------------------------------


@dataclass
class PsroLite:
    """Stateful wrapper that scans the buffer and writes the sidecar JSON.

    Holds the EMA ``prev_weights`` in memory (initialized to uniform on the
    first ``step``) and writes ``meta_weights.json`` atomically.
    """

    config: PsroConfig
    _prev_weights: Optional[Dict[str, float]] = field(default=None, init=False)
    _last_write_ok: bool = field(default=True, init=False)

    def step(self, *, epoch: int) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
        """Scan the buffer, compute weights, write the sidecar. Returns diagnostics."""
        cfg = self.config
        weights, diag = compute_prioritized_weights(
            buffer_dir=cfg.buffer_dir,
            battle_format=cfg.battle_format,
            agent_names=cfg.agent_names,
            window=cfg.window,
            min_games=cfg.min_games,
            temp=cfg.temp,
            floor=cfg.floor,
            ema=cfg.ema,
            prev_weights=self._prev_weights,
            novelty_gamma=cfg.novelty_gamma,
            novelty_gamma0=cfg.novelty_gamma0,
            cap_ratio=cfg.cap_ratio,
        )
        self._prev_weights = dict(weights)
        self._last_write_ok = self._write_sidecar(weights)
        diag["_epoch"] = epoch  # type: ignore[assignment]
        diag["_sidecar_write_ok"] = self._last_write_ok  # type: ignore[assignment]
        diag["_weight_entropy"] = weight_entropy(weights)  # type: ignore[assignment]
        return weights, diag

    def _write_sidecar(self, weights: Dict[str, float]) -> bool:
        path = self.config.sidecar_path
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(weights, f, indent=2, sort_keys=True)
            os.replace(tmp, path)
            return True
        except OSError:
            return False

    def trim_buffer(self) -> int:
        """One-time eviction of the FIFO down to ``buffer_trim`` files.

        Deletes the oldest on-disk trajectories until at most ``buffer_trim``
        files remain in the format dir. Returns the number of files removed.
        Called once at the start epoch to accelerate turnover of the
        uniform-sampled backlog. No-op if ``buffer_trim`` is ``None``.
        """
        target = self.config.buffer_trim
        if target is None:
            return 0
        fmt_dir = os.path.join(
            os.path.abspath(self.config.buffer_dir), self.config.battle_format
        )
        files = _scan_recent_files(fmt_dir, window=0)
        # ``_scan_recent_files`` returns newest-first; evict the oldest tail.
        files.sort(key=lambda x: x[0])  # oldest first
        num_to_remove = max(len(files) - target, 0)
        removed = 0
        for _, path in files[:num_to_remove]:
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
        return removed


# ---------------------------------------------------------------------------
# Sidecar reader (used by the env and the FIFO sampler)
# ---------------------------------------------------------------------------


def read_sidecar(
    path: str, last_mtime: Optional[float]
) -> Tuple[Optional[Dict[str, float]], Optional[float]]:
    """Read the sidecar JSON if its mtime changed; cache by mtime.

    Returns ``(weights, mtime)``. If the file is missing, stale (same mtime),
    or unparseable, returns ``(None, last_mtime)`` so callers keep their
    previous weights / uniform fallback.
    """
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None, last_mtime
    if last_mtime is not None and mtime == last_mtime:
        return None, last_mtime
    try:
        with open(path, "r") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return None, last_mtime
    if not isinstance(raw, dict):
        return None, last_mtime
    weights: Dict[str, float] = {}
    for k, v in raw.items():
        try:
            weights[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    if not weights:
        return None, last_mtime
    return weights, mtime
