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
_TRAJ_FILENAME_RE = re.compile(
    r"_vs_(?P<opponent>.+?)_"
    r"(?P<ts>\d{2}-\d{2}-\d{4}-\d{2}:\d{2}:\d{2})_"
    r"(?P<result>WIN|LOSS)"
    r"\.json(?:\.lz4)?$"
)


def parse_trajectory_filename(filename: str) -> Optional[Tuple[str, str]]:
    """Return ``(opponent_short_label, result)`` from a trajectory filename.

    ``result`` is ``"WIN"`` or ``"LOSS"``. Returns ``None`` if the filename does
    not match the metamon collection format (e.g. human replay filenames).
    """
    base = os.path.basename(filename)
    m = _TRAJ_FILENAME_RE.search(base)
    if m is None:
        return None
    return m.group("opponent"), m.group("result")


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

    @property
    def sidecar_path(self) -> str:
        return os.path.join(
            os.path.abspath(self.buffer_dir), self.battle_format, "meta_weights.json"
        )


def _scan_recent_files(
    fmt_dir: str, window: int
) -> List[Tuple[float, str]]:
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
) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
    """Compute per-agent prioritized weights from the FIFO buffer filenames.

    Returns ``(weights, diagnostics)`` where ``weights`` maps each agent name in
    ``agent_names`` to a normalized weight (summing to 1), and ``diagnostics``
    maps each agent name to ``{n, win_rate, raw_weight, weight}``.

    The transform (PFSP-style):

    .. code-block:: python

        p_o        = (wins_o + α) / (n_o + 2α)        # Laplace-smoothed win rate
        score_o    = max(0, 0.5 - p_o)               # high when learner loses
        w_o        = score_o ** (1/τ)                # temperature; τ→∞ ⇒ uniform
        w_o        = max(floor, w_o)                 # diversity floor
        W          = w / sum(w)

    Agents with ``n_o < min_games`` fall back to the uniform weight. If every
    agent is cold or all scores are zero, returns uniform weights. EMA smoothing
    blends the new weights with ``prev_weights`` (``W_t = β·W_t + (1-β)·W_{t-1}``,
    where ``β = ema``); ``prev_weights=None`` initializes to uniform.
    """
    n_agents = len(agent_names)
    if n_agents == 0:
        return {}, {}

    fmt_dir = os.path.join(os.path.abspath(buffer_dir), battle_format)
    files = _scan_recent_files(fmt_dir, window)

    # Aggregate per-agent counts / wins.
    n_games: Dict[str, int] = {name: 0 for name in agent_names}
    wins: Dict[str, int] = {name: 0 for name in agent_names}
    unmatched = 0
    for _, path in files:
        parsed = parse_trajectory_filename(path)
        if parsed is None:
            continue
        opp_label, result = parsed
        agent = match_agent_name(opp_label, agent_names)
        if agent is None:
            unmatched += 1
            continue
        n_games[agent] += 1
        if result == "WIN":
            wins[agent] += 1

    # Raw per-agent weights via the prioritized transform.
    raw: Dict[str, float] = {}
    for name in agent_names:
        n_o = n_games[name]
        if n_o < min_games:
            # Cold opponent: uniform fallback (resolved into the mix below).
            raw[name] = float("nan")
            continue
        p_o = (wins[name] + laplace_alpha) / (n_o + 2.0 * laplace_alpha)
        score = max(0.0, 0.5 - p_o)
        if score <= 0.0:
            # We dominate this opponent: floor keeps a non-zero diversity mass.
            w = floor
        else:
            w = max(floor, score ** (1.0 / max(temp, 1e-9)))
        raw[name] = w

    # If every agent is cold, return uniform.
    if all(math.isnan(v) for v in raw.values()):
        uniform = 1.0 / n_agents
        weights = {name: uniform for name in agent_names}
        diag = {
            name: {
                "n": n_games[name],
                "win_rate": None,
                "raw_weight": None,
                "weight": uniform,
            }
            for name in agent_names
        }
        if unmatched:
            diag["_unmatched_files"] = unmatched  # type: ignore[assignment]
        return weights, diag

    # Cold agents inherit the uniform mass (mean of the *active* agents' raw
    # weights would bias the mix; instead give them the uniform share so they
    # neither vanish nor dominate while we gather games).
    active = [v for v in raw.values() if not math.isnan(v)]
    uniform_share = 1.0 / n_agents if active else 0.0
    w_vec = {name: (uniform_share if math.isnan(raw[name]) else raw[name]) for name in agent_names}

    total = sum(w_vec.values())
    if total <= 0.0:
        uniform = 1.0 / n_agents
        weights = {name: uniform for name in agent_names}
    else:
        weights = {name: w_vec[name] / total for name in agent_names}

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
        diag[name] = {
            "n": n_o,
            "win_rate": win_rate,
            "raw_weight": None if math.isnan(raw[name]) else raw[name],
            "weight": weights[name],
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
