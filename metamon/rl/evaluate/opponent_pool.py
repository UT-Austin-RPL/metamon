"""Opponent pool config for vectorized Showdown eval and training.

Same YAML shape as ladder self-play (``agents`` + ``defaults``) and h2h
``policies``. Reuses :mod:`metamon.rl.evaluate.common` for loading, merging,
and sampling — one shared opponent per env ``reset()``.
"""

from __future__ import annotations

import glob
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from metamon.env import get_metamon_teams
from metamon.env.wrappers import TeamSet
from metamon.rl.evaluate.common import (
    PolicySpec,
    expand_agent_pool_entries,
    load_config,
    merge_defaults,
    sample_policy_from_merged,
)

_EPOCH_RE = re.compile(r"policy_epoch_(\d+)\.pt$")


def _discover_checkpoints(
    model_name: str,
    include_latest: bool,
    min_epoch: Optional[int],
    max_epoch: Optional[int],
    step: Optional[int],
) -> List[int]:
    """List a local model's saved epochs (oldest→newest), optionally + ``-1`` latest.

    Reads ``{save_dir}/{run}/ckpts/policy_weights/policy_epoch_{N}.pt`` for the
    registered ``model_name``. Returns ``[]`` if the model can't be instantiated
    yet (e.g. the finetune run's ckpt dir does not exist), so an early collector
    simply sees an empty self slice and falls back to static/parent pool rows.
    The rolling ``latest/policy.pt`` (``-1``) is appended as the newest entry only
    when ``include_latest`` and the file already exists on disk.
    """
    from metamon.rl.pretrained import LATEST_CHECKPOINT, get_pretrained_model

    try:
        model = get_pretrained_model(model_name)
    except Exception:
        return []
    ckpt_dir = getattr(model, "local_ckpt_dir", None)
    if not ckpt_dir:
        return []

    epochs: List[int] = []
    for path in glob.glob(
        os.path.join(ckpt_dir, "policy_weights", "policy_epoch_*.pt")
    ):
        m = _EPOCH_RE.search(os.path.basename(path))
        if m:
            epochs.append(int(m.group(1)))
    epochs = sorted(set(epochs))
    if min_epoch is not None:
        epochs = [e for e in epochs if e >= min_epoch]
    if max_epoch is not None:
        epochs = [e for e in epochs if e <= max_epoch]
    if step and int(step) > 1:
        # Subsample from the newest end so the most recent epochs are always kept.
        epochs = epochs[::-1][:: int(step)][::-1]

    result: List[int] = list(epochs)
    if include_latest:
        latest_path = model.get_path_to_checkpoint(LATEST_CHECKPOINT)
        if os.path.exists(latest_path):
            result.append(LATEST_CHECKPOINT)
    return result


def _recency_num_agents(n_ckpts: int, rho: float, total: Optional[int]) -> List[int]:
    """Integer per-checkpoint row weights, geometrically skewed toward the newest.

    ``weight_i = rho ** rank_i`` with ``rank 0`` = oldest, so ``rho > 1`` favors
    recent checkpoints. Weights are scaled to a soft budget of ``total`` rows
    (default ``2 * n_ckpts``) and rounded, with each checkpoint guaranteed >= 1
    row. Rounding keeps the counts monotonic in ``rank`` (i.e. never fewer rows
    for a newer checkpoint), so the actual row total may differ slightly from
    ``total``.
    """
    if n_ckpts <= 0:
        return []
    weights = [rho**rank for rank in range(n_ckpts)]
    if total is None:
        total = 2 * n_ckpts
    total = max(int(total), n_ckpts)  # guarantee >= 1 row per checkpoint
    s = sum(weights)
    return [max(1, round(total * w / s)) for w in weights]


def _expand_discover_agent(
    base_name: str, merged: dict, spec: Dict[str, Any]
) -> List[Tuple[str, dict]]:
    """Expand a ``checkpoints: {discover: true, ...}`` entry into per-ckpt pool rows.

    Discovers the agent's saved epochs (+ optional rolling ``latest``), assigns each
    a recency-skewed integer ``num_agents``, and emits ordinary weighted pool rows
    (one config per checkpoint). Returns ``[]`` when nothing is discovered yet.
    """
    from metamon.rl.pretrained import LATEST_CHECKPOINT

    model_name = merged["model_name"]
    include_latest = bool(spec.get("include_latest", False))
    rho = float(spec.get("recency_rho", 1.5))
    total = spec.get("total_num_agents", None)
    discovered = _discover_checkpoints(
        model_name,
        include_latest=include_latest,
        min_epoch=spec.get("min_epoch"),
        max_epoch=spec.get("max_epoch"),
        step=spec.get("step"),
    )
    if not discovered:
        return []

    counts = _recency_num_agents(
        len(discovered), rho, int(total) if total is not None else None
    )
    rows: List[Tuple[str, dict]] = []
    for ckpt, n in zip(discovered, counts):
        label = "latest" if ckpt == LATEST_CHECKPOINT else str(ckpt)
        row_merged = dict(merged)
        row_merged["checkpoints"] = [ckpt]
        row_merged["num_agents"] = n
        rows.extend(expand_agent_pool_entries(f"{base_name}_ckpt{label}", row_merged))
    return rows


# Seconds between filesystem re-scans of ``discover: true`` agents. Discovery is
# cheap (a glob + lightweight registry instantiation, no weight loading), but a
# small TTL avoids re-globbing on every single env reset when they come fast.
DEFAULT_DISCOVER_REFRESH_SECONDS = 30.0


def _split_pool_entries(
    raw: Dict[str, Any],
) -> Tuple[List[Tuple[str, dict]], List[Tuple[str, dict, Dict[str, Any]]]]:
    """Split a pool dict into static rows and (deferred) discover specs.

    Returns ``(static_rows, discover_specs)`` where ``static_rows`` are the fully
    expanded ``(name, merged)`` rows for ordinary agents, and ``discover_specs``
    are ``(base_name, merged, spec)`` tuples for agents whose ``checkpoints`` is a
    ``discover: true`` mapping. Discover specs are expanded lazily (and re-scanned
    over time) by :class:`OpponentPoolConfig` so a running collector picks up newly
    saved self-checkpoints, not just the ones present at process start.
    """
    defaults = raw.get("defaults", {})
    agents = raw.get("agents")
    if agents is None:
        agents = raw.get("policies")
    if not agents:
        raise ValueError(
            "Opponent pool config must define an 'agents' or 'policies' section"
        )

    static_rows: List[Tuple[str, dict]] = []
    discover_specs: List[Tuple[str, dict, Dict[str, Any]]] = []
    for base_name, agent_config in agents.items():
        merged = merge_defaults(defaults, agent_config or {})
        if "model_name" not in merged:
            merged["model_name"] = base_name
        checkpoints = merged.get("checkpoints")
        if isinstance(checkpoints, dict) and checkpoints.get("discover"):
            discover_specs.append((base_name, merged, checkpoints))
        else:
            static_rows.extend(expand_agent_pool_entries(base_name, merged))
    return static_rows, discover_specs


def parse_opponent_pool_dict(raw: Dict[str, Any]) -> List[Tuple[str, dict]]:
    """Parse ``agents`` / ``policies`` + ``defaults`` into (name, merged_config) rows.

    Each agent is expanded by its merged ``num_agents`` field (default 1). A value
    of ``N`` adds ``N`` equally-weighted rows, matching ladder self-play expansion.

    An agent whose ``checkpoints`` is a mapping with ``discover: true`` is instead
    expanded into one recency-weighted row per discovered saved epoch (see
    :func:`_expand_discover_agent`), letting a training pool track a run's evolving
    self-checkpoints without hand-editing the YAML. This returns a one-shot
    snapshot; :class:`OpponentPoolConfig` re-scans discover agents over time.
    """
    static_rows, discover_specs = _split_pool_entries(raw)
    rows = list(static_rows)
    for base_name, merged, spec in discover_specs:
        rows.extend(_expand_discover_agent(base_name, merged, spec))
    return rows


class OpponentPoolConfig:
    """Parsed opponent pool — sample one shared opponent per env ``reset()``.

    Static agents are expanded once. ``discover: true`` agents are re-scanned from
    disk at most every ``discover_refresh_seconds`` (see
    :meth:`_maybe_refresh_discovered`), so a long-running collector's self-play
    slice grows as the finetune saves new epochs — without relaunching the process.
    """

    def __init__(
        self,
        agents: List[Tuple[str, dict]],
        battle_format: str,
        rng: Optional[random.Random] = None,
        discover_specs: Optional[List[Tuple[str, dict, Dict[str, Any]]]] = None,
        discover_refresh_seconds: float = DEFAULT_DISCOVER_REFRESH_SECONDS,
    ):
        self.static_agents = list(agents)
        self.discover_specs = list(discover_specs or [])
        if not self.static_agents and not self.discover_specs:
            raise ValueError("OpponentPoolConfig requires at least one agent entry")
        self.battle_format = battle_format
        self.rng = rng or random.Random()
        self._team_cache: Dict[str, TeamSet] = {}
        self._discover_refresh_seconds = max(0.0, float(discover_refresh_seconds))
        self._discovered_rows: List[Tuple[str, dict]] = []
        self._discover_next_refresh = 0.0
        if self.discover_specs:
            self._maybe_refresh_discovered(force=True)

    @classmethod
    def from_dict(
        cls,
        raw: Dict[str, Any],
        battle_format: str,
        rng: Optional[random.Random] = None,
    ) -> "OpponentPoolConfig":
        static_rows, discover_specs = _split_pool_entries(raw)
        refresh = float(
            raw.get("discover_refresh_seconds", DEFAULT_DISCOVER_REFRESH_SECONDS)
        )
        return cls(
            agents=static_rows,
            battle_format=battle_format,
            rng=rng,
            discover_specs=discover_specs,
            discover_refresh_seconds=refresh,
        )

    def _maybe_refresh_discovered(self, force: bool = False) -> None:
        """Re-expand ``discover: true`` agents if the refresh TTL has elapsed."""
        if not self.discover_specs:
            return
        now = time.monotonic()
        if not force and now < self._discover_next_refresh:
            return
        rows: List[Tuple[str, dict]] = []
        for base_name, merged, spec in self.discover_specs:
            rows.extend(_expand_discover_agent(base_name, merged, spec))
        self._discovered_rows = rows
        self._discover_next_refresh = now + self._discover_refresh_seconds

    @property
    def agents(self) -> List[Tuple[str, dict]]:
        """All currently active rows (static + most recently discovered)."""
        return self.static_agents + self._discovered_rows

    def sample_opponent(self) -> PolicySpec:
        """Pick an agent, then sample checkpoint / temperature / team set."""
        self._maybe_refresh_discovered()
        population = self.agents
        if not population:
            raise RuntimeError(
                "Opponent pool is empty: no static agents and discover agents have "
                "not found any checkpoints yet. Add a static fallback entry or wait "
                "for the run to save an epoch / latest/policy.pt."
            )
        name, merged = self.rng.choice(population)
        return sample_policy_from_merged(name, merged)

    def team_set_for(self, team_set_name: str) -> TeamSet:
        if team_set_name not in self._team_cache:
            self._team_cache[team_set_name] = get_metamon_teams(
                self.battle_format, team_set_name
            )
        return self._team_cache[team_set_name]


def make_simple_opponent_pool_dict(
    opponent_agent: str,
    team_set: str = "competitive",
    checkpoint: Optional[int] = None,
    temperature: float = 1.0,
    battle_backend: str = "metamon",
) -> Dict[str, Any]:
    """Minimal one-agent pool (``--eval_type metamon --opponent_agent`` + ``--team_set``)."""
    return {
        "defaults": {
            "team_set": team_set,
            "battle_backend": battle_backend,
            "checkpoints": [checkpoint],
            "temperatures": [temperature],
        },
        "agents": {
            opponent_agent: {"model_name": opponent_agent},
        },
    }


def load_simple_opponent_pool(
    opponent_agent: str,
    battle_format: str,
    team_set: str = "competitive",
    checkpoint: Optional[int] = None,
    temperature: float = 1.0,
    battle_backend: str = "metamon",
    rng: Optional[random.Random] = None,
) -> OpponentPoolConfig:
    raw = make_simple_opponent_pool_dict(
        opponent_agent=opponent_agent,
        team_set=team_set,
        checkpoint=checkpoint,
        temperature=temperature,
        battle_backend=battle_backend,
    )
    return OpponentPoolConfig.from_dict(raw, battle_format=battle_format, rng=rng)


def load_opponent_pool(
    config_path: str,
    battle_format: str,
    template_vars: Optional[Dict[str, str]] = None,
) -> OpponentPoolConfig:
    raw = load_config(config_path, template_vars=template_vars)
    return OpponentPoolConfig.from_dict(raw, battle_format=battle_format)
