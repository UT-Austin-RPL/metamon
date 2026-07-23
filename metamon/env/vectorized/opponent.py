"""Batched in-the-loop opponents for the vectorized Showdown env.

The env synchronizes all lanes at each decision cycle and asks the opponent for
one action per *active* lane in a single call, so opponent inference is amortized
across the batch. Two implementations are provided:

  * :class:`RandomBatchedOpponent` — no NN; returns random action indices (the env
    repairs any illegal pick against the legal mask). Useful for smoke tests.
  * :class:`AmagoBatchedOpponent` — wraps a metamon policy via
    :class:`~metamon.env.vectorized.amago_policy.AmagoLadderPolicyDriver`, using
    the same ``rl2`` / ``time_idx`` bookkeeping as ``QueueOnLocalLadder``.
  * :class:`ConfigBatchedOpponent` — one shared policy for all lanes; on full env
    ``reset()``, sample an opponent from an :class:`~metamon.rl.evaluate.opponent_pool.OpponentPoolConfig`.
    Pool entries may mix action dimensions: each cached policy bundle owns its own
    ``rl2`` buffer and hidden state; ``configure()`` swaps bundles and reinitializes.
"""

from __future__ import annotations

import gc
import json
import math
import os
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
import torch

from .amago_policy import AmagoLadderPolicyDriver

if TYPE_CHECKING:
    from metamon.env.wrappers import TeamSet
    from metamon.rl.evaluate.common import PolicySpec


class BatchedOpponent(ABC):
    """Interface the env uses to drive the in-the-loop opponent."""

    @abstractmethod
    def act(self, active: np.ndarray, obs_list: List[dict]) -> np.ndarray:
        """Return an action index per lane.

        Only entries where ``active[i]`` is True are guaranteed meaningful (and
        only those lanes advance any internal recurrent state). ``obs_list`` has
        one obs dict per lane (the env supplies a cached obs for inactive lanes).
        """

    def observe(self, lane_idx: int, reward: float, action_idx: int) -> None:
        """Record the reward/action for ``lane_idx`` (rl2 bookkeeping)."""

    def reset_lanes(self, done_mask: np.ndarray) -> None:
        """Reset per-lane recurrent state for finished lanes."""

    def reset_all(self) -> None:
        """Reset all per-lane state (called on env.reset)."""


class RandomBatchedOpponent(BatchedOpponent):
    def __init__(
        self, num_lanes: int, action_dim: int, rng: Optional[np.random.Generator] = None
    ):
        self.num_lanes = num_lanes
        self.action_dim = action_dim
        self.rng = rng or np.random.default_rng()

    def act(self, active: np.ndarray, obs_list: List[dict]) -> np.ndarray:
        return self.rng.integers(0, self.action_dim, size=self.num_lanes).astype(
            np.int64
        )


class AmagoBatchedOpponent(BatchedOpponent):
    """Wrap an AMAGO policy with ladder-identical rollout bookkeeping."""

    def __init__(
        self,
        policy: torch.nn.Module,
        device: torch.device,
        num_lanes: int,
        action_dim: int,
        hidden_state=None,
        sample: bool = True,
    ):
        self._driver = AmagoLadderPolicyDriver(
            policy=policy,
            device=device,
            num_lanes=num_lanes,
            action_dim=action_dim,
            hidden_state=hidden_state,
            sample=sample,
        )
        self.action_dim = int(action_dim)

    def act(self, active: np.ndarray, obs_list: List[dict]) -> np.ndarray:
        return self._driver.act(active, obs_list)

    def observe(self, lane_idx: int, reward: float, action_idx: int) -> None:
        self._driver.observe(lane_idx, reward, action_idx)

    def reset_lanes(self, done_mask: np.ndarray) -> None:
        self._driver.reset_lanes(done_mask)

    def reset_all(self) -> None:
        self._driver.reset_all()


class ConfigBatchedOpponent(BatchedOpponent):
    """One opponent shared by all lanes; resample from config on env ``reset()`` only."""

    def __init__(
        self,
        config: "OpponentPoolConfig",
        num_lanes: int,
        device: torch.device,
        sample: bool = True,
        cache_size: int = 1,
        weights_path: Optional[str] = None,
        quota_min_games: Optional[int] = None,
        quota_window: int = 128,
    ):
        from metamon.rl.evaluate.opponent_pool import OpponentPoolConfig

        if not isinstance(config, OpponentPoolConfig):
            raise TypeError(f"config must be OpponentPoolConfig, got {type(config)}")
        self.config = config
        self.num_lanes = int(num_lanes)
        self.device = device
        self.sample = sample
        # PSRO-Lite sidecar: optional ``meta_weights.json`` keyed by agent row
        # name. Re-read only when mtime changes; fall back to uniform on any
        # error. ``None`` disables the reader entirely (val/ladder unchanged).
        self._weights_path = weights_path
        self._weights_mtime: Optional[float] = None
        # Quota-based diversification: a rolling window of recent ``configure()``
        # assignments (one per env reset = ``num_lanes`` games against one shared
        # opponent). Each agent row is guaranteed at least ``quota_min_games``
        # games over the window — dominated, ladder-strong policies can never
        # fall to ~0 games played (which previously triggered the cold-fallback
        # weight spike). The surplus (window slots beyond all quotas) is sampled
        # by the PSRO-Lite weights, so prioritization still tilts toward weaker
        # matchups on the margin. ``quota_min_games=None`` / ``<= 0`` disables
        # the quota (pure weighted sampling, the previous behavior).
        self._quota_window = max(1, int(quota_window))
        min_assignments = 0
        if quota_min_games is not None and quota_min_games > 0 and self.num_lanes > 0:
            min_assignments = max(1, int(math.ceil(quota_min_games / self.num_lanes)))
        self._quota_min_assignments = min_assignments
        self._quota_recent: "deque[str]" = deque(maxlen=self._quota_window)
        # Bound how many opponent policies stay resident on the GPU. Each cached
        # bundle holds a full policy (60-200M params) plus per-lane KV caches, so
        # an unbounded cache OOMs collectors that resample a new opponent every
        # epoch. LRU-evict and free GPU memory beyond this many distinct opponents.
        self._cache_size = max(1, int(cache_size))
        self.current_spec: Optional["PolicySpec"] = None
        self.current_team: Optional["TeamSet"] = None
        self._active_key: Optional[str] = None
        self._bundle: Optional[AmagoBatchedOpponent] = None
        self._cache: "OrderedDict[str, AmagoBatchedOpponent]" = OrderedDict()

    def _make_bundle(self, spec: "PolicySpec") -> AmagoBatchedOpponent:
        from metamon.rl.pretrained import get_pretrained_model

        model = get_pretrained_model(spec.model_name)
        agent = model.initialize_agent(
            checkpoint=spec.checkpoint,
            log=False,
            action_temperature=spec.temperature,
        )
        action_dim = model.action_space.gym_space.n
        agent.policy.to(self.device)
        agent.policy.eval()
        return AmagoBatchedOpponent(
            policy=agent.policy,
            device=self.device,
            num_lanes=self.num_lanes,
            action_dim=action_dim,
            sample=self.sample,
        )

    def _free_bundle(self, bundle: AmagoBatchedOpponent) -> None:
        """Drop a bundle's GPU tensors (policy weights + per-lane KV caches)."""
        driver = getattr(bundle, "_driver", None)
        if driver is not None:
            driver.hidden_state = None
            driver.policy = None

    def _maybe_refresh_weights(self) -> None:
        """Re-read the PSRO-Lite sidecar if it changed; apply to the pool."""
        if self._weights_path is None:
            return
        try:
            mtime = os.path.getmtime(self._weights_path)
        except OSError:
            # No sidecar yet (e.g. before psro_start_epoch) → stay uniform.
            self._weights_mtime = None
            return
        if mtime == self._weights_mtime:
            return
        try:
            with open(self._weights_path, "r") as f:
                raw = json.load(f)
        except (OSError, ValueError):
            return
        if not isinstance(raw, dict):
            return
        # Align sidecar weights (keyed by agent row name) to ``self.config.agents``
        # rows. Missing agents and non-finite values fall back to uniform via
        # ``set_weights`` all-zero/None handling.
        names = [row[0] for row in self.config.agents]
        aligned: List[float] = []
        for name in names:
            try:
                aligned.append(float(raw.get(name, 0.0)))
            except (TypeError, ValueError):
                aligned.append(0.0)
        try:
            self.config.set_weights(aligned)
        except ValueError:
            self.config.set_weights(None)
        self._weights_mtime = mtime

    def _sample_with_quota(self) -> "PolicySpec":
        """Two-phase opponent draw: guaranteed representation, then weighted surplus.

        One ``configure()`` call assigns one shared opponent to all ``num_lanes``
        lanes for one battle, so the quota is tracked in units of *assignments*
        (``quota_min_games`` is converted to assignments via ``num_lanes``).

        Over the rolling ``quota_window`` most recent assignments, every agent
        row is guaranteed at least ``quota_min_assignments`` assignments — i.e.
        at least ``quota_min_assignments * num_lanes`` games. Any window slots
        beyond the union of quotas are filled by the PSRO-Lite weighted sample
        (or uniform when no sidecar/weights), so prioritization still tilts the
        *surplus* toward weaker matchups without ever starving a policy.

        If the window is too small to satisfy every agent's quota
        (``n_agents * min > window``), the quota is infeasible and we fall back
        to pure weighted sampling rather than starving the surplus entirely.
        """
        names = [row[0] for row in self.config.agents]
        n_agents = len(names)
        min_a = self._quota_min_assignments
        if min_a <= 0 or n_agents == 0:
            return self.config.sample_opponent()
        if n_agents * min_a > self._quota_window:
            # Window can't hold all quotas → can't guarantee representation; fall
            # back to weighted sampling (raise the window or lower the floor).
            return self.config.sample_opponent()
        # Per-agent assignment counts within the current rolling window.
        counts: Dict[str, int] = {nm: 0 for nm in names}
        for nm in self._quota_recent:
            if nm in counts:
                counts[nm] += 1
        under = [nm for nm in names if counts[nm] < min_a]
        if under:
            # Pick the agent furthest below its quota (largest deficit); random
            # tie-break so identical deficits don't lock onto one row.
            max_deficit = max(min_a - counts[nm] for nm in under)
            tied = [nm for nm in under if (min_a - counts[nm]) == max_deficit]
            pick = self.config.rng.choice(tied)
            spec = self.config.sample_opponent_for_agent(pick)
        else:
            # All quotas satisfied → spend the surplus on the weighted sample.
            spec = self.config.sample_opponent()
        self._quota_recent.append(spec.name)
        return spec

    def configure(self, spec: Optional["PolicySpec"] = None) -> "PolicySpec":
        """Activate one sampled (or explicit) opponent for all lanes."""
        self._maybe_refresh_weights()
        sampled = spec is None
        if sampled:
            spec = self._sample_with_quota()
        self.current_spec = spec
        self.current_team = self.config.team_set_for(spec.team_set)
        key = spec.unique_key
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            self._cache[key] = self._make_bundle(spec)
            # LRU-evict the oldest opponents and reclaim their GPU memory.
            evicted = False
            while len(self._cache) > self._cache_size:
                _, old_bundle = self._cache.popitem(last=False)
                self._free_bundle(old_bundle)
                del old_bundle
                evicted = True
            if evicted:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        self._bundle = self._cache[key]
        self._active_key = key
        self._bundle.reset_all()
        return spec

    def _require_bundle(self) -> AmagoBatchedOpponent:
        if self._bundle is None:
            raise RuntimeError(
                "ConfigBatchedOpponent.configure() must run before act()"
            )
        return self._bundle

    def act(self, active: np.ndarray, obs_list: List[dict]) -> np.ndarray:
        return self._require_bundle().act(active, obs_list)

    def observe(self, lane_idx: int, reward: float, action_idx: int) -> None:
        self._require_bundle().observe(lane_idx, reward, action_idx)

    def reset_lanes(self, done_mask: np.ndarray) -> None:
        if self._bundle is not None:
            self._bundle.reset_lanes(done_mask)

    def reset_all(self) -> None:
        if self._bundle is not None:
            self._bundle.reset_all()
