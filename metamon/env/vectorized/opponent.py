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
"""

from __future__ import annotations

from abc import ABC, abstractmethod
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
    ):
        from metamon.rl.evaluate.opponent_pool import OpponentPoolConfig

        if not isinstance(config, OpponentPoolConfig):
            raise TypeError(f"config must be OpponentPoolConfig, got {type(config)}")
        self.config = config
        self.num_lanes = int(num_lanes)
        self.device = device
        self.sample = sample
        self.current_spec: Optional["PolicySpec"] = None
        self.current_team: Optional["TeamSet"] = None
        self._active_key: Optional[str] = None
        self._bundle: Optional[AmagoBatchedOpponent] = None
        self._cache: Dict[str, AmagoBatchedOpponent] = {}

    def _make_bundle(self, spec: "PolicySpec") -> AmagoBatchedOpponent:
        from metamon.rl.pretrained import get_pretrained_model

        model = get_pretrained_model(spec.model_name)
        agent = model.initialize_agent(
            checkpoint=spec.checkpoint,
            log=False,
            action_temperature=spec.temperature,
        )
        action_dim = model.action_space.gym_space.n
        if self._cache:
            existing = next(iter(self._cache.values()))
            if existing.action_dim != action_dim:
                raise ValueError(
                    "Opponent pool models must share action_dim; "
                    f"got {action_dim} for {spec.model_name}"
                )
        agent.policy.to(self.device)
        agent.policy.eval()
        return AmagoBatchedOpponent(
            policy=agent.policy,
            device=self.device,
            num_lanes=self.num_lanes,
            action_dim=action_dim,
            sample=self.sample,
        )

    def configure(self, spec: Optional["PolicySpec"] = None) -> "PolicySpec":
        """Activate one sampled (or explicit) opponent for all lanes."""
        if spec is None:
            spec = self.config.sample_opponent()
        self.current_spec = spec
        self.current_team = self.config.team_set_for(spec.team_set)
        key = spec.unique_key
        if key not in self._cache:
            self._cache[key] = self._make_bundle(spec)
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
