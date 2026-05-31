"""Batched in-the-loop opponents for the vectorized Showdown env.

The env synchronizes all lanes at each decision cycle and asks the opponent for
one action per *active* lane in a single call, so opponent inference is amortized
across the batch. Two implementations are provided:

  * :class:`RandomBatchedOpponent` — no NN; returns random action indices (the env
    repairs any illegal pick against the legal mask). Useful for smoke tests.
  * :class:`AmagoBatchedOpponent` — wraps a metamon ``PretrainedModel`` policy,
    reproducing the batched forward from
    ``metamon.env.pokepy_battle.vector_env._opponent_actions`` (per-lane hidden
    state with snapshot/restore for inactive lanes, rl2 = [reward, prev_action]).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import torch

from .obs_utils import numpy_obs_to_torch, stack_obs_dicts


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
    """Wrap an AMAGO policy as a batched opponent (adapted from pokepy env)."""

    def __init__(
        self,
        policy: torch.nn.Module,
        device: torch.device,
        num_lanes: int,
        action_dim: int,
        hidden_state=None,
        sample: bool = True,
    ):
        self.policy = policy
        self.device = device
        self.num_lanes = int(num_lanes)
        self.action_dim = int(action_dim)
        self.sample = sample
        # rl2 = concat([reward, prev_action_one_hot]); width = action_dim + 1.
        self.rl2s = np.zeros((self.num_lanes, self.action_dim + 1), dtype=np.float32)
        self.step_counts = np.zeros((self.num_lanes,), dtype=np.int64)
        if hidden_state is None:
            hidden_state = self.policy.traj_encoder.init_hidden_state(
                self.num_lanes, self.device
            )
        self.hidden_state = hidden_state

    def _snapshot_hidden(self, inactive: np.ndarray) -> Optional[dict]:
        if not inactive.any():
            return None
        hs = self.hidden_state
        idx = np.where(inactive)[0]
        return {
            "idx": idx,
            "seq_lens": hs.seq_lens[idx].clone(),
            "key": hs.key_cache.data[:, idx].clone(),
            "val": hs.val_cache.data[:, idx].clone(),
        }

    def _restore_hidden(self, saved: Optional[dict]) -> None:
        if saved is None:
            return
        hs = self.hidden_state
        idx = saved["idx"]
        hs.seq_lens[idx] = saved["seq_lens"]
        hs.key_cache.data[:, idx] = saved["key"]
        hs.val_cache.data[:, idx] = saved["val"]

    def act(self, active: np.ndarray, obs_list: List[dict]) -> np.ndarray:
        saved = self._snapshot_hidden(~active)
        obs_batch = stack_obs_dicts(obs_list)
        torch_obs = numpy_obs_to_torch(obs_batch, self.device)
        rl2s = torch.from_numpy(self.rl2s).to(self.device).unsqueeze(1)
        time_idxs = (
            torch.from_numpy(self.step_counts).to(self.device).unsqueeze(1).unsqueeze(1)
        )
        with torch.no_grad():
            actions, self.hidden_state = self.policy.get_actions(
                obs=torch_obs,
                rl2s=rl2s,
                time_idxs=time_idxs,
                hidden_state=self.hidden_state,
                sample=self.sample,
            )
        self._restore_hidden(saved)
        return actions.squeeze(1).cpu().numpy().astype(np.int64)

    def observe(self, lane_idx: int, reward: float, action_idx: int) -> None:
        self.step_counts[lane_idx] += 1
        self.rl2s[lane_idx] = 0.0
        self.rl2s[lane_idx, 0] = float(reward)
        if 0 <= action_idx < self.action_dim:
            self.rl2s[lane_idx, 1 + action_idx] = 1.0

    def reset_lanes(self, done_mask: np.ndarray) -> None:
        if not done_mask.any():
            return
        for i in np.where(done_mask)[0]:
            self.step_counts[i] = 0
            self.rl2s[i] = 0.0
        self.hidden_state = self.policy.traj_encoder.reset_hidden_state(
            self.hidden_state, torch.as_tensor(done_mask, device=self.device)
        )

    def reset_all(self) -> None:
        self.step_counts[:] = 0
        self.rl2s[:] = 0.0
        self.hidden_state = self.policy.traj_encoder.init_hidden_state(
            self.num_lanes, self.device
        )
