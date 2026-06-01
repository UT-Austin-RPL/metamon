"""AMAGO policy drivers that mirror QueueOnLocalLadder / MetamonAMAGOWrapper semantics.

Each :class:`AmagoLadderPolicyDriver` tracks per-lane ``rl2`` and ``time_idx`` the same
way ``AMAGOEnv`` + ``SequenceWrapper.current_timestep`` do during
``Experiment.interact`` (``rl2 = concat(reward, prev_action_one_hot)``,
``time_idx = step_count``).  Vectorized eval uses one driver for the evaluated agent
and the opponent stack uses the same class so both policies see identical inputs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .obs_utils import numpy_obs_to_torch, stack_obs_dicts, unstack_obs_dicts


class AmagoLadderPolicyDriver:
    """Batched policy forward matching AMAGO ladder rollouts."""

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
        """Return action indices; only ``active`` lanes advance recurrent state."""
        actions = np.zeros((self.num_lanes,), dtype=np.int64)
        if not active.any():
            return actions

        saved = self._snapshot_hidden(~active)
        obs_batch = stack_obs_dicts(obs_list)
        torch_obs = numpy_obs_to_torch(obs_batch, self.device)
        rl2s = torch.from_numpy(self.rl2s).to(self.device).unsqueeze(1)
        time_idxs = (
            torch.from_numpy(self.step_counts).to(self.device).unsqueeze(1).unsqueeze(1)
        )
        with torch.no_grad():
            act_out, self.hidden_state = self.policy.get_actions(
                obs=torch_obs,
                rl2s=rl2s,
                time_idxs=time_idxs,
                hidden_state=self.hidden_state,
                sample=self.sample,
            )
        self._restore_hidden(saved)
        out = act_out.squeeze(1).cpu().numpy().astype(np.int64).reshape(-1)
        actions[active] = out[active]
        return actions

    def observe(self, lane_idx: int, reward: float, action_idx: int) -> None:
        """Record env feedback after a committed decision (matches AMAGOEnv.step)."""
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


def vectorized_ladder_eval(
    policy: torch.nn.Module,
    device: torch.device,
    make_env: callable,
    total_battles: int,
    action_dim: int,
    sample: bool = True,
    timesteps: Optional[int] = None,
) -> Dict[str, Any]:
    """Run vectorized Showdown eval with symmetric ladder-style policy drivers.

    The evaluated agent and in-the-loop opponent both use :class:`AmagoLadderPolicyDriver`
    with identical ``get_actions`` bookkeeping.  This replaces AMAGO ``interact`` for
    batched Showdown eval so the eval agent is not on a different code path than the
    opponent.
    """
    from .opponent import AmagoBatchedOpponent

    wrapped = make_env()
    env = wrapped._metamon_env if hasattr(wrapped, "_metamon_env") else wrapped.env
    num_lanes = env.batched_envs
    eval_actor = AmagoBatchedOpponent(
        policy=policy,
        device=device,
        num_lanes=num_lanes,
        action_dim=action_dim,
        sample=sample,
    )
    policy.eval()
    env.bind_eval_policy(eval_actor)

    if timesteps is None:
        timesteps = max(total_battles * 250 // num_lanes, 250)

    obs, _info = env.reset()
    episodes_done = 0
    steps = 0
    returns: List[float] = []
    wins: List[float] = []
    valid_ratios: List[float] = []
    episode_return = np.zeros((num_lanes,), dtype=np.float64)

    while episodes_done < total_battles and steps < timesteps:
        obs_list = unstack_obs_dicts(obs)
        active = np.ones((num_lanes,), dtype=bool)
        actions = eval_actor.act(active, obs_list)
        obs, rewards, terminated, truncated, info = env.step(actions)
        steps += 1
        episode_return += rewards

        done = terminated | truncated
        if done.any():
            done_idx = np.where(done)[0]
            for i in done_idx:
                episodes_done += 1
                returns.append(float(episode_return[i]))
                episode_return[i] = 0.0
                won = info.get("won")
                if isinstance(won, list):
                    if won[i] is not None:
                        wins.append(float(won[i]))
                elif won is not None:
                    wins.append(float(won))
                valid = info.get("valid_action_count")
                invalid = info.get("invalid_action_count")
                if (
                    isinstance(valid, list)
                    and isinstance(invalid, list)
                    and valid[i] is not None
                    and invalid[i] is not None
                ):
                    denom = valid[i] + invalid[i]
                    if denom > 0:
                        valid_ratios.append(valid[i] / denom)

    env.close()

    env_name = getattr(wrapped, "env_name", "metamon")
    results: Dict[str, Any] = {
        f"Average Total Return in {env_name}": (
            float(np.mean(returns)) if returns else 0.0
        ),
        "Average Total Return (Across All Env Names)": (
            float(np.mean(returns)) if returns else 0.0
        ),
    }
    if returns:
        results[f"Bottom Quintile Total Return in {env_name}"] = float(
            np.percentile(returns, 20)
        )
    if wins:
        results[f"Average Win Rate in {env_name}"] = float(np.mean(wins))
    if valid_ratios:
        results[f"Average Valid Actions in {env_name}"] = float(np.mean(valid_ratios))
    return results
