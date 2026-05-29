"""Multiprocess orchestrator over VectorizedPokepyEnv worker slices."""

from __future__ import annotations

import copy
from multiprocessing import get_context
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

from metamon.env.pokepy_battle.profiling import StepProfileAccumulator
from metamon.env.pokepy_battle.vector_env import VectorizedPokepyEnv, _stack_obs_dicts
from metamon.env.wrappers import TeamSet
from metamon.interface import ActionSpace, ObservationSpace, RewardFunction
from metamon.rl.pretrained import PretrainedModel, get_pretrained_registry_name


def _resolve_opponent_device(
    opponent_gpu_idx: Optional[int],
    fallback: Optional[torch.device] = None,
) -> torch.device:
    if opponent_gpu_idx is not None:
        return torch.device(f"cuda:{int(opponent_gpu_idx)}")
    if fallback is not None:
        return fallback
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _worker_main(conn, worker_kwargs: dict) -> None:
    """Persistent worker: owns a sub-VectorizedPokepyEnv and serves reset/step."""
    from metamon.env.pokepy_battle.vector_env import _instantiate_pokepy_env
    from metamon.rl.pretrained import get_pretrained_model, get_pretrained_registry_name

    opponent_registry_name = worker_kwargs.pop("opponent_registry_name")
    opponent_model = get_pretrained_model(opponent_registry_name)
    worker_kwargs.pop("num_workers", None)
    worker_kwargs["opponent_model"] = opponent_model
    worker_kwargs["opponent_policy"] = None

    env = _instantiate_pokepy_env(**worker_kwargs)

    while True:
        cmd, payload = conn.recv()
        if cmd == "reset":
            conn.send(env.reset())
        elif cmd == "step":
            conn.send(env.step(payload))
        elif cmd == "profile":
            conn.send(getattr(env, "profile_summary", lambda: {})())
        elif cmd == "close":
            conn.close()
            break
        else:
            raise RuntimeError(f"unknown worker command {cmd!r}")


def _build_batched_observation_space(
    eval_obs_space: ObservationSpace,
    eval_action_space: ActionSpace,
    batched_envs: int,
) -> gym.spaces.Dict:
    base_space = eval_obs_space.gym_space
    if isinstance(base_space, gym.spaces.Dict):
        spaces = {}
        for k, space in base_space.spaces.items():
            if isinstance(space, gym.spaces.Box):
                batched_shape = (batched_envs,) + space.shape
                spaces[k] = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=batched_shape,
                    dtype=space.dtype,
                )
            else:
                spaces[k] = space
        obs_space = gym.spaces.Dict(spaces)
    else:
        obs_space = base_space
    obs_space["illegal_actions"] = gym.spaces.Box(
        low=0,
        high=1,
        shape=(batched_envs, eval_action_space.gym_space.n),
        dtype=bool,
    )
    return obs_space


def _unbatch_obs_dict(obs: Dict[str, np.ndarray]) -> List[Dict[str, np.ndarray]]:
    """Split leading batch dim of stacked obs dict into per-lane dicts."""
    if not obs:
        return []
    batch_size = next(iter(obs.values())).shape[0]
    return [{k: v[i] for k, v in obs.items()} for i in range(batch_size)]


class MultiprocessVectorizedPokepyEnv:
    """Fan ``batched_envs`` lanes across ``num_workers`` persistent subprocesses.

    Presents the same interface as :class:`VectorizedPokepyEnv` for AMAGO
    ``already_vectorized`` mode: ``batched_envs=N``, stacked obs, auto-resets.
    Each worker reloads its own opponent policy onto ``opponent_device``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        player_team_set: TeamSet,
        opponent_team_set: TeamSet,
        opponent_model: PretrainedModel,
        opponent_checkpoint: Optional[int],
        eval_obs_space: ObservationSpace,
        eval_action_space: ActionSpace,
        eval_reward_function: RewardFunction,
        batched_envs: int,
        num_workers: int,
        opponent_gpu_idx: Optional[int] = None,
        opponent_sample: bool = True,
        eval_player_side: int = 0,
        battle_format: str = "gen9ou",
        turn_limit: int = 200,
        save_trajectories_to: Optional[str] = None,
        save_results_to: Optional[str] = None,
        player_username: Optional[str] = None,
    ):
        if int(batched_envs) < int(num_workers):
            raise ValueError(
                f"batched_envs ({batched_envs}) must be >= num_workers ({num_workers})"
            )
        if int(batched_envs) % int(num_workers) != 0:
            raise ValueError(
                f"batched_envs ({batched_envs}) must divide evenly by "
                f"num_workers ({num_workers})"
            )

        self.batched_envs = int(batched_envs)
        self.num_workers = int(num_workers)
        self.lanes_per_worker = self.batched_envs // self.num_workers
        self.eval_side = int(eval_player_side)
        self.opponent_gpu_idx = opponent_gpu_idx
        self.opponent_device = _resolve_opponent_device(opponent_gpu_idx)
        self.battle_format = battle_format
        self.turn_limit = turn_limit
        self.metamon_battle_format = battle_format
        self.metamon_opponent_name = opponent_model.model_name
        self.metamon_action_space = eval_action_space
        self.metamon_obs_space = eval_obs_space
        self.eval_action_space = eval_action_space
        self.action_space = eval_action_space.gym_space
        self.observation_space = _build_batched_observation_space(
            eval_obs_space, eval_action_space, self.batched_envs
        )
        self._profile = StepProfileAccumulator()

        opponent_device_str = str(self.opponent_device)
        base_worker_kwargs = dict(
            battle_format=battle_format,
            observation_space=eval_obs_space,
            action_space=eval_action_space,
            reward_function=eval_reward_function,
            team_set=player_team_set,
            opponent_team_set=copy.deepcopy(opponent_team_set),
            opponent_checkpoint=opponent_checkpoint,
            opponent_obs_space=opponent_model.observation_space,
            opponent_action_space=opponent_model.action_space,
            turn_limit=turn_limit,
            opponent_sample=opponent_sample,
            eval_player_side=eval_player_side,
            save_trajectories_to=save_trajectories_to,
            save_results_to=save_results_to,
            device=opponent_device_str,
            opponent_gpu_idx=opponent_gpu_idx,
            batched_envs=self.lanes_per_worker,
            opponent_registry_name=get_pretrained_registry_name(opponent_model),
            player_username=player_username,
        )

        ctx = get_context("spawn")
        self._conns: List[Any] = []
        self._processes: List[Any] = []
        for wid in range(self.num_workers):
            worker_kwargs = copy.deepcopy(base_worker_kwargs)
            if player_username:
                worker_kwargs["player_username"] = f"{player_username}-w{wid}"
            parent_conn, child_conn = ctx.Pipe(duplex=True)
            proc = ctx.Process(
                target=_worker_main,
                args=(child_conn, worker_kwargs),
                daemon=True,
            )
            proc.start()
            child_conn.close()
            self._conns.append(parent_conn)
            self._processes.append(proc)

    @property
    def env_name(self) -> str:
        return f"{self.metamon_battle_format}_vs_{self.metamon_opponent_name}"

    def profile_summary(self) -> Dict[str, float]:
        summaries = []
        for conn in self._conns:
            conn.send(("profile", None))
            summaries.append(conn.recv())
        merged = dict(self._profile.summary())
        if summaries:
            worker_steps = sum(s.get("steps", 0) for s in summaries)
            merged["worker_lane_loop_s"] = sum(
                s.get("lane_loop_s", 0) for s in summaries
            )
            merged["worker_steps"] = worker_steps
        return merged

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        del seed, options
        obs_list: List[dict] = []
        legal_actions: List[list] = []
        for conn in self._conns:
            conn.send(("reset", None))
            obs, info = conn.recv()
            for lane_obs in _unbatch_obs_dict(obs):
                obs_list.append(lane_obs)
            legal_actions.extend(info["legal_actions"])
        return _stack_obs_dicts(obs_list), {"legal_actions": legal_actions}

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions).reshape(self.batched_envs)
        obs_list: List[dict] = []
        rewards_parts: List[np.ndarray] = []
        terminated_parts: List[np.ndarray] = []
        truncated_parts: List[np.ndarray] = []
        infos_per_lane: List[dict] = [{} for _ in range(self.batched_envs)]
        legal_actions: List[list] = []

        offset = 0
        for conn in self._conns:
            slice_actions = actions[offset : offset + self.lanes_per_worker]
            conn.send(("step", slice_actions))
            obs, rewards, terminated, truncated, info = conn.recv()

            for lane_obs in _unbatch_obs_dict(obs):
                obs_list.append(lane_obs)
            rewards_parts.append(rewards)
            terminated_parts.append(terminated)
            truncated_parts.append(truncated)
            legal_actions.extend(info["legal_actions"])

            for key, values in info.items():
                if key == "legal_actions":
                    continue
                if not isinstance(values, list):
                    continue
                for lane_i, val in enumerate(values):
                    infos_per_lane[offset + lane_i][key] = val
            offset += self.lanes_per_worker

        batched_obs = _stack_obs_dicts(obs_list)
        merged_info: Dict[str, Any] = {"legal_actions": legal_actions}
        for lane_i, lane_info in enumerate(infos_per_lane):
            for key, val in lane_info.items():
                merged_info.setdefault(key, [None] * self.batched_envs)
                merged_info[key][lane_i] = val

        return (
            batched_obs,
            np.concatenate(rewards_parts),
            np.concatenate(terminated_parts),
            np.concatenate(truncated_parts),
            merged_info,
        )

    def take_long_break(self):
        pass

    def resume_from_break(self):
        pass

    def close(self) -> None:
        for conn in self._conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self._processes:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
        self._conns.clear()
        self._processes.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
