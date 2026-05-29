"""Vectorized pokepy battle env with internal NN opponent."""

from __future__ import annotations

import copy
import json
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import lz4.frame
import numpy as np
import torch

from pokepy.core.constants import PHASE_FORCED_SWITCH
from pokepy.data.loader import (
    GameData,
    IDMappings,
    load_game_data,
    load_id_mappings,
    load_move_effect_data,
)
from pokepy.data.type_charts import MODERN_TYPE_CHART
from pokepy.engine.battle_gen9 import step_battle_gen9, step_forced_switch
from pokepy.env.battle_env import init_battle_state
from pokepy.utils.gen5_prng import Gen5PRNG

from metamon.interface import (
    ActionSpace,
    ObservationSpace,
    RewardFunction,
    UniversalAction,
)
from metamon.env.pokepy_battle.action_adapter import (
    build_illegal_actions_mask,
    legal_action_indices,
    universal_action_to_pokepy,
)
from metamon.env.pokepy_battle.state_adapter import pokepy_state_to_universal
from metamon.env.pokepy_battle.team_adapter import team_set_to_pokepy_dict
from metamon.env.wrappers import TeamSet


def _load_pretrained_model(model):
    from metamon.rl.pretrained import PretrainedModel

    if not isinstance(model, PretrainedModel):
        raise TypeError(f"opponent_model must be PretrainedModel, got {type(model)}")
    return model


def _stack_obs_dicts(obs_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    if not obs_list:
        return {}
    keys = obs_list[0].keys()
    return {k: np.stack([obs[k] for obs in obs_list], axis=0) for k in keys}


def _numpy_obs_to_torch(
    obs: Dict[str, np.ndarray], device: torch.device
) -> Dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v).to(device).unsqueeze(1) for k, v in obs.items()}


class _BattleLane:
    """One parallel battle lane."""

    def __init__(
        self,
        game_data: GameData,
        mappings: IDMappings,
        move_effects,
        battle_format: str,
    ):
        self.game_data = game_data
        self.mappings = mappings
        self.move_effects = move_effects
        self.type_chart = MODERN_TYPE_CHART
        self.battle_format = battle_format
        self.state = None
        self.prng: Optional[Gen5PRNG] = None
        self.seed = 0
        self.last_universal_side0 = None
        self.last_universal_side1 = None
        self.turn_counter = 0
        self.valid_action_counter = 0
        self.invalid_action_counter = 0
        # per-battle trajectory accumulation (parsed-replay format)
        self.traj_states: List[Any] = []
        self.traj_actions: List[int] = []
        self.player_team_file: Optional[str] = None

    def reset(self, team0: dict, team1: dict, seed: int):
        self.seed = int(seed)
        self.prng = Gen5PRNG((self.seed & 0xFFFF, (self.seed >> 16) & 0xFFFF, 0, 0))
        self.state = init_battle_state(team0, team1, self.game_data, self.seed)
        self.turn_counter = 0
        self.valid_action_counter = 0
        self.invalid_action_counter = 0
        self.last_universal_side0 = pokepy_state_to_universal(
            self.state,
            self.game_data,
            self.mappings,
            format_str=self.battle_format,
            player_side=0,
        )
        self.last_universal_side1 = pokepy_state_to_universal(
            self.state,
            self.game_data,
            self.mappings,
            format_str=self.battle_format,
            player_side=1,
        )
        # initial state begins the trajectory; actions are appended each step
        self.traj_states = [self.last_universal_side0]
        self.traj_actions = []

    def step(
        self,
        side0_action: int,
        side1_action: int,
        *,
        tera0: bool = False,
        tera1: bool = False,
    ) -> Tuple[bool, float]:
        assert self.state is not None and self.prng is not None
        if int(self.state.phase) == PHASE_FORCED_SWITCH:
            r0, r1, done = step_forced_switch(
                self.state,
                side0_action,
                side=0,
                game_data=self.game_data,
                move_effects=self.move_effects,
                type_chart=self.type_chart,
                gen5_prng=self.prng,
            )
        else:
            self.turn_counter += 1
            r0, r1, done = step_battle_gen9(
                self.state,
                side0_action,
                side1_action,
                self.game_data,
                self.move_effects,
                self.type_chart,
                self.prng,
                wants_tera0=tera0,
                wants_tera1=tera1,
            )
        return bool(done), float(r0)


class VectorizedPokepyEnv(gym.Env):
    """N-lane gen9 pokepy env: side 0 is the eval agent, side 1 is a batched NN opponent."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        player_team_set: TeamSet,
        opponent_team_set: TeamSet,
        opponent_policy: torch.nn.Module,
        opponent_obs_space: ObservationSpace,
        opponent_action_space: ActionSpace,
        opponent_hidden_state,
        opponent_device: torch.device,
        eval_obs_space: ObservationSpace,
        eval_action_space: ActionSpace,
        eval_reward_function: RewardFunction,
        opponent_reward_function: Optional[RewardFunction] = None,
        batched_envs: int = 8,
        battle_format: str = "gen9ou",
        turn_limit: int = 200,
        opponent_model_name: str = "opponent",
        opponent_sample: bool = True,
        eval_player_side: int = 0,
        save_trajectories_to: Optional[str] = None,
        save_results_to: Optional[str] = None,
        player_username: Optional[str] = None,
        game_data: Optional[GameData] = None,
        mappings: Optional[IDMappings] = None,
    ):
        if not battle_format.startswith("gen9"):
            raise ValueError(
                f"pokepy backend only supports gen9 formats; got {battle_format!r}"
            )
        if eval_player_side not in (0, 1):
            raise ValueError(f"eval_player_side must be 0 or 1; got {eval_player_side}")
        # Which *physical* pokepy side the evaluated agent plays. The opponent
        # plays the other side. Used as a diagnostic to disentangle role-based
        # asymmetries (inference/metrics) from side-based ones (engine).
        self.eval_side = int(eval_player_side)
        self.opp_side = 1 - self.eval_side
        self.player_team_set = player_team_set
        self.opponent_team_set = opponent_team_set
        self.opponent_policy = opponent_policy
        self.opponent_obs_space = opponent_obs_space
        self.opponent_action_space = opponent_action_space
        self.opponent_hidden_state = opponent_hidden_state
        self.opponent_device = opponent_device
        self.opponent_sample = opponent_sample
        # Obs spaces can be *stateful* across `state_to_obs` calls within a
        # battle (e.g. DefaultObservationSpace tracks `revealed_opponents`).
        # A single shared instance would mix state across parallel lanes and
        # across consecutive battles, so each lane gets its own copy that we
        # `.reset()` per battle (mirroring one obs space per PokeEnvWrapper).
        self.eval_obs_space = eval_obs_space  # template; used for gym_space only
        self.eval_obs_spaces = [
            copy.deepcopy(eval_obs_space) for _ in range(int(batched_envs))
        ]
        self.opponent_obs_spaces = [
            copy.deepcopy(opponent_obs_space) for _ in range(int(batched_envs))
        ]
        self.eval_action_space = eval_action_space
        self.eval_reward_function = eval_reward_function
        self.opponent_reward_function = opponent_reward_function
        self.batched_envs = int(batched_envs)
        self.battle_format = battle_format
        self.turn_limit = turn_limit
        self.metamon_battle_format = battle_format
        self.metamon_opponent_name = opponent_model_name
        self.metamon_action_space = eval_action_space
        self.metamon_obs_space = eval_obs_space

        # trajectory / result saving (parsed-replay format, matches PokeEnvWrapper)
        if save_trajectories_to is not None:
            self.save_trajectories_to = os.path.join(
                save_trajectories_to, battle_format
            )
            os.makedirs(self.save_trajectories_to, exist_ok=True)
        else:
            self.save_trajectories_to = None
        self.save_results_to = save_results_to
        self.player_username = player_username or (
            f"MMVec-{''.join(str(random.randint(0, 9)) for _ in range(10))}"
        )
        self._saving = (
            self.save_trajectories_to is not None or self.save_results_to is not None
        )

        self.game_data = game_data or load_game_data()
        self.mappings = mappings or load_id_mappings()
        self.move_effects = load_move_effect_data()

        self.lanes: List[_BattleLane] = [
            _BattleLane(self.game_data, self.mappings, self.move_effects, battle_format)
            for _ in range(self.batched_envs)
        ]
        self.opponent_step_counts = np.zeros((self.batched_envs,), dtype=np.int64)
        # AMAGO's "rl2" vector is `concat([reward, prev_action_one_hot])`, so it is
        # `action_dim + 1` wide with the reward stored at index 0.
        self.opponent_rl2s = np.zeros(
            (self.batched_envs, self.opponent_action_space.gym_space.n + 1),
            dtype=np.float32,
        )
        if self.opponent_hidden_state is None:
            self.opponent_hidden_state = (
                self.opponent_policy.traj_encoder.init_hidden_state(
                    self.batched_envs, self.opponent_device
                )
            )

        # Need one initialized battle before inferring batched observation shapes.
        self._reset_lane(0)
        sample_obs, _ = self._build_side0_obs_and_info(0)
        base_space = self.eval_obs_space.gym_space
        if isinstance(base_space, gym.spaces.Dict):
            spaces = {}
            for k, space in base_space.spaces.items():
                if isinstance(space, gym.spaces.Box):
                    batched_shape = (self.batched_envs,) + space.shape
                    spaces[k] = gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=batched_shape,
                        dtype=space.dtype,
                    )
                else:
                    spaces[k] = space
            self.observation_space = gym.spaces.Dict(spaces)
        else:
            self.observation_space = base_space
        self.observation_space["illegal_actions"] = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.batched_envs, self.eval_action_space.gym_space.n),
            dtype=bool,
        )
        self.action_space = eval_action_space.gym_space

    @property
    def env_name(self) -> str:
        return f"{self.metamon_battle_format}_vs_{self.metamon_opponent_name}"

    def _sample_lane_teams(self) -> Tuple[dict, dict, int, Optional[str]]:
        team0 = team_set_to_pokepy_dict(self.player_team_set, mappings=self.mappings)
        player_team_file = self.player_team_set.most_recent_team_file
        team1 = team_set_to_pokepy_dict(self.opponent_team_set, mappings=self.mappings)
        seed = random.randint(0, 2**31 - 1)
        return team0, team1, seed, player_team_file

    @staticmethod
    def _lane_cached_universal(lane: "_BattleLane", side: int):
        # `last_universal_side{0,1}` are kept in sync with `lane.state` by both
        # `_BattleLane.reset` and the step loop, indexed by *physical* pokepy side.
        return lane.last_universal_side0 if side == 0 else lane.last_universal_side1

    def _build_side0_obs_and_info(self, lane_idx: int) -> Tuple[dict, dict]:
        """Build the evaluated agent's obs/info (it plays physical side `self.eval_side`)."""
        lane = self.lanes[lane_idx]
        self._sync_lane_universal(lane)
        universal = self._lane_cached_universal(lane, self.eval_side)
        obs = self.eval_obs_spaces[lane_idx].state_to_obs(universal)
        illegal = build_illegal_actions_mask(
            self.eval_action_space,
            universal,
            lane.state,
            self.game_data,
            self.mappings,
            player_side=self.eval_side,
        )
        obs["illegal_actions"] = illegal
        legal = legal_action_indices(
            self.eval_action_space,
            universal,
            lane.state,
            self.game_data,
            self.mappings,
            player_side=self.eval_side,
        )
        return obs, {"legal_actions": legal}

    def _build_side1_obs(self, lane_idx: int) -> dict:
        """Build the opponent's obs (it plays physical side `self.opp_side`)."""
        lane = self.lanes[lane_idx]
        self._sync_lane_universal(lane)
        universal = self._lane_cached_universal(lane, self.opp_side)
        obs = self.opponent_obs_spaces[lane_idx].state_to_obs(universal)
        illegal = build_illegal_actions_mask(
            self.opponent_action_space,
            universal,
            lane.state,
            self.game_data,
            self.mappings,
            player_side=self.opp_side,
        )
        obs["illegal_actions"] = illegal
        return obs

    def _sync_lane_universal(self, lane: _BattleLane) -> None:
        lane.last_universal_side0 = pokepy_state_to_universal(
            lane.state,
            self.game_data,
            self.mappings,
            format_str=self.battle_format,
            player_side=0,
        )
        lane.last_universal_side1 = pokepy_state_to_universal(
            lane.state,
            self.game_data,
            self.mappings,
            format_str=self.battle_format,
            player_side=1,
        )

    def _side0_forced_switch_pending(self, lane: _BattleLane) -> bool:
        return int(lane.state.phase) == PHASE_FORCED_SWITCH

    def _amago_decision_pending(self, lane: _BattleLane) -> bool:
        """Outer agent (main AMAGO loop): act on returned obs when True."""
        if self._side0_forced_switch_pending(lane):
            return self.eval_side == 0
        return True

    def _inner_opponent_decision_pending(self, lane: _BattleLane) -> bool:
        """Inner opponent (env batched inference): act when True."""
        if self._side0_forced_switch_pending(lane):
            return self.eval_side == 1
        return True

    def _inner_active_mask(
        self,
        lane_indices: Optional[List[int]] = None,
        *,
        terminated: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Lanes whose inner opponent actually commits a timestep this forward pass."""
        active = np.zeros((self.batched_envs,), dtype=bool)
        lanes = (
            range(self.batched_envs)
            if lane_indices is None
            else (int(i) for i in lane_indices)
        )
        for i in lanes:
            if terminated is not None and terminated[i]:
                continue
            if self._inner_opponent_decision_pending(self.lanes[i]):
                active[i] = True
        return active

    def _snapshot_inner_hidden_state(self, inactive: np.ndarray) -> Optional[dict]:
        if not inactive.any():
            return None
        hs = self.opponent_hidden_state
        idx = np.where(inactive)[0]
        return {
            "idx": idx,
            "seq_lens": hs.seq_lens[idx].clone(),
            "key": hs.key_cache.data[:, idx].clone(),
            "val": hs.val_cache.data[:, idx].clone(),
        }

    def _restore_inner_hidden_state(self, saved: Optional[dict]) -> None:
        if saved is None:
            return
        hs = self.opponent_hidden_state
        idx = saved["idx"]
        hs.seq_lens[idx] = saved["seq_lens"]
        hs.key_cache.data[:, idx] = saved["key"]
        hs.val_cache.data[:, idx] = saved["val"]

    def _opponent_actions(self, active: np.ndarray) -> np.ndarray:
        """Batched inner forward pass. Only ``active`` lanes advance hidden state."""
        inactive = ~active
        saved = self._snapshot_inner_hidden_state(inactive)
        obs_list = [self._build_side1_obs(i) for i in range(self.batched_envs)]
        obs_batch = _stack_obs_dicts(obs_list)
        torch_obs = _numpy_obs_to_torch(obs_batch, self.opponent_device)
        rl2s = (
            torch.from_numpy(self.opponent_rl2s).to(self.opponent_device).unsqueeze(1)
        )
        time_idxs = (
            torch.from_numpy(self.opponent_step_counts)
            .to(self.opponent_device)
            .unsqueeze(1)
            .unsqueeze(1)
        )
        with torch.no_grad():
            actions, self.opponent_hidden_state = self.opponent_policy.get_actions(
                obs=torch_obs,
                rl2s=rl2s,
                time_idxs=time_idxs,
                hidden_state=self.opponent_hidden_state,
                sample=self.opponent_sample,
            )
        self._restore_inner_hidden_state(saved)
        return actions.squeeze(1).cpu().numpy().astype(np.int64)

    def _legal_action_indices_for_lane(
        self,
        lane_idx: int,
        *,
        action_space: ActionSpace,
        player_side: int,
    ) -> list[int]:
        lane = self.lanes[lane_idx]
        universal = self._lane_cached_universal(lane, player_side)
        return legal_action_indices(
            action_space,
            universal,
            lane.state,
            self.game_data,
            self.mappings,
            player_side=player_side,
        )

    def _resolve_legal_action_idx(
        self,
        lane_idx: int,
        action_idx: int,
        *,
        action_space: ActionSpace,
        player_side: int,
        count: bool = False,
    ) -> int:
        """Keep the agent's action if legal, else substitute a random legal one.

        Mirrors ``PokeEnvWrapper.action_to_move`` + ``on_invalid_order``: the
        actor masks illegal actions, but masking can occasionally fail for rare
        edge cases where the true option set is unknown. Rather than crash, we
        fall back to a random legal action (matching the Showdown path). When
        ``count`` is True we record valid/invalid like the main env so the
        "Valid Actions" metric surfaces how often the fallback fires (~99%+).
        """
        lane = self.lanes[lane_idx]
        legal = self._legal_action_indices_for_lane(
            lane_idx, action_space=action_space, player_side=player_side
        )
        action_idx = int(action_idx)
        is_legal = action_idx in legal
        if count:
            if is_legal:
                lane.valid_action_counter += 1
            else:
                lane.invalid_action_counter += 1
        if is_legal or not legal:
            if not legal:
                print(
                    f"[pokepy fallback] lane {lane_idx}: NO legal actions "
                    f"(action={action_idx}, player_side={player_side}, "
                    f"count={count}); passing raw action through"
                )
            return action_idx
        fallback = int(random.choice(legal))
        print(
            f"[pokepy fallback] lane {lane_idx}: illegal action {action_idx} "
            f"-> random legal {fallback} (legal={legal}, player_side={player_side}, "
            f"count={count})"
        )
        return fallback

    def _convert_agent_action(
        self,
        lane_idx: int,
        action_idx: int,
        *,
        action_space: ActionSpace,
        player_side: int,
    ) -> Tuple[int, bool, UniversalAction]:
        lane = self.lanes[lane_idx]
        universal = self._lane_cached_universal(lane, player_side)
        ua = action_space.agent_output_to_action(universal, int(action_idx))
        pokepy_a, tera = universal_action_to_pokepy(
            ua,
            universal,
            lane.state,
            self.mappings,
            player_side=player_side,
        )
        return pokepy_a, tera, ua

    def _update_opponent_rl2(
        self, lane_idx: int, prev_opp_universal, opp_action_idx: int
    ) -> None:
        lane = self.lanes[lane_idx]
        self.opponent_step_counts[lane_idx] += 1
        opp_reward = float(
            self.opponent_reward_function(
                prev_opp_universal,
                self._lane_cached_universal(lane, self.opp_side),
            )
        )
        self.opponent_rl2s[lane_idx] = 0
        self.opponent_rl2s[lane_idx, 0] = opp_reward
        n = self.opponent_action_space.gym_space.n
        if 0 <= opp_action_idx < n:
            self.opponent_rl2s[lane_idx, 1 + opp_action_idx] = 1.0

    def _run_side0_forced_switch(self, lane_idx: int, side0_action: int) -> bool:
        lane = self.lanes[lane_idx]
        done, _ = lane.step(side0_action, 0)
        self._sync_lane_universal(lane)
        return done

    def _apply_inner_forced_switch(self, lane_idx: int, opp_action_idx: int) -> bool:
        prev_opp = self._lane_cached_universal(self.lanes[lane_idx], self.opp_side)
        opp_action_idx = self._resolve_legal_action_idx(
            lane_idx,
            opp_action_idx,
            action_space=self.opponent_action_space,
            player_side=0,
        )
        pokepy_a, _, _ = self._convert_agent_action(
            lane_idx,
            opp_action_idx,
            action_space=self.opponent_action_space,
            player_side=0,
        )
        done = self._run_side0_forced_switch(lane_idx, pokepy_a)
        self._update_opponent_rl2(lane_idx, prev_opp, opp_action_idx)
        return done

    def _catch_up_inner(
        self, lane_indices: Optional[List[int]] = None
    ) -> Dict[int, bool]:
        """Run inner opponent while outer AMAGO waits on side-0 forced switches."""
        if self.eval_side != 1:
            return {}
        lanes = (
            list(range(self.batched_envs))
            if lane_indices is None
            else [int(i) for i in lane_indices]
        )
        ended: Dict[int, bool] = {}
        while True:
            pending = [
                i
                for i in lanes
                if i not in ended
                and self._inner_opponent_decision_pending(self.lanes[i])
                and not self._amago_decision_pending(self.lanes[i])
            ]
            if not pending:
                break
            active = np.zeros((self.batched_envs,), dtype=bool)
            for lane_i in pending:
                active[lane_i] = True
            opp_actions = self._opponent_actions(active)
            for lane_i in pending:
                if self._apply_inner_forced_switch(lane_i, int(opp_actions[lane_i])):
                    ended[lane_i] = True
        return ended

    def _finish_lane(
        self,
        lane_idx: int,
        *,
        done: bool,
        terminated: np.ndarray,
        truncated: np.ndarray,
        infos: List[dict],
    ) -> None:
        lane = self.lanes[lane_idx]
        hit_time_limit = lane.turn_counter > self.turn_limit
        terminated[lane_idx] = done or hit_time_limit
        truncated[lane_idx] = hit_time_limit
        infos[lane_idx]["won"] = bool(lane.state.winner == self.eval_side)
        infos[lane_idx]["valid_action_count"] = lane.valid_action_counter
        infos[lane_idx]["invalid_action_count"] = lane.invalid_action_counter
        if self._saving:
            self._save_lane_trajectory(lane_idx)
        self._reset_lane(lane_idx)
        self.opponent_step_counts[lane_idx] = 0
        self.opponent_rl2s[lane_idx] = 0

    def _reset_lane(self, lane_idx: int):
        team0, team1, seed, player_team_file = self._sample_lane_teams()
        self.lanes[lane_idx].reset(team0, team1, seed)
        self.lanes[lane_idx].player_team_file = player_team_file
        self.opponent_step_counts[lane_idx] = 0
        self.opponent_rl2s[lane_idx] = 0
        # Clear stateful obs-space accumulators (e.g. revealed_opponents) so the
        # new battle in this lane starts from a clean perspective.
        self.eval_obs_spaces[lane_idx].reset()
        self.opponent_obs_spaces[lane_idx].reset()

    def _save_lane_trajectory(self, lane_idx: int) -> None:
        """Write a finished lane's battle to disk in the parsed-replay format."""
        lane = self.lanes[lane_idx]
        won = bool(lane.state.winner == self.eval_side)
        result = "WIN" if won else "LOSS"
        battle_id = "".join(str(random.randint(0, 9)) for _ in range(10))
        timestamp = datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
        opponent_name = self.metamon_opponent_name

        if self.save_trajectories_to is not None:
            filename = (
                f"metamon-{self.metamon_battle_format}-{battle_id}_Unrated_"
                f"{self.player_username}_vs_{opponent_name}_{timestamp}_{result}.json.lz4"
            )
            # matches the format of the parsed replay dataset: one more state than
            # actions, with a blank (-1) final action.
            output_json = {
                "states": [s.to_dict() for s in lane.traj_states],
                "actions": lane.traj_actions + [-1],
            }
            # conservative write to avoid partial files when many lanes finish at once
            path = os.path.join(self.save_trajectories_to, filename)
            temp_path = path + ".tmp"
            with lz4.frame.open(temp_path, "wb") as f:
                f.write(json.dumps(output_json).encode("utf-8"))
            os.rename(temp_path, path)

        if self.save_results_to is not None:
            with open(self.save_results_to, "a") as f:
                f.write(
                    f"{self.player_username},{lane.player_team_file},{opponent_name},"
                    f"{result},{lane.turn_counter},{battle_id}\n"
                )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        for i in range(self.batched_envs):
            self._reset_lane(i)
        self.opponent_hidden_state = (
            self.opponent_policy.traj_encoder.init_hidden_state(
                self.batched_envs, self.opponent_device
            )
        )
        obs_list = []
        infos = []
        for i in range(self.batched_envs):
            obs, info = self._build_side0_obs_and_info(i)
            obs_list.append(obs)
            infos.append(info)
        batched_obs = _stack_obs_dicts(obs_list)
        merged_info = {"legal_actions": [info["legal_actions"] for info in infos]}
        return batched_obs, merged_info

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions).reshape(self.batched_envs)
        rewards = np.zeros((self.batched_envs,), dtype=np.float32)
        terminated = np.zeros((self.batched_envs,), dtype=bool)
        truncated = np.zeros((self.batched_envs,), dtype=bool)
        infos: List[dict] = [{} for _ in range(self.batched_envs)]

        # Inner-only phases left from last step (outer AMAGO was waiting).
        for lane_i in self._catch_up_inner():
            if not terminated[lane_i]:
                self._finish_lane(
                    lane_i,
                    done=True,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                )

        # Single batched inner forward for battle turns (skip lanes in outer-only
        # forced switch). Building the opponent's obs/legal mask is read-only wrt
        # the engine state, so it is safe to run before applying the eval action.
        battle_active = self._inner_active_mask(terminated=terminated)
        opp_actions = self._opponent_actions(battle_active)

        for i in range(self.batched_envs):
            if terminated[i]:
                continue
            lane = self.lanes[i]
            if not self._amago_decision_pending(lane):
                raise RuntimeError(
                    f"lane {i}: outer agent stepped while no decision pending "
                    f"(phase={int(lane.state.phase)}, eval_side={self.eval_side})"
                )

            self._sync_lane_universal(lane)
            prev_state = self._lane_cached_universal(lane, self.eval_side)
            # The trajectory records the agent's *original* choice (matches
            # PokeEnvWrapper), even when the fallback substitutes a legal action.
            orig_action_idx = int(actions[i])

            if self._side0_forced_switch_pending(lane):
                eval_idx = self._resolve_legal_action_idx(
                    i,
                    orig_action_idx,
                    action_space=self.eval_action_space,
                    player_side=0,
                    count=True,
                )
                eval_pokepy_a, _, _ = self._convert_agent_action(
                    i,
                    eval_idx,
                    action_space=self.eval_action_space,
                    player_side=0,
                )
                done = self._run_side0_forced_switch(i, eval_pokepy_a)
            else:
                eval_idx = self._resolve_legal_action_idx(
                    i,
                    orig_action_idx,
                    action_space=self.eval_action_space,
                    player_side=self.eval_side,
                    count=True,
                )
                eval_pokepy_a, eval_tera, _ = self._convert_agent_action(
                    i,
                    eval_idx,
                    action_space=self.eval_action_space,
                    player_side=self.eval_side,
                )
                opp_idx = self._resolve_legal_action_idx(
                    i,
                    int(opp_actions[i]),
                    action_space=self.opponent_action_space,
                    player_side=self.opp_side,
                )
                opp_pokepy_a, opp_tera, _ = self._convert_agent_action(
                    i,
                    opp_idx,
                    action_space=self.opponent_action_space,
                    player_side=self.opp_side,
                )
                prev_opp = self._lane_cached_universal(lane, self.opp_side)
                side_actions = {
                    self.eval_side: (eval_pokepy_a, eval_tera),
                    self.opp_side: (opp_pokepy_a, opp_tera),
                }
                pokepy_a0, tera0 = side_actions[0]
                pokepy_a1, tera1 = side_actions[1]
                done, _ = lane.step(pokepy_a0, pokepy_a1, tera0=tera0, tera1=tera1)
                self._sync_lane_universal(lane)
                self._update_opponent_rl2(i, prev_opp, opp_idx)
                if not done:
                    done = i in self._catch_up_inner([i])

            new_universal = self._lane_cached_universal(lane, self.eval_side)
            rewards[i] = self.eval_reward_function(prev_state, new_universal)
            if self._saving:
                lane.traj_actions.append(orig_action_idx)
                lane.traj_states.append(new_universal)

            if done or lane.turn_counter > self.turn_limit:
                self._finish_lane(
                    i,
                    done=done,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                )

        done_mask = terminated | truncated
        if done_mask.any():
            self.opponent_hidden_state = (
                self.opponent_policy.traj_encoder.reset_hidden_state(
                    self.opponent_hidden_state, done_mask
                )
            )

        # Side-0 forced switches are inner-only when eval_side==1. Resolve any
        # remaining ones before building the outer agent's next obs.
        if self.eval_side == 1:
            self._catch_up_inner()
            for i in range(self.batched_envs):
                if terminated[i]:
                    continue
                if self._inner_opponent_decision_pending(
                    self.lanes[i]
                ) and not self._amago_decision_pending(self.lanes[i]):
                    raise RuntimeError(
                        f"lane {i}: side-0 forced switch still pending before "
                        f"returning outer obs (eval_side=1)"
                    )

        obs_list = []
        legal_actions = []
        for i in range(self.batched_envs):
            obs, info = self._build_side0_obs_and_info(i)
            obs_list.append(obs)
            legal_actions.append(info["legal_actions"])
        batched_obs = _stack_obs_dicts(obs_list)
        merged_info = {"legal_actions": legal_actions}
        for i, info in enumerate(infos):
            for k, v in info.items():
                merged_info.setdefault(k, [])
                while len(merged_info[k]) <= i:
                    merged_info[k].append(None)
                merged_info[k][i] = v
        return batched_obs, rewards, terminated, truncated, merged_info

    def take_long_break(self):
        pass

    def resume_from_break(self):
        pass


def BattlePokepyVectorized(
    battle_format: str,
    observation_space: ObservationSpace,
    action_space: ActionSpace,
    reward_function: RewardFunction,
    team_set: TeamSet,
    opponent_model,
    opponent_checkpoint: Optional[int] = None,
    opponent_policy: Optional[torch.nn.Module] = None,
    opponent_obs_space: Optional[ObservationSpace] = None,
    opponent_action_space: Optional[ActionSpace] = None,
    opponent_hidden_state=None,
    batched_envs: int = 8,
    turn_limit: int = 200,
    opponent_sample: bool = True,
    eval_player_side: int = 0,
    save_trajectories_to: Optional[str] = None,
    save_results_to: Optional[str] = None,
    player_username: Optional[str] = None,
    device: Optional[str] = None,
) -> VectorizedPokepyEnv:
    """Factory: vectorized pokepy env vs a metamon PretrainedModel opponent."""
    if not battle_format.startswith("gen9"):
        raise ValueError(
            f"BattlePokepyVectorized requires a gen9 format; got {battle_format!r}"
        )
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    opponent_model = _load_pretrained_model(opponent_model)
    if opponent_policy is None:
        opponent_agent = opponent_model.initialize_agent(
            checkpoint=opponent_checkpoint, log=False
        )
        opponent_policy = opponent_agent.policy
        opponent_obs_space = opponent_model.observation_space
        opponent_action_space = opponent_model.action_space
    else:
        opponent_obs_space = opponent_obs_space or opponent_model.observation_space
        opponent_action_space = opponent_action_space or opponent_model.action_space
    opponent_policy.to(device)
    opponent_policy.eval()
    if opponent_hidden_state is None:
        opponent_hidden_state = opponent_policy.traj_encoder.init_hidden_state(
            batched_envs, device
        )
    return VectorizedPokepyEnv(
        player_team_set=team_set,
        opponent_team_set=copy.deepcopy(team_set),
        opponent_policy=opponent_policy,
        opponent_obs_space=opponent_obs_space,
        opponent_action_space=opponent_action_space,
        opponent_hidden_state=opponent_hidden_state,
        opponent_device=device,
        eval_obs_space=observation_space,
        eval_action_space=action_space,
        eval_reward_function=reward_function,
        opponent_reward_function=opponent_model.reward_function,
        batched_envs=batched_envs,
        battle_format=battle_format,
        turn_limit=turn_limit,
        opponent_model_name=opponent_model.model_name,
        opponent_sample=opponent_sample,
        eval_player_side=eval_player_side,
        save_trajectories_to=save_trajectories_to,
        save_results_to=save_results_to,
        player_username=player_username,
    )
