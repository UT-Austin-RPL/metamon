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

from pokepy import data, env
from pokepy.core import constants as pokepy_constants
from pokepy.data import loader as pokepy_loader
from pokepy.core.gen_profile import (
    is_format_supported,
    parse_battle_format,
    profile_for_format,
    registered_gens,
)
from pokepy.engine import get_engine, step_forced_switch_for_gen
from pokepy.engine import battle_gen9, switch_requests
from pokepy.utils import gen5_prng

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
from metamon.env.pokepy_battle.profiling import StepProfileAccumulator
from metamon.env.pokepy_battle.state_adapter import pokepy_state_to_universal
from metamon.env.pokepy_battle.team_adapter import team_set_to_pokepy_dict
from metamon.env.wrappers import TeamSet
from metamon.rl.pretrained import PretrainedModel


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
        game_data: pokepy_loader.GameData,
        mappings: pokepy_loader.IDMappings,
        move_effects,
        battle_format: str,
    ):
        self.game_data = game_data
        self.mappings = mappings
        self.move_effects = move_effects
        self.battle_format = battle_format
        self.gen = parse_battle_format(battle_format)
        self.profile = profile_for_format(battle_format)
        self.engine = get_engine(self.gen)
        from pokepy.data.type_charts import load_type_chart_for_gen

        self.type_chart = load_type_chart_for_gen(self.gen)
        self.state = None
        self.prng: Optional[gen5_prng.Gen5PRNG] = None
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
        # Live turn generator (mid-turn subturn yields).
        self.turn_gen = None
        # Paused mid-turn: outer agent must act on this request next step.
        self.pending_switch_request: Optional[switch_requests.SwitchRequest] = None
        # Inner choices collected while waiting for outer (simultaneous requests).
        self.held_inner_switch_choices: Optional[Dict[int, int]] = None
        # Double EOT forced switch: inner slot held until outer commits.
        self.pending_eot_inner_slot: Optional[int] = None
        self.pending_eot_inner_action_idx: Optional[int] = None
        self.pending_eot_inner_prev_opp: Optional[Any] = None
        self.universal_dirty_0 = True
        self.universal_dirty_1 = True
        self.last_side1_obs: Optional[dict] = None

    def invalidate_universal(self) -> None:
        self.universal_dirty_0 = True
        self.universal_dirty_1 = True

    def clear_turn_driver(self) -> None:
        self.turn_gen = None
        self.pending_switch_request = None
        self.held_inner_switch_choices = None
        self.pending_eot_inner_slot = None
        self.pending_eot_inner_action_idx = None
        self.pending_eot_inner_prev_opp = None

    def reset(self, team0: dict, team1: dict, seed: int):
        self.seed = int(seed)
        self.prng = gen5_prng.Gen5PRNG(
            (self.seed & 0xFFFF, (self.seed >> 16) & 0xFFFF, 0, 0)
        )
        self.state = env.init_battle_state(
            team0, team1, self.game_data, self.seed, gen=self.gen
        )
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
        self.clear_turn_driver()
        self.universal_dirty_0 = False
        self.universal_dirty_1 = False
        self.last_side1_obs = None

    def run_forced_switch(self, side: int, pokepy_action: int) -> bool:
        """Apply a forced replacement for ``side``. Returns whether the battle ended."""
        assert self.state is not None and self.prng is not None
        _r0, _r1, done = step_forced_switch_for_gen(
            self.gen,
            self.state,
            pokepy_action,
            side=int(side),
            game_data=self.game_data,
            move_effects=self.move_effects,
            type_chart=self.type_chart,
            gen5_prng=self.prng,
        )
        self.invalidate_universal()
        return bool(done)

    def start_turn(
        self,
        side0_action: int,
        side1_action: int,
        *,
        tera0: bool = False,
        tera1: bool = False,
    ) -> None:
        assert self.state is not None and self.prng is not None
        self.turn_counter += 1
        self.turn_gen = battle_gen9.step_battle_gen9_iter(
            self.state,
            side0_action,
            side1_action,
            self.game_data,
            self.move_effects,
            self.type_chart,
            self.prng,
            wants_tera0=tera0 if self.profile.has_tera else False,
            wants_tera1=tera1 if self.profile.has_tera else False,
            profile=self.profile,
        )
        self.invalidate_universal()


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
        game_data: Optional[pokepy_loader.GameData] = None,
        mappings: Optional[pokepy_loader.IDMappings] = None,
    ):
        if not is_format_supported(battle_format):
            raise ValueError(
                f"pokepy backend does not support format {battle_format!r}; "
                f"registered gens: {sorted(registered_gens())}"
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
        self.gen_profile = profile_for_format(battle_format)
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

        self.game_data = game_data or data.load_game_data(
            gen=parse_battle_format(battle_format)
        )
        self.mappings = mappings or pokepy_loader.load_id_mappings(
            gen=parse_battle_format(battle_format)
        )
        self.move_effects = pokepy_loader.load_move_effect_data(
            gen=parse_battle_format(battle_format)
        )

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
        self._profile: Optional[StepProfileAccumulator] = None

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

    def enable_profiling(self) -> StepProfileAccumulator:
        self._profile = StepProfileAccumulator()
        return self._profile

    def profile_summary(self) -> dict:
        if self._profile is None:
            return {}
        return self._profile.summary()

    @property
    def env_name(self) -> str:
        return f"{self.metamon_battle_format}_vs_{self.metamon_opponent_name}"

    def _sample_lane_teams(self) -> Tuple[dict, dict, int, Optional[str]]:
        team0 = team_set_to_pokepy_dict(
            self.player_team_set, mappings=self.mappings, profile=self.gen_profile
        )
        player_team_file = self.player_team_set.most_recent_team_file
        team1 = team_set_to_pokepy_dict(
            self.opponent_team_set, mappings=self.mappings, profile=self.gen_profile
        )
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
        self._sync_lane_universal(lane, self.eval_side)
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
        self._sync_lane_universal(lane, self.opp_side)
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
        lane.last_side1_obs = obs
        return obs

    def _sync_lane_universal(self, lane: _BattleLane, *sides: int) -> None:
        # Rebuild the requested side(s) from the live engine state. We only
        # build the side(s) actually needed (the Stage 1 win), but we do NOT
        # skip based on a dirty flag: ``available_switches`` / forced-switch
        # legality must always reflect the current active slot, and pokepy's
        # ``get_action_mask`` (live ground truth) can desync from a stale cache
        # if any engine mutation forgets to invalidate. Recomputing the needed
        # side unconditionally keeps obs/action masks correct.
        if not sides:
            sides = (0, 1)
        if 0 in sides:
            lane.last_universal_side0 = pokepy_state_to_universal(
                lane.state,
                self.game_data,
                self.mappings,
                format_str=self.battle_format,
                player_side=0,
            )
        if 1 in sides:
            lane.last_universal_side1 = pokepy_state_to_universal(
                lane.state,
                self.game_data,
                self.mappings,
                format_str=self.battle_format,
                player_side=1,
            )

    def _forced_switch_side(self, lane: _BattleLane) -> int:
        return int(getattr(lane.state, "forced_switch_side", -1))

    def _forced_switch_pending(self, lane: _BattleLane) -> bool:
        return int(lane.state.phase) == pokepy_constants.PHASE_FORCED_SWITCH

    def _side_has_forced_switch(self, lane: _BattleLane, side: int) -> bool:
        if not self._forced_switch_pending(lane):
            return False
        fs = self._forced_switch_side(lane)
        return fs == int(side) or fs == 2

    def _lane_takes_battle_turn(self, lane: _BattleLane) -> bool:
        """True when the lane is about to run a fresh, full battle turn."""
        return lane.pending_switch_request is None and not self._forced_switch_pending(
            lane
        )

    def _amago_decision_pending(self, lane: _BattleLane) -> bool:
        """Outer agent (main AMAGO loop): act on returned obs when True."""
        if lane.pending_switch_request is not None:
            return self.eval_side in lane.pending_switch_request.sides
        if self._forced_switch_pending(lane):
            return self._side_has_forced_switch(lane, self.eval_side)
        return True

    def _inner_opponent_decision_pending(self, lane: _BattleLane) -> bool:
        """Inner opponent (env batched inference): act when True."""
        if lane.pending_switch_request is not None:
            return self.opp_side in lane.pending_switch_request.sides
        if self._forced_switch_pending(lane):
            return self._side_has_forced_switch(lane, self.opp_side)
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
            lane = self.lanes[i]
            if not self._lane_takes_battle_turn(lane):
                continue
            if self._inner_opponent_decision_pending(lane):
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
        obs_list: List[dict] = []
        for i in range(self.batched_envs):
            if active[i]:
                obs = self._build_side1_obs(i)
            else:
                obs = self.lanes[i].last_side1_obs
                if obs is None:
                    obs = self._build_side1_obs(i)
            obs_list.append(obs)
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
        # Rebuild this side from live engine state so legality matches pokepy's
        # action mask (avoids stale active-slot / available_switches desync).
        self._sync_lane_universal(lane, player_side)
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
                if os.environ.get("POKEPY_MASK_DEBUG"):
                    self._dump_empty_legal(lane_idx, action_idx, player_side)
            return action_idx
        fallback = int(random.choice(legal))
        print(
            f"[pokepy fallback] lane {lane_idx}: illegal action {action_idx} "
            f"-> random legal {fallback} (legal={legal}, player_side={player_side}, "
            f"count={count})"
        )
        return fallback

    def _dump_empty_legal(
        self, lane_idx: int, action_idx: int, player_side: int
    ) -> None:
        """Diagnostic: explain why metamon produced an empty legal set while
        pokepy's ground-truth mask may have legal actions. Gated by
        POKEPY_MASK_DEBUG to keep normal runs quiet."""
        from pokepy.engine.action_mask import get_action_mask

        lane = self.lanes[lane_idx]
        self._sync_lane_universal(lane, player_side)
        us = self._lane_cached_universal(lane, player_side)
        state = lane.state
        pk = get_action_mask(state, player_side, self.game_data)
        pk_legal = [i for i in range(len(pk)) if pk[i]]
        active = int(
            state.battle_state[
                pokepy_constants.OFF_META
                + (
                    pokepy_constants.M_ACTIVE0
                    if player_side == 0
                    else pokepy_constants.M_ACTIVE1
                )
            ]
        )
        print("  ---- EMPTY-LEGAL DUMP ----")
        print(
            f"    phase={int(state.phase)} fs_side={getattr(state,'forced_switch_side',-1)} "
            f"fs_slot={getattr(state,'forced_switch_slot',-1)} active_slot={active}"
        )
        print(f"    pokepy_mask_legal={pk_legal}")
        print(f"    us.forced_switch={us.forced_switch} can_tera={us.can_tera}")
        print(
            f"    us.active='{us.player_active_pokemon.name}' "
            f"moves={[m.name for m in us.player_active_pokemon.moves]}"
        )
        print(f"    us.available_switches={[p.name for p in us.available_switches]}")
        trace = []
        for a in sorted(
            UniversalAction.maybe_valid_actions(us), key=lambda x: x.action_idx
        ):
            idx = int(a.action_idx)
            try:
                pa, tera = universal_action_to_pokepy(
                    a, us, state, self.mappings, player_side=player_side
                )
                ok = 0 <= pa < len(pk) and bool(pk[pa])
                trace.append(
                    (idx, "->", pa, "T" if tera else "", "ok" if ok else "PKILLEGAL")
                )
            except Exception as e:  # noqa: BLE001
                trace.append((idx, "CONVERR", repr(e)))
        print(f"    candidate_trace={trace}")
        sp = state.team_species if player_side == 0 else state.opp_species
        hp = [
            int(
                state.battle_state[
                    (
                        pokepy_constants.OFF_SIDE0
                        if player_side == 0
                        else pokepy_constants.OFF_SIDE1
                    )
                    + s * pokepy_constants.POKEMON_SIZE
                    + 1
                ]
            )
            for s in range(6)
        ]
        fl = [
            int(
                state.battle_state[
                    (
                        pokepy_constants.OFF_SIDE0
                        if player_side == 0
                        else pokepy_constants.OFF_SIDE1
                    )
                    + s * pokepy_constants.POKEMON_SIZE
                    + 15
                ]
            )
            & 0x1
            for s in range(6)
        ]
        print(f"    team_species={list(map(int, sp))} hp={hp} fainted={fl}")
        print("  --------------------------")

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

    def _resolve_switch_slot_from_agent(
        self,
        lane_idx: int,
        action_idx: int,
        *,
        action_space: ActionSpace,
        player_side: int,
        count: bool = False,
    ) -> int:
        idx = self._resolve_legal_action_idx(
            lane_idx,
            action_idx,
            action_space=action_space,
            player_side=player_side,
            count=count,
        )
        try:
            pokepy_a, _, _ = self._convert_agent_action(
                lane_idx,
                idx,
                action_space=action_space,
                player_side=player_side,
            )
            return switch_requests.slot_from_pokepy_action(pokepy_a)
        except ValueError:
            lane = self.lanes[lane_idx]
            from pokepy.engine.action_mask import get_action_mask

            mask = get_action_mask(lane.state, player_side, self.game_data)
            legal_pokepy = [a for a in range(4, 10) if mask[a]]
            if not legal_pokepy:
                raise
            fallback = int(random.choice(legal_pokepy))
            print(
                f"[pokepy fallback] lane {lane_idx}: switch action {idx} "
                f"-> pokepy {fallback} (player_side={player_side})"
            )
            return switch_requests.slot_from_pokepy_action(fallback)

    def _run_forced_switch(self, lane_idx: int, side: int, pokepy_action: int) -> bool:
        lane = self.lanes[lane_idx]
        done = lane.run_forced_switch(side, pokepy_action)
        self._sync_lane_universal(lane)
        return done

    def _apply_inner_forced_switch(self, lane_idx: int, opp_action_idx: int) -> bool:
        prev_opp = self._lane_cached_universal(self.lanes[lane_idx], self.opp_side)
        pokepy_a = (
            self._resolve_switch_slot_from_agent(
                lane_idx,
                opp_action_idx,
                action_space=self.opponent_action_space,
                player_side=self.opp_side,
            )
            + 4
        )
        done = self._run_forced_switch(lane_idx, self.opp_side, pokepy_a)
        self._update_opponent_rl2(lane_idx, prev_opp, opp_action_idx)
        return done

    def _catch_up_inner(
        self, lane_indices: Optional[List[int]] = None
    ) -> Dict[int, bool]:
        """Run inner opponent while outer AMAGO waits on forced switches / subturns."""
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
                lane = self.lanes[lane_i]
                if lane.pending_switch_request is not None:
                    result = self._advance_lane_generator_inner_only(
                        lane_i, int(opp_actions[lane_i])
                    )
                    if result[0] is not None and result[0]:
                        ended[lane_i] = True
                    continue
                if self._apply_inner_forced_switch(lane_i, int(opp_actions[lane_i])):
                    ended[lane_i] = True
        return ended

    def _collect_double_ko_inner_choices(self, terminated: np.ndarray) -> None:
        """Batched inner forward for double-KO (forced_switch_side == 2) lanes.

        Both sides owe a replacement on the identical pre-switch state. The
        outer agent's choice arrives via the next ``env.step`` action; the inner
        choice is collected here (no peeking at the outer's pick) and held until
        the outer commits, so switch-ins resolve speed-ordered in
        ``_resolve_eot_forced_switch``.
        """
        pending = [
            i
            for i in range(self.batched_envs)
            if not terminated[i]
            and self._forced_switch_side(self.lanes[i]) == 2
            and self.lanes[i].pending_eot_inner_slot is None
        ]
        if not pending:
            return
        active = np.zeros((self.batched_envs,), dtype=bool)
        for lane_i in pending:
            active[lane_i] = True
        opp_actions = self._opponent_actions(active)
        for lane_i in pending:
            lane = self.lanes[lane_i]
            slot = self._resolve_switch_slot_from_agent(
                lane_i,
                int(opp_actions[lane_i]),
                action_space=self.opponent_action_space,
                player_side=self.opp_side,
            )
            lane.pending_eot_inner_slot = slot
            lane.pending_eot_inner_action_idx = int(opp_actions[lane_i])
            lane.pending_eot_inner_prev_opp = self._lane_cached_universal(
                lane, self.opp_side
            )

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

    def _send_switch_choices_to_generator(
        self, lane_idx: int, choices: Dict[int, int]
    ) -> Optional[Tuple[bool, float]]:
        """Resume a paused turn generator with chosen replacement slot(s)."""
        lane = self.lanes[lane_idx]
        gen = lane.turn_gen
        if gen is None:
            raise RuntimeError(f"lane {lane_idx}: no turn generator to resume")
        try:
            nxt = gen.send(choices)
        except StopIteration as stop:
            lane.clear_turn_driver()
            lane.invalidate_universal()
            _r0, _r1, done = stop.value
            return bool(done), float(_r0)
        lane.invalidate_universal()
        if not isinstance(nxt, switch_requests.SwitchRequest):
            raise RuntimeError(f"lane {lane_idx}: unexpected generator yield {nxt!r}")
        lane.pending_switch_request = nxt
        return None

    def _advance_lane_generator_inner_only(
        self, lane_idx: int, opp_action_idx: int
    ) -> Tuple[Optional[bool], Optional[float]]:
        """Resolve an inner-owned mid-turn subturn and keep advancing."""
        lane = self.lanes[lane_idx]
        req = lane.pending_switch_request
        if req is None or self.opp_side not in req.sides:
            raise RuntimeError(f"lane {lane_idx}: no inner switch request pending")
        prev_opp = self._lane_cached_universal(lane, self.opp_side)
        slot = self._resolve_switch_slot_from_agent(
            lane_idx,
            opp_action_idx,
            action_space=self.opponent_action_space,
            player_side=self.opp_side,
        )
        self._update_opponent_rl2(lane_idx, prev_opp, opp_action_idx)
        choices: Dict[int, int] = dict(lane.held_inner_switch_choices or {})
        choices[self.opp_side] = slot
        if self.eval_side in req.sides:
            lane.held_inner_switch_choices = choices
            lane.pending_switch_request = req
            return None, None
        lane.pending_switch_request = None
        lane.held_inner_switch_choices = None
        result = self._send_switch_choices_to_generator(lane_idx, choices)
        if result is None:
            return self._drain_lane_generator_inner(lane_idx)
        return result

    def _drain_lane_generator_inner(
        self, lane_idx: int
    ) -> Tuple[Optional[bool], Optional[float]]:
        """Keep resolving inner-only subturns until blocked on outer or turn ends."""
        while True:
            lane = self.lanes[lane_idx]
            req = lane.pending_switch_request
            if req is None:
                return None, None
            if self.eval_side in req.sides:
                return None, None
            if self.opp_side not in req.sides:
                return None, None
            active = np.zeros((self.batched_envs,), dtype=bool)
            active[lane_idx] = True
            opp_actions = self._opponent_actions(active)
            result = self._advance_lane_generator_inner_only(
                lane_idx, int(opp_actions[lane_idx])
            )
            if result[0] is not None:
                return result
            if lane.pending_switch_request is None:
                continue
            if self.eval_side in lane.pending_switch_request.sides:
                return None, None

    def _resolve_eot_forced_switch(
        self,
        lane_idx: int,
        eval_action_idx: int,
        *,
        count: bool,
    ) -> bool:
        lane = self.lanes[lane_idx]
        fs = self._forced_switch_side(lane)
        eval_slot = self._resolve_switch_slot_from_agent(
            lane_idx,
            eval_action_idx,
            action_space=self.eval_action_space,
            player_side=self.eval_side,
            count=count,
        )
        eval_pokepy = eval_slot + 4
        if fs == 2:
            inner_slot = lane.pending_eot_inner_slot
            if inner_slot is None:
                raise RuntimeError(
                    f"lane {lane_idx}: double forced switch missing inner choice"
                )
            inner_pokepy = int(inner_slot) + 4
            # Map each player's chosen slot to its physical side, then resolve
            # side 1 BEFORE side 0. step_forced_switch(side=1) stashes a
            # `pending_opp_switch_in` when the opponent (side 0) is still
            # fainted, so the subsequent side-0 call resolves BOTH switch-in
            # abilities in speed order (Showdown runSwitch -> speedSort), rather
            # than physical-side order.
            side_acts = {self.eval_side: eval_pokepy, self.opp_side: inner_pokepy}
            done = self._run_forced_switch(lane_idx, 1, side_acts[1])
            # side 1's replacement may have fainted to hazards (rare): it leaves
            # the lane in FORCED_SWITCH for side 1, which the side-0 call below
            # would otherwise clobber when it resets the phase.
            side1_still_pending = not done and self._side_has_forced_switch(lane, 1)
            if not done:
                done = self._run_forced_switch(lane_idx, 0, side_acts[0])
            if (
                not done
                and side1_still_pending
                and not self._forced_switch_pending(lane)
            ):
                lane.state.phase = np.int8(pokepy_constants.PHASE_FORCED_SWITCH)
                lane.state.forced_switch_side = np.int8(1)
            elif not done and side1_still_pending and self._forced_switch_pending(lane):
                lane.state.forced_switch_side = np.int8(2)
            # Advance inner rl2/hidden-state bookkeeping for its subturn now that
            # the replacement (and its switch-in effects) have resolved.
            if lane.pending_eot_inner_prev_opp is not None:
                self._update_opponent_rl2(
                    lane_idx,
                    lane.pending_eot_inner_prev_opp,
                    int(lane.pending_eot_inner_action_idx or 0),
                )
            lane.pending_eot_inner_slot = None
            lane.pending_eot_inner_action_idx = None
            lane.pending_eot_inner_prev_opp = None
            return done
        side = self.eval_side
        return self._run_forced_switch(lane_idx, side, eval_pokepy)

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions).reshape(self.batched_envs)
        rewards = np.zeros((self.batched_envs,), dtype=np.float32)
        terminated = np.zeros((self.batched_envs,), dtype=bool)
        truncated = np.zeros((self.batched_envs,), dtype=bool)
        infos: List[dict] = [{} for _ in range(self.batched_envs)]
        profile = self._profile

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

        # Double-KO lanes: collect the inner replacement on the pre-switch state
        # before the outer commits (its choice arrives via this step's action).
        self._collect_double_ko_inner_choices(terminated)

        # Single batched inner forward for battle turns (skip lanes in a forced
        # switch / mid-turn subturn). Building the opponent's obs/legal mask is
        # read-only wrt the engine state, so it is safe to run before applying
        # the eval action.
        battle_active = self._inner_active_mask(terminated=terminated)
        if profile is not None:
            t_opp = time.perf_counter()
        opp_actions = self._opponent_actions(battle_active)
        if profile is not None:
            profile.opponent_forward_s += time.perf_counter() - t_opp

        if profile is not None:
            t_lane = time.perf_counter()
        for i in range(self.batched_envs):
            if terminated[i]:
                continue
            lane = self.lanes[i]
            if not self._amago_decision_pending(lane):
                raise RuntimeError(
                    f"lane {i}: outer agent stepped while no decision pending "
                    f"(phase={int(lane.state.phase)}, eval_side={self.eval_side}, "
                    f"fs={self._forced_switch_side(lane)})"
                )

            self._sync_lane_universal(lane, self.eval_side)
            prev_state = self._lane_cached_universal(lane, self.eval_side)
            orig_action_idx = int(actions[i])
            done = False
            if profile is not None:
                profile.lane_steps += 1

            # Resume mid-turn subturn paused for outer agent last step.
            if lane.pending_switch_request is not None and self.eval_side in (
                lane.pending_switch_request.sides
            ):
                eval_slot = self._resolve_switch_slot_from_agent(
                    i,
                    orig_action_idx,
                    action_space=self.eval_action_space,
                    player_side=self.eval_side,
                    count=True,
                )
                choices = dict(lane.held_inner_switch_choices or {})
                choices[self.eval_side] = eval_slot
                lane.pending_switch_request = None
                lane.held_inner_switch_choices = None
                turn_result = self._send_switch_choices_to_generator(i, choices)
                if turn_result is not None:
                    done = turn_result[0]
                    self._sync_lane_universal(lane, self.eval_side)
                    if not done:
                        ended_inner = self._catch_up_inner([i])
                        if i in ended_inner:
                            done = True
                else:
                    inner_done = self._drain_lane_generator_inner(i)
                    if inner_done[0] is not None:
                        done = inner_done[0]
                        self._sync_lane_universal(lane, self.eval_side)
                    elif lane.pending_switch_request is not None:
                        continue

            elif self._forced_switch_pending(lane) and self._side_has_forced_switch(
                lane, self.eval_side
            ):
                done = self._resolve_eot_forced_switch(i, orig_action_idx, count=True)

            elif lane.turn_gen is not None:
                raise RuntimeError(
                    f"lane {i}: turn generator active without pending outer request"
                )

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
                self._sync_lane_universal(lane, self.opp_side)
                prev_opp = self._lane_cached_universal(lane, self.opp_side)
                side_actions = {
                    self.eval_side: (eval_pokepy_a, eval_tera),
                    self.opp_side: (opp_pokepy_a, opp_tera),
                }
                pokepy_a0, tera0 = side_actions[0]
                pokepy_a1, tera1 = side_actions[1]
                lane.start_turn(pokepy_a0, pokepy_a1, tera0=tera0, tera1=tera1)
                self._sync_lane_universal(lane, self.eval_side)
                self._update_opponent_rl2(i, prev_opp, opp_idx)
                try:
                    req = next(lane.turn_gen)
                except StopIteration as stop:
                    lane.clear_turn_driver()
                    lane.invalidate_universal()
                    done = bool(stop.value[2])
                    self._sync_lane_universal(lane, self.eval_side)
                    if not done:
                        ended_inner = self._catch_up_inner([i])
                        if i in ended_inner:
                            done = True
                else:
                    lane.invalidate_universal()
                    if isinstance(req, switch_requests.SwitchRequest):
                        lane.pending_switch_request = req
                        if (
                            self.opp_side in req.sides
                            and self.eval_side not in req.sides
                        ):
                            inner_done = self._drain_lane_generator_inner(i)
                            if inner_done[0] is not None:
                                done = inner_done[0]
                            elif (
                                lane.pending_switch_request is not None
                                and self.eval_side in lane.pending_switch_request.sides
                            ):
                                continue
                        elif self.eval_side in req.sides:
                            continue
                    else:
                        raise RuntimeError(f"lane {i}: unexpected yield {req!r}")

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

        if profile is not None:
            profile.lane_loop_s += time.perf_counter() - t_lane

        done_mask = terminated | truncated
        if done_mask.any():
            self.opponent_hidden_state = (
                self.opponent_policy.traj_encoder.reset_hidden_state(
                    self.opponent_hidden_state, done_mask
                )
            )

        # Resolve inner-only forced switches / held subturns before outer obs.
        self._catch_up_inner()
        for i in range(self.batched_envs):
            if terminated[i]:
                continue
            lane = self.lanes[i]
            if (
                lane.pending_switch_request is not None
                and self.eval_side in lane.pending_switch_request.sides
            ):
                continue
            if self._inner_opponent_decision_pending(
                lane
            ) and not self._amago_decision_pending(lane):
                raise RuntimeError(
                    f"lane {i}: inner forced switch still pending before "
                    f"returning outer obs (eval_side={self.eval_side})"
                )

        if profile is not None:
            t_obs = time.perf_counter()
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
        if profile is not None:
            profile.build_obs_s += time.perf_counter() - t_obs
            profile.step_count += 1
        return batched_obs, rewards, terminated, truncated, merged_info

    def take_long_break(self):
        pass

    def resume_from_break(self):
        pass


class PokepyEnv(VectorizedPokepyEnv):
    """Single-battle (non-vectorized) pokepy env — for debugging and as a reference.

    Presents a standard single-env gym interface (no leading batch dimension on
    obs/rewards/dones). Wrap with :class:`~metamon.rl.metamon_to_amago.MetamonAMAGOWrapper`
    and run AMAGO in ``env_mode="sync"`` with ``parallel_actors=1`` — NOT
    ``already_vectorized``.

    Inherits low-level adapters from :class:`VectorizedPokepyEnv` (obs building,
    action conversion, NN inference, forced-switch resolution) so behaviour stays
    identical to the batched path. The control flow is rewritten as a short linear
    loop with none of the batching machinery.

    ``eval_player_side`` (0 or 1) selects which physical pokepy side the outer
    AMAGO agent plays; the inner opponent plays the other side. Read top to bottom:

      * :meth:`step` — apply the eval agent's action, score it, recycle on done.
      * :meth:`_advance` — route that action to whichever decision point the eval
        agent owes (fresh move, mid-turn replacement, or EOT forced switch).
      * :meth:`_drive` — advance the engine, querying the inner opponent inline at
        each of its own decision points, until the eval agent is needed again.
    """

    def __init__(self, *args, **kwargs):
        batched = kwargs.get("batched_envs", 1)
        if int(batched) != 1:
            raise ValueError(
                f"PokepyEnv is single-battle; batched_envs must be 1, got {batched}"
            )
        kwargs["batched_envs"] = 1
        super().__init__(*args, **kwargs)
        self._init_sync_observation_space()

    def _init_sync_observation_space(self) -> None:
        """Unbatched gym spaces for AMAGO ``sync`` mode (``parallel_actors=1``).

        The parent always prefixes observation shapes with ``batched_envs``; undo
        that here so AMAGO's SequenceWrapper adds the sole batch dim itself.
        """
        base_space = self.eval_obs_space.gym_space
        if isinstance(base_space, gym.spaces.Dict):
            spaces = {}
            for k, space in base_space.spaces.items():
                if isinstance(space, gym.spaces.Box):
                    spaces[k] = gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=space.shape,
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
            shape=(self.eval_action_space.gym_space.n,),
            dtype=bool,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self._reset_lane(0)
        self.opponent_hidden_state = (
            self.opponent_policy.traj_encoder.init_hidden_state(1, self.opponent_device)
        )
        obs, info = self._build_side0_obs_and_info(0)
        return obs, {"legal_actions": info["legal_actions"]}

    @property
    def lane(self) -> "_BattleLane":
        return self.lanes[0]

    # ---- inner-opponent queries (unbatched) ------------------------------

    def _inner_move(self) -> int:
        """Ask the inner opponent for an action; advance its hidden state once.

        Reuses the inherited batched forward with a single active lane (the
        inactive-lane snapshot/restore is a no-op), so inference is identical to
        the vectorized path — only without the batch.
        """
        active = np.zeros((1,), dtype=bool)
        active[0] = True
        return int(self._opponent_actions(active)[0])

    def _inner_switch_choice(self) -> int:
        """Ask the inner opponent for a replacement slot at its own subturn.

        Captures the opponent's pre-decision view and advances its rl2 vector,
        matching how the inner side is scored everywhere else.
        """
        prev_opp = self._lane_cached_universal(self.lane, self.opp_side)
        raw = self._inner_move()
        slot = self._resolve_switch_slot_from_agent(
            0, raw, action_space=self.opponent_action_space, player_side=self.opp_side
        )
        self._update_opponent_rl2(0, prev_opp, raw)
        return slot

    # ---- the battle turn ------------------------------------------------

    def _start_turn(self, eval_action: int) -> bool:
        """Both policies choose a move; begin the turn. Returns ``done``.

        If a mid-turn pivot (U-turn / Eject) fires, the suspended generator's
        ``SwitchRequest`` is parked on the lane for :meth:`_drive` to handle.
        """
        lane = self.lane
        eval_idx = self._resolve_legal_action_idx(
            0,
            eval_action,
            action_space=self.eval_action_space,
            player_side=self.eval_side,
            count=True,
        )
        eval_pokepy, eval_tera, _ = self._convert_agent_action(
            0, eval_idx, action_space=self.eval_action_space, player_side=self.eval_side
        )
        prev_opp = self._lane_cached_universal(lane, self.opp_side)
        opp_idx = self._resolve_legal_action_idx(
            0,
            self._inner_move(),
            action_space=self.opponent_action_space,
            player_side=self.opp_side,
        )
        opp_pokepy, opp_tera, _ = self._convert_agent_action(
            0,
            opp_idx,
            action_space=self.opponent_action_space,
            player_side=self.opp_side,
        )
        # Map (eval, opp) choices onto physical sides 0/1 for the engine.
        by_side = {
            self.eval_side: (eval_pokepy, eval_tera),
            self.opp_side: (opp_pokepy, opp_tera),
        }
        (a0, t0), (a1, t1) = by_side[0], by_side[1]
        lane.start_turn(a0, a1, tera0=t0, tera1=t1)
        self._sync_lane_universal(lane)
        self._update_opponent_rl2(0, prev_opp, opp_idx)

        try:
            req = next(lane.turn_gen)
        except StopIteration as stop:
            lane.clear_turn_driver()
            self._sync_lane_universal(lane)
            return bool(stop.value[2])
        if not isinstance(req, switch_requests.SwitchRequest):
            raise RuntimeError(f"unexpected generator yield {req!r}")
        lane.pending_switch_request = req
        return False

    def _drive(self) -> bool:
        """Advance the engine + inner opponent until the eval agent is needed.

        Returns ``done``. When it returns False the lane is paused exactly where
        the eval agent owes its next decision.
        """
        lane = self.lane
        while True:
            req = lane.pending_switch_request
            if req is not None:
                if self.eval_side in req.sides:
                    # Eval agent must choose. For a simultaneous (double) pivot,
                    # take the inner pick now on the identical pre-switch state
                    # so neither side peeks at the other's replacement.
                    if self.opp_side in req.sides and (
                        lane.held_inner_switch_choices is None
                        or self.opp_side not in lane.held_inner_switch_choices
                    ):
                        lane.held_inner_switch_choices = {
                            self.opp_side: self._inner_switch_choice()
                        }
                    return False
                # Inner-only pivot: pick a replacement, resume the turn, repeat.
                result = self._send_switch_choices_to_generator(
                    0, {self.opp_side: self._inner_switch_choice()}
                )
                self._sync_lane_universal(lane)
                if result is not None and result[0]:
                    return True
                continue

            if self._forced_switch_pending(lane):
                if self._side_has_forced_switch(lane, self.eval_side):
                    # Eval agent owes an end-of-turn switch. On a double-KO,
                    # collect the inner replacement now on the same pre-switch
                    # state; the eval choice arrives on the next ``step``.
                    if (
                        self._forced_switch_side(lane) == 2
                        and lane.pending_eot_inner_slot is None
                    ):
                        self._collect_double_ko_inner_choices(
                            np.zeros((1,), dtype=bool)
                        )
                    return False
                # Inner-only end-of-turn forced switch.
                if self._apply_inner_forced_switch(0, self._inner_move()):
                    return True
                continue

            # Battle phase, nothing pending: the eval agent owes its next move.
            return False

    def _advance(self, eval_action: int) -> bool:
        """Apply the eval agent's action at its current decision point."""
        lane = self.lane
        req = lane.pending_switch_request

        if req is not None and self.eval_side in req.sides:
            # Eval agent's mid-turn replacement (plus any held inner pick).
            eval_slot = self._resolve_switch_slot_from_agent(
                0,
                eval_action,
                action_space=self.eval_action_space,
                player_side=self.eval_side,
                count=True,
            )
            choices = dict(lane.held_inner_switch_choices or {})
            choices[self.eval_side] = eval_slot
            lane.pending_switch_request = None
            lane.held_inner_switch_choices = None
            result = self._send_switch_choices_to_generator(0, choices)
            self._sync_lane_universal(lane)
            if result is not None and result[0]:
                return True
            return self._drive()

        if self._forced_switch_pending(lane) and self._side_has_forced_switch(
            lane, self.eval_side
        ):
            if self._resolve_eot_forced_switch(0, eval_action, count=True):
                return True
            return self._drive()

        if self._start_turn(eval_action):
            return True
        return self._drive()

    def step(self, action):
        action = int(np.asarray(action).reshape(-1)[0])
        lane = self.lane
        terminated = False
        truncated = False
        info: dict = {}

        if not self._amago_decision_pending(lane):
            raise RuntimeError(
                f"PokepyEnv: stepped while no eval decision pending "
                f"(phase={int(lane.state.phase)}, eval_side={self.eval_side}, "
                f"fs={self._forced_switch_side(lane)})"
            )

        self._sync_lane_universal(lane)
        prev_state = self._lane_cached_universal(lane, self.eval_side)
        done = self._advance(action)

        new_state = self._lane_cached_universal(lane, self.eval_side)
        reward = float(self.eval_reward_function(prev_state, new_state))
        if self._saving:
            lane.traj_actions.append(action)
            lane.traj_states.append(new_state)

        if done or lane.turn_counter > self.turn_limit:
            term_arr = np.zeros((1,), dtype=bool)
            trunc_arr = np.zeros((1,), dtype=bool)
            infos = [info]
            self._finish_lane(
                0, done=done, terminated=term_arr, truncated=trunc_arr, infos=infos
            )
            info = infos[0]
            terminated = bool(term_arr[0])
            truncated = bool(trunc_arr[0])
            self.opponent_hidden_state = (
                self.opponent_policy.traj_encoder.reset_hidden_state(
                    self.opponent_hidden_state, term_arr | trunc_arr
                )
            )

        obs, obs_info = self._build_side0_obs_and_info(0)
        info["legal_actions"] = obs_info["legal_actions"]
        return obs, reward, terminated, truncated, info


def _resolve_opponent_device(
    device: Optional[str],
    opponent_gpu_idx: Optional[int],
) -> torch.device:
    if opponent_gpu_idx is not None:
        return torch.device(f"cuda:{int(opponent_gpu_idx)}")
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def _instantiate_pokepy_env(
    *,
    battle_format: str,
    observation_space: ObservationSpace,
    action_space: ActionSpace,
    reward_function: RewardFunction,
    team_set: TeamSet,
    opponent_model: PretrainedModel,
    opponent_checkpoint: Optional[int] = None,
    opponent_policy: Optional[torch.nn.Module] = None,
    opponent_obs_space: Optional[ObservationSpace] = None,
    opponent_action_space: Optional[ActionSpace] = None,
    opponent_hidden_state=None,
    opponent_team_set: Optional[TeamSet] = None,
    batched_envs: int = 8,
    turn_limit: int = 200,
    opponent_sample: bool = True,
    eval_player_side: int = 0,
    save_trajectories_to: Optional[str] = None,
    save_results_to: Optional[str] = None,
    player_username: Optional[str] = None,
    device: Optional[str] = None,
    opponent_gpu_idx: Optional[int] = None,
) -> VectorizedPokepyEnv:
    """Build a single-process pokepy env (used by workers and num_workers=1)."""
    if not is_format_supported(battle_format):
        raise ValueError(
            f"pokepy env requires a registered format; got {battle_format!r} "
            f"(registered gens: {sorted(registered_gens())})"
        )
    if not isinstance(opponent_model, PretrainedModel):
        raise TypeError(
            f"opponent_model must be PretrainedModel, got {type(opponent_model)}"
        )

    opponent_device = _resolve_opponent_device(device, opponent_gpu_idx)
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
    opponent_policy.to(opponent_device)
    opponent_policy.eval()
    if opponent_hidden_state is None:
        opponent_hidden_state = opponent_policy.traj_encoder.init_hidden_state(
            batched_envs, opponent_device
        )

    env_cls = PokepyEnv if batched_envs == 1 else VectorizedPokepyEnv
    return env_cls(
        player_team_set=team_set,
        opponent_team_set=opponent_team_set or copy.deepcopy(team_set),
        opponent_policy=opponent_policy,
        opponent_obs_space=opponent_obs_space,
        opponent_action_space=opponent_action_space,
        opponent_hidden_state=opponent_hidden_state,
        opponent_device=opponent_device,
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


def BattlePokepyVectorized(
    battle_format: str,
    observation_space: ObservationSpace,
    action_space: ActionSpace,
    reward_function: RewardFunction,
    team_set: TeamSet,
    opponent_model: PretrainedModel,
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
    num_workers: int = 1,
    opponent_gpu_idx: Optional[int] = None,
):
    """Factory: vectorized pokepy env vs a metamon PretrainedModel opponent."""
    if int(num_workers) > 1:
        if int(batched_envs) == 1:
            raise ValueError("num_workers > 1 requires batched_envs > 1")
        if opponent_policy is not None:
            raise ValueError(
                "multiprocess pokepy env cannot accept a prebuilt opponent_policy; "
                "pass opponent_model + opponent_checkpoint so workers reload"
            )
        from metamon.env.pokepy_battle.multiprocess_env import (
            MultiprocessVectorizedPokepyEnv,
        )

        return MultiprocessVectorizedPokepyEnv(
            player_team_set=team_set,
            opponent_team_set=copy.deepcopy(team_set),
            opponent_model=opponent_model,
            opponent_checkpoint=opponent_checkpoint,
            eval_obs_space=observation_space,
            eval_action_space=action_space,
            eval_reward_function=reward_function,
            batched_envs=batched_envs,
            num_workers=int(num_workers),
            opponent_gpu_idx=opponent_gpu_idx,
            opponent_sample=opponent_sample,
            eval_player_side=eval_player_side,
            battle_format=battle_format,
            turn_limit=turn_limit,
            save_trajectories_to=save_trajectories_to,
            save_results_to=save_results_to,
            player_username=player_username,
        )

    return _instantiate_pokepy_env(
        battle_format=battle_format,
        observation_space=observation_space,
        action_space=action_space,
        reward_function=reward_function,
        team_set=team_set,
        opponent_model=opponent_model,
        opponent_checkpoint=opponent_checkpoint,
        opponent_policy=opponent_policy,
        opponent_obs_space=opponent_obs_space,
        opponent_action_space=opponent_action_space,
        opponent_hidden_state=opponent_hidden_state,
        batched_envs=batched_envs,
        turn_limit=turn_limit,
        opponent_sample=opponent_sample,
        eval_player_side=eval_player_side,
        save_trajectories_to=save_trajectories_to,
        save_results_to=save_results_to,
        player_username=player_username,
        device=device,
        opponent_gpu_idx=opponent_gpu_idx,
    )
