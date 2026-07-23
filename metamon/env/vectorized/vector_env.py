"""Vectorized Showdown battle env with an in-the-loop batched NN opponent.

N battles run inside one Node process (``battle_host.js``) hosting N
``BattleStream``s; Python advances them in lockstep and batches the opponent's
neural-network forward pass across lanes. By default the evaluated agent plays
Showdown ``p1`` and the in-the-loop opponent plays ``p2``; ``eval_player_side``
selects the other physical side instead (diagnostic for side-based asymmetries).

Control flow is *request driven* (Showdown resolves all mid-turn sequencing for
us): each ``step`` consumes the agent's action for the lane's current eval-side
decision (a normal move or a forced switch), applies the opponent's decision for
the same cycle when there is one, advances the sim, and then auto-resolves any
cycles where only the opponent must act before parking each lane back at the next
eval-side decision.

The agent-facing contracts (``ObservationSpace``, ``DefaultActionSpace`` →
``Discrete(13)``, ``RewardFunction``, ``illegal_actions`` in the obs) match
``VectorizedPokepyEnv`` so this env is a drop-in alternative.
"""

from __future__ import annotations

import copy
import json
import os
import random
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import lz4.frame
import numpy as np

from metamon.interface import (
    ActionSpace,
    ObservationSpace,
    RewardFunction,
    UniversalState,
)

from .action_adapter import DEFAULT_CHOICE, action_idx_to_choice
from .lane import (
    AGENT_KINDS,
    KIND_FORCESWITCH,
    KIND_MOVE,
    KIND_TEAMPREVIEW,
    SIDES,
    StreamBattleLane,
)
from .obs_utils import stack_obs_dicts
from .opponent import BatchedOpponent, ConfigBatchedOpponent, RandomBatchedOpponent
from metamon.rl.evaluate.opponent_pool import OpponentPoolConfig, load_opponent_pool
from .sim_process import ShowdownSimProcess, ShowdownSimProcessError, make_sim_process
from .team_adapter import player_spec


class VectorizedShowdownEnv(gym.Env):
    """N-lane Showdown env with a batched in-the-loop NN opponent."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        player_team_set,
        opponent_team_set,
        opponent: BatchedOpponent,
        opponent_obs_space: ObservationSpace,
        opponent_action_space: ActionSpace,
        eval_obs_space: ObservationSpace,
        eval_action_space: ActionSpace,
        eval_reward_function: RewardFunction,
        opponent_reward_function: Optional[RewardFunction] = None,
        batched_envs: int = 8,
        battle_format: str = "gen9ou",
        turn_limit: int = 200,
        opponent_model_name: str = "opponent",
        player_username: Optional[str] = None,
        save_trajectories_to: Optional[str] = None,
        save_results_to: Optional[str] = None,
        node_path: str = "node",
        showdown_dist: Optional[str] = None,
        n_workers: int = 1,
        seed: Optional[int] = None,
        eval_player_side: int = 0,
    ):
        if eval_player_side not in (0, 1):
            raise ValueError(f"eval_player_side must be 0 or 1; got {eval_player_side}")
        # Which Showdown side the outer AMAGO agent plays; opponent plays the other.
        self.eval_side = "p1" if int(eval_player_side) == 0 else "p2"
        self.opp_side = "p2" if self.eval_side == "p1" else "p1"

        self.player_team_set = player_team_set
        self.opponent_team_set = opponent_team_set
        self.opponent = opponent
        self.batched_envs = int(batched_envs)
        self.battle_format = battle_format
        self.turn_limit = turn_limit
        self.metamon_battle_format = battle_format
        self.metamon_opponent_name = opponent_model_name

        # Obs spaces can be *stateful* within a battle (e.g. revealed opponents);
        # one copy per lane per role, reset at the start of each battle.
        self.eval_obs_space = eval_obs_space
        self.eval_obs_spaces = [
            copy.deepcopy(eval_obs_space) for _ in range(self.batched_envs)
        ]
        self.opponent_obs_space = opponent_obs_space
        self.opponent_obs_spaces = [
            copy.deepcopy(opponent_obs_space) for _ in range(self.batched_envs)
        ]
        self.eval_action_space = eval_action_space
        self.opponent_action_space = opponent_action_space
        self.eval_reward_function = eval_reward_function
        self.opponent_reward_function = opponent_reward_function
        self.metamon_action_space = eval_action_space
        self.metamon_obs_space = eval_obs_space

        self.player_username = player_username or (
            f"MMVecSD-{''.join(str(random.randint(0, 9)) for _ in range(8))}"
        )

        if save_trajectories_to is not None:
            self.save_trajectories_to = os.path.join(
                save_trajectories_to, battle_format
            )
            os.makedirs(self.save_trajectories_to, exist_ok=True)
        else:
            self.save_trajectories_to = None
        if save_results_to is not None:
            # ``save_results_to`` is a *directory* (mirroring the non-vectorized
            # PokeEnvWrapper); we write one per-battle CSV per env instance inside
            # ``<save_results_to>/<battle_format>/``. The CSV records both the
            # player's and the opponent's sampled team file so per-team win rates
            # can be recovered per policy/opponent for downstream replay weighting.
            self.save_results_to = os.path.join(save_results_to, battle_format)
            os.makedirs(self.save_results_to, exist_ok=True)
            self.save_results_to = os.path.join(
                self.save_results_to,
                f"battle_log_{self.player_username}_{battle_format}.csv",
            )
            if not os.path.exists(self.save_results_to):
                with open(self.save_results_to, "a") as f:
                    f.write(
                        "Player Username, Team File, Opponent Team File, "
                        "Opponent Username, Result, Turn Count, Battle ID\n"
                    )
        else:
            self.save_results_to = None
        self._saving = (
            self.save_trajectories_to is not None or self.save_results_to is not None
        )
        self._traj_states: List[List[UniversalState]] = [
            [] for _ in range(self.batched_envs)
        ]
        self._traj_actions: List[List[int]] = [[] for _ in range(self.batched_envs)]

        self._rng = random.Random(seed)

        self.lanes: List[StreamBattleLane] = [
            StreamBattleLane(i, battle_format) for i in range(self.batched_envs)
        ]
        self._lane_steps = np.zeros((self.batched_envs,), dtype=np.int64)
        self._team_files: List[Optional[str]] = [None] * self.batched_envs
        self._opp_team_files: List[Optional[str]] = [None] * self.batched_envs
        self._last_opp_obs: List[Optional[dict]] = [None] * self.batched_envs
        # Actions for the in-flight ``step`` (re-applied on trap re-prompts).
        self._step_eval_actions: Optional[np.ndarray] = None
        self._step_opp_actions: Optional[np.ndarray] = None
        # Latest resolved action index per (lane, side) this step (updated on re-prompt).
        self._committed_side_actions: Dict[Tuple[int, str], int] = {}
        # Eval POV at the eval-side decision this step's action answers; diffed
        # against the next decision's POV for the reward.
        self._cycle_prev_eval: List[Optional[UniversalState]] = [
            None
        ] * self.batched_envs

        self.proc = make_sim_process(
            num_lanes=self.batched_envs,
            n_workers=n_workers,
            node_path=node_path,
            showdown_dist=showdown_dist,
        )
        for i, lane in enumerate(self.lanes):
            self.proc.register_lane(i, lane)

        self._profile = os.environ.get("METAMON_VEC_PROFILE") == "1"
        self._profile_stats: Dict[str, float] = {}
        self._profile_steps = 0
        self._profile_reported = False

        self.observation_space = self._build_observation_space()
        self.action_space = eval_action_space.gym_space

    # ----- gym spaces ------------------------------------------------------

    def _build_observation_space(self) -> gym.spaces.Space:
        base_space = self.eval_obs_space.gym_space
        if isinstance(base_space, gym.spaces.Dict):
            spaces = {}
            for k, space in base_space.spaces.items():
                if isinstance(space, gym.spaces.Box):
                    spaces[k] = gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.batched_envs,) + space.shape,
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
            shape=(self.batched_envs, self.eval_action_space.gym_space.n),
            dtype=bool,
        )
        return obs_space

    @property
    def env_name(self) -> str:
        return f"{self.metamon_battle_format}_vs_{self.metamon_opponent_name}"

    @contextmanager
    def _profile_section(self, name: str):
        if not self._profile:
            yield
            return
        t0 = time.perf_counter()
        yield
        self._profile_stats[name] = self._profile_stats.get(name, 0.0) + (
            time.perf_counter() - t0
        )

    def profile_report(self) -> str:
        if not self._profile_stats:
            return "no profile data (set METAMON_VEC_PROFILE=1)"
        total = sum(self._profile_stats.values())
        lines = [f"profile over {self._profile_steps} steps (total {total:.3f}s):"]
        for name, secs in sorted(
            self._profile_stats.items(), key=lambda kv: kv[1], reverse=True
        ):
            pct = 100.0 * secs / total if total else 0.0
            per_step = secs / max(self._profile_steps, 1)
            lines.append(
                f"  {name}: {secs:.3f}s ({pct:.1f}%, {per_step*1000:.2f}ms/step)"
            )
        return "\n".join(lines)

    # ----- lane lifecycle --------------------------------------------------

    @staticmethod
    def _coerce_policy_spec(value: Any) -> Optional["PolicySpec"]:
        from metamon.rl.evaluate.common import PolicySpec

        if value is None:
            return None
        if isinstance(value, PolicySpec):
            return value
        if isinstance(value, dict):
            name = value.get("name") or value.get("model_name", "opponent")
            return PolicySpec(
                name=str(name),
                model_name=str(value.get("model_name", name)),
                checkpoint=value.get("checkpoint"),
                temperature=float(value.get("temperature", 1.0)),
                team_set=str(value.get("team_set", "competitive")),
                battle_backend=str(value.get("battle_backend", "metamon")),
            )
        raise TypeError(
            f"reset(options['opponent']) must be PolicySpec, dict, or None; "
            f"got {type(value)}"
        )

    def _sync_opponent_model_spaces(self, model_name: str) -> None:
        """Refresh per-lane opponent obs/action spaces when the pool opponent changes."""
        from metamon.rl.pretrained import get_pretrained_model

        model = get_pretrained_model(model_name)
        self.opponent_obs_space = model.observation_space
        self.opponent_action_space = model.action_space
        self.opponent_reward_function = model.reward_function
        self.opponent_obs_spaces = [
            copy.deepcopy(self.opponent_obs_space) for _ in range(self.batched_envs)
        ]

    def _configure_opponent_for_reset(self, spec: Optional[Any] = None) -> None:
        """Swap the shared in-the-loop opponent (full env reset only)."""
        if not isinstance(self.opponent, ConfigBatchedOpponent):
            return
        resolved = self._coerce_policy_spec(spec)
        active = self.opponent.configure(resolved)
        self.opponent_team_set = self.opponent.current_team
        self.metamon_opponent_name = active.short_label
        self._sync_opponent_model_spaces(active.model_name)

    def _start_lane(self, i: int) -> None:
        lane = self.lanes[i]
        lane.reset_state()
        self.eval_obs_spaces[i].reset()
        self.opponent_obs_spaces[i].reset()
        self._lane_steps[i] = 0
        self._last_opp_obs[i] = None
        self._cycle_prev_eval[i] = None
        eval_team = self.player_team_set
        opp_team = self.opponent_team_set
        if self.eval_side == "p1":
            p1_team, p2_team = eval_team, opp_team
        else:
            p1_team, p2_team = opp_team, eval_team
        p1_spec, p1_file = player_spec(f"p1-{i}", p1_team, self.battle_format)
        p2_spec, p2_file = player_spec(f"p2-{i}", p2_team, self.battle_format)
        self._team_files[i] = p1_file if self.eval_side == "p1" else p2_file
        self._opp_team_files[i] = p2_file if self.eval_side == "p1" else p1_file
        seed = self._random_seed()
        self.proc.start_battle(i, self.battle_format, p1=p1_spec, p2=p2_spec, seed=seed)

    def _init_lane_trajectory(self, i: int) -> None:
        """Seed per-lane trajectory buffers with the eval POV at battle start."""
        self._traj_states[i] = [self.lanes[i].universal_state(self.eval_side)]
        self._traj_actions[i] = []

    def _random_seed(self):
        # Showdown PRNG seed: four 16-bit ints (sodium seed also accepted as str).
        return [self._rng.randint(0, 0xFFFF) for _ in range(4)]

    # ----- obs / legality --------------------------------------------------

    def _illegal_mask(self, legal: List[int], n: int) -> np.ndarray:
        mask = np.ones((n,), dtype=bool)
        for idx in legal:
            if 0 <= idx < n:
                mask[idx] = False
        return mask

    def _build_eval_obs(self, i: int) -> dict:
        obs, _info = self._build_eval_obs_and_info(i)
        return obs

    def _append_lane_choices(
        self,
        choices: List[Tuple[int, str, str]],
        lane_idx: int,
        side_choices: Dict[str, str],
    ) -> None:
        """Append one lane's choices in physical Showdown order (p1, then p2)."""
        for side in SIDES:
            choice = side_choices.get(side)
            if choice is not None:
                choices.append((lane_idx, side, choice))

    def _build_eval_obs_and_info(self, i: int) -> Tuple[dict, dict]:
        lane = self.lanes[i]
        state = lane.universal_state(self.eval_side)
        obs = self.eval_obs_spaces[i].state_to_obs(state)
        legal = lane.legal_action_indices(self.eval_side, self.eval_action_space, state)
        obs["illegal_actions"] = self._illegal_mask(
            legal, self.eval_action_space.gym_space.n
        )
        return obs, {"legal_actions": legal}

    def _build_opp_obs(self, i: int) -> dict:
        lane = self.lanes[i]
        state = lane.universal_state(self.opp_side)
        obs = self.opponent_obs_spaces[i].state_to_obs(state)
        legal = lane.legal_action_indices(
            self.opp_side, self.opponent_action_space, state
        )
        obs["illegal_actions"] = self._illegal_mask(
            legal, self.opponent_action_space.gym_space.n
        )
        self._last_opp_obs[i] = obs
        return obs

    # ----- action resolution ----------------------------------------------

    def _resolve_action(
        self, i: int, side: str, action_space: ActionSpace, raw_idx: int
    ) -> Tuple[int, str]:
        """Repair an illegal action against the legal mask, then build a choice.

        Mirrors ``PokeEnvWrapper.action_to_move`` + ``on_invalid_order``: keep the
        action if legal, else substitute a random legal one; if there is genuinely
        no legal action, fall back to Showdown's ``default`` (always accepted).

        Note we do *not* try to second-guess Showdown beyond metamon's own legality
        (``definitely_valid_actions``). Some rejections are unknowable at choice
        time because they depend on hidden info (e.g. an attempted switch while the
        active Pokemon is secretly trapped by Shadow Tag / Arena Trap / Magnet
        Pull, surfaced only as ``maybeTrapped``). Those produce a single-side
        ``|error|`` re-prompt that :meth:`_pump_settle` handles.
        """
        lane = self.lanes[i]
        state = lane.universal_state(side)
        legal = lane.legal_action_indices(side, action_space, state)
        idx = int(raw_idx)
        if idx not in legal:
            idx = int(self._rng.choice(legal)) if legal else idx
        choice = action_idx_to_choice(idx, lane.battle(side), lane.last_request[side])
        if choice is None:
            choice = DEFAULT_CHOICE
        return idx, choice

    def _retry_committed_side(self, i: int, side: str) -> None:
        """Re-answer from the action already committed this step.

        Used for trap ``|error|`` re-prompts and mid-cycle force-switches while the
        outer agent's single action for this ``step`` is fixed. Re-applies that
        intent against the updated legal mask (``_resolve_action`` repairs illegal
        picks, e.g. move -> random legal switch on force-switch requests).
        """
        if side == self.eval_side:
            action_space = self.eval_action_space
            raw = self._committed_side_actions.get((i, side))
            if raw is None and self._step_eval_actions is not None:
                raw = int(self._step_eval_actions[i])
            raw = int(raw) if raw is not None else 0
        else:
            action_space = self.opponent_action_space
            raw = self._committed_side_actions.get((i, side))
            if raw is None and self._step_opp_actions is not None:
                raw = int(self._step_opp_actions[i])
            raw = int(raw) if raw is not None else 0
        used_idx, choice = self._resolve_action(i, side, action_space, raw)
        self._committed_side_actions[(i, side)] = int(used_idx)
        self.proc.choose(i, side, choice)

    def _choose_opponent_side(self, i: int) -> None:
        """Answer a single opponent-side request with the batched opponent policy."""
        active = np.zeros((self.batched_envs,), dtype=bool)
        active[i] = True
        actions = self.opponent.act(active, self._opp_obs_list(active))
        used_idx, choice = self._resolve_action(
            i,
            self.opp_side,
            self.opponent_action_space,
            int(actions[i]),
        )
        self._committed_side_actions[(i, self.opp_side)] = int(used_idx)
        self.proc.choose(i, self.opp_side, choice)

    def _send_side(self, i: int, side: str, *, reprompt: bool = False) -> None:
        """Re-answer a single side after an ``|error|`` re-prompt.

        Used by :meth:`_pump_settle` when Showdown re-prompts one side after an
        ``|error|`` (e.g. a now-revealed trap). The updated request reflects the
        new reality (``trapped: true`` removes switches), so re-applying this
        step's committed action against it yields a legal choice without any
        special-casing.
        """
        lane = self.lanes[i]
        kind = lane.request_kind(side)
        if kind == KIND_TEAMPREVIEW or lane.battle(side).active_pokemon is None:
            # Teampreview, or a state we can't safely decode yet: let Showdown
            # auto-pick (always accepted), keeping the lockstep barrier intact.
            self.proc.choose(i, side, DEFAULT_CHOICE)
            lane.reprompt_pending[side] = False
            return
        self._retry_committed_side(i, side)
        lane.reprompt_pending[side] = False

    # ----- pumping ---------------------------------------------------------

    def _pump_ready(self, lane_indices: List[int]) -> None:
        if not lane_indices:
            return

        def ready() -> bool:
            return all(self.lanes[i].decision_ready() for i in lane_indices)

        self.proc.pump_until(ready)

    # Request kinds that Showdown expects a choice for (vs. `wait`/`done`).
    _ANSWERABLE_KINDS = (KIND_MOVE, KIND_FORCESWITCH, KIND_TEAMPREVIEW)

    def _pump_settle(self, lane_indices: List[int]) -> None:
        """Pump until every lane reaches its next decision cycle (or ends).

        Auto-resolves opponent-only follow-ups (e.g. the opponent fainted and
        must switch while the eval side waits) and repairs single-side
        ``|error|`` re-prompts for either side. Eval-side move/force-switch
        requests are *never* answered here: the lane is simply left parked at
        that decision for the next ``step`` to consume, mirroring poke-env's
        "ask the agent once per request" model. A KO force-switch is therefore
        just the next step's decision, not a special mid-turn case.
        """
        if not lane_indices:
            return

        answered = {
            i: {s: self.lanes[i].request_serial[s] for s in SIDES} for i in lane_indices
        }
        err_handled = {i: {s: -1 for s in SIDES} for i in lane_indices}

        def ready() -> bool:
            done = True
            for i in lane_indices:
                lane = self.lanes[i]
                if lane.decision_ready():
                    continue
                done = False
                for s in SIDES:
                    other = self.opp_side if s == self.eval_side else self.eval_side
                    advanced = lane.request_serial[s] > answered[i][s]
                    other_advanced = lane.request_serial[other] > answered[i][other]
                    if (
                        advanced
                        and not other_advanced
                        and lane.reprompt_pending[s]
                        and lane.request_kind(s) in self._ANSWERABLE_KINDS
                        and lane._side_ready(s)
                    ):
                        self._send_side(i, s, reprompt=True)
                        answered[i][s] = lane.request_serial[s]
                    elif (
                        advanced
                        and not other_advanced
                        and s == self.opp_side
                        and lane.request_kind(s) in self._ANSWERABLE_KINDS
                        and lane._side_ready(s)
                        and not lane.reprompt_pending[s]
                    ):
                        # Opponent-only follow-up: answer it so the cycle can
                        # resolve. (An asymmetric *eval* request just means we
                        # are waiting for the other side's request to arrive so
                        # the lane can park at the eval decision -- keep pumping.)
                        self._choose_opponent_side(i)
                        answered[i][s] = lane.request_serial[s]
                    elif (
                        not advanced
                        and lane.error[s]
                        and err_handled[i][s] != lane.request_serial[s]
                    ):
                        self._retry_committed_side(i, s)
                        err_handled[i][s] = lane.request_serial[s]
            return done

        self.proc.pump_until(ready)

    def _opp_obs_list(self, opp_active: np.ndarray) -> List[dict]:
        obs_list: List[dict] = []
        for i in range(self.batched_envs):
            if opp_active[i]:
                obs_list.append(self._build_opp_obs(i))
            else:
                cached = self._last_opp_obs[i]
                obs_list.append(
                    cached if cached is not None else self._build_opp_obs(i)
                )
        return obs_list

    def _advance_lanes(self, lane_indices: List[int]) -> None:
        """Auto-resolve teampreview + opponent-only cycles until each lane is
        parked at an eval-side agent decision (or ended)."""
        pending = [i for i in lane_indices if not self.lanes[i].ended]
        while pending:
            # Wait-only barrier (no choices owed yet this iteration); the previous
            # iteration's _pump_settle already cleared any re-prompts.
            self._pump_ready(pending)
            opp_active = np.zeros((self.batched_envs,), dtype=bool)
            auto: List[int] = []
            for i in pending:
                lane = self.lanes[i]
                if lane.ended:
                    continue
                if lane.request_kind(self.eval_side) in AGENT_KINDS:
                    continue  # eval owes a decision -> parked, stop advancing it
                auto.append(i)
                if lane.request_kind(self.opp_side) in AGENT_KINDS:
                    opp_active[i] = True
            if not auto:
                break
            opp_actions = (
                self.opponent.act(opp_active, self._opp_obs_list(opp_active))
                if opp_active.any()
                else None
            )
            opp_pending: List[Tuple[int, int, UniversalState]] = []
            choices: List[Tuple[int, str, str]] = []
            for i in auto:
                lane = self.lanes[i]
                side_choices: Dict[str, str] = {}
                k1 = lane.request_kind(self.eval_side)
                k2 = lane.request_kind(self.opp_side)
                if k1 == KIND_TEAMPREVIEW:
                    side_choices[self.eval_side] = DEFAULT_CHOICE
                if k2 == KIND_TEAMPREVIEW:
                    side_choices[self.opp_side] = DEFAULT_CHOICE
                elif k2 in AGENT_KINDS:
                    prev_opp = lane.universal_state(self.opp_side)
                    self._step_opp_actions[i] = int(opp_actions[i])
                    used_idx, choice = self._resolve_action(
                        i,
                        self.opp_side,
                        self.opponent_action_space,
                        int(opp_actions[i]),
                    )
                    side_choices[self.opp_side] = choice
                    self._committed_side_actions[(i, self.opp_side)] = int(used_idx)
                    opp_pending.append((i, int(used_idx), prev_opp))
                self._append_lane_choices(choices, i, side_choices)
                lane.mark_settled()
            if choices:
                self.proc.choose_batch(choices)
            self._pump_settle(auto)
            self._record_opp_rewards(opp_pending)
            pending = auto

    def _record_opp_rewards(
        self, opp_pending: List[Tuple[int, int, UniversalState]]
    ) -> None:
        if self.opponent_reward_function is None:
            for i, used_idx, _ in opp_pending:
                final_idx = self._committed_side_actions.get(
                    (i, self.opp_side), int(used_idx)
                )
                self.opponent.observe(i, 0.0, final_idx)
            return
        for i, used_idx, prev_opp in opp_pending:
            new_opp = self.lanes[i].universal_state(self.opp_side)
            reward = float(self.opponent_reward_function(prev_opp, new_opp))
            final_idx = self._committed_side_actions.get(
                (i, self.opp_side), int(used_idx)
            )
            self.opponent.observe(i, reward, final_idx)

    # ----- gym API ---------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng.seed(seed)
            np.random.seed(seed)
        options = options or {}
        self._configure_opponent_for_reset(options.get("opponent"))
        self.opponent.reset_all()
        for i in range(self.batched_envs):
            self._start_lane(i)
        self._advance_lanes(list(range(self.batched_envs)))
        if self._saving:
            for i in range(self.batched_envs):
                self._init_lane_trajectory(i)

        obs_list, infos = [], []
        for i in range(self.batched_envs):
            obs, info = self._build_eval_obs_and_info(i)
            obs_list.append(obs)
            infos.append(info)
        batched_obs = stack_obs_dicts(obs_list)
        merged_info = {"legal_actions": [info["legal_actions"] for info in infos]}
        return batched_obs, merged_info

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions).reshape(self.batched_envs)
        self._step_eval_actions = actions.copy()
        self._step_opp_actions = np.full((self.batched_envs,), -1, dtype=np.int64)
        self._committed_side_actions = {}
        rewards = np.zeros((self.batched_envs,), dtype=np.float32)
        terminated = np.zeros((self.batched_envs,), dtype=bool)
        truncated = np.zeros((self.batched_envs,), dtype=bool)
        infos: List[dict] = [{} for _ in range(self.batched_envs)]
        eval_actions: List[Optional[int]] = [None] * self.batched_envs
        opp_pending: List[Tuple[int, int, UniversalState]] = []
        acted_lanes: List[int] = []

        # Lanes parked at an eval-side decision (a normal move or a KO-induced
        # force-switch). Exactly one outer-agent decision is consumed per lane
        # per step, mirroring poke-env / ``QueueOnLocalLadder``: a force-switch
        # is simply the next step's decision, not a special mid-turn case.
        acting = [
            i
            for i in range(self.batched_envs)
            if not self.lanes[i].ended
            and self.lanes[i].needs_agent_decision(self.eval_side)
            and self.lanes[i]._side_ready(self.eval_side)
        ]

        with self._profile_section("resolve_actions"):
            # The opponent moves simultaneously with the eval side on a normal
            # turn (and switches alongside an eval force-switch on a double KO),
            # so answer it in the same cycle when it also owes a decision.
            opp_active = np.zeros((self.batched_envs,), dtype=bool)
            for i in acting:
                if self.lanes[i].needs_agent_decision(self.opp_side) and self.lanes[
                    i
                ]._side_ready(self.opp_side):
                    opp_active[i] = True
            with self._profile_section("opponent_act"):
                opp_actions = (
                    self.opponent.act(opp_active, self._opp_obs_list(opp_active))
                    if opp_active.any()
                    else None
                )
            choices: List[Tuple[int, str, str]] = []
            for i in acting:
                lane = self.lanes[i]
                self._cycle_prev_eval[i] = lane.universal_state(self.eval_side)
                eval_used_idx, eval_choice = self._resolve_action(
                    i, self.eval_side, self.eval_action_space, int(actions[i])
                )
                eval_actions[i] = int(eval_used_idx)
                side_choices: Dict[str, str] = {self.eval_side: eval_choice}
                self._committed_side_actions[(i, self.eval_side)] = int(eval_used_idx)
                if opp_active[i]:
                    prev_opp = lane.universal_state(self.opp_side)
                    self._step_opp_actions[i] = int(opp_actions[i])
                    opp_used_idx, opp_choice = self._resolve_action(
                        i,
                        self.opp_side,
                        self.opponent_action_space,
                        int(opp_actions[i]),
                    )
                    side_choices[self.opp_side] = opp_choice
                    self._committed_side_actions[(i, self.opp_side)] = int(opp_used_idx)
                    opp_pending.append((i, int(opp_used_idx), prev_opp))
                self._append_lane_choices(choices, i, side_choices)
                lane.mark_settled()
                acted_lanes.append(i)
            with self._profile_section("choose_batch"):
                self.proc.choose_batch(choices)
            with self._profile_section("pump_settle"):
                self._pump_settle(acting)

        self._record_opp_rewards(opp_pending)

        # Auto-resolve opponent-only cycles (e.g. opponent fainted, eval side waits)
        # until every live lane is parked back at an eval-side decision.
        with self._profile_section("advance_lanes"):
            self._advance_lanes(list(range(self.batched_envs)))

        restarted: List[int] = []
        with self._profile_section("rewards_restart"):
            for i in acted_lanes:
                lane = self.lanes[i]
                self._lane_steps[i] += 1
                new_eval = lane.universal_state(self.eval_side)
                prev = self._cycle_prev_eval[i]
                if prev is not None:
                    rewards[i] = float(self.eval_reward_function(prev, new_eval))
                self._cycle_prev_eval[i] = None
                if self._saving and eval_actions[i] is not None:
                    self._traj_actions[i].append(int(eval_actions[i]))
                    self._traj_states[i].append(new_eval)
                hit_limit = self._lane_steps[i] >= self.turn_limit
                if lane.ended or hit_limit:
                    terminated[i] = bool(lane.ended) or hit_limit
                    truncated[i] = hit_limit and not lane.ended
                    infos[i]["won"] = bool(new_eval.battle_won)
                    if self._saving:
                        self._save_lane_outcome(i, new_eval)
                    self._start_lane(i)
                    restarted.append(i)

        done_mask = terminated | truncated
        if done_mask.any():
            self.opponent.reset_lanes(done_mask)
        if restarted:
            with self._profile_section("advance_lanes"):
                self._advance_lanes(restarted)
            if self._saving:
                for i in restarted:
                    self._init_lane_trajectory(i)

        obs_list, legal_actions = [], []
        with self._profile_section("build_obs"):
            for i in range(self.batched_envs):
                obs, info = self._build_eval_obs_and_info(i)
                obs_list.append(obs)
                legal_actions.append(info["legal_actions"])
            batched_obs = stack_obs_dicts(obs_list)

        if self._profile:
            self._profile_steps += 1

        merged_info: Dict[str, Any] = {"legal_actions": legal_actions}
        for i, info in enumerate(infos):
            for k, v in info.items():
                merged_info.setdefault(k, [None] * self.batched_envs)
                merged_info[k][i] = v
        return batched_obs, rewards, terminated, truncated, merged_info

    def _save_lane_outcome(self, i: int, final_state: UniversalState) -> None:
        """Write a finished lane's trajectory and/or result log (parsed-replay format)."""
        result = "WIN" if final_state.battle_won else "LOSS"
        battle_id = "".join(str(random.randint(0, 9)) for _ in range(10))
        timestamp = datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
        opponent_name = self.metamon_opponent_name

        if self.save_trajectories_to is not None:
            filename = (
                f"metamon-{self.metamon_battle_format}-{battle_id}_Unrated_"
                f"{self.player_username}_vs_{opponent_name}_{timestamp}_{result}.json.lz4"
            )
            output_json = {
                "states": [s.to_dict() for s in self._traj_states[i]],
                "actions": self._traj_actions[i] + [-1],
            }
            path = os.path.join(self.save_trajectories_to, filename)
            temp_path = path + ".tmp"
            with lz4.frame.open(temp_path, "wb") as f:
                f.write(json.dumps(output_json).encode("utf-8"))
            os.rename(temp_path, path)

        if self.save_results_to is not None:
            with open(self.save_results_to, "a") as f:
                f.write(
                    f"{self.player_username},{self._team_files[i]},"
                    f"{self._opp_team_files[i]},{opponent_name},{result},"
                    f"{int(self._lane_steps[i])},"
                    f"{battle_id}\n"
                )

    def close(self) -> None:
        if self._profile and self._profile_steps and not self._profile_reported:
            self._profile_reported = True
            print(self.profile_report(), flush=True)
        if getattr(self, "proc", None) is not None:
            self.proc.close()
            self.proc = None

    def take_long_break(self):
        pass

    def resume_from_break(self):
        pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class ShowdownEnv(VectorizedShowdownEnv):
    """Single-battle Showdown env — for debugging and AMAGO ``sync`` eval.

    Presents a standard single-env gym interface (no leading batch dimension on
    obs/rewards/dones). Wrap with
    :class:`~metamon.rl.metamon_to_amago.MetamonAMAGOWrapper` and run AMAGO in
    ``env_mode="sync"`` with ``parallel_actors=1`` — NOT ``already_vectorized``.
    """

    def __init__(self, *args, **kwargs):
        batched = kwargs.get("batched_envs", 1)
        if int(batched) != 1:
            raise ValueError(
                f"ShowdownEnv is single-battle; batched_envs must be 1, got {batched}"
            )
        kwargs["batched_envs"] = 1
        super().__init__(*args, **kwargs)
        self._init_sync_observation_space()

    def _init_sync_observation_space(self) -> None:
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
        batched_obs, merged_info = super().reset(seed=seed, options=options)
        obs = {k: v[0] for k, v in batched_obs.items()}
        return obs, {"legal_actions": merged_info["legal_actions"][0]}

    def step(self, action):
        batched_obs, rewards, terminated, truncated, merged_info = super().step(
            np.asarray([action])
        )
        obs = {k: v[0] for k, v in batched_obs.items()}
        info = {"legal_actions": merged_info["legal_actions"][0]}
        for k, v in merged_info.items():
            if k == "legal_actions":
                continue
            info[k] = v[0] if isinstance(v, list) else v
        return (
            obs,
            float(rewards[0]),
            bool(terminated[0]),
            bool(truncated[0]),
            info,
        )


def _resolve_opponent_device(device, opponent_gpu_idx):
    import torch

    if opponent_gpu_idx is not None:
        return torch.device(f"cuda:{int(opponent_gpu_idx)}")
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def BattleAgainstMetamon(
    battle_format: str,
    observation_space: ObservationSpace,
    action_space: ActionSpace,
    reward_function: RewardFunction,
    team_set,
    opponent_model,
    opponent_checkpoint: Optional[int] = None,
    opponent_policy=None,
    opponent_obs_space: Optional[ObservationSpace] = None,
    opponent_action_space: Optional[ActionSpace] = None,
    opponent_hidden_state=None,
    opponent_team_set=None,
    batched_envs: int = 8,
    turn_limit: int = 200,
    opponent_sample: bool = True,
    eval_player_side: int = 0,
    save_trajectories_to: Optional[str] = None,
    save_results_to: Optional[str] = None,
    player_username: Optional[str] = None,
    device: Optional[str] = None,
    opponent_gpu_idx: Optional[int] = None,
    node_path: str = "node",
    showdown_dist: Optional[str] = None,
    n_workers: int = 1,
    seed: Optional[int] = None,
):
    """Factory: vectorized Showdown env vs a metamon ``PretrainedModel`` opponent.

    ``batched_envs=1`` returns :class:`ShowdownEnv` for AMAGO ``sync`` mode;
    ``batched_envs>1`` returns :class:`VectorizedShowdownEnv` for
    ``already_vectorized``. Compatible with the metamon eval harness and
    ``VectorizedMetamonAMAGOWrapper`` (``illegal_actions`` in obs).
    """
    from metamon.rl.pretrained import PretrainedModel
    from .opponent import AmagoBatchedOpponent

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

    opponent = AmagoBatchedOpponent(
        policy=opponent_policy,
        device=opponent_device,
        num_lanes=int(batched_envs),
        action_dim=opponent_action_space.gym_space.n,
        hidden_state=opponent_hidden_state,
        sample=opponent_sample,
    )

    env_cls = ShowdownEnv if int(batched_envs) == 1 else VectorizedShowdownEnv
    return env_cls(
        player_team_set=team_set,
        opponent_team_set=opponent_team_set or copy.deepcopy(team_set),
        opponent=opponent,
        opponent_obs_space=opponent_obs_space,
        opponent_action_space=opponent_action_space,
        eval_obs_space=observation_space,
        eval_action_space=action_space,
        eval_reward_function=reward_function,
        opponent_reward_function=opponent_model.reward_function,
        batched_envs=batched_envs,
        battle_format=battle_format,
        turn_limit=turn_limit,
        opponent_model_name=opponent_model.model_name,
        eval_player_side=eval_player_side,
        player_username=player_username,
        save_trajectories_to=save_trajectories_to,
        save_results_to=save_results_to,
        node_path=node_path,
        showdown_dist=showdown_dist,
        n_workers=n_workers,
        seed=seed,
    )


# Backwards-compatible alias from early vectorized rollout.
BattleShowdownVectorized = BattleAgainstMetamon


def BattleAgainstOpponentPool(
    battle_format: str,
    observation_space: ObservationSpace,
    action_space: ActionSpace,
    reward_function: RewardFunction,
    team_set,
    opponent_config_path: Optional[str] = None,
    opponent_config: Optional[OpponentPoolConfig] = None,
    opponent_config_template_vars: Optional[Dict[str, str]] = None,
    opponent_weights_path: Optional[str] = None,
    batched_envs: int = 8,
    turn_limit: int = 200,
    opponent_sample: bool = True,
    eval_player_side: int = 0,
    save_trajectories_to: Optional[str] = None,
    save_results_to: Optional[str] = None,
    player_username: Optional[str] = None,
    device: Optional[str] = None,
    opponent_gpu_idx: Optional[int] = None,
    node_path: str = "node",
    showdown_dist: Optional[str] = None,
    n_workers: int = 1,
    seed: Optional[int] = None,
):
    """Factory: one shared opponent sampled from an opponent pool config.

    Each full env ``reset()`` calls :meth:`OpponentPoolConfig.sample_opponent`
    (pick an agent, then sample checkpoint / temperature / team set). All lanes
    share that opponent until the next ``reset()``. Pair with AMAGO
    ``force_reset_on_every=True`` for diversity between training epochs.
    """
    from metamon.rl.pretrained import get_pretrained_model

    if opponent_config is None:
        if opponent_config_path is None:
            raise ValueError("Provide opponent_config or opponent_config_path")
        opponent_config = load_opponent_pool(
            opponent_config_path,
            battle_format=battle_format,
            template_vars=opponent_config_template_vars,
        )
    if seed is not None:
        opponent_config.rng.seed(seed)
    opponent_device = _resolve_opponent_device(device, opponent_gpu_idx)

    probe = opponent_config.sample_opponent()
    probe_model = get_pretrained_model(probe.model_name)
    opponent_obs_space = probe_model.observation_space
    opponent_action_space = probe_model.action_space
    opponent_reward_function = probe_model.reward_function

    opponent = ConfigBatchedOpponent(
        config=opponent_config,
        num_lanes=int(batched_envs),
        device=opponent_device,
        sample=opponent_sample,
        weights_path=opponent_weights_path,
    )

    env_cls = ShowdownEnv if int(batched_envs) == 1 else VectorizedShowdownEnv
    return env_cls(
        player_team_set=team_set,
        opponent_team_set=copy.deepcopy(team_set),
        opponent=opponent,
        opponent_obs_space=opponent_obs_space,
        opponent_action_space=opponent_action_space,
        eval_obs_space=observation_space,
        eval_action_space=action_space,
        eval_reward_function=reward_function,
        opponent_reward_function=opponent_reward_function,
        batched_envs=batched_envs,
        battle_format=battle_format,
        turn_limit=turn_limit,
        opponent_model_name="opponent-pool",
        eval_player_side=eval_player_side,
        player_username=player_username,
        save_trajectories_to=save_trajectories_to,
        save_results_to=save_results_to,
        node_path=node_path,
        showdown_dist=showdown_dist,
        n_workers=n_workers,
        seed=seed,
    )
