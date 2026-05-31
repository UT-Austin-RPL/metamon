"""Vectorized Showdown battle env with an in-the-loop batched NN opponent.

N battles run inside one Node process (``battle_host.js``) hosting N
``BattleStream``s; Python advances them in lockstep and batches the opponent's
neural-network forward pass across lanes. The evaluated agent always plays p1;
the opponent plays p2.

Control flow is *request driven* (Showdown resolves all mid-turn sequencing for
us): each ``step`` consumes the agent's action for the lane's current p1 decision
(a normal move or a forced switch), applies the opponent's p2 decision for the
same cycle when there is one, advances the sim, and then auto-resolves any cycles
where only the opponent must act (e.g. the opponent replacing a fainted Pokemon
while p1 waits) before parking each lane back at the next p1 decision.

The agent-facing contracts (``ObservationSpace``, ``DefaultActionSpace`` →
``Discrete(13)``, ``RewardFunction``, ``illegal_actions`` in the obs) match
``VectorizedPokepyEnv`` so this env is a drop-in alternative.
"""

from __future__ import annotations

import copy
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
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
from .opponent import BatchedOpponent, RandomBatchedOpponent
from .sim_process import ShowdownSimProcess
from .team_adapter import player_spec

EVAL_SIDE = "p1"
OPP_SIDE = "p2"


class VectorizedShowdownEnv(gym.Env):
    """N-lane Showdown env: p1 is the eval agent, p2 is a batched NN opponent."""

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
        save_results_to: Optional[str] = None,
        node_path: str = "node",
        showdown_dist: Optional[str] = None,
        seed: Optional[int] = None,
    ):
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

        self.save_results_to = save_results_to
        self.player_username = player_username or (
            f"MMVecSD-{''.join(str(random.randint(0, 9)) for _ in range(8))}"
        )

        self._rng = random.Random(seed)

        self.lanes: List[StreamBattleLane] = [
            StreamBattleLane(i, battle_format) for i in range(self.batched_envs)
        ]
        self._lane_steps = np.zeros((self.batched_envs,), dtype=np.int64)
        self._team_files: List[Optional[str]] = [None] * self.batched_envs
        self._last_opp_obs: List[Optional[dict]] = [None] * self.batched_envs

        self.proc = ShowdownSimProcess(node_path=node_path, showdown_dist=showdown_dist)
        for i, lane in enumerate(self.lanes):
            self.proc.register_lane(i, lane)

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

    # ----- lane lifecycle --------------------------------------------------

    def _start_lane(self, i: int) -> None:
        lane = self.lanes[i]
        lane.reset_state()
        self.eval_obs_spaces[i].reset()
        self.opponent_obs_spaces[i].reset()
        self._lane_steps[i] = 0
        self._last_opp_obs[i] = None
        p1_spec, p1_file = player_spec(
            f"p1-{i}", self.player_team_set, self.battle_format
        )
        p2_spec, _ = player_spec(f"p2-{i}", self.opponent_team_set, self.battle_format)
        self._team_files[i] = p1_file
        seed = self._random_seed()
        self.proc.start_battle(i, self.battle_format, p1=p1_spec, p2=p2_spec, seed=seed)

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

    def _build_eval_obs_and_info(self, i: int) -> Tuple[dict, dict]:
        lane = self.lanes[i]
        state = lane.universal_state(EVAL_SIDE)
        obs = self.eval_obs_spaces[i].state_to_obs(state)
        legal = lane.legal_action_indices(EVAL_SIDE, self.eval_action_space, state)
        obs["illegal_actions"] = self._illegal_mask(
            legal, self.eval_action_space.gym_space.n
        )
        return obs, {"legal_actions": legal}

    def _build_opp_obs(self, i: int) -> dict:
        lane = self.lanes[i]
        state = lane.universal_state(OPP_SIDE)
        obs = self.opponent_obs_spaces[i].state_to_obs(state)
        legal = lane.legal_action_indices(OPP_SIDE, self.opponent_action_space, state)
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

    def _send_side(self, i: int, side: str) -> None:
        """(Re)answer a single side from its current request.

        Used by :meth:`_pump_settle` when Showdown re-prompts one side after an
        ``|error|`` (e.g. a now-revealed trap). The updated request reflects the
        new reality (``trapped: true`` removes switches), so resolving against it
        yields a legal choice without any special-casing.
        """
        lane = self.lanes[i]
        kind = lane.request_kind(side)
        if kind == KIND_TEAMPREVIEW or lane.battle(side).active_pokemon is None:
            # Teampreview, or a state we can't safely decode yet: let Showdown
            # auto-pick (always accepted), keeping the lockstep barrier intact.
            self.proc.choose(i, side, DEFAULT_CHOICE)
            return
        action_space = (
            self.eval_action_space if side == EVAL_SIDE else self.opponent_action_space
        )
        state = lane.universal_state(side)
        legal = lane.legal_action_indices(side, action_space, state)
        if legal:
            idx = int(self._rng.choice(legal))
            choice = (
                action_idx_to_choice(idx, lane.battle(side), lane.last_request[side])
                or DEFAULT_CHOICE
            )
        else:
            choice = DEFAULT_CHOICE
        self.proc.choose(i, side, choice)

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
        """Pump until every lane reaches its next decision cycle, re-answering any
        side Showdown re-prompts mid-cycle.

        The caller has just sent this cycle's choices. Normally the turn resolves
        and *both* sides receive their next request together (``decision_ready``).
        But an illegal choice that is only detectable with hidden info — most
        commonly an attempted switch while secretly trapped — is rejected with an
        ``|error|`` and a fresh request to *that side only*. Its ``request_serial``
        advances while the other side's does not, so a naive both-sides barrier
        would deadlock waiting on output that never comes. We detect such a
        single-side re-prompt and answer it from the updated request until the
        cycle truly completes.
        """
        if not lane_indices:
            return

        # Highest request_serial we have already answered for each side. The
        # caller just answered the current request, so start from there.
        answered = {
            i: {s: self.lanes[i].request_serial[s] for s in SIDES} for i in lane_indices
        }
        # request_serial at which we last sent a `default` to clear an |error|
        # that did not re-emit a request (avoids resending while the turn resolves).
        err_handled = {i: {s: -1 for s in SIDES} for i in lane_indices}

        def ready() -> bool:
            done = True
            for i in lane_indices:
                lane = self.lanes[i]
                if lane.decision_ready():
                    continue
                done = False
                # Not at the next cycle yet. A normal turn resolution re-requests
                # *both* sides together; an error re-prompt advances only the
                # erroring side (or, for some errors, re-emits no request at all).
                for s in SIDES:
                    other = OPP_SIDE if s == EVAL_SIDE else EVAL_SIDE
                    advanced = lane.request_serial[s] > answered[i][s]
                    other_advanced = lane.request_serial[other] > answered[i][other]
                    if (
                        advanced
                        and not other_advanced
                        and lane.request_kind(s) in self._ANSWERABLE_KINDS
                    ):
                        # Re-emitted single-side request (e.g. now-revealed trap):
                        # answer it from the updated request. Never act when both
                        # advanced (that is the next cycle whose |switch| public
                        # log may not be applied yet; wait, don't pre-empt).
                        self._send_side(i, s)
                        answered[i][s] = lane.request_serial[s]
                    elif (
                        not advanced
                        and lane.error[s]
                        and err_handled[i][s] != lane.request_serial[s]
                    ):
                        # Showdown rejected our choice without re-emitting a request
                        # (e.g. a stale forme name). Let it auto-pick a legal action,
                        # mirroring poke-env's re-choose-on-error, so the turn can
                        # resolve instead of deadlocking the barrier.
                        self.proc.choose(i, s, DEFAULT_CHOICE)
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
        parked at a p1 agent decision (or ended)."""
        pending = list(lane_indices)
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
                if lane.request_kind(EVAL_SIDE) in AGENT_KINDS:
                    continue  # eval owes a decision -> parked, stop advancing it
                auto.append(i)
                if lane.request_kind(OPP_SIDE) in AGENT_KINDS:
                    opp_active[i] = True
            if not auto:
                break
            opp_actions = (
                self.opponent.act(opp_active, self._opp_obs_list(opp_active))
                if opp_active.any()
                else None
            )
            opp_pending: List[Tuple[int, int, UniversalState]] = []
            for i in auto:
                lane = self.lanes[i]
                k1 = lane.request_kind(EVAL_SIDE)
                k2 = lane.request_kind(OPP_SIDE)
                if k1 == KIND_TEAMPREVIEW:
                    self.proc.choose(i, EVAL_SIDE, DEFAULT_CHOICE)
                if k2 == KIND_TEAMPREVIEW:
                    self.proc.choose(i, OPP_SIDE, DEFAULT_CHOICE)
                elif k2 in AGENT_KINDS:
                    prev_opp = lane.universal_state(OPP_SIDE)
                    used_idx, choice = self._resolve_action(
                        i, OPP_SIDE, self.opponent_action_space, int(opp_actions[i])
                    )
                    self.proc.choose(i, OPP_SIDE, choice)
                    opp_pending.append((i, used_idx, prev_opp))
                lane.mark_settled()
            self._pump_settle(auto)
            self._record_opp_rewards(opp_pending)
            pending = auto

    def _record_opp_rewards(
        self, opp_pending: List[Tuple[int, int, UniversalState]]
    ) -> None:
        if self.opponent_reward_function is None:
            for i, used_idx, _ in opp_pending:
                self.opponent.observe(i, 0.0, used_idx)
            return
        for i, used_idx, prev_opp in opp_pending:
            new_opp = self.lanes[i].universal_state(OPP_SIDE)
            reward = float(self.opponent_reward_function(prev_opp, new_opp))
            self.opponent.observe(i, reward, used_idx)

    # ----- gym API ---------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng.seed(seed)
            np.random.seed(seed)
        self.opponent.reset_all()
        for i in range(self.batched_envs):
            self._start_lane(i)
        self._advance_lanes(list(range(self.batched_envs)))

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
        rewards = np.zeros((self.batched_envs,), dtype=np.float32)
        terminated = np.zeros((self.batched_envs,), dtype=bool)
        truncated = np.zeros((self.batched_envs,), dtype=bool)
        infos: List[dict] = [{} for _ in range(self.batched_envs)]

        # Opponent's decision for the *current* (parked) cycle: only lanes where
        # p2 must also act simultaneously (normal turn, or double-KO switch).
        opp_active = np.zeros((self.batched_envs,), dtype=bool)
        for i in range(self.batched_envs):
            if self.lanes[i].request_kind(OPP_SIDE) in AGENT_KINDS:
                opp_active[i] = True
        opp_actions = (
            self.opponent.act(opp_active, self._opp_obs_list(opp_active))
            if opp_active.any()
            else None
        )

        prev_eval: List[UniversalState] = [
            lane.universal_state(EVAL_SIDE) for lane in self.lanes
        ]
        opp_pending: List[Tuple[int, int, UniversalState]] = []
        for i in range(self.batched_envs):
            lane = self.lanes[i]
            _, eval_choice = self._resolve_action(
                i, EVAL_SIDE, self.eval_action_space, int(actions[i])
            )
            self.proc.choose(i, EVAL_SIDE, eval_choice)
            if opp_active[i]:
                prev_opp = lane.universal_state(OPP_SIDE)
                used_idx, opp_choice = self._resolve_action(
                    i, OPP_SIDE, self.opponent_action_space, int(opp_actions[i])
                )
                self.proc.choose(i, OPP_SIDE, opp_choice)
                opp_pending.append((i, used_idx, prev_opp))
            lane.mark_settled()

        self._pump_settle(list(range(self.batched_envs)))
        self._record_opp_rewards(opp_pending)

        # Auto-resolve opponent-only cycles (e.g. opponent fainted, p1 waits)
        # until every live lane is parked back at a p1 decision.
        self._advance_lanes(list(range(self.batched_envs)))

        restarted: List[int] = []
        for i in range(self.batched_envs):
            lane = self.lanes[i]
            self._lane_steps[i] += 1
            new_eval = lane.universal_state(EVAL_SIDE)
            rewards[i] = float(self.eval_reward_function(prev_eval[i], new_eval))
            hit_limit = self._lane_steps[i] >= self.turn_limit
            if lane.ended or hit_limit:
                terminated[i] = bool(lane.ended) or hit_limit
                truncated[i] = hit_limit and not lane.ended
                infos[i]["won"] = bool(new_eval.battle_won)
                if self.save_results_to is not None:
                    self._save_result(i, new_eval)
                self._start_lane(i)
                restarted.append(i)

        done_mask = terminated | truncated
        if done_mask.any():
            self.opponent.reset_lanes(done_mask)
        if restarted:
            self._advance_lanes(restarted)

        obs_list, legal_actions = [], []
        for i in range(self.batched_envs):
            obs, info = self._build_eval_obs_and_info(i)
            obs_list.append(obs)
            legal_actions.append(info["legal_actions"])
        batched_obs = stack_obs_dicts(obs_list)

        merged_info: Dict[str, Any] = {"legal_actions": legal_actions}
        for i, info in enumerate(infos):
            for k, v in info.items():
                merged_info.setdefault(k, [None] * self.batched_envs)
                merged_info[k][i] = v
        return batched_obs, rewards, terminated, truncated, merged_info

    def _save_result(self, i: int, final_state: UniversalState) -> None:
        result = "WIN" if final_state.battle_won else "LOSS"
        battle_id = "".join(str(random.randint(0, 9)) for _ in range(10))
        timestamp = datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
        with open(self.save_results_to, "a") as f:
            f.write(
                f"{self.player_username},{self._team_files[i]},"
                f"{self.metamon_opponent_name},{result},{int(self._lane_steps[i])},"
                f"{battle_id}\n"
            )

    def close(self) -> None:
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


def _resolve_opponent_device(device, opponent_gpu_idx):
    import torch

    if opponent_gpu_idx is not None:
        return torch.device(f"cuda:{int(opponent_gpu_idx)}")
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def BattleShowdownVectorized(
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
    save_results_to: Optional[str] = None,
    player_username: Optional[str] = None,
    device: Optional[str] = None,
    opponent_gpu_idx: Optional[int] = None,
    node_path: str = "node",
    showdown_dist: Optional[str] = None,
    seed: Optional[int] = None,
) -> VectorizedShowdownEnv:
    """Factory: vectorized Showdown env vs a metamon ``PretrainedModel`` opponent.

    Parallels ``BattlePokepyVectorized`` so it slots into the same eval/training
    harness, and produces an env compatible with ``VectorizedMetamonAMAGOWrapper``
    (``illegal_actions`` in obs, ``env_mode='already_vectorized'``).
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

    return VectorizedShowdownEnv(
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
        player_username=player_username,
        save_results_to=save_results_to,
        node_path=node_path,
        showdown_dist=showdown_dist,
        seed=seed,
    )
