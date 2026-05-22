from __future__ import annotations

import atexit
from collections import deque
from dataclasses import dataclass, field
import json
import math
import os
from typing import Any, Iterable, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

EPS = 1e-8


@dataclass(frozen=True)
class EnsembleMemberSpec:
    """Configuration for one inference-only ensemble member."""

    model_name: str
    checkpoint: Optional[int] = None
    gxe: float = 0.5
    proposer_bias: float = 1.0
    judge_bias: float = 1.0
    shortlist_k: int = 2
    proposal_roles: tuple[str, ...] = ()
    action_temperature: float = 1.0


@dataclass
class _EnsembleMemberRuntime:
    spec: EnsembleMemberSpec
    policy: nn.Module
    device: torch.device
    experiment: Any


@dataclass
class _StallTransition:
    state_key: tuple[Any, ...]
    action: int
    reward: float


@dataclass
class _StallTrackerState:
    transitions: deque[_StallTransition] = field(
        default_factory=lambda: deque(maxlen=24)
    )
    pending_state_key: Optional[tuple[Any, ...]] = None
    pending_action: Optional[int] = None


@dataclass
class _EnsembleHiddenState:
    member_hidden: list[Any]
    stall_trackers: list[_StallTrackerState]


@dataclass(frozen=True)
class _ProposerVariant:
    proposer_idx: int
    role: str
    allowed_actions: tuple[int, ...]
    weight: float


@dataclass
class _AnchorDeviationMetrics:
    total_decisions: int = 0
    forced_switch_decisions: int = 0
    unforced_decisions: int = 0
    pre_override_non_anchor: int = 0
    pre_override_non_anchor_forced: int = 0
    pre_override_non_anchor_unforced: int = 0
    final_non_anchor: int = 0
    final_non_anchor_forced: int = 0
    final_non_anchor_unforced: int = 0
    anchor_consensus_swaps: int = 0
    single_judge_decisions: int = 0
    full_rerank_decisions: int = 0
    total_shortlist_size: int = 0
    total_proposer_variants: int = 0
    total_judges: int = 0

    def to_dict(self) -> dict[str, Any]:
        total = self.total_decisions
        forced = self.forced_switch_decisions
        unforced = self.unforced_decisions
        return {
            "total_decisions": total,
            "forced_switch_decisions": forced,
            "unforced_decisions": unforced,
            "pre_override_non_anchor": self.pre_override_non_anchor,
            "pre_override_non_anchor_forced": self.pre_override_non_anchor_forced,
            "pre_override_non_anchor_unforced": self.pre_override_non_anchor_unforced,
            "pre_override_non_anchor_rate": (
                self.pre_override_non_anchor / total if total else 0.0
            ),
            "final_non_anchor": self.final_non_anchor,
            "final_non_anchor_forced": self.final_non_anchor_forced,
            "final_non_anchor_unforced": self.final_non_anchor_unforced,
            "final_non_anchor_rate": self.final_non_anchor / total if total else 0.0,
            "final_non_anchor_rate_forced": (
                self.final_non_anchor_forced / forced if forced else 0.0
            ),
            "final_non_anchor_rate_unforced": (
                self.final_non_anchor_unforced / unforced if unforced else 0.0
            ),
            "anchor_consensus_swaps": self.anchor_consensus_swaps,
            "anchor_consensus_swap_rate": (
                self.anchor_consensus_swaps / total if total else 0.0
            ),
            "single_judge_decisions": self.single_judge_decisions,
            "single_judge_rate": (
                self.single_judge_decisions / total if total else 0.0
            ),
            "full_rerank_decisions": self.full_rerank_decisions,
            "full_rerank_rate": (self.full_rerank_decisions / total if total else 0.0),
            "avg_shortlist_size": (self.total_shortlist_size / total if total else 0.0),
            "avg_proposer_variants": (
                self.total_proposer_variants / total if total else 0.0
            ),
            "avg_judges": self.total_judges / total if total else 0.0,
        }


def _normalize(values: list[float]) -> list[float]:
    total = sum(max(v, 0.0) for v in values)
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [max(v, 0.0) / total for v in values]


def _normalize_with_floor(
    values: list[float], anchor_pos: Optional[int], floor: float
) -> list[float]:
    weights = _normalize(values)
    if anchor_pos is None or not (0 <= anchor_pos < len(weights)):
        return weights
    floor = min(max(floor, 0.0), 1.0)
    if weights[anchor_pos] >= floor:
        return weights
    if len(weights) == 1:
        return [1.0]
    remainder = 1.0 - floor
    other_total = sum(w for i, w in enumerate(weights) if i != anchor_pos)
    if other_total <= EPS:
        new_weights = [0.0] * len(weights)
        new_weights[anchor_pos] = 1.0
        return new_weights
    new_weights = []
    for idx, weight in enumerate(weights):
        if idx == anchor_pos:
            new_weights.append(floor)
        else:
            new_weights.append(weight / other_total * remainder)
    return new_weights


def _zscore(values: torch.Tensor) -> torch.Tensor:
    if values.numel() <= 1:
        return torch.zeros_like(values)
    mean = values.mean()
    std = values.std(unbiased=False)
    if float(std) < EPS:
        return torch.zeros_like(values)
    return (values - mean) / std


def _space_signature(space: gym.Space) -> Any:
    if isinstance(space, gym.spaces.Dict):
        return (
            "dict",
            tuple((k, _space_signature(v)) for k, v in sorted(space.spaces.items())),
        )
    if isinstance(space, gym.spaces.Box):
        return ("box", tuple(space.shape), str(space.dtype))
    if isinstance(space, gym.spaces.Discrete):
        return ("discrete", int(space.n))
    if isinstance(space, gym.spaces.Text):
        return ("text", int(space.max_length), int(space.min_length))
    return (space.__class__.__name__, repr(space))


def _parse_member_devices(num_members: int) -> list[torch.device]:
    raw = os.environ.get("METAMON_ENSEMBLE_MEMBER_DEVICES")
    if raw:
        device_names = [part.strip() for part in raw.split(",") if part.strip()]
    elif torch.cuda.is_available():
        device_names = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    else:
        device_names = ["cpu"]
    if not device_names:
        device_names = ["cpu"]
    devices = []
    for idx in range(num_members):
        name = device_names[idx % len(device_names)]
        if name.isdigit():
            name = f"cuda:{name}"
        devices.append(torch.device(name))
    return devices


class _EnsembleTrajEncoderProxy:
    """Mimics the tiny subset of TrajEncoder used by AMAGO eval loops."""

    def __init__(self, members: list[_EnsembleMemberRuntime]):
        self.members = members

    def init_hidden_state(self, batch_size: int, device: torch.device):
        return _EnsembleHiddenState(
            member_hidden=[
                member.policy.traj_encoder.init_hidden_state(batch_size, member.device)
                for member in self.members
            ],
            stall_trackers=[_StallTrackerState() for _ in range(batch_size)],
        )

    def reset_hidden_state(self, hidden_state, dones):
        if hidden_state is None:
            return None
        if isinstance(dones, torch.Tensor):
            dones = dones.detach().cpu().numpy()
        dones = np.asarray(dones, dtype=bool)
        if isinstance(hidden_state, _EnsembleHiddenState):
            member_hidden = hidden_state.member_hidden
            stall_trackers = list(hidden_state.stall_trackers)
        else:
            member_hidden = hidden_state
            stall_trackers = [_StallTrackerState() for _ in range(len(dones))]
        reset_member_hidden = [
            member.policy.traj_encoder.reset_hidden_state(member_hidden, dones)
            for member, member_hidden in zip(self.members, member_hidden)
        ]
        for idx, done in enumerate(dones.tolist()):
            if done:
                stall_trackers[idx] = _StallTrackerState()
        return _EnsembleHiddenState(
            member_hidden=reset_member_hidden,
            stall_trackers=stall_trackers,
        )


class HeuristicRouterEnsemblePolicy(nn.Module):
    """Inference-only proposer/judge ensemble over a fixed set of pretrained policies.

    The router is intentionally heuristic rather than trained: it uses member GXE priors,
    per-turn uncertainty, proposer disagreement, and action-count features to decide which
    experts should propose candidates and which should judge them.
    """

    def __init__(
        self,
        members: list[_EnsembleMemberRuntime],
        action_dim: int,
    ):
        super().__init__()
        self.members = members
        self.action_dim = action_dim
        self.traj_encoder = _EnsembleTrajEncoderProxy(members)
        self.anchor_idx = 0
        self._anchor_metrics_path = os.environ.get(
            "METAMON_ENSEMBLE_DEBUG_METRICS_PATH"
        )
        self._anchor_metrics = _AnchorDeviationMetrics()
        if self._anchor_metrics_path:
            atexit.register(self._flush_anchor_metrics)

    def eval(self):
        super().eval()
        for member in self.members:
            member.policy.eval()
        return self

    @staticmethod
    def _is_move_action(action: int) -> bool:
        return action <= 3 or action >= 9

    @staticmethod
    def _is_switch_action(action: int) -> bool:
        return 4 <= action <= 8

    def _extract_state_summary(
        self,
        *,
        obs: dict[str, torch.Tensor],
        time_idxs: torch.Tensor,
        batch_idx: int,
    ) -> dict[str, float]:
        turn_idx = int(time_idxs[batch_idx].reshape(-1)[-1].item())
        summary = {
            "turn_idx": float(turn_idx),
            "resource_edge": 0.0,
            "player_remaining": 0.0,
            "opponent_remaining": 0.0,
            "player_active_hp": 0.0,
            "opponent_active_hp": 0.0,
        }
        numbers = obs.get("numbers")
        if numbers is None:
            return summary
        current_numbers = numbers[batch_idx, -1].detach().float().cpu()
        if current_numbers.numel() < 34:
            return summary

        opponent_remaining = float(current_numbers[0].clamp(0.0, 1.0).item() * 6.0)
        player_active_hp = float(current_numbers[1].clamp(0.0, 1.0).item())
        switch_hps = [max(float(value.item()), 0.0) for value in current_numbers[28:33]]
        player_remaining = float(player_active_hp > 0.02) + sum(
            hp > 0.02 for hp in switch_hps
        )
        opponent_active_hp = float(current_numbers[33].clamp(0.0, 1.0).item())

        active_edge = player_active_hp - opponent_active_hp
        alive_edge = (player_remaining - opponent_remaining) / 6.0
        reserve_edge = (
            sum(switch_hps) / max(len(switch_hps), 1) - 0.5 if switch_hps else 0.0
        )
        resource_edge = 0.55 * active_edge + 0.35 * alive_edge + 0.10 * reserve_edge
        summary.update(
            {
                "resource_edge": resource_edge,
                "player_remaining": player_remaining,
                "opponent_remaining": opponent_remaining,
                "player_active_hp": player_active_hp,
                "opponent_active_hp": opponent_active_hp,
            }
        )
        return summary

    def _record_anchor_decision(
        self,
        *,
        default_anchor_action: int,
        anchor_action: int,
        proposed_best_action: int,
        final_action: int,
        forced_switch: bool,
        single_judge: bool,
        full_rerank: bool,
        shortlist_size: int,
        proposer_variant_count: int,
        judge_count: int,
    ) -> None:
        if not self._anchor_metrics_path:
            return
        metrics = self._anchor_metrics
        metrics.total_decisions += 1
        metrics.anchor_consensus_swaps += int(anchor_action != default_anchor_action)
        metrics.single_judge_decisions += int(single_judge)
        metrics.full_rerank_decisions += int(full_rerank)
        metrics.total_shortlist_size += shortlist_size
        metrics.total_proposer_variants += proposer_variant_count
        metrics.total_judges += judge_count
        if forced_switch:
            metrics.forced_switch_decisions += 1
        else:
            metrics.unforced_decisions += 1
        if proposed_best_action != anchor_action:
            metrics.pre_override_non_anchor += 1
            if forced_switch:
                metrics.pre_override_non_anchor_forced += 1
            else:
                metrics.pre_override_non_anchor_unforced += 1
        if final_action != anchor_action:
            metrics.final_non_anchor += 1
            if forced_switch:
                metrics.final_non_anchor_forced += 1
            else:
                metrics.final_non_anchor_unforced += 1
        if metrics.total_decisions % 50 == 0:
            self._flush_anchor_metrics()

    def _flush_anchor_metrics(self) -> None:
        if not self._anchor_metrics_path:
            return
        os.makedirs(os.path.dirname(self._anchor_metrics_path), exist_ok=True)
        payload = {
            "anchor_model_index": self.anchor_idx,
            "anchor_model_name": self.members[self.anchor_idx].spec.model_name,
            "anchor_checkpoint": self.members[self.anchor_idx].spec.checkpoint,
            **self._anchor_metrics.to_dict(),
        }
        with open(self._anchor_metrics_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def get_actions(
        self,
        obs: dict[str, torch.Tensor],
        rl2s: torch.Tensor,
        time_idxs: torch.Tensor,
        hidden_state=None,
        sample: bool = True,
    ):
        batch_size = next(iter(obs.values())).shape[0]
        output_device = next(iter(obs.values())).device
        if hidden_state is None:
            hidden_state = self.traj_encoder.init_hidden_state(
                batch_size, output_device
            )
        if isinstance(hidden_state, _EnsembleHiddenState):
            member_hidden_state = hidden_state.member_hidden
            stall_trackers = list(hidden_state.stall_trackers)
        else:
            member_hidden_state = hidden_state
            stall_trackers = [_StallTrackerState() for _ in range(batch_size)]

        member_steps = []
        next_hidden = []
        for member, member_hidden in zip(self.members, member_hidden_state):
            member_obs = {
                key: value.to(member.device, non_blocking=True)
                for key, value in obs.items()
            }
            member_rl2s = rl2s.to(member.device, non_blocking=True)
            member_time_idxs = time_idxs.to(member.device, non_blocking=True)
            traj_emb, member_hidden = member.policy.get_state_embedding(
                obs=member_obs,
                rl2s=member_rl2s,
                time_idxs=member_time_idxs,
                hidden_state=member_hidden,
            )
            straight_from_obs = {
                key: member_obs[key] for key in member.policy.pass_obs_keys_to_actor
            }
            action_dist = member.policy.actor(
                traj_emb,
                straight_from_obs=straight_from_obs,
            )
            probs = action_dist.probs[:, -1, -1, :].detach()
            member_steps.append(
                {
                    "member": member,
                    "obs": member_obs,
                    "traj_emb": traj_emb[:, -1:, :].detach(),
                    "probs": probs,
                }
            )
            next_hidden.append(member_hidden)

        illegal_actions = obs["illegal_actions"][:, -1, :].bool()
        actions = []
        for batch_idx in range(batch_size):
            legal_actions = (
                (~illegal_actions[batch_idx]).nonzero(as_tuple=True)[0].tolist()
            )
            if not legal_actions:
                actions.append(0)
                continue
            if len(legal_actions) == 1:
                actions.append(legal_actions[0])
                continue
            tracker = stall_trackers[batch_idx]
            current_state_key = self._make_cycle_key(
                obs=obs,
                batch_idx=batch_idx,
                legal_actions=legal_actions,
            )
            state_summary = self._extract_state_summary(
                obs=obs,
                time_idxs=time_idxs,
                batch_idx=batch_idx,
            )
            prev_reward = float(rl2s[batch_idx, -1, 0].item())
            self._finalize_stall_transition(
                tracker=tracker,
                prev_reward=prev_reward,
            )
            chosen = self._choose_action_for_batch(
                batch_idx=batch_idx,
                legal_actions=legal_actions,
                member_steps=member_steps,
                sample=sample,
                tracker=tracker,
                current_state_key=current_state_key,
                state_summary=state_summary,
            )
            tracker.pending_state_key = current_state_key
            tracker.pending_action = chosen
            actions.append(chosen)

        actions = torch.tensor(actions, device=output_device, dtype=torch.uint8)
        return actions.view(batch_size, 1, 1), _EnsembleHiddenState(
            member_hidden=next_hidden,
            stall_trackers=stall_trackers,
        )

    def _make_cycle_key(
        self,
        obs: dict[str, torch.Tensor],
        batch_idx: int,
        legal_actions: list[int],
    ) -> tuple[Any, ...]:
        key_parts: list[Any] = [tuple(legal_actions)]
        text_tokens = obs.get("text_tokens")
        if text_tokens is not None:
            key_parts.append(
                tuple(
                    text_tokens[batch_idx, -1].detach().cpu().to(torch.int16).tolist()
                )
            )
        numbers = obs.get("numbers")
        if numbers is not None:
            coarse_numbers = torch.round(
                numbers[batch_idx, -1].detach().float().cpu() * 4.0
            ).to(torch.int16)
            key_parts.append(tuple(coarse_numbers.tolist()))
        return tuple(key_parts)

    def _finalize_stall_transition(
        self,
        tracker: _StallTrackerState,
        prev_reward: float,
    ) -> None:
        if tracker.pending_state_key is None or tracker.pending_action is None:
            return
        if not math.isfinite(prev_reward):
            prev_reward = 0.0
        tracker.transitions.append(
            _StallTransition(
                state_key=tracker.pending_state_key,
                action=tracker.pending_action,
                reward=prev_reward,
            )
        )
        tracker.pending_state_key = None
        tracker.pending_action = None

    def _stall_penalties(
        self,
        tracker: _StallTrackerState,
        current_state_key: tuple[Any, ...],
        forced_switch: bool,
    ) -> dict[int, float]:
        if forced_switch:
            return {}
        history = list(tracker.transitions)
        if len(history) < 4:
            return {}

        penalties: dict[int, float] = {}
        reward_cap = 0.08
        mean_reward_cap = 0.025

        def low_progress(window: list[_StallTransition]) -> bool:
            magnitudes = [abs(step.reward) for step in window]
            return (
                max(magnitudes, default=0.0) <= reward_cap
                and sum(magnitudes) / max(len(magnitudes), 1) <= mean_reward_cap
            )

        # Only react to clearly persistent cycles such as AAA or ABABAB.
        for period in (1, 2):
            if len(history) < 3 * period:
                continue
            recent = history[-3 * period :]
            blocks = [recent[idx * period : (idx + 1) * period] for idx in range(3)]
            state_blocks = [[step.state_key for step in block] for block in blocks]
            action_blocks = [[step.action for step in block] for block in blocks]
            if not (
                state_blocks[0] == state_blocks[1] == state_blocks[2]
                and action_blocks[0] == action_blocks[1] == action_blocks[2]
            ):
                continue
            if current_state_key != state_blocks[-1][0]:
                continue
            if not low_progress(recent):
                continue
            cycle_action = action_blocks[-1][0]
            penalty = 0.07 + 0.02 * (period == 1)
            penalties[cycle_action] = max(penalties.get(cycle_action, 0.0), penalty)

        if len(history) >= 4:
            recent = history[-4:]
            if (
                all(step.state_key == current_state_key for step in recent)
                and len({step.action for step in recent}) == 1
                and low_progress(recent)
            ):
                repeat_action = recent[-1].action
                penalties[repeat_action] = max(penalties.get(repeat_action, 0.0), 0.12)

        return penalties

    def _apply_action_penalties(
        self,
        shortlist: list[int],
        final_scores: torch.Tensor,
        penalties: dict[int, float],
    ) -> torch.Tensor:
        if not penalties:
            return final_scores
        adjusted = final_scores.clone()
        for idx, action in enumerate(shortlist):
            adjusted[idx] -= penalties.get(action, 0.0)
        return adjusted

    def _build_proposer_variants(
        self,
        *,
        legal_actions: list[int],
        proposer_weights: list[float],
        features: list[dict[str, Any]],
        anchor_action: int,
        forced_switch: bool,
    ) -> tuple[list[_ProposerVariant], bool]:
        move_actions = tuple(
            action for action in legal_actions if self._is_move_action(action)
        )
        switch_actions = tuple(
            action for action in legal_actions if self._is_switch_action(action)
        )
        mixed_choice = not forced_switch and bool(move_actions) and bool(switch_actions)

        variants: list[_ProposerVariant] = []
        for proposer_idx, proposer_weight in enumerate(proposer_weights):
            if proposer_weight <= 0.0:
                continue
            info = features[proposer_idx]
            counter_anchor_actions = tuple(
                action for action in legal_actions if action != anchor_action
            )
            if not mixed_choice:
                role_scores: list[tuple[str, tuple[int, ...], float]] = [
                    (
                        "any",
                        tuple(legal_actions),
                        0.80 + 0.20 * info["top_prob"],
                    )
                ]
                if counter_anchor_actions:
                    role_scores.append(
                        (
                            "counter_anchor",
                            counter_anchor_actions,
                            info["counter_anchor_mass"]
                            * (1.10 if info["top_action"] != anchor_action else 0.90),
                        )
                    )
                total_role_score = sum(
                    score for _, _, score in role_scores if score > 0.0
                )
                for role, allowed_actions, role_score in role_scores:
                    if role_score <= 0.0 or total_role_score <= 0.0:
                        continue
                    variants.append(
                        _ProposerVariant(
                            proposer_idx=proposer_idx,
                            role=role,
                            allowed_actions=allowed_actions,
                            weight=proposer_weight * role_score / total_role_score,
                        )
                    )
                continue

            role_scores = []
            if info["move_mass"] >= 0.15:
                role_scores.append(
                    (
                        "move",
                        move_actions,
                        info["move_mass"]
                        * (1.10 if self._is_move_action(info["top_action"]) else 0.90),
                    )
                )
            if info["switch_mass"] >= 0.15:
                role_scores.append(
                    (
                        "switch",
                        switch_actions,
                        info["switch_mass"]
                        * (
                            1.10 if self._is_switch_action(info["top_action"]) else 0.90
                        ),
                    )
                )
            if counter_anchor_actions and info["counter_anchor_mass"] >= 0.20:
                role_scores.append(
                    (
                        "counter_anchor",
                        counter_anchor_actions,
                        info["counter_anchor_mass"]
                        * (1.10 if info["top_action"] != anchor_action else 0.85),
                    )
                )
            if not role_scores:
                role_scores = (
                    [("counter_anchor", counter_anchor_actions, 1.0)]
                    if counter_anchor_actions
                    else [("any", tuple(legal_actions), 1.0)]
                )
            allowed_roles = info["spec"].proposal_roles
            if allowed_roles:
                filtered_role_scores = [
                    (role, allowed_actions, role_score)
                    for role, allowed_actions, role_score in role_scores
                    if role in allowed_roles
                ]
                if filtered_role_scores:
                    role_scores = filtered_role_scores
            total_role_score = sum(score for _, _, score in role_scores if score > 0.0)
            for role, allowed_actions, role_score in role_scores:
                if not allowed_actions or role_score <= 0.0 or total_role_score <= 0.0:
                    continue
                variants.append(
                    _ProposerVariant(
                        proposer_idx=proposer_idx,
                        role=role,
                        allowed_actions=allowed_actions,
                        weight=proposer_weight * role_score / total_role_score,
                    )
                )
        return variants, mixed_choice

    def _select_proposer_variants(
        self,
        *,
        proposer_variants: list[_ProposerVariant],
        disagreement: float,
        forced_switch: bool,
    ) -> list[_ProposerVariant]:
        if len(proposer_variants) <= 1:
            return proposer_variants

        max_variants = 3 + int(disagreement > 0.30) + int(forced_switch)
        if len(proposer_variants) <= max_variants:
            selected = proposer_variants
        else:

            def variant_score(variant: _ProposerVariant) -> float:
                role_bonus = 1.0
                if variant.role == "counter_anchor":
                    role_bonus += 0.08 + 0.08 * float(disagreement > 0.25)
                elif variant.role == "switch":
                    role_bonus += 0.04 * float(forced_switch)
                return variant.weight * role_bonus

            ranked = sorted(
                proposer_variants,
                key=variant_score,
                reverse=True,
            )
            selected: list[_ProposerVariant] = []
            seen: set[tuple[int, str]] = set()

            def maybe_add(variant: _ProposerVariant) -> None:
                key = (variant.proposer_idx, variant.role)
                if key in seen or len(selected) >= max_variants:
                    return
                selected.append(variant)
                seen.add(key)

            anchor_variants = [
                variant for variant in ranked if variant.proposer_idx == self.anchor_idx
            ]
            if anchor_variants:
                maybe_add(anchor_variants[0])

            non_anchor_variants = [
                variant for variant in ranked if variant.proposer_idx != self.anchor_idx
            ]
            if non_anchor_variants:
                maybe_add(non_anchor_variants[0])

            for variant in ranked:
                maybe_add(variant)
                if len(selected) >= max_variants:
                    break

        total_weight = sum(variant.weight for variant in selected)
        if total_weight <= 0.0:
            return selected
        return [
            _ProposerVariant(
                proposer_idx=variant.proposer_idx,
                role=variant.role,
                allowed_actions=variant.allowed_actions,
                weight=variant.weight / total_weight,
            )
            for variant in selected
        ]

    def _masked_action_distribution(
        self,
        *,
        step: dict[str, Any],
        batch_idx: int,
        allowed_actions: tuple[int, ...],
    ) -> tuple[list[int], torch.Tensor]:
        allowed_list = list(allowed_actions)
        masked_probs = step["probs"][batch_idx, allowed_list].float()
        masked_probs = masked_probs / masked_probs.sum().clamp(min=EPS)
        ranked_local = torch.argsort(masked_probs, descending=True)
        ranked_actions = [allowed_list[idx] for idx in ranked_local.tolist()]
        ranked_scores = masked_probs[ranked_local]
        return ranked_actions, ranked_scores

    def _shortlist_k_for_variant(
        self,
        *,
        info: dict[str, Any],
        proposer_idx: int,
        role: str,
        allowed_actions: tuple[int, ...],
        mixed_choice: bool,
    ) -> int:
        if role == "move":
            shortlist_k = (
                1
                + int(proposer_idx == self.anchor_idx and len(allowed_actions) >= 2)
                + int(
                    proposer_idx == self.anchor_idx
                    and len(allowed_actions) >= 3
                    and info["entropy"] > 0.45
                )
            )
        elif role == "switch":
            shortlist_k = 1
        elif role == "counter_anchor":
            shortlist_k = 1 + int(
                proposer_idx == self.anchor_idx
                and len(allowed_actions) >= 4
                and mixed_choice
            )
        else:
            shortlist_k = (
                info["spec"].shortlist_k
                + int(proposer_idx == self.anchor_idx)
                + int(info["entropy"] > 0.45 and proposer_idx == self.anchor_idx)
            )
        return min(len(allowed_actions), shortlist_k)

    def _route_shortlist(
        self,
        *,
        legal_actions: list[int],
        anchor_action: int,
        candidate_support: dict[int, float],
        candidate_members: dict[int, set[int]],
        candidate_roles: dict[int, set[str]],
        features: list[dict[str, Any]],
        disagreement: float,
        mixed_choice: bool,
        anchor_top_prob: float,
        state_summary: dict[str, float],
        strong_anchor: bool,
    ) -> tuple[list[int], dict[int, float]]:
        if not candidate_support:
            return [anchor_action], {anchor_action: anchor_top_prob}

        actions = sorted(candidate_support)
        support_tensor = torch.tensor(
            [candidate_support[action] for action in actions], dtype=torch.float32
        )
        member_tensor = torch.tensor(
            [len(candidate_members.get(action, set())) for action in actions],
            dtype=torch.float32,
        )
        role_tensor = torch.tensor(
            [len(candidate_roles.get(action, set())) for action in actions],
            dtype=torch.float32,
        )
        non_anchor_tensor = torch.tensor(
            [
                sum(
                    member_idx != self.anchor_idx
                    for member_idx in candidate_members.get(action, set())
                )
                for action in actions
            ],
            dtype=torch.float32,
        )
        strength_tensor = torch.tensor(
            [
                sum(
                    features[member_idx]["strength"]
                    for member_idx in candidate_members.get(action, set())
                )
                for action in actions
            ],
            dtype=torch.float32,
        )

        router_scores = 0.45 * _zscore(support_tensor)
        router_scores = router_scores + 0.25 * _zscore(member_tensor)
        router_scores = router_scores + 0.10 * _zscore(role_tensor)
        router_scores = router_scores + 0.15 * _zscore(non_anchor_tensor)
        router_scores = router_scores + 0.10 * _zscore(strength_tensor)
        late_turn = state_summary["turn_idx"] >= 30.0
        desperation = state_summary["resource_edge"] < -0.10 or (
            late_turn and state_summary["resource_edge"] < 0.02
        )
        stabilize = state_summary["resource_edge"] > 0.14 and (
            state_summary["player_remaining"] >= state_summary["opponent_remaining"]
        )
        if desperation:
            router_scores = router_scores + 0.08 * _zscore(non_anchor_tensor)
        if stabilize:
            router_scores = router_scores + 0.05 * _zscore(strength_tensor)
        if anchor_action in actions:
            anchor_bonus = 0.04 + 0.03 * stabilize - 0.02 * desperation
            if strong_anchor and not desperation:
                anchor_bonus += 0.03
            router_scores[actions.index(anchor_action)] += anchor_bonus

        shortlist_k = min(
            len(actions),
            3
            + int(len(legal_actions) >= 5)
            + int(disagreement > 0.30)
            + int(mixed_choice),
        )
        shortlist_k = min(
            len(actions),
            shortlist_k + int(desperation) + int(late_turn and mixed_choice),
        )
        ranked_indices = sorted(
            range(len(actions)),
            key=lambda idx: (
                float(router_scores[idx].item()),
                float(support_tensor[idx].item()),
                float(strength_tensor[idx].item()),
                float(member_tensor[idx].item()),
                -actions[idx],
            ),
            reverse=True,
        )
        shortlist = [actions[idx] for idx in ranked_indices[:shortlist_k]]

        if anchor_action not in shortlist:
            if len(shortlist) < shortlist_k:
                shortlist.append(anchor_action)
            else:
                replace_idx = len(shortlist) - 1
                for idx in range(len(shortlist) - 1, -1, -1):
                    if shortlist[idx] != anchor_action:
                        replace_idx = idx
                        break
                shortlist[replace_idx] = anchor_action

        if mixed_choice:
            for predicate in (self._is_move_action, self._is_switch_action):
                if any(predicate(action) for action in shortlist):
                    continue
                candidate_idxs = [
                    idx for idx, action in enumerate(actions) if predicate(action)
                ]
                if not candidate_idxs:
                    continue
                best_idx = max(
                    candidate_idxs,
                    key=lambda idx: (
                        float(router_scores[idx].item()),
                        float(support_tensor[idx].item()),
                        float(strength_tensor[idx].item()),
                        -actions[idx],
                    ),
                )
                replacement = actions[best_idx]
                replace_idx = len(shortlist) - 1
                for idx in range(len(shortlist) - 1, -1, -1):
                    if shortlist[idx] != anchor_action and not predicate(
                        shortlist[idx]
                    ):
                        replace_idx = idx
                        break
                shortlist[replace_idx] = replacement

        shortlist = list(dict.fromkeys(shortlist))
        proposal_support = {
            action: candidate_support.get(action, 0.0) for action in shortlist
        }
        proposal_support.setdefault(anchor_action, anchor_top_prob)
        return shortlist, proposal_support

    def _choose_action_for_batch(
        self,
        batch_idx: int,
        legal_actions: list[int],
        member_steps: list[dict[str, Any]],
        sample: bool,
        tracker: _StallTrackerState,
        current_state_key: tuple[Any, ...],
        state_summary: dict[str, float],
    ) -> int:
        features = []
        top_actions = []
        for step in member_steps:
            probs = step["probs"][batch_idx, legal_actions].float()
            probs = probs / probs.sum().clamp(min=EPS)
            entropy = float(
                -torch.sum(probs * probs.clamp(min=EPS).log()).item()
                / max(math.log(len(legal_actions)), 1.0)
            )
            top_idx = int(torch.argmax(probs).item())
            top_action = legal_actions[top_idx]
            sorted_probs = torch.sort(probs, descending=True).values
            margin = float(
                sorted_probs[0].item() - sorted_probs[1].item()
                if sorted_probs.numel() > 1
                else 1.0
            )
            features.append(
                {
                    "entropy": entropy,
                    "certainty": 1.0 - entropy,
                    "top_prob": float(sorted_probs[0].item()),
                    "margin": margin,
                    "top_action": top_action,
                    "strength": step["member"].spec.gxe,
                    "spec": step["member"].spec,
                }
            )
            top_actions.append(top_action)

        anchor = features[self.anchor_idx]
        default_anchor_action = anchor["top_action"]
        default_anchor_margin = anchor["margin"]
        default_anchor_top_prob = anchor["top_prob"]

        vote_weights: dict[int, float] = {}
        vote_counts: dict[int, int] = {}
        for info in features:
            action = info["top_action"]
            vote_weights[action] = vote_weights.get(action, 0.0) + (
                0.80 * info["strength"] + 0.20 * info["certainty"]
            )
            vote_counts[action] = vote_counts.get(action, 0) + 1
        consensus_action = max(
            sorted(vote_weights),
            key=lambda action: (
                vote_counts[action],
                vote_weights[action],
                -action,
            ),
        )
        consensus_count = vote_counts[consensus_action]
        anchor_action = default_anchor_action
        if (
            consensus_action != default_anchor_action
            and consensus_count >= 2
            and vote_weights[consensus_action]
            > vote_weights.get(default_anchor_action, 0.0)
            and default_anchor_top_prob < 0.62
        ):
            anchor_action = consensus_action

        anchor_probs = member_steps[self.anchor_idx]["probs"][
            batch_idx, legal_actions
        ].float()
        anchor_probs = anchor_probs / anchor_probs.sum().clamp(min=EPS)
        anchor_prob_map = {
            action: float(prob.item())
            for action, prob in zip(legal_actions, anchor_probs)
        }
        anchor_top_prob = anchor_prob_map.get(anchor_action, default_anchor_top_prob)
        if anchor_action == default_anchor_action:
            anchor_margin = default_anchor_margin
        else:
            sorted_anchor_probs = sorted(anchor_prob_map.values(), reverse=True)
            anchor_margin = max(
                anchor_top_prob
                - (sorted_anchor_probs[0] if sorted_anchor_probs else 0.0),
                0.0,
            )
        for info, step in zip(features, member_steps):
            full_probs = step["probs"][batch_idx, legal_actions].float()
            full_probs = full_probs / full_probs.sum().clamp(min=EPS)
            info["move_mass"] = float(
                sum(
                    prob.item()
                    for action, prob in zip(legal_actions, full_probs)
                    if self._is_move_action(action)
                )
            )
            info["switch_mass"] = float(
                sum(
                    prob.item()
                    for action, prob in zip(legal_actions, full_probs)
                    if self._is_switch_action(action)
                )
            )
            info["counter_anchor_mass"] = float(
                sum(
                    prob.item()
                    for action, prob in zip(legal_actions, full_probs)
                    if action != anchor_action
                )
            )
        forced_switch = all(action >= 4 for action in legal_actions)
        stall_penalties = self._stall_penalties(
            tracker=tracker,
            current_state_key=current_state_key,
            forced_switch=forced_switch,
        )
        late_turn = state_summary["turn_idx"] >= 30.0
        desperation = state_summary["resource_edge"] < -0.10 or (
            late_turn and state_summary["resource_edge"] < 0.02
        )
        stabilize = state_summary["resource_edge"] > 0.14 and (
            state_summary["player_remaining"] >= state_summary["opponent_remaining"]
        )
        disagreement = 1.0 - (
            max(top_actions.count(action) for action in set(top_actions))
            / max(len(top_actions), 1)
        )

        proposer_scores = []
        judge_scores = []
        for idx, info in enumerate(features):
            proposer = info["spec"].proposer_bias * (
                0.70 * info["strength"]
                + 0.20 * info["entropy"]
                + 0.10 * float(info["top_action"] != anchor_action)
            )
            judge = info["spec"].judge_bias * (
                0.85 * info["strength"] + 0.15 * info["certainty"]
            )
            if forced_switch:
                proposer *= 1.05
                judge *= 1.05
            if disagreement > 0.35:
                proposer *= 1.05
            if idx == self.anchor_idx:
                proposer *= 1.35
                judge *= 1.45
            if desperation:
                if idx == self.anchor_idx:
                    proposer *= 0.95
                    judge *= 0.96
                else:
                    proposer *= 1.04 + 0.06 * float(info["top_action"] != anchor_action)
                    judge *= 1.02
            if stabilize:
                if idx == self.anchor_idx:
                    proposer *= 1.04
                    judge *= 1.08
                elif info["top_action"] != anchor_action:
                    proposer *= 0.94
                    judge *= 0.97
            if info["strength"] < 0.50:
                proposer *= 0.75
                judge *= 0.50
            proposer_scores.append(proposer)
            judge_scores.append(judge)

        anchor_judge_score = judge_scores[self.anchor_idx]
        best_other_judge = max(
            (
                judge_scores[idx]
                for idx in range(len(member_steps))
                if idx != self.anchor_idx
            ),
            default=0.0,
        )
        anchor_dominant_judge = (
            anchor_judge_score > 0.0 and best_other_judge <= 0.20 * anchor_judge_score
        )

        if anchor_dominant_judge:
            num_judges = 1
        elif len(member_steps) <= 3:
            num_judges = len(member_steps)
        else:
            num_judges = min(
                len(member_steps),
                2
                + int(disagreement > 0.35 and anchor_margin < 0.15)
                + int(desperation and len(member_steps) >= 4),
            )

        judge_order = [self.anchor_idx]
        for idx in sorted(
            (i for i in range(len(member_steps)) if i != self.anchor_idx),
            key=lambda i: judge_scores[i],
            reverse=True,
        ):
            if len(judge_order) >= num_judges:
                break
            judge_order.append(idx)

        strong_anchor = features[self.anchor_idx]["strength"] >= 0.82
        proposer_floor = 0.18 if desperation else 0.22
        judge_floor = 0.40 if desperation else 0.45
        if strong_anchor and not desperation:
            proposer_floor += 0.05
            judge_floor += 0.08
        proposer_weights = _normalize_with_floor(
            proposer_scores,
            anchor_pos=self.anchor_idx,
            floor=proposer_floor,
        )
        judge_weights = _normalize_with_floor(
            [judge_scores[idx] for idx in judge_order],
            anchor_pos=judge_order.index(self.anchor_idx),
            floor=judge_floor,
        )
        proposer_variants, mixed_choice = self._build_proposer_variants(
            legal_actions=legal_actions,
            proposer_weights=proposer_weights,
            features=features,
            anchor_action=anchor_action,
            forced_switch=forced_switch,
        )
        proposer_variants = self._select_proposer_variants(
            proposer_variants=proposer_variants,
            disagreement=disagreement,
            forced_switch=forced_switch,
        )

        candidate_support: dict[int, float] = {}
        candidate_members: dict[int, set[int]] = {}
        candidate_roles: dict[int, set[str]] = {}
        for variant in proposer_variants:
            if not variant.allowed_actions or variant.weight <= 0.0:
                continue
            proposer_idx = variant.proposer_idx
            step = member_steps[proposer_idx]
            info = features[proposer_idx]
            shortlist_k = self._shortlist_k_for_variant(
                info=info,
                proposer_idx=proposer_idx,
                role=variant.role,
                allowed_actions=variant.allowed_actions,
                mixed_choice=mixed_choice,
            )
            ranked_actions, ranked_scores = self._masked_action_distribution(
                step=step,
                batch_idx=batch_idx,
                allowed_actions=variant.allowed_actions,
            )
            for action, action_score in zip(
                ranked_actions[:shortlist_k],
                ranked_scores[:shortlist_k].tolist(),
            ):
                candidate_support[action] = candidate_support.get(action, 0.0) + (
                    variant.weight * float(action_score)
                )
                candidate_members.setdefault(action, set()).add(proposer_idx)
                candidate_roles.setdefault(action, set()).add(variant.role)
        candidate_support.setdefault(anchor_action, anchor_top_prob)
        candidate_members.setdefault(anchor_action, set()).add(self.anchor_idx)
        candidate_roles.setdefault(anchor_action, set()).add("anchor_guard")
        shortlist, proposal_support = self._route_shortlist(
            legal_actions=legal_actions,
            anchor_action=anchor_action,
            candidate_support=candidate_support,
            candidate_members=candidate_members,
            candidate_roles=candidate_roles,
            features=features,
            disagreement=disagreement,
            mixed_choice=mixed_choice,
            anchor_top_prob=anchor_top_prob,
            state_summary=state_summary,
            strong_anchor=strong_anchor,
        )
        if not shortlist:
            shortlist = [consensus_action]
            proposal_support = {consensus_action: 1.0}

        shortlist, final_scores = self._judge_shortlist(
            batch_idx=batch_idx,
            shortlist=shortlist,
            judge_order=judge_order,
            judge_weights=judge_weights,
            member_steps=member_steps,
            forced_switch=forced_switch,
            proposal_support=proposal_support,
        )
        final_scores = self._apply_action_penalties(
            shortlist=shortlist,
            final_scores=final_scores,
            penalties=stall_penalties,
        )

        full_rerank = False
        if len(shortlist) < len(legal_actions):
            sorted_scores = torch.sort(final_scores, descending=True).values
            margin = float(
                sorted_scores[0].item() - sorted_scores[1].item()
                if sorted_scores.numel() > 1
                else float("inf")
            )
            if anchor_dominant_judge or disagreement > 0.25 or margin < 0.16:
                shortlist, final_scores = self._judge_shortlist(
                    batch_idx=batch_idx,
                    shortlist=legal_actions,
                    judge_order=judge_order,
                    judge_weights=judge_weights,
                    member_steps=member_steps,
                    forced_switch=forced_switch,
                    proposal_support=proposal_support,
                )
                final_scores = self._apply_action_penalties(
                    shortlist=shortlist,
                    final_scores=final_scores,
                    penalties=stall_penalties,
                )
                full_rerank = True

        best_local = int(torch.argmax(final_scores).item())
        best_action = shortlist[best_local]
        if best_action == anchor_action:
            self._record_anchor_decision(
                default_anchor_action=default_anchor_action,
                anchor_action=anchor_action,
                proposed_best_action=best_action,
                final_action=best_action,
                forced_switch=forced_switch,
                single_judge=len(judge_order) == 1,
                full_rerank=full_rerank,
                shortlist_size=len(shortlist),
                proposer_variant_count=len(proposer_variants),
                judge_count=len(judge_order),
            )
            return best_action

        anchor_score = float("-inf")
        if anchor_action in shortlist:
            anchor_score = float(final_scores[shortlist.index(anchor_action)].item())
        best_score = float(final_scores[best_local].item())
        supporting_members = sum(
            idx != self.anchor_idx for idx in candidate_members.get(best_action, set())
        )
        supporting_roles = len(candidate_roles.get(best_action, set()))
        alt_strength_sum = sum(
            features[idx]["strength"]
            for idx in candidate_members.get(best_action, set())
            if idx != self.anchor_idx
        )
        strong_alt_support = sum(
            features[idx]["strength"] >= 0.62
            for idx in candidate_members.get(best_action, set())
            if idx != self.anchor_idx
        )
        allow_override = (
            alt_strength_sum >= 1.25
            or (
                alt_strength_sum >= 0.62
                and supporting_roles >= 2
                and anchor_top_prob < 0.58
                and disagreement > 0.18
                and best_score - anchor_score > 0.10
            )
            or (
                supporting_members >= 2
                and supporting_roles >= 2
                and best_score - anchor_score > 0.14
            )
            or (
                desperation
                and alt_strength_sum >= 0.62
                and supporting_roles >= 2
                and best_score - anchor_score > 0.08
            )
        )
        if strong_anchor and not desperation:
            allow_override = allow_override and (
                alt_strength_sum >= 1.45
                or (
                    supporting_members >= 2
                    and supporting_roles >= 2
                    and best_score - anchor_score > 0.12
                )
            )
        override_margin = 0.05 if anchor_margin < 0.08 else 0.10
        if strong_anchor and not desperation:
            override_margin += 0.03
        if len(judge_order) >= 2 and len(shortlist) <= 4 and not full_rerank:
            override_margin = max(0.03, override_margin - 0.015)
            if supporting_members >= 1 and supporting_roles >= 2:
                allow_override = allow_override or (
                    alt_strength_sum >= 0.62 and best_score - anchor_score > 0.07
                )
        if desperation:
            override_margin = max(0.03, override_margin - 0.03)
        if stabilize:
            override_margin += 0.02
        if not forced_switch and (
            not allow_override or best_score - anchor_score < override_margin
        ):
            self._record_anchor_decision(
                default_anchor_action=default_anchor_action,
                anchor_action=anchor_action,
                proposed_best_action=best_action,
                final_action=anchor_action,
                forced_switch=forced_switch,
                single_judge=len(judge_order) == 1,
                full_rerank=full_rerank,
                shortlist_size=len(shortlist),
                proposer_variant_count=len(proposer_variants),
                judge_count=len(judge_order),
            )
            return anchor_action
        self._record_anchor_decision(
            default_anchor_action=default_anchor_action,
            anchor_action=anchor_action,
            proposed_best_action=best_action,
            final_action=best_action,
            forced_switch=forced_switch,
            single_judge=len(judge_order) == 1,
            full_rerank=full_rerank,
            shortlist_size=len(shortlist),
            proposer_variant_count=len(proposer_variants),
            judge_count=len(judge_order),
        )
        return best_action

    def _judge_shortlist(
        self,
        batch_idx: int,
        shortlist: list[int],
        judge_order: list[int],
        judge_weights: list[float],
        member_steps: list[dict[str, Any]],
        forced_switch: bool,
        proposal_support: dict[int, float],
    ) -> tuple[list[int], torch.Tensor]:
        single_judge = len(judge_order) == 1
        if single_judge:
            critic_mix = 0.35 if forced_switch else 0.15
            proposal_bonus = 0.18
        else:
            critic_mix = 0.25 if forced_switch else 0.10
            proposal_bonus = 0.50
        actor_mix = 1.0 - critic_mix
        final_scores = torch.zeros(len(shortlist), dtype=torch.float32)
        for judge_weight, judge_idx in zip(judge_weights, judge_order):
            step = member_steps[judge_idx]
            probs = step["probs"][batch_idx, shortlist].float().cpu()
            q_vals = self._score_candidates_with_critic(
                step=step,
                batch_idx=batch_idx,
                shortlist=shortlist,
            ).cpu()
            member_score = actor_mix * _zscore(probs)
            member_score = member_score + critic_mix * _zscore(q_vals)
            final_scores += judge_weight * member_score
        proposer_scores = torch.tensor(
            [proposal_support.get(action, 0.0) for action in shortlist],
            dtype=torch.float32,
        )
        final_scores += proposal_bonus * _zscore(proposer_scores)
        return shortlist, final_scores

    def _score_candidates_with_critic(
        self,
        step: dict[str, Any],
        batch_idx: int,
        shortlist: list[int],
    ) -> torch.Tensor:
        member = step["member"]
        policy = member.policy
        device = member.device
        traj_emb = step["traj_emb"][batch_idx : batch_idx + 1]
        num_gammas = len(policy.gammas)
        action_tensor = torch.tensor(shortlist, device=device, dtype=torch.long)
        one_hot = F.one_hot(action_tensor, num_classes=self.action_dim).float()
        one_hot = one_hot.view(1, len(shortlist), 1, 1, self.action_dim)
        one_hot = one_hot.repeat(1, 1, 1, num_gammas, 1)
        traj_emb = traj_emb.repeat(len(shortlist), 1, 1)
        critic_actions = policy.actor.policy_dist.action_from_buffer(one_hot)
        critic_values = policy.critics(traj_emb, critic_actions)
        if hasattr(policy.critics, "bin_dist_to_raw_vals"):
            critic_values = policy.critics.bin_dist_to_raw_vals(critic_values)
        else:
            critic_values = policy.popart(critic_values, normalized=False)
        critic_values = critic_values.mean(dim=3)
        return critic_values[0, :, 0, -1, 0].detach()


def build_heuristic_ensemble_experiment(
    *,
    reference_model_name: str,
    member_specs: list[EnsembleMemberSpec],
    expected_obs_space,
    expected_action_space,
    log: bool,
    action_temperature: float,
):
    import gin
    from metamon.rl.pretrained import get_pretrained_model

    if not member_specs:
        raise ValueError("Ensemble requires at least one member")
    devices = _parse_member_devices(len(member_specs))
    expected_obs_sig = _space_signature(expected_obs_space.gym_space)
    expected_action_n = expected_action_space.gym_space.n

    runtimes: list[_EnsembleMemberRuntime] = []
    reference_experiment = None
    reference_policy = None

    for idx, (spec, device) in enumerate(zip(member_specs, devices)):
        builder = get_pretrained_model(spec.model_name)
        if builder.action_space.gym_space.n != expected_action_n:
            raise ValueError(
                f"Ensemble member {spec.model_name} uses {builder.action_space.gym_space.n} actions, "
                f"expected {expected_action_n}"
            )
        if _space_signature(builder.observation_space.gym_space) != expected_obs_sig:
            raise ValueError(
                f"Observation space mismatch for {spec.model_name}; only compatible "
                "models may be combined in this ensemble."
            )

        checkpoint = spec.checkpoint
        if idx == 0:
            # Reuse the first compatible member's AMAGO shell so non-Kakuna
            # anchors can still participate in the same proposer/judge pipeline.
            gin.clear_config()
            reference_builder = builder
            reference_experiment = reference_builder.initialize_agent(
                checkpoint=checkpoint,
                log=log,
                action_temperature=action_temperature * spec.action_temperature,
            )
            reference_policy = reference_experiment.policy
            reference_policy.to(device)
            reference_policy.eval()
            runtimes.append(
                _EnsembleMemberRuntime(
                    spec=spec,
                    policy=reference_policy,
                    device=device,
                    experiment=reference_experiment,
                )
            )
            continue

        gin.clear_config()
        experiment = builder.initialize_agent(
            checkpoint=checkpoint,
            log=False,
            action_temperature=action_temperature * spec.action_temperature,
        )
        policy = experiment.policy
        policy.to(device)
        policy.eval()
        runtimes.append(
            _EnsembleMemberRuntime(
                spec=spec,
                policy=policy,
                device=device,
                experiment=experiment,
            )
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert reference_experiment is not None
    ensemble_policy = HeuristicRouterEnsemblePolicy(
        members=runtimes,
        action_dim=expected_action_n,
    )
    if os.environ.get("METAMON_ENSEMBLE_VERBOSE", "").lower() in {"1", "true", "yes"}:
        roster = ", ".join(
            f"{runtime.spec.model_name}@{runtime.spec.checkpoint or 'default'}->{runtime.device}"
            for runtime in runtimes
        )
        print(f"Ensemble roster: {roster}")
    reference_experiment.policy_aclr = ensemble_policy
    reference_experiment.sample_actions_val = False
    reference_experiment._ensemble_members = runtimes
    reference_experiment._ensemble_policy = ensemble_policy
    return reference_experiment
