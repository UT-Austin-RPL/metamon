from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
from torch import nn

from metamon.rl.experimental.ensemblev2.action_remap import (
    ActionTranslator,
    CANONICAL_ACTION_DIM,
)
from metamon.rl.experimental.ensemblev2.config import (
    EnsembleV2Config,
    EnsembleV2MemberSpec,
    member_prefix,
)
from metamon.rl.experimental.ensemblev2.decision import (
    EnsembleDecision,
    EnsembleDecisionContext,
    MemberStepFeatures,
    get_ensemble_decision,
)
from metamon.rl.experimental.ensemblev2.features import MemberFeatureExtractor
from metamon.rl.experimental.ensemblev2.observation import ENSEMBLE_STATE_KEY
from metamon.rl.experimental.ensemblev2.analysis import compute_disagreement_features
from metamon.rl.experimental.ensemblev2.logging import FeatureLogger
from metamon.rl.metamon_to_amago import pop_battle_outcome


def _parse_member_devices(num_members: int) -> list[torch.device]:
    raw = os.environ.get("METAMON_ENSEMBLEV2_MEMBER_DEVICES")
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


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class _EnsembleV2MemberRuntime:
    spec: EnsembleV2MemberSpec
    policy: nn.Module
    device: torch.device
    experiment: Any
    obs_keys: list[str]
    prefix: str
    translator: ActionTranslator
    extractor: MemberFeatureExtractor
    model_name: str
    checkpoint: Optional[int]
    action_space_name: str
    gammas: list[float]
    is_anchor: bool

    def description(self) -> dict[str, Any]:
        return {
            "member_index": None,
            "model_name": self.model_name,
            "checkpoint": self.checkpoint,
            "action_space_name": self.action_space_name,
            "is_anchor": self.is_anchor,
            "gammas": self.gammas,
        }


@dataclass
class _EnsembleV2Hidden:
    member_hidden: list[Any]
    history: list[list[dict[str, Any]]]


class _EnsembleV2TrajEncoderProxy:
    """Mimics the subset of TrajEncoder used by AMAGO eval loops.

    Holds per-member TrajEncoder hidden state plus a per-lane battle history of
    step records, and flushes a lane's history to the :class:`FeatureLogger` when
    that lane's battle finishes.
    """

    def __init__(
        self,
        members: list[_EnsembleV2MemberRuntime],
        logger: FeatureLogger,
    ):
        self.members = members
        self.logger = logger

    def init_hidden_state(self, batch_size: int, device: torch.device):
        return _EnsembleV2Hidden(
            member_hidden=[
                member.policy.traj_encoder.init_hidden_state(batch_size, member.device)
                for member in self.members
            ],
            history=[[] for _ in range(batch_size)],
        )

    def reset_hidden_state(self, hidden_state, dones):
        if hidden_state is None:
            return None
        if isinstance(dones, torch.Tensor):
            dones = dones.detach().cpu().numpy()
        dones = np.asarray(dones, dtype=bool).reshape(-1)

        if isinstance(hidden_state, _EnsembleV2Hidden):
            member_hidden = hidden_state.member_hidden
            history = list(hidden_state.history)
        else:
            member_hidden = hidden_state
            history = [[] for _ in range(len(dones))]

        reset_member_hidden = [
            member.policy.traj_encoder.reset_hidden_state(member_hidden_i, dones)
            for member, member_hidden_i in zip(self.members, member_hidden)
        ]
        for idx, done in enumerate(dones.tolist()):
            if done:
                won = pop_battle_outcome(idx)
                self.logger.log_battle(history[idx], won=won)
                history[idx] = []
        return _EnsembleV2Hidden(
            member_hidden=reset_member_hidden,
            history=history,
        )


class EnsembleV2Policy(nn.Module):
    """Inference-only ensemble over members with heterogeneous obs/action spaces.

    Splits the combined namespaced observation, runs each member, gathers per-gamma
    legal-normalized action distributions and per-action Q-values (remapped to
    canonical universal indices), then routes the final action through the pluggable
    :func:`make_ensembled_decision` hook. Outputs are in the canonical
    ``DefaultActionSpace`` (13) which is also the env's action space, so no
    outbound translation is needed.
    """

    def __init__(
        self,
        members: list[_EnsembleV2MemberRuntime],
        anchor_index: int,
        decision: EnsembleDecision,
        decision_name: str = "anchor",
        gather_q: Optional[bool] = None,
    ):
        super().__init__()
        self.members = members
        self.anchor_index = anchor_index
        self.decision = decision
        self.decision_name = decision_name
        self.action_dim = CANONICAL_ACTION_DIM
        if gather_q is None:
            gather_q = _env_flag("METAMON_ENSEMBLEV2_GATHER_Q", True)
        # Anchor-vs-others OOD diagnostics (logged per step). Needs >= 2 members
        # and Q gathering for the value/uncertainty features.
        self.analyze = (
            _env_flag("METAMON_ENSEMBLEV2_ANALYZE", True) and len(members) >= 2
        )
        if self.analyze and not gather_q:
            gather_q = True
        self.gather_q = gather_q

        member_descriptions = []
        for idx, member in enumerate(members):
            desc = member.description()
            desc["member_index"] = idx
            member_descriptions.append(desc)
        self.logger = FeatureLogger.from_env(
            member_descriptions, meta={"decision": decision_name}
        )
        self.traj_encoder = _EnsembleV2TrajEncoderProxy(members, self.logger)

    def eval(self):
        super().eval()
        for member in self.members:
            member.policy.eval()
        return self

    def _state_hashes(
        self, obs: dict[str, torch.Tensor], batch_size: int
    ) -> list[Optional[str]]:
        """Compact per-lane signature of the current state (anchor obs hash).

        Quantizes the anchor member's current-timestep observation and hashes it
        per lane so stateless deciders can detect repeated states (cycle/stall
        breaking). Obs-space-agnostic: works for any member's keys/dtypes.
        """
        anchor = self.members[self.anchor_index]
        parts = []
        for key in anchor.obs_keys:
            tensor = obs[f"{anchor.prefix}/{key}"][:, -1].reshape(batch_size, -1)
            if tensor.is_floating_point():
                # Coarse quantization (quarters) matching the v1 cycle key. Fine
                # bucketing (e.g. *32) lets small HP drift change the hash every
                # turn, so the stall breaker almost never fires.
                tensor = torch.round(tensor * 4.0).to(torch.int32)
            else:
                tensor = tensor.to(torch.int32)
            parts.append(tensor.cpu())
        if not parts:
            return [None] * batch_size
        combined = torch.cat(parts, dim=1).numpy()
        return [
            hashlib.blake2b(combined[b].tobytes(), digest_size=8).hexdigest()
            for b in range(batch_size)
        ]

    def get_actions(
        self,
        obs: dict[str, torch.Tensor],
        rl2s: torch.Tensor,
        time_idxs: torch.Tensor,
        hidden_state=None,
        sample: bool = True,
    ):
        first_value = next(iter(obs.values()))
        batch_size = first_value.shape[0]
        output_device = first_value.device

        if hidden_state is None:
            hidden_state = self.traj_encoder.init_hidden_state(
                batch_size, output_device
            )
        if isinstance(hidden_state, _EnsembleV2Hidden):
            member_hidden = hidden_state.member_hidden
            history = hidden_state.history
        else:
            member_hidden = hidden_state
            history = [[] for _ in range(batch_size)]

        canonical_illegal = obs["illegal_actions"].bool()  # (B, L, 13)

        next_hidden: list[Any] = []
        per_member_probs: list[np.ndarray] = []
        per_member_q: list[Optional[np.ndarray]] = []
        per_member_q_std: list[Optional[np.ndarray]] = []
        for member in self.members:
            member_obs = {key: obs[f"{member.prefix}/{key}"] for key in member.obs_keys}
            nh, probs_np, q_np, q_std_np = member.extractor.forward(
                member_obs=member_obs,
                rl2s=rl2s,
                time_idxs=time_idxs,
                hidden_state=member_hidden[len(next_hidden)],
                canonical_illegal=canonical_illegal,
                gather_q=self.gather_q,
            )
            next_hidden.append(nh)
            per_member_probs.append(probs_np)
            per_member_q.append(q_np)
            per_member_q_std.append(q_std_np)

        illegal_np = canonical_illegal[:, -1, :].cpu().numpy()  # (B, 13)
        turn_np = time_idxs.reshape(batch_size, -1)[:, -1].cpu().numpy()
        prev_reward_np = rl2s[:, -1, 0].float().cpu().numpy()
        state_hashes = self._state_hashes(obs, batch_size)
        opp_rem_t = obs.get(ENSEMBLE_STATE_KEY)
        opp_rem_np = (
            None
            if opp_rem_t is None
            else opp_rem_t[:, -1].reshape(batch_size, -1)[:, 0].float().cpu().numpy()
        )

        actions: list[int] = []
        for b in range(batch_size):
            legal = [a for a in range(self.action_dim) if not bool(illegal_np[b, a])]
            if not legal:
                actions.append(0)
                continue

            member_feats: list[MemberStepFeatures] = []
            for idx, member in enumerate(self.members):
                q_vals = per_member_q[idx]
                q_stds = per_member_q_std[idx]
                if q_vals is None:
                    q_b = np.full(
                        (len(member.gammas), self.action_dim),
                        np.nan,
                        dtype=np.float32,
                    )
                else:
                    q_b = q_vals[b]
                q_std_b = None if q_stds is None else q_stds[b]
                member_feats.append(
                    MemberStepFeatures(
                        member_index=idx,
                        model_name=member.model_name,
                        checkpoint=member.checkpoint,
                        action_space_name=member.action_space_name,
                        is_anchor=(idx == self.anchor_index),
                        gammas=member.gammas,
                        probs=per_member_probs[idx][b],
                        q_values=q_b,
                        q_std=q_std_b,
                    )
                )

            context = EnsembleDecisionContext(
                turn_idx=int(turn_np[b]),
                legal_actions=legal,
                members=member_feats,
                anchor_index=self.anchor_index,
                prev_reward=float(prev_reward_np[b]),
                history=history[b],
                state_hash=state_hashes[b],
                opponents_remaining=(
                    None if opp_rem_np is None else float(opp_rem_np[b])
                ),
            )
            chosen = self.decision(context)
            if self.analyze:
                feats = compute_disagreement_features(context)
                if feats:
                    context.diagnostics["disagreement"] = feats
            history[b].append(context.step_record(chosen))
            actions.append(int(chosen))

        actions_t = torch.tensor(actions, device=output_device, dtype=torch.uint8)
        return actions_t.view(batch_size, 1, 1), _EnsembleV2Hidden(
            member_hidden=next_hidden,
            history=history,
        )


def build_ensemblev2_experiment(
    *,
    config: EnsembleV2Config,
    log: bool,
    action_temperature: float,
):
    """Load each member, wrap them in an :class:`EnsembleV2Policy`, and return an
    AMAGO experiment shell (reusing the anchor/first member's) for eval.

    Unlike the v1 ensemble builder, this performs NO observation-space or
    action-space compatibility check: members may differ in both.
    """
    import gin
    from metamon.rl.pretrained import get_pretrained_model

    if not config.members:
        raise ValueError("EnsembleV2 requires at least one member")

    devices = _parse_member_devices(len(config.members))
    runtimes: list[_EnsembleV2MemberRuntime] = []
    reference_experiment = None

    for idx, (spec, device) in enumerate(zip(config.members, devices)):
        builder = get_pretrained_model(spec.model_name)
        gin.clear_config()
        experiment = builder.initialize_agent(
            checkpoint=spec.checkpoint,
            log=log and idx == 0,
            action_temperature=action_temperature,
        )
        policy = experiment.policy
        policy.to(device)
        policy.eval()

        translator = ActionTranslator(builder.action_space)
        extractor = MemberFeatureExtractor(
            policy=policy, device=device, translator=translator
        )
        resolved_checkpoint = (
            spec.checkpoint
            if spec.checkpoint is not None
            else builder.default_checkpoint
        )
        runtime = _EnsembleV2MemberRuntime(
            spec=spec,
            policy=policy,
            device=device,
            experiment=experiment,
            obs_keys=list(builder.observation_space.gym_space.spaces.keys()),
            prefix=member_prefix(idx),
            translator=translator,
            extractor=extractor,
            model_name=spec.model_name,
            checkpoint=resolved_checkpoint,
            action_space_name=type(builder.action_space).__name__,
            gammas=[float(g) for g in policy.gammas.tolist()],
            is_anchor=(idx == config.anchor_index),
        )
        runtimes.append(runtime)

        if idx == 0:
            reference_experiment = experiment
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    assert reference_experiment is not None
    decision = get_ensemble_decision(config.decision, **config.decision_kwargs)
    ensemble_policy = EnsembleV2Policy(
        members=runtimes,
        anchor_index=config.anchor_index,
        decision=decision,
        decision_name=config.decision,
    )
    if _env_flag("METAMON_ENSEMBLEV2_VERBOSE", False):
        roster = ", ".join(
            f"{r.model_name}@{r.checkpoint}->{r.device}"
            + ("[anchor]" if r.is_anchor else "")
            for r in runtimes
        )
        log_path = ensemble_policy.logger.path
        print(f"EnsembleV2 roster: {roster}")
        print(f"EnsembleV2 decision: {config.decision}")
        print(f"EnsembleV2 feature log: {log_path or 'disabled'}")

    reference_experiment.policy_aclr = ensemble_policy
    reference_experiment.sample_actions_val = False
    reference_experiment._ensemblev2_policy = ensemble_policy
    reference_experiment._ensemblev2_members = runtimes
    return reference_experiment
