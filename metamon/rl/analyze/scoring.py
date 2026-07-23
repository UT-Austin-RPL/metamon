"""Persistent pretrained-model session and full-sequence action-prob scoring."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import gin
import numpy as np
import torch
import torch.nn.functional as F
from amago.loading import RLData

from metamon.interface import UniversalAction, UniversalState
from metamon.rl.analyze.actions import action_label
from metamon.rl.pretrained import (
    ALL_PRETRAINED_MODELS,
    get_analyze_compatible_pretrained_model_names,
    get_pretrained_model,
)

# Analyze loads many architectures in one process. Metamon/AMAGO mark some
# nets with @torch.compile, which specializes on shapes like d_model; scoring
# Tauros (512) then Kakuna (400) hits dynamo recompile_limit and can 500.
# Disable compile here — ladder serving uses one policy per process.
torch._dynamo.config.disable = True


@dataclass
class LoadedModel:
    key: str
    name: str
    checkpoint: Optional[int]
    temperature: float
    device: str
    builder: Any
    experiment: Any
    policy: Any
    max_seq_len: int


class ModelSession:
    """Keep multiple pretrained policies resident on one device."""

    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device)
        self.models: dict[str, LoadedModel] = {}

    @staticmethod
    def available_names() -> list[str]:
        return get_analyze_compatible_pretrained_model_names()

    def loaded(self) -> list[dict]:
        out = []
        for m in self.models.values():
            out.append(
                {
                    "key": m.key,
                    "name": m.name,
                    "checkpoint": m.checkpoint,
                    "temperature": m.temperature,
                    "device": m.device,
                    "max_seq_len": m.max_seq_len,
                }
            )
        return out

    def _make_key(
        self,
        name: str,
        checkpoint: Optional[int],
        temperature: float = 1.0,
    ) -> str:
        parts = [name]
        if checkpoint is not None:
            parts.append(str(int(checkpoint)))
        if abs(float(temperature) - 1.0) > 1e-6:
            parts.append(f"t{float(temperature):g}")
        return "@".join(parts)

    def load(
        self,
        name: str,
        checkpoint: Optional[int] = None,
        temperature: float = 1.0,
    ) -> dict:
        temperature = float(temperature)
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        key = self._make_key(name, checkpoint, temperature)
        if key in self.models:
            return {"key": key, "status": "already_loaded"}
        cls = ALL_PRETRAINED_MODELS.get(name)
        if cls is None:
            raise ValueError(f"Unknown pretrained model: {name}")
        if not getattr(cls, "analyze_compatible", True):
            raise ValueError(
                f"{name} is an ensemble router and is not supported by analyze "
                "(set PretrainedModel.analyze_compatible to opt in)."
            )

        builder = get_pretrained_model(name)
        gin.clear_config()
        experiment = builder.initialize_agent(
            checkpoint=checkpoint,
            log=False,
            action_temperature=temperature,
        )
        policy = experiment.policy
        policy.to(self.device)
        policy.eval()
        max_seq_len = int(
            getattr(policy, "max_seq_len", None)
            or getattr(experiment, "max_seq_len", 128)
            or 128
        )
        resolved_ckpt = (
            checkpoint
            if checkpoint is not None
            else getattr(builder, "default_checkpoint", None)
        )
        self.models[key] = LoadedModel(
            key=key,
            name=name,
            checkpoint=resolved_ckpt,
            temperature=temperature,
            device=str(self.device),
            builder=builder,
            experiment=experiment,
            policy=policy,
            max_seq_len=max_seq_len,
        )
        return {
            "key": key,
            "name": name,
            "checkpoint": resolved_ckpt,
            "temperature": temperature,
            "device": str(self.device),
            "status": "loaded",
            "max_seq_len": max_seq_len,
        }

    def unload(self, key: str) -> dict:
        if key not in self.models:
            # also allow bare name if unique
            matches = [k for k in self.models if k == key or k.startswith(key + "@")]
            if len(matches) == 1:
                key = matches[0]
            else:
                raise KeyError(key)
        del self.models[key]
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return {"key": key, "status": "unloaded"}


def _states_actions_to_dataset_tuple(
    states: list[UniversalState],
    actions: list[int],
    observation_space,
    action_space,
    reward_function,
):
    """Mirror MetamonDataset.load_filename packaging (no disk)."""
    observation_space.reset()
    obs_list = [observation_space.state_to_obs(s) for s in states]
    nested_obs = defaultdict(list)
    for o in obs_list:
        for k, v in o.items():
            nested_obs[k].append(v)

    action_infos = {"chosen": [], "legal": [], "missing": []}
    for s, a_idx in zip(states, actions[:-1]):
        universal_action = UniversalAction(action_idx=int(a_idx))
        action_infos["chosen"].append(
            action_space.action_to_agent_output(s, universal_action)
        )
        action_infos["legal"].append(
            set(
                action_space.action_to_agent_output(s, l)
                for l in UniversalAction.maybe_valid_actions(s)
            )
        )
        action_infos["missing"].append(universal_action.missing)

    rewards = np.array(
        [reward_function(s_t, s_t1) for s_t, s_t1 in zip(states[:-1], states[1:])],
        dtype=np.float32,
    )
    dones = np.zeros_like(rewards, dtype=bool)
    if len(dones):
        dones[-1] = True
    return dict(nested_obs), action_infos, rewards, dones


def _to_rl_data(data, action_space) -> RLData:
    """Same conversion as MetamonAMAGODataset._process_data."""
    obs, action_infos, rewards, dones = data
    num_actions = action_space.gym_space.n
    actions_torch = F.one_hot(
        torch.tensor(action_infos["chosen"]).long().clamp(min=0),
        num_classes=num_actions,
    ).float()

    illegal_actions = torch.ones((len(action_infos["chosen"]) + 1, num_actions)).bool()
    for i, legal_actions in enumerate(action_infos["legal"]):
        for legal_agent_action in legal_actions:
            illegal_actions[i, int(legal_agent_action)] = False

    obs_torch = {k: torch.from_numpy(np.stack(v, axis=0)) for k, v in obs.items()}
    missing_acts = torch.tensor(action_infos["missing"] + [True]).unsqueeze(-1)
    obs_torch["missing_action_mask"] = missing_acts
    obs_torch["illegal_actions"] = illegal_actions
    rewards_torch = torch.from_numpy(rewards).unsqueeze(-1)
    dones_torch = torch.from_numpy(dones).unsqueeze(-1)
    time_idxs = torch.arange(len(action_infos["chosen"]) + 1).long().unsqueeze(-1)
    return RLData(
        obs=obs_torch,
        actions=actions_torch,
        rews=rewards_torch,
        dones=dones_torch,
        time_idxs=time_idxs,
    )


def _batchify(rl_data: RLData, device: torch.device):
    """Add batch dim B=1 and move to device."""
    obs = {k: v.unsqueeze(0).to(device) for k, v in rl_data.obs.items()}
    rl2s = rl_data.rl2s.unsqueeze(0).to(device)
    time_idxs = rl_data.time_idxs.unsqueeze(0).to(device)
    return obs, rl2s, time_idxs


def _probs_from_full_sequence(
    policy, obs, rl2s, time_idxs, n_decisions: int
) -> np.ndarray:
    """Parallel training-style forward when the whole traj fits in max_seq_len."""
    traj_emb, _ = policy.get_state_embedding(
        obs=obs, rl2s=rl2s, time_idxs=time_idxs, hidden_state=None
    )
    action_dists = policy.actor(
        traj_emb,
        straight_from_obs={k: obs[k] for k in policy.pass_obs_keys_to_actor},
    )
    # (B, L, G, A) → (n_decisions, A) test-time gamma
    return action_dists.probs[0, :n_decisions, -1, :].float().cpu().numpy()


def _probs_from_kv_rollout(
    policy, obs, rl2s, time_idxs, n_decisions: int
) -> np.ndarray:
    """Match ladder interact: L=1 steps with rolling Tformer KV cache.

    Mirrors ``Experiment.interact`` → ``Agent.get_actions(..., hidden_state=...)``
    and ``TformerHiddenState.update`` / ``Cache.roll_back`` once the cache is
    full (``seq_lens == max_seq_len``).
    """
    device = time_idxs.device
    # Must be eval: Transformer.forward only uses the KV path when not training.
    was_training = policy.training
    policy.eval()
    hidden = policy.init_hidden_state(batch_size=1, device=device)
    T_obs = time_idxs.shape[1]
    probs_list: list[np.ndarray] = []
    try:
        for t in range(T_obs):
            obs_t = {k: v[:, t : t + 1] for k, v in obs.items()}
            rl2s_t = rl2s[:, t : t + 1]
            time_t = time_idxs[:, t : t + 1]
            traj_emb, hidden = policy.get_state_embedding(
                obs=obs_t,
                rl2s=rl2s_t,
                time_idxs=time_t,
                hidden_state=hidden,
            )
            if t >= n_decisions:
                continue
            action_dists = policy.actor(
                traj_emb,
                straight_from_obs={k: obs_t[k] for k in policy.pass_obs_keys_to_actor},
            )
            # (B=1, L=1, G, A) → test-time gamma
            probs_list.append(action_dists.probs[0, 0, -1, :].float().cpu().numpy())
    finally:
        policy.train(was_training)
    if not probs_list:
        action_dim = int(obs["illegal_actions"].shape[-1])
        return np.zeros((0, action_dim), dtype=np.float32)
    return np.stack(probs_list, axis=0)


def _turn_prob_payload(
    builder,
    states: list[UniversalState],
    actions: list[int],
    probs: np.ndarray,
) -> list[dict[str, Any]]:
    turns_out = []
    action_dim = probs.shape[-1] if len(probs) else 0
    n_decisions = len(probs)
    for t in range(n_decisions):
        state = states[t]
        gt_univ = int(actions[t])
        missing = gt_univ < 0
        legal_idxs = []
        for ua in UniversalAction.maybe_valid_actions(state):
            agent_a = builder.action_space.action_to_agent_output(state, ua)
            legal_idxs.append(int(agent_a))
        legal_idxs = sorted(set(legal_idxs))
        p = probs[t].astype(np.float64)
        p_legal = p[legal_idxs] if legal_idxs else p
        z = float(p_legal.sum())
        if z > 1e-12:
            p_disp = (p_legal / z).tolist()
        else:
            p_disp = [0.0] * len(legal_idxs)
        labels = [action_label(state, i) for i in legal_idxs]
        p_gt = None
        agree = None
        if not missing:
            gt_agent = int(
                builder.action_space.action_to_agent_output(
                    state, UniversalAction(action_idx=gt_univ)
                )
            )
            p_gt = float(p[gt_agent]) if gt_agent < action_dim else None
            argmax = (
                legal_idxs[int(np.argmax(p_disp))] if legal_idxs else int(np.argmax(p))
            )
            agree = argmax == gt_agent

        turns_out.append(
            {
                "turn": t,
                "missing": missing,
                "gt": None if missing else gt_univ,
                "gt_label": action_label(state, gt_univ),
                "legal": legal_idxs,
                "labels": labels,
                "probs": p_disp,
                "p_gt": p_gt,
                "argmax": (
                    None if not legal_idxs else legal_idxs[int(np.argmax(p_disp))]
                ),
                "agree": agree,
            }
        )
    return turns_out


@torch.no_grad()
def score_replay_with_model(
    loaded: LoadedModel,
    states: list[UniversalState],
    actions: list[int],
) -> dict[str, Any]:
    """Score action probs for every decision turn (ladder-matched for long battles)."""
    builder = loaded.builder
    policy = loaded.policy
    device = torch.device(loaded.device)

    data = _states_actions_to_dataset_tuple(
        states,
        actions,
        builder.observation_space,
        builder.action_space,
        builder.reward_function,
    )
    rl_data = _to_rl_data(data, builder.action_space)
    T_obs = rl_data.obs["illegal_actions"].shape[0]
    n_decisions = len(rl_data.actions)
    max_len = loaded.max_seq_len
    # Obs length exceeds context → same sliding KV path as online interact.
    rolled_context = T_obs > max_len

    obs, rl2s, time_idxs = _batchify(rl_data, device)
    if rolled_context:
        probs = _probs_from_kv_rollout(policy, obs, rl2s, time_idxs, n_decisions)
    else:
        probs = _probs_from_full_sequence(policy, obs, rl2s, time_idxs, n_decisions)

    turns_out = _turn_prob_payload(builder, states, actions, probs)

    return {
        "model_key": loaded.key,
        "model_name": loaded.name,
        "checkpoint": loaded.checkpoint,
        "truncated": False,
        "rolled_context": rolled_context,
        "max_seq_len": loaded.max_seq_len,
        "num_turns_scored": len(turns_out),
        "turns": turns_out,
    }


def score_replay(
    session: ModelSession,
    states: list[UniversalState],
    actions: list[int],
) -> dict[str, Any]:
    if not session.models:
        return {"models": [], "warning": "No models loaded"}
    T_obs = len(states)
    long_battle = any(T_obs > m.max_seq_len for m in session.models.values())
    results = []
    for loaded in session.models.values():
        results.append(score_replay_with_model(loaded, states, actions))
    out: dict[str, Any] = {"models": results, "long_battle": long_battle}
    if long_battle:
        out["message"] = (
            "This is a long battle, please wait — scoring with sliding "
            "KV-cache context (same as ladder rollouts)."
        )
    return out
