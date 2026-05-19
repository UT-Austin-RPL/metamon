import copy
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

from metamon.interface import (
    ActionSpace,
    UniversalState,
    consistent_move_order,
    consistent_pokemon_order,
)


ACTION_COLUMNS = [
    "rank",
    "agent_action",
    "action",
    "probability",
    "legal",
    "selected",
]


@dataclass
class PreviewSnapshot:
    status: str = "Waiting for the first policy decision..."
    updated_at: float = field(default_factory=time.time)
    battle: str = ""
    turn: Optional[int] = None
    value_estimate: Optional[float] = None
    value_by_gamma: list[float] = field(default_factory=list)
    selected_action: Optional[int] = None
    selected_action_label: str = ""
    action_rows: list[list[Any]] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class PreviewStateStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._snapshot = PreviewSnapshot()

    def update(self, snapshot: PreviewSnapshot):
        with self._lock:
            self._snapshot = snapshot

    def get(self) -> PreviewSnapshot:
        with self._lock:
            return copy.deepcopy(self._snapshot)


def launch_gradio_preview(
    store: PreviewStateStore,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False,
):
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError(
            "The Showdown AI preview UI requires gradio. Install it with `pip install gradio`."
        ) from exc

    def read_snapshot():
        snapshot = store.get()
        age = max(0.0, time.time() - snapshot.updated_at)
        summary = [
            "# Metamon AI Preview",
            f"**Status:** {snapshot.status}",
            f"**Last update:** {age:.1f}s ago",
        ]
        if snapshot.battle:
            summary.append(f"**Battle:** `{snapshot.battle}`")
        if snapshot.turn is not None:
            summary.append(f"**Turn:** {snapshot.turn}")
        if snapshot.value_estimate is not None:
            summary.append(
                f"**Value-head estimate:** `{snapshot.value_estimate:.4f}`"
            )
        if snapshot.selected_action is not None:
            summary.append(
                f"**Selected action:** `{snapshot.selected_action}` {snapshot.selected_action_label}"
            )
        if snapshot.error:
            summary.append(f"**Preview error:** `{snapshot.error}`")

        state_json = json.dumps(snapshot.state, indent=2, sort_keys=True)
        value_json = {
            "policy_gamma_value": snapshot.value_estimate,
            "all_gamma_values": snapshot.value_by_gamma,
        }
        return "\n\n".join(summary), snapshot.action_rows, value_json, state_json

    with gr.Blocks(title="Metamon AI Preview") as demo:
        gr.Markdown(
            "Live policy preview for the running Showdown battle. "
            "The action table is the model's full distribution for the current decision."
        )
        with gr.Row():
            summary = gr.Markdown()
            value = gr.JSON(label="Value Head")
        actions = gr.Dataframe(
            headers=ACTION_COLUMNS,
            datatype=["number", "number", "str", "number", "bool", "bool"],
            interactive=False,
            label="Policy Distribution",
        )
        state = gr.Code(label="Current State", language="json", lines=24)
        refresh = gr.Button("Refresh")
        refresh.click(read_snapshot, outputs=[summary, actions, value, state])
        if hasattr(gr, "Timer"):
            timer = gr.Timer(1.0)
            timer.tick(read_snapshot, outputs=[summary, actions, value, state])
        demo.load(read_snapshot, outputs=[summary, actions, value, state])

    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        prevent_thread_lock=True,
        quiet=True,
    )
    return demo


def _unwrap_attr(obj: Any, attr: str) -> Any:
    current = obj
    seen = set()
    for _ in range(16):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        if hasattr(current, attr):
            return getattr(current, attr)
        current = getattr(current, "env", None)
    return None


def _current_battle_tag(env: Any) -> str:
    battle = _unwrap_attr(env, "current_battle")
    return getattr(battle, "battle_tag", "") if battle is not None else ""


def _current_turn(env: Any) -> Optional[int]:
    battle = _unwrap_attr(env, "current_battle")
    return getattr(battle, "turn", None) if battle is not None else None


def _describe_action(action_space: ActionSpace, state: UniversalState, idx: int) -> str:
    universal = action_space.agent_output_to_action(state=state, agent_output=idx)
    action_idx = universal.action_idx
    tera = action_idx >= 9
    if tera:
        action_idx -= 9

    if 0 <= action_idx <= 3:
        moves = consistent_move_order(state.player_active_pokemon.moves)
        if action_idx < len(moves):
            prefix = "Tera " if tera else "Move "
            return f"{prefix}{action_idx + 1}: {moves[action_idx].name}"
        return f"{'Tera ' if tera else ''}Move {action_idx + 1}: unavailable"

    if 4 <= action_idx <= 8:
        switches = consistent_pokemon_order(state.available_switches)
        switch_idx = action_idx - 4
        if switch_idx < len(switches):
            return f"Switch {switch_idx + 1}: {switches[switch_idx].name}"
        return f"Switch {switch_idx + 1}: unavailable"

    return f"Action {universal.action_idx}"


def _policy_preview_forward(
    experiment,
    obs: dict[str, torch.Tensor],
    rl2s: torch.Tensor,
    time_idxs: torch.Tensor,
    hidden_state: Any,
):
    policy = experiment.policy
    tstep_emb = policy.tstep_encoder(obs=obs, rl2s=rl2s)
    traj_emb_t, next_hidden_state = policy.traj_encoder(
        tstep_emb, time_idxs=time_idxs, hidden_state=hidden_state
    )
    straight_from_obs = {k: obs[k] for k in policy.pass_obs_keys_to_actor}
    action_dists = policy.actor(
        traj_emb_t,
        straight_from_obs=straight_from_obs,
    )
    if experiment.sample_actions:
        actions = action_dists.sample()
    elif policy.discrete:
        actions = torch.argmax(action_dists.probs, dim=-1, keepdim=True)
    else:
        actions = action_dists.mean
    actions = actions[..., -1, :]
    dtype = torch.uint8 if (policy.discrete or policy.multibinary) else torch.float32

    value_by_gamma = []
    value_estimate = None
    value_error = None
    if policy.discrete and hasattr(action_dists, "probs"):
        try:
            action_dim = action_dists.probs.shape[-1]
            num_gammas = action_dists.probs.shape[-2]
            all_actions = torch.eye(
                action_dim,
                device=traj_emb_t.device,
                dtype=traj_emb_t.dtype,
            )
            all_actions = all_actions.view(action_dim, 1, 1, 1, action_dim).expand(
                action_dim, 1, 1, num_gammas, action_dim
            )
            critic_output = policy.critics(traj_emb_t, all_actions)
            if hasattr(critic_output, "probs") and hasattr(
                policy.critics, "bin_dist_to_raw_vals"
            ):
                q_all_actions = policy.critics.bin_dist_to_raw_vals(critic_output)
            else:
                q_all_actions = critic_output
            q_all_actions = q_all_actions.mean(3).squeeze(-1)
            probs_by_gamma = action_dists.probs[0, -1].transpose(0, 1)
            q_by_gamma = q_all_actions[:, 0, -1, :]
            values = (q_by_gamma * probs_by_gamma).sum(0).detach().float().cpu().tolist()
            value_by_gamma = [float(v) for v in values]
            value_estimate = value_by_gamma[-1] if value_by_gamma else None
        except Exception as exc:
            value_error = f"value estimate unavailable: {exc!r}"

    probs = None
    if hasattr(action_dists, "probs"):
        probs = action_dists.probs[0, -1, -1, :].detach().float().cpu().numpy()

    return (
        actions.to(dtype=dtype),
        next_hidden_state,
        probs,
        value_estimate,
        value_by_gamma,
        value_error,
    )


def _as_policy_tensors(current_timestep, device):
    obs, rl2s, time_idxs = current_timestep
    obs = {
        key: torch.from_numpy(value).to(device).unsqueeze(1)
        for key, value in obs.items()
    }
    rl2s = torch.from_numpy(rl2s).to(device).unsqueeze(1)
    time_idxs = torch.from_numpy(time_idxs).to(device).unsqueeze(1)
    return obs, rl2s, time_idxs


def _build_snapshot(
    env: Any,
    action_space: ActionSpace,
    probs: Optional[np.ndarray],
    legal_actions: list[int],
    selected_action: Optional[int],
    value_estimate: Optional[float],
    value_by_gamma: list[float],
    error: Optional[str] = None,
) -> PreviewSnapshot:
    state = _unwrap_attr(env, "_most_recent_state")
    if state is None:
        return PreviewSnapshot(status="Waiting for battle state...", error=error)

    action_rows = []
    selected_label = ""
    if probs is not None:
        legal = set(int(a) for a in legal_actions)
        order = np.argsort(-probs)
        for rank, action_idx in enumerate(order, start=1):
            action_idx = int(action_idx)
            label = _describe_action(action_space, state, action_idx)
            if selected_action == action_idx:
                selected_label = label
            action_rows.append(
                [
                    rank,
                    action_idx,
                    label,
                    round(float(probs[action_idx]), 6),
                    action_idx in legal,
                    selected_action == action_idx,
                ]
            )

    return PreviewSnapshot(
        status="Running",
        battle=_current_battle_tag(env),
        turn=_current_turn(env),
        value_estimate=value_estimate,
        value_by_gamma=value_by_gamma,
        selected_action=selected_action,
        selected_action_label=selected_label,
        action_rows=action_rows,
        state=state.to_dict(),
        error=error,
    )


def run_showdown_with_preview(
    experiment,
    make_env,
    action_space: ActionSpace,
    timesteps: int,
    episodes: Optional[int],
    server_name: str,
    server_port: int,
    share: bool,
) -> dict[str, float]:
    from amago.envs.amago_env import SequenceWrapper

    store = PreviewStateStore()
    launch_gradio_preview(
        store=store,
        server_name=server_name,
        server_port=server_port,
        share=share,
    )
    print(f"AI preview UI: http://{server_name}:{server_port}")

    policy = experiment.policy
    policy.eval()
    env = SequenceWrapper(make_env(), save_trajs_to=None, save_every=None)
    env.reset()
    hidden_state = policy.traj_encoder.init_hidden_state(1, experiment.DEVICE)

    episodes_finished = 0
    try:
        for _ in range(timesteps):
            obs, rl2s, time_idxs = _as_policy_tensors(
                env.current_timestep, experiment.DEVICE
            )
            selected_action = None
            preview_error = None
            with torch.no_grad():
                with experiment.caster():
                    try:
                        (
                            actions,
                            next_hidden_state,
                            probs,
                            value_estimate,
                            values,
                            preview_error,
                        ) = (
                            _policy_preview_forward(
                                experiment=experiment,
                                obs=obs,
                                rl2s=rl2s,
                                time_idxs=time_idxs,
                                hidden_state=hidden_state,
                            )
                        )
                    except Exception as exc:
                        preview_error = repr(exc)
                        actions, next_hidden_state = policy.get_actions(
                            obs=obs,
                            rl2s=rl2s,
                            time_idxs=time_idxs,
                            sample=experiment.sample_actions,
                            hidden_state=hidden_state,
                        )
                        probs, value_estimate, values = None, None, []

            # SequenceWrapper is wrapping one AMAGOEnv directly, not a vector env.
            # DiscreteActionWrapper expects shape (1,) here; AMAGO's vector eval loop
            # uses shape (num_envs, 1), which leaks through to Showdown as an array.
            action_np = actions[0, 0].cpu().numpy()
            if action_np.size:
                selected_action = int(np.asarray(action_np).reshape(-1)[0])
            legal_actions = _unwrap_attr(env, "_most_recent_legal_actions") or []
            store.update(
                _build_snapshot(
                    env=env,
                    action_space=action_space,
                    probs=probs,
                    legal_actions=legal_actions,
                    selected_action=selected_action,
                    value_estimate=value_estimate,
                    value_by_gamma=values,
                    error=preview_error,
                )
            )

            *_, terminated, truncated, _ = env.step(action_np)
            done = np.logical_or(terminated, truncated)
            hidden_state = policy.traj_encoder.reset_hidden_state(
                next_hidden_state, done
            )
            if done.any():
                episodes_finished += int(done.sum())
                if episodes is not None and episodes_finished >= episodes:
                    break
                env.reset()
    finally:
        env.close()

    logs = experiment.policy_metrics([env.return_history], [env.special_history])
    experiment.log(logs, key="test")
    return logs
