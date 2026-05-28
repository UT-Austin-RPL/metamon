#!/usr/bin/env python
"""Visualize policy logit/probability changes on one Lapras replay state."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import html
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("METAMON_CACHE_DIR", "/home/eddie/metamon_cache")

import gin
import numpy as np
import torch

from metamon.data.parsed_replay_dset import ParsedReplayDataset
from metamon.interface import (
    UniversalAction,
    UniversalState,
    consistent_move_order,
    consistent_pokemon_order,
)
from metamon.rl.metamon_to_amago import MetamonAMAGODataset
from metamon.rl.pretrained import get_pretrained_model

# Ensure local Gen 1 model registrations are loaded when this script is invoked
# directly.
import metamon.rl.gen1_binary_models  # noqa: F401


DEFAULT_REPLAY_ROOT = Path(
    "/home/eddie/metamon/trajectories/lapras/splits/v1/train/lapras"
)
DEFAULT_OUTPUT_DIR = Path("/home/eddie/metamon/evals/logit_visualizations")

MODEL_SPECS = (
    ("Articuno", 40),
    ("lapras_bc_kl_anchor_actor", 0),
    ("lapras_bc_kl_anchor_actor", 2),
    ("lapras_bc_kl_anchor_actor", 5),
)


@dataclass(frozen=True)
class Situation:
    replay: str
    turn: int
    score: float
    active: str
    opponent: str
    legal_actions: tuple[int, ...]
    recorded_action: int
    recorded_label: str


@dataclass(frozen=True)
class PolicyOutput:
    model: str
    checkpoint: int
    logits: np.ndarray
    probs: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Articuno and Lapras finetune policy logits/probabilities "
            "on the exact same Gen 1 replay-derived decision."
        )
    )
    parser.add_argument(
        "--replay_root",
        type=Path,
        default=DEFAULT_REPLAY_ROOT,
        help="Root containing the Lapras replay split and index.csv.",
    )
    parser.add_argument(
        "--replay",
        default=None,
        help=(
            "Replay path, path relative to replay_root, or basename. If omitted, "
            "the script auto-selects a nontrivial candidate."
        ),
    )
    parser.add_argument(
        "--turn",
        type=int,
        default=None,
        help="Decision turn index in dataset action order. If omitted, auto-select.",
    )
    parser.add_argument(
        "--top_k_situations",
        type=int,
        default=5,
        help="Number of candidate situations to print when scanning.",
    )
    parser.add_argument(
        "--max_replays_to_scan",
        type=int,
        default=25,
        help="Maximum replay files to scan when auto-selecting a situation.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV and SVG outputs.",
    )
    parser.add_argument(
        "--unmasked_logits",
        action="store_true",
        help=(
            "Temporarily disable the actor illegal-action mask only for the "
            "direct actor_network_forward logit call. Probabilities still come "
            "from policy.actor with its normal mask and MetamonDiscrete clipping."
        ),
    )
    return parser.parse_args()


def finite_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.9g}"


def sanitize_stem(name: str) -> str:
    stem = Path(name).name
    for suffix in (".json.lz4", ".json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)


def short_replay_name(filename: str, replay_root: Path) -> str:
    path = Path(filename)
    try:
        return str(path.relative_to(replay_root))
    except ValueError:
        return path.name


def move_label(move: Any) -> str:
    return getattr(move, "name", str(move))


def pokemon_label(pokemon: Any) -> str:
    name = getattr(pokemon, "name", str(pokemon))
    hp_pct = getattr(pokemon, "hp_pct", None)
    status = getattr(pokemon, "status", None)
    details = []
    if hp_pct is not None:
        details.append(f"{100.0 * float(hp_pct):.0f}%")
    if status and status != "nostatus":
        details.append(str(status))
    return f"{name} ({', '.join(details)})" if details else name


def action_labels(state: UniversalState) -> list[str]:
    labels: list[str] = []
    moves = consistent_move_order(state.player_active_pokemon.moves)[:4]
    switches = consistent_pokemon_order(state.available_switches)[:5]
    for idx in range(4):
        if idx < len(moves):
            labels.append(f"{idx}: move {move_label(moves[idx])}")
        else:
            labels.append(f"{idx}: move <empty>")
    for idx in range(5):
        action_idx = 4 + idx
        if idx < len(switches):
            labels.append(f"{action_idx}: switch {pokemon_label(switches[idx])}")
        else:
            labels.append(f"{action_idx}: switch <empty>")
    return labels


def legal_agent_actions(state: UniversalState, action_space: Any) -> set[int]:
    legal = set()
    for action in UniversalAction.maybe_valid_actions(state):
        legal.add(action_space.action_to_agent_output(state, copy.copy(action)))
    return legal


def recorded_agent_action(
    state: UniversalState, action_space: Any, raw_action_idx: int
) -> int:
    return action_space.action_to_agent_output(
        state, UniversalAction(action_idx=int(raw_action_idx))
    )


def resolve_replay(dataset: ParsedReplayDataset, replay: str | None) -> str:
    if replay is None:
        if not dataset.filenames:
            raise ValueError("No replay files were found in the replay root.")
        return dataset.filenames[0]

    replay_path = Path(replay)
    candidates: list[Path] = []
    if replay_path.is_absolute():
        candidates.append(replay_path)
    else:
        candidates.append(Path(dataset.dset_root) / replay_path)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    matches = []
    for filename in dataset.filenames:
        path = Path(filename)
        if path.name == replay or str(path).endswith(replay):
            matches.append(filename)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        examples = "\n".join(f"  {m}" for m in matches[:10])
        raise ValueError(f"Replay is ambiguous. Matches include:\n{examples}")
    raise FileNotFoundError(f"Replay was not found under {dataset.dset_root}: {replay}")


def load_states_and_raw_actions(
    dataset: ParsedReplayDataset, filename: str
) -> tuple[list[UniversalState], list[int]]:
    raw = dataset._load_json(filename)
    states = [UniversalState.from_dict(copy.deepcopy(s)) for s in raw["states"]]
    raw_actions = [int(a) for a in raw["actions"][:-1]]
    return states, raw_actions


def score_situation(
    filename: str,
    turn: int,
    state: UniversalState,
    raw_action_idx: int,
    total_actions: int,
    action_space: Any,
) -> Situation:
    legal = legal_agent_actions(state, action_space)
    recorded = recorded_agent_action(state, action_space, raw_action_idx)
    labels = action_labels(state)
    has_move = any(0 <= a <= 3 for a in legal)
    has_switch = any(4 <= a <= 8 for a in legal)
    midgame_bonus = max(0.0, 1.0 - abs(turn - (total_actions / 2)) / max(total_actions, 1))
    score = 0.0
    score += 100.0 if has_move and has_switch else 0.0
    score += 20.0 if recorded >= 0 else 0.0
    score += 20.0 if recorded in legal else 0.0
    score += 15.0 if 0 < len(legal) < 9 else 0.0
    score += 10.0 if turn >= 3 else 0.0
    score += 10.0 * midgame_bonus
    if state.forced_switch:
        score -= 25.0
    return Situation(
        replay=filename,
        turn=turn,
        score=score,
        active=state.player_active_pokemon.name,
        opponent=state.opponent_active_pokemon.name,
        legal_actions=tuple(sorted(int(a) for a in legal)),
        recorded_action=recorded,
        recorded_label=labels[recorded] if 0 <= recorded < len(labels) else "missing",
    )


def find_situations(
    dataset: ParsedReplayDataset,
    action_space: Any,
    max_replays_to_scan: int,
    top_k: int,
) -> list[Situation]:
    situations: list[Situation] = []
    for filename in dataset.filenames[: max(0, max_replays_to_scan)]:
        try:
            states, raw_actions = load_states_and_raw_actions(dataset, filename)
        except Exception as exc:
            print(f"Skipping unreadable replay {filename}: {exc}")
            continue
        for turn, raw_action_idx in enumerate(raw_actions):
            if turn >= len(states):
                break
            situation = score_situation(
                filename=filename,
                turn=turn,
                state=states[turn],
                raw_action_idx=raw_action_idx,
                total_actions=len(raw_actions),
                action_space=action_space,
            )
            legal = set(situation.legal_actions)
            if any(0 <= a <= 3 for a in legal) and any(4 <= a <= 8 for a in legal):
                situations.append(situation)
    situations.sort(key=lambda s: (-s.score, s.replay, s.turn))
    return situations[: max(1, top_k)]


def choose_situation(
    dataset: ParsedReplayDataset,
    action_space: Any,
    replay: str | None,
    turn: int | None,
    top_k: int,
    max_replays_to_scan: int,
) -> Situation:
    if replay is not None:
        filename = resolve_replay(dataset, replay)
        states, raw_actions = load_states_and_raw_actions(dataset, filename)
        if not raw_actions:
            raise ValueError(f"Replay has no actions: {filename}")
        if turn is None:
            candidates = [
                score_situation(
                    filename,
                    i,
                    states[i],
                    raw_action_idx,
                    len(raw_actions),
                    action_space,
                )
                for i, raw_action_idx in enumerate(raw_actions)
                if i < len(states)
            ]
            candidates.sort(key=lambda s: (-s.score, s.turn))
            return candidates[0]
        if turn < 0 or turn >= len(raw_actions):
            raise ValueError(
                f"Turn {turn} is outside valid decision range 0..{len(raw_actions) - 1}"
            )
        return score_situation(
            filename, turn, states[turn], raw_actions[turn], len(raw_actions), action_space
        )

    situations = find_situations(dataset, action_space, max_replays_to_scan, top_k)
    if situations:
        print("Candidate situations:")
        for idx, situation in enumerate(situations, start=1):
            replay_name = short_replay_name(situation.replay, Path(dataset.dset_root))
            print(
                f"  {idx}. turn={situation.turn} score={situation.score:.1f} "
                f"active={situation.active} vs={situation.opponent} "
                f"legal={list(situation.legal_actions)} recorded={situation.recorded_label} "
                f"replay={replay_name}"
            )
        return situations[0]

    filename = resolve_replay(dataset, None)
    states, raw_actions = load_states_and_raw_actions(dataset, filename)
    if not raw_actions:
        raise ValueError(f"Replay has no actions: {filename}")
    fallback_turn = 0 if turn is None else turn
    return score_situation(
        filename,
        fallback_turn,
        states[fallback_turn],
        raw_actions[fallback_turn],
        len(raw_actions),
        action_space,
    )


def build_dataset(builder: Any, replay_root: Path) -> ParsedReplayDataset:
    return ParsedReplayDataset(
        observation_space=builder.observation_space,
        action_space=builder.action_space,
        reward_function=builder.reward_function,
        dset_root=str(replay_root),
        formats=["gen1ou"],
        verbose=False,
        shuffle=False,
        use_cached_filenames=True,
    )


def build_prefix_tensors(
    dataset: ParsedReplayDataset, filename: str, turn: int, device: torch.device
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, Any]]:
    data = dataset.load_filename(filename)
    obs, action_infos, _rewards, _dones = data
    if turn < 0 or turn >= len(action_infos["chosen"]):
        raise ValueError(
            f"Turn {turn} is outside valid decision range 0..{len(action_infos['chosen']) - 1}"
        )
    rl_data = MetamonAMAGODataset(dataset)._process_data(data)
    seq = slice(0, turn + 1)
    obs_tensors = {
        key: value[seq].unsqueeze(0).to(device)
        for key, value in rl_data.obs.items()
    }
    rl2s = rl_data.rl2s[seq].unsqueeze(0).to(device)
    time_idxs = rl_data.time_idxs[seq].unsqueeze(0).to(device)
    info = {
        "chosen": action_infos["chosen"][turn],
        "legal": sorted(int(a) for a in action_infos["legal"][turn]),
        "missing": action_infos["missing"][turn],
        "seq_len": turn + 1,
    }
    return obs_tensors, rl2s, time_idxs, info


def action_vector(tensor: torch.Tensor, name: str) -> np.ndarray:
    if tensor.ndim == 4:
        values = tensor[0, -1, -1, :]
    elif tensor.ndim == 3:
        values = tensor[0, -1, :]
    else:
        raise ValueError(f"{name} has unexpected shape {tuple(tensor.shape)}")
    arr = values.detach().float().cpu().numpy()
    if arr.shape[-1] != 9:
        raise ValueError(f"{name} reduced to {arr.shape[-1]} actions, expected 9")
    return arr


def forward_policy(
    model_name: str,
    checkpoint: int,
    obs_cpu: dict[str, torch.Tensor],
    rl2s_cpu: torch.Tensor,
    time_idxs_cpu: torch.Tensor,
    unmasked_logits: bool,
) -> PolicyOutput:
    gin.clear_config()
    experiment = get_pretrained_model(model_name).initialize_agent(
        checkpoint=checkpoint,
        log=False,
        action_temperature=1.0,
    )
    policy = experiment.policy
    policy.eval()
    device = torch.device(experiment.DEVICE)
    obs = {key: value.to(device) for key, value in obs_cpu.items()}
    rl2s = rl2s_cpu.to(device)
    time_idxs = time_idxs_cpu.to(device)
    pass_obs_keys = policy.pass_obs_keys_to_actor or ()
    straight_from_obs = {k: obs[k] for k in pass_obs_keys}

    with torch.inference_mode():
        tstep_emb = policy.tstep_encoder(obs=obs, rl2s=rl2s)
        s_rep, _ = policy.traj_encoder(
            tstep_emb, time_idxs=time_idxs, hidden_state=None
        )
        had_mask_attr = hasattr(policy.actor, "mask_illegal_actions")
        old_mask_value = getattr(policy.actor, "mask_illegal_actions", None)
        if unmasked_logits and had_mask_attr:
            policy.actor.mask_illegal_actions = False
        try:
            raw_logits = policy.actor.actor_network_forward(
                s_rep,
                straight_from_obs=straight_from_obs,
            )
        finally:
            if unmasked_logits and had_mask_attr:
                policy.actor.mask_illegal_actions = old_mask_value
        dist = policy.actor(s_rep, straight_from_obs=straight_from_obs)
        if not hasattr(dist, "probs"):
            raise ValueError(f"{model_name} checkpoint {checkpoint} did not return probs")
        logits = action_vector(raw_logits, "raw_logits")
        probs = action_vector(dist.probs, "probs")

    del experiment, policy, obs, rl2s, time_idxs, raw_logits, dist, s_rep, tstep_emb
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return PolicyOutput(model=model_name, checkpoint=checkpoint, logits=logits, probs=probs)


def make_rows(
    outputs: list[PolicyOutput],
    labels: list[str],
    legal: set[int],
    recorded: int,
) -> list[dict[str, Any]]:
    baseline = outputs[0]
    rows: list[dict[str, Any]] = []
    for output in outputs:
        for action_idx in range(9):
            logit = float(output.logits[action_idx])
            prob = float(output.probs[action_idx])
            base_logit = float(baseline.logits[action_idx])
            base_prob = float(baseline.probs[action_idx])
            if math.isfinite(logit) and math.isfinite(base_logit):
                delta_logit = logit - base_logit
            else:
                delta_logit = math.nan
            rows.append(
                {
                    "model": output.model,
                    "checkpoint": output.checkpoint,
                    "action_idx": action_idx,
                    "action_label": labels[action_idx],
                    "legal": action_idx in legal,
                    "recorded_action": action_idx == recorded,
                    "raw_logit": logit,
                    "prob": prob,
                    "delta_logit_vs_articuno": delta_logit,
                    "delta_prob_vs_articuno": prob - base_prob,
                }
            )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "model",
        "checkpoint",
        "action_idx",
        "action_label",
        "legal",
        "recorded_action",
        "raw_logit",
        "prob",
        "delta_logit_vs_articuno",
        "delta_prob_vs_articuno",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in (
                "raw_logit",
                "prob",
                "delta_logit_vs_articuno",
                "delta_prob_vs_articuno",
            ):
                out[key] = finite_float(float(out[key]))
            writer.writerow(out)


def color_for_prob(prob: float) -> str:
    prob = min(max(prob, 0.0), 1.0)
    base = 245 - int(125 * prob)
    green = 248 - int(70 * prob)
    blue = 255 - int(165 * prob)
    return f"rgb({base},{green},{blue})"


def svg_text(x: float, y: float, text: str, **attrs: Any) -> str:
    default_attrs = {
        "font_family": "Arial, sans-serif",
        "font_size": "13",
        "fill": "#18212f",
    }
    default_attrs.update(attrs)
    attr_text = " ".join(
        f'{key.replace("_", "-")}="{html.escape(str(value), quote=True)}"'
        for key, value in default_attrs.items()
    )
    return f'<text x="{x:.1f}" y="{y:.1f}" {attr_text}>{html.escape(text)}</text>'


def write_svg(
    rows: list[dict[str, Any]],
    outputs: list[PolicyOutput],
    labels: list[str],
    legal: set[int],
    recorded: int,
    situation: Situation,
    replay_root: Path,
    path: Path,
    unmasked_logits: bool,
) -> None:
    model_keys = [(o.model, o.checkpoint) for o in outputs]
    compare_keys = model_keys[1:]
    row_by_key_action = {
        (row["model"], row["checkpoint"], row["action_idx"]): row for row in rows
    }
    finite_deltas = [
        float(row["delta_logit_vs_articuno"])
        for row in rows
        if row["checkpoint"] != 40 and math.isfinite(float(row["delta_logit_vs_articuno"]))
    ]
    max_abs = max([1.0] + [abs(v) for v in finite_deltas])
    scale_min, scale_max = -max_abs, max_abs

    width = 1320
    top = 118
    action_row_h = 78
    bottom = 72
    height = top + 9 * action_row_h + bottom
    label_x = 28
    plot_x = 330
    plot_w = 410
    table_x = 790
    table_cell_w = 112
    table_cell_h = 38
    zero_x = plot_x + (-scale_min / (scale_max - scale_min)) * plot_w
    colors = {
        ("lapras_bc_kl_anchor_actor", 0): "#4c78a8",
        ("lapras_bc_kl_anchor_actor", 2): "#f58518",
        ("lapras_bc_kl_anchor_actor", 5): "#54a24b",
    }

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(
            28,
            34,
            "Lapras policy deltas vs Articuno checkpoint 40",
            font_size=22,
            font_weight=700,
        ),
    ]
    replay_name = short_replay_name(situation.replay, replay_root)
    mode = "unmasked actor-head logits" if unmasked_logits else "actor_network_forward logits"
    parts.append(
        svg_text(
            28,
            59,
            f"Replay: {replay_name} | turn {situation.turn} | active {situation.active} vs {situation.opponent}",
            font_size=13,
            fill="#4b5563",
        )
    )
    parts.append(
        svg_text(
            28,
            80,
            f"Recorded: {situation.recorded_label} | Logit mode: {mode} | Probabilities use policy.actor mask+clip",
            font_size=13,
            fill="#4b5563",
        )
    )

    legend_x = plot_x
    for idx, key in enumerate(compare_keys):
        color = colors.get(key, "#777777")
        x = legend_x + idx * 140
        parts.append(f'<rect x="{x}" y="93" width="16" height="10" fill="{color}"/>')
        parts.append(
            svg_text(
                x + 22,
                103,
                f"ckpt {key[1]}",
                font_size=12,
                fill="#374151",
            )
        )

    parts.append(svg_text(plot_x, top - 10, "delta logit", font_size=12, fill="#374151"))
    for key_idx, key in enumerate(model_keys):
        x = table_x + key_idx * table_cell_w
        label = "Articuno 40" if key_idx == 0 else f"Lapras {key[1]}"
        parts.append(
            svg_text(
                x + table_cell_w / 2,
                top - 10,
                label,
                font_size=12,
                fill="#374151",
                text_anchor="middle",
            )
        )

    for tick in np.linspace(scale_min, scale_max, 5):
        x = plot_x + ((float(tick) - scale_min) / (scale_max - scale_min)) * plot_w
        parts.append(
            f'<line x1="{x:.1f}" y1="{top - 2}" x2="{x:.1f}" y2="{height - bottom + 8}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            svg_text(
                x,
                height - bottom + 28,
                f"{tick:.2f}",
                font_size=11,
                fill="#6b7280",
                text_anchor="middle",
            )
        )
    parts.append(
        f'<line x1="{zero_x:.1f}" y1="{top - 2}" x2="{zero_x:.1f}" y2="{height - bottom + 8}" stroke="#111827" stroke-width="1.2"/>'
    )

    for action_idx, label in enumerate(labels):
        y = top + action_idx * action_row_h
        is_legal = action_idx in legal
        is_recorded = action_idx == recorded
        bg = "#f9fafb" if action_idx % 2 == 0 else "#ffffff"
        if not is_legal:
            bg = "#f3f4f6"
        parts.append(
            f'<rect x="18" y="{y - 18}" width="{width - 36}" height="{action_row_h}" fill="{bg}"/>'
        )
        if is_recorded:
            parts.append(
                f'<rect x="18" y="{y - 18}" width="{width - 36}" height="{action_row_h}" fill="none" stroke="#c58a00" stroke-width="2"/>'
            )
        label_fill = "#111827" if is_legal else "#6b7280"
        suffix = ""
        if not is_legal:
            suffix += " [illegal]"
        if is_recorded:
            suffix += " [recorded]"
        parts.append(
            svg_text(
                label_x,
                y + 19,
                label + suffix,
                font_size=13,
                fill=label_fill,
            )
        )

        bar_h = 13
        for model_idx, key in enumerate(compare_keys):
            row = row_by_key_action[(key[0], key[1], action_idx)]
            delta = float(row["delta_logit_vs_articuno"])
            bar_y = y - 3 + model_idx * 18
            if math.isfinite(delta):
                x_delta = plot_x + ((delta - scale_min) / (scale_max - scale_min)) * plot_w
                x0, x1 = sorted((zero_x, x_delta))
                parts.append(
                    f'<rect x="{x0:.1f}" y="{bar_y:.1f}" width="{max(x1 - x0, 1.0):.1f}" height="{bar_h}" fill="{colors.get(key, "#777777")}"/>'
                )
                label_anchor = "start" if delta >= 0 else "end"
                label_x_pos = x1 + 4 if delta >= 0 else x0 - 4
                parts.append(
                    svg_text(
                        label_x_pos,
                        bar_y + 11,
                        f"{delta:+.2f}",
                        font_size=10,
                        fill="#374151",
                        text_anchor=label_anchor,
                    )
                )
            else:
                parts.append(
                    f'<line x1="{plot_x}" y1="{bar_y + bar_h / 2:.1f}" x2="{plot_x + plot_w}" y2="{bar_y + bar_h / 2:.1f}" stroke="#d1d5db" stroke-width="1" stroke-dasharray="4 3"/>'
                )
                parts.append(
                    svg_text(
                        zero_x + 6,
                        bar_y + 11,
                        "masked",
                        font_size=10,
                        fill="#6b7280",
                    )
                )

        for model_idx, key in enumerate(model_keys):
            row = row_by_key_action[(key[0], key[1], action_idx)]
            prob = float(row["prob"])
            x = table_x + model_idx * table_cell_w
            cell_y = y - 8
            parts.append(
                f'<rect x="{x:.1f}" y="{cell_y:.1f}" width="{table_cell_w - 8}" height="{table_cell_h}" fill="{color_for_prob(prob)}" stroke="#d1d5db"/>'
            )
            parts.append(
                svg_text(
                    x + (table_cell_w - 8) / 2,
                    cell_y + 24,
                    f"{100.0 * prob:.1f}%",
                    font_size=12,
                    fill="#111827",
                    text_anchor="middle",
                )
            )

    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_builder = get_pretrained_model("Articuno")
    dataset = build_dataset(baseline_builder, args.replay_root)
    if baseline_builder.action_space.gym_space.n != 9:
        raise ValueError(
            f"Expected MinimalActionSpace with 9 actions, got {baseline_builder.action_space.gym_space.n}"
        )

    situation = choose_situation(
        dataset=dataset,
        action_space=baseline_builder.action_space,
        replay=args.replay,
        turn=args.turn,
        top_k=args.top_k_situations,
        max_replays_to_scan=args.max_replays_to_scan,
    )
    states, raw_actions = load_states_and_raw_actions(dataset, situation.replay)
    state = states[situation.turn]
    labels = action_labels(state)
    legal = legal_agent_actions(state, baseline_builder.action_space)
    recorded = recorded_agent_action(
        state, baseline_builder.action_space, raw_actions[situation.turn]
    )

    print()
    print(f"Selected replay: {short_replay_name(situation.replay, args.replay_root)}")
    print(f"Turn index: {situation.turn}")
    print(f"Active Pokemon: {state.player_active_pokemon.name}")
    print(f"Opponent active Pokemon: {state.opponent_active_pokemon.name}")
    print(f"Legal actions: {sorted(legal)}")
    print(f"Recorded action: {recorded} ({labels[recorded] if 0 <= recorded < 9 else 'missing'})")

    cpu_device = torch.device("cpu")
    obs_cpu, rl2s_cpu, time_idxs_cpu, tensor_info = build_prefix_tensors(
        dataset=dataset,
        filename=situation.replay,
        turn=situation.turn,
        device=cpu_device,
    )
    print(f"History prefix length: {tensor_info['seq_len']} timesteps")
    print(f"Dataset legal actions at turn: {tensor_info['legal']}")
    print(f"Dataset recorded action missing: {tensor_info['missing']}")

    outputs: list[PolicyOutput] = []
    for model_name, checkpoint in MODEL_SPECS:
        print(f"Forwarding {model_name} checkpoint {checkpoint}...")
        outputs.append(
            forward_policy(
                model_name=model_name,
                checkpoint=checkpoint,
                obs_cpu=obs_cpu,
                rl2s_cpu=rl2s_cpu,
                time_idxs_cpu=time_idxs_cpu,
                unmasked_logits=args.unmasked_logits,
            )
        )

    rows = make_rows(outputs, labels, legal, recorded)
    safe_stem = sanitize_stem(short_replay_name(situation.replay, args.replay_root))
    mode_suffix = "unmasked" if args.unmasked_logits else "masked"
    csv_path = args.output_dir / f"{safe_stem}_turn{situation.turn}_{mode_suffix}_policy_logits.csv"
    svg_path = args.output_dir / f"{safe_stem}_turn{situation.turn}_{mode_suffix}_policy_logits.svg"
    write_csv(rows, csv_path)
    write_svg(
        rows=rows,
        outputs=outputs,
        labels=labels,
        legal=legal,
        recorded=recorded,
        situation=situation,
        replay_root=args.replay_root,
        path=svg_path,
        unmasked_logits=args.unmasked_logits,
    )

    print()
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote SVG: {svg_path}")
    if not args.unmasked_logits:
        print(
            "Note: raw_logit uses actor_network_forward with the actor mask enabled, "
            "so illegal actions may be -inf. Probabilities come from MetamonDiscrete "
            "after clipping/renormalization, so illegal actions can still show the "
            "clip floor. Re-run with --unmasked_logits for pre-mask actor-head logits."
        )
    else:
        print(
            "Note: raw_logit uses pre-mask actor-head logits; probabilities are still "
            "from policy.actor after the normal mask and MetamonDiscrete "
            "clipping/renormalization."
        )


if __name__ == "__main__":
    main()
