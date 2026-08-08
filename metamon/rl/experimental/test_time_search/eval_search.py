"""Eval entry point: frozen Gen1 OU policy with optional oracle root MC search.

Runs the vectorized Showdown env with the frozen ``MiniOnlinePsroV1_4`` policy
as both the evaluated agent and the live opponent, and optionally wraps the
evaluated agent's action selection with test-time search.

Usage::

    uv run python -m metamon.rl.experimental.test_time_search.eval_search \\
        --agent MiniOnlinePsroV1_4 --checkpoint 740 --format gen1ou \\
        --search-mode none --total-battles 50

    uv run python -m metamon.rl.experimental.test_time_search.eval_search \\
        --agent MiniOnlinePsroV1_4 --checkpoint 740 --format gen1ou \\
        --search-mode oracle-root-mc --rollouts-per-action 16 --search-depth 1 \\
        --search-beta 1.0 --total-battles 50
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from metamon.env.vectorized.amago_policy import AmagoLadderPolicyDriver
from metamon.env.vectorized.obs_utils import unstack_obs_dicts
from metamon.env.vectorized.opponent import AmagoBatchedOpponent
from metamon.env.vectorized.vector_env import BattleAgainstMetamon
from metamon.rl.pretrained import get_pretrained_model

from .config import SearchConfig
from .search_driver import SearchEvalRunner


def run_search_eval(
    agent_name: str,
    checkpoint: int,
    battle_format: str,
    team_set_name: str,
    config: SearchConfig,
    total_battles: int,
    num_parallel: int = 4,
    device: str = "cuda",
    opponent_agent: Optional[str] = None,
    opponent_checkpoint: Optional[int] = None,
    seed: Optional[int] = None,
    action_temperature: float = 1.0,
    eval_player_side: int = 0,
) -> Dict:
    """Run vectorized eval and return win-rate + search diagnostics.

    Seeding: when ``seed`` is set, all stochastic sources are fixed so the
    baseline (``search_mode=none``) and search runs are a **paired** comparison:
    same Showdown battle PRNG, same team draws, and same frozen-policy sampling
    stream. At non-searched decisions both runs therefore take identical actions;
    the only difference is the search-selected action at searched decisions.
    """
    import metamon.env
    import random as _random

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    num_parallel = max(
        2, int(num_parallel)
    )  # search eval needs batched (VectorizedShowdownEnv) obs
    if seed is not None:
        _random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        # Deterministic cuDNN (reduces GPU non-determinism). NOTE: the frozen
        # checkpoint uses FlashAttention-2, which is itself non-deterministic on
        # GPU, so bit-identical runs are not possible without disabling flash
        # attn (which would break checkpoint loading). Fixed seeds + sufficient
        # game counts are the practical control; report CIs honestly.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    model = get_pretrained_model(agent_name)
    agent = model.initialize_agent(
        checkpoint=checkpoint, log=False, action_temperature=action_temperature
    )
    policy = agent.policy.to(dev)
    policy.eval()
    action_dim = model.action_space.gym_space.n

    # Opponent: the frozen policy (self-play default) or a named opponent.
    opp_name = opponent_agent or agent_name
    opp_ckpt = opponent_checkpoint if opponent_checkpoint is not None else checkpoint
    opp_model = model if opp_name == agent_name else get_pretrained_model(opp_name)
    opp_agent = opp_model.initialize_agent(
        checkpoint=opp_ckpt, log=False, action_temperature=1.0
    )
    opp_policy = opp_agent.policy.to(dev)
    opp_policy.eval()

    team_set = metamon.env.get_metamon_teams(battle_format, team_set_name)
    env = BattleAgainstMetamon(
        battle_format=battle_format,
        observation_space=model.observation_space,
        action_space=model.action_space,
        reward_function=model.reward_function,
        team_set=team_set,
        opponent_model=opp_model,
        opponent_checkpoint=opp_ckpt,
        opponent_sample=True,
        batched_envs=num_parallel,
        n_workers=1,
        eval_player_side=eval_player_side,
        seed=seed,
        device=str(dev),
    )
    print(
        f"Env: {env.batched_envs} lanes, {battle_format}, eval_side={env.eval_side}, "
        f"search={config.search_mode}"
    )

    eval_driver = AmagoLadderPolicyDriver(
        policy=policy,
        device=dev,
        num_lanes=env.batched_envs,
        action_dim=action_dim,
        sample=True,
    )
    runner = SearchEvalRunner(
        env=env,
        eval_driver=eval_driver,
        opponent=env.opponent,
        eval_policy=policy,
        opponent_policy=opp_policy,
        eval_action_space=model.action_space,
        opponent_action_space=opp_model.action_space,
        eval_reward_function=model.reward_function,
        opponent_reward_function=opp_model.reward_function,
        config=config,
        device=dev,
        action_dim=action_dim,
        battle_format=battle_format,
        # The critic is trained with reward_multiplier=10.0 (set on the
        # MultiTaskAgent via gin). It lives on agent.policy, not the experiment
        # handle ``agent``; fall back to 10.0 only if absent (returns audit).
        reward_multiplier=float(
            getattr(getattr(agent, "policy", agent), "reward_multiplier", 10.0)
        ),
    )

    wins: List[float] = []
    n = env.batched_envs
    obs, info = env.reset()
    steps = 0
    max_steps = max(total_battles * 300 // n + 100, 300)
    battle_idx = 0

    try:
        while len(wins) < total_battles and steps < max_steps:
            steps += 1
            obs_list = unstack_obs_dicts(obs)
            actions = np.zeros(n, dtype=np.int64)
            any_search = False
            for i in range(n):
                lane = env.lanes[i]
                if lane.ended or not lane.needs_agent_decision(env.eval_side):
                    actions[i] = 0
                    continue
                legal = info["legal_actions"][i]
                runner._battle_id = f"b{battle_idx}"
                runner._decision_counter = steps
                if config.search_enabled and (
                    steps % config.search_every_n_decisions == 0
                ):
                    try:
                        action, rec = runner.search_root(i, obs_list[i], legal)
                        actions[i] = action
                        any_search = True
                    except Exception as exc:  # noqa: BLE001
                        if config.search_error_policy == "raise":
                            # Research runs must fail loudly: search_root already
                            # cleaned up its branches before re-raising.
                            raise
                        # base_fallback: log and fall back to base for this decision.
                        print(
                            f"[search] lane {i} failed ({exc!r}); falling back to base"
                        )
                        active = np.zeros(n, dtype=bool)
                        active[i] = True
                        actions[i] = int(eval_driver.act(active, obs_list)[i])
                else:
                    active = np.zeros(n, dtype=bool)
                    active[i] = True
                    actions[i] = int(eval_driver.act(active, obs_list)[i])
            obs, rewards, terminated, truncated, info = env.step(actions)
            for i in range(n):
                eval_driver.observe(i, float(rewards[i]), int(actions[i]))
            done = terminated | truncated
            if done.any():
                for i in np.where(done)[0]:
                    won = info.get("won")
                    w = (
                        float(won[i])
                        if isinstance(won, list) and won[i] is not None
                        else (float(won) if won is not None else 0.0)
                    )
                    wins.append(w)
                    battle_idx += 1
                    eval_driver.reset_lanes(
                        np.array([i == j for j in range(n)], dtype=bool)
                    )
                    env.opponent.reset_lanes(
                        np.array([i == j for j in range(n)], dtype=bool)
                    )
    finally:
        runner.close()
        env.close()

    win_rate = float(np.mean(wins)) if wins else 0.0
    # aggregate diagnostics
    recs = runner.root_records
    agg = {}
    if recs:
        changed = [r for r in recs if r.changed_argmax]
        agg = {
            "n_searched_roots": len(recs),
            "frac_changed_argmax": float(len(changed) / len(recs)),
            "mean_kl_to_base": float(np.mean([r.kl_to_base for r in recs])),
            "mean_latency_ms": float(np.mean([r.latency_ms for r in recs])),
            "mean_search_q_std": float(np.mean([max(r.search_q_std) for r in recs])),
        }
    return {
        "agent": agent_name,
        "checkpoint": checkpoint,
        "battle_format": battle_format,
        "search_mode": config.search_mode,
        "search_ablation": config.search_ablation,
        "rollouts_per_action": config.search_rollouts_per_action,
        "search_depth": config.search_depth,
        "search_beta": config.search_beta,
        "opponent": opp_name,
        "eval_player_side": env.eval_side,
        "seed": seed,
        "total_battles": len(wins),
        "win_rate": win_rate,
        "per_battle_wins": [float(w) for w in wins],
        "diagnostics": agg,
    }


def build_config_from_args(args) -> SearchConfig:
    cfg = SearchConfig(
        search_mode=args.search_mode,
        search_rollouts_per_action=args.rollouts_per_action,
        search_depth=args.search_depth,
        search_beta=args.search_beta,
        search_rollout_temperature=args.rollout_temperature,
        search_root_selection=args.root_selection,
        search_critic_horizon=args.critic_horizon,
        search_lane_batch_size=args.lane_batch_size,
        search_seed=args.search_seed,
        search_every_n_decisions=args.search_every_n,
        search_log_roots=args.search_log_roots,
        search_max_root_actions=args.max_root_actions,
        search_root_prob_threshold=args.root_prob_threshold,
        search_policy_prior_floor=args.policy_prior_floor,
        search_include_intermediate_rewards=args.include_intermediate_rewards,
        search_value_normalization=args.search_value_normalization,
        search_ablation=args.search_ablation,
        search_chance_mode=args.search_chance_mode,
        search_root_opponent_coupling=args.root_opponent_coupling,
        search_leaf_value_mode=args.leaf_value_mode,
        search_root_candidate_mode=args.root_candidate_mode,
        search_cumulative_mass_threshold=args.cumulative_mass_threshold,
        search_min_root_actions=args.min_root_actions,
        search_value_scale_mode=args.value_scale_mode,
        search_global_advantage_scale=args.global_advantage_scale,
        search_magnet_alpha=args.magnet_alpha,
        search_error_policy=args.error_policy,
        search_log_branch_details=args.log_branch_details,
        search_z_gate=args.z_gate,
        search_adaptive_beta=args.adaptive_beta,
        search_adaptive_k=args.adaptive_k,
        search_k_pilot=args.k_pilot,
        search_k_max=args.k_max,
        search_k_batch=args.k_batch,
        search_k_z_stop=args.k_z_stop,
        search_win_head_path=args.win_head_path,
    )
    if getattr(args, "legacy_prototype", False):
        cfg.apply_legacy_prototype_defaults()
    return cfg


def _bool_arg(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).lower() in ("1", "true", "yes", "y")


def add_cli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--agent", default="MiniOnlinePsroV1_4")
    parser.add_argument("--checkpoint", type=int, default=740)
    parser.add_argument("--format", default="gen1ou")
    parser.add_argument("--team_set", default="competitive")
    parser.add_argument("--opponent_agent", default=None)
    parser.add_argument("--opponent_checkpoint", type=int, default=None)
    parser.add_argument("--total_battles", type=int, default=50)
    parser.add_argument("--num_parallel", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--search_mode", default="none", choices=["none", "oracle-root-mc"]
    )
    parser.add_argument("--rollouts_per_action", type=int, default=16)
    parser.add_argument("--search_depth", type=int, default=0)
    parser.add_argument("--search_beta", type=float, default=1.0)
    parser.add_argument("--rollout_temperature", type=float, default=1.0)
    parser.add_argument(
        "--root_selection", default="sample", choices=["sample", "argmax"]
    )
    parser.add_argument("--critic_horizon", type=int, default=None)
    parser.add_argument("--lane_batch_size", type=int, default=64)
    parser.add_argument("--search_seed", type=int, default=0)
    parser.add_argument("--search_every_n", type=int, default=1)
    parser.add_argument("--search_log_roots", default=None)
    parser.add_argument("--max_root_actions", type=int, default=None)
    parser.add_argument(
        "--root_prob_threshold",
        type=float,
        default=0.0,
        help=(
            "relative_threshold candidate mode: prune root actions whose base "
            "prob is below this fraction of the max legal prob (0.0 = keep all; "
            "the legacy prototype used 0.05). The base argmax is always kept."
        ),
    )
    parser.add_argument("--policy_prior_floor", type=float, default=0.0)
    parser.add_argument(
        "--include_intermediate_rewards",
        type=_bool_arg,
        default=True,
        help="accumulate discounted intermediate rewards (True, MC return) vs leaf-only",
    )
    parser.add_argument(
        "--search_value_normalization",
        type=_bool_arg,
        default=False,
        help="legacy per-root z-scoring of advantages (off in primary research mode)",
    )
    parser.add_argument(
        "--search_ablation",
        default="single_anchor_kl",
        choices=[
            "single_anchor_kl",
            "kl_anchor",
            "confidence_gated_kl",
            "magnetic_kl",
            "argmax_q",
            "softmax_q",
            "base_only",
        ],
    )
    # --- research-mode flags (skill §7/§9/§10/§11/§12/§15/§19) ---
    parser.add_argument(
        "--search_chance_mode",
        default="resample_crn",
        choices=["resample_crn", "inherited_trunk_rng"],
        help="resample_crn: branch-only Showdown PRNG seed per rollout index k "
        "(common random numbers across candidate actions). "
        "inherited_trunk_rng: future-chance oracle DIAGNOSTIC only.",
    )
    parser.add_argument(
        "--root_opponent_coupling",
        type=_bool_arg,
        default=True,
        help="couple opponent root action across candidate actions per rollout k",
    )
    parser.add_argument(
        "--leaf_value_mode",
        default="policy_expectation",
        choices=[
            "policy_expectation",
            "sampled_action",
            "root_critic_only",
            "win_head",
        ],
        help="policy_expectation: exact V_pi=sum_a pi(a)Q(h,a). sampled_action: "
        "legacy single-action bootstrap. root_critic_only: no rollout. "
        "win_head (kimi-search M3): terminal-aligned leaf value from a trained "
        "win-probability head (requires --win_head_path).",
    )
    parser.add_argument(
        "--root_candidate_mode",
        default="all_legal",
        choices=["all_legal", "relative_threshold", "cumulative_mass"],
    )
    parser.add_argument("--cumulative_mass_threshold", type=float, default=0.99)
    parser.add_argument("--min_root_actions", type=int, default=2)
    parser.add_argument(
        "--value_scale_mode",
        default="raw",
        choices=["raw", "environment_units", "global_standardized", "legacy_zscore"],
        help="advantage units beta is expressed in (skill §11).",
    )
    parser.add_argument("--global_advantage_scale", type=float, default=None)
    parser.add_argument(
        "--win_head_path",
        default=None,
        help="kimi-search M3: path to a trained WinHead checkpoint. Required "
        "when --leaf_value_mode win_head.",
    )
    parser.add_argument("--magnet_alpha", type=float, default=0.0)
    parser.add_argument(
        "--z_gate",
        type=float,
        default=0.0,
        help="Phase C (skill §37): z-score threshold for the confidence gate. "
        "When > 0, the update is suppressed (returns pi_base) if the best "
        "action's min paired z-score < this threshold. Requires per-branch "
        "rollout data (not root_critic_only). 0 = no gating.",
    )
    parser.add_argument(
        "--adaptive_beta",
        type=_bool_arg,
        default=False,
        help="Phase C: scale beta with confidence (beta_eff = beta * z_gate / "
        "max(min_z, z_gate)) so the update strengthens when the signal is "
        "statistically separated. Only active when z_gate > 0.",
    )
    # --- Phase B: adaptive-K (skill §37) ---
    parser.add_argument(
        "--adaptive_k",
        type=_bool_arg,
        default=False,
        help="Phase B: multi-round adaptive-K with z-score early stopping. "
        "Starts with k_pilot rollouts/action, adds k_batch per round, stops "
        "when the best action's paired z-score >= k_z_stop or k_max is reached. "
        "Recommended for D=0 only.",
    )
    parser.add_argument(
        "--k_pilot", type=int, default=4, help="initial rollouts/action"
    )
    parser.add_argument("--k_max", type=int, default=64, help="maximum rollouts/action")
    parser.add_argument(
        "--k_batch", type=int, default=4, help="additional rollouts/action per round"
    )
    parser.add_argument(
        "--k_z_stop", type=float, default=2.0, help="z-score early-stopping threshold"
    )
    parser.add_argument(
        "--error_policy",
        default="raise",
        choices=["raise", "base_fallback"],
        help="raise: research runs fail loudly. base_fallback: legacy silent fallback.",
    )
    parser.add_argument("--log_branch_details", type=_bool_arg, default=False)
    parser.add_argument(
        "--legacy_prototype",
        action="store_true",
        help="restore the pre-correction prototype defaults (skill §28) so the "
        "historical 100-game result stays reproducible under a labeled mode.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen Gen1 OU eval with optional test-time search"
    )
    add_cli(parser)
    args = parser.parse_args()
    config = build_config_from_args(args)
    results = run_search_eval(
        agent_name=args.agent,
        checkpoint=args.checkpoint,
        battle_format=args.format,
        team_set_name=args.team_set,
        config=config,
        total_battles=args.total_battles,
        num_parallel=args.num_parallel,
        device=args.device,
        opponent_agent=args.opponent_agent,
        opponent_checkpoint=args.opponent_checkpoint,
        seed=args.seed,
        action_temperature=args.temperature,
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
