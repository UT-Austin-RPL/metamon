#!/usr/bin/env python
"""Evaluate the anchored Lapras specialist against TaurosV0.

This runs a coordinated local challenge matchup, records the same local
CSV/JSONL outputs as the H2H launcher, and logs the aggregate result plus
output files to W&B.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


os.environ.setdefault("METAMON_CACHE_DIR", "/home/eddie/metamon_cache")

from metamon.rl.evaluate.common import MatchupSpec, PolicySpec, run_matchup_pair
from metamon.rl.evaluate.results import ResultsTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Lapras anchored specialist vs TaurosV0 H2H evaluation."
    )
    parser.add_argument(
        "--agent",
        default="lapras_bc_kl_anchor_l9_actor",
        help="Pretrained model registry name for the Lapras-side policy.",
    )
    parser.add_argument(
        "--agent_name",
        default="LaprasKLAnchorL9Actor",
        help="Display name used in local result tables and W&B metrics.",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=7,
        help="Lapras checkpoint epoch to evaluate.",
    )
    parser.add_argument("--opponent", default="TaurosV0")
    parser.add_argument("--opponent_name", default="TaurosV0")
    parser.add_argument("--opponent_checkpoint", type=int, default=None)
    parser.add_argument("--battles", type=int, default=100)
    parser.add_argument("--format", default="gen1ou")
    parser.add_argument(
        "--agent_team_set",
        default="competitive",
        help="Team set for the agent-side policy.",
    )
    parser.add_argument(
        "--opponent_team_set",
        default="competitive",
        help="Team set for the opponent policy.",
    )
    parser.add_argument(
        "--team_set",
        default=None,
        help="Legacy shortcut: if set, use this team set for both policies.",
    )
    parser.add_argument("--battle_backend", default="metamon")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--opponent_temperature", type=float, default=1.0)
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=[0],
        help="GPU IDs. If one GPU is supplied, both sides share it.",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/eddie/metamon/evals/lapras_bc_kl_anchor_l9_actor_vs_taurosv0",
    )
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--acceptor_startup_delay", type=float, default=10.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save_trajectories", action="store_true")
    parser.add_argument(
        "--wandb_project",
        default=os.environ.get("METAMON_WANDB_PROJECT"),
        help="W&B project. Defaults to METAMON_WANDB_PROJECT.",
    )
    parser.add_argument(
        "--wandb_entity",
        default=os.environ.get("METAMON_WANDB_ENTITY"),
        help="W&B entity. Defaults to METAMON_WANDB_ENTITY.",
    )
    parser.add_argument(
        "--wandb_run_name",
        default="lapras_bc_kl_anchor_l9_actor-vs-taurosv0-100b",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Run locally without logging the aggregate result to W&B.",
    )
    return parser.parse_args()


def _team_sets(args: argparse.Namespace) -> tuple[str, str]:
    if args.team_set is not None:
        return args.team_set, args.team_set
    return args.agent_team_set, args.opponent_team_set


def _build_matchup(args: argparse.Namespace) -> MatchupSpec:
    agent_team_set, opponent_team_set = _team_sets(args)
    lapras = PolicySpec(
        name=args.agent_name,
        model_name=args.agent,
        checkpoint=args.checkpoint,
        temperature=args.temperature,
        team_set=agent_team_set,
        battle_backend=args.battle_backend,
    )
    tauros = PolicySpec(
        name=args.opponent_name,
        model_name=args.opponent,
        checkpoint=args.opponent_checkpoint,
        temperature=args.opponent_temperature,
        team_set=opponent_team_set,
        battle_backend=args.battle_backend,
    )
    return MatchupSpec(
        policy_a=lapras,
        policy_b=tauros,
        n_battles=args.battles,
        battle_format=args.format,
    )


def _log_to_wandb(args: argparse.Namespace, result, output_dir: Path) -> None:
    if args.no_wandb:
        return

    import wandb

    config = vars(args).copy()
    agent_team_set, opponent_team_set = _team_sets(args)
    config["resolved_agent_team_set"] = agent_team_set
    config["resolved_opponent_team_set"] = opponent_team_set
    init_kwargs = dict(
        name=args.wandb_run_name,
        job_type="eval",
        config=config,
    )
    if args.wandb_project:
        init_kwargs["project"] = args.wandb_project
    if args.wandb_entity:
        init_kwargs["entity"] = args.wandb_entity

    run = wandb.init(**init_kwargs)
    try:
        metrics = {
            "eval/lapras_wins": result.policy_a_wins,
            "eval/taurosv0_wins": result.policy_b_wins,
            "eval/total_battles": result.total_battles,
            "eval/lapras_win_rate": result.policy_a_win_rate,
            "eval/taurosv0_win_rate": 1.0 - result.policy_a_win_rate,
        }
        wandb.log(metrics)

        table = wandb.Table(
            columns=[
                "agent",
                "checkpoint",
                "opponent",
                "opponent_checkpoint",
                "agent_team_set",
                "opponent_team_set",
                "battles",
                "wins",
                "losses",
                "win_rate",
            ]
        )
        table.add_data(
            args.agent,
            args.checkpoint,
            args.opponent,
            args.opponent_checkpoint,
            agent_team_set,
            opponent_team_set,
            result.total_battles,
            result.policy_a_wins,
            result.policy_b_wins,
            result.policy_a_win_rate,
        )
        wandb.log({"eval/summary": table})

        artifact = wandb.Artifact(
            name=f"{args.wandb_run_name}-results",
            type="eval-results",
            metadata={
                "matchup_id": result.matchup_id,
                "lapras_win_rate": result.policy_a_win_rate,
                "total_battles": result.total_battles,
                "agent_team_set": agent_team_set,
                "opponent_team_set": opponent_team_set,
            },
        )
        artifact.add_dir(str(output_dir))
        run.log_artifact(artifact)
    finally:
        wandb.finish()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    matchup = _build_matchup(args)
    gpu_a = args.gpus[0]
    gpu_b = args.gpus[1] if len(args.gpus) > 1 else args.gpus[0]

    tracker = ResultsTracker(str(output_dir))
    if tracker.is_completed(matchup.matchup_id):
        result = tracker._completed[matchup.matchup_id]
    else:
        pair = run_matchup_pair(
            matchup=matchup,
            gpu_a=gpu_a,
            gpu_b=gpu_b,
            output_dir=str(output_dir),
            timeout=args.timeout,
            acceptor_startup_delay=args.acceptor_startup_delay,
            verbose=args.verbose,
            save_trajectories=args.save_trajectories,
        )
        result = tracker.record_from_results_dir(
            matchup_id=matchup.matchup_id,
            policy_a_name=matchup.policy_a.short_label,
            policy_b_name=matchup.policy_b.short_label,
            results_dir=str(Path(pair.matchup_dir) / "results"),
            challenger_username=pair.challenger_username,
        )
        if result is None:
            raise RuntimeError(f"No completed battle results found in {pair.matchup_dir}")

    tracker.print_win_matrix()
    tracker.write_win_matrix_csv()
    _log_to_wandb(args, result, output_dir)

    print(
        f"{result.policy_a_name} vs {result.policy_b_name}: "
        f"{result.policy_a_wins}-{result.policy_b_wins} "
        f"({result.policy_a_win_rate:.1%}) over {result.total_battles} battles"
    )
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
