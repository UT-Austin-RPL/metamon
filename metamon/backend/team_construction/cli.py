from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from tqdm import tqdm

from .core import Team
from .pokemon_pool import (
    build_species_clause_keys,
    get_eligible_pokemon,
    load_pool_artifact,
    pool_pokemon_sets,
    save_pool_artifact,
    team_ids_to_showdown,
    team_string_to_ids,
)
from .restricted_game import (
    build_payoff_matrix,
    build_strategy_pool,
    build_strategy_pool_double_oracle,
    payoff_antisymmetry_error,
    solve_zero_sum_equilibrium,
)
from .artifacts import load_artifact, save_artifact
from .matchup import run_matchup
from .model_fit import fit_baseline_model, fit_interaction_model
from .model_scoring import interaction_matrices, make_scorer
from .coordinate_ascent import (
    coordinate_ascent_best_team,
    coordinate_ascent_multi_start,
    objective_vs_fixed_opponent,
    objective_vs_mixture,
    objective_vs_metagame,
    sample_opponent_teams,
    top_theta_init_team,
)
from .simulation import (
    SimulationMetadata,
    load_examples_jsonl,
    make_active_matchup_sampler,
    make_uniform_matchup_sampler,
    sample_team,
    save_simulation_metadata,
    simulate_battles,
)


def _parse_float_csv(raw: str) -> list[float]:
    out = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not out:
        raise ValueError(f"Expected comma-separated float list, got '{raw}'")
    return out


def _load_pool(pool_path: Path):
    pool_data = load_pool_artifact(pool_path)
    pokemon_sets = pool_pokemon_sets(pool_data)
    species = [ps.species for ps in pokemon_sets]
    format_id = str(pool_data.get("format_id") or "")
    if format_id:
        species_clause_keys = build_species_clause_keys(format_id, pokemon_sets)
    else:
        species_clause_keys = {idx: idx for idx in range(len(pokemon_sets))}
    return pool_data, pokemon_sets, species, species_clause_keys


def _assert_model_pool_compat(model: dict, pokemon_sets) -> None:
    if "num_pokemon" not in model:
        return
    expected = int(model["num_pokemon"])
    actual = len(pokemon_sets)
    if expected != actual:
        raise ValueError(
            f"Model expects {expected} Pokemon features, but pool has {actual}. "
            "Use a model and pool built from the same eligible set."
        )


def _team_to_species(team: Team, species: Sequence[str]) -> list[str]:
    return [species[idx] for idx in team]


def _print_team(label: str, team: Team, species: Sequence[str]) -> None:
    names = _team_to_species(team, species)
    print(f"{label}: {', '.join(names)}")


def _print_restart_summary(runs: list[dict], species: Sequence[str]) -> None:
    if not runs:
        return
    print("Restart summary:")
    for row in runs:
        marker = "*" if row.get("selected") else " "
        start_names = ", ".join(species[idx] for idx in row["start_team"])
        final_names = ", ".join(species[idx] for idx in row["final_team"])
        print(
            f"  {marker} restart={row['restart_index']:>2} swaps={row['accepted_swaps']:>2} "
            f"objective={row['objective']:.6f}"
        )
        print(f"    start=[{start_names}]")
        print(f"    final=[{final_names}]")


def _print_baseline_report(
    model: dict, species: Sequence[str], top_k: int = 30
) -> None:
    theta = np.asarray(model["theta"], dtype=np.float64)
    order = np.argsort(theta)[::-1]

    print("\nBaseline theta ranking (descending):")
    for rank, idx in enumerate(order[: min(top_k, len(order))], start=1):
        print(f"  {rank:>2}. {species[idx]}\t{theta[idx]:+.5f}")


def _top_k_pairs(
    scores: np.ndarray, names: Sequence[str], k: int
) -> list[tuple[str, float]]:
    order = np.argsort(scores)[::-1]
    out: list[tuple[str, float]] = []
    for idx in order:
        if not np.isfinite(scores[idx]):
            continue
        out.append((names[idx], float(scores[idx])))
        if len(out) >= k:
            break
    return out


def _print_interaction_report(
    model: dict, species: Sequence[str], k_detail: int = 3
) -> None:
    theta, synergy_matrix, matchup_matrix = interaction_matrices(model)
    order = np.argsort(theta)[::-1]

    print("\nInteraction theta ranking (descending):")
    for rank, idx in enumerate(order, start=1):
        print(f"  {rank:>2}. {species[idx]}\t{theta[idx]:+.5f}")

    print("\nPer-Pokemon interaction summary:")
    for idx in order:
        partner_scores = synergy_matrix[idx].copy()
        partner_scores[idx] = -np.inf
        partners = _top_k_pairs(partner_scores, species, k=k_detail)

        counter_scores = matchup_matrix[idx].copy()  # i counters j
        counter_scores[idx] = -np.inf
        counters = _top_k_pairs(counter_scores, species, k=k_detail)

        countered_by_scores = matchup_matrix[:, idx].copy()  # j counters i
        countered_by_scores[idx] = -np.inf
        countered_by = _top_k_pairs(countered_by_scores, species, k=k_detail)

        fmt = lambda rows: ", ".join(f"{name} ({score:+.3f})" for name, score in rows)
        print(f"  {species[idx]}:")
        print(f"    top synergy partners: {fmt(partners)}")
        print(f"    top counters: {fmt(counters)}")
        print(f"    countered-by: {fmt(countered_by)}")


def _maybe_attach_pool_metadata(
    model: dict, pool_data: dict, species: Sequence[str]
) -> dict:
    out = dict(model)
    out["format_id"] = pool_data.get("format_id")
    out["species"] = list(species)
    out["pool_size"] = len(species)
    return out


def _merge_jsonl_files(inputs: Sequence[Path], out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with out.open("w", encoding="utf-8") as fout:
        for src in inputs:
            with src.open("r", encoding="utf-8") as fin:
                for line in fin:
                    row = line.strip()
                    if not row:
                        continue
                    fout.write(row + "\n")
                    total += 1
    return total


def _count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _wandb_log_pool_snapshot(
    *,
    wandb_run,
    wandb_mod,
    pokemon_sets,
    top_k: int = 20,
) -> None:
    if wandb_run is None:
        return
    if not pokemon_sets:
        return

    usage_pct = np.asarray(
        [float(ps.usage) * 100.0 for ps in pokemon_sets], dtype=np.float64
    )
    order = np.argsort(usage_pct)[::-1]
    top_idxs = order[: min(top_k, len(order))]
    top_rows = [
        [
            pokemon_sets[idx].species,
            float(usage_pct[idx]),
            str(pokemon_sets[idx].ability or ""),
        ]
        for idx in top_idxs
    ]
    top_table = wandb_mod.Table(
        data=top_rows,
        columns=["species", "usage_pct", "ability"],
    )
    wandb_run.log(
        {
            "pool/usage_pct_histogram": wandb_mod.Histogram(usage_pct),
            "pool/top_species_table": top_table,
            "pool/top_species_usage_bar": wandb_mod.plot.bar(
                top_table,
                "species",
                "usage_pct",
                title=f"Top {len(top_rows)} Pool Usage (%)",
            ),
        }
    )


def _wandb_log_model_snapshot(
    *,
    wandb_run,
    wandb_mod,
    stage_prefix: str,
    model: dict,
    species: Sequence[str],
    top_k: int = 20,
) -> None:
    if wandb_run is None:
        return

    payload: dict[str, object] = {}
    theta = np.asarray(model.get("theta", []), dtype=np.float64)
    if theta.size:
        order = np.argsort(theta)[::-1]
        top_idxs = order[: min(top_k, len(order))]
        top_rows = [[species[idx], float(theta[idx])] for idx in top_idxs]
        top_table = wandb_mod.Table(data=top_rows, columns=["species", "theta"])
        payload[f"{stage_prefix}/theta_histogram"] = wandb_mod.Histogram(theta)
        payload[f"{stage_prefix}/top_theta_table"] = top_table
        payload[f"{stage_prefix}/top_theta_bar"] = wandb_mod.plot.bar(
            top_table,
            "species",
            "theta",
            title=f"{stage_prefix}: Top Theta Weights",
        )

    fit_info = model.get("fit", {}) if isinstance(model.get("fit", {}), dict) else {}
    val_metrics = fit_info.get("val_metrics")
    if isinstance(val_metrics, dict):
        for metric_name, metric_value in val_metrics.items():
            payload[f"{stage_prefix}/val_{metric_name}"] = float(metric_value)

    tuning_rows: list[list[float]] = []
    tuning = fit_info.get("tuning", [])
    if isinstance(tuning, list):
        for row in tuning:
            if not isinstance(row, dict):
                continue
            metrics = row.get("val_metrics")
            if not isinstance(metrics, dict):
                continue
            tuning_rows.append(
                [
                    float(row.get("C", 0.0)),
                    float(metrics.get("log_loss", float("nan"))),
                    float(metrics.get("accuracy", float("nan"))),
                    float(metrics.get("brier", float("nan"))),
                ]
            )
    if tuning_rows:
        tuning_rows.sort(key=lambda r: r[0])
        tuning_table = wandb_mod.Table(
            data=tuning_rows,
            columns=["C", "log_loss", "accuracy", "brier"],
        )
        payload[f"{stage_prefix}/c_tuning_table"] = tuning_table
        payload[f"{stage_prefix}/c_vs_log_loss"] = wandb_mod.plot.line(
            tuning_table,
            "C",
            "log_loss",
            title=f"{stage_prefix}: C vs Validation Log Loss",
        )
        payload[f"{stage_prefix}/c_vs_accuracy"] = wandb_mod.plot.line(
            tuning_table,
            "C",
            "accuracy",
            title=f"{stage_prefix}: C vs Validation Accuracy",
        )

    # Interaction-model-specific summaries (if available).
    try:
        _, synergy_matrix, matchup_matrix = interaction_matrices(model)
        n = min(len(species), int(synergy_matrix.shape[0]))
        if n >= 2:
            synergy_rows: list[list[object]] = []
            for i in range(n):
                for j in range(i + 1, n):
                    synergy_rows.append(
                        [f"{species[i]} + {species[j]}", float(synergy_matrix[i, j])]
                    )
            synergy_rows.sort(key=lambda r: float(r[1]), reverse=True)
            synergy_rows = synergy_rows[: min(top_k, len(synergy_rows))]
            if synergy_rows:
                synergy_table = wandb_mod.Table(
                    data=synergy_rows,
                    columns=["pair", "synergy_score"],
                )
                payload[f"{stage_prefix}/top_synergy_pairs_table"] = synergy_table
                payload[f"{stage_prefix}/top_synergy_pairs_bar"] = wandb_mod.plot.bar(
                    synergy_table,
                    "pair",
                    "synergy_score",
                    title=f"{stage_prefix}: Top Synergy Pairs",
                )

            counter_rows: list[list[object]] = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    counter_rows.append(
                        [f"{species[i]} > {species[j]}", float(matchup_matrix[i, j])]
                    )
            counter_rows.sort(key=lambda r: float(r[1]), reverse=True)
            counter_rows = counter_rows[: min(top_k, len(counter_rows))]
            if counter_rows:
                counter_table = wandb_mod.Table(
                    data=counter_rows,
                    columns=["counter_pair", "counter_score"],
                )
                payload[f"{stage_prefix}/top_counter_pairs_table"] = counter_table
                payload[f"{stage_prefix}/top_counter_pairs_bar"] = wandb_mod.plot.bar(
                    counter_table,
                    "counter_pair",
                    "counter_score",
                    title=f"{stage_prefix}: Top Counter Pairs",
                )
    except Exception:
        pass

    if payload:
        wandb_run.log(payload)


def _wandb_log_metagame_outputs(
    *,
    wandb_run,
    wandb_mod,
    best_team: Team,
    avg_prob: float,
    history: list[dict],
    restart_runs: list[dict],
    species: Sequence[str],
) -> None:
    if wandb_run is None:
        return

    payload: dict[str, object] = {
        "metagame/best_team_avg_win_prob": float(avg_prob),
    }

    best_team_rows = [
        [slot + 1, int(member), species[member]]
        for slot, member in enumerate(best_team)
    ]
    payload["metagame/best_team_table"] = wandb_mod.Table(
        data=best_team_rows,
        columns=["slot", "pokemon_id", "species"],
    )

    history_rows: list[list[object]] = []
    for step, row in enumerate(history):
        history_rows.append(
            [
                int(step),
                str(row.get("event", "")),
                float(row.get("objective", 0.0)),
            ]
        )
    if history_rows:
        history_table = wandb_mod.Table(
            data=history_rows,
            columns=["step", "event", "objective"],
        )
        payload["metagame/objective_trace_table"] = history_table
        payload["metagame/objective_trace_line"] = wandb_mod.plot.line(
            history_table,
            "step",
            "objective",
            title="Coordinate-Ascent Objective Trace",
        )

    restart_rows: list[list[object]] = []
    for row in restart_runs:
        final_team = row.get("final_team", [])
        restart_rows.append(
            [
                int(row.get("restart_index", -1)),
                int(row.get("accepted_swaps", 0)),
                float(row.get("objective", 0.0)),
                bool(row.get("selected", False)),
                ", ".join(species[idx] for idx in final_team),
            ]
        )
    if restart_rows:
        payload["metagame/restarts_table"] = wandb_mod.Table(
            data=restart_rows,
            columns=[
                "restart_index",
                "accepted_swaps",
                "objective",
                "selected",
                "final_team",
            ],
        )

    wandb_run.log(payload)


def _wandb_log_benchmark_outputs(
    *,
    wandb_run,
    wandb_mod,
    benchmark: dict | None,
) -> None:
    if wandb_run is None or not benchmark or not benchmark.get("ran", False):
        return

    payload: dict[str, object] = {}
    pipeline_wr = float(benchmark.get("pipeline_win_rate", 0.0))
    random_mean = float(benchmark.get("random_mean_win_rate", 0.0))
    random_std = float(benchmark.get("random_std_win_rate", 0.0))
    margin = float(benchmark.get("margin_vs_random_mean", 0.0))
    payload["benchmark/pipeline_win_rate"] = pipeline_wr
    payload["benchmark/random_mean_win_rate"] = random_mean
    payload["benchmark/random_std_win_rate"] = random_std
    payload["benchmark/margin_vs_random_mean"] = margin

    compare_table = wandb_mod.Table(
        data=[
            ["pipeline_team", pipeline_wr],
            ["random_mean", random_mean],
        ],
        columns=["group", "win_rate"],
    )
    payload["benchmark/pipeline_vs_random_bar"] = wandb_mod.plot.bar(
        compare_table,
        "group",
        "win_rate",
        title="Pipeline Team vs Random-Team Baseline",
    )

    random_samples = benchmark.get("random_samples", [])
    if isinstance(random_samples, list) and random_samples:
        rows: list[list[object]] = []
        random_wrs: list[float] = []
        for idx, row in enumerate(random_samples, start=1):
            wr = float(row.get("win_rate", 0.0))
            random_wrs.append(wr)
            rows.append(
                [
                    int(idx),
                    wr,
                    ", ".join(str(name) for name in row.get("team_species", [])),
                ]
            )
        random_table = wandb_mod.Table(
            data=rows,
            columns=["sample_idx", "win_rate", "team_species"],
        )
        payload["benchmark/random_samples_table"] = random_table
        payload["benchmark/random_win_rate_histogram"] = wandb_mod.Histogram(
            np.asarray(random_wrs, dtype=np.float64)
        )
        payload["benchmark/random_win_rate_scatter"] = wandb_mod.plot.scatter(
            random_table,
            "sample_idx",
            "win_rate",
            title="Random-Team Win Rate Samples",
        )

    wandb_run.log(payload)


def _wandb_log_equilibrium_outputs(
    *,
    wandb_run,
    wandb_mod,
    eq_payload: dict,
    support_tol: float,
) -> None:
    if wandb_run is None:
        return

    payload: dict[str, object] = {}

    equilibrium = eq_payload.get("equilibrium", {})
    row_mixture = np.asarray(equilibrium.get("row_mixture", []), dtype=np.float64)
    strategy_species = eq_payload.get("strategy_species", [])

    if row_mixture.size:
        payload["equilibrium/game_value"] = float(equilibrium.get("game_value", 0.0))
        payload["equilibrium/row_mixture_histogram"] = wandb_mod.Histogram(row_mixture)

        mix_rows: list[list[object]] = []
        for idx, prob in enumerate(row_mixture.tolist()):
            label = ""
            if idx < len(strategy_species):
                label = ", ".join(str(name) for name in strategy_species[idx])
            mix_rows.append([int(idx), float(prob), label])

        mix_table = wandb_mod.Table(
            data=mix_rows,
            columns=["strategy_idx", "probability", "team_species"],
        )
        payload["equilibrium/row_mixture_table"] = mix_table
        payload["equilibrium/row_mixture_bar"] = wandb_mod.plot.bar(
            mix_table,
            "strategy_idx",
            "probability",
            title="Equilibrium Row Mixture",
        )

        support_rows = [row for row in mix_rows if float(row[1]) > float(support_tol)]
        if support_rows:
            payload["equilibrium/support_table"] = wandb_mod.Table(
                data=support_rows,
                columns=["strategy_idx", "probability", "team_species"],
            )

    payoff = np.asarray(eq_payload.get("payoff_matrix", []), dtype=np.float64)
    if payoff.size:
        payload["equilibrium/payoff_value_histogram"] = wandb_mod.Histogram(
            payoff.reshape(-1)
        )
        payload["equilibrium/antisymmetry_error"] = float(
            eq_payload.get("antisymmetry_error", 0.0)
        )

    oracle_rows: list[list[float]] = []
    pool_expansion = eq_payload.get("pool_expansion", {})
    if isinstance(pool_expansion, dict):
        for row in pool_expansion.get("double_oracle_iterations", []):
            if not isinstance(row, dict):
                continue
            oracle_rows.append(
                [
                    float(row.get("iteration", 0)),
                    float(row.get("game_value", 0.0)),
                    float(row.get("best_response_value", 0.0)),
                    float(row.get("exploitability", 0.0)),
                ]
            )
    if oracle_rows:
        oracle_table = wandb_mod.Table(
            data=oracle_rows,
            columns=[
                "iteration",
                "game_value",
                "best_response_value",
                "exploitability",
            ],
        )
        payload["equilibrium/double_oracle_table"] = oracle_table
        payload["equilibrium/exploitability_line"] = wandb_mod.plot.line(
            oracle_table,
            "iteration",
            "exploitability",
            title="Double-Oracle Exploitability",
        )
        payload["equilibrium/game_value_line"] = wandb_mod.plot.line(
            oracle_table,
            "iteration",
            "game_value",
            title="Double-Oracle Game Value",
        )

    if payload:
        wandb_run.log(payload)


def _wandb_log_run_artifact(
    *,
    wandb_run,
    wandb_mod,
    artifact_name: str,
    file_paths: Sequence[Path | None],
) -> None:
    if wandb_run is None:
        return
    artifact = wandb_mod.Artifact(name=artifact_name, type="team_construction_run")
    added = 0
    for path in file_paths:
        if path is None:
            continue
        p = Path(path)
        if not p.exists():
            continue
        artifact.add_file(str(p), name=p.name)
        added += 1
    if added > 0:
        wandb_run.log_artifact(artifact)


def _init_custom_teamset_dir(
    cache_dir: Path, set_name: str, battle_format: str
) -> Path:
    target_dir = cache_dir / "teams" / set_name / battle_format
    target_dir.mkdir(parents=True, exist_ok=True)
    for old in target_dir.glob(f"*.{battle_format}_team"):
        old.unlink()
    return target_dir


def _write_custom_team(team_dir: Path, battle_format: str, team_text: str) -> None:
    team_file = team_dir / f"team_0001.{battle_format}_team"
    team_file.write_text(team_text.strip() + "\n", encoding="utf-8")


def _run_matchup_with_retry(
    *,
    battle_format: str,
    num_battles: int,
    model_name: str,
    team_set_a: str,
    team_set_b: str,
    gpu_a: int,
    gpu_b: int,
    work_dir: Path,
    checkpoint: int | None,
    max_retries: int,
    retry_sleep_sec: float,
    print_match_stats: bool,
) -> dict:
    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            return run_matchup(
                battle_format=battle_format,
                num_battles=num_battles,
                model_name=model_name,
                team_set_a=team_set_a,
                team_set_b=team_set_b,
                gpu_a=gpu_a,
                gpu_b=gpu_b,
                work_dir=work_dir,
                checkpoint=checkpoint,
                print_match_stats=print_match_stats,
            )
        except RuntimeError as exc:
            last_error = str(exc)
            if attempt >= max_retries:
                break
            time.sleep(retry_sleep_sec)
    raise RuntimeError(
        f"run_matchup failed for gpu pair ({gpu_a},{gpu_b}) after retries: {last_error}"
    )


def cmd_build_pool(args: argparse.Namespace) -> None:
    pokemon_sets = get_eligible_pokemon(
        format_id=args.format,
        usage_month=args.usage_month,
        usage_threshold=args.usage_threshold,
        rank=args.rank,
        replication_movesets_json=args.replication_movesets_json,
        manual_sets_json=args.manual_sets_json,
        strict_max_evs=args.strict_max_evs,
    )

    save_pool_artifact(
        args.out,
        format_id=args.format,
        usage_month=args.usage_month,
        usage_threshold=args.usage_threshold,
        pokemon_sets=pokemon_sets,
        metadata={
            "rank": args.rank,
            "replication_movesets_json": (
                str(args.replication_movesets_json)
                if args.replication_movesets_json
                else None
            ),
            "manual_sets_json": (
                str(args.manual_sets_json) if args.manual_sets_json else None
            ),
            "strict_max_evs": bool(args.strict_max_evs),
        },
    )

    print(f"Saved pool with {len(pokemon_sets)} Pokemon to {args.out}")
    print("Top by usage:")
    for row in pokemon_sets[: min(15, len(pokemon_sets))]:
        print(f"  {row.species}: {row.usage * 100:.3f}%")


def cmd_simulate(args: argparse.Namespace) -> None:
    pool_data, pokemon_sets, species, species_clause_keys = _load_pool(args.pool)
    format_id = args.format or str(pool_data.get("format_id") or "")
    if not format_id:
        raise ValueError(
            "Format missing: provide --format or include it in the pool artifact"
        )

    pool_ids = list(range(len(pokemon_sets)))
    rng = random.Random(args.seed)
    sampling_metadata: dict[str, object] = {
        "strategy": args.sampling_strategy,
        "team_size": int(args.team_size),
        "replace": bool(args.replace),
    }

    if args.sampling_strategy == "uniform":
        sampler = make_uniform_matchup_sampler(
            pool_ids,
            team_size=args.team_size,
            replace=args.replace,
            rng=rng,
            species_clause_keys=species_clause_keys,
        )
    elif args.sampling_strategy == "active":
        if args.active_model is None:
            raise ValueError(
                "--active-model is required when --sampling-strategy active"
            )
        active_model = load_artifact(args.active_model)
        _assert_model_pool_compat(active_model, pokemon_sets)
        pair_eval = make_scorer(active_model)
        sampler = make_active_matchup_sampler(
            pool_ids,
            pair_evaluator=pair_eval,
            team_size=args.team_size,
            replace=args.replace,
            rng=rng,
            candidate_pool_size=args.active_candidate_pool_size,
            uniform_mix=args.active_uniform_mix,
            min_uncertainty=args.active_min_uncertainty,
            species_clause_keys=species_clause_keys,
        )
        sampling_metadata.update(
            {
                "active_model": str(args.active_model),
                "active_candidate_pool_size": int(args.active_candidate_pool_size),
                "active_uniform_mix": float(args.active_uniform_mix),
                "active_min_uncertainty": float(args.active_min_uncertainty),
            }
        )
    else:
        raise ValueError(f"Unknown sampling strategy: {args.sampling_strategy}")

    heuristic_aliases = {
        "simpleheuristicsplayer",
        "simple_heuristics_player",
        "simpleheuristics",
    }
    normalized_agent = str(args.agent).strip().lower()
    resolved_backend = str(args.backend)
    metamon_model_name: str | None = None
    if resolved_backend == "poke_env" and normalized_agent not in heuristic_aliases:
        resolved_backend = "metamon"
        metamon_model_name = str(args.agent)
        print(
            "[simulate] Non-heuristic --agent detected under backend='poke_env'; "
            f"routing simulation through backend='metamon' with model '{metamon_model_name}'."
        )
    elif resolved_backend == "metamon":
        metamon_model_name = str(args.agent)
    if resolved_backend == "metamon":
        sampling_metadata.update(
            {
                "metamon_model_name": metamon_model_name,
                "metamon_checkpoint": args.checkpoint,
                "metamon_gpu_a": int(args.gpu_a),
                "metamon_gpu_b": int(args.gpu_b),
                "metamon_work_dir": str(args.work_dir),
            }
        )

    team_to_showdown = None
    if resolved_backend in {"poke_env", "metamon"}:
        team_to_showdown = lambda team: team_ids_to_showdown(team, pokemon_sets)

    examples = simulate_battles(
        n=args.n,
        sampler=sampler,
        agent_class=args.agent,
        format_id=format_id,
        concurrency=args.concurrency,
        seed=args.seed,
        backend=resolved_backend,
        team_to_showdown=team_to_showdown,
        timeout_sec=args.timeout_sec,
        max_retries=args.max_retries,
        retry_sleep_sec=args.retry_sleep_sec,
        metamon_model_name=metamon_model_name,
        metamon_checkpoint=args.checkpoint,
        metamon_gpu_a=args.gpu_a,
        metamon_gpu_b=args.gpu_b,
        metamon_work_dir=args.work_dir,
        metamon_print_match_stats=args.print_match_stats,
        incremental_out=args.out,
        incremental_flush_every=args.flush_every,
        progress_desc=(
            f"Simulating {args.sampling_strategy} battles"
            if args.sampling_strategy
            else "Simulating battles"
        ),
    )

    meta_path = args.metadata_out or args.out.with_suffix(
        args.out.suffix + ".meta.json"
    )
    save_simulation_metadata(
        meta_path,
        SimulationMetadata(
            format_id=format_id,
            agent_name=args.agent,
            n_battles=args.n,
            seed=args.seed,
            backend=resolved_backend,
            concurrency=args.concurrency,
            sampling_strategy=args.sampling_strategy,
            sampling_metadata=sampling_metadata,
        ),
    )

    extra = {
        "pokemon_index": {
            species_name: idx for idx, species_name in enumerate(species)
        },
        "team_size": args.team_size,
        "replace": bool(args.replace),
        "backend": resolved_backend,
        "agent": args.agent,
        "sampling_strategy": args.sampling_strategy,
        "sampling_metadata": sampling_metadata,
    }
    with meta_path.open("r+", encoding="utf-8") as f:
        payload = json.load(f)
        payload.update(extra)
        f.seek(0)
        json.dump(payload, f, indent=2, sort_keys=True)
        f.truncate()

    print(f"Simulated {len(examples)} battles -> {args.out}")
    print(f"Saved simulation metadata -> {meta_path}")


def cmd_fit_baseline(args: argparse.Namespace) -> None:
    pool_data, pokemon_sets, species, _ = _load_pool(args.pool)
    examples = load_examples_jsonl(args.input)

    model = fit_baseline_model(
        examples,
        num_pokemon=len(pokemon_sets),
        val_fraction=args.val_fraction,
        split_seed=args.seed,
        max_iter=args.max_iter,
    )
    model = _maybe_attach_pool_metadata(model, pool_data, species)

    save_artifact(args.out, model)
    print(f"Saved baseline model -> {args.out}")

    val_metrics = model.get("fit", {}).get("val_metrics")
    if val_metrics is not None:
        print(
            "Validation metrics: "
            + ", ".join(f"{k}={v:.6f}" for k, v in val_metrics.items())
        )
    _print_baseline_report(model, species)


def cmd_fit_interaction(args: argparse.Namespace) -> None:
    pool_data, pokemon_sets, species, _ = _load_pool(args.pool)
    examples = load_examples_jsonl(args.input)

    if args.tune_C:
        c_values = _parse_float_csv(args.c_values)
    else:
        c_values = [float(args.C)]

    model = fit_interaction_model(
        examples,
        num_pokemon=len(pokemon_sets),
        c_values=c_values,
        val_fraction=args.val_fraction,
        split_seed=args.seed,
        max_iter=args.max_iter,
    )
    model = _maybe_attach_pool_metadata(model, pool_data, species)

    save_artifact(args.out, model)
    print(f"Saved interaction model -> {args.out}")

    tuning = model.get("fit", {}).get("tuning", [])
    if tuning:
        print("C tuning results:")
        for row in tuning:
            vm = row.get("val_metrics")
            if vm is None:
                print(f"  C={row['C']:.6g}: no validation split")
            else:
                print(
                    f"  C={row['C']:.6g}: log_loss={vm['log_loss']:.6f}, "
                    f"accuracy={vm['accuracy']:.6f}, brier={vm['brier']:.6f}"
                )

    print(f"Selected best C: {model['fit']['best_C']}")
    _print_interaction_report(model, species, k_detail=args.detail_k)


def _resolve_init_team(
    *,
    init_mode: str,
    init_team_text: str | None,
    team_size: int,
    theta: np.ndarray,
    pool_size: int,
    pokemon_sets,
    seed: int,
    species_clause_keys: dict[int, object],
) -> Team:
    if init_mode == "top_theta":
        return top_theta_init_team(
            theta,
            team_size=team_size,
            species_clause_keys=species_clause_keys,
        )
    if init_mode == "random":
        rng = random.Random(seed)
        return sample_team(
            list(range(pool_size)),
            team_size=team_size,
            replace=False,
            rng=rng,
            species_clause_keys=species_clause_keys,
        )
    if init_mode == "explicit":
        if not init_team_text:
            raise ValueError("--init-team is required when --init explicit")
        return team_string_to_ids(init_team_text, pokemon_sets, team_size=team_size)
    raise ValueError(f"Unknown init mode: {init_mode}")


def cmd_best_response(args: argparse.Namespace) -> None:
    _, pokemon_sets, species, species_clause_keys = _load_pool(args.pool)
    model = load_artifact(args.model)
    _assert_model_pool_compat(model, pokemon_sets)
    pair_eval = make_scorer(model)

    if args.vs_team_file is not None:
        opponent_text = args.vs_team_file.read_text(encoding="utf-8")
    elif args.vs is not None:
        opponent_text = args.vs
    else:
        raise ValueError("Provide --vs or --vs-team-file")

    opponent = team_string_to_ids(opponent_text, pokemon_sets, team_size=args.team_size)

    theta = np.asarray(model["theta"], dtype=np.float64)
    init_team = _resolve_init_team(
        init_mode=args.init,
        init_team_text=args.init_team,
        team_size=args.team_size,
        theta=theta,
        pool_size=len(pokemon_sets),
        pokemon_sets=pokemon_sets,
        seed=args.seed,
        species_clause_keys=species_clause_keys,
    )

    objective = objective_vs_fixed_opponent(pair_eval, opponent)
    best_team, history, restart_runs = coordinate_ascent_multi_start(
        objective,
        primary_init=init_team,
        pool_ids=list(range(len(pokemon_sets))),
        team_size=args.team_size,
        random_restarts=args.random_restarts,
        seed=args.seed + 101,
        species_clause_keys=species_clause_keys,
    )

    win_prob = float(pair_eval(best_team, opponent))

    _print_team("Opponent team", opponent, species)
    _print_team("Best response", best_team, species)
    print(f"Predicted win probability: {win_prob:.6f}")
    _print_restart_summary(restart_runs, species)

    print("Accepted swaps (objective monotonicity trace):")
    for row in history:
        if row["event"] == "init":
            print(f"  init objective={row['objective']:.6f}")
        else:
            print(
                f"  slot {row['slot']} out={species[row['out']]} in={species[row['in']]} "
                f"objective={row['objective']:.6f}"
            )


def _optimize_metagame_team(
    *,
    model: dict,
    pokemon_sets,
    team_size: int,
    n_opponents: int,
    seed: int,
    init_mode: str,
    init_team_text: str | None,
    random_restarts: int,
    species_clause_keys: dict[int, object],
) -> tuple[Team, float, list[dict], list[Team], list[dict]]:
    pair_eval = make_scorer(model)
    theta = np.asarray(model["theta"], dtype=np.float64)

    init_team = _resolve_init_team(
        init_mode=init_mode,
        init_team_text=init_team_text,
        team_size=team_size,
        theta=theta,
        pool_size=len(pokemon_sets),
        pokemon_sets=pokemon_sets,
        seed=seed,
        species_clause_keys=species_clause_keys,
    )

    opponents = sample_opponent_teams(
        list(range(len(pokemon_sets))),
        n=n_opponents,
        team_size=team_size,
        seed=seed + 17,
        replace=False,
        species_clause_keys=species_clause_keys,
    )
    objective = objective_vs_metagame(pair_eval, opponents)
    best_team, history, restart_runs = coordinate_ascent_multi_start(
        objective,
        primary_init=init_team,
        pool_ids=list(range(len(pokemon_sets))),
        team_size=team_size,
        random_restarts=random_restarts,
        seed=seed + 271,
        species_clause_keys=species_clause_keys,
    )
    avg_prob = float(objective(best_team))
    return best_team, avg_prob, history, opponents, restart_runs


def cmd_optimize_metagame(args: argparse.Namespace) -> None:
    _, pokemon_sets, species, species_clause_keys = _load_pool(args.pool)
    model = load_artifact(args.model)
    _assert_model_pool_compat(model, pokemon_sets)

    best_team, avg_prob, history, _, restart_runs = _optimize_metagame_team(
        model=model,
        pokemon_sets=pokemon_sets,
        team_size=args.team_size,
        n_opponents=args.N,
        seed=args.seed,
        init_mode=args.init,
        init_team_text=args.init_team,
        random_restarts=args.random_restarts,
        species_clause_keys=species_clause_keys,
    )

    _print_team("Metagame-optimized team", best_team, species)
    print(f"Predicted average win probability vs sampled metagame: {avg_prob:.6f}")
    _print_restart_summary(restart_runs, species)

    print("Accepted swaps (objective monotonicity trace):")
    for row in history:
        if row["event"] == "init":
            print(f"  init objective={row['objective']:.6f}")
        else:
            print(
                f"  slot {row['slot']} out={species[row['out']]} in={species[row['in']]} "
                f"objective={row['objective']:.6f}"
            )


def cmd_equilibrium(args: argparse.Namespace) -> None:
    _, pokemon_sets, species, species_clause_keys = _load_pool(args.pool)
    model = load_artifact(args.model)
    _assert_model_pool_compat(model, pokemon_sets)
    pair_eval = make_scorer(model)

    if args.seed_team_from == "metagame":
        seed_team, avg_prob, _, _, seed_restart_runs = _optimize_metagame_team(
            model=model,
            pokemon_sets=pokemon_sets,
            team_size=args.team_size,
            n_opponents=args.metagame_N,
            seed=args.seed,
            init_mode="top_theta",
            init_team_text=None,
            random_restarts=args.metagame_random_restarts,
            species_clause_keys=species_clause_keys,
        )
        _print_team("Seed team (metagame optimized)", seed_team, species)
        print(f"Seed-team average win probability: {avg_prob:.6f}")
        _print_restart_summary(seed_restart_runs, species)
    elif args.seed_team_from == "top_theta":
        seed_team = top_theta_init_team(
            np.asarray(model["theta"]),
            team_size=args.team_size,
            species_clause_keys=species_clause_keys,
        )
        _print_team("Seed team (top theta)", seed_team, species)
    else:
        if not args.seed_team:
            raise ValueError("--seed-team is required when --seed-team-from explicit")
        seed_team = team_string_to_ids(
            args.seed_team, pokemon_sets, team_size=args.team_size
        )
        _print_team("Seed team (explicit)", seed_team, species)

    theta = np.asarray(model["theta"], dtype=np.float64)
    init_for_br = top_theta_init_team(
        theta,
        team_size=args.team_size,
        species_clause_keys=species_clause_keys,
    )
    pool_ids = list(range(len(pokemon_sets)))
    br_restart_traces: list[list[dict]] = []

    if args.pool_expansion == "last_response":
        br_counter = 0

        def _best_response_to(team_last: Team) -> Team:
            nonlocal br_counter
            objective = objective_vs_fixed_opponent(pair_eval, team_last)
            best, _, restart_runs = coordinate_ascent_multi_start(
                objective,
                primary_init=init_for_br,
                pool_ids=pool_ids,
                team_size=args.team_size,
                random_restarts=args.br_random_restarts,
                seed=args.seed + 10000 + br_counter,
                species_clause_keys=species_clause_keys,
            )
            br_counter += 1
            br_restart_traces.append(restart_runs)
            return best

        strategies = build_strategy_pool(
            seed_team,
            best_response=_best_response_to,
            max_size=args.max_size,
            stop_on_cycle=True,
        )
        oracle_iterations: list[dict] = []
    else:
        br_counter = 0

        def _best_response_to_mixture(
            restricted_strategies: Sequence[Team], row_mixture: np.ndarray
        ) -> Team:
            nonlocal br_counter
            objective = objective_vs_mixture(
                pair_eval, restricted_strategies, row_mixture
            )
            best, _, restart_runs = coordinate_ascent_multi_start(
                objective,
                primary_init=init_for_br,
                pool_ids=pool_ids,
                team_size=args.team_size,
                random_restarts=args.br_random_restarts,
                seed=args.seed + 20000 + br_counter,
                species_clause_keys=species_clause_keys,
            )
            br_counter += 1
            br_restart_traces.append(restart_runs)
            return best

        strategies, oracle_iterations = build_strategy_pool_double_oracle(
            seed_team,
            pair_evaluator=pair_eval,
            best_response_to_mixture=_best_response_to_mixture,
            max_size=args.max_size,
            stop_on_cycle=True,
            exploitability_tol=args.exploitability_tol,
        )

    payoff = build_payoff_matrix(strategies, pair_eval)
    antisym_err = payoff_antisymmetry_error(payoff)
    eq = solve_zero_sum_equilibrium(payoff)

    row_mix = np.asarray(eq["row_mixture"], dtype=np.float64)
    print("\nRestricted strategy pool:")
    for idx, team in enumerate(strategies):
        _print_team(f"  t{idx}", team, species)

    if oracle_iterations:
        print("\nDouble-oracle iterations:")
        for row in oracle_iterations:
            print(
                f"  iter={row['iteration']:>2} pool={row['pool_size']:>2} "
                f"game_value={row['game_value']:+.6f} "
                f"br_value={row['best_response_value']:+.6f} "
                f"exploitability={row['exploitability']:+.6e}"
            )

    print(f"\nPayoff antisymmetry max error |A + A^T|_max = {antisym_err:.6e}")

    print("\nEquilibrium mixture (row player):")
    support = []
    for idx, prob in enumerate(row_mix):
        if prob > args.support_tol:
            support.append((idx, float(prob)))
            _print_team(f"  w={prob:.6f} team t{idx}", strategies[idx], species)

    if not support:
        print("  (No probabilities above support tolerance.)")

    print(f"Game value: {float(eq['game_value']):+.6f}")

    if args.out is not None:
        payload = {
            "strategies": [list(team) for team in strategies],
            "strategy_species": [[species[i] for i in team] for team in strategies],
            "payoff_matrix": payoff.tolist(),
            "antisymmetry_error": float(antisym_err),
            "equilibrium": {
                "chosen_index": int(eq["chosen_index"]),
                "row_mixture": row_mix.tolist(),
                "col_mixture": np.asarray(eq["col_mixture"], dtype=np.float64).tolist(),
                "game_value": float(eq["game_value"]),
            },
            "pool_expansion": {
                "mode": args.pool_expansion,
                "br_random_restarts": int(args.br_random_restarts),
                "exploitability_tol": float(args.exploitability_tol),
                "double_oracle_iterations": oracle_iterations,
                "br_restart_traces": br_restart_traces,
            },
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"Saved equilibrium artifact -> {args.out}")


def cmd_checks(args: argparse.Namespace) -> None:
    _, pokemon_sets, _, species_clause_keys = _load_pool(args.pool)
    model = load_artifact(args.model)
    _assert_model_pool_compat(model, pokemon_sets)
    pair_eval = make_scorer(model)

    rng = random.Random(args.seed)
    pool_ids = list(range(len(pokemon_sets)))

    sym_errors: list[float] = []
    for _ in range(args.num_pairs):
        a = sample_team(
            pool_ids,
            team_size=args.team_size,
            replace=False,
            rng=rng,
            species_clause_keys=species_clause_keys,
        )
        b = sample_team(
            pool_ids,
            team_size=args.team_size,
            replace=False,
            rng=rng,
            species_clause_keys=species_clause_keys,
        )
        p_ab = float(pair_eval(a, b))
        p_ba = float(pair_eval(b, a))
        sym_errors.append(abs((p_ab + p_ba) - 1.0))

    max_sym = max(sym_errors) if sym_errors else 0.0
    mean_sym = float(np.mean(sym_errors)) if sym_errors else 0.0

    opp = sample_team(
        pool_ids,
        team_size=args.team_size,
        replace=False,
        rng=rng,
        species_clause_keys=species_clause_keys,
    )
    init = top_theta_init_team(
        np.asarray(model["theta"]),
        team_size=args.team_size,
        species_clause_keys=species_clause_keys,
    )
    objective = objective_vs_fixed_opponent(pair_eval, opp)
    _, history = coordinate_ascent_best_team(
        objective,
        init_team=init,
        pool_ids=pool_ids,
        team_size=args.team_size,
        species_clause_keys=species_clause_keys,
    )

    monotonic = True
    prev = None
    for row in history:
        value = float(row["objective"])
        if prev is not None and value + 1e-12 < prev:
            monotonic = False
            break
        prev = value

    strategies = [
        sample_team(
            pool_ids,
            team_size=args.team_size,
            replace=False,
            rng=rng,
            species_clause_keys=species_clause_keys,
        )
        for _ in range(args.payoff_pool_size)
    ]
    payoff = build_payoff_matrix(strategies, pair_eval)
    antisym = payoff_antisymmetry_error(payoff)

    sim_rng_1 = random.Random(args.seed + 1000)
    sim_rng_2 = random.Random(args.seed + 1000)
    sampler_1 = make_uniform_matchup_sampler(
        pool_ids,
        team_size=args.team_size,
        replace=False,
        rng=sim_rng_1,
        species_clause_keys=species_clause_keys,
    )
    sampler_2 = make_uniform_matchup_sampler(
        pool_ids,
        team_size=args.team_size,
        replace=False,
        rng=sim_rng_2,
        species_clause_keys=species_clause_keys,
    )
    examples_1 = simulate_battles(
        n=args.repro_n,
        sampler=sampler_1,
        agent_class="SimpleHeuristicsPlayer",
        format_id=str(model.get("format_id") or "gen1ou"),
        concurrency=1,
        seed=args.seed,
        backend="synthetic",
    )
    examples_2 = simulate_battles(
        n=args.repro_n,
        sampler=sampler_2,
        agent_class="SimpleHeuristicsPlayer",
        format_id=str(model.get("format_id") or "gen1ou"),
        concurrency=1,
        seed=args.seed,
        backend="synthetic",
    )

    reproducible = examples_1 == examples_2

    print("Symmetry check:")
    print(f"  mean |p(A,B)+p(B,A)-1| = {mean_sym:.6e}")
    print(f"  max  |p(A,B)+p(B,A)-1| = {max_sym:.6e}")

    print("Search monotonicity check:")
    print(f"  accepted swaps: {max(0, len(history) - 1)}")
    print(f"  monotonic objective: {monotonic}")

    print("Payoff antisymmetry check:")
    print(f"  max |A + A^T| = {antisym:.6e}")

    print("Reproducibility check (synthetic simulator):")
    print(f"  deterministic examples match: {reproducible}")


def cmd_run_all(args: argparse.Namespace) -> None:
    if args.vs is not None and args.vs_team_file is not None:
        raise ValueError("Provide at most one of --vs and --vs-team-file")

    if (args.num_batches is None) != (args.batch_size is None):
        raise ValueError("Provide both --num-batches and --batch-size, or neither.")
    if args.num_batches is not None and args.num_batches <= 0:
        raise ValueError(f"--num-batches must be > 0, got {args.num_batches}")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError(f"--batch-size must be > 0, got {args.batch_size}")

    legacy_n = None
    if args.num_batches is not None and args.batch_size is not None:
        legacy_n = int(args.num_batches) * int(args.batch_size)

    n_uniform = int(
        args.n_uniform if args.n_uniform is not None else (legacy_n or 5000)
    )
    n_active = int(args.n_active if args.n_active is not None else (legacy_n or 5000))
    if n_uniform <= 0:
        raise ValueError(f"--n-uniform must be > 0, got {n_uniform}")
    if n_active < 0:
        raise ValueError(f"--n-active must be >= 0, got {n_active}")
    if args.eval_num_random_teams <= 0:
        raise ValueError(
            f"--eval-num-random-teams must be > 0, got {args.eval_num_random_teams}"
        )
    used_active = n_active > 0

    eval_battles_per_team = int(
        args.eval_battles_per_team
        if args.eval_battles_per_team is not None
        else (args.batch_size or 16)
    )
    if eval_battles_per_team <= 0:
        raise ValueError(
            f"--eval-battles-per-team must be > 0, got {eval_battles_per_team}"
        )

    eval_gpu_a = int(
        args.eval_gpu_a if args.eval_gpu_a is not None else (args.gpu_a or 0)
    )
    eval_gpu_b = int(
        args.eval_gpu_b if args.eval_gpu_b is not None else (args.gpu_b or 1)
    )
    sim_gpu_a = int(args.sim_gpu_a if args.sim_gpu_a is not None else (args.gpu_a or 0))
    sim_gpu_b = int(args.sim_gpu_b if args.sim_gpu_b is not None else (args.gpu_b or 1))
    eval_opponent_team_set = args.eval_opponent_team_set or args.opponent_team_set
    eval_team_set = (
        args.eval_team_set or f"team_construction_pipeline_eval_{args.format}"
    )
    eval_matchup_max_retries = int(
        args.eval_matchup_max_retries
        if args.eval_matchup_max_retries is not None
        else args.max_retries
    )
    eval_matchup_retry_sleep_sec = float(
        args.eval_matchup_retry_sleep_sec
        if args.eval_matchup_retry_sleep_sec is not None
        else args.retry_sleep_sec
    )

    data_root = Path(args.data_root)
    if args.reset_data_root and data_root.exists():
        shutil.rmtree(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    sim_work_dir = (
        args.sim_work_dir
        if args.sim_work_dir is not None
        else data_root / "team_construction_sim_battles"
    )
    sim_work_dir.mkdir(parents=True, exist_ok=True)

    pool_path = data_root / "pool.json"
    uniform_data_path = data_root / "battles_uniform.jsonl"
    active_data_path = data_root / "battles_active.jsonl"
    train_data_path = data_root / "battles_train.jsonl"
    uniform_model_path = data_root / "interaction_uniform.pkl"
    final_model_path = data_root / "interaction_final.pkl"
    equilibrium_path = data_root / "equilibrium.json"
    manifest_path = data_root / "run_all_manifest.json"

    phase_steps = ["build_pool", "simulate_uniform", "fit_uniform", "fit_final"]
    if used_active:
        phase_steps.insert(3, "simulate_active")
    phase_steps.append("optimize_metagame")
    if args.eval_enable:
        phase_steps.append("kakuna_benchmark")
    if args.vs is not None or args.vs_team_file is not None:
        phase_steps.append("best_response")
    if not args.skip_equilibrium:
        phase_steps.append("equilibrium")
    if not args.skip_checks:
        phase_steps.append("checks")
    phase_steps.append("write_manifest")
    phase_bar = tqdm(
        total=len(phase_steps),
        desc="[run-all] phases",
        unit="phase",
        dynamic_ncols=True,
    )

    wandb_run = None
    wandb_mod = None
    stage_counter = 0
    if args.log_wandb:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is not installed. Install it or run without --log-wandb."
            ) from exc
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                "pipeline": "run-all",
                "format": args.format,
                "usage_month": args.usage_month,
                "usage_threshold": args.usage_threshold,
                "rank": args.rank,
                "backend": args.backend,
                "agent": args.agent,
                "sim_checkpoint": args.sim_checkpoint,
                "sim_gpu_a": sim_gpu_a,
                "sim_gpu_b": sim_gpu_b,
                "sim_work_dir": str(sim_work_dir),
                "sim_print_match_stats": bool(args.sim_print_match_stats),
                "team_size": args.team_size,
                "replace": bool(args.replace),
                "seed": args.seed,
                "concurrency": args.concurrency,
                "timeout_sec": args.timeout_sec,
                "max_retries": args.max_retries,
                "retry_sleep_sec": args.retry_sleep_sec,
                "flush_every": args.flush_every,
                "n_uniform": n_uniform,
                "n_active": n_active,
                "val_fraction": args.val_fraction,
                "max_iter": args.max_iter,
                "tune_C": bool(args.tune_C),
                "c_values": args.c_values,
                "C": args.C,
                "metagame_N": args.metagame_N,
                "metagame_random_restarts": args.metagame_random_restarts,
                "best_response_random_restarts": args.best_response_random_restarts,
                "pool_expansion": args.pool_expansion,
                "br_random_restarts": args.br_random_restarts,
                "exploitability_tol": args.exploitability_tol,
                "max_size": args.max_size,
                "support_tol": args.support_tol,
                "eval_enable": bool(args.eval_enable),
                "eval_model_name": args.eval_model_name,
                "eval_checkpoint": args.eval_checkpoint,
                "eval_opponent_team_set": eval_opponent_team_set,
                "eval_team_set": eval_team_set,
                "eval_battles_per_team": eval_battles_per_team,
                "eval_num_random_teams": args.eval_num_random_teams,
                "eval_gpu_a": eval_gpu_a,
                "eval_gpu_b": eval_gpu_b,
                "eval_matchup_max_retries": eval_matchup_max_retries,
                "eval_matchup_retry_sleep_sec": eval_matchup_retry_sleep_sec,
                "eval_margin_tol": args.eval_margin_tol,
                "fail_on_no_improvement": bool(args.fail_on_no_improvement),
                "legacy_opponent_team_set": args.opponent_team_set,
                "legacy_learner_team_set": args.learner_team_set,
                "legacy_num_batches": args.num_batches,
                "legacy_batch_size": args.batch_size,
                "legacy_gpu_a": args.gpu_a,
                "legacy_gpu_b": args.gpu_b,
                "legacy_epsilon_start": args.epsilon_start,
                "legacy_epsilon_end": args.epsilon_end,
                "legacy_thompson_candidate_pool_size": args.thompson_candidate_pool_size,
                "legacy_weight_team": args.weight_team,
                "legacy_weight_pokemon": args.weight_pokemon,
                "legacy_weight_moves": args.weight_moves,
            },
        )
        wandb_mod = wandb
        wandb.define_metric("run_all_stage")
        wandb.define_metric("*", step_metric="run_all_stage")

    def _wandb_log(stage_name: str, **metrics: float | int | str | bool) -> None:
        nonlocal stage_counter
        if wandb_run is None:
            return
        stage_counter += 1
        payload = {
            "run_all_stage": stage_counter,
            "run_all_stage_name": stage_name,
        }
        payload.update(metrics)
        wandb_run.log(payload)

    def _wandb_safe(callable_obj, *args, **kwargs) -> None:
        if wandb_run is None or wandb_mod is None:
            return
        try:
            callable_obj(*args, **kwargs)
        except Exception as exc:
            print(
                f"[run-all] WARN: W&B rich logging failed ({type(exc).__name__}: {exc})"
            )

    def _phase_done(name: str) -> None:
        phase_bar.update(1)
        phase_bar.set_postfix_str(name)

    try:
        print("[run-all] Stage: build pool")
        cmd_build_pool(
            argparse.Namespace(
                format=args.format,
                usage_month=args.usage_month,
                usage_threshold=args.usage_threshold,
                rank=args.rank,
                replication_movesets_json=args.replication_movesets_json,
                manual_sets_json=args.manual_sets_json,
                strict_max_evs=args.strict_max_evs,
                out=pool_path,
            )
        )
        pool_data = load_pool_artifact(pool_path)
        pokemon_sets = pool_pokemon_sets(pool_data)
        species = [ps.species for ps in pokemon_sets]
        pool_format = str(pool_data.get("format_id") or args.format)
        species_clause_keys = build_species_clause_keys(pool_format, pokemon_sets)
        _wandb_log("build_pool", pool_size=len(pokemon_sets))
        _wandb_safe(
            _wandb_log_pool_snapshot,
            wandb_run=wandb_run,
            wandb_mod=wandb_mod,
            pokemon_sets=pokemon_sets,
            top_k=max(10, min(30, len(pokemon_sets))),
        )
        _phase_done("build_pool")

        print("[run-all] Stage: simulate uniform dataset")
        cmd_simulate(
            argparse.Namespace(
                pool=pool_path,
                n=n_uniform,
                seed=args.seed,
                team_size=args.team_size,
                replace=args.replace,
                format=args.format,
                agent=args.agent,
                backend=args.backend,
                sampling_strategy="uniform",
                active_model=None,
                active_candidate_pool_size=args.active_candidate_pool_size,
                active_uniform_mix=args.active_uniform_mix,
                active_min_uncertainty=args.active_min_uncertainty,
                concurrency=args.concurrency,
                timeout_sec=args.timeout_sec,
                max_retries=args.max_retries,
                retry_sleep_sec=args.retry_sleep_sec,
                checkpoint=args.sim_checkpoint,
                gpu_a=sim_gpu_a,
                gpu_b=sim_gpu_b,
                work_dir=sim_work_dir,
                print_match_stats=args.sim_print_match_stats,
                flush_every=args.flush_every,
                out=uniform_data_path,
                metadata_out=None,
            )
        )
        uniform_rows = _count_jsonl_rows(uniform_data_path)
        _wandb_log("simulate_uniform", uniform_rows=uniform_rows)
        _phase_done("simulate_uniform")

        print("[run-all] Stage: fit interaction model on uniform dataset")
        cmd_fit_interaction(
            argparse.Namespace(
                input=uniform_data_path,
                pool=pool_path,
                out=uniform_model_path,
                val_fraction=args.val_fraction,
                seed=args.seed,
                max_iter=args.max_iter,
                tune_C=args.tune_C,
                c_values=args.c_values,
                C=args.C,
                detail_k=args.detail_k,
            )
        )
        uniform_model = load_artifact(uniform_model_path)
        uniform_fit = uniform_model.get("fit", {})
        _wandb_log(
            "fit_uniform",
            uniform_best_C=float(uniform_fit.get("best_C", 0.0)),
            uniform_n_examples=int(uniform_fit.get("n_examples", 0)),
        )
        _wandb_safe(
            _wandb_log_model_snapshot,
            wandb_run=wandb_run,
            wandb_mod=wandb_mod,
            stage_prefix="uniform_model",
            model=uniform_model,
            species=species,
            top_k=max(10, min(30, len(species))),
        )
        _phase_done("fit_uniform")

        if used_active:
            print("[run-all] Stage: simulate active dataset")
            cmd_simulate(
                argparse.Namespace(
                    pool=pool_path,
                    n=n_active,
                    seed=args.seed + 1,
                    team_size=args.team_size,
                    replace=args.replace,
                    format=args.format,
                    agent=args.agent,
                    backend=args.backend,
                    sampling_strategy="active",
                    active_model=uniform_model_path,
                    active_candidate_pool_size=args.active_candidate_pool_size,
                    active_uniform_mix=args.active_uniform_mix,
                    active_min_uncertainty=args.active_min_uncertainty,
                    concurrency=args.concurrency,
                    timeout_sec=args.timeout_sec,
                    max_retries=args.max_retries,
                    retry_sleep_sec=args.retry_sleep_sec,
                    checkpoint=args.sim_checkpoint,
                    gpu_a=sim_gpu_a,
                    gpu_b=sim_gpu_b,
                    work_dir=sim_work_dir,
                    print_match_stats=args.sim_print_match_stats,
                    flush_every=args.flush_every,
                    out=active_data_path,
                    metadata_out=None,
                )
            )

            total_rows = _merge_jsonl_files(
                [uniform_data_path, active_data_path], out=train_data_path
            )
            print(
                f"[run-all] Merged uniform+active datasets -> {train_data_path} "
                f"(rows={total_rows})"
            )
            _wandb_log(
                "simulate_active",
                active_rows=_count_jsonl_rows(active_data_path),
                merged_rows=total_rows,
            )
            _phase_done("simulate_active")

            print("[run-all] Stage: refit interaction model on merged dataset")
            cmd_fit_interaction(
                argparse.Namespace(
                    input=train_data_path,
                    pool=pool_path,
                    out=final_model_path,
                    val_fraction=args.val_fraction,
                    seed=args.seed + 1,
                    max_iter=args.max_iter,
                    tune_C=args.tune_C,
                    c_values=args.c_values,
                    C=args.C,
                    detail_k=args.detail_k,
                )
            )
        else:
            train_data_path = uniform_data_path
            shutil.copyfile(uniform_model_path, final_model_path)
            print(
                "[run-all] Active stage disabled (n_active=0); "
                f"reusing uniform model -> {final_model_path}"
            )
            _wandb_log("simulate_active", active_rows=0, merged_rows=uniform_rows)

        final_model = load_artifact(final_model_path)
        final_fit = final_model.get("fit", {})
        _wandb_log(
            "fit_final",
            final_best_C=float(final_fit.get("best_C", 0.0)),
            final_n_examples=int(final_fit.get("n_examples", 0)),
        )
        _wandb_safe(
            _wandb_log_model_snapshot,
            wandb_run=wandb_run,
            wandb_mod=wandb_mod,
            stage_prefix="final_model",
            model=final_model,
            species=species,
            top_k=max(10, min(30, len(species))),
        )
        _phase_done("fit_final")

        print("[run-all] Stage: metagame optimization")
        best_team, avg_prob, history, _, restart_runs = _optimize_metagame_team(
            model=final_model,
            pokemon_sets=pokemon_sets,
            team_size=args.team_size,
            n_opponents=args.metagame_N,
            seed=args.seed,
            init_mode=args.init,
            init_team_text=args.init_team,
            random_restarts=args.metagame_random_restarts,
            species_clause_keys=species_clause_keys,
        )
        _print_team("Metagame-optimized team", best_team, species)
        print(f"Predicted average win probability vs sampled metagame: {avg_prob:.6f}")
        _print_restart_summary(restart_runs, species)
        print("Accepted swaps (objective monotonicity trace):")
        for row in history:
            if row["event"] == "init":
                print(f"  init objective={row['objective']:.6f}")
            else:
                print(
                    f"  slot {row['slot']} out={species[row['out']]} in={species[row['in']]} "
                    f"objective={row['objective']:.6f}"
                )
        _wandb_log(
            "optimize_metagame",
            metagame_N=int(args.metagame_N),
            metagame_avg_prob=float(avg_prob),
        )
        _wandb_safe(
            _wandb_log_metagame_outputs,
            wandb_run=wandb_run,
            wandb_mod=wandb_mod,
            best_team=best_team,
            avg_prob=avg_prob,
            history=history,
            restart_runs=restart_runs,
            species=species,
        )
        _phase_done("optimize_metagame")

        benchmark: dict | None = None
        if args.eval_enable:
            print("[run-all] Stage: Kakuna benchmark vs random-team baseline")
            if args.backend == "synthetic":
                reason = "backend is synthetic"
                print(f"[run-all] Skipping Kakuna benchmark ({reason})")
                benchmark = {"enabled": True, "ran": False, "skipped_reason": reason}
                _wandb_log("kakuna_benchmark", skipped=True, skipped_reason=reason)
                _phase_done("kakuna_benchmark")
            else:
                if not eval_opponent_team_set:
                    raise ValueError(
                        "--eval-opponent-team-set (or --opponent-team-set) is required "
                        "when --eval-enable is set."
                    )
                cache_dir_env = os.environ.get("METAMON_CACHE_DIR")
                if not cache_dir_env:
                    raise ValueError(
                        "METAMON_CACHE_DIR must be set to run Kakuna benchmark evaluation."
                    )
                cache_dir = Path(cache_dir_env)
                eval_work_dir = (
                    args.eval_work_dir
                    if args.eval_work_dir is not None
                    else data_root / "team_construction_eval_battles"
                )
                eval_work_dir.mkdir(parents=True, exist_ok=True)
                eval_team_dir = _init_custom_teamset_dir(
                    cache_dir, eval_team_set, args.format
                )

                pipeline_showdown = team_ids_to_showdown(best_team, pokemon_sets)
                _write_custom_team(eval_team_dir, args.format, pipeline_showdown)
                pipeline_result = _run_matchup_with_retry(
                    battle_format=args.format,
                    num_battles=eval_battles_per_team,
                    model_name=args.eval_model_name,
                    team_set_a=eval_team_set,
                    team_set_b=eval_opponent_team_set,
                    gpu_a=eval_gpu_a,
                    gpu_b=eval_gpu_b,
                    work_dir=eval_work_dir,
                    checkpoint=args.eval_checkpoint,
                    max_retries=eval_matchup_max_retries,
                    retry_sleep_sec=eval_matchup_retry_sleep_sec,
                    print_match_stats=args.eval_print_match_stats,
                )
                pipeline_wr = float(pipeline_result["acceptor_summary"]["win_rate"])

                rng_eval = random.Random(args.seed + 98765)
                random_wrs: list[float] = []
                random_rows: list[dict] = []
                pool_ids = list(range(len(pokemon_sets)))
                random_eval_iter = tqdm(
                    range(args.eval_num_random_teams),
                    desc="[run-all] random baseline",
                    unit="team",
                    dynamic_ncols=True,
                    leave=False,
                )
                for i in random_eval_iter:
                    team = sample_team(
                        pool_ids,
                        team_size=args.team_size,
                        replace=False,
                        rng=rng_eval,
                        species_clause_keys=species_clause_keys,
                    )
                    _write_custom_team(
                        eval_team_dir,
                        args.format,
                        team_ids_to_showdown(team, pokemon_sets),
                    )
                    result = _run_matchup_with_retry(
                        battle_format=args.format,
                        num_battles=eval_battles_per_team,
                        model_name=args.eval_model_name,
                        team_set_a=eval_team_set,
                        team_set_b=eval_opponent_team_set,
                        gpu_a=eval_gpu_a,
                        gpu_b=eval_gpu_b,
                        work_dir=eval_work_dir,
                        checkpoint=args.eval_checkpoint,
                        max_retries=eval_matchup_max_retries,
                        retry_sleep_sec=eval_matchup_retry_sleep_sec,
                        print_match_stats=False,
                    )
                    wr = float(result["acceptor_summary"]["win_rate"])
                    random_wrs.append(wr)
                    random_rows.append(
                        {
                            "team_ids": list(team),
                            "team_species": [species[idx] for idx in team],
                            "win_rate": wr,
                        }
                    )
                    print(
                        f"[run-all] random baseline {i + 1}/{args.eval_num_random_teams} "
                        f"win_rate={wr:.3f}"
                    )

                random_mean = float(np.mean(random_wrs))
                random_std = float(np.std(random_wrs, ddof=0))
                margin = float(pipeline_wr - random_mean)
                outperform = bool(margin > float(args.eval_margin_tol))
                print(
                    "[run-all] benchmark summary: "
                    f"pipeline_wr={pipeline_wr:.3f} random_mean={random_mean:.3f} "
                    f"random_std={random_std:.3f} margin={margin:+.3f} "
                    f"outperform={outperform}"
                )

                benchmark = {
                    "enabled": True,
                    "ran": True,
                    "model_name": args.eval_model_name,
                    "checkpoint": args.eval_checkpoint,
                    "opponent_team_set": eval_opponent_team_set,
                    "team_set_a": eval_team_set,
                    "battles_per_team": eval_battles_per_team,
                    "num_random_teams": int(args.eval_num_random_teams),
                    "gpu_a": eval_gpu_a,
                    "gpu_b": eval_gpu_b,
                    "pipeline_team_ids": list(best_team),
                    "pipeline_team_species": [species[idx] for idx in best_team],
                    "pipeline_win_rate": pipeline_wr,
                    "random_mean_win_rate": random_mean,
                    "random_std_win_rate": random_std,
                    "margin_vs_random_mean": margin,
                    "margin_tol": float(args.eval_margin_tol),
                    "outperform_random": outperform,
                    "random_samples": random_rows,
                }
                _wandb_log(
                    "kakuna_benchmark",
                    pipeline_win_rate=pipeline_wr,
                    random_mean_win_rate=random_mean,
                    random_std_win_rate=random_std,
                    margin_vs_random_mean=margin,
                    margin_tol=float(args.eval_margin_tol),
                    outperform_random=outperform,
                )
                _wandb_safe(
                    _wandb_log_benchmark_outputs,
                    wandb_run=wandb_run,
                    wandb_mod=wandb_mod,
                    benchmark=benchmark,
                )
                _phase_done("kakuna_benchmark")
                if args.fail_on_no_improvement and not outperform:
                    raise RuntimeError(
                        "Pipeline team did not beat random-team baseline on Kakuna benchmark. "
                        "Revisit model/search settings."
                    )
        else:
            benchmark = {"enabled": False, "ran": False}
            _wandb_log("kakuna_benchmark", skipped=True, skipped_reason="disabled")

        if args.vs is not None or args.vs_team_file is not None:
            print("[run-all] Optional stage: best-response vs provided team")
            cmd_best_response(
                argparse.Namespace(
                    model=final_model_path,
                    pool=pool_path,
                    team_size=args.team_size,
                    vs=args.vs,
                    vs_team_file=args.vs_team_file,
                    init=args.init,
                    init_team=args.init_team,
                    seed=args.seed,
                    random_restarts=args.best_response_random_restarts,
                )
            )
            _wandb_log("best_response", ran=True)
            _phase_done("best_response")

        if not args.skip_equilibrium:
            print("[run-all] Stage: equilibrium solve")
            cmd_equilibrium(
                argparse.Namespace(
                    model=final_model_path,
                    pool=pool_path,
                    team_size=args.team_size,
                    seed=args.seed,
                    seed_team_from=args.seed_team_from,
                    seed_team=args.seed_team,
                    pool_expansion=args.pool_expansion,
                    metagame_N=args.metagame_N,
                    metagame_random_restarts=args.metagame_random_restarts,
                    br_random_restarts=args.br_random_restarts,
                    exploitability_tol=args.exploitability_tol,
                    max_size=args.max_size,
                    support_tol=args.support_tol,
                    out=equilibrium_path,
                )
            )
            if equilibrium_path.exists():
                eq_payload = json.loads(equilibrium_path.read_text(encoding="utf-8"))
                row_mix = np.asarray(
                    eq_payload["equilibrium"]["row_mixture"], dtype=np.float64
                )
                support_size = int(np.sum(row_mix > float(args.support_tol)))
                _wandb_safe(
                    _wandb_log_equilibrium_outputs,
                    wandb_run=wandb_run,
                    wandb_mod=wandb_mod,
                    eq_payload=eq_payload,
                    support_tol=float(args.support_tol),
                )
            else:
                support_size = 0
            _wandb_log("equilibrium", support_size=support_size)
            _phase_done("equilibrium")
        else:
            print("[run-all] Skipping equilibrium stage (--skip-equilibrium)")
            _wandb_log("equilibrium", skipped=True)

        if not args.skip_checks:
            print("[run-all] Final checks")
            cmd_checks(
                argparse.Namespace(
                    model=final_model_path,
                    pool=pool_path,
                    team_size=args.team_size,
                    seed=args.seed,
                    num_pairs=args.check_num_pairs,
                    payoff_pool_size=args.check_payoff_pool_size,
                    repro_n=args.check_repro_n,
                )
            )
            _wandb_log("checks", ran=True)
            _phase_done("checks")
        else:
            print("[run-all] Skipping checks (--skip-checks)")
            _wandb_log("checks", skipped=True)

        manifest = {
            "format": args.format,
            "usage_month": args.usage_month,
            "backend": args.backend,
            "agent": args.agent,
            "seed": int(args.seed),
            "team_size": int(args.team_size),
            "n_uniform": int(n_uniform),
            "n_active": int(n_active),
            "used_active": bool(used_active),
            "simulation": {
                "checkpoint": args.sim_checkpoint,
                "gpu_a": sim_gpu_a,
                "gpu_b": sim_gpu_b,
                "work_dir": str(sim_work_dir),
                "print_match_stats": bool(args.sim_print_match_stats),
            },
            "eval": {
                "enabled": bool(args.eval_enable),
                "model_name": args.eval_model_name,
                "checkpoint": args.eval_checkpoint,
                "opponent_team_set": eval_opponent_team_set,
                "team_set_a": eval_team_set,
                "battles_per_team": eval_battles_per_team,
                "num_random_teams": int(args.eval_num_random_teams),
                "gpu_a": eval_gpu_a,
                "gpu_b": eval_gpu_b,
                "matchup_max_retries": eval_matchup_max_retries,
                "matchup_retry_sleep_sec": eval_matchup_retry_sleep_sec,
                "margin_tol": float(args.eval_margin_tol),
                "fail_on_no_improvement": bool(args.fail_on_no_improvement),
                "result": benchmark,
            },
            "legacy_args": {
                "opponent_team_set": args.opponent_team_set,
                "learner_team_set": args.learner_team_set,
                "num_batches": args.num_batches,
                "batch_size": args.batch_size,
                "gpu_a": args.gpu_a,
                "gpu_b": args.gpu_b,
                "epsilon_start": args.epsilon_start,
                "epsilon_end": args.epsilon_end,
                "thompson_candidate_pool_size": args.thompson_candidate_pool_size,
                "weight_team": args.weight_team,
                "weight_pokemon": args.weight_pokemon,
                "weight_moves": args.weight_moves,
            },
            "paths": {
                "pool": str(pool_path),
                "uniform_dataset": str(uniform_data_path),
                "active_dataset": str(active_data_path) if used_active else None,
                "train_dataset": str(train_data_path),
                "uniform_model": str(uniform_model_path),
                "final_model": str(final_model_path),
                "equilibrium": (
                    str(equilibrium_path) if not args.skip_equilibrium else None
                ),
            },
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        _wandb_log("complete", manifest_path=str(manifest_path))
        _wandb_safe(
            _wandb_log_run_artifact,
            wandb_run=wandb_run,
            wandb_mod=wandb_mod,
            artifact_name=(
                f"team-construction-{args.format}-{args.usage_month}-"
                f"{int(time.time())}"
            ),
            file_paths=[
                pool_path,
                uniform_data_path,
                active_data_path if used_active else None,
                train_data_path,
                uniform_model_path,
                final_model_path,
                equilibrium_path if not args.skip_equilibrium else None,
                manifest_path,
            ],
        )
        _phase_done("write_manifest")
        print(f"[run-all] Done. Manifest -> {manifest_path}")
    finally:
        phase_bar.close()
        if wandb_run is not None:
            wandb_run.finish()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end team-construction pipeline: pool build, simulation, "
            "logistic fitting, coordinate-ascent search, and restricted-game equilibrium."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "build-pool", help="Build eligible Pokemon pool and standardized sets"
    )
    p.add_argument("--format", required=True, help="Battle format, e.g. gen1ou")
    p.add_argument(
        "--usage-month",
        required=True,
        help="Usage month in YYYY-MM (paper replication uses 2025-07)",
    )
    p.add_argument("--usage-threshold", type=float, default=0.001)
    p.add_argument("--rank", type=int, default=1500)
    p.add_argument("--replication-movesets-json", type=Path, default=None)
    p.add_argument("--manual-sets-json", type=Path, default=None)
    p.add_argument("--strict-max-evs", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_build_pool)

    p = sub.add_parser("simulate", help="Generate battle dataset via simulation")
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--team-size", type=int, default=6)
    p.add_argument("--replace", action="store_true")
    p.add_argument("--format", default=None)
    p.add_argument("--agent", default="SimpleHeuristicsPlayer")
    p.add_argument(
        "--backend", choices=["poke_env", "metamon", "synthetic"], default="poke_env"
    )
    p.add_argument(
        "--sampling-strategy",
        choices=["uniform", "active"],
        default="uniform",
        help="uniform=random team pairs, active=model-guided uncertain pairs",
    )
    p.add_argument(
        "--active-model",
        type=Path,
        default=None,
        help="Model artifact used for active sampling uncertainty scores",
    )
    p.add_argument("--active-candidate-pool-size", type=int, default=256)
    p.add_argument("--active-uniform-mix", type=float, default=0.25)
    p.add_argument("--active-min-uncertainty", type=float, default=1e-6)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--timeout-sec", type=float, default=240.0)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--retry-sleep-sec", type=float, default=2.0)
    p.add_argument("--checkpoint", type=int, default=None)
    p.add_argument("--gpu-a", type=int, default=0)
    p.add_argument("--gpu-b", type=int, default=1)
    p.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/team_prediction/team_construction_battles"),
        help="Used by backend=metamon (matchup worker scratch/results directory).",
    )
    p.add_argument("--print-match-stats", action="store_true")
    p.add_argument("--flush-every", type=int, default=50)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--metadata-out", type=Path, default=None)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("fit-baseline", help="Fit no-interaction logistic model")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-iter", type=int, default=2000)
    p.set_defaults(func=cmd_fit_baseline)

    p = sub.add_parser("fit-interaction", help="Fit interaction logistic model")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-iter", type=int, default=3000)
    p.add_argument("--tune-C", action="store_true")
    p.add_argument("--c-values", default="0.01,0.1,1,10,100")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--detail-k", type=int, default=3)
    p.set_defaults(func=cmd_fit_interaction)

    p = sub.add_parser(
        "best-response", help="Find best response to a fixed opponent team"
    )
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--team-size", type=int, default=6)
    p.add_argument(
        "--vs", type=str, default=None, help="Opponent team as names or team string"
    )
    p.add_argument("--vs-team-file", type=Path, default=None)
    p.add_argument(
        "--init", choices=["top_theta", "random", "explicit"], default="top_theta"
    )
    p.add_argument("--init-team", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--random-restarts", type=int, default=8)
    p.set_defaults(func=cmd_best_response)

    p = sub.add_parser(
        "optimize-metagame",
        help="Optimize average win probability against sampled metagame teams",
    )
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--team-size", type=int, default=6)
    p.add_argument("--N", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--init", choices=["top_theta", "random", "explicit"], default="top_theta"
    )
    p.add_argument("--init-team", type=str, default=None)
    p.add_argument("--random-restarts", type=int, default=8)
    p.set_defaults(func=cmd_optimize_metagame)

    p = sub.add_parser(
        "equilibrium",
        help="Expand restricted strategy pool via iterated BR and solve zero-sum equilibrium",
    )
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--team-size", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--seed-team-from",
        choices=["metagame", "top_theta", "explicit"],
        default="metagame",
    )
    p.add_argument("--seed-team", type=str, default=None)
    p.add_argument(
        "--pool-expansion",
        choices=["double_oracle", "last_response"],
        default="double_oracle",
    )
    p.add_argument("--metagame-N", type=int, default=100)
    p.add_argument("--metagame-random-restarts", type=int, default=8)
    p.add_argument("--br-random-restarts", type=int, default=8)
    p.add_argument("--exploitability-tol", type=float, default=1e-6)
    p.add_argument("--max-size", type=int, default=25)
    p.add_argument("--support-tol", type=float, default=1e-6)
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(func=cmd_equilibrium)

    p = sub.add_parser(
        "run-all",
        help="Single-command end-to-end run: pool -> simulate -> fit -> optimize -> equilibrium -> checks",
    )
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument(
        "--reset-data-root",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Delete --data-root before running (use --no-reset-data-root to keep existing files).",
    )
    p.add_argument(
        "--format",
        "--battle-format",
        dest="format",
        required=True,
        help="Battle format, e.g. gen9ou",
    )
    p.add_argument("--usage-month", required=True, help="Usage month in YYYY-MM")
    p.add_argument("--usage-threshold", type=float, default=0.001)
    p.add_argument("--rank", type=int, default=1500)
    p.add_argument("--replication-movesets-json", type=Path, default=None)
    p.add_argument("--manual-sets-json", type=Path, default=None)
    p.add_argument("--strict-max-evs", action="store_true")

    p.add_argument("--opponent-team-set", default=None)
    p.add_argument("--learner-team-set", default=None)
    p.add_argument("--num-batches", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--gpu-a", type=int, default=None)
    p.add_argument("--gpu-b", type=int, default=None)
    p.add_argument("--epsilon-start", type=float, default=None)
    p.add_argument("--epsilon-end", type=float, default=None)
    p.add_argument("--thompson-candidate-pool-size", type=int, default=None)
    p.add_argument("--weight-team", type=float, default=None)
    p.add_argument("--weight-pokemon", type=float, default=None)
    p.add_argument("--weight-moves", type=float, default=None)

    p.add_argument(
        "--backend", choices=["poke_env", "metamon", "synthetic"], default="poke_env"
    )
    p.add_argument("--agent", default="Kakuna")
    p.add_argument("--team-size", type=int, default=6)
    p.add_argument("--replace", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--timeout-sec", type=float, default=240.0)
    p.add_argument(
        "--max-retries",
        "--matchup-max-retries",
        dest="max_retries",
        type=int,
        default=2,
    )
    p.add_argument(
        "--retry-sleep-sec",
        "--matchup-retry-sleep-sec",
        dest="retry_sleep_sec",
        type=float,
        default=2.0,
    )
    p.add_argument("--sim-checkpoint", type=int, default=None)
    p.add_argument("--sim-gpu-a", type=int, default=None)
    p.add_argument("--sim-gpu-b", type=int, default=None)
    p.add_argument("--sim-work-dir", type=Path, default=None)
    p.add_argument("--sim-print-match-stats", action="store_true")
    p.add_argument("--flush-every", type=int, default=50)

    p.add_argument("--n-uniform", type=int, default=None)
    p.add_argument("--n-active", type=int, default=None)
    p.add_argument("--active-candidate-pool-size", type=int, default=256)
    p.add_argument("--active-uniform-mix", type=float, default=0.25)
    p.add_argument("--active-min-uncertainty", type=float, default=1e-6)

    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--max-iter", type=int, default=3000)
    p.add_argument(
        "--tune-C",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tune interaction regularization C over --c-values.",
    )
    p.add_argument("--c-values", default="0.01,0.1,1,10,100")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--detail-k", type=int, default=3)

    p.add_argument(
        "--init", choices=["top_theta", "random", "explicit"], default="top_theta"
    )
    p.add_argument("--init-team", type=str, default=None)
    p.add_argument("--metagame-N", type=int, default=100)
    p.add_argument("--metagame-random-restarts", type=int, default=16)
    p.add_argument("--best-response-random-restarts", type=int, default=16)

    p.add_argument("--vs", type=str, default=None)
    p.add_argument("--vs-team-file", type=Path, default=None)

    p.add_argument(
        "--seed-team-from",
        choices=["metagame", "top_theta", "explicit"],
        default="metagame",
    )
    p.add_argument("--seed-team", type=str, default=None)
    p.add_argument(
        "--pool-expansion",
        choices=["double_oracle", "last_response"],
        default="double_oracle",
    )
    p.add_argument("--br-random-restarts", type=int, default=16)
    p.add_argument("--exploitability-tol", type=float, default=1e-6)
    p.add_argument("--max-size", type=int, default=25)
    p.add_argument("--support-tol", type=float, default=1e-6)
    p.add_argument("--skip-equilibrium", action="store_true")

    p.add_argument("--skip-checks", action="store_true")
    p.add_argument("--check-num-pairs", type=int, default=64)
    p.add_argument("--check-payoff-pool-size", type=int, default=8)
    p.add_argument("--check-repro-n", type=int, default=32)

    p.add_argument(
        "--eval-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Kakuna benchmark comparing pipeline team vs random-team baseline.",
    )
    p.add_argument("--eval-model-name", default="Kakuna")
    p.add_argument("--eval-checkpoint", type=int, default=None)
    p.add_argument("--eval-opponent-team-set", default=None)
    p.add_argument("--eval-team-set", default=None)
    p.add_argument("--eval-battles-per-team", type=int, default=None)
    p.add_argument("--eval-num-random-teams", type=int, default=8)
    p.add_argument("--eval-gpu-a", type=int, default=None)
    p.add_argument("--eval-gpu-b", type=int, default=None)
    p.add_argument("--eval-work-dir", type=Path, default=None)
    p.add_argument("--eval-matchup-max-retries", type=int, default=None)
    p.add_argument("--eval-matchup-retry-sleep-sec", type=float, default=None)
    p.add_argument("--eval-margin-tol", type=float, default=0.0)
    p.add_argument("--eval-print-match-stats", action="store_true")
    p.add_argument("--fail-on-no-improvement", action="store_true")

    p.add_argument("--log-wandb", action="store_true")
    p.add_argument("--wandb-project", default="team_construction")
    p.add_argument(
        "--wandb-entity",
        default=os.environ.get("METAMON_WANDB_ENTITY", None),
    )
    p.add_argument("--wandb-run-name", default=None)
    p.set_defaults(func=cmd_run_all)

    p = sub.add_parser("checks", help="Run minimal correctness checks")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--team-size", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-pairs", type=int, default=64)
    p.add_argument("--payoff-pool-size", type=int, default=8)
    p.add_argument("--repro-n", type=int, default=32)
    p.set_defaults(func=cmd_checks)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
