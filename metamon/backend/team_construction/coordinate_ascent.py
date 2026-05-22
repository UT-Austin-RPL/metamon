from __future__ import annotations

import random
from typing import Callable, Mapping, Sequence

from .core import Team, canonical_team
from .simulation import sample_team

Evaluator = Callable[[Team], float]
PairEvaluator = Callable[[Team, Team], float]


def _team_is_species_clause_legal(
    team: Sequence[int],
    species_clause_keys: Mapping[int, object] | None,
) -> bool:
    if species_clause_keys is None:
        return True
    keys = [species_clause_keys.get(int(member), int(member)) for member in team]
    return len(set(keys)) == len(keys)


def top_theta_init_team(
    theta: Sequence[float],
    team_size: int = 6,
    species_clause_keys: Mapping[int, object] | None = None,
) -> Team:
    ranked = sorted(range(len(theta)), key=lambda idx: float(theta[idx]), reverse=True)
    if len(ranked) < team_size:
        raise ValueError(f"Need at least {team_size} Pokemon, got {len(ranked)}")
    if species_clause_keys is None:
        return canonical_team(ranked[:team_size], team_size=team_size)
    selected: list[int] = []
    seen_keys: set[object] = set()
    for idx in ranked:
        key = species_clause_keys.get(int(idx), int(idx))
        if key in seen_keys:
            continue
        selected.append(int(idx))
        seen_keys.add(key)
        if len(selected) == team_size:
            break
    if len(selected) < team_size:
        raise ValueError(
            "Need at least "
            f"{team_size} unique species-clause groups, got {len(selected)}"
        )
    return canonical_team(selected, team_size=team_size)


def coordinate_ascent_best_team(
    evaluator: Evaluator,
    init_team: Team,
    pool_ids: Sequence[int],
    *,
    team_size: int = 6,
    tol: float = 1e-12,
    species_clause_keys: Mapping[int, object] | None = None,
) -> tuple[Team, list[dict]]:
    """Coordinate-ascent local search as specified in the PDF workflow."""

    pool = sorted({int(x) for x in pool_ids})
    if len(pool) < team_size:
        raise ValueError(f"Pool has only {len(pool)} Pokemon, need {team_size}")
    if species_clause_keys is not None:
        unique_clause_groups = {
            species_clause_keys.get(member, member) for member in pool
        }
        if len(unique_clause_groups) < team_size:
            raise ValueError(
                "Pool has only "
                f"{len(unique_clause_groups)} unique species-clause groups, need {team_size}"
            )
    pool_set = set(pool)

    team = list(canonical_team(init_team, team_size=team_size))
    if any(idx not in pool_set for idx in team):
        raise ValueError(
            "init_team contains Pokemon IDs not in pool_ids: "
            + ", ".join(str(idx) for idx in team if idx not in pool_set)
        )
    if not _team_is_species_clause_legal(team, species_clause_keys):
        raise ValueError("init_team violates species clause.")
    current = canonical_team(team, team_size=team_size)
    current_obj = float(evaluator(current))

    history = [
        {
            "event": "init",
            "team": list(current),
            "objective": current_obj,
        }
    ]

    pos = 0
    while pos < team_size:
        incumbent = team[pos]
        occupied_keys: set[object] = set()
        if species_clause_keys is not None:
            occupied_keys = {
                species_clause_keys.get(member, member)
                for idx, member in enumerate(team)
                if idx != pos
            }
        best_candidate = incumbent
        best_team = current
        best_obj = current_obj

        for candidate in pool:
            if candidate == incumbent:
                continue
            if candidate in team:
                continue
            if (
                species_clause_keys is not None
                and species_clause_keys.get(candidate, candidate) in occupied_keys
            ):
                continue

            trial = list(team)
            trial[pos] = candidate
            trial_team = canonical_team(trial, team_size=team_size)
            trial_obj = float(evaluator(trial_team))

            if trial_obj > best_obj + tol:
                best_obj = trial_obj
                best_candidate = candidate
                best_team = trial_team

        if best_candidate != incumbent:
            team[pos] = best_candidate
            current = best_team
            current_obj = best_obj
            history.append(
                {
                    "event": "swap",
                    "slot": pos,
                    "out": int(incumbent),
                    "in": int(best_candidate),
                    "objective": current_obj,
                    "team": list(current),
                }
            )
            pos = 0
        else:
            pos += 1

    return current, history


def coordinate_ascent_multi_start(
    evaluator: Evaluator,
    *,
    primary_init: Team,
    pool_ids: Sequence[int],
    team_size: int = 6,
    random_restarts: int = 0,
    seed: int = 0,
    tol: float = 1e-12,
    species_clause_keys: Mapping[int, object] | None = None,
) -> tuple[Team, list[dict], list[dict]]:
    """Run coordinate ascent from multiple starts and keep the best local optimum."""

    if random_restarts < 0:
        raise ValueError(f"random_restarts must be >= 0, got {random_restarts}")

    pool = sorted({int(x) for x in pool_ids})
    primary = canonical_team(primary_init, team_size=team_size)

    rng = random.Random(seed)
    init_teams: list[Team] = [primary]
    seen = {primary}
    target_starts = 1 + random_restarts
    attempts = 0
    max_attempts = max(1000, target_starts * 100)
    while len(init_teams) < target_starts and attempts < max_attempts:
        attempts += 1
        candidate = sample_team(
            pool,
            team_size=team_size,
            replace=False,
            rng=rng,
            species_clause_keys=species_clause_keys,
        )
        if candidate in seen:
            continue
        init_teams.append(candidate)
        seen.add(candidate)

    best_team: Team | None = None
    best_obj = float("-inf")
    best_history: list[dict] = []
    best_restart_idx = 0
    run_summaries: list[dict] = []

    for restart_idx, init in enumerate(init_teams):
        final_team, history = coordinate_ascent_best_team(
            evaluator,
            init_team=init,
            pool_ids=pool,
            team_size=team_size,
            tol=tol,
            species_clause_keys=species_clause_keys,
        )
        final_obj = float(history[-1]["objective"])
        accepted_swaps = max(0, len(history) - 1)
        run_summaries.append(
            {
                "restart_index": restart_idx,
                "start_team": list(init),
                "final_team": list(final_team),
                "objective": final_obj,
                "accepted_swaps": accepted_swaps,
            }
        )

        better = final_obj > best_obj + tol
        tied = abs(final_obj - best_obj) <= tol and (
            best_team is None or tuple(final_team) < tuple(best_team)
        )
        if better or tied:
            best_team = final_team
            best_obj = final_obj
            best_history = history
            best_restart_idx = restart_idx

    if best_team is None:
        raise RuntimeError(
            "coordinate_ascent_multi_start failed to produce any candidate"
        )

    for row in run_summaries:
        row["selected"] = row["restart_index"] == best_restart_idx

    return best_team, best_history, run_summaries


def objective_vs_fixed_opponent(
    pair_evaluator: PairEvaluator,
    opponent_team: Team,
) -> Evaluator:
    def _objective(team: Team) -> float:
        return float(pair_evaluator(team, opponent_team))

    return _objective


def objective_vs_metagame(
    pair_evaluator: PairEvaluator,
    opponent_teams: Sequence[Team],
) -> Evaluator:
    if not opponent_teams:
        raise ValueError("opponent_teams cannot be empty")

    def _objective(team: Team) -> float:
        score = 0.0
        for opp in opponent_teams:
            score += float(pair_evaluator(team, opp))
        return score / len(opponent_teams)

    return _objective


def objective_vs_mixture(
    pair_evaluator: PairEvaluator,
    opponent_teams: Sequence[Team],
    opponent_weights: Sequence[float],
) -> Evaluator:
    if not opponent_teams:
        raise ValueError("opponent_teams cannot be empty")
    if len(opponent_teams) != len(opponent_weights):
        raise ValueError(
            f"opponent_teams/opponent_weights length mismatch: "
            f"{len(opponent_teams)} vs {len(opponent_weights)}"
        )

    weights = [float(w) for w in opponent_weights]
    tol = 1e-12
    if any(w < -tol for w in weights):
        raise ValueError("opponent_weights must be nonnegative")
    weights = [max(0.0, w) for w in weights]
    total = sum(weights)
    if total <= tol:
        raise ValueError("sum(opponent_weights) must be > 0")
    normalized = [w / total for w in weights]

    def _objective(team: Team) -> float:
        score = 0.0
        for weight, opp in zip(normalized, opponent_teams):
            score += weight * float(pair_evaluator(team, opp))
        return score

    return _objective


def sample_opponent_teams(
    pool_ids: Sequence[int],
    *,
    n: int,
    team_size: int,
    seed: int,
    replace: bool = False,
    species_clause_keys: Mapping[int, object] | None = None,
) -> list[Team]:
    rng = random.Random(seed)
    return [
        sample_team(
            pool_ids,
            team_size=team_size,
            replace=replace,
            rng=rng,
            species_clause_keys=species_clause_keys,
        )
        for _ in range(n)
    ]
