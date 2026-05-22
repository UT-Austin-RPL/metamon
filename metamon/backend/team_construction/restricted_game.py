from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .core import Team, canonical_team

PairEvaluator = Callable[[Team, Team], float]
BestResponseFn = Callable[[Team], Team]
MixtureBestResponseFn = Callable[[Sequence[Team], np.ndarray], Team]


def build_strategy_pool(
    seed_team: Team,
    *,
    best_response: BestResponseFn,
    max_size: int,
    stop_on_cycle: bool = True,
) -> list[Team]:
    if max_size <= 0:
        raise ValueError(f"max_size must be > 0, got {max_size}")

    pool: list[Team] = [canonical_team(seed_team)]
    seen = {pool[0]}

    while len(pool) < max_size:
        last = pool[-1]
        br = canonical_team(best_response(last))

        if br in seen:
            if stop_on_cycle:
                break
            return pool

        pool.append(br)
        seen.add(br)

    return pool


def expected_payoff_vs_mixture(
    team: Team,
    strategies: Sequence[Team],
    mixture: np.ndarray,
    pair_evaluator: PairEvaluator,
) -> float:
    if len(strategies) != int(mixture.shape[0]):
        raise ValueError(
            f"strategies/mixture length mismatch: {len(strategies)} vs {mixture.shape[0]}"
        )

    weights = np.asarray(mixture, dtype=np.float64)
    tol = 1e-12
    if np.any(weights < -tol):
        raise ValueError("mixture contains negative probabilities")
    weights = np.clip(weights, 0.0, None)
    total = float(np.sum(weights))
    if total <= tol:
        raise ValueError("mixture probabilities sum to zero")
    weights = weights / total

    value = 0.0
    for weight, opponent in zip(weights, strategies):
        p_win = float(pair_evaluator(team, opponent))
        value += float(weight) * (2.0 * p_win - 1.0)
    return float(value)


def build_strategy_pool_double_oracle(
    seed_team: Team,
    *,
    pair_evaluator: PairEvaluator,
    best_response_to_mixture: MixtureBestResponseFn,
    max_size: int,
    stop_on_cycle: bool = True,
    exploitability_tol: float = 1e-6,
) -> tuple[list[Team], list[dict]]:
    """Expand restricted strategies via best response to the current equilibrium mixture."""

    if max_size <= 0:
        raise ValueError(f"max_size must be > 0, got {max_size}")

    pool: list[Team] = [canonical_team(seed_team)]
    seen = {pool[0]}
    iterations: list[dict] = []

    while len(pool) < max_size:
        payoff = build_payoff_matrix(pool, pair_evaluator)
        eq = solve_zero_sum_equilibrium(payoff)
        row_mix = np.asarray(eq["row_mixture"], dtype=np.float64)
        br = canonical_team(best_response_to_mixture(pool, row_mix))
        br_value = expected_payoff_vs_mixture(br, pool, row_mix, pair_evaluator)
        game_value = float(eq["game_value"])
        exploitability = float(br_value - game_value)

        iterations.append(
            {
                "iteration": len(iterations),
                "pool_size": len(pool),
                "game_value": game_value,
                "best_response_value": br_value,
                "exploitability": exploitability,
                "best_response": list(br),
                "row_mixture": row_mix.tolist(),
            }
        )

        if exploitability <= exploitability_tol:
            break

        if br in seen:
            if stop_on_cycle:
                break
            return pool, iterations

        pool.append(br)
        seen.add(br)

    return pool, iterations


def build_payoff_matrix(
    strategies: Sequence[Team], pair_evaluator: PairEvaluator
) -> np.ndarray:
    k = len(strategies)
    if k == 0:
        raise ValueError("strategies cannot be empty")

    payoff = np.zeros((k, k), dtype=np.float64)

    for i in range(k):
        payoff[i, i] = 0.0
        for j in range(i + 1, k):
            p_ij = float(pair_evaluator(strategies[i], strategies[j]))
            value = 2.0 * p_ij - 1.0
            payoff[i, j] = value
            payoff[j, i] = -value

    return payoff


def payoff_antisymmetry_error(payoff: np.ndarray) -> float:
    return float(np.max(np.abs(payoff + payoff.T)))


def _normalize_mixture(mixture: np.ndarray, *, tol: float = 1e-12) -> np.ndarray:
    weights = np.asarray(mixture, dtype=np.float64).reshape(-1)
    if weights.size == 0:
        raise ValueError("mixture cannot be empty")
    weights = np.where(np.isfinite(weights), weights, 0.0)
    weights = np.clip(weights, 0.0, None)
    total = float(np.sum(weights))
    if total <= tol:
        return np.full(weights.shape, 1.0 / float(weights.size), dtype=np.float64)
    return weights / total


def solve_zero_sum_equilibrium(
    payoff: np.ndarray,
    *,
    symmetric_tol: float = 1e-6,
) -> dict:
    matrix = np.asarray(payoff, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"payoff must be square, got shape {matrix.shape}")

    try:
        import nashpy as nash
    except ImportError as exc:
        raise ImportError(
            "nashpy is required for equilibrium solving. Install nashpy to continue."
        ) from exc

    game = nash.Game(matrix, -matrix)
    solver = "support_enumeration"

    try:
        equilibria = list(game.support_enumeration())
    except Exception:
        equilibria = []

    if not equilibria:
        solver = "vertex_enumeration"
        try:
            equilibria = list(game.vertex_enumeration())
        except Exception:
            equilibria = []

    if equilibria:
        chosen_idx = 0
        for idx, (eq_row, eq_col) in enumerate(equilibria):
            if np.allclose(eq_row, eq_col, atol=symmetric_tol, rtol=0.0):
                chosen_idx = idx
                break

        sigma_row, sigma_col = equilibria[chosen_idx]
        sigma_row = _normalize_mixture(np.asarray(sigma_row, dtype=np.float64))
        sigma_col = _normalize_mixture(np.asarray(sigma_col, dtype=np.float64))
    else:
        solver = "linear_program"
        try:
            sigma_row, sigma_col = game.linear_program()
        except Exception as exc:
            raise RuntimeError(
                "nashpy failed to compute equilibrium via support_enumeration, "
                "vertex_enumeration, and linear_program"
            ) from exc

        chosen_idx = -1
        sigma_row = _normalize_mixture(np.asarray(sigma_row, dtype=np.float64))
        sigma_col = _normalize_mixture(np.asarray(sigma_col, dtype=np.float64))

    if payoff_antisymmetry_error(matrix) <= symmetric_tol:
        sigma_sym = _normalize_mixture(0.5 * (sigma_row + sigma_col))
        sigma_row = sigma_sym
        sigma_col = sigma_sym

    value = float(np.dot(sigma_row, matrix @ sigma_col))

    return {
        "chosen_index": int(chosen_idx),
        "solver": solver,
        "row_mixture": sigma_row,
        "col_mixture": sigma_col,
        "game_value": value,
        "all_equilibria": [
            {
                "row": np.asarray(r, dtype=np.float64),
                "col": np.asarray(c, dtype=np.float64),
            }
            for r, c in equilibria
        ],
    }
