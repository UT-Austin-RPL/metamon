from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .core import BattleExample
from .feature_baseline import build_baseline_matrix
from .feature_interaction import (
    InteractionFeatureLayout,
    build_interaction_matrix,
    matchup_matrix_to_vector,
    matchup_vector_to_matrix,
)
from .simulation import augment_swap_symmetry, split_before_augmentation


def _require_sklearn_logreg():
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for model fitting. Install scikit-learn to continue."
        ) from exc
    return LogisticRegression


def _binary_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    probs = np.clip(probs.astype(np.float64), 1e-9, 1.0 - 1e-9)
    y = y_true.astype(np.float64)
    log_loss = float(-np.mean(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs)))
    preds = (probs >= 0.5).astype(np.int64)
    accuracy = float(np.mean(preds == y_true))
    brier = float(np.mean((probs - y) ** 2))
    return {
        "log_loss": log_loss,
        "accuracy": accuracy,
        "brier": brier,
    }


def _require_two_classes(y: np.ndarray, *, context: str) -> None:
    labels = np.unique(y)
    if labels.size < 2:
        raise ValueError(
            f"{context} requires both classes 0/1, but observed labels={labels.tolist()}"
        )


def _center_theta(theta: np.ndarray) -> np.ndarray:
    if theta.size == 0:
        return theta
    return theta - float(np.mean(theta))


def _center_alpha(alpha: np.ndarray) -> np.ndarray:
    if alpha.size == 0:
        return alpha
    return alpha - float(np.mean(alpha))


def _center_and_antisymmetrize_beta(
    layout: InteractionFeatureLayout,
    beta: np.ndarray,
) -> np.ndarray:
    matrix = matchup_vector_to_matrix(layout, beta)
    mask = ~np.eye(layout.num_pokemon, dtype=bool)
    offdiag = matrix[mask]
    if offdiag.size > 0:
        matrix[mask] = offdiag - float(np.mean(offdiag))

    matrix = 0.5 * (matrix - matrix.T)
    np.fill_diagonal(matrix, 0.0)
    return matchup_matrix_to_vector(layout, matrix)


def fit_baseline_model(
    examples: Sequence[BattleExample],
    *,
    num_pokemon: int,
    val_fraction: float = 0.15,
    split_seed: int = 0,
    max_iter: int = 2000,
) -> dict:
    if not examples:
        raise ValueError("Need at least one training example to fit baseline model.")
    LogisticRegression = _require_sklearn_logreg()

    train_orig, val_orig = split_before_augmentation(
        examples, val_fraction=val_fraction, seed=split_seed
    )
    if not train_orig:
        raise ValueError(
            "Training split is empty. Lower --val-fraction or provide more examples."
        )
    train_aug = augment_swap_symmetry(train_orig)
    val_aug = augment_swap_symmetry(val_orig)

    x_train, y_train = build_baseline_matrix(train_aug, num_pokemon=num_pokemon)
    _require_two_classes(y_train, context="Baseline training")
    model = LogisticRegression(max_iter=max_iter)
    model.fit(x_train, y_train)

    val_metrics = None
    if val_aug:
        x_val, y_val = build_baseline_matrix(val_aug, num_pokemon=num_pokemon)
        val_probs = model.predict_proba(x_val)[:, 1]
        val_metrics = _binary_metrics(y_val, val_probs)

    full_aug = augment_swap_symmetry(list(examples))
    x_full, y_full = build_baseline_matrix(full_aug, num_pokemon=num_pokemon)
    _require_two_classes(y_full, context="Baseline refit")
    final_model = LogisticRegression(max_iter=max_iter)
    final_model.fit(x_full, y_full)

    theta = final_model.coef_[0].astype(np.float64)
    theta = _center_theta(theta)
    intercept = float(final_model.intercept_[0])

    return {
        "model_type": "baseline",
        "num_pokemon": int(num_pokemon),
        "theta": theta,
        "intercept": intercept,
        "fit": {
            "val_fraction": float(val_fraction),
            "split_seed": int(split_seed),
            "max_iter": int(max_iter),
            "n_examples": int(len(examples)),
            "n_train_original": int(len(train_orig)),
            "n_val_original": int(len(val_orig)),
            "n_train_augmented": int(len(train_aug)),
            "n_val_augmented": int(len(val_aug)),
            "val_metrics": val_metrics,
        },
    }


def fit_interaction_model(
    examples: Sequence[BattleExample],
    *,
    num_pokemon: int,
    c_values: Iterable[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    val_fraction: float = 0.15,
    split_seed: int = 0,
    max_iter: int = 3000,
) -> dict:
    if not examples:
        raise ValueError("Need at least one training example to fit interaction model.")
    LogisticRegression = _require_sklearn_logreg()

    layout = InteractionFeatureLayout(num_pokemon=num_pokemon)
    c_grid = [float(c) for c in c_values]
    if not c_grid:
        raise ValueError("c_values must contain at least one candidate")

    train_orig, val_orig = split_before_augmentation(
        examples, val_fraction=val_fraction, seed=split_seed
    )
    if not train_orig:
        raise ValueError(
            "Training split is empty. Lower --val-fraction or provide more examples."
        )
    train_aug = augment_swap_symmetry(train_orig)
    val_aug = augment_swap_symmetry(val_orig)

    x_train, y_train = build_interaction_matrix(train_aug, layout=layout)
    _require_two_classes(y_train, context="Interaction training")
    x_val = y_val = None
    if val_aug:
        x_val, y_val = build_interaction_matrix(val_aug, layout=layout)

    tuning_rows: list[dict] = []
    best_c = c_grid[0]
    best_loss = float("inf")

    for c in c_grid:
        model = LogisticRegression(
            penalty="l2",
            C=c,
            max_iter=max_iter,
            fit_intercept=False,
            solver="lbfgs",
        )
        model.fit(x_train, y_train)

        if x_val is not None and y_val is not None:
            val_probs = model.predict_proba(x_val)[:, 1]
            metrics = _binary_metrics(y_val, val_probs)
            val_loss = metrics["log_loss"]
        else:
            metrics = None
            val_loss = 0.0

        row = {
            "C": float(c),
            "val_metrics": metrics,
            "val_loss": float(val_loss),
        }
        tuning_rows.append(row)

        if val_loss < best_loss:
            best_loss = val_loss
            best_c = c

    full_aug = augment_swap_symmetry(list(examples))
    x_full, y_full = build_interaction_matrix(full_aug, layout=layout)
    _require_two_classes(y_full, context="Interaction refit")
    final_model = LogisticRegression(
        penalty="l2",
        C=best_c,
        max_iter=max_iter,
        fit_intercept=False,
        solver="lbfgs",
    )
    final_model.fit(x_full, y_full)

    coef = final_model.coef_[0].astype(np.float64)

    start = layout.main_offset
    end = start + layout.num_main
    theta = coef[start:end]

    start = layout.synergy_offset
    end = start + layout.num_synergy
    alpha = coef[start:end]

    start = layout.matchup_offset
    end = start + layout.num_matchup
    beta = coef[start:end]

    theta = _center_theta(theta)
    alpha = _center_alpha(alpha)
    beta = _center_and_antisymmetrize_beta(layout, beta)

    return {
        "model_type": "interaction",
        "num_pokemon": int(num_pokemon),
        "layout": {
            "num_pokemon": int(layout.num_pokemon),
            "main_offset": int(layout.main_offset),
            "num_main": int(layout.num_main),
            "synergy_offset": int(layout.synergy_offset),
            "num_synergy": int(layout.num_synergy),
            "matchup_offset": int(layout.matchup_offset),
            "num_matchup": int(layout.num_matchup),
            "n_features": int(layout.n_features),
        },
        "theta": theta,
        "alpha": alpha,
        "beta": beta,
        "intercept": 0.0,
        "fit": {
            "val_fraction": float(val_fraction),
            "split_seed": int(split_seed),
            "max_iter": int(max_iter),
            "C_candidates": c_grid,
            "best_C": float(best_c),
            "tuning": tuning_rows,
            "n_examples": int(len(examples)),
            "n_train_original": int(len(train_orig)),
            "n_val_original": int(len(val_orig)),
            "n_train_augmented": int(len(train_aug)),
            "n_val_augmented": int(len(val_aug)),
        },
    }
