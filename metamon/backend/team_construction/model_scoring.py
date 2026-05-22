from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .core import Team
from .feature_interaction import (
    InteractionFeatureLayout,
    matchup_vector_to_matrix,
    synergy_vector_to_matrix,
)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    z = np.exp(x)
    return float(z / (1.0 + z))


@dataclass
class BaselineScorer:
    theta: np.ndarray
    intercept: float

    def logit(self, team_a: Team, team_b: Team) -> float:
        return float(
            self.intercept
            + np.sum(self.theta[list(team_a)], dtype=np.float64)
            - np.sum(self.theta[list(team_b)], dtype=np.float64)
        )

    def predict(self, team_a: Team, team_b: Team) -> float:
        return _sigmoid(self.logit(team_a, team_b))


@dataclass
class InteractionScorer:
    theta: np.ndarray
    synergy_matrix: np.ndarray
    matchup_matrix: np.ndarray
    intercept: float = 0.0

    def __post_init__(self) -> None:
        self._intrinsic_cache: dict[Team, float] = {}

    def _team_intrinsic(self, team: Team) -> float:
        cached = self._intrinsic_cache.get(team)
        if cached is not None:
            return cached

        ids = list(team)
        value = float(np.sum(self.theta[ids], dtype=np.float64))
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                value += float(self.synergy_matrix[ids[i], ids[j]])

        self._intrinsic_cache[team] = value
        return value

    def logit(self, team_a: Team, team_b: Team) -> float:
        a = list(team_a)
        b = list(team_b)
        intrinsic = self._team_intrinsic(team_a) - self._team_intrinsic(team_b)
        matchup = float(np.sum(self.matchup_matrix[np.ix_(a, b)], dtype=np.float64))
        return float(self.intercept + intrinsic + matchup)

    def predict(self, team_a: Team, team_b: Team) -> float:
        return _sigmoid(self.logit(team_a, team_b))


def make_scorer(model_artifact: dict) -> Callable[[Team, Team], float]:
    model_type = model_artifact.get("model_type")
    if model_type == "baseline":
        scorer = BaselineScorer(
            theta=np.asarray(model_artifact["theta"], dtype=np.float64),
            intercept=float(model_artifact.get("intercept", 0.0)),
        )
        return scorer.predict

    if model_type == "interaction":
        num_pokemon = int(model_artifact["num_pokemon"])
        layout = InteractionFeatureLayout(num_pokemon=num_pokemon)
        theta = np.asarray(model_artifact["theta"], dtype=np.float64)
        alpha = np.asarray(model_artifact["alpha"], dtype=np.float64)
        beta = np.asarray(model_artifact["beta"], dtype=np.float64)
        scorer = InteractionScorer(
            theta=theta,
            synergy_matrix=synergy_vector_to_matrix(layout, alpha),
            matchup_matrix=matchup_vector_to_matrix(layout, beta),
            intercept=float(model_artifact.get("intercept", 0.0)),
        )
        return scorer.predict

    raise ValueError(f"Unknown model_type in artifact: {model_type}")


def predict_win_prob(team_A_ids: Team, team_B_ids: Team, params: dict) -> float:
    return float(make_scorer(params)(team_A_ids, team_B_ids))


def interaction_matrices(
    model_artifact: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if model_artifact.get("model_type") != "interaction":
        raise ValueError("interaction_matrices requires an interaction model artifact")

    num_pokemon = int(model_artifact["num_pokemon"])
    layout = InteractionFeatureLayout(num_pokemon=num_pokemon)
    theta = np.asarray(model_artifact["theta"], dtype=np.float64)
    alpha = np.asarray(model_artifact["alpha"], dtype=np.float64)
    beta = np.asarray(model_artifact["beta"], dtype=np.float64)
    synergy_matrix = synergy_vector_to_matrix(layout, alpha)
    matchup_matrix = matchup_vector_to_matrix(layout, beta)
    return theta, synergy_matrix, matchup_matrix
