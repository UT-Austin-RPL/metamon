from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import BattleExample, Team
from .feature_sparse import feature_dicts_to_csr


@dataclass(frozen=True)
class InteractionFeatureLayout:
    num_pokemon: int

    @property
    def main_offset(self) -> int:
        return 0

    @property
    def num_main(self) -> int:
        return self.num_pokemon

    @property
    def synergy_offset(self) -> int:
        return self.main_offset + self.num_main

    @property
    def num_synergy(self) -> int:
        return self.num_pokemon * (self.num_pokemon - 1) // 2

    @property
    def matchup_offset(self) -> int:
        return self.synergy_offset + self.num_synergy

    @property
    def num_matchup(self) -> int:
        return self.num_pokemon * (self.num_pokemon - 1)

    @property
    def n_features(self) -> int:
        return self.matchup_offset + self.num_matchup

    def synergy_index(self, i: int, k: int) -> int:
        if i == k:
            raise ValueError("Synergy index undefined for identical IDs")
        if i > k:
            i, k = k, i
        start = i * (2 * self.num_pokemon - i - 1) // 2
        return self.synergy_offset + start + (k - i - 1)

    def matchup_index(self, i: int, j: int) -> int:
        if i == j:
            raise ValueError("Matchup index undefined for identical IDs")
        j_compact = j if j < i else j - 1
        return self.matchup_offset + i * (self.num_pokemon - 1) + j_compact


def _add_synergy_terms(
    out: dict[int, float],
    team: Team,
    sign: float,
    layout: InteractionFeatureLayout,
) -> None:
    for i in range(len(team)):
        for j in range(i + 1, len(team)):
            idx = layout.synergy_index(team[i], team[j])
            out[idx] = out.get(idx, 0.0) + sign


def _add_matchup_terms(
    out: dict[int, float],
    team_a: Team,
    team_b: Team,
    layout: InteractionFeatureLayout,
) -> None:
    for i in team_a:
        for j in team_b:
            if i == j:
                continue
            idx = layout.matchup_index(i, j)
            out[idx] = out.get(idx, 0.0) + 1.0


def build_interaction_feature_dict(
    team_a: Team,
    team_b: Team,
    layout: InteractionFeatureLayout,
) -> dict[int, float]:
    out: dict[int, float] = {}

    for idx in team_a:
        out[idx] = out.get(idx, 0.0) + 1.0
    for idx in team_b:
        out[idx] = out.get(idx, 0.0) - 1.0

    _add_synergy_terms(out, team_a, +1.0, layout)
    _add_synergy_terms(out, team_b, -1.0, layout)

    _add_matchup_terms(out, team_a, team_b, layout)

    return out


def build_interaction_matrix(
    examples: list[BattleExample], layout: InteractionFeatureLayout
):
    rows = [
        build_interaction_feature_dict(ex.team_A, ex.team_B, layout) for ex in examples
    ]
    x = feature_dicts_to_csr(rows, n_features=layout.n_features)
    y = np.array([ex.y for ex in examples], dtype=np.int64)
    return x, y


def synergy_vector_to_matrix(
    layout: InteractionFeatureLayout, alpha: np.ndarray
) -> np.ndarray:
    matrix = np.zeros((layout.num_pokemon, layout.num_pokemon), dtype=np.float64)
    ptr = 0
    for i in range(layout.num_pokemon):
        for k in range(i + 1, layout.num_pokemon):
            value = float(alpha[ptr])
            matrix[i, k] = value
            matrix[k, i] = value
            ptr += 1
    return matrix


def matchup_vector_to_matrix(
    layout: InteractionFeatureLayout, beta: np.ndarray
) -> np.ndarray:
    matrix = np.zeros((layout.num_pokemon, layout.num_pokemon), dtype=np.float64)
    ptr = 0
    for i in range(layout.num_pokemon):
        for j in range(layout.num_pokemon):
            if i == j:
                continue
            matrix[i, j] = float(beta[ptr])
            ptr += 1
    return matrix


def matchup_matrix_to_vector(
    layout: InteractionFeatureLayout, matrix: np.ndarray
) -> np.ndarray:
    values: list[float] = []
    for i in range(layout.num_pokemon):
        for j in range(layout.num_pokemon):
            if i == j:
                continue
            values.append(float(matrix[i, j]))
    return np.asarray(values, dtype=np.float64)
