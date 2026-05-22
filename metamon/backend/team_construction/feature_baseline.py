from __future__ import annotations

import numpy as np

from .core import BattleExample, Team
from .feature_sparse import feature_dicts_to_csr


def build_baseline_feature_dict(
    team_a: Team,
    team_b: Team,
    num_pokemon: int,
) -> dict[int, float]:
    out: dict[int, float] = {}
    for idx in team_a:
        if idx < 0 or idx >= num_pokemon:
            raise IndexError(f"Pokemon ID out of range: {idx}")
        out[idx] = out.get(idx, 0.0) + 1.0
    for idx in team_b:
        if idx < 0 or idx >= num_pokemon:
            raise IndexError(f"Pokemon ID out of range: {idx}")
        out[idx] = out.get(idx, 0.0) - 1.0
    return out


def build_baseline_matrix(examples: list[BattleExample], num_pokemon: int):
    rows = [
        build_baseline_feature_dict(ex.team_A, ex.team_B, num_pokemon)
        for ex in examples
    ]
    x = feature_dicts_to_csr(rows, n_features=num_pokemon)
    y = np.array([ex.y for ex in examples], dtype=np.int64)
    return x, y
