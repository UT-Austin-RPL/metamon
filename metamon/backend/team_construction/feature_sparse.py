from __future__ import annotations

from typing import Sequence


def feature_dicts_to_csr(feature_dicts: Sequence[dict[int, float]], n_features: int):
    """Convert sparse dict rows into a scipy CSR matrix."""

    try:
        from scipy import sparse
    except ImportError as exc:
        raise ImportError(
            "scipy is required for sparse feature matrices. Install scipy to continue."
        ) from exc

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for row_idx, feat_dict in enumerate(feature_dicts):
        for col_idx, value in feat_dict.items():
            if value == 0:
                continue
            rows.append(row_idx)
            cols.append(int(col_idx))
            data.append(float(value))

    return sparse.csr_matrix(
        (data, (rows, cols)), shape=(len(feature_dicts), n_features)
    )
