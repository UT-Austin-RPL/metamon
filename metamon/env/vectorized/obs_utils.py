"""Observation batching helpers for the vectorized Showdown env.

Copied/adapted from ``metamon.env.pokepy_battle.vector_env`` (these helpers are
NN/observation-generic, not pokepy-specific) so this package has no dependency on
the soon-to-be-removed pokepy backend.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


def stack_obs_dicts(obs_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    if not obs_list:
        return {}
    keys = obs_list[0].keys()
    return {k: np.stack([obs[k] for obs in obs_list], axis=0) for k in keys}


def unstack_obs_dicts(obs: Dict[str, np.ndarray]) -> List[Dict[str, np.ndarray]]:
    if not obs:
        return []
    batch = next(iter(obs.values())).shape[0]
    return [{k: v[i] for k, v in obs.items()} for i in range(batch)]


def numpy_obs_to_torch(
    obs: Dict[str, np.ndarray], device: torch.device
) -> Dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v).to(device).unsqueeze(1) for k, v in obs.items()}
