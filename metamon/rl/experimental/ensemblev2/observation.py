from __future__ import annotations

from typing import Sequence

import gymnasium as gym
import numpy as np

from metamon.interface import ObservationSpace, UniversalState
from metamon.rl.experimental.ensemblev2.config import member_prefix

# Extra scalar features pulled straight off the UniversalState (obs-space-agnostic)
# and threaded to deciders via the policy. Keys are namespaced under ``ensemble/``
# so they never collide with a member's own obs keys.
ENSEMBLE_STATE_KEY = "ensemble/opponents_remaining"


class MultiObservationSpace(ObservationSpace):
    """Fan one ``UniversalState`` out to N members with different obs spaces.

    Each member keeps its own (already tokenized) :class:`ObservationSpace`. On
    every step we recompute each member's observation from the same eval-side
    ``UniversalState`` and merge them into a single flat dict whose keys are
    namespaced ``m{i}/<key>`` so the env can batch them and :class:`EnsembleV2Policy`
    can split them back out.

    This mirrors how the vectorized env already runs two different observation
    spaces (eval vs. opponent) on a single battle; here we run N eval-side spaces.

    Note: the ``illegal_actions`` key is NOT added here. The env appends a single
    canonical (``DefaultActionSpace``) illegal mask to the obs, and the policy
    translates it per-member.
    """

    def __init__(self, member_obs_spaces: Sequence[ObservationSpace]):
        if not member_obs_spaces:
            raise ValueError("MultiObservationSpace requires at least one member")
        self.member_obs_spaces = list(member_obs_spaces)
        self.prefixes = [member_prefix(i) for i in range(len(self.member_obs_spaces))]

    def reset(self):
        for space in self.member_obs_spaces:
            space.reset()

    @property
    def gym_space(self) -> gym.spaces.Dict:
        merged: dict[str, gym.spaces.Space] = {}
        for prefix, space in zip(self.prefixes, self.member_obs_spaces):
            sub_space = space.gym_space
            for key, value in sub_space.spaces.items():
                merged[f"{prefix}/{key}"] = value
        merged[ENSEMBLE_STATE_KEY] = gym.spaces.Box(
            low=0.0, high=6.0, shape=(1,), dtype=np.float32
        )
        return gym.spaces.Dict(merged)

    def state_to_obs(self, state: UniversalState) -> dict:
        merged: dict = {}
        for prefix, space in zip(self.prefixes, self.member_obs_spaces):
            member_obs = space.state_to_obs(state)
            for key, value in member_obs.items():
                merged[f"{prefix}/{key}"] = value
        merged[ENSEMBLE_STATE_KEY] = np.array(
            [float(state.opponents_remaining)], dtype=np.float32
        )
        return merged

    def member_obs_keys(self, index: int) -> list[str]:
        """Original (un-namespaced) obs keys for member ``index``."""
        return list(self.member_obs_spaces[index].gym_space.spaces.keys())
