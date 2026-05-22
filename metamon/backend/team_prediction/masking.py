"""Masker classes for creating (x, y) training pairs by masking team attributes."""

import copy
import ctypes
import random
from typing import Tuple

import torch.multiprocessing as mp

from metamon.backend.team_prediction.team import TeamSet, PokemonSet


class TeamMasker:
    """
    Base masker: creates (x, y) pairs by randomly masking team attributes.

    - attrs_prob_range: probability range for masking individual attributes
      (any attribute including name can be masked; high values may mask entire Pokemon)
    """

    def __init__(
        self,
        attrs_prob_range: Tuple[float, float] = (0.1, 1.0),
        include_stats: bool = False,
    ):
        self.attrs_prob_range = attrs_prob_range
        self.include_stats = include_stats

    def set_step(self, step: int) -> None:
        """Update training step (for curriculum subclasses)."""
        pass

    def _mask_pokemon(self, pokemon: PokemonSet) -> PokemonSet:
        """Mask a random subset of attributes (possibly all of them)."""
        data = pokemon.to_dict()
        maskable = pokemon.get_maskable_attrs(include_stats=self.include_stats)

        if not maskable:
            return PokemonSet.from_dict(data)

        # Discrete sampling: uniformly choose count from [min_count, max_count]
        min_frac, max_frac = self.attrs_prob_range
        min_count = max(1, round(min_frac * len(maskable)))
        max_count = max(min_count, round(max_frac * len(maskable)))
        num_to_mask = random.randint(min_count, max_count)

        for key, subkey in random.sample(maskable, num_to_mask):
            if subkey is None:
                if key == "name":
                    data["name"] = PokemonSet.MISSING_NAME
                elif key == "ability":
                    data["ability"] = PokemonSet.MISSING_ABILITY
                elif key == "item":
                    data["item"] = PokemonSet.MISSING_ITEM
                elif key == "tera_type":
                    data["tera_type"] = PokemonSet.MISSING_TERA_TYPE
            else:
                if key == "moves":
                    data["moves"][subkey] = PokemonSet.MISSING_MOVE
                elif key == "evs":
                    data["evs"][subkey] = PokemonSet.MISSING_EV
                elif key == "ivs":
                    data["ivs"][subkey] = PokemonSet.MISSING_IV

        return PokemonSet.from_dict(data)

    def mask(self, team: TeamSet) -> Tuple[TeamSet, TeamSet]:
        """Mask a team. Returns (masked_x, ground_truth_y)."""
        y = copy.deepcopy(team)
        x = copy.deepcopy(team)
        x.lead = self._mask_pokemon(x.lead)
        x.reserve = [self._mask_pokemon(p) for p in x.reserve]
        return x, y

    def __repr__(self) -> str:
        return f"TeamMasker(attrs={self.attrs_prob_range})"


class NamesOnlyMasker(TeamMasker):
    """Toy masker: only masks Pokemon names."""

    def __init__(self, mask_all: bool = True):
        super().__init__()
        self.mask_all = mask_all

    def mask(self, team: TeamSet) -> Tuple[TeamSet, TeamSet]:
        y = copy.deepcopy(team)
        x = copy.deepcopy(team)

        all_pokemon = [x.lead] + list(x.reserve)
        if self.mask_all:
            indices = list(range(len(all_pokemon)))
        else:
            k = random.randint(1, len(all_pokemon))
            indices = random.sample(range(len(all_pokemon)), k)

        for i in indices:
            all_pokemon[i].name = PokemonSet.MISSING_NAME

        x.lead = all_pokemon[0]
        x.reserve = all_pokemon[1:]
        return x, y

    def __repr__(self) -> str:
        return f"NamesOnlyMasker(mask_all={self.mask_all})"


class CurriculumMasker(TeamMasker):
    """Masker with curriculum: masking rate anneals from min to max over warmup steps."""

    def __init__(
        self,
        warmup_steps: int = 20_000,
        attrs_prob: float = 1.0,
        min_attrs_prob: float = 0.25,
        include_stats: bool = False,
    ):
        self.include_stats = include_stats
        self.warmup_steps = warmup_steps
        self._attrs_prob = attrs_prob
        self._min_attrs_prob = min_attrs_prob
        self._shared_step = mp.Value(ctypes.c_int, 0)

    def set_step(self, step: int) -> None:
        self._shared_step.value = step

    @property
    def _step(self) -> int:
        return self._shared_step.value

    @property
    def progress(self) -> float:
        return min(self._step / max(self.warmup_steps, 1), 1.0)

    @property
    def attrs_prob_range(self) -> Tuple[float, float]:
        current = self._min_attrs_prob + self.progress * (
            self._attrs_prob - self._min_attrs_prob
        )
        return (0.0, current)

    def __repr__(self) -> str:
        return (
            f"CurriculumMasker(step={self._step}/{self.warmup_steps}, "
            f"progress={self.progress:.1%}, attrs=[0,{self.attrs_prob_range[1]:.2f}])"
        )
