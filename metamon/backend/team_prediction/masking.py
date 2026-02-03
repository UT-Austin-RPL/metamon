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

    - pokemon_prob_range: probability range for fully masking reserve Pokemon (Gen 1-4 only)
    - attrs_prob_range: probability range for masking individual attributes
    - name_only_prob: probability of creating "name visible, all attrs masked" examples
      (for coverage of intermediate states during iterative decoding)
    """

    def __init__(
        self,
        pokemon_prob_range: Tuple[float, float] = (0.0, 0.15),
        attrs_prob_range: Tuple[float, float] = (0.1, 0.5),
        include_stats: bool = False,
        name_only_prob: float = 0.05,
    ):
        self.pokemon_prob_range = pokemon_prob_range
        self.attrs_prob_range = attrs_prob_range
        self.include_stats = include_stats
        self.name_only_prob = name_only_prob

    def set_step(self, step: int) -> None:
        """Update training step (for curriculum subclasses)."""
        pass

    def _mask_pokemon(self, pokemon: PokemonSet) -> PokemonSet:
        """Mask a fraction of available attributes."""
        data = pokemon.to_dict()
        maskable = pokemon.get_maskable_attrs(include_stats=self.include_stats)

        if not maskable:
            return PokemonSet.from_dict(data)

        mask_prob = random.uniform(*self.attrs_prob_range)
        num_to_mask = int(mask_prob * len(maskable))
        num_to_mask = max(1, min(num_to_mask, len(maskable) - 1))

        for key, subkey in random.sample(maskable, num_to_mask):
            if key == "name":
                continue
            if subkey is None:
                if key == "ability":
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

    def _mask_attrs_only(self, pokemon: PokemonSet) -> PokemonSet:
        """
        Mask all attributes but keep the name visible.

        Creates "name-only" states that occur during iterative decoding
        after committing a Pokemon's name but before its attributes.
        """
        data = pokemon.to_dict()
        data["ability"] = PokemonSet.MISSING_ABILITY
        data["item"] = PokemonSet.MISSING_ITEM
        data["tera_type"] = PokemonSet.MISSING_TERA_TYPE
        data["moves"] = [PokemonSet.MISSING_MOVE] * 4
        if self.include_stats:
            data["nature"] = PokemonSet.MISSING_NATURE
            data["evs"] = [PokemonSet.MISSING_EV] * 6
            data["ivs"] = [PokemonSet.MISSING_IV] * 6
        return PokemonSet.from_dict(data)

    def mask(self, team: TeamSet) -> Tuple[TeamSet, TeamSet]:
        """Mask a team. Returns (masked_x, ground_truth_y)."""
        gen = team.gen
        y = copy.deepcopy(team)
        x = copy.deepcopy(team)

        # Lead Pokemon: standard masking or name-only
        if random.random() < self.name_only_prob:
            x.lead = self._mask_attrs_only(x.lead)
        else:
            x.lead = self._mask_pokemon(x.lead)

        pokemon_prob = random.uniform(*self.pokemon_prob_range)
        if gen >= 5:
            pokemon_prob = 0.0

        masked_reserve = []
        for p in x.reserve:
            if random.random() < pokemon_prob:
                # Gen 1-4: fully mask this Pokemon
                masked_reserve.append(PokemonSet.missing_pokemon(gen=gen))
            elif random.random() < self.name_only_prob:
                # Name-only: keep name visible, mask all attrs
                masked_reserve.append(self._mask_attrs_only(p))
            else:
                # Standard: mask some fraction of attrs
                masked_reserve.append(self._mask_pokemon(p))
        x.reserve = masked_reserve

        return x, y

    def __repr__(self) -> str:
        return (
            f"TeamMasker(pokemon={self.pokemon_prob_range}, "
            f"attrs={self.attrs_prob_range}, name_only={self.name_only_prob})"
        )


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
    """Masker with curriculum: masking rates anneal from min to max over warmup steps."""

    def __init__(
        self,
        warmup_steps: int = 20_000,
        pokemon_prob: float = 0.15,
        attrs_prob: float = 0.5,
        min_pokemon_prob: float = 0.0,
        min_attrs_prob: float = 0.1,
        include_stats: bool = False,
        name_only_prob: float = 0.05,
    ):
        self.include_stats = include_stats
        self.name_only_prob = name_only_prob
        self.warmup_steps = warmup_steps
        self._pokemon_prob = pokemon_prob
        self._attrs_prob = attrs_prob
        self._min_pokemon_prob = min_pokemon_prob
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
    def pokemon_prob_range(self) -> Tuple[float, float]:
        current = self._min_pokemon_prob + self.progress * (
            self._pokemon_prob - self._min_pokemon_prob
        )
        return (0.0, current)

    @property
    def attrs_prob_range(self) -> Tuple[float, float]:
        current = self._min_attrs_prob + self.progress * (
            self._attrs_prob - self._min_attrs_prob
        )
        return (0.0, current)

    def __repr__(self) -> str:
        return (
            f"CurriculumMasker(step={self._step}/{self.warmup_steps}, "
            f"progress={self.progress:.1%}, pokemon=[0,{self.pokemon_prob_range[1]:.2f}], "
            f"attrs=[0,{self.attrs_prob_range[1]:.2f}])"
        )
