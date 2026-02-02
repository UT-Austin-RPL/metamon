import copy
import ctypes
import random
from typing import Tuple

import torch.multiprocessing as mp

from metamon.backend.team_prediction.team import TeamSet, PokemonSet


class TeamMasker:

    def __init__(
        self,
        pokemon_prob_range: Tuple[float, float] = (0.0, 0.15),
        attrs_prob_range: Tuple[float, float] = (0.1, 0.5),
        include_stats: bool = False,
    ):
        self.pokemon_prob_range = pokemon_prob_range
        self.attrs_prob_range = attrs_prob_range
        self.include_stats = include_stats

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
        num_to_mask = max(0, min(num_to_mask, len(maskable) - 1))

        if num_to_mask == 0:
            return PokemonSet.from_dict(data)

        for key, subkey in random.sample(maskable, num_to_mask):
            if key == "name":
                continue  # name (and all the rest of the details) masked via pokemon_prob, not here
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

    def mask(self, team: TeamSet) -> Tuple[TeamSet, TeamSet]:
        """Mask a team. Returns (masked_team, ground_truth_team)."""
        gen = team.gen
        y = copy.deepcopy(team)
        y.shuffle()
        x = copy.deepcopy(y)

        # it's impossible for the lead to be entirely masked
        x.lead = self._mask_pokemon(x.lead)

        # reserves (switches) have a chance to be completely masked.
        # otherwise, mask some fraction of the available attributes.
        pokemon_prob = random.uniform(*self.pokemon_prob_range)
        if gen >= 5:
            pokemon_prob = 0.0
        masked_reserve = []
        for p in x.reserve:
            if random.random() < pokemon_prob:
                masked_reserve.append(PokemonSet.missing_pokemon(gen=gen))
            else:
                masked_reserve.append(self._mask_pokemon(p))
        x.reserve = masked_reserve

        return x, y

    def __repr__(self) -> str:
        return (
            f"TeamMasker(pokemon={self.pokemon_prob_range}, "
            f"attrs={self.attrs_prob_range}, include_stats={self.include_stats})"
        )


class NamesOnlyMasker(TeamMasker):
    """Toy problem: only mask Pokemon names."""

    def __init__(self, mask_all: bool = True):
        super().__init__()
        self.mask_all = mask_all

    def mask(self, team: TeamSet) -> Tuple[TeamSet, TeamSet]:
        y = copy.deepcopy(team)
        y.shuffle()
        x = copy.deepcopy(y)

        all_pokemon = [x.lead] + list(x.reserve)
        if self.mask_all:
            indices_to_mask = list(range(len(all_pokemon)))
        else:
            k = random.randint(1, len(all_pokemon))
            indices_to_mask = random.sample(range(len(all_pokemon)), k)

        for i, p in enumerate(all_pokemon):
            if i in indices_to_mask:
                p.name = PokemonSet.MISSING_NAME

        x.lead = all_pokemon[0]
        x.reserve = all_pokemon[1:]
        return x, y

    def __repr__(self) -> str:
        return f"NamesOnlyMasker(mask_all={self.mask_all})"


class CurriculumMasker(TeamMasker):
    """TeamMasker with curriculum: ranges grow from 0 to target over warmup steps."""

    def __init__(
        self,
        warmup_steps: int = 20_000,
        pokemon_prob: float = 0.15,
        attrs_prob: float = 0.5,
        include_stats: bool = False,
    ):
        super().__init__(include_stats=include_stats)
        self.warmup_steps = warmup_steps
        self._pokemon_prob = pokemon_prob
        self._attrs_prob = attrs_prob
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
        # Range grows from [0,0] to [0, pokemon_prob]
        return (0.0, self.progress * self._pokemon_prob)

    @property
    def attrs_prob_range(self) -> Tuple[float, float]:
        # Range grows from [0,0] to [0, attrs_prob]
        return (0.0, self.progress * self._attrs_prob)

    def __repr__(self) -> str:
        return (
            f"CurriculumMasker(step={self._step}/{self.warmup_steps}, "
            f"progress={self.progress:.1%}, pokemon=[0,{self.pokemon_prob_range[1]:.2f}], "
            f"attrs=[0,{self.attrs_prob_range[1]:.2f}])"
        )
