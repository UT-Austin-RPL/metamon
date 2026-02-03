import copy
import ctypes
import random
from typing import Tuple, List

import torch.multiprocessing as mp

from metamon.backend.team_prediction.team import TeamSet, PokemonSet


def _pokemon_sort_key(p: PokemonSet) -> Tuple[int, str]:
    """Sort key: visible Pokemon first (alphabetically), then missing."""
    if p.name == PokemonSet.MISSING_NAME:
        return (1, "")  # Missing Pokemon sort last
    return (0, p.name)


def _move_sort_key(move: str) -> Tuple[int, str]:
    """Sort key: visible moves first (alphabetically), then missing, then <nomove>."""
    if move == PokemonSet.NO_MOVE:
        return (2, move)  # Empty slots last
    if move == PokemonSet.MISSING_MOVE:
        return (1, move)  # Missing moves second
    return (0, move)  # Real moves first, alphabetically


def _compute_ordering(items: List, sort_key) -> List[int]:
    """Compute permutation indices to sort items by key."""
    indexed = [(i, sort_key(item)) for i, item in enumerate(items)]
    indexed.sort(key=lambda x: x[1])
    return [i for i, _ in indexed]


def _apply_ordering(items: List, order: List[int]) -> List:
    """Apply permutation to reorder items."""
    return [items[i] for i in order]


class TeamMasker:
    """
    Base masker class handling masking and sequence generation.

    Sequence ordering: visible items first (alphabetically), then masked items.
    This ordering is determined by x (masked version) and applied to both x and y.
    """

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
        num_to_mask = max(1, min(num_to_mask, len(maskable) - 1))

        for key, subkey in random.sample(maskable, num_to_mask):
            if key == "name":
                continue  # name masked via pokemon_prob, not here
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
        x = copy.deepcopy(team)

        # Lead is never fully masked, just some attributes
        x.lead = self._mask_pokemon(x.lead)

        # Reserve Pokemon: some chance to be fully masked (Gen 1-4 only)
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

    def _pokemon_to_seq(self, pokemon: PokemonSet) -> Tuple[List[str], List[bool]]:
        """Convert a single Pokemon to sequence tokens (unordered moves)."""
        seq = [
            f"Mon: {pokemon.name}",
            f"Ability: {pokemon.ability}",
            f"Item: {pokemon.item}",
            f"Tera Type: {pokemon.tera_type}",
        ]
        seq += [f"Move: {move}" for move in pokemon.moves]

        if self.include_stats:
            seq.append(f"Nature: {pokemon.nature}")
            seq += [f"EVs: {ev}" for ev in pokemon.evs]
            seq += [f"IV: {iv}" for iv in pokemon.ivs]

        mask = [bool(pokemon.missing_regex.search(word)) for word in seq]
        return seq, mask

    def _pokemon_to_seq_ordered(
        self, x_pokemon: PokemonSet, y_pokemon: PokemonSet
    ) -> Tuple[List[str], List[str], List[bool]]:
        """
        Convert Pokemon pair to sequences with coordinated move ordering.
        Order determined by x's visible moves, applied to both.
        """
        # Compute move ordering based on x's visible state
        move_order = _compute_ordering(x_pokemon.moves, _move_sort_key)

        # Reorder moves in both x and y
        x_moves_ordered = _apply_ordering(x_pokemon.moves, move_order)
        y_moves_ordered = _apply_ordering(y_pokemon.moves, move_order)

        # Build sequences
        x_seq = [
            f"Mon: {x_pokemon.name}",
            f"Ability: {x_pokemon.ability}",
            f"Item: {x_pokemon.item}",
            f"Tera Type: {x_pokemon.tera_type}",
        ]
        x_seq += [f"Move: {move}" for move in x_moves_ordered]

        y_seq = [
            f"Mon: {y_pokemon.name}",
            f"Ability: {y_pokemon.ability}",
            f"Item: {y_pokemon.item}",
            f"Tera Type: {y_pokemon.tera_type}",
        ]
        y_seq += [f"Move: {move}" for move in y_moves_ordered]

        if self.include_stats:
            x_seq.append(f"Nature: {x_pokemon.nature}")
            x_seq += [f"EVs: {ev}" for ev in x_pokemon.evs]
            x_seq += [f"IV: {iv}" for iv in x_pokemon.ivs]

            y_seq.append(f"Nature: {y_pokemon.nature}")
            y_seq += [f"EVs: {ev}" for ev in y_pokemon.evs]
            y_seq += [f"IV: {iv}" for iv in y_pokemon.ivs]

        # Prediction mask: where x is missing but y is not
        x_mask = [bool(x_pokemon.missing_regex.search(word)) for word in x_seq]
        y_mask = [bool(y_pokemon.missing_regex.search(word)) for word in y_seq]
        pred_mask = [xm and not ym for xm, ym in zip(x_mask, y_mask)]

        return x_seq, y_seq, pred_mask

    def to_seq(self, team: TeamSet) -> Tuple[List[str], List[bool]]:
        """
        Convert a single team to sequence format (for inference).
        Orders: visible Pokemon first (alphabetically), then missing.
        Within each Pokemon: visible moves first, then missing, then <nomove>.

        Returns (sequence, needs_prediction_mask).
        """
        all_pokemon = [team.lead] + list(team.reserve)

        # Order Pokemon: lead stays first, reserve sorted by visibility
        reserve_order = _compute_ordering(team.reserve, _pokemon_sort_key)
        ordered_pokemon = [team.lead] + _apply_ordering(team.reserve, reserve_order)

        # Build sequence
        seq = [f"Format: {team.format}"]
        mask = [False]

        for pokemon in ordered_pokemon:
            # Order moves within this Pokemon
            move_order = _compute_ordering(pokemon.moves, _move_sort_key)
            ordered_moves = _apply_ordering(pokemon.moves, move_order)

            p_seq = [
                f"Mon: {pokemon.name}",
                f"Ability: {pokemon.ability}",
                f"Item: {pokemon.item}",
                f"Tera Type: {pokemon.tera_type}",
            ]
            p_seq += [f"Move: {move}" for move in ordered_moves]

            if self.include_stats:
                p_seq.append(f"Nature: {pokemon.nature}")
                p_seq += [f"EVs: {ev}" for ev in pokemon.evs]
                p_seq += [f"IV: {iv}" for iv in pokemon.ivs]

            p_mask = [bool(pokemon.missing_regex.search(word)) for word in p_seq]
            seq.extend(p_seq)
            mask.extend(p_mask)

        return seq, mask

    def to_seq_pair(
        self, x: TeamSet, y: TeamSet
    ) -> Tuple[List[str], List[str], List[bool]]:
        """
        Convert (x, y) team pair to sequences with coordinated ordering.

        Ordering is determined by x's visible state:
        - Pokemon: lead first, then reserve with visible names (alphabetically), then fully masked
        - Moves: visible first (alphabetically), then missing, then <nomove>

        Returns (x_seq, y_seq, pred_mask).
        """
        # Order reserve Pokemon based on x's visible names
        reserve_order = _compute_ordering(x.reserve, _pokemon_sort_key)

        x_reserve_ordered = _apply_ordering(x.reserve, reserve_order)
        y_reserve_ordered = _apply_ordering(y.reserve, reserve_order)

        x_all = [x.lead] + x_reserve_ordered
        y_all = [y.lead] + y_reserve_ordered

        # Build sequences
        x_seq = [f"Format: {x.format}"]
        y_seq = [f"Format: {y.format}"]
        pred_mask = [False]

        for x_pokemon, y_pokemon in zip(x_all, y_all):
            px_seq, py_seq, p_mask = self._pokemon_to_seq_ordered(x_pokemon, y_pokemon)
            x_seq.extend(px_seq)
            y_seq.extend(py_seq)
            pred_mask.extend(p_mask)

        return x_seq, y_seq, pred_mask

    def mask_to_seq(self, team: TeamSet) -> Tuple[List[str], List[str], List[bool]]:
        """
        Convenience method: mask team and convert to sequences in one call.
        Returns (x_seq, y_seq, pred_mask).
        """
        x, y = self.mask(team)
        return self.to_seq_pair(x, y)

    def __repr__(self) -> str:
        return (
            f"TeamMasker(pokemon={self.pokemon_prob_range}, "
            f"attrs={self.attrs_prob_range}, include_stats={self.include_stats})"
        )


class NoOpMasker(TeamMasker):
    """
    No-op masker for inference.

    Does not mask anything, but still provides the standard sequence ordering
    (visible items first, missing at end) for consistency with training.
    """

    def __init__(self, include_stats: bool = False):
        super().__init__(
            pokemon_prob_range=(0.0, 0.0),
            attrs_prob_range=(0.0, 0.0),
            include_stats=include_stats,
        )

    def mask(self, team: TeamSet) -> Tuple[TeamSet, TeamSet]:
        """Identity: return team unchanged (but deep copied)."""
        t = copy.deepcopy(team)
        return t, t

    def __repr__(self) -> str:
        return f"NoOpMasker(include_stats={self.include_stats})"


class NamesOnlyMasker(TeamMasker):
    """Toy problem: only mask Pokemon names."""

    def __init__(self, mask_all: bool = True, include_stats: bool = False):
        super().__init__(include_stats=include_stats)
        self.mask_all = mask_all

    def mask(self, team: TeamSet) -> Tuple[TeamSet, TeamSet]:
        y = copy.deepcopy(team)
        x = copy.deepcopy(team)

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
    """TeamMasker with curriculum: ranges grow from min to target over warmup steps."""

    def __init__(
        self,
        warmup_steps: int = 20_000,
        pokemon_prob: float = 0.15,
        attrs_prob: float = 0.5,
        min_pokemon_prob: float = 0.0,
        min_attrs_prob: float = 0.1,
        include_stats: bool = False,
    ):
        self.include_stats = include_stats
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
