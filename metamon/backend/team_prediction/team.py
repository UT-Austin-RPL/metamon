import random
import warnings
import re
import os
import copy
from datetime import date
import functools
from dataclasses import dataclass
from typing import List, Optional, Tuple

import metamon
from metamon.backend.replay_parser.replay_state import (
    Pokemon,
    Nothing,
    unknown,
)
from metamon.backend.replay_parser.str_parsing import pokemon_name
from metamon.backend.team_prediction.usage_stats import get_usage_stats
from metamon.backend.showdown_dex import Dex


def moveset_size(pokemon_name: str, gen: int) -> int:
    # attempts to handle cases where we would expect a Pokemon to have less than 4 moves
    stat = get_usage_stats(f"gen{gen}ubers")
    try:
        moves = len(set(stat[pokemon_name]["moves"].keys()) - {"Nothing"})
    except KeyError:
        return 4
    moveset = min(moves, 4)
    return moveset


def _one_hidden_power(move_name: str) -> str:
    # used to map all hidden power moves to the same name
    if move_name.startswith("Hidden Power"):
        return "Hidden Power"
    elif move_name.startswith("hiddenpower"):
        return "hiddenpower"
    else:
        return move_name


@functools.total_ordering
@dataclass
class PokemonSet:
    """
    Represents a Pokemon's moveset, ability, item, nature, and EVs/IVs during team prediction.

    Defines a useful ordering between Pokemon where PokemonSet A < PokemonSet B if B
    is a superset of the information revealed in A.
    """

    name: str
    gen: int
    moves: List[str]
    ability: str
    item: str
    nature: str
    evs: List[int]
    ivs: List[int]
    tera_type: str

    NO_MOVE = "<nomove>"
    NO_ABILITY = "<noability>"
    NO_ITEM = "<noitem>"
    NO_NATURE = "<nonature>"
    NO_TERA_TYPE = "<notera>"

    MISSING_NAME = "$missing_name$"
    MISSING_MOVE = "$missing_move$"
    MISSING_ABILITY = "$missing_ability$"
    MISSING_ITEM = "$missing_item$"
    MISSING_EV = "$missing_ev$"
    MISSING_IV = "$missing_iv$"
    MISSING_NATURE = "$missing_nature$"
    MISSING_TERA_TYPE = "$missing_tera$"

    @classmethod
    def get_teamfile_name(cls, given_name: str, gen: int) -> tuple[str, str]:
        if given_name == cls.MISSING_NAME:
            return given_name, given_name
        dex = Dex.from_gen(gen)
        try:
            entry = dex.get_pokedex_entry(given_name)
        except KeyError:
            return given_name, given_name
        name = entry.get("name", given_name)
        if "battleOnly" in entry:
            name = entry["battleOnly"]
        base_species = entry.get("baseSpecies", name)
        return name, base_species

    def __post_init__(self):
        assert len(self.evs) == 6
        assert len(self.ivs) == 6
        assert self.nature is not None
        assert self.item is not None
        assert self.ability is not None
        self.missing_strings = [
            self.MISSING_NAME,
            self.MISSING_MOVE,
            self.MISSING_ABILITY,
            self.MISSING_ITEM,
            self.MISSING_NATURE,
            self.MISSING_TERA_TYPE,
        ]
        self.missing_regex = re.compile("|".join(map(re.escape, self.missing_strings)))
        self.moves = [_one_hidden_power(move) for move in self.moves]
        # override names with official pokedex lookup
        self.name, self.base_species = self.get_teamfile_name(self.name, self.gen)

    def __hash__(self):
        moves_frozen = frozenset(self.moves) - {self.MISSING_MOVE}
        evs_tuple = tuple(self.evs)
        ivs_tuple = tuple(self.ivs)
        return hash(
            (
                self.name,
                self.gen,
                moves_frozen,
                self.ability,
                self.item,
                self.nature,
                evs_tuple,
                ivs_tuple,
                self.tera_type,
            )
        )

    def __len__(self):
        return self.revealed_details

    def additional_details(self, other) -> Optional[dict]:
        """
        Returns a dictionary of the details in `other` that are not in `self`,
        a.k.a. newly revealed details during team prediction.

        If the two sets are not consistent, returns None.
        """
        if not isinstance(other, PokemonSet):
            raise ValueError("other must be a PokemonSet")
        if not self.is_consistent_with(other):
            return None
        other = other.to_dict()
        current = self.to_dict()
        diff = {}
        for key, value in other.items():
            if key == "moves":
                other_moves = set(value) - {self.MISSING_MOVE, self.NO_MOVE}
                our_moves = set(current["moves"]) - {self.MISSING_MOVE, self.NO_MOVE}
                diff_moves = other_moves - our_moves
                if diff_moves:
                    diff["moves"] = diff_moves
            elif current[key] != value:
                diff[key] = value
        return diff

    @property
    def revealed_details(self) -> int:
        """
        Counts the number of details revealed in this PokemonSet.

        This is used to skip comparisons between PokemonSets, because
        A cannot be < B if A has more revealed details than B.
        """
        score = (
            int(self.name != self.MISSING_NAME)
            + int(self.ability != self.MISSING_ABILITY)
            + int(self.item != self.MISSING_ITEM)
            + int(self.nature != self.MISSING_NATURE)
            + sum(int(move != self.MISSING_MOVE) for move in self.moves)
            + sum(int(ev != self.MISSING_EV) for ev in self.evs)
            + sum(int(iv != self.MISSING_IV) for iv in self.ivs)
            + int(self.tera_type != self.MISSING_TERA_TYPE)
        )
        return score

    @property
    def revealed_moves(self) -> int:
        """Counts the number of moves revealed in this PokemonSet."""
        return len(set(self.moves) - {self.MISSING_MOVE})

    @property
    def set_key(self) -> tuple:
        """Canonical (species, moves, item, ability) key — the competitive definition of a "set".

        Two PokemonSets with the same species, moveset (unordered), item, and
        ability are considered the same set, regardless of nature, EVs, IVs,
        or tera type.  Sentinel / missing values are stripped before comparison.
        """
        _skip = {self.MISSING_MOVE, self.NO_MOVE}
        moves = frozenset(m for m in self.moves if m not in _skip)
        return (self.name, moves, self.item, self.ability)

    @classmethod
    def max_relevant_attrs(cls, gen: int, include_stats: bool = False) -> int:
        """
        Get the maximum number of relevant attributes per Pokemon for a generation.
        """
        count = 0
        if gen <= 4:
            # name is only relevant in Gen 1-4 (no team preview)
            count += 1
        count += 4  # 4 moves always
        if gen >= 2:
            count += 1  # item
        if gen >= 3:
            count += 1  # ability
        if gen == 9:
            count += 1  # tera_type
        if include_stats:
            count += 1  # nature
            count += 6  # evs
            count += 6  # ivs
        return count

    def revealed_relevant_attrs(self, include_stats: bool = False) -> int:
        """
        Count the number of revealed attributes that are "relevant" for this Pokemon.
        """
        count = 0
        # name is only relevant in Gen 1-4 (no team preview)
        if self.gen <= 4 and self.name != self.MISSING_NAME:
            count += 1
        # ability (gen 3+)
        if self.gen >= 3 and self.ability != self.MISSING_ABILITY:
            count += 1
        # item (gen 2+)
        if self.gen >= 2 and self.item != self.MISSING_ITEM:
            count += 1
        # tera_type (gen 9 only)
        if self.gen == 9 and self.tera_type != self.MISSING_TERA_TYPE:
            count += 1
        # moves (always, up to 4)
        for move in self.moves:
            if move != self.MISSING_MOVE and move != self.NO_MOVE:
                count += 1
        if include_stats:
            if self.nature != self.MISSING_NATURE:
                count += 1
            for ev in self.evs:
                if ev != self.MISSING_EV:
                    count += 1
            for iv in self.ivs:
                if iv != self.MISSING_IV:
                    count += 1
        return count

    def revealed_score(self, include_stats: bool = False) -> float:
        """
        Compute the revealed score for this Pokemon.

        Score = revealed_relevant_attrs / max_relevant_attrs
        """
        max_attrs = PokemonSet.max_relevant_attrs(self.gen, include_stats)
        if max_attrs == 0:
            return 0.0
        return self.revealed_relevant_attrs(include_stats) / max_attrs

    @property
    def is_present(self) -> bool:
        """Check if this Pokemon is actually present (not a placeholder for unknown mon)."""
        return self.name != self.MISSING_NAME

    def get_maskable_attrs(self, include_stats: bool = False) -> list:
        """
        Get list of (key, subkey) tuples for revealed attributes that can be masked.
        Respects generation constraints (no abilities in gen1-2, no tera in gen1-8, etc.)
        """
        maskable = []

        if self.name != self.MISSING_NAME:
            maskable.append(("name", None))
        if self.gen >= 3 and self.ability != self.MISSING_ABILITY:
            maskable.append(("ability", None))
        if self.gen >= 2 and self.item != self.MISSING_ITEM:
            maskable.append(("item", None))
        if self.gen == 9 and self.tera_type != self.MISSING_TERA_TYPE:
            maskable.append(("tera_type", None))

        for i, move in enumerate(self.moves):
            if move != self.MISSING_MOVE and move != self.NO_MOVE:
                maskable.append(("moves", i))

        if include_stats:
            for i, ev in enumerate(self.evs):
                if ev != self.MISSING_EV:
                    maskable.append(("evs", i))
            for i, iv in enumerate(self.ivs):
                if iv != self.MISSING_IV:
                    maskable.append(("ivs", i))

        return maskable

    def __eq__(self, other):
        if not isinstance(other, PokemonSet):
            return False
        possible = (
            self.name == other.name
            and self.ability == other.ability
            and self.item == other.item
            and self.nature == other.nature
            and self.evs == other.evs
            and self.ivs == other.ivs
            and self.gen == other.gen
            and self.tera_type == other.tera_type
        )
        if possible and (set(self.moves) - {self.MISSING_MOVE}) == (
            set(other.moves) - {other.MISSING_MOVE}
        ):
            return True
        return False

    def __lt__(self, other):
        return self.is_consistent_with(other) and self != other

    def is_consistent_with(self, other) -> bool:
        """
        Determines whether this Pokemon is "consistent" with another Pokemon,
        where "consistent" means that there is no information we know about this
        Pokemon that is contradicted by the other. For example, the partial version
        of a Pokemon revealed by a replay would be consistent with a correct prediction
        of the rest of the set.
        """
        if self.name != other.name:
            return False
        if self.gen != other.gen:
            return False
        if self.ability != self.MISSING_ABILITY and self.ability != other.ability:
            return False
        if self.item != self.MISSING_ITEM and self.item != other.item:
            return False
        if self.nature != self.MISSING_NATURE and self.nature != other.nature:
            return False
        if (
            self.tera_type != self.MISSING_TERA_TYPE
            and self.tera_type != other.tera_type
        ):
            return False
        for our_move in self.moves:
            if our_move != self.MISSING_MOVE and our_move not in other.moves:
                return False
        for our_ev, other_ev in zip(self.evs, other.evs):
            if our_ev != self.MISSING_EV and our_ev != other_ev:
                return False
        for our_iv, other_iv in zip(self.ivs, other.ivs):
            if our_iv != self.MISSING_IV and our_iv != other_iv:
                return False
        return True

    @classmethod
    def default_moves(cls, name: str, gen: int):
        if name == cls.MISSING_NAME:
            return [cls.MISSING_MOVE] * 4
        m = moveset_size(name, gen)
        # we are trying to catch cases where a full moveset would actually be less than 4 moves
        return [cls.MISSING_MOVE] * m + [cls.NO_MOVE] * (4 - m)

    @classmethod
    def default_ivs(cls, gen: int):
        # mirroring Showdown logic where IVs are assumed to be 31
        return [31] * 6

    @classmethod
    def default_evs(cls, gen: int):
        return [252] * 6 if gen <= 2 else [cls.MISSING_EV] * 6

    @classmethod
    def default_nature(cls, gen: int):
        # the trick in gens before nature is for the backend to pick a neutral nature.
        # here we set to NO_NATURE and then ignore it in the output string.
        return cls.NO_NATURE if gen <= 2 else cls.MISSING_NATURE

    @classmethod
    def default_item(cls, gen: int):
        return cls.NO_ITEM if gen <= 1 else cls.MISSING_ITEM

    @classmethod
    def default_ability(cls, gen: int):
        return cls.NO_ABILITY if gen <= 2 else cls.MISSING_ABILITY

    @classmethod
    def default_tera_type(cls, gen: int):
        return cls.NO_TERA_TYPE if gen != 9 else cls.MISSING_TERA_TYPE

    @classmethod
    def from_ReplayPokemon(cls, pokemon: Optional[Pokemon], gen: int):
        """
        Used to convert between the Pokemon we are filling in the replay parser
        and this PokemonSet format used for team prediction.
        """

        if pokemon is None:
            return cls.missing_pokemon(gen=gen)
        moves = [m.name for m in pokemon.had_moves.values()]
        while len(moves) < moveset_size(pokemon.name, pokemon.gen):
            moves.append(cls.MISSING_MOVE)

        # maintaining the replay parser's distinction between "known to be None" and "unrevealed"
        if pokemon.gen == 1 or pokemon.had_item == Nothing.NO_ITEM:
            item = cls.NO_ITEM
        elif unknown(pokemon.had_item):
            item = cls.MISSING_ITEM
        else:
            item = pokemon.had_item
        if pokemon.gen <= 2 or pokemon.had_ability == Nothing.NO_ABILITY:
            ability = cls.NO_ABILITY
        elif unknown(pokemon.had_ability):
            ability = cls.MISSING_ABILITY
        else:
            ability = pokemon.had_ability
        if pokemon.gen == 9 and pokemon.tera_type is not None:
            tera_type = pokemon.tera_type
        else:
            tera_type = cls.default_tera_type(gen)

        return cls(
            name=pokemon.name,
            gen=pokemon.gen,
            moves=moves,
            ability=ability,
            item=item,
            nature=cls.default_nature(gen),
            evs=cls.default_evs(gen),
            ivs=cls.default_ivs(gen),
            tera_type=tera_type,
        )

    def fill_from_PokemonSet(self, other):
        """
        Used to merge the results of a team prediction into existing team info.
        """
        if not isinstance(other, PokemonSet):
            raise ValueError("other must be a PokemonSet")
        if not pokemon_name(self.name) == pokemon_name(other.name):
            raise ValueError("other must have the same name")
        if self.base_species != self.MISSING_NAME and pokemon_name(
            self.base_species
        ) != pokemon_name(other.base_species):
            raise ValueError("other must have the same base species")
        self.base_species = other.base_species
        new_moves = list(set(other.moves) - set(self.moves))
        for move in self.moves:
            if move == self.MISSING_MOVE:
                if new_moves:
                    new_move = new_moves.pop()
                    self.moves[self.moves.index(move)] = new_move
        if self.ability == self.MISSING_ABILITY:
            self.ability = other.ability
        if self.item == self.MISSING_ITEM:
            self.item = other.item
        if self.nature == self.MISSING_NATURE:
            self.nature = other.nature
        if self.tera_type == self.MISSING_TERA_TYPE:
            self.tera_type = other.tera_type
        for idx, ev in enumerate(self.evs):
            if ev == self.MISSING_EV:
                self.evs[idx] = other.evs[idx]
        for idx, iv in enumerate(self.ivs):
            if iv == self.MISSING_IV:
                self.ivs[idx] = other.ivs[idx]

    def to_str(self):
        """
        Outputs the poke-paste-style string for this PokemonSet.
        """
        evs = "EVs: "
        for desc, ev_val in zip(["HP", "Atk", "Def", "SpA", "SpD", "Spe"], self.evs):
            evs += f"{ev_val} {desc}"
            if desc != "Spe":
                evs += " / "
        if self.nature != self.NO_NATURE:
            evs += f"\n{self.nature} Nature"
        ivs = "IVs: "
        for desc, iv_val in zip(["HP", "Atk", "Def", "SpA", "SpD", "Spe"], self.ivs):
            ivs += f"{iv_val} {desc}"
            if desc != "Spe":
                ivs += " / "

        start = f"{self.name}"
        if self.item != self.NO_ITEM:
            start += f" @ {self.item}"
        if self.tera_type != self.NO_TERA_TYPE:
            start += f"\nTera Type: {self.tera_type}"
        moves = "\n".join([f"- {move}" for move in self.moves])
        ability_str = self.ability if self.ability != self.NO_ABILITY else "No Ability"
        return start + f"\nAbility: {ability_str}\n{evs}\n{ivs}\n{moves}"

    @classmethod
    def from_showdown_block(cls, block: str, gen: int):
        """
        Creates a PokemonSet from a poke-paste string.
        """
        block = block.replace("\u200b", "")
        lines = [line.strip() for line in block.strip().split("\n") if line.strip()]

        # Parse first line: handle nickname, gender, and item
        first = lines[0]
        if "@" in first:
            name_part, item_part = first.split("@", 1)
            item = item_part.strip() or cls.default_item(gen)
        else:
            name_part = first
            item = cls.default_item(gen)
        name_raw = name_part.strip()
        # Extract parenthetical contents and pick the species (ignore gender flags)
        contents = re.findall(r"\(([^)]+)\)", name_raw)
        species = None
        for content in contents:
            c = content.strip()
            if c.upper() in ("M", "F"):
                continue
            species = c
            break
        if species:
            name = species
        else:
            # Remove any parentheses (nicknames or gender)
            name = re.sub(r"\s*\([^)]*\)", "", name_raw).strip()

        # Set defaults based on gen
        evs = cls.default_evs(gen)
        ivs = cls.default_ivs(gen)
        nature = cls.default_nature(gen)
        ability = cls.default_ability(gen)
        moves = cls.default_moves(name, gen)
        tera_type = cls.default_tera_type(gen)

        for line in lines[1:]:
            if line.startswith("Ability:"):
                if gen > 2:
                    ability = line.split(":", 1)[1].strip()
                    if ability == "No Ability":
                        ability = cls.NO_ABILITY
            elif line.startswith("EVs:"):
                evs = [0] * 6 if gen > 2 else [252] * 6
                for part in line[4:].split("/"):
                    stat = part.strip().split(" ")
                    if len(stat) == 2:
                        val, stat_name = stat
                        idx = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"].index(stat_name)
                        if val != cls.MISSING_EV:
                            evs[idx] = int(val)
                        else:
                            evs[idx] = cls.MISSING_EV
            elif line.startswith("IVs:"):
                for part in line[4:].split("/"):
                    stat = part.strip().split(" ")
                    if len(stat) == 2:
                        val, stat_name = stat
                        idx = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"].index(stat_name)
                        if val != cls.MISSING_IV:
                            ivs[idx] = int(val)
                        else:
                            ivs[idx] = cls.MISSING_IV
            elif line.endswith("Nature"):
                if gen >= 3:
                    nature = line.split()[0].strip()
            elif line.startswith("Tera Type:"):
                if gen == 9:
                    tera_type = line.split(":")[1].strip()
            elif line.startswith("- "):
                move_raw = line[2:].strip()
                # if multiple options, take the first option
                if "/" in move_raw:
                    move_raw = move_raw.split("/", 1)[0].strip()

                if cls.MISSING_MOVE in moves:
                    moves[moves.index(cls.MISSING_MOVE)] = move_raw
                elif len(moves) < 4:
                    moves.append(move_raw)
                else:
                    raise ValueError(f"Team has too many moves: {moves}")

        return cls(
            name=name,
            gen=gen,
            moves=moves,
            ability=ability,
            item=item,
            evs=evs,
            ivs=ivs,
            nature=nature,
            tera_type=tera_type,
        )

    def to_dict(self):
        return {
            "name": self.name,
            "moves": self.moves,
            "gen": self.gen,
            "ability": self.ability,
            "item": self.item,
            "evs": self.evs,
            "ivs": self.ivs,
            "nature": self.nature,
            "tera_type": self.tera_type,
        }

    @classmethod
    def from_dict(cls, d: dict):
        # backwards compatibility with existing
        # replay set dicts before gen9 update
        if "tera_type" not in d:
            warnings.warn(
                "tera_type not found in PokemonSet.from_dict. if gen9, this is a bug."
            )
            d["tera_type"] = cls.NO_TERA_TYPE

        return cls(
            name=d["name"],
            gen=d["gen"],
            ability=d["ability"],
            item=d["item"],
            nature=d["nature"],
            moves=d["moves"],
            evs=d["evs"],
            ivs=d["ivs"],
            tera_type=d["tera_type"],
        )

    @classmethod
    def from_seq(cls, seq: List[str], gen: int, include_stats: bool = True):
        """
        Creates a PokemonSet from the sequence format, which may have been predicted by a model.
        """
        name = seq[0].split(":")[1].strip()
        ability = seq[1].split(":")[1].strip()
        item = seq[2].split(":")[1].strip()
        tera_type = seq[3].split(":")[1].strip()
        moves = [move.split(":")[1].strip() for move in seq[4:8]]
        if include_stats:
            nature = seq[8].split(":")[1].strip()
            evs = [ev.split(":")[1].strip() for ev in seq[9:15]]
            ivs = [iv.split(":")[1].strip() for iv in seq[15:21]]
        else:
            nature = cls.default_nature(gen)
            evs = cls.default_evs(gen)
            ivs = cls.default_ivs(gen)
        return cls(
            name=name,
            gen=gen,
            ability=ability,
            item=item,
            nature=nature,
            moves=moves,
            evs=evs,
            ivs=ivs,
            tera_type=tera_type,
        )

    @classmethod
    def missing_pokemon(cls, gen: int):
        return cls(
            name=PokemonSet.MISSING_NAME,
            gen=gen,
            ability=PokemonSet.default_ability(gen),
            item=PokemonSet.default_item(gen),
            nature=PokemonSet.default_nature(gen),
            moves=[PokemonSet.MISSING_MOVE] * 4,
            evs=PokemonSet.default_evs(gen),
            ivs=PokemonSet.default_ivs(gen),
            tera_type=PokemonSet.default_tera_type(gen),
        )

    def masked(self, mask_attrs_prob: float = 0.1):
        """
        Randomly sets some of the known attributes of this PokemonSet to be missing,
        so that we may learn to predict them.
        """
        data = self.to_dict()
        data["name"] = self.name
        # Mask nature, item, ability
        if random.random() < mask_attrs_prob:
            data["nature"] = self.MISSING_NATURE
        if random.random() < mask_attrs_prob:
            data["item"] = self.MISSING_ITEM
        if random.random() < mask_attrs_prob:
            data["ability"] = self.MISSING_ABILITY
        if random.random() < mask_attrs_prob:
            data["tera_type"] = self.MISSING_TERA_TYPE
        # Mask moves
        masked_moves = []
        for move in data["moves"]:
            if random.random() < mask_attrs_prob:
                masked_moves.append(self.MISSING_MOVE)
            else:
                masked_moves.append(move)
        data["moves"] = masked_moves
        # Mask EVs/IVs --- it only makes sense to do this as all-or-nothing
        if random.random() < mask_attrs_prob:
            data["evs"] = [self.MISSING_EV] * 6
        if random.random() < mask_attrs_prob:
            data["ivs"] = [self.MISSING_IV] * 6
        return type(self).from_dict(data)


@functools.total_ordering
@dataclass
class Roster:
    """
    A simplified version of a Team that only tracks Pokemon names.

    Used for simple two-step prediction strategy of Step 1) predict team, Step 2) predict movesets for each member of that team.
    """

    lead: str
    reserve: frozenset[str]

    def to_dict(self):
        return {
            "lead": self.lead,
            "reserve": list(self.reserve),
        }

    @classmethod
    def from_dict(cls, d: dict):
        return cls(lead=d["lead"], reserve=frozenset(d["reserve"]))

    @property
    def known_pokemon(self):
        return set([self.lead] + list(self.reserve)) - {PokemonSet.MISSING_NAME}

    def __len__(self):
        return int(self.lead != PokemonSet.MISSING_NAME) + sum(
            r != PokemonSet.MISSING_NAME for r in self.reserve
        )

    def __hash__(self):
        return hash((self.lead, self.reserve))

    def __eq__(self, other):
        return self.lead == other.lead and self.reserve == other.reserve

    def is_consistent_with(self, other) -> bool:
        if self.lead != other.lead:
            return False
        for our_pokemon in self.reserve:
            if our_pokemon == PokemonSet.MISSING_NAME:
                continue
            elif our_pokemon not in other.reserve:
                return False
        return True

    def additional_details(self, other) -> Optional[frozenset[str]]:
        """
        Returns a set of the Pokemon names in `other` that are not in `self`,
        a.k.a. newly revealed details during team prediction.

        If the two sets are not consistent, returns None.
        """
        if not isinstance(other, Roster):
            raise ValueError("other must be a Roster")
        if not self.is_consistent_with(other):
            return None
        return other.known_pokemon - self.known_pokemon

    def __lt__(self, other):
        return self.is_consistent_with(other) and self != other


@dataclass
class TeamSet:
    """
    Represents an entire team during team prediction.

    Mostly splits the functionality into something that needs to be done for each
    Pokemon on the team and calls the PokemonSet version.
    """

    lead: PokemonSet
    reserve: List[PokemonSet]
    format: str

    @property
    def gen(self) -> int:
        return metamon.backend.format_to_gen(self.format)

    @property
    def known_pokemon(self) -> List[PokemonSet]:
        return [p for p in self.pokemon if p.name != PokemonSet.MISSING_NAME]

    def revealed_score(self, include_stats: bool = False) -> float:
        """
        Compute the revealed score for the entire team.

        Score = (total revealed relevant attrs) / (total possible relevant attrs)
        """
        gen = self.gen
        all_pokemon = [self.lead] + list(self.reserve)

        total_revealed = 0
        total_possible = 0

        for pokemon in all_pokemon:
            total_possible += PokemonSet.max_relevant_attrs(gen, include_stats)
            total_revealed += pokemon.revealed_relevant_attrs(include_stats)

        if total_possible == 0:
            return 0.0

        return total_revealed / total_possible

    def __eq__(self, other):
        """
        Note that in gen1-4 the leads need to match, but the rest of the roster can be in any order
        and we'd still consider the teams equal.
        """
        return (
            self.format == other.format
            and self.lead == other.lead
            and set(self.reserve) == set(other.reserve)
        )

    def is_consistent_with(self, other) -> bool:
        """
        Determines whether this team is "consistent" with another team,
        where "consistent" means that there is no information we know about this
        team that is contradicted by the other. For example, the partial version
        of a team revealed by a replay would be consistent with a correct prediction
        of the rest of the team.
        """
        if self.format != other.format:
            return False
        if not self.lead.is_consistent_with(other.lead):
            return False

        # check rest of pokemon where order doesn't matter
        our_names = {p.name for p in self.reserve if p.name != PokemonSet.MISSING_NAME}
        other_names = {p.name for p in other.reserve}
        if not our_names.issubset(other_names):
            return False
        other_dict = {p.name: p for p in other.reserve}
        for our_pokemon in self.reserve:
            if our_pokemon.name == PokemonSet.MISSING_NAME:
                continue
            if not our_pokemon.is_consistent_with(other_dict[our_pokemon.name]):
                return False
        return True

    @property
    def pokemon(self):
        return [self.lead] + self.reserve

    def to_str(self):
        """
        Outputs the poke-paste-style string.
        """
        out = f"{self.lead.to_str()}"
        for p in self.reserve:
            out += f"\n\n{p.to_str()}"
        return out

    @classmethod
    def from_showdown_file(cls, path: str, format: str):
        """
        Creates a TeamSet from a showdown file.
        """
        with open(path, "r") as f:
            content = f.read()
        gen = metamon.backend.format_to_gen(format)
        blocks = [block for block in content.strip().split("\n\n") if block.strip()]
        pokemons = [PokemonSet.from_showdown_block(block, gen=gen) for block in blocks]
        if not pokemons:
            raise ValueError("No Pokémon found in file.")
        lead = pokemons[0]
        reserve = pokemons[1:]
        return cls(lead=lead, reserve=reserve, format=format)

    def write_to_file(self, path: str):
        with open(path, "w") as f:
            f.write(self.to_str())

    def to_dict(self):
        out = {self.lead.name: self.lead.to_dict()}
        for p in self.reserve:
            out[p.name] = p.to_dict()
        return out

    @classmethod
    def from_seq(cls, seq: List[str], include_stats: bool = True):
        format = seq[0].split(":")[1].strip()
        gen = metamon.backend.format_to_gen(format)
        poke_seq_len = 21 if include_stats else 8
        lead = PokemonSet.from_seq(
            seq[1 : poke_seq_len + 1], gen=gen, include_stats=include_stats
        )
        idx = poke_seq_len + 1
        reserve = []
        while idx < len(seq):
            reserve.append(
                PokemonSet.from_seq(
                    seq[idx : idx + poke_seq_len], gen=gen, include_stats=include_stats
                )
            )
            idx += poke_seq_len
        return cls(lead=lead, reserve=reserve, format=format)

    def fill_from_Roster(self, roster: Roster):
        """
        Fill in missing Pokemon names from a Roster.
        """
        if (
            self.lead.name == PokemonSet.MISSING_NAME
            and roster.lead != PokemonSet.MISSING_NAME
        ):
            self.lead.name = roster.lead
        if any(p.name == PokemonSet.MISSING_NAME for p in self.reserve):
            roster_pokemon = set(roster.reserve)
            current_pokemon = set(p.name for p in self.reserve) - {
                PokemonSet.MISSING_NAME
            }
            new_pokemon = roster_pokemon - current_pokemon
            for pokemon in self.pokemon:
                if pokemon.name == PokemonSet.MISSING_NAME and new_pokemon:
                    pokemon.name = new_pokemon.pop()


def _pokemon_sort_key(x_p: PokemonSet, y_p: PokemonSet) -> Tuple[int, int, str]:
    """
    Sort key: visible first, then masked-with-label, then masked-no-label.

    Order:
    1. Visible Pokemon (alphabetically)
    2. Masked Pokemon with labels (alphabetically by y's name)
    3. Masked Pokemon without labels (y is also $missing_name$)
    """
    if x_p.name == PokemonSet.MISSING_NAME:
        # If y is also missing, no ground truth - sort after labeled positions
        if y_p.name == PokemonSet.MISSING_NAME:
            return (2, 0, y_p.name)
        # Masked with label - sort alphabetically by y
        return (1, 0, y_p.name)
    return (0, 0, x_p.name)


def _move_sort_key(x_move: str, y_move: str) -> Tuple[int, int, str]:
    """
    Sort key: visible first, then masked-with-label, then masked-no-label, then <nomove>.

    Order:
    1. Visible moves (alphabetically)
    2. Masked moves with labels (alphabetically by y)
    3. Masked moves without labels (y is also $missing_move$)
    4. <nomove> empty slots
    """
    if x_move == PokemonSet.NO_MOVE:
        return (3, 0, x_move)
    if x_move == PokemonSet.MISSING_MOVE:
        # If y is also missing, no ground truth - sort after labeled positions
        if y_move == PokemonSet.MISSING_MOVE:
            return (2, 0, y_move)
        # Masked with label - sort alphabetically by y
        return (1, 0, y_move)
    return (0, 0, x_move)


def _compute_ordering(x_items: List, y_items: List, sort_key) -> List[int]:
    """Compute permutation indices to sort items by key (using y as tie-breaker for masked x)."""
    indexed = [(i, sort_key(x, y)) for i, (x, y) in enumerate(zip(x_items, y_items))]
    indexed.sort(key=lambda x: x[1])
    return [i for i, _ in indexed]


def _apply_ordering(items: List, order: List[int]) -> List:
    """Apply permutation to reorder items."""
    return [items[i] for i in order]


class Team2Seq:
    """
    Converts TeamSets to model-ready sequences and token IDs with standard ordering.
    Ordering rules:
    - Pokemon: lead first, then reserve by visible name (alphabetically), then masked
    - Moves: visible first (alphabetically), then $missing_move$, then <nomove>
    """

    NUM_POKEMON = 6
    ATTRS_PER_POKEMON_BASE = 8  # name, ability, item, tera, 4 moves
    ATTRS_PER_POKEMON_WITH_STATS = 21  # + nature + 6 evs + 6 ivs

    @staticmethod
    def seq_len(include_stats: bool = False) -> int:
        """Compute total sequence length for a team."""
        attrs = (
            Team2Seq.ATTRS_PER_POKEMON_WITH_STATS
            if include_stats
            else Team2Seq.ATTRS_PER_POKEMON_BASE
        )
        return 1 + Team2Seq.NUM_POKEMON * attrs  # format token + pokemon

    def __init__(self, include_stats: bool = False):
        self.include_stats = include_stats
        # Attributes per Pokemon in sequence
        self._attrs_per_pokemon = 8  # name, ability, item, tera, move0-3
        if include_stats:
            self._attrs_per_pokemon += 1 + 6 + 6  # nature + evs + ivs

    @property
    def vocab(self):
        from metamon.backend.team_prediction.vocabulary import get_vocab

        return get_vocab()

    def get_pokemon_indices(self, positions: "torch.Tensor") -> "torch.Tensor":
        """Which Pokemon (0-5) does each position belong to? Format (pos 0) → -1."""
        import torch

        # Pos 1-8 → Pokemon 0, Pos 9-16 → Pokemon 1, etc.
        return torch.where(
            positions == 0, -1, (positions - 1) // self._attrs_per_pokemon
        )

    def get_name_positions(self, pokemon_indices: "torch.Tensor") -> "torch.Tensor":
        """Get the sequence position of each Pokemon's name. Index -1 → 0."""
        import torch

        # Pokemon 0 name at pos 1, Pokemon 1 name at pos 9, etc.
        return torch.where(
            pokemon_indices < 0, 0, 1 + pokemon_indices * self._attrs_per_pokemon
        )

    def get_all_name_positions(self) -> list[int]:
        """Get sequence positions of all 6 Pokemon names."""
        return [1 + p * self._attrs_per_pokemon for p in range(6)]

    def get_move_positions_for_pokemon(self, pokemon_idx: int) -> list[int]:
        """Get the 4 move sequence positions for a specific Pokemon (0-5)."""
        # Sequence layout: format(0), then per pokemon: name, ability, item, tera, move0-3
        # Pokemon 0 moves at positions 5,6,7,8
        # Pokemon 1 moves at positions 13,14,15,16, etc.
        base = (
            1 + pokemon_idx * self._attrs_per_pokemon + 4
        )  # +4 skips name,ability,item,tera
        return [base, base + 1, base + 2, base + 3]

    def encode(
        self, team: TeamSet
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """
        Encode a team to token IDs for inference.

        Returns:
            (tokens, type_ids, pred_mask) where pred_mask indicates missing tokens.
        """
        import torch

        seq, mask = self.to_seq(team)
        tokens, type_ids = self.vocab.pokeset_seq_to_ints(seq)
        return (
            torch.from_numpy(tokens).long(),
            torch.from_numpy(type_ids).long(),
            torch.tensor(mask),
        )

    def encode_pair(
        self, x: TeamSet, y: TeamSet
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """
        Encode (masked, ground_truth) pair to token IDs for training.

        Returns:
            (x_tokens, type_ids, y_tokens, pred_mask)
        """
        import torch

        x_seq, y_seq, pred_mask = self.to_seq_pair(x, y)
        x_tokens, x_type_ids = self.vocab.pokeset_seq_to_ints(x_seq)
        y_tokens, y_type_ids = self.vocab.pokeset_seq_to_ints(y_seq)
        assert (x_type_ids == y_type_ids).all()
        return (
            torch.from_numpy(x_tokens).long(),
            torch.from_numpy(x_type_ids).long(),
            torch.from_numpy(y_tokens).long(),
            torch.tensor(pred_mask),
        )

    def decode(self, tokens: "torch.Tensor") -> TeamSet:
        """Decode token IDs back to a TeamSet."""
        seq = self.vocab.ints_to_pokeset_seq(tokens.cpu().numpy())
        return TeamSet.from_seq(seq, include_stats=self.include_stats)

    def _pokemon_to_seq(self, p: PokemonSet, moves: List[str]) -> List[str]:
        """Convert a single Pokemon to sequence tokens with given move order."""
        seq = [
            f"Mon: {p.name}",
            f"Ability: {p.ability}",
            f"Item: {p.item}",
            f"Tera Type: {p.tera_type}",
        ]
        seq += [f"Move: {m}" for m in moves]
        if self.include_stats:
            seq.append(f"Nature: {p.nature}")
            seq += [f"EVs: {ev}" for ev in p.evs]
            seq += [f"IV: {iv}" for iv in p.ivs]
        return seq

    def _pokemon_pair_to_seq(
        self, x_pokemon: PokemonSet, y_pokemon: PokemonSet
    ) -> Tuple[List[str], List[str], List[bool]]:
        """Convert Pokemon pair to sequences with coordinated move ordering."""
        move_order = _compute_ordering(x_pokemon.moves, y_pokemon.moves, _move_sort_key)
        x_moves = _apply_ordering(x_pokemon.moves, move_order)
        y_moves = _apply_ordering(y_pokemon.moves, move_order)

        x_seq = self._pokemon_to_seq(x_pokemon, x_moves)
        y_seq = self._pokemon_to_seq(y_pokemon, y_moves)

        # pred_mask: True where x is missing but y has a real value
        x_mask = [bool(x_pokemon.missing_regex.search(w)) for w in x_seq]
        y_mask = [bool(y_pokemon.missing_regex.search(w)) for w in y_seq]
        pred_mask = [xm and not ym for xm, ym in zip(x_mask, y_mask)]

        return x_seq, y_seq, pred_mask

    def to_seq(self, team: TeamSet) -> Tuple[List[str], List[bool]]:
        """
        Convert a team to sequence format (for inference).
        Returns (sequence, needs_prediction_mask).

        The mask indicates which tokens are missing and need prediction.
        """
        # Get the ordered sequence (use to_seq_pair for consistent ordering)
        seq, _, _ = self.to_seq_pair(team, team)

        # Compute mask: True for any missing token
        # (different from to_seq_pair which computes xm and not ym)
        mask = [bool(team.lead.missing_regex.search(w)) for w in seq]
        return seq, mask

    def to_seq_pair(
        self, x: TeamSet, y: TeamSet
    ) -> Tuple[List[str], List[str], List[bool]]:
        """
        Convert (x, y) pair to sequences with coordinated ordering.
        Ordering determined by x's visible state (with y as tie-breaker for masked).
        Returns (x_seq, y_seq, pred_mask).
        """
        reserve_order = _compute_ordering(x.reserve, y.reserve, _pokemon_sort_key)
        x_all = [x.lead] + _apply_ordering(x.reserve, reserve_order)
        y_all = [y.lead] + _apply_ordering(y.reserve, reserve_order)

        x_seq = [f"Format: {x.format}"]
        y_seq = [f"Format: {y.format}"]
        pred_mask = [False]

        for xp, yp in zip(x_all, y_all):
            px, py, pm = self._pokemon_pair_to_seq(xp, yp)
            x_seq.extend(px)
            y_seq.extend(py)
            pred_mask.extend(pm)

        return x_seq, y_seq, pred_mask

    def compute_permutation(self, team: TeamSet) -> List[int]:
        """
        Compute the permutation that to_seq applies to put team in sorted order.
        """
        # Position 0 (Format) stays fixed
        permutation = [0]

        # Compute Pokemon ordering (lead stays first, reserve gets sorted)
        # Pass team.reserve twice since we don't have separate ground truth
        reserve_order = _compute_ordering(team.reserve, team.reserve, _pokemon_sort_key)
        pokemon_order = [0] + [
            i + 1 for i in reserve_order
        ]  # 0=lead, then reserve indices

        # For each Pokemon in new order, compute its attribute positions
        all_pokemon = [team.lead] + list(team.reserve)
        for old_p_idx in pokemon_order:
            pokemon = all_pokemon[old_p_idx]
            old_pokemon_start = 1 + old_p_idx * self._attrs_per_pokemon

            # First 4 attributes (name, ability, item, tera) keep relative order
            for attr_offset in range(4):
                permutation.append(old_pokemon_start + attr_offset)

            # Moves get reordered within Pokemon (pass moves twice, no ground truth)
            move_order = _compute_ordering(pokemon.moves, pokemon.moves, _move_sort_key)
            for old_move_idx in move_order:
                permutation.append(old_pokemon_start + 4 + old_move_idx)

            # Stats if included (nature, evs, ivs keep relative order)
            if self.include_stats:
                for stat_offset in range(1 + 6 + 6):
                    permutation.append(old_pokemon_start + 8 + stat_offset)

        return permutation
