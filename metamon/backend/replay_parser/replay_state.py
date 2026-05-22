import copy
import re
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime
from collections import namedtuple

from metamon.backend.replay_parser.exceptions import *
from metamon.backend.showdown_dex import Dex
from metamon.backend.replay_parser.pe_datatypes import (
    PEEffect,
    PEField,
    PEMove,
    PESideCondition,
    PEStatus,
    PEWeather,
)
from metamon.backend.replay_parser.str_parsing import move_name, pokemon_name


class Nothing(Enum):
    """
    `None` means "unknown", while these values mean
    "Known to be missing or N/A"
    """

    NO_ITEM = auto()
    NO_ABILITY = auto()
    NO_WEATHER = auto()
    NO_STATUS = auto()
    NO_TERA_TYPE = auto()


class BackwardMarkers(Enum):
    """
    Mark info with "all we know is that we definitely can't know this"
    """

    FORCE_UNKNOWN = auto()


def unknown(x: Any) -> bool:
    """
    Check if a value is considered "unknown"
    """
    return x is None or x == BackwardMarkers.FORCE_UNKNOWN


# organize some info for tracking down edge cases

TargetedBy = namedtuple("TargetedBy", ["pokemon", "move"])

Targeting = namedtuple("Targeting", ["pokemon", "move"])

Replacement = namedtuple("Replacement", ["replaced", "replaced_with", "turn_range"])


class Winner(Enum):
    TIE = 0
    PLAYER_1 = 1
    PLAYER_2 = 2


class Boosts:
    """
    Stat stage boosts
    """

    def __init__(self, atk_=0, spa_=0, def_=0, spd_=0, spe_=0, accuracy_=0, evasion_=0):
        self.atk_: int = atk_
        self.spa_: int = spa_
        self.def_: int = def_
        self.spd_: int = spd_
        self.spe_: int = spe_
        self.accuracy_: int = accuracy_
        self.evasion_: int = evasion_

    @property
    def stat_attrs(self):
        return [
            f"{s}_" for s in ["atk", "spa", "def", "spd", "spe", "accuracy", "evasion"]
        ]

    def set_to_with_str(self, s: str, value: int):
        if hasattr(self, f"{s}_"):
            setattr(self, f"{s}_", value)
        else:
            raise RareValueError(f"Unknown stat: '{s}'")

    def change_with_str(self, s: str, value: int):
        try:
            stage = getattr(self, f"{s}_")
        except AttributeError:
            raise RareValueError(f"Unknown stat: '{s}'")
        else:
            # stat "stages" are always in [-6, 6]
            setattr(self, f"{s}_", min(max(stage + value, -6), 6))

    def get_boost(self, s: str):
        return getattr(self, f"{s}_")

    def to_dict(self):
        return {
            "atk": self.atk_,
            "spa": self.spa_,
            "def": self.def_,
            "spd": self.spd_,
            "spe": self.spe_,
            "accuracy": self.accuracy_,
            "evasion": self.evasion_,
        }


class Move(PEMove):
    """
    A wrapper around poke-env's Move object with its own pp counter
    """

    def __init__(self, name: str, gen: int):
        # in an attempt to handle `choice` messages that give names in a case/space insensitive format,
        # we'll go from the name parsed from the replay --> dex id --> dex json's official move name
        lookup_name = move_name(name)
        self.lookup_name = lookup_name
        try:
            super().__init__(move_id=self.lookup_name, gen=gen)
            self.charge_move = bool(self.entry.get("flags", {}).get("charge", False))
            self.name = self.entry.get("name", name)
        except:
            raise MovedexMissingEntry(name, self.lookup_name)
        self.gen_ = gen
        self.pp = self.current_pp  # split from poke-env PP counter
        self.maximum_pp = self.pp

    def set_pp(self, pp: int):
        self.pp = pp
        self._current_pp = pp

    def __deepcopy__(self, memo):
        # poke-env stores a ton of data in each Move. it is faster
        # to remake the object than copy it.
        new = self.__class__(name=self.name, gen=self.gen_)
        new.pp = self.pp
        new._current_pp = self.pp
        self.maximum_pp = self.maximum_pp
        memo[id(self)] = new
        return new

    def from_transform(self):
        new_move = Move(name=self.name, gen=self.gen_)
        new_move.set_pp(5)
        new_move.maximum_pp = 5
        return new_move

    def __eq__(self, other):
        return self.name == other.name and self.pp == other.pp

    def __repr__(self):
        return f"{self.name} ({self.pp})"


class Pokemon:

    def __init__(self, name: str, lvl: int, gen: int):
        # basic info
        self.name: Optional[str] = None  # changes on forme change
        self.had_name: Optional[str] = None  # never changes
        self.nickname: Optional[str] = None  # player-assigned username
        self.unique_id: str = str(
            uuid.uuid4()
        )  # unique id used internally to avoid matching by names at all
        self.lvl: int = lvl
        self.gen: int = gen

        # pokedex lookup
        self.active_ability: Optional[str] = None
        self.had_ability: Optional[str] = None
        self.type: Optional[List[str]] = None
        self.had_type: Optional[List[str]] = None
        self.tera_type: Optional[str] = None
        self.base_stats: Dict[str, int] = {}

        self.active_item: Optional[str] = None
        self.had_item: Optional[str] = None
        self.moves: Dict[str, Move] = {}
        self.had_moves: Dict[str, Move] = {}
        self.move_change_to_from: Dict[str, str] = {}

        self.last_used_move: Move = None
        self.boosts: Boosts = Boosts()
        self.status: PEStatus | Nothing = Nothing.NO_STATUS
        self.effects: Dict[PEEffect, int] = {}
        self.current_hp: int = None
        self.max_hp: int = None

        self.transformed_this_turn: bool = False
        self.transformed_into: Optional[Pokemon] = None

        # within-turn state (reset on next turn)
        self.protected: bool = False
        self.last_target: Optional[Targeting] = None
        self.last_targeted_by: Optional[TargetedBy] = None
        self.tricking: Optional["Pokemon"] = None
        self.update_pokedex_info(name)

    def __eq__(self, other):
        # the unique id is what ultimately matters. the showdown messages
        # ID pokemon by species/form/level/status, and we need to map that
        # info to the correct Pokemon object.
        if other is None:
            return False
        return self.unique_id == other.unique_id

    @staticmethod
    def _lookup_pokedex_info(name: str, gen: int):
        pokedex = Dex.from_gen(gen).pokedex
        lookup_name = pokemon_name(name)
        try:
            pokedex_info = pokedex[lookup_name]
        except KeyError:
            raise PokedexMissingEntry(name, lookup_name)
        return pokedex_info

    def update_pokedex_info(self, name: str):
        pokedex_info = self._lookup_pokedex_info(name, gen=self.gen)
        self.name = pokedex_info["name"]
        if self.had_name is None:
            self.had_name = pokedex_info["baseSpecies"]
        self.type = pokedex_info["types"]
        if self.had_type is None:
            self.had_type = copy.deepcopy(self.type)
        self.base_stats = pokedex_info["baseStats"]
        # use the pokedex to reveal info
        possible_abilities = list(pokedex_info["abilities"].values())
        required_ability = pokedex_info.get("requiredAbility", None)
        if required_ability is not None:
            self.reveal_ability(required_ability)
        elif len(possible_abilities) == 1:
            only_ability = possible_abilities[0]
            if only_ability == "No Ability":
                only_ability = Nothing.NO_ABILITY
            self.reveal_ability(only_ability)
        required_item = pokedex_info.get("requiredItem", None)
        if required_item is not None:
            self.reveal_item(required_item)
        required_tera = pokedex_info.get("requiredTeraType", None)
        if required_tera is not None and self.gen == 9:
            self.tera_type = required_tera

    def on_switch_out(self):
        # many temporary effects and changes revert on switch out
        self.boosts = Boosts()
        self.transformed_into = None
        self.moves = copy.deepcopy(self.had_moves)
        self.active_ability = self.had_ability
        self.move_change_to_from = {}
        self.type = self.had_type
        self.effects = {}

    def on_end_of_turn(self):
        # "within-turn state" is reset
        self.last_target = None
        self.last_targeted_by = None
        self.protected = False
        self.tricking = None
        self.transformed_this_turn = False

    def fresh_like(self):
        # returns a version of this pokemon before it entered a battle
        fresh = copy.deepcopy(self)
        fresh.boosts = Boosts()
        fresh.status = Nothing.NO_STATUS
        fresh.effects = {}
        fresh.current_hp = 100
        fresh.max_hp = 100
        fresh.transformed_into = None
        fresh.active_ability = fresh.had_ability
        fresh.active_item = fresh.had_item
        fresh.moves = copy.deepcopy(fresh.had_moves)
        fresh.type = fresh.had_type
        fresh.on_end_of_turn()
        return fresh

    def mimic(self, move_name: str):
        """
        TODO/Dev note: Mimic is really hard from the replay POV and this isn't perfect.

        - Copying moves that we actually already knew, but had not yet revealed...
          then reveal before the switching / move_change_to_from system fixes it.
                - Similarly: copying a move we probably didn't already know, but inferred
                  from the random generation in the backward pass.
                - probably fixable.

        - Copying moves from a pokemon that is transformed into us. Did we
          always have that move?
                - probably fixable.

        - PP tracking should reference back to the move we copied, but this doesn't
          work the same in every generation.
                - Truly accurate PP tracking seems like a lost cause anyway, becuase
                  it breaks in situations that are a lot more common than Mimic.

        - And so on. We at least catch the basic case of copying a new move you didn't
          have and being able to use it until you switch out or faint in a few turns.
        """
        if "Mimic" not in self.moves or self.last_target is None:
            raise MimicMiss("Mimic not in moveset")
        if self.last_target.move != "Mimic":
            raise MimicMiss("Lost reference to Mimic target")
        copied_move = Move(name=move_name, gen=self.gen)
        # discovers a move the PS viewer doesn't reveal...
        self.last_target.pokemon.reveal_move(copy.deepcopy(copied_move))
        # gen1 PP tracking slightly flawed; won't subtract from Mimic
        pp = self.moves["Mimic"].pp if self.gen == 1 else 5
        copied_move.set_pp(pp)
        copied_move.maximum_pp = pp
        self.move_change_to_from[copied_move.name] = "Mimic"
        del self.moves["Mimic"]
        # by putting the new Move in the move dict ourselves, we prevent
        # its use from "discovering" a move we never had.
        self.moves[copied_move.name] = copied_move

    def get_pp_for_move_name(self, move_name: str) -> Optional[int]:
        if move_name in self.moves:
            return self.moves[move_name].pp

    def reveal_move(self, move: Move):
        if move.name in {"Struggle", "Recharge", "Fight"}:
            return

        if self.transformed_into is not None:
            if move.name not in self.moves:
                tform_move = move.from_transform()
                self.moves[move.name] = tform_move
        else:
            if move.name not in self.moves:
                self.moves[move.name] = move
                self.had_moves[move.name] = copy.deepcopy(move)

    def reveal_ability(self, ability: str):
        self.active_ability = ability
        if self.had_ability is None and not self.transformed_into:
            self.had_ability = ability

    def reveal_item(self, item: str):
        self.active_item = item
        if self.had_item is None:
            self.had_item = item

    def transform(self, other):
        # too complicated to change the name, stats, ability, & types here.
        # only change things that won't carry forward after switching out.
        # we try to take care of the rest at the very end (`resolve_transforms`)
        self.transformed_this_turn = True
        self.transformed_into = other
        self.boosts = copy.deepcopy(other.boosts)
        self.active_ability = other.active_ability
        self.moves = {k: v.from_transform() for k, v in other.moves.items()}

    @property
    def last_used_move_name(self) -> Optional[str]:
        return self.last_used_move.name if self.last_used_move is not None else None

    @property
    def last_used_move_pp(self) -> Optional[int]:
        return self.last_used_move.pp if self.last_used_move is not None else None

    def use_move(self, move: Move, pp_used: int):
        self.last_used_move = move
        if move.name == "Struggle":
            return

        self.reveal_move(move)
        if move.name not in self.moves:
            return

        if self.transformed_into is None and move.name in self.had_moves:
            # subtract pp from had_moves (the moves we brought to the battle)
            curr_pp = self.had_moves[move.name].pp
            # you can always use the move 1 more time when curr_pp == 1, setting pp = 0
            self.had_moves[move.name].set_pp(
                curr_pp - (pp_used if curr_pp > 1 else min(pp_used, 1))
            )

        # always subtract pp from current movset
        curr_pp = self.moves[move.name].pp
        self.moves[move.name].set_pp(
            curr_pp - (pp_used if curr_pp > 1 else min(pp_used, 1))
        )

    def backfill_info(self, future_mon):
        """
        Update this Pokemon's info based on a version of itself from later in the battle
        (when we've hopefully learned more about it)
        """
        if future_mon != self:
            raise ValueError(
                "Trying to transfer properties between two different pokemon!"
            )

        def _fill_if(attr):
            if unknown(getattr(self, attr)):
                setattr(self, attr, copy.deepcopy(getattr(future_mon, attr)))

        def _backup_move(move_t1, moveset):
            move_t = copy.deepcopy(move_t1)
            # moves are usually discovered after they are used the first time (pp -= 1)
            move_t.pp = min(move_t1.maximum_pp, move_t1.pp + 1)
            moveset[move_t.name] = move_t

        # if info has been discovered in the future_mon that isn't
        # known now, copy it.
        _fill_if("had_item")
        _fill_if("had_ability")
        _fill_if("had_name")
        _fill_if("max_hp")
        _fill_if("current_hp")
        _fill_if("tera_type")

        if self.active_item is None:
            assert self.had_item is not None
            self.active_item = self.had_item

        if self.active_ability is None:
            assert self.had_ability is not None
            self.active_ability = self.had_ability

        for move_name, future_move in future_mon.had_moves.items():
            if move_name not in self.had_moves:
                _backup_move(future_move, self.had_moves)

        if len(self.had_moves.keys()) > 4:
            raise TooManyMoves(self)

        move_change_from_to = {v: k for k, v in self.move_change_to_from.items()}
        if self.transformed_into is None:
            for move_name, had_move in self.had_moves.items():
                if move_name not in self.moves and move_name not in move_change_from_to:
                    self.moves[move_name] = had_move

    @staticmethod
    def identify_from_details(
        s: str, gen: int, get_base_species: bool = False
    ) -> tuple[str, int]:
        """
        pokemon info from showdown `DETAILS` arg

        https://github.com/smogon/pokemon-showdown/blob/master/sim/SIM-PROTOCOL.md#identifying-pok%C3%A9mon
        """
        name = s.replace(", shiny", "")  # chop off shiny
        name = re.sub(r",\s*tera:.*$", "", name)  # chop off tera info
        name = re.sub(r",\s*[MF]$", "", name)  # chop off extra gender info
        # find level (only provided when not lvl 100)
        match = re.search(r", L\d{1,3}", name)
        if match:
            lvl = match.group()
            name = name.replace(lvl, "")
            lvl = int(lvl[3:])
        else:
            lvl = 100
        # get the standardized name and base species from the pokedex
        name_out = name
        dex = Dex.from_gen(gen)
        try:
            entry = dex.get_pokedex_entry(name)
        except KeyError:
            if get_base_species:
                # give it a shot i guess. if it's not in the dex,
                # the replay parser is probably going to fail later anyway.
                name_out = name_out.split("-")[0].strip()
        else:
            name_out = entry.get(
                "baseSpecies" if get_base_species else "name", name_out
            )

        return name_out, lvl

    def __repr__(self):
        return f"{self.name} - {self.active_ability} - {self.active_item} : {self._moveset_str}"

    @property
    def _moveset_str(self):
        return " ".join(str(m) for m in self.moves.values())

    def start_effect(self, effect: PEEffect):
        if effect not in self.effects:
            self.effects[effect] = 0
        elif effect.is_action_countable:
            self.effects[effect] += 1

    def end_effect(self, effect: PEEffect):
        if effect in self.effects:
            self.effects.pop(effect)

    def __str__(self):
        items = [
            f"name={self.name}",
            f"\t\tlvl={self.lvl}",
            f"\t\tboosts={self.boosts}",
            f"\t\tstatus={self.status}",
            f"\t\tmoves={self._moveset_str}",
            f"\t\tability={self.had_ability}",
            f"\t\thad_item={self.had_item}",
            f"\t\tactive_item={self.active_item}",
            f"\t\teffects={self.effects}",
            f"\t\thp={self.current_hp}/{self.max_hp}",
            f"\t\ttera_type={self.tera_type}\n",
        ]
        return "\n".join(items)

    def fill_from_PokemonSet(self, pokemon_set):
        """
        Fill unknown details based on the outputs of our team prediction module.
        """
        if not self.had_name == pokemon_set.base_species:
            raise ValueError(
                f"species/name mixup: {self.had_name} != {pokemon_set.base_species}"
            )

        item = pokemon_set.item
        if item == pokemon_set.NO_ITEM:
            item = Nothing.NO_ITEM
        elif item == pokemon_set.MISSING_ITEM:
            item = None
        self.had_item = item

        ability = pokemon_set.ability
        if ability == pokemon_set.NO_ABILITY:
            ability = Nothing.NO_ABILITY
        elif ability == pokemon_set.MISSING_ABILITY:
            ability = None
        self.had_ability = ability

        tera_type = pokemon_set.tera_type
        if tera_type == pokemon_set.NO_TERA_TYPE:
            tera_type = Nothing.NO_TERA_TYPE
        elif tera_type == pokemon_set.MISSING_TERA_TYPE:
            tera_type = None
        self.tera_type = tera_type

        pokemon_set_moves = set(
            move_name(move)
            for move in pokemon_set.moves
            if (
                move != pokemon_set.MISSING_MOVE
                and move != "Struggle"
                and move != pokemon_set.NO_MOVE
            )
        )
        moves_to_add = pokemon_set_moves - set(self.had_moves.keys())
        while len(self.had_moves.keys()) < 4 and moves_to_add:
            choice = moves_to_add.pop()
            new_move = Move(name=choice, gen=self.gen)
            self.had_moves[new_move.name] = new_move

        if self.max_hp is None:
            assert self.current_hp is None
            self.max_hp = 100
            self.current_hp = 100
        return self


@dataclass
class Action:
    name: str
    user: Optional[Pokemon]
    target: Optional[Pokemon]
    is_noop: bool = False
    is_switch: bool = False
    is_tera: bool = False
    is_revival: bool = False

    def __repr__(self):
        return f"Action: {self.name}"

    def __str__(self):
        items = [
            f"name={self.name}",
            f"is_switch={self.is_switch}",
            f"target={'None' if self.target is None else self.target.name}",
        ]
        return ",".join(items)


@dataclass
class Turn:
    pokemon_1: List[Optional[Pokemon]] = field(default_factory=lambda: [None] * 6)
    pokemon_2: List[Optional[Pokemon]] = field(default_factory=lambda: [None] * 6)
    active_pokemon_1: List[Optional[Pokemon]] = field(
        default_factory=lambda: [None, None]
    )
    active_pokemon_2: List[Optional[Pokemon]] = field(
        default_factory=lambda: [None, None]
    )
    moves_1: List[Optional[Action]] = field(default_factory=lambda: [None, None])
    choices_1: List[Optional[Action]] = field(default_factory=lambda: [None, None])
    moves_2: List[Optional[Action]] = field(default_factory=lambda: [None, None])
    choices_2: List[Optional[Action]] = field(default_factory=lambda: [None, None])
    weather: PEWeather | Nothing = Nothing.NO_WEATHER
    battle_field: Dict[PEField, int] = field(default_factory=dict)
    conditions_1: Dict[PESideCondition, int] = field(default_factory=dict)
    conditions_2: Dict[PESideCondition, int] = field(default_factory=dict)
    replacements_1: List[Replacement] = field(default_factory=list)
    replacements_2: List[Replacement] = field(default_factory=list)
    turn_number: int = None
    is_force_switch: bool = False
    subturns: List = field(default_factory=list)
    can_tera_1: bool = False
    can_tera_2: bool = False
    teampreview_1: List[Pokemon] = field(default_factory=list)
    teampreview_2: List[Pokemon] = field(default_factory=list)

    def get_active_pokemon(self, p1: bool) -> Optional[Pokemon]:
        return self.active_pokemon_1 if p1 else self.active_pokemon_2

    def get_pokemon(self, p1: bool) -> Optional[Pokemon]:
        return self.pokemon_1 if p1 else self.pokemon_2

    def get_switches(self, p1: bool) -> List[Pokemon]:
        return self.available_switches_1 if p1 else self.available_switches_2

    def get_team_dict(self, p1: bool) -> List[Optional[Pokemon]]:
        role = "p1" if p1 else "p2"
        pokemon = self.pokemon_1 if p1 else self.pokemon_2
        return {f"{role} {p.name}": p for p in pokemon if p is not None}

    def get_moves(self, p1: bool) -> List[Optional[Action]]:
        return self.moves_1 if p1 else self.moves_2

    def get_conditions(self, p1: bool) -> Dict[PESideCondition, int]:
        return self.conditions_1 if p1 else self.conditions_2

    def get_available_switches(self, p1: bool) -> List[Pokemon]:
        return self.available_switches_1 if p1 else self.available_switches_2

    def get_replacements(self, p1: bool) -> List[Replacement]:
        return self.replacements_1 if p1 else self.replacements_2

    def get_pokemon_from_nickname(self, s: str) -> Optional[Pokemon]:
        if s in ["", "null"]:
            return None
        side_id = s[1:3]
        nickname = s.split(":")[1].strip()
        p1 = "1" in side_id
        side = self.get_pokemon(p1=p1)
        for pokemon in side:
            if pokemon is not None and pokemon.nickname == nickname:
                return pokemon
        return None

    def get_pokemon_list_from_str(self, s: str) -> List[Optional[Pokemon]]:
        sub_str = s[0:2]
        if sub_str == "p1":
            return self.pokemon_1
        elif sub_str == "p2":
            return self.pokemon_2
        else:
            raise RareValueError(f"Unknown player: {sub_str}")

    def get_active_pokemon_from_str(self, s: str) -> List[Optional[Pokemon]]:
        sub_str = s[0:2]
        if sub_str == "p1":
            return self.active_pokemon_1
        elif sub_str == "p2":
            return self.active_pokemon_2
        else:
            raise RareValueError(f"Unknown player: {sub_str}")

    def get_pokemon_from_str(
        self, showdown_msg: str, fallback_to_nickname: bool = True
    ) -> Optional[Pokemon]:
        if showdown_msg in ["", "null"]:
            return None
        # get active pokemon from slot id first
        sub_str = showdown_msg[1:3]
        if sub_str == "1a" or sub_str == "1:":
            poke = self.active_pokemon_1[0]
        elif sub_str == "1b":
            poke = self.active_pokemon_1[1]
        elif sub_str == "2a" or sub_str == "2:":
            poke = self.active_pokemon_2[0]
        elif sub_str == "2b":
            poke = self.active_pokemon_2[1]
        else:
            raise RareValueError(f"Unknown player in '{showdown_msg}'")
        if poke is None:
            raise RareValueError(f"No pokemon present in slot {sub_str}")
        if ":" in showdown_msg:
            # discover nickname
            nickname = showdown_msg.split(":")[1].strip()
            # attempt to assign nickname
            if poke.nickname is None:
                poke.nickname = nickname
            elif poke.nickname != nickname and fallback_to_nickname:
                # experimental: hard to say when this is a good idea.
                # there are some messages where the reference is meant
                # to be identified by nickname, so it should be ok
                # to check by position first, and then fallback to nickname.
                # this relies on the idea that the first time we need to id
                # a pokemon by nickname comes after a message that would discover
                # the nickname. so far the only cases I know of are Revival Blessing
                # and curestatus from heal bell, though there are probably more.
                poke_by_nickname = self.get_pokemon_from_nickname(showdown_msg)
                if poke_by_nickname is not None:
                    return poke_by_nickname
        return poke

    def get_teampreview(self, p1: bool) -> List[Optional[Pokemon]]:
        return self.teampreview_1 if p1 else self.teampreview_2

    def player_id_to_action_idx(self, move_str: str) -> Tuple[int, int]:
        sub_str = move_str[1:3]
        if sub_str == "1a" or sub_str == "1:":
            return 1, 0
        elif sub_str == "1b":
            return 1, 1
        elif sub_str == "2a" or sub_str == "2:":
            return 2, 0
        elif sub_str == "2b":
            return 2, 1
        raise RareValueError(f"Unknown player in '{move_str}'")

    def pokemon_to_action_idx(self, pokemon: Pokemon) -> Optional[Tuple[int, int]]:
        for team_idx, team in enumerate([self.active_pokemon_1, self.active_pokemon_2]):
            for slot_idx, slot in enumerate(team):
                if slot == pokemon:
                    # teams were apparently 1-indexed...
                    return team_idx + 1, slot_idx
        return None

    def on_end_of_turn(self) -> None:
        for pokemon in self.all_pokemon:
            if pokemon:
                pokemon.on_end_of_turn()

    def create_next_turn(self) -> "Turn":
        next_turn = copy.deepcopy(self)
        next_turn.moves_1 = [None, None]
        next_turn.moves_2 = [None, None]
        next_turn.choices_1 = [None, None]
        next_turn.choices_2 = [None, None]
        next_turn.subturns = []
        next_turn.turn_number += 1
        next_turn.replacements_1 = []
        next_turn.replacements_2 = []
        return next_turn

    def create_subturn(self, force_switch: bool) -> "Turn":
        subturn = copy.deepcopy(self)
        subturn.subturns = []
        subturn.is_force_switch = force_switch
        return subturn

    def remove_empty_subturn(self, team: int, slot: int) -> None:
        for subturn in self.subturns:
            if subturn.turn is None and subturn.team == team and subturn.slot == slot:
                self.subturns.remove(subturn)

    def mark_forced_switch(self, move_str: str) -> None:
        # make a blank subturn
        team, slot = self.player_id_to_action_idx(move_str)
        # remove an existing forced switch that wasn't filled (lots of edge cases here)
        self.remove_empty_subturn(team, slot)
        subturn = Subturn(turn=None, team=team, slot=slot, action=None)
        self.subturns.append(subturn)

    def _available_switches(self, for_team_1: bool) -> List[Pokemon]:
        active = self.active_pokemon_1 if for_team_1 else self.active_pokemon_2
        team = self.pokemon_1 if for_team_1 else self.pokemon_2
        active_ids = {a.unique_id for a in active if a is not None}
        return [
            p
            for p in team
            if p is not None
            and p.status != PEStatus.FNT
            and p.unique_id not in active_ids
        ]

    @property
    def available_switches_1(self) -> List[Pokemon]:
        return self._available_switches(for_team_1=True)

    @property
    def available_switches_2(self) -> List[Pokemon]:
        return self._available_switches(for_team_1=False)

    @property
    def pokemon2id(self) -> Dict[Pokemon, str]:
        return {
            pokemon: pokemon.unique_id
            for pokemon in self.all_pokemon
            if pokemon is not None
        }

    @property
    def id2pokemon(self) -> Dict[str, Pokemon]:
        return {
            pokemon.unique_id: pokemon
            for pokemon in self.all_pokemon
            if pokemon is not None
        }

    def get_pokemon_by_uid(self, uid: str) -> Optional[Pokemon]:
        for pokemon in self.all_pokemon:
            if pokemon and pokemon.unique_id == uid:
                return pokemon

    def set_move_attribute(
        self,
        s: str,
        move_name: Optional[str] = None,
        is_noop: Optional[bool] = None,
        is_switch: Optional[bool] = None,
        user: Optional[Pokemon] = None,
        target: Optional[Pokemon] = None,
        is_tera: Optional[bool] = None,
    ) -> None:
        # "p1a", "p2a", ...
        if s[1] == "1":
            moves_list = self.moves_1
        elif s[1] == "2":
            moves_list = self.moves_2
        else:
            raise RareValueError(f"Unknown player: '{s}'")
        if s[2] == "a" or s[2] == ":":
            index = 0
        elif s[2] == "b":
            index = 1
        else:
            raise RareValueError(f"Unknown index: '{s}'")

        if moves_list[index] is None:
            # create new Action
            moves_list[index] = Action(
                name=move_name or None,
                is_noop=is_noop or False,
                is_switch=is_switch or False,
                user=user or None,
                target=target or None,
                is_tera=is_tera or False,
            )
        else:
            # adjust existing Action
            if move_name is not None:
                moves_list[index].name = move_name
            if is_switch is not None:
                moves_list[index].is_switch = is_switch
            if user is not None:
                moves_list[index].user = user
            if target is not None:
                moves_list[index].target = target
            if is_noop is not None:
                moves_list[index].is_noop = is_noop
            if is_tera is not None:
                moves_list[index].is_tera = is_tera

    def __repr__(self) -> str:
        poke_1_str = "\n\t\t".join([str(x) for x in self.pokemon_1])
        poke_2_str = "\n\t\t".join([str(x) for x in self.pokemon_2])
        move_1_str = ", ".join(["None" if x is None else str(x) for x in self.moves_1])
        move_2_str = ", ".join(["None" if x is None else str(x) for x in self.moves_2])
        items = [
            f"weather={self.weather}",
            f"\tpokemon_1={poke_1_str}",
            f"\tpokemon_2={poke_2_str}",
            f"\tactive_pokemon_1={self.active_pokemon_1}",
            f"\tactive_pokemon_2={self.active_pokemon_2}",
            f"\tmoves_1={move_1_str}",
            f"\tmoves_2={move_2_str}",
        ]
        return ",\n".join(items)

    @property
    def all_pokemon(self) -> List[Optional[Pokemon]]:
        return self.pokemon_1 + self.pokemon_2

    @property
    def all_active_pokemon(self) -> List[Optional[Pokemon]]:
        return self.active_pokemon_1 + self.active_pokemon_2


@dataclass
class Subturn:
    turn: Optional[Turn]
    team: int
    slot: int
    action: Optional[Action]

    @property
    def unfilled(self) -> bool:
        return self.turn is None

    def matches_slot(self, team: int, slot: int) -> bool:
        return self.team == team and self.slot == slot

    def fill_turn(self, turn: Turn) -> None:
        self.turn = turn


@dataclass
class ParsedReplay:
    gameid: str
    time_played: datetime
    format: Optional[str] = None
    ratings: List[Optional[int | str]] = field(default_factory=lambda: [None, None])
    players: List[Optional[str]] = field(default_factory=lambda: [None, None])
    gen: Optional[int] = None
    turnlist: List[Turn] = field(default_factory=lambda: [Turn(turn_number=0)])
    rules: List[str] = field(default_factory=list)
    winner: Optional[Winner] = None
    check_warnings: Set[WarningFlags] = field(default_factory=set)

    def __getitem__(self, i):
        return self.turnlist[i]

    def has_warning(self, warning: WarningFlags) -> bool:
        return warning in self.check_warnings

    def add_warning(self, warning: WarningFlags) -> None:
        self.check_warnings.add(warning)

    def get_pov_turnlist(
        self, from_p1_pov: bool, start_from_turn: int = 0
    ) -> list[Turn]:
        flat = []
        for turn in self.turnlist[start_from_turn:]:
            for subturn in turn.subturns:
                if subturn.turn is not None and subturn.team == (
                    1 if from_p1_pov else 2
                ):
                    flat.append(subturn.turn)
            flat.append(turn)
        return flat

    @property
    def flattened_turnlist(self) -> list[Turn]:
        flat = []
        for turn in self.turnlist:
            for subturn in turn.subturns:
                if subturn.turn is not None:
                    flat.append(subturn.turn)
            flat.append(turn)
        return flat

    @property
    def replay_url(self) -> str:
        return f"https://replay.pokemonshowdown.com/{self.gameid}"

    def __str__(self):
        turnlist_str = "\n\n\t".join(
            [f"Turn {i} -> {str(x)}" for i, x in enumerate(self.turnlist)]
        )
        rules_str = "\n\t".join([str(x) for x in self.rules])
        poke_1_str = "\n\t".join([str(x) for x in self.turnlist[-1].pokemon_1])
        poke_2_str = "\n\t".join([str(x) for x in self.turnlist[-1].pokemon_2])
        items = [
            f"format={self.format}",
            f"gen={self.gen}",
            f"rating={self.rating}",
            f"players={self.players}",
            f"pokemon 1={poke_1_str}",
            f"pokemon 2={poke_2_str}",
            f"rules={rules_str}",
            f"winner={self.winner}",
            f"turnlist={turnlist_str}",
        ]
        return ",\n".join(items)


@dataclass
class ReplayState:
    format: str
    force_switch: bool
    active_pokemon: Pokemon
    opponent_active_pokemon: Pokemon
    available_switches: List[Pokemon]
    player_prev_move: Move
    opponent_prev_move: Move
    opponent_team: List[Pokemon]
    player_conditions: Dict[PESideCondition, int]
    opponent_conditions: Dict[PESideCondition, int]
    battle_field: Dict[PEField, int]
    weather: PEWeather | Nothing
    battle_won: bool
    battle_lost: bool
    can_tera: bool
    opponent_teampreview: List[Pokemon]
