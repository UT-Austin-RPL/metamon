import copy
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple

from metamon.data.replay_dataset.parsed_replays.replay_parser.exceptions import *

from poke_env.data import to_id_str
from poke_env.data.gen_data import GenData
from poke_env.environment import Effect as PEEffect
from poke_env.environment import Field as PEField
from poke_env.environment import Move as PEMove
from poke_env.environment import SideCondition as PESideCondition
from poke_env.environment import Status as PEStatus
from poke_env.environment import Weather as PEWeather


class Nothing(Enum):
    """
    `None` means "unknown", while these values mean
    "Known to be missing or N/A"
    """

    NO_ITEM = auto()
    NO_ABILITY = auto()
    NO_WEATHER = auto()
    NO_STATUS = auto()


def _one_hidden_power(move_name: str) -> str:
    # used to map all hidden power moves to the same name
    if move_name.startswith("Hidden Power"):
        return "Hidden Power"
    elif move_name.startswith("hiddenpower"):
        return "hiddenpower"
    else:
        return move_name


@dataclass
class Boosts:
    atk_: int = 0
    spa_: int = 0
    def_: int = 0
    spd_: int = 0
    spe_: int = 0
    accuracy_: int = 0
    evasion_: int = 0

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


class Move(PEMove):
    def __init__(self, name: str, gen: int):
        # in an attempt to handle `choice` messages that give names in a case/space insensitive format,
        # we'll go from the name parsed from the replay --> poke_env id --> poke_env's official move name
        name = _one_hidden_power(name)
        self.lookup_name = to_id_str(name)
        try:
            super().__init__(move_id=self.lookup_name, gen=gen)
            self.charge_move = bool(self.entry["flags"].get("charge"))
            self.name = self.entry["name"]
        except:
            raise MovedexMissingEntry(name, self.lookup_name)
        self.gen_ = gen
        self.pp = self.current_pp  # split from poke-env PP counter
        self.maximum_pp = self.pp

    def __deepcopy__(self, memo):
        # poke-env stores a ton of data in each Move. it is faster
        # to remake the object than copy it.
        new = self.__class__(name=self.name, gen=self.gen_)
        new.pp = self.pp
        self.maximum_pp = self.maximum_pp
        memo[id(self)] = new
        return new

    def __eq__(self, other):
        return self.name == other.name and self.pp == other.pp

    def __repr__(self):
        return f"{self.name} ({self.pp})"


class Pokemon:
    def __init__(self, name: str, lvl: int, gen: int):
        # basic info
        self.name = name
        self.had_name = name
        self.unique_id: str = str(uuid.uuid4())
        self.lvl = lvl
        self.gen = gen

        # pokedex lookup
        pokedex = GenData.from_gen(gen).pokedex
        self.lookup_name = to_id_str(name)
        try:
            pokedex_info = pokedex[self.lookup_name]
        except KeyError:
            raise PokedexMissingEntry(name, self.lookup_name)
        self.type = pokedex_info["types"]
        self.base_stats = pokedex_info["baseStats"]

        # poke-env will assign an abilty as "known" when there
        # is only one option for a pokemon. showdown displays the
        # full list of "Possible Abilities" but waits to assign
        # until the ability is genuinely revealed in sim messages.
        possible_abilities = list(pokedex_info["abilities"].values())
        self.active_ability = None
        if len(possible_abilities) == 1:
            only_ability = possible_abilities[0]
            if only_ability == "No Ability":
                self.active_ability = Nothing.NO_ABILITY
            else:
                self.active_ability = only_ability
        self.had_ability = self.active_ability

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
        self.transformed_into = None

        # within-turn state (reset on next turn)
        self.protected: bool = False
        self.last_target = None
        self.last_targeted_by = None
        self.tricking = None

    def __eq__(self, other):
        # the unique id is what ultimately matters. the showdown messages
        # ID pokemon by species/form/level/status, and we need to map that
        # info to the correct Pokemon object.
        if other is None:
            return False
        return self.unique_id == other.unique_id

    def on_switch_out(self):
        # many temporary effects and changes revert on switch out
        self.boosts = Boosts()
        self.transformed_into = None
        self.moves = copy.deepcopy(self.had_moves)
        self.active_ability = self.had_ability
        self.move_change_to_from = {}

    def on_end_of_turn(self):
        # "within-turn state" is reset
        self.last_target = None
        self.last_targeted_by = None
        self.protected = False
        self.tricking = None

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
        fresh.on_end_of_turn()
        return fresh

    def mimic(self, move_name: str, gen: int):
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
        if "Mimic" not in self.moves:
            raise MimicMiss("Mimic not in moveset")
        if self.last_target[1] != "Mimic":
            raise MimicMiss("Lost reference to Mimic target")
        copied_move = Move(name=move_name, gen=gen)
        # discovers a move the PS viewer doesn't reveal...
        self.last_target[0].reveal_move(copy.deepcopy(copied_move))
        # gen1 PP tracking slightly flawed; won't subtract from Mimic
        pp = self.moves["Mimic"].pp if gen == 1 else 5
        copied_move.pp, copied_move.maximum_pp = pp, pp
        self.move_change_to_from[copied_move.name] = "Mimic"
        del self.moves["Mimic"]
        # by putting the new Move in the move dict ourselves, we prevent
        # its use from "discovering" a move we never had.
        self.moves[copied_move.name] = copied_move

    def get_pp_for_move_name(self, move_name: str) -> Optional[int]:
        if move_name in self.moves:
            return self.moves[move_name].pp

    def reveal_move(self, move: Move):
        if move.name == "Struggle":
            return

        if self.transformed_into is not None:
            if move.name not in self.moves:
                tform_move = copy.deepcopy(move)
                tform_move.pp = 5
                tform_move.maximum_pp = 5
                self.moves[move.name] = tform_move
        else:
            if move.name not in self.moves:
                self.moves[move.name] = move
                self.had_moves[move.name] = copy.deepcopy(move)

    def reveal_ability(self, ability: str):
        self.active_ability = ability
        if self.had_ability is None and not self.transformed_into:
            self.had_ability = ability

    def transform(self, other):
        # too complicated to change the name, stats, ability, & types here.
        # only change things that won't carry forward after switching out.
        # we try to take care of the rest at the very end (`resolve_transforms`)
        self.transformed_into = other
        self.moves = copy.deepcopy(other.moves)
        self.boosts = copy.deepcopy(other.boosts)
        self.status = other.status
        self.active_ability = other.active_ability
        for move in self.moves.values():
            move.pp = 5
            move.maximum_pp = 5

    def use_move(self, move: Move, pp_used: int):
        self.last_used_move = move
        if move.name == "Struggle":
            return

        self.reveal_move(move)
        if self.transformed_into is None and move.name in self.had_moves:
            # subtract pp from had_moves (the moves we brought to the battle)
            curr_pp = self.had_moves[move.name].pp
            # you can always use the move 1 more time when curr_pp == 1, setting pp = 0
            self.had_moves[move.name].pp -= pp_used if curr_pp > 1 else min(pp_used, 1)

        # always subtract pp from current movset
        curr_pp = self.moves[move.name].pp
        self.moves[move.name].pp -= pp_used if curr_pp > 1 else min(pp_used, 1)

    def backfill_info(self, future_mon):
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

        if self.active_item is None:
            assert self.had_item is not None
            self.active_item = self.had_item

        if self.active_ability is None:
            assert self.had_ability is not None
            self.active_ability = self.had_ability

        for move_name, future_move in future_mon.had_moves.items():
            if move_name not in self.had_moves:
                _backup_move(future_move, self.had_moves)
        assert len(self.had_moves.keys()) <= 4

        move_change_from_to = {v: k for k, v in self.move_change_to_from.items()}
        if self.transformed_into is None:
            for move_name, had_move in self.had_moves.items():
                if move_name not in self.moves and move_name not in move_change_from_to:
                    self.moves[move_name] = had_move

    @staticmethod
    def identify_from_details(s: str) -> tuple[str, int]:
        """
        pokemon info from showdown `DETAILS` arg

        https://github.com/smogon/pokemon-showdown/blob/master/sim/SIM-PROTOCOL.md#identifying-pok%C3%A9mon
        """
        name = s.replace(", shiny", "")  # chop off shiny
        name = re.sub(r",\s*[MF]$", "", name)  # chop off extra gender info
        # find level (only provided when not lvl 100)
        match = re.search(r", L\d{1,3}", name)
        if match:
            lvl = match.group()
            name = name.replace(lvl, "")
            lvl = int(lvl[3:])
        else:
            lvl = 100
        return name, lvl

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
            f"\t\thp={self.current_hp}/{self.max_hp}\n",
        ]
        return ",\n".join(items)

    def fill_from_PokemonSet(self, pokemon_set):
        if not self.name == pokemon_set.name:
            raise ValueError("other must have the same name")
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
        pokemon_set_moves = set(
            _one_hidden_power(move)
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
    turn_number: int = None
    is_force_switch: bool = False
    subturns: List = field(default_factory=list)

    def on_end_of_turn(self):
        for pokemon in self.all_pokemon:
            if pokemon:
                pokemon.on_end_of_turn()

    def create_subturn(self, force_switch: bool):
        subturn = copy.deepcopy(self)
        subturn.subturns = []
        subturn.is_force_switch = force_switch
        return subturn

    def remove_empty_subturn(self, team: int, slot: int):
        for subturn in self.subturns:
            if subturn.turn is None and subturn.team == team and subturn.slot == slot:
                self.subturns.remove(subturn)

    def create_next_turn(self):
        next_turn = copy.deepcopy(self)
        # create blank actions
        next_turn.moves_1 = [None, None]
        next_turn.moves_2 = [None, None]
        next_turn.choices_1 = [None, None]
        next_turn.choices_2 = [None, None]
        next_turn.subturns = []
        next_turn.turn_number += 1
        return next_turn

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
    def available_switches_1(self):
        return self._available_switches(for_team_1=True)

    @property
    def available_switches_2(self):
        return self._available_switches(for_team_1=False)

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

    def mark_forced_switch(self, move_str: str):
        # make a blank subturn
        team, slot = self.player_id_to_action_idx(move_str)
        # remove an existing forced switch that wasn't filled (lots of edge cases here)
        self.remove_empty_subturn(team, slot)
        subturn = Subturn(turn=None, team=team, slot=slot, action=None)
        self.subturns.append(subturn)

    def get_pokemon_from_str(self, s: str) -> Optional[Pokemon]:
        if s in ["", "null"]:
            return None
        sub_str = s[1:3]
        if sub_str == "1a" or sub_str == "1:":
            poke = self.active_pokemon_1[0]
        elif sub_str == "1b":
            poke = self.active_pokemon_1[1]
        elif sub_str == "2a" or sub_str == "2:":
            poke = self.active_pokemon_2[0]
        elif sub_str == "2b":
            poke = self.active_pokemon_2[1]
        else:
            raise RareValueError(f"Unknown player in '{s}'")
        if poke is None:
            raise RareValueError(f"No pokemon present in slot {sub_str}")
        return poke

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

    @property
    def pokemon2id(self):
        return {
            pokemon: pokemon.unique_id
            for pokemon in self.all_pokemon
            if pokemon is not None
        }

    @property
    def id2pokemon(self):
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
    ):
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
                name=move_name,
                is_noop=is_noop or False,
                is_switch=is_switch or False,
                user=user,
                target=target,
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
    def all_pokemon(self):
        return self.pokemon_1 + self.pokemon_2

    @property
    def all_active_pokemon(self):
        return self.active_pokemon_1 + self.active_pokemon_2


@dataclass
class Subturn:
    turn: Optional[Turn]
    team: int
    slot: int
    action: Optional[Action]

    @property
    def unfilled(self):
        return self.turn is None

    def matches_slot(self, team, slot):
        return self.team == team and self.slot == slot

    def fill_turn(self, turn):
        self.turn = turn


class Winner(Enum):
    TIE = 0
    PLAYER_1 = 1
    PLAYER_2 = 2


class BackwardMarkers(Enum):
    # mark info with "all we know is that we definitely can't know this"
    FORCE_UNKNOWN = auto()


def unknown(x):
    return x is None or x == BackwardMarkers.FORCE_UNKNOWN


@lru_cache
def get_pokedex_and_moves(format: str) -> Tuple[dict[str, Any], dict[str, Any]]:
    if format[:3] != "gen":
        raise RareValueError(f"Unknown format: {format}")
    gen = int(format[3])
    gen_data = GenData.from_gen(gen)
    return gen_data.pokedex, gen_data.moves


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
