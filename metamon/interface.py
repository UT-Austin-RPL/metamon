import copy
import re
import warnings
from dataclasses import dataclass, asdict
from typing import Optional, List, Any, Set
from abc import ABC, abstractmethod

import numpy as np
import gymnasium as gym
import string

from poke_env.environment import (
    Battle,
    Move,
    Pokemon,
    Field,
    Effect,
    SideCondition,
    Status,
    PokemonType,
)
from poke_env.player import BattleOrder, Player

import metamon
from metamon.config import format_for_agent
from metamon.tokenizer import PokemonTokenizer, UNKNOWN_TOKEN
from metamon.backend.replay_parser.replay_state import (
    Move as ReplayMove,
    Pokemon as ReplayPokemon,
    Action as ReplayAction,
    ReplayState,
    Nothing as ReplayNothing,
)
from metamon.backend.replay_parser.str_parsing import (
    clean_no_numbers,
    clean_name,
    pokemon_name,
    move_name,
)

ALL_OBSERVATION_SPACES = {}
ALL_ACTION_SPACES = {}
ALL_REWARD_FUNCTIONS = {}


def register_observation_space(name: Optional[str] = None):
    """
    Decorator to register observation space classes.

    Args:
        name: Optional custom name for the observation space. If not provided, uses the class name.

    Usage:
        @register_observation_space()
        class MyObservationSpace(ObservationSpace):
            pass

        @register_observation_space("CustomName")
        class AnotherObservationSpace(ObservationSpace):
            pass
    """

    def _register(cls):
        obs_name = name if name is not None else cls.__name__
        if obs_name in ALL_OBSERVATION_SPACES:
            raise ValueError(f"Observation space '{obs_name}' is already registered!")
        ALL_OBSERVATION_SPACES[obs_name] = cls
        return cls

    return _register


def register_action_space(name: Optional[str] = None):
    """
    Decorator to register action space classes.

    Args:
        name: Optional custom name for the action space. If not provided, uses the class name.

    Usage:
        @register_action_space()
        class MyActionSpace(ActionSpace):
            pass

        @register_action_space("CustomName")
        class AnotherActionSpace(ActionSpace):
            pass
    """

    def _register(cls):
        action_name = name if name is not None else cls.__name__
        if action_name in ALL_ACTION_SPACES:
            raise ValueError(f"Action space '{action_name}' is already registered!")
        ALL_ACTION_SPACES[action_name] = cls
        return cls

    return _register


def register_reward_function(name: Optional[str] = None):
    """
    Decorator to register reward function classes.

    Args:
        name: Optional custom name for the reward function. If not provided, uses the class name.

    Usage:
        @register_reward_function()
        class MyRewardFunction(RewardFunction):
            pass

        @register_reward_function("CustomName")
        class AnotherRewardFunction(RewardFunction):
            pass
    """

    def _register(cls):
        reward_name = name if name is not None else cls.__name__
        if reward_name in ALL_REWARD_FUNCTIONS:
            raise ValueError(f"Reward function '{reward_name}' is already registered!")
        ALL_REWARD_FUNCTIONS[reward_name] = cls
        return cls

    return _register


def get_observation_space_names():
    """Get all registered observation space names."""
    return sorted(ALL_OBSERVATION_SPACES.keys())


def get_action_space_names():
    """Get all registered action space names."""
    return sorted(ALL_ACTION_SPACES.keys())


def get_reward_function_names():
    """Get all registered reward function names."""
    return sorted(ALL_REWARD_FUNCTIONS.keys())


def get_observation_space(name: str):
    """Get an instantiated observation space object by name."""
    if name not in ALL_OBSERVATION_SPACES:
        raise ValueError(
            f"Unknown observation space '{name}' (available: {get_observation_space_names()})"
        )
    return ALL_OBSERVATION_SPACES[name]()


def get_action_space(name: str):
    """Get an instantiated action space object by name."""
    if name not in ALL_ACTION_SPACES:
        raise ValueError(
            f"Unknown action space '{name}' (available: {get_action_space_names()})"
        )
    return ALL_ACTION_SPACES[name]()


def get_reward_function(name: str):
    """Get an instantiated reward function object by name."""
    if name not in ALL_REWARD_FUNCTIONS:
        raise ValueError(
            f"Unknown reward function '{name}' (available: {get_reward_function_names()})"
        )
    return ALL_REWARD_FUNCTIONS[name]()


def consistent_pokemon_order(pokemon):
    """
    Sorts Pokémon alphabetically according to their active species
    name (which would include "formes") in lowercase with special
    characters removed.
    """
    if not pokemon:
        return []
    if isinstance(pokemon[0], Pokemon):
        key = lambda p: pokemon_name(p.species)
    elif isinstance(pokemon[0], str):
        key = lambda p: pokemon_name(p)
    elif isinstance(pokemon[0], UniversalPokemon):
        key = lambda p: pokemon_name(p.name)
    elif isinstance(pokemon[0], ReplayPokemon):
        key = lambda p: pokemon_name(p.name)
    else:
        raise ValueError(
            f"Unrecognized `pokemon` list format of type {type(pokemon)}: {pokemon}"
        )
    return sorted(pokemon, key=key)


def consistent_move_order(moves):
    """
    Sorts moves alphabetically according to their name in lowercase
    with special characters removed.
    """
    if not moves:
        return []
    if isinstance(moves[0], Move):
        key = lambda m: move_name(m.id)
    elif isinstance(moves[0], str):
        key = lambda m: move_name(m)
    elif isinstance(moves[0], UniversalMove):
        key = lambda m: move_name(m.name)
    elif isinstance(moves[0], ReplayMove):
        key = lambda m: move_name(m.name)
    else:
        raise ValueError(
            f"Unrecognized `moves` list format of type {type(moves[0])}: {moves}"
        )
    return sorted(moves, key=key)


@dataclass
class UniversalMove:
    """An object that represents a move in the backend-agnostic "Universal" format.

    Rarely constructed directly. Instead, use one of the following factory methods:
        - UniversalMove.from_Move(move) - when move is from poke-env
        - UniversalMove.from_ReplayMove(move) - when move is from the replay parser
        - UniversalMove.from_dict(data) - when move is a dict from the parsed replay
            dataset on disk
    """

    name: str
    move_type: str
    category: str
    base_power: int
    accuracy: float
    priority: int
    current_pp: int
    max_pp: int

    @classmethod
    def blank_move(cls):
        return cls(
            name="nomove",
            move_type="nomove",
            category="nomove",
            base_power=0,
            accuracy=1.0,
            priority=0,
            current_pp=0,
            max_pp=0,
        )

    @classmethod
    def from_ReplayMove(cls, move: Optional[ReplayMove]):
        universal_move = cls.from_Move(move)
        if move is not None:
            universal_move.current_pp = move.pp
            universal_move.max_pp = move.maximum_pp
        return universal_move

    @classmethod
    def from_Move(cls, move: Optional[Move]):
        if move is None:
            return cls.blank_move()
        assert isinstance(move, Move)
        return cls(
            name=move_name(move.id),
            category=clean_name(move.category.name),
            base_power=move.base_power,
            move_type=clean_name(move.type.name),
            priority=move.priority,
            accuracy=move.accuracy,
            current_pp=move.current_pp,
            max_pp=move.max_pp,
        )


@dataclass
class UniversalPokemon:
    """An object that represents a pokemon in the backend-agnostic "Universal" format.

    Rarely constructed directly. Instead, use one of the following factory methods:
        - UniversalPokemon.from_ReplayPokemon(pokemon) - when pokemon is from the replay
            parser
        - UniversalPokemon.from_Pokemon(pokemon) - when pokemon is from poke-env
        - UniversalPokemon.from_dict(data) - when pokemon is a dict from the parsed
            replay dataset on disk
    """

    name: str
    hp_pct: float
    types: str
    item: str
    ability: str
    lvl: int
    status: str
    effect: str
    moves: list[UniversalMove]

    atk_boost: int
    spa_boost: int
    def_boost: int
    spd_boost: int
    spe_boost: int
    accuracy_boost: int
    evasion_boost: int

    base_atk: int
    base_spa: int
    base_def: int
    base_spd: int
    base_spe: int
    base_hp: int

    # version-specific
    tera_type: str
    base_species: str

    @staticmethod
    def universal_items(item_rep: Optional[str | ReplayNothing]) -> str:
        if item_rep is None or item_rep == "unknown_item":
            item_str = "unknownitem"
        elif item_rep == ReplayNothing.NO_ITEM:
            item_str = item_rep.name
        elif isinstance(item_rep, str) and item_rep.strip() in {
            "",
            "No Item",
            "noitem",
        }:
            item_str = ReplayNothing.NO_ITEM.name
        else:
            item_str = item_rep
        return clean_no_numbers(item_str)

    @staticmethod
    def universal_abilities(ability_rep: Optional[str | ReplayNothing]) -> str:
        if ability_rep is None or ability_rep == "unknown_ability":
            ability_str = "unknownability"
        elif ability_rep == ReplayNothing.NO_ABILITY:
            ability_str = ability_rep.name
        elif isinstance(ability_rep, str) and ability_rep.strip() in {
            "",
            "No Ability",
            "noability",
        }:
            ability_str = ReplayNothing.NO_ABILITY.name
        else:
            ability_str = ability_rep
        return clean_no_numbers(ability_str)

    @staticmethod
    def universal_effects(effect: Optional[Effect]) -> str:
        if not effect:
            return "noeffect"
        return clean_no_numbers(effect.name)

    @staticmethod
    def universal_status(status_rep: Status | ReplayNothing) -> str:
        if status_rep is None or status_rep == ReplayNothing.NO_STATUS:
            return "nostatus"
        assert isinstance(status_rep, Status)
        return clean_no_numbers(status_rep.name)

    @staticmethod
    def universal_types(type_rep: list, force_two: bool = True) -> str:
        if force_two:
            while len(type_rep) < 2:
                type_rep.append(None)
        type_strs = []
        for type in type_rep:
            if type is None or type == ReplayNothing.NO_TERA_TYPE:
                type_strs.append("notype")
            elif isinstance(type, PokemonType):
                type_strs.append(clean_name(type.name))
            elif isinstance(type, str):
                type_strs.append(clean_name(type))
        return " ".join(sorted(type_strs))

    @classmethod
    def from_ReplayPokemon(cls, pokemon: ReplayPokemon):
        assert isinstance(pokemon, ReplayPokemon)
        # NOTE: new replay parser lets movesets go over 4 for some dittos... so temporary fix here
        moves = [
            UniversalMove.from_ReplayMove(move)
            for move in pokemon.moves.values()
            if move is not None
        ][:4]
        stats = {f"base_{stat}": val for stat, val in pokemon.base_stats.items()}
        boosts = {
            f"{stat}boost": getattr(pokemon.boosts, stat)
            for stat in pokemon.boosts.stat_attrs
        }
        if pokemon.effects:
            most_recent_effect = min(pokemon.effects.keys(), key=pokemon.effects.get)
        else:
            most_recent_effect = None
        return cls(
            name=pokemon_name(pokemon.name),
            base_species=pokemon_name(pokemon.had_name),
            hp_pct=float(pokemon.current_hp) / pokemon.max_hp,
            types=cls.universal_types(pokemon.type),
            tera_type=cls.universal_types([pokemon.tera_type], force_two=False),
            item=cls.universal_items(pokemon.active_item),
            ability=cls.universal_abilities(pokemon.active_ability),
            lvl=pokemon.lvl,
            status=cls.universal_status(pokemon.status),
            effect=cls.universal_effects(most_recent_effect),
            moves=moves,
            **(boosts | stats),
        )

    @classmethod
    def from_Pokemon(cls, pokemon: Pokemon):
        # do not use Battle.available_moves
        # NOTE: new replay parser lets movesets go over 4 for some dittos... so temporary fix here
        moves = [UniversalMove.from_Move(move) for move in pokemon.moves.values()][:4]
        boosts = {f"{stat}_boost": boost for stat, boost in pokemon.boosts.items()}
        stats = {f"base_{stat}": val for stat, val in pokemon.base_stats.items()}
        if pokemon.effects:
            most_recent_effect = min(pokemon.effects.keys(), key=pokemon.effects.get)
        else:
            most_recent_effect = None
        return cls(
            name=pokemon_name(pokemon.species),
            base_species=pokemon_name(pokemon.base_species),
            hp_pct=float(pokemon.current_hp_fraction),
            types=cls.universal_types(pokemon.types),
            tera_type=cls.universal_types([pokemon.tera_type], force_two=False),
            item=cls.universal_items(pokemon.item),
            ability=cls.universal_abilities(pokemon.ability),
            lvl=pokemon.level,
            status=cls.universal_status(pokemon.status),
            effect=cls.universal_effects(most_recent_effect),
            moves=moves,
            **(boosts | stats),
        )

    @classmethod
    def from_dict(cls, data: dict):
        # NOTE: new replay parser lets movesets go over 4 for some dittos... so temporary fix here
        data["moves"] = [UniversalMove(**m) for m in data["moves"][:4]]
        if "tera_type" not in data:
            # if missing --> old version of the dataset --> gen 1-4 --> no tera
            data["tera_type"] = cls.universal_types([None], force_two=False)
        if "base_species" not in data:
            # if missing --> old version of the dataset --> gen 1-4 --> we can get away with this
            data["base_species"] = data["name"].split("-")[0].strip()
        return cls(**data)

    @staticmethod
    def metamon_to_poke_env(pokemon: ReplayPokemon, is_active: bool) -> Pokemon:
        """
        Straight-through conversion from metamon replay parser Pokemon object
        to poke-env Pokemon object. An ugly alternative to adding a
        `update_from_metamon` equivalent in poke-env.Pokemon. Used by metamon
        battle backend.
        """
        if pokemon is None:
            return None
        p = Pokemon(gen=pokemon.gen)
        p._base_stats = pokemon.base_stats
        p._type_1 = PokemonType.from_name(pokemon.type[0])
        p._type_2 = (
            PokemonType.from_name(pokemon.type[1]) if len(pokemon.type) > 1 else None
        )
        p._ability = pokemon.had_ability
        p._level = pokemon.lvl
        p._max_hp = pokemon.max_hp
        p._moves = {m.lookup_name: m for m in pokemon.moves.values()}
        for m in p._moves.values():
            m.set_pp(m.pp)
        p._name = pokemon.nickname
        p._species = clean_name(pokemon.name)
        p._active = is_active
        p._boosts = pokemon.boosts.to_dict()
        p._current_hp = pokemon.current_hp
        p._effects = pokemon.effects
        p._item = pokemon.active_item
        p._status = pokemon.status
        p._temporary_ability = pokemon.active_ability
        p._previous_move = pokemon.last_used_move
        p._terastallized_type = (
            PokemonType.from_name(pokemon.tera_type) if pokemon.tera_type else None
        )
        return p


@dataclass
class UniversalState:
    """An object that represents a state in the backend-agnostic "Universal" format.

    Rarely constructed directly. Instead, use one of the following factory methods:
        - UniversalState.from_ReplayState(state) - when coming from a ReplayState
            object in the replay parser
        - UniversalState.from_Battle(battle) - when coming from a Battle object in the
            online poke-env
        - UniversalState.from_dict(data) - when state is a dict from the parsed replay
            dataset on disk
    """

    format: str
    player_active_pokemon: UniversalPokemon
    opponent_active_pokemon: UniversalPokemon
    available_switches: List[UniversalPokemon]
    player_prev_move: UniversalMove
    opponent_prev_move: UniversalMove
    opponents_remaining: int
    player_conditions: str
    opponent_conditions: str
    weather: str
    battle_field: str
    forced_switch: bool
    battle_won: bool
    battle_lost: bool

    # version-specific
    can_tera: bool  # added v3-beta
    opponent_teampreview: List[str]  # added v3

    @property
    def agent_format(self) -> str:
        """The format as presented to the agent, with Showdown variants normalized."""
        return format_for_agent(self.format)

    @staticmethod
    def universal_conditions(condition_rep) -> str:
        if not condition_rep:
            return "noconditions"
        most_recent = max(condition_rep.keys(), key=condition_rep.get)
        assert isinstance(most_recent, SideCondition)
        return clean_no_numbers(most_recent.name)

    @staticmethod
    def universal_field(field_rep) -> str:
        if not field_rep:
            return "nofield"
        most_recent = max(field_rep.keys(), key=field_rep.get)
        assert isinstance(most_recent, Field)
        return clean_no_numbers(most_recent.name)

    @staticmethod
    def universal_weather(weather_rep) -> str:
        if not weather_rep or weather_rep == ReplayNothing.NO_WEATHER:
            return "noweather"
        if isinstance(weather_rep, dict):
            weather_rep = list(weather_rep.keys())[0]
        return clean_no_numbers(weather_rep.name)

    # fmt: off
    @classmethod
    def from_ReplayState(cls, state: ReplayState):
        assert isinstance(state, ReplayState)
        format = re.sub(r"\[|\]| ", "", state.format).lower()
        active = UniversalPokemon.from_ReplayPokemon(state.active_pokemon)
        opponent = UniversalPokemon.from_ReplayPokemon(state.opponent_active_pokemon)
        switches = [UniversalPokemon.from_ReplayPokemon(p) for p in state.available_switches]
        opponents_remaining = 6 - sum(p.status == Status.FNT for p in state.opponent_team if p is not None)
        opponent_teampreview = [pokemon_name(p.had_name) for p in state.opponent_teampreview]
        return cls(
            format=format,
            player_active_pokemon=active,
            opponent_active_pokemon=opponent,
            available_switches=switches,
            player_prev_move=UniversalMove.from_ReplayMove(state.player_prev_move),
            opponent_prev_move=UniversalMove.from_ReplayMove(state.opponent_prev_move),
            player_conditions=cls.universal_conditions(state.player_conditions),
            opponent_conditions=cls.universal_conditions(state.opponent_conditions),
            weather=cls.universal_weather(state.weather),
            battle_field=cls.universal_field(state.battle_field),
            forced_switch=state.force_switch,
            opponents_remaining=opponents_remaining,
            battle_won=state.battle_won,
            battle_lost=state.battle_lost,
            can_tera=state.can_tera,
            opponent_teampreview=opponent_teampreview,
        )

    @classmethod
    def from_Battle(cls, battle: Battle):
        # do not use Battle.available_switches or Battle.available_moves
        format = battle.battle_tag.split("-")[1]
        weather = cls.universal_weather(battle.weather)
        battle_field = cls.universal_field(battle.fields)
        player_conditions = cls.universal_conditions(battle.side_conditions)
        opponent_conditions = cls.universal_conditions(battle.opponent_side_conditions)
        active = UniversalPokemon.from_Pokemon(battle.active_pokemon)
        opponent = UniversalPokemon.from_Pokemon(battle.opponent_active_pokemon)
        if battle.reviving:
            possible_switches = [p for p in battle.team.values() if p.fainted and not p.active]
        else:
            possible_switches = [p for p in battle.team.values() if not p.fainted and not p.active]
        switches = [UniversalPokemon.from_Pokemon(p) for p in possible_switches]
        player_prev_move = UniversalMove.from_Move(battle.active_pokemon.previous_move)
        opponent_prev_move = UniversalMove.from_Move(battle.opponent_active_pokemon.previous_move)
        # NOTE: always assumes 6 in the party, and this will probably never change for backwards compat
        opponents_remaining = 6 - sum(p.status == Status.FNT for p in battle.opponent_team.values())
        opponent_teampreview = [pokemon_name(p.base_species) for p in battle.teampreview_opponent_team if p is not None]
        force_switch = battle.force_switch
        if isinstance(force_switch, list):
            force_switch = force_switch[0]

        return cls(
            format=format,
            player_active_pokemon=active,
            opponent_active_pokemon=opponent,
            available_switches=switches,
            player_prev_move=player_prev_move,
            opponent_prev_move=opponent_prev_move,
            player_conditions=player_conditions,
            opponent_conditions=opponent_conditions,
            weather=weather,
            battle_field=battle_field,
            forced_switch=force_switch,
            battle_won=battle.won if battle.won else False,
            battle_lost=battle.lost if battle.lost else False,
            opponents_remaining=opponents_remaining,
            can_tera=battle.can_tera is not None,
            opponent_teampreview=opponent_teampreview,
        )
    # fmt: on

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        # convert nested Pokemon objects
        data["player_active_pokemon"] = UniversalPokemon.from_dict(
            data["player_active_pokemon"]
        )
        data["opponent_active_pokemon"] = UniversalPokemon.from_dict(
            data["opponent_active_pokemon"]
        )
        data["available_switches"] = [
            UniversalPokemon.from_dict(p) for p in data["available_switches"]
        ]
        # convert nested Move objects
        data["player_prev_move"] = UniversalMove(**data["player_prev_move"])
        data["opponent_prev_move"] = UniversalMove(**data["opponent_prev_move"])

        if "can_tera" not in data:
            # backwards compat (if it's missing; it's an old version of the dataset
            # --> gen 1-4 --> no tera)
            data["can_tera"] = False

        if "opponent_teampreview" not in data:
            # backwards compat (if it's missing; it's an old version of the dataset
            # --> gen 1-4 --> no teampreview)
            data["opponent_teampreview"] = []

        return cls(**data)


class UniversalAction:
    def __init__(self, action_idx: int):
        self.action_idx = action_idx

    @property
    def missing(self) -> bool:
        return self.action_idx == -1

    def __eq__(self, other: "UniversalAction") -> bool:
        return self.action_idx == other.action_idx

    def __repr__(self):
        return str(self.action_idx)

    def __hash__(self):
        return hash(self.action_idx)

    @classmethod
    def from_ReplayAction(
        cls, state: ReplayState, action: ReplayAction
    ) -> Optional["UniversalAction"]:
        action_idx = None
        if action is None or (action.name is None and action.is_tera):
            # action was never revealed
            # (or tera animation was shown but the rest of the action was never revealed)
            action_idx = -1
        elif action.is_noop:
            assert action.name == "Recharge"
            action_idx = 0
        elif action.name in {"Struggle", "Fight"}:
            action_idx = 0
        elif action.is_switch or action.is_revival:
            for switch_idx, available_switch in enumerate(
                consistent_pokemon_order(state.available_switches)
            ):
                if available_switch.unique_id == action.target.unique_id:
                    action_idx = 4 + switch_idx
                    break
        else:
            move_options = list(state.active_pokemon.moves.values())
            for move_idx, move in enumerate(consistent_move_order(move_options)):
                if move.name == action.name:
                    action_idx = move_idx
                    if action.is_tera:
                        action_idx += 9
                    break
        if action_idx is None:
            return None
        return cls(action_idx)

    @classmethod
    def maybe_valid_actions(cls, state: UniversalState) -> Set["UniversalAction"]:
        legal = []
        if not state.forced_switch:
            moves = len(state.player_active_pokemon.moves)
            legal.extend(range(moves))
            if state.can_tera:
                legal.extend(range(9, 9 + moves))
        legal.extend(range(4, 4 + len(state.available_switches)))
        return set(UniversalAction(action_idx=action_idx) for action_idx in legal)

    @classmethod
    def definitely_valid_actions(
        cls, state: UniversalState, battle: Battle
    ) -> Set["UniversalAction"]:
        maybe_legal = cls.maybe_valid_actions(state)
        definitely_legal = set()
        for action in maybe_legal:
            order = cls.action_idx_to_BattleOrder(battle, action_idx=action.action_idx)
            if order is not None:
                definitely_legal.add(action)
        return definitely_legal

    @staticmethod
    def action_idx_to_BattleOrder(
        battle: Battle, action_idx: int
    ) -> Optional[BattleOrder]:
        valid_moves = {m.id for m in battle.available_moves}
        if valid_moves == {"recharge"}:
            # there is only one option; take it so it doesn't count as an invalid action
            return Player.create_order(battle.available_moves[0])
        elif valid_moves == {"struggle"}:
            # override the options so that all the move indices are struggle but switches are valid.
            # note that the replay version sets every Struggle in the dataset to index 0, so this
            # is giving a little room for error.
            move_options = [battle.available_moves[0]] * 4
        elif "fight" in valid_moves:
            # new in ~march 2026: a "fight" button in gen1, which tells you your only
            # options are to "fight" or potentially switch. Similar to struggle, the agent
            # will see its regular 4 moves and switches (if applicable) but all of the moves
            # will map to clicking "fight".
            move_options = [battle.available_moves[0]] * 4
        else:
            # standard: pick from the active pokemon's moves
            move_options = consistent_move_order(
                list(battle.active_pokemon.moves.values())
            )

        valid_switches = {p.name for p in battle.available_switches}
        if not battle.reviving:
            switch_options = consistent_pokemon_order(
                [
                    p
                    for p in list(battle.team.values())
                    if not p.fainted and not p.active
                ]
            )
        else:
            switch_options = consistent_pokemon_order(
                [p for p in list(battle.team.values()) if p.fainted and not p.active]
            )

        wants_tera = False
        can_tera = battle.can_tera is not None
        if action_idx >= 9:
            wants_tera = True
            action_idx -= 9

        if action_idx <= 3 and not battle.force_switch:
            # pick one of up to 4 available moves
            if action_idx < len(move_options):
                selected_move = move_options[action_idx]
                if selected_move.id in valid_moves:
                    # NOTE: giving the player a little help on invalid tera requests here
                    order = Player.create_order(
                        selected_move, terastallize=wants_tera and can_tera
                    )
                    return order
        if 4 <= action_idx <= 8:
            # switch to one of up to 5 alternative pokemon
            action_idx -= 4
            if action_idx < len(switch_options):
                selected_switch = switch_options[action_idx]
                if selected_switch.name in valid_switches:
                    order = Player.create_order(selected_switch)
                    return order

        # Q: "what happens when we pick an invalid action? (order = None)"
        # A : up to env's `on_invalid_order` to pick one
        return None

    def to_BattleOrder(self, battle: Battle) -> Optional[BattleOrder]:
        return UniversalAction.action_idx_to_BattleOrder(
            battle, action_idx=self.action_idx
        )


class ActionSpace(ABC):

    @property
    @abstractmethod
    def gym_space(self) -> gym.spaces.Discrete:
        raise NotImplementedError

    @abstractmethod
    def agent_output_to_action(
        self, state: UniversalState, agent_output: Any
    ) -> UniversalAction:
        raise NotImplementedError

    @abstractmethod
    def action_to_agent_output(
        self, state: UniversalState, action: UniversalAction
    ) -> Any:
        raise NotImplementedError


@register_action_space()
class DefaultActionSpace(ActionSpace):

    @property
    def gym_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(13)

    def agent_output_to_action(
        self, state: UniversalState, agent_output: int
    ) -> UniversalAction:
        return UniversalAction(action_idx=int(agent_output))

    def action_to_agent_output(
        self, state: UniversalState, action: UniversalAction
    ) -> int:
        return action.action_idx


@register_action_space()
class MinimalActionSpace(DefaultActionSpace):

    @property
    def gym_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(9)

    def agent_output_to_action(
        self, state: UniversalState, agent_output: int
    ) -> UniversalAction:
        action_idx = int(agent_output)
        if action_idx >= 9:
            # map all gimmick move actions to regular move actions
            action_idx -= 9
        return UniversalAction(action_idx=action_idx)

    def action_to_agent_output(
        self, state: UniversalState, action: UniversalAction
    ) -> int:
        if action.action_idx >= 9:
            # map all gimmick move actions to regular move actions
            action.action_idx -= 9
        return action.action_idx


class RewardFunction(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def __name__(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def __call__(self, last_state: UniversalState, state: UniversalState) -> float:
        raise NotImplementedError


@register_reward_function()
class DefaultShapedReward(RewardFunction):
    """The default reward function used by the paper.

    See the Appendix for a full description.
    """

    def __call__(self, last_state: UniversalState, state: UniversalState) -> float:
        active_now = state.player_active_pokemon
        active_prev = None
        for pokemon in [
            last_state.player_active_pokemon,
            *last_state.available_switches,
        ]:
            if pokemon.base_species == active_now.base_species:
                active_prev = pokemon
                break
        if active_prev is None:
            # this used to trigger a crash, but is now allowed because revival blessing in gen9 will break it
            hp_gain = 0.0
            took_status = 0.0
        else:
            hp_gain = active_now.hp_pct - active_prev.hp_pct
            took_status = float(
                active_now.status != "nostatus" and active_prev.status == "nostatus"
            )
        opp_now = state.opponent_active_pokemon
        opp_prev = last_state.opponent_active_pokemon
        if opp_now.base_species == opp_prev.base_species:
            damage_done = opp_prev.hp_pct - opp_now.hp_pct
            gave_status = float(
                opp_now.status != "nostatus" and opp_prev.status == "nostatus"
            )
        else:
            damage_done, gave_status = 0.0, 0.0
        lost_pokemon = float(
            len(last_state.available_switches) > len(state.available_switches)
        )
        removed_pokemon = float(
            last_state.opponents_remaining > state.opponents_remaining
        )
        if state.battle_won:
            victory = 1.0
        elif state.battle_lost:
            victory = -1.0
        else:
            victory = 0.0
        reward = (
            1.0 * (damage_done + hp_gain)
            + 0.5 * (gave_status - took_status)
            + 1.0 * (removed_pokemon - lost_pokemon)
            + 100.0 * victory
        )
        return reward


@register_reward_function()
class AggressiveShapedReward(RewardFunction):
    """
    Edits the default reward function so that the sparse reward is +200 for winning / +0 for losing.
    This discourages the original policies' annoying tendency to cling to lost positions.
    Gamble and try to win! Also removes shaping for status conditions.
    """

    def __call__(self, last_state: UniversalState, state: UniversalState) -> float:
        active_now = state.player_active_pokemon
        active_prev = None
        for pokemon in [
            last_state.player_active_pokemon,
            *last_state.available_switches,
        ]:
            if pokemon.base_species == active_now.base_species:
                active_prev = pokemon
                break
        hp_gain = 0.0 if active_prev is None else active_now.hp_pct - active_prev.hp_pct
        opp_now = state.opponent_active_pokemon
        opp_prev = last_state.opponent_active_pokemon
        if opp_now.base_species == opp_prev.base_species:
            damage_done = opp_prev.hp_pct - opp_now.hp_pct
        else:
            damage_done = 0.0
        lost_pokemon = float(
            len(last_state.available_switches) > len(state.available_switches)
        )
        removed_pokemon = float(
            last_state.opponents_remaining > state.opponents_remaining
        )
        victory = float(state.battle_won)
        reward = (
            1.0 * (damage_done + hp_gain)
            + 2.0 * (removed_pokemon - lost_pokemon)
            + 200.0 * victory
        )
        return reward


@register_reward_function()
class BinaryReward(RewardFunction):
    """A sparse variant of the default reward function."""

    def __call__(self, last_state: UniversalState, state: UniversalState) -> float:
        if state.battle_won:
            return 100.0
        elif state.battle_lost:
            return -100.0
        return 0.0


class ObservationSpace(ABC):
    def __init__(self, *args, **kwargs):
        self.reset()
        pass

    def __name__(self) -> str:
        return self.__class__.__name__

    def reset(self):
        """Clear any internal state (between battles)."""
        pass

    @property
    def tokenizable(self) -> dict[str, int]:
        """Return a dictionary of tokenizable keys and their expected (max) length."""
        return {}

    @property
    @abstractmethod
    def gym_space(self) -> gym.spaces.Space:
        """Return the observation space for this observation type."""
        raise NotImplementedError

    @abstractmethod
    def state_to_obs(self, state: UniversalState):
        raise NotImplementedError

    def __call__(self, state: UniversalState) -> dict[str, np.ndarray]:
        obs = self.state_to_obs(state)
        return obs


@register_observation_space()
class DefaultObservationSpace(ObservationSpace):
    """The default observation space used by the paper.

    Observations become a dictionary with two keys:
        - "numbers": A 48-dimensional vector of numerical features
        - "text": A string of text features with inconsistent length, but a consistent
            number of whitespace-separated words.
    """

    @property
    def gym_space(self):
        return gym.spaces.Dict(
            {
                "numbers": gym.spaces.Box(
                    low=-10.0,
                    high=10.0,
                    shape=(48,),
                    dtype=np.float32,
                ),
                "text": gym.spaces.Text(
                    max_length=900,
                    min_length=800,
                    charset=set(string.ascii_lowercase)
                    | set(str(n) for n in range(0, 10))
                    | {"<", ">"},
                ),
            }
        )

    @property
    def tokenizable(self) -> dict[str, int]:
        return {
            "text": 87,
        }

    def _get_move_string_features(self, move: UniversalMove, active: bool) -> list[str]:
        out = [clean_name(move.name)]
        if active:
            out += [clean_name(move.move_type), clean_name(move.category)]
        return out

    def _get_move_pad_string(self, active: bool) -> list[str]:
        out = ["<blank>"]
        if active:
            out += ["<blank>", "<blank>"]
        return out

    def _get_move_numerical_features(
        self, move: UniversalMove, active: bool
    ) -> list[float]:
        if not active:
            return []
        return [move.base_power / 200.0, move.accuracy, move.priority / 5.0]

    def _get_move_pad_numerical(self, active: bool) -> list[float]:
        if not active:
            return []
        return [-2.0] * 3

    def _get_pokemon_string_features(
        self, pokemon: UniversalPokemon, active: bool
    ) -> list[str]:
        out = [pokemon.name, pokemon.item, pokemon.ability]
        if active:
            out += [pokemon.types, pokemon.effect, pokemon.status]
        else:
            out += ["<moveset>"]
            move_num = -1
            for move_num, move in enumerate(consistent_move_order(pokemon.moves)):
                out += self._get_move_string_features(move, active=False)
            while move_num < 3:
                out += self._get_move_pad_string(active=False)
                move_num += 1
        return out

    def _get_opponent_pokemon_string_features(
        self, pokemon: UniversalPokemon, active: bool
    ) -> list[str]:
        return self._get_pokemon_string_features(pokemon, active)

    def _get_pokemon_pad_string(self, active: bool) -> list[str]:
        blanks = 3 + (4 if active else 5)
        return ["<blank>"] * blanks

    def _get_pokemon_numerical_features(
        self, pokemon: UniversalPokemon, active: bool
    ) -> list[float]:
        out = [pokemon.hp_pct]
        if active:
            stat = lambda s: getattr(pokemon, f"base_{s}") / 255.0
            boost = lambda b: getattr(pokemon, f"{b}_boost") / 6.0
            out.append(pokemon.lvl / 100.0)
            out += map(stat, ["atk", "spa", "def", "spd", "spe", "hp"])
            out += map(
                boost, ["atk", "spa", "def", "spd", "spe", "accuracy", "evasion"]
            )
        return out

    def _get_pokemon_pad_numerical(self, active: bool) -> list[float]:
        blanks = 1 + (14 if active else 0)
        return [-2.0] * blanks

    def state_to_obs(self, state: UniversalState) -> dict[str, np.ndarray]:
        player_str = ["<player>"] + self._get_pokemon_string_features(
            state.player_active_pokemon, active=True
        )
        numerical = [
            state.opponents_remaining / 6.0
        ] + self._get_pokemon_numerical_features(
            state.player_active_pokemon, active=True
        )

        # consistent move order
        move_str, move_num = [], -1
        for move_num, move in enumerate(
            consistent_move_order(state.player_active_pokemon.moves)
        ):
            move_str += ["<move>"] + self._get_move_string_features(move, active=True)
            numerical += self._get_move_numerical_features(move, active=True)

        while move_num < 3:
            move_str += ["<move>"] + self._get_move_pad_string(active=True)
            numerical += self._get_move_pad_numerical(active=True)
            move_num += 1

        # consistent switch order
        switch_str, switch_num = [], -1
        for switch_num, switch in enumerate(
            consistent_pokemon_order(state.available_switches)
        ):
            switch_str += ["<switch>"] + self._get_pokemon_string_features(
                switch, active=False
            )
            numerical += self._get_pokemon_numerical_features(switch, active=False)
        while switch_num < 4:
            switch_str += ["<switch>"] + self._get_pokemon_pad_string(active=False)
            numerical += self._get_pokemon_pad_numerical(active=False)
            switch_num += 1

        force_switch = "<forcedswitch>" if state.forced_switch else "<anychoice>"
        opponent_str = ["<opponent>"] + self._get_opponent_pokemon_string_features(
            state.opponent_active_pokemon, active=True
        )
        numerical += self._get_pokemon_numerical_features(
            state.opponent_active_pokemon, active=True
        )
        global_str = ["<conditions>"] + [
            state.weather,
            state.player_conditions,
            state.opponent_conditions,
        ]
        prev_move_str = (
            ["<player_prev>"]
            + self._get_move_string_features(state.player_prev_move, active=False)
            + ["<opp_prev>"]
            + self._get_move_string_features(state.opponent_prev_move, active=False)
        )
        full_text_list = (
            [f"<{state.agent_format}>", force_switch]
            + player_str
            + move_str
            + switch_str
            + opponent_str
            + global_str
            + prev_move_str
        )
        # length should be 85 (type features have 2 words --> final word length of 87)
        text = " ".join(full_text_list)
        text = np.array(text, dtype=np.str_)
        numbers = np.array(numerical, dtype=np.float32)
        return {"text": text, "numbers": numbers}


@register_observation_space()
class ExpandedObservationSpace(DefaultObservationSpace):
    """Adds PP, the opponent's revealed party, and edge case sleep/freeze flags to DefaultObservationSpace.

    The DefaultObservationSpace used by the paper makes Pokémon more long-term-memory-intensive
    than it strictly needs to be:

    1. Sleep/freeze clause relies on remembering our move and the opponent active Pokémon's status
        at previous timesteps.

    2. PP counts can only be inferred by recalling prev_move features at previous timesteps.

    3. The opponent's full team must be inferred from recalling the active Pokémon at previous
        timesteps.

    This observation space moves some of that information into every timestep. Also adds tera types for gen 9.
    """

    def reset(self):
        # reset the history-dependent features at the start of each battle
        self.any_opponent_asleep = False
        self.any_opponent_frozen = False
        self.revealed_opponents = set()

    @property
    def gym_space(self):
        base_space = super().gym_space
        base_space["numbers"] = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            # adds 4 PP features + 2 sleep/freeze flags + 1 can_tera flag
            shape=(48 + 7,),
            dtype=np.float32,
        )
        return base_space

    @property
    def tokenizable(self) -> dict[str, int]:
        # adds 6 new tokens for the revealed party
        # adds 6 new tokens for the tera types of our party, 1 for the opponent
        return {"text": 87 + 13}

    def _get_pokemon_string_features(
        self, pokemon: UniversalPokemon, active: bool
    ) -> list[str]:
        base = super()._get_pokemon_string_features(pokemon, active)
        base.append(pokemon.tera_type)
        return base

    def _get_pokemon_pad_string(self, active: bool) -> list[str]:
        blanks = 4 + (4 if active else 5)
        return ["<blank>"] * blanks

    def _get_move_numerical_features(
        self, move: UniversalMove, active: np.bool
    ) -> list[float]:
        out = super()._get_move_numerical_features(move, active)
        if active:
            pp_ratio = move.current_pp / move.max_pp
            # there's a reason the original obs space doesn't have PP counts ---
            # they are not accurate in replays. Compromise by discretizing to
            # "low pp" warnings that would minimize off-by-one shift:
            pp_warning = (pp_ratio >= 0.5) + (pp_ratio >= 0.25) + (pp_ratio > 0)
            out.append(float(pp_warning))
        return out

    def _get_move_pad_numerical(self, active: bool) -> list[float]:
        if not active:
            return []
        return [-2.0] * 4

    def state_to_obs(self, state: UniversalState):
        # get default observation + PP features
        obs = super().state_to_obs(state)

        opponent = state.opponent_active_pokemon
        # (sleep/freeze clause only activates when *we* put the opponent to sleep/freeze,
        # which is not what's being tracked here, but this covers the main failure case
        # and the subtlety has been learnable without this feature.)
        self.any_opponent_asleep |= opponent.status == "slp"
        self.any_opponent_frozen |= opponent.status == "frz"
        new_features = [
            self.any_opponent_asleep,
            self.any_opponent_frozen,
            state.can_tera,
        ]
        obs["numbers"] = np.concatenate([obs["numbers"], new_features])

        # add a list of revealed opponents padded to length 6 while reusing
        # the existing <blank> token to avoid making a new vocabulary.
        self.revealed_opponents.add(opponent.base_species)
        revealed = [opp_name for opp_name in sorted(self.revealed_opponents)]
        while len(revealed) < 6:
            revealed.append("<blank>")
        obs["text"] = np.array(
            obs["text"].item() + " " + " ".join(revealed[:6]), dtype=np.str_
        )
        return obs


@register_observation_space()
class TeamPreviewObservationSpace(ExpandedObservationSpace):

    @property
    def tokenizable(self) -> dict[str, int]:
        # adds 6 new tokens for teampreview
        return {"text": 87 + 13 + 6}

    def state_to_obs(self, state: UniversalState):
        obs = super().state_to_obs(state)
        teampreview = [opp_name for opp_name in sorted(state.opponent_teampreview)]
        while len(teampreview) < 6:
            teampreview.append("<blank>")
        obs["text"] = np.array(
            obs["text"].item() + " " + " ".join(teampreview[:6]), dtype=np.str_
        )
        return obs


@register_observation_space()
class OpponentMoveObservationSpace(TeamPreviewObservationSpace):
    """
    Trades some text tokens to make space for the opponent's revealed moves.
    """

    def _get_move_string_features(self, move: UniversalMove, active: bool) -> list[str]:
        out = [clean_name(move.name)]
        if active:
            # save 4 tokens
            out += [clean_name(move.move_type)]
        return out

    def _get_move_pad_string(self, active: bool) -> list[str]:
        return ["<blank>"] * (2 if active else 1)

    def _get_opponent_pokemon_string_features(
        self, pokemon: UniversalPokemon, active: bool
    ) -> list[str]:
        base = self._get_pokemon_string_features(pokemon, active)
        # add 4 tokens
        moves = ["<blank>"] * 4
        for i, move in enumerate(consistent_move_order(pokemon.moves)[:4]):
            moves[i] = clean_name(move.name)
        return base + moves


@register_observation_space()
class GroupedObservationSpace(ObservationSpace):
    """
    Groups observations by entity for use with a shared Pokemon encoder.

    Unlike DefaultObservationSpace which concatenates all features into single
    "text" and "numbers" arrays, this space outputs separate arrays for each
    Pokemon and a misc array for global state.
    """

    POKEMON_TEXT_LEN = 12  # name, item, ability, tera, types×2, effect, status, moves×4
    POKEMON_NUM_LEN = 31  # hp, lvl, stats×6, boosts×7, (bp, acc, pri, pp)×4
    MISC_TEXT_LEN = (
        20  # format, switch, weather, field, conds×2, prev×2, revealed×6, preview×6
    )
    MISC_NUM_LEN = 4  # opp_remaining, sleep, freeze, can_tera
    NUM_SWITCHES = 5

    def reset(self):
        self.any_opponent_asleep = False
        self.any_opponent_frozen = False
        self.revealed_opponents = set()

    @property
    def tokenizable(self) -> dict[str, int]:
        return {
            "text_active_pokemon": self.POKEMON_TEXT_LEN,
            "text_switch_0": self.POKEMON_TEXT_LEN,
            "text_switch_1": self.POKEMON_TEXT_LEN,
            "text_switch_2": self.POKEMON_TEXT_LEN,
            "text_switch_3": self.POKEMON_TEXT_LEN,
            "text_switch_4": self.POKEMON_TEXT_LEN,
            "text_opponent_active_pokemon": self.POKEMON_TEXT_LEN,
            "text_misc": self.MISC_TEXT_LEN,
        }

    @property
    def gym_space(self) -> gym.spaces.Dict:
        spaces = {}
        for key in self.tokenizable:
            spaces[key] = gym.spaces.Text(max_length=500, min_length=0)
        spaces["numbers_active_pokemon"] = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(self.POKEMON_NUM_LEN,), dtype=np.float32
        )
        for i in range(self.NUM_SWITCHES):
            spaces[f"numbers_switch_{i}"] = gym.spaces.Box(
                low=-10.0, high=10.0, shape=(self.POKEMON_NUM_LEN,), dtype=np.float32
            )
        spaces["numbers_opponent_active_pokemon"] = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(self.POKEMON_NUM_LEN,), dtype=np.float32
        )
        spaces["numbers_misc"] = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(self.MISC_NUM_LEN,), dtype=np.float32
        )
        return gym.spaces.Dict(spaces)

    def _get_universal_pokemon_text(
        self, pokemon: UniversalPokemon, is_active: bool = True
    ) -> list[str]:
        out = [
            pokemon.name,
            pokemon.item,
            pokemon.ability,
            pokemon.tera_type,
        ]
        type_parts = pokemon.types.split()
        out.extend(type_parts[:2] + ["notype"] * (2 - len(type_parts)))
        out.append(pokemon.effect)
        out.append(pokemon.status)
        # (sorted order, padded to 4)
        for move in self._pad_moves(pokemon.moves):
            out.append(clean_name(move.name) if move else "<blank>")

        return out

    def _get_universal_pokemon_numbers(
        self, pokemon: UniversalPokemon, is_active: bool = True
    ) -> list[float]:
        out = [pokemon.hp_pct, pokemon.lvl / 100.0]
        for stat in ("atk", "spa", "def", "spd", "spe", "hp"):
            out.append(getattr(pokemon, f"base_{stat}") / 255.0)
        if is_active:
            for boost in ("atk", "spa", "def", "spd", "spe", "accuracy", "evasion"):
                out.append(getattr(pokemon, f"{boost}_boost") / 6.0)
        else:
            out.extend([0.0] * 7)
        # (sorted order, padded to 4)
        for move in self._pad_moves(pokemon.moves):
            if move:
                pp_ratio = move.current_pp / move.max_pp if move.max_pp > 0 else 0.0
                pp_warning = (pp_ratio >= 0.5) + (pp_ratio >= 0.25) + (pp_ratio > 0)
                out.extend(
                    [
                        move.base_power / 200.0,
                        move.accuracy,
                        move.priority / 5.0,
                        float(pp_warning),
                    ]
                )
            else:
                out.extend([-2.0] * 4)

        return out

    def _pad_moves(
        self, moves: list[UniversalMove], n: int = 4
    ) -> list[Optional[UniversalMove]]:
        sorted_moves = consistent_move_order(moves)[:n]
        return sorted_moves + [None] * (n - len(sorted_moves))

    def _get_blank_pokemon_text(self) -> list[str]:
        # padding for empty switch slots."""
        return ["<blank>"] * self.POKEMON_TEXT_LEN

    def _get_blank_pokemon_numbers(self) -> list[float]:
        # padding for empty switch slots."""
        return [-2.0] * self.POKEMON_NUM_LEN

    def _get_misc_text(self, state: UniversalState) -> list[str]:
        # global text features: format, conditions, prev moves, opponent team
        # note: "nofield" isn't in the tokenizer, so we map it to "<blank>"
        battle_field = (
            state.battle_field if state.battle_field != "nofield" else "<blank>"
        )
        out = [
            f"<{state.agent_format}>",
            "<forcedswitch>" if state.forced_switch else "<anychoice>",
            state.weather,
            battle_field,
            state.player_conditions,
            state.opponent_conditions,
            clean_name(state.player_prev_move.name),
            clean_name(state.opponent_prev_move.name),
        ]
        revealed = sorted(self.revealed_opponents)[:6]
        out.extend(revealed + ["<blank>"] * (6 - len(revealed)))
        teampreview = sorted(state.opponent_teampreview)[:6]
        out.extend(teampreview + ["<blank>"] * (6 - len(teampreview)))
        return out

    def _get_misc_numbers(self, state: UniversalState) -> list[float]:
        # global numeric features
        return [
            state.opponents_remaining / 6.0,
            float(self.any_opponent_asleep),
            float(self.any_opponent_frozen),
            float(state.can_tera),
        ]

    def state_to_obs(self, state: UniversalState) -> dict[str, np.ndarray]:
        obs = {}

        # update history-dependent state tracking
        opponent = state.opponent_active_pokemon
        self.any_opponent_asleep |= opponent.status == "slp"
        self.any_opponent_frozen |= opponent.status == "frz"
        self.revealed_opponents.add(opponent.base_species)

        # player active
        obs["text_active_pokemon"] = self._get_universal_pokemon_text(
            state.player_active_pokemon, is_active=True
        )
        obs["numbers_active_pokemon"] = self._get_universal_pokemon_numbers(
            state.player_active_pokemon, is_active=True
        )

        # reserve team (sorted order, padded to NUM_SWITCHES)
        switches = consistent_pokemon_order(state.available_switches)
        for i in range(self.NUM_SWITCHES):
            if i < len(switches):
                obs[f"text_switch_{i}"] = self._get_universal_pokemon_text(
                    switches[i], is_active=False
                )
                obs[f"numbers_switch_{i}"] = self._get_universal_pokemon_numbers(
                    switches[i], is_active=False
                )
            else:
                obs[f"text_switch_{i}"] = self._get_blank_pokemon_text()
                obs[f"numbers_switch_{i}"] = self._get_blank_pokemon_numbers()

        # opponent active
        obs["text_opponent_active_pokemon"] = self._get_universal_pokemon_text(
            state.opponent_active_pokemon, is_active=True
        )
        obs["numbers_opponent_active_pokemon"] = self._get_universal_pokemon_numbers(
            state.opponent_active_pokemon, is_active=True
        )

        # misc (global state)
        obs["text_misc"] = self._get_misc_text(state)
        obs["numbers_misc"] = self._get_misc_numbers(state)

        # temporary assert checks to verify lengths
        for key in obs:
            if key.startswith("text"):
                expected = (
                    self.MISC_TEXT_LEN if key == "text_misc" else self.POKEMON_TEXT_LEN
                )
                assert (
                    len(obs[key]) == expected
                ), f"{key}: expected {expected}, got {len(obs[key])}"
            else:
                expected = (
                    self.MISC_NUM_LEN if key == "numbers_misc" else self.POKEMON_NUM_LEN
                )
                assert (
                    len(obs[key]) == expected
                ), f"{key}: expected {expected}, got {len(obs[key])}"

        for key in obs:
            if key.startswith("text"):
                obs[key] = np.array(" ".join(obs[key]), dtype=np.str_)
            else:
                obs[key] = np.array(obs[key], dtype=np.float32)

        return obs


@register_observation_space()
class PatchPokeAgentTeraBug(ObservationSpace):
    """
    Intentionally reintroduces a bug in the "pokeagent" backend that has been fixed.
    This allows models that should use the "pokeagent" backend to still make sound decisions
    when using the "metamon" backend by setting all tera_type to "notype".
    """

    def __init__(self, base_obs_space: ObservationSpace):
        self.base_obs_space = base_obs_space
        super().__init__()

    def reset(self):
        self.base_obs_space.reset()

    @property
    def gym_space(self):
        return self.base_obs_space.gym_space

    @property
    def tokenizable(self):
        return self.base_obs_space.tokenizable

    def state_to_obs(self, state: UniversalState):
        patched_state = copy.deepcopy(state)
        # patch player pokemon tera types to "notype"
        patched_state.player_active_pokemon.tera_type = "notype"
        for pokemon in patched_state.available_switches:
            pokemon.tera_type = "notype"
        return self.base_obs_space.state_to_obs(patched_state)


# Register patched versions of observation spaces for PokeAgent Challenge compatibility
@register_observation_space("PAC-ExpandedObservationSpace")
class PACExpandedObservationSpace(PatchPokeAgentTeraBug):
    def __init__(self):
        super().__init__(ExpandedObservationSpace())


@register_observation_space("PAC-TeamPreviewObservationSpace")
class PACTeamPreviewObservationSpace(PatchPokeAgentTeraBug):
    def __init__(self):
        super().__init__(TeamPreviewObservationSpace())


@register_observation_space("PAC-OpponentMoveObservationSpace")
class PACOpponentMoveObservationSpace(PatchPokeAgentTeraBug):
    def __init__(self):
        super().__init__(OpponentMoveObservationSpace())


class TokenizedObservationSpace(ObservationSpace):
    """An observation space that tokenizes specified keys of the default observation space.

    Splits text into whitespace-separated words and runs them through a simple
    vocabulary lookup, which usually has been generated by tracking unique words across
    the entire replay dataset. Useful for turning the text features of the default
    observation space into an array with constant shape.
    """

    def __init__(
        self,
        base_obs_space: ObservationSpace,
        tokenizer: PokemonTokenizer,
    ):
        self.base_obs_space = base_obs_space
        self.tokenizer = tokenizer

    def reset(self):
        self.base_obs_space.reset()

    @property
    def gym_space(self):
        tokenizable = self.base_obs_space.tokenizable
        base_space = copy.deepcopy(self.base_obs_space.gym_space)
        new_space_dict = {
            key: space
            for key, space in base_space.spaces.items()
            if key not in tokenizable
        }
        for tokenizable_key, tokenizable_length in tokenizable.items():
            low_token = min(UNKNOWN_TOKEN, 0)
            high_token = max(UNKNOWN_TOKEN, len(self.tokenizer))
            new_space_dict[f"{tokenizable_key}_tokens"] = gym.spaces.Box(
                low=low_token,
                high=high_token,
                shape=(tokenizable_length,),
                dtype=np.int32,
            )

        return gym.spaces.Dict(new_space_dict)

    def state_to_obs(self, state: UniversalState):
        obs = self.base_obs_space.state_to_obs(state)
        for tokenizable_key in self.base_obs_space.tokenizable.keys():
            base_obs_key = obs.pop(tokenizable_key)
            obs[f"{tokenizable_key}_tokens"] = self.tokenizer.tokenize(
                base_obs_key.tolist()
            )
        return obs
