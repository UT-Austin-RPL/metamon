from functools import lru_cache
import re
from dataclasses import dataclass, asdict
from typing import Optional, List
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
from poke_env.data import to_id_str

import metamon
from metamon.data import DATA_PATH
from metamon.data.replay_dataset.parsed_replays.replay_parser.replay_state import (
    Move as ReplayMove,
    Pokemon as ReplayPokemon,
    Action as ReplayAction,
    ReplayState,
    Nothing as ReplayNothing,
)


def pokemon_name(name: str) -> str:
    return clean_name(name)


def clean_name(name: str) -> str:
    return to_id_str(str(name)).strip()


@lru_cache(2**13)
def clean_no_numbers(name: str) -> str:
    return "".join(char for char in str(name) if char.isalpha()).lower().strip()


def consistent_pokemon_order(pokemon):
    if not pokemon:
        return []
    if isinstance(pokemon[0], Pokemon):
        key = lambda p: clean_name(p.species)
    elif isinstance(pokemon[0], str):
        key = lambda p: clean_name(p)
    elif isinstance(pokemon[0], UniversalPokemon):
        key = lambda p: clean_name(p.name)
    elif isinstance(pokemon[0], ReplayPokemon):
        key = lambda p: clean_name(p.name)
    else:
        raise ValueError(
            f"Unrecognized `pokemon` list format of type {type(pokemon)}: {pokemon}"
        )
    return sorted(pokemon, key=key)


def consistent_move_order(moves):
    if not moves:
        return []
    if isinstance(moves[0], Move):
        key = lambda m: clean_name(m.id)
    elif isinstance(moves[0], str):
        key = lambda m: clean_name(m)
    elif isinstance(moves[0], UniversalMove):
        key = lambda m: clean_name(m.name)
    elif isinstance(moves[0], ReplayMove):
        key = lambda m: clean_name(m.name)
    else:
        raise ValueError(
            f"Unrecognized `moves` list format of type {type(moves[0])}: {moves}"
        )
    return sorted(moves, key=key)


@lru_cache(9)
def hidden_power_reference(gen: int) -> Move:
    # we will map all hidden powers to this move
    return Move(move_id="hiddenpower", gen=gen)


@dataclass
class UniversalMove:
    name: str
    move_type: str
    category: str
    base_power: int
    accuracy: float
    priority: int
    current_pp: int
    max_pp: int

    def __post_init__(self):
        for name, should_be in self.__annotations__.items():
            if not isinstance(self.__dict__[name], should_be):
                actually_is = type(self.__dict__[name])
                raise TypeError(f"UniversalMove `{name}` has type {actually_is}")

    def get_string_features(self, active: bool) -> str:
        out = clean_name(self.name)
        if active:
            out += f" {clean_name(self.move_type)} {clean_name(self.category)}"
        return out

    @staticmethod
    def get_pad_string(active: bool) -> str:
        out = "<blank>"
        if active:
            out += " <blank> <blank>"
        return out

    def get_numerical_features(self, active: bool) -> list[float]:
        if not active:
            return []
        # notably missing PP, which (for now) is too unreliable across replay parser vs. poke-env vs. actual pokemon showdown
        return [self.base_power / 200.0, self.accuracy, self.priority / 5.0]

    @staticmethod
    def get_pad_numerical(active: bool) -> list[float]:
        if not active:
            return []
        return [-2.0] * 3

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
    def from_ReplayMove(cls, move: ReplayMove):
        # ReplayMove overrides Move but has
        # a different pp tracker
        universal_move = cls.from_Move(move)
        if move is not None:
            universal_move.current_pp = move.pp
            universal_move.max_pp = move.maximum_pp
        return universal_move

    @classmethod
    def from_Move(cls, move: Move):
        if move is None:
            return cls.blank_move()
        assert isinstance(move, Move)
        if move.id.startswith("hiddenpower"):
            # we map every hidden power to the typeless version
            # because the types don't show up in replays
            reference = hidden_power_reference(move._gen)
        else:
            reference = move
        return cls(
            name=reference.id,
            category=reference.category.name,
            base_power=reference.base_power,
            move_type=reference.type.name,
            priority=reference.priority,
            accuracy=reference.accuracy,
            # always use `move` for pp tracking
            current_pp=move.current_pp,
            max_pp=move.max_pp,
        )


@dataclass
class UniversalPokemon:
    name: str
    hp_pct: float
    types: list[str]
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

    def __post_init__(self):
        for name, should_be in self.__annotations__.items():
            if should_be in {str, bool, int, float} and not isinstance(
                self.__dict__[name], should_be
            ):
                actually_is = type(self.__dict__[name])
                raise TypeError(f"UniversalPokemon `{name}` has type {actually_is}")

    @classmethod
    def from_dict(cls, data: dict):
        args = data.copy()
        args["moves"] = [UniversalMove(**m) for m in data["moves"]]
        return cls(**args)

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
    def universal_effects(effect_rep: dict[Effect, int]) -> str:
        if not effect_rep:
            return "noeffect"
        most_recent = min(effect_rep.keys(), key=effect_rep.get)
        assert isinstance(most_recent, Effect)
        # get rid of poke-env's effect timing system (Effect.FALLEN5, ... Effect.FALLEN1)
        return clean_no_numbers(most_recent.name)

    @staticmethod
    def universal_status(status_rep: Status | ReplayNothing) -> str:
        if status_rep is None or status_rep == ReplayNothing.NO_STATUS:
            return "nostatus"
        assert isinstance(status_rep, Status)
        return clean_no_numbers(status_rep.name)

    @staticmethod
    def universal_types(type_rep: list) -> str:
        while len(type_rep) < 2:
            type_rep.append(None)

        type_strs = []
        for type in type_rep:
            if type is None:
                type_strs.append("notype")
            elif isinstance(type, PokemonType):
                type_strs.append(clean_name(type.name))
            elif isinstance(type, str):
                type_strs.append(clean_name(type))

        assert len(type_strs) == 2
        return " ".join(sorted(type_strs))

    def get_moveset_string(self) -> str:
        out = ""
        move_num = -1
        for move_num, move in enumerate(consistent_move_order(self.moves)):
            out += f" {move.get_string_features(active=False)}"
        while move_num < 3:
            out += f" {UniversalMove.get_pad_string(active=False)}"
            move_num += 1
        return out.strip()

    def get_string_features(self, active: bool) -> str:
        out = f"{self.name} {self.item} {self.ability}"
        if active:
            out += f" {self.types} {self.effect} {self.status}"
        else:
            out += f" <moveset> {self.get_moveset_string()}"
        return out.strip()

    @staticmethod
    def get_pad_string(active: bool) -> str:
        blanks = 3 + (4 if active else 5)
        return " ".join(["<blank>"] * blanks)

    def get_numerical_features(self, active: bool) -> list[float]:
        out = [self.hp_pct]
        if active:
            stat = lambda s: getattr(self, f"base_{s}") / 255.0
            boost = lambda b: getattr(self, f"{b}_boost") / 6.0
            out.append(self.lvl / 100.0)
            out += map(stat, ["atk", "spa", "def", "spd", "spe", "hp"])
            out += map(
                boost, ["atk", "spa", "def", "spd", "spe", "accuracy", "evasion"]
            )
        return out

    @staticmethod
    def get_pad_numerical(active: bool) -> list[float]:
        blanks = 1 + (14 if active else 0)
        return [-2.0] * blanks

    @classmethod
    def from_ReplayPokemon(cls, pokemon: ReplayPokemon):
        assert isinstance(pokemon, ReplayPokemon)
        moves = [
            UniversalMove.from_ReplayMove(move)
            for move in pokemon.moves.values()
            if move is not None
        ]
        stats = {f"base_{stat}": val for stat, val in pokemon.base_stats.items()}
        boosts = {
            f"{stat}boost": getattr(pokemon.boosts, stat)
            for stat in pokemon.boosts.stat_attrs
        }

        item = cls.universal_items(pokemon.active_item)
        ability = cls.universal_abilities(pokemon.active_ability)
        status = cls.universal_status(pokemon.status)
        effect = cls.universal_effects(pokemon.effects)
        types = cls.universal_types(pokemon.type)

        # TODO: some confusion over whether to use `had_name` or `name`.
        # `name` makes more sense b/c it's updated with form change, but
        # from what I can tell poke-env never adjusts the equiv. `species`
        # prop, which would be more similar to `had_name`.
        name = clean_name(pokemon.had_name)

        return cls(
            name=name,
            hp_pct=float(pokemon.current_hp) / pokemon.max_hp,
            types=types,
            item=item,
            ability=ability,
            lvl=pokemon.lvl,
            status=status,
            effect=effect,
            moves=moves,
            **(boosts | stats),
        )

    @classmethod
    def from_Pokemon(cls, pokemon: Pokemon):
        moves = [UniversalMove.from_Move(move) for move in pokemon.moves.values()]
        boosts = {f"{stat}_boost": boost for stat, boost in pokemon.boosts.items()}
        stats = {f"base_{stat}": val for stat, val in pokemon.base_stats.items()}
        status = cls.universal_status(pokemon.status)
        effect = cls.universal_effects(pokemon.effects)
        item = cls.universal_items(pokemon.item)
        ability = cls.universal_abilities(pokemon.ability)
        types = cls.universal_types(pokemon.types)

        return cls(
            name=clean_name(pokemon.species),
            hp_pct=float(pokemon.current_hp_fraction),
            types=types,
            item=item,
            ability=ability,
            lvl=pokemon.level,
            status=status,
            effect=effect,
            moves=moves,
            **(boosts | stats),
        )


@dataclass
class UniversalState:
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

    def __post_init__(self):
        for name, should_be in self.__annotations__.items():
            if should_be in {
                str,
                bool,
                UniversalPokemon,
                UniversalMove,
            } and not isinstance(self.__dict__[name], should_be):
                actually_is = type(self.__dict__[name])
                raise TypeError(f"UniversalState `{name}` has type {actually_is}")

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
        )

    @classmethod
    def from_Battle(cls, battle: Battle):
        format = battle.battle_tag.split("-")[1]
        weather = cls.universal_weather(battle.weather)
        battle_field = cls.universal_field(battle.fields)
        player_conditions = cls.universal_conditions(battle.side_conditions)
        opponent_conditions = cls.universal_conditions(battle.opponent_side_conditions)
        active = UniversalPokemon.from_Pokemon(battle.active_pokemon)
        opponent = UniversalPokemon.from_Pokemon(battle.opponent_active_pokemon)
        possible_switches = [p for p in battle.team.values() if not p.fainted and not p.active]
        switches = [UniversalPokemon.from_Pokemon(p) for p in possible_switches]
        player_prev_move = UniversalMove.from_Move(battle.active_pokemon.previous_move)
        opponent_prev_move = UniversalMove.from_Move(battle.opponent_active_pokemon.previous_move)
        opponents_remaining = 6 - sum(p.status == Status.FNT for p in battle.opponent_team.values())

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
        )
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        args = data.copy()
        # convert nested Pokemon objects
        args["player_active_pokemon"] = UniversalPokemon.from_dict(data["player_active_pokemon"])
        args["opponent_active_pokemon"] = UniversalPokemon.from_dict(data["opponent_active_pokemon"])
        args["available_switches"] = [UniversalPokemon.from_dict(p) for p in data["available_switches"]]
        # convert nested Move objects
        args["player_prev_move"] = UniversalMove(**args["player_prev_move"])
        args["opponent_prev_move"] = UniversalMove(**args["opponent_prev_move"])
        # create UniversalState from the processed dict
        return cls(**args)


def replaystate_action_to_idx(
    state: ReplayState, action: ReplayAction
) -> Optional[int]:
    # *can* return None, but replay parser will throw an exception if it does.
    action_idx = None
    if action is None or action.is_noop:
        action_idx = -1

    elif action.name == "Struggle":
        action_idx = 0

    elif action.is_switch:
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
                break

    return action_idx


def action_idx_to_battle_order(
    battle: Battle, action_idx: int
) -> Optional[BattleOrder]:
    if action_idx > 8:
        raise ValueError(
            f"Invalid `action_idx` {action_idx}. The global action space is bounded {0, ..., 8}"
        )

    # we'll only submit an action if it's valid, but we pick from a longer list determined
    # by rules that are easier to keep track of elsewhere
    valid_moves = battle.available_moves
    valid_switches = battle.available_switches
    move_options = consistent_move_order(list(battle.active_pokemon.moves.values()))
    switch_options = consistent_pokemon_order(
        [p for p in list(battle.team.values()) if not p.fainted and not p.active]
    )

    order = None
    if action_idx <= 3 and not battle.force_switch:
        # pick one of up to 4 available moves
        if action_idx < len(move_options):
            selected_move = move_options[action_idx]
            if selected_move in valid_moves:
                order = Player.create_order(selected_move)

    elif 4 <= action_idx <= 8:
        # switch to one of up to 5 alternative pokemon
        action_idx -= 4
        if action_idx < len(switch_options):
            selected_switch = switch_options[action_idx]
            if selected_switch in valid_switches:
                order = Player.create_order(selected_switch)

    # Q: "what happens when we pick an invalid action? (order = None)"
    # A : env.ShowdownEnv.on_invalid_order
    return order


class RewardFunction(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def __name__(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def __call__(self, last_state: UniversalState, state: UniversalState) -> float:
        raise NotImplementedError


class DefaultShapedReward(RewardFunction):
    def __call__(self, last_state: UniversalState, state: UniversalState) -> float:
        active_now = state.player_active_pokemon
        active_prev = None
        for pokemon in [
            last_state.player_active_pokemon,
            *last_state.available_switches,
        ]:
            if pokemon.name == active_now.name:
                active_prev = pokemon
                break
        assert active_prev is not None

        hp_gain = active_now.hp_pct - active_prev.hp_pct
        took_status = float(
            active_now.status != "nostatus" and active_prev.status == "nostatus"
        )

        opp_now = state.opponent_active_pokemon
        opp_prev = last_state.opponent_active_pokemon
        if opp_now.name == opp_prev.name:
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


class BinaryReward(RewardFunction):
    def __call__(self, last_state: UniversalState, state: UniversalState) -> float:
        if state.battle_won:
            return 100.0
        elif state.battle_lost:
            return -100.0
        return 0.0


class ObservationSpace(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def __name__(self) -> str:
        return self.__class__.__name__

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
        # Verify observation matches the defined gym space
        if not self.gym_space.contains(obs):
            raise ValueError(
                f"Observation {obs} does not comply with defined gym space {self.gym_space}"
            )
        return obs


class DefaultObservationSpace(ObservationSpace):
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

    def state_to_obs(self, state: UniversalState):
        player_str = (
            f"<player> {state.player_active_pokemon.get_string_features(active=True)}"
        )
        numerical = [
            state.opponents_remaining / 6.0
        ] + state.player_active_pokemon.get_numerical_features(active=True)

        # consistent move order
        move_str, move_num = "", -1
        for move_num, move in enumerate(
            consistent_move_order(state.player_active_pokemon.moves)
        ):
            move_str += f" <move> {move.get_string_features(active=True)}"
            numerical += move.get_numerical_features(active=True)

        while move_num < 3:
            move_str += f" <move> {UniversalMove.get_pad_string(active=True)}"
            numerical += UniversalMove.get_pad_numerical(active=True)
            move_num += 1

        # consistent switch order
        switch_str, switch_num = "", -1
        for switch_num, switch in enumerate(
            consistent_pokemon_order(state.available_switches)
        ):
            switch_str += f" <switch> {switch.get_string_features(active=False)}"
            numerical += switch.get_numerical_features(active=False)
        while switch_num < 4:
            switch_str += f" <switch> {UniversalPokemon.get_pad_string(active=False)}"
            numerical += UniversalPokemon.get_pad_numerical(active=False)
            switch_num += 1

        force_switch = "<forcedswitch>" if state.forced_switch else "<anychoice>"
        opponent_str = f"<opponent> {state.opponent_active_pokemon.get_string_features(active=True)}"
        numerical += state.opponent_active_pokemon.get_numerical_features(active=True)
        global_str = f"<conditions> {state.weather} {state.player_conditions} {state.opponent_conditions}"
        prev_move_str = f"<player_prev> {state.player_prev_move.get_string_features(active=False)} <opp_prev> {state.opponent_prev_move.get_string_features(active=False)}"

        text = np.array(
            f"<{state.format}> {force_switch} {player_str} {move_str.strip()} {switch_str.strip()} {opponent_str} {global_str} {prev_move_str}",
            dtype=np.str_,
        )
        numbers = np.array(numerical, dtype=np.float32)
        return {"text": text, "numbers": numbers}
