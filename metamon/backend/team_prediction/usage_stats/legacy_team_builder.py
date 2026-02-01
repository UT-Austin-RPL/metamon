import random
import datetime
import warnings
from typing import Iterable

from termcolor import colored
import numpy as np

import metamon
from metamon.backend.team_prediction.usage_stats import (
    get_usage_stats,
    DEFAULT_USAGE_RANK,
)
from metamon.backend.team_prediction.usage_stats.constants import (
    HIDDEN_POWER_IVS,
    HIDDEN_POWER_DVS,
    INCOMPATIBLE_MOVES,
)


class PokemonStatsLookupError(KeyError):
    def __init__(self, pokemon: str, format: str):
        message = f"Pokemon name `{pokemon}` could not be found in Smogon Statistics for format `{format}`"
        super().__init__(message)


def weighted_random_choice(data: dict[str, float], n: int) -> list[str]:
    if "Other" in data:
        del data["Other"]
    if "Nothing" in data:
        del data["Nothing"]
    if not data:
        # blank predictions happen for rare pokemon where usages stats don't exist,
        # get mapped to "Nothing" in the replay parser, and eventually trigger a skip
        return [""]
    # random choose four moves from moveset using their percent, should not duplicate
    p = np.array(list(data.values()))
    p = p / np.sum(p)
    n = min(n, len(data))
    return np.random.choice(list(data.keys()), size=n, replace=False, p=p).tolist()


class TeamBuilder:
    def __init__(
        self,
        format: str,
        start_date: datetime.date,
        end_date: datetime.date,
        rank: int = DEFAULT_USAGE_RANK,
        verbose: bool = False,
        remove_banned: bool = False,
    ):
        self.format = format
        self.gen = metamon.backend.format_to_gen(format)
        self.stat = get_usage_stats(
            format,
            start_date,
            end_date,
            rank=rank,
        )
        if remove_banned:
            self.stat.remove_banned_pm()
        self.verbose = verbose
        # use for different hp ivs
        self.isgen2 = format.startswith("gen2")
        self.isgen7 = format.startswith("gen7")

    def check_valid_moves(self, pokemon, moves):
        # TODO: this is probably not needed anymore but further changes will
        # wait until the next backend update....
        hp_flag = False
        if len(set(moves)) < len(moves):
            return False
        for move in moves:
            if "Hidden Power" in move:
                if hp_flag:
                    return False
                hp_flag = True

        if pokemon in INCOMPATIBLE_MOVES[self.format[:4]]:
            for incompatible_move_pair in INCOMPATIBLE_MOVES[self.format[:4]][pokemon]:
                # If the other move is in the moveset, return False
                if len(set(incompatible_move_pair) & set(moves)) > 1:
                    return False
        return True

    def get_valid_moves(
        self, pokemon: str, moves: dict[str, float], selected_moves=None
    ) -> list[str]:
        if selected_moves is None:
            selected_moves = []

        def _remove_incompatible_moves(move):
            del moves[move]
            if pokemon in INCOMPATIBLE_MOVES[self.format[:4]]:
                for incompatible_move_pair in INCOMPATIBLE_MOVES[self.format[:4]][
                    pokemon
                ]:
                    # also delete the incompatible move
                    if move in incompatible_move_pair:
                        for m in incompatible_move_pair:
                            if m in moves:
                                del moves[m]
            if "Hidden Power" in move:
                # delete all other hidden power moves
                for m in list(moves.keys()):
                    if "Hidden Power" in m:
                        del moves[m]

        for move in selected_moves:
            _remove_incompatible_moves(move)
        while len(selected_moves) < 4 and len(moves) > 0:
            move = weighted_random_choice(moves, 1)[0]
            _remove_incompatible_moves(move)
            selected_moves.append(move)
        return selected_moves

    def generate_new_team(self, existing_pokemon=Iterable[str] | str):
        current_team = []

        # fill with completions of existing team
        if isinstance(existing_pokemon, Iterable) and existing_pokemon:
            for p in existing_pokemon:
                try:
                    current_team.append(self.generate_moveset(p))
                except Exception as e:
                    raise e
        elif isinstance(existing_pokemon, str):
            try:
                current_team.append(self.generate_moveset(existing_pokemon))
            except Exception as e:
                raise e

        # fill out the rest of the team by picking related pokemon
        if len(current_team) == 0:
            # hack: we need an uppercase name to kick this off, so pick a random
            # id, then a random teammate from that entry...
            warnings.warn(
                colored(
                    "No existing pokemon provided, picking a random starter!", "red"
                )
            )
            starter_id = random.choice(self.stat.usage)
            starter_name = weighted_random_choice(
                self.stat[starter_id]["teammates"], 1
            )[0]
            current_team.append(self.generate_moveset(starter_name))
        while len(current_team) < 6:
            tries = 0
            while tries < 10:
                starter = random.choice(current_team)["name"]
                next_choice = self.select_teammate(starter, current_team)
                if next_choice and not self.name_in_team(next_choice, current_team):
                    break
                tries += 1
            else:
                raise ValueError(
                    f"Failed to find a valid teammate for team {current_team} after 10 tries"
                )
            current_team.append(self.generate_moveset(next_choice))
        return current_team

    def name_in_team(self, name: str, team: list[dict]) -> bool:
        for p in team:
            if p["name"] == name:
                return True
        return False

    def select_teammate(
        self, pokemon_name: str, current_team: list[dict]
    ) -> str | None:
        poke_stat = self.stat[pokemon_name]
        teammates = poke_stat["teammates"]
        # random choose one from teammates not in current team using their percent, teammates is a dict
        valid_teammates = {
            name: freq
            for name, freq in teammates.items()
            if not self.name_in_team(name, current_team)
        }
        if not valid_teammates:
            return None
        teammate = weighted_random_choice(valid_teammates, 1)[0]
        return teammate

    def generate_moveset(self, pokemon):
        poke_stat = self.stat[pokemon]
        abilities = poke_stat["abilities"]
        items = poke_stat["items"]
        spreads = poke_stat["spreads"]
        moves = poke_stat["moves"]
        selected_moves = self.get_valid_moves(pokemon, moves.copy())
        tera_types = poke_stat["tera_types"]

        ivs = "31/31/31/31/31/31"
        for move in selected_moves:
            if "Hidden Power" in move:
                # parse from format "Hidden Power [type]"
                hp_type = move.split(" ")[-1]
                ivs = HIDDEN_POWER_IVS[hp_type]
                # gen2 use DVs instead of IVs
                if self.isgen2:
                    ivs = HIDDEN_POWER_DVS[hp_type]
                # gen7 hyper training allows any hp
                if self.isgen7:
                    ivs = "31/0/31/31/31/31"
                if self.verbose:
                    print(f"Hidden Power {hp_type} detected, IVs set to {ivs}")
                break

        # Gen 1-2 had no abilities; gen 1 had no held items.
        # Usage stats contain bogus data for these, so we override.
        if self.gen <= 2:
            ability = "No Ability"
        else:
            ability = weighted_random_choice(abilities, 1)[0]
        if self.gen <= 1:
            item = ""
        else:
            item = weighted_random_choice(items, 1)[0]

        return {
            "name": pokemon,
            "ability": ability,
            "item": item,
            "spread": weighted_random_choice(spreads, 1)[0],
            "tera_type": weighted_random_choice(tera_types, 1)[0],
            "IVs": ivs,
            "moves": selected_moves,
        }

    def generate_partial_moveset(
        self,
        pokemon,
        ability=None,
        item=None,
        spread=None,
        ivs=None,
        selected_moves=None,
    ):
        poke_stat = self.stat[pokemon]
        abilities = poke_stat["abilities"]
        items = poke_stat["items"]
        spreads = poke_stat["spreads"]
        moves = poke_stat["moves"]
        if selected_moves is not None and not self.check_valid_moves(
            pokemon, selected_moves
        ):
            print(f"Invalid moves for {pokemon}, regenerating moveset")
            selected_moves = None

        # get valid moves
        valid_moves = self.get_valid_moves(pokemon, moves.copy(), selected_moves)
        assert isinstance(valid_moves, list), "selected_moves should be a list"
        if not ivs:
            ivs = "31/31/31/31/31/31"
            for move in valid_moves:
                if "Hidden Power" in move:
                    # parse from format "Hidden Power [type]"
                    hp_type = move.split(" ")[-1]
                    ivs = HIDDEN_POWER_IVS[hp_type]
                    # gen2 use DVs instead of IVs
                    if self.isgen2:
                        ivs = HIDDEN_POWER_DVS[hp_type]
                    # gen7 hyper training allows any hp
                    if self.isgen7:
                        ivs = "31/0/31/31/31/31"
                    if self.verbose:
                        print(f"Hidden Power {hp_type} detected, IVs set to {ivs}")
                    break

        # Gen 1-2 had no abilities; gen 1 had no held items.
        # Usage stats contain bogus data for these, so we override.
        if self.gen <= 2:
            ability = "No Ability"
        elif not ability:
            ability = weighted_random_choice(abilities, 1)[0]

        if self.gen <= 1:
            item = ""
        elif not item:
            item = weighted_random_choice(items, 1)[0]

        if not spread:
            spread = weighted_random_choice(spreads, 1)[0]

        return {
            "name": pokemon,
            "ability": ability,
            "item": item,
            "spread": spread,
            "IVs": ivs,
            "moves": valid_moves,
        }

    def pokemon_to_str(self, pokemon):
        name = pokemon["name"]
        ability = pokemon["ability"]
        item = pokemon["item"]
        nature, evs = pokemon["spread"].split(":")
        hp, atk, df, spa, spd, spe = evs.split("/")
        hp_i, atk_i, df_i, spa_i, spd_i, spe_i = pokemon["IVs"].split("/")
        # check if 6 evs are all zero
        if (
            hp == "0"
            and atk == "0"
            and df == "0"
            and spa == "0"
            and spd == "0"
            and spe == "0"
        ):
            hp = atk = df = spa = spd = spe = "84"
        moves = pokemon["moves"]
        return (
            f"{name} @ {item}\nAbility: {ability}\n"
            + f"EVs: {hp} HP / {atk} Atk / {df} Def / {spa} SpA / {spd} SpD / {spe} Spe\n{nature} Nature\n"
            + f"IVs: {hp_i} HP / {atk_i} Atk / {df_i} Def / {spa_i} SpA / {spd_i} SpD / {spe_i} Spe\n- "
            + "\n- ".join(moves)
        )

    def team_to_str(self, team):
        return "\n\n".join([self.pokemon_to_str(x) for x in team])


def same_team_members(team1, team2):
    # check the names of the pokemons in two teams are all the same
    names1 = [x["name"] for x in team1]
    names2 = [x["name"] for x in team2]
    return set(names1) == set(names2)
