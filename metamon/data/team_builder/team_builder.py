import random
import numpy as np
from metamon.data.team_builder.stat_reader import PreloadedSmogonStat
from metamon.data.team_builder.constants import (
    HIDDEN_POWER_IVS,
    HIDDEN_POWER_DVS,
    INCOMPATIBLE_MOVES,
)


class PokemonStatsLookupError(KeyError):
    def __init__(self, pokemon: str, format: str):
        message = f"Pokemon name `{pokemon}` could not be found in Smogon Statistics for format `{format}`"
        super().__init__(message)


class TeamBuilder:
    def __init__(self, format, verbose=True, remove_banned=True, inclusive=False):
        self.format = format
        self.stat = PreloadedSmogonStat(format, verbose=verbose, inclusive=inclusive)
        if remove_banned:
            self.stat.remove_banned_pm()
        self.verbose = verbose

        # use for different hp ivs
        self.isgen2 = format.startswith("gen2")
        self.isgen7 = format.startswith("gen7")
        self.team_pool = []
        self.current_team = []

    def check_valid_moves(self, pokemon, moves):
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

    def get_valid_moves(self, pokemon, moves, selected_moves=None):
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
            move = random_choice(moves, 1)[0]
            _remove_incompatible_moves(move)
            selected_moves.append(move)
        return selected_moves

    def generate_new_team(self, pokemon=None):
        self.current_team = []

        # is pokemon is a list
        if type(pokemon) == list and len(pokemon) > 0:
            for p in pokemon:
                try:
                    self.current_team.append(self.generate_moveset(p))
                except Exception as e:
                    raise PokemonStatsLookupError(p, self.format)
            if len(self.current_team) > 0:
                pokemon = self.current_team[-1]["name"]
        elif type(pokemon) == str:
            try:
                self.current_team.append(self.generate_moveset(pokemon))
            except Exception as e:
                raise PokemonStatsLookupError(pokemon, self.format)
        if len(self.current_team) == 0:
            # select first pokemon purely random, top cut to 100
            pokemon = self.get_random_pm()
            self.current_team.append(self.generate_moveset(pokemon))

        while len(self.current_team) < 6:
            pokemon = self.select_teammates(pokemon)
            while (
                not pokemon
                or pokemon not in self.stat.usage
                or self.in_current_team(pokemon)
            ):
                pokemon = self.get_random_pm()
            self.current_team.append(self.generate_moveset(pokemon))
        self.team_pool.append(self.current_team)
        return self.current_team

    def get_random_pm(self):
        return random.choice(self.stat.usage[:100])

    def select_teammates(self, pokemon):
        poke_stat = self.stat[pokemon]
        teammates = poke_stat["teammates"]
        # random choose one from teammates not in current team using their percent, teammates is a dict
        valid_teammates = [x for x in teammates.keys() if not self.in_current_team(x)]
        if len(valid_teammates) == 0:
            return None
        teammate = random.choices(
            valid_teammates, weights=[teammates[x] for x in valid_teammates], k=1
        )[0]
        return teammate

    def generate_moveset(self, pokemon):
        poke_stat = self.stat[pokemon]
        abilities = poke_stat["abilities"]
        items = poke_stat["items"]
        spreads = poke_stat["spreads"]
        moves = poke_stat["moves"]
        selected_moves = self.get_valid_moves(pokemon, moves.copy())

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
        return {
            "name": pokemon,
            "ability": random_choice(abilities, 1)[0],
            "item": random_choice(items, 1)[0],
            "spread": random_choice(spreads, 1)[0],
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

        if not ability:
            ability = random_choice(abilities, 1)[0]

        if not item:
            item = random_choice(items, 1)[0]

        if not spread:
            spread = random_choice(spreads, 1)[0]

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

    def in_current_team(self, pokemon):
        for p in self.current_team:
            if p["name"] == pokemon:
                return True
        return False


def random_choice(data, n):
    if "Other" in data:
        del data["Other"]
    if "Nothing" in data:
        del data["Nothing"]
    if not data:
        return [" "]
    # random choose four moves from moveset using their percent, should not duplicate
    p = np.array(list(data.values()))
    p = p / np.sum(p)
    n = min(n, len(data))
    return np.random.choice(list(data.keys()), size=n, replace=False, p=p).tolist()


def same_team_members(team1, team2):
    # check the names of the pokemons in two teams are all the same
    names1 = [x["name"] for x in team1]
    names2 = [x["name"] for x in team2]
    return set(names1) == set(names2)
