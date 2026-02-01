from abc import ABC, abstractmethod
import copy
import math
import random

import poke_env
from poke_env.player import Player, BattleOrder
from poke_env.environment import (
    Pokemon,
    PokemonType,
    Move,
    MoveCategory,
    Battle,
    SideCondition,
    Weather,
    Status,
    Effect,
)

from metamon.baselines import GEN_DATA
from metamon.backend.team_prediction.usage_stats import get_usage_stats


class Baseline(Player, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.randomize()

    def get_gen_format(self, battle: Battle) -> tuple[int, str]:
        """
        Enable generation and format-specific logic by parsing
        the current gen and format from the showdown wrapper
        """
        tag = battle.battle_tag
        _, battle_format, _ = tag.split("-")
        gen = int(battle_format[3])
        format = battle_format[4:]
        return gen, format

    def boost_move_scores(
        self,
        battle: Battle,
        attack_w: float = 1.0,
        defense_w: float = 1.0,
        spattack_w: float = 1.0,
        spdefense_w: float = 1.0,
        speed_w: float = 1.0,
    ) -> dict[Move, float]:
        """
        Rank the available_moves by a heuristic
        according to which stats they boost and by how much
        """
        scores = {}
        for move in battle.available_moves:
            boosts = move.boosts
            if boosts is None:
                scores[move] = 0
                continue
            score = 0
            if "spa" in boosts:
                score += spattack_w * boosts["spa"]
            if "spd" in boosts:
                score += spdefense_w * boosts["spd"]
            if "atk" in boosts:
                score += attack_w * boosts["atk"]
            if "def" in boosts:
                score += defense_w * boosts["def"]
            if "spe" in boosts:
                score += speed_w * boosts["spe"]
            scores[move] = score
        return scores

    def heal_move_scores(self, battle: Battle) -> dict[Move, float]:
        """
        Rank the available_moves by the fraction of health they recover
        """
        scores = {}
        for move in battle.available_moves:
            scores[move] = move.heal
        return scores

    def type_advantage(
        self, type: PokemonType, opponent_mon: Pokemon, battle: Battle
    ) -> float:
        """
        Calculate "super effective", "effective", "not very effective"
        multipliers for a move type given the target's type(s)
        """
        if type is None:
            return 0.0
        elif type in {PokemonType.THREE_QUESTION_MARKS, PokemonType.STELLAR}:
            return 1.0

        gen, _ = self.get_gen_format(battle)
        type_chart = GEN_DATA[gen].type_chart
        opp_types = [t for t in opponent_mon.types if t is not None]
        type_advantage = 1.0
        for opp_type in opp_types:
            if opp_type in {PokemonType.THREE_QUESTION_MARKS, PokemonType.STELLAR}:
                multiplier = 1.0
            else:
                multiplier = type_chart[opp_type.name][type.name]
            type_advantage *= multiplier
        return type_advantage

    def _stat_from_base_stats_early_gens(self, mon: Pokemon, stat: str):
        """
        Recreate current stat given base stats and level -- assuming best DV / StatExp
        """
        base = mon.base_stats[stat]
        lvl = mon.level
        # just assume the best stats since there's no tactical trade-off here...
        dv = 15
        statexp = 65_535
        inner = math.floor(
            ((base + dv) * 2 + math.floor(math.ceil(math.sqrt(statexp)) / 4.0))
            * lvl
            / 100.0
        )
        if stat == "hp":
            return inner + lvl + 10
        else:
            return inner + 5

    def _stat_from_base_stats_gen3plus(
        self, mon: Pokemon, stat: str, iv: int, ev: int, nature_change: str = "neutral"
    ) -> int:
        """
        Recreate current stat given a pokemon's base stats, level, iv, and ev
        (Using gen3+ version)

        https://bulbapedia.bulbagarden.net/wiki/Stat
        """
        base = mon.base_stats[stat]
        lvl = mon.level
        inner = math.floor(((2 * base + iv + math.floor(ev / 4)) * lvl) / 100)
        if stat == "hp":
            estimate = inner + lvl + 10
        else:
            if nature_change == "neutral":
                nat = 1.0
            elif nature_change == "hurts":
                nat = 0.9
            elif nature_change == "helps":
                nat = 1.1
            else:
                return ValueError(
                    f"Invalid `nature_change` arg {nature_change}. Options are 'neutral', 'hurts', 'helps'"
                )
            estimate = math.floor((inner + 5) * nat)
        return estimate

    def outspeed_chance(self, user: Pokemon, target: Pokemon, battle: Battle) -> float:
        """
        Heuristic to determine our chances of outspeeding an opponent given
        that we know its species and level but not its EV/IV/Nature.

        Most useful when returns 0 (never outspeeds) or 1 (always outspeeds).
        c in (0, 1) means we outspeed c% of the range of possible speed values.
        But we don't actually have a good prior for the missing EV/IV/Nature range
        so this should be taken lightly...
        """
        user_speed = user.stats["spe"]
        # use boost-adjusted speed estimates
        target_speed_low = self.infer_stats(target, battle, assume="worst")[1]["spe"]
        target_speed_high = (
            self.infer_stats(target, battle, assume="best")[1]["spe"] + 1
        )
        chance = (user_speed - target_speed_low) / (
            target_speed_high - target_speed_low
        )
        return max(min(chance, 1.0), 0.0)

    def find_ko_move(
        self, user: Pokemon, target: Pokemon, battle: Battle, assume_worst: bool
    ) -> Move | None:
        """
        Returns a move from `user` that we estimate will KO the target. If none
        found, returns None.
        """
        damage_estimate = (
            self.worst_case_damage if assume_worst else self.expected_damage
        )
        move_damage = [
            (m, damage_estimate(user, m, target, battle)) for m in user.moves.values()
        ]
        best_move, best_damage = max(move_damage, key=lambda x: x[1])
        if best_damage > target.current_hp:
            return best_move
        return None

    def should_emergency_switch(
        self, battle: Battle, must_be_slower: bool, must_be_low_hp: bool
    ) -> bool:
        """
        Do we need to switch out our current pokemon before it faints?
        """
        if battle.trapped:
            return False
        user = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        ineffective = (
            max(
                self.type_advantage(move.type, opp, battle)
                for move in user.moves.values()
            )
            < 1
        )
        slower = self.outspeed_chance(user, opp, battle=battle) < 0.5
        low_hp = user.current_hp_fraction < 0.25
        switch = ineffective or (Effect.PERISH2 in user.effects)
        if must_be_slower:
            switch = switch and slower
        if must_be_low_hp:
            switch = switch and low_hp
        return switch

    def find_outspeed_ko_switch(
        self, battle: Battle, assume_worst: bool
    ) -> Pokemon | None:
        """
        Returns a pokemon from the user's party that we estimate will
        outspeed and KO the opponent's current active pokemon. Returns
        None when no option found.
        """
        if not battle.available_switches:
            return None
        opp = battle.opponent_active_pokemon
        for switch in battle.available_switches:
            if self.outspeed_chance(switch, opp, battle=battle) < 1:
                continue
            ko_move = self.find_ko_move(switch, opp, battle, assume_worst=assume_worst)
            if ko_move is not None and ko_move.priority >= 0:
                return switch
        return None

    def infer_stats(self, mon: Pokemon, battle: Battle, assume: str) -> dict[str, int]:
        """
        Fill in any missing statistics by computing them from known info
        and reasonable guesses for EV/IV values
        """
        # fmt: off
        gen, _ = self.get_gen_format(battle)

        # infer default stats
        stats = mon.stats or {
            "hp": None,
            "spd": None,
            "spa": None,
            "atk": None,
            "def": None,
            "spe": None,
        }

        if assume == "best":
            ev = 252
        elif assume == "worst":
            ev = 0
        elif assume == "uniform":
            ev = 510 // 6
        else:
            raise RuntimeError(
                f"Invalid assumption `assume = {assume}` for infer_stats"
            )

        for stat, val in stats.items():
            if val is None:
                if gen > 2:
                    stats[stat] = self._stat_from_base_stats_gen3plus(mon, stat, iv=31, ev=ev)
                else:
                    stats[stat] = self._stat_from_base_stats_early_gens(mon, stat)

        # stat boosts
        modifiers = [0.25, 0.28, 0.33, 0.4, 0.5, 0.66, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        boosted_stats = copy.deepcopy(stats)
        boosts = mon.boosts
        for stat, val in boosted_stats.items():
            if stat in boosts:
                boost = boosts[stat]
                multiplier = modifiers[boost + 6]
                boosted_stats[stat] = round(val * multiplier)

        # fmt: on
        return stats, boosted_stats

    def current_accuracy(
        self, user: Pokemon, move: Move, target: Pokemon, battle: Battle
    ) -> float:
        """
        Get the current accuracy of a move, accounting for changes to accuracy/evasion stats

        https://bulbapedia.bulbagarden.net/wiki/Stat_modifier
        """
        # fmt: off
        base_accuracy = move.accuracy
        accuracy_stat = user.boosts["accuracy"]
        evasion_stat = target.boosts["evasion"]
        gen, _ = self.get_gen_format(battle)

        if gen == 1:
            mods = [0.25, 0.28, 0.33, 0.4, 0.5, 0.66, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        elif gen == 2:
            mods = [0.33, 0.36, 0.43, 0.5, 0.6, 0.75, 1.0, 1.33, 1.66, 2.0, 2.33, 2.66, 3.0]
        elif 3 <= gen <= 4:
            mods = [0.33, 0.36, 0.43, 0.5, 0.6, 0.75, 1.0, 1.33, 1.66, 2.0, 2.50, 2.66, 3.0]
        else:
            mods = [0.333, 0.375, 0.428, 0.5, 0.6, 0.75, 1.0, 1.33, 1.66, 2.0, 2.333, 2.666, 3.0]

        if gen <= 2:
            accuracy = base_accuracy * mods[accuracy_stat + 6] * mods[evasion_stat + 6]
        else:
            accuracy = base_accuracy * mods[accuracy_stat - evasion_stat + 6]

        if gen == 1:
            # "gen 1 miss"
            accuracy = min(accuracy, 255. / 256.)

        # fmt: on
        return accuracy

    def damage_roll(
        self, user: Player, move: Move, target: Pokemon, battle: Battle
    ) -> float:
        """
        Calculate damage with random RNG and a chance to miss
        """
        accuracy = self.current_accuracy(user, move, target, battle)
        hits = random.random() <= accuracy
        return (
            move.expected_hits
            * hits
            * self.damage_equation(
                user,
                move,
                target,
                battle,
                critical_hit=False,
                rng="random",
                assume_stats="uniform",
            )
        )

    def crit_chance(self, user: Pokemon, move: Move, battle: Battle) -> float:
        """
        [0, 1] prob of landing a critical hit, by generation

        https://bulbapedia.bulbagarden.net/wiki/Critical_hit
        """
        gen, _ = self.get_gen_format(battle)
        stage = min(move.crit_ratio, 4)
        if gen == 1:
            return user.base_stats["spe"] * 100.0 / 512 / 100.0
        elif gen == 2:
            return [0.0664, 0.125, 0.25, 0.332, 0.5][stage]
        elif 3 <= gen <= 5:
            return [0.0625, 0.125, 0.25, 0.333, 0.5][stage]
        elif gen == 6:
            return [0.0625, 0.125, 0.5, 1.0, 1.0][stage]
        else:
            return [0.0417, 0.125, 0.5, 1.0, 1.0][stage]

    def expected_damage(self, user, move, target, battle):
        """
        Calculate the (naive) expectation of damage dealt
        """
        crit_chance = self.crit_chance(user, move, battle)
        raw_damage = self.damage_equation(
            user,
            move,
            target,
            battle,
            critical_hit=False,
            rng="mean",
            assume_stats="uniform",
        )
        crit_damage = self.damage_equation(
            user,
            move,
            target,
            battle,
            critical_hit=True,
            rng="mean",
            assume_stats="uniform",
        )
        accuracy = self.current_accuracy(user, move, target, battle)
        exp_damage = accuracy * move.expected_hits * (
            (1.0 - crit_chance) * raw_damage
        ) + (crit_chance * crit_damage)
        return exp_damage

    def worst_case_damage(self, user, move, target, battle):
        """
        Calculate the (naive) worst-case-scenario for damage dealt
        """
        min_hits = move.n_hit[0]
        damage = (
            min_hits
            * self.damage_equation(
                user,
                move,
                target,
                battle,
                critical_hit=False,
                rng="min",
                assume_stats="best",
                disable_rounding=True,
            )
            * (self.current_accuracy(user, move, target, battle) >= (255.0 / 256.0))
        )  # counting gen 1 misses as perfect accuracy
        return damage

    def damage_equation(
        self,
        user: Pokemon,
        move: Move,
        target: Pokemon,
        battle: Battle,
        critical_hit: bool = False,
        rng: str = "mean",
        assume_stats: str = "uniform",
        disable_rounding: bool = False,
    ) -> int | float:
        """
        A simplified version of the pokemon damage calculation


        https://bulbapedia.bulbagarden.net/wiki/Damage
        """
        assert rng in ["max", "mean", "random", "min"]
        assert assume_stats in ["uniform", "best", "worst"]
        gen, _ = self.get_gen_format(battle)

        partial_protect = 1.0
        attack_raw_stats, attack_boosted_stats = self.infer_stats(
            user, battle=battle, assume=assume_stats
        )
        attack_stats = attack_raw_stats if critical_hit else attack_boosted_stats
        off_category = move.category
        if off_category == MoveCategory.PHYSICAL:
            attack = attack_stats["atk"]
            if SideCondition.REFLECT in battle.opponent_side_conditions:
                partial_protect *= 0.5
        elif off_category == MoveCategory.SPECIAL:
            attack = attack_stats["spa"]
            if SideCondition.LIGHT_SCREEN in battle.opponent_side_conditions:
                partial_protect *= 0.5
        elif off_category == MoveCategory.STATUS:
            return 0.0

        defense_raw_stats, defense_boosted_stats = self.infer_stats(
            target, battle=battle, assume=assume_stats
        )
        defense_stats = defense_raw_stats if critical_hit else defense_boosted_stats
        def_category = move.defensive_category
        if def_category == MoveCategory.PHYSICAL:
            defense = defense_stats["def"]
        elif def_category == MoveCategory.SPECIAL:
            defense = defense_stats["spd"]

        base_power_num = (
            ((2 * user.level) / 5.0)
            * partial_protect
            * move.base_power
            * (attack / defense)
        )
        base_power = (base_power_num / 50.0) + 2

        weather = battle.weather
        weather_mul = 1.0
        if Weather.RAINDANCE in weather:
            if move.type == PokemonType.WATER:
                weather_mul *= 1.5
            elif move.type == PokemonType.FIRE:
                weather_mul *= 0.5
        if Weather.SUNNYDAY in weather:
            if move.type == PokemonType.FIRE:
                weather_mul *= 1.5
            elif move.type == PokemonType.WATER:
                weather_mul *= 0.5

        if not critical_hit:
            critical = 1.0
        elif gen == 1:
            critical = (2 * user.level + 5) / (user.level + 5)
        elif 2 <= gen <= 5:
            critical = 2.0
        else:
            critical = 1.5

        if rng == "random":
            rng_mul = random.uniform(0.85, 1.0)
        elif rng == "min":
            rng_mul = 0.85
        elif rng == "max":
            rng_mul = 1.0
        elif rng == "mean":
            rng_mul = 0.925

        stab = 1.0
        if move.type in user.types:
            stab = 2.0 if user.is_terastallized else 1.5

        burn = (
            0.5
            if (move.category == MoveCategory.PHYSICAL and user.status == Status.BRN)
            else 1.0
        )
        type_multiplier = self.type_advantage(move.type, target, battle)

        damage = (
            base_power
            * weather_mul
            * critical
            * rng_mul
            * stab
            * type_multiplier
            * burn
        )
        damage = math.ceil(damage) if not disable_rounding else damage
        return damage

    def switch_scores(
        self,
        switches: list[Pokemon],
        battle: Battle,
        check_w: float = 0.0,
        min_check_val: float = 0.3,
        def_type_disadvantage_w: float = 0.0,
        off_type_advantage_w: float = 0.0,
        speed_w: float = 0.0,
        lvl_w: float = 0.0,
        hp_w: float = 0.0,
        status_free_w: float = 0.0,
    ) -> dict[Pokemon, float]:
        """
        Assign heuristic scores to available switches
        based on a linear combination of factors.
        Customize weights for different switching behavior.
        Please note that the terms corresponding to each weight
        are on very different scales.
        """
        opp = battle.opponent_active_pokemon
        gen, format = self.get_gen_format(battle)

        if check_w > 0:
            smogon_stats = get_usage_stats(f"gen{gen}{format.lower()}")

        switch_scores = {}
        for switch in switches:
            score = 0

            # Human "checks and counters" statistics [0, 1] or N/A
            if check_w > 0:
                # TODO: maybe revisit species/base_species after forme overhaul but it's
                # probably fine.
                opp_stats = smogon_stats[opp.base_species]
                if opp_stats is not None:
                    check_stats = {
                        poke_env.to_id_str(k): v
                        for k, v in opp_stats.get("checks", {}).items()
                    }
                    check_val = check_stats.get(switch.base_species, 0)
                    if check_val > min_check_val:
                        score += check_w * check_val

            # Defense Type Disadvantage [-4, 0]
            score += def_type_disadvantage_w * -sum(
                [
                    self.type_advantage(type, switch, battle)
                    for type in opp.types
                    if type is not None
                ]
            )

            # Offensive Type Advantage [0, 4]
            damage_move_types = [
                m.type for m in switch.moves.values() if m.base_power > 0
            ]
            score += off_type_advantage_w * sum(
                [
                    self.type_advantage(type, opp, battle)
                    * float(type in damage_move_types)
                    for type in switch.types
                    if type is not None
                ]
            )
            # Level [0, 100]
            score += lvl_w * switch.level
            # Speed [0, 400+]
            score += (
                speed_w
                * self.infer_stats(switch, battle=battle, assume="uniform")[1]["spe"]
            )
            # Current HP *As Fraction* [0, 1]
            score += hp_w * switch.current_hp_fraction
            # No Status [0, 1]
            score += status_free_w * float(switch.status is None)
            switch_scores[switch] = score
        return switch_scores

    def force_use_gimmick(self, battle: Battle, order: BattleOrder) -> BattleOrder:
        """
        Modify order to use generational power-up gimmick if on the last pokemon.
        Default strategy of game NPCs.
        """

        # last full HP pokemon remaining or last pokemon
        if (
            len([m for m in battle.team.values() if m.current_hp_fraction == 1]) == 1
            and battle.active_pokemon.current_hp_fraction == 1
        ) or len([m for m in battle.team.values() if m.fainted is False]) == 1:
            if battle.can_dynamax:
                order.dynamax = True
            if battle.can_tera:
                order.terastallize = True
            if battle.can_mega_evolve:
                order.mega = True
            if battle.can_z_move:
                order.z_move = True

        return order

    @abstractmethod
    def randomize(self):
        pass

    @property
    def lose_rate(self) -> float:
        return self.n_lost_battles / self.n_finished_battles

    @property
    def tie_rate(self) -> float:
        return self.n_tied_battles / self.n_finished_battles
