import copy
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from metamon.data.replay_dataset.parsed_replays.replay_parser import checks
from metamon.data.replay_dataset.parsed_replays.replay_parser.exceptions import *
from metamon.data.replay_dataset.parsed_replays.replay_parser.replay_state import (
    Action,
    BackwardMarkers,
    Boosts,
    Move,
    Nothing,
    Pokemon,
    Turn,
    Winner,
)
from metamon.data.replay_dataset.parsed_replays.replay_parser.str_parsing import *

from poke_env.data import to_id_str
from poke_env.environment import STACKABLE_CONDITIONS
from poke_env.environment import Effect as PEEffect
from poke_env.environment import Field as PEField
from poke_env.environment import SideCondition as PESideCondition
from poke_env.environment import Status as PEStatus
from poke_env.environment import Weather as PEWeather


@dataclass
class ParsedReplay:
    gameid: str
    format: str
    time_played: datetime
    ratings: List[Optional[int | str]] = field(default_factory=lambda: [None, None])
    players: List[Optional[str]] = field(default_factory=lambda: [None, None])
    gen: Optional[int] = None
    turnlist: List[Turn] = field(default_factory=lambda: [Turn(turn_number=0)])
    rules: List[str] = field(default_factory=list)
    winner: Optional[Winner] = None

    def __getitem__(self, i):
        return self.turnlist[i]

    @property
    def flattened_turnlist(self):
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


class SpecialCategories:
    # https://bulbapedia.bulbagarden.net/wiki/Category:Moves_that_switch_the_user_out
    MOVES_THAT_SWITCH_THE_USER_OUT = {
        "Baton Pass",
        "Chilly Reception",
        "Flip Turn",
        "Parting Shot",
        "Shed Tail",
        "Teleport",
        "U-turn",
        "Volt Switch",
    }

    # https://bulbapedia.bulbagarden.net/wiki/Category:Moves_that_call_other_moves
    MOVE_OVERRIDE = {
        "Metronome",
        "Me First",
        "Copycat",
        "Nature Power",
        "Magic Coat",
        "Mirror Move",
        "Assist",
        "Snatch",
    }
    MOVE_OVERRIDE_BUT_REVEAL_ANYWAY = {"Sleep Talk"}
    MOVE_IGNORE_ITEMS = {"Custap Berry"}
    CONSECUTIVE_MOVES = {
        "Rollout",
        "Outrage",
        "Thrash",
        "Uproar",
        "Petal Dance",
        "Ice Ball",
    }

    GEN1_PP_ROLLOVERS = {"Bind", "Wrap", "Fire Spin", "Clamp"}

    # https://bulbapedia.bulbagarden.net/wiki/Category:Moves_that_restore_HP
    RESTORES_PP = {"Lunar Dance"}
    RESTORES_STATUS = {"Healing Wish", "Lunar Dance"}

    ABILITY_STEALS_ABILITY = {"Trace"}

    # https://bulbapedia.bulbagarden.net/wiki/Category:Item-manipulating_moves
    # we are missing some of these; lookout for UnhandledFromMoveItemLogic
    ITEM_APPROVED_SKIP = {"Knock Off", "Recycle", "Fling", "Corrosive Gas"}
    ITEM_UNNAMED_STOLEN = {"Trick", "Switcheroo"}
    ITEM_NAMED_STOLEN = {"Thief", "Covet"}


REPLAY_IGNORES = {
    "",
    "-anim",
    "badge",
    "bigerror",  # usually auto-tie warnings
    "c",
    "c:",
    "chatmsg-raw",
    "-crit",  # redundant
    "chat",
    "clearpoke",
    "debug",
    "error",
    "-fieldactivate",  # redundant
    "gametype",
    "hidelines",  # undocumented, no idea
    "-hint",
    "hint",
    "html",
    "-hitcount",
    "inactive",  # battle timer
    "inactiveoff",  # battle timer
    "j",
    "J",
    "join",
    "leave",
    "l",
    "L",
    "-miss",
    "message",
    "-message",  # chat
    "n",
    "-nothing",  # redundant for a move that did "absolutely nothing"
    "-notarget",  # for move target
    "-ohko",
    "-prepare",  # extra move info
    "-primal",  # based soley on action
    "raw",
    "rated",
    "-resisted",
    "start",
    "-supereffective",
    "-singlemove",
    "seed",
    "teampreview",
    "tier",
    "t:",  # timer
    "upkeep",
    "uhtml",
    "unlink",  # disconnect or spectator removed
    "-zbroken",  # z-move hits through protect
}

# fmt: off
def parse_row(replay: ParsedReplay, row: List[str]):
    """
    https://github.com/smogon/pokemon-showdown/blob/master/sim/SIM-PROTOCOL.md
    
    and https://github.com/hsahovic/poke-env/blob/master/src/poke_env/environment/abstract_battle.py

    When in doubt, match the poke-env version to cut "sim2sim" offline/online gap.
    """
    curr_turn = replay.turnlist[-1]

    name, *data = row

    if name in REPLAY_IGNORES:
        return

    if name == "gen":
        # |gen|GENNUM
        replay.gen = int(data[0])
        if replay.gen >= 5:
            raise SoftLockedGen(replay.gen)

    elif name == "player":
        # |player|PLAYER|USERNAME|AVATAR|RATING
        if len(data) < 2 or not data[1]:
            # skip reintroductions
            return
        if data[0] == "p1":
            slot = 0
        elif data[0] == "p2":
            slot = 1
        else:
            raise RareValueError(f"Could not parse player slot from player id `{data[0]}`")
        replay.players[slot] = to_id_str(data[1])
        if len(data) >= 4 and data[3]:
            replay.ratings[slot] = int(data[3])
        else:
            replay.ratings[slot] = "Unrated"

    elif name == "teamsize":
        # |teamsize|PLAYER|NUMBER
        player, size = data
        size = int(size)
        if size != 6:
            raise UnusualTeamSize(size)
        blank_team = [None] * size
        if player == "p1":
            curr_turn.pokemon_1 = blank_team
        elif player == "p2":
            curr_turn.pokemon_2 = blank_team

    elif name == "turn":
        # |turn|NUMBER
        checks.check_forced_switching(curr_turn)
        assert curr_turn.turn_number is not None
        new_turn = curr_turn.create_next_turn()
        # saves the within-turn-state for the previous turn, but does not continue it
        new_turn.on_end_of_turn() 
        replay.turnlist.append(new_turn)

    elif name == "win":
        # |win|USER
        winner_name = to_id_str(data[0])
        if winner_name == replay.players[0]:
            replay.winner = Winner.PLAYER_1
        elif winner_name == replay.players[1]:
            replay.winner = Winner.PLAYER_2
        else:
            raise RareValueError(
                f"Unknown winner: {winner_name} with players: {replay.players}"
            )

    elif name == "choice":
        # https://github.com/smogon/pokemon-showdown/blob/master/sim/SIM-PROTOCOL.md#sending-decisions
        # `choice` messaegs reveal players action choices directly instead of waiting to see the outcome of the move/switch.
        # they are not present in every replay, but when they are, they can help us fill missing actions.
        # It would be possible to catch many more choices if we could use the numeric arg format of some
        # messages (e.g. `move 1`). We'd need to infer a mapping between the numeric args and the move names.
        for player_idx, player_choice in enumerate(data):
            if player_choice:
                for poke_idx, poke_choice in enumerate(player_choice.split(",")):
                    msg = poke_choice.split(" ")
                    command = msg[0]
                    args = re.sub(r'\d+', '', " ".join(msg[1:])).strip()
                    if command == "move" and args and args.lower() not in {"recharge", "struggle"}:
                        user_pokemon = curr_turn.active_pokemon_1[poke_idx] if player_idx == 0 else curr_turn.active_pokemon_2[poke_idx]
                        move = Move(name=args, gen=replay.gen)
                        choice = Action(name=move.name, is_switch=False, is_noop=False, user=user_pokemon, target=None)
                        user_pokemon.reveal_move(move)
                        if player_idx == 0:
                            curr_turn.choices_1[poke_idx] = choice
                        else:
                            curr_turn.choices_2[poke_idx] = choice

    elif name == "tie":
        # |tie
        replay.winner = Winner.TIE

    elif name == "rule":
        # |rule|RULE: DESCRIPTION
        replay.rules.append(data[0])

    elif name == "poke":
        # |poke|PLAYER|DETAILS|ITEM
        poke_list = curr_turn.get_pokemon_list_from_str(data[0])
        assert isinstance(poke_list, list)
        if None not in poke_list:
            raise UnusualTeamSize(len(poke_list) + 1)
        poke_name, lvl = Pokemon.identify_from_details(data[1])
        insert_at = poke_list.index(None)
        poke_list[insert_at] = Pokemon(name=poke_name, lvl=lvl, gen=replay.gen)

    elif name == "switch" or name == "drag":
        # |switch|POKEMON|DETAILS|HP STATUS or |drag|POKEMON|DETAILS|HP STATUS
        if len(data) < 3:
            raise UnfinishedMessageException(row)

        # fill the forced switch state
        switch_team, switch_slot = curr_turn.player_id_to_action_idx(data[0])
        is_force_switch = False
        for subturn_idx, subturn in enumerate(curr_turn.subturns):
            if subturn.unfilled and subturn.matches_slot(switch_team, switch_slot):
                is_force_switch = True
                subturn.fill_turn(curr_turn.create_subturn(is_force_switch))
                break

        # id switch out
        poke_list = curr_turn.get_pokemon_list_from_str(data[0])
        assert poke_list
        active_poke_list = curr_turn.get_active_pokemon_from_str(data[0])
        current_active = active_poke_list[switch_slot]
        if current_active is not None:
            current_active.on_switch_out()

        # id switch in
        poke_name, lvl = Pokemon.identify_from_details(data[1])
        known_names = [p.name if p else None for p in poke_list]
        known_first_names = [p.had_name if p else None for p in poke_list]
        if poke_name in known_names:
            # previously identified pokemon
            poke = poke_list[known_names.index(poke_name)]
        elif poke_name in known_first_names:
            # previously identified, but known by name that has been changing (forms)
            poke = poke_list[known_first_names.index(poke_name)]
        else:
            # discovered by switching in
            poke = Pokemon(name=poke_name, lvl=lvl, gen=replay.gen)
            if None not in poke_list: raise CantIDSwitchIn(data[1], poke_list)
            insert_at = poke_list.index(None)
            poke_list[insert_at] = poke
        active_poke_list[switch_slot] = poke
        cur_hp, max_hp = parse_hp_fraction(data[2])
        poke.max_hp = max_hp
        poke.current_hp = cur_hp

        if name == "switch":
            if is_force_switch:
                curr_turn.subturns[subturn_idx].action = Action(name="Switch", user=current_active, target=poke, is_switch=True)
            else:
                curr_turn.set_move_attribute(
                    data[0][:3], move_name="Switch", is_noop=False, is_switch=True, user=current_active, target=poke,
                )

    elif name == "move":
        # |move|POKEMON|MOVE|TARGET
        if len(data) < 2:
            raise UnfinishedMessageException(row)

        # id pokemon
        poke_str = data[0][:3]
        pokemon = curr_turn.get_pokemon_from_str(data[0])
        move_name = data[1]
        move = Move(name=move_name, gen=replay.gen)
        target_pokemon = curr_turn.get_pokemon_from_str(data[2]) if len(data) > 2 else None
        probably_repeat_move = False
        extra_from_message = None
        for d in reversed(data):
            if "[from]" in d:
                extra_from_message = d
                break

        if move_name in SpecialCategories.MOVES_THAT_SWITCH_THE_USER_OUT:
            notarget = any('[notarget]' in d for d in data)
            protected = target_pokemon.protected if target_pokemon else False
            if not notarget and not protected:
                curr_turn.mark_forced_switch(data[0])

        if extra_from_message:
            # fishing for "moves called by moves that call other moves", which should be prevented
            # from being incorrectly added to a pokemon's true moveset.
            override_risk = move_name in SpecialCategories.CONSECUTIVE_MOVES or move.charge_move

            if "[from]move:" in extra_from_message or "[from]ability:" in extra_from_message:
                # MODERN REPLAY VERSION
                is_move = "[from]move" in extra_from_message
                is_ability = "[from]ability" in extra_from_message
                if is_move:
                    from_move = parse_move_from_extra(extra_from_message)
                    probably_repeat_move = from_move.lower() == move_name.lower()
                    if from_move in SpecialCategories.MOVE_OVERRIDE_BUT_REVEAL_ANYWAY | SpecialCategories.MOVE_OVERRIDE:
                        if override_risk:
                            raise CalledForeignConsecutive(row)
                        if from_move in SpecialCategories.MOVE_OVERRIDE_BUT_REVEAL_ANYWAY:
                            pokemon.reveal_move(move)
                        return
                elif is_ability:
                    raise UnimplementedMoveFromMoveAbility(data)
            else:
                # OLD REPLAY VERSION
                ability_or_move = parse_extra(extra_from_message)
                probably_repeat_move = ability_or_move.lower() == move_name.lower()
                probably_item = ability_or_move in ({pokemon.had_item, pokemon.active_item} | SpecialCategories.MOVE_IGNORE_ITEMS)
                if not (probably_repeat_move or probably_item):
                    if ability_or_move in SpecialCategories.MOVE_OVERRIDE_BUT_REVEAL_ANYWAY | SpecialCategories.MOVE_OVERRIDE:
                        if override_risk:
                            raise CalledForeignConsecutive(row)
                        if ability_or_move in SpecialCategories.MOVE_OVERRIDE_BUT_REVEAL_ANYWAY:
                            pokemon.reveal_move(move)
                        return
                probably_ability = ability_or_move.lower() not in {"lockedmove", "pursuit"} and not probably_item and not probably_repeat_move
                if probably_ability:
                    raise UnimplementedMoveFromMoveAbility(data)


        # how much PP is used?
        pressured = target_pokemon is not None and target_pokemon.active_ability == "Pressure" and target_pokemon != pokemon and '[notarget]' not in data
        if move.charge_move:
            if "[still]" in data:
                pp_used = 0
            else:
                pp_used = 1 + pressured
        elif replay.gen == 1:
            # gen1 partial trapping PP counting edge cases
            if probably_repeat_move:
                # the turns that auto apply partial trapping moves w/o PP have `move USER MOVE TARGET [from]MOVE`
                pp_used = 0
            elif move_name in SpecialCategories.GEN1_PP_ROLLOVERS and pokemon.get_pp_for_move_name(move_name) == 0:
                # (https://www.smogon.com/rb/articles/rby_trapping)
                pp_used = -63
            else:
                pp_used = 1
        else:
            pp_used = 1 + pressured
        # known PP counter flaw: consecutively executed moves (https://bulbapedia.bulbagarden.net/wiki/Category:Consecutively_executed_moves).
        # also breaks the PS replay viewer (https://replay.pokemonshowdown.com/gen2ou-1136717194); we just aren't given enough info to tell
        # when the move should/shouldn't use PP.

        # use move
        pokemon.use_move(move, pp_used=pp_used)
        if pokemon.transformed_into is not None:
            pokemon.transformed_into.reveal_move(copy.deepcopy(move))

        # create edge between pokemon to help track down special cases
        pokemon.last_target = (target_pokemon, move_name) 
        if target_pokemon:
            target_pokemon.last_targeted_by = (pokemon, move_name)

        # create Action
        curr_turn.set_move_attribute(
            s=poke_str,
            move_name=move.name,
            is_noop=False,
            is_switch=False,
            user=pokemon,
            target=target_pokemon,
        )

        
    elif name == "-damage" or name == "-heal":
        # |-damage|POKEMON|HP STATUS or |-heal|POKEMON|HP STATUS
        if len(data) < 2 or (len(data) == 2 and not data[-1]):
            raise UnfinishedMessageException(" ".join(row))

        # take or heal damage
        pokemon = curr_turn.get_pokemon_from_str(data[0])
        assert pokemon is not None
        if "fnt" in data[1] or data[1] == "0" or data[1][:2] == "0 ":
            pokemon.current_hp = 0
            pokemon.status = PEStatus.FNT
        else:
            if "/" not in data[1]:
                raise UnfinishedMessageException(row)
            cur_hp, max_hp = parse_hp_fraction(data[1])
            pokemon.current_hp = cur_hp
            pokemon.max_hp = max_hp

        if len(data) > 2:
            # parse extra info for items and abilities
            found_item, found_ability, found_move, found_of_pokemon = parse_from_effect_of(data)
            if found_item:
                if name == "-heal":
                    # like the ability/heal combo just below, the "[of]" pokemon here is misleading
                    of_pokemon = pokemon
                    if of_pokemon.active_item != Nothing.NO_ITEM:
                        of_pokemon.active_item = found_item
                    if of_pokemon.had_item is None:
                        of_pokemon.had_item = found_item
                else:
                    if found_of_pokemon:
                        of_pokemon = curr_turn.get_pokemon_from_str(found_of_pokemon)
                    else:
                        of_pokemon = pokemon
                    of_pokemon.active_item = found_item
                    if of_pokemon.had_item is None:
                        of_pokemon.had_item = found_item

            if found_ability:
                if name == "-heal":
                    # poke-env comments say the [of] pokemon is the wrong side here
                    # (|-heal|p2a: Quagsire|100/100|[from] ability: Water Absorb|[of] p1a: Genesect
                    # is healing Quagsire from Quagsire's Water Absorb ability)
                    of_pokemon = pokemon
                else:
                    if found_of_pokemon:
                        of_pokemon = curr_turn.get_pokemon_from_str(found_of_pokemon)
                    else:
                        of_pokemon = pokemon
                of_pokemon.reveal_ability(found_ability)

            if found_move:
                if found_move in SpecialCategories.RESTORES_PP:
                    for move_name, move in pokemon.moves.items():
                        move.pp = move.maximum_pp
                        if move_name in pokemon.had_moves:
                            had_move = pokemon.had_moves[move_name]
                            had_move.pp = had_move.maximum_pp

                if found_move in SpecialCategories.RESTORES_STATUS and pokemon.status != PEStatus.FNT:
                    pokemon.status = Nothing.NO_STATUS
    
    elif name == "-sethp":
        # |-sethp|POKEMON|HP
        pokemon = curr_turn.get_pokemon_from_str(data[0])
        assert pokemon is not None
        cur_hp, max_hp = parse_hp_fraction(data[1])
        if pokemon.max_hp:
            assert max_hp == pokemon.max_hp
        pokemon.current_hp = cur_hp

    elif name == "faint":
        # |faint|POKEMON
        pokemon = curr_turn.get_pokemon_from_str(data[0])
        assert pokemon is not None
        pokemon.current_hp = 0
        pokemon.status = PEStatus.FNT
        curr_turn.mark_forced_switch(data[0])

    elif name == "-status" or name == "-curestatus":
        # |-status|POKEMON|STATUS or |-curestatus|POKEMON|STATUS
        pokemon = curr_turn.get_pokemon_from_str(data[0])
        assert pokemon is not None
        status = PEStatus[data[1].upper()]
        if name == "-status":
            pokemon.status = status
        elif pokemon.status == status:
            pokemon.status = Nothing.NO_STATUS

    elif name == "-boost" or name == "-unboost":
        # |-boost|POKEMON|STAT|AMOUNT or |-unboost|POKEMON|STAT|AMOUNT
        if len(data) < 3:
            raise UnfinishedMessageException(row)
        change = int(data[2])
        if name == "-unboost":
            change *= -1
        pokemon = curr_turn.get_pokemon_from_str(data[0])
        assert pokemon is not None
        pokemon.boosts.change_with_str(data[1], change)

    elif name == "-swapboost":
        # |-swapboost|SOURCE|TARGET|STATS
        pokemon_1 = curr_turn.get_pokemon_from_str(data[0])
        pokemon_2 = curr_turn.get_pokemon_from_str(data[1])
        if "[from]" in data[2]:
            if "Heart Swap" in data[2]:
                stats = ["atk", "spa", "def", "spd", "spe", "accuracy", "evasion"]
            elif "Guard Swap" in data[2]:
                stats = ["def", "spd"]
            else:
                raise UnimplementedSwapboost(row)
        else:
            stats = data[2].split(", ")
        temp = copy.deepcopy(pokemon_1.boosts)
        for stat in stats:
            pokemon_1.boosts.set_to_with_str(stat, pokemon_2.boosts.get_boost(stat))
            pokemon_2.boosts.set_to_with_str(stat, temp.get_boost(stat))

    elif name == "swap":
        # |swap|POKEMON|POSITION
        raise UnimplementedMessage(row)

    elif name == "-ability":
        # |-ability|POKEMON|ABILITY|[from]EFFECT
        if len(data) < 2:
            raise UnfinishedMessageException(row)
        pokemon = curr_turn.get_pokemon_from_str(data[0])
        ability = parse_ability(data[1])
        found_item, found_ability, found_move, found_mon = parse_from_effect_of(data)
        if found_mon and found_ability:
            if found_ability in SpecialCategories.ABILITY_STEALS_ABILITY:
                # ['p1a: Porygon2', 'Clear Body', '[from] ability: Trace', '[of] p2a: Dragapult']
                # Porygon2 has the Trace ability, copying Clear Body from Dragapult
                pokemon.active_ability = ability # # porygon now has clear body
                if pokemon.had_ability is None:
                    pokemon.had_ability = found_ability # porygon used to have trace
                curr_turn.get_pokemon_from_str(found_mon).reveal_ability(ability) # dragapult has clear body
            else:
                raise UnhandledFromOfAbilityLogic(row)
        elif (found_item or found_mon or found_move) and found_ability:
            raise UnhandledFromOfAbilityLogic(row)
        else:
            pokemon.reveal_ability(ability)

    elif name == "-endability":
        # |-endability|POKEMON
        pokemon = curr_turn.get_pokemon_from_str(data[0])
        pokemon.active_ability = Nothing.NO_ABILITY
        if len(data) > 1:
            ability = parse_ability(data[1])
            if pokemon.had_ability is None:
                pokemon.had_ability = ability

    elif name == "-sidestart" or name == "-sideend" or name == "-swapsideconditions":
        # |-sidestart|SIDE|CONDITION or |-sideend|SIDE|CONDITION or |-swapsideconditions
        if "start" in name or "end" in name:
            side_str = data[0][0:2]
            if side_str == "p1":
                side = curr_turn.conditions_1
            elif side_str == "p2":
                side = curr_turn.conditions_2
            else:
                raise RareValueError(f"Can't find side conditions from identifier `{data[0]}`")
            if len(data) < 2:
                raise UnfinishedMessageException(row)
            condition = PESideCondition.from_showdown_message(data[1])
            if "start" in name:
                if condition in STACKABLE_CONDITIONS:
                    side[condition] = side.get(condition, 0) + 1
                elif condition not in side:
                    side[condition] = curr_turn.turn_number
            else:
                if condition in side and condition != PESideCondition.UNKNOWN:
                    side.pop(condition)
        else:
            curr_turn.conditions_1, curr_turn.conditions_2 = curr_turn.conditions_2, curr_turn.conditions_1

    elif name == "-weather":
        # |-weather|WEATHER
        if data[0] == "none":
            curr_turn.weather = Nothing.NO_WEATHER
        else:
            curr_turn.weather = PEWeather.from_showdown_message(data[0])
        found_item, found_ability, found_move, found_of_mon = parse_from_effect_of(data)
        if found_of_mon:
            pokemon = curr_turn.get_pokemon_from_str(found_of_mon)
            assert pokemon is not None
            if found_ability:
                pokemon.reveal_ability(found_ability)

    elif name == "-activate":
        # |-activate|EFFECT
        # also the catch-all message PS sends for minor edge cases
        pokemon = curr_turn.get_pokemon_from_str(data[0])
        assert pokemon is not None
        if data[1].startswith("ability:"):
            ability = parse_ability(data[1])
            pokemon.reveal_ability(ability)
            return

        effect = PEEffect.from_showdown_message(data[1])
        # edge cases
        found_item, found_ability, found_move, found_mon = parse_from_effect_of(data)
        if effect == PEEffect.TRICK:
            if found_mon:
                found_mon = curr_turn.get_pokemon_from_str(found_mon)
                pokemon.tricking = found_mon
                found_mon.tricking = pokemon
            else:
                raise TrickError(row)
        elif effect == PEEffect.MIMIC:
            pokemon.mimic(move_name=data[2], gen=replay.gen)
        elif effect in [PEEffect.LEPPA_BERRY, PEEffect.MYSTERY_BERRY]:
            # https://bulbapedia.bulbagarden.net/wiki/Category:PP-restoring_items
            pp_gained = 10 if effect == PEEffect.LEPPA_BERRY else 5
            move_name = data[2]
            if move_name in pokemon.moves:
                pokemon.moves[move_name].pp += pp_gained
                if move_name in pokemon.had_moves:
                    pokemon.had_moves[move_name].pp += pp_gained
        # this catches so much redundant info that probably shouldn't be called
        # a volatile status (and isn't displayed like that on PS)
        # but it's 1:1 with poke-env
        pokemon.start_effect(effect)

    elif name == "-item" or name == "-enditem":
        # |-item|POKEMON|ITEM|[from]EFFECT or |-enditem|POKEMON|ITEM|[from]EFFECT
        pokemon = curr_turn.get_pokemon_from_str(data[0])
        item = data[1]
        if pokemon is None:
            raise RareValueError(f"Could not find pokemon from {data[0]}")

        # adjust active item
        if "end" in name:
            pokemon.active_item = Nothing.NO_ITEM
        else:
            pokemon.active_item = item

        found_item, found_ability, found_move, found_mon = parse_from_effect_of(data)
        if found_move:
            if found_move in SpecialCategories.ITEM_APPROVED_SKIP:
                pass
            elif found_move in SpecialCategories.ITEM_UNNAMED_STOLEN:
                # item is stolen from the opponent using a move
                if not pokemon.tricking:
                    raise TrickError(row)
                if pokemon.tricking.had_item is None:
                    pokemon.tricking.had_item = item
            elif found_move in SpecialCategories.ITEM_NAMED_STOLEN:
                # item is stolen from a named opponent.
                if "end" in name:
                    pokemon_that_had_the_item = pokemon
                else:
                    pokemon_that_had_the_item = curr_turn.get_pokemon_from_str(found_mon)
                # remove item 
                pokemon_that_had_the_item.active_item = Nothing.NO_ITEM
                if pokemon_that_had_the_item.had_item is None:
                    pokemon_that_had_the_item.had_item = item
                # if we still don't know the pokemon's item here, make sure we never fill it in
                if pokemon.had_item is None:
                    pokemon.had_item = BackwardMarkers.FORCE_UNKNOWN
            else:
                raise UnhandledFromMoveItemLogic(row)

        elif pokemon.had_item is None:
            pokemon.had_item = item

    elif name == "-terastallize" or name == "-zpower" or name == "-mega":
        raise SoftLockedGen(replay.gen)

    elif name == "-transform":
        # |-transform|POKEMON|SPECIES
        user = curr_turn.get_pokemon_from_str(data[0])
        target = curr_turn.get_pokemon_from_str(data[1])
        _, found_ability, _, _ = parse_from_effect_of(data)
        if found_ability:
            user.reveal_ability(found_ability)
        user.transform(target)

    elif name in ["-fieldstart", "-fieldend"]:
        # |-fieldstart|CONDITION or |-fieldend|CONDITION
        field_condition = PEField.from_showdown_message(data[0])
        if name == "-fieldstart":
            found_item, found_ability, found_move, found_of_mon = parse_from_effect_of(data)
            if found_of_mon and found_ability:
                pokemon = curr_turn.get_pokemon_from_str(found_of_mon)
                assert pokemon is not None
                pokemon.reveal_ability(found_ability)

            if field_condition.is_terrain:
                curr_turn.battle_field = {f : t for f, t in curr_turn.battle_field.items() if not f.is_terrain}
            curr_turn.battle_field[field_condition] = curr_turn.turn_number
        else:
            if field_condition != PEField.UNKNOWN:
                curr_turn.battle_field.pop(field_condition)

    elif name == "-cureteam":
        # |-cureteam|POKEMON
        poke_list = curr_turn.get_pokemon_list_from_str(data[0])
        for poke in poke_list:
            if poke and poke.status != PEStatus.FNT:
                poke.status = Nothing.NO_STATUS

    elif name in ["-start", "-end"]:
        # |-start|POKEMON|EFFECT or # |-end|POKEMON|EFFECT
        pokemon = curr_turn.get_pokemon_from_str(data[0])
        assert pokemon is not None
        effect = PEEffect.from_showdown_message(data[1])
        if effect == PEEffect.MIMIC:
            # 1 of 2 ways PS will tell you which move Mimic copies 
            # (depending on gen or replay date it's hard to tell)
            pokemon.mimic(move_name=data[2], gen=replay.gen)
        found_item, found_ability, found_move, found_mon = parse_from_effect_of(data[2:])
        if "start" in name:
            pokemon.start_effect(effect)
        else:
            pokemon.end_effect(effect)
        if found_item or found_ability or found_mon or found_move:
            if found_mon is None:
                of_pokemon = pokemon
            else:
                of_pokemon = curr_turn.get_pokemon_from_str(found_mon)
            if found_ability:
                of_pokemon.reveal_ability(found_ability)

    elif name == "-setboost":
        # |-setboost|POKEMON|STAT|AMOUNT
        pokemon = curr_turn.get_pokemon_from_str(data[0])
        assert pokemon is not None
        pokemon.boosts.set_to_with_str(data[1], int(data[2]))

    elif name == "-clearboost":
        # |-clearboost|POKEMON
        pokemon = curr_turn.get_pokemon_from_str(data[0])
        assert pokemon is not None
        pokemon.boosts = Boosts()

    elif name == "-clearpositiveboost":
        # |-clearpositiveboost|TARGET|POKEMON|EFFECT
        pokemon = curr_turn.get_pokemon_from_str(data[0])
        for stat in pokemon.boosts.stat_attrs:
            current = getattr(pokemon.boosts, stat)
            setattr(pokemon.boosts, stat, min(current, 0))

    elif name == "-clearnegativeboost":
        # |-clearnegativeboost|POKEMON
        pokemon = curr_turn.get_pokemon_from_str(data[0])
        for stat in pokemon.boosts.stat_attrs:
            current = getattr(pokemon.boosts, stat)
            setattr(pokemon.boosts, stat, max(current, 0))

    elif name == "-copyboost":
        # |-copyboost|SOURCE|TARGET
        source = curr_turn.get_pokemon_from_str(data[0])
        target = curr_turn.get_pokemon_from_str(data[1])
        assert source is not None and target is not None
        source.boosts = copy.deepcopy(target.boosts)

    elif name == "-clearallboost":
        # |-clearallboost
        for active in [curr_turn.active_pokemon_1, curr_turn.active_pokemon_2]:
            for pokemon in active:
                if pokemon:
                    pokemon.boosts = Boosts()

    elif name == "-restoreboost":
        # |-restoreboost|p2a: Gorebyss|[silent]
        pokemon = curr_turn.get_pokemon_from_str(data[0])
        assert pokemon is not None
        boosts = pokemon.boosts
        for stat_name in boosts.stat_attrs:
            stage = getattr(boosts, stat_name)
            setattr(boosts, stat_name, max(stage, 0))

    elif name == "-invertboost":
        # |-invertboost|POKEMON
        pokemon = curr_turn.get_pokemon_from_str(data[0])
        assert pokemon is not None
        boosts = pokemon.boosts
        for stat_name in boosts.stat_attrs:
            inv = -getattr(boosts, stat_name)
            setattr(boosts, stat_name, inv)

    elif name == "-mustrecharge":
        # |-mustrecharge|POKEMON
        # the action labels default to None, so we do nothing here.
        pass

    elif name == "cant":
        # |cant|POKEMON|REASON or |cant|POKEMON|REASON|MOVE
        # pokemon cannot move and we usually aren't told what the player's preferred action was.
        # the action labels default to None, so we do nothing here.
        pass

    elif name == "-immune":
        # | -immune | POKEMON
        pokemon = curr_turn.get_pokemon_from_str(data[0])
        found_item, found_ability, found_move, found_mon = parse_from_effect_of(data)
        if found_ability:
            pokemon.reveal_ability(found_ability)

    elif name == "detailschange" or name == "-formechange":
        # |detailschange|POKEMON|DETAILS|HP STATUS or |-formechange|POKEMON|SPECIES|HP STATUS
        pokemon = curr_turn.get_pokemon_from_str(data[0])
        if pokemon.had_name is None:
            pokemon.had_name = pokemon.name
        name, lvl = Pokemon.identify_from_details(data[1])
        pokemon.name = name
        pokemon.lvl = lvl
        found_item, found_ability, found_move, found_mon = parse_from_effect_of(data)
        if found_ability:
            pokemon.reveal_ability(found_ability)

    elif name == "replace":
        # |replace|POKEMON|DETAILS|HP STATUS
        raise ZoroarkException

    elif name == "-burst":
        # |-burst|POKEMON|SPECIES|ITEM
        raise UnimplementedMessage(row)

    elif name == "-fail":
        # |-fail|POKEMON|ACTION
        pokemon = curr_turn.get_pokemon_from_str(data[0])
        if pokemon.last_used_move is not None and pokemon.last_used_move.name in SpecialCategories.MOVES_THAT_SWITCH_THE_USER_OUT:
            # a move we thought would create a forced switch actually failed
            team, slot = curr_turn.player_id_to_action_idx(data[0])
            curr_turn.remove_empty_subturn(team=team, slot=slot)

    elif name == "-singleturn":
        # ['-singleturn', 'p2a: Abomasnow', 'Protect']
        pokemon = curr_turn.get_pokemon_from_str(data[0])
        effect = PEEffect.from_showdown_message(data[1])
        if effect == PEEffect.PROTECT:
            pokemon.protected = True
    
    else:
        if data and data[0].startswith(">>>"):
            # leaked browser console messages?
            pass
        else:
            raise UnimplementedMessage(row)


def forward_fill(replay: ParsedReplay, log: list[list[str]], verbose: bool = False) -> ParsedReplay:
    for row in log:
        if row:
            if verbose:
                print(f"{replay.gameid} {row}")
            parse_row(replay, row)

    checks.check_noun_spelling(replay)
    checks.check_finished(replay)
    checks.check_replay_rules(replay)
    checks.check_forward_consistency(replay)
    return replay
