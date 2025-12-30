import copy
from typing import List

from metamon.backend.replay_parser import checks
from metamon.backend.replay_parser.replay_state import (
    Action,
    BackwardMarkers,
    Boosts,
    Move,
    Nothing,
    Pokemon,
    Winner,
    TargetedBy,
    Targeting,
    Replacement,
    ParsedReplay,
)
from metamon.backend.replay_parser.pe_datatypes import (
    PEEffect,
    PEField,
    PESideCondition,
    PEStatus,
    PEWeather,
    PEStackableConditions,
)
from metamon.backend.replay_parser.str_parsing import *
from metamon.backend.replay_parser.exceptions import *


class SimProtocol:
    """State-tracking from Showdown "battle" (sim protocol) messages

    https://github.com/smogon/pokemon-showdown/blob/master/sim/SIM-PROTOCOL.md

    Originally based on (and intended to be 1:1 with)
    https://github.com/hsahovic/poke-env/blob/master/src/poke_env/environment/abstract_battle.py
    except that it emphasized "offline" situation where we don't expect Showdown
    "request" messages to help us out, and identified failure cases that should be skipped
    because that help was truly needed.

    Now that there is a `metamon` battle backend, more flexible changes are allowed.
    """

    IGNORES = {
        "",
        "-anim",
        "askreg",
        "badge",
        "bigerror",  # usually auto-tie warnings
        "c",
        "c:",
        "chatmsg-raw",
        "-crit",  # redundant
        "chat",
        "clearpoke",
        "debug",
        "deinit",
        "error",
        "-fieldactivate",  # redundant
        "gametype",
        "hidelines",  # undocumented, no idea
        "-hint",
        "hint",
        "html",
        "-hitcount",
        "init",
        "inactive",  # battle timer
        "inactiveoff",  # battle timer
        "j",
        "J",
        "join",
        "leave",
        "l",
        "L",
        "message",
        "-message",  # chat
        "-miss",
        "n",
        "-nothing",  # redundant for a move that did "absolutely nothing"
        "-notarget",  # for move target
        "-ohko",
        "-prepare",  # extra move info
        "-primal",  # based soley on action
        "raw",
        "rated",
        "request",
        "-resisted",
        "start",
        "-supereffective",
        "-singlemove",
        "seed",
        "sentchoice",
        "teampreview",
        "title",
        "tier",
        "t:",  # timer
        "upkeep",
        "uhtml",
        "uhtmlchange",
        "unlink",  # disconnect or spectator removed
        "-zbroken",  # z-move hits through protect
    }

    # https://bulbapedia.bulbagarden.net/wiki/Category:Moves_that_switch_the_user_out
    MOVES_THAT_SWITCH_THE_USER_OUT = {
        "Baton Pass",
        "Chilly Reception",
        "Flip Turn",
        "Parting Shot",
        "Shed Tail",
        # "Teleport", # technically does this but always fails in trainer battles
        "U-turn",
        "Volt Switch",
    }

    # https://bulbapedia.bulbagarden.net/wiki/Category:Moves_that_switch_the_target_out
    MOVES_THAT_SWITCH_THE_TARGET_OUT = {
        "Whirlwind",
        "Roar",
        "Dragon Tail",
        "Circle Throw",
    }

    FORCES_REVIVAL = {
        "Revival Blessing",
    }

    # https://bulbapedia.bulbagarden.net/wiki/Category:Moves_that_call_other_moves
    # moves that call another move, where the move that is chosen does not necessarily
    # reveal a move this Pokemon actually has.
    MOVE_OVERRIDE = {
        "Assist",
        "Copycat",
        "Me First",
        "Metronome",
        "Mirror Move",
        "Nature Power",
        "Snatch",
        "Magic Coat",
    }
    # moves that call another move, where the move that is chosen is from the Pokemon's moveset
    # and therefore might reveal information.
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

    # partial trapping moves that cause the gen 1 PP rollover to 63
    GEN1_PP_ROLLOVERS = {"Bind", "Wrap", "Fire Spin", "Clamp"}

    # https://bulbapedia.bulbagarden.net/wiki/Category:Moves_that_restore_HP
    RESTORES_PP = {"Lunar Dance"}

    RESTORES_STATUS = {"Healing Wish", "Lunar Dance"}

    # approved to ignore in `move` [from] ability: messages
    MOVE_CAUSED_BY_ABILITY = {"Magic Bounce", "Dancer"}

    ABILITY_STEALS_ABILITY = {"Trace"}

    # heal messages associated with key ability should indicate that
    # the value move failed to force a switch
    ABILITY_CAUSES_MOVE_TO_FAIL = {
        "Water Absorb": "Flip Turn",
        "Dry Skin": "Flip Turn",
        "Lightning Rod": "Volt Switch",
        "Volt Absorb": "Volt Switch",
        "Storm Drain": "Flip Turn",
    }

    # https://bulbapedia.bulbagarden.net/wiki/Category:Item-manipulating_moves
    # we are missing some of these; lookout for UnhandledFromMoveItemLogic
    ITEM_APPROVED_SKIP = {"Knock Off", "Recycle", "Fling", "Corrosive Gas"}
    ITEM_UNNAMED_STOLEN = {"Trick", "Switcheroo"}
    ITEM_NAMED_STOLEN = {"Thief", "Covet"}
    ITEMS_THAT_SWITCH_THE_USER_OUT = {"Eject Button", "Eject Pack"}
    ITEMS_THAT_SWITCH_THE_ATTACKER_OUT = {"Red Card"}

    # https://bulbapedia.bulbagarden.net/wiki/Skill_Swap_(move)
    # I can't tell if Showdown will send the skill swap messages when these happen. keep track of them for now...
    SKILL_SWAP_FAILS = {
        "Wonder Guard",
        "Multitype",
        "Illusion",
        "Stance Change",
        "Schooling",
        "Comatose",
        "Shields Down",
        "Disguise",
        "RKS System",
        "Battle Bond",
        "Power Construct",
        "Ice Face",
        "Gulp Missile",
        "Neutralizing Gas",
        "As One",
        "Zero to Hero",
        "Commander",
        "Protosynthesis",
        "Quark Drive",
        "Orichalcum Pulse",
        "Hadron Engine",
        "Poison Puppeteer",
    }

    def __init__(self, replay: ParsedReplay):
        self.replay = replay

    @property
    def curr_turn(self):
        return self.replay.turnlist[-1]

    def _parse_gen(self, args: List[str]):
        """
        |gen|GENNUM
        """
        self.replay.gen = int(args[0])
        if not (self.replay.gen <= 4 or self.replay.gen == 9):
            raise SoftLockedGen(self.replay.gen)
        if self.replay.gen == 9:
            self.curr_turn.can_tera_1 = True
            self.curr_turn.can_tera_2 = True

    def _parse_player(self, args: List[str]):
        """
        |player|PLAYER|USERNAME|AVATAR|RATING
        """
        if len(args) < 2 or not args[1]:
            # skip reintroductions
            return
        if args[0] == "p1":
            slot = 0
        elif args[0] == "p2":
            slot = 1
        else:
            raise RareValueError(
                f"Could not parse player slot from player id `{args[0]}`"
            )
        self.replay.players[slot] = clean_name(args[1])
        if len(args) >= 4 and args[3]:
            self.replay.ratings[slot] = int(args[3])
        else:
            self.replay.ratings[slot] = "Unrated"

    def _parse_teamsize(self, args: List[str]):
        """
        |teamsize|PLAYER|NUMBER
        """
        player, size = args
        size = int(size)
        assert len(self.curr_turn.pokemon_1) == 6
        assert len(self.curr_turn.pokemon_2) == 6
        if player == "p1":
            team = self.curr_turn.pokemon_1
        elif player == "p2":
            team = self.curr_turn.pokemon_2
        while len(team) > size and None in team:
            team.remove(None)
        if size != 6:
            raise UnusualTeamSize(size)

    def _parse_turn(self, args: List[str]):
        """
        |turn|NUMBER
        """
        assert self.curr_turn.turn_number is not None
        new_turn = self.curr_turn.create_next_turn()
        # saves the within-turn-state for the previous turn, but does not continue it
        new_turn.on_end_of_turn()
        self.replay.turnlist.append(new_turn)

    def _parse_win(self, args: List[str]):
        """
        |win|USER
        """
        winner_name = clean_name(args[0])
        if winner_name == self.replay.players[0]:
            self.replay.winner = Winner.PLAYER_1
        elif winner_name == self.replay.players[1]:
            self.replay.winner = Winner.PLAYER_2
        else:
            raise RareValueError(
                f"Unknown winner: {winner_name} with players: {self.replay.players}"
            )

    def _parse_choice(self, args: List[str]):
        """
        |choice|PLAYER_CHOICES

        https://github.com/smogon/pokemon-showdown/blob/master/sim/SIM-PROTOCOL.md#sending-decisions
        `choice` messages reveal player action choices directly instead of waiting to see the outcome
        of the move/switch. They are not present in every replay, but when they are, they can help us
        fill missing actions. It would be possible to catch many more choices if we could use the numeric
        arg format of some messages (e.g. `move 1`). We'd need to infer a mapping between the numeric args
        and the move names.
        """
        for player_idx, player_choice in enumerate(args):
            if not player_choice:
                continue
            for poke_idx, poke_choice in enumerate(player_choice.split(",")):
                msg = poke_choice.split(" ")
                command = msg[0]
                choice_args = re.sub(r"\d+", "", " ".join(msg[1:])).strip()
                if (
                    command == "move"
                    and choice_args
                    and choice_args.lower() not in {"recharge", "struggle"}
                ):
                    user_pokemon = (
                        self.curr_turn.active_pokemon_1[poke_idx]
                        if player_idx == 0
                        else self.curr_turn.active_pokemon_2[poke_idx]
                    )
                    move = Move(name=choice_args, gen=self.replay.gen)
                    choice = Action(
                        name=move.name,
                        is_switch=False,
                        is_noop=False,
                        user=user_pokemon,
                        target=None,
                    )
                    user_pokemon.reveal_move(move)
                    if player_idx == 0:
                        self.curr_turn.choices_1[poke_idx] = choice
                    else:
                        self.curr_turn.choices_2[poke_idx] = choice

    def _parse_poke(self, args: List[str]):
        """
        |poke|PLAYER|DETAILS|ITEM
        """
        gen = self.replay.gen
        poke_list = self.curr_turn.get_pokemon_list_from_str(args[0])
        assert isinstance(poke_list, list)
        if None not in poke_list:
            raise UnusualTeamSize(len(poke_list) + 1)
        poke_name, lvl = Pokemon.identify_from_details(args[1], gen=gen)
        insert_at = poke_list.index(None)
        new_pokemon = Pokemon(name=poke_name, lvl=lvl, gen=gen)
        # add to team, where it will change throughout the battle
        poke_list[insert_at] = new_pokemon
        sub_str = args[0][0:2]
        # add to teampreview, where it will be frozen
        if sub_str == "p1":
            self.curr_turn.teampreview_1.append(copy.deepcopy(new_pokemon))
        elif sub_str == "p2":
            self.curr_turn.teampreview_2.append(copy.deepcopy(new_pokemon))
        else:
            raise RareValueError(f"Unknown player: {sub_str}")

    def get_or_create_pokemon_from_details(
        self, details: str, poke_list: list[Pokemon]
    ) -> Pokemon:
        # i think matching by `had_name` alone is enough now that it's the official
        # base species, but am unwilling to risk that change. So, for now, match by
        # `name` first (like we always have), then fallback to `had_name`.
        poke_name, lvl = Pokemon.identify_from_details(details, gen=self.replay.gen)
        for p in poke_list:
            if p and p.name == poke_name:
                return p
        poke_base_name, _ = Pokemon.identify_from_details(
            details, gen=self.replay.gen, get_base_species=True
        )
        for p in poke_list:
            if p and p.had_name == poke_base_name:
                return p
        # if we get here, it's a newly introduced pokemon.
        poke = Pokemon(name=poke_name, lvl=lvl, gen=self.replay.gen)
        if None not in poke_list:
            raise CantIDSwitchIn(details, poke_list)
        insert_at = poke_list.index(None)
        poke_list[insert_at] = poke
        return poke

    def _parse_switch_drag(self, args: List[str], name: str):
        """
        |switch|POKEMON|DETAILS|HP STATUS or |drag|POKEMON|DETAILS|HP STATUS
        """
        if len(args) < 3:
            raise UnfinishedMessageException([name] + args)

        # fill the forced switch state
        switch_team, switch_slot = self.curr_turn.player_id_to_action_idx(args[0])
        is_force_switch = False
        player_subturn = None
        for subturn in self.curr_turn.subturns:
            if subturn.matches_slot(switch_team, switch_slot):
                is_force_switch = True
                player_subturn = subturn
            if subturn.unfilled:
                subturn.fill_turn(self.curr_turn.create_subturn(True))

        # id switch out
        poke_list = self.curr_turn.get_pokemon_list_from_str(args[0])
        assert poke_list
        active_poke_list = self.curr_turn.get_active_pokemon_from_str(args[0])
        current_active = active_poke_list[switch_slot]
        if current_active is not None:
            current_active.on_switch_out()

        # id switch in
        poke = self.get_or_create_pokemon_from_details(args[1], poke_list)
        active_poke_list[switch_slot] = poke
        cur_hp, max_hp = parse_hp_fraction(args[2])
        poke.max_hp = max_hp
        poke.current_hp = cur_hp

        if name == "switch":
            # mark intentional "Switch" action
            if is_force_switch:
                player_subturn.action = Action(
                    name="Switch", user=current_active, target=poke, is_switch=True
                )
            else:
                self.curr_turn.set_move_attribute(
                    args[0][:3],
                    move_name="Switch",
                    is_noop=False,
                    is_switch=True,
                    user=current_active,
                    target=poke,
                )

    def _parse_move(self, args: List[str]):
        """
        |move|POKEMON|MOVE|TARGET
        """
        if len(args) < 2:
            raise UnfinishedMessageException(["move"] + args)

        # id pokemon
        poke_str = args[0][:3]
        pokemon = self.curr_turn.get_pokemon_from_str(
            args[0], fallback_to_nickname=False
        )
        move_name = args[1]
        move = Move(name=move_name, gen=self.replay.gen)
        probably_repeat_move = False

        # id target
        target_pokemon, target_team_idx, target_slot_idx = None, None, None
        if len(args) > 2:
            target_pokemon = self.curr_turn.get_pokemon_from_str(
                args[2], fallback_to_nickname=False
            )
            if target_pokemon:
                target_team_idx, target_slot_idx = (
                    self.curr_turn.player_id_to_action_idx(args[2])
                )

        # find extra info from the message
        extra_from_message = None
        for d in reversed(args):
            if "[from]" in d:
                extra_from_message = d
                break

        # forced selection of another pokemon
        if move_name in SimProtocol.MOVES_THAT_SWITCH_THE_USER_OUT:
            notarget = any("[notarget]" in d for d in args)
            protected = target_pokemon.protected if target_pokemon else False
            missed = any("[miss]" in d for d in args)
            if not notarget and not protected and not missed:
                self.curr_turn.mark_forced_switch(args[0])
        elif move_name in SimProtocol.FORCES_REVIVAL:
            self.curr_turn.mark_forced_switch(args[0])

        if extra_from_message:
            # fishing for "moves called by moves that call other moves", which should be prevented
            # from being incorrectly added to a pokemon's true moveset.
            override_risk = (
                move_name in SimProtocol.CONSECUTIVE_MOVES
                or move.charge_move
                or move_name in SimProtocol.GEN1_PP_ROLLOVERS
            )

            # two equivalent logic blocks:
            if "move:" in extra_from_message or "ability:" in extra_from_message:
                # 1. parse [from] effect [of] messages that specify whether they are talking
                # about a move or ability, which are much less ambiguous, but not always appear.
                _, is_ability, is_move, _ = parse_from_effect_of([extra_from_message])
                if is_move:
                    from_move = parse_move_from_extra(extra_from_message)
                    probably_repeat_move = from_move.lower() == move_name.lower()
                    if (
                        from_move
                        in SimProtocol.MOVE_OVERRIDE_BUT_REVEAL_ANYWAY
                        | SimProtocol.MOVE_OVERRIDE
                    ):
                        if override_risk and len(pokemon.had_moves) < 4:
                            raise CalledForeignConsecutive(["move"] + args)
                        if from_move in SimProtocol.MOVE_OVERRIDE_BUT_REVEAL_ANYWAY:
                            pokemon.reveal_move(move)
                        return
                elif is_ability:
                    if is_ability in SimProtocol.MOVE_CAUSED_BY_ABILITY:
                        # this "move" was automatically called by an ability,
                        # and shouldn't be considered a "move" that reveals anything
                        # about this Pokemon or uses PP --- early exit.
                        if (
                            move_name in SimProtocol.MOVES_THAT_SWITCH_THE_USER_OUT
                            and target_pokemon is not None
                            and target_pokemon.last_used_move_name == move_name
                        ):
                            # the move being stolen by the ability is cancelling the opponent's forced switch
                            self.curr_turn.remove_empty_subturn(
                                team=target_team_idx, slot=target_slot_idx
                            )
                        return
                    else:
                        raise UnimplementedMoveFromMoveAbility(args)
            else:
                # 2. parse messages that do not specify move: or ability:. I used to think these
                # were caused by ancient replays (because specifying move/ability is clearly better)
                # but they are still present in recent battles so I no longer know the cause of this.
                ability_or_move = parse_extra(extra_from_message)
                probably_repeat_move = ability_or_move.lower() == move_name.lower()
                probably_item = ability_or_move in (
                    {pokemon.had_item, pokemon.active_item}
                    | SimProtocol.MOVE_IGNORE_ITEMS
                )
                if not (probably_repeat_move or probably_item):
                    if (
                        ability_or_move
                        in SimProtocol.MOVE_OVERRIDE_BUT_REVEAL_ANYWAY
                        | SimProtocol.MOVE_OVERRIDE
                    ):
                        if override_risk and len(pokemon.had_moves) < 4:
                            raise CalledForeignConsecutive(["move"] + args)
                        if (
                            ability_or_move
                            in SimProtocol.MOVE_OVERRIDE_BUT_REVEAL_ANYWAY
                        ):
                            pokemon.reveal_move(move)
                        return
                probably_ability = (
                    ability_or_move.lower() not in {"lockedmove", "pursuit"}
                    and not probably_item
                    and not probably_repeat_move
                )
                if probably_ability:
                    if ability_or_move in SimProtocol.MOVE_CAUSED_BY_ABILITY:
                        return
                    else:
                        raise UnimplementedMoveFromMoveAbility(args)

        # how much PP is used?
        pressured = (
            target_pokemon is not None
            and target_pokemon.active_ability == "Pressure"
            and target_pokemon != pokemon
            and "[notarget]" not in args
        )
        if move.charge_move:
            if "[still]" in args:
                pp_used = 0
            else:
                pp_used = 1 + pressured
        elif self.replay.gen == 1:
            # gen1 partial trapping PP counting edge cases
            if probably_repeat_move:
                # the turns that auto apply partial trapping moves w/o PP have `move USER MOVE TARGET [from]MOVE`
                pp_used = 0
            elif (
                move_name in SimProtocol.GEN1_PP_ROLLOVERS
                and pokemon.get_pp_for_move_name(move_name) == 0
            ):
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
        pokemon.last_target = Targeting(
            pokemon=target_pokemon,
            move=move_name,
        )
        if target_pokemon:
            target_pokemon.last_targeted_by = TargetedBy(
                pokemon=pokemon,
                move=move_name,
            )
        # create Action
        self.curr_turn.set_move_attribute(
            s=poke_str,
            move_name=move.name,
            is_noop=False,
            is_switch=False,
            user=pokemon,
            target=target_pokemon,
        )

    def _parse_damage_heal(self, args: List[str], name: str):
        """
        |-damage|POKEMON|HP STATUS or |-heal|POKEMON|HP STATUS
        """
        if len(args) < 2 or (len(args) == 2 and not args[-1]):
            raise UnfinishedMessageException([name] + args)

        pokemon = self.curr_turn.get_pokemon_from_str(args[0])
        assert pokemon is not None

        if len(args) > 2:
            # parse extra info for items and abilities
            found_item, found_ability, found_move, found_of_pokemon = (
                parse_from_effect_of(args)
            )
            if found_move:
                if found_move in SimProtocol.FORCES_REVIVAL:
                    switch_team, switch_slot = self.curr_turn.player_id_to_action_idx(
                        args[0]
                    )
                    for subturn in self.curr_turn.subturns:
                        if subturn.matches_slot(switch_team, switch_slot):
                            if subturn.unfilled:
                                subturn.fill_turn(self.curr_turn.create_subturn(True))
                            # hardcoded action type that gets converted to similar action idices to "switch"
                            subturn.action = Action(
                                name="$Forced Revival$",
                                user=None,
                                target=pokemon,
                                is_switch=False,
                                is_revival=True,
                            )
                            break
                    if pokemon.status == PEStatus.FNT:
                        pokemon.status = Nothing.NO_STATUS
                if found_move in SimProtocol.RESTORES_PP:
                    for move_name, move in pokemon.moves.items():
                        move.pp = move.maximum_pp
                        if move_name in pokemon.had_moves:
                            had_move = pokemon.had_moves[move_name]
                            had_move.pp = had_move.maximum_pp
                if (
                    found_move in SimProtocol.RESTORES_STATUS
                    and pokemon.status != PEStatus.FNT
                ):
                    pokemon.status = Nothing.NO_STATUS
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
                        of_pokemon = self.curr_turn.get_pokemon_from_str(
                            found_of_pokemon
                        )
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
                    # dealing with edge case of switching move failure due to the target's ability
                    self._cancel_opponent_switch_based_on_user_ability(
                        user_pokemon=pokemon,
                        based_on_ability=found_ability,
                    )
                else:
                    of_pokemon = (
                        self.curr_turn.get_pokemon_from_str(found_of_pokemon)
                        if found_of_pokemon
                        else pokemon
                    )
                # reveal found ability
                of_pokemon.reveal_ability(found_ability)

        # take or heal damage
        if "fnt" in args[1] or args[1] == "0" or args[1][:2] == "0 ":
            pokemon.current_hp = 0
            pokemon.status = PEStatus.FNT
        else:
            if "/" not in args[1]:
                raise UnfinishedMessageException([name] + args)
            cur_hp, max_hp = parse_hp_fraction(args[1])
            pokemon.current_hp = cur_hp
            pokemon.max_hp = max_hp

    def _parse_sethp(self, args: List[str]):
        """
        |-sethp|POKEMON|HP
        """
        pokemon = self.curr_turn.get_pokemon_from_str(args[0])
        assert pokemon is not None
        cur_hp, max_hp = parse_hp_fraction(args[1])
        if pokemon.max_hp:
            assert max_hp == pokemon.max_hp
        pokemon.current_hp = cur_hp

    def _parse_faint(self, args: List[str]):
        """
        |faint|POKEMON
        """
        pokemon = self.curr_turn.get_pokemon_from_str(args[0])
        assert pokemon is not None
        pokemon.current_hp = 0
        pokemon.status = PEStatus.FNT
        self.curr_turn.mark_forced_switch(args[0])

    def _parse_status_curestatus(self, args: List[str], name: str):
        """
        |-status|POKEMON|STATUS or |-curestatus|POKEMON|STATUS
        """
        pokemon = self.curr_turn.get_pokemon_from_str(args[0])
        found_item, found_ability, found_move, found_mon = parse_from_effect_of(args)
        assert pokemon is not None
        status = PEStatus[args[1].upper()]
        if name == "-status":
            pokemon.status = status
        elif pokemon.status == status:
            pokemon.status = Nothing.NO_STATUS
        if found_mon and found_ability:
            found_mon = self.curr_turn.get_pokemon_from_str(found_mon)
            found_mon.reveal_ability(found_ability)
        if found_mon and found_item:
            raise ForwardException(
                "Detected `status` message with unimplemented behavior"
            )

    def _parse_boost_unboost(self, args: List[str], name: str):
        """
        |-boost|POKEMON|STAT|AMOUNT or |-unboost|POKEMON|STAT|AMOUNT
        """
        if len(args) < 3:
            raise UnfinishedMessageException([name] + args)
        change = int(args[2])
        if name == "-unboost":
            change *= -1
        pokemon = self.curr_turn.get_pokemon_from_str(args[0])
        assert pokemon is not None
        pokemon.boosts.change_with_str(args[1], change)

    def _parse_swapboost(self, args: List[str]):
        """
        |-swapboost|SOURCE|TARGET|STATS
        """
        pokemon_1 = self.curr_turn.get_pokemon_from_str(args[0])
        pokemon_2 = self.curr_turn.get_pokemon_from_str(args[1])
        if "[from]" in args[2]:
            if "Heart Swap" in args[2]:
                stats = ["atk", "spa", "def", "spd", "spe", "accuracy", "evasion"]
            elif "Guard Swap" in args[2]:
                stats = ["def", "spd"]
            else:
                raise UnimplementedSwapboost(["swapboost"] + args)
        else:
            stats = args[2].split(", ")
        temp = copy.deepcopy(pokemon_1.boosts)
        for stat in stats:
            pokemon_1.boosts.set_to_with_str(stat, pokemon_2.boosts.get_boost(stat))
            pokemon_2.boosts.set_to_with_str(stat, temp.get_boost(stat))

    def _parse_swap(self, args: List[str]):
        """
        |swap|POKEMON|POSITION
        """
        raise UnimplementedMessage(["swap"] + args)

    def _parse_ability(self, args: List[str]):
        """
        |-ability|POKEMON|ABILITY|[from]EFFECT
        """
        if len(args) < 2:
            raise UnfinishedMessageException(["-ability"] + args)
        pokemon = self.curr_turn.get_pokemon_from_str(args[0])
        ability = parse_ability(args[1])
        found_item, found_ability, found_move, found_mon = parse_from_effect_of(args)
        self._cancel_opponent_switch_based_on_user_ability(
            user_pokemon=pokemon,
            based_on_ability=ability,
        )
        if found_mon and found_ability:
            if found_ability in SimProtocol.ABILITY_STEALS_ABILITY:
                # ['p1a: Porygon2', 'Clear Body', '[from] ability: Trace', '[of] p2a: Dragapult']
                # Porygon2 has the Trace ability, copying Clear Body from Dragapult
                pokemon.active_ability = ability  # # porygon now has clear body
                if pokemon.had_ability is None:
                    pokemon.had_ability = found_ability  # porygon used to have trace
                self.curr_turn.get_pokemon_from_str(found_mon).reveal_ability(
                    ability
                )  # dragapult has clear body
            else:
                raise UnhandledFromOfAbilityLogic(["-ability"] + args)
        elif (found_item or found_mon or found_move) and found_ability:
            raise UnhandledFromOfAbilityLogic(["-ability"] + args)
        else:
            pokemon.reveal_ability(ability)

    def _parse_endability(self, args: List[str]):
        """
        |-endability|POKEMON
        """
        pokemon = self.curr_turn.get_pokemon_from_str(args[0])
        pokemon.active_ability = Nothing.NO_ABILITY
        if len(args) > 1:
            ability = parse_ability(args[1])
            if pokemon.had_ability is None:
                pokemon.had_ability = ability

    def _parse_side_conditions(self, args: List[str], name: str):
        """
        |-sidestart|SIDE|CONDITION or |-sideend|SIDE|CONDITION or |-swapsideconditions
        """
        if "start" in name or "end" in name:
            side_str = args[0][0:2]
            if side_str == "p1":
                side = self.curr_turn.conditions_1
            elif side_str == "p2":
                side = self.curr_turn.conditions_2
            else:
                raise RareValueError(
                    f"Can't find side conditions from identifier `{args[0]}`"
                )
            if len(args) < 2:
                raise UnfinishedMessageException([name] + args)
            condition = PESideCondition.from_showdown_message(args[1])
            if "start" in name:
                if condition in PEStackableConditions:
                    side[condition] = side.get(condition, 0) + 1
                elif condition not in side:
                    side[condition] = self.curr_turn.turn_number
            else:
                if condition in side and condition != PESideCondition.UNKNOWN:
                    side.pop(condition)
        else:
            self.curr_turn.conditions_1, self.curr_turn.conditions_2 = (
                self.curr_turn.conditions_2,
                self.curr_turn.conditions_1,
            )

    def _parse_weather(self, args: List[str]):
        """
        |-weather|WEATHER
        """
        if args[0] == "none":
            self.curr_turn.weather = Nothing.NO_WEATHER
        else:
            self.curr_turn.weather = PEWeather.from_showdown_message(args[0])
        found_item, found_ability, found_move, found_of_mon = parse_from_effect_of(args)
        if found_of_mon:
            pokemon = self.curr_turn.get_pokemon_from_str(found_of_mon)
            assert pokemon is not None
            if found_ability:
                pokemon.reveal_ability(found_ability)

    def _parse_activate(self, args: List[str]):
        """
        |-activate|EFFECT

        also the catch-all message PS sends for minor edge cases
        """
        pokemon = self.curr_turn.get_pokemon_from_str(args[0])
        assert pokemon is not None
        if args[1].startswith("ability:"):
            ability = parse_ability(args[1])
            pokemon.reveal_ability(ability)
            return

        effect = PEEffect.from_showdown_message(args[1])
        # edge cases
        found_item, found_ability, found_move, found_mon = parse_from_effect_of(args)
        if effect == PEEffect.TRICK:
            if found_mon:
                found_mon = self.curr_turn.get_pokemon_from_str(found_mon)
                pokemon.tricking = found_mon
                found_mon.tricking = pokemon
            else:
                raise TrickError(["-activate"] + args)
        elif effect == PEEffect.MIMIC:
            pokemon.mimic(move_name=args[2])
            self.replay.add_warning(WarningFlags.MIMIC)
        elif effect in [PEEffect.LEPPA_BERRY, PEEffect.MYSTERY_BERRY]:
            # https://bulbapedia.bulbagarden.net/wiki/Category:PP-restoring_items
            pp_gained = 10 if effect == PEEffect.LEPPA_BERRY else 5
            move_name = args[2]
            if move_name in pokemon.moves:
                pokemon.moves[move_name].pp += pp_gained
                if move_name in pokemon.had_moves:
                    pokemon.had_moves[move_name].pp += pp_gained
        elif (
            effect == PEEffect.SKILL_SWAP
            and pokemon.last_target is not None
            and pokemon.last_target.pokemon is not None
        ):
            target = pokemon.last_target.pokemon
            target_ability = target.active_ability
            pokemon_ability = pokemon.active_ability
            if (
                target_ability in SimProtocol.SKILL_SWAP_FAILS
                or pokemon_ability in SimProtocol.SKILL_SWAP_FAILS
            ):
                raise ForwardException("Detected Skill Swap failure with patch TODO")
            target.active_ability = pokemon_ability
            pokemon.active_ability = target_ability
        pokemon.start_effect(effect)

    def _parse_item_enditem(self, args: List[str], name: str):
        """
        |-item|POKEMON|ITEM|[from]EFFECT or |-enditem|POKEMON|ITEM|[from]EFFECT
        """
        pokemon = self.curr_turn.get_pokemon_from_str(args[0])
        item = args[1]
        if pokemon is None:
            raise RareValueError(f"Could not find pokemon from {args[0]}")

        found_item, found_ability, found_move, found_mon = parse_from_effect_of(args)
        if found_move:
            if found_move in SimProtocol.ITEM_APPROVED_SKIP:
                pass
            elif found_move in SimProtocol.ITEM_UNNAMED_STOLEN:
                # item is stolen from the opponent using a move
                if not pokemon.tricking:
                    raise TrickError([name] + args)
                if pokemon.tricking.had_item is None:
                    pokemon.tricking.had_item = item
            elif found_move in SimProtocol.ITEM_NAMED_STOLEN:
                # item is stolen from a named opponent.
                if "end" in name:
                    pokemon_that_had_the_item = pokemon
                else:
                    pokemon_that_had_the_item = self.curr_turn.get_pokemon_from_str(
                        found_mon
                    )
                # remove item
                pokemon_that_had_the_item.active_item = Nothing.NO_ITEM
                if pokemon_that_had_the_item.had_item is None:
                    pokemon_that_had_the_item.had_item = item
                # if we still don't know the pokemon's item here, make sure we never fill it in
                if pokemon.had_item is None:
                    pokemon.had_item = BackwardMarkers.FORCE_UNKNOWN
            else:
                raise UnhandledFromMoveItemLogic([name] + args)
        elif pokemon.had_item is None:
            pokemon.had_item = item

        # adjust active item
        if "end" in name:
            pokemon.active_item = Nothing.NO_ITEM
            if (
                item in SimProtocol.ITEMS_THAT_SWITCH_THE_USER_OUT
                and found_move is None
            ):
                # catch Eject Button and Eject Pack messages (which - if activated - would not have an item component?)
                self.curr_turn.mark_forced_switch(args[0])
                self._cancel_opponent_switch_based_on_user_item(
                    user_pokemon=pokemon,
                    based_on_item=item,
                )
            elif (
                item in SimProtocol.ITEMS_THAT_SWITCH_THE_ATTACKER_OUT
                and found_mon is not None
            ):
                team, slot = self.curr_turn.player_id_to_action_idx(found_mon)
                self.curr_turn.remove_empty_subturn(team=team, slot=slot)
        else:
            pokemon.active_item = item

    def _parse_terastallize(self, args: List[str]):
        """
        |-terastallize|POKEMON|TYPE
        """
        pokemon = self.curr_turn.get_pokemon_from_str(args[0])
        poke_str = args[0][:3]
        self.curr_turn.set_move_attribute(
            s=poke_str,
            is_tera=True,
        )
        pokemon.type = [args[1]]
        pokemon.tera_type = args[1]
        team, _ = self.curr_turn.player_id_to_action_idx(poke_str)
        if team == 1:
            self.curr_turn.can_tera_1 = False
        else:
            self.curr_turn.can_tera_2 = False

    def _parse_zpower_mega(self, args: List[str]):
        """
        |-zpower|... or |-mega|...
        """
        raise SoftLockedGen(self.replay.gen)

    def _parse_transform(self, args: List[str]):
        """
        |-transform|POKEMON|SPECIES
        """
        user = self.curr_turn.get_pokemon_from_str(args[0])
        target = self.curr_turn.get_pokemon_from_str(args[1])
        _, found_ability, _, _ = parse_from_effect_of(args)
        if found_ability:
            user.reveal_ability(found_ability)
        user.transform(target)
        self.replay.add_warning(WarningFlags.TRANSFORM)

    def _parse_field_conditions(self, args: List[str], name: str):
        """
        |-fieldstart|CONDITION or |-fieldend|CONDITION
        """
        field_condition = PEField.from_showdown_message(args[0])
        if name == "-fieldstart":
            found_item, found_ability, found_move, found_of_mon = parse_from_effect_of(
                args
            )
            if found_of_mon and found_ability:
                pokemon = self.curr_turn.get_pokemon_from_str(found_of_mon)
                assert pokemon is not None
                pokemon.reveal_ability(found_ability)

            if field_condition.is_terrain:
                self.curr_turn.battle_field = {
                    f: t
                    for f, t in self.curr_turn.battle_field.items()
                    if not f.is_terrain
                }
            self.curr_turn.battle_field[field_condition] = self.curr_turn.turn_number
        else:
            if field_condition != PEField.UNKNOWN:
                self.curr_turn.battle_field.pop(field_condition)

    def _parse_cureteam(self, args: List[str]):
        """
        |-cureteam|POKEMON
        """
        poke_list = self.curr_turn.get_pokemon_list_from_str(args[0])
        for poke in poke_list:
            if poke and poke.status != PEStatus.FNT:
                poke.status = Nothing.NO_STATUS

    def _parse_start_end(self, args: List[str], name: str):
        """
        # |-start|POKEMON|EFFECT or # |-end|POKEMON|EFFECT
        """
        pokemon = self.curr_turn.get_pokemon_from_str(args[0])
        assert pokemon is not None
        effect = PEEffect.from_showdown_message(args[1])
        if effect == PEEffect.MIMIC:
            # 1 of 2 ways PS will tell you which move Mimic copies
            # (depending on gen or replay date it's hard to tell)
            pokemon.mimic(move_name=args[2])
            self.replay.add_warning(WarningFlags.MIMIC)
        found_item, found_ability, found_move, found_mon = parse_from_effect_of(
            args[2:]
        )
        if "start" in name:
            pokemon.start_effect(effect)
        else:
            pokemon.end_effect(effect)
        if found_item or found_ability or found_mon or found_move:
            if found_mon is None:
                of_pokemon = pokemon
            else:
                of_pokemon = self.curr_turn.get_pokemon_from_str(found_mon)
            if found_ability:
                of_pokemon.reveal_ability(found_ability)

    def _parse_setboost(self, args: List[str]):
        """
        |-setboost|POKEMON|STAT|AMOUNT
        """
        pokemon = self.curr_turn.get_pokemon_from_str(args[0])
        assert pokemon is not None
        pokemon.boosts.set_to_with_str(args[1], int(args[2]))

    def _parse_clearboost(self, args: List[str]):
        """
        |-clearboost|POKEMON
        """
        pokemon = self.curr_turn.get_pokemon_from_str(args[0])
        assert pokemon is not None
        pokemon.boosts = Boosts()

    def _parse_clearpositiveboost(self, args: List[str]):
        """
        |-clearpositiveboost|TARGET|POKEMON|EFFECT
        """
        pokemon = self.curr_turn.get_pokemon_from_str(args[0])
        for stat in pokemon.boosts.stat_attrs:
            current = getattr(pokemon.boosts, stat)
            setattr(pokemon.boosts, stat, min(current, 0))

    def _parse_clearnegativeboost(self, args: List[str]):
        """
        |-clearnegativeboost|POKEMON
        """
        pokemon = self.curr_turn.get_pokemon_from_str(args[0])
        for stat in pokemon.boosts.stat_attrs:
            current = getattr(pokemon.boosts, stat)
            setattr(pokemon.boosts, stat, max(current, 0))

    def _parse_copyboost(self, args: List[str]):
        """
        |-copyboost|SOURCE|TARGET
        """
        source = self.curr_turn.get_pokemon_from_str(args[0])
        target = self.curr_turn.get_pokemon_from_str(args[1])
        assert source is not None and target is not None
        source.boosts = copy.deepcopy(target.boosts)

    def _parse_clearallboost(self, args: List[str]):
        """
        |-clearallboost
        """
        for active in [
            self.curr_turn.active_pokemon_1,
            self.curr_turn.active_pokemon_2,
        ]:
            for pokemon in active:
                if pokemon:
                    pokemon.boosts = Boosts()

    def _parse_restoreboost(self, args: List[str]):
        """
        |-restoreboost|p2a: Gorebyss|[silent]
        """
        pokemon = self.curr_turn.get_pokemon_from_str(args[0])
        assert pokemon is not None
        boosts = pokemon.boosts
        for stat_name in boosts.stat_attrs:
            stage = getattr(boosts, stat_name)
            setattr(boosts, stat_name, max(stage, 0))

    def _parse_invertboost(self, args: List[str]):
        """
        |-invertboost|POKEMON
        """
        pokemon = self.curr_turn.get_pokemon_from_str(args[0])
        assert pokemon is not None
        boosts = pokemon.boosts
        for stat_name in boosts.stat_attrs:
            inv = -getattr(boosts, stat_name)
            setattr(boosts, stat_name, inv)

    def _parse_mustrecharge(self, args: List[str]):
        """
        |-mustrecharge|POKEMON
        """
        self.curr_turn.set_move_attribute(
            s=args[0][:3],
            move_name="Recharge",
            is_noop=True,
            is_switch=False,
            user=self.curr_turn.get_pokemon_from_str(args[0]),
            target=None,
        )

    def _parse_cant(self, args: List[str]):
        """
        |cant|POKEMON|REASON or |cant|POKEMON|REASON|MOVE
        """
        # pokemon cannot move and we usually aren't told what the player's preferred action was.
        # the action labels default to None, so we do nothing here.
        pass

    def _parse_immune(self, args: List[str]):
        """
        |-immune|POKEMON
        """
        pokemon = self.curr_turn.get_pokemon_from_str(args[0])
        found_item, found_ability, found_move, found_mon = parse_from_effect_of(args)
        if found_ability:
            pokemon.reveal_ability(found_ability)
        self._cancel_opponent_switch_based_on_user_immunity(
            immune_pokemon=pokemon,
        )

    def _parse_detailschange_formechange(self, args: List[str]):
        """
        |detailschange|POKEMON|DETAILS|HP STATUS or |-formechange|POKEMON|SPECIES|HP STATUS
        """
        pokemon = self.curr_turn.get_pokemon_from_str(args[0])
        if pokemon.had_name is None:
            pokemon.had_name = pokemon.name
        name, lvl = Pokemon.identify_from_details(args[1], gen=self.replay.gen)
        pokemon.update_pokedex_info(name)
        found_item, found_ability, found_move, found_mon = parse_from_effect_of(args)
        if found_ability:
            pokemon.reveal_ability(found_ability)

    def _parse_replace(self, args: List[str]):
        """
        |replace|POKEMON|DETAILS|HP STATUS
        """
        # find the pokemon that is being replaced
        replace_team, replace_slot = self.curr_turn.player_id_to_action_idx(args[0])
        active_pokemon = self.curr_turn.get_active_pokemon_from_str(args[0])
        to_replace = active_pokemon[replace_slot]

        # find the zoroark that is replacing it
        replace_with = None
        poke_list = self.curr_turn.get_pokemon_list_from_str(args[0])
        replace_with_name, _ = Pokemon.identify_from_details(
            args[1], gen=self.replay.gen
        )
        for p in poke_list:
            if p is not None and p.name == replace_with_name:
                replace_with = p
                break
        if replace_with is None or not replace_with.name.startswith("Zoroark"):
            raise ZoroarkException

        # everything that has happened to `replace` since it was last switched
        # in has been wrong. So go back and find the last time it was inactive.
        for prev_turn in self.replay.turnlist[::-1]:
            active_on_turn = prev_turn.get_active_pokemon_from_str(args[0])
            if to_replace not in active_on_turn:
                break
        old_version = copy.deepcopy(prev_turn.get_pokemon_by_uid(to_replace.unique_id))
        for idx, p in enumerate(poke_list):
            if p.unique_id == old_version.unique_id:
                poke_list[idx] = old_version
                break

        # update the active zoroark to the general condition of the mistaken pokemon
        replace_with.status = to_replace.status
        replace_with.current_hp = to_replace.current_hp
        replace_with.max_hp = to_replace.max_hp
        replace_with.boosts = to_replace.boosts
        active_pokemon[replace_slot] = replace_with

        # we're going to have to go back and fix all this during the "backward" pass
        # and it's probably not going to work anyway...
        replacement = Replacement(
            replaced=to_replace,
            replaced_with=replace_with,
            turn_range=(prev_turn.turn_number, self.curr_turn.turn_number),
        )
        if replace_team == 1:
            self.curr_turn.replacements_1.append(replacement)
        else:
            self.curr_turn.replacements_2.append(replacement)
        self.replay.add_warning(WarningFlags.ZOROARK)

    def _parse_burst(self, args: List[str]):
        """
        |-burst|POKEMON|SPECIES|ITEM
        """
        raise UnimplementedMessage(["-burst"] + args)

    def _parse_fail(self, args: List[str]):
        """
        |-fail|POKEMON|ACTION
        """
        pokemon = self.curr_turn.get_pokemon_from_str(args[0])
        from_item, from_ability, from_move, from_mon = parse_from_effect_of(args)
        if from_item is not None and from_mon is not None:
            pokemon.reveal_item(from_item)
        if from_ability is not None and from_mon is not None:
            pokemon.reveal_ability(from_ability)
        self._cancel_user_switch_based_on_failure(user_pokemon=pokemon)
        self._cancel_opponent_parting_shot(
            user_pokemon=pokemon, extra_condition=any("unboost" in s for s in args)
        )

    def _parse_singleturn(self, args: List[str]):
        """
        |-singleturn|POKEMON|EFFECT
        """
        pokemon = self.curr_turn.get_pokemon_from_str(args[0])
        effect = PEEffect.from_showdown_message(args[1])
        if effect == PEEffect.PROTECT:
            pokemon.protected = True

    def _parse_tie(self, args: List[str]):
        """
        |tie
        """
        self.replay.winner = Winner.TIE

    def _parse_rule(self, args: List[str]):
        """
        |rule|RULE: DESCRIPTION
        """
        self.replay.rules.append(args[0])

    def _parse_block(self, args: List[str]):
        """
        |-block|POKEMON|EFFECT|MOVE|ATTACKER
        """
        *_, from_mon = parse_from_effect_of(args)
        if from_mon:
            pokemon = self.curr_turn.get_pokemon_from_str(from_mon)
        else:
            pokemon = self.curr_turn.get_pokemon_from_str(args[0])
        if from_mon and "ability" in args[1]:
            ability = parse_ability(args[1])
            pokemon.reveal_ability(ability)
            self._cancel_opponent_parting_shot(
                user_pokemon=pokemon, extra_condition=True
            )
        else:
            raise ForwardException(
                "Detected `block` message with unimplemented behavior"
            )

    def interpret_message(self, message: List[str]):
        """Interpret and process a single Showdown battle protocol message.

        Parses messages according to the Showdown sim protocol and updates the
        replay state accordingly. Each message is a list where the first element
        is the message type and subsequent elements are the arguments.

        References:
            https://github.com/smogon/pokemon-showdown/blob/master/sim/SIM-PROTOCOL.md
            https://github.com/hsahovic/poke-env/blob/master/src/poke_env/environment/abstract_battle.py

        Args:
            message: List of strings representing a protocol message, where
                    message[0] is the message type and message[1:] are the arguments.
        """
        name, *data = message
        if name in self.IGNORES:
            return
        if name == "gen":
            self._parse_gen(data)
        elif name == "tier":
            # |tier|TIER
            self.replay.format = data[0]
        elif name == "player":
            self._parse_player(data)
        elif name == "teamsize":
            self._parse_teamsize(data)
        elif name == "turn":
            self._parse_turn(data)
        elif name == "win":
            self._parse_win(data)
        elif name == "choice":
            self._parse_choice(data)
        elif name == "tie":
            self._parse_tie(data)
        elif name == "rule":
            self._parse_rule(data)
        elif name == "poke":
            self._parse_poke(data)
        elif name == "switch" or name == "drag":
            self._parse_switch_drag(data, name)
        elif name == "move":
            self._parse_move(data)
        elif name == "-damage" or name == "-heal":
            self._parse_damage_heal(data, name)
        elif name == "-sethp":
            self._parse_sethp(data)
        elif name == "faint":
            self._parse_faint(data)
        elif name == "-status" or name == "-curestatus":
            self._parse_status_curestatus(data, name)
        elif name == "-boost" or name == "-unboost":
            self._parse_boost_unboost(data, name)
        elif name == "-swapboost":
            self._parse_swapboost(data)
        elif name == "swap":
            self._parse_swap(data)
        elif name == "-ability":
            self._parse_ability(data)
        elif name == "-endability":
            self._parse_endability(data)
        elif (
            name == "-sidestart" or name == "-sideend" or name == "-swapsideconditions"
        ):
            self._parse_side_conditions(data, name)
        elif name == "-weather":
            self._parse_weather(data)
        elif name == "-activate":
            self._parse_activate(data)
        elif name == "-item" or name == "-enditem":
            self._parse_item_enditem(data, name)
        elif name == "-terastallize":
            self._parse_terastallize(data)
        elif name == "-zpower" or name == "-mega":
            self._parse_zpower_mega(data)
        elif name == "-transform":
            self._parse_transform(data)
        elif name == "-fieldstart" or name == "-fieldend":
            self._parse_field_conditions(data, name)
        elif name == "-cureteam":
            self._parse_cureteam(data)
        elif name in ["-start", "-end"]:
            self._parse_start_end(data, name)
        elif name == "-setboost":
            self._parse_setboost(data)
        elif name == "-clearboost":
            self._parse_clearboost(data)
        elif name == "-clearpositiveboost":
            self._parse_clearpositiveboost(data)
        elif name == "-clearnegativeboost":
            self._parse_clearnegativeboost(data)
        elif name == "-copyboost":
            self._parse_copyboost(data)
        elif name == "-clearallboost":
            self._parse_clearallboost(data)
        elif name == "-restoreboost":
            self._parse_restoreboost(data)
        elif name == "-invertboost":
            self._parse_invertboost(data)
        elif name == "-mustrecharge":
            self._parse_mustrecharge(data)
        elif name == "cant":
            self._parse_cant(data)
        elif name == "-immune":
            self._parse_immune(data)
        elif name == "detailschange" or name == "-formechange":
            self._parse_detailschange_formechange(data)
        elif name == "replace":
            self._parse_replace(data)
        elif name == "-burst":
            self._parse_burst(data)
        elif name == "-fail":
            self._parse_fail(data)
        elif name == "-singleturn":
            self._parse_singleturn(data)
        elif name == "-block":
            self._parse_block(data)
        else:
            if data and data[0].startswith(">>>"):
                # leaked browser console messages?
                pass
            else:
                raise UnimplementedMessage(message)

    def _cancel_opponent_switch_based_on_user_ability(
        self, user_pokemon: Pokemon, based_on_ability: str
    ) -> bool:
        """Cancel an opponent's switch if the user's ability was activated by a switch-out move.

        Args:
            curr_turn: The current turn being processed
            user_pokemon: The Pokemon that had its ability activated
            based_on_ability: The name of the ability that was activated

        Returns:
            bool: True if the switch was cancelled, False otherwise
        """
        curr_turn = self.curr_turn
        if (
            based_on_ability not in SimProtocol.ABILITY_CAUSES_MOVE_TO_FAIL
            or not user_pokemon.last_targeted_by
        ):
            return False

        if (
            user_pokemon.last_targeted_by.move
            != SimProtocol.ABILITY_CAUSES_MOVE_TO_FAIL[based_on_ability]
        ):
            return False

        subturn_slot = curr_turn.pokemon_to_action_idx(
            user_pokemon.last_targeted_by.pokemon
        )
        if not subturn_slot:
            return False

        curr_turn.remove_empty_subturn(team=subturn_slot[0], slot=subturn_slot[1])
        return True

    def _cancel_opponent_switch_based_on_user_item(
        self, user_pokemon: Pokemon, based_on_item: str
    ) -> bool:
        """Cancel an opponent's switch if the user's item was activated by a switch-out move.

        Args:
            curr_turn: The current turn being processed
            user_pokemon: The Pokemon that had its item activated
            based_on_item: The name of the item that was activated

        Returns:
            bool: True if the switch was cancelled, False otherwise
        """
        curr_turn = self.curr_turn
        last_targeted_by = user_pokemon.last_targeted_by
        if (
            based_on_item not in SimProtocol.ITEMS_THAT_SWITCH_THE_USER_OUT
            or not last_targeted_by
        ):
            return False

        if (
            not last_targeted_by.pokemon
            or last_targeted_by.move not in SimProtocol.MOVES_THAT_SWITCH_THE_USER_OUT
        ):
            return False

        subturn_slot = curr_turn.pokemon_to_action_idx(last_targeted_by.pokemon)
        if not subturn_slot:
            return False

        curr_turn.remove_empty_subturn(team=subturn_slot[0], slot=subturn_slot[1])
        return True

    def _cancel_opponent_switch_based_on_user_immunity(
        self, immune_pokemon: Pokemon
    ) -> bool:
        """Cancel an opponent's switch if the immune Pokemon was targeted by a switch-out move.

        Args:
            curr_turn: The current turn being processed
            immune_pokemon: The Pokemon that is immune to the move

        Returns:
            bool: True if the switch was cancelled, False otherwise
        """
        curr_turn = self.curr_turn
        last_targeted_by = immune_pokemon.last_targeted_by
        if not last_targeted_by:
            return False
        if last_targeted_by.move not in SimProtocol.MOVES_THAT_SWITCH_THE_USER_OUT:
            return False
        subturn_slot = curr_turn.pokemon_to_action_idx(last_targeted_by.pokemon)
        if not subturn_slot:
            return False
        curr_turn.remove_empty_subturn(team=subturn_slot[0], slot=subturn_slot[1])
        return True

    def _cancel_user_switch_based_on_failure(self, user_pokemon: Pokemon) -> bool:
        """Cancel a user's switch if their move failed and it was a switch-out move.

        Args:
            curr_turn: The current turn being processed
            user_pokemon: The Pokemon that failed to use its move

        Returns:
            bool: True if the switch was cancelled, False otherwise
        """
        curr_turn = self.curr_turn
        if (
            user_pokemon.last_used_move is not None
            and user_pokemon.last_used_move.name
            in SimProtocol.MOVES_THAT_SWITCH_THE_USER_OUT
        ):
            team_slot = curr_turn.pokemon_to_action_idx(user_pokemon)
            if team_slot:
                curr_turn.remove_empty_subturn(team=team_slot[0], slot=team_slot[1])
                return True
        return False

    def _cancel_opponent_parting_shot(
        self, user_pokemon: Pokemon, extra_condition: bool
    ) -> bool:
        # https://bulbapedia.bulbagarden.net/wiki/Parting_Shot_(move)
        if user_pokemon.last_targeted_by is None:
            return False
        last_targeted_by = user_pokemon.last_targeted_by
        if (
            self.replay.gen >= 7
            and last_targeted_by.move in {"Parting Shot"}
            and extra_condition
        ):
            opponent_team_slot = self.curr_turn.pokemon_to_action_idx(
                last_targeted_by.pokemon
            )
            if opponent_team_slot is not None:
                opponent_team, opponent_slot = opponent_team_slot
                self.curr_turn.remove_empty_subturn(
                    team=opponent_team, slot=opponent_slot
                )
                return True
        return False


def forward_fill(
    replay: ParsedReplay, log: list[list[str]], verbose: bool = False
) -> ParsedReplay:
    sim_protocol = SimProtocol(replay)
    for message in log:
        if not message:
            continue
        if verbose:
            print(f"{replay.gameid} {message}")
        sim_protocol.interpret_message(message)

    checks.check_forced_switching(replay)
    checks.check_noun_spelling(replay)
    checks.check_finished(replay)
    checks.check_replay_rules(replay)
    checks.check_forward_consistency(replay)
    checks.check_name_permanence(replay)
    checks.check_tera_consistency(replay)
    return replay
