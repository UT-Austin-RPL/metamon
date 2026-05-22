from typing import Optional

from metamon.backend.replay_parser.exceptions import *
from metamon.backend.replay_parser.str_parsing import clean_name
from metamon.backend.replay_parser.replay_state import Nothing, unknown


def check_finished(replay):
    if replay.winner is None or len(replay.turnlist) < 5:
        # silent unfinished message, incomplete download/log, or insta-forfeit
        raise UnfinishedReplayException(replay.replay_url)


def check_replay_rules(replay):
    # if the replay didn't use species clause then we probably didn't track
    # the switches correctly. luckily this is a very common rule.
    species_clause = False
    for rule in replay.rules:
        if rule.startswith("Scalemons Mod"):
            raise Scalemons(replay)
        species_clause |= rule.startswith("Species Clause")
    if not species_clause:
        raise NoSpeciesClause(replay)

    # check species clause was effectively maintained
    names = lambda team: [p.name for p in team if p is not None]
    for turn in replay:
        names_1 = names(turn.pokemon_1)
        names_2 = names(turn.pokemon_2)
        if len(names_1) != len(set(names_1)):
            raise ForwardVerify(f"Found duplicate pokemon names on Team 1: {names_1}")
        if len(names_2) != len(set(names_2)):
            raise ForwardVerify(f"Found duplicate pokemon names on Team 2: {names_2}")


def check_forward_consistency(replay):
    # we did not "discover" too many unique pokemon.
    known_ids = set()
    for turn in replay.turnlist:
        for pokemon in turn.all_pokemon:
            if pokemon:
                known_ids.add(pokemon.unique_id)
    last_turn = replay.turnlist[-1]
    expected_max_ids = len([p is not None for p in last_turn.all_pokemon])
    if len(known_ids) > expected_max_ids:
        raise ForwardVerify(
            f"Expected <= {expected_max_ids} unique Pokemon, but found {len(known_ids)}"
        )

    for uid in known_ids:
        # once we found a pokemon, it never switched teams
        found_p1 = (False, None)
        found_p2 = (False, None)
        for i, turn in enumerate(replay.turnlist):
            if (
                uid in [p.unique_id for p in turn.pokemon_1 if p is not None]
                and not found_p1[0]
            ):
                found_p1 = (True, i)
            if (
                uid in [p.unique_id for p in turn.pokemon_2 if p is not None]
                and not found_p2[0]
            ):
                found_p2 = (True, i)
            if found_p1[0] and found_p2[0]:
                raise ForwardVerify(f"Found the same pokemon ID on both teams!")
        found_p1, on1 = found_p1
        found_p2, on2 = found_p2
        assert found_p1 or found_p2
        assert on1 is not None or on2 is not None
        assert on1 is None or on2 is None

        # ...and we never lose track of it
        if found_p1:
            for turn in replay.turnlist[on1:]:
                if uid not in [p.unique_id for p in turn.pokemon_1 if p is not None]:
                    raise ForwardVerify(f"Pokemon ID vanished on Team 1")
        elif found_p2:
            for turn in replay.turnlist[on2:]:
                if uid not in [p.unique_id for p in turn.pokemon_2 if p is not None]:
                    raise ForwardVerify(f"Pokemon ID vanished on Team 2")

        # ...or its item, ability, and full moveset.
        on = on1 or on2
        found_ability = False
        found_item = False
        found_move = False
        for turn in replay.turnlist[on:]:
            pokemon_t = turn.get_pokemon_by_uid(uid)

            if found_item and pokemon_t.had_item is None:
                raise ForwardVerify(f"Lost track of {pokemon_t.name} Item")
            elif (not found_item) and pokemon_t.had_item is not None:
                found_item = True
            if found_ability and pokemon_t.had_ability is None:
                raise ForwardVerify(f"Lost track of {pokemon_t.name} Ability")
            elif (not found_ability) and pokemon_t.had_ability is not None:
                found_ability = True
            if found_move and not pokemon_t.had_moves:
                raise ForwardVerify(f"Lost track of a {pokemon_t.name} MoveSet")
            elif (not found_move) and pokemon_t.had_moves:
                found_move = True

            # check moveset and PP
            if pokemon_t.moves != pokemon_t.had_moves:
                # rare moveset discrepancy edge cases
                had_but_missing = [
                    m
                    for m in (
                        set(pokemon_t.had_moves.keys()) - set(pokemon_t.moves.keys())
                    )
                ]
                has_but_unknown = [
                    m
                    for m in (
                        set(pokemon_t.moves.keys()) - set(pokemon_t.had_moves.keys())
                    )
                ]
                if pokemon_t.transformed_into is not None:
                    # explained by transform
                    pass
                elif (has_but_unknown or had_but_missing) and (
                    "Mimic" in pokemon_t.had_moves.keys()
                ):
                    # explained by Mimic
                    pass
                else:
                    raise ForwardVerify(f"Inconsistent MoveSet for {pokemon_t.name}")

            if len(pokemon_t.moves) > 4 or len(pokemon_t.had_moves) > 4:
                raise ForwardVerify(f"Found too many moves for {pokemon_t.name}")
            if pokemon_t.moves:
                lowest_pp_move = min(pokemon_t.moves.values(), key=lambda m: m.pp)
                if lowest_pp_move.pp < 0:
                    raise ForwardVerify(f"{pokemon_t.name} PP of {lowest_pp_move} < 0")


def check_name_permanence(replay):
    uid_to_had_name = {}
    for turn in replay:
        for pokemon in turn.all_pokemon:
            if pokemon is None:
                continue
            if pokemon.unique_id in uid_to_had_name:
                if uid_to_had_name[pokemon.unique_id] != pokemon.had_name:
                    raise ForwardVerify(f"Found had_name change for {pokemon.name}")
            else:
                uid_to_had_name[pokemon.unique_id] = pokemon.had_name


def check_noun_spelling(replay):
    for turn in replay:
        for pokemon in turn.all_pokemon:
            if pokemon is None:
                continue
            for poke_attr in [
                "name",
                "had_name",
                "active_item",
                "active_ability",
                "active_item",
            ]:
                val = getattr(pokemon, poke_attr)
                if isinstance(val, str):
                    if clean_name(val) == val:
                        raise ForwardVerify(
                            f"Potential clean_name --> Proper Name mismatch: {val}, probably caused by all-lowercase replay logs"
                        )


def check_filled_mon(pokemon):
    p = pokemon
    if (
        not isinstance(p.base_stats, dict)
        or unknown(p.name)
        or unknown(p.had_name)
        or unknown(p.had_ability)
        or unknown(p.active_ability)
        or unknown(p.had_item)
        or unknown(p.active_item)
        or (isinstance(p.active_item, str) and not p.active_item.strip())
        or (isinstance(p.active_ability, str) and not p.active_ability.strip())
        or unknown(p.current_hp)
        or unknown(p.status)
        or unknown(p.max_hp)
        or unknown(p.lvl)
        or not p.base_stats
        or None in p.base_stats.values()
        or unknown(p.type)
    ):
        raise BackwardException(f"Pokemon info has not been filled: {p}")

    if pokemon.gen == 9 and pokemon.tera_type == Nothing.NO_TERA_TYPE:
        raise BackwardException(f"Pokemon {pokemon.name} has no tera type")

    if pokemon.gen > 2 and pokemon.had_ability == Nothing.NO_ABILITY:
        raise BackwardException(f"Pokemon {pokemon.name} has no ability")

    if pokemon.gen > 2 and pokemon.had_item == Nothing.NO_ITEM:
        raise BackwardException(f"Pokemon {pokemon.name} has no item")

    moveset_size = len(pokemon.moves)
    if moveset_size > 4:
        raise TooManyMoves(p)

    # sanity check on annonying spelling changes across move names
    moves_by_lookup = set([m.lookup_name for m in pokemon.moves.values()])
    if len(moves_by_lookup) != moveset_size:
        raise ForwardVerify(
            f"Found duplicate moves for {pokemon.name}: {moves_by_lookup}"
        )


def check_info_filled(replay):
    for turn in replay:
        for pokemon in turn.all_pokemon:
            check_filled_mon(pokemon)


def check_action_alignment(replay):
    for turn, team_actions in zip(replay.povturnlist, replay.actionlist):
        active = turn.active_pokemon_1 if replay.from_p1_pov else turn.active_pokemon_2
        switches = turn.get_switches(replay.from_p1_pov)
        for active_pokemon, action in zip(active, team_actions):
            if action is None or action.name in {"Struggle", "Fight"} or action.is_noop:
                # considered a "no-op"
                continue
            elif action.name == "Switch":
                # standard switch
                if action.target in switches and action.is_switch:
                    continue
            elif action.name in active_pokemon.moves.keys() and not action.is_switch:
                # standard move
                continue
            elif action.name is None and action.is_tera:
                # revealed only the choice to tera (at the start of the turn),
                # but never found out what the move was...
                continue
            elif action.is_revival:
                pokemon = turn.get_pokemon(replay.from_p1_pov)
                if action.target in pokemon and action.target not in switches:
                    # our revival choice is on our team but had fainted (can't be switched to)
                    continue
            raise ActionMisaligned(active_pokemon, action)


def check_action_idxs(
    univeral_states: list,
    actions: list,
    action_idxs: list[int],
    gen: int,
):
    # too lazy to fix circular import for the runtime checks
    from metamon.interface import UniversalAction

    tera = 0
    for state, action, action_idx in zip(univeral_states, actions, action_idxs):
        # check missing actions
        if action is None and action_idx != -1:
            raise ActionIndexError(f"Action is None but action_idx is not -1")
        elif action is None or action_idx == -1:
            continue
        if action_idx > 13 or action_idx < -1:
            raise ActionIndexError(f"Action index {action_idx} is out of bounds")
        # check tera by action idx
        if action_idx >= 9:
            tera += 1
        if tera and gen != 9:
            raise ActionIndexError(f"Found Tera action in gen {gen}")
        if tera > 1:
            raise ActionIndexError(f"Found {tera} Tera actions")
        if action.name in {"Struggle", "Recharge", "Fight"} and action_idx != 0:
            # check struggle and recharge special case move overrides
            raise ActionIndexError(
                f"{action.name} is action index {action_idx}; expected to be 0"
            )
        if state.forced_switch and not (action.is_switch or action.is_revival):
            # check forced switch leads to switch
            raise ActionIndexError(
                f"Forced switch {state.forced_switch} != action {action.is_switch}"
            )
        if action_idx > 9 and (not state.can_tera or not action.is_tera):
            # check tera action index is valid
            raise ActionIndexError(
                f"Found Tera action index {action_idx} but can_tera is False"
            )
        if action.is_switch and (action_idx <= 3 or action_idx >= 9):
            # check move actions become move action indices
            raise ActionIndexError(f"Expected switch action index")
        elif not action.is_switch and not action.is_revival and 4 <= action_idx <= 8:
            # check switch action indices
            raise ActionIndexError(f"Expected move action index")
        # check action index is considered legal by mask helper
        maybe_legal = UniversalAction.maybe_valid_actions(state)
        if UniversalAction(action_idx) not in maybe_legal:
            raise ActionIndexError(
                f"Action index {action_idx} is not found to be legal by UniversalAction"
            )


def check_tera_consistency(replay):
    gen = replay.gen
    can_tera_1 = True
    can_tera_2 = True
    ever_tera_1 = False
    ever_tera_2 = False
    for turn in replay:
        if gen != 9 and (turn.can_tera_1 or turn.can_tera_2):
            raise ForwardVerify("Found Tera in gen != 9")
        if not can_tera_1 and turn.can_tera_1:
            raise MultipleTera("p1")
        if not can_tera_2 and turn.can_tera_2:
            raise MultipleTera("p2")
        ever_tera_1 |= turn.can_tera_1
        ever_tera_2 |= turn.can_tera_2
        can_tera_1 &= turn.can_tera_1
        can_tera_2 &= turn.can_tera_2
    if gen == 9 and not (ever_tera_1 and ever_tera_2):
        raise ForwardVerify("Found no Tera available in gen 9")


def check_forced_switching(replay):
    for turn in replay.turnlist[:-1]:
        # was there a turn where we 1) had to switch, 2) could switch, but 3) didn't record it?
        for subturn in turn.subturns:
            if subturn.turn is None or subturn.action is None:
                switches = (
                    turn.available_switches_1
                    if subturn.team == 1
                    else turn.available_switches_2
                )
                if len(switches) > 0:
                    raise ForceSwitchMishandled(subturn)
