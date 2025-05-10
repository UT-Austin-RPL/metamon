from metamon.data.replay_dataset.parsed_replays.replay_parser.exceptions import *
from metamon.data.replay_dataset.parsed_replays.replay_parser.forward import (
    SpecialCategories,
)
from poke_env.environment import Effect as PEEffect
from poke_env.data import to_id_str


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
                    if to_id_str(val) == val:
                        raise ForwardVerify(
                            f"Potential to_id_str --> Proper Name mismatch: {val}, sometimes caused by all-lowercase logs"
                        )


def check_filled_mon(pokemon):
    p = pokemon
    if (
        not isinstance(p.base_stats, dict)
        or p.had_ability is None
        or p.active_ability is None
        or p.had_item is None
        or p.active_item is None
        or (isinstance(p.active_item, str) and not p.active_item.strip())
        or (isinstance(p.active_ability, str) and not p.active_ability.strip())
        or p.current_hp is None
        or p.status is None
        or p.max_hp is None
        or p.lvl is None
        or not p.base_stats
        or None in p.base_stats.values()
        or p.type is None
    ):
        raise BackwardException(f"Pokemon info has not been filled: {p}")

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
        for active_pokemon, action in zip(active, team_actions):
            if (
                action is None
                or action.name in ["Switch", "Struggle"]
                or action.is_noop
            ):
                pass
            elif action.name in active_pokemon.moves.keys():
                pass
            else:
                raise ActionMisaligned(active_pokemon, action)


def check_forced_switching(turn):
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
