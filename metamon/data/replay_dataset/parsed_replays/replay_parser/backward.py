import copy
import re
from typing import List

from metamon.data.replay_dataset.parsed_replays.replay_parser import checks, forward
from metamon.data.replay_dataset.parsed_replays.replay_parser.exceptions import *
from metamon.data.replay_dataset.parsed_replays.replay_parser.replay_state import (
    Action,
    Boosts,
    Move,
    Nothing,
    Pokemon,
    Turn,
    Winner,
    get_pokedex_and_moves,
    unknown,
)
from metamon.data.team_prediction.predictor import TeamPredictor
from metamon.data.team_prediction.team import TeamSet, PokemonSet
from metamon.data.team_builder.team_builder import PokemonStatsLookupError


def fill_missing_team_info(
    battle_format: str, poke_list: List[Pokemon], team_predictor: TeamPredictor
) -> List[Pokemon]:
    """
    Team prediction works by:

    1. Converting the team we've gathered here in the replay parser to the format expected by the team_prediction module
    2. Predicting the team with a TeamPredictor
    3. Filling missing information with the predicted team
    """

    # 1. Convert the team to the format expected by the team_prediction module
    gen = int(battle_format.split("gen")[1][0])
    poke_names = [p.name for p in poke_list if p is not None]
    converted_poke = [PokemonSet.from_ReplayPokemon(p, gen=gen) for p in poke_list]
    revealed_team = TeamSet(
        lead=converted_poke[0], reserve=converted_poke[1:], format=battle_format
    )

    # 2. Predict the team
    predicted_team = team_predictor.predict(revealed_team)
    if not revealed_team.is_consistent_with(predicted_team):
        raise InconsistentTeamPrediction(revealed_team, predicted_team)

    # 3. Filling missing information with the predicted team
    pokemon_to_add = [
        poke for poke in predicted_team.pokemon if poke.name not in poke_names
    ]
    while None in poke_list and pokemon_to_add:
        generated = pokemon_to_add.pop(0)
        new_pokemon = Pokemon(name=generated.name, lvl=100, gen=gen)
        poke_list[poke_list.index(None)] = new_pokemon

    if None in poke_list:
        raise BackwardException(
            f"Could not fill in all missing pokemon for {poke_list} with {predicted_team}"
        )

    names = [p.name for p in poke_list]
    if len(names) != len(set(names)):
        raise BackwardException(f"Duplicate pokemon names in {names}")

    for p in poke_list:
        for match in predicted_team.pokemon:
            if match.name == p.name:
                break
        else:
            raise BackwardException(f"Could not find match for {p.name}")
        p.fill_from_PokemonSet(match)

    return poke_list, revealed_team


class POVReplay:
    def __init__(
        self,
        replay: forward.ParsedReplay,
        filled_replay: forward.ParsedReplay,
        from_p1_pov: bool,
        revealed_team: TeamSet,
    ):
        if replay.gameid != filled_replay.gameid:
            raise ValueError("Using replays of different games to construct POVReplay")

        self.from_p1_pov = from_p1_pov
        self.revealed_team = revealed_team

        # copy replay metadata
        self.replay_url = filled_replay.replay_url
        self.gameid = filled_replay.gameid
        self.format = filled_replay.format
        self.gen = filled_replay.gen
        self.time_played = filled_replay.time_played
        self.rules = filled_replay.rules

        # rating and winner from POV
        self.rating = filled_replay.ratings[0 if from_p1_pov else 1]
        self.winner = filled_replay.winner == (
            Winner.PLAYER_1 if from_p1_pov else Winner.PLAYER_2
        )

        self.fill_one_side(replay, filled_replay)
        self.align_states_actions(replay)

    def fill_one_side(self, replay, filled_replay):
        # take spectator replay and reveal one entire team from filled_replay
        assert len(replay.flattened_turnlist) == len(filled_replay.flattened_turnlist)
        for turn, filled_turn in zip(
            replay.flattened_turnlist, filled_replay.flattened_turnlist
        ):
            if self.from_p1_pov:
                turn.pokemon_1 = filled_turn.pokemon_1
                turn.active_pokemon_1 = filled_turn.active_pokemon_1
            else:
                turn.pokemon_2 = filled_turn.pokemon_2
                turn.active_pokemon_2 = filled_turn.active_pokemon_2

    def align_states_actions(self, replay: forward.ParsedReplay):
        self.povturnlist = []
        self.actionlist = []
        for idx, (turn_t, turn_t1) in enumerate(
            zip(replay.turnlist, replay.turnlist[1:])
        ):
            # subturns freeze the sim midturn, which we currently
            # only use to replicate forced switches.
            for subturn in turn_t.subturns:
                if subturn.turn is not None and subturn.team == (
                    1 if self.from_p1_pov else 2
                ):
                    action = [None, None]
                    action[subturn.slot] = subturn.action
                    self.povturnlist.append(subturn.turn)
                    self.actionlist.append(action)

            self.povturnlist.append(
                turn_t
            )  # turn_t holds the state at the very end of the turn
            # and the action we clicked between turns is held in the next turn
            moves = turn_t1.moves_1 if self.from_p1_pov else turn_t1.moves_2
            choices = turn_t1.choices_1 if self.from_p1_pov else turn_t1.choices_2
            actionlist = [None, None]
            for move_idx, (move, choice) in enumerate(zip(moves, choices)):
                if move is not None:
                    # we default to the original system of *used* moves
                    actionlist[move_idx] = move
                elif choice is not None:
                    # if the move was missing, but a `choice` message was parsed,
                    # we can fall back to that.
                    actionlist[move_idx] = choice
            self.actionlist.append(actionlist)

        # add final state
        self.povturnlist.append(turn_t1)
        self.actionlist.append([None, None])


def add_filled_final_turn(
    replay: forward.ParsedReplay, team_predictor: TeamPredictor
) -> tuple[forward.ParsedReplay, tuple[TeamSet, TeamSet]]:
    # add an extra turn to a replay with all missing information guessed
    # by sampling from the TeamBuilder. this extra turn can then be moved
    # backwards through the replay and discareded.
    filled_turn = replay[-1].create_next_turn()
    filled_turn.on_end_of_turn()
    filled_turn.pokemon_1, revealed_team_1 = fill_missing_team_info(
        replay.format, replay[-1].pokemon_1, team_predictor=team_predictor
    )
    filled_turn.pokemon_2, revealed_team_2 = fill_missing_team_info(
        replay.format, replay[-1].pokemon_2, team_predictor=team_predictor
    )
    replay.turnlist.append(filled_turn)
    return replay, (revealed_team_1, revealed_team_2)


def resolve_transforms(replay):
    base_stats = {
        pokemon.unique_id: pokemon.base_stats for pokemon in replay[0].all_pokemon
    }
    types = {pokemon.unique_id: pokemon.type for pokemon in replay[0].all_pokemon}

    for prev_turn, turn in zip(replay.turnlist, replay.turnlist[1:]):
        for pokemon in turn.all_pokemon:
            prev_ids = prev_turn.id2pokemon
            on_prev_turn = prev_ids[pokemon.unique_id]

            if pokemon.transformed_into is not None:
                transformed_on_prev = prev_ids[pokemon.transformed_into.unique_id]

                if on_prev_turn.transformed_into != pokemon.transformed_into:
                    # first turn of transformation
                    for move_name in transformed_on_prev.moves.keys():
                        # when we got the -transform message on the forward pass
                        # we copied the moves we already knew about it. now we
                        # add the ones we didn't discover until the backward pass.
                        if move_name not in pokemon.moves:
                            tformed_move = Move(move_name, gen=replay.gen)
                            tformed_move.pp, tformed_move.maximum_pp = 5, 5
                            pokemon.moves[tformed_move.name] = tformed_move

                    # on the forward pass we avoided changing things that would
                    # not reset on switch out. now we make those changes manually.
                    # poke-env does NOT switch the `species` property on transform,
                    # so we don't change the `name`.
                    pokemon.base_stats = transformed_on_prev.base_stats
                    # ...except for hp
                    pokemon.base_stats["hp"] = base_stats[pokemon.unique_id]["hp"]
                    pokemon.active_ability = transformed_on_prev.active_ability
                    pokemon.type = transformed_on_prev.type

                elif pokemon.transformed_into == on_prev_turn.transformed_into:
                    # continued transformation
                    for move_name, move in on_prev_turn.moves.items():
                        # we may have discovered more of the opponents moveset
                        # during the transformation, so the moves we added above
                        # may already be in `moves` with correct PP tracking.
                        if move_name not in pokemon.moves:
                            assert move.maximum_pp == 5
                            pokemon.moves[move_name] = move

                    pokemon.base_stats = on_prev_turn.base_stats
                    pokemon.active_ability = on_prev_turn.active_ability
                    pokemon.type = on_prev_turn.type

            elif on_prev_turn.transformed_into is not None:
                # forward pass logic switches most things back
                pokemon.base_stats = base_stats[pokemon.unique_id]
                pokemon.active_ability = pokemon.had_ability
                pokemon.type = types[pokemon.unique_id]


def backward_fill(
    replay: forward.ParsedReplay, team_predictor: TeamPredictor
) -> tuple[POVReplay, POVReplay]:
    cleaned_format = re.sub(r"\[|\]| ", "", replay.format).lower()
    pokedex, _ = get_pokedex_and_moves(cleaned_format)

    # fill in missing team info at the end of the forward pass
    replay_filled, (revealed_team_1, revealed_team_2) = add_filled_final_turn(
        copy.deepcopy(replay), team_predictor=team_predictor
    )

    # copy that info across the trajectory
    flat_turnlist = replay_filled.flattened_turnlist
    for turn_t, turn_t1 in zip(flat_turnlist[-2::-1], flat_turnlist[::-1]):
        prev_ids = turn_t.id2pokemon
        # first we move information backwards from the current team roster
        # to the previous timestep
        for prev_team, team in (
            (turn_t.pokemon_1, turn_t1.pokemon_1),
            (turn_t.pokemon_2, turn_t1.pokemon_2),
        ):
            for pokemon in team:
                if pokemon.unique_id in prev_ids:
                    prev_pokemon = prev_ids[pokemon.unique_id]
                    prev_pokemon.backfill_info(pokemon)
                else:
                    # pokemon discovered in turn_t1 enters turn_t "fresh"
                    assert None in prev_team
                    prev_team[prev_team.index(None)] = pokemon.fresh_like()

    # chop off the extra filled turn
    replay_filled.turnlist = replay_filled.turnlist[:-1]
    resolve_transforms(replay_filled)
    checks.check_info_filled(replay_filled)

    from_p1 = POVReplay(
        copy.deepcopy(replay),
        copy.deepcopy(replay_filled),
        from_p1_pov=True,
        revealed_team=revealed_team_1,
    )
    checks.check_action_alignment(from_p1)
    from_p2 = POVReplay(
        replay, replay_filled, from_p1_pov=False, revealed_team=revealed_team_2
    )
    checks.check_action_alignment(from_p2)
    return from_p1, from_p2
