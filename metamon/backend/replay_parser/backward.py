import copy
import datetime
from typing import List, Optional
import collections

import metamon
from metamon.backend.replay_parser import checks
from metamon.backend.replay_parser.exceptions import *
from metamon.backend.replay_parser.replay_state import (
    Action,
    Pokemon,
    Turn,
    Winner,
    BackwardMarkers,
    Replacement,
    ParsedReplay,
)
from metamon.backend.team_prediction.predictor import TeamPredictor
from metamon.backend.team_prediction.team import TeamSet, PokemonSet


def fill_missing_team_info(
    battle_format: str,
    date_played: datetime.date,
    poke_list: List[Pokemon],
    team_predictor: TeamPredictor,
    rating: Optional[int | str] = None,
    gameid: Optional[str] = None,
) -> List[Pokemon]:
    """
    Team prediction works by:

    1. Converting the team we've gathered here in the replay parser to the format expected by the team_prediction module
    2. Predicting the team with a TeamPredictor
    3. Filling missing information with the predicted team
    """

    # 1. Convert the team to the format expected by the team_prediction module
    gen = metamon.backend.format_to_gen(battle_format)
    existing_species = set(p.had_name for p in poke_list if p is not None)
    converted_poke = [PokemonSet.from_ReplayPokemon(p, gen=gen) for p in poke_list]
    # TODO/NOTE: this method of finding the lead pokemon doesn't hold for gen9,
    # where the entire team is revealed at the start of the battle in an unrelated order.
    revealed_team = TeamSet(
        lead=converted_poke[0], reserve=converted_poke[1:], format=battle_format
    )

    # 2. Predict the team
    try:
        predicted_team = team_predictor.predict(
            revealed_team, date=date_played, rating=rating, gameid=gameid
        )
    except Exception as e:
        raise BackwardException(f"Error predicting team: {e}")
    if not revealed_team.is_consistent_with(predicted_team):
        raise InconsistentTeamPrediction(revealed_team, predicted_team)

    # 3. Filling missing information with the predicted team
    pokemon_to_add = [
        poke
        for poke in predicted_team.pokemon
        if poke.base_species not in existing_species
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
            if match.base_species == p.had_name:
                break
        else:
            raise BackwardException(f"Could not find match for {p.name}")
        p.fill_from_PokemonSet(match)

        if (
            p.had_item == BackwardMarkers.FORCE_UNKNOWN
            or p.had_ability == BackwardMarkers.FORCE_UNKNOWN
        ):
            raise BackwardException(
                f"Leaked BackwardMarkers.FORCE_UNKNOWN for {p.had_item} or {p.had_ability} with predicted match {match.name}"
            )

    return poke_list, revealed_team


class POVReplay:
    def __init__(
        self,
        replay: ParsedReplay,
        filled_replay: ParsedReplay,
        from_p1_pov: bool,
        revealed_team: TeamSet,
    ):
        if replay.gameid != filled_replay.gameid:
            raise ValueError("Using replays of different games to construct POVReplay")
        self.from_p1_pov = from_p1_pov
        self.replay = replay
        self.filled_replay = filled_replay
        self.revealed_team = revealed_team

        # copy replay metadata
        self.gameid = filled_replay.gameid
        self.time_played = filled_replay.time_played
        self.format = filled_replay.format
        self.replay_url = filled_replay.replay_url
        self.gen = filled_replay.gen
        self.rules = filled_replay.rules
        self.check_warnings = filled_replay.check_warnings
        # rating and winner from POV
        self.rating = filled_replay.ratings[0 if from_p1_pov else 1]
        self.winner = filled_replay.winner == (
            Winner.PLAYER_1 if from_p1_pov else Winner.PLAYER_2
        )

        self._povturnlist: list[Turn] = []
        self._actionlist: list[list[Optional[Action]]] = []
        self._fill_one_side(replay, filled_replay)
        self._resolve_transforms()
        self._resolve_zoroark()
        self._align_states_actions(replay)

    @property
    def povturnlist(self) -> list[Turn]:
        return self._povturnlist

    @property
    def actionlist(self) -> list[list[Optional[Action]]]:
        return self._actionlist

    def _flatten_turnlist_from_pov(self, start_from_turn: int = 0) -> list[Turn]:
        flat = []
        for turn in self.replay.turnlist[start_from_turn:]:
            for subturn in self._flatten_subturns_from_pov(turn):
                flat.append(subturn.turn)
            flat.append(turn)
        return flat

    def _flatten_subturns_from_pov(self, turn: Turn):
        for subturn in turn.subturns:
            if subturn.turn is not None and subturn.team == (
                1 if self.from_p1_pov else 2
            ):
                yield subturn

    def _resolve_transforms(self):
        replay = self.replay
        filled_replay = self.filled_replay
        if not replay.has_warning(WarningFlags.TRANSFORM):
            return

        # find the turn where transformations begin
        transforms = collections.deque()
        for i, filled_turn in enumerate(filled_replay.turnlist):
            active_pokemon = filled_turn.get_active_pokemon(self.from_p1_pov)
            for p in active_pokemon:
                if (
                    p is not None
                    and p.transformed_this_turn
                    and p.transformed_into is not None
                ):
                    transforms.append((i, p.unique_id, p.transformed_into.unique_id))
        while transforms:
            i, poke_id, tformed_id = transforms.popleft()
            filled_turn = filled_replay.turnlist[i]
            transformed_into = filled_turn.id2pokemon[tformed_id]
            opp_moves_on_transform = transformed_into.moves
            # skip to the end of the transformation, where we've found
            # as many moves as we'll ever find...
            last_moveset = {}
            for turn in replay.turnlist[i:]:
                player_pov = turn.id2pokemon[poke_id]
                if player_pov.transformed_into is None:
                    break
                last_moveset = player_pov.moves
            last_moveset = copy.deepcopy(last_moveset)
            # fill the last moveset with moves the transformed opponent supposedly
            # had on the transformation turn
            for opp_move_name, opp_move in opp_moves_on_transform.items():
                if opp_move_name not in last_moveset and len(last_moveset) < 4:
                    fixed_move = opp_move.from_transform()
                    last_moveset[opp_move_name] = fixed_move
            # now go through the whole transformation window inserting moves we'll use
            # later (or will never use at at all -- but the opponent had them)
            # TODO: v3-beta lets movesets go over 4... forcing a fix on the interface side.
            # need to revisit.
            transform_active = False
            for turn in self._flatten_turnlist_from_pov(start_from_turn=i):
                # `transform_active` needed in case the transformation actually happens
                # after a forced switch on the same turn.
                player_pov = turn.id2pokemon[poke_id]
                if player_pov.transformed_into is not None:
                    transform_active = True
                if transform_active and player_pov.transformed_into is None:
                    break  # done
                if transform_active:
                    for move in last_moveset.values():
                        player_pov.reveal_move(move)

    def _resolve_zoroark(self):
        replay = self.replay
        if not replay.has_warning(WarningFlags.ZOROARK):
            return

        def _broken_switch(action: Action, replacement: Replacement):
            return action and action.is_switch and action.target == replacement.replaced

        def _fix_turn(turn: Turn, replacement: Replacement):
            for t in self._flatten_subturns_from_pov(turn):
                action = t.action
                if _broken_switch(action, replacement):
                    action.target = replacement.replaced_with
            for move_action in turn.get_moves(self.from_p1_pov):
                if _broken_switch(move_action, replacement):
                    move_action.target = replacement.replaced_with
            for t in [s.turn for s in self._flatten_subturns_from_pov(turn)] + [turn]:
                active = t.get_active_pokemon(self.from_p1_pov)
                if replacement.replaced_with in active:
                    return True
                for p in t.get_active_pokemon(self.from_p1_pov):
                    if p is not None and p == replacement.replaced:
                        true_active_pokemon = t.get_pokemon_by_uid(
                            replacement.replaced_with.unique_id
                        )
                        p.moves = true_active_pokemon.moves
            return False

        for turn in replay.turnlist:
            for replacement in turn.get_replacements(self.from_p1_pov):
                fixed = False
                for tstep in range(*replacement.turn_range):
                    turn = replay.turnlist[tstep]
                    fixed = _fix_turn(turn, replacement)
                    if fixed:
                        break

    def _fill_one_side(self, replay, filled_replay):
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

    def _align_states_actions(self, replay: ParsedReplay):
        self._povturnlist = []
        self._actionlist = []
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
                    self._povturnlist.append(subturn.turn)
                    self._actionlist.append(action)

            self._povturnlist.append(
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
            self._actionlist.append(actionlist)

        # add final state
        self._povturnlist.append(turn_t1)
        self._actionlist.append([None, None])


def add_filled_final_turn(
    replay: ParsedReplay, team_predictor: TeamPredictor
) -> tuple[ParsedReplay, tuple[TeamSet, TeamSet]]:
    # add an extra turn to a replay with all missing information guessed
    # by sampling from the TeamBuilder. this extra turn can then be moved
    # backwards through the replay and discareded.
    filled_turn = replay[-1].create_next_turn()
    filled_turn.on_end_of_turn()
    date_played = replay.time_played.date()
    filled_turn.pokemon_1, revealed_team_1 = fill_missing_team_info(
        replay.format,
        date_played=date_played,
        poke_list=replay[-1].pokemon_1,
        team_predictor=team_predictor,
        rating=replay.ratings[0],
        gameid=replay.gameid,
    )
    filled_turn.pokemon_2, revealed_team_2 = fill_missing_team_info(
        replay.format,
        date_played=date_played,
        poke_list=replay[-1].pokemon_2,
        team_predictor=team_predictor,
        rating=replay.ratings[1],
        gameid=replay.gameid,
    )
    replay.turnlist.append(filled_turn)
    return replay, (revealed_team_1, revealed_team_2)


def backward_fill(
    replay: ParsedReplay, team_predictor: TeamPredictor
) -> tuple[POVReplay, POVReplay]:
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
