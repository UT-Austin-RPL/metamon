import copy
import os
import random
import json
from abc import ABC, abstractmethod
from functools import lru_cache
from collections import deque
from typing import Set, List, Tuple
import numpy as np

from metamon.data.team_builder.team_builder import TeamBuilder, PokemonStatsLookupError
from metamon.data.team_builder.stat_reader import PreloadedSmogonStat
from metamon.data.team_prediction.team import TeamSet, PokemonSet, Roster


class TeamPredictor(ABC):
    def __init__(self):
        pass

    def predict(self, team: TeamSet) -> TeamSet:
        copy_team = copy.deepcopy(team)
        return self.fill_team(copy_team)

    @abstractmethod
    def fill_team(self, team: TeamSet):
        raise NotImplementedError


@lru_cache(maxsize=16)
def get_legacy_teambuilder(format: str):
    return TeamBuilder(format, verbose=False, remove_banned=False, inclusive=True)


class NaiveUsagePredictor(TeamPredictor):
    """
    The original paper strategy. We use the names of the pokemon we have already know
    to guess the full team, then fill in the movesets using usage stats. Every decision
    is made independently by sampling from unconditioned usage distributions. This has
    some downsides that are explained / improved by the ReplayPredictor. Every detail that
    was missing in the orignial team is then filled by grabbing the same attribute from
    the generated team.

    This team_prediction module didn't exist in the original version. Because i'm lazy,
    this works by converting to the old format, running the legacy predictor, and then
    converting back.
    """

    def fill_team(self, team: TeamSet):
        team_builder = get_legacy_teambuilder(team.format)
        gen = int(team.format.split("gen")[1][0])
        pokemon = [team.lead] + team.reserve
        # use legacy team builder to generate a team of 6 Pokémon based on the ones we already
        # have and their common teammates. Teammates and all moves/items/abilities are sampled
        # from a premade set of Showdown usage statistics.
        existing_names = [p.name for p in pokemon if p.name != PokemonSet.MISSING_NAME]
        try:
            sample_team = team_builder.generate_new_team(existing_names)
        except PokemonStatsLookupError as e:
            raise e

        # convert from the output of the old team builder to the new PokemonSet format
        cleaned_dict = {}
        for poke in sample_team:
            if poke["name"] == PokemonSet.MISSING_NAME:
                raise ValueError("Missing name in sample team")
            spread = poke["spread"]
            nature, evs = spread.split(":")
            nature = nature.strip()
            evs = [int(x) for x in evs.split("/")]
            ivs = [int(x) for x in poke["IVs"].split("/")]
            item = poke["item"].strip()
            ability = poke["ability"].strip()
            if ability == "No Ability":
                ability = PokemonSet.NO_ABILITY
            if not item:
                item = PokemonSet.NO_ITEM
            cleaned_dict[poke["name"]] = {
                "moves": poke["moves"],
                "gen": gen,
                "nature": nature,
                "evs": evs,
                "ivs": ivs,
                "ability": ability,
                "item": item,
            }

        sample_team = []
        for key, val in cleaned_dict.items():
            as_pokemon_dict = val | {"name": key}
            try:
                sample_team.append(PokemonSet.from_dict(as_pokemon_dict))
            except Exception as e:
                raise e

        # Build a mapping from name to PokemonSet for fast lookup
        sample_team_map = {p.name: p for p in sample_team}
        # Prepare a queue of new Pokemon to fill missing slots
        new_pokemon = deque([p for p in sample_team if p.name not in existing_names])
        merged_team = []
        for p in team.pokemon:
            if p.name == PokemonSet.MISSING_NAME:
                new_choice = new_pokemon.popleft()
                merged_team.append(copy.deepcopy(new_choice))
            else:
                new_p = sample_team_map.get(p.name)
                filled_p = copy.deepcopy(p)
                filled_p.fill_from_PokemonSet(new_p)
                merged_team.append(filled_p)
        final_team = TeamSet(
            lead=merged_team[0], reserve=merged_team[1:], format=team.format
        )
        return final_team


@lru_cache(maxsize=4)
def load_replay_stats_by_format(format: str):
    """
    This loads large json files that are created by the `generate_replay_stats` script.
    """
    pokemon_set_path = os.path.join(
        os.path.dirname(__file__),
        "replay_stats",
        f"{format}_pokemon.json",
    )
    team_roster_path = os.path.join(
        os.path.dirname(__file__),
        "replay_stats",
        f"{format}_team_rosters.json",
    )
    if not os.path.exists(pokemon_set_path) or not os.path.exists(team_roster_path):
        raise FileNotFoundError(
            f"Stat files not found for format {format}. To use ReplayPredictor, you must first run `python -m metamon.data.team_prediction.generate_replay_stats`"
        )
    with open(pokemon_set_path, "r") as f:
        pokemon_sets = json.load(f)
    with open(team_roster_path, "r") as f:
        team_rosters = json.load(f)["rosters"]

    rosters = []
    for rw in team_rosters:
        r = Roster.from_dict(rw["team"])
        w = rw["weight"]
        rosters.append((r, w))

    sets = {}
    for name, all_sets in pokemon_sets.items():
        sets[name] = [
            (PokemonSet.from_dict(p | {"name": name}), w) for p, w in all_sets
        ]

    return sets, rosters


class ReplayPredictor(NaiveUsagePredictor):
    """
    NaiveUsagePredictor *independently* samples missing details from PS usage stats.
    The flaw with this is that it creates unlikely combinations of pokemon and movesets.
    For example, if there are 2 common movesets for a particular Pokemon, it will sample
    a very unlikely combination of moves from both sets.

    ReplayPredictor instead matches the current revealed team to a set of candidates
    discovered from every replay in the dataset. If we think of revealed teams as nodes
    on a graph, where an edge A -> B exists if team B could have been made from team A
    by revealing more information, then these `candidate` teams are the leaf nodes. By sampling
    from the most commonly implied candidates, we are restricting our predictions to real +
    reasonably popular choices. Some hacky logic based on usage stats is still needed to filter
    candidates that are frequently valid but rarely actually used (see `score_pokemon`).

    Currently only supports gen1ou, gen2ou, gen3ou, and gen4ou. Falls back to NaiveUsagePredictor
    otherwise.
    """

    def __init__(
        self,
        top_k_consistent_teams: int = 20,
        top_k_consistent_movesets: int = 15,
        top_k_scored_teams: int = 10,
        top_k_scored_movesets: int = 3,
    ):
        self.stat_format = None
        self.top_k_consistent_teams = top_k_consistent_teams
        self.top_k_consistent_movesets = top_k_consistent_movesets
        self.top_k_scored_teams = top_k_scored_teams
        self.top_k_scored_movesets = top_k_scored_movesets

    def _load_data(self, format: str):
        self.smogon_stat = PreloadedSmogonStat(format, verbose=False, inclusive=True)
        self.pokemon_sets, self.team_rosters = load_replay_stats_by_format(format)

    def _sample_from_top_k(self, choices, probs: List[float], k: int) -> float:
        """
        Sample from the top k choices, weighted by their probabilities.
        """
        probs = np.array(probs)
        k = min(k, len(probs))
        # grab the k highest probs
        topk_idx = np.argpartition(probs, -k)[-k:]
        new_probs = probs[topk_idx]
        new_probs /= new_probs.sum()
        new_choices = [choices[i] for i in topk_idx]
        return random.choices(new_choices, weights=new_probs, k=1)[0]

    def score_roster(self, current_roster: Roster, candidate_roster: Roster) -> float:
        """
        Score a candidate roster based on how likely it is to have been made from the current roster.
        """
        score = 1.0
        eps = 1e-6

        # we only want to judge the likelihood of new info
        new_pokemon = current_roster.additional_details(candidate_roster)
        if len(new_pokemon) < 6 - len(current_roster):
            # heavily downweight candidates that don't fill our team
            score *= eps

        # grab p(teammate | pokemon) for all pokemon in our team where that info exists
        teammates = []
        for p in current_roster.known_pokemon:
            try:
                teammates.append(self.smogon_stat[p]["teammates"])
            except KeyError:
                continue

        for p in new_pokemon:
            # pick pokemon that are very likely to appear alongside
            # at least one of our current pokemon.
            max_teammate_weight = eps
            for teammate in teammates:
                if p in teammate:
                    max_teammate_weight = max(max_teammate_weight, teammate[p])
            score *= max_teammate_weight
        return score

    def sample_roster_from_candidates(
        self, current_roster: Roster, candidates: List[Tuple[Roster, float]]
    ):
        weights = [self.score_roster(current_roster, r) for r, _ in candidates]
        choices = [r for r, _ in candidates]
        roster = self._sample_from_top_k(choices, weights, k=self.top_k_scored_teams)
        return copy.deepcopy(roster)

    def score_pokemon(
        self, current_pokemon: PokemonSet, candidate_pokemon: PokemonSet
    ) -> float:
        """
        Scores a diff between a current Pokemon and a candidate predicted Pokemon.

        The basic problem we need to address is the difference between a moveset that is frequently *possible* vs. a moveset that is frequently *used*.
        This occurs when Pokemon almost always use 1-2 key moves but rarely reveal the rest such that the common moves carry very little info.

        For example, in gen1ou, it is common for Tauros to reveal {Earthquake, Body Slam, Hyper Beam} during a battle.
        Our candidates computed by the replay stats will find us all the movesets that have these three moves.
        There are many choices for the 4th move that have appeared at some point. The "weights" assigned to each candidate
        from the replay stats will give options that look something like this:
            Blizzard, weight=6812
            Fire Blast, weight=5574
            Rest, weight=5507
            Stomp, weight=5507
            ...
            Thunder, weight=4709
            Swords Dance, weight=3333
        Stomp/Thunder/Swords Dance are dramatically overrepresented because many replays are consistent with these movesets,
        but in reality these would rarely be the 4th move. A team that revealed {Fire Blast, Earthquake, Hyper Beam} would not have
        this problem. The 4th move is clearly Body Slam.

        This function adjusts candidates based on the usage stats of their suggested additions, e.g. P(Fire Blast|Tauros) >> P(Thunder|Tauros).
        """
        name = current_pokemon.name
        eps = 1e-6

        # we only want to judge the likelihood of new info
        diff = current_pokemon.additional_details(candidate_pokemon)
        if diff is None:
            return eps

        try:
            smogon_stat = self.smogon_stat[name]
        except KeyError:
            return eps

        score = 1.0

        new_moves = diff.get("moves", [])
        if len(new_moves) < (
            len(current_pokemon.moves) - current_pokemon.revealed_moves
        ):
            # if the candidate does not reveal all the missing moves, downweight
            score *= eps

        if (
            current_pokemon.ability == PokemonSet.MISSING_ABILITY
            and "ability" not in diff
        ):
            # if the candidate does not reveal the missing ability, downweight
            score *= eps

        if current_pokemon.item == PokemonSet.MISSING_ITEM and "item" not in diff:
            # if the candidate does not reveal the missing item, downweight
            score *= eps

        if "moves" in diff:
            move_stats = smogon_stat["moves"]
            for new_move in diff["moves"]:
                if new_move not in move_stats:
                    score *= eps
                else:
                    score *= move_stats[new_move]

        if "ability" in diff:
            if diff["ability"] not in smogon_stat["abilities"]:
                score *= eps
            else:
                score *= smogon_stat["abilities"][diff["ability"]]

        if "item" in diff:
            if diff["item"] not in smogon_stat["items"]:
                score *= eps
            else:
                score *= smogon_stat["items"][diff["item"]]

        return score

    def sample_pokemon_from_candidates(
        self,
        current_pokemon: PokemonSet,
        candidates: List[Tuple[PokemonSet, float]],
    ):
        weights = [self.score_pokemon(current_pokemon, p) for p, _ in candidates]
        choices = [p for p, _ in candidates]
        pokemon = self._sample_from_top_k(
            choices, weights, k=self.top_k_scored_movesets
        )
        return copy.deepcopy(pokemon)

    def emergency_fill_team(self, team: TeamSet):
        found_pokemon = set([p.name for p in team.known_pokemon])
        extra_pokemon = set()
        # we use this to deduplicate form changes like "Rotom-Wash" and "Rotom-Heat"
        _form_agnostic = lambda pnames: set(map(lambda x: x.split("-")[0], pnames))

        tries = 0
        while len(extra_pokemon | found_pokemon) < 6 and tries < 100:
            tries += 1
            for pokemon in team.known_pokemon:
                try:
                    teammates = self.smogon_stat[pokemon.name]["teammates"]
                except KeyError:
                    continue
                # get all the teammates in the PS usage stats
                already_found = _form_agnostic(found_pokemon | extra_pokemon)
                emergency_options = {
                    k: v
                    for k, v in teammates.items()
                    if k.split("-")[0] not in already_found
                }
                if emergency_options:
                    # sample a new pokemon according to usage weights
                    name_choices = list(emergency_options.keys())
                    ps_usage_weights = list(emergency_options.values())
                    choice = random.choices(
                        name_choices, weights=ps_usage_weights, k=1
                    )[0]
                    extra_pokemon.add(choice)
                    break

        assert len(extra_pokemon) == 6 - len(found_pokemon)

        for extra in extra_pokemon:
            for pokemon in team.pokemon:
                if pokemon.name == PokemonSet.MISSING_NAME:
                    pokemon.name = extra
                    break

    def fill_team(self, team: TeamSet) -> TeamSet:
        if team.format not in {"gen1ou", "gen2ou", "gen3ou", "gen4ou"}:
            # we only trust our stats for the big OU formats for now
            return super().fill_team(team)

        if self.stat_format != team.format:
            # load the stats on a format change
            self._load_data(team.format)

        # first we fill our team of 6 pokemon names
        lead_name = team.lead.name
        reserve_names = [p.name for p in team.reserve]
        roster = Roster(lead=lead_name, reserve=reserve_names)
        # from the set of "most revealed" rosters implied by every replay in the dataset,
        # grab the ones that do not contradict the roster we have so far
        candidate_rosters = [
            (r, w) for r, w in self.team_rosters if roster.is_consistent_with(r)
        ]
        if candidate_rosters:
            # the "most revealed" rosters have weights associated with the number of replays
            # that *may* use them. It's not quite right to sample from these weights, but
            # taking the top k is a decent start.
            top_k_candidate_rosters = sorted(
                candidate_rosters, key=lambda x: x[1], reverse=True
            )[: self.top_k_consistent_teams]
            roster = self.sample_roster_from_candidates(roster, top_k_candidate_rosters)
            assert roster.lead == lead_name
            # this fills the $missing_name$ slots on the team, but movesets will be blank
            team.fill_from_Roster(roster)
            # due to the definition of a "most revealed" roster (which is basically a leaf node
            # on a directed graph of all the teams ever revealed by a replay with an edge
            # from team A --> team B if team B could be a more revealed version of team A)
            # there may still be missing Pokemon on the roster.

        # if we still need Pokemon, we prefer to naive guess them here instead of waiting
        # for NaiveUsagePredictor, because our moveset prediction below is much improved.
        if len(team.known_pokemon) < 6:
            self.emergency_fill_team(team)

        # now we fill in the movesets following similar logic
        for pokemon in team.pokemon:
            candidate_sets = [
                (p, w)
                for p, w in self.pokemon_sets[pokemon.name]
                if pokemon.is_consistent_with(p)
            ]
            if candidate_sets:
                top_k_candidate_sets = sorted(
                    candidate_sets, key=lambda x: x[1], reverse=True
                )[: self.top_k_consistent_movesets]
                new_pokemon = self.sample_pokemon_from_candidates(
                    pokemon, top_k_candidate_sets
                )
                pokemon.fill_from_PokemonSet(new_pokemon)

        # fall back to old method for any remaining info
        return super().fill_team(team)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict team from partial information"
    )
    parser.add_argument(
        "format", type=str, help="Showdown battle format (e.g., gen1ou)"
    )
    args = parser.parse_args()

    team_files = os.listdir(
        os.path.join(os.environ["METAMON_TEAMFILE_PATH"], f"{args.format}_teams")
    )
    team_file = os.path.join(
        os.environ["METAMON_TEAMFILE_PATH"],
        f"{args.format}_teams",
        random.choice(team_files),
    )
    print(f"Using team file: {team_file}")

    naive_predictor = NaiveUsagePredictor()
    og_team = copy.deepcopy(TeamSet.from_showdown_file(team_file, args.format))
    naive_team = naive_predictor.predict(og_team)
    consistent = og_team.is_consistent_with(naive_team)
    print(f"Consistent: {consistent}")
    improved_predictor = ReplayPredictor()
    improved_team = improved_predictor.predict(og_team)
    consistent = og_team.is_consistent_with(improved_team)
    print(f"Consistent: {consistent}")
    # Print teams side by side
    og_lines = og_team.to_str().split("\n")
    naive_lines = naive_team.to_str().split("\n")
    improved_lines = improved_team.to_str().split("\n")

    # Pad lines to equal length
    max_len = max(len(og_lines), len(naive_lines), len(improved_lines))
    og_lines += [""] * (max_len - len(og_lines))
    naive_lines += [""] * (max_len - len(naive_lines))
    improved_lines += [""] * (max_len - len(improved_lines))

    # Print header
    print(f"{'Original Team':<40}{'Naive Prediction':<40}{'Improved Prediction':<40}")
    print("-" * 120)

    # Print lines side by side
    for og, naive, improved in zip(og_lines, naive_lines, improved_lines):
        print(f"{og:<40}{naive:<40}{improved:<40}")
