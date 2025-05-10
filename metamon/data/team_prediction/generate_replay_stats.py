"""
A questionably efficient script that loads a set of team files and finds all unique Pokémon movesets and team rosters (6 Pokemon names)
implied by those files. It then calculates all the possible fully revealed versions of those movesets and rosters, and counts the number
of team files that *may have* used them.
"""

import os
import random
from collections import defaultdict
from typing import Set

import tqdm
from torch.utils.data import DataLoader

from metamon.data.team_prediction.dataset import TeamDataset
from metamon.data.team_prediction.team import TeamSet, PokemonSet, Roster


def load_replay_teams(
    revealed_team_dir: str, format: str, max_teams: int, num_workers: int = 10
):
    teams = []
    pokemon_sets = defaultdict(list)
    team_rosters = []

    dataset = TeamDataset(revealed_team_dir, format, max_teams=max_teams)
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=lambda x: x,
    )

    for batch in tqdm.tqdm(dataloader, desc="Loading Team Files", colour="blue"):
        for result in batch:
            if result is None:
                continue
            team, pokemon_dict, team_roster = result
            teams.append(team)
            for name, pokemon in pokemon_dict.items():
                pokemon_sets[name].append(pokemon)
            team_rosters.append(team_roster)

    if not teams:
        raise ValueError(f"No valid team files found in {revealed_team_dir}")

    print(f"Loaded {len(teams)} teams with {len(pokemon_sets)} unique Pokémon")
    return teams, pokemon_sets, team_rosters


def _find_consistency_leaves(sets: list[PokemonSet | TeamSet | Set]):
    unique = list(set(sets))
    n = len(unique)
    lengths = [len(s) for s in unique]
    is_leaf = [True] * n

    for i in tqdm.trange(n, desc="Finding consistency leaves", colour="green"):
        if not is_leaf[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            # If i is consistent with j, then i is not a leaf
            # use lengths as easy check to skip comparisons
            if lengths[i] < lengths[j]:
                if unique[i] < unique[j]:
                    is_leaf[i] = False
                    break

    return [unique[i] for i in range(n) if is_leaf[i]]


def find_pokemon_consistency_weights(sets: list[PokemonSet | TeamSet | Set]):
    leaves = _find_consistency_leaves(sets)
    weights = [0 for _ in leaves]
    for s in tqdm.tqdm(sets, desc="Calculating consistency weights", colour="red"):
        for idx, l in enumerate(leaves):
            if s <= l:
                weights[idx] += 1
    return list(zip(leaves, weights))


def _process_roster_batch(args):
    batch_sets, leaves = args
    result = [0] * len(leaves)

    for s in batch_sets:
        for idx, leaf in enumerate(leaves):
            if s <= leaf:
                result[idx] += 1

    return result


def find_team_consistency_weights(sets: list[Roster], num_workers: int = 10):
    leaves = _find_consistency_leaves(sets)
    print(f"Found {len(leaves)} consistency leaves")

    batch_size = max(100, len(sets) // (num_workers * 2))
    batches = [sets[i : i + batch_size] for i in range(0, len(sets), batch_size)]
    print(f"Processing {len(batches)} batches in parallel")

    from multiprocessing import Pool

    with Pool(processes=num_workers) as pool:
        all_results = list(
            tqdm.tqdm(
                pool.imap(
                    _process_roster_batch, [(batch, leaves) for batch in batches]
                ),
                total=len(batches),
                desc="Calculating consistency weights",
                colour="red",
            )
        )

    # sum up the weights from all batches
    final_weights = [0] * len(leaves)
    for batch_result in all_results:
        for i, count in enumerate(batch_result):
            final_weights[i] += count

    return list(zip(leaves, final_weights))


if __name__ == "__main__":
    import argparse
    import os
    from multiprocessing import Pool
    from collections import defaultdict
    import json

    parser = argparse.ArgumentParser(description="Generate replay stats")
    parser.add_argument(
        "--revealed_team_dir",
        type=str,
        default=None,
        help="Directory containing revealed team files. Defaults to METAMON_TEAMFILE_PATH env var",
    )
    parser.add_argument(
        "--format", type=str, default="gen9ou", help="Format to analyze"
    )
    parser.add_argument(
        "--max_teams", type=int, default=1000, help="Maximum number of teams to load"
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="Number of worker processes"
    )
    args = parser.parse_args()

    revealed_team_dir = args.revealed_team_dir
    if revealed_team_dir is None:
        revealed_team_dir = os.environ.get("METAMON_TEAMFILE_PATH", None)
        if revealed_team_dir is None:
            raise ValueError("METAMON_TEAMFILE_PATH environment variable not set")

    teams, pokemon_sets, team_rosters = load_replay_teams(
        revealed_team_dir,
        format=args.format,
        max_teams=args.max_teams,
        num_workers=args.workers,
    )

    def process_pokemon_sets(args):
        pokemon_name, all_sets = args
        weights = find_pokemon_consistency_weights(all_sets)
        sorted_sets = sorted(weights, key=lambda x: x[1], reverse=True)
        return pokemon_name, [
            (set_i.to_dict(), weight_i) for set_i, weight_i in sorted_sets
        ]

    pokemon_sets_items = list(pokemon_sets.items())
    with Pool(processes=args.workers if args.workers > 0 else None) as pool:
        results = pool.map(process_pokemon_sets, pokemon_sets_items)

    processed_sets = dict(results)
    output_dir = os.path.join(os.path.dirname(__file__), "replay_stats")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{args.format}_pokemon.json"), "w") as f:
        json.dump(processed_sets, f, indent=2)

    weights = find_team_consistency_weights(team_rosters, num_workers=args.workers)
    results = {
        "rosters": [
            {
                "team": roster.to_dict(),
                "weight": weight,
            }
            for roster, weight in sorted(weights, key=lambda x: x[1], reverse=True)
        ],
    }
    with open(os.path.join(output_dir, f"{args.format}_team_rosters.json"), "w") as f:
        json.dump(results, f, indent=2)
