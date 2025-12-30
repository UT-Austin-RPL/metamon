"""
(More or less) salvages self-play trajectories collected during the PokéAgent Challenge
(using what's now called the "pokeagent" backend) by filling in missing tera types.

If gen9ou trajectory .json's have "notype" for tera type, run your gen9ou dataset through this script
and then finetune an existing model (e.g. Abra) to enable strong play on the "metamon" battle backend.
"""

import os
import json
import argparse
from datetime import date
from typing import List, Callable
from glob import iglob
import numpy as np
from tqdm import tqdm
import lz4.frame
from multiprocessing import Pool, cpu_count

from metamon.interface import UniversalState
from metamon.backend.team_prediction.usage_stats import get_usage_stats
from metamon.backend.replay_parser.str_parsing import clean_name


USAGE_STATS = get_usage_stats(
    "gen9ou", start_date=date(2022, 1, 1), end_date=date(2025, 5, 31)
)


def process_trajectory_file(args_tuple):
    input_path, output_path = args_tuple

    try:
        with lz4.frame.open(input_path, "rb") as f:
            data = json.loads(f.read().decode("utf-8"))
        states = [UniversalState.from_dict(s) for s in data["states"]]

        if states[0].format == "gen9ou":
            # step 1: find any tera types that are correctly recorded
            our_pokemon = [states[0].player_active_pokemon.base_species]
            for p in states[0].available_switches:
                our_pokemon.append(p.base_species)
            known_tera_types = {p: None for p in our_pokemon}
            for state in states:
                pokemon = [state.player_active_pokemon] + state.available_switches
                for p in pokemon:
                    if p.tera_type != "notype":
                        if known_tera_types[p.base_species] is None:
                            known_tera_types[p.base_species] = p.tera_type

            # step 2: for any pokemon with an unknown tera type, sample a tera type based on usage stats
            for p, tera_type in known_tera_types.items():
                if tera_type is not None:
                    continue
                tera_type_stats = USAGE_STATS[p].get("tera_types", {}).copy()
                if "Nothing" in tera_type_stats:
                    del tera_type_stats["Nothing"]
                if "Other" in tera_type_stats:
                    del tera_type_stats["Other"]
                if not tera_type_stats:
                    return (input_path, False, "Missing tera_types in usage stats")
                total = sum(tera_type_stats.values())
                normalized_probs = {k: v / total for k, v in tera_type_stats.items()}
                types, probs = zip(*normalized_probs.items())
                sampled_tera_type = clean_name(np.random.choice(types, p=probs).item())
                known_tera_types[p] = sampled_tera_type

            # step 3: relabel with the replacement tera types
            for state in states:
                pokemon = [state.player_active_pokemon] + state.available_switches
                for p in pokemon:
                    if p.base_species in known_tera_types and p.tera_type == "notype":
                        p.tera_type = known_tera_types[p.base_species]
                    if p.tera_type == "notype":
                        return (input_path, False, "Found notype after patching")

        output_json = {
            "states": [s.to_dict() for s in states],
            "actions": data["actions"],  # Actions remain unchanged
        }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        temp_path = output_path + ".tmp"
        with lz4.frame.open(temp_path, "wb") as f:
            f.write(json.dumps(output_json).encode("utf-8"))
        os.rename(temp_path, output_path)

        return (input_path, True, None)

    except Exception as e:
        return (input_path, False, str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Patch trajectories by transforming UniversalStates"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory containing input trajectory files (.json.lz4)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save patched trajectory files",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=".json.lz4",
        help="File pattern to match (default: .json.lz4)",
    )
    parser.add_argument(
        "--filelist",
        type=str,
        default=None,
        help="Path to text file with absolute paths to trajectory files (one per line). If provided, --input_dir is ignored.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of parallel workers (default: 16)",
    )

    args = parser.parse_args()

    if args.filelist is None and args.input_dir is None:
        raise ValueError("Must provide either --filelist or --input_dir")

    os.makedirs(args.output_dir, exist_ok=True)

    trajectory_files = []

    if args.filelist:
        print(f"Loading file list from: {args.filelist}", flush=True)
        with open(args.filelist, "r") as f:
            for line in tqdm(f, desc="Reading file list"):
                input_path = line.strip()
                if not input_path:
                    continue
                filename = os.path.basename(input_path)
                output_path = os.path.join(args.output_dir, filename)
                trajectory_files.append((input_path, output_path))
        print(f"\nLoaded {len(trajectory_files)} files from filelist", flush=True)
    else:
        if not os.path.exists(args.input_dir):
            raise ValueError(f"Input directory does not exist: {args.input_dir}")

        pattern_glob = os.path.join(args.input_dir, "**", f"*{args.pattern}")
        print(f"Searching for files matching: {pattern_glob}")

        input_dir_prefix_len = len(args.input_dir.rstrip(os.sep)) + 1

        for input_path in tqdm(
            iglob(pattern_glob, recursive=True), desc="Finding trajectory files"
        ):
            rel_path = input_path[input_dir_prefix_len:]
            output_path = os.path.join(args.output_dir, rel_path)
            trajectory_files.append((input_path, output_path))

        print(f"\nFound {len(trajectory_files)} trajectory files", flush=True)

    num_workers = args.num_workers
    print(f"Using {num_workers} parallel workers", flush=True)
    print(
        "Initializing worker pool (each worker needs to load usage stats, this may take 30-60 seconds)...",
        flush=True,
    )

    errors = []
    with Pool(processes=num_workers) as pool:
        print("Worker pool initialized, starting processing...")
        chunksize = max(1, len(trajectory_files) // (num_workers * 100))
        print(f"Using chunksize of {chunksize}")
        results = list(
            tqdm(
                pool.imap_unordered(
                    process_trajectory_file, trajectory_files, chunksize=chunksize
                ),
                total=len(trajectory_files),
                desc="Processing trajectories",
            )
        )
        for input_path, success, error_msg in results:
            if not success:
                errors.append((input_path, error_msg))

    if errors:
        print(f"\nEncountered {len(errors)} errors:")
        for input_path, error_msg in errors[:10]:
            print(f"  {input_path}: {error_msg}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        print(
            f"Successfully processed {len(trajectory_files) - len(errors)}/{len(trajectory_files)} trajectories"
        )
    else:
        print(f"Successfully processed {len(trajectory_files)} trajectories")

    print(f"Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
