import os
import json
import argparse
from collections import defaultdict
from tqdm import tqdm

from metamon.backend.team_prediction.usage_stats.format_rules import (
    Tier,
)
from metamon.backend.team_prediction.usage_stats.stat_reader import (
    SmogonStat,
    merge_movesets,
)

VALID_TIERS = [Tier.UBERS, Tier.OU, Tier.UU, Tier.RU, Tier.NU, Tier.PU]


def main(args):
    total_iterations = 9 * 12 * 12 * len(VALID_TIERS)
    with tqdm(total=total_iterations, desc="Processing movesets") as pbar:
        for gen in range(1, 10):
            for year in range(2014, 2027):
                for month in range(1, 13):
                    date = f"{year}-{month:02d}"
                    stat_dir = os.path.join(args.smogon_stat_dir)
                    valid_movesets_by_rank = defaultdict(list)
                    for format in VALID_TIERS:
                        format_name = f"gen{gen}{format.name.lower()}"

                        ranks = SmogonStat.available_ranks(
                            format_name,
                            raw_stats_dir=stat_dir,
                            date=date,
                        )

                        for rank in ranks:
                            stat = SmogonStat(
                                format_name,
                                raw_stats_dir=stat_dir,
                                date=date,
                                rank=rank,
                                verbose=False,
                            )
                            if not stat.movesets:
                                continue

                            path = os.path.join(
                                args.save_dir,
                                "movesets_data",
                                f"gen{gen}",
                                f"{format.name.lower()}",
                                str(rank),
                                f"{date}.json",
                            )
                            os.makedirs(os.path.dirname(path), exist_ok=True)
                            with open(path, "w") as f:
                                json.dump(stat.movesets, f)

                            check_cheatsheet = {
                                mon: stat.movesets[mon]["checks"]
                                for mon in stat.movesets.keys()
                            }
                            path = os.path.join(
                                args.save_dir,
                                "checks_data",
                                f"gen{gen}",
                                f"{format.name.lower()}",
                                str(rank),
                                f"{date}.json",
                            )
                            os.makedirs(os.path.dirname(path), exist_ok=True)
                            with open(path, "w") as f:
                                json.dump(check_cheatsheet, f)

                            valid_movesets_by_rank[rank].append(stat.movesets)
                        pbar.update(1)

                    for rank, tier_movesets in valid_movesets_by_rank.items():
                        inclusive_movesets = merge_movesets(tier_movesets)
                        path = os.path.join(
                            args.save_dir,
                            "movesets_data",
                            f"gen{gen}",
                            "all_tiers",
                            str(rank),
                            f"{date}.json",
                        )
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        with open(path, "w") as f:
                            json.dump(inclusive_movesets, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create usage jsons")
    parser.add_argument(
        "--smogon_stat_dir",
        type=str,
        help="Path to the scraped smogon stat directory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Path to the save directory",
    )
    args = parser.parse_args()
    main(args)
