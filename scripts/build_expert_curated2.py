#!/usr/bin/env python3
"""Build expert_curated2 from the current expert_curated seed + ranked pool fills.

Seed: all teams currently in expert_curated/gen1ou (post Chansey/Tauros lead purge).
Fill to TARGET size with admissible Gengar- or Alakazam-lead teams from the
smallg1online_v2 collector only logs curated team paths; full expert
per-team rankings come from mini_online_g1_expert_v2_results (~1.37M battles).

Admissibility (same as original expert_curated build):
  - roster must contain Tauros
  - may drop at most 1 of {Chansey, Tauros, Snorlax}
  - Snorlax moveset not in bottom half (rank >= SNORLAX_BAD_RANK)
  - Tauros moveset win% >= most-popular Tauros set

Banned leads for additions: Chansey, Exeggutor, Snorlax, and anything not
Gengar/Alakazam for the fill slots.
"""

from __future__ import annotations

import csv
import glob
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from metamon.backend.team_prediction.team import PokemonSet, TeamSet as PredictionTeamSet

TARGET = 50
SEARCH_DEPTH = 200
SNORLAX_BAD_RANK = 11
BIG3 = {"Chansey", "Tauros", "Snorlax"}
FILL_LEADS = {"Gengar", "Alakazam"}
BANNED_LEADS = {"Chansey", "Exeggutor", "Snorlax"}

TEAMS_ROOT = Path(os.path.expanduser("~/local_metamon_cache/teams"))
SRC = TEAMS_ROOT / "expert" / "gen1ou"
SEED_DIR = TEAMS_ROOT / "expert_curated" / "gen1ou"
DST = TEAMS_ROOT / "expert_curated2" / "gen1ou"
RESULTS = Path("/home/jake/metamon_local/mini_online_g1_expert_v2_results")
NFS_DST = Path("/mnt/nfs_client/jake/metamon_cache/teams/expert_curated2/gen1ou")
SKIP = {PokemonSet.NO_MOVE, PokemonSet.MISSING_MOVE}


def parse_lead_raw(path: Path) -> str:
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            stripped = line.strip()
            if stripped.startswith(
                (
                    "-",
                    "EVs:",
                    "IVs:",
                    "Level:",
                    "Ability:",
                    "Shiny:",
                    "Happiness:",
                    "Nature:",
                    "Tera Type:",
                )
            ):
                continue
            if line[:1].isspace():
                continue
            name = stripped.split(" @ ")[0].strip()
            m = re.match(r"^.+\(([^)]+)\)$", name)
            return m.group(1).strip() if m else name
    raise ValueError(f"no lead in {path}")


def parse_team(fn: str) -> tuple[list[str], dict[str, frozenset]]:
    t = PredictionTeamSet.from_showdown_file(str(SRC / fn), "gen1ou")
    species = [p.name for p in t.pokemon]
    moves = {
        p.name: frozenset(m for m in p.moves if m not in SKIP) for p in t.pokemon
    }
    return species, moves


def load_battle_stats() -> tuple[dict[str, int], dict[str, int]]:
    na, wa = defaultdict(int), defaultdict(int)
    for fp in glob.glob(str(RESULTS / "collect_results_seed*.csv")):
        with open(fp, newline="") as f:
            for row in csv.reader(f):
                if len(row) < 4:
                    continue
                # V2 logs may point at curated paths; normalize to expert basename
                tk = os.path.basename(row[1])
                if not tk.endswith(".gen1ou_team"):
                    continue
                na[tk] += 1
                wa[tk] += row[3] == "WIN"
    return na, wa


def moveset_stats(
    species: str,
    files: list[str],
    info: dict,
    na: dict[str, int],
    wa: dict[str, int],
) -> tuple[dict, dict, float]:
    by = defaultdict(lambda: {"b": 0, "w": 0, "teams": 0})
    for fn in files:
        moves = info[fn][1]
        if species in moves and fn in na:
            by[moves[species]]["b"] += na[fn]
            by[moves[species]]["w"] += wa[fn]
            by[moves[species]]["teams"] += 1
    ordered = sorted(by.items(), key=lambda x: x[1]["w"] / x[1]["b"], reverse=True)
    rank = {mv: i + 1 for i, (mv, _) in enumerate(ordered)}
    wr = {mv: d["w"] / d["b"] for mv, d in by.items()}
    popular = max(by.items(), key=lambda x: x[1]["teams"])[0]
    return rank, wr, wr[popular]


def main():
    na, wa = load_battle_stats()
    files = sorted(f for f in os.listdir(SRC) if f.endswith(".gen1ou_team"))
    info = {fn: parse_team(fn) for fn in files}
    leads = {fn: parse_lead_raw(SRC / fn) for fn in files}

    sn_rank, sn_wr, _ = moveset_stats("Snorlax", files, info, na, wa)
    tau_rank, tau_wr, tau_popular_wr = moveset_stats("Tauros", files, info, na, wa)

    def admissible(fn: str) -> tuple[bool, list[str]]:
        species, moves = info[fn]
        reasons = []
        if leads[fn] in BANNED_LEADS:
            reasons.append(f"banned lead {leads[fn]}")
        if "Tauros" not in species:
            reasons.append("drops Tauros")
        if len(BIG3 - set(species)) > 1:
            reasons.append("drops >1 big three")
        if "Snorlax" in moves and sn_rank[moves["Snorlax"]] >= SNORLAX_BAD_RANK:
            reasons.append(
                f"bad Snorlax set (#{sn_rank[moves['Snorlax']]}/{len(sn_rank)})"
            )
        if "Tauros" in moves and tau_wr[moves["Tauros"]] < tau_popular_wr - 1e-9:
            reasons.append(
                f"Tauros set below most-popular (#{tau_rank[moves['Tauros']]}/{len(tau_rank)})"
            )
        return (not reasons, reasons)

    ranked = sorted(
        [f for f in files if f in na], key=lambda t: wa[t] / na[t], reverse=True
    )
    orig_rank = {t: i + 1 for i, t in enumerate(ranked)}

    # Seed: current expert_curated files -> orig expert names
    seed_mapping = {
        r["new_name"]: r["orig_name"]
        for r in csv.DictReader(open(SEED_DIR / "ranking_mapping.csv"))
    }
    seed_orig = set()
    for p in sorted(SEED_DIR.glob("team_*.gen1ou_team")):
        orig = seed_mapping.get(p.name)
        if orig is None:
            raise RuntimeError(f"no mapping for seed file {p.name}")
        seed_orig.add(orig)

    seed_teams = sorted(seed_orig)
    if len(seed_teams) != len(list(SEED_DIR.glob("team_*.gen1ou_team"))):
        raise RuntimeError("seed orig count mismatch")

    n_add = TARGET - len(seed_teams)
    if n_add < 0:
        raise RuntimeError(f"seed has {len(seed_teams)} teams > TARGET {TARGET}")

    # Count lead gaps vs expert-ish targets at n=50
    seed_leads = Counter(leads[t] for t in seed_teams)
    target_alak = 10
    target_geng = 8
    need_alak = max(0, target_alak - seed_leads.get("Alakazam", 0))
    need_geng = max(0, target_geng - seed_leads.get("Gengar", 0))
    # Split available slots between the two (prioritize larger gap)
    add_alak = min(need_alak, n_add)
    add_geng = min(need_geng, n_add - add_alak)
    # If still short, take best remaining of either lead
    remaining_slots = n_add - add_alak - add_geng

    candidates = []
    for fn in ranked[:SEARCH_DEPTH]:
        if fn in seed_orig:
            continue
        if leads[fn] not in FILL_LEADS:
            continue
        ok, _ = admissible(fn)
        if not ok:
            continue
        candidates.append(fn)

    by_lead: dict[str, list[str]] = {"Alakazam": [], "Gengar": []}
    for fn in candidates:
        by_lead[leads[fn]].append(fn)

    added: list[str] = []
    for lead, quota in (("Alakazam", add_alak), ("Gengar", add_geng)):
        added.extend(by_lead[lead][:quota])
    if remaining_slots:
        rest = [fn for fn in candidates if fn not in added]
        added.extend(rest[:remaining_slots])

    final_orig = seed_teams + added
    final_orig = sorted(set(final_orig), key=lambda t: wa[t] / na[t], reverse=True)

    if len(final_orig) < TARGET:
        print(
            f"WARNING: only {len(final_orig)} teams (wanted {TARGET}); "
            f"added {len(added)} of {n_add} requested"
        )

    # Write output
    if DST.parent.exists():
        shutil.rmtree(DST.parent)
    DST.mkdir(parents=True)

    rows = []
    for i, orig in enumerate(final_orig):
        new_name = f"team_{i:04d}.gen1ou_team"
        shutil.copyfile(SRC / orig, DST / new_name)
        rows.append(
            [
                i,
                new_name,
                orig,
                orig_rank.get(orig, ""),
                na.get(orig, 0),
                f"{wa[orig] / na[orig]:.4f}" if orig in na else "",
                leads[orig],
                "seed" if orig in seed_orig else "added",
            ]
        )

    with open(DST / "ranking_mapping.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "curated_rank",
                "new_name",
                "orig_name",
                "orig_fullpool_rank",
                "n_battles",
                "win_pct",
                "lead",
                "source",
            ]
        )
        w.writerows(rows)

    lead_dist = Counter(r[6] for r in rows)
    print(f"Wrote {len(rows)} teams -> {DST}")
    print(f"  seed: {len(seed_teams)}  added: {len(added)}")
    print(f"  added breakdown: Alakazam {sum(1 for t in added if leads[t]=='Alakazam')}, "
          f"Gengar {sum(1 for t in added if leads[t]=='Gengar')}")
    print("Lead distribution:")
    for lead, c in lead_dist.most_common():
        print(f"  {lead:<12} {c:3d}  ({100*c/len(rows):5.1f}%)")

    print("\nAdded teams:")
    for orig in added:
        ok, reasons = admissible(orig)
        print(
            f"  #{orig_rank[orig]:>3} {orig:<28} {leads[orig]:<9} "
            f"{wa[orig]/na[orig]*100:.2f}%"
        )

    # Mirror to NFS if available
    if Path("/mnt/nfs_client/jake/metamon_cache/teams").exists():
        if NFS_DST.parent.exists():
            shutil.rmtree(NFS_DST.parent)
        shutil.copytree(DST.parent, NFS_DST.parent)
        print(f"Mirrored -> {NFS_DST.parent}")


if __name__ == "__main__":
    main()
