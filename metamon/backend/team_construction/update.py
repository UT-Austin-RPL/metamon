import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


def _norm(name: str) -> str:
    return name.strip().lower()


def load_pokemon_pool(index_csv: Path) -> Tuple[List[str], Dict[str, str]]:
    if not index_csv.exists():
        raise FileNotFoundError(f"Index CSV not found: {index_csv}")
    pool: List[str] = []
    canonical: Dict[str, str] = {}
    with index_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "pokemon_name" not in (reader.fieldnames or []):
            raise ValueError(f"{index_csv} must contain a pokemon_name column")
        for row in reader:
            name = (row.get("pokemon_name") or "").strip()
            if not name:
                continue
            key = _norm(name)
            canonical[key] = name
            pool.append(name)
    if not pool:
        raise ValueError(f"No pokemon names found in {index_csv}")
    pool = sorted(set(pool))
    return pool, canonical


def canonicalize_team(team: List[str], canonical: Dict[str, str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for p in team:
        key = _norm(p)
        name = canonical.get(key, p.strip())
        if name and name not in seen:
            out.append(name)
            seen.add(name)
    return out


def default_state() -> dict:
    return {
        "version": 1,
        "metric": "beta_bernoulli_name_bandit",
        "matches": 0,
        "pokemon": {},
    }


def default_feature_state() -> dict:
    return {
        "version": 2,
        "metric": "beta_bernoulli_team_feature_bandit",
        "matches": 0,
        "pokemon": {},
        "moves": {},
        "teams": {},
    }


def load_state(path: Path) -> dict:
    if not path.exists():
        return default_state()
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def ensure_feature_bandit_state(state: dict | None) -> dict:
    if not state or state.get("metric") != "beta_bernoulli_team_feature_bandit":
        return default_feature_state()
    state.setdefault("version", 2)
    state.setdefault("matches", 0)
    state.setdefault("pokemon", {})
    state.setdefault("moves", {})
    state.setdefault("teams", {})
    return state


def ensure_pokemon(state: dict, name: str) -> None:
    if name not in state["pokemon"]:
        state["pokemon"][name] = {"alpha": 1.0, "beta": 1.0, "wins": 0, "losses": 0}


def ensure_feature(state_bucket: dict, name: str) -> None:
    if name not in state_bucket:
        state_bucket[name] = {"alpha": 1.0, "beta": 1.0, "wins": 0, "losses": 0}


def posterior_mean(state: dict, name: str) -> float:
    ensure_pokemon(state, name)
    stats = state["pokemon"][name]
    return stats["alpha"] / (stats["alpha"] + stats["beta"])


def feature_posterior_mean(state_bucket: dict, name: str) -> float:
    ensure_feature(state_bucket, name)
    stats = state_bucket[name]
    return stats["alpha"] / (stats["alpha"] + stats["beta"])


def thompson_sample(state: dict, name: str, rng: random.Random) -> float:
    ensure_pokemon(state, name)
    stats = state["pokemon"][name]
    return rng.betavariate(stats["alpha"], stats["beta"])


def feature_thompson_sample(state_bucket: dict, name: str, rng: random.Random) -> float:
    ensure_feature(state_bucket, name)
    stats = state_bucket[name]
    return rng.betavariate(stats["alpha"], stats["beta"])


def thompson_score_team(
    state: dict,
    *,
    team_id: str,
    pokemon_names: List[str],
    moves: List[str],
    rng: random.Random,
    weight_team: float = 0.35,
    weight_pokemon: float = 0.40,
    weight_moves: float = 0.25,
) -> float:
    state = ensure_feature_bandit_state(state)

    pokemon_unique = sorted({p.strip() for p in pokemon_names if p.strip()})
    moves_unique = sorted({m.strip() for m in moves if m.strip()})

    team_sample = feature_thompson_sample(state["teams"], team_id, rng)
    if pokemon_unique:
        p_samples = [
            feature_thompson_sample(state["pokemon"], p, rng) for p in pokemon_unique
        ]
        pokemon_sample = sum(p_samples) / len(p_samples)
    else:
        pokemon_sample = 0.5

    if moves_unique:
        m_samples = [
            feature_thompson_sample(state["moves"], m, rng) for m in moves_unique
        ]
        move_sample = sum(m_samples) / len(m_samples)
    else:
        move_sample = 0.5

    total = weight_team + weight_pokemon + weight_moves
    if total <= 0:
        return team_sample
    return (
        (weight_team * team_sample)
        + (weight_pokemon * pokemon_sample)
        + (weight_moves * move_sample)
    ) / total


def _update_bucket_from_outcome(
    bucket: dict, keys: List[str], *, won: bool, draw: bool
) -> None:
    unique_keys = {k for k in keys if k}
    for key in unique_keys:
        ensure_feature(bucket, key)
        stats = bucket[key]
        if draw:
            stats["alpha"] += 0.5
            stats["beta"] += 0.5
        elif won:
            stats["alpha"] += 1.0
            stats["wins"] += 1
        else:
            stats["beta"] += 1.0
            stats["losses"] += 1


def update_from_feature_match(
    state: dict,
    *,
    team_a_id: str,
    team_a_pokemon: List[str],
    team_a_moves: List[str],
    team_b_id: str,
    team_b_pokemon: List[str],
    team_b_moves: List[str],
    winner: str,
) -> None:
    state = ensure_feature_bandit_state(state)

    if winner not in {"a", "b", "draw"}:
        raise ValueError("winner must be one of: a, b, draw")

    draw = winner == "draw"
    a_won = winner == "a"
    b_won = winner == "b"

    _update_bucket_from_outcome(state["teams"], [team_a_id], won=a_won, draw=draw)
    _update_bucket_from_outcome(state["teams"], [team_b_id], won=b_won, draw=draw)

    _update_bucket_from_outcome(state["pokemon"], team_a_pokemon, won=a_won, draw=draw)
    _update_bucket_from_outcome(state["pokemon"], team_b_pokemon, won=b_won, draw=draw)

    _update_bucket_from_outcome(state["moves"], team_a_moves, won=a_won, draw=draw)
    _update_bucket_from_outcome(state["moves"], team_b_moves, won=b_won, draw=draw)

    state["matches"] += 1


def update_from_match(
    state: dict, team_a: List[str], team_b: List[str], winner: str
) -> None:
    a_set = set(team_a)
    b_set = set(team_b)
    all_names = a_set.union(b_set)
    for p in all_names:
        ensure_pokemon(state, p)

    if winner == "a":
        winners, losers = a_set, b_set
        draw = False
    elif winner == "b":
        winners, losers = b_set, a_set
        draw = False
    elif winner == "draw":
        draw = True
        winners, losers = set(), set()
    else:
        raise ValueError("winner must be one of: a, b, draw")

    if draw:
        for p in all_names:
            state["pokemon"][p]["alpha"] += 0.5
            state["pokemon"][p]["beta"] += 0.5
    else:
        for p in winners:
            state["pokemon"][p]["alpha"] += 1.0
            state["pokemon"][p]["wins"] += 1
        for p in losers:
            state["pokemon"][p]["beta"] += 1.0
            state["pokemon"][p]["losses"] += 1

    state["matches"] += 1


def propose_team(
    current_team: List[str],
    pool: List[str],
    state: dict,
    replacements: int,
    rng: random.Random,
) -> List[str]:
    if replacements <= 0:
        return list(current_team)

    replacements = min(replacements, len(current_team))
    team_samples = {p: thompson_sample(state, p, rng) for p in current_team}
    to_drop = sorted(current_team, key=lambda p: team_samples[p])[:replacements]
    remaining = [p for p in current_team if p not in set(to_drop)]

    candidates = [p for p in pool if p not in set(remaining)]
    candidate_scores = [(p, thompson_sample(state, p, rng)) for p in candidates]
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    additions = [p for p, _ in candidate_scores[:replacements]]

    updated = remaining + additions
    return updated[: len(current_team)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Update standalone name-bandit state from a match result and propose "
            "new team names."
        )
    )
    parser.add_argument(
        "--team-a",
        nargs="+",
        required=True,
        help="Team A pokemon names (typically 6).",
    )
    parser.add_argument(
        "--team-b",
        nargs="+",
        required=True,
        help="Team B pokemon names (typically 6).",
    )
    parser.add_argument(
        "--winner",
        required=True,
        choices=["a", "b", "draw"],
        help="Winner of the match.",
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=Path("/tmp/team_construction/gen1ou_pokemon_team_index.csv"),
        help="Pokemon pool CSV.",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=Path("/tmp/team_construction/name_bandit_state.json"),
        help="Path to persistent bandit state JSON.",
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        default=Path("/tmp/team_construction/match_history.jsonl"),
        help="Path to append-only history log.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing state/history files before applying this update.",
    )
    parser.add_argument(
        "--replacements-loser",
        type=int,
        default=1,
        help="How many pokemon to replace on the losing team.",
    )
    parser.add_argument(
        "--replacements-winner",
        type=int,
        default=0,
        help="How many pokemon to replace on the winning team.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pool, canonical = load_pokemon_pool(args.index_csv)
    team_a = canonicalize_team(args.team_a, canonical)
    team_b = canonicalize_team(args.team_b, canonical)
    if not team_a or not team_b:
        raise ValueError("Both teams must contain at least one pokemon name.")

    if args.reset:
        if args.state_path.exists():
            args.state_path.unlink()
        if args.history_path.exists():
            args.history_path.unlink()

    state = load_state(args.state_path)
    update_from_match(state, team_a=team_a, team_b=team_b, winner=args.winner)

    rng = random.Random(args.seed + state["matches"])
    if args.winner == "a":
        team_a_next = propose_team(
            team_a, pool, state, replacements=args.replacements_winner, rng=rng
        )
        team_b_next = propose_team(
            team_b, pool, state, replacements=args.replacements_loser, rng=rng
        )
    elif args.winner == "b":
        team_a_next = propose_team(
            team_a, pool, state, replacements=args.replacements_loser, rng=rng
        )
        team_b_next = propose_team(
            team_b, pool, state, replacements=args.replacements_winner, rng=rng
        )
    else:
        team_a_next = propose_team(
            team_a, pool, state, replacements=args.replacements_loser, rng=rng
        )
        team_b_next = propose_team(
            team_b, pool, state, replacements=args.replacements_loser, rng=rng
        )

    save_state(args.state_path, state)
    args.history_path.parent.mkdir(parents=True, exist_ok=True)
    with args.history_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "match_num": state["matches"],
                    "winner": args.winner,
                    "team_a": team_a,
                    "team_b": team_b,
                    "team_a_next": team_a_next,
                    "team_b_next": team_b_next,
                }
            )
            + "\n"
        )

    print(f"Updated state: {args.state_path} (matches={state['matches']})")
    print(f"Team A current: {team_a}")
    print(f"Team B current: {team_b}")
    print(f"Winner: {args.winner}")
    print(f"Team A next: {team_a_next}")
    print(f"Team B next: {team_b_next}")
    print("Posterior means for Team A next:")
    for p in team_a_next:
        print(f"  {p}: {posterior_mean(state, p):.3f}")
    print("Posterior means for Team B next:")
    for p in team_b_next:
        print(f"  {p}: {posterior_mean(state, p):.3f}")


if __name__ == "__main__":
    main()
