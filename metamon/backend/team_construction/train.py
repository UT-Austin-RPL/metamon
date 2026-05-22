import argparse
import concurrent.futures
import csv
import itertools
import json
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

from metamon.backend.team_construction.matchup import run_matchup
from metamon.backend.team_construction.teams.parse import (
    parse_teams,
    parse_team_file,
    team_id_from_filename,
    write_moveset_index_csv,
    write_team_index_csv,
)
from metamon.backend.team_construction.update import (
    ensure_feature_bandit_state,
    feature_posterior_mean,
    load_pokemon_pool,
    load_state,
    posterior_mean,
    save_state,
    thompson_sample,
    thompson_score_team,
    update_from_feature_match,
    update_from_match,
)

TEAM_FILE_RE = re.compile(r"^team_(\d+)\.(gen[0-9][a-z0-9]*)_team$", re.IGNORECASE)


def _init_custom_teamset_dir(
    cache_dir: Path, set_name: str, battle_format: str
) -> Path:
    target_dir = cache_dir / "teams" / set_name / battle_format
    target_dir.mkdir(parents=True, exist_ok=True)
    for old in target_dir.glob(f"*.{battle_format}_team"):
        old.unlink()
    return target_dir


def _set_custom_team(
    team_dir: Path, battle_format: str, source_team_file: Path
) -> None:
    shutil.copyfile(source_team_file, team_dir / f"team_0001.{battle_format}_team")


def _resolve_team_source_dir(
    explicit_source_dir: Path | None,
    cache_dir: Path | None,
    team_set: str,
    battle_format: str,
) -> Path:
    if explicit_source_dir is not None:
        return explicit_source_dir
    if cache_dir is None:
        raise ValueError(
            "Provide --team-source-dir or set METAMON_CACHE_DIR to infer team source."
        )
    return cache_dir / "teams" / team_set / battle_format


def _team_features_from_file(
    team_file: Path,
    team_feature_cache: Dict[str, tuple[List[str], List[str]]],
) -> tuple[List[str], List[str]]:
    key = str(team_file)
    if key in team_feature_cache:
        return team_feature_cache[key]
    parsed = parse_team_file(team_file)
    names = [name for name, _ in parsed if name]
    moves: List[str] = []
    for _, move_tuple in parsed:
        moves.extend([m for m in move_tuple if m])
    team_feature_cache[key] = (names, moves)
    return names, moves


def _team_id_for_path(path: Path, source_team_dir: Path) -> str:
    base = team_id_from_filename(path)
    try:
        rel = str(path.relative_to(source_team_dir))
    except ValueError:
        rel = path.name
    return f"{base}:{rel}"


def _build_team_pool(
    source_team_dir: Path,
    battle_format: str,
    team_feature_cache: Dict[str, tuple[List[str], List[str]]],
) -> List[dict]:
    team_files = sorted(source_team_dir.rglob(f"*.{battle_format}_team"))
    if not team_files:
        raise FileNotFoundError(
            f"No .{battle_format}_team files found under {source_team_dir}"
        )

    pool: List[dict] = []
    for team_file in team_files:
        pokemon, moves = _team_features_from_file(team_file, team_feature_cache)
        if not pokemon:
            continue
        pool.append(
            {
                "team_id": _team_id_for_path(team_file, source_team_dir),
                "team_file": team_file,
                "pokemon": pokemon,
                "moves": moves,
            }
        )
    if not pool:
        raise RuntimeError(f"No parsable teams found in {source_team_dir}")
    return pool


def _select_candidate_team(
    *,
    pool: List[dict],
    state: dict,
    epsilon: float,
    rng: random.Random,
    candidate_pool_size: int,
    weight_team: float,
    weight_pokemon: float,
    weight_moves: float,
) -> tuple[dict, str]:
    if not pool:
        raise ValueError("Team pool is empty.")

    if rng.random() < epsilon:
        return rng.choice(pool), "explore_random"

    candidates = pool
    if 0 < candidate_pool_size < len(pool):
        candidates = rng.sample(pool, k=candidate_pool_size)

    best_team = candidates[0]
    best_score = thompson_score_team(
        state,
        team_id=best_team["team_id"],
        pokemon_names=best_team["pokemon"],
        moves=best_team["moves"],
        rng=rng,
        weight_team=weight_team,
        weight_pokemon=weight_pokemon,
        weight_moves=weight_moves,
    )
    for candidate in candidates[1:]:
        score = thompson_score_team(
            state,
            team_id=candidate["team_id"],
            pokemon_names=candidate["pokemon"],
            moves=candidate["moves"],
            rng=rng,
            weight_team=weight_team,
            weight_pokemon=weight_pokemon,
            weight_moves=weight_moves,
        )
        if score > best_score:
            best_score = score
            best_team = candidate
    return best_team, "exploit_thompson"


def _winner_from_result(result: str) -> str:
    r = result.upper().strip()
    if r == "WIN":
        return "a"
    if r == "LOSS":
        return "b"
    return "draw"


def _parse_gpu_list(raw: str) -> List[int]:
    out = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not out:
        raise ValueError("GPU list cannot be empty")
    return out


def _hardlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def _mirror_teamset_to_local_cache(
    source_team_dir: Path,
    local_cache_dir: Path,
    team_set_name: str,
    battle_format: str,
) -> Path:
    dst_dir = local_cache_dir / "teams" / team_set_name / battle_format
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(source_team_dir.rglob(f"*.{battle_format}_team")):
        _hardlink_or_copy(src, dst_dir / src.name)
    return dst_dir


def _load_team_index_mapping(index_csv: Path) -> Dict[str, set[str]]:
    mapping: Dict[str, set[str]] = {}
    with index_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("pokemon_name") or "").strip().lower()
            if not name:
                continue
            ids = {
                x.strip() for x in (row.get("team_ids") or "").split(",") if x.strip()
            }
            if ids:
                mapping[name] = ids
    return mapping


def _build_team_file_lookup(team_dir: Path) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    for path in sorted(team_dir.rglob("*.gen*_team")):
        match = TEAM_FILE_RE.match(path.name)
        if match:
            lookup[match.group(1)] = path
    return lookup


def _retrieve_team_fast(
    candidate_names: Sequence[str],
    team_ids_by_name: Dict[str, set[str]],
    team_file_by_id: Dict[str, Path],
) -> tuple[Path, List[str], int]:
    seen = set()
    ordered: List[str] = []
    for name in candidate_names:
        key = name.strip().lower()
        if key and key not in seen:
            ordered.append(key)
            seen.add(key)
    available = [n for n in ordered if n in team_ids_by_name and team_ids_by_name[n]]
    if not available:
        raise ValueError("No available names found in team index mapping.")

    for k in range(len(available), 0, -1):
        for combo in itertools.combinations(available, k):
            intersection = set(team_ids_by_name[combo[0]])
            for name in combo[1:]:
                intersection &= team_ids_by_name[name]
                if not intersection:
                    break
            if not intersection:
                continue
            for team_id in sorted(intersection):
                path = team_file_by_id.get(team_id)
                if path is not None:
                    return path, list(combo), k
    raise ValueError("Could not retrieve a matching team from candidate names.")


def _team_names_from_file(team_file: Path, cache: Dict[str, List[str]]) -> List[str]:
    key = str(team_file)
    if key in cache:
        return cache[key]
    content = team_file.read_text(encoding="utf-8", errors="replace")
    blocks = [block for block in content.split("\n\n") if block.strip()]
    names: List[str] = []
    for block in blocks:
        header = block.splitlines()[0].strip()
        if " @ " in header:
            header = header.split(" @ ", 1)[0].strip()
        if header.endswith(")") and "(" in header:
            header = header.rsplit("(", 1)[1].rstrip(")").strip()
        names.append(header)
    deduped: List[str] = []
    seen = set()
    for name in names:
        if name and name not in seen:
            deduped.append(name)
            seen.add(name)
    cache[key] = deduped
    return deduped


def _select_candidate_names(
    *,
    pool: List[str],
    state: dict,
    team_size: int,
    epsilon: float,
    rng: random.Random,
) -> tuple[List[str], str]:
    if len(pool) < team_size:
        raise ValueError(f"Need at least {team_size} pokemon in pool, got {len(pool)}")
    if rng.random() < epsilon:
        return rng.sample(pool, k=team_size), "explore_random"
    scored = [(name, thompson_sample(state, name, rng)) for name in pool]
    scored.sort(key=lambda item: item[1], reverse=True)
    return [name for name, _ in scored[:team_size]], "exploit_thompson"


def _run_matchup_with_retry(
    *,
    battle_format: str,
    num_battles: int,
    model_name: str,
    team_set_a: str,
    team_set_b: str,
    gpu_a: int,
    gpu_b: int,
    work_dir: Path,
    checkpoint: int | None,
    print_match_stats: bool,
    max_retries: int,
    retry_sleep_sec: float,
) -> Dict[str, object]:
    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            return run_matchup(
                battle_format=battle_format,
                num_battles=num_battles,
                model_name=model_name,
                team_set_a=team_set_a,
                team_set_b=team_set_b,
                gpu_a=gpu_a,
                gpu_b=gpu_b,
                work_dir=work_dir,
                checkpoint=checkpoint,
                print_match_stats=print_match_stats,
            )
        except RuntimeError as exc:
            last_error = str(exc)
            if attempt >= max_retries:
                break
            time.sleep(retry_sleep_sec)
    raise RuntimeError(
        f"run_matchup failed for gpu pair ({gpu_a},{gpu_b}) after retries: {last_error}"
    )


def _run_single_mode(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)

    if args.reset_tmp and args.data_root.exists():
        print(f"[setup] deleting {args.data_root}")
        shutil.rmtree(args.data_root)
    args.data_root.mkdir(parents=True, exist_ok=True)

    cache_dir_env = os.environ.get("METAMON_CACHE_DIR")
    cache_dir = Path(cache_dir_env) if cache_dir_env else None
    source_team_dir = _resolve_team_source_dir(
        explicit_source_dir=args.team_source_dir,
        cache_dir=cache_dir,
        team_set=args.opponent_team_set,
        battle_format=args.battle_format,
    )
    print(f"[setup] source teams: {source_team_dir}")

    team_index_csv = args.data_root / f"{args.battle_format}_pokemon_team_index.csv"
    moveset_index_csv = (
        args.data_root / f"{args.battle_format}_pokemon_moveset_index.csv"
    )
    state_path = args.data_root / "name_bandit_state.json"
    history_path = args.data_root / "match_history.jsonl"
    batch_history_path = args.data_root / "batch_history.jsonl"
    matchup_work_dir = args.data_root / "team_construction_battles"

    if args.reset_tmp or not (team_index_csv.exists() and moveset_index_csv.exists()):
        print("[setup] parsing teams -> csv indexes")
        pokemon_to_teams, pokemon_to_movesets = parse_teams(
            source_team_dir, fallback_format=None
        )
        write_team_index_csv(pokemon_to_teams, team_index_csv)
        write_moveset_index_csv(pokemon_to_movesets, moveset_index_csv)
    else:
        print("[setup] using cached csv indexes")

    team_feature_cache: Dict[str, tuple[List[str], List[str]]] = {}
    team_pool = _build_team_pool(
        source_team_dir=source_team_dir,
        battle_format=args.battle_format,
        team_feature_cache=team_feature_cache,
    )
    print(f"[setup] parsed team pool size={len(team_pool)}")

    state = load_state(state_path)
    state = ensure_feature_bandit_state(state if not args.reset_tmp else None)
    save_state(state_path, state)

    wandb_run = None
    if args.log_wandb:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is not installed. Install it or run without --log-wandb."
            ) from exc
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                "execution_mode": "single",
                "battle_format": args.battle_format,
                "opponent_team_set": args.opponent_team_set,
                "learner_team_set": args.learner_team_set,
                "model_name": args.model_name,
                "checkpoint": args.checkpoint,
                "num_batches": args.num_batches,
                "batch_size": args.batch_size,
                "thompson_candidate_pool_size": args.thompson_candidate_pool_size,
                "weight_team": args.weight_team,
                "weight_pokemon": args.weight_pokemon,
                "weight_moves": args.weight_moves,
                "epsilon_start": args.epsilon_start,
                "epsilon_end": args.epsilon_end,
                "seed": args.seed,
            },
        )
        wandb.define_metric("batch")
        wandb.define_metric("*", step_metric="batch")
        print(
            f"[wandb] enabled project={args.wandb_project} "
            f"entity={args.wandb_entity or '<default>'}"
        )

    if cache_dir is None:
        raise ValueError("METAMON_CACHE_DIR must be set for learner team-set writes.")

    learner_team_dir = _init_custom_teamset_dir(
        cache_dir=cache_dir,
        set_name=args.learner_team_set,
        battle_format=args.battle_format,
    )
    team_name_cache: Dict[str, tuple[List[str], List[str]]] = {}
    cumulative_wins = 0.0
    cumulative_games = 0

    batch = 0
    while True:
        batch += 1
        if args.num_batches is None:
            epsilon = args.epsilon_start
        else:
            t = (batch - 1) / max(1, args.num_batches - 1)
            epsilon = args.epsilon_start + t * (args.epsilon_end - args.epsilon_start)
            epsilon = max(0.0, min(1.0, epsilon))

        print(f"[batch {batch:03d}] epsilon={epsilon:.3f}")
        batch_score = 0.0
        batch_games = 0
        batch_explore = 0
        batch_modes: List[str] = []

        for match_idx in range(1, args.batch_size + 1):
            selected_team, selection_mode = _select_candidate_team(
                pool=team_pool,
                state=state,
                epsilon=epsilon,
                rng=rng,
                candidate_pool_size=args.thompson_candidate_pool_size,
                weight_team=args.weight_team,
                weight_pokemon=args.weight_pokemon,
                weight_moves=args.weight_moves,
            )
            batch_modes.append(selection_mode)
            if selection_mode.startswith("explore"):
                batch_explore += 1

            team_file = selected_team["team_file"]
            team_a_id = selected_team["team_id"]
            team_a_names = selected_team["pokemon"]
            team_a_moves = selected_team["moves"]
            _set_custom_team(
                team_dir=learner_team_dir,
                battle_format=args.battle_format,
                source_team_file=team_file,
            )
            print(
                f"[batch {batch:03d} match {match_idx:02d}] mode={selection_mode} "
                f"team={team_file.name} team_id={team_a_id}"
            )

            result = None
            for attempt in range(args.matchup_max_retries + 1):
                try:
                    result = run_matchup(
                        battle_format=args.battle_format,
                        num_battles=1,
                        model_name=args.model_name,
                        team_set_a=args.learner_team_set,
                        team_set_b=args.opponent_team_set,
                        gpu_a=args.gpu_a,
                        gpu_b=args.gpu_b,
                        work_dir=matchup_work_dir,
                        checkpoint=args.checkpoint,
                        print_match_stats=args.print_match_stats,
                    )
                    break
                except RuntimeError as exc:
                    if attempt >= args.matchup_max_retries:
                        print(
                            f"[batch {batch:03d} match {match_idx:02d}] "
                            f"failed after retries: {exc}"
                        )
                    else:
                        print(
                            f"[batch {batch:03d} match {match_idx:02d}] "
                            f"worker failure, retry {attempt + 1}/{args.matchup_max_retries}"
                        )
                        time.sleep(args.matchup_retry_sleep_sec)
            if result is None:
                continue

            acceptor_matches = result["acceptor_matches"]
            challenger_matches = result["challenger_matches"]
            if not acceptor_matches or not challenger_matches:
                continue

            a_match = acceptor_matches[0]
            b_match = challenger_matches[0]
            team_b_path = b_match.get("team_file_path", "")
            team_b_names: List[str] = []
            team_b_moves: List[str] = []
            team_b_id = ""
            if team_b_path:
                team_b_file = Path(team_b_path)
                team_b_names, team_b_moves = _team_features_from_file(
                    team_b_file, team_name_cache
                )
                team_b_id = _team_id_for_path(team_b_file, source_team_dir)

            winner = _winner_from_result(a_match["result"])
            score = 1.0 if winner == "a" else 0.5 if winner == "draw" else 0.0
            batch_score += score
            batch_games += 1
            cumulative_wins += score
            cumulative_games += 1

            if team_b_names and team_b_id:
                update_from_feature_match(
                    state,
                    team_a_id=team_a_id,
                    team_a_pokemon=team_a_names,
                    team_a_moves=team_a_moves,
                    team_b_id=team_b_id,
                    team_b_pokemon=team_b_names,
                    team_b_moves=team_b_moves,
                    winner=winner,
                )
                history_path.parent.mkdir(parents=True, exist_ok=True)
                with history_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "batch": batch,
                                "match_idx": match_idx,
                                "winner": winner,
                                "team_a_id": team_a_id,
                                "team_a": team_a_names,
                                "team_a_move_count": len(team_a_moves),
                                "team_b_id": team_b_id,
                                "team_b": team_b_names,
                                "team_b_move_count": len(team_b_moves),
                                "battle_id": a_match.get("battle_id", ""),
                                "selection_mode": selection_mode,
                            }
                        )
                        + "\n"
                    )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "batch": batch,
                        "match_idx_in_batch": match_idx,
                        "match_win_score": score,
                        "match_selection_explore": (
                            1 if selection_mode.startswith("explore") else 0
                        ),
                        "match_team_a_move_count": len(team_a_moves),
                    }
                )

        save_state(state_path, state)

        batch_wr = batch_score / max(1, batch_games)
        cumulative_wr = cumulative_wins / max(1, cumulative_games)
        top_names = sorted(
            state["pokemon"].keys(),
            key=lambda name: feature_posterior_mean(state["pokemon"], name),
            reverse=True,
        )[:6]
        top_moves = sorted(
            state["moves"].keys(),
            key=lambda move: feature_posterior_mean(state["moves"], move),
            reverse=True,
        )[:10]
        print(
            f"[batch {batch:03d}] batch_wr={batch_wr:.3f} cum_wr={cumulative_wr:.3f} "
            f"games={batch_games}/{args.batch_size} "
            f"explore_rate={batch_explore / max(1, args.batch_size):.3f} "
            f"top_pokemon={top_names}"
        )

        with batch_history_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "batch": batch,
                        "batch_win_rate": batch_wr,
                        "cumulative_win_rate": cumulative_wr,
                        "epsilon": epsilon,
                        "batch_games": batch_games,
                        "target_games": args.batch_size,
                        "explore_count": batch_explore,
                        "explore_rate": batch_explore / max(1, args.batch_size),
                        "top_pokemon": top_names,
                        "top_moves": top_moves,
                        "modes": batch_modes,
                    }
                )
                + "\n"
            )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "batch": batch,
                    "batch_win_rate": batch_wr,
                    "cumulative_win_rate": cumulative_wr,
                    "epsilon": epsilon,
                    "batch_games": batch_games,
                    "explore_rate": batch_explore / max(1, args.batch_size),
                    "top_pokemon": ", ".join(top_names),
                    "top_moves": ", ".join(top_moves),
                }
            )

        if args.num_batches is not None and batch >= args.num_batches:
            break

    if wandb_run is not None:
        wandb_run.finish()
    print(f"[done] state={state_path} batch_history={batch_history_path}")


def _run_multi_mode(args: argparse.Namespace) -> None:
    if args.team_source_dir is None:
        raise ValueError("--team-source-dir is required when --execution-mode multi")

    num_batches = args.num_batches if args.num_batches is not None else 100
    acceptor_gpus = _parse_gpu_list(args.acceptor_gpus)
    challenger_gpus = _parse_gpu_list(args.challenger_gpus)
    if len(acceptor_gpus) != len(challenger_gpus):
        raise ValueError("acceptor-gpus and challenger-gpus must have the same length")
    gpu_pairs = list(zip(acceptor_gpus, challenger_gpus))

    rng = random.Random(args.seed)
    if args.reset_tmp and args.data_root.exists():
        shutil.rmtree(args.data_root)
    args.data_root.mkdir(parents=True, exist_ok=True)

    source_team_dir = args.team_source_dir

    local_cache_dir = args.data_root / "metamon_cache"
    os.environ["METAMON_CACHE_DIR"] = str(local_cache_dir)
    _mirror_teamset_to_local_cache(
        source_team_dir=source_team_dir,
        local_cache_dir=local_cache_dir,
        team_set_name=args.opponent_team_set,
        battle_format=args.battle_format,
    )

    team_index_csv = args.data_root / f"{args.battle_format}_pokemon_team_index.csv"
    moveset_index_csv = (
        args.data_root / f"{args.battle_format}_pokemon_moveset_index.csv"
    )
    state_path = args.data_root / "name_bandit_state.json"
    history_path = args.data_root / "match_history.jsonl"
    batch_history_path = args.data_root / "batch_history.jsonl"
    matchup_work_dir = args.data_root / "team_construction_battles"

    if args.reset_tmp or not (team_index_csv.exists() and moveset_index_csv.exists()):
        pokemon_to_teams, pokemon_to_movesets = parse_teams(
            source_team_dir, fallback_format=None
        )
        write_team_index_csv(pokemon_to_teams, team_index_csv)
        write_moveset_index_csv(pokemon_to_movesets, moveset_index_csv)

    pool, _ = load_pokemon_pool(team_index_csv)
    team_ids_by_name = _load_team_index_mapping(team_index_csv)
    team_file_by_id = _build_team_file_lookup(source_team_dir)
    if not team_file_by_id:
        raise ValueError(f"No team files found in {source_team_dir}")

    state = load_state(state_path)
    if args.reset_tmp:
        state = {
            "version": 1,
            "metric": "beta_bernoulli_name_bandit",
            "matches": 0,
            "pokemon": {},
        }
        save_state(state_path, state)

    learner_team_dir = _init_custom_teamset_dir(
        cache_dir=local_cache_dir,
        set_name=args.learner_team_set,
        battle_format=args.battle_format,
    )
    team_name_cache: Dict[str, List[str]] = {}
    cumulative_wins = 0.0
    cumulative_games = 0

    wandb_run = None
    if args.log_wandb:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is not installed. Install it or run without --log-wandb."
            ) from exc

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                "execution_mode": "multi",
                "battle_format": args.battle_format,
                "num_batches": num_batches,
                "batch_size_per_gpu": args.batch_size_per_gpu,
                "gpu_pairs": gpu_pairs,
                "epsilon_start": args.epsilon_start,
                "epsilon_end": args.epsilon_end,
            },
        )
        wandb.define_metric("batch")
        wandb.define_metric("*", step_metric="batch")

    for batch in range(1, num_batches + 1):
        t = (batch - 1) / max(1, num_batches - 1)
        epsilon = args.epsilon_start + t * (args.epsilon_end - args.epsilon_start)
        epsilon = max(0.0, min(1.0, epsilon))

        candidate_names, selection_mode = _select_candidate_names(
            pool=pool,
            state=state,
            team_size=args.team_size,
            epsilon=epsilon,
            rng=rng,
        )
        team_file, matched_norm_names, match_size = _retrieve_team_fast(
            candidate_names=candidate_names,
            team_ids_by_name=team_ids_by_name,
            team_file_by_id=team_file_by_id,
        )
        team_a_names = _team_names_from_file(team_file, team_name_cache)
        _set_custom_team(
            team_dir=learner_team_dir,
            battle_format=args.battle_format,
            source_team_file=team_file,
        )

        futures = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(gpu_pairs)
        ) as executor:
            for gpu_a, gpu_b in gpu_pairs:
                futures.append(
                    executor.submit(
                        _run_matchup_with_retry,
                        battle_format=args.battle_format,
                        num_battles=args.batch_size_per_gpu,
                        model_name=args.model_name,
                        team_set_a=args.learner_team_set,
                        team_set_b=args.opponent_team_set,
                        gpu_a=gpu_a,
                        gpu_b=gpu_b,
                        work_dir=matchup_work_dir,
                        checkpoint=args.checkpoint,
                        print_match_stats=args.print_match_stats,
                        max_retries=args.matchup_max_retries,
                        retry_sleep_sec=args.matchup_retry_sleep_sec,
                    )
                )
            results = [future.result() for future in futures]

        batch_score = 0.0
        batch_games = 0
        for result in results:
            a_matches = result["acceptor_matches"]
            b_matches = result["challenger_matches"]
            for a_match, b_match in zip(a_matches, b_matches):
                team_b_path = b_match.get("team_file_path", "")
                team_b_names = (
                    _team_names_from_file(Path(team_b_path), team_name_cache)
                    if team_b_path
                    else []
                )
                winner = _winner_from_result(a_match["result"])
                if team_b_names:
                    update_from_match(state, team_a_names, team_b_names, winner)
                    history_path.parent.mkdir(parents=True, exist_ok=True)
                    with history_path.open("a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                {
                                    "batch": batch,
                                    "winner": winner,
                                    "team_a": team_a_names,
                                    "team_b": team_b_names,
                                    "battle_id": a_match.get("battle_id", ""),
                                }
                            )
                            + "\n"
                        )

                score = 1.0 if winner == "a" else 0.5 if winner == "draw" else 0.0
                batch_score += score
                batch_games += 1

        save_state(state_path, state)
        batch_wr = batch_score / max(1, batch_games)
        cumulative_wins += batch_score
        cumulative_games += batch_games
        cumulative_wr = cumulative_wins / max(1, cumulative_games)
        top_names = sorted(
            pool, key=lambda name: posterior_mean(state, name), reverse=True
        )[:6]

        with batch_history_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "batch": batch,
                        "epsilon": epsilon,
                        "selection_mode": selection_mode,
                        "candidate_names": candidate_names,
                        "matched_names": matched_norm_names,
                        "team_file": str(team_file),
                        "batch_games": batch_games,
                        "batch_win_rate": batch_wr,
                        "cumulative_win_rate": cumulative_wr,
                        "top_names": top_names,
                    }
                )
                + "\n"
            )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "batch": batch,
                    "batch_win_rate": batch_wr,
                    "cumulative_win_rate": cumulative_wr,
                    "epsilon": epsilon,
                    "match_size": match_size,
                    "selection_explore": (
                        1 if selection_mode.startswith("explore") else 0
                    ),
                    "top_names": ", ".join(top_names),
                    "chosen_team_file": team_file.name,
                    "batch_games": batch_games,
                }
            )

        print(
            f"[batch {batch:03d}] wr={batch_wr:.3f} cum={cumulative_wr:.3f} "
            f"games={batch_games} mode={selection_mode} epsilon={epsilon:.3f}"
        )

    if wandb_run is not None:
        wandb_run.finish()
    print(f"[done] state={state_path} batch_history={batch_history_path}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in {"pipeline", "new"}:
        from metamon.backend.team_construction.cli import main as pipeline_main

        pipeline_main(sys.argv[2:])
        return

    parser = argparse.ArgumentParser(
        description=(
            "Team-construction trainer. Use --execution-mode single (legacy single-GPU), "
            "--execution-mode multi (legacy multi-GPU), or `pipeline` for the new model-based flow."
        )
    )

    parser.add_argument(
        "--execution-mode", choices=["single", "multi"], default="single"
    )

    parser.add_argument("--battle-format", default="gen1ou")
    parser.add_argument("--opponent-team-set", default="competitive")
    parser.add_argument("--learner-team-set", default="team_construction_learner")
    parser.add_argument("--model-name", default="Kakuna")
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument(
        "--num-batches",
        type=int,
        default=None,
        help=(
            "Number of batches. Omit for perpetual training in single mode; "
            "defaults to 100 in multi mode."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batch-size-per-gpu", type=int, default=16)
    parser.add_argument("--team-size", type=int, default=6)

    parser.add_argument("--gpu-a", type=int, default=0)
    parser.add_argument("--gpu-b", type=int, default=1)
    parser.add_argument("--acceptor-gpus", type=str, default="0,1,2,3")
    parser.add_argument("--challenger-gpus", type=str, default="4,5,6,7")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/tmp/team_construction"),
        help="Working directory for indexes/state/history/logs.",
    )
    parser.add_argument(
        "--team-source-dir",
        type=Path,
        default=None,
        help="Optional source team directory (required for multi mode).",
    )
    parser.add_argument(
        "--reset-tmp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete --data-root before training (use --no-reset-tmp to append).",
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=0.35,
        help="Initial exploration probability.",
    )
    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=0.05,
        help="Final exploration probability at last batch.",
    )
    parser.add_argument(
        "--thompson-candidate-pool-size",
        type=int,
        default=2048,
        help=(
            "Max teams to score per exploit step in single mode (sampled from full pool). "
            "Use <=0 to score all teams."
        ),
    )
    parser.add_argument(
        "--weight-team",
        type=float,
        default=0.35,
        help="Weight for team-identity Thompson prior (single mode).",
    )
    parser.add_argument(
        "--weight-pokemon",
        type=float,
        default=0.40,
        help="Weight for Pokemon-level Thompson prior (single mode).",
    )
    parser.add_argument(
        "--weight-moves",
        type=float,
        default=0.25,
        help="Weight for move-level Thompson prior (single mode).",
    )
    parser.add_argument(
        "--print-match-stats",
        action="store_true",
        help="Print per-match stats from matchup.py.",
    )

    parser.add_argument("--log-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="team_construction")
    parser.add_argument(
        "--wandb-entity",
        default=os.environ.get("METAMON_WANDB_ENTITY", None),
    )
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--matchup-max-retries",
        type=int,
        default=3,
        help="Retry count when matchup workers fail.",
    )
    parser.add_argument(
        "--matchup-retry-sleep-sec",
        type=float,
        default=5.0,
        help="Sleep duration between matchup retries.",
    )
    args = parser.parse_args()

    if args.execution_mode == "single":
        _run_single_mode(args)
    else:
        _run_multi_mode(args)


if __name__ == "__main__":
    main()
