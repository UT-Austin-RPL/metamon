from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import os
import random
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from tqdm import tqdm

from .core import (
    BattleExample,
    Team,
    battle_example_from_json_dict,
    battle_example_to_json_dict,
)


@dataclass(frozen=True)
class SimulationMetadata:
    format_id: str
    agent_name: str
    n_battles: int
    seed: int
    backend: str
    concurrency: int
    sampling_strategy: str = "uniform"
    sampling_metadata: dict | None = None


def sample_team(
    pool_ids: Sequence[int],
    team_size: int = 6,
    replace: bool = False,
    rng: random.Random | None = None,
    species_clause_keys: Mapping[int, object] | None = None,
) -> Team:
    """Sample one legal team as a sorted tuple of distinct Pokemon IDs."""

    if team_size <= 0:
        raise ValueError(f"team_size must be > 0, got {team_size}")
    pool = sorted({int(x) for x in pool_ids})
    if not pool:
        raise ValueError("pool_ids is empty")
    if rng is None:
        rng = random.Random()

    if species_clause_keys is not None:
        members_by_key: dict[object, list[int]] = {}
        for member in pool:
            key = species_clause_keys.get(member, member)
            members_by_key.setdefault(key, []).append(member)
        keys = sorted(members_by_key.keys(), key=str)
        if len(keys) < team_size:
            raise ValueError(
                "Pool has too few unique species-clause groups for sampling: "
                f"{len(keys)} < team_size={team_size}"
            )
        chosen_keys = rng.sample(keys, k=team_size)
        sampled = [int(rng.choice(members_by_key[key])) for key in chosen_keys]
    else:
        if len(pool) < team_size:
            raise ValueError(
                f"Pool too small for sampling: {len(pool)} < team_size={team_size}"
            )
        if replace:
            sampled = []
            seen: set[int] = set()
            max_attempts = max(100, 10 * team_size)
            attempts = 0
            while len(sampled) < team_size and attempts < max_attempts:
                attempts += 1
                pick = int(rng.choice(pool))
                if pick in seen:
                    continue
                sampled.append(pick)
                seen.add(pick)
            if len(sampled) < team_size:
                remaining = [x for x in pool if x not in seen]
                rng.shuffle(remaining)
                sampled.extend(remaining[: team_size - len(sampled)])
        else:
            sampled = rng.sample(pool, k=team_size)

    if len(set(sampled)) != len(sampled):
        raise ValueError("Duplicate Pokemon in sampled team. Use replace=False.")
    if species_clause_keys is not None:
        clause_members = [species_clause_keys.get(member, member) for member in sampled]
        if len(set(clause_members)) != len(clause_members):
            raise ValueError("Sampled team violates species clause.")
    return tuple(sorted(int(x) for x in sampled))


def make_uniform_matchup_sampler(
    pool_ids: Sequence[int],
    *,
    team_size: int,
    replace: bool,
    rng: random.Random,
    species_clause_keys: Mapping[int, object] | None = None,
) -> Callable[[], tuple[Team, Team]]:
    """Uniform metagame sampler used by the paper-style training setup."""

    def _sample() -> tuple[Team, Team]:
        return (
            sample_team(
                pool_ids,
                team_size=team_size,
                replace=replace,
                rng=rng,
                species_clause_keys=species_clause_keys,
            ),
            sample_team(
                pool_ids,
                team_size=team_size,
                replace=replace,
                rng=rng,
                species_clause_keys=species_clause_keys,
            ),
        )

    return _sample


def make_active_matchup_sampler(
    pool_ids: Sequence[int],
    *,
    pair_evaluator: Callable[[Team, Team], float],
    team_size: int,
    replace: bool,
    rng: random.Random,
    candidate_pool_size: int = 256,
    uniform_mix: float = 0.25,
    min_uncertainty: float = 1e-6,
    species_clause_keys: Mapping[int, object] | None = None,
) -> Callable[[], tuple[Team, Team]]:
    """Model-guided active sampler that prioritizes uncertain team-vs-team pairs."""

    if candidate_pool_size < 2:
        raise ValueError(f"candidate_pool_size must be >= 2, got {candidate_pool_size}")
    if not 0.0 <= uniform_mix <= 1.0:
        raise ValueError(f"uniform_mix must be in [0,1], got {uniform_mix}")
    if min_uncertainty <= 0.0:
        raise ValueError(f"min_uncertainty must be > 0, got {min_uncertainty}")

    bank: list[Team] = []
    seen: set[Team] = set()
    attempts = 0
    max_attempts = max(1000, candidate_pool_size * 20)
    while len(bank) < candidate_pool_size and attempts < max_attempts:
        attempts += 1
        team = sample_team(
            pool_ids,
            team_size=team_size,
            replace=replace,
            rng=rng,
            species_clause_keys=species_clause_keys,
        )
        if team in seen:
            continue
        bank.append(team)
        seen.add(team)

    if len(bank) < 2:
        return make_uniform_matchup_sampler(
            pool_ids,
            team_size=team_size,
            replace=replace,
            rng=rng,
            species_clause_keys=species_clause_keys,
        )

    pairs: list[tuple[Team, Team]] = []
    weights: list[float] = []
    for i in range(len(bank)):
        for j in range(i + 1, len(bank)):
            a = bank[i]
            b = bank[j]
            p = float(pair_evaluator(a, b))
            if not (p == p) or p < 0.0 or p > 1.0:
                p = 0.5
            p = min(max(p, 1e-9), 1.0 - 1e-9)
            uncertainty = 4.0 * p * (1.0 - p)
            pairs.append((a, b))
            weights.append(max(min_uncertainty, uncertainty))

    if not pairs:
        return make_uniform_matchup_sampler(
            pool_ids,
            team_size=team_size,
            replace=replace,
            rng=rng,
            species_clause_keys=species_clause_keys,
        )

    idxs = list(range(len(pairs)))

    def _sample() -> tuple[Team, Team]:
        if rng.random() < uniform_mix:
            return (
                sample_team(
                    pool_ids,
                    team_size=team_size,
                    replace=replace,
                    rng=rng,
                    species_clause_keys=species_clause_keys,
                ),
                sample_team(
                    pool_ids,
                    team_size=team_size,
                    replace=replace,
                    rng=rng,
                    species_clause_keys=species_clause_keys,
                ),
            )
        chosen = rng.choices(idxs, weights=weights, k=1)[0]
        a, b = pairs[chosen]
        return (a, b) if rng.random() < 0.5 else (b, a)

    return _sample


def _synthetic_outcome(team_a: Team, team_b: Team, seed: int) -> int:
    """Deterministic fallback simulator useful for CI and correctness tests."""

    key = f"{seed}|{','.join(map(str, team_a))}|{','.join(map(str, team_b))}".encode(
        "utf-8"
    )
    digest = hashlib.sha256(key).digest()
    draw = int.from_bytes(digest[:8], byteorder="big", signed=False)
    rng = random.Random(draw)
    return 1 if rng.random() < 0.5 else 0


def _run_single_poke_env_battle(
    *,
    team_a_showdown: str,
    team_b_showdown: str,
    format_id: str,
    timeout_sec: float,
) -> int:
    """Play one battle between two identical heuristic policies; return y in {0,1}."""

    from poke_env import AccountConfiguration, LocalhostServerConfiguration
    from poke_env.player.baselines import SimpleHeuristicsPlayer

    async def _main() -> int:
        username_a = f"tcA-{uuid.uuid4().hex[:12]}"
        username_b = f"tcB-{uuid.uuid4().hex[:12]}"
        player_a = SimpleHeuristicsPlayer(
            battle_format=format_id,
            team=team_a_showdown,
            account_configuration=AccountConfiguration(username_a, None),
            server_configuration=LocalhostServerConfiguration,
            max_concurrent_battles=1,
        )
        player_b = SimpleHeuristicsPlayer(
            battle_format=format_id,
            team=team_b_showdown,
            account_configuration=AccountConfiguration(username_b, None),
            server_configuration=LocalhostServerConfiguration,
            max_concurrent_battles=1,
        )

        await player_a.battle_against(player_b, n_battles=1)
        return 1 if int(getattr(player_a, "n_won_battles", 0)) > 0 else 0

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return int(
            loop.run_until_complete(asyncio.wait_for(_main(), timeout=timeout_sec))
        )
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        asyncio.set_event_loop(None)
        loop.close()


def _run_single_metamon_battle(
    *,
    team_a_showdown: str,
    team_b_showdown: str,
    format_id: str,
    model_name: str,
    seed: int,
    gpu_a: int,
    gpu_b: int,
    work_dir: Path | None,
    checkpoint: int | None,
    print_match_stats: bool,
) -> int:
    """Play one battle using metamon pretrained policies; return y in {0,1}.

    Team A is assigned to the acceptor role, so y=1 means Team A wins.
    """

    from .matchup import run_matchup

    cache_dir_env = os.environ.get("METAMON_CACHE_DIR")
    if not cache_dir_env:
        raise ValueError(
            "METAMON_CACHE_DIR must be set for backend='metamon' simulation."
        )
    cache_dir = Path(cache_dir_env)
    teams_root = cache_dir / "teams"

    run_id = uuid.uuid4().hex[:10]
    team_set_a = f"team_construction_sim_a_{run_id}"
    team_set_b = f"team_construction_sim_b_{run_id}"
    team_dir_a = teams_root / team_set_a / format_id
    team_dir_b = teams_root / team_set_b / format_id

    team_dir_a.mkdir(parents=True, exist_ok=True)
    team_dir_b.mkdir(parents=True, exist_ok=True)
    (team_dir_a / f"team_0001.{format_id}_team").write_text(
        team_a_showdown.strip() + "\n",
        encoding="utf-8",
    )
    (team_dir_b / f"team_0001.{format_id}_team").write_text(
        team_b_showdown.strip() + "\n",
        encoding="utf-8",
    )

    try:
        result = run_matchup(
            battle_format=format_id,
            num_battles=1,
            model_name=model_name,
            team_set_a=team_set_a,
            team_set_b=team_set_b,
            gpu_a=gpu_a,
            gpu_b=gpu_b,
            work_dir=(
                work_dir or Path("/tmp/team_prediction/team_construction_battles")
            ),
            checkpoint=checkpoint,
            print_match_stats=print_match_stats,
        )

        acceptor_matches = result.get("acceptor_matches", [])
        if acceptor_matches:
            outcome = str(acceptor_matches[0].get("result", "")).strip().upper()
            if outcome == "WIN":
                return 1
            if outcome == "LOSS":
                return 0
            if outcome == "DRAW":
                # Preserve binary labels while avoiding systematic draw bias.
                return 1 if random.Random(seed).random() < 0.5 else 0

        wr = float(result["acceptor_summary"]["win_rate"])
        if wr > 0.5:
            return 1
        if wr < 0.5:
            return 0
        return 1 if random.Random(seed + 17).random() < 0.5 else 0
    finally:
        shutil.rmtree(team_dir_a.parent, ignore_errors=True)
        shutil.rmtree(team_dir_b.parent, ignore_errors=True)


def _result_to_binary(result_raw: str, *, seed: int) -> int:
    outcome = str(result_raw).strip().upper()
    if outcome == "WIN":
        return 1
    if outcome == "LOSS":
        return 0
    if outcome == "DRAW":
        return 1 if random.Random(seed).random() < 0.5 else 0
    raise ValueError(f"Unknown matchup result '{result_raw}'")


def _simulate_metamon_batch(
    *,
    pairs: Sequence[tuple[Team, Team]],
    team_to_showdown: Callable[[Team], str],
    format_id: str,
    model_name: str,
    seed: int,
    gpu_a: int,
    gpu_b: int,
    work_dir: Path | None,
    checkpoint: int | None,
    print_match_stats: bool,
    max_retries: int,
    retry_sleep_sec: float,
) -> list[BattleExample]:
    """Run all metamon battles in one matchup call to avoid per-battle agent reinit."""

    from .matchup import run_matchup

    if not pairs:
        return []

    cache_dir_env = os.environ.get("METAMON_CACHE_DIR")
    if not cache_dir_env:
        raise ValueError(
            "METAMON_CACHE_DIR must be set for backend='metamon' simulation."
        )
    cache_dir = Path(cache_dir_env)
    teams_root = cache_dir / "teams"

    run_id = uuid.uuid4().hex[:10]
    team_set_a = f"team_construction_sim_batch_a_{run_id}"
    team_set_b = f"team_construction_sim_batch_b_{run_id}"
    team_dir_a = teams_root / team_set_a / format_id
    team_dir_b = teams_root / team_set_b / format_id
    team_dir_a.mkdir(parents=True, exist_ok=True)
    team_dir_b.mkdir(parents=True, exist_ok=True)

    def _write_team_bank(
        team_dir: Path,
        teams: Sequence[Team],
    ) -> tuple[dict[str, Team], dict[str, Team]]:
        by_path: dict[str, Team] = {}
        by_name: dict[str, Team] = {}
        for idx, team in enumerate(teams, start=1):
            team_file = team_dir / f"team_{idx:04d}.{format_id}_team"
            team_file.write_text(
                team_to_showdown(team).strip() + "\n", encoding="utf-8"
            )
            resolved = str(team_file.resolve())
            by_path[resolved] = team
            by_name[team_file.name] = team
        return by_path, by_name

    try:
        sampled_a = [a for a, _ in pairs]
        sampled_b = [b for _, b in pairs]
        a_by_path, a_by_name = _write_team_bank(team_dir_a, sampled_a)
        b_by_path, b_by_name = _write_team_bank(team_dir_b, sampled_b)

        last_error: Exception | None = None
        result: dict | None = None
        for attempt in range(max_retries + 1):
            try:
                result = run_matchup(
                    battle_format=format_id,
                    num_battles=len(pairs),
                    model_name=model_name,
                    team_set_a=team_set_a,
                    team_set_b=team_set_b,
                    gpu_a=gpu_a,
                    gpu_b=gpu_b,
                    work_dir=(
                        work_dir
                        or Path("/tmp/team_prediction/team_construction_battles")
                    ),
                    checkpoint=checkpoint,
                    print_match_stats=print_match_stats,
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt >= max_retries:
                    raise RuntimeError(
                        f"batch metamon simulation failed after retries: {last_error}"
                    ) from last_error
                time.sleep(retry_sleep_sec)

        assert result is not None
        acceptor_matches = result.get("acceptor_matches", [])
        challenger_matches = result.get("challenger_matches", [])
        if not acceptor_matches or not challenger_matches:
            raise RuntimeError("metamon batch simulation produced empty match logs.")

        n = min(len(pairs), len(acceptor_matches), len(challenger_matches))
        if n <= 0:
            raise RuntimeError("metamon batch simulation produced zero usable matches.")

        examples: list[BattleExample] = []
        for idx in range(n):
            a_row = acceptor_matches[idx]
            b_row = challenger_matches[idx]

            a_path_raw = str(a_row.get("team_file_path", ""))
            b_path_raw = str(b_row.get("team_file_path", ""))
            a_path = str(Path(a_path_raw).resolve()) if a_path_raw else ""
            b_path = str(Path(b_path_raw).resolve()) if b_path_raw else ""

            team_a = a_by_path.get(a_path)
            team_b = b_by_path.get(b_path)
            if team_a is None and a_path_raw:
                team_a = a_by_name.get(Path(a_path_raw).name)
            if team_b is None and b_path_raw:
                team_b = b_by_name.get(Path(b_path_raw).name)
            if team_a is None or team_b is None:
                raise RuntimeError(
                    f"Could not map team files back to sampled teams: "
                    f"a='{a_path_raw}', b='{b_path_raw}'"
                )

            y = _result_to_binary(str(a_row.get("result", "")), seed=seed + idx)
            examples.append(BattleExample(team_A=team_a, team_B=team_b, y=int(y)))

        if n < len(pairs):
            raise RuntimeError(
                f"metamon batch simulation returned fewer matches ({n}) "
                f"than requested ({len(pairs)})."
            )
        return examples
    finally:
        shutil.rmtree(team_dir_a.parent, ignore_errors=True)
        shutil.rmtree(team_dir_b.parent, ignore_errors=True)


def _simulate_one(
    *,
    team_a: Team,
    team_b: Team,
    seed: int,
    backend: str,
    format_id: str,
    team_to_showdown: Callable[[Team], str] | None,
    timeout_sec: float,
    max_retries: int,
    retry_sleep_sec: float,
    metamon_model_name: str | None = None,
    metamon_checkpoint: int | None = None,
    metamon_gpu_a: int = 0,
    metamon_gpu_b: int = 1,
    metamon_work_dir: Path | None = None,
    metamon_print_match_stats: bool = False,
) -> BattleExample:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            if backend == "synthetic":
                y = _synthetic_outcome(team_a, team_b, seed=seed)
            elif backend == "poke_env":
                if team_to_showdown is None:
                    raise ValueError(
                        "team_to_showdown is required for poke_env simulation"
                    )
                y = _run_single_poke_env_battle(
                    team_a_showdown=team_to_showdown(team_a),
                    team_b_showdown=team_to_showdown(team_b),
                    format_id=format_id,
                    timeout_sec=timeout_sec,
                )
            elif backend == "metamon":
                if team_to_showdown is None:
                    raise ValueError(
                        "team_to_showdown is required for metamon simulation"
                    )
                if not metamon_model_name:
                    raise ValueError(
                        "metamon_model_name is required for backend='metamon'"
                    )
                y = _run_single_metamon_battle(
                    team_a_showdown=team_to_showdown(team_a),
                    team_b_showdown=team_to_showdown(team_b),
                    format_id=format_id,
                    model_name=metamon_model_name,
                    seed=seed,
                    gpu_a=metamon_gpu_a,
                    gpu_b=metamon_gpu_b,
                    work_dir=metamon_work_dir,
                    checkpoint=metamon_checkpoint,
                    print_match_stats=metamon_print_match_stats,
                )
            else:
                raise ValueError(f"Unknown simulation backend '{backend}'")
            return BattleExample(team_A=team_a, team_B=team_b, y=int(y))
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(retry_sleep_sec)

    raise RuntimeError(
        f"simulate battle failed after retries for teams {team_a} vs {team_b}: {last_error}"
    )


def simulate_battles(
    n: int,
    sampler: Callable[[], tuple[Team, Team]],
    agent_class: str,
    format_id: str,
    concurrency: int,
    *,
    seed: int = 0,
    backend: str = "poke_env",
    team_to_showdown: Callable[[Team], str] | None = None,
    timeout_sec: float = 240.0,
    max_retries: int = 2,
    retry_sleep_sec: float = 2.0,
    metamon_model_name: str | None = None,
    metamon_checkpoint: int | None = None,
    metamon_gpu_a: int = 0,
    metamon_gpu_b: int = 1,
    metamon_work_dir: Path | None = None,
    metamon_print_match_stats: bool = False,
    incremental_out: Path | None = None,
    incremental_flush_every: int = 50,
    show_progress: bool = True,
    progress_desc: str | None = None,
) -> list[BattleExample]:
    """Run team-vs-team simulations and return supervised battle examples.

    backends:
      - synthetic: deterministic CI/debug fallback
      - poke_env: SimpleHeuristicsPlayer vs SimpleHeuristicsPlayer
      - metamon: pretrained policy (e.g., Kakuna) vs itself via run_matchup
    """

    if n <= 0:
        return []
    if concurrency <= 0:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")
    if backend == "poke_env":
        normalized = agent_class.strip().lower()
        valid = {
            "simpleheuristicsplayer",
            "simple_heuristics_player",
            "simpleheuristics",
        }
        if normalized not in valid:
            raise ValueError(
                "Only SimpleHeuristicsPlayer is currently supported for backend='poke_env'. "
                f"Got agent_class='{agent_class}'."
            )
    elif backend == "metamon":
        if not metamon_model_name:
            raise ValueError(
                "metamon_model_name is required for backend='metamon'. "
                "Pass a pretrained model name like 'Kakuna'."
            )
        if team_to_showdown is None:
            raise ValueError("team_to_showdown is required for backend='metamon'")

    pairs = [sampler() for _ in range(n)]

    if backend == "metamon":
        examples = _simulate_metamon_batch(
            pairs=pairs,
            team_to_showdown=team_to_showdown,
            format_id=format_id,
            model_name=metamon_model_name,
            seed=seed,
            gpu_a=metamon_gpu_a,
            gpu_b=metamon_gpu_b,
            work_dir=metamon_work_dir,
            checkpoint=metamon_checkpoint,
            print_match_stats=metamon_print_match_stats,
            max_retries=max_retries,
            retry_sleep_sec=retry_sleep_sec,
        )
        if incremental_out is not None:
            incremental_out.parent.mkdir(parents=True, exist_ok=True)
            with incremental_out.open("w", encoding="utf-8") as out_handle:
                for idx, example in enumerate(examples, start=1):
                    out_handle.write(
                        json.dumps(battle_example_to_json_dict(example)) + "\n"
                    )
                    if idx % max(1, incremental_flush_every) == 0:
                        out_handle.flush()
        return examples

    out_handle = None
    if incremental_out is not None:
        incremental_out.parent.mkdir(parents=True, exist_ok=True)
        out_handle = incremental_out.open("w", encoding="utf-8")

    examples: list[BattleExample] = []

    def _job(payload: tuple[int, Team, Team]) -> BattleExample:
        idx, team_a, team_b = payload
        return _simulate_one(
            team_a=team_a,
            team_b=team_b,
            seed=seed + idx,
            backend=backend,
            format_id=format_id,
            team_to_showdown=team_to_showdown,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            retry_sleep_sec=retry_sleep_sec,
            metamon_model_name=metamon_model_name,
            metamon_checkpoint=metamon_checkpoint,
            metamon_gpu_a=metamon_gpu_a,
            metamon_gpu_b=metamon_gpu_b,
            metamon_work_dir=metamon_work_dir,
            metamon_print_match_stats=metamon_print_match_stats,
        )

    jobs = [(idx, pair[0], pair[1]) for idx, pair in enumerate(pairs)]

    executor = None
    try:
        if concurrency == 1:
            iterator = map(_job, jobs)
        else:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)
            iterator = executor.map(_job, jobs)

        with tqdm(
            total=len(jobs),
            desc=progress_desc or f"Simulating {n} battles",
            unit="battle",
            dynamic_ncols=True,
            disable=not show_progress,
        ) as pbar:
            for idx, example in enumerate(iterator, start=1):
                examples.append(example)
                if out_handle is not None:
                    out_handle.write(
                        json.dumps(battle_example_to_json_dict(example)) + "\n"
                    )
                    if idx % max(1, incremental_flush_every) == 0:
                        out_handle.flush()
                pbar.update(1)
    finally:
        if out_handle is not None:
            out_handle.flush()
            out_handle.close()
        if executor is not None:
            executor.shutdown(wait=True)

    return examples


def augment_swap_symmetry(examples: Sequence[BattleExample]) -> list[BattleExample]:
    """Add swapped team/order examples (x, y) -> (swap(x), 1-y)."""

    out: list[BattleExample] = []
    for example in examples:
        out.append(example)
        out.append(example.swapped())
    return out


def split_before_augmentation(
    examples: Sequence[BattleExample],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[BattleExample], list[BattleExample]]:
    """Split originals first so augmented pairs stay in the same split."""

    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    n_val = int(round(len(indices) * val_fraction))
    val_idx = set(indices[:n_val])

    train: list[BattleExample] = []
    val: list[BattleExample] = []
    for idx, example in enumerate(examples):
        if idx in val_idx:
            val.append(example)
        else:
            train.append(example)
    return train, val


def save_examples_jsonl(path: Path, examples: Iterable[BattleExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(battle_example_to_json_dict(ex)) + "\n")


def load_examples_jsonl(path: Path) -> list[BattleExample]:
    out: list[BattleExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(battle_example_from_json_dict(json.loads(line)))
    return out


def save_simulation_metadata(path: Path, metadata: SimulationMetadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(metadata), indent=2, sort_keys=True), encoding="utf-8"
    )
