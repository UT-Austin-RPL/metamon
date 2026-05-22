import argparse
import atexit
import json
import os
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
import random
import shutil
import subprocess
from typing import List, Optional
import tqdm

from poke_env.teambuilder import ConstantTeambuilder

from metamon.backend.team_prediction.team import TeamSet
from metamon.backend.team_prediction.team_index import refresh_team_index
from metamon.env import BattleAgainstBaseline
from metamon.baselines.heuristic.basic import RandomBaseline
from metamon.interface import (
    TokenizedObservationSpace,
    DefaultObservationSpace,
    DefaultShapedReward,
    MinimalActionSpace,
)
from metamon.tokenizer import get_tokenizer

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PERSISTENT_VALIDATOR = None
_PERSISTENT_VALIDATOR_DISABLED = False
_SHOWDOWN_PROCESS_MARKERS = (
    "dist/server/sockets.js",
    "dist/server/room-battle.js",
    "pokemon-showdown",
)


def _valid_showdown_dist(dist: Path) -> bool:
    return (dist / "sim" / "team-validator.js").exists()


def _dist_candidates(root: Path) -> List[Path]:
    return [root / "dist", root / "pokemon-showdown" / "dist"]


def _showdown_roots_from_path(path: Path) -> List[Path]:
    roots: List[Path] = []
    seen: set[Path] = set()
    current = path.resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if candidate.name == "dist" and _valid_showdown_dist(candidate):
            root = candidate.parent
        else:
            root = candidate
        for dist in _dist_candidates(root):
            if _valid_showdown_dist(dist) and root not in seen:
                seen.add(root)
                roots.append(root)
                break
        if roots:
            break
    return roots


def _showdown_roots_from_running_processes() -> List[Path]:
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return []

    roots: List[Path] = []
    seen: set[Path] = set()

    def add_root(root: Path) -> None:
        root = root.resolve()
        if root in seen:
            return
        for dist in _dist_candidates(root):
            if _valid_showdown_dist(dist):
                seen.add(root)
                roots.append(root)
                return

    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        cmdline_file = entry / "cmdline"
        try:
            raw = cmdline_file.read_bytes()
        except OSError:
            continue
        if not raw:
            continue
        cmdline = raw.replace(b"\0", b" ").decode("utf-8", errors="ignore")
        if not any(marker in cmdline for marker in _SHOWDOWN_PROCESS_MARKERS):
            continue

        for part in raw.split(b"\0"):
            if not part:
                continue
            text = part.decode("utf-8", errors="ignore")
            if "dist/server/" not in text:
                continue
            path = Path(text)
            if "dist" not in path.parts:
                continue
            root = Path(*path.parts[: path.parts.index("dist")])
            add_root(root)

        try:
            add_root((entry / "cwd").resolve())
        except OSError:
            pass

    return roots


def _candidate_showdown_roots(repo_root: Path) -> List[Path]:
    seen: set[Path] = set()
    roots: List[Path] = []

    def add(path: Path) -> None:
        for root in _showdown_roots_from_path(path):
            resolved = root.resolve()
            if resolved not in seen:
                seen.add(resolved)
                roots.append(resolved)

    extra = os.environ.get("SHOWDOWN_ROOT")
    if extra:
        add(Path(extra))

    for root in _showdown_roots_from_running_processes():
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            roots.append(resolved)

    showdown_bin = shutil.which("pokemon-showdown")
    if showdown_bin:
        add(Path(showdown_bin))

    bundled = repo_root / "server" / "pokemon-showdown"
    if bundled.exists():
        add(bundled)

    return roots


def _candidate_node_cwds(repo_root: Path) -> List[Path]:
    return _candidate_showdown_roots(repo_root)


def _showdown_node_paths(repo_root: Path) -> List[str]:
    """NODE_PATH entries so tools/persistent_showdown_validator.js can require pokemon-showdown."""
    seen: set[str] = set()
    out: List[str] = []
    for cwd in _candidate_node_cwds(repo_root):
        for base in (cwd, cwd / "pokemon-showdown"):
            node_modules = base / "node_modules"
            if (node_modules / "pokemon-showdown").exists():
                path = str(node_modules.resolve())
                if path not in seen:
                    seen.add(path)
                    out.append(path)
    return out


def _resolve_node_cwd(repo_root: Path) -> Path:
    for cwd in _candidate_node_cwds(repo_root):
        for base in (cwd, cwd / "pokemon-showdown"):
            if (base / "node_modules" / "pokemon-showdown").exists():
                return base
            if (base / "node_modules" / ".bin" / "pokemon-showdown").exists():
                return base
    return repo_root


def _resolve_showdown_dist(repo_root: Path) -> Optional[Path]:
    explicit = os.environ.get("SHOWDOWN_DIST")
    if explicit:
        dist = Path(explicit)
        if (dist / "sim" / "team-validator.js").exists():
            return dist.resolve()
        return None
    for root in _candidate_showdown_roots(repo_root):
        for dist in _dist_candidates(root):
            if (dist / "sim" / "team-validator.js").exists():
                return dist.resolve()
    return None


def _validator_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    showdown_dist = _resolve_showdown_dist(repo_root)
    if showdown_dist is not None:
        env["SHOWDOWN_DIST"] = str(showdown_dist)
    else:
        node_paths = _showdown_node_paths(repo_root)
        if node_paths:
            existing = env.get("NODE_PATH", "")
            combined = os.pathsep.join(node_paths + ([existing] if existing else []))
            env["NODE_PATH"] = combined
    return env


def _find_showdown_bin(repo_root: Path) -> Optional[str]:
    showdown_bin = shutil.which("pokemon-showdown")
    if showdown_bin:
        return showdown_bin
    for cwd in _candidate_node_cwds(repo_root):
        for base in (cwd, cwd / "pokemon-showdown"):
            bin_path = base / "node_modules" / ".bin" / "pokemon-showdown"
            if bin_path.exists():
                return str(bin_path)
    return None


def _resolve_showdown_validate_cmd(
    format_id: str, cmd: Optional[List[str]]
) -> List[str]:
    if cmd is not None:
        return cmd + [format_id]
    showdown_bin = _find_showdown_bin(_REPO_ROOT)
    if showdown_bin:
        return [showdown_bin, "validate-team", format_id]
    return ["npx", "pokemon-showdown", "validate-team", format_id]


class PersistentShowdownValidator:
    def __init__(self, repo_root: Path):
        self._repo_root = repo_root
        self._script_path = repo_root / "tools" / "persistent_showdown_validator.js"
        if not self._script_path.exists():
            raise FileNotFoundError(f"Missing validator script at {self._script_path}")
        self._cwd = _resolve_node_cwd(repo_root)
        self._proc = self._start_process()
        if not self._ping():
            self.close()
            raise RuntimeError("Persistent validator failed to start")

    def _start_process(self) -> subprocess.Popen:
        return subprocess.Popen(
            ["node", str(self._script_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=str(self._cwd),
            env=_validator_env(self._repo_root),
            bufsize=1,
        )

    def _send(self, payload: dict) -> Optional[dict]:
        if self._proc.poll() is not None:
            return None
        if self._proc.stdin is None or self._proc.stdout is None:
            return None
        try:
            self._proc.stdin.write(json.dumps(payload) + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError:
            return None
        line = self._proc.stdout.readline()
        if not line:
            return None
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    def _ping(self) -> bool:
        response = self._send({"format": "gen1ou", "team": ""})
        return response is not None

    def validate(self, team_str: str, format_id: str) -> tuple[bool, List[str]]:
        response = self._send({"format": format_id, "team": team_str})
        if response is None:
            raise RuntimeError("Validator process is not responding")
        ok = bool(response.get("ok"))
        errors = response.get("errors") or []
        return ok, [str(err) for err in errors]

    def close(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            try:
                if self._proc.stdin is not None:
                    self._proc.stdin.close()
                self._proc.terminate()
                self._proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None


_POOL_VALIDATOR: Optional["PersistentShowdownValidator"] = None


@dataclass
class ValidateFileResult:
    file: str
    passed: bool
    errors: Optional[List[str]] = None
    parse_error: Optional[str] = None


def _init_validate_pool() -> None:
    global _POOL_VALIDATOR
    _POOL_VALIDATOR = PersistentShowdownValidator(_REPO_ROOT)


def _validate_file(
    filename: str,
    input_dir: str,
    output_dir: str,
    format_id: str,
    validator: PersistentShowdownValidator,
    write_output: bool,
) -> ValidateFileResult:
    filepath = os.path.join(input_dir, filename)
    try:
        team = TeamSet.from_showdown_file(filepath, format=format_id)
        team_str = team.to_str()
    except Exception as exc:
        return ValidateFileResult(file=filename, passed=False, parse_error=str(exc))

    try:
        ok, errors = validator.validate(team_str, format_id)
    except Exception as exc:
        return ValidateFileResult(file=filename, passed=False, errors=[str(exc)])

    if not ok:
        return ValidateFileResult(file=filename, passed=False, errors=errors)

    if write_output:
        out_dir = os.path.join(output_dir, format_id)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, filename), "w") as f:
            f.write(team_str)

    return ValidateFileResult(file=filename, passed=True)


def _validate_file_pool(args: tuple[str, str, str, str]) -> ValidateFileResult:
    if _POOL_VALIDATOR is None:
        raise RuntimeError("Validator pool was not initialized")
    filename, input_dir, output_dir, format_id = args
    return _validate_file(
        filename,
        input_dir,
        output_dir,
        format_id,
        _POOL_VALIDATOR,
        write_output=True,
    )


def _list_team_files(input_dir: str) -> List[str]:
    """List team files once each, in deterministic order before optional shuffle."""
    seen: set[str] = set()
    files: List[str] = []
    for name in sorted(os.listdir(input_dir)):
        if not name.endswith("team"):
            continue
        if name in seen:
            raise RuntimeError(f"Duplicate team filename in {input_dir}: {name}")
        seen.add(name)
        files.append(name)
    return files


def _default_workers() -> int:
    return max(1, min(16, cpu_count() - 2))


def _validate_directory(
    format_id: str,
    input_path: str,
    output_path: str,
    workers: int,
    print_errors: bool = True,
) -> tuple[int, int, int]:
    input_dir = os.path.join(input_path, format_id)
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = _list_team_files(input_dir)
    random.shuffle(files)
    # One tuple per file; pool.imap_unordered dispatches each item to exactly one worker.
    work = [(f, input_dir, output_path, format_id) for f in files]

    passed = 0
    rejected = 0
    parse_errors = 0
    seen_results: set[str] = set()

    def handle_result(result: ValidateFileResult) -> None:
        nonlocal passed, rejected, parse_errors
        if result.file in seen_results:
            raise RuntimeError(f"Processed {result.file} more than once")
        seen_results.add(result.file)
        if result.passed:
            passed += 1
            return
        if result.parse_error is not None:
            parse_errors += 1
            if print_errors:
                print(result.parse_error)
        else:
            rejected += 1
            if print_errors and result.errors:
                print(result.errors)

    if workers <= 1:
        validator = PersistentShowdownValidator(_REPO_ROOT)
        try:
            for filename in tqdm.tqdm(files):
                result = _validate_file(
                    filename,
                    input_dir,
                    output_path,
                    format_id,
                    validator,
                    write_output=True,
                )
                handle_result(result)
        finally:
            validator.close()
    else:
        chunksize = max(1, len(work) // (workers * 8))
        with Pool(workers, initializer=_init_validate_pool) as pool:
            for result in tqdm.tqdm(
                pool.imap_unordered(_validate_file_pool, work, chunksize=chunksize),
                total=len(work),
            ):
                handle_result(result)

    if len(seen_results) != len(files):
        raise RuntimeError(
            f"Expected {len(files)} team results, got {len(seen_results)}"
        )
    if passed + rejected + parse_errors != len(files):
        raise RuntimeError(
            f"Result counts do not match input files: "
            f"{passed + rejected + parse_errors} vs {len(files)}"
        )

    output_format_dir = os.path.join(output_path, format_id)
    if passed:
        index_path, index_count = refresh_team_index(output_format_dir, format_id)
        print(f"Wrote {index_path} ({index_count:,} teams)")

    return passed, rejected, parse_errors


def _get_persistent_validator() -> Optional[PersistentShowdownValidator]:
    global _PERSISTENT_VALIDATOR, _PERSISTENT_VALIDATOR_DISABLED
    if _PERSISTENT_VALIDATOR_DISABLED:
        return None
    if _PERSISTENT_VALIDATOR is None:
        try:
            _PERSISTENT_VALIDATOR = PersistentShowdownValidator(_REPO_ROOT)
            atexit.register(_PERSISTENT_VALIDATOR.close)
        except Exception as exc:  # pragma: no cover - best-effort optimization
            print(f"Persistent validator unavailable, falling back to CLI: {exc}")
            _PERSISTENT_VALIDATOR_DISABLED = True
            return None
    return _PERSISTENT_VALIDATOR


def validate_showdown_team(
    team_str: str,
    format_id: str = "gen1ou",
    cmd: Optional[List[str]] = None,
) -> bool:
    validator = _get_persistent_validator()
    if validator is not None:
        global _PERSISTENT_VALIDATOR_DISABLED
        try:
            ok, errors = validator.validate(team_str, format_id)
        except Exception as exc:  # pragma: no cover - best-effort optimization
            validator.close()
            _PERSISTENT_VALIDATOR_DISABLED = True
            print(f"Persistent validator failed, falling back to CLI: {exc}")
        else:
            if ok:
                return True
            print(errors)
            return False

    full_cmd = _resolve_showdown_validate_cmd(format_id, cmd)

    proc = subprocess.run(full_cmd, input=team_str, text=True, capture_output=True)

    if proc.returncode == 0:
        return True
    else:
        output = proc.stdout.strip().splitlines() + proc.stderr.strip().splitlines()
        print(output)
        return False


def env_verify_team(team_str: str, format_id: str = "gen1ou") -> bool:
    team_set = ConstantTeambuilder(team_str)
    obs_space = TokenizedObservationSpace(
        base_obs_space=DefaultObservationSpace(),
        tokenizer=get_tokenizer("DefaultObservationSpace-v0"),
    )
    reward_fn = DefaultShapedReward()
    env = BattleAgainstBaseline(
        battle_format=format_id,
        team_set=team_set,
        opponent_type=RandomBaseline,
        observation_space=obs_space,
        action_space=MinimalActionSpace(),
        reward_function=reward_fn,
    )
    env._INIT_RETRIES = 2
    env._TIME_BETWEEN_RETRIES = 0.05
    try:
        env.reset()
        env.step(env.action_space.sample())
    except Exception as e:
        del env
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate and rewrite Pokemon Showdown teams."
    )
    parser.add_argument(
        "format", type=str, help="The format to process (e.g. gen1ou, gen4uu)"
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to input directory containing team files",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to output directory for verified teams",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Parallel workers (each runs its own Showdown validator process). "
            f"Suggested: {_default_workers()} on this machine."
        ),
    )

    args = parser.parse_args()
    print(f"Processing format: {args.format} (workers={args.workers})")

    passed, rejected, parse_errors = _validate_directory(
        format_id=args.format,
        input_path=args.input_path,
        output_path=args.output_path,
        workers=args.workers,
    )
    total = passed + rejected + parse_errors
    if total:
        print(
            f"Done: {passed:,}/{total:,} passed "
            f"({passed / total * 100:.1f}%), "
            f"{rejected:,} invalid, {parse_errors:,} parse errors"
        )
    else:
        print("Done: no team files found")
