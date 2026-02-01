import argparse
import atexit
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
from typing import List, Optional
import tqdm

from poke_env.teambuilder import ConstantTeambuilder

from metamon.backend.team_prediction.team import TeamSet
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


def _candidate_node_cwds(repo_root: Path) -> List[Path]:
    return [repo_root, repo_root / "server" / "pokemon-showdown"]


def _resolve_node_cwd(repo_root: Path) -> Path:
    for cwd in _candidate_node_cwds(repo_root):
        if (cwd / "node_modules" / "pokemon-showdown").exists():
            return cwd
        if (cwd / "node_modules" / ".bin" / "pokemon-showdown").exists():
            return cwd
    return repo_root


def _find_showdown_bin(repo_root: Path) -> Optional[str]:
    local_bins = [
        repo_root / "node_modules" / ".bin" / "pokemon-showdown",
        repo_root
        / "server"
        / "pokemon-showdown"
        / "node_modules"
        / ".bin"
        / "pokemon-showdown",
    ]
    for bin_path in local_bins:
        if bin_path.exists():
            return str(bin_path)
    return shutil.which("pokemon-showdown")


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

    args = parser.parse_args()
    print(f"Processing format: {args.format}")

    path = os.path.join(args.input_path, args.format)
    if os.path.isdir(path):
        files = os.listdir(path)
        random.shuffle(files)
        for file in tqdm.tqdm(files):
            if file.endswith("team"):
                filename = os.path.join(path, file)
                format = path.split("/")[-1]
                try:
                    team = TeamSet.from_showdown_file(filename, format=format)
                    team_str = team.to_str()
                except Exception as e:
                    print(e)
                    continue

                if not validate_showdown_team(team_str, format):
                    continue
                # if not env_verify_team(team_str, format):
                #    continue

                os.makedirs(os.path.join(args.output_path, format), exist_ok=True)
                with open(os.path.join(args.output_path, format, file), "w") as f:
                    f.write(team_str)
