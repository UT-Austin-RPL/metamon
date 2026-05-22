"""
Shared utilities for auto-evaluation launchers (h2h, sweep, ladder_self_play).

Provides:
    - PolicySpec / MatchupSpec: data classes for describing policies and matchups
    - GPU distribution
    - Subprocess management for running matchup workers
    - Config parsing helpers
"""

import gc
import hashlib
import itertools
import os
import random
import re
import subprocess
import time
import yaml
from argparse import ArgumentParser
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


@dataclass
class PolicySpec:
    """A fully-specified policy configuration for evaluation.

    Attributes:
        name: Display name (used in win matrix labels, usernames, etc.)
        model_name: Pretrained model identifier (must match a key in pretrained.py).
        checkpoint: Checkpoint epoch to load (None = model default).
        temperature: Action sampling temperature.
        team_set: Team set name.
        battle_backend: Showdown state-parsing backend.
    """

    name: str
    model_name: str
    checkpoint: Optional[int]
    temperature: float
    team_set: str
    battle_backend: str

    @property
    def short_label(self) -> str:
        """Compact label for display (e.g. matrix headers)."""
        parts = [self.name]
        if self.checkpoint is not None:
            parts.append(f"ckpt{self.checkpoint}")
        if self.temperature != 1.0:
            parts.append(f"t{self.temperature}")
        if self.team_set:
            parts.append(self.team_set)
        return "-".join(parts)

    @property
    def unique_key(self) -> str:
        """Deterministic key that uniquely identifies this policy configuration."""
        return f"{self.model_name}_ckpt{self.checkpoint}_t{self.temperature}_{self.team_set}_{self.battle_backend}"


@dataclass
class MatchupSpec:
    """A single head-to-head matchup to run.

    Attributes:
        policy_a: The first policy (will be the challenger).
        policy_b: The second policy (will be the acceptor).
        n_battles: Number of battles to play.
        battle_format: Pokemon Showdown battle format (e.g. "gen1ou").
        matchup_id: Deterministic unique identifier for crash recovery.
    """

    policy_a: PolicySpec
    policy_b: PolicySpec
    n_battles: int
    battle_format: str
    matchup_id: str = ""

    def __post_init__(self):
        if not self.matchup_id:
            self.matchup_id = (
                f"{self.policy_a.unique_key}__vs__{self.policy_b.unique_key}"
            )


# ---------------------------------------------------------------------------
# Config templating — ${var} and ${var:default} placeholders
# ---------------------------------------------------------------------------

_TEMPLATE_RE = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")


def discover_template_vars(config_path: str) -> Dict[str, Optional[str]]:
    """Scan a config file for ``${var}`` and ``${var:default}`` placeholders.

    Only non-comment lines are inspected (lines whose first non-whitespace
    character is ``#`` are skipped).

    Returns:
        Ordered dict mapping variable names to their default value
        (``None`` if no default was specified → the variable is required).
    """
    with open(config_path, "r") as f:
        lines = f.readlines()
    found: Dict[str, Optional[str]] = {}
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        # also ignore inline comments (everything after ' #')
        code_part = line.split(" #")[0]
        for m in _TEMPLATE_RE.finditer(code_part):
            name, default = m.group(1), m.group(2)
            if name not in found:
                found[name] = default
    return found


def resolve_templates(text: str, values: Dict[str, str]) -> str:
    """Replace ``${var}`` / ``${var:default}`` in *text* with provided values.

    YAML comment lines (starting with ``#``) are left untouched.
    """

    def _replacer(m: re.Match) -> str:
        name, default = m.group(1), m.group(2)
        if name in values:
            return str(values[name])
        if default is not None:
            return default
        raise ValueError(f"Template variable ${{{name}}} has no value and no default")

    resolved_lines = []
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            resolved_lines.append(line)
        else:
            # resolve only the code portion (before inline comment)
            resolved_lines.append(_TEMPLATE_RE.sub(_replacer, line))
    return "".join(resolved_lines)


def add_template_args(
    parser: ArgumentParser, config_path: Optional[str]
) -> Dict[str, Optional[str]]:
    """Discover template variables in *config_path* and add them to *parser*.

    Call this **after** adding all standard arguments but **before**
    ``parser.parse_args()``.  Each ``${var}`` becomes ``--var`` (required);
    each ``${var:default}`` becomes ``--var`` with the given default.

    Returns:
        The discovered variables dict (name → default-or-None).
    """
    if not config_path or not os.path.exists(config_path):
        return {}
    tvars = discover_template_vars(config_path)
    for name, default in tvars.items():
        parser.add_argument(
            f"--{name}",
            required=(default is None),
            default=default,
            help=f"Template variable (from config). "
            + ("Required." if default is None else f"Default: {default}"),
        )
    return tvars


def get_template_values(
    args, template_vars: Dict[str, Optional[str]]
) -> Dict[str, str]:
    """Extract resolved template variable values from parsed args."""
    return {name: str(getattr(args, name)) for name in template_vars}


# ---------------------------------------------------------------------------
# Unified value-list expansion and weighted random choice
# ---------------------------------------------------------------------------

_RANGE_RE = re.compile(
    r"^range\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*(?:,\s*(-?[\d.]+))?\s*\)$"
)
_LINSPACE_RE = re.compile(
    r"^linspace\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(\d+)\s*\)$"
)


def expand_value_list(value) -> list:
    """Expand a config value into a flat Python list.

    Supports the following forms in YAML configs:

    * ``"range(start, stop[, step])"``  — integer or float range, stop exclusive.
    * ``"linspace(start, stop, n)"``    — n evenly-spaced floats, endpoints inclusive.
    * ``[1, 2, 3]``                     — returned as-is.
    * ``42`` / ``null``                 — wrapped in a single-element list.

    The string forms are the canonical shorthand used by sweep, h2h, and
    ladder-self-play configs.  Plain lists are always accepted as a fall-through
    so that hand-enumerated values never need special escaping.
    """
    if isinstance(value, str):
        stripped = value.strip()

        m = _RANGE_RE.match(stripped)
        if m:
            start_s, stop_s, step_s = m.group(1), m.group(2), m.group(3)
            if any("." in s for s in (start_s, stop_s, step_s or "0")):
                start, stop = float(start_s), float(stop_s)
                step = float(step_s) if step_s else 1.0
                result, v = [], start
                while v < stop:
                    result.append(round(v, 6))
                    v += step
                return result
            else:
                return list(
                    range(int(start_s), int(stop_s), int(step_s) if step_s else 1)
                )

        m = _LINSPACE_RE.match(stripped)
        if m:
            start, stop, n = float(m.group(1)), float(m.group(2)), int(m.group(3))
            if n < 1:
                raise ValueError(f"linspace n must be >= 1, got {n}")
            if n == 1:
                return [round(start, 6)]
            step = (stop - start) / (n - 1)
            return [round(start + i * step, 6) for i in range(n)]

        # Plain string (e.g. a team-set name) — treat as a scalar.
        return [value]

    if isinstance(value, list):
        return value
    return [value]


def random_choice(value):
    """Draw a single random element from a config value.

    Accepts all forms understood by :func:`expand_value_list`, plus one
    additional form for non-uniform sampling:

    * ``{weighted: {option_a: w1, option_b: w2, ...}}``
      — draws from the keys with probability proportional to the weights.

    This lets configs express e.g.::

        team_set:
          weighted:
            elite_sets_filled: 4
            competitive: 1

    instead of repeating ``elite_sets_filled`` four times in a plain list.
    """
    if isinstance(value, dict) and "weighted" in value:
        mapping = value["weighted"]
        population = list(mapping.keys())
        weights = [float(mapping[k]) for k in population]
        return random.choices(population, weights=weights, k=1)[0]
    return random.choice(expand_value_list(value))


def _format_value_for_display(value) -> str:
    """Human-readable summary of a raw config value (for preview tables)."""
    if isinstance(value, dict) and "weighted" in value:
        parts = ", ".join(f"{k}({v})" for k, v in value["weighted"].items())
        return f"weighted({parts})"
    lst = expand_value_list(value)
    if len(lst) == 1:
        return str(lst[0])
    return f"[{lst[0]}…{lst[-1]}] ({len(lst)})"


def load_config(
    config_path: str, template_vars: Optional[Dict[str, str]] = None
) -> dict:
    """Load and validate a YAML config file.

    If *template_vars* is provided, ``${var}`` / ``${var:default}``
    placeholders in the raw YAML text are resolved before parsing.
    """
    with open(config_path, "r") as f:
        text = f.read()
    if template_vars:
        text = resolve_templates(text, template_vars)
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must be a YAML mapping, got {type(raw)}")
    return raw


def merge_defaults(defaults: dict, overrides: dict) -> dict:
    """Merge per-policy overrides on top of defaults."""
    merged = {**defaults}
    merged.update({k: v for k, v in overrides.items() if v is not None})
    return merged


def build_policy_spec(name: str, config: dict, defaults: dict) -> PolicySpec:
    """Build a PolicySpec from a per-policy config dict + defaults.

    The policy name is used as model_name unless model_name is explicitly set.
    """
    merged = merge_defaults(defaults, config)
    return PolicySpec(
        name=name,
        model_name=merged.get("model_name", name),
        checkpoint=merged.get("checkpoint", None),
        temperature=float(merged.get("temperature", 1.0)),
        team_set=merged.get("team_set", "competitive"),
        battle_backend=merged.get("battle_backend", "metamon"),
    )


def expand_variants(name: str, config: dict, defaults: dict) -> List[PolicySpec]:
    """Expand a policy entry that may have a 'variants' list.

    Without variants: returns a single PolicySpec.
    With variants: returns one PolicySpec per variant, named {name}-1, {name}-2, ...
        Each variant dict is merged on top of the base config (which is merged on top of defaults).
    """
    variants = config.get("variants", None)
    if variants is None:
        return [build_policy_spec(name, config, defaults)]

    base = {k: v for k, v in config.items() if k != "variants"}
    if "model_name" not in base:
        base["model_name"] = name
    policies = []
    for i, variant in enumerate(variants, 1):
        variant_config = {**base, **variant}
        variant_name = f"{name}-{i}"
        policies.append(build_policy_spec(variant_name, variant_config, defaults))
    return policies


# ---------------------------------------------------------------------------
# GPU distribution
# ---------------------------------------------------------------------------


def distribute_across_gpus(items: List, gpus: List[int]) -> Dict[int, List]:
    """Round-robin distribute items across GPUs."""
    assignments = {gpu: [] for gpu in gpus}
    for i, item in enumerate(items):
        gpu_id = gpus[i % len(gpus)]
        assignments[gpu_id].append(item)
    return assignments


# ---------------------------------------------------------------------------
# Subprocess management
# ---------------------------------------------------------------------------


def run_subprocess(
    cmd: List[str],
    gpu_id: int,
    timeout: int = 3600,
    verbose: bool = False,
    cwd: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """Run a subprocess on a specific GPU with timeout handling.

    Args:
        cmd: Command and arguments.
        gpu_id: GPU to assign via CUDA_VISIBLE_DEVICES.
        timeout: Seconds before killing the process.
        verbose: If True, stream stdout/stderr in real-time.
        cwd: Working directory for the subprocess.

    Returns:
        CompletedProcess with returncode (and captured output if not verbose).
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    kwargs = dict(env=env, cwd=cwd, text=True)
    if not verbose:
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1)

    process = None
    try:
        process = subprocess.Popen(cmd, **kwargs)
        process.wait(timeout=timeout)
        return subprocess.CompletedProcess(
            cmd,
            process.returncode,
            stdout=process.stdout.read() if not verbose and process.stdout else "",
            stderr=process.stderr.read() if not verbose and process.stderr else "",
        )
    except subprocess.TimeoutExpired:
        if process:
            process.kill()
            process.wait()
        return subprocess.CompletedProcess(
            cmd, returncode=-1, stdout="", stderr="TIMEOUT"
        )
    except Exception as e:
        if process and process.poll() is None:
            process.kill()
            process.wait()
        return subprocess.CompletedProcess(cmd, returncode=-1, stdout="", stderr=str(e))
    finally:
        if process:
            for stream in (process.stdout, process.stderr):
                if stream:
                    try:
                        stream.close()
                    except Exception:
                        pass
            del process
        gc.collect()


@dataclass
class MatchupPairResult:
    """Return value from run_matchup_pair."""

    challenger_proc: subprocess.CompletedProcess
    acceptor_proc: subprocess.CompletedProcess
    matchup_dir: str
    challenger_username: str


def run_matchup_pair(
    matchup: MatchupSpec,
    gpu_a: int,
    gpu_b: int,
    output_dir: str,
    timeout: int = 3600,
    acceptor_startup_delay: float = 5.0,
    verbose: bool = False,
    save_trajectories: bool = False,
) -> MatchupPairResult:
    """Run both sides of a matchup as coordinated subprocesses.

    Launches acceptor first, waits for it to come online, then launches
    challenger.  Both sides write per-battle CSV logs to a shared
    ``results/`` directory inside the matchup directory (handled by
    ``PokeEnvWrapper``).
    """
    # Generate unique usernames for this matchup.
    # Showdown caps usernames at 18 chars. Use a hash of the matchup_id to
    # guarantee uniqueness even when many matchups share a long common prefix.
    short_hash = hashlib.md5(matchup.matchup_id.encode()).hexdigest()[:8]
    username_a = f"h2h-A-{short_hash}"  # 14 chars
    username_b = f"h2h-B-{short_hash}"  # 14 chars

    matchup_dir = os.path.join(output_dir, matchup.matchup_id)
    os.makedirs(matchup_dir, exist_ok=True)
    results_dir = os.path.join(matchup_dir, "results")

    serve_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "serve_matchup.py"
    )

    def _build_cmd(
        policy: PolicySpec, username: str, opponent_username: str, role: str
    ):
        cmd = [
            "python",
            serve_script,
            "--model_name",
            policy.model_name,
            "--username",
            username,
            "--opponent_username",
            opponent_username,
            "--role",
            role,
            "--format",
            matchup.battle_format,
            "--n_battles",
            str(matchup.n_battles),
            "--team_set",
            policy.team_set,
            "--battle_backend",
            policy.battle_backend,
            "--temperature",
            str(policy.temperature),
            "--save_results_to",
            results_dir,
        ]
        if policy.checkpoint is not None:
            cmd.extend(["--checkpoint", str(policy.checkpoint)])
        if save_trajectories:
            traj_dir = os.path.join(matchup_dir, "trajectories")
            os.makedirs(traj_dir, exist_ok=True)
            cmd.extend(["--save_trajectories_to", traj_dir])
        return cmd

    # Acceptor (policy_b) launches first
    acceptor_cmd = _build_cmd(matchup.policy_b, username_b, username_a, "acceptor")
    env_acceptor = os.environ.copy()
    env_acceptor["CUDA_VISIBLE_DEVICES"] = str(gpu_b)

    kwargs_acceptor = dict(env=env_acceptor, text=True)
    if not verbose:
        kwargs_acceptor.update(
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1
        )

    acceptor_proc = subprocess.Popen(acceptor_cmd, **kwargs_acceptor)

    # Wait for acceptor to connect
    time.sleep(acceptor_startup_delay)

    # Challenger (policy_a) launches second
    challenger_cmd = _build_cmd(matchup.policy_a, username_a, username_b, "challenger")
    challenger_result = run_subprocess(
        challenger_cmd, gpu_a, timeout=timeout, verbose=verbose
    )

    # Wait for acceptor to finish too
    try:
        acceptor_proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        acceptor_proc.kill()
        acceptor_proc.wait()

    acceptor_result = subprocess.CompletedProcess(
        acceptor_cmd,
        acceptor_proc.returncode,
        stdout=(
            acceptor_proc.stdout.read() if not verbose and acceptor_proc.stdout else ""
        ),
        stderr=(
            acceptor_proc.stderr.read() if not verbose and acceptor_proc.stderr else ""
        ),
    )

    for stream in (acceptor_proc.stdout, acceptor_proc.stderr):
        if stream:
            try:
                stream.close()
            except Exception:
                pass

    return MatchupPairResult(
        challenger_proc=challenger_result,
        acceptor_proc=acceptor_result,
        matchup_dir=matchup_dir,
        challenger_username=username_a,
    )
