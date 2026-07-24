"""PSRO-Lite concerns for online RL, split out of :mod:`metamon.rl.online_rl`.

CLI flag definitions, config construction, sidecar path resolution, and the
mtime-cached weight provider used by the learner's FIFO sampler. All PSRO-Lite
behavior is default-off (uniform sampling identical unless explicitly enabled);
see ``docs/psro_lite_plan.md`` for the design.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

from metamon.rl.psro_lite import PsroConfig, read_sidecar


def add_psro_cli_args(parser) -> None:
    """Register ``--psro_*`` flags on ``parser`` (all default-off)."""
    parser.add_argument(
        "--psro_weighting",
        action="store_true",
        help="Enable PSRO-Lite prioritized opponent sampling at collection time "
        "(writes meta_weights.json sidecar from the collector). No-op before "
        "--psro_start_epoch.",
    )
    parser.add_argument(
        "--psro_start_epoch",
        type=int,
        default=0,
        help="First epoch to write/apply PSRO-Lite weights (default 0 = always; "
        "the live run sets 1000 for a mid-run switchover).",
    )
    parser.add_argument(
        "--psro_temp",
        type=float,
        default=1.0,
        help="Prioritization temperature τ; small ⇒ sharp, large ⇒ uniform.",
    )
    parser.add_argument(
        "--psro_floor",
        type=float,
        default=0.05,
        help="Per-opponent diversity floor (non-zero mass for every opponent).",
    )
    parser.add_argument(
        "--psro_min_games",
        type=int,
        default=20,
        help="Minimum games vs an opponent before weighting it (else uniform fallback).",
    )
    parser.add_argument(
        "--psro_window",
        type=int,
        default=50000,
        help="Number of most-recent buffer files to score (rolling window).",
    )
    parser.add_argument(
        "--psro_update_interval",
        type=int,
        default=5,
        help="Epochs between PSRO-Lite weight updates (one forced update on the "
        "start epoch itself).",
    )
    parser.add_argument(
        "--psro_ema",
        type=float,
        default=0.7,
        help="EMA smoothing factor β for weights across updates (0=no smoothing).",
    )
    parser.add_argument(
        "--psro_solver",
        type=str,
        default="prioritized",
        choices=["prioritized", "nash"],
        help="Weight solver. 'prioritized' (PFSP-style) is v1; 'nash' is reserved "
        "for v3 (requires pool-vs-pool eval).",
    )
    parser.add_argument(
        "--psro_fifo_reweight",
        action="store_true",
        help="Per-trajectory FIFO reweighting: the learner's online 40%% mixture "
        "samples files in proportion to the current per-opponent weight instead "
        "of uniformly (fixes buffer lag at a mid-run switchover).",
    )
    parser.add_argument(
        "--psro_buffer_trim",
        type=int,
        default=None,
        help="If set, evict the FIFO down to this many files once at "
        "--psro_start_epoch to accelerate turnover of the uniform-sampled "
        "backlog (e.g. 50000).",
    )
    # Diversification quota: guarantees every pool agent a minimum number of
    # games over a rolling window so dominated, ladder-strong policies never
    # fall to ~0 games played (which previously triggered the cold-fallback
    # weight spike). The PSRO-Lite weights then tilt the *surplus* (window
    # slots beyond all quotas) toward weaker matchups.
    parser.add_argument(
        "--psro_quota_min_games",
        type=int,
        default=0,
        help="Per-agent minimum games over the rolling --psro_quota_window. "
        "0 disables the quota (pure weighted sampling). One configure() call "
        "assigns one shared opponent to all lanes for a battle, so the quota "
        "is enforced in units of ceil(min_games / lanes) assignments. Default "
        "0; the launch scripts set this when PSRO is on.",
    )
    parser.add_argument(
        "--psro_quota_window",
        type=int,
        default=128,
        help="Rolling window (in env reset / configure() calls) over which the "
        "per-agent quota is enforced. Must be >= n_agents * ceil(min_games / "
        "lanes) or the quota is infeasible and falls back to weighted sampling.",
    )
    parser.add_argument(
        "--psro_novelty",
        type=float,
        default=0.0,
        help="Decaying novelty bonus γ added to each opponent's raw weight: "
        "γ/(n+γ0). 0 (default) disables — the collection quota is the primary "
        "exploration mechanism. Set >0 to give genuinely novel opponents a "
        "small, n-decaying bump on top of the floor.",
    )
    parser.add_argument(
        "--psro_cap",
        type=float,
        default=None,
        help="Weight-ratio cap R: hard-bounds each raw weight to R*floor as a "
        "safety net against solver spikes. None (default) disables.",
    )


def psro_sidecar_path(buffer_dir: str, battle_format: str) -> str:
    """Path to the ``meta_weights.json`` sidecar shared between collector/learner."""
    return os.path.join(os.path.abspath(buffer_dir), battle_format, "meta_weights.json")


def load_pool_agent_names(opponent_config_path: str, battle_format: str) -> list[str]:
    """Return the row names (``agents[i][0]``) of the training opponent pool."""
    from metamon.rl.evaluate.opponent_pool import load_opponent_pool

    pool = load_opponent_pool(opponent_config_path, battle_format=battle_format)
    return [row[0] for row in pool.agents]


def make_psro_config_from_args(args, *, battle_format: str) -> Optional[PsroConfig]:
    """Build a :class:`PsroConfig` from CLI args, or ``None`` if PSRO-Lite is off."""
    if not args.psro_weighting:
        return None
    agent_names = load_pool_agent_names(args.train_pool, battle_format)
    return PsroConfig(
        buffer_dir=args.buffer_dir,
        battle_format=battle_format,
        agent_names=agent_names,
        start_epoch=args.psro_start_epoch,
        update_interval=args.psro_update_interval,
        window=args.psro_window,
        min_games=args.psro_min_games,
        temp=args.psro_temp,
        floor=args.psro_floor,
        ema=args.psro_ema,
        solver=args.psro_solver,
        fifo_reweight=args.psro_fifo_reweight,
        buffer_trim=args.psro_buffer_trim,
        novelty_gamma=args.psro_novelty,
        cap_ratio=args.psro_cap,
    )


def make_opponent_weight_provider(
    sidecar_path: str,
) -> Callable[[], dict[str, float]]:
    """Return a callable that reads the PSRO-Lite sidecar (cached by mtime).

    Returns the last-seen weights dict (or ``{}`` → uniform fallback) so the
    learner's FIFO sampler never hard-fails when the sidecar is momentarily
    absent (e.g. before ``psro_start_epoch`` or during an atomic rewrite).
    """
    state = {"mtime": None, "weights": None}

    def provider() -> dict[str, float]:
        weights, mtime = read_sidecar(sidecar_path, state["mtime"])
        if weights is not None:
            state["weights"] = weights
            state["mtime"] = mtime
        return state["weights"] or {}

    return provider


def resolve_quota(args) -> tuple[Optional[int], int]:
    """Extract ``(opponent_quota_min_games, opponent_quota_window)`` from args.

    Returns ``(None, window)`` when the quota is disabled (``min_games <= 0``).
    """
    min_games = args.psro_quota_min_games if args.psro_quota_min_games > 0 else None
    return min_games, args.psro_quota_window


def log_psro_status(
    psro_config: Optional[PsroConfig],
    *,
    sidecar_path: str,
    fifo_reweight: bool,
    buffer_trim: Optional[int],
    quota_min_games: Optional[int],
    quota_window: int,
) -> None:
    """Print the PSRO-Lite / quota status block (no-op when PSRO is off)."""
    if psro_config is not None:
        print(
            f"  PSRO-Lite: ON (start_epoch={psro_config.start_epoch}, "
            f"solver={psro_config.solver}, temp={psro_config.temp}, "
            f"floor={psro_config.floor}, min_games={psro_config.min_games}, "
            f"window={psro_config.window}, "
            f"update_interval={psro_config.update_interval}, "
            f"ema={psro_config.ema})"
        )
        print(f"  PSRO agents: {psro_config.agent_names}")
        if fifo_reweight:
            print(f"  PSRO FIFO reweighting: ON (sidecar={sidecar_path})")
        if buffer_trim is not None:
            print(f"  PSRO buffer trim: {buffer_trim} at start epoch")
    if quota_min_games is not None:
        print(
            f"  PSRO quota: min {quota_min_games} games/agent over "
            f"window={quota_window} resets"
        )
