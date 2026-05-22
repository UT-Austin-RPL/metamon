import os
import sys
import threading
import time
from argparse import ArgumentParser
from collections import defaultdict
from typing import List, Dict

from metamon.rl.evaluate.common import (
    distribute_across_gpus,
    load_config,
    merge_defaults,
    run_subprocess,
)


class _StatsTracker:
    """Thread-safe counter for battle-generation throughput."""

    def __init__(self):
        self._lock = threading.Lock()
        self._start_time = time.monotonic()
        self._total_battles = 0
        self._total_runs = 0
        self._failed_runs = 0
        self._battles_by_agent: Dict[str, int] = defaultdict(int)

    def record_run(self, username: str, n_battles: int, success: bool):
        with self._lock:
            self._total_runs += 1
            if success:
                self._total_battles += n_battles
                self._battles_by_agent[username] += n_battles
            else:
                self._failed_runs += 1

    def summary(self) -> str:
        with self._lock:
            elapsed = time.monotonic() - self._start_time
            elapsed_h = elapsed / 3600
            bph = self._total_battles / elapsed_h if elapsed_h > 0 else 0
            lines = [
                f"  Elapsed : {elapsed/60:.1f} min",
                f"  Battles : {self._total_battles:,}  ({bph:,.0f} battles/hr)",
                f"  Runs    : {self._total_runs} completed, {self._failed_runs} failed",
                "  By agent:",
            ]
            for agent, count in sorted(
                self._battles_by_agent.items(), key=lambda x: -x[1]
            ):
                lines.append(f"    {agent:<30} {count:>6,} battles")
            return "\n".join(lines)


def get_usernames(config_path: str) -> List[str]:
    """Expand agents based on num_agents field"""
    raw_config = load_config(config_path)

    # validate structure
    if "agents" not in raw_config:
        raise ValueError("Config must have 'agents' section")

    defaults = raw_config.get("defaults", {})
    agents = raw_config.get("agents", {})

    # validate defaults
    required_defaults = [
        "team_set",
        "battle_backend",
        "checkpoints",
        "temperatures",
        "num_agents",
    ]
    missing_defaults = [field for field in required_defaults if field not in defaults]
    if missing_defaults:
        raise ValueError(
            f"defaults section missing required fields: {', '.join(missing_defaults)}"
        )

    expanded_usernames = []
    for base_username, agent_config in agents.items():
        # validate required fields
        if "model_name" not in agent_config and "model_name" not in defaults:
            raise ValueError(
                f"Agent {base_username} missing required field: model_name"
            )

        # expand based on num_agents
        merged_config = merge_defaults(defaults, agent_config or {})
        num_agents = merged_config.get("num_agents", 1)
        # handle None/null values in yaml
        if num_agents is None:
            num_agents = 1

        if num_agents == 1:
            expanded_usernames.append(base_username)
        else:
            # add numbered copies
            for i in range(1, num_agents + 1):
                expanded_username = f"{base_username}-{i}"
                expanded_usernames.append(expanded_username)

    print(
        f"Found {len(agents)} base agents, expanded to {len(expanded_usernames)} total: {', '.join(expanded_usernames)}"
    )
    return expanded_usernames


def get_agent_details(config_path: str) -> List[dict]:
    """Get full agent details for preview. Returns list of dicts with username, model_name, etc."""
    raw_config = load_config(config_path)
    defaults = raw_config.get("defaults", {})
    agents = raw_config.get("agents", {})

    details = []
    for base_username, agent_config in agents.items():
        merged = merge_defaults(defaults, agent_config or {})
        num_agents = merged.get("num_agents", 1) or 1

        usernames = (
            [base_username]
            if num_agents == 1
            else [f"{base_username}-{i}" for i in range(1, num_agents + 1)]
        )
        for username in usernames:
            details.append(
                {
                    "username": username,
                    "model_name": merged.get("model_name", base_username),
                    "checkpoint": merged.get("checkpoints"),
                    "temperature": merged.get("temperatures"),
                    "team_set": merged.get("team_set"),
                    "battle_backend": merged.get("battle_backend"),
                }
            )
    return details


def run_username_on_gpu_continuous(
    gpu_id: int,
    username: str,
    format_name: str,
    config_path: str,
    n_challenges: int = 50,
    startup_delay: int = 0,
    restart_delay: int = 60,
    timeout: int = 2700,
    save_trajectories_to: str = None,
    verbose: bool = False,
    stats: "_StatsTracker | None" = None,
):
    if startup_delay > 0:
        print(
            f"Waiting {startup_delay} seconds before starting {username} on GPU {gpu_id}..."
        )
        time.sleep(startup_delay)

    serve_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "serve_model.py"
    )

    run_count = 0
    while True:
        run_count += 1
        t0 = time.monotonic()

        cmd = [
            "python",
            serve_script,
            "--username",
            username,
            "--format",
            format_name,
            "--n_challenges",
            str(n_challenges),
            "--config",
            config_path,
        ]
        if save_trajectories_to:
            cmd.extend(["--save_trajectories_to", save_trajectories_to])

        result = run_subprocess(cmd, gpu_id, timeout=timeout, verbose=verbose)
        elapsed = time.monotonic() - t0

        if result.returncode == 0:
            bps = n_challenges / elapsed if elapsed > 0 else 0
            print(
                f"✓ {username} GPU{gpu_id} run#{run_count} "
                f"— {n_challenges} battles in {elapsed:.0f}s "
                f"({bps:.2f} battles/s)"
            )
            if stats:
                stats.record_run(username, n_challenges, success=True)
        elif result.stderr == "TIMEOUT":
            print(
                f"⏰ {username} GPU{gpu_id} run#{run_count} timed out after {timeout}s"
            )
            if stats:
                stats.record_run(username, 0, success=False)
        else:
            print(
                f"✗ {username} GPU{gpu_id} run#{run_count} failed (code {result.returncode})"
            )
            if not verbose and result.stderr:
                print(f"  stderr: {result.stderr[:500]}")
            if stats:
                stats.record_run(username, 0, success=False)

        time.sleep(restart_delay)


def run_all_usernames_parallel(
    format_name: str,
    gpus: List[int],
    config_path: str,
    n_challenges: int = 50,
    restart_delay: int = 60,
    timeout: int = 2700,
    save_trajectories_to: str = None,
    verbose: bool = False,
    stats_interval: int = 300,
):
    usernames = get_usernames(config_path)

    print(f"Running usernames: {', '.join(usernames)}")
    print(f"Available GPUs: {gpus}")
    print(f"Format: {format_name}")
    print(f"Config: {config_path}")
    print(f"Challenges per username: {n_challenges}")
    print(f"Restart delay: {restart_delay} seconds")
    print(f"Timeout per run: {timeout} seconds ({timeout//60} minutes)")
    print(f"Stats interval: every {stats_interval}s")
    if save_trajectories_to:
        print(f"Saving trajectories to: {save_trajectories_to}")
    print("-" * 50)

    gpu_assignments = distribute_across_gpus(usernames, gpus)
    for gpu_id, usernames_for_gpu in gpu_assignments.items():
        print(f"GPU {gpu_id}: {', '.join(usernames_for_gpu)}")
    print("-" * 50)

    stats = _StatsTracker()

    threads = []
    startup_delay = 0
    for gpu_id, usernames_for_gpu in gpu_assignments.items():
        for username in usernames_for_gpu:
            thread = threading.Thread(
                target=run_username_on_gpu_continuous,
                args=(
                    gpu_id,
                    username,
                    format_name,
                    config_path,
                    n_challenges,
                    startup_delay,
                    restart_delay,
                    timeout,
                    save_trajectories_to,
                    verbose,
                    stats,
                ),
                daemon=True,
            )
            threads.append(thread)
            thread.start()
            startup_delay += 2

    print(f"\n✓ All {len(threads)} bots launched and running continuously!")
    print("Press Ctrl+C to stop all bots")
    print("-" * 50)

    try:
        last_print = time.monotonic()
        while True:
            time.sleep(10)
            if time.monotonic() - last_print >= stats_interval:
                print(f"\n{'─'*50}")
                print("THROUGHPUT STATS")
                print(stats.summary())
                print(f"{'─'*50}\n")
                last_print = time.monotonic()
    except KeyboardInterrupt:
        print("\n\nFinal stats:")
        print(stats.summary())
        print("\nShutting down all bots...")
        sys.exit(0)


def main():
    parser = ArgumentParser(
        description="Run serve_model.py for all usernames across multiple GPUs (self-play)"
    )
    parser.add_argument(
        "--format",
        required=True,
        choices=["gen1ou", "gen2ou", "gen3ou", "gen4ou", "gen9ou"],
        help="The battle format to use",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        required=True,
        help="List of GPU IDs to use (e.g., --gpus 0 1 2 3)",
    )
    parser.add_argument(
        "--config",
        default="example_config.yaml",
        help="Path to YAML config file (default: example_config.yaml)",
    )
    parser.add_argument(
        "--n_challenges",
        type=int,
        default=50,
        help="Number of challenges per username (default: 50)",
    )
    parser.add_argument(
        "--restart_delay",
        type=int,
        default=80,
        help="Seconds to wait before relaunching each bot after completion (default: 80)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=2700,
        help="Timeout in seconds for each bot run (default: 2700 = 45 minutes)",
    )
    parser.add_argument(
        "--save_trajectories_to",
        required=True,
        help="Base directory to save trajectories (will create subdirs per model)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print error messages from failed runs",
    )
    parser.add_argument(
        "--stats_interval",
        type=int,
        default=300,
        help="Print throughput summary every this many seconds (default: 300)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview agents without launching them.",
    )

    args = parser.parse_args()

    # validate GPUs
    if not args.gpus:
        print("Error: At least one GPU ID must be specified")
        sys.exit(1)

    # convert config path to absolute path so subprocesses can find it
    config_path = os.path.abspath(args.config)

    if args.dry_run:
        from metamon.rl.evaluate.preview import preview_ladder_agents

        agents = get_agent_details(config_path)
        preview_ladder_agents(agents)
        return

    # run continuously
    run_all_usernames_parallel(
        args.format,
        args.gpus,
        config_path,
        args.n_challenges,
        args.restart_delay,
        args.timeout,
        args.save_trajectories_to,
        args.verbose,
        args.stats_interval,
    )


if __name__ == "__main__":
    main()
