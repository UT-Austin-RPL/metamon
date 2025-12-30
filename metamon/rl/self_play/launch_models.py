import gc
import os
import random
import subprocess
import sys
import threading
import time
import yaml
from argparse import ArgumentParser
from typing import List, Dict


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
):
    if startup_delay > 0:
        print(
            f"Waiting {startup_delay} seconds before starting {username} on GPU {gpu_id}..."
        )
        time.sleep(startup_delay)

    run_count = 0
    while True:
        run_count += 1
        print(f"\n{'='*60}")
        print(
            f"[Run #{run_count}] Starting {username} on GPU {gpu_id} for format {format_name} with {n_challenges} challenges..."
        )
        print(f"{'='*60}")

        # set GPU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        cmd = [
            "python",
            "serve_model.py",
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

        process = None
        try:
            if verbose:
                # verbose mode: stream output in real-time
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                    text=True,
                )
            else:
                # quiet mode: capture output
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )

            # wait for completion
            try:
                process.wait(timeout=timeout)
                if process.returncode == 0:
                    print(
                        f"✓ {username} on GPU {gpu_id} [Run #{run_count}] completed successfully"
                    )
                else:
                    print(
                        f"✗ {username} on GPU {gpu_id} [Run #{run_count}] failed with code {process.returncode}"
                    )
                    if verbose:
                        # stderr already printed in real-time
                        pass
                    else:
                        stderr_output = process.stderr.read()
                        if stderr_output:
                            print(f"Error output from {username}:")
                            print(stderr_output)
            except subprocess.TimeoutExpired:
                print(
                    f"⏰ {username} on GPU {gpu_id} [Run #{run_count}] timed out after {timeout} seconds"
                )
                process.kill()
                process.wait()

        except Exception as e:
            print(
                f"✗ {username} on GPU {gpu_id} [Run #{run_count}] failed with exception: {e}"
            )
            if process and process.poll() is None:
                process.kill()
                process.wait()

        finally:
            # cleanup resources
            if process:
                if process.poll() is None:
                    process.kill()
                    try:
                        process.wait(timeout=5)
                    except:
                        pass

                if hasattr(process, "stdout") and process.stdout:
                    process.stdout.close()
                if hasattr(process, "stderr") and process.stderr:
                    process.stderr.close()

                del process

            gc.collect()

        print(f"Waiting {restart_delay} seconds before relaunching {username}...")
        time.sleep(restart_delay)


def get_usernames(config_path: str) -> List[str]:
    """Expand agents based on num_agents field"""
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    # validate structure
    if "agents" not in raw_config:
        raise ValueError("Config must have 'agents' section")

    defaults = raw_config.get("defaults", {})
    agents = raw_config.get("agents", {})

    # validate defaults
    required_defaults = ["team_set", "battle_backend", "checkpoints", "num_agents"]
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
        merged_config = {**defaults, **agent_config}
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


def distribute_across_gpus(
    usernames: List[str], gpus: List[int]
) -> Dict[int, List[str]]:
    gpu_assignments = {gpu: [] for gpu in gpus}
    for i, username in enumerate(usernames):
        gpu_id = gpus[i % len(gpus)]
        gpu_assignments[gpu_id].append(username)
    return gpu_assignments


def run_all_usernames_parallel(
    format_name: str,
    gpus: List[int],
    config_path: str,
    n_challenges: int = 50,
    restart_delay: int = 60,
    timeout: int = 2700,
    save_trajectories_to: str = None,
    verbose: bool = False,
):
    usernames = get_usernames(config_path)

    print(f"Running usernames: {', '.join(usernames)}")
    print(f"Available GPUs: {gpus}")
    print(f"Format: {format_name}")
    print(f"Config: {config_path}")
    print(f"Challenges per username: {n_challenges}")
    print(f"Restart delay: {restart_delay} seconds")
    print(f"Timeout per run: {timeout} seconds ({timeout//60} minutes)")
    if save_trajectories_to:
        print(f"Saving trajectories to: {save_trajectories_to}")
    print("-" * 50)

    # distribute usernames across GPUs
    gpu_assignments = distribute_across_gpus(usernames, gpus)

    for gpu_id, usernames_for_gpu in gpu_assignments.items():
        print(f"GPU {gpu_id}: {', '.join(usernames_for_gpu)}")
    print("-" * 50)

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
                ),
                daemon=True,
            )
            threads.append(thread)
            thread.start()
            startup_delay += (
                10  # increased delay to allow bots to connect and start challenging
            )

    print(f"\n✓ All {len(threads)} bots launched and running continuously!")
    print("Press Ctrl+C to stop all bots")
    print("-" * 50)

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n\nShutting down all bots...")
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
        default="earlygen_config.yaml",
        help="Path to YAML config file (default: earlygen_config.yaml)",
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

    args = parser.parse_args()

    # validate GPUs
    if not args.gpus:
        print("Error: At least one GPU ID must be specified")
        sys.exit(1)

    # convert config path to absolute path so subprocesses can find it
    config_path = os.path.abspath(args.config)

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
    )


if __name__ == "__main__":
    main()
