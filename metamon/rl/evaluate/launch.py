"""
Shared launcher for challenge-based evaluation (h2h and sweep).

Takes a list of MatchupSpecs and runs them with a thread pool,
tracking results for crash recovery.
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from metamon.rl.evaluate.common import (
    MatchupSpec,
    MatchupPairResult,
    distribute_across_gpus,
    run_matchup_pair,
)
from metamon.rl.evaluate.results import ResultsTracker


def run_all_matchups(
    matchups: List[MatchupSpec],
    gpus: List[int],
    output_dir: str,
    max_concurrent: int = None,
    timeout: int = 3600,
    acceptor_startup_delay: float = 10.0,
    verbose: bool = False,
    save_trajectories: bool = False,
):
    """Run a list of matchups with crash recovery and a thread pool.

    Args:
        matchups: All matchups to run.
        gpus: Available GPU IDs.
        output_dir: Root directory for results, team logs, etc.
        max_concurrent: Max matchups to run in parallel (default = len(gpus)).
        timeout: Seconds before killing a matchup.
        acceptor_startup_delay: Seconds to wait after launching acceptor before challenger.
        verbose: Stream subprocess output in real-time.
        save_trajectories: Whether to save trajectory files.
    """
    if max_concurrent is None:
        max_concurrent = len(gpus)

    tracker = ResultsTracker(output_dir)

    # Filter out completed matchups (crash recovery)
    remaining = [m for m in matchups if not tracker.is_completed(m.matchup_id)]
    n_skipped = len(matchups) - len(remaining)

    print(f"\n{'='*60}")
    print(f"  Head-to-Head Launcher")
    print(f"{'='*60}")
    print(f"  Total matchups:     {len(matchups)}")
    print(f"  Already completed:  {n_skipped}")
    print(f"  Remaining:          {len(remaining)}")
    print(f"  Max concurrent:     {max_concurrent}")
    print(f"  GPUs:               {gpus}")
    print(f"  Output:             {output_dir}")
    print(f"  Timeout:            {timeout}s ({timeout // 60}min)")
    print(f"{'='*60}\n")

    if not remaining:
        print("All matchups already completed!")
        tracker.print_win_matrix()
        tracker.write_win_matrix_csv()
        return

    # Assign GPU pairs for each matchup (round-robin across available GPUs)
    # Each matchup needs 2 GPU slots: one for challenger, one for acceptor
    gpu_pairs = []
    for i, matchup in enumerate(remaining):
        gpu_a = gpus[(2 * i) % len(gpus)]
        gpu_b = gpus[(2 * i + 1) % len(gpus)]
        gpu_pairs.append((gpu_a, gpu_b))

    completed_count = n_skipped
    failed_count = 0

    def _run_one(matchup: MatchupSpec, gpu_a: int, gpu_b: int):
        label = f"{matchup.policy_a.short_label} vs {matchup.policy_b.short_label}"
        print(f"▶ Starting: {label}  (GPUs {gpu_a},{gpu_b})")
        t0 = time.time()

        pair = run_matchup_pair(
            matchup=matchup,
            gpu_a=gpu_a,
            gpu_b=gpu_b,
            output_dir=output_dir,
            timeout=timeout,
            acceptor_startup_delay=acceptor_startup_delay,
            verbose=verbose,
            save_trajectories=save_trajectories,
        )

        elapsed = time.time() - t0

        # Parse results from the CSVs written by PokeEnvWrapper
        results_dir = os.path.join(pair.matchup_dir, "results")
        result = tracker.record_from_results_dir(
            matchup_id=matchup.matchup_id,
            policy_a_name=matchup.policy_a.short_label,
            policy_b_name=matchup.policy_b.short_label,
            results_dir=results_dir,
            challenger_username=pair.challenger_username,
        )

        if result is not None:
            print(
                f"✓ Completed: {label}  "
                f"({result.policy_a_wins}W-{result.policy_b_wins}L, "
                f"{elapsed:.0f}s)"
            )
            return True
        else:
            print(f"✗ Failed: {label}  ({elapsed:.0f}s)")
            if not verbose:
                if pair.challenger_proc.stderr:
                    print(f"  Challenger stderr: {pair.challenger_proc.stderr[:500]}")
                if pair.acceptor_proc.stderr:
                    print(f"  Acceptor stderr: {pair.acceptor_proc.stderr[:500]}")
            return False

    try:
        with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
            futures = {}
            for matchup, (gpu_a, gpu_b) in zip(remaining, gpu_pairs):
                future = pool.submit(_run_one, matchup, gpu_a, gpu_b)
                futures[future] = matchup

            for future in as_completed(futures):
                matchup = futures[future]
                try:
                    success = future.result()
                    if success:
                        completed_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    print(f"✗ Exception in {matchup.matchup_id}: {e}")
                    failed_count += 1

                print(
                    f"  Progress: {completed_count}/{len(matchups)} completed, "
                    f"{failed_count} failed"
                )

    except KeyboardInterrupt:
        print("\n\nInterrupted! Partial results saved.")

    # Final summary
    print(f"\n{'='*60}")
    print(f"  Completed: {completed_count}/{len(matchups)}  |  Failed: {failed_count}")
    print(f"{'='*60}")
    tracker.print_win_matrix()
    tracker.write_win_matrix_csv()
