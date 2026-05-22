"""
Sweep evaluation launcher.

Usage:
    python -m metamon.rl.evaluate.sweep \\
        --config sweep_config.yaml \\
        --gpus 0 1 2 3 \\
        --output_dir ./sweep_results

    # With template variables:
    python -m metamon.rl.evaluate.sweep \\
        --config sweep_template.yaml \\
        --model SyntheticRLV2 \\
        --gpus 0 1 --output_dir ./results

Add --dry_run to preview all matchups without running them.

Config files may contain ``${var}`` template placeholders that become
extra CLI arguments automatically.  See ``metamon.rl.evaluate.common``
for details.
"""

import os
import sys
from argparse import ArgumentParser

from metamon.rl.evaluate.common import add_template_args, get_template_values
from metamon.rl.evaluate.sweep.config import parse_sweep_config
from metamon.rl.evaluate.launch import run_all_matchups
from metamon.rl.evaluate.preview import preview_matchups


def main():
    # --- first pass: discover --config so we can scan for template vars ---
    pre_parser = ArgumentParser(add_help=False)
    pre_parser.add_argument("--config")
    pre_args, _ = pre_parser.parse_known_args()

    # --- full parser ---
    parser = ArgumentParser(
        description="Run sweep evaluation against a fixed opponent."
    )
    parser.add_argument("--config", required=True, help="Path to sweep YAML config.")
    parser.add_argument(
        "--gpus", nargs="+", type=int, required=True, help="GPU IDs to use."
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory for results and logs."
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=None,
        help="Max concurrent matchups (default = number of GPUs).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout per matchup in seconds (default: 3600 = 1 hour).",
    )
    parser.add_argument(
        "--acceptor_startup_delay",
        type=float,
        default=10.0,
        help="Seconds to wait for acceptor before launching challenger.",
    )
    parser.add_argument(
        "--save_trajectories",
        action="store_true",
        help="Save trajectory files for each matchup.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Stream subprocess output."
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview matchups without running them.",
    )

    # auto-discover template variables and add them as CLI args
    template_vars = add_template_args(parser, pre_args.config)

    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    tpl_values = get_template_values(args, template_vars) if template_vars else None
    matchups = parse_sweep_config(config_path, template_vars=tpl_values)

    if args.dry_run:
        preview_matchups(matchups, mode="sweep", template_values=tpl_values)
        return

    run_all_matchups(
        matchups=matchups,
        gpus=args.gpus,
        output_dir=args.output_dir,
        max_concurrent=args.max_concurrent,
        timeout=args.timeout,
        acceptor_startup_delay=args.acceptor_startup_delay,
        verbose=args.verbose,
        save_trajectories=args.save_trajectories,
    )


if __name__ == "__main__":
    main()
