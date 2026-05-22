"""
Config parsing for sweep evaluation.

Sweep mode evaluates one policy across a parameter grid against a fixed opponent.

Example config:

    battle_format: gen1ou
    battles_per_matchup: 50

    defaults:
      team_set: modern_replays_v2
      battle_backend: metamon

    opponent:
      model_name: Kadabra
      checkpoint: null
      temperature: 1.0

    sweep:
      model_name: SyntheticRLV2
      checkpoints: "range(2, 50, 2)"        # or a list: [null, 32, 36, 40]
      temperatures: "linspace(0.5, 3.0, 6)" # or a list: [1.0, 1.5, 2.0]

Generates a cartesian product of checkpoints × temperatures, each played
against the fixed opponent.

Shorthand syntax for sweep values:
  - range(start, stop, step)  — Python-style, stop exclusive. Best for int checkpoints.
  - linspace(start, stop, n)  — N evenly-spaced points, both endpoints inclusive. Best for floats.
"""

import itertools
from typing import Dict, List, Optional

from metamon.rl.evaluate.common import (
    PolicySpec,
    MatchupSpec,
    load_config,
    build_policy_spec,
    merge_defaults,
    expand_value_list,
)


def parse_sweep_config(
    config_path: str, template_vars: Optional[Dict[str, str]] = None
) -> List[MatchupSpec]:
    """Parse a sweep YAML config into a list of MatchupSpecs."""
    raw = load_config(config_path, template_vars=template_vars)
    defaults = raw.get("defaults", {})
    battle_format = raw["battle_format"]
    n_battles = raw.get("battles_per_matchup", 50)

    if "opponent" not in raw:
        raise ValueError("sweep config must have an 'opponent' section")
    if "sweep" not in raw:
        raise ValueError("sweep config must have a 'sweep' section")

    # Build the fixed opponent
    opp_config = raw["opponent"]
    opp_name = opp_config.get("name", opp_config.get("model_name", "opponent"))
    opponent = build_policy_spec(opp_name, opp_config, defaults)

    # Build the sweep grid
    sweep_config = raw["sweep"]
    sweep_model = sweep_config.get("model_name")
    if sweep_model is None:
        raise ValueError("sweep section must have 'model_name'")

    base_sweep = {
        k: v
        for k, v in sweep_config.items()
        if k not in ("checkpoints", "temperatures")
    }

    checkpoints = expand_value_list(sweep_config.get("checkpoints", [None]))
    temperatures = expand_value_list(sweep_config.get("temperatures", [1.0]))

    # Cartesian product
    matchups = []
    for ckpt, temp in itertools.product(checkpoints, temperatures):
        # Use the bare model name; short_label will append ckpt/temp for display
        variant_config = {**base_sweep, "checkpoint": ckpt, "temperature": temp}
        policy = build_policy_spec(sweep_model, variant_config, defaults)

        matchups.append(
            MatchupSpec(
                policy_a=policy,
                policy_b=opponent,
                n_battles=n_battles,
                battle_format=battle_format,
            )
        )

    return matchups
