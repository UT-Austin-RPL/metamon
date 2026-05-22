"""
Config parsing for head-to-head evaluation.

Example config:

    battle_format: gen1ou
    battles_per_matchup: 50

    defaults:
      team_set: modern_replays_v2
      battle_backend: metamon
      checkpoint: null
      temperature: 1.0

    policies:
      # Simple: outer key = model_name
      Kadabra: {}
      SyntheticRLV2:
        checkpoint: 40

      # Override model_name if display name differs
      MyAlias:
        model_name: Alakazam2
        checkpoint: 48

      # Variants: expand one model into multiple configs
      Kadabra:
        variants:
          - team_set: modern_replays_v2
          - team_set: competitive
          - { team_set: competitive, checkpoint: 20 }
"""

from itertools import combinations
from typing import List, Optional

from metamon.rl.evaluate.common import (
    PolicySpec,
    MatchupSpec,
    load_config,
    expand_variants,
)


def parse_h2h_config(
    config_path: str, template_vars: Optional[dict] = None
) -> List[MatchupSpec]:
    """Parse a head-to-head YAML config into a list of MatchupSpecs.

    Generates all unordered pairs of policies.
    """
    raw = load_config(config_path, template_vars=template_vars)
    defaults = raw.get("defaults", {})
    battle_format = raw["battle_format"]
    n_battles = raw.get("battles_per_matchup", 50)

    if "policies" not in raw:
        raise ValueError("h2h config must have a 'policies' section")

    # Expand all policies (including variants)
    all_policies: List[PolicySpec] = []
    for name, policy_config in raw["policies"].items():
        if policy_config is None:
            policy_config = {}
        all_policies.extend(expand_variants(name, policy_config, defaults))

    if len(all_policies) < 2:
        raise ValueError(
            f"h2h config must define at least 2 policies, got {len(all_policies)}"
        )

    # Generate all unordered pairs
    matchups = []
    for a, b in combinations(all_policies, 2):
        matchups.append(
            MatchupSpec(
                policy_a=a,
                policy_b=b,
                n_battles=n_battles,
                battle_format=battle_format,
            )
        )

    return matchups
