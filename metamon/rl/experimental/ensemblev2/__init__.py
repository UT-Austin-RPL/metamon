"""EnsembleV2: a configurable, heterogeneous multi-model inference ensemble.

Unlike the v1 ``ensemble`` package, EnsembleV2 does NOT assume members share an
observation space or an action space. Every member is run on its own observation
(recomputed from the shared ``UniversalState`` by a :class:`MultiObservationSpace`)
and all cross-model comparison happens in the canonical ``UniversalAction`` index
space (the 13-wide ``DefaultActionSpace`` layout).

Each step we gather, for every member: the per-gamma legal-normalized action
distribution and the per-gamma Q-value of each action, all remapped onto canonical
universal action indices. These features (plus the full per-battle history) are
handed to a single pluggable ``make_ensembled_decision`` hook that returns the
chosen universal action. The baseline hook simply returns the anchor model's
argmax. All gathered features are logged to disk to enable future learned deciders.
"""

from metamon.rl.experimental.ensemblev2.config import (
    EnsembleV2MemberSpec,
    EnsembleV2Config,
    load_ensemblev2_presets,
    get_ensemblev2_preset,
    member_prefix,
)
from metamon.rl.experimental.ensemblev2.decision import (
    MemberStepFeatures,
    EnsembleDecisionContext,
    EnsembleDecision,
    register_ensemble_decision,
    get_ensemble_decision,
    get_ensemble_decision_names,
)

# Import concrete strategies so they register on package import.
from metamon.rl.experimental.ensemblev2 import deciders  # noqa: F401
from metamon.rl.experimental.ensemblev2.deciders import (
    AnchorPassthroughDecision,
    AnchorQArgmaxDecision,
    MeanProbDecision,
    SampleMeanProbDecision,
    MajorityVoteDecision,
    AnchorGatedMajorityDecision,
    HeuristicSafetyDecision,
    SafeAnchorDecision,
    SafeAnchorQDecision,
    SafeAnchorQGammaDecision,
    SafeAnchorQUnforcedSwitchDecision,
    SafeConsensusDecision,
    SafeSampleConsensusDecision,
    SafeMajorityDecision,
    SafeAnchorGatedMajorityDecision,
    SafeTeammatesDecision,
)
from metamon.rl.experimental.ensemblev2.runtime import (
    EnsembleV2Policy,
    build_ensemblev2_experiment,
)

__all__ = [
    "EnsembleV2MemberSpec",
    "EnsembleV2Config",
    "load_ensemblev2_presets",
    "get_ensemblev2_preset",
    "member_prefix",
    "MemberStepFeatures",
    "EnsembleDecisionContext",
    "EnsembleDecision",
    "register_ensemble_decision",
    "get_ensemble_decision",
    "get_ensemble_decision_names",
    "AnchorPassthroughDecision",
    "AnchorQArgmaxDecision",
    "MeanProbDecision",
    "SampleMeanProbDecision",
    "MajorityVoteDecision",
    "AnchorGatedMajorityDecision",
    "HeuristicSafetyDecision",
    "SafeAnchorDecision",
    "SafeAnchorQDecision",
    "SafeAnchorQGammaDecision",
    "SafeAnchorQUnforcedSwitchDecision",
    "SafeConsensusDecision",
    "SafeSampleConsensusDecision",
    "SafeMajorityDecision",
    "SafeAnchorGatedMajorityDecision",
    "SafeTeammatesDecision",
    "EnsembleV2Policy",
    "build_ensemblev2_experiment",
]
