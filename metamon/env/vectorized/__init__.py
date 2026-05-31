"""Vectorized Showdown simulation env for Metamon.

Runs many battles in parallel inside a single Node process hosting N
``BattleStream``s (see ``battle_host.js``) and batches the in-the-loop opponent's
neural-network inference across lanes. By default the evaluated agent plays Showdown
``p1``; pass ``eval_player_side=1`` for ``p2``. New code lives entirely under this
package and uses the installed ``pokemon-showdown`` npm package.
"""

from .lane import StreamBattleLane
from .amago_policy import AmagoLadderPolicyDriver, vectorized_ladder_eval
from .opponent import (
    AmagoBatchedOpponent,
    BatchedOpponent,
    ConfigBatchedOpponent,
    RandomBatchedOpponent,
)
from .sim_process import ShowdownSimProcess, ShowdownSimProcessError, make_sim_process
from .vector_env import (
    BattleAgainstMetamon,
    BattleAgainstOpponentPool,
    BattleShowdownVectorized,
    ShowdownEnv,
    VectorizedShowdownEnv,
)

__all__ = [
    "StreamBattleLane",
    "ShowdownSimProcess",
    "ShowdownSimProcessError",
    "make_sim_process",
    "BatchedOpponent",
    "RandomBatchedOpponent",
    "AmagoBatchedOpponent",
    "AmagoLadderPolicyDriver",
    "ConfigBatchedOpponent",
    "vectorized_ladder_eval",
    "VectorizedShowdownEnv",
    "ShowdownEnv",
    "BattleAgainstMetamon",
    "BattleAgainstOpponentPool",
    "BattleShowdownVectorized",
]
