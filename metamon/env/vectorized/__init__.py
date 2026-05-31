"""Vectorized Showdown simulation env for Metamon.

Runs many battles in parallel inside a single Node process hosting N
``BattleStream``s (see ``battle_host.js``) and batches the in-the-loop opponent's
neural-network inference across lanes. The evaluated agent plays p1; the opponent
plays p2. New code lives entirely under this package; it imports the installed
``pokemon-showdown`` npm package and does not touch ``server/pokemon-showdown``.
"""

from .lane import StreamBattleLane
from .opponent import (
    AmagoBatchedOpponent,
    BatchedOpponent,
    RandomBatchedOpponent,
)
from .sim_process import ShowdownSimProcess, ShowdownSimProcessError
from .vector_env import BattleShowdownVectorized, VectorizedShowdownEnv

__all__ = [
    "StreamBattleLane",
    "ShowdownSimProcess",
    "ShowdownSimProcessError",
    "BatchedOpponent",
    "RandomBatchedOpponent",
    "AmagoBatchedOpponent",
    "VectorizedShowdownEnv",
    "BattleShowdownVectorized",
]
