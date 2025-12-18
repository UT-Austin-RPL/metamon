"""
Team Analytics Tool for Metamon

Analyzes Pokémon battle trajectories to provide insights on:
- Win rates by team composition
- Performance by archetype (species presence/absence)
- Lead Pokemon performance
- Matchup matrices
"""

from .parser import TrajectoryParser, BattleRecord
from .database import TeamAnalyticsDB
from .analytics import AnalyticsEngine
from .export import TeamExporter

__all__ = [
    "TrajectoryParser",
    "BattleRecord",
    "TeamAnalyticsDB",
    "AnalyticsEngine",
    "TeamExporter",
]
