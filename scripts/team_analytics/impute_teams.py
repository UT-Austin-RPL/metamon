"""
Team Imputation

Uses paired battle perspectives to fill in incomplete opponent team data.
"""

from typing import List, Dict
from collections import defaultdict
from .parser import BattleRecord


def impute_opponent_teams(records: List[BattleRecord]) -> List[BattleRecord]:
    """
    Impute missing opponent team data by matching battle_id pairs.

    For each battle, there should be 2 records (one from each player's perspective).
    We can use Player A's known full team to fill in Player B's opponent team, and vice versa.

    Args:
        records: List of BattleRecord objects

    Returns:
        List of BattleRecord objects with imputed opponent teams
    """
    # Group records by battle_id
    battles_by_id: Dict[str, List[BattleRecord]] = defaultdict(list)
    for record in records:
        battles_by_id[record.battle_id].append(record)

    imputed_records = []
    imputed_count = 0
    unmatched_count = 0

    for battle_id, battle_records in battles_by_id.items():
        if len(battle_records) == 2:
            # We have both perspectives - can impute
            record1, record2 = battle_records

            # Verify they're actually opponents (one should be WIN, one LOSS)
            if record1.result != record2.result:
                # Impute record1's opponent team from record2's player team
                if len(record1.opponent_team_species) < len(record2.player_team_species):
                    record1_updated = BattleRecord(
                        battle_id=record1.battle_id,
                        filename=record1.filename,
                        player_name=record1.player_name,
                        opponent_name=record1.opponent_name,
                        result=record1.result,
                        num_turns=record1.num_turns,
                        date=record1.date,
                        rating=record1.rating,
                        player_lead=record1.player_lead,
                        opponent_lead=record2.player_lead,  # Use record2's lead
                        player_team_species=record1.player_team_species,
                        opponent_team_species=record2.player_team_species,  # Full team!
                        player_team_hash=record1.player_team_hash,
                        opponent_team_hash=record2.player_team_hash,  # Correct hash
                        player_team_json=record1.player_team_json,
                        opponent_team_json=record2.player_team_json,
                    )
                    imputed_count += 1
                else:
                    record1_updated = record1

                # Impute record2's opponent team from record1's player team
                if len(record2.opponent_team_species) < len(record1.player_team_species):
                    record2_updated = BattleRecord(
                        battle_id=record2.battle_id,
                        filename=record2.filename,
                        player_name=record2.player_name,
                        opponent_name=record2.opponent_name,
                        result=record2.result,
                        num_turns=record2.num_turns,
                        date=record2.date,
                        rating=record2.rating,
                        player_lead=record2.player_lead,
                        opponent_lead=record1.player_lead,  # Use record1's lead
                        player_team_species=record2.player_team_species,
                        opponent_team_species=record1.player_team_species,  # Full team!
                        player_team_hash=record2.player_team_hash,
                        opponent_team_hash=record1.player_team_hash,  # Correct hash
                        player_team_json=record2.player_team_json,
                        opponent_team_json=record1.player_team_json,
                    )
                    imputed_count += 1
                else:
                    record2_updated = record2

                imputed_records.extend([record1_updated, record2_updated])
            else:
                # Both have same result - shouldn't happen, keep as-is
                imputed_records.extend(battle_records)
        else:
            # Unmatched - keep original
            imputed_records.extend(battle_records)
            unmatched_count += len(battle_records)

    print(f"Imputation results:")
    print(f"  Records with imputed opponent teams: {imputed_count}")
    print(f"  Unmatched records (no pairing): {unmatched_count}")
    print(f"  Total records: {len(imputed_records)}")

    return imputed_records
