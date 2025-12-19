"""
Analytics Engine

Provides high-level query functions for team performance analysis.
"""

from typing import List, Optional, Dict, Any
import pandas as pd
from .database import TeamAnalyticsDB


class AnalyticsEngine:
    """High-level analytics queries with mirror match filtering."""

    def __init__(self, db: TeamAnalyticsDB):
        """
        Initialize analytics engine.

        Args:
            db: TeamAnalyticsDB instance
        """
        self.db = db
        self.conn = db.get_connection()

    def _build_archetype_filter(
        self,
        must_have: Optional[List[str]] = None,
        must_not_have: Optional[List[str]] = None,
        opp_must_have: Optional[List[str]] = None,
        opp_must_not_have: Optional[List[str]] = None,
    ) -> str:
        """
        Build WHERE clause for archetype filtering.

        Args:
            must_have: Player team must contain these species
            must_not_have: Player team must NOT contain these species
            opp_must_have: Opponent team must contain these species
            opp_must_not_have: Opponent team must NOT contain these species

        Returns:
            SQL WHERE clause string
        """
        conditions = []

        if must_have:
            for species in must_have:
                conditions.append(f"list_contains(player_team_species, '{species}')")

        if must_not_have:
            for species in must_not_have:
                conditions.append(f"NOT list_contains(player_team_species, '{species}')")

        if opp_must_have:
            for species in opp_must_have:
                conditions.append(f"list_contains(opponent_team_species, '{species}')")

        if opp_must_not_have:
            for species in opp_must_not_have:
                conditions.append(f"NOT list_contains(opponent_team_species, '{species}')")

        if conditions:
            return " AND " + " AND ".join(conditions)
        return ""

    def win_rate_by_team(
        self,
        exclude_mirrors: bool = True,
        min_battles: int = 5,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Calculate win rate for each unique team.

        Args:
            exclude_mirrors: Exclude mirror matches
            min_battles: Minimum battles required
            limit: Maximum number of teams to return

        Returns:
            DataFrame with columns: [team_hash, team_species, wins, losses, total_battles, win_rate, avg_turns]
        """
        mirror_filter = "AND player_team_hash != opponent_team_hash" if exclude_mirrors else ""

        query = f"""
            SELECT
                player_team_hash as team_hash,
                player_team_species as team_species,
                SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                COUNT(*) as total_battles,
                CAST(SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) AS FLOAT) /
                    NULLIF(SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) + SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END), 0) as win_rate,
                AVG(num_turns) as avg_turns
            FROM battles
            WHERE player_team_hash != 'empty'
            {mirror_filter}
            GROUP BY player_team_hash, player_team_species
            HAVING COUNT(*) >= {min_battles}
            ORDER BY win_rate DESC, total_battles DESC
            LIMIT {limit}
        """

        return self.conn.execute(query).df()

    def win_rate_by_archetype(
        self,
        must_have: Optional[List[str]] = None,
        must_not_have: Optional[List[str]] = None,
        opp_must_have: Optional[List[str]] = None,
        opp_must_not_have: Optional[List[str]] = None,
        exclude_mirrors: bool = True
    ) -> Dict[str, Any]:
        """
        Calculate win rate for teams matching archetype criteria.

        Returns:
            Dict with: win_rate, total_battles, wins, losses, avg_turns
        """
        archetype_filter = self._build_archetype_filter(
            must_have, must_not_have, opp_must_have, opp_must_not_have
        )
        mirror_filter = "AND player_team_hash != opponent_team_hash" if exclude_mirrors else ""

        query = f"""
            SELECT
                COUNT(*) as total_battles,
                SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                CAST(SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) AS FLOAT) /
                    NULLIF(SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) + SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END), 0) as win_rate,
                AVG(num_turns) as avg_turns
            FROM battles
            WHERE 1=1
            {archetype_filter}
            {mirror_filter}
        """

        result = self.conn.execute(query).fetchone()

        if result and result[0] > 0:
            return {
                'total_battles': result[0],
                'wins': result[1],
                'losses': result[2],
                'win_rate': result[3],
                'avg_turns': result[4],
            }
        else:
            return {
                'total_battles': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'avg_turns': 0.0,
            }

    def win_rate_by_lead(
        self,
        exclude_mirrors: bool = True,
        min_battles: int = 10
    ) -> pd.DataFrame:
        """
        Calculate win rate by lead Pokemon.

        Excludes battles where the opponent has the same lead
        to avoid 50% win rate bias from mirror matchups.

        Args:
            exclude_mirrors: Exclude mirror matches
            min_battles: Minimum battles required

        Returns:
            DataFrame with columns: [lead, wins, losses, total_battles, win_rate, avg_turns]
        """
        mirror_filter = "AND player_team_hash != opponent_team_hash" if exclude_mirrors else ""

        query = f"""
            SELECT
                player_lead as lead,
                SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                COUNT(*) as total_battles,
                CAST(SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) AS FLOAT) /
                    NULLIF(SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) + SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END), 0) as win_rate,
                AVG(num_turns) as avg_turns
            FROM battles
            WHERE player_lead != 'Unknown'
              AND player_lead != opponent_lead
            {mirror_filter}
            GROUP BY player_lead
            HAVING COUNT(*) >= {min_battles}
            ORDER BY win_rate DESC, total_battles DESC
        """

        return self.conn.execute(query).df()

    def matchup_matrix(
        self,
        archetypes: List[Dict[str, Any]],
        exclude_mirrors: bool = True
    ) -> pd.DataFrame:
        """
        Calculate win rate matrix for archetype matchups.

        Args:
            archetypes: List of archetype definitions, each with:
                - name: Display name
                - must_have: List of species (optional)
                - must_not_have: List of species (optional)
            exclude_mirrors: Exclude mirror matches

        Returns:
            DataFrame with archetypes as rows and columns, win rates as values
        """
        results = []

        for player_archetype in archetypes:
            row = {'archetype': player_archetype['name']}

            for opp_archetype in archetypes:
                # Build filters
                archetype_filter = self._build_archetype_filter(
                    must_have=player_archetype.get('must_have'),
                    must_not_have=player_archetype.get('must_not_have'),
                    opp_must_have=opp_archetype.get('must_have'),
                    opp_must_not_have=opp_archetype.get('must_not_have'),
                )
                mirror_filter = "AND player_team_hash != opponent_team_hash" if exclude_mirrors else ""

                query = f"""
                    SELECT
                        COUNT(*) as total,
                        CAST(SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) AS FLOAT) /
                            NULLIF(SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) + SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END), 0) as win_rate
                    FROM battles
                    WHERE 1=1
                    {archetype_filter}
                    {mirror_filter}
                """

                result = self.conn.execute(query).fetchone()
                win_rate = result[1] if result and result[0] > 0 else None

                row[opp_archetype['name']] = win_rate

            results.append(row)

        return pd.DataFrame(results)

    def get_teams_by_filter(
        self,
        must_have: Optional[List[str]] = None,
        must_not_have: Optional[List[str]] = None,
        exclude_mirrors: bool = True,
        min_win_rate: Optional[float] = None,
        max_win_rate: Optional[float] = None,
        min_battles: int = 5,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Get list of teams matching filter criteria.

        Returns:
            DataFrame with team details and performance stats
        """
        archetype_filter = self._build_archetype_filter(must_have, must_not_have)
        mirror_filter = "AND player_team_hash != opponent_team_hash" if exclude_mirrors else ""

        query = f"""
            WITH team_stats AS (
                SELECT
                    player_team_hash as team_hash,
                    player_team_species as team_species,
                    player_team_json,
                    COUNT(*) as total_battles,
                    SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                    CAST(SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) AS FLOAT) /
                        NULLIF(SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) + SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END), 0) as win_rate,
                    AVG(num_turns) as avg_turns
                FROM battles
                WHERE player_team_hash != 'empty'
                {mirror_filter}
                {archetype_filter}
                GROUP BY player_team_hash, player_team_species, player_team_json
                HAVING COUNT(*) >= {min_battles}
            )
            SELECT * FROM team_stats
            WHERE 1=1
        """

        if min_win_rate is not None:
            query += f" AND win_rate >= {min_win_rate}"

        if max_win_rate is not None:
            query += f" AND win_rate <= {max_win_rate}"

        query += f"""
            ORDER BY win_rate DESC, total_battles DESC
            LIMIT {limit}
        """

        return self.conn.execute(query).df()

    def species_usage_stats(
        self,
        exclude_mirrors: bool = True
    ) -> pd.DataFrame:
        """
        Get usage statistics for each species.

        Excludes battles where the opponent also has the target species
        to avoid 50% win rate bias from mirror matchups.

        Returns:
            DataFrame with: [species, appearances, avg_win_rate, avg_turns]
        """
        mirror_filter = "AND player_team_hash != opponent_team_hash" if exclude_mirrors else ""

        query = f"""
            WITH species_battles AS (
                SELECT
                    unnest(player_team_species) as species,
                    result,
                    num_turns,
                    opponent_team_species
                FROM battles
                WHERE 1=1
                {mirror_filter}
            )
            SELECT
                species,
                COUNT(*) as appearances,
                CAST(SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) AS FLOAT) /
                    NULLIF(SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) + SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END), 0) as avg_win_rate,
                AVG(num_turns) as avg_turns
            FROM species_battles
            WHERE species != 'Unknown'
              AND NOT list_contains(opponent_team_species, species)
            GROUP BY species
            ORDER BY appearances DESC
        """

        return self.conn.execute(query).df()

    def lead_matchup_matrix(
        self,
        exclude_mirrors: bool = True,
        min_battles: int = 10
    ) -> pd.DataFrame:
        """
        Calculate win rate matrix for lead matchups.

        Aggregates from both battle perspectives to ensure symmetry.
        For mirror matchups (same lead vs same lead), enforces 50% win rate.

        Returns:
            DataFrame with player leads as rows, opponent leads as columns
        """
        # Get top leads
        top_leads = self.conn.execute(f"""
            SELECT player_lead, COUNT(*) as count
            FROM battles
            WHERE player_lead != 'Unknown'
            GROUP BY player_lead
            ORDER BY count DESC
            LIMIT 20
        """).fetchall()

        lead_names = [row[0] for row in top_leads]

        results = []
        mirror_filter = "AND player_team_hash != opponent_team_hash" if exclude_mirrors else ""

        for player_lead in lead_names:
            row = {'player_lead': player_lead}

            for opp_lead in lead_names:
                # For mirror matchups, force 50% win rate
                if player_lead == opp_lead:
                    row[opp_lead] = 0.5
                else:
                    # Aggregate from BOTH perspectives to handle asymmetric data
                    query = f"""
                        WITH perspective_a AS (
                            SELECT
                                COUNT(*) as total,
                                SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                                SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses
                            FROM battles
                            WHERE player_lead = '{player_lead}'
                              AND opponent_lead = '{opp_lead}'
                              {mirror_filter}
                        ),
                        perspective_b AS (
                            SELECT
                                COUNT(*) as total,
                                SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as opp_wins,
                                SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as opp_losses
                            FROM battles
                            WHERE player_lead = '{opp_lead}'
                              AND opponent_lead = '{player_lead}'
                              {mirror_filter}
                        )
                        SELECT
                            COALESCE((SELECT wins FROM perspective_a), 0) + COALESCE((SELECT opp_losses FROM perspective_b), 0) as total_wins,
                            COALESCE((SELECT losses FROM perspective_a), 0) + COALESCE((SELECT opp_wins FROM perspective_b), 0) as total_losses,
                            COALESCE((SELECT total FROM perspective_a), 0) + COALESCE((SELECT total FROM perspective_b), 0) as total_battles
                    """

                    result = self.conn.execute(query).fetchone()
                    total_wins, total_losses, total_battles = result

                    if total_battles >= min_battles and (total_wins + total_losses) > 0:
                        win_rate = total_wins / (total_wins + total_losses)
                        row[opp_lead] = win_rate
                    else:
                        row[opp_lead] = None

            results.append(row)

        return pd.DataFrame(results)

    def get_lead_variations_for_teams(
        self,
        team_hashes: Optional[List[str]] = None,
        exclude_mirrors: bool = True
    ) -> pd.DataFrame:
        """
        Get lead variation statistics for teams.

        Shows which leads were used with each team and their performance.

        Args:
            team_hashes: Optional list of team hashes to filter (all teams if None)
            exclude_mirrors: Exclude mirror matches

        Returns:
            DataFrame with: [team_hash, team_species, player_lead, lead_usage_count, lead_win_rate]
        """
        mirror_filter = "AND player_team_hash != opponent_team_hash" if exclude_mirrors else ""
        team_filter = ""

        if team_hashes:
            # Safely escape team hashes for SQL
            hash_list = ", ".join([f"'{h}'" for h in team_hashes])
            team_filter = f"AND player_team_hash IN ({hash_list})"

        query = f"""
            SELECT
                player_team_hash as team_hash,
                player_team_species as team_species,
                player_lead,
                COUNT(*) as lead_usage_count,
                SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as lead_wins,
                CAST(SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) AS FLOAT) /
                    NULLIF(SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) + SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END), 0) as lead_win_rate
            FROM battles
            WHERE player_team_hash != 'empty'
              AND player_lead != 'Unknown'
            {mirror_filter}
            {team_filter}
            GROUP BY player_team_hash, player_team_species, player_lead
            ORDER BY player_team_hash, lead_usage_count DESC
        """

        return self.conn.execute(query).df()
