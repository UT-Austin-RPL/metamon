"""
Gradio Web Interface for Team Analytics

Interactive dashboard for analyzing battle trajectories.
"""

import gradio as gr
import pandas as pd
from typing import Optional, List
from pathlib import Path
import tempfile

from .database import TeamAnalyticsDB
from .analytics import AnalyticsEngine
from .export import TeamExporter


class TeamAnalyticsApp:
    """Gradio application for team analytics."""

    def __init__(self, db: TeamAnalyticsDB):
        """
        Initialize app with database.

        Args:
            db: TeamAnalyticsDB instance
        """
        self.db = db
        self.analytics = AnalyticsEngine(db)
        self.exporter = TeamExporter()

        # Cache all species for dropdowns
        self.all_species = self.db.get_all_species()

    def get_database_overview(self):
        """Get database statistics for overview tab."""
        stats = self.db.get_database_stats()

        summary = f"""
# Database Overview

**Total Battles:** {stats['total_battles']:,}
**Unique Teams:** {stats['unique_player_teams']:,}
**Date Range:** {stats['date_range']['min']} to {stats['date_range']['max']}

## Top 10 Species by Usage

"""
        for species_data in stats['top_species']:
            summary += f"- **{species_data['species']}**: {species_data['count']:,} appearances\n"

        summary += "\n## Top 10 Leads by Usage\n\n"
        for lead_data in stats['top_leads']:
            summary += f"- **{lead_data['lead']}**: {lead_data['count']:,} games\n"

        return summary

    def query_team_performance(
        self,
        exclude_mirrors: bool,
        min_battles: int,
        limit: int
    ):
        """Query team performance."""
        df = self.analytics.win_rate_by_team(
            exclude_mirrors=exclude_mirrors,
            min_battles=min_battles,
            limit=limit
        )

        # Format team_species as readable string
        if not df.empty:
            df['team'] = df['team_species'].apply(lambda x: ', '.join(x))
            df['win_rate'] = df['win_rate'].apply(lambda x: f"{x:.1%}")
            df['avg_turns'] = df['avg_turns'].apply(lambda x: f"{x:.1f}")

            display_df = df[['team', 'wins', 'losses', 'total_battles', 'win_rate', 'avg_turns']]
            return display_df
        else:
            return pd.DataFrame()

    def query_archetype_performance(
        self,
        must_have: List[str],
        must_not_have: List[str],
        opp_must_have: List[str],
        opp_must_not_have: List[str],
        exclude_mirrors: bool
    ):
        """Query archetype performance."""
        result = self.analytics.win_rate_by_archetype(
            must_have=must_have if must_have else None,
            must_not_have=must_not_have if must_not_have else None,
            opp_must_have=opp_must_have if opp_must_have else None,
            opp_must_not_have=opp_must_not_have if opp_must_not_have else None,
            exclude_mirrors=exclude_mirrors
        )

        summary = f"""
## Archetype Performance

**Total Battles:** {result['total_battles']:,}
**Wins:** {result['wins']:,}
**Losses:** {result['losses']:,}
**Win Rate:** {result['win_rate']:.1%}
**Avg Turn Count:** {result['avg_turns']:.1f}

### Filters Applied:
- **Team must have:** {', '.join(must_have) if must_have else 'None'}
- **Team must NOT have:** {', '.join(must_not_have) if must_not_have else 'None'}
- **Opponent must have:** {', '.join(opp_must_have) if opp_must_have else 'None'}
- **Opponent must NOT have:** {', '.join(opp_must_not_have) if opp_must_not_have else 'None'}
- **Exclude mirrors:** {exclude_mirrors}
"""
        return summary

    def query_lead_performance(
        self,
        exclude_mirrors: bool,
        min_battles: int
    ):
        """Query lead performance."""
        df = self.analytics.win_rate_by_lead(
            exclude_mirrors=exclude_mirrors,
            min_battles=min_battles
        )

        if not df.empty:
            df['win_rate'] = df['win_rate'].apply(lambda x: f"{x:.1%}")
            df['avg_turns'] = df['avg_turns'].apply(lambda x: f"{x:.1f}")
            return df
        else:
            return pd.DataFrame()

    def query_species_usage(
        self,
        exclude_mirrors: bool
    ):
        """Query species usage statistics."""
        df = self.analytics.species_usage_stats(
            exclude_mirrors=exclude_mirrors
        )

        if not df.empty:
            df['avg_win_rate'] = df['avg_win_rate'].apply(lambda x: f"{x:.1%}")
            df['avg_turns'] = df['avg_turns'].apply(lambda x: f"{x:.1f}")
            return df
        else:
            return pd.DataFrame()

    def export_filtered_teams(
        self,
        must_have: List[str],
        must_not_have: List[str],
        exclude_mirrors: bool,
        min_win_rate: float,
        min_battles: int,
        limit: int
    ):
        """Export teams matching filters."""
        teams_df = self.analytics.get_teams_by_filter(
            must_have=must_have if must_have else None,
            must_not_have=must_not_have if must_not_have else None,
            exclude_mirrors=exclude_mirrors,
            min_win_rate=min_win_rate / 100.0,  # Convert from percentage
            min_battles=min_battles,
            limit=limit
        )

        if teams_df.empty:
            return None, "No teams match the specified filters."

        # Create temporary ZIP file
        temp_dir = tempfile.mkdtemp()
        output_file = Path(temp_dir) / "teams_export.zip"

        self.exporter.export_teams_to_zip(
            teams_df,
            str(output_file),
            include_csv=True
        )

        return str(output_file), f"Exported {len(teams_df)} teams to ZIP archive."

    def build_ui(self):
        """Build Gradio interface."""
        with gr.Blocks(title="Team Analytics Dashboard") as app:
            gr.Markdown("# Metamon Team Analytics Dashboard")
            gr.Markdown("Analyze Pokemon battle trajectories by team composition, archetype, and lead performance.")

            with gr.Tabs():
                # Tab 1: Overview
                with gr.Tab("Overview"):
                    gr.Markdown("## Database Statistics")
                    overview_text = gr.Markdown(self.get_database_overview())

                # Tab 2: Team Performance
                with gr.Tab("Team Performance"):
                    gr.Markdown("## Win Rate by Team Composition")

                    with gr.Row():
                        team_exclude_mirrors = gr.Checkbox(
                            label="Exclude Mirror Matches",
                            value=True
                        )
                        team_min_battles = gr.Slider(
                            minimum=1,
                            maximum=50,
                            value=5,
                            step=1,
                            label="Minimum Battles"
                        )
                        team_limit = gr.Slider(
                            minimum=10,
                            maximum=500,
                            value=100,
                            step=10,
                            label="Max Teams to Show"
                        )

                    team_query_btn = gr.Button("Run Query", variant="primary")
                    team_results = gr.Dataframe(label="Team Performance Results")

                    team_query_btn.click(
                        fn=self.query_team_performance,
                        inputs=[team_exclude_mirrors, team_min_battles, team_limit],
                        outputs=team_results
                    )

                # Tab 3: Archetype Analysis
                with gr.Tab("Archetype Analysis"):
                    gr.Markdown("## Performance by Team Archetype")
                    gr.Markdown("Define archetypes by species presence/absence and analyze win rates.")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Player Team Filters")
                            arch_must_have = gr.Dropdown(
                                choices=self.all_species,
                                multiselect=True,
                                label="Team MUST have these species"
                            )
                            arch_must_not_have = gr.Dropdown(
                                choices=self.all_species,
                                multiselect=True,
                                label="Team must NOT have these species"
                            )

                        with gr.Column():
                            gr.Markdown("### Opponent Team Filters")
                            arch_opp_must_have = gr.Dropdown(
                                choices=self.all_species,
                                multiselect=True,
                                label="Opponent MUST have these species"
                            )
                            arch_opp_must_not_have = gr.Dropdown(
                                choices=self.all_species,
                                multiselect=True,
                                label="Opponent must NOT have these species"
                            )

                    arch_exclude_mirrors = gr.Checkbox(
                        label="Exclude Mirror Matches",
                        value=True
                    )

                    arch_query_btn = gr.Button("Run Query", variant="primary")
                    arch_results = gr.Markdown(label="Archetype Performance")

                    arch_query_btn.click(
                        fn=self.query_archetype_performance,
                        inputs=[
                            arch_must_have,
                            arch_must_not_have,
                            arch_opp_must_have,
                            arch_opp_must_not_have,
                            arch_exclude_mirrors
                        ],
                        outputs=arch_results
                    )

                # Tab 4: Lead Analysis
                with gr.Tab("Lead Analysis"):
                    gr.Markdown("## Win Rate by Lead Pokemon")

                    with gr.Row():
                        lead_exclude_mirrors = gr.Checkbox(
                            label="Exclude Mirror Matches",
                            value=True
                        )
                        lead_min_battles = gr.Slider(
                            minimum=5,
                            maximum=100,
                            value=10,
                            step=5,
                            label="Minimum Battles"
                        )

                    lead_query_btn = gr.Button("Run Query", variant="primary")
                    lead_results = gr.Dataframe(label="Lead Performance Results")

                    lead_query_btn.click(
                        fn=self.query_lead_performance,
                        inputs=[lead_exclude_mirrors, lead_min_battles],
                        outputs=lead_results
                    )

                # Tab 5: Species Usage
                with gr.Tab("Species Usage"):
                    gr.Markdown("## Pokemon Usage Statistics")

                    species_exclude_mirrors = gr.Checkbox(
                        label="Exclude Mirror Matches",
                        value=True
                    )

                    species_query_btn = gr.Button("Run Query", variant="primary")
                    species_results = gr.Dataframe(label="Species Usage Results")

                    species_query_btn.click(
                        fn=self.query_species_usage,
                        inputs=[species_exclude_mirrors],
                        outputs=species_results
                    )

                # Tab 6: Export Teams
                with gr.Tab("Export Teams"):
                    gr.Markdown("## Export Filtered Teams")
                    gr.Markdown("Filter teams and export to ZIP archive (includes JSON files + CSV summary).")

                    with gr.Row():
                        export_must_have = gr.Dropdown(
                            choices=self.all_species,
                            multiselect=True,
                            label="Team MUST have these species"
                        )
                        export_must_not_have = gr.Dropdown(
                            choices=self.all_species,
                            multiselect=True,
                            label="Team must NOT have these species"
                        )

                    with gr.Row():
                        export_exclude_mirrors = gr.Checkbox(
                            label="Exclude Mirror Matches",
                            value=True
                        )
                        export_min_wr = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=50,
                            step=5,
                            label="Minimum Win Rate (%)"
                        )
                        export_min_battles = gr.Slider(
                            minimum=1,
                            maximum=50,
                            value=5,
                            step=1,
                            label="Minimum Battles"
                        )
                        export_limit = gr.Slider(
                            minimum=10,
                            maximum=1000,
                            value=100,
                            step=10,
                            label="Max Teams to Export"
                        )

                    export_btn = gr.Button("Export Teams", variant="primary")
                    export_file = gr.File(label="Download ZIP")
                    export_status = gr.Textbox(label="Status")

                    export_btn.click(
                        fn=self.export_filtered_teams,
                        inputs=[
                            export_must_have,
                            export_must_not_have,
                            export_exclude_mirrors,
                            export_min_wr,
                            export_min_battles,
                            export_limit
                        ],
                        outputs=[export_file, export_status]
                    )

        return app

    def launch(self, **kwargs):
        """Launch the Gradio app."""
        app = self.build_ui()
        app.launch(**kwargs)
