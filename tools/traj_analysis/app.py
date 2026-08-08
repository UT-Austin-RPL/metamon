"""Gradio demo for the squirtle agent's ladder battles.

Shows:
  Tab 1 — Team performance vs opponent strength (win-rate heatmap + bars).
  Tab 2 — Per-battle model evaluation curve (V(s) per turn, delta, pokemon remaining).
  Tab 3 — Aggregate eval summaries: by turn number and by pokemon remaining per side.

Data: cache produced by build_cache.py (battles.parquet, turns.parquet, values.npz).

Usage:
  .venv/bin/python tools/traj_analysis/app.py [--cache DIR] [--port 7860]
"""

import argparse
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import gradio as gr

DEFAULT_CACHE = os.path.expanduser("~/metamon/trajectories/squirtle/eval_cache")
GAMMA_MAIN = 6  # gamma = 0.999

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_CACHE_DIR = None
_BATTLES = None
_TURNS = None
_VALUES = None

TEAM_INFO = {
    "team_0180": (
        "Triple Boom",
        "Jynx / Chansey / Snorlax / Gengar / Starmie / Tauros",
        "Triple Self-Destruct+Explosion, Rest Tauros",
    ),
    "team_0208": (
        "Clamp Cloyster double-sleep",
        "Jynx / Cloyster / Snorlax / Chansey / Tauros / Alakazam",
        "Clamp Cloyster trapping, Sing + Lovely Kiss double sleep",
    ),
    "team_0232": (
        "Reflect Snorlax (best vs TaurosV0)",
        "Gengar / Jynx / Chansey / Snorlax / Alakazam / Tauros",
        "Reflect Snorlax, Rest Tauros",
    ),
    "team_0259": (
        "Reflect Snorlax + dual Rest",
        "Jynx / Cloyster / Snorlax / Chansey / Tauros / Alakazam",
        "Reflect Snorlax, Rest on Lax+Tauros, dual Ice",
    ),
    "team_0353": (
        "Rhydon SubRock",
        "Jynx / Starmie / Chansey / Snorlax / Tauros / Rhydon",
        "Rhydon Substitute+Rock Slide, Sing Chansey, Reflect Lax",
    ),
}

_TIER_CACHE = {}  # opponent -> tier


def load_cache(cache_dir):
    global _CACHE_DIR, _BATTLES, _TURNS, _VALUES
    _CACHE_DIR = cache_dir
    _BATTLES = pd.read_parquet(os.path.join(cache_dir, "battles.parquet"))
    _TURNS = pd.read_parquet(os.path.join(cache_dir, "turns.parquet"))
    _VALUES = np.load(os.path.join(cache_dir, "values.npz"), allow_pickle=True)


def battles() -> pd.DataFrame:
    return _BATTLES


def turns() -> pd.DataFrame:
    return _TURNS


def values():
    return _VALUES


def opponent_tier(opponent: str) -> str:
    """Tier an opponent by the squirtle agent's Laplace-smoothed win rate vs them."""
    if opponent in _TIER_CACHE:
        return _TIER_CACHE[opponent]
    b = _BATTLES
    wins = int(((b.opponent == opponent) & (b.result == "WIN")).sum())
    games = int((b.opponent == opponent).sum())
    wr = (wins + 1.0) / (games + 2.0)  # Laplace alpha=1
    if wr < 0.45:
        tier = "Strong opponent"
    elif wr < 0.60:
        tier = "Mid opponent"
    else:
        tier = "Weak opponent"
    _TIER_CACHE[opponent] = tier
    return tier


def team_label(team: str) -> str:
    if team in TEAM_INFO:
        return f"{team} · {TEAM_INFO[team][0]}"
    return team or "unknown"


TIER_ORDER = ["Weak opponent", "Mid opponent", "Strong opponent"]


# ---------------------------------------------------------------------------
# Tab 1 — Team x opponent-strength performance
# ---------------------------------------------------------------------------


def _team_tier_table():
    b = _BATTLES.copy()
    b["tier"] = b.opponent.map(opponent_tier)
    b["team_label"] = b.team.map(team_label)
    return b


def plot_heatmap(battles_sub: pd.DataFrame) -> go.Figure:
    """Win-rate heatmap: team rows x opponent-strength columns."""
    b = battles_sub.copy()
    b["tier"] = b.opponent.map(opponent_tier)
    team_labels = sorted(b.team.map(team_label).unique())
    tiers = TIER_ORDER
    matrix = np.full((len(team_labels), len(tiers)), np.nan)
    counts = np.full((len(team_labels), len(tiers)), 0)
    for i, tl in enumerate(team_labels):
        for j, t in enumerate(tiers):
            sub = b[(b.team_label == tl) & (b.tier == t)]
            if len(sub):
                matrix[i, j] = 100.0 * (sub.result == "WIN").mean()
                counts[i, j] = len(sub)
    text = [
        [
            f"{matrix[i, j]:.0f}%" if not np.isnan(matrix[i, j]) else ""
            for j in range(len(tiers))
        ]
        for i in range(len(team_labels))
    ]
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=tiers,
            y=team_labels,
            zmin=0,
            zmax=100,
            colorscale="RdYlGn",
            text=text,
            texttemplate="%{text}",
            colorbar=dict(title="Win rate %"),
            hovertemplate="team=%{y}<br>opponent tier=%{x}<br>win rate=%{z:.1f}%<br>games=%{customdata}<extra></extra>",
            customdata=counts,
        )
    )
    fig.update_layout(
        title="Squirtle win rate by team vs opponent strength",
        xaxis_title="Opponent strength (agent win-rate based)",
        yaxis_title="Team",
        height=420,
        margin=dict(t=60, b=40, l=10, r=10),
        template="plotly_white",
    )
    return fig


def plot_team_bars(battles_sub: pd.DataFrame) -> go.Figure:
    b = battles_sub.copy()
    agg = b.groupby(b.team.map(team_label)).agg(
        games=("file", "count"),
        wins=("result", lambda r: (r == "WIN").sum()),
    )
    agg["winrate"] = 100.0 * agg.wins / agg.games
    agg = agg.sort_values("winrate")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=agg.winrate,
            y=agg.index,
            orientation="h",
            text=[f"{w:.1f}%" for w in agg.winrate],
            textposition="outside",
            marker_color=["#c0392b" if w < 50 else "#27ae60" for w in agg.winrate],
            customdata=agg.games,
            hovertemplate="team=%{y}<br>win rate=%{x:.1f}%<br>games=%{customdata}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Win rate by team (all battles)",
        xaxis=dict(title="Win rate %", range=[0, 100]),
        yaxis_title="",
        height=420,
        margin=dict(t=60, b=40, l=10, r=10),
        template="plotly_white",
    )
    return fig


def tab1_heatmap(team_filter, tier_filter):
    b = _team_tier_table()
    if team_filter != "All teams":
        b = b[b.team_label == team_filter]
    if tier_filter != "All tiers":
        b = b[b.tier == tier_filter]
    return plot_heatmap(b)


def tab1_bars(team_filter, tier_filter):
    b = _team_tier_table()
    if team_filter != "All teams":
        b = b[b.team_label == team_filter]
    if tier_filter != "All tiers":
        b = b[b.tier == tier_filter]
    return plot_team_bars(b)


def tab1_table(team_filter, tier_filter):
    b = _team_tier_table()
    if team_filter != "All teams":
        b = b[b.team_label == team_filter]
    if tier_filter != "All tiers":
        b = b[b.tier == tier_filter]
    agg = (
        b.groupby(["team_label", "tier"], dropna=False)
        .agg(
            battles=("file", "count"),
            wins=("result", lambda r: (r == "WIN").sum()),
        )
        .reset_index()
    )
    agg["win_rate"] = (100.0 * agg.wins / agg.battles).round(1)
    agg["losses"] = agg.battles - agg.wins
    agg = agg.sort_values(["tier", "team_label"])
    return agg[["team_label", "tier", "battles", "wins", "losses", "win_rate"]]


def tab1_kpis(team_filter, tier_filter):
    b = _team_tier_table()
    if team_filter != "All teams":
        b = b[b.team_label == team_filter]
    if tier_filter != "All tiers":
        b = b[b.tier == tier_filter]
    n = len(b)
    wr = 100.0 * (b.result == "WIN").sum() / n if n else 0.0
    return (
        n,
        round(wr, 1),
        int((b.result == "WIN").sum()),
        int((b.result == "LOSS").sum()),
    )


# ---------------------------------------------------------------------------
# Tab 2 — per-battle eval curve
# ---------------------------------------------------------------------------


def battle_options(team_filter, tier_filter, result_filter):
    b = _team_tier_table()
    if team_filter != "All teams":
        b = b[b.team_label == team_filter]
    if tier_filter != "All tiers":
        b = b[b.tier == tier_filter]
    if result_filter != "All results":
        b = b[b.result == result_filter]
    b = b.sort_values("file")
    opts = []
    for _, row in b.iterrows():
        label = (
            f"{row.opponent} · {row.result} · {team_label(row.team)} · "
            f"{row.n_turns} turns · {row.file[:38]}"
        )
        opts.append((label, row.file))
    return opts


def _turns_for(file: str) -> pd.DataFrame:
    return _TURNS[_TURNS.file == file].reset_index(drop=True)


def _battle_row(file: str) -> pd.Series:
    return _BATTLES[_BATTLES.file == file].iloc[0]


def plot_battle(
    file: str,
    gamma_idx: int,
    show_delta: bool,
    show_q: bool,
    show_adv: bool,
    show_remaining: bool,
    show_mean: bool,
) -> go.Figure:
    b = _battle_row(file)
    tdf = _turns_for(file)
    L = int(b.n_turns)
    v_s = _VALUES["v_s"][_BATTLES.index[_BATTLES.file == file][0], :L, :]
    q_sa = _VALUES["q_sa"][_BATTLES.index[_BATTLES.file == file][0], :L, :]
    adv = _VALUES["advantage"][_BATTLES.index[_BATTLES.file == file][0], :L, :]
    gammas = _VALUES["gammas"]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    x = np.arange(L)

    # main eval line
    fig.add_trace(
        go.Scatter(
            x=x,
            y=v_s[:, gamma_idx],
            name=f"V(s) γ={gammas[gamma_idx]:.3f}",
            line=dict(color="#1f77b4", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(31,119,180,0.08)",
            hovertemplate="turn %{x}<br>V(s)=%{y:.1f}<extra></extra>",
        ),
        secondary_y=False,
    )
    if show_mean:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=v_s.mean(1),
                name="V(s) mean γ",
                line=dict(color="#7f7f7f", width=1.2, dash="dot"),
                hovertemplate="turn %{x}<br>mean V=%{y:.1f}<extra></extra>",
            ),
            secondary_y=False,
        )

    if show_delta:
        dv = np.concatenate([[0.0], np.diff(v_s[:, gamma_idx])])
        colors = ["#d62728" if d < 0 else "#2ca02c" for d in dv]
        fig.add_trace(
            go.Bar(
                x=x,
                y=dv,
                name="ΔV per turn",
                marker_color=colors,
                opacity=0.55,
                hovertemplate="turn %{x}<br>ΔV=%{y:.1f}<extra></extra>",
            ),
            secondary_y=False,
        )

    if show_q:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=q_sa[:, gamma_idx],
                name="Q(s,a)",
                line=dict(color="#9467bd", width=1.4),
                hovertemplate="turn %{x}<br>Q(s,a)=%{y:.1f}<extra></extra>",
            ),
            secondary_y=False,
        )
    if show_adv:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=adv[:, gamma_idx],
                name="Advantage",
                line=dict(color="#e377c2", width=1.4),
                hovertemplate="turn %{x}<br>A(s,a)=%{y:.1f}<extra></extra>",
            ),
            secondary_y=False,
        )

    if show_remaining:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=tdf.player_remaining,
                name="Squirtle remaining",
                line=dict(color="#f39c12", width=2, shape="hv"),
                hovertemplate="turn %{x}<br>player remaining=%{y}<extra></extra>",
            ),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=tdf.opp_remaining,
                name="Opponent remaining",
                line=dict(color="#555555", width=2, dash="dash", shape="hv"),
                hovertemplate="turn %{x}<br>opponent remaining=%{y}<extra></extra>",
            ),
            secondary_y=True,
        )

    result_color = "#27ae60" if b.result == "WIN" else "#c0392b"
    fig.add_annotation(
        x=L - 1,
        y=v_s[-1, gamma_idx],
        text=f"{b.result} ({L} turns)",
        showarrow=True,
        arrowhead=2,
        ax=40,
        ay=-30,
        font=dict(size=13, color=result_color),
    )
    fig.update_layout(
        title=f"{b.opponent} — {b.result} — {team_label(b.team)}",
        height=520,
        margin=dict(t=70, b=40, l=10, r=10),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(title_text="turn")
    fig.update_yaxes(title_text="model evaluation V(s)", secondary_y=False)
    fig.update_yaxes(
        title_text="pokemon remaining", secondary_y=True, range=[-0.5, 6.5], dtick=1
    )
    return fig


def battle_info(file: str):
    b = _battle_row(file)
    tdf = _turns_for(file)
    tier = opponent_tier(b.opponent)
    info = TEAM_INFO.get(b.team)
    comp = info[1] if info else b.roster.replace(",", " / ")
    tech = info[2] if info else ""
    dropped = (tdf.opp_remaining.diff() < 0).sum()
    fainted = (tdf.player_remaining.diff() < 0).sum()
    lines = [
        f"**Battle:** `{b.file}`",
        f"**Opponent:** {b.opponent} — *{tier}*",
        f"**Result:** {b.result} ({b.n_turns} turns)",
        f"**Team:** {team_label(b.team)}",
        f"**Composition:** {comp}",
        f"**Tech:** {tech}",
        f"**KOs dealt / taken:** {int(dropped)} / {int(fainted)}",
        f"**V(s₀) → V(s_final):** {b.v0:.0f} → {b.v_final:.0f}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tab 3 — aggregate summaries
# ---------------------------------------------------------------------------


def _agg_frame(team_filter, tier_filter, result_filter, gamma_kind):
    t = _TURNS.copy()
    t["tier"] = t.opponent.map(opponent_tier)
    t["team_label"] = t.team.map(team_label)
    if team_filter != "All teams":
        t = t[t.team_label == team_filter]
    if tier_filter != "All tiers":
        t = t[t.tier == tier_filter]
    if result_filter != "All results":
        t = t[t.result == result_filter]
    return t


def plot_by_turn(
    team_filter,
    tier_filter,
    result_filter,
    gamma_kind,
    max_turn,
    split_result,
    show_band,
) -> go.Figure:
    t = _agg_frame(team_filter, tier_filter, result_filter, gamma_kind)
    col = "v_main" if gamma_kind == "γ=0.999" else "v_mean"
    t = t[t.turn <= max_turn]
    fig = go.Figure()
    groups = [("All", None)] if not split_result else [("WIN", "WIN"), ("LOSS", "LOSS")]
    colors = {"WIN": "#27ae60", "LOSS": "#c0392b", "All": "#1f77b4"}
    for label, res in groups:
        sub = t if res is None else t[t.result == res]
        if len(sub) == 0:
            continue
        agg = sub.groupby("turn")[col].agg(["mean", "std", "count"])
        x = agg.index
        fig.add_trace(
            go.Scatter(
                x=x,
                y=agg["mean"],
                name=label,
                mode="lines",
                line=dict(color=colors[label], width=2.5),
                hovertemplate="turn %{x}<br>mean eval=%{y:.1f}<extra></extra>",
            )
        )
        if show_band and len(agg) > 1:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=agg["mean"] + agg["std"],
                    mode="lines",
                    line=dict(width=0, color=colors[label]),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=agg["mean"] - agg["std"],
                    mode="lines",
                    line=dict(width=0, color=colors[label]),
                    fill="tonexty",
                    fillcolor=f"rgba({int(colors[label][1:3],16)},{int(colors[label][3:5],16)},{int(colors[label][5:7],16)},0.15)",
                    name=f"{label} ±1σ",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
    fig.update_layout(
        title=f"Mean model evaluation by turn ({gamma_kind})",
        xaxis_title="turn",
        yaxis_title="model evaluation",
        height=460,
        margin=dict(t=60, b=40, l=10, r=10),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def plot_delta_by_turn(
    team_filter, tier_filter, result_filter, max_turn, split_result
) -> go.Figure:
    t = _agg_frame(team_filter, tier_filter, result_filter, "γ=0.999")
    t = t[t.turn <= max_turn].sort_values(["file", "turn"])
    t["dv"] = t.groupby("file")["v_main"].diff().fillna(0.0)
    fig = go.Figure()
    groups = [("All", None)] if not split_result else [("WIN", "WIN"), ("LOSS", "LOSS")]
    colors = {"WIN": "#27ae60", "LOSS": "#c0392b", "All": "#1f77b4"}
    for label, res in groups:
        sub = t if res is None else t[t.result == res]
        agg = sub.groupby("turn")["dv"].agg(["mean"])
        fig.add_trace(
            go.Scatter(
                x=agg.index,
                y=agg["mean"],
                name=f"ΔV {label}",
                mode="lines",
                line=dict(color=colors[label], width=1.8),
                hovertemplate="turn %{x}<br>mean ΔV=%{y:.2f}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Mean per-turn change in model evaluation (γ=0.999)",
        xaxis_title="turn",
        yaxis_title="mean ΔV (this turn − previous)",
        height=420,
        margin=dict(t=60, b=40, l=10, r=10),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def plot_remaining_heatmap(
    team_filter, tier_filter, result_filter, gamma_kind, metric
) -> go.Figure:
    t = _agg_frame(team_filter, tier_filter, result_filter, gamma_kind)
    col = "v_main" if gamma_kind == "γ=0.999" else "v_mean"
    t = t.dropna(subset=["player_remaining", "opp_remaining"])
    if metric == "Mean eval":
        agg = t.groupby(["player_remaining", "opp_remaining"])[col].mean().unstack()
        title = f"Mean model evaluation by pokemon remaining ({gamma_kind})"
        zmin, zmax = None, None
        colorscale = "RdYlGn"
    elif metric == "Win rate":
        wins = (
            t[t.result == "WIN"].groupby(["player_remaining", "opp_remaining"]).size()
        )
        tot = t.groupby(["player_remaining", "opp_remaining"]).size()
        agg = (100.0 * wins / tot).unstack()
        title = "Win rate % by pokemon remaining"
        zmin, zmax = 0, 100
        colorscale = "RdYlGn"
    else:  # counts
        agg = t.groupby(["player_remaining", "opp_remaining"]).size().unstack()
        title = "Turn counts by pokemon remaining"
        zmin, zmax = None, None
        colorscale = "Blues"
    return _heatmap_fig(
        agg,
        title,
        "squirtle remaining",
        "opponent remaining",
        zmin=zmin,
        zmax=zmax,
        colorscale=colorscale,
    )


def _heatmap_fig(
    agg: pd.DataFrame, title, ylab, xlab, zmin=None, zmax=None, colorscale="RdYlGn"
) -> go.Figure:
    agg = agg.reindex(index=range(6, 0, -1), columns=range(1, 7))
    text = [
        [
            f"{agg.loc[i, j]:.0f}" if not pd.isna(agg.loc[i, j]) else ""
            for j in range(1, 7)
        ]
        for i in range(6, 0, -1)
    ]
    fig = go.Figure(
        data=go.Heatmap(
            z=agg.values,
            x=list(range(1, 7)),
            y=list(range(6, 0, -1)),
            zmin=zmin,
            zmax=zmax,
            colorscale=colorscale,
            text=text,
            texttemplate="%{text}",
            hovertemplate=f"{xlab}=%{{x}}<br>{ylab}=%{{y}}<br>value=%{{z:.1f}}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=xlab.capitalize(),
        yaxis_title=ylab.capitalize(),
        height=460,
        margin=dict(t=60, b=40, l=10, r=10),
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def build_app():
    team_choices = ["All teams"] + sorted(battles().team.map(team_label).unique())
    tier_choices = ["All tiers"] + TIER_ORDER
    result_choices = ["All results", "WIN", "LOSS"]

    with gr.Blocks(title="Squirtle Ladder Battles — Analysis") as demo:
        gr.Markdown(
            "# Squirtle vs. humans — ladder battle analysis\n"
            f"**{len(battles())} battles** on the Showdown ladder using the 5 **smog_ladder** teams, "
            f"**{int((battles().result == 'WIN').sum())} wins / {int((battles().result == 'LOSS').sum())} losses** "
            f"({100.0 * (battles().result == 'WIN').mean():.1f}%), vs **{battles().opponent.nunique()}** humans. "
            "Model evaluation = the squirtle agent's value estimate V(s) at each position (γ=0.999 unless noted)."
        )

        with gr.Tab("Teams vs opponent strength"):
            with gr.Row():
                t1_team = gr.Dropdown(team_choices, value="All teams", label="Team")
                t1_tier = gr.Dropdown(
                    tier_choices, value="All tiers", label="Opponent strength"
                )
            with gr.Row():
                kpi_games = gr.Number(label="Battles", info="*filtered*")
                kpi_wr = gr.Number(label="Win rate %")
                kpi_wins = gr.Number(label="Wins")
                kpi_losses = gr.Number(label="Losses")
            with gr.Row():
                heat = gr.Plot(label="Win rate heatmap")
                bars = gr.Plot(label="Win rate by team")
            with gr.Row():
                table = gr.Dataframe(label="Breakdown", interactive=False)

            def _refresh(tm, ti):
                k = tab1_kpis(tm, ti)
                return tab1_heatmap(tm, ti), tab1_bars(tm, ti), tab1_table(tm, ti), *k

            for ctrl in (t1_team, t1_tier):
                ctrl.change(
                    _refresh,
                    [t1_team, t1_tier],
                    [heat, bars, table, kpi_games, kpi_wr, kpi_wins, kpi_losses],
                )
            demo.load(
                _refresh,
                [t1_team, t1_tier],
                [heat, bars, table, kpi_games, kpi_wr, kpi_wins, kpi_losses],
            )

        with gr.Tab("Single battle evaluation"):
            with gr.Row():
                t2_team = gr.Dropdown(team_choices, value="All teams", label="Team")
                t2_tier = gr.Dropdown(
                    tier_choices, value="All tiers", label="Opponent strength"
                )
                t2_result = gr.Dropdown(
                    result_choices, value="All results", label="Result"
                )
            battle_sel = gr.Dropdown(
                label="Battle",
                scale=3,
                choices=battle_options("All teams", "All tiers", "All results"),
                value=None,
            )
            with gr.Row():
                gamma_idx = gr.Slider(
                    0,
                    6,
                    value=6,
                    step=1,
                    label=f"gamma index (0..6: "
                    f"{np.round(values()['gammas'], 3).tolist()})",
                )
                show_delta = gr.Checkbox(value=True, label="Show ΔV (per-turn change)")
                show_q = gr.Checkbox(value=False, label="Show Q(s,a)")
                show_adv = gr.Checkbox(value=False, label="Show advantage")
                show_rem = gr.Checkbox(value=True, label="Show pokemon remaining")
                show_mean = gr.Checkbox(value=False, label="Show mean over gammas")
            info_md = gr.Markdown()
            battle_plot = gr.Plot()

            def _update_options(tm, ti, tr):
                return gr.Dropdown(choices=battle_options(tm, ti, tr), value=None)

            for ctrl in (t2_team, t2_tier, t2_result):
                ctrl.change(_update_options, [t2_team, t2_tier, t2_result], battle_sel)
            demo.load(_update_options, [t2_team, t2_tier, t2_result], battle_sel)

            def _plot(file, gi, d, q, a, r, m):
                if file is None:
                    return "Select a battle.", None
                return battle_info(file), plot_battle(file, int(gi), d, q, a, r, m)

            battle_sel.change(
                _plot,
                [
                    battle_sel,
                    gamma_idx,
                    show_delta,
                    show_q,
                    show_adv,
                    show_rem,
                    show_mean,
                ],
                [info_md, battle_plot],
            )
            for ctrl in (gamma_idx, show_delta, show_q, show_adv, show_rem, show_mean):
                ctrl.change(
                    _plot,
                    [
                        battle_sel,
                        gamma_idx,
                        show_delta,
                        show_q,
                        show_adv,
                        show_rem,
                        show_mean,
                    ],
                    [info_md, battle_plot],
                )

        with gr.Tab("Aggregate evaluation"):
            with gr.Row():
                t3_team = gr.Dropdown(team_choices, value="All teams", label="Team")
                t3_tier = gr.Dropdown(
                    tier_choices, value="All tiers", label="Opponent strength"
                )
                t3_result = gr.Dropdown(
                    result_choices, value="All results", label="Result"
                )
            with gr.Row():
                t3_gamma = gr.Radio(
                    ["γ=0.999", "mean γ"], value="γ=0.999", label="Evaluation"
                )
                t3_split = gr.Checkbox(value=True, label="Split wins/losses")
                t3_band = gr.Checkbox(value=True, label="Show ±1σ band")
                t3_maxturn = gr.Slider(10, 250, value=150, step=5, label="Max turn")
            by_turn = gr.Plot(label="Mean eval by turn")
            delta_turn = gr.Plot(label="Mean ΔV by turn")
            with gr.Row():
                t3_metric = gr.Radio(
                    ["Mean eval", "Win rate", "Turn counts"],
                    value="Mean eval",
                    label="Metric",
                )
                rem_heat = gr.Plot(label="By pokemon remaining")

            def _agg_turn(tm, ti, tr, g, sp, band, mt):
                return (
                    plot_by_turn(tm, ti, tr, g, mt, sp, band),
                    plot_delta_by_turn(tm, ti, tr, mt, sp),
                )

            def _rem(tm, ti, tr, g, metric):
                return plot_remaining_heatmap(tm, ti, tr, g, metric)

            all_t3 = [
                t3_team,
                t3_tier,
                t3_result,
                t3_gamma,
                t3_split,
                t3_band,
                t3_maxturn,
            ]
            for ctrl in all_t3:
                ctrl.change(_agg_turn, all_t3, [by_turn, delta_turn])
            for ctrl in (t3_team, t3_tier, t3_result, t3_gamma, t3_metric):
                ctrl.change(
                    _rem, [t3_team, t3_tier, t3_result, t3_gamma, t3_metric], rem_heat
                )
            demo.load(_agg_turn, all_t3, [by_turn, delta_turn])
            demo.load(
                _rem, [t3_team, t3_tier, t3_result, t3_gamma, t3_metric], rem_heat
            )

    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    load_cache(args.cache)
    demo = build_app()
    demo.queue().launch(
        server_port=args.port, share=args.share, show_error=True, theme=gr.themes.Soft()
    )


if __name__ == "__main__":
    main()
