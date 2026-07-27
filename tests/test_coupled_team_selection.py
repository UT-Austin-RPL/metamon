"""Unit tests for coupled team-pool selection in vectorized Showdown.

Covers :func:`metamon.env.vectorized.team_adapter.coupled_player_specs` and the
:class:`~metamon.env.wrappers.WeightedMixedTeamSet` component-level methods that
back it. The goal of the coupling is that both sides of a collection battle draw
from the **same team pool** per battle, eliminating the composition mismatch
(learner draws ``smogon_pass2`` while an ``@schedule`` opponent independently
draws ``gl_05_26``) that previously inflated PSRO-Lite per-opponent win rates
relative to same-pool ladder play.

Stubs are used so the tests do not need real team files on disk.
"""

from __future__ import annotations

import os

from metamon.env.vectorized.team_adapter import coupled_player_specs
from metamon.env.wrappers import WeightedMixedTeamSet


class StubTeamSet:
    """Minimal stand-in for a plain (pinned) :class:`TeamSet`.

    Quacks like ``TeamSet`` for the attributes ``coupled_player_specs`` touches:
    ``yield_team`` (returns a packed team string), ``most_recent_team_file``
    (the source file path), and ``team_file_dir`` (the set's directory, used to
    match pools). Not a ``WeightedMixedTeamSet`` so it is treated as "pinned".
    """

    def __init__(self, set_name: str, team_label: str | None = None):
        self.team_file_dir = f"/cache/teams/{set_name}"
        self.battle_format = "gen1ou"
        label = team_label or set_name
        self._most_recent_team_file = f"/cache/teams/{set_name}/gen1ou/team_{label}.txt"
        self._team_label = label

    @property
    def most_recent_team_file(self) -> str:
        return self._most_recent_team_file

    def yield_team(self) -> str:
        return f"PACKED:{self._team_label}"

    def block_team(self, packed_team: str) -> bool:
        return False


def _mix(set_names, weights):
    """Build a real WeightedMixedTeamSet over stub components (no schedule)."""
    comps = [StubTeamSet(n) for n in set_names]
    return WeightedMixedTeamSet(team_sets=comps, weights=weights)


def _teamset_of_file(path):
    """Recover the teamset name from a stub team file path (mirrors the env)."""
    parts = path.replace("\\", "/").split("/")
    for idx, part in enumerate(parts):
        if part == "teams" and idx + 1 < len(parts):
            return parts[idx + 1]
    return None


# ---------------------------------------------------------------------------
# WeightedMixedTeamSet component-level methods
# ---------------------------------------------------------------------------


def test_sample_component_index_respects_weights():
    mix = _mix(["gl_05_26", "smogon_pass2", "smogon_pass2_selected"], [0.0, 1.0, 0.0])
    # All weight on index 1 → always samples component 1.
    for _ in range(50):
        assert mix.sample_component_index() == 1


def test_yield_team_from_component_sets_most_recent_file():
    mix = _mix(["gl_05_26", "smogon_pass2", "smogon_pass2_selected"], [0.0, 1.0, 0.0])
    team = mix.yield_team_from_component(2)
    assert team == "PACKED:smogon_pass2_selected"
    assert mix.most_recent_team_file.endswith(
        "smogon_pass2_selected/gen1ou/team_smogon_pass2_selected.txt"
    )


def test_index_for_team_file_dir_finds_matching_component():
    mix = _mix(["gl_05_26", "smogon_pass2", "smogon_pass2_selected"], [1.0, 1.0, 1.0])
    assert mix.index_for_team_file_dir("/cache/teams/smogon_pass2") == 1
    assert mix.index_for_team_file_dir("/cache/teams/gl_05_26") == 0
    assert mix.index_for_team_file_dir("/cache/teams/nonexistent") is None
    assert mix.index_for_team_file_dir(None) is None


def test_yield_team_uses_sample_and_from_component():
    mix = _mix(["gl_05_26", "smogon_pass2"], [0.0, 1.0])
    team = mix.yield_team()
    assert team == "PACKED:smogon_pass2"
    assert _teamset_of_file(mix.most_recent_team_file) == "smogon_pass2"


# ---------------------------------------------------------------------------
# coupled_player_specs: the core coupling behavior
# ---------------------------------------------------------------------------


def test_both_mix_draw_from_same_component():
    """The @schedule-vs-@schedule case: one component pick, both sides share it."""
    set_names = ["gl_05_26", "smogon_pass2", "smogon_pass2_selected"]
    # All weight on component 1 (smogon_pass2) so the pick is deterministic.
    eval_mix = _mix(set_names, [0.0, 1.0, 0.0])
    opp_mix = _mix(set_names, [1.0, 1.0, 1.0])  # weights irrelevant when coupled
    p1_spec, p1_file, p2_spec, p2_file = coupled_player_specs(
        "p1-0", "p2-0", eval_mix, opp_mix, "gen1ou", eval_side="p1"
    )
    # Both sides drew a team.
    assert "team" in p1_spec and "team" in p2_spec
    # Both files come from the SAME team pool (smogon_pass2).
    assert _teamset_of_file(p1_file) == "smogon_pass2"
    assert _teamset_of_file(p2_file) == "smogon_pass2"


def test_both_mix_curriculum_drives_the_pick():
    """The eval (learner) side's weights drive the component pick, not the opp's."""
    set_names = ["gl_05_26", "smogon_pass2", "smogon_pass2_selected"]
    # Eval weights force component 0 (gl_05_26); opp weights force component 2.
    eval_mix = _mix(set_names, [1.0, 0.0, 0.0])
    opp_mix = _mix(set_names, [0.0, 0.0, 1.0])
    _, p1_file, _, p2_file = coupled_player_specs(
        "p1-0", "p2-0", eval_mix, opp_mix, "gen1ou", eval_side="p1"
    )
    # Both follow the eval curriculum → gl_05_26, NOT the opp's component 2.
    assert _teamset_of_file(p1_file) == "gl_05_26"
    assert _teamset_of_file(p2_file) == "gl_05_26"


def test_eval_mix_opp_pinned_matches_opponent_pool():
    """Opponent pinned to one set: eval draws from the matching mix component."""
    set_names = ["gl_05_26", "smogon_pass2", "smogon_pass2_selected"]
    # Eval weights would pick gl_05_26 on their own, but the opponent is pinned
    # to smogon_pass2 → the eval must match the opponent's pool.
    eval_mix = _mix(set_names, [1.0, 0.0, 0.0])
    opp_pinned = StubTeamSet("smogon_pass2")
    _, p1_file, _, p2_file = coupled_player_specs(
        "p1-0", "p2-0", eval_mix, opp_pinned, "gen1ou", eval_side="p1"
    )
    assert _teamset_of_file(p1_file) == "smogon_pass2"
    assert _teamset_of_file(p2_file) == "smogon_pass2"


def test_eval_mix_opp_pinned_unmatched_falls_back_independently():
    """A pinned opponent whose set is NOT in the eval mix → independent draw."""
    eval_mix = _mix(["gl_05_26", "smogon_pass2"], [1.0, 0.0])
    opp_pinned = StubTeamSet("competitive")  # not one of the mix components
    _, p1_file, _, p2_file = coupled_player_specs(
        "p1-0", "p2-0", eval_mix, opp_pinned, "gen1ou", eval_side="p1"
    )
    # Eval falls back to its own weighted pick (gl_05_26); opp draws competitive.
    assert _teamset_of_file(p1_file) == "gl_05_26"
    assert _teamset_of_file(p2_file) == "competitive"


def test_opp_mix_eval_pinned_matches_eval_pool():
    """Symmetric: eval pinned, opponent is the mix → opponent matches eval's pool."""
    set_names = ["gl_05_26", "smogon_pass2", "smogon_pass2_selected"]
    opp_mix = _mix(set_names, [1.0, 1.0, 1.0])
    eval_pinned = StubTeamSet("smogon_pass2_selected")
    _, p1_file, _, p2_file = coupled_player_specs(
        "p1-0", "p2-0", eval_pinned, opp_mix, "gen1ou", eval_side="p1"
    )
    assert _teamset_of_file(p1_file) == "smogon_pass2_selected"
    assert _teamset_of_file(p2_file) == "smogon_pass2_selected"


def test_both_plain_independent_draws():
    """Both plain (e.g. validator: competitive vs competitive) → same pool already."""
    eval_plain = StubTeamSet("competitive", team_label="eval")
    opp_plain = StubTeamSet("competitive", team_label="opp")
    p1_spec, p1_file, p2_spec, p2_file = coupled_player_specs(
        "p1-0", "p2-0", eval_plain, opp_plain, "gen1ou", eval_side="p1"
    )
    assert p1_spec["team"] == "PACKED:eval"
    assert p2_spec["team"] == "PACKED:opp"
    # Same pool (competitive) for both, independently sampled.
    assert _teamset_of_file(p1_file) == "competitive"
    assert _teamset_of_file(p2_file) == "competitive"


def test_random_format_no_teams_drawn():
    """Random formats: the sim generates teams, so no team field on either side."""
    eval_mix = _mix(["gl_05_26", "smogon_pass2"], [1.0, 1.0])
    opp_mix = _mix(["gl_05_26", "smogon_pass2"], [1.0, 1.0])
    p1_spec, p1_file, p2_spec, p2_file = coupled_player_specs(
        "p1-0", "p2-0", eval_mix, opp_mix, "gen1randombattle", eval_side="p1"
    )
    assert "team" not in p1_spec
    assert "team" not in p2_spec
    assert p1_file is None and p2_file is None


def test_eval_side_assignment_p2():
    """When the eval side is p2, the eval team is assigned to the p2 spec/file."""
    set_names = ["gl_05_26", "smogon_pass2"]
    eval_mix = _mix(set_names, [0.0, 1.0])  # forces smogon_pass2
    opp_mix = _mix(set_names, [1.0, 1.0])
    p1_spec, p1_file, p2_spec, p2_file = coupled_player_specs(
        "p1-0", "p2-0", eval_mix, opp_mix, "gen1ou", eval_side="p2"
    )
    # eval is p2 → the p2 file is the learner's (smogon_pass2) draw.
    assert _teamset_of_file(p2_file) == "smogon_pass2"
    # p1 is the opponent, also smogon_pass2 (coupled).
    assert _teamset_of_file(p1_file) == "smogon_pass2"


def test_none_team_set_handled():
    """A None team set on one side must not crash (independent draw for the other)."""
    eval_plain = StubTeamSet("competitive")
    p1_spec, p1_file, p2_spec, p2_file = coupled_player_specs(
        "p1-0", "p2-0", eval_plain, None, "gen1ou", eval_side="p1"
    )
    assert p1_spec["team"] == "PACKED:competitive"
    assert "team" not in p2_spec
    assert p2_file is None
