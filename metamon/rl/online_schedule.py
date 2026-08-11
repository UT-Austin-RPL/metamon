"""Team-mix schedule glue for online RL — epoch-driven curriculum without restarts.

Reconstructed integration that wires :class:`~metamon.env.wrappers.TeamMixSchedule`
+ :class:`~metamon.env.wrappers.EpochRef` into the online RL collector, so the
player team set and any opponent-pool agent whose ``team_set`` is ``"@schedule"``
both follow a shared epoch-driven weight schedule loaded from a YAML file
(see ``metamon/rl/configs/team_schedules/`` and
``docs/teamset_curriculum_proposal.md``).

The schedule is **required** when the training opponent pool contains any
``"@schedule"`` agent (the pool raises if no schedule is provided). It is
optional otherwise — when absent, the collector falls back to a static team set
/ mix spec (the historical behavior).

How it fits together
--------------------

1. ``add_schedule_cli_args`` registers ``--train_team_schedule`` (path to a
   schedule YAML).
2. :func:`make_schedule_state` loads the YAML into a :class:`TeamMixSchedule`
   and wraps it + a fresh :class:`EpochRef` (epoch 0) in a
   :class:`ScheduleState`. This is created once in
   :func:`~metamon.rl.online_rl.create_online_experiment` and shared between the
   player team set and the opponent pool so both sides draw from the same
   epoch-driven mix.
3. :func:`resolve_train_team_set` builds the player's team set for the collector
   env: schedule-aware (:func:`get_metamon_team_set_from_schedule`) when a
   schedule is set, else the legacy static set / mix spec.
4. The :class:`~metamon.rl.metamon_to_amago.MetamonOnlineExperiment` stores the
   ``EpochRef`` and bumps it to ``self.epoch`` at the start of each
   ``collect_new_training_data`` cycle, so schedule-aware team sets lazily
   refresh their weights on the next ``yield_team()``.
5. On resume, :func:`log_schedule_start` prints the v1-style
   ``"Team mix schedule: starting at epoch N (from latest training_state)"``
   banner and the caller sets ``epoch_ref.epoch`` to the resumed epoch so the
   schedule picks up at the right phase.
"""

from __future__ import annotations

from typing import Optional

from metamon.env import (
    EpochRef,
    TeamMixSchedule,
    get_metamon_team_set_from_schedule,
    get_metamon_team_set_or_mix,
)


class ScheduleState:
    """A loaded :class:`TeamMixSchedule` + its shared :class:`EpochRef`.

    Both the collector's player team set and the opponent pool's ``"@schedule"``
    agents hold a reference to the *same* ``epoch_ref`` so one bump per
    collection cycle advances both. The ``EpochRef`` starts at epoch 0; the
    caller sets it to the resumed epoch after ``load_checkpoint`` on resume, or
    leaves it at 0 for a fresh start.
    """

    __slots__ = ("schedule", "epoch_ref")

    def __init__(self, schedule: TeamMixSchedule, epoch_ref: EpochRef):
        self.schedule = schedule
        self.epoch_ref = epoch_ref

    @property
    def set_names(self) -> list[str]:
        return self.schedule.set_names

    def phase_summary(self) -> str:
        """One-line per-phase weight summary for logging, e.g.
        ``"e0: gl_05_26=100%,smogon_pass2=0%,... | e940: ..."``."""
        parts = []
        names = self.schedule.set_names
        for epoch, row in zip(self.schedule._epochs, self.schedule._weight_rows):
            weights = ", ".join(f"{n}={w:.0%}" for n, w in zip(names, row))
            parts.append(f"e{epoch}: {weights}")
        return " | ".join(parts)


def add_schedule_cli_args(parser) -> None:
    """Register ``--train_team_schedule`` on ``parser``."""
    parser.add_argument(
        "--train_team_schedule",
        type=str,
        default=None,
        help="Path to a team-mix schedule YAML (epoch-driven, no restart needed). "
        "The collector's player team set and any opponent-pool agent whose "
        'team_set is "@schedule" both follow this schedule via a shared '
        "EpochRef. See docs/teamset_curriculum_proposal.md. Required when the "
        'training opponent pool uses "@schedule"; optional otherwise.',
    )


def make_schedule_state(schedule_path: Optional[str]) -> Optional[ScheduleState]:
    """Load a schedule YAML into a :class:`ScheduleState`, or ``None`` if disabled.

    Returns ``None`` when ``schedule_path`` is falsy (static team set / mix spec
    behavior). Raises if the path is set but cannot be loaded.
    """
    if not schedule_path:
        return None
    schedule = TeamMixSchedule.from_yaml_file(schedule_path)
    return ScheduleState(schedule=schedule, epoch_ref=EpochRef(epoch=0))


def resolve_train_team_set(
    battle_format: str,
    *,
    team_set_name: str,
    team_mix_spec: Optional[str],
    schedule_state: Optional[ScheduleState],
):
    """Build the collector's player team set.

    When ``schedule_state`` is set, returns a schedule-aware
    :class:`WeightedMixedTeamSet` (initial weights from the schedule's epoch-0
    entry; lazily refreshed from ``epoch_ref.epoch`` on each ``yield_team``).
    Otherwise falls back to the legacy static set / mix spec.
    """
    if schedule_state is not None:
        return get_metamon_team_set_from_schedule(
            battle_format, schedule_state.schedule, schedule_state.epoch_ref
        )
    if team_mix_spec:
        return get_metamon_team_set_or_mix(battle_format, team_mix_spec)
    return get_metamon_team_set_or_mix(battle_format, team_set_name)


def log_schedule_start(
    schedule_state: Optional[ScheduleState], *, resume_epoch: Optional[int]
) -> None:
    """Print the v1-style ``"Team mix schedule: starting at epoch N"`` banner.

    ``resume_epoch`` is the resumed epoch (from ``--resume_training_state``) when
    resuming, or ``None`` for a fresh start (epoch 0). No-op when no schedule.
    """
    if schedule_state is None:
        return
    start_epoch = resume_epoch if resume_epoch is not None else 0
    source = "from latest training_state" if resume_epoch is not None else "fresh start"
    print(f"  Team mix schedule: starting at epoch {start_epoch} ({source})")
    print(f"  Team mix schedule: {schedule_state.phase_summary()}")
