"""Optional step-level profiling for VectorizedPokepyEnv."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class StepProfileAccumulator:
    """Wall-time buckets accumulated across env steps (seconds)."""

    opponent_forward_s: float = 0.0
    lane_loop_s: float = 0.0
    build_obs_s: float = 0.0
    step_count: int = 0
    lane_steps: int = 0

    def reset(self) -> None:
        self.opponent_forward_s = 0.0
        self.lane_loop_s = 0.0
        self.build_obs_s = 0.0
        self.step_count = 0
        self.lane_steps = 0

    def summary(self) -> Dict[str, float]:
        if self.step_count == 0:
            return {}
        total = self.opponent_forward_s + self.lane_loop_s + self.build_obs_s
        per_step = total / self.step_count
        per_lane_ms = 1000.0 * self.lane_loop_s / max(self.lane_steps, 1)
        return {
            "steps": float(self.step_count),
            "total_s": total,
            "per_step_s": per_step,
            "fps": self.step_count / max(total, 1e-9),
            "opponent_forward_s": self.opponent_forward_s,
            "lane_loop_s": self.lane_loop_s,
            "build_obs_s": self.build_obs_s,
            "opponent_forward_pct": 100.0 * self.opponent_forward_s / max(total, 1e-9),
            "lane_loop_pct": 100.0 * self.lane_loop_s / max(total, 1e-9),
            "build_obs_pct": 100.0 * self.build_obs_s / max(total, 1e-9),
            "per_lane_loop_ms": per_lane_ms,
        }

    def format_summary(self) -> str:
        s = self.summary()
        if not s:
            return "no steps recorded"
        return (
            f"steps={int(s['steps'])} fps={s['fps']:.1f} "
            f"opp_fwd={s['opponent_forward_pct']:.1f}% "
            f"lane_loop={s['lane_loop_pct']:.1f}% "
            f"build_obs={s['build_obs_pct']:.1f}% "
            f"per_lane_loop_ms={s['per_lane_loop_ms']:.2f}"
        )


def merge_profile_summaries(
    summaries: List[Dict[str, float]],
) -> Dict[str, float]:
    """Merge per-run profile summaries (e.g. across batched_envs sweeps)."""
    if not summaries:
        return {}
    total_steps = sum(s.get("steps", 0) for s in summaries)
    total_s = sum(s.get("total_s", 0) for s in summaries)
    return {
        "runs": float(len(summaries)),
        "steps": total_steps,
        "total_s": total_s,
        "fps": total_steps / max(total_s, 1e-9),
        "mean_per_lane_loop_ms": sum(
            s.get("per_lane_loop_ms", 0) * s.get("steps", 0) for s in summaries
        )
        / max(total_steps, 1),
    }
