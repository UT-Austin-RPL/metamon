"""
Curriculum learning components for team prediction.

Implements dynamic masking schedules and hierarchical training strategies.
"""

import math
from typing import Literal


class MaskingScheduler:
    """
    Gradually increase masking difficulty during training.

    Starts with easy tasks (low masking) and progressively makes prediction harder.
    This helps the model learn basic patterns before tackling full team prediction.
    """

    def __init__(
        self,
        initial_mask_prob: float = 0.05,
        final_mask_prob: float = 0.3,
        warmup_steps: int = 10000,
        schedule: Literal["linear", "cosine", "exponential"] = "cosine",
    ):
        """
        Args:
            initial_mask_prob: Starting masking probability (easier)
            final_mask_prob: Final masking probability (harder)
            warmup_steps: Number of steps to ramp up difficulty
            schedule: Type of schedule ('linear', 'cosine', 'exponential')
        """
        assert 0 <= initial_mask_prob <= final_mask_prob <= 1.0
        assert warmup_steps > 0

        self.initial_prob = initial_mask_prob
        self.final_prob = final_mask_prob
        self.warmup_steps = warmup_steps
        self.schedule = schedule

    def get_mask_prob(self, step: int) -> float:
        """
        Get masking probability for current training step.

        Args:
            step: Current training step

        Returns:
            Masking probability in [initial_prob, final_prob]
        """
        if step >= self.warmup_steps:
            return self.final_prob

        progress = step / self.warmup_steps

        if self.schedule == "linear":
            return self.initial_prob + (self.final_prob - self.initial_prob) * progress

        elif self.schedule == "cosine":
            # Smooth acceleration using cosine
            cosine_progress = (1 - math.cos(progress * math.pi)) / 2
            return (
                self.initial_prob
                + (self.final_prob - self.initial_prob) * cosine_progress
            )

        elif self.schedule == "exponential":
            # Slow start, fast finish
            exp_progress = (math.exp(progress) - 1) / (math.e - 1)
            return (
                self.initial_prob + (self.final_prob - self.initial_prob) * exp_progress
            )

        raise ValueError(f"Unknown schedule: {self.schedule}")


class HierarchicalMaskingScheduler:
    """
    Three-phase curriculum that progressively adds masking complexity:

    Phase 1 (0-33%): Only mask Pokemon names (learn species from movesets)
    Phase 2 (33-66%): Mask names + moves (learn common move combinations)
    Phase 3 (66-100%): Mask all attributes (full team prediction)

    This mimics the natural difficulty progression in team prediction.
    """

    def __init__(
        self,
        total_steps: int = 100000,
        phase1_ratio: float = 0.33,
        phase2_ratio: float = 0.33,
    ):
        """
        Args:
            total_steps: Total training steps
            phase1_ratio: Fraction of training for phase 1
            phase2_ratio: Fraction of training for phase 2
        """
        assert 0 < phase1_ratio + phase2_ratio < 1.0

        self.total_steps = total_steps
        self.phase1_end = int(total_steps * phase1_ratio)
        self.phase2_end = int(total_steps * (phase1_ratio + phase2_ratio))

    def get_masking_config(self, step: int) -> dict:
        """
        Get masking configuration for current phase.

        Returns:
            Dictionary with masking settings:
            - mask_pokemon_prob: Probability to mask entire Pokemon
            - mask_attrs_prob: Probability to mask individual attributes
            - toy_names_only: Whether to only mask names
        """
        if step < self.phase1_end:
            # Phase 1: Only names
            return {
                "mask_pokemon_prob": 0.0,
                "mask_attrs_prob": 0.0,
                "toy_names_only": True,
                "phase": 1,
            }

        elif step < self.phase2_end:
            # Phase 2: Names + moves
            progress = (step - self.phase1_end) / (self.phase2_end - self.phase1_end)
            move_mask_prob = 0.1 + 0.1 * progress  # Ramp from 0.1 to 0.2

            return {
                "mask_pokemon_prob": 0.0,
                "mask_attrs_prob": move_mask_prob,
                "toy_names_only": False,
                "mask_moves_only": True,  # New flag - only mask moves, not items/abilities
                "phase": 2,
            }

        else:
            # Phase 3: Full masking
            progress = (step - self.phase2_end) / (self.total_steps - self.phase2_end)
            pokemon_mask_prob = 0.05 + 0.15 * progress  # Ramp from 0.05 to 0.2
            attrs_mask_prob = 0.1 + 0.2 * progress  # Ramp from 0.1 to 0.3

            return {
                "mask_pokemon_prob": pokemon_mask_prob,
                "mask_attrs_prob": attrs_mask_prob,
                "toy_names_only": False,
                "mask_moves_only": False,
                "phase": 3,
            }


class AdaptiveMaskingScheduler:
    """
    Adjust masking difficulty based on model performance.

    If model is doing well, increase difficulty. If struggling, ease up.
    This creates a personalized curriculum for each training run.
    """

    def __init__(
        self,
        base_mask_prob: float = 0.15,
        min_mask_prob: float = 0.05,
        max_mask_prob: float = 0.4,
        target_accuracy: float = 0.7,
        adjustment_rate: float = 0.001,
        eval_window: int = 1000,
    ):
        """
        Args:
            base_mask_prob: Starting masking probability
            min_mask_prob: Minimum allowed masking probability
            max_mask_prob: Maximum allowed masking probability
            target_accuracy: Target validation accuracy
            adjustment_rate: How quickly to adjust difficulty
            eval_window: How often to evaluate and adjust (in steps)
        """
        self.current_prob = base_mask_prob
        self.min_prob = min_mask_prob
        self.max_prob = max_mask_prob
        self.target_accuracy = target_accuracy
        self.adjustment_rate = adjustment_rate
        self.eval_window = eval_window

        self.recent_accuracies = []

    def update(self, step: int, accuracy: float):
        """
        Update masking probability based on recent performance.

        Args:
            step: Current training step
            accuracy: Recent validation accuracy
        """
        self.recent_accuracies.append(accuracy)

        if step % self.eval_window == 0 and len(self.recent_accuracies) > 0:
            avg_accuracy = sum(self.recent_accuracies) / len(self.recent_accuracies)

            # If doing well, make it harder
            if avg_accuracy > self.target_accuracy + 0.05:
                self.current_prob = min(
                    self.max_prob, self.current_prob + self.adjustment_rate
                )

            # If struggling, make it easier
            elif avg_accuracy < self.target_accuracy - 0.05:
                self.current_prob = max(
                    self.min_prob, self.current_prob - self.adjustment_rate
                )

            self.recent_accuracies = []

    def get_mask_prob(self, step: int) -> float:
        """Get current masking probability."""
        return self.current_prob


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # Compare different schedules
    steps = np.arange(0, 50000)

    schedules = {
        "Linear": MaskingScheduler(0.05, 0.3, 20000, "linear"),
        "Cosine": MaskingScheduler(0.05, 0.3, 20000, "cosine"),
        "Exponential": MaskingScheduler(0.05, 0.3, 20000, "exponential"),
    }

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for name, scheduler in schedules.items():
        probs = [scheduler.get_mask_prob(s) for s in steps]
        plt.plot(steps, probs, label=name, linewidth=2)

    plt.xlabel("Training Step")
    plt.ylabel("Masking Probability")
    plt.title("Curriculum Learning Schedules")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Show hierarchical phases
    plt.subplot(1, 2, 2)
    hierarchical = HierarchicalMaskingScheduler(total_steps=100000)

    phases = []
    pokemon_probs = []
    attr_probs = []

    for s in range(0, 100000, 1000):
        config = hierarchical.get_masking_config(s)
        phases.append(config["phase"])
        pokemon_probs.append(config["mask_pokemon_prob"])
        attr_probs.append(config["mask_attrs_prob"])

    steps_hierarchical = range(0, 100000, 1000)
    plt.plot(steps_hierarchical, pokemon_probs, label="Pokemon Masking", linewidth=2)
    plt.plot(steps_hierarchical, attr_probs, label="Attribute Masking", linewidth=2)

    # Mark phase boundaries
    plt.axvline(
        x=33000, color="gray", linestyle="--", alpha=0.5, label="Phase Boundaries"
    )
    plt.axvline(x=66000, color="gray", linestyle="--", alpha=0.5)

    plt.xlabel("Training Step")
    plt.ylabel("Masking Probability")
    plt.title("Hierarchical Curriculum (3 Phases)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("curriculum_schedules.png", dpi=150, bbox_inches="tight")
    print("Saved curriculum visualization to curriculum_schedules.png")
