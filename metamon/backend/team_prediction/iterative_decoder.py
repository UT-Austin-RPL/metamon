"""
Iterative decoding for team prediction using MaskGIT-style approach.

Instead of predicting all masked tokens in one shot, iteratively refine
predictions by keeping high-confidence predictions and re-predicting
low-confidence ones.

Reference: MaskGIT (Chang et al., 2022) - https://arxiv.org/abs/2202.04200
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Literal, Optional
import math


class IterativeTeamDecoder:
    """
    Iteratively refine team predictions using confidence-based masking.

    Algorithm:
    1. Start with partially masked team
    2. Predict all masked tokens
    3. Keep predictions with confidence > threshold
    4. Re-mask low-confidence predictions
    5. Repeat for T iterations

    The confidence threshold decreases over iterations, so we become
    less selective as we approach the final prediction.
    """

    def __init__(
        self,
        model,
        vocab,
        num_iterations: int = 8,
        confidence_schedule: Literal["linear", "cosine", "root"] = "cosine",
        min_confidence: float = 0.3,
        max_confidence: float = 0.9,
        temperature: float = 1.0,
    ):
        """
        Args:
            model: TeamTransformer model
            vocab: Vocabulary object for filtering
            num_iterations: Number of refinement iterations
            confidence_schedule: How to decay confidence threshold
            min_confidence: Final confidence threshold (iteration T)
            max_confidence: Initial confidence threshold (iteration 0)
            temperature: Softmax temperature (higher = more diverse)
        """
        self.model = model
        self.vocab = vocab
        self.num_iterations = num_iterations
        self.confidence_schedule = confidence_schedule
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        self.temperature = temperature

    @torch.no_grad()
    def decode(
        self,
        x_tokens: torch.Tensor,
        type_ids: torch.Tensor,
        pred_mask: torch.Tensor,
        return_all_iterations: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, list]:
        """
        Iteratively decode masked team.

        Args:
            x_tokens: Initial tokens [batch_size, seq_len]
            type_ids: Type IDs for filtering [batch_size, seq_len]
            pred_mask: Mask indicating what needs prediction [batch_size, seq_len]
            return_all_iterations: If True, return predictions at each iteration

        Returns:
            Completed team tokens (and optionally list of intermediate results)
        """
        self.model.eval()

        current_tokens = x_tokens.clone()
        current_mask = pred_mask.clone()

        all_iterations = [] if return_all_iterations else None

        for t in range(self.num_iterations):
            # Forward pass
            logits = self.model(current_tokens, type_ids)

            # Apply temperature
            logits = logits / self.temperature

            # Compute probabilities with vocab filtering
            probs = F.softmax(logits, dim=-1)
            filtered_probs = self.vocab.filter_probs(probs, type_ids)

            # Get predictions and confidences
            confidences, predictions = filtered_probs.max(dim=-1)

            # Confidence threshold for this iteration
            threshold = self._get_threshold(t)

            # Keep high-confidence predictions
            keep_mask = (confidences > threshold) & current_mask
            current_tokens[keep_mask] = predictions[keep_mask]
            current_mask[keep_mask] = False

            if return_all_iterations:
                all_iterations.append(
                    {
                        "iteration": t,
                        "tokens": current_tokens.clone(),
                        "mask": current_mask.clone(),
                        "threshold": threshold,
                        "remaining": current_mask.sum().item(),
                    }
                )

            # Early stopping if all predicted
            if not current_mask.any():
                break

        if return_all_iterations:
            return current_tokens, all_iterations
        return current_tokens

    def _get_threshold(self, iteration: int) -> float:
        """
        Compute confidence threshold for current iteration.

        Threshold decreases over iterations so we become less selective.
        """
        if iteration >= self.num_iterations - 1:
            return self.min_confidence

        progress = iteration / (self.num_iterations - 1)

        if self.confidence_schedule == "linear":
            return (
                self.max_confidence
                - (self.max_confidence - self.min_confidence) * progress
            )

        elif self.confidence_schedule == "cosine":
            # Smooth decay
            cosine_progress = (1 + math.cos(progress * math.pi)) / 2
            return (
                self.min_confidence
                + (self.max_confidence - self.min_confidence) * cosine_progress
            )

        elif self.confidence_schedule == "root":
            # Aggressive early, conservative later
            root_progress = 1 - math.sqrt(1 - progress)
            return (
                self.max_confidence
                - (self.max_confidence - self.min_confidence) * root_progress
            )

        raise ValueError(f"Unknown schedule: {self.confidence_schedule}")


class AdaptiveIterativeDecoder(IterativeTeamDecoder):
    """
    Adaptive version that adjusts strategy based on prediction difficulty.

    Uses different numbers of iterations and confidence thresholds depending
    on how much of the team is masked.
    """

    def __init__(
        self, model, vocab, min_iterations: int = 4, max_iterations: int = 12, **kwargs
    ):
        super().__init__(model, vocab, **kwargs)
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations

    @torch.no_grad()
    def decode(
        self,
        x_tokens: torch.Tensor,
        type_ids: torch.Tensor,
        pred_mask: torch.Tensor,
        return_all_iterations: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, list]:
        """
        Adaptive decoding with dynamic iteration count.
        """
        # Determine difficulty based on masking rate
        mask_rate = pred_mask.float().mean().item()

        # More iterations for harder problems
        num_iterations = int(
            self.min_iterations
            + (self.max_iterations - self.min_iterations) * mask_rate
        )

        # Temporarily override num_iterations
        original_iters = self.num_iterations
        self.num_iterations = num_iterations

        result = super().decode(x_tokens, type_ids, pred_mask, return_all_iterations)

        # Restore
        self.num_iterations = original_iters

        return result


class IterativeTrainingWrapper:
    """
    Wrapper for training with iterative decoding simulation.

    During training, we simulate the iterative process to help the model
    learn to produce well-calibrated confidence scores.
    """

    def __init__(
        self,
        model,
        vocab,
        num_training_iterations: int = 3,
        confidence_threshold: float = 0.7,
    ):
        """
        Args:
            model: TeamTransformer model
            vocab: Vocabulary object
            num_training_iterations: Simulate this many iterations during training
            confidence_threshold: Confidence to use for unmasking during training
        """
        self.model = model
        self.vocab = vocab
        self.num_training_iterations = num_training_iterations
        self.confidence_threshold = confidence_threshold

    def forward_with_unmasking(
        self,
        x_tokens: torch.Tensor,
        type_ids: torch.Tensor,
        y_tokens: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with iterative unmasking.

        Returns:
            Tuple of (final_logits, final_pred_mask, num_unmasked)
        """
        current_tokens = x_tokens.clone()
        current_mask = pred_mask.clone()

        total_unmasked = 0

        for t in range(self.num_training_iterations):
            # Forward pass
            logits = self.model(current_tokens, type_ids)

            if t < self.num_training_iterations - 1:
                # Intermediate iteration - do unmasking
                with torch.no_grad():
                    probs = F.softmax(logits, dim=-1)
                    filtered_probs = self.vocab.filter_probs(probs, type_ids)
                    confidences, predictions = filtered_probs.max(dim=-1)

                    # Unmask high-confidence predictions
                    unmask = (confidences > self.confidence_threshold) & current_mask
                    current_tokens[unmask] = predictions[unmask]
                    current_mask[unmask] = False

                    total_unmasked += unmask.sum().item()

        # Return final logits and remaining mask
        return logits, current_mask, total_unmasked


def compare_single_vs_iterative(
    model,
    vocab,
    x_tokens: torch.Tensor,
    type_ids: torch.Tensor,
    pred_mask: torch.Tensor,
    y_tokens: torch.Tensor,
    num_iterations: int = 8,
) -> dict:
    """
    Compare single-shot vs iterative decoding performance.

    Useful for evaluating whether iterative refinement helps.
    """
    model.eval()

    results = {}

    # Single-shot prediction
    with torch.no_grad():
        logits = model(x_tokens, type_ids)
        probs = F.softmax(logits, dim=-1)
        filtered_probs = vocab.filter_probs(probs, type_ids)
        single_shot_preds = filtered_probs.argmax(dim=-1)

        single_shot_acc = ((single_shot_preds == y_tokens) * pred_mask).float().mean()
        results["single_shot_accuracy"] = single_shot_acc.item()

    # Iterative decoding
    decoder = IterativeTeamDecoder(model, vocab, num_iterations=num_iterations)

    iterative_preds, iterations = decoder.decode(
        x_tokens, type_ids, pred_mask, return_all_iterations=True
    )

    iterative_acc = ((iterative_preds == y_tokens) * pred_mask).float().mean()
    results["iterative_accuracy"] = iterative_acc.item()

    # Accuracy at each iteration
    results["accuracy_by_iteration"] = []
    for it in iterations:
        acc = ((it["tokens"] == y_tokens) * pred_mask).float().mean()
        results["accuracy_by_iteration"].append(
            {
                "iteration": it["iteration"],
                "accuracy": acc.item(),
                "remaining_masked": it["remaining"],
            }
        )

    return results


# Example usage
if __name__ == "__main__":
    from metamon.backend.team_prediction.model import TeamTransformer
    from metamon.backend.team_prediction.vocabulary import Vocabulary

    # Create model and vocab
    model = TeamTransformer(
        max_seq_len=64,
        d_model=256,
        nhead=4,
        num_layers=3,
    )
    model.eval()

    vocab = Vocabulary()

    # Create dummy data
    batch_size = 2
    seq_len = 49

    x_tokens = torch.randint(0, len(vocab.tokenizer), (batch_size, seq_len))
    type_ids = torch.randint(0, 9, (batch_size, seq_len))
    pred_mask = torch.randint(0, 2, (batch_size, seq_len)).bool()
    y_tokens = torch.randint(0, len(vocab.tokenizer), (batch_size, seq_len))

    print("Testing iterative decoder...")

    # Standard iterative decoding
    decoder = IterativeTeamDecoder(
        model, vocab, num_iterations=8, confidence_schedule="cosine"
    )

    decoded_tokens, iterations = decoder.decode(
        x_tokens, type_ids, pred_mask, return_all_iterations=True
    )

    print(f"\nIterative decoding completed in {len(iterations)} iterations")
    for it in iterations:
        print(
            f"  Iteration {it['iteration']}: "
            f"threshold={it['threshold']:.3f}, "
            f"remaining={it['remaining']}"
        )

    # Compare single-shot vs iterative
    print("\nComparing single-shot vs iterative...")
    comparison = compare_single_vs_iterative(
        model, vocab, x_tokens, type_ids, pred_mask, y_tokens
    )

    print(f"Single-shot accuracy: {comparison['single_shot_accuracy']:.3f}")
    print(f"Iterative accuracy: {comparison['iterative_accuracy']:.3f}")
    print(
        f"Improvement: {comparison['iterative_accuracy'] - comparison['single_shot_accuracy']:.3f}"
    )
