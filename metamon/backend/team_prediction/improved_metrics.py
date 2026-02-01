"""
Enhanced evaluation metrics for team prediction.

Provides comprehensive evaluation beyond simple token accuracy, including:
- Per-attribute accuracy (Pokemon, moves, items, abilities)
- Weighted metrics (more important attributes weighted higher)
- Top-k accuracy
- Consistency checking
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
import numpy as np
from collections import defaultdict


class TeamPredictionMetrics:
    """
    Comprehensive metrics for evaluating team prediction quality.
    """

    def __init__(self, vocab):
        """
        Args:
            vocab: Vocabulary object with type_ids and masks
        """
        self.vocab = vocab

        # Define importance weights for different attribute types
        self.attribute_weights = {
            "Mon": 3.0,  # Pokemon names most important
            "Move": 2.0,  # Moves very important
            "Ability": 1.5,  # Abilities/items moderately important
            "Item": 1.5,
            "Nature": 1.0,  # EVs/IVs/Nature less critical
            "EV": 0.5,
            "IV": 0.5,
            "Tera Type": 1.5,  # Tera type important in Gen 9
        }

    def compute_all_metrics(
        self,
        logits: torch.Tensor,
        y_tokens: torch.Tensor,
        pred_mask: torch.Tensor,
        type_ids: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.

        Args:
            logits: Model predictions [batch_size, seq_len, vocab_size]
            y_tokens: Ground truth tokens [batch_size, seq_len]
            pred_mask: Mask indicating which positions to evaluate [batch_size, seq_len]
            type_ids: Type IDs for each position [batch_size, seq_len]

        Returns:
            Dictionary of metric names to values
        """
        metrics = {}

        # Basic accuracy
        metrics["token_accuracy"] = self._token_accuracy(logits, y_tokens, pred_mask)

        # Weighted accuracy (emphasize important attributes)
        metrics["weighted_accuracy"] = self._weighted_accuracy(
            logits, y_tokens, pred_mask, type_ids
        )

        # Per-attribute accuracy
        attr_metrics = self._per_attribute_accuracy(
            logits, y_tokens, pred_mask, type_ids
        )
        metrics.update(attr_metrics)

        # Top-k accuracy
        for k in [3, 5, 10]:
            metrics[f"top_{k}_accuracy"] = self._topk_accuracy(
                logits, y_tokens, pred_mask, k
            )

        # Confidence calibration
        metrics["confidence"] = self._average_confidence(logits, pred_mask)
        metrics["calibration_error"] = self._calibration_error(
            logits, y_tokens, pred_mask
        )

        return metrics

    def _token_accuracy(
        self,
        logits: torch.Tensor,
        y_tokens: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> float:
        """Standard token-level accuracy."""
        preds = logits.argmax(dim=-1)
        correct = ((preds == y_tokens) * pred_mask).sum().item()
        total = max(pred_mask.sum().item(), 1)
        return correct / total

    def _weighted_accuracy(
        self,
        logits: torch.Tensor,
        y_tokens: torch.Tensor,
        pred_mask: torch.Tensor,
        type_ids: torch.Tensor,
    ) -> float:
        """
        Accuracy weighted by attribute importance.
        More important attributes (Pokemon names, moves) count more.
        """
        preds = logits.argmax(dim=-1)
        correct = (preds == y_tokens) * pred_mask

        # Create weight tensor
        weights = torch.ones_like(pred_mask, dtype=torch.float32)

        for type_name, weight in self.attribute_weights.items():
            type_id = self.vocab.type_ids.get(type_name)
            if type_id is not None:
                weights[type_ids == type_id] = weight

        weighted_correct = (correct.float() * weights).sum().item()
        weighted_total = (pred_mask.float() * weights).sum().item()

        return weighted_correct / max(weighted_total, 1.0)

    def _per_attribute_accuracy(
        self,
        logits: torch.Tensor,
        y_tokens: torch.Tensor,
        pred_mask: torch.Tensor,
        type_ids: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute accuracy separately for each attribute type.
        Helps identify which aspects the model struggles with.
        """
        preds = logits.argmax(dim=-1)
        correct = (preds == y_tokens) * pred_mask

        metrics = {}

        for type_name, type_id in self.vocab.type_ids.items():
            # Skip format (always correct, not predicted)
            if type_name == "Format":
                continue

            # Get mask for this attribute type
            attr_mask = (type_ids == type_id) & pred_mask

            if attr_mask.sum() > 0:
                attr_correct = (correct * attr_mask).sum().item()
                attr_total = attr_mask.sum().item()
                accuracy = attr_correct / attr_total

                # Use lowercase with underscores for metric name
                metric_name = f"{type_name.lower().replace(' ', '_')}_accuracy"
                metrics[metric_name] = accuracy

        return metrics

    def _topk_accuracy(
        self,
        logits: torch.Tensor,
        y_tokens: torch.Tensor,
        pred_mask: torch.Tensor,
        k: int,
    ) -> float:
        """
        Top-k accuracy: is the correct token in the top-k predictions?

        This is more forgiving than exact match and better captures
        cases where the model is "close" to correct.
        """
        # Get top-k predictions
        topk_preds = logits.topk(k, dim=-1).indices  # [batch, seq_len, k]

        # Check if ground truth is in top-k
        y_expanded = y_tokens.unsqueeze(-1).expand_as(topk_preds)
        in_topk = (topk_preds == y_expanded).any(dim=-1)

        correct = (in_topk * pred_mask).sum().item()
        total = max(pred_mask.sum().item(), 1)

        return correct / total

    def _average_confidence(
        self,
        logits: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> float:
        """
        Average confidence (max probability) of predictions.
        Helps understand if model is certain or uncertain.
        """
        probs = F.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1).values

        masked_probs = max_probs * pred_mask
        avg_confidence = masked_probs.sum().item() / max(pred_mask.sum().item(), 1)

        return avg_confidence

    def _calibration_error(
        self,
        logits: torch.Tensor,
        y_tokens: torch.Tensor,
        pred_mask: torch.Tensor,
        num_bins: int = 10,
    ) -> float:
        """
        Expected Calibration Error (ECE).

        Measures whether predicted probabilities match actual accuracy.
        Low ECE means model knows when it's right/wrong.
        """
        probs = F.softmax(logits, dim=-1)
        confidences = probs.max(dim=-1).values
        predictions = logits.argmax(dim=-1)
        accuracies = (predictions == y_tokens).float()

        # Only consider masked positions
        confidences = confidences[pred_mask]
        accuracies = accuracies[pred_mask]

        if len(confidences) == 0:
            return 0.0

        # Bin predictions by confidence
        ece = 0.0
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)

        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            # Find predictions in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

            if in_bin.sum() > 0:
                bin_confidence = confidences[in_bin].mean()
                bin_accuracy = accuracies[in_bin].mean()
                bin_size = in_bin.sum().float()

                # Weighted by bin size
                ece += (bin_size / len(confidences)) * abs(
                    bin_confidence - bin_accuracy
                )

        return ece.item()


class DetailedEvaluator:
    """
    Provides detailed evaluation including team-level metrics.
    """

    def __init__(self, vocab):
        self.vocab = vocab
        self.metrics = TeamPredictionMetrics(vocab)

    def evaluate_teams(
        self,
        predicted_teams: List[List[str]],
        true_teams: List[List[str]],
    ) -> Dict[str, float]:
        """
        Evaluate at the team level (not just tokens).

        Args:
            predicted_teams: List of predicted team sequences
            true_teams: List of ground truth team sequences

        Returns:
            Dictionary of team-level metrics
        """
        metrics = {}

        # Pokemon identification accuracy
        pokemon_correct = 0
        pokemon_total = 0

        # Moveset accuracy (all 4 moves correct for a Pokemon)
        moveset_correct = 0
        moveset_total = 0

        for pred_team, true_team in zip(predicted_teams, true_teams):
            # Extract Pokemon names
            pred_pokemon = self._extract_pokemon(pred_team)
            true_pokemon = self._extract_pokemon(true_team)

            # Pokemon identification
            for pred, true in zip(pred_pokemon, true_pokemon):
                if pred == true:
                    pokemon_correct += 1
                pokemon_total += 1

            # Moveset accuracy
            for i, (pred_mon, true_mon) in enumerate(zip(pred_pokemon, true_pokemon)):
                if pred_mon == true_mon:
                    pred_moves = self._extract_moves(pred_team, i)
                    true_moves = self._extract_moves(true_team, i)

                    if set(pred_moves) == set(true_moves):
                        moveset_correct += 1
                    moveset_total += 1

        metrics["pokemon_identification_accuracy"] = pokemon_correct / max(
            pokemon_total, 1
        )
        metrics["moveset_accuracy"] = moveset_correct / max(moveset_total, 1)

        return metrics

    def _extract_pokemon(self, team_seq: List[str]) -> List[str]:
        """Extract Pokemon names from team sequence."""
        pokemon = []
        for token in team_seq:
            if token.startswith("Mon: "):
                pokemon.append(token.split("Mon: ")[1])
        return pokemon

    def _extract_moves(self, team_seq: List[str], pokemon_idx: int) -> List[str]:
        """Extract moves for a specific Pokemon."""
        # Each Pokemon has 8 tokens (in no-stats mode)
        start_idx = 1 + pokemon_idx * 8  # +1 for format token
        moves = []

        for i in range(start_idx + 4, start_idx + 8):
            if i < len(team_seq) and team_seq[i].startswith("Move: "):
                moves.append(team_seq[i].split("Move: ")[1])

        return moves


def compute_loss_and_metrics(
    logits: torch.Tensor,
    y_tokens: torch.Tensor,
    pred_mask: torch.Tensor,
    type_ids: torch.Tensor,
    vocab,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    Combined loss and metrics computation.

    Args:
        logits: Model predictions [batch_size, seq_len, vocab_size]
        y_tokens: Ground truth tokens [batch_size, seq_len]
        pred_mask: Mask indicating which positions to evaluate [batch_size, seq_len]
        type_ids: Type IDs for each position [batch_size, seq_len]
        vocab: Vocabulary object
        ignore_index: Index to ignore in loss computation

    Returns:
        Tuple of (loss, metrics_dict)
    """
    B, L, V = logits.shape

    # Compute cross-entropy loss
    loss = F.cross_entropy(
        logits.view(-1, V),
        y_tokens.view(-1),
        reduction="none",
        ignore_index=ignore_index,
    )

    # Apply prediction mask
    num_preds = max(pred_mask.sum().item(), 1)
    loss = (loss * pred_mask.view(-1)).sum() / num_preds

    # Compute all metrics
    metrics_computer = TeamPredictionMetrics(vocab)
    metrics = metrics_computer.compute_all_metrics(
        logits, y_tokens, pred_mask, type_ids
    )

    return loss, metrics


# Example usage
if __name__ == "__main__":
    from metamon.backend.team_prediction.vocabulary import Vocabulary

    # Create dummy data for testing
    vocab = Vocabulary()
    batch_size = 4
    seq_len = 49  # Format + 6 Pokemon * 8 attributes
    vocab_size = len(vocab.tokenizer)

    logits = torch.randn(batch_size, seq_len, vocab_size)
    y_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    pred_mask = torch.randint(0, 2, (batch_size, seq_len)).bool()
    type_ids = torch.randint(0, 9, (batch_size, seq_len))

    # Compute metrics
    loss, metrics = compute_loss_and_metrics(
        logits, y_tokens, pred_mask, type_ids, vocab
    )

    print(f"Loss: {loss.item():.4f}")
    print("\nMetrics:")
    for name, value in sorted(metrics.items()):
        print(f"  {name}: {value:.4f}")
