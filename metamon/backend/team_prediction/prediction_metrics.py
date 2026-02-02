import torch
import torch.nn.functional as F
from typing import Dict, Optional
from collections import defaultdict


class TeamPredictionMetrics:
    """Computes evaluation metrics for team prediction."""

    def __init__(self, vocab):
        self.vocab = vocab
        self.attribute_weights = vocab.attribute_weights

    def compute_all_metrics(
        self,
        logits: torch.Tensor,
        y_tokens: torch.Tensor,
        pred_mask: torch.Tensor,
        type_ids: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute all evaluation metrics."""
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
        """Accuracy weighted by attribute importance."""
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
        """Compute accuracy separately for each attribute type."""
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
        """Top-k accuracy: is the correct token in the top-k predictions?"""
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
        """Average confidence (max probability) of predictions."""
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
        """Expected Calibration Error (ECE)."""
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


class EvaluationAccumulator:
    """Accumulates evaluation statistics across batches for per-generation metrics."""

    def __init__(self, vocab):
        self.vocab = vocab
        self.metrics_computer = TeamPredictionMetrics(vocab)
        self.attribute_weights = vocab.attribute_weights

        self.total_correct = 0
        self.total_count = 0
        self.total_loss = 0.0
        self.num_batches = 0
        self.weighted_correct = 0.0
        self.weighted_total = 0.0

        self.gen_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        self.attr_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        self.gen_attr_stats = defaultdict(
            lambda: defaultdict(lambda: {"correct": 0, "total": 0})
        )

    def _extract_gen_from_tokens(self, x_tokens: torch.Tensor) -> torch.Tensor:
        """Extract generation number for each sample from the format token."""
        batch_size = x_tokens.shape[0]
        gens = torch.zeros(batch_size, dtype=torch.long, device=x_tokens.device)

        # First token is the format token
        format_token_ids = x_tokens[:, 0]

        for i, token_id in enumerate(format_token_ids):
            gens[i] = self.vocab.format_token_to_gen.get(token_id.item(), 0)

        return gens

    def add_batch(
        self,
        logits: torch.Tensor,
        y_tokens: torch.Tensor,
        pred_mask: torch.Tensor,
        type_ids: torch.Tensor,
        x_tokens: torch.Tensor,
        loss: Optional[torch.Tensor] = None,
    ):
        """Add a batch to the accumulator."""
        preds = logits.argmax(dim=-1)
        correct = (preds == y_tokens) & pred_mask

        # Extract generation for each sample
        gens = self._extract_gen_from_tokens(x_tokens)

        # Overall stats
        self.total_correct += correct.sum().item()
        self.total_count += pred_mask.sum().item()
        if loss is not None:
            self.total_loss += loss.item()
        self.num_batches += 1

        # Weighted stats - create weight tensor
        weights = torch.ones_like(pred_mask, dtype=torch.float32)
        for type_name, weight in self.attribute_weights.items():
            type_id = self.vocab.type_ids.get(type_name)
            if type_id is not None:
                weights[type_ids == type_id] = weight
        self.weighted_correct += (correct.float() * weights).sum().item()
        self.weighted_total += (pred_mask.float() * weights).sum().item()

        # Per-generation stats
        for b in range(x_tokens.shape[0]):
            gen = gens[b].item()
            sample_correct = correct[b].sum().item()
            sample_total = pred_mask[b].sum().item()

            self.gen_stats[gen]["correct"] += sample_correct
            self.gen_stats[gen]["total"] += sample_total

            # Per-attribute stats for this sample
            for type_name, type_id in self.vocab.type_ids.items():
                if type_name == "Format":
                    continue

                attr_mask = (type_ids[b] == type_id) & pred_mask[b]
                if attr_mask.sum() > 0:
                    attr_correct = (correct[b] & attr_mask).sum().item()
                    attr_total = attr_mask.sum().item()

                    # Overall per-attribute
                    self.attr_stats[type_name]["correct"] += attr_correct
                    self.attr_stats[type_name]["total"] += attr_total

                    # Per-gen per-attribute
                    self.gen_attr_stats[gen][type_name]["correct"] += attr_correct
                    self.gen_attr_stats[gen][type_name]["total"] += attr_total

    def compute_metrics(self) -> Dict[str, float]:
        """Compute final metrics from accumulated statistics."""
        metrics = {}

        # Overall accuracy
        metrics["token_accuracy"] = self.total_correct / max(self.total_count, 1)
        metrics["weighted_accuracy"] = self.weighted_correct / max(
            self.weighted_total, 1.0
        )
        metrics["loss"] = self.total_loss / max(self.num_batches, 1)

        # Per-attribute accuracy (overall)
        for attr_name, stats in self.attr_stats.items():
            if stats["total"] > 0:
                metric_name = f"{attr_name.lower().replace(' ', '_')}_accuracy"
                metrics[metric_name] = stats["correct"] / stats["total"]

        # Per-generation accuracy
        for gen, stats in sorted(self.gen_stats.items()):
            if gen > 0 and stats["total"] > 0:  # Skip unknown gen (0)
                metrics[f"gen{gen}_accuracy"] = stats["correct"] / stats["total"]
                metrics[f"gen{gen}_count"] = stats["total"]

        # Per-generation per-attribute (only for gens with data)
        for gen, attr_dict in sorted(self.gen_attr_stats.items()):
            if gen == 0:
                continue
            for attr_name, stats in attr_dict.items():
                if stats["total"] > 0:
                    metric_name = (
                        f"gen{gen}_{attr_name.lower().replace(' ', '_')}_accuracy"
                    )
                    metrics[metric_name] = stats["correct"] / stats["total"]

        return metrics


def compute_loss_and_metrics(
    logits: torch.Tensor,
    y_tokens: torch.Tensor,
    pred_mask: torch.Tensor,
    type_ids: torch.Tensor,
    vocab,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Combined loss and metrics computation for training."""
    vocab_size = logits.shape[-1]

    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        y_tokens.view(-1),
        reduction="none",
    )
    num_preds = max(pred_mask.sum().item(), 1)
    loss = (loss * pred_mask.view(-1)).sum() / num_preds

    metrics_computer = TeamPredictionMetrics(vocab)
    metrics = metrics_computer.compute_all_metrics(
        logits, y_tokens, pred_mask, type_ids
    )

    return loss, metrics
