import torch
import torch.nn.functional as F
from typing import Literal, Optional, List
from dataclasses import dataclass, field
import math


@dataclass
class IterativeDecodingStats:
    mask_ratios: List[float] = field(default_factory=list)
    remaining_counts: List[int] = field(default_factory=list)
    committed_counts: List[int] = field(
        default_factory=list
    )  # tokens committed per iteration
    confidences_per_iter: List[torch.Tensor] = field(default_factory=list)
    num_iterations_used: int = 0
    total_masked: int = 0

    def add_iteration(
        self,
        iteration: int,
        mask_ratio: float,
        remaining: int,
        committed: int,
        confidences: torch.Tensor,
        current_mask: torch.Tensor,
    ):
        self.mask_ratios.append(mask_ratio)
        self.remaining_counts.append(remaining)
        self.committed_counts.append(committed)
        if current_mask.any():
            self.confidences_per_iter.append(confidences[current_mask].cpu())
        else:
            self.confidences_per_iter.append(torch.tensor([]))
        self.num_iterations_used = iteration + 1


class IterativeStatsAccumulator:
    def __init__(self, num_iterations: int):
        self.num_iterations = num_iterations
        self.total_masked = 0
        self.mask_ratios: Optional[List[float]] = None
        self.remaining_counts: List[List[int]] = [[] for _ in range(num_iterations)]
        self.committed_counts: List[List[int]] = [[] for _ in range(num_iterations)]
        self.confidences: List[List[torch.Tensor]] = [[] for _ in range(num_iterations)]

    def add_batch(self, stats: IterativeDecodingStats):
        self.total_masked += stats.total_masked
        if self.mask_ratios is None:
            # same for all batches
            self.mask_ratios = stats.mask_ratios
        for i, (remaining, committed, conf) in enumerate(
            zip(
                stats.remaining_counts,
                stats.committed_counts,
                stats.confidences_per_iter,
            )
        ):
            self.remaining_counts[i].append(remaining)
            self.committed_counts[i].append(committed)
            if len(conf) > 0:
                self.confidences[i].append(conf)

    def compute_results(self) -> dict:
        """
        Returns:
            - mask_ratios: list[float] target mask ratio per iteration
            - remaining_frac: list[float] actual fraction remaining per iteration
            - committed_per_iter: list[int] total committed tokens per iteration
            - confidences: list[Tensor] concatenated confidences per iteration
        """
        remaining_frac = []
        committed_per_iter = []
        if self.total_masked > 0:
            for i in range(self.num_iterations):
                if self.remaining_counts[i]:
                    remaining_frac.append(
                        sum(self.remaining_counts[i]) / self.total_masked
                    )
                    committed_per_iter.append(sum(self.committed_counts[i]))
                else:
                    break

        confidences = []
        for i in range(self.num_iterations):
            if self.confidences[i]:
                confidences.append(torch.cat(self.confidences[i], dim=0))
            else:
                confidences.append(torch.tensor([]))

        return {
            "mask_ratios": self.mask_ratios or [],
            "remaining_frac": remaining_frac,
            "committed_per_iter": committed_per_iter,
            "confidences": confidences,
        }


class IterativeTeamDecoder:
    """
    MaskGIT-style iterative decoding
    """

    def __init__(
        self,
        model,
        vocab,
        num_iterations: int = 8,
        mask_schedule: Literal["linear", "cosine"] = "cosine",
        temperature: float = 1.0,
        top_p: float = 0.9,
        deterministic: bool = False,
    ):
        """
        Args:
            model: TeamTransformer model
            vocab: Vocabulary object for filtering
            num_iterations: Number of refinement iterations (T)
            mask_schedule: How mask ratio decreases ("cosine" recommended)
            temperature: Sampling temperature (lower = more deterministic)
            top_p: Nucleus sampling threshold (keep smallest set with cumulative prob >= top_p)
            deterministic: If True, use argmax instead of sampling (for fair comparison with one-shot)
        """
        self.model = model
        self.vocab = vocab
        self.num_iterations = num_iterations
        self.mask_schedule = mask_schedule
        self.temperature = temperature
        self.top_p = top_p
        self.deterministic = deterministic

    def _nucleus_filter(self, probs: torch.Tensor) -> torch.Tensor:
        """Apply nucleus (top-p) filtering and renormalize."""
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs - sorted_probs > self.top_p
        sorted_probs[sorted_mask] = 0.0
        filtered_probs = torch.zeros_like(probs)
        filtered_probs.scatter_(-1, sorted_indices, sorted_probs)
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
        return filtered_probs

    def _gamma(self, ratio: float) -> float:
        """
        Mask ratio schedule: gamma(r) gives fraction of tokens still masked at progress r.
        """
        if self.mask_schedule == "linear":
            return 1.0 - ratio
        elif self.mask_schedule == "cosine":
            return math.cos(ratio * math.pi / 2)
        raise ValueError(f"Unknown schedule: {self.mask_schedule}")

    @torch.no_grad()
    def decode(
        self,
        x_tokens: torch.Tensor,
        type_ids: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, IterativeDecodingStats]:
        """
        MaskGIT-style iterative decoding.

        Args:
            x_tokens: Initial tokens [batch_size, seq_len]
            type_ids: Type IDs for filtering [batch_size, seq_len]
            pred_mask: Mask indicating what needs prediction [batch_size, seq_len]

        Returns:
            Tuple of (completed_tokens, stats)
        """
        self.model.eval()
        batch_size, seq_len = x_tokens.shape
        device = x_tokens.device

        current_tokens = x_tokens.clone()
        current_mask = pred_mask.clone()

        initial_n_masked = pred_mask.sum(dim=1)  # [batch_size]

        stats = IterativeDecodingStats()
        stats.total_masked = pred_mask.sum().item()

        for t in range(self.num_iterations):
            n_masked = current_mask.sum(dim=1)
            # early exit if all tokens committed
            if not current_mask.any():
                break

            # forward pass
            logits = self.model(current_tokens, type_ids)
            probs = F.softmax(logits, dim=-1)
            filtered_probs = self.vocab.filter_probs(probs, type_ids)

            if self.deterministic:
                # argmax (matches one-shot behavior)
                confidences, predictions = filtered_probs.max(dim=-1)
            else:
                # stochastic sampling with temperature and nucleus filtering
                scaled_logits = logits / self.temperature
                scaled_probs = F.softmax(scaled_logits, dim=-1)
                scaled_filtered = self.vocab.filter_probs(scaled_probs, type_ids)
                flat_probs = scaled_filtered.view(-1, scaled_filtered.shape[-1])
                flat_probs = self._nucleus_filter(flat_probs)
                sampled_flat = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)
                predictions = sampled_flat.view(batch_size, seq_len)
                # confidence = probability of the sampled token (from unscaled probs)
                confidences = filtered_probs.gather(
                    -1, predictions.unsqueeze(-1)
                ).squeeze(-1)
            confidences = torch.where(
                current_mask, confidences, torch.tensor(float("inf"), device=device)
            )

            is_last_iter = t == self.num_iterations - 1
            progress = (t + 1) / self.num_iterations
            target_mask_ratio = 0.0 if is_last_iter else self._gamma(progress)

            total_committed_this_iter = 0
            for b in range(batch_size):
                # TODO: vectorize this
                if n_masked[b] == 0:
                    continue

                # calculate how many tokens to commit
                n_remain = max(
                    0, math.ceil(target_mask_ratio * initial_n_masked[b].item())
                )
                n_remain = min(n_remain, n_masked[b].item())
                n_to_commit = n_masked[b].item() - n_remain
                n_to_commit = max(1, n_to_commit)
                n_to_commit = min(n_to_commit, n_masked[b].item())
                sample_mask = current_mask[b]
                masked_confidences = confidences[b][sample_mask]
                # select n_to_commit tokens to add to the prediction
                k = min(n_to_commit, masked_confidences.numel())
                _, topk_local_idx = masked_confidences.topk(k)
                masked_positions = sample_mask.nonzero(as_tuple=True)[0]
                commit_positions = masked_positions[topk_local_idx]
                # commit selected tokens
                current_tokens[b, commit_positions] = predictions[b, commit_positions]
                current_mask[b, commit_positions] = False
                total_committed_this_iter += len(commit_positions)

            stats.add_iteration(
                iteration=t,
                mask_ratio=target_mask_ratio,
                remaining=current_mask.sum().item(),
                committed=total_committed_this_iter,
                confidences=confidences,
                current_mask=current_mask,
            )

        return current_tokens, stats
