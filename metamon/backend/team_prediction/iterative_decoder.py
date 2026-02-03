import torch
import torch.nn.functional as F
from typing import Literal, Optional, List
from dataclasses import dataclass, field
import math

from metamon.backend.team_prediction.team import TeamSet, Team2Seq


@dataclass
class IterativeDecodingStats:
    mask_ratios: List[float] = field(default_factory=list)
    remaining_counts: List[int] = field(default_factory=list)
    committed_counts: List[int] = field(default_factory=list)
    names_committed_counts: List[int] = field(default_factory=list)
    confidences_per_iter: List[torch.Tensor] = field(default_factory=list)
    tokens_per_iter: List[torch.Tensor] = field(default_factory=list)
    num_iterations_used: int = 0
    total_masked: int = 0

    def add_iteration(
        self,
        iteration: int,
        mask_ratio: float,
        remaining: int,
        committed: int,
        names_committed: int,
        masked_confidences: torch.Tensor,
        current_tokens: Optional[torch.Tensor] = None,
    ):
        self.mask_ratios.append(mask_ratio)
        self.remaining_counts.append(remaining)
        self.committed_counts.append(committed)
        self.names_committed_counts.append(names_committed)
        self.confidences_per_iter.append(masked_confidences)
        if current_tokens is not None:
            self.tokens_per_iter.append(current_tokens.cpu().clone())
        self.num_iterations_used = iteration + 1


class IterativeStatsAccumulator:
    def __init__(self, num_iterations: int):
        self.num_iterations = num_iterations
        self.total_masked = 0
        self.mask_ratios: Optional[List[float]] = None
        self.remaining_counts: List[List[int]] = [[] for _ in range(num_iterations)]
        self.committed_counts: List[List[int]] = [[] for _ in range(num_iterations)]
        self.names_committed_counts: List[List[int]] = [
            [] for _ in range(num_iterations)
        ]
        self.confidences: List[List[torch.Tensor]] = [[] for _ in range(num_iterations)]

    def add_batch(self, stats: IterativeDecodingStats):
        self.total_masked += stats.total_masked
        if self.mask_ratios is None:
            # same for all batches
            self.mask_ratios = stats.mask_ratios
        for i, (remaining, committed, names_committed, conf) in enumerate(
            zip(
                stats.remaining_counts,
                stats.committed_counts,
                stats.names_committed_counts,
                stats.confidences_per_iter,
            )
        ):
            self.remaining_counts[i].append(remaining)
            self.committed_counts[i].append(committed)
            self.names_committed_counts[i].append(names_committed)
            if len(conf) > 0:
                self.confidences[i].append(conf)

    def compute_results(self) -> dict:
        """
        Returns:
            - mask_ratios: list[float] target mask ratio per iteration
            - remaining_frac: list[float] actual fraction remaining per iteration
            - committed_per_iter: list[int] total committed tokens per iteration
            - names_committed_per_iter: list[int] total names committed per iteration
            - confidences: list[Tensor] concatenated confidences per iteration
        """
        remaining_frac = []
        committed_per_iter = []
        names_committed_per_iter = []
        if self.total_masked > 0:
            for i in range(self.num_iterations):
                if self.remaining_counts[i]:
                    remaining_frac.append(
                        sum(self.remaining_counts[i]) / self.total_masked
                    )
                    committed_per_iter.append(sum(self.committed_counts[i]))
                    names_committed_per_iter.append(sum(self.names_committed_counts[i]))
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
            "names_committed_per_iter": names_committed_per_iter,
            "confidences": confidences,
        }


class IterativeTeamDecoder:
    """
    MaskGIT-style iterative decoding with re-sorting after each fill.

    After each iteration, the filled-in tokens are converted back to a TeamSet,
    re-sorted using Team2Seq to maintain the canonical ordering invariant
    (visible items first alphabetically, then masked items).
    """

    def __init__(
        self,
        model,
        num_iterations: int = 8,
        mask_schedule: Literal["linear", "cosine"] = "cosine",
        temperature: float = 1.0,
        top_p: float = 0.9,
        deterministic: bool = False,
        include_stats: bool = False,
    ):
        """
        Args:
            model: TeamTransformer model
            num_iterations: Number of refinement iterations (T)
            mask_schedule: How mask ratio decreases ("cosine" recommended)
            temperature: Sampling temperature (lower = more deterministic)
            top_p: Nucleus sampling threshold (keep smallest set with cumulative prob >= top_p)
            deterministic: If True, use argmax instead of sampling (for fair comparison with one-shot)
            include_stats: Whether sequences include nature/EVs/IVs
        """
        self.model = model
        self.num_iterations = num_iterations
        self.mask_schedule = mask_schedule
        self.temperature = temperature
        self.top_p = top_p
        self.deterministic = deterministic
        self.t2s = Team2Seq(include_stats=include_stats)

    @property
    def vocab(self):
        return self.t2s.vocab

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

    def _compute_resort_permutation(self, tokens: torch.Tensor) -> List[int]:
        """Compute permutation to put tokens in canonical order."""
        team = self.t2s.decode(tokens)
        return self.t2s.compute_permutation(team)

    def _apply_permutation(self, tensor: torch.Tensor, perm: List[int]) -> torch.Tensor:
        return tensor[perm]

    @torch.no_grad()
    def decode(
        self,
        x_tokens: torch.Tensor,
        type_ids: torch.Tensor,
        pred_mask: torch.Tensor,
        track_tokens: bool = False,
    ) -> tuple[torch.Tensor, IterativeDecodingStats]:
        """
        MaskGIT-style iterative decoding with re-sorting to deal with structured
        semi-ordered sequence format of our Pokemon teams.
        """
        self.model.eval()
        batch_size, seq_len = x_tokens.shape
        device = x_tokens.device

        current_tokens = x_tokens.clone()
        current_type_ids = type_ids.clone()
        current_mask = pred_mask.clone()

        initial_n_masked = pred_mask.sum(dim=1)  # [batch_size]

        stats = IterativeDecodingStats()
        stats.total_masked = pred_mask.sum().item()
        NAME_TYPE_ID = self.vocab.type_ids["Mon"]
        MOVE_TYPE_ID = self.vocab.type_ids["Move"]

        # initial state for visualization
        if track_tokens:
            stats.tokens_per_iter.append(current_tokens.cpu().clone())

        for t in range(self.num_iterations):
            n_masked = current_mask.sum(dim=1)
            # early exit if all tokens committed
            if not current_mask.any():
                break

            # forward pass
            logits = self.model(current_tokens, current_type_ids)
            probs = F.softmax(logits, dim=-1)
            filtered_probs = self.vocab.filter_probs(probs, current_type_ids)

            if self.deterministic:
                # argmax (matches one-shot behavior)
                confidences, predictions = filtered_probs.max(dim=-1)
            else:
                # stochastic sampling with temperature and nucleus filtering
                scaled_logits = logits / self.temperature
                scaled_probs = F.softmax(scaled_logits, dim=-1)
                scaled_filtered = self.vocab.filter_probs(
                    scaled_probs, current_type_ids
                )
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

            iter_confidences = (
                confidences[current_mask].cpu()
                if current_mask.any()
                else torch.tensor([])
            )

            total_committed_this_iter = 0
            total_names_committed_this_iter = 0
            for b in range(batch_size):
                if n_masked[b] == 0:
                    continue

                # how many tokens to commit
                n_remain = max(
                    0, math.ceil(target_mask_ratio * initial_n_masked[b].item())
                )
                n_remain = min(n_remain, n_masked[b].item())
                n_to_commit = n_masked[b].item() - n_remain
                n_to_commit = max(1, n_to_commit)
                n_to_commit = min(n_to_commit, n_masked[b].item())

                masked_positions = current_mask[b].nonzero(as_tuple=True)[0]
                masked_type_ids = current_type_ids[b][masked_positions]
                masked_confs = confidences[b][masked_positions]

                # identify which masked positions are Pokemon names
                is_name = masked_type_ids == NAME_TYPE_ID
                name_positions = masked_positions[is_name]
                name_confs = (
                    masked_confs[is_name] if name_positions.numel() > 0 else None
                )

                # select top-k by confidence
                _, topk_idx = masked_confs.topk(min(n_to_commit, masked_confs.numel()))
                candidate_positions = masked_positions[topk_idx]
                candidate_types = current_type_ids[b][candidate_positions]

                # at least one name is committed if any remain masked
                # (so we always make progress on unblocking attributes)
                names_in_candidates = candidate_positions[
                    candidate_types == NAME_TYPE_ID
                ]
                if name_positions.numel() > 0 and names_in_candidates.numel() == 0:
                    # no names in top-k, add the most confident name
                    best_name_idx = name_confs.argmax()
                    best_name_pos = name_positions[best_name_idx : best_name_idx + 1]
                    candidate_positions = torch.cat(
                        [candidate_positions, best_name_pos]
                    )

                # filter: allow attribute only if its Pokemon's name is visible
                # OR the name is also being committed this iteration
                candidate_types = current_type_ids[b][candidate_positions]
                cand_pokemon_idx = self.t2s.get_pokemon_indices(candidate_positions)
                cand_name_pos = self.t2s.get_name_positions(cand_pokemon_idx)

                # which Pokemon have names being committed this iteration?
                name_pokemon_idx = cand_pokemon_idx[candidate_types == NAME_TYPE_ID]

                # keep if: (1) it's a name, (2) its name is visible, (3) its name is being committed
                is_candidate_name = candidate_types == NAME_TYPE_ID
                name_already_visible = ~current_mask[b, cand_name_pos]
                name_being_committed = torch.isin(cand_pokemon_idx, name_pokemon_idx)

                keep = is_candidate_name | name_already_visible | name_being_committed
                commit_positions = candidate_positions[keep]

                if commit_positions.numel() == 0:
                    continue

                # count names being committed
                commit_types = current_type_ids[b][commit_positions]
                names_in_commit = (commit_types == NAME_TYPE_ID).sum().item()

                # commit selected tokens
                current_tokens[b, commit_positions] = predictions[b, commit_positions]
                current_mask[b, commit_positions] = False
                total_committed_this_iter += len(commit_positions)
                total_names_committed_this_iter += names_in_commit

                # re-sort only if names or moves were committed
                has_ordering_change = (
                    names_in_commit > 0 or (commit_types == MOVE_TYPE_ID).any()
                )
                if has_ordering_change:
                    perm = self._compute_resort_permutation(current_tokens[b])
                    current_tokens[b] = self._apply_permutation(current_tokens[b], perm)
                    current_type_ids[b] = self._apply_permutation(
                        current_type_ids[b], perm
                    )
                    current_mask[b] = self._apply_permutation(current_mask[b], perm)

            tokens_for_viz = current_tokens.clone() if track_tokens else None

            stats.add_iteration(
                iteration=t,
                mask_ratio=target_mask_ratio,
                remaining=current_mask.sum().item(),
                committed=total_committed_this_iter,
                names_committed=total_names_committed_this_iter,
                masked_confidences=iter_confidences,
                current_tokens=tokens_for_viz,
            )

        return current_tokens, stats
