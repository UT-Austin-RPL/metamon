import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Literal, Optional, List
from dataclasses import dataclass, field
import math

from metamon.backend.team_prediction.team import TeamSet, Team2Seq, PokemonSet


class Decoder(ABC):
    """Abstract base class for team prediction decoders."""

    @property
    @abstractmethod
    def num_iterations(self) -> int:
        """Number of decoding iterations."""
        pass

    @property
    @abstractmethod
    def vocab(self):
        """Vocabulary for token filtering."""
        pass


@dataclass
class IterativeDecodingStats:
    mask_ratios: List[float] = field(default_factory=list)
    remaining_counts: List[int] = field(default_factory=list)
    committed_counts: List[int] = field(default_factory=list)
    names_committed_counts: List[int] = field(default_factory=list)
    moves_committed_counts: List[int] = field(default_factory=list)
    confidences_per_iter: List[torch.Tensor] = field(default_factory=list)
    name_confidences_per_iter: List[torch.Tensor] = field(default_factory=list)
    move_confidences_per_iter: List[torch.Tensor] = field(default_factory=list)
    tokens_per_iter: List[torch.Tensor] = field(default_factory=list)
    # Uniqueness constraint diagnostics
    names_reset_counts: List[int] = field(default_factory=list)
    moves_reset_counts: List[int] = field(default_factory=list)
    num_iterations_used: int = 0
    total_masked: int = 0

    def add_iteration(
        self,
        iteration: int,
        mask_ratio: float,
        remaining: int,
        committed: int,
        names_committed: int,
        moves_committed: int,
        masked_confidences: torch.Tensor,
        name_confidences: torch.Tensor,
        move_confidences: torch.Tensor,
        names_reset: int,
        moves_reset: int,
        current_tokens: Optional[torch.Tensor] = None,
    ):
        self.mask_ratios.append(mask_ratio)
        self.remaining_counts.append(remaining)
        self.committed_counts.append(committed)
        self.names_committed_counts.append(names_committed)
        self.moves_committed_counts.append(moves_committed)
        self.confidences_per_iter.append(masked_confidences)
        self.name_confidences_per_iter.append(name_confidences)
        self.move_confidences_per_iter.append(move_confidences)
        self.names_reset_counts.append(names_reset)
        self.moves_reset_counts.append(moves_reset)
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
        self.moves_committed_counts: List[List[int]] = [
            [] for _ in range(num_iterations)
        ]
        self.confidences: List[List[torch.Tensor]] = [[] for _ in range(num_iterations)]
        self.name_confidences: List[List[torch.Tensor]] = [
            [] for _ in range(num_iterations)
        ]
        self.move_confidences: List[List[torch.Tensor]] = [
            [] for _ in range(num_iterations)
        ]
        # Uniqueness constraint diagnostics
        self.names_reset_counts: List[List[int]] = [[] for _ in range(num_iterations)]
        self.moves_reset_counts: List[List[int]] = [[] for _ in range(num_iterations)]

    def add_batch(self, stats: IterativeDecodingStats):
        self.total_masked += stats.total_masked
        if self.mask_ratios is None:
            # same for all batches
            self.mask_ratios = stats.mask_ratios
        for i, (
            remaining,
            committed,
            names_committed,
            moves_committed,
            conf,
            name_conf,
            move_conf,
            names_reset,
            moves_reset,
        ) in enumerate(
            zip(
                stats.remaining_counts,
                stats.committed_counts,
                stats.names_committed_counts,
                stats.moves_committed_counts,
                stats.confidences_per_iter,
                stats.name_confidences_per_iter,
                stats.move_confidences_per_iter,
                stats.names_reset_counts,
                stats.moves_reset_counts,
            )
        ):
            self.remaining_counts[i].append(remaining)
            self.committed_counts[i].append(committed)
            self.names_committed_counts[i].append(names_committed)
            self.moves_committed_counts[i].append(moves_committed)
            if len(conf) > 0:
                self.confidences[i].append(conf)
            if len(name_conf) > 0:
                self.name_confidences[i].append(name_conf)
            if len(move_conf) > 0:
                self.move_confidences[i].append(move_conf)
            self.names_reset_counts[i].append(names_reset)
            self.moves_reset_counts[i].append(moves_reset)

    def compute_results(self) -> dict:
        """Aggregate batch stats into a summary dict (mask ratios, commits, confidences)."""
        remaining_frac = []
        committed_per_iter = []
        names_committed_per_iter = []
        moves_committed_per_iter = []
        names_reset_per_iter = []
        moves_reset_per_iter = []
        if self.total_masked > 0:
            for i in range(self.num_iterations):
                if self.remaining_counts[i]:
                    remaining_frac.append(
                        sum(self.remaining_counts[i]) / self.total_masked
                    )
                    committed_per_iter.append(sum(self.committed_counts[i]))
                    names_committed_per_iter.append(sum(self.names_committed_counts[i]))
                    moves_committed_per_iter.append(sum(self.moves_committed_counts[i]))
                    names_reset_per_iter.append(sum(self.names_reset_counts[i]))
                    moves_reset_per_iter.append(sum(self.moves_reset_counts[i]))
                else:
                    break

        def _concat_tensors(tensor_lists):
            result = []
            for i in range(self.num_iterations):
                if tensor_lists[i]:
                    result.append(torch.cat(tensor_lists[i], dim=0))
                else:
                    result.append(torch.tensor([]))
            return result

        return {
            "mask_ratios": self.mask_ratios or [],
            "remaining_frac": remaining_frac,
            "committed_per_iter": committed_per_iter,
            "names_committed_per_iter": names_committed_per_iter,
            "moves_committed_per_iter": moves_committed_per_iter,
            "names_reset_per_iter": names_reset_per_iter,
            "moves_reset_per_iter": moves_reset_per_iter,
            "confidences": _concat_tensors(self.confidences),
            "name_confidences": _concat_tensors(self.name_confidences),
            "move_confidences": _concat_tensors(self.move_confidences),
        }


class OneShotDecoder(Decoder):
    """
    Single-pass decoding with temperature and nucleus sampling.

    Used for one-shot evaluation with the same sampling options as iterative decoding.
    """

    def __init__(
        self,
        model,
        temperature: float = 1.0,
        top_p: float = 0.9,
        deterministic: bool = False,
        include_stats: bool = False,
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.deterministic = deterministic
        self.t2s = Team2Seq(include_stats=include_stats)

    @property
    def num_iterations(self) -> int:
        return 1

    @property
    def vocab(self):
        return self.t2s.vocab

    def _nucleus_filter(self, probs: torch.Tensor) -> torch.Tensor:
        """Apply nucleus (top-p) filtering and renormalize."""
        if self.top_p >= 1.0:
            return probs
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs - sorted_probs > self.top_p
        sorted_probs[sorted_mask] = 0.0
        filtered_probs = torch.zeros_like(probs)
        filtered_probs.scatter_(-1, sorted_indices, sorted_probs)
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
        return filtered_probs

    @torch.no_grad()
    def decode(
        self,
        x_tokens: torch.Tensor,
        type_ids: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> torch.Tensor:
        self.model.eval()
        device = x_tokens.device

        logits = self.model(x_tokens, type_ids)

        # Apply temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature

        probs = torch.softmax(logits, dim=-1)

        # Filter by valid token types
        probs = self.vocab.filter_probs(probs, type_ids)

        # Apply nucleus sampling filter
        probs = self._nucleus_filter(probs)

        # Sample or argmax
        if self.deterministic:
            sampled = probs.argmax(dim=-1)
        else:
            flat_probs = probs.view(-1, probs.shape[-1])
            sampled = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)
            sampled = sampled.view(probs.shape[:-1])

        # Only replace masked positions
        pred_tokens = x_tokens.clone()
        pred_tokens[pred_mask] = sampled[pred_mask]

        return pred_tokens


class IterativeTeamDecoder(Decoder):
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
        self.model = model
        self._num_iterations = num_iterations
        self.mask_schedule = mask_schedule
        self.temperature = temperature
        self.top_p = top_p
        self.deterministic = deterministic
        self.t2s = Team2Seq(include_stats=include_stats)
        # cache special token IDs for uniqueness constraints
        self._missing_name_id = int(
            self.vocab.tokenizer.tokenize([f"Mon: {PokemonSet.MISSING_NAME}"])[0]
        )
        self._missing_move_id = int(
            self.vocab.tokenizer.tokenize([f"Move: {PokemonSet.MISSING_MOVE}"])[0]
        )
        self._no_move_id = int(
            self.vocab.tokenizer.tokenize([f"Move: {PokemonSet.NO_MOVE}"])[0]
        )

    @property
    def num_iterations(self) -> int:
        return self._num_iterations

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

    def _reset_duplicates(
        self,
        positions: List[int],
        tokens: torch.Tensor,
        mask: torch.Tensor,
        probs: torch.Tensor,
        missing_id: int,
        skip_ids: set,
    ) -> int:
        """Reset duplicate tokens at given positions, keeping highest confidence."""
        # Get visible (non-masked) tokens at these positions
        visible = [(p, tokens[p].item()) for p in positions if not mask[p]]
        if len(visible) < 2:
            return 0

        # Group by token value
        by_token = defaultdict(list)
        for pos, tok in visible:
            by_token[tok].append(pos)

        tokens_reset = 0
        for tok, pos_list in by_token.items():
            if len(pos_list) > 1 and tok not in skip_ids:
                # Keep highest confidence, reset others
                confs = [probs[p, tok].item() for p in pos_list]
                best_idx = confs.index(max(confs))
                for i, p in enumerate(pos_list):
                    if i != best_idx:
                        tokens[p] = missing_id
                        mask[p] = True
                        tokens_reset += 1
        return tokens_reset

    def _enforce_uniqueness_constraints(
        self,
        b: int,
        current_tokens: torch.Tensor,
        current_mask: torch.Tensor,
        filtered_probs: torch.Tensor,
    ) -> tuple[int, int]:
        """
        Enforce unique pokemon names and moves per pokemon (reset duplicates to $missing_*$).
        Returns (names_reset, moves_reset).
        """
        # 1. Pokemon names (6 positions, no duplicates allowed)
        names_reset = self._reset_duplicates(
            self.t2s.get_all_name_positions(),
            current_tokens[b],
            current_mask[b],
            filtered_probs[b],
            self._missing_name_id,
            skip_ids={self._missing_name_id},
        )

        # 2. Moves per Pokemon (4 positions each, <nomove> can repeat)
        moves_reset = 0
        for pokemon_idx in range(6):
            moves_reset += self._reset_duplicates(
                self.t2s.get_move_positions_for_pokemon(pokemon_idx),
                current_tokens[b],
                current_mask[b],
                filtered_probs[b],
                self._missing_move_id,
                skip_ids={self._missing_move_id, self._no_move_id},
            )

        return names_reset, moves_reset

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

            # Collect confidences by type for diagnostics
            iter_confidences = (
                confidences[current_mask].cpu()
                if current_mask.any()
                else torch.tensor([])
            )
            # Name confidences
            name_mask = current_mask & (current_type_ids == NAME_TYPE_ID)
            iter_name_confidences = (
                confidences[name_mask].cpu() if name_mask.any() else torch.tensor([])
            )
            # Move confidences
            move_mask = current_mask & (current_type_ids == MOVE_TYPE_ID)
            iter_move_confidences = (
                confidences[move_mask].cpu() if move_mask.any() else torch.tensor([])
            )

            total_committed_this_iter = 0
            total_names_committed_this_iter = 0
            total_moves_committed_this_iter = 0
            total_names_reset_this_iter = 0
            total_moves_reset_this_iter = 0
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
                masked_confs = confidences[b][masked_positions]

                # select top-k most confident tokens to commit
                _, topk_idx = masked_confs.topk(min(n_to_commit, masked_confs.numel()))
                commit_positions = masked_positions[topk_idx]

                if commit_positions.numel() == 0:
                    continue

                # count names and moves being committed
                commit_types = current_type_ids[b][commit_positions]
                names_in_commit = (commit_types == NAME_TYPE_ID).sum().item()
                moves_in_commit = (commit_types == MOVE_TYPE_ID).sum().item()

                # commit selected tokens
                current_tokens[b, commit_positions] = predictions[b, commit_positions]
                current_mask[b, commit_positions] = False
                total_committed_this_iter += len(commit_positions)
                total_names_committed_this_iter += names_in_commit
                total_moves_committed_this_iter += moves_in_commit

                # Enforce uniqueness: no duplicate pokemon names, no duplicate moves per pokemon
                # Only do this if we have more iterations to fix it
                if not is_last_iter:
                    names_reset, moves_reset = self._enforce_uniqueness_constraints(
                        b, current_tokens, current_mask, filtered_probs
                    )
                    total_committed_this_iter -= names_reset + moves_reset
                    total_names_committed_this_iter -= names_reset
                    total_moves_committed_this_iter -= moves_reset
                    total_names_reset_this_iter += names_reset
                    total_moves_reset_this_iter += moves_reset

                # re-sort to maintain canonical ordering
                perm = self._compute_resort_permutation(current_tokens[b])
                current_tokens[b] = self._apply_permutation(current_tokens[b], perm)
                current_type_ids[b] = self._apply_permutation(current_type_ids[b], perm)
                current_mask[b] = self._apply_permutation(current_mask[b], perm)

            tokens_for_viz = current_tokens.clone() if track_tokens else None

            stats.add_iteration(
                iteration=t,
                mask_ratio=target_mask_ratio,
                remaining=current_mask.sum().item(),
                committed=total_committed_this_iter,
                names_committed=total_names_committed_this_iter,
                moves_committed=total_moves_committed_this_iter,
                masked_confidences=iter_confidences,
                name_confidences=iter_name_confidences,
                move_confidences=iter_move_confidences,
                names_reset=total_names_reset_this_iter,
                moves_reset=total_moves_reset_this_iter,
                current_tokens=tokens_for_viz,
            )

        return current_tokens, stats
