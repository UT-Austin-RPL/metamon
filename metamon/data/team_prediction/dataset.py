import os
import random
import pathlib
from typing import List, Tuple, Optional, Union, Iterable, Literal, Dict, Set

import torch
from torch.utils.data import Dataset

from metamon.data.team_prediction.team import TeamSet, Roster, PokemonSet
from metamon.data.team_prediction.vocabulary import Vocabulary
from metamon.data import DATA_PATH
from poke_env.data import to_id_str


class TeamDataset(Dataset):
    def __init__(
        self, team_file_dir: str, format: str, max_teams: Optional[int] = None
    ):
        self.team_path = os.path.join(team_file_dir, f"{format}_teams")
        if not os.path.exists(self.team_path):
            raise ValueError(f"Team directory {self.team_path} does not exist")
        self.filenames = os.listdir(self.team_path)
        random.shuffle(self.filenames)
        if max_teams is not None:
            self.filenames = self.filenames[:max_teams]
        self.format = format

    def __len__(self):
        return len(self.filenames)

    def __getitem__(
        self, idx
    ) -> Tuple[TeamSet, Dict[str, PokemonSet], Set[str]] | None:
        team_file = self.filenames[idx]
        team_path_full = os.path.join(self.team_path, team_file)
        try:
            team = TeamSet.from_showdown_file(team_path_full, self.format)
        except Exception as e:
            print(f"Error loading team file {team_path_full}: {e}")
            return None
        pokemon_sets = {p.name: p for p in team.pokemon}
        team_roster = Roster(team.lead.name, frozenset(p.name for p in team.reserve))
        return (team, pokemon_sets, team_roster)


class TeamPredictionDataset(Dataset):
    def __init__(
        self,
        data_dir: Union[str, Iterable[str]],
        mask_pokemon_prob_range: Tuple[float, float],
        mask_attrs_prob_range: Tuple[float, float],
        split: Literal["train", "val"] = "train",
        validation_ratio: float = 0.1,
        seed: Optional[int] = None,
    ):
        """
        Args:
            data_dir: Directory or iterable of directories containing .team files (will be searched recursively)
            split: Whether this is the training or validation split
            validation_ratio: Fraction of data to use for validation
            mask_pokemon_prob_range: Range of probabilities to use for masking an entire Pokemon
            mask_attrs_prob_range: Range of probabilities to use for masking an indivudal attribute
            seed: Random seed for reproducibility
        """
        (
            self.mask_pokemon_prob_low,
            self.mask_pokemon_prob_high,
        ) = mask_pokemon_prob_range
        self.mask_attrs_prob_low, self.mask_attrs_prob_high = mask_attrs_prob_range
        assert self.mask_pokemon_prob_low <= self.mask_pokemon_prob_high
        assert self.mask_attrs_prob_low <= self.mask_attrs_prob_high
        assert 0 <= validation_ratio <= 1, "validation_ratio must be in [0, 1)"

        self.vocab = Vocabulary()
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        # Accept a string or an iterable of strings for data_dir
        if isinstance(data_dir, str):
            data_dirs = [data_dir]
        else:
            data_dirs = list(data_dir)

        # Collect all team files
        team_files_set = set()
        for d in data_dirs:
            d_path = pathlib.Path(d)
            for f in d_path.rglob("*"):
                if f.is_file() and f.suffix.endswith("team"):
                    team_files_set.add(str(f))
        all_team_files = sorted(team_files_set)

        # Create deterministic train/val split
        n_total = len(all_team_files)
        n_val = int(n_total * validation_ratio)

        # Use a separate random state for splitting to ensure same split regardless of other randomness
        split_rng = random.Random(seed)
        indices = list(range(n_total))
        split_rng.shuffle(indices)

        val_indices = set(indices[:n_val])

        # Assign files based on split
        if split == "train":
            self.team_files = [
                f for i, f in enumerate(all_team_files) if i not in val_indices
            ]
        else:  # val
            self.team_files = [
                f for i, f in enumerate(all_team_files) if i in val_indices
            ]

        print(f"Created {split} split with {len(self.team_files)} team files")

    def __len__(self) -> int:
        return len(self.team_files)

    def _mask(self, seq: list[str]) -> list[str]:
        mask_out = []
        for token in seq:
            if random.random() < self.mask_pokemon_prob:
                mask_out.append(self.vocab.special_tokens["Mon"])
            else:
                mask_out.append(token)
        return mask_out

    def __getitem__(self, idx: int) -> Tuple[TeamSet, TeamSet]:
        """
        Returns:
            x: Masked team
            x_type_ids: Type indicating ints (pokemon, ability, item, etc.)
            y: Complete team (ground truth)
            pred_mask: Mask indicating which values are eligible for loss function
        """
        path = self.team_files[idx]
        # Extract format from file extension (e.g. .gen4ou_team -> gen4ou)
        format = to_id_str(os.path.splitext(path)[1].split("_")[0])
        assert format.startswith("gen"), f"Invalid format: {format}"
        team = TeamSet.from_showdown_file(path, format=format)
        mask_pokemon_prob = random.uniform(
            self.mask_pokemon_prob_low, self.mask_pokemon_prob_high
        )
        mask_attrs_prob = random.uniform(
            self.mask_attrs_prob_low, self.mask_attrs_prob_high
        )
        x, y = team.to_prediction_pair(
            mask_pokemon_prob=mask_pokemon_prob,
            mask_attrs_prob=mask_attrs_prob,
        )
        x_seq, x_needs_pred = x.to_seq(include_stats=False)
        y_seq, y_needs_pred = y.to_seq(include_stats=False)
        # we will only train on values that are missing from x but provided by y
        pred_mask = torch.logical_and(
            torch.tensor(x_needs_pred), ~torch.tensor(y_needs_pred)
        )
        x_tokens, x_type_ids = self.vocab.pokeset_seq_to_ints(x_seq)
        y_tokens, y_type_ids = self.vocab.pokeset_seq_to_ints(y_seq)
        assert len(x_tokens) == len(x_type_ids)
        assert len(y_tokens) == len(y_type_ids)
        assert len(x_tokens) == (7 * 6) + 1
        assert (x_type_ids == y_type_ids).all()
        x_tokens = torch.from_numpy(x_tokens).long()
        x_type_ids = torch.from_numpy(x_type_ids).long()
        y_tokens = torch.from_numpy(y_tokens).long()
        return x_tokens, x_type_ids, y_tokens, pred_mask


class CompetitiveTeamPredictionDataset(TeamPredictionDataset):
    def __init__(
        self,
        mask_pokemon_prob_range: Tuple[float, float] = (0.1, 0.1),
        mask_attrs_prob_range: Tuple[float, float] = (0.1, 0.1),
    ):
        team_dirs = []
        for gen in range(1, 5):
            for tier in ["ou", "uu", "ubers", "nu"]:
                team_dirs.append(
                    os.path.join(DATA_PATH, "teams", f"gen{gen}", tier, "competitive")
                )
        super().__init__(
            data_dir=team_dirs,
            split="val",
            validation_ratio=1.0,
            mask_pokemon_prob_range=mask_pokemon_prob_range,
            mask_attrs_prob_range=mask_attrs_prob_range,
        )


if __name__ == "__main__":
    # Test dataset loading
    dataset = CompetitiveTeamPredictionDataset()

    print(f"Dataset size: {len(dataset)}")

    # Test loading a single item
    x, type_ids, y = dataset[0]
    print("\nMasked team (x):")
    print(x)
    print("\nType IDs:")
    print(type_ids)
    print("\nComplete team (y):")
    print(y)
