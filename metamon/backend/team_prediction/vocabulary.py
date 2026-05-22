import os
from functools import lru_cache
from typing import Optional
from collections import defaultdict
from datetime import date

import numpy as np
import torch
import tqdm
from poke_env.data import to_id_str

import metamon
from metamon.tokenizer import PokemonTokenizer, UNKNOWN_TOKEN
from metamon.backend.team_prediction.team import PokemonSet, TeamSet
from metamon.backend.team_prediction.usage_stats import get_usage_stats


def create_vocabularies(scan_dataset: bool = False):
    # Initialize tokenizers for each vocabulary type
    team_tokenizer = PokemonTokenizer()
    team_tokenizer.unfreeze()
    for gen in [1, 2, 3, 4, 9]:
        for tier in ["ou", "uu", "ubers", "nu"]:
            format = f"Format: gen{gen}{tier}"
            team_tokenizer.add_token_for(format)
    # Add special tokens
    team_tokenizer.add_token_for(f"Mon: {PokemonSet.MISSING_NAME}")
    team_tokenizer.add_token_for(f"Ability: {PokemonSet.MISSING_ABILITY}")
    team_tokenizer.add_token_for(f"Ability: {PokemonSet.NO_ABILITY}")
    team_tokenizer.add_token_for(f"Item: {PokemonSet.MISSING_ITEM}")
    team_tokenizer.add_token_for(f"Item: {PokemonSet.NO_ITEM}")
    team_tokenizer.add_token_for(f"Nature: {PokemonSet.MISSING_NATURE}")
    team_tokenizer.add_token_for(f"Nature: {PokemonSet.NO_NATURE}")
    team_tokenizer.add_token_for(f"Move: {PokemonSet.MISSING_MOVE}")
    team_tokenizer.add_token_for(f"Move: {PokemonSet.NO_MOVE}")
    team_tokenizer.add_token_for(f"EV: {PokemonSet.MISSING_EV}")
    team_tokenizer.add_token_for(f"IV: {PokemonSet.MISSING_IV}")
    team_tokenizer.add_token_for(f"Tera Type: {PokemonSet.MISSING_TERA_TYPE}")
    team_tokenizer.add_token_for(f"Tera Type: {PokemonSet.NO_TERA_TYPE}")

    # Populate vocabularies from Smogon stats
    for gen in [1, 2, 3, 4, 9]:
        for tier in ["ou", "uu", "ubers", "nu"]:
            format = f"gen{gen}{tier}"
            stat = get_usage_stats(
                format,
                start_date=date(2015, 1, 1),
                end_date=date(2025, 1, 1),
            )

            for pokemon_name, data in stat._inclusive.items():
                # because usage stats keys are now lowercase, we take the long way around
                # and add all the teammates as tokens
                for partner in data["teammates"]:
                    partner = partner.strip()
                    team_tokenizer.add_token_for(f"Mon: {partner}")

                for ability in data["abilities"]:
                    ability = ability.strip()
                    if ability != "No Ability":
                        team_tokenizer.add_token_for(f"Ability: {ability}")

                for move in data["moves"]:
                    move = move.strip()
                    # have to fold Hidden Power types into one move since
                    # that's how they'd appear in replays and team files.
                    if move.startswith("Hidden Power"):
                        move = "Hidden Power"
                    team_tokenizer.add_token_for(f"Move: {move}")

                for item in data["items"]:
                    item = item.strip()
                    if item != "Nothing":
                        team_tokenizer.add_token_for(f"Item: {item}")

                for spread in data["spreads"]:
                    nature = spread.split(":")[0].strip()
                    team_tokenizer.add_token_for(f"Nature: {nature}")

                for tera_type in data["tera_types"]:
                    tera_type = tera_type.strip()
                    if tera_type != "Nothing":
                        team_tokenizer.add_token_for(f"Tera Type: {tera_type}")

    # Optionally scan team files from the training dataset for any tokens not in usage stats
    if scan_dataset:
        data_dir = metamon.data.download.download_revealed_teams()
        index_path = os.path.join(data_dir, "index.csv")

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"index.csv not found at {index_path}. "
                "Run the dataset once with use_cached_filenames=False to generate it."
            )

        with open(index_path, "r") as f:
            lines = f.read().splitlines()[1:]  # skip header
        team_files = [os.path.join(data_dir, line) for line in lines if line]

        from metamon.backend.team_prediction.team import Team2Seq

        t2s = Team2Seq(include_stats=False)
        print(f"Scanning {len(team_files)} team files for unknown tokens...")
        for path in tqdm.tqdm(team_files, desc="Scanning team files"):
            try:
                format_str = to_id_str(os.path.splitext(path)[1].split("_")[0])
                team = TeamSet.from_showdown_file(path, format=format_str)
                seq, _ = t2s.to_seq(team)
                for token in seq:
                    team_tokenizer.add_token_for(token)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue

    # Sort all tokenizers
    team_tokenizer.sort_tokens()
    team_tokenizer.freeze()
    return team_tokenizer


class TeamTokenizer(PokemonTokenizer):
    def __init__(self):
        super().__init__()
        self._inv_data = None

    def load_tokens_from_disk(self, path):
        super().load_tokens_from_disk(path)
        self._inv_data = {v: k for k, v in self._initial_ids.items()}
        return self

    def tokenize(self, seq: list[str]) -> np.ndarray:
        for i, s in enumerate(seq):
            if s.startswith("Move:") and "Hidden Power" in s:
                seq[i] = "Move: Hidden Power"
        out = np.array([self[s] for s in seq], dtype=np.int32)
        for i, token in enumerate(out):
            if token == UNKNOWN_TOKEN:
                print(f"Unknown token: {seq[i]}")
        return out

    def invert(self, tokens: np.ndarray) -> list[str]:
        out = []
        for token in tokens:
            if token in self._inv_data:
                out.append(self._inv_data[token])
            else:
                out.append(f"<unknown>")
        return out


class Vocabulary:
    def __init__(self):
        vocab_path = os.path.join(os.path.dirname(__file__), "vocab.json")
        self.tokenizer = TeamTokenizer().load_tokens_from_disk(vocab_path)
        prefixes = [
            "Format:",
            "Mon:",
            "Ability:",
            "Item:",
            "Nature:",
            "Move:",
            "EV:",
            "IV:",
            "Tera Type:",
        ]
        for prefix in prefixes:
            attr_name = f"{prefix.lower().rstrip(':').replace(' ', '_')}_mask"
            setattr(
                self,
                attr_name,
                [
                    i
                    for i, token in enumerate(self.tokenizer.all_words)
                    if token.startswith(prefix)
                ],
            )

        self.missing_mask = [
            i
            for i, token in enumerate(self.tokenizer.all_words)
            if token.count("$") == 2
        ]

        self.masks = {
            "format": self.format_mask,
            "mon": self.mon_mask,
            "ability": self.ability_mask,
            "item": self.item_mask,
            "nature": self.nature_mask,
            "move": self.move_mask,
            "ev": self.ev_mask,
            "iv": self.iv_mask,
            "tera_type": self.tera_type_mask,
            "missing": self.missing_mask,
        }
        self.type_ids = defaultdict(
            lambda: UNKNOWN_TOKEN,
            {
                "Format": 0,
                "Mon": 1,
                "Ability": 2,
                "Item": 3,
                "Nature": 4,
                "Move": 5,
                "EV": 6,
                "IV": 7,
                "Tera Type": 8,
            },
        )
        # Importance weights for weighted accuracy metric
        self.attribute_weights = {
            "Mon": 3.0,
            "Move": 2.0,
            "Ability": 1.5,
            "Item": 1.5,
            "Tera Type": 1.5,
            "Nature": 1.0,
            "EV": 0.5,
            "IV": 0.5,
        }
        self.type_id_to_mask = {
            0: self.format_mask,
            1: self.mon_mask,
            2: self.ability_mask,
            3: self.item_mask,
            4: self.nature_mask,
            5: self.move_mask,
            6: self.ev_mask,
            7: self.iv_mask,
            8: self.tera_type_mask,
        }

        # Map format token IDs to generation numbers
        # e.g., token for "Format: gen1ou" -> 1
        self.format_token_to_gen = {}
        for token_id in self.format_mask:
            token_str = self.tokenizer.all_words[token_id]  # e.g., "Format: gen1ou"
            # Extract gen number from format string (format is "Format: genXtier")
            format_part = token_str.split(": ")[1] if ": " in token_str else token_str
            for gen in range(1, 10):
                if format_part.lower().startswith(f"gen{gen}"):
                    self.format_token_to_gen[token_id] = gen
                    break

    def pokeset_seq_to_ints(self, seq: list[str]) -> np.ndarray:
        tokens = self.tokenizer.tokenize(seq)
        type_ids = np.array(
            [self.type_ids[s.split(":")[0].strip()] for s in seq], dtype=np.int32
        )
        return tokens, type_ids

    def ints_to_pokeset_seq(self, ints: np.ndarray) -> list[str]:
        return self.tokenizer.invert(ints)

    def filter_probs(self, probs: torch.Tensor, type_ids: torch.Tensor) -> torch.Tensor:
        # probs: [batch_size, seq_len, vocab_size], type_ids: [batch_size, seq_len]
        # Initialize mask and flatten batch+seq dims
        B, L, V = probs.shape
        mask = torch.zeros_like(probs)

        # For each type_id, set allowed vocab indices
        for type_id, mask_indices in self.type_id_to_mask.items():
            # TODO: speedup
            if not mask_indices:
                continue
            for b in range(B):
                for l in range(L):
                    if type_ids[b, l] == type_id:
                        mask[b, l, mask_indices] = 1.0

        # ban prediction of missing values
        mask[:, :, self.missing_mask] = 0.0

        # renormalize probs
        filtered = probs * mask
        return filtered / (filtered.sum(dim=-1, keepdim=True) + 1e-10)


@lru_cache(maxsize=1)
def get_vocab() -> Vocabulary:
    """Cached Vocabulary singleton (read-only after init)."""
    return Vocabulary()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate vocabulary for team prediction"
    )
    parser.add_argument(
        "--scan-dataset",
        action="store_true",
        help="Scan the training dataset for additional tokens not in usage stats",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="vocab.json",
        help="Output path for the vocabulary file",
    )
    args = parser.parse_args()

    vocabularies = create_vocabularies(scan_dataset=args.scan_dataset)
    vocabularies.save_tokens_to_disk(args.output)
