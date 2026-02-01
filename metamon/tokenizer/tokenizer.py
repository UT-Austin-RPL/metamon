import json
import os
from datetime import date

import numpy as np

from metamon import SUPPORTED_BATTLE_FORMATS
from metamon.backend.replay_parser.str_parsing import (
    clean_no_numbers,
    pokemon_name,
    move_name,
    clean_name,
)

UNKNOWN_TOKEN: int = -1


class PokemonTokenizer:
    def __init__(self):
        self._initial_ids: dict[str, int] = {}
        self._new_ids: dict[str, int] = {}
        self._frozen: bool = True
        self.name: str = "custom"

    def unfreeze(self):
        self._frozen = False

    def freeze(self):
        self._frozen = True

    def __len__(self):
        return len(self._initial_ids.keys()) + len(self._new_ids.keys())

    @property
    def all_words(self) -> list[str]:
        return list(self._initial_ids.keys()) + list(self._new_ids.keys())

    @property
    def new_token(self):
        return len(self)

    def __getitem__(self, string: str) -> int:
        if string in self._initial_ids:
            return self._initial_ids[string]
        if string in self._new_ids:
            return self._new_ids[string]
        return UNKNOWN_TOKEN

    def save_tokens_to_disk(self, path):
        with open(path, "w") as f:
            json.dump({**self._initial_ids, **self._new_ids}, f)

    def load_tokens_from_disk(self, path):
        with open(path, "r") as f:
            self._initial_ids = json.load(f)
        return self

    def load_tokens(self, tokens: dict[str, int]):
        self._initial_ids = tokens
        return self

    def add_token_for(self, string: str) -> None:
        if string in self._initial_ids:
            return
        if string in self._new_ids:
            return
        print(f"Adding: `{string}`")
        self._new_ids[string] = self.new_token

    def sort_tokens(self) -> None:
        self._new_ids = {
            k: i + len(self._initial_ids)
            for i, k in enumerate(sorted(self._new_ids.keys()))
        }

    def tokenize(self, text: str) -> np.ndarray:
        words = text.split(" ")
        if not self._frozen:
            for word in words:
                self.add_token_for(word)
        return np.array([self[word] for word in words], dtype=np.int32)


PREMADE_TOKEN_LISTS = {
    # pre-history token lists for backwards compatibility
    # with old models before release
    "allreplays-v1": "allreplaysv1.json",
    "allreplays-v2": "allreplaysv2.json",
    "allreplays-v3": "allreplaysv3.json",
    # post v1.0 official token lists -- now named by the observation space
    # they are confirmed to be compatible with
    "DefaultObservationSpace-v0": "DefaultObservationSpace-v0.json",
    # adds ~1k new words for gen 9
    "DefaultObservationSpace-v1": "DefaultObservationSpace-v1.json",
}


def get_tokenizer(choice: str) -> PokemonTokenizer:
    tokenizer = PokemonTokenizer()
    if choice not in PREMADE_TOKEN_LISTS:
        raise KeyError(
            f"`get_tokenizer` `choice = {choice}` is invalid. Options are: {list(PREMADE_TOKEN_LISTS.keys())}"
        )
    path = os.path.join(os.path.dirname(__file__), PREMADE_TOKEN_LISTS[choice])
    tokenizer.load_tokens_from_disk(path)
    tokenizer.name = choice
    return tokenizer


if __name__ == "__main__":
    from argparse import ArgumentParser
    import tqdm

    from metamon.interface import (
        get_observation_space,
        DefaultShapedReward,
        DefaultActionSpace,
    )
    from metamon.data import ParsedReplayDataset
    from metamon.backend.team_prediction.usage_stats import get_usage_stats

    parser = ArgumentParser()
    parser.add_argument("--parsed_replay_root", required=True)
    parser.add_argument("--start_tokens", type=str, default=None)
    parser.add_argument("--save_tokens", type=str, default=None)
    parser.add_argument("--obs_space", type=str, default="DefaultObservationSpace")
    args = parser.parse_args()

    tokenizer = PokemonTokenizer()
    tokenizer.unfreeze()
    if args.start_tokens:
        tokenizer.load_tokens_from_disk(args.start_tokens)

    # catch stray names from Smogon stats
    for format in SUPPORTED_BATTLE_FORMATS:
        stat = get_usage_stats(format)
        for pokemon_name_str, data in tqdm.tqdm(stat._inclusive.items()):
            tokenizer.add_token_for(pokemon_name(pokemon_name_str))

            for ability in data["abilities"]:
                ability = ability.strip()
                if ability != "No Ability":
                    tokenizer.add_token_for(clean_no_numbers(ability))

            for move in data["moves"]:
                move = move.strip()
                tokenizer.tokenize(move_name(move))

            for item in data["items"]:
                item = item.strip()
                if item != "Nothing":
                    tokenizer.tokenize(clean_no_numbers(item))

            for spread in data["spreads"]:
                nature = spread.split(":")[0].strip()
                tokenizer.tokenize(clean_no_numbers(nature))

    dset = ParsedReplayDataset(
        dset_root=args.parsed_replay_root,
        observation_space=get_observation_space(args.obs_space),
        action_space=DefaultActionSpace(),
        reward_function=DefaultShapedReward(),
        verbose=True,
        shuffle=True,
    )
    total_dataset_size = 0
    for obs_seq, *_ in tqdm.tqdm(dset):
        for text_obs in obs_seq["text"]:
            total_dataset_size += 1
            tokenizer.tokenize(text_obs.tolist())
    print(f"Total dataset size: {total_dataset_size}")

    tokenizer.sort_tokens()

    if args.save_tokens:
        tokenizer.save_tokens_to_disk(args.save_tokens)

    original_tokenizer = PokemonTokenizer()
    original_tokenizer.load_tokens_from_disk(args.start_tokens)
    new_tokenizer = PokemonTokenizer()
    new_tokenizer.load_tokens_from_disk(args.save_tokens)

    for token, id in original_tokenizer._initial_ids.items():
        if token not in new_tokenizer._initial_ids:
            print(f"Token `{token}` is missing from the new tokenizer")
        elif new_tokenizer._initial_ids[token] != id:
            print(f"Token `{token}` has the wrong id in the new tokenizer")

    for word in original_tokenizer.all_words:
        if word not in new_tokenizer.all_words:
            print(f"Word `{word}` is missing from the new tokenizer")
