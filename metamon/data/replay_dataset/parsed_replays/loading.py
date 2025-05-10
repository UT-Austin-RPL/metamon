import os
import json
from typing import Optional
from datetime import datetime
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import tqdm

from metamon.interface import ObservationSpace, RewardFunction, UniversalState


class ParsedReplayDataset(Dataset):
    def __init__(
        self,
        dset_root: str,
        observation_space: ObservationSpace,
        reward_function: RewardFunction,
        formats: Optional[list[str]] = None,
        wins_losses_both: str = "both",
        min_rating: Optional[int] = None,
        max_rating: Optional[int] = None,
        min_date: Optional[datetime] = None,
        max_date: Optional[datetime] = None,
        verbose: bool = False,
    ):
        assert dset_root is not None and os.path.exists(dset_root)
        formats = formats or [
            f"gen{g}{t}" for g in range(1, 5) for t in ["ou", "uu", "nu", "ubers"]
        ]
        self.observation_space = observation_space
        self.reward_function = reward_function
        self.filenames = []

        def _rating_to_int(rating: str) -> int:
            try:
                return int(rating)
            except ValueError:
                return 1000

        bar = lambda it, desc: (
            it if not verbose else tqdm.tqdm(it, desc=desc, colour="red")
        )

        for format in formats:
            path = os.path.join(dset_root, format)
            for filename in bar(os.listdir(path), desc=f"Finding {format} battles"):
                if not filename.endswith(".json"):
                    continue
                battle_id, rating, p1_name, _, p2_name, mm_dd_yyyy, result = filename[
                    :-5
                ].split("_")
                rating = _rating_to_int(rating)
                date = datetime.strptime(mm_dd_yyyy, "%m-%d-%Y")
                battle_id = (
                    battle_id.replace("[", "").replace("]", "").replace(" ", "").lower()
                )
                if (
                    format not in battle_id
                    or (min_rating is not None and rating < min_rating)
                    or (max_rating is not None and rating > max_rating)
                    or (min_date is not None and date < min_date)
                    or (max_date is not None and date > max_date)
                    or (wins_losses_both == "wins" and result != "WIN")
                    or (wins_losses_both == "losses" and result != "LOSS")
                ):
                    continue
                self.filenames.append(os.path.join(path, filename))

        if verbose:
            print(f"Dataset contains {len(self.filenames)} battles")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        breakpoint()
        with open(self.filenames[i], "r") as f:
            data = json.load(f)
        states = [UniversalState.from_dict(s) for s in data["states"]]
        obs = [self.observation_space.state_to_obs(s) for s in states]
        actions = torch.LongTensor(data["actions"])
        rewards = torch.Tensor(
            [
                self.reward_function(s_t, s_t1)
                for s_t, s_t1 in zip(states[:-1], states[1:])
            ]
        )
        dones = torch.zeros_like(rewards, dtype=bool)
        dones[-1] = True
        return obs, actions, rewards, dones


if __name__ == "__main__":
    from metamon.interface import DefaultObservationSpace, DefaultShapedReward

    dset = ParsedReplayDataset(
        dset_root="/mnt/nfs_client/jake/metamon_parsed_hf_replays",
        observation_space=DefaultObservationSpace(),
        reward_function=DefaultShapedReward(),
        verbose=True,
    )
    print(len(dset))
    obs, actions, rewards, dones = dset[0]
