import os

DATA_PATH = os.path.dirname(__file__)

__all__ = [
    "DATA_PATH",
    "MetamonDataset",
    "ParsedReplayDataset",
    "SelfPlayDataset",
    "raw_replay_util",
]


def __getattr__(name: str):
    if name == "MetamonDataset":
        from .parsed_replay_dset import MetamonDataset

        return MetamonDataset
    if name == "ParsedReplayDataset":
        from .parsed_replay_dset import ParsedReplayDataset

        return ParsedReplayDataset
    if name == "SelfPlayDataset":
        from .parsed_replay_dset import SelfPlayDataset

        return SelfPlayDataset
    if name == "raw_replay_util":
        from . import raw_replay_util

        return raw_replay_util
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
