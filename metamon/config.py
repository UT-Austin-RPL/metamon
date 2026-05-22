import os

# officially supported Showdown rulesets.
# gen1-4 nu/uu/ubers were supported by the original metamon paper,
# but their rules change often and their datasets are rarely updated,
# so our models would be very out-of-sync with current metagame.
SUPPORTED_BATTLE_FORMATS = [
    "gen1ou",
    "gen1uu",
    "gen1nu",
    "gen1ubers",
    "gen2ou",
    "gen2uu",
    "gen2nu",
    "gen2ubers",
    "gen3ou",
    "gen3uu",
    "gen3nu",
    "gen3ubers",
    "gen4ou",
    "gen4uu",
    "gen4nu",
    "gen4ubers",
    "gen9ou",
]


# play format A without OOD inputs by telling the agent it is playing format B.
FORMAT_ALIASES = {
    "gen1oulongtimer": "gen1ou",
    "gen9oulongtimer": "gen9ou",
}


def format_for_agent(fmt: str) -> str:
    """Lets metamon play non-standard Showdown formats by pretending they're something else"""
    return FORMAT_ALIASES.get(fmt.lower(), fmt.lower())


METAMON_CACHE_DIR = os.environ.get("METAMON_CACHE_DIR", None)
