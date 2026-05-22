import os
from importlib.metadata import version

__version__ = "1.6.0"

# ANSI color codes
_YELLOW = "\033[38;5;228m"
_BLUE = "\033[94m"
_CYAN = "\033[96m"
_RED = "\033[91m"
_WHITE = "\033[97m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"

_METAMON_LOGO_LINES = [
    r"    __  ___     __                            ",
    r"   /  |/  /__  / /_____ _____ ___  ____  ____ ",
    "  / /|_/ / _ \\/ __/ __ `/ __ `__ \\/ __ \\/ __ \\",
    " / /  / /  __/ /_/ /_/ / / / / / / /_/ / / / /",
    "/_/  /_/\\___/\\__/\\__,_/_/ /_/ /_/\\____/_/ /_/ ",
]


def print_banner():
    print(f'{_BLUE}╔{"═" * 60}╗{_RESET}')

    for line in _METAMON_LOGO_LINES:
        padding = 60 - len(line)
        print(
            f'{_BLUE}║{_RESET}{_YELLOW}{_BOLD}{line}{" " * padding}{_RESET}{_BLUE}║{_RESET}'
        )

    print(f'{_BLUE}╠{"═" * 60}╣{_RESET}')
    tagline = f"Pokémon Showdown RL  •  v{__version__}  •  UT-Austin-RPL/metamon"
    pad_left = (60 - len(tagline)) // 2
    pad_right = 60 - len(tagline) - pad_left
    print(
        f'{_BLUE}║{_RESET}{" " * pad_left}{_WHITE}{tagline}{_RESET}{" " * pad_right}{_BLUE}║{_RESET}'
    )
    print(f'{_BLUE}╚{"═" * 60}╝{_RESET}')
    print()


poke_env_version = version("poke-env")

if not os.environ.get("METAMON_ALLOW_ANY_POKE_ENV"):
    if poke_env_version != "0.8.3.3":
        raise ImportError(
            f"poke-env version {poke_env_version} is not officially supported.\n"
            f"Please install version '0.8.3.3', found here: https://github.com/UT-Austin-RPL/poke-env).\n"
            f"This error is here to prevent silent bugs. If you are sure you want to use a\n"
            f"different version of poke-env, set the METAMON_ALLOW_ANY_POKE_ENV environment\n"
            f"variable to True."
        )

from .config import SUPPORTED_BATTLE_FORMATS, METAMON_CACHE_DIR
