import re
from typing import Optional
from functools import lru_cache

from metamon.backend.replay_parser.exceptions import *


@lru_cache(2**13)
def clean_no_numbers(name: str) -> str:
    return "".join(char for char in str(name) if char.isalpha()).lower()


@lru_cache(2**13)
def clean_name(name: str) -> str:
    return "".join(char for char in name if char.isalnum()).lower()


def pokemon_name(name: str) -> str:
    return clean_name(name).strip()


def _cleanup_move_id(move_id: str) -> str:
    if move_id.startswith("hiddenpower"):
        # map all hidden power moves to the same name
        return "hiddenpower"
    elif move_id == "vicegrip":
        return "visegrip"
    # i think these are covered by clean_no_numbers now, but just in case...
    elif move_id.startswith("return"):
        return "return"
    elif move_id.startswith("frustration"):
        return "frustration"
    else:
        return move_id


def move_name(name: str) -> str:
    return _cleanup_move_id(clean_no_numbers(name)).strip()


def parse_hp_fraction(raw: str) -> tuple[int, int]:
    # Showdown may suffix the denominator with g/y/r (HP bar color hints).
    m = re.search(r"(\d+)/(\d+)(?:[gyr])?\b", raw)
    if not m:
        raise StrParsingException("parse_hp_fraction", raw)
    return int(m.group(1)), int(m.group(2))


def parse_condition(raw: str) -> str:
    condition = raw.replace("move:", "").lower().strip().replace(" ", "")
    if "ability" in raw or ":" in condition:
        raise StrParsingException("parse_condition", raw)
    return condition


def parse_move_from_extra(raw: str) -> str:
    move = re.sub(r"\[from\]\s?move:\s?", "", raw).strip()
    if "[from]" not in raw or "move" not in raw or not move:
        raise StrParsingException("parse_move_from_extra", raw)
    return move


def parse_mon_from_extra(raw: str) -> str:
    mon = re.sub(r"\[of\]\s?", "", raw).strip()
    if "[of]" not in raw or not mon:
        raise StrParsingException("parse_mon_from_extra", raw)
    return mon


def parse_ability_from_extra(raw: str) -> str:
    ability = re.sub(r"\[from\]\s?ability:\s?", "", raw).strip()
    if "[from]" not in raw or "ability" not in raw or not ability:
        raise StrParsingException("parse_ability_from_extra", raw)
    return ability


def parse_item_from_extra(raw: str) -> str:
    item = re.sub(r"\[from\]\s?item:\s?", "", raw).strip()
    if "[from]" not in raw or "item" not in raw or not item:
        raise StrParsingException("parse_item_from_extra", raw)
    return item


def parse_extra(raw: str) -> str:
    *_, info = raw.split("[from]")
    if "[from]" not in raw or not info:
        raise StrParsingException("parse_extra", raw)
    info = info.strip()
    if info.startswith("move:"):
        info = info.replace("move:", "").strip()
    return info


def parse_ability(raw: str) -> str:
    ability = raw.replace("ability:", "").strip()
    if not ability or ":" in ability:
        raise StrParsingException("parse_ability", raw)
    return ability


def parse_item(raw: str) -> str:
    item = raw.replace("item:", "").strip()
    if not item or ":" in item:
        raise StrParsingException("parse_item", raw)
    return item


def parse_from_effect_of(message: list[str]) -> tuple[Optional[str]]:
    """
    for the misc. "[from] item/ability/move [of] id: pokemon" messages
    """
    item, ability, move, of_pokemon = None, None, None, None
    for arg in message:
        if "[from]" in arg:
            if "item" in arg:
                item = parse_item_from_extra(arg)
            if "ability" in arg:
                ability = parse_ability_from_extra(arg)
            if "move" in arg:
                move = parse_move_from_extra(arg)
        if "[of]" in arg:
            of_pokemon = parse_mon_from_extra(arg)
    return item, ability, move, of_pokemon
