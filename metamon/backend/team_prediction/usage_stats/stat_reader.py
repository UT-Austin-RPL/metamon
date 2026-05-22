import os
import re
import json
import datetime
import functools
import warnings
from typing import Optional, List

from termcolor import colored
import metamon
from metamon.config import format_for_agent
from metamon.data.download import download_usage_stats
from metamon.backend.team_prediction.usage_stats.format_rules import (
    get_valid_pokemon,
    Tier,
)
from metamon.backend.replay_parser.str_parsing import pokemon_name
from metamon.backend.showdown_dex.dex import Dex

TIER_MAP = {
    "ubers": Tier.UBERS,
    "ou": Tier.OU,
    "uu": Tier.UU,
    "ru": Tier.RU,
    "nu": Tier.NU,
    "pu": Tier.PU,
    "lc": Tier.LC,
}

EARLIEST_USAGE_STATS_DATE = datetime.date(2014, 1, 1)
LATEST_USAGE_STATS_DATE = datetime.date(2026, 4, 1)
DEFAULT_USAGE_RANK = 1500

ELITE_REPLAY_SOURCES = ("smogtours",)


class UsageStatsLoadError(Exception):
    """Raised when usage stats cannot be loaded at the requested rank/time window."""


def assert_usage_rank_available(format: str, rank: int) -> None:
    available = list_available_usage_ranks(format)
    if not available:
        raise UsageStatsLoadError(
            f"No usage rank directories found for {format}. "
            f"Run `python -m metamon download usage-stats`."
        )
    if rank not in available:
        raise UsageStatsLoadError(
            f"Usage rank {rank} is not available for {format}. "
            f"Available ranks on disk: {available}."
        )


def assert_usage_stats_loaded(
    format: str,
    start_date: datetime.date,
    end_date: datetime.date,
    requested_rank: int,
    stats: "PreloadedSmogonUsageStats",
) -> None:
    if stats.rank != requested_rank:
        raise UsageStatsLoadError(
            f"Usage stats for {format} fell back from rank={requested_rank} "
            f"to rank={stats.rank} between {start_date} and {end_date}. "
            f"Refusing to use a different skill tier silently."
        )
    if not stats.movesets:
        raise UsageStatsLoadError(
            f"Usage stats for {format} at rank={requested_rank} "
            f"between {start_date} and {end_date} loaded empty movesets."
        )


def _is_elite_replay_source(gameid: Optional[str]) -> bool:
    if not gameid:
        return False
    gameid_lower = gameid.lower()
    return any(src in gameid_lower for src in ELITE_REPLAY_SOURCES)


def resolve_effective_rating(
    rating: Optional[int | str],
    gameid: Optional[str],
    format: str,
) -> int:
    if isinstance(rating, int):
        return rating
    if _is_elite_replay_source(gameid):
        ranks = list_available_usage_ranks(format)
        positive = [r for r in ranks if r > 0]
        if positive:
            return max(positive)
    return DEFAULT_USAGE_RANK


def rating_to_usage_rank(effective_rating: int, available_ranks: List[int]) -> int:
    if not available_ranks:
        raise UsageStatsLoadError(
            "Cannot map rating to usage rank: no rank directories found on disk."
        )
    eligible = [r for r in available_ranks if r <= effective_rating]
    return max(eligible) if eligible else min(available_ranks)


def resolve_usage_rank(
    format: str,
    rating: Optional[int | str] = None,
    gameid: Optional[str] = None,
) -> int:
    available_ranks = list_available_usage_ranks(format)
    if not available_ranks:
        raise UsageStatsLoadError(
            f"No usage rank directories found for {format}. "
            f"Run `python -m metamon download usage-stats`."
        )
    effective = resolve_effective_rating(rating, gameid, format)
    rank = rating_to_usage_rank(effective, available_ranks)
    assert_usage_rank_available(format, rank)
    return rank


def rank_from_moveset_filename(fmt: str, filename: str) -> Optional[int]:
    """
    Extract the baseline/rank from a Smogon moveset filename.
    Examples: gen1ou-0.txt, gen1ou-1500.txt, gen1ou-1630.txt, gen1ou-1760.txt
    Returns rank as int or None if not applicable.
    """
    if not filename.startswith(fmt):
        return None
    if filename.endswith(".txt.gz") or filename.endswith(".gz"):
        return None
    if not filename.endswith(".txt"):
        return None

    stem = filename[:-4]
    if stem == fmt:
        # Smogon convention: no explicit baseline means 1500.
        return 1500

    m = re.match(rf"^{re.escape(fmt)}-(\d+(?:\.\d+)?)$", stem)
    if not m:
        return None
    return int(float(m.group(1)))


def list_available_ranks_in_moveset_dir(moveset_dir: str, fmt: str) -> List[int]:
    if not os.path.isdir(moveset_dir):
        return []
    ranks = set()
    for fn in os.listdir(moveset_dir):
        r = rank_from_moveset_filename(fmt, fn)
        if r is not None:
            ranks.add(r)
    return sorted(ranks)


def list_available_usage_ranks(format: str) -> List[int]:
    """
    List available baseline/rank subdirectories in the processed usage-stats dataset
    for a given format (e.g., gen4ou).
    """
    gen, tier = int(format[3]), format[4:]
    usage_stats_path = download_usage_stats(gen)
    base = os.path.join(usage_stats_path, "movesets_data", f"gen{gen}", f"{tier}")
    if not os.path.isdir(base):
        return []
    ranks = []
    for d in os.listdir(base):
        if os.path.isdir(os.path.join(base, d)) and re.fullmatch(r"\d+", d):
            ranks.append(int(d))
    return sorted(ranks)


def parse_pokemon_moveset(file_path):
    moveset_data_list = {
        "name": [],
        "count": [],
        "abilities": [],
        "items": [],
        "spreads": [],
        "moves": [],
        "tera_types": [],
        "teammates": [],
        "checks": [],
    }

    def p_name(data_cache):
        if not data_cache:
            return moveset_data_list
        name = data_cache[0][2:-2].strip()
        moveset_data_list["name"].append(name)
        return moveset_data_list

    def p_count(data_cache):
        if not data_cache:
            return moveset_data_list
        count = int(data_cache[0][2:-2].split(":")[1].strip())
        moveset_data_list["count"].append(count)
        return moveset_data_list

    def p_abilities(data_cache):
        if not data_cache:
            moveset_data_list["abilities"].append({})
            return moveset_data_list
        _abilities = {}
        assert "Abilities" in data_cache[0], "Abilities not found"
        for line in data_cache[1:]:
            line_split = line[2:-2].strip().split()
            name = " ".join(line_split[:-1])
            percent = line_split[-1]
            # percent is in string format of xx%
            _abilities[name] = float(percent[:-1]) / 100
        moveset_data_list["abilities"].append(_abilities)
        # assert _abilities, "abilities is empty"
        return moveset_data_list

    def p_items(data_cache):
        if not data_cache:
            moveset_data_list["items"].append({})
            return moveset_data_list
        _items = {}
        assert "Items" in data_cache[0], "Items not found"
        for line in data_cache[1:]:
            line_split = line[2:-2].strip().split()
            name = " ".join(line_split[:-1])
            percent = line_split[-1]
            # percent is in string format of xx%
            _items[name] = float(percent[:-1]) / 100
        moveset_data_list["items"].append(_items)
        # assert _items, "items is empty"
        return moveset_data_list

    def p_spreads(data_cache):
        if not data_cache:
            moveset_data_list["spreads"].append({})
            return moveset_data_list
        _spreads = {}
        assert "Spreads" in data_cache[0], "Spreads not found"
        for line in data_cache[1:]:
            nature_ev, percent = line[2:-2].strip().split()
            # percent is in string format of xx%
            _spreads[nature_ev] = float(percent[:-1]) / 100
        moveset_data_list["spreads"].append(_spreads)
        # assert _spreads, "spreads is empty"
        return moveset_data_list

    def p_moves(data_cache):
        if not data_cache:
            moveset_data_list["moves"].append({})
            return moveset_data_list
        _moves = {}
        assert "Moves" in data_cache[0], "Moves not found"
        for line in data_cache[1:]:
            line_split = line[2:-2].strip().split()
            name = " ".join(line_split[:-1])
            percent = line_split[-1]
            _moves[name] = float(percent[:-1]) / 100
        # assert _moves, "moves is empty"
        moveset_data_list["moves"].append(_moves)
        return moveset_data_list

    def p_tera_types(data_cache):
        if not data_cache:
            moveset_data_list["tera_types"].append({})
            return moveset_data_list
        _tera_types = {}
        assert "Tera Types" in data_cache[0], "Tera Types not found"
        for line in data_cache[1:]:
            line_split = line[2:-2].strip().split()
            name = " ".join(line_split[:-1])
            percent = line_split[-1]
            _tera_types[name] = float(percent[:-1]) / 100
        moveset_data_list["tera_types"].append(_tera_types)
        return moveset_data_list

    def p_teammates(data_cache):
        if not data_cache:
            moveset_data_list["teammates"].append({})
            return moveset_data_list
        _teammates = {}
        assert "Teammates" in data_cache[0], "Teammates not found"
        for line in data_cache[1:]:
            line_split = line[2:-2].strip().split()
            name = " ".join(line_split[:-1])
            percent = line_split[-1]
            if percent.startswith("+") or percent.startswith("-"):
                percent = percent[1:]
            # sometimes the data will have a duplicate line and cause error, just skip that line
            try:
                _teammates[name] = float(percent[:-1]) / 100
            except:
                continue
        # assert _teammates, "teammates is empty"
        moveset_data_list["teammates"].append(_teammates)
        return moveset_data_list

    def p_checks(data_cache):
        if not data_cache:
            moveset_data_list["checks"].append({})
            return moveset_data_list
        _checks = {}
        assert "Checks and Counters" in data_cache[0], "Checks and Counters not found"
        for i in range(1, len(data_cache), 2):
            line = data_cache[i]
            line_split = line[2:-2].strip().split()
            name = " ".join(line_split[:-2])
            percent = line_split[-2]
            # percent is in string format of xx%
            _checks[name] = float(percent) / 100
        moveset_data_list["checks"].append(_checks)
        # assert _checks, "checks is empty"
        return moveset_data_list

    with open(file_path, "r") as file:
        file_content = file.read()

    if "Tera Types" in file_content:
        section_order = [
            p_name,
            p_count,
            p_abilities,
            p_items,
            p_spreads,
            p_moves,
            p_tera_types,
            p_teammates,
            p_checks,
            lambda _: None,
        ]
    else:
        section_order = [
            p_name,
            p_count,
            p_abilities,
            p_items,
            p_spreads,
            p_moves,
            p_teammates,
            p_checks,
            lambda _: None,
        ]

    lines = file_content.split("\n")
    current_section = -1

    data_cache = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("+-"):
            section_order[current_section](data_cache)
            current_section = (current_section + 1) % len(section_order)
            data_cache = []
        elif stripped.startswith("|"):
            if (
                current_section >= 0
                and section_order[current_section] is section_order[-1]
            ):
                # After the trailing noop section (double boundary between
                # Pokemon), the next row is the following Pokemon's name.
                current_section = 0
                data_cache = [" " + stripped]
            else:
                data_cache.append(" " + stripped)

    n = len(moveset_data_list["name"])
    if len(moveset_data_list["tera_types"]) == 0:
        moveset_data_list["tera_types"] = [{"Nothing": 1.0} for _ in range(n)]
    else:
        while len(moveset_data_list["tera_types"]) < n:
            moveset_data_list["tera_types"].append({"Nothing": 1.0})

    for key, default in (
        ("count", 0),
        ("abilities", {}),
        ("items", {}),
        ("spreads", {}),
        ("moves", {}),
        ("teammates", {}),
        ("checks", {}),
    ):
        while len(moveset_data_list[key]) < n:
            moveset_data_list[key].append(default)

    moveset_data = {}
    for i in range(n):
        _name = moveset_data_list["name"][i]
        _count = moveset_data_list["count"][i]
        _abilities = moveset_data_list["abilities"][i]
        _items = moveset_data_list["items"][i]
        _spreads = moveset_data_list["spreads"][i]
        _moves = moveset_data_list["moves"][i]
        _tera_types = moveset_data_list["tera_types"][i]
        _teammates = moveset_data_list["teammates"][i]
        _checks = moveset_data_list["checks"][i]
        moveset_data[_name] = {
            "count": _count,
            "abilities": _abilities,
            "items": _items,
            "spreads": _spreads,
            "moves": _moves,
            "tera_types": _tera_types,
            "teammates": _teammates,
            "checks": _checks,
        }
    return moveset_data


def merge_movesets(movesets):
    result = {}
    for moveset in movesets:
        for pokemon, data in moveset.items():
            # add up count first
            count = data["count"]
            if pokemon not in result:
                result[pokemon] = {}
                result[pokemon]["count"] = 0
            for key, value in data.items():
                if key == "count":
                    result[pokemon][key] += value
                    continue
                # key is the entry name, e.g. "abilities", "items", "spreads", "moves", "teammates", "checks"
                if key not in result[pokemon]:
                    result[pokemon][key] = {}
                # k is the counted thing, v is the percentage
                for k, v in value.items():
                    if k not in result[pokemon][key]:
                        result[pokemon][key][k] = 0
                    result[pokemon][key][k] += v * count

    # recalculating the percentage
    for pokemon, data in result.items():
        count = data["count"]
        for key, value in data.items():
            if key != "count":
                for k, v in value.items():
                    result[pokemon][key][k] /= count

    return result


class SmogonStat:
    def __init__(
        self,
        format: str,
        raw_stats_dir: str,
        date=None,
        rank: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        if date and type(date) == str:
            dates = [date]
        elif date and type(date) == list:
            dates = date
        else:
            dates = os.listdir(raw_stats_dir)

        self.data_paths = [os.path.join(raw_stats_dir, date) for date in dates]
        self.format = format
        self.rank = rank
        self.verbose = verbose

        self._movesets = {}
        self._inclusive = {}
        self._usage = None
        self._available_ranks: List[int] = []
        self._load()
        self._name_conversion = {
            pokemon_name(pokemon): pokemon for pokemon in self._movesets.keys()
        }

    @staticmethod
    def available_ranks(format: str, raw_stats_dir: str, date: str) -> List[int]:
        moveset_dir = os.path.join(raw_stats_dir, date, "moveset")
        return list_available_ranks_in_moveset_dir(moveset_dir, format)

    @property
    def available_ranks_loaded(self) -> List[int]:
        return list(self._available_ranks)

    def _load(self):
        moveset_paths = []
        for data_path in self.data_paths:
            moveset_path = os.path.join(data_path, "moveset")
            if os.path.exists(moveset_path):
                moveset_paths.append(moveset_path)

        if len(moveset_paths) == 0:
            if self.verbose:
                print(f"No moveset data found for {self.format} in {self.data_paths}")
            self._movesets = {}
            self._available_ranks = []
            return

        _movesets = []
        ranks_seen = set()
        for moveset_path in moveset_paths:
            files_by_rank = {}
            for fn in os.listdir(moveset_path):
                r = rank_from_moveset_filename(self.format, fn)
                if r is None:
                    continue
                files_by_rank.setdefault(r, []).append(fn)

            ranks_seen.update(files_by_rank.keys())

            if not files_by_rank:
                continue

            if self.rank is None:
                available = sorted(files_by_rank.keys())
                raise ValueError(
                    f"SmogonStat requires a baseline/rank for {self.format}. "
                    f"Available ranks in {moveset_path}: {available}"
                )
            filenames = files_by_rank.get(self.rank, [])

            for fn in filenames:
                fp = os.path.join(moveset_path, fn)
                try:
                    _movesets.append(parse_pokemon_moveset(fp))
                except Exception as e:
                    if self.verbose:
                        warnings.warn(colored(f"Failed parsing {fp}: {e}", "red"))

        self._available_ranks = sorted(ranks_seen)

        if not _movesets:
            self._movesets = {}
            return

        self._movesets = {
            pokemon_name(k): v for k, v in merge_movesets(_movesets).items()
        }

    def pretty_print(self, key):
        data = self[key]
        print(f"------ {key} {data['count']} ------\n")
        print("\tMoveset:")
        sorted_moves = sorted(data["moves"].items(), key=lambda x: x[1], reverse=True)
        for i, (move, usage) in enumerate(sorted_moves[:10]):
            print(f"\t\t{i+1}. {move} ({usage * 100: .1f}%)")
        print("\n\tTeammates:")
        sorted_mates = sorted(
            data["teammates"].items(), key=lambda x: x[1], reverse=True
        )
        for i, (mate, usage) in enumerate(sorted_mates[:5]):
            print(f"\t\t{i+1}. {mate} ({usage * 100: .1f}%)")
        print("\n\tChecks:")
        sorted_checks = sorted(data["checks"].items(), key=lambda x: x[1], reverse=True)
        for i, (check, usage) in enumerate(sorted_checks[:5]):
            print(f"\t\t{i+1}. {check} ({usage * 100: .1f}%)")

    def remove_banned_pm(self):
        valid_pm_dict = get_valid_pokemon(self.format)
        tier = TIER_MAP[self.format[4:]]
        valid_pm = []
        # get all pokemon valid in this tier
        for t in valid_pm_dict.keys():
            if t >= tier:
                valid_pm.extend(valid_pm_dict[t])

        if self.verbose:
            print(f"Total {len(valid_pm)} valid pokemon for {self.format}")
        # remove pokemon that not in this tier
        for pm in list(self._movesets.keys()):
            if re.sub(r"[^a-zA-Z0-9]", "", pm) not in valid_pm:
                if self.verbose:
                    print(f"Remove {pm} from {self.format}")
                del self._movesets[pm]

    @property
    def movesets(self):
        return self._movesets

    @property
    def usage(self):
        if self._usage is None:
            # create a list of pokemon names, sorted by count
            self._usage = list(
                sorted(
                    self._movesets.keys(),
                    key=lambda x: self._movesets[x]["count"],
                    reverse=True,
                )
            )
        return self._usage


def load_between_dates(
    dir_path: str,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    warn_if_empty: bool = True,
) -> dict:
    start_date = datetime.date(start_year, start_month, 1)
    end_date = datetime.date(end_year, end_month, 1)

    if start_date > LATEST_USAGE_STATS_DATE:
        start_date = LATEST_USAGE_STATS_DATE
    elif start_date < EARLIEST_USAGE_STATS_DATE:
        start_date = EARLIEST_USAGE_STATS_DATE

    if end_date > LATEST_USAGE_STATS_DATE:
        end_date = LATEST_USAGE_STATS_DATE
    elif end_date < EARLIEST_USAGE_STATS_DATE:
        end_date = EARLIEST_USAGE_STATS_DATE

    selected_data = []
    for json_file in os.listdir(dir_path):
        if not json_file.endswith(".json"):
            continue
        year, month = json_file.replace(".json", "").split("-")
        date = datetime.date(year=int(year), month=int(month), day=1)
        if not start_date <= date <= end_date:
            continue
        with open(os.path.join(dir_path, json_file), "r") as file:
            data = json.load(file)
        selected_data.append(data)
    if not selected_data and warn_if_empty:
        warnings.warn(
            colored(
                f"No Showdown usage stats found in {dir_path} between {start_date} and {end_date}",
                "red",
            )
        )
    return merge_movesets(selected_data)


class PreloadedSmogonUsageStats(SmogonStat):
    def __init__(
        self,
        format,
        start_date: datetime.date,
        end_date: datetime.date,
        rank: int = DEFAULT_USAGE_RANK,
        load_nearest_lower_rank: bool = True,
        search_lower_ranks_on_miss: bool = True,
        verbose: bool = True,
    ):
        self.format = format.strip().lower()
        self.rank = int(rank)
        self.start_date = start_date
        self.end_date = end_date
        self.verbose = verbose
        self._usage = None
        gen, tier = int(self.format[3]), self.format[4:]
        self.gen = gen
        usage_stats_path = download_usage_stats(gen)
        movesets_base = os.path.join(
            usage_stats_path, "movesets_data", f"gen{gen}", f"{tier}"
        )
        inclusive_base = os.path.join(
            usage_stats_path, "movesets_data", f"gen{gen}", "all_tiers"
        )
        movesets_path = os.path.join(movesets_base, str(self.rank))
        inclusive_path = os.path.join(inclusive_base, str(self.rank))

        def _avail_ranks(base: str) -> list[int]:
            if not os.path.isdir(base):
                return []
            ranks = []
            for d in os.listdir(base):
                if os.path.isdir(os.path.join(base, d)) and re.fullmatch(r"\d+", d):
                    ranks.append(int(d))
            return sorted(ranks)

        def _nearest_lower_rank(target: int, candidates: list[int]) -> Optional[int]:
            lower = [r for r in candidates if r < target]
            return max(lower) if lower else None

        if not os.path.isdir(movesets_path):
            avail = _avail_ranks(movesets_base)
            fallback_rank = (
                _nearest_lower_rank(self.rank, avail)
                if load_nearest_lower_rank
                else None
            )
            if fallback_rank is not None:
                if self.verbose:
                    warnings.warn(
                        colored(
                            f"Requested rank={self.rank} not found for {self.format}. "
                            f"Falling back to nearest rank={fallback_rank}.",
                            "yellow",
                        )
                    )
                self.rank = fallback_rank
                movesets_path = os.path.join(movesets_base, str(self.rank))
                inclusive_path = os.path.join(inclusive_base, str(self.rank))
            else:
                raise FileNotFoundError(
                    f"Movesets data not found for {self.format} at rank={self.rank}. "
                    f"Available ranks: {avail}. "
                    f"Run `python -m metamon download usage-stats` to get the latest data."
                )

        if not os.path.isdir(inclusive_path):
            avail = _avail_ranks(inclusive_base)
            raise FileNotFoundError(
                f"All-tiers movesets not found for gen{gen} at rank={self.rank}. "
                f"Available ranks: {avail}. "
                f"Run `python -m metamon download usage-stats` to get the latest data."
            )

        # data is split by year and month
        if not os.path.exists(movesets_path) or not os.path.exists(inclusive_path):
            raise FileNotFoundError(
                f"Movesets data not found for {format}. Run `python -m metamon download usage-stats` to download the data."
            )
        self._movesets = load_between_dates(
            movesets_path,
            start_year=start_date.year,
            start_month=start_date.month,
            end_year=end_date.year,
            end_month=end_date.month,
        )
        if not self._movesets:
            raise FileNotFoundError(
                f"No usage stats found for {self.format} at rank={self.rank} "
                f"between {start_date} and {end_date} in {movesets_path}."
            )
        self._inclusive = load_between_dates(
            inclusive_path,
            start_year=EARLIEST_USAGE_STATS_DATE.year,
            start_month=EARLIEST_USAGE_STATS_DATE.month,
            end_year=LATEST_USAGE_STATS_DATE.year,
            end_month=LATEST_USAGE_STATS_DATE.month,
        )
        self._lower_rank_fallbacks: list[tuple[int, dict, dict]] = []
        if search_lower_ranks_on_miss:
            avail = _avail_ranks(movesets_base)
            lower_ranks = [r for r in avail if r < self.rank]
            lower_ranks.sort(reverse=True)
            for r in lower_ranks:
                lower_movesets_path = os.path.join(movesets_base, str(r))
                lower_inclusive_path = os.path.join(inclusive_base, str(r))
                if not os.path.isdir(lower_movesets_path) or not os.path.isdir(
                    lower_inclusive_path
                ):
                    continue
                lower_movesets = load_between_dates(
                    lower_movesets_path,
                    start_year=start_date.year,
                    start_month=start_date.month,
                    end_year=end_date.year,
                    end_month=end_date.month,
                    warn_if_empty=False,
                )
                lower_inclusive = load_between_dates(
                    lower_inclusive_path,
                    start_year=EARLIEST_USAGE_STATS_DATE.year,
                    start_month=EARLIEST_USAGE_STATS_DATE.month,
                    end_year=LATEST_USAGE_STATS_DATE.year,
                    end_month=LATEST_USAGE_STATS_DATE.month,
                    warn_if_empty=False,
                )
                if lower_movesets or lower_inclusive:
                    self._lower_rank_fallbacks.append(
                        (r, lower_movesets, lower_inclusive)
                    )

    def _load(self):
        pass

    def _inclusive_search(self, key):
        # check the stats for this specific tier and time period first
        key_id = pokemon_name(key)
        recent = self._movesets.get(key_id, {})
        alltime = self._inclusive.get(key_id, {})
        if not (recent or alltime or self._lower_rank_fallbacks):
            return None

        no_info = {"Nothing": 1.0}

        def _apply_field_fallback(primary: dict, fallback: dict) -> dict:
            if not fallback:
                return primary
            if not primary:
                return fallback
            for field, value in fallback.items():
                if value == no_info:
                    continue
                if field not in primary or primary.get(field) == no_info:
                    primary[field] = value
            return primary

        # Start with tier stats for the requested rank; do not use all_tiers yet.
        primary = recent if recent else {}

        # First, walk downward through lower-rank tier stats.
        for _, lower_recent, _ in self._lower_rank_fallbacks:
            primary = _apply_field_fallback(primary, lower_recent.get(key_id, {}))

        # If still missing, fall back to all_tiers for the requested rank.
        primary = _apply_field_fallback(primary, alltime)

        # Finally, use lower-rank all_tiers as a last resort.
        for _, _, lower_alltime in self._lower_rank_fallbacks:
            primary = _apply_field_fallback(primary, lower_alltime.get(key_id, {}))

        return primary if primary else None

    def __getitem__(self, key):
        entry = Dex.from_gen(self.gen).get_pokedex_entry(key)
        species, base_species = entry.get("name", key), entry.get("baseSpecies", key)
        lookup = self._inclusive_search(species)
        if lookup is not None:
            return lookup
        lookup = self._inclusive_search(base_species)
        if lookup is not None:
            return lookup
        raise KeyError(f"Pokemon {key} not found in {self.format}")


def get_usage_stats(
    format,
    start_date: Optional[datetime.date] = None,
    end_date: Optional[datetime.date] = None,
    rank: int = DEFAULT_USAGE_RANK,
    load_nearest_lower_rank: bool = True,
    search_lower_ranks_on_miss: bool = True,
) -> PreloadedSmogonUsageStats:
    format = format_for_agent(format)
    if start_date is None or start_date < EARLIEST_USAGE_STATS_DATE:
        start_date = EARLIEST_USAGE_STATS_DATE
    else:
        # force to start of months to prevent cache miss (we only have monthly stats anyway)
        start_date = datetime.date(start_date.year, start_date.month, 1)
    if end_date is None or end_date > LATEST_USAGE_STATS_DATE:
        end_date = LATEST_USAGE_STATS_DATE
    else:
        # force to start of months to prevent cache miss (we only have monthly stats anyway)
        end_date = datetime.date(end_date.year, end_date.month, 1)
    return _cached_smogon_stats(
        format,
        start_date,
        end_date,
        int(rank),
        load_nearest_lower_rank,
        search_lower_ranks_on_miss,
    )


@functools.lru_cache(maxsize=64)
def _cached_smogon_stats(
    format,
    start_date: datetime.date,
    end_date: datetime.date,
    rank: int,
    load_nearest_lower_rank: bool,
    search_lower_ranks_on_miss: bool,
):
    print(
        f"Loading usage stats for {format} between {start_date} and {end_date} (rank={rank})"
    )
    assert_usage_rank_available(format, rank)
    stats = PreloadedSmogonUsageStats(
        format=format,
        start_date=start_date,
        end_date=end_date,
        rank=rank,
        load_nearest_lower_rank=load_nearest_lower_rank,
        search_lower_ranks_on_miss=search_lower_ranks_on_miss,
        verbose=False,
    )
    assert_usage_stats_loaded(format, start_date, end_date, rank, stats)
    return stats


if __name__ == "__main__":
    stats = get_usage_stats(
        "gen9ou",
        datetime.date(2023, 1, 1),
        datetime.date(2025, 6, 1),
    )
    print(len(stats.usage))
    for mon in sorted(
        stats.movesets.keys(), key=lambda m: stats[m]["count"], reverse=True
    )[:5]:
        stats.pretty_print(mon)
        print()
