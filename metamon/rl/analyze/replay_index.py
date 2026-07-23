"""Scan metamon replay directories and load individual battles for the analyze UI."""

from __future__ import annotations

import base64
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import lz4.frame

from metamon.interface import UniversalState
from metamon.rl.analyze.actions import action_label, legal_action_labels, sprite_id

_BLANK_MOVES = frozenset({"", "nomove", "none", "null"})


def _pretty_action(label: str) -> str:
    """Light display cleanup for move/switch labels."""
    text = (label or "").strip()
    if not text or text.lower() in _BLANK_MOVES or text.lower() == "unrevealed":
        return ""
    # "Switch tauros" / "bodyslam" → title-ish tokens
    return " ".join(part.capitalize() for part in text.replace("_", " ").split())


def _opponent_will_use(
    state: UniversalState, next_state: Optional[UniversalState]
) -> str:
    """Infer foe action this turn from the following observation.

    Species change must win over ``opponent_prev_move``: after a switch the next
    state's prev-move often belongs to the *incoming* mon's history (e.g. Alakazam's
    old Thunder Wave), not what the mon that started the turn did.
    """
    if next_state is None:
        return ""
    cur = (
        state.opponent_active_pokemon.base_species
        or state.opponent_active_pokemon.name
        or ""
    ).lower()
    nxt = (
        next_state.opponent_active_pokemon.base_species
        or next_state.opponent_active_pokemon.name
        or ""
    ).lower()
    if nxt and cur != nxt:
        return _pretty_action(f"Switch {nxt}")
    mv = ""
    if next_state.opponent_prev_move is not None:
        mv = str(next_state.opponent_prev_move.name or "")
    if mv.lower() not in _BLANK_MOVES:
        return _pretty_action(mv)
    return ""


@dataclass
class ReplayMeta:
    id: str
    path: str
    rel_path: str
    format: str
    rating: Optional[int]
    player: str
    opponent: str
    date: str
    result: str  # WIN | LOSS | UNKNOWN
    filename: str


@dataclass
class TurnSummary:
    turn: int
    player_active: str
    player_active_base: str
    player_hp_pct: float
    player_status: str
    opponent_active: str
    opponent_active_base: str
    opponent_hp_pct: float
    opponent_status: str
    weather: str
    forced_switch: bool
    player_conditions: str
    opponent_conditions: str
    available_switches: list[dict]
    chosen_action: Optional[int]
    chosen_label: str
    missing: bool
    legal: dict[int, str]  # idx -> label (JSON keys will be str)
    player_will_use: str = ""
    opponent_will_use: str = ""
    player_party: list[dict] = field(default_factory=list)
    opponent_party: list[dict] = field(default_factory=list)


@dataclass
class ReplayDetail:
    meta: ReplayMeta
    num_turns: int
    turns: list[TurnSummary]
    player_party: list[dict]  # final revealed roster (convenience)
    opponent_party: list[dict]


def parse_filename_meta(filename: str, format_name: str) -> Optional[dict]:
    """Parse metadata from a metamon trajectory filename (no directory)."""
    name = filename[:-9] if filename.endswith(".json.lz4") else filename
    if name.endswith(".json"):
        name = name[:-5]
    parts = name.split("_")
    if len(parts) < 7 or "vs" not in parts[2:-2]:
        return None
    battle_id = parts[0]
    rating_str = parts[1]
    date_str = parts[-2]
    result = parts[-1]
    fmt_norm = format_name.replace("[", "").replace("]", "").replace(" ", "").lower()
    bid_norm = battle_id.replace("[", "").replace("]", "").replace(" ", "").lower()
    if fmt_norm not in bid_norm:
        return None
    mid = parts[2:-2]
    try:
        vs_i = mid.index("vs")
    except ValueError:
        return None
    player = "_".join(mid[:vs_i])
    opponent = "_".join(mid[vs_i + 1 :])
    try:
        rating = int(rating_str)
    except ValueError:
        rating = None
    if result not in ("WIN", "LOSS"):
        result = "UNKNOWN"
    return {
        "battle_id": battle_id,
        "rating": rating,
        "player": player,
        "opponent": opponent,
        "date": date_str,
        "result": result,
    }


def _discover_formats(root: str, formats: Optional[list[str]]) -> list[str]:
    if formats:
        return list(formats)
    found = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isdir(path) and not name.startswith("."):
            # skip non-format helper dirs
            if name in ("revealed_teams",):
                continue
            found.append(name)
        elif name.endswith(".tar"):
            found.append(name[: -len(".tar")])
    return found


def _walk_format_files(root: str, format_name: str) -> list[str]:
    format_dir = os.path.join(root, format_name)
    if not os.path.isdir(format_dir):
        return []
    out = []
    for dirpath, _, files in os.walk(format_dir):
        for f in files:
            if f.endswith((".json", ".json.lz4")):
                out.append(os.path.relpath(os.path.join(dirpath, f), root))
    return out


def _load_index_csv(root: str) -> Optional[list[str]]:
    index_path = os.path.join(root, "index.csv")
    if not os.path.isfile(index_path):
        return None
    rels = []
    with open(index_path, "r") as f:
        # first line may be header "filename"
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if i == 0 and line.lower() in ("filename", "path", "rel_path"):
                continue
            rels.append(line)
    return rels


class ReplayIndex:
    """In-memory index of metamon `.json.lz4` trajectories under a root."""

    def __init__(
        self,
        root: str,
        formats: Optional[list[str]] = None,
        use_index_csv: bool = True,
    ):
        self.root = os.path.abspath(root)
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Replay root not found: {self.root}")
        self.formats = _discover_formats(self.root, formats)
        self.replays: list[ReplayMeta] = []
        self._by_id: dict[str, ReplayMeta] = {}
        self._build(use_index_csv=use_index_csv)

    def _make_id(self, rel_path: str) -> str:
        # URL-safe id without path separators (FastAPI-friendly).
        return base64.urlsafe_b64encode(rel_path.encode("utf-8")).decode("ascii")

    def _build(self, use_index_csv: bool):
        rel_paths: list[tuple[str, str]] = []  # (format, rel)
        csv_rels = _load_index_csv(self.root) if use_index_csv else None
        if csv_rels is not None and self.formats:
            fmt_set = set(self.formats)
            for rel in csv_rels:
                fmt = rel.split(os.sep, 1)[0]
                if fmt in fmt_set:
                    rel_paths.append((fmt, rel))
        else:
            for fmt in self.formats:
                for rel in _walk_format_files(self.root, fmt):
                    rel_paths.append((fmt, rel))

        for fmt, rel in rel_paths:
            fname = os.path.basename(rel)
            parsed = parse_filename_meta(fname, fmt)
            if parsed is None:
                continue
            rid = self._make_id(rel)
            meta = ReplayMeta(
                id=rid,
                path=os.path.join(self.root, rel),
                rel_path=rel,
                format=fmt,
                rating=parsed["rating"],
                player=parsed["player"],
                opponent=parsed["opponent"],
                date=parsed["date"],
                result=parsed["result"],
                filename=fname,
            )
            self.replays.append(meta)
            self._by_id[rid] = meta

        # Newest-ish first by date string (MM-DD-YYYY… lexicographic is imperfect but ok)
        self.replays.sort(key=lambda m: (m.date, m.filename), reverse=True)

    def list_replays(
        self,
        offset: int = 0,
        limit: int = 100,
        result: Optional[str] = None,
        opponent: Optional[str] = None,
        format: Optional[str] = None,
        q: Optional[str] = None,
    ) -> dict[str, Any]:
        items = self.replays
        if result:
            r = result.upper()
            items = [m for m in items if m.result == r]
        if format:
            items = [m for m in items if m.format == format]
        if opponent:
            o = opponent.lower()
            items = [m for m in items if o in m.opponent.lower()]
        if q:
            qq = q.lower()
            items = [
                m
                for m in items
                if qq in m.filename.lower()
                or qq in m.player.lower()
                or qq in m.opponent.lower()
                or qq in m.format.lower()
            ]
        total = len(items)
        page = items[offset : offset + limit]
        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "items": [asdict(m) for m in page],
        }

    def get(self, replay_id: str) -> ReplayMeta:
        if replay_id not in self._by_id:
            raise KeyError(replay_id)
        return self._by_id[replay_id]

    @staticmethod
    def load_raw(path: str) -> tuple[list[UniversalState], list[int]]:
        if path.endswith(".json.lz4"):
            with lz4.frame.open(path, "rb") as f:
                data = json.loads(f.read().decode("utf-8"))
        else:
            with open(path, "r") as f:
                data = json.load(f)
        states = [UniversalState.from_dict(s) for s in data["states"]]
        actions = list(data["actions"])
        return states, actions

    def load_detail(self, replay_id: str) -> ReplayDetail:
        meta = self.get(replay_id)
        states, actions = self.load_raw(meta.path)
        # decision turns = len(states) - 1 (final action is -1 sentinel)
        n = max(len(states) - 1, 0)
        turns: list[TurnSummary] = []

        # Revealed roster order (stable) + KO tracking that updates each turn.
        player_roster: dict[str, dict] = {}
        opp_roster: dict[str, dict] = {}
        opp_fainted: set[str] = set()
        opp_last_hp: dict[str, float] = {}
        prev_opp_key: Optional[str] = None
        prev_opp_remaining: Optional[int] = None

        def _entry(poke) -> tuple[str, dict]:
            key = poke.base_species or poke.name
            return key, {
                "name": poke.name,
                "base_species": poke.base_species or poke.name,
                "sprite_id": sprite_id(poke.base_species or poke.name),
            }

        for t in range(n):
            s = states[t]
            a_idx = int(actions[t]) if t < len(actions) else -1
            missing = a_idx < 0
            pa = s.player_active_pokemon
            oa = s.opponent_active_pokemon

            switches = []
            player_alive: set[str] = set()
            pa_key, pa_info = _entry(pa)
            player_roster.setdefault(pa_key, pa_info)
            if float(pa.hp_pct) > 0:
                player_alive.add(pa_key)
            for p in s.available_switches:
                key, info = _entry(p)
                player_roster.setdefault(key, info)
                switches.append(
                    {
                        "name": p.name,
                        "base_species": p.base_species,
                        "hp_pct": float(p.hp_pct),
                        "status": p.status or "",
                        "sprite_id": sprite_id(p.base_species or p.name),
                    }
                )
                if float(p.hp_pct) > 0:
                    player_alive.add(key)

            # Player: anything revealed but not active/switchable is KO'd.
            player_party = [
                {
                    **info,
                    "fainted": key not in player_alive,
                    "active": key == pa_key,
                }
                for key, info in player_roster.items()
            ]

            # Opponent bench isn't in the traj — only active + opponents_remaining.
            # KOs often happen between decision turns, so the fainted mon never
            # appears at hp<=0. Mark the previous active when rem drops on a
            # species change (voluntary switches keep rem unchanged).
            oa_key, oa_info = _entry(oa)
            opp_roster.setdefault(oa_key, oa_info)
            rem = int(s.opponents_remaining)
            if prev_opp_key is not None and prev_opp_key != oa_key:
                rem_dropped = (
                    prev_opp_remaining is not None and rem < prev_opp_remaining
                )
                if rem_dropped or opp_last_hp.get(prev_opp_key, 1.0) <= 0:
                    opp_fainted.add(prev_opp_key)
            opp_last_hp[oa_key] = float(oa.hp_pct)
            if float(oa.hp_pct) <= 0 or (oa.status or "").lower() in ("fnt", "fainted"):
                opp_fainted.add(oa_key)
            # Active can't be both alive on field and fainted in the roster.
            if float(oa.hp_pct) > 0 and (oa.status or "").lower() not in (
                "fnt",
                "fainted",
            ):
                opp_fainted.discard(oa_key)
            prev_opp_key = oa_key
            prev_opp_remaining = rem

            opponent_party = [
                {
                    **info,
                    "fainted": key in opp_fainted,
                    "active": key == oa_key,
                }
                for key, info in opp_roster.items()
            ]

            legal = legal_action_labels(s)
            chosen = action_label(s, a_idx)
            next_state = states[t + 1] if t + 1 < len(states) else None
            player_will = "" if missing else _pretty_action(chosen)
            turns.append(
                TurnSummary(
                    turn=t,
                    player_active=pa.name,
                    player_active_base=pa.base_species or pa.name,
                    player_hp_pct=float(pa.hp_pct),
                    player_status=pa.status or "",
                    opponent_active=oa.name,
                    opponent_active_base=oa.base_species or oa.name,
                    opponent_hp_pct=float(oa.hp_pct),
                    opponent_status=oa.status or "",
                    weather=str(s.weather or ""),
                    forced_switch=bool(s.forced_switch),
                    player_conditions=str(s.player_conditions or ""),
                    opponent_conditions=str(s.opponent_conditions or ""),
                    available_switches=switches,
                    chosen_action=None if missing else a_idx,
                    chosen_label=chosen,
                    missing=missing,
                    legal={int(k): v for k, v in legal.items()},
                    player_will_use=player_will,
                    opponent_will_use=_opponent_will_use(s, next_state),
                    player_party=player_party,
                    opponent_party=opponent_party,
                )
            )

        last = turns[-1] if turns else None
        return ReplayDetail(
            meta=meta,
            num_turns=n,
            turns=turns,
            player_party=list(last.player_party) if last else [],
            opponent_party=list(last.opponent_party) if last else [],
        )

    def detail_to_dict(self, detail: ReplayDetail) -> dict:
        return {
            "meta": asdict(detail.meta),
            "num_turns": detail.num_turns,
            "turns": [asdict(t) for t in detail.turns],
            "player_party": detail.player_party,
            "opponent_party": detail.opponent_party,
        }
