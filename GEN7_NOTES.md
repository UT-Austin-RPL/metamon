# Gen 7 Port — Notes

Full pipeline pass rate: **196/200** bootstrap replays (top-100 ELO: 100/100, recent-100: 96/100).

Remaining failures are data quality: 3 incomplete replay downloads, 1 unusual team size. One replay additionally drops a single POV side (Kartana used a Normal-type Z-move but its usage stats contain no Normal-type damaging move to substitute).

Regression check on frozen 25-replay corpora (re-run 2026-06-10): gen1ou 25/25, gen2ou 25/25, gen3ou 24/25 (1 incomplete download), gen4ou 25/25, gen9ou 25/25. Zero regressions.

---

## What was added

**Parser**

- `_parse_gen`: gen 7 unlocked; initializes `can_z_1/2` and `can_mega_1/2` at battle start
- `_parse_zpower`: sets `is_zmove=True` on the action, consumes `can_z`
- `_parse_mega`: sets `is_mega=True` on the action, consumes `can_mega`; forme change handled by the `detailschange` that precedes it
- `_parse_burst`: Ultra Burst (Necrozma) shares the `can_mega` slot
- Damaging Z-move names stripped from `had_moves` after use — they're one-turn transforms of the base move, not permanent move slots. Status Z-moves keep the base move's name in the log and are left alone.
- `check_gimmick_consistency`: mirrors `check_tera_consistency` for Z/mega flags; split by type (mega valid in gens 6–7, Z only gen 7)

**Z-move action resolution**

The log records the Z-move name (e.g. "Corkscrew Crash"), not the base move, but action indexing needs the base move's slot. Resolution happens in three layers:

1. `forward.py` resolves the base move by type when it's already in `had_moves`
2. `backward.py` (`_enforce_zmove_consistency`): a damaging Z-move proves the user carries a base move of the crystal's type and holds the crystal. If team prediction filled the moveset without one, the least common predicted move is swapped for the most common move of the required type from usage stats; if the item was never revealed, it's replaced with the crystal.
3. `from_ReplayAction` has a final type-based fallback against the post-backward moveset

A gimmick revealed without a move (e.g. mega evolved then flinched) maps to action `-1`, same as the existing tera handling.

**Interface**

- `UniversalState`: `can_z`, `can_mega` fields; backwards-compat in `from_dict`
- `from_Battle`: wired to `battle.can_z_move` / `battle.can_mega_evolve`
- `action_idx_to_BattleOrder`: +9 actions map to mega/Z/tera based on what's available; Z-move only applied if the selected move matches the held crystal
- `Gen7ObservationSpace`: `DefaultObservationSpace` + `can_z`/`can_mega` appended to numbers (50 dims)

**Tokenizer**

- `DefaultObservationSpace-v1-gen7.json`: built from 200 gen7ou replays; 2902 tokens. Append-only on top of `DefaultObservationSpace-v1` (existing token ids unchanged).
- Adds gen7 mons (Tapus, Kartana, Celesteela, Ash-Greninja, etc.), Z-crystals, Z-move names, mega formes

**Other**

- `SUPPORTED_BATTLE_FORMATS`: gen7ou added
- `BASELINES_BY_GEN`: gen 7 added (fixes `KeyError: 7` in heuristic baselines)
- `PreloadedSmogonUsageStats`: falls back to full history when a date-windowed load returns empty (handles retired formats queried with recent replay dates)
- `FORMAT_LATEST_DATES` removed in favour of the fallback approach above

---

## Bugs fixed along the way

- `check_action_idxs`: the tera counter incremented for any `action_idx >= 9`, so every gen7 Z-move/mega action failed with `ActionIndexError`. Now only `is_tera` actions count.
- Usage stats lookups during Z-move fixups go through the forme name (`pokemon.name`), not the base species — `Ninetales-Alola` was getting Kanto Ninetales movesets.
- Status Z-moves (Z-Conversion etc.) no longer have their base move wrongly stripped from the moveset.

---

## Known limitations / TODOs

**Online play**
The online/self-play path for gen7 needs the poke-env layer to handle `-zpower` / `-mega` protocol messages. `can_z` and `can_mega` in `from_Battle` are wired, but this path hasn't been exercised yet — likely needs work analogous to what PR #26 did for gen9.

**Z-move PP**
When a Z-move is used, the base move's PP is not decremented. PP tracking is already approximate in the codebase; this is a known gap.

**Mega forme name**
PR #26 hit a bug (commit e481fde) where offline data used the original species name while poke-env updated to the mega forme. Not yet verified by replay diff whether our parser handles this correctly in all cases.

**Team sets**
The gen7ou team files currently used locally are hand-written placeholders, not a curated competitive set.
