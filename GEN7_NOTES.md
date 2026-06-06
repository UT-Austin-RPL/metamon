# Gen 7 Port — Notes

Parser pipeline is working. 196/200 bootstrap replays pass forward + backward fill.
The 4 failures are data quality (incomplete downloads, early forfeit) — not fixable in code.

---

## What was added

**Parser**
- `_parse_gen`: gen 7 unlocked; initializes `can_z_1/2` and `can_mega_1/2` at battle start
- `_parse_zpower`: sets `is_zmove=True` on the action, consumes `can_z`
- `_parse_mega`: sets `is_mega=True` on the action, consumes `can_mega`; forme change handled by the `detailschange` that precedes it
- `_parse_burst`: Ultra Burst (Necrozma) shares the `can_mega` slot
- Z-move names stripped from `had_moves` after use — they're one-turn transforms of the base move, not permanent move slots
- `check_gimmick_consistency`: mirrors `check_tera_consistency` for Z/mega flags; split by type (mega valid in gens 6–7, Z only gen 7)

**Interface**
- `UniversalState`: `can_z`, `can_mega` fields; backwards-compat in `from_dict`
- `from_Battle`: wired to `battle.can_z_move` / `battle.can_mega_evolve`
- `action_idx_to_BattleOrder`: +9 actions map to mega/Z/tera based on what's available; Z-move only applied if the selected move matches the held crystal
- `Gen7ObservationSpace`: `DefaultObservationSpace` + `can_z`/`can_mega` appended to numbers (50 dims)

**Tokenizer**
- `DefaultObservationSpace-v1-gen7.json`: built from 200 gen7ou replays; 2639 tokens
- Adds gen7 mons (Tapus, Kartana, Celesteela, Ash-Greninja, etc.), Z-crystals, Z-move names, mega formes

**Other**
- `SUPPORTED_BATTLE_FORMATS`: gen7ou added
- `BASELINES_BY_GEN`: gen 7 added (fixes `KeyError: 7` in heuristic baselines)
- `PreloadedSmogonUsageStats`: falls back to full history when a date-windowed load returns empty (handles retired formats queried with recent replay dates)
- `FORMAT_LATEST_DATES` removed in favour of the fallback approach above

---

## Known limitations / TODOs before PR

**Replay diff not done**
Jake's bar for merge is a turn-by-turn comparison of the parser's reconstructed state against the Showdown replay viewer. Needs someone with gen7 game knowledge to run it.

**poke-env fork**
Online / self-play path for gen7 needs the poke-env fork to handle `-zpower` / `-mega` protocol messages. `can_z` and `can_mega` in `from_Battle` are wired, but the fork itself may need updates analogous to what PR #26 did for gen9.

**Z-move PP**
When a Z-move is used, the base move's PP is not decremented (we don't track which base move a Z-move came from without a crystal→type lookup). PP tracking is already approximate in the codebase; this is a known gap.

**Mega forme name**
PR #26 hit a bug (commit e481fde) where offline data used the original species name while poke-env updated to the mega forme. Not yet verified by replay diff whether our parser handles this correctly in all cases.

**Team sets**
Two hand-written gen7ou teams in `.cache/teams/competitive/gen7ou/`. Jake's HF dataset will ship proper competitive team sets; delete and replace these when it lands.
