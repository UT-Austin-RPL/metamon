Adds gen7ou: replay parser, observation space, tokenizer, and action mapping for mega/Z-moves (sharing the +9 slots like tera).

The main issue is Z-moves: the log shows the Z-move name, not the base move, so the base move is resolved by crystal type in forward fill when already revealed, otherwise enforced in team prediction and a final fallback in from_ReplayAction.

- All gen7 replays sampled by me (100 from high ELO and 100 recent replays) passed.
- No regressions on gen1–4/9
- Tokenizer is append-only on v1 (existing ids unchanged)
- gen7uu and gen7nu also enabled, but not all gen7 formats are solid yet: gen7ubers is left out for now (a few parser bugs around Marshadow's Z-move name, Ultra Burst + Z-move, and a team-prediction crash).

TODO: update README, and online play is wired but still needs to be verified.
