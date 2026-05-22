import argparse
import csv
import os
import re
import sys
from glob import glob
from collections import defaultdict, Counter
from itertools import combinations
import textwrap
import math

try:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
except Exception as _e:
    np = None
    LogisticRegression = None


def canonical_team_name(team_file_path: str) -> str:
    """Derive a team name from the last 3 path parts.

    Example: /.../teams/competitive/gen1ou/team_1026.gen1ou_team
    -> competitive-gen1ou-team_1026
    """
    if team_file_path is None or team_file_path == "":
        return "UNKNOWN"
    parts = team_file_path.strip().split("/")
    if len(parts) >= 3:
        last_three = parts[-3:]
        folder_a, folder_b, filename = last_three
        # strip extension at first '.'
        base = filename.split(".")[0]
        return f"{folder_a}-{folder_b}-{base}"
    # fallback: use filename stem only
    filename = os.path.basename(team_file_path)
    base = filename.split(".")[0]
    return base


def last_three_parts(team_file_path: str):
    parts = (team_file_path or "").strip().split("/")
    if len(parts) >= 3:
        return parts[-3], parts[-2], parts[-1]
    return None, None, None


def resolve_team_file(cache_dir: str, team_file_path: str) -> str | None:
    """Resolve team file under METAMON_CACHE_DIR/teams/ using last three parts.

    Fallbacks: original path if exists, else None.
    """
    a, b, filename = last_three_parts(team_file_path)
    if cache_dir:
        candidate = os.path.join(
            cache_dir, "teams", *(p for p in [a, b] if p), filename or ""
        )
        if filename and os.path.isfile(candidate):
            return candidate
    if team_file_path and os.path.isfile(team_file_path):
        return team_file_path
    return None


_POKEMON_LINE_RE = re.compile(r"^\s*([^@\n]+?)(?:\s*@|\s*$)")


def parse_showdown_team_species(path: str) -> list[str]:
    """Parse a Showdown team export file and return list of species names.

    Extracts the leading species token on each mon block's first line,
    stripping gender and form parentheses, keeping hyphenated forms.
    """
    try:
        with open(path, "r") as f:
            text = f.read()
    except Exception:
        return []
    # Split on blank lines into mon blocks
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    species: list[str] = []
    for block in blocks:
        first_line = block.strip().splitlines()[0]
        m = _POKEMON_LINE_RE.match(first_line)
        if not m:
            continue
        name = m.group(1).strip()
        # strip gender/level/etc in parentheses, but keep hyphen forms
        if "(" in name:
            name = name.split("(", 1)[0].strip()
        if name:
            species.append(name)
    return species


def read_rows_from_csv(path: str):
    """Yield rows from a CSV, skipping header if present.

    Rows are lists of strings. Handles files that may be partially written
    or missing trailing columns. Skips empty/bad rows.
    """
    try:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            first = True
            for row in reader:
                if not row or all(cell.strip() == "" for cell in row):
                    continue
                if first:
                    first = False
                    # header detection
                    if row[0].strip().lower() in {"player username", "player_username"}:
                        continue
                yield row
    except Exception as e:
        print(f"Warning: failed to read {path}: {e}", file=sys.stderr)


def aggregate_win_rates(paths: list[str], cache_dir: str | None):
    by_team = defaultdict(lambda: {"wins": 0, "games": 0, "paths": set()})
    by_pokemon = defaultdict(lambda: {"wins": 0, "games": 0})
    files_seen = 0
    for p in paths:
        for row in read_rows_from_csv(p):
            # Expected columns (may be missing end columns):
            # 0: Player Username
            # 1: Team File (full path)
            # 2: Opponent Username
            # 3: Result (WIN/LOSS)
            # 4: Turn Count
            # 5: Battle ID (optional)
            if len(row) < 4:
                continue
            team_file = row[1].strip()
            result = row[3].strip().upper()
            team_name = canonical_team_name(team_file)
            agg = by_team[team_name]
            agg["games"] += 1
            if result == "WIN":
                agg["wins"] += 1
            # per-Pokemon aggregation
            cache_root = cache_dir or os.environ.get("METAMON_CACHE_DIR", "")
            resolved = resolve_team_file(cache_root, team_file)
            if resolved:
                try:
                    agg["paths"].add(resolved)
                except Exception:
                    pass
                mons = parse_showdown_team_species(resolved)
                for mon in mons:
                    pstats = by_pokemon[mon]
                    pstats["games"] += 1
                    if result == "WIN":
                        pstats["wins"] += 1
        files_seen += 1
    ranked = []
    for name, stats in by_team.items():
        games = stats["games"]
        wins = stats["wins"]
        losses = max(0, games - wins)
        win_rate = wins / games if games > 0 else 0.0
        ranked.append((name, games, wins, losses, win_rate))
    ranked.sort(key=lambda x: (x[4], x[1]), reverse=True)  # by win_rate then games
    ranked_pokemon = []
    for name, stats in by_pokemon.items():
        games = stats["games"]
        wins = stats["wins"]
        losses = max(0, games - wins)
        win_rate = wins / games if games > 0 else 0.0
        ranked_pokemon.append((name, games, wins, losses, win_rate))
    ranked_pokemon.sort(key=lambda x: (x[4], x[1]), reverse=True)
    # Map team name -> list of candidate file paths
    team_paths = {name: sorted(list(by_team[name]["paths"])) for name, *_ in ranked}
    return ranked, ranked_pokemon, files_seen, team_paths


def collect_samples(paths: list[str], cache_dir: str | None):
    """Collect per-battle samples with parsed species and outcome.

    Returns list of dicts: {"species": set[str], "y": int, "team_path": str, "team_name": str,
    "format": str}
    """
    samples = []
    cache_root = cache_dir or os.environ.get("METAMON_CACHE_DIR", "")
    for p in paths:
        for row in read_rows_from_csv(p):
            if len(row) < 4:
                continue
            team_file = row[1].strip()
            result = row[3].strip().upper()
            y = 1 if result == "WIN" else 0
            resolved = resolve_team_file(cache_root, team_file)
            if not resolved:
                # still keep the row for outcome-only analyses, but species unknown
                samples.append(
                    {
                        "species": set(),
                        "y": y,
                        "team_path": team_file,
                        "team_name": canonical_team_name(team_file),
                        "format": last_three_parts(team_file)[1] or "",
                    }
                )
                continue
            mons = set(parse_showdown_team_species(resolved))
            a, b, _ = last_three_parts(resolved)
            samples.append(
                {
                    "species": mons,
                    "y": y,
                    "team_path": resolved,
                    "team_name": canonical_team_name(resolved),
                    "format": b or "",
                }
            )
    return samples


def _build_design_matrix(
    samples,
    min_games_species: int,
    add_top_pairs: int,
    include_format: bool,
):
    """Build X, y, feature_names from samples.

    - Filters species by min_games occurrence
    - Adds top-K frequent pair features (presence of both)
    - Optionally includes format fixed effects
    """
    species_counter = Counter()
    pair_counter = Counter()
    for s in samples:
        species = list(s["species"]) if s["species"] else []
        species_counter.update(species)
        if add_top_pairs > 0 and len(species) >= 2:
            pair_counter.update(tuple(sorted(p)) for p in combinations(species, 2))

    keep_species = sorted(
        [sp for sp, c in species_counter.items() if c >= min_games_species]
    )
    species_to_idx = {sp: i for i, sp in enumerate(keep_species)}

    keep_pairs = []
    if add_top_pairs > 0:
        keep_pairs = [pair for pair, _ in pair_counter.most_common(add_top_pairs)]
    pair_to_idx = {pair: i for i, pair in enumerate(keep_pairs)}

    formats = []
    if include_format:
        formats = sorted({s["format"] for s in samples if s["format"]})
    fmt_to_idx = {f: i for i, f in enumerate(formats)}

    n = len(samples)
    d_species = len(keep_species)
    d_pairs = len(keep_pairs)
    d_format = len(formats)
    d_total = d_species + d_pairs + d_format
    X = np.zeros((n, d_total), dtype=np.float32)
    y = np.zeros((n,), dtype=np.int64)

    for i, s in enumerate(samples):
        y[i] = int(s["y"]) if s["y"] in (0, 1) else 0
        sp_set = s["species"]
        for sp in sp_set:
            j = species_to_idx.get(sp)
            if j is not None:
                X[i, j] = 1.0
        if d_pairs:
            for pair, jrel in pair_to_idx.items():
                a, b = pair
                if a in sp_set and b in sp_set:
                    X[i, d_species + jrel] = 1.0
        if d_format:
            f = s["format"]
            jfmt = fmt_to_idx.get(f)
            if jfmt is not None:
                X[i, d_species + d_pairs + jfmt] = 1.0

    feature_names = []
    feature_names.extend([f"mon:{sp}" for sp in keep_species])
    feature_names.extend([f"pair:{a}&{b}" for (a, b) in keep_pairs])
    feature_names.extend([f"format:{f}" for f in formats])
    return X, y, feature_names


def fit_presence_logistic(
    samples,
    min_games_species: int,
    add_top_pairs: int,
    include_format: bool,
    penalty: str,
    C: float,
    max_iter: int,
):
    if np is None or LogisticRegression is None:
        print("sklearn/numpy not available; skipping logistic regression.")
        return None
    X, y, feature_names = _build_design_matrix(
        samples,
        min_games_species=min_games_species,
        add_top_pairs=add_top_pairs,
        include_format=include_format,
    )
    if X.shape[1] == 0:
        print("No features after filtering; skipping logistic regression.")
        return None
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    if penalty == "l1":
        # liblinear handles l1 but not multinomial; our y is binary
        model = LogisticRegression(
            penalty="l1", C=C, solver=solver, max_iter=max_iter, fit_intercept=True
        )
    else:
        model = LogisticRegression(
            penalty="l2", C=C, solver=solver, max_iter=max_iter, fit_intercept=True
        )
    model.fit(X, y)
    coefs = model.coef_.reshape(-1)
    intercept = float(model.intercept_.reshape(()))
    odds = np.exp(coefs)
    results = list(zip(feature_names, coefs.tolist(), odds.tolist()))
    # sort by odds ratio descending
    results.sort(key=lambda t: t[2], reverse=True)
    return {
        "feature_importances": results,
        "intercept": intercept,
        "num_samples": int(X.shape[0]),
        "num_features": int(X.shape[1]),
    }


def print_logit_results(res, top_k: int, bottom_k: int):
    if not res:
        return
    feats = res["feature_importances"]
    print("\n" + "=" * 80)
    print("Presence logistic regression (win ~ mons [+ pairs] [+ format])")
    print(
        f"Samples: {res['num_samples']}, Features: {res['num_features']}, Intercept OR: {math.exp(res['intercept']):.3f}"
    )
    print("=" * 80)
    if top_k > 0:
        print("Top features by odds ratio:\n")
        print(f"{'Feature':40}  {'Coef':>9}  {'OddsRatio':>10}")
        print("-" * 64)
        for name, coef, orat in feats[:top_k]:
            print(f"{name:40}  {coef:9.3f}  {orat:10.3f}")


def print_best_worst_side_by_side(best_list, worst_list):
    """Print best and worst team summaries side-by-side for quick comparison.

    Each side shows: Team, Games, Wins, Losses, WinRate.
    """
    # Determine widths
    name_w = 44
    left_header = f"{'Best Team':{name_w}}  {'G':>4}  {'W':>4}  {'L':>4}  {'WR':>6}"
    right_header = f"{'Worst Team':{name_w}}  {'G':>4}  {'W':>4}  {'L':>4}  {'WR':>6}"
    sep = "-" * (len(left_header) + 4 + len(right_header))
    print("Best vs Worst (side-by-side):\n")
    print(left_header + "    " + right_header)
    print(sep)
    rows = max(len(best_list), len(worst_list))
    for i in range(rows):
        if i < len(best_list):
            b_name, b_g, b_w, b_l, b_wr = best_list[i]
            left = f"{b_name:{name_w}}  {b_g:4d}  {b_w:4d}  {b_l:4d}  {b_wr:6.3f}"
        else:
            left = f"{'':{name_w}}  {'':>4}  {'':>4}  {'':>4}  {'':>6}"
        if i < len(worst_list):
            w_name, w_g, w_w, w_l, w_wr = worst_list[i]
            right = f"{w_name:{name_w}}  {w_g:4d}  {w_w:4d}  {w_l:4d}  {w_wr:6.3f}"
        else:
            right = f"{'':{name_w}}  {'':>4}  {'':>4}  {'':>4}  {'':>6}"
        print(left + "    " + right)
    print()


def expand_inputs(inputs: list[str]) -> list[str]:
    paths: list[str] = []
    for p in inputs:
        if os.path.isdir(p):
            # add all csv files under this directory (non-recursive)
            paths.extend(sorted(glob(os.path.join(p, "*.csv"))))
        else:
            # allow glob patterns
            expanded = glob(p)
            if expanded:
                paths.extend(sorted(expanded))
            else:
                paths.append(p)
    # drop duplicates while preserving order
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def _read_team_text(team_path: str) -> str:
    try:
        with open(team_path, "r") as tf:
            return tf.read().rstrip("\n")
    except Exception as e:
        return f"<error reading file: {e}>"


def print_team_contents_columns(
    team_rows,
    team_paths: dict[str, list[str]],
    title: str,
    columns: int = 2,
    col_width: int = 60,
):
    """Print full team files in a multi-column layout.

    team_rows: list of (name, games, wins, losses, win_rate)
    team_paths: mapping name -> [paths]
    """
    space_between = 4
    total_width = columns * col_width + (columns - 1) * space_between
    print(title)
    print("-" * total_width)
    # Build blocks (list of list[str]) for each team
    blocks = []
    for name, games, wins, losses, wr in team_rows:
        paths_for_team = team_paths.get(name, [])
        header = f"{name} | G={games} W={wins} L={losses} WR={wr:.3f}"
        if paths_for_team:
            team_path = paths_for_team[0]
            header2 = f"File: {team_path}"
            content = _read_team_text(team_path)
        else:
            header2 = "File: <not found>"
            content = "<no content>"
        # wrap lines to column width
        lines = []
        for h in (header, header2):
            lines.extend(textwrap.wrap(h, width=col_width) or [h])
        for line in content.splitlines():
            wrapped = textwrap.wrap(line, width=col_width)
            lines.extend(wrapped if wrapped else [""])
        blocks.append(lines)
    # Print in rows of `columns`
    for i in range(0, len(blocks), columns):
        row_blocks = blocks[i : i + columns]
        max_h = max(len(b) for b in row_blocks)
        for h in range(max_h):
            parts = []
            for b in row_blocks:
                cell = b[h] if h < len(b) else ""
                parts.append(f"{cell:<{col_width}}")
            print((" " * space_between).join(parts))
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze win rates by team from team log CSVs."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="CSV file(s), directories, or glob patterns (e.g., team_logs/*.csv)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=50,
        help="Show top N teams by win rate (deprecated in favor of --best-teams)",
    )
    parser.add_argument(
        "--best-teams", type=int, default=None, help="Show best K teams by win rate"
    )
    parser.add_argument(
        "--worst-teams", type=int, default=0, help="Also show worst K teams by win rate"
    )
    parser.add_argument(
        "--top-pokemon", type=int, default=50, help="Show top N Pokemon by win rate"
    )
    parser.add_argument(
        "--min-games-pokemon",
        type=int,
        default=10,
        help="Minimum games to include a Pokemon in ranking",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.environ.get("METAMON_CACHE_DIR", ""),
        help="Path to METAMON_CACHE_DIR (for resolving team files)",
    )
    parser.add_argument(
        "--no-team-content",
        action="store_true",
        help="Do not print team file contents in best/worst team sections",
    )
    # modeling options
    parser.add_argument(
        "--logit", action="store_true", help="Fit a logistic regression presence model"
    )
    parser.add_argument(
        "--logit-penalty",
        type=str,
        choices=["l1", "l2"],
        default="l2",
        help="Regularization penalty for logistic regression",
    )
    parser.add_argument(
        "--logit-C",
        type=float,
        default=1.0,
        help="Inverse regularization strength for logistic regression",
    )
    parser.add_argument(
        "--logit-iter",
        type=int,
        default=200,
        help="Max iterations for logistic regression",
    )
    parser.add_argument(
        "--logit-min-games",
        type=int,
        default=25,
        help="Minimum battles required to include a Pokemon as a feature",
    )
    parser.add_argument(
        "--logit-top-pairs",
        type=int,
        default=0,
        help="Add top-K frequent pair presence features (0 disables)",
    )
    parser.add_argument(
        "--logit-include-format",
        action="store_true",
        help="Include format fixed effects in the model",
    )
    parser.add_argument(
        "--logit-top-features",
        type=int,
        default=25,
        help="Show top-K features by odds ratio",
    )
    parser.add_argument(
        "--logit-bottom-features",
        type=int,
        default=25,
        help="Show bottom-K features by odds ratio",
    )
    args = parser.parse_args()

    paths = expand_inputs(args.inputs)
    if not paths:
        print("No input files found.")
        return 1

    ranked, ranked_pokemon, files_seen, team_paths = aggregate_win_rates(
        paths, cache_dir=args.cache_dir
    )
    print("=" * 80)
    print(f"Analyzed {files_seen} file(s), {len(ranked)} team(s) found.")
    print("=" * 80 + "\n")
    best_k = args.best_teams if args.best_teams is not None else args.top
    if best_k and best_k > 0:
        # Prepare best and worst lists for side-by-side view (names-only summary)
        best_rows = ranked[:best_k]
        worst_rows = []
        if args.worst_teams and args.worst_teams > 0 and ranked:
            worst_rows = list(reversed(ranked[-args.worst_teams :]))
        if best_rows or worst_rows:
            print_best_worst_side_by_side(best_rows, worst_rows)
        if not args.no_team_content:
            print_team_contents_columns(
                best_rows, team_paths, title="Best teams (full contents, columns):\n"
            )
    if args.worst_teams and args.worst_teams > 0 and ranked:
        if not args.no_team_content:
            worst_slice = list(reversed(ranked[-args.worst_teams :]))
            print_team_contents_columns(
                worst_slice, team_paths, title="Worst teams (full contents, columns):\n"
            )
    # Pokemon ranking
    if ranked_pokemon:
        print("\nPokemon rankings (filtered):\n")
        print(
            f"{'Pokemon':28}  {'Games':>5}  {'Wins':>5}  {'Losses':>6}  {'WinRate':>7}"
        )
        print("-" * 60)
        shown = 0
        for name, games, wins, losses, wr in ranked_pokemon:
            if games < args.min_games_pokemon:
                continue
            print(f"{name:28}  {games:5d}  {wins:5d}  {losses:6d}  {wr:7.3f}")
            shown += 1
            if shown >= args.top_pokemon:
                break
    # logistic regression analysis
    if args.logit:
        samples = collect_samples(paths, cache_dir=args.cache_dir)
        res = fit_presence_logistic(
            samples,
            min_games_species=args.logit_min_games,
            add_top_pairs=args.logit_top_pairs,
            include_format=args.logit_include_format,
            penalty=args.logit_penalty,
            C=args.logit_C,
            max_iter=args.logit_iter,
        )
        print_logit_results(
            res, top_k=args.logit_top_features, bottom_k=args.logit_bottom_features
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
