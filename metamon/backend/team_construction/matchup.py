import argparse
import csv
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List


def _run_serve_matchup(
    *,
    model_name: str,
    username: str,
    opponent_username: str,
    role: str,
    battle_format: str,
    n_battles: int,
    team_set: str,
    gpu_id: int,
    save_results_to: Path,
    checkpoint: int | None = None,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "metamon.rl.evaluate.serve_matchup",
        "--model_name",
        model_name,
        "--username",
        username,
        "--opponent_username",
        opponent_username,
        "--role",
        role,
        "--format",
        battle_format,
        "--n_battles",
        str(n_battles),
        "--team_set",
        team_set,
        "--battle_backend",
        "metamon",
        "--temperature",
        "1.0",
        "--save_results_to",
        str(save_results_to),
    ]
    if checkpoint is not None:
        cmd += ["--checkpoint", str(checkpoint)]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return subprocess.Popen(cmd, env=env)


def _norm_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def _row_get(row: dict, key: str) -> str:
    if key in row and row[key] is not None:
        return str(row[key])
    wanted = _norm_key(key)
    for k, v in row.items():
        if k is None:
            continue
        if _norm_key(str(k)) == wanted and v is not None:
            return str(v)
    return ""


def _canonical_result(raw: str) -> str:
    value = raw.strip().upper()
    if value in {"WIN", "WON", "1"}:
        return "WIN"
    if value in {"DRAW", "TIE", "0.5"}:
        return "DRAW"
    if value in {"LOSS", "LOSE", "LOST", "0"}:
        return "LOSS"
    return "UNKNOWN"


def _as_int(value: str, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _read_player_rows(results_dir: Path, player_username: str) -> List[dict]:
    rows: List[dict] = []
    for csv_file in sorted(results_dir.rglob("*.csv")):
        with csv_file.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                player = _row_get(row, "Player Username").strip()
                if player == player_username:
                    rows.append(row)
    return rows


def _summarize(rows: List[dict]) -> Dict[str, float]:
    wins = 0
    draws = 0
    losses = 0
    unknown = 0
    for row in rows:
        result = _canonical_result(_row_get(row, "Result"))
        if result == "WIN":
            wins += 1
        elif result == "DRAW":
            draws += 1
        elif result == "LOSS":
            losses += 1
        else:
            unknown += 1
    scored = wins + draws + losses
    win_rate = (wins + 0.5 * draws) / scored if scored else 0.0
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "unknown": unknown,
        "win_rate": win_rate,
        "logged_battles": scored,
    }


def _match_records(rows: List[dict]) -> List[dict]:
    out: List[dict] = []
    for i, row in enumerate(rows, start=1):
        out.append(
            {
                "match_idx": i,
                "battle_id": _row_get(row, "Battle ID").strip(),
                "result": _canonical_result(_row_get(row, "Result")),
                "turn_count": _as_int(_row_get(row, "Turn Count"), default=0),
                "team_file_path": _row_get(row, "Team File").strip(),
                "team_file": Path(_row_get(row, "Team File").strip()).name,
            }
        )
    return out


def run_matchup(
    *,
    battle_format: str,
    num_battles: int,
    model_name: str,
    team_set_a: str,
    team_set_b: str,
    gpu_a: int,
    gpu_b: int,
    work_dir: Path,
    checkpoint: int | None = None,
    print_match_stats: bool = False,
) -> Dict[str, object]:
    run_id = uuid.uuid4().hex[:10]
    acceptor_username = f"tca{run_id}"
    challenger_username = f"tcb{run_id}"
    acceptor_results = work_dir / f"results_acceptor_{run_id}"
    challenger_results = work_dir / f"results_challenger_{run_id}"
    acceptor_results.mkdir(parents=True, exist_ok=True)
    challenger_results.mkdir(parents=True, exist_ok=True)

    print(
        f"[matchup] format={battle_format} n={num_battles} model={model_name} "
        f"team_set_a={team_set_a} team_set_b={team_set_b}"
    )
    print(
        f"[matchup] acceptor={acceptor_username} gpu={gpu_a} "
        f"challenger={challenger_username} gpu={gpu_b}"
    )
    start = time.time()
    acceptor_proc = _run_serve_matchup(
        model_name=model_name,
        username=acceptor_username,
        opponent_username=challenger_username,
        role="acceptor",
        battle_format=battle_format,
        n_battles=num_battles,
        team_set=team_set_a,
        gpu_id=gpu_a,
        save_results_to=acceptor_results,
        checkpoint=checkpoint,
    )
    time.sleep(2.0)
    challenger_proc = _run_serve_matchup(
        model_name=model_name,
        username=challenger_username,
        opponent_username=acceptor_username,
        role="challenger",
        battle_format=battle_format,
        n_battles=num_battles,
        team_set=team_set_b,
        gpu_id=gpu_b,
        save_results_to=challenger_results,
        checkpoint=checkpoint,
    )

    rc_challenger = challenger_proc.wait()
    rc_acceptor = acceptor_proc.wait()
    elapsed = time.time() - start
    if rc_challenger != 0 or rc_acceptor != 0:
        raise RuntimeError(
            f"Battle workers failed (acceptor_rc={rc_acceptor}, challenger_rc={rc_challenger})."
        )

    acceptor_rows = _read_player_rows(acceptor_results, acceptor_username)
    challenger_rows = _read_player_rows(challenger_results, challenger_username)
    if not acceptor_rows:
        raise RuntimeError(f"No result rows found for {acceptor_username}")
    if not challenger_rows:
        raise RuntimeError(f"No result rows found for {challenger_username}")

    acceptor_summary = _summarize(acceptor_rows)
    challenger_summary = _summarize(challenger_rows)
    acceptor_matches = _match_records(acceptor_rows)
    challenger_matches = _match_records(challenger_rows)

    print("[matchup] done")
    print(
        f"[acceptor] W/D/L/U = {acceptor_summary['wins']}/{acceptor_summary['draws']}/"
        f"{acceptor_summary['losses']}/{acceptor_summary['unknown']} "
        f"win_rate={acceptor_summary['win_rate']:.3f}"
    )
    print(
        f"[challenger] W/D/L/U = {challenger_summary['wins']}/{challenger_summary['draws']}/"
        f"{challenger_summary['losses']}/{challenger_summary['unknown']} "
        f"win_rate={challenger_summary['win_rate']:.3f}"
    )
    print(f"[matchup] elapsed={elapsed:.1f}s")

    if print_match_stats:
        running = 0.0
        for m in acceptor_matches:
            if m["result"] == "WIN":
                running += 1.0
            elif m["result"] == "DRAW":
                running += 0.5
            print(
                f"[acceptor match {m['match_idx']:02d}] id={m['battle_id']} "
                f"result={m['result']} turns={m['turn_count']} "
                f"team={m['team_file']} running_wr={running / m['match_idx']:.3f}"
            )

    return {
        "run_id": run_id,
        "elapsed_sec": elapsed,
        "acceptor_results_dir": str(acceptor_results),
        "challenger_results_dir": str(challenger_results),
        "acceptor_summary": acceptor_summary,
        "challenger_summary": challenger_summary,
        "acceptor_matches": acceptor_matches,
        "challenger_matches": challenger_matches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one standalone Kakuna-vs-Kakuna matchup batch."
    )
    parser.add_argument("--battle-format", default="gen1ou")
    parser.add_argument("--num-battles", type=int, default=16)
    parser.add_argument("--model-name", default="Kakuna")
    parser.add_argument("--team-set-a", default="competitive")
    parser.add_argument("--team-set-b", default="competitive")
    parser.add_argument("--gpu-a", type=int, default=0)
    parser.add_argument("--gpu-b", type=int, default=1)
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/team_prediction/team_construction_battles"),
    )
    parser.add_argument("--print-match-stats", action="store_true")
    args = parser.parse_args()

    run_matchup(
        battle_format=args.battle_format,
        num_battles=args.num_battles,
        model_name=args.model_name,
        team_set_a=args.team_set_a,
        team_set_b=args.team_set_b,
        gpu_a=args.gpu_a,
        gpu_b=args.gpu_b,
        work_dir=args.work_dir,
        checkpoint=args.checkpoint,
        print_match_stats=args.print_match_stats,
    )


if __name__ == "__main__":
    main()
