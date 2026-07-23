"""
python -m metamon.rl.analyze REPLAY_DIR
python -m metamon.rl.analyze --parsed-replays gen2ou
python -m metamon.rl.analyze REPLAY_DIR --debug-models
"""

from __future__ import annotations

import argparse
import os
import sys

# TEMPORARY debug preset — remove once the analyze UI workflow is settled.
DEBUG_MODEL_ASSORTMENT: list[tuple[str, int]] = [
    ("TaurosV2", 800),
    ("SyntheticRLV2", 48),
    ("TaurosV2", 409),
    ("SmallG1OnlineV3", 250),
    ("Kakuna", 34),
]


def _require_web_deps():
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError:
        print(
            "metamon.rl.analyze requires FastAPI and uvicorn.\n"
            "  pip install 'metamon[analyze]'\n"
            "  # or: pip install fastapi uvicorn",
            file=sys.stderr,
        )
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Browse metamon replays and compare pretrained action distributions."
    )
    parser.add_argument(
        "replay_dir",
        nargs="?",
        default=None,
        help="Root directory with {format}/*.json.lz4 (or nested YYYY/MM).",
    )
    parser.add_argument(
        "--parsed-replays",
        nargs="*",
        metavar="FORMAT",
        default=None,
        help=(
            "Use $METAMON_CACHE_DIR/parsed-replays. Optional format filters "
            "(e.g. gen2ou). With no formats, index all format subdirs."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device for loaded analysis models (default: cuda:0).",
    )
    parser.add_argument(
        "--debug-models",
        action="store_true",
        help=(
            "TEMPORARY: preload a fixed debug assortment "
            "(TaurosV2@800, SyntheticRLV2@48, TaurosV2@409, "
            "SmallG1OnlineV3@250, Kakuna@34)."
        ),
    )
    args = parser.parse_args(argv)

    _require_web_deps()

    formats = None
    if args.parsed_replays is not None:
        from metamon.config import METAMON_CACHE_DIR

        if not METAMON_CACHE_DIR:
            print("METAMON_CACHE_DIR is not set.", file=sys.stderr)
            sys.exit(1)
        replay_root = os.path.join(METAMON_CACHE_DIR, "parsed-replays")
        formats = list(args.parsed_replays) if args.parsed_replays else None
    elif args.replay_dir:
        replay_root = os.path.abspath(args.replay_dir)
    else:
        parser.error("Provide REPLAY_DIR or --parsed-replays")

    if not os.path.isdir(replay_root):
        print(f"Replay root not found: {replay_root}", file=sys.stderr)
        sys.exit(1)

    import uvicorn

    from metamon.rl.analyze.server import create_app

    preload = DEBUG_MODEL_ASSORTMENT if args.debug_models else None
    app = create_app(
        replay_root=replay_root,
        formats=formats,
        device=args.device,
        preload_models=preload,
    )
    print(f"Metamon analyze")
    print(f"  Replays: {replay_root}")
    if formats:
        print(f"  Formats: {', '.join(formats)}")
    print(f"  Device:  {args.device}")
    if preload:
        print(
            "  Debug models: "
            + ", ".join(f"{n}@{c}" for n, c in preload)
            + "  (TEMPORARY --debug-models)"
        )
    print(f"  Open:    http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
