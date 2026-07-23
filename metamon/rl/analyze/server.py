"""FastAPI app for the metamon replay analyze browser."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from metamon.rl.analyze.replay_index import ReplayIndex
from metamon.rl.analyze.scoring import ModelSession, score_replay
from metamon.rl.analyze.sprites import get_cropped_gen1_sprite

STATIC_DIR = Path(__file__).resolve().parent / "static"


class LoadModelBody(BaseModel):
    name: str
    checkpoint: Optional[int] = None
    temperature: float = 1.0


def create_app(
    replay_root: str,
    formats: Optional[list[str]] = None,
    device: str = "cuda:0",
    preload_models: Optional[list[tuple[str, Optional[int]]]] = None,
) -> FastAPI:
    app = FastAPI(title="Metamon Analyze", version="0.1.0")
    index = ReplayIndex(root=replay_root, formats=formats)
    session = ModelSession(device=device)
    # Cache last opened raw traj for rescoring when models change
    cache: dict[str, Any] = {"replay_id": None, "states": None, "actions": None}

    if preload_models:
        print(f"Preloading {len(preload_models)} debug model(s)…")
        for name, checkpoint in preload_models:
            label = f"{name}@{checkpoint}" if checkpoint is not None else name
            print(f"  loading {label} …", flush=True)
            try:
                session.load(name, checkpoint=checkpoint)
                print(f"  loaded  {label}", flush=True)
            except Exception as e:
                print(f"  FAILED  {label}: {e}", flush=True)

    app.state.replay_root = replay_root
    app.state.index = index
    app.state.session = session
    app.state.cache = cache

    @app.get("/api/health")
    def health():
        return {
            "ok": True,
            "replay_root": replay_root,
            "num_replays": len(index.replays),
            "formats": index.formats,
            "device": device,
            "loaded_models": [m["key"] for m in session.loaded()],
        }

    @app.get("/api/replays")
    def list_replays(
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=500),
        result: Optional[str] = None,
        opponent: Optional[str] = None,
        format: Optional[str] = None,
        q: Optional[str] = None,
    ):
        return index.list_replays(
            offset=offset,
            limit=limit,
            result=result,
            opponent=opponent,
            format=format,
            q=q,
        )

    @app.get("/api/replays/{replay_id}")
    def get_replay(replay_id: str):
        try:
            detail = index.load_detail(replay_id)
        except KeyError:
            raise HTTPException(404, f"Replay not found: {replay_id}")
        except Exception as e:
            raise HTTPException(500, f"Failed to load replay: {e}") from e
        states, actions = ReplayIndex.load_raw(detail.meta.path)
        cache["replay_id"] = replay_id
        cache["states"] = states
        cache["actions"] = actions
        return index.detail_to_dict(detail)

    @app.get("/api/models/available")
    def models_available():
        return {"names": ModelSession.available_names()}

    @app.get("/api/models/loaded")
    def models_loaded():
        return {"models": session.loaded()}

    @app.post("/api/models/load")
    def models_load(body: LoadModelBody):
        try:
            info = session.load(
                body.name,
                checkpoint=body.checkpoint,
                temperature=body.temperature,
            )
        except Exception as e:
            raise HTTPException(400, str(e)) from e
        return info

    @app.delete("/api/models/{model_key}")
    def models_unload(model_key: str):
        try:
            return session.unload(model_key)
        except KeyError:
            raise HTTPException(404, f"Model not loaded: {model_key}")

    @app.get("/api/score/{replay_id}")
    def score(replay_id: str):
        if cache.get("replay_id") != replay_id or cache.get("states") is None:
            try:
                meta = index.get(replay_id)
            except KeyError:
                raise HTTPException(404, f"Replay not found: {replay_id}")
            states, actions = ReplayIndex.load_raw(meta.path)
            cache["replay_id"] = replay_id
            cache["states"] = states
            cache["actions"] = actions
        try:
            return score_replay(session, cache["states"], cache["actions"])
        except Exception as e:
            raise HTTPException(500, f"Scoring failed: {e}") from e

    @app.get("/api/sprites/{folder}/{sprite_id}.png")
    def gen1_sprite(folder: str, sprite_id: str):
        """Cropped gen1 / gen1-back art (empty canvas trimmed)."""
        try:
            png = get_cropped_gen1_sprite(folder, sprite_id.lower())
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except FileNotFoundError:
            raise HTTPException(404, f"Sprite not found: {folder}/{sprite_id}")
        except Exception as e:
            raise HTTPException(502, f"Failed to fetch sprite: {e}") from e
        return Response(
            content=png,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=604800"},
        )

    @app.get("/")
    def root():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app
