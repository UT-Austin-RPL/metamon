"""Showdown sprite helpers for the analyze UI.

Gen1 sheets sit in a 96×96 canvas with large empty padding. Back sprites are
half-body with a flat crop; fronts are full-ish but still float above our
pedestal if shown uncropped. Showdown offsets via spriteData.y in its 3D
scene — we trim to the painted alpha bbox for the 2D analyze UI.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from functools import lru_cache
from io import BytesIO

from PIL import Image

_SPRITE_ID_RE = re.compile(r"^[a-z0-9]+$")
_FOLDER_RE = re.compile(r"^gen1(-back)?$")
_SHOWDOWN = "https://play.pokemonshowdown.com/sprites"
_UA = "metamon-analyze/0.1"


def _fetch_png(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read()


def crop_gen1_png(
    raw: bytes, *, scale: int = 3, keep_flat_bottom: bool = False
) -> bytes:
    """Trim empty canvas; nearest-neighbor upscale.

    keep_flat_bottom: for gen1-back half-body sheets — don't pad under the crop.
    """
    im = Image.open(BytesIO(raw)).convert("RGBA")
    bbox = im.split()[-1].getbbox()
    if bbox is None:
        out = im
    else:
        left, top, right, bottom = bbox
        pad = 2
        left = max(0, left - pad)
        top = max(0, top - pad)
        right = min(im.width, right + pad)
        if not keep_flat_bottom:
            bottom = min(im.height, bottom + pad)
        out = im.crop((left, top, right, bottom))
    if scale > 1:
        out = out.resize(
            (out.width * scale, out.height * scale),
            resample=Image.Resampling.NEAREST,
        )
    buf = BytesIO()
    out.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


@lru_cache(maxsize=1024)
def get_cropped_gen1_sprite(folder: str, sprite_id: str) -> bytes:
    if not _FOLDER_RE.match(folder):
        raise ValueError(f"invalid sprite folder: {folder!r}")
    if not _SPRITE_ID_RE.match(sprite_id):
        raise ValueError(f"invalid sprite id: {sprite_id!r}")
    url = f"{_SHOWDOWN}/{folder}/{sprite_id}.png"
    try:
        raw = _fetch_png(url)
    except urllib.error.HTTPError as e:
        raise FileNotFoundError(sprite_id) from e
    return crop_gen1_png(raw, keep_flat_bottom=folder.endswith("-back"))


# Back-compat alias used by older imports / tests.
def get_cropped_gen1_back(sprite_id: str) -> bytes:
    return get_cropped_gen1_sprite("gen1-back", sprite_id)
