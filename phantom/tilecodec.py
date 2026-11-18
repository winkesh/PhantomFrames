"""Whole-frame tile JPEG coding: the native batch codec when it is built, else OpenCV one tile at a time.

The native codec (phantom/native/_tilecodec.cpp, libjpeg-turbo) codes a frame's tiles in one call on several
threads with the GIL released; OpenCV costs one Python call per tile (576 at 720p, 1296 at 1080p), which is where
the time goes. Both give standard baseline 4:2:0 JPEG tiles, and the native decoder returns the same pixels as
cv2.imdecode for the same bytes (same library, same accurate DCT and upsampling).

PHANTOM_NATIVE_CODEC=0 forces the OpenCV path; PHANTOM_CODEC_THREADS sets the native thread count.
"""
from __future__ import annotations

import os

import numpy as np

from . import tiles as T
from .config import TILE, TILES_X

try:
    from .native import _tilecodec as _native
except ImportError:
    _native = None
if os.environ.get("PHANTOM_NATIVE_CODEC", "1") == "0":
    _native = None

NATIVE = _native is not None
THREADS = int(os.environ.get("PHANTOM_CODEC_THREADS", "0")) or min(8, os.cpu_count() or 4)


def encode_tiles(frame: np.ndarray, tile_ids, quality: int) -> list[bytes]:
    """JPEG-encode the given tiles of an HxWx3 BGR frame, in order."""
    ids = list(tile_ids)
    if _native is not None:
        return _native.encode_tiles(np.ascontiguousarray(frame), ids, TILE, int(quality), THREADS)
    tv = T.tile_view(np.ascontiguousarray(frame))
    return [T.encode_tile(tv[i // TILES_X, :, i % TILES_X], quality) for i in ids]


def decode_tiles(jpegs: list) -> tuple[np.ndarray, np.ndarray]:
    """Decode tile JPEGs: (ok mask, (n, TILE, TILE, 3) BGR tiles; rows where ok is False are undefined)."""
    n = len(jpegs)
    out = np.empty((n, TILE, TILE, 3), np.uint8)
    if _native is not None:
        ok = np.frombuffer(_native.decode_tiles(jpegs, out, TILE, THREADS), np.uint8).astype(bool)
        return ok, out
    ok = np.zeros(n, bool)
    for i, data in enumerate(jpegs):
        tile = T.decode_tile_raw(data)
        if tile is not None:
            out[i], ok[i] = tile, True
    return ok, out
