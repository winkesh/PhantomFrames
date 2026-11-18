"""Tile grid, per-tile JPEG coding and loss masks."""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from .config import HEIGHT, N_TILES, TILE, TILES_X, TILES_Y, WIDTH


def tile_rect(tid: int) -> tuple[int, int, int, int]:
    """(y0, y1, x0, x1) of a tile."""
    ty, tx = divmod(tid, TILES_X)
    return ty * TILE, (ty + 1) * TILE, tx * TILE, (tx + 1) * TILE


def split(frame: np.ndarray) -> list[np.ndarray]:
    assert frame.shape[:2] == (HEIGHT, WIDTH), frame.shape
    t = frame.reshape(TILES_Y, TILE, TILES_X, TILE, 3).swapaxes(1, 2)
    return [t[i // TILES_X, i % TILES_X] for i in range(N_TILES)]


def encode_tile(tile: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", tile, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return buf.tobytes()


def decode_tile_raw(data: bytes) -> np.ndarray | None:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None or img.shape != (TILE, TILE, 3):
        return None
    return img


@lru_cache(maxsize=60000)
def _decode_cached(data: bytes) -> np.ndarray | None:
    img = decode_tile_raw(data)
    if img is not None:
        img.setflags(write=False)
    return img


def decode_tile(data: bytes) -> np.ndarray | None:
    """JPEG-decode one tile (memoised: the simulator decodes the same tiles in many runs)."""
    return _decode_cached(bytes(data))


def paste(frame: np.ndarray, tid: int, tile: np.ndarray) -> None:
    y0, y1, x0, x1 = tile_rect(tid)
    frame[y0:y1, x0:x1] = tile


# ---- whole-frame tile operations, vectorised: a frame seen as a (rows, TILE, cols, TILE, 3) grid of tiles, so
# copying, blanking or pasting k tiles is one indexed assignment touching only those tiles (a masked copy over a
# 720p frame costs ~4 ms; this costs ~0.1 ms for a typical 10% loss).

def tile_view(frame: np.ndarray) -> np.ndarray:
    """(TILES_Y, TILE, TILES_X, TILE, 3) view of a contiguous HxWx3 frame (writes go to the frame)."""
    return frame.reshape(TILES_Y, TILE, TILES_X, TILE, frame.shape[2])


def tile_grid(mask: np.ndarray) -> np.ndarray:
    """(TILES_Y, TILES_X) bool grid from a tile-aligned HxW pixel mask."""
    return mask[::TILE, ::TILE]


def tile_coords(tids) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(sorted(tids) if isinstance(tids, (set, frozenset)) else tids, dtype=np.intp)
    return t // TILES_X, t % TILES_X


def copy_tiles(dst: np.ndarray, src: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """dst's tiles where grid is True replaced by src's (in place); returns dst."""
    ys, xs = np.nonzero(grid)
    if len(ys):
        tile_view(dst)[ys, :, xs] = tile_view(src)[ys, :, xs]
    return dst


def fill_tiles(dst: np.ndarray, grid: np.ndarray, value: int = 0) -> np.ndarray:
    ys, xs = np.nonzero(grid)
    if len(ys):
        tile_view(dst)[ys, :, xs] = value
    return dst


def paste_many(frame: np.ndarray, tids: np.ndarray, stack: np.ndarray) -> None:
    """Paste stack[k] (TILE x TILE x 3) at tile tids[k], for all k at once."""
    if len(tids):
        ys, xs = tile_coords(tids)
        tile_view(frame)[ys, :, xs] = stack


def tile_mask_to_pixels(missing: set[int] | np.ndarray) -> np.ndarray:
    """Boolean HxW mask, True where pixels are missing."""
    m = np.zeros((TILES_Y, TILES_X), bool)
    if len(missing):
        ys, xs = tile_coords(missing)
        m[ys, xs] = True
    return np.ascontiguousarray(np.broadcast_to(m[:, None, :, None], (TILES_Y, TILE, TILES_X, TILE))
                                ).reshape(HEIGHT, WIDTH)


def resize_frame(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)


def letterbox(img: np.ndarray) -> np.ndarray:
    """Fit any input into WIDTHxHEIGHT without distortion (black bars), e.g. a 4:3 webcam."""
    h, w = img.shape[:2]
    if (w, h) == (WIDTH, HEIGHT):
        return img
    s = min(WIDTH / w, HEIGHT / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    out = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    y, x = (HEIGHT - nh) // 2, (WIDTH - nw) // 2
    out[y:y + nh, x:x + nw] = small
    return out
