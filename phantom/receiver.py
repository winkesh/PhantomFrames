"""Receiver: reassemble packets into frames, recover with FEC, and report which tiles are missing."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import tilecodec as TC
from . import tiles as T
from .config import HEIGHT, N_TILES, WIDTH
from .packet import FLAG_PARITY, Packet, bitmap_ids, unpack_tiles
from .sender import xor_bytes


@dataclass
class FrameState:
    frame_id: int
    n_pkts: int = 0
    fec_k: int = 0
    media: dict = field(default_factory=dict)     # pkt_idx -> payload
    parity: dict = field(default_factory=dict)    # group idx -> payload
    skip: set = field(default_factory=set)
    first_arrival: float | None = None
    decoded: tuple | None = None                  # (tile ids, stacked tiles), filled by assemble()
    _n_decoded: int = -1

    def add(self, p: Packet, now: float = 0.0) -> None:
        self.n_pkts = p.n_pkts
        self.fec_k = p.fec_k
        if self.first_arrival is None:   # skip map is identical in every packet of a frame
            self.skip = bitmap_ids(p.skip_map)
        if self.first_arrival is None:
            self.first_arrival = now
        if p.flags & FLAG_PARITY:
            self.parity[p.pkt_idx] = p.payload
        else:
            self.media[p.pkt_idx] = p.payload

    def fec_recover(self) -> int:
        """Recover single losses per parity group. Returns number recovered."""
        if not self.fec_k:
            return 0
        rec = 0
        for gi, par in self.parity.items():
            idx = range(gi * self.fec_k, min((gi + 1) * self.fec_k, self.n_pkts))
            missing = [i for i in idx if i not in self.media]
            if len(missing) == 1:
                self.media[missing[0]] = xor_bytes([par] + [self.media[i] for i in idx if i in self.media])
                rec += 1
        return rec

    @property
    def complete(self) -> bool:
        return self.n_pkts > 0 and len(self.media) >= self.n_pkts


@dataclass
class Assembled:
    frame_id: int
    image: np.ndarray            # tiles that arrived pasted over the previous output
    missing: set                 # tile ids lost in the network
    any_packet: bool             # False when nothing at all arrived for this frame


def decode_state(state: FrameState, memo: bool) -> tuple[np.ndarray, np.ndarray]:
    """JPEG-decode every tile that arrived, once per FrameState: (tile ids, (k, TILE, TILE, 3) tiles).
    memo: reuse decodes of identical tile bytes (the simulator replays the same tiles across many runs);
    the live app decodes fresh tiles, where the cache would only cost memory.
    One thread on purpose: a 40x40 decode is so short that per-call overhead under the GIL dominates, and a
    thread pool measured slower (9.8 ms single vs 10-13 ms pooled for 576 tiles at 720p)."""
    if state.decoded is None or len(state.media) != state._n_decoded:
        state.fec_recover()
        items = [(tid, data) for payload in state.media.values() for tid, data in unpack_tiles(payload)
                 if tid < N_TILES]
        if not memo and TC.NATIVE:            # live app: whole frame in one native call (same pixels as cv2)
            ok, stack = TC.decode_tiles([d for _, d in items])
            ids = np.asarray([tid for tid, _ in items], np.intp)
            state.decoded = (ids[ok], stack[ok]) if not ok.all() else (ids, stack)
        else:
            dec = T.decode_tile if memo else T.decode_tile_raw
            ids, tiles = [], []
            for tid, data in items:
                tile = dec(data)
                if tile is not None:
                    ids.append(tid); tiles.append(tile)
            stack = np.stack(tiles) if tiles else np.zeros((0, T.TILE, T.TILE, 3), np.uint8)
            state.decoded = (np.asarray(ids, np.intp), stack)
        state._n_decoded = len(state.media)
    return state.decoded


def assemble(state: FrameState | None, frame_id: int, prev: np.ndarray | None, pool=None,
             memo: bool | None = None) -> Assembled:
    """Build the partial frame. Skipped (unchanged) tiles are copied from prev; lost tiles reported.
    memo: memoise tile decodes (default: on unless a pool is passed, as the live app used to)."""
    base = prev.copy() if prev is not None else np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    if state is None or not state.media and not state.parity:
        return Assembled(frame_id, base, set(range(N_TILES)), False)
    ids, stack = decode_state(state, memo=(pool is None) if memo is None else memo)
    T.paste_many(base, ids, stack)
    missing = set(range(N_TILES)) - set(ids.tolist()) - state.skip
    return Assembled(frame_id, base, missing, True)
