"""Per-step timing of the demo's sender and receiver at the current PHANTOM_RES (default 360p).

    PHANTOM_RES=720 python -m tools.stage_profile

Runs the real Engine steps on the held-out game clip at 10% loss and prints the median time of each
step, so the slowest parts of the frame budget (33 ms at 30 fps) are visible.
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phantom.app.engine import Engine  # noqa: E402
from phantom.channel import make_loss  # noqa: E402
from phantom.config import HEIGHT, N_TILES, WIDTH  # noqa: E402
from phantom.receiver import FrameState, assemble, decode_state  # noqa: E402
from phantom.tiles import letterbox, tile_mask_to_pixels  # noqa: E402
from phantom.video import load_video  # noqa: E402


def main(n: int = 60):
    e = Engine(udp_port=0)
    e.update(loss=0.10)
    frames = [letterbox(f) for f in load_video(ROOT / "data/test/game_forza.avi", max_frames=n + 10)]
    T = defaultdict(list)
    model = e.models[e.settings.model]
    hist_b, hist_p = [], []
    for k, f in enumerate(frames):
        t = time.perf_counter(); pkts = e.pipe.pk.packetize(f); T["send: packetize (JPEG + RTP)"].append(time.perf_counter() - t)
        t = time.perf_counter()
        lm = make_loss("bernoulli", 0.10, 1.0, seed=k); st = FrameState(k)
        for p in pkts:
            if not lm.lost():
                st.add(p)
        T["send: channel + FrameState.add"].append(time.perf_counter() - t)
        t = time.perf_counter(); decode_state(st, memo=False); T["recv: JPEG decode"].append(time.perf_counter() - t)
        t = time.perf_counter(); a_b = assemble(st, k, hist_b[-1] if hist_b else None, memo=False); T["recv: assemble #1 (paste)"].append(time.perf_counter() - t)
        t = time.perf_counter(); a_p = assemble(st, k, hist_p[-1] if hist_p else None, memo=False); T["recv: assemble #2 (paste)"].append(time.perf_counter() - t)
        t = time.perf_counter(); mask = tile_mask_to_pixels(a_p.missing); T["recv: mask"].append(time.perf_counter() - t)
        t = time.perf_counter(); out_b = e.baselines["none"](a_b.image, mask, hist_b) if a_p.missing else a_b.image; T["recv: baseline"].append(time.perf_counter() - t)
        t = time.perf_counter(); out_p = model(a_p.image, mask, hist_p) if a_p.missing else a_p.image; T["recv: repair"].append(time.perf_counter() - t)
        hist_b = (hist_b + [out_b])[-3:]; hist_p = (hist_p + [out_p])[-3:]
        t = time.perf_counter(); comp = np.hstack([out_b, out_p]); T["recv: compose"].append(time.perf_counter() - t)
        t = time.perf_counter(); cv2.PSNR(out_b, f); cv2.PSNR(out_p, f); T["recv: 2x PSNR"].append(time.perf_counter() - t)
    print(f"{HEIGHT}p ({WIDTH}x{HEIGHT}, {N_TILES} tiles), repair: {getattr(model, 'label', '')}")
    tot = defaultdict(float)
    for k, v in T.items():
        ms = 1000 * float(np.median(v[10:]))
        tot[k.split(":")[0]] += ms
        print(f"  {k:36s} {ms:6.1f} ms")
    for k, v in tot.items():
        print(f"  {k + ' total':36s} {v:6.1f} ms")


if __name__ == "__main__":
    main()
