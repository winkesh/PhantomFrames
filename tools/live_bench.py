"""Frame rate of the running demo engine, end to end, with a browser-style H.264 stream attached.

    python -m tools.live_bench --res 720                      # sender/receiver processes + NVENC (the app default)
    python -m tools.live_bench --res 720 --single-process --encoder x264   # the old setup

Plays the held-out game clip through the real Engine at 10% loss for --secs seconds and reports the frames per
second the receiver delivered (the number the demo page shows) and the CPU use of the whole process tree.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="720", choices=["360", "720", "1080"])
    ap.add_argument("--single-process", action="store_true")
    ap.add_argument("--encoder", default="auto", choices=["auto", "nvenc", "x264", "none"])
    ap.add_argument("--secs", type=float, default=20)
    ap.add_argument("--clip", default="data/test/game_forza.avi")
    a = ap.parse_args()
    os.environ["PHANTOM_RES"] = a.res
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from phantom.app import server
    from phantom.app.engine import Engine
    from phantom.config import HEIGHT, WIDTH

    kw = {} if a.single_process else {"processes": True}
    try:
        e = Engine(udp_port=0, **kw)
    except TypeError:                          # an older engine without the processes option
        e = Engine(udp_port=0)
    e.update(loss=0.10)
    e.set_source(str(ROOT / a.clip), "file", "bench")
    stop = threading.Event()
    if a.encoder != "none":                    # a browser tab: steady 30 fps H.264 of the composite
        server.ENCODER = a.encoder
        w, h = 2 * WIDTH, HEIGHT
        level = "5.1" if w * h > 8192 * 256 else "4.0"
        if hasattr(server, "h264_encoder_args"):
            enc_args = server.h264_encoder_args("10M", level)
        else:                                  # the server before NVENC support: its libx264 settings
            enc_args = ["-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-profile:v", "high",
                        "-level:v", level, "-pix_fmt", "yuv420p", "-g", "30", "-bf", "0", "-b:v", "10M",
                        "-maxrate", "10M", "-bufsize", "4M"]
        cmd = [server.ffmpeg_exe(), "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
               "-r", "30", "-i", "-", *enc_args, "-f", "null", "-"]
        enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        q = e.subscribe()

        def feed():
            last = np.zeros((h, w, 3), np.uint8)
            t_next = time.perf_counter()
            while not stop.is_set():
                try:
                    while True:
                        last = q.get_nowait()
                except Exception:
                    pass
                enc.stdin.write(last.tobytes())
                t_next += 1 / 30
                time.sleep(max(0.0, t_next - time.perf_counter()))
        threading.Thread(target=feed, daemon=True).start()
    time.sleep(4)                              # warm-up (engines, caches)
    try:
        import psutil
        procs = [psutil.Process()] + psutil.Process().children(recursive=True)
        for p in procs:
            p.cpu_percent(None)
    except ImportError:
        procs = []
    n0, t0 = len(e.stats), time.time()
    fids0 = e.stats[-1]["fid"] if e.stats else 0
    time.sleep(a.secs)
    fids1 = e.stats[-1]["fid"] if e.stats else 0
    dt = time.time() - t0
    cpu = sum(p.cpu_percent(None) for p in procs) / (os.cpu_count() or 1) if procs else float("nan")
    infer = [r["infer_ms"] for r in list(e.stats)[-60:] if r["tiles_lost"]]
    last = list(e.stats)[-60:]
    med = lambda k: float(np.median([r[k] for r in last if k in r])) if any(k in r for r in last) else float("nan")
    stop.set()
    mode = "single process (threads)" if a.single_process else "sender + receiver processes"
    used = a.encoder if a.encoder != "auto" else (
        "nvenc" if getattr(server, "nvenc_available", lambda: False)() else "x264")
    print(f"{a.res}p, {mode}, encoder={used}: "
          f"{(fids1 - fids0) / dt:.1f} fps delivered, repair {np.median(infer) if infer else 0:.1f} ms, "
          f"CPU {cpu:.0f}% of all cores; receiver per frame {med('recv_ms'):.1f} ms (decode {med('dec_ms'):.1f})",
          flush=True)
    for p in getattr(e, "_procs", []):        # stop the worker processes before leaving
        p.terminate()
    os._exit(0)


if __name__ == "__main__":
    main()
