"""Sender machine: one command.

    python -m phantom.send                         # webcam, finds the receiver on this network by itself
    python -m phantom.send --src clip.mp4          # any video file (loops), at its own frame rate
    python -m phantom.send --src https://.../x.m3u8   # or a stream URL OpenCV can open (http, HLS, RTSP)
    python -m phantom.send --to 192.168.1.42       # skip broadcast discovery: send straight to that receiver
    python -m phantom.send --res 720               # force a resolution (normally the receiver's is used)

Start the web demo on the receiving machine first (python -m phantom.app [--res 720]); it listens for senders
and tells each sender which resolution it runs, so both ends match without setting anything here.
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import time

from .live.discovery import discover   # no phantom.config import yet: the resolution is not fixed until below


def main():
    ap = argparse.ArgumentParser(description="Phantom Frames sender")
    ap.add_argument("--to", help="receiver IP (or IP:port); found automatically when left out")
    ap.add_argument("--src", default="0",
                    help="webcam index (0 = first camera), a video file, or a stream URL (http/HLS/RTSP)")
    ap.add_argument("--port", type=int, default=5004)
    ap.add_argument("--res", choices=["360", "720", "1080"],
                    help="stream resolution; default: whatever the receiver runs")
    ap.add_argument("--quality", type=int, default=75, help="JPEG quality of the tiles (lower = fewer Mbit/s)")
    a = ap.parse_args()

    if a.to:
        host, _, port = a.to.partition(":")
        ip, port = socket.gethostbyname(host), int(port or a.port)
        found = discover(port, target=ip, tries=2)          # ask it directly which resolution it runs
        res = found[1] if found else None
        if found is None and not a.res:
            print(f"[warn] {ip}:{port} did not answer; sending at {os.environ.get('PHANTOM_RES', '360')}p. "
                  f"If the demo shows nothing, pass the receiver's --res here too.", flush=True)
    else:
        print("Looking for a receiver on this network...", flush=True)
        found = discover(a.port)
        if found is None:
            sys.exit("No receiver found. Start the web demo (python -m phantom.app) on the receiving machine, or give its "
                     "address:  python -m phantom.send --to <receiver-ip>  (it prints its IP when it starts).")
        ip, port, res = found[0], a.port, found[1]
        print(f"Found receiver at {ip} ({res}p)", flush=True)
    if a.res:
        if res and int(a.res) != res:
            print(f"[warn] the receiver runs {res}p but --res {a.res} was given: it will reject these packets", flush=True)
        res = int(a.res)
    if res:
        os.environ["PHANTOM_RES"] = str(res)                 # before anything imports phantom.config
    from .config import HEIGHT, WIDTH
    from .live.endpoints import LiveSender
    from . import tilecodec

    try:
        tx = LiveSender(a.src, (ip, port), quality=a.quality).start()
    except RuntimeError as e:
        sys.exit(f"{e}. For the webcam, close other apps using it (Teams, Zoom, Camera) and allow desktop apps "
                 f"to use the camera in Windows privacy settings; or try --src 1.")
    src = tx.source
    what = f"webcam {a.src}" if str(a.src).isdigit() else a.src
    print(f"Streaming {what} ({src.size[0]}x{src.size[1]}) as {WIDTH}x{HEIGHT} at {tx.fps:.0f} fps to {ip}:{port}"
          f"   (tile codec: {'native' if tilecodec.NATIVE else 'OpenCV'}; Ctrl+C to stop)", flush=True)
    last, t = 0, time.time()
    try:
        while True:
            time.sleep(2)
            now = time.time()
            rate = (tx.bytes_sent - last) * 8 / (now - t) / 1e6
            fb = (f"receiver sees {tx.reported_loss:.0%} packet loss" if now - tx.last_feedback < 2
                  else "no reply from the receiver (is it running? firewall?)")
            print(f"frame {tx.pk.frame_id:6d}   {rate:5.1f} Mbit/s   {fb}", flush=True)
            last, t = tx.bytes_sent, now
    except KeyboardInterrupt:
        tx.stop()


if __name__ == "__main__":
    main()
