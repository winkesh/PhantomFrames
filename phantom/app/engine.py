"""Live streaming engine behind the demo app.

source (file / URL / webcam) -> frames -> real Packetizer (RTP packets, interleaved JPEG tiles)
  -> lossy channel (same Bernoulli / Gilbert-Elliott models as the UDP relay)
  -> real receiver (FrameState + assemble)
or: another machine running `python -m phantom.send` -> real UDP packets on port 5004 -> loss simulated on
    arrival -> the same receiver path (the app answers the sender's discovery broadcast and switches to
    the network stream when one starts)
  -> left:  what the loss does (no repair | normal decoder freeze | copy last tile)
     right: Phantom regenerating the lost tiles (RepairNet, or framegen with one frame of look-ahead)
  -> composite [left | right] published to the video streams + per-frame stats.

Two ways to run it:
  processes=True (the app): the sender (JPEG + packets + loss) and the receiver (decode + repair + compose) each
      run in their own process, like two real machines, and the web server, source reader and UDP listener stay
      in this one. Each stage then has a core and a GIL of its own: tile coding is ~1,000 tiny cv2 calls a
      frame at 720p, which threads in one process could not run in parallel. Frames move through shared-memory
      rings (no pickling of images); only the packets themselves (~100 KB a frame) go through queues.
  processes=False (tools/benchmarks): everything in this process, stages on threads, as before.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from multiprocessing import shared_memory
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from .. import conceal as C
from ..channel import make_loss
from ..config import FPS, HEIGHT, N_TILES, WIDTH
from ..live.endpoints import LiveReceiver, local_ips
from ..packet import HEADER_SIZE
from ..receiver import FrameState, assemble
from ..sender import Packetizer
from ..tiles import tile_mask_to_pixels

TARGET_FPS = FPS
IN_SLOTS, OUT_SLOTS = 6, 4          # shared-memory frames in flight: source -> sender/receiver, receiver -> app


@dataclass
class Settings:
    loss: float = 0.10          # packet loss rate
    burst: float = 1.0          # mean burst length in packets (1 = independent/random loss)
    baseline: str = "none"      # left panel: none | freeze | copy
    model: str = "repairnet"    # right panel: repairnet | framegen
    left: str = "loss"          # loss | original
    outline: bool = False       # outline lost tiles on the left panel
    paused: bool = False


def letterbox(img: np.ndarray) -> np.ndarray:
    """Fit any input into WIDTHxHEIGHT without distortion (black bars)."""
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


class Source:
    def __init__(self, spec: str, kind: str, label: str):
        self.spec, self.kind, self.label = spec, kind, label
        self.cap = cv2.VideoCapture(int(spec) if kind == "webcam" else spec)
        if not self.cap.isOpened():
            raise IOError(f"could not open {label}")
        ok, first = self.cap.read()
        if not ok:
            raise IOError(f"{label} opened but no frames could be decoded")
        self.first = first
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if 1 < fps < 241 else 30.0
        self.size = (first.shape[1], first.shape[0])
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if kind == "file" else 0
        # high-fps sources are decimated to ~30 fps (60 fps game capture -> every 2nd frame)
        self.stride = max(1, round(self.fps / TARGET_FPS))
        self.out_fps = self.fps / self.stride

    def read(self):
        if self.first is not None:
            f, self.first = self.first, None
            return f
        for _ in range(self.stride - 1):
            self.cap.grab()
        ok, img = self.cap.read()
        if not ok and self.kind in ("file", "url"):  # loop files / VOD URLs: reopen (live URLs reconnect)
            self.cap.release()
            self.cap = cv2.VideoCapture(self.spec)
            ok, img = self.cap.read()
        return img if ok else None

    def close(self):
        self.cap.release()


class NetSource:
    """A stream arriving over UDP from `python -m phantom.send` on another machine."""
    kind, n, out_fps = "network", 0, float(FPS)

    def __init__(self, peer_ip: str):
        self.label = f"Live from {peer_ip}"
        self.size = (WIDTH, HEIGHT)

    def close(self):
        pass


def load_models(weights: str, framegen: str, backend: str, repair_res: str) -> dict:
    models: dict[str, C.Concealer] = {}
    if Path(weights).exists():
        models["repairnet"] = C.make_repairnet(weights, backend, repair_res)
    if Path(framegen).exists():
        fb = models.get("repairnet") or C.FlowExtrapolate()
        models["framegen"] = C.FramegenInterp(framegen, fallback=fb)
    if not models:
        models["flow"] = C.FlowExtrapolate()
    return models


def make_baselines() -> dict:
    return {"none": C.NoRecovery(), "freeze": C.Freeze(), "copy": C.CopyTile()}


class Pipeline:
    """The sender and receiver halves of the demo, with no threads or processes of their own.
    send(): frame -> real packets -> lossy channel. recv(): packets -> both panels -> composite + stats."""

    def __init__(self, models: dict | None = None, baselines: dict | None = None, pool=None):
        self.models, self.baselines, self.pool = models or {}, baselines or {}, pool
        self.reset()

    def reset(self):
        self.pk = Packetizer(quality=80, interleave=True)
        self.pk.pool = self.pool
        self.hist_b: list = []
        self.hist_p: list = []
        self.pending = None                         # framegen: (frame, original) waiting for frame t+1
        self.fid = 0
        self._loss_key, self.loss_model = None, None

    # ---- sender half
    def send(self, orig: np.ndarray, s: Settings) -> dict:
        t0 = time.perf_counter()
        pkts = self.pk.packetize(orig)              # real packetizer
        key = (s.loss, s.burst)                     # lossy channel
        if key != self._loss_key:
            self.loss_model = make_loss("ge" if s.burst > 1.01 else "bernoulli", s.loss, s.burst,
                                        seed=int(time.time()))
            self._loss_key = key
        st = FrameState(self.fid)
        lost = 0
        for p in pkts:
            if self.loss_model.lost():
                lost += 1
            else:
                st.add(p)
        cur = dict(fid=self.fid, st=st, pkts=len(pkts), lost=lost, bytes=sum(p.size for p in pkts), t0=t0)
        self.fid += 1
        return cur

    # ---- receiver half
    def recv(self, cur: dict, orig: np.ndarray | None, s: Settings) -> list:
        """Returns [(composite, stats, missing tiles)]: framegen needs frame t+1 before it can repair frame t,
        so it returns nothing for the first frame and then one result per call."""
        if s.model == "framegen":
            prev, self.pending = self.pending, (cur, None if orig is None else orig.copy())
            return [] if prev is None else [self._receive(prev[0], prev[1], s, nxt=cur)]
        out = []
        if self.pending is not None:                # switched away from framegen: flush
            out.append(self._receive(*self.pending, s, nxt=None))
            self.pending = None
        out.append(self._receive(cur, orig, s, nxt=None))
        return out

    def _receive(self, f: dict, orig: np.ndarray | None, s: Settings, nxt: dict | None):
        st = f["st"]
        tr = time.perf_counter()
        a_b = assemble(st, f["fid"], self.hist_b[-1] if self.hist_b else None, memo=False)
        dec = (time.perf_counter() - tr) * 1000
        a_p = assemble(st, f["fid"], self.hist_p[-1] if self.hist_p else None, memo=False)
        miss = a_p.missing
        mask = tile_mask_to_pixels(miss)
        out_b = self.baselines[s.baseline](a_b.image, mask, self.hist_b) if miss else a_b.image
        model = self.models[s.model]
        ti = time.perf_counter()
        if not miss:
            out_p = a_p.image
        elif s.model == "framegen" and nxt is not None:
            n_asm = assemble(nxt["st"], nxt["fid"], a_p.image, memo=False)
            out_p = model(a_p.image, mask, self.hist_p, nxt=n_asm.image, nxt_missing_frac=len(n_asm.missing) / N_TILES)
        else:
            out_p = model(a_p.image, mask, self.hist_p)
        infer = (time.perf_counter() - ti) * 1000 if miss else 0.0
        for h, o in ((self.hist_b, out_b), (self.hist_p, out_p)):
            h.append(o)
            if len(h) > 3:
                h.pop(0)
        show_orig = s.left == "original" and orig is not None   # a live network stream has no original
        left = orig if show_orig else out_b
        if s.outline and miss and not show_orig:
            left = left.copy()
            edge = mask & ~cv2.erode(mask.view(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            left[edge] = (60, 60, 230)
        comp = np.hstack([left, out_p])
        stat = dict(
            t=time.time(), fid=f["fid"], pkts=f["pkts"], lost=f["lost"], bytes=f["bytes"],
            tiles_lost=len(miss),
            psnr_b=None if orig is None else min(60.0, cv2.PSNR(out_b, orig)),
            psnr_p=None if orig is None else min(60.0, cv2.PSNR(out_p, orig)),
            infer_ms=infer, proc_ms=(time.perf_counter() - f["t0"]) * 1000,
            dec_ms=dec, recv_ms=(time.perf_counter() - tr) * 1000)
        return comp, stat, sorted(miss)


# ---------------------------------------------------------------- worker processes (top level: spawn imports them)
def _next(q):
    """q.get() for a worker process that exits as soon as the app is gone (closed, crashed or killed), so a
    worker can never outlive it holding the GPU."""
    parent = mp.parent_process()
    while True:
        try:
            return q.get(timeout=1.0)
        except queue.Empty:
            if parent is not None and not parent.is_alive():
                os._exit(0)


def full_speed():
    """Windows 11 may run windowless background processes in efficiency mode (EcoQoS: lower clocks,
    efficiency cores), which slows the worker processes most when the CPU is lightly loaded. Opt this process
    out of execution-speed throttling (a per-process setting; nothing system-wide changes)."""
    if os.name != "nt" or os.environ.get("PHANTOM_ALLOW_THROTTLING"):
        return
    try:
        import ctypes
        from ctypes import wintypes

        class _PPTS(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]
        st = _PPTS(1, 0x1 | 0x4, 0)            # execution speed + timer resolution: controlled, not throttled
        k32 = ctypes.windll.kernel32
        k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.SetProcessInformation(k32.GetCurrentProcess(), 4, ctypes.byref(st), ctypes.sizeof(st))
    except Exception:
        pass


def _limit_threads(torch_too: bool = False):
    """Thread budget of a worker process (PHANTOM_WORKER_THREADS, 0 = library defaults). OpenCV and PyTorch
    each default to one thread per logical CPU, so several processes together can oversubscribe the cores."""
    full_speed()
    n = int(os.environ.get("PHANTOM_WORKER_THREADS", "0"))
    if n > 0:
        cv2.setNumThreads(n)
        if torch_too:
            import torch
            torch.set_num_threads(n)


def _attach(name: str, slots: int, shape: tuple):
    shm = shared_memory.SharedMemory(name=name)
    return shm, np.ndarray((slots, *shape), np.uint8, buffer=shm.buf)


def _sender_proc(in_name, send_q, recv_q, in_free):
    _limit_threads()
    shm, ring = _attach(in_name, IN_SLOTS, (HEIGHT, WIDTH, 3))
    pool_n = int(os.environ.get("PHANTOM_SENDER_POOL", "4"))
    pipe, s, gen = Pipeline(pool=ThreadPoolExecutor(pool_n) if pool_n > 0 else None), Settings(), 0
    while True:
        msg = _next(send_q)
        kind = msg[0]
        if kind == "settings":
            s = Settings(**msg[1])
        elif kind == "reset":
            gen = msg[1]
            pipe.reset()
        elif kind == "frame":
            _, g, slot = msg
            if g != gen:
                in_free.put(slot)
                continue
            try:
                cur = pipe.send(ring[slot], s)
            except Exception as e:                   # keep the app alive; show the error in the UI
                recv_q.put(("error", f"sender error: {e!r}"))
                in_free.put(slot)
                continue
            recv_q.put(("frame", g, slot, cur))       # blocks when the receiver is behind: backpressure


def _receiver_proc(cfg, in_name, out_name, recv_q, res_q, in_free, out_free):
    _limit_threads(torch_too=True)
    shm_in, ring_in = _attach(in_name, IN_SLOTS, (HEIGHT, WIDTH, 3))       # keep the handles: dropping them
    shm_out, ring_out = _attach(out_name, OUT_SLOTS, (HEIGHT, 2 * WIDTH, 3))  # unmaps the rings
    try:
        models = load_models(*cfg)
    except Exception as e:
        res_q.put(("ready", {}, f"model loading failed: {e!r}"))
        return
    res_q.put(("ready", {k: getattr(m, "label", "") for k, m in models.items()}, ""))
    pipe, s, gen = Pipeline(models, make_baselines()), Settings(), 0
    while True:
        msg = _next(recv_q)
        kind = msg[0]
        if kind == "settings":
            s = Settings(**msg[1])
        elif kind == "reset":
            gen = msg[1]
            pipe.reset()
        elif kind == "error":
            res_q.put(msg)
        elif kind == "frame":
            _, g, slot, cur = msg
            if g != gen:
                if slot is not None:
                    in_free.put(slot)
                continue
            try:
                results = pipe.recv(cur, None if slot is None else ring_in[slot], s)
            except Exception as e:
                res_q.put(("error", f"receiver error: {e!r}"))
                results = []
            if slot is not None:
                in_free.put(slot)
            for comp, stat, missing in results:
                try:
                    oslot = out_free.get(timeout=2.0)
                except queue.Empty:                   # the app stopped reading: drop
                    continue
                ring_out[oslot] = comp
                res_q.put(("out", g, oslot, stat, missing, bool(cur.get("net"))))


# ---------------------------------------------------------------- the engine the web app talks to
class Engine:
    def __init__(self, weights: str = "weights/repairnet.pt", framegen: str = "weights/framegen.pt",
                 udp_port: int = 5004, backend: str = "auto", repair_res: str = "auto", processes: bool = False):
        self.settings = Settings()
        self.lock = threading.Lock()
        self.processes = processes
        self.baselines = make_baselines()
        self.source: Source | None = None
        self.source_error = ""
        self.subscribers: list[queue.Queue] = []
        self.latest: np.ndarray | None = None
        self.stats: deque = deque(maxlen=300)
        self._gen = 0                               # bumps on every source change
        cfg = (weights, framegen, backend, repair_res)
        full_speed()                                # this process too (reader, UDP listener, stream encoder feed)
        if processes:
            self._start_processes(cfg)
        else:
            self.pool = ThreadPoolExecutor(8)
            self.models = load_models(*cfg)
            self.pipe = Pipeline(self.models, self.baselines, self.pool)
            self.frames: queue.Queue = queue.Queue(maxsize=3)    # reader -> sender
            self.net_q: queue.Queue = queue.Queue(maxsize=3)     # sender / network -> receiver
        if self.settings.model not in self.models:
            self.settings.model = next(iter(self.models))
        self._push_settings()
        # receiving from another machine: always listening, answers discovery broadcasts
        self.udp_port, self.net_error, self.rx = udp_port, "", None
        self._net_ssrc, self._net_key, self._avg_pkts = None, None, 0.0
        self.ips = local_ips()
        try:
            self.rx = LiveReceiver(udp_port, deadline_ms=150)
        except OSError as e:
            self.net_error = f"cannot listen on UDP {udp_port}: {e}"
        threading.Thread(target=self._reader, daemon=True).start()
        if processes:
            threading.Thread(target=self._dispatch, daemon=True).start()
        else:
            threading.Thread(target=self._sender, daemon=True).start()
            threading.Thread(target=self._worker, daemon=True).start()
        if self.rx is not None:
            threading.Thread(target=self._net_loop, daemon=True).start()

    def _start_processes(self, cfg):
        ctx = mp.get_context("spawn")               # same behaviour on Windows and Linux; CUDA-safe
        fsize, osize = HEIGHT * WIDTH * 3, HEIGHT * 2 * WIDTH * 3
        self._shm_in = shared_memory.SharedMemory(create=True, size=IN_SLOTS * fsize)
        self._shm_out = shared_memory.SharedMemory(create=True, size=OUT_SLOTS * osize)
        self.ring_in = np.ndarray((IN_SLOTS, HEIGHT, WIDTH, 3), np.uint8, buffer=self._shm_in.buf)
        self.ring_out = np.ndarray((OUT_SLOTS, HEIGHT, 2 * WIDTH, 3), np.uint8, buffer=self._shm_out.buf)
        self.send_q, self.recv_q, self.res_q = ctx.Queue(), ctx.Queue(maxsize=8), ctx.Queue()
        self.in_free, self.out_free = ctx.Queue(), ctx.Queue()
        for i in range(IN_SLOTS):
            self.in_free.put(i)
        for i in range(OUT_SLOTS):
            self.out_free.put(i)
        self._procs = [
            ctx.Process(target=_sender_proc, name="phantom-sender", daemon=True,
                        args=(self._shm_in.name, self.send_q, self.recv_q, self.in_free)),
            ctx.Process(target=_receiver_proc, name="phantom-receiver", daemon=True,
                        args=(cfg, self._shm_in.name, self._shm_out.name, self.recv_q, self.res_q,
                              self.in_free, self.out_free)),
        ]
        for p in self._procs:
            p.start()
        kind, labels, err = self.res_q.get(timeout=300)          # the receiver loads the models
        if err or not labels:
            raise RuntimeError(err or "no repair model could be loaded")
        self.models = {k: SimpleNamespace(label=v) for k, v in labels.items()}

    # ------------------------------------------------------------------ control
    def _push_settings(self):
        if self.processes:
            d = asdict(self.settings)
            self.send_q.put(("settings", d))
            self.recv_q.put(("settings", d))

    def update(self, **kw):
        with self.lock:
            for k, v in kw.items():
                if hasattr(self.settings, k):
                    cur = getattr(self.settings, k)
                    setattr(self.settings, k, type(cur)(v) if not isinstance(cur, bool) else bool(v))
            s = self.settings
            s.loss = min(0.6, max(0.0, s.loss))
            s.burst = min(16.0, max(1.0, s.burst))
            if s.baseline not in self.baselines:
                s.baseline = "none"
            if s.model not in self.models:
                s.model = next(iter(self.models))
            self._push_settings()
        return asdict(self.settings)

    def set_source(self, spec: str, kind: str, label: str):
        src = Source(spec, kind, label) if kind != "network" else NetSource(spec)   # IOError if unopenable
        with self.lock:
            old, self.source = self.source, src
            self._gen += 1
            self.stats.clear()
            self.source_error = ""
            if self.processes:
                self.send_q.put(("reset", self._gen))
                self.recv_q.put(("reset", self._gen))
            else:
                self.pipe.reset()
        if not self.processes:
            for q in (self.frames, self.net_q):
                while not q.empty():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break
        if old:
            old.close()
        return {"label": src.label, "kind": src.kind, "size": src.size, "fps": round(src.out_fps, 1),
                "frames": src.n}

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=2)
        self.subscribers.append(q)
        return q

    def unsubscribe(self, q):
        if q in self.subscribers:
            self.subscribers.remove(q)

    # ------------------------------------------------------------------ stages
    def _reader(self):
        while True:
            src = self.source
            if src is None or src.kind == "network" or self.settings.paused:
                time.sleep(0.05)
                continue
            gen = self._gen
            t0 = time.perf_counter()
            img = src.read()
            if img is None:
                self.source_error = f"{src.label}: stream ended or stalled"
                time.sleep(0.2)
                continue
            frame = letterbox(img)
            if gen != self._gen:
                continue
            if self.processes:
                try:
                    slot = self.in_free.get(timeout=1.0)  # waits while every slot is in flight: backpressure
                except queue.Empty:
                    continue
                self.ring_in[slot] = frame
                self.send_q.put(("frame", gen, slot))
            else:
                try:
                    self.frames.put((gen, frame), timeout=1.0)
                except queue.Full:
                    continue
            if src.kind in ("file", "url"):         # play at the source's own frame rate (live streams
                                                    # already arrive at that rate, so this only waits if early)
                dt = 1.0 / src.out_fps - (time.perf_counter() - t0)
                if dt > 0:
                    time.sleep(dt)

    def _dispatch(self):
        """processes=True: results from the receiver process -> latest frame, subscribers, stats, feedback."""
        while True:
            msg = self.res_q.get()
            if msg[0] == "error":
                self.source_error = msg[1]
                continue
            if msg[0] != "out":
                continue
            _, gen, oslot, stat, missing, net = msg
            comp = None
            if gen == self._gen:
                comp = self.ring_out[oslot].copy()
            self.out_free.put(oslot)
            if comp is None:
                continue
            if net and self.rx is not None:
                self.rx.report_missing(set(missing))   # loss-aware refresh feedback to the remote sender
            self._publish(comp, stat)

    def _publish(self, comp: np.ndarray, stat: dict):
        self.latest = comp
        for q in list(self.subscribers):
            if q.full():
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
            q.put_nowait(comp)
        self.stats.append(stat)

    def _sender(self):
        while True:
            try:
                gen, frame = self.frames.get(timeout=0.5)
            except queue.Empty:
                continue
            if gen != self._gen:
                continue
            try:
                cur = self._send(frame)
                self.net_q.put((gen, cur), timeout=2.0)
            except queue.Full:
                pass
            except Exception as e:                   # keep the app alive; show the error in the UI
                self.source_error = f"sender error: {e!r}"
                time.sleep(0.1)

    def _worker(self):
        while True:
            try:
                gen, cur = self.net_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if gen != self._gen:
                continue
            try:
                self._recv(cur)
            except Exception as e:
                self.source_error = f"receiver error: {e!r}"
                time.sleep(0.1)

    def _net_loop(self):
        """Receive real UDP packets from another machine and feed them into the receiver stage."""
        rx = self.rx
        while True:
            src = self.source
            on = src is not None and src.kind == "network"
            s = self.settings
            key = (s.loss, s.burst) if on else None
            if key != self._net_key:                  # the loss slider acts on arrival for network streams
                rx.loss_model = (make_loss("ge" if s.burst > 1.01 else "bernoulli", s.loss, s.burst,
                                           seed=int(time.time())) if on else None)
                self._net_key = key
            frames = rx.poll()
            if rx.ssrc is not None and rx.ssrc != self._net_ssrc and rx.peer:
                self._net_ssrc = rx.ssrc              # a sender just started: show its stream
                try:
                    self.set_source(rx.peer[0], "network", "")
                except Exception as e:
                    self.net_error = repr(e)
                continue
            if not on or s.paused:
                continue
            gen = self._gen
            for fid, st in frames:
                n = st.n_pkts if st else 0
                if n:
                    self._avg_pkts = 0.9 * self._avg_pkts + 0.1 * n if self._avg_pkts else n
                else:                                 # nothing at all arrived for this frame
                    n = round(self._avg_pkts)
                got = len(st.media) if st else 0
                nbytes = sum(len(p) + HEADER_SIZE for p in st.media.values()) if st else 0
                cur = dict(fid=fid, st=st, pkts=n, lost=max(0, n - got), bytes=nbytes,
                           t0=time.perf_counter(), net=True)
                try:                                  # receiver stage behind: drop rather than lag
                    if self.processes:
                        self.recv_q.put_nowait(("frame", gen, None, cur))
                    else:
                        self.net_q.put_nowait((gen, cur))
                except queue.Full:
                    pass

    def _snapshot(self) -> Settings:
        with self.lock:
            return Settings(**asdict(self.settings))

    # ---- in-process steps (processes=False; also what tools/res_bench.py times)
    def _process(self, orig: np.ndarray):
        """Single-threaded path (profiling / tests): send then receive."""
        self._recv(self._send(orig))

    def _send(self, orig: np.ndarray) -> dict:
        cur = self.pipe.send(orig, self._snapshot())
        cur["orig"] = orig
        return cur

    def _recv(self, cur: dict):
        for comp, stat, missing in self.pipe.recv(cur, cur.get("orig"), self._snapshot()):
            if cur.get("net") and self.rx is not None:
                self.rx.report_missing(set(missing))
            self._publish(comp, stat)

    # ------------------------------------------------------------------ stats
    def state(self) -> dict:
        st = list(self.stats)
        recent = [r for r in st if r["t"] > time.time() - 1.5]
        win = st[-30:]
        agg = {}
        if win:
            sent = sum(r["pkts"] for r in win)
            dur = max(1e-3, win[-1]["t"] - win[0]["t"]) if len(win) > 1 else 1.0
            withloss = [r["infer_ms"] for r in win if r["tiles_lost"]]
            pb = [r["psnr_b"] for r in win if r["psnr_b"] is not None]
            pp = [r["psnr_p"] for r in win if r["psnr_p"] is not None]
            agg = dict(
                measured_loss=sum(r["lost"] for r in win) / max(1, sent),
                pkts_per_frame=sent / len(win),
                tiles_lost=win[-1]["tiles_lost"],
                tiles_lost_avg=float(np.mean([r["tiles_lost"] for r in win])),
                psnr_b=float(np.mean(pb)) if pb else None,      # None: live stream, no original to compare
                psnr_p=float(np.mean(pp)) if pp else None,
                infer_ms=float(np.median(withloss)) if withloss else 0.0,
                fps=(len(recent) - 1) / max(1e-3, recent[-1]["t"] - recent[0]["t"]) if len(recent) > 2 else 0.0,
                mbps=sum(r["bytes"] for r in win) * 8 / dur / 1e6 if len(win) > 1 else 0.0,
            )
        src = self.source
        rx = self.rx
        hist = [r for r in st[-240:] if r["psnr_p"] is not None]
        return dict(
            settings=asdict(self.settings),
            models=list(self.models), baselines=list(self.baselines),
            model_info={k: getattr(v, "label", "") for k, v in self.models.items()},
            source=None if src is None else dict(label=src.label, kind=src.kind, size=src.size,
                                                 fps=round(src.out_fps, 1)),
            error=self.source_error, res=[WIDTH, HEIGHT], tiles=N_TILES, agg=agg,
            hist_b=[round(r["psnr_b"], 2) for r in hist],
            hist_p=[round(r["psnr_p"], 2) for r in hist],
            hist_loss=[round(r["tiles_lost"] / N_TILES, 3) for r in st[-240:]],
            net=dict(listening=rx is not None, port=self.udp_port, ips=self.ips, error=self.net_error,
                     sender=rx.peer[0] if rx is not None and rx.peer else None,
                     active=rx is not None and time.time() - rx.last_rx < 2.0),
        )
