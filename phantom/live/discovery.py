"""Finding the receiver on the local network, and learning its stream resolution.

Kept free of phantom.config on purpose: the resolution is fixed when phantom.config is imported, so the sender
first asks the receiver which resolution it runs, then sets PHANTOM_RES to match, then imports the rest.

    sender -> broadcast or unicast  DISCOVER
    receiver -> sender              ANNOUNCE + its frame height ("PHANTOM-FRAMES!720")
"""
from __future__ import annotations

import ipaddress
import socket
import time

DISCOVER = b"PHANTOM-FRAMES?"      # sender -> broadcast: is there a receiver on this network?
ANNOUNCE = b"PHANTOM-FRAMES!"      # receiver -> sender: yes, here (followed by its resolution, e.g. b"720")


def announce(height: int) -> bytes:
    return ANNOUNCE + str(height).encode()


def parse_announce(data: bytes) -> int | None:
    """The receiver's resolution from its reply (360 for an older receiver that sends none), else None."""
    if not data.startswith(ANNOUNCE):
        return None
    tail = data[len(ANNOUNCE):]
    return int(tail) if tail.isdigit() else 360


def local_ips() -> list[str]:
    """This machine's LAN addresses (the default route's first)."""
    ips = []
    try:                                    # the address of the default route (no packet is sent)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        ips += socket.gethostbyname_ex(socket.gethostname())[2]
    except OSError:
        pass
    seen, out = set(), []
    for ip in ips:
        if ip not in seen and not ip.startswith(("127.", "169.254.")):
            seen.add(ip)
            out.append(ip)
    return out


def discover(port: int, target: str | None = None, tries: int = 4, wait: float = 1.0) -> tuple[str, int] | None:
    """Ask for a receiver: broadcast on every local network, or ask `target` directly.
    Returns (receiver IP, its resolution), or None when nobody answers."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.settimeout(wait)
    if target:
        targets = {target}
    else:
        targets = {"255.255.255.255"}
        for ip in local_ips():          # subnet broadcasts too (assume /24; covers home Wi-Fi and phone hotspots)
            targets.add(str(ipaddress.ip_network(f"{ip}/24", strict=False).broadcast_address))
    try:
        for _ in range(tries):
            for t in targets:
                try:
                    s.sendto(DISCOVER, (t, port))
                except OSError:
                    pass
            t_end = time.time() + wait
            while time.time() < t_end:
                try:
                    data, addr = s.recvfrom(64)
                except (socket.timeout, ConnectionResetError, OSError):
                    break
                res = parse_announce(data)
                if res is not None:
                    return addr[0], res
    finally:
        s.close()
    return None
