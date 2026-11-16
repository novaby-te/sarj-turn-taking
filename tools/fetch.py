"""Parallel, resumable download (many HTTP range requests at once). Used for model weights.

    .venv/bin/python -m tools.fetch URL OUTFILE [--parts 16]
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import urllib.request


def size_of(url):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers["Content-Length"]), r.geturl()


def part(url, start, end, path, done):
    have = os.path.getsize(path) if os.path.exists(path) else 0
    while start + have <= end:
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start + have}-{end}", "User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=60) as r, open(path, "ab") as f:
                while True:
                    b = r.read(1 << 16)
                    if not b:
                        break
                    f.write(b)
                    done[0] += len(b)
            have = os.path.getsize(path)
        except Exception:
            time.sleep(2)
            have = os.path.getsize(path) if os.path.exists(path) else 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--parts", type=int, default=16)
    a = ap.parse_args(argv)
    n, final = size_of(a.url)
    k = a.parts
    bounds = [(i * n // k, (i + 1) * n // k - 1) for i in range(k)]
    done = [0]
    ths = []
    for i, (s, e) in enumerate(bounds):
        p = f"{a.out}.part{i}"
        done[0] += os.path.getsize(p) if os.path.exists(p) else 0
        t = threading.Thread(target=part, args=(final, s, e, p, done), daemon=True)
        t.start()
        ths.append(t)
    t0 = time.time()
    while any(t.is_alive() for t in ths):
        time.sleep(5)
        print(f"{done[0] / 1e6:8.1f} / {n / 1e6:.1f} MB  ({done[0] / max(1, time.time() - t0) / 1e6:.2f} MB/s avg)", flush=True)
    with open(a.out, "wb") as f:
        for i in range(k):
            p = f"{a.out}.part{i}"
            with open(p, "rb") as g:
                f.write(g.read())
            os.remove(p)
    assert os.path.getsize(a.out) == n, "size mismatch"
    print("done", a.out)


if __name__ == "__main__":
    sys.exit(main())
