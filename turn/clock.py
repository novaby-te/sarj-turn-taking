"""Time and slow work, the same code in a live run and in the fast simulator.

Live (real harness / live mic): Async runs the work on a background thread; on_frame never blocks.
Simulation (TT_SIM=1): Async runs the work immediately but only reveals the result once the simulated
stream clock has advanced by the MEASURED compute time, so the simulator charges the true latency.
"""
from __future__ import annotations

import os
import queue
import threading
import time

SIM = os.environ.get("TT_SIM") == "1"
_now_ms = 0.0


def set_now(ms: float) -> None:
    global _now_ms
    _now_ms = ms


class Async:
    """One slow worker (e.g. speech recognition). submit() a job, poll() for finished results."""

    def __init__(self, fn):
        self.fn = fn
        self.pending = []          # sim: (ready_at_ms, tag, result)
        self.busy = False
        if not SIM:
            self.q_in: queue.Queue = queue.Queue(maxsize=1)
            self.q_out: queue.Queue = queue.Queue()
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            tag, args = self.q_in.get()
            t0 = time.perf_counter()
            try:
                res = self.fn(*args)
            except Exception as e:  # never kill the worker
                res = {"error": repr(e)}
            self.q_out.put((tag, res, (time.perf_counter() - t0) * 1000))
            self.busy = False

    def submit(self, tag, *args) -> bool:
        """Start a job unless one is already running. Returns True if it started."""
        if self.busy:
            return False
        self.busy = True
        if SIM:
            t0 = time.perf_counter()
            res = self.fn(*args)
            dt = (time.perf_counter() - t0) * 1000
            if isinstance(res, dict) and "ms" in res:      # cached result: charge the latency measured originally
                dt = max(dt, float(res["ms"]))
            self.pending.append((_now_ms + dt, tag, res, dt))
        else:
            self.q_in.put((tag, args))
        return True

    def poll(self):
        """Finished results: list of (tag, result, compute_ms)."""
        out = []
        if SIM:
            keep = []
            for ready, tag, res, dt in self.pending:
                (out if ready <= _now_ms else keep).append((ready, tag, res, dt))
            self.pending = keep
            if not keep:
                self.busy = False
            return [(t, r, d) for _, t, r, d in out]
        while True:
            try:
                out.append(self.q_out.get_nowait())
            except queue.Empty:
                return out
