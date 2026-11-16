"""Echo suppression using the clean agent channel as a reference.

On speakerphone the agent's own voice comes back on the caller's line (NV-D03). We learn, per call and online,
the delay and the per-band gain of that leak, and subtract its predicted energy from the caller's spectrum.
What is left ("residual") is what the caller (or their room) actually added.
"""
from __future__ import annotations

import numpy as np

NFFT = 512
NB = 24                     # frequency bands
HIST = 40                   # agent history kept: 40 frames = 800 ms (max echo delay)
EDGES = np.unique(np.geomspace(2, NFFT // 2, NB + 1).astype(int))
WIN = np.hanning(320)


def band_power(frame: np.ndarray) -> np.ndarray:
    spec = np.abs(np.fft.rfft(frame.astype(np.float64) / 32768 * WIN, NFFT)) ** 2
    return np.add.reduceat(spec, EDGES[:-1])[: len(EDGES) - 1] + 1e-12


class EchoSuppressor:
    def __init__(self):
        self.reset()

    def reset(self):
        nb = len(EDGES) - 1
        self.ahist = np.zeros((HIST, nb))
        self.ae = []                 # agent log energy (recent)
        self.ce = []                 # caller log energy (recent)
        self.lag = 0
        self.lag_corr = 0.0
        self.gain = np.zeros(nb)     # per band power gain agent->caller
        self.ratios = []
        self.n = 0

    def process(self, agent: np.ndarray, caller: np.ndarray, agent_active: bool) -> tuple[float, float]:
        """Returns (residual_db, echo_db): caller level after removing predicted echo, and the predicted echo level."""
        pa, pc = band_power(agent), band_power(caller)
        self.ahist = np.roll(self.ahist, 1, axis=0)
        self.ahist[0] = pa
        self.n += 1
        self.ae.append(np.log(pa.sum()))
        self.ce.append(np.log(pc.sum()))
        if len(self.ae) > 200:
            self.ae.pop(0)
            self.ce.pop(0)
        if self.n % 25 == 0 and len(self.ae) >= 150:
            self._update_lag()
        echo = np.zeros_like(pc)
        if self.lag_corr > 0.45:
            ref = self.ahist[self.lag]
            if agent_active and ref.sum() > 1e-7:
                r = pc / ref
                self.ratios.append(r)
                if len(self.ratios) > 150:
                    self.ratios.pop(0)
                if len(self.ratios) >= 20:
                    self.gain = np.percentile(np.array(self.ratios), 30, axis=0)
            echo = 2.0 * self.gain * ref
        resid = np.maximum(pc - echo, 1e-12)
        to_db = lambda p: 10 * np.log10(p.sum() / (np.sum(WIN ** 2) * NFFT / 2) + 1e-12)  # noqa: E731
        return float(to_db(resid)), float(to_db(echo + 1e-12))

    def _update_lag(self):
        a = np.array(self.ae[-150:])
        c = np.array(self.ce[-150:])
        best, bl = 0.0, 0
        for lag in range(0, HIST):
            x = a[: len(a) - lag] if lag else a
            y = c[lag:]
            if x.std() < 1e-6 or y.std() < 1e-6:
                continue
            r = float(np.corrcoef(x, y)[0, 1])
            if r > best:
                best, bl = r, lag
        # smooth: only move when clearly better
        if best > self.lag_corr * 0.9 or best > 0.6:
            self.lag, self.lag_corr = bl, best
        else:
            self.lag_corr *= 0.97
