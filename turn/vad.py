"""Silero VAD (v5, ONNX, ~2 MB, MIT licence) fed with 20 ms frames. Returns P(speech) for the latest audio."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

MODEL = Path(__file__).resolve().parents[1] / "models" / "silero_vad.onnx"
CHUNK = 512        # Silero needs 512-sample chunks at 16 kHz
CONTEXT = 64


class SileroVAD:
    _session = None

    def __init__(self):
        if SileroVAD._session is None:
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            SileroVAD._session = ort.InferenceSession(str(MODEL), so, providers=["CPUExecutionProvider"])
        self.s = SileroVAD._session
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), np.float32)
        self.context = np.zeros((1, CONTEXT), np.float32)
        self.buf = np.zeros(0, np.float32)
        self.prob = 0.0

    def push(self, frame_i16: np.ndarray) -> float:
        self.buf = np.concatenate([self.buf, frame_i16.astype(np.float32) / 32768.0])
        while len(self.buf) >= CHUNK:
            chunk = self.buf[:CHUNK][None, :]
            self.buf = self.buf[CHUNK:]
            x = np.concatenate([self.context, chunk], axis=1)
            out, self.state = self.s.run(None, {"input": x, "state": self.state, "sr": np.array(16000, np.int64)})
            self.context = chunk[:, -CONTEXT:]
            self.prob = float(out[0, 0])
        return self.prob
