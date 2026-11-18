"""End of turn from the VOICE (not the words): Pipecat Smart Turn v3.2 (BSD-2-Clause, 8 MB int8 ONNX, Whisper-tiny
encoder + linear head, 23 languages incl. Arabic). Given the caller's last <= 8 s, it returns P(the turn is complete):
a falling, finished intonation scores high; a held «اممم…» or a mid-list pause scores low.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import EXT_DIR

MODEL = EXT_DIR / "smart-turn" / "smart-turn-v3.2-cpu.onnx"
_sess = None
_fe = None


def available() -> bool:
    return MODEL.exists()


def p_complete(audio_i16: np.ndarray) -> float | None:
    global _sess, _fe
    if not available() or len(audio_i16) < 3200:
        return None
    try:
        if _sess is None:
            import onnxruntime as ort
            from transformers import WhisperFeatureExtractor
            so = ort.SessionOptions()
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            so.inter_op_num_threads = 1
            so.intra_op_num_threads = 2
            _sess = ort.InferenceSession(str(MODEL), sess_options=so, providers=["CPUExecutionProvider"])
            _fe = WhisperFeatureExtractor(chunk_length=8)
        x = audio_i16[-8 * 16000:].astype(np.float32) / 32768.0
        f = _fe(x, sampling_rate=16000, return_tensors="np", padding="max_length", max_length=8 * 16000,
                truncation=True, do_normalize=True).input_features.astype(np.float32)
        return float(_sess.run(None, {"input_features": f})[0].reshape(-1)[0])
    except Exception:
        return None
