"""Whose voice is it? Speaker embeddings compared with the caller's own voice (learned from their first answer).

Backend 1 (default): 3D-Speaker CAM++ (Apache-2.0, 7.2M params, ONNX via sherpa-onnx, ~10 ms per call).
  Held-out AUC same-caller vs other voice 0.855 (Resemblyzer 0.819); caller ≈0.49 vs others ≈0.20.
Backend 2 (fallback): Resemblyzer GE2E (Apache-2.0).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from .config import EXT_DIR

CAMPP = EXT_DIR / "spk" / "campp.onnx"
_ex = None
_res = None
BACKEND = "campp" if CAMPP.exists() else "resemblyzer"


def _campp(x: np.ndarray) -> np.ndarray:
    global _ex
    if _ex is None:
        import sherpa_onnx
        _ex = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(CAMPP), num_threads=1))
    s = _ex.create_stream()
    s.accept_waveform(16000, x.astype(np.float32) / 32768)
    s.input_finished()
    e = np.array(_ex.compute(s), dtype=np.float32)
    return e / (np.linalg.norm(e) + 1e-9)


def _resemblyzer(x: np.ndarray) -> np.ndarray:
    global _res
    if _res is None:
        warnings.filterwarnings("ignore")
        from resemblyzer import VoiceEncoder
        _res = VoiceEncoder("cpu", verbose=False)
    return _res.embed_utterance(x.astype(np.float32) / 32768)


def embed(audio_i16: np.ndarray) -> np.ndarray | None:
    if len(audio_i16) < 16000 * 0.25:
        return None
    try:
        return _campp(audio_i16) if BACKEND == "campp" else _resemblyzer(audio_i16)
    except Exception:
        return None
