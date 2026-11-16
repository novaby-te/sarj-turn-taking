"""Whose voice is it? A small speaker encoder (Resemblyzer, GE2E d-vector, ~17 MB, CPU, ~5 ms per call) compares
what we hear now with the caller's own voice, learned from their first answer (U0).
TV voices and people across the room score low; the caller scores high, even when shouting."""
from __future__ import annotations

import warnings

import numpy as np

_enc = None


def _encoder():
    global _enc
    if _enc is None:
        warnings.filterwarnings("ignore")
        from resemblyzer import VoiceEncoder
        _enc = VoiceEncoder("cpu", verbose=False)
    return _enc


def embed(audio_i16: np.ndarray) -> np.ndarray | None:
    if len(audio_i16) < 16000 * 0.25:
        return None
    try:
        return _encoder().embed_utterance(audio_i16.astype(np.float32) / 32768)
    except Exception:
        return None
