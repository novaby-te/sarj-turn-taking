"""Speech recognition of the caller's words, on-device (Whisper through Apple MLX).

The prompt primes Whisper with the short Gulf words that matter here, so «إيه / هيه / مم / لحظة» come out as
words instead of being dropped. A disk cache (keyed by the exact audio) makes repeated simulator runs fast
while still charging the latency that was MEASURED the first time.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parents[1]
EXT = Path.home() / "Downloads" / "sarj-ext-data"
MODEL = os.environ.get("TT_ASR_MODEL", str(EXT / "whisper-large-v3-turbo-q4"))
PROMPT = os.environ.get("TT_ASR_PROMPT", "إيه، أها، مم، هيه، زين، طيب، لا لا، لحظة، اسمع، خلاص.")
CACHE_DIR = KIT / ".cache" / "asr"
_cache_on = os.environ.get("TT_ASR_CACHE", "1" if os.environ.get("TT_SIM") == "1" else "0") == "1"


def _key(audio: np.ndarray) -> str:
    h = hashlib.sha1(audio.tobytes())
    h.update(Path(MODEL).name.encode() + PROMPT.encode())
    return h.hexdigest()


def warmup():
    transcribe(np.zeros(16000, np.int16), use_cache=False)


def transcribe(audio_i16: np.ndarray, use_cache: bool = True) -> dict:
    """Returns {"text": str, "ms": compute time}. Never raises."""
    if MODEL == "none":                       # acoustic-only mode (no words)
        return {"text": "", "no_speech": 1.0, "ms": 0.0}
    key = _key(audio_i16)
    path = CACHE_DIR / key[:2] / f"{key}.json"
    if use_cache and _cache_on and path.exists():
        return json.loads(path.read_text(encoding="utf-8")) | {"cached": True}
    import mlx_whisper
    t0 = time.perf_counter()
    audio = audio_i16.astype(np.float32) / 32768.0
    try:
        r = mlx_whisper.transcribe(audio, path_or_hf_repo=MODEL, language="ar", temperature=0.0,
                                   condition_on_previous_text=False, initial_prompt=PROMPT or None,
                                   no_speech_threshold=0.6, logprob_threshold=-1.0,
                                   compression_ratio_threshold=2.4, verbose=None,
                                   sample_len=40)
        segs = [s for s in r.get("segments", []) if s.get("no_speech_prob", 0) < 0.8]
        text = " ".join(s["text"].strip() for s in segs).strip()
        nsp = min([s.get("no_speech_prob", 1.0) for s in r.get("segments", [])] or [1.0])
    except Exception as e:  # pragma: no cover
        print("ASR error", e)
        return {"text": "", "no_speech": 1.0, "ms": round((time.perf_counter() - t0) * 1000, 1), "error": repr(e)}
    out = {"text": text, "no_speech": round(float(nsp), 3), "ms": round((time.perf_counter() - t0) * 1000, 1)}
    if use_cache and _cache_on:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out
