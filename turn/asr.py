"""Speech recognition of the caller's words, on-device (Whisper through Apple MLX).

The prompt primes Whisper with the short Gulf words that matter here, so «إيه / هيه / مم / لحظة» come out as
words instead of being dropped. A disk cache (keyed by the exact audio) makes repeated simulator runs fast
while still charging the latency that was MEASURED the first time.
"""
from __future__ import annotations

import hashlib
import threading
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
_LOCK = threading.Lock()      # one Whisper call at a time (live demo: the turn-taker and the dialogue share it)
# Speed-up: Whisper always encodes 30 s of audio even for a 1 s «لحظة». With CTX_S > 0 we encode only a short
# window (the encoder's positional embedding is sliced), which cuts the latency. 0 = standard Whisper.
CTX_S = float(os.environ.get("TT_ASR_CTX", "10"))
_cache_on = os.environ.get("TT_ASR_CACHE", "1" if os.environ.get("TT_SIM") == "1" else "0") == "1"


def _key(audio: np.ndarray) -> str:
    h = hashlib.sha1(audio.tobytes())
    h.update(Path(MODEL).name.encode() + PROMPT.encode() + b"v2" + (f"ctx{CTX_S}".encode() if CTX_S else b""))
    return h.hexdigest()


def warmup():
    transcribe(np.zeros(16000, np.int16), use_cache=False)


def transcribe(audio_i16: np.ndarray, use_cache: bool = True) -> dict:
    """Returns {"text": str, "ms": compute time}. Never raises. Thread-safe."""
    with _LOCK:
        return _transcribe(audio_i16, use_cache)


def _transcribe(audio_i16: np.ndarray, use_cache: bool = True) -> dict:
    if MODEL == "none":                       # acoustic-only mode (no words)
        return {"text": "", "no_speech": 1.0, "ms": 0.0}
    key = _key(audio_i16)
    path = CACHE_DIR / key[:2] / f"{key}.json"
    if use_cache and _cache_on and path.exists():
        return json.loads(path.read_text(encoding="utf-8")) | {"cached": True}
    import mlx_whisper
    t0 = time.perf_counter()
    audio = audio_i16.astype(np.float32) / 32768.0
    if CTX_S and len(audio) < (CTX_S - 0.5) * 16000:
        out = _short(audio, t0)
        if out is not None:
            if use_cache and _cache_on:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
            return out
    try:
        r = mlx_whisper.transcribe(audio, path_or_hf_repo=MODEL, language="ar", temperature=0.0,
                                   condition_on_previous_text=False, initial_prompt=PROMPT or None,
                                   no_speech_threshold=0.6, logprob_threshold=-1.0,
                                   compression_ratio_threshold=2.4, verbose=None,
                                   sample_len=40)
        segs = [s for s in r.get("segments", []) if s.get("no_speech_prob", 0) < 0.8]
        text = " ".join(s["text"].strip() for s in segs).strip()
        nsp = min([s.get("no_speech_prob", 1.0) for s in r.get("segments", [])] or [1.0])
        lp = [s.get("avg_logprob", -5.0) for s in segs]
        logprob = float(np.mean(lp)) if lp else -5.0
    except Exception as e:  # pragma: no cover
        print("ASR error", e)
        return {"text": "", "no_speech": 1.0, "ms": round((time.perf_counter() - t0) * 1000, 1), "error": repr(e)}
    out = {"text": text, "no_speech": round(float(nsp), 3), "logprob": round(logprob, 3),
           "ms": round((time.perf_counter() - t0) * 1000, 1)}
    if use_cache and _cache_on:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


_patched = False


def _short(audio: np.ndarray, t0: float) -> dict | None:
    """Decode with a shortened audio context (whisper.cpp's audio_ctx idea, in MLX)."""
    global _patched
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_whisper import whisper as W
        from mlx_whisper.audio import log_mel_spectrogram
        from mlx_whisper.decoding import DecodingOptions, decode
        from mlx_whisper.transcribe import ModelHolder
        if not _patched:
            def enc_call(self, x):
                x = nn.gelu(self.conv1(x))
                x = nn.gelu(self.conv2(x))
                x = x + self._positional_embedding[: x.shape[1]]
                for block in self.blocks:
                    x, _, _ = block(x)
                return self.ln_post(x)
            W.AudioEncoder.__call__ = enc_call
            _patched = True
        model = ModelHolder.get_model(MODEL, mx.float16)
        n_frames = int(CTX_S * 100)
        mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels)
        if mel.shape[0] < n_frames:
            mel = mx.pad(mel, [(0, n_frames - mel.shape[0]), (0, 0)])
        mel = mel[:n_frames].astype(mx.float16)
        opts = DecodingOptions(language="ar", temperature=0.0, sample_len=40, prompt=PROMPT or None,
                               without_timestamps=True, fp16=True)
        r = decode(model, mel, opts)
        text = r.text.strip() if r.no_speech_prob < 0.8 else ""
        return {"text": text, "no_speech": round(float(r.no_speech_prob), 3), "logprob": round(float(r.avg_logprob), 3),
                "ms": round((time.perf_counter() - t0) * 1000, 1), "ctx": CTX_S}
    except Exception as e:  # fall back to standard Whisper
        print("short-context ASR failed:", e)
        return None
