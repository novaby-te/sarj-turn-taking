"""The agent's voice. Two engines:
  edge  - Microsoft neural Saudi voices (natural, needs internet, ~1-3 s to the first sound)
  say   - macOS «Majed» (offline, fast, more robotic) - automatic fallback
Short fillers («طيب»، «إيه نعم») are synthesised once and cached on disk so they play instantly while the
brain is still thinking, the way a person says «طيب…» before answering.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import subprocess
import tempfile
from pathlib import Path

import numpy as np

CACHE = Path.home() / "Downloads" / "sarj-ext-data" / "voice-cache"
VOICES = {"f": "ar-SA-ZariyahNeural", "m": "ar-SA-HamedNeural"}
FILLERS = ["طيب", "إيه نعم", "أبشر", "طيب، لحظة"]


def _key(engine, voice, text):
    return hashlib.sha1(f"{engine}|{voice}|{text}".encode()).hexdigest()


async def _edge(text: str, voice: str, timeout: float = 8.0) -> np.ndarray:
    import edge_tts
    import soundfile as sf
    from scipy import signal

    async def go():
        b = b""
        async for ch in edge_tts.Communicate(text, voice, rate="+5%").stream():
            if ch["type"] == "audio":
                b += ch["data"]
        return b
    data = await asyncio.wait_for(go(), timeout)
    x, sr = sf.read(io.BytesIO(data), dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    return (np.clip(signal.resample_poly(x, 16000, sr), -1, 1) * 32767 * 0.9).astype(np.int16)


_best = None


def best_say_voice() -> str:
    """The best Arabic macOS voice installed: Premium > Enhanced > the default compact «Majed»."""
    global _best
    if _best is None:
        lines = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout.splitlines()
        import re
        ar = [m.group(1).strip() for l in lines if (m := re.match(r"^(.*?)\s+ar_\w+", l))]
        _best = next((v for tag in ("Premium", "Enhanced") for v in ar if tag in v), ar[0] if ar else "Majed")
    return _best


def _say(text: str) -> np.ndarray:
    import soundfile as sf
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "x.wav"
        subprocess.run(["say", "-v", best_say_voice(), "-r", "185", "-o", str(out), "--file-format=WAVE",
                        "--data-format=LEI16@16000", text], check=True, capture_output=True)
        x, _ = sf.read(out, dtype="int16")
    return x


async def tts(text: str, gender: str = "f", engine: str = "edge") -> tuple[np.ndarray, str]:
    """Returns (16 kHz int16 audio, engine actually used). Never raises."""
    import sys
    mac = sys.platform == "darwin"
    if engine == "say" and not mac:                  # no macOS «say» on Windows / Linux: use Edge
        engine = "edge"
    voice = VOICES.get(gender, VOICES["f"]) if engine == "edge" else best_say_voice()
    k = _key(engine, voice, text)
    f = CACHE / f"{k}.npy"
    if f.exists():
        return np.load(f), engine
    if engine == "edge":
        try:
            x = await _edge(text, voice)
            CACHE.mkdir(parents=True, exist_ok=True)
            if len(text) < 40:
                np.save(f, x)
            return x, "edge"
        except Exception:
            if not mac:
                return np.zeros(1600, np.int16), "none"
    x = await asyncio.to_thread(_say, text)
    if len(text) < 40:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(f, x)
    return x, "say"


async def warm_fillers(gender: str = "f", engine: str = "edge"):
    for t in FILLERS:
        await tts(t, gender, engine)
