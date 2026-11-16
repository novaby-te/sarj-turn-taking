"""The agent's brain: a local language model behind an OpenAI-style HTTP server (mlx_lm.server), in its own
process so it never competes with Whisper inside our process. Streams the reply sentence by sentence so the
voice can start on the first sentence while the rest is still being written.

Models (tested on this Mac, Apple M5 24 GB):
  gemma-3-12b-it-qat-4bit  - best Arabic of the ones that fit (default once downloaded)
  Qwen3-14B-MLX-4bit       - already in LM Studio; correct facts but stiffer, more formal Arabic
"""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

EXT = Path.home() / "Downloads" / "sarj-ext-data"
MODELS = {
    "gemma-12b": EXT / "gemma-3-12b-it-qat-4bit",
    "qwen3-14b": Path.home() / ".lmstudio" / "models" / "lmstudio-community" / "Qwen3-14B-MLX-4bit",
}
PORT = 8780
_proc = None


def available() -> dict:
    ok = {}
    for k, p in MODELS.items():
        idx = p / "model.safetensors.index.json"
        if idx.exists():
            need = set(json.loads(idx.read_text())["weight_map"].values())
            ok[k] = all((p / f).exists() for f in need)
    return ok


def default_model() -> str:
    av = available()
    return os.environ.get("TT_BRAIN") or next((k for k in ("gemma-12b", "qwen3-14b") if av.get(k)), "qwen3-14b")


def _up() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", PORT), timeout=0.3):
            return True
    except OSError:
        return False


def start(model: str | None = None) -> str:
    """Start (or restart with another model) the brain server. Returns the model name."""
    global _proc, _current
    model = model or default_model()
    if _proc and _proc.poll() is None and globals().get("_current") == model:
        return model
    stop()
    _proc = subprocess.Popen([sys.executable, "-m", "mlx_lm.server", "--model", str(MODELS[model]), "--port", str(PORT),
                              "--max-tokens", "160"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _current = model
    t0 = time.time()
    while time.time() - t0 < 180:                      # ready = it answers, not just that the port is open
        try:
            if _up():
                warm()
                return model
        except Exception:
            pass
        if _proc.poll() is not None:
            raise RuntimeError("brain server exited")
        time.sleep(0.5)
    raise RuntimeError("brain server did not start")


def stop():
    global _proc
    if _proc and _proc.poll() is None:
        _proc.terminate()
        try:
            _proc.wait(5)
        except subprocess.TimeoutExpired:
            _proc.kill()
    _proc = None


def warm():
    for _ in stream([{"role": "user", "content": "هلا"}], max_tokens=4):
        pass


def normalize(messages: list[dict]) -> list[dict]:
    """Strict chat templates (Gemma) need: one system message first, then user/assistant alternating, user first."""
    sys_txt = "\n".join(m["content"] for m in messages if m["role"] == "system")
    out: list[dict] = []
    for m in messages:
        if m["role"] == "system":
            continue
        if out and out[-1]["role"] == m["role"]:
            out[-1] = {"role": m["role"], "content": out[-1]["content"] + " " + m["content"]}
        else:
            out.append(dict(m))
    if not out or out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": "(بدأت المكالمة ورد العميل على الخط)"})
    if out[-1]["role"] != "user":
        out.append({"role": "user", "content": "(كمّل)"})
    if sys_txt:
        if _current_is_gemma():                 # Gemma has no system role: put it in front of the first user turn
            out[0] = {"role": "user", "content": sys_txt + "\n\n" + out[0]["content"]}
        else:
            out.insert(0, {"role": "system", "content": sys_txt})
    return out


def _current_is_gemma() -> bool:
    return str(globals().get("_current", "")).startswith("gemma")


def stream(messages: list[dict], max_tokens: int = 120, temperature: float = 0.6):
    """Yield text pieces as the model writes them."""
    messages = normalize(messages)
    if globals().get("_current", "").startswith("qwen3") and messages and messages[0]["role"] == "system":
        messages = [{"role": "system", "content": messages[0]["content"] + "\n/no_think"}] + messages[1:]
    body = json.dumps({"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body, {"Content-Type": "application/json"})
    in_think = False
    with urllib.request.urlopen(req, timeout=60) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            try:
                d = json.loads(line[5:])["choices"][0].get("delta", {}).get("content") or ""
            except (ValueError, KeyError, IndexError):
                continue
            if "<think>" in d:
                in_think = True
            if in_think:
                if "</think>" in d:
                    in_think = False
                    d = d.split("</think>", 1)[1]
                else:
                    continue
            if d:
                yield d


_SENT = re.compile(r"(.+?[\.\!\؟\?…]+|.{22,}?[،,])(\s+|$)", re.S)


def sentences(messages: list[dict]):
    """Yield complete, speakable sentences as soon as each one is finished."""
    buf = ""
    for piece in stream(messages):
        buf += piece
        while True:
            m = _SENT.match(buf)
            if not m:
                break
            s = clean(m.group(1))
            buf = buf[m.end():]
            if s:
                yield s
    if clean(buf):
        yield clean(buf)


def clean(s: str) -> str:
    s = re.sub(r"[\*\#\_\(\)\[\]\"«»“”]|<[^>]+>", " ", s)
    s = re.sub(r"[\U0001F300-\U0001FAFF☀-➿]", "", s)
    return re.sub(r"\s+", " ", s).strip()
