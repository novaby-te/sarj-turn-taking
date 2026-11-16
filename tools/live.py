"""Live microphone test: the agent talks through your speakers, you interrupt (or just say «إيه»), and you see
what the system decided, what it heard and how fast. You can label each try; labelled tries are saved as new
calls in data/live (same format as the kit) so the next `tools.loop` scores the model on YOUR voice too.

    .venv/bin/python -m tools.live          # opens http://127.0.0.1:8770
"""
from __future__ import annotations

import asyncio
import io
import json
import sys
import threading
import time
import wave
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
from ttbench.dataset import read_wav_stereo, write_wav_stereo  # noqa: E402

HTTP_PORT, WS_PORT = 8770, 8771
LIVE = KIT / "data" / "live"


def wav_bytes(x: np.ndarray) -> bytes:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(x.astype("<i2").tobytes())
    return b.getvalue()


def dev_calls():
    out = []
    for f in sorted((KIT / "data" / "dev" / "labels").glob("*.json")):
        lab = json.loads(f.read_text())
        u = max(lab["agent_utterances"], key=lambda u: u["end_ms"] - u["start_ms"])
        out.append({"sid": lab["scenario_id"], "text": u["text"], "start": u["start_ms"], "end": u["end_ms"]})
    return out


def tts(text: str) -> np.ndarray:
    import edge_tts
    import soundfile as sf
    from scipy import signal

    async def go():
        c = edge_tts.Communicate(text, "ar-SA-ZariyahNeural")
        b = b""
        async for ch in c.stream():
            if ch["type"] == "audio":
                b += ch["data"]
        return b
    x, sr = sf.read(io.BytesIO(asyncio.run(go())), dtype="float32")
    x = signal.resample_poly(x, 16000, sr)
    return (np.clip(x, -1, 1) * 32767 * 0.8).astype(np.int16)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            return self._send((KIT / "tools" / "live.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/calls":
            return self._send(json.dumps(dev_calls(), ensure_ascii=False).encode(), "application/json")
        if u.path == "/agent":
            sid = q["sid"][0]
            c = next(c for c in dev_calls() if c["sid"] == sid)
            st = read_wav_stereo(KIT / "data" / "dev" / "audio" / f"{sid}.wav")
            return self._send(wav_bytes(st[c["start"] * 16:c["end"] * 16, 0]), "audio/wav")
        if u.path == "/tts":
            text = q["text"][0][:400]
            return self._send(wav_bytes(tts(text)), "audio/wav")
        self.send_error(404)


async def ws_handler(ws):
    from turn import policy
    tt = None
    frames_a, frames_c = [], []
    agent_text = ""
    t_start = None
    decisions = []
    sent_trace = 0
    async for raw in ws:
        if isinstance(raw, bytes):
            if tt is None:
                continue
            fr = np.frombuffer(raw, dtype="<i2").reshape(-1, 2)
            frames_a.append(fr[:, 0].copy())
            frames_c.append(fr[:, 1].copy())
            d = tt.on_frame(fr[:, 0], fr[:, 1])
            ms = len(frames_a) * 20
            if d:
                dec = d["decision"] if isinstance(d, dict) else d
                decisions.append((ms, dec))
                await ws.send(json.dumps({"type": "decision", "decision": dec, "t": ms}))
            if len(frames_a) % 5 == 0:
                lvl = 20 * np.log10(np.sqrt(np.mean(fr[:, 1].astype(np.float64) ** 2)) / 32768 + 1e-9)
                await ws.send(json.dumps({"type": "meter", "vad": round(tt.vad.prob, 2), "db": round(float(lvl), 1),
                                          "ref": round(tt.ref_db, 1), "t": ms}))
            while sent_trace < len(tt.trace):
                await ws.send(json.dumps({"type": "trace", **{k: v for k, v in tt.trace[sent_trace].items() if k != "f"}}, ensure_ascii=False))
                sent_trace += 1
            continue
        msg = json.loads(raw)
        if msg["type"] == "start":
            tt = policy.TurnTaker()
            tt.on_scenario_start()
            frames_a, frames_c, decisions, sent_trace = [], [], [], 0
            agent_text = msg.get("text", "")
            if msg.get("calibrate"):
                tt.calibrating = True
                await ws.send(json.dumps({"type": "calibrating"}))   # caller says «هلا، معك» first (like U0)
            else:
                tt.on_agent_start(agent_text)
                await ws.send(json.dumps({"type": "started"}))
            t_start = time.time()
        elif msg["type"] == "agent_start" and tt:
            tt.calibrating = False
            tt.on_agent_start(agent_text)
            await ws.send(json.dumps({"type": "started", "ref_db": round(tt.ref_db, 1),
                                      "voiceprint": tt.ref_emb is not None}))
        elif msg["type"] == "agent_end" and tt:
            tt.on_agent_end()
            agent_end_ms = len(frames_a) * 20
            await ws.send(json.dumps({"type": "agent_end", "t": agent_end_ms}))
        elif msg["type"] == "calibrate":
            pass
        elif msg["type"] == "save" and frames_a:
            sid = save_try(frames_a, frames_c, agent_text, msg["expected"], msg.get("agent_end_ms"), tt.trace)
            await ws.send(json.dumps({"type": "saved", "sid": sid}))


def save_try(fa, fc, agent_text, expected, agent_end_ms, trace):
    """Store a labelled try as a kit-format call. Onset/offset from the system's own voice detection;
    evident = onset + 500 ms (we do not know the exact moment, the label says so)."""
    LIVE.mkdir(parents=True, exist_ok=True)
    (LIVE / "audio").mkdir(exist_ok=True)
    (LIVE / "labels").mkdir(exist_ok=True)
    n = len(list((LIVE / "labels").glob("*.json"))) + 1
    sid = f"LV-{n:03d}"
    pre = np.zeros(16000, np.int16)       # 1 s of silence before, so the call starts like the others
    a = np.concatenate([pre, np.concatenate(fa)])
    c = np.concatenate([pre, np.concatenate(fc)])
    write_wav_stereo(LIVE / "audio" / f"{sid}.wav", np.stack([a, c], 1))
    cands = [t for t in trace if t["kind"] == "candidate"]
    onset = (cands[0]["t"] if cands else 1000) + 1000
    ends = [t["t"] for t in trace if t["kind"] in ("candidate_end", "STOP")]
    offset = max(onset + 300, (ends[-1] if ends else onset + 1200) + 1000)
    dur = len(a) // 16
    end_agent = (agent_end_ms or dur - 1000) + 1000
    ev = {"event_id": f"{sid}.e1", "kind": "interruption", "expected": expected, "onset_ms": onset, "offset_ms": min(offset, dur - 10)}
    if expected == "STOP":
        ev["evident_ms"] = onset + 500
        ev["deadline_ms"] = onset + 2000
    else:
        ev["window_end_ms"] = ev["offset_ms"] + 500
    lab = {"schema_version": "1.0", "scenario_id": sid, "split": "live", "category": "GI" if expected == "STOP" else "BC",
           "subtype": "live", "difficulty": 2, "dialect": "live-user", "elicitation": "naturalistic",
           "speakers": [{"speaker_id": "U", "role": "caller", "source": "live microphone"}],
           "acoustic": {"conditions": ["live-mic"]},
           "audio": {"file": f"{sid}.wav", "sample_rate": 16000, "channels": ["agent", "user"], "duration_ms": dur},
           "context": {"use_case": "live test", "language": "ar"},
           "agent_utterances": [{"utt_id": "A1", "start_ms": 1000, "end_ms": min(end_agent, dur - 10), "text": agent_text}],
           "user_segments": [], "events": [ev], "provenance": {"label_status": "user_labelled_live"}}
    (LIVE / "labels" / f"{sid}.json").write_text(json.dumps(lab, ensure_ascii=False, indent=1), encoding="utf-8")
    ids = sorted(p.stem for p in (LIVE / "labels").glob("*.json"))
    (LIVE / "manifest.json").write_text(json.dumps({"dataset": "live", "split": "live", "scenarios": [{"scenario_id": i} for i in ids],
                                                    "thresholds": {"T_full_ms": 300, "T_dead_ms": 1500, "E_full_ms": 600, "E_dead_ms": 2000, "E_miss_ms": 3000}},
                                                   ensure_ascii=False, indent=1))
    return sid


def main():
    from websockets.asyncio.server import serve
    from turn.asr import warmup
    print("loading Whisper ...", flush=True)
    warmup()
    srv = ThreadingHTTPServer(("127.0.0.1", HTTP_PORT), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    async def run():
        async with serve(ws_handler, "127.0.0.1", WS_PORT, max_size=2 ** 20):
            print(f"open http://127.0.0.1:{HTTP_PORT}", flush=True)
            webbrowser.open(f"http://127.0.0.1:{HTTP_PORT}")
            await asyncio.Future()
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
