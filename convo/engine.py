"""A full, human-like phone conversation built on the turn-taking model.

    caller audio ─► TurnTaker (our competition model)
                      ├─ KEEP   («إيه»، كحة، تلفزيون)       → the agent carries on
                      ├─ STOP   (a real interruption)        → the agent stops mid-word and listens
                      └─ TAKE_TURN (the caller has finished)  → Whisper reads the whole turn
                                                                 → the brain (local LLM) writes a reply
                                                                 → the voice speaks it, sentence by sentence
    After an interruption the brain knows exactly what the agent managed to say and what it did not, so it
    answers the question first and then picks its point back up («وزي ما كنت أقول…»).
"""
from __future__ import annotations

import asyncio
import base64
import io
import random
import time
import wave

import numpy as np

from turn import lexicon, policy
from turn.asr import transcribe

from . import brain, voice
from .scenarios import SCENARIOS, as_male, system_prompt

SILENCE_NUDGE_MS = 7000      # caller silent this long after the agent finished → a gentle «معي؟»
FILLER_AFTER_S = 0.9         # if the first sentence is not ready by then, say «طيب…» like a person would


def wav_b64(x: np.ndarray) -> str:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(x.astype("<i2").tobytes())
    return base64.b64encode(b.getvalue()).decode()


class Conversation:
    def __init__(self, send, scenario: str = "payment", engine: str = "edge"):
        self.send = send                         # async fn(dict) → browser
        self.engine = engine
        self.sc = SCENARIOS[scenario] if engine == "edge" else as_male(SCENARIOS[scenario])
        self.tt = policy.TurnTaker()
        self.tt.on_scenario_start()
        self.history = [{"role": "system", "content": system_prompt(self.sc)}]
        self.state = "idle"                      # speaking | listening | thinking | ended
        self.chunks: dict[int, dict] = {}        # audio chunks sent for the current agent turn
        self.next_id = 0
        self.turn_text = ""                      # what the agent is saying in this turn (planned so far)
        self.task = None                         # the running reply (brain + voice)
        self.interrupted_at = None               # caller audio start of the interruption
        self.remaining = ""                      # what the agent had not said yet when interrupted
        self.listen_since = None
        self.turn_start_ms = None
        self.ms = 0
        self.seen_trace = 0
        self.stats = []
        self.t_eot = None
        self.ending = False
        self.phase = "greeting"                  # greeting → point → free conversation
        self.stop_ms = None
        self.planned_done = True

    # ------------------------------------------------------------------ speaking
    def synth(self, text: str) -> asyncio.Task:
        return asyncio.create_task(voice.tts(text, self.sc["gender"], self.engine))

    async def say_chunk(self, text: str, filler: bool = False, audio: asyncio.Task | None = None):
        x, used = await (audio or self.synth(text))
        if self.state not in ("speaking", "thinking"):
            return
        cid = self.next_id
        self.next_id += 1
        self.chunks[cid] = {"text": text, "n": len(x), "filler": filler, "played": False}
        self.browser_idle = False
        if self.state == "thinking" or not self.tt.agent_active:
            self.state = "speaking"
            self.turn_text = text
            self.tt.on_agent_start(text)
            if self.t_eot is not None:
                self.stats.append({"kind": "reply", "ms": round((time.time() - self.t_eot) * 1000)})
                await self.send({"type": "stat", "kind": "reply", "ms": self.stats[-1]["ms"]})
                self.t_eot = None
            await self.send({"type": "status", "state": "speaking"})
        else:
            self.turn_text += " " + text
            self.tt.agent_text = self.turn_text
        if not filler and getattr(self, "t_answer", None) is not None:
            self.stats.append({"kind": "answer", "ms": round((time.time() - self.t_answer) * 1000)})
            await self.send({"type": "stat", "kind": "answer", "ms": self.stats[-1]["ms"]})
            self.t_answer = None
        await self.send({"type": "say", "id": cid, "wav": wav_b64(x), "text": text, "voice": used})

    async def speak_lines(self, lines: list[str]):
        self.state = "thinking"
        self.planned_done = False
        tasks = [self.synth(line) for line in lines]          # synthesise all lines at once, speak in order
        for line, t in zip(lines, tasks):
            if self.state not in ("speaking", "thinking"):
                return
            await self.say_chunk(line, audio=t)
        self.planned_done = True
        await self.on_idle(check=True)

    async def reply(self, user_text: str, interrupted: bool):
        """Brain → voice, pipelined: sentence 1 is spoken while sentence 2 is being written."""
        self.state = "thinking"
        self.planned_done = False
        await self.send({"type": "status", "state": "thinking"})
        note = ""
        if interrupted and self.remaining:
            note = (f"(ملاحظة لك: العميل قاطعك قبل ما تقول: «{self.remaining}». رد على كلامه أول، "
                    f"وبعدين ارجع للنقطة إذا لسا مهمة.) ")
        msgs = list(self.history) + [{"role": "user", "content": note + user_text}]
        if not user_text.startswith("("):
            self.history.append({"role": "user", "content": user_text})
        q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def produce():
            try:
                for s in brain.sentences(msgs):
                    loop.call_soon_threadsafe(q.put_nowait, s)
            except Exception as e:  # brain down → say so, do not hang
                loop.call_soon_threadsafe(q.put_nowait, f"__ERR__{e}")
            loop.call_soon_threadsafe(q.put_nowait, None)
        fut = loop.run_in_executor(None, produce)
        said, filler_done, t0 = [], False, time.time()
        pending: list = []                                  # (sentence, synthesis task) in order

        async def flush(block: bool):
            while pending and (block or pending[0][1].done()):
                s, t = pending.pop(0)
                if self.state not in ("speaking", "thinking"):
                    return
                await self.say_chunk(s, audio=t)
        while True:
            try:
                s = await asyncio.wait_for(q.get(), timeout=max(0.05, FILLER_AFTER_S - (time.time() - t0)) if not filler_done else 30)
            except asyncio.TimeoutError:
                if not filler_done:
                    filler_done = True
                    await self.say_chunk(random.choice(voice.FILLERS[:3]), filler=True)
                continue
            if s is None:
                await flush(True)
                break
            if s.startswith("__ERR__"):
                await self.send({"type": "error", "text": "العقل (النموذج اللغوي) ما رد: " + s[7:]})
                break
            if self.state not in ("speaking", "thinking"):
                break                                      # interrupted while still writing: drop the rest
            if filler_done and said == [] and s.startswith(("طيب", "إيه نعم", "أبشر")) and len(s) < 12:
                continue
            said.append(s)
            pending.append((s, self.synth(s)))
            await flush(len(said) == 1)                     # the first sentence goes out as soon as it is ready
        await fut
        self.planned_done = True
        if self.state == "thinking":                       # nothing was said at all
            self.state = "listening"
            self.listen_since = self.ms
        if any("مع السلامة" in s for s in said):
            self.ending = True
        await self.on_idle(check=True)

    # ------------------------------------------------------------------ browser events
    async def on_played(self, cid: int):
        if cid in self.chunks:
            self.chunks[cid]["played"] = True

    async def on_idle(self, check: bool = False):
        """The browser played everything queued. If the brain is done too, the agent's turn is over."""
        if check and not self.browser_idle:
            return
        if self.state == "speaking" and self.planned_done and all(c["played"] for c in self.chunks.values()):
            spoken = " ".join(c["text"] for c in self.chunks.values() if not c["filler"])
            if spoken:
                self.history.append({"role": "assistant", "content": spoken})
            self.chunks.clear()
            self.tt.on_agent_end()
            self.state = "listening"
            self.listen_since = self.ms
            self.turn_start_ms = None
            await self.send({"type": "status", "state": "listening"})
            if self.ending:
                self.state = "ended"
                await self.send({"type": "status", "state": "ended"})

    async def on_stopped(self, cid: int, played: int):
        """The browser faded the agent out. Work out exactly what the caller heard."""
        heard, rest = [], []
        for i in sorted(self.chunks):
            c = self.chunks[i]
            if c["filler"]:
                continue
            if c["played"]:
                heard.append(c["text"])
            elif i == cid and c["n"]:
                w = c["text"].split()
                k = int(len(w) * min(1.0, played / c["n"]))
                heard.append(" ".join(w[:k]))
                rest.append(" ".join(w[k:]))
            else:
                rest.append(c["text"])
        said = " ".join(h for h in heard if h).strip()
        self.remaining = " ".join(r for r in rest if r).strip()
        if said:
            self.history.append({"role": "assistant", "content": said + "…"})
        self.chunks.clear()

    # ------------------------------------------------------------------ every 20 ms of audio
    async def on_frame(self, agent: np.ndarray, caller: np.ndarray):
        self.ms += 20
        d = self.tt.on_frame(agent, caller)
        dec = (d["decision"] if isinstance(d, dict) else d) if d else None
        while self.seen_trace < len(self.tt.trace):          # show the caller's acknowledgements as they happen
            r = self.tt.trace[self.seen_trace]
            self.seen_trace += 1
            if r["kind"] == "candidate_end" and r.get("text") and self.state == "speaking":
                await self.send({"type": "chat", "role": "caller", "text": r["text"], "note": "تأييد — كمّل"})
            if r["kind"] == "asr" and self.state in ("speaking", "listening"):
                await self.send({"type": "hearing", "text": r.get("text", ""), "why": r.get("why", "")})
        if dec == "STOP" and self.state in ("speaking", "thinking"):
            c = self.tt.cand
            self.interrupted_at = c["start"] if c else self.ms - 600
            self.stats.append({"kind": "stop", "ms": self.ms - self.interrupted_at})
            self.state = "listening"
            if self.task and not self.task.done():
                self.task.cancel()
            self.stop_ms = self.ms
            await self.send({"type": "stop", "latency": self.ms - self.interrupted_at, "why": self.tt.trace[-1].get("why", "")})
            await self.send({"type": "status", "state": "listening"})
            self.tt.on_agent_end()
            self.listen_since = self.ms
        elif dec == "TAKE_TURN" and self.state == "listening":
            start = self.interrupted_at if self.interrupted_at is not None else (self.tt.turn["start"] if self.tt.turn else self.ms - 3000)
            await self.caller_finished(start)
        elif self.state == "listening" and self.interrupted_at is not None and self.tt.turn is None \
                and self.ms - self.stop_ms > 900 and self.tt.vad.prob < 0.3:
            await self.caller_finished(self.interrupted_at)     # a short «لحظة» that ended before we stopped
        elif self.state == "listening" and self.listen_since is not None and not self.tt.turn \
                and self.ms - self.listen_since > SILENCE_NUDGE_MS and len(self.history) > 1:
            self.listen_since = self.ms
            self.task = asyncio.create_task(self.reply("(العميل ساكت ما رد. تأكد بلطف إنه معك)", False))

    async def caller_finished(self, start_ms: float):
        audio = self.tt._audio_since(max(start_ms - 150, self.ms - 29000))
        interrupted = self.interrupted_at is not None
        self.interrupted_at = None
        self.t_eot = self.t_answer = time.time()
        r = await asyncio.to_thread(transcribe, audio, False)
        text = r.get("text", "").strip()
        a = lexicon.analyse(text)
        if not text or a["hallucination"] or not a["words"]:
            self.tt.turn = None
            return                                            # noise, not words: keep listening
        await self.send({"type": "chat", "role": "caller", "text": text, "note": "قاطع" if interrupted else ""})
        self.tt.turn = None
        if self.task and not self.task.done():
            self.task.cancel()
        if self.phase == "greeting":                          # answer to «معي الأستاذ؟» → the agent makes its point
            self.phase = "talk"
            self.history.append({"role": "user", "content": text})
            self.state = "thinking"
            self.planned_done = False
            self.task = asyncio.create_task(self._first_point())
            return
        self.task = asyncio.create_task(self.reply(text, interrupted))

    async def _first_point(self):
        parts = [p.strip() + "،" for p in self.sc["first_point"].split("،") if p.strip()]
        parts[-1] = parts[-1].rstrip("،")
        lines, cur = [], ""
        for p in parts:                                        # group into natural phrases of ~8-14 words
            cur = (cur + " " + p).strip()
            if len(cur.split()) >= 8:
                lines.append(cur)
                cur = ""
        if cur:
            lines.append(cur)
        await self.send({"type": "status", "state": "speaking"})
        await self.speak_lines(lines)

    async def start(self):
        self.state = "thinking"
        self.planned_done = False
        await self.speak_lines(self.sc["opening"])
