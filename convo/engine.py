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
import websockets

from turn import lexicon, policy
from turn.asr import transcribe

from . import brain, voice
from .live_score import LiveScore
from .live_voice import LiveVoice
from .scenarios import SCENARIOS, as_male, system_prompt

SILENCE_NUDGE_MS = 7000      # caller silent this long after the agent finished → a gentle «معي؟»
ECHO_MARGIN = 8.0            # the caller must be this many times (≈9 dB) louder than the agent's echo
FILLER_AFTER_S = 0.9         # if the first sentence is not ready by then, say «طيب…» like a person would
INTERRUPT_SETTLE_MS = 600    # the caller has been quiet this long after interrupting: answer now
STRAY_SOUND_MS = 200         # less voiced sound than this after a stop never completes a turn
LIVE_CHUNK_BYTES = 9600      # 200 ms of 24 kHz audio: enough to ride out small network gaps
LIVE_ERRORS = (OSError, KeyError, websockets.WebSocketException)


def wav_b64(x: np.ndarray) -> str:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(voice.RATE)
        w.writeframes(x.astype("<i2").tobytes())
    return base64.b64encode(b.getvalue()).decode()


class Conversation:
    def __init__(self, send, scenario: str = "payment", engine: str = "edge"):
        self.events: list = []
        async def _send(m, _orig=send):
            if m.get("type") in ("say", "pcm", "stop", "status", "chat"):
                self.events.append({"t": self.ms, **{k: v for k, v in m.items() if k not in ("wav", "pcm")}})
            await _orig(m)
        self.send = _send                        # async fn(dict) → browser (and logged)
        self.engine = engine
        self.reply_engine = "edge" if engine == "live" else engine   # used when the live voice is unavailable
        self.live = None
        self.sc = SCENARIOS[scenario] if engine in ("edge", "live", "eleven") else as_male(SCENARIOS[scenario])
        self.tt = policy.TurnTaker()
        self.tt.on_scenario_start()
        self.score = LiveScore()
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
        self.agent_pow: list = []
        self.coupling: list = []
        self.gated = 0
        self.echo_flags: list = []
        self.rec_a: list = []
        self.rec_c: list = []
        self.phase = "greeting"                  # greeting → point → free conversation
        self.stop_ms = None
        self.planned_done = True

    # ------------------------------------------------------------------ speaking
    def synth(self, text: str, engine: str | None = None) -> asyncio.Task:
        return asyncio.create_task(voice.tts(text, self.sc["gender"], engine or self.engine))

    async def say_chunk(self, text: str, filler: bool = False, audio: asyncio.Task | None = None):
        x, used = await (audio or self.synth(text))
        if self.state not in ("speaking", "thinking"):
            return
        cid = await self._open_chunk(text, round(len(x) * 16000 / voice.RATE), filler)
        await self.send({"type": "say", "id": cid, "wav": wav_b64(x), "text": text, "voice": used})

    async def _play_pcm(self, pcm: bytes, text: str):
        cid = await self._open_chunk(text, round(len(pcm) / 2 * 16000 / voice.RATE), False)
        await self.send({"type": "pcm", "id": cid, "pcm": base64.b64encode(pcm).decode(), "text": text, "voice": "live"})

    async def _open_chunk(self, text: str, n: int, filler: bool) -> int:
        """Register audio that is about to play. `n` counts samples at 16 kHz, as the browser reports them."""
        cid = self.next_id
        self.next_id += 1
        self.chunks[cid] = {"text": text, "n": n, "filler": filler, "played": False}
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
        return cid

    async def _connect_live(self):
        told = [f"«{' '.join(self.sc['opening'])}»", f"«{self.sc['first_point']}»"]
        past = [f"{'أنت' if m['role'] == 'assistant' else 'العميل'}: {m['content']}" for m in self.history if m["role"] != "system"]
        recap = "بدأت المكالمة بقولك: " + " ثم ".join(told) + ("\nسجل المكالمة حتى الآن:\n" + "\n".join(past) if past else "")
        live = LiveVoice(system_prompt(self.sc) + "\n\n" + recap, self.sc["gender"])
        await live.connect()
        self.live = live

    async def _connect_quietly(self):
        try:
            await self._connect_live()
        except LIVE_ERRORS as e:
            print(f"live voice unavailable: {type(e).__name__}", flush=True)

    async def _stream_live(self, text: str) -> bool:
        """Speak the reply while it is generated. False means nothing was said, so the text path should answer."""
        buffered, words, spoken, started = bytearray(), "", "", False
        try:
            if self.live is None or self.live.closed:
                await self._connect_live()
            async for pcm, piece in self.live.speak(text):
                if self.state not in ("speaking", "thinking"):
                    return True
                buffered += pcm
                words += piece
                spoken += piece
                if len(buffered) >= LIVE_CHUNK_BYTES:
                    await self._play_pcm(bytes(buffered), words)
                    buffered, words, started = bytearray(), "", True
            if buffered and self.state in ("speaking", "thinking"):
                await self._play_pcm(bytes(buffered), words)
                started = True
        except LIVE_ERRORS as e:
            print(f"live voice failed: {type(e).__name__}", flush=True)
            self.shutdown()
            return started
        finally:
            if self.live is not None:
                self.live.cancel()
        if "مع السلامة" in spoken:
            self.ending = True
        return started

    async def send_score(self):
        await self.send({"type": "score", **self.score.snapshot()})

    def shutdown(self):
        if self.live is not None:
            asyncio.create_task(self.live.close())
            self.live = None

    async def speak_lines(self, lines: list[str]):
        self.state = "thinking"
        self.planned_done = False
        if self.engine == "live":
            lines = [" ".join(lines)]              # one recording: natural flow, no gaps
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
        if self.engine == "live" and await self._stream_live(note + user_text):
            await self._finish_reply()
            return
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
                    filler = random.choice(voice.FILLERS[:3])
                    await self.say_chunk(filler, filler=True, audio=self.synth(filler, self.reply_engine))
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
            pending.append((s, self.synth(s, self.reply_engine)))
            await flush(len(said) == 1)                     # the first sentence goes out as soon as it is ready
        await fut
        if any("مع السلامة" in s for s in said):
            self.ending = True
        await self._finish_reply()

    async def _finish_reply(self):
        self.planned_done = True
        if self.state == "thinking":                       # nothing was said at all
            self.state = "listening"
            self.listen_since = self.ms
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
        if getattr(self, "resume_pending", False):
            self.resume_pending = False
            await self._resume()

    async def _resume(self):
        """The stop was a backchannel after all («إيه، تمام»): carry on from where the caller heard us stop,
        like a person would, without asking the brain."""
        rest, self.remaining, self.interrupted_at = self.remaining, "", None
        self.tt.turn = None
        if not rest:
            return
        if self.engine == "live":
            await self.send({"type": "hearing", "text": "", "why": "كان تأييد: يكمّل كلامه"})
            self.stats.append({"kind": "resume", "ms": self.ms - (self.stop_ms or self.ms)})
            self.task = asyncio.create_task(self.reply(f"(العميل كان يوافقك بس. كمّل كلامك من النقطة اللي وقفت عندها: «{rest}»)", False))
            return
        words = rest.split()
        lines = [" ".join(words[i:i + 12]) for i in range(0, len(words), 12)]
        await self.send({"type": "hearing", "text": "", "why": "كان تأييد: يكمّل كلامه"})
        self.stats.append({"kind": "resume", "ms": self.ms - (self.stop_ms or self.ms)})
        self.task = asyncio.create_task(self.speak_lines(lines))

    # ------------------------------------------------------------------ echo gate (laptop speakers)
    def echo_gate(self, agent: np.ndarray, caller: np.ndarray) -> np.ndarray:
        """Learn how much of the agent's own voice comes back into the mic (speaker→mic coupling), and while the
        agent is audible treat the mic as the caller only when it is clearly louder than that echo."""
        pa = float(np.mean(agent.astype(np.float64) ** 2))
        pc = float(np.mean(caller.astype(np.float64) ** 2))
        self.agent_pow.append(pa)
        if len(self.agent_pow) > 25:                     # 500 ms of agent history: delay + room reverb
            self.agent_pow.pop(0)
        ref = max(self.agent_pow)
        if pa > 1e5:                                     # agent clearly playing (> about -50 dBFS)
            self.coupling.append(pc / pa)
            if len(self.coupling) > 400:
                self.coupling.pop(0)
        echo_only = False
        if ref >= 1e5 and len(self.coupling) >= 25:
            k = float(np.percentile(self.coupling, 60))  # typical echo level relative to the agent
            echo_only = pc < ECHO_MARGIN * k * ref
        self.echo_flags.append(echo_only)
        if len(self.echo_flags) > 1500:
            self.echo_flags.pop(0)
        self.gated += echo_only
        return caller

    # ------------------------------------------------------------------ every 20 ms of audio
    async def on_frame(self, agent: np.ndarray, caller: np.ndarray):
        self.ms += 20
        self.rec_a.append(agent.copy())
        self.rec_c.append(caller.copy())
        caller = self.echo_gate(agent, caller)
        d = self.tt.on_frame(agent, caller)
        dec = (d["decision"] if isinstance(d, dict) else d) if d else None
        while self.seen_trace < len(self.tt.trace):          # show the caller's acknowledgements as they happen
            r = self.tt.trace[self.seen_trace]
            self.seen_trace += 1
            if self._is_echo(r):
                continue                                  # the agent's own voice coming back: not the caller
            if r["kind"] == "candidate_end" and r.get("text") and self.state == "speaking":
                await self.send({"type": "chat", "role": "caller", "text": r["text"], "note": "تأييد — كمّل"})
                self.score.kept(self.ms, r["text"])
                await self.send_score()
            if r["kind"] == "asr" and self.state in ("speaking", "listening"):
                await self.send({"type": "hearing", "text": r.get("text", ""), "why": r.get("why", "")})
        if dec == "STOP" and self.state in ("speaking", "thinking") and not self._real_words():
            # demo guard: no clear caller words (the agent's own voice from the speakers, noise) → keep talking
            self.tt.stopped, self.tt.cand = False, None
            await self.send({"type": "hearing", "text": "", "why": "تجاهل صوت بدون كلام واضح (صدى/ضجيج)"})
            dec = None
        if dec == "STOP" and self.state in ("speaking", "thinking"):
            c = self.tt.cand
            self.interrupted_at = c["start"] if c else self.ms - 600
            self.stats.append({"kind": "stop", "ms": self.ms - self.interrupted_at})
            self.score.stopped(self.interrupted_at, self.ms, (c or {}).get("text", ""))
            await self.send_score()
            self.state = "listening"
            if self.task and not self.task.done():
                self.task.cancel()
            self.stop_ms = self.ms
            await self.send({"type": "stop", "latency": self.ms - self.interrupted_at, "why": self.tt.trace[-1].get("why", "")})
            await self.send({"type": "status", "state": "listening"})
            self.tt.on_agent_end()
            self.listen_since = self.ms
        elif dec == "RESUME" and self.state == "listening" and self.interrupted_at is not None:
            self.score.took_back(self.ms)
            await self.send_score()
            if self.chunks:                                   # the browser has not reported where it stopped yet
                self.resume_pending = True
            else:
                await self._resume()
        elif dec == "TAKE_TURN" and self.state == "listening":
            start = self.interrupted_at if self.interrupted_at is not None else (self.tt.turn["start"] if self.tt.turn else self.ms - 3000)
            await self.caller_finished(start)
        elif self.state == "listening" and self.interrupted_at is not None and self._interruption_over():
            await self.caller_finished(self.interrupted_at)     # a short «كم قيمتها؟» that ended around the stop
        elif self.state == "listening" and self.listen_since is not None and not self.tt.turn \
                and self.ms - self.listen_since > SILENCE_NUDGE_MS and len(self.history) > 1:
            self.listen_since = self.ms
            self.task = asyncio.create_task(self.reply("(العميل ساكت ما رد. تأكد بلطف إنه معك)", False))

    def _interruption_over(self) -> bool:
        """The caller interrupted briefly and went quiet. A stray sound after the stop must not keep the agent waiting."""
        turn = self.tt.turn
        if turn is not None and turn["voiced"] >= STRAY_SOUND_MS:
            return False                                  # a real turn: the end-of-turn logic will answer it
        last_sound = max(self.stop_ms, turn["last"]) if turn is not None else self.stop_ms
        return self.ms - last_sound > INTERRUPT_SETTLE_MS and self.tt.vad.prob < 0.3

    def _is_echo(self, r: dict) -> bool:
        text = r.get("text", "")
        a = lexicon.analyse(text, self.turn_text) if text else None
        if a and a["echoed"] / max(1, a["n"]) >= 0.5:
            return True
        n = 50                                            # the last second of audio
        recent = self.echo_flags[-n:]
        return bool(recent) and sum(1 for e in recent if not e) / len(recent) < 0.3

    def _real_words(self) -> bool:
        """Stop only for clear words from the caller that are not the agent's own sentence coming back."""
        c = self.tt.cand
        f = (c or {}).get("feats")
        if not f or f["n"] < 1 or f["echo_ratio"] >= 0.5:
            return False
        n = max(1, int((self.ms - c["start"]) // 20))
        recent = self.echo_flags[-n:]
        if recent and sum(1 for e in recent if not e) / len(recent) < 0.3:
            return False                                 # mostly the agent's own voice coming back
        return True

    async def caller_finished(self, start_ms: float):
        audio = self.tt._audio_since(max(start_ms - 150, self.ms - 29000))
        interrupted = self.interrupted_at is not None
        self.interrupted_at = None
        self.t_eot = self.t_answer = time.time()
        r = await asyncio.to_thread(transcribe, audio, False)
        text = r.get("text", "").strip()
        last_agent = next((m["content"] for m in reversed(self.history) if m["role"] == "assistant"), "")
        a = lexicon.analyse(text, last_agent + " " + self.turn_text)
        own_voice = a["echoed"] / max(1, a["n"]) >= 0.6 and (not interrupted or a["n"] >= 5)   # a short question may reuse the agent's words
        if not text or a["hallucination"] or not a["words"] or own_voice:
            self.tt.turn = None
            if interrupted:                                   # the agent stopped for the caller: never leave them in silence
                self.task = asyncio.create_task(self.reply("(قاطعك العميل بس ما سمعت كلامه بوضوح. اعتذر باختصار واطلب منه يعيد)", False))
            return                                            # noise, not words: keep listening
        only_ack = a["bid"] == 0 and a["content"] == 0 and not a["repeated_ambig"] and not a["question"]
        if interrupted and only_ack and self.remaining:
            self.score.was_acknowledgement(text)
            await self.send_score()
            await self.send({"type": "chat", "role": "caller", "text": text, "note": "تأييد — يكمّل"})
            await self._resume()
            return
        if interrupted:
            self.score.heard(text)
            await self.send_score()
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
        if self.engine == "live":
            await self.send({"type": "status", "state": "speaking"})
            await self.speak_lines([self.sc["first_point"]])
            return
        parts = [p.strip() + "،" for p in self.sc["first_point"].split("،") if p.strip()]
        parts[-1] = parts[-1].rstrip("،")
        lines, cur = [], ""
        for p in parts:                                        # whole phrases: short pieces sound choppy
            cur = (cur + " " + p).strip()
            if len(cur.split()) >= 14:
                lines.append(cur)
                cur = ""
        if cur:
            lines.append(cur)
        await self.send({"type": "status", "state": "speaking"})
        await self.speak_lines(lines)

    async def start(self):
        self.state = "thinking"
        self.planned_done = False
        if self.engine == "live":
            asyncio.create_task(self._connect_quietly())
        await self.speak_lines(self.sc["opening"])


def save_session(conv: "Conversation", folder, d=None):
    """Keep the demo call (both channels + what the model decided) for diagnosis."""
    import json
    from pathlib import Path
    from ttbench.dataset import write_wav_stereo
    if not conv.rec_a:
        return None
    d = Path(d) if d else Path(folder) / time.strftime("%m%d-%H%M%S")
    d.mkdir(parents=True, exist_ok=True)
    write_wav_stereo(d / "call.wav", np.stack([np.concatenate(conv.rec_a), np.concatenate(conv.rec_c)], 1))
    (d / "trace.json").write_text(json.dumps({"trace": [{k: v for k, v in r.items() if k != "f"} for r in conv.tt.trace],
                                              "history": conv.history, "stats": conv.stats, "gated_frames": conv.gated, "score": conv.score.snapshot()},
                                             ensure_ascii=False, indent=1), encoding="utf-8")
    (d / "events.json").write_text(json.dumps(conv.events, ensure_ascii=False), encoding="utf-8")
    return d
