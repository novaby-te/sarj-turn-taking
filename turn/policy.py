"""The turn-taking brain. Four layers, each one cheap to explain:

  1. WHO is making sound?   echo removed with the agent's clean channel, Silero VAD, and the level compared with
                            the caller's OWN voice, learned from their first answer (U0). TV, a child across the
                            room, a horn or the agent's echo are quieter than, or not, the caller's voice.
  2. WHAT did they say?     on-device Whisper on the caller's audio so far, re-read every ~250 ms.
  3. WHAT does it MEAN?     Gulf/Saudi lexicon: acknowledgement (إيه، أها، هيه) vs floor bid (لا، لحظة، بس، اسمع)
                            vs new content; words the caller repeats from the agent's own sentence = saying it along.
  4. DECIDE                 a learned score over those features (models/policy.json, trained on generated calls)
                            with transparent rules as the fallback; a sustained-loud-speech safety net.
  + END OF TURN             wait longer when the answer is unfinished (a filler, fewer digits than the agent asked for).

Every number below is a parameter the tuner (tools/tune.py) can move.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

from . import clock, lexicon
from .echo import EchoSuppressor
from .vad import SileroVAD

FRAME_MS = 20
VAD_TH = 0.5             # Silero probability that counts as speech
REL_DB = 10.0            # speech must be at most this far below the caller's own level (U0)
SNR_DB = 9.0             # ... and this far above the room's noise floor
STOP_REL_DB = 8.0        # a stop needs the candidate's loudness within this of U0 (rejects side speech, TV)
MIN_VOICED_MS = 100      # voiced audio before the first ASR read
ASR_EVERY_MS = 150       # re-read the growing segment this often
GAP_END_MS = 1300        # silence that closes a candidate
CONTENT_MAX_AGE_S = 3.0  # content heard only after this much continuous sound, and quiet = background talk
MIN_LOGPROB = -0.9       # content words count only if Whisper was reasonably sure
CONTENT_MIN = 2          # new-content words that make a floor bid
LONG_MS = 2000           # safety net: sustained loud speech this long -> stop even without words
LONG_REL_DB = 3.0
DEFAULT_REF_DB = -21.0   # caller level before U0 has been heard
AGENT_TAIL_MS = 200
EOT_BASE_MS = 600        # silence after a complete answer before TAKE_TURN
EOT_LONG_MS = 2500       # silence allowed when the answer is unfinished
NEVER_STOP = 0           # 1: only record features (used to build training data)
USE_SPEAKER = 1          # compare with the caller's own voiceprint
SPK_MIN = 0.40           # below this similarity it is someone else (TV, a child across the room)...
SPK_LOUD_DB = -2.0       # ...unless it is clearly louder than the caller usually is
HYB_TH = 0.2            # hybrid: below this learned probability a rule-based stop is vetoed
PROMOTE_TH = 0.95        # hybrid: above this learned probability the model may stop on its own (1.01 = never)
PROMOTE_REL_DB = 4.0     # ...only when the voice is about as loud as the caller's
USE_MODEL = 2            # 1: learned score (if models/policy.json exists), 0: rules only
MODEL_TH = 0.5

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "policy.json"
FEATURES = ["bid", "content", "bc", "filler", "echoed", "repeated_ambig", "ambig_alone", "question", "n",
            "voiced_s", "rel_db", "only_bc", "agent_q_prev", "since_agent_start_s", "first_word_bid", "spk_sim", "cand_age_s", "logprob", "echo_ratio"]


def _load_model():
    try:
        return json.loads(MODEL_PATH.read_text())
    except (OSError, ValueError):
        return None


class TurnTaker:
    def __init__(self, asr_fn=None):
        from .asr import transcribe
        from . import speaker
        self._speaker = speaker

        def job(audio):
            r = dict((asr_fn or transcribe)(audio))
            e = speaker.embed(audio[-16000 * 3:]) if USE_SPEAKER else None
            r["spk"] = e
            return r
        self.asr = clock.Async(job)
        self.vad = SileroVAD()
        self.echo = EchoSuppressor()
        self.model = _load_model() if USE_MODEL else None
        self.trace: list = []          # what happened and why (for the dashboard)
        self.reset_call()

    # ------------------------------------------------------------------ call / agent events
    def reset_call(self):
        self.vad.reset()
        self.echo.reset()
        self.ms = 0
        self.ref_levels: list[float] = []
        self.ref_audio: list[np.ndarray] = []
        self.ref_emb = None
        self.ref_emb_n = 0
        self.ref_db = DEFAULT_REF_DB
        self.floor = deque(maxlen=250)
        self.ring = deque(maxlen=1500)      # last 30 s of caller audio, 20 ms frames
        self.agent_active = False
        self.agent_end_ms = -1e9
        self.agent_text = ""
        self.agent_start_ms = 0
        self.prev_agent_question = False
        self.stopped = False
        self.cand = None
        self.cand_id = 0
        self.turn = None                     # caller turn while the agent is silent (end-of-turn track)
        self.prev_speech = False

    def on_scenario_start(self):
        self.reset_call()

    def on_agent_start(self, text: str):
        self.prev_agent_question = self.agent_text.strip().endswith(("؟", "?")) if self.agent_text else False
        self.agent_active, self.agent_text, self.stopped = True, text, False
        self.agent_start_ms = self.ms
        self.turn = None

    def on_agent_end(self):
        self.agent_active = False
        self.agent_end_ms = self.ms

    # ------------------------------------------------------------------ helpers
    def _floor_db(self) -> float:
        return float(np.percentile(self.floor, 20)) if len(self.floor) > 25 else -70.0

    def _audio_since(self, start_ms: float) -> np.ndarray:
        n = int(max(0, (self.ms - start_ms) // FRAME_MS))
        frames = list(self.ring)[-n:] if n else []
        return np.concatenate(frames) if frames else np.zeros(0, np.int16)

    def _log(self, kind: str, **kw):
        self.trace.append({"t": self.ms, "kind": kind, **kw})

    # ------------------------------------------------------------------ every 20 ms
    def on_frame(self, agent: np.ndarray, caller: np.ndarray):
        self.ms += FRAME_MS
        speaking = self.agent_active or (self.ms - self.agent_end_ms) <= AGENT_TAIL_MS
        raw_db = 20 * np.log10(np.sqrt(np.mean(caller.astype(np.float64) ** 2)) / 32768 + 1e-9)
        resid_db, echo_db = self.echo.process(agent, caller, self.agent_active)
        p = self.vad.push(caller)
        self.ring.append(caller.copy())
        if p < 0.3:
            self.floor.append(raw_db)
        floor = self._floor_db()
        level_ok = resid_db > self.ref_db - REL_DB and resid_db > floor + SNR_DB
        # opening a candidate needs the caller's level; once open, quieter syllables of the same word still count
        is_speech = p > VAD_TH and (level_ok or (self.cand is not None and resid_db > floor + SNR_DB - 3))

        # learn the caller's own voice level from what they say while the agent is silent (U0 first)
        if not speaking and p > 0.6 and raw_db > floor + 6:
            self.ref_levels.append(raw_db)
            if len(self.ref_levels) >= 8:
                self.ref_db = float(np.percentile(self.ref_levels[-150:], 80))
            if len(self.ref_audio) < 200:                       # up to 4 s of the caller's own voice
                self.ref_audio.append(caller.copy())
                n = len(self.ref_audio)
                if USE_SPEAKER and n >= 40 and n - self.ref_emb_n >= 25:
                    self.ref_emb = self._speaker.embed(np.concatenate(self.ref_audio))
                    self.ref_emb_n = n

        out = None
        for tag, res, ms in self.asr.poll():
            out = self._on_asr(tag, res, ms) or out
        if speaking:
            if not self.stopped:
                out = self._while_agent_speaks(is_speech, resid_db) or out
        else:
            self.cand = None
            out = self._end_of_turn(is_speech) or out
        self.prev_speech = is_speech
        return out

    # ------------------------------------------------------------------ core: should the agent stop?
    def _while_agent_speaks(self, is_speech: bool, db: float):
        c = self.cand
        if c is None:
            if not is_speech:
                return None
            self.cand_id += 1
            c = self.cand = {"id": self.cand_id, "start": self.ms - 120, "voiced": 0, "last": self.ms,
                             "levels": [], "sub": -1e9, "text": "", "feats": None}
            self._log("candidate", level_rel=round(db - self.ref_db, 1))
        if is_speech:
            c["voiced"] += FRAME_MS
            c["last"] = self.ms
            c["levels"].append(db)
        elif self.ms - c["last"] > GAP_END_MS:
            self._log("candidate_end", text=c["text"])
            self.cand = None
            return None
        pause_start = self.prev_speech and not is_speech
        due = self.ms - c["sub"] >= ASR_EVERY_MS
        ready = c["voiced"] >= MIN_VOICED_MS or (pause_start and c["voiced"] >= 60)
        if ready and (due or pause_start) and self.ms - c["last"] < 400:
            if self.asr.submit(("cand", c["id"], self.ms), self._audio_since(c["start"])):
                c["sub"] = self.ms
        # safety net: long, loud, sustained speech even if the words are not understood yet
        rel = self._cand_rel(c)
        not_caller = c.get("spk") is not None and c["spk"] < SPK_MIN
        if c["voiced"] >= LONG_MS and rel >= -LONG_REL_DB and not not_caller and not (c["feats"] and c["feats"]["only_bc"]):
            return self._stop("sustained loud speech", c)
        return None

    def _cand_rel(self, c) -> float:
        return float(np.percentile(c["levels"], 80)) - self.ref_db if c["levels"] else -99.0

    def _features(self, a: dict, c: dict) -> dict:
        return {"bid": a["bid"], "content": a["content"], "bc": a["bc"], "filler": a["filler"], "echoed": a["echoed"],
                "repeated_ambig": int(a["repeated_ambig"]), "ambig_alone": int(a["ambig"] > 0 and not a["repeated_ambig"]),
                "question": int(a["question"]), "n": a["n"], "voiced_s": c["voiced"] / 1000, "rel_db": self._cand_rel(c),
                "only_bc": int(a["n"] > 0 and a["bid"] == 0 and a["content"] == 0 and not a["repeated_ambig"]),
                "agent_q_prev": int(self.prev_agent_question),
                "since_agent_start_s": (self.ms - self.agent_start_ms) / 1000,
                "first_word_bid": int(a["first_bid"] == 0), "spk_sim": c.get("spk", 0.6),
                "cand_age_s": (self.ms - c["start"]) / 1000,
                "logprob": c.get("logprob", -1.0), "echo_ratio": a["echoed"] / max(1, a["n"])}

    def score(self, f: dict) -> tuple[float, str]:
        """P(the caller wants the floor). Learned weights if available, else transparent rules."""
        if f["rel_db"] < -STOP_REL_DB:
            return 0.0, "too quiet for the caller (side speech / TV)"
        if f["spk_sim"] < SPK_MIN and f["rel_db"] < SPK_LOUD_DB:
            return 0.05, f"not the caller's voice (similarity {f['spk_sim']:.2f})"
        rp, why = self._rules(f)
        if not self.model:
            return rp, why
        z = self.model["bias"] + sum(self.model["w"][k] * (f[k] - self.model["mu"][k]) / (self.model["sd"][k] or 1.0)
                                     for k in FEATURES)
        mp = float(1 / (1 + np.exp(-z)))
        if USE_MODEL == 1:                       # learned score alone
            return mp, f"model p={mp:.2f}"
        # USE_MODEL == 2 (hybrid): the rules propose, the learned model can veto an unlikely stop
        if rp >= MODEL_TH and mp < HYB_TH:
            return 0.25, f"{why}, but the model doubts it (p={mp:.2f})"
        if rp < MODEL_TH and mp >= PROMOTE_TH and f["n"] >= 1 and f["rel_db"] > -PROMOTE_REL_DB:
            return mp, f"model is sure (p={mp:.2f}) although the words are unclear"
        return rp, f"{why} (model p={mp:.2f})"

    def _rules(self, f: dict) -> tuple[float, str]:
        if f["bid"] >= 1:
            return 1.0, "floor-bid word"
        if f["repeated_ambig"]:
            return 0.9, "«خلاص/طيب» with more after it"
        if f["content"] >= CONTENT_MIN:
            if f["logprob"] < MIN_LOGPROB:
                return 0.3, f"words unclear (confidence {f['logprob']:.2f})"
            if f["cand_age_s"] > CONTENT_MAX_AGE_S and f["rel_db"] < -4:
                return 0.2, "long, quiet, continuous talk without a floor bid: background"
            return 0.85, f"{f['content']} content words"
        if f["question"] and f["n"] >= 2:
            return 0.8, "question"
        return 0.1, "acknowledgement / filler" if f["n"] else "no words"

    def _on_asr(self, tag, res, ms):
        if tag[0] == "turn":
            if self.turn is not None and tag[1] == self.turn["id"]:
                self.turn["text"] = res.get("text", "")
                self._log("asr_turn", text=self.turn["text"], ms=round(ms))
            return None
        c = self.cand
        if c is None or tag[1] != c["id"]:
            return None
        text = res.get("text", "")
        c["logprob"] = res.get("logprob", -1.0)
        if res.get("spk") is not None and self.ref_emb is not None:
            c["spk"] = float(np.dot(res["spk"], self.ref_emb))
        a = lexicon.analyse(text, self.agent_text)
        f = self._features(a, c)
        c["text"], c["feats"] = text, f
        p, why = self.score(f)
        self._log("asr", text=text, ms=round(ms), p=round(p, 2), why=why, rel_db=round(f["rel_db"], 1), f=f,
                  cand_start=c["start"])
        speaking = self.agent_active or (self.ms - self.agent_end_ms) <= AGENT_TAIL_MS
        if speaking and not self.stopped and p >= MODEL_TH and not NEVER_STOP:
            return self._stop(why, c)
        return None

    def _stop(self, why, c):
        if NEVER_STOP:
            return None
        self.stopped = True
        self._log("STOP", why=why, text=c["text"])
        label = "interruption"
        return {"decision": "STOP", "label": label, "confidence": 0.8}

    # ------------------------------------------------------------------ optional: end of turn
    def _end_of_turn(self, is_speech: bool):
        t = self.turn
        if is_speech:
            if t is None:
                self.cand_id += 1
                t = self.turn = {"id": self.cand_id, "start": self.ms - 120, "last": self.ms, "text": "", "done": False,
                                 "sub": -1e9, "voiced": 0}
            t["done"] = False                  # they carried on: the turn (and its transcript) continues
            t["last"] = self.ms
            t["voiced"] += FRAME_MS
            return None
        if t is None or t["done"] or t["voiced"] < 200:
            return None
        silence = self.ms - t["last"]
        if silence >= 160 and self.ms - t["sub"] > 400 and t["sub"] < t["last"]:
            if self.asr.submit(("turn", t["id"], self.ms), self._audio_since(max(t["start"], self.ms - 29000))):
                t["sub"] = self.ms
        need = lexicon.expected_digits(self.agent_text)
        unfinished = lexicon.is_incomplete(t["text"]) or (need is not None and lexicon.count_digits(t["text"]) < need)
        wait = EOT_LONG_MS if unfinished else EOT_BASE_MS
        if silence >= wait:
            t["done"] = True
            self._log("TAKE_TURN", text=t["text"], waited=silence)
            return "TAKE_TURN"
        return None
