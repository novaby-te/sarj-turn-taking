"""A live estimate of the competition score during a demo call.

The official scorer needs labels that say what the caller meant. A live call has none, so each moment starts with the
system's own decision as its label, and the person testing can correct it on the page. Credits and the final formula
are the official ones from ttbench.score. Ambiguous moments are not labelled here, so Core = 100 x F.
"""
from __future__ import annotations

from statistics import fmean

from ttbench.score import DEFAULT_THRESHOLDS, harmonic, stop_credit

EVIDENT_AFTER_ONSET_MS = 500        # when the intent became clear is unknown live: the same assumption as our labelled tries


class LiveScore:
    def __init__(self):
        self.events: list[dict] = []

    def _add(self, t: float, text: str, action: str, expected: str, latency_ms: float | None = None) -> dict:
        event = {"id": len(self.events), "t": t, "text": text, "action": action, "expected": expected, "latency_ms": latency_ms}
        self.events.append(event)
        return event

    def stopped(self, onset_ms: float, stop_ms: float, text: str) -> None:
        latency = max(0.0, stop_ms - (onset_ms + EVIDENT_AFTER_ONSET_MS))
        self._add(stop_ms, text, "stopped", "STOP", round(latency))

    def kept(self, t: float, text: str) -> None:
        self._add(t, text, "kept", "KEEP")

    def _last_stop(self) -> dict | None:
        return next((e for e in reversed(self.events) if e["action"] in ("stopped", "resumed")), None)

    def heard(self, text: str) -> None:
        """The full transcript of the interruption is ready."""
        event = self._last_stop()
        if event is not None and text:
            event["text"] = text

    def took_back(self, resume_ms: float) -> None:
        """The agent stopped, then resumed: an acknowledgement after all."""
        event = self._last_stop()
        if event is not None:
            quick = resume_ms - event["t"] <= DEFAULT_THRESHOLDS["hiccup_ms"]
            event.update(action="resumed" if quick else "stopped", expected="KEEP")

    def was_acknowledgement(self, text: str) -> None:
        """The agent stopped, but the caller only acknowledged."""
        event = self._last_stop()
        if event is not None:
            event.update(expected="KEEP", text=text or event["text"])

    def relabel(self, event_id: int, expected: str) -> None:
        """The person testing says what the agent should have done."""
        if expected in ("STOP", "KEEP") and isinstance(event_id, int) and 0 <= event_id < len(self.events):
            self.events[event_id]["expected"] = expected

    @staticmethod
    def credit(event: dict) -> float:
        if event["expected"] == "STOP":
            return stop_credit(event["latency_ms"], DEFAULT_THRESHOLDS) if event["action"] == "stopped" else 0.0
        return {"kept": 1.0, "resumed": 0.5}.get(event["action"], 0.0)

    def snapshot(self) -> dict:
        events = [{**e, "credit": round(self.credit(e), 2)} for e in self.events]
        stops = [self.credit(e) for e in self.events if e["expected"] == "STOP"]
        keeps = [self.credit(e) for e in self.events if e["expected"] == "KEEP"]
        r = fmean(stops) if stops else None
        s = fmean(keeps) if keeps else 1.0          # no mistaken stop so far
        f = None if r is None else harmonic(r, s)
        return {"core": None if f is None else round(100 * f, 1), "R": None if r is None else round(r, 3), "S": round(s, 3),
                "F": None if f is None else round(f, 3), "events": events}
