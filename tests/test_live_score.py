import asyncio

import numpy as np

from convo.engine import Conversation
from convo.live_score import LiveScore
from ttbench.score import DEFAULT_THRESHOLDS, harmonic, stop_credit


def test_no_score_before_the_first_interruption():
    score = LiveScore()
    score.kept(1000, "إيه")
    snap = score.snapshot()
    assert snap["core"] is None and snap["S"] == 1.0


def test_a_fast_stop_earns_full_credit():
    score = LiveScore()
    score.stopped(1000, 1700, "لحظة")
    snap = score.snapshot()
    assert snap["events"][0]["latency_ms"] == 200
    assert snap["R"] == 1.0 and snap["core"] == 100.0


def test_a_late_stop_uses_the_official_curve():
    score = LiveScore()
    score.stopped(1000, 2400, "لحظة")
    assert score.snapshot()["R"] == round(stop_credit(900, DEFAULT_THRESHOLDS), 3)


def test_the_core_formula_matches_the_official_one():
    score = LiveScore()
    score.stopped(1000, 2400, "لحظة")
    score.kept(5000, "إيه")
    score.stopped(8000, 8500, "إيه")
    score.relabel(2, "KEEP")
    snap = score.snapshot()
    r, s = stop_credit(900, DEFAULT_THRESHOLDS), 0.5
    assert snap["core"] == round(100 * harmonic(r, s), 1)


def test_correcting_a_missed_interruption_lowers_the_score():
    score = LiveScore()
    score.stopped(1000, 1600, "لحظة")
    score.kept(5000, "كم المبلغ")
    score.relabel(1, "STOP")
    assert score.snapshot()["R"] == 0.5


def test_a_quick_take_back_earns_half_credit():
    score = LiveScore()
    score.stopped(1000, 1600, "إيه")
    score.took_back(1900)
    event = score.snapshot()["events"][0]
    assert (event["expected"], event["action"], event["credit"]) == ("KEEP", "resumed", 0.5)


def test_a_slow_take_back_is_a_false_stop():
    score = LiveScore()
    score.stopped(1000, 1600, "إيه")
    score.took_back(2600)
    assert score.snapshot()["events"][0]["credit"] == 0.0


def test_invalid_corrections_are_ignored():
    score = LiveScore()
    score.kept(1000, "إيه")
    score.relabel(5, "STOP")
    score.relabel(0, "MAYBE")
    assert score.snapshot()["events"][0]["expected"] == "KEEP"


def test_a_real_stop_in_a_call_reaches_the_page():
    sent = []

    async def send(message):
        sent.append(message)

    async def run():
        conv = Conversation(send, "payment", "edge")
        conv.state = "speaking"
        conv.ms = 3000
        feats = {"n": 1, "echo_ratio": 0.0, "bid": 1, "content": 0, "repeated_ambig": False}
        conv.tt.cand = {"start": 2400, "feats": feats, "text": "وقف"}
        conv.tt.trace.append({"t": 3000, "kind": "STOP", "why": "floor-bid word"})
        conv.seen_trace = 1
        conv.echo_flags = [False] * 50
        conv.tt.on_frame = lambda agent, caller: "STOP"
        silence = np.zeros(320, np.int16)
        await conv.on_frame(silence, silence)

    asyncio.run(run())
    score = [m for m in sent if m["type"] == "score"][-1]
    assert score["events"][0]["text"] == "وقف" and score["events"][0]["action"] == "stopped"
