"""v3 training tools: contamination filter, hard test material, and the student gate (no models needed)."""
import json

import numpy as np

from tools import decontam, hard_bank, student_eval


def test_a_copied_test_sentence_is_caught_even_with_spelling_changes():
    idx = decontam.Index([decontam.norm("لا لا لحظة، أنا ما طلبت هذا أبداً")])
    assert idx.hit("لا لا لحظه انا ما طلبت هذا ابدا")
    assert idx.hit("لا لا لحظة أنا ما طلبت هذا") is not None          # most of it is the same sentence
    assert idx.hit("لا والله ما طلبت شي من عندكم") is None


def test_everyday_short_phrases_are_never_dropped():
    idx = decontam.Index([decontam.norm("إيه تمام")])
    assert idx.hit("إيه تمام") is None


def test_hard_material_is_balanced_and_marked():
    kinds = {t["kind"] for t in hard_bank.TEXT}
    assert {"single_ack", "ack_then_bid", "thinking", "room_talk", "echo", "asr_junk"} <= kinds
    labels = [t["label"] for t in hard_bank.TEXT]
    assert labels.count("STOP") >= 20 and labels.count("KEEP") >= 30
    assert all("*" in s for s in hard_bank.BANK["stop"])                # the evident word is marked for the labels
    assert set(hard_bank.BANK) == {"a1", "stop", "bc", "fillers", "side", "tv", "talkover", "am"}   # synth_calls --bank keys


def test_the_gate_rejects_a_student_that_stops_more_often():
    base = {"held_out": {"S": 0.986, "F": 0.956}, "hard": {"errors": [1, 2, 3]}}
    more_false_stops = {"held_out": {"S": 0.97, "F": 0.97}, "hard": {"errors": [1]}}
    better = {"held_out": {"S": 0.986, "F": 0.965}, "hard": {"errors": [1]}}
    assert not student_eval.gate(base, more_false_stops)[0]
    assert student_eval.gate(base, better)[0]


def test_metrics_count_false_stops_and_misses():
    m = student_eval.metrics(np.array([0.9, 0.1, 0.8, 0.2]), np.array([1, 1, 0, 0]))
    assert (m["false_stops"], m["misses"], m["R"], m["S"]) == (1, 1, 0.5, 0.5)
