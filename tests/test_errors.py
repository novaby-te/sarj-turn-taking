from tools.errors import classify, noise_of

LABELS = {"A": {"dialect": "najdi", "acoustic": {"conditions": ["quiet"]}},
          "B": {"dialect": "gulf-omani", "acoustic": {"conditions": ["car"]}}}
TRACES = {"A": [{"t": 1000, "kind": "asr", "text": "إيه", "why": "filler", "f": {"sem_p": 0.2}},
                {"t": 1100, "kind": "STOP", "text": "إيه طيب", "why": "2 content words"}],
          "B": [{"t": 5200, "kind": "asr", "text": "لحظة", "why": "unclear", "f": {"sem_p": 0.7}}]}


def event(sid, outcome, **extra):
    return {"scenario_id": sid, "event_id": f"{sid}.e1", "category": "BC", "onset_ms": 900 if sid == "A" else 5000,
            "outcome": outcome, "credit": 0.0, **extra}


def score(*events):
    return {"events": list(events), "thresholds": {"T_full_ms": 300}}


def test_each_error_type_is_recognised():
    found = classify(score(event("A", "false_stop"), event("B", "miss"), event("A", "yield", latency_ms=650),
                           event("A", "eot", patience_items=[{"item": "pause", "passed": False}])), LABELS, TRACES)
    assert [e["type"] for e in found] == ["false_stop", "missed", "late", "premature_take_turn"]


def test_correct_behaviour_is_not_an_error():
    ok = score(event("A", "kept"), event("A", "yield", latency_ms=250),
               event("A", "eot", patience_items=[{"item": "pause", "passed": True}]))
    assert classify(ok, LABELS, TRACES) == []


def test_examples_carry_the_evidence():
    stop = classify(score(event("A", "false_stop")), LABELS, TRACES)[0]
    assert (stop["text"], stop["rule"], stop["sem_p"], stop["dialect"], stop["noise"]) == ("إيه طيب", "2 content words", 0.2, "najdi", "clean")
    miss = classify(score(event("B", "miss")), LABELS, TRACES)[0]
    assert (miss["text"], miss["sem_p"], miss["noise"]) == ("لحظة", 0.7, "car")


def test_stray_stops_count_as_false_stops():
    run = score()
    run["scenarios"] = [{"scenario_id": "A", "category": "BC", "spurious_stops": [1100]}]
    stray = classify(run, LABELS, TRACES)[0]
    assert (stray["type"], stray["event"], stray["text"]) == ("false_stop", "A.stray", "إيه طيب")


def test_noise_defaults_to_clean():
    assert noise_of({}) == "clean"
