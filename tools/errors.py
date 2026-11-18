"""Where did a run lose points? Sorts every mistake into four types and breaks them down by category, dialect and noise.

    .venv/bin/python -m tools.errors runs/sim-novel-test-latest
    .venv/bin/python -m tools.errors runs/sim-novel-test-latest --data data/novel-test --examples 10

Types: false stop, missed interruption, late stop (after the full-credit time), premature take-turn.
Each example shows the call, the time, what the system heard, the rule that fired and the meaning score (sem_p).
The full report is written to errors.md inside the run folder.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
from ttbench.score import CLEAN_CONDITIONS, DEFAULT_THRESHOLDS  # noqa: E402

TYPES = {"false_stop": "False stop", "missed": "Missed interruption", "late": "Late stop",
         "premature_take_turn": "Premature take-turn"}
STOP_LOOKBACK_MS = 300        # a stop may start just before the caller's onset
STOP_WINDOW_MS = 8000
MISS_WINDOW_MS = 4000


def noise_of(label: dict) -> str:
    conditions = label.get("acoustic", {}).get("conditions") or ["quiet"]
    return "clean" if set(conditions) <= CLEAN_CONDITIONS else "+".join(sorted(conditions))


def _sem_before(trace: list[dict], t: float):
    reads = [x for x in trace if x.get("kind") == "asr" and x.get("t", 0) <= t and x.get("f", {}).get("sem_p", -1) >= 0]
    return round(reads[-1]["f"]["sem_p"], 2) if reads else None


def _evidence(kind: str, event: dict, trace: list[dict]) -> dict:
    """What the system heard and why it acted, taken from the run's trace."""
    onset = event["onset_ms"]
    if kind == "missed":
        reads = [x for x in trace if x.get("kind") == "asr" and onset <= x.get("t", 0) <= onset + MISS_WINDOW_MS]
        last = reads[-1] if reads else {}
        return {"t": last.get("t", onset), "text": last.get("text", ""), "rule": last.get("why", "no reading"),
                "sem_p": _sem_before(trace, last.get("t", onset))}
    want = "TAKE_TURN" if kind == "premature_take_turn" else "STOP"
    hits = [x for x in trace if x.get("kind") == want and onset - STOP_LOOKBACK_MS <= x.get("t", 0) <= onset + STOP_WINDOW_MS]
    hit = hits[0] if hits else {}
    return {"t": hit.get("t", onset), "text": hit.get("text", ""), "rule": hit.get("why", ""),
            "sem_p": _sem_before(trace, hit.get("t", onset))}


def error_type(event: dict, full_credit_ms: float) -> str | None:
    outcome = event.get("outcome")
    if outcome == "false_stop" and event.get("expected", "KEEP") == "KEEP":
        return "false_stop"
    if outcome == "miss":
        return "missed"
    if outcome == "yield" and (event.get("latency_ms") or 0) > full_credit_ms:
        return "late"
    if outcome == "eot" and any(not item["passed"] for item in event.get("patience_items", [])):
        return "premature_take_turn"
    return None


def classify(score: dict, labels: dict, traces: dict) -> list[dict]:
    full_credit_ms = score.get("thresholds", DEFAULT_THRESHOLDS)["T_full_ms"]
    errors = []
    for event in score["events"]:
        kind = error_type(event, full_credit_ms)
        if kind is None:
            continue
        sid = event["scenario_id"]
        label = labels.get(sid, {})
        errors.append({"type": kind, "call": sid, "event": event["event_id"], "category": event["category"],
                       "dialect": label.get("dialect", "unknown"), "noise": noise_of(label),
                       "latency_ms": event.get("latency_ms"), "credit": event.get("credit", 0.0),
                       **_evidence(kind, event, traces.get(sid, []))})
    for scenario in score.get("scenarios", []):              # stops while nothing labelled was happening
        sid = scenario["scenario_id"]
        label = labels.get(sid, {})
        for t in scenario.get("spurious_stops", []):
            hit = next((x for x in traces.get(sid, []) if x.get("kind") == "STOP" and x.get("t", 0) >= t - STOP_LOOKBACK_MS), {})
            errors.append({"type": "false_stop", "call": sid, "event": f"{sid}.stray", "category": scenario.get("category", ""),
                           "dialect": label.get("dialect", "unknown"), "noise": noise_of(label), "latency_ms": None, "credit": 0.0,
                           "t": t, "text": hit.get("text", ""), "rule": hit.get("why", ""), "sem_p": _sem_before(traces.get(sid, []), t)})
    return errors


def find_data(run_dir: Path, score: dict) -> Path:
    """The data folder holding this run's labels."""
    sid = score["events"][0]["scenario_id"]
    for labels in sorted((KIT / "data").glob("*/labels")):
        if (labels / f"{sid}.json").exists():
            return labels.parent
    raise SystemExit(f"No labels found for {run_dir.name}. Pass --data.")


def table(errors: list[dict], field: str) -> list[str]:
    counts = Counter((e[field], e["type"]) for e in errors)
    rows = [f"| {field} | " + " | ".join(TYPES.values()) + " | Total |", "|---|" + "---|" * (len(TYPES) + 1)]
    for value in sorted({e[field] for e in errors}):
        cells = [counts[(value, t)] for t in TYPES]
        rows.append(f"| {value} | " + " | ".join(map(str, cells)) + f" | {sum(cells)} |")
    return rows


def report(run_dir: Path, score: dict, errors: list[dict], examples: int) -> str:
    totals = Counter(e["type"] for e in errors)
    rates = score["rates"]
    lines = [f"# Error report: {run_dir.name}", "",
             f"Core {score['core']['core_score']:.2f}. Scorer totals: {rates['false_stops']} false stops, {rates['misses']} misses.", "",
             "| Type | Count |", "|---|---|", *[f"| {name} | {totals[t]} |" for t, name in TYPES.items()]]
    for field in ("category", "dialect", "noise"):
        lines += ["", f"## By {field}", "", *table(errors, field)]
    for t, name in TYPES.items():
        chosen = sorted((e for e in errors if e["type"] == t), key=lambda e: e["credit"])[:examples]
        if chosen:
            lines += ["", f"## {name}: examples", "", "| Call | Time (ms) | Heard | Rule | sem_p |", "|---|---|---|---|---|"]
            lines += [f"| {e['event']} | {e['t']} | {e['text'].replace('|', ' ')} | {e['rule'].replace('|', ' ')} | {e['sem_p']} |" for e in chosen]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run folder, e.g. runs/sim-novel-test-latest")
    ap.add_argument("--data", default=None, help="data folder (found automatically when omitted)")
    ap.add_argument("--examples", type=int, default=5, help="examples to show per error type")
    a = ap.parse_args(argv)
    run_dir = Path(a.run)
    score = json.loads((run_dir / "score.json").read_text(encoding="utf-8"))
    traces = json.loads((run_dir / "trace.json").read_text(encoding="utf-8"))
    data = Path(a.data) if a.data else find_data(run_dir, score)
    labels = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (data / "labels").glob("*.json")}
    errors = classify(score, labels, traces)
    text = report(run_dir, score, errors, a.examples)
    (run_dir / "errors.md").write_text(text, encoding="utf-8")
    totals = Counter(e["type"] for e in errors)
    print("  ".join(f"{name}: {totals[t]}" for t, name in TYPES.items()))
    ok = totals["false_stop"] == score["rates"]["false_stops"] and totals["missed"] == score["rates"]["misses"]
    print(f"Matches score.json: {'yes' if ok else 'NO'}  ({score['rates']['false_stops']} false stops, {score['rates']['misses']} misses)")
    print("Report:", run_dir / "errors.md")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
