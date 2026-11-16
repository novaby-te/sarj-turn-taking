"""The feedback loop, automated: try parameter changes, keep the ones that raise the score.

Coordinate search over the policy's knobs, scored on a TUNING set (generated + stressed calls); every accepted
change is re-checked on the organisers' dev calls, which are never used to choose (that is how we avoid
fooling ourselves). Results are appended to runs/tune-log.csv.

    .venv/bin/python -m tools.tune --tune data/synth,data/dev-aug --check data/dev --rounds 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
from tools.sim import simulate  # noqa: E402

SPACE = {
    "policy.REL_DB": [7.0, 8.5, 10.0, 12.0, 14.0],
    "policy.STOP_REL_DB": [5.0, 6.5, 8.0, 10.0, 12.0],
    "policy.VAD_TH": [0.35, 0.5, 0.65],
    "policy.SNR_DB": [6.0, 9.0, 12.0],
    "policy.MIN_VOICED_MS": [100, 160, 240],
    "policy.ASR_EVERY_MS": [160, 220, 300],
    "policy.MODEL_TH": [0.35, 0.5, 0.65, 0.8],
    "policy.CONTENT_MIN": [1, 2, 3],
    "policy.LONG_MS": [1400, 2000, 2600],
}


def evaluate(params: dict, sets: list[str]) -> float:
    scores = []
    for d in sets:
        r = simulate(str(KIT / "my_system.py"), KIT / d, overrides=params)
        scores.append(r["core"]["core_score"])
    return sum(scores) / len(scores)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", default="data/synth,data/dev-aug")
    ap.add_argument("--check", default="data/dev")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--only", default=None, help="comma list of knobs to search")
    a = ap.parse_args(argv)
    tune_sets, check_sets = a.tune.split(","), a.check.split(",")
    space = {k: v for k, v in SPACE.items() if not a.only or k.split(".")[-1] in a.only.split(",")}
    best: dict = {}
    best_s = evaluate(best, tune_sets)
    print(f"start: tune {best_s:.1f}  check {evaluate(best, check_sets):.1f}")
    log = open(KIT / "runs" / "tune-log.csv", "a")
    for rnd in range(a.rounds):
        for k, vals in space.items():
            for v in vals:
                trial = {**best, k: v}
                t0 = time.time()
                s = evaluate(trial, tune_sets)
                log.write(f"{time.strftime('%H:%M:%S')},{rnd},{k},{v},{s:.2f}\n")
                log.flush()
                if s > best_s + 0.2:
                    best, best_s = trial, s
                    print(f"  better: {k}={v} -> tune {s:.1f} (check {evaluate(best, check_sets):.1f})  [{time.time() - t0:.0f}s]")
    print("best params:", json.dumps(best))
    print(f"final: tune {best_s:.1f}  check {evaluate(best, check_sets):.1f}")
    (KIT / "runs" / "tune-best.json").write_text(json.dumps(best, indent=1))


if __name__ == "__main__":
    main()
