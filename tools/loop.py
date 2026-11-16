"""ONE command for the feedback loop:  run -> score -> compare with last time -> dashboard.

    .venv/bin/python -m tools.loop                    # dev + dev-aug + synth (+ live labels if any), then the dashboard
    .venv/bin/python -m tools.loop --note "lexicon v2"
    .venv/bin/python -m tools.loop --sets dev         # quicker

Each run is saved under runs/sim-<set>-latest and appended to runs/sim-history.csv with your note, so the
dashboard can show whether a change helped (and where it hurt).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
from tools.sim import line, simulate  # noqa: E402

SETS = {"dev": "data/dev", "dev-aug": "data/dev-aug", "synth": "data/synth", "live": "data/live"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="dev,dev-aug,synth,live")
    ap.add_argument("--note", default="")
    ap.add_argument("--set", action="append", default=[], help="NAME=VALUE parameter override")
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args(argv)
    ov = dict(s.split("=", 1) for s in a.set)
    hist = KIT / "runs" / "sim-history.csv"
    if not hist.exists():
        hist.write_text("when,name,data,core,R,S,A,eot,false_stops,misses,median_lat,overrides\n")
    version = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=KIT, capture_output=True, text=True).stdout.strip()
    for s in a.sets.split(","):
        d = KIT / SETS.get(s, s)
        if not (d / "manifest.json").exists():
            continue
        t0 = time.time()
        r = simulate(str(KIT / "my_system.py"), d, overrides=ov, out_dir=KIT / "runs" / f"sim-{s}-latest")
        c = r["core"]
        print(f"{s:8s} {line(r)}  ({time.time() - t0:.0f}s)")
        with open(hist, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M')},{(a.note or version).replace(',', ' ')},{s},{c['core_score']:.1f},"
                    f"{c['R']:.3f},{c['S']:.3f},{'' if c['A'] is None else round(c['A'], 3)},{r['eot']['eot_score']:.1f},"
                    f"{r['rates']['false_stops']},{r['rates']['misses']},{r['latency_ms']['median']},\"{' '.join(a.set)}\"\n")
    from tools.dashboard import build
    out = build()
    print("dashboard:", out)
    if not a.no_open:
        subprocess.run(["open", str(out)])


if __name__ == "__main__":
    main()
