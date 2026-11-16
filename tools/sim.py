"""Fast offline evaluator: runs a system in-process on every call (in parallel), then scores it with the
OFFICIAL scorer (ttbench.score). Same timing rules as the harness, without the 10-minute real-time wait.

    .venv/bin/python -m tools.sim                         # my_system.py on data/dev, prints the score
    .venv/bin/python -m tools.sim --data data/synth       # on our own generated calls
    .venv/bin/python -m tools.sim --set STOP_MS=400       # override a parameter of the system (for tuning)

How time works here (mirrors ttbench/harness.py):
  frame n is delivered at stream time n*20 ms; a decision returned by on_frame is stamped
  n*20 + the wall-clock time on_frame took. Slow components (ASR) run through turn.clock.Async, which in
  simulation charges their MEASURED compute time before their result becomes visible. So latencies here
  are honest, as long as the machine is not overloaded.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
os.environ["TT_SIM"] = "1"          # must be set before turn.clock / turn.asr are imported anywhere

from ttbench.dataset import GAP_MS, lay_out_stream, load_split  # noqa: E402
from ttbench.harness import build_stream  # noqa: E402
from ttbench.protocol import FRAME_MS, FRAME_SAMPLES  # noqa: E402
from ttbench.score import load_labels, score  # noqa: E402


def load_system_class(path: str):
    spec = importlib.util.spec_from_file_location("system_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _apply_overrides(mod, overrides: dict):
    for k, v in overrides.items():
        target = mod
        if "." in k:                       # e.g. policy.STOP_MS -> turn.policy module
            modname, k = k.rsplit(".", 1)
            target = sys.modules.get(f"turn.{modname}") or importlib.import_module(f"turn.{modname}")
        setattr(target, k, type(getattr(target, k))(v) if hasattr(target, k) else v)


def run_one(args) -> tuple[str, list[dict], dict]:
    """Run one scenario through a fresh system. Returns (scenario_id, decisions, trace)."""
    system_path, data_dir, sid, overrides = args
    os.environ["TT_SIM"] = "1"
    from turn import clock
    mod = load_system_class(system_path)
    _apply_overrides(mod, overrides)
    try:                                   # load Whisper before the call, as the live server does at start-up
        from turn import asr
        if asr.MODEL != "none" and not getattr(asr, "_warm", False):
            asr.warmup()
            asr._warm = True
    except ImportError:
        pass
    sc = load_split(Path(data_dir), [sid])
    sc = lay_out_stream(sc)
    stream, controls = build_stream(sc)
    n_frames = len(stream) // FRAME_SAMPLES
    system = mod.MySystem()
    out = []
    t_compute = 0.0
    for n in range(n_frames):
        clock.set_now(n * FRAME_MS)
        for msg in controls.get(n, []):
            t = msg["type"]
            if t == "scenario_start":
                system.on_scenario_start(msg.get("context", {}))
            elif t == "agent_utterance_start":
                system.on_agent_start(msg.get("text", ""), msg.get("duration_ms", 0))
            elif t == "agent_utterance_end":
                system.on_agent_end()
        fr = stream[n * FRAME_SAMPLES:(n + 1) * FRAME_SAMPLES]
        t0 = time.perf_counter()
        res = system.on_frame(fr[:, 0], fr[:, 1])
        dt = (time.perf_counter() - t0) * 1000
        t_compute += dt
        if res is None:
            continue
        for d in (res if isinstance(res, list) else [res]):
            m = {"decision": d} if isinstance(d, str) else dict(d)
            out.append({"stream_ms": round(n * FRAME_MS + dt, 2), "msg": {"type": "decision", **m}})
    trace = getattr(system, "trace", [])
    return sid, out, {"compute_ms": t_compute, "frames": n_frames, "trace": trace}


def simulate(system_path: str, data_dir: Path, only: list[str] | None = None, overrides: dict | None = None,
             workers: int | None = None, out_dir: Path | None = None) -> dict:
    labels, manifest = load_labels(data_dir)
    ids = [s for s in labels if not only or s in only]
    jobs = [(system_path, str(data_dir), sid, overrides or {}) for sid in ids]
    # Whisper on the GPU: parallel workers would slow each other down and inflate the measured latency.
    asr_on = os.environ.get("TT_ASR_MODEL", "") != "none"
    workers = workers or (1 if asr_on else min(8, os.cpu_count() or 4))
    if workers == 1 and asr_on:            # one process, cache-friendly, honest latency
        os.environ["TT_SIM"] = "1"
        results = [run_one(j) for j in jobs]
    elif workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(workers) as ex:
            results = list(ex.map(run_one, jobs))
    else:
        results = [run_one(j) for j in jobs]
    # lay the calls out exactly like the harness and shift each call's decisions to its place in the stream
    layout, t = [], 0
    log = []
    traces = {}
    compute = 0.0
    frames = 0
    for sid, decs, info in results:
        dur = labels[sid]["audio"]["duration_ms"]
        dur = -(-dur // FRAME_MS) * FRAME_MS
        layout.append({"scenario_id": sid, "opaque_id": sid, "start_ms": t, "duration_ms": dur})
        for d in decs:
            log.append({"dir": "rx", "stream_ms": d["stream_ms"] + t, "msg": d["msg"]})
        traces[sid] = info["trace"]
        compute += info["compute_ms"]
        frames += info["frames"]
        t += dur + GAP_MS
    meta = {"scenarios": layout, "hello": {"team": "sim", "system_version": "sim"}, "official": False,
            "speed": "sim", "errors": []}
    result = score(labels, meta, log, manifest.get("thresholds"))
    result["sim"] = {"avg_frame_compute_ms": round(compute / max(frames, 1), 3), "frames": frames}
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        with open(out_dir / "log.jsonl", "w", encoding="utf-8") as f:
            for i, line in enumerate(log):
                f.write(json.dumps({"seq": i, **line}, ensure_ascii=False) + "\n")
        (out_dir / "score.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "trace.json").write_text(json.dumps(traces, ensure_ascii=False), encoding="utf-8")
        from ttbench.report import write_report
        write_report(result, labels, out_dir / "report.md")
    return result


def line(r: dict) -> str:
    c = r["core"]
    return (f"core {c['core_score']:5.1f}  R {c['R']:.2f}  S {c['S']:.2f}  A {c['A'] if c['A'] is None else round(c['A'], 2)}"
            f"  EOT {r['eot']['eot_score']:.1f}  false_stops {r['rates']['false_stops']}  misses {r['rates']['misses']}"
            f"  median_lat {r['latency_ms']['median']}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", default=str(KIT / "my_system.py"))
    p.add_argument("--data", default=str(KIT / "data" / "dev"))
    p.add_argument("--calls", default=None, help="comma-separated ids (default: all)")
    p.add_argument("--set", action="append", default=[], help="NAME=VALUE parameter override (repeatable)")
    p.add_argument("--name", default=None, help="save the run to runs/sim-<name> (for the dashboard)")
    p.add_argument("--workers", type=int, default=None)
    a = p.parse_args(argv)
    ov = dict(s.split("=", 1) for s in a.set)
    out = KIT / "runs" / f"sim-{a.name}" if a.name else None
    t0 = time.time()
    r = simulate(a.system, Path(a.data), a.calls.split(",") if a.calls else None, ov, a.workers, out)
    print(line(r), f"  ({time.time() - t0:.1f}s, {r['sim']['avg_frame_compute_ms']} ms/frame)")
    for e in r["events"]:
        mark = "OK " if e["credit"] >= 0.99 else ("~  " if e["credit"] > 0 else "XX ")
        print(f"  {mark}{e['event_id']:12s} {e['expected']:9s} {e['outcome']:10s} {e['credit']:.2f}  {e.get('detail', '')}")
    for s in r["scenarios"]:
        if s["spurious_stops"]:
            print(f"  XX {s['scenario_id']:12s} stray stops at {s['spurious_stops']}")
    if out:
        hist = KIT / "runs" / "sim-history.csv"
        new = not hist.exists()
        with open(hist, "a", encoding="utf-8") as f:
            if new:
                f.write("when,name,data,core,R,S,A,eot,false_stops,misses,median_lat,overrides\n")
            c = r["core"]
            f.write(f"{time.strftime('%Y-%m-%d %H:%M')},{a.name},{Path(a.data).name},{c['core_score']:.1f},{c['R']:.3f},"
                    f"{c['S']:.3f},{'' if c['A'] is None else round(c['A'], 3)},{r['eot']['eot_score']:.1f},"
                    f"{r['rates']['false_stops']},{r['rates']['misses']},{r['latency_ms']['median']},"
                    f"\"{' '.join(a.set)}\"\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
