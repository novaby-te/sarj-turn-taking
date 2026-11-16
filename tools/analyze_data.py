"""Analyse every development call: levels, noise, echo, timing, pitch. Writes analysis/data.json + per-call SVG strips.

    .venv/bin/python -m tools.analyze_data [--data data/dev]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
from ttbench.dataset import read_wav_stereo  # noqa: E402

SR = 16000
HOP = 320  # 20 ms


def frame_db(x: np.ndarray) -> np.ndarray:
    n = len(x) // HOP
    f = x[: n * HOP].astype(np.float64).reshape(n, HOP) / 32768
    return 20 * np.log10(np.sqrt((f ** 2).mean(1)) + 1e-9)


def f0_track(x: np.ndarray, fmin=75, fmax=400) -> np.ndarray:
    """Simple autocorrelation pitch per 20 ms hop on 40 ms windows; NaN when unvoiced."""
    x = x.astype(np.float64) / 32768
    win = 640
    out = []
    lo, hi = SR // fmax, SR // fmin
    for i in range(0, len(x) - win, HOP):
        w = x[i:i + win] * np.hanning(win)
        if np.sqrt((w ** 2).mean()) < 0.003:
            out.append(np.nan)
            continue
        ac = np.correlate(w, w, "full")[win - 1:]
        ac /= ac[0] + 1e-12
        k = lo + int(np.argmax(ac[lo:hi]))
        out.append(SR / k if ac[k] > 0.45 else np.nan)
    return np.array(out)


def segs_frames(start_ms, end_ms):
    return slice(int(start_ms // 20), int(end_ms // 20))


def echo_estimate(agent: np.ndarray, caller: np.ndarray, spans) -> dict:
    """How much of the agent's own voice leaks into the caller channel, and with what delay."""
    best = {"corr": 0.0, "lag_ms": None, "gain_db": None}
    for s, e in spans:
        a = agent[int(s * 16):int(e * 16)].astype(np.float64)
        c = caller[int(s * 16):int(e * 16)].astype(np.float64)
        if len(a) < 16000 or a.std() < 50:
            continue
        a, c = a[:48000], c[:48000]
        A = np.fft.rfft(a, 2 * len(a))
        C = np.fft.rfft(c, 2 * len(a))
        xc = np.fft.irfft(C * np.conj(A))[: 8000]          # caller lagging agent by 0..500 ms
        k = int(np.argmax(np.abs(xc)))
        corr = abs(xc[k]) / (np.linalg.norm(a) * np.linalg.norm(c) + 1e-9)
        if corr > best["corr"]:
            g = np.std(c) / (np.std(a) + 1e-9)
            best = {"corr": round(float(corr), 3), "lag_ms": round(k / 16, 1), "gain_db": round(float(20 * np.log10(corr * g + 1e-9)), 1)}
    return best


def analyse(data_dir: Path) -> dict:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    calls = []
    for entry in manifest["scenarios"]:
        sid = entry["scenario_id"]
        lab = json.loads((data_dir / "labels" / f"{sid}.json").read_text(encoding="utf-8"))
        audio = read_wav_stereo(data_dir / "audio" / lab["audio"]["file"])
        ag, ca = audio[:, 0], audio[:, 1]
        adb, cdb = frame_db(ag), frame_db(ca)
        n = len(cdb)
        user_mask = np.zeros(n, bool)
        for s in lab["user_segments"]:
            user_mask[segs_frames(s["start_ms"], s["end_ms"])] = True
        for ev in lab["events"]:
            user_mask[segs_frames(ev["onset_ms"], ev.get("offset_ms", ev.get("turn_end_ms", ev["onset_ms"])))] = True
        agent_mask = np.zeros(n, bool)
        for u in lab["agent_utterances"]:
            agent_mask[segs_frames(u["start_ms"], u["end_ms"])] = True
        bg_during_agent = cdb[agent_mask & ~user_mask]
        bg_silence = cdb[~agent_mask & ~user_mask]
        u0 = next(s for s in lab["user_segments"] if s["seg_id"] == "U0")
        u0db = cdb[segs_frames(u0["start_ms"], u0["end_ms"])]
        u0_level = float(np.percentile(u0db, 80))
        noise_floor = float(np.percentile(bg_silence, 50)) if len(bg_silence) else None
        bg_agent_p90 = float(np.percentile(bg_during_agent, 90)) if len(bg_during_agent) else None
        f0 = f0_track(ca)
        segs = []
        for s in lab["user_segments"]:
            sl = segs_frames(s["start_ms"], s["end_ms"])
            d = cdb[sl]
            pf = f0[sl.start:min(sl.stop, len(f0))]
            voiced = pf[~np.isnan(pf)]
            slope = None
            if len(voiced) >= 6:
                tail = voiced[-min(10, len(voiced)):]
                slope = round(float(np.polyfit(np.arange(len(tail)), 12 * np.log2(tail / tail[0]), 1)[0] * 50), 1)  # semitones / s
            segs.append({"seg_id": s["seg_id"], "kind": s["kind"], "text": s.get("text", ""),
                         "delivery": s.get("delivery", ""), "start_ms": s["start_ms"], "end_ms": s["end_ms"],
                         "dur_ms": s["end_ms"] - s["start_ms"], "level_p80": round(float(np.percentile(d, 80)), 1),
                         "rel_u0_db": round(float(np.percentile(d, 80)) - u0_level, 1),
                         "f0_median": None if not len(voiced) else round(float(np.median(voiced)), 0),
                         "f0_tail_slope_st_per_s": slope,
                         "overlaps_agent": bool(agent_mask[sl].any())})
        evs = []
        for ev in lab["events"]:
            off = ev.get("offset_ms", ev.get("turn_end_ms"))
            d = cdb[segs_frames(ev["onset_ms"], off)]
            evs.append({"event_id": ev["event_id"], "kind": ev["kind"], "expected": ev["expected"],
                        "preferred": ev.get("preferred"), "onset_ms": ev["onset_ms"], "offset_ms": off,
                        "evident_ms": ev.get("evident_ms"), "dur_ms": off - ev["onset_ms"],
                        "evident_delay_ms": (ev["evident_ms"] - ev["onset_ms"]) if "evident_ms" in ev else None,
                        "level_p80": round(float(np.percentile(d, 80)), 1),
                        "rel_u0_db": round(float(np.percentile(d, 80)) - u0_level, 1),
                        "rationale": ev.get("rationale", ""), "pauses": ev.get("pauses")})
        spans = [(u["start_ms"], u["end_ms"]) for u in lab["agent_utterances"]]
        # strip for the HTML: 100 ms resolution
        k = 5
        m = n // k
        strip = {"caller_db": [round(float(v), 1) for v in cdb[: m * k].reshape(m, k).max(1)],
                 "agent_db": [round(float(v), 1) for v in adb[: m * k].reshape(m, k).max(1)]}
        calls.append({"scenario_id": sid, "category": lab["category"], "subtype": lab["subtype"],
                      "difficulty": lab["difficulty"], "dialect": lab["dialect"],
                      "conditions": lab["acoustic"]["conditions"], "snr_db": lab["acoustic"].get("snr_db"),
                      "duration_ms": lab["audio"]["duration_ms"], "use_case": lab["context"]["use_case"],
                      "speaker": lab["speakers"][0], "agent_voice": lab.get("agent_voice"),
                      "agent_utterances": [{"utt_id": u["utt_id"], "start_ms": u["start_ms"], "end_ms": u["end_ms"],
                                            "is_question": u.get("is_question"), "text": u["text"]} for u in lab["agent_utterances"]],
                      "u0_level": round(u0_level, 1), "noise_floor": None if noise_floor is None else round(noise_floor, 1),
                      "bg_during_agent_p90": None if bg_agent_p90 is None else round(bg_agent_p90, 1),
                      "echo": echo_estimate(ag, ca, spans), "segments": segs, "events": evs, "strip": strip})
    return {"calls": calls}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(KIT / "data" / "dev"))
    p.add_argument("--out", default=str(KIT / "analysis" / "data.json"))
    a = p.parse_args(argv)
    r = analyse(Path(a.data))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    for c in r["calls"]:
        print(f"{c['scenario_id']} {c['dialect']:14s} {'|'.join(c['conditions']):18s} U0 {c['u0_level']:6.1f}  floor {c['noise_floor']}  "
              f"bgAgent90 {c['bg_during_agent_p90']}  echo {c['echo']}")
        for e in c["events"]:
            print(f"     {e['expected']:9s} dur {e['dur_ms']:6d} evid+{e['evident_delay_ms']}  rel_u0 {e['rel_u0_db']:+.1f}")
        for s in c["segments"]:
            if s["seg_id"] != "U0":
                print(f"       seg {s['seg_id']:4s} {s['kind']:12s} rel {s['rel_u0_db']:+.1f} f0 {s['f0_median']} slope {s['f0_tail_slope_st_per_s']} {s['text'][:30]}")


if __name__ == "__main__":
    main()
