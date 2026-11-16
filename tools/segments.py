"""The labelled segment bank: every sound in every call, with what it truly is, so each LAYER can be scored on its
own (not only the final score). Writes analysis/segments.jsonl.

Per segment:
  cls      bid (wants the floor) · bc (acknowledgement) · filler (thinking) · answer (U0) · nonspeech (cough, laugh,
           horn...) · other_speaker (TV, someone in the room) · echo (the agent's own voice back)
  same     1 = the caller's own voice, 0 = someone/something else, None = not a voice
  near     1 = close to the mic (the caller), 0 = far (TV, room, echo), None = unknown
  intent   1 = should STOP, 0 = should KEEP, None = either / not during agent speech

    .venv/bin/python -m tools.segments
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
SETS = ["dev", "dev-aug", "synth", "synth-test"]


def build() -> list[dict]:
    out = []
    for s in SETS:
        d = KIT / "data" / s
        if not (d / "manifest.json").exists():
            continue
        for e in json.loads((d / "manifest.json").read_text())["scenarios"]:
            lab = json.loads((d / "labels" / f"{e['scenario_id']}.json").read_text())
            u0 = next((x for x in lab["user_segments"] if x["seg_id"] == "U0"), None)
            ev_by_seg = {sid: ev for ev in lab["events"] for sid in ev.get("segment_ids", [])}
            used = set()
            for x in lab["user_segments"]:
                ev = ev_by_seg.get(x["seg_id"])
                if ev is None:   # generated calls: match by time
                    ev = next((v for v in lab["events"] if v["kind"] == "interruption"
                               and abs(v["onset_ms"] - x["start_ms"]) < 400), None)
                kind = x["kind"]
                if x["seg_id"] == "U0":
                    cls, same, near, intent = "answer", 1, 1, None
                elif kind == "side_speech":
                    cls, same, near, intent = "other_speaker", 0, 0, 0
                elif kind == "nonverbal":
                    cls, same, near, intent = "nonspeech", None, None, 0
                elif kind == "filler":
                    cls, same, near, intent = "filler", 1, 1, 0
                elif kind == "backchannel":
                    cls, same, near, intent = "bc", 1, 1, 0
                else:
                    if ev is None or ev["kind"] != "interruption":
                        cls, intent = "turn", None            # an answer while the agent is silent (EOT)
                    elif ev["expected"] == "STOP":
                        cls, intent = "bid", 1
                    elif ev["expected"] == "EITHER":
                        cls, intent = ("bid" if ev.get("preferred") == "STOP" else "bc"), None
                    else:
                        cls, intent = "bc", 0
                    same, near = 1, 1
                if ev is not None:
                    used.add(ev["event_id"])
                out.append({"set": s, "sid": lab["scenario_id"], "seg": x["seg_id"], "start": x["start_ms"], "end": x["end_ms"],
                            "cls": cls, "same": same, "near": near, "intent": intent, "text": x.get("text", ""),
                            "delivery": x.get("delivery", ""), "dialect": lab["dialect"],
                            "conditions": lab["acoustic"]["conditions"], "evident": ev.get("evident_ms") if ev else None,
                            "u0": [u0["start_ms"], u0["end_ms"]] if u0 else None})
            for ev in lab["events"]:                         # events without a caller segment: TV, echo, background
                if ev["event_id"] in used or ev["kind"] != "interruption":
                    continue
                r = ev.get("rationale", "").lower()
                if "tv" in r or "تلفزيون" in r:
                    cls, same, near = "other_speaker", 0, 0
                elif "speakerphone" in r or "echo" in r or "صدى" in r or "agent" in r:
                    cls, same, near = "echo", 0, 0
                else:
                    cls, same, near = "nonspeech", None, None
                out.append({"set": s, "sid": lab["scenario_id"], "seg": ev["event_id"], "start": ev["onset_ms"],
                            "end": min(ev["offset_ms"], ev["onset_ms"] + 4000), "cls": cls, "same": same, "near": near,
                            "intent": 0, "text": "", "delivery": ev.get("rationale", ""), "dialect": lab["dialect"],
                            "conditions": lab["acoustic"]["conditions"], "evident": None,
                            "u0": [u0["start_ms"], u0["end_ms"]] if u0 else None})
    return out


def main():
    segs = build()
    p = KIT / "analysis" / "segments.jsonl"
    p.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in segs) + "\n", encoding="utf-8")
    from collections import Counter
    print(len(segs), "segments")
    for s in SETS:
        print(f"  {s:11s}", dict(Counter(x["cls"] for x in segs if x["set"] == s)))


if __name__ == "__main__":
    sys.exit(main())
