"""How well and how fast does Whisper hear the callers? Transcribes every caller segment of the dev calls
(and partial prefixes, as the live system does) and writes analysis/asr_bench.json.

    TT_ASR_MODEL=~/Downloads/sarj-ext-data/whisper-large-v3-turbo-q4 .venv/bin/python -m tools.asr_bench
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
from ttbench.dataset import read_wav_stereo  # noqa: E402
from turn import asr, lexicon  # noqa: E402


def main():
    asr.warmup()
    rows = []
    for f in sorted((KIT / "data" / "dev" / "labels").glob("*.json")):
        lab = json.loads(f.read_text())
        st = read_wav_stereo(KIT / "data" / "dev" / "audio" / lab["audio"]["file"])
        items = [(s["seg_id"], s["start_ms"], s["end_ms"], s.get("text", ""), s["kind"]) for s in lab["user_segments"]]
        for e in lab["events"]:
            if not e.get("segment_ids"):
                items.append(("EV", e["onset_ms"], min(e["offset_ms"], e["onset_ms"] + 4000), "(background / no caller words)", "background"))
        for seg, s, e, ref, kind in items:
            x = st[max(0, s - 150) * 16:(e + 150) * 16, 1]
            r = asr.transcribe(x, use_cache=False)
            a = lexicon.analyse(r["text"])
            pre = None
            if e - s > 900:            # the first 600 ms only: what the live system knows early
                p = asr.transcribe(st[max(0, s - 150) * 16:(s + 600) * 16, 1], use_cache=False)
                pre = p["text"]
            rows.append({"scenario_id": lab["scenario_id"], "seg": seg, "kind": kind, "ref": ref, "heard": r["text"],
                         "heard_600ms": pre, "ms": r["ms"], "audio_ms": e - s + 300,
                         "bid": a["bid"], "bc": a["bc"], "content": a["content"]})
            print(f"{lab['scenario_id']} {seg:4s} {r['ms']:6.0f}ms  ref «{ref}»  heard «{r['text']}»" + (f"  [600ms: «{pre}»]" if pre else ""))
    out = KIT / "analysis" / "asr_bench.json"
    out.write_text(json.dumps({"model": asr.MODEL, "rows": rows}, ensure_ascii=False, indent=1))
    ms = sorted(r["ms"] for r in rows)
    print(f"median {ms[len(ms) // 2]:.0f} ms, p90 {ms[int(0.9 * len(ms))]:.0f} ms")


if __name__ == "__main__":
    main()
