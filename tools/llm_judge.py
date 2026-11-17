"""Can a large language model decide STOP / KEEP from the words alone? And how fast?

Takes every labelled caller moment of the segment bank (analysis/segfeat.jsonl: real floor bids, backchannels,
fillers, noise, TV / other people, echo of the agent), transcribes it with the system's ASR (Cohere by
default), pairs it with the sentence the agent was saying, and asks several "judges":

    rules        our word rules (turn/lexicon.py): STOP if a bid word, «خلاص خلاص», or >= 2 content words
    claude:<id>  Claude through the Claude Code CLI (`claude -p`, uses your subscription; no API key needed)
    mlx:<name>   a local model through convo/brain.py (mlx_lm.server): qwen3-14b, gemma-12b

The LLMs see the SAME text as the rules (words only, no sound), in batches of 20, and answer STOP/KEEP.
Results: analysis/llm_judge.json (+ a per-class table printed).

    .venv/bin/python -m tools.llm_judge --judges rules,claude:claude-opus-5-5,mlx:qwen3-14b,mlx:gemma-12b
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
import os  # noqa: E402
os.environ.setdefault("TT_SIM", "1")          # use the ASR disk cache
from ttbench.dataset import read_wav_stereo  # noqa: E402
from turn import lexicon  # noqa: E402

OUT = KIT / "analysis" / "llm_judge.json"
TEXTS = KIT / "analysis" / "llm_judge_texts.json"

SYSTEM = """You judge turn-taking in Saudi customer-service phone calls (Arabic, Gulf/Najdi/Hijazi dialects).
The AGENT is speaking. While it speaks, the CALLER line picks up something. Decide for each item:
STOP = the caller wants the floor: a question, objection, correction, «لحظة/بس/اسمع/لا لا», new information, a request.
KEEP = no need to stop: a backchannel or acknowledgement («إيه، أها، تمام، طيب، صح، معك، زين»), a filler («اممم»),
noise, a TV or another person in the room, the agent's own words echoing back, or nothing intelligible.
The caller text comes from speech recognition and may contain errors.
Answer ONLY a JSON object mapping each item number to "STOP" or "KEEP", e.g. {"1":"KEEP","2":"STOP"}."""


def agent_at(lab: dict, t_ms: float) -> str:
    cur = ""
    for u in lab["agent_utterances"]:
        if u["start_ms"] <= t_ms + 200:
            cur = u["text"]
    return cur


def build_items(engine_tag: str) -> list[dict]:
    from turn import asr
    cache = json.loads(TEXTS.read_text()) if TEXTS.exists() else {}
    rows = [json.loads(l) for l in (KIT / "analysis" / "segfeat.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("intent") in (0, 1)]
    labs, audio = {}, {}
    items = []
    for r in rows:
        key = f"{r['set']}/{r['sid']}/{r['seg']}/{engine_tag}"
        if key not in cache:
            lp = KIT / "data" / r["set"] / "labels" / f"{r['sid']}.json"
            if lp not in labs:
                labs[lp] = json.loads(lp.read_text())
                audio.clear()
                audio[lp] = read_wav_stereo(KIT / "data" / r["set"] / "audio" / labs[lp]["audio"]["file"])[:, 1]
            x = audio[lp][int(r["start"] * 16):int(r["end"] * 16)][:16000 * 9]
            cache[key] = {"text": asr.transcribe(x).get("text", "") if len(x) >= 1600 else "",
                          "agent": agent_at(labs[lp], r["start"])}
        items.append({"id": f"{r['set']}/{r['sid']}/{r['seg']}", "set": r["set"], "cls": r["cls"], "intent": r["intent"],
                      **cache[key]})
    TEXTS.write_text(json.dumps(cache, ensure_ascii=False))
    return items


def judge_rules(items):
    out = {}
    for it in items:
        a = lexicon.analyse(it["text"], it["agent"])
        out[it["id"]] = "STOP" if (a["bid"] > 0 or a["repeated_ambig"] or a["content"] >= 2) else "KEEP"
    return out, {}


def _prompt(batch):
    lines = []
    for i, it in enumerate(batch, 1):
        txt = it["text"].strip() or "(لا كلام مفهوم)"
        lines.append(f"{i}. الوكيل يقول: «{it['agent'][:160]}»\n   العميل: «{txt}»")
    return "\n".join(lines)


def _parse(reply: str, n: int) -> list[str | None]:
    m = re.search(r"\{.*\}", reply, re.S)
    d = {}
    if m:
        try:
            d = json.loads(m.group(0))
        except ValueError:
            d = {k: v for k, v in re.findall(r'"(\d+)"\s*:\s*"(STOP|KEEP)"', m.group(0))}
    return [str(d.get(str(i), "")).upper() or None for i in range(1, n + 1)]


def judge_claude(items, model, bs=20, effort="low"):
    out, timing = {}, []
    for k in range(0, len(items), bs):
        batch = items[k:k + bs]
        cmd = ["claude", "-p", "--model", model, "--effort", effort, "--tools", "", "--system-prompt", SYSTEM,
               "--no-session-persistence", "--strict-mcp-config", "--setting-sources", "", _prompt(batch)]
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=300)
        timing.append(time.time() - t0)
        for it, v in zip(batch, _parse(r.stdout, len(batch))):
            out[it["id"]] = v
        print(f"   {model} {k + len(batch)}/{len(items)}  {timing[-1]:.1f}s", flush=True)
    return out, {"batch_s": timing}


def judge_mlx(items, name, bs=20):
    from convo import brain
    brain.start(name)
    out, timing = {}, []
    try:
        for k in range(0, len(items), bs):
            batch = items[k:k + bs]
            t0 = time.time()
            reply = "".join(brain.stream([{"role": "system", "content": SYSTEM}, {"role": "user", "content": _prompt(batch)}],
                                         max_tokens=400, temperature=0.0))
            timing.append(time.time() - t0)
            for it, v in zip(batch, _parse(reply, len(batch))):
                out[it["id"]] = v
            print(f"   {name} {k + len(batch)}/{len(items)}  {timing[-1]:.1f}s", flush=True)
    finally:
        brain.stop()
    return out, {"batch_s": timing}


def score(items, pred):
    by = defaultdict(lambda: [0, 0])
    tp = fp = fn = tn = miss = 0
    for it in items:
        p = pred.get(it["id"])
        if p is None:
            miss += 1
            p = "KEEP"
        ok = (p == "STOP") == (it["intent"] == 1)
        by[it["cls"]][0] += ok
        by[it["cls"]][1] += 1
        if it["intent"] == 1:
            tp += p == "STOP"
            fn += p != "STOP"
        else:
            fp += p == "STOP"
            tn += p != "STOP"
    R = tp / max(1, tp + fn)          # stops when it should
    S = tn / max(1, tn + fp)          # keeps going when it should
    F = 2 * R * S / max(1e-9, R + S)
    return {"R": round(R, 3), "S": round(S, 3), "F": round(F, 3), "acc": round((tp + tn) / len(items), 3),
            "no_answer": miss, "per_class": {c: f"{a}/{n}" for c, (a, n) in sorted(by.items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", default="rules,claude:claude-opus-5-5,mlx:qwen3-14b")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from turn import asr
    items = build_items(asr.BACKEND)
    if a.limit:
        items = items[:a.limit]
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res["_asr"] = asr.BACKEND
    res["_n"] = len(items)
    for j in a.judges.split(","):
        print(f"== {j}", flush=True)
        if j == "rules":
            pred, t = judge_rules(items)
        elif j.startswith("claude:"):
            pred, t = judge_claude(items, j.split(":", 1)[1])
        elif j.startswith("mlx:"):
            pred, t = judge_mlx(items, j.split(":", 1)[1])
        else:
            continue
        res[j] = {**score(items, pred), **t, "pred": pred}
        print("  ", {k: v for k, v in res[j].items() if k not in ("pred", "batch_s")}, flush=True)
    res["_items"] = items
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
