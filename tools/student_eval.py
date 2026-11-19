"""Compare meaning students (sem.onnx) on text only, so it runs anywhere (cloud CPU included, no recogniser).

Two test sets, neither used for training:
  * held-out bank: analysis/llm_judge.json items of sets dev, dev-aug, synth-test. The caller text is what Cohere
    really heard on the Mac, so the student is judged on real recogniser output.
  * hard suite: data/hard_text_v1.json (tools/hard_bank.py), the live-demo failure categories (one-word
    acknowledgements, «طيب بس…», thinking aloud, room talk, echo, recogniser junk).

    python -m tools.student_eval --base models/student-base --cand models/student-v3 [--out runs/student-eval.json]
Exit code 1 when the candidate fails the gate (see GATE), so scripts can stop there.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parents[1]
HELD_OUT = ("dev", "dev-aug", "synth-test")
MAX_LEN = 64
# The candidate replaces the official student only if, on held-out real-recogniser text, it stops for no more
# acknowledgements than today (false stops are what broke #42) and it is better overall, and on the hard suite it
# makes fewer mistakes.
GATE = {"max_S_drop": 0.0, "min_F_gain": 0.005, "hard_errors": "fewer"}


class Student:
    def __init__(self, folder: Path):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.sess = ort.InferenceSession(str(folder / "sem.onnx"), providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tok.enable_truncation(MAX_LEN)
        self.tok.no_padding()
        self.tt = "token_type_ids" in {i.name for i in self.sess.get_inputs()}

    def p_stop(self, agent: str, caller: str) -> float:
        if not caller.strip():
            return 0.0
        enc = self.tok.encode(" ".join(agent.split()[-24:]), caller.strip())   # same input as turn/semantic.py
        ids = np.array([enc.ids], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
        if self.tt:
            feeds["token_type_ids"] = np.array([enc.type_ids], dtype=np.int64)
        z = self.sess.run(None, feeds)[0][0]
        e = np.exp(z - z.max())
        return float(e[1] / e.sum())


def metrics(p: np.ndarray, y: np.ndarray, th: float = 0.5) -> dict:
    R = float(((p > th) & (y == 1)).sum() / max(1, (y == 1).sum()))
    S = float(((p <= th) & (y == 0)).sum() / max(1, (y == 0).sum()))
    return {"R": round(R, 3), "S": round(S, 3), "F": round(2 * R * S / max(1e-9, R + S), 3), "n": int(len(y)),
            "false_stops": int(((p > th) & (y == 0)).sum()), "misses": int(((p <= th) & (y == 1)).sum())}


def load_sets() -> dict[str, list[dict]]:
    bank = json.loads((KIT / "analysis" / "llm_judge.json").read_text(encoding="utf-8"))["_items"]
    out = {"held_out": [dict(agent=b["agent"], text=b["text"], y=int(b["intent"]), cat=b.get("cls", "")) for b in bank
                        if b["set"] in HELD_OUT and b.get("intent") is not None]}
    hard = KIT / "data" / "hard_text_v1.json"
    if hard.exists():
        out["hard"] = [dict(agent=h["agent"], text=h["caller"], y=int(h["label"] == "STOP"), cat=h["kind"])
                       for h in json.loads(hard.read_text(encoding="utf-8"))]
    return out


def evaluate(student: Student, items: list[dict]) -> dict:
    p = np.array([student.p_stop(i["agent"], i["text"]) for i in items])
    y = np.array([i["y"] for i in items])
    res = metrics(p, y)
    by = defaultdict(list)
    for k, i in enumerate(items):
        by[i["cat"]].append(k)
    res["by_category"] = {c: metrics(p[ix], y[ix]) for c, ix in sorted(by.items())}
    res["errors"] = [{"agent": items[k]["agent"][-60:], "text": items[k]["text"], "y": int(y[k]), "p": round(float(p[k]), 3),
                      "cat": items[k]["cat"]} for k in range(len(items)) if (p[k] > 0.5) != (y[k] == 1)]
    return res


def gate(base: dict, cand: dict) -> tuple[bool, list[str]]:
    why = []
    b, c = base["held_out"], cand["held_out"]
    if c["S"] < b["S"] - GATE["max_S_drop"]:
        why.append(f"held-out S fell {b['S']} -> {c['S']} (more false stops)")
    if c["F"] < b["F"] + GATE["min_F_gain"]:
        why.append(f"held-out F not better by {GATE['min_F_gain']}: {b['F']} -> {c['F']}")
    if "hard" in base and len(cand["hard"]["errors"]) >= len(base["hard"]["errors"]):
        why.append(f"hard suite errors not fewer: {len(base['hard']['errors'])} -> {len(cand['hard']['errors'])}")
    return not why, why


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(KIT / "models" / "student"))
    ap.add_argument("--cand", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    sets = load_sets()
    report = {}
    for name, folder in (("base", a.base), ("cand", a.cand)):
        if folder:
            s = Student(Path(folder))
            report[name] = {k: evaluate(s, v) for k, v in sets.items()}
    for name, r in report.items():
        line = " · ".join(f"{k}: R {v['R']} S {v['S']} F {v['F']} (false stops {v['false_stops']}, misses {v['misses']}, n {v['n']})"
                          for k, v in r.items())
        print(f"{name:5s} {line}")
    ok = True
    if "cand" in report:
        ok, why = gate(report["base"], report["cand"])
        report["gate"] = {"pass": ok, "why": why, "rule": GATE}
        print("GATE", "PASS" if ok else "FAIL", *why, sep="\n  ")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
