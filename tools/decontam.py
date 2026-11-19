"""Keep test sentences out of training data (CLAUDE.md rule 3), automatically.

Test texts: every user segment of the test sets (novel-test, novel-test-2, hard-test if present), the blind banks
(data/novel_bank*.json) and the hard text suite (data/hard_text_v1.json). A training item is dropped when, after
normalising spelling, it equals a test text or shares most of its character 3-grams with one (Jaccard ≥ 0.7).
Items shorter than 3 words are never dropped: «إيه تمام» is everyday Arabic, not a copied sentence.

    python -m tools.decontam                       # cleans data/teacher/raw/*.json in place, report in runs/
    python -m tools.decontam --check FILE.json     # only report
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
JACCARD = 0.7
MIN_WORDS = 3


def norm(t: str) -> str:
    t = re.sub(r"[ً-ْـ]", "", t)                 # diacritics, tatweel
    t = re.sub("[أإآ]", "ا", t).replace("ة", "ه").replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    t = re.sub(r"\*", "", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def grams(t: str) -> set:
    s = f" {t} "
    return {s[i:i + 3] for i in range(len(s) - 2)}


def _strings(x) -> list[str]:
    if isinstance(x, str):
        return [x]
    if isinstance(x, list):
        return [s for v in x for s in _strings(v)]
    if isinstance(x, dict):
        return [s for v in x.values() for s in _strings(v)]
    return []


def test_texts() -> list[str]:
    out = []
    for d in ("novel-test", "novel-test-2", "hard-test"):
        for p in (KIT / "data" / d / "labels").glob("*.json"):
            lab = json.loads(p.read_text(encoding="utf-8"))
            out += [s.get("text", "") for s in lab.get("user_segments", [])]
    for p in list((KIT / "data").glob("novel_bank*.json")) + [KIT / "data" / "hard_bank_v1.json"]:
        if p.exists():
            out += _strings(json.loads(p.read_text(encoding="utf-8")))
    hard = KIT / "data" / "hard_text_v1.json"
    if hard.exists():
        out += [h["caller"] for h in json.loads(hard.read_text(encoding="utf-8"))]
    return [t for t in {norm(t) for t in out} if len(t.split()) >= MIN_WORDS]


class Index:
    def __init__(self, texts: list[str]):
        self.texts, self.g = texts, [grams(t) for t in texts]
        self.exact = set(texts)
        self.inv = defaultdict(set)
        for k, g in enumerate(self.g):
            for x in g:
                self.inv[x].add(k)

    def hit(self, text: str) -> str | None:
        t = norm(text)
        if len(t.split()) < MIN_WORDS:
            return None
        if t in self.exact:
            return t
        g = grams(t)
        counts = defaultdict(int)
        for x in g:
            for k in self.inv.get(x, ()):
                counts[k] += 1
        for k, c in counts.items():
            if c / (len(g) + len(self.g[k]) - c) >= JACCARD:
                return self.texts[k]
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(KIT / "data" / "teacher" / "raw"))
    ap.add_argument("--check", default=None, help="a single JSON list of items with a 'caller' field; report only")
    a = ap.parse_args(argv)
    idx = Index(test_texts())
    print(f"{len(idx.texts)} test texts indexed")
    files = [Path(a.check)] if a.check else sorted(Path(a.dir).glob("*.json"))
    dropped = []
    for p in files:
        items = json.loads(p.read_text(encoding="utf-8"))
        keep = []
        for x in items:
            h = idx.hit(x.get("caller", ""))
            if h:
                dropped.append({"file": p.name, "caller": x["caller"], "matches": h})
            else:
                keep.append(x)
        if not a.check and len(keep) != len(items):
            p.write_text(json.dumps(keep, ensure_ascii=False), encoding="utf-8")
    rep = KIT / "runs" / "decontam-report.json"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(dropped, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"dropped {len(dropped)} items from {len(files)} files (report: {rep})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
