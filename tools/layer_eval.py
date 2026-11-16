"""Score every LEVEL of the hierarchy on its own, on the labelled segment bank (tools/segments.py).

For each segment we compute all cues (acoustic levels 1-3, voiceprint, VAD, words) and measure how well each cue
separates the classes each level is responsible for (AUC: 0.5 = useless, 1.0 = perfect). Training set (synth)
and held-out sets (dev-aug, synth-test, dev) are reported separately, so a cue that only works on our own
generated voices is caught. Then a small learned model per level is trained on synth and tested on held-out data.

    .venv/bin/python -m tools.layer_eval            # features are cached in analysis/segfeat.jsonl
    .venv/bin/python -m tools.layer_eval --refresh
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
import os  # noqa: E402
os.environ.setdefault("TT_SIM", "1")
from ttbench.dataset import read_wav_stereo  # noqa: E402
from turn import acoustic, lexicon  # noqa: E402

ACOUSTIC = ["level_rel", "dyn_rel", "hf_rel", "tilt_rel", "centroid_rel", "decay_rel", "per_rel", "f0_rel", "f0_range",
            "f0_slope", "rate_rel", "voiced", "dur", "far_score", "arousal"]
OTHER = ["spk_sim", "vad_frac", "asr_bid", "asr_bc", "asr_content", "asr_n", "asr_logprob", "asr_echo", "asr_halluc"]
TASKS = {   # level → (name, positive selector, negative selector)
    "L2 هل هو صوت العميل؟": (lambda s: s["same"] == 1 and s["cls"] != "answer", lambda s: s["same"] == 0),
    "L2 قريب ولا بعيد؟": (lambda s: s["near"] == 1 and s["cls"] != "answer", lambda s: s["near"] == 0),
    "L2 كلام ولا صوت بلا كلام؟": (lambda s: s["cls"] in ("bid", "bc", "filler"), lambda s: s["cls"] == "nonspeech"),
    "L3/L4 طلب الدور ولا تأييد (من نفس العميل)": (lambda s: s["cls"] == "bid", lambda s: s["cls"] in ("bc", "filler")),
    "L5 يسكت ولا يكمّل (كل الحالات)": (lambda s: s["intent"] == 1, lambda s: s["intent"] == 0),
}


def compute(segs: list[dict]) -> list[dict]:
    from turn import asr, speaker
    from turn.vad import SileroVAD
    asr.warmup()
    vad = SileroVAD()
    audio_cache: dict = {}
    u0_cache: dict = {}
    out = []
    t0 = time.time()
    for i, s in enumerate(segs):
        key = (s["set"], s["sid"])
        if key not in audio_cache:
            audio_cache.clear()
            lab = json.loads((KIT / "data" / s["set"] / "labels" / f"{s['sid']}.json").read_text())
            audio_cache[key] = (read_wav_stereo(KIT / "data" / s["set"] / "audio" / lab["audio"]["file"])[:, 1], lab)
        c, lab = audio_cache[key]
        x = c[int(s["start"] * 16):int(s["end"] * 16)]
        if len(x) < 1600:
            continue
        if key not in u0_cache:
            u = c[int(s["u0"][0] * 16):int(s["u0"][1] * 16)] if s["u0"] else None
            u0_cache[key] = (acoustic.segment_profile(u) if u is not None and len(u) > 1600 else None,
                             speaker.embed(u) if u is not None else None)
        ref, ref_emb = u0_cache[key]
        f = acoustic.relative(acoustic.segment_profile(x), ref)
        e = speaker.embed(x[-16000 * 3:])
        f["spk_sim"] = float(np.dot(e, ref_emb)) if e is not None and ref_emb is not None else 0.6
        vad.reset()
        probs = [vad.push(x[k:k + 320]) for k in range(0, len(x) - 320, 320)]
        f["vad_frac"] = float(np.mean(np.array(probs) > 0.5)) if probs else 0.0
        r = asr.transcribe(x[:16000 * 9])
        agent = " ".join(u["text"] for u in lab["agent_utterances"])
        a = lexicon.analyse(r.get("text", ""), agent)
        f.update(asr_bid=a["bid"] + int(a["repeated_ambig"]), asr_bc=a["bc"], asr_content=a["content"], asr_n=a["n"],
                 asr_logprob=r.get("logprob", -1.0), asr_echo=a["echoed"] / max(1, a["n"]), asr_halluc=int(a["hallucination"]),
                 asr_text=r.get("text", ""))
        out.append({**s, **{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in f.items()}})
        if i % 100 == 0:
            print(f"  {i}/{len(segs)}  {time.time() - t0:.0f}s", flush=True)
    return out


def auc(pos, neg):
    from sklearn.metrics import roc_auc_score
    if len(pos) < 3 or len(neg) < 3:
        return None
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    v = np.r_[pos, neg]
    a = roc_auc_score(y, v)
    return max(a, 1 - a), ("+" if a >= 0.5 else "−")       # direction: + means higher value → positive class


def report(rows: list[dict]) -> dict:
    from sklearn.linear_model import LogisticRegression
    res = {}
    feats = ACOUSTIC + OTHER
    for task, (P, N) in TASKS.items():
        res[task] = {"cues": {}, "learned": {}}
        for sets, tag in ((["synth"], "train"), (["dev-aug", "synth-test", "dev"], "held_out")):
            sub = [r for r in rows if r["set"] in sets]
            pos = [r for r in sub if P(r)]
            neg = [r for r in sub if N(r)]
            res[task]["n_" + tag] = (len(pos), len(neg))
            for f in feats:
                a = auc([r[f] or 0 for r in pos], [r[f] or 0 for r in neg])
                if a:
                    res[task]["cues"].setdefault(f, {})[tag] = a
        # a small learned model for this level: trained on synth, tested on held-out
        tr = [r for r in rows if r["set"] == "synth" and (P(r) or N(r))]
        te = [r for r in rows if r["set"] != "synth" and (P(r) or N(r))]
        if len({P(r) for r in tr}) == 2 and len({P(r) for r in te}) == 2:
            for name, fs in (("acoustic only", ACOUSTIC + ["spk_sim", "vad_frac"]), ("words only", [f for f in OTHER if f.startswith("asr")]),
                             ("all levels", feats)):
                X = np.array([[r[f] or 0 for f in fs] for r in tr], float)
                y = np.array([P(r) for r in tr], int)
                mu, sd = X.mean(0), X.std(0) + 1e-6
                clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000).fit((X - mu) / sd, y)
                Xt = np.array([[r[f] or 0 for f in fs] for r in te], float)
                yt = np.array([P(r) for r in te], int)
                pr = clf.predict_proba((Xt - mu) / sd)[:, 1]
                from sklearn.metrics import roc_auc_score
                res[task]["learned"][name] = {"auc": round(float(roc_auc_score(yt, pr)), 3),
                                              "acc": round(float(((pr > 0.5) == yt).mean()), 3)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    cache = KIT / "analysis" / "segfeat.jsonl"
    if a.refresh or not cache.exists():
        segs = [json.loads(l) for l in (KIT / "analysis" / "segments.jsonl").read_text().splitlines() if l.strip()]
        segs.sort(key=lambda s: (s["set"], s["sid"]))
        rows = compute(segs)
        cache.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    rows = [json.loads(l) for l in cache.read_text().splitlines() if l.strip()]
    res = report(rows)
    (KIT / "analysis" / "layer_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for task, r in res.items():
        print(f"\n== {task}   (train pos/neg {r['n_train']}, held-out pos/neg {r['n_held_out']})")
        top = sorted(r["cues"].items(), key=lambda kv: -(kv[1].get("held_out", (0, ""))[0]))[:8]
        for f, v in top:
            tr, ho = v.get("train", (0, "")), v.get("held_out", (0, ""))
            print(f"   {f:14s} train {tr[0]:.2f}{tr[1]}   held-out {ho[0]:.2f}{ho[1]}")
        for name, m in r["learned"].items():
            print(f"   learned [{name}]  held-out AUC {m['auc']}  acc {m['acc']}")


if __name__ == "__main__":
    main()
