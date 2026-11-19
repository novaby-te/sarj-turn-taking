"""Train the MEANING student (turn/semantic.py): CAMeLBERT-DA fine-tuned to copy Claude's STOP/KEEP judgement.

Training data
  * data/teacher/raw/*.json  - thousands of caller moments written + labelled by Claude Opus (tools/teacher_gen.py)
  * the real speech-recognition output of our generated TRAINING calls (set "synth"), so it also learns how Cohere
    actually writes what it hears
  Each STOP moment also teaches its growing prefixes from the word where a listener knows («لا لا» -> STOP); each
  KEEP moment teaches its prefixes too. Spelling noise like the recogniser's (hamza, ة/ه, glued words) is added.
Validation = teacher files never trained on (whole dialect x domain calls). Test = the 381 labelled moments of the
segment bank from sets the student never saw (dev, dev-aug, synth-test) + all 381 for comparison with Claude.

    .venv/bin/python -m tools.train_student            # -> models/student/{sem.onnx, tokenizer.json, meta.json}

v3 options (both off by default, so the official recipe is unchanged):
  --agent-trunc P   with probability P a training row sees only the first words of the agent sentence. At run time
                    turn/policy.py passes heard_context(): the previous sentence + the part spoken SO FAR, not the
                    end of a sentence the caller has not heard yet. This closes that train/run gap.
  --anchor DIR      also learn the probabilities of an existing student (e.g. models/student-base) on every row:
                    loss = (1-W)*CE(teacher label) + W*CE(anchor probability), W = --anchor-w. Keeps what the
                    official student knows (its teacher data is not published) while adding the new teacher data,
                    so the new student does not drift into extra false stops (the failure of EXPERIMENTS #42).
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
from turn.config import EXT_DIR  # noqa: E402

EXT = EXT_DIR
BASES = {"camelbert-da": EXT / "camelbert-da", "saudibert": EXT / "saudibert"}
BASE = BASES["camelbert-da"]
OUT = KIT / "models" / "student"
MAX_LEN = 64
rng = random.Random(11)


def noisy(t: str) -> str:
    """Spelling noise like the recogniser's."""
    ops = [lambda s: re.sub("[أإآ]", "ا", s), lambda s: s.replace("ة", "ه"), lambda s: s.replace("ى", "ي"),
           lambda s: re.sub(r"(\S) (\S)", r"\1\2", s, count=1), lambda s: s.replace("ال", "", 1) if len(s) > 8 else s]
    for op in rng.sample(ops, rng.randint(1, 2)):
        t = op(t)
    return t


def trunc_agent(agent: str) -> str:
    """The agent sentence as heard when the caller cuts in: its first k words (k >= 3)."""
    w = agent.split()
    return " ".join(w[:rng.randint(3, len(w))]) if len(w) > 4 else agent


def teacher_rows() -> tuple[list, list]:
    tr, va = [], []
    for p in sorted((KIT / "data" / "teacher" / "raw").glob("*.json")):
        dest = va if int(p.stem) % 10 == 0 else tr
        for x in json.loads(p.read_text()):
            agent, caller, y = x.get("agent", ""), x["caller"].strip(), int(x["label"] == "STOP")
            words = caller.split()
            if not words:
                continue
            dest.append((agent, caller, y, x.get("kind", "")))
            if dest is va:
                continue
            if y == 1:
                cue = int(x.get("cue") or 1)
                for k in range(max(1, cue), len(words)):          # prefixes once the bid is evident
                    if rng.random() < 0.5:
                        dest.append((agent, " ".join(words[:k]), 1, "bid_prefix"))
            elif len(words) > 1 and rng.random() < 0.6:
                dest.append((agent, " ".join(words[:rng.randint(1, len(words) - 1)]), 0, x.get("kind", "") + "_prefix"))
            if rng.random() < 0.35:
                dest.append((agent, noisy(caller), y, x.get("kind", "") + "_noisy"))
    return tr, va


def bank_rows() -> list[dict]:
    p = KIT / "analysis" / "llm_judge.json"
    return json.loads(p.read_text())["_items"] if p.exists() else []


def main():
    global BASE, OUT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="camelbert-da", choices=list(BASES))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--from-ckpt", action="store_true", help="skip training, export the saved checkpoint")
    ap.add_argument("--agent-trunc", type=float, default=0.0, help="share of rows with a truncated agent sentence")
    ap.add_argument("--anchor", default=None, help="student folder whose probabilities are also learned")
    ap.add_argument("--anchor-w", type=float, default=0.5)
    ap.add_argument("--anchor-mode", default="all", choices=["all", "agree"],
                    help="agree: learn the anchor only where it agrees with the teacher label (the official student "
                         "misses most openers and questions; copying it there would copy that bias)")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--quant", default="dynamic", choices=["dynamic", "dynamic_pc", "none"],
                    help="dynamic: int8 as before. dynamic_pc: per-channel int8 (closer to fp32). none: ship fp32 "
                         "(about 4x larger and slower, but exactly what was trained)")
    ap.add_argument("--keep-fp32", action="store_true", help="also keep sem_fp32.onnx next to sem.onnx")
    a = ap.parse_args()
    BASE, OUT = BASES[a.base], Path(a.out)
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tr, va = teacher_rows()
    bank = bank_rows()
    tr += [(b["agent"], b["text"], int(b["intent"]), "real_asr") for b in bank if b["set"] == "synth" and b["text"].strip()]
    tr += [(b["agent"], b["text"], int(b["intent"]), "real_asr") for b in bank if b["set"] == "synth" and b["text"].strip()]
    if a.agent_trunc > 0:
        tr += [(trunc_agent(r[0]), r[1], r[2], r[3] + "_trunc") for r in tr if rng.random() < a.agent_trunc]
    rng.shuffle(tr)
    soft = None
    if a.anchor:                                                    # anchor probabilities, same input as at run time
        from tools.student_eval import Student
        anc = Student(Path(a.anchor))
        t1 = time.time()
        soft = [anc.p_stop(r[0], r[1]) for r in tr]
        if a.anchor_mode == "agree":                                 # disagreement: fall back to the teacher label
            n_dis = sum((q > 0.5) != (r[2] == 1) for q, r in zip(soft, tr))
            soft = [q if (q > 0.5) == (r[2] == 1) else float(r[2]) for q, r in zip(soft, tr)]
            print(f"anchor agree-only: {n_dis} disagreeing rows use the teacher label", flush=True)
        print(f"anchor {a.anchor}: {len(soft)} rows in {time.time() - t1:.0f}s, w={a.anchor_w}", flush=True)
    print(f"train {len(tr)} (STOP {sum(r[2] for r in tr)}) · val {len(va)}", flush=True)

    tok = AutoTokenizer.from_pretrained(str(BASE), use_fast=True)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = EXT / "student-ckpt" / f"{a.base}-{OUT.name}"
    src = ckpt if a.from_ckpt else BASE
    model = AutoModelForSequenceClassification.from_pretrained(str(src), num_labels=2, use_safetensors=True).to(dev)

    def batches(rows, bs, shuffle):
        idx = list(range(len(rows)))
        if shuffle:
            rng.shuffle(idx)
        for i in range(0, len(idx), bs):
            b = [rows[j] for j in idx[i:i + bs]]
            enc = tok([" ".join(r[0].split()[-24:]) or "." for r in b], [r[1] for r in b],
                      truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
            y = torch.tensor([r[2] for r in b], device=dev)
            if soft is not None and shuffle:
                yield {k: v.to(dev) for k, v in enc.items()}, (y, torch.tensor([soft[j] for j in idx[i:i + bs]], device=dev))
            else:
                yield {k: v.to(dev) for k, v in enc.items()}, y

    def evaluate(rows):
        model.eval()
        ps = []
        with torch.no_grad():
            for x, _ in batches(rows, 128, False):
                ps.append(torch.softmax(model(**x).logits, -1)[:, 1].float().cpu().numpy())
        model.train()
        return np.concatenate(ps) if ps else np.zeros(0)

    epochs, bs, lr = (0 if a.from_ckpt else a.epochs), 32, a.lr
    steps = max(1, epochs * ((len(tr) + bs - 1) // bs))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / (0.06 * steps)) * max(0.0, (steps - s) / steps))
    model.train()
    t0, step = time.time(), 0
    pv, yv = evaluate(va), np.array([r[2] for r in va])
    for ep in range(epochs):
        for x, y in batches(tr, bs, True):
            if isinstance(y, tuple):                                # hard teacher label + anchor probability
                y, q = y
                logp = torch.log_softmax(model(**x).logits, -1)
                hard = torch.nn.functional.nll_loss(logp, y)
                anchor = -(q * logp[:, 1] + (1 - q) * logp[:, 0]).mean()
                loss = (1 - a.anchor_w) * hard + a.anchor_w * anchor
            else:
                loss = torch.nn.functional.cross_entropy(model(**x).logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 100 == 0:
                print(f"  ep {ep} step {step}/{steps} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
        pv = evaluate(va)
        yv = np.array([r[2] for r in va])
        vR = ((pv > 0.5) & (yv == 1)).sum() / max(1, (yv == 1).sum())
        vS = ((pv <= 0.5) & (yv == 0)).sum() / max(1, (yv == 0).sum())
        print(f"  epoch {ep}: val acc {((pv > 0.5) == yv).mean():.3f} R {vR:.3f} S {vS:.3f}", flush=True)

    # ---------------- export: ONNX fp32 -> int8, tokenizer.json, check they agree
    OUT.mkdir(parents=True, exist_ok=True)
    model = model.to("cpu").eval()
    if not a.from_ckpt:                                             # keep the trained weights outside the repo
        model.save_pretrained(str(ckpt), safe_serialization=True)
        tok.save_pretrained(str(ckpt))
    tok.save_pretrained(str(OUT / "hf_tok"))
    (OUT / "tokenizer.json").write_text((OUT / "hf_tok" / "tokenizer.json").read_text())
    ex = tok(["حياك الله"], ["إيه"], return_tensors="pt")
    if "token_type_ids" not in ex:
        import torch as _t
        ex["token_type_ids"] = _t.zeros_like(ex["input_ids"])
    fp32 = OUT / "sem_fp32.onnx"

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.m(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits

    args = (ex["input_ids"], ex["attention_mask"], ex["token_type_ids"])
    names = ["input_ids", "attention_mask", "token_type_ids"]
    dyn = {n: {0: "b", 1: "t"} for n in names}
    try:
        torch.onnx.export(Wrap(model), args, str(fp32), input_names=names, output_names=["logits"],
                          dynamic_axes=dyn | {"logits": {0: "b"}}, opset_version=17, dynamo=False)
    except TypeError:
        torch.onnx.export(Wrap(model), args, str(fp32), input_names=names, output_names=["logits"],
                          dynamic_axes=dyn | {"logits": {0: "b"}}, opset_version=17)
    from onnxruntime.quantization import QuantType, quantize_dynamic
    if a.quant == "none":
        import shutil
        shutil.copy(fp32, OUT / "sem.onnx")
    else:
        quantize_dynamic(str(fp32), str(OUT / "sem.onnx"), weight_type=QuantType.QInt8, per_channel=a.quant == "dynamic_pc")
    if not a.keep_fp32:
        fp32.unlink()
    for f in (OUT / "hf_tok").glob("*"):
        f.unlink()
    (OUT / "hf_tok").rmdir()

    # ---------------- test on the segment bank (never trained on except set "synth") with the real runtime
    import importlib
    import os
    os.environ["TT_SEM_DIR"] = str(OUT)
    from turn import semantic
    importlib.reload(semantic)
    t1 = time.perf_counter()
    sp = [semantic.p_stop(b["agent"], b["text"]) for b in bank]
    ms = (time.perf_counter() - t1) * 1000 / max(1, len(bank))
    res = {"val_acc": float(((pv > 0.5) == yv).mean()), "ms_per_call": round(ms, 2), "n_train": len(tr),
           "recipe": {"epochs": a.epochs, "lr": a.lr, "agent_trunc": a.agent_trunc, "anchor": a.anchor,
                      "anchor_w": a.anchor_w if a.anchor else None,
                      "anchor_mode": a.anchor_mode if a.anchor else None, "quant": a.quant}}
    for name, sets in (("held_out", ("dev", "dev-aug", "synth-test")), ("all", ("dev", "dev-aug", "synth-test", "synth"))):
        idx = [i for i, b in enumerate(bank) if b["set"] in sets]
        y = np.array([bank[i]["intent"] for i in idx])
        p = np.array([0.0 if sp[i] is None else sp[i] for i in idx])
        R = float(((p > 0.5) & (y == 1)).sum() / max(1, (y == 1).sum()))
        S = float(((p <= 0.5) & (y == 0)).sum() / max(1, (y == 0).sum()))
        res[name] = {"R": round(R, 3), "S": round(S, 3), "F": round(2 * R * S / max(1e-9, R + S), 3), "n": len(idx)}
    (OUT / "meta.json").write_text(json.dumps({"base": a.base,
                                               "teacher": "claude-opus-5-5", "max_len": MAX_LEN, **res}, ensure_ascii=False, indent=1))
    print("RESULT", json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
