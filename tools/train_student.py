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
BASE = Path.home() / "Downloads" / "sarj-ext-data" / "camelbert-da"
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
    import torch
    from transformers import BertForSequenceClassification, BertTokenizerFast
    tr, va = teacher_rows()
    bank = bank_rows()
    tr += [(b["agent"], b["text"], int(b["intent"]), "real_asr") for b in bank if b["set"] == "synth" and b["text"].strip()]
    tr += [(b["agent"], b["text"], int(b["intent"]), "real_asr") for b in bank if b["set"] == "synth" and b["text"].strip()]
    rng.shuffle(tr)
    print(f"train {len(tr)} (STOP {sum(r[2] for r in tr)}) · val {len(va)}", flush=True)

    tok = BertTokenizerFast(vocab_file=str(BASE / "vocab.txt"), do_lower_case=False, tokenize_chinese_chars=True)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(str(BASE), num_labels=2, use_safetensors=True).to(dev)

    def batches(rows, bs, shuffle):
        idx = list(range(len(rows)))
        if shuffle:
            rng.shuffle(idx)
        for i in range(0, len(idx), bs):
            b = [rows[j] for j in idx[i:i + bs]]
            enc = tok([" ".join(r[0].split()[-24:]) or "." for r in b], [r[1] for r in b],
                      truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
            yield {k: v.to(dev) for k, v in enc.items()}, torch.tensor([r[2] for r in b], device=dev)

    def evaluate(rows):
        model.eval()
        ps = []
        with torch.no_grad():
            for x, _ in batches(rows, 128, False):
                ps.append(torch.softmax(model(**x).logits, -1)[:, 1].float().cpu().numpy())
        model.train()
        return np.concatenate(ps) if ps else np.zeros(0)

    epochs, bs, lr = 3, 32, 3e-5
    steps = epochs * ((len(tr) + bs - 1) // bs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / (0.06 * steps)) * max(0.0, (steps - s) / steps))
    model.train()
    t0, step = time.time(), 0
    for ep in range(epochs):
        for x, y in batches(tr, bs, True):
            loss = torch.nn.functional.cross_entropy(model(**x).logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 100 == 0:
                print(f"  ep {ep} step {step}/{steps} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
        pv = evaluate(va)
        yv = np.array([r[2] for r in va])
        print(f"  epoch {ep}: val acc {((pv > 0.5) == yv).mean():.3f}", flush=True)

    # ---------------- export: ONNX fp32 -> int8, tokenizer.json, check they agree
    OUT.mkdir(parents=True, exist_ok=True)
    model = model.to("cpu").eval()
    tok.save_pretrained(str(OUT / "hf_tok"))
    (OUT / "tokenizer.json").write_text((OUT / "hf_tok" / "tokenizer.json").read_text())
    ex = tok(["حياك الله"], ["إيه"], return_tensors="pt")
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
    quantize_dynamic(str(fp32), str(OUT / "sem.onnx"), weight_type=QuantType.QInt8)
    fp32.unlink()
    for f in (OUT / "hf_tok").glob("*"):
        f.unlink()
    (OUT / "hf_tok").rmdir()

    # ---------------- test on the segment bank (never trained on except set "synth") with the real runtime
    import importlib
    from turn import semantic
    importlib.reload(semantic)
    t1 = time.perf_counter()
    sp = [semantic.p_stop(b["agent"], b["text"]) for b in bank]
    ms = (time.perf_counter() - t1) * 1000 / max(1, len(bank))
    res = {"val_acc": float(((pv > 0.5) == yv).mean()), "ms_per_call": round(ms, 2), "n_train": len(tr)}
    for name, sets in (("held_out", ("dev", "dev-aug", "synth-test")), ("all", ("dev", "dev-aug", "synth-test", "synth"))):
        idx = [i for i, b in enumerate(bank) if b["set"] in sets]
        y = np.array([bank[i]["intent"] for i in idx])
        p = np.array([0.0 if sp[i] is None else sp[i] for i in idx])
        R = float(((p > 0.5) & (y == 1)).sum() / max(1, (y == 1).sum()))
        S = float(((p <= 0.5) & (y == 0)).sum() / max(1, (y == 0).sum()))
        res[name] = {"R": round(R, 3), "S": round(S, 3), "F": round(2 * R * S / max(1e-9, R + S), 3), "n": len(idx)}
    (OUT / "meta.json").write_text(json.dumps({"base": "CAMeL-Lab/bert-base-arabic-camelbert-da (Apache-2.0)",
                                               "teacher": "claude-opus-5-5", "max_len": MAX_LEN, **res}, ensure_ascii=False, indent=1))
    print("RESULT", json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
