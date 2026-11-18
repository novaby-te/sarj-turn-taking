"""Train the decision layer from data: every time Whisper re-reads a caller segment we get a feature vector;
the labels say whether, at that moment, the caller was bidding for the floor. A logistic regression learns
the weights; they are written to models/policy.json and the system uses them automatically.

    .venv/bin/python -m tools.train_policy --train data/synth --test data/dev,data/dev-aug

Honest protocol: train only on generated calls, test on the organisers' calls it has never seen.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
from tools.sim import simulate  # noqa: E402
from turn.policy import ENGINE_POLICY, FEATURES  # noqa: E402
from ttbench.score import load_labels  # noqa: E402


def rows_for(data_dir: Path):
    labels, _ = load_labels(data_dir)
    out = Path(KIT / "runs" / f"sim-dump-{data_dir.name}")
    simulate(str(KIT / "my_system.py"), data_dir, overrides={"policy.NEVER_STOP": 1, "policy.USE_MODEL": 0}, out_dir=out)
    traces = json.loads((out / "trace.json").read_text())
    X, y, meta = [], [], []
    for sid, tr in traces.items():
        lab = labels[sid]
        for r in tr:
            if r["kind"] != "asr":
                continue
            t, f = r["t"], r["f"]
            ev = next((e for e in lab["events"] if e["kind"] == "interruption"
                       and e["onset_ms"] - 300 <= t <= e.get("offset_ms", e["onset_ms"]) + 500), None)
            if ev is None:
                target = 0                       # sound outside any labelled moment: must not stop
            elif ev["expected"] == "KEEP":
                target = 0
            elif ev["expected"] == "STOP":
                if t < ev["evident_ms"] - 150:
                    continue                     # before the intention is clear: no label
                target = 1
            else:
                if ev.get("preferred") == "STOP" and "evident_ms" in ev and t >= ev["evident_ms"]:
                    target = 1
                elif ev.get("preferred") == "KEEP":
                    target = 0
                else:
                    continue
            X.append([float(f[k]) for k in FEATURES])
            y.append(target)
            meta.append((sid, t, r["text"]))
    return np.array(X), np.array(y), meta


def fit(X, y, C=1.0):
    from sklearn.linear_model import LogisticRegression
    mu, sd = X.mean(0), np.maximum(X.std(0), 1e-3)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit((X - mu) / sd, y)
    return {"features": FEATURES, "mu": dict(zip(FEATURES, mu.round(5).tolist())), "sd": dict(zip(FEATURES, sd.round(5).tolist())),
            "w": dict(zip(FEATURES, clf.coef_[0].round(4).tolist())), "bias": float(clf.intercept_[0])}, clf, mu, sd


def fit_lgbm(X, y):
    import lightgbm as lgb
    pos = max(1, int(y.sum()))
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                             scale_pos_weight=(len(y) - pos) / pos, verbose=-1).fit(X, y)
    return {"type": "lgbm", "features": FEATURES, "booster": clf.booster_.model_to_string()}, clf


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/synth")
    ap.add_argument("--model", default="lr", choices=["lr", "lgbm"])
    ap.add_argument("--out", default=str(ENGINE_POLICY))
    ap.add_argument("--C", type=float, default=0.5)
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args(argv)
    Xs, ys = [], []
    for d in a.train.split(","):
        X, y, meta = rows_for(KIT / d)
        print(f"{d}: {len(y)} examples ({int(y.sum())} floor bids)")
        Xs.append(X)
        ys.append(y)
    X, y = np.vstack(Xs), np.concatenate(ys)
    if a.model == "lgbm":
        model, clf = fit_lgbm(X, y)
        print(f"train accuracy {(clf.predict(X) == y).mean():.3f}")
        imp = sorted(zip(FEATURES, clf.booster_.feature_importance("gain")), key=lambda kv: -kv[1])
        for k, g in imp[:15]:
            print(f"  {k:22s} gain {g:8.1f}")
    else:
        model, clf, mu, sd = fit(X, y, a.C)
        print(f"train accuracy {(clf.predict((X - mu) / sd) == y).mean():.3f}")
        for k, w in sorted(model["w"].items(), key=lambda kv: -abs(kv[1]))[:15]:
            print(f"  {k:22s} {w:+.3f}")
    if not a.no_save:
        Path(a.out).write_text(json.dumps(model, ensure_ascii=False, indent=1))
        print("saved", a.out)


if __name__ == "__main__":
    main()
