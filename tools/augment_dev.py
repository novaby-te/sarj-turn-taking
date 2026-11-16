"""Stress-test copies of the development calls: same words and labels, harsher line conditions.

For each dev call, variants: phone line + street, quieter caller + car, speakerphone echo + household,
and a louder background ESC-50 event. Labels are unchanged (the caller says the same thing), so any score drop
is pure robustness loss.

    .venv/bin/python -m tools.augment_dev --out data/dev-aug
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
from ttbench.dataset import read_wav_stereo, write_wav_stereo  # noqa: E402
from tools.synth_calls import bed, esc_clip, phone, place, reverb, set_level  # noqa: E402

VARIANTS = {
    "p": ("phone-nb|street", dict(phone=True, bed="street", gain_db=0)),
    "q": ("car|quiet-caller", dict(phone=False, bed="car", gain_db=-6)),
    "e": ("speakerphone|household", dict(phone=False, bed="household", gain_db=-2, echo=True)),
    "x": ("cafe|event", dict(phone=True, bed="cafe", gain_db=0, event=True)),
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(KIT / "data" / "dev"))
    ap.add_argument("--out", default=str(KIT / "data" / "dev-aug"))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    src, out = Path(a.data), Path(a.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    scen = []
    for e in manifest["scenarios"]:
        sid = e["scenario_id"]
        lab = json.loads((src / "labels" / f"{sid}.json").read_text(encoding="utf-8"))
        st = read_wav_stereo(src / "audio" / lab["audio"]["file"]).astype(np.float32) / 32768
        agent, caller = st[:, 0], st[:, 1]
        for v, (cond, o) in VARIANTS.items():
            c = caller * 10 ** (o["gain_db"] / 20)
            n = len(c)
            if o.get("echo"):
                lag = rng.uniform(120, 300)
                ech = np.concatenate([np.zeros(int(lag * 16), np.float32), agent])[:n]
                ech = signal.sosfilt(signal.butter(2, [250, 3500], "band", fs=16000, output="sos"), ech)
                c = c + reverb(ech, rng, 0.25)[:n] * 10 ** (rng.uniform(-24, -16) / 20)
            c = c + bed(o["bed"], n, rng).astype(np.float32)
            if o.get("event"):
                x = set_level(esc_clip(rng.choice(["dog", "car_horn", "door_wood_knock", "crying_baby"]), rng), -26)
                place(c, x[: 2 * 16000], rng.uniform(2000, max(2500, n / 16 - 4000)))
            if o["phone"]:
                c = phone(c)
            new = f"{sid}-{v}"
            stereo = (np.clip(np.stack([agent, c], 1), -1, 1) * 32767).astype(np.int16)
            write_wav_stereo(out / "audio" / f"{new}.wav", stereo)
            l2 = json.loads(json.dumps(lab))
            l2["scenario_id"] = new
            l2["audio"]["file"] = f"{new}.wav"
            l2["acoustic"]["conditions"] = sorted(set(lab["acoustic"]["conditions"]) | set(cond.split("|")) - {"quiet-caller", "event"})
            for ev in l2["events"]:
                ev["event_id"] = ev["event_id"].replace(sid, new)
            (out / "labels" / f"{new}.json").write_text(json.dumps(l2, ensure_ascii=False, indent=1), encoding="utf-8")
            scen.append({"scenario_id": new})
    m2 = {k: v for k, v in manifest.items() if k != "scenarios"} | {"split": "dev-aug", "scenarios": scen}
    (out / "manifest.json").write_text(json.dumps(m2, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(scen)} calls to {out}")


if __name__ == "__main__":
    main()
