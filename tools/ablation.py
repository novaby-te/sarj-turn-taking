"""What does each layer contribute? Turns one layer off at a time and measures the unseen sets.

    .venv/bin/python -m tools.ablation          # -> analysis/ablation.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
from tools.sim import simulate  # noqa: E402
from turn import echo  # noqa: E402

SETS = ("novel-test", "dev-aug")
CONDITIONS = {
    "full system": {},
    "no echo suppression": {"_no_echo": True},
    "no level gate": {"policy.REL_DB": 99.0, "policy.STOP_REL_DB": 99.0},
    "no speaker check": {"policy.USE_SPEAKER": 0},
    "rules only (no learned model)": {"policy.USE_MODEL": 0},
    "no meaning model": {"policy.USE_SEM": 0},
    "no RESUME": {"policy.RESUME_P": 0.0},
}


def _raw_level(self, agent, caller, agent_active):
    level = 20 * np.log10(np.sqrt(np.mean(caller.astype(np.float64) ** 2)) / 32768 + 1e-9)
    return float(level), -99.0


def main() -> int:
    original = echo.EchoSuppressor.process
    results = {}
    for name, settings in CONDITIONS.items():
        overrides = {k: v for k, v in settings.items() if not k.startswith("_")}
        echo.EchoSuppressor.process = _raw_level if settings.get("_no_echo") else original
        results[name] = {}
        for s in SETS:
            r = simulate(str(KIT / "my_system.py"), KIT / "data" / s, overrides=overrides)
            results[name][s] = {"core": round(r["core"]["core_score"], 2), "false_stops": r["rates"]["false_stops"],
                                "misses": r["rates"]["misses"], "median_ms": r["latency_ms"]["median"], "p90_ms": r["latency_ms"]["p90"]}
            print(f"{name:32s} {s:11s} {results[name][s]}", flush=True)
    echo.EchoSuppressor.process = original
    (KIT / "analysis").mkdir(exist_ok=True)
    (KIT / "analysis" / "ablation.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
