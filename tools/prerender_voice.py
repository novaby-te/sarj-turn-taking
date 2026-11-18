"""Generate the agent's fixed lines once, in the same Gemini voice as its live replies, so calls start without waiting.

    .venv/bin/python -m tools.prerender_voice

The audio is stored outside git, in the voice cache. The Gemini free tier allows 3 speech requests a minute.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
from convo import voice  # noqa: E402
from convo.scenarios import SCENARIOS  # noqa: E402

WAIT_S = 22


async def main() -> int:
    missing = 0
    for name, scenario in SCENARIOS.items():
        for text in (" ".join(scenario["opening"]), scenario["first_point"]):
            if voice.has_live_line(text, scenario["gender"]):
                print("ready  ", name, flush=True)
                continue
            x, used = await voice.tts(text, scenario["gender"], "live")
            ok = used == "gemini"
            missing += not ok
            print("created" if ok else "FAILED ", name, f"{len(x) / voice.RATE:.1f}s", flush=True)
            await asyncio.sleep(WAIT_S)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
