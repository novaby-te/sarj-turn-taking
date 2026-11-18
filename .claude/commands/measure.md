---
description: Run the gate measurement (dev, dev-aug, novel-test, synth-test) and compare with the last entry
---
1. Make sure nothing heavy is running (one heavy job at a time).
2. Run: `python -m tools.loop --sets dev,dev-aug,novel-test,synth-test --note "$ARGUMENTS"`
3. Read the last rows of `runs/sim-history.csv` and show a before/after table per set: Core, R, S, false stops, misses, median latency.
4. Check gates G1–G5 from docs/TECH_SPEC.md §7.2 and say PASS/FAIL for each.
5. Do not change any code in this command.
