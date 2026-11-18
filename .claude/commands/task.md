---
description: Start a task from the CLAUDE.md queue (e.g. /task T1)
---
Task: $ARGUMENTS

1. Re-read the row for this task in CLAUDE.md §4 and the TECH_SPEC sections it references.
2. Create a branch (`fix/`, `feat/` or `exp/` + short name) from an up-to-date `main`.
3. State the exit criteria in one short list.
4. If the task touches `turn/`, run `/measure baseline-$ARGUMENTS` first.
5. Implement inside the owning module only. Small commits.
6. Measure again, show before/after, report gates, add an entry to docs/EXPERIMENTS.md.
7. Push the branch and open a PR with `gh pr create`. Never push to main. Stop and ask before anything irreversible.
