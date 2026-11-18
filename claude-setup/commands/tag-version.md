---
description: Tag the current main as a working version with its score (e.g. /tag-version v1.1)
---
1. Confirm you are on `main`, it is clean, and it matches `origin/main`.
2. Take the latest dev, dev-aug and novel-test Core from `runs/sim-history.csv` (and the last real-time dev run if there is one).
3. Create the tag: `git tag -a $ARGUMENTS -m "dev NN.N · dev-aug NN.N · novel NN.N"` with the real numbers.
4. Add a row to the score table in README.md on a branch `docs/tag-$ARGUMENTS` and open a PR.
5. Ask the user before `git push origin $ARGUMENTS`.
