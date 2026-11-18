# CLAUDE.md — instructions for Claude Code (Sarj turn-taking project)

Claude Code reads this file automatically whenever the project opens. It says **what we want, in what order, and when a task counts as done**. Read all of it before any change. If anything here conflicts with an explicit request from the user, the user's request wins.

## 0. Read first
1. `docs/PRD.md`: what we build and why, the current status, phases, and questions for the mentor.
2. `docs/TECH_SPEC.md`: how every module works, parameters, data partitioning, and gates G1–G7.
3. `INTERFACE.md`, `SCORING.md`, `DATA.md`, `SCRIPTS.md`: the official kit contract. These override everything.
4. `docs/EXPERIMENTS.md`: what has been tried, and what it scored.

## 1. Context in one paragraph
We are a student team in the Sarj AI hackathon. The harness streams a 2-channel call (agent + caller, 16 kHz, 20 ms frames) to `my_system.py` over WebSocket :8765. We decide only **STOP / RESUME / TAKE_TURN**. Full credit needs a STOP within 300 ms of the caller's intent becoming evident. Final rank = 80 % hidden human Arabic benchmark + 20 % judges.

Current scores (`runs/sim-history.csv` and `runs/final-dev-official`):

| dev | dev-aug | novel-test | blind novel-test-2 | real-time dev |
|---|---|---|---|---|
| 99.3 | 96.6 | **84.1** | **85.5** | 99.0 |

The gap is **generalisation**: lexicon errors on new phrasing, not ASR.

## 2. Rules (never break these)
1. **Never modify `ttbench/`.** It is the official harness.
2. **Measure every change under `turn/`** before you say it is done:
   ```bash
   python -m tools.loop --sets dev,dev-aug,novel-test,synth-test --note "<what changed>"
   ```
   Reject the change, or report it as a regression, if `dev-aug` or `novel-test` drops by more than 0.3, or false stops rise (gates G1–G7, TECH_SPEC §7.2).
3. **Never copy `novel-test` or `novel-test-2` sentences** into the lexicon, training data or tests. Fix the *category* of error instead. Open `novel-test-2` only once, when the user says so.
4. **Never delete files** without asking the user. Move them to `models/_archive/` or `_to_delete/` instead.
5. **No secrets in the repo.** API keys live only in `.env` (git-ignored). Never print a key, and never read `.env` aloud in output.
6. **No audio storage** from the checkpoint or final runs. Recordings in `data/live/` are git-ignored and need consent.
7. **Cross-platform.** Every change must run on Mac (Cohere/MLX) and Windows (faster-whisper). No hard-coded `~/Downloads/...`, and no Mac-only commands such as `open`.
8. **One heavy job at a time** (sim, training, generation). The laptop overloads otherwise.
9. **Git workflow:**
   - a new branch per task (`fix/...`, `exp/...`, `feat/...`);
   - small commits;
   - a PR into `main`;
   - **never push directly to `main`**;
   - never rewrite history;
   - after a measured improvement is merged, tag it: `git tag -a vX.Y -m "dev NN.N · dev-aug NN.N · novel NN.N"`, and update the score table in `README.md`.
10. **One module per function.** Fix a problem inside the module that owns it (see TECH_SPEC §4.1). Don't spread a fix across layers.
11. **The scored path stays local.** Cloud APIs are allowed only in the demo brain (`convo/`), data generation and offline analysis. They are optional, and they are declared in `WRITEUP.md`.
12. After each task, add an entry to `docs/EXPERIMENTS.md`: what changed, numbers before and after, decision.

## 3. How to work
- Do **one task at a time**, in the order below, unless the user picks one.
- At the start of each task:
  - create the branch;
  - restate the task's exit criteria;
  - run the baseline measurement if the task touches `turn/`.
- At the end:
  - show a before/after table;
  - say which gates passed;
  - open the PR (`gh pr create`), or say what is blocking.
- **Stop and ask** before anything irreversible: deleting, force-pushing, changing a threshold the gates depend on, or opening `novel-test-2`.
- If a fix fails twice, stop and explain. Don't keep trying random changes.

## 4. Task queue

| ID | Task | Phase | Exit criteria |
|---|---|---|---|
| **T0** | Decide `fix1` (uncommitted changes in `turn/lexicon.py`, `turn/policy.py`, `tools/synth_calls.py`). Measure stash vs applied on all 4 sets | P1 | Before/after table shown to the user; user decides keep or drop |
| **T1** | Cross-platform hygiene: (a) `tools/loop.py`: replace `open` with a platform-aware opener; add `--quick`; add `novel-test` to `SETS`. (b) `turn/config.py` with `TT_EXT_DIR`, replacing the hard-coded path in all 11 files (`grep -rn "sarj-ext-data"`). (c) `.env` loader + `.env.example`. (d) Share `novel-test` via a GitHub release `data-v1` plus a fetch step in `tools/fetch.py`. (e) Load `models/policy-{BACKEND}.json` for every backend, and train `policy-faster.json` | P1 | `pytest -q` green; `python -m tools.sim` works on Mac **and** Windows; tag `v1.0` |
| **T2** | Demo brain via the Gemini API: `TT_BRAIN=gemini` and `GEMINI_API_KEY` in `.env`, model `gemini-3.5-flash-lite`, handled in `convo/brain.py`. Add `--local-brain` to `setup.sh` so the 15 GB local model is not downloaded by default | P1 | Demo answers through Gemini; Claude Haiku takes over when Gemini fails |
| **T3** | Disk and models cleanup (**ask first**): mark the official models in `models/README.md`; move other policy and student variants to `models/_archive/` | P1 | User approved; `models/README.md` exists |
| **T4** | `tools/errors.py`: the 4 error types (false stop, missed, late, premature take-turn) × category × dialect × noise, with examples (call id, time, text, rule, `sem_p`) | P2 | Totals match `score.json` (9 false stops, 4 misses on `runs/sim-fix1-novel-test`) |
| **T5** | Generalisation (TECH_SPEC §5.7–5.8): split the lexicon into BID vs CUE; add the thinking-aloud rule; add teacher categories (openers in 14 dialects, thinking aloud, room talk) with **new** phrases; retrain the student and policy | P3 | `novel-test` ≥ 90, G1–G7 pass |
| **T6** | Early acoustic stop `EARLY_MS` (default 0) per TECH_SPEC §5.10. Also try `ASR_EVERY_MS=100` | P4 | Share of stops later than 300 ms lower, gates pass; otherwise left off |
| **T7** | Prosody for a lone ambiguous word («طيب»، «خلاص») | P4 | AM category up, no new false stops |
| **T8** | `novel-test-2` via `--bank data/novel_bank2.json` (blind bank). Measure **once** when the user says so | P5 | Result recorded; no tuning afterwards |
| **T9** | `WRITEUP.md` (2–4 pages) per TECH_SPEC §10: ablation, ≥ 5 failure cases with Arabic examples, declared services | P6 | Reviewed by the user |
| **T10** | `README.md`: setup for Mac and Windows, how to measure, the score table per tag | P6 | A new teammate can run `tools.sim` from README alone |
| T11 (optional) | Slow-brain RESUME confirmation `TT_SLOWBRAIN` (TECH_SPEC §5.11): measure the round trip first | — | Only if p90 RTT < 300 ms and the gates pass |

## 5. Commands
```bash
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m tools.sim                    # dev, fast
python -m tools.sim --data data/novel-test
python -m tools.loop --quick --note "..."
python -m ttbench.play --calls all     # official real-time run (~10 min, nothing else running)
python -m ttbench.play --data data/live --calls all
python -m ttbench.summary runs/<name> --failures
python -m ttbench.ready
pytest -q
```
Slash commands in `.claude/commands/`: `/measure`, `/task`, `/tag-version`. The source copies live in `claude-setup/`. If `.claude/` is missing, run `bash claude-setup/install.sh` once.

## 6. Starting prompt (for the user)
> Read CLAUDE.md, docs/PRD.md and docs/TECH_SPEC.md fully. Create branch `fix/team-ready`, commit CLAUDE.md, docs/, claude-setup/, .claude/ and .gitignore. Then do T0 only. Stop, and show me the before/after numbers and your recommendation.
