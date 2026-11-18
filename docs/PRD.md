# PRD — Arabic Turn-Taking Decision Engine (Sarj AI Hackathon)

**Version:** 1.1 · **Date:** 2026-11-18 · **Owner:** Saud (team lead) · **Status:** Active

| Document | Answers | Audience |
|---|---|---|
| `docs/PRD.md` (this file) | **What** we build, **why**, current status, the plan, and how we measure | Team, mentor, Claude Code |
| `docs/TECH_SPEC.md` | **How** every module works (algorithms, parameters, data, gates) | Developers, Claude Code |
| `CLAUDE.md` | The **task queue** and the rules Claude Code must follow | Claude Code |

Official sources that override this document if they conflict: `CHALLENGE.md`, `INTERFACE.md`, `SCORING.md`, `DATA.md`, `SCRIPTS.md`, the organiser brief (15 Nov 2026), and the mentor's team guidance.

---

## 1. The problem in simple words

A voice agent is talking to a caller on the phone, in Arabic. While it talks, the caller makes sounds. Some sounds mean "go on" («إيه»، «أها»، «مم»), some are not the caller at all (TV, a child, a cough, the agent's own echo on speakerphone), and some mean "stop, I want to speak" («لا لا لحظة»، «كم المبلغ؟»، «أيوه… بس أبي أغيّر الموعد»).

Our system listens to the call **20 ms at a time** and decides, while the agent is still speaking, whether the agent should **keep talking** or **stop now**. Optionally it also decides when the caller has **finished** so the agent can answer.

**We do not build a voice assistant.** The agent's voice, its text, and the call audio already come from the starter-kit harness. We build only the decision layer.

## 2. Goals and success metrics

**How we are scored** (from `SCORING.md` and the organiser brief):

- Final rank = **80 % Core score** on a hidden Arabic benchmark + **20 % judges**. The judges' 20 % splits into originality and reasoning 10, practical applicability 5, and clarity of demo and evaluation 5.
- `Core = 100 × (0.9·F + 0.1·A)`, where `F = 2RS/(R+S)`.
  - **R (responsiveness)** = credit for real interruptions. Credit is 1.0 if the agent stops within 300 ms of the moment the intent becomes evident, falls linearly to 0.4 at 1500 ms, and is 0 after that.
  - **S (restraint)** = credit for keeping talking through acknowledgements, hesitation and noise, computed per category and **multiplied** by a "no stray stop" item. A wrong stop taken back with `RESUME` within 500 ms earns 0.5.
  - **A** = credit on genuinely ambiguous cases: 1.0 for the preferred behaviour, 0.6 for the other.
- End-of-turn (`TAKE_TURN`) has its own leaderboard and does not count in the total.

**Targets before code freeze:**

| Metric | Target | Measured with |
|---|---|---|
| Core on `dev` (24 calls), **real-time run** | ≥ 97 | `python -m ttbench.play --calls all` |
| Core on unseen phrasing (`novel-test`, later `novel-test-2`) | ≥ 92 | `tools.sim` / `tools.loop` |
| Core on degraded audio (`dev-aug`) | ≥ 96, never lower than before | `tools.loop` |
| Missed interruptions on `dev` | 0 | scorer report |
| False stops per minute of agent speech | ≤ 0.2 | scorer report |
| Stop latency from `evident` | median ≤ 300 ms, p90 ≤ 700 ms | real-time run |
| Dialect gap (Najdi vs others) and noise gap (quiet vs noisy) | ≤ 3 points | `tools/errors.py` |
| Laptop memory during the live demo | < 6 GB | Activity Monitor / Task Manager |
| 30-minute soak test | no crash, every `ping` answered | soak run |
| `WRITEUP.md` | complete, 2–4 pages, ≥ 5 failure cases with Arabic examples | review |

## 3. Constraints

1. **Local first.** The scored decision path runs fully offline on a laptop, with zero network calls. Any cloud API is optional, used only outside the scored path (demo brain, data generation), and **declared** in the write-up.
2. **Speed.** Full credit needs the stop within **0.3 s** after the intent is evident. The harness timestamps a decision when it **arrives**, so every millisecond of compute counts.
3. **Start from the starter kit.** `my_system.py` and the `ttbench` server interface stay as given. `ttbench/` is never modified.
4. **Load before the call.** The harness waits only 10 s for `hello`, so all models load and warm up before we send it.
5. **Rules.** Arabic audio, focused on Saudi and Gulf dialects. No customer data. Recordings only with consent. **Never store audio from the checkpoint or final runs.** Code freeze by commit hash at the end of Day 5. One restart allowed within 60 s.
6. **IP.** Sarj receives a perpetual commercial licence to the submission, so anything shipped must be Apache-2.0, MIT or BSD. Data under CC BY-NC (for example ESC-50) is used for generation and testing only, and is declared.
7. **Hardware.** The primary target is an Apple-Silicon Mac (Cohere ASR on MLX). The secondary target is Windows or Linux on CPU (faster-whisper). The final scored run happens on the Mac.

## 4. Scope

**In scope**

- The Core track: STOP and KEEP while the agent speaks, `RESUME` to take back a wrong stop, and a `label` on every decision.
- The EOT track: `TAKE_TURN`, with patience for fillers and partly read numbers.
- Our own evaluation tools: a fast simulator with measured latency, a measurement loop, a dashboard, a categorised error report, and an ablation study.
- Data pipelines: a call generator in the kit's format, degraded-audio augmentation, a teacher (LLM) data generator, the student model, and training for the decision layer.
- A demo app: live microphone with a speaking agent, replay of calls with a decision timeline, and a results view.
- `WRITEUP.md`, the experiment log, and the dialect analysis.

**Out of scope**

- Telephony integration.
- Building a voice assistant from scratch.
- Cloud inference inside the decision path, and GPU servers.
- Modifying `ttbench/` or the WebSocket protocol.
- Ready-made agent platforms that make the turn-taking decision for us.
- Rewriting git history before the final.

## 5. System flow: from one audio frame to STOP or RESUME

```
harness ──(binary frame every 20 ms: 1280 B, ch0 = agent, ch1 = caller)──► my_system.py
  │
  ├─ SYNC LOOP (every frame, < 5 ms)
  │    L0 echo suppression    : remove the agent's own voice leaking into the caller channel
  │    L1 voice + level gate  : Silero VAD + loudness relative to THIS caller's own voice and the noise floor
  │    state machine          : opens a "candidate" when the caller starts speaking during agent speech
  │                             └─ every 150 ms (and at each pause) submit the growing segment ─┐
  │                                                                                             ▼
  ├─ BACKGROUND WORKER (one job at a time, ~100–150 ms)
  │    L4  ASR (Cohere Transcribe Arabic on Mac / faster-whisper elsewhere) ─► text + confidence
  │    L4b meaning model (CAMeLBERT student distilled from an LLM, ~7 ms)     ─► P(wants the floor)
  │    L2  speaker match (CAM++ voiceprint vs the caller's first answer)      ─► same caller?
  │    L3  prosody and effort relative to the caller (pitch, slope, effort)   ─► how it was said
  │
  └─ FUSION (on each worker result, synchronous)
       L4a lexicon: floor-bid words vs cues vs backchannels vs fillers
       L5  hard gates → transparent rules → learned model (veto / promote) → STOP if p ≥ 0.5
       RESUME watch: if a weak stop turns out to be a backchannel → RESUME within 460 ms
       L6  end of turn (agent silent): wait 600 ms, or 2500 ms if the answer looks unfinished → TAKE_TURN
  ▼
decision JSON sent immediately: {"type":"decision","decision":"STOP","label":"interruption", ...}
```

## 6. Modules: inputs and outputs

Each module lives in its own file, and each one is fixed inside that file only.

| Module | File | Input | Output |
|---|---|---|---|
| Entry point | `my_system.py` | WebSocket messages | `hello`, decisions, `pong` |
| Settings | `turn/config.py` (to add) | `.env`, environment | paths and flags |
| L0 echo | `turn/echo.py` | agent frame, caller frame | `resid_db`, `echo_db` |
| L1 VAD | `turn/vad.py` | caller frame | P(speech) |
| L2 speaker | `turn/speaker.py` | ≤ 3 s of audio | L2-normalised embedding |
| L3 prosody | `turn/acoustic.py` | segment + the caller's reference profile | `ac_*` relative features |
| L4 ASR | `turn/asr.py` | int16 audio | `{text, logprob, no_speech, ms}` |
| L4a lexicon | `turn/lexicon.py` | text + the agent text heard so far | lexical features |
| L4b meaning | `turn/semantic.py` | agent tail + caller text | `sem_p` |
| L5 fusion, RESUME, L6 EOT | `turn/policy.py` (+ `turn/eot.py`) | all features | STOP, RESUME, TAKE_TURN |
| Worker and clock | `turn/clock.py` | jobs | results with measured latency (also in the simulator) |
| Demo agent (L7) | `convo/` | the caller's full turn | spoken reply (demo only) |

## 7. Current status (as of 2026-11-17) and gaps

The system already covers every layer above. Scores come from `runs/sim-history.csv`, using the official scorer:

| Set | What it measures | Core | False stops | Misses |
|---|---|---|---|---|
| `dev` (24) | calls we have seen | 99.3 | 0 | 0 |
| `dev-aug` (96) | same calls, with noise, phone line and speakerphone | 96.3 | 0 | 0 |
| `synth-test` | new voices, **our own sentences** | 99.3 | 1 | 0 |
| `synth-heldvoice` | voices never heard, our sentences | 99.5 | 0 | 0 |
| `novel-test` (120) | **new phrasing** | **82.2** | 9 | 4 |
| B1 baseline from Sarj | stops at any voice | 41.0 | — | — |

The last official real-time run on `dev` scored 96.5, matching the simulator.

**Main gap: generalisation (17 points).** The trace (`runs/sim-fix1-novel-test/trace.json`) shows ASR heard correctly. The lexicon misread the words:

| Failure type | Example (what the system heard) | Root cause |
|---|---|---|
| Vocative or negation treated as a floor bid | «يعطيك العافية يا اخوي» (BC-N037), «ما عدا» (BC-N044) | `BID` contains ما، يا أخي، اخوي |
| Self-talk or talking to someone nearby | «الثانية أيش» (HT-N100), «وين حطيتي» (HT-N101) | Question words are in `BID` |
| Thinking aloud | «يعني إيه تقريباً…» (HT-N099), «حمام الأربعاء…» (HT-N103) | The "≥ 2 content words" rule ignores hesitation |
| Unknown interruption opener, so the stop comes late | «اعذرني…» (GI-N017), «من جد» (GI-N011) | Not in the lexicon; the stop waits for 3 content words (15 of 41 interruptions stopped after 300 ms) |
| Ambiguous word alone | «طيب» (AM-N053), «ايه طيب» (AM-N047) | Prosody is not used for a lone ambiguous word |

**Engineering gaps**

1. `WRITEUP.md` is still the empty template, and it is mandatory.
2. Uncommitted change `fix1` in `turn/lexicon.py` and `turn/policy.py` has not been decided (rule check pending).
3. `tools.loop` crashes on Windows (it calls the Mac-only `open` command).
4. `data/novel-test` exists on one laptop only; it is not in git or in a release.
5. Windows uses `policy.json`, which was trained on MLX-Whisper words, not faster-whisper words.
6. `~/Downloads/sarj-ext-data` is hard-coded in 11 files.
7. Six policy variants and three student variants sit in `models/` with no "official" marker.
8. The Mac runs heavy work in parallel (Gemma 26B uses about 15 GB) and becomes overloaded.

## 8. Key decisions

| # | Decision | Why |
|---|---|---|
| D1 | The decision path is 100 % local | Latency is scored on arrival; the venue's internet is not guaranteed; local is easier to defend under "practical applicability" |
| D2 | No GPU server, and no cloud ASR | Too complex for a 6-day hackathon; local Cohere on MLX takes about 97 ms per read |
| D3 | The demo brain runs through an API (OpenRouter, Gemma 4 26B) with Claude Haiku as fallback | Frees about 15 GB of RAM; it is outside the scored path; it is declared |
| D4 | The mentor's "Gemini as slow brain" idea is **optional, off by default**. It is allowed only to confirm or correct a stop with `RESUME`, and only if the measured round trip fits inside the 460 ms RESUME window. Its default use is offline: labelling errors and generating teacher data | The mentor recommends measuring the round trip first; our distilled student already brings LLM judgement into 7 ms |
| D5 | Decisions are gated on **unseen** sets (`novel-test`, `dev-aug`, held-out voices), not on `dev` | The hidden set has human, unscripted callers; `dev` has few real interruptions, so we never tune to individual calls |
| D6 | The lexicon gets two tiers: strong `BID` words stop at once; `CUE` words stop only when meaning confirms | This fixes the largest group of false stops |

## 9. Phased build plan

Phase 0 is done. Each later phase ends with a measurement using the kit's tools, and **each phase ships as a tagged version with its score**.

| Phase | Goal | Work | Exit criteria (measured) |
|---|---|---|---|
| **P0 — done (v0.1 to v0.9)** | An end-to-end system | VAD + level → ASR + lexicon → speaker print → learned fusion → Cohere → meaning student → RESUME → demo app | `dev` real-time run 96.5; simulator matches |
| **P1 — Hygiene, runs everywhere** | Every teammate can run and measure | Decide `fix1`; `turn/config.py` + `.env`; `TT_EXT_DIR`; fix `loop` on Windows; `--quick`; share `novel-test` via a release; `policy-{engine}.json`; demo brain via API; mark the official models | `pytest` passes; `tools.sim` prints `core` on Mac and Windows; tag `v1.0` |
| **P2 — Measure the errors properly** | Know what to fix first | `tools/errors.py`: the 4 error types × category × dialect × noise, with examples; feed in the checkpoint results | Totals match `score.json` (9 false stops, 4 misses on `fix1-novel-test`) |
| **P3 — Generalisation** | Raise `novel-test` | Two-tier lexicon (BID/CUE); a thinking-aloud rule; more teacher categories (openers in 14 dialects, thinking aloud, talking to someone nearby) and a retrained student | `novel-test` ≥ 90, `dev-aug` not lower, false stops not higher |
| **P4 — Speed and prosody** | Stop within 300 ms | An early acoustic stop path (off by default); prosody for a lone «طيب/خلاص»; try `ASR_EVERY_MS = 100` | Share of stops later than 300 ms lower; the G1–G5 gates pass |
| **P5 — Human data and final check** | Evidence for unscripted speech | 20–30 consented recordings (mic + real phone call on speakerphone); build `novel-test-2` from a sentence bank written blind, and run it **once** | Results recorded in `EXPERIMENTS.md` and `WRITEUP.md` |
| **P6 — Freeze and deliver** | Submit | Real-time run on the freeze commit; `python -m ttbench.ready`; `WRITEUP.md`; commit hash + external services list; soak test | Delivered before the end of Day 5 |

## 10. Testing and measurement

**Three audio sources** (the mentor's guidance). Run them all through the same tools, and explain any large gap between them: noise, dialect, mic quality, or synthetic voice.

1. **`dev` from the kit** (24 calls with labels). The **official number always comes from a real-time run** (`ttbench.play`). Fast simulator runs are for experiments only.
2. **Our microphone and phone recordings**, labelled in the kit's format (see `SCRIPTS.md`), in `data/live`.
3. **Synthetic calls**: `synth`, `synth-test`, `synth-heldvoice`, `novel-test`. These use Edge TTS and ElevenLabs voices, and the free ElevenLabs tier is non-commercial, which we declare.

**Commands**

```bash
python -m tools.sim                                     # dev, about one minute (simulator, measured latency)
python -m tools.sim --data data/novel-test              # one set
python -m tools.loop --quick --note "what changed"      # dev + dev-aug + novel-test → sim-history.csv + dashboard
python -m ttbench.play --calls all                      # official real-time run (~10 min)
python -m ttbench.play --data data/live --calls all     # our own recordings, same tool
python -m ttbench.summary runs/<name> --failures        # every decision and its time, plus the failure table
python -m tools.errors runs/<name>                      # error types × dialect × noise (to build in P2)
python -m ttbench.ready                                 # before submission
```

**Acceptance gates** for any change under `turn/`. Full definitions are in TECH_SPEC §7.

- `novel-test` and `dev-aug` must not drop by more than 0.3.
- False stops must not rise.
- `dev` must stay ≥ 97 with 0 misses.
- Latency must not get worse by more than 50 ms.
- Tests must pass.
- Every change gets an entry in `docs/EXPERIMENTS.md`.

**The rule:** better on seen sets but worse on unseen sets = **reject**.

## 11. Versioning and Git workflow

- `main` always holds a working system. **Never push to `main` directly.**
- One branch per module or experiment, for example `exp/lexicon-tiers` or `fix/windows-loop`.
- Merge through a PR only after the gates pass.
- Tag every working version, and record its score in the README table:

  ```bash
  git tag -a v1.0 -m "dev 99.3 · dev-aug 96.3 · novel 82.2"
  git push origin v1.0
  ```

- Secrets live in `.env`, which is git-ignored. Large audio and model files are git-ignored too and are shared through GitHub Releases.
- The final run uses **exactly** the frozen commit hash.

## 12. Team roles

Roles are flexible, but every role has one clear owner. **Every member must be able to explain the path from the first frame to a STOP**, because judges may ask anyone.

| Role | Owns |
|---|---|
| Tech lead and integration | PRD, repo, `my_system.py`, fusion (L5), gates, freeze, run day |
| Audio and decision signals | L0–L3, the early acoustic path, prosody for ambiguous words |
| ASR and language | L4 engines, hallucination filters, lexicon (L4a), end of turn (L6) |
| Data and testing | Generator, augmentation, holdouts, teacher and student, recordings, labels, error report |
| Presentation and write-up (1–2 people from day one) | Screenshots and results for every phase, demo app, `WRITEUP.md`, the 7-minute talk |

## 13. Deliverables and presentation

**Mandatory**

- A real-time system connected to the harness.
- Our own evaluation on `dev`, with at least 5 explained failure cases.
- `WRITEUP.md`, 2–4 pages, declaring every API, model and dataset.
- Commit hash and the external services list by the end of Day 5.

**Stretch** (we do all of them)

- `label` on every decision.
- `RESUME`.
- `TAKE_TURN`.
- Dialect analysis.
- Live microphone demo on Day 6.

**Talk: 7 minutes + 3 minutes Q&A** (organiser brief)

| Time | Part | Content |
|---|---|---|
| 0:00–0:45 | Problem | Two Arabic clips: «إيه والله صادق» vs «إيه… بس أبي أغيّر» |
| 0:45–2:15 | Idea | The layer stack, comparison against the caller's own voice, and the distillation (3.5 s → 7 ms) |
| 2:15–4:15 | Live demo | «إيه» (keeps talking), «لا لا لحظة» (stops), TV (ignored), with the layer view on screen |
| 4:15–5:45 | Evidence | Scores on unseen sets, ablation, latency, dialect gap |
| 5:45–6:30 | Honest failures | Two Arabic failure cases |
| 6:30–7:00 | Practicality | Runs on a laptop, cost per call, what deployment would take |

## 14. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Overfitting to synthetic data; the hidden set is human and unscripted | High | High | Gate on unseen sets; collect human recordings; `novel-test-2` opened once |
| Tuning on `novel-test` until it is memorised | High | High | Fix categories, never copy its sentences; final judgement on `novel-test-2` |
| The early acoustic stop increases false stops | Medium | High | Off by default; protected by speaker and level checks; covered by RESUME; enabled only if the gates pass |
| The laptop is overloaded during an official run | Low after cleanup | High | Run only `my_system.py`; reboot first; follow the run-day checklist |
| Thresholds change after Sarj calibrates them | Medium | Low | Read thresholds from `manifest.json` |
| The demo fails (noise or internet) | Medium | Medium | Headset, local fallback brain, recorded backup video |

## 15. Questions for the mentor

1. Is the Day 3 checkpoint run definitely tonight (participants 6:00–7:30 pm)? Do we get per-category results and per-event details, or only totals?
2. Ambiguous (AM) cases in the hidden set: are they all labelled EITHER, or are some STOP?
3. Will the provisional thresholds (300/1500 ms, 600/2000/3000 ms) change before the final, and when will the final values be published?
4. Roughly what share of the hidden set is human-recorded or unscripted? Is any of it real phone audio rather than simulated narrowband?
5. For the final, does the harness connect over the venue LAN to our laptop's IP? Which port and network should we prepare?
6. For Q&A: is the talk 7 + 3 minutes (brief) or 8 + 5 (CHALLENGE.md)?
7. Can we use an LLM-generated teacher dataset and an ElevenLabs free-tier synthetic test set if we declare them?
8. Is a demo brain through an API acceptable, given that the scored path is fully local?

## 16. External services (summary; full register in TECH_SPEC Appendix A)

Cohere Transcribe Arabic (Apache-2.0) · Whisper large-v3-turbo / faster-whisper (MIT) · Silero VAD (MIT) · 3D-Speaker CAM++ via sherpa-onnx (Apache-2.0) · CAMeLBERT-DA (Apache-2.0) · Smart Turn v3.2 (BSD-2, optional) · LLM teacher (Claude) for training data only · Gemma 4 via OpenRouter for the demo brain only · Edge TTS and ElevenLabs for synthetic test voices and demo voice · ESC-50 (CC BY-NC, generation only).
