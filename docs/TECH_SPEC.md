# TECH_SPEC — Arabic Turn-Taking Decision Engine

**Version:** 1.1 · **Date:** 2026-11-18 · **Companion to:** `docs/PRD.md` · **Task queue:** `CLAUDE.md`

This document is complete enough for a developer, or Claude Code, to build, run, measure and defend the whole system without the original authors. Parameter values are the **current defaults in the code** (`turn/policy.py` and friends). If code and this document disagree, the code wins, and this document gets fixed in the same PR.

Normative sources, which override this spec: `INTERFACE.md` (protocol), `SCORING.md` (metric), `DATA.md` (labels), `CHALLENGE.md` and the organiser brief (rules), and the mentor's team guidance (workflow).

---

## 1. Product and goals

### 1.1 Definition
A real-time WebSocket service. It receives a 2-channel call stream (ch0 = agent, ch1 = caller) in 20 ms frames, and emits turn-taking decisions:

- `STOP`: the agent must stop speaking, because the caller is taking the floor.
- `RESUME`: take back a recent wrong `STOP`.
- `TAKE_TURN`: the caller has finished, and the agent may speak (EOT track).

### 1.2 Scoring (summary of `SCORING.md`)
- `Core = 100 × (0.9·F + 0.1·A)`, with `F = 2RS/(R+S)`.
- R (responsiveness) is per interruption event. Latency is `t_decision_arrival − t_evident`. Credit is 1.0 when latency ≤ 300 ms, falls linearly to 0.4 at 1500 ms, and is 0 after that or if the event is missed.
- S (restraint) is the mean over non-interruption categories (backchannel, hesitation, noise, side speech, echo) × a "no stray stop" item. A `STOP` taken back by `RESUME` within 500 ms earns 0.5 instead of 0.
- A (ambiguity): EITHER-labelled events. The preferred action earns 1.0, the other earns 0.6.
- Final rank = 80 % hidden-set Core + 20 % judges (originality 10, practicality 5, clarity 5).

### 1.3 Win strategy
1. **Generalise, don't memorise.** Every change is gated on unseen sets (§7).
2. **Be fast where it pays.** Bid words stop on the first ASR read, about 250 ms after onset.
3. **Be relative, not absolute.** Loudness, pitch and voice are compared with *this* caller's own first answer (U0), never with global constants.
4. **Layered veto.** Cheap acoustic gates reject most noise before any ASR. The learned and semantic layers only veto or promote rule decisions.
5. **Safety net.** `RESUME` recovers half the credit on wrong stops.
6. **Explainability.** Every decision carries a `label` and a trace line that a teammate can read aloud to a judge.
7. **Honest evaluation.** Ablation, a dialect gap analysis, and failure cases with Arabic examples.

## 2. Constraints and assumptions

| ID | Constraint |
|---|---|
| C1 | The scored path is 100 % local and makes zero network calls |
| C2 | The decision must arrive within 300 ms of `evident` for full credit; compute is charged on arrival |
| C3 | Start from the starter kit. Never modify `ttbench/`. `my_system.py` keeps the `ttbench.server` contract |
| C4 | `hello` must be sent within 10 s of connection, so models load and warm up **before** connecting |
| C5 | Shipped code and weights must be Apache-2.0, MIT or BSD. CC BY-NC data is for generation and testing only, and is declared |
| C6 | Never store checkpoint or final audio. Recordings need consent. No customer data |
| C7 | Runs on an Apple-Silicon Mac (primary) and on Windows or Linux CPU (secondary) |

**Assumptions**

- The hidden set contains human, unscripted Gulf speech with noise and dialect variety.
- Thresholds are read from `manifest.json` and may change before the final.
- The final run is on the team's Mac.

## 3. Interface contract (summary; `INTERFACE.md` governs)

- Transport: WebSocket on `ws://<host>:8765`.
- Server → harness: `{"type":"hello","team":"sarj-saud", ...}` sent once at start; then decision messages:
  `{"type":"decision","decision":"STOP"|"RESUME"|"TAKE_TURN","label":"interruption|backchannel|...","t_ms":<frame time>}`.
- Harness → server:
  - binary audio frames: 1280 bytes = 320 samples × 2 channels × int16, 16 kHz, interleaved, one every 20 ms;
  - JSON control messages, including agent text and segment events (see `INTERFACE.md`);
  - `ping`, which must be answered with `pong`.
- The harness timestamps each decision when it **arrives**, so the system must never queue decisions behind compute.

## 4. Architecture

### 4.1 Layers

| Layer | File | Purpose | Cost |
|---|---|---|---|
| L0 echo | `turn/echo.py` | Spectral comparison of the caller channel with the last 800 ms of agent audio (NFFT 512, Hann 320). Gives `resid_db` and `echo_db` | < 1 ms / frame |
| L1 VAD + level | `turn/vad.py` + policy | Silero VAD (512-sample chunks), plus a level gate relative to the caller reference U0 and the noise floor | < 1 ms |
| L2 speaker | `turn/speaker.py` | CAM++ (sherpa-onnx) embedding of ≤ 3 s, cosine vs the U0 print; Resemblyzer fallback | ~15 ms |
| L3 prosody | `turn/acoustic.py` | YIN f0, slope, range, effort, HF ratio, dynamics, decay, rate, all relative to U0 | ~5 ms |
| L4 ASR | `turn/asr.py` | Cohere Transcribe Arabic (MLX 8-bit) on Mac; faster-whisper `large-v3-turbo` elsewhere; MLX-Whisper legacy | ~97 ms on Mac |
| L4a lexicon | `turn/lexicon.py` | Gulf word classes: BID, WEAK_BID, AMBIG, POLITE, WAIT_WORDS, backchannels, fillers; agent-echo check | < 1 ms |
| L4b meaning | `turn/semantic.py` | CAMeLBERT-DA student (ONNX, max 64 tokens) distilled from an LLM teacher; outputs `sem_p` | ~7 ms |
| L5 fusion | `turn/policy.py` | Hard gates → rules → learned LR (veto/promote) → semantic veto/promote → STOP; RESUME watcher | < 1 ms |
| L6 EOT | `turn/policy.py`, `turn/eot.py` | Silence timers with optional Smart Turn v3.2 | < 1 ms (+10 ms Smart Turn) |
| L7 demo agent | `convo/` | Brain (API or local) + TTS for the live demo only, **not scored** | — |

### 4.2 Concurrency model
- **Sync loop** (the WebSocket handler thread) processes every frame: L0, L1, the state machine, and fusion on any finished worker result. It must never block for more than 5 ms.
- **Background worker** (`turn/clock.py`): single-flight. At most one ASR + L2/L3/L4b job runs at a time. A newer submission for the same candidate replaces a queued one, so stale reads are dropped.
- **Simulator honesty:** with `TT_SIM=1` the simulator advances a virtual clock by the worker's measured wall time, so fast runs are charged real latency. A simulator number can still be optimistic, so the official number is always from `ttbench.play`.

### 4.3 State machines

**Candidate (per caller utterance while the agent speaks)**

```
IDLE ──voiced ≥ MIN_VOICED_MS & level gate──► CANDIDATE
CANDIDATE ──every ASR_EVERY_MS / at pause──► submit(segment) ──result──► FUSE
FUSE ──p ≥ 0.5──► STOPPED (emit STOP, start RESUME watch)
FUSE ──p < 0.5──► CANDIDATE
CANDIDATE ──silence ≥ GAP_END_MS──► IDLE
CANDIDATE ──loud speech ≥ LONG_MS (within LONG_REL_DB of U0)──► STOPPED (safety net)
STOPPED ──sound ended ≥ RESUME_GAP_MS, voiced ≤ RESUME_MAX_VOICED_MS, full-read p < RESUME_P,
          elapsed ≤ RESUME_MS, no bid word──► emit RESUME ──► IDLE
```

**EOT (agent silent, caller speaking)**

```
LISTENING ──silence ≥ EOT_BASE_MS & answer complete──► TAKE_TURN
LISTENING ──silence & answer unfinished (filler, trailing connective, partial number)──► wait up to EOT_LONG_MS
(optional) Smart Turn P(complete) < EOT_SMART_TH ──► wait EOT_MID_MS
```

### 4.4 Repository layout

```
my_system.py        entry point (ttbench.server subclass)
turn/               decision engine, one module per function (see 4.1)
convo/              demo agent: brain.py, tts, dialogue loop
tools/              sim, loop, dashboard, synth_calls, augment_dev, teacher_gen,
                    train_policy, train_student, layer_eval, live, get_model, (errors.py: to add)
models/             silero_vad.onnx, policy-*.json, student*/  (large files via release)
data/               dev (kit), dev-aug, synth*, novel-test, live (git-ignored audio)
runs/               run outputs and sim-history.csv
docs/               PRD.md, TECH_SPEC.md, EXPERIMENTS.md, html explainers
ttbench/            KIT — READ ONLY
```

## 5. Component specifications

### 5.1 Entry point (`my_system.py`)
- Load and warm all models (ASR, VAD, speaker, student) **before** returning from setup, then send `hello`.
- Forward frames to `TurnTaker.on_frame(agent, caller, t_ms)`. Send each returned decision immediately.
- Answer `ping` from the handler thread, never behind the worker.

### 5.2 L0 echo suppression
- Keep a history of 40 agent frames (800 ms, the maximum echo delay).
- For each caller frame, compute band energies on a geometric band grid and the best-matching delayed agent spectrum. The output is `resid_db` (caller energy not explained by the agent) and `echo_db`.
- A frame counts as caller speech only if `resid_db` passes the level gate. The lexicon also flags `echoed` when the ASR text repeats words of the agent text heard in the last few seconds.

### 5.3 L1 VAD and level gate
- Speech requires Silero P ≥ `VAD_TH` = 0.5.
- The level must be no more than `REL_DB` = 10 dB below U0, and at least `SNR_DB` = 9 dB above the noise floor.
- U0 is the caller's level in their first answer. Before U0 exists, use `DEFAULT_REF_DB` = −21 dBFS.
- A stop needs the candidate's level within `STOP_REL_DB` = 8 dB of U0. This rejects TV and side speech.

### 5.4 L2 speaker verification
- Enrol from U0, using the first ≥ 1.5 s of clean caller speech.
- Similarity below `SPK_MIN` = 0.30 means another speaker, and the candidate is vetoed **unless** its level is ≥ U0 + `SPK_LOUD_DB` (−2 dB). This protects a caller who moves to the phone's loudspeaker.
- `USE_SPEAKER = 1` by default.

### 5.5 L3 prosody (relative)
- Features in `AC_KEYS`: `effort_rel, hf_rel, dyn_rel, decay_rel, per_rel, f0_rel, f0_range, f0_slope, rate_rel, voiced, …`. All are divided by or subtracted from the caller's U0 profile.
- Used by the learned fusion model. Phase P4 adds a dedicated rule for a lone ambiguous word (§5.10).

### 5.6 L4 ASR
- `TT_ASR` selects the backend: `cohere` (Mac default), `faster` (Windows/Linux default), or `whisper` (MLX-Whisper).
- The prompt (`TT_ASR_PROMPT`) primes Gulf backchannels and bid words. Context is the last `TT_ASR_CTX` = 10 s.
- Output: `{text, logprob, no_speech, ms}`.
- Hallucination filters:
  - Cohere's `فراغ` output (after a run of `@` tokens) means "no speech".
  - Bracketed Cohere text is treated as unreliable.
  - Content words count only if `logprob ≥ MIN_LOGPROB` (−0.9).
- A cache (`.cache/asr`) is on only in the simulator, so it never affects live runs.
- A global lock means one ASR call at a time; in the live demo the turn-taker and the dialogue share the model.

### 5.7 L4a lexicon (Gulf)
- **Classes:** `BID` (strong floor bids: «لا لا»، «لحظة»، «اسمع»، «طيب بس»…), `WEAK_BID`, `AMBIG` («طيب»، «خلاص»، «زين»), `POLITE`, `WAIT_WORDS`, backchannels («إيه»، «أها»، «مم»، «صح»، «والله»), and fillers («يعني»، «امم»، «ااه»).
- **Features:** `bid, content, bc, filler, echoed, repeated_ambig, ambig_alone, question, n, …` (see `FEATURES` in `policy.py`).
- `_strip_speaker_label` removes any "speaker:" prefixes that ASR emits.
- **Planned change (P3, task T5): two-tier split.**
  - `BID` keeps only unambiguous floor grabs, which stop on the first read.
  - New `CUE` holds question, negation and vocative words (ما، ايش، وين، يا أخي، اخوي…). These stop only if `sem_p ≥ 0.5` or there are ≥ 2 content words with no hesitation.
- **Planned (T5): thinking-aloud rule.** If the text contains a filler plus a trailing or partial structure (for example «يعني… تقريباً…»), content words do not count toward `CONTENT_MIN`.

### 5.8 L4b meaning student
- **Input:** `[agent tail, last ~12 words] [SEP] [caller text]`, max 64 tokens. **Output:** `sem_p` = P(caller wants the floor).
- The model lives in `TT_SEM_DIR` (default `models/student`) as `sem.onnx` + `tokenizer.json`. It is enabled when both files exist and `TT_SEM=1`.
- **Pipeline:** `tools/teacher_gen.py` → the teacher LLM labels synthetic and paraphrased (agent tail, caller text) pairs → `tools/train_student.py` fine-tunes CAMeLBERT-DA → exports ONNX.
- The teacher is used offline only. The student runs locally in about 7 ms.

### 5.9 L5 fusion
Order of evaluation on each worker result:

1. **Hard gates:**
   - fails the level or SNR gate → KEEP;
   - speaker veto (§5.4) → KEEP;
   - `echoed` with no bid → KEEP.
2. **Rules** (`_rules()`):
   - a BID word → STOP;
   - content ≥ `CONTENT_MIN` (2) new words, heard less than `CONTENT_MAX_AGE_S` (3 s) into continuous sound, and confident → STOP;
   - backchannel, filler or a lone ambiguous word → KEEP.
3. **Learned model** (`USE_MODEL = 2`, hybrid). Logistic regression over lexical + acoustic + speaker features, weights in `models/policy-{backend}.json`:
   - if a rule says STOP but model p < `HYB_TH` (0.2) → veto;
   - if the rules say KEEP but p ≥ `PROMOTE_TH` (0.95) and level within `PROMOTE_REL_DB` (4 dB) → STOP.
4. **Meaning:**
   - content words with `sem_p < SEM_LO` (0.25) do not stop the agent (TV, echo, chatter);
   - `sem_p ≥ SEM_HI` (0.90) with 2+ words, level within `SEM_REL_DB` (3 dB) and speaker similarity ≥ `SEM_SPK` (0.35) may stop even without a bid word, overruling the learned veto.
5. **Safety net:** sustained loud speech ≥ `LONG_MS` (2000 ms) within `LONG_REL_DB` (3 dB) → STOP.
6. **Label:** every decision carries the category that fired (`interruption`, `backchannel`, `noise`, `side_speech`, `echo`, `hesitation`, `ambiguous`).

### 5.10 Planned: early acoustic stop (P4, task T6)
- `EARLY_MS` (default 0 = off). When > 0, the engine may STOP before any ASR result if all of these hold:
  - voiced ≥ `EARLY_MS`;
  - level ≥ U0 − 3 dB;
  - speaker similarity ≥ 0.5;
  - rising effort.
- RESUME covers mistakes. It is enabled only if gates G1–G5 pass with it on.
- **Lone ambiguous word** («طيب»، «خلاص»): use f0 slope and effort relative to U0. Rising pitch or high effort → treat as a bid; flat → backchannel.

### 5.11 RESUME
- Allowed inside `RESUME_MS` = 460 ms after a STOP. This leaves 40 ms of margin before the scorer's 500 ms.
- Conditions:
  - the sound ended ≥ `RESUME_GAP_MS` (120 ms) ago;
  - total voiced ≤ `RESUME_MAX_VOICED_MS` (1200 ms);
  - the full-segment re-read gives p < `RESUME_P` (0.3);
  - no BID word.
- **Optional slow-brain confirmation** (mentor suggestion, off by default, `TT_SLOWBRAIN=1`). After a weak STOP, send (agent tail, caller text) to an LLM (Gemini Flash / OpenRouter). If it answers "backchannel" **within the RESUME window**, emit RESUME.
  - Its round-trip time must be logged.
  - If p90 > 300 ms on the venue network, it stays off.
  - It is never allowed to emit STOP.

### 5.12 L6 end of turn
- TAKE_TURN after `EOT_BASE_MS` = 600 ms of silence when the answer looks complete.
- Wait up to `EOT_LONG_MS` = 2500 ms when it looks unfinished: it ends in a filler or connective («و»، «بس»، «يعني»), contains a partly read number or date, or has a WAIT_WORD («لحظة»).
- Optional: Smart Turn v3.2 (`EOT_SMART = 1`). If P(complete) < 0.5, wait `EOT_MID_MS` = 1500 ms.

### 5.13 Parameter table (defaults)

| Name | Value | Module |
|---|---|---|
| FRAME_MS | 20 | policy |
| VAD_TH | 0.5 | L1 |
| REL_DB / SNR_DB / STOP_REL_DB | 10 / 9 / 8 dB | L1 |
| DEFAULT_REF_DB | −21 dBFS | L1 |
| MIN_VOICED_MS | 100 | state machine |
| ASR_EVERY_MS | 150 (try 100 in P4) | state machine |
| GAP_END_MS | 1300 | state machine |
| CONTENT_MIN / CONTENT_MAX_AGE_S | 2 / 3.0 s | rules |
| MIN_LOGPROB | −0.9 | ASR filter |
| LONG_MS / LONG_REL_DB | 2000 / 3 dB | safety net |
| SPK_MIN / SPK_LOUD_DB | 0.30 / −2 dB | L2 |
| USE_MODEL / MODEL_TH / HYB_TH / PROMOTE_TH / PROMOTE_REL_DB | 2 / 0.5 / 0.2 / 0.95 / 4 dB | L5 |
| SEM_LO / SEM_HI / SEM_REL_DB / SEM_SPK | 0.25 / 0.90 / 3 dB / 0.35 | L4b |
| RESUME_MS / RESUME_GAP_MS / RESUME_MAX_VOICED_MS / RESUME_P | 460 / 120 / 1200 / 0.3 | RESUME |
| EOT_BASE_MS / EOT_LONG_MS / EOT_MID_MS / EOT_SMART_TH | 600 / 2500 / 1500 / 0.5 | L6 |
| EARLY_MS (planned) | 0 (off) | L5 |

## 6. Data and training

### 6.1 Dataset registry and anti-leakage partitioning

| Set | Content | Use | May tune on it? |
|---|---|---|---|
| DEV (`data/dev`) | 24 Sarj calls | Official sanity check, real-time run | Only at category level, never per call |
| AUG (`dev-aug`) | DEV × {noise, narrowband phone, speakerphone, reverb} = 96 | Robustness gate | No |
| SYNTH-TRAIN (`synth`) | Generated calls from `tools/synth_calls.py` | Train the LR policy | Yes |
| HOLDOUT-V (`synth-test`, `synth-heldvoice`) | Unseen voices / sentences | Gate | No |
| HOLDOUT-P (`novel-test`, 120) | Phrasing written independently of the lexicon | Main generalisation gate | **Category-level fixes only; never copy its sentences** |
| HOLDOUT-P2 (`novel-test-2`) | New blind sentence bank (`data/novel_bank2.json`) | Final check, **opened once** | Never |
| LIVE (`data/live`) | Consented mic and phone recordings | Human-speech evidence | No |
| TEACHER | LLM-labelled text pairs | Student training | Yes |
| BANK | Sentence banks per category | Generation | — |

Rule: if a fix is motivated by a HOLDOUT-P failure, the fix must be expressed as a category rule or as new **different** training phrases, and it must also help another set.

### 6.2 Label schema
Same as `DATA.md`: per event `category` ∈ {interruption, backchannel, hesitation, noise, side_speech, echo, ambiguous}, `onset_ms`, `evident_ms`, expected action, and for EOT `turn_end_ms`. Our own recordings use exactly the same schema (`SCRIPTS.md` shows the script format).

### 6.3 Generators
- `tools/synth_calls.py` generates calls from scenario banks with Edge TTS / ElevenLabs voices, mixes agent and caller, and writes labels.
- `tools/augment_dev.py` adds noise (ESC-50, home and street), narrowband phone codec, speakerphone echo, and reverb.
- ElevenLabs free tier is non-commercial use only. It is test data only, and it is declared.

### 6.4 Policy training
`tools/train_policy.py`:
1. run the system with `NEVER_STOP=1` to dump features per candidate read;
2. join with labels;
3. fit LR (LightGBM is kept for comparison only);
4. write `models/policy-{backend}.json`. **Train per ASR backend**, because the word features differ between engines.

### 6.5 Teacher → student
`tools/teacher_gen.py` generates and labels pairs. `tools/train_student.py` trains CAMeLBERT-DA for 3 epochs and exports `models/student/sem.onnx`.
- Planned categories (T5): interruption openers in 14 dialects (اعذرني، من جد، لحظة لو سمحت…), thinking aloud, and speech aimed at someone in the room.

### 6.6 LIVE protocol
- Each teammate records 5–8 short calls by playing an agent clip while speaking naturally. Cover: quiet room, café, car, and speakerphone.
- Get written consent. Recordings live only on the team drive and in `data/live/`, which is git-ignored.
- Label with the `DATA.md` schema and run with `python -m ttbench.play --data data/live --calls all`.

## 7. Evaluation

### 7.1 Tools

| Command | What it does |
|---|---|
| `python -m tools.sim [--data DIR]` | Fast simulator with measured latency → `runs/sim-*/score.json`, `trace.json` |
| `python -m tools.loop --quick --note "..."` | dev + dev-aug + novel-test → appends `runs/sim-history.csv`, rebuilds the dashboard |
| `python -m tools.loop --sets dev,dev-aug,novel-test,synth-test` | Full gate run |
| `python -m ttbench.play --calls all [--data DIR]` | **Official** real-time run |
| `python -m ttbench.summary runs/<name> --failures` | Per-decision table and failures |
| `python -m tools.layer_eval` | Per-layer accuracy on the evaluation bank |
| `python -m tools.errors runs/<name>` (to add, T4) | Error taxonomy × category × dialect × noise |
| `python -m ttbench.ready` | Pre-submission check |
| `pytest -q` | Unit and protocol tests |

### 7.2 Gates (a PR that changes `turn/` must pass all of them)

| Gate | Condition |
|---|---|
| G1 | `novel-test` Core ≥ previous − 0.3 |
| G2 | `dev-aug` Core ≥ previous − 0.3 |
| G3 | False stops (sum over sets) not higher than before |
| G4 | `dev` Core ≥ 97 and 0 missed interruptions |
| G5 | Median stop latency not more than 50 ms worse |
| G6 | `pytest -q` green, `ttbench.ready` passes |
| G7 | Entry in `docs/EXPERIMENTS.md`: what changed, numbers before/after, decision |

### 7.3 Error taxonomy (workshop)
1. **False stop:** STOP on a backchannel, hesitation, noise, side speech or echo.
2. **Missed interruption:** no STOP before 1500 ms.
3. **Late decision:** STOP between 300 and 1500 ms after `evident`.
4. **Premature take-turn:** TAKE_TURN while the caller was mid-thought.

`tools/errors.py` reports each one with the call id, time, ASR text, the fired rule and `sem_p`, broken down by dialect and noise condition.

### 7.4 Ablation (for the write-up)
Run `novel-test` and `dev-aug` with each layer off in turn:
- no echo;
- no level gate;
- no speaker check (`USE_SPEAKER=0`);
- rules only (`USE_MODEL=0`);
- no meaning model (`TT_SEM=0`);
- no RESUME.

Report ΔCore, Δfalse stops and Δlatency.

### 7.5 Checkpoint run (Day 3)
Submit the current `main` tag. Record per-category results in `EXPERIMENTS.md`, then feed them into `tools/errors.py` priorities. Do not store the audio.

## 8. Non-functional requirements

| Area | Requirement |
|---|---|
| Latency budget | Onset → first ASR submit ~100 ms (MIN_VOICED) + ASR ~100 ms + fusion < 5 ms ⇒ **~200 ms typical**; worst case (queued read) ~380 ms |
| Resources | Scored process < 4 GB RAM; whole demo < 6 GB; nothing else heavy running during official runs |
| Reliability | 30-minute soak with no crash; every `ping` answered; reconnect-safe; one restart within 60 s allowed |
| Determinism | Same input in the simulator gives the same decisions (ASR cache on in sim) |
| Observability | `trace.json` per run: per read, the text, features, p, rule fired, worker ms |
| Privacy | No audio written outside `data/` and `runs/` (both git-ignored); checkpoint and final audio never saved |
| Portability | All paths come from `turn/config.py` / env (`TT_EXT_DIR`); no `open`-only or Mac-only calls in tools |

## 9. Local operations

### 9.1 Environment
- Python 3.11, virtualenv `.venv`.
- Mac: `./setup.sh` (MLX, Cohere, CAM++). Windows: `setup_windows.ps1` + `requirements-windows.txt` (see `WINDOWS.md`).

### 9.2 Models (in `TT_EXT_DIR`, default `~/Downloads/sarj-ext-data`, or in `models/`)

| Model | Path | Size | License |
|---|---|---|---|
| Silero VAD | `models/silero_vad.onnx` | 2 MB | MIT |
| Cohere Transcribe Arabic MLX 8-bit | `$TT_EXT_DIR/cohere-ar-mlx8` | ~1 GB | Apache-2.0 |
| Whisper large-v3-turbo (q4 MLX / faster-whisper) | `$TT_EXT_DIR/whisper-large-v3-turbo-q4` or HF cache | ~0.8 GB | MIT |
| CAM++ | `$TT_EXT_DIR/spk/campp.onnx` | 28 MB | Apache-2.0 |
| Student (CAMeLBERT-DA) | `models/student/` | ~420 MB | Apache-2.0 |
| Smart Turn v3.2 (optional) | `$TT_EXT_DIR/smart-turn/` | 8 MB | BSD-2 |
| Policy weights | `models/policy-{cohere,faster,whisper}.json` | KB | ours |

`python -m tools.get_model` downloads what is missing.

### 9.3 Environment variables

| Var | Default | Meaning |
|---|---|---|
| `TT_EXT_DIR` | `~/Downloads/sarj-ext-data` | External model folder (to add via `turn/config.py`) |
| `TT_ASR` | `cohere` on Mac, `faster` otherwise | ASR backend |
| `TT_FW_MODEL` | `large-v3-turbo` | faster-whisper model |
| `TT_ASR_PROMPT`, `TT_ASR_CTX`, `TT_ASR_CACHE` | see asr.py | ASR tuning |
| `TT_SEM`, `TT_SEM_DIR` | `1`, `models/student` | Meaning student |
| `TT_SIM` | unset | `1` inside the simulator |
| `TT_BRAIN` | auto | Demo brain: `gemini`, `local-url`, `claude-haiku`, `gemma4-26b`… |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | —, `gemini-3.5-flash-lite` | Gemini API demo brain |
| `TT_BRAIN_URL`, `TT_BRAIN_MODEL`, `TT_BRAIN_KEY` | — | Any other OpenAI-compatible endpoint |
| `ANTHROPIC_API_KEY` | — | Claude brain / teacher |
| `TT_SLOWBRAIN` (planned) | `0` | Optional LLM RESUME confirmation |

Keys live only in `.env` (git-ignored); `.env.example` lists the names.

### 9.4 Run-day runbook
1. Reboot the laptop. Close the browser, IDE indexers and Docker. Plug in power.
2. `git checkout <freeze-hash>`, then `git status` must show it clean.
3. `source .venv/bin/activate && python -m ttbench.ready`.
4. Start `python my_system.py`. Wait for "models warm". Confirm the harness gets `hello`.
5. Do not run anything else. Watch the logs for slow worker times over 250 ms.
6. Afterwards, check that no audio was written, and record the result in `EXPERIMENTS.md`.

## 10. Deliverables and presentation

| Deliverable | Owner | Done when |
|---|---|---|
| Real-time system | Tech lead | `ttbench.ready` passes on the freeze hash |
| `dev` evaluation + ≥ 5 failure cases | Data & testing | In `WRITEUP.md` with Arabic text, times and the cause |
| `WRITEUP.md` (2–4 pages) | Write-up | Sections below; every API, model and dataset declared |
| Commit hash + services list | Tech lead | Sent before the end of Day 5 |
| Demo app (`/demo`, port 8790) | Presentation | Live mic, replay, results view; offline fallback |
| Slides + backup video | Presentation | Rehearsed twice, at 7 min |

**WRITEUP structure:**
1. Problem and approach.
2. Architecture (layer table + diagram).
3. Why relative-to-caller signals.
4. Data and anti-leakage.
5. Results: per set, per category, latency, dialect and noise gaps.
6. Ablation.
7. Failure cases (≥ 5).
8. Practicality: cost, CPU/RAM, deployment.
9. Declared services and licences.
10. What we'd do next.

**Expected Q&A** (any teammate must answer):
- "Walk us from one frame to a STOP."
- "Why not just a VAD?" (B1 = 41).
- "How do you avoid overfitting to dev?"
- "What happens with a TV in the background?"
- "What's your latency and where does it go?"
- "What breaks it?"
- "Which parts are cloud?" (none in the scored path).

**Live demo protocol:**
- Use a headset mic.
- Script: «إيه» (keeps talking) → «أها مم» (keeps talking) → TV clip (ignored) → «لا لا لحظة» (stops) → «طيب» flat vs rising.
- Keep the layer view visible.
- If the network fails, switch to the local brain or the recorded video.

## 11. Team, milestones, Definition of Done

**Roles** (one owner each; every member can explain frame → STOP):

| ID | Role | Owns |
|---|---|---|
| R1 | Tech lead and integration | PRD/spec, repo, `my_system.py`, L5, gates, tags, freeze |
| R2 | Audio and decision signals | L0–L3, early stop, prosody |
| R3 | ASR and language | L4, L4a, L6 |
| R4 | Data and testing | generators, holdouts, teacher/student, LIVE, `tools/errors.py` |
| R5 | Presentation and write-up | phase documentation with screenshots from day one, demo, `WRITEUP.md`, talk |

**Milestones**

| ID | Milestone | Tag |
|---|---|---|
| M1 | P1 done: runs and measures on Mac and Windows | v1.0 |
| M2 | P2 error report live; checkpoint results analysed | v1.1 |
| M3 | P3: `novel-test` ≥ 90 with gates green | v1.2 |
| M4 | P4: latency and prosody improvements | v1.3 |
| M5 | P5: LIVE + `novel-test-2` measured once | v1.4 |
| M6 | Freeze: real-time run, write-up, hash sent | v2.0-final |

Tag format: `git tag -a vX.Y -m "dev NN.N · dev-aug NN.N · novel NN.N"`, and the README score table is updated in the same PR.

**Definition of Done (any task):**
- Code lives in the right module.
- Gates G1–G7 pass, or the task is non-`turn/` with tests passing.
- `EXPERIMENTS.md` entry written.
- PR merged into `main` via review.
- No secrets, no audio, nothing in `ttbench/` touched.

## 12. Risks

| Risk | Mitigation |
|---|---|
| Overfitting synthetic / `novel-test` | §6.1 partitioning; `novel-test-2` opened once; LIVE human data |
| ASR hallucination on noise | `فراغ`/bracket filters, logprob gate, level/speaker gates before ASR |
| Worker queueing causes late stops | Single-flight with stale-drop; `ASR_EVERY_MS` tuning; latency gate G5 |
| Echo on speakerphone | L0 + `echoed` lexicon flag + AUG speakerphone set |
| Laptop overload on run day | §9.4 runbook; demo brain on API |
| Slow-brain network latency | Off by default; never allowed to STOP; RTT logged |
| Licence issues | Appendix A register; only permissive weights shipped |

## Appendix A — Dependency and licence register

| Component | Use | Licence | In scored path |
|---|---|---|---|
| Silero VAD | L1 | MIT | yes |
| Cohere Transcribe Arabic | L4 (Mac) | Apache-2.0 | yes |
| Whisper / faster-whisper / MLX-Whisper | L4 alt | MIT | yes |
| 3D-Speaker CAM++ via sherpa-onnx | L2 | Apache-2.0 | yes |
| Resemblyzer (fallback) | L2 | Apache-2.0 | fallback |
| CAMeLBERT-DA | L4b student base | Apache-2.0 | yes |
| Smart Turn v3.2 | L6 optional | BSD-2 | optional |
| onnxruntime, numpy, websockets | runtime | MIT / BSD / BSD | yes |
| Claude (Anthropic API) | teacher labels; optional demo brain | commercial API | **no** |
| Gemini API (`gemini-3.5-flash-lite`) | demo brain; optional offline error labelling | commercial API | **no** |
| Edge TTS, ElevenLabs | synthetic test voices, demo voice | ToS (ElevenLabs free = non-commercial) | **no** |
| ESC-50 | noise for augmentation | CC BY-NC | **no** |

## Appendix B — Glossary
- **U0**: the caller's first answer, the reference for level, voice and pitch.
- **Evident**: the labelled moment the caller's intent becomes clear.
- **Candidate**: a caller sound during agent speech that is being judged.
- **Bid**: a word or phrase that claims the floor.
- **Backchannel**: a short "I'm listening" sound.
- **EOT**: end of turn.
- **Hybrid fusion**: rules decide, and the model may veto or promote.
- **Teacher/student**: a large LLM labels data offline, and a small local model learns it.

## Appendix C — References
`CHALLENGE.md`, `INTERFACE.md`, `SCORING.md`, `DATA.md`, `SCRIPTS.md`, `SUBMISSION.md`, `WINDOWS.md`, `docs/04-architecture.html`, `docs/EXPERIMENTS.md`, the organiser brief (15 Nov 2026), the KSU turn-taking workshop slides, and the mentor's team guidance (Abdulaziz Alghoraibi).
