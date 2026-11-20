#!/usr/bin/env bash
# v3 training run, start to finish, on the cloud machine (after cloud/setup_linux.sh). Resumable: rerun to continue.
#
#   STAGES="teacher clean train eval hardset e2e package" bash cloud/run_v3.sh
#
# Settings (environment):
#   TEACHER_BACKEND  claude (Claude Code CLI, default) | anthropic (ANTHROPIC_API_KEY) | gemini (GEMINI_API_KEY)
#   TEACHER_MODEL    default claude-opus-5-5 (claude/anthropic) or gemini-3.5-flash-lite (gemini)
#   CALLS            teacher calls per focus (40 items each), default 60  → 5 focuses × 60 × 40 ≈ 12k new items
#   EPOCHS           default 3
#   FOCUSES          teacher focuses to run, default "hard pairs live answers default" (also: offtopic)
#   WORKERS          parallel teacher requests, default 4 (use 2 when the API answers 503 "high demand")
# The official model in models/student is never touched: the candidate goes to models/student-v3.
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export TT_EXT_DIR="${TT_EXT_DIR:-$HOME/sarj-ext-data}"
STAGES="${STAGES:-teacher clean train eval hardset package}"
BACKEND="${TEACHER_BACKEND:-claude}"
if [ "$BACKEND" = gemini ]; then MODEL="${TEACHER_MODEL:-gemini-3.5-flash-lite}"; else MODEL="${TEACHER_MODEL:-claude-opus-5-5}"; fi
CALLS="${CALLS:-60}"
OUT=runs/v3
mkdir -p "$OUT"
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/run.log"; }
has () { [[ " $STAGES " == *" $1 "* ]]; }

if has teacher; then
  # new files start at 5000 so they never overwrite the team's original teacher files (0000-0999)
  # each focus has a fixed file range, so a later run with FOCUSES="offtopic default" only fills those ranges
  for focus in ${FOCUSES:-hard pairs live answers default}; do
    case $focus in hard) off=5000;; pairs) off=6000;; live) off=7000;; answers) off=8000;; default) off=9000;; offtopic) off=10000;; *) echo "unknown focus $focus"; exit 2;; esac
    f=$focus; [ "$f" = default ] && f=""
    log "teacher: focus=${focus} calls=${CALLS} backend=${BACKEND} model=${MODEL}"
    python -m tools.teacher_gen --calls "$CALLS" --workers "${WORKERS:-4}" --backend "$BACKEND" --model "$MODEL" --focus "$f" --offset "$off" \
      | tee -a "$OUT/teacher.log" | grep -E "FAILED|ALLDONE" || true
  done
fi

if has clean; then
  log "decontaminate: drop anything close to novel-test / hard test texts"
  python -m tools.decontam | tee -a "$OUT/run.log"
  cp runs/decontam-report.json "$OUT/"
fi

if has train; then
  log "train student v3 (CPU) epochs=${EPOCHS:-3}"
  python -m tools.train_student --out models/student-v3 --epochs "${EPOCHS:-3}" 2>&1 | tee "$OUT/train.log" | grep -E "train |epoch|RESULT" || true
fi

if has eval; then
  log "text gate: official student vs v3 (held-out real recogniser text + hard suite)"
  set +e
  python -m tools.student_eval --base models/student-base --cand models/student-v3 --out "$OUT/student-eval.json" | tee -a "$OUT/run.log"
  echo $? > "$OUT/gate.exit"
  set -e
fi

if has hardset; then
  log "render the hard audio test set (160 calls, noisy, phone band, new sentences)"
  python -m tools.synth_calls --bank data/hard_bank_v1.json --hard --n 160 --seed 31 --prefix H --out data/hard-test | tail -2
fi

if has e2e; then
  # Rough whole-system check with faster-whisper (the Mac uses Cohere, so only the base-vs-v3 DIFFERENCE means something)
  for who in base v3; do
    dir=models/student-base; [ $who = v3 ] && dir=models/student-v3
    log "e2e (faster-whisper) student=$who"
    TT_ASR=faster TT_SEM_DIR="$dir" python -m tools.loop --sets dev,dev-aug,novel-test,synth-test,data/hard-test \
      --note "cloud e2e faster student=$who" --no-open | tee -a "$OUT/e2e-$who.log"
  done
fi

if has package; then
  log "package for the Mac: student v3 + hard test set + reports"
  rm -rf "$OUT/pkg" && mkdir -p "$OUT/pkg"
  python - <<'PY'
import shutil
shutil.make_archive("runs/v3/pkg/student-v3", "zip", "models/student-v3")
shutil.make_archive("runs/v3/pkg/hard-test", "zip", "data", "hard-test")
PY
  if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
    tag="v3-candidate-$(date +%Y%m%d-%H%M)"
    gh release create "$tag" runs/v3/pkg/*.zip "$OUT/student-eval.json" --prerelease --title "$tag" \
      --notes "Candidate from cloud/run_v3.sh. Not official until measured on the Mac (docs/CLOUD_TRAINING.md §4)."
    log "uploaded release $tag"
  else
    # no release rights here: split into parts under GitHub's 100 MB file limit, for a separate artifacts branch
    (cd "$OUT/pkg" && for z in *.zip; do split -b 90m -d "$z" "$z.part-" && rm "$z"; done)
    log "split into parts in $OUT/pkg (join on the Mac: cat X.zip.part-* > X.zip)"
  fi
fi
log "DONE stages: $STAGES"
