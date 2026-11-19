#!/usr/bin/env bash
# One-time setup of a Linux cloud machine (Claude Code cloud session, CPU only) for the v3 training run.
# Everything heavy lives here, never on a team laptop.     bash cloud/setup_linux.sh
set -euo pipefail
cd "$(dirname "$0")/.."
KIT=$(pwd)
export TT_EXT_DIR="${TT_EXT_DIR:-$HOME/sarj-ext-data}"
mkdir -p "$TT_EXT_DIR" data models runs
REL=https://github.com/o6lq/sarj-turn-taking/releases/download
PY=$(command -v python3.12 || command -v python3.11 || command -v python3)
echo "python: $($PY --version)  ext: $TT_EXT_DIR"

# 1) Python packages, CPU builds (no Apple MLX here)
[ -d .venv ] || $PY -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q --index-url https://download.pytorch.org/whl/cpu torch
pip install -q numpy scipy soundfile onnxruntime onnx tokenizers "transformers>=4.44" huggingface_hub \
    edge-tts websockets pytest scikit-learn faster-whisper anthropic

# 2) the base model the student is fine-tuned from (CAMeLBERT-DA, Apache-2.0)
if [ ! -f "$TT_EXT_DIR/camelbert-da/config.json" ]; then
  python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("CAMeL-Lab/bert-base-arabic-camelbert-da", local_dir="$TT_EXT_DIR/camelbert-da")
PY
fi

# 3) the official student as the baseline to beat (models-v1 release)
mkdir -p models/student-base
# BASE_TAG: the release holding the student the Mac really uses (see docs/CLOUD_TRAINING.md §0)
BASE_TAG="${BASE_TAG:-models-v1}"
[ -f models/student-base/sem.onnx ] || curl -fsSL -o models/student-base/sem.onnx "$REL/$BASE_TAG/student-sem.onnx"
[ -f models/student-base/tokenizer.json ] || curl -fsSL -o models/student-base/tokenizer.json "$REL/$BASE_TAG/student-tokenizer.json"
sha256sum models/student-base/sem.onnx

# 4) data sets (data-v1 release). novel-test is for TESTING and decontamination only, never training.
fetch_set () {
  local name=$1
  [ -f "data/$name/manifest.json" ] && return 0
  local tmp; tmp=$(mktemp -d)
  curl -fsSL -o "$tmp/$name.zip" "$REL/data-v1/$name.zip"
  python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$tmp/$name.zip" "$tmp/x"
  local m; m=$(find "$tmp/x" -name manifest.json -maxdepth 3 | head -1)
  mkdir -p "data/$name" && cp -r "$(dirname "$m")"/. "data/$name/"
  rm -rf "$tmp"
  echo "data/$name: $(ls data/$name/labels | wc -l) calls"
}
for s in synth synth-test dev-aug novel-test; do fetch_set "$s"; done

# 5) ESC-50 noises for generating new calls (CC BY-NC 3.0, research use)
if [ ! -f "$TT_EXT_DIR/ESC-50-master/meta/esc50.csv" ]; then
  curl -fsSL -o "$TT_EXT_DIR/esc50.zip" https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip
  python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$TT_EXT_DIR/esc50.zip" "$TT_EXT_DIR"
  rm "$TT_EXT_DIR/esc50.zip"
fi

# 6) the teacher data the official student was trained on, when the team has published it (optional)
if [ ! -d data/teacher/raw ] && curl -fsIL "$REL/teacher-v1/teacher-raw.zip" >/dev/null 2>&1; then
  curl -fsSL -o /tmp/teacher-raw.zip "$REL/teacher-v1/teacher-raw.zip"
  mkdir -p data/teacher && python -c "import zipfile; zipfile.ZipFile('/tmp/teacher-raw.zip').extractall('data/teacher')"
fi
python - <<'PY'
import glob, json
print("teacher items already here:", sum(len(json.load(open(p, encoding="utf-8"))) for p in glob.glob("data/teacher/raw/*.json")))
PY

python -m tools.hard_bank
echo "SETUP DONE"
