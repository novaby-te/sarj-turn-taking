#!/bin/zsh
# One-time setup on a teammate's Mac (Apple Silicon). Safe to run again: every step skips what already exists.
#   ./setup.sh            # code + models + our generated data
#   ./setup.sh --no-data  # skip the ~500 MB of generated calls
set -e
cd "$(dirname "$0")"
EXT="$HOME/Downloads/sarj-ext-data"          # big files live outside git
mkdir -p "$EXT"

echo "1/4  Python 3.12 environment"
PY=$(command -v python3.12 || echo /opt/homebrew/bin/python3.12)
if [ ! -x "$PY" ]; then echo "Need Python 3.12:  brew install python@3.12"; exit 1; fi
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements-full.txt

echo "2/4  Whisper (speech-to-text, 464 MB)"
W="$EXT/whisper-large-v3-turbo-q4"
mkdir -p "$W"
[ -f "$W/config.json" ] || curl -sL -o "$W/config.json" https://huggingface.co/mlx-community/whisper-large-v3-turbo-q4/resolve/main/config.json
[ -f "$W/weights.npz" ] || .venv/bin/python tools/fetch.py https://huggingface.co/mlx-community/whisper-large-v3-turbo-q4/resolve/main/weights.npz "$W/weights.npz" --parts 16

echo "2b/4 Cohere Transcribe Arabic (default speech-to-text, 2.4 GB, Apache-2.0, 8-bit MLX build)"
C="$EXT/cohere-ar-mlx8"; mkdir -p "$C"
CR=https://huggingface.co/majentik/cohere-transcribe-arabic-07-2026-MLX-8bit/resolve/main
for f in PROVENANCE.md config.json generation_config.json model.safetensors.index.json normalizer.json preprocessor_config.json processor_config.json special_tokens_map.json tokenizer.json tokenizer.model tokenizer_config.json; do
  [ -f "$C/$f" ] || curl -sfL -o "$C/$f" "$CR/$f"; done
[ -f "$C/model.safetensors" ] || .venv/bin/python tools/fetch.py "$CR/model.safetensors" "$C/model.safetensors" --parts 16

echo "2c/4 Meaning student (models/student, 110 MB, GitHub release models-v1)"
mkdir -p models/student
[ -f models/student/sem.onnx ] || gh release download models-v1 -R o6lq/sarj-turn-taking -p student-sem.onnx -O models/student/sem.onnx
[ -f models/student/tokenizer.json ] || gh release download models-v1 -R o6lq/sarj-turn-taking -p student-tokenizer.json -O models/student/tokenizer.json
echo "2d/4 Gemma 4 26B-A4B (agent brain, 15 GB, Apache-2.0)"
.venv/bin/python -m tools.get_model mlx-community/gemma-4-26b-a4b-it-4bit "$EXT/gemma-4-26b-a4b-4bit"

echo "3/4  Generated training/test calls (GitHub release data-v1)"
if [[ "$1" != "--no-data" ]]; then
  for set in synth synth-test dev-aug; do
    if [ ! -f "data/$set/manifest.json" ]; then
      gh release download data-v1 -R o6lq/sarj-turn-taking -p "$set.zip" -D /tmp --clobber
      unzip -q -o "/tmp/$set.zip" -d data && rm "/tmp/$set.zip"
    fi
  done
fi

echo "4/4  Check: the official scorer on Sarj's 24 calls (about a minute)"
.venv/bin/python -m tools.sim | head -1
echo "Done. Try it:  .venv/bin/python -m tools.live    (opens the live page)"
