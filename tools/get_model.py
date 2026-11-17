"""Download a whole Hugging Face model repo (public, ungated) safely and resumably.

Only weight formats that cannot run code are accepted (safetensors / onnx / npz / gguf); pickle files (.bin, .pt,
.pth, .ckpt) are refused. Big files go through tools/fetch.py (parallel ranges, resumable).

    .venv/bin/python -m tools.get_model mlx-community/gemma-4-26b-a4b-it-4bit ~/Downloads/sarj-ext-data/gemma-4-26b-a4b-4bit
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
UNSAFE = (".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle")
BIG = 50 * 1024 * 1024


def files(repo: str, rev: str = "main") -> list[dict]:
    url = f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=1"
    with urllib.request.urlopen(url, timeout=30) as r:
        return [f for f in json.load(r) if f.get("type") == "file"]


def main():
    repo, dest = sys.argv[1], Path(sys.argv[2]).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    fl = files(repo)
    bad = [f["path"] for f in fl if f["path"].endswith(UNSAFE)]
    if bad and not any(f["path"].endswith((".safetensors", ".onnx", ".npz", ".gguf")) for f in fl):
        sys.exit(f"refused: {repo} only ships pickle weights {bad[:3]}")
    for f in fl:
        p = f["path"]
        if p.endswith(UNSAFE) or p.startswith(".") or p.endswith((".md", ".png", ".jpg", ".gif")) and p != "README.md":
            continue
        out = dest / p
        size = f.get("lfs", {}).get("size") or f.get("size") or 0
        if out.exists() and (not size or out.stat().st_size == size):
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{repo}/resolve/main/{p}"
        if size > BIG:
            subprocess.run([sys.executable, str(KIT / "tools" / "fetch.py"), url, str(out), "--parts", "16"], check=True)
        else:
            urllib.request.urlretrieve(url, out)
        print("ok", p, size, flush=True)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
