"""Project settings: reads the git-ignored `.env` file into the environment."""
from __future__ import annotations

import os
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]


def load_env(path: Path = KIT / ".env") -> None:
    """Load KEY=VALUE lines from `.env`. Values already set in the environment win, and empty values are skipped."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        value = value.strip().strip("'\"")
        if value:
            os.environ.setdefault(key.strip(), value)


load_env()
EXT_DIR = Path(os.environ.get("TT_EXT_DIR") or Path.home() / "Downloads" / "sarj-ext-data").expanduser()   # big models and data live outside git
