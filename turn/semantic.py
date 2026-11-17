"""Level 4b, MEANING: does the caller want the floor, judging by what the words mean in context?

A small Arabic BERT (CAMeLBERT-DA, trained on billions of dialectal Arabic words, Apache-2.0) fine-tuned on
thousands of caller moments written and labelled by Claude Opus (the teacher), across 14 dialects and 12 business
domains (tools/teacher_gen.py -> tools/train_student.py). It reads [agent sentence] + [caller words] and returns
P(STOP). The teacher needs ~3.5 s per decision; this student needs a few milliseconds on a laptop CPU, so it can
sit inside the 20 ms decision loop. ONNX (int8) + the `tokenizers` library only: runs on macOS, Windows, Linux.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parents[1]
DIR = Path(os.environ.get("TT_SEM_DIR", str(KIT / "models" / "student")))
MAX_LEN = 64
_sess = None
_tok = None
_lock = threading.Lock()


def available() -> bool:
    return (DIR / "sem.onnx").exists() and (DIR / "tokenizer.json").exists() and os.environ.get("TT_SEM", "1") == "1"


def _load():
    global _sess, _tok
    if _sess is None:
        import onnxruntime as ort
        from tokenizers import Tokenizer
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1
        _sess = ort.InferenceSession(str(DIR / "sem.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
        _tok = Tokenizer.from_file(str(DIR / "tokenizer.json"))
        _tok.enable_truncation(MAX_LEN)
        _tok.no_padding()


def agent_tail(agent_text: str, n_words: int = 24) -> str:
    return " ".join(agent_text.split()[-n_words:])


def p_stop(agent_text: str, caller_text: str) -> float | None:
    """P(the caller wants the floor) from meaning alone. None if the model is not installed or there are no words."""
    if not caller_text.strip() or not available():
        return None
    with _lock:
        _load()
        enc = _tok.encode(agent_tail(agent_text), caller_text.strip())
    ids = np.array([enc.ids], dtype=np.int64)
    feeds = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
    if "token_type_ids" in {i.name for i in _sess.get_inputs()}:
        feeds["token_type_ids"] = np.array([enc.type_ids], dtype=np.int64)
    logits = _sess.run(None, feeds)[0][0]
    e = np.exp(logits - logits.max())
    return float(e[1] / e.sum())


def warmup():
    if available():
        p_stop("حياك الله", "إيه")
