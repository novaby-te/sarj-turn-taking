"""The demo brain's cloud path, against a fake server on this machine (no network, no real key)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from convo import brain
from turn.config import load_env


@pytest.fixture
def server():
    seen = {"status": 200}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["auth"] = self.headers.get("Authorization")
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(seen["status"])
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            if seen["status"] == 200:
                for piece in ("هلا ", "والله"):
                    chunk = {"choices": [{"delta": {"content": piece}}]}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    seen["url"] = f"http://127.0.0.1:{httpd.server_port}/chat/completions"
    yield seen
    httpd.shutdown()


@pytest.fixture
def gemini(server, monkeypatch):
    monkeypatch.setattr(brain, "GEMINI_URL", server["url"])
    monkeypatch.setattr(brain, "_current", "gemini", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    return server


def test_gemini_streams_with_the_key_and_model(gemini):
    assert "".join(brain.stream([{"role": "user", "content": "هلا"}])) == "هلا والله"
    assert gemini["auth"] == "Bearer test-key"
    assert gemini["body"]["model"] == brain.GEMINI_MODEL


def test_claude_haiku_takes_over_when_gemini_fails(gemini, monkeypatch):
    gemini["status"] = 503
    used = []
    monkeypatch.setattr(brain, "_claude_ready", lambda: True)
    monkeypatch.setattr(brain, "_claude_stream", lambda m, t, name=None: used.append(name) or iter(["بديل"]))
    assert "".join(brain.stream([{"role": "user", "content": "هلا"}])) == "بديل"
    assert used == ["claude-haiku"]


def test_error_is_raised_when_no_fallback_exists(gemini, monkeypatch):
    gemini["status"] = 503
    monkeypatch.setattr(brain, "_claude_ready", lambda: False)
    with pytest.raises(OSError):
        list(brain.stream([{"role": "user", "content": "هلا"}]))


def test_gemini_is_the_default_once_it_has_a_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("TT_BRAIN", "gemini")
    assert brain.default_model() == "gemini"
    assert next(iter(brain.available())) == "gemini"


def test_without_a_key_a_cloud_brain_still_beats_the_local_model(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("TT_BRAIN_URL", raising=False)
    monkeypatch.setenv("TT_BRAIN", "gemini")
    monkeypatch.setattr(brain, "_claude_ready", lambda: True)
    assert brain.default_model() == "claude-haiku"


def test_load_env_keeps_existing_values_and_skips_empty_ones(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# note\nTT_A=from-file\nTT_B='quoted'\nTT_C=\nTT_D=from-file\n", encoding="utf-8")
    for key in ("TT_A", "TT_B", "TT_C"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TT_D", "from-shell")
    load_env(env)
    import os
    assert (os.environ["TT_A"], os.environ["TT_B"], os.environ["TT_D"]) == ("from-file", "quoted", "from-shell")
    assert "TT_C" not in os.environ
    monkeypatch.delenv("TT_A")
    monkeypatch.delenv("TT_B")
