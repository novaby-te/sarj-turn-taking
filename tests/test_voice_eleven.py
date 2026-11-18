"""The ElevenLabs voice option, against a fake server on this machine (no network, no real key)."""
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from convo import voice


@pytest.fixture
def eleven(monkeypatch, tmp_path):
    seen = {"status": 200}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"voices": [{"voice_id": "woman", "labels": {"gender": "female"}},
                                          {"voice_id": "man", "labels": {"gender": "male"}}]}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            seen.update(path=self.path, key=self.headers.get("xi-api-key"),
                        body=json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(seen["status"])
            self.end_headers()
            if seen["status"] == 200:
                self.wfile.write(np.array([1, 2, 3], dtype="<i2").tobytes())

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setattr(voice, "ELEVEN_URL", f"http://127.0.0.1:{httpd.server_port}")
    monkeypatch.setattr(voice, "CACHE", tmp_path)
    monkeypatch.setattr(voice, "_eleven_voices", {})
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    monkeypatch.delenv("ELEVENLABS_VOICE_ID", raising=False)
    yield seen
    httpd.shutdown()


def test_speaks_with_a_voice_of_the_right_gender(eleven):
    x, used = asyncio.run(voice.tts("هلا والله", "m", "eleven"))
    assert used == "eleven" and x.tolist() == [1, 2, 3]
    assert eleven["path"].startswith("/text-to-speech/man?output_format=pcm_24000")
    assert eleven["key"] == "test-key" and eleven["body"]["text"] == "هلا والله"


def test_a_chosen_voice_wins(eleven, monkeypatch):
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "my-voice")
    asyncio.run(voice.tts("هلا", "f", "eleven"))
    assert eleven["path"].startswith("/text-to-speech/my-voice?")


def test_falls_back_when_elevenlabs_fails(eleven, monkeypatch):
    eleven["status"] = 401

    async def edge(text, name, timeout=8.0):
        return np.zeros(4, np.int16)
    monkeypatch.setattr(voice, "_edge", edge)
    assert asyncio.run(voice.tts("هلا", "f", "eleven"))[1] == "edge"
