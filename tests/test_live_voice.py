import asyncio
import base64

from convo.live_voice import LiveVoice


def audio(data=b"ab", words=""):
    return {"serverContent": {"modelTurn": {"parts": [{"inlineData": {"data": base64.b64encode(data).decode()}}]},
                              "outputTranscription": {"text": words}}}


DONE = {"serverContent": {"turnComplete": True}}


def drain(voice):
    items = []
    while not voice._queue.empty():
        items.append(voice._queue.get_nowait())
    return items


def test_a_reply_streams_until_the_turn_completes():
    voice = LiveVoice("", "m")
    voice.active = True
    voice.handle(audio(b"ab", "هلا"))
    voice.handle(DONE)
    assert drain(voice) == [(b"ab", "هلا"), None]
    assert not voice.active


def test_audio_from_a_replaced_turn_is_dropped():
    voice = LiveVoice("", "m")
    voice.active = voice._replaced = True
    voice.handle(audio(b"old"))
    voice.handle({"serverContent": {"interrupted": True}})
    voice.handle(DONE)
    voice.handle(audio(b"new"))
    assert drain(voice) == [(b"new", "")]
    assert voice.active


def test_audio_after_the_caller_interrupts_is_dropped():
    voice = LiveVoice("", "m")
    voice.active = True
    voice.cancel()
    voice.handle(audio(b"late"))
    voice.handle(DONE)
    assert drain(voice) == []
    assert not voice.active


def test_speaking_on_a_closed_session_fails_clearly():
    async def run():
        async for _ in LiveVoice("", "m").speak("هلا"):
            pass
    try:
        asyncio.run(run())
    except ConnectionError:
        return
    raise AssertionError("expected ConnectionError")
