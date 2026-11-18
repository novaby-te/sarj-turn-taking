from types import SimpleNamespace

from convo import brain
from convo.engine import Conversation


def call(feats, echo_flags=None):
    conv = Conversation.__new__(Conversation)
    conv.tt = SimpleNamespace(cand={"feats": feats, "start": 0})
    conv.ms = 1000
    conv.echo_flags = [False] * 50 if echo_flags is None else echo_flags
    return conv


def words(n=1, echo_ratio=0.0):
    return {"n": n, "echo_ratio": echo_ratio, "bid": 0, "content": n, "repeated_ambig": False}


def test_a_single_word_from_the_caller_can_stop_the_agent():
    assert call(words(1))._real_words()


def test_the_agents_own_voice_cannot_stop_it():
    assert not call(words(3, echo_ratio=0.8))._real_words()
    assert not call(words(2), echo_flags=[True] * 50)._real_words()


def test_no_words_cannot_stop_the_agent():
    assert not call(words(0))._real_words()


def test_short_comma_phrases_stay_in_one_sentence():
    assert brain._SENT.match("حياك الله، حبيت أذكرك بالدفعة") is None
    assert brain._SENT.match("حياك الله، حبيت أذكرك بالدفعة. ").group(1) == "حياك الله، حبيت أذكرك بالدفعة."
