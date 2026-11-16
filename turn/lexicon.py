"""What the caller's words mean for turn taking, in Saudi and Gulf Arabic.

Three groups:
  BACKCHANNEL  "I'm following, carry on"         إيه، أها، مم، زين، هيه، صح، والله ...
  FILLER       "thinking"                         اممم، ااا، يعني ...
  BID          "I want the floor"                 لا، لحظة، بس، اسمع، استنى، كم، وش، ممكن، أبي ...
Everything else is CONTENT: new information is itself a claim on the floor once there are enough words.
Words the caller repeats from the agent's own sentence (saying it along with the agent) count as backchannel.
"""
from __future__ import annotations

import re

_DIAC = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_PUNCT = re.compile(r"[^\w\s؀-ۿ]|[_\d٠-٩]|[،؛؟]")


def norm(t: str) -> str:
    t = _DIAC.sub("", t)
    t = re.sub("[إأآٱ]", "ا", t)
    t = t.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    t = _PUNCT.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def squeeze(w: str) -> str:
    """ممممم -> مم, هههه -> هه, اييييه -> ايه (elongation)."""
    return re.sub(r"(.)\1{2,}", r"\1\1", w)


BACKCHANNEL = {norm(w) for w in """
ايه ايوه ايوا ايوة اي اى ايي اها اه اهاه اهه مم ممم ام هم همم هيه هييه هي يب يس
نعم اكيد اكيده طبعا تمام زين صح صحيح مضبوط بالضبط معك معاك معاي معايا وياك فاهم فاهمه فاهمك
اوكي اوك اوكيه اوكى ماشي حلو حلوه جميل ممتاز الله والله صادق صادقه عارف عارفه طيب
هه ههه هههه ههههه ههههههه
""".split()}
FILLER = {norm(w) for w in "امم اممم ااا اا ام يعني ممم اه اهه هممم".split()}
# words that open a turn, ask for one, object, or ask a question
BID = {norm(w) for w in """
لا لحظه لحظة دقيقه ثانيه ثواني استنى استني اصبر انتظر انتظري بس اسمع اسمعي اسمعني شوف شوفي
ممكن ابي ابغى ابغي ابا ودي اريد بدي عندي سؤال استفسار كم وش ايش شنو شو ليش ليه متى وين كيف مين منو اي
مو مب مش ما مافي ماني مابي مابغى
سوري عفوا لوسمحت لو سمحت سمحت يااخي اخي اخوي خويي
غلط خطا ماطلبت
""".split()} - {"اي"}
# «خلاص» and «طيب» are the brief's ambiguous pair: repeated or followed by more words = floor bid
AMBIG = {norm(w) for w in "خلاص طيب".split()}

# Whisper sometimes "hears" these in noise or silence: never trust them
HALLUCINATIONS = [norm(p) for p in [
    "اشتركوا في القناة", "اشترك في القناة", "شكرا للمشاهده", "ترجمه نانسي قنقر", "نانسي قنقر", "موسيقى",
    "السلام عليكم ورحمه الله", "تابعونا", "لا تنسوا الاشتراك", "سبحان الله", "قناه", "المترجم",
]]


def cut_loop(words: list[str]) -> list[str]:
    """Whisper sometimes gets stuck repeating: «أها، ممتازة، بس، صحيح، بس، صحيح، بس...». Everything from the
    first repeated cycle on is unreliable, so keep only what came before it."""
    for n, reps in ((1, 4), (2, 3), (3, 3)):          # «لا لا لا» is real; four of the same is a loop
        for i in range(len(words) - reps * n + 1):
            a = words[i:i + n]
            if all(words[i + k * n:i + (k + 1) * n] == a for k in range(1, reps)):
                return words[:i]
    return words


def analyse(text: str, agent_text: str = "") -> dict:
    """Turn the caller's words into simple, explainable features."""
    t = norm(text)
    halluc = any(h and h in t for h in HALLUCINATIONS)
    full = [squeeze(w) for w in t.split()]
    raw = cut_loop(full)
    if len(raw) < len(full):          # the decoder looped: this whole read is unreliable
        halluc = True
    words = raw if not halluc else []
    agent_words = set(norm(agent_text).split())
    bc = fill = bid = content = echoed = amb = 0
    first_bid = None
    for i, w in enumerate(words):
        if w in BID:
            bid += 1
            first_bid = i if first_bid is None else first_bid
        elif w in AMBIG:
            amb += 1
        elif w in BACKCHANNEL or w.startswith("هه") or re.fullmatch(r"(ا|ي|ه|م)+", w or "x"):
            bc += 1
        elif w in FILLER:
            fill += 1
        elif len(w) >= 3 and w in agent_words:
            echoed += 1
        else:
            content += 1
    repeated_amb = amb >= 2 or (amb >= 1 and (content + bid) >= 1)
    question = "؟" in text or "?" in text
    return {"words": words, "n": len(words), "bc": bc, "filler": fill, "bid": bid, "content": content,
            "echoed": echoed, "ambig": amb, "repeated_ambig": repeated_amb, "question": question,
            "first_bid": first_bid, "hallucination": halluc}


ARABIC_DIGITS = {norm(w): i for i, w in enumerate(
    "صفر واحد اثنين ثلاثه اربعه خمسه سته سبعه ثمانيه تسعه".split())}
ARABIC_DIGITS.update({"تلاته": 3, "اتنين": 2, "تنين": 2, "تمانيه": 8, "سابع": 7, "سبعه": 7, "ستّه": 6,
                      "اثنين": 2, "ثنين": 2, "ثلاث": 3, "اربع": 4, "خمس": 5, "ست": 6, "سبع": 7, "ثمان": 8, "تسع": 9})
NUMBER_WORDS = {"واحد": 1, "اثنين": 2, "ثلاث": 3, "اربع": 4, "خمس": 5, "ست": 6, "سبع": 7, "ثمان": 8, "تسع": 9, "عشر": 10}


def count_digits(text: str) -> int:
    t = norm(text)
    n = sum(1 for w in t.split() if w in ARABIC_DIGITS or w.lstrip("و") in ARABIC_DIGITS)
    return n + len(re.findall(r"\d", text))


def expected_digits(agent_text: str) -> int | None:
    """«هو سبع أرقام» -> 7. The agent tells us how long the caller's answer should be."""
    t = norm(agent_text)
    for m in re.finditer(r"(\S+) (ارقام|رقم|خانات)", t):
        w = m.group(1)
        for k, v in NUMBER_WORDS.items():
            if w.startswith(k):
                return v
    return None


def is_incomplete(text: str) -> bool:
    """Ends on a word that promises more: a filler, «و», «يعني», «بس»..."""
    words = norm(text).split()
    if not words:
        return False
    last = squeeze(words[-1])
    if any(w in WAIT_WORDS for w in words[:4]) and len(words) <= 4:
        return True                    # «لحظة أشوفه» / «خلني أشوف»: they asked us to wait
    return last in FILLER or last in {"و", "يعني", "بس", "لان", "عشان", "اللي", "في", "من", "على", "مع"} or text.rstrip().endswith("…")


WAIT_WORDS = {norm(w) for w in "لحظه لحظة ثانيه دقيقه اشوف اشوفه اشوفوا خلني خليني استنى انتظر".split()}
