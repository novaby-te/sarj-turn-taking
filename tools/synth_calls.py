"""Generate NEW Arabic calls in exactly the starter-kit format (stereo WAV + label JSON + manifest), so the
official scorer and our simulator run on them unchanged.

Sources (all disclosed in the write-up):
  * voices: Microsoft Edge neural TTS (edge-tts), 12 Gulf voices (SA, KW, BH, QA, AE, OM), random pitch/rate
            per call so speakers differ; word timings give the exact moment the decisive word ends (evident_ms)
  * noises: ESC-50 (K. Piczak, CC BY-NC 3.0): coughs, laughter, sneezes, horns, engines, babies, dogs, doors...
  * effects (ours): telephone narrow-band + mu-law, speakerphone echo of the agent, car / street / AC hum /
            household beds, TV speech with room reverb, side speech from another voice
Only synthetic Arabic text is sent to the TTS service; no personal data.

    .venv/bin/python -m tools.synth_calls --n 160 --out data/synth --seed 1
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
from ttbench.dataset import write_wav_stereo  # noqa: E402

EXT = Path.home() / "Downloads" / "sarj-ext-data"
TTS_CACHE = EXT / "tts-cache"
ESC = EXT / "ESC-50-master"
SR = 16000

CALLER_VOICES = {
    "ar-SA-HamedNeural": ("sa-najdi", "m"), "ar-SA-ZariyahNeural": ("sa-najdi", "f"),
    "ar-KW-FahedNeural": ("gulf-kuwaiti", "m"), "ar-KW-NouraNeural": ("gulf-kuwaiti", "f"),
    "ar-BH-AliNeural": ("gulf-bahraini", "m"), "ar-BH-LailaNeural": ("gulf-bahraini", "f"),
    "ar-QA-MoazNeural": ("gulf-qatari", "m"), "ar-QA-AmalNeural": ("gulf-qatari", "f"),
    "ar-AE-HamdanNeural": ("gulf-emirati", "m"), "ar-AE-FatimaNeural": ("gulf-emirati", "f"),
    "ar-OM-AbdullahNeural": ("gulf-omani", "m"), "ar-OM-AyshaNeural": ("gulf-omani", "f"),
}
AGENT_VOICES = ["ar-SA-HamedNeural", "ar-SA-ZariyahNeural", "ar-JO-TaimNeural", "ar-JO-SanaNeural"]
TV_VOICES = ["ar-EG-ShakirNeural", "ar-SY-LaithNeural", "ar-LB-RamiNeural", "ar-EG-SalmaNeural"]

AGENTS = ["نوف", "خالد", "منى", "فيصل", "سارة", "عبدالله", "هيفاء", "ماجد"]
COMPANIES = ["شركة الريان للتأمين", "مطعم السفرة", "بنك الخليج", "تطبيق مشوار", "عيادات الندى", "طيران نسمة",
             "مكتبة جرير", "شركة زاد للتوصيل"]
CALLERS = [("الأستاذ", "محمد"), ("الأستاذ", "سلطان"), ("الأستاذة", "نورة"), ("الأستاذة", "عبير"), ("الأستاذ", "تركي"),
           ("الأستاذة", "منيرة"), ("الأخ", "بدر"), ("الأخت", "هيا")]
A0 = ["السلام عليكم، معك {a} من {c}. معي {t} {n}؟", "مساء الخير، معك {a} من {c}. أكلّم {t} {n}؟",
      "هلا، معك {a} من {c}. معي {t} {n}؟"]
A1 = [
    "حياك الله. أتصل عليك بخصوص طلبك الأخير، الطلب جاهز في المستودع وبيطلع مع المندوب بكرة الصبح، والمندوب بيتصل عليك قبل ما يوصل بنص ساعة، وإذا ما كنت موجود تقدر تحدد موعد ثاني من التطبيق.",
    "يا هلا. حبيت أبلغك إن وثيقة التأمين حقتك بتنتهي آخر الشهر، وعندنا عرض تجديد بخصم خمسة عشر بالمية إذا جددت هالأسبوع، والتجديد يتم أونلاين وما يحتاج تزور الفرع أبد.",
    "أهلين. بخصوص الحجز اللي سويته، الطاولة محجوزة لك يوم الجمعة الساعة تسع بالليل لستة أشخاص، ونبي نأكد إذا عندكم أي طلبات خاصة أو حساسية من أكل معين.",
    "حياك الله. نبي نذكرك إن بطاقتك الائتمانية عليها مبلغ مستحق، والحد الأدنى للسداد مية وخمسين ريال، وآخر موعد للسداد يوم الأحد الجاي، وتقدر تسدد من التطبيق أو من أي صراف.",
    "هلا والله. رحلتك إلى أبها تأخرت ساعتين بسبب الأحوال الجوية، والموعد الجديد للإقلاع الساعة أربع العصر، وبطاقة صعود الطائرة بتوصلك برسالة نصية قبل الموعد بثلاث ساعات.",
    "أهلين. نتايج الفحص طلعت والحمدلله كلها سليمة، بس الدكتور حاب يشوفك مرة ثانية بعد شهر عشان يتابع الضغط، وعندنا مواعيد متاحة الصبح والعصر طول الأسبوع.",
    "يا هلا. طلب الكتب حقك وصل الفرع، وتقدر تستلمه من اليوم لين نهاية الأسبوع، ولا تنسى تجيب رقم الطلب والهوية، وإذا تبي توصيل للبيت نقدر نرتب لك بعشرين ريال.",
    "حياك الله. اشتراكك الشهري بيتجدد تلقائياً بعد ثلاث أيام بنفس السعر، وإذا حاب تغير الباقة أو توقف التجديد تقدر تسوي هذا من الإعدادات في التطبيق، أو أسويه لك الحين.",
    "أهلين فيك. بخصوص الشكوى اللي رفعتها، الفريق الفني راجع الموضوع وتبين إن فيه خلل في العداد، وبنرسل فني يركب عداد جديد يوم الثلاثاء من الساعة ثمان إلى الساعة اثنعش الظهر.",
    "هلا. حبيت أوضح لك طريقة الدفع بالتقسيط: المبلغ الإجمالي ألفين وأربع مية ريال، ينقسم على ست دفعات، كل دفعة أربع مية ريال، وأول دفعة تنخصم اليوم، والباقي كل شهر بنفس التاريخ.",
    "يا هلا. عندنا لك عرض خاص على باقة الإنترنت المنزلي، سرعة ألف ميقا بنفس سعر باقتك الحالية لأول ست شهور، والتركيب مجاني، والفني يقدر يجيك خلال يومين من الموافقة.",
    "حياك الله. نبي نأكد عنوان التوصيل: حي الملقا، شارع الأمير محمد بن سعد، بيت رقم سبعة وعشرين، بجنب المسجد، وإذا فيه أي تعديل على العنوان قل لي الحين عشان نحدثه.",
]
A1_QUESTIONS = ["حياك الله. وش أنسب وقت نتصل عليك فيه عشان نأكد الموعد؟", "ممكن تعطيني رقم الطلب لو سمحت؟ هو ستة أرقام، تلقاه في رسالة التأكيد.",
                "طيب، وش اسم الحي اللي تبي التوصيل له؟", "ممكن تعطيني آخر أربع أرقام من البطاقة؟"]
U0 = ["إيه نعم.", "إيه معك، تفضل.", "أيوه أنا، هلا.", "هلا، إيه نعم.", "إي نعم، تفضل.", "هيه، مساء النور.",
      "وعليكم السلام، إيه.", "نعم، تفضل.", "إيه هلا والله."]

# *word* marks the word whose end makes the intention evident
STOP_LINES = [
    "لا لا *لحظة*، أنا ما طلبت هذا.", "*لحظة* لو سمحت.", "*لحظة*.", "*استنى* استنى، وش قلت؟", "*اسمع*، أنا مو فاضي الحين.",
    "أيوه… *بس* أبي أغير العنوان.", "إيه *بس* كم المبلغ بالضبط؟", "*عفواً*، ممكن تعيد آخر شي؟", "*لا*، الموعد ما يناسبني.",
    "*ممكن* سؤال؟", "*وش* يعني رسوم إدارية؟", "*كم* السعر؟", "*متى* بيوصل الطلب؟", "طيب *بس* أنا دفعت الأسبوع الماضي.",
    "*أبي* أكلم مشرف لو سمحت.", "*مو* هذا اللي اتفقنا عليه.", "*دقيقة* دقيقة، خلني أجيب قلم.", "*سوري* بقاطعك، الرقم غلط.",
    "*شوف*، أنا ألغيت الاشتراك من زمان.", "*ليش* انخصم المبلغ مرتين؟", "لا لا لا *غلط*، العنوان تغير.",
    "الطلب *وصلني* أمس الحمدلله.", "أنا بالدوام *الحين*، كلمني بعد العصر.", "العنوان *تغيّر*، صرت ساكن بجدة.",
    "*يا أخي* هذا ثالث مرة تتصلون علي!", "*وين* الفرع بالضبط؟", "إيه زين… *بس* ما أبي توصيل.",
]
TALKOVER = ["يا أخي *هذا* ثالث مرة أسمع نفس الكلام! صار لي أسبوعين والمشكلة ما انحلت!",
            "لا لا *اسمعني* أنت، أنا اتصلت عليكم خمس مرات وكل مرة نفس الرد!",
            "*خلاص* خلاص، ما أبي أي عرض، لا تتصلون علي مرة ثانية لو سمحت!"]
BC_LINES = ["إيه.", "أها.", "مم.", "إيه إيه.", "زين.", "تمام.", "صح.", "أكيد.", "إيه والله.", "إيه والله صادق، صح.",
            "أيوه أيوه.", "هيه.", "هيه… هيه.", "إي إي… زين.", "ماشي.", "أوكي.", "نعم نعم.", "معك معك.", "حلو.", "أها، تمام.",
            "إيه، فاهم عليك.", "مم مم."]
FILLERS = ["اممم…", "ااا…", "ممم…", "امم… امم…"]
SIDE_SPEECH = ["يُمّه! وين الشاحن؟", "يبه تعال شوف!", "خلاص لا تصارخ!", "وين المفتاح؟ يُمّه!", "أبي ماي!", "سكّر الباب!"]
TV_LINES = ["وفي نشرة الأخبار لهذا اليوم، أعلنت الجهات المختصة عن افتتاح عدد من المشاريع الجديدة في مختلف المناطق.",
            "وتشير التوقعات إلى استمرار الأجواء المعتدلة خلال الأيام المقبلة مع فرصة لهطول أمطار خفيفة.",
            "وانتهت المباراة بالتعادل الإيجابي بهدف لكل فريق في لقاء شهد حضوراً جماهيرياً كبيراً."]
EOT_TURNS = [  # chunks and the pause after each: (text, pause_ms, hold) ; last pause ignored
    [("اممم…", 900, "required"), ("بعد المغرب…", 700, "optional"), ("يعني تقريباً الساعة سبع.", 0, None)],
    [("لحظة أشوفه…", 2000, "required"), ("ثلاثة، سبعة،", 1000, "required"), ("تسعة، واحد،", 900, "required"), ("أربعة، خمسة.", 0, None)],
    [("ااا…", 1000, "required"), ("حي الياسمين.", 0, None)],
    [("خلني أشوف…", 1500, "required"), ("أربعة، اثنين،", 900, "required"), ("ستة، ثمانية.", 0, None)],
]
ESC_NV = ["coughing", "laughing", "sneezing", "breathing", "snoring", "drinking_sipping"]
ESC_BG = ["car_horn", "engine", "siren", "dog", "crying_baby", "door_wood_knock", "train", "vacuum_cleaner",
          "church_bells", "footsteps", "keyboard_typing", "washing_machine", "clock_alarm", "glass_breaking", "helicopter"]


# ---------------------------------------------------------------- TTS with cache
def _tts_key(voice, text, rate, pitch):
    return hashlib.sha1(f"{voice}|{text}|{rate}|{pitch}".encode()).hexdigest()


async def _tts_one(sem, voice, text, rate, pitch):
    import edge_tts
    key = _tts_key(voice, text, rate, pitch)
    wav, js = TTS_CACHE / f"{key}.wav", TTS_CACHE / f"{key}.json"
    if wav.exists() and js.exists():
        return
    async with sem:
        for attempt in range(4):
            try:
                c = edge_tts.Communicate(text.replace("*", ""), voice, rate=rate, pitch=pitch, boundary="WordBoundary")
                audio, words = b"", []
                async for ch in c.stream():
                    if ch["type"] == "audio":
                        audio += ch["data"]
                    elif ch["type"] == "WordBoundary":
                        words.append({"w": ch["text"], "s": ch["offset"] / 1e4, "d": ch["duration"] / 1e4})
                x, sr = sf.read(io.BytesIO(audio), dtype="float32")
                if x.ndim > 1:
                    x = x.mean(1)
                x = signal.resample_poly(x, SR, sr).astype(np.float32)
                TTS_CACHE.mkdir(parents=True, exist_ok=True)
                sf.write(wav, x, SR)
                js.write_text(json.dumps({"voice": voice, "text": text, "words": words}, ensure_ascii=False))
                return
            except Exception as e:  # network hiccup: retry
                if attempt == 3:
                    print("TTS failed", voice, text[:30], e)
                await asyncio.sleep(1 + attempt)


def tts_batch(items):
    async def go():
        sem = asyncio.Semaphore(12)
        await asyncio.gather(*[_tts_one(sem, *it) for it in set(items)])
    asyncio.run(go())


OFFLINE = False            # --offline: no network; reuse cached utterances and perturb them locally
PERTURB: dict = {}         # voice -> speed/pitch factor for the call being rendered (consistent within a call)
_INDEX: dict = {}


def _index():
    if not _INDEX:
        for js in TTS_CACHE.glob("*.json"):
            m = json.loads(js.read_text(encoding="utf-8"))
            _INDEX.setdefault((m["voice"], m["text"]), js.stem)
            _INDEX.setdefault(("*", m["text"]), js.stem)
    return _INDEX


def perturb(x, words, f):
    """Speed perturbation (Kaldi-style): factor f>1 = slower and lower-pitched. Word timings scale with it."""
    if abs(f - 1) < 1e-3:
        return x, words
    from fractions import Fraction
    fr = Fraction(f).limit_denominator(50)
    y = signal.resample_poly(x, fr.numerator, fr.denominator).astype(np.float32)
    return y, [{"w": w["w"], "s": w["s"] * f, "d": w["d"] * f} for w in words]


def tts_get(voice, text, rate, pitch):
    key = _tts_key(voice, text, rate, pitch)
    if not (TTS_CACHE / f"{key}.wav").exists() and OFFLINE:
        key = _index().get((voice, text)) or _index().get(("*", text))
        if key is None:
            raise KeyError(f"no cached audio for «{text}»")
    x, _ = sf.read(TTS_CACHE / f"{key}.wav", dtype="float32")
    meta = json.loads((TTS_CACHE / f"{key}.json").read_text(encoding="utf-8"))
    return perturb(x, meta["words"], PERTURB.get(voice, 1.0))


# ---------------------------------------------------------------- audio helpers
def trim(x, thr_db=-45):
    e = 20 * np.log10(np.abs(x) + 1e-9)
    idx = np.where(e > thr_db)[0]
    return x[max(0, idx[0] - 160): idx[-1] + 160] if len(idx) else x


def level_p80(x):
    n = len(x) // 320
    if n == 0:
        return -90.0
    f = x[: n * 320].reshape(n, 320)
    d = 20 * np.log10(np.sqrt((f ** 2).mean(1)) + 1e-9)
    d = d[d > d.max() - 30]
    return float(np.percentile(d, 80))


def set_level(x, target_db):
    return x * 10 ** ((target_db - level_p80(x)) / 20)


def pink(n, rng):
    w = rng.standard_normal(n)
    b, a = [0.049922035, -0.095993537, 0.050612699, -0.004408786], [1, -2.494956002, 2.017265875, -0.522189400]
    return signal.lfilter(b, a, w)


def bed(cond, n, rng):
    if cond == "quiet":
        x = pink(n, rng)
        return set_level(x, -62 + rng.uniform(-3, 3))
    if cond == "ac-hum":
        t = np.arange(n) / SR
        f0 = rng.choice([50, 60])
        h = sum(np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6)) / k for k in range(1, 6))
        return set_level(h + 0.5 * pink(n, rng), -44 + rng.uniform(-4, 4))
    if cond == "car":
        b = signal.lfilter([1], [1, -0.995], rng.standard_normal(n))
        b = signal.sosfilt(signal.butter(4, 300, "low", fs=SR, output="sos"), b)
        return set_level(b, -36 + rng.uniform(-4, 4))
    if cond in ("street", "cafe"):
        return set_level(pink(n, rng), -35 + rng.uniform(-4, 4))
    if cond == "household":
        return set_level(pink(n, rng), -48 + rng.uniform(-4, 4))
    return set_level(pink(n, rng), -60)


def reverb(x, rng, rt=0.4):
    n = int(rt * SR)
    ir = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
    ir[0] = 1.0
    y = signal.fftconvolve(x, ir)[: len(x) + n]
    return y / (np.abs(y).max() + 1e-9) * np.abs(x).max()


def phone(x):
    y = signal.resample_poly(x, 1, 2)
    y = signal.sosfilt(signal.butter(4, [300, 3400], "band", fs=8000, output="sos"), y)
    mu = 255
    y = np.clip(y, -1, 1)
    q = np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)
    q = np.round(q * 127) / 127
    y = np.sign(q) * ((1 + mu) ** np.abs(q) - 1) / mu
    return signal.resample_poly(y, 2, 1)[: len(x)]


_esc_meta = None


def esc_clip(category, rng):
    global _esc_meta
    if _esc_meta is None:
        with open(ESC / "meta" / "esc50.csv") as f:
            _esc_meta = list(csv.DictReader(f))
    rows = [r for r in _esc_meta if r["category"] == category]
    r = rows[rng.integers(len(rows))]
    x, sr = sf.read(ESC / "audio" / r["filename"], dtype="float32")
    x = signal.resample_poly(x, SR, sr)
    return trim(x, -40)


def place(buf, x, at_ms):
    a = int(at_ms * 16)
    b = min(len(buf), a + len(x))
    if b > a:
        buf[a:b] += x[: b - a]


def words_span(words, offset_ms):
    """Speech onset/offset (ms) from TTS word timings."""
    return offset_ms + words[0]["s"], offset_ms + words[-1]["s"] + words[-1]["d"]


def evident_from(words, text, offset_ms):
    """End of the *marked* word (or the second word) = the moment the intention is clear."""
    toks = text.replace("…", " ").split()
    marked = [i for i, t in enumerate(toks) if "*" in t]
    if marked:
        target = toks[marked[0]].strip("*،.؟!,").replace("*", "")
        for w in words:
            if w["w"].strip("*،.؟!,") == target or target.startswith(w["w"]):
                return offset_ms + w["s"] + w["d"]
    w = words[min(1, len(words) - 1)]
    return offset_ms + w["s"] + w["d"]


# ---------------------------------------------------------------- one call
def plan(i, rng):
    kinds = ["GI"] * 26 + ["BC"] * 20 + ["AM"] * 8 + ["OVs"] * 6 + ["OVk"] * 6 + ["NV"] * 10 + ["NVe"] * 4 + \
            ["BGtv"] * 5 + ["BGside"] * 6 + ["BGnoise"] * 8 + ["HTf"] * 6 + ["EOT"] * 5
    kind = kinds[i % len(kinds)] if i < len(kinds) else rng.choice(kinds)
    cv = rng.choice(list(CALLER_VOICES))
    av = rng.choice([v for v in AGENT_VOICES if v != cv])
    p = {"kind": kind, "caller_voice": cv, "agent_voice": av,
         "c_rate": f"{int(rng.integers(-12, 13)):+d}%", "c_pitch": f"{int(rng.integers(-25, 26)):+d}Hz",
         "a0": rng.choice(A0).format(a=rng.choice(AGENTS), c=rng.choice(COMPANIES), t=(cn := CALLERS[rng.integers(len(CALLERS))])[0], n=cn[1]),
         "a1": rng.choice(A1_QUESTIONS if kind == "EOT" else A1), "u0": rng.choice(U0)}
    cond = rng.choice(["quiet", "quiet", "ac-hum", "car", "street", "household", "cafe"])
    p["cond"] = [cond] + (["phone-nb"] if rng.random() < 0.45 else [])
    if kind == "GI":
        p["u1"] = [rng.choice(STOP_LINES)]
    elif kind == "BC":
        p["u1"] = list(rng.choice(BC_LINES, size=int(rng.integers(1, 4)), replace=False))
    elif kind == "AM":
        p["u1"] = [rng.choice(["طيب.", "خلاص خلاص، *فهمت*.", "طيب؟ … *بس* كم المبلغ؟", "طيب طيب."])]
    elif kind == "OVs":
        p["u1"] = [rng.choice(TALKOVER)]
    elif kind == "HTf":
        p["u1"] = [rng.choice(FILLERS)]
    elif kind == "BGside":
        p["side"] = rng.choice(SIDE_SPEECH)
        p["side_voice"] = rng.choice([v for v in CALLER_VOICES if v != cv])
    elif kind == "BGtv":
        p["tv"] = rng.choice(TV_LINES)
        p["tv_voice"] = rng.choice(TV_VOICES)
    elif kind == "EOT":
        p["eot"] = EOT_TURNS[1 if "رقم" in p["a1"] or "أرقام" in p["a1"] else rng.integers(len(EOT_TURNS))]
        if "أربع أرقام" in p["a1"]:
            p["eot"] = EOT_TURNS[3]
        if "الحي" in p["a1"]:
            p["eot"] = EOT_TURNS[2]
    if kind == "NVe":
        p["cond"] = ["speakerphone"]
    return p


def offline_plan(i, rng):
    """Like plan(), but only with sentences already synthesised FOR THAT VOICE (so a caller keeps one voice)."""
    idx = _index()
    by_voice: dict = {}
    for (v, t) in idx:
        if v != "*":
            by_voice.setdefault(v, set()).add(t)
    greet = lambda t: t.startswith(("السلام عليكم، معك", "مساء الخير، معك", "هلا، معك"))  # noqa: E731
    for _ in range(200):
        p = plan(i, rng)
        cv, av = p["caller_voice"], p["agent_voice"]
        T, A = by_voice.get(cv, set()), by_voice.get(av, set())
        a0 = [t for t in A if greet(t)]
        a1 = [t for t in A if t in (A1_QUESTIONS if p["kind"] == "EOT" else A1)]
        u0 = [t for t in T if t in U0]
        if not (a0 and a1 and u0):
            continue
        p["a0"], p["a1"], p["u0"] = rng.choice(a0), rng.choice(a1), rng.choice(u0)
        bank = {"GI": STOP_LINES, "BC": BC_LINES, "AM": ["طيب.", "خلاص خلاص، *فهمت*.", "طيب؟ … *بس* كم المبلغ؟", "طيب طيب."],
                "OVs": TALKOVER, "HTf": FILLERS}.get(p["kind"])
        if bank is not None:
            have = [t for t in bank if t in T]
            if not have:
                continue
            p["u1"] = list(rng.choice(have, size=min(len(have), len(p.get("u1", [1]))), replace=False))
        if p["kind"] == "OVk":
            p["kind"], p["u1"] = "BC", [rng.choice([t for t in BC_LINES if t in T] or ["إيه."])]
            if p["u1"][0] not in T:
                continue
        if p["kind"] == "BGside":
            pairs = [(v, t) for (v, t) in idx if v != "*" and t in SIDE_SPEECH and v != cv]
            if not pairs:
                continue
            p["side_voice"], p["side"] = pairs[rng.integers(len(pairs))]
        if p["kind"] == "BGtv":
            pairs = [(v, t) for (v, t) in idx if v != "*" and t in TV_LINES]
            if not pairs:
                continue
            p["tv_voice"], p["tv"] = pairs[rng.integers(len(pairs))]
        if p["kind"] == "EOT" and not all(c in T for c, _, _ in p["eot"]):
            continue
        return p
    raise RuntimeError("could not build an offline plan")


def tts_items(p):
    cv, av = p["caller_voice"], p["agent_voice"]
    r, pt = p["c_rate"], p["c_pitch"]
    items = [(av, p["a0"], "+0%", "+0Hz"), (av, p["a1"], "+0%", "+0Hz"), (cv, p["u0"], r, pt)]
    for line in p.get("u1", []):
        items.append((cv, line, r, pt))
    if "side" in p:
        items.append((p["side_voice"], p["side"], "+15%", "+40Hz"))
    if "tv" in p:
        items.append((p["tv_voice"], p["tv"], "+0%", "+0Hz"))
    for chunk, _, _ in p.get("eot", []):
        items.append((cv, chunk, r, pt))
    return items


CAT = {"GI": "GI", "BC": "BC", "AM": "AM", "OVs": "OV", "OVk": "OV", "NV": "NV", "NVe": "NV", "BGtv": "BG",
       "BGside": "BG", "BGnoise": "BG", "HTf": "HT", "EOT": "HT"}


def render(i, p, rng, prefix):
    sid = f"{CAT[p['kind']]}-{prefix}{i:03d}"
    cv, av = p["caller_voice"], p["agent_voice"]
    PERTURB.clear()
    if OFFLINE:                                   # a «new» caller and agent voice for this call
        PERTURB[cv] = float(rng.uniform(0.88, 1.12))
        PERTURB[av] = float(rng.uniform(0.94, 1.06))
    a0, a0w = tts_get(av, p["a0"], "+0%", "+0Hz")
    a1, a1w = tts_get(av, p["a1"], "+0%", "+0Hz")
    u0, u0w = tts_get(cv, p["u0"], p["c_rate"], p["c_pitch"])
    a0, a1 = set_level(trim(a0), -20), set_level(trim(a1), -20)
    caller_ref = -20 + rng.uniform(-4, 3)
    u0 = set_level(trim(u0), caller_ref)
    t_a0 = 300
    a0_end = t_a0 + len(a0) / 16
    t_u0 = a0_end + rng.uniform(250, 500)
    u0_end = t_u0 + len(u0) / 16
    t_a1 = u0_end + rng.uniform(400, 1300)
    a1_end = t_a1 + len(a1) / 16
    dur = a1_end + 2500
    if p["kind"] == "EOT":
        dur = a1_end + 400 + sum(len(tts_get(cv, c, p["c_rate"], p["c_pitch"])[0]) / 16 + pz for c, pz, _ in p["eot"]) + 3500
    n = int(dur * 16)
    agent = np.zeros(n, np.float32)
    caller = np.zeros(n, np.float32)
    place(agent, a0, t_a0)
    place(agent, a1, t_a1)
    place(caller, u0, t_u0)
    segs = [{"seg_id": "U0", "speaker_id": "C", "start_ms": round(t_u0), "end_ms": round(u0_end), "kind": "speech", "text": p["u0"]}]
    events = []
    a1_dur = a1_end - t_a1

    def ev(kind, expected, onset, offset, evident=None, preferred=None, rationale=""):
        e = {"event_id": f"{sid}.e{len(events) + 1}", "kind": kind, "expected": expected, "onset_ms": round(onset),
             "offset_ms": round(offset), "rationale": rationale}
        if expected in ("KEEP", "EITHER"):
            e["window_end_ms"] = round(offset + 500)
        if evident is not None:
            e["evident_ms"] = round(evident)
            if expected == "STOP":
                e["deadline_ms"] = round(evident + 1500)
        if preferred:
            e["preferred"] = preferred
        events.append(e)

    k = p["kind"]
    if k in ("GI", "AM", "OVs", "HTf", "BC"):
        lines = p["u1"]
        slots = sorted(rng.uniform(0.2, 0.75, size=len(lines)))
        last_end = 0
        for j, (line, frac) in enumerate(zip(lines, slots)):
            x, w = tts_get(cv, line, p["c_rate"], p["c_pitch"])
            off = max(t_a1 + frac * a1_dur, last_end + 1500)
            lvl = caller_ref + (rng.uniform(-9, -2) if k in ("BC", "HTf") else rng.uniform(-3, 2) if k != "OVs" else rng.uniform(2, 6))
            x = set_level(x, lvl)
            st = off + 0
            place(caller, x, st)
            on, of = words_span(w, st)
            last_end = of
            segs.append({"seg_id": f"U1{chr(97 + j)}", "speaker_id": "C", "start_ms": round(on), "end_ms": round(of),
                         "kind": "backchannel" if k == "BC" else "filler" if k == "HTf" else "speech", "text": line.replace("*", "")})
            if k == "GI" or k == "OVs":
                evd = evident_from(w, line, st) if k == "GI" else on + 400
                ev("interruption", "STOP", on, of, evd, rationale="floor bid")
            elif k == "AM":
                if line.startswith("طيب.") or line == "طيب طيب.":
                    ev("interruption", "EITHER", on, of, preferred="KEEP", rationale="«طيب» alone")
                elif "خلاص" in line:
                    ev("interruption", "EITHER", on, of, evident_from(w, line, st), preferred="STOP", rationale="«خلاص خلاص»")
                else:
                    ev("interruption", "STOP", on, of, evident_from(w, line, st), rationale="rising «طيب» then a question")
            else:
                ev("interruption", "KEEP", on, of, rationale="acknowledgement" if k == "BC" else "thinking aloud")
    elif k == "OVk":
        # say a word from the agent's sentence along with it, then «أيوه»
        cands = [w for w in a1w[4:-4] if len(w["w"].strip("،.")) >= 5]
        w0 = cands[rng.integers(len(cands))] if cands else a1w[len(a1w) // 2]
        line = f"{w0['w'].strip('،.')}، أيوه."
        tts_batch([(cv, line, p["c_rate"], p["c_pitch"])])
        x, w = tts_get(cv, line, p["c_rate"], p["c_pitch"])
        x = set_level(x, caller_ref + rng.uniform(-8, -3))
        st = t_a1 + w0["s"] - w[0]["s"] + rng.uniform(-80, 80)
        place(caller, x, st)
        on, of = words_span(w, st)
        segs.append({"seg_id": "U1", "speaker_id": "C", "start_ms": round(on), "end_ms": round(of), "kind": "backchannel", "text": line})
        ev("interruption", "KEEP", on, of, rationale="saying the agent's word along with it")
    elif k in ("NV", "BGnoise"):
        cat = rng.choice(ESC_NV if k == "NV" else ESC_BG)
        x = esc_clip(cat, rng)[: int(2.5 * SR)]
        x = set_level(x, caller_ref + (rng.uniform(-6, 1) if k == "NV" else rng.uniform(-10, 3)))
        st = t_a1 + rng.uniform(0.25, 0.7) * a1_dur
        place(caller, x, st)
        segs.append({"seg_id": "X1", "speaker_id": "C", "start_ms": round(st), "end_ms": round(st + len(x) / 16), "kind": "nonverbal", "text": "", "delivery": cat})
        ev("interruption", "KEEP", st, st + len(x) / 16, rationale=f"{cat}; nothing said")
    elif k == "NVe":
        pass  # echo added below; the whole A1 is a KEEP window
    elif k == "BGside":
        x, w = tts_get(p["side_voice"], p["side"], "+15%", "+40Hz")
        x = reverb(set_level(x, caller_ref + rng.uniform(-16, -9)), rng, 0.5)
        st = t_a1 + rng.uniform(0.2, 0.6) * a1_dur
        place(caller, x, st)
        segs.append({"seg_id": "X1", "speaker_id": "S2", "start_ms": round(st), "end_ms": round(st + len(x) / 16), "kind": "side_speech", "text": p["side"]})
        ev("interruption", "KEEP", st, st + len(x) / 16, rationale="someone else in the room")
    elif k == "BGtv":
        x, w = tts_get(p["tv_voice"], p["tv"], "+0%", "+0Hz")
        x = reverb(set_level(x, caller_ref + rng.uniform(-20, -12)), rng, 0.6)
        x = np.tile(x, 3)[: int((a1_end - t_a1) * 16)]
        st = t_a1 + 300
        place(caller, x, st)
        ev("interruption", "KEEP", st, st + len(x) / 16, rationale="TV in the background")
    elif k == "EOT":
        t = a1_end + rng.uniform(300, 700)
        onset = None
        pauses = []
        for j, (chunk, pz, hold) in enumerate(p["eot"]):
            x, w = tts_get(cv, chunk, p["c_rate"], p["c_pitch"])
            x = set_level(trim(x), caller_ref)
            place(caller, x, t)
            on, of = words_span(w, t)
            onset = onset if onset is not None else on
            t_end = t + len(x) / 16
            if j < len(p["eot"]) - 1:
                pauses.append({"start_ms": round(of), "end_ms": round(t_end + pz), "hold": hold})
                t = t_end + pz
            else:
                turn_end = of
        segs.append({"seg_id": "U1", "speaker_id": "C", "start_ms": round(onset), "end_ms": round(turn_end), "kind": "speech",
                     "text": " … ".join(c for c, _, _ in p["eot"])})
        events.append({"event_id": f"{sid}.e1", "kind": "end_of_turn", "expected": "TAKE_TURN", "onset_ms": round(onset),
                       "turn_end_ms": round(turn_end), "pauses": pauses, "rationale": "answer with pauses"})

    conds = list(p["cond"])
    if k == "NVe" or "speakerphone" in conds:
        lag = rng.uniform(120, 320)
        e = np.concatenate([np.zeros(int(lag * 16), np.float32), agent])[:n]
        e = signal.sosfilt(signal.butter(2, [250, 3500], "band", fs=SR, output="sos"), e)
        e = reverb(e, rng, 0.25) [:n] * 10 ** (rng.uniform(-26, -16) / 20)
        caller[: len(e)] += e[:n]
        ev("interruption", "KEEP", t_a1 + 300, a1_end, rationale="speakerphone: the agent's own voice comes back")
    base = conds[0] if conds[0] != "speakerphone" else "quiet"
    caller += bed(base, n, rng).astype(np.float32)
    if base in ("street", "cafe") and rng.random() < 0.6:
        place(caller, set_level(esc_clip(rng.choice(["car_horn", "engine", "footsteps"]), rng), caller_ref - 14), rng.uniform(1000, dur - 3000))
    if "phone-nb" in conds:
        caller = phone(caller).astype(np.float32)
    stereo = np.stack([np.clip(agent, -1, 1), np.clip(caller, -1, 1)], 1)
    stereo = (stereo * 32767).astype(np.int16)
    dialect, gender = CALLER_VOICES[cv]
    label = {
        "schema_version": "1.0", "scenario_id": sid, "split": "synth", "category": CAT[k], "subtype": k,
        "difficulty": 2, "dialect": dialect, "elicitation": "scripted",
        "speakers": [{"speaker_id": "C", "role": "caller", "source": f"edge-tts:{cv} {p['c_rate']} {p['c_pitch']}", "gender": gender, "dialect": dialect}],
        "agent_voice": av, "acoustic": {"conditions": conds, "noise_source_ids": ["ESC-50", "procedural"]},
        "audio": {"file": f"{sid}.wav", "sample_rate": SR, "channels": ["agent", "user"], "duration_ms": int(len(stereo) / 16)},
        "context": {"use_case": "synthetic", "language": "ar"},
        "agent_utterances": [{"utt_id": "A0", "start_ms": t_a0, "end_ms": round(a0_end), "text": p["a0"], "is_question": True},
                             {"utt_id": "A1", "start_ms": round(t_a1), "end_ms": round(a1_end), "text": p["a1"], "is_question": p["a1"].endswith("؟")}],
        "user_segments": segs, "events": events,
        "provenance": {"production_mode": "synthetic", "label_status": "generated", "generator": "tools/synth_calls.py"},
    }
    return sid, stereo, label


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--out", default=str(KIT / "data" / "synth"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--prefix", default="S")
    ap.add_argument("--offline", action="store_true", help="no network: reuse cached voices, perturb them")
    ap.add_argument("--voices", default=None, help="comma list of locale tags to keep, e.g. SA,KW,BH,AE")
    ap.add_argument("--exclude-voices", default=None, help="comma list of locale tags to drop, e.g. QA,OM")
    a = ap.parse_args(argv)
    global OFFLINE
    OFFLINE = a.offline
    keep = lambda v: (not a.voices or any(f"-{t}-" in v for t in a.voices.split(","))) and \
        (not a.exclude_voices or not any(f"-{t}-" in v for t in a.exclude_voices.split(",")))  # noqa: E731
    for v in [v for v in CALLER_VOICES if not keep(v)]:
        CALLER_VOICES.pop(v)
    rng = np.random.default_rng(a.seed)
    if OFFLINE:
        plans = [offline_plan(i, rng) for i in range(a.n)]
    else:
        plans = [plan(i, rng) for i in range(a.n)]
    if not OFFLINE:
        items = [it for p in plans for it in tts_items(p)]
        print(f"TTS: {len(set(items))} unique utterances")
        tts_batch(items)
    out = Path(a.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    scen = []
    for i, p in enumerate(plans):
        try:
            sid, stereo, label = render(i, p, rng, a.prefix)
        except Exception as e:
            print("skip", i, p["kind"], repr(e))
            continue
        write_wav_stereo(out / "audio" / f"{sid}.wav", stereo)
        (out / "labels" / f"{sid}.json").write_text(json.dumps(label, ensure_ascii=False, indent=1), encoding="utf-8")
        scen.append({"scenario_id": sid})
    manifest = {"dataset": "sarj-turn-taking-ar-synth", "split": out.name, "label_status": "generated",
                "audio_source": "edge-tts Gulf voices + ESC-50 + procedural effects",
                "thresholds": {"T_full_ms": 300, "T_dead_ms": 1500, "E_full_ms": 600, "E_dead_ms": 2000, "E_miss_ms": 3000},
                "scenarios": scen}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(scen)} calls to {out}")


if __name__ == "__main__":
    main()
