"""Claude as the TEACHER: writes thousands of labelled caller moments in Saudi / Gulf / other Arabic, so a small
fast student model (tools/train_student.py) can learn Claude's judgement and decide in ~10 ms.

Each call to Claude asks for 40 items for one (dialect, business domain) pair, mixed on purpose so the hard
contrasts sit side by side («إيه والله صادق» = KEEP vs «إيه بس أبي أسأل» = STOP). Items look like what our speech
recognition really writes: no diacritics, some spelling noise, sometimes a misheard word.

    .venv/bin/python -m tools.teacher_gen --calls 120 --workers 4          # -> data/teacher/raw/*.json
Runs through the Claude Code CLI on the user's subscription (no API key needed). Resumable: finished calls are kept.
Other teachers: --backend anthropic (ANTHROPIC_API_KEY) or --backend gemini (GEMINI_API_KEY; --model gemini-...).
v3 focuses: --focus pairs (minimal pairs: same start, KEEP vs STOP) and --focus live (the live-demo failure kinds).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
OUT = KIT / "data" / "teacher" / "raw"

DIALECTS = [
    "نجدي (الرياض والقصيم)", "حجازي (جدة ومكة)", "شرقاوي (الأحساء والدمام)", "جنوبي سعودي (أبها وجازان ونجران)",
    "شمالي سعودي (حائل والجوف)", "سعودي أبيض (لهجة المكالمات العامة)", "كويتي", "إماراتي", "قطري وبحريني", "عماني",
    "مقيم مصري يتكلم مع شركة سعودية", "مقيم شامي (سوري/لبناني/أردني)", "مقيم يمني أو سوداني",
    "غير ناطق بالعربية (عربي مكسّر، مثل عامل باكستاني أو فلبيني)",
]
DOMAINS = [
    "تحصيل دفعات وأقساط متأخرة", "مواعيد عيادة ومستشفى", "توصيل طلبات وشحنات", "مبيعات باقات اتصالات وإنترنت",
    "بنك (بطاقة، حوالة، تحديث بيانات)", "تأمين سيارات وطبي", "خدمات حكومية (موعد، معاملة، مخالفة)",
    "صيانة سيارات ووكالة", "إيجار وعقار", "حجز طيران وفنادق", "مطاعم وطلبات أكل", "كهرباء ومياه وفواتير",
]

SYSTEM = """You write training data for an Arabic voice agent's turn-taking model. Output ONLY a JSON array, no prose."""

PROMPT = """اكتب ٤٠ عنصر تدريب لمكالمة هاتفية من شركة لعميل.
المجال: {domain}. لهجة العميل: {dialect}.

كل عنصر = لحظة: الوكيل (موظف الشركة) يتكلم بجملة، وخط العميل التقط كلام. المطلوب نعرف: هل يسكت الوكيل (STOP) ولا يكمّل (KEEP).
التوزيع بالضبط:
- ١٢ STOP: العميل يبي الكلمة. اعتراض، سؤال («كم؟»، «متى؟»، «مين معي؟»)، تصحيح معلومة، «لحظة/بس/اسمع/لا لا/خلاص»، طلب يعيد («ها؟»، «نعم؟»، «ما سمعت»)، مشكلة خط («ألو ألو؟»)، معلومة جديدة، زعل، استئذان مؤدب («معليش أقاطعك»). نوّع: قصيرة جداً (كلمة)، وطويلة، وتبدأ بتأييد ثم طلب («إيه بس…»، «تمام، طيب سؤال»).
- ١٠ KEEP تأييد: «إيه، أها، تمام، طيب، صح، معك، زين، أكيد، ماشي، أوكي، ايوه، هممم، الله يعطيك العافية، صادق، والله، فاهم عليك، يا هلا، الله يسلمك، ضحكة هههه»، وبعضها طويلة أو مكررة («إيه إيه صح صح والله») بدون طلب.
- ٤ KEEP حشو: «اممم، يعني، ااه، خلني أشوف» (يفكر، ما يبي يقاطع).
- ٦ KEEP صوت شخص ثاني واضح من الكلمات إنه مو موجّه للوكيل: نشرة أخبار أو إعلان بالتلفزيون (فصحى)، تعليق مباراة، أغنية، طفل يكلم أمه/أبوه، شخص بالبيت ينادي أحد باسمه، قراءة قرآن أو أذان.
- ٤ KEEP صدى: كلمات من جملة الوكيل نفسها ترجع مشوّهة وناقصة (مثل ما يلتقطها مايك العميل).
- ٤ KEEP تفريغ غلط: كلام فاضي يكتبه نظام التفريغ على ضجيج («ترجمة نانسي قنقر»، «اشتركوا في القناة»، حروف مقطعة، كلمة إنجليزية عشوائية).

قواعد:
- جملة الوكيل («agent»): لهجة سعودية بيضاء مهذبة من نفس المجال، ١٠–٣٠ كلمة، كأنها الجملة اللي كان يقولها لحظة التداخل. نوّعها.
- كلام العميل («caller»): مثل ما يكتبه نظام تفريغ: بدون تشكيل، و٣٠٪ منها فيها أخطاء إملاء أو كلمة مسموعة غلط أو كلمتين ملصوقة. بلهجة العميل المحددة فعلاً (مفرداتها الحقيقية).
- لكل STOP حط «cue»: رقم الكلمة (من ١) اللي عندها السامع يعرف إن العميل يبي الكلمة.
- لا تكرر نفس الصياغة، ولا تنسخ الأمثلة اللي فوق حرفياً. خلّها واقعية ومتنوعة وصعبة.

الصيغة (JSON array فقط):
[{{"agent": "...", "caller": "...", "label": "STOP", "kind": "bid", "cue": 2}}, {{"agent": "...", "caller": "...", "label": "KEEP", "kind": "backchannel"}}, ...]
kind واحد من: bid, backchannel, filler, other_speaker, echo, asr_noise"""


PROMPT_ANSWERS = """اكتب ٤٠ عنصر تدريب لمكالمة هاتفية من شركة لعميل. المجال: {domain}. لهجة العميل: {dialect}.
التركيز: الوكيل سأل سؤال أو عرض خيارات، وبعدها كمّل كلامه، والعميل رد فوقه.
«agent» = اللي سمعه العميل من الوكيل لحظة التداخل: جملة السؤال السابقة + بداية الجملة الحالية (١٥–٣٠ كلمة).
التوزيع:
- ١٤ STOP: العميل يجاوب السؤال أو يختار خيار أو يعطي المعلومة المطلوبة والوكيل لسا يتكلم («إي الأحد زين احجزه»، «لا الثلاثاء أحسن»، «الساعة أربع تناسبني»، «رقمي خمسة خمسة…»، «اسمي محمد العتيبي»، «إيه احجزه»).
- ٨ STOP: الاثنين بدوا مع بعض: الوكيل بدأ جملة جديدة والعميل بدأ يتكلم بطلب أو سؤال أو معلومة.
- ٦ STOP: سؤال توضيح قصير وسط عرض الخيارات («الثلاثاء الساعة كم؟»، «كم قلت؟»، «وين الفرع؟»).
- ٦ KEEP: تأييد صافي («تمام»، «أها»، «طيب»، «إيه إيه») والوكيل ما سأل شي، يشرح بس.
- ٦ KEEP: حشو أو تفكير بصوت عالي بدون جواب («اممم خلني أشوف»، «ااه»، «يعني…») أو كلام لشخص ثاني موجود معه («لحظة يا ولد»).
كلام العميل مثل ما يكتبه نظام التفريغ: بدون تشكيل، و٣٠٪ فيها أخطاء إملاء. بلهجته الحقيقية. لا تكرر الصياغات.
لكل STOP حط «cue» (رقم الكلمة اللي عندها السامع يعرف).
JSON array فقط: [{{"agent": "...", "caller": "...", "label": "STOP", "kind": "answer", "cue": 2}}, ...]
kind واحد من: answer, overlap_start, clarify, backchannel, filler"""


PROMPT_HARD = """اكتب ٤٠ عنصر تدريب لمكالمة هاتفية من شركة لعميل. المجال: {domain}. لهجة العميل: {dialect}.
التركيز: الحالات الصعبة اللي الكلمات لحالها ما تكفي فيها، والمعنى هو اللي يفرّق.
«agent» = جملة الوكيل لحظة التداخل (١٠–٣٠ كلمة، لهجة سعودية بيضاء مهذبة).
التوزيع بالضبط:
- ١٢ STOP افتتاحية مقاطعة: العميل يبدأ بعبارة تطلب الكلمة بمفردات لهجته الحقيقية (اعتذار قبل المقاطعة، لفت انتباه، تعجب يتبعه طلب)، وأحياناً تنقطع الجملة بعد الافتتاحية مباشرة.
- ٦ STOP سؤال قصير عن كلام الوكيل نفسه: يذكر شي قاله الوكيل ويستفسر عنه.
- ٨ KEEP تفكير بصوت عالي: العميل يكلم نفسه وهو يتذكر تاريخ أو رقم أو مكان، بتردد وكلام ناقص، وممكن فيه كلمة سؤال موجهة لنفسه، وما يطلب شي من الوكيل.
- ٨ KEEP كلام لشخص بالغرفة: يكلم طفله أو زوجته أو زميله عن شي ما له علاقة بالمكالمة، وفيه أمر أو سؤال أو نداء باسم، وواضح إنه مو موجّه للوكيل.
- ٦ KEEP تأييد يشبه الاعتراض: مجاملة أو موافقة فيها كلمة نفي أو نداء أو دعاء، ومعناها «كمّل».
قواعد:
- كلام العميل مثل ما يكتبه نظام التفريغ: بدون تشكيل، و٣٠٪ فيها أخطاء إملاء أو كلمة مسموعة غلط. نوّع الطول من كلمتين إلى ١٢ كلمة.
- لا تكرر الصياغات، وخلّ كل عنصر مختلف عن غيره.
- لكل STOP حط «cue»: رقم الكلمة (من ١) اللي عندها السامع يعرف إن العميل يبي الكلمة.
JSON array فقط: [{{"agent": "...", "caller": "...", "label": "STOP", "kind": "opener", "cue": 2}}, ...]
kind واحد من: opener, question, thinking, room_talk, backchannel"""


PROMPT_PAIRS = """اكتب ٤٠ عنصر تدريب لمكالمة هاتفية من شركة لعميل. المجال: {domain}. لهجة العميل: {dialect}.
التركيز: أزواج متقابلة. كل زوج = عنصرين بنفس جملة الوكيل، وكلام العميل يبدأ بنفس الكلمة أو الكلمتين، بس واحد معناه «كمّل» (KEEP) والثاني «أبي الكلمة» (STOP).
اكتب ٢٠ زوج، ونوّع البدايات: تأييد («إيه»، «طيب»، «تمام»، «زين»، «أوكي»)، نفي («لا»)، نداء («يا أخي»)، دعاء («الله يعافيك»)، تفكير («امم»)، كلمة سؤال.
أمثلة على الفكرة بس (لا تنسخها): «طيب» ↔ «طيب وإذا تأخرت؟» · «لا عادي كمّل» ↔ «لا مو أنا صاحب الطلب».
«agent» = جملة الوكيل لحظة التداخل (١٠–٣٠ كلمة، لهجة سعودية بيضاء مهذبة). كلام العميل مثل ما يكتبه نظام التفريغ: بدون تشكيل، و٣٠٪ فيها أخطاء إملاء.
لكل STOP حط «cue»: رقم الكلمة (من ١) اللي عندها السامع يعرف إن العميل يبي الكلمة. الزوجين ورا بعض في المصفوفة.
JSON array فقط: [{{"agent": "...", "caller": "...", "label": "KEEP", "kind": "pair_keep"}}, {{"agent": "...", "caller": "...", "label": "STOP", "kind": "pair_stop", "cue": 2}}, ...]"""


PROMPT_LIVE = """اكتب ٤٠ عنصر تدريب لمكالمة هاتفية من شركة لعميل. المجال: {domain}. لهجة العميل: {dialect}.
التركيز: الأخطاء اللي تصير في المكالمات الحقيقية على مايك لابتوب.
التوزيع بالضبط:
- ٨ KEEP كلمة وحدة أو كلمتين تأييد أو شكر أو «نعم»، بالشكل اللي يكتبه نظام التفريغ (أحياناً ناقصة حرف أو ملصوقة).
- ٦ KEEP كلام الوكيل نفسه راجع من سماعة اللابتوب للمايك: مقطع من جملة الوكيل («agent») بكلماته، ناقص أو مشوّه شوي.
- ٦ KEEP تفريغ غلط على ضجيج: جملة فصحى طويلة ما لها علاقة، أو اسم متحدث مع نقطتين، أو تكرار حرف أو كلمة، أو عبارة ترجمة فيديو.
- ١٠ STOP مقاطعة حقيقية تبدأ بكلمة تأييد ثم طلب أو اعتراض أو سؤال.
- ١٠ STOP مقاطعة قصيرة جداً (كلمة إلى ثلاث) واضحة إنها طلب للكلمة.
«agent» = جملة الوكيل لحظة التداخل (١٠–٣٠ كلمة، لهجة سعودية بيضاء مهذبة). لا تكرر الصياغات.
لكل STOP حط «cue»: رقم الكلمة (من ١) اللي عندها السامع يعرف إن العميل يبي الكلمة.
JSON array فقط: [{{"agent": "...", "caller": "...", "label": "KEEP", "kind": "single_ack"}}, ...]
kind واحد من: single_ack, echo, asr_noise, ack_then_bid, short_bid"""


GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


def ask(backend: str, model: str, prompt: str) -> str:
    """One teacher answer as text. claude = the Claude Code CLI; anthropic / gemini = their HTTP APIs."""
    if backend == "claude":
        cmd = ["claude", "-p", "--model", model, "--effort", "low", "--tools", "", "--system-prompt", SYSTEM,
               "--no-session-persistence", "--strict-mcp-config", "--setting-sources", "", prompt]
        return subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=600).stdout
    if backend == "anthropic":
        import anthropic
        r = anthropic.Anthropic().messages.create(model=model, max_tokens=16000, system=SYSTEM,
                                                  messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in r.content if b.type == "text")
    import urllib.request
    body = {"model": model, "temperature": 0.9, "max_tokens": 16000,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
    req = urllib.request.Request(GEMINI_URL, json.dumps(body).encode(), {
        "Content-Type": "application/json", "Authorization": f"Bearer {os.environ['GEMINI_API_KEY']}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def one(job: tuple[int, str, str], model: str, prompt: str = PROMPT, backend: str = "claude") -> str:
    i, dialect, domain = job
    out = OUT / f"{i:04d}.json"
    if out.exists():
        return "skip"
    for attempt in range(3):
        t0 = time.time()
        try:
            text = ask(backend, model, prompt.format(dialect=dialect, domain=domain))
        except Exception as e:                                # network / quota: try again, then give up on this call
            print(f"{i}: {type(e).__name__} {e}", flush=True)
            time.sleep(5 * (attempt + 1))
            continue
        m = re.search(r"\[.*\]", text, re.S)
        try:
            items = json.loads(m.group(0)) if m else []
        except ValueError:
            items = []
        items = [x for x in items if isinstance(x, dict) and x.get("caller") and x.get("label") in ("STOP", "KEEP")]
        if len(items) >= 30:
            for x in items:
                x.update(dialect=dialect, domain=domain)
            out.write_text(json.dumps(items, ensure_ascii=False))
            return f"{i}: {len(items)} items in {time.time() - t0:.0f}s ({dialect[:12]} / {domain[:14]})"
    return f"{i}: FAILED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=120)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--backend", default="claude", choices=["claude", "anthropic", "gemini"])
    ap.add_argument("--focus", default="", choices=["", "answers", "hard", "pairs", "live"])
    ap.add_argument("--offset", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = list(itertools.product(DIALECTS, DOMAINS))
    random.Random(7).shuffle(pairs)
    if a.focus:
        random.Random(13).shuffle(pairs)
    jobs = [(a.offset + i, d, m) for i, (d, m) in enumerate(itertools.islice(itertools.cycle(pairs), a.calls))]
    prompt = {"answers": PROMPT_ANSWERS, "hard": PROMPT_HARD, "pairs": PROMPT_PAIRS, "live": PROMPT_LIVE}.get(a.focus, PROMPT)
    with ThreadPoolExecutor(a.workers) as ex:
        for msg in ex.map(lambda j: one(j, a.model, prompt, a.backend), jobs):
            print(msg, flush=True)
    n = sum(len(json.loads(p.read_text())) for p in OUT.glob("*.json"))
    print(f"ALLDONE {n} items", flush=True)


if __name__ == "__main__":
    sys.exit(main())
