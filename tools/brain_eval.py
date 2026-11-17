"""Which brain should the talking agent use? Same 12 call moments for every model, measured the same way.

Each case is a real moment of a demo call (convo/scenarios.py), built exactly the way convo/engine.py builds it:
the system prompt, the agent's opening, the caller's answer, and, for interruptions, the half sentence the agent
managed to say plus the same «ملاحظة لك: العميل قاطعك…» note. We then measure:

  * speed  : time to the first full sentence (what the caller waits for) and to the whole reply
  * facts  : did it give the right fact (amount, time, name…) and NOT invent what it doesn't know
  * dialect: Saudi words vs formal MSA vs other dialects (Egyptian / Levantine / Moroccan)

    .venv/bin/python -m tools.brain_eval --brains qwen3-14b,gemma-12b,claude-opus,claude-haiku
Writes analysis/brain_eval.json (all replies, so people can read and judge them too).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
from convo import brain  # noqa: E402
from convo.scenarios import SCENARIOS, as_male, system_prompt  # noqa: E402

OUT = KIT / "analysis" / "brain_eval.json"
DONT_KNOW = r"أتأكد|اتأكد|بتأكد|نتأكد|أتحقق|اتحقق|ما عندي|ماعندي|ما أقدر أأكد|برجع لك|ارجع لك|أرجع لك|بشيّك|بشيك|ما أعرف|ما اعرف|مو متأكد"
# case: scenario, where the agent was cut (None = it finished its point), what the caller said, what a good answer has
CASES = [
    ("payment", "حبيت أذكرك إن عندك دفعة", "كم القيمة؟", r"ثلاث ?مي[هة] ?و ?خمسين|٣٥٠|350", "interrupt: amount"),
    ("payment", None, "أنا دفعت أمس", r"يوم عمل|يظهر", "fact: paid already"),
    ("payment", None, "فيه خصم لو سددت اليوم؟", DONT_KNOW, "unknown: must not invent a discount"),
    ("clinic", "أتصل عليك أأكد موعدك يوم الأحد", "لا لا لحظة، الأحد ما يناسبني", r"الاثنين|الإثنين|الثلاثاء|الثلوث", "interrupt: offer the alternatives"),
    ("clinic", None, "الكشف عليه تأمين؟", r"مغطى|يغطي|يشمل|التأمين", "fact: insurance"),
    ("clinic", None, "وش اسم الدكتورة؟", r"نورة|نوره", "fact: doctor's name"),
    ("delivery", "طلبك طلع من المستودع اليوم، والمندوب بيوصلك", "بس أنا مو بالبيت بكرة", r"تغي[ّ]?ر|نغي[ّ]?ر|موعد ثاني|مجان|مرة وحدة|وقت ثاني|عنوان ثاني", "interrupt: offer a change"),
    ("delivery", None, "كم المبلغ؟", r"مي[هة] ?و ?عشرين|١٢٠|120", "fact: amount"),
    ("delivery", None, "تقدر تجيبه اليوم؟", DONT_KNOW + r"|بكرة|بكره", "unknown: must not promise today"),
    ("telecom", "عندنا لك عرض على باقة", "مو مهتم الحين الله يعافيك", r".", "interrupt: respect the no"),
    ("telecom", None, "فيه عقد؟", r"بدون عقد|ما فيه عقد|مافيه عقد|لا يوجد عقد|ما في عقد|بلا عقد", "fact: no contract"),
    ("telecom", None, "كم أدفع الحين على باقتي؟", r"مي[هة] ?و ?تسعين|١٩٠|190", "fact: current price"),
]
SAUDI = "وش ابغى أبغى ابي أبي تبي تبغى تبغين الحين زين أبشر ابشر حياك عشان مو اللي كذا ودك وايد يعطيك طال عمرك تامر هلا".split()
MSA = "سوف لديك لديكم هل ترغب يرجى نود حيث بإمكانك يمكنك يسعدني أود سيتم".split()
OTHER = "ديال دابا عايز عاوز إزاي ازاي كده دلوقتي ماتعرفش شو هلق بدك منيح كتير هيك".split()


def build(case):
    scn, cut, said, _, _ = case
    sc = as_male(SCENARIOS[scn])
    h = [{"role": "system", "content": system_prompt(sc)},
         {"role": "assistant", "content": " ".join(sc["opening"])},
         {"role": "user", "content": "إيه نعم، هلا"}]
    if cut:
        rest = sc["first_point"].split(cut, 1)[-1].strip()
        h.append({"role": "assistant", "content": cut})
        note = f"(ملاحظة لك: العميل قاطعك قبل ما تقول: «{rest}». رد على كلامه أول، وبعدين ارجع للنقطة إذا لسا مهمة.) "
        h.append({"role": "user", "content": note + said})
    else:
        h.append({"role": "assistant", "content": sc["first_point"]})
        h.append({"role": "user", "content": said})
    return h


def dialect(text: str) -> dict:
    words = re.findall(r"[؀-ۿ]+", text)
    return {"saudi": sum(w in SAUDI for w in words), "msa": sum(w in MSA for w in words),
            "other": sum(w in OTHER for w in words)}


def run(name: str) -> dict:
    brain.start(name)
    rows = []
    try:
        for case in CASES:
            t0 = time.time()
            first, sents = None, []
            for s in brain.sentences(build(case)):
                first = first or time.time() - t0
                sents.append(s)
            total = time.time() - t0
            reply = " ".join(sents)
            ok = bool(re.search(case[3], reply))
            rows.append({"case": case[4], "said": case[2], "reply": reply, "first_s": round(first or total, 2),
                         "total_s": round(total, 2), "fact_ok": ok, **dialect(reply)})
            print(f"  {name:12s} {rows[-1]['first_s']:5.2f}s {'✓' if ok else '✗'} {case[4]:38s} | {reply[:90]}", flush=True)
    finally:
        brain.stop()
    fs = sorted(r["first_s"] for r in rows)
    return {"rows": rows, "median_first_s": fs[len(fs) // 2], "facts_ok": sum(r["fact_ok"] for r in rows),
            "saudi": sum(r["saudi"] for r in rows), "msa": sum(r["msa"] for r in rows), "other": sum(r["other"] for r in rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brains", default="qwen3-14b,gemma-12b,claude-opus")
    a = ap.parse_args()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for b in a.brains.split(","):
        if not brain.available().get(b):
            print("skip (not available):", b)
            continue
        print("==", b, flush=True)
        res[b] = run(b)
        r = res[b]
        print(f"   median first sentence {r['median_first_s']}s · facts {r['facts_ok']}/{len(CASES)} · "
              f"saudi {r['saudi']} msa {r['msa']} other {r['other']}", flush=True)
        OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
