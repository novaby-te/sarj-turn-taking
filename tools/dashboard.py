"""Build the measurement dashboard (Arabic, RTL) from the latest simulator runs.

    .venv/bin/python -m tools.dashboard        # -> ~/Downloads/لوحة قياس سرج.html
"""
from __future__ import annotations

import csv
import html
import json
import sys
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
from ttbench.dataset import read_wav_stereo  # noqa: E402

OUT = Path.home() / "Downloads" / "لوحة قياس سرج.html"
SETS = [("dev", "data/dev", "بيانات سرج (٢٤ مكالمة)"), ("dev-aug", "data/dev-aug", "نفس المكالمات بظروف أصعب (٩٦)"),
        ("synth", "data/synth", "مكالمات ولّدناها (١٦٠)"), ("live", "data/live", "تجاربك بالمايك")]
CAT_AR = {"GI": "مقاطعة حقيقية", "BC": "تأييد (إيه/أها)", "HT": "تردد وتفكير", "BG": "خلفية (تلفزيون/شارع)",
          "NV": "أصوات بلا كلام", "AM": "كلمات محيّرة", "OV": "تداخل", "DV": "لهجات"}
OUTCOME_AR = {"yield": "سكت", "miss": "ما سكت", "kept": "كمّل", "false_stop": "سكت غلط", "hiccup": "سكت ورجع", "eot": "نهاية دور"}
EXP_AR = {"STOP": "لازم يسكت", "KEEP": "لازم يكمّل", "EITHER": "الاثنين مقبول", "TAKE_TURN": "يرد بعد ما يخلص"}


def esc(s):
    return html.escape(str(s if s is not None else ""))


def energy(path: Path, step_ms=100):
    st = read_wav_stereo(path).astype(np.float64) / 32768
    k = 16 * step_ms
    n = len(st) // k
    out = []
    for ch in (0, 1):
        f = st[: n * k, ch].reshape(n, k)
        out.append((20 * np.log10(np.sqrt((f ** 2).mean(1)) + 1e-9)).round(1))
    return out


def timeline_svg(lab: dict, decisions: list, trace: list, audio_path: Path) -> str:
    dur = lab["audio"]["duration_ms"]
    W, H = 1000, 128
    x = lambda ms: ms / dur * W  # noqa: E731
    ag, ca = energy(audio_path)
    parts = [f'<svg viewBox="0 0 {W} {H}" class="tl" preserveAspectRatio="none" role="img">']
    for u in lab["agent_utterances"]:
        parts.append(f'<rect x="{x(u["start_ms"]):.1f}" y="6" width="{x(u["end_ms"]) - x(u["start_ms"]):.1f}" height="14" class="agent"><title>الوكيل: {esc(u["text"])}</title></rect>')
    for e in lab["events"]:
        off = e.get("offset_ms", e.get("turn_end_ms", e["onset_ms"]))
        cls = {"STOP": "ev-stop", "KEEP": "ev-keep", "EITHER": "ev-either", "TAKE_TURN": "ev-eot"}[e["expected"]]
        parts.append(f'<rect x="{x(e["onset_ms"]):.1f}" y="26" width="{max(2, x(off) - x(e["onset_ms"])):.1f}" height="70" class="{cls}"><title>{EXP_AR[e["expected"]]}: {esc(e.get("rationale", ""))}</title></rect>')
        if e.get("evident_ms"):
            parts.append(f'<line x1="{x(e["evident_ms"]):.1f}" x2="{x(e["evident_ms"]):.1f}" y1="26" y2="96" class="evident"/>')
    pts = " ".join(f"{x(i * 100):.1f},{96 - max(0, min(70, (v + 70))):.1f}" for i, v in enumerate(ca))
    parts.append(f'<polyline points="{pts}" class="caller"/>')
    for s in lab["user_segments"]:
        if s.get("text"):
            parts.append(f'<rect x="{x(s["start_ms"]):.1f}" y="98" width="{max(2, x(s["end_ms"]) - x(s["start_ms"])):.1f}" height="8" class="useg"><title>المتصل: {esc(s["text"])}</title></rect>')
    for t, d in decisions:
        cls = "dec-stop" if d == "STOP" else "dec-eot" if d == "TAKE_TURN" else "dec-resume"
        parts.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="2" y2="{H - 2}" class="{cls}"><title>{d} عند {t / 1000:.2f} ث</title></line>')
    for r in trace:
        if r["kind"] == "asr" and r.get("text"):
            parts.append(f'<circle cx="{x(r["t"]):.1f}" cy="116" r="3.2" class="asr"><title>{r["t"] / 1000:.2f} ث — سمع: «{esc(r["text"])}» ({esc(r.get("why", ""))}, {r.get("ms", 0):.0f} ms)</title></circle>')
    parts.append("</svg>")
    return "".join(parts)


def history_svg(rows: list[dict]) -> str:
    if not rows:
        return "<p class='muted'>لا يوجد سجل بعد.</p>"
    W, H, P = 1000, 220, 30
    series = {}
    for r in rows:
        series.setdefault(r["data"], []).append(r)
    n = max(len(v) for v in series.values())
    parts = [f'<svg viewBox="0 0 {W} {H}" class="hist" role="img">']
    for y in (0, 25, 50, 75, 100):
        yy = H - P - y / 100 * (H - 2 * P)
        parts.append(f'<line x1="{P}" x2="{W - 10}" y1="{yy:.1f}" y2="{yy:.1f}" class="grid"/><text x="{P - 6}" y="{yy + 4:.1f}" class="ax">{y}</text>')
    for i, (name, rs) in enumerate(series.items()):
        pts = []
        for j, r in enumerate(rs):
            xx = P + (j + (n - len(rs))) / max(1, n - 1) * (W - P - 20)
            yy = H - P - float(r["core"]) / 100 * (H - 2 * P)
            pts.append((xx, yy, r))
        parts.append(f'<polyline points="{" ".join(f"{a:.1f},{b:.1f}" for a, b, _ in pts)}" class="s s{i}"/>')
        for a, b, r in pts:
            parts.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="4" class="p{i}"><title>{esc(r["data"])} — {esc(r["core"])} — {esc(r["name"])} — {esc(r["when"])}</title></circle>')
        parts.append(f'<text x="{W - 12}" y="{18 + i * 16}" class="lg l{i}" text-anchor="end">● {esc(name)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def build() -> Path:
    hist_path = KIT / "runs" / "sim-history.csv"
    rows = list(csv.DictReader(open(hist_path, encoding="utf-8"))) if hist_path.exists() else []
    rows = [r for r in rows if r.get("data") in {s for s, _, _ in SETS}]
    cards, cats_html, fails_html, tls = [], [], [], []
    for key, folder, title in SETS:
        run = KIT / "runs" / f"sim-{key}-latest"
        if not (run / "score.json").exists():
            continue
        r = json.loads((run / "score.json").read_text())
        traces = json.loads((run / "trace.json").read_text()) if (run / "trace.json").exists() else {}
        c = r["core"]
        prev = [x for x in rows if x["data"] == key]
        delta = ""
        if len(prev) >= 2:
            dlt = float(prev[-1]["core"]) - float(prev[-2]["core"])
            delta = f'<span class="{"up" if dlt >= 0 else "down"}">{dlt:+.1f}</span>'
        cards.append(f"""<div class="card"><div class="ct">{esc(title)}</div><div class="big">{c['core_score']:.1f} {delta}</div>
<div class="kv"><span>R (يسكت لما لازم)</span><b>{c['R']:.2f}</b></div><div class="kv"><span>S (يكمّل لما لازم)</span><b>{c['S']:.2f}</b></div>
<div class="kv"><span>A (المحيّر)</span><b>{'—' if c['A'] is None else f"{c['A']:.2f}"}</b></div>
<div class="kv"><span>سكتات غلط</span><b>{r['rates']['false_stops']}</b></div><div class="kv"><span>مقاطعات فاتته</span><b>{r['rates']['misses']}</b></div>
<div class="kv"><span>متوسط التأخير</span><b>{'—' if r['latency_ms']['median'] is None else f"{r['latency_ms']['median']:.0f} ms"}</b></div>
<div class="kv"><span>نهاية الدور (اختياري)</span><b>{r['eot']['eot_score']:.1f}</b></div></div>""")
        trs = "".join(f"<tr><td>{esc(k)} — {esc(CAT_AR.get(k, ''))}</td><td>{'—' if v['R'] is None else f'{v['R']:.2f}'}</td><td>{'—' if v['S'] is None else f'{v['S']:.2f}'}</td>"
                      f"<td>{v['stop_yielded']}/{v['n_stop']}</td><td>{v['keep_passed']}/{v['n_keep']}</td><td>{v['protected_clean']}/{v['n_scenarios']}</td></tr>"
                      for k, v in r["categories"].items())
        cats_html.append(f"<details {'open' if key == 'dev' else ''}><summary>{esc(title)} — حسب الفئة</summary><table><tr><th>الفئة</th><th>R</th><th>S</th><th>سكت لما لازم</th><th>كمّل لما لازم</th><th>مكالمات بلا سكتة غلط</th></tr>{trs}</table></details>")
        bad = [e for e in r["events"] if e["credit"] < 1 and e["kind"] == "interruption"]
        stray = [s for s in r["scenarios"] if s["spurious_stops"]]
        li = []
        for e in bad[:60]:
            heard = [t["text"] for t in traces.get(e["scenario_id"], []) if t["kind"] == "asr" and t.get("text")
                     and e["onset_ms"] - 300 <= t["t"] <= e["onset_ms"] + 6000]
            li.append(f"<tr><td>{esc(e['event_id'])}</td><td>{EXP_AR.get(e['expected'], e['expected'])}</td><td>{OUTCOME_AR.get(e['outcome'], e['outcome'])}</td>"
                      f"<td>{e['credit']:.2f}</td><td>{esc(e.get('detail', ''))}</td><td class='heard'>{esc(' ← '.join(heard[-3:]))}</td></tr>")
        for s in stray[:30]:
            li.append(f"<tr><td>{esc(s['scenario_id'])}</td><td>—</td><td>سكتة بلا سبب</td><td>{s['credit']:.2f}</td><td>عند {esc(s['spurious_stops'])} ms</td><td></td></tr>")
        fails_html.append(f"<details {'open' if key == 'dev' else ''}><summary>{esc(title)} — الأخطاء ({len(bad)} حدث + {len(stray)} مكالمة فيها سكتة بلا سبب)</summary>"
                          f"<table><tr><th>الحدث</th><th>المطلوب</th><th>اللي صار</th><th>الدرجة</th><th>التفصيل</th><th>وش سمع النموذج</th></tr>{''.join(li)}</table></details>")
        if key in ("dev", "live"):
            log = [json.loads(l) for l in (run / "log.jsonl").read_text().splitlines() if l.strip()]
            meta = json.loads((run / "run.json").read_text())
            for s in meta["scenarios"]:
                sid = s["scenario_id"]
                lab = json.loads((KIT / folder / "labels" / f"{sid}.json").read_text())
                decs = [(l["stream_ms"] - s["start_ms"], l["msg"]["decision"]) for l in log
                        if s["start_ms"] <= l["stream_ms"] < s["start_ms"] + s["duration_ms"] + 1500]
                ev = [e for e in r["events"] if e["scenario_id"] == sid]
                ok = all(e["credit"] >= 0.99 for e in ev if e["kind"] == "interruption") and not any(
                    x["spurious_stops"] for x in r["scenarios"] if x["scenario_id"] == sid)
                wav = KIT / folder / "audio" / lab["audio"]["file"]
                heard = " | ".join(f"{t['t'] / 1000:.1f}ث «{t['text']}»" for t in traces.get(sid, []) if t["kind"] == "asr" and t.get("text"))
                tls.append(f"""<div class="call {'ok' if ok else 'bad'}"><div class="ch"><b>{esc(sid)}</b> <span class="tag">{esc(CAT_AR.get(lab['category'], ''))}</span>
<span class="tag">{esc(lab['dialect'])}</span><span class="tag">{esc('، '.join(lab['acoustic']['conditions']))}</span>
<span class="res">{'✓' if ok else '✗'} {' · '.join(f"{EXP_AR.get(e['expected'], '')}: {OUTCOME_AR.get(e['outcome'], '')} ({e['credit']:.2f})" for e in ev)}</span></div>
{timeline_svg(lab, decs, traces.get(sid, []), wav)}
<div class="heardline">سمع: {esc(heard) or '—'}</div><audio controls preload="none" src="file://{esc(wav)}"></audio></div>""")
    page = f"""<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>لوحة قياس سرج</title><style>
:root{{--bg:#f6f5f2;--card:#fff;--ink:#1c1c1c;--muted:#6b6b6b;--line:#e3e0da;--agent:#7b8cde;--stop:#e5484d;--keep:#30a46c;--either:#f5a524;--eot:#8e4ec6;--caller:#222}}
@media (prefers-color-scheme:dark){{:root{{--bg:#141414;--card:#1e1e1e;--ink:#eee;--muted:#9a9a9a;--line:#333;--caller:#ddd}}}}
body{{background:var(--bg);color:var(--ink);font-family:-apple-system,"SF Arabic","Geeza Pro",Tahoma,sans-serif;margin:0;padding:20px 16px;line-height:1.6}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:18px;margin:28px 0 10px}} .muted{{color:var(--muted)}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}} .ct{{font-size:13px;color:var(--muted)}}
.big{{font-size:34px;font-weight:700}} .up{{color:var(--keep);font-size:16px}} .down{{color:var(--stop);font-size:16px}}
.kv{{display:flex;justify-content:space-between;font-size:13px;border-top:1px dashed var(--line);padding:3px 0}}
table{{border-collapse:collapse;width:100%;font-size:13px;background:var(--card)}} td,th{{border:1px solid var(--line);padding:5px 8px;text-align:right;vertical-align:top}}
details{{margin:8px 0}} summary{{cursor:pointer;font-weight:600}} .heard{{font-size:12px;color:var(--muted)}}
.call{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px;margin:10px 0}} .call.bad{{border-color:var(--stop)}}
.ch{{font-size:13px;display:flex;flex-wrap:wrap;gap:6px;align-items:center}} .tag{{background:var(--bg);border-radius:6px;padding:0 6px;font-size:12px}}
.res{{margin-inline-start:auto;font-size:12px}} .heardline{{font-size:12px;color:var(--muted);margin:4px 0}}
svg.tl{{width:100%;height:128px;direction:ltr}} svg.hist{{width:100%;height:auto;direction:ltr}}
.agent{{fill:var(--agent);opacity:.7}} .ev-stop{{fill:var(--stop);opacity:.18}} .ev-keep{{fill:var(--keep);opacity:.18}} .ev-either{{fill:var(--either);opacity:.22}} .ev-eot{{fill:var(--eot);opacity:.15}}
.evident{{stroke:var(--stop);stroke-dasharray:3 2;stroke-width:1.5}} .caller{{fill:none;stroke:var(--caller);stroke-width:1}} .useg{{fill:var(--muted);opacity:.5}}
.dec-stop{{stroke:var(--stop);stroke-width:3}} .dec-eot{{stroke:var(--eot);stroke-width:2;stroke-dasharray:4 2}} .dec-resume{{stroke:var(--keep);stroke-width:2}}
.asr{{fill:var(--either)}} .grid{{stroke:var(--line)}} .ax{{font-size:11px;fill:var(--muted);text-anchor:end}}
.s{{fill:none;stroke-width:2}} .s0,.l0{{stroke:#3e63dd;fill:#3e63dd}} .s1,.l1{{stroke:#e5484d;fill:#e5484d}} .s2,.l2{{stroke:#30a46c;fill:#30a46c}} .s3,.l3{{stroke:#8e4ec6;fill:#8e4ec6}}
.p0{{fill:#3e63dd}} .p1{{fill:#e5484d}} .p2{{fill:#30a46c}} .p3{{fill:#8e4ec6}} .s{{fill:none}} .lg{{font-size:12px;stroke:none}}
.legend span{{display:inline-block;margin-inline-end:12px;font-size:12px}} .sw{{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:middle;margin-inline-end:4px}}
</style></head><body>
<h1>لوحة قياس نموذج «متى يسكت الوكيل»</h1><div class="muted">تتحدّث كل مرة تشغّل <code>.venv/bin/python -m tools.loop</code>. نفس طريقة حساب سرج الرسمية (ttbench.score)، بدون انتظار الوقت الحقيقي.</div>
<h2>آخر نتيجة</h2><div class="cards">{''.join(cards)}</div>
<h2>هل التعديلات تحسّن؟ (سجل الدرجات)</h2><div class="card">{history_svg(rows)}</div>
<h2>حسب الفئة</h2>{''.join(cats_html)}
<h2>وين أخطأ النموذج (هذي اللي نشتغل عليها)</h2>{''.join(fails_html)}
<h2>كل مكالمة على خط زمني</h2>
<div class="legend"><span><i class="sw" style="background:var(--agent)"></i>الوكيل يتكلم</span><span><i class="sw" style="background:var(--stop);opacity:.4"></i>لازم يسكت (الخط المتقطع = لحظة الوضوح)</span>
<span><i class="sw" style="background:var(--keep);opacity:.4"></i>لازم يكمّل</span><span><i class="sw" style="background:var(--either)"></i>نقطة = النموذج قرأ كلام (مرّر الماوس)</span>
<span><i class="sw" style="background:var(--stop)"></i>خط أحمر عريض = قرار STOP</span><span>الخط الأسود = صوت المتصل</span></div>
{''.join(tls)}
</body></html>"""
    OUT.write_text(page, encoding="utf-8")
    return OUT


if __name__ == "__main__":
    print(build())
