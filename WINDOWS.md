# تشغيل المشروع على ويندوز (للتيم)

الوقت: ٢٠–٤٠ دقيقة، أغلبه تحميل. كل خطوة تتخطى نفسها إذا انعادت.

## وش يشتغل على ويندوز، ووش للماك بس

| الجزء | على الماك | على ويندوز |
|---|---|---|
| نموذج القرار كامل (L0–L5، RESUME، طبقة المعنى) | ✅ | ✅ نفس الكود |
| تفريغ الكلام | Cohere Arabic (أدق، عبر MLX) | **faster-whisper** (Whisper turbo). مع كرت NVIDIA سريع، وعلى CPU أبطأ (٠٫٥–١٫٥ ث للقراءة) |
| بصمة الصوت، وكاشف الكلام، ونموذج المعنى | ✅ | ✅ (ONNX) |
| عقل الوكيل (المكالمة الكاملة) | Gemma 4 محلي | **LM Studio** أو **Ollama** (Gemma 4 E4B أو 12B)، أو Claude |
| صوت الوكيل | ماجد (محلي) أو Edge | **Edge** (يحتاج إنترنت) |
| المحاكي والدرجة الرسمية وصفحة العرض | ✅ | ✅ |

> الدرجات الرسمية في التقرير مأخوذة على الماك بـ Cohere. على ويندوز بـ Whisper توقّع درجة أقل شوي، وهذا طبيعي.

## ١) التثبيت (مرة وحدة)

افتح **PowerShell** وانسخ سطر سطر:

```powershell
winget install -e --id Python.Python.3.12
winget install -e --id Git.Git
winget install -e --id GitHub.cli
```

سكّر PowerShell وافتحه من جديد، وبعدها:

```powershell
gh auth login
cd $HOME\Documents
gh repo clone o6lq/sarj-turn-taking
cd sarj-turn-taking
powershell -ExecutionPolicy Bypass -File setup_windows.ps1
```

في `gh auth login` اختر: GitHub.com، ثم HTTPS، ثم Login with a web browser، وكمّل الدخول من المتصفح بحسابك.

آخر خطوة في السكربت تشغّل المحاكي على مكالمات سرج الـ٢٤ وتطبع سطر يبدأ بـ `core`. إذا طلع، كل شي شغّال.

## ٢) الاستخدام اليومي

```powershell
cd $HOME\Documents\sarj-turn-taking
git pull
.\.venv\Scripts\python -m tools.sim                          # درجة سرج على مكالماتهم
.\.venv\Scripts\python -m tools.sim --data data\dev-aug      # النسخ الأصعب
.\.venv\Scripts\python -m tools.sim --data data\synth-test   # مكالمات ما شافها
.\.venv\Scripts\python -m tools.live                         # صفحة التجربة والعرض
```

صفحة العرض: افتح **http://127.0.0.1:8790/demo** في Chrome.

## ٣) عقل الوكيل على ويندوز (للمكالمة الكاملة فقط)

نموذج القرار ما يحتاج عقل. هذا بس لتبويب «جرّب حي» اللي فيه الوكيل يرد.

**الخيار أ: LM Studio (الأسهل)**
1. حمّل LM Studio من lmstudio.ai.
2. من البحث حمّل **gemma-4-e4b-it** (أو gemma-3-12b-it إذا جهازك فيه ١٦ GB رام أو أكثر).
3. من تبويب Developer شغّل السيرفر (Start Server، المنفذ 1234).
4. قبل ما تشغّل الصفحة:
```powershell
$env:TT_BRAIN_URL = "http://127.0.0.1:1234"
$env:TT_BRAIN_MODEL = "gemma-4-e4b-it"      # نفس الاسم اللي يطلع في LM Studio
.\.venv\Scripts\python -m tools.live
```
5. في الصفحة اختر العقل «LM Studio / Ollama».

**الخيار ب: Ollama**: `ollama pull gemma3:12b`، وبعدها `$env:TT_BRAIN_URL = "http://127.0.0.1:11434"` و `$env:TT_BRAIN_MODEL = "gemma3:12b"`.

**الخيار ج: Claude**: إذا عندك Claude Code مثبّت ومسجّل، يظهر «Claude Opus / Haiku» في الصفحة تلقائياً.

## ٤) إعدادات مفيدة

| المتغير | وش يسوي |
|---|---|
| `$env:TT_FW_MODEL = "medium"` | نموذج تفريغ أصغر لو الجهاز بطيء (الافتراضي `large-v3-turbo`) |
| `$env:TT_ASR = "none"` | بدون تفريغ (الصوت بس). أسرع شي، لكن ما يفهم الكلمات |
| `$env:TT_SEM = "0"` | يطفي طبقة المعنى (للمقارنة) |

## ٥) مشاكل شائعة

| المشكلة | الحل |
|---|---|
| `py` غير معروف | سكّر PowerShell وافتحه من جديد بعد تثبيت بايثون |
| `gh` يطلب صلاحية | `gh auth login`، وتأكد إن حسابك عنده وصول للريبو |
| تحميل Whisper بطيء | أول مرة بس (~١٫٦ GB). بعدها محفوظ |
| ما فيه صوت للوكيل | Edge يحتاج إنترنت. تأكد من الاتصال |
| المايك مكتوم أو يلقط صدى | استخدم **سماعة رأس**، واختر «سماعة رأس» في الصفحة |
| Execution Policy | شغّل السكربت بالسطر اللي فيه `-ExecutionPolicy Bypass` |

## ٦) قبل ما ترفع أي تعديل

```powershell
.\.venv\Scripts\python -m tools.sim                       # لازم ما تنزل الدرجة
git checkout -b اسمك/وصف-التعديل
git add -A; git commit -m "وش سويت وليش"
git push -u origin HEAD
gh pr create --fill
```

لا ترفع تسجيلات صوت ولا بيانات سرج خارج الريبو. الريبو خاص.
