# One-time setup on Windows (PowerShell). Safe to run again: every step skips what already exists.
#   powershell -ExecutionPolicy Bypass -File setup_windows.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$EXT = Join-Path $HOME "Downloads\sarj-ext-data"          # big files live outside git (same place as on the Mac)
New-Item -ItemType Directory -Force -Path $EXT | Out-Null

Write-Host "1/5  Python 3.12 environment"
if (-not (Test-Path .venv)) { py -3.12 -m venv .venv }
.\.venv\Scripts\python -m pip install -q --upgrade pip
.\.venv\Scripts\python -m pip install -q -r requirements-windows.txt

Write-Host "2/5  Voiceprint model CAM++ (28 MB, Apache-2.0)"
$SPK = Join-Path $EXT "spk"; New-Item -ItemType Directory -Force -Path $SPK | Out-Null
$CAM = Join-Path $SPK "campp.onnx"
if (-not (Test-Path $CAM)) { Invoke-WebRequest "https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx" -OutFile $CAM }

Write-Host "3/5  Meaning model (models\student, 110 MB, GitHub release models-v1)"
New-Item -ItemType Directory -Force -Path models\student | Out-Null
if (-not (Test-Path models\student\sem.onnx)) { gh release download models-v1 -R o6lq/sarj-turn-taking -p student-sem.onnx -O models\student\sem.onnx }
if (-not (Test-Path models\student\tokenizer.json)) { gh release download models-v1 -R o6lq/sarj-turn-taking -p student-tokenizer.json -O models\student\tokenizer.json }

Write-Host "4/5  Generated test calls (GitHub release data-v1)"
foreach ($set in "synth-test", "dev-aug") {
  if (-not (Test-Path "data\$set\manifest.json")) {
    gh release download data-v1 -R o6lq/sarj-turn-taking -p "$set.zip" -D $env:TEMP --clobber
    Expand-Archive -Force (Join-Path $env:TEMP "$set.zip") -DestinationPath data
  }
}

Write-Host "5/5  Check: the official scorer on Sarj's 24 calls (first run downloads Whisper turbo, ~1.6 GB)"
.\.venv\Scripts\python -m tools.sim
Write-Host "Done. Live page:  .\.venv\Scripts\python -m tools.live   then open http://127.0.0.1:8790/demo"
