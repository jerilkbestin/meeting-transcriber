# One-time setup for teams-notetaker on Windows.
# Run from this folder in PowerShell:
#     powershell -ExecutionPolicy Bypass -File .\setup-windows.ps1

$ErrorActionPreference = "Stop"

Write-Host "Creating virtual environment (.venv)..." -ForegroundColor Cyan
py -3 -m venv .venv

Write-Host "Installing dependencies..." -ForegroundColor Cyan
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "`nRunning the offline self-test..." -ForegroundColor Cyan
& .\.venv\Scripts\python.exe selftest.py

if (-not $env:ANTHROPIC_API_KEY) {
    Write-Host "`nNo ANTHROPIC_API_KEY set — that is fine." -ForegroundColor Cyan
    Write-Host "Transcription runs locally and costs nothing. Every run writes"
    Write-Host "notes-prompt.md, which you can paste straight into claude.ai."
    Write-Host "Or run a local model:  ollama serve  +  --notes local"
}

Write-Host "`nDone. Next steps:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python -m teams_notetaker devices"
Write-Host "  python -m teams_notetaker record --title `"My Meeting`""
Write-Host "`nRemember to tell people on the call that you are recording." -ForegroundColor Yellow
