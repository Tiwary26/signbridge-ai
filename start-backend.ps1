# Starts the SignBridge AI FastAPI Backend
Write-Host "Starting SignBridge AI FastAPI Backend on http://localhost:8000..." -ForegroundColor Cyan
Set-Location -Path "$PSScriptRoot\backend"
& ".\venv\Scripts\uvicorn.exe" app.main:app --host 0.0.0.0 --port 8000 --reload
