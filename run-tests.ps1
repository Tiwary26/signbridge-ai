# Runs automated PyTest test suite
Write-Host "Running SignBridge AI PyTest Suite..." -ForegroundColor Cyan
Set-Location -Path "$PSScriptRoot\backend"
& ".\venv\Scripts\pytest.exe" tests/ -v
