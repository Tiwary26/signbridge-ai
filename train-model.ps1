# Runs PyTorch model training
Write-Host "Running SignBridge AI PyTorch Model Training..." -ForegroundColor Cyan
Set-Location -Path "$PSScriptRoot"
& ".\backend\venv\Scripts\python.exe" ml\scripts\train_model.py --epochs 30 --batch-size 32 --lr 0.001
