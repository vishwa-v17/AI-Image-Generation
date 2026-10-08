Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " Starting Text-to-Image Generator (Generative AI Mini-Project)" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan

if (Test-Path ".\venv\Scripts\Activate.ps1") {
    & ".\venv\Scripts\Activate.ps1"
    python app.py
} else {
    python app.py
}
