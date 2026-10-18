Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "   Matrix Reconstruction and Error Correction App" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

$streamlitPath = if (Test-Path ".\venv\Scripts\streamlit.exe") { ".\venv\Scripts\streamlit.exe" } elseif (Test-Path ".\.venv\Scripts\streamlit.exe") { ".\.venv\Scripts\streamlit.exe" } else { $null }
if ($streamlitPath) {
    Write-Host "Launching Streamlit application on local server..." -ForegroundColor Green
    & $streamlitPath run app.py
} else {
    Write-Host "Virtual environment not detected. Creating and installing dependencies..." -ForegroundColor Yellow
    python -m venv venv
    .\venv\Scripts\pip install -r requirements.txt
    & .\venv\Scripts\streamlit.exe run app.py
}
