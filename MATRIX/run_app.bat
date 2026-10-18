@echo off
echo ========================================================
echo   Matrix Reconstruction and Error Correction App
echo ========================================================
echo.

IF EXIST "venv\Scripts\streamlit.exe" (
    echo Starting Streamlit application...
    "venv\Scripts\streamlit.exe" run app.py
) ELSE IF EXIST ".venv\Scripts\streamlit.exe" (
    echo Starting Streamlit application...
    ".venv\Scripts\streamlit.exe" run app.py
) ELSE (
    echo Virtual environment not found. Please run:
    echo python -m venv venv
    echo .\venv\Scripts\pip install -r requirements.txt
    pause
)
