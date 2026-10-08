@echo off
echo ========================================================
echo Starting Text-to-Image Generator (Generative AI Mini-Project)
echo ========================================================

IF EXIST "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    python app.py
) ELSE (
    python app.py
)
pause
