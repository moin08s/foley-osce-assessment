@echo off
setlocal
title Foley Catheterization OSCE Assessment - DY Patil Medical College

echo ===========================================================================
echo  D Y PATIL MEDICAL COLLEGE - AI OSCE CATHETERIZATION ASSESSMENT ENGINE
echo ===========================================================================

cd /d "C:\Medical"

if "%~1"=="" (
    echo [*] Checking C:\Medical\new_videos for incoming recordings...
    ".venv\Scripts\python.exe" -m catheter_ai_project.src.assess_new_video
) else (
    echo [*] Assessing video: %~1
    ".venv\Scripts\python.exe" -m catheter_ai_project.src.assess_new_video "%~1"
)

echo.
echo ===========================================================================
echo Processing finished. Check reports in catheter_ai_project\reports
echo ===========================================================================
pause
