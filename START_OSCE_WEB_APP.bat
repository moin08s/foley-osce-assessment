@echo off
setlocal
title D Y Patil Medical College - OSCE Catheterization Web Suite

echo ===========================================================================
echo  D Y PATIL MEDICAL COLLEGE - AI OSCE CATHETERIZATION WEB SUITE
echo ===========================================================================
echo [*] Starting local edge AI web server at http://127.0.0.1:8000 ...
echo [*] Features:
echo     - One-click assessment of clinical videos
echo     - Interactive 13-step OSCE Checklist with timestamps & confidence
echo     - Native HTML5 video player with step jump seeking
echo     - Procedural sequence & critical safety audits
echo     - Downloadable official scorecards & JSON reports
echo ===========================================================================

cd /d "C:\Medical"

start "" http://127.0.0.1:8000

"C:\Users\Moin\AppData\Local\Programs\Python\Python312\python.exe" web_app.py

pause
