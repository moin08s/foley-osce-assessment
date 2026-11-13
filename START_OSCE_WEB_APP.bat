@echo off
setlocal
title D Y Patil Medical College - OSCE Catheterization Web Suite

echo ===========================================================================
echo  D Y PATIL MEDICAL COLLEGE - AI OSCE CATHETERIZATION WEB SUITE
echo ===========================================================================
echo [*] Starting local edge AI web server at http://127.0.0.1:8000 ...
echo [*] Features:
echo     - Option A: Live Video Assessment (Webcam / Simulated stream)
echo     - Option B: Upload Video & Automated Slicing
echo     - Live Procedural Checklist & Dwell Timer Updates
echo     - Verifiable Step Video Clips Gallery
echo     - Human-in-the-Loop Editable Rubric & Active Continuous Learning
echo ===========================================================================

cd /d "C:\Medical"

start "" http://127.0.0.1:8000

".venv\Scripts\python.exe" -m catheter_ai_project.src.web_app

pause
