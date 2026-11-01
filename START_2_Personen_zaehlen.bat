@echo off
rem Double-click this file in Windows Explorer.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PY=python
if exist "venv\Scripts\python.exe" set PY=venv\Scripts\python.exe
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
if not exist "videos\restaurant.mp4" %PY% main.py testvideo
echo.
echo === Personen zaehlen - Fenster mit Q schliessen ===
echo Beim ersten Start wird das YOLO-Modell heruntergeladen, bitte kurz warten.
%PY% main.py 1.3
echo.
pause
