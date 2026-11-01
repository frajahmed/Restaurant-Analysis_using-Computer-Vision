@echo off
rem Double-click this file in Windows Explorer.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PY=python
if exist "venv\Scripts\python.exe" set PY=venv\Scripts\python.exe
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
if not exist "videos\restaurant.mp4" %PY% main.py testvideo
echo.
echo === Dashboard - der Browser oeffnet sich gleich von selbst ===
echo Zum Beenden dieses Fenster schliessen.
set OPEN_BROWSER=1
%PY% main.py
echo.
pause
