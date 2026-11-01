@echo off
rem Double-click this file in Windows Explorer.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PY=python
if exist "venv\Scripts\python.exe" set PY=venv\Scripts\python.exe
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
echo.
echo === Tests (ohne Video) ===
%PY% main.py test
echo.
pause
