@echo off
call "%~dp0env_win.bat"
cd /d "%REPO%"
.venv-win\Scripts\python.exe experiment\win\kernel_check.py %*
