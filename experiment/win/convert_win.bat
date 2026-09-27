@echo off
rem Native-Windows HF safetensors -> FTW conversion (ft checkpoint) in the .venv-win environment.
rem Usage: convert_win.bat --model <hf_dir> --out <ftw_dir> [ft checkpoint options]
call "%~dp0env_win.bat"
"%REPO%\.venv-win\Scripts\ft.exe" checkpoint %*
