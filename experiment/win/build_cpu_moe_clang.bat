@echo off
rem Rebuild the CPU MoE executor with clang-cl (arg: clang, default) or restore the MSVC build (arg: msvc).
call "%~dp0env_win.bat"
cd /d "%REPO%"
.venv-win\Scripts\python.exe experiment\win\build_cpu_moe_clang.py %*
