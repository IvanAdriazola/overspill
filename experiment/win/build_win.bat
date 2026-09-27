@echo off
rem Build + editable-install the fork into .venv-win (native Windows).
call "%~dp0env_win.bat"
cd /d "%REPO%"
nvcc --version | findstr release
uv pip install --python .venv-win\Scripts\python.exe --no-sources --no-build-isolation -e . 2>&1
.venv-win\Scripts\python.exe -c "import freetoken.kernel._cpu_moe as m, freetoken.kernel._pinned_tensor, freetoken.kernel._ple_store; print('extensions OK', m.__file__)"
