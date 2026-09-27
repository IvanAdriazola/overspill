@echo off
rem Native-Windows Overspill serving Qwen3.8-Flash-Next (FTW on C:) - or any model via FT_MODEL_DIR (the API name stays
rem "flashnext"). %1 = cpu | mixed (default) | stock (FreeToken's default strategy, no CPU layers); more args pass through.
rem Same engine flags as experiment/serve_flashnext.sh (WSL).
call "%~dp0env_win.bat"
if not defined FT_MODEL_DIR set "FT_MODEL_DIR=C:\AIModels\flashnext_ftw"
if not defined FT_FILE_BANKS set "FT_FILE_BANKS=1"
if not defined FT_EMBED_HOST set "FT_EMBED_HOST=1"
if not defined FT_HEAD_HOST set "FT_HEAD_HOST=1"
if not defined FT_HEAD_I8 set "FT_HEAD_I8=1"
if not defined FT_CPU_PREFILL_MAX set "FT_CPU_PREFILL_MAX=256"
if not defined FT_WILLNEED set "FT_WILLNEED=1"
set "PYTHONUNBUFFERED=1"
set "MODE=%~1"
if "%MODE%"=="" set "MODE=mixed"
shift
set "STRATEGY=--moe-strategy offload --moe-cpu-layers auto"
if /i "%MODE%"=="cpu" set "STRATEGY=--moe-strategy cpu"
if /i "%MODE%"=="stock" set "STRATEGY="
echo model dir: %FT_MODEL_DIR%  mode: %MODE%
"%REPO%\.venv-win\Scripts\ft.exe" serve --model "%FT_MODEL_DIR%" --served-model-name flashnext --text-model-only --host 127.0.0.1 --port 1919 %STRATEGY% --max-running-requests 1 --cuda-graph-max-bs 1 %1 %2 %3 %4 %5 %6 %7 %8
