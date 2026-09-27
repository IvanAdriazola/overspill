@echo off
rem Native-Windows Overspill environment: MSVC x64 toolchain + CUDA 13.0 from the nvidia-cuda-* pip wheels in .venv-win.
rem Usage: call experiment\win\env_win.bat   (then run python / ft from .venv-win)
set "REPO=%~dp0..\.."
for %%I in ("%REPO%") do set "REPO=%%~fI"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_HOME=%REPO%\.venv-win\Lib\site-packages\nvidia\cu13"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%CUDA_HOME%\bin\x86_64;%REPO%\.venv-win\Scripts;%PATH%"
set "TORCH_CUDA_ARCH_LIST=8.6"
set "TVM_FFI_CUDA_ARCH_LIST=8.6"
set "FLASHINFER_CUDA_ARCH_LIST=8.6"
set "DISTUTILS_USE_SDK=1"
