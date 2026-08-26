@echo off
setlocal
set "PROJECT_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PROJECT_PYTHON%" (
    >&2 echo [python] project venv not found: "%PROJECT_PYTHON%"
    >&2 echo [python] create .venv first, then retry.
    exit /b 2
)
set "PYTHONPATH=%~dp0src;%~dp0;%PYTHONPATH%"
"%PROJECT_PYTHON%" %*
exit /b %ERRORLEVEL%
