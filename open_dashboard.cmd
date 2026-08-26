@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0open_dashboard.ps1" %*
exit /b %ERRORLEVEL%
