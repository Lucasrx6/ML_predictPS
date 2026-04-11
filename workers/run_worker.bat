@echo off
REM ============================================================
REM ML PS Volume Worker — execução agendada
REM ============================================================

cd /d C:\ml_workspace

REM Ativa o virtualenv
call venv_ml\Scripts\activate.bat

REM Roda o worker
python workers\ml_ps_volume_worker.py

REM Não deixa janela aberta se rodou via scheduler
exit /b %ERRORLEVEL%