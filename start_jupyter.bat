@echo off
REM ============================================================
REM Start Jupyter Lab - ML Workspace
REM Hospital Anchieta Ceilandia
REM ============================================================

cd /d C:\ml_workspace

REM Ativa o virtualenv
call venv_ml\Scripts\activate.bat

REM Verifica se ativou
if "%VIRTUAL_ENV%"=="" (
    echo [ERRO] Falha ao ativar o venv_ml
    pause
    exit /b 1
)

echo ============================================================
echo ML WORKSPACE - JUPYTER LAB
echo ============================================================
echo Python: %VIRTUAL_ENV%
echo Workspace: %CD%
echo.
echo Iniciando Jupyter Lab...
echo (Para encerrar, feche esta janela ou pressione Ctrl+C duas vezes)
echo ============================================================
echo.

REM Inicia o Jupyter Lab dentro da pasta notebooks
jupyter lab --notebook-dir=notebooks --no-browser --ip=127.0.0.1