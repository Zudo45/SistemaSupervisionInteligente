@echo off
REM Pagina sin link publico (por ejemplo, sin internet).
cd /d "%~dp0"
title Sistema de Supervision Inteligente (local) - NO CERRAR
call :preparar || exit /b 1
if exist "resultados	unel.json" del "resultados	unel.json"
".venv\Scripts\python.exe" -m streamlit run app.py --server.headless false
pause
exit /b

:preparar
if exist ".venv\Scripts\python.exe" exit /b 0
echo Preparando el entorno por primera vez (puede tardar varios minutos)...
py -3.12 -m venv .venv 2>nul || python -m venv .venv
if not exist ".venv\Scripts\python.exe" (
    echo No se encontro Python 3.12. Instalalo desde https://www.python.org y vuelve a intentar.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requisitos.txt
exit /b 0
