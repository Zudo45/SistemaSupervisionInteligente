@echo off
REM Sistema de Supervision Inteligente.
REM La primera vez instala todo. Luego crea un link publico con una clave
REM nueva, inicia la pagina y abre el navegador en http://localhost:8501.
REM NO cerrar la ventana mientras se use la pagina.
cd /d "%~dp0"
title Sistema de Supervision Inteligente - NO CERRAR
call :preparar || exit /b 1
".venv\Scripts\python.exe" integracion\publicar.py
echo.
echo La pagina se detuvo.
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
