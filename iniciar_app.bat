@echo off
REM Inicia la pagina del modulo IMAGEN.
REM La primera vez crea el entorno virtual e instala las librerias.
cd /d "%~dp0"
call :preparar || exit /b 1
".venv\Scripts\python.exe" -m streamlit run app.py
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
