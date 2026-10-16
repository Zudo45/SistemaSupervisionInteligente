@echo off
REM Inicia el sistema del grupo VIDEO por separado (su ventana de camara).
REM Las capturas de la alarma se guardan en video\alertas y la pagina las procesa.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Primero ejecuta iniciar_app.bat para preparar el entorno.
    pause
    exit /b 1
)
cd video
"..\.venv\Scripts\python.exe" principal.py
