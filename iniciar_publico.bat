@echo off
REM Inicia la pagina con un link publico (tunel de Cloudflare) para
REM que se pueda abrir desde otras redes. Las visitas necesitan la clave
REM que aparece en esta ventana y en la barra lateral de la pagina.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Primero ejecuta iniciar_app.bat una vez para instalar el entorno.
    pause
    exit /b 1
)

if not exist "herramientas\cloudflared.exe" (
    echo Descargando cloudflared desde la pagina oficial de Cloudflare...
    if not exist herramientas mkdir herramientas
    powershell -NoProfile -Command "Invoke-WebRequest -Uri https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe -OutFile herramientas\cloudflared.exe"
)

".venv\Scripts\python.exe" integracion\publicar.py
pause
