# ============================================================
# PAGINA PUBLICA (TUNEL DE CLOUDFLARE)
# Inicia la pagina y dos tuneles de Cloudflare (sin cuenta):
#   - pagina: puerto 8501
#   - video de la camara: puerto 8502
# Los links publicos y una clave para las visitas se guardan en
# resultados/tunel.json, que la pagina lee para mostrarlos.
#
# Uso (desde la raiz del proyecto): iniciar_publico.bat
# ============================================================

import os
import re
import sys
import json
import time
import secrets
import threading
import subprocess

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLOUDFLARED = os.path.join(RAIZ, "herramientas", "cloudflared.exe")
RUTA_TUNEL = os.path.join(RAIZ, "resultados", "tunel.json")

PUERTO_PAGINA = 8501
PUERTO_VIDEO = 8502

PATRON_LINK = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def abrir_tunel(puerto, nombre, links):
    """
    Lanza cloudflared y guarda el link publico cuando aparece en su salida.
    """

    proceso = subprocess.Popen(
        [CLOUDFLARED, "tunnel", "--no-autoupdate", "--url", f"http://localhost:{puerto}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    def leer():
        for linea in proceso.stdout:
            encontrado = PATRON_LINK.search(linea)
            if encontrado and nombre not in links:
                links[nombre] = encontrado.group(0)

    threading.Thread(target=leer, daemon=True).start()

    return proceso


def main():

    if not os.path.exists(CLOUDFLARED):
        print("No se encontro herramientas/cloudflared.exe")
        sys.exit(1)

    os.makedirs(os.path.dirname(RUTA_TUNEL), exist_ok=True)

    if os.path.exists(RUTA_TUNEL):
        os.remove(RUTA_TUNEL)

    links = {}
    procesos = [
        abrir_tunel(PUERTO_PAGINA, "pagina", links),
        abrir_tunel(PUERTO_VIDEO, "video", links)
    ]

    print("Creando los links publicos...")

    inicio = time.time()

    while len(links) < 2 and time.time() - inicio < 60:
        time.sleep(0.5)

    if len(links) < 2:
        print("No se pudieron crear los links publicos (revisa la conexion a internet).")
        for p in procesos:
            p.terminate()
        sys.exit(1)

    clave = secrets.token_hex(3).upper()

    with open(RUTA_TUNEL, "w", encoding="utf-8") as f:
        json.dump({**links, "clave": clave, "pid": os.getpid()}, f, indent=2)

    print()
    print("=" * 60)
    print(f"  Link publico de la pagina: {links['pagina']}")
    print(f"  Clave para las visitas:    {clave}")
    print("=" * 60)
    print("  Comparte el link y la clave. Cierra esta ventana para terminar.")
    print()

    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", os.path.join(RAIZ, "app.py")],
            cwd=RAIZ
        )
    finally:
        for p in procesos:
            p.terminate()
        if os.path.exists(RUTA_TUNEL):
            os.remove(RUTA_TUNEL)


if __name__ == "__main__":
    main()
