# ============================================================
# INICIO DE LA PAGINA (con link publico)
# Lo ejecuta iniciar_app.bat. Pasos:
#   1. Descarga cloudflared (Cloudflare, sin cuenta) si falta.
#   2. Crea dos tuneles: pagina (8501) y video de la camara (8502).
#      Los links y una clave nueva se guardan en resultados/tunel.json,
#      que la pagina lee para mostrarlos en la barra lateral.
#   3. Inicia la pagina y abre el navegador en http://localhost:8501.
# Sin internet, sigue en modo local (sin link publico).
# Al cerrar la ventana se apaga todo.
# ============================================================

import os
import re
import sys
import json
import time
import secrets
import threading
import subprocess
import webbrowser
import urllib.request

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLOUDFLARED = os.path.join(RAIZ, "herramientas", "cloudflared.exe")
URL_CLOUDFLARED = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download/"
    "cloudflared-windows-amd64.exe"
)
RUTA_TUNEL = os.path.join(RAIZ, "resultados", "tunel.json")

PUERTO_PAGINA = 8501
PUERTO_VIDEO = 8502
ESPERA_TUNEL_SEG = 45

PATRON_LINK = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def descargar_cloudflared():

    if os.path.exists(CLOUDFLARED):
        return True

    print("Descargando cloudflared (programa oficial de Cloudflare)...")

    try:
        os.makedirs(os.path.dirname(CLOUDFLARED), exist_ok=True)
        urllib.request.urlretrieve(URL_CLOUDFLARED, CLOUDFLARED + ".tmp")
        os.replace(CLOUDFLARED + ".tmp", CLOUDFLARED)
        return True
    except Exception as error:  # noqa: BLE001
        print(f"No se pudo descargar cloudflared: {error}")
        return False


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


def crear_link_publico():
    """
    Devuelve (procesos de los tuneles, datos del link) o ([], None).
    """

    if not descargar_cloudflared():
        return [], None

    print("Creando el link publico...")

    links = {}
    procesos = [
        abrir_tunel(PUERTO_PAGINA, "pagina", links),
        abrir_tunel(PUERTO_VIDEO, "video", links)
    ]

    inicio = time.time()

    while len(links) < 2 and time.time() - inicio < ESPERA_TUNEL_SEG:
        time.sleep(0.5)

    if len(links) < 2:
        for p in procesos:
            p.terminate()
        return [], None

    return procesos, {**links, "clave": secrets.token_hex(3).upper(), "pid": os.getpid()}


def abrir_navegador_cuando_este_lista():

    def esperar():
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://localhost:{PUERTO_PAGINA}", timeout=1)
                webbrowser.open(f"http://localhost:{PUERTO_PAGINA}")
                return
            except Exception:  # noqa: BLE001
                time.sleep(1)

    threading.Thread(target=esperar, daemon=True).start()


def main():

    os.makedirs(os.path.dirname(RUTA_TUNEL), exist_ok=True)

    # Un link de un inicio anterior ya no existe
    if os.path.exists(RUTA_TUNEL):
        os.remove(RUTA_TUNEL)

    procesos, tunel = crear_link_publico()

    print()
    print("=" * 64)

    if tunel:
        with open(RUTA_TUNEL, "w", encoding="utf-8") as f:
            json.dump(tunel, f, indent=2)

        print(f"  Link publico (para los demas): {tunel['pagina']}")
        print(f"  Clave para las visitas:        {tunel['clave']}")
    else:
        print("  Sin link publico (sin internet?). La pagina funciona en modo local.")

    print(f"  En esta laptop:                http://localhost:{PUERTO_PAGINA}")
    print("=" * 64)
    print("  NO cierres esta ventana: al cerrarla se apaga la pagina y el link.")
    print("  Puedes minimizarla.")
    print()

    abrir_navegador_cuando_este_lista()

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
