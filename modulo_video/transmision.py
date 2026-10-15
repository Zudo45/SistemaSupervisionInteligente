# ============================================================
# TRANSMISION DE LA CAMARA (MJPEG)
# Servidor HTTP minimo que envia los fotogramas de la sesion de
# camara como un flujo MJPEG (como una camara IP). El navegador lo
# reproduce de forma nativa en una etiqueta <img>, sin recargar la
# pagina en cada fotograma.
# ============================================================

import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np


PUERTO_VIDEO = 8502
FPS_TRANSMISION = 10
FRONTERA = "fotograma"


def cartel(texto, detalle=""):
    """
    Fotograma de relleno cuando la camara no esta transmitiendo.
    """

    imagen = np.full((480, 640, 3), (32, 18, 11), dtype=np.uint8)

    for i, (linea, escala) in enumerate(((texto, 0.9), (detalle, 0.6))):

        if not linea:
            continue

        (ancho, _), _ = cv2.getTextSize(linea, cv2.FONT_HERSHEY_SIMPLEX, escala, 2)
        cv2.putText(imagen, linea, ((640 - ancho) // 2, 230 + i * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, escala, (200, 200, 200), 2, cv2.LINE_AA)

    return cv2.imencode(".jpg", imagen)[1].tobytes()


class ServidorVideo:

    def __init__(self, sesion, puerto=PUERTO_VIDEO):

        self.sesion = sesion
        self.puerto = None

        carteles = {
            "apagada": cartel("Camara apagada", "Pulsa Iniciar camara"),
            "modelo": cartel("Iniciando...", "Cargando el modelo de VIDEO"),
            "abriendo": cartel("Iniciando...", "Abriendo la camara")
        }

        def fotograma_actual():

            if sesion.fotograma_jpeg:
                return sesion.fotograma_jpeg
            if not sesion.activa:
                return carteles["apagada"]
            if not sesion.modelo_listo.is_set():
                return carteles["modelo"]
            return carteles["abriendo"]

        class Manejador(BaseHTTPRequestHandler):

            def log_message(self, *args):
                pass

            def do_GET(self):

                if not self.path.startswith("/video.mjpg"):
                    self.send_response(404)
                    self.end_headers()
                    return

                self.send_response(200)
                self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={FRONTERA}")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                try:
                    while True:
                        jpeg = fotograma_actual()
                        self.wfile.write(
                            f"--{FRONTERA}\r\nContent-Type: image/jpeg\r\n"
                            f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                        )
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        time.sleep(1 / FPS_TRANSMISION)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

        try:
            servidor = ThreadingHTTPServer(("0.0.0.0", puerto), Manejador)
            servidor.daemon_threads = True
        except OSError:
            return

        self.puerto = puerto
        threading.Thread(target=servidor.serve_forever, daemon=True).start()
