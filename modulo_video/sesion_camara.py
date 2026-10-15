# ============================================================
# MODULO VIDEO EN LA PAGINA WEB
# Ejecuta el sistema del grupo VIDEO dentro de la pagina: lee la
# camara de la laptop donde corre la pagina, detecta con el modelo
# de VIDEO y aplica la misma alarma que principal.py (objeto no
# permitido visible TIEMPO_ALERTA segundos).
#
# Se reutilizan las funciones originales de VIDEO:
#   modulos.deteccion  -> detectar_objetos, dibujar_detecciones
#   modulos.registro   -> guardar_alerta (captura + sonido)
#   principal          -> es_no_permitido, normalizar
#
# Cuando suena la alarma se dispara el evento "alerta": la captura
# se guarda en alertas/ y se encola para que IMAGEN la procese.
# ============================================================

import os
import queue
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from modulo_imagen.config_imagen import CARPETA_PROYECTO  # agrega el proyecto al path
from modulo_imagen.recepcion import detectar_objetos
from modulo_imagen import vigilante

from configuracion.config import CAMARA, TIEMPO_ALERTA
from modulos.deteccion import dibujar_detecciones
from modulos.registro import guardar_alerta
from principal import es_no_permitido, normalizar


def abrir_camara(indice):
    """
    En Windows, DirectShow abre la camara en menos de un segundo
    (el metodo por defecto tarda varios). Si falla, se usa el normal.
    """

    if os.name == "nt":
        camara = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
        if camara.isOpened():
            return camara
        camara.release()

    return cv2.VideoCapture(indice)


class SesionCamara:
    """
    Una sola sesion por pagina: todos los que abren la pagina ven la
    misma camara. Solo la laptop donde corre la pagina la controla.
    """

    def __init__(self):

        self.activa = False
        self.error = None
        self.inicio_sesion = None

        self.fotograma_jpeg = None
        self.fps = 0.0
        self.detecciones = []
        self.tiempo_mayor = 0.0
        self.alerta_activa = False

        self.alertas = deque(maxlen=50)      # alertas de esta sesion
        self.registro = deque(maxlen=30)     # mensajes para la pagina

        self._hilo = None
        self._detener = threading.Event()

        # La primera deteccion de YOLO tarda varios segundos: se hace
        # al cargar la pagina para que la camara arranque enseguida
        self.modelo_listo = threading.Event()
        threading.Thread(target=self._precalentar, daemon=True).start()

        # Eventos "alerta" hacia IMAGEN
        self._eventos = queue.Queue()
        threading.Thread(target=self._atender_eventos, daemon=True).start()

    # --------------------------------------------------------
    # CONTROL
    # --------------------------------------------------------

    def iniciar(self):

        if self.activa:
            return

        if self._hilo and self._hilo.is_alive():
            self._hilo.join(timeout=3)

        # Se marca activa de inmediato (abrir la camara tarda unos segundos)
        self.activa = True
        self.error = None
        self._detener.clear()
        self._hilo = threading.Thread(target=self._bucle, daemon=True)
        self._hilo.start()

    def detener(self):
        """
        No espera a que la camara se cierre: el hilo termina por su cuenta.
        """

        self._detener.set()
        self.activa = False
        self.fotograma_jpeg = None

    def _precalentar(self):

        detectar_objetos(np.zeros((480, 640, 3), dtype=np.uint8))
        self.modelo_listo.set()
        self._anotar("Modelo de VIDEO cargado")

    def _anotar(self, mensaje):

        self.registro.appendleft(f"{datetime.now():%H:%M:%S}  {mensaje}")

    # --------------------------------------------------------
    # BUCLE DE CAMARA (misma logica que principal.py)
    # --------------------------------------------------------

    def _bucle(self):

        self.modelo_listo.wait()

        if self._detener.is_set():
            return

        camara = abrir_camara(CAMARA)

        if not camara.isOpened():
            self.error = "No se pudo abrir la camara (puede estar en uso por otro programa)."
            self._anotar(self.error)
            self.activa = False
            return

        self.inicio_sesion = time.time()
        self._anotar("Camara iniciada")

        inicio_deteccion = {}
        alerta_generada = {}
        ultimo = time.time()

        try:

            while not self._detener.is_set():

                correcto, frame = camara.read()

                if not correcto:
                    self.error = "Se perdio la senal de la camara."
                    self._anotar(self.error)
                    break

                detecciones = detectar_objetos(frame)
                ahora = time.time()
                objetos_actuales = set()

                for deteccion in detecciones:

                    clase = deteccion["clase"]

                    if not es_no_permitido(clase):
                        continue

                    nombre = normalizar(clase)
                    objetos_actuales.add(nombre)

                    if nombre not in inicio_deteccion:
                        inicio_deteccion[nombre] = ahora
                        alerta_generada[nombre] = False

                    if (ahora - inicio_deteccion[nombre] >= TIEMPO_ALERTA
                            and not alerta_generada[nombre]):

                        # Captura limpia (antes de dibujar), como en principal.py
                        ruta = guardar_alerta(frame, clase)
                        alerta_generada[nombre] = True
                        self._disparar_alerta(ruta, clase)

                # Reiniciar el contador de los objetos que ya no se ven
                for nombre in list(inicio_deteccion):
                    if nombre not in objetos_actuales:
                        del inicio_deteccion[nombre]
                        alerta_generada.pop(nombre, None)

                self.tiempo_mayor = max(
                    (ahora - inicio_deteccion[n] for n in objetos_actuales),
                    default=0.0
                )
                self.alerta_activa = any(alerta_generada.get(n) for n in objetos_actuales)
                self.detecciones = detecciones

                # Fotograma para la pagina, con los dibujos de VIDEO
                vista = frame.copy()

                cv2.putText(vista, f"Tiempo: {int(self.tiempo_mayor)}s / {TIEMPO_ALERTA}s",
                            (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

                if self.alerta_activa:
                    cv2.putText(vista, "ALERTA: MATERIAL NO PERMITIDO", (20, 85),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

                vista = dibujar_detecciones(vista, detecciones)

                correcto, jpeg = cv2.imencode(".jpg", vista, [cv2.IMWRITE_JPEG_QUALITY, 80])

                if correcto:
                    self.fotograma_jpeg = jpeg.tobytes()

                self.fps = 0.8 * self.fps + 0.2 / max(ahora - ultimo, 1e-3)
                ultimo = ahora

        finally:

            camara.release()

            # Solo si sigue siendo la sesion vigente (no una nueva)
            if self._hilo is threading.current_thread():
                self.activa = False
                self.fotograma_jpeg = None
            self.tiempo_mayor = 0.0
            self.alerta_activa = False
            self._anotar("Camara detenida")

    # --------------------------------------------------------
    # EVENTO "ALERTA" -> IMAGEN
    # --------------------------------------------------------

    def _disparar_alerta(self, ruta, clase):

        nombre = os.path.basename(ruta)

        self.alertas.appendleft({
            "archivo": nombre,
            "objeto": clase,
            "hora": datetime.now().strftime("%H:%M:%S")
        })

        self._anotar(f"ALERTA: {clase} -> captura enviada a IMAGEN")
        self._eventos.put(nombre)

    def _atender_eventos(self):

        while True:

            nombre = self._eventos.get()

            try:
                vigilante.revisar()
                self._anotar(f"IMAGEN proceso {nombre}")
            except Exception as error:  # noqa: BLE001
                self._anotar(f"IMAGEN no pudo procesar {nombre}: {error}")
