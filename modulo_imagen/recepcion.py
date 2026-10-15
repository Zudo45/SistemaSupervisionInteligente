# ============================================================
# RECEPCION DE VIDEO
# Simula la salida del grupo VIDEO: fotogramas + detecciones
# (clase, confianza, coordenadas) generadas por su modelo.
# ============================================================

import os
import sys
import json
import glob
import threading

import cv2

from modulo_imagen.config_imagen import (
    CARPETA_PROYECTO,
    LADO_MAXIMO_FRAME,
    INTERVALO_MUESTREO_SEG
)

if CARPETA_PROYECTO not in sys.path:
    sys.path.insert(0, CARPETA_PROYECTO)

# Funcion de deteccion original del grupo VIDEO (modelo best.pt)
from modulos.deteccion import detectar_objetos as _detectar_video  # noqa: E402


# La camara de VIDEO y el procesamiento de IMAGEN pueden usar el modelo
# al mismo tiempo (hilos distintos): se usa de a uno.
_candado_modelo = threading.Lock()


def detectar_objetos(frame):

    with _candado_modelo:
        return _detectar_video(frame)


EXTENSIONES_IMAGEN = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


# ============================================================
# REDIMENSIONAR
# ============================================================

def redimensionar(frame, lado_maximo=LADO_MAXIMO_FRAME):

    alto, ancho = frame.shape[:2]

    escala = lado_maximo / max(alto, ancho)

    if escala >= 1:
        return frame

    return cv2.resize(
        frame,
        None,
        fx=escala,
        fy=escala,
        interpolation=cv2.INTER_AREA
    )


# ============================================================
# FUENTES DE FOTOGRAMAS
# ============================================================

def frames_desde_carpeta(carpeta):

    rutas = sorted(
        ruta for ruta in glob.glob(os.path.join(carpeta, "*"))
        if ruta.lower().endswith(EXTENSIONES_IMAGEN)
    )

    for indice, ruta in enumerate(rutas):

        frame = cv2.imread(ruta)

        if frame is None:
            continue

        yield {
            "frame_id": indice,
            "origen": os.path.basename(ruta),
            "timestamp": float(indice),
            "frame": redimensionar(frame)
        }


def frames_desde_video(ruta, intervalo=INTERVALO_MUESTREO_SEG):

    video = cv2.VideoCapture(ruta)

    fps = video.get(cv2.CAP_PROP_FPS) or 30

    salto = max(1, int(round(fps * intervalo)))

    numero = 0
    indice = 0

    while True:

        correcto, frame = video.read()

        if not correcto:
            break

        if numero % salto == 0:

            yield {
                "frame_id": indice,
                "origen": os.path.basename(ruta),
                "timestamp": round(numero / fps, 2),
                "frame": redimensionar(frame)
            }

            indice += 1

        numero += 1

    video.release()


def frame_desde_bytes(datos, nombre):

    import numpy as np

    arreglo = np.frombuffer(datos, dtype=np.uint8)

    frame = cv2.imdecode(arreglo, cv2.IMREAD_COLOR)

    if frame is None:
        return None

    return {
        "frame_id": 0,
        "origen": nombre,
        "timestamp": 0.0,
        "frame": redimensionar(frame)
    }


# ============================================================
# RECIBIR SALIDA DE VIDEO
# ============================================================

def recibir_de_video(paquete):
    """
    Ejecuta el detector de VIDEO sobre el fotograma y devuelve
    el registro en el formato que entregaria ese grupo.
    """

    frame = paquete["frame"]

    alto, ancho = frame.shape[:2]

    # Se pasa una copia: VIDEO podria dibujar sobre el frame
    detecciones = detectar_objetos(frame.copy())

    registro = {
        "frame_id": paquete["frame_id"],
        "origen": paquete["origen"],
        "timestamp": paquete["timestamp"],
        "ancho": ancho,
        "alto": alto,
        "detecciones": [
            {**d, "confianza": round(float(d["confianza"]), 4)}
            for d in detecciones
        ]
    }

    # Datos de la alarma de VIDEO (si el fotograma es una alerta)
    if paquete.get("alerta"):
        registro["alerta"] = paquete["alerta"]

    return registro


# ============================================================
# GUARDAR / CARGAR SALIDA SIMULADA
# ============================================================

def guardar_salida_video(registros, ruta_json):

    os.makedirs(os.path.dirname(ruta_json), exist_ok=True)

    with open(ruta_json, "w", encoding="utf-8") as archivo:
        json.dump(registros, archivo, indent=2, ensure_ascii=False)


def cargar_salida_video(ruta_json):

    with open(ruta_json, "r", encoding="utf-8") as archivo:
        return json.load(archivo)
