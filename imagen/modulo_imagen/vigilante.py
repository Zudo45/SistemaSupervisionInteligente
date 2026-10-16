# ============================================================
# VIGILANTE DE ALERTAS DE VIDEO
# Revisa la carpeta donde VIDEO guarda las capturas de sus alarmas
# (alertas/alerta_<objeto>_<fecha>_<hora>.jpg) y procesa
# automaticamente cada captura nueva con el pipeline de IMAGEN.
# ============================================================

import os
import re
import json
import time
import threading
import unicodedata
from datetime import datetime

import cv2

from modulo_imagen.config_imagen import (
    CARPETA_ALERTAS_VIDEO,
    CARPETA_EN_VIVO
)
from modulo_imagen.recepcion import redimensionar, EXTENSIONES_IMAGEN
from modulo_imagen.pipeline import procesar_frame
from modulo_imagen.exportacion import resultado_a_dict, guardar_resultados_dicts
from modulo_imagen.visualizacion import dibujar_video


PATRON_ALERTA = re.compile(
    r"alerta_(?P<objeto>.+)_(?P<fecha>\d{4}-\d{2}-\d{2})_(?P<hora>\d{2}-\d{2}-\d{2})",
    re.IGNORECASE
)

# Evita que dos hilos (camara de VIDEO y pagina) procesen a la vez
_candado = threading.Lock()

CARPETA_IMAGENES = os.path.join(CARPETA_EN_VIVO, "imagenes")
CARPETA_DATOS = os.path.join(CARPETA_EN_VIVO, "datos")


# ============================================================
# UTILIDADES
# ============================================================

def normalizar(texto):

    texto = unicodedata.normalize("NFD", str(texto).lower().strip())

    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def leer_alerta(nombre_archivo):
    """
    Extrae el objeto y la fecha del nombre que pone VIDEO.
    """

    base = os.path.splitext(nombre_archivo)[0]

    coincidencia = PATRON_ALERTA.fullmatch(base)

    if not coincidencia:
        return {"objeto": "desconocido", "fecha": None}

    fecha = f"{coincidencia['fecha']} {coincidencia['hora'].replace('-', ':')}"

    return {
        "objeto": normalizar(coincidencia["objeto"]),
        "fecha": fecha
    }


def archivo_listo(ruta, espera=0.3):
    """
    Evita leer una imagen que todavia se esta copiando.
    """

    try:
        tamano = os.path.getsize(ruta)
        time.sleep(espera)
        return tamano > 0 and tamano == os.path.getsize(ruta)
    except OSError:
        return False


def _ruta_datos(base):

    return os.path.join(CARPETA_DATOS, f"{base}.json")


# ============================================================
# ESTADO
# ============================================================

def alertas_en_carpeta():

    if not os.path.isdir(CARPETA_ALERTAS_VIDEO):
        return []

    archivos = [
        a for a in os.listdir(CARPETA_ALERTAS_VIDEO)
        if a.lower().endswith(EXTENSIONES_IMAGEN)
    ]

    return sorted(archivos, key=lambda a: os.path.getmtime(os.path.join(CARPETA_ALERTAS_VIDEO, a)))


def pendientes():

    return [
        a for a in alertas_en_carpeta()
        if not os.path.exists(_ruta_datos(os.path.splitext(a)[0]))
    ]


def procesadas():
    """
    Resumenes de las alertas ya procesadas, de la mas reciente a la
    mas antigua.
    """

    if not os.path.isdir(CARPETA_DATOS):
        return []

    resultados = []

    for archivo in os.listdir(CARPETA_DATOS):

        if not archivo.endswith(".json"):
            continue

        ruta = os.path.join(CARPETA_DATOS, archivo)

        with open(ruta, "r", encoding="utf-8") as f:
            datos = json.load(f)

        # Si la captura ya no esta en la carpeta de alertas, se
        # descartan sus resultados
        if not os.path.exists(os.path.join(CARPETA_ALERTAS_VIDEO, datos["archivo"])):
            os.remove(ruta)
            for imagen in datos.get("imagenes", {}).values():
                if os.path.exists(imagen):
                    os.remove(imagen)
            continue

        resultados.append(datos)

    return sorted(resultados, key=lambda r: r["procesado_en"], reverse=True)


# ============================================================
# PROCESAMIENTO
# ============================================================

def cargar_paquete(nombre_archivo, frame_id=0):

    ruta = os.path.join(CARPETA_ALERTAS_VIDEO, nombre_archivo)

    frame = cv2.imread(ruta)

    if frame is None:
        return None

    alerta = leer_alerta(nombre_archivo)

    # Si el nombre no trae la fecha (archivo renombrado), se usa la del archivo
    fecha = alerta["fecha"] or datetime.fromtimestamp(
        os.path.getmtime(ruta)
    ).strftime("%Y-%m-%d %H:%M:%S")

    return {
        "frame_id": frame_id,
        "origen": nombre_archivo,
        "timestamp": fecha,
        "alerta": alerta,
        "frame": redimensionar(frame)
    }


def procesar_alerta(nombre_archivo, opciones=None):

    ruta = os.path.join(CARPETA_ALERTAS_VIDEO, nombre_archivo)

    if not archivo_listo(ruta):
        return None

    paquete = cargar_paquete(nombre_archivo, frame_id=len(procesadas()))

    if paquete is None:
        return None

    inicio = time.time()

    resultado = procesar_frame(paquete, opciones)

    base = os.path.splitext(nombre_archivo)[0]

    os.makedirs(CARPETA_IMAGENES, exist_ok=True)
    os.makedirs(CARPETA_DATOS, exist_ok=True)

    imagenes = resultado["imagenes"]

    rutas = {
        "original": os.path.join(CARPETA_IMAGENES, f"{base}_video.jpg"),
        "procesada": os.path.join(CARPETA_IMAGENES, f"{base}_imagen.jpg"),
        "mejorada": os.path.join(CARPETA_IMAGENES, f"{base}_mejorada.jpg")
    }

    # Imagen mejorada limpia (sin recuadros): la que recibe PREDICCION
    cv2.imwrite(rutas["mejorada"], imagenes["mejorado"], [cv2.IMWRITE_JPEG_QUALITY, 92])

    cv2.imwrite(rutas["original"], dibujar_video(
        imagenes["capturado"], resultado["registro_video"]["detecciones"]
    ))
    cv2.imwrite(rutas["procesada"], dibujar_video(
        imagenes["mejorado"], resultado["detecciones_despues"]
    ))

    datos = {
        **resultado_a_dict(resultado),
        "archivo": nombre_archivo,
        "procesado_en": datetime.now().isoformat(timespec="seconds"),
        "segundos_proceso": round(time.time() - inicio, 2),
        "decisiones": resultado["decisiones"],
        "imagenes": rutas
    }

    with open(_ruta_datos(base), "w", encoding="utf-8") as f:
        json.dump(datos, f, indent=2, ensure_ascii=False)

    exportar_acumulado()

    # Evento hacia PREDICCION: imagen mejorada + datos de IMAGEN
    try:
        from integracion.prediccion import predecir_alerta
        predecir_alerta(datos)
    except Exception as error:  # noqa: BLE001
        print(f"PREDICCION no pudo procesar {nombre_archivo}: {error}")

    return datos


def revisar(opciones=None, maximo=None, bloquear=True):
    """
    Procesa las alertas nuevas. Devuelve la lista de las procesadas.
    Con bloquear=False no espera si otro hilo ya esta procesando.
    """

    if not _candado.acquire(blocking=bloquear):
        return []

    try:

        nuevas = []

        for nombre in pendientes()[:maximo]:

            datos = procesar_alerta(nombre, opciones)

            if datos:
                nuevas.append(datos)

        return nuevas

    finally:
        _candado.release()


def exportar_acumulado():
    """
    JSON y CSV con todas las alertas procesadas (para otros grupos).
    """

    datos = sorted(procesadas(), key=lambda r: r["procesado_en"])

    return guardar_resultados_dicts(datos, CARPETA_EN_VIVO, "resultados_alertas")
