# ============================================================
# CONEXION IMAGEN -> PREDICCION
# Cuando IMAGEN termina de procesar una alerta, se envian a
# PREDICCION la imagen mejorada (limpia) y los datos de IMAGEN:
#   - integrar_fuentes(): motor de reglas de PREDICCION, que recibe
#     el JSON de IMAGEN tal cual (su adaptador ya lo admite).
#   - procesar_frame(): analisis de PREDICCION (rostro, emociones,
#     atencion, proyeccion) sobre cada estudiante, recortando cada
#     persona detectada por VIDEO en la imagen mejorada.
#
# El codigo de PREDICCION (carpeta backend/) se usa sin modificar.
# ============================================================

import os
import sys
import json
import base64
import threading
from datetime import datetime

import cv2

from modulo_imagen.config_imagen import (
    CARPETA_RESULTADOS,
    CARPETA_ALERTAS_VIDEO,
    CLASES_NO_PERMITIDAS,
    MARGEN_RECORTE
)
from modulo_imagen.recepcion import detectar_objetos

CARPETA_MODULO = os.path.dirname(os.path.abspath(__file__))

if CARPETA_MODULO not in sys.path:
    sys.path.insert(0, CARPETA_MODULO)

from backend.modulos import vision_service  # noqa: E402
from backend.modulos.prediccion.integracion import integrar_fuentes  # noqa: E402

from configuracion.config import TIEMPO_ALERTA  # noqa: E402


CARPETA_PREDICCIONES = os.path.join(CARPETA_RESULTADOS, "prediccion")

_candado = threading.Lock()


# ------------------------------------------------------------
# PREDICCION carga su propia copia del modelo de VIDEO y fija los
# hilos de PyTorch para todo el programa. Aqui se le hace usar el
# mismo modelo de VIDEO que ya usa la pagina (mismas clases y salida).
# ------------------------------------------------------------

def _detectar_con_modelo_compartido(imagen):

    return [
        {**d, "confianza": round(float(d["confianza"]), 2)}
        for d in detectar_objetos(imagen)
    ]


vision_service.detectar_objetos_en_frame = _detectar_con_modelo_compartido
vision_service._modelo_yolo = "modelo de VIDEO compartido"


# ============================================================
# UTILIDADES
# ============================================================

def _ruta_prediccion(archivo_alerta):

    return os.path.join(CARPETA_PREDICCIONES, os.path.splitext(archivo_alerta)[0] + ".json")


def _a_base64(imagen):

    return base64.b64encode(cv2.imencode(".jpg", imagen)[1].tobytes()).decode()


def _recortar(imagen, caja, margen=MARGEN_RECORTE):

    alto, ancho = imagen.shape[:2]

    mx = int((caja["x2"] - caja["x1"]) * margen)
    my = int((caja["y2"] - caja["y1"]) * margen)

    x1, y1 = max(0, caja["x1"] - mx), max(0, caja["y1"] - my)
    x2, y2 = min(ancho, caja["x2"] + mx), min(alto, caja["y2"] + my)

    return imagen[y1:y2, x1:x2]


# ============================================================
# EVENTO: ALERTA PROCESADA POR IMAGEN
# ============================================================

def predecir_alerta(datos):
    """
    datos: resultado de IMAGEN para una alerta (lo que guarda el
    vigilante), con la ruta de la imagen mejorada.
    """

    with _candado:

        imagen = cv2.imread(datos["imagenes"]["mejorada"])

        if imagen is None:
            raise FileNotFoundError("No se encontro la imagen mejorada de IMAGEN")

        # 1. Motor de reglas de PREDICCION con el JSON de IMAGEN
        try:
            motor = integrar_fuentes({"imagen": datos})
            error_motor = None
        except Exception as error:  # noqa: BLE001
            motor, error_motor = None, str(error)

        # 2. Analisis por estudiante (cada persona detectada por VIDEO)
        objetos = datos["objetos"]
        personas = [o for o in objetos if o["clase"] == "persona"]

        # Objetos no permitidos asociados a cada persona (indice de IMAGEN)
        asociados = {i: [] for i in range(len(personas))}

        for o in objetos:
            indice = o["caracteristicas"].get("persona_asociada")
            if o["clase"] in CLASES_NO_PERMITIDAS and indice is not None and indice in asociados:
                asociados[indice].append(o["clase"])

        # Numeracion de izquierda a derecha, como se ven en la imagen
        orden = sorted(range(len(personas)), key=lambda i: personas[i]["x1"])

        estudiantes = []

        for numero, i in enumerate(orden, start=1):

            persona = personas[i]
            recorte = _recortar(imagen, persona)

            analisis = vision_service.procesar_frame(
                _a_base64(recorte),
                modo="captura",
                aplicar_mejora=False,          # IMAGEN ya mejoro la imagen
                segundos_distraccion=float(TIEMPO_ALERTA) if asociados[i] else 0.0
            )
            analisis.pop("imagen_mejorada_hd", None)

            estudiantes.append({
                "numero": numero,
                "etiqueta": f"Estudiante {numero}",
                "confianza_persona": persona["confianza_video"],
                "caja": {k: persona[k] for k in ("x1", "y1", "x2", "y2")},
                "objetos_no_permitidos": asociados[i],
                "analisis": analisis
            })

        # Sin personas detectadas: se analiza la imagen completa
        if not estudiantes:

            analisis = vision_service.procesar_frame(
                _a_base64(imagen), modo="captura", aplicar_mejora=False,
                segundos_distraccion=float(TIEMPO_ALERTA)
            )
            analisis.pop("imagen_mejorada_hd", None)

            estudiantes.append({
                "numero": 1,
                "etiqueta": "Imagen completa",
                "confianza_persona": None,
                "caja": None,
                "objetos_no_permitidos": [
                    o["clase"] for o in objetos if o["clase"] in CLASES_NO_PERMITIDAS
                ],
                "analisis": analisis
            })

        resultado = {
            "archivo": datos["archivo"],
            "alerta_video": datos.get("alerta_video"),
            "timestamp": datos.get("timestamp"),
            "imagen_mejorada": datos["imagenes"]["mejorada"],
            "generado": datetime.now().isoformat(timespec="seconds"),
            "motor": motor,
            "error_motor": error_motor,
            "estudiantes": estudiantes
        }

        os.makedirs(CARPETA_PREDICCIONES, exist_ok=True)

        with open(_ruta_prediccion(datos["archivo"]), "w", encoding="utf-8") as f:
            json.dump(resultado, f, indent=2, ensure_ascii=False)

        return resultado


# ============================================================
# CONSULTA
# ============================================================

def listar_predicciones():
    """
    Predicciones de las alertas que siguen en la carpeta, de la mas
    antigua a la mas reciente.
    """

    if not os.path.isdir(CARPETA_PREDICCIONES):
        return []

    lista = []

    for nombre in os.listdir(CARPETA_PREDICCIONES):

        if not nombre.endswith(".json"):
            continue

        ruta = os.path.join(CARPETA_PREDICCIONES, nombre)

        with open(ruta, "r", encoding="utf-8") as f:
            datos = json.load(f)

        if not os.path.exists(os.path.join(CARPETA_ALERTAS_VIDEO, datos["archivo"])):
            os.remove(ruta)
            continue

        lista.append(datos)

    return sorted(lista, key=lambda d: (d.get("timestamp") or "", d["generado"]))


def completar_pendientes():
    """
    Genera la prediccion de las alertas que IMAGEN ya proceso pero que
    aun no la tienen (por ejemplo, procesadas antes de esta conexion).
    """

    from modulo_imagen import vigilante

    for datos in vigilante.procesadas():

        if os.path.exists(_ruta_prediccion(datos["archivo"])):
            continue

        mejorada = datos.get("imagenes", {}).get("mejorada")

        if mejorada and os.path.exists(mejorada):
            predecir_alerta(datos)
        else:
            # Procesada con una version anterior: IMAGEN la vuelve a procesar
            os.remove(vigilante._ruta_datos(os.path.splitext(datos["archivo"])[0]))
