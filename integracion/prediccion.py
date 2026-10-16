# ============================================================
# CONEXION IMAGEN -> PREDICCION
# Cuando IMAGEN termina de procesar una alerta, se envian a
# PREDICCION la imagen mejorada (limpia) y los datos de IMAGEN:
#   - integrar_fuentes(): motor de reglas de PREDICCION, que recibe
#     el JSON de IMAGEN tal cual (su adaptador ya lo admite).
#   - procesar_captura_imagen(): modelos entrenados de PREDICCION
#     (alumnos, 7 emociones, causas, consecuencias y riesgo).
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
    CLASES_NO_PERMITIDAS
)
from modulo_imagen.recepcion import _candado_modelo, _video

from modulo_imagen.config_imagen import CARPETA_PREDICCION  # noqa: E402

# El codigo de PREDICCION (prediccion/backend) se importa como "backend"
if CARPETA_PREDICCION not in sys.path:
    sys.path.append(CARPETA_PREDICCION)

from backend.entrenamiento_prediccion import predictor  # noqa: E402
from backend.modulos.prediccion.integracion import integrar_fuentes  # noqa: E402


CARPETA_PREDICCIONES = os.path.join(CARPETA_RESULTADOS, "prediccion")

# Acciones que PREDICCION entiende para cada objeto de VIDEO
ACCIONES = {
    "telefono": "uso_celular",
    "audifonos": "uso_audifonos"
}

_candado = threading.Lock()


# ------------------------------------------------------------
# PREDICCION carga su propia copia del modelo de VIDEO (best.pt).
# Se le pasa el mismo modelo que ya usa la pagina, protegido con el
# mismo candado, para no cargarlo dos veces ni usarlo a la vez.
# ------------------------------------------------------------

class _ModeloCompartido:

    def __init__(self, modelo):
        self._modelo = modelo

    def __call__(self, *args, **kwargs):
        with _candado_modelo:
            return self._modelo(*args, **kwargs)


predictor._YOLO_BEST = _ModeloCompartido(_video.modelo_personalizado)


# ============================================================
# UTILIDADES
# ============================================================

def _ruta_prediccion(archivo_alerta):

    return os.path.join(CARPETA_PREDICCIONES, os.path.splitext(archivo_alerta)[0] + ".json")


def _a_base64(imagen):

    return base64.b64encode(cv2.imencode(".jpg", imagen)[1].tobytes()).decode()


def _caja_en_pixeles(caja_relativa, ancho, alto):

    return {
        "x1": int(caja_relativa["x"] * ancho),
        "y1": int(caja_relativa["y"] * alto),
        "x2": int((caja_relativa["x"] + caja_relativa["w"]) * ancho),
        "y2": int((caja_relativa["y"] + caja_relativa["h"]) * alto)
    }


def _dentro(objeto, caja, margen=0.15):
    """
    El centro del objeto cae dentro de la caja del alumno (ampliada).
    """

    mx = (caja["x2"] - caja["x1"]) * margen
    my = (caja["y2"] - caja["y1"]) * margen
    cx = (objeto["x1"] + objeto["x2"]) / 2
    cy = (objeto["y1"] + objeto["y2"]) / 2

    return caja["x1"] - mx <= cx <= caja["x2"] + mx and caja["y1"] - my <= cy <= caja["y2"] + my


def _distancia(objeto, caja):
    """
    Distancia del centro del objeto al centro de la cabeza/torso del
    alumno (parte superior de su recuadro).
    """

    cx = (objeto["x1"] + objeto["x2"]) / 2
    cy = (objeto["y1"] + objeto["y2"]) / 2
    px = (caja["x1"] + caja["x2"]) / 2
    py = caja["y1"] + (caja["y2"] - caja["y1"]) * 0.35

    return ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5


def _recortar(imagen, caja, margen=0.15):

    alto, ancho = imagen.shape[:2]
    mx = int((caja["x2"] - caja["x1"]) * margen)
    my = int((caja["y2"] - caja["y1"]) * margen)

    x1, y1 = max(0, caja["x1"] - mx), max(0, caja["y1"] - my)
    x2, y2 = min(ancho, caja["x2"] + mx), min(alto, caja["y2"] + my)

    return imagen[y1:y2, x1:x2], (x1, y1)


# Campos de analisis que se toman del recorte con contexto
CAMPOS_ANALISIS = (
    "estado", "accionDetectada", "estadoClase", "atencion", "tipoAtencion",
    "emociones", "causas", "consecuencias", "score", "nivelRiesgo",
    "esRiesgoAlto", "alertaExamen", "narrativa"
)


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

        alto, ancho = imagen.shape[:2]

        # 1. Motor de reglas de PREDICCION con el JSON de IMAGEN
        try:
            motor = integrar_fuentes({"imagen": datos})
            error_motor = None
        except Exception as error:  # noqa: BLE001
            motor, error_motor = None, str(error)

        # 2. Modelos entrenados de PREDICCION sobre la imagen completa:
        #    detectan a todos los alumnos
        captura = predictor.procesar_captura_imagen(_a_base64(imagen), None)

        # 3. Objetos no permitidos que VIDEO detecto e IMAGEN verifico
        objetos = [
            o for o in datos["objetos"]
            if o["clase"] in CLASES_NO_PERMITIDAS
            and o["estado"] in ("confirmada", "recuperada")
        ]

        # 4. Cada objeto se asigna a UN alumno: el mas cercano entre los
        #    que lo contienen (en un aula los recuadros se superponen).
        #    Solo ese alumno se vuelve a analizar con ese contexto.
        estudiantes = captura.get("estudiantes", [])

        for estudiante in estudiantes:
            estudiante["caja"] = _caja_en_pixeles(estudiante["box"], ancho, alto)

        asignados = {id(e): [] for e in estudiantes}

        for o in objetos:

            candidatos = [e for e in estudiantes if _dentro(o, e["caja"])]

            if candidatos:
                elegido = min(candidatos, key=lambda e: _distancia(o, e["caja"]))
                asignados[id(elegido)].append(o)

        for estudiante in estudiantes:

            caja = estudiante["caja"]
            suyos = asignados[id(estudiante)]
            estudiante["objetos_no_permitidos"] = sorted({o["clase"] for o in suyos})

            if not suyos:
                continue

            recorte, (ox, oy) = _recortar(imagen, caja)

            contexto = {
                "objetos": estudiante["objetos_no_permitidos"],
                "acciones_detectadas": sorted({
                    ACCIONES[c] for c in estudiante["objetos_no_permitidos"] if c in ACCIONES
                })
            }

            individual = predictor.procesar_captura_imagen(_a_base64(recorte), contexto)

            if individual.get("estudiantes"):

                analisis = individual["estudiantes"][0]

                for campo in CAMPOS_ANALISIS:
                    if campo in analisis:
                        estudiante[campo] = analisis[campo]

                # La narrativa del recorte se refiere al alumno como "Estudiante 1"
                if isinstance(estudiante.get("narrativa"), str):
                    estudiante["narrativa"] = estudiante["narrativa"].replace(
                        "Estudiante 1", estudiante["nombre"]
                    )

            # Objeto (de VIDEO) con mayor confianza, para dibujarlo
            principal = max(suyos, key=lambda o: o["confianza_procesada"] or o["confianza_video"])
            estudiante["distractor"] = True
            estudiante["objeto_caja"] = {
                "clase": principal["clase"],
                "confianza": principal["confianza_procesada"] or principal["confianza_video"],
                **{k: principal[k] for k in ("x1", "y1", "x2", "y2")}
            }

        resultado = {
            "archivo": datos["archivo"],
            "alerta_video": datos.get("alerta_video"),
            "timestamp": datos.get("timestamp"),
            "imagen_mejorada": datos["imagenes"]["mejorada"],
            "generado": datetime.now().isoformat(timespec="seconds"),
            "motor": motor,
            "error_motor": error_motor,
            "captura": {k: v for k, v in captura.items() if k != "estudiantes"},
            "estudiantes": captura.get("estudiantes", [])
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

        alerta_existe = os.path.exists(os.path.join(CARPETA_ALERTAS_VIDEO, datos["archivo"]))

        # Alertas borradas o predicciones con el formato anterior
        if not alerta_existe or "captura" not in datos:
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

    listar_predicciones()  # descarta predicciones obsoletas

    for datos in vigilante.procesadas():

        if os.path.exists(_ruta_prediccion(datos["archivo"])):
            continue

        mejorada = datos.get("imagenes", {}).get("mejorada")

        if mejorada and os.path.exists(mejorada):
            predecir_alerta(datos)
        else:
            # Procesada con una version anterior: IMAGEN la vuelve a procesar
            os.remove(vigilante._ruta_datos(os.path.splitext(datos["archivo"])[0]))
