# ============================================================
# GENERACION DE RESULTADOS
# JSON (detalle por fotograma) y CSV (una fila por objeto)
# listos para el grupo PREDICCION.
# ============================================================

import os
import csv
import json
from datetime import datetime


CAMPOS_DETECCION = (
    "clase", "x1", "y1", "x2", "y2",
    "confianza_video", "confianza_capturada", "confianza_procesada",
    "variacion_confianza",
    "estado", "iou_verificacion"
)


def resultado_a_dict(resultado):

    registro = resultado["registro_video"]

    return {
        "frame_id": registro["frame_id"],
        "origen": registro["origen"],
        "timestamp": registro["timestamp"],
        "ancho": registro["ancho"],
        "alto": registro["alto"],
        "alerta_video": registro.get("alerta"),
        "calidad": resultado["calidad"],
        "resumen": resultado["resumen"],
        "objetos": [
            {
                "id": objeto["id"],
                **{c: objeto["deteccion"][c] for c in CAMPOS_DETECCION},
                "caracteristicas": objeto["caracteristicas"]
            }
            for objeto in resultado["objetos"]
        ]
    }


# Campos internos que no se exportan a otros grupos
CAMPOS_INTERNOS = ("imagenes", "decisiones", "archivo", "segundos_proceso")


def _limpiar(datos):

    return {k: v for k, v in datos.items() if k not in CAMPOS_INTERNOS}


def generar_json_dicts(frames):

    return {
        "modulo": "IMAGEN",
        "fuente": "VIDEO (SistemaSupervisionExamenes)",
        "generado": datetime.now().isoformat(timespec="seconds"),
        "total_frames": len(frames),
        "frames": [_limpiar(f) for f in frames]
    }


def filas_csv_dicts(frames):

    filas = []

    for datos in frames:

        alerta = datos.get("alerta_video") or {}

        for objeto in datos["objetos"]:

            fila = {
                "frame_id": datos["frame_id"],
                "origen": datos["origen"],
                "timestamp": datos["timestamp"],
                "alerta_objeto": alerta.get("objeto"),
                "alerta_fecha": alerta.get("fecha"),
                "objeto_id": objeto["id"]
            }

            for campo in CAMPOS_DETECCION:
                fila[campo] = objeto[campo]

            fila.update(objeto["caracteristicas"])

            filas.append(fila)

    return filas


def guardar_resultados_dicts(frames, carpeta, nombre="resultados_imagen"):

    os.makedirs(carpeta, exist_ok=True)

    ruta_json = os.path.join(carpeta, f"{nombre}.json")
    ruta_csv = os.path.join(carpeta, f"{nombre}.csv")

    with open(ruta_json, "w", encoding="utf-8") as archivo:
        json.dump(generar_json_dicts(frames), archivo, indent=2, ensure_ascii=False)

    filas = filas_csv_dicts(frames)

    if filas:

        columnas = list(dict.fromkeys(k for f in filas for k in f))

        with open(ruta_csv, "w", newline="", encoding="utf-8-sig") as archivo:
            escritor = csv.DictWriter(archivo, fieldnames=columnas)
            escritor.writeheader()
            escritor.writerows(filas)

    return ruta_json, ruta_csv


# Variantes que reciben directamente la salida de pipeline.procesar_frame

def generar_json(resultados):

    return generar_json_dicts([resultado_a_dict(r) for r in resultados])


def filas_csv(resultados):

    return filas_csv_dicts([resultado_a_dict(r) for r in resultados])


def guardar_resultados(resultados, carpeta, nombre="resultados_imagen"):

    return guardar_resultados_dicts(
        [resultado_a_dict(r) for r in resultados], carpeta, nombre
    )
