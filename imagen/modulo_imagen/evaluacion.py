# ============================================================
# EVALUACION COMPARATIVA
# Se simulan malas condiciones de captura y se compara la salida
# del modelo de VIDEO (best.pt, sin cambios):
#   ANTES:   VIDEO sobre el frame capturado (sistema actual)
#   DESPUES: VIDEO sobre el frame procesado por IMAGEN
# ============================================================

import pandas as pd

from modulo_imagen.config_imagen import CLASES_NO_PERMITIDAS
from modulo_imagen.pipeline import procesar_frame


ESCENARIOS = {
    "normal": {},
    "poca_luz": {"oscuridad": 0.7},
    "ruido": {"ruido_simulado": 30},
    "desenfoque": {"desenfoque": 9},
    "luz+ruido": {"oscuridad": 0.6, "ruido_simulado": 20}
}


def _resumir(detecciones, prefijo):

    personas = [d for d in detecciones if d["clase"] == "persona"]
    objetos = [d for d in detecciones if d["clase"] in CLASES_NO_PERMITIDAS]

    def media(lista):
        if not lista:
            return 0.0
        return round(sum(d["confianza"] for d in lista) / len(lista), 4)

    return {
        f"{prefijo}_detecciones": len(detecciones),
        f"{prefijo}_personas": len(personas),
        f"{prefijo}_no_permitidos": len(objetos),
        f"{prefijo}_conf_media": media(detecciones),
        f"{prefijo}_conf_personas": media(personas)
    }


def evaluar(paquetes, escenarios=None, opciones=None, progreso=None):
    """
    paquetes: lista de fotogramas (ver recepcion.py)
    progreso: funcion opcional progreso(actual, total, texto)
    Devuelve un DataFrame con una fila por (fotograma, escenario).
    """

    escenarios = escenarios or ESCENARIOS
    paquetes = list(paquetes)

    total = len(paquetes) * len(escenarios)
    filas = []
    paso = 0

    for paquete in paquetes:

        for nombre, degradacion in escenarios.items():

            paso += 1

            if progreso:
                progreso(paso, total, f"{paquete['origen']} / {nombre}")

            resultado = procesar_frame(
                paquete,
                {**(opciones or {}), **degradacion}
            )

            comparacion = resultado["resumen"]["comparacion_video"]
            calidad = resultado["calidad"]

            fila = {
                "origen": paquete["origen"],
                "escenario": nombre,
                **_resumir(resultado["detecciones_capturado"], "antes"),
                **_resumir(resultado["detecciones_despues"], "despues"),
                "mantenidas": comparacion["mantenidas"],
                "perdidas": comparacion["perdidas"],
                "solo_tras_procesamiento": comparacion["solo_tras_procesamiento"],
                "brillo_antes": calidad["capturado"]["brillo"],
                "brillo_despues": calidad["mejorado"]["brillo"],
                "contraste_antes": calidad["capturado"]["contraste"],
                "contraste_despues": calidad["mejorado"]["contraste"],
                "nitidez_antes": calidad["capturado"]["nitidez"],
                "nitidez_despues": calidad["mejorado"]["nitidez"],
                "ruido_antes": calidad["capturado"]["ruido"],
                "ruido_despues": calidad["mejorado"]["ruido"]
            }

            if "vs_original" in calidad:
                fila.update({
                    "psnr_antes": calidad["vs_original"]["capturado"]["psnr"],
                    "psnr_despues": calidad["vs_original"]["mejorado"]["psnr"],
                    "ssim_antes": calidad["vs_original"]["capturado"]["ssim"],
                    "ssim_despues": calidad["vs_original"]["mejorado"]["ssim"]
                })

            filas.append(fila)

    return pd.DataFrame(filas)


def resumen_por_escenario(tabla):

    conteos = [
        "antes_detecciones", "despues_detecciones",
        "antes_personas", "despues_personas",
        "antes_no_permitidos", "despues_no_permitidos",
        "mantenidas", "perdidas", "solo_tras_procesamiento"
    ]

    promedios = [
        c for c in tabla.columns
        if c not in conteos and c not in ("origen", "escenario")
    ]

    agrupado = tabla.groupby("escenario", sort=False)

    return agrupado[conteos].sum().join(agrupado[promedios].mean().round(3))
