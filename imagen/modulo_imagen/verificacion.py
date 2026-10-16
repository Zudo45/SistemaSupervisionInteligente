# ============================================================
# VERIFICACION DE DETECCIONES
# IMAGEN no detecta: mide si su procesamiento ayuda a VIDEO.
#   1. Se toman las detecciones de VIDEO sobre el frame capturado.
#   2. Se ejecuta el MISMO modelo de VIDEO (best.pt, sin cambios)
#      sobre el frame procesado por IMAGEN.
#   3. Se emparejan ambas salidas con las detecciones originales de
#      VIDEO y se compara la confianza (ver verificar()).
# ============================================================

from modulo_imagen.config_imagen import (
    GRUPO_CLASE,
    IOU_COINCIDENCIA
)
from modulo_imagen.recepcion import detectar_objetos
from modulo_imagen.caracteristicas import iou


def grupo(clase):

    return GRUPO_CLASE.get(clase, clase)


def emparejar(antes, despues):
    """
    Empareja detecciones del mismo grupo de clase por IoU (greedy,
    de mayor a menor IoU). Devuelve {indice_antes: indice_despues}.
    """

    pares = sorted(
        (
            (iou(a, d), i, j)
            for i, a in enumerate(antes)
            for j, d in enumerate(despues)
            if grupo(a["clase"]) == grupo(d["clase"])
        ),
        reverse=True
    )

    asignados = {}
    usados = set()

    for valor, i, j in pares:

        if valor < IOU_COINCIDENCIA:
            break

        if i in asignados or j in usados:
            continue

        asignados[i] = j
        usados.add(j)

    return asignados


def verificar(frame_procesado, referencia, detecciones_capturado):
    """
    frame_procesado: frame tras preprocesamiento y mejora
    referencia: detecciones originales de VIDEO (lo que entrego VIDEO)
    detecciones_capturado: salida de VIDEO sobre el frame que recibe
                           IMAGEN (igual a `referencia` si no hay
                           simulacion de condiciones de captura)

    Para cada objeto de la referencia se busca si VIDEO lo sigue viendo
    en el frame recibido (antes) y en el frame procesado (despues):
      confirmada:    lo ve antes y despues
      no_confirmada: lo ve antes, pero no despues del procesamiento
      recuperada:    no lo ve antes, pero si despues del procesamiento
      no_detectada:  no lo ve ni antes ni despues

    Devuelve (verificadas, detecciones_despues).
    """

    detecciones_despues = [
        {**d, "confianza": round(float(d["confianza"]), 4)}
        for d in detectar_objetos(frame_procesado.copy())
    ]

    pares_antes = emparejar(referencia, detecciones_capturado)
    pares_despues = emparejar(referencia, detecciones_despues)

    verificadas = []

    for i, original in enumerate(referencia):

        antes = detecciones_capturado[pares_antes[i]]["confianza"] if i in pares_antes else 0.0
        despues = detecciones_despues[pares_despues[i]] if i in pares_despues else None
        conf_despues = despues["confianza"] if despues else 0.0

        if antes and despues:
            estado = "confirmada"
        elif antes:
            estado = "no_confirmada"
        elif despues:
            estado = "recuperada"
        else:
            estado = "no_detectada"

        verificadas.append({
            **original,
            "confianza_video": original["confianza"],
            "confianza_capturada": antes,
            "confianza_procesada": conf_despues,
            "variacion_confianza": round(conf_despues - antes, 4),
            "estado": estado,
            "iou_verificacion": round(iou(original, despues), 3) if despues else 0.0
        })

    return verificadas, detecciones_despues


def comparar_salidas(antes, despues):
    """
    Metricas agregadas de la salida de VIDEO antes y despues del
    procesamiento de IMAGEN.
    """

    def media(lista):
        return round(sum(d["confianza"] for d in lista) / len(lista), 4) if lista else 0.0

    pares = emparejar(antes, despues)

    emparejadas_antes = [antes[i] for i in pares]
    emparejadas_despues = [despues[j] for j in pares.values()]

    return {
        "detecciones_antes": len(antes),
        "detecciones_despues": len(despues),
        "confianza_media_antes": media(antes),
        "confianza_media_despues": media(despues),
        "mantenidas": len(pares),
        "perdidas": len(antes) - len(pares),
        "solo_tras_procesamiento": len(despues) - len(pares),
        "variacion_confianza_mantenidas": round(
            media(emparejadas_despues) - media(emparejadas_antes), 4
        )
    }
