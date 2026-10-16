# ============================================================
# SEGMENTACION
# Separa cada objeto detectado por VIDEO del fondo dentro de
# su region (caja delimitadora).
# ============================================================

import cv2
import numpy as np

from modulo_imagen.config_imagen import MARGEN_RECORTE


# Lado maximo al que se reduce una region antes de GrabCut
# (GrabCut es costoso en regiones grandes)
LADO_MAXIMO_GRABCUT = 400


# ============================================================
# RECORTE DE REGIONES
# ============================================================

def caja_con_margen(deteccion, ancho, alto, margen=MARGEN_RECORTE):

    x1, y1 = deteccion["x1"], deteccion["y1"]
    x2, y2 = deteccion["x2"], deteccion["y2"]

    mx = int((x2 - x1) * margen)
    my = int((y2 - y1) * margen)

    return (
        max(0, x1 - mx),
        max(0, y1 - my),
        min(ancho, x2 + mx),
        min(alto, y2 + my)
    )


def recortar(imagen, deteccion, margen=MARGEN_RECORTE):
    """
    Devuelve el recorte y su desplazamiento (ox, oy) dentro
    de la imagen completa.
    """

    alto, ancho = imagen.shape[:2]

    x1, y1, x2, y2 = caja_con_margen(deteccion, ancho, alto, margen)

    return imagen[y1:y2, x1:x2].copy(), (x1, y1)


# ============================================================
# METODOS DE SEGMENTACION
# ============================================================

def segmentar_otsu(region):
    """
    Umbral de Otsu sobre la imagen en gris suavizada. Se elige la
    polaridad cuyo componente toca menos el borde de la region
    (el fondo suele tocar los bordes).
    """

    gris = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (5, 5), 0)

    _, mascara = cv2.threshold(
        gris, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    def proporcion_borde(m):
        borde = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]])
        return (borde > 0).mean()

    if proporcion_borde(mascara) > proporcion_borde(255 - mascara):
        mascara = 255 - mascara

    return mascara


def segmentar_grabcut(region, rect, iteraciones=3):
    """
    GrabCut inicializado con el rectangulo de la deteccion:
    todo lo de fuera del rectangulo es fondo seguro.
    """

    alto, ancho = region.shape[:2]

    escala = min(1.0, LADO_MAXIMO_GRABCUT / max(alto, ancho))

    if escala < 1.0:
        pequena = cv2.resize(region, None, fx=escala, fy=escala,
                             interpolation=cv2.INTER_AREA)
        rect = tuple(int(v * escala) for v in rect)
    else:
        pequena = region

    x, y, w, h = rect

    if w < 5 or h < 5:
        return segmentar_otsu(region)

    mascara = np.zeros(pequena.shape[:2], np.uint8)
    fondo = np.zeros((1, 65), np.float64)
    frente = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(pequena, mascara, (x, y, w, h), fondo, frente,
                    iteraciones, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return segmentar_otsu(region)

    binaria = np.where(
        (mascara == cv2.GC_FGD) | (mascara == cv2.GC_PR_FGD),
        255, 0
    ).astype(np.uint8)

    if escala < 1.0:
        binaria = cv2.resize(binaria, (ancho, alto),
                             interpolation=cv2.INTER_NEAREST)

    return binaria


def limpiar_mascara(mascara):
    """
    Apertura + cierre morfologico y se conserva solo el
    componente conexo mas grande.
    """

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)

    contornos, _ = cv2.findContours(
        mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contornos:
        return mascara, None

    mayor = max(contornos, key=cv2.contourArea)

    limpia = np.zeros_like(mascara)
    cv2.drawContours(limpia, [mayor], -1, 255, cv2.FILLED)

    return limpia, mayor


# ============================================================
# SEGMENTAR UN OBJETO DETECTADO
# ============================================================

def segmentar_objeto(imagen, deteccion, metodo="grabcut"):
    """
    Devuelve un diccionario con:
      recorte: region con margen alrededor de la caja
      origen: (ox, oy) del recorte en la imagen completa
      mascara: mascara binaria del objeto (tamano del recorte)
      contorno: contorno principal (coordenadas del recorte)
      bordes: bordes Canny de la region
    """

    recorte, (ox, oy) = recortar(imagen, deteccion)

    # Rectangulo de la caja original dentro del recorte
    rect = (
        deteccion["x1"] - ox,
        deteccion["y1"] - oy,
        deteccion["x2"] - deteccion["x1"],
        deteccion["y2"] - deteccion["y1"]
    )

    if metodo == "grabcut":
        mascara = segmentar_grabcut(recorte, rect)
    else:
        mascara = segmentar_otsu(recorte)

    mascara, contorno = limpiar_mascara(mascara)

    gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(cv2.GaussianBlur(gris, (5, 5), 0), 50, 150)

    return {
        "recorte": recorte,
        "origen": (ox, oy),
        "rect": rect,
        "mascara": mascara,
        "contorno": contorno,
        "bordes": bordes
    }


def aislar_objeto(recorte, mascara, atenuacion=0.25):
    """
    Resalta el objeto atenuando el fondo (para visualizar).
    """

    fondo = (recorte * atenuacion).astype(np.uint8)

    return np.where(mascara[:, :, None] > 0, recorte, fondo)
