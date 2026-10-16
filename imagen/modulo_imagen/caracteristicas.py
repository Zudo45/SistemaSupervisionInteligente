# ============================================================
# EXTRACCION DE CARACTERISTICAS
# Tamano, posicion, forma, color y textura de cada objeto.
# ============================================================

import cv2
import numpy as np


NOMBRES_COLOR = [
    (10, "rojo"),
    (25, "naranja"),
    (35, "amarillo"),
    (85, "verde"),
    (130, "azul"),
    (160, "morado"),
    (180, "rojo")
]


# ============================================================
# UTILIDADES GEOMETRICAS
# ============================================================

def iou(a, b):

    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])

    interseccion = max(0, x2 - x1) * max(0, y2 - y1)

    if interseccion == 0:
        return 0.0

    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])

    return interseccion / float(area_a + area_b - interseccion)


def proporcion_dentro(interior, exterior):
    """
    Fraccion del area de `interior` que cae dentro de `exterior`.
    """

    x1 = max(interior["x1"], exterior["x1"])
    y1 = max(interior["y1"], exterior["y1"])
    x2 = min(interior["x2"], exterior["x2"])
    y2 = min(interior["y2"], exterior["y2"])

    interseccion = max(0, x2 - x1) * max(0, y2 - y1)

    area = (interior["x2"] - interior["x1"]) * (interior["y2"] - interior["y1"])

    return interseccion / float(area) if area else 0.0


def zona_en_frame(cx, cy):
    """
    Zona en una rejilla de 3x3 (cx, cy normalizados 0-1).
    """

    filas = ["superior", "media", "inferior"]
    columnas = ["izquierda", "centro", "derecha"]

    return (
        f"{filas[min(2, int(cy * 3))]}-"
        f"{columnas[min(2, int(cx * 3))]}"
    )


def nombre_color(tono, saturacion, valor):

    if valor < 50:
        return "negro"

    if saturacion < 40:
        return "blanco" if valor > 180 else "gris"

    for limite, nombre in NOMBRES_COLOR:
        if tono < limite:
            return nombre

    return "rojo"


# ============================================================
# CARACTERISTICAS DE UN OBJETO
# ============================================================

def extraer_caracteristicas(imagen, deteccion, segmento, personas):
    """
    imagen: frame completo
    deteccion: deteccion (formato VIDEO)
    segmento: resultado de segmentacion.segmentar_objeto
    personas: lista de detecciones de clase persona
    """

    alto, ancho = imagen.shape[:2]

    x1, y1 = deteccion["x1"], deteccion["y1"]
    x2, y2 = deteccion["x2"], deteccion["y2"]

    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2 / ancho, (y1 + y2) / 2 / alto

    caracteristicas = {
        # Tamano y posicion
        "ancho_px": int(w),
        "alto_px": int(h),
        "area_px": int(w * h),
        "area_relativa": round(w * h / float(ancho * alto), 5),
        "centro_x": round(cx, 4),
        "centro_y": round(cy, 4),
        "zona": zona_en_frame(cx, cy),
        "relacion_aspecto": round(w / float(h), 3) if h else 0.0
    }

    # --------------------------------------------------------
    # Forma (a partir del contorno segmentado)
    # --------------------------------------------------------

    contorno = segmento["contorno"]
    mascara = segmento["mascara"]

    if contorno is not None and cv2.contourArea(contorno) > 0:

        area = cv2.contourArea(contorno)
        perimetro = cv2.arcLength(contorno, True)
        casco = cv2.convexHull(contorno)
        area_casco = cv2.contourArea(casco)

        caracteristicas.update({
            "area_contorno": round(float(area), 1),
            "perimetro": round(float(perimetro), 1),
            "circularidad": round(4 * np.pi * area / perimetro ** 2, 4) if perimetro else 0.0,
            "solidez": round(area / area_casco, 4) if area_casco else 0.0,
            "relleno_caja": round(area / float(w * h), 4) if w * h else 0.0,
            "vertices_aprox": int(len(cv2.approxPolyDP(contorno, 0.02 * perimetro, True)))
        })

    else:

        caracteristicas.update({
            "area_contorno": 0.0,
            "perimetro": 0.0,
            "circularidad": 0.0,
            "solidez": 0.0,
            "relleno_caja": 0.0,
            "vertices_aprox": 0
        })

    # --------------------------------------------------------
    # Color y textura (dentro de la mascara)
    # --------------------------------------------------------

    recorte = segmento["recorte"]

    pixeles = mascara > 0

    if not pixeles.any():
        pixeles = np.ones(mascara.shape, dtype=bool)

    hsv = cv2.cvtColor(recorte, cv2.COLOR_BGR2HSV)

    tono = float(np.median(hsv[:, :, 0][pixeles]))
    saturacion = float(np.median(hsv[:, :, 1][pixeles]))
    valor = float(np.median(hsv[:, :, 2][pixeles]))

    gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)

    caracteristicas.update({
        "color_dominante": nombre_color(tono, saturacion, valor),
        "brillo_region": round(float(gris[pixeles].mean()), 2),
        "contraste_region": round(float(gris[pixeles].std()), 2),
        "nitidez_region": round(float(cv2.Laplacian(gris, cv2.CV_64F).var()), 2),
        "densidad_bordes": round(float((segmento["bordes"] > 0).mean()), 4)
    })

    # --------------------------------------------------------
    # Relacion con la persona mas cercana
    # --------------------------------------------------------

    mejor = None
    mejor_dentro = 0.0

    for indice, persona in enumerate(personas):

        dentro = proporcion_dentro(deteccion, persona)

        if dentro > mejor_dentro:
            mejor, mejor_dentro = indice, dentro

    if mejor is not None:

        persona = personas[mejor]

        px = (persona["x1"] + persona["x2"]) / 2 / ancho
        py = (persona["y1"] + persona["y2"]) / 2 / alto

        caracteristicas.update({
            "persona_asociada": mejor,
            "proporcion_dentro_persona": round(mejor_dentro, 3),
            "distancia_a_persona": round(float(np.hypot(cx - px, cy - py)), 4)
        })

    else:

        caracteristicas.update({
            "persona_asociada": None,
            "proporcion_dentro_persona": 0.0,
            "distancia_a_persona": None
        })

    return caracteristicas
