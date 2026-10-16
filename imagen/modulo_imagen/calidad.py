# ============================================================
# METRICAS DE CALIDAD DE IMAGEN
# Permiten comparar objetivamente antes y despues.
# ============================================================

import cv2
import numpy as np

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)


def estimar_ruido(gris):
    """
    Estimacion rapida de la desviacion del ruido (Immerkaer, 1996):
    se filtra con una mascara que anula bordes y zonas suaves, y
    lo que queda es aproximadamente ruido.
    """

    mascara = np.array([
        [1, -2, 1],
        [-2, 4, -2],
        [1, -2, 1]
    ], dtype=np.float64)

    alto, ancho = gris.shape

    respuesta = cv2.filter2D(gris.astype(np.float64), -1, mascara)

    suma = np.abs(respuesta[1:-1, 1:-1]).sum()

    return suma * np.sqrt(0.5 * np.pi) / (6.0 * (ancho - 2) * (alto - 2))


def medir_calidad(imagen):
    """
    brillo: media de intensidad (0-255)
    contraste: desviacion estandar de intensidad
    nitidez: varianza del Laplaciano (mayor = mas nitida)
    ruido: sigma estimada del ruido (menor = mas limpia)
    entropia: informacion del histograma (bits)
    """

    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    histograma = cv2.calcHist([gris], [0], None, [256], [0, 256]).ravel()
    probabilidad = histograma / histograma.sum()
    probabilidad = probabilidad[probabilidad > 0]

    return {
        "brillo": round(float(gris.mean()), 2),
        "contraste": round(float(gris.std()), 2),
        "nitidez": round(float(cv2.Laplacian(gris, cv2.CV_64F).var()), 2),
        "ruido": round(float(estimar_ruido(gris)), 3),
        "entropia": round(float(-(probabilidad * np.log2(probabilidad)).sum()), 3)
    }


def comparar_con_referencia(referencia, procesada):
    """
    PSNR y SSIM respecto a una imagen de referencia
    (por ejemplo, la original antes de degradarla).
    """

    gris_ref = cv2.cvtColor(referencia, cv2.COLOR_BGR2GRAY)
    gris_pro = cv2.cvtColor(procesada, cv2.COLOR_BGR2GRAY)

    return {
        "psnr": round(float(peak_signal_noise_ratio(gris_ref, gris_pro)), 2),
        "ssim": round(float(structural_similarity(gris_ref, gris_pro)), 4)
    }
