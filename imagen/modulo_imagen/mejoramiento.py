# ============================================================
# MEJORAMIENTO DE IMAGEN
# Contraste, brillo, nitidez y ampliacion de regiones.
# ============================================================

import cv2
import numpy as np


# ============================================================
# BRILLO Y CONTRASTE
# ============================================================

def ajustar_brillo_contraste(imagen, alfa=1.0, beta=0):
    """
    g(x) = alfa * f(x) + beta
    alfa: contraste (1.0 = sin cambio)
    beta: brillo (0 = sin cambio)
    """

    return cv2.convertScaleAbs(imagen, alpha=alfa, beta=beta)


def estiramiento_contraste(imagen, recorte=0.5):
    """
    Estiramiento lineal del histograma por percentiles:
    el percentil `recorte` pasa a 0 y el (100 - recorte) a 255.
    """

    resultado = np.empty_like(imagen)

    for canal in range(imagen.shape[2]):

        bajo, alto = np.percentile(
            imagen[:, :, canal],
            (recorte, 100 - recorte)
        )

        if alto - bajo < 1:
            resultado[:, :, canal] = imagen[:, :, canal]
            continue

        estirado = (imagen[:, :, canal].astype(np.float32) - bajo)
        estirado *= 255.0 / (alto - bajo)

        resultado[:, :, canal] = np.clip(estirado, 0, 255)

    return resultado


# ============================================================
# NITIDEZ
# ============================================================

def enfocar(imagen, cantidad=1.0, sigma=2.0):
    """
    Mascara de desenfoque (unsharp masking):
    resultado = original + cantidad * (original - suavizada)
    """

    if cantidad <= 0:
        return imagen.copy()

    suavizada = cv2.GaussianBlur(imagen, (0, 0), sigma)

    return cv2.addWeighted(
        imagen, 1.0 + cantidad,
        suavizada, -cantidad,
        0
    )


def enfocar_laplaciano(imagen):

    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)

    return cv2.filter2D(imagen, -1, kernel)


# ============================================================
# AMPLIACION (super-resolucion por interpolacion)
# ============================================================

def ampliar(imagen, lado_minimo):
    """
    Amplia la imagen con interpolacion bicubica hasta que su
    lado mayor alcance `lado_minimo`. Devuelve la imagen y la
    escala aplicada (para devolver coordenadas al original).
    """

    alto, ancho = imagen.shape[:2]

    escala = lado_minimo / max(alto, ancho)

    if escala <= 1:
        return imagen, 1.0

    ampliada = cv2.resize(
        imagen,
        None,
        fx=escala,
        fy=escala,
        interpolation=cv2.INTER_CUBIC
    )

    return ampliada, escala


# ============================================================
# MEJORAMIENTO COMPLETO
# ============================================================

def mejorar(
    imagen,
    contraste="estiramiento",
    alfa=1.0,
    beta=0,
    nitidez=0.8,
    sigma_nitidez=2.0
):
    """
    contraste: ninguno | estiramiento | manual
    sigma_nitidez: radio del desenfoque que se compensa (mayor para
                   imagenes mas borrosas)
    """

    if contraste == "estiramiento":
        resultado = estiramiento_contraste(imagen)
    elif contraste == "manual":
        resultado = ajustar_brillo_contraste(imagen, alfa, beta)
    else:
        resultado = imagen.copy()

    return enfocar(resultado, nitidez, sigma_nitidez)
