# ============================================================
# PREPROCESAMIENTO
# Reduccion de ruido y correccion de iluminacion.
# ============================================================

import cv2
import numpy as np


# ============================================================
# REDUCCION DE RUIDO
# ============================================================

def reducir_ruido(imagen, metodo="mediana", fuerza=5):
    """
    metodo: ninguno | mediana | gaussiano | bilateral | nlmeans
    fuerza: tamano de kernel (mediana/gaussiano) o intensidad.
    """

    if metodo == "ninguno":
        return imagen.copy()

    k = int(fuerza) | 1  # kernel impar

    if metodo == "mediana":
        return cv2.medianBlur(imagen, k)

    if metodo == "gaussiano":
        return cv2.GaussianBlur(imagen, (k, k), 0)

    if metodo == "bilateral":
        return cv2.bilateralFilter(imagen, k, 15 * fuerza, 15 * fuerza)

    if metodo == "nlmeans":
        return cv2.fastNlMeansDenoisingColored(
            imagen, None, fuerza, fuerza, 7, 21
        )

    raise ValueError(f"Metodo de ruido desconocido: {metodo}")


# ============================================================
# CORRECCION DE ILUMINACION
# ============================================================

def correccion_gamma(imagen, gamma):

    tabla = np.array([
        ((i / 255.0) ** (1.0 / gamma)) * 255
        for i in range(256)
    ]).astype(np.uint8)

    return cv2.LUT(imagen, tabla)


def gamma_automatica(imagen):
    """
    Calcula el gamma que lleva el brillo medio a 0.5.
    """

    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    media = np.clip(gris.mean() / 255.0, 0.01, 0.99)

    gamma = np.log(media) / np.log(0.5)

    return float(np.clip(gamma, 0.3, 3.0))


def clahe(imagen, limite=2.0, rejilla=8):
    """
    Ecualizacion adaptativa sobre el canal L (espacio LAB)
    para no alterar los colores.
    """

    lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    operador = cv2.createCLAHE(
        clipLimit=limite,
        tileGridSize=(rejilla, rejilla)
    )

    l = operador.apply(l)

    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def ecualizacion_global(imagen):

    ycrcb = cv2.cvtColor(imagen, cv2.COLOR_BGR2YCrCb)

    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])

    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def corregir_iluminacion(imagen, metodo="clahe", limite_clahe=2.0):
    """
    metodo: ninguno | clahe | ecualizacion | gamma_auto | gamma_auto+clahe
    """

    if metodo == "ninguno":
        return imagen.copy()

    if metodo == "clahe":
        return clahe(imagen, limite_clahe)

    if metodo == "ecualizacion":
        return ecualizacion_global(imagen)

    if metodo == "gamma_auto":
        return correccion_gamma(imagen, gamma_automatica(imagen))

    if metodo == "gamma_auto+clahe":
        corregida = correccion_gamma(imagen, gamma_automatica(imagen))
        return clahe(corregida, limite_clahe)

    raise ValueError(f"Metodo de iluminacion desconocido: {metodo}")


# ============================================================
# PREPROCESAMIENTO COMPLETO
# ============================================================

def preprocesar(
    imagen,
    ruido="mediana",
    fuerza_ruido=3,
    iluminacion="clahe",
    limite_clahe=2.0
):

    sin_ruido = reducir_ruido(imagen, ruido, fuerza_ruido)

    return corregir_iluminacion(sin_ruido, iluminacion, limite_clahe)


# ============================================================
# DEGRADACION (para experimentos)
# Simula condiciones malas de camara: poca luz, ruido y
# desenfoque, para medir cuanto recupera el procesamiento.
# ============================================================

def degradar(imagen, oscuridad=0.0, ruido=0.0, desenfoque=0, semilla=0):
    """
    oscuridad: 0..1 (0 = sin cambio, 0.7 = muy oscuro)
    ruido: desviacion estandar del ruido gaussiano (0..60)
    desenfoque: tamano de kernel del desenfoque (0 = sin cambio)
    """

    resultado = imagen.astype(np.float32)

    if oscuridad > 0:
        resultado *= (1.0 - oscuridad)

    if ruido > 0:
        generador = np.random.default_rng(semilla)
        resultado += generador.normal(0, ruido, resultado.shape)

    resultado = np.clip(resultado, 0, 255).astype(np.uint8)

    if desenfoque > 0:
        k = int(desenfoque) | 1
        resultado = cv2.GaussianBlur(resultado, (k, k), 0)

    return resultado
