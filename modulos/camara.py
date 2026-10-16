import cv2
import time

from configuracion.config import CAMARA


def iniciar_camara():

    # En Windows suele funcionar mejor con DirectShow
    camara = cv2.VideoCapture(
        CAMARA,
        cv2.CAP_DSHOW
    )

    if not camara.isOpened():
        print("Error: No se pudo abrir la cámara")
        return None

    # Resolución moderada para evitar lag
    camara.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    camara.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    # Intentar reducir buffer
    camara.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    # Darle tiempo a la cámara para inicializar
    time.sleep(0.5)

    return camara


def obtener_frame(camara):

    if camara is None:
        return None

    correcto, frame = camara.read()

    if correcto and frame is not None:
        return frame

    # Segundo intento por si el primer frame falla
    time.sleep(0.03)

    correcto, frame = camara.read()

    if correcto and frame is not None:
        return frame

    return None


def cerrar_camara(camara):

    if camara is not None:
        camara.release()

    cv2.destroyAllWindows()