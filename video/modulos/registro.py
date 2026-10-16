import os
import cv2
from datetime import datetime
import pygame


# ============================================================
# RUTAS
# ============================================================

CARPETA_PROYECTO = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

CARPETA_ALERTAS = os.path.join(
    CARPETA_PROYECTO,
    "alertas"
)

CARPETA_CAPTURAS = os.path.join(
    CARPETA_PROYECTO,
    "capturas"
)

RUTA_SONIDO = os.path.join(
    CARPETA_PROYECTO,
    "sonido",
    "alerta.mp3"
)


# Crear carpetas si no existen
os.makedirs(CARPETA_ALERTAS, exist_ok=True)
os.makedirs(CARPETA_CAPTURAS, exist_ok=True)


# ============================================================
# AUDIO
# ============================================================

audio_iniciado = False


def iniciar_audio():

    global audio_iniciado

    if audio_iniciado:
        return True

    try:

        pygame.mixer.init()

        audio_iniciado = True

        print("Audio iniciado correctamente.")

        print(
            f"Archivo de sonido: {RUTA_SONIDO}"
        )

        return True

    except Exception as e:

        print(
            f"No se pudo iniciar el audio: {e}"
        )

        audio_iniciado = False

        return False


# ============================================================
# REPRODUCIR ALERTA
# ============================================================

def reproducir_alerta():

    # Asegurar que pygame este iniciado
    if not iniciar_audio():
        return

    # Verificar que exista el archivo
    if not os.path.isfile(RUTA_SONIDO):

        print(
            "ERROR: No se encontro el archivo de sonido:"
        )

        print(RUTA_SONIDO)

        return

    try:

        # Detener cualquier sonido anterior
        pygame.mixer.music.stop()

        # Cargar nuevamente el audio
        pygame.mixer.music.load(
            RUTA_SONIDO
        )

        # Reproducir
        pygame.mixer.music.play()

        print(
            "ALERTA SONORA REPRODUCIDA"
        )

    except Exception as e:

        print(
            f"Error al reproducir alerta: {e}"
        )


# ============================================================
# GUARDAR ALERTA
# ============================================================

def guardar_alerta(frame, clase):

    ahora = datetime.now()

    fecha = ahora.strftime(
        "%Y-%m-%d"
    )

    hora = ahora.strftime(
        "%H-%M-%S"
    )

    nombre = (
        f"alerta_{clase}_{fecha}_{hora}.jpg"
    )

    ruta_alerta = os.path.join(
        CARPETA_ALERTAS,
        nombre
    )

    # ========================================================
    # GUARDAR IMAGEN
    # ========================================================

    cv2.imwrite(
        ruta_alerta,
        frame
    )

    print(
        f"ALERTA GUARDADA: {ruta_alerta}"
    )

    # ========================================================
    # REPRODUCIR SONIDO
    # ========================================================

    reproducir_alerta()

    return ruta_alerta