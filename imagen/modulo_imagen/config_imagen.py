# ============================================================
# CONFIGURACION DEL MODULO IMAGEN
# ============================================================

import os
import sys


# Estructura del sistema: cada grupo en su carpeta
#   video/  imagen/  prediccion/  integracion/
CARPETA_IMAGEN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAIZ = os.path.dirname(CARPETA_IMAGEN)
CARPETA_VIDEO = os.path.join(RAIZ, "video")
CARPETA_PREDICCION = os.path.join(RAIZ, "prediccion")

# El codigo de VIDEO (modulos/, configuracion/, principal.py) y el de
# la integracion se importan desde sus carpetas
for _carpeta in (CARPETA_VIDEO, RAIZ):
    if _carpeta not in sys.path:
        sys.path.append(_carpeta)

CARPETA_RESULTADOS = os.path.join(
    RAIZ,
    "resultados"
)

# Carpeta donde VIDEO guarda las capturas de sus alarmas
# (la misma que usa video/modulos/registro.py)
CARPETA_ALERTAS_VIDEO = os.path.join(
    CARPETA_VIDEO,
    "alertas"
)

# Resultados del procesamiento de las alertas
CARPETA_EN_VIVO = os.path.join(
    CARPETA_RESULTADOS,
    "alertas"
)

# Cada cuantos segundos se revisa la carpeta de alertas
INTERVALO_REVISION_SEG = 2


# ============================================================
# RECEPCION
# ============================================================

# Los fotogramas se llevan a una resolucion tipica de camara
# antes de simular la deteccion del grupo VIDEO.
LADO_MAXIMO_FRAME = 1280

# Cada cuantos segundos se toma un fotograma de un video.
INTERVALO_MUESTREO_SEG = 1.0


# ============================================================
# CLASES
# ============================================================

# Mismos objetos no permitidos que usa principal.py de VIDEO
CLASES_NO_PERMITIDAS = {
    "telefono",
    "cuaderno",
    "libro",
    "audifonos",
    "reloj"
}

# Cuaderno y libro se consideran el mismo grupo al comparar,
# igual que en la visualizacion de VIDEO ("Cuaderno/Libro").
GRUPO_CLASE = {
    "cuaderno": "cuaderno/libro",
    "libro": "cuaderno/libro"
}


# ============================================================
# SEGMENTACION Y VERIFICACION
# ============================================================

# Margen alrededor de cada caja al recortar (proporcion)
MARGEN_RECORTE = 0.15

# Lado minimo al que se amplia una region para visualizarla
LADO_MINIMO_REGION = 400

# IoU minimo para emparejar una deteccion antes y despues
# del procesamiento
IOU_COINCIDENCIA = 0.30
