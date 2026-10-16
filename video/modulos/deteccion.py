import cv2
import os
import torch
from ultralytics import YOLO


# ==========================================================
# RUTA DEL PROYECTO
# ==========================================================

CARPETA_PROYECTO = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# ==========================================================
# RUTAS DE MODELOS
# ==========================================================

RUTA_MODELO_PERSONALIZADO = os.path.join(
    CARPETA_PROYECTO,
    "modelo_ia",
    "best.pt"
)

RUTA_MODELO_PHONE_LAPTOP = os.path.join(
    CARPETA_PROYECTO,
    "modelo_ia",
    "best_phone_laptop.pt"
)

RUTA_MODELO_GENERAL = os.path.join(
    CARPETA_PROYECTO,
    "modelo_ia",
    "yolov8n.pt"
)

RUTA_MODELO_AUDIFONOS = os.path.join(
    CARPETA_PROYECTO,
    "modelo_audifonos",
    "best_audifonos.pt"
)


# ==========================================================
# CARGAR MODELOS
# ==========================================================

print("Cargando modelos...")

modelo_personalizado = YOLO(
    RUTA_MODELO_PERSONALIZADO
)

modelo_phone_laptop = YOLO(
    RUTA_MODELO_PHONE_LAPTOP
)

modelo_general = YOLO(
    RUTA_MODELO_GENERAL
)

print("Cargando modelo de audifonos...")

modelo_audifonos = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path=RUTA_MODELO_AUDIFONOS,
    trust_repo=True
)

modelo_audifonos.conf = 0.25

print("Modelos cargados correctamente.")


# ==========================================================
# CONFIGURACION DE RENDIMIENTO
# ==========================================================

IMGSZ = 416

# Phone/Laptop se ejecuta cada frame
# para que ByteTrack pueda seguir el celular correctamente.
FRECUENCIA_PHONE_LAPTOP = 1

# Estos modelos se ejecutan con menor frecuencia
# para reducir el lag.
FRECUENCIA_AUDIFONOS = 2
FRECUENCIA_GENERAL = 4


# ==========================================================
# CACHE
# ==========================================================

contador_frames = 0

cache_phone_laptop = []
cache_audifonos = []
cache_general = []


# ==========================================================
# CLASES DEL MODELO PERSONALIZADO
# ==========================================================

CLASES = {
    0: "persona",
    1: "mochila",
    2: "telefono",
    3: "cuaderno",
    4: "libro",
    5: "audifonos",
    6: "reloj",
    7: "laptop"
}


# ==========================================================
# UMBRALES
# ==========================================================

UMBRAL = {
    "persona": 0.20,
    "mochila": 0.15,
    "audifonos": 0.15,
    "reloj": 0.15,
    "cuaderno": 0.80,
    "libro": 0.80
}

UMBRAL_PHONE_LAPTOP = 0.30
UMBRAL_TELEFONO_GENERAL = 0.30
UMBRAL_AUDIFONOS = 0.25


# ==========================================================
# NORMALIZAR CLASES
# ==========================================================

def normalizar_clase(nombre):

    nombre = str(nombre).lower().strip()

    if (
        "phone" in nombre
        or "mobile" in nombre
        or "cell" in nombre
        or "telefono" in nombre
    ):
        return "telefono"

    if "laptop" in nombre:
        return "laptop"

    if (
        "tablet" in nombre
        or nombre == "tab"
    ):
        return "tablet"

    return nombre


# ==========================================================
# CALCULAR IOU
# ==========================================================

def calcular_iou(det1, det2):

    x1 = max(
        det1["x1"],
        det2["x1"]
    )

    y1 = max(
        det1["y1"],
        det2["y1"]
    )

    x2 = min(
        det1["x2"],
        det2["x2"]
    )

    y2 = min(
        det1["y2"],
        det2["y2"]
    )

    ancho = max(
        0,
        x2 - x1
    )

    alto = max(
        0,
        y2 - y1
    )

    interseccion = (
        ancho * alto
    )

    area1 = max(
        0,
        det1["x2"] - det1["x1"]
    ) * max(
        0,
        det1["y2"] - det1["y1"]
    )

    area2 = max(
        0,
        det2["x2"] - det2["x1"]
    ) * max(
        0,
        det2["y2"] - det2["y1"]
    )

    union = (
        area1
        + area2
        - interseccion
    )

    if union <= 0:
        return 0.0

    return (
        interseccion / union
    )


# ==========================================================
# MODELO PRINCIPAL
# best.pt
# ==========================================================

def detectar_modelo_personalizado(frame):

    detecciones = []

    resultados = modelo_personalizado.predict(
        source=frame,
        conf=0.15,
        iou=0.45,
        imgsz=IMGSZ,
        verbose=False
    )

    for resultado in resultados:

        if resultado.boxes is None:
            continue

        for caja in resultado.boxes:

            confianza = float(
                caja.conf[0]
            )

            clase_id = int(
                caja.cls[0]
            )

            if clase_id not in CLASES:
                continue

            nombre = CLASES[
                clase_id
            ]

            # Telefono y laptop son manejados
            # por el modelo especializado.
            #
            # Audifonos NO se excluye:
            # best.pt sigue detectandolos.
            if nombre in {
                "telefono",
                "laptop"
            }:
                continue

            umbral = UMBRAL.get(
                nombre,
                0.25
            )

            if confianza < umbral:
                continue

            x1, y1, x2, y2 = map(
                int,
                caja.xyxy[0]
            )

            detecciones.append({
                "clase": nombre,
                "confianza": confianza,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "modelo": "personalizado"
            })

    return detecciones


# ==========================================================
# PHONE / LAPTOP
# CON SEGUIMIENTO BYTETRACK
# ==========================================================

def detectar_phone_laptop(frame):

    detecciones = []

    # persist=True:
    # mantiene internamente el seguimiento del mismo objeto.
    resultados = modelo_phone_laptop.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=UMBRAL_PHONE_LAPTOP,
        iou=0.45,
        imgsz=IMGSZ,
        verbose=False
    )

    for resultado in resultados:

        if resultado.boxes is None:
            continue

        for caja in resultado.boxes:

            confianza = float(
                caja.conf[0]
            )

            clase_id = int(
                caja.cls[0]
            )

            nombre_original = (
                modelo_phone_laptop.names[
                    clase_id
                ]
            )

            nombre = normalizar_clase(
                nombre_original
            )

            if nombre not in {
                "telefono",
                "laptop"
            }:
                continue

            x1, y1, x2, y2 = map(
                int,
                caja.xyxy[0]
            )

            # ==============================================
            # ID INTERNO
            # NO SE MUESTRA EN LA CAMARA
            # ==============================================

            track_id = None

            if caja.id is not None:

                track_id = int(
                    caja.id[0]
                )

            detecciones.append({
                "clase": nombre,
                "confianza": confianza,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "modelo": "phone_laptop",
                "track_id": track_id
            })

    return detecciones


# ==========================================================
# YOLO GENERAL
# RESPALDO PARA TELEFONO
# ==========================================================

def detectar_telefono_general(frame):

    detecciones = []

    resultados = modelo_general.predict(
        source=frame,
        classes=[67],
        conf=UMBRAL_TELEFONO_GENERAL,
        iou=0.45,
        imgsz=IMGSZ,
        verbose=False
    )

    for resultado in resultados:

        if resultado.boxes is None:
            continue

        for caja in resultado.boxes:

            confianza = float(
                caja.conf[0]
            )

            x1, y1, x2, y2 = map(
                int,
                caja.xyxy[0]
            )

            detecciones.append({
                "clase": "telefono",
                "confianza": confianza,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "modelo": "general",
                "track_id": None
            })

    return detecciones


# ==========================================================
# MODELO ESPECIALIZADO AUDIFONOS
# ==========================================================

def detectar_audifonos_especializado(frame):

    detecciones = []

    resultados = modelo_audifonos(
        frame,
        size=IMGSZ
    )

    resultados_xyxy = (
        resultados.xyxy[0]
    )

    alto_frame, ancho_frame = (
        frame.shape[:2]
    )

    for deteccion in resultados_xyxy:

        x1 = int(
            deteccion[0]
        )

        y1 = int(
            deteccion[1]
        )

        x2 = int(
            deteccion[2]
        )

        y2 = int(
            deteccion[3]
        )

        confianza = float(
            deteccion[4]
        )

        clase_id = int(
            deteccion[5]
        )

        nombre = str(
            modelo_audifonos.names[
                clase_id
            ]
        ).lower().strip()

        if nombre not in [
            "earphone",
            "earphones",
            "headphone",
            "headphones",
            "earbud",
            "earbuds"
        ]:
            continue

        if (
            confianza
            <
            UMBRAL_AUDIFONOS
        ):
            continue

        ancho_caja = (
            x2 - x1
        )

        alto_caja = (
            y2 - y1
        )

        porcentaje_ancho = (
            ancho_caja
            / ancho_frame
        )

        porcentaje_alto = (
            alto_caja
            / alto_frame
        )

        # Evitar detectar toda la cara.
        if porcentaje_ancho > 0.22:
            continue

        if porcentaje_alto > 0.35:
            continue

        # Evitar cajas demasiado pequeñas.
        if (
            ancho_caja < 8
            or alto_caja < 8
        ):
            continue

        detecciones.append({
            "clase": "audifonos",
            "confianza": confianza,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "modelo": "audifonos_especializado"
        })

    return detecciones


# ==========================================================
# RESOLVER CONFLICTOS PHONE / LAPTOP
# ==========================================================

def resolver_conflictos(
    especializadas,
    generales
):

    resultado = list(
        especializadas
    )

    for telefono_general in generales:

        conflicto = False

        for especializada in especializadas:

            iou = calcular_iou(
                telefono_general,
                especializada
            )

            if (
                especializada["clase"] == "laptop"
                and iou >= 0.35
            ):
                conflicto = True
                break

            if (
                especializada["clase"] == "telefono"
                and iou >= 0.35
            ):
                conflicto = True
                break

        if not conflicto:

            resultado.append(
                telefono_general
            )

    return resultado


# ==========================================================
# AUDIFONOS SOBRE FALSO TELEFONO
# ==========================================================

def resolver_conflictos_audifonos(
    detecciones,
    audifonos
):

    resultado = []

    for deteccion in detecciones:

        eliminar = False

        if (
            deteccion["clase"]
            == "telefono"
        ):

            for audifono in audifonos:

                iou = calcular_iou(
                    deteccion,
                    audifono
                )

                if iou >= 0.30:

                    eliminar = True
                    break

        if not eliminar:

            resultado.append(
                deteccion
            )

    resultado.extend(
        audifonos
    )

    return resultado


# ==========================================================
# ELIMINAR DUPLICADOS
# ==========================================================

def eliminar_duplicados(
    detecciones
):

    detecciones = sorted(
        detecciones,
        key=lambda x: x["confianza"],
        reverse=True
    )

    finales = []

    for deteccion in detecciones:

        duplicado = False

        for guardada in finales:

            if (
                deteccion["clase"]
                != guardada["clase"]
            ):
                continue

            iou = calcular_iou(
                deteccion,
                guardada
            )

            if iou >= 0.40:

                duplicado = True
                break

        if not duplicado:

            finales.append(
                deteccion
            )

    return finales


# ==========================================================
# DETECTAR OBJETOS
# ==========================================================

def detectar_objetos(frame):

    global contador_frames
    global cache_phone_laptop
    global cache_audifonos
    global cache_general

    contador_frames += 1


    # ======================================================
    # MODELO PRINCIPAL
    # CADA FRAME
    # ======================================================

    detecciones_personalizadas = (
        detectar_modelo_personalizado(
            frame
        )
    )


    # ======================================================
    # PHONE / LAPTOP
    # CADA FRAME PARA TRACKING
    # ======================================================

    if (
        contador_frames
        % FRECUENCIA_PHONE_LAPTOP
        == 0
        or not cache_phone_laptop
    ):

        cache_phone_laptop = (
            detectar_phone_laptop(
                frame
            )
        )

    detecciones_phone_laptop = (
        cache_phone_laptop
    )


    # ======================================================
    # AUDIFONOS
    # CADA 2 FRAMES
    # ======================================================

    if (
        contador_frames
        % FRECUENCIA_AUDIFONOS
        == 0
        or not cache_audifonos
    ):

        cache_audifonos = (
            detectar_audifonos_especializado(
                frame
            )
        )

    detecciones_audifonos = (
        cache_audifonos
    )


    # ======================================================
    # YOLO GENERAL
    # CADA 4 FRAMES
    # ======================================================

    if (
        contador_frames
        % FRECUENCIA_GENERAL
        == 0
        or not cache_general
    ):

        cache_general = (
            detectar_telefono_general(
                frame
            )
        )

    detecciones_general = (
        cache_general
    )


    # ======================================================
    # RESOLVER TELEFONO / LAPTOP
    # ======================================================

    detecciones_dispositivos = (
        resolver_conflictos(
            detecciones_phone_laptop,
            detecciones_general
        )
    )


    # ======================================================
    # COMBINAR
    # ======================================================

    detecciones = (
        detecciones_personalizadas
        +
        detecciones_dispositivos
    )


    # ======================================================
    # AGREGAR AUDIFONOS
    # ======================================================

    detecciones = (
        resolver_conflictos_audifonos(
            detecciones,
            detecciones_audifonos
        )
    )


    # ======================================================
    # QUITAR DUPLICADOS
    # ======================================================

    detecciones = (
        eliminar_duplicados(
            detecciones
        )
    )

    return detecciones


# ==========================================================
# DIBUJAR DETECCIONES
# ==========================================================

def dibujar_detecciones(
    frame,
    detecciones
):

    objetos_permitidos = {
        "persona",
        "mochila"
    }

    nombres_pantalla = {
        "persona": "Persona",
        "mochila": "Mochila",
        "telefono": "Telefono",
        "audifonos": "Audifonos",
        "reloj": "Reloj",
        "cuaderno": "Cuaderno/Libro",
        "libro": "Cuaderno/Libro",
        "laptop": "Laptop"
    }

    for deteccion in detecciones:

        clase = deteccion[
            "clase"
        ]

        confianza = deteccion[
            "confianza"
        ]

        x1 = deteccion["x1"]
        y1 = deteccion["y1"]
        x2 = deteccion["x2"]
        y2 = deteccion["y2"]

        # AZUL = permitido
        if clase in objetos_permitidos:

            color = (
                255,
                0,
                0
            )

        # ROJO = no permitido
        else:

            color = (
                0,
                0,
                255
            )

        etiqueta = nombres_pantalla.get(
            clase,
            clase
        )

        # IMPORTANTE:
        # El track_id NO se muestra.
        texto = (
            f"{etiqueta} "
            f"{confianza:.2f}"
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        posicion_y = (
            y1 - 10
        )

        if posicion_y < 30:

            posicion_y = (
                y1 + 30
            )

        cv2.putText(
            frame,
            texto,
            (
                x1,
                posicion_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA
        )

    return frame