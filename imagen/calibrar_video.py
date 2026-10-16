# ============================================================
# CALIBRACION DE LA ALARMA DE VIDEO (para la demo de IMAGEN)
#
# Se ejecuta el modelo de VIDEO (best.pt, sin cambios) sobre la
# camara mientras el usuario sigue una serie de pasos (con y sin
# objetos). NO se guarda video ni imagenes: solo se anotan, por
# cada fotograma, las clases detectadas y su confianza.
#
# Con esos numeros se eligen, por objeto, el umbral de confianza
# y la tolerancia a parpadeos que hacen que la alarma suene con
# los objetos reales y no con los distractores.
#
# Uso:
#   python calibrar_video.py              (captura + analisis)
#   python calibrar_video.py --analizar   (solo analisis)
# ============================================================

import os
import sys
import json
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

import modulo_imagen.config_imagen  # noqa: F401  (agrega video/ al path)

from configuracion.config import CAMARA, TIEMPO_ALERTA
from modulos.deteccion import UMBRAL

try:
    # Version nueva de VIDEO (varios modelos): se calibra su modelo principal
    from modulos.deteccion import modelo_personalizado as modelo
except ImportError:
    from modulos.deteccion import modelo


CARPETA_CALIBRACION = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "calibracion"
)

RUTA_FRAMES = os.path.join(CARPETA_CALIBRACION, "frames.csv")
RUTA_DETECCIONES = os.path.join(CARPETA_CALIBRACION, "detecciones.csv")
RUTA_RESULTADO = os.path.join(CARPETA_CALIBRACION, "calibracion_video.json")

DURACION_PASO = 30          # segundos que se registra cada paso
PREPARACION = 6             # segundos para prepararse (no se registra)
CONFIANZA_REGISTRO = 0.05   # se registra todo lo que el modelo "cree" ver

# detectar_objetos de VIDEO filtra internamente con conf=0.15,
# por eso ningun umbral calibrado puede ser menor.
UMBRAL_MINIMO = 0.15

TOLERANCIAS = [0.0, 0.25, 0.5, 1.0, 1.5]

GRUPOS = {
    "telefono": {"telefono"},
    "cuaderno/libro": {"cuaderno", "libro"},
    "audifonos": {"audifonos"},
    "reloj": {"reloj"}
}

PASOS = [
    {"id": "sin_objetos", "tipo": "negativo", "objetivo": None,
     "texto": "Sentado normal, manos sobre el escritorio, SIN objetos"},
    {"id": "lapicero", "tipo": "negativo", "objetivo": None,
     "texto": "Escribe con un LAPICERO en una hoja; muevelo y sostenlo como si fuera un celular"},
    {"id": "celular_mano", "tipo": "positivo", "objetivo": "telefono",
     "texto": "CELULAR en la mano: vertical, horizontal, de costado, alto y bajo"},
    {"id": "celular_escritorio", "tipo": "positivo", "objetivo": "telefono",
     "texto": "CELULAR sobre el escritorio (pantalla arriba); miralo de reojo"},
    {"id": "cuaderno", "tipo": "positivo", "objetivo": "cuaderno/libro",
     "texto": "CUADERNO abierto en el escritorio, escribe; luego levantalo un poco"},
    {"id": "audifonos", "tipo": "positivo", "objetivo": "audifonos",
     "texto": "AUDIFONOS puestos; gira la cabeza a ambos lados"},
    {"id": "reloj", "tipo": "positivo", "objetivo": "reloj",
     "texto": "RELOJ en la muneca; mueve el brazo (S = saltar si no tienes)"}
]

PASOS_POCA_LUZ = [
    {"id": "poca_luz_sin_objetos", "tipo": "negativo", "objetivo": None,
     "texto": "POCA LUZ: sentado normal, SIN objetos"},
    {"id": "poca_luz_celular", "tipo": "positivo", "objetivo": "telefono",
     "texto": "POCA LUZ: CELULAR en la mano, varias posiciones"}
]


# ============================================================
# INTERFAZ EN PANTALLA
# ============================================================

def escribir(frame, lineas, color=(255, 255, 255)):

    alto, ancho = frame.shape[:2]
    escala = ancho / 1000
    espacio = int(38 * escala) + 10

    cv2.rectangle(frame, (0, 0), (ancho, espacio * len(lineas) + 15), (0, 0, 0), cv2.FILLED)

    for i, linea in enumerate(lineas):
        cv2.putText(frame, linea, (15, espacio * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8 * escala + 0.2, color, 2, cv2.LINE_AA)


def mostrar(frame):

    alto, ancho = frame.shape[:2]
    escala = 960 / ancho

    cv2.imshow("Calibracion alarma VIDEO", cv2.resize(frame, None, fx=escala, fy=escala))

    return cv2.waitKey(1) & 0xFF


def leer(camara):

    correcto, frame = camara.read()

    if not correcto:
        raise RuntimeError("No se pudo leer la camara")

    return frame


# ============================================================
# CAPTURA
# ============================================================

def esperar_tecla(camara, lineas, teclas):

    while True:

        frame = leer(camara)
        escribir(frame, lineas)
        tecla = mostrar(frame)

        for t in teclas:
            if tecla == ord(t):
                return t


def ejecutar_paso(camara, paso, indice, total, frames, detecciones, luz):
    """
    Devuelve False si el usuario pide salir.
    """

    inicio = time.time()

    while time.time() - inicio < PREPARACION:

        frame = leer(camara)
        restante = PREPARACION - int(time.time() - inicio)
        escribir(frame, [
            f"Paso {indice}/{total} - preparate ({restante}s)",
            paso["texto"]
        ], (0, 255, 255))

        tecla = mostrar(frame)

        if tecla == ord("q"):
            return False
        if tecla == ord("s"):
            return True

    inicio = time.time()
    numero = 0

    while True:

        transcurrido = time.time() - inicio

        if transcurrido >= DURACION_PASO:
            break

        frame = leer(camara)

        resultado = modelo.predict(
            source=frame,
            conf=CONFIANZA_REGISTRO,
            iou=0.45,
            verbose=False
        )[0]

        alto, ancho = frame.shape[:2]
        cantidad = 0

        for caja in resultado.boxes:

            x1, y1, x2, y2 = map(float, caja.xyxy[0])

            detecciones.append({
                "paso": paso["id"],
                "frame": numero,
                "clase": modelo.names[int(caja.cls[0])],
                "confianza": round(float(caja.conf[0]), 4),
                "area_relativa": round((x2 - x1) * (y2 - y1) / (ancho * alto), 5)
            })

            cantidad += 1

        frames.append({
            "paso": paso["id"],
            "tipo": paso["tipo"],
            "objetivo": paso["objetivo"] or "",
            "luz": luz,
            "frame": numero,
            "t": round(transcurrido, 3),
            "detecciones": cantidad
        })

        numero += 1

        escribir(frame, [
            f"Paso {indice}/{total} - REGISTRANDO {DURACION_PASO - int(transcurrido)}s"
            "  (solo numeros, sin video)",
            paso["texto"],
            "S = saltar paso   Q = terminar"
        ], (0, 255, 0))

        tecla = mostrar(frame)

        if tecla == ord("q"):
            return False
        if tecla == ord("s"):
            break

    return True


def capturar():

    os.makedirs(CARPETA_CALIBRACION, exist_ok=True)

    camara = cv2.VideoCapture(CAMARA)

    if not camara.isOpened():
        print("No se pudo abrir la camara")
        return False

    frames = []
    detecciones = []

    try:

        tecla = esperar_tecla(camara, [
            "Ajusta la camara: de tu cabeza hasta el escritorio",
            "ESPACIO = empezar   Q = salir"
        ], " q")

        if tecla == "q":
            return False

        continuar = True

        for i, paso in enumerate(PASOS, start=1):

            continuar = ejecutar_paso(camara, paso, i, len(PASOS), frames, detecciones, "normal")

            if not continuar:
                break

        if continuar:

            tecla = esperar_tecla(camara, [
                "Opcional: baja la luz (deja solo la pantalla)",
                "L = registrar con poca luz   Q = terminar"
            ], "lq")

            if tecla == "l":
                for i, paso in enumerate(PASOS_POCA_LUZ, start=1):
                    if not ejecutar_paso(camara, paso, i, len(PASOS_POCA_LUZ),
                                         frames, detecciones, "baja"):
                        break

    finally:

        camara.release()
        cv2.destroyAllWindows()

        pd.DataFrame(frames).to_csv(RUTA_FRAMES, index=False)
        pd.DataFrame(
            detecciones,
            columns=["paso", "frame", "clase", "confianza", "area_relativa"]
        ).to_csv(RUTA_DETECCIONES, index=False)

        print(f"Registrados {len(frames)} fotogramas en {CARPETA_CALIBRACION}")

    return True


# ============================================================
# ANALISIS
# ============================================================

def puntajes(frames, detecciones, clases):
    """
    Confianza maxima de las clases indicadas en cada fotograma
    (0 si no aparecen).
    """

    filtradas = detecciones[detecciones["clase"].isin(clases)]
    maximos = filtradas.groupby(["paso", "frame"])["confianza"].max()

    indice = pd.MultiIndex.from_frame(frames[["paso", "frame"]])

    return pd.Series(maximos.reindex(indice).fillna(0.0).values, index=frames.index)


def simular_alarma(tiempos, valores, umbral, tolerancia, tiempo_alerta):
    """
    Reproduce la logica de alarma de principal.py: el objeto debe
    mantenerse visible `tiempo_alerta` segundos; si deja de verse, el
    contador se reinicia. Con tolerancia > 0, ausencias menores a
    `tolerancia` segundos no reinician el contador.
    Devuelve el segundo en que suena la alarma, o None.
    """

    inicio = None
    ausente_desde = None

    for t, v in zip(tiempos, valores):

        if v >= umbral:

            ausente_desde = None

            if inicio is None:
                inicio = t

            if t - inicio >= tiempo_alerta:
                return round(float(t), 1)

        elif inicio is not None:

            if ausente_desde is None:
                ausente_desde = t

            if t - ausente_desde >= tolerancia:
                inicio = None
                ausente_desde = None

    return None


def parpadeos(valores, umbral):
    """
    Veces que el objeto deja de detectarse tras haberse detectado.
    """

    visible = np.asarray(valores) >= umbral

    return int(np.sum(visible[:-1] & ~visible[1:])) if len(visible) > 1 else 0


def analizar():

    frames = pd.read_csv(RUTA_FRAMES)
    detecciones = pd.read_csv(RUTA_DETECCIONES)

    frames["objetivo"] = frames["objetivo"].fillna("")

    normal = frames["luz"] == "normal"
    negativos = normal & (frames["tipo"] == "negativo")

    umbrales = dict(UMBRAL)
    resumen = {}
    alarmas = []
    tolerancia_elegida = 0.0

    for grupo, clases in GRUPOS.items():

        positivos = normal & (frames["objetivo"] == grupo)

        if not positivos.any():
            resumen[grupo] = {"estado": "sin datos (paso saltado)"}
            continue

        valor = puntajes(frames, detecciones, clases)

        pos = valor[positivos]
        neg = valor[negativos]

        umbral_original = min(UMBRAL[c] for c in clases)

        # Barrido de umbrales
        candidatos = []

        for u in np.round(np.arange(UMBRAL_MINIMO, 0.96, 0.05), 2):

            sensibilidad = float((pos >= u).mean())
            falsos = float((neg >= u).mean()) if len(neg) else 0.0

            candidatos.append((u, sensibilidad, falsos))

        # Mayor sensibilidad con a lo sumo 1% de fotogramas negativos
        # por encima del umbral; si no existe, se maximiza la diferencia.
        validos = [c for c in candidatos if c[2] <= 0.01]

        if validos:
            elegido = max(validos, key=lambda c: (c[1], c[0]))
        else:
            elegido = max(candidatos, key=lambda c: c[1] - c[2])

        umbral = float(elegido[0])

        if pos.max() < UMBRAL_MINIMO:
            estado = "el modelo de VIDEO no reconoce este objeto en esta escena"
        elif elegido[1] < 0.3:
            estado = "deteccion debil"
        else:
            estado = "ok"

        # Si no se puede calibrar, se mantiene el umbral de VIDEO
        if estado == "ok":
            for clase in clases:
                umbrales[clase] = umbral
        else:
            umbral = umbral_original

        resumen[grupo] = {
            "estado": estado,
            "umbral_original": umbral_original,
            "umbral_calibrado": umbral,
            "confianza_mediana_presente": round(float(pos[pos > 0].median()), 3) if (pos > 0).any() else 0.0,
            "confianza_maxima_ausente": round(float(neg.max()), 3) if len(neg) else 0.0,
            "fotogramas_detectado_original": round(float((pos >= umbral_original).mean()), 3),
            "fotogramas_detectado_calibrado": round(elegido[1], 3),
            "falsos_original": round(float((neg >= umbral_original).mean()), 3) if len(neg) else 0.0,
            "falsos_calibrado": round(elegido[2], 3),
            "parpadeos_original": parpadeos(pos, umbral_original),
            "parpadeos_calibrado": parpadeos(pos, umbral)
        }

    # --------------------------------------------------------
    # Simulacion de la alarma en cada paso
    # --------------------------------------------------------

    def simular_todo(usar_calibrado, tolerancia):
        """
        Para cada paso: {grupo: segundo en que suena la alarma}.
        """

        filas = []

        for paso, datos in frames.groupby("paso", sort=False):

            sonidos = {}

            for grupo, clases in GRUPOS.items():

                if usar_calibrado:
                    umbral = umbrales[next(iter(clases))]
                else:
                    umbral = min(UMBRAL[c] for c in clases)

                valor = puntajes(datos, detecciones, clases)

                t = simular_alarma(datos["t"].values, valor.values, umbral,
                                   tolerancia, TIEMPO_ALERTA)

                if t is not None:
                    sonidos[grupo] = t

            filas.append({
                "paso": paso,
                "tipo": datos["tipo"].iloc[0],
                "objetivo": datos["objetivo"].iloc[0],
                "luz": datos["luz"].iloc[0],
                "sonidos": sonidos
            })

        return filas

    def correcta(fila):
        """
        Positivo: suena la alarma de su objeto. Negativo: no suena nada.
        """

        if fila["tipo"] == "negativo":
            return not fila["sonidos"]

        return fila["objetivo"] in fila["sonidos"]

    def texto(sonidos):

        return ", ".join(f"{g} ({t}s)" for g, t in sonidos.items()) or "-"

    calibrables = {g for g, r in resumen.items() if r.get("estado") == "ok"}

    # Menor tolerancia con la que todos los pasos (con luz normal y de
    # objetos calibrables) se comportan correctamente
    for tolerancia in TOLERANCIAS:

        tolerancia_elegida = tolerancia

        filas = simular_todo(True, tolerancia)

        if all(
            correcta(f) for f in filas
            if f["luz"] == "normal"
            and (f["tipo"] == "negativo" or f["objetivo"] in calibrables)
        ):
            break

    antes = simular_todo(False, 0.0)
    despues = simular_todo(True, tolerancia_elegida)

    for a, d in zip(antes, despues):
        alarmas.append({
            "paso": a["paso"],
            "tipo": a["tipo"],
            "luz": a["luz"],
            "alarma_original": texto(a["sonidos"]),
            "correcta_original": correcta(a),
            "alarma_calibrada": texto(d["sonidos"]),
            "correcta_calibrada": correcta(d)
        })

    # Persona (informativo)
    persona = puntajes(frames, detecciones, {"persona"})
    tasa_persona = round(float((persona[normal] >= UMBRAL["persona"]).mean()), 3)

    resultado = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "camara": CAMARA,
        "fotogramas": int(len(frames)),
        "fps_aproximado": round(float(frames.groupby("paso")["frame"].count().sum()
                                      / frames.groupby("paso")["t"].max().sum()), 1),
        "tiempo_alerta": TIEMPO_ALERTA,
        "tolerancia_seg": tolerancia_elegida,
        "umbrales": umbrales,
        "persona_detectada_en_fotogramas": tasa_persona,
        "resumen": resumen,
        "alarma_por_paso": alarmas
    }

    with open(RUTA_RESULTADO, "w", encoding="utf-8") as archivo:
        json.dump(resultado, archivo, indent=2, ensure_ascii=False)

    # --------------------------------------------------------
    # Reporte
    # --------------------------------------------------------

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    print("\n=== CALIBRACION POR OBJETO ===")
    print(pd.DataFrame(resumen).T.to_string())

    print(f"\nTolerancia a parpadeos elegida: {tolerancia_elegida} s")
    print(f"Persona detectada en {tasa_persona:.0%} de los fotogramas")

    print("\n=== ALARMA POR PASO (antes -> despues de calibrar) ===")
    print(pd.DataFrame(alarmas).to_string(index=False))

    print(f"\nGuardado en: {RUTA_RESULTADO}")

    return resultado


if __name__ == "__main__":

    if "--analizar" not in sys.argv:
        if not capturar():
            sys.exit(0)

    analizar()
