# ============================================================
# PIPELINE DEL MODULO IMAGEN
# VIDEO -> preprocesamiento -> mejoramiento -> verificacion
#       -> segmentacion -> caracteristicas -> resultado
# ============================================================

from modulo_imagen.config_imagen import CLASES_NO_PERMITIDAS
from modulo_imagen.recepcion import recibir_de_video, detectar_objetos
from modulo_imagen.preprocesamiento import preprocesar, degradar
from modulo_imagen.mejoramiento import mejorar
from modulo_imagen.calidad import medir_calidad, comparar_con_referencia
from modulo_imagen.verificacion import verificar, comparar_salidas, grupo
from modulo_imagen.segmentacion import segmentar_objeto
from modulo_imagen.caracteristicas import extraer_caracteristicas


OPCIONES_POR_DEFECTO = {
    # adaptativo: los filtros se eligen segun la calidad medida
    # manual: se usan los valores de abajo tal cual
    "modo": "adaptativo",

    # Degradacion simulada de la camara (experimentos)
    "oscuridad": 0.0,
    "ruido_simulado": 0.0,
    "desenfoque": 0,

    # Preprocesamiento
    "ruido": "mediana",
    "fuerza_ruido": 3,
    "iluminacion": "clahe",
    "limite_clahe": 1.5,

    # Mejoramiento
    "contraste": "estiramiento",
    "alfa": 1.0,
    "beta": 0,
    "nitidez": 0.5,
    "sigma_nitidez": 2.0,

    # Segmentacion
    "segmentacion": "grabcut"
}


# Umbrales del modo adaptativo
RUIDO_ALTO = 8.0
RUIDO_MEDIO = 3.0
BRILLO_BAJO = 80.0
NITIDEZ_BAJA = 50.0
NITIDEZ_MUY_BAJA = 15.0
BRILLO_REFERENCIA = 128.0
CONTRASTE_BAJO = 40.0


def decidir_opciones(calidad):
    """
    Elige los filtros de preprocesamiento y mejora a partir de
    las metricas del fotograma capturado. Devuelve las opciones
    y la lista de decisiones tomadas (para mostrarlas).
    """

    opciones = {}
    decisiones = []

    brillo = calidad["brillo"]
    nitidez = calidad["nitidez"]

    # Si la imagen es oscura, la correccion de iluminacion
    # amplificara el ruido en proporcion a la ganancia de brillo
    ganancia = BRILLO_REFERENCIA / brillo if brillo < BRILLO_BAJO else 1.0
    ruido = calidad["ruido"] * ganancia

    if ganancia > 1.0:
        decisiones.append(
            f"Ganancia de brillo esperada x{ganancia:.1f}: el ruido "
            f"efectivo pasa de {calidad['ruido']:.1f} a {ruido:.1f}"
        )

    # Ruido
    if ruido > RUIDO_ALTO:
        fuerza = int(min(30, max(5, ruido * 0.8)))
        opciones.update({"ruido": "nlmeans", "fuerza_ruido": fuerza, "nitidez": 0.0})
        decisiones.append(
            f"Ruido alto (sigma={ruido:.1f}): filtro Non-Local Means "
            f"(h={fuerza}) y sin realce de nitidez para no amplificar ruido"
        )
    elif ruido > RUIDO_MEDIO:
        opciones.update({"ruido": "bilateral", "fuerza_ruido": 7, "nitidez": 0.3})
        decisiones.append(
            f"Ruido moderado (sigma={ruido:.1f}): filtro bilateral, "
            "conserva bordes"
        )
    else:
        opciones.update({"ruido": "mediana", "fuerza_ruido": 3})
        decisiones.append(f"Ruido bajo (sigma={ruido:.1f}): filtro de mediana 3x3")

    oscura = brillo < BRILLO_BAJO
    ruidosa = ruido > RUIDO_ALTO

    # Iluminacion y contraste
    if oscura and ruidosa:
        opciones.update({
            "iluminacion": "gamma_auto+clahe", "limite_clahe": 1.0,
            "contraste": "estiramiento"
        })
        decisiones.append(
            f"Imagen oscura y ruidosa (brillo={brillo:.0f}): gamma "
            "automatica + CLAHE limitado + estiramiento de contraste"
        )
    elif oscura:
        opciones.update({
            "iluminacion": "gamma_auto+clahe", "limite_clahe": 1.0,
            "contraste": "ninguno", "nitidez": 0.3
        })
        decisiones.append(
            f"Imagen oscura (brillo={brillo:.0f}): gamma automatica + "
            "CLAHE suave"
        )
    elif ruidosa or calidad["contraste"] < CONTRASTE_BAJO:
        opciones.update({
            "iluminacion": "clahe", "limite_clahe": 1.5,
            "contraste": "estiramiento"
        })
        decisiones.append(
            f"Contraste a recuperar (contraste={calidad['contraste']:.0f}): "
            "CLAHE + estiramiento de contraste"
        )
    else:
        opciones.update({"iluminacion": "ninguno", "contraste": "ninguno"})
        decisiones.append(
            f"Iluminacion y contraste adecuados (brillo={brillo:.0f}, "
            f"contraste={calidad['contraste']:.0f}): no se modifican"
        )

    # Nitidez. En una imagen oscura la varianza del Laplaciano
    # tambien es baja, por eso solo se evalua si no es oscura.
    if not oscura and nitidez < NITIDEZ_MUY_BAJA and ruido <= RUIDO_MEDIO:
        opciones.update({"nitidez": 1.2, "sigma_nitidez": 3.0})
        decisiones.append(
            f"Imagen muy borrosa (var. Laplaciano={nitidez:.0f}): "
            "mascara de desenfoque fuerte (cantidad 1.2, radio 3)"
        )
    elif not oscura and nitidez < NITIDEZ_BAJA and ruido <= RUIDO_MEDIO:
        opciones.update({"nitidez": 0.8, "sigma_nitidez": 2.5})
        decisiones.append(
            f"Imagen borrosa (var. Laplaciano={nitidez:.0f}): "
            "mascara de desenfoque (cantidad 0.8, radio 2.5)"
        )
    elif not ruidosa and "nitidez" not in opciones:
        opciones["nitidez"] = 0.3

    return opciones, decisiones


def procesar_frame(paquete, opciones=None):
    """
    paquete: {"frame_id", "origen", "timestamp", "frame"}
    Devuelve un diccionario con imagenes intermedias y datos.
    """

    op = {**OPCIONES_POR_DEFECTO, **(opciones or {})}

    original = paquete["frame"]

    # --------------------------------------------------------
    # 0. Condiciones de captura (degradacion opcional)
    # --------------------------------------------------------

    capturado = degradar(
        original,
        oscuridad=op["oscuridad"],
        ruido=op["ruido_simulado"],
        desenfoque=op["desenfoque"]
    )

    # --------------------------------------------------------
    # 1. Recepcion: salida del grupo VIDEO
    # --------------------------------------------------------

    # Lo que VIDEO entrego: su deteccion sobre el fotograma original
    registro = recibir_de_video(paquete)

    # Salida de VIDEO sobre el fotograma que recibe IMAGEN. Solo difiere
    # de la original si se simulan malas condiciones de captura.
    simulacion = bool(op["oscuridad"] or op["ruido_simulado"] or op["desenfoque"])

    if simulacion:
        detecciones_capturado = [
            {**d, "confianza": round(float(d["confianza"]), 4)}
            for d in detectar_objetos(capturado.copy())
        ]
    else:
        detecciones_capturado = registro["detecciones"]

    calidad_capturado = medir_calidad(capturado)

    decisiones = []

    if op["modo"] == "adaptativo":
        elegidas, decisiones = decidir_opciones(calidad_capturado)
        op.update(elegidas)

    # --------------------------------------------------------
    # 2. Preprocesamiento
    # --------------------------------------------------------

    preprocesado = preprocesar(
        capturado,
        ruido=op["ruido"],
        fuerza_ruido=op["fuerza_ruido"],
        iluminacion=op["iluminacion"],
        limite_clahe=op["limite_clahe"]
    )

    # --------------------------------------------------------
    # 3. Mejoramiento
    # --------------------------------------------------------

    mejorado = mejorar(
        preprocesado,
        contraste=op["contraste"],
        alfa=op["alfa"],
        beta=op["beta"],
        nitidez=op["nitidez"],
        sigma_nitidez=op["sigma_nitidez"]
    )

    # --------------------------------------------------------
    # 4. Verificacion: mismo modelo de VIDEO antes y despues
    # --------------------------------------------------------

    verificadas, detecciones_despues = verificar(
        mejorado, registro["detecciones"], detecciones_capturado
    )

    # --------------------------------------------------------
    # 5. Segmentacion + 6. Caracteristicas
    # --------------------------------------------------------

    personas = [d for d in verificadas if d["clase"] == "persona"]

    objetos = []

    for indice, deteccion in enumerate(verificadas):

        segmento = segmentar_objeto(mejorado, deteccion, op["segmentacion"])

        objetos.append({
            "id": indice,
            "deteccion": deteccion,
            "segmento": segmento,
            "caracteristicas": extraer_caracteristicas(
                mejorado, deteccion, segmento, personas
            )
        })

    # --------------------------------------------------------
    # Calidad y resumen
    # --------------------------------------------------------

    calidad = {
        "capturado": calidad_capturado,
        "mejorado": medir_calidad(mejorado)
    }

    if simulacion:
        calidad["vs_original"] = {
            "capturado": comparar_con_referencia(original, capturado),
            "mejorado": comparar_con_referencia(original, mejorado)
        }

    estados = [o["deteccion"]["estado"] for o in objetos]

    # Los objetos no permitidos son los que reporto VIDEO
    no_permitidos = [
        o["deteccion"]["clase"] for o in objetos
        if o["deteccion"]["clase"] in CLASES_NO_PERMITIDAS
    ]

    if "alerta" in registro:
        objeto_alerta = registro["alerta"]["objeto"]
        registro["alerta"]["objeto_redetectado"] = any(
            grupo(d["clase"]) == grupo(objeto_alerta)
            for d in registro["detecciones"]
        )

    resumen = {
        "detecciones_video": len(registro["detecciones"]),
        "confirmadas": estados.count("confirmada"),
        "no_confirmadas": estados.count("no_confirmada"),
        "objetos_no_permitidos_video": sorted(set(no_permitidos)),
        "recuperadas": estados.count("recuperada"),
        "no_detectadas": estados.count("no_detectada"),
        "comparacion_video": comparar_salidas(
            detecciones_capturado, detecciones_despues
        )
    }

    return {
        "registro_video": registro,
        "opciones": op,
        "decisiones": decisiones,
        "imagenes": {
            "original": original,
            "capturado": capturado,
            "preprocesado": preprocesado,
            "mejorado": mejorado
        },
        "detecciones_capturado": detecciones_capturado,
        "detecciones_despues": detecciones_despues,
        "objetos": objetos,
        "calidad": calidad,
        "resumen": resumen
    }
