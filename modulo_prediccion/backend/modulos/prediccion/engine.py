"""Motor explicable y sin estado. Los índices son heurísticos, no probabilidades."""
from collections import defaultdict
from math import isfinite
from backend.modulos.video.adapter import ALIAS

MODALIDADES = {"imagen", "video", "audio"}
PESOS = {"telefono": .85, "audifonos": .8, "libro": .7,
         "cuaderno": .7, "reloj": .5, "multiples_personas": .65,
         "voz_adicional": .65, "consulta_externa": .8}
CONTEXTUALES = {"persona", "laptop", "mochila", "silencio", "sin_alerta"}


def numero(value, campo, minimo=0, maximo=1):
    if isinstance(value, bool):
        raise ValueError(f"{campo}: se requiere un número")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{campo}: se requiere un número") from None
    if not isfinite(value) or not minimo <= value <= maximo:
        raise ValueError(f"{campo}: fuera del intervalo [{minimo}, {maximo}]")
    return value


def predecir(datos):
    if not isinstance(datos, dict):
        raise ValueError("Se requiere un objeto JSON")
    eventos = datos.get("eventos", [])
    if not isinstance(eventos, list) or len(eventos) > 10000:
        raise ValueError("eventos debe ser una lista de hasta 10000 elementos")
    ventana = numero(datos.get("ventana_segundos", 60), "ventana_segundos", 1, 3600)
    permitidos = datos.get("objetos_permitidos", [])
    if not isinstance(permitidos, list) or not all(isinstance(x, str) for x in permitidos):
        raise ValueError("objetos_permitidos debe ser una lista de textos")
    permitidos = {ALIAS.get(x.lower(), x.lower()) for x in permitidos}
    normalizados = []
    ids = set()
    for evento in eventos:
        if not isinstance(evento, dict):
            raise ValueError("Cada evento debe ser un objeto")
        modalidad = evento.get("modalidad")
        if modalidad not in MODALIDADES:
            raise ValueError("modalidad debe ser imagen, video o audio")
        identificador = evento.get("id")
        if not isinstance(identificador, str) or not identificador or len(identificador) > 200:
            raise ValueError("Cada evento requiere un id de texto, de hasta 200 caracteres")
        if identificador in ids:
            continue
        ids.add(identificador)
        senal = evento.get("senal")
        if not isinstance(senal, str) or not senal:
            raise ValueError("Cada evento requiere una senal de texto")
        senal = ALIAS.get(senal.lower(), senal.lower())
        normalizados.append({
            "id": identificador, "modalidad": modalidad, "senal": senal,
            "segundo": numero(evento.get("segundo"), "segundo", 0, 86400),
            "confianza": numero(evento.get("confianza"), "confianza"),
            "calidad": numero(evento.get("calidad", 1), "calidad"),
            "modelo": str(evento.get("modelo", "no_especificado"))[:200],
        })
    ahora = numero(datos.get("segundo_actual", max((e["segundo"] for e in normalizados), default=0)),
                   "segundo_actual", 0, 86400)
    recientes = [e for e in normalizados if 0 <= ahora - e["segundo"] <= ventana]
    grupos = defaultdict(list)
    descartados = []
    desconocidas = set()
    disponibles = set()
    for e in recientes:
        if e["calidad"] < .4 or e["confianza"] < .5:
            descartados.append(e["id"])
            continue
        if e["senal"] not in PESOS and e["senal"] not in CONTEXTUALES:
            desconocidas.add(e["senal"])
            continue
        disponibles.add(e["modalidad"])
        if e["senal"] in PESOS and e["senal"] not in permitidos:
            grupos[e["senal"]].append(e)
    evidencia = []
    for senal, grupo in sorted(grupos.items()):
        # Imagen y video pueden describir el mismo fotograma: no sumarlos como evidencia independiente.
        fuerza = max(e["confianza"] * e["calidad"] *
                     (1 - .5 * (ahora - e["segundo"]) / ventana) for e in grupo)
        segundos = {int(e["segundo"]) for e in grupo}
        persistencia = len(segundos) >= 3 and max(segundos) - min(segundos) >= 5
        score = min(1, PESOS[senal] * fuerza * (1.1 if persistencia else 1))
        evidencia.append({"senal": senal, "indice": round(score * 100, 2),
                          "persistente": persistencia, "modalidades": sorted({e["modalidad"] for e in grupo}),
                          "eventos": [e["id"] for e in grupo],
                          "explicacion": "Señal repetida durante al menos 5 segundos" if persistencia else "Señal puntual; requiere verificar contexto"})
    # Máximo conservador: evita elevar el riesgo por detecciones correlacionadas.
    indice = max((e["indice"] for e in evidencia), default=0) if disponibles else None
    nivel = "datos_insuficientes" if indice is None else (
        "alto" if indice >= 75 else "moderado" if indice >= 40 else "bajo")
    return {
        "version_motor": "1.1.0", "metodo": "reglas_explicables_no_calibradas",
        "indice_revision": indice, "nivel_revision": nivel,
        "requiere_revision_humana": bool(evidencia),
        "evidencia": evidencia, "modalidades_disponibles": sorted(disponibles),
        "modalidades_faltantes": sorted(MODALIDADES - disponibles),
        "eventos_descartados": descartados, "senales_sin_adaptador": sorted(desconocidas),
        "ventana": {"desde": max(0, ahora - ventana), "hasta": ahora},
        "recomendacion": "Verificar las señales y las reglas del examen con el docente" if evidencia else (
            "Solicitar resultados válidos a los módulos de origen" if indice is None else "Continuar observación; ausencia de alertas no demuestra ausencia de incidentes"),
        "limitaciones": ["El índice no es una probabilidad de fraude ni una predicción de notas"],
    }
