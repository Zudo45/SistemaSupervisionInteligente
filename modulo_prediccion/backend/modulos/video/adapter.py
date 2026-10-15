"""Convierte las detecciones de VIDEO al contrato común de PREDICCIÓN."""
from math import isfinite
import re


ALIAS = {
    "cell phone": "telefono",
    "celular": "telefono",
    "phone": "telefono",
    "headphones": "audifonos",
    "airpods": "audifonos",
    "watch": "reloj",
    "cuaderno/libro": "libro",
    "person": "persona",
    "backpack": "mochila",
}


def _numero(valor, campo, minimo=0, maximo=86400):
    if isinstance(valor, bool):
        raise ValueError(f"video.{campo}: se requiere un número")
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        raise ValueError(f"video.{campo}: se requiere un número") from None
    if not isfinite(numero) or not minimo <= numero <= maximo:
        raise ValueError(f"video.{campo}: fuera del intervalo [{minimo}, {maximo}]")
    return numero


def adaptar_video(datos):
    if not isinstance(datos, dict):
        raise ValueError("video debe ser un objeto")
    if "schema_version" in datos and (type(datos["schema_version"]) is not int or datos["schema_version"] != 1):
        raise ValueError("video.schema_version: solo se admite la versión 1")
    alerta = "frames" not in datos
    if alerta:
        if (datos.get("schema_version") != 1 or datos.get("modulo") != "VIDEO"
                or not isinstance(datos.get("evento_id"), str)
                or not re.fullmatch(r"[a-f0-9]{32}", datos["evento_id"])
                or not all(campo in datos for campo in ("timestamp", "detecciones", "frame_id"))):
            raise ValueError("video: se requiere frames o una alerta VIDEO schema_version 1 con evento_id, frame_id, timestamp y detecciones")
        modelos = datos.get("modelos", {})
        if not isinstance(modelos, dict):
            raise ValueError("video.modelos debe ser un objeto")
        modelo = ", ".join(sorted(modelos))[:200] or "no_especificado"
        frames = [datos]
    else:
        frames = datos["frames"]
        modelo = str(datos.get("modelo", "SistemaSupervisionExamenes/best.pt"))[:200]
    if not isinstance(frames, list):
        raise ValueError("video.frames debe ser una lista")
    eventos = []
    for indice_frame, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError("Cada elemento de video.frames debe ser un objeto")
        frame_id = str(frame.get("frame_id", indice_frame))
        if alerta:
            frame_id = f"alerta:{datos['evento_id']}"
        segundo = _numero(frame.get("timestamp") if alerta else
                          frame.get("segundo", frame.get("timestamp")), "timestamp (segundos; máximo 86400)")
        calidad = 1.0 if alerta else _numero(frame.get("calidad", 1), "calidad", 0, 1)
        detecciones = frame.get("detecciones", [])
        if not isinstance(detecciones, list):
            raise ValueError("video.frames[].detecciones debe ser una lista")
        personas = []
        for indice, deteccion in enumerate(detecciones):
            if not isinstance(deteccion, dict):
                raise ValueError("Cada detección de video debe ser un objeto")
            clase = deteccion.get("clase")
            if not isinstance(clase, str) or not clase.strip():
                raise ValueError("Cada detección de video requiere clase")
            senal = ALIAS.get(clase.strip().lower(), clase.strip().lower())
            confianza = _numero(deteccion.get("confianza"), "confianza", 0, 1)
            identificador = (f"{frame_id}:deteccion:{indice}" if alerta else
                             str(deteccion.get("id", f"{frame_id}:{indice}")))
            eventos.append({
                "id": f"video:{identificador}",
                "modalidad": "video",
                "senal": senal,
                "segundo": segundo,
                "confianza": confianza,
                "calidad": calidad,
                "modelo": modelo,
            })
            if senal == "persona":
                personas.append(confianza)
        if len(personas) > 1:
            personas.sort(reverse=True)
            eventos.append({
                "id": f"video:{frame_id}:multiples_personas",
                "modalidad": "video",
                "senal": "multiples_personas",
                "segundo": segundo,
                "confianza": personas[1],
                "calidad": calidad,
                "modelo": modelo,
            })
    return eventos
