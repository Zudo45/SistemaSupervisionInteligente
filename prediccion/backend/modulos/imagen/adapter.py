"""Convierte la exportación JSON de IMAGEN al contrato de PREDICCIÓN."""
from datetime import datetime
from math import isfinite
from hashlib import sha256
import json

from backend.modulos.video.adapter import ALIAS


ESTADOS_VALIDOS = {"confirmada", "recuperada"}


def _numero(valor, campo, minimo=0, maximo=1):
    if isinstance(valor, bool):
        raise ValueError(f"imagen.{campo}: se requiere un número")
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        raise ValueError(f"imagen.{campo}: se requiere un número") from None
    if not isfinite(numero) or not minimo <= numero <= maximo:
        raise ValueError(f"imagen.{campo}: fuera del intervalo [{minimo}, {maximo}]")
    return numero


def _segundos(frames, origen_temporal=None, exigir_origen_temporal=False):
    valores = [frame.get("segundo", frame.get("timestamp")) for frame in frames]
    if all(isinstance(valor, (int, float)) and not isinstance(valor, bool) for valor in valores):
        return [_numero(valor, "timestamp", 0, 86400) for valor in valores]
    if not all(isinstance(valor, str) and valor.strip() for valor in valores):
        raise ValueError("imagen.timestamp debe contener segundos o fechas ISO consistentes; no se admiten vacíos ni mezclas")
    if exigir_origen_temporal and origen_temporal is None:
        raise ValueError("imagen.origen_temporal: se requiere una fecha ISO explícita para sincronizar con VIDEO")
    try:
        fechas = [datetime.fromisoformat(str(valor).replace("Z", "+00:00")) for valor in valores]
        origen = (datetime.fromisoformat(str(origen_temporal).replace("Z", "+00:00"))
                  if origen_temporal is not None else min(fechas))
        segundos = [(fecha - origen).total_seconds() for fecha in fechas]
    except (TypeError, ValueError):
        raise ValueError("imagen.timestamp y origen_temporal deben ser fechas ISO con zonas horarias consistentes") from None
    return [_numero(valor, "timestamp (segundos; máximo 86400)", 0, 86400) for valor in segundos]


def _confianza(objeto):
    for campo in ("confianza_procesada", "confianza_capturada", "confianza_video", "confianza"):
        if objeto.get(campo) is not None:
            return _numero(objeto[campo], campo)
    raise ValueError("Cada objeto de imagen requiere una confianza")


def adaptar_imagen(datos, *, exigir_origen_temporal=False):
    if not isinstance(datos, dict):
        raise ValueError("imagen debe ser un objeto")
    if "frames" in datos:
        frames = datos["frames"]
    elif all(campo in datos for campo in ("frame_id", "timestamp", "objetos")):
        frames = [datos]
    else:
        raise ValueError("imagen requiere frames o un fotograma con frame_id, timestamp y objetos")
    if not isinstance(frames, list):
        raise ValueError("imagen.frames debe ser una lista")
    if not all(isinstance(frame, dict) for frame in frames):
        raise ValueError("Cada elemento de imagen.frames debe ser un objeto")
    sesion_id = datos.get("sesion_id")
    if sesion_id is not None and (not isinstance(sesion_id, str) or not sesion_id.strip()):
        raise ValueError("imagen.sesion_id debe ser un texto no vacío")
    segundos = _segundos(frames, datos.get("origen_temporal"), exigir_origen_temporal) if frames else []
    modelo = str(datos.get("modelo", "SistemaSupervisionExamenes-Imagen"))[:200]
    eventos = []
    for indice_frame, (frame, segundo) in enumerate(zip(frames, segundos)):
        frame_id = str(frame.get("frame_id", indice_frame))
        # El productor reutiliza IDs por frame: aislar origen, tiempo y sesión.
        if "origen" in frame or sesion_id is not None:
            contexto_id = json.dumps([sesion_id, frame.get("origen"),
                                      frame.get("segundo", frame.get("timestamp"))], ensure_ascii=False)
            frame_id = f"{sha256(contexto_id.encode('utf-8')).hexdigest()[:24]}:{frame_id}"
        calidad = frame.get("calidad_normalizada", 1)
        calidad = _numero(calidad, "calidad_normalizada")
        objetos = frame.get("objetos", [])
        if not isinstance(objetos, list):
            raise ValueError("imagen.frames[].objetos debe ser una lista")
        for indice, objeto in enumerate(objetos):
            if not isinstance(objeto, dict):
                raise ValueError("Cada objeto detectado por imagen debe ser un objeto JSON")
            estado = objeto.get("estado")
            if estado is not None and estado not in ESTADOS_VALIDOS:
                continue
            clase = objeto.get("clase")
            if not isinstance(clase, str) or not clase.strip():
                raise ValueError("Cada objeto de imagen requiere clase")
            identificador = str(objeto.get("id", f"{frame_id}:{indice}"))
            eventos.append({
                "id": f"imagen:{frame_id}:{identificador}",
                "modalidad": "imagen",
                "senal": ALIAS.get(clase.strip().lower(), clase.strip().lower()),
                "segundo": segundo,
                "confianza": _confianza(objeto),
                "calidad": calidad,
                "modelo": modelo,
            })
    return eventos
