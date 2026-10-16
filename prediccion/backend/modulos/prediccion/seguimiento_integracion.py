"""Puente de resultados externos al motor B; AUDIO permanece pendiente."""
from datetime import datetime, timedelta

from backend.modulos.imagen.adapter import adaptar_imagen
from backend.modulos.video.adapter import adaptar_video
from .seguimiento_contrato import EntradaSeguimiento, Observacion


def reunir_observaciones(entrada: EntradaSeguimiento):
    sesiones = {s.sesion_id: s for s in entrada.sesiones}
    observaciones = list(entrada.observaciones) + list(entrada.historial)
    if any(o.modulo == "audio" for o in observaciones):
        raise ValueError("AUDIO pendiente: no existe contrato productor ni adaptador; no se admiten detecciones acústicas")
    for fuente in entrada.fuentes:
        if fuente.modulo == "audio":
            raise ValueError("AUDIO pendiente: no existe contrato productor ni adaptador")
        if fuente.sesion_id not in sesiones:
            raise ValueError("Fuente de resultados referencia una sesión desconocida")
        sesion = sesiones[fuente.sesion_id]
        # Una sesión explícita es el reloj común también para fuentes sueltas.
        if fuente.modulo == "imagen":
            if "origen_temporal" in fuente.datos:
                try:
                    origen = datetime.fromisoformat(str(fuente.datos["origen_temporal"]).replace("Z", "+00:00"))
                except ValueError:
                    raise ValueError("IMAGEN requiere origen_temporal ISO válido") from None
                inicio = sesion.inicio if origen.tzinfo is not None else sesion.inicio.replace(tzinfo=None)
                if origen != inicio:
                    raise ValueError("IMAGEN origen_temporal no coincide con el inicio explícito de la sesión")
            eventos = adaptar_imagen(fuente.datos, exigir_origen_temporal=True)
        else:
            eventos = adaptar_video(fuente.datos)
        conocidos = {e["id"] for e in eventos}
        if set(fuente.evidencias) - conocidos:
            raise ValueError("La referencia de correlación apunta a un evento no generado por el adaptador")
        for evento in eventos:
            observaciones.append(Observacion(
                id=evento["id"], sesion_id=fuente.sesion_id, modulo=fuente.modulo,
                segundo=evento["segundo"], fecha_hora=sesion.inicio + timedelta(seconds=evento["segundo"]),
                origen=fuente.origen, tipo="objeto", valor=evento["senal"],
                confianza=evento["confianza"], calidad=evento["calidad"],
                evidencia_id=fuente.evidencias.get(evento["id"]),
            ))
    return observaciones
