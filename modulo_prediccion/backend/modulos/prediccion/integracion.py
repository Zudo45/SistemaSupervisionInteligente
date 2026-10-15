"""Une las salidas de los equipos sin ejecutar ni entrenar sus modelos."""
from backend.modulos.imagen.adapter import adaptar_imagen
from backend.modulos.video.adapter import adaptar_video

from .engine import predecir


def integrar_fuentes(datos):
    if not isinstance(datos, dict):
        raise ValueError("Se requiere un objeto JSON")
    eventos = []
    fuentes = []
    if datos.get("video") is not None:
        eventos.extend(adaptar_video(datos["video"]))
        fuentes.append("video")
    if datos.get("imagen") is not None:
        eventos.extend(adaptar_imagen(datos["imagen"], exigir_origen_temporal=datos.get("video") is not None))
        fuentes.append("imagen")
    if datos.get("audio") is not None:
        raise ValueError("El adaptador de audio está pendiente hasta recibir su repositorio y contrato")

    entrada = {
        "eventos": eventos,
        "ventana_segundos": datos.get("ventana_segundos", 60),
        "objetos_permitidos": datos.get("objetos_permitidos", []),
    }
    if datos.get("segundo_actual") is not None:
        entrada["segundo_actual"] = datos["segundo_actual"]
    resultado = predecir(entrada)
    resultado["integracion"] = {
        "fuentes_recibidas": fuentes,
        "eventos_generados": len(eventos),
        "audio": "pendiente_repositorio",
    }
    return resultado
