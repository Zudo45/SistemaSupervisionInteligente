"""Rutas del motor multimodal; integradas por backend.app."""
import json

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from .engine import predecir
from .integracion import integrar_fuentes

router = APIRouter(tags=["Predicción multimodal"])


@router.get("/salud")
def salud():
    return {"estado": "ok", "motor_listo": True, "requiere_pesos": False}


@router.post("/prediccion")
async def prediccion(request: Request):
    return await _procesar(request, predecir)


@router.post("/prediccion/integrar")
async def integrar(request: Request):
    return await _procesar(request, integrar_fuentes)


async def _procesar(request, funcion):
    contenido = bytearray()
    async for fragmento in request.stream():
        contenido.extend(fragmento)
        if len(contenido) > 2 * 1024 * 1024:
            raise HTTPException(413, "El JSON supera 2 MB")
    try:
        datos = json.loads(contenido)
        return await run_in_threadpool(funcion, datos)
    except (ValueError, UnicodeDecodeError) as error:
        raise HTTPException(422, str(error)) from error
