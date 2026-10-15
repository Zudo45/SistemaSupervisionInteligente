"""API local de demostración: sin persistencia, perfiles públicos ni datos reales."""
import ipaddress
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from .seguimiento import analizar_estudiantes
from .seguimiento_contrato import EntradaSeguimiento, SalidaSeguimiento


class RutaDemoLocal(APIRoute):
    def get_route_handler(self):
        manejar = super().get_route_handler()

        async def protegido(request: Request):
            host = request.client.host if request.client else ""
            try:
                local = ipaddress.ip_address(host).is_loopback
            except ValueError:
                local = host == "testclient"  # Transporte en memoria de las pruebas.
            if not local or any(h in request.headers for h in ("forwarded", "x-forwarded-for", "x-real-ip")):
                return JSONResponse(status_code=403, content={"detail": "Seguimiento disponible solo como demo sintética local; requiere autenticación antes de un despliegue público."}, headers={"Cache-Control": "no-store"})
            cuerpo = bytearray()
            async for bloque in request.stream():
                cuerpo.extend(bloque)
                if len(cuerpo) > 2 * 1024 * 1024:
                    return JSONResponse(status_code=413, content={"detail": "El JSON supera 2 MB"}, headers={"Cache-Control": "no-store"})
            request._body = bytes(cuerpo)  # Reutiliza la lectura acotada en la validación de FastAPI.
            try:
                respuesta = await manejar(request)
            except RequestValidationError as error:
                # No reflejar el cuerpo ni los valores personales del input en errores.
                detalle = [{"campo": ".".join(map(str, e["loc"])), "mensaje": e["msg"]} for e in error.errors()]
                respuesta = JSONResponse(status_code=422, content={"detail": detalle})
            except HTTPException as error:
                respuesta = JSONResponse(status_code=error.status_code, content={"detail": error.detail})
            respuesta.headers["Cache-Control"] = "no-store"
            return respuesta

        return protegido


router = APIRouter(tags=["Seguimiento sintético por estudiante"], route_class=RutaDemoLocal)


@router.post("/prediccion/estudiantes", response_model=SalidaSeguimiento,
             summary="Evaluar seguimiento por estudiante (demo sintética local)")
async def seguimiento(entrada: EntradaSeguimiento):
    """No almacena historiales ni envía avisos. Identidad requiere referencias explícitas.

    AUDIO permanece pendiente. No se aceptan datos reales ni conexiones remotas.
    """
    try:
        return await run_in_threadpool(analizar_estudiantes, entrada)
    except ValidationError as error:
        detalle = [{"campo": ".".join(map(str, e["loc"])), "mensaje": e["msg"]} for e in error.errors()]
        raise HTTPException(422, detalle) from None
    except ValueError as error:
        raise HTTPException(422, str(error)) from None


@router.get("/prediccion/estudiantes/ejemplos/{caso}", summary="Obtener un escenario sintético local")
async def ejemplo_seguimiento(caso: str):
    from pathlib import Path
    permitidos = {"juan", "sin_alerta", "identidad_desconocida", "insuficientes",
                  "contradictorio", "duplicados", "aislado", "cambio_patron"}
    if caso not in permitidos:
        raise HTTPException(404, "Ejemplo no disponible")
    ruta = Path(__file__).resolve().parents[3] / "ejemplos" / f"seguimiento_{caso}_sintetico.json"
    return json.loads(ruta.read_text(encoding="utf-8"))
