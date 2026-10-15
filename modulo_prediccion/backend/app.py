import json
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.modulos.prediccion.api import router as motor_router
from backend.modulos.prediccion.seguimiento_api import router as seguimiento_router

app = FastAPI(title="AulaPredict · Predicción", version="3.1.0")
origenes_frontend = os.getenv(
    "FRONTEND_ORIGINS",
    "http://localhost:8001,http://127.0.0.1:8001,https://prediccion-frontend.onrender.com",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origen.strip() for origen in origenes_frontend if origen.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(motor_router, prefix="/api")
app.include_router(seguimiento_router, prefix="/api")


@app.get("/api/modulos")
def modulos():
    salida = []
    for nombre in ("imagen", "video", "audio", "prediccion"):
        ruta = RAIZ / "backend" / "modulos" / nombre / "modulo.json"
        config = json.loads(ruta.read_text(encoding="utf-8"))
        config["carpeta"] = f"backend/modulos/{nombre}"
        config["archivos_presentes"] = [
            artefacto for artefacto in config["artefactos"]
            if (RAIZ / artefacto).is_file()
        ]
        salida.append(config)
    return {"modulos": salida}


from fastapi import HTTPException, Request
from starlette.concurrency import run_in_threadpool
from backend.modulos.vision_service import procesar_frame, _obtener_ruta_modelo


@app.get("/api/vision/estado")
def vision_estado():
    ruta_yolo = _obtener_ruta_modelo()
    return {
        "estado": "listo",
        "modelo_yolo_disponible": bool(ruta_yolo and ruta_yolo.is_file()),
        "ruta_modelo_yolo": str(ruta_yolo) if ruta_yolo else None,
        "mejora_imagen": "CLAHE + Unsharp + Contraste activo",
        "seguimiento_facial": "HaarCascade + Geometría + Landmarks",
        "emociones": ["alegria", "neutral", "tristeza", "estres", "fatiga"],
        "prediccion_academica": "Escala 0-20 explicable"
    }


@app.post("/api/vision/analizar")
async def vision_analizar(request: Request):
    try:
        cuerpo = await request.json()
    except Exception:
        raise HTTPException(400, "Cuerpo JSON inválido")

    imagen_b64 = cuerpo.get("imagen_b64")
    if not imagen_b64 or not isinstance(imagen_b64, str):
        raise HTTPException(422, "Se requiere imagen_b64 en formato base64")

    modo = cuerpo.get("modo", "en_vivo")
    aplicar_mejora = bool(cuerpo.get("aplicar_mejora", True))
    segundos = float(cuerpo.get("segundos_distraccion", 0.0))

    try:
        return await run_in_threadpool(
            procesar_frame, imagen_b64, modo, aplicar_mejora, segundos
        )
    except Exception as e:
        raise HTTPException(500, f"Error en procesamiento de visión: {str(e)}")


app.mount("/", StaticFiles(directory=RAIZ / "frontend", html=True), name="prueba-local")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8001, reload=False)
