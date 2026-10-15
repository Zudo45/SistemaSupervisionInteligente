# PREDICCIÓN: integración de equipos

El motor recibe resultados de inferencia, no fotos, audio ni video. `integracion.py` coordina los adaptadores de cada equipo y `engine.py` combina sus eventos mediante reglas explicables.

## Contrato de integración

`POST /api/prediccion/integrar` acepta:

- `video.frames[]`: `frame_id`, `timestamp`, `calidad` opcional y `detecciones[]` con `clase` y `confianza`.
- `imagen.frames[]`: formato exportado por IMAGEN, con `objetos[]`, estado y confianzas antes/después del procesamiento.
- `objetos_permitidos`: materiales autorizados para esa evaluación.
- `segundo_actual` y `ventana_segundos`: contexto temporal de la sesión.

El adaptador de VIDEO normaliza nombres en español e inglés y genera `multiples_personas` cuando corresponde. El adaptador de IMAGEN descarta estados `no_confirmada` y `no_detectada`. Ambos preservan la modalidad y la versión informada del modelo.

AUDIO se mantiene pendiente hasta recibir su repositorio y un ejemplo real. No se inventará su contrato.

## Criterio actual

El índice 0–100 es heurístico y no calibrado. Agrupa evidencia correlacionada usando el máximo, filtra baja confianza/calidad y distingue señales puntuales de persistentes. Un resultado alto solicita revisión humana; nunca demuestra fraude ni autoriza una sanción automática.

PREDICCIÓN no entrena modelos ni procesa medios: recibe resultados externos. Las rutas de inferencia locales anteriores están archivadas fuera del repositorio.

## Motor B: seguimiento por estudiante

`seguimiento.py` agrupa observaciones por estudiante/sesión y aplica reglas
configurables junto a contexto académico. `seguimiento_contrato.py` define
esquemas Pydantic; `seguimiento_integracion.py` reutiliza adaptadores existentes;
`seguimiento_api.py` expone una API de demostración exclusivamente local y sintética.
Los índices A/B no se suman y las rutas anteriores conservan sus respuestas.

No existe identificación biométrica: cada evento necesita asociación explícita,
verificada y referenciada. No se atribuyen detecciones por pista local o geometría.
No se guardan historiales. AUDIO sigue pendiente y emociones no asignan puntos.
Contrato, reglas y ejemplos: `docs/SEGUIMIENTO_ESTUDIANTES.md`.
