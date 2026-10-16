# Integración con IMAGEN

Fuente externa: [Zudo45/SistemaSupervisionExamenes-Imagen](https://github.com/Zudo45/SistemaSupervisionExamenes-Imagen).

El equipo de IMAGEN conserva su pipeline, entrenamiento, interfaz y dependencias. Este directorio contiene únicamente `adapter.py`, que lee su exportación JSON por fotograma y transforma objetos confirmados o recuperados en eventos para PREDICCIÓN.

PREDICCIÓN no vuelve a ejecutar YOLO ni modifica métricas de IMAGEN. `calidad_normalizada`, cuando se suministra, debe estar entre 0 y 1; las métricas sin calibrar de brillo, ruido o nitidez se conservan en el sistema de origen y no se convierten automáticamente.

Se admite el acumulado `frames` y el JSON individual del vigilante. Formatos
desconocidos se rechazan. Sin `calidad_normalizada`, el factor es 1 provisional.
Con VIDEO, las fechas requieren `origen_temporal` explícito; sin VIDEO se
mantiene la conversión relativa al mínimo del lote. Todos los segundos se validan
en 0–86400. `sesion_id`, origen y tiempo delimitan IDs para evitar colisiones.

Referencia inspeccionada: commit `eddc6234`, 8 de octubre de 2026. No exporta
emociones ni identidad. Ejemplo sintético: `ejemplos/imagen_exportacion_sintetica.json`;
pendiente de validar contra una exportación real del equipo. Detalles completos
en `docs/CONTRATO_MODULOS.md`.
