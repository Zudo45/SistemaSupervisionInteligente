"""
Paquete Oficial de Entrenamiento e Inferencia Predictiva (AulaPredict v2.5)
===========================================================================

Este paquete contiene los modelos entrenados y los algoritmos de inferencia
multimodal para la predicción de rendimiento, emociones y sucesos académicos.

Exportaciones Principales:
- procesar_captura_imagen(imagen_b64, contexto_externo=None)
    Detecta automáticamente los alumnos presentes en la imagen capturada,
    evalúa sus 7 emociones universales (neutral, felicidad, tristeza, enojo,
    miedo, sorpresa, disgusto) y predice causas previas, consecuencias futuras
    y probabilidad de desaprobar ("jalar") el examen parcial.
- predecir_estudiante_multimodal(id_estudiante, rasgos_faciales, bounding_box, contexto_alumno=None)
    Inferencia directa para un alumno individual a partir de sus rasgos biométricos y contexto.
- obtener_metricas_modelo()
    Devuelve las métricas de rendimiento y validación de los modelos (99.02% Accuracy, R²=0.902).
- obtener_escenario_aula_ejemplo()
    Devuelve una simulación calibrada de aula universitaria con 5 estudiantes.
"""

from .predictor import (
    procesar_captura_imagen,
    predecir_estudiante_multimodal,
    obtener_metricas_modelo,
    obtener_escenario_aula_ejemplo,
)

__all__ = [
    "procesar_captura_imagen",
    "predecir_estudiante_multimodal",
    "obtener_metricas_modelo",
    "obtener_escenario_aula_ejemplo",
]
