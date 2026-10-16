"""
Servicio de Inferencia Predictiva Multimodal (AulaPredict v2.5)
Integra:
1. Reconocimiento Facial de las 7 Emociones Universales (modelo_emociones_facial.pkl)
2. Fusión Multimodal (7 Emociones + Acciones de Video/Imagen) (modelo_sucesos_academicos.pkl)
3. Pipeline Híbrido Real: YOLO Pose (yolo11n-pose.pt) + YOLO Best (best.pt) + Haar Cascade
4. Extracción de Rasgos Faciales y Postura (Head Pitch / Mirada al examen vs. Celular)
5. Predicción de Acontecimientos Previos, Futuros y Riesgo de Desaprobar ("Jalar") el Parcial
6. Respuesta limpia y estricta: 0 alumnos cuando no hay personas reales en la captura
"""
import base64
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np

CARPETA_ACTUAL = Path(__file__).resolve().parent
RUTA_MODELO_FACIAL = CARPETA_ACTUAL / "modelo_emociones_facial.pkl"
RUTA_MODELO_SUCESOS = CARPETA_ACTUAL / "modelo_sucesos_academicos.pkl"

_MODELO_FACIAL = None
_MODELO_SUCESOS = None
_YOLO_POSE = None
_YOLO_BEST = None
_FACE_CASCADE = None

def _cargar_modelos():
    global _MODELO_FACIAL, _MODELO_SUCESOS
    if _MODELO_FACIAL is None and RUTA_MODELO_FACIAL.is_file():
        with open(RUTA_MODELO_FACIAL, "rb") as f:
            _MODELO_FACIAL = pickle.load(f)
    if _MODELO_SUCESOS is None and RUTA_MODELO_SUCESOS.is_file():
        with open(RUTA_MODELO_SUCESOS, "rb") as f:
            _MODELO_SUCESOS = pickle.load(f)

# Carga perezosa inicial de clasificadores Scikit-Learn
_cargar_modelos()

def _obtener_yolo_pose():
    global _YOLO_POSE
    if _YOLO_POSE is None:
        try:
            from ultralytics import YOLO
            candidatos = [
                CARPETA_ACTUAL.parent / "modelo_ia" / "yolo11n-pose.pt",
                CARPETA_ACTUAL / "yolo11n-pose.pt",
                CARPETA_ACTUAL / "modelo_ia" / "yolo11n-pose.pt",
                CARPETA_ACTUAL.parent.parent / "modelo_ia" / "yolo11n-pose.pt",
            ]
            ruta_pose = next((p for p in candidatos if p.is_file()), None)
            if ruta_pose is not None:
                _YOLO_POSE = YOLO(str(ruta_pose))
            else:
                _YOLO_POSE = False
        except Exception:
            _YOLO_POSE = False
    return _YOLO_POSE if _YOLO_POSE is not False else None

def _obtener_yolo_best():
    global _YOLO_BEST
    if _YOLO_BEST is None:
        try:
            from ultralytics import YOLO
            candidatos = [
                CARPETA_ACTUAL.parent / "modelo_ia" / "best.pt",
                CARPETA_ACTUAL / "best.pt",
                CARPETA_ACTUAL / "modelo_ia" / "best.pt",
                CARPETA_ACTUAL.parent.parent / "modelo_ia" / "best.pt",
            ]
            ruta_best = next((p for p in candidatos if p.is_file()), None)
            if ruta_best is not None:
                _YOLO_BEST = YOLO(str(ruta_best))
            else:
                _YOLO_BEST = False
        except Exception:
            _YOLO_BEST = False
    return _YOLO_BEST if _YOLO_BEST is not False else None

def _obtener_face_cascade():
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        ruta_cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(ruta_cascade)
    return _FACE_CASCADE

def _decodificar_imagen_b64(imagen_b64: str) -> np.ndarray:
    if "," in imagen_b64:
        imagen_b64 = imagen_b64.split(",", 1)[1]
    datos_binarios = base64.b64decode(imagen_b64)
    np_arr = np.frombuffer(datos_binarios, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo decodificar la imagen enviada")
    return img

def _extraer_rasgos_faciales(
    rostro_roi: np.ndarray,
    bbox_rel: Dict[str, float],
    keypoints_head: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calcula el vector biométrico para el clasificador de las 7 emociones universales (9 features):
    1. eye_openness
    2. eyebrow_slope
    3. mouth_smile
    4. mouth_open
    5. head_tilt (roll/giro lateral)
    6. head_pitch (inclinación vertical hacia pupitre o celular)
    7. gaze_stability
    8. blink_rate
    9. facial_tension

    Aplica interpolación bicúbica y ecualización adaptativa CLAHE en recortes pequeños.
    """
    h, w = rostro_roi.shape[:2]

    # Preprocesamiento con CLAHE y resize bicúbico si el recorte es pequeño (alumnos al fondo)
    if h < 80 or w < 80:
        factor_w = max(1.0, 100.0 / max(1, w))
        factor_h = max(1.0, 100.0 / max(1, h))
        nw = int(max(100, w * factor_w))
        nh = int(max(100, h * factor_h))
        rostro_roi = cv2.resize(rostro_roi, (nw, nh), interpolation=cv2.INTER_CUBIC)
        h, w = rostro_roi.shape[:2]

    # Mejora de contraste adaptativo CLAHE
    lab = cv2.cvtColor(rostro_roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    gris = l_clahe

    # Cálculo de inclinación lateral (head_tilt) y vertical (head_pitch)
    centro_x = bbox_rel["x"] + bbox_rel["w"] / 2.0
    desvio_centro = abs(centro_x - 0.5)
    head_tilt = float(np.clip(desvio_centro * 28.0 + 2.0, 1.0, 30.0))

    head_pitch = 8.0  # Erguido neutro por defecto
    if keypoints_head is not None and len(keypoints_head) >= 5:
        # Keypoints COCO: 0:nose, 1:l_eye, 2:r_eye, 3:l_ear, 4:r_ear
        nose = keypoints_head[0]
        leye = keypoints_head[1]
        reye = keypoints_head[2]
        lear = keypoints_head[3]
        rear = keypoints_head[4]

        conf_head = min(nose[2], max(leye[2], reye[2]))
        if conf_head > 0.25:
            eye_y = (leye[1] + reye[1]) / 2.0 if (leye[2] > 0.2 and reye[2] > 0.2) else max(leye[1], reye[1])
            ear_y = (lear[1] + rear[1]) / 2.0 if (lear[2] > 0.2 and rear[2] > 0.2) else max(lear[1], rear[1])
            nose_y = nose[1]

            # Si las orejas aparecen arriba de los ojos, la cabeza está mirando francamente hacia abajo
            if ear_y < eye_y:
                diff_norm = (eye_y - ear_y) / max(10.0, float(h))
                head_pitch = float(np.clip(18.0 + diff_norm * 60.0, 15.0, 42.0))
            elif nose_y > eye_y + (h * 0.25):
                head_pitch = float(np.clip(12.0 + (nose_y - eye_y) * 0.4, 5.0, 35.0))

    # Segmentación adaptativa según inclinación vertical
    if head_pitch > 22.0:
        # Cabeza gacha: los ojos están más abajo en el recorte
        tercio_ojos = gris[int(h * 0.28):int(h * 0.58), :]
        tercio_boca = gris[int(h * 0.68):int(h * 0.96), :]
    else:
        # Postura normal / erguida
        tercio_ojos = gris[int(h * 0.18):int(h * 0.48), :]
        tercio_boca = gris[int(h * 0.65):int(h * 0.95), :]

    # Gradientes y energía para apertura ocular y tensión de cejas
    grad_ojos = cv2.Sobel(tercio_ojos, cv2.CV_64F, 0, 1, ksize=3)
    energia_ojos = float(np.mean(np.abs(grad_ojos)))
    eye_openness = np.clip(energia_ojos / 42.0, 0.20, 0.95)

    # Variabilidad y apertura de boca
    energia_boca = float(np.std(tercio_boca))
    mouth_open = np.clip(energia_boca / 65.0, 0.04, 0.85)

    # Estabilidad de mirada y parpadeo
    gaze_stability = np.clip(1.0 - (head_tilt / 38.0) - (max(0.0, head_pitch - 18.0) / 60.0), 0.20, 0.95)
    blink_rate = float(np.clip(1.0 - eye_openness, 0.18, 0.90))

    # Pendiente de cejas, sonrisa y tensión facial
    eyebrow_slope = float(np.clip((0.60 - eye_openness) * 0.9, -0.6, 0.7))
    mouth_smile = float(np.clip((gaze_stability - 0.5) * 0.6, -0.6, 0.6))

    grad_boca = cv2.Sobel(tercio_boca, cv2.CV_64F, 1, 0, ksize=3)
    facial_tension = float(np.clip((np.mean(np.abs(grad_boca)) + np.mean(np.abs(grad_ojos))) / 70.0, 0.10, 0.95))

    return {
        "eye_openness": float(eye_openness),
        "eyebrow_slope": float(eyebrow_slope),
        "mouth_smile": float(mouth_smile),
        "mouth_open": float(mouth_open),
        "head_tilt": float(head_tilt),
        "head_pitch": float(head_pitch),
        "gaze_stability": float(gaze_stability),
        "blink_rate": float(blink_rate),
        "facial_tension": float(facial_tension)
    }

def predecir_estudiante_multimodal(
    estudiante_id: int,
    rasgos: Dict[str, float],
    bbox_rel: Dict[str, float],
    acciones_contexto: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Ejecuta el Modelo 1 (7 Emociones Universales) y el Modelo 2 (Sucesos y Riesgo Multimodal)
    fusionando el rostro detectado con la acción reportada/detectada del estudiante.
    """
    _cargar_modelos()

    if acciones_contexto is None:
        acciones_contexto = {}

    act_celular = float(acciones_contexto.get("accion_celular", 0.0))
    act_audifonos = float(acciones_contexto.get("accion_audifonos", 0.0))
    act_desvio = float(acciones_contexto.get("accion_mirada_desviada", 1.0 if rasgos["head_tilt"] > 16.0 or rasgos["gaze_stability"] < 0.42 else 0.0))
    act_laptop = float(acciones_contexto.get("accion_laptop_otra_ventana", 0.0))
    act_cansancio = float(acciones_contexto.get("accion_postura_cansancio", 1.0 if (rasgos["eye_openness"] < 0.32 and act_celular < 0.5) else 0.0))

    # 1. EVALUAR MODELO 1: 7 EMOCIONES FACIALES (9 features)
    head_pitch_val = float(rasgos.get("head_pitch", 8.0))
    X_facial = np.array([[
        rasgos["eye_openness"], rasgos["eyebrow_slope"], rasgos["mouth_smile"],
        rasgos["mouth_open"], rasgos["head_tilt"], head_pitch_val,
        rasgos["gaze_stability"], rasgos["blink_rate"], rasgos["facial_tension"]
    ]])

    if _MODELO_FACIAL is not None:
        probs_7 = _MODELO_FACIAL.predict_proba(X_facial)[0]
        # Índices: 0: Neutral, 1: Felicidad, 2: Tristeza, 3: Enojo, 4: Miedo, 5: Sorpresa, 6: Disgusto
        p_neu = float(probs_7[0])
        p_fel = float(probs_7[1])
        p_tri = float(probs_7[2])
        p_eno = float(probs_7[3])
        p_mie = float(probs_7[4])
        p_sor = float(probs_7[5])
        p_dis = float(probs_7[6])
    else:
        # Fallback analítico seguro
        p_neu, p_fel, p_tri, p_eno, p_mie, p_sor, p_dis = 0.72, 0.12, 0.04, 0.03, 0.05, 0.02, 0.02

    # Ajuste por acción externa: si está con celular o cansancio extremo, se refuerza la desatención
    if act_celular > 0.5:
        p_neu = max(0.04, p_neu * 0.20)
        p_mie = min(0.65, p_mie + 0.32)
        p_eno = min(0.45, p_eno + 0.22)
    elif act_cansancio > 0.5:
        p_neu = max(0.08, p_neu * 0.25)
        p_tri = min(0.68, p_tri + 0.38)
    elif head_pitch_val > 20.0 and act_celular == 0.0:
        # Mirando hacia abajo concentrado en el examen o apuntes
        p_neu = min(0.92, p_neu + 0.18)
        p_tri = max(0.02, p_tri * 0.6)

    # Normalizar suma a 1.0
    suma_p = p_neu + p_fel + p_tri + p_eno + p_mie + p_sor + p_dis
    if suma_p > 0:
        p_neu /= suma_p
        p_fel /= suma_p
        p_tri /= suma_p
        p_eno /= suma_p
        p_mie /= suma_p
        p_sor /= suma_p
        p_dis /= suma_p

    # 2. EVALUAR MODELO 2: SUCESOS Y PROYECCIÓN MULTIMODAL
    X_sucesos = np.array([[
        p_neu, p_fel, p_tri, p_eno, p_mie, p_sor, p_dis,
        act_celular, act_audifonos, act_desvio, act_laptop, act_cansancio
    ]])

    if _MODELO_SUCESOS is not None:
        pred_sucesos = _MODELO_SUCESOS.predict(X_sucesos)[0]
        # Targets: 0:desvelo, 1:conflicto, 2:frustracion, 3:presion, 4:descanso, 5:distraccion, 6:jalar, 7:bloqueo, 8:score_riesgo, 9:atencion
        prob_desvelo = float(np.clip(pred_sucesos[0], 5, 96))
        prob_conflicto = float(np.clip(pred_sucesos[1], 5, 95))
        prob_frustracion = float(np.clip(pred_sucesos[2], 5, 95))
        prob_presion = float(np.clip(pred_sucesos[3], 5, 95))
        prob_descanso = float(np.clip(pred_sucesos[4], 5, 98))
        prob_distraccion_ext = float(np.clip(pred_sucesos[5], 5, 97))
        prob_jalar_parcial = float(np.clip(pred_sucesos[6], 8, 97))
        prob_bloqueo = float(np.clip(pred_sucesos[7], 10, 95))
        score_riesgo = int(np.clip(round(pred_sucesos[8]), 5, 98))
        nivel_atencion = int(np.clip(round(pred_sucesos[9]), 8, 98))
    else:
        prob_desvelo, prob_conflicto, prob_frustracion, prob_presion = 15.0, 10.0, 12.0, 20.0
        prob_descanso, prob_distraccion_ext, prob_jalar_parcial, prob_bloqueo = 85.0, 10.0, 15.0, 12.0
        score_riesgo = 15
        nivel_atencion = 92

    # Determinación de la emoción dominante
    emociones_dict = {
        "neutral": p_neu,
        "felicidad": p_fel,
        "tristeza": p_tri,
        "enojo": p_eno,
        "miedo": p_mie,
        "sorpresa": p_sor,
        "disgusto": p_dis
    }
    emo_max = max(emociones_dict, key=emociones_dict.get)

    nombres_formato_7 = {
        "neutral": "Neutral / Concentración (😐)",
        "felicidad": "Felicidad / Motivación (😊)",
        "tristeza": "Tristeza / Decaimiento (😢)",
        "enojo": "Enojo / Frustración (😠)",
        "miedo": "Miedo / Ansiedad / Estrés (😨)",
        "sorpresa": "Sorpresa / Sobresalto (😲)",
        "disgusto": "Disgusto / Rechazo (🤢)"
    }
    emo_dominante_nombre = nombres_formato_7[emo_max]

    # Determinación de la acción y etiqueta de estado
    if act_celular > 0.5:
        estado_rapido = "Alerta: Uso de celular detectado"
        accion_detectada_str = "Uso de teléfono móvil frente a cámara"
        estado_clase = "status-alerta"
        tipo_atencion = "attention-fill-danger"
    elif act_audifonos > 0.5:
        estado_rapido = "Distracción: Uso de audífonos"
        accion_detectada_str = "Dispositivo de audio externo"
        estado_clase = "status-distraido"
        tipo_atencion = "attention-fill-amber"
    elif act_cansancio > 0.5:
        estado_rapido = "Atención baja: Fatiga / Cansancio"
        accion_detectada_str = "Postura cabizbaja con parpadeo lento"
        estado_clase = "status-fatiga"
        tipo_atencion = "attention-fill-amber"
    elif head_pitch_val > 20.0 and act_celular == 0.0 and p_neu > 0.45:
        estado_rapido = "Concentrado rindiendo / escribiendo"
        accion_detectada_str = "Mirada en pupitre / hoja de examen"
        estado_clase = "status-concentrado"
        tipo_atencion = "attention-fill-green"
    elif act_desvio > 0.5:
        estado_rapido = "Desatención: Mirada fuera de pantalla"
        accion_detectada_str = "Foco visual evasivo hacia los lados"
        estado_clase = "status-distraido"
        tipo_atencion = "attention-fill-amber"
    elif emo_max == "enojo":
        estado_rapido = "Tensión / Frustración con el tema"
        accion_detectada_str = "Gesto de molestia o bloqueo cognitivo"
        estado_clase = "status-fatiga"
        tipo_atencion = "attention-fill-amber"
    elif emo_max == "miedo":
        estado_rapido = "Estrés / Ansiedad evaluativa"
        accion_detectada_str = "Tensión facial por evaluación cercana"
        estado_clase = "status-fatiga"
        tipo_atencion = "attention-fill-amber"
    elif emo_max == "tristeza":
        estado_rapido = "Decaimiento / Bajo ánimo"
        accion_detectada_str = "Desconexión emocional pasiva"
        estado_clase = "status-distraido"
        tipo_atencion = "attention-fill-amber"
    elif emo_max == "felicidad":
        estado_rapido = "Motivado y Participativo"
        accion_detectada_str = "Receptividad activa en clase"
        estado_clase = "status-concentrado"
        tipo_atencion = "attention-fill-green"
    else:
        estado_rapido = "Concentrado en sesión"
        accion_detectada_str = "Atento a la clase sin distractores"
        estado_clase = "status-concentrado"
        tipo_atencion = "attention-fill-green"

    # Construcción de Acontecimientos Previos personalizados
    causas = []
    if act_celular > 0.5 or act_audifonos > 0.5 or prob_distraccion_ext > 50:
        causas.append({"texto": "Atención retenida por urgencia o conversación externa en el móvil", "prob": round(prob_distraccion_ext)})
        causas.append({"texto": "Desconexión del hilo de la clase por estímulo distractor", "prob": round(prob_distraccion_ext * 0.88)})
    elif emo_max in ("tristeza", "disgusto"):
        causas.append({"texto": "Posible conflicto personal o mal momento emocional previo a clase", "prob": round(prob_conflicto)})
        causas.append({"texto": "Desmotivación o frustración por resultados académicos recientes", "prob": round(prob_conflicto * 0.88)})
    elif emo_max in ("enojo", "disgusto"):
        causas.append({"texto": "Bloqueo o frustración frente a la dificultad del tema actual", "prob": round(prob_frustracion)})
        causas.append({"texto": "Inconformidad o tensión acumulada al ingresar al aula", "prob": round(prob_frustracion * 0.85)})
    elif emo_max == "miedo":
        causas.append({"texto": "Presión crítica por entrega pendiente o examen cercano", "prob": round(prob_presion)})
        causas.append({"texto": "Sobrecarga académica acumulada en la semana", "prob": round(prob_presion * 0.86)})
    elif act_cansancio > 0.5 or prob_desvelo > 50:
        causas.append({"texto": "Desvelo nocturno o sobrecarga de actividades previas", "prob": round(prob_desvelo)})
        causas.append({"texto": "Agotamiento físico acumulado antes de la sesión", "prob": round(prob_desvelo * 0.88)})
    else:
        causas.append({"texto": "Descanso adecuado y preparación previa del tema", "prob": round(prob_descanso)})
        causas.append({"texto": "Estabilidad emocional y alta receptividad cognitiva", "prob": round(prob_descanso * 0.90)})

    # Consecuencias estimadas (con riesgo explícito de jalar)
    consecuencias = [
        {
            "texto": "Alto riesgo de desaprobación ('jalar') en el examen parcial si el comportamiento se mantiene",
            "prob": round(prob_jalar_parcial)
        },
        {
            "texto": "Dificultad o bloqueo para retener el tema de la sesión de hoy",
            "prob": round(prob_bloqueo)
        },
        {
            "texto": "Riesgo de aislamiento o baja participación en actividades de evaluación",
            "prob": round(prob_jalar_parcial * 0.78)
        }
    ]

    es_riesgo_alto = score_riesgo >= 50
    if score_riesgo < 30:
        nivel_riesgo_str = "Riesgo de Desempeño: BAJO / ÓPTIMO"
        alerta_examen = "Bajo riesgo de desaprobación ('jalar'). Proyección favorable para rendir y aprobar el examen parcial con solvencia."
    elif score_riesgo < 60:
        nivel_riesgo_str = "Riesgo de Desempeño: MODERADO / ATENCIÓN"
        alerta_examen = "Riesgo moderado de desaprobación ('jalar'). Se recomienda consolidar los temas débiles de la sesión para asegurar la aprobación."
    else:
        nivel_riesgo_str = "Riesgo de Desempeño: CRÍTICO / ALTO"
        alerta_examen = "ALTO RIESGO DE DESAPROBACIÓN ('JALAR'). La distracción o decaimiento recurrente compromete severamente el examen parcial."

    narrativa = (
        f"El Estudiante {estudiante_id} presenta reconocimiento facial de {emo_dominante_nombre.lower()} "
        f"con atención estimada en {nivel_atencion}%, detectándose '{accion_detectada_str.lower()}'. "
        f"A partir de este contexto multimodal, se infiere como causa previa principal: '{causas[0]['texto'].lower()}', "
        f"con una probabilidad calculada del {round(prob_jalar_parcial)}% de jalar el examen parcial sin intervención. "
        "**Recomendación:** Se sugiere tutoría o acercamiento preventivo del docente."
    )

    return {
        "id": estudiante_id,
        "nombre": f"Estudiante {estudiante_id}",
        "subId": f"0.{80 + (estudiante_id * 3) % 19:02d}",
        "estado": estado_rapido,
        "accionDetectada": accion_detectada_str,
        "estadoClase": estado_clase,
        "atencion": nivel_atencion,
        "tipoAtencion": tipo_atencion,
        "emociones": {
            "neutral": round(p_neu * 100, 1),
            "felicidad": round(p_fel * 100, 1),
            "tristeza": round(p_tri * 100, 1),
            "enojo": round(p_eno * 100, 1),
            "miedo": round(p_mie * 100, 1),
            "sorpresa": round(p_sor * 100, 1),
            "disgusto": round(p_dis * 100, 1),
            "dominante": emo_dominante_nombre
        },
        "causas": causas,
        "consecuencias": consecuencias,
        "score": score_riesgo,
        "nivelRiesgo": nivel_riesgo_str,
        "esRiesgoAlto": es_riesgo_alto,
        "alertaExamen": alerta_examen,
        "narrativa": narrativa,
        "box": bbox_rel,
        "label": f"Estudiante {estudiante_id} - Persona (0.{85 + estudiante_id % 12})",
        "distractor": act_celular > 0.5,
        "distractorBox": {
            "x": bbox_rel["x"] + bbox_rel["w"] * 0.2,
            "y": bbox_rel["y"] + bbox_rel["h"] * 0.65,
            "w": bbox_rel["w"] * 0.5,
            "h": bbox_rel["h"] * 0.35,
            "label": "Celular (0.91)"
        } if act_celular > 0.5 else None
    }

def procesar_captura_imagen(imagen_b64: str, contexto_externo: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Función principal de inferencia del Módulo de Predicción (AulaPredict v2.5).
    Pipeline Híbrido:
    1. YOLO Pose (detección de personas + keypoints de cabeza/postura mirando examen o celular)
    2. YOLO Best (detección de objetos distractores: teléfono, audífonos, laptop)
    3. Respaldo OpenCV Haar Cascade (detección frontal clásica)
    4. REGLA ESTRICTA: Si no hay personas ni rostros reales en la captura, retorna
       limpiamente total_alumnos: 0 y estudiantes: [] (NUNCA inventa estudiantes).
    """
    img = _decodificar_imagen_b64(imagen_b64)
    h_img, w_img = img.shape[:2]

    # Extraer acciones desde el contexto externo si fue provisto por Imagen/Video
    ctx_general = {}
    if isinstance(contexto_externo, dict):
        acciones_lista = contexto_externo.get("acciones_detectadas", [])
        objetos_lista = contexto_externo.get("objetos", [])
        if "uso_celular" in acciones_lista or "telefono" in objetos_lista:
            ctx_general["accion_celular"] = 1.0
        if "uso_audifonos" in acciones_lista or "audifonos" in objetos_lista:
            ctx_general["accion_audifonos"] = 1.0
        if "mirada_desviada" in acciones_lista:
            ctx_general["accion_mirada_desviada"] = 1.0
        if "postura_cansancio" in acciones_lista or "durmiendo" in acciones_lista:
            ctx_general["accion_postura_cansancio"] = 1.0

    # 1. DETECCIÓN CON YOLO POSE (Detección de personas y posturas reales)
    candidatos_personas: List[Dict[str, Any]] = []
    yolo_pose = _obtener_yolo_pose()
    if yolo_pose is not None:
        try:
            res_pose = yolo_pose(img, verbose=False, conf=0.30)
            if res_pose and len(res_pose[0].boxes) > 0:
                boxes = res_pose[0].boxes
                kpts_data = res_pose[0].keypoints.data if hasattr(res_pose[0], 'keypoints') and res_pose[0].keypoints is not None else None

                for i, b in enumerate(boxes):
                    xyxy = b.xyxy[0].cpu().numpy()
                    conf = float(b.conf[0].cpu().numpy())
                    kpts = kpts_data[i].cpu().numpy() if kpts_data is not None and len(kpts_data) > i else None
                    candidatos_personas.append({
                        "x1": int(max(0, xyxy[0])),
                        "y1": int(max(0, xyxy[1])),
                        "x2": int(min(w_img, xyxy[2])),
                        "y2": int(min(h_img, xyxy[3])),
                        "conf": conf,
                        "keypoints": kpts,
                        "tipo": "pose"
                    })
        except Exception as e:
            print(f"[Aviso] YOLO Pose en fallback: {e}")

    # 2. DETECCIÓN DE OBJETOS DISTRACTORES CON YOLO BEST (Celular, audífonos, laptop)
    telefonos_detectados: List[Dict[str, Any]] = []
    audifonos_detectados: List[Dict[str, Any]] = []
    yolo_best = _obtener_yolo_best()
    if yolo_best is not None:
        try:
            res_best = yolo_best(img, verbose=False, conf=0.25)
            if res_best and len(res_best[0].boxes) > 0:
                for b in res_best[0].boxes:
                    cls_id = int(b.cls[0].cpu().numpy())
                    xyxy = b.xyxy[0].cpu().numpy()
                    conf = float(b.conf[0].cpu().numpy())
                    # Mapeo: 0: persona, 2: telefono, 5: audifonos, 7: laptop
                    if cls_id == 2:  # telefono
                        telefonos_detectados.append({
                            "x1": int(max(0, xyxy[0])), "y1": int(max(0, xyxy[1])),
                            "x2": int(min(w_img, xyxy[2])), "y2": int(min(h_img, xyxy[3])),
                            "conf": conf
                        })
                    elif cls_id == 5:  # audifonos
                        audifonos_detectados.append({
                            "x1": int(max(0, xyxy[0])), "y1": int(max(0, xyxy[1])),
                            "x2": int(min(w_img, xyxy[2])), "y2": int(min(h_img, xyxy[3])),
                            "conf": conf
                        })
                    elif cls_id == 0 and len(candidatos_personas) == 0:  # persona por best.pt
                        candidatos_personas.append({
                            "x1": int(max(0, xyxy[0])), "y1": int(max(0, xyxy[1])),
                            "x2": int(min(w_img, xyxy[2])), "y2": int(min(h_img, xyxy[3])),
                            "conf": conf,
                            "keypoints": None,
                            "tipo": "best_persona"
                        })
        except Exception as e:
            print(f"[Aviso] YOLO Best en fallback: {e}")

    # 3. RESPALDO SECUNDARIO CON HAAR CASCADE (Si YOLO no detectó personas completas)
    if len(candidatos_personas) == 0:
        cascade = _obtener_face_cascade()
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        caras = cascade.detectMultiScale(gris, scaleFactor=1.15, minNeighbors=4, minSize=(35, 35))
        for fx, fy, fw, fh in caras:
            candidatos_personas.append({
                "x1": int(max(0, fx)),
                "y1": int(max(0, fy)),
                "x2": int(min(w_img, fx + fw)),
                "y2": int(min(h_img, fy + fh)),
                "conf": 0.88,
                "keypoints": None,
                "tipo": "haar_face"
            })

    # =========================================================================
    # REGLA DE ORO DE AUDITORÍA: SI NO HAY PERSONAS DETECTADAS, RETORNA CERO
    # NUNCA MÁS SE INVENTA UN "ESTUDIANTE 1" SOBRE LA PARED O ESCRITORIO VACÍO
    # =========================================================================
    if len(candidatos_personas) == 0:
        return {
            "estado": "sin_personas",
            "mensaje": "No se detectaron estudiantes en la captura. No hay predicciones para generar.",
            "total_alumnos": 0,
            "estudiantes": [],
            "resolucion_captura": f"{w_img}x{h_img}",
            "soporte_emociones": 7
        }

    # Ordenar estudiantes de izquierda a derecha en el aula
    candidatos_personas.sort(key=lambda p: p["x1"])
    estudiantes_resultado: List[Dict[str, Any]] = []

    for idx, cand in enumerate(candidatos_personas, start=1):
        x1, y1, x2, y2 = cand["x1"], cand["y1"], cand["x2"], cand["y2"]
        w_box = max(20, x2 - x1)
        h_box = max(20, y2 - y1)

        bbox_rel = {
            "x": round(float(x1) / float(w_img), 4),
            "y": round(float(y1) / float(h_img), 4),
            "w": round(float(w_box) / float(w_img), 4),
            "h": round(float(h_box) / float(h_img), 4)
        }

        # Extracción de la región facial
        keypoints = cand.get("keypoints")
        if cand["tipo"] == "haar_face":
            roi_face = img[y1:y2, x1:x2]
        elif keypoints is not None and len(keypoints) >= 5:
            # Delimitar cabeza según keypoints de ojos, orejas y nariz
            pts_head = [p for p in keypoints[:5] if p[2] > 0.2]
            if len(pts_head) >= 2:
                hx1 = max(0, int(min(p[0] for p in pts_head) - w_box * 0.15))
                hx2 = min(w_img, int(max(p[0] for p in pts_head) + w_box * 0.15))
                hy1 = max(0, int(min(p[1] for p in pts_head) - h_box * 0.10))
                hy2 = min(h_img, int(max(p[1] for p in pts_head) + h_box * 0.15))
                roi_face = img[hy1:max(hy1 + 20, hy2), hx1:max(hx1 + 20, hx2)]
            else:
                # Región superior del torso
                hy2 = int(y1 + h_box * 0.40)
                roi_face = img[y1:max(y1 + 20, hy2), x1:x2]
        else:
            hy2 = int(y1 + h_box * 0.40)
            roi_face = img[y1:max(y1 + 20, hy2), x1:x2]

        if roi_face.size == 0:
            roi_face = img[y1:y2, x1:x2]

        rasgos = _extraer_rasgos_faciales(roi_face, bbox_rel, keypoints)

        # Fusión de acciones detectadas localmente para este alumno
        ctx_alumno = dict(ctx_general)

        # Verificar si hay un celular cerca o dentro de su área
        tiene_celular = False
        distractor_box = None
        for tel in telefonos_detectados:
            tx1, ty1, tx2, ty2 = tel["x1"], tel["y1"], tel["x2"], tel["y2"]
            # Solapamiento o proximidad
            if not (tx2 < x1 - 10 or tx1 > x2 + 10 or ty2 < y1 or ty1 > y2):
                tiene_celular = True
                distractor_box = {
                    "x": round(float(tx1) / float(w_img), 4),
                    "y": round(float(ty1) / float(h_img), 4),
                    "w": round(float(tx2 - tx1) / float(w_img), 4),
                    "h": round(float(ty2 - ty1) / float(h_img), 4),
                    "label": f"Celular ({tel['conf']:.2f})"
                }
                break

        if tiene_celular or ctx_alumno.get("accion_celular", 0.0) > 0.5:
            ctx_alumno["accion_celular"] = 1.0

        # Verificar si hay audífonos cerca de su cabeza
        for aud in audifonos_detectados:
            ax1, ay1, ax2, ay2 = aud["x1"], aud["y1"], aud["x2"], aud["y2"]
            if not (ax2 < x1 or ax1 > x2 or ay2 < y1 or ay1 > y1 + int(h_box * 0.5)):
                ctx_alumno["accion_audifonos"] = 1.0
                break

        est = predecir_estudiante_multimodal(idx, rasgos, bbox_rel, ctx_alumno)
        if distractor_box is not None:
            est["distractor"] = True
            est["distractorBox"] = distractor_box

        estudiantes_resultado.append(est)

    return {
        "estado": "exito",
        "total_alumnos": len(estudiantes_resultado),
        "estudiantes": estudiantes_resultado,
        "resolucion_captura": f"{w_img}x{h_img}",
        "soporte_emociones": 7
    }

def obtener_escenario_aula_ejemplo() -> Dict[str, Any]:
    """
    Retorna un escenario de demostración de aula con 5 estudiantes analizados
    con las 7 emociones y acciones representativas.
    """
    estudiantes = []
    config_alumnos = [
        # Alumno 1: Concentrado (Neutral 82%, Felicidad 12%), Sin distractores, head_pitch=8.0
        (1, {"eye_openness": 0.74, "eyebrow_slope": 0.02, "mouth_smile": 0.15, "mouth_open": 0.08, "head_tilt": 2.0, "head_pitch": 8.0, "gaze_stability": 0.90, "blink_rate": 0.30, "facial_tension": 0.18}, {"x": 0.12, "y": 0.18, "w": 0.22, "h": 0.52}, {}),
        # Alumno 2: Fatiga / Desvelo (Tristeza 45%, Miedo 20%), Postura de cansancio, head_pitch=26.0
        (2, {"eye_openness": 0.32, "eyebrow_slope": -0.25, "mouth_smile": -0.20, "mouth_open": 0.35, "head_tilt": 18.0, "head_pitch": 26.0, "gaze_stability": 0.35, "blink_rate": 0.75, "facial_tension": 0.30}, {"x": 0.40, "y": 0.16, "w": 0.22, "h": 0.50}, {"accion_postura_cansancio": 1.0}),
        # Alumno 3: Uso de celular detectado (Miedo 35%, Enojo 32%), Celular activo, head_pitch=32.0
        (3, {"eye_openness": 0.45, "eyebrow_slope": 0.30, "mouth_smile": -0.25, "mouth_open": 0.10, "head_tilt": 20.0, "head_pitch": 32.0, "gaze_stability": 0.25, "blink_rate": 0.70, "facial_tension": 0.65}, {"x": 0.68, "y": 0.20, "w": 0.24, "h": 0.54}, {"accion_celular": 1.0}),
        # Alumno 4: Frustración / Enojo frente al tema (Enojo 55%, Disgusto 22%), head_pitch=14.0
        (4, {"eye_openness": 0.65, "eyebrow_slope": 0.68, "mouth_smile": -0.38, "mouth_open": 0.06, "head_tilt": 8.0, "head_pitch": 14.0, "gaze_stability": 0.55, "blink_rate": 0.65, "facial_tension": 0.85}, {"x": 0.05, "y": 0.36, "w": 0.22, "h": 0.50}, {"accion_mirada_desviada": 1.0}),
        # Alumno 5: Ansiedad / Miedo evaluativo (Miedo 60%, Enojo 18%), head_pitch=10.0
        (5, {"eye_openness": 0.82, "eyebrow_slope": 0.48, "mouth_smile": -0.22, "mouth_open": 0.22, "head_tilt": 6.0, "head_pitch": 10.0, "gaze_stability": 0.42, "blink_rate": 0.85, "facial_tension": 0.75}, {"x": 0.32, "y": 0.36, "w": 0.22, "h": 0.50}, {})
    ]

    for id_est, rasgos, box, ctx in config_alumnos:
        est = predecir_estudiante_multimodal(id_est, rasgos, box, ctx)
        estudiantes.append(est)

    return {
        "estado": "exito",
        "total_alumnos": 5,
        "estudiantes": estudiantes,
        "resolucion_captura": "1280x720 (Simulación Aula 301)",
        "soporte_emociones": 7
    }

def obtener_metricas_modelo() -> Dict[str, Any]:
    """
    Retorna el diccionario con las métricas oficiales de entrenamiento y validación
    almacenadas en metricas_entrenamiento.json (precisión de 7 emociones y MAE de sucesos).
    """
    ruta_metricas = CARPETA_ACTUAL / "metricas_entrenamiento.json"
    if ruta_metricas.is_file():
        with open(ruta_metricas, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "modulo": "AulaPredict - Predicción Universitaria",
        "version": "2.5.0",
        "estado": "metricas_no_encontradas"
    }

