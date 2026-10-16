"""
Generador de Dataset de Características Faciales y Proyección Académica
Módulo de Predicción (AulaPredict) - Sistema de Supervisión Inteligente
Soporta las 7 Emociones Universales + Ingesta Multimodal de Acciones
"""
import numpy as np
import pandas as pd

NOMBRES_7_EMOCIONES = [
    "neutral",        # 0: Neutral / Concentración (😐)
    "felicidad",      # 1: Felicidad / Motivación (😊)
    "tristeza",       # 2: Tristeza / Decaimiento (😢)
    "enojo",          # 3: Enojo / Frustración (😠)
    "miedo",          # 4: Miedo / Ansiedad / Estrés (😨)
    "sorpresa",       # 5: Sorpresa / Sobresalto (😲)
    "disgusto"        # 6: Disgusto / Rechazo (🤢)
]

def generar_dataset_facial(n_samples: int = 5600, random_state: int = 42):
    """
    Genera un conjunto sintético calibrado de métricas faciales geométricas
    asociadas a las 7 emociones faciales universales:
    0: Neutral / Concentración (😐)
    1: Felicidad / Motivación (😊)
    2: Tristeza / Decaimiento (😢)
    3: Enojo / Frustración (😠)
    4: Miedo / Ansiedad / Estrés (😨)
    5: Sorpresa / Sobresalto (😲)
    6: Disgusto / Rechazo (🤢)
    """
    rng = np.random.RandomState(random_state)
    muestras_por_clase = n_samples // 7
    registros = []

    # 0: NEUTRAL / CONCENTRACIÓN (😐)
    # Rasgos: Ojos abiertos y enfocados, cejas relajadas, boca neutra, cabeza erguida o mirando examen/mesa con atención
    for _ in range(muestras_por_clase):
        head_pitch = float(np.clip(rng.normal(10.0, 7.0), 0.0, 36.0))
        registros.append({
            "eye_openness": float(np.clip(rng.normal(0.72, 0.07), 0.50, 0.95)),
            "eyebrow_slope": float(rng.normal(0.02, 0.08)),
            "mouth_smile": float(rng.normal(0.05, 0.08)),
            "mouth_open": float(np.clip(rng.normal(0.08, 0.04), 0.02, 0.20)),
            "head_tilt": float(np.clip(rng.normal(2.5, 3.0), 0.0, 15.0)),
            "head_pitch": head_pitch,
            "gaze_stability": float(np.clip(rng.normal(0.88, 0.06), 0.65, 0.99)),
            "blink_rate": float(np.clip(rng.normal(0.35, 0.08), 0.15, 0.55)),
            "facial_tension": float(np.clip(rng.normal(0.18, 0.06), 0.05, 0.35)),
            "label_emocion": 0,
            "nombre_emocion": "neutral"
        })

    # 1: FELICIDAD / MOTIVACIÓN (😊)
    # Rasgos: Sonrisa marcada (comisuras elevadas), ojos ligeramente achinados por mejillas altas, cabeza erguida
    for _ in range(muestras_por_clase):
        head_pitch = float(np.clip(rng.normal(4.0, 3.0), 0.0, 16.0))
        registros.append({
            "eye_openness": float(np.clip(rng.normal(0.68, 0.08), 0.45, 0.90)),
            "eyebrow_slope": float(rng.normal(0.10, 0.09)),
            "mouth_smile": float(np.clip(rng.normal(0.65, 0.12), 0.35, 0.98)),
            "mouth_open": float(np.clip(rng.normal(0.20, 0.08), 0.05, 0.45)),
            "head_tilt": float(np.clip(rng.normal(3.0, 3.5), 0.0, 18.0)),
            "head_pitch": head_pitch,
            "gaze_stability": float(np.clip(rng.normal(0.85, 0.08), 0.60, 0.98)),
            "blink_rate": float(np.clip(rng.normal(0.38, 0.09), 0.18, 0.60)),
            "facial_tension": float(np.clip(rng.normal(0.22, 0.07), 0.05, 0.40)),
            "label_emocion": 1,
            "nombre_emocion": "felicidad"
        })

    # 2: TRISTEZA / DECAIMIENTO (😢)
    # Rasgos: Comisuras caídas, extremos internos de cejas elevados/caídos en bordes, mirada baja/dispersa, cabeza gacha
    for _ in range(muestras_por_clase):
        head_pitch = float(np.clip(rng.normal(25.0, 8.0), 5.0, 45.0))
        registros.append({
            "eye_openness": float(np.clip(rng.normal(0.42, 0.08), 0.20, 0.65)),
            "eyebrow_slope": float(rng.normal(-0.45, 0.12)),
            "mouth_smile": float(np.clip(rng.normal(-0.55, 0.12), -0.95, -0.25)),
            "mouth_open": float(np.clip(rng.normal(0.12, 0.06), 0.02, 0.30)),
            "head_tilt": float(np.clip(rng.normal(14.0, 6.0), 2.0, 35.0)),
            "head_pitch": head_pitch,
            "gaze_stability": float(np.clip(rng.normal(0.28, 0.10), 0.08, 0.55)),
            "blink_rate": float(np.clip(rng.normal(0.48, 0.10), 0.25, 0.75)),
            "facial_tension": float(np.clip(rng.normal(0.35, 0.10), 0.15, 0.60)),
            "label_emocion": 2,
            "nombre_emocion": "tristeza"
        })

    # 3: ENOJO / FRUSTRACIÓN (😠)
    # Rasgos: Cejas muy fruncidas hacia abajo/centro, boca apretada o tensa, mirada fija penetrante o bloqueada
    for _ in range(muestras_por_clase):
        head_pitch = float(np.clip(rng.normal(10.0, 6.0), 0.0, 26.0))
        registros.append({
            "eye_openness": float(np.clip(rng.normal(0.62, 0.10), 0.35, 0.85)),
            "eyebrow_slope": float(np.clip(rng.normal(0.72, 0.12), 0.40, 0.98)),
            "mouth_smile": float(np.clip(rng.normal(-0.35, 0.12), -0.75, -0.05)),
            "mouth_open": float(np.clip(rng.normal(0.08, 0.04), 0.01, 0.20)),
            "head_tilt": float(np.clip(rng.normal(6.0, 4.0), 0.0, 20.0)),
            "head_pitch": head_pitch,
            "gaze_stability": float(np.clip(rng.normal(0.70, 0.12), 0.40, 0.95)),
            "blink_rate": float(np.clip(rng.normal(0.65, 0.12), 0.35, 0.90)),
            "facial_tension": float(np.clip(rng.normal(0.85, 0.09), 0.60, 0.99)),
            "label_emocion": 3,
            "nombre_emocion": "enojo"
        })

    # 4: MIEDO / ANSIEDAD / ESTRÉS (😨)
    # Rasgos: Ojos abiertos de par en par con tensión, cejas levantadas y juntas, parpadeo alto, boca entreabierta tensa
    for _ in range(muestras_por_clase):
        head_pitch = float(np.clip(rng.normal(8.0, 5.0), 0.0, 22.0))
        registros.append({
            "eye_openness": float(np.clip(rng.normal(0.80, 0.08), 0.58, 0.98)),
            "eyebrow_slope": float(np.clip(rng.normal(0.48, 0.14), 0.18, 0.85)),
            "mouth_smile": float(np.clip(rng.normal(-0.25, 0.10), -0.55, 0.00)),
            "mouth_open": float(np.clip(rng.normal(0.25, 0.09), 0.08, 0.55)),
            "head_tilt": float(np.clip(rng.normal(8.0, 5.0), 0.0, 25.0)),
            "head_pitch": head_pitch,
            "gaze_stability": float(np.clip(rng.normal(0.40, 0.14), 0.15, 0.70)),
            "blink_rate": float(np.clip(rng.normal(0.86, 0.08), 0.65, 0.99)),
            "facial_tension": float(np.clip(rng.normal(0.78, 0.10), 0.50, 0.98)),
            "label_emocion": 4,
            "nombre_emocion": "miedo"
        })

    # 5: SORPRESA / SOBRESALTO (😲)
    # Rasgos: Ojos muy abiertos, cejas completamente arqueadas hacia arriba, boca marcadamente abierta redondeada
    for _ in range(muestras_por_clase):
        head_pitch = float(np.clip(rng.normal(3.0, 3.0), 0.0, 14.0))
        registros.append({
            "eye_openness": float(np.clip(rng.normal(0.88, 0.06), 0.70, 0.99)),
            "eyebrow_slope": float(np.clip(rng.normal(0.60, 0.12), 0.30, 0.95)),
            "mouth_smile": float(rng.normal(0.05, 0.10)),
            "mouth_open": float(np.clip(rng.normal(0.68, 0.12), 0.40, 0.96)),
            "head_tilt": float(np.clip(rng.normal(5.0, 4.0), 0.0, 20.0)),
            "head_pitch": head_pitch,
            "gaze_stability": float(np.clip(rng.normal(0.65, 0.15), 0.30, 0.95)),
            "blink_rate": float(np.clip(rng.normal(0.25, 0.08), 0.08, 0.45)),
            "facial_tension": float(np.clip(rng.normal(0.45, 0.12), 0.20, 0.75)),
            "label_emocion": 5,
            "nombre_emocion": "sorpresa"
        })

    # 6: DISGUSTO / RECHAZO (🤢)
    # Rasgos: Nariz y cejas arrugadas, labio superior levantado/torcido, mirada evasiva o de desaprobación
    for _ in range(muestras_por_clase):
        head_pitch = float(np.clip(rng.normal(14.0, 6.0), 2.0, 32.0))
        registros.append({
            "eye_openness": float(np.clip(rng.normal(0.48, 0.09), 0.25, 0.70)),
            "eyebrow_slope": float(np.clip(rng.normal(0.35, 0.12), 0.10, 0.70)),
            "mouth_smile": float(np.clip(rng.normal(-0.48, 0.14), -0.85, -0.15)),
            "mouth_open": float(np.clip(rng.normal(0.18, 0.08), 0.04, 0.40)),
            "head_tilt": float(np.clip(rng.normal(12.0, 6.0), 2.0, 30.0)),
            "head_pitch": head_pitch,
            "gaze_stability": float(np.clip(rng.normal(0.35, 0.12), 0.10, 0.65)),
            "blink_rate": float(np.clip(rng.normal(0.55, 0.12), 0.25, 0.85)),
            "facial_tension": float(np.clip(rng.normal(0.70, 0.12), 0.45, 0.95)),
            "label_emocion": 6,
            "nombre_emocion": "disgusto"
        })

    df = pd.DataFrame(registros).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df

def generar_dataset_sucesos_y_riesgo(n_samples: int = 5000, random_state: int = 42):
    """
    Genera dataset para entrenar el modelo probabilístico fusionando:
    - Las 7 emociones universales reconocidas en el rostro.
    - Las señales de acción recibidas de los módulos de Imagen/Video o detectadas:
      `accion_celular`, `accion_audifonos`, `accion_mirada_desviada`,
      `accion_laptop_otra_ventana`, `accion_postura_cansancio`.
    """
    rng = np.random.RandomState(random_state)
    registros = []

    for _ in range(n_samples):
        # 1. Distribución de las 7 emociones universales (Dirichlet)
        alpha = rng.exponential(scale=1.0, size=7) + 0.1
        emociones = rng.dirichlet(alpha)
        p_neu, p_fel, p_tri, p_eno, p_mie, p_sor, p_dis = emociones

        # 2. Acciones del alumno (recibidas o detectadas)
        # Correlación natural: si hay disgusto/enojo/tristeza es más probable mirar celular o desviar mirada
        p_celular = 0.50 * p_tri + 0.40 * p_eno + 0.30 * p_dis + 0.10 * p_neu
        p_cansancio = 0.55 * p_tri + 0.35 * p_eno + 0.10 * p_neu
        p_desvio = 0.45 * p_dis + 0.40 * p_mie + 0.35 * p_tri + 0.10 * p_neu

        accion_celular = 1.0 if rng.rand() < p_celular else 0.0
        accion_audifonos = 1.0 if rng.rand() < (0.25 * accion_celular + 0.15 * p_dis) else 0.0
        accion_mirada_desviada = 1.0 if (rng.rand() < p_desvio or accion_celular == 1.0) else 0.0
        accion_laptop_otra_ventana = 1.0 if rng.rand() < (0.30 * p_dis + 0.20 * accion_celular) else 0.0
        accion_postura_cansancio = 1.0 if rng.rand() < p_cansancio else 0.0

        # 3. Predicción de Acontecimientos Previos (Causas Probables)
        # - Desvelo o sobrecarga previa:
        prob_desvelo = np.clip(
            accion_postura_cansancio * 55 + p_tri * 35 + p_mie * 20 + rng.normal(5, 4),
            5, 96
        )
        # - Conflicto personal o mal momento previo:
        prob_conflicto = np.clip(
            p_tri * 60 + p_dis * 25 + p_eno * 15 + rng.normal(5, 4),
            5, 95
        )
        # - Bloqueo / Frustración por dificultad del tema actual:
        prob_frustracion_tema = np.clip(
            p_eno * 65 + p_dis * 25 + p_mie * 15 + rng.normal(4, 4),
            5, 95
        )
        # - Presión crítica por entregas o examen cercano:
        prob_presion_evaluacion = np.clip(
            p_mie * 70 + p_eno * 20 + rng.normal(5, 4),
            5, 95
        )
        # - Descanso adecuado y preparación previa del tema:
        prob_descanso_preparacion = np.clip(
            (p_neu * 60 + p_fel * 35) * (1.0 - 0.6 * accion_celular - 0.4 * accion_postura_cansancio) + rng.normal(5, 3),
            4, 98
        )
        # - Desconexión por distracción externa (celular / laptop / conversación):
        prob_distraccion_externa = np.clip(
            accion_celular * 65 + accion_audifonos * 15 + accion_mirada_desviada * 20 + rng.normal(4, 3),
            5, 97
        )

        # 4. Predicción de Acontecimientos Futuros y Riesgo de Desaprobación (Jalar)
        # Factor acumulado de riesgo:
        riesgo_emocional = (
            p_tri * 0.35 +
            p_eno * 0.30 +
            p_mie * 0.35 +
            p_dis * 0.25 -
            p_neu * 0.45 -
            p_fel * 0.40
        )
        riesgo_conductual = (
            accion_celular * 0.40 +
            accion_audifonos * 0.20 +
            accion_mirada_desviada * 0.25 +
            accion_laptop_otra_ventana * 0.20 +
            accion_postura_cansancio * 0.25
        )
        
        # Score heurístico de riesgo de desempeño (0 a 100)
        score_riesgo = np.clip((riesgo_emocional + riesgo_conductual) * 60 + 35 + rng.normal(0, 3), 5, 98)

        # Probabilidad de jalar el examen parcial si el estado persiste
        prob_jalar_parcial = np.clip(score_riesgo * 0.95 + accion_celular * 10 + rng.normal(0, 3), 8, 97)
        # Dificultad o bloqueo para retener el tema de la sesión de hoy
        prob_bloqueo_retencion = np.clip(score_riesgo * 0.88 + p_eno * 12 + accion_postura_cansancio * 10 + rng.normal(0, 3), 10, 95)
        # Nivel de atención estimado (inverso del riesgo)
        nivel_atencion = np.clip(100 - score_riesgo * 0.90 + rng.normal(0, 3), 8, 98)

        registros.append({
            "p_neutral": p_neu,
            "p_felicidad": p_fel,
            "p_tristeza": p_tri,
            "p_enojo": p_eno,
            "p_miedo": p_mie,
            "p_sorpresa": p_sor,
            "p_disgusto": p_dis,
            "accion_celular": accion_celular,
            "accion_audifonos": accion_audifonos,
            "accion_mirada_desviada": accion_mirada_desviada,
            "accion_laptop_otra_ventana": accion_laptop_otra_ventana,
            "accion_postura_cansancio": accion_postura_cansancio,
            "prob_desvelo": prob_desvelo,
            "prob_conflicto": prob_conflicto,
            "prob_frustracion_tema": prob_frustracion_tema,
            "prob_presion_evaluacion": prob_presion_evaluacion,
            "prob_descanso_preparacion": prob_descanso_preparacion,
            "prob_distraccion_externa": prob_distraccion_externa,
            "prob_jalar_parcial": prob_jalar_parcial,
            "prob_bloqueo_retencion": prob_bloqueo_retencion,
            "score_riesgo": score_riesgo,
            "nivel_atencion": nivel_atencion
        })

    return pd.DataFrame(registros)
