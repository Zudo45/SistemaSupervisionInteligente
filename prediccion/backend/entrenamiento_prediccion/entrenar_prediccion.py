"""
Script de Entrenamiento Real de Modelos de Predicción (AulaPredict v2.5)
1. Modelo 1: Clasificador Facial de las 7 Emociones Universales (RandomForest) -> modelo_emociones_facial.pkl
2. Modelo 2: Inferencia Multimodal de Sucesos y Riesgo Académico (7 Emociones + Acciones) -> modelo_sucesos_academicos.pkl
"""
import os
import sys
import json
import pickle
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, mean_absolute_error, r2_score

CARPETA_ACTUAL = Path(__file__).resolve().parent
RAIZ = CARPETA_ACTUAL.parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from backend.entrenamiento_prediccion.dataset_generator import (
    generar_dataset_facial,
    generar_dataset_sucesos_y_riesgo,
    NOMBRES_7_EMOCIONES
)

def entrenar_modelo_7_emociones():
    print("=" * 75)
    print(">>> 1. ENTRENAMIENTO DEL MODELO FACIAL: 7 EMOCIONES UNIVERSALES")
    print("=" * 75)

    df_facial = generar_dataset_facial(n_samples=5600, random_state=42)
    feature_cols = [
        "eye_openness", "eyebrow_slope", "mouth_smile", "mouth_open",
        "head_tilt", "head_pitch", "gaze_stability", "blink_rate", "facial_tension"
    ]
    X = df_facial[feature_cols].values
    y = df_facial["label_emocion"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline_facial = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=180,
            max_depth=14,
            min_samples_split=3,
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipeline_facial.fit(X_train, y_train)
    y_pred = pipeline_facial.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    nombres_legibles = [
        "Neutral (😐)", "Felicidad (😊)", "Tristeza (😢)",
        "Enojo (😠)", "Miedo (😨)", "Sorpresa (😲)", "Disgusto (🤢)"
    ]
    report = classification_report(
        y_test, y_pred,
        target_names=nombres_legibles,
        output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"Precisión Global (Accuracy): {acc * 100:.2f}%")
    print("\nReporte de Clasificación por Emoción Universal:")
    for clase in nombres_legibles:
        if clase in report:
            m = report[clase]
            print(f"  - {clase:<22}: Precision={m['precision']:.3f} | Recall={m['recall']:.3f} | F1={m['f1-score']:.3f}")

    ruta_modelo = CARPETA_ACTUAL / "modelo_emociones_facial.pkl"
    with open(ruta_modelo, "wb") as f:
        pickle.dump(pipeline_facial, f)
    print(f"\n[OK] Modelo de 7 emociones guardado en: {ruta_modelo.name}")

    return {
        "accuracy": float(acc),
        "clases": NOMBRES_7_EMOCIONES,
        "clases_legibles": nombres_legibles,
        "classification_report": report,
        "confusion_matrix": cm,
        "features": feature_cols
    }

def entrenar_modelo_sucesos_multimodal():
    print("\n" + "=" * 75)
    print(">>> 2. ENTRENAMIENTO DEL MODELO DE SUCESOS Y RIESGO (7 EMOCIONES + ACCIÓN)")
    print("=" * 75)

    df_sucesos = generar_dataset_sucesos_y_riesgo(n_samples=6000, random_state=42)
    feature_cols = [
        "p_neutral", "p_felicidad", "p_tristeza", "p_enojo", "p_miedo", "p_sorpresa", "p_disgusto",
        "accion_celular", "accion_audifonos", "accion_mirada_desviada",
        "accion_laptop_otra_ventana", "accion_postura_cansancio"
    ]
    target_cols = [
        "prob_desvelo", "prob_conflicto", "prob_frustracion_tema", "prob_presion_evaluacion",
        "prob_descanso_preparacion", "prob_distraccion_externa", "prob_jalar_parcial",
        "prob_bloqueo_retencion", "score_riesgo", "nivel_atencion"
    ]

    X = df_sucesos[feature_cols].values
    y = df_sucesos[target_cols].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    regresor = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        ))
    ])

    regresor.fit(X_train, y_train)
    y_pred = regresor.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Error Absoluto Medio (MAE en escala %): {mae:.2f}%")
    print(f"Coeficiente de Determinación R²: {r2:.4f}")
    print("\nDesglose de MAE por variable predictiva:")
    mae_por_col = mean_absolute_error(y_test, y_pred, multioutput="raw_values")
    for col, m in zip(target_cols, mae_por_col):
        print(f"  - {col:<32}: MAE = {m:.2f}%")

    ruta_modelo = CARPETA_ACTUAL / "modelo_sucesos_academicos.pkl"
    with open(ruta_modelo, "wb") as f:
        pickle.dump(regresor, f)
    print(f"\n[OK] Modelo de sucesos multimodal guardado en: {ruta_modelo.name}")

    return {
        "mae_global": float(mae),
        "r2_score": float(r2),
        "mae_por_variable": {col: float(m) for col, m in zip(target_cols, mae_por_col)},
        "features": feature_cols,
        "targets": target_cols
    }

def main():
    print("Iniciando pipeline de entrenamiento oficial de Predicción (AulaPredict v2.5)...\n")
    metricas_facial = entrenar_modelo_7_emociones()
    metricas_sucesos = entrenar_modelo_sucesos_multimodal()

    metricas_totales = {
        "modulo": "AulaPredict - Predicción Universitaria",
        "version": "2.5.0",
        "soporte": "7 Emociones Universales + Ingesta Multimodal (Imagen + Acción)",
        "fecha": "2026-10-15",
        "modelo_facial_7_emociones": metricas_facial,
        "modelo_sucesos_multimodal": metricas_sucesos
    }

    ruta_json = CARPETA_ACTUAL / "metricas_entrenamiento.json"
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(metricas_totales, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 75)
    print("Pipeline de entrenamiento completado exitosamente.")
    print("Artefactos generados:")
    print("  1. modelo_emociones_facial.pkl (7 clases)")
    print("  2. modelo_sucesos_academicos.pkl (7 emociones + 5 acciones)")
    print("  3. metricas_entrenamiento.json")
    print("=" * 75)

if __name__ == "__main__":
    main()
