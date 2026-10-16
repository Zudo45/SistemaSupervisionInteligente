# Módulo IMAGEN

Continuación del sistema del grupo VIDEO. **VIDEO detecta; IMAGEN mejora, analiza y caracteriza lo detectado**, y mide con métricas si su procesamiento mejora los resultados de VIDEO.

El código de VIDEO (`principal.py`, `modulos/`, `modelo_ia/best.pt`) no se modifica: el módulo IMAGEN solo importa su función `detectar_objetos`.

## Flujo

```
fotograma ─► VIDEO (best.pt) ─► recepción ─► preprocesamiento adaptativo ─► mejoramiento
                                                                              │
     JSON / CSV para PREDICCIÓN ◄─ características ◄─ segmentación ◄─ verificación
```

| Etapa | Archivo | Técnicas |
|---|---|---|
| 1. Recepción | `modulo_imagen/recepcion.py` | Fotogramas de imágenes o video + detecciones de VIDEO (clase, confianza, caja) |
| 2. Preprocesamiento | `modulo_imagen/preprocesamiento.py` | Mediana, gaussiano, bilateral, Non-Local Means, gamma automática, CLAHE, ecualización |
| 3. Mejoramiento | `modulo_imagen/mejoramiento.py` | Estiramiento de contraste, brillo/contraste lineal, unsharp masking, ampliación bicúbica |
| Modo adaptativo | `modulo_imagen/pipeline.py` | Elige los filtros según el ruido, brillo, contraste y nitidez medidos |
| 4. Segmentación | `modulo_imagen/segmentacion.py` | GrabCut, Otsu, morfología, contornos, Canny |
| 5. Características | `modulo_imagen/caracteristicas.py` | Tamaño, posición, forma, color (HSV), textura y relación con la persona |
| 6. Verificación | `modulo_imagen/verificacion.py` | Mismo modelo de VIDEO antes y después del procesamiento: confianza y detecciones |
| Calidad | `modulo_imagen/calidad.py` | Brillo, contraste, nitidez (var. Laplaciano), ruido (Immerkær), entropía, PSNR, SSIM |
| 7. Visualización | `../app.py` (sección Imágenes) | Página web del sistema |
| Conexión | `modulo_imagen/vigilante.py` | Procesa automáticamente las capturas de la alarma de VIDEO |
| 8. Exportación | `modulo_imagen/exportacion.py` | JSON por fotograma y CSV por objeto |

## Conexión con VIDEO (alertas)

Cuando la alarma de VIDEO se dispara, su programa guarda la captura en `video/alertas/alerta_<objeto>_<fecha>_<hora>.jpg`. La página de IMAGEN vigila esa carpeta cada 2 segundos y procesa automáticamente cada captura nueva:

- imágenes anotadas en `../resultados/alertas/imagenes/`
- resultado por alerta en `../resultados/alertas/datos/`
- salida acumulada en `../resultados/alertas/resultados_alertas.json` y `.csv`

La página permite recorrer las alertas con las flechas ◀ ▶ o con las miniaturas. También se puede vigilar la carpeta sin interfaz con `vigilar_alertas.py`.

`calibrar_video.py` mide, con la cámara y sin guardar imágenes, cómo responde el modelo de VIDEO a cada objeto; sus resultados (`calibracion/calibracion_video.json`) muestran que la detección parpadea entre fotogramas en una webcam.

## Uso

El módulo se usa desde la página del sistema (ver el `README.md` principal). Herramientas propias de IMAGEN, desde esta carpeta:

```
..\.venv\Scripts\python vigilar_alertas.py      # procesa las alertas de VIDEO sin la página
..\.venv\Scripts\python calibrar_video.py       # mide la respuesta del modelo de VIDEO con la cámara
```
