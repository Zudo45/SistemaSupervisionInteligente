# Sistema de Supervisión Inteligente

Sistema de supervisión de exámenes del curso de Percepción. Integra en una sola página web el trabajo de los grupos **VIDEO**, **IMAGEN** y **PREDICCIÓN**, conectados por eventos:

```
VIDEO  ──alerta──▶  IMAGEN  ──imagen mejorada + JSON──▶  PREDICCIÓN
cámara, detección     preprocesamiento, mejora,          alumnos, emociones,
y alarma (5 s)        segmentación, características,     causas, consecuencias
                      verificación                        y riesgo
```

## Estructura

| Carpeta | Grupo | Contenido |
|---|---|---|
| `video/` | VIDEO | Su sistema original: `principal.py`, `modulos/`, modelos y sonido de alarma |
| `imagen/` | IMAGEN | `modulo_imagen/` (procesamiento), calibración y vigilante de alertas |
| `prediccion/` | PREDICCIÓN | Su `backend/` (motor de reglas y modelos entrenados) y el modelo de pose |
| `integracion/` | Integración | Cámara de VIDEO dentro de la página y eventos entre módulos |
| `app.py` | Integración | Página web (Streamlit) con las secciones Video, Imágenes y Predicciones |

Cada grupo trabaja **solo en su carpeta**. El código de VIDEO y de PREDICCIÓN se mantiene sin modificar; la integración se adapta a él desde `integracion/`.

Repositorios de origen:

- VIDEO: https://github.com/cesarcabanillas1921-bit/SistemaSupervisionExamenes
- IMAGEN: https://github.com/Zudo45/SistemaSupervisionExamenes-Imagen
- PREDICCIÓN: https://github.com/alessandrorr1007-debug/Prediccion_GD

## Ramas

| Rama | Uso |
|---|---|
| `main` | Versión estable, la que se presenta. Siempre debe funcionar |
| `video` | Cambios del grupo VIDEO en prueba |
| `imagen` | Cambios del grupo IMAGEN en prueba |
| `prediccion` | Cambios del grupo PREDICCIÓN en prueba |

Forma de trabajo:

1. Trabajar en la rama del grupo: `git switch imagen` (por ejemplo).
2. Modificar solo la carpeta del grupo y subir los cambios: `git push`.
3. Cuando funcione, abrir un **Pull Request** de la rama hacia `main` en GitHub para que se revise e integre.

## Inicio rápido (Windows)

1. Instalar **Python 3.12** marcando "Add Python to PATH".
2. Descargar el repositorio (`git clone` o **Code → Download ZIP**).
3. Doble clic en **`iniciar_app.bat`**: la primera vez crea el entorno e instala las librerías (necesita internet y tarda varios minutos); luego abre la página en `http://localhost:8501`.
4. En la sección **Video**, pulsar **Iniciar cámara**. Cuando un objeto no permitido se mantiene 5 segundos, la captura pasa automáticamente a **Imágenes** y luego a **Predicciones**.

Otras formas de uso:

```
iniciar_video.bat                                   # sistema de VIDEO en su propia ventana
.venv\Scripts\python -m streamlit run app.py        # página, desde la terminal
```

Desde otra PC de la misma red, la página se abre con la dirección que aparece en su barra lateral (`http://<IP>:8501`). El video de la cámara usa además el puerto `8502`.

### Acceso desde otras redes (link público)

Doble clic en **`iniciar_publico.bat`** (después de haber usado `iniciar_app.bat` una vez). Descarga `cloudflared` (programa oficial de Cloudflare, sin cuenta) y crea un link público `https://…trycloudflare.com`, que funciona desde cualquier red (datos móviles, otra wifi).

- El link y una **clave** aparecen en la ventana negra y en la barra lateral de la página.
- Quien entra por el link debe ingresar la clave y **solo puede mirar**: los botones de la cámara solo están en la laptop anfitriona.
- El link cambia cada vez que se inicia. Al cerrar la ventana negra, el link deja de funcionar.

## Datos y privacidad

Las capturas de las alarmas (`video/alertas/`) y los resultados generados (`resultados/`) **no se suben** al repositorio (ver `.gitignore`).
