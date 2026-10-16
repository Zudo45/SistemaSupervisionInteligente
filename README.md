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
2. Descargar el repositorio (`git clone` o **Code → Download ZIP**) y descomprimirlo.
3. Doble clic en **`iniciar_app.bat`**. Hace todo:
   - la primera vez instala las librerías (necesita internet y tarda varios minutos);
   - crea un **link público** (`https://…trycloudflare.com`) y una **clave** nuevos;
   - inicia la página y abre el navegador en `http://localhost:8501`.
4. **No cerrar la ventana negra** mientras se use la página (se puede minimizar). Al cerrarla se apagan la página y el link.
5. En la sección **Video**, pulsar **Iniciar cámara**. Cuando un objeto no permitido se mantiene 5 segundos, la captura pasa automáticamente a **Imágenes** y luego a **Predicciones**.

### Link público y clave

- Aparecen en la ventana negra y en la barra lateral de la página (**Acceso por internet**). Cambian cada vez que se inicia.
- Se comparten con quienes quieran ver la página desde otra red (datos móviles, otra wifi). Al entrar deben escribir la clave y **solo pueden mirar**: los botones de la cámara solo están en la laptop que inició la página (`http://localhost:8501`).
- `cloudflared` es el programa oficial de Cloudflare (no requiere cuenta); se descarga solo la primera vez en `herramientas/`.

### Otros lanzadores

```
iniciar_local.bat      # página sin link público (por ejemplo, sin internet)
iniciar_video.bat      # sistema de VIDEO en su propia ventana
```

Desde otra PC de la misma red también se puede entrar con la dirección de la barra lateral (`http://<IP>:8501`).

## Datos y privacidad

Las capturas de las alarmas (`video/alertas/`) y los resultados generados (`resultados/`) **no se suben** al repositorio (ver `.gitignore`).
