# Sistema de Supervisión Inteligente

Sistema de supervisión de exámenes mediante inteligencia artificial y visión por computadora.

## ¿De qué trata el proyecto?

El proyecto busca apoyar la supervisión de evaluaciones mediante una cámara y modelos de inteligencia artificial entrenados con YOLO.

El sistema analiza en tiempo real lo que aparece frente a la cámara y permite detectar distintos elementos relacionados con el estudiante.

## ¿Qué hace?

- Detecta personas.
- Detecta mochilas.
- Detecta teléfonos.
- Detecta cuadernos/libros.
- Detecta audífonos.
- Detecta relojes.
- Detecta laptops.
- Muestra las detecciones mediante cuadros de colores.
- Los elementos permitidos se muestran en azul.
- Los elementos no permitidos se muestran en rojo.
- Inicia un contador cuando detecta un elemento no permitido.
- Si permanece durante 5 segundos, genera una alerta.
- Guarda una captura de la alerta.
- Reproduce un sonido de alerta.
- Realiza seguimiento interno del teléfono para evitar que el contador se reinicie cuando el celular se mueve o deja de ser detectado momentáneamente.
- El contador se reinicia cuando el objeto deja de aparecer realmente y posteriormente vuelve a ser detectado.

---

## 1. Instalar Python

Se recomienda utilizar Python 3.12.

Descargar desde:

https://www.python.org/downloads/release/python-31210/

Durante la instalación marcar:

```text
Add Python to PATH
```

---

## 2. Descargar el proyecto

Clonar el repositorio:

```bash
git clone https://github.com/cesarcabanillas1921-bit/SistemaSupervisionExamenes.git
```

Entrar a la carpeta:

```bash
cd SistemaSupervisionExamenes
```

---

## 3. Instalar las librerías

Ejecutar:

```bash
pip install -r requisitos.txt
```

La primera ejecución puede demorar un poco debido a la carga y descarga de algunos componentes necesarios para los modelos.

---

## 4. Ejecutar el sistema

Conectar o habilitar la cámara y ejecutar:

```bash
python principal.py
```

---

## 5. Salir del sistema

Presionar:

```text
Q
```

---

## Modelos utilizados

El sistema utiliza diferentes modelos especializados para mejorar la detección.

### Modelo principal

```text
modelo_ia/best.pt
```

Se utiliza para detectar elementos como:

- Persona
- Mochila
- Cuaderno/Libro
- Audífonos
- Reloj

### Modelo especializado en teléfono y laptop

```text
modelo_ia/best_phone_laptop.pt
```

Se utiliza para mejorar la diferenciación entre:

- Teléfono
- Laptop

### Modelo general YOLO

```text
modelo_ia/yolov8n.pt
```

Se utiliza como apoyo adicional para la detección de teléfonos.

### Modelo especializado en audífonos

```text
modelo_audifonos/best_audifonos.pt
```

Se utiliza para reforzar la detección de audífonos, especialmente cuando se encuentran colocados en la oreja.

---

## Sonido de alerta

El archivo de audio utilizado para las alertas se encuentra en:

```text
sonido/alerta.mp3
```

No eliminar los modelos ni el archivo de sonido, ya que son necesarios para el funcionamiento del sistema.

---

## Funcionamiento de las alertas

Cuando el sistema detecta un objeto no permitido:

```text
0s → 1s → 2s → 3s → 4s → 5s
```

Al llegar a 5 segundos:

- Se genera una alerta.
- Se guarda una captura.
- Se reproduce un sonido.
- El contador continúa mientras el objeto permanezca visible.

Ejemplo:

```text
Tiempo: 6s / 5s
Tiempo: 7s / 5s
Tiempo: 8s / 5s
```

En el caso del teléfono, el sistema realiza seguimiento interno para evitar reinicios innecesarios del contador cuando el objeto se mueve o deja de ser detectado momentáneamente.

---

## Estructura principal

```text
SistemaSupervisionExamenes/
│
├── alertas/
├── capturas/
│
├── configuracion/
│   └── config.py
│
├── modelo_audifonos/
│   ├── best_audifonos.pt
│   └── earphones.yaml
│
├── modelo_ia/
│   ├── best.pt
│   ├── best_phone_laptop.pt
│   ├── last.pt
│   └── yolov8n.pt
│
├── modulos/
│   ├── camara.py
│   ├── comportamiento.py
│   ├── deteccion.py
│   └── registro.py
│
├── sonido/
│   └── alerta.mp3
│
├── entrenar_modelo.py
├── filtrar_dataset.py
├── principal.py
├── probar_audifonos.py
├── README.md
└── requisitos.txt
```

---

## Mejoras futuras

- Ampliar el conjunto de datos con condiciones de baja iluminación.
- Agregar imágenes con alta exposición a la luz.
- Agregar imágenes de baja resolución.
- Mejorar la detección de objetos parcialmente ocultos.
- Mejorar la detección de cuadernos y hojas.
- Mejorar la detección de relojes y smartwatches.
- Optimizar el rendimiento para reducir el uso de CPU.
- Integrar análisis de comportamiento del estudiante.
- Detectar movimientos o conductas sospechosas durante una evaluación.

---

## Apoyo

Si este proyecto te resulta útil:

- ⭐ Dale una estrella al repositorio.
- 🍴 Haz un fork del proyecto.
- 💬 Comparte sugerencias o mejoras.
