# Configuración del sistema

RUTA_MODELO = "modelo_ia/yolov8n.pt"

CAMARA = 0

CONFIANZA_MINIMA = 0.55

# Tiempo que debe permanecer visible un objeto
# antes de generar una alerta
TIEMPO_ALERTA = 5

CARPETA_ALERTAS = "alertas"
CARPETA_CAPTURAS = "capturas"

# Objetos que generan alerta
OBJETOS_NO_PERMITIDOS = [
    "cell phone",
    "book",
    "cuaderno",
    "headphones"
]

# Traducción para mostrar en pantalla
TRADUCCION_CLASES = {
    "person": "Persona",
    "cell phone": "Telefono movil",
    "backpack": "Mochila",
    "book": "Cuaderno",
    "headphones": "Audifonos"
}