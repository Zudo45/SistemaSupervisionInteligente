# Organización por equipos

Cada equipo mantiene sus modelos, pesos, entrenamiento y entorno. PREDICCIÓN conserva un `modulo.json` con el repositorio y el estado de integración, además de un adaptador pequeño para convertir resultados al contrato común.

- `video/adapter.py`: recibe fotogramas ya inferidos con `clase`, `confianza` y `timestamp`.
- `imagen/adapter.py`: recibe la exportación JSON de IMAGEN y conserva objetos `confirmada` o `recuperada`.
- `audio/`: pendiente hasta recibir el repositorio y un ejemplo real de salida.
- `prediccion/`: une eventos y calcula un índice explicable para revisión humana.

Los adaptadores no ejecutan entrenamientos ni alteran los resultados originales. Las referencias al evento de origen permanecen en sus identificadores y versiones de modelo.
