# ============================================================
# VIGILANTE DE ALERTAS (sin interfaz)
# Procesa automaticamente cada captura que VIDEO guarda en
# alertas/. La aplicacion web hace lo mismo mientras esta abierta.
# Uso:
#   python vigilar_alertas.py
# ============================================================

import time

from modulo_imagen.config_imagen import (
    CARPETA_ALERTAS_VIDEO,
    INTERVALO_REVISION_SEG
)
from modulo_imagen.vigilante import revisar


if __name__ == "__main__":

    print(f"Vigilando: {CARPETA_ALERTAS_VIDEO}  (Ctrl+C para salir)")

    while True:

        for datos in revisar():
            r = datos["resumen"]["comparacion_video"]
            print(
                f"Procesada {datos['archivo']}: objeto={datos['alerta_video']['objeto']} "
                f"detecciones={r['detecciones_antes']}->{r['detecciones_despues']} "
                f"confianza={r['confianza_media_antes']:.2f}->{r['confianza_media_despues']:.2f} "
                f"({datos['segundos_proceso']}s)"
            )

        time.sleep(INTERVALO_REVISION_SEG)
