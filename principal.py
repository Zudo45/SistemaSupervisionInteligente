import cv2
import time
import unicodedata

from modulos.camara import (
    iniciar_camara,
    obtener_frame,
    cerrar_camara
)

from modulos.deteccion import (
    detectar_objetos,
    dibujar_detecciones
)

from modulos.registro import (
    guardar_alerta
)

from configuracion.config import (
    TIEMPO_ALERTA
)


# ============================================================
# OBJETOS NO PERMITIDOS
# ============================================================

OBJETOS_NO_PERMITIDOS = {
    "telefono",
    "celular",
    "phone",
    "audifonos",
    "airpods",
    "reloj",
    "cuaderno",
    "libro",
    "cuaderno/libro"
}


# ============================================================
# CONTROL DEL TIEMPO
# ============================================================

inicio_deteccion = {}

alerta_generada = {}

# Guarda la ultima vez que realmente
# vimos cada objeto.
ultima_vez_visto = {}


# ============================================================
# TOLERANCIA DEL TELEFONO
# ============================================================

# Si YOLO pierde momentaneamente el celular
# durante menos de este tiempo,
# NO reiniciamos el contador.
#
# Esto evita que el contador vuelva a 0
# por uno o varios frames perdidos.

TOLERANCIA_TELEFONO = 1.0


# ============================================================
# NORMALIZAR TEXTO
# ============================================================

def normalizar(texto):

    texto = str(
        texto
    ).lower().strip()

    texto = unicodedata.normalize(
        "NFD",
        texto
    )

    texto = "".join(
        c for c in texto
        if unicodedata.category(c) != "Mn"
    )

    return texto


# ============================================================
# VERIFICAR SI ES OBJETO NO PERMITIDO
# ============================================================

def es_no_permitido(clase):

    nombre = normalizar(
        clase
    )

    if nombre in OBJETOS_NO_PERMITIDOS:

        return True

    # Cuaderno / libro
    if "cuaderno" in nombre:

        return True

    if "libro" in nombre:

        return True

    # Telefono
    if "telefono" in nombre:

        return True

    if "celular" in nombre:

        return True

    if "phone" in nombre:

        return True

    # Audifonos
    if "audifono" in nombre:

        return True

    if "airpod" in nombre:

        return True

    # Reloj
    if "reloj" in nombre:

        return True

    if "watch" in nombre:

        return True

    return False


# ============================================================
# VERIFICAR SI ES TELEFONO
# ============================================================

def es_telefono(nombre):

    nombre = normalizar(
        nombre
    )

    if "telefono" in nombre:

        return True

    if "celular" in nombre:

        return True

    if "phone" in nombre:

        return True

    return False


# ============================================================
# EJECUTAR SISTEMA
# ============================================================

def ejecutar_sistema():

    print(
        "Sistema Inteligente de Supervision de Examenes iniciado"
    )

    camara = iniciar_camara()

    if camara is None:

        print(
            "No se pudo iniciar la camara"
        )

        return


    # ========================================================
    # BUCLE PRINCIPAL
    # ========================================================

    while True:


        # ====================================================
        # FRAME
        # ====================================================

        frame = obtener_frame(
            camara
        )

        if frame is None:

            break


        # ====================================================
        # DETECCION
        # ====================================================

        detecciones = detectar_objetos(
            frame
        )

        tiempo_actual = (
            time.time()
        )

        objetos_actuales = set()


        # ====================================================
        # PROCESAR DETECCIONES
        # ====================================================

        for deteccion in detecciones:

            clase = deteccion[
                "clase"
            ]

            nombre = normalizar(
                clase
            )


            # ------------------------------------------------
            # SOLO OBJETOS NO PERMITIDOS
            # ------------------------------------------------

            if es_no_permitido(
                clase
            ):

                objetos_actuales.add(
                    nombre
                )

                # ============================================
                # GUARDAR ULTIMA VEZ QUE VIMOS EL OBJETO
                # ============================================

                ultima_vez_visto[
                    nombre
                ] = tiempo_actual


                # ============================================
                # INICIO DEL CONTADOR
                # ============================================

                if (
                    nombre
                    not in inicio_deteccion
                ):

                    inicio_deteccion[
                        nombre
                    ] = tiempo_actual

                    alerta_generada[
                        nombre
                    ] = False

                    print(
                        f"INICIO CONTADOR: {clase}"
                    )


                # ============================================
                # TIEMPO TRANSCURRIDO
                # ============================================

                tiempo_visible = (
                    tiempo_actual
                    -
                    inicio_deteccion[
                        nombre
                    ]
                )

                print(
                    f"{clase}: "
                    f"{tiempo_visible:.1f} segundos"
                )


                # ============================================
                # ALERTA A LOS 5 SEGUNDOS
                # ============================================

                if (
                    tiempo_visible
                    >=
                    TIEMPO_ALERTA

                    and

                    not alerta_generada.get(
                        nombre,
                        False
                    )
                ):

                    guardar_alerta(
                        frame,
                        clase
                    )

                    alerta_generada[
                        nombre
                    ] = True

                    print(
                        f"ALERTA CONFIRMADA: {clase}"
                    )


        # ====================================================
        # MANTENER TELEFONO DURANTE PERDIDAS MOMENTANEAS
        # ====================================================

        objetos_en_seguimiento = list(
            inicio_deteccion.keys()
        )

        for objeto in objetos_en_seguimiento:

            # Si está detectado normalmente
            # no hacemos nada.
            if objeto in objetos_actuales:

                continue


            # ================================================
            # TELEFONO
            # ================================================

            if es_telefono(
                objeto
            ):

                ultima = (
                    ultima_vez_visto.get(
                        objeto,
                        0
                    )
                )

                tiempo_sin_ver = (
                    tiempo_actual
                    -
                    ultima
                )


                # ============================================
                # AUN CONSIDERAMOS QUE ES EL MISMO CELULAR
                # ============================================

                if (
                    tiempo_sin_ver
                    <=
                    TOLERANCIA_TELEFONO
                ):

                    objetos_actuales.add(
                        objeto
                    )

                    continue


            # ================================================
            # EL OBJETO REALMENTE DESAPARECIO
            # ================================================

            print(
                f"OBJETO RETIRADO: {objeto}"
            )

            if (
                objeto
                in inicio_deteccion
            ):

                del inicio_deteccion[
                    objeto
                ]

            if (
                objeto
                in alerta_generada
            ):

                del alerta_generada[
                    objeto
                ]

            if (
                objeto
                in ultima_vez_visto
            ):

                del ultima_vez_visto[
                    objeto
                ]


        # ====================================================
        # CALCULAR CONTADOR
        # ====================================================

        mayor_tiempo = 0

        for objeto in objetos_actuales:

            if (
                objeto
                in inicio_deteccion
            ):

                tiempo_objeto = (
                    tiempo_actual
                    -
                    inicio_deteccion[
                        objeto
                    ]
                )

                if (
                    tiempo_objeto
                    >
                    mayor_tiempo
                ):

                    mayor_tiempo = (
                        tiempo_objeto
                    )


        segundos_contador = int(
            mayor_tiempo
        )


        # ====================================================
        # MOSTRAR CONTADOR SIEMPRE
        # ====================================================

        cv2.putText(
            frame,
            (
                f"Tiempo: "
                f"{segundos_contador}s / "
                f"{TIEMPO_ALERTA}s"
            ),
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            3
        )


        # ====================================================
        # MOSTRAR ALERTA
        # ====================================================

        alerta_activa = False

        for objeto in objetos_actuales:

            if alerta_generada.get(
                objeto,
                False
            ):

                alerta_activa = True

                break


        if alerta_activa:

            cv2.putText(
                frame,
                "ALERTA: MATERIAL NO PERMITIDO",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                3
            )


        # ====================================================
        # DIBUJAR DETECCIONES
        # ====================================================

        frame = dibujar_detecciones(
            frame,
            detecciones
        )


        # ====================================================
        # MOSTRAR CAMARA
        # ====================================================

        cv2.imshow(
            "Sistema de Supervision Inteligente",
            frame
        )

        tecla = (
            cv2.waitKey(1)
            &
            0xFF
        )

        if tecla == ord(
            "q"
        ):

            break


    # ========================================================
    # CERRAR
    # ========================================================

    cerrar_camara(
        camara
    )

    cv2.destroyAllWindows()

    print(
        "Sistema de supervision finalizado"
    )


# ============================================================
# EJECUTAR
# ============================================================

if __name__ == "__main__":

    ejecutar_sistema()