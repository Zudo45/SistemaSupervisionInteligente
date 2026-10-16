import os
import shutil

# ============================================================
# CONFIGURACIÓN
# ============================================================

DATASET_ORIGINAL = "dataset"
DATASET_FILTRADO = "dataset_filtrado"

CONJUNTOS = ["train", "valid", "test"]

# ID original -> ID nuevo
MAPEO_CLASES = {
    37: 0,  # person -> persona
    29: 1,  # backpack -> mochila
    14: 2,  # Electronic-devices-phone -> telefono
    35: 3,  # notebook -> cuaderno
    31: 4,  # book -> libro

    # Todos se unen como "audifonos"
    0: 5,   # AirPods
    10: 5,  # Electronic-devices-earphones
    11: 5,  # Electronic-devices-headphones

    28: 6,  # Watch -> reloj
    34: 7   # laptop -> laptop
}

NOMBRES_CLASES = [
    "persona",
    "mochila",
    "telefono",
    "cuaderno",
    "libro",
    "audifonos",
    "reloj",
    "laptop"
]


# ============================================================
# ELIMINAR VERSION ANTERIOR
# ============================================================

if os.path.exists(DATASET_FILTRADO):
    print("Eliminando dataset filtrado anterior...")
    shutil.rmtree(DATASET_FILTRADO)


# ============================================================
# CREAR ESTRUCTURA
# ============================================================

for conjunto in CONJUNTOS:

    os.makedirs(
        os.path.join(
            DATASET_FILTRADO,
            conjunto,
            "images"
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            DATASET_FILTRADO,
            conjunto,
            "labels"
        ),
        exist_ok=True
    )


# ============================================================
# PROCESAR DATASET
# ============================================================

for conjunto in CONJUNTOS:

    print(f"\nProcesando: {conjunto}")

    carpeta_imagenes = os.path.join(
        DATASET_ORIGINAL,
        conjunto,
        "images"
    )

    carpeta_labels = os.path.join(
        DATASET_ORIGINAL,
        conjunto,
        "labels"
    )

    # Si no existe la carpeta labels, continuar
    if not os.path.exists(carpeta_labels):

        print(
            f"No existe la carpeta labels en "
            f"{conjunto}. Se omite."
        )

        continue

    # Si no existe images, continuar
    if not os.path.exists(carpeta_imagenes):

        print(
            f"No existe la carpeta images en "
            f"{conjunto}. Se omite."
        )

        continue

    carpeta_salida_imagenes = os.path.join(
        DATASET_FILTRADO,
        conjunto,
        "images"
    )

    carpeta_salida_labels = os.path.join(
        DATASET_FILTRADO,
        conjunto,
        "labels"
    )

    total_imagenes = 0

    # ========================================================
    # RECORRER ETIQUETAS
    # ========================================================

    for archivo_label in os.listdir(carpeta_labels):

        if not archivo_label.endswith(".txt"):
            continue

        ruta_label = os.path.join(
            carpeta_labels,
            archivo_label
        )

        # Leer etiquetas
        with open(
            ruta_label,
            "r",
            encoding="utf-8"
        ) as archivo:

            lineas = archivo.readlines()

        nuevas_lineas = []

        # ====================================================
        # FILTRAR CLASES
        # ====================================================

        for linea in lineas:

            partes = linea.strip().split()

            if len(partes) < 5:
                continue

            try:
                clase_original = int(partes[0])
            except ValueError:
                continue

            # Comprobar si la clase nos interesa
            if clase_original in MAPEO_CLASES:

                nueva_clase = MAPEO_CLASES[
                    clase_original
                ]

                # Mantener coordenadas YOLO
                nuevas_lineas.append(
                    f"{nueva_clase} "
                    f"{partes[1]} "
                    f"{partes[2]} "
                    f"{partes[3]} "
                    f"{partes[4]}\n"
                )

        # ====================================================
        # SI LA IMAGEN TIENE UNA CLASE QUE NOS INTERESA
        # ====================================================

        if nuevas_lineas:

            nombre_base = os.path.splitext(
                archivo_label
            )[0]

            imagen_encontrada = None

            # Buscar imagen correspondiente
            for extension in [
                ".jpg",
                ".jpeg",
                ".png",
                ".webp"
            ]:

                posible_imagen = os.path.join(
                    carpeta_imagenes,
                    nombre_base + extension
                )

                if os.path.exists(posible_imagen):

                    imagen_encontrada = posible_imagen
                    break

            # =================================================
            # COPIAR IMAGEN Y ETIQUETA
            # =================================================

            if imagen_encontrada:

                shutil.copy2(
                    imagen_encontrada,
                    os.path.join(
                        carpeta_salida_imagenes,
                        os.path.basename(
                            imagen_encontrada
                        )
                    )
                )

                nueva_ruta_label = os.path.join(
                    carpeta_salida_labels,
                    archivo_label
                )

                with open(
                    nueva_ruta_label,
                    "w",
                    encoding="utf-8"
                ) as archivo:

                    archivo.writelines(
                        nuevas_lineas
                    )

                total_imagenes += 1

    print(
        f"Imágenes conservadas en {conjunto}: "
        f"{total_imagenes}"
    )


# ============================================================
# CREAR DATA.YAML
# ============================================================

ruta_yaml = os.path.join(
    DATASET_FILTRADO,
    "data.yaml"
)

with open(
    ruta_yaml,
    "w",
    encoding="utf-8"
) as archivo:

    archivo.write(
        "train: ../train/images\n"
    )

    archivo.write(
        "val: ../valid/images\n"
    )

    archivo.write(
        "test: ../test/images\n\n"
    )

    archivo.write(
        f"nc: {len(NOMBRES_CLASES)}\n"
    )

    archivo.write("names: [")

    archivo.write(
        ", ".join(
            f"'{nombre}'"
            for nombre in NOMBRES_CLASES
        )
    )

    archivo.write("]\n")


# ============================================================
# RESULTADO FINAL
# ============================================================

print("\n========================================")
print("DATASET FILTRADO CORRECTAMENTE")
print("========================================")

print(
    f"Clases finales: {len(NOMBRES_CLASES)}"
)

print(
    f"Ubicación: {DATASET_FILTRADO}"
)

print("\nClases:")

for i, nombre in enumerate(NOMBRES_CLASES):

    print(
        f"{i}: {nombre}"
    )

print("\nArchivo creado:")

print(
    f"{DATASET_FILTRADO}/data.yaml"
)
