# ============================================================
# MODULO IMAGEN - APLICACION WEB
# Uso:
#   python -m streamlit run app.py
# ============================================================

import os
import sys
import json
import base64
import socket
from datetime import datetime

import cv2

# Cada grupo esta en su carpeta: se agregan al path antes de importar
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagen"))

import pandas as pd
import streamlit as st

from modulo_imagen.config_imagen import (
    CARPETA_ALERTAS_VIDEO,
    CARPETA_EN_VIVO,
    INTERVALO_REVISION_SEG,
    LADO_MINIMO_REGION
)
from modulo_imagen.pipeline import procesar_frame, OPCIONES_POR_DEFECTO
from modulo_imagen.exportacion import generar_json
from modulo_imagen.evaluacion import evaluar, resumen_por_escenario
from modulo_imagen.segmentacion import aislar_objeto
from modulo_imagen.mejoramiento import ampliar
from modulo_imagen import vigilante
from modulo_imagen.verificacion import emparejar
from integracion.camara_video import SesionCamara, TIEMPO_ALERTA, es_no_permitido
from integracion import prediccion as integracion_prediccion
from modulo_imagen.visualizacion import (
    dibujar_video,
    superponer_mascara,
    histograma_gris,
    a_rgb
)


st.set_page_config(
    page_title="Supervision de Examenes",
    page_icon="🖼️",
    layout="wide"
)


# ============================================================
# ESTILOS
# ============================================================

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1400px;}
.encabezado {
    padding: 1.4rem 1.6rem; border-radius: 16px; margin-bottom: 1.2rem;
    background: linear-gradient(120deg, #0F766E 0%, #134E4A 45%, #131C2E 100%);
    border: 1px solid #1F2A40;
}
.encabezado h1 {margin: 0; font-size: 2rem; color: #F8FAFC;}
.encabezado p {margin: .3rem 0 0; color: #CCFBF1; font-size: .95rem;}
.flujo {display: flex; gap: .4rem; flex-wrap: wrap; margin-top: .8rem;}
.flujo span {
    background: rgba(15, 23, 42, .55); color: #E2E8F0; border-radius: 999px;
    padding: .2rem .7rem; font-size: .78rem; border: 1px solid rgba(45, 212, 191, .35);
}
.pastilla {
    display: inline-block; border-radius: 999px; padding: .15rem .65rem;
    font-size: .8rem; font-weight: 600; margin-right: .3rem;
}
.p-alerta {background: #7F1D1D; color: #FECACA;}
.p-ok {background: #134E4A; color: #99F6E4;}
.p-info {background: #1E293B; color: #CBD5E1; border: 1px solid #334155;}
.vivo {display: inline-flex; align-items: center; gap: .45rem; color: #99F6E4; font-size: .85rem;}
.vivo::before {
    content: ""; width: 9px; height: 9px; border-radius: 50%; background: #2DD4BF;
    box-shadow: 0 0 0 0 rgba(45, 212, 191, .7); animation: pulso 1.8s infinite;
}
@keyframes pulso {
    0% {box-shadow: 0 0 0 0 rgba(45, 212, 191, .6);}
    70% {box-shadow: 0 0 0 9px rgba(45, 212, 191, 0);}
    100% {box-shadow: 0 0 0 0 rgba(45, 212, 191, 0);}
}
.contador {text-align: center; font-size: 1.05rem; padding-top: .35rem; color: #E2E8F0;}
.contador b {color: #2DD4BF;}
div[data-testid="stMetric"] {
    background: #131C2E; border: 1px solid #1F2A40; border-radius: 12px; padding: .7rem 1rem;
}
.vacio {
    text-align: center; padding: 3rem 1rem; border: 1px dashed #334155;
    border-radius: 16px; color: #94A3B8;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# ACCESO POR INTERNET (TUNEL)
# Con iniciar_publico.bat la pagina tiene un link publico. Las
# visitas que llegan por ese link necesitan la clave y no pueden
# manejar la camara; en la laptop anfitriona se entra directo.
# ============================================================

RUTA_TUNEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados", "tunel.json")


def datos_tunel():

    try:
        with open(RUTA_TUNEL, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def es_visita_remota():

    cabeceras = st.context.headers
    host = cabeceras.get("Host", "")

    return bool(cabeceras.get("Cf-Connecting-Ip")) or host.endswith(".trycloudflare.com")


def pedir_clave():

    tunel = datos_tunel()

    if not es_visita_remota() or not tunel or st.session_state.get("clave_ok"):
        return

    _, centro, _ = st.columns([1, 1.2, 1])

    with centro:

        st.markdown("<div style='height:12vh'></div>", unsafe_allow_html=True)
        st.markdown("## Sistema de Supervisión Inteligente")
        st.caption("Acceso por internet: ingresa la clave que te compartieron.")

        with st.form("clave_visita"):

            clave = st.text_input("Clave", type="password")

            if st.form_submit_button("Ingresar", width="stretch"):

                if clave.strip().upper() == tunel["clave"]:
                    st.session_state["clave_ok"] = True
                    st.rerun()

                st.error("Clave incorrecta")

    st.stop()


pedir_clave()


# ============================================================
# UTILIDADES
# ============================================================

def ip_local():

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "localhost"


def mostrar(imagen, titulo=None):

    st.image(a_rgb(imagen), caption=titulo, width="stretch")


def pastilla(texto, tipo="info"):

    return f"<span class='pastilla p-{tipo}'>{texto}</span>"


@st.cache_data(show_spinner="Analizando imagen...")
def procesar(paquete, opciones):

    return procesar_frame(paquete, dict(opciones))


@st.cache_data(show_spinner=False)
def cargar_alerta(nombre_archivo, modificado):
    """
    `modificado` invalida la cache si el archivo cambia.
    """

    return vigilante.cargar_paquete(nombre_archivo)


def tabla_calidad(calidad):

    filas = []

    for metrica in ("brillo", "contraste", "nitidez", "ruido", "entropia"):
        filas.append({
            "metrica": metrica,
            "antes": calidad["capturado"][metrica],
            "despues": calidad["mejorado"][metrica]
        })

    if "vs_original" in calidad:
        for metrica in ("psnr", "ssim"):
            filas.append({
                "metrica": f"{metrica} (vs. original)",
                "antes": calidad["vs_original"]["capturado"][metrica],
                "despues": calidad["vs_original"]["mejorado"][metrica]
            })

    return pd.DataFrame(filas).set_index("metrica")


# ============================================================
# ANALISIS DE UN FOTOGRAMA (pestanas)
# ============================================================

def mostrar_analisis(resultado, clave):

    imagenes = resultado["imagenes"]
    registro = resultado["registro_video"]
    objetos = resultado["objetos"]

    pestanas = st.tabs([
        "📥 Recepcion",
        "🧹 Preprocesamiento",
        "✨ Mejoramiento",
        "✂️ Segmentacion",
        "📐 Caracteristicas",
        "✅ Verificacion",
        "📤 Exportacion"
    ])

    # --------------------------------------------------------
    # RECEPCION
    # --------------------------------------------------------

    with pestanas[0]:

        st.caption(
            "Fotograma y detecciones tal como los entrego VIDEO (su modelo best.pt, sin cambios). "
            "La simulacion de condiciones de captura no afecta a esta pestana."
        )

        c1, c2 = st.columns([3, 2])

        with c1:
            mostrar(dibujar_video(imagenes["original"], registro["detecciones"]),
                    f"{registro['origen']} · {registro['ancho']}x{registro['alto']}")

        with c2:

            alerta = registro.get("alerta")

            if alerta:
                st.markdown(
                    pastilla(f"Alarma: {alerta['objeto']}", "alerta")
                    + pastilla(alerta["fecha"] or "sin fecha"),
                    unsafe_allow_html=True
                )
                if not alerta.get("objeto_redetectado"):
                    st.caption(
                        "El modelo de VIDEO no vuelve a encontrar el objeto de la alarma "
                        "en esta captura (su deteccion parpadea entre fotogramas). "
                        "La imagen se procesa igual."
                    )

            if registro["detecciones"]:
                st.dataframe(pd.DataFrame(registro["detecciones"]), hide_index=True)
            else:
                st.info("VIDEO no reporto detecciones en este fotograma.")

            with st.expander("Registro recibido (JSON)"):
                st.json(registro)

    # --------------------------------------------------------
    # PREPROCESAMIENTO
    # --------------------------------------------------------

    with pestanas[1]:

        if resultado["decisiones"]:
            st.markdown("**Decisiones del preprocesamiento adaptativo**")
            for decision in resultado["decisiones"]:
                st.markdown(f"- {decision}")

        op = resultado["opciones"]
        st.caption(
            f"Filtro de ruido: {op['ruido']} ({op['fuerza_ruido']}) · "
            f"Iluminacion: {op['iluminacion']} (CLAHE {op['limite_clahe']})"
        )

        c1, c2 = st.columns(2)

        with c1:
            mostrar(imagenes["capturado"], "Recibido por IMAGEN")
        with c2:
            mostrar(imagenes["preprocesado"], "Preprocesado")

        st.markdown("**Histograma de intensidades**")
        st.line_chart(pd.DataFrame({
            "recibido": histograma_gris(imagenes["capturado"]),
            "preprocesado": histograma_gris(imagenes["preprocesado"])
        }), color=["#64748B", "#2DD4BF"])

    # --------------------------------------------------------
    # MEJORAMIENTO
    # --------------------------------------------------------

    with pestanas[2]:

        op = resultado["opciones"]
        st.caption(f"Contraste: {op['contraste']} · Nitidez (unsharp masking): {op['nitidez']}")

        c1, c2 = st.columns(2)

        with c1:
            mostrar(imagenes["capturado"], "Antes")
        with c2:
            mostrar(imagenes["mejorado"], "Despues")

        calidad = resultado["calidad"]

        columnas = st.columns(4)

        for columna, metrica in zip(columnas, ("brillo", "contraste", "nitidez", "ruido")):

            antes = calidad["capturado"][metrica]
            despues = calidad["mejorado"][metrica]

            columna.metric(
                metrica.capitalize(),
                f"{despues:.1f}",
                f"{despues - antes:+.1f}",
                delta_color="inverse" if metrica == "ruido" else "normal"
            )

        st.dataframe(tabla_calidad(calidad))

    # --------------------------------------------------------
    # SEGMENTACION
    # --------------------------------------------------------

    with pestanas[3]:

        if not objetos:
            st.info("No hay objetos detectados por VIDEO para segmentar.")
        else:
            etiquetas = [
                f"{o['id']}: {o['deteccion']['clase']} ({o['deteccion']['confianza_video']:.2f})"
                for o in objetos
            ]

            # Por defecto, el primer objeto no permitido
            inicial = next(
                (i for i, o in enumerate(objetos) if o["deteccion"]["clase"] != "persona"), 0
            )

            elegido = objetos[etiquetas.index(
                st.selectbox("Objeto", etiquetas, index=inicial, key=f"seg_{clave}")
            )]
            segmento = elegido["segmento"]

            recorte = segmento["recorte"]
            ampliado, _ = ampliar(recorte, LADO_MINIMO_REGION)

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                mostrar(ampliado, "Region (ampliada, bicubica)")
            with c2:
                st.image(segmento["mascara"], caption="Mascara (GrabCut)", width="stretch")
            with c3:
                mostrar(superponer_mascara(recorte, segmento["mascara"], segmento["contorno"]),
                        "Contorno")
            with c4:
                mostrar(aislar_objeto(recorte, segmento["mascara"]), "Objeto aislado")

            st.image(segmento["bordes"], caption="Bordes (Canny)", width=360)

    # --------------------------------------------------------
    # CARACTERISTICAS
    # --------------------------------------------------------

    with pestanas[4]:

        if objetos:
            st.dataframe(pd.DataFrame([
                {"id": o["id"], "clase": o["deteccion"]["clase"], **o["caracteristicas"]}
                for o in objetos
            ]).set_index("id"))

            st.caption(
                "Tamano y posicion (area, centro, zona 3x3), forma (circularidad, solidez, "
                "relleno), color dominante (HSV), textura (nitidez, densidad de bordes) "
                "y relacion con la persona."
            )
        else:
            st.info("No hay objetos detectados.")

    # --------------------------------------------------------
    # VERIFICACION
    # --------------------------------------------------------

    with pestanas[5]:

        st.caption(
            "El mismo modelo de VIDEO se ejecuta sobre la imagen que recibe IMAGEN (antes) "
            "y sobre la imagen procesada (despues). Cada objeto que VIDEO entrego se busca "
            "en ambas: confirmada, no confirmada, recuperada o no detectada."
        )

        c1, c2 = st.columns(2)

        with c1:
            mostrar(dibujar_video(imagenes["capturado"], resultado["detecciones_capturado"]),
                    "Antes: VIDEO sobre la imagen recibida")
        with c2:
            mostrar(dibujar_video(imagenes["mejorado"], resultado["detecciones_despues"]),
                    "Despues: VIDEO sobre la imagen procesada por IMAGEN")

        comparacion = resultado["resumen"]["comparacion_video"]

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Detecciones", comparacion["detecciones_despues"],
                  comparacion["detecciones_despues"] - comparacion["detecciones_antes"])
        m2.metric("Confianza media", f"{comparacion['confianza_media_despues']:.2f}",
                  f"{comparacion['confianza_media_despues'] - comparacion['confianza_media_antes']:+.2f}")
        m3.metric("Mantenidas", comparacion["mantenidas"])
        m4.metric("Perdidas", comparacion["perdidas"], delta_color="off")

        if objetos:
            st.dataframe(pd.DataFrame([
                {
                    "id": o["id"],
                    "clase": o["deteccion"]["clase"],
                    "VIDEO original": o["deteccion"]["confianza_video"],
                    "antes (recibida)": o["deteccion"]["confianza_capturada"],
                    "despues (procesada)": o["deteccion"]["confianza_procesada"],
                    "variacion": o["deteccion"]["variacion_confianza"],
                    "estado": o["deteccion"]["estado"]
                }
                for o in objetos
            ]).set_index("id"))

    # --------------------------------------------------------
    # EXPORTACION
    # --------------------------------------------------------

    with pestanas[6]:

        datos = generar_json([resultado])

        st.caption(
            "Al terminar el procesamiento, IMAGEN envía automáticamente a PREDICCIÓN "
            "la imagen mejorada (sin recuadros) y estos datos."
        )

        st.image(a_rgb(imagenes["mejorado"]), caption="Imagen mejorada que recibe PREDICCIÓN",
                 width=420)

        st.json(datos, expanded=False)


# ============================================================
# BARRA LATERAL
# ============================================================

CARPETA_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

# Se acepta el logo en PNG o JPG
RUTA_LOGO = next(
    (os.path.join(CARPETA_ASSETS, n) for n in ("logo.png", "logo.jpg", "logo.jpeg")
     if os.path.exists(os.path.join(CARPETA_ASSETS, n))),
    None
)

with st.sidebar:

    if RUTA_LOGO:
        with open(RUTA_LOGO, "rb") as f:
            logo = base64.b64encode(f.read()).decode()
        tipo = "png" if RUTA_LOGO.endswith(".png") else "jpeg"
        st.markdown(
            "<div style='text-align:center;margin-bottom:.6rem'>"
            f"<img src='data:image/{tipo};base64," + logo + "' style='width:120px;"
            "background:#fff;border-radius:50%;padding:10px'></div>",
            unsafe_allow_html=True
        )

    st.markdown("### Sistema de Supervisión Inteligente")
    st.caption("Aula Inteligente para Educación Universitaria")

    modulo = st.radio(
        "Módulo",
        ["VIDEO", "IMAGEN", "PREDICCION"],
        format_func={
            "VIDEO": ":material/videocam: Video",
            "IMAGEN": ":material/image: Imágenes",
            "PREDICCION": ":material/psychology: Predicciones"
        }.get,
        label_visibility="collapsed"
    )

    segmentacion = "grabcut"

    if modulo == "IMAGEN":
        st.divider()
        st.markdown("**Segmentacion**")
        segmentacion = st.selectbox(
            "Metodo de segmentacion",
            ["grabcut", "otsu"],
            format_func={"grabcut": "GrabCut (modelos de color)", "otsu": "Otsu (umbral automatico)"}.get,
            label_visibility="collapsed"
        )

    tunel = datos_tunel()

    if tunel and not es_visita_remota():
        st.divider()
        st.markdown("**Acceso por internet**")
        st.code(tunel["pagina"], language=None)
        st.markdown(f"Clave para las visitas: **{tunel['clave']}**")

    if not es_visita_remota():
        st.divider()
        st.markdown("**Acceso desde otra PC (misma red)**")
        st.code(f"http://{ip_local()}:8501", language=None)


# El analisis principal trabaja siempre sobre la captura real de VIDEO
opciones = {"segmentacion": segmentacion}

opciones_clave = tuple(sorted({**OPCIONES_POR_DEFECTO, **opciones}.items()))


# ============================================================
# MODULO VIDEO
# ============================================================

@st.cache_resource
def obtener_sesion():
    """
    Sesion de camara compartida por todos los que abren la pagina.
    """

    return SesionCamara()


def es_anfitrion():
    """
    Solo la laptop donde corre la pagina puede manejar la camara.
    """

    return st.context.ip_address in (None, "127.0.0.1", "::1") and not es_visita_remota()


NOMBRES_OBJETO = {
    "telefono": "Teléfono",
    "laptop": "Laptop",
    "cuaderno": "Cuaderno/Libro",
    "libro": "Cuaderno/Libro",
    "mochila": "Mochila",
    "audifonos": "Audífonos",
    "reloj": "Reloj",
    "persona": "Persona",
    "desconocido": "Desconocido"
}

ICONOS_OBJETO = {
    "Teléfono": ":material/smartphone:",
    "Laptop": ":material/laptop_windows:",
    "Cuaderno/Libro": ":material/menu_book:",
    "Mochila": ":material/backpack:",
    "Audífonos": ":material/headphones:",
    "Reloj": ":material/watch:",
    "Persona": ":material/person:"
}


def nombre_objeto(clase):

    return NOMBRES_OBJETO.get(clase, str(clase).capitalize())


def lista_de_alertas():
    """
    Todas las capturas de la alarma guardadas en alertas/, de la mas
    reciente a la mas antigua.
    """

    filas = []

    for archivo in vigilante.alertas_en_carpeta():

        ruta = os.path.join(CARPETA_ALERTAS_VIDEO, archivo)
        datos = vigilante.leer_alerta(archivo)

        if datos["fecha"]:
            fecha, hora = datos["fecha"].split(" ")
            fecha = "/".join(reversed(fecha.split("-")))
        else:
            modificado = datetime.fromtimestamp(os.path.getmtime(ruta))
            fecha, hora = modificado.strftime("%d/%m/%Y"), modificado.strftime("%H:%M:%S")

        objeto = nombre_objeto(datos["objeto"])

        filas.append({
            "archivo": archivo,
            "ruta": ruta,
            "modificado": os.path.getmtime(ruta),
            "objeto": objeto,
            "nombre": f"Uso de {objeto.lower()} detectado",
            "fecha": fecha,
            "hora": hora,
            "extension": os.path.splitext(archivo)[1].lstrip(".").upper()
        })

    return sorted(filas, key=lambda f: f["modificado"], reverse=True)


@st.cache_data(show_spinner=False, max_entries=200)
def miniatura(ruta, modificado):

    imagen = cv2.imread(ruta)

    if imagen is None:
        return None

    imagen = cv2.resize(imagen, (96, int(96 * imagen.shape[0] / imagen.shape[1])))

    return "data:image/jpeg;base64," + base64.b64encode(
        cv2.imencode(".jpg", imagen)[1].tobytes()
    ).decode()


@st.fragment(run_every=1.0)
def estado_en_vivo():

    sesion = obtener_sesion()

    if not sesion.activa:
        texto = "⚪ Apagada"
    elif not sesion.modelo_listo.is_set():
        texto = "⏳ Cargando modelo"
    elif not sesion.fotograma_jpeg:
        texto = "⏳ Abriendo cámara"
    elif sesion.alerta_activa:
        texto = "🔴 Alerta"
    else:
        texto = "🟢 En vivo"

    st.markdown(f"<div style='text-align:right;font-weight:600'>{texto}</div>",
                unsafe_allow_html=True)

    # Si la camara cambia de estado por su cuenta, se actualizan los botones
    if st.session_state.get("camara_activa") != sesion.activa:
        st.session_state["camara_activa"] = sesion.activa
        st.rerun()


@st.fragment(run_every=1.0)
def contador_alarma():

    sesion = obtener_sesion()

    if sesion.error:
        st.error(sesion.error)

    st.progress(
        min(sesion.tiempo_mayor / TIEMPO_ALERTA, 1.0),
        text=f"Objeto no permitido a la vista: {sesion.tiempo_mayor:.1f} s de {TIEMPO_ALERTA} s"
    )


@st.fragment(run_every=1.0)
def detecciones_actuales():

    sesion = obtener_sesion()

    conteo = {nombre: 0 for nombre in ICONOS_OBJETO}

    if sesion.activa:
        for d in sesion.detecciones:
            nombre = nombre_objeto(d["clase"])
            if nombre in conteo:
                conteo[nombre] += 1

    for nombre, icono in ICONOS_OBJETO.items():

        cantidad = conteo[nombre]
        alerta = cantidad and nombre != "Persona" and nombre != "Mochila" and nombre != "Laptop"

        c1, c2 = st.columns([4, 1], vertical_alignment="center")
        c1.markdown(f"{icono}&nbsp;&nbsp;{nombre}")
        c2.markdown(pastilla(str(cantidad), "alerta" if alerta else "info"), unsafe_allow_html=True)


@st.fragment(run_every=1.0)
def detalle_del_evento():

    sesion = obtener_sesion()
    alertas = lista_de_alertas()

    # Aviso y seleccion automatica cuando llega una alerta nueva
    if alertas:
        ultima = alertas[0]["archivo"]
        if st.session_state.get("ultima_alerta") != ultima:
            if st.session_state.get("ultima_alerta"):
                st.toast(f"🚨 {alertas[0]['nombre']} · enviada a IMAGEN")
            st.session_state["ultima_alerta"] = ultima
            st.session_state["alerta_sel"] = ultima

    if not alertas:
        st.info("Aún no hay alertas. Aparecerán aquí cuando suene la alarma de VIDEO.")
        return

    elegida = next(
        (a for a in alertas if a["archivo"] == st.session_state.get("alerta_sel")),
        alertas[0]
    )

    st.error(elegida["nombre"], icon=":material/error:")

    c1, c2 = st.columns([1.1, 1])

    with c1:
        st.image(elegida["ruta"], width="stretch")

    with c2:
        for icono, etiqueta, valor in (
            (":material/alarm:", "Nombre de alerta", elegida["nombre"]),
            (ICONOS_OBJETO.get(elegida["objeto"], ":material/help:"), "Objeto identificado", elegida["objeto"]),
            (":material/calendar_month:", "Fecha", elegida["fecha"]),
            (":material/schedule:", "Hora", elegida["hora"]),
            (":material/image:", "Extensión", elegida["extension"])
        ):
            # Los ":" del valor (por ejemplo la hora) se escapan para que
            # no se interpreten como codigos de iconos
            valor = str(valor).replace(":", "&#58;")
            st.markdown(f"{icono} <small>{etiqueta}</small><br><b>{valor}</b>", unsafe_allow_html=True)

    procesada = elegida["archivo"] not in vigilante.pendientes()
    st.caption("✅ Procesada por IMAGEN" if procesada else "⏳ IMAGEN está procesando esta captura")

    with open(elegida["ruta"], "rb") as f:
        st.download_button(
            "Descargar imagen", f.read(), file_name=elegida["archivo"],
            icon=":material/download:", type="primary", width="stretch",
            key=f"descargar_{elegida['archivo']}"
        )


@st.fragment(run_every=2.0)
def lista_de_capturas():

    alertas = lista_de_alertas()

    t1, t2, t3 = st.columns([3, 1.2, 1.2], vertical_alignment="bottom")

    t1.markdown("##### :material/photo_library: Lista de capturas y alertas")

    fecha = t2.date_input("Fecha", value=None, format="DD/MM/YYYY",
                          label_visibility="collapsed", key="filtro_fecha")

    tipos = ["Todo tipo"] + sorted({a["objeto"] for a in alertas})
    tipo = t3.selectbox("Tipo", tipos, label_visibility="collapsed", key="filtro_tipo")

    if fecha:
        alertas = [a for a in alertas if a["fecha"] == fecha.strftime("%d/%m/%Y")]

    if tipo != "Todo tipo":
        alertas = [a for a in alertas if a["objeto"] == tipo]

    if not alertas:
        st.caption("No hay capturas para mostrar.")
        return

    tabla = pd.DataFrame([
        {
            "#": i + 1,
            "Captura": miniatura(a["ruta"], a["modificado"]),
            "Nombre de alerta": a["nombre"],
            "Objeto identificado": a["objeto"],
            "Fecha": a["fecha"],
            "Hora": a["hora"],
            "Extensión": a["extension"]
        }
        for i, a in enumerate(alertas)
    ])

    evento = st.dataframe(
        tabla,
        hide_index=True,
        width="stretch",
        row_height=48,
        on_select="rerun",
        selection_mode="single-row",
        key="tabla_capturas",
        column_config={
            "#": st.column_config.NumberColumn(width="small"),
            "Captura": st.column_config.ImageColumn(width="small")
        }
    )

    filas = evento.selection.rows

    if filas and st.session_state.get("alerta_sel") != alertas[filas[0]]["archivo"]:
        st.session_state["alerta_sel"] = alertas[filas[0]]["archivo"]
        st.rerun()


def mostrar_modulo_video():

    t1, t2 = st.columns([3, 2], vertical_alignment="bottom")

    with t1:
        st.markdown("# :material/videocam: Video")
        st.caption("Gestión de capturas de imágenes y alertas detectadas durante los exámenes.")

    sesion = obtener_sesion()
    st.session_state["camara_activa"] = sesion.activa

    with t2:

        if es_anfitrion():

            b1, b2 = st.columns(2)

            if b1.button("Iniciar cámara", icon=":material/play_arrow:", type="primary",
                         width="stretch", disabled=sesion.activa):
                sesion.iniciar()
                st.rerun()

            if b2.button("Detener cámara", icon=":material/stop:", width="stretch",
                         disabled=not sesion.activa):
                sesion.detener()
                st.rerun()

    c1, c2, c3 = st.columns([1.55, 0.95, 1.55])

    with c1, st.container(border=True):

        h1, h2 = st.columns([2, 1], vertical_alignment="center")
        h1.markdown("##### :material/videocam: Cámara en tiempo real")

        with h2:
            estado_en_vivo()

        if sesion.video.puerto:
            # Flujo MJPEG servido por la laptop de la pagina, con el mismo
            # nombre/IP con el que se abrio la pagina
            tunel = datos_tunel()

            if es_visita_remota() and tunel:
                fuente = f"{tunel['video']}/video.mjpg?k={tunel['clave']}"
            else:
                host = st.context.headers.get("Host", "localhost").split(":")[0]
                fuente = f"http://{host}:{sesion.video.puerto}/video.mjpg"

            st.markdown(
                f"<img style='width:100%;border-radius:10px' src='{fuente}'>",
                unsafe_allow_html=True
            )
        else:
            st.error("No se pudo iniciar la transmisión de video (puerto ocupado).")

        contador_alarma()

    with c2, st.container(border=True):
        st.markdown("##### :material/format_list_bulleted: Detecciones actuales")
        detecciones_actuales()

    with c3, st.container(border=True):
        st.markdown("##### :material/visibility: Detalle del evento")
        detalle_del_evento()

    with st.container(border=True):
        lista_de_capturas()


# ============================================================
# MODULO PREDICCION
# ============================================================

EMOCIONES = [
    ("neutral", "Neutral / Concentración"),
    ("felicidad", "Felicidad"),
    ("tristeza", "Tristeza / Desatención"),
    ("enojo", "Enojo / Frustración"),
    ("miedo", "Miedo / Ansiedad"),
    ("sorpresa", "Sorpresa / Sobresalto"),
    ("disgusto", "Disgusto")
]


def ir_a_prediccion(indice):

    st.session_state["pred_actual"] = indice
    st.session_state["pred_seguir_ultima"] = False
    st.session_state["pred_estudiante"] = None


def imagen_con_estudiantes(prediccion, seleccionado):
    """
    Imagen mejorada por IMAGEN con el recuadro de cada alumno y el
    objeto no permitido que se le asigno.
    """

    imagen = cv2.imread(prediccion["imagen_mejorada"])

    if imagen is None:
        return None

    grosor = max(2, max(imagen.shape[:2]) // 400)
    escala = max(0.5, max(imagen.shape[:2]) / 1600)

    def caja(c, color, texto):
        cv2.rectangle(imagen, (c["x1"], c["y1"]), (c["x2"], c["y2"]), color, grosor)
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
        y = max(th + 10, c["y1"])
        cv2.rectangle(imagen, (c["x1"], y - th - 10), (c["x1"] + tw + 8, y), color, cv2.FILLED)
        cv2.putText(imagen, texto, (c["x1"] + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    escala, (20, 20, 20), max(1, grosor - 1), cv2.LINE_AA)

    for e in prediccion["estudiantes"]:

        color = (191, 212, 45) if e["id"] == seleccionado else (235, 160, 60)
        caja(e["caja"], color, f"{e['nombre']} ({e['subId']})")

        if e.get("objeto_caja"):
            o = e["objeto_caja"]
            caja(o, (60, 60, 230), f"{nombre_objeto(o['clase'])} {o['confianza']:.2f}")

    return imagen


@st.fragment(run_every=2.0)
def vigilar_predicciones():
    """
    Completa en segundo plano las predicciones pendientes y salta a la
    mas reciente cuando llega una nueva.
    """

    vigilante.revisar(maximo=1, bloquear=False)
    integracion_prediccion.completar_pendientes()

    lista = integracion_prediccion.listar_predicciones()
    ultima = lista[-1]["archivo"] if lista else None

    if ultima != st.session_state.get("pred_ultima"):
        nueva = st.session_state.get("pred_ultima") is not None
        st.session_state["pred_ultima"] = ultima
        if lista and st.session_state.get("pred_seguir_ultima", True):
            st.session_state["pred_actual"] = len(lista) - 1
            st.session_state["pred_estudiante"] = None
        if nueva:
            st.toast("🧠 Nueva predicción disponible")
        st.rerun()


def lista_con_probabilidad(elementos):

    for el in elementos:
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;gap:.6rem;"
            f"padding:.35rem .6rem;margin-bottom:.3rem;border-radius:8px;background:#1E293B'>"
            f"<span>• {el['texto']}</span><b style='white-space:nowrap'>{el['prob']}% prob.</b></div>",
            unsafe_allow_html=True
        )


def mostrar_modulo_prediccion():

    st.markdown("# :material/psychology: Panel de Inferencia de Predicciones")
    st.caption(
        "Resultados del módulo PREDICCIÓN sobre las capturas que procesa IMAGEN: estado y "
        "emociones de cada alumno, sucesos probables y proyección académica."
    )

    vigilar_predicciones()

    lista = integracion_prediccion.listar_predicciones()

    if not lista:
        st.info("Aún no hay predicciones. Aparecerán cuando IMAGEN procese una alerta de VIDEO.")
        return

    total = len(lista)
    indice = min(st.session_state.get("pred_actual", total - 1), total - 1)
    prediccion = lista[indice]
    estudiantes = prediccion["estudiantes"]
    alerta = prediccion.get("alerta_video") or {}

    # Alumno seleccionado: por defecto, el de mayor riesgo
    elegido = st.session_state.get("pred_estudiante")
    estudiante = next((e for e in estudiantes if e["id"] == elegido), None)

    if estudiante is None and estudiantes:
        estudiante = max(estudiantes, key=lambda e: e["score"])

    # --------------------------------------------------------
    # Fila 1: imagen capturada y alumnos
    # --------------------------------------------------------

    c1, c2 = st.columns([1.35, 1.65])

    with c1, st.container(border=True):

        st.markdown("##### :material/photo_camera: Imagen capturada")

        imagen = imagen_con_estudiantes(prediccion, estudiante["id"] if estudiante else None)

        if imagen is not None:
            st.image(a_rgb(imagen), width="stretch")

        n1, n2, n3 = st.columns([1, 2.2, 1], vertical_alignment="center")

        n1.button("◀", width="stretch", disabled=indice == 0, key="pred_ant",
                  on_click=ir_a_prediccion, args=(indice - 1,))
        n2.markdown(
            f"<div class='contador'>Captura <b>{indice + 1}</b> de <b>{total}</b><br>"
            f"<small>{nombre_objeto(alerta.get('objeto', 'desconocido'))} · "
            f"{prediccion.get('timestamp') or ''}</small></div>",
            unsafe_allow_html=True
        )
        n3.button("▶", width="stretch", disabled=indice == total - 1, key="pred_sig",
                  on_click=ir_a_prediccion, args=(indice + 1,))

    with c2, st.container(border=True):

        st.markdown("##### :material/groups: Selección y estado de alumnos en aula")

        if not estudiantes:
            st.info(prediccion["captura"].get(
                "mensaje", "No se detectaron estudiantes en la captura."
            ))

        columnas = st.columns(2)

        for posicion, e in enumerate(estudiantes):

            with columnas[posicion % 2], st.container(border=True):

                riesgo = "🔴 " if e.get("esRiesgoAlto") else ""

                st.markdown(
                    f"**{riesgo}{e['nombre']} (ID: {e['subId']})**  \n"
                    f"{e['estado']}  \n"
                    f"**{e['atencion']}% Atención**"
                )

                st.button(
                    "Seleccionado" if estudiante and e["id"] == estudiante["id"] else "Ver análisis",
                    key=f"est_{indice}_{e['id']}",
                    width="stretch",
                    type="primary" if estudiante and e["id"] == estudiante["id"] else "secondary",
                    on_click=lambda i=e["id"]: st.session_state.update(pred_estudiante=i)
                )

    if not estudiante:
        return

    # --------------------------------------------------------
    # Fila 2: emociones, sucesos y proyeccion del alumno
    # --------------------------------------------------------

    emociones = estudiante["emociones"]
    motor = prediccion.get("motor") or {}

    d1, d2, d3 = st.columns(3)

    with d1, st.container(border=True):

        st.markdown(f"##### :material/mood: Estado y emociones · {estudiante['nombre']}")

        for clave, etiqueta in sorted(EMOCIONES, key=lambda x: -float(emociones.get(x[0], 0))):
            valor = float(emociones.get(clave, 0))
            st.progress(min(valor / 100, 1.0), text=f"{etiqueta} · {valor:.0f}%")

        st.markdown(pastilla(f"Dominante: {emociones['dominante']}", "alerta"),
                    unsafe_allow_html=True)

        st.caption(f"Acción detectada: {estudiante.get('accionDetectada', '-')}")

    with d2, st.container(border=True):

        st.markdown(f"##### :material/format_list_bulleted: Sucesos probables · {estudiante['nombre']}")

        st.markdown("**Acontecimientos previos (causas probables)**")
        lista_con_probabilidad(estudiante.get("causas", []))

        st.markdown("**Acontecimientos futuros (consecuencias estimadas)**")
        lista_con_probabilidad(estudiante.get("consecuencias", []))

    with d3, st.container(border=True):

        st.markdown("##### :material/trending_up: Análisis y proyección académica")

        st.metric(estudiante.get("nivelRiesgo", "Riesgo de desempeño"), f"{estudiante['score']} / 100")

        if estudiante.get("alertaExamen"):
            st.error(estudiante["alertaExamen"], icon=":material/warning:")

        if estudiante.get("narrativa"):
            st.info(estudiante["narrativa"], icon=":material/smart_toy:")

        if estudiante.get("objetos_no_permitidos"):
            st.caption(
                "Objeto detectado por VIDEO y verificado por IMAGEN: "
                + ", ".join(nombre_objeto(o) for o in estudiante["objetos_no_permitidos"])
            )

        if motor:
            st.caption(
                f"Motor de reglas de PREDICCIÓN: revisión {motor.get('nivel_revision', '-')} "
                f"(índice {motor.get('indice_revision', 0):.0f}/100)"
                + (" · requiere revisión humana" if motor.get("requiere_revision_humana") else "")
            )

    st.caption(
        ":material/info: Resultados del módulo PREDICCIÓN (AulaPredict). Son estimaciones a "
        "partir de una imagen: no constituyen diagnóstico, nota real ni prueba de fraude, y "
        "requieren revisión humana."
    )


if modulo == "VIDEO":
    mostrar_modulo_video()
    st.stop()

if modulo == "PREDICCION":
    mostrar_modulo_prediccion()
    st.stop()


# ============================================================
# ENCABEZADO
# ============================================================

st.markdown("""
<div class="encabezado">
  <h1>Modulo IMAGEN</h1>
  <p>VIDEO detecta. IMAGEN mejora, analiza y caracteriza lo detectado,
     y mide si su procesamiento mejora los resultados de VIDEO.</p>
  <div class="flujo">
    <span>1 · Recepcion</span><span>2 · Preprocesamiento adaptativo</span>
    <span>3 · Mejoramiento</span><span>4 · Segmentacion</span>
    <span>5 · Caracteristicas</span><span>6 · Verificacion</span>
    <span>7 · Exportacion</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# PANEL DE ALERTAS DE VIDEO
# ============================================================

def ir_a(indice):

    st.session_state["alerta_actual"] = indice
    st.session_state["seguir_ultima"] = False


@st.fragment(run_every=INTERVALO_REVISION_SEG)
def panel_alertas():

    # Procesa como maximo una alerta nueva por ciclo para no bloquear
    pendientes = vigilante.pendientes()

    if pendientes:
        with st.spinner(f"Procesando alerta nueva: {pendientes[0]}"):
            vigilante.revisar(maximo=1, bloquear=False)

    lista = list(reversed(vigilante.procesadas()))   # mas antigua primero
    total = len(lista)

    ultima = lista[-1]["archivo"] if lista else None

    # Si llega una alerta nueva y se esta siguiendo la ultima, saltar a ella
    if ultima != st.session_state.get("ultima_vista"):
        st.session_state["ultima_vista"] = ultima
        if st.session_state.get("seguir_ultima", True) and total:
            st.session_state["alerta_actual"] = total - 1
            st.rerun()

    # Estado
    c1, c2, c3, c4 = st.columns(4)

    c1.markdown("<div class='vivo'>Vigilando carpeta de alertas de VIDEO</div>",
                unsafe_allow_html=True)
    c1.caption(os.path.relpath(CARPETA_ALERTAS_VIDEO))
    c2.metric("Alertas procesadas", total)
    c3.metric("En cola", len(vigilante.pendientes()))

    if total:
        variaciones = [
            r["resumen"]["comparacion_video"]["confianza_media_despues"]
            - r["resumen"]["comparacion_video"]["confianza_media_antes"]
            for r in lista
        ]
        c4.metric("Variacion media de confianza", f"{sum(variaciones) / total:+.2f}")

    if not total:
        st.markdown(
            "<div class='vacio'>Esperando capturas de la alarma de VIDEO…<br>"
            "<small>Cada captura que VIDEO guarde en su carpeta de alertas se procesara "
            "automaticamente.</small></div>",
            unsafe_allow_html=True
        )
        return

    indice = min(st.session_state.get("alerta_actual", total - 1), total - 1)
    actual = lista[indice]

    # Navegacion
    st.write("")
    n1, n2, n3 = st.columns([1, 3, 1])

    n1.button("◀ Anterior", width="stretch", disabled=indice == 0,
              on_click=ir_a, args=(indice - 1,))
    n2.markdown(
        f"<div class='contador'>Alerta <b>{indice + 1}</b> de <b>{total}</b> · "
        f"{actual['alerta_video']['objeto']} · {actual['alerta_video']['fecha'] or ''}</div>",
        unsafe_allow_html=True
    )
    n3.button("Siguiente ▶", width="stretch", disabled=indice == total - 1,
              on_click=ir_a, args=(indice + 1,))

    # Antes / despues
    comparacion = actual["resumen"]["comparacion_video"]

    i1, i2 = st.columns(2)

    with i1:
        st.image(actual["imagenes"]["original"], width="stretch",
                 caption="Captura de VIDEO con sus detecciones")
    with i2:
        st.image(actual["imagenes"]["procesada"], width="stretch",
                 caption="Procesada por IMAGEN · lo que VIDEO detecta en ella")

    st.markdown(
        pastilla(f"Alarma: {actual['alerta_video']['objeto']}", "alerta")
        + pastilla(
            f"Confianza {comparacion['confianza_media_antes']:.2f} → "
            f"{comparacion['confianza_media_despues']:.2f}",
            "ok" if comparacion["confianza_media_despues"] >= comparacion["confianza_media_antes"] else "info"
        )
        + pastilla(f"{len(actual['objetos'])} objetos analizados")
        + pastilla(f"Procesada en {actual['segundos_proceso']} s"),
        unsafe_allow_html=True
    )

    # Miniaturas
    st.write("")
    st.caption("Todas las alertas")

    por_fila = 8

    for inicio in range(0, total, por_fila):

        columnas = st.columns(por_fila)

        for desplazamiento, columna in enumerate(columnas):

            j = inicio + desplazamiento

            if j >= total:
                break

            with columna:
                st.image(lista[j]["imagenes"]["procesada"], width="stretch")
                st.button(
                    ("● " if j == indice else "") + f"{j + 1}",
                    key=f"mini_{j}",
                    width="stretch",
                    type="primary" if j == indice else "secondary",
                    on_click=ir_a,
                    args=(j,)
                )

    # El analisis detallado esta fuera del fragmento: si cambia la
    # alerta seleccionada, se recarga toda la pagina
    if st.session_state.get("archivo_actual") != actual["archivo"]:
        st.session_state["archivo_actual"] = actual["archivo"]
        st.rerun()


# ============================================================
# EVALUACION COMPARATIVA (comun a ambos modos)
# ============================================================

def seccion_evaluacion(paquetes, ruta_detalle, descripcion):

    st.markdown("### Evaluacion comparativa")
    st.caption(
        f"{descripcion} Cada imagen se degrada en 5 escenarios (normal, poca luz, ruido, "
        "desenfoque, luz+ruido) y se compara la salida del modelo de VIDEO antes y "
        "despues del procesamiento de IMAGEN."
    )

    if len(paquetes) < 2:
        st.info(
            f"Hay {len(paquetes)} imagen(es). La evaluacion funciona, pero los promedios "
            "son mas representativos con 2 o mas."
        )

    if paquetes and st.button(f"Ejecutar evaluacion ({len(paquetes)} imagenes)", type="primary"):

        barra = st.progress(0.0)

        tabla = evaluar(
            paquetes,
            opciones={"segmentacion": segmentacion},
            progreso=lambda i, n, t: barra.progress(i / n, text=t)
        )

        os.makedirs(os.path.dirname(ruta_detalle), exist_ok=True)
        tabla.to_csv(ruta_detalle, index=False, encoding="utf-8-sig")
        barra.empty()

    if not os.path.exists(ruta_detalle):
        st.info("Aun no hay resultados. Pulsa el boton para ejecutar la evaluacion.")
        return

    tabla = pd.read_csv(ruta_detalle)
    resumen = resumen_por_escenario(tabla)

    st.caption(f"Ultima evaluacion: {tabla['origen'].nunique()} imagenes.")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Confianza media de VIDEO en personas**")
        st.bar_chart(
            resumen[["antes_conf_personas", "despues_conf_personas"]]
            .rename(columns={"antes_conf_personas": "antes", "despues_conf_personas": "despues"}),
            stack=False, color=["#64748B", "#2DD4BF"]
        )

    with c2:
        st.markdown("**SSIM respecto a la imagen original**")
        st.bar_chart(
            resumen[["ssim_antes", "ssim_despues"]].dropna()
            .rename(columns={"ssim_antes": "antes", "ssim_despues": "despues"}),
            stack=False, color=["#64748B", "#2DD4BF"]
        )

    st.markdown("**Resumen por escenario**")
    st.dataframe(resumen.T)

    with st.expander("Detalle por imagen"):
        st.dataframe(tabla, hide_index=True)


# ============================================================
# PRUEBA DE ROBUSTEZ (experimento)
# Se empeora a proposito la captura y se mide si el procesamiento
# de IMAGEN la recupera. El modelo de VIDEO se usa sin cambios,
# solo para medir que detecta en cada imagen.
# ============================================================

def seccion_robustez(paquete, clave):

    st.markdown(f"### Prueba de robustez · `{paquete['origen']}`")
    st.caption(
        "Experimento: ¿que pasaria si la camara de VIDEO hubiera capturado una imagen "
        "oscura, con ruido o borrosa? Se degrada la captura a proposito y se compara "
        "con la imagen procesada por IMAGEN. La deteccion de VIDEO no se modifica: su "
        "modelo solo se usa para medir que reconoce en cada imagen."
    )

    s1, s2, s3 = st.columns(3)

    oscuridad = s1.slider("Poca luz", 0.0, 0.9, 0.0, 0.05, key=f"luz_{clave}")
    ruido_simulado = s2.slider("Ruido de sensor", 0, 60, 0, 5, key=f"ruido_{clave}")
    desenfoque = s3.slider("Desenfoque", 0, 20, 0, 1, key=f"desenfoque_{clave}")

    if not (oscuridad or ruido_simulado or desenfoque):
        st.info("Mueve uno de los controles para degradar la captura.")
        return

    opciones_prueba = tuple(sorted({
        **OPCIONES_POR_DEFECTO,
        **opciones,
        "oscuridad": oscuridad,
        "ruido_simulado": ruido_simulado,
        "desenfoque": desenfoque
    }.items()))

    resultado = procesar(paquete, opciones_prueba)

    imagenes = resultado["imagenes"]
    objetos = resultado["objetos"]

    c1, c2, c3 = st.columns(3)

    with c1:
        mostrar(dibujar_video(imagenes["original"], resultado["registro_video"]["detecciones"]),
                "1 · Captura original de VIDEO")
    with c2:
        mostrar(dibujar_video(imagenes["capturado"], resultado["detecciones_capturado"]),
                "2 · Degradada (lo que VIDEO veria)")
    with c3:
        mostrar(dibujar_video(imagenes["mejorado"], resultado["detecciones_despues"]),
                "3 · Procesada por IMAGEN (lo que VIDEO detecta)")

    if resultado["decisiones"]:
        st.markdown("**Decisiones del preprocesamiento adaptativo**")
        for decision in resultado["decisiones"]:
            st.markdown(f"- {decision}")

    d1, d2 = st.columns(2)

    with d1:
        st.markdown("**Calidad de imagen**")
        tabla = tabla_calidad(resultado["calidad"]).rename(
            columns={"antes": "degradada", "despues": "procesada"}
        )
        st.dataframe(tabla)
        st.caption("PSNR y SSIM comparan con la captura original: mas alto = mas parecida.")

    with d2:
        st.markdown("**Que reconoce el modelo de VIDEO** (sin cambios)")
        st.dataframe(pd.DataFrame([
            {
                "clase": o["deteccion"]["clase"],
                "original": o["deteccion"]["confianza_video"],
                "degradada": o["deteccion"]["confianza_capturada"],
                "procesada": o["deteccion"]["confianza_procesada"],
                "estado": o["deteccion"]["estado"]
            }
            for o in objetos
        ]), hide_index=True)

    # Detecciones de la imagen degradada que no corresponden a ningun
    # objeto original (confusiones del modelo)
    emparejadas = set(emparejar(
        resultado["registro_video"]["detecciones"], resultado["detecciones_capturado"]
    ).values())

    otras = [
        d for j, d in enumerate(resultado["detecciones_capturado"])
        if j not in emparejadas
    ]

    if otras:
        st.caption(
            "En la imagen degradada VIDEO tambien reporta: "
            + ", ".join(f"{d['clase']} ({d['confianza']:.2f})" for d in otras)
            + ". Son confusiones causadas por la degradacion."
        )


vista = st.segmented_control(
    "Vista", ["Analisis", "Prueba de robustez", "Evaluacion comparativa"],
    default="Analisis", label_visibility="collapsed"
) or "Analisis"


# ============================================================
# CONTENIDO: ALERTAS DE VIDEO
# ============================================================

if vista == "Evaluacion comparativa":

    paquetes = []

    for nombre in vigilante.alertas_en_carpeta():
        ruta = os.path.join(CARPETA_ALERTAS_VIDEO, nombre)
        paquete = cargar_alerta(nombre, os.path.getmtime(ruta))
        if paquete:
            paquetes.append(paquete)

    seccion_evaluacion(
        paquetes,
        os.path.join(CARPETA_EN_VIVO, "evaluacion_alertas_detalle.csv"),
        "Sobre las capturas reales de la alarma de VIDEO."
    )

elif vista == "Prueba de robustez":

    archivos = vigilante.alertas_en_carpeta()

    if not archivos:
        st.info("Aun no hay alertas de VIDEO.")
    else:
        actual = st.session_state.get("archivo_actual")
        archivo = st.selectbox(
            "Alerta", archivos,
            index=archivos.index(actual) if actual in archivos else len(archivos) - 1
        )
        ruta = os.path.join(CARPETA_ALERTAS_VIDEO, archivo)
        paquete = cargar_alerta(archivo, os.path.getmtime(ruta))

        if paquete:
            seccion_robustez(paquete, archivo)

else:

    panel_alertas()

    archivo = st.session_state.get("archivo_actual")
    ruta = os.path.join(CARPETA_ALERTAS_VIDEO, archivo) if archivo else None

    if ruta and os.path.exists(ruta):

        st.divider()
        st.markdown(f"### Analisis de IMAGEN · `{archivo}`")

        paquete = cargar_alerta(archivo, os.path.getmtime(ruta))

        if paquete:
            mostrar_analisis(procesar(paquete, opciones_clave), archivo)

        # ------------------------------------------------
        # Resumen de todas las alertas (salida para otros grupos)
        # ------------------------------------------------

        st.divider()
        st.markdown("### Resumen de todas las alertas")
        st.caption(
            "Salida acumulada de IMAGEN: un registro por alerta procesada. "
            "Es lo que otros modulos (por ejemplo, PREDICCION) pueden leer."
        )

        todas = list(reversed(vigilante.procesadas()))

        st.dataframe(pd.DataFrame([
            {
                "#": n + 1,
                "archivo": r["archivo"],
                "alarma": (r.get("alerta_video") or {}).get("objeto"),
                "fecha": (r.get("alerta_video") or {}).get("fecha"),
                "objetos": len(r["objetos"]),
                "confianza antes": r["resumen"]["comparacion_video"]["confianza_media_antes"],
                "confianza despues": r["resumen"]["comparacion_video"]["confianza_media_despues"],
                "brillo antes": r["calidad"]["capturado"]["brillo"],
                "brillo despues": r["calidad"]["mejorado"]["brillo"],
                "ruido antes": r["calidad"]["capturado"]["ruido"],
                "ruido despues": r["calidad"]["mejorado"]["ruido"]
            }
            for n, r in enumerate(todas)
        ]), hide_index=True)

