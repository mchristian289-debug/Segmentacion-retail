import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(
    page_title="Segmentación de Clientes - Retail Online",
    layout="wide"
)

# ------------------------------------------------------------------
# Caja de explicación con color personalizado
# ------------------------------------------------------------------

def explicacion(texto, color="#1f77b4"):
    st.markdown(
        f"""
        <div style="
            background-color:{color}15;
            border-left: 5px solid {color};
            padding: 12px 16px;
            border-radius: 6px;
            margin-bottom: 18px;
        ">
        {texto}
        </div>
        """,
        unsafe_allow_html=True
    )

COLOR_EDA = "#1f77b4"        # azul
COLOR_PREP = "#6c757d"       # gris
COLOR_RFM = "#6f42c1"        # morado
COLOR_KMEANS = "#198754"     # verde
COLOR_SEGMENTOS = "#fd7e14"  # naranja
COLOR_ARBOL = "#b02a37"      # rojo vino
COLOR_CONCLUSION = "#20c997" # teal

# ------------------------------------------------------------------
# Carga y procesamiento de datos (con cache)
# ------------------------------------------------------------------

@st.cache_data
def cargar_datos():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    return pd.read_excel(url)


@st.cache_data
def preparar_datos(datos):
    datos_limpios = datos[
        (~datos["InvoiceNo"].astype(str).str.startswith("C")) &
        (datos["Quantity"] > 0) &
        (datos["UnitPrice"] > 0) &
        (datos["CustomerID"].notnull())
    ].copy()

    datos_limpios["ValorTotal"] = datos_limpios["Quantity"] * datos_limpios["UnitPrice"]
    datos_limpios["InvoiceDate"] = pd.to_datetime(datos_limpios["InvoiceDate"])

    return datos_limpios


@st.cache_data
def calcular_rfm(datos_limpios):
    fecha_referencia = datos_limpios["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = datos_limpios.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (fecha_referencia - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("ValorTotal", "sum")
    ).reset_index()

    return rfm


@st.cache_data
def calcular_codo(rfm_escalado):
    inercias = []
    for k in range(1, 11):
        modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
        modelo.fit(rfm_escalado)
        inercias.append(modelo.inertia_)
    return inercias


@st.cache_data
def segmentar(rfm, k):
    scaler = StandardScaler()
    rfm_escalado = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
    rfm = rfm.copy()
    rfm["Segmento"] = modelo.fit_predict(rfm_escalado)

    return rfm, rfm_escalado


@st.cache_data
def entrenar_arbol(rfm_segmentado, profundidad):
    X = rfm_segmentado[["Recency", "Frequency", "Monetary"]]
    y = rfm_segmentado["Segmento"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    arbol = DecisionTreeClassifier(max_depth=profundidad, random_state=42)
    arbol.fit(X_train, y_train)
    y_pred = arbol.predict(X_test)

    reporte = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    exactitud = accuracy_score(y_test, y_pred)

    return arbol, exactitud, reporte


# ------------------------------------------------------------------
# Carga inicial
# ------------------------------------------------------------------

st.title("Segmentación Inteligente de Clientes en Retail Online")
st.caption("Producto Mínimo Viable — EDA, modelo RFM, K-Means y árbol de decisión explicativo")

with st.spinner("Cargando y procesando datos (puede tardar un poco la primera vez)..."):
    datos = cargar_datos()
    datos_limpios = preparar_datos(datos)
    rfm = calcular_rfm(datos_limpios)

st.sidebar.header("Configuración")
k = st.sidebar.slider("Número de segmentos (k)", min_value=2, max_value=8, value=4)
profundidad_arbol = st.sidebar.slider("Profundidad del árbol de decisión", min_value=2, max_value=5, value=3)

rfm_segmentado, rfm_escalado = segmentar(rfm, k)

tab_eda, tab_rfm, tab_seg, tab_arbol, tab_conclusion = st.tabs(
    ["📊 EDA", "🧮 RFM", "🎯 Segmentación", "🌳 Árbol de decisión", "✅ Conclusiones"]
)

# ------------------------------------------------------------------
# TAB EDA
# ------------------------------------------------------------------

with tab_eda:
    st.header("Análisis Exploratorio de Datos")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Filas totales", f"{datos.shape[0]:,}")
    col2.metric("Facturas únicas", f"{datos['InvoiceNo'].nunique():,}")
    col3.metric("Clientes únicos", f"{datos['CustomerID'].nunique():,}")
    col4.metric("Países", datos["Country"].nunique())

    explicacion(
        "El dataset original tiene más de 540.000 filas, una por cada producto dentro de una factura "
        "(no una fila por factura completa). Cubre cerca de un año de transacciones, sobre todo del Reino Unido.",
        COLOR_EDA
    )

    st.subheader("Valores faltantes")
    nulos = datos.isnull().sum()
    nulos = nulos[nulos > 0]
    st.bar_chart(nulos)

    explicacion(
        "CustomerID suele tener alrededor de 135.000 valores nulos (cerca del 25% del dataset): son ventas "
        "registradas sin identificar al cliente. Como este reto se centra en el comportamiento por cliente, "
        "esas filas se descartan más adelante.",
        COLOR_EDA
    )

    st.subheader("Distribución de Quantity y UnitPrice (rango acotado para poder verlas)")
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4))
    datos[datos["Quantity"].between(0, 50)]["Quantity"].hist(bins=30, ax=ejes[0], color=COLOR_EDA)
    ejes[0].set_title("Quantity (0 a 50)")
    datos[datos["UnitPrice"].between(0, 20)]["UnitPrice"].hist(bins=30, ax=ejes[1], color=COLOR_EDA)
    ejes[1].set_title("UnitPrice (0 a 20)")
    st.pyplot(fig)

    explicacion(
        "Ambas distribuciones están muy sesgadas hacia la izquierda: la mayoría de las compras son de pocas "
        "unidades y precios bajos, con una cola larga de compras grandes o productos caros. Esto es típico "
        "en datos de retail y confirma por qué conviene filtrar outliers antes de calcular RFM.",
        COLOR_EDA
    )

    st.subheader("Cancelaciones y valores inválidos")
    cancelaciones = datos["InvoiceNo"].astype(str).str.startswith("C").sum()
    negativos_qty = (datos["Quantity"] <= 0).sum()
    negativos_precio = (datos["UnitPrice"] <= 0).sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Filas de cancelaciones", f"{cancelaciones:,}")
    col2.metric("Quantity ≤ 0", f"{negativos_qty:,}")
    col3.metric("UnitPrice ≤ 0", f"{negativos_precio:,}")

    explicacion(
        "Estas filas no representan compras reales (son devoluciones o ajustes contables), así que se "
        "excluyen en la fase de preparación de datos.",
        COLOR_EDA
    )

# ------------------------------------------------------------------
# TAB RFM
# ------------------------------------------------------------------

with tab_rfm:
    st.header("Preparación de datos e ingeniería de variables RFM")

    col1, col2 = st.columns(2)
    col1.metric("Filas antes de filtrar", f"{datos.shape[0]:,}")
    col2.metric("Filas después de filtrar", f"{datos_limpios.shape[0]:,}")

    explicacion(
        "Se filtran cancelaciones, cantidades o precios inválidos, y clientes sin identificar. Queda "
        "alrededor del 72% del dataset original, con transacciones reales de clientes identificados.",
        COLOR_PREP
    )

    st.subheader("Tabla RFM por cliente")
    st.dataframe(rfm.head(20), use_container_width=True)

    explicacion(
        "Cada fila resume el comportamiento de un cliente: Recency (días desde su última compra), "
        "Frequency (número de facturas distintas) y Monetary (gasto total). Se pasa de cientos de miles "
        "de transacciones a una sola fila por cliente.",
        COLOR_RFM
    )

    st.subheader("Distribución de las variables RFM")
    fig, ejes = plt.subplots(1, 3, figsize=(15, 4))
    rfm["Recency"].hist(bins=30, ax=ejes[0], color=COLOR_RFM)
    ejes[0].set_title("Recency (días)")
    rfm["Frequency"].hist(bins=30, ax=ejes[1], color=COLOR_RFM)
    ejes[1].set_title("Frequency (facturas)")
    rfm["Monetary"].hist(bins=30, ax=ejes[2], color=COLOR_RFM)
    ejes[2].set_title("Monetary (gasto total)")
    st.pyplot(fig)

    explicacion(
        "Frequency y Monetary suelen salir muy sesgadas: la mayoría de los clientes compran pocas veces y "
        "gastan montos moderados, con un grupo pequeño de compradores mayoristas muy por encima del resto. "
        "Por eso las variables se normalizan antes de aplicar K-Means — de lo contrario Monetary dominaría "
        "el cálculo de distancias solo por tener números más grandes.",
        COLOR_RFM
    )

# ------------------------------------------------------------------
# TAB SEGMENTACIÓN
# ------------------------------------------------------------------

with tab_seg:
    st.header("Segmentación con K-Means")

    st.subheader("Método del codo")
    inercias = calcular_codo(rfm_escalado)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, 11), inercias, marker="o", color=COLOR_KMEANS)
    ax.set_xlabel("Número de clusters (k)")
    ax.set_ylabel("Inercia")
    st.pyplot(fig)

    explicacion(
        "La inercia siempre baja al aumentar k, pero el punto donde la curva deja de bajar de forma "
        "pronunciada (el 'codo') suele estar entre k=4 y k=5 en datasets RFM como este. Usa el slider de "
        "la izquierda para ajustar el número de segmentos según lo que veas aquí.",
        COLOR_KMEANS
    )

    st.subheader(f"Perfil de los {k} segmentos")
    perfil_segmentos = rfm_segmentado.groupby("Segmento").agg(
        Clientes=("CustomerID", "count"),
        Recency_promedio=("Recency", "mean"),
        Frequency_promedio=("Frequency", "mean"),
        Monetary_promedio=("Monetary", "mean"),
        Ingreso_total=("Monetary", "sum")
    ).round(1)
    perfil_segmentos["Porcentaje_ingresos"] = (
        perfil_segmentos["Ingreso_total"] / perfil_segmentos["Ingreso_total"].sum() * 100
    ).round(1)

    st.dataframe(perfil_segmentos, use_container_width=True)

    explicacion(
        "El segmento con Recency baja, Frequency alta y Monetary alto es el de clientes de mayor valor: "
        "suele ser una fracción pequeña de los clientes pero un porcentaje alto de los ingresos (patrón "
        "80/20). El segmento con Recency muy alta son clientes inactivos o en riesgo de abandono.",
        COLOR_SEGMENTOS
    )

    col_izq, col_der = st.columns(2)
    with col_izq:
        st.subheader("Clientes por segmento")
        fig1, ax1 = plt.subplots()
        perfil_segmentos["Clientes"].plot(kind="bar", ax=ax1, color=COLOR_SEGMENTOS)
        ax1.set_xlabel("Segmento")
        ax1.set_ylabel("Clientes")
        st.pyplot(fig1)

    with col_der:
        st.subheader("Ingresos por segmento (%)")
        fig2, ax2 = plt.subplots()
        perfil_segmentos["Porcentaje_ingresos"].plot(kind="bar", ax=ax2, color=COLOR_SEGMENTOS)
        ax2.set_xlabel("Segmento")
        ax2.set_ylabel("% de ingresos totales")
        st.pyplot(fig2)

    explicacion(
        "Es común que el segmento más pequeño en número de clientes sea el que concentra el mayor "
        "porcentaje de ingresos — esa es la evidencia visual de por qué conviene tratar a los clientes "
        "de forma diferenciada en vez de aplicarles a todos la misma estrategia.",
        COLOR_SEGMENTOS
    )

    st.subheader("Clusters: Frequency vs. Monetary")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    dispersion = ax3.scatter(
        rfm_segmentado["Frequency"],
        rfm_segmentado["Monetary"],
        c=rfm_segmentado["Segmento"],
        cmap="viridis",
        alpha=0.6
    )
    ax3.set_xlabel("Frequency (número de facturas)")
    ax3.set_ylabel("Monetary (gasto total)")
    legend1 = ax3.legend(*dispersion.legend_elements(), title="Segmento")
    ax3.add_artist(legend1)
    st.pyplot(fig3)

    st.subheader("Detalle de clientes")
    st.dataframe(
        rfm_segmentado.sort_values("Monetary", ascending=False),
        use_container_width=True
    )

# ------------------------------------------------------------------
# TAB ÁRBOL DE DECISIÓN
# ------------------------------------------------------------------

with tab_arbol:
    st.header("Modelo supervisado explicativo (árbol de decisión)")

    explicacion(
        "El árbol aprende a predecir el segmento de un cliente a partir de sus variables RFM. El objetivo "
        "no es maximizar la exactitud, sino obtener reglas simples e interpretables que expliquen qué hace "
        "que un cliente pertenezca a cada segmento.",
        COLOR_ARBOL
    )

    arbol, exactitud, reporte = entrenar_arbol(rfm_segmentado, profundidad_arbol)

    st.metric("Exactitud del árbol", f"{exactitud:.3f}")

    explicacion(
        "Como el árbol está reconstruyendo los mismos segmentos que ya definió K-Means a partir de estas "
        "tres variables, la exactitud suele ser muy alta (típicamente por encima de 0.90). Eso es normal "
        "aquí: no se está prediciendo algo desconocido, sino explicando con reglas simples una segmentación "
        "que ya existe.",
        COLOR_ARBOL
    )

    st.subheader("Reglas del árbol")
    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(
        arbol,
        feature_names=["Recency", "Frequency", "Monetary"],
        class_names=[f"Segmento {c}" for c in sorted(rfm_segmentado["Segmento"].unique())],
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax
    )
    st.pyplot(fig)

    explicacion(
        "Cada división es una regla legible, del tipo 'si Monetary es mayor a cierto valor, entonces...'. "
        "Las primeras divisiones (más cerca de la raíz) son las variables que más separan a los segmentos "
        "— en datasets RFM casi siempre es Monetary o Frequency la que aparece primero.",
        COLOR_ARBOL
    )

    st.subheader("Importancia de cada variable")
    importancias = pd.Series(arbol.feature_importances_, index=["Recency", "Frequency", "Monetary"])
    fig, ax = plt.subplots(figsize=(6, 3))
    importancias.sort_values().plot(kind="barh", ax=ax, color=COLOR_ARBOL)
    st.pyplot(fig)

# ------------------------------------------------------------------
# TAB CONCLUSIONES
# ------------------------------------------------------------------

with tab_conclusion:
    st.header("Conclusiones")

    explicacion(
        """
        <b>EDA:</b> el dataset tiene valores faltantes en CustomerID, cancelaciones y outliers, típicos de datos transaccionales reales.<br><br>
        <b>Preparación de datos:</b> se filtraron cancelaciones, valores inválidos y clientes sin identificar.<br><br>
        <b>RFM:</b> se transformaron las transacciones en un perfil de comportamiento por cliente.<br><br>
        <b>K-Means:</b> se segmentó a los clientes en grupos con comportamientos similares.<br><br>
        <b>Interpretación:</b> cada segmento se describió en términos de negocio (clientes, gasto promedio, % de ingresos).<br><br>
        <b>Árbol de decisión:</b> se obtuvieron reglas simples que explican cada segmento.
        """,
        COLOR_CONCLUSION
    )
