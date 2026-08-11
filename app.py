import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="Segmentación de Clientes - Retail Online",
    layout="wide"
)

# ------------------------------------------------------------------
# Carga y procesamiento de datos (con cache para no recalcular cada vez)
# ------------------------------------------------------------------

@st.cache_data
def cargar_datos():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    datos = pd.read_excel(url)
    return datos


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
def segmentar(rfm, k):
    scaler = StandardScaler()
    rfm_escalado = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
    rfm = rfm.copy()
    rfm["Segmento"] = modelo.fit_predict(rfm_escalado)

    return rfm


# ------------------------------------------------------------------
# Interfaz
# ------------------------------------------------------------------

st.title("Segmentación Inteligente de Clientes en Retail Online")
st.caption("Producto Mínimo Viable — modelo RFM + K-Means")

with st.spinner("Cargando y procesando datos (puede tardar un poco la primera vez)..."):
    datos = cargar_datos()
    datos_limpios = preparar_datos(datos)
    rfm = calcular_rfm(datos_limpios)

st.sidebar.header("Configuración")
k = st.sidebar.slider("Número de segmentos (k)", min_value=2, max_value=8, value=4)

rfm_segmentado = segmentar(rfm, k)

# ------------------------------------------------------------------
# KPIs
# ------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric("Clientes totales", f"{rfm_segmentado.shape[0]:,}")
col2.metric("Número de segmentos", k)
col3.metric("Ingreso total", f"£{rfm_segmentado['Monetary'].sum():,.0f}")
col4.metric("Ingreso promedio por cliente", f"£{rfm_segmentado['Monetary'].mean():,.0f}")

st.divider()

# ------------------------------------------------------------------
# Perfil de segmentos
# ------------------------------------------------------------------

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

col_izq, col_der = st.columns(2)

with col_izq:
    st.subheader("Clientes por segmento")
    fig1, ax1 = plt.subplots()
    perfil_segmentos["Clientes"].plot(kind="bar", ax=ax1, color="steelblue")
    ax1.set_xlabel("Segmento")
    ax1.set_ylabel("Clientes")
    st.pyplot(fig1)

with col_der:
    st.subheader("Ingresos por segmento (%)")
    fig2, ax2 = plt.subplots()
    perfil_segmentos["Porcentaje_ingresos"].plot(kind="bar", ax=ax2, color="darkorange")
    ax2.set_xlabel("Segmento")
    ax2.set_ylabel("% de ingresos totales")
    st.pyplot(fig2)

st.subheader("Representación de los clusters (Frequency vs. Monetary)")
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

st.subheader("Tabla resumen: métricas RFM por segmento")
st.dataframe(perfil_segmentos, use_container_width=True)

st.subheader("Detalle de clientes")
st.dataframe(
    rfm_segmentado.sort_values("Monetary", ascending=False),
    use_container_width=True
)
