# Segmentación Inteligente de Clientes en Retail Online

Reto de bootcamp: segmentación de clientes con el modelo RFM (Recency, Frequency, Monetary) y K-Means, más un Producto Mínimo Viable (dashboard) construido con Streamlit.

## Contenido

- `notebook/` — análisis completo (EDA, RFM, K-Means, árbol de decisión).
- `app.py` — dashboard interactivo (Producto Mínimo Viable).
- `requirements.txt` — dependencias del proyecto.

## Ejecutar el dashboard en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Datos

El dashboard carga automáticamente el dataset *Online Retail* directamente desde el repositorio de la UCI, así que no necesitas subir ningún archivo de datos.

## Despliegue

Este proyecto está pensado para desplegarse en [Streamlit Community Cloud](https://share.streamlit.io), apuntando al archivo `app.py` de este repositorio.
