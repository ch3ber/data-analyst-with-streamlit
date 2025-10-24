# Aplicación de regresión lineal simple con Streamlit.
# El código y comentarios están en inglés, pero la interfaz está en español.

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="Regresión Lineal Simple (Demo)", layout="centered")

st.title("Regresión Lineal Simple (Demo)")
st.write(
    "Esta aplicación ajusta un modelo de regresión lineal simple entre dos variables numéricas. "
    "Sube un archivo CSV, selecciona la variable independiente (X) y la variable dependiente (Y), "
    "entrena el modelo, realiza una predicción y visualiza la línea de regresión."
)

# -----------------------------
# 1) Cargar datos
# -----------------------------
st.header("1. Cargar datos")

st.caption(
    "Sube un archivo CSV que contenga al menos dos columnas numéricas. "
    "Los encabezados de las columnas se usarán como nombres de variables."
)

uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])

sample_df = pd.DataFrame(
    {
        "edad": [20, 18, 30, 12, 30],
        "salario": [20000, 4000, 3000, 0, 60000],
    }
)

# Permitir descarga de CSV de ejemplo
st.download_button(
    label="Descargar CSV de ejemplo",
    data=sample_df.to_csv(index=False).encode("utf-8"),
    file_name="datos_ejemplo.csv",
    mime="text/csv",
    help="Haz clic para descargar un CSV mínimo que puedes modificar localmente.",
)

# Seleccionar fuente de datos
df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"No se pudo leer el archivo CSV subido: {e}")
        df = None
else:
    st.info(
        "Aún no has subido un archivo. Puedes usar el CSV de ejemplo anterior o subir uno propio."
    )

if df is None:
    st.stop()

# Vista previa
st.subheader("Vista previa de los datos")
st.dataframe(df.head())

# -----------------------------
# 2) Seleccionar variables
# -----------------------------
st.header("2. Seleccionar variables")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if len(numeric_cols) < 2:
    st.error("El conjunto de datos debe contener al menos dos columnas numéricas.")
    st.stop()

x_col = st.selectbox("Variable independiente (X)", numeric_cols, index=0)
y_col = st.selectbox(
    "Variable dependiente (Y)", numeric_cols, index=min(1, len(numeric_cols) - 1)
)

work_df = df[[x_col, y_col]].copy()
n_before = len(work_df)
work_df = work_df.dropna()
n_after = len(work_df)
if n_after < n_before:
    st.warning(
        f"Se eliminaron {n_before - n_after} filas debido a valores faltantes en '{x_col}' o '{y_col}'."
    )

X = work_df[[x_col]].values
y = work_df[y_col].values

# -----------------------------
# 3) Entrenar el modelo
# -----------------------------
st.header("3. Entrenar el modelo")
model = LinearRegression()
model.fit(X, y)

slope = float(model.coef_[0])
intercept = float(model.intercept_)
y_pred = model.predict(X)
r2 = float(r2_score(y, y_pred))

st.subheader("Ecuación del modelo")
st.latex(rf"{y_col} = {slope:.4f}\,{x_col} + {intercept:.4f}")

st.subheader("Coeficiente de determinación ($R^2$)")
st.write(f"R² = {r2:.4f}")

# -----------------------------
# 4) Realizar una predicción
# -----------------------------
st.header("4. Realizar una predicción")

default_x = float(np.nanmean(X)) if len(X) else 0.0
x_value = st.number_input(
    f"Introduce un valor para {x_col}:", value=float(np.round(default_x, 2))
)
pred_value = float(model.predict(np.array([[x_value]]))[0])
st.success(f"Predicción para {y_col} cuando {x_col} = {x_value:.2f}: {pred_value:.2f}")

# -----------------------------
# 5) Visualización del modelo
# -----------------------------
st.header("5. Visualización del modelo")

fig = plt.figure(figsize=(6, 4))
plt.scatter(X, y, label="Datos reales")
x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(x_line)
plt.plot(x_line, y_line, label="Línea de regresión", color="red")
plt.scatter(
    [x_value], [pred_value], s=60, marker="o", label="Predicción", color="green"
)

plt.xlabel(x_col)
plt.ylabel(y_col)
plt.legend(loc="best")
plt.tight_layout()

st.pyplot(fig)

# -----------------------------
# 6) Descargar resultados (opcional)
# -----------------------------
st.header("6. Descargar resultados (opcional)")

result_df = work_df.copy()
result_df["prediccion_" + y_col] = y_pred
result_df["residuo"] = work_df[y_col] - y_pred

buffer = io.BytesIO()
result_df.to_csv(buffer, index=False)
st.download_button(
    label="Descargar resultados ajustados (CSV)",
    data=buffer.getvalue(),
    file_name="resultados_ajustados.csv",
    mime="text/csv",
)
