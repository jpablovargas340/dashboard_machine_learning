"""
Streamlit dashboard (tabs/ventanas) para monitorear un modelo de riesgo de diabetes
y explicarlo de forma interpretativa (con fórmulas explicadas + lectura de gráficos).

Cómo correr:
  1) Exporta artefactos:
       python train_export.py --input diabetes.xlsx --out_dir .
  2) Ejecuta Streamlit:
       streamlit run streamlit_app.py

Archivos esperados:
  - model.joblib
  - model_metrics.json
  - feature_importance_top10.csv
  - (opcional) diabetes.xlsx   (para EDA, curvas ROC/PR y análisis por umbral)
"""

from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
)

alt.data_transformers.disable_max_rows()


# -----------------------------
# Carga de artefactos (cache)
# -----------------------------
@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


@st.cache_data
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_feature_importance(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "abs_coef" not in df.columns and "coef" in df.columns:
        df["abs_coef"] = df["coef"].abs()
    return df


@st.cache_data
def load_dataset_excel(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    return pd.read_excel(path)


# -----------------------------
# Helpers visuales
# -----------------------------
def kpi(label: str, value: float, help_text: str = "", suffix: str = "") -> None:
    with st.container(border=True):
        st.markdown(f"**{label}**")
        st.markdown(
            f"<div style='font-size:30px; font-weight:800;'>{value:.3f}{suffix}</div>",
            unsafe_allow_html=True,
        )
        if help_text:
            st.caption(help_text)


def heatmap_confusion(cm: np.ndarray) -> alt.Chart:
    df_cm = pd.DataFrame(
        {
            "Real": ["No diabetes", "No diabetes", "Diabetes", "Diabetes"],
            "Predicción": ["No diabetes", "Diabetes", "No diabetes", "Diabetes"],
            "Valor": [cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]],
        }
    )
    base = (
        alt.Chart(df_cm)
        .mark_rect()
        .encode(
            x=alt.X("Predicción:N", title="Predicción del modelo"),
            y=alt.Y("Real:N", title="Etiqueta real"),
            color=alt.Color("Valor:Q", title="Conteo"),
            tooltip=["Real:N", "Predicción:N", "Valor:Q"],
        )
        .properties(height=260)
    )
    text = (
        alt.Chart(df_cm)
        .mark_text(fontSize=18, fontWeight="bold")
        .encode(x="Predicción:N", y="Real:N", text="Valor:Q")
    )
    return base + text


def bar_feature_importance_signed(df_fi: pd.DataFrame, top_n: int = 12) -> alt.Chart:
    df_plot = df_fi.copy()
    if "coef" not in df_plot.columns:
        df_plot["coef"] = df_plot["abs_coef"]

    df_plot["signo"] = np.where(df_plot["coef"] >= 0, "Aumenta riesgo", "Reduce riesgo")
    df_plot = df_plot.sort_values("abs_coef", ascending=False).head(top_n).copy()

    chart = (
        alt.Chart(df_plot)
        .mark_bar()
        .encode(
            x=alt.X("coef:Q", title="Coeficiente (dirección del efecto)"),
            y=alt.Y("feature:N", sort="-x", title="Variable"),
            color=alt.Color("signo:N", title="Interpretación"),
            tooltip=["feature:N", "coef:Q", "abs_coef:Q", "signo:N"],
        )
        .properties(height=380)
    )
    return chart


def dist_target_chart(df: pd.DataFrame) -> alt.Chart:
    dist = df["diabetes"].value_counts().rename_axis("diabetes").reset_index(name="conteo")
    dist["porcentaje"] = (dist["conteo"] / dist["conteo"].sum() * 100).round(2)
    dist["diabetes"] = dist["diabetes"].astype(str)

    return (
        alt.Chart(dist)
        .mark_bar()
        .encode(
            x=alt.X("diabetes:N", title="Diabetes (0 = No, 1 = Sí)"),
            y=alt.Y("conteo:Q", title="Cantidad"),
            tooltip=["diabetes:N", "conteo:Q", "porcentaje:Q"],
        )
        .properties(height=260, title="Distribución de la variable objetivo")
    )


def means_by_class_chart(df: pd.DataFrame, vars_: list[str]) -> alt.Chart:
    medias = df.groupby("diabetes")[vars_].mean().reset_index()
    long = medias.melt("diabetes", var_name="variable", value_name="media")
    long["diabetes"] = long["diabetes"].astype(str)

    return (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("variable:N", title="Variable"),
            y=alt.Y("media:Q", title="Media"),
            column=alt.Column("diabetes:N", title="Clase (diabetes)"),
            tooltip=["variable:N", "media:Q", "diabetes:N"],
        )
        .properties(height=250, title="Comparación de medias por clase")
    )


def histogram_chart(df: pd.DataFrame, col: str, bins: int = 35) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=bins), title=col),
            y=alt.Y("count():Q", title="Frecuencia"),
            tooltip=[alt.Tooltip("count():Q", title="Frecuencia")],
        )
        .properties(height=260, title=f"Histograma de {col}")
    )


def kde_chart(df: pd.DataFrame, col: str) -> alt.Chart:
    return (
        alt.Chart(df)
        .transform_density(col, as_=[col, "densidad"])
        .mark_area(opacity=0.6)
        .encode(
            x=alt.X(f"{col}:Q", title=col),
            y=alt.Y("densidad:Q", title="Densidad"),
            tooltip=[alt.Tooltip(f"{col}:Q", title=col), alt.Tooltip("densidad:Q", title="Densidad")],
        )
        .properties(height=260, title=f"Densidad de {col}")
    )


# -----------------------------
# Evaluación adicional (si hay dataset)
# -----------------------------
@st.cache_data
def compute_test_predictions(
    data_path: str,
    model_path: str,
    test_size: float = 0.20,
    seed: int = 42
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Recalcula un split estratificado sobre el dataset y obtiene probabilidades en test.
    Útil para curvas ROC/PR y para estudiar umbrales.
    """
    if not os.path.exists(data_path) or not os.path.exists(model_path):
        return None

    df = pd.read_excel(data_path)
    if "diabetes" not in df.columns:
        return None

    X = df.drop(columns=["diabetes"])
    y = df["diabetes"].astype(int)

    _, X_te, _, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    model = joblib.load(model_path)
    proba = model.predict_proba(X_te)[:, 1]
    return y_te.values, proba


def threshold_metrics_table(y_true: np.ndarray, y_proba: np.ndarray) -> pd.DataFrame:
    thresholds = np.linspace(0.05, 0.95, 19)
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        rows.append(
            {
                "umbral": float(t),
                "recall": recall_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred),
            }
        )
    return pd.DataFrame(rows)


def threshold_curve_chart(df_thr: pd.DataFrame) -> alt.Chart:
    df_long = df_thr.melt("umbral", var_name="métrica", value_name="valor")
    return (
        alt.Chart(df_long)
        .mark_line(point=True)
        .encode(
            x=alt.X("umbral:Q", title="Umbral (τ)"),
            y=alt.Y("valor:Q", title="Valor"),
            color=alt.Color("métrica:N", title="Métrica"),
            tooltip=["umbral:Q", "métrica:N", alt.Tooltip("valor:Q", format=".3f")],
        )
        .properties(height=280, title="Trade-off por umbral (recall vs carga operativa)")
    )


def roc_pr_charts(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[alt.Chart, alt.Chart]:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    df_roc = pd.DataFrame({"fpr": fpr, "tpr": tpr})

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    df_pr = pd.DataFrame({"recall": recall, "precision": precision})

    roc_chart = (
        alt.Chart(df_roc)
        .mark_line()
        .encode(
            x=alt.X("fpr:Q", title="FPR = FP / (FP + TN)"),
            y=alt.Y("tpr:Q", title="TPR (Recall) = TP / (TP + FN)"),
            tooltip=[alt.Tooltip("fpr:Q", format=".3f"), alt.Tooltip("tpr:Q", format=".3f")],
        )
        .properties(height=260, title=f"Curva ROC (AUC ≈ {roc_auc:.3f})")
    )

    pr_chart = (
        alt.Chart(df_pr)
        .mark_line()
        .encode(
            x=alt.X("recall:Q", title="Recall = TP / (TP + FN)"),
            y=alt.Y("precision:Q", title="Precision = TP / (TP + FP)"),
            tooltip=[alt.Tooltip("recall:Q", format=".3f"), alt.Tooltip("precision:Q", format=".3f")],
        )
        .properties(height=260, title=f"Curva Precision-Recall (AUC ≈ {pr_auc:.3f})")
    )

    return roc_chart, pr_chart


# -----------------------------
# App principal
# -----------------------------
def main() -> None:
    st.set_page_config(page_title="Diabetes Risk – Monitoring", layout="wide")

    st.markdown(
        """
        <div style="display:flex; align-items:baseline; gap:14px;">
          <h1 style="margin-bottom:0;">Diabetes Risk</h1>
          <span style="opacity:.7; font-size:16px;">Tablero interpretativo: desempeño, variables e insights</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.header("Configuración")
    model_path = st.sidebar.text_input("Ruta del modelo", "model.joblib")
    metrics_path = st.sidebar.text_input("Ruta métricas (JSON)", "model_metrics.json")
    fi_path = st.sidebar.text_input("Ruta feature importance (CSV)", "feature_importance_top10.csv")
    data_path = st.sidebar.text_input("Ruta dataset (opcional)", "diabetes.xlsx")
    st.sidebar.divider()

    if not (os.path.exists(model_path) and os.path.exists(metrics_path) and os.path.exists(fi_path)):
        st.warning(
            "Faltan archivos requeridos. Debes tener: model.joblib, model_metrics.json, feature_importance_top10.csv.\n\n"
            "Sugerencia: `python train_export.py --input diabetes.xlsx --out_dir .`"
        )
        st.stop()

    model = load_model(model_path)
    metrics = load_json(metrics_path)
    df_fi = load_feature_importance(fi_path)
    df_data = load_dataset_excel(data_path)

    tab_resumen, tab_desempeno, tab_interpret, tab_eda, tab_pred = st.tabs(
        ["📌 Resumen", "📊 Desempeño", "🧠 Interpretabilidad", "🔎 EDA", "🧾 Predicción"]
    )

    # -----------------------------
    # TAB: RESUMEN
    # -----------------------------
    with tab_resumen:
        c1, c2 = st.columns([1.15, 0.85], gap="large")

        with c1:
            st.subheader("Objetivo del tablero")
            st.write(
                "Este tablero no es solo una “demo”: busca explicar qué hace el modelo, cómo se evalúa "
                "y cómo se debería usar en decisiones clínicas. El foco del negocio es minimizar falsos negativos, "
                "es decir, evitar que un paciente con diabetes quede sin detectar."
            )

            st.markdown("**Métrica principal (negocio): Sensibilidad / Recall**")
            st.latex(r"\text{Recall}=\frac{TP}{TP+FN}")
            st.write(
                "Aquí, **TP** son los casos con diabetes correctamente detectados y **FN** son los casos con diabetes "
                "que el modelo no detectó. Por eso, cuando sube recall, normalmente bajan los falsos negativos."
            )

        with c2:
            st.subheader("KPIs del modelo final (test)")
            m = metrics["model"]
            kpi("Recall (sensibilidad)", float(m["recall"]), "Prioridad clínica: reducir FN")
            kpi("Precisión", float(m["precision"]), "Carga operativa: cuántas alertas son correctas")
            kpi("F1-score", float(m["f1"]), "Balance general")
            kpi("AUC-ROC", float(m["roc_auc"]), "Separación global entre clases")

        st.divider()
        balance = metrics.get("class_balance", {})
        if balance:
            st.caption(
                f"Distribución (dataset): diabetes=1 ≈ {balance['positive_rate']*100:.1f}% "
                f"| diabetes=0 ≈ {balance['negative_rate']*100:.1f}%"
            )
            st.write(
                "Interpretación: como la clase positiva es minoritaria, un modelo podría tener buena accuracy "
                "prediciendo casi todo como 0. Por eso no se toma accuracy como métrica principal."
            )

    # -----------------------------
    # TAB: DESEMPEÑO
    # -----------------------------
    with tab_desempeno:
        st.subheader("Desempeño y lectura operativa")

        m = metrics["model"]
        cm = np.array(m["confusion_matrix"], dtype=int)

        c1, c2 = st.columns([1, 1], gap="large")

        with c1:
            st.markdown("**Matriz de confusión (¿en qué se equivoca el modelo?)**")
            st.altair_chart(heatmap_confusion(cm), use_container_width=True)

            tp = int(cm[1, 1])
            fn = int(cm[1, 0])
            fp = int(cm[0, 1])
            tn = int(cm[0, 0])

            st.write(
                f"Interpretación: **TP={tp}** (diabéticos detectados), **FN={fn}** (diabéticos no detectados), "
                f"**FP={fp}** (personas sin diabetes marcadas como riesgo), **TN={tn}** (no diabéticos correctos)."
            )
            st.write(
                "En salud, lo más crítico es **FN**, porque es gente que sí tiene diabetes y el sistema no la prioriza. "
                "FP generan carga extra, pero suelen ser un costo aceptable si el recall es alto."
            )

        with c2:
            st.markdown("**Curvas y umbrales (cómo ajustar el modelo a la operación)**")
            pred_pack = compute_test_predictions(data_path, model_path)

            if pred_pack is None:
                st.info(
                    "Si agregas diabetes.xlsx en el sidebar, aquí aparecen curvas ROC/PR y el análisis por umbral."
                )
            else:
                y_true, y_proba = pred_pack

                df_thr = threshold_metrics_table(y_true, y_proba)
                st.altair_chart(threshold_curve_chart(df_thr), use_container_width=True)

                st.write(
                    "Interpretación del gráfico por umbral: cuando **bajas τ**, normalmente sube el recall "
                    "(detectas más casos) pero baja la precisión (aparecen más falsos positivos). Cuando **subes τ**, "
                    "pasa lo contrario. La elección de τ debe alinearse con la capacidad real de atención."
                )

                roc_chart, pr_chart = roc_pr_charts(y_true, y_proba)
                st.altair_chart(roc_chart, use_container_width=True)
                st.altair_chart(pr_chart, use_container_width=True)

                st.write(
                    "Lectura rápida: ROC resume la capacidad de separación general, pero en escenarios desbalanceados "
                    "la curva Precision-Recall suele ser más informativa porque se enfoca en el desempeño de la clase positiva."
                )

                with st.expander("Ver tabla de métricas por umbral"):
                    st.dataframe(df_thr, use_container_width=True)

    # -----------------------------
    # TAB: INTERPRETABILIDAD
    # -----------------------------
    with tab_interpret:
        st.subheader("Interpretabilidad del modelo (por qué predice lo que predice)")

        st.write(
            "En regresión logística, el modelo estima una probabilidad a partir de una combinación lineal de variables. "
            "La ecuación se puede expresar como:"
        )
        st.latex(r"\log\left(\frac{p}{1-p}\right)=\beta_0+\beta_1x_1+\cdots+\beta_kx_k")

        st.markdown("**¿Qué significa cada símbolo?**")
        st.write(
            "Aquí, **p** representa la probabilidad estimada de diabetes (clase 1). El término **p/(1-p)** se conoce como "
            "odds o razón de probabilidades. El logaritmo de los odds se modela como una suma: **β0** es el intercepto "
            "(término base) y cada **βi** es un coeficiente que multiplica a su variable **xi**. "
            "Cuando **βi** es positivo, aumentar **xi** tiende a aumentar el riesgo; si es negativo, tiende a disminuirlo. "
            "Como las variables numéricas están escaladas, el tamaño del coeficiente es comparable entre ellas."
        )

        st.divider()

        c1, c2 = st.columns([1, 1], gap="large")

        with c1:
            top_n = st.slider("Top N variables", min_value=8, max_value=25, value=12, step=1)
            st.altair_chart(bar_feature_importance_signed(df_fi, top_n=top_n), use_container_width=True)

            st.write(
                "Interpretación del gráfico: el eje X muestra el coeficiente con su signo. Las barras hacia la derecha "
                "representan variables que aumentan la probabilidad, y hacia la izquierda variables que la reducen. "
                "El orden por importancia se basa en el valor absoluto del coeficiente."
            )

        with c2:
            st.markdown("**Tabla (detalle)**")
            st.dataframe(df_fi.sort_values("abs_coef", ascending=False).head(top_n), use_container_width=True)

            st.markdown("**Cómo traducir esto a acciones**")
            st.write(
                "Si variables clínicas como HbA1c y glucosa aparecen arriba, el modelo está aprendiendo señales relevantes. "
                "Esto ayuda a justificar decisiones: por ejemplo, aumentar la frecuencia de control o priorizar confirmación "
                "para pacientes con valores elevados en estas variables."
            )

    # -----------------------------
    # TAB: EDA
    # -----------------------------
    with tab_eda:
        st.subheader("EDA (evidencia para respaldar insights)")

        if df_data is None:
            st.info("Para ver EDA, agrega diabetes.xlsx en la ruta indicada en el sidebar.")
        else:
            c1, c2 = st.columns([1, 1], gap="large")

            with c1:
                st.altair_chart(dist_target_chart(df_data), use_container_width=True)
                st.write(
                    "Interpretación: se observa un desbalance fuerte. Esto explica por qué el modelo se enfoca en recall "
                    "y por qué se usan estrategias como class_weight."
                )

            with c2:
                vars_key = ["HbA1c_level", "blood_glucose_level", "bmi", "age"]
                st.altair_chart(means_by_class_chart(df_data, vars_key), use_container_width=True)
                st.write(
                    "Interpretación: las medias en glucosa y HbA1c suelen ser más altas en el grupo con diabetes. "
                    "BMI y edad también tienden a ser mayores. Esto respalda el comportamiento del modelo y los insights."
                )

            st.divider()

            st.markdown("### Distribución de variables (elige una y analízala)")
            var_sel = st.selectbox("Variable numérica", ["HbA1c_level", "blood_glucose_level", "bmi", "age"])

            c3, c4 = st.columns([1, 1], gap="large")
            with c3:
                st.altair_chart(histogram_chart(df_data, var_sel), use_container_width=True)
                st.write(
                    "Cómo leer el histograma: permite ver dónde se concentra la población (barras más altas) "
                    "y si existen colas hacia valores extremos. En variables clínicas, las colas altas suelen señalar "
                    "perfiles de riesgo."
                )

            with c4:
                st.altair_chart(kde_chart(df_data, var_sel), use_container_width=True)
                st.write(
                    "Cómo leer la densidad: es una versión suavizada del histograma. Sirve para notar si la distribución "
                    "es simétrica o sesgada. Sesgo hacia la derecha (cola alta) suele indicar presencia de valores elevados."
                )

            st.divider()
            st.markdown("### Conclusión interpretativa del EDA")
            st.write(
                "En general, EDA y modelo cuentan una historia consistente: los indicadores metabólicos (glucosa y HbA1c) "
                "explican gran parte del riesgo, y variables como BMI y edad ayudan a perfilar pacientes que deben priorizarse. "
                "Esta consistencia es importante porque en salud no basta con que el modelo acierte: también debe ser coherente y justificable."
            )

    # -----------------------------
    # TAB: PREDICCIÓN
    # -----------------------------
    with tab_pred:
        st.subheader("Predicción individual (cómo se debería usar en la práctica)")

        st.write(
            "Este módulo muestra una predicción a nivel de paciente. En un entorno real, la predicción debe integrarse "
            "con un flujo clínico: recolección de datos confiables, validación, decisión con umbral definido por operación, "
            "y finalmente confirmación con exámenes o criterio médico."
        )

        st.markdown("**Paso a paso recomendado (flujo real):**")
        st.write(
            "1) Capturar variables con controles de calidad (rangos válidos, unidades y consistencia). "
            "2) Generar la probabilidad con el modelo. "
            "3) Comparar contra un umbral τ definido por la institución (según capacidad de exámenes y riesgo clínico). "
            "4) Si se supera τ, se activa un protocolo de confirmación (prueba, consulta, seguimiento). "
            "5) Registrar resultado final para monitorear desempeño y retrain futuro."
        )

        st.divider()

        st.markdown("**Regla de decisión del sistema**")
        st.latex(r"\hat{y}=\mathbb{1}\left[p(\text{diabetes}=1\mid x)\ge \tau\right]")
        st.write(
            "Interpretación: **p(diabetes=1 | x)** es la probabilidad estimada dada la información del paciente **x** "
            "(sus variables). **τ (tau)** es el umbral de decisión. El símbolo **𝟙[·]** es un indicador: vale 1 si la condición "
            "se cumple y 0 si no. Entonces, si la probabilidad supera τ, el sistema marca ‘riesgo alto’."
        )

        st.divider()

        # Valores sugeridos
        gender_vals = ["Female", "Male", "Other"]
        smoke_vals = ["No Info", "current", "ever", "former", "never", "not current"]

        with st.form("form_pred"):
            c1, c2, c3 = st.columns(3)

            with c1:
                gender = st.selectbox("gender", gender_vals)
                age = st.number_input("age", min_value=0.0, max_value=100.0, value=43.0)
                hypertension = st.selectbox("hypertension", [0, 1], index=0)

            with c2:
                heart_disease = st.selectbox("heart_disease", [0, 1], index=0)
                smoking_history = st.selectbox("smoking_history", smoke_vals)
                bmi = st.number_input("bmi", min_value=5.0, max_value=120.0, value=27.32)

            with c3:
                hba1c = st.number_input("HbA1c_level", min_value=2.0, max_value=15.0, value=5.8)
                glucose = st.number_input("blood_glucose_level", min_value=50.0, max_value=400.0, value=140.0)
                threshold = st.slider("Umbral (τ)", min_value=0.05, max_value=0.95, value=0.50, step=0.01)

            submit = st.form_submit_button("Calcular riesgo")

        if submit:
            fila = pd.DataFrame(
                [
                    {
                        "gender": gender,
                        "age": float(age),
                        "hypertension": int(hypertension),
                        "heart_disease": int(heart_disease),
                        "smoking_history": smoking_history,
                        "bmi": float(bmi),
                        "HbA1c_level": float(hba1c),
                        "blood_glucose_level": int(glucose),
                    }
                ]
            )

            proba = float(model.predict_proba(fila)[:, 1][0])
            pred = int(proba >= threshold)

            c1, c2 = st.columns([1, 1], gap="large")

            with c1:
                with st.container(border=True):
                    st.markdown("**Probabilidad estimada**")
                    st.markdown(
                        f"<div style='font-size:40px; font-weight:900;'>{proba:.3f}</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption("p(diabetes=1 | x). Esto NO es diagnóstico; es estimación de riesgo.")

            with c2:
                with st.container(border=True):
                    st.markdown("**Clasificación por umbral**")
                    etiqueta = "Riesgo alto (1)" if pred == 1 else "Riesgo bajo (0)"
                    st.markdown(
                        f"<div style='font-size:28px; font-weight:900;'>{etiqueta}</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption("Decisión operacional: activar (o no) protocolo de confirmación.")

            st.divider()

            st.markdown("**¿Qué debería pasar después de esta predicción?**")
            if pred == 1:
                st.write(
                    "Como el caso quedó en ‘riesgo alto’, lo recomendado es: (1) confirmar con pruebas clínicas o valoración médica, "
                    "(2) priorizar seguimiento cercano y educación en hábitos, y (3) registrar el resultado final para auditoría "
                    "y mejora continua del modelo."
                )
            else:
                st.write(
                    "Como el caso quedó en ‘riesgo bajo’, no significa que el paciente esté libre de riesgo en el largo plazo. "
                    "Lo recomendado es mantener controles rutinarios y, si existen factores externos no modelados (historia familiar, dieta, etc.), "
                    "considerarlos en la decisión clínica."
                )


if __name__ == "__main__":
    main()
