# app.py
"""
BIOQUIFI - Registro y análisis de germinación y crecimiento de semillas/plántulas

Instalación:
    1. Crear entorno virtual
       python -m venv .venv
       source .venv/bin/activate      # Linux/macOS
       .venv\Scripts\activate         # Windows

    2. Instalar dependencias
       pip install -r requirements.txt

    3. Ejecutar la aplicación
       streamlit run app.py

Descripción:
    Esta app permite registrar experimentos de germinación y crecimiento, validar
    datos, analizar diferencias por tratamiento y exportar resultados en csv/xlsx
    y reportes en PDF.
"""

from __future__ import annotations

import io
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.analysis import (
    build_boxplot_figure,
    build_growth_curve_plot,
    build_summary_report,
    compute_analysis,
    export_report_pdf,
    generate_summary_dataframe,
)
from src.data_manager import (
    create_empty_structure,
    export_dataframe,
    load_uploaded_dataframe,
    validate_experiment,
)
from src.sample_data import create_dummy_experiment

st.set_page_config(
    page_title="BIOQUIFI - Germinación",
    page_icon="🌱",
    layout="wide",
)


# ----------------------------
# Inicialización de sesión
# ----------------------------
def init_session_state() -> None:
    if "experiment_data" not in st.session_state:
        st.session_state.experiment_data = create_dummy_experiment()
    if "experiment_meta" not in st.session_state:
        st.session_state.experiment_meta = {
            "nombre": "Experimento BIOQUIFI",
            "temperatura_c": 25,
            "humedad_pct": 60,
            "fotoperiodo_h": 12,
            "sustrato": "Arena + turba",
            "metodo": "Germinación en placas",
            "fecha": "2026-09-29",
            "notas": "Datos de ejemplo para prueba de la aplicación.",
        }
    if "variables" not in st.session_state:
        st.session_state.variables = ["longitud_raiz", "alto_plantula", "peso_fresco"]
    if "max_day" not in st.session_state:
        st.session_state.max_day = 10


init_session_state()


# ----------------------------
# Utilidades para interfaz
# ----------------------------
def render_metadata_form() -> None:
    meta = st.session_state.experiment_meta
    with st.sidebar:
        st.header("🧪 Configuración del experimento")
        st.session_state.experiment_meta["nombre"] = st.text_input(
            "Nombre del experimento",
            value=meta.get("nombre", "Experimento BIOQUIFI"),
        )
        st.session_state.experiment_meta["temperatura_c"] = st.number_input(
            "Temperatura (°C)",
            min_value=0.0,
            max_value=60.0,
            value=float(meta.get("temperatura_c", 25.0)),
            step=0.1,
        )
        st.session_state.experiment_meta["humedad_pct"] = st.number_input(
            "Humedad relativa (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(meta.get("humedad_pct", 60.0)),
            step=1.0,
        )
        st.session_state.experiment_meta["fotoperiodo_h"] = st.number_input(
            "Fotoperiodo (h)",
            min_value=0,
            max_value=24,
            value=int(meta.get("fotoperiodo_h", 12)),
        )
        st.session_state.experiment_meta["sustrato"] = st.text_input(
            "Sustrato",
            value=meta.get("sustrato", "Arena + turba"),
        )
        st.session_state.experiment_meta["metodo"] = st.text_input(
            "Método",
            value=meta.get("metodo", "Germinación en placas"),
        )
        st.session_state.experiment_meta["fecha"] = st.date_input(
            "Fecha de inicio",
            value=pd.to_datetime(meta.get("fecha", "2026-09-29")).date(),
        ).isoformat()
        st.session_state.experiment_meta["notas"] = st.text_area(
            "Notas",
            value=meta.get("notas", ""),
            height=120,
        )

        st.markdown("---")
        st.subheader("Diseño experimental")
        tratamientos = st.text_input(
            "Tratamientos (separados por coma)",
            value="Control, Tratamiento A, Tratamiento B",
            help="Ejemplo: Control, NaCl 50 mM, NaCl 100 mM",
        )
        n_replicas = st.number_input(
            "Réplicas por tratamiento",
            min_value=1,
            max_value=50,
            value=10,
        )
        max_day = st.number_input(
            "Número de días de medición",
            min_value=1,
            max_value=120,
            value=10,
        )
        variables = st.text_input(
            "Variables de crecimiento (separadas por coma)",
            value="longitud_raiz, alto_plantula, peso_fresco",
            help="Ejemplo: longitud_raiz, alto_plantula, peso_fresco",
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generar estructura"):
                df = create_empty_structure(
                    tratamientos=tratamientos,
                    replicas=n_replicas,
                    variables=variables,
                    max_day=max_day,
                )
                st.session_state.experiment_data = df
                st.session_state.variables = [
                    v.strip() for v in variables.split(",") if v.strip()
                ]
                st.session_state.max_day = max_day
                st.success("Estructura creada correctamente.")
        with col2:
            if st.button("Datos de ejemplo"):
                df = create_dummy_experiment()
                st.session_state.experiment_data = df
                st.session_state.variables = ["longitud_raiz", "alto_plantula", "peso_fresco"]
                st.session_state.max_day = 10
                st.success("Datos de ejemplo cargados.")

        uploaded_file = st.file_uploader(
            "Importar CSV o Excel",
            type=["csv", "xlsx", "xls"],
            help="Importa un archivo con la estructura del experimento.",
        )
        if uploaded_file is not None:
            df = load_uploaded_dataframe(uploaded_file)
            if df is not None:
                st.session_state.experiment_data = df
                st.session_state.variables = [
                    c.split("_d")[0] for c in df.columns if "_d" in c and c.endswith("_d1")
                ]
                st.success("Archivo importado correctamente.")

        st.markdown("---")
        st.caption("Exportar datos")
        if not st.session_state.experiment_data.empty:
            csv_data = export_dataframe(st.session_state.experiment_data, "csv")
            st.download_button(
                "Descargar CSV",
                data=csv_data,
                file_name="bioquifi_experimento.csv",
                mime="text/csv",
            )
            xlsx_data = export_dataframe(st.session_state.experiment_data, "xlsx")
            st.download_button(
                "Descargar Excel",
                data=xlsx_data,
                file_name="bioquifi_experimento.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


render_metadata_form()


# ----------------------------
# Encabezado principal
# ----------------------------
st.title("🌱 BIOQUIFI · Gestión y análisis de germinación")
st.caption("Plataforma para estudiantes de Ingeniería Biológica para analizar germinación y crecimiento de semillas/plántulas.")

st.write("**Experimento actual:**", st.session_state.experiment_meta.get("nombre", "Sin nombre"))

# Mostrar metadata del experimento
with st.expander("📌 Información del experimento", expanded=False):
    meta = st.session_state.experiment_meta
    cols = st.columns(3)
    cols[0].markdown(f"**Temperatura:** {meta.get('temperatura_c', '')} °C")
    cols[1].markdown(f"**Humedad:** {meta.get('humedad_pct', '')} %")
    cols[2].markdown(f"**Fotoperiodo:** {meta.get('fotoperiodo_h', '')} h")
    cols = st.columns(3)
    cols[0].markdown(f"**Sustrato:** {meta.get('sustrato', '')}")
    cols[1].markdown(f"**Método:** {meta.get('metodo', '')}")
    cols[2].markdown(f"**Fecha:** {meta.get('fecha', '')}")
    st.text_area("Notas", value=meta.get("notas", ""), disabled=True, height=80)


# ----------------------------
# Pestañas principales
# ----------------------------
tab_ingreso, tab_analisis, tab_reporte = st.tabs(["📝 Ingreso de datos", "📊 Análisis", "📄 Reporte"])

with tab_ingreso:
    st.subheader("Tabla editable del experimento")
    df = st.session_state.experiment_data.copy()
    if df.empty:
        st.warning("Aún no hay datos cargados. Use 'Generar estructura' o 'Datos de ejemplo'.")
    else:
        # Mostrar editor de tabla con filas dinámicas
        edited = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "id_unidad": st.column_config.TextColumn("ID unidad", help="Identificador único de cada semilla o plántula."),
                "tratamiento": st.column_config.SelectboxColumn(
                    "Tratamiento",
                    options=sorted(df["tratamiento"].dropna().unique().tolist() + ["Control"]),
                    required=True,
                ),
                "replica": st.column_config.NumberColumn("Réplica", min_value=1, step=1, required=True),
                "germinado": st.column_config.CheckboxColumn("Germinó", help="1 = sí, 0 = no"),
                "dia_germinacion": st.column_config.NumberColumn("Día germinación", min_value=1, max_value=st.session_state.max_day, step=1),
                "observaciones": st.column_config.TextColumn("Notas", width="large"),
            },
        )

        st.session_state.experiment_data = edited

        # Validaciones simples
        issues = validate_experiment(edited, st.session_state.max_day)
        if issues:
            st.warning("Se detectaron algunas validaciones:")
            for item in issues:
                st.write(f"- {item}")
        else:
            st.success("Datos validados correctamente: no se detectaron inconsistencias evidentes.")

        st.download_button(
            "Descargar la tabla actual",
            data=export_dataframe(edited, "csv"),
            file_name="bioquifi_tabla_actual.csv",
            mime="text/csv",
        )

        st.caption("Tip: puedes editar directamente la tabla, asignar tratamiento, día de germinación y mediciones de día por día.")

with tab_analisis:
    st.subheader("Análisis estadístico automático")
    if st.session_state.experiment_data.empty:
        st.warning("No hay datos para analizar.")
    else:
        variables = [
            c.split("_d")[0] for c in st.session_state.experiment_data.columns if "_d" in c and c.endswith("_d1")
        ]
        if not variables:
            st.warning("No se encontraron variables de crecimiento. Genera la estructura del experimento primero.")
        else:
            variable = st.selectbox("Variable para analizar", variables)
            analysis = compute_analysis(st.session_state.experiment_data, variables=variables, selected_variable=variable)

            st.markdown("### Indicadores clave")
            kpis = analysis["kpis"]
            col1, col2, col3 = st.columns(3)
            col1.metric("Germinación promedio", f"{kpis['germinacion_promedio']:.1f}%")
            col2.metric("Tiempo medio de germinación", f"{kpis['tiempo_medio_germinacion']:.2f} días")
            col3.metric("Tratamiento con mayor germinación", kpis["tratamiento_mas_alto"])

            st.markdown("### Descriptivos por tratamiento")
            desc_df = analysis["descriptivos"]
            st.dataframe(desc_df, use_container_width=True)

            st.markdown("### ANOVA y prueba post-hoc de Tukey")
            anova_table = analysis["anova"]
            st.dataframe(anova_table, use_container_width=True)

            st.markdown("### Interpretación sencilla")
            st.info(analysis["interpretacion"])

            st.markdown("### Gráficos")
            fig_box = build_boxplot_figure(st.session_state.experiment_data, selected_variable=variable)
            st.pyplot(fig_box, clear_figure=True)

            fig_curve = build_growth_curve_plot(st.session_state.experiment_data, selected_variable=variable)
            st.plotly_chart(fig_curve, use_container_width=True)

            if analysis.get("barplot") is not None:
                st.bar_chart(analysis["barplot"], x="tratamiento", y="media", use_container_width=True)

with tab_reporte:
    st.subheader("Reporte resumido y descarga")
    if st.session_state.experiment_data.empty:
        st.warning("No hay datos para generar reportes.")
    else:
        report = build_summary_report(st.session_state.experiment_data)
        st.dataframe(report["resumen_general"], use_container_width=True)

        st.markdown("### Resumen interpretativo")
        st.write(report["interpretacion_general"])

        if st.button("Generar PDF"):
            pdf_buffer = export_report_pdf(st.session_state.experiment_data)
            st.download_button(
                "Descargar PDF",
                data=pdf_buffer.getvalue(),
                file_name="bioquifi_reporte.pdf",
                mime="application/pdf",
            )

        st.caption("También puedes exportar los gráficos o los datos de resumen desde la sección de análisis.")


st.sidebar.markdown("---")
st.sidebar.caption("BIOQUIFI © 2026 · Diseño para investigación en germinación y crecimiento")


# Fin de la app

