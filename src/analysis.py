# src/analysis.py
"""Análisis estadístico, visualización y reportes del experimento."""

from __future__ import annotations

import io
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


# -------------------------------------
# Preparación de datos
# -------------------------------------
def get_measurement_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if "_d" in col]


def get_variables_from_columns(df: pd.DataFrame) -> List[str]:
    cols = get_measurement_columns(df)
    if not cols:
        return []
    return sorted({c.split("_d")[0] for c in cols})


def melt_growth_data(df: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
    """Transforma columnas en formato ancho a largo para análisis longitudinal."""
    rows = []
    for _, row in df.iterrows():
        tratamiento = row.get("tratamiento")
        replica = row.get("replica")
        for variable in variables:
            for day in range(1, 100):
                col = f"{variable}_d{day}"
                if col not in df.columns:
                    break
                value = row.get(col)
                if pd.notna(value):
                    rows.append(
                        {
                            "tratamiento": tratamiento,
                            "replica": replica,
                            "variable": variable,
                            "dia": day,
                            "valor": float(value),
                        }
                    )
    return pd.DataFrame(rows)


def get_final_day_value(df: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
    """Obtiene la última medición válida por unidad y variable."""
    rows = []
    for _, row in df.iterrows():
        for variable in variables:
            values = []
            day = 1
            while True:
                col = f"{variable}_d{day}"
                if col not in df.columns:
                    break
                val = row.get(col)
                if pd.notna(val):
                    values.append((day, float(val)))
                day += 1
            if values:
                final_day, final_value = values[-1]
                rows.append(
                    {
                        "tratamiento": row.get("tratamiento"),
                        "replica": row.get("replica"),
                        "variable": variable,
                        "dia_final": final_day,
                        "valor_final": final_value,
                    }
                )
    return pd.DataFrame(rows)


# -------------------------------------
# Indicadores y métricas
# -------------------------------------
def compute_germination_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula porcentaje de germinación y tiempo medio de germinación por tratamiento."""
    metrics = []
    for treatment, group in df.groupby("tratamiento", dropna=False):
        n_total = len(group)
        n_germinated = int(group["germinado"].fillna(False).astype(bool).sum())
        pct_germ = (n_germinated / n_total) * 100 if n_total else 0.0
        germinated_days = group.loc[group["germinado"].fillna(False).astype(bool), "dia_germinacion"]
        mean_germination_time = germinated_days.mean() if not germinated_days.empty else np.nan
        metrics.append(
            {
                "tratamiento": treatment,
                "total_unidades": n_total,
                "germinadas": n_germinated,
                "porcentaje_germinacion": pct_germ,
                "tiempo_medio_germinacion": mean_germination_time,
            }
        )
    return pd.DataFrame(metrics)


def compute_descriptives(df: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
    """Calcula estadísticos descriptivos por tratamiento y variable."""
    final_df = get_final_day_value(df, variables)
    if final_df.empty:
        return pd.DataFrame()

    summary = (
        final_df.groupby(["tratamiento", "variable"], dropna=False)["valor_final"]
        .agg([
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max",
        ])
        .reset_index()
    )
    summary["cv_pct"] = np.where(
        summary["mean"].abs() > 0,
        (summary["std"] / summary["mean"]) * 100,
        np.nan,
    )
    summary.columns = [
        "tratamiento",
        "variable",
        "n",
        "media",
        "mediana",
        "desv_estandar",
        "min",
        "max",
        "cv_pct",
    ]
    return summary


def run_anova_tukey(df: pd.DataFrame, variables: List[str]) -> Dict[str, pd.DataFrame]:
    """Aplica ANOVA de un factor y Tukey para la última medición de cada variable."""
    final_df = get_final_day_value(df, variables)
    results = {}
    for variable in variables:
        subset = final_df[final_df["variable"] == variable].copy()
        if subset.empty:
            continue
        groups = [group["valor_final"].dropna().to_numpy() for _, group in subset.groupby("tratamiento")]
        valid = [g for g in groups if len(g) > 1]
        if len(valid) < 2:
            table = pd.DataFrame({"variable": [variable], "mensaje": ["Se requieren al menos 2 grupos con más de 1 dato."]})
            results[variable] = table
            continue
        f_value, p_value = stats.f_oneway(*valid)
        tukey = pairwise_tukeyhsd(endog=subset["valor_final"].dropna(), groups=subset["tratamiento"].dropna(), alpha=0.05)
        tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
        results[variable] = pd.DataFrame(
            {
                "variable": [variable],
                "f_value": [f_value],
                "p_value": [p_value],
                "interpretacion": ["Hay diferencias significativas" if p_value < 0.05 else "No hay diferencias significativas"],
            }
        )
        results[f"{variable}_tukey"] = tukey_df
    return results


# -------------------------------------
# Gráficos
# -------------------------------------
def build_boxplot_figure(df: pd.DataFrame, selected_variable: str) -> plt.Figure:
    """Genera un boxplot por tratamiento para una variable seleccionada."""
    fig, ax = plt.subplots(figsize=(9, 5))
    data = []
    labels = []
    for treatment, group in df.groupby("tratamiento", dropna=False):
        values = []
        day_col = 1
        while True:
            col = f"{selected_variable}_d{day_col}"
            if col not in df.columns:
                break
            vals = pd.to_numeric(group[col], errors="coerce").dropna().tolist()
            values.extend(vals)
            day_col += 1
        if values:
            data.append(values)
            labels.append(treatment)
    if not data:
        ax.text(0.5, 0.5, "Sin datos para graficar", ha="center", va="center")
        return fig
    ax.boxplot(data, labels=labels, patch_artist=True)
    ax.set_title(f"Boxplot de {selected_variable} por tratamiento")
    ax.set_xlabel("Tratamiento")
    ax.set_ylabel(selected_variable)
    fig.tight_layout()
    return fig


def build_growth_curve_plot(df: pd.DataFrame, selected_variable: str):
    """Genera curva de crecimiento media por tratamiento."""
    growth_rows = []
    for treatment, group in df.groupby("tratamiento", dropna=False):
        for day in range(1, 101):
            col = f"{selected_variable}_d{day}"
            if col not in df.columns:
                break
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            if values.empty:
                continue
            growth_rows.append({"tratamiento": treatment, "dia": day, "valor_promedio": values.mean()})
    if not growth_rows:
        return px.line(pd.DataFrame({"tratamiento": [], "dia": [], "valor_promedio": []}), x="dia", y="valor_promedio", color="tratamiento")
    growth = pd.DataFrame(growth_rows)
    fig = px.line(
        growth,
        x="dia",
        y="valor_promedio",
        color="tratamiento",
        title=f"Curva de crecimiento promedio ({selected_variable})",
        markers=True,
    )
    fig.update_layout(xaxis_title="Día", yaxis_title=selected_variable)
    return fig


# -------------------------------------
# Reporte resumen
# -------------------------------------
def generate_summary_dataframe(df: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
    """Crea una tabla resumen general del experimento."""
    germination = compute_germination_metrics(df)
    final_df = get_final_day_value(df, variables)
    summary = []
    for treatment in df["tratamiento"].dropna().unique():
        row = {"tratamiento": treatment}
        treatment_metrics = germination[germination["tratamiento"] == treatment]
        if not treatment_metrics.empty:
            row["germinacion_pct"] = treatment_metrics["porcentaje_germinacion"].iloc[0]
            row["tiempo_medio_germinacion"] = treatment_metrics["tiempo_medio_germinacion"].iloc[0]
        for variable in variables:
            var_df = final_df[(final_df["tratamiento"] == treatment) & (final_df["variable"] == variable)]
            if not var_df.empty:
                row[f"{variable}_media"] = var_df["valor_final"].mean()
                row[f"{variable}_sd"] = var_df["valor_final"].std(ddof=1)
            else:
                row[f"{variable}_media"] = np.nan
                row[f"{variable}_sd"] = np.nan
        summary.append(row)
    return pd.DataFrame(summary)


def build_summary_report(df: pd.DataFrame) -> Dict[str, object]:
    """Genera un reporte en formato de resumen para mostrar en la app."""
    variables = get_variables_from_columns(df)
    if not variables:
        variables = ["longitud_raiz", "alto_plantula"]

    germination = compute_germination_metrics(df)
    descriptives = compute_descriptives(df, variables)
    summary = generate_summary_dataframe(df, variables)
    anova = run_anova_tukey(df, variables)

    interpretations = []
    for variable, result in anova.items():
        if variable.endswith("_tukey"):
            continue
        p = float(result["p_value"].iloc[0]) if not result.empty else np.nan
        if np.isnan(p):
            interpretations.append(f"{variable}: sin suficientes datos para evaluar la significancia.")
        elif p < 0.05:
            interpretations.append(f"{variable}: existen diferencias significativas entre tratamientos (p = {p:.3f}).")
        else:
            interpretations.append(f"{variable}: no se detectaron diferencias significativas entre tratamientos (p = {p:.3f}).")

    report = {
        "resumen_general": summary,
        "interpretacion_general": "\n".join(interpretations),
        "germination": germination,
        "descriptivos": descriptives,
        "anova": anova,
    }
    return report


def compute_analysis(df: pd.DataFrame, variables: List[str], selected_variable: str) -> Dict[str, object]:
    """Función central para análisis completo de la app."""
    germination = compute_germination_metrics(df)
    descriptives = compute_descriptives(df, variables)
    anova = run_anova_tukey(df, variables)

    # Kpis generales
    avg_pct = germination["porcentaje_germinacion"].mean() if not germination.empty else 0.0
    mgt = germination["tiempo_medio_germinacion"].mean() if not germination.empty else np.nan
    best = germination.loc[germination["porcentaje_germinacion"].idxmax(), "tratamiento"] if not germination.empty else "N/A"

    # Panel de resultados para la variable seleccionada
    selected_desc = descriptives[descriptives["variable"] == selected_variable] if not descriptives.empty else pd.DataFrame()
    variable_anova = anova.get(selected_variable, pd.DataFrame())

    # Interpretación variable por variable
    text = []
    if not variable_anova.empty:
        p_val = float(variable_anova["p_value"].iloc[0])
        if p_val < 0.05:
            text.append(f"Para {selected_variable}, hubo diferencias significativas entre tratamientos (p = {p_val:.3f}).")
        else:
            text.append(f"Para {selected_variable}, no hubo diferencias significativas entre tratamientos (p = {p_val:.3f}).")
    else:
        text.append(f"Para {selected_variable}, no hubo suficientes datos para evaluar significancia.")

    # Gráfico de barra con media por tratamiento
    barplot_df = selected_desc[["tratamiento", "media"]].rename(columns={"media": "media"}) if not selected_desc.empty else pd.DataFrame()

    return {
        "kpis": {
            "germinacion_promedio": avg_pct,
            "tiempo_medio_germinacion": mgt,
            "tratamiento_mas_alto": best,
        },
        "descriptivos": selected_desc,
        "anova": variable_anova,
        "interpretacion": " ".join(text),
        "barplot": barplot_df,
    }


def export_report_pdf(df: pd.DataFrame) -> io.BytesIO:
    """Genera un PDF con un resumen básico del experimento."""
    variables = get_variables_from_columns(df)
    if not variables:
        variables = ["longitud_raiz", "alto_plantula"]
    summary = build_summary_report(df)
    buffer = io.BytesIO()
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []

    story.append(Paragraph("BIOQUIFI - Reporte de germinación", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Resumen general", styles["Heading2"]))

    germination = summary["germination"]
    if not germination.empty:
        rows = [["Tratamiento", "Germinación %", "Tiempo medio (días)"]]
        for _, row in germination.iterrows():
            rows.append([
                row["tratamiento"],
                f"{row['porcentaje_germinacion']:.2f}",
                f"{row['tiempo_medio_germinacion']:.2f}" if pd.notna(row["tiempo_medio_germinacion"]) else "N/A",
            ])
        table = Table(rows)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#CFF3D5")),
            ("GRID", (0, 0), (-1, -1), 1, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]))
        story.append(table)

    story.append(Spacer(1, 12))
    story.append(Paragraph("Interpretación", styles["Heading2"]))
    story.append(Paragraph(summary["interpretacion_general"], styles["BodyText"]))

    doc.build(story)
    buffer.seek(0)
    return buffer

