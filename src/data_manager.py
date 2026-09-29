# src/data_manager.py
"""Funciones para crear, validar e importar/exportar datos experimentales."""

from __future__ import annotations

import io
from typing import List, Optional

import pandas as pd


def parse_list(values: str) -> List[str]:
    """Convierte una cadena separada por comas en una lista limpia."""
    if values is None:
        return []
    return [v.strip() for v in str(values).split(",") if v.strip()]


def build_growth_columns(variable_names: List[str], max_day: int) -> List[str]:
    """Construye columnas del tipo: longitud_raiz_d1, longitud_raiz_d2, ..."""
    columns = []
    for variable in variable_names:
        for day in range(1, max_day + 1):
            columns.append(f"{variable}_d{day}")
    return columns


def create_empty_structure(
    tratamientos: str,
    replicas: int,
    variables: str,
    max_day: int,
) -> pd.DataFrame:
    """Genera la estructura base del experimento para ingreso manual."""
    treatment_list = parse_list(tratamientos)
    variable_list = parse_list(variables)

    if not treatment_list:
        treatment_list = ["Control"]
    if not variable_list:
        variable_list = ["longitud_raiz", "alto_plantula"]

    records = []
    for treatment in treatment_list:
        for rep in range(1, replicas + 1):
            row = {
                "id_unidad": f"{treatment.replace(' ', '_')}_{rep}",
                "tratamiento": treatment,
                "replica": rep,
                "germinado": False,
                "dia_germinacion": None,
                "observaciones": "",
            }
            for col in build_growth_columns(variable_list, max_day):
                row[col] = None
            records.append(row)

    return pd.DataFrame(records)


def validate_experiment(df: pd.DataFrame, max_day: int) -> List[str]:
    """Valida coherencia básica de los datos."""
    issues = []

    if df.empty:
        return ["La tabla está vacía."]

    # Validar tratamiento y réplica
    if df["tratamiento"].isna().any():
        issues.append("Hay tratamientos faltantes.")
    if df["replica"].isna().any():
        issues.append("Hay réplicas faltantes.")

    for idx, row in df.iterrows():
        if pd.notna(row.get("dia_germinacion")):
            if row["dia_germinacion"] < 1 or row["dia_germinacion"] > max_day:
                issues.append(f"Fila {idx}: el día de germinación debe estar entre 1 y {max_day}.")
        if bool(row.get("germinado")) and pd.isna(row.get("dia_germinacion")):
            issues.append(f"Fila {idx}: la semilla germinó, pero no se registró el día de germinación.")

    # Validar columnas de medición
    measurement_columns = [c for c in df.columns if "_d" in c]
    for col in measurement_columns:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.isna().all():
            continue
        if series.notna().any() and (series[series.notna()] < 0).any():
            issues.append(f"La columna {col} tiene valores negativos, revise la medición.")

    # Advertencias de valores faltantes en mediciones
    if measurement_columns:
        missing = df[measurement_columns].isna().sum().sum()
        if missing > 0:
            issues.append(f"Hay {missing} mediciones faltantes en la estructura experimental.")

    return issues


def export_dataframe(df: pd.DataFrame, file_format: str) -> bytes:
    """Exporta un dataframe como CSV o Excel."""
    if file_format == "csv":
        return df.to_csv(index=False).encode("utf-8")
    if file_format == "xlsx":
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False)
        return buffer.getvalue()
    raise ValueError("Formato no soportado: use 'csv' o 'xlsx'.")


def load_uploaded_dataframe(uploaded_file) -> Optional[pd.DataFrame]:
    """Carga un archivo CSV o Excel importado por el usuario."""
    filename = getattr(uploaded_file, "name", "")
    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    return None

