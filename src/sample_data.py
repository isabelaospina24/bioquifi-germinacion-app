# src/sample_data.py
"""Generación de datos dummy para probar la app inmediatamente."""

from __future__ import annotations

import numpy as np
import pandas as pd


def create_dummy_experiment() -> pd.DataFrame:
    """Genera un conjunto de datos de ejemplo realista para germinación y crecimiento."""
    rng = np.random.default_rng(42)
    tratamientos = ["Control", "Tratamiento A", "Tratamiento B", "Tratamiento C"]
    variables = ["longitud_raiz", "alto_plantula", "peso_fresco"]
    max_day = 10
    rows = []

    for treatment in tratamientos:
        for rep in range(1, 11):
            germinacion = int(rng.integers(1, 10) >= 3)
            if germinacion:
                day_germ = int(rng.integers(2, 8))
            else:
                day_germ = None

            row = {
                "id_unidad": f"{treatment.replace(' ', '_')}_{rep}",
                "tratamiento": treatment,
                "replica": rep,
                "germinado": bool(germinacion),
                "dia_germinacion": day_germ,
                "observaciones": "",
            }

            for var in variables:
                for day in range(1, max_day + 1):
                    if not germinacion:
                        value = np.nan
                    else:
                        if day < day_germ:
                            value = np.nan
                        else:
                            base = {
                                "longitud_raiz": 0.8 + rep * 0.15 + (day * 0.7),
                                "alto_plantula": 0.5 + rep * 0.12 + (day * 0.6),
                                "peso_fresco": 0.08 + rep * 0.02 + (day * 0.04),
                            }[var]
                            if treatment == "Control":
                                base *= 1.1
                            elif treatment == "Tratamiento A":
                                base *= 1.3
                            elif treatment == "Tratamiento B":
                                base *= 0.9
                            else:
                                base *= 0.75
                            value = round(float(base + rng.normal(0, 0.18)), 3)
                    row[f"{var}_d{day}"] = value

            rows.append(row)

    return pd.DataFrame(rows)

