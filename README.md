# README.md
# BIOQUIFI - Aplicación para análisis de germinación y crecimiento

Aplicación desarrollada en Python con Streamlit para registrar y analizar datos experimentales de germinación y crecimiento de semillas/plántulas.

## Características principales

- Ingreso de datos editable para múltiples tratamientos y réplicas.
- Registro de días de germinación y mediciones diarias de crecimiento.
- Validaciones básicas de coherencia de datos.
- Importación y exportación a CSV/Excel.
- Análisis estadístico con descriptivos, germinación, ANOVA y Tukey.
- Visualización de boxplots, barras y curvas de crecimiento.
- Descarga de resumen y reporte PDF.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución

```bash
streamlit run app.py
```

## Estructura del proyecto

```text
.
├── app.py
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── analysis.py
│   ├── data_manager.py
│   └── sample_data.py
└── .gitignore
```

## Uso rápido

1. Abre la app en el navegador.
2. Crea la estructura del experimento o carga datos de ejemplo.
3. Edita la tabla con los tratamientos, días de germinación y mediciones.
4. Revisa la validación de datos.
5. Abre la pestaña de análisis para ver descriptivos y comparaciones.
6. Descarga el CSV/Excel o el reporte PDF.

